"""The ``astro`` signal through atlas-run (ATLAS.runner): config, build, run.

A config with an ``astro`` block must give exactly the model one builds by
hand with ``load_astro_flow``, ``set_flow_tspan``, ``make_red_noise`` and
``pspace_log_prior``; every misconfiguration must be caught by
``validate_run`` before anything is launched; validation must not import the
flow libraries; and a run must save named astrophysical parameters.
"""
import importlib.util
import json
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("flax")
pytest.importorskip("distrax")

import jax.numpy as jnp  # noqa: E402
from numpyro.infer.util import log_density  # noqa: E402

from ATLAS.astro.flows import load_astro_flow  # noqa: E402
from ATLAS.astro.flows.flax import FlaxAstroFlow  # noqa: E402
from ATLAS.data import PTA_Data  # noqa: E402
from ATLAS.model import model_maker  # noqa: E402
from ATLAS.model_builder import ModelBuilder  # noqa: E402
from ATLAS.psd_functions import powerlaw  # noqa: E402
from ATLAS.runner import data_io  # noqa: E402
from ATLAS.runner.build import build_model  # noqa: E402
from ATLAS.runner.config import ConfigError, expand_runs, load_config, validate_run  # noqa: E402
from ATLAS.runner.execute import _model_and_args, run_one  # noqa: E402
from ATLAS.signals import signals_utils as sutils  # noqa: E402
from ATLAS.signals.correlated.astro_gwb import set_flow_tspan  # noqa: E402

from . import harness as H  # noqa: E402
from .test_flax_flows import F, _flow  # noqa: E402

pytestmark = pytest.mark.astro

ROOT = Path(__file__).resolve().parent.parent
NPSR, N_IRN = 3, 6
IRN = [[-20.0, -11.0], [0.0, 7.0]]
# two of Pandora's phenom parameters with normal priors (finite everywhere)
NAMES = ["gsmf_phi0_log10", "mmb_scatter_dex"]
TIGHT = 1e-12
C_OFF, SLAB = 10.0 ** -8.5, 0.30103    # offset-mode flows: rho offset and zero-slab depth


def save_flow(path, tspan, **meta):
    '''

    Save a small random-weight 'rho|theta' flow "trained" at Tspan `tspan`.

    Arguments
    ------------------------------
    path : pathlib.Path
    	Directory to save the flow to.
    tspan : float
    	Training Tspan [s]; the flow's frequencies are its first F harmonics.
    meta : dict
    	Extra metadata, or replacements (e.g. param_space=None to drop it).

    Returns
    ------------------------------
    flow : FlaxAstroFlow
    	The saved flow.

    '''
    base = _flow("coupling_rqs", nf_type=meta.pop("nf_type", "rho|theta"))
    metadata = dict(param_names=NAMES, param_space="PS_Pandora_Phenom", tspan=float(tspan),
                    f_conv=list(np.asarray(sutils.get_harmonic_frequencies(F, tspan))))
    metadata.update(meta)
    metadata = {k: v for k, v in metadata.items() if v is not None}
    flow = FlaxAstroFlow(base.model, base.nf_type, base.mapping, base.n_params, metadata=metadata)
    flow.save(str(path))
    return flow


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    '''

    Synthetic pulsars (pickled, as for data.source: enterprise_pickle), their
    white-noise dictionary, a priors file, a hook file and the flows.

    Arguments
    ------------------------------
    tmp_path_factory : pytest.TempPathFactory
    	Module-scoped temporary directory factory.

    Returns
    ------------------------------
    files : dict
    	Paths ('dir'), the flow and its Tspan, and the data's Tspan.

    '''
    tmp = tmp_path_factory.mktemp("runner_astro")
    psrs = H.load_psrs("synth", NPSR)
    with open(tmp / "psrs.pkl", "wb") as fh:
        pickle.dump(psrs, fh)
    m = H.build(npsr=NPSR, model_string="ltm|unc", n_irn=N_IRN)
    names = [n for cov in m.wn.cov_matrices for n in cov.get_param_names()]
    (tmp / "wn.json").write_text(json.dumps(dict(zip(names, np.asarray(m.wn_vec).tolist()))))

    tspan_data = float(m.data.pta_tspan)
    tspan_flow = 0.99 * tspan_data                 # a shorter flow span: set, with a warning
    flow = save_flow(tmp / "flow", tspan_flow)
    save_flow(tmp / "flow_no_pspace", tspan_flow, param_space=None)
    save_flow(tmp / "flow_theta_rho", tspan_flow, nf_type="theta|rho")
    # ng20 offset-mode flows (sampled in y = log10(rho + c)), with and without a zero slab
    save_flow(tmp / "flow_slab", tspan_flow, zero_handling="offset", rho_offset=C_OFF, slab_dex=SLAB)
    save_flow(tmp / "flow_no_slab", tspan_flow, zero_handling="offset", rho_offset=C_OFF, slab_dex=0.0)

    sd_lo = float(flow.mean_ast[1] - 0.5 * flow.half_range_ast[1])
    sd_hi = float(flow.mean_ast[1] + 0.5 * flow.half_range_ast[1])
    priors = dict(irn=dict(log10_A=IRN[0], gamma=IRN[1]),
                  astro_gwb=dict(halflog10_rho=[-10.5, -5.5], mmb_scatter_dex=[sd_lo, sd_hi]))
    (tmp / "priors.yaml").write_text(yaml.safe_dump(priors))
    (tmp / "hook.py").write_text("import jax.numpy as jnp\n\n"
                                 "def lp(theta):\n"
                                 "    return -0.5 * jnp.sum((theta - 0.1) ** 2)\n")
    return dict(dir=tmp, flow=flow, tspan_flow=tspan_flow, tspan_data=tspan_data,
                fixed={NAMES[0]: float(flow.mean_ast[0])}, theta_box=(sd_lo, sd_hi))


def make_run(files, name="astro", edit=None, **astro):
    '''

    Write a YAML config with an astro block, load and expand it.

    Arguments
    ------------------------------
    files : dict
    	The `files` fixture.
    name : str
    	Run name (also its output directory). (Default: 'astro')
    edit : callable or None
    	``edit(cfg)`` changes the config dict before it is written. (Default: None)
    astro : dict
    	``model.signals.astro`` entries, over the defaults below.

    Returns
    ------------------------------
    run : dict
    	The single resolved run.

    '''
    tmp = files["dir"]
    a = dict(flow="flow", priors="astro_gwb", fixed=dict(files["fixed"]), log_prior="pspace")
    a.update(astro)
    cfg = dict(
        config_version=1,
        output=dict(root="runs"),
        data=dict(source="enterprise_pickle", pickle="psrs.pkl", noise_dict="wn.json",
                  cache_dir="cache"),
        # design_matrix: leave M alone (the synthetic pulsars carry no fitpars labels)
        timing=dict(linear_timing=True, marg_timing=False,
                    design_matrix=dict(columns="all", keep_offset=False)),
        white_noise=dict(vary=False, include_ecorr=True),
        frequency_bins=dict(num_irn_bins=N_IRN, num_gwb_bins=None, num_dm_bins=None,
                            num_chrom_bins=None),
        model=dict(basis_string="ltm|unc+astro->unc", priors_file="priors.yaml",
                   signals=dict(unc=dict(psd="powerlaw", priors="irn"), astro=a)),
        sampler=dict(marg_over_non_gwb=True, num_warmup=20, num_samples=20, seed=0,
                     nuts=dict(target_accept_prob=0.8, max_tree_depth=5, dense_mass=False)),
        runs=[dict(name=name)])
    if edit is not None:
        edit(cfg)
    path = tmp / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    (run,) = expand_runs(load_config(path))
    return run


def runner_log_density(run, params):
    '''

    Build a run the way atlas-run does and evaluate its model's log density.

    Arguments
    ------------------------------
    run : dict
    	A resolved run.
    params : dict or callable
    	Site values, or ``params(built)`` returning them.

    Returns
    ------------------------------
    value : float
    	The log density.
    built : ATLAS.runner.build.Built
    	The built model.

    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = build_model(run, *validate_run(run))
    model, args = _model_and_args(run, b)
    p = params(b) if callable(params) else params
    return float(log_density(model, (), args, p)[0]), b


def reference_log_density(run, flow, params, **astro_kw):
    '''

    The same model built by hand from the Python API, without the runner.

    Arguments
    ------------------------------
    run : dict
    	The run whose data settings are reproduced (pulsars, noise dictionary).
    flow : FlaxAstroFlow or ZukoAstroFlow
    	The flow, loaded independently of the runner.
    params : dict
    	Site values.
    astro_kw : dict
    	Astro keywords of `make_red_noise`.

    Returns
    ------------------------------
    value : float
    	The log density.

    '''
    psrs = data_io.load_pulsars(run)
    data_io.restrict_design_matrix(run, psrs)
    noise = data_io.load_noise_dict(run)
    data = PTA_Data(psrs, num_irn_bins=N_IRN, num_gwb_bins=None, num_dm_bins=None,
                    num_det_bins=None, adaptus_basis=None, adaptus_size=None,
                    linear_timing=True, marg_timing=False, noise_dict=noise, dm_ref_freq=1400.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        set_flow_tspan(data, flow)
    mb = ModelBuilder(data=data)
    wn = mb.make_white_noise(stabilize_TNT=False, include_ecorr=True)
    rn = mb.make_red_noise("ltm|unc+astro->unc", irn_psd_function=powerlaw,
                           irn_lower_bound_psd=jnp.array([IRN[0][0], IRN[1][0]]),
                           irn_upper_bound_psd=jnp.array([IRN[0][1], IRN[1][1]]),
                           astro_flow=flow, **astro_kw)
    wn_vec = wn.params_dict_to_vector(noise)
    raw = jnp.concat(data.raw_residuals)
    helpers = rn.get_helpers(reff=raw, white_noise_params=wn_vec)
    return float(log_density(model_maker, (), dict(raw_residuals=raw, super_sig=rn,
                                                   marg_over_non_gwb=True, helpers=helpers),
                             params)[0])


def point(b, seed=0):
    '''

    Site values inside every prior box of a built model (partial marginalization).

    Arguments
    ------------------------------
    b : ATLAS.runner.build.Built
    	The built model.
    seed : int
    	Seed of the draw. (Default: 0)

    Returns
    ------------------------------
    params : dict
    	Values of 'red_noise', 'z_a' and 'astro_params'.

    '''
    rng = np.random.default_rng(seed)
    rn, sig = b.rn, b.rn.astro_signal
    lo, hi = (np.asarray(a) for a in (rn.model.lower_prior_lim_all, rn.model.upper_prior_lim_all))
    a_lo, a_hi = np.asarray(sig.astro_lower), np.asarray(sig.astro_upper)
    return dict(red_noise=jnp.asarray(lo + (hi - lo) * rng.uniform(0.2, 0.8, lo.shape)),
                z_a=jnp.asarray(rng.normal(size=(rn.npsrs, 2 * rn.model.crn_bins))),
                astro_params=jnp.asarray(a_lo + (a_hi - a_lo) * rng.uniform(0.2, 0.8, a_lo.shape)))


# ---- validation ----------------------------------------------------------------------------

def test_valid_config(files):
    parsed, resolved = validate_run(make_run(files))
    assert parsed["shared_names"] == ["unc", "astro"]
    assert resolved["astro"] == dict(rho=[-10.5, -5.5],
                                     theta={NAMES[1]: list(files["theta_box"])},
                                     fixed=files["fixed"], log_prior="pspace")


def _set(path, value):
    def edit(cfg):
        *head, last = path.split(".")
        d = cfg
        for k in head:
            d = d.setdefault(k, {})
        d[last] = value
    return edit


@pytest.mark.parametrize("astro, edit, match", [
    (dict(flow=None), None, "needs model.signals.astro.flow"),
    (dict(flow=None, mapping="m.npz"), None, "needs model.signals.astro.pickle"),
    (dict(pickle="p.pkl", mapping="m.npz"), None, "not both"),
    (dict(flow="nowhere"), None, "holds neither"),
    (dict(flow="flow_theta_rho"), None, "needs a 'rho|theta' flow"),
    (dict(fixed={"nope": 1.0}), None, "are not flow parameters"),
    (dict(priors="missing"), None, "is not a key of the priors file"),
    (dict(log_prior="nonsense"), None, "must be null, pspace or path/file.py:function"),
    (dict(log_prior="nope.py:lp"), None, "log_prior file not found"),
    (dict(log_prior=None, pspace="PS_NG20_Fiducial"), None, "only used with log_prior: pspace"),
    (dict(flow="flow_no_pspace"), None, "pspace needs model.signals.astro.pspace"),
    (dict(tspan_policy="data"), None, "tspan_policy must be flow or strict"),
    (dict(tspan_rtol=0.0), None, "tspan_rtol must be a positive number"),
    (dict(n_prior_draws=-1), None, "n_prior_draws must be an integer >= 0"),
    (dict(flow="flow_no_slab"), None, "without a zero slab"),
    ({}, _set("frequency_bins.gwb_user_freqs", [1e-9, 2e-9]), "gwb_user_freqs cannot be used"),
    ({}, _set("frequency_bins.num_irn_bins", F - 1), f"num_irn_bins must be >= {F}"),
    ({}, _set("sampler.beta", 0.5), "TI evidence is not supported for 'astro'"),
    ({}, _set("model.basis_string", "ltm|unc+astro->unc;cor"), "cannot be combined"),
])
def test_validation_errors(files, astro, edit, match):
    run = make_run(files, name="bad", edit=edit, **astro)
    with pytest.raises(ConfigError, match=match):
        validate_run(run)


@pytest.mark.parametrize("entry, match", [
    (dict(nope=[0.0, 1.0]), "is not a flow parameter"),
    (dict(halflog10_rho=[[-10.0, -5.0]] * (F - 1)), f"{F - 1} halflog10_rho bounds"),
    (dict(halflog10_rho=[-5.0, -10.0]), "halflog10_rho must be"),
    ({NAMES[0]: [-3.0, -2.0]}, "is fixed"),
    ({NAMES[1]: [1.0, 0.0]}, "low < high"),
])
def test_priors_entry_errors(files, entry, match):
    (files["dir"] / "priors_bad.yaml").write_text(yaml.safe_dump(dict(
        irn=dict(log10_A=IRN[0], gamma=IRN[1]), astro_gwb=entry)))
    run = make_run(files, name="bad_priors",
                   edit=_set("model.priors_file", "priors_bad.yaml"))
    with pytest.raises(ConfigError, match=match):
        validate_run(run)


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None, reason="torch is installed")
def test_torch_flow_without_torch(files, tmp_path):
    """An ng20 flow directory needs torch to load: said at validation, not at run time."""
    ng20 = files["dir"] / "ng20_stub"
    ng20.mkdir(exist_ok=True)
    np.savez(ng20 / "transform_data.npz", B=5.0, mean=np.zeros(5), half_range=np.ones(5),
             ast_param_idxs=np.arange(2), rho_idxs=np.arange(2, 5))
    (ng20 / "zuko_flow.pkl").write_bytes(b"")
    with pytest.raises(ConfigError, match="torch"):
        validate_run(make_run(files, name="stub", flow="ng20_stub", log_prior=None))


def test_validation_imports_no_flow_libraries(files):
    """atlas-run check reads the flow's description without flax, torch or holodeck."""
    make_run(files, name="light")
    code = ("import sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from ATLAS.runner.config import load_config, expand_runs, validate_run\n"
            f"(run,) = expand_runs(load_config({str(files['dir'] / 'light.yaml')!r}))\n"
            "validate_run(run)\n"
            "heavy = [m for m in ('flax', 'torch', 'zuko', 'holodeck') if m in sys.modules]\n"
            "assert not heavy, heavy\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True,
                       text=True, env={"JAX_PLATFORMS": "cpu", "PATH": "/usr/bin:/bin",
                                       "HOME": str(Path.home())})
    assert r.returncode == 0, r.stderr[-2000:]


def test_shipped_configs_validate():
    for path in ("ATLAS/runner/templates/run_config.yaml",
                 "ATLAS/runner/examples/astro/astro_pandora_style.yaml"):
        for run in expand_runs(load_config(ROOT / path)):
            validate_run(run, check_files=False)


# ---- build -----------------------------------------------------------------------------------

def test_runner_model_equals_python_model(files):
    '''

    The runner's model (pspace prior, a parameter fixed by name, rho and
    theta boxes from the priors file, the flow's Tspan) has the same log
    density as the model built by hand.

    Arguments
    ------------------------------
    files : dict
    	The `files` fixture.

    Returns
    ------------------------------
    None

    '''
    from ATLAS.astro.priors import pspace_log_prior

    run = make_run(files)
    got, b = runner_log_density(run, point)
    assert b.data.pta_tspan == files["tspan_flow"]
    assert b.rn.astro_signal.astro_param_names == [NAMES[1]]
    flow = load_astro_flow(str(files["dir"] / "flow"))
    lo = np.asarray(flow.mean_ast - flow.half_range_ast).copy()
    hi = np.asarray(flow.mean_ast + flow.half_range_ast).copy()
    lo[1], hi[1] = files["theta_box"]
    want = reference_log_density(
        run, flow, point(b), fixed_astro=files["fixed"], astro_log_prior=pspace_log_prior(
            "PS_Pandora_Phenom", NAMES), astro_lower_bound=lo, astro_upper_bound=hi,
        gwb_lower_bound_psd=np.full(F, -10.5), gwb_upper_bound_psd=np.full(F, -5.5))
    assert np.isfinite(got) and abs(got - want) <= TIGHT * abs(want)


def test_tspan_policy_reaches_the_build(files):
    run = make_run(files, name="warn")
    with pytest.warns(UserWarning, match="shorter"):
        build_model(run, *validate_run(run))
    strict = make_run(files, name="strict", tspan_policy="strict")
    with pytest.raises(ValueError, match="Tspan mismatch"):
        build_model(strict, *validate_run(strict))


def test_hook_log_prior(files):
    '''

    A ``file.py:function`` log prior adds exactly its value at the full theta.

    Arguments
    ------------------------------
    files : dict
    	The `files` fixture.

    Returns
    ------------------------------
    None

    '''
    hooked, b = runner_log_density(make_run(files, name="hook", log_prior="hook.py:lp"), point)
    plain, _ = runner_log_density(make_run(files, name="plain", log_prior=None), point)
    sig = b.rn.astro_signal
    theta = np.asarray(sig.astro_container).copy()
    theta[sig.varied] = np.asarray(point(b)["astro_params"])
    want = float(-0.5 * np.sum((theta - 0.1) ** 2))
    assert abs((hooked - plain) - want) <= TIGHT * max(1.0, abs(hooked))


def test_ti_and_flow_prior_are_rejected_in_the_model(files):
    run = make_run(files, name="ti")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = build_model(run, *validate_run(run))
    raw = jnp.concat(b.data.raw_residuals)
    with pytest.raises(ValueError, match="not supported for 'astro'"):
        model_maker(raw, b.rn, True, beta=0.5)
    with pytest.raises(ValueError, match="cannot be combined with an 'astro' block"):
        model_maker(raw, b.rn, True, gwb_prior_flow=object())


# ---- a run -----------------------------------------------------------------------------------

def test_run_saves_named_astro_parameters(files):
    run = make_run(files, name="e2e")
    status = run_one(run)
    assert status["state"] == "done", status.get("traceback")
    out = Path(run["run"]["output_dir"])
    with np.load(out / "samples.npz") as s:
        assert list(s["astro_params__names"]) == [NAMES[1]]
        assert s["astro_params"].shape[-1] == 1
        rho = [n for n in s["red_noise__names"] if "rho" in n]
        assert rho == [f"astro_halflog10_rho_{i}" for i in range(F)]
        assert np.all(np.isfinite(s["astro_params"]))
    summary = json.loads((out / "summary.json").read_text())
    assert NAMES[1] in summary["parameters"] and "astro_halflog10_rho_0" in summary["parameters"]

    # run_pandora's spectrum outputs: drop mode, so log10 rho is the sampled variable
    with np.load(out / "samples.npz") as s:
        rho = s["red_noise"][..., -F:].reshape(-1, F)
    post = np.load(out / "gwb_log10_rho.npy")
    assert np.array_equal(post, rho)
    assert summary["p_rho_zero"]["posterior"] == [0.0] * F
    assert len(summary["p_rho_zero"]["flow_prior"]) == F       # PS_Pandora_Phenom draws
    q = summary["gwb_log10_rho"]["log10_rho_0"]
    assert np.isclose(q["median"], np.median(post[:, 0]), rtol=0, atol=1e-12)
    assert q["ci68"] == list(np.percentile(post[:, 0], [16, 84]))


def test_zero_slab_run(files):
    '''

    An offset-mode flow through the runner: parameters astro_log10_rho_plus_c_i,
    the box from the bottom of the zero slab, and gwb_log10_rho.npy / p_rho_zero
    counting slab draws as rho = 0, as run_pandora reports them.

    Arguments
    ------------------------------
    files : dict
    	The `files` fixture.

    Returns
    ------------------------------
    None

    '''
    run = make_run(files, name="slab", flow="flow_slab", n_prior_draws=0)
    status = run_one(run)
    assert status["state"] == "done", status.get("traceback")
    out = Path(run["run"]["output_dir"])
    with np.load(out / "samples.npz") as s:
        names = list(s["red_noise__names"])
        y = s["red_noise"][..., -F:].reshape(-1, F)
    assert names[-F:] == [f"astro_log10_rho_plus_c_{i}" for i in range(F)]
    assert np.all(y >= np.log10(C_OFF) - SLAB)
    post = np.load(out / "gwb_log10_rho.npy")
    zero = 10.0**y - C_OFF <= 0
    assert np.array_equal(np.isnan(post), zero)
    assert np.allclose(post[~zero], np.log10(10.0**y[~zero] - C_OFF), rtol=1e-14)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["p_rho_zero"] == dict(posterior=list(np.mean(zero, axis=0)))   # no prior draws


# ---- an ng20 flow (atlas-env-2) -------------------------------------------------------------

def test_ng20_offset_flow_through_the_runner(files):
    '''

    An ng20astromethods flow directory (zuko_flow.pkl + transform_data.npz,
    offset mode with a zero slab) gives the same model through the runner as
    loaded by hand.

    Arguments
    ------------------------------
    files : dict
    	The `files` fixture.

    Returns
    ------------------------------
    None

    '''
    torch = pytest.importorskip("torch")
    pytest.importorskip("zuko")
    from ATLAS.astro.flows import zuko as zf
    from .test_flax_flows import _mapping

    ng20 = files["dir"] / "ng20_offset"
    ng20.mkdir(exist_ok=True)
    torch.manual_seed(5)
    torch.save([zf.make_zuko_flow(F, 2, hidden_features=[16, 16])], ng20 / "zuko_flow.pkl")
    m = _mapping(5)
    tspan = files["tspan_flow"]
    np.savez(ng20 / "transform_data.npz", B=5.0, mean=m["mean"], half_range=m["half_range"],
             ast_param_idxs=np.arange(2), rho_idxs=np.arange(2, 2 + F),
             context_param_names=np.array(NAMES), marginalized_param_names=np.array([], dtype=str),
             fobs=np.asarray(sutils.get_harmonic_frequencies(F, tspan)), tspan=tspan, n_freqs=F,
             zero_handling="offset", rho_offset=C_OFF, slab_dex=SLAB, slab_shape="truncnorm",
             pspace_class="PS_Pandora_Phenom")
    run = make_run(files, name="ng20", flow="ng20_offset", priors=None)
    got, b = runner_log_density(run, point)
    sig = b.rn.astro_signal
    assert sig.rho_offset == C_OFF and sig.slab_dex == SLAB
    assert b.rn.model.get_param_names()[b.rn.common_param_slice][0] == "astro_log10_rho_plus_c_0"

    from ATLAS.astro.priors import pspace_log_prior
    want = reference_log_density(run, load_astro_flow(str(ng20)), point(b),
                                 fixed_astro=files["fixed"],
                                 astro_log_prior=pspace_log_prior("PS_Pandora_Phenom", NAMES))
    assert np.isfinite(got) and abs(got - want) <= 1e-10 * abs(want)
