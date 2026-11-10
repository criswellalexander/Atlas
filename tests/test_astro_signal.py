"""The `astro` signal type: a free-spectrum HD GWB with a flow hyperprior.

The reference for every exact check is the same model built with a `cor`
block (`gwb_free_spectrum`, HD ORF, the same rho bounds): the astro model must
add exactly the flow density and the theta prior to its joint density, and
nothing else. The flax-only tests run in atlas-env-1; the zuko conversion and
the posterior comparison with the standalone `AstroInferenceModel` (PTMCMC,
zuko flow) need atlas-env-2.
"""
import warnings

import numpy as np
import pytest

pytest.importorskip("flax")
pytest.importorskip("distrax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from numpyro.infer import MCMC, NUTS, init_to_value  # noqa: E402
from numpyro.infer.util import log_density  # noqa: E402

from ATLAS.data import PTA_Data  # noqa: E402
from ATLAS.model import model_maker  # noqa: E402
from ATLAS.model_builder import ModelBuilder  # noqa: E402
from ATLAS.nMatrix.base import WhiteCov  # noqa: E402
from ATLAS.psd_functions import gwb_free_spectrum, hd_orf, powerlaw  # noqa: E402
from ATLAS.samplers.canetoadracing import MultiHMCGibbs  # noqa: E402
from ATLAS.signals import signals_utils as sutils  # noqa: E402
from ATLAS.signals.correlated.astro_gwb import AstroGWB  # noqa: E402

from . import harness as H  # noqa: E402
from .test_flax_flows import F, P, _flow  # noqa: E402

pytestmark = pytest.mark.astro

NPSR, N_IRN = 3, 6
IRN_LO, IRN_HI = jnp.array([-20.0, 0.0]), jnp.array([-11.0, 7.0])
TIGHT = 1e-12


def make_data(linear_timing=True, marg_timing=False):
    '''

    Synthetic PTA data shared by the astro and reference models.

    Arguments
    ------------------------------
    linear_timing : bool
    	Keep the linear timing model. (Default: True)
    marg_timing : bool
    	Marginalize the timing model. (Default: False)

    Returns
    ------------------------------
    data : ATLAS.data.PTA_Data
    	Data with F GWB bins and N_IRN intrinsic bins, and its white-noise
    	covariance built (as `make_white_noise` does before `make_red_noise`).
    psrs : list
    	The synthetic pulsars.

    '''
    psrs = H.load_psrs("synth", NPSR)
    data = PTA_Data(
        psrs, num_gwb_bins=F, num_irn_bins=N_IRN, num_dm_bins=None, num_det_bins=None,
        adaptus_basis=None, adaptus_size=None, fixed_white_noise_params=None,
        linear_timing=linear_timing, marg_timing=marg_timing, diag_white_cov=False,
        fixed_res=False, timfiles=None, parfiles=None, noise_dict=None, dm_ref_freq=H.DM_REF_FREQ)
    WhiteCov(data=data, stabilize_TNT=False, include_ecorr=True)
    return data, psrs


def make_flow(data, kind="coupling_rqs", seed=0):
    '''

    A small random-weight 'rho|theta' flow trained "at" the data's frequencies.

    Arguments
    ------------------------------
    data : ATLAS.data.PTA_Data
    	Supplies the Tspan, hence the frequencies recorded in the metadata.
    kind : str
    	Flow architecture, as in `test_flax_flows._flow`. (Default: 'coupling_rqs')
    seed : int
    	Weight and mapping seed. (Default: 0)

    Returns
    ------------------------------
    flow : ATLAS.astro.flows.flax.FlaxAstroFlow
    	The flow, with `f_conv` and `tspan` in its metadata.

    '''
    flow = _flow(kind, seed=seed)
    flow.metadata.update(f_conv=list(np.asarray(sutils.get_harmonic_frequencies(F, data.pta_tspan))),
                         tspan=float(data.pta_tspan))
    return flow


def build(data, psrs, model_string, flow, common="astro", **astro_kw):
    '''

    Build the astro model, or the reference model with a `cor` block in its
    place (free spectrum, HD, the flow's training range as the rho box).

    Arguments
    ------------------------------
    data : ATLAS.data.PTA_Data
    	The data.
    psrs : list
    	The pulsars (for the white-noise vector).
    model_string : str
    	Combination string written with 'astro'; 'cor' replaces it for the reference.
    flow : ATLAS.astro.flows.flax.FlaxAstroFlow
    	The astro flow (also sets the reference's rho bounds).
    common : str
    	'astro' or 'cor'. (Default: 'astro')
    astro_kw : dict
    	Extra `make_red_noise` keywords for the astro block.

    Returns
    ------------------------------
    rn : ATLAS.signals.factorized.base.SuperSignal
    	The assembled signal.
    helpers : tuple
    	TNT, TNr, rNr, logdet_N at the harness white noise.
    wn_vec : jax.Array
    	The white-noise vector the helpers were built with.

    '''
    lo = flow.mean_gwb - flow.half_range_gwb
    hi = flow.mean_gwb + flow.half_range_gwb
    kw = dict(irn_psd_function=powerlaw, irn_lower_bound_psd=IRN_LO, irn_upper_bound_psd=IRN_HI,
              gwb_lower_bound_psd=lo, gwb_upper_bound_psd=hi)
    if common == "astro":
        kw.update(astro_flow=flow, **astro_kw)
    else:
        model_string = model_string.replace("astro", "cor")
        kw.update(gwb_psd_function=gwb_free_spectrum, orf_function=hd_orf)
    rn = ModelBuilder(data=data).make_red_noise(model_string, **kw)
    wn_vec = H.white_noise_vector(psrs, True)
    helpers = rn.get_helpers(reff=jnp.concat(data.raw_residuals), white_noise_params=wn_vec)
    return rn, helpers, wn_vec


@pytest.fixture(scope="module")
def setup():
    data, psrs = make_data()
    return data, psrs, make_flow(data)


def point(rn, flow, marg, seed=0, theta=None):
    '''

    A parameter point inside every prior box.

    Arguments
    ------------------------------
    rn : SuperSignal
    	The model (sets the z_a shape and the red-noise box).
    flow : FlaxAstroFlow
    	The flow (sets the theta draw).
    marg : bool
    	marg_over_non_gwb, which sets the z_a shape.
    seed : int
    	Seed of the draw. (Default: 0)
    theta : array_like or None
    	Varied theta; None draws it inside the training range. (Default: None)

    Returns
    ------------------------------
    params : dict
    	Values of 'red_noise', 'z_a' and 'astro_params'.

    '''
    rng = np.random.default_rng(seed)
    lo, hi = (np.asarray(a) for a in (rn.model.lower_prior_lim_all, rn.model.upper_prior_lim_all))
    red = lo + (hi - lo) * rng.uniform(0.2, 0.8, size=lo.shape)
    n_z = 2 * rn.model.crn_bins if marg else rn.nmodes
    if theta is None:
        theta = np.asarray(flow.mean_ast) + 0.3 * np.asarray(flow.half_range_ast) * rng.uniform(-1, 1, P)
    return dict(red_noise=jnp.asarray(red),
                z_a=jnp.asarray(rng.normal(size=(rn.npsrs, n_z))),
                astro_params=jnp.asarray(theta))


def maker_kwargs(rn, helpers, wn_vec, marg, vary_white):
    raw = jnp.concat(rn.data.raw_residuals)
    if vary_white:
        return dict(raw_residuals=raw, super_sig=rn, marg_over_non_gwb=marg, vary_white=True,
                    wn_lower_bound=wn_vec - 1.0, wn_upper_bound=wn_vec + 1.0)
    return dict(raw_residuals=raw, super_sig=rn, marg_over_non_gwb=marg, helpers=helpers)


# ---- construction -----------------------------------------------------------------------

@pytest.mark.parametrize("model_string", ["ltm|unc+astro->unc", "ltm|unc;astro"])
def test_construction(setup, model_string):
    data, psrs, flow = setup
    rn, _, _ = build(data, psrs, model_string, flow)
    sig = rn.signal_map["astro"]
    assert isinstance(sig, AstroGWB) and rn.has_astro and rn.has_common and not rn.has_cor
    assert rn.common_name == "astro" and rn.astro_signal is sig
    # default boxes are the training ranges
    m = rn.model
    rho_lo = np.asarray(m.lower_prior_lim_all)[rn.common_param_slice]
    rho_hi = np.asarray(m.upper_prior_lim_all)[rn.common_param_slice]
    assert np.array_equal(rho_lo, np.asarray(flow.mean_gwb - flow.half_range_gwb))
    assert np.array_equal(rho_hi, np.asarray(flow.mean_gwb + flow.half_range_gwb))
    assert np.array_equal(sig.astro_lower, flow.mean_ast - flow.half_range_ast)
    assert np.array_equal(sig.astro_upper, flow.mean_ast + flow.half_range_ast)
    names = m.get_param_names()
    assert names[rn.common_param_slice] == [f"astro_halflog10_rho_{i}" for i in range(F)]
    assert not any(n.startswith("gwb_") for n in names)
    assert sig.astro_param_names == ["a", "b"]


def test_cor_models_are_unchanged(setup):
    data, psrs, flow = setup
    rn, _, _ = build(data, psrs, "ltm|unc+astro->unc", flow, common="cor")
    assert rn.has_cor and not rn.has_astro and rn.common_name == "cor" and rn.astro_signal is None
    assert [n for n in rn.model.get_param_names() if "rho" in n] == \
        [f"gwb_halflog10_rho_{i}" for i in range(F)]


def test_construction_errors(setup):
    data, psrs, flow = setup
    mb = ModelBuilder(data=data)
    base = dict(irn_psd_function=powerlaw, irn_lower_bound_psd=IRN_LO, irn_upper_bound_psd=IRN_HI)
    with pytest.raises(ValueError, match="cannot be combined"):
        mb.make_red_noise("ltm|unc+astro->unc;cor", astro_flow=flow, gwb_psd_function=gwb_free_spectrum,
                          orf_function=hd_orf, gwb_lower_bound_psd=-jnp.ones(F) * 9,
                          gwb_upper_bound_psd=-jnp.ones(F) * 5, **base)
    with pytest.raises(ValueError, match="needs a trained flow"):
        mb.make_red_noise("ltm|unc+astro->unc", **base)
    with pytest.raises(ValueError, match="'rho\\|theta' flow"):
        AstroGWB(data, _flow("coupling_rqs", nf_type="theta|rho"))
    other_tspan = make_flow(data)
    other_tspan.metadata["f_conv"] = list(np.asarray(other_tspan.metadata["f_conv"]) * 1.01)
    with pytest.raises(ValueError, match="pta_tspan"):
        AstroGWB(data, other_tspan)
    with pytest.raises(ValueError, match="halflog10_rho bounds"):
        AstroGWB(data, flow, lower_bound_psd=-jnp.ones(F + 1) * 9)
    with pytest.raises(ValueError, match="astrophysical bounds"):
        AstroGWB(data, flow, astro_lower=jnp.zeros(P + 1))


def test_construction_warnings(setup):
    data, psrs, flow = setup
    lo, hi = flow.mean_gwb - flow.half_range_gwb, flow.mean_gwb + flow.half_range_gwb
    with pytest.warns(UserWarning, match="clip"):
        AstroGWB(data, flow, lower_bound_psd=lo + 0.1, upper_bound_psd=hi)
    with pytest.warns(UserWarning, match="extrapolating"):
        AstroGWB(data, flow, astro_upper=flow.mean_ast + 2 * flow.half_range_ast)
    no_freqs = make_flow(data)
    del no_freqs.metadata["f_conv"]
    with pytest.warns(UserWarning, match="f_conv"):
        AstroGWB(data, no_freqs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        AstroGWB(data, flow)          # defaults: no warning


# ---- exact joint density --------------------------------------------------------------

def extra_log_prior(theta):
    # a Gaussian on the first varied parameter, to exercise astro_log_prior
    return -0.5 * ((theta[0] - 0.1) / 0.7) ** 2


@pytest.mark.parametrize("fixed", [False, True], ids=["all-varied", "one-fixed"])
@pytest.mark.parametrize("vary_white", [False, True], ids=["fixed-wn", "vary-wn"])
@pytest.mark.parametrize("model_string", ["ltm|unc+astro->unc", "ltm|unc;astro"],
                         ids=["nested", "separate"])
@pytest.mark.parametrize("marg", [False, True], ids=["reparam", "partial-marg"])
def test_log_density_adds_exactly_the_hyperprior(setup, marg, model_string, vary_white, fixed):
    '''

    log p_astro - log p_cor = log p_flow(rho | theta) + astro_log_prior(theta)
    - sum log(theta box widths), at the same site values.

    Arguments
    ------------------------------
    setup : tuple
    	Module fixture (data, pulsars, flow).
    marg : bool
    	marg_over_non_gwb branch.
    model_string : str
    	Nested or separate common-process layout.
    vary_white : bool
    	Sample the white noise (helpers rebuilt in the model).
    fixed : bool
    	Hold the first astrophysical parameter fixed.

    Returns
    ------------------------------
    None

    '''
    data, psrs, flow = setup
    astro_kw = dict(astro_log_prior=extra_log_prior)
    if fixed:
        astro_kw["fixed_astro"] = {0: float(flow.mean_ast[0]) + 0.05}
    rn_a, helpers_a, wn_vec = build(data, psrs, model_string, flow, **astro_kw)
    rn_c, helpers_c, _ = build(data, psrs, model_string, flow, common="cor")
    sig = rn_a.astro_signal

    params = point(rn_a, flow, marg)
    params["astro_params"] = params["astro_params"][sig.varied]
    if vary_white:
        params["white_noise"] = wn_vec
    got = float(log_density(model_maker, (), maker_kwargs(rn_a, helpers_a, wn_vec, marg, vary_white),
                            params)[0])
    ref_params = {k: v for k, v in params.items() if k != "astro_params"}
    ref = float(log_density(model_maker, (), maker_kwargs(rn_c, helpers_c, wn_vec, marg, vary_white),
                            ref_params)[0])

    theta_full = np.asarray(sig.astro_container).copy()
    theta_full[sig.varied] = np.asarray(params["astro_params"])
    rho = params["red_noise"][rn_a.common_param_slice]
    expected = (float(flow.log_prob(rho, jnp.asarray(theta_full)))
                + float(extra_log_prior(params["astro_params"]))
                - float(np.sum(np.log(np.asarray(sig.astro_upper) - np.asarray(sig.astro_lower)))))
    assert abs((got - ref) - expected) <= TIGHT * max(abs(got), abs(ref))


def test_all_theta_fixed_has_no_site(setup):
    data, psrs, flow = setup
    fixed = {i: float(v) for i, v in enumerate(np.asarray(flow.mean_ast))}
    rn, helpers, wn_vec = build(data, psrs, "ltm|unc+astro->unc", flow, fixed_astro=fixed)
    params = point(rn, flow, True)
    del params["astro_params"]
    _, trace = log_density(model_maker, (), maker_kwargs(rn, helpers, wn_vec, True, False), params)
    assert "astro_params" not in trace and "astro_hyperprior" in trace
    rho = params["red_noise"][rn.common_param_slice]
    assert np.isclose(float(jnp.sum(trace["astro_hyperprior"]["fn"].log_factor)),
                      float(flow.log_prob(rho, flow.mean_ast)), rtol=TIGHT, atol=0)


def test_gradients_match_finite_differences(setup):
    data, psrs, flow = setup
    rn, helpers, wn_vec = build(data, psrs, "ltm|unc+astro->unc", flow)
    kw = maker_kwargs(rn, helpers, wn_vec, True, False)
    params = point(rn, flow, True)
    f = jax.jit(lambda p: log_density(model_maker, (), kw, p)[0])
    grad = jax.grad(f)(params)
    h = 1e-5
    # every rho and theta component, and one intrinsic-noise parameter
    checks = [("red_noise", i) for i in range(rn.common_param_slice.start, rn.common_param_slice.stop)]
    checks += [("astro_params", i) for i in range(P)] + [("red_noise", 0)]
    for site, i in checks:
        up = dict(params, **{site: params[site].at[i].add(h)})
        dn = dict(params, **{site: params[site].at[i].add(-h)})
        fd = (float(f(up)) - float(f(dn))) / (2 * h)
        ad = float(grad[site][i])
        assert abs(fd - ad) <= 1e-5 * max(1.0, abs(ad)), (site, i, fd, ad)


def test_log_prior_matches_standalone_model():
    '''

    Ties the signal to Stage 1c (itself exact against Pandora): the flow term
    of `AstroInferenceModel.lnlikelihood` equals `AstroGWB.log_prior` at the
    same point, with the same parameter order.

    Arguments
    ------------------------------
    None

    Returns
    ------------------------------
    None

    '''
    from ATLAS.astro.inference import AstroInferenceModel, HDAstroRedModel
    from ATLAS.signals.signals_utils import blockPsrs2sigma, blockVec2sigmaVec

    data, psrs = make_data(linear_timing=False, marg_timing=True)
    flow = make_flow(data)
    rn_c, (TNT, TNr, rNr, logdet_N), _ = build(data, psrs, "unc+astro->unc", flow, common="cor")
    red = HDAstroRedModel(blockPsrs2sigma(TNT), blockVec2sigmaVec(TNr), rn_c.model, data.pta_tspan,
                          rNr=rNr, logdet_N=logdet_N)
    fixed = {1: float(flow.mean_ast[1])}
    lo, hi = flow.mean_ast - flow.half_range_ast, flow.mean_ast + flow.half_range_ast
    standalone = AstroInferenceModel(red, flow, lo, hi, astro_log_prior=extra_log_prior,
                                     fixed_astro=fixed)
    rn_a, _, _ = build(data, psrs, "unc+astro->unc", flow, fixed_astro=fixed,
                       astro_log_prior=extra_log_prior)

    params = point(rn_a, flow, True)
    xs = np.asarray(params["red_noise"])
    theta_v = np.asarray(params["astro_params"])[rn_a.astro_signal.varied]
    assert red.param_names[red.gwb_slice] == [n.replace("astro_", "gwb_") for n in
                                               rn_a.model.get_param_names()[rn_a.common_param_slice]]
    flow_term = standalone.lnlikelihood(np.concatenate([xs, theta_v])) - float(red.lnlikelihood(xs)[0])
    ours = float(rn_a.astro_signal.log_prior(params["red_noise"][rn_a.common_param_slice], theta_v))
    assert abs(flow_term - ours) <= TIGHT * max(1.0, abs(ours))


# ---- samplers -----------------------------------------------------------------------

def init_values(rn, flow, marg):
    p = point(rn, flow, marg, seed=3)
    return init_to_value(values=p)


@pytest.mark.parametrize("marg", [False, True], ids=["reparam", "partial-marg"])
def test_nuts_runs(setup, marg):
    data, psrs, flow = setup
    rn, helpers, wn_vec = build(data, psrs, "ltm|unc+astro->unc", flow)
    mcmc = MCMC(NUTS(model_maker, init_strategy=init_values(rn, flow, marg), max_tree_depth=6),
                num_warmup=30, num_samples=30, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), extra_fields=("potential_energy",),
             **maker_kwargs(rn, helpers, wn_vec, marg, False))
    theta = np.asarray(mcmc.get_samples()["astro_params"])
    sig = rn.astro_signal
    assert np.all(np.isfinite(np.asarray(mcmc.get_extra_fields()["potential_energy"])))
    assert np.all((theta > np.asarray(sig.astro_lower)) & (theta < np.asarray(sig.astro_upper)))


@pytest.mark.parametrize("groups", [
    [["red_noise", "astro_params"], ["z_a"]],
    [["red_noise"], ["astro_params"], ["z_a"]],
], ids=["two-blocks", "three-blocks"])
def test_multihmcgibbs_runs(setup, groups):
    data, psrs, flow = setup
    rn, helpers, wn_vec = build(data, psrs, "ltm|unc+astro->unc", flow)
    kernels = [NUTS(model_maker, init_strategy=init_values(rn, flow, True), max_tree_depth=5)
               for _ in groups]
    mcmc = MCMC(MultiHMCGibbs(kernels, groups), num_warmup=20, num_samples=20, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(1), **maker_kwargs(rn, helpers, wn_vec, True, False))
    samples = mcmc.get_samples()
    assert set(samples) >= {"red_noise", "astro_params", "z_a"}
    assert all(np.all(np.isfinite(np.asarray(v))) for v in samples.values())


# ---- zuko (atlas-env-2) -------------------------------------------------------------------

def zuko_flow(data, seed=1):
    '''

    A random-weight zuko 'rho|theta' flow with the data's frequencies.

    Arguments
    ------------------------------
    data : ATLAS.data.PTA_Data
    	Supplies the frequencies recorded in the metadata.
    seed : int
    	torch seed of the weights. (Default: 1)

    Returns
    ------------------------------
    zflow : ATLAS.astro.flows.zuko.ZukoAstroFlow
    	The flow.

    '''
    torch = pytest.importorskip("torch")
    pytest.importorskip("zuko")
    from ATLAS.astro.flows import zuko as zf
    from .test_flax_flows import _mapping

    torch.manual_seed(seed)
    f_conv = list(np.asarray(sutils.get_harmonic_frequencies(F, data.pta_tspan)))
    return zf.ZukoAstroFlow(zf.make_zuko_flow(F, P, hidden_features=[16, 16]), "rho|theta",
                            _mapping(seed), P,
                            metadata=dict(param_names=["a", "b"], f_conv=f_conv,
                                          tspan=float(data.pta_tspan)))


def test_zuko_flow_is_converted(setup):
    data, psrs, _ = setup
    zflow = zuko_flow(data)
    sig = AstroGWB(data, zflow)
    rho = np.asarray(zflow.mean_gwb) + 0.2
    theta = np.asarray(zflow.mean_ast) - 0.1
    want = float(zflow.log_prob(rho[None], theta[None])[0])
    got = float(sig.log_prior(jnp.asarray(rho), jnp.asarray(theta)))
    assert abs(got - want) <= 1e-10 * max(1.0, abs(want))


@pytest.mark.slow
def test_posterior_matches_standalone_ptmcmc(tmp_path):
    '''

    NUTS on the global-fit model and the standalone PTMCMC model target the
    same posterior over (intrinsic noise, rho, theta) at fixed white noise.
    Means agree to a fraction of the posterior width, and widths to 20%.

    Arguments
    ------------------------------
    tmp_path : pathlib.Path
    	PTMCMC output directory.

    Returns
    ------------------------------
    None

    '''
    pytest.importorskip("PTMCMCSampler")
    from ATLAS.astro.inference import AstroInferenceModel, HDAstroRedModel
    from ATLAS.signals.signals_utils import blockPsrs2sigma, blockVec2sigmaVec

    from ATLAS.astro.flows.flax import FlaxAstroFlow

    data, psrs = make_data(linear_timing=False, marg_timing=True)
    zflow = zuko_flow(data)
    flow = FlaxAstroFlow.from_zuko(zflow)
    rn_c, (TNT, TNr, rNr, logdet_N), _ = build(data, psrs, "unc+astro->unc", flow, common="cor")
    red = HDAstroRedModel(blockPsrs2sigma(TNT), blockVec2sigmaVec(TNr), rn_c.model, data.pta_tspan,
                          rNr=rNr, logdet_N=logdet_N)
    lo, hi = np.asarray(zflow.mean_ast - zflow.half_range_ast), np.asarray(zflow.mean_ast + zflow.half_range_ast)
    standalone = AstroInferenceModel(red, zflow, lo, hi)
    standalone.sample(200_000, str(tmp_path), seed=0, resume=False)
    # the saved chain is thinned (every 10th step); drop the first quarter as burn-in
    chain = np.loadtxt(tmp_path / "chain_1.txt")
    chain = chain[len(chain) // 4:, :red.lower.size + len(lo)]

    rn, helpers, wn_vec = build(data, psrs, "unc+astro->unc", flow)
    mcmc = MCMC(NUTS(model_maker, init_strategy=init_values(rn, flow, True)),
                num_warmup=1000, num_samples=4000, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), **maker_kwargs(rn, helpers, wn_vec, True, False))
    s = mcmc.get_samples()
    nuts = np.concatenate([np.asarray(s["red_noise"]), np.asarray(s["astro_params"])], axis=1)

    std = chain.std(0)
    assert np.all(np.abs(nuts.mean(0) - chain.mean(0)) < 0.2 * std)
    assert np.all(np.abs(nuts.std(0) / std - 1) < 0.2)


# ---- GPU --------------------------------------------------------------------------------

@pytest.mark.gpu
def test_gpu_matches_cpu(setup):
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("no GPU")
    data, psrs, flow = setup
    rn, helpers, wn_vec = build(data, psrs, "ltm|unc+astro->unc", flow)
    kw = maker_kwargs(rn, helpers, wn_vec, True, False)
    params = point(rn, flow, True)
    gpu = float(jax.jit(lambda p: log_density(model_maker, (), kw, p)[0])(params))
    with jax.default_device(jax.devices("cpu")[0]):
        cpu = float(jax.jit(lambda p: log_density(model_maker, (), kw, p)[0])(
            jax.device_put(params, jax.devices("cpu")[0])))
    assert abs(gpu - cpu) <= 1e-10 * abs(cpu)
    mcmc = MCMC(NUTS(model_maker, init_strategy=init_values(rn, flow, True), max_tree_depth=5),
                num_warmup=10, num_samples=10, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), **kw)
    assert np.all(np.isfinite(np.asarray(mcmc.get_samples()["astro_params"])))
