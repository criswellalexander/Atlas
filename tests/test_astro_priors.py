"""holodeck parameter-space priors as JAX log densities (ATLAS.astro.priors).

Each term is checked against scipy, holodeck's own draws must all lie in the
support, and the 2D hardening block must be normalized. With
``ATLAS_RUN_PANDORA_DIR`` set to the directory holding ng20astromethods'
``run_pandora.py`` (atlas-env-2), the whole prior is also checked against its
``build_astro_prior``, which this module ports.
"""
import importlib.util
import logging
import os
import sys

import numpy as np
import pytest
from scipy import stats

holodeck = pytest.importorskip("holodeck")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from holodeck.librarian import lib_tools  # noqa: E402

from ATLAS.astro.flows import trainset  # noqa: E402
from ATLAS.astro.priors import get_param_space, pspace_log_prior  # noqa: E402

pytestmark = pytest.mark.astro
holodeck.log.setLevel(logging.ERROR)

NG20 = "PS_NG20_Fiducial"
PHENOM = "PS_Pandora_Phenom"


def _space(name, nsamples=None, seed=0):
    if name == PHENOM:
        return trainset.pandora_phenom_param_space(nsamples=nsamples, seed=seed)
    return holodeck.librarian.param_spaces_dict[name](nsamples=nsamples, seed=seed)


def _dist(ps, cls):
    return next(d for d in ps._parameters if type(d) is cls)


def _names(d):
    return list(d.name) if isinstance(d.name, tuple) else [d.name]


def test_phenom_matches_scipy():
    """Uniform and normal terms, with the context in a different order than the space."""
    ps = _space(PHENOM, nsamples=200)
    order = list(np.random.default_rng(0).permutation(len(ps.param_names)))
    names = [ps.param_names[i] for i in order]
    theta = ps.param_samples[:, order]
    ref = np.zeros(len(theta))
    for d in ps._parameters:
        x = theta[:, names.index(d.name)]
        if type(d) is lib_tools.PD_Uniform:
            ref += np.where((x >= d._lo) & (x <= d._hi), -np.log(d._hi - d._lo), -np.inf)
        else:
            ref += stats.norm(d._mean, d._stdev).logpdf(x)
    got = np.asarray(pspace_log_prior(PHENOM, names)(theta))
    assert np.all(np.isfinite(got))
    assert np.allclose(got, ref, rtol=0, atol=1e-12)


def test_ng20_draws_lie_in_the_support():
    ps = _space(NG20, nsamples=300)
    got = np.asarray(pspace_log_prior(ps, ps.param_names)(ps.param_samples))
    assert got.shape == (300,) and np.all(np.isfinite(got))


def test_mvnormal_term_matches_scipy():
    """Only the GSMF block in the context: every other block is marginalized."""
    ps = _space(NG20, nsamples=50)
    d = _dist(ps, lib_tools.PD_MVNormal)
    names = _names(d)
    theta = ps.param_samples[:, [ps.param_names.index(n) for n in names]]
    got = np.asarray(pspace_log_prior(ps, names)(theta))
    ref = stats.multivariate_normal(d._frozen_dist.mean, d._frozen_dist.cov).logpdf(theta)
    assert np.allclose(got, ref, rtol=1e-13, atol=1e-12)


def test_uniform_term_is_zero_outside():
    ps = _space(NG20)
    d = _dist(ps, lib_tools.PD_Uniform)
    lp = pspace_log_prior(ps, [d.name])
    x = np.array([d._lo, 0.5 * (d._lo + d._hi), d._hi, d._lo - 1e-9, d._hi + 1e-9])[:, None]
    want = np.full(5, -np.log(d._hi - d._lo))
    want[3:] = -np.inf
    assert np.array_equal(np.asarray(lp(x)), want)


def test_2d_hardening_block():
    """Uniform in area over its support: constant inside, -inf below y_lo(x),
    and normalized over its box (Monte Carlo)."""
    ps = _space(NG20, nsamples=300)
    d = _dist(ps, lib_tools.PD_2D_Uniform_Variable_Ymin)
    names = _names(d)
    lp = pspace_log_prior(ps, names)
    draws = ps.param_samples[:, [ps.param_names.index(n) for n in names]]
    vals = np.asarray(lp(draws))
    assert np.all(np.isfinite(vals)) and np.ptp(vals) == 0

    x = np.linspace(d._x_lo, d._x_hi, 7)
    y_lo = d._y_lo_func(x, d._parspace_defaults, absmin=d._y_abs_lo, **d._y_lo_kwargs)
    below = np.stack([x, y_lo - 1e-3], axis=1)
    assert np.all(np.asarray(lp(below)) == -np.inf)

    rng = np.random.default_rng(1)
    n = 400_000
    y_min = d._y_abs_lo
    pts = np.stack([rng.uniform(d._x_lo, d._x_hi, n), rng.uniform(y_min, d._y_hi, n)], axis=1)
    dens = np.exp(np.asarray(lp(pts)))
    box = (d._x_hi - d._x_lo) * (d._y_hi - y_min)
    integral = box * dens.mean()
    assert abs(integral - 1) < 4 * box * dens.std() / np.sqrt(n)


def test_jit_and_gradient():
    ps = _space(NG20, nsamples=5)
    lp = pspace_log_prior(ps, ps.param_names)
    theta = jnp.asarray(ps.param_samples)
    assert np.array_equal(np.asarray(jax.jit(lp)(theta)), np.asarray(lp(theta)))
    grad = jax.vmap(jax.grad(lp))(theta)
    assert np.all(np.isfinite(np.asarray(grad)))
    # the GSMF block's gradient is the MVN score
    d = _dist(ps, lib_tools.PD_MVNormal)
    cols = [ps.param_names.index(n) for n in _names(d)]
    score = -np.linalg.solve(d._frozen_dist.cov, (ps.param_samples[:, cols] - d._frozen_dist.mean).T).T
    assert np.allclose(np.asarray(grad)[:, cols], score, rtol=1e-10, atol=1e-10)


def test_errors():
    ps = _space(NG20)
    gsmf = _names(_dist(ps, lib_tools.PD_MVNormal))
    with pytest.raises(NotImplementedError, match="only partly"):
        pspace_log_prior(ps, gsmf[:3])
    with pytest.raises(NotImplementedError, match="PD_Uniform_Log"):
        pspace_log_prior("PS_Astro_Strong_Hard_All",
                         get_param_space("PS_Astro_Strong_Hard_All").param_names)
    with pytest.raises(ValueError, match="not in the parameter space"):
        pspace_log_prior(ps, ["theta_0", "theta_1"])   # e.g. a Pandora flow without names
    with pytest.raises(KeyError, match="unknown parameter space"):
        get_param_space("PS_Not_A_Space")
    assert get_param_space(PHENOM).name == PHENOM


# ---- identity with run_pandora -----------------------------------------------------------

def _run_pandora():
    where = os.environ.get("ATLAS_RUN_PANDORA_DIR")
    if not where:
        pytest.skip("set ATLAS_RUN_PANDORA_DIR to the directory holding run_pandora.py")
    spec = importlib.util.spec_from_file_location("run_pandora", os.path.join(where, "run_pandora.py"))
    module = importlib.util.module_from_spec(spec)
    # run_pandora imports its sibling pta_matrices.py (validate_flow it locates itself)
    sys.path.insert(0, where)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        pytest.skip(f"run_pandora does not import here: {exc}")
    finally:
        sys.path.remove(where)
        for name in ("validate_flow", "pta_matrices", "train_flow"):
            sys.modules.pop(name, None)
    return module


@pytest.mark.parametrize("fixed", [{}, {"mmb_zplaw_amp": 0.0, "gmr_normz": -0.2}])
def test_matches_run_pandora(fixed):
    rp = _run_pandora()
    ps = _space(NG20, nsamples=200)
    names = list(ps.param_names)
    theta = ps.param_samples.copy()
    for n, v in fixed.items():
        theta[:, names.index(n)] = v
    varied = [i for i, n in enumerate(names) if n not in fixed]
    ref_fn = rp.build_astro_prior(ps, names, fixed)
    ref = np.array([ref_fn(t[varied]) for t in theta])
    got = np.asarray(pspace_log_prior(ps, names)(theta))
    assert np.all(np.isfinite(ref))
    assert np.allclose(got, ref, rtol=1e-13, atol=1e-12)


def test_offset_psd_matches_run_pandora():
    """The zero-slab PSD of an offset-mode astro block is run_pandora's offset_free_spectrum,
    except that slab bins carry RHO_FLOOR**2 instead of 0 (see astro_gwb.RHO_FLOOR)."""
    rp = _run_pandora()
    pytest.importorskip("flax")
    from ATLAS.signals.correlated.astro_gwb import RHO_FLOOR, _offset_free_spectrum

    c = 1e-9
    y = np.linspace(np.log10(c) - 0.5, np.log10(c) + 2.0, 41)
    ours = np.asarray(_offset_free_spectrum(c)(None, None, *y))[:, 0]
    theirs = np.asarray(rp.offset_free_spectrum(c)(None, None, *y))[:, 0]
    slab = theirs == 0
    assert slab.any() and (~slab).any()
    assert np.array_equal(ours[~slab], theirs[~slab])
    assert np.all(ours[slab] == RHO_FLOOR**2)
