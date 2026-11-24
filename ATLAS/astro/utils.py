"""
Helpers for :mod:`ATLAS.astro`: training-set file layout,
frequency conventions, the per-draw simulation worker, GWB prior bounds, and
the GWB spectrum summaries of an astro run (as ng20astromethods' run_pandora
writes them).
"""

import glob
import json
import os
import re

import numpy as np


def conversion_freqs(freqs, tspan, n_freqs):
    """Frequencies for the ``hc -> log10_rho`` conversion (see ``astro.gwb_frequencies``)."""
    if freqs is None:
        return np.arange(1 / tspan, (n_freqs + .001) / tspan, 1 / tspan)
    if isinstance(freqs, str):
        raise ValueError(f"freqs must be None or an array, got {freqs!r}")
    f = np.asarray(freqs, dtype=np.float64)
    if f.shape != (n_freqs,):
        raise ValueError(f"freqs array has shape {f.shape}, expected ({n_freqs},)")
    return f


def default_tag(tspan):
    """Pandora's file suffix, e.g. ``'20yrs'``."""
    from holodeck.constants import YR

    return f"{round(tspan / YR, 6):g}yrs"


def metadata_path(save_dir, tag):
    return os.path.join(save_dir, f"trainset_{tag}_metadata.json")


def draw_path(save_dir, idx, tag):
    return os.path.join(save_dir, f"{idx}_{tag}.npy")


def read_metadata(save_dir, tag):
    """The training set's metadata JSON, or ``{}`` if there is none."""
    path = metadata_path(save_dir, tag)
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def draw_indices_on_disk(save_dir, tag):
    """Sorted draw indices of the ``{i}_{tag}.npy`` files in ``save_dir``."""
    pattern = re.compile(rf"^(\d+)_{re.escape(tag)}\.npy$")
    found = (pattern.match(os.path.basename(p))
             for p in glob.glob(os.path.join(save_dir, f"*_{tag}.npy")))
    return np.array(sorted(int(m.group(1)) for m in found if m), dtype=int)


def simulate_and_save(pspace, idx, params, path, tspan, n_freqs, n_real, freqs,
                      seed, sam_shape):
    """joblib worker: simulate one draw and save it unless it is non-finite."""
    # looked up at call time so tests can patch trainset.simulate_log10_rho
    from ATLAS.astro.flows import trainset

    rho = trainset.simulate_log10_rho(pspace, params, tspan, n_freqs, n_real, freqs=freqs,
                                   seed=seed, sam_shape=sam_shape)
    if not np.isfinite(rho).all():
        return idx, False
    np.save(path, rho)
    return idx, True


def gwb_bounds(gwb_psd_function, bounds, rho_bounds, crn_bins):
    """GWB PSD prior bounds: given, or ``rho_bounds`` per bin for a free spectrum."""
    import jax.numpy as jnp
    from ATLAS.psd_functions import gwb_free_spectrum

    if bounds is not None:
        return jnp.asarray(bounds[0]), jnp.asarray(bounds[1])
    if gwb_psd_function is not gwb_free_spectrum:
        raise ValueError("gwb_bounds is required unless the GWB PSD is gwb_free_spectrum")
    return jnp.full(crn_bins, float(rho_bounds[0])), jnp.full(crn_bins, float(rho_bounds[1]))


# ---- GWB spectrum of an astro run (run_pandora's postprocess) ----------------------

def gwb_log10_rho(samples, flow):
    """Samples of an astro model's sampled GWB variable -> log10 rho, NaN where rho = 0.

    For an offset-mode flow the samples are y = log10(rho + c); rho = 10^y - c,
    mapped draw by draw (no Jacobian: they are samples, not densities), and the
    zero slab (rho <= 0) becomes NaN.  Otherwise they already are log10 rho.
    """
    x = np.asarray(samples, dtype=np.float64)
    if flow.metadata.get("zero_handling") != "offset":
        return x
    rho = 10.0**x - float(flow.metadata["rho_offset"])
    out = np.full_like(x, np.nan)
    out[rho > 0] = np.log10(rho[rho > 0])
    return out


def flow_prior_log10_rho(flow, pspace, context_names, container, varied, lower, upper, n, seed):
    """The flow prior on the spectrum, in log10 rho: one flow draw per theta.

    theta comes from ``n`` draws of the holodeck parameter space (a name or an
    instance; ``context_names`` picks the flow's columns), with the fixed values
    of ``container`` inserted and kept only inside the varied parameters' box
    (``lower``, ``upper``), as run_pandora does.  With fixed parameters these
    are draws from the marginal, not the conditional, prior of the varied ones.
    """
    import jax
    import jax.numpy as jnp
    from ATLAS.astro.priors import get_param_space

    if isinstance(pspace, str):
        pspace = get_param_space(pspace, nsamples=int(n), seed=seed)
    names = list(pspace.param_names)
    theta = np.asarray(pspace.param_samples)[:, [names.index(nn) for nn in context_names]]
    fixed = np.setdiff1d(np.arange(len(context_names)), varied)
    theta[:, fixed] = np.asarray(container)[fixed]
    inside = np.all((theta[:, varied] > lower) & (theta[:, varied] < upper), axis=1)
    draws = flow.sample(jax.random.key(0 if seed is None else seed), (),
                        jnp.asarray(theta[inside]))                  # (n_inside, F)
    return gwb_log10_rho(np.asarray(draws), flow)


def spectrum_summary(post, prior=None):
    """``p_rho_zero`` and ``gwb_log10_rho`` summary blocks, as in run_pandora.

    ``post`` / ``prior`` are (N, F) log10 rho samples with NaN where rho = 0
    (``gwb_log10_rho``).  Each bin's intervals use its rho > 0 samples.
    """
    p_post = np.mean(np.isnan(post), axis=0)
    p_zero = dict(posterior=p_post.tolist())
    if prior is not None:
        p_zero["flow_prior"] = np.mean(np.isnan(prior), axis=0).tolist()
    bins = {}
    for kk in range(post.shape[1]):
        vals = post[np.isfinite(post[:, kk]), kk]
        qq = np.percentile(vals, [2.5, 16, 50, 84, 97.5]) if vals.size else np.full(5, np.nan)
        bins[f"log10_rho_{kk}"] = dict(median=float(qq[2]), ci68=[float(qq[1]), float(qq[3])],
                                       ci95=[float(qq[0]), float(qq[4])], p_rho_zero=float(p_post[kk]))
    return dict(p_rho_zero=p_zero, gwb_log10_rho=bins)
