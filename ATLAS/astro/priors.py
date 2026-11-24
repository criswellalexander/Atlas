'''

Astrophysical priors on the flow's context parameters theta.

``pspace_log_prior`` turns a holodeck parameter space into a pure-JAX log
prior density over the flow's context parameters, for use as
``AstroGWB``'s ``astro_log_prior``. It is a port of ``build_astro_prior`` in
ng20astromethods' ``run_pandora.py``, with the same terms and constants.

holodeck is only needed to build the prior; the returned function is plain
JAX.

'''

import logging

import numpy as np
import jax.numpy as jnp
from jax.scipy import stats as jstats

__all__ = ["get_param_space", "pspace_log_prior"]

# Name of Atlas's own Pandora phenom space (ATLAS.astro.flows.trainset), which
# is not in holodeck's registry
PANDORA_PHENOM = "PS_Pandora_Phenom"


def get_param_space(name):
    '''

    Build a holodeck parameter space by name, without drawing samples.

    Arguments
    ------------------------------
    name : str
    	A key of ``holodeck.librarian.param_spaces_dict``, or
    	'PS_Pandora_Phenom' for ``ATLAS.astro.flows.trainset.pandora_phenom_param_space``.

    Returns
    ------------------------------
    pspace : holodeck.librarian.lib_tools._Param_Space
    	The parameter space, built with ``nsamples=None``.

    '''
    if name == PANDORA_PHENOM:
        from ATLAS.astro.flows.trainset import pandora_phenom_param_space
        return pandora_phenom_param_space(nsamples=None)
    import holodeck.librarian
    spaces = holodeck.librarian.param_spaces_dict
    if name not in spaces:
        raise KeyError(f"unknown parameter space {name!r}; available: "
                       f"{sorted(spaces) + [PANDORA_PHENOM]}")
    return spaces[name](nsamples=None)


def pspace_log_prior(pspace, context_names, n_grid=2001):
    '''

    Log prior density of a holodeck parameter space over the flow's context
    parameters, as a pure-JAX function of the full theta (flow context order).

    The terms are those of ``run_pandora.build_astro_prior``:
    ``PD_Uniform`` (-log(hi - lo) inside, -inf outside), ``PD_Normal``,
    ``PD_MVNormal``, and ``PD_2D_Uniform_Variable_Ymin``. The last is uniform
    in area over {x_lo <= x <= x_hi, y_lo(x) <= y <= y_hi}, holodeck's
    sampling density: y_lo(x) is tabulated once on ``n_grid`` points and the
    area integrated with the trapezoid rule. A distribution none of whose
    parameters are context parameters is marginalized and drops out.

    The density is exactly -inf outside the support (a uniform range, or below
    y_lo(x)). Inside a sampler's box that is wider than the support, NUTS sees
    these edges as rejected (divergent) trajectories; they do not bias the
    posterior.

    Fixed parameters need no special handling: evaluating the joint density
    at the fixed values gives the conditional density of the varied ones up
    to a constant.

    Arguments
    ------------------------------
    pspace : holodeck.librarian.lib_tools._Param_Space or str
    	The parameter space, or its name (see ``get_param_space``).
    context_names : list of str
    	The flow's context parameter names, in flow order.
    n_grid : int
    	Points on which y_lo(x) of a 2D variable-ymin block is tabulated.
    	(Default: 2001)

    Returns
    ------------------------------
    log_prior : callable
    	``log_prior(theta) -> jax.Array``; theta has shape (..., len(context_names))
    	and the result the leading shape. Works under jit, grad and vmap.

    '''
    from holodeck.librarian import lib_tools

    if isinstance(pspace, str):
        pspace = get_param_space(pspace)
    context_names = list(context_names)

    terms = []                                  # (kind, context columns, constants)
    for pdist in pspace._parameters:
        dnames = pdist.name if isinstance(pdist.name, tuple) else (pdist.name,)
        in_ctx = [nn in context_names for nn in dnames]
        if not any(in_ctx):
            continue                            # marginalized: integrates out
        if not all(in_ctx):
            raise NotImplementedError(
                f"the multivariate prior block {dnames} is only partly in the flow's context")
        if pdist._clip is not None:
            raise NotImplementedError(
                f"the prior on {dnames} is clipped, which puts point masses at the clip edges")
        cols = np.array([context_names.index(nn) for nn in dnames])
        kind = type(pdist)
        if kind is lib_tools.PD_Uniform:
            terms.append(("uniform", cols, (pdist._lo, pdist._hi, -np.log(pdist._hi - pdist._lo))))
        elif kind is lib_tools.PD_Normal:
            terms.append(("normal", cols, (pdist._mean, pdist._stdev)))
        elif kind is lib_tools.PD_MVNormal:
            frozen = pdist._frozen_dist
            terms.append(("mvnormal", cols, (jnp.asarray(frozen.mean), jnp.asarray(frozen.cov))))
        elif kind is lib_tools.PD_2D_Uniform_Variable_Ymin:
            # y_lo(x) is smooth and non-increasing; a direct call costs ~0.5 ms
            # and logs warnings, so tabulate it once (as run_pandora does)
            import holodeck
            level = holodeck.log.level
            holodeck.log.setLevel(logging.ERROR)
            try:
                x_grid = np.linspace(pdist._x_lo, pdist._x_hi, n_grid)
                y_lo = pdist._y_lo_func(x_grid, pdist._parspace_defaults,
                                        absmin=pdist._y_abs_lo, **pdist._y_lo_kwargs)
            finally:
                holodeck.log.setLevel(level)
            area = np.trapezoid(np.clip(pdist._y_hi - y_lo, 0.0, None), x_grid)
            terms.append(("2d_ymin", cols, (jnp.asarray(x_grid), jnp.asarray(y_lo),
                                            pdist._y_hi, -np.log(area))))
        else:
            raise NotImplementedError(
                f"no log density implemented for the prior {kind.__name__} on {dnames}")

    def log_prior(theta):
        theta = jnp.asarray(theta, jnp.float64)
        total = jnp.zeros(theta.shape[:-1])
        for kind, cols, pars in terms:
            vals = theta[..., cols]
            if kind == "uniform":
                lo, hi, lnorm = pars
                inside = (vals[..., 0] >= lo) & (vals[..., 0] <= hi)
                total = total + jnp.where(inside, lnorm, -jnp.inf)
            elif kind == "normal":
                total = total + jstats.norm.logpdf(vals[..., 0], *pars)
            elif kind == "mvnormal":
                total = total + jstats.multivariate_normal.logpdf(vals, *pars)
            else:
                x_grid, y_lo, y_hi, lnorm = pars
                xx, yy = vals[..., 0], vals[..., 1]
                inside = ((xx >= x_grid[0]) & (xx <= x_grid[-1])
                          & (yy >= jnp.interp(xx, x_grid, y_lo)) & (yy <= y_hi))
                total = total + jnp.where(inside, lnorm, -jnp.inf)
        return total

    return log_prior
