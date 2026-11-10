import warnings

import numpy as np
import jax.numpy as jnp

from ATLAS import psd_functions
from ATLAS.signals.correlated.base import Correlated


class AstroGWB(Correlated):
    '''

    Hellings-Downs free-spectrum GWB whose amplitudes carry a hierarchical
    astrophysical prior, as in Pandora. A trained conditional normalizing flow
    p(log10_rho | theta) is the prior on the free-spectrum amplitudes, and the
    astrophysical parameters theta are sampled alongside the rest of the model
    (``model_maker`` adds the ``'astro_params'`` site and the
    ``'astro_hyperprior'`` factor). The joint prior on (log10_rho, theta) is a
    uniform box times the flow density, so the flow is truncated to the box.

    The sampled amplitudes are ``halflog10_rho = 0.5*log10(<a^T a>)``, which is
    the flow's log10_rho. The frequency bins are those the flow was trained at:
    ``data.pta_tspan`` must equal the flow's training Tspan.

    Arguments
    ------------------------------
    data : ATLAS.data.PTA_Data
    	The PTA data; supplies the pulsar positions and ``pta_tspan``.
    flow : ATLAS.astro.flows.flax.FlaxAstroFlow or ATLAS.astro.flows.zuko.ZukoAstroFlow
    	A trained ``'rho|theta'`` flow. A zuko flow is converted to flax with
    	``FlaxAstroFlow.from_zuko``.
    name : str
    	Signal name used in the combination string. (Default: 'astro')
    lower_bound_psd : array_like or None
    	Lower prior bounds on the halflog10_rho of each bin; None uses the
    	flow's training range, mean - half_range. (Default: None)
    upper_bound_psd : array_like or None
    	Upper prior bounds on the halflog10_rho of each bin; None uses the
    	flow's training range, mean + half_range. (Default: None)
    astro_lower : array_like or None
    	Lower uniform prior bounds of the varied astrophysical parameters (a
    	full-length array is also accepted); None uses the flow's training
    	range. (Default: None)
    astro_upper : array_like or None
    	Upper uniform prior bounds of the varied astrophysical parameters, as
    	for ``astro_lower``. (Default: None)
    astro_log_prior : callable or None
    	Extra log prior ``f(theta_varied) -> scalar`` on the varied
    	astrophysical parameters, e.g. Gaussians; it must be traceable by JAX.
    	(Default: None)
    fixed_astro : dict or None
    	``{index: value}`` of astrophysical parameters held fixed, indexed in
    	the flow's context order. (Default: None)
    freq_rtol : float
    	Relative tolerance when matching the flow's training frequencies to
    	the model's. (Default: 1e-10)

    Returns
    ------------------------------
    None

    '''

    # Prefix of this signal's parameter names, e.g. astro_halflog10_rho_0
    param_prefix = 'astro'

    def __init__(self,
                 data,
                 flow,
                 name='astro',
                 lower_bound_psd=None,
                 upper_bound_psd=None,
                 astro_lower=None,
                 astro_upper=None,
                 astro_log_prior=None,
                 fixed_astro=None,
                 freq_rtol=1e-10):
        # Flax (and zuko, for conversion) are [astro] extras: import them here
        # so the core never needs them.
        from ATLAS.astro.flows.flax import FlaxAstroFlow

        if not isinstance(flow, FlaxAstroFlow):
            flow = FlaxAstroFlow.from_zuko(flow)
        if flow.nf_type != 'rho|theta':
            raise ValueError(f"AstroGWB needs a 'rho|theta' flow, got {flow.nf_type!r}")
        self.flow = flow
        nfreqs = flow.n_freqs

        # Training ranges of log10_rho and theta, in physical units
        train_lo_rho = np.asarray(flow.mean_gwb - flow.half_range_gwb)
        train_hi_rho = np.asarray(flow.mean_gwb + flow.half_range_gwb)
        train_lo_ast = np.asarray(flow.mean_ast - flow.half_range_ast)
        train_hi_ast = np.asarray(flow.mean_ast + flow.half_range_ast)

        # ---- free-spectrum amplitudes: box defaults to the training range ----
        lower_rho = train_lo_rho if lower_bound_psd is None else np.asarray(lower_bound_psd, dtype=np.float64)
        upper_rho = train_hi_rho if upper_bound_psd is None else np.asarray(upper_bound_psd, dtype=np.float64)
        if lower_rho.shape != (nfreqs,) or upper_rho.shape != (nfreqs,):
            raise ValueError(
                f"the flow models {nfreqs} frequency bins, so the halflog10_rho bounds need "
                f"shape ({nfreqs},); got {lower_rho.shape} and {upper_rho.shape}")
        if np.any(lower_rho > train_lo_rho) or np.any(upper_rho < train_hi_rho):
            warnings.warn(
                "the halflog10_rho prior bounds clip the flow's training range; the "
                "astrophysical prior is truncated to the narrower box", stacklevel=2)

        super().__init__(
            data=data,
            name=name,
            psd_function=psd_functions.gwb_free_spectrum,
            orf_function=psd_functions.hd_orf,
            upper_bound_psd=jnp.asarray(upper_rho),
            lower_bound_psd=jnp.asarray(lower_rho),
            orf='hd',
            nfreqs=nfreqs,
        )
        # The base class assumes one shared range for every bin; record the real box.
        self.parameter_range = jnp.stack([jnp.asarray(lower_rho), jnp.asarray(upper_rho)], axis=1)
        self.log_prior_volume = jnp.sum(jnp.log(self.parameter_range[:, 1] - self.parameter_range[:, 0]))

        # ---- the flow must be evaluated at the bins it was trained at ----
        f_conv = flow.metadata.get('f_conv')
        if f_conv is None:
            warnings.warn(
                "the flow's metadata has no 'f_conv', so its training frequencies cannot be "
                "checked against the model's", stacklevel=2)
        elif not np.allclose(np.asarray(f_conv), np.asarray(self.freqs), rtol=freq_rtol, atol=0):
            raise ValueError(
                f"the flow was trained at frequencies {np.asarray(f_conv)} (Tspan "
                f"{flow.metadata.get('tspan')}) but the model uses {np.asarray(self.freqs)} "
                f"(Tspan {self.tspan}); set data.pta_tspan to the flow's Tspan before building "
                "the model, since every Fourier basis uses it")

        # ---- astrophysical parameters: varied subset and its prior box ----
        self.n_astro = int(flow.n_params)
        fixed_astro = dict(fixed_astro or {})
        container = np.zeros(self.n_astro)
        for i, v in fixed_astro.items():
            container[int(i)] = v
        self.astro_container = jnp.asarray(container)
        self.varied = np.array([i for i in range(self.n_astro) if i not in fixed_astro], dtype=int)
        self.n_varied = len(self.varied)

        lower_ast = train_lo_ast if astro_lower is None else np.asarray(astro_lower, dtype=np.float64)
        upper_ast = train_hi_ast if astro_upper is None else np.asarray(astro_upper, dtype=np.float64)
        # Full-length bounds (defaults included) are reduced to the varied parameters
        if lower_ast.shape == (self.n_astro,) and self.n_varied != self.n_astro:
            lower_ast = lower_ast[self.varied]
        if upper_ast.shape == (self.n_astro,) and self.n_varied != self.n_astro:
            upper_ast = upper_ast[self.varied]
        if lower_ast.shape != (self.n_varied,) or upper_ast.shape != (self.n_varied,):
            raise ValueError(
                f"astrophysical bounds need length {self.n_varied} (the varied parameters) "
                f"or {self.n_astro}; got {lower_ast.shape} and {upper_ast.shape}")
        if np.any(lower_ast < train_lo_ast[self.varied]) or np.any(upper_ast > train_hi_ast[self.varied]):
            warnings.warn(
                "the astrophysical prior bounds extend beyond the flow's training range; the "
                "flow is extrapolating there", stacklevel=2)
        self.astro_lower = jnp.asarray(lower_ast)
        self.astro_upper = jnp.asarray(upper_ast)

        names = flow.metadata.get('param_names') or [f"theta_{i}" for i in range(self.n_astro)]
        self.astro_param_names = [names[i] for i in self.varied]
        self.astro_log_prior = astro_log_prior

    def log_prior(self, log10_rho, theta_varied):
        '''

        Hierarchical log prior of the free-spectrum amplitudes given the
        astrophysical parameters: the flow's physical-units density (Jacobian
        included) plus the optional extra prior on theta. Pure JAX, so it can
        be jitted and differentiated inside the NumPyro model.

        Arguments
        ------------------------------
        log10_rho : jax.Array
        	Free-spectrum amplitudes halflog10_rho, shape (nfreqs,).
        theta_varied : jax.Array
        	Varied astrophysical parameters, shape (n_varied,).

        Returns
        ------------------------------
        log_prior : jax.Array
        	Scalar log p_flow(log10_rho | theta) + astro_log_prior(theta_varied).

        '''
        # The flow's context is the full theta: fixed values with the varied ones filled in
        theta = self.astro_container.at[self.varied].set(theta_varied)
        log_prior = self.flow.log_prob(log10_rho, theta)
        if self.astro_log_prior is not None:
            log_prior = log_prior + self.astro_log_prior(theta_varied)
        return log_prior
