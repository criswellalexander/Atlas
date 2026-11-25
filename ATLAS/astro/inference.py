"""
Hierarchical astrophysical inference of the GWB with PTMCMC (ported from Pandora).

* :class:`HDAstroRedModel` is the red-noise model and the dense Hellings-Downs
  likelihood marginalized over all Fourier coefficients, built on Atlas's
  TNT/TNr and ``CorrelatedPulsarRedNoise``.
* :class:`AstroInferenceModel` is Pandora's ``AstroInferenceModel``: the joint
  posterior of red noise and astrophysical parameters, with a conditional flow
  (``ATLAS.astro.flows``) as the prior on the GWB spectrum, sampled with
  PTMCMCSampler.
* :func:`pandora_to_atlas_order` / :func:`atlas_to_pandora_order` swap each
  pulsar's intrinsic-noise pair between Pandora's and Atlas's orderings.

PTMCMCSampler and the flows are imported lazily.
"""

import os

import numpy as np

from ATLAS.astro.utils import gwb_bounds as _gwb_bounds

__all__ = [
    "HDAstroRedModel",
    "AstroInferenceModel",
    "pandora_to_atlas_order",
    "atlas_to_pandora_order",
]


class HDAstroRedModel:
    """
    Red noise for hierarchical astro inference: intrinsic power laws plus an HD GWB.

    The likelihood is marginalized over all Fourier coefficients with a dense
    cross-pulsar covariance, as in Pandora's ``AstroInferenceModel``::

        ln L = 1/2 (TNr^T Sigma^-1 TNr - ln|Sigma| - ln|phi|),  Sigma = TNT + phi^-1

    with the timing model marginalized and the white noise fixed inside TNT
    and TNr. phi, phi^-1 and the prior bounds come from Atlas's
    ``CorrelatedPulsarRedNoise``; build the model with :meth:`from_pta_data`
    (Atlas computes TNT/TNr) or :meth:`from_matrices` (precomputed ones).

    The parameter vector is Atlas's: ``[log10_A, gamma]`` per pulsar, then the
    GWB PSD parameters (``halflog10_rho`` per bin for a free spectrum), then
    any free ORF parameters. Pandora orders each pulsar's pair
    ``[gamma, log10_A]``; see :func:`pandora_to_atlas_order`.

    Attributes
    ----------
    lower, upper : ndarray
        Uniform prior bounds.
    param_names : list of str
    n_irn_params : int
        Length of the intrinsic red-noise block at the front of ``xs``.
    gwb_slice : slice
        Position of the GWB PSD parameters in ``xs``.
    gwb_is_free_spectrum : bool
    crn_bins, int_bins : int
    tspan : float
    f_common : ndarray
        GWB frequencies, ``i / tspan``.
    """

    def __init__(self, TNT, TNr, model, tspan, rNr=0.0, logdet_N=0.0):
        import jax
        import jax.numpy as jnp
        from ATLAS.psd_functions import gwb_free_spectrum

        self.model = model
        self.TNT = jnp.asarray(TNT)
        self.TNr = jnp.asarray(TNr)
        self.npsr = int(model.Npulsars)
        n = self.TNT.shape[0]
        if self.TNT.shape != (n, n) or self.TNr.shape != (n,) or n % self.npsr:
            raise ValueError(f"TNT {self.TNT.shape} / TNr {self.TNr.shape} do not match "
                             f"{self.npsr} pulsars")
        self.kmax = n // self.npsr
        self.int_bins = int(model.irn_bins)
        if self.kmax != 2 * self.int_bins:
            raise ValueError(f"TNT has {self.kmax} columns per pulsar but the red-noise "
                             f"model has {self.int_bins} bins (expected {2 * self.int_bins})")
        self.crn_bins = int(model.crn_bins)
        self.tspan = float(tspan)
        self.f_common = np.asarray(model.f_common)[:, 0]
        self.lower = np.asarray(model.lower_prior_lim_all, dtype=np.float64)
        self.upper = np.asarray(model.upper_prior_lim_all, dtype=np.float64)
        self.param_names = list(model.get_param_names())
        self.n_irn_params = int(model.irn_end_idx)
        self.gwb_slice = slice(int(model.gtm_end_idx), int(model.gwb_psd_end_idx))
        self.gwb_is_free_spectrum = model.gwb_psd_func is gwb_free_spectrum
        self.orf_fixed = bool(model.orf_fixed)
        self.rNr, self.logdet_N = float(rNr), float(logdet_N)
        self._lnl = jax.jit(self._lnlikelihood)

    @classmethod
    def from_pta_data(cls, data, white_noise_params, irn_log10_A_bounds=(-18.0, -11.0),
                      irn_gamma_bounds=(1.0, 7.0), rho_bounds=(-12.0, -4.0),
                      gwb_psd_function=None, orf_function=None, gwb_bounds=None,
                      include_ecorr=None, tspan=None):
        """
        Build TNT/TNr and the red-noise model with Atlas.

        Parameters
        ----------
        data : ATLAS.data.PTA_Data
            Built with ``marg_timing=True``, ``linear_timing=False``,
            ``num_gwb_bins`` (``crn_bins``) and ``num_irn_bins >= num_gwb_bins``.
        white_noise_params : dict or array_like
            Fixed white noise: a noise dictionary, or Atlas's flat vector.
        irn_log10_A_bounds, irn_gamma_bounds : (float, float)
            Intrinsic red-noise priors (Pandora's ``UniformPrior`` defaults).
        rho_bounds : (float, float)
            Prior on each ``halflog10_rho`` of a free-spectrum GWB.
        gwb_psd_function, orf_function : callable, optional
            Default: ``psd_functions.gwb_free_spectrum`` and ``hd_orf``.
        gwb_bounds : (lower, upper), optional
            GWB PSD bounds; required unless the PSD is a free spectrum.
        include_ecorr : bool or None
            Passed to ``WhiteCov`` (``None`` auto-detects).
        tspan : float, optional
            Overrides ``data.pta_tspan`` before the Fourier bases are built,
            e.g. to match the span a flow was trained on.
        """
        import jax.numpy as jnp
        from ATLAS.model_builder import ModelBuilder
        from ATLAS.nMatrix.base import WhiteCov
        from ATLAS.psd_functions import gwb_free_spectrum, hd_orf, powerlaw
        from ATLAS.signals.signals_utils import blockPsrs2sigma, blockVec2sigmaVec

        if not data.marg or data.linear_timing:
            raise ValueError("HDAstroRedModel needs PTA_Data(marg_timing=True, linear_timing=False)")
        gwb_psd_function = gwb_free_spectrum if gwb_psd_function is None else gwb_psd_function
        orf_function = hd_orf if orf_function is None else orf_function
        gwb_lo, gwb_hi = _gwb_bounds(gwb_psd_function, gwb_bounds, rho_bounds,
                                     data.num_gwb_bins)
        if tspan is not None:
            data.pta_tspan = float(tspan)

        wn = WhiteCov(data=data, stabilize_TNT=False, include_ecorr=include_ecorr)
        if isinstance(white_noise_params, dict):
            white_noise_params = wn.params_dict_to_vector(white_noise_params)
        rn = ModelBuilder(data=data).make_red_noise(
            "unc+cor->unc", use_pulsar_tspan=False,
            irn_psd_function=powerlaw, gwb_psd_function=gwb_psd_function,
            orf_function=orf_function,
            irn_lower_bound_psd=jnp.array([irn_log10_A_bounds[0], irn_gamma_bounds[0]]),
            irn_upper_bound_psd=jnp.array([irn_log10_A_bounds[1], irn_gamma_bounds[1]]),
            gwb_lower_bound_psd=gwb_lo, gwb_upper_bound_psd=gwb_hi,
        )
        TNT, TNr, rNr, logdet_N = rn.get_helpers(reff=jnp.concat(data.raw_residuals),
                                                 white_noise_params=jnp.asarray(white_noise_params))
        return cls(blockPsrs2sigma(TNT), blockVec2sigmaVec(TNr), rn.model, data.pta_tspan,
                   rNr=rNr, logdet_N=logdet_N)

    @classmethod
    def from_matrices(cls, TNT, TNr, psr_pos, tspan, crn_bins, int_bins=None,
                      irn_log10_A_bounds=(-18.0, -11.0), irn_gamma_bounds=(1.0, 7.0),
                      rho_bounds=(-12.0, -4.0), gwb_psd_function=None, orf_function=None,
                      gwb_bounds=None, stabilize_delta=None, psr_names=None):
        """
        Use precomputed TNT/TNr (e.g. Pandora's ``data/15yr_TNT_and_TNr_*.npz``).

        Parameters
        ----------
        TNT : (Np * 2 * int_bins, Np * 2 * int_bins) array_like
            Dense, pulsar-major, columns ``[sin f1, cos f1, sin f2, ...]``.
        TNr : (Np * 2 * int_bins,) array_like
        psr_pos : (Np, 3) array_like
            Unit vectors to the pulsars.
        psr_names : list of str, optional
            Used in parameter names (default ``psr0``, ``psr1``, ...).
        tspan : float
            Span defining the Fourier frequencies ``i / tspan``.
        crn_bins, int_bins : int
            GWB and intrinsic-noise bins (``int_bins`` defaults to ``crn_bins``).
        stabilize_delta : float, optional
            Pandora's ``matrix_stabilization``: add ``delta`` to the diagonal
            of TNT's correlation matrix and rescale by ``1 / (1 + delta)``.
        Other parameters
            As in :meth:`from_pta_data`.
        """
        import jax.numpy as jnp
        from ATLAS.parameterized import CorrelatedPulsarRedNoise
        from ATLAS.psd_functions import gwb_free_spectrum, hd_orf, powerlaw
        from ATLAS.signals.correlated.utils import make_gwb_model
        from ATLAS.signals.factorized.utils import make_irn_model
        from ATLAS.signals.signals_utils import get_harmonic_frequencies

        int_bins = crn_bins if int_bins is None else int_bins
        gwb_psd_function = gwb_free_spectrum if gwb_psd_function is None else gwb_psd_function
        orf_function = hd_orf if orf_function is None else orf_function
        gwb_lo, gwb_hi = _gwb_bounds(gwb_psd_function, gwb_bounds, rho_bounds, crn_bins)
        irn_func, irn_help = make_irn_model(
            powerlaw, jnp.array([irn_log10_A_bounds[0], irn_gamma_bounds[0]]),
            jnp.array([irn_log10_A_bounds[1], irn_gamma_bounds[1]]))
        gwb_func, orf_func, gwb_help = make_gwb_model(gwb_psd_function, orf_function,
                                                      gwb_lo, gwb_hi)
        psr_pos = np.asarray(psr_pos, dtype=np.float64)
        if psr_names is None:
            psr_names = [f"psr{i}" for i in range(len(psr_pos))]
        model = CorrelatedPulsarRedNoise(
            psr_pos=psr_pos, Npulsars=len(psr_pos),
            signal_indices={"unc": slice(0, 2 * int_bins), "cor": slice(0, 2 * crn_bins)},
            linear_timing_model_size=0,
            gwb_psd_func=gwb_func, orf_func=orf_func, gwb_helper_dictionary=gwb_help,
            crn_bins=crn_bins, f_common=get_harmonic_frequencies(crn_bins, tspan),
            irn_psd_func=irn_func, irn_helper_dictionary=irn_help, irn_bins=int_bins,
            f_irn=get_harmonic_frequencies(int_bins, tspan), pulsar_names=psr_names,
        )
        TNT = jnp.asarray(TNT)
        if stabilize_delta:
            D = jnp.outer(jnp.sqrt(TNT.diagonal()), jnp.sqrt(TNT.diagonal()))
            corr = TNT / D + stabilize_delta * jnp.eye(TNT.shape[0])
            TNT = D * corr / (1 + stabilize_delta)
        return cls(TNT, TNr, model, tspan)

    def _lnlikelihood(self, xs, TNT, TNr):
        import jax.numpy as jnp
        import jax.scipy as jsp
        from ATLAS.signals.signals_utils import blockModes2sigma

        phi = self.model.get_phi_mat(xs)
        psd_common = self.model.get_phi_diag(xs)[1]
        phiinv, logdet_phi = self.model.get_phi_mat_inv(phi)
        cf = jsp.linalg.cho_factor(TNT + blockModes2sigma(phiinv), lower=False)
        mu = jsp.linalg.cho_solve(cf, TNr)
        logdet_sigma = 2 * jnp.log(cf[0].diagonal()).sum()
        return 0.5 * (jnp.dot(TNr, mu) - logdet_sigma - logdet_phi), psd_common

    def lnlikelihood(self, xs, include_constant=False):
        """
        Marginal log likelihood and the common-process PSD.

        Parameters
        ----------
        xs : array_like
            Red-noise parameters (see the class docstring).
        include_constant : bool
            Add ``-1/2 (r^T N^-1 r + ln|N|)`` (available from
            :meth:`from_pta_data`). Pandora omits it.

        Returns
        -------
        lnl : jax scalar
        psd_common : (crn_bins, 1) jax array
            ``rho^2`` per GWB bin; ``0.5 * log10(psd_common)`` is ``log10_rho``.
        """
        import jax.numpy as jnp

        lnl, psd = self._lnl(jnp.asarray(xs, dtype=jnp.float64), self.TNT, self.TNr)
        if include_constant:
            lnl = lnl - 0.5 * (self.rNr + self.logdet_N)
        return lnl, psd


def pandora_to_atlas_order(xs, n_psr):
    """Swap each pulsar's ``[gamma, log10_A]`` (Pandora) to ``[log10_A, gamma]`` (Atlas)."""
    out = np.array(xs, dtype=np.float64, copy=True)
    pairs = out[..., :2 * n_psr].reshape(out.shape[:-1] + (n_psr, 2))
    out[..., :2 * n_psr] = pairs[..., ::-1].reshape(out.shape[:-1] + (2 * n_psr,))
    return out


def atlas_to_pandora_order(xs, n_psr):
    """Inverse of :func:`pandora_to_atlas_order` (the swap is its own inverse)."""
    return pandora_to_atlas_order(xs, n_psr)


class AstroInferenceModel:
    """
    Joint posterior of red noise and astrophysical parameters (Pandora's
    ``AstroInferenceModel``), sampled with PTMCMCSampler::

        p(xs, theta | data) ∝ L_HD(xs) p_flow(log10_rho(xs) | theta) p(theta)

    ``log10_rho = 0.5 * log10(psd_common)`` is the GWB spectrum implied by the
    red-noise parameters; the flow scores it given the astrophysical
    parameters ``theta``. The sampled vector is ``[red-noise params | varied
    theta]``.

    Parameters
    ----------
    red_model : HDAstroRedModel
    flow : ATLAS.astro.flows.zuko.ZukoAstroFlow
        A ``'rho|theta'`` flow trained at the same frequencies as ``red_model``
        (checked against the flow's ``f_conv``).
    astro_lower, astro_upper : array_like
        Uniform prior bounds of the varied astrophysical parameters (full
        length is also accepted when some are fixed).
    astro_log_prior : callable, optional
        Extra log prior on the varied parameters, e.g. Gaussians; Pandora's
        ``astro_additional_prior_func``.
    fixed_astro : dict, optional
        ``{index: value}`` of astrophysical parameters held fixed, indexed in
        the flow's context order.
    seed : int, optional
        Seeds the jump proposals' generator.

    Differences from Pandora
    ------------------------
    * The prior is the normalized uniform density (Pandora returns -8.01),
      and the flow density includes its Jacobian, so log posteriors are
      proper densities.
    * The flow-draw proposal conditions on the full ``theta`` (Pandora passed
      only the varied parameters, which breaks with fixed parameters), and it
      is used only when the GWB is a free spectrum (Pandora wrote ``rho`` into
      whatever the GWB parameters were).
    * Proposals use one seeded generator; ``sample(seed=...)`` reproduces a
      chain exactly. ``seed=0`` is honoured.
    * The flow's frequencies must match the red-noise model's.
    * Atlas does not rescale TNT/TNr (Pandora's ``renorm_const``), so the
      flow-draw proposal needs no unit offset; Pandora omitted it.
    """

    def __init__(self, red_model, flow, astro_lower, astro_upper, astro_log_prior=None,
                 fixed_astro=None, seed=None, freq_rtol=1e-10):
        if flow.nf_type != "rho|theta":
            raise ValueError(f"need a 'rho|theta' flow, got {flow.nf_type!r}")
        self.red, self.flow = red_model, flow
        self.n_astro = int(flow.n_params)
        fixed_astro = dict(fixed_astro or {})
        self.astro_container = np.zeros(self.n_astro)
        for i, v in fixed_astro.items():
            self.astro_container[int(i)] = v
        self.varied = np.array([i for i in range(self.n_astro) if i not in fixed_astro], dtype=int)
        self.n_varied = len(self.varied)

        astro_lower, astro_upper = (np.asarray(a, dtype=np.float64) for a in (astro_lower, astro_upper))
        if astro_lower.shape == (self.n_astro,) and self.n_varied != self.n_astro:
            astro_lower, astro_upper = astro_lower[self.varied], astro_upper[self.varied]
        if astro_lower.shape != (self.n_varied,) or astro_upper.shape != (self.n_varied,):
            raise ValueError(f"astro bounds must have length {self.n_varied} (varied parameters)")
        self.n_red = len(red_model.lower)
        self.lower = np.concatenate([red_model.lower, astro_lower])
        self.upper = np.concatenate([red_model.upper, astro_upper])
        self.ln_prior_value = -float(np.sum(np.log(self.upper - self.lower)))
        names = flow.metadata.get("param_names") or [f"astro_{i}" for i in range(self.n_astro)]
        self.param_names = red_model.param_names + [names[i] for i in self.varied]
        self.astro_log_prior = astro_log_prior if astro_log_prior is not None else (lambda x: 0.0)

        n_f = len(flow.mean_gwb)
        if n_f != red_model.crn_bins:
            raise ValueError(f"flow models {n_f} frequency bins but the red-noise model has "
                             f"{red_model.crn_bins} GWB bins")
        f_conv = flow.metadata.get("f_conv")
        if f_conv is not None and not np.allclose(f_conv, red_model.f_common, rtol=freq_rtol, atol=0):
            flow_tspan = flow.metadata.get("tspan")
            raise ValueError(
                f"flow was trained at frequencies {np.asarray(f_conv)} (Tspan {flow_tspan}) but the "
                f"red-noise model uses {red_model.f_common} (Tspan {red_model.tspan}); "
                "rebuild the red-noise model with a matching tspan")

        self.rng = np.random.default_rng(seed)

    # ---- posterior ------------------------------------------------------------

    def full_astro(self, varied):
        """The flow's context: fixed values with ``varied`` filled in."""
        theta = self.astro_container.copy()
        theta[self.varied] = varied
        return theta

    def lnlikelihood(self, xs):
        """``ln L_HD + ln p_flow(log10_rho | theta) + astro_log_prior(theta)`` (Pandora's ``get_lnliklihood``)."""
        xs = np.asarray(xs, dtype=np.float64)
        theta_v = xs[self.n_red:]
        lnl, psd = self.red.lnlikelihood(xs[:self.n_red])
        log10_rho = np.asarray(0.5 * np.log10(np.asarray(psd).T))
        try:
            lp = self.flow.log_prob(log10_rho, self.full_astro(theta_v)[None])
        except AssertionError:
            return -np.inf
        return float(np.asarray(lnl) + lp[0] + self.astro_log_prior(theta_v))

    def lnprior(self, xs):
        """Normalized uniform prior on the box (strict inequalities, as in Pandora)."""
        xs = np.asarray(xs)
        inside = np.all((xs > self.lower) & (xs < self.upper))
        return self.ln_prior_value if inside else -np.inf

    def make_initial_guess(self, seed=None):
        """A uniform draw from the prior box (``seed`` gives a fresh generator)."""
        rng = self.rng if seed is None else np.random.default_rng(seed)
        return rng.uniform(self.lower, self.upper)

    # ---- jump proposals (PTMCMCSampler signature) ------------------------------

    def _redraw(self, x, lo, hi):
        q = np.array(x, copy=True)
        i = self.rng.integers(lo, hi)
        q[i] = self.rng.uniform(self.lower[i], self.upper[i])
        return q, 0.0

    def draw_from_prior(self, x, iter, beta):
        """Redraw one parameter from its prior."""
        return self._redraw(x, 0, len(x))

    def draw_from_red_prior(self, x, iter, beta):
        """Redraw one intrinsic red-noise parameter."""
        return self._redraw(x, 0, self.red.n_irn_params)

    def draw_from_nonIR_prior(self, x, iter, beta):
        """Redraw one parameter outside the intrinsic red-noise block."""
        return self._redraw(x, self.red.n_irn_params, len(x))

    def draw_from_astro_prior(self, x, iter, beta):
        """Redraw one astrophysical parameter."""
        return self._redraw(x, self.n_red, len(x))

    def draw_from_gwb_prior(self, x, iter, beta):
        """Independence proposal: ``log10_rho ~ p_flow(. | theta)``."""
        return self._gwb_flow_draw(x, int(self.rng.integers(2**62)))

    def _gwb_flow_draw(self, x, torch_seed):
        q = np.array(x, copy=True)
        theta = self.full_astro(q[self.n_red:])
        sl = self.red.gwb_slice
        current = q[sl][None]
        new = self.flow.sample(1, theta, seed=torch_seed)
        lqxy = (self.flow.log_prob(current, theta[None], jacobian=False)
                - self.flow.log_prob(new, theta[None], jacobian=False))
        q[sl] = new[0]
        return q, float(lqxy[0])

    # ---- sampling --------------------------------------------------------------

    def _write_pars(self, outdir, names):
        with open(os.path.join(outdir, "pars.txt"), "w") as fh:
            fh.write("\n".join(names) + "\n")

    def sample(self, niter, outdir, x0=None, seed=None, resume=True, load_cov=False,
               include_groups=True, include_IRN_groups=False, **sample_kwargs):
        """
        Run PTMCMCSampler (Pandora's ``sample``).

        Parameters
        ----------
        niter : int
        outdir : str
            Chains (``chain_1.txt``), ``pars.txt`` and, with ``load_cov``,
            the starting ``cov.npy``.
        x0 : array_like, optional
            Start; default a prior draw.
        seed : int, optional
            Seeds PTMCMCSampler and the proposals, so the chain is reproducible.
        resume : bool
            Continue an existing chain in ``outdir`` (Pandora's default).
        load_cov : bool
            Start from ``outdir/cov.npy`` instead of ``0.01**2 * I``.
        include_groups, include_IRN_groups : bool
            Pandora's extra jump groups: the non-intrinsic block (and the ORF
            parameters if free), and the intrinsic block.
        sample_kwargs
            Passed to ``PTSampler.sample`` (Pandora's jump weights SCAM 30,
            AM 15, DE 50 are the defaults here).
        """
        from PTMCMCSampler.PTMCMCSampler import PTSampler

        os.makedirs(outdir, exist_ok=True)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        x0 = self.make_initial_guess() if x0 is None else np.asarray(x0, dtype=np.float64)
        ndim = len(x0)
        cov = (np.load(os.path.join(outdir, "cov.npy")) if load_cov
               else np.diag(np.ones(ndim) * 0.01**2))
        n_irn = self.red.n_irn_params
        groups = [list(range(ndim))]
        if include_IRN_groups and n_irn:
            groups += [list(range(n_irn))] * 2
        if include_groups:
            groups += [list(range(n_irn, ndim))] * 2
            if not self.red.orf_fixed:
                groups += [list(range(self.red.gwb_slice.stop, ndim))] * 2

        sampler = PTSampler(ndim, self.lnlikelihood, self.lnprior, cov, groups=groups,
                            outDir=outdir, resume=resume, seed=seed)
        sampler.addProposalToCycle(self.draw_from_prior, 10)
        if n_irn:
            sampler.addProposalToCycle(self.draw_from_red_prior, 10)
        sampler.addProposalToCycle(self.draw_from_nonIR_prior, 10)
        if self.n_varied:
            sampler.addProposalToCycle(self.draw_from_astro_prior, 10)
        if self.red.gwb_is_free_spectrum:
            sampler.addProposalToCycle(self.draw_from_gwb_prior, 10)
        self._write_pars(outdir, self.param_names)
        kwargs = dict(SCAMweight=30, AMweight=15, DEweight=50)
        kwargs.update(sample_kwargs)
        sampler.sample(x0, niter, **kwargs)
        return sampler
