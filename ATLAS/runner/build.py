"""Resolved run config -> PTA_Data, white noise, red noise, timing model."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial

import numpy as np
import jax.numpy as jnp

from ATLAS.runner import data_io
from ATLAS.runner.config import ConfigError, load_priors

log = logging.getLogger("ATLAS.runner")

__all__ = ["Built", "build_model"]


@dataclass
class Built:
    data: object
    wn: object
    rn: object
    tm_model: object
    wn_vec: object            # fixed white-noise vector, or None
    wn_lower: object
    wn_upper: object
    present: tuple
    psr_names: list
    residual_offset: object = None   # subtracted from the residuals (GTM/Adaptus mean)

    @property
    def residuals(self):
        """The residuals the likelihood sees: raw minus ``residual_offset``."""
        r = jnp.concat(self.data.raw_residuals)
        return r if self.residual_offset is None else r - self.residual_offset


def _bins(fb, key, present, sig):
    return fb[key] if sig in present else None


def _wn_overrides(run, wn):
    """Optional white-noise bounds from the priors file (``white_noise:``)."""
    entry = load_priors(run).get("white_noise")
    if not entry:
        return
    attrs = {"efac": ("lower_efac", "upper_efac"),
             "log10_equad": ("lower_log10equad", "upper_log10equad"),
             "log10_ecorr": ("lower_log10ecorr", "upper_log10ecorr")}
    unknown = set(entry) - set(attrs)
    if unknown:
        raise ConfigError(f"priors white_noise: unknown keys {sorted(unknown)}; "
                          f"expected {sorted(attrs)}")
    for key, (lo_attr, hi_attr) in attrs.items():
        if key in entry:
            lo, hi = entry[key]
            for cov in wn.cov_matrices:
                setattr(cov, lo_attr, float(lo)); setattr(cov, hi_attr, float(hi))


def _load_astro_flow(run):
    """The ``astro`` flow: a flow directory (Atlas or ng20), or a Pandora pickle + mapping."""
    from ATLAS.astro.flows import load_astro_flow
    from ATLAS.runner.config import resolve_path
    a = run["model"]["signals"]["astro"]
    if a["mapping"] is not None:
        return load_astro_flow(pickle=resolve_path(run, a["pickle"]),
                               mapping=resolve_path(run, a["mapping"]))
    # an ng20 pickle name is relative to the flow directory
    return load_astro_flow(resolve_path(run, a["flow"]), pickle=a["pickle"])


def _astro_kwargs(run, res, flow):
    """``make_red_noise`` keywords of the ``astro`` block.

    ``res`` is ``resolved['astro']`` from ``validate_run``.  Bounds not in the
    priors entry stay at the flow's training range.
    """
    from ATLAS.runner.config import load_function
    a = run["model"]["signals"]["astro"]
    names = list(flow.metadata.get("param_names") or
                 [f"theta_{i}" for i in range(flow.n_params)])
    unknown = sorted(set(res["theta"]) | set(res["fixed"]))
    unknown = [n for n in unknown if n not in names]
    if unknown:
        raise ConfigError(f"model.signals.astro: {unknown} are not flow parameters {names}")
    n_f = int(len(flow.mean_gwb))
    kw = dict(astro_flow=flow, fixed_astro=dict(res["fixed"]) or None)

    rho = res["rho"]
    if rho is not None:
        pairs = np.asarray([rho] * n_f if np.ndim(rho) == 1 else rho, dtype=np.float64)
        if pairs.shape != (n_f, 2):
            raise ConfigError(f"model.signals.astro: {len(pairs)} halflog10_rho bounds for the "
                              f"flow's {n_f} bins")
        kw["gwb_upper_bound_psd"] = pairs[:, 1]
        if flow.metadata.get("zero_handling") == "offset":
            # as in run_pandora: the box starts at the zero slab, so only upper edges apply
            log.info("offset-mode flow: the halflog10_rho lower edges are ignored (the box "
                     "starts at the zero slab)")
        else:
            kw["gwb_lower_bound_psd"] = pairs[:, 0]
    if res["theta"]:
        # full-length boxes at the training range, named overrides on top
        lo = np.asarray(flow.mean_ast - flow.half_range_ast, dtype=np.float64).copy()
        hi = np.asarray(flow.mean_ast + flow.half_range_ast, dtype=np.float64).copy()
        for n, (l, h) in res["theta"].items():
            lo[names.index(n)], hi[names.index(n)] = l, h
        kw["astro_lower_bound"], kw["astro_upper_bound"] = lo, hi

    if res["log_prior"] == "pspace":
        from ATLAS.astro.priors import pspace_log_prior
        pspace = a["pspace"] or flow.metadata.get("param_space")
        if pspace is None:
            raise ConfigError("model.signals.astro.log_prior: pspace needs model.signals.astro.pspace: "
                              "this flow does not record its parameter space")
        kw["astro_log_prior"] = pspace_log_prior(pspace, names)
    elif res["log_prior"] is not None:
        kw["astro_log_prior"] = load_function(run, res["log_prior"])
    return kw


def build_model(run, parsed, resolved):
    """Everything ``model_maker`` needs for one run.

    ``parsed`` / ``resolved`` come from ``config.validate_run``.
    """
    from ATLAS.data import PTA_Data
    from ATLAS.model_builder import ModelBuilder

    present = tuple(parsed["shared_names"] + parsed["separate_names"])
    d, fb, tm_cfg, sig = run["data"], run["frequency_bins"], run["timing"], run["model"]["signals"]

    psrs = data_io.load_pulsars(run)
    names = [str(p.name) for p in psrs]
    tm_params = None
    if tm_cfg["nonlinear"]["enabled"] or tm_cfg["design_matrix"]["columns"] in ("sampled", "not_sampled"):
        tm_params = data_io.timing_params_for(run, psrs)
        if tm_params is None and tm_cfg["nonlinear"]["enabled"]:
            # every fitted parameter, read BEFORE the design matrix is
            # restricted (make_timing_model would otherwise read the
            # restricted labels)
            tm_params = [[("XDOT" if l.upper() == "A1DOT" else l)
                          for l in data_io._design(p)[1] if data_io._canon(l) != "OFFSET"]
                         for p in psrs]
    # before PTA_Data / WhiteCov: both take their own SVD of psr.Mmat
    data_io.restrict_design_matrix(run, psrs, sampled=tm_params)
    noise_dict = data_io.load_noise_dict(run)

    adaptus_basis = gtm_psd = gtm_means = None
    if "gtm" in present:
        adaptus_basis, gtm_psd, gtm_means = data_io.load_gtm(run, psrs)

    parfiles = timfiles = None
    if tm_cfg["nonlinear"]["enabled"]:
        files = data_io.pulsar_files(run, names)
        parfiles = [files[n][0] for n in names]
        timfiles = [files[n][1] for n in names]

    data = PTA_Data(
        psrs,
        num_irn_bins=_bins(fb, "num_irn_bins", present, "unc"),
        num_gwb_bins=_bins(fb, "num_gwb_bins", present, "cor"),
        num_dm_bins=_bins(fb, "num_dm_bins", present, "dm"),
        num_det_bins=_bins(fb, "num_det_bins", present, "det"),
        adaptus_basis=adaptus_basis,
        adaptus_size=(adaptus_basis[0].shape[1] if adaptus_basis is not None else None),
        linear_timing=tm_cfg["linear_timing"],
        marg_timing=tm_cfg["marg_timing"],
        noise_dict=noise_dict,
        dm_ref_freq=float(d["dm_ref_freq"]),
        psr_dists_dict=data_io.load_distances(run),
        parfiles=parfiles, timfiles=timfiles,
    )
    astro_flow = None
    if "astro" in present:
        # the flow fixes the GWB frequencies: every Fourier basis uses its Tspan
        from ATLAS.signals.correlated.astro_gwb import set_flow_tspan
        astro_flow = _load_astro_flow(run)
        a = sig["astro"]
        set_flow_tspan(data, astro_flow, policy=a["tspan_policy"], rtol=float(a["tspan_rtol"]))
    mb = ModelBuilder(data=data)

    wn_cfg = run["white_noise"]
    ecorr = {"auto": None, True: True, False: False}[wn_cfg["include_ecorr"]]
    wn = mb.make_white_noise(stabilize_TNT=bool(wn_cfg["stabilize_TNT"]), include_ecorr=ecorr)
    _wn_overrides(run, wn)
    wn_lower, wn_upper = (jnp.asarray(x) for x in wn.get_prior_bounds())
    wn_vec = wn.params_dict_to_vector(noise_dict) if noise_dict is not None else None

    kw = dict(use_pulsar_tspan=bool(fb["use_pulsar_tspan"]))
    for which in ("irn", "gwb", "dm", "chrom"):
        uf = data_io.user_freqs(run, which)
        if uf is not None:
            kw[f"{which}_user_freqs"] = jnp.asarray(uf)

    def psd_of(s):
        r = resolved[s]
        fixed = sig[s]["fixed"] or {}
        fn = partial(r["psd"], **fixed) if fixed else r["psd"]
        return fn, jnp.asarray(r["lower"]), jnp.asarray(r["upper"])

    if "unc" in present:
        kw["irn_psd_function"], kw["irn_lower_bound_psd"], kw["irn_upper_bound_psd"] = psd_of("unc")
    if "cor" in present:
        kw["gwb_psd_function"], kw["gwb_lower_bound_psd"], kw["gwb_upper_bound_psd"] = psd_of("cor")
        kw["orf_function"] = resolved["cor"]["orf"]
        if resolved["cor"]["orf_names"]:
            kw["lower_bound_orf"] = jnp.asarray(resolved["cor"]["orf_lower"])
            kw["upper_bound_orf"] = jnp.asarray(resolved["cor"]["orf_upper"])
    if "astro" in present:
        kw.update(_astro_kwargs(run, resolved["astro"], astro_flow))
    if "dm" in present:
        kw["dm_psd_function"], kw["dm_lower_bound_psd"], kw["dm_upper_bound_psd"] = psd_of("dm")
    if "chrom" in present or "cquad" in present:
        idx = data_io.chrom_index_for(run, psrs)
        kw["chrom_index"] = idx
    if "chrom" in present:
        kw["chrom_psd_function"], kw["chrom_lower_bound_psd"], kw["chrom_upper_bound_psd"] = psd_of("chrom")
        kw["num_chrom_bins"] = int(fb["num_chrom_bins"])
    if "cquad" in present:
        kw["cquad_log10_amp"] = float(sig["cquad"]["log10_amp"])
        kw["cquad_project_timing"] = bool(sig["cquad"]["project_timing"])
    if "sw" in present:
        kw["sw_geometry"] = data_io.sw_geometry(run, psrs, compute=True)
        swk = data_io.sw_kwargs_for(run, psrs)
        if "fixed_log10_sigma" not in swk:
            swk["lower_bound_psd"] = tuple(resolved["sw"]["lower"])
            swk["upper_bound_psd"] = tuple(resolved["sw"]["upper"])
        kw["sw_kwargs"] = swk
    if "det" in present:
        from ATLAS.runner import registry
        kw["det_delay_function"] = registry.waveform(sig["det"]["waveform"])
        kw["det_parameter_bounds"] = np.stack([resolved["det"]["lower"],
                                               resolved["det"]["upper"]], axis=1)
    if "gtm" in present and sig["gtm"]["psd"] == "free_spectrum":
        kw["gtm_lower_bound_psd"] = resolved["gtm"]["lower"]
        kw["gtm_upper_bound_psd"] = resolved["gtm"]["upper"]
    elif "gtm" in present and gtm_psd is not None:
        kw["gt_psd_val"] = jnp.asarray(gtm_psd)

    rn = mb.make_red_noise(run["model"]["basis_string"], **kw)

    tm_model = None
    if tm_cfg["nonlinear"]["enabled"]:
        tm_model = mb.make_timing_model(
            explicit_timing_model_params_to_sample=tm_params)

    log.info("built %s: %d pulsars, %d T-matrix columns, %d red-noise parameters",
             run["run"]["name"], len(psrs), rn.nmodes, rn.model.upper_prior_lim_all.shape[0])
    offset = None
    if gtm_means is not None:
        if tm_model is not None:
            raise ConfigError("a gtm mean is subtracted from the data residuals, but with "
                              "nonlinear timing the residuals come from the timing model; "
                              "drop mean_pattern or the nonlinear timing")
        offset = jnp.asarray(np.concatenate(gtm_means))
    return Built(data, wn, rn, tm_model, wn_vec, wn_lower, wn_upper, present, names, offset)
