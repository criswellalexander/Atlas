"""Run configuration: defaults, dotted-path overrides, sweeps and validation.

A config file holds one base setup and a list of runs.  Each run is the base
with a few values replaced through dotted paths::

    runs:
      - name: hd_chi_sampled
        overrides:
          model.signals.chrom.varied_index: true

``expand_runs`` turns the file into one fully resolved dict per run, and
``validate_run`` checks a resolved run against ATLAS before anything is
launched -- signal names, PSD/ORF names, prior coverage, bin counts, and the
setting combinations ATLAS cannot run.
"""
from __future__ import annotations

import copy
import glob
import importlib.util
import itertools
import json
from pathlib import Path

import yaml

from ATLAS.runner import registry

__all__ = ["DEFAULTS", "SIGNALS", "load_config", "expand_runs", "validate_run", "substitute_vars",
           "ConfigError", "set_dotted", "get_dotted", "load_priors", "prior_bounds",
           "load_function"]

SIGNALS = ("unc", "cor", "astro", "dm", "gtm", "chrom", "cquad", "sw", "det")
# relative order the separate blocks must keep (partial marginalisation lays
# the non-GWB phi out IRN | DM | GTM | extras, and det must come last)
_BLOCK_ORDER = ("dm", "gtm", "chrom", "cquad", "sw", "det")

DEFAULTS = {
    "config_version": 1,
    # free-form values substituted into any string as ${name} (after overrides)
    "vars": {},
    "ray": {
        "address": None,
        "num_cpus_per_run": 1,
        "num_gpus_per_run": 0,
        "max_concurrent_runs": None,
        "max_retries": 1,
        "env_vars": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "JAX_ENABLE_X64": "true"},
    },
    "output": {
        "root": None,
        "run_dir": "{root}/{run_name}",
        "format": "npz",
        "save": {"samples": True, "red_coefficients": False, "z_a": False,
                 "resolved_config": True, "environment": True},
        "skip_completed": True,
        "overwrite": False,
    },
    "data": {
        "source": "par_tim",
        "par_files": None,
        "tim_files": None,
        "pickle": None,
        "npz_dir": None,
        "npz_files": None,
        "pulsars": "all",
        "exclude": [],
        "timing_package": "pint",
        "ephem": "DE440",
        "noise_dict": None,
        "pulsar_distances": None,
        "dm_ref_freq": 1400.0,
        "cache_dir": None,
        "load_jobs": 4,
        "file_exclude": [],
    },
    "timing": {
        "linear_timing": True,
        "marg_timing": True,
        "nonlinear": {"enabled": False, "params_to_sample": None,
                      "parametrization": "klam"},
        "design_matrix": {"columns": "all", "keep_offset": True},
    },
    "white_noise": {"vary": False, "include_ecorr": "auto", "stabilize_TNT": False},
    "frequency_bins": {
        "num_irn_bins": 30, "num_gwb_bins": 14, "num_dm_bins": 30,
        "num_chrom_bins": 30, "num_det_bins": None, "use_pulsar_tspan": False,
        "irn_user_freqs": None, "gwb_user_freqs": None,
        "dm_user_freqs": None, "chrom_user_freqs": None,
    },
    "model": {
        "basis_string": "unc+cor->unc ; dm",
        "priors_file": None,
        "signals": {
            "unc": {"psd": "powerlaw", "fixed": {}, "priors": "irn_powerlaw"},
            "cor": {"psd": "powerlaw", "fixed": {}, "orf": "hd_orf", "priors": "gwb_powerlaw_free_gamma",
                    # GWB spectrum prior from a trained SASFlow (replaces the uniform box):
                    # {file: flow.npz, flow_num_layers: 4, hidden_size: 128,
                    #  mlp_num_layers: 2, num_bins: 8, B: 5.0, mass_samples: 200000}
                    "prior_flow": None},
            # HD free-spectrum GWB with a hierarchical astrophysical prior: a trained
            # p(log10_rho | theta) flow, theta sampled (ATLAS.signals.correlated.astro_gwb)
            "astro": {"flow": None, "pickle": None, "mapping": None, "priors": None,
                      "fixed": {}, "log_prior": None, "pspace": None,
                      "tspan_policy": "flow", "tspan_rtol": 1e-3, "n_prior_draws": 20000},
            "dm": {"psd": "powerlaw", "fixed": {}, "priors": "dm_powerlaw", "varied_index": False},
            "chrom": {"psd": "powerlaw", "fixed": {}, "priors": "chrom_powerlaw", "chrom_index": 4.0,
                      "varied_index": False, "index_bounds": [0.0, 7.0]},
            "cquad": {"log10_amp": 0.0, "project_timing": True},
            "sw": {"geometry": "pint", "settings_from_par": True,
                   "defaults": {"kernel": "RIDGE", "dt_days": 30.0, "interp_kind": "linear"},
                   "fixed_log10_sigma": None, "priors": "sw_ridge"},
            "gtm": {"basis_file": None, "basis_pattern": None, "mean_pattern": None,
                    "from_pulsar": False, "index": None, "size": None,
                    "psd_file": None, "psd_value": None,
                    "psd": "fixed", "priors": "gtm_free_spectrum"},
            "det": {"waveform": "cw_delay_evolve_float64", "priors": "cw"},
        },
    },
    "sampler": {
        "kind": "nuts",
        "model": "atlas",
        "seed_offset_var": None,
        "beta": None,
        "switch_overrides": None,
        "dense_timing_mass": False,
        "init_white_noise": None,
        "marg_over_non_gwb": False,
        "nuts": {"target_accept_prob": 0.8, "max_tree_depth": 10, "dense_mass": False},
        "num_warmup": 500,
        "num_samples": 1000,
        "num_chains": 1,
        "chain_method": "sequential",
        "thinning": 1,
        "seed": 0,
        "progress_bar": False,
    },
}

# dicts whose keys are free-form (not checked against DEFAULTS)
_FREE_FORM = ("vars", "sampler.switch_overrides", "timing.design_matrix.columns", "ray.env_vars", "model.signals.*.fixed", "model.signals.sw.defaults",
              "model.signals.cor.prior_flow",
              "timing.nonlinear.params_to_sample")


class ConfigError(ValueError):
    """A config that cannot run.  Raised before anything is launched."""


# --------------------------------------------------------------------------- #
#  dict helpers
# --------------------------------------------------------------------------- #

def _deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and v:
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _is_free_form(path):
    parts = path.split(".")
    for pat in _FREE_FORM:
        pp = pat.split(".")
        if len(parts) >= len(pp) and all(a == b or a == "*" for a, b in zip(pp, parts)):
            return True
    return False


def _check_keys(d, ref, prefix=""):
    """Unknown keys are almost always typos (``num_irn_bin``) -- reject them."""
    for k, v in d.items():
        path = f"{prefix}{k}"
        if _is_free_form(path):
            continue
        if k not in ref:
            raise ConfigError(f"unknown config key {path!r}; expected one of {sorted(ref)}")
        if isinstance(v, dict) and isinstance(ref[k], dict) and ref[k]:
            _check_keys(v, ref[k], path + ".")


def get_dotted(d, path):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def set_dotted(d, path, value):
    """Replace ``d[a][b][c]`` for ``path = "a.b.c"``.  The whole value is
    replaced (a dict is not merged), and the path must already exist."""
    parts = path.split(".")
    cur = d
    for i, part in enumerate(parts[:-1]):
        if not isinstance(cur, dict) or part not in cur:
            raise ConfigError(f"override {path!r}: {'.'.join(parts[:i + 1])!r} does not exist")
        cur = cur[part]
    if not isinstance(cur, dict):
        raise ConfigError(f"override {path!r}: parent is not a section")
    if parts[-1] not in cur and not _is_free_form(".".join(parts[:-1]) + ".x"):
        raise ConfigError(f"override {path!r}: unknown key {parts[-1]!r}; "
                          f"expected one of {sorted(cur)}")
    cur[parts[-1]] = copy.deepcopy(value)


# --------------------------------------------------------------------------- #
#  loading and expansion
# --------------------------------------------------------------------------- #

def load_config(path):
    """Read a run-config YAML file and merge it over ``DEFAULTS``."""
    path = Path(path).resolve()
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    runs = raw.pop("runs", None)
    sweeps = raw.pop("sweeps", None)
    _check_keys(raw, DEFAULTS)
    cfg = _deep_merge(DEFAULTS, raw)
    cfg["runs"] = runs if runs is not None else [{"name": "run"}]
    cfg["sweeps"] = sweeps or []
    cfg["_config_path"] = str(path)
    cfg["_config_dir"] = str(path.parent)
    return cfg


_VAR = __import__("re").compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def substitute_vars(obj, variables, _what="config"):
    """Replace ``${name}`` in every string of ``obj`` by ``variables[name]``.

    A string that is exactly ``${name}`` takes the value itself (so numbers and
    lists stay numbers and lists); otherwise the value is formatted into the
    string.  Values may refer to other variables.
    """
    def sub(v):
        if isinstance(v, dict):
            return {k: sub(x) for k, x in v.items()}
        if isinstance(v, list):
            return [sub(x) for x in v]
        if not isinstance(v, str) or "${" not in v:
            return v
        for _ in range(10):
            m = _VAR.fullmatch(v)
            if m:
                if m.group(1) not in variables:
                    raise ConfigError(f"{_what}: unknown variable ${{{m.group(1)}}}")
                v = copy.deepcopy(variables[m.group(1)])
                if not (isinstance(v, str) and "${" in v):
                    return sub(v) if isinstance(v, (dict, list)) else v
                continue
            missing = [n for n in _VAR.findall(v) if n not in variables]
            if missing:
                raise ConfigError(f"{_what}: unknown variable(s) {missing} in {v!r}; "
                                  f"defined: {sorted(variables)}")
            v = _VAR.sub(lambda mm: str(variables[mm.group(1)]), v)
            if "${" not in v:
                return v
        raise ConfigError(f"{_what}: variables refer to each other in a loop: {v!r}")
    return sub(obj)


def _grid_values(key, vals):
    """A grid entry: a list, or ``{range: [start, stop(, step)]}``."""
    if isinstance(vals, dict):
        if set(vals) != {"range"}:
            raise ConfigError(f"sweep grid {key!r}: a dict entry must be {{range: [start, stop]}}")
        return list(range(*[int(x) for x in vals["range"]]))
    if not isinstance(vals, list) or not vals:
        raise ConfigError(f"sweep grid {key!r}: give a non-empty list or {{range: [start, stop]}}")
    return vals


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return "-".join(_fmt(x) for x in v)
    return str(v).replace("/", "_").replace(" ", "")


def expand_runs(cfg):
    """One fully resolved dict per run: base + overrides (+ sweep grids).

    Each resolved run carries ``run.name``, ``run.index``, ``run.output_dir``
    and a seed (``sampler.seed + index`` unless the run sets its own).
    """
    base = {k: v for k, v in cfg.items() if k not in ("runs", "sweeps")}
    specs = []
    for r in cfg["runs"]:
        if "name" not in r:
            raise ConfigError(f"every run needs a name: {r}")
        if "sweep" in r:
            raise ConfigError(f"run {r['name']!r}: put sweeps under the top-level 'sweeps:'")
        extra = set(r) - {"name", "overrides"}
        if extra:
            raise ConfigError(f"run {r['name']!r}: unknown keys {sorted(extra)} "
                              "(use 'overrides' with dotted paths)")
        specs.append((r["name"], dict(r.get("overrides") or {})))
    for sw in cfg["sweeps"]:
        name, grid = sw.get("name"), sw.get("grid") or {}
        if not name or not grid:
            raise ConfigError(f"a sweep needs 'name' and 'grid': {sw}")
        keys = list(grid)
        for combo in itertools.product(*[_grid_values(k, grid[k]) for k in keys]):
            ov = dict(sw.get("base_overrides") or {})
            ov.update(zip(keys, combo))
            tag = "__".join(f"{k.split('.')[-1]}={_fmt(v)}" for k, v in zip(keys, combo))
            specs.append((f"{name}__{tag}", ov))

    names = [n for n, _ in specs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ConfigError(f"duplicate run names: {dup}")

    runs = []
    for i, (name, overrides) in enumerate(specs):
        run = copy.deepcopy(base)
        for path, value in overrides.items():
            set_dotted(run, path, value)
        variables = dict(run["vars"] or {}, run_name=name, run_index=i)
        run["vars"] = substitute_vars(run["vars"] or {}, variables, f"run {name!r} vars")
        variables.update(run["vars"])
        run = {k: (v if k in ("vars", "_config_path", "_config_dir")
                   else substitute_vars(v, variables, f"run {name!r}"))
               for k, v in run.items()}
        if "sampler.seed" not in overrides:
            off_var = run["sampler"]["seed_offset_var"]
            if off_var is not None:
                if off_var not in variables:
                    raise ConfigError(f"run {name!r}: sampler.seed_offset_var {off_var!r} "
                                      "is not a variable")
                run["sampler"]["seed"] = int(run["sampler"]["seed"]) + int(variables[off_var])
            else:
                run["sampler"]["seed"] = int(run["sampler"]["seed"]) + i
        root = run["output"]["root"]
        if root is None:
            raise ConfigError("output.root is required")
        root = str(Path(cfg["_config_dir"], root)) if not Path(root).is_absolute() else root
        run["run"] = {"name": name, "index": i, "overrides": overrides,
                      "output_dir": run["output"]["run_dir"].format(root=root, run_name=name)}
        runs.append(run)

    dirs = [r["run"]["output_dir"] for r in runs]
    if len(set(dirs)) != len(dirs):
        raise ConfigError("two runs write to the same output directory; "
                          "check output.run_dir uses {run_name}")
    return runs


# --------------------------------------------------------------------------- #
#  priors
# --------------------------------------------------------------------------- #

def _resolve(cfg, p):
    if p is None:
        return None
    p = Path(p)
    return str(p if p.is_absolute() else Path(cfg["_config_dir"], p))


def load_priors(run):
    path = _resolve(run, run["model"]["priors_file"])
    if path is None:
        raise ConfigError("model.priors_file is required")
    if not Path(path).exists():
        raise ConfigError(f"priors file not found: {path}")
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def prior_bounds(entry, names, what):
    """``{param: [lo, hi]}`` -> (lower, upper) lists in ``names`` order.

    A free spectrum may give one pair under ``halflog10_rho`` (broadcast to
    every bin) or a list with one pair per bin.
    """
    entry = {k: v for k, v in (entry or {}).items() if k != "orf"}
    fs = registry.FREE_SPECTRUM_SENTINEL
    if names and all(n.startswith(fs + "_") for n in names) and fs in entry:
        val = entry.pop(fs)
        pairs = [val] * len(names) if _is_pair(val) else list(val)
        if len(pairs) != len(names):
            raise ConfigError(f"{what}: {len(pairs)} bounds for {len(names)} free-spectrum bins")
        entry.update(zip(names, pairs))
    missing = [n for n in names if n not in entry]
    extra = [k for k in entry if k not in names]
    if missing or extra:
        raise ConfigError(f"{what}: priors must cover exactly the sampled parameters "
                          f"{names}; missing {missing}, unexpected {extra}")
    lo, hi = [], []
    for n in names:
        pair = entry[n]
        if not _is_pair(pair) or not float(pair[0]) < float(pair[1]):
            raise ConfigError(f"{what}: bound for {n!r} must be [low, high] with low < high, got {pair}")
        lo.append(float(pair[0])); hi.append(float(pair[1]))
    return lo, hi


def _is_pair(v):
    return isinstance(v, (list, tuple)) and len(v) == 2 and \
        all(isinstance(x, (int, float)) for x in v)


# --------------------------------------------------------------------------- #
#  validation
# --------------------------------------------------------------------------- #

def _bins_key(sig):
    return {"unc": "num_irn_bins", "cor": "num_gwb_bins", "dm": "num_dm_bins",
            "chrom": "num_chrom_bins", "det": "num_det_bins"}.get(sig)


def _files(spec, cfg):
    if spec is None:
        return []
    items = [spec] if isinstance(spec, str) else list(spec)
    out = []
    for it in items:
        it = _resolve(cfg, it)
        hits = sorted(glob.glob(it))
        out += hits if hits else [it]
    return out


def validate_run(run, check_files=True):
    """Raise ``ConfigError`` (listing every problem) if ``run`` cannot run.

    Returns the parsed basis string and the per-signal priors resolved to
    bound arrays, which the builder reuses.
    """
    from ATLAS.signals.signals_utils import parse_basis_string

    errs = []
    name = run["run"]["name"]
    m, fb, tm, smp = run["model"], run["frequency_bins"], run["timing"], run["sampler"]

    try:
        parsed = parse_basis_string(m["basis_string"])
    except Exception as exc:
        raise ConfigError(f"[{name}] bad basis_string {m['basis_string']!r}: {exc}")
    present = parsed["shared_names"] + parsed["separate_names"]
    unknown = [s for s in present if s not in SIGNALS]
    if unknown:
        errs.append(f"unknown signals {unknown} in basis_string; known: {list(SIGNALS)}")

    # -- layout ---------------------------------------------------------------
    for s in ("chrom", "cquad", "sw", "det", "gtm", "dm"):
        if s in parsed["shared_names"] and s != "dm":
            errs.append(f"{s!r} must be a separate block (after ';'), not shared")
    order = [s for s in parsed["separate_names"] if s in _BLOCK_ORDER]
    if order != sorted(order, key=_BLOCK_ORDER.index):
        errs.append(f"separate blocks must keep the order {list(_BLOCK_ORDER)}; got {order}")
    if tm["linear_timing"] and not tm["marg_timing"] and not parsed["include_timing"]:
        errs.append("linear_timing=true with marg_timing=false needs 'ltm|' at the "
                    "start of basis_string (or set marg_timing: true)")
    if parsed["include_timing"] and (tm["marg_timing"] or not tm["linear_timing"]):
        errs.append("'ltm|' needs timing.linear_timing: true and timing.marg_timing: false")

    # -- bins -------------------------------------------------------------------
    for s in present:
        key = _bins_key(s)
        if key and not (isinstance(fb.get(key), int) and fb[key] > 0):
            errs.append(f"signal {s!r} needs frequency_bins.{key} > 0 (got {fb.get(key)})")
    if parsed["representative"] == "unc" and "cor" in parsed["shared_names"] \
            and isinstance(fb["num_gwb_bins"], int) and fb["num_gwb_bins"] > fb["num_irn_bins"]:
        errs.append("with 'unc+cor->unc' the GWB bins sit inside the IRN basis: "
                    "num_gwb_bins must be <= num_irn_bins")

    # -- sampler and flags ----------------------------------------------------------
    if smp["kind"] != "nuts":
        errs.append(f"sampler.kind {smp['kind']!r} is not supported yet (only 'nuts')")
    if smp["chain_method"] not in ("sequential", "vectorized", "parallel"):
        errs.append(f"sampler.chain_method must be sequential|vectorized|parallel")
    if smp["marg_over_non_gwb"] and "cor" not in present and "astro" not in present:
        errs.append("sampler.marg_over_non_gwb needs a 'cor' or 'astro' signal")
    sig = m["signals"]
    if sig["dm"]["varied_index"]:
        if "dm" not in present:
            errs.append("model.signals.dm.varied_index needs 'dm' in basis_string")
        if not (run["white_noise"]["vary"] or tm["nonlinear"]["enabled"]):
            errs.append("model.signals.dm.varied_index needs white_noise.vary or "
                        "timing.nonlinear.enabled (the helpers must be rebuilt per step)")
    custom = smp["model"] != "atlas"
    if smp["beta"] is not None:
        try:
            b = float(smp["beta"])
        except (TypeError, ValueError):
            b = None
        if b is None or not 0.0 <= b <= 1.0:
            errs.append(f"sampler.beta must be a number in [0, 1], got {smp['beta']!r}")
        if not smp["marg_over_non_gwb"]:
            errs.append("sampler.beta (thermodynamic integration) needs "
                        "sampler.marg_over_non_gwb: true (it tempers partial_marg_lnposterior)")
        if "astro" in present:
            errs.append("sampler.beta: TI evidence is not supported for 'astro' models yet "
                        "(the box x flow prior on log10_rho is not normalized over the box)")
        elif "cor" not in present:
            errs.append("sampler.beta needs a 'cor' block in basis_string")
        if run["white_noise"]["stabilize_TNT"]:
            errs.append("sampler.beta needs white_noise.stabilize_TNT: false "
                        "(the stabilising shift biases ln Z)")
        if custom:
            errs.append("sampler.beta works with sampler.model: atlas only")
    iwn = smp["init_white_noise"]
    if iwn not in (None, "noise_dict"):
        errs.append("sampler.init_white_noise must be null or noise_dict")
    if iwn == "noise_dict" and not run["white_noise"]["vary"]:
        errs.append("sampler.init_white_noise only matters with white_noise.vary: true")
    if iwn == "noise_dict" and not run["data"]["noise_dict"]:
        errs.append("sampler.init_white_noise: noise_dict needs data.noise_dict")
    if smp["dense_timing_mass"] and not tm["nonlinear"]["enabled"]:
        errs.append("sampler.dense_timing_mass needs timing.nonlinear.enabled "
                    "(there are no sampled timing parameters otherwise)")
    so = smp["switch_overrides"]
    if so is not None:
        if smp["beta"] is None:
            errs.append("sampler.switch_overrides (model-switch TI) needs sampler.beta")
        if not isinstance(so, dict) or not so:
            errs.append("sampler.switch_overrides must be a non-empty {dotted.path: value} "
                        "map that turns this run into model A (beta = 0)")
        else:
            bad = [k for k in so if not k.startswith("model.signals.")]
            if bad:
                errs.append(f"sampler.switch_overrides may only change model.signals.* "
                            f"(same data, basis and white noise in both models); got {bad}")
    if custom and (not isinstance(smp["model"], str) or ":" not in smp["model"]):
        errs.append("sampler.model must be 'atlas' or 'path/to/file.py:function'")
    if sig["chrom"]["varied_index"] and not ({"chrom", "cquad"} & set(present)):
        errs.append("model.signals.chrom.varied_index needs 'chrom' or 'cquad' in basis_string")
    if tm["nonlinear"]["enabled"] and importlib.util.find_spec("jug") is None:
        errs.append("timing.nonlinear.enabled needs JUG installed")
    if tm["nonlinear"]["enabled"] and not run["data"]["par_files"]:
        errs.append("timing.nonlinear.enabled needs data.par_files / data.tim_files "
                    "(JUG reads them, whatever data.source is)")
    cols = tm["design_matrix"]["columns"]
    if not (cols in ("all", "sampled", "not_sampled") or isinstance(cols, (list, dict))):
        errs.append("timing.design_matrix.columns must be all | sampled | not_sampled | "
                    "[names] | {psr: [names]}")
    if cols != "all" and not (tm["marg_timing"] or tm["linear_timing"]):
        errs.append("timing.design_matrix.columns only matters with timing.marg_timing "
                    "or timing.linear_timing (otherwise the design matrix is not used)")
    if cols in ("sampled", "not_sampled") and not tm["nonlinear"]["enabled"] \
            and tm["nonlinear"]["params_to_sample"] is None:
        errs.append(f"timing.design_matrix.columns: {cols} takes its parameter list from "
                    "timing.nonlinear.params_to_sample, which is not set (with nonlinear "
                    "timing off there is no default list); add params_to_sample")
    if tm["nonlinear"]["enabled"] and tm["marg_timing"] and cols in ("all", "sampled"):
        errs.append("nonlinear timing with marg_timing: true would marginalise the sampled "
                    "parameters' columns as well (double counting); set "
                    "timing.design_matrix.columns: not_sampled")
    pts = tm["nonlinear"]["params_to_sample"]
    if not (pts is None or pts == "from_pulsar" or isinstance(pts, (list, dict))):
        errs.append("timing.nonlinear.params_to_sample must be null, 'from_pulsar', a list, "
                    "{psr: [...]}, {exclude: [patterns]} or {file: ..., key: ..., index: ...}")
    for k in ("irn_user_freqs", "gwb_user_freqs", "dm_user_freqs", "chrom_user_freqs"):
        v = fb[k]
        if not (v is None or isinstance(v, (list, str, dict))):
            errs.append(f"frequency_bins.{k} must be null, a list, 'file.npz:key' or "
                        "{file: ..., key: ...}")
    if run["output"]["format"] not in ("npz", "netcdf"):
        errs.append("output.format must be npz or netcdf")
    if run["output"]["format"] == "netcdf" and netcdf_engine() is None:
        errs.append("output.format netcdf needs xarray plus netCDF4, or h5netcdf with "
                    "h5py; or use npz")

    # -- PSDs, ORFs, priors -----------------------------------------------------------
    resolved = {}
    try:
        priors = load_priors(run)
    except ConfigError as exc:
        errs.append(str(exc)); priors = {}
    for s in ("unc", "cor", "dm", "chrom"):
        if s not in present:
            continue
        spec = sig[s]
        try:
            fn = registry.psd(spec["psd"], signal=s)
            names = registry.psd_param_names(fn, fixed=tuple(spec["fixed"] or {}),
                                             n_bins=fb[_bins_key(s)])
            if spec["priors"] not in priors:
                raise ConfigError(f"model.signals.{s}.priors = {spec['priors']!r} "
                                  "is not a key of the priors file")
            lo, hi = prior_bounds(priors[spec["priors"]], names, f"priors[{spec['priors']}]")
            resolved[s] = dict(psd=fn, names=names, lower=lo, upper=hi)
            if s == "cor":
                ofn = registry.orf(spec["orf"])
                onames = registry.orf_param_names(ofn)
                olo = ohi = None
                if onames:
                    olo, ohi = prior_bounds(priors[spec["priors"]].get("orf"), onames,
                                            f"priors[{spec['priors']}].orf")
                resolved[s].update(orf=ofn, orf_names=onames, orf_lower=olo, orf_upper=ohi)
        except (KeyError, ConfigError, ValueError) as exc:
            errs.append(f"signal {s!r}: {exc}")
    if "gtm" in present and sig["gtm"]["psd"] == "free_spectrum":
        try:
            key = sig["gtm"]["priors"]
            if key not in priors:
                raise ConfigError(f"model.signals.gtm.priors = {key!r} is not a key of the "
                                  "priors file")
            entry = priors[key] or {}
            pair = entry.get("halflog10_rho")
            if not _is_pair(pair) or not float(pair[0]) < float(pair[1]):
                raise ConfigError(f"priors[{key}] needs halflog10_rho: [low, high] "
                                  "(one pair, used for every PCA column)")
            resolved["gtm"] = dict(lower=float(pair[0]), upper=float(pair[1]))
        except ConfigError as exc:
            errs.append(f"signal 'gtm': {exc}")
    if "sw" in present and sig["sw"]["fixed_log10_sigma"] is None:
        try:
            if sig["sw"]["priors"] not in priors:
                raise ConfigError(f"model.signals.sw.priors = {sig['sw']['priors']!r} "
                                  "is not a key of the priors file")
            lo, hi = prior_bounds(priors[sig["sw"]["priors"]], ["log10_amp"],
                                  f"priors[{sig['sw']['priors']}]")
            resolved["sw"] = dict(lower=lo, upper=hi)
        except ConfigError as exc:
            errs.append(f"signal 'sw': {exc}")
    if "sw" in present and sig["sw"]["geometry"] not in ("pint", "approx", "stored"):
        errs.append("model.signals.sw.geometry must be pint, approx or stored")
    if "det" in present:
        try:
            registry.waveform(sig["det"]["waveform"])
            entry = priors.get(sig["det"]["priors"]) or {}
            if not entry:
                raise ConfigError(f"priors[{sig['det']['priors']!r}] is empty or missing")
            lo, hi = prior_bounds(entry, list(entry), f"priors[{sig['det']['priors']}]")
            resolved["det"] = dict(names=list(entry), lower=lo, upper=hi)
        except (KeyError, ConfigError) as exc:
            errs.append(f"signal 'det': {exc}")
        if run["data"]["pulsar_distances"] is None:
            errs.append("a 'det' signal needs data.pulsar_distances")
    pf = sig["cor"].get("prior_flow")
    if pf is not None:
        allowed = {"file", "flow_num_layers", "hidden_size", "mlp_num_layers", "num_bins", "B",
                   "mass_samples"}
        if not isinstance(pf, dict) or "file" not in pf:
            errs.append("model.signals.cor.prior_flow must be null or {file: flow.npz, ...}")
        elif set(pf) - allowed:
            errs.append(f"model.signals.cor.prior_flow: unknown keys {sorted(set(pf) - allowed)}; "
                        f"expected {sorted(allowed)}")
        elif "cor" not in present:
            errs.append("model.signals.cor.prior_flow needs a 'cor' block in the basis string")
        elif check_files and not Path(_resolve(run, pf["file"])).exists():
            errs.append(f"model.signals.cor.prior_flow.file not found: {pf['file']}")
    if "astro" in present:
        astro_errs, resolved["astro"] = _validate_astro(run, parsed, priors, check_files)
        errs += astro_errs
    g = sig["gtm"]
    n_src = sum(bool(x) for x in (g["basis_file"], g["basis_pattern"], g["from_pulsar"]))
    if "gtm" in present and n_src != 1:
        errs.append("a 'gtm' signal needs exactly one of model.signals.gtm.basis_file, "
                    "basis_pattern or from_pulsar")
    if g["psd_file"] and g["psd_value"] is not None:
        errs.append("model.signals.gtm: give psd_file or psd_value, not both")
    if g["psd"] not in ("fixed", "free_spectrum"):
        errs.append("model.signals.gtm.psd must be fixed or free_spectrum")
    if "gtm" in present and g["psd"] == "free_spectrum":
        if g["psd_file"] or g["psd_value"] is not None:
            errs.append("model.signals.gtm: psd free_spectrum samples the variances; "
                        "remove psd_file / psd_value")
        if g["size"] is not None and isinstance(g["size"], int) and g["size"] % 2:
            errs.append("model.signals.gtm.size must be even")

    # -- files --------------------------------------------------------------------
    if check_files:
        d = run["data"]
        src = d["source"]
        if src == "par_tim":
            pars, tims = _files(d["par_files"], run), _files(d["tim_files"], run)
            if not pars or not all(Path(p).exists() for p in pars):
                errs.append(f"data.par_files matched nothing: {d['par_files']}")
            if not tims or not all(Path(t).exists() for t in tims):
                errs.append(f"data.tim_files matched nothing: {d['tim_files']}")
        elif src == "enterprise_pickle":
            if not d["pickle"] or not Path(_resolve(run, d["pickle"])).exists():
                errs.append(f"data.pickle not found: {d['pickle']}")
        elif src == "atlas_npz":
            if d["npz_files"]:
                pat = _resolve(run, d["npz_files"])
                pulsars = d["pulsars"] if d["pulsars"] != "all" else None
                if "{psr}" in pat and pulsars:
                    miss = [pat.replace("{psr}", n) for n in pulsars
                            if not Path(pat.replace("{psr}", n)).exists()]
                    if miss:
                        errs.append(f"data.npz_files: {len(miss)} file(s) missing, e.g. {miss[:3]}")
                elif not glob.glob(pat.replace("{psr}", "*")):
                    errs.append(f"data.npz_files matched nothing: {d['npz_files']}")
            elif not d["npz_dir"] or not Path(_resolve(run, d["npz_dir"])).is_dir():
                errs.append(f"data.npz_dir not found: {d['npz_dir']} (or set data.npz_files)")
            if tm["nonlinear"]["enabled"]:
                pars, tims = _files(d["par_files"], run), _files(d["tim_files"], run)
                if not pars or not all(Path(p).exists() for p in pars) or \
                        not tims or not all(Path(t).exists() for t in tims):
                    errs.append("nonlinear timing: data.par_files / data.tim_files matched nothing")
        else:
            errs.append("data.source must be par_tim | enterprise_pickle | atlas_npz")
        for key in ("noise_dict", "pulsar_distances"):
            if d[key] is not None and not Path(_resolve(run, d[key])).exists():
                errs.append(f"data.{key} not found: {d[key]}")
        if d["noise_dict"] is None and not run["white_noise"]["vary"]:
            errs.append("fixed white noise (white_noise.vary: false) needs data.noise_dict")
        if d["cache_dir"] is None:
            errs.append("data.cache_dir is required")
        ci = sig["chrom"]["chrom_index"]
        if isinstance(ci, str) and not Path(_resolve(run, ci)).exists():
            errs.append(f"model.signals.chrom.chrom_index file not found: {ci}")
        for key in ("basis_file", "psd_file"):
            f = sig["gtm"][key]
            if "gtm" in present and f and not Path(_resolve(run, f)).exists():
                errs.append(f"model.signals.gtm.{key} not found: {f}")
        if "gtm" in present and g["basis_pattern"] and d["pulsars"] != "all":
            for key in ("basis_pattern", "mean_pattern"):
                pat = g[key]
                if pat:
                    miss = [n for n in d["pulsars"]
                            if not Path(_resolve(run, pat.replace("{psr}", n))).exists()]
                    if miss:
                        errs.append(f"model.signals.gtm.{key}: no file for {miss[:5]}")
        if custom:
            f = smp["model"].rsplit(":", 1)[0]
            if not Path(_resolve(run, f)).exists():
                errs.append(f"sampler.model file not found: {f}")
        for k in ("irn_user_freqs", "gwb_user_freqs", "dm_user_freqs", "chrom_user_freqs"):
            v = fb[k]
            f = v.get("file") if isinstance(v, dict) else (v.rsplit(":", 1)[0] if isinstance(v, str) else None)
            if f and not Path(_resolve(run, f)).exists():
                errs.append(f"frequency_bins.{k}: file not found: {f}")

    if errs:
        raise ConfigError(f"run {name!r} cannot run:\n" + "\n".join(f"  - {e}" for e in errs))
    return parsed, resolved


# ng20astromethods flow directory: torch.save([flow]) plus its transform metadata
_NG20_PICKLE, _NG20_MAPPING = "zuko_flow.pkl", "transform_data.npz"


def _flow_info(run, a):
    """What an astro flow's files say about it, read without flax or torch.

    ``a`` is ``model.signals.astro``.  Returns ``nf_type``, ``n_params``,
    ``n_freqs``, ``param_names``, ``param_space``, ``offset`` (an offset-mode
    flow, sampled in y = log10(rho + c)) and ``slab_dex``, or None when the files
    cannot tell (a Pandora pickle + mapping: the mapping alone does not
    separate rho from theta) or are missing.
    """
    import numpy as np

    if a["flow"] is None:
        return None
    flow = Path(_resolve(run, a["flow"]))
    config, ng20 = flow / "flow_config.json", flow / _NG20_MAPPING
    if config.exists():
        # Atlas flax or zuko flow: the metadata sits at the top level of the config
        c = json.loads(config.read_text())
        n_params, n_total = int(c["n_params"]), len(c["mapping"]["mean"])
        nf_type, names, pspace = c["nf_type"], c.get("param_names"), c.get("param_space")
        zero, slab = c.get("zero_handling"), c.get("slab_dex", 0.0)
    elif ng20.exists():
        with np.load(ng20) as z:
            n_params, n_total = len(z["ast_param_idxs"]), len(z["mean"])
            names = [str(n) for n in z["context_param_names"]] if "context_param_names" in z.files else None
            pspace = str(z["pspace_class"]) if "pspace_class" in z.files else None
            zero = str(z["zero_handling"]) if "zero_handling" in z.files else None
            # flows saved before the zero slab load with slab_dex = 0, as in validate_flow
            slab = float(z["slab_dex"]) if "slab_dex" in z.files else 0.0
        nf_type = "rho|theta"
    else:
        return None
    return dict(nf_type=nf_type, n_params=n_params, n_freqs=n_total - n_params,
                param_names=list(names or [f"theta_{i}" for i in range(n_params)]),
                param_space=pspace, offset=zero == "offset", slab_dex=float(slab or 0.0))


def _validate_astro(run, parsed, priors, check_files):
    """Problems with an ``astro`` block, and its resolved settings for the builder.

    Checks that need the flow's description (bins, parameter names) use
    ``_flow_info`` and are skipped when it is unavailable; ``AstroGWB`` and the
    builder check again at build time.
    """
    a, fb = run["model"]["signals"]["astro"], run["frequency_bins"]
    present = parsed["shared_names"] + parsed["separate_names"]
    errs = []
    if "cor" in present:
        errs.append("'cor' and 'astro' cannot be combined yet: use one common process")

    # -- the flow: an Atlas or ng20 directory, or a Pandora pickle + mapping --------
    pandora = a["mapping"] is not None
    if pandora and a["pickle"] is None:
        errs.append("model.signals.astro.mapping (a Pandora flow) needs model.signals.astro.pickle")
    if pandora and a["flow"] is not None:
        errs.append("model.signals.astro: give flow (a flow directory) or pickle + mapping "
                    "(a Pandora flow), not both")
    if not pandora and a["flow"] is None:
        errs.append("an 'astro' block needs model.signals.astro.flow (a flow directory) or "
                    "pickle + mapping (a Pandora flow)")
    needs_torch = pandora
    if check_files and not errs:
        if pandora:
            for key in ("pickle", "mapping"):
                if not Path(_resolve(run, a[key])).exists():
                    errs.append(f"model.signals.astro.{key} not found: {a[key]}")
        else:
            flow = Path(_resolve(run, a["flow"]))
            if (flow / "flow_config.json").exists():
                pass
            elif (flow / _NG20_MAPPING).exists():
                pkl = flow / (a["pickle"] or _NG20_PICKLE)
                if not pkl.exists():
                    errs.append(f"model.signals.astro.flow: no {pkl.name} in {a['flow']}")
            else:
                errs.append(f"model.signals.astro.flow {a['flow']} holds neither flow_config.json "
                            f"(an Atlas flow) nor {_NG20_MAPPING} (an ng20astromethods flow)")
    info = None
    if not errs:
        try:
            info = _flow_info(run, a)
        except Exception as exc:
            errs.append(f"model.signals.astro.flow: cannot read the flow's description: {exc}")
    if info is not None:
        if info["nf_type"] != "rho|theta":
            errs.append(f"model.signals.astro.flow is a {info['nf_type']!r} flow; 'astro' needs "
                        "a 'rho|theta' flow")
        if info["offset"] and info["slab_dex"] <= 0:
            errs.append("model.signals.astro.flow is an offset-mode flow without a zero slab "
                        "(slab_dex = 0), which cannot be sampled; retrain with transform.slab_dex > 0")
        flow = Path(_resolve(run, a["flow"]))
        if (flow / "flow_config.json").exists():
            needs_torch = "format" not in json.loads((flow / "flow_config.json").read_text())
        else:
            needs_torch = True                     # an ng20 pickle
    # flax evaluates every flow in the model; torch only loads the zuko formats
    missing = [m for m in ("flax", "distrax") if importlib.util.find_spec(m) is None]
    if needs_torch:
        missing += [m for m in ("torch", "zuko") if importlib.util.find_spec(m) is None]
    if missing:
        errs.append(f"model.signals.astro: this flow needs {missing}, which are not installed "
                    "(torch-backed flows can be converted once with "
                    "FlaxAstroFlow.from_zuko(flow).save(directory))")
    names = info["param_names"] if info else None

    # -- fixed parameters and prior bounds ------------------------------------------
    fixed = a["fixed"] or {}
    if not isinstance(fixed, dict):
        errs.append("model.signals.astro.fixed must be {parameter name: value}")
        fixed = {}
    elif names is not None and set(fixed) - set(names):
        errs.append(f"model.signals.astro.fixed: {sorted(set(fixed) - set(names))} are not flow "
                    f"parameters {names}")
    rho, theta = None, {}
    if a["priors"] is not None:
        what = f"priors[{a['priors']}]"
        if a["priors"] not in priors:
            errs.append(f"model.signals.astro.priors = {a['priors']!r} is not a key of the "
                        "priors file")
        else:
            theta = dict(priors[a["priors"]] or {})
            rho = theta.pop("halflog10_rho", None)
            pairs = [rho] if _is_pair(rho) else (list(rho) if isinstance(rho, list) else None)
            if rho is not None and (not pairs or not all(_is_pair(p) and p[0] < p[1] for p in pairs)):
                errs.append(f"{what}: halflog10_rho must be one [low, high] pair (every bin) or a "
                            "list with one pair per bin")
            elif rho is not None and not _is_pair(rho) and info is not None \
                    and len(pairs) != info["n_freqs"]:
                errs.append(f"{what}: {len(pairs)} halflog10_rho bounds for the flow's "
                            f"{info['n_freqs']} bins")
            for k, v in theta.items():
                if names is not None and k not in names:
                    errs.append(f"{what}: {k!r} is not a flow parameter {names}")
                elif k in fixed:
                    errs.append(f"{what}: {k!r} is fixed (model.signals.astro.fixed); drop its bounds")
                elif not (_is_pair(v) and v[0] < v[1]):
                    errs.append(f"{what}: bound for {k!r} must be [low, high] with low < high, got {v}")

    # -- extra log prior on theta ------------------------------------------------------
    lp = a["log_prior"]
    if lp not in (None, "pspace") and not (isinstance(lp, str) and ":" in lp):
        errs.append("model.signals.astro.log_prior must be null, pspace or path/file.py:function")
    elif isinstance(lp, str) and ":" in lp and check_files \
            and not Path(_resolve(run, lp.rsplit(":", 1)[0])).exists():
        errs.append(f"model.signals.astro.log_prior file not found: {lp.rsplit(':', 1)[0]}")
    # a Pandora pickle never records its space; an unreadable flow is checked at build time
    if lp == "pspace" and a["pspace"] is None and \
            (pandora or (info is not None and not info["param_space"])):
        errs.append("model.signals.astro.log_prior: pspace needs model.signals.astro.pspace "
                    "(a holodeck parameter-space name) or a flow that records param_space")
    if a["pspace"] is not None and lp != "pspace":
        errs.append("model.signals.astro.pspace is only used with log_prior: pspace")

    # -- Fourier basis -------------------------------------------------------------------
    if a["tspan_policy"] not in ("flow", "strict"):
        errs.append("model.signals.astro.tspan_policy must be flow or strict")
    if not isinstance(a["tspan_rtol"], (int, float)) or isinstance(a["tspan_rtol"], bool) \
            or a["tspan_rtol"] <= 0:
        errs.append("model.signals.astro.tspan_rtol must be a positive number")
    n = a["n_prior_draws"]
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        errs.append("model.signals.astro.n_prior_draws must be an integer >= 0 (0 skips the "
                    "flow-prior spectrum in summary.json)")
    if fb["gwb_user_freqs"] is not None:
        errs.append("frequency_bins.gwb_user_freqs cannot be used with 'astro': the flow fixes "
                    "the GWB frequencies")
    if info is not None and parsed["representative"] == "unc" and "astro" in parsed["shared_names"] \
            and isinstance(fb["num_irn_bins"], int) and info["n_freqs"] > fb["num_irn_bins"]:
        errs.append(f"with 'unc+astro->unc' the flow's {info['n_freqs']} GWB bins sit inside the "
                    f"IRN basis: frequency_bins.num_irn_bins must be >= {info['n_freqs']}")

    return errs, dict(rho=rho, theta=theta, fixed=dict(fixed), log_prior=lp)


def load_function(run, spec):
    """``'path/to/file.py:function'`` (path relative to the config file) -> the function."""
    f, fn = spec.rsplit(":", 1)
    path = _resolve(run, f)
    module_spec = importlib.util.spec_from_file_location(f"atlas_custom_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(mod)
    if not hasattr(mod, fn):
        raise AttributeError(f"{path} has no function {fn!r}")
    return getattr(mod, fn)


def netcdf_engine():
    """An xarray engine that can write NetCDF groups, or None."""
    has = lambda m: importlib.util.find_spec(m) is not None
    if not has("xarray"):
        return None
    if has("netCDF4"):
        return "netcdf4"
    if has("h5netcdf") and has("h5py"):
        return "h5netcdf"
    return None


def resolve_path(run, p):
    """Public form of the path resolution used throughout (config-relative)."""
    return _resolve(run, p)


def dump_run(run, path):
    """Write a resolved run as YAML (functions and private keys dropped)."""
    clean = {k: v for k, v in run.items() if not k.startswith("_")}
    clean["run"] = dict(clean["run"])
    with open(path, "w") as fh:
        yaml.safe_dump(json.loads(json.dumps(clean, default=str)), fh, sort_keys=False)
