# ATLAS/signals

See the top-level [README](../../README.md) for the likelihood, the model-string
syntax and the resulting column layout.

- `factorized/` — `Red`, `GaussianTiming` and `SuperSignal`: basis construction,
  helper assembly and the three likelihood entry points.
- `correlated/` — `Correlated`: the background signal and its overlap reduction
  function. `AstroGWB` (`astro_gwb.py`) is the `astro` background: an HD free
  spectrum with a normalizing-flow prior over astrophysical parameters.
- `deterministic/` — continuous-wave and other deterministic signals. No
  end-to-end path currently runs; see "Known issues" in the top-level README.
- `timing/` — the non-linear timing model, built on JUG.
- `signals_utils.py` — model-string parser, column bookkeeping
  (`merge_slices`, `block_slice`, `vec_slice`) and the timing-model SVD.
