"""
Conditional normalizing flows for the astrophysical GWB: training sets
(``trainset``), training (``train``), the zuko and flax flows (``zuko``,
``flax``) and shared training utilities (``utils``).

Submodules are not imported here: ``zuko`` needs torch and ``flax`` needs
flax/distrax/optax, so each is imported only when used. ``load_astro_flow``
loads a trained flow from any supported format.
"""

import json
import os

__all__ = ["load_astro_flow"]

# ng20astromethods' flow directory: torch.save([flow]) plus its transform metadata
NG20_PICKLE, NG20_MAPPING = "zuko_flow.pkl", "transform_data.npz"


def load_astro_flow(path=None, pickle=None, mapping=None, device="cpu"):
    '''

    Load a trained flow from any of the formats Atlas reads, detected from
    the files present. The flow is returned as stored: ``AstroGWB`` converts
    zuko flows to flax itself.

    - ``path`` holding ``flow_config.json``: a flow saved by
      ``FlaxAstroFlow.save`` or ``ZukoAstroFlow.save``.
    - ``path`` holding ``transform_data.npz``: an ng20astromethods flow
      (``pickle`` defaults to ``zuko_flow.pkl``), with its metadata.
    - ``pickle`` and ``mapping``: a Pandora ``NFMaker`` pickle and its
      ``*_mapping_data.npy.npz`` (relative to ``path`` if given).

    Pickled flows are unpickled with torch, which runs code from the file, so
    only load files you trust. Every format but flax needs torch and zuko.

    Arguments
    ------------------------------
    path : str or None
    	Flow directory. (Default: None)
    pickle : str or None
    	Pickled flow, for the Pandora and ng20 formats. (Default: None)
    mapping : str or None
    	Pandora mapping file; selects the Pandora format. (Default: None)
    device : str
    	Torch device of a zuko flow. (Default: 'cpu')

    Returns
    ------------------------------
    flow : ATLAS.astro.flows.flax.FlaxAstroFlow or ATLAS.astro.flows.zuko.ZukoAstroFlow
    	The loaded flow.

    '''
    def at(name):
        return name if path is None or os.path.isabs(name) else os.path.join(path, name)

    if mapping is not None:
        if pickle is None:
            raise ValueError("a Pandora flow needs both pickle and mapping")
        return _zuko().ZukoAstroFlow.from_pandora(at(pickle), at(mapping), device=device)
    if path is None:
        raise ValueError("give a flow directory (path) or a Pandora pickle and mapping")
    config = os.path.join(path, "flow_config.json")
    if os.path.exists(config):
        with open(config) as fh:
            is_flax = "format" in json.load(fh)
        if is_flax:
            from ATLAS.astro.flows.flax import FlaxAstroFlow
            return FlaxAstroFlow.load(path)
        return _zuko().ZukoAstroFlow.load(path, device=device)
    if os.path.exists(os.path.join(path, NG20_MAPPING)):
        return _zuko().ZukoAstroFlow.from_pandora(at(pickle or NG20_PICKLE), at(NG20_MAPPING),
                                                  device=device)
    raise FileNotFoundError(f"{path} holds neither flow_config.json (an Atlas flow) nor "
                            f"{NG20_MAPPING} (an ng20astromethods flow)")


def _zuko():
    try:
        from ATLAS.astro.flows import zuko
    except ImportError as exc:
        raise ImportError(
            "this flow is stored with torch and needs torch and zuko (e.g. atlas-env-2); "
            "convert it once with FlaxAstroFlow.from_zuko(flow).save(directory) to use it "
            "without torch") from exc
    return zuko
