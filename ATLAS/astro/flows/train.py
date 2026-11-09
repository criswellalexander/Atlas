"""
One-call flow training on a training set (Pandora's full training path).

:func:`train_astro_flow` normalizes a
:class:`~ATLAS.astro.flows.trainset.TrainingSet` to Pandora's ``[-B, B]``
space, optionally holds out a validation split for Hellinger-distance early
stopping, and trains either a zuko (torch) flow with
``ATLAS.astro.flows.zuko.FlowTrainer`` or a flax (JAX) flow with
``ATLAS.astro.flows.flax.FlaxFlowTrainer``. Both backends are imported lazily,
so this module imports without torch or flax.
"""

__all__ = ["train_astro_flow"]


def train_astro_flow(training_set, save_dir, nf_type="rho|theta", B=5, val=None,
                     split_seeds=(0, 1), steps=10_000, batch_size=512, save_freq=1_000,
                     mode="diagonal", device=None, spline_bins=8, hidden_dims=(512, 512),
                     transforms=3, train_kwargs=None, backend="zuko", architecture="zuko_nsf",
                     arch_kwargs=None, seed=0):
    """
    Train a conditional flow on a training set (Pandora's full training path).

    Normalizes ``training_set`` to ``[-B, B]`` (the same map Pandora saves in
    ``*_mapping_data.npy.npz``), optionally holds out a validation set for
    Hellinger-distance early stopping, trains a flow, and saves it.

    Parameters
    ----------
    training_set : TrainingSet
    save_dir : str
        Receives checkpoints, ``flow_config.json`` and the final weights;
        reload with ``ZukoAstroFlow.load(save_dir)`` or
        ``FlaxAstroFlow.load(save_dir)``.
    nf_type : {'rho|theta', 'theta|rho', 'rho'}
    B : float
    val : (n_val_draws, n_val_real) or None
        Validation split (see :meth:`TrainingSet.split`). Not available for
        ``'theta|rho'``.
    split_seeds : (int, int)
    steps, batch_size, save_freq, mode
        See ``ATLAS.astro.flows.zuko.FlowTrainer.train``. ``mode='flat'`` reproduces the
        loop in ``AstroInferenceUpdated.ipynb``.
    device, spline_bins, hidden_dims, transforms
        See ``ATLAS.astro.flows.zuko.FlowTrainer``. The notebook uses
        ``hidden_dims=[512] * 8``. With ``backend='flax'`` and
        ``architecture='zuko_nsf'`` they define the same flow in JAX (and
        ``device`` is ignored: JAX uses its default device).
    train_kwargs : dict, optional
        Further ``train`` arguments (``learning_rate``, ``seed``,
        ``patience``, ...).
    backend : {'zuko', 'flax'}
        Train with torch (``ATLAS.astro.flows.zuko.FlowTrainer``) or JAX
        (``ATLAS.astro.flows.flax.FlaxFlowTrainer``).
    architecture : {'zuko_nsf', 'coupling_rqs'}
        Flax architecture (``backend='flax'`` only).
    arch_kwargs : dict, optional
        Flax architecture arguments; for ``'zuko_nsf'`` they default to
        ``spline_bins``, ``hidden_dims`` and ``transforms``.
    seed : int
        Weight initialization for ``backend='flax'`` (the zuko backend
        follows ``torch.manual_seed``).

    Returns
    -------
    flow : ATLAS.astro.flows.zuko.ZukoAstroFlow or ATLAS.astro.flows.flax.FlaxAstroFlow
    history : dict
    """
    from ATLAS.astro.flows.utils import NF_TYPES

    if backend not in ("zuko", "flax"):
        raise ValueError(f"backend must be 'zuko' or 'flax', got {backend!r}")
    if nf_type not in NF_TYPES:
        raise ValueError(f"nf_type must be one of {NF_TYPES}, got {nf_type!r}")
    mapping = training_set.normalization(B)
    if val is not None:
        if nf_type == "theta|rho":
            raise ValueError("validation is only available for 'rho|theta' and 'rho' flows")
        train_set, val_set = training_set.split(*val, seeds=split_seeds)
    else:
        train_set, val_set = training_set, None

    rho_n, ast_n, _ = train_set.normalize(mapping=mapping)
    inputs, context = {"rho|theta": (rho_n, ast_n), "theta|rho": (ast_n, rho_n),
                       "rho": (rho_n, None)}[nf_type]

    validation = None
    if val_set is not None:
        v_rho, v_ast, _ = val_set.normalize(mapping=mapping)
        if nf_type == "rho|theta":
            # (n_ctx, n_val_real, n_f) -> (n_ctx * n_f, n_val_real), Pandora's layout
            validation = (v_rho.transpose(0, 2, 1).reshape(-1, v_rho.shape[1]), v_ast)
        else:
            # unconditional: all held-out samples of each frequency, (n_f, n)
            validation = (v_rho.reshape(-1, v_rho.shape[-1]).T, None)

    metadata = dict(nf_type=nf_type, n_params=training_set.n_params, mapping=mapping,
                    param_names=list(training_set.param_names),
                    f_conv=None if training_set.f_conv is None else list(training_set.f_conv),
                    tspan=training_set.tspan)
    flow_meta = {k: metadata[k] for k in ("param_names", "f_conv", "tspan")}
    if backend == "flax":
        from ATLAS.astro.flows import flax as ff

        if arch_kwargs is None and architecture == "zuko_nsf":
            arch_kwargs = dict(bins=spline_bins, hidden_features=list(hidden_dims),
                               transforms=transforms)
        trainer = ff.FlaxFlowTrainer(inputs, context, save_dir, B, architecture=architecture,
                                     arch_kwargs=arch_kwargs, metadata=metadata, seed=seed)
        history = trainer.train(steps, batch_size, save_freq, mode=mode, validation=validation,
                                **(train_kwargs or {}))
        flow = ff.FlaxAstroFlow(trainer.model, nf_type, mapping, training_set.n_params,
                                metadata=flow_meta)
    else:
        from ATLAS.astro.flows import zuko as zf

        trainer = zf.FlowTrainer(inputs, context, save_dir, B, device=device,
                                 spline_bins=spline_bins, hidden_dims=hidden_dims,
                                 transforms=transforms, metadata=metadata)
        history = trainer.train(steps, batch_size, save_freq, mode=mode, validation=validation,
                                **(train_kwargs or {}))
        flow = zf.ZukoAstroFlow(trainer.flow, nf_type, mapping, training_set.n_params,
                                device=trainer.device, metadata=flow_meta)
    flow.save(save_dir)
    return flow, history
