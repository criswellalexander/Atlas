"""
Hierarchical astrophysical inference of the gravitational-wave background.

Ported from Pandora: an astrophysical population model (holodeck) is emulated
by a conditional normalizing flow ``p(log10_rho | theta)`` over the GWB free
spectrum, which then acts as a hierarchical prior on the GWB in a PTA analysis.

Layout
------
``ATLAS.astro.flows.trainset``
    holodeck training sets of ``log10_rho`` given astrophysical parameters.
``ATLAS.astro.flows.train``
    ``train_astro_flow``: one-call flow training on a training set.
``ATLAS.astro.flows.zuko`` / ``ATLAS.astro.flows.flax``
    The flows: zuko (torch, Pandora's) and flax (JAX, differentiable).
``ATLAS.astro.flows.utils``
    Batch sampling and Hellinger-distance validation shared by both trainers.
``ATLAS.astro.inference``
    The standalone hierarchical model (``HDAstroRedModel``,
    ``AstroInferenceModel``) sampled with PTMCMC.
``ATLAS.astro.utils``
    Training-set file layout and prior-bound helpers.

Submodules are not imported here, so ``import ATLAS.astro`` does not pull in
holodeck, torch, flax or PTMCMCSampler (the ``[astro]`` extra).
"""
