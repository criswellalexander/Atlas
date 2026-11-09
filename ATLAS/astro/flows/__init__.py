"""
Conditional normalizing flows for the astrophysical GWB: training sets
(``trainset``), training (``train``), the zuko and flax flows (``zuko``,
``flax``) and shared training utilities (``utils``).

Submodules are not imported here: ``zuko`` needs torch and ``flax`` needs
flax/distrax/optax, so each is imported only when used.
"""
