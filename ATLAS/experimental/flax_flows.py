"""
Flax NNX port of the RQS coupling flows in ``flows.py``.

``Flow`` and ``ConditionalFlow`` here are drop-in replacements for the Haiku
classes of the same name: same constructors, same fit modes, same public
methods, same power-affine normalisation.  Differences from ``flows.py``:

- The network is an ``nnx.Module`` held in ``self.model``; the optimiser is an
  ``nnx.Optimizer`` in ``self.optimizer``.  There is no ``self.params`` /
  ``self.opt_state`` pytree to thread by hand (``params`` is a read-only view).
- Jitted methods receive the weights as an argument instead of closing over
  ``self.params``, so results always reflect the current weights.  In
  ``flows.py`` the first call baked the weights in as constants and later
  calls after ``fit()`` / ``load_params()`` silently reused them.  Inference
  methods also write nothing back to the model, so they can be called inside
  an outer ``jax.jit`` / ``grad`` / ``vmap`` (as astro.py does).
- ``ConditionalFlow.sample`` uses each row of ``c`` for its own sample.  The
  Haiku version went through distrax's vmapped ``Transformed.sample``, under
  which the conditioner only ever saw ``c[0]``.
- ``save_params`` writes parameters keyed by their path in the module tree;
  ``load_params`` rejects files from the Haiku version (retrain instead).

"""

import os
import pickle
import random
from functools import partial

import distrax
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax import nnx
from tqdm.auto import trange

jax.config.update("jax_enable_x64", True)

_PARAMS_FORMAT = "atlas-flax-nnx-v1"

# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers 
# ─────────────────────────────────────────────────────────────────────────────

def _fit_normalizer(min_x, max_x, p, B):
    """
    Fit the power-affine normalizer from user-supplied range statistics.

    The forward direction (physical → normalized) is:
        dummy = sign(x) * |x|^exp(alpha)          # power stretch
        norm  = (dummy - mean) / half_range * B   # affine to [-B, B]

    The inverse (normalized → physical) is:
        dummy = half_range/B * norm + mean
        x     = sign(dummy) * |dummy|^(1/exp(alpha))

    Returns
    -------
    dict with keys: alpha, mean, half_range, B
    """
    min_x = np.asarray(min_x, dtype=np.float64)
    max_x = np.asarray(max_x, dtype=np.float64)
    alpha = np.log(float(p))
    gamma = np.exp(alpha)
    dummy_min = np.sign(min_x) * np.abs(min_x) ** gamma
    dummy_max = np.sign(max_x) * np.abs(max_x) ** gamma
    mean      = (dummy_max + dummy_min) / 2.0
    half_range = (dummy_max - dummy_min) / 2.0
    return dict(alpha=alpha, mean=mean, half_range=half_range, B=float(B))


def _from_coefficients_np(arr, norm):
    """Physical → normalized  (numpy, used during batched training reads)."""
    alpha, mean, half_range, B = norm['alpha'], norm['mean'], norm['half_range'], norm['B']
    dummy = np.sign(arr) * np.abs(arr) ** np.exp(alpha)
    return (dummy - mean) / half_range * B


def _to_coefficients_np(arr, norm):
    """Normalized → physical  (numpy, used at inference)."""
    alpha, mean, half_range, B = norm['alpha'], norm['mean'], norm['half_range'], norm['B']
    dummy = half_range / B * arr + mean
    return np.sign(dummy) * np.abs(dummy) ** (1.0 / np.exp(alpha))


def _from_coefficients_jnp(arr, alpha, mean, half_range, B):
    """Physical → normalized  (jax, used inside jit-compiled methods)."""
    dummy = jnp.sign(arr) * jnp.abs(arr) ** jnp.exp(alpha)
    return (dummy - mean) / half_range * B


def _to_coefficients_jnp(arr, alpha, mean, half_range, B):
    """Normalized → physical  (jax, used inside jit-compiled methods)."""
    dummy = half_range / B * arr + mean
    return jnp.sign(dummy) * jnp.abs(dummy) ** (1.0 / jnp.exp(alpha))


def _logdet_from_coefficients_jnp(arr, alpha, mean, half_range, B):
    """
    Log |det J| of the physical → normalized map.
    J is diagonal so log|det J| = sum_i log|df_i/dx_i|.

    df/dx = exp(alpha) * |x|^(exp(alpha)-1) * (B / half_range)
    """
    gamma = jnp.exp(alpha)
    return jnp.sum(
        jnp.log(gamma)
        + (gamma - 1.0) * jnp.log(jnp.abs(arr))
        + jnp.log(B / half_range),
        axis=-1,
    )


def _logdet_to_coefficients_jnp(arr, alpha, mean, half_range, B):
    """
    Log |det J| of the normalized → physical map.
    """
    beta = 1.0 / jnp.exp(alpha)
    dummy = half_range / B * arr + mean
    return jnp.sum(
        jnp.log(half_range / B)
        + jnp.log(beta)
        + (beta - 1.0) * jnp.log(jnp.abs(dummy)),
        axis=-1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Memmapped-safe batch iterator
# ─────────────────────────────────────────────────────────────────────────────

def _iter_batches(data, context, x_norm, c_norm, batch_size, key):
    """
    Yield (x_batch, c_batch) pairs in random order without loading the
    full dataset into memory.

    Shuffling draws a random permutation of row indices; each batch then
    reads only `batch_size` rows from the (possibly memmapped) arrays,
    normalises them on the fly, and converts to jax arrays.  Peak memory is
    O(batch_size * (D + C)), regardless of N.  The final partial batch is
    dropped so the jitted step is not retraced for a new shape.
    """
    N = data.shape[0]
    perm = np.array(jr.permutation(key, N))          # numpy for fancy indexing
    for start in range(0, N, batch_size):
        idx = perm[start: start + batch_size]
        if len(idx) < batch_size:
            continue
        # Fancy-index into mmap → materialises only this slice
        x_raw = np.asarray(data[idx], dtype=np.float64)
        c_raw = np.asarray(context[idx], dtype=np.float64)
        x_batch = jnp.asarray(_from_coefficients_np(x_raw, x_norm))
        c_batch = jnp.asarray(_from_coefficients_np(c_raw, c_norm))
        yield x_batch, c_batch


def _norm_tuple(norm):
    """Normalizer dict → (alpha, mean, half_range, B) as jax float64 arrays."""
    return (jnp.float64(norm['alpha']), jnp.asarray(norm['mean'], dtype=jnp.float64),
            jnp.asarray(norm['half_range'], dtype=jnp.float64), jnp.float64(norm['B']))


# ─────────────────────────────────────────────────────────────────────────────
# NNX modules
# ─────────────────────────────────────────────────────────────────────────────

class _Conditioner(nnx.Module):
    """
    MLP that maps [x_masked ‖ c] to RQS parameters of shape (..., D, 3K+1).

    Hidden layers are ReLU-activated, including the last one (Haiku's
    ``hk.nets.MLP(activate_final=True)``).  The output layer is
    zero-initialised, which makes every spline start as the identity, so a
    freshly built flow is exactly the base N(0, I).
    """

    def __init__(self, in_features, D, hidden_sizes, num_params, *, rngs):
        sizes = [in_features, *hidden_sizes]
        self.hidden = nnx.List([
            nnx.Linear(a, b, param_dtype=jnp.float64, rngs=rngs)
            for a, b in zip(sizes[:-1], sizes[1:])
        ])
        self.out = nnx.Linear(
            sizes[-1], D * num_params,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            param_dtype=jnp.float64, rngs=rngs,
        )
        self.D = D
        self.num_params = num_params

    def __call__(self, x_masked, c=None):
        h = x_masked if c is None else jnp.concatenate([x_masked, c], axis=-1)
        for layer in self.hidden:
            h = jax.nn.relu(layer(h))
        return self.out(h).reshape(h.shape[:-1] + (self.D, self.num_params))


class _CouplingRQSFlow(nnx.Module):
    """
    Stack of masked-coupling RQS layers over a standard-normal base, shared
    by ``Flow`` (C = 0) and ``ConditionalFlow``.

    Layer i transforms the features where the alternating ``arange(D) % 2``
    mask is False, with spline parameters produced from the features where it
    is True (and the context, if any).  The distrax objects are rebuilt on
    every call; only the conditioner weights are state.  As in ``flows.py``,
    the chain is wrapped in ``Inverse`` so that ``log_prob`` runs the layers
    data → latent and ``forward`` runs them latent → data.
    """

    def __init__(self, D, C, num_layers, hidden_sizes, num_bins, B, *, rngs):
        self.D = int(D)
        self.C = int(C)
        self.num_bins = int(num_bins)
        # Must be a plain Python float -- RationalQuadraticSpline compares
        # range_min >= range_max at construction, which cannot be traced.
        self.B = float(B)
        num_params = 3 * self.num_bins + 1
        self.conditioners = nnx.List([
            _Conditioner(self.D + self.C, self.D, hidden_sizes, num_params, rngs=rngs)
            for _ in range(num_layers)
        ])

    def _bijector(self, c=None):
        B = self.B

        def bijector_fn(params):
            return distrax.RationalQuadraticSpline(
                params, range_min=-B - 1, range_max=B + 1
            )

        mask = (jnp.arange(self.D) % 2).astype(bool)
        layers = []
        for conditioner in self.conditioners:
            layers.append(distrax.MaskedCoupling(
                mask=mask, bijector=bijector_fn,
                conditioner=partial(conditioner, c=c),
            ))
            mask = jnp.logical_not(mask)
        return distrax.Inverse(distrax.Chain(layers))

    def log_prob(self, u, c=None):
        base = distrax.Independent(
            distrax.Normal(jnp.zeros(self.D), jnp.ones(self.D)),
            reinterpreted_batch_ndims=1,
        )
        return distrax.Transformed(base, self._bijector(c)).log_prob(u)

    def forward(self, z, c=None):
        return self._bijector(c).forward(z)

    def forward_and_log_det(self, z, c=None):
        return self._bijector(c).forward_and_log_det(z)

    def inverse(self, u, c=None):
        return self._bijector(c).inverse(u)


# ─────────────────────────────────────────────────────────────────────────────
# Jitted functions
#
# Flax 0.12 modules (and nnx.Optimizer) are JAX pytrees, so every function
# here is a plain jax.jit that takes the model as an ordinary argument:
#   - the weights are an argument, never a closed-over constant, so a call
#     after fit() / load_params() sees the new values;
#   - nothing is written back to the caller's module, so the inference
#     functions can be called inside someone else's jax.jit / grad / vmap.
#     astro.py does exactly that (the flow is a static argument of an outer
#     jax.jit).  nnx.jit, by contrast, propagates state back out of the call
#     and raises TraceContextError under an outer transform;
#   - it is also the cheapest dispatch.  nnx.jit and nnx.state() walk the
#     module graph in Python on every call (~0.3 ms here), which tripled the
#     per-call cost of a batch-1 log_prob and slowed training ~2x
#     (see bench/bench_flows.py).
# _train_step returns the updated (model, optimizer) instead of mutating its
# arguments; the classes thread them through the epoch and write them back
# into self.model / self.optimizer once at the end (_commit).
#
# xn / cn are (alpha, mean, half_range, B) normalizer tuples for data and
# context; c and cn are None for the unconditional Flow.
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_normalize(c, cn):
    return None if c is None else _from_coefficients_jnp(c, *cn)


@jax.jit
def _log_prob_phys(model, x, c, xn, cn):
    """log p(x | c) in physical space."""
    u = _from_coefficients_jnp(x, *xn)
    return (model.log_prob(u, _maybe_normalize(c, cn))
            + _logdet_from_coefficients_jnp(x, *xn))


@jax.jit
def _forward_phys(model, z, c, xn, cn):
    """Latent z → physical x."""
    u = model.forward(z, _maybe_normalize(c, cn))
    return _to_coefficients_jnp(u, *xn)


@jax.jit
def _forward_phys_with_logdet(model, z, c, xn, cn):
    """Latent z → (physical x, log|det J_{z→x}|)."""
    u, logdet_flow = model.forward_and_log_det(z, _maybe_normalize(c, cn))
    return (_to_coefficients_jnp(u, *xn),
            logdet_flow + _logdet_to_coefficients_jnp(u, *xn))


@jax.jit
def _inverse_phys(model, x, c, xn, cn):
    """Physical x → latent z."""
    return model.inverse(_from_coefficients_jnp(x, *xn), _maybe_normalize(c, cn))


@jax.jit
def _train_step(model, optimizer, u, c):
    """
    One Adam step on the mean NLL of normalized data.

    Returns (model, optimizer, loss) with the updated weights and optimiser
    state; the arguments passed in are left untouched.
    """
    def loss_fn(m):
        return -jnp.mean(m.log_prob(u, c))

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return model, optimizer, loss


def _commit(dst_model, dst_optimizer, model, optimizer):
    """Write trained (model, optimizer) back into the long-lived objects."""
    nnx.update(dst_model, nnx.state(model))
    nnx.update(dst_optimizer, nnx.state(optimizer))


# ─────────────────────────────────────────────────────────────────────────────
# Parameter / checkpoint (de)serialisation
# ─────────────────────────────────────────────────────────────────────────────

def _path_str(path):
    return "/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path)


def _flat_params(model):
    pure = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    return {_path_str(p): np.asarray(v)
            for p, v in jax.tree_util.tree_flatten_with_path(pure)[0]}


def _save_params(model, path):
    save_dict = _flat_params(model)
    save_dict["__format__"] = np.array(_PARAMS_FORMAT)
    np.savez(path, **save_dict)


def _load_params(model, path):
    data = np.load(path, allow_pickle=False)
    keys = set(data.files)
    if "__format__" not in keys or str(data["__format__"]) != _PARAMS_FORMAT:
        origin = ("the Haiku implementation in flows.py"
                  if any(k.startswith("leaf_") for k in keys) else "an unknown source")
        raise ValueError(
            f"{path} holds parameters from {origin}, which cannot be loaded into "
            f"the Flax NNX flow.  Retrain the flow with flax_flows.py.")

    state = nnx.state(model, nnx.Param)
    pure = nnx.to_pure_dict(state)
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(pure)
    want = {_path_str(p): np.shape(v) for p, v in leaves_with_path}
    got = {k: data[k].shape for k in keys - {"__format__"}}
    if want != got:
        differ = sorted(k for k in want.keys() | got.keys() if want.get(k) != got.get(k))
        raise ValueError(
            f"{path} was written by a flow with a different architecture; "
            f"mismatched parameters: {differ[:6]}{' ...' if len(differ) > 6 else ''}")

    new_leaves = [jnp.asarray(data[_path_str(p)]) for p, _ in leaves_with_path]
    nnx.replace_by_pure_dict(state, jax.tree_util.tree_unflatten(treedef, new_leaves))
    nnx.update(model, state)


def _to_numpy_pure(node):
    return jax.tree_util.tree_map(np.asarray, nnx.to_pure_dict(nnx.state(node)))


def _restore_pure(node, pure):
    state = nnx.state(node)
    nnx.replace_by_pure_dict(state, jax.tree_util.tree_map(jnp.asarray, pure))
    nnx.update(node, state)


# ─────────────────────────────────────────────────────────────────────────────
# Flow
# ─────────────────────────────────────────────────────────────────────────────

class Flow:
    """
    Rational Quadratic Spline normalizing flow with custom power-law
    preprocessing (Flax NNX implementation).

    Maps between three spaces:

        physical space (original data)
            ↔  power-affine normalisation
        normalized bounded space [-B, B]
            ↔  stack of masked-coupling RQS layers
        latent Gaussian space N(0, I)

    Preprocessing, per feature:
        x_norm = (sign(x) * |x|^gamma - mean) / half_range * B,  gamma = p
    with mean / half_range taken from data_min / data_max if given, otherwise
    from the power-stretched training data.  Jacobian corrections for this
    map are included in ``log_prob`` and ``forward_pass_with_logdet``.

    The architecture follows the pyprobml spline-flow notebook
    (Murphy et al. 2021) that ``flows.py`` was adapted from.
    """

    def __init__(
        self,
        data,
        data_min = jnp.array([False]),
        data_max = jnp.array([False]),
        tset_paths = jnp.array([False]),
        flow_num_layers=4,
        hidden_size=128,
        mlp_num_layers=2,
        num_bins=8,
        learning_rate=1e-4,
        B=6.0,
        p=0.3,
        seed=0,
    ):
        """
        Parameters
        ----------
        data : array (N, D)
            Training dataset
        data_min, data_max : array (D,), optional
            Physical-space feature bounds for the normalizer.  If omitted,
            the bounds are taken from `data`.
        tset_paths : list of str, optional
            .npy training-set files for fit(mode='from_disk').
        flow_num_layers : int
            Number of coupling layers
        hidden_size : int
            Width of MLP hidden layers
        mlp_num_layers : int
            Number of hidden layers in conditioner networks
        num_bins : int
            Number of bins in spline transform
        learning_rate : float
            Adam optimizer learning rate
        B : float
            Bound for normalized space
        p : float
            Power transform parameter
        seed : int
            Random seed
        """
        self.tset_paths = tset_paths

        self.N, self.D = data.shape
        self.B = float(B)
        self.alpha = jnp.log(p)
        self.gamma = jnp.exp(self.alpha)

        # ------------------------------------------------------------
        # Power transform parameters
        # ------------------------------------------------------------
        if data_min.any() and data_max.any():
            min_x = data_min
            max_x = data_max
            dummy_min = np.sign(min_x) * np.abs(min_x) ** self.gamma
            dummy_max = np.sign(max_x) * np.abs(max_x) ** self.gamma
            self.mean_x      = (dummy_max + dummy_min) / 2.0
            self.half_range  = (dummy_max - dummy_min) / 2.0
        else:
            dummy = jnp.sign(data) * jnp.abs(data) ** self.gamma
            min_x = dummy.min(axis =0)
            max_x = dummy.max(axis = 0)
            self.mean_x = (max_x + min_x) / 2.0
            self.half_range = (max_x - min_x) / 2.0

        self._xn = (jnp.float64(self.alpha), jnp.asarray(self.mean_x, dtype=jnp.float64),
                    jnp.asarray(self.half_range, dtype=jnp.float64), jnp.float64(self.B))

        # ------------------------------------------------------------
        # Normalize data
        # ------------------------------------------------------------
        self.x_norm = self.to_unit_interval(data)

        # ------------------------------------------------------------
        # Flow config
        # ------------------------------------------------------------
        self.event_shape = (self.D,)
        self.flow_num_layers = flow_num_layers
        self.hidden_sizes = [hidden_size] * mlp_num_layers
        self.num_bins = num_bins
        self.learning_rate = learning_rate

        # ------------------------------------------------------------
        # RNG, model, optimizer
        # ------------------------------------------------------------
        self.key = jr.PRNGKey(seed)
        self.key, init_key = jr.split(self.key)
        self.model = _CouplingRQSFlow(
            self.D, 0, self.flow_num_layers, self.hidden_sizes,
            self.num_bins, self.B, rngs=nnx.Rngs(params=init_key),
        )
        self.optimizer = nnx.Optimizer(
            self.model, optax.adam(self.learning_rate), wrt=nnx.Param
        )

    @property
    def params(self):
        """Read-only view of the current parameters (an ``nnx.State``)."""
        return nnx.state(self.model, nnx.Param)

    # ================================================================
    # Data transforms
    # ================================================================
    @partial(jax.jit, static_argnums=0)
    def from_unit_interval(self, arr):
        """
        Transform normalized values back to physical space.

        Implements the inverse of the preprocessing transform.
        """
        return _to_coefficients_jnp(arr, *self._xn)

    @partial(jax.jit, static_argnums=0)
    def to_unit_interval(self, arr):
        """
        Transform physical data into normalized bounded space [-B, B].
        """
        return _from_coefficients_jnp(arr, *self._xn)

    @partial(jax.jit, static_argnums=0)
    def logdet_to_unit_interval(self, arr):
        """
        Log absolute determinant of the Jacobian of the transformation
        from physical space to normalized space.
        """
        return _logdet_from_coefficients_jnp(arr, *self._xn)

    @partial(jax.jit, static_argnums=0)
    def logdet_from_unit_interval(self, arr):
        """
        Log absolute determinant of the Jacobian of the transformation
        from normalized space back to physical space.
        """
        return _logdet_to_coefficients_jnp(arr, *self._xn)

    # ================================================================
    # Batching
    # ================================================================

    @staticmethod
    def get_batches(data, batch_size, key):
        """
        Generate shuffled mini-batches of data.
        """
        N = data.shape[0]
        idx = jax.random.permutation(key, N)
        for i in range(0, N, batch_size):
            yield data[idx[i:i + batch_size]]

    # ================================================================
    # Training
    # ================================================================

    def fit(
        self,
        mode = 'pre-loaded',
        num_epochs=100,
        batch_size=256,
        checkpoint=None,
    ):
        """
        Train the model using mini-batch gradient descent.

        mode : str
            'pre-loaded' : epochs over the normalized training data.
            'from_disk'  : each "epoch" is one step on a random batch from a
                           random file in tset_paths.
        num_epochs : int
        batch_size : int
        checkpoint : path or None
            Write training state here after every epoch if not None,
            and resume from last epoch recorded here if it already exists.
        """
        if mode == 'pre-loaded':
            # load from last checkpointed epoch
            start_epoch = 0
            if checkpoint is not None and os.path.exists(checkpoint):
                start_epoch = self._load_checkpoint(checkpoint, batch_size)
                print(f"[flow] resuming from {checkpoint} at epoch "
                        f"{start_epoch}/{num_epochs}", flush=True)
            self.resumed_from = start_epoch # needed for interpreting wall-clock timing on a resumed run

            model, optimizer = self.model, self.optimizer
            try:
                for ep in trange(start_epoch, num_epochs,
                                initial=start_epoch, total=num_epochs,
                                colour = 'blue', desc = "Training the flow from pre-loaded data"):

                    self.key, subkey = jr.split(self.key)

                    for batch in self.get_batches(self.x_norm, batch_size, subkey):
                        model, optimizer, _ = _train_step(model, optimizer, batch, None)

                    if checkpoint is not None:
                        _commit(self.model, self.optimizer, model, optimizer)
                        self._save_checkpoint(checkpoint, ep + 1, batch_size)
            finally:
                # Also on interrupt, so completed steps are not lost.
                _commit(self.model, self.optimizer, model, optimizer)

        if mode == 'from_disk':
            if checkpoint is not None:
                err = "Checkpointing is only supported for mode='pre-loaded'."
                raise ValueError(err)
            model, optimizer = self.model, self.optimizer
            try:
                for _ in trange(num_epochs, colour = 'blue', desc = "Training the flow from disk"):
                    rand_idx = random.randint(0, len(self.tset_paths) - 1)
                    training_set = jnp.load(self.tset_paths[rand_idx])
                    idxs = jnp.array(random.sample(range(training_set.shape[0]),
                                    k = min(batch_size, training_set.shape[0])))
                    batch =  self.to_unit_interval(training_set[idxs])
                    model, optimizer, _ = _train_step(model, optimizer, batch, None)
            finally:
                _commit(self.model, self.optimizer, model, optimizer)

    # ================================================================
    # Public API
    # ================================================================

    def log_prob(self, x):
        """
        Compute log-probability in physical space.
        """
        return _log_prob_phys(self.model, x, None, self._xn, None)

    def sample(self, num_samples):
        """
        Draw samples in physical space.
        """
        self.key, subkey = jr.split(self.key)
        z = jr.normal(subkey, (num_samples, self.D))
        return _forward_phys(self.model, z, None, self._xn, None)

    def forward_pass(self, z):
        """
        Map latent samples to physical space.
        """
        return _forward_phys(self.model, z, None, self._xn, None)

    def backward_pass(self, target_samp):
        """
        Map physical-space samples to latent space.
        """
        return _inverse_phys(self.model, target_samp, None, self._xn, None)

    def forward_pass_with_logdet(self, z):
        """
        Forward transform with full Jacobian correction.

        Returns (x, log|det J_{z→x}|), including the denormalisation Jacobian.
        """
        return _forward_phys_with_logdet(self.model, z, None, self._xn, None)

    def save_params(self, flow_save_path):
        """Write the parameters to an .npz keyed by module path."""
        _save_params(self.model, flow_save_path)

    def load_params(self, flow_load_path):
        """
        Load parameters written by ``save_params``.  Raises ValueError for
        files from a different architecture or from the Haiku ``flows.py``.
        """
        _load_params(self.model, flow_load_path)

    # ------------- Checkpointing -----------
    # To resume flow training if interrupted partway through

    def _checkpoint_signature(self, batch_size):
        # Configuration must match for a checkpoint to belong to this run.
        # Prevents resuming into a different state
        return dict(
            framework='flax-nnx',
            N=int(self.N), D=int(self.D),
            flow_num_layers=int(self.flow_num_layers),
            hidden_sizes=[int(h) for h in self.hidden_sizes],
            num_bins=int(self.num_bins),
            learning_rate=float(self.learning_rate),
            B=float(self.B), gamma=float(self.gamma),
            batch_size=int(batch_size),
        )

    def _save_checkpoint(self, path, epoch, batch_size):
        """Params, optimizer state, RNG key and epoch counter."""
        # The optimizer state (Adam moments and step count) is saved, not just
        # the parameters: resuming without the moments would restart them at
        # zero and put a visible transient in the loss.
        blob = dict(
            model=_to_numpy_pure(self.model),
            optimizer=_to_numpy_pure(self.optimizer),
            key=np.asarray(self.key),
            epoch=int(epoch),
            signature=self._checkpoint_signature(batch_size),
        )
        # Write to a temporary file and rename.  The atomic rename is the whole
        # reason this is safe to interrupt: a kill during the write leaves the
        # PREVIOUS checkpoint intact rather than a half-written file that cannot
        # be unpickled -- which would turn one lost epoch into all of them.
        tmp = f'{path}.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def _load_checkpoint(self, path, batch_size):
        """Restore training state; return the number of epochs already done."""
        with open(path, 'rb') as f:
            blob = pickle.load(f)

        want, got = self._checkpoint_signature(batch_size), blob['signature']
        if got != want:
            differ = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
            raise ValueError(
                f"{path} was written by a different configuration: "
                f"(checkpoint, this run) differ on {differ}.  Train into a "
                f"different directory, or delete the checkpoint to start over.")

        _restore_pure(self.model, blob['model'])
        _restore_pure(self.optimizer, blob['optimizer'])
        # Restoring the key is what makes a resume CONTINUE the RNG stream.
        # Without it the resumed epochs would replay the batch order of the
        # epochs already trained on.
        self.key = jnp.asarray(blob['key'])
        return int(blob['epoch'])


# ─────────────────────────────────────────────────────────────────────────────
# ConditionalFlow
# ─────────────────────────────────────────────────────────────────────────────

class ConditionalFlow:
    """
    Conditional Rational Quadratic Spline normalizing flow (Flax NNX).

    Models p(x | c) where c is a conditioning context vector.

    Both data and context are independently normalised through a
    power-affine map:

        physical → normalized:
            dummy = sign(x) * |x|^exp(alpha)
            norm  = (dummy - mean) / half_range * B

        normalized → physical:
            dummy = half_range/B * norm + mean
            x     = sign(dummy) * |dummy|^(1/exp(alpha))

    The user supplies the physical-space range [min_x, max_x] for both
    arrays, plus p and B, so no data statistics need to be computed from
    the full dataset — making the class fully compatible with memmapped
    inputs.

    Conditioning: every coupling layer's conditioner MLP receives
    [x_masked ‖ c_norm] as input.
    """

    def __init__(
        self,
        # ── data / context (memmapped or in-memory) ──────────────────────
        data,
        context,
        # ── normalisation ranges (per-feature 1-D arrays) ────────────────
        data_min = None,
        data_max = None,
        context_min = None,
        context_max = None,
        normalizer_dict = jnp.array([False]),
        tset_paths = None,
        last_feature_index = None,
        last_feature_index_for_context = None,
        flow_save_path = None,
        flow_load_path = None,
        # ── shared normalisation hyperparameters ─────────────────────────
        p=0.1,
        B=4.0,
        # ── flow architecture ─────────────────────────────────────────────
        flow_num_layers=4,
        hidden_size=128,
        mlp_num_layers=2,
        num_bins=8,
        # ── optimisation ─────────────────────────────────────────────────
        learning_rate=1e-4,
        seed=0,
    ):
        """
        Parameters
        ----------
        data : array-like (N, D)
            Target variables.  May be a numpy memmap — only batch-sized
            slices are ever read.
        context : array-like (N, C)
            Conditioning variables.  Same memory constraints as data.
        data_min, data_max : array-like (D,)
            Per-feature physical-space bounds of `data`.
        context_min, context_max : array-like (C,)
            Per-feature physical-space bounds of `context`.
        normalizer_dict : dict, optional
            Pre-fitted normalizers {'data': ..., 'context': ...}; if given,
            the min/max arguments are ignored.
        tset_paths : list of str, optional
            .npy training-set files for fit modes 'simple' and
            'broadcastable_context'.
        last_feature_index, last_feature_index_for_context : int, optional
            Column split of the training-set files: data is
            [:, :last_feature_index], context is
            [:, last_feature_index:last_feature_index_for_context].
        p : float
            Power parameter shared by both normalizers.  alpha = log(p).
        B : float
            Target half-width of the normalized space for both arrays.
        flow_num_layers : int
            Number of RQS masked-coupling layers.
        hidden_size : int
            Width of each conditioner MLP hidden layer.
        mlp_num_layers : int
            Number of hidden layers in each conditioner MLP.
        num_bins : int
            Number of spline bins per RQS layer.
        learning_rate : float
            Adam learning rate.
        seed : int
            PRNG seed.
        """
        self.N, self.D = data.shape[0], data.shape[1]
        self.C = context.shape[1]
        self.tset_paths = tset_paths
        self.last_feature_idx = last_feature_index
        self.last_feature_index_for_context = last_feature_index_for_context

        self.flow_save_path = flow_save_path
        self.flow_load_path = flow_load_path

        # Keep references to the raw arrays (may be memmapped)
        self._data    = data
        self._context = context

        # ── Normalizers ───────────────────────────────────────────────────
        # Fit from user-supplied ranges; never touches the full data array.
        if any(normalizer_dict):
            self.x_norm = normalizer_dict['data']
            self.c_norm = normalizer_dict['context']
        else:
            self.x_norm = _fit_normalizer(data_min,    data_max,    p, B)
            self.c_norm = _fit_normalizer(context_min, context_max, p, B)

        self._xn = _norm_tuple(self.x_norm)
        self._cn = _norm_tuple(self.c_norm)

        # ── Flow config ───────────────────────────────────────────────────
        self.event_shape    = (self.D,)
        self.flow_num_layers = flow_num_layers
        self.hidden_sizes    = [hidden_size] * mlp_num_layers
        self.num_bins        = num_bins
        self.learning_rate   = learning_rate

        self.key = jr.PRNGKey(seed)
        self.key, init_key = jr.split(self.key)
        self.model = _CouplingRQSFlow(
            self.D, self.C, self.flow_num_layers, self.hidden_sizes,
            self.num_bins, float(self.x_norm['B']), rngs=nnx.Rngs(params=init_key),
        )
        self.optimizer = nnx.Optimizer(
            self.model, optax.adam(self.learning_rate), wrt=nnx.Param
        )

    @property
    def params(self):
        """Read-only view of the current parameters (an ``nnx.State``)."""
        return nnx.state(self.model, nnx.Param)

    # ====================================================================
    # Normalisation
    # ====================================================================

    @partial(jax.jit, static_argnums=0)
    def normalize_data(self, x):
        """Physical data → normalized [-B, B]."""
        return _from_coefficients_jnp(x, *self._xn)

    @partial(jax.jit, static_argnums=0)
    def denormalize_data(self, x_norm):
        """Normalized → physical data."""
        return _to_coefficients_jnp(x_norm, *self._xn)

    @partial(jax.jit, static_argnums=0)
    def normalize_context(self, c):
        """Physical context → normalized [-B, B]."""
        return _from_coefficients_jnp(c, *self._cn)

    @partial(jax.jit, static_argnums=0)
    def denormalize_context(self, c_norm):
        """Normalized → physical context."""
        return _to_coefficients_jnp(c_norm, *self._cn)

    @partial(jax.jit, static_argnums=0)
    def _logdet_normalize_data(self, x):
        return _logdet_from_coefficients_jnp(x, *self._xn)

    @partial(jax.jit, static_argnums=0)
    def _logdet_denormalize_data(self, x_norm):
        return _logdet_to_coefficients_jnp(x_norm, *self._xn)

    # ====================================================================
    # Training
    # ====================================================================

    def fit(self, mode = 'pre-loaded', num_epochs=100, batch_size=256, context_array = None):
        """
        Train the flow.

        mode='pre-loaded' : epochs over self._data / self._context.  Batches
            are read lazily, normalised on the fly and discarded, so only
            `batch_size` rows are ever resident -- safe with arbitrarily
            large memmapped arrays.
        mode='simple' : each "epoch" is one step on a random batch from a
            random file in tset_paths, split into data / context columns.
        mode='broadcastable_context' : as 'simple', but file i is paired
            with the single context vector context_array[i], broadcast
            across the batch.
        """
        model, optimizer = self.model, self.optimizer
        try:
            if mode == 'pre-loaded':
                for _ in trange(num_epochs, colour = 'green', desc = "Training the flow the pre-loaded way"):
                    self.key, subkey = jr.split(self.key)
                    for x_batch, c_batch in _iter_batches(
                        self._data, self._context,
                        self.x_norm, self.c_norm,
                        batch_size, subkey,
                    ):
                        model, optimizer, _ = _train_step(model, optimizer, x_batch, c_batch)
            elif mode == 'simple':
                for _ in trange(num_epochs, colour = 'blue', desc = "Training the flow the simple way"):
                    rand_idx = random.randint(0, len(self.tset_paths) - 1)
                    training_set = jnp.load(self.tset_paths[rand_idx])
                    idxs = jnp.array(random.sample(range(training_set.shape[0]), k = min(batch_size, training_set.shape[0])))
                    training_set = training_set[idxs]
                    x_batch =  self.normalize_data(training_set[:, :self.last_feature_idx])
                    c_batch =  self.normalize_context(training_set[:, self.last_feature_idx:self.last_feature_index_for_context])
                    model, optimizer, _ = _train_step(model, optimizer, x_batch, c_batch)

            elif mode == 'broadcastable_context':
                for _ in trange(num_epochs, colour = 'blue', desc = "Training the flow the broadcastable_context way"):
                    rand_idx = random.randint(0, len(context_array) - 1)
                    training_set = jnp.load(self.tset_paths[rand_idx])
                    idxs = jnp.array(random.sample(range(training_set.shape[0]), k = min(batch_size, training_set.shape[0])))
                    training_set = training_set[idxs]
                    x_batch =  self.normalize_data(training_set[:, :self.last_feature_idx])
                    c_batch =  self.normalize_context(context_array[rand_idx:rand_idx+1])
                    c_batch = jnp.broadcast_to(c_batch, (x_batch.shape[0], c_batch.shape[-1]))
                    model, optimizer, _ = _train_step(model, optimizer, x_batch, c_batch)
        finally:
            # Also on interrupt, so completed steps are not lost.
            _commit(self.model, self.optimizer, model, optimizer)

    def live_fit(self, x_batch, c_batch):
        """
        One optimiser step on already-normalized (x_batch, c_batch).

        Each call writes the result back into self.model, which walks the
        module graph (~1 ms); for many steps in a row, fit() is cheaper.
        """
        model, optimizer, _ = _train_step(self.model, self.optimizer, x_batch, c_batch)
        _commit(self.model, self.optimizer, model, optimizer)

    # ====================================================================
    # Public API  (inputs/outputs always in physical space)
    # ====================================================================

    def log_prob(self, x, c):
        """
        log p(x | c) in physical space.

        Parameters
        ----------
        x : (batch, D)  — physical-space data
        c : (batch, C)  — physical-space context
        """
        return _log_prob_phys(self.model, x, c, self._xn, self._cn)

    def sample(self, c, num_samples):
        """
        Draw samples from p(x | c).

        Parameters
        ----------
        c : (C,), (1, C) or (num_samples, C) — physical-space context.
            A single context row is broadcast to all samples; otherwise
            row i conditions sample i.
        num_samples : int

        Returns
        -------
        samples : (num_samples, D)  in physical space.
        """
        c = jnp.atleast_2d(jnp.asarray(c, dtype=jnp.float64))
        if c.shape[0] == 1:
            c = jnp.broadcast_to(c, (num_samples, c.shape[1]))
        elif c.shape[0] != num_samples:
            raise ValueError(
                f"context has {c.shape[0]} rows; expected 1 or num_samples={num_samples}")

        self.key, subkey = jr.split(self.key)
        z = jr.normal(subkey, (num_samples, self.D))
        return _forward_phys(self.model, z, c, self._xn, self._cn)

    def forward_pass(self, z, c):
        """
        Map latent z ~ N(0, I) to physical space given context c.

        Parameters
        ----------
        z : (batch, D)
        c : (batch, C)  — physical-space context
        """
        return _forward_phys(self.model, z, c, self._xn, self._cn)

    def forward_pass_with_logdet(self, z, c):
        """
        Forward transform with full Jacobian log-determinant.

        Parameters
        ----------
        z : (batch, D)
        c : (batch, C)  — physical-space context

        Returns
        -------
        x           : (batch, D)   physical-space samples
        total_logdet: (batch,)     log |det J_total|
        """
        return _forward_phys_with_logdet(self.model, z, c, self._xn, self._cn)

    def save_params(self, flow_save_path):
        """Write the parameters to an .npz keyed by module path."""
        _save_params(self.model, flow_save_path)

    def load_params(self, flow_load_path):
        """
        Load parameters written by ``save_params``.  Raises ValueError for
        files from a different architecture or from the Haiku ``flows.py``.
        """
        _load_params(self.model, flow_load_path)
