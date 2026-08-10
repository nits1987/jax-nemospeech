"""Integrated Optax training step for the FastConformer RNN-T PoC."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Mapping

from flax import struct
from flax.core import FrozenDict, freeze
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import ParakeetConfig
from .model import ParakeetRNNT
from .rnnt_loss import rnnt_loss_from_joint, rnnt_loss_from_logits

Array = jax.Array
Batch = Mapping[str, Array]


@dataclass(frozen=True)
class OptimizerConfig:
    """PoC optimizer defaults; the pinned NeMo recipe remains the oracle."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_gradient_norm: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    epsilon: float = 1e-8


class RNNTTrainState(train_state.TrainState):
    """Full state required for deterministic continuation."""

    batch_stats: FrozenDict[str, Any] = struct.field(pytree_node=True)
    rng: Array = struct.field(pytree_node=True)
    data_cursor: Array = struct.field(pytree_node=True)


def build_optimizer(config: OptimizerConfig) -> optax.GradientTransformation:
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.max_gradient_norm <= 0:
        raise ValueError("max_gradient_norm must be positive")
    return optax.chain(
        optax.clip_by_global_norm(config.max_gradient_norm),
        optax.adamw(
            learning_rate=config.learning_rate,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.epsilon,
            weight_decay=config.weight_decay,
        ),
    )


def validate_batch(batch: Batch, config: ParakeetConfig) -> None:
    required = {"features", "feature_lengths", "tokens", "token_lengths"}
    missing = required.difference(batch)
    if missing:
        raise KeyError(f"batch is missing required fields: {sorted(missing)}")
    features = batch["features"]
    feature_lengths = batch["feature_lengths"]
    tokens = batch["tokens"]
    token_lengths = batch["token_lengths"]
    if features.ndim != 3 or features.shape[-1] != config.encoder.num_mel_bins:
        raise ValueError(
            "features must have shape [B,T,num_mel_bins]; got "
            f"{features.shape}, expected mel width {config.encoder.num_mel_bins}"
        )
    batch_size = features.shape[0]
    if feature_lengths.shape != (batch_size,):
        raise ValueError("feature_lengths must have shape [B]")
    if tokens.ndim != 2 or tokens.shape[0] != batch_size:
        raise ValueError("tokens must have shape [B,U]")
    if token_lengths.shape != (batch_size,):
        raise ValueError("token_lengths must have shape [B]")


def create_train_state(
    model: ParakeetRNNT,
    rng: Array,
    example_batch: Batch,
    *,
    optimizer: OptimizerConfig = OptimizerConfig(),
) -> RNNTTrainState:
    """Initialize every model collection and the optimizer state."""

    validate_batch(example_batch, model.config)
    params_rng, dropout_rng, state_rng = jax.random.split(rng, 3)
    variables = model.init(
        {"params": params_rng, "dropout": dropout_rng},
        example_batch["features"],
        example_batch["feature_lengths"],
        example_batch["tokens"],
        example_batch["token_lengths"],
        train=False,
        # Calling the joint during init ensures its parameters are present.
        compute_joint=True,
    )
    params = variables["params"]
    batch_stats = variables.get("batch_stats", freeze({}))
    return RNNTTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=build_optimizer(optimizer),
        batch_stats=batch_stats,
        rng=state_rng,
        data_cursor=jnp.asarray(0, dtype=jnp.int32),
    )


def _tree_all_finite(tree: Any) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))


def _tree_l2_norm(tree: Any) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    squared = [jnp.sum(jnp.square(leaf.astype(jnp.float32))) for leaf in leaves]
    return jnp.sqrt(jnp.sum(jnp.stack(squared)))


def _parameter_count(params: Any) -> int:
    return sum(int(np.prod(leaf.shape)) for leaf in jax.tree_util.tree_leaves(params))


def parameter_count(params: Any) -> int:
    """Return the number of scalar trainable parameters."""

    return _parameter_count(params)


def _joint_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Select only parameters touched by ``ParakeetRNNT.joint``.

    Passing the entire 1.07B parameter tree as a scan constant is unnecessary
    and can inflate the compiled streamed-loss program. Gradients through this
    selected mapping still flow back to the corresponding leaves of ``params``.
    """

    names = ("encoder_projector", "joint_head")
    return {name: params[name] for name in names}


def _loss_with_variables(
    params: Any,
    state: RNNTTrainState,
    batch: Batch,
    *,
    model: ParakeetRNNT,
    dropout_rng: Array,
    materialize_joint: bool,
) -> tuple[Array, tuple[FrozenDict[str, Any], dict[str, Array]]]:
    variables = {"params": params, "batch_stats": state.batch_stats}
    output, mutated = state.apply_fn(
        variables,
        batch["features"],
        batch["feature_lengths"],
        batch["tokens"],
        batch["token_lengths"],
        train=True,
        compute_joint=materialize_joint,
        rngs={"dropout": dropout_rng},
        mutable=["batch_stats"],
    )
    if materialize_joint:
        loss = rnnt_loss_from_logits(
            output["joint_logits"],
            batch["tokens"],
            output["encoder_lengths"],
            batch["token_lengths"],
            blank_id=model.config.blank_id,
            reduction="mean_batch",
        )
    else:

        def joint_fn(joint_params: Any, encoder_frame: Array, predictor: Array) -> Array:
            return state.apply_fn(
                {"params": joint_params},
                encoder_frame,
                predictor,
                method=model.joint,
            )

        loss = rnnt_loss_from_joint(
            output["encoder"],
            output["predictor"],
            joint_fn,
            batch["tokens"],
            output["encoder_lengths"],
            batch["token_lengths"],
            blank_id=model.config.blank_id,
            reduction="mean_batch",
            joint_params=_joint_params(params),
        )
    auxiliary = {
        "mean_encoder_frames": jnp.mean(output["encoder_lengths"].astype(jnp.float32)),
        "mean_target_tokens": jnp.mean(batch["token_lengths"].astype(jnp.float32)),
    }
    return loss, (mutated.get("batch_stats", state.batch_stats), auxiliary)


def train_step(
    state: RNNTTrainState,
    batch: Batch,
    *,
    model: ParakeetRNNT,
    materialize_joint: bool = False,
) -> tuple[RNNTTrainState, dict[str, Array]]:
    """Run encoder, predictor, streamed joint/loss, backward, and optimizer."""

    dropout_rng, next_rng = jax.random.split(state.rng)
    (loss, (batch_stats, auxiliary)), gradients = jax.value_and_grad(
        _loss_with_variables, has_aux=True
    )(
        state.params,
        state,
        batch,
        model=model,
        dropout_rng=dropout_rng,
        materialize_joint=materialize_joint,
    )
    updates, _ = state.tx.update(gradients, state.opt_state, state.params)
    metrics = {
        "loss": loss,
        "gradient_norm": _tree_l2_norm(gradients),
        "update_norm": _tree_l2_norm(updates),
        "parameters_finite": _tree_all_finite(state.params),
        "gradients_finite": _tree_all_finite(gradients),
        **auxiliary,
    }
    state = state.apply_gradients(
        grads=gradients,
        batch_stats=batch_stats,
        rng=next_rng,
        data_cursor=state.data_cursor + batch["features"].shape[0],
    )
    return state, metrics


def make_train_step(
    model: ParakeetRNNT,
    *,
    materialize_joint: bool = False,
):
    """Return a JIT-compiled train step with static model/loss policy."""

    return jax.jit(
        partial(train_step, model=model, materialize_joint=materialize_joint)
    )


def eval_loss(
    state: RNNTTrainState,
    batch: Batch,
    *,
    model: ParakeetRNNT,
    materialize_joint: bool = False,
) -> Array:
    """Compute deterministic loss without updating BatchNorm state."""

    output = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        batch["features"],
        batch["feature_lengths"],
        batch["tokens"],
        batch["token_lengths"],
        train=False,
        compute_joint=materialize_joint,
    )
    if materialize_joint:
        return rnnt_loss_from_logits(
            output["joint_logits"],
            batch["tokens"],
            output["encoder_lengths"],
            batch["token_lengths"],
            blank_id=model.config.blank_id,
            reduction="mean_batch",
        )

    def joint_fn(joint_params: Any, encoder_frame: Array, predictor: Array) -> Array:
        return state.apply_fn(
            {"params": joint_params}, encoder_frame, predictor, method=model.joint
        )

    return rnnt_loss_from_joint(
        output["encoder"],
        output["predictor"],
        joint_fn,
        batch["tokens"],
        output["encoder_lengths"],
        batch["token_lengths"],
        blank_id=model.config.blank_id,
        reduction="mean_batch",
        joint_params=_joint_params(state.params),
    )


def make_synthetic_batch(
    config: ParakeetConfig,
    *,
    batch_size: int = 1,
    feature_frames: int = 24,
    target_steps: int = 3,
    seed: int = 0,
) -> dict[str, Array]:
    """Create one static, deterministic bucket for smoke/fitment tests."""

    if batch_size <= 0 or feature_frames < config.encoder.subsampling_factor:
        raise ValueError("batch_size must be positive and feature_frames must survive subsampling")
    if target_steps < 0:
        raise ValueError("target_steps must be non-negative")
    generator = np.random.default_rng(seed)
    features = generator.normal(
        size=(batch_size, feature_frames, config.encoder.num_mel_bins)
    ).astype(np.float32)
    token_ids = np.asarray(
        [index for index in range(config.vocab_size) if index != config.blank_id],
        dtype=np.int32,
    )
    tokens = generator.choice(token_ids, size=(batch_size, target_steps)).astype(np.int32)
    feature_lengths = np.full((batch_size,), feature_frames, dtype=np.int32)
    token_lengths = np.full((batch_size,), target_steps, dtype=np.int32)
    if batch_size > 1:
        feature_lengths[-1] = max(config.encoder.subsampling_factor, feature_frames - 3)
        token_lengths[-1] = max(0, target_steps - 1)
        tokens[-1, token_lengths[-1] :] = config.pad_id
    return {
        "features": jnp.asarray(features),
        "feature_lengths": jnp.asarray(feature_lengths),
        "tokens": jnp.asarray(tokens),
        "token_lengths": jnp.asarray(token_lengths),
    }


__all__ = [
    "OptimizerConfig",
    "RNNTTrainState",
    "create_train_state",
    "eval_loss",
    "make_synthetic_batch",
    "make_train_step",
    "parameter_count",
    "train_step",
    "validate_batch",
]
