# Copyright 2026 fcrnnt-jax contributors
# Portions Copyright © 2023 Apple Inc.
# Adapted and modified from Apple AXLearn's Apache-2.0 transducer code.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Classic RNN-T loss with an accelerator-friendly JAX implementation.

The diagonal alignment dynamic program and its analytic VJP are adapted from
``axlearn.common.transducer`` in Apple AXLearn (Copyright 2023 Apple Inc.,
Apache-2.0), with substantial changes for a standalone, batched, length-based
API and an FP32-only numerical path. See ``THIRD_PARTY_NOTICES.md``.

There are two public entry points:

* :func:`rnnt_loss_from_logits` accepts a complete ``[B, T, U+1, V]`` tensor.
  It is intended for small tests and framework-parity fixtures.
* :func:`rnnt_loss_from_joint` evaluates a supplied joint network one acoustic
  frame at a time. It retains only blank and ground-truth-token edge log
  probabilities, ``O(B*T*U)``, rather than complete ``O(B*T*U*V)`` logits.

Both functions implement *classic* RNN-T: blank and non-blank symbols share a
single vocabulary softmax. This is deliberately not the HAT factorization.
The final transition is a blank from ``(T-1, U)`` to the terminal state.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp


Array = jax.Array
Reduction = Literal["none", "sum", "mean_batch", "mean", "mean_volume"]

_VALID_REDUCTIONS = ("none", "sum", "mean_batch", "mean", "mean_volume")
_NEG_INF = -1.0e30


def _static_blank_id(blank_id: int, vocab_size: int) -> int:
    """Returns a Python blank id, producing a useful error for dynamic ids."""
    try:
        value = operator.index(blank_id)
    except TypeError as exc:
        raise TypeError(
            "blank_id must be a static Python integer; close over it when using "
            "jax.jit, or mark it as a static argument"
        ) from exc
    if not 0 <= value < vocab_size:
        raise ValueError(f"blank_id={value} is outside vocabulary [0, {vocab_size})")
    return value


def _validate_reduction(reduction: str) -> None:
    if reduction not in _VALID_REDUCTIONS:
        choices = ", ".join(repr(value) for value in _VALID_REDUCTIONS)
        raise ValueError(f"invalid reduction {reduction!r}; expected one of {choices}")


def _validate_integer_array(name: str, value: Array, ndim: int) -> None:
    if value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {value.shape}")
    if not jnp.issubdtype(value.dtype, jnp.integer):
        raise TypeError(f"{name} must have an integer dtype, got {value.dtype}")


def _validate_batch_inputs(
    *,
    labels: Array,
    logit_lengths: Array,
    label_lengths: Array,
    batch_size: int,
    max_time: int,
    max_labels: int,
) -> None:
    _validate_integer_array("labels", labels, 2)
    _validate_integer_array("logit_lengths", logit_lengths, 1)
    _validate_integer_array("label_lengths", label_lengths, 1)
    if labels.shape != (batch_size, max_labels):
        raise ValueError(
            "labels shape must match the batch and U dimensions: "
            f"expected {(batch_size, max_labels)}, got {labels.shape}"
        )
    if logit_lengths.shape != (batch_size,):
        raise ValueError(
            f"logit_lengths must have shape {(batch_size,)}, got {logit_lengths.shape}"
        )
    if label_lengths.shape != (batch_size,):
        raise ValueError(
            f"label_lengths must have shape {(batch_size,)}, got {label_lengths.shape}"
        )
    if batch_size == 0:
        raise ValueError("empty batches are not supported")
    if max_time == 0:
        raise ValueError("the static acoustic time dimension must be at least one")


def _tilt(x: Array, pad_value: float) -> Array:
    """Tilts ``[R, C]`` so that lattice diagonals become rows.

    Adapted and modified from Apple AXLearn's transducer implementation.
    ``C`` must be positive.
    """
    rows, columns = x.shape
    x = jnp.pad(x, ((0, columns), (0, 0)), constant_values=pad_value)
    x = x.T.reshape(-1)
    x = x[:-columns]
    return x.reshape((columns, rows + columns - 1)).T


def _untilt(y: Array) -> Array:
    """Inverse of :func:`_tilt` for a positive column count."""
    tilted_rows, columns = y.shape
    rows = tilted_rows + 1 - columns
    y = jnp.pad(y.T.reshape(-1), (0, columns))
    return y.reshape((columns, rows + columns))[:, :rows].T


def _prefix_log_probs(log_prob_blank: Array, log_prob_label: Array) -> Array:
    """Computes forward alignment log probabilities ``alpha[t, u]``.

    ``log_prob_blank`` has shape ``[T, U+1]`` and ``log_prob_label`` has
    shape ``[T+1, U]``. The computation proceeds a diagonal at a time, as in
    Bagby et al., *Efficient Implementation of Recurrent Neural Network
    Transducer in TensorFlow* (2018).
    """
    max_labels = log_prob_label.shape[1]

    # The empty-transcript lattice has only vertical blank transitions. This
    # explicit branch also avoids applying _tilt to a zero-column array.
    if max_labels == 0:
        initial = jnp.zeros((1, 1), dtype=jnp.float32)
        return jnp.concatenate(
            [initial, jnp.cumsum(log_prob_blank[:, :1], axis=0)], axis=0
        )

    tilted_blank = _tilt(log_prob_blank, _NEG_INF)
    tilted_label = _tilt(log_prob_label, _NEG_INF)
    initial = jnp.full((max_labels + 1,), _NEG_INF, dtype=jnp.float32)
    initial = initial.at[0].set(0.0)

    def next_diagonal(carry: Array, edges: tuple[Array, Array]) -> tuple[Array, Array]:
        blank, label = edges
        via_blank = blank + carry
        via_label = jnp.pad(
            label + carry[:-1], (1, 0), constant_values=-jnp.inf
        )
        result = jnp.logaddexp(via_blank, via_label)
        return result, result

    _, diagonals = jax.lax.scan(
        next_diagonal, initial, (tilted_blank, tilted_label)
    )
    tilted_prefix = jnp.concatenate([initial[None, :], diagonals], axis=0)
    return _untilt(tilted_prefix)


def _suffix_log_probs(log_prob_blank: Array, log_prob_label: Array) -> Array:
    """Computes backward alignment log probabilities ``beta[t, u]``."""
    reversed_prefix = _prefix_log_probs(
        log_prob_blank[::-1, ::-1], log_prob_label[::-1, ::-1]
    )
    return reversed_prefix[::-1, ::-1]


@jax.custom_vjp
def _alignment_log_prob(log_prob_blank: Array, log_prob_label: Array) -> Array:
    """Returns log P(labels | acoustics) for one fixed-size padded lattice."""
    return _prefix_log_probs(log_prob_blank, log_prob_label)[-1, -1]


def _alignment_log_prob_fwd(
    log_prob_blank: Array, log_prob_label: Array
) -> tuple[Array, tuple[Array, Array, Array]]:
    prefix = _prefix_log_probs(log_prob_blank, log_prob_label)
    return prefix[-1, -1], (log_prob_blank, log_prob_label, prefix)


def _alignment_log_prob_bwd(
    residual: tuple[Array, Array, Array], cotangent: Array
) -> tuple[Array, Array]:
    log_prob_blank, log_prob_label, prefix = residual
    suffix = _suffix_log_probs(log_prob_blank, log_prob_label)
    total = suffix[0, 0]

    # Edge posterior probabilities are the derivatives of log P with respect
    # to edge log probabilities. Computing them analytically avoids asking
    # autodiff to retain every diagonal scan intermediate.
    blank_log_posterior = (
        prefix[:-1, :] + log_prob_blank + suffix[1:, :] - total
    )
    label_log_posterior = (
        prefix[:, :-1] + log_prob_label + suffix[:, 1:] - total
    )
    grad_blank = jnp.exp(blank_log_posterior)
    grad_label = jnp.exp(label_log_posterior)
    return cotangent * grad_blank, cotangent * grad_label


_alignment_log_prob.defvjp(_alignment_log_prob_fwd, _alignment_log_prob_bwd)


def _pad_one_lattice(
    log_prob_blank: Array,
    log_prob_label: Array,
    logit_length: Array,
    label_length: Array,
) -> tuple[Array, Array]:
    """Embeds a variable-size lattice in a fixed-size equivalent lattice.

    Real edges lead to ``(T, U)``. Zero-cost synthetic edges then lead down to
    ``(Tmax, U)`` and right to ``(Tmax, Umax)``. All other edges are disabled.
    This construction lets every batch item use the same custom-VJP program.
    """
    max_time, max_u_plus_one = log_prob_blank.shape
    max_labels = max_u_plus_one - 1

    time = jnp.arange(max_time, dtype=jnp.int32)[:, None]
    output = jnp.arange(max_u_plus_one, dtype=jnp.int32)[None, :]

    before_last_frame = (time < logit_length - 1) & (output <= label_length)
    terminal_blank = (time == logit_length - 1) & (output == label_length)
    real_blank = before_last_frame | terminal_blank
    synthetic_blank = (time >= logit_length) & (output == label_length)
    padded_blank = jnp.where(
        real_blank,
        log_prob_blank,
        jnp.where(synthetic_blank, 0.0, _NEG_INF),
    )

    padded_label_source = jnp.pad(
        log_prob_label, ((0, 1), (0, 0)), constant_values=_NEG_INF
    )
    label_time = jnp.arange(max_time + 1, dtype=jnp.int32)[:, None]
    label_output = jnp.arange(max_labels, dtype=jnp.int32)[None, :]
    real_label = (label_time < logit_length) & (label_output < label_length)
    synthetic_label = (label_time == max_time) & (label_output >= label_length)
    padded_label = jnp.where(
        real_label,
        padded_label_source,
        jnp.where(synthetic_label, 0.0, _NEG_INF),
    )
    return padded_blank, padded_label


def _reduce_losses(
    losses: Array, label_lengths: Array, reduction: Reduction
) -> Array:
    """Applies NeMo-compatible reduction formulas.

    Empty transcripts use a denominator of one for ``mean`` and
    ``mean_volume``. NeMo training batches normally contain non-empty targets;
    the clamp makes the mathematically valid U=0 edge case finite.
    """
    _validate_reduction(reduction)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean_batch":
        return losses.mean()
    lengths = label_lengths.astype(jnp.float32)
    if reduction == "mean":
        return (losses / jnp.maximum(lengths, 1.0)).mean()
    return losses.sum() / jnp.maximum(lengths.sum(), 1.0)


def _loss_from_edge_log_probs(
    log_prob_blank: Array,
    log_prob_label: Array,
    logit_lengths: Array,
    label_lengths: Array,
    reduction: Reduction,
) -> Array:
    """Computes a reduced loss from blank and target-label edge scores."""
    # Loss DP is intentionally FP32 even when the model/joint uses BF16.
    log_prob_blank = log_prob_blank.astype(jnp.float32)
    log_prob_label = log_prob_label.astype(jnp.float32)
    _, max_time, max_u_plus_one = log_prob_blank.shape
    max_labels = max_u_plus_one - 1

    valid_lengths = (
        (logit_lengths >= 1)
        & (logit_lengths <= max_time)
        & (label_lengths >= 0)
        & (label_lengths <= max_labels)
    )
    # Clip before building the lattice so invalid dynamic values cannot produce
    # out-of-bounds or structurally impossible programs under jax.jit. A NaN
    # sentinel below reports invalid examples without a host callback.
    safe_logit_lengths = jnp.clip(logit_lengths, 1, max_time).astype(jnp.int32)
    safe_label_lengths = jnp.clip(label_lengths, 0, max_labels).astype(jnp.int32)

    padded_blank, padded_label = jax.vmap(_pad_one_lattice)(
        log_prob_blank,
        log_prob_label,
        safe_logit_lengths,
        safe_label_lengths,
    )
    log_likelihood = jax.vmap(_alignment_log_prob)(padded_blank, padded_label)
    losses = jnp.where(valid_lengths, -log_likelihood, jnp.nan)
    return _reduce_losses(losses, safe_label_lengths, reduction)


def _target_edge_log_probs(log_probs: Array, labels: Array, blank_id: int) -> Array:
    """Selects ground-truth label edges and marks invalid active ids as NaN."""
    batch_size, max_time, max_u_plus_one, vocab_size = log_probs.shape
    max_labels = max_u_plus_one - 1
    if max_labels == 0:
        return jnp.empty((batch_size, max_time, 0), dtype=jnp.float32)

    target_is_valid = (
        (labels >= 0) & (labels < vocab_size) & (labels != blank_id)
    )
    safe_labels = jnp.clip(labels, 0, vocab_size - 1)
    indices = jnp.broadcast_to(
        safe_labels[:, None, :, None], (batch_size, max_time, max_labels, 1)
    )
    selected = jnp.take_along_axis(
        log_probs[:, :, :-1, :], indices, axis=-1
    )[..., 0]
    return jnp.where(target_is_valid[:, None, :], selected, jnp.nan)


def rnnt_loss_from_logits(
    logits: Array,
    labels: Array,
    logit_lengths: Array,
    label_lengths: Array,
    *,
    blank_id: int,
    reduction: Reduction = "mean_batch",
) -> Array:
    """Computes classic RNN-T loss from complete joint logits.

    Args:
        logits: Floating tensor ``[B, T, U+1, V]``. This API materializes the
            complete vocabulary lattice and is therefore intended for parity
            tests and small examples, not Parakeet-scale training.
        labels: Integer target ids ``[B, U]``. Values beyond each corresponding
            ``label_length`` are ignored and may contain padding sentinels.
        logit_lengths: Number of valid acoustic frames per example, ``[B]``.
            Each value must be in ``[1, T]``.
        label_lengths: Number of valid target labels per example, ``[B]``.
            Each value must be in ``[0, U]``.
        blank_id: Static vocabulary id for the classic RNN-T blank symbol.
        reduction: ``none``, ``sum``, ``mean_batch``, ``mean``, or
            ``mean_volume``. The formulas match NeMo's RNN-T wrapper.

    Returns:
        FP32 per-example negative log likelihoods for ``none``; otherwise an
        FP32 scalar. Invalid dynamic lengths or active target ids produce NaN.

    Note:
        When applying :func:`jax.jit` directly to this function, mark
        ``blank_id`` and ``reduction`` static. Closing over them in a jitted
        training-step function is usually simpler.
    """
    logits = jnp.asarray(logits)
    labels = jnp.asarray(labels)
    logit_lengths = jnp.asarray(logit_lengths)
    label_lengths = jnp.asarray(label_lengths)
    if logits.ndim != 4:
        raise ValueError(f"logits must have rank 4 [B,T,U+1,V], got {logits.shape}")
    if not jnp.issubdtype(logits.dtype, jnp.floating):
        raise TypeError(f"logits must have a floating dtype, got {logits.dtype}")

    batch_size, max_time, max_u_plus_one, vocab_size = logits.shape
    if max_u_plus_one == 0:
        raise ValueError("logits U+1 dimension must be at least one")
    if vocab_size < 2:
        raise ValueError("RNN-T vocabulary must contain at least two symbols")
    blank_id = _static_blank_id(blank_id, vocab_size)
    _validate_reduction(reduction)
    _validate_batch_inputs(
        labels=labels,
        logit_lengths=logit_lengths,
        label_lengths=label_lengths,
        batch_size=batch_size,
        max_time=max_time,
        max_labels=max_u_plus_one - 1,
    )

    # Softmax and every subsequent loss operation are explicitly FP32.
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    log_prob_blank = log_probs[..., blank_id]
    log_prob_label = _target_edge_log_probs(log_probs, labels, blank_id)
    return _loss_from_edge_log_probs(
        log_prob_blank,
        log_prob_label,
        logit_lengths,
        label_lengths,
        reduction,
    )


def rnnt_loss_from_joint(
    encoder_outputs: Array,
    predictor_outputs: Array,
    joint_fn: Callable[..., Array],
    labels: Array,
    logit_lengths: Array,
    label_lengths: Array,
    *,
    blank_id: int,
    reduction: Reduction = "mean_batch",
    joint_params: object | None = None,
) -> Array:
    """Computes classic RNN-T loss while streaming the joint over time.

    Args:
        encoder_outputs: Acoustic representations ``[B, T, ...]``.
        predictor_outputs: Predictor representations ``[B, U+1, ...]``.
        joint_fn: With no ``joint_params``, a callable receiving one acoustic
            frame ``[B, ...]`` and all predictor positions ``[B, U+1, ...]``.
            When ``joint_params`` is supplied, its signature is
            ``joint_fn(joint_params, encoder_frame, predictor_outputs)``. It
            must return floating logits ``[B, U+1, V]`` and is rematerialized
            during backward.
        labels: Integer target ids ``[B, U]``.
        logit_lengths: Valid acoustic lengths ``[B]``, in ``[1, T]``.
        label_lengths: Valid target lengths ``[B]``, in ``[0, U]``.
        blank_id: Static vocabulary id for the classic RNN-T blank symbol.
        reduction: See :func:`rnnt_loss_from_logits`.
        joint_params: Optional differentiable parameter pytree passed explicitly
            to ``joint_fn``. Prefer this form in training code so joint-network
            parameter ownership is visible at the loss call site. Closing over
            traced parameters also differentiates correctly, but is less clear.

    Returns:
        FP32 loss, shaped according to ``reduction``.

    Memory contract:
        Only one ``[B, U+1, V]`` joint frame is live at a time. The persistent
        edge tensors are ``[B, T, U+1]`` and ``[B, T, U]``. ``joint_fn`` must
        itself avoid broadcasting the acoustic input over the entire T axis.
    """
    encoder_outputs = jnp.asarray(encoder_outputs)
    predictor_outputs = jnp.asarray(predictor_outputs)
    labels = jnp.asarray(labels)
    logit_lengths = jnp.asarray(logit_lengths)
    label_lengths = jnp.asarray(label_lengths)
    if encoder_outputs.ndim < 2:
        raise ValueError(
            "encoder_outputs must have shape [B,T,...], got "
            f"{encoder_outputs.shape}"
        )
    if predictor_outputs.ndim < 2:
        raise ValueError(
            "predictor_outputs must have shape [B,U+1,...], got "
            f"{predictor_outputs.shape}"
        )
    if not callable(joint_fn):
        raise TypeError("joint_fn must be callable")

    batch_size, max_time = encoder_outputs.shape[:2]
    if predictor_outputs.shape[0] != batch_size:
        raise ValueError("encoder_outputs and predictor_outputs batch sizes differ")
    max_u_plus_one = predictor_outputs.shape[1]
    if max_u_plus_one == 0:
        raise ValueError("predictor_outputs U+1 dimension must be at least one")
    _validate_reduction(reduction)
    _validate_batch_inputs(
        labels=labels,
        logit_lengths=logit_lengths,
        label_lengths=label_lengths,
        batch_size=batch_size,
        max_time=max_time,
        max_labels=max_u_plus_one - 1,
    )

    # Validate static-ness now; the vocabulary range is checked when the joint
    # output shape is available while tracing the scan body.
    try:
        blank_id = operator.index(blank_id)
    except TypeError as exc:
        raise TypeError(
            "blank_id must be a static Python integer; close over it when using "
            "jax.jit, or mark it as a static argument"
        ) from exc

    def one_frame_edges(
        carry: None, encoder_frame: Array
    ) -> tuple[None, tuple[Array, Array]]:
        if joint_params is None:
            frame_logits = joint_fn(encoder_frame, predictor_outputs)
        else:
            frame_logits = joint_fn(
                joint_params, encoder_frame, predictor_outputs
            )
        frame_logits = jnp.asarray(frame_logits)
        expected_prefix = (batch_size, max_u_plus_one)
        if frame_logits.ndim != 3 or frame_logits.shape[:2] != expected_prefix:
            raise ValueError(
                "joint_fn must return [B,U+1,V]; expected prefix "
                f"{expected_prefix}, got {frame_logits.shape}"
            )
        if not jnp.issubdtype(frame_logits.dtype, jnp.floating):
            raise TypeError(
                f"joint_fn must return floating logits, got {frame_logits.dtype}"
            )
        vocab_size = frame_logits.shape[-1]
        if vocab_size < 2:
            raise ValueError("RNN-T vocabulary must contain at least two symbols")
        checked_blank_id = _static_blank_id(blank_id, vocab_size)

        frame_log_probs = jax.nn.log_softmax(
            frame_logits.astype(jnp.float32), axis=-1
        )
        blank = frame_log_probs[..., checked_blank_id]
        max_labels = max_u_plus_one - 1
        if max_labels == 0:
            target = jnp.empty((batch_size, 0), dtype=jnp.float32)
        else:
            target_is_valid = (
                (labels >= 0)
                & (labels < vocab_size)
                & (labels != checked_blank_id)
            )
            safe_labels = jnp.clip(labels, 0, vocab_size - 1)
            target = jnp.take_along_axis(
                frame_log_probs[:, :-1, :], safe_labels[..., None], axis=-1
            )[..., 0]
            target = jnp.where(target_is_valid, target, jnp.nan)
        return None, (blank, target)

    # Rematerializing the per-frame joint in the scan backward pass prevents
    # complete vocabulary logits from becoming persistent T-axis activations.
    scan_body = jax.checkpoint(one_frame_edges)
    _, (blank_time_major, label_time_major) = jax.lax.scan(
        scan_body, None, jnp.swapaxes(encoder_outputs, 0, 1)
    )
    log_prob_blank = jnp.swapaxes(blank_time_major, 0, 1)
    log_prob_label = jnp.swapaxes(label_time_major, 0, 1)
    return _loss_from_edge_log_probs(
        log_prob_blank,
        log_prob_label,
        logit_lengths,
        label_lengths,
        reduction,
    )


__all__ = ["rnnt_loss_from_joint", "rnnt_loss_from_logits"]
