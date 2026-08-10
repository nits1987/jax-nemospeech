"""Deliberately naive NumPy reference for the classic RNN-T loss.

This module exists so that :mod:`fcrnnt_jax.rnnt_loss` is never its own oracle.
It is written for auditability rather than speed: explicit Python loops over the
lattice, FP64 throughout, and no shared code with the JAX implementation. It is
the reference used by the CPU correctness matrix and by the NeMo fixture
harness when a fixture ships values but no gradients.

Conventions match :mod:`fcrnnt_jax.rnnt_loss` exactly:

* ``alpha[t, u]`` is the log probability of reaching lattice node ``(t, u)``
  after consuming ``t`` acoustic frames and emitting ``u`` labels.
* A blank at ``(t, u)`` advances to ``(t + 1, u)``; label ``labels[u]`` advances
  to ``(t, u + 1)``.
* The total log likelihood is ``alpha[T, U]``, so the final transition is a
  blank taken from ``(T - 1, U)``.
* Blank and labels share one vocabulary softmax (classic RNN-T, not HAT).

Nothing here is JAX-aware; every function takes and returns NumPy arrays.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


Reduction = Literal["none", "sum", "mean_batch", "mean", "mean_volume"]

_VALID_REDUCTIONS = ("none", "sum", "mean_batch", "mean", "mean_volume")


def _validate_reduction(reduction: str) -> None:
    if reduction not in _VALID_REDUCTIONS:
        choices = ", ".join(repr(value) for value in _VALID_REDUCTIONS)
        raise ValueError(f"invalid reduction {reduction!r}; expected one of {choices}")


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Return an FP64 log-softmax over the trailing axis."""

    values = np.asarray(logits, dtype=np.float64)
    maximum = np.max(values, axis=-1, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _validate_example(
    log_probs: np.ndarray,
    labels: np.ndarray,
    logit_length: int,
    label_length: int,
    blank_id: int,
) -> None:
    max_time, max_u_plus_one, vocab_size = log_probs.shape
    if not 1 <= logit_length <= max_time:
        raise ValueError(
            f"logit_length={logit_length} must be in [1, {max_time}]"
        )
    if not 0 <= label_length <= max_u_plus_one - 1:
        raise ValueError(
            f"label_length={label_length} must be in [0, {max_u_plus_one - 1}]"
        )
    if not 0 <= blank_id < vocab_size:
        raise ValueError(f"blank_id={blank_id} is outside vocabulary [0, {vocab_size})")
    active = labels[:label_length]
    if active.size:
        if int(active.min()) < 0 or int(active.max()) >= vocab_size:
            raise ValueError("active target ids must lie inside the vocabulary")
        if int((active == blank_id).sum()):
            raise ValueError("active target ids must not contain the blank id")


def _alpha(
    log_probs: np.ndarray,
    labels: np.ndarray,
    logit_length: int,
    label_length: int,
    blank_id: int,
) -> np.ndarray:
    alpha = np.full((logit_length + 1, label_length + 1), -np.inf, dtype=np.float64)
    alpha[0, 0] = 0.0
    for time in range(logit_length):
        for output in range(label_length + 1):
            score = alpha[time, output]
            if score == -np.inf:
                continue
            alpha[time + 1, output] = np.logaddexp(
                alpha[time + 1, output], score + log_probs[time, output, blank_id]
            )
            if output < label_length:
                token = int(labels[output])
                alpha[time, output + 1] = np.logaddexp(
                    alpha[time, output + 1], score + log_probs[time, output, token]
                )
    return alpha


def _beta(
    log_probs: np.ndarray,
    labels: np.ndarray,
    logit_length: int,
    label_length: int,
    blank_id: int,
) -> np.ndarray:
    """Suffix scores ``beta[t, u]`` for completing the lattice from ``(t, u)``."""

    beta = np.full((logit_length + 1, label_length + 1), -np.inf, dtype=np.float64)
    beta[logit_length, label_length] = 0.0
    for time in range(logit_length, -1, -1):
        for output in range(label_length, -1, -1):
            if time == logit_length and output == label_length:
                continue
            score = -np.inf
            if time < logit_length:
                score = np.logaddexp(
                    score, log_probs[time, output, blank_id] + beta[time + 1, output]
                )
                if output < label_length:
                    token = int(labels[output])
                    score = np.logaddexp(
                        score, log_probs[time, output, token] + beta[time, output + 1]
                    )
            beta[time, output] = score
    return beta


def example_loss(
    logits: np.ndarray,
    labels: np.ndarray,
    logit_length: int,
    label_length: int,
    blank_id: int,
) -> float:
    """Return the negative log likelihood for one ``[T, U+1, V]`` lattice."""

    log_probs = log_softmax(logits)
    labels = np.asarray(labels)
    _validate_example(log_probs, labels, logit_length, label_length, blank_id)
    alpha = _alpha(log_probs, labels, logit_length, label_length, blank_id)
    return float(-alpha[logit_length, label_length])


def example_loss_and_logit_grad(
    logits: np.ndarray,
    labels: np.ndarray,
    logit_length: int,
    label_length: int,
    blank_id: int,
) -> tuple[float, np.ndarray]:
    """Return the loss and ``d(loss)/d(logits)`` for one padded lattice.

    Gradients are produced from analytic edge posteriors and then pushed through
    the softmax Jacobian. Entries outside the valid ``[0, T) x [0, U]`` region
    are exactly zero, which is what makes padding invariance testable.
    """

    log_probs = log_softmax(logits)
    labels = np.asarray(labels)
    _validate_example(log_probs, labels, logit_length, label_length, blank_id)
    alpha = _alpha(log_probs, labels, logit_length, label_length, blank_id)
    beta = _beta(log_probs, labels, logit_length, label_length, blank_id)
    total = float(alpha[logit_length, label_length])

    # d(-log P) / d(log p_edge) is the negated posterior probability of that edge.
    grad_log_probs = np.zeros_like(log_probs)
    for time in range(logit_length):
        for output in range(label_length + 1):
            occupancy = alpha[time, output]
            if occupancy == -np.inf:
                continue
            blank_suffix = beta[time + 1, output]
            if blank_suffix != -np.inf:
                grad_log_probs[time, output, blank_id] -= np.exp(
                    occupancy + log_probs[time, output, blank_id] + blank_suffix - total
                )
            if output < label_length:
                token = int(labels[output])
                label_suffix = beta[time, output + 1]
                if label_suffix != -np.inf:
                    grad_log_probs[time, output, token] -= np.exp(
                        occupancy
                        + log_probs[time, output, token]
                        + label_suffix
                        - total
                    )

    # Chain through log_softmax: dL/dz = g - softmax(z) * sum(g).
    probabilities = np.exp(log_probs)
    totals = np.sum(grad_log_probs, axis=-1, keepdims=True)
    grad_logits = grad_log_probs - probabilities * totals
    # Zero the padded region explicitly; sum(g) is zero there, but being
    # explicit keeps the padding-invariance test independent of that identity.
    grad_logits[logit_length:, :, :] = 0.0
    grad_logits[:, label_length + 1 :, :] = 0.0
    return float(-alpha[logit_length, label_length]), grad_logits


def batch_losses(
    logits: np.ndarray,
    labels: np.ndarray,
    logit_lengths: np.ndarray,
    label_lengths: np.ndarray,
    blank_id: int,
) -> np.ndarray:
    """Return per-example FP64 losses for a padded ``[B, T, U+1, V]`` batch."""

    logits = np.asarray(logits)
    labels = np.asarray(labels)
    if logits.ndim != 4:
        raise ValueError(f"logits must have rank 4 [B,T,U+1,V], got {logits.shape}")
    logit_lengths = np.asarray(logit_lengths)
    label_lengths = np.asarray(label_lengths)
    return np.asarray(
        [
            example_loss(
                logits[index],
                labels[index],
                int(logit_lengths[index]),
                int(label_lengths[index]),
                blank_id,
            )
            for index in range(logits.shape[0])
        ],
        dtype=np.float64,
    )


def reduction_weights(
    label_lengths: np.ndarray, reduction: Reduction, batch_size: int
) -> np.ndarray:
    """Return ``d(reduced_loss)/d(per_example_loss)`` for a reduction.

    The clamped denominators match :func:`fcrnnt_jax.rnnt_loss._reduce_losses`,
    including the ``U=0`` edge case.
    """

    _validate_reduction(reduction)
    lengths = np.asarray(label_lengths, dtype=np.float64)
    if reduction == "none":
        return np.ones((batch_size,), dtype=np.float64)
    if reduction == "sum":
        return np.ones((batch_size,), dtype=np.float64)
    if reduction == "mean_batch":
        return np.full((batch_size,), 1.0 / batch_size, dtype=np.float64)
    if reduction == "mean":
        return 1.0 / (np.maximum(lengths, 1.0) * batch_size)
    return np.full((batch_size,), 1.0 / max(float(lengths.sum()), 1.0), dtype=np.float64)


def reduce_losses(
    losses: np.ndarray, label_lengths: np.ndarray, reduction: Reduction
) -> np.ndarray:
    """Apply a NeMo-compatible reduction to per-example losses."""

    _validate_reduction(reduction)
    losses = np.asarray(losses, dtype=np.float64)
    if reduction == "none":
        return losses
    weights = reduction_weights(label_lengths, reduction, losses.shape[0])
    return np.asarray(float(np.sum(losses * weights)))


def batch_loss_and_logit_grad(
    logits: np.ndarray,
    labels: np.ndarray,
    logit_lengths: np.ndarray,
    label_lengths: np.ndarray,
    blank_id: int,
    reduction: Reduction = "mean_batch",
) -> tuple[np.ndarray, np.ndarray]:
    """Return the reduced loss and its gradient with respect to ``logits``.

    For ``reduction="none"`` the gradient is that of ``sum(losses)``, which is
    the convention used when comparing against ``jax.grad`` of a summed loss.
    """

    _validate_reduction(reduction)
    logits = np.asarray(logits)
    if logits.ndim != 4:
        raise ValueError(f"logits must have rank 4 [B,T,U+1,V], got {logits.shape}")
    labels = np.asarray(labels)
    logit_lengths = np.asarray(logit_lengths)
    label_lengths = np.asarray(label_lengths)

    losses = np.empty((logits.shape[0],), dtype=np.float64)
    gradients = np.zeros(logits.shape, dtype=np.float64)
    for index in range(logits.shape[0]):
        loss, gradient = example_loss_and_logit_grad(
            logits[index],
            labels[index],
            int(logit_lengths[index]),
            int(label_lengths[index]),
            blank_id,
        )
        losses[index] = loss
        gradients[index] = gradient

    weights = reduction_weights(label_lengths, reduction, logits.shape[0])
    reduced = reduce_losses(losses, label_lengths, reduction)
    return reduced, gradients * weights[:, None, None, None]


__all__ = [
    "batch_loss_and_logit_grad",
    "batch_losses",
    "example_loss",
    "example_loss_and_logit_grad",
    "log_softmax",
    "reduce_losses",
    "reduction_weights",
]
