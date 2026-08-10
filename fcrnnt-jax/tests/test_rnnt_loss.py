"""CPU conformance tests for the standalone classic RNN-T loss."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fcrnnt_jax.rnnt_loss import rnnt_loss_from_joint, rnnt_loss_from_logits


def _np_log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    maximum = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _np_rnnt_loss(
    logits: np.ndarray,
    labels: np.ndarray,
    logit_length: int,
    label_length: int,
    blank_id: int,
) -> float:
    """Straightforward NumPy alpha recursion used only as a test oracle."""
    log_probs = _np_log_softmax(logits)
    alpha = np.full((logit_length + 1, label_length + 1), -np.inf)
    alpha[0, 0] = 0.0
    for time in range(logit_length):
        for output in range(label_length + 1):
            score = alpha[time, output]
            alpha[time + 1, output] = np.logaddexp(
                alpha[time + 1, output],
                score + log_probs[time, output, blank_id],
            )
            if output < label_length:
                token = labels[output]
                alpha[time, output + 1] = np.logaddexp(
                    alpha[time, output + 1],
                    score + log_probs[time, output, token],
                )
    return float(-alpha[logit_length, label_length])


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(2, 4, 3, 5)).astype(np.float32)
    # The last value for example 1 is deliberately an invalid padding sentinel.
    labels = np.asarray([[1, 3], [2, -99]], dtype=np.int32)
    logit_lengths = np.asarray([4, 2], dtype=np.int32)
    label_lengths = np.asarray([2, 1], dtype=np.int32)
    return logits, labels, logit_lengths, label_lengths, 4


def test_matches_numpy_reference_for_mixed_lengths() -> None:
    logits, labels, logit_lengths, label_lengths, blank_id = _fixture()
    expected = np.asarray(
        [
            _np_rnnt_loss(
                logits[index],
                labels[index],
                int(logit_lengths[index]),
                int(label_lengths[index]),
                blank_id,
            )
            for index in range(logits.shape[0])
        ],
        dtype=np.float32,
    )
    actual = rnnt_loss_from_logits(
        logits,
        labels,
        logit_lengths,
        label_lengths,
        blank_id=blank_id,
        reduction="none",
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-5, atol=2e-5)
    assert actual.dtype == jnp.float32


@pytest.mark.parametrize(
    ("reduction", "expected_fn"),
    [
        ("sum", lambda values, lengths: values.sum()),
        ("mean_batch", lambda values, lengths: values.mean()),
        ("mean", lambda values, lengths: (values / lengths).mean()),
        ("mean_volume", lambda values, lengths: values.sum() / lengths.sum()),
    ],
)
def test_nemo_reductions(reduction: str, expected_fn) -> None:
    logits, labels, logit_lengths, label_lengths, blank_id = _fixture()
    values = np.asarray(
        rnnt_loss_from_logits(
            logits,
            labels,
            logit_lengths,
            label_lengths,
            blank_id=blank_id,
            reduction="none",
        )
    )
    actual = rnnt_loss_from_logits(
        logits,
        labels,
        logit_lengths,
        label_lengths,
        blank_id=blank_id,
        reduction=reduction,
    )
    expected = expected_fn(values, label_lengths.astype(np.float32))
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-6)


def test_gradient_matches_central_finite_difference() -> None:
    logits = jnp.asarray(
        [[[[0.2, -0.1, 0.4], [0.1, 0.3, -0.2]],
          [[-0.4, 0.2, 0.5], [0.7, -0.3, 0.1]]]],
        dtype=jnp.float32,
    )
    labels = jnp.asarray([[1]], dtype=jnp.int32)
    logit_lengths = jnp.asarray([2], dtype=jnp.int32)
    label_lengths = jnp.asarray([1], dtype=jnp.int32)

    def loss_fn(value: jax.Array) -> jax.Array:
        return rnnt_loss_from_logits(
            value,
            labels,
            logit_lengths,
            label_lengths,
            blank_id=2,
            reduction="sum",
        )

    analytic = np.asarray(jax.grad(loss_fn)(logits))
    base = np.asarray(logits)
    numeric = np.empty_like(base)
    epsilon = 2e-3
    for index in np.ndindex(base.shape):
        plus = base.copy()
        minus = base.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numeric[index] = (
            float(loss_fn(jnp.asarray(plus))) - float(loss_fn(jnp.asarray(minus)))
        ) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numeric, rtol=6e-3, atol=2e-3)
    assert np.isfinite(analytic).all()


def test_padding_is_ignored_and_has_zero_gradient() -> None:
    logits, labels, logit_lengths, label_lengths, blank_id = _fixture()
    baseline = rnnt_loss_from_logits(
        logits,
        labels,
        logit_lengths,
        label_lengths,
        blank_id=blank_id,
        reduction="none",
    )

    changed = logits.copy()
    changed[1, 2:, :, :] = 1000.0  # acoustic padding
    changed[1, :, 2, :] = -1000.0  # target-state padding
    candidate = rnnt_loss_from_logits(
        changed,
        labels,
        logit_lengths,
        label_lengths,
        blank_id=blank_id,
        reduction="none",
    )
    np.testing.assert_allclose(np.asarray(candidate), np.asarray(baseline), rtol=0, atol=0)

    def second_example_loss(value: jax.Array) -> jax.Array:
        return rnnt_loss_from_logits(
            value,
            jnp.asarray(labels),
            jnp.asarray(logit_lengths),
            jnp.asarray(label_lengths),
            blank_id=blank_id,
            reduction="none",
        )[1]

    gradient = np.asarray(jax.grad(second_example_loss)(jnp.asarray(logits)))
    np.testing.assert_array_equal(gradient[1, 2:, :, :], 0.0)
    np.testing.assert_array_equal(gradient[1, :, 2, :], 0.0)
    np.testing.assert_array_equal(gradient[0], 0.0)


def test_empty_target_single_frame_and_bfloat16_use_fp32() -> None:
    logits = jnp.asarray([[[[0.25, -0.5, 1.0]]]], dtype=jnp.bfloat16)
    labels = jnp.empty((1, 0), dtype=jnp.int32)
    logit_lengths = jnp.asarray([1], dtype=jnp.int32)
    label_lengths = jnp.asarray([0], dtype=jnp.int32)
    actual = rnnt_loss_from_logits(
        logits,
        labels,
        logit_lengths,
        label_lengths,
        blank_id=2,
        reduction="none",
    )
    expected = -jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)[0, 0, 0, 2]
    np.testing.assert_allclose(np.asarray(actual[0]), np.asarray(expected), rtol=1e-6)
    assert actual.dtype == jnp.float32

    # Token-normalized reductions are defined with a denominator clamp for U=0.
    assert math.isfinite(
        float(
            rnnt_loss_from_logits(
                logits,
                labels,
                logit_lengths,
                label_lengths,
                blank_id=2,
                reduction="mean_volume",
            )
        )
    )


def test_empty_target_inside_padded_batch_and_t1_nonempty_target() -> None:
    rng = np.random.default_rng(29)
    logits = rng.normal(size=(2, 3, 3, 4)).astype(np.float32)
    labels = np.asarray([[-17, -17], [1, 2]], dtype=np.int32)
    logit_lengths = np.asarray([3, 1], dtype=np.int32)
    label_lengths = np.asarray([0, 2], dtype=np.int32)
    expected = np.asarray(
        [
            _np_rnnt_loss(logits[0], labels[0], 3, 0, 3),
            _np_rnnt_loss(logits[1], labels[1], 1, 2, 3),
        ]
    )
    actual = rnnt_loss_from_logits(
        logits,
        labels,
        logit_lengths,
        label_lengths,
        blank_id=3,
        reduction="none",
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-5, atol=2e-5)


def test_jit_value_and_gradient() -> None:
    logits, labels, logit_lengths, label_lengths, blank_id = _fixture()

    @jax.jit
    def compiled(value: jax.Array) -> tuple[jax.Array, jax.Array]:
        fn = lambda tensor: rnnt_loss_from_logits(
            tensor,
            labels,
            logit_lengths,
            label_lengths,
            blank_id=blank_id,
            reduction="sum",
        )
        return fn(value), jax.grad(fn)(value)

    value, gradient = compiled(jnp.asarray(logits))
    assert value.dtype == jnp.float32
    assert gradient.shape == logits.shape
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(gradient)).all()


def test_streamed_joint_matches_full_logits_and_trains_joint_params() -> None:
    rng = np.random.default_rng(19)
    batch_size, max_time, max_labels, hidden, vocab = 2, 3, 2, 4, 5
    encoder = jnp.asarray(
        rng.normal(size=(batch_size, max_time, hidden)).astype(np.float32)
    )
    predictor = jnp.asarray(
        rng.normal(size=(batch_size, max_labels + 1, hidden)).astype(np.float32)
    )
    labels = jnp.asarray([[1, 2], [3, -1]], dtype=jnp.int32)
    logit_lengths = jnp.asarray([3, 2], dtype=jnp.int32)
    label_lengths = jnp.asarray([2, 1], dtype=jnp.int32)
    params = {
        "encoder_kernel": jnp.asarray(
            rng.normal(scale=0.2, size=(hidden, vocab)).astype(np.float32)
        ),
        "predictor_kernel": jnp.asarray(
            rng.normal(scale=0.2, size=(hidden, vocab)).astype(np.float32)
        ),
        "bias": jnp.asarray(rng.normal(scale=0.1, size=(vocab,)).astype(np.float32)),
    }

    def joint_fn(parameters, encoder_frame, all_predictor):
        acoustic = encoder_frame @ parameters["encoder_kernel"]
        lexical = all_predictor @ parameters["predictor_kernel"]
        return jnp.tanh(acoustic[:, None, :] + lexical) + parameters["bias"]

    def full_logits(parameters):
        acoustic = encoder @ parameters["encoder_kernel"]
        lexical = predictor @ parameters["predictor_kernel"]
        return (
            jnp.tanh(acoustic[:, :, None, :] + lexical[:, None, :, :])
            + parameters["bias"]
        )

    def direct_loss(parameters):
        return rnnt_loss_from_logits(
            full_logits(parameters),
            labels,
            logit_lengths,
            label_lengths,
            blank_id=4,
            reduction="mean_volume",
        )

    def streamed_loss(parameters):
        return rnnt_loss_from_joint(
            encoder,
            predictor,
            joint_fn,
            labels,
            logit_lengths,
            label_lengths,
            blank_id=4,
            reduction="mean_volume",
            joint_params=parameters,
        )

    direct_value, direct_grad = jax.value_and_grad(direct_loss)(params)
    streamed_value, streamed_grad = jax.jit(jax.value_and_grad(streamed_loss))(params)
    np.testing.assert_allclose(
        np.asarray(streamed_value), np.asarray(direct_value), rtol=2e-5, atol=2e-5
    )
    for name in params:
        np.testing.assert_allclose(
            np.asarray(streamed_grad[name]),
            np.asarray(direct_grad[name]),
            rtol=5e-5,
            atol=5e-5,
        )
        assert np.linalg.norm(np.asarray(streamed_grad[name])) > 0


def test_dynamic_validation_is_jit_safe() -> None:
    logits = jnp.zeros((1, 2, 2, 3), dtype=jnp.float32)
    labels = jnp.asarray([[1]], dtype=jnp.int32)

    @jax.jit
    def loss_with_lengths(times, outputs):
        return rnnt_loss_from_logits(
            logits,
            labels,
            times,
            outputs,
            blank_id=2,
            reduction="none",
        )

    valid = loss_with_lengths(
        jnp.asarray([2], dtype=jnp.int32), jnp.asarray([1], dtype=jnp.int32)
    )
    invalid = loss_with_lengths(
        jnp.asarray([0], dtype=jnp.int32), jnp.asarray([1], dtype=jnp.int32)
    )
    assert np.isfinite(np.asarray(valid)).all()
    assert np.isnan(np.asarray(invalid)).all()


def test_static_contract_validation() -> None:
    with pytest.raises(ValueError, match="rank 4"):
        rnnt_loss_from_logits(
            jnp.zeros((1, 2, 3)),
            jnp.zeros((1, 1), dtype=jnp.int32),
            jnp.ones((1,), dtype=jnp.int32),
            jnp.ones((1,), dtype=jnp.int32),
            blank_id=2,
        )
    with pytest.raises(ValueError, match="outside vocabulary"):
        rnnt_loss_from_logits(
            jnp.zeros((1, 2, 2, 3)),
            jnp.ones((1, 1), dtype=jnp.int32),
            jnp.asarray([2], dtype=jnp.int32),
            jnp.asarray([1], dtype=jnp.int32),
            blank_id=3,
        )
    with pytest.raises(ValueError, match="invalid reduction"):
        rnnt_loss_from_logits(
            jnp.zeros((1, 2, 2, 3)),
            jnp.ones((1, 1), dtype=jnp.int32),
            jnp.asarray([2], dtype=jnp.int32),
            jnp.asarray([1], dtype=jnp.int32),
            blank_id=2,
            reduction="average",
        )
