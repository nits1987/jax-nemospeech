from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fcrnnt_jax.config import FastConformerConfig, ParakeetConfig
from fcrnnt_jax.model import ParakeetRNNT, subsample_lengths


def _batch(config: ParakeetConfig):
    features = jax.random.normal(
        jax.random.key(1), (2, 17, config.encoder.num_mel_bins)
    )
    feature_lengths = jnp.asarray([17, 9], dtype=jnp.int32)
    tokens = jnp.asarray([[1, 2, 3, 4], [2, 3, 0, 0]], dtype=jnp.int32)
    token_lengths = jnp.asarray([4, 2], dtype=jnp.int32)
    return features, feature_lengths, tokens, token_lengths


def test_production_factory_matches_public_architecture() -> None:
    config = ParakeetConfig.parakeet_1_1b()

    assert config.encoder.num_hidden_layers == 42
    assert config.encoder.hidden_size == 1024
    assert config.encoder.intermediate_size == 4096
    assert config.encoder.num_attention_heads == 8
    assert config.encoder.subsampling_factor == 8
    assert config.encoder.subsampling_conv_channels == 256
    assert config.encoder.conv_kernel_size == 9
    assert config.predictor.hidden_size == 640
    assert config.predictor.num_layers == 2
    assert config.vocab_size == 1025
    assert config.blank_id == 1024


def test_config_rejects_invalid_attention_and_subsampling() -> None:
    with pytest.raises(ValueError, match="divisible"):
        FastConformerConfig(hidden_size=18, num_attention_heads=4)
    with pytest.raises(ValueError, match="power of two"):
        FastConformerConfig(subsampling_factor=6)


def test_subsample_lengths_are_three_rounds_of_ceiling_division() -> None:
    config = ParakeetConfig.tiny().encoder
    lengths = jnp.asarray([0, 1, 8, 9, 16, 17], dtype=jnp.int32)

    actual = subsample_lengths(lengths, config)

    np.testing.assert_array_equal(actual, np.asarray([0, 1, 1, 2, 2, 3]))


def test_forward_shape_length_and_padding_contracts() -> None:
    config = ParakeetConfig.tiny(vocab_size=8)
    model = ParakeetRNNT(config)
    batch = _batch(config)
    variables = model.init(
        jax.random.key(0), *batch, train=False, compute_joint=True
    )

    output = model.apply(
        variables, *batch, train=False, compute_joint=True
    )

    assert output["encoder"].shape == (2, 3, 16)
    np.testing.assert_array_equal(output["encoder_lengths"], np.asarray([3, 2]))
    assert output["predictor"].shape == (2, 5, 16)
    np.testing.assert_array_equal(output["predictor_lengths"], np.asarray([5, 3]))
    assert output["joint_logits"].shape == (2, 3, 5, 8)
    np.testing.assert_array_equal(output["encoder"][1, 2], np.zeros(16))
    np.testing.assert_array_equal(output["predictor"][1, 3:], np.zeros((2, 16)))


def test_default_forward_does_not_materialize_joint_lattice() -> None:
    config = ParakeetConfig.tiny(vocab_size=8)
    model = ParakeetRNNT(config)
    batch = _batch(config)
    variables = model.init(jax.random.key(0), *batch, train=False)

    output = jax.jit(lambda *args: model.apply(variables, *args, train=False))(*batch)

    assert output["joint_logits"] is None


def test_masked_feature_values_cannot_change_valid_encoder_outputs() -> None:
    config = ParakeetConfig.tiny(vocab_size=8)
    model = ParakeetRNNT(config)
    features, lengths, tokens, token_lengths = _batch(config)
    variables = model.init(
        jax.random.key(0), features, lengths, tokens, token_lengths, train=False
    )
    changed = features.at[1, 9:, :].set(1e4)

    original, original_lengths = model.apply(
        variables, features, lengths, train=False, method=model.encode
    )
    modified, modified_lengths = model.apply(
        variables, changed, lengths, train=False, method=model.encode
    )

    np.testing.assert_array_equal(original_lengths, modified_lengths)
    np.testing.assert_allclose(original[1, :2], modified[1, :2], rtol=1e-5, atol=1e-5)


def test_frame_joint_returns_normalized_fp32_log_probs() -> None:
    config = ParakeetConfig.tiny(vocab_size=8)
    model = ParakeetRNNT(config)
    batch = _batch(config)
    variables = model.init(jax.random.key(0), *batch, train=False)
    output = model.apply(variables, *batch, train=False)

    log_probs = model.apply(
        variables,
        output["encoder"][:, 0],
        output["predictor"],
        method=model.joint_frame_log_probs,
    )

    assert log_probs.shape == (2, 5, 8)
    assert log_probs.dtype == jnp.float32
    np.testing.assert_allclose(jnp.exp(log_probs).sum(-1), 1.0, rtol=1e-5, atol=1e-5)


def test_frame_joint_raw_logits_have_loss_facing_shape() -> None:
    config = ParakeetConfig.tiny(vocab_size=8)
    model = ParakeetRNNT(config)
    batch = _batch(config)
    variables = model.init(jax.random.key(0), *batch, train=False)
    output = model.apply(variables, *batch, train=False)

    logits = model.apply(
        variables,
        output["encoder"][:, 0],
        output["predictor"],
        method=model.joint,
    )

    assert logits.shape == (2, 5, 8)


def test_zero_length_item_stays_zero_and_finite() -> None:
    config = ParakeetConfig.tiny(vocab_size=8)
    model = ParakeetRNNT(config)
    features, _, tokens, token_lengths = _batch(config)
    lengths = jnp.asarray([17, 0], dtype=jnp.int32)
    variables = model.init(
        jax.random.key(0), features, lengths, tokens, token_lengths, train=False
    )

    output = model.apply(
        variables, features, lengths, tokens, token_lengths, train=False
    )

    assert bool(jnp.all(jnp.isfinite(output["encoder"])))
    np.testing.assert_array_equal(output["encoder"][1], np.zeros((3, 16)))


def test_training_updates_batch_stats_and_has_finite_gradients() -> None:
    config = ParakeetConfig.tiny(vocab_size=8)
    model = ParakeetRNNT(config)
    batch = _batch(config)
    variables = model.init(
        {"params": jax.random.key(0), "dropout": jax.random.key(1)},
        *batch,
        train=True,
        compute_joint=True,
    )

    def objective(params):
        output, updates = model.apply(
            {"params": params, "batch_stats": variables["batch_stats"]},
            *batch,
            train=True,
            compute_joint=True,
            rngs={"dropout": jax.random.key(2)},
            mutable=["batch_stats"],
        )
        return jnp.mean(output["joint_logits"].astype(jnp.float32) ** 2), updates

    (loss, updates), gradients = jax.value_and_grad(objective, has_aux=True)(
        variables["params"]
    )

    assert bool(jnp.isfinite(loss))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(gradients))
    old_means = [
        value
        for path, value in jax.tree_util.tree_leaves_with_path(variables["batch_stats"])
        if path[-1].key == "mean"
    ]
    new_means = [
        value
        for path, value in jax.tree_util.tree_leaves_with_path(updates["batch_stats"])
        if path[-1].key == "mean"
    ]
    assert any(not bool(jnp.array_equal(old, new)) for old, new in zip(old_means, new_means))


def test_shape_contracts_fail_early() -> None:
    config = ParakeetConfig.tiny(vocab_size=8)
    model = ParakeetRNNT(config)
    batch = _batch(config)
    variables = model.init(jax.random.key(0), *batch, train=False)

    with pytest.raises(ValueError, match="mel bins"):
        model.apply(
            variables,
            jnp.zeros((2, 17, 3)),
            batch[1],
            batch[2],
            batch[3],
            train=False,
        )
    with pytest.raises(ValueError, match="feature_lengths"):
        model.apply(
            variables,
            batch[0],
            jnp.ones((2, 1), dtype=jnp.int32),
            batch[2],
            batch[3],
            train=False,
        )
