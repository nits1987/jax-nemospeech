from __future__ import annotations

import jax
import numpy as np

from fcrnnt_jax.config import ParakeetConfig
from fcrnnt_jax.model import ParakeetRNNT
from fcrnnt_jax.training import (
    OptimizerConfig,
    create_train_state,
    eval_loss,
    make_synthetic_batch,
    make_train_step,
    parameter_count,
)


def test_streamed_full_training_step_updates_joint_and_encoder_params():
    config = ParakeetConfig.tiny(vocab_size=8)
    model = ParakeetRNNT(config)
    batch = make_synthetic_batch(config, feature_frames=17, target_steps=2)
    state = create_train_state(
        model,
        jax.random.key(0),
        batch,
        optimizer=OptimizerConfig(learning_rate=2e-3, weight_decay=0.0),
    )
    before_encoder = np.asarray(
        state.params["encoder"]["subsampling"]["conv_0"]["kernel"]
    ).copy()
    before_joint = np.asarray(state.params["joint_head"]["kernel"]).copy()

    next_state, metrics = make_train_step(model)(state, batch)

    assert np.isfinite(float(metrics["loss"]))
    assert bool(metrics["gradients_finite"])
    assert int(next_state.step) == 1
    assert int(next_state.data_cursor) == 1
    assert not np.array_equal(
        before_encoder,
        np.asarray(next_state.params["encoder"]["subsampling"]["conv_0"]["kernel"]),
    )
    assert not np.array_equal(before_joint, np.asarray(next_state.params["joint_head"]["kernel"]))


def test_streamed_and_materialized_eval_loss_match():
    config = ParakeetConfig.tiny(vocab_size=8)
    model = ParakeetRNNT(config)
    batch = make_synthetic_batch(config, feature_frames=17, target_steps=2)
    state = create_train_state(model, jax.random.key(0), batch)

    streamed = eval_loss(state, batch, model=model, materialize_joint=False)
    materialized = eval_loss(state, batch, model=model, materialize_joint=True)

    np.testing.assert_allclose(streamed, materialized, rtol=1e-5, atol=1e-5)
    assert parameter_count(state.params) > 0

