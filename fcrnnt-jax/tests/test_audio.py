import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fcrnnt_jax.audio import AudioConfig, frame_lengths, log_mel_spectrogram


def test_log_mel_shape_lengths_mask_and_jit():
    config = AudioConfig(n_fft=16, win_length=8, hop_length=4, n_mels=4)
    samples = jnp.stack([jnp.arange(20), jnp.arange(20)], axis=0).astype(jnp.float32)
    lengths = jnp.asarray([20, 12], dtype=jnp.int32)

    features, output_lengths = jax.jit(
        lambda x, n: log_mel_spectrogram(x, n, config)
    )(samples, lengths)

    assert features.shape == (2, 4, 4)
    np.testing.assert_array_equal(output_lengths, [4, 2])
    np.testing.assert_array_equal(np.asarray(features[1, 2:]), 0.0)
    assert np.isfinite(np.asarray(features)).all()


def test_frame_lengths_clamps_short_audio_to_zero():
    config = AudioConfig(n_fft=16, win_length=8, hop_length=4, n_mels=4)
    np.testing.assert_array_equal(frame_lengths(jnp.asarray([0, 7, 8, 12]), config), [0, 0, 1, 2])


def test_invalid_audio_config_is_rejected():
    with pytest.raises(ValueError, match="win_length"):
        AudioConfig(n_fft=8, win_length=16).validate()

