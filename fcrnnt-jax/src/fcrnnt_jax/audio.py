"""Small JAX log-mel frontend used by the PoC.

The public NeMo checkpoint remains the frontend oracle.  This module is useful
for end-to-end execution and synthetic qualification, while saved NeMo features
should be used when isolating model parity.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

Array = jax.Array


@dataclass(frozen=True)
class AudioConfig:
    """Frontend parameters with Parakeet-like defaults."""

    sample_rate: int = 16_000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float | None = None
    preemphasis: float = 0.97
    log_floor: float = 1e-10
    normalize: bool = True

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0 < self.win_length <= self.n_fft:
            raise ValueError("win_length must be in [1, n_fft]")
        if self.hop_length <= 0 or self.n_mels <= 0:
            raise ValueError("hop_length and n_mels must be positive")
        nyquist = self.sample_rate / 2
        f_max = nyquist if self.f_max is None else self.f_max
        if not 0 <= self.f_min < f_max <= nyquist:
            raise ValueError("expected 0 <= f_min < f_max <= Nyquist")


def frame_lengths(sample_lengths: Array, config: AudioConfig) -> Array:
    """Return non-centred STFT frame counts for each example."""

    lengths = jnp.asarray(sample_lengths, dtype=jnp.int32)
    return jnp.maximum(0, 1 + (lengths - config.win_length) // config.hop_length)


def length_mask(lengths: Array, max_length: int) -> Array:
    """Return a boolean ``[batch, max_length]`` validity mask."""

    lengths = jnp.asarray(lengths, dtype=jnp.int32)
    return jnp.arange(max_length, dtype=jnp.int32)[None, :] < lengths[:, None]


def _hz_to_mel(frequency: Array) -> Array:
    return 2595.0 * jnp.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mels: Array) -> Array:
    return 700.0 * (jnp.power(10.0, mels / 2595.0) - 1.0)


def mel_filterbank(config: AudioConfig, dtype: jnp.dtype = jnp.float32) -> Array:
    """Construct a triangular ``[fft_bins, n_mels]`` filterbank."""

    config.validate()
    f_max = config.sample_rate / 2 if config.f_max is None else config.f_max
    mel_edges = jnp.linspace(
        _hz_to_mel(jnp.asarray(config.f_min)),
        _hz_to_mel(jnp.asarray(f_max)),
        config.n_mels + 2,
        dtype=dtype,
    )
    hz_edges = _mel_to_hz(mel_edges)
    frequencies = jnp.linspace(
        0.0, config.sample_rate / 2, config.n_fft // 2 + 1, dtype=dtype
    )[:, None]
    lower, centre, upper = hz_edges[:-2], hz_edges[1:-1], hz_edges[2:]
    rising = (frequencies - lower) / jnp.maximum(centre - lower, 1e-12)
    falling = (upper - frequencies) / jnp.maximum(upper - centre, 1e-12)
    return jnp.maximum(0.0, jnp.minimum(rising, falling)).astype(dtype)


def log_mel_spectrogram(
    samples: Array,
    sample_lengths: Array,
    config: AudioConfig = AudioConfig(),
) -> tuple[Array, Array]:
    """Compute time-major log-mel features.

    Args:
        samples: Zero-padded mono waveforms with shape ``[batch, samples]``.
        sample_lengths: Number of valid samples in each waveform.
        config: Static frontend configuration.

    Returns:
        ``(features, lengths)`` where features have shape
        ``[batch, frames, n_mels]``. Invalid frames are exactly zero.
    """

    config.validate()
    samples = jnp.asarray(samples, dtype=jnp.float32)
    sample_lengths = jnp.asarray(sample_lengths, dtype=jnp.int32)
    if samples.ndim != 2:
        raise ValueError(f"samples must have rank 2, got {samples.shape}")
    if sample_lengths.shape != (samples.shape[0],):
        raise ValueError("sample_lengths must have shape [batch]")

    # A static padded input creates a static number of frames, avoiding a new
    # TPU compilation for every utterance in a bucket.
    padded_size = max(samples.shape[1], config.win_length)
    samples = jnp.pad(samples, ((0, 0), (0, padded_size - samples.shape[1])))
    first = samples[:, :1]
    samples = jnp.concatenate(
        [first, samples[:, 1:] - config.preemphasis * samples[:, :-1]], axis=1
    )
    max_frames = max(1, 1 + (padded_size - config.win_length) // config.hop_length)
    offsets = (
        jnp.arange(max_frames, dtype=jnp.int32)[:, None] * config.hop_length
        + jnp.arange(config.win_length, dtype=jnp.int32)[None, :]
    )
    frames = jnp.take(samples, offsets, axis=1)
    window = jnp.hanning(config.win_length).astype(jnp.float32)
    frames = frames * window[None, None, :]
    spectrum = jnp.fft.rfft(frames, n=config.n_fft, axis=-1)
    power = jnp.square(jnp.abs(spectrum)).astype(jnp.float32)
    mel = power @ mel_filterbank(config)
    features = jnp.log(jnp.maximum(mel, config.log_floor))

    lengths = jnp.minimum(frame_lengths(sample_lengths, config), max_frames)
    mask = length_mask(lengths, max_frames)
    if config.normalize:
        count = jnp.maximum(lengths, 1).astype(jnp.float32)[:, None, None]
        mean = jnp.sum(jnp.where(mask[..., None], features, 0.0), axis=1, keepdims=True) / count
        variance = (
            jnp.sum(
                jnp.where(mask[..., None], jnp.square(features - mean), 0.0),
                axis=1,
                keepdims=True,
            )
            / count
        )
        features = (features - mean) * jax.lax.rsqrt(variance + 1e-5)
    features = jnp.where(mask[..., None], features, 0.0)
    return features, lengths

