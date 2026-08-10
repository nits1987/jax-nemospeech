import hashlib
import json
from pathlib import Path
import wave

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fcrnnt_jax.audio import AudioConfig, frame_lengths, load_pcm16_wav, log_mel_spectrogram
from fcrnnt_jax.cli import main


def _write_wav(
    path: Path,
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
    sample_count: int = 800,
) -> None:
    values = np.arange(sample_count * channels, dtype=np.int16)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(values.astype("<i2").tobytes())


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


def test_load_pcm16_wav_returns_normalized_mono_samples(tmp_path: Path):
    path = tmp_path / "speech.wav"
    _write_wav(path)

    loaded = load_pcm16_wav(path)

    assert loaded.samples.shape == (1, 800)
    assert loaded.samples.dtype == np.float32
    np.testing.assert_array_equal(loaded.sample_lengths, [800])
    assert loaded.sample_rate == 16_000
    assert loaded.channels == 1
    assert loaded.sample_width_bytes == 2
    assert loaded.duration_seconds == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("sample_rate", "channels", "message"),
    [(8_000, 1, "sample rate"), (16_000, 2, "mono")],
)
def test_load_pcm16_wav_rejects_incompatible_audio(
    tmp_path: Path, sample_rate: int, channels: int, message: str
):
    path = tmp_path / "incompatible.wav"
    _write_wav(path, sample_rate=sample_rate, channels=channels)

    with pytest.raises(ValueError, match=message):
        load_pcm16_wav(path)


def test_load_pcm16_wav_rejects_empty_audio(tmp_path: Path):
    path = tmp_path / "empty.wav"
    _write_wav(path, sample_count=0)

    with pytest.raises(ValueError, match="at least one"):
        load_pcm16_wav(path)


def test_audio_smoke_writes_frontend_evidence_without_asr_claim(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    audio_path = tmp_path / "speech.wav"
    transcript_path = tmp_path / "speech.txt"
    artifact_path = tmp_path / "frontend.npz"
    _write_wav(audio_path)
    transcript_path.write_text("A small TEST transcript.\n", encoding="utf-8")
    expected_hash = hashlib.sha256(audio_path.read_bytes()).hexdigest()

    exit_code = main(
        [
            "audio-smoke",
            "--audio",
            str(audio_path),
            "--reference-transcript-file",
            str(transcript_path),
            "--expected-audio-sha256",
            expected_hash,
            "--output",
            str(artifact_path),
        ]
    )

    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert exit_code == 0
    assert report["status"] == "pass"
    assert report["validation_scope"] == "audio-ingestion-and-jax-frontend-only"
    assert report["asr_decode_executed"] is False
    assert report["reference"]["normalized_text"] == "a small test transcript"
    assert report["frontend"]["feature_shape"] == [1, 3, 80]
    assert "hypothesis" not in report
    assert "wer" not in report
    with np.load(artifact_path, allow_pickle=False) as archive:
        assert archive["features"].shape == (1, 3, 80)
        assert archive["reference_transcript"].item() == "A small TEST transcript."
