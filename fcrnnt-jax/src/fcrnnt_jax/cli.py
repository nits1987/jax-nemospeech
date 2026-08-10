"""Operator-facing smoke and fitment commands."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import sys
import time
import traceback
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .audio import AudioConfig, load_pcm16_wav, log_mel_spectrogram
from .checkpoint import restore_checkpoint, save_checkpoint, tree_fingerprint
from .config import ParakeetConfig
from .model import PARITY_GAPS, ParakeetRNNT
from .rnnt_loss import rnnt_loss_from_logits
from .training import (
    OptimizerConfig,
    create_train_state,
    eval_loss,
    make_synthetic_batch,
    make_train_step,
    parameter_count,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (jax.Array, np.ndarray)):
        array = np.asarray(value)
        return array.item() if array.ndim == 0 else array.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def _versions() -> dict[str, str]:
    names = ["jax", "jaxlib", "flax", "optax", "orbax-checkpoint", "numpy"]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _base_report(command: str) -> dict[str, Any]:
    return {
        "command": command,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "python": platform.python_version(),
        "versions": _versions(),
    }


def _print_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, sort_keys=True, default=_json_default))


def _config_for_preset(name: str) -> ParakeetConfig:
    if name == "tiny":
        return ParakeetConfig.tiny(vocab_size=8)
    if name == "parakeet-1.1b":
        return ParakeetConfig.parakeet_1_1b()
    raise ValueError(f"unknown preset: {name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_transcript(path: Path) -> tuple[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"reference transcript not found: {resolved}")
    text = " ".join(resolved.read_text(encoding="utf-8").split())
    if not text:
        raise ValueError("reference transcript must not be empty")
    normalized = " ".join(re.findall(r"[a-z0-9']+", text.lower()))
    if not normalized:
        raise ValueError("reference transcript has no words after normalization")
    return text, normalized


def _logaddexp(a: float, b: float) -> float:
    return float(np.logaddexp(a, b))


def _numpy_single_rnnt_loss(
    logits: np.ndarray,
    labels: np.ndarray,
    logit_length: int,
    label_length: int,
    blank_id: int,
) -> float:
    """Independent small-lattice oracle used only by the built-in CLI smoke."""

    selected = logits[:logit_length, : label_length + 1].astype(np.float64)
    maximum = np.max(selected, axis=-1, keepdims=True)
    log_probs = selected - maximum - np.log(
        np.sum(np.exp(selected - maximum), axis=-1, keepdims=True)
    )
    alpha = np.full((logit_length + 1, label_length + 1), -np.inf)
    alpha[0, 0] = 0.0
    for t in range(logit_length):
        for u in range(label_length + 1):
            alpha[t + 1, u] = _logaddexp(
                alpha[t + 1, u], alpha[t, u] + log_probs[t, u, blank_id]
            )
            if u < label_length:
                token = int(labels[u])
                alpha[t, u + 1] = _logaddexp(
                    alpha[t, u + 1], alpha[t, u] + log_probs[t, u, token]
                )
    return -float(alpha[logit_length, label_length])


def _numpy_batch_loss(
    logits: np.ndarray,
    labels: np.ndarray,
    logit_lengths: np.ndarray,
    label_lengths: np.ndarray,
    blank_id: int,
    reduction: str,
) -> np.ndarray:
    losses = np.asarray(
        [
            _numpy_single_rnnt_loss(logits[i], labels[i], int(t), int(u), blank_id)
            for i, (t, u) in enumerate(zip(logit_lengths, label_lengths, strict=True))
        ],
        dtype=np.float64,
    )
    if reduction == "none":
        return losses
    if reduction == "sum":
        return np.asarray(losses.sum())
    if reduction == "mean_batch":
        return np.asarray(losses.mean())
    denominators = np.maximum(label_lengths.astype(np.float64), 1.0)
    if reduction == "mean":
        return np.asarray((losses / denominators).mean())
    if reduction == "mean_volume":
        return np.asarray(losses.sum() / max(float(label_lengths.sum()), 1.0))
    raise ValueError(f"unsupported reduction: {reduction}")


def _loss_fixture(path: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if path is None:
        generator = np.random.default_rng(17)
        logits = generator.normal(size=(1, 2, 3, 4)).astype(np.float32)
        labels = np.asarray([[1, 2]], dtype=np.int32)
        logit_lengths = np.asarray([2], dtype=np.int32)
        label_lengths = np.asarray([2], dtype=np.int32)
        blank_id = 3
        reduction = "mean_batch"
        expected_loss = _numpy_batch_loss(
            logits, labels, logit_lengths, label_lengths, blank_id, reduction
        )
        return (
            {
                "logits": logits,
                "labels": labels,
                "logit_lengths": logit_lengths,
                "label_lengths": label_lengths,
                "blank_id": blank_id,
                "reduction": reduction,
                "expected_loss": expected_loss,
            },
            {"kind": "built-in-numpy-oracle", "seed": 17},
        )

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"fixture not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "logits",
            "labels",
            "logit_lengths",
            "label_lengths",
            "blank_id",
            "expected_loss",
            "expected_logits_grad",
        }
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"fixture missing arrays: {sorted(missing)}")
        reduction = str(archive["reduction"].item()) if "reduction" in archive else "mean_batch"
        fixture = {name: np.asarray(archive[name]) for name in required}
        fixture["blank_id"] = int(fixture["blank_id"].item())
        fixture["reduction"] = reduction
    return fixture, {"kind": "external-reference", "path": str(path), "sha256": _sha256(path)}


def command_devices(_: argparse.Namespace) -> int:
    report = _base_report("devices")
    devices = jax.devices()
    matrix = jnp.ones((2048, 2048), dtype=jnp.bfloat16)
    started = time.perf_counter()
    product = jax.jit(lambda value: value @ value)(matrix)
    jax.block_until_ready(product)
    matmul_seconds = time.perf_counter() - started
    matmul_finite = bool(jnp.all(jnp.isfinite(product)))
    passed = bool(devices) and matmul_finite
    report.update(
        {
            "status": "pass" if passed else "fail",
            "device_count": jax.device_count(),
            "local_device_count": jax.local_device_count(),
            "process_count": jax.process_count(),
            "process_index": jax.process_index(),
            "devices": [str(device) for device in devices],
            "platforms": sorted({device.platform for device in devices}),
            "matmul": {
                "shape": [2048, 2048],
                "dtype": "bfloat16",
                "compile_and_execute_seconds": matmul_seconds,
                "finite": matmul_finite,
            },
        }
    )
    _print_report(report)
    return 0 if passed else 1


def command_loss_smoke(args: argparse.Namespace) -> int:
    fixture, fixture_evidence = _loss_fixture(args.fixture)
    logits_np = fixture["logits"].astype(np.float32)
    labels = jnp.asarray(fixture["labels"])
    logit_lengths = jnp.asarray(fixture["logit_lengths"])
    label_lengths = jnp.asarray(fixture["label_lengths"])
    blank_id = int(fixture["blank_id"])
    reduction = str(fixture["reduction"])

    def scalar_loss(value: jax.Array) -> jax.Array:
        result = rnnt_loss_from_logits(
            value,
            labels,
            logit_lengths,
            label_lengths,
            blank_id=blank_id,
            reduction=reduction,
        )
        return jnp.sum(result)

    actual_loss = rnnt_loss_from_logits(
        jnp.asarray(logits_np),
        labels,
        logit_lengths,
        label_lengths,
        blank_id=blank_id,
        reduction=reduction,
    )
    gradient = jax.grad(scalar_loss)(jnp.asarray(logits_np))
    jax.block_until_ready(gradient)
    expected_loss = np.asarray(fixture["expected_loss"], dtype=np.float64)
    actual_np = np.asarray(actual_loss)
    loss_abs_error = float(np.max(np.abs(actual_np - expected_loss)))
    loss_ok = bool(np.allclose(actual_np, expected_loss, rtol=1e-5, atol=1e-5))
    gradient_finite = bool(np.isfinite(np.asarray(gradient)).all())
    gradient_comparison: dict[str, Any] = {"finite": gradient_finite}
    gradient_ok = gradient_finite

    if "expected_logits_grad" in fixture:
        expected_gradient = np.asarray(fixture["expected_logits_grad"], dtype=np.float32)
        if expected_gradient.shape != logits_np.shape:
            raise ValueError("expected_logits_grad shape must match logits")
        actual_gradient = np.asarray(gradient)
        numerator = float(np.vdot(actual_gradient.ravel(), expected_gradient.ravel()))
        denominator = float(
            np.linalg.norm(actual_gradient.ravel()) * np.linalg.norm(expected_gradient.ravel())
        )
        cosine = numerator / denominator if denominator else float("nan")
        maximum_error = float(np.max(np.abs(actual_gradient - expected_gradient)))
        gradient_ok = bool(cosine >= 0.99999 and np.isfinite(maximum_error))
        gradient_comparison.update(
            {"cosine_similarity": cosine, "minimum_cosine_similarity": 0.99999, "max_abs_error": maximum_error}
        )
    elif args.fixture is None:
        epsilon = 1e-3
        numerical = np.empty_like(logits_np)
        for index in np.ndindex(logits_np.shape):
            plus = logits_np.copy()
            minus = logits_np.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            plus_loss = _numpy_batch_loss(
                plus,
                np.asarray(labels),
                np.asarray(logit_lengths),
                np.asarray(label_lengths),
                blank_id,
                reduction,
            ).sum()
            minus_loss = _numpy_batch_loss(
                minus,
                np.asarray(labels),
                np.asarray(logit_lengths),
                np.asarray(label_lengths),
                blank_id,
                reduction,
            ).sum()
            numerical[index] = (plus_loss - minus_loss) / (2 * epsilon)
        maximum_error = float(np.max(np.abs(np.asarray(gradient) - numerical)))
        gradient_ok = bool(gradient_finite and maximum_error <= 5e-3)
        gradient_comparison.update(
            {"reference": "central-finite-difference", "max_abs_error": maximum_error, "max_abs_error_limit": 5e-3}
        )

    passed = loss_ok and gradient_ok and bool(np.isfinite(actual_np).all())
    report = _base_report("loss-smoke")
    report.update(
        {
            "status": "pass" if passed else "fail",
            "fixture": fixture_evidence,
            "shape": list(logits_np.shape),
            "dtype": str(logits_np.dtype),
            "blank_id": blank_id,
            "reduction": reduction,
            "expected_loss": expected_loss,
            "actual_loss": actual_np,
            "loss_max_abs_error": loss_abs_error,
            "loss_tolerance": {"atol": 1e-5, "rtol": 1e-5},
            "gradient": gradient_comparison,
        }
    )
    _print_report(report)
    return 0 if passed else 1


def command_model_smoke(args: argparse.Namespace) -> int:
    config = _config_for_preset(args.preset)
    model = ParakeetRNNT(config)
    frame_count = 17 if args.preset == "tiny" else 64
    batch = make_synthetic_batch(config, feature_frames=frame_count, target_steps=3)
    init_started = time.perf_counter()
    variables = model.init(
        jax.random.key(7),
        batch["features"],
        batch["feature_lengths"],
        batch["tokens"],
        batch["token_lengths"],
        train=False,
        compute_joint=True,
    )
    init_seconds = time.perf_counter() - init_started
    apply_fn = jax.jit(
        lambda current_variables, current_batch: model.apply(
            current_variables,
            current_batch["features"],
            current_batch["feature_lengths"],
            current_batch["tokens"],
            current_batch["token_lengths"],
            train=False,
            compute_joint=True,
        )
    )
    started = time.perf_counter()
    output = apply_fn(variables, batch)
    jax.block_until_ready(output["joint_logits"])
    compiled_forward_seconds = time.perf_counter() - started
    finite = all(
        bool(np.isfinite(np.asarray(output[name])).all())
        for name in ("encoder", "predictor", "joint_logits")
    )
    report = _base_report("model-smoke")
    report.update(
        {
            "status": "pass" if finite else "fail",
            "preset": args.preset,
            "compatibility_claim": "shape-fitment-only",
            "parameter_count": parameter_count(variables["params"]),
            "input_shapes": {name: list(value.shape) for name, value in batch.items()},
            "output_shapes": {
                "encoder": list(output["encoder"].shape),
                "predictor": list(output["predictor"].shape),
                "joint_logits": list(output["joint_logits"].shape),
            },
            "encoder_lengths": output["encoder_lengths"],
            "predictor_lengths": output["predictor_lengths"],
            "compute_dtype": config.encoder.compute_dtype,
            "init_seconds": init_seconds,
            "compiled_forward_seconds": compiled_forward_seconds,
            "parity_gaps": list(PARITY_GAPS),
        }
    )
    _print_report(report)
    return 0 if finite else 1


def command_audio_smoke(args: argparse.Namespace) -> int:
    audio_path = args.audio.expanduser().resolve()
    transcript_path = args.reference_transcript_file.expanduser().resolve()
    audio_sha256 = _sha256(audio_path)
    if args.expected_audio_sha256 is not None:
        expected_sha256 = args.expected_audio_sha256.lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("--expected-audio-sha256 must contain 64 hexadecimal characters")
        if audio_sha256 != expected_sha256:
            raise ValueError(
                f"audio SHA-256 mismatch: expected {expected_sha256}, got {audio_sha256}"
            )

    loaded = load_pcm16_wav(audio_path)
    reference, normalized_reference = _reference_transcript(transcript_path)
    config = AudioConfig(sample_rate=loaded.sample_rate)
    frontend = jax.jit(lambda samples, lengths: log_mel_spectrogram(samples, lengths, config))
    started = time.perf_counter()
    features, feature_lengths = frontend(
        jnp.asarray(loaded.samples), jnp.asarray(loaded.sample_lengths)
    )
    jax.block_until_ready(features)
    elapsed = time.perf_counter() - started
    features_np = np.asarray(features)
    feature_lengths_np = np.asarray(feature_lengths)
    finite = bool(np.isfinite(features_np).all())
    nonempty = bool(feature_lengths_np.size == 1 and feature_lengths_np[0] > 0)
    expected_shape = (
        1,
        max(1, 1 + (loaded.frame_count - config.win_length) // config.hop_length),
        config.n_mels,
    )
    passed = finite and nonempty and features_np.shape == expected_shape

    artifact: dict[str, Any] | None = None
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        if output_path.exists():
            raise FileExistsError(f"refusing to replace existing audio artifact: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            samples=loaded.samples,
            sample_lengths=loaded.sample_lengths,
            features=features_np,
            feature_lengths=feature_lengths_np,
            audio_sha256=np.asarray(audio_sha256),
            reference_transcript=np.asarray(reference),
            normalized_reference_transcript=np.asarray(normalized_reference),
            frontend_config_json=np.asarray(json.dumps(asdict(config), sort_keys=True)),
        )
        artifact = {
            "path": str(output_path),
            "sha256": _sha256(output_path),
        }

    report = _base_report("audio-smoke")
    report.update(
        {
            "status": "pass" if passed else "fail",
            "validation_scope": "audio-ingestion-and-jax-frontend-only",
            "asr_decode_executed": False,
            "asr_decode_blocker": (
                "requires converted Parakeet checkpoint, matching tokenizer, and RNN-T decoder"
            ),
            "audio": {
                "path": str(audio_path),
                "sha256": audio_sha256,
                "sample_rate_hz": loaded.sample_rate,
                "channels": loaded.channels,
                "sample_width_bits": loaded.sample_width_bytes * 8,
                "sample_count": loaded.frame_count,
                "duration_seconds": loaded.duration_seconds,
                "peak_absolute_amplitude": float(np.max(np.abs(loaded.samples))),
                "rms_amplitude": float(np.sqrt(np.mean(np.square(loaded.samples)))),
            },
            "reference": {
                "path": str(transcript_path),
                "sha256": _sha256(transcript_path),
                "role": "operator-supplied-ground-truth-not-model-output",
                "text": reference,
                "normalized_text": normalized_reference,
                "word_count": len(normalized_reference.split()),
            },
            "frontend": {
                "config": asdict(config),
                "feature_shape": list(features_np.shape),
                "feature_lengths": feature_lengths_np,
                "dtype": str(features_np.dtype),
                "finite": finite,
                "compile_and_execute_seconds": elapsed,
            },
            "artifact": artifact,
        }
    )
    _print_report(report)
    return 0 if passed else 1


def _block_metrics(metrics: dict[str, jax.Array]) -> None:
    jax.block_until_ready(metrics["loss"])


def command_train_smoke(args: argparse.Namespace) -> int:
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    workdir = args.workdir.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = workdir / "checkpoint"
    if checkpoint_dir.exists():
        raise FileExistsError(f"refusing to replace previous smoke checkpoint: {checkpoint_dir}")
    config = ParakeetConfig.tiny(vocab_size=8)
    model = ParakeetRNNT(config)
    batch = make_synthetic_batch(config, feature_frames=17, target_steps=2, seed=11)
    state = create_train_state(
        model,
        jax.random.key(11),
        batch,
        optimizer=OptimizerConfig(learning_rate=3e-3, weight_decay=0.0),
    )
    initial_loss = float(eval_loss(state, batch, model=model))
    step_fn = make_train_step(model)
    losses: list[float] = []
    started = time.perf_counter()
    all_finite = math.isfinite(initial_loss)
    for _ in range(args.steps):
        state, metrics = step_fn(state, batch)
        _block_metrics(metrics)
        current = float(metrics["loss"])
        losses.append(current)
        all_finite = all_finite and math.isfinite(current) and bool(metrics["gradients_finite"])
    elapsed = time.perf_counter() - started
    final_loss = float(eval_loss(state, batch, model=model))
    loss_ratio = final_loss / initial_loss
    maximum_ratio = 0.50
    passed = all_finite and math.isfinite(final_loss) and loss_ratio <= maximum_ratio
    metadata = save_checkpoint(
        checkpoint_dir,
        state,
        metadata={
            "command": "train-smoke",
            "steps": args.steps,
            "preset": "tiny",
            "initial_loss": initial_loss,
            "final_loss": final_loss,
        },
    )
    report = _base_report("train-smoke")
    report.update(
        {
            "status": "pass" if passed else "fail",
            "steps": args.steps,
            "workdir": str(workdir),
            "checkpoint": str(checkpoint_dir),
            "checkpoint_tree_sha256": metadata["tree_sha256"],
            "parameter_count": parameter_count(state.params),
            "input_shapes": {name: list(value.shape) for name, value in batch.items()},
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_ratio": loss_ratio,
            "required_loss_ratio_max": maximum_ratio,
            "all_finite": all_finite,
            "elapsed_seconds": elapsed,
            "steps_per_second_including_compile": args.steps / elapsed,
            "first_step_loss": losses[0],
            "last_step_training_loss": losses[-1],
        }
    )
    (workdir / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _print_report(report)
    return 0 if passed else 1


def _memory_stats() -> dict[str, Any] | None:
    try:
        stats = jax.local_devices()[0].memory_stats()
    except (AttributeError, RuntimeError, TypeError):
        return None
    if not stats:
        return None
    return {
        str(key): value.item() if hasattr(value, "item") else value
        for key, value in stats.items()
        if isinstance(value, (str, int, float, bool, np.number))
    }


def command_benchmark(args: argparse.Namespace) -> int:
    if args.steps <= 0 or args.warmup < 0:
        raise ValueError("--steps must be positive and --warmup non-negative")
    config = _config_for_preset(args.preset)
    frames = args.feature_frames
    targets = args.target_steps
    if frames is None:
        frames = 64 if args.preset == "tiny" else 256
    if targets is None:
        targets = 8 if args.preset == "tiny" else 32
    model = ParakeetRNNT(config)
    batch = make_synthetic_batch(
        config,
        batch_size=args.batch_size,
        feature_frames=frames,
        target_steps=targets,
        seed=23,
    )
    init_started = time.perf_counter()
    state = create_train_state(
        model,
        jax.random.key(23),
        batch,
        optimizer=OptimizerConfig(learning_rate=1e-4, weight_decay=0.0),
    )
    init_seconds = time.perf_counter() - init_started
    step_fn = make_train_step(model)

    compile_started = time.perf_counter()
    state, metrics = step_fn(state, batch)
    _block_metrics(metrics)
    compile_and_first_step_seconds = time.perf_counter() - compile_started
    finite = bool(metrics["gradients_finite"]) and math.isfinite(float(metrics["loss"]))

    for _ in range(args.warmup):
        state, metrics = step_fn(state, batch)
        _block_metrics(metrics)
        finite = finite and bool(metrics["gradients_finite"]) and math.isfinite(float(metrics["loss"]))

    durations: list[float] = []
    losses: list[float] = []
    for _ in range(args.steps):
        started = time.perf_counter()
        state, metrics = step_fn(state, batch)
        _block_metrics(metrics)
        durations.append(time.perf_counter() - started)
        losses.append(float(metrics["loss"]))
        finite = finite and bool(metrics["gradients_finite"]) and math.isfinite(losses[-1])

    times = np.asarray(durations)
    mean_seconds = float(times.mean())
    audio_seconds_per_step = args.batch_size * frames * 0.01
    report = _base_report("benchmark")
    report.update(
        {
            "status": "pass" if finite else "fail",
            "preset": args.preset,
            "step_scope": "encoder+predictor+streamed-joint+rnnt-loss+backward+optimizer",
            "compatibility_claim": "synthetic-shape-fitment-not-checkpoint-or-wer-parity",
            "parameter_count": parameter_count(state.params),
            "batch_size": args.batch_size,
            "feature_frames": frames,
            "target_steps": targets,
            "feature_frame_seconds_assumption": 0.01,
            "input_shapes": {name: list(value.shape) for name, value in batch.items()},
            "compute_dtype": config.encoder.compute_dtype,
            "init_seconds": init_seconds,
            "compile_and_first_step_seconds": compile_and_first_step_seconds,
            "warmup_steps_after_compile": args.warmup,
            "measured_steps": args.steps,
            "mean_step_seconds": mean_seconds,
            "median_step_seconds": float(np.median(times)),
            "p95_step_seconds": float(np.percentile(times, 95)),
            "steps_per_second": 1.0 / mean_seconds,
            "audio_seconds_per_chip_second": audio_seconds_per_step / mean_seconds,
            "first_measured_loss": losses[0],
            "last_measured_loss": losses[-1],
            "all_finite": finite,
            "device_memory_stats": _memory_stats(),
        }
    )
    _print_report(report)
    return 0 if finite else 1


def _tree_max_abs_difference(left: Any, right: Any) -> float:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    if left_structure != right_structure:
        return float("inf")
    maximum = 0.0
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_dtype = getattr(left_leaf, "dtype", None)
        if hasattr(jax.dtypes, "prng_key") and jax.dtypes.issubdtype(
            left_dtype, jax.dtypes.prng_key
        ):
            left_value = np.asarray(jax.device_get(jax.random.key_data(left_leaf)))
            right_value = np.asarray(jax.device_get(jax.random.key_data(right_leaf)))
            difference = 0.0 if np.array_equal(left_value, right_value) else np.inf
        else:
            left_value = np.asarray(jax.device_get(left_leaf))
            right_value = np.asarray(jax.device_get(right_leaf))
            if np.issubdtype(left_value.dtype, np.inexact):
                difference = np.max(
                    np.abs(left_value.astype(np.float64) - right_value.astype(np.float64))
                )
            else:
                difference = (
                    0.0 if np.array_equal(left_value, right_value) else np.inf
                )
        maximum = max(maximum, float(difference))
    return maximum


def command_checkpoint_smoke(args: argparse.Namespace) -> int:
    directory = args.directory.expanduser().resolve()
    if directory.exists():
        raise FileExistsError(f"refusing to replace existing checkpoint: {directory}")
    config = ParakeetConfig.tiny(vocab_size=8)
    model = ParakeetRNNT(config)
    batch = make_synthetic_batch(config, feature_frames=17, target_steps=2, seed=31)
    state = create_train_state(model, jax.random.key(31), batch)
    step_fn = make_train_step(model)
    saved_state, first_metrics = step_fn(state, batch)
    _block_metrics(first_metrics)
    control_state, control_metrics = step_fn(saved_state, batch)
    _block_metrics(control_metrics)

    written_metadata = save_checkpoint(
        directory,
        saved_state,
        metadata={"command": "checkpoint-smoke", "saved_step": int(saved_state.step)},
    )
    restored_state, restored_metadata = restore_checkpoint(directory, target=saved_state)
    resumed_state, resumed_metrics = step_fn(restored_state, batch)
    _block_metrics(resumed_metrics)
    max_difference = _tree_max_abs_difference(control_state, resumed_state)
    loss_difference = abs(float(control_metrics["loss"]) - float(resumed_metrics["loss"]))
    tolerance = 1e-6
    passed = (
        int(restored_state.step) == int(saved_state.step)
        and int(resumed_state.step) == int(saved_state.step) + 1
        and max_difference <= tolerance
        and loss_difference <= tolerance
        and written_metadata == restored_metadata
    )
    report = _base_report("checkpoint-smoke")
    report.update(
        {
            "status": "pass" if passed else "fail",
            "directory": str(directory),
            "saved_step": int(saved_state.step),
            "restored_step": int(restored_state.step),
            "resumed_step": int(resumed_state.step),
            "control_next_loss": float(control_metrics["loss"]),
            "resumed_next_loss": float(resumed_metrics["loss"]),
            "next_loss_abs_difference": loss_difference,
            "state_max_abs_difference": max_difference,
            "tolerance": tolerance,
            "tree_sha256": restored_metadata["tree_sha256"],
            "restored_tree_sha256": tree_fingerprint(restored_state),
        }
    )
    _print_report(report)
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fcrnnt_jax.cli",
        description="FastConformer RNN-T JAX/TPU qualification commands",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    devices = subparsers.add_parser("devices", help="report the active JAX devices")
    devices.set_defaults(handler=command_devices)

    loss = subparsers.add_parser("loss-smoke", help="validate classic RNN-T value and gradients")
    loss.add_argument("--fixture", type=Path, help="external NeMo/reference .npz fixture")
    loss.set_defaults(handler=command_loss_smoke)

    model = subparsers.add_parser("model-smoke", help="compile and run model forward topology")
    model.add_argument("--preset", choices=("tiny", "parakeet-1.1b"), default="tiny")
    model.set_defaults(handler=command_model_smoke)

    audio = subparsers.add_parser(
        "audio-smoke",
        help="validate PCM WAV ingestion and the JAX frontend (does not decode ASR)",
    )
    audio.add_argument("--audio", type=Path, required=True, help="mono 16-kHz PCM16 WAV")
    audio.add_argument(
        "--reference-transcript-file",
        type=Path,
        required=True,
        help="UTF-8 ground-truth transcript; it is not treated as model output",
    )
    audio.add_argument(
        "--expected-audio-sha256",
        help="optional pinned SHA-256 that must match before audio is decoded",
    )
    audio.add_argument("--output", type=Path, help="optional frontend evidence .npz")
    audio.set_defaults(handler=command_audio_smoke)

    train = subparsers.add_parser("train-smoke", help="overfit one deterministic tiny batch")
    train.add_argument("--steps", type=int, default=100)
    train.add_argument("--workdir", type=Path, default=Path("artifacts/train-smoke"))
    train.set_defaults(handler=command_train_smoke)

    benchmark = subparsers.add_parser("benchmark", help="time complete post-compile training steps")
    benchmark.add_argument("--preset", choices=("tiny", "parakeet-1.1b"), default="tiny")
    benchmark.add_argument("--steps", type=int, default=20)
    benchmark.add_argument("--warmup", type=int, default=3)
    benchmark.add_argument("--batch-size", type=int, default=1)
    benchmark.add_argument("--feature-frames", type=int)
    benchmark.add_argument("--target-steps", type=int)
    benchmark.set_defaults(handler=command_benchmark)

    checkpoint = subparsers.add_parser("checkpoint-smoke", help="save, restore, and resume exactly")
    checkpoint.add_argument("--directory", type=Path, default=Path("artifacts/checkpoint-smoke"))
    checkpoint.set_defaults(handler=command_checkpoint_smoke)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # Operator logs need both a traceback and final JSON.
        traceback.print_exc()
        _print_report(
            {
                **_base_report(args.command),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
