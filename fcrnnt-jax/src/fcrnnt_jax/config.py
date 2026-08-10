"""Configuration objects for the FastConformer RNN-T proof of concept.

The production factory mirrors the public ``nvidia/parakeet-rnnt-1.1b``
architecture metadata.  It intentionally describes the neural network only;
audio feature extraction, optimization, sharding, and checkpoint conversion are
separate concerns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp


_DTYPES: dict[str, Any] = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
}


def resolve_dtype(name: str) -> Any:
    """Resolve a serialized dtype name to a JAX dtype."""

    try:
        return _DTYPES[name]
    except KeyError as exc:
        allowed = ", ".join(sorted(_DTYPES))
        raise ValueError(f"unsupported dtype {name!r}; expected one of: {allowed}") from exc


@dataclass(frozen=True)
class FastConformerConfig:
    """FastConformer encoder dimensions and regularization."""

    num_mel_bins: int = 80
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 42
    num_attention_heads: int = 8
    conv_kernel_size: int = 9
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    subsampling_conv_kernel_size: int = 3
    subsampling_conv_stride: int = 2
    max_position_embeddings: int = 5000
    hidden_activation: str = "silu"
    dropout_rate: float = 0.1
    activation_dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    position_dropout_rate: float = 0.0
    layerdrop_rate: float = 0.1
    attention_bias: bool = True
    convolution_bias: bool = True
    scale_input: bool = True
    batch_norm_momentum: float = 0.9
    batch_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    compute_dtype: str = "bfloat16"
    param_dtype: str = "float32"

    def __post_init__(self) -> None:
        positive = {
            "num_mel_bins": self.num_mel_bins,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "conv_kernel_size": self.conv_kernel_size,
            "subsampling_factor": self.subsampling_factor,
            "subsampling_conv_channels": self.subsampling_conv_channels,
            "subsampling_conv_kernel_size": self.subsampling_conv_kernel_size,
            "subsampling_conv_stride": self.subsampling_conv_stride,
            "max_position_embeddings": self.max_position_embeddings,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.hidden_size % 2:
            raise ValueError("hidden_size must be even for sinusoidal relative positions")
        if self.conv_kernel_size % 2 != 1:
            raise ValueError("conv_kernel_size must be odd for length-preserving convolution")
        if self.subsampling_conv_kernel_size % 2 != 1:
            raise ValueError("subsampling_conv_kernel_size must be odd")
        if self.subsampling_factor & (self.subsampling_factor - 1):
            raise ValueError("subsampling_factor must be a power of two")
        stages = self.subsampling_factor.bit_length() - 1
        if self.subsampling_conv_stride**stages != self.subsampling_factor:
            raise ValueError(
                "subsampling_factor must equal subsampling_conv_stride ** log2(subsampling_factor)"
            )
        for name, value in {
            "dropout_rate": self.dropout_rate,
            "activation_dropout_rate": self.activation_dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "position_dropout_rate": self.position_dropout_rate,
            "layerdrop_rate": self.layerdrop_rate,
            "batch_norm_momentum": self.batch_norm_momentum,
        }.items():
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}")
        if self.batch_norm_epsilon <= 0.0:
            raise ValueError("batch_norm_epsilon must be positive")
        if self.initializer_range <= 0.0:
            raise ValueError("initializer_range must be positive")
        if self.hidden_activation not in {"silu", "relu"}:
            raise ValueError("hidden_activation must be 'silu' or 'relu'")
        resolve_dtype(self.compute_dtype)
        resolve_dtype(self.param_dtype)

    @property
    def subsampling_stages(self) -> int:
        return self.subsampling_factor.bit_length() - 1

    @property
    def subsampled_mel_bins(self) -> int:
        """Frequency width after the same-padded strided convolutions."""

        size = self.num_mel_bins
        padding = (self.subsampling_conv_kernel_size - 1) // 2
        for _ in range(self.subsampling_stages):
            size = (
                size + 2 * padding - self.subsampling_conv_kernel_size
            ) // self.subsampling_conv_stride + 1
        return size


@dataclass(frozen=True)
class PredictorConfig:
    """RNN-T prediction-network dimensions."""

    hidden_size: int = 640
    num_layers: int = 2
    dropout_rate: float = 0.0
    use_bias: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("predictor hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("predictor num_layers must be positive")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("predictor dropout_rate must be in [0, 1)")


@dataclass(frozen=True)
class ParakeetConfig:
    """Complete FastConformer RNN-T model configuration.

    ``vocab_size`` includes the RNN-T blank class.  For the public 1.1B
    checkpoint, text pieces occupy ids 0..1023 and blank is id 1024.
    """

    vocab_size: int = 1025
    blank_id: int = 1024
    pad_id: int = 0
    joint_hidden_size: int = 640
    joint_activation: str = "relu"
    max_symbols_per_step: int = 10
    encoder: FastConformerConfig = field(default_factory=FastConformerConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)

    def __post_init__(self) -> None:
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must include at least one token and blank")
        if not 0 <= self.blank_id < self.vocab_size:
            raise ValueError("blank_id must be in [0, vocab_size)")
        if not 0 <= self.pad_id < self.vocab_size:
            raise ValueError("pad_id must be in [0, vocab_size)")
        if self.joint_hidden_size <= 0:
            raise ValueError("joint_hidden_size must be positive")
        if self.joint_activation not in {"relu", "silu"}:
            raise ValueError("joint_activation must be 'relu' or 'silu'")
        if self.max_symbols_per_step <= 0:
            raise ValueError("max_symbols_per_step must be positive")

    @classmethod
    def parakeet_1_1b(cls) -> "ParakeetConfig":
        """Return the public Parakeet RNN-T 1.1B architectural dimensions."""

        return cls(
            vocab_size=1025,
            blank_id=1024,
            pad_id=0,
            joint_hidden_size=640,
            joint_activation="relu",
            max_symbols_per_step=10,
            encoder=FastConformerConfig(
                num_mel_bins=80,
                hidden_size=1024,
                intermediate_size=4096,
                num_hidden_layers=42,
                num_attention_heads=8,
                conv_kernel_size=9,
                subsampling_factor=8,
                subsampling_conv_channels=256,
                subsampling_conv_kernel_size=3,
                subsampling_conv_stride=2,
                max_position_embeddings=5000,
                hidden_activation="silu",
                dropout_rate=0.1,
                activation_dropout_rate=0.1,
                attention_dropout_rate=0.1,
                position_dropout_rate=0.0,
                layerdrop_rate=0.1,
                attention_bias=True,
                convolution_bias=True,
                scale_input=True,
                compute_dtype="bfloat16",
                param_dtype="float32",
            ),
            predictor=PredictorConfig(hidden_size=640, num_layers=2),
        )

    @classmethod
    def tiny(
        cls,
        vocab_size: int = 16,
        blank_id: int | None = None,
    ) -> "ParakeetConfig":
        """Return a CPU-friendly config that preserves all architectural paths."""

        resolved_blank = vocab_size - 1 if blank_id is None else blank_id
        return cls(
            vocab_size=vocab_size,
            blank_id=resolved_blank,
            pad_id=0,
            joint_hidden_size=16,
            joint_activation="relu",
            max_symbols_per_step=4,
            encoder=FastConformerConfig(
                num_mel_bins=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                conv_kernel_size=9,
                subsampling_factor=8,
                subsampling_conv_channels=4,
                subsampling_conv_kernel_size=3,
                subsampling_conv_stride=2,
                max_position_embeddings=256,
                hidden_activation="silu",
                dropout_rate=0.0,
                activation_dropout_rate=0.0,
                attention_dropout_rate=0.0,
                position_dropout_rate=0.0,
                layerdrop_rate=0.0,
                scale_input=True,
                compute_dtype="float32",
                param_dtype="float32",
            ),
            predictor=PredictorConfig(hidden_size=16, num_layers=2),
        )


__all__ = [
    "FastConformerConfig",
    "ParakeetConfig",
    "PredictorConfig",
    "resolve_dtype",
]
