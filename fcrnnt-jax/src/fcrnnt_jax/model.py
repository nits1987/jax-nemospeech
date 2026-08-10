"""Flax implementation of the Parakeet FastConformer RNN-T model skeleton.

Tensor contracts
----------------
* Acoustic features: ``[batch, frames, mel_bins]``.
* Acoustic lengths: ``[batch]`` in pre-subsampling frames.
* Encoder output: ``[batch, ceil(frames / 8), encoder_dim]``.
* Transcript tokens: ``[batch, target_steps]`` without the RNN-T start blank.
* Predictor output: ``[batch, target_steps + 1, predictor_dim]``.  A blank is
  prepended internally, matching the standard RNN-T lattice convention.
* Full joint logits (opt-in): ``[batch, encoder_steps, target_steps + 1,
  vocab_size]``.

The full joint lattice is disabled by default because production RNN-T loss
must consume one acoustic frame/chunk at a time.  ``joint`` returns raw logits
for that path; ``joint_frame_log_probs`` is a convenience wrapper.

Remaining parity work is deliberately visible in ``PARITY_GAPS``.  This file
implements the complete topology but does not claim checkpoint or WER parity
until those items have passed against the pinned NeMo reference.
"""

from __future__ import annotations

import math
from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp

from .config import FastConformerConfig, ParakeetConfig, PredictorConfig, resolve_dtype


Array = jax.Array


PARITY_GAPS = (
    "NeMo-to-Flax parameter conversion and tensor-by-tensor golden tests are not implemented here.",
    "Training-time LayerDrop selects the residual after evaluating the block, so a dropped block can still update "
    "BatchNorm statistics; disable LayerDrop for strict update-parity tests until control flow is ported.",
    "The LSTM stores one combined bias per layer; PyTorch/NeMo input and recurrent biases must be summed on import.",
    "Streaming/local attention, predictor-state caching, and greedy/beam decoding are outside this PoC model slice.",
    "Audio feature extraction and augmentation are external and require their own NeMo parity tests.",
)


def sequence_mask(lengths: Array, max_length: int) -> Array:
    """Return a boolean ``[batch, max_length]`` mask."""

    return jnp.arange(max_length)[None, :] < lengths[:, None]


def subsample_lengths(lengths: Array, config: FastConformerConfig) -> Array:
    """Apply the exact same-padded Conv2D length formula used by Parakeet."""

    result = lengths.astype(jnp.int32)
    padding = (config.subsampling_conv_kernel_size - 1) // 2
    for _ in range(config.subsampling_stages):
        result = (
            result + 2 * padding - config.subsampling_conv_kernel_size
        ) // config.subsampling_conv_stride + 1
    return jnp.maximum(result, 0)


def _activation(name: str, value: Array) -> Array:
    if name == "silu":
        return nn.silu(value)
    if name == "relu":
        return nn.relu(value)
    raise ValueError(f"unsupported activation: {name}")


def _kernel_init(stddev: float):
    return nn.initializers.normal(stddev=stddev)


class FastConformerSubsampling(nn.Module):
    """Three-stage depthwise-striding Conv2D subsampler for factor 8."""

    config: FastConformerConfig

    @nn.compact
    def __call__(self, features: Array, lengths: Array) -> tuple[Array, Array]:
        cfg = self.config
        dtype = resolve_dtype(cfg.compute_dtype)
        param_dtype = resolve_dtype(cfg.param_dtype)
        padding_size = (cfg.subsampling_conv_kernel_size - 1) // 2
        padding = ((padding_size, padding_size), (padding_size, padding_size))
        kernel = (cfg.subsampling_conv_kernel_size,) * 2
        strides = (cfg.subsampling_conv_stride,) * 2

        input_mask = sequence_mask(lengths, features.shape[1])
        x = jnp.where(input_mask[:, :, None], features, 0).astype(dtype)
        x = x[:, :, :, None]
        current_lengths = lengths.astype(jnp.int32)

        x = nn.Conv(
            features=cfg.subsampling_conv_channels,
            kernel_size=kernel,
            strides=strides,
            padding=padding,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=_kernel_init(cfg.initializer_range),
            name="conv_0",
        )(x)
        current_lengths = (
            current_lengths + 2 * padding_size - cfg.subsampling_conv_kernel_size
        ) // cfg.subsampling_conv_stride + 1
        x = nn.relu(x)
        x = jnp.where(sequence_mask(current_lengths, x.shape[1])[:, :, None, None], x, 0)

        for stage in range(1, cfg.subsampling_stages):
            x = nn.Conv(
                features=cfg.subsampling_conv_channels,
                kernel_size=kernel,
                strides=strides,
                padding=padding,
                feature_group_count=cfg.subsampling_conv_channels,
                use_bias=True,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=_kernel_init(cfg.initializer_range),
                name=f"depthwise_conv_{stage}",
            )(x)
            current_lengths = (
                current_lengths + 2 * padding_size - cfg.subsampling_conv_kernel_size
            ) // cfg.subsampling_conv_stride + 1
            x = nn.Conv(
                features=cfg.subsampling_conv_channels,
                kernel_size=(1, 1),
                padding="VALID",
                use_bias=True,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=_kernel_init(cfg.initializer_range),
                name=f"pointwise_conv_{stage}",
            )(x)
            x = nn.relu(x)
            x = jnp.where(sequence_mask(current_lengths, x.shape[1])[:, :, None, None], x, 0)

        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        x = nn.Dense(
            cfg.hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=_kernel_init(cfg.initializer_range),
            name="output_projection",
        )(x)
        output_mask = sequence_mask(current_lengths, x.shape[1])
        return jnp.where(output_mask[:, :, None], x, 0), jnp.maximum(current_lengths, 0)


def _relative_position_embeddings(length: int, width: int, dtype: Any) -> Array:
    """Transformer-XL sinusoidal positions ordered ``T-1 .. -(T-1)``."""

    positions = jnp.arange(length - 1, -length, -1, dtype=jnp.float32)
    inv_freq = 1.0 / (
        10000.0 ** (jnp.arange(0, width, 2, dtype=jnp.float32) / float(width))
    )
    angles = positions[:, None] * inv_freq[None, :]
    embeddings = jnp.stack((jnp.sin(angles), jnp.cos(angles)), axis=-1)
    return embeddings.reshape(2 * length - 1, width).astype(dtype)


def _relative_shift(scores: Array) -> Array:
    """Align Transformer-XL relative logits with content logits."""

    batch, heads, query_length, position_length = scores.shape
    padded = jnp.pad(scores, ((0, 0), (0, 0), (0, 0), (1, 0)))
    shifted = padded.reshape(batch, heads, -1, query_length)[:, :, 1:, :]
    return shifted.reshape(batch, heads, query_length, position_length)


class RelativePositionAttention(nn.Module):
    """Non-causal Transformer-XL relative-position multi-head attention."""

    config: FastConformerConfig

    @nn.compact
    def __call__(self, x: Array, mask: Array, *, train: bool) -> Array:
        cfg = self.config
        dtype = resolve_dtype(cfg.compute_dtype)
        param_dtype = resolve_dtype(cfg.param_dtype)
        num_heads = cfg.num_attention_heads
        head_dim = cfg.hidden_size // num_heads
        dense_kwargs = dict(
            features=cfg.hidden_size,
            use_bias=cfg.attention_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=_kernel_init(cfg.initializer_range),
        )

        query = nn.Dense(name="query", **dense_kwargs)(x)
        key = nn.Dense(name="key", **dense_kwargs)(x)
        value = nn.Dense(name="value", **dense_kwargs)(x)
        query = query.reshape(x.shape[0], x.shape[1], num_heads, head_dim).transpose(0, 2, 1, 3)
        key = key.reshape(x.shape[0], x.shape[1], num_heads, head_dim).transpose(0, 2, 1, 3)
        value = value.reshape(x.shape[0], x.shape[1], num_heads, head_dim).transpose(0, 2, 1, 3)

        bias_u = self.param(
            "bias_u",
            _kernel_init(cfg.initializer_range),
            (num_heads, head_dim),
            param_dtype,
        ).astype(dtype)
        bias_v = self.param(
            "bias_v",
            _kernel_init(cfg.initializer_range),
            (num_heads, head_dim),
            param_dtype,
        ).astype(dtype)

        positions = _relative_position_embeddings(x.shape[1], cfg.hidden_size, dtype)
        relative_keys = nn.Dense(
            cfg.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=_kernel_init(cfg.initializer_range),
            name="relative_key",
        )(positions)
        relative_keys = relative_keys.reshape(2 * x.shape[1] - 1, num_heads, head_dim)

        content_scores = jnp.einsum(
            "bhtd,bhsd->bhts", query + bias_u[None, :, None, :], key
        )
        position_scores = jnp.einsum(
            "bhtd,rhd->bhtr", query + bias_v[None, :, None, :], relative_keys
        )
        position_scores = _relative_shift(position_scores)[..., : x.shape[1]]
        scores = (content_scores + position_scores) / math.sqrt(head_dim)

        key_mask = mask[:, None, None, :]
        scores = jnp.where(key_mask, scores, jnp.asarray(-1e30, scores.dtype))
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(dtype)
        weights = nn.Dropout(rate=cfg.attention_dropout_rate, name="attention_dropout")(
            weights, deterministic=not train
        )
        attended = jnp.einsum("bhts,bhsd->bhtd", weights, value)
        attended = attended.transpose(0, 2, 1, 3).reshape(x.shape)
        output = nn.Dense(name="output", **dense_kwargs)(attended)
        return jnp.where(mask[:, :, None], output, 0)


class FeedForward(nn.Module):
    config: FastConformerConfig

    @nn.compact
    def __call__(self, x: Array, *, train: bool) -> Array:
        cfg = self.config
        dtype = resolve_dtype(cfg.compute_dtype)
        param_dtype = resolve_dtype(cfg.param_dtype)
        x = nn.Dense(
            cfg.intermediate_size,
            use_bias=cfg.attention_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=_kernel_init(cfg.initializer_range),
            name="linear_1",
        )(x)
        x = _activation(cfg.hidden_activation, x)
        x = nn.Dropout(rate=cfg.activation_dropout_rate, name="activation_dropout")(
            x, deterministic=not train
        )
        return nn.Dense(
            cfg.hidden_size,
            use_bias=cfg.attention_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=_kernel_init(cfg.initializer_range),
            name="linear_2",
        )(x)


class ConformerConvolution(nn.Module):
    config: FastConformerConfig

    @nn.compact
    def __call__(self, x: Array, mask: Array, *, train: bool) -> Array:
        cfg = self.config
        dtype = resolve_dtype(cfg.compute_dtype)
        param_dtype = resolve_dtype(cfg.param_dtype)
        init = _kernel_init(cfg.initializer_range)

        x = nn.Dense(
            2 * cfg.hidden_size,
            use_bias=cfg.convolution_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init,
            name="pointwise_conv_1",
        )(x)
        gate, value = jnp.split(x, 2, axis=-1)
        x = gate * jax.nn.sigmoid(value)
        x = jnp.where(mask[:, :, None], x, 0)
        x = nn.Conv(
            cfg.hidden_size,
            kernel_size=(cfg.conv_kernel_size,),
            padding="SAME",
            feature_group_count=cfg.hidden_size,
            use_bias=cfg.convolution_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init,
            name="depthwise_conv",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=cfg.batch_norm_momentum,
            epsilon=cfg.batch_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            name="batch_norm",
        )(x)
        x = _activation(cfg.hidden_activation, x)
        x = nn.Dense(
            cfg.hidden_size,
            use_bias=cfg.convolution_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init,
            name="pointwise_conv_2",
        )(x)
        return jnp.where(mask[:, :, None], x, 0)


class FastConformerBlock(nn.Module):
    config: FastConformerConfig

    @nn.compact
    def __call__(self, x: Array, mask: Array, *, train: bool) -> Array:
        cfg = self.config
        dtype = resolve_dtype(cfg.compute_dtype)
        param_dtype = resolve_dtype(cfg.param_dtype)
        norm = dict(dtype=dtype, param_dtype=param_dtype, epsilon=1e-5)
        block_input = x

        y = nn.LayerNorm(name="norm_feed_forward_1", **norm)(x)
        x = x + 0.5 * FeedForward(cfg, name="feed_forward_1")(y, train=train)
        x = jnp.where(mask[:, :, None], x, 0)

        y = nn.LayerNorm(name="norm_self_attention", **norm)(x)
        x = x + RelativePositionAttention(cfg, name="self_attention")(y, mask, train=train)
        x = jnp.where(mask[:, :, None], x, 0)

        y = nn.LayerNorm(name="norm_convolution", **norm)(x)
        x = x + ConformerConvolution(cfg, name="convolution")(y, mask, train=train)
        x = jnp.where(mask[:, :, None], x, 0)

        y = nn.LayerNorm(name="norm_feed_forward_2", **norm)(x)
        x = x + 0.5 * FeedForward(cfg, name="feed_forward_2")(y, train=train)
        x = nn.LayerNorm(name="norm_output", **norm)(x)
        x = jnp.where(mask[:, :, None], x, 0)

        if train and cfg.layerdrop_rate:
            keep = jax.random.bernoulli(
                self.make_rng("dropout"), p=1.0 - cfg.layerdrop_rate
            )
            x = jnp.where(keep, x, block_input)
        return jnp.where(mask[:, :, None], x, 0)


class FastConformerEncoder(nn.Module):
    config: FastConformerConfig

    @nn.compact
    def __call__(self, features: Array, lengths: Array, *, train: bool) -> tuple[Array, Array]:
        cfg = self.config
        dtype = resolve_dtype(cfg.compute_dtype)
        x, output_lengths = FastConformerSubsampling(cfg, name="subsampling")(
            features, lengths
        )
        if cfg.scale_input:
            x = x * math.sqrt(cfg.hidden_size)
        x = nn.Dropout(rate=cfg.dropout_rate, name="input_dropout")(
            x, deterministic=not train
        )
        mask = sequence_mask(output_lengths, x.shape[1])
        x = jnp.where(mask[:, :, None], x.astype(dtype), 0)
        for layer_index in range(cfg.num_hidden_layers):
            x = FastConformerBlock(cfg, name=f"layer_{layer_index}")(
                x, mask, train=train
            )
        return jnp.where(mask[:, :, None], x, 0), output_lengths


class LSTMLayer(nn.Module):
    """Batch-major LSTM layer with PyTorch-compatible IFGO gate ordering."""

    hidden_size: int
    use_bias: bool
    compute_dtype: str
    param_dtype: str
    initializer_range: float

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        dtype = resolve_dtype(self.compute_dtype)
        param_dtype = resolve_dtype(self.param_dtype)
        input_size = inputs.shape[-1]
        input_kernel = self.param(
            "input_kernel",
            _kernel_init(self.initializer_range),
            (input_size, 4 * self.hidden_size),
            param_dtype,
        ).astype(dtype)
        recurrent_kernel = self.param(
            "recurrent_kernel",
            _kernel_init(self.initializer_range),
            (self.hidden_size, 4 * self.hidden_size),
            param_dtype,
        ).astype(dtype)
        if self.use_bias:
            bias = self.param(
                "bias", nn.initializers.zeros, (4 * self.hidden_size,), param_dtype
            ).astype(dtype)
        else:
            bias = jnp.zeros((4 * self.hidden_size,), dtype=dtype)

        batch = inputs.shape[0]
        initial = (
            jnp.zeros((batch, self.hidden_size), dtype=dtype),
            jnp.zeros((batch, self.hidden_size), dtype=dtype),
        )

        def step(carry: tuple[Array, Array], current: Array):
            cell, hidden = carry
            gates = current @ input_kernel + hidden @ recurrent_kernel + bias
            input_gate, forget_gate, candidate, output_gate = jnp.split(gates, 4, axis=-1)
            cell = jax.nn.sigmoid(forget_gate) * cell + jax.nn.sigmoid(
                input_gate
            ) * jnp.tanh(candidate)
            hidden = jax.nn.sigmoid(output_gate) * jnp.tanh(cell)
            return (cell, hidden), hidden

        _, outputs = jax.lax.scan(step, initial, inputs.swapaxes(0, 1))
        return outputs.swapaxes(0, 1)


class PredictionNetwork(nn.Module):
    config: PredictorConfig
    vocab_size: int
    blank_id: int
    output_size: int
    compute_dtype: str
    param_dtype: str
    initializer_range: float

    @nn.compact
    def __call__(
        self,
        tokens: Array,
        token_lengths: Array | None = None,
        *,
        train: bool,
    ) -> Array:
        if tokens.ndim != 2:
            raise ValueError(f"tokens must have shape [B, U], got {tokens.shape}")
        batch, target_steps = tokens.shape
        if token_lengths is None:
            token_lengths = jnp.full((batch,), target_steps, dtype=jnp.int32)
        if token_lengths.shape != (batch,):
            raise ValueError(
                f"token_lengths must have shape {(batch,)}, got {token_lengths.shape}"
            )

        dtype = resolve_dtype(self.compute_dtype)
        param_dtype = resolve_dtype(self.param_dtype)
        valid_tokens = sequence_mask(token_lengths, target_steps)
        safe_tokens = jnp.where(valid_tokens, tokens, 0)
        start = jnp.full((batch, 1), self.blank_id, dtype=tokens.dtype)
        decoder_inputs = jnp.concatenate((start, safe_tokens), axis=1)
        x = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=_kernel_init(self.initializer_range),
            name="embedding",
        )(decoder_inputs)

        for layer_index in range(self.config.num_layers):
            x = LSTMLayer(
                hidden_size=self.config.hidden_size,
                use_bias=self.config.use_bias,
                compute_dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
                initializer_range=self.initializer_range,
                name=f"lstm_{layer_index}",
            )(x)
            if layer_index + 1 < self.config.num_layers:
                x = nn.Dropout(
                    rate=self.config.dropout_rate, name=f"dropout_{layer_index}"
                )(x, deterministic=not train)

        x = nn.Dense(
            self.output_size,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=_kernel_init(self.initializer_range),
            name="output_projection",
        )(x)
        predictor_lengths = token_lengths.astype(jnp.int32) + 1
        return jnp.where(sequence_mask(predictor_lengths, x.shape[1])[:, :, None], x, 0)


class ParakeetRNNT(nn.Module):
    """Config-driven FastConformer encoder and classic RNN-T prediction head.

    ``compute_joint=False`` is the safe default.  RNN-T loss code should call
    ``joint_frame_log_probs`` through ``Module.apply(..., method=...)`` for one
    encoder frame at a time.
    """

    config: ParakeetConfig

    def setup(self) -> None:
        cfg = self.config
        dtype = cfg.encoder.compute_dtype
        param_dtype = cfg.encoder.param_dtype
        init = _kernel_init(cfg.encoder.initializer_range)
        self.fastconformer = FastConformerEncoder(cfg.encoder, name="encoder")
        self.prediction_network = PredictionNetwork(
            config=cfg.predictor,
            vocab_size=cfg.vocab_size,
            blank_id=cfg.blank_id,
            output_size=cfg.joint_hidden_size,
            compute_dtype=dtype,
            param_dtype=param_dtype,
            initializer_range=cfg.encoder.initializer_range,
            name="predictor",
        )
        self.encoder_projector = nn.Dense(
            cfg.joint_hidden_size,
            dtype=resolve_dtype(dtype),
            param_dtype=resolve_dtype(param_dtype),
            kernel_init=init,
            name="encoder_projector",
        )
        self.joint_head = nn.Dense(
            cfg.vocab_size,
            dtype=resolve_dtype(dtype),
            param_dtype=resolve_dtype(param_dtype),
            kernel_init=init,
            name="joint_head",
        )

    def encode(self, features: Array, feature_lengths: Array, *, train: bool = False):
        """Encode ``[B,T,F]`` features and return ``(states, ceil(lengths/8))``."""

        if features.ndim != 3:
            raise ValueError(f"features must have shape [B, T, F], got {features.shape}")
        if features.shape[-1] != self.config.encoder.num_mel_bins:
            raise ValueError(
                f"expected {self.config.encoder.num_mel_bins} mel bins, got {features.shape[-1]}"
            )
        if feature_lengths.shape != (features.shape[0],):
            raise ValueError(
                "feature_lengths must have shape [B]; "
                f"expected {(features.shape[0],)}, got {feature_lengths.shape}"
            )
        return self.fastconformer(features, feature_lengths, train=train)

    def predict(
        self,
        tokens: Array,
        token_lengths: Array | None = None,
        *,
        train: bool = False,
    ) -> Array:
        """Return predictor states with a blank/BOS state prepended."""

        return self.prediction_network(tokens, token_lengths, train=train)

    def joint(self, encoder: Array, predictor: Array) -> Array:
        """Materialize joint logits for supplied encoder and predictor states.

        ``encoder`` is ``[B,T,D]`` or ``[B,D]``; ``predictor`` is ``[B,U,H]``
        or ``[B,H]``.  Leading sequence axes are broadcast to produce
        ``[B,T,U,V]`` (or the naturally squeezed frame/step equivalent).
        """

        if encoder.ndim not in (2, 3):
            raise ValueError("encoder must have shape [B,D] or [B,T,D]")
        if predictor.ndim not in (2, 3):
            raise ValueError("predictor must have shape [B,H] or [B,U,H]")
        if encoder.shape[0] != predictor.shape[0]:
            raise ValueError("encoder and predictor batch sizes must match")
        encoder_projected = self.encoder_projector(encoder)
        predictor_projected = predictor
        if encoder.ndim == 3 and predictor.ndim == 3:
            encoder_projected = encoder_projected[:, :, None, :]
            predictor_projected = predictor_projected[:, None, :, :]
        elif encoder.ndim == 2 and predictor.ndim == 3:
            encoder_projected = encoder_projected[:, None, :]
        elif encoder.ndim == 3 and predictor.ndim == 2:
            predictor_projected = predictor_projected[:, None, :]
        hidden = _activation(
            self.config.joint_activation, encoder_projected + predictor_projected
        )
        return self.joint_head(hidden)

    def joint_frame_log_probs(self, encoder_frame: Array, predictor: Array) -> Array:
        """Return fp32 log-probs for one frame without a ``T``-wide lattice.

        Args:
            encoder_frame: ``[B,D]`` (preferred) or ``[B,1,D]``.
            predictor: ``[B,U,H]`` or a single ``[B,H]`` predictor state.
        """

        if encoder_frame.ndim == 3:
            if encoder_frame.shape[1] != 1:
                raise ValueError("encoder_frame with rank 3 must have shape [B,1,D]")
            encoder_frame = encoder_frame[:, 0, :]
        if encoder_frame.ndim != 2:
            raise ValueError("encoder_frame must have shape [B,D] or [B,1,D]")
        return jax.nn.log_softmax(
            self.joint(encoder_frame, predictor).astype(jnp.float32), axis=-1
        )

    def __call__(
        self,
        features: Array,
        feature_lengths: Array,
        tokens: Array,
        token_lengths: Array | None = None,
        *,
        train: bool = False,
        compute_joint: bool = False,
    ) -> dict[str, Any]:
        """Run the encoder/predictor and optionally materialize full logits."""

        if tokens.ndim != 2:
            raise ValueError(f"tokens must have shape [B,U], got {tokens.shape}")
        if tokens.shape[0] != features.shape[0]:
            raise ValueError("features and tokens batch sizes must match")
        if token_lengths is None:
            token_lengths = jnp.full(
                (tokens.shape[0],), tokens.shape[1], dtype=jnp.int32
            )
        encoder, encoder_lengths = self.encode(
            features, feature_lengths, train=train
        )
        predictor = self.predict(tokens, token_lengths, train=train)
        output: dict[str, Any] = {
            "encoder": encoder,
            "encoder_lengths": encoder_lengths,
            "predictor": predictor,
            "predictor_lengths": token_lengths.astype(jnp.int32) + 1,
            "joint_logits": None,
        }
        if compute_joint:
            output["joint_logits"] = self.joint(encoder, predictor)
        else:
            # Linen parameters are lazy.  Touch one lattice cell so a normal
            # ``model.init(..., compute_joint=False)`` still creates the joint
            # parameters needed later by the frame-wise RNN-T loss.
            _ = self.joint(encoder[:, 0, :], predictor[:, 0, :])
        return output


__all__ = [
    "FastConformerEncoder",
    "PARITY_GAPS",
    "ParakeetRNNT",
    "PredictionNetwork",
    "sequence_mask",
    "subsample_lengths",
]
