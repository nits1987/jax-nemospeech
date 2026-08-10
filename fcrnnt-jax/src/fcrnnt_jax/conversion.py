"""Strict, declarative tensor conversion utilities.

This module deliberately separates the *conversion engine* from a model-specific
mapping.  It can read NumPy ``.npz`` and safetensors files, apply an explicit set
of :class:`MappingRule` objects, validate the result against expected shapes or a
target template, and emit a machine-readable audit report.

There is no built-in Parakeet 1.1B mapping here.  Such a mapping must be produced
from the exact NeMo checkpoint/config pair and reviewed independently; the
converter fails closed when a source or target is not accounted for.

The implementation has no JAX dependency.  It returns NumPy arrays in a nested
dictionary that can be frozen or device-put by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


ArrayLike = Any
JsonObject = dict[str, Any]
Transform = Callable[[tuple[np.ndarray, ...], Mapping[str, Any]], np.ndarray]

MAPPING_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1


class ConversionError(ValueError):
    """Base class for deterministic conversion failures."""


class MappingValidationError(ConversionError):
    """Raised before conversion when a mapping is incomplete or ambiguous."""


class TensorTransformError(ConversionError):
    """Raised when a transform cannot be applied to its source tensor(s)."""


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def flatten_pytree(tree: Mapping[str, Any], *, separator: str = ".") -> dict[str, Any]:
    """Flatten a nested string-keyed mapping into path-keyed leaves.

    This intentionally supports mappings only, which matches Flax parameter
    trees without importing Flax.  Empty mappings and keys containing the path
    separator are rejected because they cannot round-trip unambiguously.
    Values are returned unchanged.
    """

    if not separator:
        raise ValueError("separator must be non-empty")
    if not _is_mapping(tree):
        raise TypeError(f"tree must be a mapping, got {type(tree).__name__}")

    flattened: dict[str, Any] = {}

    def visit(node: Mapping[str, Any], prefix: tuple[str, ...]) -> None:
        if not node:
            path = separator.join(prefix) or "<root>"
            raise ValueError(f"empty mapping at {path!r} cannot be represented")
        for raw_key, value in node.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise ValueError("all pytree keys must be non-empty strings")
            if separator in raw_key:
                raise ValueError(
                    f"pytree key {raw_key!r} contains separator {separator!r}"
                )
            path = (*prefix, raw_key)
            if _is_mapping(value):
                visit(value, path)
            else:
                flattened[separator.join(path)] = value

    visit(tree, ())
    return flattened


def unflatten_pytree(
    flattened: Mapping[str, Any], *, separator: str = "."
) -> dict[str, Any]:
    """Build a nested dictionary from path-keyed leaves.

    Prefix collisions (for example, both ``"a"`` and ``"a.b"``) are rejected.
    """

    if not separator:
        raise ValueError("separator must be non-empty")
    if not _is_mapping(flattened):
        raise TypeError("flattened must be a mapping")

    root: dict[str, Any] = {}
    for path, value in flattened.items():
        if not isinstance(path, str) or not path:
            raise ValueError("all flattened paths must be non-empty strings")
        parts = path.split(separator)
        if any(not part for part in parts):
            raise ValueError(f"invalid flattened path {path!r}")

        cursor = root
        for part in parts[:-1]:
            if part not in cursor:
                child: dict[str, Any] = {}
                cursor[part] = child
                cursor = child
            elif _is_mapping(cursor[part]):
                existing = cursor[part]
                cursor = existing  # type: ignore[assignment]
            else:
                raise ValueError(f"path collision while inserting {path!r}")

        leaf = parts[-1]
        if leaf in cursor:
            raise ValueError(f"duplicate or prefix collision at path {path!r}")
        cursor[leaf] = value
    return root


def _as_tensor(value: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ConversionError(f"tensor {name!r} has unsupported object dtype")
    return array


def tensor_sha256(value: ArrayLike) -> str:
    """Return a stable SHA-256 over a tensor's dtype, shape, and C-order bytes."""

    array = _as_tensor(value, name="<unnamed>")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(b"fcrnnt-jax-tensor-v1\0")
    digest.update(contiguous.dtype.str.encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def tensor_set_sha256(tensors: Mapping[str, ArrayLike]) -> str:
    """Return an order-independent digest for a named collection of tensors."""

    digest = hashlib.sha256()
    digest.update(b"fcrnnt-jax-tensor-set-v1\0")
    for name in _sorted_tensor_names(tensors):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor_sha256(tensors[name]).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class TensorInfo:
    """Inventory record for one named tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    sha256: str

    def to_dict(self) -> JsonObject:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "nbytes": self.nbytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TensorInfo":
        return cls(
            name=str(value["name"]),
            shape=tuple(int(dimension) for dimension in value["shape"]),
            dtype=str(value["dtype"]),
            nbytes=int(value["nbytes"]),
            sha256=str(value["sha256"]),
        )


def inventory_tensors(tensors: Mapping[str, ArrayLike]) -> tuple[TensorInfo, ...]:
    """Return sorted shape/dtype/size/hash records for named tensors."""

    records: list[TensorInfo] = []
    for name in _sorted_tensor_names(tensors):
        array = _as_tensor(tensors[name], name=name)
        records.append(
            TensorInfo(
                name=name,
                shape=tuple(int(dimension) for dimension in array.shape),
                dtype=array.dtype.str,
                nbytes=int(array.nbytes),
                sha256=tensor_sha256(array),
            )
        )
    return tuple(records)


def _sorted_tensor_names(tensors: Mapping[str, Any]) -> list[str]:
    names = list(tensors)
    if any(not isinstance(name, str) or not name for name in names):
        raise ConversionError("tensor names must be non-empty strings")
    return sorted(names)


def load_tensor_file(path: str | Path) -> dict[str, np.ndarray]:
    """Load a flat tensor dictionary from ``.npz`` or ``.safetensors``.

    Safetensors support is optional at import time.  Install ``safetensors`` to
    read those files; NumPy-only users can still use the rest of this module.
    """

    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix == ".npz":
        try:
            with np.load(source_path, allow_pickle=False) as archive:
                return {name: np.array(archive[name], copy=True) for name in archive.files}
        except (OSError, ValueError) as error:
            raise ConversionError(f"failed to read NPZ file {source_path}: {error}") from error

    if suffix == ".safetensors":
        try:
            from safetensors.numpy import load_file  # type: ignore[import-not-found]
        except ImportError as error:
            raise ConversionError(
                "reading .safetensors requires the optional 'safetensors' package"
            ) from error
        try:
            loaded = load_file(str(source_path))
        except Exception as error:  # safetensors exposes format-specific exceptions.
            raise ConversionError(
                f"failed to read safetensors file {source_path}: {error}"
            ) from error
        return {name: np.asarray(value) for name, value in loaded.items()}

    raise ConversionError(
        f"unsupported tensor file extension {source_path.suffix!r}; "
        "expected .npz or .safetensors"
    )


def inventory_tensor_file(path: str | Path) -> tuple[TensorInfo, ...]:
    """Load a supported tensor file and return its inventory."""

    return inventory_tensors(load_tensor_file(path))


def _normalize_sources(source: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(source, str):
        sources = (source,)
    else:
        sources = tuple(source)
    if not sources or any(not isinstance(item, str) or not item for item in sources):
        raise ValueError("MappingRule.source must contain non-empty tensor names")
    if len(set(sources)) != len(sources):
        raise ValueError(f"MappingRule.source contains duplicates: {sources!r}")
    return sources


def _normalize_source_shapes(
    shapes: Sequence[Sequence[int]] | None,
) -> tuple[tuple[int, ...], ...] | None:
    if shapes is None:
        return None
    normalized = tuple(tuple(int(dimension) for dimension in shape) for shape in shapes)
    if any(any(dimension < 0 for dimension in shape) for shape in normalized):
        raise ValueError("source shape dimensions must be non-negative")
    return normalized


@dataclass(frozen=True)
class MappingRule:
    """One explicit source-to-target conversion rule.

    ``source`` may be a single tensor name or a tuple for a transform that
    intentionally combines tensors (for example, PyTorch's two LSTM biases).
    ``source_shapes`` follows the same order.  ``target_shape`` is strongly
    recommended when no target template is supplied.
    """

    source: str | tuple[str, ...]
    target: str
    transform: str = "identity"
    source_shapes: tuple[tuple[int, ...], ...] | None = None
    target_shape: tuple[int, ...] | None = None
    target_dtype: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sources = _normalize_sources(self.source)
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("MappingRule.target must be a non-empty string")
        if not isinstance(self.transform, str) or not self.transform:
            raise ValueError("MappingRule.transform must be a non-empty string")
        source_shapes = _normalize_source_shapes(self.source_shapes)
        if source_shapes is not None and len(source_shapes) != len(sources):
            raise ValueError(
                "MappingRule.source_shapes must have one shape per source tensor"
            )
        target_shape = (
            None
            if self.target_shape is None
            else tuple(int(dimension) for dimension in self.target_shape)
        )
        if target_shape is not None and any(dimension < 0 for dimension in target_shape):
            raise ValueError("target shape dimensions must be non-negative")
        if self.target_dtype is not None:
            np.dtype(self.target_dtype)  # Validate eagerly.
        if not _is_mapping(self.options):
            raise TypeError("MappingRule.options must be a mapping")

        object.__setattr__(self, "source", sources[0] if len(sources) == 1 else sources)
        object.__setattr__(self, "source_shapes", source_shapes)
        object.__setattr__(self, "target_shape", target_shape)
        object.__setattr__(self, "options", dict(self.options))

    @property
    def sources(self) -> tuple[str, ...]:
        return _normalize_sources(self.source)

    def to_dict(self) -> JsonObject:
        sources = self.sources
        result: JsonObject = {
            "source": sources[0] if len(sources) == 1 else list(sources),
            "target": self.target,
            "transform": self.transform,
        }
        if self.source_shapes is not None:
            result["source_shapes"] = [list(shape) for shape in self.source_shapes]
        if self.target_shape is not None:
            result["target_shape"] = list(self.target_shape)
        if self.target_dtype is not None:
            result["target_dtype"] = self.target_dtype
        if self.options:
            result["options"] = dict(self.options)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MappingRule":
        allowed = {
            "source",
            "target",
            "transform",
            "source_shapes",
            "target_shape",
            "target_dtype",
            "options",
        }
        unexpected = set(value) - allowed
        if unexpected:
            raise ValueError(f"unexpected MappingRule fields: {sorted(unexpected)}")
        return cls(
            source=value["source"],
            target=value["target"],
            transform=value.get("transform", "identity"),
            source_shapes=value.get("source_shapes"),
            target_shape=value.get("target_shape"),
            target_dtype=value.get("target_dtype"),
            options=value.get("options", {}),
        )


@dataclass(frozen=True)
class MappingSpec:
    """Versioned collection of mapping rules and human-readable metadata."""

    rules: tuple[MappingRule, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = MAPPING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules", tuple(self.rules))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.schema_version != MAPPING_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported mapping schema version {self.schema_version}; "
                f"expected {MAPPING_SCHEMA_VERSION}"
            )

    def to_dict(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "metadata": dict(self.metadata),
            "rules": [rule.to_dict() for rule in self.rules],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MappingSpec":
        allowed = {"schema_version", "metadata", "rules"}
        unexpected = set(value) - allowed
        if unexpected:
            raise ValueError(f"unexpected mapping fields: {sorted(unexpected)}")
        return cls(
            schema_version=int(value.get("schema_version", MAPPING_SCHEMA_VERSION)),
            metadata=value.get("metadata", {}),
            rules=tuple(MappingRule.from_dict(rule) for rule in value["rules"]),
        )


def write_mapping(
    path: str | Path,
    mapping: MappingSpec | Sequence[MappingRule],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a mapping spec as stable, reviewable JSON."""

    if isinstance(mapping, MappingSpec):
        if metadata is not None:
            raise ValueError("metadata cannot be overridden for an existing MappingSpec")
        spec = mapping
    else:
        spec = MappingSpec(tuple(mapping), metadata or {})
    _write_json(path, spec.to_dict())


def load_mapping(path: str | Path) -> MappingSpec:
    """Load and validate a versioned JSON mapping spec."""

    return MappingSpec.from_dict(_read_json(path))


def _single_source(
    values: tuple[np.ndarray, ...], name: str, *, ndim: int | None = None
) -> np.ndarray:
    if len(values) != 1:
        raise TensorTransformError(f"{name} expects exactly one source tensor")
    value = values[0]
    if ndim is not None and value.ndim != ndim:
        raise TensorTransformError(
            f"{name} expects rank {ndim}, received shape {value.shape}"
        )
    return value


def _identity(values: tuple[np.ndarray, ...], _: Mapping[str, Any]) -> np.ndarray:
    return _single_source(values, "identity")


def _dense_transpose(
    values: tuple[np.ndarray, ...], _: Mapping[str, Any]
) -> np.ndarray:
    return _single_source(values, "dense_transpose", ndim=2).T


def _conv_transpose(
    values: tuple[np.ndarray, ...], _: Mapping[str, Any], *, spatial_rank: int
) -> np.ndarray:
    value = _single_source(values, f"pytorch_conv{spatial_rank}d", ndim=spatial_rank + 2)
    # PyTorch: [out, in/groups, spatial...]. Flax/Linen: [spatial..., in/groups, out].
    axes = (*range(2, spatial_rank + 2), 1, 0)
    return np.transpose(value, axes)


def _pytorch_conv1d(
    values: tuple[np.ndarray, ...], options: Mapping[str, Any]
) -> np.ndarray:
    return _conv_transpose(values, options, spatial_rank=1)


def _pytorch_conv2d(
    values: tuple[np.ndarray, ...], options: Mapping[str, Any]
) -> np.ndarray:
    return _conv_transpose(values, options, spatial_rank=2)


def _pytorch_conv3d(
    values: tuple[np.ndarray, ...], options: Mapping[str, Any]
) -> np.ndarray:
    return _conv_transpose(values, options, spatial_rank=3)


def _depthwise_conv(
    values: tuple[np.ndarray, ...], options: Mapping[str, Any], *, spatial_rank: int
) -> np.ndarray:
    name = f"pytorch_depthwise_conv{spatial_rank}d"
    value = _single_source(values, name, ndim=spatial_rank + 2)
    if value.shape[1] != 1:
        raise TensorTransformError(
            f"{name} requires PyTorch in_channels/groups == 1, got shape {value.shape}"
        )

    layout = options.get("layout", "flax_grouped")
    if layout == "flax_grouped":
        return np.transpose(value, (*range(2, spatial_rank + 2), 1, 0))
    if layout != "channel_multiplier":
        raise TensorTransformError(
            f"{name} layout must be 'flax_grouped' or 'channel_multiplier'"
        )

    try:
        in_channels = int(options["in_channels"])
    except (KeyError, TypeError, ValueError) as error:
        raise TensorTransformError(
            f"{name} with channel_multiplier layout requires integer in_channels"
        ) from error
    if in_channels <= 0 or value.shape[0] % in_channels:
        raise TensorTransformError(
            f"{name} output channels {value.shape[0]} are not divisible by "
            f"in_channels={in_channels}"
        )
    multiplier = value.shape[0] // in_channels
    # PyTorch orders output channels as input_channel * multiplier + multiplier_index.
    spatial_shape = value.shape[2:]
    reshaped = value[:, 0, ...].reshape(in_channels, multiplier, *spatial_shape)
    return np.transpose(reshaped, (*range(2, spatial_rank + 2), 0, 1))


def _pytorch_depthwise_conv1d(
    values: tuple[np.ndarray, ...], options: Mapping[str, Any]
) -> np.ndarray:
    return _depthwise_conv(values, options, spatial_rank=1)


def _pytorch_depthwise_conv2d(
    values: tuple[np.ndarray, ...], options: Mapping[str, Any]
) -> np.ndarray:
    return _depthwise_conv(values, options, spatial_rank=2)


def _gate_orders(options: Mapping[str, Any]) -> tuple[str, str]:
    source = str(options.get("source_gate_order", "ifgo"))
    target = str(options.get("target_gate_order", "ifgo"))
    if not source or len(set(source)) != len(source):
        raise TensorTransformError("source_gate_order must contain unique gate labels")
    if len(target) != len(source) or set(target) != set(source):
        raise TensorTransformError(
            "target_gate_order must be a permutation of source_gate_order"
        )
    return source, target


def _reorder_gates(
    value: np.ndarray, *, axis: int, source_order: str, target_order: str
) -> np.ndarray:
    if value.shape[axis] % len(source_order):
        raise TensorTransformError(
            f"gate axis size {value.shape[axis]} is not divisible by "
            f"gate count {len(source_order)}"
        )
    chunks = np.split(value, len(source_order), axis=axis)
    by_gate = dict(zip(source_order, chunks))
    return np.concatenate([by_gate[gate] for gate in target_order], axis=axis)


def _pytorch_lstm_kernel(
    values: tuple[np.ndarray, ...], options: Mapping[str, Any]
) -> np.ndarray:
    """Convert one PyTorch LSTM weight ``[gates*H, input]`` to ``[input, gates*H]``."""

    value = _single_source(values, "pytorch_lstm_kernel", ndim=2)
    source_order, target_order = _gate_orders(options)
    return _reorder_gates(
        value, axis=0, source_order=source_order, target_order=target_order
    ).T


def _pytorch_lstm_combined_kernel(
    values: tuple[np.ndarray, ...], options: Mapping[str, Any]
) -> np.ndarray:
    """Combine PyTorch input/recurrent weights into one ``[input+hidden, gates*H]`` kernel."""

    if len(values) != 2 or any(value.ndim != 2 for value in values):
        raise TensorTransformError(
            "pytorch_lstm_combined_kernel expects input and recurrent rank-2 weights"
        )
    if values[0].shape[0] != values[1].shape[0]:
        raise TensorTransformError(
            "LSTM input and recurrent weights must have equal gate-axis sizes"
        )
    source_order, target_order = _gate_orders(options)
    converted = [
        _reorder_gates(
            value, axis=0, source_order=source_order, target_order=target_order
        ).T
        for value in values
    ]
    return np.concatenate(converted, axis=0)


def _pytorch_lstm_bias(
    values: tuple[np.ndarray, ...], options: Mapping[str, Any]
) -> np.ndarray:
    """Convert one bias, or sum PyTorch's input/recurrent LSTM biases."""

    if len(values) not in (1, 2) or any(value.ndim != 1 for value in values):
        raise TensorTransformError(
            "pytorch_lstm_bias expects one or two rank-1 bias tensors"
        )
    if len(values) == 2 and values[0].shape != values[1].shape:
        raise TensorTransformError("PyTorch LSTM bias tensors must have equal shapes")
    value = values[0] if len(values) == 1 else values[0] + values[1]
    source_order, target_order = _gate_orders(options)
    return _reorder_gates(
        value, axis=0, source_order=source_order, target_order=target_order
    )


TRANSFORMS: Mapping[str, Transform] = {
    "identity": _identity,
    "dense_transpose": _dense_transpose,
    "pytorch_conv1d": _pytorch_conv1d,
    "pytorch_conv2d": _pytorch_conv2d,
    "pytorch_conv3d": _pytorch_conv3d,
    "pytorch_depthwise_conv1d": _pytorch_depthwise_conv1d,
    "pytorch_depthwise_conv2d": _pytorch_depthwise_conv2d,
    "pytorch_lstm_kernel": _pytorch_lstm_kernel,
    "pytorch_lstm_combined_kernel": _pytorch_lstm_combined_kernel,
    "pytorch_lstm_bias": _pytorch_lstm_bias,
}


@dataclass(frozen=True)
class ConvertedTensor:
    """Audit record for one successfully converted target tensor."""

    sources: tuple[str, ...]
    target: str
    transform: str
    source_shapes: tuple[tuple[int, ...], ...]
    target_shape: tuple[int, ...]
    target_dtype: str
    sha256: str

    def to_dict(self) -> JsonObject:
        return {
            "sources": list(self.sources),
            "target": self.target,
            "transform": self.transform,
            "source_shapes": [list(shape) for shape in self.source_shapes],
            "target_shape": list(self.target_shape),
            "target_dtype": self.target_dtype,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConvertedTensor":
        return cls(
            sources=tuple(str(name) for name in value["sources"]),
            target=str(value["target"]),
            transform=str(value["transform"]),
            source_shapes=tuple(
                tuple(int(dimension) for dimension in shape)
                for shape in value["source_shapes"]
            ),
            target_shape=tuple(int(dimension) for dimension in value["target_shape"]),
            target_dtype=str(value["target_dtype"]),
            sha256=str(value["sha256"]),
        )


@dataclass(frozen=True)
class ConversionReport:
    """Machine-readable provenance for a successful strict conversion."""

    source_inventory: tuple[TensorInfo, ...]
    converted: tuple[ConvertedTensor, ...]
    source_set_sha256: str
    target_set_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_inventory", tuple(self.source_inventory))
        object.__setattr__(self, "converted", tuple(self.converted))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.schema_version != REPORT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported report schema version {self.schema_version}; "
                f"expected {REPORT_SCHEMA_VERSION}"
            )

    def to_dict(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "metadata": dict(self.metadata),
            "source_set_sha256": self.source_set_sha256,
            "target_set_sha256": self.target_set_sha256,
            "source_inventory": [item.to_dict() for item in self.source_inventory],
            "converted": [item.to_dict() for item in self.converted],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConversionReport":
        allowed = {
            "schema_version",
            "metadata",
            "source_set_sha256",
            "target_set_sha256",
            "source_inventory",
            "converted",
        }
        unexpected = set(value) - allowed
        if unexpected:
            raise ValueError(f"unexpected report fields: {sorted(unexpected)}")
        return cls(
            schema_version=int(value.get("schema_version", REPORT_SCHEMA_VERSION)),
            metadata=value.get("metadata", {}),
            source_set_sha256=str(value["source_set_sha256"]),
            target_set_sha256=str(value["target_set_sha256"]),
            source_inventory=tuple(
                TensorInfo.from_dict(item) for item in value["source_inventory"]
            ),
            converted=tuple(
                ConvertedTensor.from_dict(item) for item in value["converted"]
            ),
        )


@dataclass(frozen=True)
class ConversionResult:
    """Nested converted parameters, flat tensors, and their audit report."""

    tree: Mapping[str, Any]
    flat_tensors: Mapping[str, np.ndarray]
    report: ConversionReport


def _flat_target_template(template: Mapping[str, Any]) -> dict[str, Any]:
    # A flat mapping from dotted parameter names is a convenient target template.
    # Recurse only when the mapping actually contains nested mappings.
    if any(_is_mapping(value) for value in template.values()):
        return flatten_pytree(template)
    return dict(template)


def _validate_rules(
    sources: Mapping[str, np.ndarray],
    rules: Sequence[MappingRule],
    *,
    target_template: Mapping[str, Any] | None,
    strict_source: bool,
    strict_target: bool,
) -> dict[str, Any] | None:
    errors: list[str] = []
    seen_sources: dict[str, int] = {}
    seen_targets: dict[str, int] = {}

    for index, rule in enumerate(rules):
        target_parts = rule.target.split(".")
        if any(not part for part in target_parts):
            errors.append(f"invalid target path {rule.target!r} in rule {index}")
        if rule.target in seen_targets:
            errors.append(
                f"duplicate target {rule.target!r} in rules "
                f"{seen_targets[rule.target]} and {index}"
            )
        else:
            seen_targets[rule.target] = index
        for source in rule.sources:
            if source in seen_sources:
                errors.append(
                    f"duplicate source {source!r} in rules "
                    f"{seen_sources[source]} and {index}"
                )
            else:
                seen_sources[source] = index

    sorted_targets = sorted(seen_targets)
    for index, target in enumerate(sorted_targets[:-1]):
        following = sorted_targets[index + 1]
        if following.startswith(target + "."):
            errors.append(
                f"prefix-colliding targets {target!r} and {following!r}"
            )

    missing_sources = sorted(set(seen_sources) - set(sources))
    if missing_sources:
        errors.append(f"missing source tensors: {missing_sources}")
    if strict_source:
        unexpected_sources = sorted(set(sources) - set(seen_sources))
        if unexpected_sources:
            errors.append(f"unexpected source tensors: {unexpected_sources}")

    flat_template = (
        None if target_template is None else _flat_target_template(target_template)
    )
    if flat_template is not None:
        mapped_targets = set(seen_targets)
        template_targets = set(flat_template)
        unexpected_targets = sorted(mapped_targets - template_targets)
        if unexpected_targets:
            errors.append(f"mapping targets absent from template: {unexpected_targets}")
        if strict_target:
            missing_targets = sorted(template_targets - mapped_targets)
            if missing_targets:
                errors.append(f"unmapped target tensors: {missing_targets}")

    if errors:
        raise MappingValidationError("invalid mapping:\n- " + "\n- ".join(errors))
    return flat_template


def convert_tensors(
    source_tensors: Mapping[str, ArrayLike],
    rules: Sequence[MappingRule] | MappingSpec,
    *,
    target_template: Mapping[str, Any] | None = None,
    strict_source: bool = True,
    strict_target: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> ConversionResult:
    """Apply a reviewed mapping and return a nested Flax-compatible dictionary.

    By default every source must be consumed exactly once.  If a target template
    is supplied, every template leaf must also be produced exactly once and each
    result must match its shape.  Rule-level source/target shapes are validated
    in addition to the template.  Duplicate source use and duplicate targets are
    always errors.
    """

    source_arrays: dict[str, np.ndarray] = {}
    for name, value in source_tensors.items():
        if not isinstance(name, str) or not name:
            raise ConversionError("source tensor names must be non-empty strings")
        if _is_mapping(value):
            raise ConversionError(
                "source_tensors must be flat; call flatten_pytree explicitly for nested trees"
            )
        source_arrays[name] = _as_tensor(value, name=name)

    if isinstance(rules, MappingSpec):
        mapping_rules = rules.rules
        report_metadata = dict(rules.metadata)
        if metadata:
            report_metadata.update(metadata)
    else:
        mapping_rules = tuple(rules)
        report_metadata = dict(metadata or {})

    flat_template = _validate_rules(
        source_arrays,
        mapping_rules,
        target_template=target_template,
        strict_source=strict_source,
        strict_target=strict_target,
    )

    converted: dict[str, np.ndarray] = {}
    records: list[ConvertedTensor] = []
    for index, rule in enumerate(mapping_rules):
        values = tuple(source_arrays[name] for name in rule.sources)
        actual_source_shapes = tuple(
            tuple(int(dimension) for dimension in value.shape) for value in values
        )
        if (
            rule.source_shapes is not None
            and actual_source_shapes != rule.source_shapes
        ):
            raise ConversionError(
                f"rule {index} ({rule.target!r}) source shape mismatch: "
                f"expected {rule.source_shapes}, got {actual_source_shapes}"
            )

        transform = TRANSFORMS.get(rule.transform)
        if transform is None:
            raise MappingValidationError(
                f"rule {index} ({rule.target!r}) uses unknown transform "
                f"{rule.transform!r}; available: {sorted(TRANSFORMS)}"
            )
        try:
            output = _as_tensor(
                transform(values, rule.options), name=rule.target
            )
        except ConversionError as error:
            raise TensorTransformError(
                f"rule {index} ({rule.target!r}, {rule.transform}) failed: {error}"
            ) from error
        except (TypeError, ValueError, IndexError) as error:
            raise TensorTransformError(
                f"rule {index} ({rule.target!r}, {rule.transform}) failed: {error}"
            ) from error

        if rule.target_dtype is not None:
            output = output.astype(np.dtype(rule.target_dtype), copy=False)
        output_shape = tuple(int(dimension) for dimension in output.shape)
        if rule.target_shape is not None and output_shape != rule.target_shape:
            raise ConversionError(
                f"rule {index} ({rule.target!r}) target shape mismatch: "
                f"expected {rule.target_shape}, got {output_shape}"
            )
        if flat_template is not None:
            expected = _as_tensor(flat_template[rule.target], name=rule.target)
            if output_shape != expected.shape:
                raise ConversionError(
                    f"rule {index} ({rule.target!r}) template shape mismatch: "
                    f"expected {expected.shape}, got {output_shape}"
                )

        converted[rule.target] = output
        records.append(
            ConvertedTensor(
                sources=rule.sources,
                target=rule.target,
                transform=rule.transform,
                source_shapes=actual_source_shapes,
                target_shape=output_shape,
                target_dtype=output.dtype.str,
                sha256=tensor_sha256(output),
            )
        )

    report = ConversionReport(
        source_inventory=inventory_tensors(source_arrays),
        converted=tuple(records),
        source_set_sha256=tensor_set_sha256(source_arrays),
        target_set_sha256=tensor_set_sha256(converted),
        metadata=report_metadata,
    )
    return ConversionResult(
        tree=unflatten_pytree(converted),
        flat_tensors=converted,
        report=report,
    )


def write_report(path: str | Path, report: ConversionReport) -> None:
    """Write a successful conversion report as stable JSON."""

    _write_json(path, report.to_dict())


def load_report(path: str | Path) -> ConversionReport:
    """Load and validate a conversion report written by :func:`write_report`."""

    return ConversionReport.from_dict(_read_json(path))


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: str | Path) -> JsonObject:
    source = Path(path)
    try:
        parsed = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConversionError(f"failed to read JSON file {source}: {error}") from error
    if not isinstance(parsed, dict):
        raise ConversionError(f"JSON file {source} must contain an object")
    return parsed


__all__ = [
    "ConversionError",
    "ConversionReport",
    "ConversionResult",
    "ConvertedTensor",
    "MappingRule",
    "MappingSpec",
    "MappingValidationError",
    "TensorInfo",
    "TensorTransformError",
    "TRANSFORMS",
    "convert_tensors",
    "flatten_pytree",
    "inventory_tensor_file",
    "inventory_tensors",
    "load_mapping",
    "load_report",
    "load_tensor_file",
    "tensor_set_sha256",
    "tensor_sha256",
    "unflatten_pytree",
    "write_mapping",
    "write_report",
]
