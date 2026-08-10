"""Orbax checkpoint helpers with explicit metadata and integrity evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import jax
import numpy as np
import orbax.checkpoint as ocp

_METADATA_FILE = "fcrnnt_metadata.json"


def tree_fingerprint(tree: Any) -> str:
    """Return a deterministic SHA-256 over PyTree paths, shapes, dtypes and bytes."""

    digest = hashlib.sha256()
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    for path, leaf in path_leaves:
        original_shape = getattr(leaf, "shape", ())
        original_dtype = getattr(leaf, "dtype", type(leaf).__name__)
        if hasattr(jax.dtypes, "prng_key") and jax.dtypes.issubdtype(
            original_dtype, jax.dtypes.prng_key
        ):
            # Typed PRNG keys intentionally reject implicit NumPy conversion.
            value = np.asarray(jax.device_get(jax.random.key_data(leaf)))
        else:
            value = np.asarray(jax.device_get(leaf))
        digest.update(jax.tree_util.keystr(path).encode("utf-8"))
        digest.update(str(original_shape).encode("ascii"))
        digest.update(str(original_dtype).encode("ascii"))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _normalise_metadata(metadata: Mapping[str, Any] | None, item: Any) -> dict[str, Any]:
    result = dict(metadata or {})
    result["tree_sha256"] = tree_fingerprint(item)
    result["format"] = "fcrnnt-jax-orbax-v1"
    # Fail early rather than writing metadata that cannot be replayed.
    json.dumps(result, sort_keys=True)
    return result


def save_checkpoint(
    directory: str | Path,
    item: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Save a complete training PyTree and return the written metadata.

    ``item`` should contain parameters, mutable model state, optimizer state,
    step, RNG and data cursor when those are part of the experiment.  The
    helper intentionally refuses to replace an existing directory unless the
    caller explicitly sets ``overwrite=True``.
    """

    path = Path(directory).expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    written_metadata = _normalise_metadata(metadata, item)

    checkpointer = ocp.StandardCheckpointer()
    try:
        checkpointer.save(path, item, force=overwrite)
        checkpointer.wait_until_finished()
    finally:
        close = getattr(checkpointer, "close", None)
        if close is not None:
            close()
    (path / _METADATA_FILE).write_text(
        json.dumps(written_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return written_metadata


def restore_checkpoint(
    directory: str | Path,
    *,
    target: Any,
    verify_fingerprint: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Restore an Orbax PyTree and its PoC metadata.

    A freshly initialized target is required so Orbax restores the intended
    TrainState structure, dtypes and sharding instead of guessing from storage
    metadata.  This is also the safe cross-device restore path.
    """

    path = Path(directory).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {path}")
    metadata_path = path / _METADATA_FILE
    if not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    checkpointer = ocp.StandardCheckpointer()
    try:
        item = checkpointer.restore(path, target=target)
    finally:
        close = getattr(checkpointer, "close", None)
        if close is not None:
            close()
    if verify_fingerprint:
        actual = tree_fingerprint(item)
        expected = metadata.get("tree_sha256")
        if not expected or actual != expected:
            raise ValueError(
                f"checkpoint fingerprint mismatch: expected {expected!r}, got {actual!r}"
            )
    return item, metadata
