from pathlib import Path

import jax.numpy as jnp
import jax
import numpy as np
import pytest

from fcrnnt_jax.checkpoint import restore_checkpoint, save_checkpoint, tree_fingerprint


def test_orbax_round_trip_and_metadata(tmp_path: Path):
    item = {
        "params": {"kernel": jnp.arange(6, dtype=jnp.float32).reshape(2, 3)},
        "step": jnp.asarray(7, dtype=jnp.int32),
        "rng": jax.random.key(1),
    }
    directory = tmp_path / "checkpoint"
    written = save_checkpoint(directory, item, metadata={"experiment": "unit-test"})
    restored, metadata = restore_checkpoint(directory, target=item)

    assert written == metadata
    assert metadata["experiment"] == "unit-test"
    assert metadata["tree_sha256"] == tree_fingerprint(restored)
    np.testing.assert_array_equal(restored["params"]["kernel"], item["params"]["kernel"])
    assert int(restored["step"]) == 7


def test_existing_checkpoint_is_not_silently_replaced(tmp_path: Path):
    directory = tmp_path / "checkpoint"
    save_checkpoint(directory, {"value": jnp.asarray(1)})
    with pytest.raises(FileExistsError, match="already exists"):
        save_checkpoint(directory, {"value": jnp.asarray(2)})
