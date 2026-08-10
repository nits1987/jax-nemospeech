from __future__ import annotations

import json

import numpy as np
import pytest

from fcrnnt_jax.conversion import (
    ConversionError,
    MappingRule,
    MappingSpec,
    MappingValidationError,
    TensorTransformError,
    convert_tensors,
    flatten_pytree,
    inventory_tensor_file,
    inventory_tensors,
    load_mapping,
    load_report,
    load_tensor_file,
    tensor_set_sha256,
    tensor_sha256,
    unflatten_pytree,
    write_mapping,
    write_report,
)


def test_flatten_and_unflatten_pytree_round_trip() -> None:
    tree = {
        "params": {
            "encoder": {"kernel": np.arange(6).reshape(2, 3)},
            "joint": {"bias": np.zeros(4)},
        }
    }

    flat = flatten_pytree(tree)

    assert set(flat) == {"params.encoder.kernel", "params.joint.bias"}
    restored = unflatten_pytree(flat)
    np.testing.assert_array_equal(
        restored["params"]["encoder"]["kernel"], tree["params"]["encoder"]["kernel"]
    )
    np.testing.assert_array_equal(
        restored["params"]["joint"]["bias"], tree["params"]["joint"]["bias"]
    )


@pytest.mark.parametrize(
    "flat",
    [
        {"a": np.array(1), "a.b": np.array(2)},
        {"a.b": np.array(1), "a": np.array(2)},
        {"a..b": np.array(1)},
    ],
)
def test_unflatten_rejects_ambiguous_paths(flat: dict[str, np.ndarray]) -> None:
    with pytest.raises(ValueError, match="collision|invalid"):
        unflatten_pytree(flat)


def test_tensor_hash_and_inventory_are_deterministic_and_shape_sensitive() -> None:
    original = np.arange(6, dtype=np.float32).reshape(2, 3)
    same = np.array(original, order="F")
    reshaped = original.reshape(3, 2)

    assert tensor_sha256(original) == tensor_sha256(same)
    assert tensor_sha256(original) != tensor_sha256(reshaped)
    assert tensor_set_sha256({"b": original, "a": reshaped}) == tensor_set_sha256(
        {"a": reshaped, "b": original}
    )

    inventory = inventory_tensors({"z": original, "a": reshaped})
    assert [item.name for item in inventory] == ["a", "z"]
    assert inventory[0].shape == (3, 2)
    assert inventory[0].dtype == np.dtype(np.float32).str
    assert inventory[0].nbytes == reshaped.nbytes


def test_strict_conversion_applies_dense_conv_and_depthwise_transforms() -> None:
    source = {
        "dense.weight": np.arange(6, dtype=np.float32).reshape(3, 2),
        "conv.weight": np.arange(24, dtype=np.float32).reshape(4, 2, 3),
        "depthwise.weight": np.arange(12, dtype=np.float32).reshape(4, 1, 3),
    }
    rules = (
        MappingRule(
            source="dense.weight",
            target="params.dense.kernel",
            transform="dense_transpose",
            source_shapes=((3, 2),),
            target_shape=(2, 3),
        ),
        MappingRule(
            source="conv.weight",
            target="params.conv.kernel",
            transform="pytorch_conv1d",
            target_shape=(3, 2, 4),
        ),
        MappingRule(
            source="depthwise.weight",
            target="params.depthwise.kernel",
            transform="pytorch_depthwise_conv1d",
            options={"layout": "channel_multiplier", "in_channels": 2},
            target_shape=(3, 2, 2),
        ),
    )
    template = {
        "params": {
            "dense": {"kernel": np.empty((2, 3), dtype=np.float32)},
            "conv": {"kernel": np.empty((3, 2, 4), dtype=np.float32)},
            "depthwise": {"kernel": np.empty((3, 2, 2), dtype=np.float32)},
        }
    }

    result = convert_tensors(
        source,
        rules,
        target_template=template,
        metadata={"checkpoint": "synthetic"},
    )

    np.testing.assert_array_equal(
        result.flat_tensors["params.dense.kernel"], source["dense.weight"].T
    )
    np.testing.assert_array_equal(
        result.flat_tensors["params.conv.kernel"],
        np.transpose(source["conv.weight"], (2, 1, 0)),
    )
    expected_depthwise = source["depthwise.weight"][:, 0, :].reshape(2, 2, 3)
    expected_depthwise = np.transpose(expected_depthwise, (2, 0, 1))
    np.testing.assert_array_equal(
        result.flat_tensors["params.depthwise.kernel"], expected_depthwise
    )
    assert result.report.metadata == {"checkpoint": "synthetic"}
    assert len(result.report.converted) == 3
    assert result.report.target_set_sha256 == tensor_set_sha256(result.flat_tensors)
    assert result.tree["params"]["dense"]["kernel"].shape == (2, 3)


def test_pytorch_lstm_transforms_combine_biases_and_reorder_gates() -> None:
    hidden = 2

    def gate_matrix(width: int) -> np.ndarray:
        return np.concatenate(
            [np.full((hidden, width), value, np.float32) for value in (1, 2, 3, 4)]
        )

    def gate_bias(scale: int) -> np.ndarray:
        return np.concatenate(
            [np.full((hidden,), scale * value, np.float32) for value in (1, 2, 3, 4)]
        )

    source = {
        "weight_ih_l0": gate_matrix(3),
        "weight_hh_l0": gate_matrix(2) * 10,
        "bias_ih_l0": gate_bias(1),
        "bias_hh_l0": gate_bias(10),
    }
    options = {"source_gate_order": "ifgo", "target_gate_order": "igfo"}
    rules = (
        MappingRule(
            source=("weight_ih_l0", "weight_hh_l0"),
            target="params.lstm.kernel",
            transform="pytorch_lstm_combined_kernel",
            target_shape=(5, 8),
            options=options,
        ),
        MappingRule(
            source=("bias_ih_l0", "bias_hh_l0"),
            target="params.lstm.bias",
            transform="pytorch_lstm_bias",
            target_shape=(8,),
            options=options,
        ),
    )

    result = convert_tensors(source, rules)

    kernel = result.flat_tensors["params.lstm.kernel"]
    # Target gate order is i, g, f, o. Input and recurrent rows are concatenated.
    np.testing.assert_array_equal(kernel[0, :], [1, 1, 3, 3, 2, 2, 4, 4])
    np.testing.assert_array_equal(kernel[3, :], [10, 10, 30, 30, 20, 20, 40, 40])
    np.testing.assert_array_equal(
        result.flat_tensors["params.lstm.bias"],
        [11, 11, 33, 33, 22, 22, 44, 44],
    )


def test_mapping_fails_closed_for_missing_and_unexpected_sources() -> None:
    rules = [MappingRule("required", "params.kernel")]

    with pytest.raises(MappingValidationError, match="missing source tensors"):
        convert_tensors({"different": np.ones(1)}, rules)

    with pytest.raises(MappingValidationError, match="unexpected source tensors"):
        convert_tensors(
            {"required": np.ones(1), "not_mapped": np.ones(1)}, rules
        )

    result = convert_tensors(
        {"required": np.ones(1), "not_mapped": np.ones(1)},
        rules,
        strict_source=False,
    )
    assert set(result.flat_tensors) == {"params.kernel"}


@pytest.mark.parametrize(
    ("rules", "message"),
    [
        (
            [MappingRule("x", "a"), MappingRule("x", "b")],
            "duplicate source",
        ),
        (
            [MappingRule("x", "a"), MappingRule("y", "a")],
            "duplicate target",
        ),
        (
            [MappingRule("x", "a"), MappingRule("y", "a.kernel")],
            "prefix-colliding targets",
        ),
    ],
)
def test_mapping_rejects_duplicate_consumption(
    rules: list[MappingRule], message: str
) -> None:
    with pytest.raises(MappingValidationError, match=message):
        convert_tensors({"x": np.ones(1), "y": np.ones(1)}, rules)


def test_target_template_requires_complete_mapping() -> None:
    source = {"x": np.ones(2)}
    template = {"params": {"x": np.ones(2), "y": np.ones(3)}}

    with pytest.raises(MappingValidationError, match="unmapped target tensors"):
        convert_tensors(source, [MappingRule("x", "params.x")], target_template=template)


def test_rule_and_template_shape_validation() -> None:
    with pytest.raises(ConversionError, match="source shape mismatch"):
        convert_tensors(
            {"x": np.ones((2, 3))},
            [MappingRule("x", "params.x", source_shapes=((3, 2),))],
        )

    with pytest.raises(ConversionError, match="target shape mismatch"):
        convert_tensors(
            {"x": np.ones((2, 3))},
            [MappingRule("x", "params.x", target_shape=(3, 2))],
        )

    with pytest.raises(ConversionError, match="template shape mismatch"):
        convert_tensors(
            {"x": np.ones((2, 3))},
            [MappingRule("x", "params.x")],
            target_template={"params": {"x": np.ones((3, 2))}},
        )


def test_unknown_and_invalid_transforms_raise_contextual_errors() -> None:
    with pytest.raises(MappingValidationError, match="unknown transform"):
        convert_tensors(
            {"x": np.ones(1)},
            [MappingRule("x", "params.x", transform="not-a-transform")],
        )

    with pytest.raises(TensorTransformError, match="requires PyTorch"):
        convert_tensors(
            {"x": np.ones((2, 2, 3))},
            [
                MappingRule(
                    "x", "params.x", transform="pytorch_depthwise_conv1d"
                )
            ],
        )


def test_mapping_and_report_json_round_trip(tmp_path) -> None:
    spec = MappingSpec(
        rules=(
            MappingRule(
                "linear.weight",
                "params.linear.kernel",
                transform="dense_transpose",
                source_shapes=((3, 2),),
                target_shape=(2, 3),
                target_dtype="float32",
            ),
        ),
        metadata={"model": "synthetic", "complete": False},
    )
    mapping_path = tmp_path / "mapping.json"
    write_mapping(mapping_path, spec)

    loaded_spec = load_mapping(mapping_path)
    assert loaded_spec == spec
    assert json.loads(mapping_path.read_text(encoding="utf-8"))["schema_version"] == 1

    result = convert_tensors(
        {"linear.weight": np.arange(6).reshape(3, 2)}, loaded_spec
    )
    report_path = tmp_path / "conversion-report.json"
    write_report(report_path, result.report)

    loaded_report = load_report(report_path)
    assert loaded_report == result.report


def test_npz_load_and_inventory(tmp_path) -> None:
    path = tmp_path / "checkpoint.npz"
    np.savez(path, encoder=np.arange(4), joint=np.ones((2, 3), dtype=np.float32))

    tensors = load_tensor_file(path)
    inventory = inventory_tensor_file(path)

    assert set(tensors) == {"encoder", "joint"}
    assert [item.name for item in inventory] == ["encoder", "joint"]
    np.testing.assert_array_equal(tensors["encoder"], np.arange(4))


def test_bad_mapping_json_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad-mapping.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rules": [{"source": "x", "target": "y", "surprise": True}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected MappingRule fields"):
        load_mapping(path)


def test_object_dtype_is_rejected() -> None:
    with pytest.raises(ConversionError, match="object dtype"):
        convert_tensors(
            {"x": np.array([object()], dtype=object)},
            [MappingRule("x", "params.x")],
        )
