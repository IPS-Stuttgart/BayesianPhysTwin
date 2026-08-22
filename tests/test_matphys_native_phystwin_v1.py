import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _preparer():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "remote"
        / "prepare_matphys_native_phystwin_case_v1.py"
    )
    spec = importlib.util.spec_from_file_location("matphys_native_preparer_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_case_arrays_preserve_released_structure_order() -> None:
    preparer = _preparer()
    observed = np.arange(36, dtype=np.float32).reshape(3, 4, 3)
    surface = np.array([[20.0, 21.0, 22.0]], dtype=np.float32)
    interior = np.array([[30.0, 31.0, 32.0]], dtype=np.float32)
    controller = np.arange(18, dtype=np.float32).reshape(3, 2, 3)

    values = preparer._case_arrays(
        {
            "object_points": observed,
            "surface_points": surface,
            "interior_points": interior,
            "controller_points": controller,
        }
    )

    np.testing.assert_array_equal(values[0], observed)
    np.testing.assert_array_equal(values[3], controller)
    np.testing.assert_array_equal(
        values[4], np.concatenate((observed[0], surface, interior), axis=0)
    )


def test_native_checkpoint_field_keeps_object_then_controller_order() -> None:
    preparer = _preparer()
    full = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

    object_field, complete = preparer._checkpoint_spring_fields(
        {"spring_Y": full, "num_object_springs": 3},
        total_spring_count=4,
        object_spring_count=3,
    )

    assert object_field.dtype == np.float32
    assert object_field.tobytes() == full[:3].tobytes()
    assert complete.tobytes() == full.tobytes()


def test_native_checkpoint_field_rejects_graph_mismatch() -> None:
    preparer = _preparer()

    with pytest.raises(ValueError, match="reconstructed graph disagree"):
        preparer._checkpoint_spring_fields(
            {
                "spring_Y": np.array([10.0, 20.0], dtype=np.float32),
                "num_object_springs": 2,
            },
            total_spring_count=3,
            object_spring_count=2,
        )
    with pytest.raises(ValueError, match="object-spring count changed"):
        preparer._checkpoint_spring_fields(
            {
                "spring_Y": np.array([10.0, 20.0, 30.0], dtype=np.float32),
                "num_object_springs": 1,
            },
            total_spring_count=3,
            object_spring_count=2,
        )
