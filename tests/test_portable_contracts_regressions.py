from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin._canonical_contracts import (
    immutable_integer_array,
    integer_array,
)
from bayesian_phystwin._portable_contracts import source_artifact_mapping
from bayesian_phystwin._simulation_based_calibration_core import (
    weighted_randomized_pit,
)


def test_canonical_integer_arrays_are_little_endian_on_every_host() -> None:
    source = np.asarray([1, 256, -2], dtype=np.dtype(">i8"))
    expected = np.asarray([1, 256, -2], dtype=np.dtype("<i8"))

    owned = integer_array(source, name="values")
    frozen = immutable_integer_array(source, name="values")

    assert owned.dtype.str == "<i8"
    assert frozen.dtype.str == "<i8"
    assert owned.tobytes(order="C") == expected.tobytes(order="C")
    assert frozen.tobytes(order="C") == expected.tobytes(order="C")
    assert not frozen.flags.writeable
    with pytest.raises(ValueError):
        frozen.setflags(write=True)


@pytest.mark.parametrize(
    "path",
    [
        "/absolute/artifact.npy",
        "../artifact.npy",
        "raw/../artifact.npy",
        "C:/artifact.npy",
        "raw\\artifact.npy",
        "raw//artifact.npy",
        "raw/artifact.npy/",
        " raw/artifact.npy ",
    ],
)
def test_source_artifact_mapping_rejects_nonportable_paths(path: str) -> None:
    with pytest.raises(ValueError, match="canonical paths.*relative POSIX"):
        source_artifact_mapping(
            {path: "a" * 64},
            name="source_artifacts",
        )


def test_source_artifact_mapping_preserves_a_canonical_path() -> None:
    mapping = source_artifact_mapping(
        {"raw/object-1/artifact.npy": "a" * 64},
        name="source_artifacts",
    )
    assert mapping == {"raw/object-1/artifact.npy": "a" * 64}
    with pytest.raises(TypeError, match="immutable"):
        mapping["raw/object-1/artifact.npy"] = "b" * 64


def test_randomized_pit_normalizes_large_finite_weights_without_overflow() -> None:
    ordinary = weighted_randomized_pit(
        [0.0, 1.0],
        0.5,
        weights=[2.0, 1.0],
    )
    large = weighted_randomized_pit(
        [0.0, 1.0],
        0.5,
        weights=[1.0e308, 5.0e307],
    )

    assert ordinary == pytest.approx(2.0 / 3.0)
    assert large == pytest.approx(ordinary)
