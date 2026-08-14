from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.observation_belief import ObservationBeliefV1
from bayesian_phystwin.observation_structured_covariance import (
    OBSERVATION_STRUCTURED_COVARIANCE_ADAPTER_SCHEMA,
    observation_point_ids,
    structured_covariance_from_observation_belief,
    write_observation_structured_covariance,
)
from bayesian_phystwin.structured_point_covariance_io import (
    load_structured_point_covariance,
)

_SOURCE_DIGEST = "1" * 64
_CALIBRATION_DIGEST = "2" * 64


def _belief(*, factor_names: tuple[str, ...] | None = None) -> ObservationBeliefV1:
    names = (
        ("gauge_x", "gauge_y", "process")
        if factor_names is None
        else factor_names
    )
    count = 4
    rank = len(names)
    factor = np.arange(count * 3 * rank, dtype=np.float64).reshape(count, 3, rank)
    factor = factor / 1000.0
    return ObservationBeliefV1(
        case_id="case-a",
        stream_id="stream-a",
        causal_frame_stop=2,
        view_names=("cam-0", "cam-1"),
        window_names=("window-0", "window-1"),
        factor_names=names,
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="3" * 40,
        source_artifact_sha256=_SOURCE_DIGEST,
        declared_frame_ids=np.asarray([0, 1], dtype=np.int64),
        mean_xyz_m=np.arange(count * 3, dtype=np.float64).reshape(count, 3) / 10.0,
        frame_ids=np.asarray([0, 0, 1, 1], dtype=np.int64),
        entity_ids=np.asarray([0, 1, 0, 1], dtype=np.int64),
        view_indices=np.asarray([0, 0, 1, 1], dtype=np.int64),
        window_indices=np.asarray([0, 0, 1, 1], dtype=np.int64),
        correlation_group_ids=np.asarray([0, 0, 1, 1], dtype=np.int64),
        factor_group_ids=np.asarray([10, 10, 20, 20], dtype=np.int64),
        prior_reliability=np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float64),
        association_probability=np.asarray([1.0, 0.9, 0.8, 0.7], dtype=np.float64),
        local_covariance_m2=np.repeat(
            (np.eye(3, dtype=np.float64) * 0.01)[None, :, :],
            count,
            axis=0,
        ),
        low_rank_factor_m=factor,
        group_ids=np.asarray([0, 1], dtype=np.int64),
        group_prior_nominal_probability=np.asarray([0.9, 0.8], dtype=np.float64),
        group_composite_weight=np.asarray([1.0, 0.75], dtype=np.float64),
        metadata={"producer": "fixture"},
    )


def _original_component_covariance(
    belief: ObservationBeliefV1,
    columns: tuple[int, ...],
) -> np.ndarray:
    dimension = 3 * belief.observation_count
    result = np.zeros((dimension, dimension), dtype=np.float64)
    for group_id in np.unique(belief.factor_group_ids):
        root = np.zeros((dimension, len(columns)), dtype=np.float64)
        for row in np.flatnonzero(belief.factor_group_ids == group_id):
            root[3 * row : 3 * row + 3] = belief.low_rank_factor_m[row][
                :,
                np.asarray(columns, dtype=np.int64),
            ]
        result += root @ root.T
    return result


def _original_dense_covariance(belief: ObservationBeliefV1) -> np.ndarray:
    dimension = 3 * belief.observation_count
    result = np.zeros((dimension, dimension), dtype=np.float64)
    for row, block in enumerate(belief.local_covariance_m2):
        result[3 * row : 3 * row + 3, 3 * row : 3 * row + 3] = block
    if belief.factor_rank:
        result += _original_component_covariance(
            belief,
            tuple(range(belief.factor_rank)),
        )
    return result


def _assert_covariance_parity(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-18)


def _mapping() -> dict[str, str]:
    return {
        "gauge_x": "gauge",
        "gauge_y": "gauge",
        "process": "process",
    }


def test_factor_group_expansion_preserves_dense_covariance_exactly() -> None:
    belief = _belief()
    covariance = structured_covariance_from_observation_belief(
        belief,
        coordinate_frame="world",
        factor_components=_mapping(),
        calibration_artifact_id=_CALIBRATION_DIGEST,
        metadata={"protocol_id": "adapter-test"},
    )

    _assert_covariance_parity(
        covariance.dense_covariance_m2(maximum_dimension=12),
        _original_dense_covariance(belief),
    )
    _assert_covariance_parity(
        covariance.shared_factors_m["gauge"].reshape(12, 4)
        @ covariance.shared_factors_m["gauge"].reshape(12, 4).T,
        _original_component_covariance(belief, (0, 1)),
    )
    _assert_covariance_parity(
        covariance.shared_factors_m["process"].reshape(12, 2)
        @ covariance.shared_factors_m["process"].reshape(12, 2).T,
        _original_component_covariance(belief, (2,)),
    )
    assert covariance.shared_component_names == ("gauge", "process")
    assert covariance.shared_rank == 6
    assert covariance.source_artifact_id == belief.artifact_id
    assert covariance.calibration_artifact_id == _CALIBRATION_DIGEST
    assert covariance.metadata["adapter_schema"] == (
        OBSERVATION_STRUCTURED_COVARIANCE_ADAPTER_SCHEMA
    )
    assert covariance.metadata["expanded_shared_ranks"] == {
        "gauge": 4,
        "process": 2,
    }
    assert covariance.metadata["caller"] == {"protocol_id": "adapter-test"}


def test_factor_groups_remain_independent_and_row_ids_are_deterministic() -> None:
    belief = _belief()
    covariance = structured_covariance_from_observation_belief(
        belief,
        coordinate_frame="world",
        factor_components=_mapping(),
    )
    ids = observation_point_ids(belief)

    assert covariance.point_ids == ids
    assert len(set(ids)) == belief.observation_count
    assert "frame=0|entity=0|view=0|window=0" in ids[0]
    assert np.array_equal(
        covariance.cross_covariance_m2(ids[0], ids[2]),
        np.zeros((3, 3), dtype=np.float64),
    )
    expected_within_group = (
        belief.low_rank_factor_m[0] @ belief.low_rank_factor_m[1].T
    )
    _assert_covariance_parity(
        covariance.cross_covariance_m2(ids[0], ids[1]),
        expected_within_group,
    )


def test_archive_roundtrip_retains_adapter_identity(tmp_path: Path) -> None:
    belief = _belief()
    path = tmp_path / "structured-covariance.npz"
    covariance = write_observation_structured_covariance(
        path,
        belief,
        coordinate_frame="world",
        factor_components=_mapping(),
    )
    loaded = load_structured_point_covariance(path)

    assert loaded.artifact_id == covariance.artifact_id
    assert loaded.descriptor() == covariance.descriptor()
    _assert_covariance_parity(
        loaded.dense_covariance_m2(maximum_dimension=12),
        _original_dense_covariance(belief),
    )
    with pytest.raises(FileExistsError):
        write_observation_structured_covariance(
            path,
            belief,
            coordinate_frame="world",
            factor_components=_mapping(),
        )


def test_empty_factor_set_preserves_only_local_covariance() -> None:
    belief = _belief(factor_names=())
    covariance = structured_covariance_from_observation_belief(
        belief,
        coordinate_frame="world",
        factor_components={},
    )

    assert covariance.shared_component_names == ()
    assert covariance.shared_rank == 0
    assert np.array_equal(
        covariance.dense_covariance_m2(maximum_dimension=12),
        _original_dense_covariance(belief),
    )


@pytest.mark.parametrize(
    "mapping",
    [
        {"gauge_x": "gauge", "gauge_y": "gauge"},
        {
            "gauge_x": "gauge",
            "gauge_y": "gauge",
            "process": "process",
            "extra": "process",
        },
    ],
)
def test_factor_mapping_must_be_exact(mapping: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="identify every factor exactly"):
        structured_covariance_from_observation_belief(
            _belief(),
            coordinate_frame="world",
            factor_components=mapping,
        )


def test_factor_component_and_rank_fail_closed() -> None:
    mapping = _mapping()
    mapping["process"] = "unsupported"
    with pytest.raises(ValueError, match="must be one of"):
        structured_covariance_from_observation_belief(
            _belief(),
            coordinate_frame="world",
            factor_components=mapping,
        )

    with pytest.raises(ValueError, match="maximum_expanded_rank"):
        structured_covariance_from_observation_belief(
            _belief(),
            coordinate_frame="world",
            factor_components=_mapping(),
            maximum_expanded_rank=5,
        )


def test_ambiguous_duplicate_factor_names_are_rejected() -> None:
    duplicate = replace(
        _belief(),
        factor_names=("duplicate", "duplicate", "process"),
    )
    with pytest.raises(ValueError, match="must be unique"):
        structured_covariance_from_observation_belief(
            duplicate,
            coordinate_frame="world",
            factor_components={"duplicate": "gauge", "process": "process"},
        )


def test_argument_types_are_not_coerced() -> None:
    belief = _belief()
    with pytest.raises(TypeError, match="observation"):
        observation_point_ids(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="observation"):
        structured_covariance_from_observation_belief(
            object(),  # type: ignore[arg-type]
            coordinate_frame="world",
            factor_components={},
        )
    with pytest.raises(TypeError, match="factor_components"):
        structured_covariance_from_observation_belief(
            belief,
            coordinate_frame="world",
            factor_components=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="metadata"):
        structured_covariance_from_observation_belief(
            belief,
            coordinate_frame="world",
            factor_components=_mapping(),
            metadata=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="coordinate_frame"):
        structured_covariance_from_observation_belief(
            belief,
            coordinate_frame=" world ",
            factor_components=_mapping(),
        )
    with pytest.raises(ValueError, match="must be an integer"):
        structured_covariance_from_observation_belief(
            belief,
            coordinate_frame="world",
            factor_components=_mapping(),
            maximum_expanded_rank=True,  # type: ignore[arg-type]
        )
