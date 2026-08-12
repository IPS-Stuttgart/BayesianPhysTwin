from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.phystwin_correspondence_gate import (
    detect_pairwise_consensus_correspondences as detect_v1,
)
from bayesian_phystwin.phystwin_correspondence_gate_v2 import (
    PairwiseCorrespondenceGateConfig,
    PairwiseCorrespondenceGateResult,
    detect_pairwise_consensus_correspondences,
    pairwise_distance_strain_m,
)


def _identity_problem(count: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinate = np.linspace(0.0, 0.8, count)
    source = np.column_stack((coordinate, coordinate**2, coordinate**3))
    return source, source.copy(), np.ones(count, dtype=np.bool_)


def _assert_matches_v1(
    source: np.ndarray,
    observed: np.ndarray,
    available: np.ndarray,
    *,
    material_ids: np.ndarray | None = None,
) -> None:
    prospective = detect_pairwise_consensus_correspondences(
        source,
        observed,
        available,
        material_ids=material_ids,
    )
    registered = detect_v1(
        source,
        observed,
        available,
        material_ids=material_ids,
    )
    assert np.array_equal(prospective.inlier_mask, registered.inlier_mask)
    assert prospective.accepted == registered.accepted
    assert prospective.decision == registered.decision
    assert prospective.available_count == registered.available_count
    assert prospective.inlier_count == registered.inlier_count
    assert prospective.inlier_fraction == registered.inlier_fraction
    assert prospective.pair_count == registered.pair_count
    assert prospective.compatible_pair_fraction == registered.compatible_pair_fraction
    assert (
        prospective.median_inlier_normalized_strain
        == registered.median_inlier_normalized_strain
    )
    assert (
        prospective.maximum_inlier_normalized_strain
        == registered.maximum_inlier_normalized_strain
    )


def _valid_result() -> PairwiseCorrespondenceGateResult:
    return PairwiseCorrespondenceGateResult(
        inlier_mask=np.ones(2, dtype=np.bool_),
        accepted=np.bool_(True),
        decision="accepted",
        available_count=np.int64(2),
        inlier_count=np.int64(2),
        inlier_fraction=np.float64(1.0),
        pair_count=np.int64(1),
        compatible_pair_fraction=np.float64(1.0),
        median_inlier_normalized_strain=np.float64(0.0),
        maximum_inlier_normalized_strain=np.float64(0.0),
    )


def test_gate_preserves_identity_and_normalizes_valid_scalars() -> None:
    source, observed, available = _identity_problem()
    config = PairwiseCorrespondenceGateConfig(
        absolute_pair_strain_m=np.float64(0.03),
        relative_pair_strain=np.float64(0.1),
        minimum_inlier_count=np.int64(9),
        minimum_inlier_fraction=np.float64(0.7),
        maximum_exact_center_count=np.int64(24),
    )

    result = detect_pairwise_consensus_correspondences(
        source,
        observed,
        available,
        material_ids=np.arange(len(source), dtype=np.int32),
        config=config,
    )

    assert result.accepted
    assert result.inlier_count == len(source)
    assert result.inlier_mask.dtype == np.dtype(np.bool_)
    assert not result.inlier_mask.flags.writeable
    assert isinstance(config.absolute_pair_strain_m, float)
    assert isinstance(config.minimum_inlier_count, int)


def test_gate_matches_registered_v1_under_rigid_motion() -> None:
    source, _, available = _identity_problem(10)
    angle = 0.4
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    observed = source @ rotation.T + np.asarray([0.2, -0.1, 0.05])
    ids = np.arange(100, 110, dtype=np.int64)

    _assert_matches_v1(source, observed, available, material_ids=ids)


def test_gate_matches_registered_v1_with_one_outlier() -> None:
    source, observed, available = _identity_problem(10)
    observed[-1] += np.asarray([2.0, -1.0, 0.5])

    _assert_matches_v1(source, observed, available)
    result = detect_pairwise_consensus_correspondences(
        source,
        observed,
        available,
    )
    assert result.accepted
    assert result.inlier_count == 9
    assert not result.inlier_mask[-1]


def test_gate_handles_insufficient_and_nonfinite_support_like_v1() -> None:
    source, observed, available = _identity_problem(10)
    available[8:] = False
    _assert_matches_v1(source, observed, available)
    insufficient = detect_pairwise_consensus_correspondences(
        source,
        observed,
        available,
    )
    assert insufficient.decision == "insufficient_available_support"

    available[:] = True
    observed[-1] = np.nan
    _assert_matches_v1(source, observed, available)
    finite = detect_pairwise_consensus_correspondences(
        source,
        observed,
        available,
    )
    assert finite.available_count == 9
    assert finite.accepted


def test_pairwise_diagnostics_match_v1_and_reject_bad_geometry() -> None:
    source, observed, _ = _identity_problem()
    strain, distance = pairwise_distance_strain_m(source, observed)
    assert np.all(strain == 0.0)
    assert distance.shape == (len(source), len(source))

    with pytest.raises(ValueError, match="source positions must be numeric"):
        pairwise_distance_strain_m(source.astype(str), observed)
    with pytest.raises(ValueError, match="observed positions must be numeric"):
        pairwise_distance_strain_m(source, observed.astype(object))
    with pytest.raises(ValueError, match="share shape"):
        pairwise_distance_strain_m(source[:, :2], observed[:, :2])
    observed[0, 0] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        pairwise_distance_strain_m(source, observed)


def test_gate_rejects_non_boolean_availability_without_coercion() -> None:
    source, observed, available = _identity_problem()
    invalid_masks = (
        available.astype(np.int64),
        available.astype(np.float64),
        np.full(available.shape, 0.5, dtype=np.float64),
        np.full(available.shape, np.nan, dtype=np.float64),
        np.asarray(available, dtype=object),
    )

    for invalid in invalid_masks:
        with pytest.raises(ValueError, match="available must contain only booleans"):
            detect_pairwise_consensus_correspondences(source, observed, invalid)


def test_gate_rejects_falsey_non_config() -> None:
    source, observed, available = _identity_problem()

    with pytest.raises(
        TypeError,
        match="config must be a PairwiseCorrespondenceGateConfig",
    ):
        detect_pairwise_consensus_correspondences(
            source,
            observed,
            available,
            config=0,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"absolute_pair_strain_m": False}, "must be a real number"),
        ({"absolute_pair_strain_m": np.inf}, "must be finite"),
        ({"absolute_pair_strain_m": 0.0}, "must be positive"),
        ({"relative_pair_strain": -0.1}, "must be nonnegative"),
        ({"minimum_inlier_count": True}, "must be an integer"),
        ({"minimum_inlier_count": 2.5}, "must be an integer"),
        ({"minimum_inlier_count": 1}, "must be at least two"),
        ({"minimum_inlier_fraction": np.nan}, "must be finite"),
        ({"minimum_inlier_fraction": 0.0}, "must lie in"),
        ({"minimum_inlier_fraction": 1.1}, "must lie in"),
        ({"maximum_exact_center_count": False}, "must be an integer"),
        ({"maximum_exact_center_count": 24.5}, "must be an integer"),
        ({"maximum_exact_center_count": 8}, "must cover"),
    ],
)
def test_config_rejects_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PairwiseCorrespondenceGateConfig(**kwargs)  # type: ignore[arg-type]


def test_gate_rejects_coerced_or_ambiguous_material_identities() -> None:
    source, observed, available = _identity_problem()
    invalid_ids = (
        np.arange(len(source), dtype=np.float64),
        np.ones(len(source), dtype=np.bool_),
        np.asarray([str(index) for index in range(len(source))]),
    )
    for material_ids in invalid_ids:
        with pytest.raises(ValueError, match="material_ids must contain integers"):
            detect_pairwise_consensus_correspondences(
                source,
                observed,
                available,
                material_ids=material_ids,
            )

    with pytest.raises(ValueError, match="shape"):
        detect_pairwise_consensus_correspondences(
            source,
            observed,
            available,
            material_ids=np.arange(len(source) - 1),
        )
    duplicate = np.arange(len(source))
    duplicate[-1] = duplicate[-2]
    with pytest.raises(ValueError, match="one unique ID"):
        detect_pairwise_consensus_correspondences(
            source,
            observed,
            available,
            material_ids=duplicate,
        )
    overflowing = np.arange(len(source), dtype=np.uint64)
    overflowing[-1] = np.uint64(np.iinfo(np.int64).max) + np.uint64(1)
    with pytest.raises(ValueError, match="signed 64-bit range"):
        detect_pairwise_consensus_correspondences(
            source,
            observed,
            available,
            material_ids=overflowing,
        )


def test_gate_rejects_bad_position_and_mask_shapes() -> None:
    source, observed, available = _identity_problem()

    with pytest.raises(ValueError, match="source positions must be numeric"):
        detect_pairwise_consensus_correspondences(
            source.astype(str),
            observed,
            available,
        )
    with pytest.raises(ValueError, match="observed positions must be numeric"):
        detect_pairwise_consensus_correspondences(
            source,
            observed.astype(str),
            available,
        )
    with pytest.raises(ValueError, match="share shape"):
        detect_pairwise_consensus_correspondences(
            source[:, :2],
            observed[:, :2],
            available,
        )
    with pytest.raises(ValueError, match="available must have shape"):
        detect_pairwise_consensus_correspondences(
            source,
            observed,
            available[:-1],
        )


def test_gate_preserves_exact_center_limit() -> None:
    source, observed, available = _identity_problem(25)
    with pytest.raises(ValueError, match="exact-consensus limit"):
        detect_pairwise_consensus_correspondences(
            source,
            observed,
            available,
        )


def test_result_normalizes_scalars_and_freezes_mask() -> None:
    result = _valid_result()

    assert result.accepted is True
    assert isinstance(result.available_count, int)
    assert isinstance(result.inlier_fraction, float)
    assert not result.inlier_mask.flags.writeable
    with pytest.raises(ValueError):
        result.inlier_mask[0] = False


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"inlier_mask": np.ones(2, dtype=np.int64)}, "only booleans"),
        ({"inlier_mask": np.ones((1, 2), dtype=np.bool_)}, "must be a vector"),
        ({"accepted": 1}, "accepted must be a boolean"),
        ({"available_count": 1.5}, "available_count must be an integer"),
        ({"available_count": -1}, "available_count is inconsistent"),
        ({"available_count": 3}, "available_count is inconsistent"),
        ({"inlier_count": 1.5}, "inlier_count must be an integer"),
        ({"inlier_count": 1}, "inlier_count is inconsistent"),
        ({"available_count": 1}, "inlier_count exceeds"),
        ({"inlier_fraction": "1"}, "inlier_fraction must be a real number"),
        ({"inlier_fraction": -0.1}, "inlier_fraction must lie"),
        ({"inlier_fraction": 1.1}, "inlier_fraction must lie"),
        ({"pair_count": 1.5}, "pair_count must be an integer"),
        ({"pair_count": -1}, "pair_count must be nonnegative"),
        (
            {"compatible_pair_fraction": np.inf},
            "compatible_pair_fraction must be finite",
        ),
        ({"compatible_pair_fraction": -0.1}, "must lie in"),
        ({"compatible_pair_fraction": 1.1}, "must lie in"),
        ({"decision": ""}, "decision must be a nonempty string"),
        ({"decision": 1}, "decision must be a nonempty string"),
        (
            {"median_inlier_normalized_strain": "0"},
            "median_inlier_normalized_strain must be a real number",
        ),
        (
            {"maximum_inlier_normalized_strain": np.inf},
            "maximum_inlier_normalized_strain must be finite",
        ),
    ],
)
def test_result_rejects_invalid_contract_fields(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_valid_result(), **changes)
