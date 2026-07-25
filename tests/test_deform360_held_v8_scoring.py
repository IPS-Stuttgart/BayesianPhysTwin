from __future__ import annotations

from copy import deepcopy
import ast
import inspect

import numpy as np
import pytest

import bayesian_phystwin.deform360_held_v8_scoring as scoring


def _arrays(identity_count: int = 50) -> dict[str, np.ndarray]:
    ids = np.arange(1000, 1000 + identity_count, dtype=np.int64)
    index = np.arange(identity_count, dtype=np.float32)
    frame_zero = np.column_stack(
        (
            np.float32(0.004) * index,
            np.float32(0.002) * (index % np.float32(7.0)),
            np.float32(0.20) + np.float32(0.001) * (index % np.float32(5.0)),
        )
    ).astype(np.float32)
    target = np.repeat(frame_zero[None], scoring.FRAME_COUNT, axis=0)
    time = np.arange(scoring.FRAME_COUNT, dtype=np.float32)[:, None]
    target[:, :, 1] += time * np.float32(0.0001)
    target[0] = frame_zero
    primary = target.copy()
    comparator = target.copy()
    comparator[1:, :, 2] += np.float32(0.002)
    visible = np.ones((scoring.FRAME_COUNT, identity_count), dtype=bool)
    valid = visible.copy()
    support = np.ones(identity_count, dtype=bool)
    excluded = np.zeros(identity_count, dtype=bool)
    excluded[: scoring.CENTER_COUNT] = True
    return {
        "primary_prediction_m": primary,
        "selected_raw_backbone_m": comparator,
        "queried_identity_ids": ids,
        "target_identity_ids": ids.copy(),
        "official_frame_zero_m": frame_zero,
        "target_points_m": target,
        "object_visibilities": visible,
        "object_motions_valid": valid,
        "shared_support_mask": support,
        "center_exclusion_mask": excluded,
        "frame_indices": np.arange(scoring.FRAME_COUNT, dtype=np.int64),
    }


def _score(
    arrays: dict[str, np.ndarray] | None = None,
    *,
    source_node_count: int | None = None,
) -> dict[str, object]:
    values = _arrays() if arrays is None else arrays
    return scoring.score_direct_official_identity_case(
        case_name="002-rope-silk-ep0003",
        object_id="002-rope-silk",
        source_node_count=source_node_count,
        **values,
    )


def _minimal_record(
    case_name: str,
    object_id: str,
    *,
    primary_chamfer: float = 0.90,
    comparator_chamfer: float = 1.0,
    primary_identity: float = 0.90,
    comparator_identity: float = 1.0,
    supported_count: int = 45,
    hidden_count: int = 32,
    excluded_count: int = scoring.CENTER_COUNT,
) -> dict[str, object]:
    official_count = 50
    return {
        "case_name": case_name,
        "object_id": object_id,
        "gate_score": {
            "primary_chamfer_m": primary_chamfer,
            "comparator_chamfer_m": comparator_chamfer,
            "primary_identity_rmse_m": primary_identity,
            "comparator_identity_rmse_m": comparator_identity,
        },
        "mask_evidence": {
            "official_identity_count": official_count,
            "supported_identity_count": supported_count,
            "support_coverage_fraction": supported_count / official_count,
            "assimilation_center_count": scoring.CENTER_COUNT,
            "center_excluded_identity_count": excluded_count,
            "hidden_supported_identity_count": hidden_count,
        },
    }


def _cohort(
    count: int,
    *,
    primary: float = 0.95,
) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    expected = {f"case-{index:02d}": f"object-{index % 3}" for index in range(count)}
    records = {
        case: _minimal_record(case, object_id, primary_chamfer=primary)
        for case, object_id in expected.items()
    }
    return expected, records


def test_module_is_version_isolated_pure_scoring_code() -> None:
    tree = ast.parse(inspect.getsource(scoring))
    imported_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module != "__future__"
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert imported_roots == {"hashlib", "math", "numpy"}
    assert imported_from == {"typing"}
    assert "open" not in called_names
    assert scoring.PROTOCOL_ID == "deform360-held-online-belief-v8.2"


def test_direct_score_uses_exact_frozen_frames_and_official_identities() -> None:
    record = _score(source_node_count=1447)

    assert record["scored_frames"] == [
        *range(20, 38),
        *range(39, 57),
        *range(58, 76),
    ]
    assert len(record["scored_frames"]) == 54
    assert record["gate_score"]["primary_chamfer_m"] == 0.0
    assert record["gate_score"]["primary_identity_rmse_m"] == 0.0
    assert record["gate_score"]["comparator_chamfer_m"] > 0.0
    assert record["gate_score"]["comparator_identity_rmse_m"] > 0.0
    assert record["paired"]["chamfer_improvement_fraction"] == 1.0
    direct = record["direct_identity"]
    assert direct["transport_performed"] is False
    assert direct["assignment_performed"] is False
    assert direct["query_performed"] is False
    assert direct["identity_order_exact"] is True


def test_one_shared_mask_excludes_unsupported_and_centers_from_both_arms() -> None:
    arrays = _arrays()
    arrays["selected_raw_backbone_m"] = arrays["target_points_m"].copy()
    arrays["shared_support_mask"][-1] = False
    arrays["primary_prediction_m"][1:, 0] += np.float32(100.0)
    arrays["selected_raw_backbone_m"][1:, -1] -= np.float32(100.0)

    record = _score(arrays)

    assert all(value == 0.0 for value in record["gate_score"].values())
    masks = record["mask_evidence"]
    assert masks["supported_identity_count"] == 49
    assert masks["assimilation_center_count"] == 16
    assert masks["center_excluded_identity_count"] == 16
    assert masks["hidden_supported_identity_count"] == 33
    assert masks["minimum_scored_identity_count"] == 33
    assert masks["single_shared_mask_for_both_arms"] is True
    assert masks["arm_specific_dropping_performed"] is False
    assert masks["density_weighting_performed"] is False
    assert len(masks["shared_support_mask_sha256"]) == 64
    assert len(masks["center_exclusion_mask_sha256"]) == 64
    assert len(masks["shared_base_mask_sha256"]) == 64
    assert len(masks["per_frame_evaluation_mask_sha256"]) == 64


def test_nonfinite_target_is_shared_masking_but_nonfinite_prediction_rejects() -> None:
    arrays = _arrays()
    arrays["target_points_m"][20, 20] = np.float32(np.nan)
    arrays["object_visibilities"][21, 21] = False
    arrays["object_motions_valid"][22, 22] = False

    record = _score(arrays)

    primary_counts = record["scores"]["primary"]["by_frame"]["identity_count"]
    comparator_counts = record["scores"]["selected_raw_backbone"]["by_frame"][
        "identity_count"
    ]
    assert primary_counts == comparator_counts
    assert primary_counts[:3] == [33, 33, 33]
    assert min(primary_counts[3:]) == 34

    broken = _arrays()
    broken["shared_support_mask"][-1] = False
    broken["primary_prediction_m"][1, -1, 0] = np.float32(np.nan)
    with pytest.raises(ValueError, match="finite globally"):
        _score(broken)


def test_identity_order_and_all_three_frame_zero_arrays_must_match_exactly() -> None:
    order_mismatch = _arrays()
    order_mismatch["target_identity_ids"] = order_mismatch["target_identity_ids"][
        ::-1
    ].copy()
    with pytest.raises(ValueError, match="identity order differs"):
        _score(order_mismatch)

    primary_x0_mismatch = _arrays()
    assert primary_x0_mismatch["primary_prediction_m"][0, 0, 0] == 0.0
    primary_x0_mismatch["primary_prediction_m"][0, 0, 0] = np.float32(-0.0)
    with pytest.raises(ValueError, match="primary frame zero.*bit-equal"):
        _score(primary_x0_mismatch)

    comparator_x0_mismatch = _arrays()
    comparator_x0_mismatch["selected_raw_backbone_m"][0, 0, 0] = np.float32(-0.0)
    with pytest.raises(ValueError, match="comparator frame zero.*bit-equal"):
        _score(comparator_x0_mismatch)

    target_x0_mismatch = _arrays()
    target_x0_mismatch["target_points_m"][0, 0, 0] = np.nextafter(
        np.float32(0.0), np.float32(1.0)
    )
    with pytest.raises(ValueError, match="target frame zero.*bit-equal"):
        _score(target_x0_mismatch)

    all_excluded = _arrays()
    all_excluded["center_exclusion_mask"][:] = True
    with pytest.raises(ValueError, match="no hidden supported identity"):
        _score(all_excluded)

    frame_index_mismatch = _arrays()
    frame_index_mismatch["frame_indices"][-1] = 74
    with pytest.raises(ValueError, match=r"arange\(76\)"):
        _score(frame_index_mismatch)


@pytest.mark.parametrize("excluded_count", [0, 1, 15, 16, 18])
def test_direct_score_accepts_variable_radius_union_exclusion_counts(
    excluded_count: int,
) -> None:
    arrays = _arrays()
    arrays["center_exclusion_mask"][:] = False
    arrays["center_exclusion_mask"][:excluded_count] = True

    record = _score(arrays)

    masks = record["mask_evidence"]
    assert masks["assimilation_center_count"] == scoring.CENTER_COUNT
    assert masks["center_excluded_identity_count"] == excluded_count
    assert masks["hidden_supported_identity_count"] == 50 - excluded_count


def test_source_cardinality_is_metadata_only_for_m_less_or_greater_than_n() -> None:
    less_than_source = _score(source_node_count=1447)
    greater_than_source = _score(source_node_count=12)

    assert less_than_source["direct_identity"]["cardinality_relation"] == (
        "official-identity-count-less-than-source-node-count"
    )
    assert greater_than_source["direct_identity"]["cardinality_relation"] == (
        "official-identity-count-greater-than-source-node-count"
    )
    assert (
        less_than_source["direct_identity"]["source_node_count_used_for_scoring"]
        is False
    )
    assert less_than_source["gate_score"] == greater_than_source["gate_score"]
    assert less_than_source["scores"] == greater_than_source["scores"]
    assert less_than_source["paired"] == greater_than_source["paired"]
    assert less_than_source["mask_evidence"] == greater_than_source["mask_evidence"]


def test_symmetric_chamfer_contains_both_euclidean_directions() -> None:
    arrays = _arrays(identity_count=18)
    arrays["selected_raw_backbone_m"] = arrays["target_points_m"].copy()
    for frame in scoring.SCORED_FRAMES:
        arrays["target_points_m"][frame, 16:, 1:] = np.float32(0.0)
        arrays["selected_raw_backbone_m"][frame, 16:, 1:] = np.float32(0.0)
        arrays["primary_prediction_m"][frame, 16:, 1:] = np.float32(0.0)
        arrays["target_points_m"][frame, 16:, 0] = np.array(
            [0.0, 2.0], dtype=np.float32
        )
        arrays["selected_raw_backbone_m"][frame, 16:, 0] = np.array(
            [0.0, 2.0], dtype=np.float32
        )
        arrays["primary_prediction_m"][frame, 16:, 0] = np.array(
            [0.0, 0.1], dtype=np.float32
        )

    record = _score(arrays)
    by_frame = record["scores"]["primary"]["by_frame"]

    assert by_frame["prediction_to_target_euclidean_chamfer_m"] == pytest.approx(
        [0.05] * 54
    )
    assert by_frame["target_to_prediction_euclidean_chamfer_m"] == pytest.approx(
        [0.95] * 54
    )
    assert by_frame["symmetric_euclidean_chamfer_m"] == pytest.approx([0.5] * 54)
    assert record["gate_score"]["primary_chamfer_m"] == pytest.approx(0.5)


def test_temporal_aggregation_is_equal_per_frame_not_pooled() -> None:
    arrays = _arrays()
    arrays["primary_prediction_m"][20, 20, 0] += np.float32(1.0)

    record = _score(arrays)

    frame_rmse = np.sqrt(1.0 / (34 * 3))
    expected = frame_rmse / 54
    assert record["gate_score"]["primary_identity_rmse_m"] == pytest.approx(expected)


def test_equal_case_and_equal_object_aggregation_are_both_explicit() -> None:
    expected = {"c1": "o1", "c2": "o1", "c3": "o2"}
    records = {
        "c1": _minimal_record("c1", "o1", primary_chamfer=1.0, comparator_chamfer=2.0),
        "c2": _minimal_record("c2", "o1", primary_chamfer=3.0, comparator_chamfer=4.0),
        "c3": _minimal_record("c3", "o2", primary_chamfer=9.0, comparator_chamfer=10.0),
    }

    result = scoring.aggregate_equal_case_and_object(
        records, expected_case_to_object=expected
    )

    assert result["equal_case_mean"]["primary_chamfer_m"] == pytest.approx(13 / 3)
    assert result["by_object_equal_case_mean"]["o1"]["primary_chamfer_m"] == 2.0
    assert result["by_object_equal_case_mean"]["o2"]["primary_chamfer_m"] == 9.0
    assert result["equal_object_mean"]["primary_chamfer_m"] == 5.5
    assert result["equal_object_mean"]["comparator_chamfer_m"] == 6.5
    assert result["weighting"]["point_or_visibility_density"] == "none"

    mismatched = deepcopy(records)
    mismatched["c2"]["object_id"] = "wrong"
    with pytest.raises(ValueError, match="case object changed"):
        scoring.aggregate_equal_case_and_object(
            mismatched, expected_case_to_object=expected
        )


def test_calibration_gate_passes_exact_boundaries_and_enforces_support() -> None:
    expected, records = _cohort(scoring.CALIBRATION_CASE_COUNT, primary=0.95)

    result = scoring.evaluate_calibration_gate(
        records, expected_case_to_object=expected
    )

    assert result["passed"] is True
    assert result["case_chamfer_wins"] == 15
    assert result["minimum_observed_support_coverage_fraction"] == 0.90
    assert result["minimum_observed_hidden_supported_identity_count"] == 32
    assert all(result["checks"].values())

    below_coverage = deepcopy(records)
    failed_case = next(iter(expected))
    below_coverage[failed_case]["mask_evidence"]["supported_identity_count"] = 44
    below_coverage[failed_case]["mask_evidence"]["support_coverage_fraction"] = 0.88
    below_coverage[failed_case]["mask_evidence"]["hidden_supported_identity_count"] = 31
    failed = scoring.evaluate_calibration_gate(
        below_coverage, expected_case_to_object=expected
    )
    assert failed["passed"] is False
    assert failed_case in failed["support_coverage_failures"]
    assert failed["checks"]["all_cases_support_coverage_at_least_0_90"] is False

    below_hidden = deepcopy(records)
    below_hidden[failed_case]["mask_evidence"]["hidden_supported_identity_count"] = 31
    failed = scoring.evaluate_calibration_gate(
        below_hidden, expected_case_to_object=expected
    )
    assert failed["checks"]["all_cases_hidden_supported_count_at_least_32"] is False


def test_calibration_gate_v7_win_and_regression_boundaries() -> None:
    expected, records = _cohort(scoring.CALIBRATION_CASE_COUNT, primary=1.0)
    for case in list(expected)[:10]:
        records[case]["gate_score"]["primary_chamfer_m"] = 0.90
    at_ten_wins = scoring.evaluate_calibration_gate(
        records, expected_case_to_object=expected
    )
    assert at_ten_wins["case_chamfer_wins"] == 10
    assert at_ten_wins["passed"] is True

    nine_wins = deepcopy(records)
    for case in expected:
        nine_wins[case]["gate_score"]["primary_chamfer_m"] = 1.0
    for case in list(expected)[:9]:
        nine_wins[case]["gate_score"]["primary_chamfer_m"] = 0.80
    failed = scoring.evaluate_calibration_gate(
        nine_wins, expected_case_to_object=expected
    )
    assert failed["case_chamfer_wins"] == 9
    assert failed["checks"]["at_least_10_of_15_chamfer_wins"] is False

    regression_boundary = deepcopy(records)
    first = next(iter(expected))
    regression_boundary[first]["gate_score"]["primary_chamfer_m"] = 1.10
    for case in list(expected)[1:]:
        regression_boundary[case]["gate_score"]["primary_chamfer_m"] = 0.80
    boundary = scoring.evaluate_calibration_gate(
        regression_boundary, expected_case_to_object=expected
    )
    assert boundary["checks"]["no_case_over_10_percent_chamfer_regression"] is True

    regression_boundary[first]["gate_score"]["primary_chamfer_m"] = 1.100001
    failed = scoring.evaluate_calibration_gate(
        regression_boundary, expected_case_to_object=expected
    )
    assert failed["checks"]["no_case_over_10_percent_chamfer_regression"] is False


def test_confirmation_gate_requires_exact_six_wins_and_sign_test() -> None:
    expected, records = _cohort(scoring.CONFIRMATION_CASE_COUNT, primary=0.95)

    result = scoring.evaluate_confirmation_gate(
        records, expected_case_to_object=expected
    )

    assert result["passed"] is True
    assert result["case_chamfer_wins"] == 6
    assert result["one_sided_sign_test_p"] == 1.0 / 64.0
    assert all(result["checks"].values())

    one_tie = deepcopy(records)
    one_tie[next(iter(expected))]["gate_score"]["primary_chamfer_m"] = 1.0
    failed = scoring.evaluate_confirmation_gate(
        one_tie, expected_case_to_object=expected
    )
    assert failed["passed"] is False
    assert failed["case_chamfer_wins"] == 5
    assert failed["one_sided_sign_test_p"] == 7.0 / 64.0
    assert failed["checks"]["all_6_cases_chamfer_win"] is False
    assert failed["checks"]["one_sided_sign_test_p_is_1_over_64"] is False


def test_gate_rejects_any_missing_or_extra_exact_case_record() -> None:
    expected, records = _cohort(scoring.CONFIRMATION_CASE_COUNT)
    records.pop(next(iter(expected)))

    with pytest.raises(ValueError, match="exact declared cohort"):
        scoring.evaluate_confirmation_gate(records, expected_case_to_object=expected)
