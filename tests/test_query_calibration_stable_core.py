from pathlib import Path

import numpy as np
import test_query_calibration as core
from test_query_calibration_validation import (
    test_calibrate_query_covariance_rejects_malformed_inputs,
    test_calibrate_query_covariance_supports_batches,
    test_covariance_transform_scalars_fail_closed,
    test_fit_requires_one_residual_and_covariance_entry_per_group,
    test_from_dict_rejects_nonmapping_or_incomplete_records,
    test_from_dict_rejects_unsupported_or_malformed_values,
    test_from_dict_rejects_valid_but_content_tampered_record,
    test_load_rejects_nonobject_json,
    test_query_array_contract_rejects_malformed_inputs,
    test_query_calibration_contract_rejects_malformed_fields,
)


def test_query_calibration_stable_core_coverage(tmp_path: Path) -> None:
    core.test_fit_uses_one_maximum_score_per_independent_group()
    core.test_calibrated_covariance_and_group_coverage_share_one_scale()
    core.test_group_score_is_invariant_to_row_order_and_duplicate_nonmaxima()
    core.test_affine_coordinate_change_preserves_mahalanobis_score()
    core.test_covariance_scale_is_compensated_by_conformal_quantile()
    core.test_group_order_is_canonical_and_content_invariant()
    core.test_impossible_coverage_fails_before_outcomes_are_inspected()
    core.test_policy_selection_with_calibration_outcomes_is_rejected()
    core.test_save_load_roundtrip_and_tamper_detection(tmp_path)
    core.test_duplicate_json_keys_fail_closed(tmp_path)
    for residual, covariance, message in (
        (np.asarray([1.0, 2.0]), np.eye(3), "shape"),
        (np.asarray([1.0, np.nan]), np.eye(2), "residual must be finite"),
        (
            np.asarray([1.0, 2.0]),
            np.asarray([[1.0, 1.0], [0.0, 1.0]]),
            "symmetric",
        ),
        (
            np.asarray([1.0, 2.0]),
            np.asarray([[1.0, 0.0], [0.0, 0.0]]),
            "positive definite",
        ),
    ):
        core.test_invalid_query_geometry_fails_closed(
            residual,
            covariance,
            message,
        )
    core.test_contract_rejects_changed_evidence_or_derived_quantile()
    core.test_duplicate_group_ids_and_noninteger_schema_versions_fail_closed(
        tmp_path
    )

    for kwargs, message in (
        ({"covariance_scale": True}, "finite real"),
        ({"covariance_scale": "1"}, "finite real"),
        ({"covariance_scale": np.nan}, "finite real"),
        ({"covariance_scale": 0.0}, "positive"),
        ({"isotropic_variance": -1.0}, "nonnegative"),
    ):
        test_covariance_transform_scalars_fail_closed(kwargs, message)
    for residual, covariance, message in (
        (np.asarray(["1"]), np.eye(1), "residual.*real numeric"),
        (np.ones(1), np.asarray([["1"]]), "covariance.*real numeric"),
        (np.ones((1, 1, 1)), np.ones((1, 1, 1)), "residual must have shape"),
        (np.ones((1, 1)), np.ones((1, 1, 1, 1)), "covariance must have shape"),
        (np.empty((0, 1)), np.empty((0, 1, 1)), "cannot be empty"),
        (np.empty((1, 0)), np.empty((1, 0, 0)), "cannot be empty"),
        (np.ones(1), np.asarray([[np.inf]]), "covariance must be finite"),
    ):
        test_query_array_contract_rejects_malformed_inputs(
            residual,
            covariance,
            message,
        )
    for changes, message in (
        (
            {
                "calibration_group_ids": (
                    "object-00",
                    "object-00",
                    *tuple(f"object-{index:02d}" for index in range(2, 9)),
                )
            },
            "unique",
        ),
        ({"calibration_group_scores": np.asarray(["1"] * 9)}, "real array"),
        ({"calibration_group_scores": np.ones((9, 1))}, "real array"),
        ({"calibration_group_scores": np.ones(8)}, "one score per group"),
        (
            {"calibration_group_scores": np.asarray([*range(1, 9), np.nan])},
            "finite and nonnegative",
        ),
        (
            {"calibration_group_scores": np.asarray([*range(1, 9), -1.0])},
            "finite and nonnegative",
        ),
        ({"predictor_frozen_before_scores": np.bool_(True)}, "must be a boolean"),
        (
            {"calibration_outcomes_used_for_selection": np.bool_(False)},
            "must be a boolean",
        ),
        ({"finite_sample_rank": 8}, "finite-group conformal rank"),
    ):
        test_query_calibration_contract_rejects_malformed_fields(changes, message)
    test_from_dict_rejects_nonmapping_or_incomplete_records()
    for field, value, message in (
        ("schema", "other", "unsupported.*schema"),
        ("schema_version", 2, "schema version"),
        ("score", "other", "unsupported.*score"),
        ("calibration_groups", (), "nonempty list"),
        ("calibration_groups", [], "nonempty list"),
        ("calibration_groups", ["bad"], "group_id and score"),
        ("calibration_groups", [{"group_id": "only"}], "group_id and score"),
    ):
        test_from_dict_rejects_unsupported_or_malformed_values(
            field,
            value,
            message,
        )
    test_from_dict_rejects_valid_but_content_tampered_record()
    for missing in ("residual", "covariance"):
        test_fit_requires_one_residual_and_covariance_entry_per_group(missing)
    for covariance, message in (
        (np.asarray(["1"]), "real numeric"),
        (np.asarray(1.0), "one or more square"),
        (np.empty((0, 0)), "one or more square"),
        (np.ones((2, 3)), "must be square"),
        (np.empty((0, 2, 2)), "nonempty and finite"),
        (np.asarray([[np.inf]]), "nonempty and finite"),
    ):
        test_calibrate_query_covariance_rejects_malformed_inputs(
            covariance,
            message,
        )
    test_calibrate_query_covariance_supports_batches()
    test_load_rejects_nonobject_json(tmp_path)
