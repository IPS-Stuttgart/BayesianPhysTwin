#!/usr/bin/env python3
"""Validate the frozen public Deform360 Transport4D confirmation design.

This validator is deliberately target blind. It reads only the committed design,
metadata reserve, and carrier-support records. It cannot load robot, tactile,
image, geometry, prediction, residual, or confirmation-outcome payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

DESIGN_SCHEMA = "bayesian-phystwin.transport4d_deform360_confirmation_design"
RESERVE_SCHEMA = "bayesian_phystwin.transport4d_deform360_reserve"
SUPPORT_SCHEMA = "bayesian-phystwin.transport4d_deform360_support_audit"
EXPECTED_PROTOCOL_ID = "transport4d-deform360-confirmation-design-v1-20260902"
EXPECTED_RESERVE_ID = (
    "0fba6a0cda4ac23fc8f900ca4632d73a22ae0559d42e48cc2ec8c5ea79030dc4"
)
EXPECTED_SUPPORT_ID = (
    "57aaea3215ad1cdf76accfc455c173bb5d84a926bfed013fa64b6ec3625befed"
)
EXPECTED_FULL_SUPPORT_SHA256 = (
    "db6737e04d40144d18f1f0d316d3035705384c4069546449d0567ee231dc2334"
)
EXPECTED_TIER_ORDER = [
    "exact_coefficients",
    "query_identifiable_effect",
    "low_dimensional_correction",
    "uncertainty_only",
    "procedure_only",
    "unsupported",
]


class DesignError(RuntimeError):
    """Raised when the frozen confirmation design or its bindings drift."""


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DesignError(f"expected a JSON object: {path}")
    return value


def canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def zero_harm_upper_bound(accepted_count: int, alpha: float = 0.05) -> float:
    """Exact one-sided Clopper-Pearson upper endpoint for zero harms."""

    if isinstance(accepted_count, bool) or accepted_count < 1:
        raise ValueError("accepted_count must be a positive integer")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return 1.0 - alpha ** (1.0 / accepted_count)


def minimum_zero_harm_accepts(
    maximum_rate: float = 0.1,
    alpha: float = 0.05,
) -> int:
    """Smallest accepted-object count giving a zero-harm upper bound at rate."""

    if not 0.0 < maximum_rate < 1.0:
        raise ValueError("maximum_rate must lie in (0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return int(math.ceil(math.log(alpha) / math.log(1.0 - maximum_rate)))


def minimum_sign_test_wins(
    object_count: int,
    alpha: float = 0.05,
) -> tuple[int, float]:
    """Return the smallest strict-win count with one-sided Bin(n, .5) tail <= alpha."""

    if isinstance(object_count, bool) or object_count < 1:
        raise ValueError("object_count must be a positive integer")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    denominator = float(2**object_count)
    for wins in range(object_count + 1):
        tail = sum(
            math.comb(object_count, value)
            for value in range(wins, object_count + 1)
        ) / denominator
        if tail <= alpha:
            return wins, tail
    raise RuntimeError("no finite sign-test threshold found")


def _require_false(mapping: dict[str, Any], names: tuple[str, ...]) -> None:
    for name in names:
        if mapping.get(name) is not False:
            raise DesignError(f"information boundary changed: {name}")


def _validate_design(design: dict[str, Any]) -> None:
    if design.get("schema") != DESIGN_SCHEMA:
        raise DesignError("unexpected confirmation-design schema")
    if design.get("schema_version") != 1:
        raise DesignError("unexpected confirmation-design schema version")
    if design.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise DesignError("confirmation-design protocol identity changed")
    if design.get("status") != (
        "frozen-before-calibration-policy-binding-and-before-confirmation-numeric-access"
    ):
        raise DesignError("confirmation design is not target closed")

    reserve = design.get("reserve_binding")
    support = design.get("support_binding")
    if not isinstance(reserve, dict) or not isinstance(support, dict):
        raise DesignError("reserve/support bindings are missing")
    if reserve.get("reserve_id") != EXPECTED_RESERVE_ID:
        raise DesignError("reserve ID changed")
    if support.get("support_id") != EXPECTED_SUPPORT_ID:
        raise DesignError("support ID changed")
    if support.get("full_support_result_sha256") != EXPECTED_FULL_SUPPORT_SHA256:
        raise DesignError("full support-result identity changed")
    if support.get("supported_calibration_object_count") != 15:
        raise DesignError("supported calibration count changed")
    if support.get("supported_confirmation_object_count") != 33:
        raise DesignError("supported confirmation count changed")
    for name in (
        "include_every_supported_confirmation_object",
        "support_negative_objects_retained",
    ):
        if support.get(name) is not True:
            raise DesignError(f"support rule weakened: {name}")
    for name in ("replacement_allowed", "split_movement_allowed"):
        if support.get(name) is not False:
            raise DesignError(f"support rule weakened: {name}")

    tiers = design.get("fixed_transport_tiers")
    if not isinstance(tiers, list):
        raise DesignError("fixed transport tiers are missing")
    tier_order = [row.get("tier") for row in tiers if isinstance(row, dict)]
    if tier_order != EXPECTED_TIER_ORDER:
        raise DesignError("transport tier order changed")
    if tiers[-1].get("action") != "exact-registered-fallback":
        raise DesignError("unsupported tier no longer returns exact fallback")

    query = design.get("registered_query")
    if not isinstance(query, dict):
        raise DesignError("registered query is missing")
    expected_query = {
        "dimension": 16,
        "sensor_count": 4,
        "future_horizon_frames": 32,
        "target_window_start_frame": 16,
        "target_frame": 48,
        "prefix_tactile_indices": [0, 12, 15, 16],
        "known_future_robot_segment_is_intervention_input": True,
        "future_target_tactile_is_scoring_only": True,
    }
    for name, expected in expected_query.items():
        if query.get(name) != expected:
            raise DesignError(f"registered query changed: {name}")

    order = design.get("information_order")
    if not isinstance(order, list) or [row.get("stage") for row in order] != [
        "bind-calibration-policy",
        "confirmation-fit-and-predict",
        "joint-prediction-seal",
        "confirmation-score-once",
    ]:
        raise DesignError("confirmation information order changed")
    seal = order[2]
    if seal.get("all_objects_must_be_sealed_before_any_target_outcome") is not True:
        raise DesignError("joint prediction seal was weakened")
    score = order[3]
    for name in (
        "target_tuning_allowed",
        "tier_reselection_allowed",
        "object_replacement_allowed",
        "retry_after_scientific_outcome_allowed",
    ):
        if score.get(name) is not False:
            raise DesignError(f"score-once rule weakened: {name}")

    statistics = design.get("statistics")
    if not isinstance(statistics, dict):
        raise DesignError("statistics block is missing")
    if statistics.get("independent_unit") != "physical-object":
        raise DesignError("independent statistical unit changed")
    if statistics.get("registered_confirmation_object_count") != 33:
        raise DesignError("registered confirmation object count changed")
    if statistics.get("minimum_accepted_objects_for-sub-10-percent-zero-harm-bound") != 29:
        raise DesignError("registered harm-bound sample size changed")
    if not math.isclose(
        float(statistics.get("zero_harm_upper_bound_if_all_33_are_accepted")),
        zero_harm_upper_bound(33),
        rel_tol=0.0,
        abs_tol=5e-11,
    ):
        raise DesignError("stored zero-harm upper bound is incorrect")

    boundary = design.get("information_boundary")
    if not isinstance(boundary, dict):
        raise DesignError("information boundary is missing")
    _require_false(
        boundary,
        (
            "confirmation_robot_numeric_payloads_opened",
            "confirmation_tactile_numeric_payloads_opened",
            "confirmation_camera_pixels_opened",
            "confirmation_geometry_or_point_cloud_opened",
            "confirmation_target_outcomes_opened",
            "confirmation_predictions_sealed",
            "confirmation_authorized",
            "paper_claim_authorized",
            "deployment_authorized",
        ),
    )


def _validate_bound_records(
    design: dict[str, Any],
    reserve: dict[str, Any],
    support: dict[str, Any],
) -> None:
    if reserve.get("schema") != RESERVE_SCHEMA:
        raise DesignError("unexpected reserve-result schema")
    if reserve.get("reserve_id") != EXPECTED_RESERVE_ID:
        raise DesignError("bound reserve result changed")
    if reserve.get("reservation_ready") is not True:
        raise DesignError("bound reserve is not ready")
    reserve_boundary = reserve.get("information_boundary")
    if not isinstance(reserve_boundary, dict):
        raise DesignError("reserve information boundary is missing")
    _require_false(
        reserve_boundary,
        (
            "robot_numeric_payload_opened",
            "tactile_numeric_payload_opened",
            "camera_pixel_opened",
            "geometry_or_point_cloud_opened",
            "target_outcome_opened",
            "confirmation_authorized",
            "paper_claim_authorized",
        ),
    )

    if support.get("schema") != SUPPORT_SCHEMA:
        raise DesignError("unexpected support-result schema")
    if support.get("support_id") != EXPECTED_SUPPORT_ID:
        raise DesignError("bound support result changed")
    if support.get("numeric_confirmation_feasible") is not True:
        raise DesignError("bound support result does not permit a full cohort")
    if support.get("supported_calibration_object_count") != 15:
        raise DesignError("bound calibration support count changed")
    if support.get("supported_confirmation_object_count") != 33:
        raise DesignError("bound confirmation support count changed")
    supported_ids = support.get("supported_confirmation_object_ids")
    if not isinstance(supported_ids, list) or len(supported_ids) != 33:
        raise DesignError("bound confirmation roster is malformed")
    if len(set(map(str, supported_ids))) != 33:
        raise DesignError("bound confirmation roster contains duplicates")
    support_boundary = support.get("information_boundary")
    if not isinstance(support_boundary, dict):
        raise DesignError("support information boundary is missing")
    _require_false(
        support_boundary,
        (
            "robot_numeric_payload_opened",
            "tactile_numeric_payload_opened",
            "camera_pixel_opened",
            "geometry_or_point_cloud_opened",
            "target_outcome_opened",
            "replacement_allowed",
            "split_movement_allowed",
            "confirmation_authorized",
            "paper_claim_authorized",
        ),
    )

    design_support = design["support_binding"]
    if design_support["supported_confirmation_object_count"] != len(supported_ids):
        raise DesignError("design and support roster counts disagree")


def validate(
    design_path: Path,
    reserve_path: Path,
    support_path: Path,
) -> dict[str, Any]:
    design = read_json(design_path)
    reserve = read_json(reserve_path)
    support = read_json(support_path)
    _validate_design(design)
    _validate_bound_records(design, reserve, support)

    minimum_accepts = minimum_zero_harm_accepts()
    sign_wins, sign_tail = minimum_sign_test_wins(33)
    result: dict[str, Any] = {
        "schema": "bayesian-phystwin.transport4d_deform360_confirmation_design_validation",
        "schema_version": 1,
        "protocol_id": design["protocol_id"],
        "reserve_id": reserve["reserve_id"],
        "support_id": support["support_id"],
        "status": "confirmation-design-valid-and-target-closed",
        "supported_calibration_object_count": 15,
        "supported_confirmation_object_count": 33,
        "minimum_zero_harm_accepts_for_95pct_upper_below_10pct": minimum_accepts,
        "zero_harm_95pct_upper_at_29_accepts": zero_harm_upper_bound(29),
        "zero_harm_95pct_upper_at_33_accepts": zero_harm_upper_bound(33),
        "minimum_strict_wins_for_one_sided_33_object_sign_test": sign_wins,
        "sign_test_tail_probability_at_threshold": sign_tail,
        "confirmation_numeric_payload_opened": False,
        "confirmation_authorized": False,
        "paper_claim_authorized": False,
        "claim_boundary": design["claim_boundary"],
    }
    result["validation_id"] = canonical_digest(result)
    return result


def render_report(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Transport4D Deform360 confirmation-design validation",
            "",
            f"Status: **{result['status']}**",
            "",
            f"- Frozen calibration objects: **{result['supported_calibration_object_count']}**",
            f"- Frozen confirmation objects: **{result['supported_confirmation_object_count']}**",
            "- Minimum zero-harm accepted objects for a one-sided 95% upper "
            f"bound below 10%: **{result['minimum_zero_harm_accepts_for_95pct_upper_below_10pct']}**",
            "- Zero-harm upper bound at 29 accepted objects: "
            f"**{100 * result['zero_harm_95pct_upper_at_29_accepts']:.3f}%**",
            "- Zero-harm upper bound at all 33 accepted objects: "
            f"**{100 * result['zero_harm_95pct_upper_at_33_accepts']:.3f}%**",
            "- Minimum strict wins for a one-sided 33-object sign test at 5%: "
            f"**{result['minimum_strict_wins_for_one_sided_33_object_sign_test']}**",
            "- Confirmation numerical access: **false**",
            "- Confirmation authorization: **false**",
            "",
            f"Validation ID: `{result['validation_id']}`",
            "",
            result["claim_boundary"],
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--reserve", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    result = validate(args.design, args.reserve, args.support)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if args.output_report is not None:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
