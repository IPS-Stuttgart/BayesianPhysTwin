"""Generate a deterministic finite-group capability-atlas calibration study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.decision_capability_atlas_v1 import (
    affine_capability_halfspaces,
    affine_decision_capability_atlas,
    capability_polygon_2d,
    polygon_area_2d,
)
from bayesian_phystwin.decision_capability_calibration_v1 import (
    DECISION_CAPABILITY_CALIBRATION_CLAIM_BOUNDARY,
    affine_box_pairwise_undercoverage_score,
    finite_group_atlas_calibration,
    statistically_corrected_halfspaces,
)
from bayesian_phystwin.decision_capability_task_uncertainty_v1 import (
    box_robust_center_halfspaces,
)

ACTIONS = ("pull_left", "hold", "pull_right")
TASK_BOUNDS = np.array([[-1.5, 1.5], [0.0, 4.0]], dtype=np.float64)
OBJECTIVE_HALF_WIDTH = np.array([0.1, 0.2], dtype=np.float64)
CALIBRATION_ALPHA = 0.1
CALIBRATION_DELTAS = np.array(
    [
        0.0,
        0.0,
        0.01,
        0.02,
        0.02,
        0.03,
        0.04,
        0.04,
        0.05,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.10,
        0.11,
        0.12,
        0.15,
        0.20,
    ],
    dtype=np.float64,
)


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def controlled_family() -> tuple[np.ndarray, ...]:
    displacement = np.array(
        [
            [-1.1, -0.1, 0.7],
            [-0.7, 0.1, 1.1],
            [-1.0, 0.0, 0.6],
            [-0.6, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    physical_risk = np.array(
        [
            [0.4, 0.05, 0.8],
            [0.8, 0.05, 0.4],
            [0.5, 0.02, 0.9],
            [0.9, 0.02, 0.5],
        ],
        dtype=np.float64,
    )
    return (
        np.full(4, 0.25, dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
        np.array([0, 0, 1, 1], dtype=np.int64),
        np.square(displacement),
        np.stack((-2.0 * displacement, physical_risk), axis=2),
    )


def action_regions(
    prior: np.ndarray,
    quotient: np.ndarray,
    classes: np.ndarray,
    intercept: np.ndarray,
    coefficient: np.ndarray,
    *,
    correction: float,
    objective_half_width: np.ndarray | None,
) -> tuple[list[dict[str, Any]], float, np.ndarray]:
    bounds = TASK_BOUNDS.copy()
    if objective_half_width is not None:
        bounds[:, 0] += objective_half_width
        bounds[:, 1] -= objective_half_width
    domain_area = float(np.prod(bounds[:, 1] - bounds[:, 0]))
    records: list[dict[str, Any]] = []
    masks = []
    x_values = np.linspace(bounds[0, 0], bounds[0, 1], 121)
    y_values = np.linspace(bounds[1, 0], bounds[1, 1], 161)
    grid = np.asarray(
        [(x, y) for y in y_values for x in x_values],
        dtype=np.float64,
    )
    for action_index, action in enumerate(ACTIONS):
        region = affine_capability_halfspaces(
            prior,
            quotient,
            classes,
            intercept,
            coefficient,
            action_index=action_index,
            regret_tolerance=0.0,
        )
        if objective_half_width is not None:
            region = box_robust_center_halfspaces(region, objective_half_width)
        if correction > 0.0:
            region = statistically_corrected_halfspaces(region, correction)
        polygon = capability_polygon_2d(region, bounds)
        area = polygon_area_2d(polygon)
        records.append(
            {
                "action": action,
                "area": area,
                "area_fraction": area / domain_area,
                "vertices": polygon.tolist(),
            }
        )
        masks.append(region.contains(grid))
    mask = np.stack(masks, axis=1)
    return records, domain_area, mask


def calibration_scores(
    prior: np.ndarray,
    quotient: np.ndarray,
    classes: np.ndarray,
    intercept: np.ndarray,
    coefficient: np.ndarray,
) -> list[dict[str, Any]]:
    region = affine_capability_halfspaces(
        prior,
        quotient,
        classes,
        intercept,
        coefficient,
        action_index=0,
        regret_tolerance=0.0,
    )
    benchmark = np.asarray(region.benchmark_action_index)
    unique_benchmark = np.unique(benchmark)
    center = np.mean(TASK_BOUNDS, axis=1)
    selected_benchmark = int(unique_benchmark[0])
    realized_intercept_base = np.zeros(len(ACTIONS), dtype=np.float64)
    realized_coefficient = np.zeros(
        (len(ACTIONS), TASK_BOUNDS.shape[0]),
        dtype=np.float64,
    )
    for benchmark_action in unique_benchmark:
        rows = np.flatnonzero(benchmark == benchmark_action)
        model_intercepts = region.regret_tolerance - region.offset[rows]
        model_values = model_intercepts + region.normal[rows] @ center
        selected_row = int(rows[int(np.argmax(model_values))])
        realized_intercept_base[benchmark_action] = (
            region.regret_tolerance - region.offset[selected_row] - 10.0
        )
        realized_coefficient[benchmark_action] = region.normal[selected_row]
        if int(benchmark_action) == selected_benchmark:
            realized_intercept_base[benchmark_action] += 10.0

    records: list[dict[str, Any]] = []
    for group_index, delta in enumerate(CALIBRATION_DELTAS):
        realized_intercept = realized_intercept_base.copy()
        realized_intercept[selected_benchmark] += float(delta)
        report = affine_box_pairwise_undercoverage_score(
            region,
            realized_intercept,
            realized_coefficient,
            TASK_BOUNDS,
        )
        records.append(
            {
                "group_index": group_index,
                "registered_discrepancy": float(delta),
                "computed_nonconformity_score": report.nonnegative_score,
                "critical_benchmark_action": ACTIONS[
                    report.critical_benchmark_action_index
                ],
                "critical_task_parameter": report.critical_task_parameter.tolist(),
            }
        )
    return records


def region_summary(
    records: list[dict[str, Any]],
    domain_area: float,
    mask: np.ndarray,
) -> dict[str, Any]:
    total_area = float(sum(float(record["area"]) for record in records))
    return {
        "domain_area": domain_area,
        "action_regions": records,
        "certified_union_area": total_area,
        "certified_union_fraction": total_area / domain_area,
        "fallback_area": domain_area - total_area,
        "fallback_fraction": 1.0 - total_area / domain_area,
        "grid_task_count": int(mask.shape[0]),
        "grid_overlap_count": int(np.count_nonzero(np.sum(mask, axis=1) > 1)),
        "grid_capable_count": int(np.count_nonzero(np.any(mask, axis=1))),
    }


def strictness_witness(
    prior: np.ndarray,
    quotient: np.ndarray,
    classes: np.ndarray,
    intercept: np.ndarray,
    coefficient: np.ndarray,
    correction: float,
) -> dict[str, Any]:
    x_values = np.linspace(TASK_BOUNDS[0, 0], TASK_BOUNDS[0, 1], 301)
    y_values = np.linspace(TASK_BOUNDS[1, 0], TASK_BOUNDS[1, 1], 401)
    grid = np.asarray(
        [(x, y) for y in y_values for x in x_values],
        dtype=np.float64,
    )
    nominal = affine_decision_capability_atlas(
        prior,
        quotient,
        classes,
        intercept,
        coefficient,
        grid,
        regret_tolerance=0.0,
    )
    corrected_masks = []
    for action_index in range(len(ACTIONS)):
        region = statistically_corrected_halfspaces(
            affine_capability_halfspaces(
                prior,
                quotient,
                classes,
                intercept,
                coefficient,
                action_index=action_index,
                regret_tolerance=0.0,
            ),
            correction,
        )
        corrected_masks.append(region.contains(grid))
    corrected = np.stack(corrected_masks, axis=1)
    nominal_unique = np.sum(nominal.robustly_optimal_action_mask, axis=1) == 1
    corrected_none = ~np.any(corrected, axis=1)
    candidates = np.flatnonzero(nominal_unique & corrected_none)
    if candidates.size == 0:
        raise RuntimeError("no nominal-only strictness witness exists")
    index = int(candidates[candidates.size // 2])
    nominal_actions = np.flatnonzero(nominal.robustly_optimal_action_mask[index])
    return {
        "task_parameter": grid[index].tolist(),
        "nominal_actions": [ACTIONS[int(value)] for value in nominal_actions],
        "statistically_corrected_actions": [],
    }


def generate_result() -> dict[str, Any]:
    prior, quotient, classes, intercept, coefficient = controlled_family()
    score_records = calibration_scores(
        prior,
        quotient,
        classes,
        intercept,
        coefficient,
    )
    scores = [record["computed_nonconformity_score"] for record in score_records]
    calibration = finite_group_atlas_calibration(
        scores,
        alpha=CALIBRATION_ALPHA,
    )

    nominal_records, nominal_domain, nominal_mask = action_regions(
        prior,
        quotient,
        classes,
        intercept,
        coefficient,
        correction=0.0,
        objective_half_width=None,
    )
    corrected_records, corrected_domain, corrected_mask = action_regions(
        prior,
        quotient,
        classes,
        intercept,
        coefficient,
        correction=calibration.correction,
        objective_half_width=None,
    )
    combined_records, combined_domain, combined_mask = action_regions(
        prior,
        quotient,
        classes,
        intercept,
        coefficient,
        correction=calibration.correction,
        objective_half_width=OBJECTIVE_HALF_WIDTH,
    )
    nominal_summary = region_summary(nominal_records, nominal_domain, nominal_mask)
    corrected_summary = region_summary(
        corrected_records,
        corrected_domain,
        corrected_mask,
    )
    combined_summary = region_summary(combined_records, combined_domain, combined_mask)

    result: dict[str, Any] = {
        "schema": "bayesian-phystwin/decision-capability-calibration-study-v1",
        "schema_version": 1,
        "status": "complete",
        "task_family": {
            "parameters": ["target_displacement", "physical_risk_weight"],
            "bounds": TASK_BOUNDS.tolist(),
            "objective_half_width": OBJECTIVE_HALF_WIDTH.tolist(),
        },
        "calibration": {
            **calibration.summary(),
            "group_score_construction": (
                "one exact continuous-task maximum per synthetic calibration group"
            ),
            "groups": score_records,
        },
        "nominal_atlas": nominal_summary,
        "statistically_corrected_atlas": corrected_summary,
        "objective_and_statistically_corrected_atlas": combined_summary,
        "strictness_witness": strictness_witness(
            prior,
            quotient,
            classes,
            intercept,
            coefficient,
            calibration.correction,
        ),
        "checks": {
            "score_matches_registered_discrepancy": bool(
                np.allclose(scores, CALIBRATION_DELTAS, atol=1e-9, rtol=0.0)
            ),
            "finite_sample_rank_attained": (
                calibration.quantile_rank <= calibration.calibration_group_count
            ),
            "statistical_correction_positive": calibration.correction > 0.0,
            "statistical_atlas_strictly_smaller": (
                corrected_summary["certified_union_fraction"]
                < nominal_summary["certified_union_fraction"]
            ),
            "combined_atlas_strictly_smaller": (
                combined_summary["certified_union_fraction"]
                < corrected_summary["certified_union_fraction"]
            ),
            "no_positive_area_overlap_on_grid": (
                nominal_summary["grid_overlap_count"] == 0
                and corrected_summary["grid_overlap_count"] == 0
                and combined_summary["grid_overlap_count"] == 0
            ),
        },
        "claim_boundary": DECISION_CAPABILITY_CALIBRATION_CLAIM_BOUNDARY,
        "evidence_role": (
            "controlled mechanism evidence; not real-data calibration or deployment evidence"
        ),
    }
    result["result_id"] = canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    result = generate_result()
    if not all(result["checks"].values()):
        raise SystemExit("statistical capability-atlas checks failed")
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.check is not None:
        if args.check.read_text(encoding="utf-8") != payload:
            raise SystemExit("checked result differs from regenerated result")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
