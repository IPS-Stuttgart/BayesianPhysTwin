"""Generate the deterministic decision-capability-atlas mechanism study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.decision_capability_atlas_v1 import (
    DECISION_CAPABILITY_ATLAS_CLAIM_BOUNDARY,
    affine_capability_halfspaces,
    affine_decision_capability_atlas,
    capability_polygon_2d,
    polygon_area_2d,
)
from bayesian_phystwin.decision_capability_task_uncertainty_v1 import (
    TASK_UNCERTAINTY_CERTIFICATE_CLAIM_BOUNDARY,
    box_robust_center_halfspaces,
    box_task_set_capability,
    norm_ball_capability_margin,
    task_uncertainty_action_mask,
)

ACTIONS = ("pull_left", "hold", "pull_right")
TASK_BOUNDS = np.array([[-1.5, 1.5], [0.0, 4.0]], dtype=np.float64)
ROBUST_TASK_HALF_WIDTH = np.array([0.1, 0.2], dtype=np.float64)


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
        displacement,
        physical_risk,
        np.square(displacement),
        np.stack((-2.0 * displacement, physical_risk), axis=2),
    )


def point_choice(
    prior: np.ndarray,
    intercept: np.ndarray,
    coefficient: np.ndarray,
    task: np.ndarray,
) -> int:
    loss = intercept + np.tensordot(coefficient, task, axes=(2, 0))
    return int(np.argmin(prior @ loss))


def generate_result() -> dict[str, Any]:
    (
        prior,
        quotient,
        classes,
        displacement,
        physical_risk,
        intercept,
        coefficient,
    ) = controlled_family()
    halfspaces_by_action = []
    polygons = []
    action_areas = []
    action_halfspace_counts = []
    for action_index, action in enumerate(ACTIONS):
        halfspaces = affine_capability_halfspaces(
            prior,
            quotient,
            classes,
            intercept,
            coefficient,
            action_index=action_index,
            regret_tolerance=0.0,
        )
        halfspaces_by_action.append(halfspaces)
        polygon = capability_polygon_2d(halfspaces, TASK_BOUNDS)
        area = polygon_area_2d(polygon)
        polygons.append(
            {
                "action": action,
                "vertices": polygon.tolist(),
                "area": area,
                "area_fraction": area / 12.0,
            }
        )
        action_areas.append(area)
        action_halfspace_counts.append(halfspaces.halfspace_count)

    box_area = float(np.prod(TASK_BOUNDS[:, 1] - TASK_BOUNDS[:, 0]))
    certified_area = float(np.sum(action_areas))
    fallback_area = box_area - certified_area
    probes = np.array(
        [
            [-1.2, 0.2],
            [0.0, 2.0],
            [1.2, 0.2],
            [-0.45, 0.0],
        ],
        dtype=np.float64,
    )
    probe_atlas = affine_decision_capability_atlas(
        prior,
        quotient,
        classes,
        intercept,
        coefficient,
        probes,
        regret_tolerance=0.0,
    )
    probe_records = []
    for index, task in enumerate(probes):
        robust = np.flatnonzero(probe_atlas.robustly_optimal_action_mask[index])
        probe_records.append(
            {
                "task": task.tolist(),
                "worst_case_regret": probe_atlas.worst_case_regret[index].tolist(),
                "robust_action": ACTIONS[int(robust[0])] if robust.size == 1 else None,
                "point_belief_action": ACTIONS[
                    point_choice(prior, intercept, coefficient, task)
                ],
            }
        )

    target = np.linspace(TASK_BOUNDS[0, 0], TASK_BOUNDS[0, 1], 121)
    risk = np.linspace(TASK_BOUNDS[1, 0], TASK_BOUNDS[1, 1], 161)
    grid = np.asarray([(x, y) for y in risk for x in target], dtype=np.float64)
    grid_atlas = affine_decision_capability_atlas(
        prior,
        quotient,
        classes,
        intercept,
        coefficient,
        grid,
        regret_tolerance=0.0,
    )
    grid_exact_count = int(np.count_nonzero(grid_atlas.exact_capability_mask))

    uncertain_centers = np.array(
        [
            [-1.2, 0.2],
            [0.0, 2.0],
            [1.2, 0.2],
            [-0.6, 0.1],
        ],
        dtype=np.float64,
    )
    uncertain_half_widths = np.array(
        [
            [0.2, 0.2],
            [0.2, 0.4],
            [0.2, 0.2],
            [0.04, 0.05],
        ],
        dtype=np.float64,
    )
    uncertain_reports = [
        box_task_set_capability(
            halfspaces,
            uncertain_centers,
            uncertain_half_widths,
        )
        for halfspaces in halfspaces_by_action
    ]
    uncertain_action_mask = task_uncertainty_action_mask(uncertain_reports)
    nominal_uncertain_atlas = affine_decision_capability_atlas(
        prior,
        quotient,
        classes,
        intercept,
        coefficient,
        uncertain_centers,
        regret_tolerance=0.0,
    )
    uncertain_records = []
    for index, (center, half_width) in enumerate(
        zip(uncertain_centers, uncertain_half_widths, strict=True)
    ):
        nominal_actions = np.flatnonzero(
            nominal_uncertain_atlas.robustly_optimal_action_mask[index]
        )
        robust_actions = np.flatnonzero(uncertain_action_mask[index])
        radii = [
            float(
                norm_ball_capability_margin(
                    halfspaces,
                    center[None, :],
                    task_norm="l2",
                ).normalized_constraint_margin[0]
            )
            for halfspaces in halfspaces_by_action
        ]
        uncertain_records.append(
            {
                "center": center.tolist(),
                "half_width": half_width.tolist(),
                "nominal_actions": [ACTIONS[int(value)] for value in nominal_actions],
                "box_robust_actions": [ACTIONS[int(value)] for value in robust_actions],
                "l2_constraint_margin_by_action": dict(
                    zip(ACTIONS, radii, strict=True)
                ),
            }
        )

    center_bounds = TASK_BOUNDS.copy()
    center_bounds[:, 0] += ROBUST_TASK_HALF_WIDTH
    center_bounds[:, 1] -= ROBUST_TASK_HALF_WIDTH
    center_domain_area = float(np.prod(center_bounds[:, 1] - center_bounds[:, 0]))
    nominal_center_areas = []
    robust_center_areas = []
    robust_center_polygons = []
    for action, halfspaces in zip(ACTIONS, halfspaces_by_action, strict=True):
        nominal_polygon = capability_polygon_2d(halfspaces, center_bounds)
        robust_halfspaces = box_robust_center_halfspaces(
            halfspaces,
            ROBUST_TASK_HALF_WIDTH,
        )
        robust_polygon = capability_polygon_2d(
            robust_halfspaces,
            center_bounds,
        )
        nominal_center_areas.append(polygon_area_2d(nominal_polygon))
        robust_area = polygon_area_2d(robust_polygon)
        robust_center_areas.append(robust_area)
        robust_center_polygons.append(
            {
                "action": action,
                "vertices": robust_polygon.tolist(),
                "area": robust_area,
                "area_fraction": robust_area / center_domain_area,
            }
        )
    nominal_center_union = float(np.sum(nominal_center_areas))
    robust_center_union = float(np.sum(robust_center_areas))

    result: dict[str, Any] = {
        "schema": "bayesian-phystwin/decision-capability-atlas-study-v2",
        "schema_version": 2,
        "status": "complete",
        "task_family": {
            "parameters": ["target_displacement", "physical_risk_weight"],
            "bounds": TASK_BOUNDS.tolist(),
            "loss": (
                "(predicted_displacement-target_displacement)^2 + "
                "physical_risk_weight*physical_risk"
            ),
            "affine_reduction": (
                "The action-independent target_displacement^2 term is dropped."
            ),
        },
        "hypotheses": {
            "prior_weights": prior.tolist(),
            "quotient_weights": quotient.tolist(),
            "class_index": classes.tolist(),
            "predicted_displacement": displacement.tolist(),
            "physical_risk": physical_risk.tolist(),
        },
        "actions": list(ACTIONS),
        "regret_tolerance": 0.0,
        "capability_regions": polygons,
        "summary": {
            "domain_area": box_area,
            "certified_union_area": certified_area,
            "certified_union_fraction": certified_area / box_area,
            "fallback_area": fallback_area,
            "fallback_fraction": fallback_area / box_area,
            "action_area_fraction": {
                action: area / box_area
                for action, area in zip(ACTIONS, action_areas, strict=True)
            },
            "halfspace_count_per_action": dict(
                zip(ACTIONS, action_halfspace_counts, strict=True)
            ),
            "point_policy_decides_outside_exact_capability_fraction": (
                fallback_area / box_area
            ),
            "grid_task_count": int(grid.shape[0]),
            "grid_exact_capability_count": grid_exact_count,
        },
        "representative_tasks": probe_records,
        "task_uncertainty": {
            "semantics": (
                "Each box contains all task objectives whose coordinates differ "
                "from the center by at most the declared half-width."
            ),
            "claim_boundary": TASK_UNCERTAINTY_CERTIFICATE_CLAIM_BOUNDARY,
            "representative_task_sets": uncertain_records,
            "fixed_box_atlas": {
                "half_width": ROBUST_TASK_HALF_WIDTH.tolist(),
                "center_bounds": center_bounds.tolist(),
                "center_domain_area": center_domain_area,
                "nominal_certified_union_area": nominal_center_union,
                "nominal_certified_union_fraction": (
                    nominal_center_union / center_domain_area
                ),
                "robust_certified_union_area": robust_center_union,
                "robust_certified_union_fraction": (
                    robust_center_union / center_domain_area
                ),
                "robust_fallback_area": center_domain_area - robust_center_union,
                "robust_fallback_fraction": (
                    1.0 - robust_center_union / center_domain_area
                ),
                "capability_regions": robust_center_polygons,
            },
        },
        "checks": {
            "all_three_actions_have_nonempty_regions": all(
                area > 0.0 for area in action_areas
            ),
            "fallback_region_nonempty": fallback_area > 0.0,
            "left_right_symmetry": abs(action_areas[0] - action_areas[2]) < 1e-12,
            "point_choice_fills_fallback_example": (
                probe_records[-1]["robust_action"] is None
                and probe_records[-1]["point_belief_action"] == "pull_left"
            ),
            "uncertain_representatives_identify_left_hold_right": (
                uncertain_records[0]["box_robust_actions"] == ["pull_left"]
                and uncertain_records[1]["box_robust_actions"] == ["hold"]
                and uncertain_records[2]["box_robust_actions"] == ["pull_right"]
            ),
            "nominally_capable_task_can_fail_under_objective_uncertainty": (
                uncertain_records[3]["nominal_actions"] == ["pull_left"]
                and uncertain_records[3]["box_robust_actions"] == []
            ),
            "objective_uncertainty_strictly_shrinks_capability": (
                robust_center_union < nominal_center_union
            ),
            "robust_left_right_symmetry": (
                abs(robust_center_areas[0] - robust_center_areas[2]) < 1e-12
            ),
        },
        "claim_boundary": DECISION_CAPABILITY_ATLAS_CLAIM_BOUNDARY,
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
        raise SystemExit("capability-atlas checks failed")
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.check is not None:
        if args.check.read_text(encoding="utf-8") != payload:
            raise SystemExit("checked result differs from regenerated result")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
