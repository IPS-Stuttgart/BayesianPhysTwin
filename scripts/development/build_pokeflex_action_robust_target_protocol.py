#!/usr/bin/env python3
"""Build the third PokeFlex repeated-action scale target protocol."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPOSITORY_ROOT = _repository_root()
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from bayesian_phystwin.pokeflex_action_robust_freshness import (  # noqa: E402
    validate_action_robust_freshness_audit,
)
from bayesian_phystwin.pokeflex_action_robust_scale import (  # noqa: E402
    validate_action_robust_scale_calibration,
)
from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (  # noqa: E402
    TARGET_PROTOCOL_ACTION_ROBUST_FRESH6_V3,
    file_sha256,
    target_protocol_sha256,
    validate_pokeflex_shrinkage_target_protocol,
)


def build_protocol(
    base: dict[str, object],
    freshness: dict[str, object],
    calibration: dict[str, object],
    *,
    freshness_path: Path,
    calibration_path: Path,
    locked_at_utc: str,
) -> dict[str, object]:
    freshness_validation = validate_action_robust_freshness_audit(freshness)
    calibration_validation = validate_action_robust_scale_calibration(calibration)
    protocol = copy.deepcopy(base)
    protocol.pop("protocol_sha256", None)
    protocol.pop("preoutcome_storage_amendment", None)
    protocol["protocol_id"] = TARGET_PROTOCOL_ACTION_ROBUST_FRESH6_V3
    protocol["locked_at_utc"] = locked_at_utc
    protocol["status"] = (
        "pre-outcome lock for repeated-action maximin shrinkage on six third-panel "
        "public takes"
    )
    protocol["claim_boundary"] = (
        "This protocol tests whether a correction magnitude that improves two "
        "opened interactions transfers to a third untouched interaction of the "
        "same physical object. It compares the frozen maximin scale with both "
        "the released checkpoint and the globally validated 0.125 correction. "
        "The six-object cohort differs from the published validation split, so "
        "the published 6.498 mm value is contextual rather than a table-SOTA gate."
    )

    source_gate = protocol["source_gate"]
    source_gate.pop("instance_scale_calibration", None)
    source_gate["action_robust_scale_calibration"] = {
        "calibration_sha256": calibration["calibration_sha256"],
        "calibration_file_sha256": file_sha256(calibration_path),
        "source_object_count": calibration["source_gate"]["source_object_count"],
        "source_action_count": calibration["source_gate"]["source_action_count"],
        "adjusted_object_count": calibration["source_gate"][
            "adjusted_object_count"
        ],
        "controls_passed": calibration["source_gate"]["controls_passed"],
        "future_take_outcomes_opened": False,
    }
    target_take_ids = list(freshness_validation["target_take_ids"])
    protocol["target_cohort"] = {
        "take_ids": target_take_ids,
        "prospective_take_ids": target_take_ids,
        "development_overlap_take_ids": [],
        "replacement_allowed": False,
        "selection_basis": (
            "minimum SHA-256 of the frozen v3 salt, NUL, and take ID within each "
            "remaining eligible object after excluding both fresh12 campaigns"
        ),
        "freshness_scope": (
            "exact take IDs are unexposed outside the registered inventory audit; "
            "physical object instances and two source interactions are opened"
        ),
    }
    protocol["freshness_audit"] = {
        "path": freshness_path.as_posix(),
        "audit_sha256": freshness["audit_sha256"],
        "audit_file_sha256": file_sha256(freshness_path),
        "public_inventory_sha256": freshness["public_archive"][
            "sorted_newline_inventory_sha256"
        ],
        "prior_exclusion_union_sha256": freshness["prior_exposure_audit"][
            "sorted_newline_union_sha256"
        ],
        "eligible_inventory_sha256": freshness["eligibility"][
            "sorted_newline_inventory_sha256"
        ],
        "selected_inventory_sha256": freshness["selection"][
            "sorted_newline_inventory_sha256"
        ],
        "selection_salt": freshness["selection"]["salt"],
        "selected_zip_sha256": freshness["selection"]["zip_sha256"],
    }
    multipliers = calibration_validation["multipliers"]
    previous_method = protocol["method"]
    protocol["method"] = {
        "selected_arm": previous_method["selected_arm"],
        "physical_prior": previous_method["physical_prior"],
        "state_update": previous_method["state_update"],
        "field": previous_method["field"],
        "base_scale": calibration["base_effective_scale"],
        "scale_policy": (
            "two-action maximin per physical object with frozen capped bank and "
            "global fallback"
        ),
        "action_robust_scale_calibration": {
            "path": calibration_path.as_posix(),
            "calibration_sha256": calibration["calibration_sha256"],
            "calibration_file_sha256": file_sha256(calibration_path),
            "multipliers": multipliers,
        },
        "effective_scale_by_object": {
            name: float(calibration["base_effective_scale"]) * multiplier
            for name, multiplier in multipliers.items()
        },
        "transfer": previous_method["transfer"],
        "required_action_history": previous_method["required_action_history"],
        "missing_required_robot_pose_action": previous_method[
            "missing_required_robot_pose_action"
        ],
        "unsupported_frame_action": previous_method["unsupported_frame_action"],
        "online_selector": (
            "none; one two-action maximin multiplier is fixed per object pre-target"
        ),
        "global_conflict_fallback": (
            "multiplier one whenever no non-default multiplier improves both source "
            "actions"
        ),
        "target_outcome_adaptation": "forbidden",
    }
    protocol["custody"]["required_prediction_seal_count"] = len(target_take_ids)
    protocol["evaluation"]["aggregation"] = (
        "equal-weight physical objects over six prospectively selected public "
        "takes; frame-level values are diagnostic"
    )
    protocol["official_reference"]["fresh12_scope"] = (
        "six third-panel public recordings; only paired checkpoint and "
        "global-scale comparisons are authorized"
    )
    protocol["gates"]["direct_metric_reference"]["scope"] = (
        "paired action-robust candidate versus released checkpoint over all six "
        "prospectively selected takes"
    )
    paired = protocol["gates"]["paired_transfer"]
    paired["bootstrap_unit"] = "physical object among all six third-panel takes"
    protocol["gates"].pop("instance_advancement", None)
    protocol["gates"]["action_robust_advancement"] = {
        "reference": "frozen global correction scale 0.125 on the same predictions",
        "relative_CD_UL1_improvement_above": 0.0,
        "bootstrap_upper_difference_mm_below": 0.0,
        "maximum_per_object_relative_regression": 0.0,
        "bootstrap_unit": "physical object among all six third-panel takes",
        "bootstrap_replicates": paired["bootstrap_replicates"],
        "bootstrap_seed": paired["bootstrap_seed"],
        "bootstrap_upper_quantile": paired["bootstrap_upper_quantile"],
    }
    protocol["forbidden"] = [
        "opening any scored deformed mesh before all six prediction seals pass the barrier",
        "using frame f Kinect or robot observations to predict frame f",
        "substituting T_WT for missing T_WE or otherwise imputing a required actuator pose",
        "changing the maximin multiplier map, support radius, metric, aggregation, or gate after target access",
        "replacing any selected take or a technically failed prediction",
        "tuning from any of the six third-panel target outcomes",
        "representing the physical objects as unseen during source development",
        "substituting a repaired-mesh or voxel Jaccard for the registered public-mesh diagnostic",
        "claiming reproduction of or direct comparison with the published eighteen-object aggregate",
        "touching any held-v8 artifact or process",
    ]
    protocol["protocol_sha256"] = target_protocol_sha256(protocol)
    validate_pokeflex_shrinkage_target_protocol(
        protocol,
        bind_action_robust_digest=False,
    )
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_protocol", type=Path)
    parser.add_argument("freshness_audit", type=Path)
    parser.add_argument("scale_calibration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--locked-at-utc", required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to replace target protocol: {args.output}")
    base = json.loads(args.base_protocol.read_text(encoding="utf-8"))
    freshness = json.loads(args.freshness_audit.read_text(encoding="utf-8"))
    calibration = json.loads(args.scale_calibration.read_text(encoding="utf-8"))
    protocol = build_protocol(
        base,
        freshness,
        calibration,
        freshness_path=args.freshness_audit,
        calibration_path=args.scale_calibration,
        locked_at_utc=args.locked_at_utc,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "protocol_sha256": protocol["protocol_sha256"],
                "target_take_ids": protocol["target_cohort"]["take_ids"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
