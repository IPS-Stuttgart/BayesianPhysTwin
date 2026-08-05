#!/usr/bin/env python3
"""Build the frozen final-two PokeFlex action-transfer protocol."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _repository_root()
sys.path.insert(0, str(ROOT / "src"))

from bayesian_phystwin.pokeflex_action_robust_all18 import (  # noqa: E402
    validate_all18_calibration,
)
from bayesian_phystwin.pokeflex_action_robust_final_freshness import (  # noqa: E402
    validate_final_freshness_audit,
)
from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (  # noqa: E402
    TARGET_PROTOCOL_ACTION_ROBUST_FRESH2_V5,
    target_protocol_sha256,
    validate_pokeflex_shrinkage_target_protocol,
)


def build_protocol(
    parent: dict[str, object],
    freshness: dict[str, object],
    all18: dict[str, object],
    *,
    locked_at_utc: str,
) -> dict[str, object]:
    """Specialize the already frozen v3 method to its final two public takes."""

    freshness_validation = validate_final_freshness_audit(freshness)
    all18_validation = validate_all18_calibration(all18)
    target_take_ids = list(freshness_validation["target_take_ids"])
    if target_take_ids != ["Pillow_T4", "PlushDice_T3"]:
        raise ValueError("final target complement changed")
    parent_multipliers = parent["method"]["action_robust_scale_calibration"][
        "multipliers"
    ]
    for object_name in ("Pillow", "PlushDice"):
        if all18_validation["multipliers"][object_name] != parent_multipliers[object_name]:
            raise ValueError("all-18 extension changed a target multiplier")

    payload = deepcopy(parent)
    payload["protocol_id"] = TARGET_PROTOCOL_ACTION_ROBUST_FRESH2_V5
    payload["locked_at_utc"] = locked_at_utc
    payload["claim_boundary"] = (
        "This protocol evaluates the frozen action-robust correction on the final "
        "two previously unscored takes in the 116-take public PokeFlex release. "
        "Both physical objects were studied earlier, so this is prospective "
        "new-action transfer, not unseen-object generalization or a reconstruction "
        "of the five unavailable records in the published validation split."
    )
    payload["target_cohort"] = {
        "take_ids": target_take_ids,
        "prospective_take_ids": target_take_ids,
        "development_overlap_take_ids": [],
        "replacement_allowed": False,
        "selection_basis": (
            "the exhaustive two-take complement after excluding all 114 public "
            "takes exposed by or before the v3 campaign"
        ),
        "freshness_scope": (
            "exact take outcomes and predictions were unexposed before this lock; "
            "the two physical object identities and earlier actions were opened"
        ),
    }
    payload["freshness_audit"] = {
        "path": (
            "configs/sota/"
            "pokeflex_action_robust_fresh2_exclusion_audit_v5.json"
        ),
        "audit_sha256": freshness["audit_sha256"],
        "audit_file_sha256": (
            "32e3b95c4449bf33e06aeefff6f25581375b8ca6ee0cf1276c41efaba0fce98b"
        ),
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
        "selected_zip_sha256": freshness["selection"]["zip_sha256"],
    }
    payload["source_gate"]["all18_source_extension"] = {
        "path": "configs/sota/pokeflex_action_robust_scale_all18_v4.json",
        "calibration_sha256": all18["calibration_sha256"],
        "calibration_file_sha256": (
            "00cdf5732f5dbf7eb0f899ebbb536260d9e66c0a151b41eec81ffaaef4aaf110"
        ),
        "source_protocol_sha256": all18["source_protocol_sha256"],
        "source_gate_passed": all18["source_gate"]["passed"],
        "parent_rows_used_by_target_unchanged": True,
        "target_objects": ["Pillow", "PlushDice"],
        "operational_target_calibration": False,
        "note": (
            "The target runner remains bound to the earlier v3 calibration. The "
            "all-18 extension is source-only corroboration and preserves both rows."
        ),
    }
    payload["custody"]["required_prediction_seal_count"] = 2
    payload["custody"]["retry_policy"] = (
        "no replacement; technical failures and exact fallbacks remain in the "
        "registered two-take cohort"
    )
    payload["evaluation"]["aggregation"] = (
        "equal-weight physical objects over the final two prospectively sealed "
        "public takes; frame-level values are diagnostic"
    )
    for gate_name in ("paired_transfer", "action_robust_advancement"):
        payload["gates"][gate_name]["bootstrap_unit"] = (
            "physical object among the final two public takes"
        )
    payload["gates"]["direct_metric_reference"]["scope"] = (
        "paired action-robust candidate versus released checkpoint on both "
        "prospectively sealed final public takes"
    )
    payload["official_reference"]["fresh12_scope"] = (
        "the final two public recordings; only paired checkpoint and global-scale "
        "comparisons are authorized"
    )
    payload["forbidden"] = [
        "opening a scored deformed mesh before both prediction seals pass the barrier",
        "using frame f Kinect or robot observations to predict frame f",
        "substituting T_WT for missing T_WE or imputing a required actuator pose",
        "changing the multiplier map, metric, aggregation, or gate after target access",
        "replacing either selected take or a technically failed prediction",
        "tuning from either final-two outcome",
        "representing the physical objects as unseen during source development",
        "claiming reproduction of the unavailable published eighteen-take split",
        "combining the final-two prospective result with retrospective actions without labeling the evidence boundary",
        "touching any held-v8 artifact or process",
    ]
    payload["protocol_sha256"] = target_protocol_sha256(payload)
    validate_pokeflex_shrinkage_target_protocol(
        payload,
        bind_action_robust_digest=False,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent",
        type=Path,
        default=(
            ROOT
            / "configs"
            / "sota"
            / "pokeflex_action_robust_shrinkage_fresh6_v3.json"
        ),
    )
    parser.add_argument(
        "--freshness",
        type=Path,
        default=(
            ROOT
            / "configs"
            / "sota"
            / "pokeflex_action_robust_fresh2_exclusion_audit_v5.json"
        ),
    )
    parser.add_argument(
        "--all18-calibration",
        type=Path,
        default=(
            ROOT
            / "configs"
            / "sota"
            / "pokeflex_action_robust_scale_all18_v4.json"
        ),
    )
    parser.add_argument("--locked-at-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = build_protocol(
        json.loads(args.parent.read_text(encoding="utf-8")),
        json.loads(args.freshness.read_text(encoding="utf-8")),
        json.loads(args.all18_calibration.read_text(encoding="utf-8")),
        locked_at_utc=args.locked_at_utc,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
