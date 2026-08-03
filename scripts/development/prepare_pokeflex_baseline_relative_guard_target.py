#!/usr/bin/env python3
"""Materialize the fresh PokeFlex guard protocol from frozen evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bayesian_phystwin.pokeflex_baseline_relative_guard_target import (  # noqa: E402
    CHECKPOINT_SHA256,
    DEVELOPMENT_COMMIT,
    DEVELOPMENT_EVALUATION_SHA256,
    MINIMUM_SUPPORTED_OBJECT_COUNT,
    MINIMUM_WIN_COUNT,
    PROTOCOL_ID,
    PROTOCOL_KIND,
    SELECTED_ARM,
    SELECTION_FILE_SHA256,
    SELECTION_MANIFEST_SHA256,
    SOURCE_PROTOCOL_SHA256,
    SOURCE_RESULT_SHA256,
    TARGET_OBJECTS,
    TARGET_TAKE_IDS,
    UPSTREAM_COMMIT,
    certificate_sha256,
    protocol_sha256,
    validate_protocol,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("development_evaluation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--locked-at-utc", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace protocol: {args.output}")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    development = json.loads(
        args.development_evaluation.read_text(encoding="utf-8")
    )
    if _sha256(args.selection) != SELECTION_FILE_SHA256:
        raise ValueError("selection bytes changed")
    if selection["selection_manifest_sha256"] != SELECTION_MANIFEST_SHA256:
        raise ValueError("selection checksum changed")
    if tuple(selection["selected_take_ids"]) != TARGET_TAKE_IDS:
        raise ValueError("selected take cohort changed")
    if _sha256(args.development_evaluation) != DEVELOPMENT_EVALUATION_SHA256:
        raise ValueError("development evaluation changed")
    certificate = development["deployment_fit"]["certificate"]
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": PROTOCOL_KIND,
        "protocol_id": PROTOCOL_ID,
        "locked_at_utc": args.locked_at_utc,
        "status": "pre-outcome lock for twelve all-history-fresh public PokeFlex takes",
        "claim_boundary": (
            "A pass supports strict improvement of the released PokeFlex Kinect "
            "checkpoint on this registered fresh-take cohort. It is not the "
            "paper's unavailable internal split; 6.498 mm is context only."
        ),
        "source_gate": {
            "protocol_sha256": SOURCE_PROTOCOL_SHA256,
            "result_sha256": SOURCE_RESULT_SHA256,
            "passed": True,
            "selected_arm": SELECTED_ARM,
        },
        "selection_audit": {
            "manifest_path": str(
                args.selection.resolve().relative_to(ROOT)
            ).replace("\\", "/"),
            "manifest_sha256": selection["selection_manifest_sha256"],
            "manifest_file_sha256": _sha256(args.selection),
            "repository_head": selection["repository_head"],
            "git_ref_count": selection["git_ref_count"],
            "git_ref_digest": selection["git_ref_digest"],
            "git_ref_tip_count": selection["git_ref_tip_count"],
            "git_ref_tip_digest": selection["git_ref_tip_digest"],
            "reachable_blob_count": selection["reachable_blob_count"],
            "reachable_blob_digest": selection["reachable_blob_digest"],
            "public_take_count": selection["public_take_count"],
            "public_take_digest": selection["public_take_digest"],
            "referenced_take_count": selection["referenced_take_count"],
            "referenced_take_digest": selection["referenced_take_digest"],
            "eligible_object_count": selection["eligible_object_count"],
            "selected_take_digest": selection["selected_take_digest"],
            "selection_was_target_free": True,
        },
        "development_guard": {
            "evaluation_path": str(
                args.development_evaluation.resolve().relative_to(ROOT)
            ).replace("\\", "/"),
            "evaluation_sha256": _sha256(args.development_evaluation),
            "git_commit": DEVELOPMENT_COMMIT,
            "claim_status": development["claim_status"],
            "feature_names": development["feature_names"],
            "certificate": certificate,
            "certificate_sha256": certificate_sha256(certificate),
        },
        "target_cohort": {
            "take_ids": list(TARGET_TAKE_IDS),
            "objects": list(TARGET_OBJECTS),
            "replacement_allowed": False,
            "freshness_scope": selection["scan_scope"],
            "selection_basis": selection["selection_rule"],
        },
        "method": {
            "selected_arm": SELECTED_ARM,
            "physical_prior": "released PokeFlex Kinect checkpoint",
            "state_update": "robust correlation-aware graph registration at f-1",
            "field": "action_local_state_relative_0.4",
            "scale": 0.125,
            "guard_features": list(development["feature_names"]),
            "guard_rule": (
                "admit iff in source support and calibrated upper regret is below zero"
            ),
            "guard_inputs": "target-free quantities through f-1 plus physical prior f",
            "target_frame_observation": "forbidden",
            "missing_required_robot_pose_action": (
                "mark update unsupported and return byte-identical released checkpoint"
            ),
            "fallback": "byte-identical released checkpoint",
        },
        "upstream": {
            "repository": "https://github.com/pokeflex-dataset/reconstruction",
            "code_commit": UPSTREAM_COMMIT,
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "custody": {
            "prediction_and_scoring_are_separate": True,
            "required_prediction_seal_count": len(TARGET_TAKE_IDS),
            "all_prediction_revisions_must_match": True,
            "implementation_checkout_must_be_clean": True,
            "target_mesh_access_before_barrier": "forbidden",
            "prediction_observation_history": "f-5 through f-1",
            "prediction_robot_history": "through f-1",
            "template_mesh": "allowed explicit upstream task input only",
            "scored_frames": "force-y at f exceeds 3 N",
            "retry_policy": "no replacement; preserve failures and exact fallbacks",
        },
        "evaluation": {
            "primary_metric": "CD_UL1_mm",
            "primary_definition": (
                "mean nearest-neighbor L1 distance from deterministic predicted "
                "surface samples to target surface samples in metric coordinates"
            ),
            "surface_sample_count": 10000,
            "surface_sample_seed": 20260720,
            "aggregation": (
                "equal scored frames within each take, then equal physical objects"
            ),
            "jaccard_definition": (
                "trimesh boolean intersection volume divided by union volume"
            ),
            "jaccard_boolean_backend": "manifold",
            "jaccard_mesh_processing": "trimesh_default",
            "jaccard_role": "non-gating diagnostic",
        },
        "gates": {
            "direct_metric_reference": {
                "candidate_CD_UL1_mm_below": 6.498,
                "gating": False,
                "scope": "cross-split published context only",
            },
            "paired_transfer": {
                "relative_CD_UL1_improvement_above": 0.0,
                "bootstrap_upper_difference_mm_below": 0.0,
                "maximum_per_object_relative_regression": 0.0,
                "minimum_object_win_fraction": 0.8,
                "minimum_object_win_count": MINIMUM_WIN_COUNT,
                "minimum_supported_object_fraction": 0.8,
                "minimum_supported_object_count": MINIMUM_SUPPORTED_OBJECT_COUNT,
                "bootstrap_unit": "physical object",
                "bootstrap_replicates": 20000,
                "bootstrap_seed": 20260720,
                "bootstrap_upper_quantile": 0.975,
            },
        },
        "forbidden": [
            "opening any selected target mesh before all prediction seals pass",
            "using target frame f observations to predict frame f",
            "changing the certificate, cohort, candidate, metric, or gates after lock",
            "replacing a failed selected take",
            "tuning from any selected-take outcome",
            "claiming identity with the unavailable published internal split",
            "touching any held-v8 artifact or process",
        ],
    }
    payload["protocol_sha256"] = protocol_sha256(payload)
    validate_protocol(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(json.dumps({"protocol_sha256": payload["protocol_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
