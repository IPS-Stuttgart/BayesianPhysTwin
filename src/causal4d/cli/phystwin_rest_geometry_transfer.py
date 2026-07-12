"""CLI for one locked source-to-target PhysTwin rest-geometry rerun."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d.phystwin_rest_geometry_transfer import (
    evaluate_phystwin_rest_geometry_transfer_case,
)
from causal4d.real_protocol import load_protocol, validate_execution_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a source execution's canonical rest correction to one locked "
            "factual, same-grasp, or new-contact target and rerun Warp."
        )
    )
    parser.add_argument("official_repo")
    parser.add_argument("protocol")
    parser.add_argument("execution_manifest")
    parser.add_argument("plan_record")
    parser.add_argument("source_correction_manifest")
    parser.add_argument("canonical_material_graph")
    parser.add_argument("target_case_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--velocity-history-frames", type=int, default=3)
    parser.add_argument("--atomic-spring-forces", action="store_true")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    execution_manifest_path = Path(args.execution_manifest)
    execution_manifest = json.loads(
        execution_manifest_path.read_text(encoding="utf-8")
    )
    validate_execution_manifest(protocol, execution_manifest, verify_files=False)
    plan_record = json.loads(Path(args.plan_record).read_text(encoding="utf-8"))
    if plan_record["target_execution_id"] != execution_manifest["execution_id"]:
        parser.error("plan record and execution manifest target different executions")
    timing = execution_manifest["timing"]
    intervention_frame = int(timing["intervention_frame"])
    rollout_start = intervention_frame
    if plan_record["target_response_prefix_allowed"]:
        rollout_start += int(timing["o_plus_prefix_frames"])
    case_root = Path(args.target_case_dir)
    result = evaluate_phystwin_rest_geometry_transfer_case(
        args.official_repo,
        case_root / "final_data.pkl",
        case_root / "inference.pkl",
        case_root / "optimal_params.pkl",
        case_root / "checkpoint.pth",
        case_root / "gt_track_3d.pkl",
        args.source_correction_manifest,
        args.output_dir,
        plan_record=plan_record,
        target_execution_id=execution_manifest["execution_id"],
        rollout_start_frame=rollout_start,
        evaluation_start_frame=rollout_start,
        evaluation_stop_frame=int(timing["frame_count"]),
        velocity_history_frames=args.velocity_history_frames,
        deterministic_spring_forces=not args.atomic_spring_forces,
        expected_protocol_id=protocol["protocol_id"],
        expected_protocol_design_sha256=protocol["design_sha256"],
        canonical_material_graph_path=args.canonical_material_graph,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
