#!/usr/bin/env python3
"""Build compact provenance for the frozen final-two PokeFlex result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _repository_root()
sys.path.insert(0, str(ROOT / "src"))

from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (  # noqa: E402
    canonical_payload_sha256,
    file_sha256,
    load_pokeflex_shrinkage_target_protocol,
)

RESULT_KIND = "PokeFlexActionRobustFresh2ResultSummary"


def _row_summary(row: dict[str, object]) -> dict[str, object]:
    baseline = float(row["baseline_mean_CD_UL1_mm"])
    global_candidate = float(row["global_candidate_mean_CD_UL1_mm"])
    candidate = float(row["candidate_mean_CD_UL1_mm"])
    return {
        "take_id": row["take_id"],
        "object_name": row["object_name"],
        "scored_frame_count": row["scored_frame_count"],
        "supported_frame_count": row["supported_frame_count"],
        "checkpoint_CD_UL1_mm": baseline,
        "global_scale_CD_UL1_mm": global_candidate,
        "action_robust_CD_UL1_mm": candidate,
        "action_robust_vs_checkpoint_relative_improvement": (
            baseline - candidate
        )
        / baseline,
        "action_robust_vs_global_relative_improvement": (
            global_candidate - candidate
        )
        / global_candidate,
    }


def build_summary(result_root: Path, protocol_path: Path) -> dict[str, object]:
    """Bind all compact evidence and preserve the preregistered gate decision."""

    protocol = load_pokeflex_shrinkage_target_protocol(protocol_path)
    target_path = result_root / "target_result.json"
    barrier_path = result_root / "prediction_barrier.json"
    target = json.loads(target_path.read_text(encoding="utf-8"))
    barrier = json.loads(barrier_path.read_text(encoding="utf-8"))
    if target["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError("target result protocol changed")
    if target["barrier_sha256"] != barrier["barrier_sha256"]:
        raise ValueError("target result barrier changed")
    if target["target_meshes_opened_after_complete_barrier"] is not True:
        raise ValueError("target outcome preceded the complete barrier")
    if target["aggregate"]["all_target_gates_passed"] is not False:
        raise ValueError("registered final-two decision unexpectedly passed")

    file_paths = sorted(
        path
        for path in result_root.rglob("*")
        if path.is_file() and path.name != "summary.json"
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": RESULT_KIND,
        "protocol_sha256": protocol["protocol_sha256"],
        "implementation_revision": barrier["implementation_revision"],
        "prediction_count": barrier["prediction_count"],
        "barrier_sha256": barrier["barrier_sha256"],
        "barrier_file_sha256": file_sha256(barrier_path),
        "target_result_file_sha256": file_sha256(target_path),
        "target_meshes_opened_after_complete_barrier": True,
        "future_mesh_read_before_barrier": False,
        "server_transfer": {
            "source_host": "gpuserver6000",
            "source_ipv4": "129.69.102.145",
            "destination_host": "gpuserver4090",
            "destination_ipv4": "129.69.102.139",
            "payload_path": "direct server LAN HTTP",
            "jump_server_in_payload_path": False,
            "destination_archives_rehashed": True,
        },
        "verification": {
            "remote_pokeflex_tests_passed": 211,
            "remote_pokeflex_tests_skipped": 3,
            "changed_file_ruff_passed": True,
        },
        "files": {
            path.relative_to(result_root).as_posix(): file_sha256(path)
            for path in file_paths
        },
        "takes": [_row_summary(row) for row in target["objects"]],
        "aggregate": target["aggregate"],
        "decision": {
            "beats_released_checkpoint": target["aggregate"]["checkpoint_pairing"][
                "passed"
            ],
            "beats_global_scale": target["aggregate"]["global_scale_advancement"][
                "passed"
            ],
            "all_preregistered_gates_passed": target["aggregate"][
                "all_target_gates_passed"
            ],
            "advance_as_strictly_superior_scale_rule": False,
            "retuning_from_final_two_outcomes_authorized": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    payload["summary_sha256"] = canonical_payload_sha256(
        payload,
        digest_field="summary_sha256",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            ROOT
            / "configs"
            / "sota"
            / "pokeflex_action_robust_shrinkage_fresh2_v5.json"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.result_root / "summary.json"
    payload = build_summary(args.result_root, args.protocol)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
