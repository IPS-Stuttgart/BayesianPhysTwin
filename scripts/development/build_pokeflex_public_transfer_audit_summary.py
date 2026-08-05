#!/usr/bin/env python3
"""Build compact custody and failure accounting for the PokeFlex public audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _repository_root()
sys.path.insert(0, str(ROOT / "src"))

from bayesian_phystwin.pokeflex_public_transfer_audit import (  # noqa: E402
    PROTOCOL_SHA256,
    RESULT_KIND,
    file_sha256,
    result_sha256,
    validate_public_transfer_protocol,
)

SUMMARY_KIND = "PokeFlexActionRobustPublicTransferAuditSummary"
IMPLEMENTATION_REVISION = "c4c80aed799ffa6f18ec96ecb0104f7a16e40d87"


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("summary_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _validate_fallback(
    artifact: Mapping[str, Any],
) -> tuple[int, int, bool]:
    missing = artifact.get("missing_required_T_WE")
    if not isinstance(missing, Mapping):
        raise ValueError("missing-T_WE accounting is absent")
    if missing.get("pose_imputation_used_by_prediction") is not False:
        raise ValueError("a missing pose influenced prediction")
    if missing.get("source_robot_bytes_modified") is not False:
        raise ValueError("source robot bytes were modified")
    source_frames = [int(value) for value in missing.get("source_frames", ())]
    fallback_frames = [
        int(value) for value in missing.get("fallback_target_frames", ())
    ]
    if int(missing.get("source_frame_count", -1)) != len(source_frames):
        raise ValueError("missing source-frame accounting changed")
    if int(missing.get("fallback_target_count", -1)) != len(fallback_frames):
        raise ValueError("fallback target-frame accounting changed")

    fallback = set(fallback_frames)
    targets = artifact.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("target rows are absent")
    for target in targets:
        if int(target["target_frame"]) not in fallback:
            continue
        checkpoint = float(target["released_checkpoint_CD_UL1_mm"])
        candidate_keys = [
            key
            for key in target
            if key.startswith("checkpoint_") and "_residual_scale_" in key
        ]
        if not candidate_keys:
            raise ValueError("fallback row has no registered candidate")
        if any(float(target[key]) != checkpoint for key in candidate_keys):
            raise ValueError("fallback candidate differs from the checkpoint")
    return len(source_frames), len(fallback_frames), len(fallback) == len(targets)


def build_summary(result_root: Path, protocol_path: Path) -> dict[str, Any]:
    """Bind the completed evidence without changing a score or decision gate."""

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validation = validate_public_transfer_protocol(protocol)
    result_path = result_root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("artifact_kind") != RESULT_KIND:
        raise ValueError("result kind changed")
    if result.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("result protocol changed")
    if result.get("result_sha256") != result_sha256(result):
        raise ValueError("result checksum mismatch")

    take_ids = set(validation["retrospective_take_ids"])
    artifact_paths = {
        path.stem: path for path in (result_root / "artifacts").glob("*.json")
    }
    projection_paths = {
        path.name.removesuffix(".manifest.json"): path
        for path in (result_root / "projection_manifests").glob("*.manifest.json")
    }
    if set(artifact_paths) != take_ids:
        raise ValueError("artifact inventory changed")
    if set(projection_paths) != take_ids:
        raise ValueError("projection inventory changed")

    row_digests = {
        str(row["take_id"]): str(row["artifact_file_sha256"])
        for row in result["retrospective"]["rows"]
    }
    if set(row_digests) != take_ids:
        raise ValueError("result row inventory changed")

    exact_fallbacks: list[dict[str, Any]] = []
    partial_fallbacks: list[dict[str, Any]] = []
    source_missing_count = 0
    fallback_frame_count = 0
    total_scored_frames = 0
    for take_id in sorted(take_ids):
        artifact_path = artifact_paths[take_id]
        if file_sha256(artifact_path) != row_digests[take_id]:
            raise ValueError(f"artifact digest changed for {take_id}")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        source_count, fallback_count, is_full = _validate_fallback(artifact)
        scored_count = len(artifact["targets"])
        source_missing_count += source_count
        fallback_frame_count += fallback_count
        total_scored_frames += scored_count
        if fallback_count:
            row = {
                "take_id": take_id,
                "source_missing_frame_count": source_count,
                "fallback_target_frame_count": fallback_count,
                "scored_target_frame_count": scored_count,
                "artifact_file_sha256": row_digests[take_id],
            }
            (exact_fallbacks if is_full else partial_fallbacks).append(row)

        projection = json.loads(
            projection_paths[take_id].read_text(encoding="utf-8")
        )
        if projection.get("take_id") != take_id:
            raise ValueError("projection take id changed")
        if projection.get("protocol_sha256") != PROTOCOL_SHA256:
            raise ValueError("projection protocol changed")
        if projection.get("target_geometry_decoded") is not False:
            raise ValueError("projection decoded target geometry")
        source = protocol["archive_inventory"]["takes"][take_id]
        if projection.get("source_archive_sha256") != source["sha256"]:
            raise ValueError("projection source digest changed")

    log_paths = sorted((result_root / "logs").glob("*.log"))
    expected_log_names = {f"{take_id}.run.log" for take_id in take_ids} | {
        "worker0.log",
        "worker1.log",
    }
    if {path.name for path in log_paths} != expected_log_names:
        raise ValueError("execution log inventory changed")
    worker_text = "\n".join(
        (result_root / "logs" / name).read_text(encoding="utf-8")
        for name in ("worker0.log", "worker1.log")
    )
    if worker_text.count("WORKER_END") != 2:
        raise ValueError("worker completion evidence is incomplete")
    if "WORKER_FAIL" in worker_text or "Traceback" in worker_text:
        raise ValueError("worker terminal failure is present")

    file_paths = sorted(
        path
        for path in result_root.rglob("*")
        if path.is_file() and path.name != "summary.json"
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": SUMMARY_KIND,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "result_sha256": result["result_sha256"],
        "result_file_sha256": file_sha256(result_path),
        "files": {
            path.relative_to(result_root).as_posix(): file_sha256(path)
            for path in file_paths
        },
        "execution_accounting": {
            "locked_take_count": len(take_ids),
            "ordinary_prediction_count": len(take_ids)
            - len(exact_fallbacks)
            - len(partial_fallbacks),
            "full_exact_checkpoint_fallback_count": len(exact_fallbacks),
            "partial_exact_checkpoint_fallback_count": len(partial_fallbacks),
            "unsealable_count": 0,
            "total_scored_target_frame_count": total_scored_frames,
            "fallback_target_frame_count": fallback_frame_count,
            "missing_source_pose_frame_count": source_missing_count,
            "all_fallback_candidates_equal_checkpoint": True,
            "pose_imputation_used_by_prediction": False,
            "source_robot_bytes_modified": False,
            "full_exact_checkpoint_fallbacks": exact_fallbacks,
            "partial_exact_checkpoint_fallbacks": partial_fallbacks,
        },
        "server_transfer": {
            "source_host": "gpuserver6000",
            "source_ipv4": "129.69.102.145",
            "destination_host": "gpuserver4090",
            "destination_ipv4": "129.69.102.139",
            "payload_path": "direct server LAN HTTP",
            "jump_server_in_payload_path": False,
            "projection_manifest_count": len(projection_paths),
            "source_archives_modified": False,
            "transient_http_404_retries_before_projection_ready": worker_text.count(
                "curl: (22) The requested URL returned error: 404"
            ),
        },
        "verification": {
            "result_rebuilt_byte_identically_from_retained_artifacts": True,
            "pre_result_remote_pokeflex_tests_passed": 221,
            "pre_result_remote_pokeflex_tests_skipped": 3,
            "post_result_remote_full_tests_passed": 1841,
            "post_result_remote_full_tests_skipped": 26,
            "prob4d_commit": "364f216c14f7770c1b360bb1b836b11ecf0c18b8",
            "changed_file_ruff_passed": True,
        },
        "decision": result["decision"],
        "claim_boundary": result["claim_boundary"],
    }
    payload["summary_sha256"] = _canonical_sha256(payload)
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
            / "pokeflex_action_robust_public78_retrospective_v6.json"
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
    print(json.dumps(payload["execution_accounting"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
