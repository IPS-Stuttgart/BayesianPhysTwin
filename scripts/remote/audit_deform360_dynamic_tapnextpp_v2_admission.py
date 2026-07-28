#!/usr/bin/env python3
"""Audit V2 camera/query admission on explicitly named open source cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from bayesian_phystwin.deform360_dynamic_query_v2 import (
    AdaptiveDynamicQueryConfig,
    build_adaptive_dynamic_query_schedule,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_admission_v2 import (
    load_complete_camera_geometry,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_physical import (
    PHYSICAL_MANIFEST_FILENAME,
    validate_dynamic_physical_artifacts,
)
from bayesian_phystwin.observation_belief import file_sha256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    return parser.parse_args()


def _processed_case_dir(root: Path, case: str) -> Path:
    object_id, episode_token = case.rsplit("-ep", maxsplit=1)
    _require(
        len(episode_token) == 4 and episode_token.isdigit(),
        f"invalid explicit source case: {case}",
    )
    return root / object_id / f"episode_{episode_token}"


def _audit_case(
    physical_root: Path,
    processed_root: Path,
    case: str,
    config: AdaptiveDynamicQueryConfig,
) -> dict[str, Any]:
    physical_case_dir = physical_root / case
    physical_dir = (
        physical_case_dir
        if (physical_case_dir / PHYSICAL_MANIFEST_FILENAME).is_file()
        else physical_case_dir / "sealed_physical"
    )
    processed_dir = _processed_case_dir(processed_root, case)
    _require(physical_dir.is_dir(), f"physical source case is missing: {case}")
    _require(processed_dir.is_dir(), f"processed source case is missing: {case}")
    manifest, physical = validate_dynamic_physical_artifacts(physical_dir)
    row: dict[str, Any] = {
        "case": case,
        "case_hash": manifest["case_hash"],
        "physical_manifest_file_sha256": file_sha256(
            physical_dir / PHYSICAL_MANIFEST_FILENAME
        ),
    }
    try:
        geometry = load_complete_camera_geometry(processed_dir)
        schedule = build_adaptive_dynamic_query_schedule(
            physical["physical_prediction_m"],
            physical["graph_basis"],
            geometry.intrinsics,
            geometry.camera_to_world,
            geometry.image_shapes_hw,
            geometry.camera_names,
            config=config,
        )
    except (OSError, RuntimeError, ValueError) as error:
        row.update(
            {
                "status": "target_free_admission_rejected",
                "reason_code": type(error).__name__,
                "reason": str(error),
            }
        )
        return row

    row.update(
        {
            "status": "target_free_admission_passed",
            "complete_camera_count": len(geometry.camera_names),
            "rejected_camera_count": len(geometry.rejected_cameras),
            "active_birth_wave_count": len(
                set(map(int, schedule.birth_frames))
            ),
            "skipped_birth_wave_count": len(schedule.skipped_birth_frames),
            "query_count": len(schedule.entity_ids),
            "camera_certificate_sha256": geometry.artifact_sha256,
            "query_schedule_sha256": schedule.artifact_sha256,
            "selected_camera_names": list(schedule.camera_panel.camera_names),
            "active_birth_frames": sorted(
                set(map(int, schedule.birth_frames))
            ),
            "skipped_birth_frames": schedule.skipped_birth_frames.tolist(),
        }
    )
    return row


def main() -> int:
    args = _parse_args()
    cases = tuple(map(str, args.case))
    _require(len(cases) == len(set(cases)), "explicit source cases are repeated")
    config = AdaptiveDynamicQueryConfig()
    physical_root = args.physical_root.resolve()
    processed_root = args.processed_root.resolve()
    rows = [
        _audit_case(physical_root, processed_root, case, config)
        for case in cases
    ]
    passed = sum(row["status"] == "target_free_admission_passed" for row in rows)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360DynamicTAPNextPPV2SourceAdmissionAudit",
        "protocol_id": "deform360-dynamic-tapnextpp-provider-v2-development",
        "status": "post_open_source_only_target_free_audit",
        "config": {
            key: value
            for key, value in vars(config).items()
        },
        "counts": {
            "explicit_open_source_cases": len(rows),
            "target_free_admission_passed": passed,
            "target_free_admission_rejected": len(rows) - passed,
        },
        "cases": rows,
        "information_boundary": {
            "explicit_open_source_cases_only": True,
            "maximum_physical_frame_read": 57,
            "maximum_rgb_depth_mask_frame_read": 57,
            "future_identity_read": False,
            "future_object_observation_read": False,
            "target_metric_read": False,
            "v1_sealed_target_cohort_read": False,
            "held_v8_artifact_read": False,
        },
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
