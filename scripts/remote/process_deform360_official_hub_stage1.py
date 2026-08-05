#!/usr/bin/env python3
"""Process the locked Deform360 calibration view with the pinned upstream code."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from bayesian_phystwin.deform360_official_hub_stage1 import (
    EXPECTED_PROCESSING_REVISION,
    load_official_hub_stage1_lock,
    validate_official_hub_stage1_processing_view,
    write_official_hub_stage1_manifest,
)

PROCESSING_REPORT_SCHEMA = (
    "bayesian-phystwin/deform360-official-hub-stage1-processing-report-v1"
)


def _load_object(path: Path, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {name}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _tree_inventory(root: Path) -> tuple[list[dict[str, object]], int]:
    files: list[dict[str, object]] = []
    total_bytes = 0
    if root.exists():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            size = path.stat().st_size
            total_bytes += size
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": size,
                    "sha256": _file_sha256(path),
                }
            )
    return files, total_bytes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--preflight-manifest", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--processing-view-root", type=Path, required=True)
    parser.add_argument("--processing-repository", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repository = args.repository.resolve()
    processing_repository = args.processing_repository.resolve()
    implementation_head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    implementation_status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if implementation_status:
        raise ValueError("Bayesian-PhysTwin processing checkout is dirty")
    processing_head = subprocess.run(
        ["git", "-C", str(processing_repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if processing_head != EXPECTED_PROCESSING_REVISION:
        raise ValueError("official processing revision changed")
    processing_status = subprocess.run(
        ["git", "-C", str(processing_repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if processing_status:
        raise ValueError("official processing checkout is dirty")

    lock = load_official_hub_stage1_lock(
        repository,
        args.protocol,
        args.selection,
    )
    preflight = _load_object(args.preflight_manifest, name="preflight manifest")
    download = _load_object(args.download_manifest, name="download manifest")
    view = _load_object(
        args.processing_view_root / "stage1_processing_view.json",
        name="processing-view manifest",
    )
    validate_official_hub_stage1_processing_view(
        view,
        preflight=preflight,
        download=download,
        view_root=args.processing_view_root,
        payload_root=args.payload_root,
        lock=lock,
    )

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise ValueError("Stage-1 processing output already exists")
    output_root.mkdir(parents=True)

    sys.path.insert(0, str(processing_repository))
    from deform360.calibration import load_calibration
    from deform360.dataset import MultiViewDataset
    from deform360.tactile import (
        DEFAULT_THRESHOLD,
        DEFAULT_TOLERANCE_US,
        TactileDataset,
        process_tactile_episode,
    )
    from deform360.undistort import undistort_episode

    rows_by_object = {str(row["object_id"]): row for row in view["objects"]}
    results: list[dict[str, object]] = []
    for selected in lock.calibration:
        object_id = selected.object_id
        object_input = args.processing_view_root / "raw" / object_id
        object_output = output_root / object_id
        result: dict[str, object] = {
            "object_id": object_id,
            "stratum": selected.stratum,
            "source_episode_id": selected.episode_id,
            "processing_episode_index": 0,
            "action": rows_by_object[object_id]["action"],
        }
        try:
            calibration = load_calibration(object_input)
            episode_output = undistort_episode(
                object_dir=object_input,
                output_dir=object_output,
                episode_index=0,
                cameras=None,
                calib=calibration,
                tol_units=100_000,
                overwrite=True,
                rebuild_timeline=False,
            )
            tactile_outputs = process_tactile_episode(
                object_dir=object_input,
                aligned_dir=object_output,
                episode_index=0,
                output_dir=None,
                sensors=None,
                threshold=DEFAULT_THRESHOLD,
                tolerance_us=DEFAULT_TOLERANCE_US,
                invalid_columns=(-1,),
                legacy_scale=True,
                out_of_tolerance="keep",
                duplicate_policy="last",
                overwrite=True,
            )
            visual = MultiViewDataset(episode_output)
            tactile = TactileDataset(episode_output)
            result.update(
                {
                    "status": "success",
                    "camera_count": len(visual.cameras),
                    "frame_count": len(visual),
                    "tactile_sensor_count": len(tactile.sensors),
                    "tactile_frame_count": len(tactile),
                    "tactile_outputs": sorted(tactile_outputs),
                }
            )
            visual.close()
            tactile.close()
        except Exception as error:
            result.update(
                {
                    "status": "retained_technical_failure",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
        files, total_bytes = _tree_inventory(object_output)
        result["output_file_count"] = len(files)
        result["output_total_bytes"] = total_bytes
        result["output_tree_sha256"] = _canonical_sha256(files)
        result["output_files"] = files
        results.append(result)

    success_count = sum(row["status"] == "success" for row in results)
    report: dict[str, object] = {
        "schema": PROCESSING_REPORT_SCHEMA,
        "schema_version": 1,
        "protocol_id": lock.protocol_id,
        "protocol_sha256": lock.protocol_sha256,
        "preflight_sha256": preflight["preflight_sha256"],
        "download_sha256": download["download_sha256"],
        "processing_view_sha256": view["processing_view_sha256"],
        "implementation_revision": implementation_head,
        "official_processing": {
            "repository": lock.processing_repository,
            "revision": processing_head,
        },
        "role": "calibration",
        "status": (
            "complete"
            if success_count == len(results)
            else "complete_with_retained_technical_failures"
        ),
        "object_count": len(results),
        "success_count": success_count,
        "retained_technical_failure_count": len(results) - success_count,
        "processing_parameters": {
            "episode_index": 0,
            "camera_tolerance_us": 100_000,
            "tactile_tolerance_us": DEFAULT_TOLERANCE_US,
            "tactile_threshold": DEFAULT_THRESHOLD,
            "tactile_invalid_columns": [-1],
            "tactile_legacy_scale": True,
            "tactile_out_of_tolerance": "keep",
            "tactile_duplicate_policy": "last",
        },
        "objects": results,
        "physical_backend_contract": {
            "minimum_node_count": 128,
            "status": "pending-reconstruction",
        },
        "information_boundary": {
            "calibration_payload_opened": True,
            "confirmation_payload_opened": False,
            "future_target_opened": False,
            "replacement_performed": False,
            "technical_failures_retained": True,
        },
    }
    report["processing_report_sha256"] = _canonical_sha256(report)
    write_official_hub_stage1_manifest(args.report, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "success_count": success_count,
                "retained_technical_failure_count": len(results) - success_count,
                "processing_report_sha256": report["processing_report_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if success_count != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
