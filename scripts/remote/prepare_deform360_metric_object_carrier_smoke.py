#!/usr/bin/env python3
"""Freeze the source-only metric object-carrier smoke before mask access."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_metric_object_carrier import (
    DEFORM360_METRIC_OBJECT_CARRIER_LOCK_SCHEMA,
    METRIC_OBJECT_CARRIER_INFORMATION_BOUNDARY,
    METRIC_OBJECT_CARRIER_POLICY,
    validate_metric_object_carrier_lock,
)
from bayesian_phystwin.deform360_tactile_metric_gauge import (
    load_tactile_metric_gauge_lock,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _git_head(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--metric-gauge-lock", type=Path, required=True)
    parser.add_argument("--metric-gauge-result", type=Path, required=True)
    parser.add_argument("--parent-provider-root", type=Path, required=True)
    parser.add_argument("--supplemental-provider-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--selector-source", type=Path, required=True)
    parser.add_argument("--sam2-repository", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repository = args.repository.resolve()
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status, "implementation checkout is dirty")
    revision = _git_head(repository)
    metric_lock_path = args.metric_gauge_lock.resolve()
    metric_lock = load_tactile_metric_gauge_lock(metric_lock_path)
    metric_result_path = args.metric_gauge_result.resolve()
    metric_result = _json(metric_result_path)
    _require(metric_result.get("status") == "admitted", "metric gauge was not admitted")
    gate = metric_result.get("gate")
    _require(isinstance(gate, dict), "metric-gauge gate is missing")
    assert isinstance(gate, dict)
    _require(gate.get("metric_gauge_authorized") is True, "metric gauge unauthorized")
    _require(gate.get("contact_anchor_authorized") is False, "contact already authorized")
    _require(
        metric_result.get("lock_id") == metric_lock["artifact_id"],
        "metric-gauge result has different parent lock",
    )
    cameras = list(metric_lock["camera_selection"]["selected_cameras"])
    _require(len(cameras) == 3, "metric-gauge panel changed")
    object_id = str(metric_lock["source_case"]["object_id"])
    episode = "episode_0000"
    source_root = args.source_root.resolve()
    parent_root = args.parent_provider_root.resolve()
    supplemental_root = args.supplemental_provider_root.resolve()
    providers = []
    for camera in cameras:
        video = source_root / object_id / episode / camera / "undistorted.mp4"
        _require(video.is_file(), f"source video is missing: {video}")
        roots = [
            root
            for root in (parent_root, supplemental_root)
            if (root / object_id / episode / camera / "predictions.json").is_file()
        ]
        _require(len(roots) == 1, f"provider root is ambiguous for {camera}")
        provider_dir = roots[0] / object_id / episode / camera
        manifest_path = provider_dir / "predictions.json"
        manifest = _json(manifest_path)
        windows = manifest.get("overlap_windows")
        _require(isinstance(windows, list), f"overlap windows missing for {camera}")
        matches = [
            row
            for row in windows
            if isinstance(row, dict)
            and row.get("start_frame") == 125
            and row.get("stop_frame") == 150
        ]
        _require(len(matches) == 1, f"carrier window changed for {camera}")
        window = provider_dir / str(matches[0]["path"])
        _require(window.is_file(), f"carrier window is missing: {window}")
        members = manifest.get("artifact_integrity", {}).get("members", [])
        member_matches = [
            row
            for row in members
            if isinstance(row, dict) and row.get("path") == matches[0]["path"]
        ]
        _require(len(member_matches) == 1, f"window integrity missing for {camera}")
        window_sha = _sha256(window)
        _require(member_matches[0].get("sha256") == window_sha, "provider window drift")
        providers.append(
            {
                "camera": camera,
                "video_path": str(video),
                "video_sha256": _sha256(video),
                "prediction_manifest_path": str(manifest_path),
                "prediction_manifest_sha256": _sha256(manifest_path),
                "window_path": str(window),
                "window_sha256": window_sha,
                "window_source_frames": [125, 150],
            }
        )
    sam2_repository = args.sam2_repository.resolve()
    selector_source = args.selector_source.resolve()
    checkpoint = args.sam2_checkpoint.resolve()
    module_source = repository / "src/bayesian_phystwin/deform360_metric_object_carrier.py"
    runner_source = repository / "scripts/remote/run_deform360_metric_object_carrier_smoke.py"
    descriptor: dict[str, object] = {
        "schema": DEFORM360_METRIC_OBJECT_CARRIER_LOCK_SCHEMA,
        "schema_version": 1,
        "status": "locked-source-only-pre-mask",
        "implementation": {
            "revision": revision,
            "runner_source_sha256": _sha256(runner_source),
            "module_source_sha256": _sha256(module_source),
        },
        "source_case": {
            "object_id": object_id,
            "processing_episode_index": 0,
            "causal_frame_stop": 150,
        },
        "parents": {
            "metric_gauge_lock": {
                "artifact_id": metric_lock["artifact_id"],
                "sha256": _sha256(metric_lock_path),
            },
            "metric_gauge_result": {
                "artifact_id": metric_result["artifact_id"],
                "sha256": _sha256(metric_result_path),
            },
        },
        "cameras": cameras,
        "reference_camera": cameras[0],
        "providers": providers,
        "sam2": {
            "repository_path": str(sam2_repository),
            "repository_revision": _git_head(sam2_repository),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "selector_source_path": str(selector_source),
            "selector_source_sha256": _sha256(selector_source),
        },
        "policy": METRIC_OBJECT_CARRIER_POLICY,
        "information_boundary": METRIC_OBJECT_CARRIER_INFORMATION_BOUNDARY,
        "claim_boundary": (
            "Source-only metric object-carrier feasibility. Admission does not "
            "authorize a contact/state update, calibration-score access, "
            "confirmation access, or a SOTA claim."
        ),
    }
    value = {"artifact_id": content_id(descriptor), **descriptor}
    validate_metric_object_carrier_lock(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"artifact_id": value["artifact_id"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
