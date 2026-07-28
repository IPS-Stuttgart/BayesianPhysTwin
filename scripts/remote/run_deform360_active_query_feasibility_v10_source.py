#!/usr/bin/env python3
"""Run the frozen target-free active-query feasibility audit on one source case."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from bayesian_phystwin.deform360_active_query_feasibility import (
    PROTOCOL_ID,
    ActiveQueryFeasibilityConfig,
    build_active_query_feasibility_audit,
    write_active_query_feasibility_artifacts,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_admission_v2 import (
    load_complete_camera_geometry,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_physical import (
    PHYSICAL_ARCHIVE_FILENAME,
    PHYSICAL_MANIFEST_FILENAME,
    validate_dynamic_physical_artifacts,
)
from bayesian_phystwin.observation_belief import file_sha256

CONFIG_RELATIVE_PATH = Path(
    "configs/sota/deform360_active_query_feasibility_source_v10.json"
)
PHYSICAL_PREFIX_FRAME_COUNT = 58


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _git_output(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_repository(repo: Path) -> str:
    revision = _git_output(repo, "rev-parse", "HEAD")
    _require(not _git_output(repo, "status", "--porcelain"), "repository is dirty")
    return revision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--processed-episode-dir", type=Path, required=True)
    parser.add_argument("--physical-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_protocol(
    repo: Path,
    case_id: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = repo / CONFIG_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("protocol_id") == PROTOCOL_ID, "protocol ID changed")
    records = {str(record["case"]): record for record in payload["cases"]}
    _require(case_id in records, "case is outside the locked source panel")
    ActiveQueryFeasibilityConfig(**payload["feasibility"])
    return path, payload, records[case_id]


def _read_frame_zero_h5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as stream:
        _require(
            "data" in stream
            and stream["data"].ndim == 3
            and len(stream["data"]) >= 1,
            f"invalid frame-zero stream: {path}",
        )
        return np.asarray(stream["data"][0])


def _load_frame_zero_depth_and_mask(
    processed_episode_dir: Path,
    camera_names: tuple[str, ...],
    *,
    depth_scale_to_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    depths: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for camera in camera_names:
        directory = processed_episode_dir / camera
        encoded_depth = _read_frame_zero_h5(directory / "rendered_depth.h5")
        mask = _read_frame_zero_h5(directory / "mask_refined.h5").astype(
            bool,
            copy=False,
        )
        _require(
            encoded_depth.shape == mask.shape,
            f"frame-zero depth and mask differ: {camera}",
        )
        depths.append(encoded_depth.astype(np.float32) * depth_scale_to_m)
        masks.append(mask)
    _require(
        len({value.shape for value in depths}) == 1,
        "complete cameras have incompatible frame-zero image shapes",
    )
    return np.stack(depths), np.stack(masks)


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    processed = args.processed_episode_dir.resolve()
    physical_dir = args.physical_dir.resolve()
    output = args.output_dir.resolve()
    revision = _require_clean_repository(repo)
    protocol_path, protocol, case_record = _load_protocol(repo, args.case)

    physical_manifest, physical = validate_dynamic_physical_artifacts(
        physical_dir
    )
    _require(
        physical_manifest.get("partition") == "source"
        and physical_manifest.get("case") == args.case,
        "physical artifact is not the locked source case",
    )
    physical_archive_path = physical_dir / PHYSICAL_ARCHIVE_FILENAME
    _require(
        file_sha256(physical_archive_path)
        == case_record["physical_archive_sha256"],
        "physical archive differs from the source lock",
    )
    geometry = load_complete_camera_geometry(
        processed,
        minimum_complete_camera_count=protocol[
            "minimum_complete_camera_count"
        ],
        frame_count=1,
    )
    depth, masks = _load_frame_zero_depth_and_mask(
        processed,
        geometry.camera_names,
        depth_scale_to_m=float(protocol["depth_scale_to_m"]),
    )
    audit = build_active_query_feasibility_audit(
        physical["physical_prediction_m"][:PHYSICAL_PREFIX_FRAME_COUNT],
        physical["graph_basis"],
        geometry.intrinsics,
        geometry.camera_to_world,
        geometry.image_shapes_hw,
        geometry.camera_names,
        depth,
        masks,
        config=ActiveQueryFeasibilityConfig(**protocol["feasibility"]),
    )
    report = write_active_query_feasibility_artifacts(
        output,
        audit,
        case_id=args.case,
        repository_revision=revision,
        protocol_path=protocol_path,
        physical_manifest_path=physical_dir / PHYSICAL_MANIFEST_FILENAME,
        physical_archive_path=physical_archive_path,
        camera_certificate_sha256=geometry.artifact_sha256,
    )
    print(
        json.dumps(
            {
                "case": args.case,
                "status": report["status"],
                "candidate_entity_count": (
                    report["audit"]["candidate_entity_count"]
                ),
                "initial_query_count": report["audit"]["initial_query_count"],
                "result_sha256": report["result_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
