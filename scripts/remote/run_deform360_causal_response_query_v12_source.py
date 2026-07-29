#!/usr/bin/env python3
"""Run the frozen target-free V12 query-feasibility audit on one source case."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from bayesian_phystwin.deform360_causal_response_preflight import (
    PROPOSAL_CAMERA_IDS,
    REGISTERED_CAMERA_IDS,
    VALIDATION_CAMERA_IDS,
)
from bayesian_phystwin.deform360_causal_response_query import (
    CausalResponseQueryConfig,
    build_causal_response_query_schedule,
    write_causal_response_query_artifacts,
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
    "configs/sota/deform360_causal_response_query_feasibility_v12.json"
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _canonical_config_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


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
    _require(
        payload.get("protocol_id")
        == "deform360-causal-response-query-feasibility-v12-source",
        "protocol ID changed",
    )
    _require(
        payload.get("config_sha256") == _canonical_config_sha256(payload),
        "protocol checksum changed",
    )
    _require(
        payload.get("parent_method_commit")
        == "d5eab1b1dcf8bb77cd7a37f9716f5846559e930c",
        "parent V12 method changed",
    )
    query = CausalResponseQueryConfig(**payload["query"])
    _require(
        query == CausalResponseQueryConfig(),
        "query settings differ from the frozen V12 method",
    )
    records = {str(record["case"]): record for record in payload["cases"]}
    _require(case_id in records, "case is outside the opened source panel")
    return path, payload, records[case_id]


def _read_frame_zero_h5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as stream:
        _require(
            "data" in stream and stream["data"].ndim == 3 and len(stream["data"]) >= 1,
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
        "registered cameras have incompatible frame-zero image shapes",
    )
    return np.stack(depths), np.stack(masks)


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    processed = args.processed_episode_dir.resolve()
    physical_dir = args.physical_dir.resolve()
    revision = _require_clean_repository(repo)
    protocol_path, protocol, case_record = _load_protocol(repo, args.case)

    physical_manifest, physical = validate_dynamic_physical_artifacts(physical_dir)
    _require(
        physical_manifest.get("partition") == "source"
        and physical_manifest.get("case") == args.case,
        "physical artifact is not the opened source case",
    )
    physical_archive_path = physical_dir / PHYSICAL_ARCHIVE_FILENAME
    _require(
        file_sha256(physical_archive_path) == case_record["physical_archive_sha256"],
        "physical archive differs from the source lock",
    )
    geometry = load_complete_camera_geometry(
        processed,
        candidate_camera_names=REGISTERED_CAMERA_IDS,
        minimum_complete_camera_count=len(REGISTERED_CAMERA_IDS),
        frame_count=1,
    )
    _require(
        geometry.camera_names == REGISTERED_CAMERA_IDS,
        "registered camera order changed",
    )
    depth, masks = _load_frame_zero_depth_and_mask(
        processed,
        geometry.camera_names,
        depth_scale_to_m=float(protocol["depth_scale_to_m"]),
    )
    proposal_indices = np.asarray(
        [geometry.camera_names.index(camera) for camera in PROPOSAL_CAMERA_IDS],
        dtype=np.int64,
    )
    validation_indices = np.asarray(
        [geometry.camera_names.index(camera) for camera in VALIDATION_CAMERA_IDS],
        dtype=np.int64,
    )
    schedule = build_causal_response_query_schedule(
        physical["physical_prediction_m"][0],
        physical["graph_basis"],
        physical["action_support"],
        geometry.intrinsics,
        geometry.camera_to_world,
        depth,
        masks,
        camera_ids=geometry.camera_names,
        proposal_camera_indices=proposal_indices,
        validation_camera_indices=validation_indices,
        config=CausalResponseQueryConfig(**protocol["query"]),
    )
    report = write_causal_response_query_artifacts(
        args.output_dir,
        schedule,
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
                "eligible_entity_count": schedule.eligible_entity_count,
                "selected_entity_count": len(schedule.entity_ids),
                "result_sha256": report["result_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
