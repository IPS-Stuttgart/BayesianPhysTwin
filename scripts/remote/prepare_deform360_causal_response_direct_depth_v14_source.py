#!/usr/bin/env python3
"""Prepare one ranked V14 source with official robot and tactile alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bayesian_phystwin.deform360_causal_response_direct_depth_cohort import (
    validate_v14_staging_queue,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_preflight import (
    deform360_v14_case_hash,
)
from bayesian_phystwin.deform360_causal_response_preflight import (
    REGISTERED_CAMERA_IDS,
    deform360_object_hash,
)
from bayesian_phystwin.deform360_fresh_source_download import (
    select_episode_causal_source_files,
)
from bayesian_phystwin.deform360_object_exclusion import file_sha256

DEFORM360_REVISION = "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
DEFORM360_SOURCE_SHA256 = {
    "undistort": "06a500ab2ced8cc960d649d9e200d6d479804ef542ba5aac8fedc5733e74aba9",
    "robot_stage": "5944301cc781f179bea96470af50273836a13fdbb367af9a89a59ce1911c11e0",
    "tactile": "eaceb4263609174cadff7f0b162f2e63a0fed6a4d5be046ffa491c36a905b688",
}
RESULT_KIND = "Deform360CausalDirectDepthSourcePreparationV14"
RESULT_CONTRACT = "deform360-causal-response-direct-depth-preparation-v14"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(
        b"deform360-causal-response-direct-depth-preparation-v14\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _download_manifest_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON artifact: {path}") from error
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_repository(repository: Path) -> str:
    revision = _git_revision(repository)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status.strip(), f"repository has uncommitted files: {repository}")
    return revision


def _parse_bimanual(metadata_path: Path, episode_id: int) -> bool:
    metadata = _read_json(metadata_path)
    sequences = metadata.get("sequences")
    _require(isinstance(sequences, Mapping), "metadata sequences are missing")
    sequence = sequences.get(str(episode_id))
    _require(isinstance(sequence, Mapping), "metadata episode is missing")
    value = sequence.get("bimanual")
    _require(value in {"yes", "no"}, "released bimanual enum changed")
    return value == "yes"


def _download_row(
    manifest: Mapping[str, Any],
    *,
    queue_rank: int,
    object_id: str,
    episode_id: int,
) -> Mapping[str, Any]:
    rows = [
        row
        for row in manifest.get("objects", ())
        if isinstance(row, Mapping)
        and row.get("queue_rank") == queue_rank
        and row.get("object_id") == object_id
        and row.get("episode_id") == episode_id
    ]
    _require(len(rows) == 1, "ranked source is absent from download manifest")
    return rows[0]


def _validate_download(
    path: Path,
    *,
    queue_path: Path,
    queue: Mapping[str, Any],
    queue_rank: int,
    object_id: str,
    episode_id: int,
    raw_object: Path,
) -> Mapping[str, Any]:
    manifest = _read_json(path)
    _require(
        manifest.get("artifact_kind") == "Deform360FreshSourceDownload"
        and manifest.get("revision") == queue["dataset"]["revision"]
        and manifest.get("queue_sha256") == queue["queue_sha256"]
        and manifest.get("queue_file_sha256") == file_sha256(queue_path)
        and manifest.get("download_scope")
        == "ranked_queued_episode_causal_source"
        and manifest.get("tactile_included") is True
        and manifest.get("manifest_sha256")
        == _download_manifest_sha256(manifest),
        "V14 ranked source download binding changed",
    )
    boundary = manifest.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("episode_payload_deserialized") is False
        and boundary.get("future_object_positions_deserialized") is False
        and boundary.get("target_metrics_opened") is False,
        "V14 source download crossed its information boundary",
    )
    row = _download_row(
        manifest,
        queue_rank=queue_rank,
        object_id=object_id,
        episode_id=episode_id,
    )
    files = sorted(path for path in raw_object.rglob("*") if path.is_file())
    _require(
        len(files) == row["file_count"]
        and sum(file.stat().st_size for file in files) == row["total_bytes"],
        "V14 raw source inventory changed",
    )
    relative_paths = [
        f"raw/{object_id}/{file.relative_to(raw_object).as_posix()}"
        for file in files
    ]
    selected = select_episode_causal_source_files(
        relative_paths,
        object_id=object_id,
        episode_id=episode_id,
        required_camera_ids=REGISTERED_CAMERA_IDS,
    )
    _require(
        set(selected) == set(relative_paths),
        "V14 raw source contains files outside the causal source scope",
    )
    metadata_path = raw_object / "metadata.json"
    _require(
        file_sha256(metadata_path) == row["metadata_sha256"],
        "V14 source metadata changed",
    )
    return row


def _output_hashes(episode: Path) -> dict[str, Any]:
    tactile_sensors = sorted(
        path.name
        for path in episode.iterdir()
        if path.is_dir() and (path / "synced_tactile.npy").is_file()
    )
    _require(bool(tactile_sensors), "aligned V14 source lacks tactile outputs")
    return {
        "alignment": file_sha256(episode / "alignment.json"),
        "undistorted_intrinsics": file_sha256(
            episode / "undistorted_intrinsics.npy"
        ),
        "extrinsics": file_sha256(episode / "extrinsics.npy"),
        "robot": file_sha256(episode / "robot" / "robot.npz"),
        "robot_metadata": file_sha256(episode / "robot" / "robot.meta.json"),
        "camera_metadata": {
            camera: file_sha256(episode / camera / "metadata.json")
            for camera in REGISTERED_CAMERA_IDS
        },
        "tactile": {
            sensor: {
                "array": file_sha256(episode / sensor / "synced_tactile.npy"),
                "alignment": file_sha256(episode / sensor / "alignment.json"),
                "metadata": file_sha256(episode / sensor / "metadata.json"),
            }
            for sensor in tactile_sensors
        },
    }


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    _require(not path.exists(), f"refusing to replace V14 preparation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["artifact_sha256"] = _canonical_sha256(payload)
    temporary = path.with_name(f".{path.name}.tmp")
    _require(not temporary.exists(), f"temporary preparation exists: {temporary}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--deform360-repo", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--candidate-rank", type=int, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    repository = args.repo.resolve()
    code_revision = _require_clean_repository(repository)
    protocol_path = args.protocol.resolve()
    protocol = _read_json(protocol_path)
    _require(
        protocol.get("protocol_id")
        == "deform360-causal-response-direct-depth-v14-source",
        "V14 protocol ID changed",
    )
    queue_path = args.queue.resolve()
    queue = validate_v14_staging_queue(queue_path)
    rank = args.candidate_rank
    _require(
        1 <= rank <= len(queue["candidates"]),
        "candidate rank is outside the frozen V14 queue",
    )
    candidate = queue["candidates"][rank - 1]
    object_id = str(candidate["object_id"])
    episode_id = int(candidate["episode_id"])
    object_hash = deform360_object_hash(object_id)
    case_hash = deform360_v14_case_hash(object_id, episode_id)

    raw_object = args.download_root.resolve() / "raw" / object_id
    _require(raw_object.is_dir(), "ranked V14 raw source is missing")
    row = _validate_download(
        args.download_manifest.resolve(),
        queue_path=queue_path,
        queue=queue,
        queue_rank=rank,
        object_id=object_id,
        episode_id=episode_id,
        raw_object=raw_object,
    )
    metadata_path = raw_object / "metadata.json"
    _require(
        file_sha256(metadata_path) == candidate["metadata_sha256"],
        "queue and downloaded metadata disagree",
    )
    bimanual = _parse_bimanual(metadata_path, episode_id)

    deform360_repository = args.deform360_repo.resolve()
    _require(
        _git_revision(deform360_repository) == DEFORM360_REVISION,
        "Deform360 revision changed",
    )
    source_paths = {
        "undistort": deform360_repository / "deform360" / "undistort.py",
        "robot_stage": (
            deform360_repository / "deform360" / "processing" / "robot_stage.py"
        ),
        "tactile": deform360_repository / "deform360" / "tactile.py",
    }
    _require(
        all(
            path.is_file()
            and file_sha256(path) == DEFORM360_SOURCE_SHA256[name]
            for name, path in source_paths.items()
        ),
        "pinned Deform360 source changed",
    )
    result_path = args.result.resolve()
    _require(
        not result_path.exists(),
        f"V14 preparation result already exists: {result_path}",
    )
    aligned_object = args.aligned_root.resolve() / object_id
    episode = aligned_object / f"episode_{episode_id:04d}"
    _require(
        not episode.exists(),
        "V14 aligned source exists without an immutable result",
    )

    base: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": RESULT_KIND,
        "contract": RESULT_CONTRACT,
        "protocol_id": protocol["protocol_id"],
        "protocol_config_sha256": protocol["config_sha256"],
        "queue_sha256": queue["queue_sha256"],
        "queue_file_sha256": file_sha256(queue_path),
        "queue_rank": rank,
        "object_hash": object_hash,
        "case_hash": case_hash,
        "category": candidate["category"],
        "repository_revision": code_revision,
        "deform360_revision": DEFORM360_REVISION,
        "source_sha256": {
            "protocol": file_sha256(protocol_path),
            "queue": file_sha256(queue_path),
            "download_manifest": file_sha256(
                args.download_manifest.resolve()
            ),
            "metadata": row["metadata_sha256"],
            **{
                f"deform360/{name}": file_sha256(path)
                for name, path in source_paths.items()
            },
        },
    }
    try:
        sys.path.insert(0, str(deform360_repository))
        from deform360 import process_tactile_episode, undistort
        from deform360.processing import robot_stage

        episode = undistort.undistort_episode(
            raw_object,
            aligned_object,
            episode_id,
            overwrite=False,
            rebuild_timeline=False,
        )
        robot_stage.process_robot_episode(
            aligned_object,
            episode_id,
            bimanual=bimanual,
            seed=0,
            overwrite=False,
            plot=False,
        )
        process_tactile_episode(
            raw_object,
            aligned_object,
            episode_id,
            overwrite=False,
        )
        alignment = _read_json(episode / "alignment.json")
        cameras = alignment.get("cameras")
        _require(
            isinstance(cameras, list)
            and set(REGISTERED_CAMERA_IDS).issubset(cameras),
            "aligned V14 source lacks the registered camera panel",
        )
        frame_count = alignment.get("frame_count")
        _require(
            isinstance(frame_count, int) and frame_count >= 76,
            "aligned V14 source is too short",
        )
        payload = {
            **base,
            "status": "prepared",
            "bimanual": bimanual,
            "aligned_frame_count": frame_count,
            "registered_camera_count": len(REGISTERED_CAMERA_IDS),
            "outputs_sha256": _output_hashes(episode),
            "information_boundary": {
                "full_rgb_decoded_for_alignment_and_robot_recovery": True,
                "tactile_decoded_for_prefix_contact_support": True,
                "object_mask_or_geometry_created": False,
                "object_response_used_for_source_selection": False,
                "future_identity_or_metric_read": False,
                "target_object_or_outcome_read": False,
                "held_v8_access": False,
            },
        }
    except Exception as error:
        print(
            f"V14 source preparation failed with {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        payload = {
            **base,
            "status": "technical_preflight_failure",
            "failure": {
                "type": type(error).__name__,
                "reason": "official-source-preparation-failed",
            },
            "information_boundary": {
                "full_rgb_may_have_been_decoded_for_alignment_or_robot_recovery": True,
                "tactile_may_have_been_decoded_for_alignment": True,
                "object_mask_or_geometry_created": False,
                "object_response_used_for_source_selection": False,
                "future_identity_or_metric_read": False,
                "target_object_or_outcome_read": False,
                "held_v8_access": False,
            },
        }
    _write_result(result_path, payload)
    print(payload["status"])
    print(payload["artifact_sha256"])
    return 0 if payload["status"] == "prepared" else 2


if __name__ == "__main__":
    raise SystemExit(main())
