#!/usr/bin/env python3
"""Repair a target-free Deform360 prefix whose FFmpeg cadence dropped frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import cv2
import numpy as np

from bayesian_phystwin.deform360_selective_virtual_sensing_artifacts import (
    BACKBONE_ARCHIVE_FILENAME,
    BACKBONE_SEAL_FILENAME,
    build_selective_backbone_seal,
    selective_case_records,
    validate_selective_backbone_seal,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_protocol import (
    EXPECTED_UPDATE_FRAMES,
    PROTOCOL_ID,
    load_selective_virtual_sensing_protocol,
)


PREDICTION_PREFIX_MANIFEST = "prediction_prefix_manifest.json"
FRAME_ZERO_RECONSTRUCTION_MANIFEST = "frame_zero_reconstruction_manifest.json"
CADENCE_REPAIR_SEAL = "prediction_prefix_cadence_repair_seal.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("result_sha256", None)
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_seal(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        payload.get("result_sha256") == _canonical_sha256(payload),
        f"JSON seal checksum changed: {path}",
    )
    return payload


def _write_json_seal(path: Path, payload: dict[str, Any]) -> None:
    payload["result_sha256"] = _canonical_sha256(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _decoded_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        return sum(1 for _ in iter(capture.grab, False))
    finally:
        capture.release()


def _decoded_rgb_sha256(path: Path, expected_count: int) -> str:
    capture = cv2.VideoCapture(str(path))
    digest = hashlib.sha256()
    count = 0
    try:
        while count < expected_count:
            okay, bgr = capture.read()
            _require(okay, f"cannot decode frame {count}: {path}")
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            digest.update(str(rgb.dtype).encode("ascii"))
            digest.update(np.asarray(rgb.shape, dtype=np.int64).tobytes())
            digest.update(rgb.tobytes())
            count += 1
        okay, _ = capture.read()
        _require(not okay, f"video exceeds {expected_count} frames: {path}")
    finally:
        capture.release()
    return digest.hexdigest()


def _first_rgb(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        okay, bgr = capture.read()
    finally:
        capture.release()
    _require(okay, f"cannot decode first frame: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _trim_video_exact(
    ffmpeg: Path,
    source: Path,
    destination: Path,
    *,
    start: int,
    count: int,
) -> None:
    subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"select='between(n,{start},{start + count - 1})',setpts=N/(30*TB)",
            "-frames:v",
            str(count),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            "-fps_mode",
            "cfr",
            str(destination),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    _require(
        _decoded_frame_count(destination) == count,
        f"repaired video does not contain {count} frames: {destination}",
    )


def _ffmpeg_version(ffmpeg: Path) -> str:
    return subprocess.run(
        [str(ffmpeg), "-version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]


def _arrays_equal(first: Path, second: Path) -> bool:
    with (
        np.load(first, allow_pickle=False) as left,
        np.load(second, allow_pickle=False) as right,
    ):
        if left.files != right.files:
            return False
        return all(np.array_equal(left[key], right[key]) for key in left.files)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-aligned-root", type=Path, required=True)
    parser.add_argument("--staged-case", type=Path, required=True)
    parser.add_argument("--backbone-case", type=Path, required=True)
    parser.add_argument("--output-staged-root", type=Path, required=True)
    parser.add_argument("--output-backbone-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_selective_virtual_sensing_protocol(protocol_path)
    staged_case = args.staged_case.resolve()
    backbone_case = args.backbone_case.resolve()
    ffmpeg = args.ffmpeg.resolve()
    _require(ffmpeg.is_file(), f"FFmpeg executable is missing: {ffmpeg}")
    prefix_manifest_path = staged_case / PREDICTION_PREFIX_MANIFEST
    frame_zero_manifest_path = staged_case / FRAME_ZERO_RECONSTRUCTION_MANIFEST
    prefix_manifest = _read_json_seal(prefix_manifest_path)
    frame_zero_manifest = _read_json_seal(frame_zero_manifest_path)
    _require(
        prefix_manifest.get("artifact_kind") == "Deform360SelectivePredictionPrefix",
        "unsupported prediction-prefix manifest",
    )
    _require(prefix_manifest.get("protocol_id") == PROTOCOL_ID, "protocol changed")
    _require(
        prefix_manifest.get("protocol_config_sha256") == protocol["config_sha256"],
        "protocol checksum changed",
    )
    case = str(prefix_manifest["case"])
    _require(staged_case.name == case == backbone_case.name, "case paths disagree")
    expected = {str(row["case"]): row for row in selective_case_records(protocol_path)}
    _require(case in expected, "case is outside the locked cohort")
    object_id = str(prefix_manifest["object_id"])
    episode_id = int(prefix_manifest["episode_id"])
    source_episode = (
        args.source_aligned_root.resolve() / object_id / f"episode_{episode_id:04d}"
    )
    _require(source_episode.is_dir(), f"source episode is missing: {source_episode}")
    start, _ = prefix_manifest["action_window"]["selected_raw_frame_range_half_open"]
    prefix_count = EXPECTED_UPDATE_FRAMES[-1] + 1
    _require(prefix_count == 58, "locked prefix length changed")
    camera_records = list(prefix_manifest["camera_records"])
    cameras = [str(row["camera"]) for row in camera_records]
    _require(
        len(cameras) >= 8 and len(cameras) == len(set(cameras)), "camera set changed"
    )

    output_staged = args.output_staged_root.resolve() / case
    output_backbone = args.output_backbone_root.resolve() / case
    _require(
        not output_staged.exists(), f"repaired staged case exists: {output_staged}"
    )
    _require(
        not output_backbone.exists(), f"repaired backbone exists: {output_backbone}"
    )
    output_staged.parent.mkdir(parents=True, exist_ok=True)
    output_backbone.parent.mkdir(parents=True, exist_ok=True)
    scratch = output_staged.with_name(f".{case}.cadence-repair-{os.getpid()}")
    _require(not scratch.exists(), f"repair scratch exists: {scratch}")
    old_prefix_result = str(prefix_manifest["result_sha256"])
    old_frame_zero_result = str(frame_zero_manifest["result_sha256"])
    input_counts: dict[str, int] = {}
    decoded_digests: dict[str, str] = {}
    first_frame_mae: dict[str, float] = {}
    try:
        shutil.copytree(staged_case, scratch)
        record_by_camera = {str(row["camera"]): row for row in camera_records}
        for camera in cameras:
            old_video = (
                staged_case / "prefix" / "episode_0000" / camera / "undistorted.mp4"
            )
            repaired_video = (
                scratch / "prefix" / "episode_0000" / camera / "undistorted.mp4"
            )
            source_video = source_episode / camera / "undistorted.mp4"
            input_counts[camera] = _decoded_frame_count(old_video)
            _require(
                input_counts[camera] < prefix_count,
                f"input prefix is not cadence-truncated: {old_video}",
            )
            temporary_video = repaired_video.with_suffix(".repairing.mp4")
            _trim_video_exact(
                ffmpeg,
                source_video,
                temporary_video,
                start=int(start),
                count=prefix_count,
            )
            old_first = _first_rgb(old_video).astype(np.float32)
            new_first = _first_rgb(temporary_video).astype(np.float32)
            _require(old_first.shape == new_first.shape, "repaired frame shape changed")
            first_frame_mae[camera] = float(np.mean(np.abs(old_first - new_first)))
            _require(
                first_frame_mae[camera] <= 8.0,
                f"repaired first frame differs unexpectedly for {camera}",
            )
            os.replace(temporary_video, repaired_video)
            record_by_camera[camera]["prefix_video_sha256"] = _sha256(repaired_video)
            decoded_digests[camera] = _decoded_rgb_sha256(repaired_video, prefix_count)

        repaired_prefix_manifest = dict(prefix_manifest)
        repaired_prefix_manifest["camera_records"] = camera_records
        repaired_prefix_manifest["target_free_operational_cadence_repair"] = {
            "superseded_result_sha256": old_prefix_result,
            "reason": (
                "FFmpeg 7 default cadence emitted 49 of 58 selected frames; "
                "explicit 30 Hz CFR materializes the frozen exact frame range"
            ),
            "source_frame_range_half_open": [int(start), int(start) + prefix_count],
            "input_decoded_frame_count_by_camera": input_counts,
            "output_decoded_frame_count": prefix_count,
            "output_decoded_rgb_sha256_by_camera": decoded_digests,
            "first_frame_mean_absolute_difference_by_camera": first_frame_mae,
            "ffmpeg_sha256": _sha256(ffmpeg),
            "ffmpeg_version": _ffmpeg_version(ffmpeg),
            "target_data_read": False,
            "future_dense_reconstruction_read": False,
            "future_particle_tracks_read": False,
        }
        repaired_prefix_path = scratch / PREDICTION_PREFIX_MANIFEST
        _write_json_seal(repaired_prefix_path, repaired_prefix_manifest)

        repaired_frame_zero_manifest = dict(frame_zero_manifest)
        repaired_frame_zero_manifest["inputs_sha256"] = dict(
            frame_zero_manifest["inputs_sha256"]
        )
        repaired_frame_zero_manifest["inputs_sha256"]["prediction_prefix_manifest"] = (
            _sha256(repaired_prefix_path)
        )
        repaired_frame_zero_manifest["target_free_operational_cadence_repair"] = {
            "superseded_result_sha256": old_frame_zero_result,
            "frame_zero_outputs_recomputed": False,
            "frame_zero_outputs_changed": False,
            "prediction_prefix_manifest_resealed": True,
            "target_data_read": False,
        }
        repaired_frame_zero_path = scratch / FRAME_ZERO_RECONSTRUCTION_MANIFEST
        _write_json_seal(repaired_frame_zero_path, repaired_frame_zero_manifest)
        os.replace(scratch, output_staged)

        old_seal = json.loads(
            (backbone_case / BACKBONE_SEAL_FILENAME).read_text(encoding="utf-8")
        )
        validate_selective_backbone_seal(
            old_seal,
            protocol_path=protocol_path,
            case_dir=backbone_case,
        )
        old_archive = backbone_case / BACKBONE_ARCHIVE_FILENAME
        with np.load(old_archive, allow_pickle=False) as stored:
            points = np.asarray(stored["frame_zero_points_m"]).copy()
        new_seal = build_selective_backbone_seal(
            protocol_path,
            output_backbone,
            object_id=object_id,
            episode_id=episode_id,
            frame_zero_points_m=points,
            frame_zero_reconstruction_manifest=(
                output_staged / FRAME_ZERO_RECONSTRUCTION_MANIFEST
            ),
            prediction_stage_manifest=output_staged / PREDICTION_PREFIX_MANIFEST,
        )
        new_archive = output_backbone / BACKBONE_ARCHIVE_FILENAME
        _require(
            _arrays_equal(old_archive, new_archive),
            "backbone scientific arrays changed during cadence reseal",
        )
        evidence: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": "Deform360PredictionPrefixCadenceRepairSeal",
            "protocol_id": PROTOCOL_ID,
            "protocol_config_sha256": protocol["config_sha256"],
            **expected[case],
            "superseded": {
                "prediction_prefix_manifest_sha256": _sha256(prefix_manifest_path),
                "frame_zero_reconstruction_manifest_sha256": _sha256(
                    frame_zero_manifest_path
                ),
                "backbone_seal_sha256": _sha256(backbone_case / BACKBONE_SEAL_FILENAME),
                "backbone_archive_sha256": _sha256(old_archive),
            },
            "repaired": {
                "prediction_prefix_manifest_sha256": _sha256(
                    output_staged / PREDICTION_PREFIX_MANIFEST
                ),
                "frame_zero_reconstruction_manifest_sha256": _sha256(
                    output_staged / FRAME_ZERO_RECONSTRUCTION_MANIFEST
                ),
                "backbone_seal_sha256": _sha256(
                    output_backbone / BACKBONE_SEAL_FILENAME
                ),
                "backbone_archive_sha256": _sha256(new_archive),
                "backbone_result_sha256": new_seal["result_sha256"],
            },
            "scientific_backbone_arrays_bit_exact": True,
            "target_data_read": False,
            "future_dense_reconstruction_read": False,
            "future_particle_tracks_read": False,
            "target_metric_read": False,
        }
        _write_json_seal(output_staged / CADENCE_REPAIR_SEAL, evidence)
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        shutil.rmtree(output_staged, ignore_errors=True)
        shutil.rmtree(output_backbone, ignore_errors=True)
        raise

    print(json.dumps(evidence, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
