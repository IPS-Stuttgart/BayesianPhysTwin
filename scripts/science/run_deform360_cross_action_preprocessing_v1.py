#!/usr/bin/env python3
"""Run and verify the locked Deform360 cross-action preprocessing stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

LOCK_SCHEMA: Final = "bayesian-phystwin/deform360-cross-action-pair-lock-v1"
PROTOCOL_SCHEMA: Final = "bayesian-phystwin/deform360-cross-action-preprocessing-v1"
VERIFICATION_SCHEMA: Final = (
    "bayesian-phystwin/deform360-cross-action-preprocessing-verification-v1"
)
EXPECTED_OBJECTS: Final = (
    "001-rope",
    "002-rope-silk",
    "003-cable",
    "081-stripe-rope",
)
EXPECTED_PAIRS: Final = {
    "001-rope": (0, 2, "move", "lift"),
    "002-rope-silk": (0, 2, "lift", "drag"),
    "003-cable": (0, 2, "lift", "drag"),
    "081-stripe-rope": (0, 2, "move", "lift"),
}
CAMERA_REQUIRED: Final = (
    "undistorted.mp4",
    "undistorted_000000.png",
    "aligned_timestamps.txt",
    "alignment.json",
    "metadata.json",
)
TACTILE_REQUIRED: Final = (
    "synced_tactile.npy",
    "alignment.json",
    "metadata.json",
)
EPISODE_REQUIRED: Final = (
    "alignment.json",
    "undistorted_intrinsics.npy",
    "extrinsics.npy",
)
ROBOT_REQUIRED: Final = ("robot.npz", "robot.meta.json")


class PreprocessingError(ValueError):
    """Raised when a frozen preprocessing contract cannot be satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreprocessingError(message)


def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PreprocessingError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs_hook,
        )
    except PreprocessingError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreprocessingError(f"cannot read JSON: {path}") from error
    require(type(value) is dict, f"JSON root must be an object: {path}")
    return value


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_id(value: Mapping[str, object], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_pair_lock(path: Path) -> dict[str, Any]:
    lock = read_json(path.resolve())
    require(lock.get("schema") == LOCK_SCHEMA, "unexpected pair-lock schema")
    require(lock.get("schema_version") == 1, "unsupported pair-lock version")
    require(
        lock.get("status") == "frozen-after-target-blind-action-binding",
        "pair lock is not frozen",
    )
    require(
        lock.get("lock_id") == content_id(lock, "lock_id"),
        "pair-lock content ID mismatch",
    )
    require(
        lock.get("raw_root")
        == "/mnt/lexar4tb/datasets/deform360/data-7fea8e2/raw",
        "raw-root identity changed",
    )
    source = lock.get("source_plan")
    require(type(source) is dict, "source-plan identity is missing")
    expected_source = {
        "run_id": 33337674563,
        "head_sha": "b627a9ba5eb3f6232475267adad24eb31f969123",
        "plan_id": "0c369d36a9db865bfe6ed00bbbb7d708f27042c992d80f293c406374247bd96a",
        "artifact_id": 9739536698,
        "artifact_digest": (
            "sha256:5628aafc473be0908873ea913c5d3b25"
            "b0d52b9189b9e71ff04672a33593862f"
        ),
        "source_roster_run_id": 33335964618,
        "source_roster_result_id": (
            "b73b0e2ee69158d59a1d1189014071a9"
            "dbb7a715de6e88d3edde9156ba999efa"
        ),
        "metadata_probe_run_id": 33337433438,
    }
    require(source == expected_source, "source-plan identity changed")
    boundary = lock.get("information_boundary")
    require(
        boundary
        == {
            "media_payload_decoded": False,
            "numeric_arrays_loaded": False,
            "score_bearing_outcomes_used": False,
            "target_future_opened": False,
        },
        "pair-lock information boundary changed",
    )
    pairs = lock.get("pairs")
    require(type(pairs) is list and len(pairs) == 4, "expected four locked pairs")
    require(
        tuple(pair.get("object_id") for pair in pairs) == EXPECTED_OBJECTS,
        "locked object roster changed",
    )
    for pair in pairs:
        require(type(pair) is dict, "locked pair must be an object")
        object_id = pair["object_id"]
        expected = EXPECTED_PAIRS[object_id]
        observed = (
            pair.get("source_episode_index"),
            pair.get("target_episode_index"),
            pair.get("source_action_family"),
            pair.get("target_action_family"),
        )
        require(observed == expected, f"locked pair changed for {object_id}")
        require(pair.get("bimanual") is False, "manuality changed")
        require(pair.get("nonprehensile") is False, "contact mode changed")
        require(pair.get("contact_anchor") == "single-edge", "anchor changed")
        cameras = pair.get("cameras")
        sensors = pair.get("tactile_streams")
        require(
            type(cameras) is list
            and len(cameras) == 12
            and cameras == sorted(set(cameras)),
            f"camera panel changed for {object_id}",
        )
        require(
            type(sensors) is list
            and len(sensors) == 4
            and sensors == sorted(set(sensors)),
            f"tactile panel changed for {object_id}",
        )
        metadata_hash = pair.get("metadata_sha256")
        require(
            type(metadata_hash) is str and len(metadata_hash) == 64,
            f"metadata identity missing for {object_id}",
        )
    return lock


def load_protocol(path: Path, lock: Mapping[str, object]) -> dict[str, Any]:
    protocol = read_json(path.resolve())
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "unexpected protocol schema")
    require(protocol.get("schema_version") == 1, "unsupported protocol version")
    require(
        protocol.get("status")
        == "authorized-retrospective-fixed-pair-preprocessing",
        "preprocessing is not authorized",
    )
    pair_lock = protocol.get("pair_lock")
    require(type(pair_lock) is dict, "pair-lock binding is missing")
    require(pair_lock.get("lock_id") == lock.get("lock_id"), "lock binding changed")
    require(
        protocol.get("raw_root") == lock.get("raw_root"),
        "protocol raw root differs from lock",
    )
    persistent = protocol.get("persistent_output_root")
    require(
        type(persistent) is str and persistent.startswith("/"),
        "persistent output root must be absolute",
    )
    upstream = protocol.get("official_processing")
    require(type(upstream) is dict, "official-processing identity is missing")
    require(
        upstream.get("repository") == "lhy0807/deform360",
        "official-processing repository changed",
    )
    require(
        upstream.get("revision")
        == "d8522a4403b766aeb387510c04e89032a56fdf35",
        "official-processing revision changed",
    )
    require(upstream.get("package_version") == "0.2.0", "package version changed")
    processing = protocol.get("processing")
    require(type(processing) is dict, "processing configuration is missing")
    require(processing.get("camera_count_per_pair") == 12, "camera count changed")
    require(
        processing.get("camera_alignment_tolerance_us") == 100000,
        "camera tolerance changed",
    )
    require(
        processing.get("tactile_alignment_tolerance_us") == 150000,
        "tactile tolerance changed",
    )
    require(
        processing.get("robot_ransac_seed") == 20260831,
        "robot seed changed",
    )
    boundary = protocol.get("information_boundary")
    require(type(boundary) is dict, "information boundary is missing")
    for key in (
        "target_dependent_model_selection",
        "physical_parameter_inference",
        "model_prediction",
        "target_scoring",
        "paper_claim_authorized",
    ):
        require(boundary.get(key) is False, f"unauthorized boundary opened: {key}")
    require(
        boundary.get("retrospective_development") is True,
        "retrospective role changed",
    )
    require(
        boundary.get("fixed_preprocessor_may_decode_selected_source_and_target_media")
        is True,
        "fixed preprocessing is not authorized",
    )
    return protocol


def worklist(lock: Mapping[str, object]) -> list[dict[str, object]]:
    result = []
    for pair in lock["pairs"]:
        result.append(
            {
                "object_id": pair["object_id"],
                "metadata_sha256": pair["metadata_sha256"],
                "source_episode_index": pair["source_episode_index"],
                "target_episode_index": pair["target_episode_index"],
                "source_action": pair["source_action"],
                "target_action": pair["target_action"],
                "cameras": pair["cameras"],
                "tactile_streams": pair["tactile_streams"],
                "bimanual": pair["bimanual"],
            }
        )
    return result


def manifest(
    protocol: Mapping[str, object],
    lock: Mapping[str, object],
    repository_revision: str,
) -> dict[str, object]:
    require(
        len(repository_revision) == 40
        and all(character in "0123456789abcdef" for character in repository_revision),
        "repository revision must be a full lowercase SHA",
    )
    value: dict[str, object] = {
        "schema": "bayesian-phystwin/deform360-cross-action-preprocessing-manifest-v1",
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "repository_revision": repository_revision,
        "pair_lock_id": lock["lock_id"],
        "source_plan": lock["source_plan"],
        "official_processing": protocol["official_processing"],
        "raw_root": protocol["raw_root"],
        "worklist": worklist(lock),
        "information_boundary": protocol["information_boundary"],
        "claim_boundary": protocol["claim_boundary"],
    }
    value["manifest_id"] = content_id(value, "manifest_id")
    return value


def process(
    protocol: Mapping[str, Any],
    lock: Mapping[str, Any],
    output_root: Path,
) -> dict[str, object]:
    from deform360.calibration import load_calibration
    from deform360.layout import camera_recordings, tactile_recordings
    from deform360.processing.robot_stage import process_robot_episode
    from deform360.tactile import process_tactile_episode
    from deform360.undistort import undistort_episode

    raw_root = Path(protocol["raw_root"]).resolve(strict=True)
    output_root = output_root.resolve()
    require(
        not output_root.exists(),
        f"unique output root already exists: {output_root}",
    )
    require(
        raw_root not in output_root.parents and output_root != raw_root,
        "output root must not be inside the raw snapshot",
    )
    output_root.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    processing = protocol["processing"]
    for item in worklist(lock):
        object_id = str(item["object_id"])
        object_dir = raw_root / object_id
        require(object_dir.is_dir(), f"raw object unavailable: {object_id}")
        metadata_path = object_dir / "metadata.json"
        require(metadata_path.is_file(), f"metadata unavailable: {object_id}")
        require(
            sha256_file(metadata_path) == item["metadata_sha256"],
            f"metadata identity changed for {object_id}",
        )
        cameras = [str(value) for value in item["cameras"]]
        sensors = [str(value) for value in item["tactile_streams"]]
        calibration = load_calibration(object_dir)
        missing_calibration = [camera for camera in cameras if camera not in calibration]
        require(
            not missing_calibration,
            f"selected cameras lack calibration for {object_id}: {missing_calibration}",
        )
        target_index = int(item["target_episode_index"])
        for camera in cameras:
            recordings = camera_recordings(object_dir / camera)
            require(
                len(recordings) > target_index,
                f"selected camera lacks target episode: {object_id}/{camera}",
            )
        for sensor in sensors:
            recordings = tactile_recordings(object_dir / sensor)
            require(
                len(recordings) > target_index,
                f"selected tactile stream lacks target episode: {object_id}/{sensor}",
            )
        aligned_object = output_root / "aligned" / object_id
        aligned_object.mkdir(parents=True)
        selection = {
            "object_id": object_id,
            "pair_lock_id": lock["lock_id"],
            "source_episode_index": item["source_episode_index"],
            "target_episode_index": item["target_episode_index"],
            "source_action": item["source_action"],
            "target_action": item["target_action"],
            "cameras": cameras,
            "tactile_streams": sensors,
            "bimanual": item["bimanual"],
        }
        write_json(aligned_object / "selection.json", selection)
        for role, episode_index in (
            ("source", int(item["source_episode_index"])),
            ("target", int(item["target_episode_index"])),
        ):
            print(
                f"[{object_id}] {role} episode {episode_index}: "
                "undistort and synchronize RGB",
                flush=True,
            )
            episode_dir = undistort_episode(
                object_dir=object_dir,
                output_dir=aligned_object,
                episode_index=episode_index,
                cameras=cameras,
                calib=calibration,
                tol_units=int(processing["camera_alignment_tolerance_us"]),
                overwrite=True,
                rebuild_timeline=False,
            )
            print(
                f"[{object_id}] {role} episode {episode_index}: align tactile",
                flush=True,
            )
            tactile_outputs = process_tactile_episode(
                object_dir=object_dir,
                aligned_dir=aligned_object,
                episode_index=episode_index,
                sensors=sensors,
                tolerance_us=int(processing["tactile_alignment_tolerance_us"]),
                out_of_tolerance="keep",
                duplicate_policy="last",
                invalid_columns=(-1,),
                legacy_scale=True,
                overwrite=True,
            )
            print(
                f"[{object_id}] {role} episode {episode_index}: recover robot state",
                flush=True,
            )
            robot_output = process_robot_episode(
                aligned_object,
                episode_index,
                bimanual=bool(item["bimanual"]),
                cameras=cameras,
                seed=int(processing["robot_ransac_seed"]),
                overwrite=True,
                plot=False,
            )
            rows.append(
                {
                    "object_id": object_id,
                    "role": role,
                    "episode_index": episode_index,
                    "episode_dir": str(episode_dir),
                    "camera_count": len(cameras),
                    "tactile_count": len(tactile_outputs),
                    "robot_output": str(robot_output),
                }
            )
    receipt: dict[str, object] = {
        "schema": "bayesian-phystwin/deform360-cross-action-preprocessing-receipt-v1",
        "schema_version": 1,
        "pair_lock_id": lock["lock_id"],
        "output_root": str(output_root),
        "episode_records": rows,
        "information_boundary": {
            "fixed_preprocessor_decoded_selected_media": True,
            "model_prediction_computed": False,
            "target_score_computed": False,
            "target_dependent_tuning_performed": False,
        },
    }
    receipt["receipt_id"] = content_id(receipt, "receipt_id")
    write_json(output_root / "processing_receipt.json", receipt)
    return receipt


def nonempty(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink() and path.stat().st_size > 0
    except OSError:
        return False


def small_json_hash(path: Path) -> str | None:
    if not nonempty(path) or path.stat().st_size > 2_000_000:
        return None
    read_json(path)
    return sha256_file(path)


def verify(
    protocol: Mapping[str, Any],
    lock: Mapping[str, Any],
    output_root: Path,
) -> dict[str, object]:
    output_root = output_root.resolve(strict=True)
    episode_records: list[dict[str, object]] = []
    for item in worklist(lock):
        object_id = str(item["object_id"])
        cameras = [str(value) for value in item["cameras"]]
        sensors = [str(value) for value in item["tactile_streams"]]
        for role, episode_index in (
            ("source", int(item["source_episode_index"])),
            ("target", int(item["target_episode_index"])),
        ):
            episode_dir = (
                output_root
                / "aligned"
                / object_id
                / f"episode_{episode_index:04d}"
            )
            episode_files = {
                name: nonempty(episode_dir / name) for name in EPISODE_REQUIRED
            }
            camera_records = []
            for camera in cameras:
                camera_dir = episode_dir / camera
                present = {
                    name: nonempty(camera_dir / name) for name in CAMERA_REQUIRED
                }
                camera_records.append(
                    {
                        "camera": camera,
                        "files_present": present,
                        "ready": all(present.values()),
                        "alignment_sha256": small_json_hash(
                            camera_dir / "alignment.json"
                        ),
                        "metadata_sha256": small_json_hash(
                            camera_dir / "metadata.json"
                        ),
                    }
                )
            tactile_records = []
            for sensor in sensors:
                sensor_dir = episode_dir / sensor
                present = {
                    name: nonempty(sensor_dir / name) for name in TACTILE_REQUIRED
                }
                tactile_records.append(
                    {
                        "sensor": sensor,
                        "files_present": present,
                        "ready": all(present.values()),
                        "alignment_sha256": small_json_hash(
                            sensor_dir / "alignment.json"
                        ),
                        "metadata_sha256": small_json_hash(
                            sensor_dir / "metadata.json"
                        ),
                    }
                )
            robot_dir = episode_dir / "robot"
            robot_files = {
                name: nonempty(robot_dir / name) for name in ROBOT_REQUIRED
            }
            ready = (
                all(episode_files.values())
                and all(row["ready"] is True for row in camera_records)
                and all(row["ready"] is True for row in tactile_records)
                and all(robot_files.values())
            )
            episode_records.append(
                {
                    "object_id": object_id,
                    "role": role,
                    "episode_index": episode_index,
                    "episode_dir": str(episode_dir),
                    "episode_files_present": episode_files,
                    "camera_records": camera_records,
                    "tactile_records": tactile_records,
                    "robot_files_present": robot_files,
                    "robot_metadata_sha256": small_json_hash(
                        robot_dir / "robot.meta.json"
                    ),
                    "ready": ready,
                }
            )
    ready_count = sum(row["ready"] is True for row in episode_records)
    verification: dict[str, object] = {
        "schema": VERIFICATION_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "pair_lock_id": lock["lock_id"],
        "output_root": str(output_root),
        "episode_record_count": len(episode_records),
        "ready_episode_count": ready_count,
        "all_selected_episodes_ready": ready_count == len(episode_records),
        "episode_records": episode_records,
        "information_boundary": {
            "video_payloads_loaded_by_verifier": False,
            "numeric_arrays_loaded_by_verifier": False,
            "model_prediction_computed": False,
            "target_score_computed": False,
            "target_dependent_tuning_performed": False,
            "full_target_media_in_verification_artifact": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    verification["verification_id"] = content_id(
        verification,
        "verification_id",
    )
    return verification


def fixture_pair_lock() -> dict[str, object]:
    pairs = []
    for object_id in EXPECTED_OBJECTS:
        source, target, source_family, target_family = EXPECTED_PAIRS[object_id]
        pairs.append(
            {
                "object_id": object_id,
                "metadata_sha256": "a" * 64,
                "source_episode_index": source,
                "target_episode_index": target,
                "source_action": f"{source_family} edge",
                "target_action": f"{target_family} edge",
                "source_action_family": source_family,
                "target_action_family": target_family,
                "contact_anchor": "single-edge",
                "bimanual": False,
                "nonprehensile": False,
                "camera_policy": "lexicographic-first-12-from-pair-common",
                "cameras": [f"cam-{index:02d}" for index in range(12)],
                "tactile_streams": [f"sensor-{index}" for index in range(4)],
                "source_pair_id": "b" * 64,
            }
        )
    value: dict[str, object] = {
        "schema": LOCK_SCHEMA,
        "schema_version": 1,
        "status": "frozen-after-target-blind-action-binding",
        "dataset": "Deform360",
        "raw_root": "/mnt/lexar4tb/datasets/deform360/data-7fea8e2/raw",
        "source_plan": {
            "run_id": 33337674563,
            "head_sha": "b627a9ba5eb3f6232475267adad24eb31f969123",
            "plan_id": "0c369d36a9db865bfe6ed00bbbb7d708f27042c992d80f293c406374247bd96a",
            "artifact_id": 9739536698,
            "artifact_digest": (
                "sha256:5628aafc473be0908873ea913c5d3b25"
                "b0d52b9189b9e71ff04672a33593862f"
            ),
            "source_roster_run_id": 33335964618,
            "source_roster_result_id": (
                "b73b0e2ee69158d59a1d1189014071a9"
                "dbb7a715de6e88d3edde9156ba999efa"
            ),
            "metadata_probe_run_id": 33337433438,
        },
        "selection": {},
        "pairs": pairs,
        "information_boundary": {
            "media_payload_decoded": False,
            "numeric_arrays_loaded": False,
            "score_bearing_outcomes_used": False,
            "target_future_opened": False,
        },
        "claim_boundary": "fixture",
    }
    value["lock_id"] = content_id(value, "lock_id")
    return value


def fixture_protocol(lock: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": PROTOCOL_SCHEMA,
        "schema_version": 1,
        "protocol_id": "fixture",
        "status": "authorized-retrospective-fixed-pair-preprocessing",
        "pair_lock": {"lock_id": lock["lock_id"]},
        "raw_root": lock["raw_root"],
        "persistent_output_root": "/tmp/output",
        "official_processing": {
            "repository": "lhy0807/deform360",
            "revision": "d8522a4403b766aeb387510c04e89032a56fdf35",
            "package_version": "0.2.0",
        },
        "processing": {
            "camera_count_per_pair": 12,
            "camera_alignment_tolerance_us": 100000,
            "tactile_alignment_tolerance_us": 150000,
            "robot_ransac_seed": 20260831,
        },
        "information_boundary": {
            "retrospective_development": True,
            "fixed_preprocessor_may_decode_selected_source_and_target_media": True,
            "target_dependent_model_selection": False,
            "physical_parameter_inference": False,
            "model_prediction": False,
            "target_scoring": False,
            "paper_claim_authorized": False,
        },
        "claim_boundary": "fixture",
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        lock = fixture_pair_lock()
        protocol = fixture_protocol(lock)
        lock_path = root / "lock.json"
        protocol_path = root / "protocol.json"
        write_json(lock_path, lock)
        write_json(protocol_path, protocol)
        loaded_lock = load_pair_lock(lock_path)
        loaded_protocol = load_protocol(protocol_path, loaded_lock)
        value = manifest(loaded_protocol, loaded_lock, "c" * 40)
        require(len(value["worklist"]) == 4, "fixture worklist changed")
        require(
            value["manifest_id"] == content_id(value, "manifest_id"),
            "bad manifest ID",
        )

        output = root / "processed"
        for item in worklist(loaded_lock):
            object_id = str(item["object_id"])
            for episode_index in (
                int(item["source_episode_index"]),
                int(item["target_episode_index"]),
            ):
                episode = (
                    output
                    / "aligned"
                    / object_id
                    / f"episode_{episode_index:04d}"
                )
                episode.mkdir(parents=True)
                for name in EPISODE_REQUIRED:
                    (episode / name).write_bytes(b"x")
                for camera in item["cameras"]:
                    camera_dir = episode / str(camera)
                    camera_dir.mkdir()
                    for name in CAMERA_REQUIRED:
                        path = camera_dir / name
                        path.write_text(
                            "{}\n" if name.endswith(".json") else "x\n"
                        )
                for sensor in item["tactile_streams"]:
                    sensor_dir = episode / str(sensor)
                    sensor_dir.mkdir()
                    for name in TACTILE_REQUIRED:
                        path = sensor_dir / name
                        path.write_text(
                            "{}\n" if name.endswith(".json") else "x\n"
                        )
                robot = episode / "robot"
                robot.mkdir()
                (robot / "robot.npz").write_bytes(b"x")
                (robot / "robot.meta.json").write_text("{}\n")
        result = verify(loaded_protocol, loaded_lock, output)
        require(
            result["all_selected_episodes_ready"] is True,
            "fixture not ready",
        )
        require(result["ready_episode_count"] == 8, "fixture count changed")
    print("self-test passed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("manifest", "process", "verify"):
        child = subparsers.add_parser(name)
        child.add_argument("--protocol", type=Path, required=True)
        child.add_argument("--pair-lock", type=Path, required=True)
        if name == "manifest":
            child.add_argument("--repository-revision", required=True)
            child.add_argument("--output", type=Path, required=True)
        else:
            child.add_argument("--output-root", type=Path, required=True)
            if name == "verify":
                child.add_argument("--output", type=Path, required=True)

    subparsers.add_parser("self-test")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "self-test":
        self_test()
        return 0
    lock = load_pair_lock(args.pair_lock)
    protocol = load_protocol(args.protocol, lock)
    if args.command == "manifest":
        value = manifest(protocol, lock, args.repository_revision)
        write_json(args.output, value)
        print(
            json.dumps(
                {
                    "manifest_id": value["manifest_id"],
                    "object_count": len(value["worklist"]),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "process":
        value = process(protocol, lock, args.output_root)
        print(
            json.dumps(
                {
                    "episode_count": len(value["episode_records"]),
                    "receipt_id": value["receipt_id"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "verify":
        value = verify(protocol, lock, args.output_root)
        write_json(args.output, value)
        print(
            json.dumps(
                {
                    "all_selected_episodes_ready": value[
                        "all_selected_episodes_ready"
                    ],
                    "ready_episode_count": value["ready_episode_count"],
                    "verification_id": value["verification_id"],
                },
                sort_keys=True,
            )
        )
        return 0 if value["all_selected_episodes_ready"] else 4
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreprocessingError as error:
        print(f"Deform360 preprocessing failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
