"""Exact prospective analysis of the open27 Deform360 raw-camera view budget.

The 2/4-view outcomes did not exist when the accompanying configuration was
frozen.  This analyzer does not build measurements or run the model.  It only
opens complete, checksum-stable outputs from the already existing raw-camera
observation evaluator, verifies that camera count was the sole experimental
factor, and applies the preregistered 4-view decision rule.

The 8-view baseline uses the older raw-camera evaluator.  Its primary arm is
``raw_measurement.recursive_rbf_ungated``.  Raw CPD is intentionally absent:
adding the newer covariance-gated evaluator to only part of this frontier
would change the method rather than isolate the camera budget.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import numpy as np


PROTOCOL_ID = "deform360-open27-raw-camera-budget-frontier-v1-development"
MEASUREMENT_PROTOCOL_ID = "deform360-raw-camera-alltracker-v1-development"
RAW_STREAM = "raw_measurement"
PRIMARY_ARM = "recursive_rbf_ungated"
COMPARATORS = ("physical_prior", "persistence")
PRIMARY_METRICS = (
    "post_update_hidden_identity_rmse_m",
    "post_update_hidden_symmetric_chamfer_m",
)
CAMERA_COUNTS = (2, 4, 8)
CENTER_COUNT = 16
UPDATE_FRAMES = (19, 38, 57)
SHARD_COUNT = 2
PANEL_ROOT = Path(
    "/mnt/corsair/florianpfaff/deform360-dense-reusable-panel-v1/independent-source-v1"
)
PROCESSED_ROOT = Path(
    "/mnt/lexar4tb/datasets/deform360/graph-action-support-independent-source-v1"
)
TRACKER_SOURCE_ROOT = Path("/mnt/corsair/florianpfaff/alltracker-molmomotion-61f5b21")
TRACKER_CHECKPOINT = Path("/mnt/corsair/florianpfaff/model-cache/alltracker.pth")
TRACKER_DEVICE = "cuda:0"


@dataclass(frozen=True)
class TreeInventory:
    """A stable content inventory plus hashes used for later bound reads."""

    root: Path
    file_count: int
    total_file_bytes: int
    inventory_sha256: str
    sha256_by_relative_path: Mapping[str, str]

    def summary(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "file_count": self.file_count,
            "total_file_bytes": self.total_file_bytes,
            "inventory_sha256": self.inventory_sha256,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _stable_file_bytes(path: Path) -> bytes:
    before = path.stat(follow_symlinks=False)
    _require(stat.S_ISREG(before.st_mode), f"inventory entry is not regular: {path}")
    with path.open("rb") as handle:
        payload = handle.read()
    after = path.stat(follow_symlinks=False)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    _require(identity_before == identity_after, f"file changed while hashing: {path}")
    _require(len(payload) == before.st_size, f"short read while hashing: {path}")
    return payload


def inventory_tree(root: str | Path) -> TreeInventory:
    """Reproduce the frozen server inventory algorithm exactly.

    Each regular file contributes
    ``relpath\0stat-percent-a\0stat-percent-s\0sha256hex\0``.  Relative paths
    are sorted by their raw UTF-8 bytes.  Permission mode is lower-case octal
    without a prefix or leading zero, matching ``stat -c %a``.
    """

    given = Path(root)
    _require(given.exists(), f"inventory root does not exist: {given}")
    _require(not given.is_symlink(), f"inventory root is a symlink: {given}")
    resolved = given.resolve(strict=True)
    _require(resolved.is_dir(), f"inventory root is not a directory: {resolved}")
    entries: list[tuple[bytes, str, int, str]] = []
    for path in resolved.rglob("*"):
        _require(not path.is_symlink(), f"symlink is forbidden in inventory: {path}")
        if path.is_dir():
            continue
        status = path.stat(follow_symlinks=False)
        _require(stat.S_ISREG(status.st_mode), f"non-regular inventory entry: {path}")
        relative = path.relative_to(resolved).as_posix()
        try:
            encoded = relative.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise ValueError(f"inventory path is not strict UTF-8: {path}") from error
        _require(b"\0" not in encoded, f"NUL in inventory path: {path}")
        payload = _stable_file_bytes(path)
        after = path.stat(follow_symlinks=False)
        _require(
            (
                status.st_dev,
                status.st_ino,
                status.st_mode,
                status.st_size,
                status.st_mtime_ns,
                status.st_ctime_ns,
            )
            == (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ),
            f"file changed while inventorying: {path}",
        )
        digest = _sha256_bytes(payload)
        mode = format(stat.S_IMODE(after.st_mode), "o")
        entries.append((encoded, mode, len(payload), digest))
    entries.sort(key=lambda record: record[0])
    digest = hashlib.sha256()
    hashes: dict[str, str] = {}
    total_bytes = 0
    for encoded, mode, size, file_sha256 in entries:
        digest.update(encoded)
        digest.update(b"\0")
        digest.update(mode.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha256.encode("ascii"))
        digest.update(b"\0")
        relative = encoded.decode("utf-8")
        hashes[relative] = file_sha256
        total_bytes += size
    return TreeInventory(
        root=resolved,
        file_count=len(entries),
        total_file_bytes=total_bytes,
        inventory_sha256=digest.hexdigest(),
        sha256_by_relative_path=hashes,
    )


def _bound_bytes(inventory: TreeInventory, relative_path: str) -> bytes:
    _require(
        relative_path in inventory.sha256_by_relative_path,
        f"file is absent from bound inventory: {relative_path}",
    )
    path = inventory.root / relative_path
    payload = _stable_file_bytes(path)
    _require(
        _sha256_bytes(payload) == inventory.sha256_by_relative_path[relative_path],
        f"file changed after inventory: {path}",
    )
    return payload


def _bound_json(inventory: TreeInventory, relative_path: str) -> dict[str, Any]:
    try:
        value = json.loads(_bound_bytes(inventory, relative_path))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {relative_path}") from error
    _require(
        isinstance(value, dict), f"JSON artifact is not an object: {relative_path}"
    )
    return value


def _validate_self_hash(value: Mapping[str, Any], *, label: str) -> None:
    claimed = value.get("result_sha256")
    unsigned = dict(value)
    unsigned.pop("result_sha256", None)
    _require(
        isinstance(claimed, str) and claimed == _canonical_sha256(unsigned),
        f"{label} canonical result hash changed",
    )


def _expected_case_metadata(
    config: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, str], dict[str, int]]:
    panel = config["panel"]
    object_episodes = panel["object_episodes"]
    _require(isinstance(object_episodes, dict), "panel object_episodes is invalid")
    cases: list[str] = []
    objects: dict[str, str] = {}
    episodes: dict[str, int] = {}
    for object_id, episode_values in object_episodes.items():
        _require(
            isinstance(object_id, str) and object_id,
            "panel has an invalid object ID",
        )
        _require(
            isinstance(episode_values, list)
            and episode_values
            and all(type(value) is int and value >= 0 for value in episode_values),
            f"panel episodes are invalid for {object_id}",
        )
        _require(
            episode_values == sorted(set(episode_values)),
            f"panel episodes are not unique and sorted for {object_id}",
        )
        for episode_id in episode_values:
            case = f"{object_id}-ep{episode_id:04d}"
            cases.append(case)
            objects[case] = object_id
            episodes[case] = episode_id
    expected = (
        "002-rope-silk-ep0002",
        "002-rope-silk-ep0005",
        "002-rope-silk-ep0006",
        "002-rope-silk-ep0007",
        "002-rope-silk-ep0009",
        "083-blanket-cloth-ep0001",
        "083-blanket-cloth-ep0002",
        "083-blanket-cloth-ep0004",
        "083-blanket-cloth-ep0005",
        "083-blanket-cloth-ep0008",
        "083-blanket-cloth-ep0009",
        "085-scarf-cloth-ep0003",
        "085-scarf-cloth-ep0004",
        "085-scarf-cloth-ep0006",
        "085-scarf-cloth-ep0008",
        "085-scarf-cloth-ep0009",
        "092-squirrel-ep0004",
        "092-squirrel-ep0005",
        "092-squirrel-ep0007",
        "092-squirrel-ep0008",
        "092-squirrel-ep0009",
        "170-spider-ep0000",
        "170-spider-ep0001",
        "170-spider-ep0003",
        "170-spider-ep0005",
        "170-spider-ep0008",
        "170-spider-ep0009",
    )
    _require(tuple(cases) == expected, "panel is not exactly the frozen open27")
    _require(panel.get("episode_count") == 27, "panel episode count changed")
    _require(panel.get("physical_object_count") == 5, "object count changed")
    return expected, objects, episodes


def _validate_config(
    config: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, str], dict[str, int]]:
    _require(config.get("schema_version") == 1, "frontier config schema changed")
    _require(config.get("protocol_id") == PROTOCOL_ID, "frontier protocol ID changed")
    _require(
        config.get("status")
        == "prospectively frozen before any 2-view or 4-view outcome was produced or inspected",
        "frontier prospective status changed",
    )
    freeze = config.get("freeze", {})
    _require(
        freeze.get("camera_count_is_only_experimental_factor") is True,
        "camera count is not frozen as the sole factor",
    )
    outcome_status = freeze.get("outcome_status_at_freeze", {})
    _require(
        outcome_status.get("2") == "not produced or inspected"
        and outcome_status.get("4") == "not produced or inspected",
        "prospective 2/4-view outcome status changed",
    )
    cases, objects, episodes = _expected_case_metadata(config)
    observation = config.get("observation", {})
    _require(
        observation.get("camera_counts") == list(CAMERA_COUNTS),
        "camera budgets changed",
    )
    _require(observation.get("center_count") == CENTER_COUNT, "center count changed")
    _require(
        observation.get("require_exact_center_id_equality_across_budgets") is True,
        "center equality is no longer required",
    )
    _require(
        observation.get("update_frames") == list(UPDATE_FRAMES),
        "update frames changed",
    )
    budget_semantics = observation.get("budget_semantics", {})
    _require(
        budget_semantics
        == {
            "all_available_cameras_used_for_frame_zero_planning": True,
            "tracked_views_after_planning_are_budgeted": True,
            "full_sensor_count_ablation": False,
        },
        "tracked-view budget semantics changed",
    )
    _require(
        observation.get("source_roots")
        == {
            "panel": str(PANEL_ROOT),
            "processed": str(PROCESSED_ROOT),
        },
        "input source roots changed",
    )
    raw_config = observation.get("raw_camera_config_except_selected_camera_count")
    _require(isinstance(raw_config, dict), "raw-camera config is missing")
    _require(
        raw_config.get("alltracker_max_side") == 512,
        "AllTracker maximum side changed",
    )
    _require(raw_config.get("center_count") == CENTER_COUNT, "raw center count changed")
    _require(
        raw_config.get("update_frames") == list(UPDATE_FRAMES),
        "raw update frames changed",
    )
    tracker = observation.get("tracker", {})
    _require(
        tracker.get("source_root") == str(TRACKER_SOURCE_ROOT)
        and tracker.get("checkpoint") == str(TRACKER_CHECKPOINT)
        and tracker.get("device") == TRACKER_DEVICE,
        "tracker execution binding changed",
    )
    evaluation = config.get("evaluation", {})
    _require(
        evaluation.get("protocol_id") == MEASUREMENT_PROTOCOL_ID,
        "evaluation protocol changed",
    )
    _require(evaluation.get("stream") == RAW_STREAM, "evaluation stream changed")
    _require(evaluation.get("primary_arm") == PRIMARY_ARM, "primary arm changed")
    _require(
        evaluation.get("comparators") == list(COMPARATORS),
        "comparators changed",
    )
    _require(
        evaluation.get("primary_metrics") == list(PRIMARY_METRICS),
        "primary metrics changed",
    )
    raw_cpd = evaluation.get("raw_cpd", {})
    _require(raw_cpd.get("included") is False, "raw CPD inclusion changed the method")
    roots = config.get("roots", {})
    _require(
        all(str(value) in roots for value in CAMERA_COUNTS)
        and "frontier_output" in roots,
        "frontier roots are incomplete",
    )
    for camera_count in CAMERA_COUNTS:
        role = roots[str(camera_count)]
        _require(
            isinstance(role, dict)
            and isinstance(role.get("measurement"), str)
            and isinstance(role.get("evaluation"), str),
            f"roots are invalid for camera count {camera_count}",
        )
    decision = config.get("decision", {})
    _require(
        decision.get("candidate_camera_count") == 4
        and decision.get("reference_camera_count") == 8
        and decision.get("descriptive_camera_counts") == [2],
        "camera decision roles changed",
    )
    gates = decision.get("go_if_all", {})
    _require(
        gates.get(
            "minimum_fraction_of_8_view_relative_improvement_retained_each_primary_metric"
        )
        == 0.8
        and gates.get("minimum_joint_case_wins_vs_physical") == 18
        and gates.get(
            "all_five_object_mean_differences_vs_physical_improve_on_both_primary_metrics"
        )
        is True
        and gates.get("maximum_case_chamfer_relative_regression_vs_physical") == 0.1,
        "frontier decision thresholds changed",
    )
    return cases, objects, episodes


def _expected_measurement_paths(cases: Sequence[str]) -> set[str]:
    paths = {
        relative
        for case in cases
        for relative in (
            f"{case}/measurement.npz",
            f"{case}/measurement_manifest.json",
        )
    }
    paths.update(f"build-shard-{index:02d}.json" for index in range(SHARD_COUNT))
    return paths


def _validate_measurement_root(
    inventory: TreeInventory,
    config: Mapping[str, Any],
    camera_count: int,
    cases: Sequence[str],
    objects: Mapping[str, str],
    episodes: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    _require(
        set(inventory.sha256_by_relative_path) == _expected_measurement_paths(cases),
        f"{camera_count}-view measurement file layout changed",
    )
    shard_cases: list[str] = []
    for shard_index in range(SHARD_COUNT):
        shard = _bound_json(inventory, f"build-shard-{shard_index:02d}.json")
        _require(
            shard.get("protocol_id") == MEASUREMENT_PROTOCOL_ID
            and shard.get("shard_count") == SHARD_COUNT
            and shard.get("shard_index") == shard_index,
            f"{camera_count}-view shard contract changed",
        )
        expected_shard_cases = list(cases[shard_index::SHARD_COUNT])
        _require(
            shard.get("cases") == expected_shard_cases
            and shard.get("case_count") == len(expected_shard_cases),
            f"{camera_count}-view shard case partition changed",
        )
        manifest_hashes = shard.get("measurement_manifest_sha256")
        _require(
            isinstance(manifest_hashes, dict)
            and set(manifest_hashes) == set(expected_shard_cases),
            f"{camera_count}-view shard manifest inventory changed",
        )
        for case in expected_shard_cases:
            _require(
                manifest_hashes[case]
                == inventory.sha256_by_relative_path[
                    f"{case}/measurement_manifest.json"
                ],
                f"{camera_count}-view shard hash differs for {case}",
            )
        shard_cases.extend(expected_shard_cases)
    _require(
        set(shard_cases) == set(cases) and len(shard_cases) == len(cases),
        f"{camera_count}-view shard coverage changed",
    )

    expected_config = dict(
        config["observation"]["raw_camera_config_except_selected_camera_count"]
    )
    expected_config["selected_camera_count"] = camera_count
    expected_tracker = config["observation"]["tracker"]
    manifests: dict[str, dict[str, Any]] = {}
    for case in cases:
        relative_manifest = f"{case}/measurement_manifest.json"
        relative_archive = f"{case}/measurement.npz"
        manifest = _bound_json(inventory, relative_manifest)
        _validate_self_hash(manifest, label=f"{camera_count}-view {case} measurement")
        _require(
            manifest.get("schema_version") == 1
            and manifest.get("artifact_kind") == "Deform360CausalRawCameraMeasurement"
            and manifest.get("protocol_id") == MEASUREMENT_PROTOCOL_ID,
            f"{camera_count}-view measurement schema changed for {case}",
        )
        _require(
            manifest.get("case") == case
            and manifest.get("object_id") == objects[case]
            and manifest.get("episode_id") == episodes[case],
            f"{camera_count}-view measurement identity changed for {case}",
        )
        _require(
            manifest.get("config") == expected_config,
            f"{camera_count}-view non-camera measurement config changed for {case}",
        )
        plan = manifest.get("plan", {})
        center_ids = plan.get("center_ids")
        selected_cameras = plan.get("selected_cameras")
        _require(
            isinstance(center_ids, list)
            and len(center_ids) == CENTER_COUNT
            and all(type(value) is int and value >= 0 for value in center_ids)
            and len(set(center_ids)) == CENTER_COUNT,
            f"{camera_count}-view center IDs are invalid for {case}",
        )
        _require(
            isinstance(selected_cameras, list)
            and len(selected_cameras) == camera_count
            and len(set(selected_cameras)) == camera_count
            and all(isinstance(value, str) and value for value in selected_cameras),
            f"{camera_count}-view selected cameras are invalid for {case}",
        )
        tracker = manifest.get("tracker", {})
        _require(
            tracker.get("name") == expected_tracker["name"]
            and tracker.get("molmomotion_revision")
            == expected_tracker["molmomotion_revision"]
            and tracker.get("source_tree") == expected_tracker["source_tree"]
            and tracker.get("runtime_source_sha256")
            == expected_tracker["runtime_source_sha256"]
            and tracker.get("checkpoint_sha256")
            == expected_tracker["checkpoint_sha256"],
            f"{camera_count}-view tracker binding changed for {case}",
        )
        _require(
            tracker == expected_tracker,
            f"{camera_count}-view tracker path or device changed for {case}",
        )
        inputs = manifest.get("inputs")
        expected_input_paths = {
            "prediction_seal": PANEL_ROOT / case / "prediction_seal.json",
            "prediction_archive": PANEL_ROOT / case / "sealed_prediction.npz",
            "intrinsics": (
                PROCESSED_ROOT / case / "episode_0000" / "undistorted_intrinsics.npy"
            ),
            "extrinsics": (PROCESSED_ROOT / case / "episode_0000" / "extrinsics.npy"),
        }
        _require(
            isinstance(inputs, dict) and set(inputs) == set(expected_input_paths),
            f"{camera_count}-view immutable inputs changed for {case}",
        )
        for input_name, expected_path in expected_input_paths.items():
            input_record = inputs[input_name]
            _require(
                isinstance(input_record, dict)
                and input_record.get("path") == str(expected_path)
                and _is_sha256(input_record.get("sha256")),
                f"{camera_count}-view {input_name} binding changed for {case}",
            )
        candidate_ids = plan.get("candidate_ids")
        _require(
            isinstance(candidate_ids, list)
            and candidate_ids == sorted(set(candidate_ids))
            and all(type(value) is int and value >= 0 for value in candidate_ids)
            and plan.get("candidate_count") == len(candidate_ids)
            and set(center_ids).issubset(candidate_ids)
            and plan.get("selection_inputs")
            == "sealed frame-zero points, calibration, and HDF5 index zero only",
            f"{camera_count}-view all-camera planning contract changed for {case}",
        )
        selected_camera_inputs = manifest.get("selected_camera_inputs")
        _require(
            isinstance(selected_camera_inputs, dict)
            and list(selected_camera_inputs) == selected_cameras,
            f"{camera_count}-view selected-camera input order changed for {case}",
        )
        expected_camera_root = PROCESSED_ROOT / case / "episode_0000"
        for camera in selected_cameras:
            _require(
                "/" not in camera and camera not in (".", ".."),
                f"{camera_count}-view camera name is unsafe for {case}",
            )
            camera_inputs = selected_camera_inputs[camera]
            _require(
                isinstance(camera_inputs, dict)
                and set(camera_inputs)
                == {"video", "frame_zero_mask", "frame_zero_depth"},
                f"{camera_count}-view camera inputs changed for {case}/{camera}",
            )
            expected_camera_paths = {
                "video": expected_camera_root / camera / "undistorted.mp4",
                "frame_zero_mask": expected_camera_root / camera / "mask_refined.h5",
                "frame_zero_depth": (
                    expected_camera_root / camera / "rendered_depth.h5"
                ),
            }
            video = camera_inputs["video"]
            _require(
                isinstance(video, dict)
                and video.get("path") == str(expected_camera_paths["video"])
                and video.get("whole_file_hashed_or_read") is False
                and set(video.get("decoded_prefix_sha256_by_update", {}))
                == {str(frame) for frame in UPDATE_FRAMES}
                and all(
                    _is_sha256(value)
                    for value in video["decoded_prefix_sha256_by_update"].values()
                ),
                f"{camera_count}-view video prefix binding changed for {case}/{camera}",
            )
            for input_name in ("frame_zero_mask", "frame_zero_depth"):
                record = camera_inputs[input_name]
                _require(
                    isinstance(record, dict)
                    and record.get("path") == str(expected_camera_paths[input_name])
                    and record.get("only_index_read") == 0
                    and record.get("whole_file_hashed_or_read") is False
                    and _is_sha256(record.get("frame_zero_array_sha256")),
                    f"{camera_count}-view {input_name} binding changed for "
                    f"{case}/{camera}",
                )
        boundary = manifest.get("information_boundary", {})
        _require(
            boundary.get("target_data_read") is False
            and boundary.get("outcome_manifest_read") is False
            and boundary.get("future_reconstruction_after_frame_zero_read") is False
            and boundary.get("maximum_video_frame_read_by_update")
            == list(UPDATE_FRAMES)
            and boundary.get("frame_zero_hdf5_indices_read") == [0],
            f"{camera_count}-view measurement crossed its boundary for {case}",
        )
        _require(
            manifest.get("output", {}).get("measurement_archive_sha256")
            == inventory.sha256_by_relative_path[relative_archive],
            f"{camera_count}-view measurement archive hash changed for {case}",
        )
        archive_payload = _bound_bytes(inventory, relative_archive)
        with np.load(io.BytesIO(archive_payload), allow_pickle=False) as stored:
            _require(
                {"center_ids", "selected_cameras", "update_frames"}.issubset(
                    stored.files
                ),
                f"{camera_count}-view measurement arrays are incomplete for {case}",
            )
            archive_centers = np.asarray(stored["center_ids"])
            archive_cameras = np.asarray(stored["selected_cameras"])
            archive_updates = np.asarray(stored["update_frames"])
            archive_candidates = np.asarray(stored["candidate_ids"])
        _require(
            archive_centers.dtype.kind in "iu"
            and archive_centers.tolist() == center_ids,
            f"{camera_count}-view archive center IDs changed for {case}",
        )
        _require(
            archive_cameras.dtype.kind == "U"
            and archive_cameras.tolist() == selected_cameras,
            f"{camera_count}-view archive camera IDs changed for {case}",
        )
        _require(
            archive_updates.dtype.kind in "iu"
            and archive_updates.tolist() == list(UPDATE_FRAMES),
            f"{camera_count}-view archive updates changed for {case}",
        )
        _require(
            archive_candidates.dtype.kind in "iu"
            and archive_candidates.tolist() == candidate_ids,
            f"{camera_count}-view archive candidate IDs changed for {case}",
        )
        updates = manifest.get("updates")
        _require(
            isinstance(updates, list)
            and [record.get("frame") for record in updates] == list(UPDATE_FRAMES),
            f"{camera_count}-view measurement updates changed for {case}",
        )
        for update in updates:
            frame = int(update["frame"])
            _require(
                update.get("maximum_video_frame_read") == frame
                and update.get("prefix_frame_range_half_open") == [0, frame + 1],
                f"{camera_count}-view causal update changed for {case}/{frame}",
            )
            update_trackers = update.get("tracker")
            _require(
                isinstance(update_trackers, list)
                and [record.get("camera") for record in update_trackers]
                == selected_cameras,
                f"{camera_count}-view update camera order changed for {case}/{frame}",
            )
            for tracker_record in update_trackers:
                camera = tracker_record["camera"]
                _require(
                    tracker_record.get("maximum_video_frame_read") == frame
                    and tracker_record.get("prefix_frame_range_half_open")
                    == [0, frame + 1]
                    and tracker_record.get("decoded_rgb_prefix_sha256")
                    == selected_camera_inputs[camera]["video"][
                        "decoded_prefix_sha256_by_update"
                    ][str(frame)],
                    f"{camera_count}-view decoded prefix changed for "
                    f"{case}/{camera}/{frame}",
                )
        manifests[case] = manifest
    return manifests


def _expected_evaluation_paths(cases: Sequence[str]) -> set[str]:
    return {
        "summary.json",
        *(relative for case in cases for relative in (f"{case}.json", f"{case}.npz")),
    }


def _metric(value: Any, *, label: str) -> float:
    _require(
        type(value) in (int, float) and math.isfinite(float(value)),
        f"{label} is not finite",
    )
    result = float(value)
    _require(result >= 0.0, f"{label} is negative")
    return result


def _validate_evaluation_root(
    inventory: TreeInventory,
    measurement_inventory: TreeInventory,
    measurement_manifests: Mapping[str, Mapping[str, Any]],
    camera_count: int,
    cases: Sequence[str],
    objects: Mapping[str, str],
    episodes: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    _require(
        set(inventory.sha256_by_relative_path) == _expected_evaluation_paths(cases),
        f"{camera_count}-view evaluation file layout changed",
    )
    summary = _bound_json(inventory, "summary.json")
    _validate_self_hash(summary, label=f"{camera_count}-view evaluation summary")
    _require(
        summary.get("schema_version") == 1
        and summary.get("protocol_id") == MEASUREMENT_PROTOCOL_ID
        and summary.get("episode_count") == 27
        and summary.get("physical_object_count") == 5,
        f"{camera_count}-view evaluation summary contract changed",
    )
    artifact_records = summary.get("artifacts")
    _require(
        isinstance(artifact_records, list)
        and [record.get("case") for record in artifact_records] == list(cases),
        f"{camera_count}-view evaluation artifact order changed",
    )
    artifact_by_case = {record["case"]: record for record in artifact_records}
    reports: dict[str, dict[str, Any]] = {}
    for case in cases:
        report_relative = f"{case}.json"
        arrays_relative = f"{case}.npz"
        artifact = artifact_by_case[case]
        _require(
            artifact.get("report_sha256")
            == inventory.sha256_by_relative_path[report_relative]
            and artifact.get("arrays_sha256")
            == inventory.sha256_by_relative_path[arrays_relative],
            f"{camera_count}-view evaluation artifact hash changed for {case}",
        )
        report = _bound_json(inventory, report_relative)
        _require(
            report.get("protocol_id") == MEASUREMENT_PROTOCOL_ID
            and report.get("case") == case
            and report.get("object_id") == objects[case]
            and report.get("episode_id") == episodes[case],
            f"{camera_count}-view evaluation identity changed for {case}",
        )
        _require(
            report.get("measurement_manifest_sha256")
            == measurement_inventory.sha256_by_relative_path[
                f"{case}/measurement_manifest.json"
            ]
            and report.get("measurement_archive_sha256")
            == measurement_inventory.sha256_by_relative_path[f"{case}/measurement.npz"]
            and report.get("measurement_result_sha256")
            == measurement_manifests[case]["result_sha256"],
            f"{camera_count}-view evaluation binds different measurements for {case}",
        )
        boundary = report.get("information_boundary", {})
        _require(
            boundary.get("measurement_hashed_before_target_open_in_this_evaluator")
            is True
            and boundary.get("measurement_builder_target_read") is False,
            f"{camera_count}-view evaluator boundary changed for {case}",
        )
        stream = report.get(RAW_STREAM)
        _require(
            isinstance(stream, dict)
            and stream.get("center_count") == CENTER_COUNT
            and stream.get("center_ids")
            == measurement_manifests[case]["plan"]["center_ids"]
            and stream.get("update_frames") == list(UPDATE_FRAMES),
            f"{camera_count}-view raw evaluation contract changed for {case}",
        )
        updates = stream.get("updates")
        _require(
            isinstance(updates, list)
            and [record.get("frame") for record in updates] == list(UPDATE_FRAMES),
            f"{camera_count}-view raw update records changed for {case}",
        )
        scores = stream.get("scores")
        _require(
            isinstance(scores, dict)
            and all(arm in scores for arm in (PRIMARY_ARM, *COMPARATORS)),
            f"{camera_count}-view score arms changed for {case}",
        )
        for arm in (PRIMARY_ARM, *COMPARATORS):
            _require(
                isinstance(scores[arm], dict),
                f"{camera_count}-view score is invalid for {case}/{arm}",
            )
            for metric_name in PRIMARY_METRICS:
                _metric(
                    scores[arm].get(metric_name),
                    label=f"{camera_count}-view {case}/{arm}/{metric_name}",
                )
        reports[case] = report

    aggregate = summary.get("aggregate", {}).get(RAW_STREAM, {})
    _require(isinstance(aggregate, dict), f"{camera_count}-view aggregate is missing")
    for arm in (PRIMARY_ARM, *COMPARATORS):
        for metric_name in PRIMARY_METRICS:
            recomputed = float(
                np.mean(
                    [
                        reports[case][RAW_STREAM]["scores"][arm][metric_name]
                        for case in cases
                    ]
                )
            )
            reported = _metric(
                aggregate.get(arm, {}).get(metric_name),
                label=f"{camera_count}-view aggregate/{arm}/{metric_name}",
            )
            _require(
                reported == recomputed,
                f"{camera_count}-view aggregate is not exactly reproducible for "
                f"{arm}/{metric_name}",
            )
    return summary, reports


def _validate_bound_8_view(
    config: Mapping[str, Any],
    measurement: TreeInventory,
    evaluation: TreeInventory,
) -> None:
    bound = config.get("bound_existing_8_view_baseline", {})
    for label, observed, expected in (
        ("measurement", measurement, bound.get("measurement", {})),
        ("evaluation", evaluation, bound.get("evaluation", {})),
    ):
        _require(
            observed.file_count == expected.get("file_count")
            and observed.total_file_bytes == expected.get("total_file_bytes")
            and observed.inventory_sha256 == expected.get("inventory_sha256"),
            f"bound 8-view {label} inventory changed",
        )
    _require(
        evaluation.sha256_by_relative_path.get("summary.json")
        == bound.get("evaluation", {}).get("summary_sha256"),
        "bound 8-view summary hash changed",
    )


def _score(
    reports: Mapping[str, Mapping[str, Any]],
    case: str,
    arm: str,
    metric_name: str,
) -> float:
    return float(reports[case][RAW_STREAM]["scores"][arm][metric_name])


def _budget_analysis(
    camera_count: int,
    reports: Mapping[str, Mapping[str, Any]],
    cases: Sequence[str],
    objects: Mapping[str, str],
) -> dict[str, Any]:
    object_ids = tuple(sorted(set(objects.values())))
    aggregate_scores = {
        arm: {
            metric_name: float(
                np.mean([_score(reports, case, arm, metric_name) for case in cases])
            )
            for metric_name in PRIMARY_METRICS
        }
        for arm in (PRIMARY_ARM, *COMPARATORS)
    }
    comparisons: dict[str, Any] = {}
    for comparator in COMPARATORS:
        metric_comparisons: dict[str, Any] = {}
        for metric_name in PRIMARY_METRICS:
            candidate = {
                case: _score(reports, case, PRIMARY_ARM, metric_name) for case in cases
            }
            baseline = {
                case: _score(reports, case, comparator, metric_name) for case in cases
            }
            differences = {case: candidate[case] - baseline[case] for case in cases}
            per_object: dict[str, Any] = {}
            for object_id in object_ids:
                object_cases = [case for case in cases if objects[case] == object_id]
                candidate_mean = float(
                    np.mean([candidate[case] for case in object_cases])
                )
                baseline_mean = float(
                    np.mean([baseline[case] for case in object_cases])
                )
                per_object[object_id] = {
                    "case_count": len(object_cases),
                    "candidate_mean_m": candidate_mean,
                    "comparator_mean_m": baseline_mean,
                    "mean_difference_m": candidate_mean - baseline_mean,
                    "relative_change": (
                        None
                        if baseline_mean == 0.0
                        else candidate_mean / baseline_mean - 1.0
                    ),
                    "case_wins": int(
                        sum(differences[case] < 0.0 for case in object_cases)
                    ),
                }
            candidate_mean = float(np.mean(list(candidate.values())))
            baseline_mean = float(np.mean(list(baseline.values())))
            object_candidate = float(
                np.mean([record["candidate_mean_m"] for record in per_object.values()])
            )
            object_baseline = float(
                np.mean([record["comparator_mean_m"] for record in per_object.values()])
            )
            metric_comparisons[metric_name] = {
                "candidate_equal_case_mean_m": candidate_mean,
                "comparator_equal_case_mean_m": baseline_mean,
                "equal_case_mean_difference_m": candidate_mean - baseline_mean,
                "relative_change": (
                    None
                    if baseline_mean == 0.0
                    else candidate_mean / baseline_mean - 1.0
                ),
                "relative_improvement": (
                    None
                    if baseline_mean == 0.0
                    else 1.0 - candidate_mean / baseline_mean
                ),
                "case_wins": int(sum(value < 0.0 for value in differences.values())),
                "object_balanced_candidate_mean_m": object_candidate,
                "object_balanced_comparator_mean_m": object_baseline,
                "object_balanced_mean_difference_m": (
                    object_candidate - object_baseline
                ),
                "object_balanced_relative_change": (
                    None
                    if object_baseline == 0.0
                    else object_candidate / object_baseline - 1.0
                ),
                "per_object": per_object,
                "per_case_difference_m": differences,
            }
        comparisons[comparator] = {
            "metrics": metric_comparisons,
            "joint_case_wins": int(
                sum(
                    all(
                        _score(reports, case, PRIMARY_ARM, metric_name)
                        < _score(reports, case, comparator, metric_name)
                        for metric_name in PRIMARY_METRICS
                    )
                    for case in cases
                )
            ),
        }
    return {
        "camera_count": camera_count,
        "role": "descriptive_only"
        if camera_count == 2
        else "candidate"
        if camera_count == 4
        else "reference",
        "aggregate_scores": aggregate_scores,
        "comparisons": comparisons,
    }


def _exact_cross_budget_checks(
    manifests_by_budget: Mapping[int, Mapping[str, Mapping[str, Any]]],
    reports_by_budget: Mapping[int, Mapping[str, Mapping[str, Any]]],
    cases: Sequence[str],
) -> dict[str, Any]:
    centers_by_case: dict[str, list[int]] = {}
    for case in cases:
        centers = [
            manifests_by_budget[camera_count][case]["plan"]["center_ids"]
            for camera_count in CAMERA_COUNTS
        ]
        _require(
            centers[0] == centers[1] == centers[2],
            f"center IDs differ across camera budgets for {case}",
        )
        centers_by_case[case] = centers[0]
        immutable_inputs = [
            manifests_by_budget[camera_count][case]["inputs"]
            for camera_count in CAMERA_COUNTS
        ]
        _require(
            immutable_inputs[0] == immutable_inputs[1] == immutable_inputs[2],
            f"immutable inputs differ across camera budgets for {case}",
        )
        planning_records = [
            {
                key: manifests_by_budget[camera_count][case]["plan"][key]
                for key in ("candidate_count", "candidate_ids", "selection_inputs")
            }
            for camera_count in CAMERA_COUNTS
        ]
        _require(
            planning_records[0] == planning_records[1] == planning_records[2],
            f"all-camera planning inputs differ across camera budgets for {case}",
        )
        tracker_bindings = [
            manifests_by_budget[camera_count][case]["tracker"]
            for camera_count in CAMERA_COUNTS
        ]
        _require(
            tracker_bindings[0] == tracker_bindings[1] == tracker_bindings[2],
            f"tracker execution differs across camera budgets for {case}",
        )
        camera_inputs_by_budget = {
            camera_count: manifests_by_budget[camera_count][case][
                "selected_camera_inputs"
            ]
            for camera_count in CAMERA_COUNTS
        }
        camera_budgets: dict[str, list[int]] = {}
        for camera_count in CAMERA_COUNTS:
            for camera in camera_inputs_by_budget[camera_count]:
                camera_budgets.setdefault(camera, []).append(camera_count)
        for camera, budgets in camera_budgets.items():
            if len(budgets) < 2:
                continue
            records = [
                camera_inputs_by_budget[camera_count][camera]
                for camera_count in budgets
            ]
            _require(
                all(record == records[0] for record in records[1:]),
                f"camera input bytes differ across budgets for {case}/{camera}",
            )
        reference_stream = reports_by_budget[8][case][RAW_STREAM]
        for camera_count in (2, 4):
            stream = reports_by_budget[camera_count][case][RAW_STREAM]
            _require(
                stream.get("belief_config") == reference_stream.get("belief_config")
                and stream.get("metric_contract")
                == reference_stream.get("metric_contract")
                and stream.get("scored_frames")
                == reference_stream.get("scored_frames"),
                f"evaluation method changed at {camera_count} views for {case}",
            )
        for arm in COMPARATORS:
            for metric_name in PRIMARY_METRICS:
                values = [
                    _score(
                        reports_by_budget[camera_count],
                        case,
                        arm,
                        metric_name,
                    )
                    for camera_count in CAMERA_COUNTS
                ]
                _require(
                    values[0] == values[1] == values[2],
                    f"{arm} changed across camera budgets for {case}/{metric_name}",
                )
    return {
        "exact_center_id_equality_across_budgets": True,
        "exact_immutable_input_hash_equality_across_budgets": True,
        "exact_all_camera_candidate_id_equality_across_budgets": True,
        "exact_tracker_path_device_and_hash_equality_across_budgets": True,
        "exact_overlapping_camera_input_hash_equality_across_budgets": True,
        "camera_invariant_physical_and_persistence_scores": True,
        "camera_invariant_belief_metric_and_scored_frame_contracts": True,
        "centers_by_case": centers_by_case,
    }


def _four_view_decision(
    analyses: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    physical_4 = analyses[4]["comparisons"]["physical_prior"]
    physical_8 = analyses[8]["comparisons"]["physical_prior"]
    retention_checks: dict[str, Any] = {}
    for metric_name in PRIMARY_METRICS:
        improvement_4 = physical_4["metrics"][metric_name]["relative_improvement"]
        improvement_8 = physical_8["metrics"][metric_name]["relative_improvement"]
        valid_reference = (
            improvement_8 is not None
            and improvement_8 > 0.0
            and improvement_4 is not None
        )
        fraction = None if not valid_reference else float(improvement_4 / improvement_8)
        retention_checks[metric_name] = {
            "four_view_relative_improvement": improvement_4,
            "eight_view_relative_improvement": improvement_8,
            "fraction_retained": fraction,
            "minimum": 0.8,
            "passed": bool(fraction is not None and fraction >= 0.8),
        }
    joint_wins = int(physical_4["joint_case_wins"])
    joint_check = {
        "observed": joint_wins,
        "minimum": 18,
        "passed": joint_wins >= 18,
    }
    object_checks: dict[str, Any] = {}
    object_ids = tuple(physical_4["metrics"][PRIMARY_METRICS[0]]["per_object"])
    for object_id in object_ids:
        differences = {
            metric_name: physical_4["metrics"][metric_name]["per_object"][object_id][
                "mean_difference_m"
            ]
            for metric_name in PRIMARY_METRICS
        }
        object_checks[object_id] = {
            "mean_difference_m": differences,
            "passed": all(value < 0.0 for value in differences.values()),
        }
    _require(len(object_checks) == 5, "object-level gate no longer has five objects")
    chamfer = PRIMARY_METRICS[1]
    chamfer_differences = physical_4["metrics"][chamfer]["per_case_difference_m"]
    relative_regressions = physical_4["metrics"][chamfer]["per_case_relative_change"]
    maximum_regression = max(relative_regressions.values())
    regression_check = {
        "maximum_observed_relative_regression": maximum_regression,
        "maximum_allowed": 0.1,
        "inclusive_bound": True,
        "passed": maximum_regression <= 0.1,
        "per_case_relative_change": relative_regressions,
        "per_case_difference_m": chamfer_differences,
    }
    checks = [
        *(record["passed"] for record in retention_checks.values()),
        joint_check["passed"],
        *(record["passed"] for record in object_checks.values()),
        regression_check["passed"],
    ]
    passed = all(checks)
    return {
        "camera_count": 4,
        "reference_camera_count": 8,
        "status": "GO" if passed else "NO_GO",
        "passed": passed,
        "retains_at_least_80_percent_of_8_view_relative_improvement": retention_checks,
        "joint_case_wins_vs_physical": joint_check,
        "all_five_objects_improve_on_both_primary_metrics": {
            "passed": all(record["passed"] for record in object_checks.values()),
            "objects": object_checks,
        },
        "case_chamfer_regression_vs_physical": regression_check,
        "two_view_role": "descriptive_only; never enters this decision",
        "tie_policy": "ties are not improvements or wins; 10% regression passes",
    }


def _add_case_relative_changes(
    analysis: dict[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
    cases: Sequence[str],
) -> None:
    for comparator in COMPARATORS:
        for metric_name in PRIMARY_METRICS:
            values: dict[str, float] = {}
            for case in cases:
                candidate = _score(reports, case, PRIMARY_ARM, metric_name)
                baseline = _score(reports, case, comparator, metric_name)
                _require(
                    baseline > 0.0,
                    f"zero comparator prevents relative change for {case}/{metric_name}",
                )
                values[case] = candidate / baseline - 1.0
            analysis["comparisons"][comparator]["metrics"][metric_name][
                "per_case_relative_change"
            ] = values


def build_frontier_report(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate complete roots and construct the preregistered result in memory."""

    cases, objects, episodes = _validate_config(config)
    roots = config["roots"]
    measurement_inventories: dict[int, TreeInventory] = {}
    evaluation_inventories: dict[int, TreeInventory] = {}
    manifests_by_budget: dict[int, dict[str, dict[str, Any]]] = {}
    reports_by_budget: dict[int, dict[str, dict[str, Any]]] = {}
    evaluation_summaries: dict[int, dict[str, Any]] = {}
    for camera_count in CAMERA_COUNTS:
        role = roots[str(camera_count)]
        measurement_inventory = inventory_tree(role["measurement"])
        evaluation_inventory = inventory_tree(role["evaluation"])
        measurement_inventories[camera_count] = measurement_inventory
        evaluation_inventories[camera_count] = evaluation_inventory
        manifests = _validate_measurement_root(
            measurement_inventory,
            config,
            camera_count,
            cases,
            objects,
            episodes,
        )
        summary, reports = _validate_evaluation_root(
            evaluation_inventory,
            measurement_inventory,
            manifests,
            camera_count,
            cases,
            objects,
            episodes,
        )
        manifests_by_budget[camera_count] = manifests
        reports_by_budget[camera_count] = reports
        evaluation_summaries[camera_count] = summary
    _validate_bound_8_view(
        config,
        measurement_inventories[8],
        evaluation_inventories[8],
    )
    cross_budget = _exact_cross_budget_checks(
        manifests_by_budget,
        reports_by_budget,
        cases,
    )
    analyses = {
        camera_count: _budget_analysis(
            camera_count,
            reports_by_budget[camera_count],
            cases,
            objects,
        )
        for camera_count in CAMERA_COUNTS
    }
    for camera_count in CAMERA_COUNTS:
        _add_case_relative_changes(
            analyses[camera_count],
            reports_by_budget[camera_count],
            cases,
        )
    decision = _four_view_decision(analyses)
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "panel": {
            "role": "outcome-open development only",
            "episode_count": len(cases),
            "physical_object_count": len(set(objects.values())),
            "cases": list(cases),
        },
        "method": {
            "stream": RAW_STREAM,
            "primary_arm": PRIMARY_ARM,
            "comparators": list(COMPARATORS),
            "primary_metrics": list(PRIMARY_METRICS),
            "camera_counts": list(CAMERA_COUNTS),
            "center_count": CENTER_COUNT,
            "update_frames": list(UPDATE_FRAMES),
            "alltracker_max_side": 512,
            "budget_semantics": (
                "dynamic tracked-view count after all-view frame-zero planning"
            ),
            "sole_changed_factor": (
                "selected_camera_count after immutable all-view frame-zero planning"
            ),
            "full_sensor_count_ablation": False,
            "raw_cpd_comparison_performed": False,
            "raw_cpd_omission_reason": config["evaluation"]["raw_cpd"]["reason"],
        },
        "input_inventories": {
            str(camera_count): {
                "measurement": measurement_inventories[camera_count].summary(),
                "evaluation": evaluation_inventories[camera_count].summary(),
                "evaluation_summary_result_sha256": evaluation_summaries[camera_count][
                    "result_sha256"
                ],
            }
            for camera_count in CAMERA_COUNTS
        },
        "cross_budget_invariants": cross_budget,
        "budgets": {
            str(camera_count): analyses[camera_count] for camera_count in CAMERA_COUNTS
        },
        "decision": decision,
        "claim_boundary": config["claim_boundary"],
    }


def analyze_camera_budget_frontier(
    config_path: str | Path,
) -> dict[str, Any]:
    """Analyze all budgets once and atomically publish ``frontier.json``."""

    path = Path(config_path).resolve(strict=True)
    config_bytes = _stable_file_bytes(path)
    try:
        config = json.loads(config_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("frontier config is invalid JSON") from error
    _require(isinstance(config, dict), "frontier config is not an object")
    report = build_frontier_report(config)
    report["config"] = {
        "path": str(path),
        "sha256": _sha256_bytes(config_bytes),
    }
    report["result_sha256"] = _canonical_sha256(report)
    destination = Path(config["roots"]["frontier_output"])
    _require(not destination.exists(), f"frontier output already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    temporary = destination / ".frontier.json.tmp"
    final = destination / "frontier.json"
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, final)
    return report


__all__ = [
    "CAMERA_COUNTS",
    "CENTER_COUNT",
    "COMPARATORS",
    "MEASUREMENT_PROTOCOL_ID",
    "PRIMARY_ARM",
    "PRIMARY_METRICS",
    "PROTOCOL_ID",
    "RAW_STREAM",
    "TreeInventory",
    "analyze_camera_budget_frontier",
    "build_frontier_report",
    "inventory_tree",
]
