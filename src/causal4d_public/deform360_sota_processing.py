"""Development-only annotation staging for the reusable Deform360 protocol."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol

import numpy as np


DEVELOPMENT_MASK_PANEL_KIND = "Deform360ReusableSotaDevelopmentMaskPanel"
LEGACY_DEVELOPMENT_MASK_PANEL_KIND = "Deform360DevelopmentSam2MaskPanel"
DEVELOPMENT_PROCESSING_STAGE_KIND = "Deform360ReusableSotaDevelopmentStage"
DEVELOPMENT_OBSERVATIONS_KIND = "Deform360ReusableSotaDevelopmentObservations"

PINNED_DEFORM360_PROCESSING_REVISION = "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
PINNED_COTRACKER_REVISION = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
PINNED_COTRACKER_CHECKPOINT_SHA256 = (
    "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"
)
DEFORM360_PCD_TAIL_FRAMES_SKIPPED = 5


class DevelopmentMaskPredictor(Protocol):
    """Small SAM2 interface needed by the development runner."""

    def select_initial_mask_with_reference(
        self,
        video_path: Path,
        reference_rgb: np.ndarray,
        reference_mask: np.ndarray,
        *,
        reference_camera: str,
    ) -> tuple[np.ndarray, dict[str, Any]]: ...

    def segment_from_initial_mask(
        self,
        video_path: Path,
        initial_mask: np.ndarray,
        *,
        initialization: Mapping[str, Any] | None = None,
    ) -> Iterator[tuple[int, np.ndarray]]: ...


FirstFrameReader = Callable[[Path], np.ndarray]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def material_identity_sha256(points_m: np.ndarray) -> str:
    """Hash ordered frame-zero material points without platform-dependent metadata."""

    points = np.ascontiguousarray(np.asarray(points_m, dtype="<f4"))
    _require(
        points.ndim == 2
        and points.shape[1] == 3
        and len(points) > 0
        and np.all(np.isfinite(points)),
        "material points must be finite (N,3)",
    )
    digest = hashlib.sha256()
    digest.update(b"deform360-material-identity-v1\0")
    digest.update(np.asarray(points.shape, dtype="<i8").tobytes())
    digest.update(points.tobytes())
    return digest.hexdigest()


def _tree_sha256(root: Path, paths: Iterator[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def authorize_development_processing(
    protocol: Mapping[str, Any],
    *,
    object_id: str,
    episode_id: int,
    role: str,
) -> dict[str, Any]:
    """Authorize only a locked development episode and fail closed otherwise."""

    config = protocol.get("config")
    _require(isinstance(config, Mapping), "reusable SOTA protocol config is missing")
    development = config.get("development_objects")
    confirmatory = config.get("confirmatory_objects")
    _require(
        isinstance(development, Mapping) and isinstance(confirmatory, Mapping),
        "reusable SOTA object panels are missing",
    )
    development_ids = {
        str(value) for category in ("1d", "2d", "3d") for value in development[category]
    }
    confirmatory_ids = {
        str(value)
        for category in ("1d", "2d", "3d")
        for value in confirmatory[category]
    }
    _require(
        object_id not in confirmatory_ids, "confirmatory object access is forbidden"
    )
    _require(object_id in development_ids, "object is outside the development panel")
    dataset = config.get("dataset")
    _require(isinstance(dataset, Mapping), "dataset split is missing")
    allowed_by_role = {
        "fit": tuple(int(value) for value in dataset["fit_episode_ids"]),
        "held-development": tuple(int(value) for value in dataset["held_episode_ids"]),
    }
    _require(role in allowed_by_role, "unknown development processing role")
    _require(
        int(episode_id) in allowed_by_role[role],
        f"episode {episode_id} is outside the {role} split",
    )
    return {
        "protocol_id": str(config["protocol_id"]),
        "protocol_config_sha256": str(protocol["config_sha256"]),
        "object_id": object_id,
        "episode_id": int(episode_id),
        "role": role,
        "development_only": True,
        "confirmatory_object_opened": False,
    }


def _read_first_rgb(video_path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - remote integration
        raise RuntimeError(
            "OpenCV is required for Deform360 mask propagation"
        ) from error
    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, bgr = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"cannot decode first frame: {video_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_mask_frame(path: Path, frame_index: int) -> np.ndarray:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - remote integration
        raise RuntimeError("h5py is required for Deform360 masks") from error
    with h5py.File(path, "r") as stream:
        values = np.asarray(stream["data"][frame_index], dtype=bool)
    _require(values.ndim == 2 and np.any(values), f"reference mask is empty: {path}")
    return values


def _write_masks(path: Path, masks: np.ndarray) -> Path:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - remote integration
        raise RuntimeError("h5py is required for Deform360 masks") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as stream:
        stream.create_dataset(
            "data",
            data=np.asarray(masks, dtype=np.uint8),
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
        )
    return path


def _load_reference_panel(
    panel_path: Path,
    *,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    _require(isinstance(panel, dict), "reference mask panel must contain an object")
    _require(
        panel.get("artifact_kind") == "Deform360DevelopmentSam2MaskPanel",
        "unexpected reference mask-panel kind",
    )
    _require(
        panel.get("result_sha256") == _canonical_sha256(panel),
        "reference mask-panel checksum mismatch",
    )
    _require(
        panel.get("protocol_id") == authorization["protocol_id"]
        and panel.get("object_id") == authorization["object_id"]
        and panel.get("role") == "development-fit",
        "reference mask panel belongs to another protocol, object, or split",
    )
    _require(
        int(panel.get("episode_id", -1)) == 1,
        "reference mask panel must use locked fit episode 1",
    )
    records = panel.get("records")
    _require(isinstance(records, list) and records, "reference mask panel is empty")
    cameras = [str(record["camera"]) for record in records]
    _require(len(cameras) == len(set(cameras)), "reference cameras are duplicated")
    _require(
        int(panel.get("accepted_camera_count", -1)) == len(cameras),
        "reference camera count changed",
    )
    return panel


def load_development_reference_mask_panel(
    panel_path: str | Path,
    *,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the source-approved camera panel for another development episode."""

    return _load_reference_panel(Path(panel_path).resolve(), authorization=authorization)


def load_development_source_mask_panel(
    panel_path: str | Path,
    *,
    authorization: Mapping[str, Any],
    reference_cameras: list[str] | tuple[str, ...],
    start_frame: int,
    frame_count: int,
) -> dict[str, Any]:
    """Validate a source-only full-episode panel before slicing a fit window."""

    _require(
        authorization.get("role") == "fit"
        and authorization.get("development_only") is True
        and authorization.get("confirmatory_object_opened") is False,
        "source mask slicing is reserved for development fit episodes",
    )
    _require(start_frame >= 0 and frame_count == 81, "invalid source mask window")
    panel_file = Path(panel_path).resolve()
    panel = json.loads(panel_file.read_text(encoding="utf-8"))
    _require(isinstance(panel, dict), "source mask panel must contain an object")
    _require(
        panel.get("artifact_kind")
        in {DEVELOPMENT_MASK_PANEL_KIND, LEGACY_DEVELOPMENT_MASK_PANEL_KIND}
        and panel.get("result_sha256") == _canonical_sha256(panel),
        "source mask-panel kind or checksum is incompatible",
    )
    _require(
        panel.get("object_id") == authorization["object_id"]
        and int(panel.get("episode_id", -1)) == int(authorization["episode_id"]),
        "source mask panel belongs to another object or episode",
    )
    if panel["artifact_kind"] == DEVELOPMENT_MASK_PANEL_KIND:
        _require(
            panel.get("authorization") == dict(authorization)
            and panel.get("role") == "fit",
            "source mask-panel authorization changed",
        )
    else:
        _require(
            panel.get("protocol_id") == authorization["protocol_id"]
            and panel.get("role") == "development-fit",
            "legacy source mask panel belongs to another protocol or split",
        )
    boundary = panel.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("confirmatory_object_opened") is False
        and boundary.get("held_episode_opened", False) is False
        and boundary.get("future_object_outcome_metric_used", False) is False,
        "source mask-panel information boundary changed",
    )
    records = panel.get("records")
    _require(isinstance(records, list) and records, "source mask panel is empty")
    records_by_camera = {str(record["camera"]): record for record in records}
    cameras = [str(camera) for camera in reference_cameras]
    _require(
        len(cameras) == len(set(cameras))
        and set(records_by_camera) == set(cameras),
        "source masks do not match the frozen reference camera panel",
    )
    stop_frame = start_frame + frame_count
    for camera in cameras:
        record = records_by_camera[camera]
        _require(
            int(record.get("frame_count", -1)) >= stop_frame,
            f"source masks are too short for {camera}",
        )
        checksum = record.get("output_sha256")
        _require(
            isinstance(checksum, str)
            and len(checksum) == 64
            and all(character in "0123456789abcdef" for character in checksum),
            f"invalid source mask checksum for {camera}",
        )
    return panel


def propagate_development_masks(
    *,
    authorization: Mapping[str, Any],
    aligned_object_root: str | Path,
    reference_annotation_root: str | Path,
    reference_panel_path: str | Path,
    output_annotation_root: str | Path,
    predictor: DevelopmentMaskPredictor,
    first_frame_reader: FirstFrameReader = _read_first_rgb,
    weak_camera_empty_frame_fraction: float = 0.25,
    minimum_nonempty_camera_count_per_frame: int = 3,
) -> dict[str, Any]:
    """Re-identify from frame zero, then propagate a fixed source camera panel."""

    _require(
        0.0 <= weak_camera_empty_frame_fraction < 1.0,
        "invalid empty-frame threshold",
    )
    _require(
        minimum_nonempty_camera_count_per_frame >= 2,
        "at least two nonempty cameras are required per frame",
    )
    object_id = str(authorization["object_id"])
    episode_id = int(authorization["episode_id"])
    aligned_root = Path(aligned_object_root).resolve()
    reference_root = Path(reference_annotation_root).resolve()
    output_root = Path(output_annotation_root).resolve()
    panel_file = Path(reference_panel_path).resolve()
    panel = _load_reference_panel(panel_file, authorization=authorization)
    reference_episode = aligned_root / "episode_0001"
    target_episode = aligned_root / f"episode_{episode_id:04d}"
    _require(
        target_episode.is_dir(), f"aligned target episode is missing: {target_episode}"
    )
    destination_episode = output_root / object_id / f"episode_{episode_id:04d}"
    _require(
        not destination_episode.exists(),
        f"development mask output already exists: {destination_episode}",
    )
    working_episode = destination_episode.with_name(
        f".{destination_episode.name}.incomplete-{os.getpid()}"
    )
    _require(not working_episode.exists(), "development mask scratch path exists")
    working_episode.parent.mkdir(parents=True, exist_ok=True)
    working_episode.mkdir()

    records = []
    nonempty_by_camera = []
    expected_frame_count: int | None = None
    started = time.time()
    try:
        for reference_record in panel["records"]:
            camera = str(reference_record["camera"])
            reference_video = reference_episode / camera / "undistorted.mp4"
            target_video = target_episode / camera / "undistorted.mp4"
            reference_mask_path = (
                reference_root / object_id / "episode_0001" / camera / "mask_refined.h5"
            )
            _require(reference_video.is_file(), f"reference video is missing: {camera}")
            _require(target_video.is_file(), f"target video is missing: {camera}")
            _require(
                reference_mask_path.is_file(), f"reference mask is missing: {camera}"
            )
            _require(
                _file_sha256(reference_mask_path) == reference_record["output_sha256"],
                f"reference mask checksum changed: {camera}",
            )
            reference_rgb = first_frame_reader(reference_video)
            reference_mask = _read_mask_frame(reference_mask_path, 0)
            initial_mask, selection = predictor.select_initial_mask_with_reference(
                target_video,
                reference_rgb,
                reference_mask,
                reference_camera=camera,
            )
            initialization = {
                "policy": "same-object-same-view-source-appearance",
                "reference_episode_id": 1,
                "reference_camera": camera,
                "object_observation_frames_used": [0],
                "future_object_observations_used": False,
                "selection": selection,
            }
            propagated = list(
                predictor.segment_from_initial_mask(
                    target_video,
                    np.asarray(initial_mask, dtype=bool),
                    initialization=initialization,
                )
            )
            frame_indices = [int(index) for index, _ in propagated]
            _require(
                frame_indices == list(range(len(propagated))) and propagated,
                f"SAM2 returned incomplete or unordered masks: {camera}",
            )
            masks = np.stack([np.asarray(mask, dtype=bool) for _, mask in propagated])
            _require(
                masks.ndim == 3 and np.any(masks[0]),
                f"SAM2 initial mask is empty: {camera}",
            )
            if expected_frame_count is None:
                expected_frame_count = len(masks)
            _require(
                len(masks) == expected_frame_count,
                "development cameras have different frame counts",
            )
            areas = np.count_nonzero(masks, axis=(1, 2))
            nonempty = areas > 0
            nonempty_by_camera.append(nonempty)
            empty_count = int(np.count_nonzero(~nonempty))
            empty_fraction = empty_count / len(masks)
            destination = working_episode / camera / "mask_refined.h5"
            _write_masks(destination, masks)
            records.append(
                {
                    "camera": camera,
                    "frame_count": len(masks),
                    "empty_frame_count": empty_count,
                    "empty_frame_fraction": empty_fraction,
                    "weak_camera": empty_fraction > weak_camera_empty_frame_fraction,
                    "area_min": int(np.min(areas)),
                    "area_median": float(np.median(areas)),
                    "area_max": int(np.max(areas)),
                    "reference_initial_area": int(np.count_nonzero(reference_mask)),
                    "target_initial_area": int(np.count_nonzero(masks[0])),
                    "selection": selection,
                    "output_sha256": _file_sha256(destination),
                }
            )

        assert expected_frame_count is not None
        nonempty_camera_count = np.count_nonzero(
            np.stack(nonempty_by_camera, axis=0), axis=0
        )
        minimum_nonempty = int(np.min(nonempty_camera_count))
        _require(
            minimum_nonempty >= minimum_nonempty_camera_count_per_frame,
            "too few nonempty camera masks at one or more frames",
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": DEVELOPMENT_MASK_PANEL_KIND,
            "authorization": dict(authorization),
            "object_id": object_id,
            "episode_id": episode_id,
            "role": str(authorization["role"]),
            "reference_episode_id": 1,
            "reference_panel_sha256": _file_sha256(panel_file),
            "camera_count": len(records),
            "frame_count": expected_frame_count,
            "weak_camera_empty_frame_fraction": weak_camera_empty_frame_fraction,
            "weak_camera_count": sum(record["weak_camera"] for record in records),
            "minimum_nonempty_camera_count_per_frame": minimum_nonempty,
            "required_nonempty_camera_count_per_frame": (
                minimum_nonempty_camera_count_per_frame
            ),
            "records": records,
            "seconds": time.time() - started,
            "information_boundary": {
                "initial_object_frames_used_for_selection": [0],
                "future_object_frames_used_only_for_development_annotation": True,
                "future_object_outcome_metric_used": False,
                "simulator_residual_used": False,
                "confirmatory_object_opened": False,
            },
            "claim_boundary": (
                "Development SAM2 producer; no SAM3 or official evaluator parity "
                "claimed."
            ),
        }
        payload["result_sha256"] = _canonical_sha256(payload)
        panel_output = working_episode / "mask_panel.json"
        panel_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(working_episode, destination_episode)
        return payload
    except Exception:
        shutil.rmtree(working_episode, ignore_errors=True)
        raise


def _replace_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists() or path.is_symlink():
        _require(overwrite, f"development processing stage already exists: {path}")
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True)


def _symlink_existing(source: Path, destination: Path) -> None:
    _require(source.exists(), f"processing input is missing: {source}")
    os.symlink(source.resolve(), destination)


def stage_development_processing_episode(
    *,
    authorization: Mapping[str, Any],
    aligned_object_root: str | Path,
    annotation_root: str | Path,
    processing_root: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build the symlink-only official-processing view for one development episode."""

    object_id = str(authorization["object_id"])
    episode_id = int(authorization["episode_id"])
    aligned_episode = Path(aligned_object_root).resolve() / f"episode_{episode_id:04d}"
    annotation_episode = (
        Path(annotation_root).resolve() / object_id / f"episode_{episode_id:04d}"
    )
    mask_panel_path = annotation_episode / "mask_panel.json"
    mask_panel = json.loads(mask_panel_path.read_text(encoding="utf-8"))
    _require(
        mask_panel.get("artifact_kind") == DEVELOPMENT_MASK_PANEL_KIND
        and mask_panel.get("result_sha256") == _canonical_sha256(mask_panel)
        and mask_panel.get("authorization") == dict(authorization),
        "development mask panel is incompatible",
    )
    output_episode = (
        Path(processing_root).resolve() / object_id / f"episode_{episode_id:04d}"
    )
    _replace_directory(output_episode, overwrite=overwrite)
    for name in (
        "alignment.json",
        "extrinsics.npy",
        "undistorted_intrinsics.npy",
        "robot",
    ):
        _symlink_existing(aligned_episode / name, output_episode / name)
    for record in mask_panel["records"]:
        camera = str(record["camera"])
        source_camera = aligned_episode / camera
        output_camera = output_episode / camera
        output_camera.mkdir()
        for name in (
            "aligned_timestamps.txt",
            "alignment.json",
            "metadata.json",
            "undistorted.mp4",
            "undistorted_000000.png",
        ):
            source = source_camera / name
            if source.exists():
                _symlink_existing(source, output_camera / name)
        mask_path = annotation_episode / camera / "mask_refined.h5"
        _require(
            _file_sha256(mask_path) == record["output_sha256"],
            f"development mask checksum changed: {camera}",
        )
        _symlink_existing(mask_path, output_camera / "mask_refined.h5")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": DEVELOPMENT_PROCESSING_STAGE_KIND,
        "authorization": dict(authorization),
        "object_id": object_id,
        "episode_id": episode_id,
        "role": str(authorization["role"]),
        "camera_count": len(mask_panel["records"]),
        "frame_count": int(mask_panel["frame_count"]),
        "mask_panel_result_sha256": mask_panel["result_sha256"],
        "mask_panel_file_sha256": _file_sha256(mask_panel_path),
        "information_boundary": {
            "development_only": True,
            "confirmatory_object_opened": False,
            "target_metric_read": False,
        },
        "claim_boundary": (
            "Development processing only; no official SAM3 or evaluator parity claim."
        ),
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    (output_episode / "development_staging.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def write_development_action_window_stage(
    path: str | Path,
    *,
    authorization: Mapping[str, Any],
    window_authorization: Mapping[str, Any],
    selected_raw_frame_range_half_open: tuple[int, int] | list[int],
    camera_count: int,
    frame_count: int,
    known_robot_action_frame_count: int | None = None,
    window_config_sha256: str,
    mask_diagnostics_sha256: str,
    initialization_diagnostics_sha256: str,
) -> dict[str, Any]:
    """Write a SOTA-compatible fit or sealed held-prediction slice."""

    role = str(authorization.get("role"))
    _require(
        role in {"fit", "held-development"}
        and authorization.get("development_only") is True
        and authorization.get("confirmatory_object_opened") is False,
        "action-window stage requires development authorization",
    )
    operation = str(window_authorization.get("operation"))
    expected_operation = {
        "fit": "development-fit-staging",
        "held-development": "development-held-prediction-staging",
    }[role]
    _require(
        operation == expected_operation
        and window_authorization.get("object_id") == authorization.get("object_id")
        and int(window_authorization.get("episode_id", -1))
        == int(authorization.get("episode_id", -2))
        and window_authorization.get("confirmatory_object_read") is False,
        "action-window authorization is incompatible",
    )
    if role == "fit":
        _require(
            window_authorization.get("held_outcome_read") is False,
            "fit window authorization may not read held outcomes",
        )
    else:
        _require(
            window_authorization.get("held_action_read") is True
            and window_authorization.get("held_object_input_frame_count") == 1
            and window_authorization.get("held_future_object_read") is False
            and window_authorization.get("held_tactile_read") is False
            and window_authorization.get(
                "prediction_seal_required_before_outcome_reveal"
            )
            is True,
            "held prediction authorization changed",
        )
    start, stop = (int(value) for value in selected_raw_frame_range_half_open)
    action_frame_count = (
        int(frame_count)
        if known_robot_action_frame_count is None
        else int(known_robot_action_frame_count)
    )
    expected_object_frames = 81 if role == "fit" else 1
    _require(
        start >= 0
        and stop - start == action_frame_count == 81
        and int(frame_count) == expected_object_frames,
        "invalid action window",
    )
    _require(camera_count >= 3, "too few cameras in the action-window stage")
    for label, value in (
        ("window config", window_config_sha256),
        ("mask diagnostics", mask_diagnostics_sha256),
        ("initialization diagnostics", initialization_diagnostics_sha256),
    ):
        _require(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value),
            f"invalid {label} checksum",
        )
    temporal_staging: dict[str, Any] = {
        "mode": "locked-action-only-window",
        "selected_raw_frame_range_half_open": [start, stop],
        "window_config_sha256": window_config_sha256,
        "window_authorization": dict(window_authorization),
    }
    information_boundary: dict[str, Any] = {
        "development_only": True,
        "confirmatory_object_opened": False,
        "target_metric_read": False,
        "window_selection_used_robot_action_and_opening_only": True,
        "window_selection_used_object_geometry_or_tactile": False,
    }
    if role == "held-development":
        temporal_staging.update(
            {
                "mode": "locked-held-prediction-window",
                "known_robot_action_frame_count": action_frame_count,
                "object_observation_frame_count": int(frame_count),
            }
        )
        information_boundary.update(
            {
                "object_observation_frames_used": [0],
                "future_object_outcome_read": False,
                "future_tactile_read": False,
                "prediction_seal_required_before_outcome_reveal": True,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": DEVELOPMENT_PROCESSING_STAGE_KIND,
        "authorization": dict(authorization),
        "object_id": str(authorization["object_id"]),
        "episode_id": int(authorization["episode_id"]),
        "role": role,
        "camera_count": int(camera_count),
        "frame_count": int(frame_count),
        "temporal_staging": temporal_staging,
        "input_sha256": {
            "mask_diagnostics": mask_diagnostics_sha256,
            "initialization_diagnostics": initialization_diagnostics_sha256,
        },
        "information_boundary": information_boundary,
        "claim_boundary": (
            "Action-window compute addendum for independent development processing; "
            "no official Deform360 evaluator or Table 4 parity claim."
        ),
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def validate_development_held_prediction_stage(
    path: str | Path,
    *,
    authorization: Mapping[str, Any],
    window_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a one-frame held initialization before a prediction is built."""

    stage_path = Path(path).resolve()
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    temporal = stage.get("temporal_staging", {})
    boundary = stage.get("information_boundary", {})
    selected = temporal.get("selected_raw_frame_range_half_open", ())
    _require(
        stage.get("artifact_kind") == DEVELOPMENT_PROCESSING_STAGE_KIND
        and stage.get("result_sha256") == _canonical_sha256(stage)
        and stage.get("authorization") == dict(authorization)
        and stage.get("role") == "held-development"
        and stage.get("frame_count") == 1
        and temporal.get("mode") == "locked-held-prediction-window"
        and temporal.get("window_authorization") == dict(window_authorization)
        and temporal.get("known_robot_action_frame_count") == 81
        and temporal.get("object_observation_frame_count") == 1
        and isinstance(selected, list)
        and len(selected) == 2
        and int(selected[1]) - int(selected[0]) == 81,
        "held prediction stage is incompatible",
    )
    _require(
        boundary.get("target_metric_read") is False
        and boundary.get("window_selection_used_robot_action_and_opening_only")
        is True
        and boundary.get("window_selection_used_object_geometry_or_tactile") is False
        and boundary.get("object_observation_frames_used") == [0]
        and boundary.get("future_object_outcome_read") is False
        and boundary.get("future_tactile_read") is False
        and boundary.get("prediction_seal_required_before_outcome_reveal") is True,
        "held prediction information boundary changed",
    )
    return {
        "passed": True,
        "object_id": str(authorization["object_id"]),
        "episode_id": int(authorization["episode_id"]),
        "role": "held-development",
        "stage_sha256": _file_sha256(stage_path),
        "stage_result_sha256": str(stage["result_sha256"]),
        "known_robot_action_frame_count": 81,
        "object_observation_frame_count": 1,
        "held_future_read": False,
    }


def _load_development_stage(
    episode_dir: Path, *, authorization: Mapping[str, Any]
) -> dict[str, Any]:
    stage_path = episode_dir / "development_staging.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    _require(
        stage.get("artifact_kind") == DEVELOPMENT_PROCESSING_STAGE_KIND
        and stage.get("result_sha256") == _canonical_sha256(stage)
        and stage.get("authorization") == dict(authorization),
        "development processing stage is incompatible",
    )
    return stage


def build_development_observations_manifest(
    *,
    authorization: Mapping[str, Any],
    processing_root: str | Path,
    deform360_processing_revision: str,
    cotracker_revision: str,
    cotracker_checkpoint: str | Path,
) -> dict[str, Any]:
    """Bind official processing outputs to one authorized development episode."""

    _require(
        deform360_processing_revision == PINNED_DEFORM360_PROCESSING_REVISION,
        "Deform360 processing revision changed",
    )
    _require(
        cotracker_revision == PINNED_COTRACKER_REVISION,
        "CoTracker revision changed",
    )
    checkpoint = Path(cotracker_checkpoint).resolve()
    _require(
        checkpoint.is_file()
        and _file_sha256(checkpoint) == PINNED_COTRACKER_CHECKPOINT_SHA256,
        "CoTracker checkpoint changed",
    )
    object_id = str(authorization["object_id"])
    episode_id = int(authorization["episode_id"])
    episode_dir = (
        Path(processing_root).resolve() / object_id / f"episode_{episode_id:04d}"
    )
    stage = _load_development_stage(episode_dir, authorization=authorization)
    frame_count = int(stage["frame_count"])
    cameras = sorted(
        path.name
        for path in episode_dir.iterdir()
        if path.is_dir() and (path / "mask_refined.h5").is_file()
    )
    _require(
        len(cameras) == int(stage["camera_count"]),
        "processed camera panel differs from the staging artifact",
    )

    splat_dir = episode_dir / "splatfacto"
    splats = [splat_dir / f"splat_{frame}.ply" for frame in range(frame_count)]
    _require(all(path.is_file() for path in splats), "reconstruction is incomplete")
    reconstruction_meta = splat_dir / "splatfacto.meta.json"
    _require(reconstruction_meta.is_file(), "reconstruction provenance is missing")

    gripper_masks: dict[str, dict[str, str]] = {}
    depth: dict[str, dict[str, str]] = {}
    tracking: dict[str, dict[str, str]] = {}
    for camera in cameras:
        camera_dir = episode_dir / camera
        gripper_mask = camera_dir / "rendered_urdf.h5"
        gripper_mask_meta = camera_dir / "rendered_urdf.meta.json"
        depth_file = camera_dir / "rendered_depth.h5"
        depth_meta = camera_dir / "rendered_depth.meta.json"
        tracking_dir = camera_dir / "tracking"
        velocity = tracking_dir / "vel.h5"
        visibility = tracking_dir / "visibility.h5"
        tracking_meta = tracking_dir / "tracking.meta.json"
        _require(
            all(
                path.is_file()
                for path in (
                    gripper_mask,
                    gripper_mask_meta,
                    depth_file,
                    depth_meta,
                    velocity,
                    visibility,
                    tracking_meta,
                )
            ),
            f"depth or tracking output is incomplete: {camera}",
        )
        gripper_masks[camera] = {
            "data_sha256": _file_sha256(gripper_mask),
            "metadata_sha256": _file_sha256(gripper_mask_meta),
        }
        depth[camera] = {
            "data_sha256": _file_sha256(depth_file),
            "metadata_sha256": _file_sha256(depth_meta),
        }
        tracking[camera] = {
            "velocity_sha256": _file_sha256(velocity),
            "visibility_sha256": _file_sha256(visibility),
            "metadata_sha256": _file_sha256(tracking_meta),
        }

    pcd_dir = episode_dir / "pcd_clean"
    point_frame_count = frame_count - DEFORM360_PCD_TAIL_FRAMES_SKIPPED
    _require(point_frame_count >= 2, "episode is too short for future evaluation")
    pcd_paths = [pcd_dir / f"{frame:06d}.npz" for frame in range(point_frame_count)]
    _require(
        all(path.is_file() for path in pcd_paths), "point-cloud output is incomplete"
    )
    pcd_meta = pcd_dir / "pcd_clean.meta.json"
    _require(pcd_meta.is_file(), "point-cloud provenance is missing")
    with np.load(pcd_paths[0], allow_pickle=False) as stored:
        frame_zero = np.asarray(stored["pts"], dtype=np.float32)
    identity_sha256 = material_identity_sha256(frame_zero)
    point_count = len(frame_zero)
    for path in pcd_paths[1:]:
        with np.load(path, allow_pickle=False) as stored:
            points = np.asarray(stored["pts"])
        _require(
            points.shape == frame_zero.shape and np.all(np.isfinite(points)),
            "advected material identities changed shape or became non-finite",
        )

    control_names = (
        "calibrate.pkl",
        "start_obj_pcd.ply",
        "split.json",
        "final_data.pkl",
        "control_points.meta.json",
    )
    control_paths = {name: episode_dir / name for name in control_names}
    _require(
        all(path.is_file() for path in control_paths.values()),
        "control-point output is incomplete",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": DEVELOPMENT_OBSERVATIONS_KIND,
        "authorization": dict(authorization),
        "object_id": object_id,
        "episode_id": episode_id,
        "role": str(authorization["role"]),
        "camera_count": len(cameras),
        "frame_count": frame_count,
        "point_frame_count": point_frame_count,
        "material_point_count": point_count,
        "material_identity_sha256": identity_sha256,
        "development_staging_result_sha256": stage["result_sha256"],
        "implementation_revision": {
            "deform360_processing": deform360_processing_revision,
            "cotracker": cotracker_revision,
        },
        "input_sha256": {
            "development_staging": _file_sha256(
                episode_dir / "development_staging.json"
            ),
            "cotracker_checkpoint": _file_sha256(checkpoint),
        },
        "output_sha256": {
            "reconstruction_metadata": _file_sha256(reconstruction_meta),
            "reconstruction_tree": _tree_sha256(splat_dir, iter(splats)),
            "gripper_masks": gripper_masks,
            "depth": depth,
            "tracking": tracking,
            "point_cloud_metadata": _file_sha256(pcd_meta),
            "point_cloud_tree": _tree_sha256(pcd_dir, iter(pcd_paths)),
            "control_points": {
                name: _file_sha256(path) for name, path in control_paths.items()
            },
        },
        "information_boundary": {
            "development_only": True,
            "future_frames_used_for_development_supervision": True,
            "prediction_metric_computed": False,
            "confirmatory_object_opened": False,
            "pokeflex_target_opened": False,
        },
        "claim_boundary": (
            "Checksummed independent development supervision; no official "
            "Deform360 evaluator or Table 4 parity claim."
        ),
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return payload


def validate_development_final_data_input(
    observations: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    final_data_path: str | Path,
) -> dict[str, Any]:
    """Bind one fit-only PhysTwin input to checksummed development supervision."""

    _require(
        observations.get("artifact_kind") == DEVELOPMENT_OBSERVATIONS_KIND
        and observations.get("result_sha256") == _canonical_sha256(observations),
        "development observation artifact is incompatible",
    )
    _require(
        observations.get("authorization") == dict(authorization)
        and observations.get("role") == "fit",
        "development observations use another authorization",
    )
    _require(
        authorization.get("development_only") is True
        and authorization.get("confirmatory_object_opened") is False,
        "development final_data input is outside the prospective panel",
    )
    final_data = Path(final_data_path).resolve()
    _require(final_data.is_file(), "development final_data input is missing")
    expected = (
        observations.get("output_sha256", {})
        .get("control_points", {})
        .get("final_data.pkl")
    )
    observed = _file_sha256(final_data)
    _require(expected == observed, "development final_data checksum changed")
    return {
        "passed": True,
        "object_id": str(authorization["object_id"]),
        "episode_id": int(authorization["episode_id"]),
        "role": "fit",
        "final_data_sha256": observed,
        "observations_result_sha256": str(observations["result_sha256"]),
        "point_frame_count": int(observations["point_frame_count"]),
        "held_outcome_read": False,
        "confirmatory_object_read": False,
    }


__all__ = [
    "DEFORM360_PCD_TAIL_FRAMES_SKIPPED",
    "DEVELOPMENT_MASK_PANEL_KIND",
    "DEVELOPMENT_OBSERVATIONS_KIND",
    "DEVELOPMENT_PROCESSING_STAGE_KIND",
    "PINNED_COTRACKER_CHECKPOINT_SHA256",
    "PINNED_COTRACKER_REVISION",
    "PINNED_DEFORM360_PROCESSING_REVISION",
    "authorize_development_processing",
    "build_development_observations_manifest",
    "load_development_reference_mask_panel",
    "load_development_source_mask_panel",
    "material_identity_sha256",
    "propagate_development_masks",
    "stage_development_processing_episode",
    "validate_development_final_data_input",
    "validate_development_held_prediction_stage",
    "write_development_action_window_stage",
]
