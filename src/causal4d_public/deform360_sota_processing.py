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
DEVELOPMENT_PROCESSING_STAGE_KIND = "Deform360ReusableSotaDevelopmentStage"


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


__all__ = [
    "DEVELOPMENT_MASK_PANEL_KIND",
    "DEVELOPMENT_PROCESSING_STAGE_KIND",
    "authorize_development_processing",
    "propagate_development_masks",
    "stage_development_processing_episode",
]
