"""MolmoMotion query preparation and inference for released PhysTwin cases."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bayesian_phystwin.phystwin_raw_cues import load_phystwin_raw_track_map


def farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    """Select a deterministic, spatially distributed subset."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] not in {2, 3}:
        raise ValueError("points must have shape (N, 2|3)")
    if not 1 <= count <= len(values):
        raise ValueError("count must lie in [1, N]")
    if not np.all(np.isfinite(values)):
        raise ValueError("points must be finite")
    centered = values - np.mean(values, axis=0, keepdims=True)
    selected = [int(np.argmax(np.sum(np.square(centered), axis=1)))]
    minimum_distance = np.sum(np.square(values - values[selected[0]]), axis=1)
    minimum_distance[selected[0]] = -np.inf
    while len(selected) < count:
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        distance = np.sum(np.square(values - values[next_index]), axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -np.inf
    return np.asarray(selected, dtype=np.int32)


@dataclass(frozen=True)
class MolmoPhysTwinQuery:
    """Eight material-point histories and raw RGB frames for MolmoMotion."""

    case_name: str
    raw_case_dir: Path
    camera_index: int
    t0_frame: int
    history_frame_indices: np.ndarray
    image_paths: tuple[Path, ...]
    node_indices: np.ndarray
    raw_track_indices: np.ndarray
    points_2d_xy: np.ndarray
    points_3d_world_history_m: np.ndarray
    camera_to_world: np.ndarray
    intrinsics: np.ndarray
    source_fps: float
    forecast_fps: float
    frame_stride: int

    def __post_init__(self) -> None:
        history = np.asarray(self.history_frame_indices, dtype=int)
        nodes = np.asarray(self.node_indices, dtype=int)
        raw_tracks = np.asarray(self.raw_track_indices, dtype=int)
        points_2d = np.asarray(self.points_2d_xy, dtype=float)
        points_3d = np.asarray(self.points_3d_world_history_m, dtype=float)
        camera_to_world = np.asarray(self.camera_to_world, dtype=float)
        intrinsics = np.asarray(self.intrinsics, dtype=float)
        point_count = len(nodes)
        expected_history = tuple(
            range(
                self.t0_frame - self.frame_stride * (len(history) - 1),
                self.t0_frame + 1,
                self.frame_stride,
            )
        )
        if history.ndim != 1 or tuple(history) != expected_history:
            raise ValueError("history frames must be regularly sampled and end at t0")
        if len(self.image_paths) != len(history):
            raise ValueError("one image path is required for each history frame")
        if raw_tracks.shape != (point_count,) or points_2d.shape != (point_count, 2):
            raise ValueError("raw tracks and 2D queries must match node_indices")
        if points_3d.shape != (len(history), point_count, 3):
            raise ValueError("3D history must have shape (H, P, 3)")
        if camera_to_world.shape != (4, 4) or intrinsics.shape != (3, 3):
            raise ValueError("camera matrices must be 4x4 and 3x3")
        if (
            not np.isfinite(self.source_fps)
            or not np.isfinite(self.forecast_fps)
            or self.source_fps <= 0.0
            or self.forecast_fps <= 0.0
            or self.frame_stride < 1
        ):
            raise ValueError("source/forecast rates and frame stride must be positive")
        if not np.isclose(self.source_fps / self.forecast_fps, self.frame_stride):
            raise ValueError("frame stride must match source_fps / forecast_fps")
        if np.any(nodes < 0) or np.any(raw_tracks < 0):
            raise ValueError("track indices must be nonnegative")
        for path in self.image_paths:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        object.__setattr__(self, "history_frame_indices", history)
        object.__setattr__(self, "node_indices", nodes)
        object.__setattr__(self, "raw_track_indices", raw_tracks)
        object.__setattr__(self, "points_2d_xy", points_2d)
        object.__setattr__(self, "points_3d_world_history_m", points_3d)
        object.__setattr__(self, "camera_to_world", camera_to_world)
        object.__setattr__(self, "intrinsics", intrinsics)

    @property
    def anchor_positions_world_m(self) -> np.ndarray:
        return self.points_3d_world_history_m[-1]

    def metadata(self) -> dict[str, Any]:
        return {
            "case": self.case_name,
            "raw_case_dir": str(self.raw_case_dir.resolve()),
            "camera_index": self.camera_index,
            "t0_frame": self.t0_frame,
            "history_frame_indices": self.history_frame_indices.tolist(),
            "image_paths": [str(path.resolve()) for path in self.image_paths],
            "node_indices": self.node_indices.tolist(),
            "raw_track_indices": self.raw_track_indices.tolist(),
            "source_fps": self.source_fps,
            "forecast_fps": self.forecast_fps,
            "frame_stride": self.frame_stride,
            "point_coordinate_contract": "raw row/column tracks converted to x/y pixels",
            "trajectory_coordinate_contract": "metric world coordinates",
            "temporal_contract": "history and future sampled at forecast_fps",
        }


def prepare_molmo_phystwin_query(
    final_data_path: str | Path,
    raw_case_dir: str | Path,
    *,
    train_end_frame: int,
    history_size: int = 3,
    point_count: int = 8,
    camera_index: int | None = None,
    forecast_fps: float = 15.0,
) -> MolmoPhysTwinQuery:
    """Recover exact raw identities and choose visible material query points."""

    final_path = Path(final_data_path)
    raw_path = Path(raw_case_dir)
    with final_path.open("rb") as handle:
        final_data = pickle.load(handle)
    object_points = np.asarray(final_data["object_points"], dtype=float)
    visible = np.asarray(final_data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(final_data["object_motions_valid"], dtype=bool)
    frame_count, track_count, coordinate_count = object_points.shape
    if coordinate_count != 3 or visible.shape != (frame_count, track_count):
        raise ValueError("final_data object arrays have incompatible shapes")
    if not history_size <= train_end_frame < frame_count:
        raise ValueError("train_end_frame cannot provide the requested history")
    if point_count < 1:
        raise ValueError("point_count must be positive")
    if not np.isfinite(forecast_fps) or forecast_fps <= 0.0:
        raise ValueError("forecast_fps must be finite and positive")
    metadata = json.loads((raw_path / "metadata.json").read_text(encoding="utf-8"))
    source_fps = float(metadata["fps"])
    ratio = source_fps / float(forecast_fps)
    frame_stride = int(round(ratio))
    if frame_stride < 1 or not np.isclose(ratio, frame_stride):
        raise ValueError("raw source fps must be an integer multiple of forecast_fps")
    t0 = train_end_frame - 1
    history_frames = np.arange(
        t0 - frame_stride * (history_size - 1),
        t0 + 1,
        frame_stride,
        dtype=int,
    )
    if history_frames[0] < 0:
        raise ValueError("train_end_frame cannot provide the requested sampled history")
    mapping = load_phystwin_raw_track_map(final_path, raw_path)
    intrinsics = np.asarray(metadata["intrinsics"], dtype=float)
    width, height = map(int, metadata["WH"])
    with (raw_path / "calibrate.pkl").open("rb") as handle:
        camera_to_world = np.asarray(pickle.load(handle), dtype=float)
    camera_count = len(mapping.track_paths)
    if intrinsics.shape != (camera_count, 3, 3) or camera_to_world.shape != (camera_count, 4, 4):
        raise ValueError("raw camera calibration does not match the track archives")

    candidates_by_camera: list[np.ndarray] = []
    for camera in range(camera_count):
        source_match = mapping.source_camera == camera
        raw_ids = mapping.source_track
        raw_visibility = mapping.visibility_by_camera[camera][
            history_frames[:, None], raw_ids[None]
        ]
        raw_tracks = mapping.tracks_by_camera[camera][history_frames[:, None], raw_ids[None]]
        # Released archives use row/column coordinates, as required by the pcd lookup.
        row = raw_tracks[..., 0]
        column = raw_tracks[..., 1]
        in_bounds = (
            (row >= 0.0)
            & (row < height)
            & (column >= 0.0)
            & (column < width)
        )
        processed_valid = np.all(
            visible[history_frames] & motion_valid[history_frames], axis=0
        )
        eligible = (
            source_match
            & processed_valid
            & np.all(raw_visibility & in_bounds, axis=0)
            & np.all(np.isfinite(object_points[history_frames]), axis=(0, 2))
        )
        candidates_by_camera.append(np.flatnonzero(eligible))
    if camera_index is None:
        camera = int(np.argmax([len(values) for values in candidates_by_camera]))
    else:
        camera = int(camera_index)
        if not 0 <= camera < camera_count:
            raise ValueError("camera_index is unavailable")
    candidates = candidates_by_camera[camera]
    if len(candidates) < point_count:
        raise ValueError(
            f"camera {camera} has only {len(candidates)} valid tracks; need {point_count}"
        )
    selected_local = farthest_point_indices(object_points[t0, candidates], point_count)
    nodes = candidates[selected_local]
    raw_ids = mapping.source_track[nodes]
    raw_row_column = mapping.tracks_by_camera[camera][t0, raw_ids]
    points_xy = raw_row_column[:, [1, 0]]
    image_paths = tuple(raw_path / "color" / str(camera) / f"{frame}.png" for frame in history_frames)
    return MolmoPhysTwinQuery(
        case_name=final_path.resolve().parent.name,
        raw_case_dir=raw_path,
        camera_index=camera,
        t0_frame=t0,
        history_frame_indices=history_frames,
        image_paths=image_paths,
        node_indices=nodes,
        raw_track_indices=raw_ids,
        points_2d_xy=points_xy,
        points_3d_world_history_m=object_points[history_frames][:, nodes],
        camera_to_world=camera_to_world[camera],
        intrinsics=intrinsics[camera],
        source_fps=source_fps,
        forecast_fps=float(forecast_fps),
        frame_stride=frame_stride,
    )


def camera_to_world_points(points_camera_m: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera_m, dtype=float)
    transform = np.asarray(camera_to_world, dtype=float)
    if points.ndim < 2 or points.shape[-1] != 3 or transform.shape != (4, 4):
        raise ValueError("camera points and transform have incompatible shapes")
    flat = points.reshape(-1, 3)
    homogeneous = np.column_stack((flat, np.ones(len(flat))))
    world = homogeneous @ transform.T
    return world[:, :3].reshape(points.shape)


@dataclass(frozen=True)
class MolmoForecastBundle:
    query: MolmoPhysTwinQuery
    forecast_ids: tuple[str, ...]
    captions: tuple[str, ...]
    future_camera_m: np.ndarray
    future_world_m: np.ndarray
    raw_text: tuple[str, ...]
    checkpoint: str

    def __post_init__(self) -> None:
        camera = np.asarray(self.future_camera_m, dtype=float)
        world = np.asarray(self.future_world_m, dtype=float)
        expected_prefix = (len(self.forecast_ids), len(self.query.node_indices))
        if camera.ndim != 4 or camera.shape[:2] != expected_prefix or camera.shape[3] != 3:
            raise ValueError("future_camera_m must have shape (K, P, F, 3)")
        if world.shape != camera.shape:
            raise ValueError("world and camera forecasts must have matching shapes")
        if len(self.captions) != len(self.forecast_ids) or len(self.raw_text) != len(self.forecast_ids):
            raise ValueError("forecast metadata must match forecast_ids")
        if len(set(self.forecast_ids)) != len(self.forecast_ids):
            raise ValueError("forecast_ids must be unique")
        if not np.all(np.isfinite(camera)) or not np.all(np.isfinite(world)):
            raise ValueError("MolmoMotion forecasts must be finite")
        object.__setattr__(self, "future_camera_m", camera)
        object.__setattr__(self, "future_world_m", world)

    @property
    def future_horizon(self) -> int:
        return int(self.future_world_m.shape[2])

    def metadata(self) -> dict[str, Any]:
        return {
            "model": "MolmoMotion",
            "checkpoint": self.checkpoint,
            "forecast_ids": list(self.forecast_ids),
            "captions": dict(zip(self.forecast_ids, self.captions, strict=True)),
            "future_horizon": self.future_horizon,
            "query": self.query.metadata(),
            "output_coordinate_frames": {
                "future_camera_m": "camera at t0",
                "future_world_m": "PhysTwin calibrated world",
            },
        }


def run_molmo_motion_forecasts(
    query: MolmoPhysTwinQuery,
    checkpoint: str | Path,
    captions: Mapping[str, str],
    *,
    future_horizon: int = 30,
    device: str = "cuda",
) -> MolmoForecastBundle:
    """Load the released checkpoint once and forecast every language control."""

    if not captions or any(not key or not value.strip() for key, value in captions.items()):
        raise ValueError("captions must map nonempty ids to nonempty text")
    if future_horizon < 1:
        raise ValueError("future_horizon must be positive")
    try:
        import torch
        from PIL import Image
        from molmo_motion import MolmoMotion, MolmoMotionProcessor
    except ImportError as error:
        raise RuntimeError(
            "MolmoMotion inference requires its released environment and package"
        ) from error
    checkpoint_path = str(Path(checkpoint).resolve())
    processor = MolmoMotionProcessor.from_pretrained(checkpoint_path)
    if processor.config.history_size != len(query.history_frame_indices):
        raise ValueError(
            "MolmoMotion checkpoint history size differs from the prepared query"
        )
    model = MolmoMotion.from_pretrained(checkpoint_path)
    if device != "cuda":
        raise ValueError("the released 4B checkpoint runner currently requires CUDA")
    model._internal = model._internal.to(torch.bfloat16).cuda()
    model.eval()
    history_frames = [Image.open(path).convert("RGB") for path in query.image_paths]
    forecast_ids = tuple(captions)
    future_camera = []
    future_world = []
    raw_text = []
    for forecast_id in forecast_ids:
        inputs = processor(
            history_frames=history_frames,
            points_2d_at_t0=torch.as_tensor(query.points_2d_xy, dtype=torch.float32),
            points_3d_history=torch.as_tensor(
                query.points_3d_world_history_m, dtype=torch.float32
            ),
            action=captions[forecast_id],
            future_horizon=future_horizon,
            c2w_at_t0=torch.as_tensor(query.camera_to_world, dtype=torch.float32),
        )
        inputs = {
            key: value.cuda() if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.predict_trajectory(**inputs)
        predicted = output.future_3d.detach().cpu().numpy().astype(np.float32)
        if predicted.shape != (len(query.node_indices), future_horizon, 3):
            raise RuntimeError(
                "MolmoMotion returned an unexpected shape: " + repr(predicted.shape)
            )
        if "<tracks" not in output.future_text:
            raise RuntimeError(
                f"MolmoMotion forecast {forecast_id!r} did not emit a track block"
            )
        future_camera.append(predicted)
        future_world.append(camera_to_world_points(predicted, query.camera_to_world))
        raw_text.append(output.future_text)
    return MolmoForecastBundle(
        query=query,
        forecast_ids=forecast_ids,
        captions=tuple(captions[key] for key in forecast_ids),
        future_camera_m=np.stack(future_camera),
        future_world_m=np.stack(future_world),
        raw_text=tuple(raw_text),
        checkpoint=checkpoint_path,
    )


def save_molmo_forecasts(path: str | Path, bundle: MolmoForecastBundle) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        forecast_ids=np.asarray(bundle.forecast_ids),
        captions=np.asarray(bundle.captions),
        future_camera_m=bundle.future_camera_m.astype(np.float32),
        future_world_m=bundle.future_world_m.astype(np.float32),
        raw_text=np.asarray(bundle.raw_text),
        node_indices=bundle.query.node_indices,
        raw_track_indices=bundle.query.raw_track_indices,
        points_2d_xy=bundle.query.points_2d_xy.astype(np.float32),
        points_3d_world_history_m=bundle.query.points_3d_world_history_m.astype(np.float32),
        camera_to_world=bundle.query.camera_to_world,
        intrinsics=bundle.query.intrinsics,
        source_fps=np.asarray(bundle.query.source_fps),
        forecast_fps=np.asarray(bundle.query.forecast_fps),
        frame_stride=np.asarray(bundle.query.frame_stride),
        camera_index=np.asarray(bundle.query.camera_index),
        t0_frame=np.asarray(bundle.query.t0_frame),
        history_frame_indices=bundle.query.history_frame_indices,
        image_paths=np.asarray([str(path.resolve()) for path in bundle.query.image_paths]),
        raw_case_dir=np.asarray(str(bundle.query.raw_case_dir.resolve())),
        case_name=np.asarray(bundle.query.case_name),
        checkpoint=np.asarray(bundle.checkpoint),
        manifest_json=np.asarray(json.dumps(bundle.metadata(), sort_keys=True)),
    )


def load_molmo_forecasts(path: str | Path) -> MolmoForecastBundle:
    with np.load(path, allow_pickle=False) as archive:
        raw_case_dir = Path(str(archive["raw_case_dir"]))
        if "source_fps" in archive.files:
            source_fps = float(archive["source_fps"])
            forecast_fps = float(archive["forecast_fps"])
            frame_stride = int(archive["frame_stride"])
        else:
            metadata_path = raw_case_dir / "metadata.json"
            source_fps = (
                float(json.loads(metadata_path.read_text(encoding="utf-8"))["fps"])
                if metadata_path.is_file()
                else 1.0
            )
            forecast_fps = source_fps
            frame_stride = 1
        query = MolmoPhysTwinQuery(
            case_name=str(archive["case_name"]),
            raw_case_dir=raw_case_dir,
            camera_index=int(archive["camera_index"]),
            t0_frame=int(archive["t0_frame"]),
            history_frame_indices=np.asarray(archive["history_frame_indices"], dtype=int),
            image_paths=tuple(Path(str(value)) for value in archive["image_paths"]),
            node_indices=np.asarray(archive["node_indices"], dtype=int),
            raw_track_indices=np.asarray(archive["raw_track_indices"], dtype=int),
            points_2d_xy=np.asarray(archive["points_2d_xy"], dtype=float),
            points_3d_world_history_m=np.asarray(
                archive["points_3d_world_history_m"], dtype=float
            ),
            camera_to_world=np.asarray(archive["camera_to_world"], dtype=float),
            intrinsics=np.asarray(archive["intrinsics"], dtype=float),
            source_fps=source_fps,
            forecast_fps=forecast_fps,
            frame_stride=frame_stride,
        )
        return MolmoForecastBundle(
            query=query,
            forecast_ids=tuple(map(str, archive["forecast_ids"])),
            captions=tuple(map(str, archive["captions"])),
            future_camera_m=np.asarray(archive["future_camera_m"], dtype=float),
            future_world_m=np.asarray(archive["future_world_m"], dtype=float),
            raw_text=tuple(map(str, archive["raw_text"])),
            checkpoint=str(archive["checkpoint"]),
        )
