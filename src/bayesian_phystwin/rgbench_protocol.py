"""Outcome-blind contracts for the RGBench online-belief study."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONTRACT = "rgbbench-online-belief-v1"
RGBENCH_COMMIT = "eddae2f28f388b4706d65d626f67bc9e34b14c68"
DATASET_REVISION = "136c00dc5f96b6b3d20427e93875a1c00d7a7cc9"
SPLIT_SALT = b"rgbbench-online-belief-v1\0"
PAPER_GARMENTS = (
    "beige_hoodie",
    "blue_dress",
    "brown_coat",
    "green_tshirt",
    "grey_pleat_skirt",
    "white_cakeskirt",
    "white_shirt",
)
ACTIONS = ("fling", "fold", "grasp")
PRIMARY_SAMPLES = ("01", "02", "03")
EXCLUDED_GARMENTS = ("grey_sunwear", "khaki_blazer")
SOURCE_GARMENTS = ("white_cakeskirt", "brown_coat", "green_tshirt")
CALIBRATION_GARMENTS = ("grey_pleat_skirt", "white_shirt")
TARGET_GARMENTS = ("blue_dress", "beige_hoodie")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _valid_hex(value: str, length: int) -> bool:
    return len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(
        b"rgbbench-online-belief-manifest-v1\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def garment_hash(garment: str) -> str:
    """Return the outcome-independent hash used to order garments."""

    return hashlib.sha256(SPLIT_SALT + garment.encode("utf-8")).hexdigest()


def garment_split(garment: str) -> str:
    """Return the frozen garment-level source/calibration/target split."""

    if garment in SOURCE_GARMENTS:
        return "source"
    if garment in CALIBRATION_GARMENTS:
        return "calibration"
    if garment in TARGET_GARMENTS:
        return "target"
    raise ValueError(f"{garment} is outside the primary RGBench cohort")


def _validate_frozen_hash_partition() -> None:
    ordered = tuple(sorted(PAPER_GARMENTS, key=garment_hash))
    _require(
        ordered[:3] == SOURCE_GARMENTS,
        "source garments differ from the salted hash partition",
    )
    _require(
        ordered[3:5] == CALIBRATION_GARMENTS,
        "calibration garments differ from the salted hash partition",
    )
    _require(
        ordered[5:] == TARGET_GARMENTS,
        "target garments differ from the salted hash partition",
    )


_validate_frozen_hash_partition()


@dataclass(frozen=True)
class RGBenchProtocolConfig:
    """Frozen public-data cohort and causal prefix settings."""

    garments: tuple[str, ...] = PAPER_GARMENTS
    actions: tuple[str, ...] = ACTIONS
    primary_samples: tuple[str, ...] = PRIMARY_SAMPLES
    branch_fraction: float = 0.25
    prefix_fit_fraction: float = 0.60
    minimum_prefix_fit_frames: int = 2
    minimum_prefix_validation_frames: int = 1
    minimum_future_frames: int = 3
    minimum_mesh_vertices: int = 128

    def __post_init__(self) -> None:
        _require(
            self.garments == PAPER_GARMENTS,
            "garments differ from the frozen paper cohort",
        )
        _require(self.actions == ACTIONS, "actions differ from the frozen cohort")
        _require(
            self.primary_samples == PRIMARY_SAMPLES,
            "primary samples differ from the published three-sample cells",
        )
        _require(
            0.0 < self.branch_fraction < 1.0,
            "branch_fraction must lie in (0, 1)",
        )
        _require(
            0.0 < self.prefix_fit_fraction < 1.0,
            "prefix_fit_fraction must lie in (0, 1)",
        )
        _require(
            self.minimum_prefix_fit_frames >= 2
            and self.minimum_prefix_validation_frames >= 1
            and self.minimum_future_frames >= 1,
            "prefix and future frame minima must be positive",
        )
        _require(
            self.minimum_mesh_vertices >= 128,
            "physical backend minimum must be at least 128 vertices",
        )


@dataclass(frozen=True)
class RGBenchCaseManifest:
    """One RGBench capture described without parsing point coordinates."""

    case_id: str
    garment: str
    action: str
    sample: str
    split: str
    data_subfolder: str
    mesh_relative_path: str
    mesh_vertex_count: int
    mesh_face_count: int
    mesh_sha256: str
    left_trajectory_sha256: str
    right_trajectory_sha256: str
    calibration_sha256: str
    camera_delay_s: float
    start_calculate_time_s: float
    end_calculate_time_s: float
    master_start_time_s: float
    evaluation_frame_count: int
    branch_index: int
    fit_stop_index: int
    point_cloud_name_sha256: str
    point_cloud_byte_count: int

    def __post_init__(self) -> None:
        _require(self.case_id == f"{self.garment}/{self.action}/{self.sample}", "bad case ID")
        _require(self.garment in PAPER_GARMENTS, "unknown garment")
        _require(self.action in ACTIONS, "unknown action")
        _require(self.sample in PRIMARY_SAMPLES, "unknown primary sample")
        _require(self.split == garment_split(self.garment), "case split changed")
        _require(bool(self.data_subfolder), "data subfolder is empty")
        _require(bool(self.mesh_relative_path), "mesh path is empty")
        _require(self.mesh_vertex_count >= 128, "mesh has too few vertices")
        _require(self.mesh_face_count >= 1, "mesh has no faces")
        for value in (
            self.mesh_sha256,
            self.left_trajectory_sha256,
            self.right_trajectory_sha256,
            self.calibration_sha256,
            self.point_cloud_name_sha256,
        ):
            _require(_valid_hex(value, 64), "case contains an invalid SHA-256")
        for value in (
            self.camera_delay_s,
            self.start_calculate_time_s,
            self.end_calculate_time_s,
            self.master_start_time_s,
        ):
            _require(math.isfinite(value), "case timing is non-finite")
        _require(
            self.end_calculate_time_s > self.start_calculate_time_s,
            "evaluation interval is empty",
        )
        _require(
            0 < self.fit_stop_index <= self.branch_index
            < self.evaluation_frame_count - 1,
            "case information boundary is invalid",
        )
        _require(self.point_cloud_byte_count > 0, "point clouds have no bytes")


@dataclass(frozen=True)
class RGBenchDatasetManifest:
    """Immutable target-free inventory of the RGBench primary cohort."""

    config: RGBenchProtocolConfig
    rgbbench_commit: str
    dataset_revision: str
    experiment_library_sha256: str
    paper_baselines_sha256: str
    cases: tuple[RGBenchCaseManifest, ...]
    artifact_sha256: str

    def __post_init__(self) -> None:
        _require(self.rgbbench_commit == RGBENCH_COMMIT, "RGBench commit changed")
        _require(
            self.dataset_revision == DATASET_REVISION,
            "RGBench dataset revision changed",
        )
        _require(
            _valid_hex(self.experiment_library_sha256, 64),
            "experiment library digest is invalid",
        )
        _require(
            _valid_hex(self.paper_baselines_sha256, 64),
            "paper baseline digest is invalid",
        )
        expected_count = (
            len(PAPER_GARMENTS) * len(ACTIONS) * len(PRIMARY_SAMPLES)
        )
        _require(len(self.cases) == expected_count, "case count changed")
        case_ids = tuple(case.case_id for case in self.cases)
        _require(
            case_ids == tuple(sorted(case_ids)) and len(case_ids) == len(set(case_ids)),
            "case IDs are not unique and sorted",
        )
        split_counts = {
            split: sum(case.split == split for case in self.cases)
            for split in ("source", "calibration", "target")
        }
        _require(
            split_counts == {"source": 27, "calibration": 18, "target": 18},
            "garment-level split counts changed",
        )
        _require(
            _canonical_sha256(self.descriptor()) == self.artifact_sha256,
            "manifest digest changed",
        )

    def descriptor(self) -> dict[str, Any]:
        return _manifest_descriptor(
            config=self.config,
            rgbbench_commit=self.rgbbench_commit,
            dataset_revision=self.dataset_revision,
            experiment_library_sha256=self.experiment_library_sha256,
            paper_baselines_sha256=self.paper_baselines_sha256,
            cases=self.cases,
            artifact_sha256=self.artifact_sha256,
        )


def _manifest_descriptor(
    *,
    config: RGBenchProtocolConfig,
    rgbbench_commit: str,
    dataset_revision: str,
    experiment_library_sha256: str,
    paper_baselines_sha256: str,
    cases: tuple[RGBenchCaseManifest, ...],
    artifact_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_kind": "RGBenchDatasetManifest",
        "contract": CONTRACT,
        "config": asdict(config),
        "rgbbench_commit": rgbbench_commit,
        "dataset_revision": dataset_revision,
        "experiment_library_sha256": experiment_library_sha256,
        "paper_baselines_sha256": paper_baselines_sha256,
        "garment_hashes": {
            garment: garment_hash(garment) for garment in PAPER_GARMENTS
        },
        "garment_splits": {
            "source": list(SOURCE_GARMENTS),
            "calibration": list(CALIBRATION_GARMENTS),
            "target": list(TARGET_GARMENTS),
        },
        "excluded_garments": {
            garment: "no published baseline; released mesh is non-manifold"
            for garment in EXCLUDED_GARMENTS
        },
        "cases": [asdict(case) for case in cases],
        "information_boundary": {
            "point_coordinates_read": False,
            "point_cloud_metrics_read": False,
            "point_cloud_filenames_and_sizes_read": True,
            "robot_trajectory_timestamps_read": True,
            "robot_trajectory_outcomes": "known intervention, not target state",
            "physical_mesh_coordinates_read": False,
            "primary_comparison_samples": list(PRIMARY_SAMPLES),
            "additional_released_samples_excluded": True,
            "calibration_requires_source_gate": True,
            "target_requires_calibration_gate": True,
        },
        "artifact_sha256": artifact_sha256,
    }


def _first_csv_time(path: Path) -> float:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames is not None and "time" in reader.fieldnames, f"{path} has no time column")
        first = next(reader, None)
    _require(first is not None, f"{path} has no rows")
    value = float(first["time"])
    _require(math.isfinite(value), f"{path} has a non-finite first timestamp")
    return value


def _mesh_counts(path: Path) -> tuple[int, int]:
    vertices = 0
    faces = 0
    with path.open("rb") as stream:
        for raw_line in stream:
            if raw_line.startswith(b"v "):
                vertices += 1
            elif raw_line.startswith(b"f "):
                faces += 1
    return vertices, faces


def _prefix_boundary(
    frame_count: int,
    config: RGBenchProtocolConfig,
) -> tuple[int, int]:
    prefix_count = max(
        math.ceil(frame_count * config.branch_fraction),
        config.minimum_prefix_fit_frames
        + config.minimum_prefix_validation_frames,
    )
    prefix_count = min(prefix_count, frame_count - config.minimum_future_frames)
    _require(
        prefix_count
        >= config.minimum_prefix_fit_frames
        + config.minimum_prefix_validation_frames,
        "case is too short for the frozen prefix and future",
    )
    fit_count = max(
        math.floor(prefix_count * config.prefix_fit_fraction),
        config.minimum_prefix_fit_frames,
    )
    fit_count = min(
        fit_count,
        prefix_count - config.minimum_prefix_validation_frames,
    )
    return prefix_count - 1, fit_count


def _case_manifest(
    dataset_root: Path,
    garment: str,
    action: str,
    sample: str,
    experiment: dict[str, Any],
    mesh_relative_path: str,
    config: RGBenchProtocolConfig,
) -> RGBenchCaseManifest:
    case_id = f"{garment}/{action}/{sample}"
    capture = dataset_root / str(experiment["data_subfolder"])
    _require(capture.is_dir(), f"missing capture directory: {case_id}")
    left = capture / "joints/left_arm_joint_states_and_end_pose.csv"
    right = capture / "joints/right_arm_joint_states_and_end_pose.csv"
    calibration = capture / "calibration/world_to_camera_transform.json"
    cloud_dir = capture / "segment_pcds"
    for path in (left, right, calibration):
        _require(path.is_file() and path.stat().st_size > 0, f"missing required stream: {path}")
    _require(cloud_dir.is_dir(), f"missing point-cloud directory: {case_id}")
    point_paths = tuple(sorted(cloud_dir.glob("pointcloud_*_segmented.pcd")))
    _require(len(point_paths) >= 1, f"{case_id} has no point clouds")
    unexpected = [
        path.name
        for path in cloud_dir.iterdir()
        if not path.is_file() or path not in point_paths
    ]
    _require(not unexpected, f"{case_id} has unexpected point-cloud entries")
    absolute_times: list[float] = []
    for path in point_paths:
        timestamp = path.name.removeprefix("pointcloud_").removesuffix(
            "_segmented.pcd"
        )
        value = float(timestamp)
        _require(math.isfinite(value), f"{case_id} has a non-finite timestamp")
        absolute_times.append(value)
    _require(
        absolute_times == sorted(absolute_times),
        f"{case_id} point-cloud filenames are not timestamp sorted",
    )
    sizes = [path.stat().st_size for path in point_paths]
    _require(all(size > 0 for size in sizes), f"{case_id} has an empty point cloud")

    master_start = min(_first_csv_time(left), _first_csv_time(right))
    camera_delay = float(experiment["camera_delay"])
    evaluation = experiment["evaluate"]
    start = float(evaluation["start_calculate_time"])
    end = float(evaluation["end_calculate_time"])
    selected_names = [
        path.name
        for path, absolute_time in zip(point_paths, absolute_times, strict=True)
        if start - 1e-9
        <= absolute_time - master_start + camera_delay
        <= end + 1e-9
    ]
    _require(selected_names, f"{case_id} has no evaluation point clouds")
    branch_index, fit_stop_index = _prefix_boundary(len(selected_names), config)

    mesh = dataset_root / "meshes" / mesh_relative_path
    _require(mesh.is_file() and mesh.stat().st_size > 0, f"missing mesh: {mesh}")
    mesh_vertex_count, mesh_face_count = _mesh_counts(mesh)
    _require(
        mesh_vertex_count >= config.minimum_mesh_vertices,
        f"{case_id} mesh violates the physical backend minimum",
    )
    _require(mesh_face_count >= 1, f"{case_id} mesh has no faces")
    point_name_digest = hashlib.sha256(
        b"rgbbench-evaluation-point-cloud-names-v1\0"
        + "\n".join(selected_names).encode("ascii")
    ).hexdigest()
    selected_set = set(selected_names)
    selected_bytes = sum(
        size
        for path, size in zip(point_paths, sizes, strict=True)
        if path.name in selected_set
    )
    return RGBenchCaseManifest(
        case_id=case_id,
        garment=garment,
        action=action,
        sample=sample,
        split=garment_split(garment),
        data_subfolder=str(experiment["data_subfolder"]),
        mesh_relative_path=mesh_relative_path,
        mesh_vertex_count=mesh_vertex_count,
        mesh_face_count=mesh_face_count,
        mesh_sha256=_sha256(mesh),
        left_trajectory_sha256=_sha256(left),
        right_trajectory_sha256=_sha256(right),
        calibration_sha256=_sha256(calibration),
        camera_delay_s=camera_delay,
        start_calculate_time_s=start,
        end_calculate_time_s=end,
        master_start_time_s=master_start,
        evaluation_frame_count=len(selected_names),
        branch_index=branch_index,
        fit_stop_index=fit_stop_index,
        point_cloud_name_sha256=point_name_digest,
        point_cloud_byte_count=int(selected_bytes),
    )


def build_rgbbench_dataset_manifest(
    dataset_root: str | Path,
    benchmark_root: str | Path,
    *,
    experiment_library: dict[str, Any],
    mesh_relative_paths: dict[str, str],
    dataset_revision: str = DATASET_REVISION,
    config: RGBenchProtocolConfig | None = None,
) -> RGBenchDatasetManifest:
    """Inventory the primary benchmark without parsing a point coordinate."""

    cfg = config or RGBenchProtocolConfig()
    root = Path(dataset_root)
    benchmark = Path(benchmark_root)
    _require(root.is_dir(), "RGBench dataset root does not exist")
    _require(benchmark.is_dir(), "RGBench checkout does not exist")
    _require(dataset_revision == DATASET_REVISION, "dataset revision changed")
    experiments = experiment_library.get("experiments")
    _require(isinstance(experiments, dict), "experiment library has no experiments")
    _require(
        set(mesh_relative_paths) == set(PAPER_GARMENTS),
        "mesh map differs from the primary garment cohort",
    )
    cases: list[RGBenchCaseManifest] = []
    for garment in PAPER_GARMENTS:
        garment_experiments = experiments.get(garment)
        _require(isinstance(garment_experiments, dict), f"missing {garment}")
        for action in ACTIONS:
            action_experiments = garment_experiments.get(action, {}).get("piper")
            _require(isinstance(action_experiments, dict), f"missing {garment}/{action}")
            for sample in PRIMARY_SAMPLES:
                experiment = action_experiments.get(sample)
                _require(
                    isinstance(experiment, dict),
                    f"missing primary case {garment}/{action}/{sample}",
                )
                cases.append(
                    _case_manifest(
                        root,
                        garment,
                        action,
                        sample,
                        experiment,
                        mesh_relative_paths[garment],
                        cfg,
                    )
                )
    cases_tuple = tuple(sorted(cases, key=lambda case: case.case_id))
    experiment_library_path = benchmark / "configs/experiment_library.yaml"
    paper_baselines_path = benchmark / "results/paper_baselines.csv"
    _require(experiment_library_path.is_file(), "experiment library is missing")
    _require(paper_baselines_path.is_file(), "paper baselines are missing")
    descriptor = _manifest_descriptor(
        config=cfg,
        rgbbench_commit=RGBENCH_COMMIT,
        dataset_revision=dataset_revision,
        experiment_library_sha256=_sha256(experiment_library_path),
        paper_baselines_sha256=_sha256(paper_baselines_path),
        cases=cases_tuple,
        artifact_sha256="0" * 64,
    )
    descriptor["artifact_sha256"] = _canonical_sha256(descriptor)
    return RGBenchDatasetManifest(
        config=cfg,
        rgbbench_commit=RGBENCH_COMMIT,
        dataset_revision=dataset_revision,
        experiment_library_sha256=descriptor["experiment_library_sha256"],
        paper_baselines_sha256=descriptor["paper_baselines_sha256"],
        cases=cases_tuple,
        artifact_sha256=descriptor["artifact_sha256"],
    )
