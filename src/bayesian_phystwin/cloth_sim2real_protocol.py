"""Outcome-blind dataset contracts for the Cloth Sim2Real benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONTRACT = "cloth-sim2real-online-belief-v1"
EXPECTED_CLOTHS = ("chequered_rag", "cotton_rag", "linen_rag")
EXPECTED_TASKS = ("dynamic", "quasi_static")
SPLIT_BY_REPEAT = {0: "source", 1: "calibration", 2: "target"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(
        b"cloth-sim2real-online-belief-v1\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ClothSim2RealProtocolConfig:
    """Frozen source/calibration/target partition and structural checks."""

    cloths: tuple[str, ...] = EXPECTED_CLOTHS
    tasks: tuple[str, ...] = EXPECTED_TASKS
    source_repeat: int = 0
    calibration_repeat: int = 1
    target_repeat: int = 2
    minimum_dynamic_frames: int = 80
    minimum_quasi_static_frames: int = 350
    branch_fraction: float = 0.25
    prefix_fit_fraction: float = 0.60
    minimum_prefix_fit_frames: int = 8
    minimum_prefix_validation_frames: int = 4

    def __post_init__(self) -> None:
        _require(
            self.cloths == EXPECTED_CLOTHS,
            "cloth identities differ from the frozen benchmark contract",
        )
        _require(
            self.tasks == EXPECTED_TASKS,
            "task identities differ from the frozen benchmark contract",
        )
        repeats = (
            self.source_repeat,
            self.calibration_repeat,
            self.target_repeat,
        )
        _require(
            repeats == (0, 1, 2) and len(set(repeats)) == 3,
            "repeat split differs from the frozen benchmark contract",
        )
        _require(
            self.minimum_dynamic_frames >= 1
            and self.minimum_quasi_static_frames >= 1,
            "minimum frame counts must be positive",
        )
        _require(
            0.0 < self.branch_fraction < 1.0,
            "branch fraction must lie in (0, 1)",
        )
        _require(
            0.0 < self.prefix_fit_fraction < 1.0,
            "prefix-fit fraction must lie in (0, 1)",
        )
        _require(
            self.minimum_prefix_fit_frames >= 1
            and self.minimum_prefix_validation_frames >= 1,
            "prefix partitions must be nonempty",
        )


@dataclass(frozen=True)
class ClothSim2RealCaseManifest:
    """One trial/task carrier described without opening point coordinates."""

    case_id: str
    cloth_id: str
    repeat_index: int
    split: str
    task: str
    frame_count: int
    branch_frame: int
    fit_stop_frame: int
    frame_name_sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        _require(bool(self.case_id.strip()), "case ID is empty")
        _require(self.cloth_id in EXPECTED_CLOTHS, "unknown cloth identity")
        _require(
            self.repeat_index in SPLIT_BY_REPEAT,
            "repeat index is outside the frozen split",
        )
        _require(
            self.split == SPLIT_BY_REPEAT[self.repeat_index],
            "case split and repeat index disagree",
        )
        _require(self.task in EXPECTED_TASKS, "unknown cloth task")
        _require(self.frame_count >= 1, "case has no frames")
        _require(
            0 < self.fit_stop_frame < self.branch_frame < self.frame_count,
            "case information boundary is invalid",
        )
        _require(_valid_sha256(self.frame_name_sha256), "frame-name digest is invalid")
        _require(self.byte_count > 0, "case has no file bytes")


@dataclass(frozen=True)
class ClothSim2RealDatasetManifest:
    """Immutable target-free inventory of the public benchmark archive."""

    config: ClothSim2RealProtocolConfig
    dataset_doi: str
    archive_sha256: str
    cases: tuple[ClothSim2RealCaseManifest, ...]
    artifact_sha256: str

    def __post_init__(self) -> None:
        _require(
            self.dataset_doi == "10.5281/zenodo.13823986",
            "dataset DOI differs from the frozen release",
        )
        _require(_valid_sha256(self.archive_sha256), "archive digest is invalid")
        expected_count = len(EXPECTED_CLOTHS) * len(EXPECTED_TASKS) * len(SPLIT_BY_REPEAT)
        _require(len(self.cases) == expected_count, "dataset case count changed")
        case_ids = tuple(case.case_id for case in self.cases)
        _require(
            case_ids == tuple(sorted(case_ids)) and len(set(case_ids)) == len(case_ids),
            "dataset cases are not unique and sorted",
        )
        split_counts = {
            split: sum(case.split == split for case in self.cases)
            for split in SPLIT_BY_REPEAT.values()
        }
        _require(
            split_counts == {"source": 6, "calibration": 6, "target": 6},
            "dataset split counts changed",
        )
        _require(
            _canonical_sha256(self.descriptor()) == self.artifact_sha256,
            "dataset manifest digest changed",
        )

    def descriptor(self) -> dict[str, Any]:
        return _manifest_descriptor(
            config=self.config,
            dataset_doi=self.dataset_doi,
            archive_sha256=self.archive_sha256,
            cases=self.cases,
            artifact_sha256=self.artifact_sha256,
        )


def _manifest_descriptor(
    *,
    config: ClothSim2RealProtocolConfig,
    dataset_doi: str,
    archive_sha256: str,
    cases: tuple[ClothSim2RealCaseManifest, ...],
    artifact_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_kind": "ClothSim2RealDatasetManifest",
        "contract": CONTRACT,
        "config": asdict(config),
        "dataset_doi": dataset_doi,
        "archive_sha256": archive_sha256,
        "cases": [asdict(case) for case in cases],
        "information_boundary": {
            "point_coordinates_read": False,
            "point_cloud_metrics_read": False,
            "frame_names_and_sizes_only": True,
            "source_repeat": config.source_repeat,
            "calibration_repeat": config.calibration_repeat,
            "target_repeat": config.target_repeat,
            "target_future_must_remain_sealed_until_gate_passes": True,
        },
        "artifact_sha256": artifact_sha256,
    }


def _frame_boundary(
    frame_count: int,
    config: ClothSim2RealProtocolConfig,
) -> tuple[int, int]:
    branch = int(frame_count * config.branch_fraction)
    branch = max(
        branch,
        config.minimum_prefix_fit_frames + config.minimum_prefix_validation_frames,
    )
    branch = min(branch, frame_count - 1)
    fit_stop = int(branch * config.prefix_fit_fraction)
    fit_stop = max(fit_stop, config.minimum_prefix_fit_frames)
    fit_stop = min(
        fit_stop,
        branch - config.minimum_prefix_validation_frames,
    )
    _require(
        0 < fit_stop < branch < frame_count,
        "case is too short for the frozen prefix partition",
    )
    return branch, fit_stop


def _case_manifest(
    dataset_root: Path,
    cloth_id: str,
    repeat_index: int,
    task: str,
    config: ClothSim2RealProtocolConfig,
) -> ClothSim2RealCaseManifest:
    case_id = f"{cloth_id}_{repeat_index}/{task}"
    cloud_dir = dataset_root / f"{cloth_id}_{repeat_index}" / task / "cloud"
    _require(cloud_dir.is_dir(), f"missing point-cloud directory: {case_id}")
    unexpected = sorted(
        path.name
        for path in cloud_dir.iterdir()
        if not path.is_file() or path.suffix != ".ply"
    )
    _require(not unexpected, f"unexpected point-cloud entries in {case_id}: {unexpected}")
    frame_paths = sorted(cloud_dir.glob("*.ply"))
    minimum = (
        config.minimum_dynamic_frames
        if task == "dynamic"
        else config.minimum_quasi_static_frames
    )
    _require(
        len(frame_paths) >= minimum,
        f"{case_id} has {len(frame_paths)} frames, expected at least {minimum}",
    )
    expected_names = [f"{index:05d}.ply" for index in range(len(frame_paths))]
    actual_names = [path.name for path in frame_paths]
    _require(
        actual_names == expected_names,
        f"{case_id} point-cloud frames are not contiguous from zero",
    )
    sizes = [path.stat().st_size for path in frame_paths]
    _require(all(size > 0 for size in sizes), f"{case_id} contains an empty PLY file")
    frame_name_sha256 = hashlib.sha256(
        b"cloth-sim2real-frame-names-v1\0"
        + "\n".join(actual_names).encode("ascii")
    ).hexdigest()
    branch_frame, fit_stop_frame = _frame_boundary(len(frame_paths), config)
    return ClothSim2RealCaseManifest(
        case_id=case_id,
        cloth_id=cloth_id,
        repeat_index=repeat_index,
        split=SPLIT_BY_REPEAT[repeat_index],
        task=task,
        frame_count=len(frame_paths),
        branch_frame=branch_frame,
        fit_stop_frame=fit_stop_frame,
        frame_name_sha256=frame_name_sha256,
        byte_count=int(sum(sizes)),
    )


def build_cloth_sim2real_dataset_manifest(
    dataset_root: str | Path,
    *,
    archive_sha256: str,
    config: ClothSim2RealProtocolConfig | None = None,
) -> ClothSim2RealDatasetManifest:
    """Inventory the public archive without parsing any point coordinate."""

    cfg = config or ClothSim2RealProtocolConfig()
    root = Path(dataset_root)
    if (root / "Benchmarking_cloth").is_dir():
        root = root / "Benchmarking_cloth"
    _require(root.is_dir(), "cloth benchmark root does not exist")
    expected_trial_dirs = {
        f"{cloth_id}_{repeat_index}"
        for cloth_id in cfg.cloths
        for repeat_index in SPLIT_BY_REPEAT
    }
    actual_trial_dirs = {
        path.name for path in root.iterdir() if path.is_dir()
    }
    _require(
        actual_trial_dirs == expected_trial_dirs,
        "cloth benchmark trial directories changed",
    )
    cases = tuple(
        sorted(
            (
                _case_manifest(root, cloth_id, repeat_index, task, cfg)
                for cloth_id in cfg.cloths
                for repeat_index in SPLIT_BY_REPEAT
                for task in cfg.tasks
            ),
            key=lambda case: case.case_id,
        )
    )
    descriptor = _manifest_descriptor(
        config=cfg,
        dataset_doi="10.5281/zenodo.13823986",
        archive_sha256=archive_sha256,
        cases=cases,
        artifact_sha256="0" * 64,
    )
    descriptor["artifact_sha256"] = _canonical_sha256(descriptor)
    return ClothSim2RealDatasetManifest(
        config=cfg,
        dataset_doi="10.5281/zenodo.13823986",
        archive_sha256=archive_sha256,
        cases=cases,
        artifact_sha256=descriptor["artifact_sha256"],
    )
