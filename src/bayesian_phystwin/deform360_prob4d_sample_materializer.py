"""Materialize source-only Prob4D calibration samples from public Deform360.

The materializer consumes integrity-bound causal MotionCrafter windows and
metric point grids reconstructed from the matching released Deform360 prefix.
It never accepts confirmation objects, reserved future frames, or target
outcomes. Dense rows are clustered across cameras and overlapping windows
before calibration so duplicated correlated evidence cannot create arbitrary
confidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import (
    canonical_relative_posix_path,
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import (
    content_id,
    exact_revision,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
)
from .deform360_calibration_visual_production import (
    validate_deform360_calibration_visual_prediction_seal,
    validate_deform360_calibration_visual_production_result,
)
from .deform360_prob4d_camera_eligibility import (
    SUPPORT_NEGATIVE_REASON,
    VISIBLE_STREAM_PLAN_SEMANTICS,
    VISIBLE_STREAM_PLAN_VERSION,
    validate_deform360_prob4d_camera_eligibility_policy,
)
from .deform360_prob4d_source_calibration import (
    POINT_CLUSTER_SEMANTICS,
    SAMPLE_SCHEMA,
    SAMPLE_SEMANTICS,
    SAMPLE_VERSION,
    Prob4DCalibrationApi,
    load_deform360_prob4d_calibration_samples,
)

PLAN_SCHEMA: Final = "bayesian-phystwin.deform360-prob4d-metric-prefix-plan"
PLAN_VERSION: Final = 1
PLAN_SEMANTICS: Final = (
    "all-successful-integrity-bound-streams-with-causal-public-metric-prefix-v1"
)
METRIC_PREFIX_ARRAYS: Final = frozenset(
    {"frame_indices", "points_world_m", "valid_mask"}
)
METRIC_SOURCE_KIND: Final = "released-deform360-robot-taxel-gauge-v1"
COORDINATE_FRAME: Final = "deform360-world"

_PLAN_V1_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "plan_id",
        "protocol_id",
        "selection_file_sha256",
        "visual_provider_spec_file_sha256",
        "metric_prior_policy_file_sha256",
        "dataset_revision",
        "processing_revision",
        "prob4d_revision",
        "motioncrafter_revision",
        "visual_production_result_id",
        "cases",
        "information_boundary",
        "claim_boundary",
    }
)
_PLAN_V2_FIELDS = _PLAN_V1_FIELDS | frozenset(
    {
        "camera_eligibility_policy_file_sha256",
        "camera_eligibility_policy_id",
        "excluded_streams",
    }
)
_CASE_FIELDS = frozenset(
    {
        "case_id",
        "object_id",
        "episode_id",
        "stratum",
        "causal_frame_range_half_open",
        "streams",
    }
)
_STREAM_FIELDS = frozenset(
    {
        "job_id",
        "camera_id",
        "prediction_manifest",
        "metric_prefix",
        "metric_calibration",
    }
)
_EXCLUDED_STREAM_FIELDS = frozenset(
    {
        "job_id",
        "object_id",
        "episode_id",
        "stratum",
        "camera_id",
        "reason",
    }
)
_FILE_FIELDS = frozenset({"path", "sha256", "byte_count"})
_BOUNDARY_FIELDS = frozenset(
    {
        "calibration_payloads_opened",
        "confirmation_payloads_opened",
        "target_outcomes_used",
        "future_frames_used",
        "replacement_allowed",
    }
)
_EXPECTED_BOUNDARY: Final = {
    "calibration_payloads_opened": True,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "future_frames_used": False,
    "replacement_allowed": False,
}


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _confined_file(root: Path, relative: object, *, name: str) -> Path:
    normalized = canonical_relative_posix_path(relative, name=name)
    candidate = root / PurePosixPath(normalized)
    _require(not candidate.is_symlink(), f"{name} must not be a symbolic link")
    path = candidate.resolve(strict=True)
    _require(root == path or root in path.parents, f"{name} escapes its root")
    _require(
        path.is_file() and not path.is_symlink(), f"{name} is not an ordinary file"
    )
    return path


def _file_record(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a file record")
    require_exact_fields(value, expected=_FILE_FIELDS, name=name)
    return {
        "path": canonical_relative_posix_path(value["path"], name=f"{name}.path"),
        "sha256": sha256_digest(value["sha256"], name=f"{name}.sha256"),
        "byte_count": genuine_integer(
            value["byte_count"], name=f"{name}.byte_count", minimum=1
        ),
    }


def _verify_record(
    root: Path, value: object, *, name: str
) -> tuple[Path, dict[str, object]]:
    record = _file_record(value, name=name)
    path = _confined_file(root, record["path"], name=f"{name}.path")
    _require(path.stat().st_size == record["byte_count"], f"{name} byte count changed")
    _require(_sha256_file(path) == record["sha256"], f"{name} SHA-256 changed")
    return path, record


def _validate_boundary(value: object) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise ValueError("information_boundary must be a JSON object")
    require_exact_fields(value, expected=_BOUNDARY_FIELDS, name="information_boundary")
    result: dict[str, bool] = {}
    for key, expected in _EXPECTED_BOUNDARY.items():
        observed = genuine_boolean(value[key], name=f"information_boundary.{key}")
        _require(observed is expected, f"information boundary changed: {key}")
        result[key] = observed
    return result


@dataclass(frozen=True)
class Deform360Prob4DMaterializationConfig:
    """Frozen source-only geometry and sampling settings."""

    covariance_cluster_size_pixels: int = 32
    maximum_metric_fit_correspondences: int = 100_000
    maximum_point_rows_per_window: int = 4_096
    minimum_point_rows_per_window: int = 32

    def __post_init__(self) -> None:
        for field, minimum in (
            ("covariance_cluster_size_pixels", 1),
            ("maximum_metric_fit_correspondences", 8),
            ("maximum_point_rows_per_window", 1),
            ("minimum_point_rows_per_window", 1),
        ):
            object.__setattr__(
                self,
                field,
                genuine_integer(getattr(self, field), name=field, minimum=minimum),
            )
        _require(
            self.minimum_point_rows_per_window <= self.maximum_point_rows_per_window,
            "minimum point rows exceed the per-window maximum",
        )


DEFAULT_MATERIALIZATION_CONFIG = Deform360Prob4DMaterializationConfig()


@dataclass(frozen=True)
class _MetricPrefix:
    frame_indices: np.ndarray
    points_world_m: np.ndarray
    valid_mask: np.ndarray


def _load_metric_prefix(
    path: Path,
    *,
    causal_range: tuple[int, int],
    image_resolution: tuple[int, int],
) -> _MetricPrefix:
    try:
        with np.load(path, allow_pickle=False) as archive:
            _require(
                set(archive.files) == METRIC_PREFIX_ARRAYS,
                "metric-prefix array names changed",
            )
            frame_indices = np.asarray(archive["frame_indices"])
            points = np.asarray(archive["points_world_m"], dtype=np.float64)
            valid = np.asarray(archive["valid_mask"])
    except (OSError, ValueError) as error:
        raise ValueError("cannot load metric-prefix archive") from error
    start, stop = causal_range
    expected_frames: np.ndarray = np.arange(start, stop, dtype=np.int64)
    _require(
        frame_indices.dtype.kind in "iu"
        and np.array_equal(frame_indices, expected_frames),
        "metric prefix does not contain exactly the causal frame range",
    )
    expected_shape = (len(expected_frames), *image_resolution)
    _require(
        points.shape == (*expected_shape, 3),
        "metric points do not match the frozen causal image grid",
    )
    _require(
        valid.dtype.kind == "b" and valid.shape == expected_shape,
        "metric valid mask does not match the frozen causal image grid",
    )
    _require(np.all(np.isfinite(points[valid])), "valid metric points are not finite")
    return _MetricPrefix(
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        points_world_m=points,
        valid_mask=np.asarray(valid, dtype=np.bool_),
    )


def _load_plan(
    path: Path,
    *,
    selection_path: Path,
    visual_provider_spec_path: Path,
    metric_prior_policy_path: Path,
    camera_eligibility_policy_path: Path | None,
) -> dict[str, Any]:
    plan = _load_json(path.resolve(strict=True), name="metric-prefix plan")
    _require(plan.get("schema") == PLAN_SCHEMA, "unsupported metric-prefix plan")
    version = genuine_integer(
        plan.get("schema_version"), name="metric-prefix plan version", minimum=1
    )
    if version == PLAN_VERSION:
        require_exact_fields(plan, expected=_PLAN_V1_FIELDS, name="metric-prefix plan")
        _require(
            plan["semantics"] == PLAN_SEMANTICS,
            "unsupported metric-prefix plan contract",
        )
        _require(
            camera_eligibility_policy_path is None,
            "version-1 plan cannot use a camera eligibility policy",
        )
    elif version == VISIBLE_STREAM_PLAN_VERSION:
        require_exact_fields(plan, expected=_PLAN_V2_FIELDS, name="metric-prefix plan")
        _require(
            plan["semantics"] == VISIBLE_STREAM_PLAN_SEMANTICS,
            "unsupported metric-prefix plan contract",
        )
        _require(
            camera_eligibility_policy_path is not None,
            "version-2 plan requires a camera eligibility policy",
        )
        eligibility_path = cast(Path, camera_eligibility_policy_path).resolve(
            strict=True
        )
        policy = validate_deform360_prob4d_camera_eligibility_policy(
            _load_json(eligibility_path, name="camera eligibility policy")
        )
        _require(
            _sha256_file(eligibility_path)
            == sha256_digest(
                plan["camera_eligibility_policy_file_sha256"],
                name="camera_eligibility_policy_file_sha256",
            )
            and plan["camera_eligibility_policy_id"] == policy["artifact_id"],
            "camera eligibility policy differs from the metric-prefix plan",
        )
    else:
        raise ValueError("unsupported metric-prefix plan contract")
    declared_id = sha256_digest(plan["plan_id"], name="plan_id")
    _require(
        declared_id
        == content_id({key: item for key, item in plan.items() if key != "plan_id"}),
        "metric-prefix plan ID changed",
    )
    for source, field in (
        (selection_path, "selection_file_sha256"),
        (visual_provider_spec_path, "visual_provider_spec_file_sha256"),
        (metric_prior_policy_path, "metric_prior_policy_file_sha256"),
    ):
        _require(
            _sha256_file(source.resolve(strict=True))
            == sha256_digest(plan[field], name=field),
            f"{field} changed",
        )
    for field in (
        "dataset_revision",
        "processing_revision",
        "prob4d_revision",
        "motioncrafter_revision",
    ):
        plan[field] = exact_revision(plan[field], name=field)
    plan["protocol_id"] = nonempty_string(plan["protocol_id"], name="protocol_id")
    plan["visual_production_result_id"] = sha256_digest(
        plan["visual_production_result_id"], name="visual_production_result_id"
    )
    plan["information_boundary"] = _validate_boundary(plan["information_boundary"])
    nonempty_string(plan["claim_boundary"], name="claim_boundary")
    return cast(dict[str, Any], plain_json(plan))


def _deterministic_rows(count: int, maximum: int, *, seed_text: str) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(count, size=maximum, replace=False))


def _spatial_clusters(
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    width: int,
    cluster_size: int,
) -> np.ndarray:
    tile_columns = int(np.ceil(width / cluster_size))
    return np.asarray(
        rows // cluster_size * tile_columns + columns // cluster_size,
        dtype=np.int64,
    )


def _fit_window_metric_gauge(
    *,
    api: Any,
    window: Any,
    metric: _MetricPrefix,
    config: Deform360Prob4DMaterializationConfig,
    seed_text: str,
) -> Any:
    frame = int(window.frame_indices[0])
    metric_index = frame - int(metric.frame_indices[0])
    _require(
        0 <= metric_index < len(metric.frame_indices),
        "window starts outside metric prefix",
    )
    active = window.valid_mask[0] & metric.valid_mask[metric_index]
    rows, columns = np.nonzero(active)
    _require(len(rows) >= 8, "metric gauge has too few supported correspondences")
    selected = _deterministic_rows(
        len(rows),
        config.maximum_metric_fit_correspondences,
        seed_text=seed_text,
    )
    rows = rows[selected]
    columns = columns[selected]
    clusters = _spatial_clusters(
        rows,
        columns,
        width=window.shape[2],
        cluster_size=config.covariance_cluster_size_pixels,
    )
    _require(
        np.unique(clusters).size >= 8,
        "metric gauge has fewer than eight independent spatial clusters",
    )
    result = api.estimate_sim3_robust(
        window.point_map[0, rows, columns],
        metric.points_world_m[metric_index, rows, columns],
        covariance_cluster_ids=clusters,
    )
    _require(
        getattr(result, "covariance_method", "") == "frame_spatial_cluster_robust_v1",
        "metric gauge did not use cluster-robust covariance",
    )
    return result


def _load_prediction_windows(
    *,
    api: Any,
    manifest_path: Path,
    seal: Mapping[str, Any],
    causal_range: tuple[int, int],
    image_resolution: tuple[int, int],
) -> tuple[dict[str, Any], list[Any]]:
    verification = api.verify_motioncrafter_prediction_manifest(
        manifest_path,
        verify_hashes=True,
        expected_run_spec_sha256=seal["run_spec_sha256"],
    )
    _require(
        verification.get("integrity_bound") is True
        and verification.get("hashes_verified") is True
        and verification.get("member_count") == seal["verified_member_count"],
        "prediction integrity verification differs from its seal",
    )
    manifest = _load_json(manifest_path, name="prediction manifest")
    raw_windows = manifest.get("overlap_windows")
    _require(
        isinstance(raw_windows, list) and bool(raw_windows),
        "prediction windows are missing",
    )
    raw_windows = cast(list[object], raw_windows)
    windows: list[Any] = []
    observed_ids: set[str] = set()
    start, stop = causal_range
    for index, raw_window in enumerate(raw_windows):
        _require(
            isinstance(raw_window, Mapping), f"prediction window {index} is invalid"
        )
        window_record = cast(Mapping[str, Any], raw_window)
        window_id = nonempty_string(
            window_record.get("window_id"), name=f"prediction window {index} ID"
        )
        _require(window_id not in observed_ids, "prediction window ID is duplicated")
        observed_ids.add(window_id)
        relative = canonical_relative_posix_path(
            window_record.get("path"), name=f"prediction window {index} path"
        )
        path = _confined_file(
            manifest_path.parent,
            relative,
            name=f"prediction window {index} path",
        )
        window = api.PredictionWindow.from_npz(path, window_id=window_id)
        _require(
            window.shape[1:] == image_resolution,
            "prediction window image resolution changed",
        )
        _require(
            int(window.frame_indices[0]) == window_record.get("start_frame")
            and int(window.frame_indices[-1]) + 1 == window_record.get("stop_frame"),
            "prediction window frame metadata changed",
        )
        _require(
            np.all((window.frame_indices >= start) & (window.frame_indices < stop)),
            "prediction window contains a frame outside the causal prefix",
        )
        windows.append(window)
    windows.sort(key=lambda item: (int(item.frame_indices[0]), str(item.window_id)))
    _require(
        int(windows[0].frame_indices[0]) == start,
        "first prediction window misses prefix start",
    )
    return manifest, windows


def _receipt_for_stream(
    *,
    production_root: Path,
    production_result: Mapping[str, Any],
    job_id: str,
    object_id: str,
    camera_id: str,
) -> dict[str, Any]:
    rows = cast(Sequence[Mapping[str, Any]], production_result["jobs"])
    matched = [row for row in rows if row["job_id"] == job_id]
    _require(len(matched) == 1, "metric-prefix plan job is absent from production")
    row = matched[0]
    _require(
        row["status"] == "succeeded", "metric-prefix plan uses a failed production job"
    )
    _require(
        row["object_id"] == object_id and row["camera_id"] == camera_id,
        "metric-prefix stream differs from production identity",
    )
    receipt_path, receipt_record = _verify_record(
        production_root, row["receipt"], name="production receipt"
    )
    receipt = validate_deform360_calibration_visual_prediction_seal(
        _load_json(receipt_path, name="prediction seal")
    )
    _require(
        receipt["job_id"] == job_id
        and receipt["object_id"] == object_id
        and receipt["camera_id"] == camera_id,
        "prediction seal differs from metric-prefix stream",
    )
    _require(
        receipt["admission_id"] == production_result["admission_id"]
        and receipt["implementation_revision"]
        == production_result["implementation_revision"]
        and receipt["provider_revision"] == production_result["provider_revision"]
        and receipt["motioncrafter_revision"]
        == production_result["motioncrafter_revision"]
        and receipt["visual_provider_lock_id"]
        == production_result["visual_provider_lock_id"]
        and receipt["model_set_id"] == production_result["model_set_id"],
        "prediction seal lineage differs from production result",
    )
    _require(
        receipt_record["sha256"] == row["receipt"]["sha256"],
        "production receipt descriptor changed",
    )
    return receipt


def _prediction_record_matches_seal(
    *,
    record: Mapping[str, object],
    seal: Mapping[str, Any],
) -> None:
    seal_manifest = _file_record(seal["prediction_manifest"], name="sealed manifest")
    expected_path = (
        PurePosixPath(cast(str, seal["output_relative_directory"]))
        / cast(str, seal_manifest["path"])
    ).as_posix()
    _require(
        record["path"] == expected_path, "prediction manifest path differs from seal"
    )
    _require(
        record["sha256"] == seal_manifest["sha256"]
        and record["byte_count"] == seal_manifest["byte_count"],
        "prediction manifest bytes differ from seal",
    )


def _copy_bound_source(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    digest = _sha256_file(source)
    _require(_sha256_file(destination) == digest, "copied source artifact changed")
    return digest


def materialize_deform360_prob4d_calibration_samples(
    *,
    plan_path: str | Path,
    production_result_path: str | Path,
    production_root: str | Path,
    prediction_root: str | Path,
    metric_root: str | Path,
    selection_path: str | Path,
    visual_provider_spec_path: str | Path,
    metric_prior_policy_path: str | Path,
    camera_eligibility_policy_path: str | Path | None = None,
    expected_processing_revision: str,
    api: Prob4DCalibrationApi | Any,
    output_directory: str | Path,
    config: Deform360Prob4DMaterializationConfig | None = None,
) -> Mapping[str, Any]:
    """Publish a portable source-only calibration-sample bundle atomically."""

    cfg = config or DEFAULT_MATERIALIZATION_CONFIG
    plan_source = Path(plan_path).resolve(strict=True)
    selection_source = Path(selection_path).resolve(strict=True)
    provider_source = Path(visual_provider_spec_path).resolve(strict=True)
    metric_policy_source = Path(metric_prior_policy_path).resolve(strict=True)
    eligibility_policy_source = (
        None
        if camera_eligibility_policy_path is None
        else Path(camera_eligibility_policy_path).resolve(strict=True)
    )
    plan = _load_plan(
        plan_source,
        selection_path=selection_source,
        visual_provider_spec_path=provider_source,
        metric_prior_policy_path=metric_policy_source,
        camera_eligibility_policy_path=eligibility_policy_source,
    )
    _require(
        plan["processing_revision"]
        == exact_revision(
            expected_processing_revision,
            name="expected_processing_revision",
        ),
        "metric-prefix processing revision changed",
    )
    production_source = Path(production_result_path).resolve(strict=True)
    production = validate_deform360_calibration_visual_production_result(
        _load_json(production_source, name="visual production result")
    )
    _require(
        production["result_id"] == plan["visual_production_result_id"],
        "metric-prefix plan uses a different production result",
    )
    _require(
        production["provider_revision"] == plan["prob4d_revision"]
        and production["motioncrafter_revision"] == plan["motioncrafter_revision"],
        "metric-prefix plan revisions differ from production",
    )
    production_root_path = Path(production_root).resolve(strict=True)
    prediction_root_path = Path(prediction_root).resolve(strict=True)
    metric_root_path = Path(metric_root).resolve(strict=True)
    provider = _load_json(provider_source, name="visual provider specification")
    motioncrafter = cast(Mapping[str, Any], provider["motioncrafter"])
    image_resolution = (
        genuine_integer(
            motioncrafter["height"], name="MotionCrafter height", minimum=1
        ),
        genuine_integer(motioncrafter["width"], name="MotionCrafter width", minimum=1),
    )

    raw_cases = plan.get("cases")
    _require(
        isinstance(raw_cases, list) and bool(raw_cases),
        "metric-prefix plan has no cases",
    )
    raw_cases = cast(list[object], raw_cases)
    seen_cases: set[str] = set()
    seen_objects: set[str] = set()
    seen_jobs: set[str] = set()
    excluded_jobs: dict[str, tuple[str, int, str, str]] = {}
    if plan["schema_version"] == VISIBLE_STREAM_PLAN_VERSION:
        raw_excluded = plan["excluded_streams"]
        _require(
            isinstance(raw_excluded, list),
            "version-2 excluded streams must be an array",
        )
        excluded_order: list[tuple[str, str, str]] = []
        for excluded_index, raw_excluded_stream in enumerate(raw_excluded):
            _require(
                isinstance(raw_excluded_stream, Mapping),
                f"excluded stream {excluded_index} is invalid",
            )
            require_exact_fields(
                raw_excluded_stream,
                expected=_EXCLUDED_STREAM_FIELDS,
                name=f"excluded stream {excluded_index}",
            )
            excluded_job_id = sha256_digest(
                raw_excluded_stream["job_id"], name="excluded stream job_id"
            )
            excluded_object_id = nonempty_string(
                raw_excluded_stream["object_id"], name="excluded stream object_id"
            )
            excluded_episode_id = genuine_integer(
                raw_excluded_stream["episode_id"],
                name="excluded stream episode_id",
                minimum=0,
            )
            excluded_stratum = nonempty_string(
                raw_excluded_stream["stratum"], name="excluded stream stratum"
            )
            excluded_camera_id = nonempty_string(
                raw_excluded_stream["camera_id"], name="excluded stream camera_id"
            )
            _require(
                raw_excluded_stream["reason"] == SUPPORT_NEGATIVE_REASON,
                "excluded stream is not a target-free visibility negative",
            )
            _require(
                excluded_job_id not in excluded_jobs,
                "version-2 plan repeats an excluded stream",
            )
            excluded_jobs[excluded_job_id] = (
                excluded_object_id,
                excluded_episode_id,
                excluded_stratum,
                excluded_camera_id,
            )
            excluded_order.append(
                (excluded_object_id, excluded_camera_id, excluded_job_id)
            )
        _require(
            excluded_order == sorted(excluded_order),
            "excluded streams are not sorted",
        )
    normalized_cases: list[dict[str, Any]] = []
    case_order: list[tuple[str, int]] = []
    for case_index, raw_case in enumerate(raw_cases):
        _require(isinstance(raw_case, Mapping), f"plan case {case_index} is invalid")
        case_record = cast(Mapping[str, Any], raw_case)
        require_exact_fields(
            case_record, expected=_CASE_FIELDS, name=f"plan case {case_index}"
        )
        case_id = nonempty_string(
            case_record["case_id"], name=f"plan case {case_index} ID"
        )
        object_id = nonempty_string(
            case_record["object_id"], name=f"plan case {case_index} object"
        )
        episode_id = genuine_integer(
            case_record["episode_id"],
            name=f"plan case {case_index} episode",
            minimum=0,
        )
        stratum = nonempty_string(
            case_record["stratum"], name=f"plan case {case_index} stratum"
        )
        _require(
            case_id not in seen_cases and object_id not in seen_objects,
            "plan repeats a case or physical object",
        )
        seen_cases.add(case_id)
        seen_objects.add(object_id)
        case_order.append((object_id, episode_id))
        raw_range = case_record["causal_frame_range_half_open"]
        _require(
            isinstance(raw_range, list)
            and len(raw_range) == 2
            and all(type(value) is int for value in raw_range),
            "plan causal frame range is invalid",
        )
        causal_range = (int(raw_range[0]), int(raw_range[1]))
        _require(causal_range[0] < causal_range[1], "plan causal frame range is empty")
        raw_streams = case_record["streams"]
        _require(
            isinstance(raw_streams, list) and len(raw_streams) >= 2,
            "every calibration object requires at least two successful streams",
        )
        streams: list[dict[str, Any]] = []
        stream_order: list[tuple[str, str]] = []
        for stream_index, raw_stream in enumerate(raw_streams):
            _require(isinstance(raw_stream, Mapping), "plan stream is invalid")
            require_exact_fields(
                raw_stream,
                expected=_STREAM_FIELDS,
                name=f"plan case {case_index} stream {stream_index}",
            )
            job_id = sha256_digest(raw_stream["job_id"], name="stream job_id")
            camera_id = nonempty_string(
                raw_stream["camera_id"], name="stream camera_id"
            )
            _require(job_id not in seen_jobs, "plan repeats a production job")
            seen_jobs.add(job_id)
            stream_order.append((camera_id, job_id))
            prediction_path, prediction_record = _verify_record(
                prediction_root_path,
                raw_stream["prediction_manifest"],
                name="prediction manifest",
            )
            metric_path, metric_record = _verify_record(
                metric_root_path,
                raw_stream["metric_prefix"],
                name="metric prefix",
            )
            calibration_path, calibration_record = _verify_record(
                metric_root_path,
                raw_stream["metric_calibration"],
                name="metric calibration",
            )
            seal = _receipt_for_stream(
                production_root=production_root_path,
                production_result=production,
                job_id=job_id,
                object_id=object_id,
                camera_id=camera_id,
            )
            _require(
                seal["episode_id"] == episode_id
                and seal["stratum"] == stratum
                and seal["causal_prefix_frame_range_half_open"] == list(causal_range),
                "prediction seal differs from plan case",
            )
            _prediction_record_matches_seal(record=prediction_record, seal=seal)
            streams.append(
                {
                    "job_id": job_id,
                    "camera_id": camera_id,
                    "prediction_path": prediction_path,
                    "prediction_record": prediction_record,
                    "metric_path": metric_path,
                    "metric_record": metric_record,
                    "calibration_path": calibration_path,
                    "calibration_record": calibration_record,
                    "seal": seal,
                }
            )
        _require(stream_order == sorted(stream_order), "plan streams are not sorted")
        normalized_cases.append(
            {
                "case_id": case_id,
                "object_id": object_id,
                "episode_id": episode_id,
                "stratum": stratum,
                "causal_range": causal_range,
                "streams": streams,
            }
        )
    _require(case_order == sorted(case_order), "plan cases are not sorted")

    case_identity_by_object = {
        cast(str, case["object_id"]): (
            cast(int, case["episode_id"]),
            cast(str, case["stratum"]),
        )
        for case in normalized_cases
    }
    for object_id, episode_id, stratum, _camera_id in excluded_jobs.values():
        _require(
            case_identity_by_object.get(object_id) == (episode_id, stratum),
            "excluded stream differs from its retained calibration case",
        )

    succeeded_jobs = {
        cast(str, row["job_id"])
        for row in cast(Sequence[Mapping[str, Any]], production["jobs"])
        if row["status"] == "succeeded"
    }
    if plan["schema_version"] == PLAN_VERSION:
        _require(
            seen_jobs == succeeded_jobs,
            "metric-prefix plan must cover every and only successful production job",
        )
    else:
        _require(
            seen_jobs.isdisjoint(excluded_jobs)
            and seen_jobs | set(excluded_jobs) == succeeded_jobs,
            "visible-stream plan does not account for every production job",
        )
        production_rows = {
            cast(str, row["job_id"]): row
            for row in cast(Sequence[Mapping[str, Any]], production["jobs"])
        }
        for job_id, identity in excluded_jobs.items():
            row = production_rows[job_id]
            object_id, _episode_id, _stratum, camera_id = identity
            _require(
                row["status"] == "succeeded"
                and row["object_id"] == object_id
                and row["camera_id"] == camera_id,
                "excluded stream differs from visual production",
            )

    output = Path(output_directory).resolve()
    _require(not output.exists(), "output directory already exists")
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        point_errors: list[np.ndarray] = []
        point_rays: list[np.ndarray] = []
        point_parallel: list[np.ndarray] = []
        point_lateral: list[np.ndarray] = []
        point_cases: list[np.ndarray] = []
        point_frames: list[np.ndarray] = []
        point_cluster_keys: list[tuple[int, int, int, int]] = []
        gauge_errors: list[np.ndarray] = []
        gauge_covariances: list[np.ndarray] = []
        gauge_cases: list[int] = []
        gauge_frames: list[int] = []
        anchors: list[np.ndarray] = []
        anchor_covariances: list[np.ndarray] = []
        output_cases: list[dict[str, Any]] = []
        source_artifacts: dict[str, str] = {}
        flattened_prediction_index = 0

        plan_copy = temporary / "source-artifacts" / "metric-prefix-plan.json"
        plan_copy.parent.mkdir(parents=True)
        shutil.copyfile(plan_source, plan_copy)
        source_artifacts[plan_copy.relative_to(temporary).as_posix()] = _sha256_file(
            plan_copy
        )
        if eligibility_policy_source is not None:
            policy_copy = (
                temporary / "source-artifacts" / "camera-eligibility-policy.json"
            )
            source_artifacts[policy_copy.relative_to(temporary).as_posix()] = (
                _copy_bound_source(eligibility_policy_source, policy_copy)
            )

        for case_index, case in enumerate(normalized_cases):
            case_predictions: list[dict[str, object]] = []
            case_metrics: list[dict[str, object]] = []
            for stream in cast(Sequence[Mapping[str, Any]], case["streams"]):
                metric = _load_metric_prefix(
                    cast(Path, stream["metric_path"]),
                    causal_range=cast(tuple[int, int], case["causal_range"]),
                    image_resolution=image_resolution,
                )
                _manifest, windows = _load_prediction_windows(
                    api=api,
                    manifest_path=cast(Path, stream["prediction_path"]),
                    seal=cast(Mapping[str, Any], stream["seal"]),
                    causal_range=cast(tuple[int, int], case["causal_range"]),
                    image_resolution=image_resolution,
                )
                window_map = {str(window.window_id): window for window in windows}
                with api.alignment_covariance_context(fallback_policy="error"):
                    alignments = [
                        api.align_windows(
                            reference,
                            moving,
                            max_correspondences=cfg.maximum_metric_fit_correspondences,
                            seed=int.from_bytes(
                                hashlib.sha256(
                                    f"{stream['job_id']}:{reference.window_id}:{moving.window_id}".encode()
                                ).digest()[:4],
                                "big",
                            ),
                            covariance_cluster_size=cfg.covariance_cluster_size_pixels,
                        )
                        for reference, moving in zip(
                            windows[:-1], windows[1:], strict=True
                        )
                    ]
                _require(
                    all(
                        getattr(alignment.result, "covariance_method", "")
                        == "frame_spatial_cluster_robust_v1"
                        for alignment in alignments
                    ),
                    "relative alignment did not use cluster-robust covariance",
                )
                evidence = api.accumulate_disagreement(window_map, alignments)
                metric_fits = {
                    str(window.window_id): _fit_window_metric_gauge(
                        api=api,
                        window=window,
                        metric=metric,
                        config=cfg,
                        seed_text=f"{stream['job_id']}:{window.window_id}:metric",
                    )
                    for window in windows
                }
                root_window = windows[0]
                root_fit = metric_fits[str(root_window.window_id)]
                anchors.append(
                    np.asarray(root_fit.transform.as_vector(), dtype=np.float64)
                )
                anchor_covariances.append(
                    np.asarray(root_fit.covariance, dtype=np.float64)
                )

                uncertainty_model = api.DepthDisagreementModel()
                for window in windows:
                    covariance = uncertainty_model.predict(
                        window, evidence[str(window.window_id)]
                    )
                    fit = metric_fits[str(window.window_id)]
                    candidate_parts: list[np.ndarray] = []
                    for local_index, frame in enumerate(window.frame_indices):
                        if int(frame) == int(window.frame_indices[0]):
                            continue
                        metric_index = int(frame) - int(metric.frame_indices[0])
                        active = (
                            window.valid_mask[local_index]
                            & metric.valid_mask[metric_index]
                        )
                        local_rows, local_columns = np.nonzero(active)
                        if len(local_rows):
                            candidate_parts.append(
                                np.column_stack(
                                    (
                                        np.full(len(local_rows), local_index),
                                        np.full(len(local_rows), metric_index),
                                        np.full(len(local_rows), int(frame)),
                                        local_rows,
                                        local_columns,
                                    )
                                )
                            )
                    _require(
                        bool(candidate_parts),
                        "prediction window has no held-prefix point rows",
                    )
                    candidates = np.concatenate(candidate_parts, axis=0).astype(
                        np.int64
                    )
                    _require(
                        len(candidates) >= cfg.minimum_point_rows_per_window,
                        "prediction window has too little held-prefix metric support",
                    )
                    selected = _deterministic_rows(
                        len(candidates),
                        cfg.maximum_point_rows_per_window,
                        seed_text=f"{stream['job_id']}:{window.window_id}:points",
                    )
                    chosen = candidates[selected]
                    local_index = chosen[:, 0]
                    metric_index = chosen[:, 1]
                    frames = chosen[:, 2]
                    rows = chosen[:, 3]
                    columns = chosen[:, 4]
                    predicted_world = fit.transform.transform_points(
                        window.point_map[local_index, rows, columns]
                    )
                    truth_world = metric.points_world_m[metric_index, rows, columns]
                    local_rays = covariance.ray_directions[local_index, rows, columns]
                    world_rays = fit.transform.rotate_directions(local_rays)
                    world_rays /= np.linalg.norm(world_rays, axis=1, keepdims=True)
                    scale_squared = float(fit.transform.scale) ** 2
                    point_errors.append(predicted_world - truth_world)
                    point_rays.append(world_rays)
                    point_parallel.append(
                        scale_squared
                        * covariance.parallel_variance[local_index, rows, columns]
                    )
                    point_lateral.append(
                        scale_squared
                        * covariance.lateral_variance[local_index, rows, columns]
                    )
                    point_cases.append(np.full(len(chosen), case_index, dtype=np.int64))
                    point_frames.append(frames)
                    point_cluster_keys.extend(
                        (
                            case_index,
                            int(frame),
                            int(row) // cfg.covariance_cluster_size_pixels,
                            int(column) // cfg.covariance_cluster_size_pixels,
                        )
                        for frame, row, column in zip(
                            frames, rows, columns, strict=True
                        )
                    )

                for alignment in alignments:
                    reference_metric = metric_fits[
                        str(alignment.reference_id)
                    ].transform
                    moving_metric = metric_fits[str(alignment.moving_id)].transform
                    true_relative = reference_metric.inverse().compose(moving_metric)
                    error = alignment.result.transform.inverse().compose(true_relative)
                    gauge_errors.append(np.asarray(error.as_vector(), dtype=np.float64))
                    gauge_covariances.append(
                        np.asarray(alignment.result.covariance, dtype=np.float64)
                    )
                    gauge_cases.append(case_index)
                    gauge_frames.append(int(np.min(alignment.common_frames)))

                source_dir = (
                    temporary
                    / "source-artifacts"
                    / f"stream-{flattened_prediction_index:04d}"
                )
                metric_destination = source_dir / "metric-prefix.npz"
                calibration_destination = source_dir / "metric-calibration.bin"
                metric_digest = _copy_bound_source(
                    cast(Path, stream["metric_path"]), metric_destination
                )
                calibration_digest = _copy_bound_source(
                    cast(Path, stream["calibration_path"]),
                    calibration_destination,
                )
                source_artifacts[
                    metric_destination.relative_to(temporary).as_posix()
                ] = metric_digest
                source_artifacts[
                    calibration_destination.relative_to(temporary).as_posix()
                ] = calibration_digest
                output_prediction_record = cast(
                    Mapping[str, Any], stream["prediction_record"]
                )
                case_predictions.append(
                    {
                        "job_id": stream["job_id"],
                        "camera_id": stream["camera_id"],
                        **dict(output_prediction_record),
                    }
                )
                case_metrics.append(
                    {
                        "job_id": stream["job_id"],
                        "camera_id": stream["camera_id"],
                        "window_id": str(root_window.window_id),
                        "frame_id": cast(tuple[int, int], case["causal_range"])[0],
                        "coordinate_frame": COORDINATE_FRAME,
                        "source_kind": METRIC_SOURCE_KIND,
                        "source_artifact_sha256": metric_digest,
                        "calibration_artifact_sha256": calibration_digest,
                    }
                )
                flattened_prediction_index += 1
            output_cases.append(
                {
                    "case_id": case["case_id"],
                    "object_id": case["object_id"],
                    "episode_id": case["episode_id"],
                    "stratum": case["stratum"],
                    "causal_frame_range_half_open": list(case["causal_range"]),
                    "prediction_manifests": case_predictions,
                    "metric_references": case_metrics,
                }
            )

        _require(bool(point_errors), "materialization produced no point residuals")
        _require(
            bool(gauge_errors), "materialization produced no relative-gauge residuals"
        )
        canonical_cluster_keys = {
            key: index for index, key in enumerate(sorted(set(point_cluster_keys)))
        }
        cluster_indices = np.asarray(
            [canonical_cluster_keys[key] for key in point_cluster_keys], dtype=np.int64
        )
        arrays = {
            "point_errors_m": np.concatenate(point_errors),
            "point_ray_directions": np.concatenate(point_rays),
            "point_parallel_variance_m2": np.concatenate(point_parallel),
            "point_lateral_variance_m2": np.concatenate(point_lateral),
            "point_case_index": np.concatenate(point_cases),
            "point_frame_id": np.concatenate(point_frames),
            "point_correlation_cluster_index": cluster_indices,
            "point_valid": np.ones(len(cluster_indices), dtype=np.bool_),
            "gauge_errors": np.asarray(gauge_errors, dtype=np.float64),
            "gauge_covariance": np.asarray(gauge_covariances, dtype=np.float64),
            "gauge_case_index": np.asarray(gauge_cases, dtype=np.int64),
            "gauge_frame_id": np.asarray(gauge_frames, dtype=np.int64),
            "anchor_global_from_local": np.asarray(anchors, dtype=np.float64),
            "anchor_covariance": np.asarray(anchor_covariances, dtype=np.float64),
            "anchor_prediction_index": np.arange(len(anchors), dtype=np.int64),
        }
        arrays_path = temporary / "samples.npz"
        np.savez_compressed(arrays_path, **arrays)
        manifest: dict[str, Any] = {
            "schema": SAMPLE_SCHEMA,
            "schema_version": SAMPLE_VERSION,
            "semantics": SAMPLE_SEMANTICS,
            "protocol_id": plan["protocol_id"],
            "selection_file_sha256": plan["selection_file_sha256"],
            "visual_provider_spec_file_sha256": plan[
                "visual_provider_spec_file_sha256"
            ],
            "metric_prior_policy_file_sha256": plan["metric_prior_policy_file_sha256"],
            "dataset_revision": plan["dataset_revision"],
            "prob4d_revision": plan["prob4d_revision"],
            "motioncrafter_revision": plan["motioncrafter_revision"],
            "visual_production_result_id": plan["visual_production_result_id"],
            "point_correlation_cluster_semantics": POINT_CLUSTER_SEMANTICS,
            "cases": output_cases,
            "arrays": {
                "path": arrays_path.name,
                "sha256": _sha256_file(arrays_path),
                "byte_count": arrays_path.stat().st_size,
            },
            "source_artifacts": dict(sorted(source_artifacts.items())),
            "information_boundary": dict(_EXPECTED_BOUNDARY),
            "claim_boundary": (
                "Source-only calibration samples from released Deform360 causal "
                "prefixes; no confirmation, target, prediction-benefit, or SOTA claim."
            ),
        }
        manifest["bundle_id"] = content_id(manifest)
        manifest_path = temporary / "samples.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        load_deform360_prob4d_calibration_samples(
            manifest_path,
            selection_path=selection_source,
            visual_provider_spec_path=provider_source,
            metric_prior_policy_path=metric_policy_source,
            prediction_root=prediction_root_path,
        )
        members = sorted(
            path
            for path in temporary.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        )
        (temporary / "SHA256SUMS").write_text(
            "".join(
                f"{_sha256_file(path)}  {path.relative_to(temporary).as_posix()}\n"
                for path in members
            ),
            encoding="ascii",
        )
        os.rename(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return frozen_finite_json_mapping(manifest, name="calibration sample manifest")


__all__ = [
    "COORDINATE_FRAME",
    "DEFAULT_MATERIALIZATION_CONFIG",
    "METRIC_PREFIX_ARRAYS",
    "METRIC_SOURCE_KIND",
    "PLAN_SCHEMA",
    "PLAN_SEMANTICS",
    "PLAN_VERSION",
    "Deform360Prob4DMaterializationConfig",
    "materialize_deform360_prob4d_calibration_samples",
]
