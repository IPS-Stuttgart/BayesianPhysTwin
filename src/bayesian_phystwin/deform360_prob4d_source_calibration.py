"""Source-only Prob4D covariance calibration for public Deform360 prefixes.

The active MotionCrafter producer exports exploratory Prob4D predictions.  This
module turns separately materialized metric-prefix residuals into the calibrated
Prob4D artifacts required by the claim-bearing provider path.  Dense rows are
collapsed into declared correlation clusters and physical objects receive equal
weight.  Confirmation identities and future frames are rejected before fitting.

Metric-prefix reconstruction is intentionally outside this module.  Its output
enters through one checksummed sample bundle so the reconstruction implementation,
the exact visual predictions, and every causal frame remain auditable inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType, ModuleType
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import (
    canonical_relative_posix_path,
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    immutable_array,
    immutable_integer_array,
    plain_json,
)
from ._portable_contracts import (
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
)

SAMPLE_SCHEMA: Final = "bayesian-phystwin.deform360-prob4d-calibration-samples"
SAMPLE_VERSION: Final = 1
SAMPLE_SEMANTICS: Final = (
    "causal-metric-prefix-residuals-with-object-and-cluster-lineage-v1"
)
RESULT_SCHEMA: Final = "bayesian-phystwin.deform360-prob4d-source-calibration"
RESULT_VERSION: Final = 1
RESULT_SEMANTICS: Final = (
    "physical-object-balanced-prob4d-point-gauge-and-metric-anchor-fit-v1"
)
POINT_CLUSTER_SEMANTICS: Final = (
    "one-effective-row-per-declared-camera-overlap-dependence-cluster-v1"
)
GAUGE_AGGREGATION_SEMANTICS: Final = (
    "equal-object-mean-of-within-object-upper-winsorized-block-ratios-v1"
)
POINT_GROUP_DEFINITION: Final = "physical-object-id"
MINIMUM_OBJECT_COUNT: Final = 8
MINIMUM_OBJECTS_PER_STRATUM: Final = 4

_ARRAY_KEYS = frozenset(
    {
        "point_errors_m",
        "point_ray_directions",
        "point_parallel_variance_m2",
        "point_lateral_variance_m2",
        "point_case_index",
        "point_frame_id",
        "point_correlation_cluster_index",
        "point_valid",
        "gauge_errors",
        "gauge_covariance",
        "gauge_case_index",
        "gauge_frame_id",
        "anchor_global_from_local",
        "anchor_covariance",
    }
)
_SAMPLE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "bundle_id",
        "protocol_id",
        "selection_file_sha256",
        "visual_provider_spec_file_sha256",
        "metric_prior_policy_file_sha256",
        "dataset_revision",
        "prob4d_revision",
        "motioncrafter_revision",
        "visual_production_result_id",
        "point_correlation_cluster_semantics",
        "cases",
        "arrays",
        "source_artifacts",
        "information_boundary",
        "claim_boundary",
    }
)
_CASE_FIELDS = frozenset(
    {
        "case_id",
        "object_id",
        "episode_id",
        "stratum",
        "causal_frame_range_half_open",
        "prediction_manifests",
        "metric_reference",
    }
)
_PREDICTION_FIELDS = frozenset(
    {"job_id", "camera_id", "path", "sha256", "byte_count"}
)
_METRIC_REFERENCE_FIELDS = frozenset(
    {
        "window_id",
        "frame_id",
        "coordinate_frame",
        "source_kind",
        "source_artifact_sha256",
        "calibration_artifact_sha256",
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
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    return genuine_integer(value, name=name, minimum=minimum)


def _file_record(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    require_exact_fields(value, expected=_FILE_FIELDS, name=name)
    return {
        "path": canonical_relative_posix_path(value["path"], name=f"{name}.path"),
        "sha256": sha256_digest(value["sha256"], name=f"{name}.sha256"),
        "byte_count": _integer(
            value["byte_count"], name=f"{name}.byte_count", minimum=1
        ),
    }


def _confined_file(root: Path, relative: str, *, name: str) -> Path:
    root_absolute = root.absolute()
    try:
        root_resolved = root_absolute.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} root does not exist: {root}") from error
    _require(root_resolved.is_dir(), f"{name} root must be a directory")
    unresolved = root_absolute.joinpath(*PurePosixPath(relative).parts)
    cursor = root_absolute
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{name} path must not contain symlinks: {relative}")
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} does not exist: {relative}") from error
    try:
        resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"{name} escapes its declared root: {relative}") from error
    _require(resolved.is_file(), f"{name} must be an ordinary file: {relative}")
    return resolved


def _verify_file_record(root: Path, record: Mapping[str, object], *, name: str) -> Path:
    path = _confined_file(root, cast(str, record["path"]), name=name)
    _require(
        path.stat().st_size == record["byte_count"], f"{name} byte_count changed"
    )
    _require(_sha256_file(path) == record["sha256"], f"{name} SHA-256 changed")
    return path


def _finite_array(value: object, *, name: str, shape_tail: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    _require(
        array.ndim == len(shape_tail) + 1 and array.shape[1:] == shape_tail,
        f"{name} must have shape (N, {', '.join(map(str, shape_tail))})",
    )
    _require(np.all(np.isfinite(array)), f"{name} contains non-finite values")
    return array


def _positive_vector(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    _require(array.ndim == 1, f"{name} must be a vector")
    _require(
        np.all(np.isfinite(array)) and np.all(array > 0.0),
        f"{name} must be finite and strictly positive",
    )
    return array


def _integer_vector(value: object, *, name: str) -> np.ndarray:
    array = immutable_integer_array(value, name=name)
    _require(array.ndim == 1, f"{name} must be a vector")
    return array


def _boolean_vector(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    _require(raw.dtype.kind == "b" and raw.ndim == 1, f"{name} must be a Boolean vector")
    return immutable_array(raw, dtype=np.bool_)


def _validate_psd_stack(value: object, *, name: str, count: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    _require(array.shape == (count, 7, 7), f"{name} must have shape ({count}, 7, 7)")
    _require(np.all(np.isfinite(array)), f"{name} contains non-finite values")
    symmetric = 0.5 * (array + np.swapaxes(array, 1, 2))
    _require(
        np.allclose(array, symmetric, atol=1e-12, rtol=1e-10),
        f"{name} must be symmetric",
    )
    _require(
        float(np.min(np.linalg.eigvalsh(symmetric))) >= -1e-12,
        f"{name} must be positive semidefinite",
    )
    return symmetric


def _sample_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = dict(plain_json(value))
    descriptor.pop("bundle_id", None)
    return descriptor


@dataclass(frozen=True)
class Deform360Prob4DCalibrationSamplesV1:
    """Validated source-only sample bundle and immutable numeric arrays."""

    manifest_path: Path
    manifest_file_sha256: str
    bundle_id: str
    protocol_id: str
    dataset_revision: str
    prob4d_revision: str
    motioncrafter_revision: str
    image_resolution: tuple[int, int]
    window_size: int
    window_overlap: int
    visual_production_result_id: str
    cases: tuple[Mapping[str, Any], ...]
    prediction_manifest_paths: tuple[tuple[Path, ...], ...]
    arrays: Mapping[str, np.ndarray]
    arrays_file_sha256: str
    source_artifact_sha256: tuple[str, ...]

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(cast(str, case["case_id"]) for case in self.cases)

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(cast(str, case["object_id"]) for case in self.cases)


def _validate_cases(
    values: object,
    *,
    calibration_records: Mapping[tuple[str, int], Mapping[str, Any]],
    confirmation_object_ids: frozenset[str],
    prediction_root: Path,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[tuple[Path, ...], ...]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("cases must be a sequence")
    normalized: list[Mapping[str, Any]] = []
    prediction_paths: list[tuple[Path, ...]] = []
    seen_cases: set[str] = set()
    seen_objects: set[str] = set()
    seen_jobs: set[str] = set()
    stratum_counts: dict[str, int] = {}
    for index, raw_case in enumerate(values):
        name = f"cases[{index}]"
        if not isinstance(raw_case, Mapping):
            raise ValueError(f"{name} must be a JSON object")
        require_exact_fields(raw_case, expected=_CASE_FIELDS, name=name)
        case_id = nonempty_string(raw_case["case_id"], name=f"{name}.case_id")
        object_id = nonempty_string(raw_case["object_id"], name=f"{name}.object_id")
        episode_id = _integer(raw_case["episode_id"], name=f"{name}.episode_id")
        stratum = nonempty_string(raw_case["stratum"], name=f"{name}.stratum")
        _require(object_id not in confirmation_object_ids, "confirmation object in samples")
        _require(case_id not in seen_cases, "duplicate calibration case_id")
        _require(object_id not in seen_objects, "one physical object appears more than once")
        locked = calibration_records.get((object_id, episode_id))
        _require(locked is not None, "sample case is not in the locked calibration cohort")
        locked_record = cast(Mapping[str, Any], locked)
        _require(locked_record["stratum"] == stratum, "sample case stratum changed")
        frame_range = raw_case["causal_frame_range_half_open"]
        _require(
            isinstance(frame_range, Sequence)
            and not isinstance(frame_range, (str, bytes))
            and len(frame_range) == 2,
            f"{name}.causal_frame_range_half_open must contain two integers",
        )
        start = _integer(frame_range[0], name=f"{name}.causal start")
        stop = _integer(frame_range[1], name=f"{name}.causal stop", minimum=1)
        _require(start < stop, f"{name} causal frame range is empty")

        raw_predictions = raw_case["prediction_manifests"]
        _require(
            isinstance(raw_predictions, Sequence)
            and not isinstance(raw_predictions, (str, bytes))
            and len(raw_predictions) >= 2,
            f"{name} requires at least two prediction manifests",
        )
        predictions: list[dict[str, object]] = []
        paths: list[Path] = []
        seen_cameras: set[str] = set()
        for prediction_index, raw_prediction in enumerate(raw_predictions):
            prediction_name = f"{name}.prediction_manifests[{prediction_index}]"
            if not isinstance(raw_prediction, Mapping):
                raise ValueError(f"{prediction_name} must be a JSON object")
            require_exact_fields(
                raw_prediction,
                expected=_PREDICTION_FIELDS,
                name=prediction_name,
            )
            job_id = sha256_digest(raw_prediction["job_id"], name=f"{prediction_name}.job_id")
            camera_id = nonempty_string(
                raw_prediction["camera_id"], name=f"{prediction_name}.camera_id"
            )
            record = _file_record(
                {field: raw_prediction[field] for field in _FILE_FIELDS},
                name=prediction_name,
            )
            _require(job_id not in seen_jobs, "prediction job appears in multiple cases")
            _require(camera_id not in seen_cameras, "camera repeated within sample case")
            seen_jobs.add(job_id)
            seen_cameras.add(camera_id)
            paths.append(_verify_file_record(prediction_root, record, name=prediction_name))
            predictions.append({"job_id": job_id, "camera_id": camera_id, **record})

        metric = raw_case["metric_reference"]
        if not isinstance(metric, Mapping):
            raise ValueError(f"{name}.metric_reference must be a JSON object")
        require_exact_fields(
            metric,
            expected=_METRIC_REFERENCE_FIELDS,
            name=f"{name}.metric_reference",
        )
        metric_frame = _integer(
            metric["frame_id"], name=f"{name}.metric_reference.frame_id"
        )
        _require(metric_frame == start, "metric prior must use the first causal frame")
        normalized_metric = {
            "window_id": nonempty_string(
                metric["window_id"], name=f"{name}.metric_reference.window_id"
            ),
            "frame_id": metric_frame,
            "coordinate_frame": nonempty_string(
                metric["coordinate_frame"],
                name=f"{name}.metric_reference.coordinate_frame",
            ),
            "source_kind": nonempty_string(
                metric["source_kind"], name=f"{name}.metric_reference.source_kind"
            ),
            "source_artifact_sha256": sha256_digest(
                metric["source_artifact_sha256"],
                name=f"{name}.metric_reference.source_artifact_sha256",
            ),
            "calibration_artifact_sha256": sha256_digest(
                metric["calibration_artifact_sha256"],
                name=f"{name}.metric_reference.calibration_artifact_sha256",
            ),
        }
        normalized.append(
            frozen_finite_json_mapping(
                {
                    "case_id": case_id,
                    "object_id": object_id,
                    "episode_id": episode_id,
                    "stratum": stratum,
                    "causal_frame_range_half_open": [start, stop],
                    "prediction_manifests": predictions,
                    "metric_reference": normalized_metric,
                },
                name=name,
            )
        )
        prediction_paths.append(tuple(paths))
        seen_cases.add(case_id)
        seen_objects.add(object_id)
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
    _require(len(normalized) >= MINIMUM_OBJECT_COUNT, "too few physical objects")
    _require(
        len(stratum_counts) >= 2
        and min(stratum_counts.values()) >= MINIMUM_OBJECTS_PER_STRATUM,
        "too few physical objects in one calibration stratum",
    )
    return tuple(normalized), tuple(prediction_paths)


def _validate_arrays(
    raw: Mapping[str, np.ndarray],
    *,
    cases: Sequence[Mapping[str, Any]],
) -> Mapping[str, np.ndarray]:
    _require(set(raw) == _ARRAY_KEYS, "calibration sample array names changed")
    case_count = len(cases)
    point_errors = _finite_array(raw["point_errors_m"], name="point_errors_m", shape_tail=(3,))
    point_count = len(point_errors)
    point_rays = _finite_array(
        raw["point_ray_directions"],
        name="point_ray_directions",
        shape_tail=(3,),
    )
    _require(len(point_rays) == point_count, "point ray count changed")
    _require(
        np.all(np.linalg.norm(point_rays, axis=1) > 1e-12),
        "point rays must be nonzero",
    )
    point_parallel = _positive_vector(
        raw["point_parallel_variance_m2"], name="point_parallel_variance_m2"
    )
    point_lateral = _positive_vector(
        raw["point_lateral_variance_m2"], name="point_lateral_variance_m2"
    )
    point_case = _integer_vector(raw["point_case_index"], name="point_case_index")
    point_frame = _integer_vector(raw["point_frame_id"], name="point_frame_id")
    point_cluster = _integer_vector(
        raw["point_correlation_cluster_index"],
        name="point_correlation_cluster_index",
    )
    point_valid = _boolean_vector(raw["point_valid"], name="point_valid")
    for name, array in (
        ("point_parallel_variance_m2", point_parallel),
        ("point_lateral_variance_m2", point_lateral),
        ("point_case_index", point_case),
        ("point_frame_id", point_frame),
        ("point_correlation_cluster_index", point_cluster),
        ("point_valid", point_valid),
    ):
        _require(len(array) == point_count, f"{name} count changed")
    _require(point_count > 0 and np.any(point_valid), "no valid point residuals")
    _require(
        np.all((point_case >= 0) & (point_case < case_count)),
        "point case index is out of range",
    )
    _require(np.all(point_cluster >= 0), "point cluster indices must be nonnegative")

    gauge_errors = _finite_array(
        raw["gauge_errors"], name="gauge_errors", shape_tail=(7,)
    )
    gauge_count = len(gauge_errors)
    _require(gauge_count > 0, "no gauge residuals")
    gauge_covariance = _validate_psd_stack(
        raw["gauge_covariance"], name="gauge_covariance", count=gauge_count
    )
    gauge_case = _integer_vector(raw["gauge_case_index"], name="gauge_case_index")
    gauge_frame = _integer_vector(raw["gauge_frame_id"], name="gauge_frame_id")
    _require(
        len(gauge_case) == gauge_count and len(gauge_frame) == gauge_count,
        "gauge index counts changed",
    )
    _require(
        np.all((gauge_case >= 0) & (gauge_case < case_count)),
        "gauge case index is out of range",
    )

    anchors = _finite_array(
        raw["anchor_global_from_local"],
        name="anchor_global_from_local",
        shape_tail=(7,),
    )
    _require(len(anchors) == case_count, "one metric transform is required per case")
    anchor_covariance = _validate_psd_stack(
        raw["anchor_covariance"], name="anchor_covariance", count=case_count
    )

    cluster_owner: dict[int, int] = {}
    for row in range(point_count):
        cluster = int(point_cluster[row])
        case_index = int(point_case[row])
        previous = cluster_owner.setdefault(cluster, case_index)
        _require(previous == case_index, "one point cluster spans physical objects")
    for case_index, case in enumerate(cases):
        start, stop = cast(Sequence[int], case["causal_frame_range_half_open"])
        case_point_rows = point_case == case_index
        point_rows = case_point_rows & point_valid
        gauge_rows = gauge_case == case_index
        _require(np.any(point_rows), "sample case has no valid point residual")
        _require(np.any(gauge_rows), "sample case has no gauge residual")
        _require(
            np.all(
                (point_frame[case_point_rows] >= start)
                & (point_frame[case_point_rows] < stop)
            ),
            "point residual uses a frame outside the causal prefix",
        )
        _require(
            np.all((gauge_frame[gauge_rows] >= start) & (gauge_frame[gauge_rows] < stop)),
            "gauge residual uses a frame outside the causal prefix",
        )

    return {
        "point_errors_m": immutable_array(point_errors, dtype=np.float64),
        "point_ray_directions": immutable_array(point_rays, dtype=np.float64),
        "point_parallel_variance_m2": immutable_array(point_parallel, dtype=np.float64),
        "point_lateral_variance_m2": immutable_array(point_lateral, dtype=np.float64),
        "point_case_index": point_case,
        "point_frame_id": point_frame,
        "point_correlation_cluster_index": point_cluster,
        "point_valid": point_valid,
        "gauge_errors": immutable_array(gauge_errors, dtype=np.float64),
        "gauge_covariance": immutable_array(gauge_covariance, dtype=np.float64),
        "gauge_case_index": gauge_case,
        "gauge_frame_id": gauge_frame,
        "anchor_global_from_local": immutable_array(anchors, dtype=np.float64),
        "anchor_covariance": immutable_array(anchor_covariance, dtype=np.float64),
    }


def load_deform360_prob4d_calibration_samples(
    manifest_path: str | Path,
    *,
    selection_path: str | Path,
    visual_provider_spec_path: str | Path,
    metric_prior_policy_path: str | Path,
    prediction_root: str | Path,
) -> Deform360Prob4DCalibrationSamplesV1:
    """Load and validate one source-only calibration sample bundle."""

    manifest_source = Path(manifest_path).resolve(strict=True)
    manifest = dict(
        load_strict_json_object(manifest_source, label="Prob4D calibration samples")
    )
    require_exact_fields(manifest, expected=_SAMPLE_FIELDS, name="sample manifest")
    _require(
        manifest["schema"] == SAMPLE_SCHEMA
        and manifest["schema_version"] == SAMPLE_VERSION
        and manifest["semantics"] == SAMPLE_SEMANTICS,
        "unsupported calibration sample contract",
    )
    supplied_id = sha256_digest(manifest["bundle_id"], name="bundle_id")
    _require(content_id(_sample_descriptor(manifest)) == supplied_id, "bundle_id changed")
    protocol_id = nonempty_string(manifest["protocol_id"], name="protocol_id")

    selection_source = Path(selection_path).resolve(strict=True)
    provider_source = Path(visual_provider_spec_path).resolve(strict=True)
    metric_source = Path(metric_prior_policy_path).resolve(strict=True)
    _require(
        _sha256_file(selection_source)
        == sha256_digest(manifest["selection_file_sha256"], name="selection_file_sha256"),
        "selection contract changed",
    )
    _require(
        _sha256_file(provider_source)
        == sha256_digest(
            manifest["visual_provider_spec_file_sha256"],
            name="visual_provider_spec_file_sha256",
        ),
        "visual-provider specification changed",
    )
    _require(
        _sha256_file(metric_source)
        == sha256_digest(
            manifest["metric_prior_policy_file_sha256"],
            name="metric_prior_policy_file_sha256",
        ),
        "metric-prior policy changed",
    )
    selection = load_strict_json_object(selection_source, label="selection contract")
    provider = load_strict_json_object(provider_source, label="visual-provider spec")
    metric_policy = load_strict_json_object(metric_source, label="metric-prior policy")
    _require(
        selection.get("protocol_id") == protocol_id
        and provider.get("protocol_id") == protocol_id
        and metric_policy.get("protocol_id") == protocol_id,
        "source contracts disagree on protocol_id",
    )
    _require(
        metric_policy.get("frame_selection") == "first retained causal frame"
        and metric_policy.get("future_frames_used") is False
        and metric_policy.get("confirmation_payloads_opened") is False,
        "metric-prior causal policy changed",
    )
    dataset = selection.get("dataset")
    _require(isinstance(dataset, Mapping), "selection dataset record is missing")
    dataset_revision = exact_revision(manifest["dataset_revision"], name="dataset_revision")
    _require(dataset.get("resolved_revision") == dataset_revision, "dataset revision changed")
    provider_record = provider.get("provider")
    motioncrafter_record = provider.get("motioncrafter")
    _require(
        isinstance(provider_record, Mapping) and isinstance(motioncrafter_record, Mapping),
        "visual-provider revisions are missing",
    )
    prob4d_revision = exact_revision(manifest["prob4d_revision"], name="prob4d_revision")
    motioncrafter_revision = exact_revision(
        manifest["motioncrafter_revision"], name="motioncrafter_revision"
    )
    _require(provider_record.get("revision") == prob4d_revision, "Prob4D revision changed")
    _require(
        motioncrafter_record.get("revision") == motioncrafter_revision,
        "MotionCrafter revision changed",
    )
    image_resolution = (
        _integer(motioncrafter_record.get("height"), name="MotionCrafter height", minimum=1),
        _integer(motioncrafter_record.get("width"), name="MotionCrafter width", minimum=1),
    )
    window_size = _integer(
        motioncrafter_record.get("window_size"),
        name="MotionCrafter window_size",
        minimum=1,
    )
    window_overlap = _integer(
        motioncrafter_record.get("overlap"), name="MotionCrafter overlap"
    )
    _require(window_overlap < window_size, "MotionCrafter overlap changed")
    _require(
        provider_record.get("api_version") == 2
        and provider_record.get("export_mode") == "exploratory",
        "source predictions must be the frozen exploratory provider-v2 export",
    )
    _require(
        manifest["point_correlation_cluster_semantics"] == POINT_CLUSTER_SEMANTICS,
        "point correlation-cluster semantics changed",
    )

    boundary = manifest["information_boundary"]
    if not isinstance(boundary, Mapping):
        raise ValueError("information_boundary must be a JSON object")
    require_exact_fields(boundary, expected=_BOUNDARY_FIELDS, name="information_boundary")
    for key, expected in _EXPECTED_BOUNDARY.items():
        observed = genuine_boolean(boundary[key], name=f"information_boundary.{key}")
        _require(observed is expected, f"information boundary changed: {key}")

    raw_selection = selection.get("selection")
    _require(isinstance(raw_selection, Mapping), "locked selection roster is missing")
    calibration = raw_selection.get("calibration")
    confirmation = raw_selection.get("confirmation")
    _require(
        isinstance(calibration, Sequence) and isinstance(confirmation, Sequence),
        "locked selection roles are missing",
    )
    calibration_records = {
        (cast(str, item["object_id"]), cast(int, item["episode_id"])): item
        for item in calibration
        if isinstance(item, Mapping)
    }
    confirmation_ids = frozenset(
        cast(str, item["object_id"])
        for item in confirmation
        if isinstance(item, Mapping)
    )
    cases, prediction_paths = _validate_cases(
        manifest["cases"],
        calibration_records=calibration_records,
        confirmation_object_ids=confirmation_ids,
        prediction_root=Path(prediction_root),
    )

    arrays_record = _file_record(manifest["arrays"], name="arrays")
    arrays_path = _verify_file_record(manifest_source.parent, arrays_record, name="arrays")
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            raw_arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError("cannot load calibration sample arrays") from error
    arrays = _validate_arrays(raw_arrays, cases=cases)

    raw_source_artifacts = manifest["source_artifacts"]
    if not isinstance(raw_source_artifacts, Mapping) or not raw_source_artifacts:
        raise ValueError("source_artifacts must be a nonempty path-to-digest mapping")
    source_digests: set[str] = set()
    for raw_path, raw_digest in raw_source_artifacts.items():
        relative = canonical_relative_posix_path(
            raw_path, name="source artifact path"
        )
        digest = sha256_digest(
            raw_digest, name=f"source_artifacts[{relative!r}]"
        )
        source = _confined_file(
            manifest_source.parent,
            relative,
            name=f"source artifact {relative}",
        )
        _require(_sha256_file(source) == digest, f"source artifact changed: {relative}")
        source_digests.add(digest)
    for case in cases:
        metric = cast(Mapping[str, Any], case["metric_reference"])
        _require(
            metric["source_artifact_sha256"] in source_digests,
            "metric reference source artifact is not present in the portable bundle",
        )
        _require(
            metric["calibration_artifact_sha256"] in source_digests,
            "metric reference calibration artifact is not present in the portable bundle",
        )
        for prediction in cast(Sequence[Mapping[str, Any]], case["prediction_manifests"]):
            source_digests.add(cast(str, prediction["sha256"]))
    source_digests.add(cast(str, arrays_record["sha256"]))
    source_digests.add(_sha256_file(manifest_source))
    return Deform360Prob4DCalibrationSamplesV1(
        manifest_path=manifest_source,
        manifest_file_sha256=_sha256_file(manifest_source),
        bundle_id=supplied_id,
        protocol_id=protocol_id,
        dataset_revision=dataset_revision,
        prob4d_revision=prob4d_revision,
        motioncrafter_revision=motioncrafter_revision,
        image_resolution=image_resolution,
        window_size=window_size,
        window_overlap=window_overlap,
        visual_production_result_id=sha256_digest(
            manifest["visual_production_result_id"], name="visual_production_result_id"
        ),
        cases=cases,
        prediction_manifest_paths=prediction_paths,
        arrays=MappingProxyType(dict(arrays)),
        arrays_file_sha256=cast(str, arrays_record["sha256"]),
        source_artifact_sha256=tuple(sorted(source_digests)),
    )


def _upper_winsorized_mean(values: np.ndarray, *, quantile: float) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    _require(array.ndim == 1 and len(array) > 0, "winsorization needs observations")
    _require(np.all(np.isfinite(array)), "winsorization values must be finite")
    _require(np.isfinite(quantile) and 0.0 < quantile <= 1.0, "invalid quantile")
    if np.all(array == array[0]):
        return max(float(array[0]), 1e-6)
    upper = float(np.quantile(array, quantile))
    clipped = np.minimum(array, upper)
    return max(math.fsum(map(float, clipped)) / len(clipped), 1e-6)


def _regularized_inverse_psd(value: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (value + value.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    floor = max(1e-12, float(np.max(eigenvalues, initial=0.0)) * 1e-12)
    return (eigenvectors * (1.0 / np.maximum(eigenvalues, floor))) @ eigenvectors.T


def fit_object_balanced_gauge_inflation(
    errors: np.ndarray,
    covariance: np.ndarray,
    case_index: np.ndarray,
    *,
    object_ids: Sequence[str],
    trim_quantile: float = 0.99,
) -> tuple[dict[str, float], Mapping[str, Any]]:
    """Fit conservative Sim(3) block factors with equal physical-object mass."""

    residual = np.asarray(errors, dtype=np.float64)
    covariances = np.asarray(covariance, dtype=np.float64)
    groups = np.asarray(case_index, dtype=np.int64)
    _require(residual.ndim == 2 and residual.shape[1] == 7, "gauge errors changed")
    _require(covariances.shape == (len(residual), 7, 7), "gauge covariance changed")
    _require(groups.shape == (len(residual),), "gauge case index changed")
    reports: list[dict[str, Any]] = []
    factors: list[tuple[float, float, float]] = []
    for group, object_id in enumerate(object_ids):
        rows = np.flatnonzero(groups == group)
        _require(len(rows) > 0, "physical object has no gauge residual")
        scale: list[float] = []
        rotation: list[float] = []
        translation: list[float] = []
        for row in rows:
            error = residual[row]
            matrix = covariances[row]
            scale.append(float(error[0] ** 2 / max(matrix[0, 0], 1e-12)))
            for values, block in (
                (rotation, slice(1, 4)),
                (translation, slice(4, 7)),
            ):
                block_error = error[block]
                information = _regularized_inverse_psd(matrix[block, block])
                values.append(float(block_error @ information @ block_error / 3.0))
        group_factors = (
            _upper_winsorized_mean(np.asarray(scale), quantile=trim_quantile),
            _upper_winsorized_mean(np.asarray(rotation), quantile=trim_quantile),
            _upper_winsorized_mean(np.asarray(translation), quantile=trim_quantile),
        )
        factors.append(group_factors)
        reports.append(
            {
                "object_id": object_id,
                "row_count": len(rows),
                "scale_factor": group_factors[0],
                "rotation_factor": group_factors[1],
                "translation_factor": group_factors[2],
            }
        )
    factor_array = np.asarray(factors, dtype=np.float64)
    result = {
        "scale": float(np.mean(factor_array[:, 0])),
        "rotation": float(np.mean(factor_array[:, 1])),
        "translation": float(np.mean(factor_array[:, 2])),
    }
    report = frozen_finite_json_mapping(
        {
            "aggregation": GAUGE_AGGREGATION_SEMANTICS,
            "group_definition": POINT_GROUP_DEFINITION,
            "group_count": len(object_ids),
            "raw_row_count": len(residual),
            "trim_quantile": trim_quantile,
            "groups": reports,
            "factors": result,
        },
        name="gauge calibration report",
    )
    return result, report


def collapse_point_correlation_clusters(
    *,
    errors: np.ndarray,
    ray_directions: np.ndarray,
    parallel_variance: np.ndarray,
    lateral_variance: np.ndarray,
    case_index: np.ndarray,
    cluster_index: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Replace each declared dense dependence cluster by one ratio-equivalent row."""

    rays = np.asarray(ray_directions, dtype=np.float64)
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    residual = np.asarray(errors, dtype=np.float64)
    parallel_error = np.sum(residual * rays, axis=1)
    total_squared = np.sum(residual**2, axis=1)
    lateral_squared = np.maximum(total_squared - parallel_error**2, 0.0)
    parallel_ratio = parallel_error**2 / np.asarray(parallel_variance, dtype=np.float64)
    lateral_ratio = lateral_squared / (2.0 * np.asarray(lateral_variance, dtype=np.float64))
    cases = np.asarray(case_index, dtype=np.int64)
    clusters = np.asarray(cluster_index, dtype=np.int64)
    active = np.asarray(valid, dtype=np.bool_)
    keys = sorted({(int(cases[row]), int(clusters[row])) for row in np.flatnonzero(active)})
    _require(bool(keys), "no valid point correlation clusters")
    synthetic_errors: np.ndarray = np.zeros((len(keys), 3), dtype=np.float64)
    synthetic_rays: np.ndarray = np.zeros((len(keys), 3), dtype=np.float64)
    synthetic_rays[:, 2] = 1.0
    synthetic_parallel: np.ndarray = np.ones(len(keys), dtype=np.float64)
    synthetic_lateral: np.ndarray = np.ones(len(keys), dtype=np.float64)
    synthetic_case: np.ndarray = np.empty(len(keys), dtype=np.int64)
    diagnostics: list[dict[str, int]] = []
    for output_row, (case, cluster) in enumerate(keys):
        rows = active & (cases == case) & (clusters == cluster)
        p_ratio = max(float(np.mean(parallel_ratio[rows])), 1e-12)
        l_ratio = max(float(np.mean(lateral_ratio[rows])), 1e-12)
        synthetic_errors[output_row, 2] = np.sqrt(p_ratio)
        synthetic_errors[output_row, 0] = np.sqrt(2.0 * l_ratio)
        synthetic_case[output_row] = case
        diagnostics.append(
            {"case_index": case, "cluster_index": cluster, "raw_row_count": int(np.sum(rows))}
        )
    report = frozen_finite_json_mapping(
        {
            "semantics": POINT_CLUSTER_SEMANTICS,
            "raw_valid_row_count": int(np.sum(active)),
            "effective_cluster_count": len(keys),
            "clusters": diagnostics,
        },
        name="point cluster report",
    )
    return (
        synthetic_errors,
        synthetic_rays,
        synthetic_parallel,
        synthetic_lateral,
        synthetic_case,
        report,
    )


@dataclass(frozen=True)
class Prob4DCalibrationApi:
    """Narrow runtime surface imported from the pinned Prob4D checkout."""

    module: ModuleType
    GaugeCovarianceCalibrationV1: Any
    MetricGaugeAnchor: Any
    PointUncertaintyCalibrationV1: Any
    Sim3: Any
    StructuredCovariance: Any
    DepthDisagreementModel: Any
    fit_group_balanced_point_uncertainty_calibration: Any
    load_gauge_covariance_calibration: Any
    load_metric_gauge_anchor: Any
    load_point_uncertainty_calibration: Any
    load_prediction_calibration_target: Any
    save_gauge_covariance_calibration: Any
    save_metric_gauge_anchor: Any
    save_point_uncertainty_calibration: Any


def load_pinned_prob4d_api(
    checkout: str | Path,
    *,
    expected_revision: str,
) -> Prob4DCalibrationApi:
    """Import the calibration API only from one clean exact Prob4D checkout."""

    root = Path(checkout).resolve(strict=True)
    _require(root.is_dir(), "Prob4D checkout must be a directory")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(head == expected_revision, "Prob4D checkout revision changed")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status.strip(), "Prob4D checkout must be clean")
    source = root / "src"
    _require(source.is_dir(), "Prob4D checkout has no src directory")
    loaded = sys.modules.get("prob4d")
    if loaded is not None:
        location = Path(cast(str, loaded.__file__)).resolve()
        _require(source in location.parents, "Prob4D was already imported elsewhere")
    sys.path.insert(0, str(source))
    try:
        import prob4d
        from prob4d.calibration import (
            GaugeCovarianceCalibrationV1,
            PointUncertaintyCalibrationV1,
            fit_group_balanced_point_uncertainty_calibration,
            load_gauge_covariance_calibration,
            load_point_uncertainty_calibration,
            save_gauge_covariance_calibration,
            save_point_uncertainty_calibration,
        )
        from prob4d.calibration_compatibility import (
            load_prediction_calibration_target,
        )
        from prob4d.provider_v2 import (
            MetricGaugeAnchor,
            load_metric_gauge_anchor,
            save_metric_gauge_anchor,
        )
        from prob4d.sim3 import Sim3
        from prob4d.uncertainty import (
            DepthDisagreementModel,
            StructuredCovariance,
        )
    finally:
        if sys.path[0] == str(source):
            sys.path.pop(0)
    return Prob4DCalibrationApi(
        module=prob4d,
        GaugeCovarianceCalibrationV1=GaugeCovarianceCalibrationV1,
        MetricGaugeAnchor=MetricGaugeAnchor,
        PointUncertaintyCalibrationV1=PointUncertaintyCalibrationV1,
        Sim3=Sim3,
        StructuredCovariance=StructuredCovariance,
        DepthDisagreementModel=DepthDisagreementModel,
        fit_group_balanced_point_uncertainty_calibration=(
            fit_group_balanced_point_uncertainty_calibration
        ),
        load_gauge_covariance_calibration=load_gauge_covariance_calibration,
        load_metric_gauge_anchor=load_metric_gauge_anchor,
        load_point_uncertainty_calibration=load_point_uncertainty_calibration,
        load_prediction_calibration_target=load_prediction_calibration_target,
        save_gauge_covariance_calibration=save_gauge_covariance_calibration,
        save_metric_gauge_anchor=save_metric_gauge_anchor,
        save_point_uncertainty_calibration=save_point_uncertainty_calibration,
    )


def _calibration_target(samples: Deform360Prob4DCalibrationSamplesV1, api: Any) -> Any:
    targets = [
        api.load_prediction_calibration_target(path)
        for paths in samples.prediction_manifest_paths
        for path in paths
    ]
    _require(bool(targets), "sample bundle has no prediction manifests")
    reference = targets[0].descriptor()
    reference.pop("manifest_sha256")
    for target in targets[1:]:
        observed = target.descriptor()
        observed.pop("manifest_sha256")
        _require(observed == reference, "prediction calibration targets differ")
    _require(
        targets[0].motioncrafter_revision == samples.motioncrafter_revision,
        "prediction manifest MotionCrafter revision changed",
    )
    _require(
        targets[0].image_resolution == samples.image_resolution
        and targets[0].window_size == samples.window_size
        and targets[0].window_overlap == samples.window_overlap,
        "prediction manifest geometry differs from the frozen provider",
    )
    _require(
        str(targets[0].model_identifier).startswith(
            "prob4d.motioncrafter-model.v2:"
        ),
        "prediction manifest does not use the frozen derived-per-call seed policy",
    )
    return targets[0]


def _artifact_record(path: Path, artifact_id: str, root: Path) -> dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "byte_count": path.stat().st_size,
    }


def fit_and_publish_deform360_prob4d_source_calibration(
    samples: Deform360Prob4DCalibrationSamplesV1,
    *,
    api: Any,
    output_directory: str | Path,
    trim_quantile: float = 0.99,
) -> Mapping[str, Any]:
    """Fit and atomically publish source-only point, gauge, and anchor artifacts."""

    _require(
        np.isfinite(trim_quantile) and 0.0 < trim_quantile <= 1.0,
        "trim_quantile must lie in (0, 1]",
    )
    target = _calibration_target(samples, api)
    arrays = samples.arrays
    point_inputs = collapse_point_correlation_clusters(
        errors=arrays["point_errors_m"],
        ray_directions=arrays["point_ray_directions"],
        parallel_variance=arrays["point_parallel_variance_m2"],
        lateral_variance=arrays["point_lateral_variance_m2"],
        case_index=arrays["point_case_index"],
        cluster_index=arrays["point_correlation_cluster_index"],
        valid=arrays["point_valid"],
    )
    (
        cluster_errors,
        cluster_rays,
        cluster_parallel,
        cluster_lateral,
        cluster_case,
        cluster_report,
    ) = point_inputs
    structured = api.StructuredCovariance(
        ray_directions=cluster_rays,
        parallel_variance=cluster_parallel,
        lateral_variance=cluster_lateral,
    )
    group_ids = np.asarray(
        [samples.object_ids[int(index)] for index in cluster_case], dtype=str
    )
    common = {
        "calibration_case_ids": tuple(sorted(samples.case_ids)),
        "source_repository": target.source_repository,
        "source_revision": samples.prob4d_revision,
        "motioncrafter_revision": target.motioncrafter_revision,
        "model_identifier": target.model_identifier,
        "image_resolution": target.image_resolution,
        "window_size": target.window_size,
        "window_overlap": target.window_overlap,
        "covariance_cluster_size": target.covariance_cluster_size,
        "input_artifact_sha256": samples.source_artifact_sha256,
    }
    point, point_report = api.fit_group_balanced_point_uncertainty_calibration(
        api.DepthDisagreementModel(),
        cluster_errors,
        structured,
        group_ids,
        group_definition=POINT_GROUP_DEFINITION,
        covariance_method=target.point_covariance_method,
        trim_quantile=trim_quantile,
        metadata={
            "bayesian_phystwin_protocol_id": samples.protocol_id,
            "calibration_sample_bundle_id": samples.bundle_id,
            "visual_production_result_id": samples.visual_production_result_id,
            "point_correlation_cluster_reduction": plain_json(cluster_report),
            "confirmation_access_authorized": False,
        },
        **common,
    )

    gauge_factors, gauge_report = fit_object_balanced_gauge_inflation(
        arrays["gauge_errors"],
        arrays["gauge_covariance"],
        arrays["gauge_case_index"],
        object_ids=samples.object_ids,
        trim_quantile=trim_quantile,
    )
    gauge = api.GaugeCovarianceCalibrationV1(
        **gauge_factors,
        count=len(samples.object_ids),
        trim_quantile=trim_quantile,
        covariance_method=target.gauge_covariance_method,
        metadata={
            "bayesian_phystwin_protocol_id": samples.protocol_id,
            "calibration_sample_bundle_id": samples.bundle_id,
            "visual_production_result_id": samples.visual_production_result_id,
            "group_balanced_gauge_covariance_calibration": plain_json(gauge_report),
            "calibration_count_semantics": "physical-object-groups",
            "confirmation_access_authorized": False,
        },
        **common,
    )

    output = Path(output_directory).absolute()
    _require(not os.path.lexists(output), f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    )
    try:
        gauge_path = temporary / f"gauge-covariance-{gauge.artifact_id}.json"
        point_path = temporary / f"point-uncertainty-{point.artifact_id}.json"
        api.save_gauge_covariance_calibration(gauge, gauge_path)
        api.save_point_uncertainty_calibration(point, point_path)
        _require(
            api.load_gauge_covariance_calibration(gauge_path).artifact_id
            == gauge.artifact_id,
            "published gauge calibration failed self-verification",
        )
        _require(
            api.load_point_uncertainty_calibration(point_path).artifact_id
            == point.artifact_id,
            "published point calibration failed self-verification",
        )
        anchor_root = temporary / "metric-anchors"
        anchor_root.mkdir()
        anchor_records: list[dict[str, object]] = []
        for case_index, case in enumerate(samples.cases):
            metric = cast(Mapping[str, Any], case["metric_reference"])
            anchor = api.MetricGaugeAnchor(
                window_id=metric["window_id"],
                global_from_local=api.Sim3.from_vector(
                    arrays["anchor_global_from_local"][case_index]
                ),
                covariance=arrays["anchor_covariance"][case_index],
                coordinate_frame=metric["coordinate_frame"],
                source_kind=metric["source_kind"],
                source_artifact_sha256=metric["source_artifact_sha256"],
                metadata={
                    "calibration_artifact_sha256": metric[
                        "calibration_artifact_sha256"
                    ],
                    "bayesian_phystwin_protocol_id": samples.protocol_id,
                    "calibration_sample_bundle_id": samples.bundle_id,
                    "case_id": case["case_id"],
                    "object_id": case["object_id"],
                    "frame_id": metric["frame_id"],
                    "future_frames_used": False,
                    "confirmation_payloads_opened": False,
                },
            )
            anchor_path = anchor_root / f"{case['case_id']}-{anchor.artifact_id}.json"
            api.save_metric_gauge_anchor(anchor_path, anchor)
            _require(
                api.load_metric_gauge_anchor(anchor_path).artifact_id == anchor.artifact_id,
                "published metric anchor failed self-verification",
            )
            anchor_records.append(
                {
                    "case_id": case["case_id"],
                    "object_id": case["object_id"],
                    **_artifact_record(anchor_path, anchor.artifact_id, temporary),
                }
            )

        descriptor: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "schema_version": RESULT_VERSION,
            "semantics": RESULT_SEMANTICS,
            "protocol_id": samples.protocol_id,
            "calibration_sample_bundle_id": samples.bundle_id,
            "calibration_sample_manifest_sha256": samples.manifest_file_sha256,
            "visual_production_result_id": samples.visual_production_result_id,
            "prob4d_revision": samples.prob4d_revision,
            "motioncrafter_revision": samples.motioncrafter_revision,
            "physical_object_count": len(samples.object_ids),
            "stratum_counts": {
                stratum: sum(case["stratum"] == stratum for case in samples.cases)
                for stratum in sorted({cast(str, case["stratum"]) for case in samples.cases})
            },
            "point_effective_cluster_count": len(cluster_errors),
            "gauge_raw_row_count": len(arrays["gauge_errors"]),
            "artifacts": {
                "gauge_covariance": _artifact_record(
                    gauge_path, gauge.artifact_id, temporary
                ),
                "point_uncertainty": _artifact_record(
                    point_path, point.artifact_id, temporary
                ),
                "metric_anchors": anchor_records,
            },
            "reports": {
                "point": plain_json(point_report.to_dict()),
                "point_correlation_clusters": plain_json(cluster_report),
                "gauge": plain_json(gauge_report),
            },
            "information_boundary": {
                **_EXPECTED_BOUNDARY,
                "confirmation_access_authorized": False,
                "calibration_gate_evaluated": False,
            },
            "claim_boundary": (
                "Source-only covariance and metric-anchor fit. This artifact does not "
                "establish transfer, predictive coverage, physical-query benefit, "
                "confirmation access, or state of the art."
            ),
        }
        result = {"result_id": content_id(descriptor), **descriptor}
        result_path = temporary / "source-calibration-result.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        checksum_paths = sorted(
            (path for path in temporary.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(temporary).as_posix(),
        )
        (temporary / "SHA256SUMS").write_text(
            "".join(
                f"{_sha256_file(path)}  {path.relative_to(temporary).as_posix()}\n"
                for path in checksum_paths
            ),
            encoding="ascii",
        )
        os.rename(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return frozen_finite_json_mapping(result, name="source calibration result")


__all__ = [
    "GAUGE_AGGREGATION_SEMANTICS",
    "MINIMUM_OBJECTS_PER_STRATUM",
    "MINIMUM_OBJECT_COUNT",
    "POINT_CLUSTER_SEMANTICS",
    "POINT_GROUP_DEFINITION",
    "RESULT_SCHEMA",
    "RESULT_SEMANTICS",
    "RESULT_VERSION",
    "SAMPLE_SCHEMA",
    "SAMPLE_SEMANTICS",
    "SAMPLE_VERSION",
    "Deform360Prob4DCalibrationSamplesV1",
    "Prob4DCalibrationApi",
    "collapse_point_correlation_clusters",
    "fit_and_publish_deform360_prob4d_source_calibration",
    "fit_object_balanced_gauge_inflation",
    "load_deform360_prob4d_calibration_samples",
    "load_pinned_prob4d_api",
]
