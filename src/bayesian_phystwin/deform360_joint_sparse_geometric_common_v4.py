"""Common contracts for Deform360 geometric v4 materialization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import plain_json
from ._portable_contracts import (
    content_id,
    exact_revision,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
)
from .deform360_joint_sparse_observability_v4 import (
    DEFORM360_JOINT_SPARSE_PROTOCOL_ID,
    default_deform360_joint_sparse_information_boundary_v4,
)

MATERIALIZER_POLICY_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-geometric-materializer-policy"
)
MATERIALIZER_POLICY_VERSION: Final = 1
MATERIALIZER_POLICY_SEMANTICS: Final = (
    "target-free-released-robot-metric-and-prediction-support-geometric-modes-v1"
)
MATERIALIZER_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-geometric-materialization"
)
MATERIALIZER_VERSION: Final = 1
V4_MANIFEST_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-observability-manifest"
)
V4_MANIFEST_VERSION: Final = 4
METRIC_BATCH_SCHEMA: Final = "bayesian-phystwin.deform360-prob4d-metric-batch"
METRIC_BATCH_VERSION: Final = 2
METRIC_BATCH_SEMANTICS: Final = (
    "target-free-robot-visible-calibration-streams-released-robot-gauge-v2"
)
METRIC_PLAN_SCHEMA: Final = "bayesian-phystwin.deform360-prob4d-metric-prefix-plan"
METRIC_PLAN_VERSION: Final = 2
METRIC_PLAN_SEMANTICS: Final = (
    "target-free-robot-visible-integrity-bound-streams-with-causal-public-metric-prefix-v2"
)
MOTIONCRAFTER_INTEGRITY_SCHEMA: Final = (
    "prob4d.motioncrafter-artifact-integrity.v1"
)
METRIC_ARRAY_MEMBERS: Final = frozenset(
    {"frame_indices.npy", "points_world_m.npy", "valid_mask.npy"}
)
PREDICTION_REQUIRED_MEMBERS: Final = frozenset(
    {"window_id.npy", "frame_indices.npy", "valid_mask.npy"}
)
PREDICTION_ALLOWED_MEMBERS: Final = frozenset(
    {
        "window_id.npy",
        "frame_indices.npy",
        "point_map.npy",
        "valid_mask.npy",
        "scene_flow.npy",
        "deform_mask.npy",
        "ray_directions.npy",
    }
)
FILE_FIELDS: Final = frozenset({"path", "sha256", "byte_count"})
PLAN_CASE_FIELDS: Final = frozenset(
    {
        "case_id",
        "object_id",
        "episode_id",
        "stratum",
        "causal_frame_range_half_open",
        "streams",
    }
)
PLAN_STREAM_FIELDS: Final = frozenset(
    {"job_id", "camera_id", "prediction_manifest", "metric_prefix", "metric_calibration"}
)
PLAN_EXCLUDED_FIELDS: Final = frozenset(
    {"job_id", "object_id", "episode_id", "stratum", "camera_id", "reason"}
)
POLICY_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "protocol_id",
        "development_cohort_role",
        "selection_artifact_sha256",
        "visual_provider_lock_id",
        "production_result_id",
        "metric_batch_result_id",
        "metric_batch_implementation_revision",
        "prob4d_revision",
        "motioncrafter_revision",
        "metric_source_kind",
        "basis_semantics",
        "gauge_semantics",
        "query_semantics",
        "world_voxel_size_m",
        "maximum_factors_per_camera_window",
        "lateral_observation_std_m",
        "axial_observation_std_m",
        "minimum_object_rms_radius_m",
        "root_gauge_prior_std_m",
        "camera_gauge_innovation_std_m",
        "correlation_group_semantics",
        "composite_weight_semantics",
        "support_mask_source",
        "robot_metric_points_used",
        "camera_calibration_used",
        "prediction_support_masks_used",
        "prediction_point_values_used",
        "prediction_residuals_used",
        "calibration_outcomes_used",
        "future_frames_used",
        "adaptive_confirmation_payloads_opened",
        "confirmation_payloads_opened",
        "target_outcomes_used",
        "replacement_allowed",
        "human_approval_required",
        "new_measurements_required",
        "claim_boundary",
        "artifact_id",
    }
)
MATERIALIZER_CLAIM_BOUNDARY: Final = (
    "Development-only structural observability materialization from released robot "
    "metric geometry and integrity-bound prediction support masks. It does not "
    "establish Prob4D calibration, BayesianPhysTwin physical-query benefit, "
    "confirmation accuracy, Causal4D benefit, deployment safety, or state of the art."
)
EXPECTED_DEVELOPMENT_BOUNDARY: Final = plain_json(
    default_deform360_joint_sparse_information_boundary_v4()
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _literal(value: object, *, name: str) -> str:
    result = nonempty_string(value, name=name)
    _require(result == result.strip(), f"{name} has surrounding whitespace")
    return result


def _positive(value: object, *, name: str) -> float:
    _require(not isinstance(value, (bool, np.bool_)), f"{name} is not numeric")
    raw = np.asarray(value)
    _require(raw.shape == () and raw.dtype.kind in "iuf", f"{name} is not scalar")
    result = float(raw.item())
    _require(np.isfinite(result) and result > 0.0, f"{name} must be finite and positive")
    return result


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{name} is invalid")
    return cast(int, value)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _ordinary_directory(path: Path, *, name: str) -> Path:
    _require(not path.is_symlink(), f"{name} must not be a symbolic link")
    result = path.resolve(strict=True)
    _require(result.is_dir() and not result.is_symlink(), f"{name} is not a directory")
    return result


def _ordinary_file(path: Path, *, name: str) -> Path:
    _require(not path.is_symlink(), f"{name} must not be a symbolic link")
    result = path.resolve(strict=True)
    _require(result.is_file() and not result.is_symlink(), f"{name} is not a file")
    return result


def _safe_relative(value: object, *, name: str) -> str:
    text = _literal(value, name=name)
    _require("\\" not in text, f"{name} is not POSIX")
    path = PurePosixPath(text)
    _require(not path.is_absolute(), f"{name} must be relative")
    _require(path.as_posix() == text, f"{name} is not canonical")
    _require(all(part not in {"", ".", ".."} for part in path.parts), f"{name} is unsafe")
    return text


def _confined_file(root: Path, relative: object, *, name: str) -> Path:
    text = _safe_relative(relative, name=name)
    current = root
    for part in PurePosixPath(text).parts:
        current /= part
        _require(not current.is_symlink(), f"{name} traverses a symbolic link")
    result = current.resolve(strict=True)
    _require(root == result or root in result.parents, f"{name} escapes its root")
    _require(result.is_file(), f"{name} is not a regular file")
    return result


def _file_record(value: object, *, name: str) -> dict[str, object]:
    _require(isinstance(value, Mapping), f"{name} is not a file record")
    record = cast(Mapping[str, Any], value)
    require_exact_fields(record, expected=FILE_FIELDS, name=name)
    return {
        "path": _safe_relative(record["path"], name=f"{name}.path"),
        "sha256": sha256_digest(record["sha256"], name=f"{name}.sha256"),
        "byte_count": _integer(record["byte_count"], name=f"{name}.byte_count", minimum=1),
    }


def _verify_record(root: Path, value: object, *, name: str) -> tuple[Path, dict[str, object]]:
    record = _file_record(value, name=name)
    path = _confined_file(root, record["path"], name=f"{name}.path")
    _require(path.stat().st_size == record["byte_count"], f"{name} byte count changed")
    _require(_sha256_file(path) == record["sha256"], f"{name} SHA-256 changed")
    return path, record


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plain_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_materializer_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen, source-independent geometric materializer policy."""

    require_exact_fields(value, expected=POLICY_FIELDS, name="geometric materializer policy")
    _require(value["schema"] == MATERIALIZER_POLICY_SCHEMA, "materializer policy schema changed")
    _require(value["schema_version"] == MATERIALIZER_POLICY_VERSION, "materializer policy version changed")
    _require(value["semantics"] == MATERIALIZER_POLICY_SEMANTICS, "materializer policy semantics changed")
    _require(value["protocol_id"] == DEFORM360_JOINT_SPARSE_PROTOCOL_ID, "materializer protocol changed")
    _require(value["development_cohort_role"] == "opened-v1-v3-development-only", "cohort role changed")
    for name in (
        "selection_artifact_sha256",
        "visual_provider_lock_id",
        "production_result_id",
        "metric_batch_result_id",
    ):
        sha256_digest(value[name], name=name)
    for name in (
        "metric_batch_implementation_revision",
        "prob4d_revision",
        "motioncrafter_revision",
    ):
        exact_revision(value[name], name=name)
    for name in (
        "world_voxel_size_m",
        "lateral_observation_std_m",
        "axial_observation_std_m",
        "minimum_object_rms_radius_m",
        "root_gauge_prior_std_m",
        "camera_gauge_innovation_std_m",
    ):
        _positive(value[name], name=name)
    _integer(
        value["maximum_factors_per_camera_window"],
        name="maximum_factors_per_camera_window",
        minimum=1,
    )
    _require(
        float(value["axial_observation_std_m"])
        >= float(value["lateral_observation_std_m"]),
        "axial observation scale is smaller than lateral scale",
    )
    for name in (
        "metric_source_kind",
        "basis_semantics",
        "gauge_semantics",
        "query_semantics",
        "correlation_group_semantics",
        "composite_weight_semantics",
        "support_mask_source",
    ):
        _literal(value[name], name=name)
    expected_flags = {
        "robot_metric_points_used": True,
        "camera_calibration_used": True,
        "prediction_support_masks_used": True,
        "prediction_point_values_used": False,
        "prediction_residuals_used": False,
        "calibration_outcomes_used": False,
        "future_frames_used": False,
        "adaptive_confirmation_payloads_opened": False,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
        "replacement_allowed": False,
        "human_approval_required": False,
        "new_measurements_required": False,
    }
    for name, expected in expected_flags.items():
        _require(type(value[name]) is bool and value[name] is expected, f"materializer boundary changed: {name}")
    _require(value["claim_boundary"] == MATERIALIZER_CLAIM_BOUNDARY, "materializer claim boundary changed")
    identity = dict(value)
    declared = sha256_digest(identity.pop("artifact_id"), name="artifact_id")
    _require(content_id(identity) == declared, "materializer policy ID changed")
    return cast(dict[str, Any], plain_json(value))


def _verify_recursive_checksums(root: Path) -> None:
    checksum = _ordinary_file(root / "SHA256SUMS", name="metric batch SHA256SUMS")
    lines = checksum.read_text(encoding="ascii").splitlines()
    observed: set[str] = set()
    for index, line in enumerate(lines):
        digest, separator, relative = line.partition("  ")
        _require(bool(separator), f"checksum line {index} is malformed")
        sha256_digest(digest, name=f"checksum line {index}")
        path = _confined_file(root, relative, name=f"checksum member {index}")
        _require(relative not in observed, "metric batch checksum path repeats")
        observed.add(relative)
        _require(_sha256_file(path) == digest, f"metric batch checksum changed: {relative}")
    expected = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    _require(observed == expected, "metric batch checksum roster changed")


def _selection_rows(value: Mapping[str, Any]) -> dict[tuple[str, int], str]:
    selection = value.get("selection")
    _require(isinstance(selection, Mapping), "selection root changed")
    calibration = cast(Mapping[str, Any], selection).get("calibration")
    _require(isinstance(calibration, list) and len(calibration) == 10, "development selection changed")
    rows: dict[tuple[str, int], str] = {}
    for index, raw in enumerate(calibration):
        _require(isinstance(raw, Mapping), f"selection row {index} changed")
        row = cast(Mapping[str, Any], raw)
        identity = (
            _literal(row.get("object_id"), name="selection object_id"),
            _integer(row.get("episode_id"), name="selection episode_id"),
        )
        stratum = _literal(row.get("stratum"), name="selection stratum")
        _require(stratum in {"sheet", "volumetric"}, "selection stratum changed")
        _require(identity not in rows, "selection repeats an object")
        rows[identity] = stratum
    return rows
