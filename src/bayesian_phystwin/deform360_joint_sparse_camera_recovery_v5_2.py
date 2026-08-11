"""Outcome-blind camera recovery for the public Deform360 v5 source panel.

The original v5.1 source runner treated one camera-local metric-gauge failure as
an object-wide technical failure.  This module provides a transparent additive
amendment: audit cameras independently, rank extra public cameras only by
released robot-geometry support, and require at least two independently passing
views.  It never reads the development suffix, a confirmation payload, or a
physical-state innovation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import genuine_integer, plain_json
from ._portable_contracts import (
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
    write_atomic_json,
)
from .deform360_calibration_visual_execution_admission import (
    validate_deform360_prepared_source_inventory,
)
from .deform360_joint_sparse_endpoint_v5 import select_reserved_endpoint_views_v5
from .deform360_joint_sparse_materializer_v5 import Deform360JointSparsePrefixFitV5
from .deform360_joint_sparse_motioncrafter_source_v5 import (
    MOTIONCRAFTER_MODEL_SET_ID,
    MOTIONCRAFTER_REVISION,
    PROB4D_REVISION,
    PROVIDER_FRAME_COUNT,
    RUN_CONFIGURATION,
    TEMPORAL_POLICY,
    _camera_video,
    _seed_schedule,
    _windows,
    validate_deform360_joint_sparse_motioncrafter_source_plan_v5,
)
from .deform360_joint_sparse_public_inputs_v5 import (
    prepare_deform360_joint_sparse_visual_window_v5,
)
from .deform360_joint_sparse_source_runner_v5 import (
    build_deform360_joint_sparse_source_prediction_plan_v5,
    validate_deform360_joint_sparse_source_prediction_plan_v5,
)
from .deform360_robot_metric_prefix import (
    METRIC_MANIFEST_FILENAME,
    METRIC_PREFIX_FILENAME,
    validate_deform360_robot_metric_prefix,
)

CAMERA_AUDIT_SCHEMA: Final = "bayesian-phystwin.deform360-joint-sparse-camera-audit"
CAMERA_AUDIT_VERSION: Final = 1
CAMERA_AUDIT_SEMANTICS: Final = (
    "independent-public-prefix-camera-gauge-audit-before-suffix-v1"
)
CAMERA_RECOVERY_PREFLIGHT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-camera-recovery-preflight"
)
CAMERA_RECOVERY_PREFLIGHT_VERSION: Final = 1
CAMERA_RECOVERY_PREFLIGHT_SEMANTICS: Final = (
    "robot-geometry-ranked-public-prefix-camera-recovery-v1"
)
CAMERA_RECOVERY_AMENDMENT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-camera-recovery-amendment"
)
CAMERA_RECOVERY_AMENDMENT_VERSION: Final = 1
CAMERA_RECOVERY_AMENDMENT_SEMANTICS: Final = (
    "additive-outcome-blind-camera-failure-granularity-repair-v1"
)
RECOVERY_PROVIDER_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-motioncrafter-camera-recovery-plan"
)
RECOVERY_PROVIDER_VERSION: Final = 1
RECOVERY_PROVIDER_SEMANTICS: Final = (
    "latest-42-of-locked-58-frame-public-prefix-ranked-camera-recovery-v1"
)
RECOVERY_PROVIDER_STATUS: Final = "locked-before-recovery-provider-inference"
PROVIDER_RUN_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-motioncrafter-source-run"
)
PROVIDER_RUN_VERSION: Final = 1

MINIMUM_INDEPENDENT_CLUSTERS: Final = 8
METRIC_CLUSTER_SIZE_PIXELS: Final = 32
MINIMUM_PASSING_CAMERAS: Final = 2
MAXIMUM_ADDITIONAL_CAMERAS: Final = 4

RECOVERY_INFORMATION_BOUNDARY: Final = {
    "public_source_prefix_payloads_authorized": True,
    "provider_outputs_opened": False,
    "development_suffix_opened": False,
    "future_object_observations_used": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "human_approval_required": False,
    "new_measurements_required": False,
}

AUDIT_INFORMATION_BOUNDARY: Final = {
    "public_source_prefix_payloads_opened": True,
    "physical_state_innovation_used": False,
    "development_suffix_opened": False,
    "future_object_observations_used": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "human_approval_used": False,
    "new_measurements_collected": False,
}

RECOVERY_POLICY: Final = {
    "camera_failure_scope": "per-camera-never-object-wide-v1",
    "minimum_passing_camera_count": MINIMUM_PASSING_CAMERAS,
    "metric_cluster_size_pixels": METRIC_CLUSTER_SIZE_PIXELS,
    "minimum_independent_clusters": MINIMUM_INDEPENDENT_CLUSTERS,
    "recovery_trigger": "fewer-than-two-passing-base-cameras",
    "candidate_set": ("all-released-nonreserved-cameras-not-already-attempted"),
    "ranking": [
        "maximum-independent-cluster-count-descending",
        "qualifying-causal-frame-count-descending",
        "total-projected-point-count-descending",
        "camera-id-ascending",
    ],
    "maximum_additional_cameras": MAXIMUM_ADDITIONAL_CAMERAS,
    "post_provider_rule": "retain-all-and-only-independently-passing-cameras",
    "insufficient_support_action": "exact-B0-physical-fallback",
    "metric_gauge_threshold_relaxation_allowed": False,
    "camera_replacement_allowed": False,
    "base_prediction_batch_mutation_allowed": False,
}

RECOVERY_CLAIM_BOUNDARY: Final = (
    "Source-prefix camera admission and provider execution only. The recovery "
    "uses public Deform360 recordings and establishes no prediction benefit, "
    "calibration, confirmation, safety, Causal4D, or state-of-the-art claim."
)

PROVIDER_RUN_INFORMATION_BOUNDARY: Final = {
    "provider_outputs_opened_for_integrity": True,
    "development_suffix_opened": False,
    "future_object_observations_used": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "human_approval_required": False,
    "new_measurements_required": False,
}
PROVIDER_RUN_CLAIM_BOUNDARY: Final = (
    "Public source-prefix provider inference and integrity only. No "
    "development suffix, prediction score, confirmation payload, or target "
    "outcome was opened."
)

_GAUGE_FAILURE_MESSAGES = frozenset(
    {
        "metric gauge lacks eight independent causal clusters",
        "metric gauge has fewer than eight independent inlier clusters",
    }
)
_FILE_FIELDS = frozenset({"path", "sha256"})
_AUDIT_CAMERA_FIELDS = frozenset(
    {
        "camera_id",
        "decoded_uniform_sha256",
        "failure_code",
        "gauge_artifact_id",
        "independent_cluster_count",
        "inlier_independent_cluster_count",
        "inlier_rmse_m",
        "metric_prefix_sha256",
        "raw_frame_index",
        "status",
    }
)
_AUDIT_OBJECT_FIELDS = frozenset(
    {
        "attempted_camera_ids",
        "camera_results",
        "failed_camera_ids",
        "object_id",
        "passing_camera_ids",
    }
)
_AUDIT_FIELDS = frozenset(
    {
        "audit_id",
        "base_source_plan_id",
        "execution_lock_id",
        "implementation_revision",
        "information_boundary",
        "objects",
        "schema",
        "schema_version",
        "semantics",
    }
)
_METRIC_SUPPORT_FIELDS = frozenset(
    {
        "artifact_id",
        "camera_id",
        "eligible",
        "frames_with_minimum_clusters",
        "manifest_file_sha256",
        "max_independent_cluster_count",
        "metric_prefix_file_sha256",
        "total_projected_point_count",
    }
)
_PREFLIGHT_OBJECT_FIELDS = frozenset(
    {
        "base_attempted_camera_ids",
        "base_passing_camera_ids",
        "candidate_metric_support",
        "object_id",
        "ranked_eligible_camera_ids",
        "recovery_required",
        "selected_recovery_camera_ids",
    }
)
_PREFLIGHT_FIELDS = frozenset(
    {
        "base_camera_audit_id",
        "base_camera_audit_file_sha256",
        "base_provider_plan_id",
        "base_provider_plan_file_sha256",
        "execution_lock_id",
        "information_boundary",
        "objects",
        "policy",
        "preflight_id",
        "schema",
        "schema_version",
        "semantics",
    }
)
_RECOVERY_PROVIDER_FIELDS = frozenset(
    {
        "base_provider_plan",
        "camera_recovery_amendment",
        "camera_recovery_preflight",
        "camera_roster_source",
        "claim_boundary",
        "implementation",
        "information_boundary",
        "job_count",
        "jobs",
        "manifest_sha256",
        "motioncrafter",
        "object_count",
        "objects",
        "prepared_source_inventory",
        "provider_lock",
        "role",
        "run_configuration",
        "schema",
        "schema_version",
        "semantics",
        "smoke_job_id",
        "source_execution_lock",
        "status",
        "temporal_policy",
    }
)
_RECOVERY_JOB_FIELDS = frozenset(
    {
        "camera",
        "episode_id",
        "job_id",
        "likelihood_eligible",
        "object_id",
        "output_relative_path",
        "seed_schedule",
        "source_episode",
        "source_frame_count",
        "source_frame_start",
        "source_frame_stop_exclusive",
        "source_video",
        "stratum",
        "windows",
    }
)
_RUN_REPORT_FIELDS = frozenset(
    {
        "claim_boundary",
        "completed_job_count",
        "completed_jobs",
        "information_boundary",
        "mode",
        "requested_job_count",
        "run_sha256",
        "runtime_revision",
        "schema",
        "schema_version",
        "shard_count",
        "shard_index",
        "source_plan_sha256",
        "status",
    }
)
_COMPLETED_JOB_FIELDS = frozenset(
    {
        "job_id",
        "prediction_manifest",
        "prediction_manifest_sha256",
        "verification",
    }
)
_VERIFICATION_FIELDS = frozenset(
    {
        "hashes_verified",
        "integrity_bound",
        "manifest_path",
        "member_count",
        "run_spec_sha256",
    }
)
_DECODED_REPORT_FIELDS = frozenset(
    {
        "completed_at_utc",
        "covariance_units",
        "fixed_prob4d_vggt_blend",
        "frame_count",
        "manifest",
        "manifest_sha256",
        "maximum_contributors",
        "method",
        "output_npz",
        "output_npz_sha256",
        "overlap_pixel_fraction",
        "prob4d_revision",
        "prob4d_root",
        "schema_version",
    }
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _identifier(value: object, *, name: str) -> str:
    result = nonempty_string(value, name=name)
    _require(result == result.strip() and "\x00" not in result, f"invalid {name}")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(value: object, *, name: str) -> str:
    result = _identifier(value, name=name)
    path = PurePosixPath(result)
    _require(
        result == path.as_posix()
        and not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"{name} must be a safe POSIX relative path",
    )
    return result


def _ordinary_root(path: str | Path) -> Path:
    requested = Path(path).absolute()
    _require(
        requested.is_dir()
        and not requested.is_symlink()
        and not any(parent.is_symlink() for parent in requested.parents),
        "input root must be an ordinary non-symlink directory",
    )
    return requested.resolve(strict=True)


def _verified_file(root: Path, value: object, *, name: str) -> Path:
    record = _mapping(value, name=name)
    require_exact_fields(record, expected=_FILE_FIELDS, name=name)
    relative = _safe_relative_path(record.get("path"), name=f"{name}.path")
    expected = sha256_digest(record.get("sha256"), name=f"{name}.sha256")
    requested = root / relative
    _require(
        requested.is_file()
        and not requested.is_symlink()
        and not any(parent.is_symlink() for parent in requested.parents),
        f"{name} must be an ordinary file",
    )
    resolved = requested.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes the input root") from error
    _require(_sha256_file(resolved) == expected, f"{name} digest changed")
    return resolved


def _content_addressed(value: Mapping[str, Any], *, identity: str) -> None:
    declared = sha256_digest(value.get(identity), name=identity)
    descriptor = dict(value)
    descriptor.pop(identity)
    _require(content_id(descriptor) == declared, f"{identity} changed")


def _cohort(lock: Mapping[str, Any]) -> tuple[str, ...]:
    cohort = _mapping(lock.get("cohort"), name="cohort")
    rows = _sequence(cohort.get("development_objects"), name="development objects")
    result = tuple(
        sorted(
            _identifier(
                _mapping(row, name="development object").get("object_id"),
                name="object_id",
            )
            for row in rows
        )
    )
    _require(len(result) == 10 and len(set(result)) == 10, "cohort changed")
    return result


def audit_deform360_joint_sparse_source_cameras_v5_2(
    *,
    lock: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    input_root: str | Path,
) -> dict[str, Any]:
    """Audit each base visual camera independently using prefix data only."""

    normalized_plan = validate_deform360_joint_sparse_source_prediction_plan_v5(
        source_plan,
        lock=lock,
    )
    root = _ordinary_root(input_root)
    cohort = _cohort(lock)
    fit = Deform360JointSparsePrefixFitV5(
        fit_object_ids=cohort,
        source_artifact_ids={
            "source-plan-id": cast(str, normalized_plan["plan_id"]),
        },
    )
    objects: list[dict[str, Any]] = []
    for raw_object in cast(Sequence[Mapping[str, Any]], normalized_plan["objects"]):
        object_id = cast(str, raw_object["object_id"])
        prefix = cast(Sequence[int], raw_object["raw_prefix_range_half_open"])
        results: list[dict[str, Any]] = []
        for raw_window in cast(
            Sequence[Mapping[str, Any]], raw_object["visual_windows"]
        ):
            camera_id = cast(str, raw_window["camera_id"])
            decoded = _verified_file(
                root,
                raw_window["decoded_uniform"],
                name=f"{object_id}/{camera_id} decoded uniform",
            )
            metric = _verified_file(
                root,
                raw_window["metric_prefix"],
                name=f"{object_id}/{camera_id} metric prefix",
            )
            common = {
                f"decoded/{object_id}/{camera_id}.npz": _sha256_file(decoded),
                f"metric/{object_id}/{camera_id}.npz": _sha256_file(metric),
            }
            try:
                _rows, gauge = prepare_deform360_joint_sparse_visual_window_v5(
                    camera_id=camera_id,
                    decoded_uniform_path=decoded,
                    metric_prefix_path=metric,
                    raw_prefix_range_half_open=(int(prefix[0]), int(prefix[1])),
                    fit=fit,
                    source_artifact_ids=common,
                    metric_cluster_size_pixels=METRIC_CLUSTER_SIZE_PIXELS,
                )
            except ValueError as error:
                message = str(error)
                if message not in _GAUGE_FAILURE_MESSAGES:
                    raise
                results.append(
                    {
                        "camera_id": camera_id,
                        "status": "rejected",
                        "failure_code": message.replace(" ", "-"),
                        "decoded_uniform_sha256": _sha256_file(decoded),
                        "metric_prefix_sha256": _sha256_file(metric),
                        "gauge_artifact_id": None,
                        "raw_frame_index": None,
                        "independent_cluster_count": None,
                        "inlier_independent_cluster_count": None,
                        "inlier_rmse_m": None,
                    }
                )
            else:
                results.append(
                    {
                        "camera_id": camera_id,
                        "status": "passed",
                        "failure_code": None,
                        "decoded_uniform_sha256": _sha256_file(decoded),
                        "metric_prefix_sha256": _sha256_file(metric),
                        "gauge_artifact_id": gauge.artifact_id,
                        "raw_frame_index": gauge.raw_frame_index,
                        "independent_cluster_count": gauge.independent_cluster_count,
                        "inlier_independent_cluster_count": (
                            gauge.inlier_independent_cluster_count
                        ),
                        "inlier_rmse_m": gauge.inlier_rmse_m,
                    }
                )
        results.sort(key=lambda item: cast(str, item["camera_id"]))
        attempted = [cast(str, item["camera_id"]) for item in results]
        passing = [
            cast(str, item["camera_id"])
            for item in results
            if item["status"] == "passed"
        ]
        failed = sorted(set(attempted) - set(passing))
        objects.append(
            {
                "object_id": object_id,
                "attempted_camera_ids": attempted,
                "passing_camera_ids": passing,
                "failed_camera_ids": failed,
                "camera_results": results,
            }
        )
    objects.sort(key=lambda item: cast(str, item["object_id"]))
    identity: dict[str, Any] = {
        "schema": CAMERA_AUDIT_SCHEMA,
        "schema_version": CAMERA_AUDIT_VERSION,
        "semantics": CAMERA_AUDIT_SEMANTICS,
        "execution_lock_id": lock["execution_lock_id"],
        "base_source_plan_id": normalized_plan["plan_id"],
        "implementation_revision": normalized_plan["implementation_revision"],
        "objects": objects,
        "information_boundary": dict(AUDIT_INFORMATION_BOUNDARY),
    }
    result = {**identity, "audit_id": content_id(identity)}
    validate_deform360_joint_sparse_camera_audit_v5_2(result, lock=lock)
    return result


def validate_deform360_joint_sparse_camera_audit_v5_2(
    value: object,
    *,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one content-addressed independent-camera audit."""

    audit = _mapping(value, name="camera audit")
    require_exact_fields(audit, expected=_AUDIT_FIELDS, name="camera audit")
    _content_addressed(audit, identity="audit_id")
    _require(
        audit.get("schema") == CAMERA_AUDIT_SCHEMA
        and audit.get("schema_version") == CAMERA_AUDIT_VERSION
        and audit.get("semantics") == CAMERA_AUDIT_SEMANTICS,
        "camera-audit contract changed",
    )
    _require(
        audit.get("execution_lock_id") == lock.get("execution_lock_id"),
        "camera audit uses a different execution lock",
    )
    exact_revision(audit.get("implementation_revision"), name="implementation revision")
    _require(
        audit.get("information_boundary") == AUDIT_INFORMATION_BOUNDARY,
        "camera-audit boundary changed",
    )
    seen: set[str] = set()
    for raw_object in _sequence(audit.get("objects"), name="audit objects"):
        row = _mapping(raw_object, name="audit object")
        require_exact_fields(row, expected=_AUDIT_OBJECT_FIELDS, name="audit object")
        object_id = _identifier(row.get("object_id"), name="object_id")
        _require(object_id not in seen, "camera audit repeats an object")
        seen.add(object_id)
        results = _sequence(row.get("camera_results"), name="camera results")
        cameras: list[str] = []
        passing: list[str] = []
        failed: list[str] = []
        for raw_result in results:
            result = _mapping(raw_result, name="camera result")
            require_exact_fields(
                result, expected=_AUDIT_CAMERA_FIELDS, name="camera result"
            )
            camera = _identifier(result.get("camera_id"), name="camera_id")
            cameras.append(camera)
            sha256_digest(
                result.get("decoded_uniform_sha256"),
                name="decoded_uniform_sha256",
            )
            sha256_digest(
                result.get("metric_prefix_sha256"), name="metric_prefix_sha256"
            )
            status = result.get("status")
            _require(status in {"passed", "rejected"}, "camera status changed")
            if status == "passed":
                _require(
                    result.get("failure_code") is None
                    and result.get("raw_frame_index") is not None
                    and result.get("independent_cluster_count") is not None
                    and result.get("inlier_independent_cluster_count") is not None
                    and result.get("inlier_rmse_m") is not None,
                    "passing camera lacks gauge evidence",
                )
                sha256_digest(result.get("gauge_artifact_id"), name="gauge_artifact_id")
                _require(
                    genuine_integer(
                        result.get("independent_cluster_count"),
                        name="independent_cluster_count",
                        minimum=MINIMUM_INDEPENDENT_CLUSTERS,
                    )
                    >= MINIMUM_INDEPENDENT_CLUSTERS
                    and genuine_integer(
                        result.get("inlier_independent_cluster_count"),
                        name="inlier_independent_cluster_count",
                        minimum=MINIMUM_INDEPENDENT_CLUSTERS,
                    )
                    >= MINIMUM_INDEPENDENT_CLUSTERS,
                    "passing camera weakens the metric gauge",
                )
                passing.append(camera)
            else:
                _require(
                    result.get("failure_code")
                    in {
                        message.replace(" ", "-") for message in _GAUGE_FAILURE_MESSAGES
                    }
                    and all(
                        result.get(field) is None
                        for field in (
                            "gauge_artifact_id",
                            "raw_frame_index",
                            "independent_cluster_count",
                            "inlier_independent_cluster_count",
                            "inlier_rmse_m",
                        )
                    ),
                    "rejected camera has unsupported failure evidence",
                )
                failed.append(camera)
        _require(
            cameras == sorted(set(cameras))
            and row.get("attempted_camera_ids") == cameras
            and row.get("passing_camera_ids") == passing
            and row.get("failed_camera_ids") == failed,
            "camera-audit accounting changed",
        )
    _require(tuple(sorted(seen)) == _cohort(lock), "camera-audit cohort changed")
    return cast(dict[str, Any], plain_json(audit))


def summarize_deform360_metric_camera_support_v5_2(
    directory: str | Path,
) -> dict[str, Any]:
    """Summarize target-free robot-geometry support for deterministic ranking."""

    manifest = validate_deform360_robot_metric_prefix(directory)
    root = Path(directory).resolve(strict=True)
    with np.load(root / METRIC_PREFIX_FILENAME, allow_pickle=False) as stored:
        valid = np.asarray(stored["valid_mask"], dtype=np.bool_)
    cluster_counts: list[int] = []
    for frame in valid:
        rows, columns = np.nonzero(frame)
        if not len(rows):
            cluster_counts.append(0)
            continue
        clusters = np.column_stack(
            (
                rows // METRIC_CLUSTER_SIZE_PIXELS,
                columns // METRIC_CLUSTER_SIZE_PIXELS,
            )
        )
        cluster_counts.append(int(len(np.unique(clusters, axis=0))))
    max_clusters = max(cluster_counts, default=0)
    qualifying = sum(count >= MINIMUM_INDEPENDENT_CLUSTERS for count in cluster_counts)
    return {
        "camera_id": manifest["camera_id"],
        "artifact_id": manifest["artifact_id"],
        "manifest_file_sha256": _sha256_file(root / METRIC_MANIFEST_FILENAME),
        "metric_prefix_file_sha256": _sha256_file(root / METRIC_PREFIX_FILENAME),
        "max_independent_cluster_count": max_clusters,
        "frames_with_minimum_clusters": qualifying,
        "total_projected_point_count": int(manifest["projected_point_count"]),
        "eligible": qualifying >= 1,
    }


def _validate_metric_support(value: object) -> dict[str, Any]:
    support = _mapping(value, name="metric camera support")
    require_exact_fields(
        support, expected=_METRIC_SUPPORT_FIELDS, name="metric camera support"
    )
    _identifier(support.get("camera_id"), name="camera_id")
    sha256_digest(support.get("artifact_id"), name="artifact_id")
    sha256_digest(support.get("manifest_file_sha256"), name="manifest_file_sha256")
    sha256_digest(
        support.get("metric_prefix_file_sha256"),
        name="metric_prefix_file_sha256",
    )
    maximum = genuine_integer(
        support.get("max_independent_cluster_count"),
        name="max_independent_cluster_count",
        minimum=0,
    )
    qualifying = genuine_integer(
        support.get("frames_with_minimum_clusters"),
        name="frames_with_minimum_clusters",
        minimum=0,
    )
    genuine_integer(
        support.get("total_projected_point_count"),
        name="total_projected_point_count",
        minimum=0,
    )
    eligible = support.get("eligible")
    _require(type(eligible) is bool, "eligible must be Boolean")
    _require(
        eligible is (maximum >= MINIMUM_INDEPENDENT_CLUSTERS and qualifying >= 1),
        "metric-camera eligibility changed",
    )
    return cast(dict[str, Any], plain_json(support))


def rank_deform360_metric_camera_support_v5_2(
    values: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Rank eligible cameras by the frozen robot-geometry-only rule."""

    supports = [_validate_metric_support(value) for value in values]
    cameras = [cast(str, item["camera_id"]) for item in supports]
    _require(len(cameras) == len(set(cameras)), "metric support repeats a camera")
    ranked = sorted(
        (item for item in supports if item["eligible"]),
        key=lambda item: (
            -cast(int, item["max_independent_cluster_count"]),
            -cast(int, item["frames_with_minimum_clusters"]),
            -cast(int, item["total_projected_point_count"]),
            cast(str, item["camera_id"]),
        ),
    )
    return tuple(cast(str, item["camera_id"]) for item in ranked)


def build_deform360_joint_sparse_camera_recovery_preflight_v5_2(
    *,
    lock: Mapping[str, Any],
    base_provider_plan: Mapping[str, Any],
    base_provider_plan_file_sha256: str,
    base_camera_audit: Mapping[str, Any],
    base_camera_audit_file_sha256: str,
    metric_root: str | Path,
) -> dict[str, Any]:
    """Select extra cameras from released prefix geometry without outcomes."""

    validate_deform360_joint_sparse_motioncrafter_source_plan_v5(base_provider_plan)
    audit = validate_deform360_joint_sparse_camera_audit_v5_2(
        base_camera_audit,
        lock=lock,
    )
    _require(
        audit["base_source_plan_id"] != base_provider_plan["manifest_sha256"],
        "source and provider plans must remain distinct artifacts",
    )
    root = _ordinary_root(metric_root)
    audit_map = {
        cast(str, row["object_id"]): row
        for row in cast(Sequence[Mapping[str, Any]], audit["objects"])
    }
    objects: list[dict[str, Any]] = []
    for base_object in cast(Sequence[Mapping[str, Any]], base_provider_plan["objects"]):
        object_id = cast(str, base_object["object_id"])
        audit_row = cast(Mapping[str, Any], audit_map[object_id])
        attempted = tuple(cast(Sequence[str], audit_row["attempted_camera_ids"]))
        passing = tuple(cast(Sequence[str], audit_row["passing_camera_ids"]))
        all_cameras = tuple(cast(Sequence[str], base_object["all_camera_ids"]))
        reserved = tuple(
            cast(Sequence[str], base_object["reserved_endpoint_camera_ids"])
        )
        recovery_required = len(passing) < MINIMUM_PASSING_CAMERAS
        candidates = tuple(
            camera
            for camera in all_cameras
            if camera not in reserved and camera not in attempted
        )
        supports: list[dict[str, Any]] = []
        if recovery_required:
            for camera in candidates:
                support = summarize_deform360_metric_camera_support_v5_2(
                    root / object_id / camera
                )
                _require(
                    support["camera_id"] == camera,
                    "metric support camera identity changed",
                )
                supports.append(support)
        ranked = list(rank_deform360_metric_camera_support_v5_2(supports))
        objects.append(
            {
                "object_id": object_id,
                "base_attempted_camera_ids": list(attempted),
                "base_passing_camera_ids": list(passing),
                "recovery_required": recovery_required,
                "candidate_metric_support": sorted(
                    supports, key=lambda row: cast(str, row["camera_id"])
                ),
                "ranked_eligible_camera_ids": ranked,
                "selected_recovery_camera_ids": ranked[:MAXIMUM_ADDITIONAL_CAMERAS],
            }
        )
    identity: dict[str, Any] = {
        "schema": CAMERA_RECOVERY_PREFLIGHT_SCHEMA,
        "schema_version": CAMERA_RECOVERY_PREFLIGHT_VERSION,
        "semantics": CAMERA_RECOVERY_PREFLIGHT_SEMANTICS,
        "execution_lock_id": lock["execution_lock_id"],
        "base_provider_plan_id": base_provider_plan["manifest_sha256"],
        "base_provider_plan_file_sha256": sha256_digest(
            base_provider_plan_file_sha256,
            name="base_provider_plan_file_sha256",
        ),
        "base_camera_audit_id": audit["audit_id"],
        "base_camera_audit_file_sha256": sha256_digest(
            base_camera_audit_file_sha256,
            name="base_camera_audit_file_sha256",
        ),
        "policy": dict(RECOVERY_POLICY),
        "objects": objects,
        "information_boundary": dict(AUDIT_INFORMATION_BOUNDARY),
    }
    result = {**identity, "preflight_id": content_id(identity)}
    validate_deform360_joint_sparse_camera_recovery_preflight_v5_2(
        result,
        lock=lock,
        base_provider_plan=base_provider_plan,
        base_camera_audit=audit,
    )
    return result


def validate_deform360_joint_sparse_camera_recovery_preflight_v5_2(
    value: object,
    *,
    lock: Mapping[str, Any],
    base_provider_plan: Mapping[str, Any],
    base_camera_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one deterministic metric-only recovery preflight."""

    validate_deform360_joint_sparse_motioncrafter_source_plan_v5(base_provider_plan)
    audit = validate_deform360_joint_sparse_camera_audit_v5_2(
        base_camera_audit,
        lock=lock,
    )
    preflight = _mapping(value, name="camera recovery preflight")
    require_exact_fields(
        preflight, expected=_PREFLIGHT_FIELDS, name="camera recovery preflight"
    )
    _content_addressed(preflight, identity="preflight_id")
    _require(
        preflight.get("schema") == CAMERA_RECOVERY_PREFLIGHT_SCHEMA
        and preflight.get("schema_version") == CAMERA_RECOVERY_PREFLIGHT_VERSION
        and preflight.get("semantics") == CAMERA_RECOVERY_PREFLIGHT_SEMANTICS,
        "camera-recovery preflight contract changed",
    )
    _require(
        preflight.get("execution_lock_id") == lock.get("execution_lock_id")
        and preflight.get("base_provider_plan_id")
        == base_provider_plan.get("manifest_sha256")
        and preflight.get("base_camera_audit_id") == audit.get("audit_id"),
        "camera-recovery lineage changed",
    )
    sha256_digest(
        preflight.get("base_provider_plan_file_sha256"),
        name="base_provider_plan_file_sha256",
    )
    sha256_digest(
        preflight.get("base_camera_audit_file_sha256"),
        name="base_camera_audit_file_sha256",
    )
    _require(preflight.get("policy") == RECOVERY_POLICY, "recovery policy changed")
    _require(
        preflight.get("information_boundary") == AUDIT_INFORMATION_BOUNDARY,
        "recovery-preflight boundary changed",
    )
    base_objects = {
        cast(str, row["object_id"]): row
        for row in cast(Sequence[Mapping[str, Any]], base_provider_plan["objects"])
    }
    audit_objects = {
        cast(str, row["object_id"]): row
        for row in cast(Sequence[Mapping[str, Any]], audit["objects"])
    }
    seen: set[str] = set()
    for raw_object in _sequence(preflight.get("objects"), name="preflight objects"):
        row = _mapping(raw_object, name="preflight object")
        require_exact_fields(
            row, expected=_PREFLIGHT_OBJECT_FIELDS, name="preflight object"
        )
        object_id = _identifier(row.get("object_id"), name="object_id")
        _require(
            object_id in base_objects
            and object_id not in seen
            and object_id in audit_objects,
            "preflight object roster changed",
        )
        seen.add(object_id)
        base = base_objects[object_id]
        audited = audit_objects[object_id]
        attempted = tuple(
            _identifier(item, name="base attempted camera")
            for item in _sequence(
                row.get("base_attempted_camera_ids"), name="base attempted cameras"
            )
        )
        passing = tuple(
            _identifier(item, name="base passing camera")
            for item in _sequence(
                row.get("base_passing_camera_ids"), name="base passing cameras"
            )
        )
        _require(
            attempted == tuple(audited["attempted_camera_ids"])
            and passing == tuple(audited["passing_camera_ids"]),
            "preflight changed the base camera audit",
        )
        recovery_required = len(passing) < MINIMUM_PASSING_CAMERAS
        _require(
            row.get("recovery_required") is recovery_required,
            "recovery trigger changed",
        )
        all_cameras = tuple(cast(Sequence[str], base["all_camera_ids"]))
        reserved = set(cast(Sequence[str], base["reserved_endpoint_camera_ids"]))
        expected_candidates = {
            camera
            for camera in all_cameras
            if camera not in reserved and camera not in attempted
        }
        supports = [
            _validate_metric_support(item)
            for item in _sequence(
                row.get("candidate_metric_support"), name="candidate metric support"
            )
        ]
        support_cameras = [cast(str, item["camera_id"]) for item in supports]
        _require(
            support_cameras == sorted(set(support_cameras)),
            "metric support cameras are not unique and sorted",
        )
        _require(
            set(support_cameras)
            == (expected_candidates if recovery_required else set()),
            "metric support candidate set changed",
        )
        ranked = list(rank_deform360_metric_camera_support_v5_2(supports))
        _require(
            row.get("ranked_eligible_camera_ids") == ranked
            and row.get("selected_recovery_camera_ids")
            == ranked[:MAXIMUM_ADDITIONAL_CAMERAS],
            "camera-recovery ranking changed",
        )
    _require(set(seen) == set(base_objects), "preflight cohort changed")
    return cast(dict[str, Any], plain_json(preflight))


def validate_deform360_joint_sparse_camera_recovery_amendment_v5_2(
    value: object,
) -> dict[str, Any]:
    """Validate the additive amendment without consulting empirical outcomes."""

    amendment = _mapping(value, name="camera recovery amendment")
    expected = frozenset(
        {
            "amendment_id",
            "base_execution_lock",
            "claim_boundary",
            "information_boundary",
            "policy",
            "schema",
            "schema_version",
            "semantics",
            "status",
            "trigger_artifacts",
        }
    )
    require_exact_fields(amendment, expected=expected, name="camera recovery amendment")
    _content_addressed(amendment, identity="amendment_id")
    _require(
        amendment.get("schema") == CAMERA_RECOVERY_AMENDMENT_SCHEMA
        and amendment.get("schema_version") == CAMERA_RECOVERY_AMENDMENT_VERSION
        and amendment.get("semantics") == CAMERA_RECOVERY_AMENDMENT_SEMANTICS
        and amendment.get("status") == "locked-before-recovery-metric-preflight",
        "camera-recovery amendment contract changed",
    )
    base = _mapping(amendment.get("base_execution_lock"), name="base execution lock")
    require_exact_fields(
        base,
        expected=frozenset({"execution_lock_id", "file_sha256"}),
        name="base execution lock",
    )
    sha256_digest(base.get("execution_lock_id"), name="execution_lock_id")
    sha256_digest(base.get("file_sha256"), name="execution lock file sha256")
    triggers = source_artifact_mapping(
        _mapping(amendment.get("trigger_artifacts"), name="trigger artifacts"),
        name="trigger artifacts",
    )
    _require(
        set(triggers)
        == {
            "base-provider-plan.json",
            "base-provider-run-report.json",
            "base-source-prediction-batch.json",
            "base-source-prediction-receipt.json",
            "base-source-prediction-plan.json",
        },
        "amendment trigger artifacts changed",
    )
    _require(amendment.get("policy") == RECOVERY_POLICY, "amendment policy changed")
    _require(
        amendment.get("information_boundary") == AUDIT_INFORMATION_BOUNDARY,
        "amendment information boundary changed",
    )
    _require(
        amendment.get("claim_boundary") == RECOVERY_CLAIM_BOUNDARY,
        "amendment claim boundary changed",
    )
    return cast(dict[str, Any], plain_json(amendment))


def _recovery_job(
    *,
    object_id: str,
    episode_id: int,
    stratum: str,
    camera: str,
    source_video: Mapping[str, Any],
    source_start: int,
    source_stop: int,
) -> dict[str, Any]:
    windows = _windows(source_start, source_stop)
    descriptor: dict[str, Any] = {
        "object_id": object_id,
        "episode_id": episode_id,
        "source_episode": "episode_0000",
        "stratum": stratum,
        "camera": camera,
        "likelihood_eligible": True,
        "source_video": dict(source_video),
        "source_frame_start": source_start,
        "source_frame_stop_exclusive": source_stop,
        "source_frame_count": PROVIDER_FRAME_COUNT,
        "windows": windows,
        "seed_schedule": _seed_schedule(cast(int, RUN_CONFIGURATION["seed"]), windows),
        "output_relative_path": (
            f"objects/{object_id}/episode_{episode_id:04d}/views/{camera}"
        ),
    }
    return {"job_id": content_id(descriptor), **descriptor}


def build_deform360_joint_sparse_motioncrafter_recovery_plan_v5_2(
    *,
    lock: Mapping[str, Any],
    execution_lock_file_sha256: str,
    inventory: Mapping[str, Any],
    base_provider_plan: Mapping[str, Any],
    base_provider_plan_file_sha256: str,
    base_camera_audit: Mapping[str, Any],
    recovery_preflight: Mapping[str, Any],
    recovery_preflight_file_sha256: str,
    amendment: Mapping[str, Any],
    amendment_file_sha256: str,
    implementation_revision: str,
    runner_source_sha256: str,
) -> dict[str, Any]:
    """Build the immutable extra-camera MotionCrafter execution plan."""

    validate_deform360_joint_sparse_motioncrafter_source_plan_v5(base_provider_plan)
    normalized_amendment = (
        validate_deform360_joint_sparse_camera_recovery_amendment_v5_2(amendment)
    )
    _require(
        normalized_amendment["base_execution_lock"]["execution_lock_id"]
        == lock.get("execution_lock_id"),
        "amendment uses a different execution lock",
    )
    preflight = validate_deform360_joint_sparse_camera_recovery_preflight_v5_2(
        recovery_preflight,
        lock=lock,
        base_provider_plan=base_provider_plan,
        base_camera_audit=base_camera_audit,
    )
    normalized_inventory = validate_deform360_prepared_source_inventory(inventory)
    _require(
        normalized_inventory["inventory_id"]
        == base_provider_plan["prepared_source_inventory"]["inventory_id"],
        "prepared inventory changed",
    )
    inventory_rows = {
        cast(str, row["object_id"]): row
        for row in cast(Sequence[Mapping[str, Any]], normalized_inventory["objects"])
    }
    base_objects = {
        cast(str, row["object_id"]): row
        for row in cast(Sequence[Mapping[str, Any]], base_provider_plan["objects"])
    }
    objects: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for preflight_object in cast(Sequence[Mapping[str, Any]], preflight["objects"]):
        selected = tuple(
            cast(Sequence[str], preflight_object["selected_recovery_camera_ids"])
        )
        if not selected:
            continue
        object_id = cast(str, preflight_object["object_id"])
        base = base_objects[object_id]
        inventory_row = inventory_rows[object_id]
        episode_id = genuine_integer(base["episode_id"], name="episode_id", minimum=0)
        stratum = _identifier(base["stratum"], name="stratum")
        provider_range = cast(Sequence[int], base["provider_range_half_open"])
        source_start, source_stop = int(provider_range[0]), int(provider_range[1])
        objects.append(
            {
                "object_id": object_id,
                "episode_id": episode_id,
                "stratum": stratum,
                "raw_prefix_range_half_open": list(base["raw_prefix_range_half_open"]),
                "provider_range_half_open": list(provider_range),
                "all_camera_ids": list(base["all_camera_ids"]),
                "reserved_endpoint_camera_ids": list(
                    base["reserved_endpoint_camera_ids"]
                ),
                "base_attempted_camera_ids": list(
                    preflight_object["base_attempted_camera_ids"]
                ),
                "base_passing_camera_ids": list(
                    preflight_object["base_passing_camera_ids"]
                ),
                "selected_recovery_camera_ids": list(selected),
            }
        )
        for camera in selected:
            jobs.append(
                _recovery_job(
                    object_id=object_id,
                    episode_id=episode_id,
                    stratum=stratum,
                    camera=camera,
                    source_video=_camera_video(inventory_row, camera),
                    source_start=source_start,
                    source_stop=source_stop,
                )
            )
    objects.sort(key=lambda row: cast(str, row["object_id"]))
    jobs.sort(key=lambda row: (cast(str, row["object_id"]), cast(str, row["camera"])))
    _require(bool(jobs), "camera recovery selected no provider jobs")
    descriptor: dict[str, Any] = {
        "schema": RECOVERY_PROVIDER_SCHEMA,
        "schema_version": RECOVERY_PROVIDER_VERSION,
        "semantics": RECOVERY_PROVIDER_SEMANTICS,
        "status": RECOVERY_PROVIDER_STATUS,
        "role": "development-source-prefix-camera-recovery",
        "implementation": {
            "revision": exact_revision(
                implementation_revision, name="implementation revision"
            ),
            "runner_source_sha256": sha256_digest(
                runner_source_sha256, name="runner source sha256"
            ),
        },
        "source_execution_lock": {
            "execution_lock_id": lock["execution_lock_id"],
            "file_sha256": sha256_digest(
                execution_lock_file_sha256, name="execution lock file sha256"
            ),
        },
        "prepared_source_inventory": dict(
            base_provider_plan["prepared_source_inventory"]
        ),
        "camera_roster_source": dict(base_provider_plan["camera_roster_source"]),
        "base_provider_plan": {
            "manifest_sha256": base_provider_plan["manifest_sha256"],
            "file_sha256": sha256_digest(
                base_provider_plan_file_sha256,
                name="base provider plan file sha256",
            ),
        },
        "camera_recovery_preflight": {
            "preflight_id": preflight["preflight_id"],
            "file_sha256": sha256_digest(
                recovery_preflight_file_sha256,
                name="recovery preflight file sha256",
            ),
        },
        "camera_recovery_amendment": {
            "amendment_id": normalized_amendment["amendment_id"],
            "file_sha256": sha256_digest(
                amendment_file_sha256, name="amendment file sha256"
            ),
        },
        "provider_lock": dict(base_provider_plan["provider_lock"]),
        "motioncrafter": dict(base_provider_plan["motioncrafter"]),
        "run_configuration": dict(RUN_CONFIGURATION),
        "temporal_policy": dict(TEMPORAL_POLICY),
        "objects": objects,
        "object_count": len(objects),
        "jobs": jobs,
        "job_count": len(jobs),
        "smoke_job_id": jobs[0]["job_id"],
        "information_boundary": dict(RECOVERY_INFORMATION_BOUNDARY),
        "claim_boundary": RECOVERY_CLAIM_BOUNDARY,
    }
    plan = {"manifest_sha256": content_id(descriptor), **descriptor}
    validate_deform360_joint_sparse_motioncrafter_recovery_plan_v5_2(plan)
    return plan


def validate_deform360_joint_sparse_motioncrafter_recovery_plan_v5_2(
    value: object,
) -> dict[str, Any]:
    """Validate one extra-camera provider plan."""

    plan = _mapping(value, name="camera recovery provider plan")
    require_exact_fields(
        plan, expected=_RECOVERY_PROVIDER_FIELDS, name="camera recovery provider plan"
    )
    _content_addressed(plan, identity="manifest_sha256")
    _require(
        plan.get("schema") == RECOVERY_PROVIDER_SCHEMA
        and plan.get("schema_version") == RECOVERY_PROVIDER_VERSION
        and plan.get("semantics") == RECOVERY_PROVIDER_SEMANTICS
        and plan.get("status") == RECOVERY_PROVIDER_STATUS
        and plan.get("role") == "development-source-prefix-camera-recovery",
        "recovery provider contract changed",
    )
    _require(
        plan.get("run_configuration") == RUN_CONFIGURATION
        and plan.get("temporal_policy") == TEMPORAL_POLICY
        and plan.get("information_boundary") == RECOVERY_INFORMATION_BOUNDARY
        and plan.get("claim_boundary") == RECOVERY_CLAIM_BOUNDARY,
        "recovery provider policy changed",
    )
    implementation = _mapping(plan.get("implementation"), name="implementation")
    exact_revision(implementation.get("revision"), name="implementation revision")
    sha256_digest(
        implementation.get("runner_source_sha256"), name="runner source sha256"
    )
    for field, id_field in (
        ("source_execution_lock", "execution_lock_id"),
        ("prepared_source_inventory", "inventory_id"),
        ("camera_roster_source", "manifest_sha256"),
        ("base_provider_plan", "manifest_sha256"),
        ("camera_recovery_preflight", "preflight_id"),
        ("camera_recovery_amendment", "amendment_id"),
    ):
        binding = _mapping(plan.get(field), name=field)
        sha256_digest(binding.get(id_field), name=f"{field}.{id_field}")
        sha256_digest(binding.get("file_sha256"), name=f"{field}.file_sha256")
    provider = _mapping(plan.get("provider_lock"), name="provider lock")
    motion = _mapping(plan.get("motioncrafter"), name="MotionCrafter")
    _require(provider.get("provider_revision") == PROB4D_REVISION, "Prob4D changed")
    _require(
        motion.get("revision") == MOTIONCRAFTER_REVISION
        and motion.get("model_set_id") == MOTIONCRAFTER_MODEL_SET_ID,
        "MotionCrafter changed",
    )
    model_set = _mapping(motion.get("model_set_manifest"), name="model set")
    _require(content_id(model_set) == MOTIONCRAFTER_MODEL_SET_ID, "model set changed")
    objects = _sequence(plan.get("objects"), name="objects")
    jobs = _sequence(plan.get("jobs"), name="jobs")
    _require(
        1 <= len(objects) <= 10
        and plan.get("object_count") == len(objects)
        and 1 <= len(jobs) <= 10 * MAXIMUM_ADDITIONAL_CAMERAS
        and plan.get("job_count") == len(jobs),
        "recovery provider counts changed",
    )
    object_map: dict[str, Mapping[str, Any]] = {}
    for raw_object in objects:
        row = _mapping(raw_object, name="recovery provider object")
        expected_fields = frozenset(
            {
                "all_camera_ids",
                "base_attempted_camera_ids",
                "base_passing_camera_ids",
                "episode_id",
                "object_id",
                "provider_range_half_open",
                "raw_prefix_range_half_open",
                "reserved_endpoint_camera_ids",
                "selected_recovery_camera_ids",
                "stratum",
            }
        )
        require_exact_fields(row, expected=expected_fields, name="recovery object")
        object_id = _identifier(row.get("object_id"), name="object_id")
        _require(object_id not in object_map, "recovery object repeats")
        all_cameras = tuple(
            _identifier(item, name="camera_id")
            for item in _sequence(row.get("all_camera_ids"), name="all cameras")
        )
        reserved = select_reserved_endpoint_views_v5(object_id, all_cameras, count=2)
        selected = tuple(
            _identifier(item, name="selected camera")
            for item in _sequence(
                row.get("selected_recovery_camera_ids"), name="selected cameras"
            )
        )
        attempted = set(
            _identifier(item, name="attempted camera")
            for item in _sequence(
                row.get("base_attempted_camera_ids"), name="attempted cameras"
            )
        )
        passing = tuple(
            _identifier(item, name="passing camera")
            for item in _sequence(
                row.get("base_passing_camera_ids"), name="passing cameras"
            )
        )
        _require(
            tuple(row.get("reserved_endpoint_camera_ids", ())) == reserved
            and len(passing) < MINIMUM_PASSING_CAMERAS
            and 1 <= len(selected) <= MAXIMUM_ADDITIONAL_CAMERAS
            and len(set(selected)) == len(selected)
            and set(selected).issubset(set(all_cameras) - set(reserved) - attempted),
            "recovery camera policy changed",
        )
        provider_range = tuple(
            genuine_integer(item, name="provider frame", minimum=0)
            for item in _sequence(
                row.get("provider_range_half_open"), name="provider range"
            )
        )
        _require(
            len(provider_range) == 2
            and provider_range[1] - provider_range[0] == PROVIDER_FRAME_COUNT,
            "recovery provider range changed",
        )
        raw_prefix = tuple(
            genuine_integer(item, name="raw prefix frame", minimum=0)
            for item in _sequence(
                row.get("raw_prefix_range_half_open"), name="raw prefix range"
            )
        )
        _require(
            len(raw_prefix) == 2
            and raw_prefix[1] - raw_prefix[0] == 58
            and provider_range == (raw_prefix[1] - PROVIDER_FRAME_COUNT, raw_prefix[1]),
            "recovery raw-prefix range changed",
        )
        object_map[object_id] = row
    _require(list(object_map) == sorted(object_map), "recovery objects are not sorted")
    seen_pairs: set[tuple[str, str]] = set()
    job_ids: list[str] = []
    for raw_job in jobs:
        job = _mapping(raw_job, name="recovery provider job")
        require_exact_fields(
            job, expected=_RECOVERY_JOB_FIELDS, name="recovery provider job"
        )
        job_id = sha256_digest(job.get("job_id"), name="job_id")
        descriptor = dict(job)
        descriptor.pop("job_id")
        _require(content_id(descriptor) == job_id, "recovery job ID changed")
        object_id = _identifier(job.get("object_id"), name="object_id")
        camera = _identifier(job.get("camera"), name="camera")
        _require(object_id in object_map, "recovery job object changed")
        row = object_map[object_id]
        _require(
            camera in row["selected_recovery_camera_ids"]
            and job.get("likelihood_eligible") is True
            and job.get("episode_id") == row["episode_id"]
            and job.get("stratum") == row["stratum"]
            and job.get("source_episode") == "episode_0000",
            "recovery job identity changed",
        )
        job_provider_range = cast(Sequence[int], row["provider_range_half_open"])
        start = genuine_integer(job.get("source_frame_start"), name="source start")
        stop = genuine_integer(
            job.get("source_frame_stop_exclusive"), name="source stop"
        )
        windows = list(_sequence(job.get("windows"), name="windows"))
        _require(
            [start, stop] == list(job_provider_range)
            and job.get("source_frame_count") == PROVIDER_FRAME_COUNT
            and windows == _windows(start, stop)
            and job.get("seed_schedule")
            == _seed_schedule(cast(int, RUN_CONFIGURATION["seed"]), windows),
            "recovery job temporal policy changed",
        )
        source = _mapping(job.get("source_video"), name="source video")
        require_exact_fields(
            source,
            expected=frozenset({"bytes", "path", "sha256"}),
            name="source video",
        )
        path = _safe_relative_path(source.get("path"), name="source video path")
        _require(
            path.endswith(f"/{camera}/undistorted.mp4")
            and type(source.get("bytes")) is int
            and cast(int, source["bytes"]) > 0,
            "recovery source video changed",
        )
        sha256_digest(source.get("sha256"), name="source video sha256")
        _require(
            _safe_relative_path(job.get("output_relative_path"), name="output path")
            == f"objects/{object_id}/episode_{int(row['episode_id']):04d}/views/{camera}",
            "recovery output path changed",
        )
        pair = (object_id, camera)
        _require(pair not in seen_pairs, "recovery job pair repeats")
        seen_pairs.add(pair)
        job_ids.append(job_id)
    expected_pairs = {
        (object_id, camera)
        for object_id, row in object_map.items()
        for camera in cast(Sequence[str], row["selected_recovery_camera_ids"])
    }
    _require(seen_pairs == expected_pairs, "recovery job roster changed")
    _require(plan.get("smoke_job_id") == job_ids[0], "recovery smoke job changed")
    return cast(dict[str, Any], plain_json(plan))


def _provider_jobs(
    plan: Mapping[str, Any], *, shard_index: int, shard_count: int
) -> tuple[Mapping[str, Any], ...]:
    jobs = cast(Sequence[Mapping[str, Any]], plan["jobs"])
    return tuple(
        job for index, job in enumerate(jobs) if index % shard_count == shard_index
    )


def _expected_provider_member_count(
    plan: Mapping[str, Any], job: Mapping[str, Any]
) -> int:
    configuration = _mapping(
        plan["run_configuration"], name="provider run configuration"
    )
    products = _sequence(configuration["products"], name="provider products")
    overlap_product = configuration["provider_consumed_product"]
    _require(
        products.count(overlap_product) == 1,
        "provider overlap product roster changed",
    )
    return len(products) - 1 + len(job["windows"])


def validate_deform360_joint_sparse_motioncrafter_recovery_run_v5_2(
    value: object,
    *,
    plan: Mapping[str, Any],
    expected_shard_index: int | None = None,
    expected_shard_count: int | None = None,
) -> dict[str, Any]:
    """Validate one complete recovery-provider run without reading a suffix."""

    normalized_plan = validate_deform360_joint_sparse_motioncrafter_recovery_plan_v5_2(
        plan
    )
    report = _mapping(value, name="recovery provider run")
    require_exact_fields(report, expected=_RUN_REPORT_FIELDS, name="provider run")
    _content_addressed(report, identity="run_sha256")
    _require(
        report.get("schema") == PROVIDER_RUN_SCHEMA
        and report.get("schema_version") == PROVIDER_RUN_VERSION
        and report.get("status") == "complete",
        "recovery provider run is incomplete or changed",
    )
    _require(
        report.get("source_plan_sha256") == normalized_plan["manifest_sha256"]
        and report.get("runtime_revision")
        == normalized_plan["implementation"]["revision"],
        "recovery provider run lineage changed",
    )
    _require(
        report.get("information_boundary") == PROVIDER_RUN_INFORMATION_BOUNDARY
        and report.get("claim_boundary") == PROVIDER_RUN_CLAIM_BOUNDARY,
        "recovery provider run boundary changed",
    )
    shard_index = genuine_integer(report.get("shard_index"), name="shard index")
    shard_count = genuine_integer(
        report.get("shard_count"), name="shard count", minimum=1
    )
    _require(shard_index < shard_count, "invalid recovery provider shard")
    if expected_shard_index is not None:
        _require(shard_index == expected_shard_index, "provider shard index changed")
    if expected_shard_count is not None:
        _require(shard_count == expected_shard_count, "provider shard count changed")
    expected_mode = "complete" if shard_count == 1 else "shard"
    _require(report.get("mode") == expected_mode, "provider run mode changed")
    expected_jobs = _provider_jobs(
        normalized_plan, shard_index=shard_index, shard_count=shard_count
    )
    completed = _sequence(report.get("completed_jobs"), name="completed jobs")
    _require(
        report.get("requested_job_count") == len(expected_jobs)
        and report.get("completed_job_count") == len(expected_jobs)
        and len(completed) == len(expected_jobs),
        "recovery provider run count changed",
    )
    normalized_completed: list[dict[str, Any]] = []
    for raw, job in zip(completed, expected_jobs, strict=True):
        row = _mapping(raw, name="completed provider job")
        require_exact_fields(
            row, expected=_COMPLETED_JOB_FIELDS, name="completed provider job"
        )
        _require(row.get("job_id") == job["job_id"], "completed job order changed")
        manifest = _identifier(
            row.get("prediction_manifest"), name="prediction manifest"
        )
        manifest_sha256 = sha256_digest(
            row.get("prediction_manifest_sha256"),
            name="prediction manifest sha256",
        )
        verification = _mapping(row.get("verification"), name="verification")
        require_exact_fields(
            verification, expected=_VERIFICATION_FIELDS, name="verification"
        )
        _require(
            verification.get("hashes_verified") is True
            and verification.get("integrity_bound") is True
            and verification.get("manifest_path") == manifest
            and genuine_integer(
                verification.get("member_count"), name="member count", minimum=1
            )
            == _expected_provider_member_count(normalized_plan, job),
            "provider verification changed",
        )
        sha256_digest(verification.get("run_spec_sha256"), name="run spec sha256")
        normalized_completed.append(
            {
                "job_id": job["job_id"],
                "prediction_manifest": manifest,
                "prediction_manifest_sha256": manifest_sha256,
                "verification": dict(verification),
            }
        )
    normalized = cast(dict[str, Any], plain_json(report))
    _require(
        normalized["completed_jobs"] == normalized_completed,
        "completed provider job normalization changed",
    )
    return normalized


def merge_deform360_joint_sparse_motioncrafter_recovery_runs_v5_2(
    *,
    plan: Mapping[str, Any],
    shard_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge an exact complete shard roster into one content-addressed report."""

    normalized_plan = validate_deform360_joint_sparse_motioncrafter_recovery_plan_v5_2(
        plan
    )
    _require(bool(shard_reports), "provider shard roster is empty")
    first = _mapping(shard_reports[0], name="provider shard")
    shard_count = genuine_integer(
        first.get("shard_count"), name="shard count", minimum=1
    )
    _require(
        shard_count == len(shard_reports),
        "provider shard roster is incomplete",
    )
    reports = [
        validate_deform360_joint_sparse_motioncrafter_recovery_run_v5_2(
            value,
            plan=normalized_plan,
            expected_shard_index=index,
            expected_shard_count=shard_count,
        )
        for index, value in enumerate(shard_reports)
    ]
    runtime_revisions = {cast(str, report["runtime_revision"]) for report in reports}
    _require(len(runtime_revisions) == 1, "provider shard runtime changed")
    completed_by_id = {
        cast(str, row["job_id"]): row
        for report in reports
        for row in cast(Sequence[Mapping[str, Any]], report["completed_jobs"])
    }
    jobs = cast(Sequence[Mapping[str, Any]], normalized_plan["jobs"])
    _require(
        set(completed_by_id) == {cast(str, job["job_id"]) for job in jobs},
        "provider shard jobs do not cover the frozen plan",
    )
    completed = [completed_by_id[cast(str, job["job_id"])] for job in jobs]
    descriptor: dict[str, Any] = {
        "schema": PROVIDER_RUN_SCHEMA,
        "schema_version": PROVIDER_RUN_VERSION,
        "source_plan_sha256": normalized_plan["manifest_sha256"],
        "runtime_revision": runtime_revisions.pop(),
        "mode": "complete",
        "shard_index": 0,
        "shard_count": 1,
        "status": "complete",
        "requested_job_count": len(jobs),
        "completed_job_count": len(completed),
        "completed_jobs": completed,
        "information_boundary": dict(PROVIDER_RUN_INFORMATION_BOUNDARY),
        "claim_boundary": PROVIDER_RUN_CLAIM_BOUNDARY,
    }
    merged = {"run_sha256": content_id(descriptor), **descriptor}
    return validate_deform360_joint_sparse_motioncrafter_recovery_run_v5_2(
        merged, plan=normalized_plan
    )


def _relative_file_record(path: Path, *, input_root: Path, name: str) -> dict[str, str]:
    _require(
        path.is_file()
        and not path.is_symlink()
        and not any(parent.is_symlink() for parent in path.parents),
        f"{name} is invalid",
    )
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(input_root)
    except ValueError as error:
        raise ValueError(f"{name} is outside the input root") from error
    return {"path": relative.as_posix(), "sha256": _sha256_file(resolved)}


def build_deform360_joint_sparse_combined_camera_audit_plan_v5_2(
    *,
    lock: Mapping[str, Any],
    base_source_plan: Mapping[str, Any],
    base_camera_audit: Mapping[str, Any],
    recovery_provider_plan: Mapping[str, Any],
    recovery_provider_run: Mapping[str, Any],
    input_root: str | Path,
    recovery_decoded_root: str | Path,
    recovery_metric_root: str | Path,
    implementation_revision: str,
) -> dict[str, Any]:
    """Build the all-attempted-camera plan without operator camera choices."""

    base = validate_deform360_joint_sparse_source_prediction_plan_v5(
        base_source_plan, lock=lock
    )
    base_audit = validate_deform360_joint_sparse_camera_audit_v5_2(
        base_camera_audit, lock=lock
    )
    _require(
        base_audit["base_source_plan_id"] == base["plan_id"],
        "base camera audit uses a different source plan",
    )
    provider = validate_deform360_joint_sparse_motioncrafter_recovery_plan_v5_2(
        recovery_provider_plan
    )
    run = validate_deform360_joint_sparse_motioncrafter_recovery_run_v5_2(
        recovery_provider_run, plan=provider
    )
    _require(
        provider["source_execution_lock"]["execution_lock_id"]
        == lock.get("execution_lock_id"),
        "recovery provider uses a different execution lock",
    )
    root = _ordinary_root(input_root)
    decoded_root = _ordinary_root(recovery_decoded_root)
    metric_root = _ordinary_root(recovery_metric_root)
    for child, name in ((decoded_root, "decoded root"), (metric_root, "metric root")):
        try:
            child.relative_to(root)
        except ValueError as error:
            raise ValueError(f"recovery {name} is outside the input root") from error
    completed = {
        cast(str, row["job_id"]): row
        for row in cast(Sequence[Mapping[str, Any]], run["completed_jobs"])
    }
    jobs = {
        (cast(str, row["object_id"]), cast(str, row["camera"])): row
        for row in cast(Sequence[Mapping[str, Any]], provider["jobs"])
    }
    selected = {
        cast(str, row["object_id"]): tuple(
            cast(Sequence[str], row["selected_recovery_camera_ids"])
        )
        for row in cast(Sequence[Mapping[str, Any]], provider["objects"])
    }
    attempted_objects: list[dict[str, Any]] = []
    for base_object in cast(Sequence[Mapping[str, Any]], base["objects"]):
        object_id = cast(str, base_object["object_id"])
        windows = [
            cast(dict[str, Any], plain_json(window))
            for window in cast(
                Sequence[Mapping[str, Any]], base_object["visual_windows"]
            )
        ]
        for camera in selected.get(object_id, ()):
            job = jobs[(object_id, camera)]
            completed_job = completed[cast(str, job["job_id"])]
            decoded = decoded_root / object_id / f"{camera}.npz"
            decoded_report_path = decoded.with_suffix(".json")
            decoded_report = load_strict_json_object(
                decoded_report_path, label="decoded-uniform recovery report"
            )
            require_exact_fields(
                decoded_report,
                expected=_DECODED_REPORT_FIELDS,
                name="decoded-uniform recovery report",
            )
            _require(
                decoded_report.get("schema_version") == 1
                and decoded_report.get("method") == "decoded uniform overlap fusion"
                and decoded_report.get("fixed_prob4d_vggt_blend") is False
                and decoded_report.get("frame_count") == PROVIDER_FRAME_COUNT
                and decoded_report.get("covariance_units")
                == "m^2 after Prob4D gauge alignment"
                and decoded_report.get("prob4d_revision") == PROB4D_REVISION,
                "decoded-uniform recovery export changed",
            )
            _require(
                decoded_report.get("manifest") == completed_job["prediction_manifest"]
                and decoded_report.get("manifest_sha256")
                == completed_job["prediction_manifest_sha256"]
                and Path(cast(str, decoded_report.get("output_npz"))).resolve()
                == decoded.resolve(strict=True)
                and decoded_report.get("output_npz_sha256") == _sha256_file(decoded),
                "decoded-uniform recovery lineage changed",
            )
            metric_directory = metric_root / object_id / camera
            metric_manifest = validate_deform360_robot_metric_prefix(metric_directory)
            _require(
                metric_manifest["object_id"] == object_id
                and metric_manifest["camera_id"] == camera,
                "recovery metric-prefix identity changed",
            )
            windows.append(
                {
                    "camera_id": camera,
                    "decoded_uniform": _relative_file_record(
                        decoded, input_root=root, name="decoded uniform"
                    ),
                    "metric_prefix": _relative_file_record(
                        metric_directory / METRIC_PREFIX_FILENAME,
                        input_root=root,
                        name="metric prefix",
                    ),
                }
            )
        windows.sort(key=lambda item: cast(str, item["camera_id"]))
        _require(
            len(windows) == len({cast(str, window["camera_id"]) for window in windows}),
            "combined audit plan repeats a camera",
        )
        attempted_objects.append(
            {
                **{
                    key: plain_json(value)
                    for key, value in base_object.items()
                    if key != "visual_windows"
                },
                "visual_windows": windows,
            }
        )
    return build_deform360_joint_sparse_source_prediction_plan_v5(
        lock=lock,
        implementation_revision=implementation_revision,
        objects=attempted_objects,
    )


def load_deform360_joint_sparse_motioncrafter_execution_plan_v5_2(
    path: str | Path,
) -> Mapping[str, Any]:
    """Load either the immutable base plan or its v5.2 recovery extension."""

    value = load_strict_json_object(path, label="MotionCrafter execution plan")
    if value.get("schema") == RECOVERY_PROVIDER_SCHEMA:
        return validate_deform360_joint_sparse_motioncrafter_recovery_plan_v5_2(value)
    validate_deform360_joint_sparse_motioncrafter_source_plan_v5(value)
    return value


def save_deform360_joint_sparse_camera_recovery_artifact_v5_2(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically save an already validated recovery artifact."""

    write_atomic_json(value, path, overwrite=overwrite)


__all__ = [
    "AUDIT_INFORMATION_BOUNDARY",
    "CAMERA_AUDIT_SCHEMA",
    "CAMERA_RECOVERY_AMENDMENT_SCHEMA",
    "CAMERA_RECOVERY_PREFLIGHT_SCHEMA",
    "MAXIMUM_ADDITIONAL_CAMERAS",
    "MINIMUM_PASSING_CAMERAS",
    "RECOVERY_POLICY",
    "RECOVERY_PROVIDER_SCHEMA",
    "audit_deform360_joint_sparse_source_cameras_v5_2",
    "build_deform360_joint_sparse_camera_recovery_preflight_v5_2",
    "build_deform360_joint_sparse_combined_camera_audit_plan_v5_2",
    "build_deform360_joint_sparse_motioncrafter_recovery_plan_v5_2",
    "load_deform360_joint_sparse_motioncrafter_execution_plan_v5_2",
    "merge_deform360_joint_sparse_motioncrafter_recovery_runs_v5_2",
    "rank_deform360_metric_camera_support_v5_2",
    "save_deform360_joint_sparse_camera_recovery_artifact_v5_2",
    "summarize_deform360_metric_camera_support_v5_2",
    "validate_deform360_joint_sparse_camera_audit_v5_2",
    "validate_deform360_joint_sparse_camera_recovery_amendment_v5_2",
    "validate_deform360_joint_sparse_camera_recovery_preflight_v5_2",
    "validate_deform360_joint_sparse_motioncrafter_recovery_plan_v5_2",
    "validate_deform360_joint_sparse_motioncrafter_recovery_run_v5_2",
]
