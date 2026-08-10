"""Equal-object failure evidence from the frozen Deform360 source-support result."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Final, cast

from ._canonical_contracts import plain_json
from .deform360_provider_failure_census_v1 import (
    validate_deform360_provider_failure_census_payload,
)
from .provider_failure_decomposition import (
    PROVIDER_FAILURE_EVIDENCE_SCHEMA,
    PROVIDER_FAILURE_EVIDENCE_VERSION,
)

DEFORM360_SOURCE_SUPPORT_FAILURE_EVIDENCE_VERSION: Final = 1
DEFORM360_SOURCE_SUPPORT_FAILURE_EVIDENCE_SCHEMA: Final = (
    PROVIDER_FAILURE_EVIDENCE_SCHEMA
)
DEFORM360_SOURCE_SUPPORT_METRIC_BATCH_SCHEMA: Final = (
    "bayesian-phystwin.deform360-prob4d-metric-batch"
)
DEFORM360_SOURCE_SUPPORT_METRIC_BATCH_VERSION: Final = 1
DEFORM360_SOURCE_SUPPORT_METRIC_BATCH_SEMANTICS: Final = (
    "all-sealed-calibration-streams-released-robot-gauge-v1"
)
DEFORM360_SOURCE_SUPPORT_NEGATIVE_REASON: Final = (
    "released-robot-geometry-outside-fixed-camera-prefix"
)
DEFORM360_SOURCE_SUPPORT_GLOBAL_REJECTION_REASON: Final = (
    "global-complete-stream-support-gate-failed-before-object-level-admission"
)
DEFORM360_SOURCE_SUPPORT_FAILURE_CLAIM_BOUNDARY: Final = (
    "Source-only equal-physical-object attribution of the frozen Deform360 "
    "313/324 complete-stream support-negative result. The transformation does "
    "not rerun a provider, delete or replace streams, infer success for an object "
    "whose downstream update was never admitted, open confirmation data, use "
    "target outcomes or future frames, or authorize provider promotion."
)
DEFORM360_SOURCE_SUPPORT_AGGREGATION_SEMANTICS: Final = (
    "All ten frozen physical objects remain rejected because the method terminated "
    "before object-level downstream admission. Unsupported provider geometry is "
    "attributed only to objects with an observed retained support-negative stream; "
    "complete-support objects remain unresolved rather than being relabelled "
    "accepted."
)

_EXPECTED_METRIC_BATCH_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "result_id",
        "implementation_revision",
        "production_result_id",
        "admission_id",
        "object_count",
        "admitted_stream_count",
        "supported_stream_count",
        "support_negative_stream_count",
        "technical_failure_stream_count",
        "supported_object_count",
        "plan_emitted",
        "plan_file",
        "status",
        "jobs",
        "source_artifacts",
        "information_boundary",
        "claim_boundary",
    }
)
_EXPECTED_JOB_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "job_id",
        "object_id",
        "episode_id",
        "stratum",
        "camera_id",
        "output_relative_directory",
        "status",
        "metric_artifact_id",
        "projected_point_count",
        "failure_reason",
        "failure_detail_sha256",
    }
)
_EXPECTED_BOUNDARY: Final[dict[str, bool]] = {
    "calibration_robot_state_access_attempted": True,
    "calibration_robot_state_opened": True,
    "calibration_camera_calibration_opened": True,
    "calibration_camera_images_opened": False,
    "calibration_tactile_payloads_opened": False,
    "rendered_depth_opened": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "future_frames_used": False,
    "replacement_allowed": False,
    "human_approval_required": False,
}
_EXPECTED_OBJECTS: Final[tuple[tuple[str, int, str, int, tuple[str, ...]], ...]] = (
    (
        "026-sock-cloth",
        7,
        "sheet",
        36,
        ("brics-odroid-002_cam0",),
    ),
    ("031-cotton-cloth", 0, "sheet", 32, ()),
    (
        "036-napkin-cloth",
        9,
        "sheet",
        32,
        ("brics-odroid-025_cam0",),
    ),
    (
        "058-roll-napkin",
        1,
        "volumetric",
        32,
        ("brics-odroid-002_cam0", "brics-odroid-007_cam1"),
    ),
    (
        "152-slime",
        8,
        "volumetric",
        32,
        (
            "brics-odroid-002_cam0",
            "brics-odroid-012_cam1",
            "brics-odroid-016_cam0",
        ),
    ),
    (
        "153-cake",
        5,
        "volumetric",
        32,
        ("brics-odroid-002_cam0", "brics-odroid-016_cam0"),
    ),
    (
        "167-glove-gray-cloth",
        0,
        "sheet",
        32,
        ("brics-odroid-002_cam0", "brics-odroid-007_cam1"),
    ),
    ("186-monster", 6, "volumetric", 32, ()),
    ("193-frog", 7, "volumetric", 32, ()),
    ("198-kneepad-cloth", 2, "sheet", 32, ()),
)
_EXPECTED_OBJECT_BY_ID: Final = {
    object_id: {
        "episode_id": episode_id,
        "stratum": stratum,
        "admitted_stream_count": admitted_stream_count,
        "support_negative_camera_ids": support_negative_camera_ids,
    }
    for (
        object_id,
        episode_id,
        stratum,
        admitted_stream_count,
        support_negative_camera_ids,
    ) in _EXPECTED_OBJECTS
}

_DIGEST = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        plain_json(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} must use literal string keys")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _exact_fields(
    value: Mapping[str, Any],
    *,
    expected: frozenset[str],
    name: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} fields changed; missing={missing}, extra={extra}")


def _nonempty_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be nonempty text")
    return cast(str, value)


def _genuine_integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a genuine integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _sha256_digest(value: object, *, name: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return cast(str, value)


def _git_revision(value: object, *, name: str) -> str:
    if type(value) is not str or _REVISION.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase full Git revision")
    return cast(str, value)


def _canonical_relative_json_path(value: object, *, name: str) -> str:
    text = _nonempty_text(value, name=name)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or path.suffix != ".json"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{name} must be a canonical relative JSON path")
    return text


@dataclass(frozen=True, slots=True)
class Deform360SourceSupportEvidenceLockV1:
    """Immutable source and Actions identities for one evidence materialization."""

    source_workflow_run_id: int
    source_workflow_run_attempt: int
    source_revision: str
    source_artifact_id: int
    source_artifact_name: str
    source_artifact_sha256: str
    metric_batch_relative_path: str
    metric_batch_sha256: str
    metric_batch_bytes: int
    metric_batch_result_id: str
    production_result_id: str
    admission_id: str
    support_receipt_sha256: str
    pipeline_receipt_sha256: str

    def __post_init__(self) -> None:
        _genuine_integer(
            self.source_workflow_run_id,
            name="source_workflow_run_id",
            minimum=1,
        )
        _genuine_integer(
            self.source_workflow_run_attempt,
            name="source_workflow_run_attempt",
            minimum=1,
        )
        _git_revision(self.source_revision, name="source_revision")
        _genuine_integer(self.source_artifact_id, name="source_artifact_id", minimum=1)
        _nonempty_text(self.source_artifact_name, name="source_artifact_name")
        _sha256_digest(self.source_artifact_sha256, name="source_artifact_sha256")
        _canonical_relative_json_path(
            self.metric_batch_relative_path,
            name="metric_batch_relative_path",
        )
        _sha256_digest(self.metric_batch_sha256, name="metric_batch_sha256")
        _genuine_integer(self.metric_batch_bytes, name="metric_batch_bytes", minimum=1)
        _sha256_digest(self.metric_batch_result_id, name="metric_batch_result_id")
        _sha256_digest(self.production_result_id, name="production_result_id")
        _sha256_digest(self.admission_id, name="admission_id")
        _sha256_digest(self.support_receipt_sha256, name="support_receipt_sha256")
        _sha256_digest(self.pipeline_receipt_sha256, name="pipeline_receipt_sha256")

    def metadata(self) -> dict[str, object]:
        """Return finite JSON metadata carried into the equal-object evidence."""

        return {
            "source_workflow_run_id": self.source_workflow_run_id,
            "source_workflow_run_attempt": self.source_workflow_run_attempt,
            "source_revision": self.source_revision,
            "source_artifact_id": self.source_artifact_id,
            "source_artifact_name": self.source_artifact_name,
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_metric_batch_relative_path": self.metric_batch_relative_path,
            "source_metric_batch_sha256": self.metric_batch_sha256,
            "source_metric_batch_bytes": self.metric_batch_bytes,
            "source_metric_batch_result_id": self.metric_batch_result_id,
            "source_production_result_id": self.production_result_id,
            "source_admission_id": self.admission_id,
            "source_support_receipt_sha256": self.support_receipt_sha256,
            "source_pipeline_receipt_sha256": self.pipeline_receipt_sha256,
        }


FROZEN_DEFORM360_SOURCE_SUPPORT_EVIDENCE_LOCK_V1: Final = (
    Deform360SourceSupportEvidenceLockV1(
        source_workflow_run_id=31297018948,
        source_workflow_run_attempt=1,
        source_revision="ded8910becbbffe958dfd18c84ad91069e7087a4",
        source_artifact_id=9033414269,
        source_artifact_name="deform360-prob4d-source-gate-31297018948-1",
        source_artifact_sha256=(
            "7247a2a260509c4c226e7ca437aff09d090abf6d2ca08f471a2143ea7d4bf7de"
        ),
        metric_batch_relative_path=(
            "bayesian-phystwin/deform360-prob4d-source-gate-v1/"
            "146f885351b2af0134b8b3d3c28a76deaa899749b1b1306e0d7061807ae95f89/"
            "ded8910becbbffe958dfd18c84ad91069e7087a4/metric-batch/"
            "metric-batch-result.json"
        ),
        metric_batch_sha256=(
            "679550aff53d3b615f63c66ee78318258893867511dd6c33100d1cf10c0f5be6"
        ),
        metric_batch_bytes=151278,
        metric_batch_result_id=(
            "f246394c84fd643b6ec8961dbcb2101a73c34e46d5eaf43961f28429aeb197eb"
        ),
        production_result_id=(
            "146f885351b2af0134b8b3d3c28a76deaa899749b1b1306e0d7061807ae95f89"
        ),
        admission_id=(
            "715ab8479bad4d97eba766cdba1a161f1f6e83e3fd597bb09a2bf8ab8dc91e15"
        ),
        support_receipt_sha256=(
            "2c14774dd0f0f96301483a46da148de392442794329da6dcb97dd61e3ca7e07f"
        ),
        pipeline_receipt_sha256=(
            "8588f6e7b3115808c49cc781a27093308b6011f528e3aabfae936daa17994dfd"
        ),
    )
)

_AGGREGATION_POLICY: Final[dict[str, object]] = {
    "schema": "bayesian_phystwin.deform360_source_support_failure_aggregation",
    "schema_version": DEFORM360_SOURCE_SUPPORT_FAILURE_EVIDENCE_VERSION,
    "source_metric_batch_schema": DEFORM360_SOURCE_SUPPORT_METRIC_BATCH_SCHEMA,
    "source_metric_batch_version": DEFORM360_SOURCE_SUPPORT_METRIC_BATCH_VERSION,
    "source_metric_batch_semantics": DEFORM360_SOURCE_SUPPORT_METRIC_BATCH_SEMANTICS,
    "statistical_unit": "physical-object",
    "accepted_decision": False,
    "direct_failure_signal": "provider_support_complete",
    "direct_failure_reason": DEFORM360_SOURCE_SUPPORT_NEGATIVE_REASON,
    "global_rejection_reason": DEFORM360_SOURCE_SUPPORT_GLOBAL_REJECTION_REASON,
    "expected_objects": [
        {
            "object_id": object_id,
            "episode_id": episode_id,
            "stratum": stratum,
            "admitted_stream_count": admitted_stream_count,
            "support_negative_camera_ids": list(support_negative_camera_ids),
        }
        for (
            object_id,
            episode_id,
            stratum,
            admitted_stream_count,
            support_negative_camera_ids,
        ) in _EXPECTED_OBJECTS
    ],
    "claim_boundary": DEFORM360_SOURCE_SUPPORT_FAILURE_CLAIM_BOUNDARY,
}
DEFORM360_SOURCE_SUPPORT_AGGREGATION_POLICY_ID: Final = _canonical_sha256(
    _AGGREGATION_POLICY
)


def _validate_metric_batch_header(
    metric_batch: Mapping[str, Any],
    *,
    lock: Deform360SourceSupportEvidenceLockV1,
) -> None:
    _exact_fields(
        metric_batch,
        expected=_EXPECTED_METRIC_BATCH_FIELDS,
        name="metric batch",
    )
    _require(
        metric_batch.get("schema") == DEFORM360_SOURCE_SUPPORT_METRIC_BATCH_SCHEMA,
        "metric batch schema changed",
    )
    _require(
        metric_batch.get("schema_version")
        == DEFORM360_SOURCE_SUPPORT_METRIC_BATCH_VERSION,
        "metric batch schema version changed",
    )
    _require(
        metric_batch.get("semantics")
        == DEFORM360_SOURCE_SUPPORT_METRIC_BATCH_SEMANTICS,
        "metric batch semantics changed",
    )
    result_id = _sha256_digest(metric_batch.get("result_id"), name="result_id")
    identity = {key: value for key, value in metric_batch.items() if key != "result_id"}
    _require(result_id == _canonical_sha256(identity), "metric batch result ID changed")
    _require(result_id == lock.metric_batch_result_id, "metric batch lock changed")
    _require(
        _git_revision(
            metric_batch.get("implementation_revision"),
            name="implementation_revision",
        )
        == lock.source_revision,
        "metric batch implementation revision changed",
    )
    _require(
        _sha256_digest(
            metric_batch.get("production_result_id"),
            name="production_result_id",
        )
        == lock.production_result_id,
        "visual-production result identity changed",
    )
    _require(
        _sha256_digest(metric_batch.get("admission_id"), name="admission_id")
        == lock.admission_id,
        "visual-production admission identity changed",
    )
    expected_counts = {
        "object_count": 10,
        "admitted_stream_count": 324,
        "supported_stream_count": 313,
        "support_negative_stream_count": 11,
        "technical_failure_stream_count": 0,
        "supported_object_count": 10,
    }
    actual_counts = {
        key: _genuine_integer(metric_batch.get(key), name=key)
        for key in expected_counts
    }
    _require(actual_counts == expected_counts, "metric batch accounting changed")
    _require(metric_batch.get("plan_emitted") is False, "metric plan must not exist")
    _require(metric_batch.get("plan_file") is None, "metric plan file must be null")
    _require(
        metric_batch.get("status") == "support-negatives-retained",
        "metric batch terminal status changed",
    )
    _nonempty_text(metric_batch.get("claim_boundary"), name="claim_boundary")

    boundary = _mapping(
        metric_batch.get("information_boundary"),
        name="metric batch information_boundary",
    )
    _require(dict(boundary) == _EXPECTED_BOUNDARY, "information boundary changed")

    source_artifacts = _mapping(
        metric_batch.get("source_artifacts"),
        name="metric batch source_artifacts",
    )
    _require(bool(source_artifacts), "metric batch source artifacts must not be empty")
    for name, digest in source_artifacts.items():
        _nonempty_text(name, name="source artifact name")
        _sha256_digest(digest, name=f"source artifact {name!r}")


def _validate_metric_jobs(
    metric_batch: Mapping[str, Any],
) -> dict[str, list[dict[str, object]]]:
    raw_jobs = _sequence(metric_batch.get("jobs"), name="metric batch jobs")
    _require(len(raw_jobs) == 324, "metric batch must retain all 324 jobs")
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    job_ids: set[str] = set()
    object_camera_pairs: set[tuple[str, str]] = set()

    for index, raw_job in enumerate(raw_jobs):
        job = _mapping(raw_job, name=f"metric job {index}")
        _exact_fields(job, expected=_EXPECTED_JOB_FIELDS, name=f"metric job {index}")
        job_id = _sha256_digest(job.get("job_id"), name=f"metric job {index} job_id")
        _require(job_id not in job_ids, "metric batch repeats a job_id")
        job_ids.add(job_id)
        object_id = _nonempty_text(
            job.get("object_id"),
            name=f"metric job {index} object_id",
        )
        expected_object = _EXPECTED_OBJECT_BY_ID.get(object_id)
        if expected_object is None:
            raise ValueError("metric batch object roster changed")
        episode_id = _genuine_integer(
            job.get("episode_id"),
            name=f"metric job {index} episode_id",
        )
        stratum = _nonempty_text(
            job.get("stratum"),
            name=f"metric job {index} stratum",
        )
        _require(
            episode_id == expected_object["episode_id"]
            and stratum == expected_object["stratum"],
            "metric job object identity changed",
        )
        camera_id = _nonempty_text(
            job.get("camera_id"),
            name=f"metric job {index} camera_id",
        )
        object_camera = (object_id, camera_id)
        _require(
            object_camera not in object_camera_pairs,
            "metric batch repeats an object/camera stream",
        )
        object_camera_pairs.add(object_camera)
        relative_output = _nonempty_text(
            job.get("output_relative_directory"),
            name=f"metric job {index} output_relative_directory",
        )
        expected_prefix = (
            f"objects/{object_id}/episode_{episode_id:04d}/views/{camera_id}"
        )
        _require(
            relative_output == expected_prefix,
            "metric job output directory changed",
        )
        status = _nonempty_text(job.get("status"), name=f"metric job {index} status")
        projected_point_count = _genuine_integer(
            job.get("projected_point_count"),
            name=f"metric job {index} projected_point_count",
        )
        normalized: dict[str, object] = {
            "job_id": job_id,
            "object_id": object_id,
            "episode_id": episode_id,
            "stratum": stratum,
            "camera_id": camera_id,
            "status": status,
            "projected_point_count": projected_point_count,
        }
        if status == "supported":
            normalized["metric_artifact_id"] = _sha256_digest(
                job.get("metric_artifact_id"),
                name=f"metric job {index} metric_artifact_id",
            )
            _require(projected_point_count > 0, "supported job has no projected points")
            _require(
                job.get("failure_reason") is None
                and job.get("failure_detail_sha256") is None,
                "supported job contains failure metadata",
            )
        elif status == "support-negative":
            _require(
                job.get("metric_artifact_id") is None,
                "support-negative job contains a metric artifact",
            )
            _require(projected_point_count == 0, "support-negative job has points")
            _require(
                job.get("failure_reason") == DEFORM360_SOURCE_SUPPORT_NEGATIVE_REASON,
                "support-negative reason changed",
            )
            _require(
                job.get("failure_detail_sha256") is None,
                "support-negative detail identity changed",
            )
        else:
            raise ValueError("metric batch contains a technical or unknown job status")
        groups[object_id].append(normalized)

    _require(
        set(groups) == set(_EXPECTED_OBJECT_BY_ID),
        "metric batch object coverage changed",
    )
    return dict(groups)


def _validate_object_groups(
    groups: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    supported_total = 0
    support_negative_total = 0
    for (
        object_id,
        episode_id,
        stratum,
        admitted_stream_count,
        expected_negative_cameras,
    ) in _EXPECTED_OBJECTS:
        rows = list(groups[object_id])
        _require(
            len(rows) == admitted_stream_count,
            f"object {object_id!r} admitted stream count changed",
        )
        _require(
            all(
                row["episode_id"] == episode_id and row["stratum"] == stratum
                for row in rows
            ),
            f"object {object_id!r} identity changed",
        )
        supported = [row for row in rows if row["status"] == "supported"]
        negatives = [row for row in rows if row["status"] == "support-negative"]
        actual_negative_cameras = tuple(
            sorted(cast(str, row["camera_id"]) for row in negatives)
        )
        _require(
            actual_negative_cameras == expected_negative_cameras,
            f"object {object_id!r} support-negative cameras changed",
        )
        supported_total += len(supported)
        support_negative_total += len(negatives)
    _require(supported_total == 313, "supported stream total changed")
    _require(support_negative_total == 11, "support-negative stream total changed")


def _build_records(
    groups: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for (
        object_id,
        episode_id,
        stratum,
        admitted_stream_count,
        _expected_negative_cameras,
    ) in _EXPECTED_OBJECTS:
        rows = list(groups[object_id])
        supported = [row for row in rows if row["status"] == "supported"]
        negatives = [row for row in rows if row["status"] == "support-negative"]
        support_complete = not negatives and len(supported) == admitted_stream_count
        result_reason = (
            DEFORM360_SOURCE_SUPPORT_GLOBAL_REJECTION_REASON
            if support_complete
            else DEFORM360_SOURCE_SUPPORT_NEGATIVE_REASON
        )
        records.append(
            {
                "case_id": object_id,
                "accepted": False,
                "result_reason": result_reason,
                "signals": {
                    "technical_valid": True,
                    "provider_support_complete": support_complete,
                    "numerically_converged": None,
                    "query_identifiable": None,
                    "gauge_or_common_mode_consistent": None,
                    "covariance_calibrated": None,
                    "material_identity_reliable": None,
                    "robust_support_sufficient": None,
                    "physical_guard_passed": None,
                },
                "metrics": {
                    "object_id": object_id,
                    "episode_id": episode_id,
                    "stratum": stratum,
                    "admitted_stream_count": admitted_stream_count,
                    "supported_stream_count": len(supported),
                    "support_negative_stream_count": len(negatives),
                    "technical_failure_stream_count": 0,
                    "support_complete_for_object": support_complete,
                    "support_negative_camera_ids": sorted(
                        cast(str, row["camera_id"]) for row in negatives
                    ),
                    "support_negative_job_ids": sorted(
                        cast(str, row["job_id"]) for row in negatives
                    ),
                },
            }
        )
    return records


def build_deform360_source_support_failure_evidence_v1(
    metric_batch: Mapping[str, object],
    *,
    lock: Deform360SourceSupportEvidenceLockV1,
) -> dict[str, object]:
    """Aggregate one frozen stream-level support result to ten equal objects."""

    batch = _mapping(metric_batch, name="metric batch")
    _validate_metric_batch_header(batch, lock=lock)
    groups = _validate_metric_jobs(batch)
    _validate_object_groups(groups)
    records = _build_records(groups)
    source_artifacts = dict(
        sorted(
            _mapping(
                batch["source_artifacts"],
                name="metric batch source_artifacts",
            ).items()
        )
    )
    payload: dict[str, object] = {
        "schema": DEFORM360_SOURCE_SUPPORT_FAILURE_EVIDENCE_SCHEMA,
        "schema_version": PROVIDER_FAILURE_EVIDENCE_VERSION,
        "provider_id": lock.production_result_id,
        "records": records,
        "metadata": {
            "split": "source-only",
            "statistical_unit": "physical-object",
            "confirmation_payloads_opened": False,
            "adaptive_confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "future_frames_used": False,
            "replacement_allowed": False,
            **lock.metadata(),
            "source_metric_batch_schema": batch["schema"],
            "source_metric_batch_schema_version": batch["schema_version"],
            "source_metric_batch_semantics": batch["semantics"],
            "source_terminal_stage": "metric-support",
            "source_terminal_status": batch["status"],
            "source_admitted_stream_count": batch["admitted_stream_count"],
            "source_supported_stream_count": batch["supported_stream_count"],
            "source_support_negative_stream_count": batch[
                "support_negative_stream_count"
            ],
            "source_technical_failure_stream_count": batch[
                "technical_failure_stream_count"
            ],
            "source_object_count": batch["object_count"],
            "source_plan_emitted": batch["plan_emitted"],
            "source_artifacts": source_artifacts,
            "aggregation_policy_id": (DEFORM360_SOURCE_SUPPORT_AGGREGATION_POLICY_ID),
            "aggregation_semantics": (DEFORM360_SOURCE_SUPPORT_AGGREGATION_SEMANTICS),
            "direct_object_failure_count": 6,
            "unresolved_global_rejection_count": 4,
            "claim_boundary": DEFORM360_SOURCE_SUPPORT_FAILURE_CLAIM_BOUNDARY,
        },
    }
    report = validate_deform360_provider_failure_census_payload(payload)
    expected_report = {
        "record_count": 10,
        "accepted_count": 0,
        "classified_rejection_count": 6,
        "unresolved_rejection_count": 4,
    }
    actual_report = {key: report[key] for key in expected_report}
    _require(actual_report == expected_report, "equal-object census accounting changed")
    primary_counts = cast(Mapping[str, object], report["primary_category_counts"])
    _require(
        primary_counts.get("unsupported-provider-geometry") == 6
        and primary_counts.get("unresolved-rejection") == 4,
        "equal-object primary attribution changed",
    )
    return payload


__all__ = [
    "DEFORM360_SOURCE_SUPPORT_AGGREGATION_POLICY_ID",
    "DEFORM360_SOURCE_SUPPORT_FAILURE_CLAIM_BOUNDARY",
    "DEFORM360_SOURCE_SUPPORT_FAILURE_EVIDENCE_SCHEMA",
    "DEFORM360_SOURCE_SUPPORT_FAILURE_EVIDENCE_VERSION",
    "DEFORM360_SOURCE_SUPPORT_GLOBAL_REJECTION_REASON",
    "DEFORM360_SOURCE_SUPPORT_NEGATIVE_REASON",
    "Deform360SourceSupportEvidenceLockV1",
    "FROZEN_DEFORM360_SOURCE_SUPPORT_EVIDENCE_LOCK_V1",
    "build_deform360_source_support_failure_evidence_v1",
]
