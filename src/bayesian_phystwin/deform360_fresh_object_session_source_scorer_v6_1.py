"""Post-seal public-source geometry scoring for Deform360 v6.1.

The module has no prefix-production interface.  It first validates the exact
100-record candidate barrier, then authorizes access to the released source
suffix.  Every candidate is scored on one B0-defined graph-node roster so that
missing projections cannot improve a challenger.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import genuine_integer, plain_json
from ._portable_contracts import (
    canonical_relative_posix_path,
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
    write_atomic_json,
)
from .deform360_fresh_object_session_candidate_runner_v6_1 import (
    CANDIDATE_BATCH_FILENAME,
    CANDIDATE_RECEIPT_FILENAME,
    validate_deform360_v61_candidate_execution_receipt,
    validate_deform360_v61_candidate_panel,
)
from .deform360_fresh_object_session_candidate_v6_1 import (
    CANDIDATE_SEAL_FILENAME,
    EVALUATION_RANGE,
    load_deform360_v61_candidate_artifact,
)
from .deform360_fresh_object_session_source_v6 import (
    B0,
    B1,
    D1_NATIVE,
    VARIANT_IDS,
    VT1_OBSERVED,
    VT1_SANDWICH,
    VT1_WORKING,
)
from .deform360_fresh_object_session_source_v6_1 import (
    assemble_deform360_v6_nested_evidence,
    build_deform360_v6_raw_nested_outcome,
    evaluate_deform360_v6_nested_source_gate,
    publish_deform360_v6_nested_evidence,
    publish_deform360_v6_nested_result,
    validate_deform360_v6_raw_nested_batch,
)
from .deform360_joint_sparse_endpoint_v5 import Deform360ReservedViewGeometryV5
from .deform360_joint_sparse_source_gate_v5 import (
    load_deform360_joint_sparse_source_execution_lock_v5,
)
from .deform360_joint_sparse_source_runner_v5 import _cohort

SOURCE_SCORING_AMENDMENT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-v6-source-scoring-amendment"
)
SOURCE_SCORING_AMENDMENT_ID: Final = (
    "e8b962a8abf228114495683cfb9ba87ee802e7405ca92f1f28b5a76df3faa371"
)
SOURCE_SCORING_AMENDMENT_FILE_SHA256: Final = (
    "c616fe1fbe19785452535772adfa937501a0fa35ab41b3c2fc995a968e60a8f1"
)
CANDIDATE_REVISION: Final = "2eb8d12e2120d58d0d678c3771d29faaeb765497"
CANDIDATE_PANEL_RECEIPT_ID: Final = (
    "db3cc4351436492db5962bc1e99f516adc38a5031140b675b45dc6d752b7559a"
)
CANDIDATE_PANEL_RECEIPT_FILE_SHA256: Final = (
    "2b96b8c92fe3be4e7ea92fd8a58fa3c3858279d2cc3d081d723d870fd84ff7ed"
)
CANDIDATE_EXECUTION_RECEIPT_ID: Final = (
    "65747822fa8380296a572811772fce88b9275a7e1148a8015e1156f520f7e369"
)
CANDIDATE_EXECUTION_RECEIPT_FILE_SHA256: Final = (
    "d2569acd499c7d21a11937a22de012418e6f15579394944ffe7399a8b89c3bf6"
)
RAW_PREDICTION_BATCH_ID: Final = (
    "d27674518f523db4fddb9cc108dd3d77321dddefeccc866b2b81044bf44ebee8"
)
RAW_PREDICTION_BATCH_FILE_SHA256: Final = (
    "b3a22f24015d0c0d8e757b5946dafd21dd61394c1c91a872a9eb56d28e2a74e6"
)
UPSTREAM_SOURCE_PLAN_ID: Final = (
    "d9b9e4df9d020e8ae076f407f61d5e1f328c68d2f4fe4d8e4ad1688d2d253100"
)
UPSTREAM_SOURCE_PLAN_FILE_SHA256: Final = (
    "08863166df11033f4a968c94a4cb5bd02869175ce3bf1c1859c8ac49be371991"
)
UPSTREAM_REVISION: Final = "913909596b71ac6ad717835ce7a87ae01e42c5ab"
EXECUTION_LOCK_ID: Final = (
    "76b74483790ace51d642889be2e3dbb22149e30f7919b5855a18066434e25189"
)
CANDIDATE_WORKFLOW_RUN_ID: Final = 31647329129
CANDIDATE_WORKFLOW_RUN_ATTEMPT: Final = 1

SOURCE_SUFFIX_AUTHORIZATION_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-v6-source-suffix-authorization"
)
SOURCE_SUFFIX_AUTHORIZATION_VERSION: Final = 1
SOURCE_ENDPOINT_MANIFEST_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-v6-source-endpoint-manifest"
)
SOURCE_ENDPOINT_MANIFEST_VERSION: Final = 1
SOURCE_SCORING_RECEIPT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-v6-source-scoring-receipt"
)
SOURCE_SCORING_RECEIPT_VERSION: Final = 1
SOURCE_SCORING_TECHNICAL_FAILURE_RECEIPT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-v6-"
    "source-scoring-technical-failure-receipt"
)
SOURCE_SCORING_TECHNICAL_FAILURE_RECEIPT_VERSION: Final = 1
TARGET_PIXEL_RANKING_DOMAIN: Final = b"v61-source-target-pixel-v1"
CANDIDATE_IDS: Final = (B0, B1, D1_NATIVE)
ENDPOINT_ARCHIVE_MEMBERS: Final = frozenset(
    {
        "camera_to_world",
        "depth_m",
        "frame_indices",
        "intrinsics",
        "object_mask",
        "raw_frame_indices",
    }
)

_AUTHORIZATION_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "source_scoring_amendment_id",
        "scorer_revision",
        "runner_name",
        "workflow_run_id",
        "workflow_run_attempt",
        "candidate_revision",
        "candidate_workflow_run_id",
        "candidate_workflow_run_attempt",
        "candidate_execution_receipt_id",
        "candidate_execution_receipt_file_sha256",
        "candidate_panel_receipt_id",
        "candidate_panel_receipt_file_sha256",
        "raw_prediction_batch_id",
        "raw_prediction_batch_file_sha256",
        "upstream_source_plan_id",
        "upstream_source_plan_file_sha256",
        "prediction_record_count",
        "technical_failure_record_count",
        "development_suffix_access_authorized",
        "confirmation_payloads_opened",
        "human_approval_required",
        "information_boundary",
        "authorization_id",
    }
)
_AUTHORIZATION_BOUNDARY: Final = {
    "all_100_raw_predictions_revalidated": True,
    "source_suffix_opened": False,
    "source_suffix_access_authorized": True,
    "future_object_observations_used_for_prediction": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_opened": False,
    "held_v8_artifacts_accessed": False,
    "human_approval_required": False,
    "human_selection_used": False,
    "replacement_allowed": False,
    "public_real_world_dataset": True,
    "new_measurements_collected": False,
    "prob4d_decoded_uniform_fusion_used": False,
    "motioncrafter_disjoint_baseline_used": True,
}
_ENDPOINT_FILE_FIELDS: Final = frozenset({"path", "sha256"})
_ENDPOINT_VIEW_FIELDS: Final = frozenset({"camera_id", "endpoint_archive"})
_ENDPOINT_OBJECT_FIELDS: Final = frozenset(
    {
        "object_id",
        "episode_id",
        "stratum",
        "all_camera_ids",
        "raw_endpoint_range_half_open",
        "reserved_views",
    }
)
_ENDPOINT_MANIFEST_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "source_scoring_amendment_id",
        "authorization_id",
        "upstream_source_plan_id",
        "processor_revision",
        "objects",
        "information_boundary",
        "manifest_id",
    }
)
_ENDPOINT_BOUNDARY: Final = {
    "candidate_predictions_sealed_before_suffix_open": True,
    "development_source_suffix_opened": True,
    "future_geometry_used_for_prediction": False,
    "reserved_views_contributed_likelihood": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_opened": False,
    "held_v8_artifacts_accessed": False,
    "human_approval_required": False,
    "human_selection_used": False,
    "replacement_allowed": False,
    "released_real_world_recordings_only": True,
}
_SCORING_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        "source_scoring_amendment_id",
        "scorer_revision",
        "runner_name",
        "workflow_run_id",
        "workflow_run_attempt",
        "authorization_id",
        "endpoint_manifest_id",
        "candidate_panel_receipt_id",
        "raw_prediction_batch_id",
        "prediction_record_count",
        "outcome_count",
        "evidence_id",
        "result_id",
        "source_gate_passed",
        "source_continuation_authorized",
        "independent_confirmation_authorized",
        "claim_authorized",
        "artifacts",
        "information_boundary",
        "receipt_id",
    }
)
_SCORING_TECHNICAL_FAILURE_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        "source_scoring_amendment_id",
        "scorer_revision",
        "runner_name",
        "workflow_run_id",
        "workflow_run_attempt",
        "candidate_panel_receipt_id",
        "raw_prediction_batch_id",
        "authorization_id",
        "terminal_stage",
        "exit_code",
        "source_suffix_access_authorized",
        "source_suffix_opened",
        "source_gate_evaluated",
        "source_gate_passed",
        "source_continuation_authorized",
        "independent_confirmation_authorized",
        "claim_authorized",
        "retained_artifacts",
        "information_boundary",
        "receipt_id",
    }
)
_SCORING_BOUNDARY = {
    "all_100_raw_predictions_revalidated_before_suffix_open": True,
    "source_suffix_opened": True,
    "future_object_observations_used_for_prediction": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_opened": False,
    "held_v8_artifacts_accessed": False,
    "human_approval_required": False,
    "human_selection_used": False,
    "replacement_allowed": False,
    "public_real_world_dataset": True,
    "new_measurements_collected": False,
    "prob4d_decoded_uniform_fusion_used": False,
    "motioncrafter_disjoint_baseline_used": True,
}


def _technical_failure_boundary(*, source_suffix_opened: bool) -> dict[str, bool]:
    return {
        "all_100_raw_predictions_revalidated_before_suffix_open": True,
        "source_suffix_opened": source_suffix_opened,
        "future_object_observations_used_for_prediction": False,
        "confirmation_payloads_opened": False,
        "target_outcomes_opened": False,
        "held_v8_artifacts_accessed": False,
        "human_approval_required": False,
        "human_selection_used": False,
        "replacement_allowed": False,
        "public_real_world_dataset": True,
        "new_measurements_collected": False,
        "prob4d_decoded_uniform_fusion_used": False,
        "motioncrafter_disjoint_baseline_used": True,
    }


class SourceEndpointSupportError(ValueError):
    """The frozen public endpoint carrier lacks preregistered support."""


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


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _content_identity(value: Mapping[str, Any], *, field: str, name: str) -> None:
    declared = sha256_digest(value.get(field), name=field)
    identity = {key: item for key, item in value.items() if key != field}
    _require(declared == content_id(identity), f"{name} identity changed")


def _publish_or_validate_json(
    value: Mapping[str, Any], path: str | Path, *, label: str
) -> dict[str, Any]:
    normalized = cast(dict[str, Any], plain_json(value))
    destination = Path(path)
    if destination.exists():
        _require(
            not destination.is_symlink()
            and load_strict_json_object(destination, label=label) == normalized,
            f"existing {label} differs",
        )
    else:
        write_atomic_json(normalized, destination, overwrite=False)
    return normalized


def load_deform360_v61_source_scoring_amendment(
    path: str | Path,
) -> dict[str, Any]:
    """Load the outcome-blind scoring contract frozen after the prefix barrier."""

    source = Path(path)
    _require(
        _sha256_file(source) == SOURCE_SCORING_AMENDMENT_FILE_SHA256,
        "source scoring amendment bytes changed",
    )
    value = load_strict_json_object(source, label="v6.1 source scoring amendment")
    _require(
        value.get("schema") == SOURCE_SCORING_AMENDMENT_SCHEMA
        and value.get("schema_version") == 1
        and value.get("amendment_id") == SOURCE_SCORING_AMENDMENT_ID,
        "source scoring amendment contract changed",
    )
    _content_identity(value, field="amendment_id", name="source scoring amendment")
    barrier = _mapping(value.get("candidate_barrier"), name="candidate_barrier")
    _require(
        barrier.get("candidate_revision") == CANDIDATE_REVISION
        and barrier.get("candidate_panel_receipt_id") == CANDIDATE_PANEL_RECEIPT_ID
        and barrier.get("candidate_execution_receipt_id")
        == CANDIDATE_EXECUTION_RECEIPT_ID
        and barrier.get("raw_prediction_batch_id") == RAW_PREDICTION_BATCH_ID
        and barrier.get("prediction_record_count") == 100
        and barrier.get("technical_failure_record_count") == 0
        and barrier.get("workflow_run_id") == CANDIDATE_WORKFLOW_RUN_ID
        and barrier.get("workflow_run_attempt") == CANDIDATE_WORKFLOW_RUN_ATTEMPT,
        "source scoring amendment binds another candidate barrier",
    )
    boundary = _mapping(value.get("information_boundary"), name="boundary")
    _require(
        boundary.get("source_suffix_opened") is False
        and boundary.get("confirmation_payloads_opened") is False
        and boundary.get("target_outcomes_opened") is False
        and boundary.get("held_v8_artifacts_accessed") is False
        and boundary.get("human_approval_required") is False
        and boundary.get("new_measurements_collected") is False,
        "source scoring amendment crossed its information boundary",
    )
    return cast(dict[str, Any], plain_json(value))


def validate_deform360_v61_source_plan(value: object) -> dict[str, Any]:
    """Validate the exact presealed public-source camera plan."""

    plan = _mapping(value, name="source plan")
    _content_identity(plan, field="plan_id", name="source plan")
    _require(
        plan.get("schema")
        == "bayesian-phystwin.deform360-joint-sparse-source-prediction-plan"
        and plan.get("schema_version") == 6
        and plan.get("plan_id") == UPSTREAM_SOURCE_PLAN_ID
        and plan.get("implementation_revision") == UPSTREAM_REVISION
        and plan.get("execution_lock_id") == EXECUTION_LOCK_ID,
        "source plan lineage changed",
    )
    objects = _sequence(plan.get("objects"), name="source plan objects")
    _require(len(objects) == 10, "source plan must contain ten units")
    seen: set[str] = set()
    for index, raw in enumerate(objects):
        row = _mapping(raw, name=f"source plan object {index}")
        object_id = _identifier(row.get("object_id"), name="object_id")
        cameras = tuple(
            _identifier(item, name="camera_id")
            for item in _sequence(row.get("all_camera_ids"), name="all_camera_ids")
        )
        reserved = tuple(
            _identifier(item, name="reserved camera")
            for item in _sequence(
                row.get("reserved_endpoint_camera_ids"),
                name="reserved_endpoint_camera_ids",
            )
        )
        prefix = tuple(
            _sequence(
                row.get("raw_prefix_range_half_open"),
                name="raw_prefix_range_half_open",
            )
        )
        _require(
            object_id not in seen
            and len(cameras) == len(set(cameras))
            and len(reserved) == len(set(reserved)) == 2
            and set(reserved) <= set(cameras)
            and len(prefix) == 2
            and all(type(item) is int for item in prefix)
            and prefix[1] - prefix[0] == EVALUATION_RANGE[0],
            "source plan object or camera roster changed",
        )
        seen.add(object_id)
    return cast(dict[str, Any], plain_json(plan))


def _load_exact_source_plan(path: str | Path) -> dict[str, Any]:
    _require(
        _sha256_file(path) == UPSTREAM_SOURCE_PLAN_FILE_SHA256,
        "upstream source plan bytes changed",
    )
    return validate_deform360_v61_source_plan(
        load_strict_json_object(path, label="upstream source plan")
    )


def build_deform360_v61_source_suffix_authorization(
    *,
    source_scoring_amendment_path: str | Path,
    execution_lock_path: str | Path,
    candidate_root: str | Path,
    candidate_execution_receipt_path: str | Path,
    upstream_source_plan_path: str | Path,
    scorer_revision: str,
    runner_name: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
) -> dict[str, Any]:
    """Revalidate the prefix barrier and authorize source suffix access."""

    load_deform360_v61_source_scoring_amendment(source_scoring_amendment_path)
    lock = load_deform360_joint_sparse_source_execution_lock_v5(execution_lock_path)
    _require(lock.get("execution_lock_id") == EXECUTION_LOCK_ID, "lock changed")
    panel = validate_deform360_v61_candidate_panel(
        execution_lock_path=execution_lock_path,
        output_root=candidate_root,
    )
    panel_path = Path(candidate_root) / CANDIDATE_RECEIPT_FILENAME
    batch_path = Path(candidate_root) / CANDIDATE_BATCH_FILENAME
    _require(
        panel.get("receipt_id") == CANDIDATE_PANEL_RECEIPT_ID
        and panel.get("candidate_revision") == CANDIDATE_REVISION
        and panel.get("raw_prediction_batch_id") == RAW_PREDICTION_BATCH_ID
        and panel.get("prediction_record_count") == 100
        and panel.get("technical_failure_record_count") == 0
        and _sha256_file(panel_path) == CANDIDATE_PANEL_RECEIPT_FILE_SHA256
        and _sha256_file(batch_path) == RAW_PREDICTION_BATCH_FILE_SHA256,
        "candidate panel differs from the frozen scoring barrier",
    )
    execution_path = Path(candidate_execution_receipt_path)
    execution = validate_deform360_v61_candidate_execution_receipt(
        load_strict_json_object(execution_path, label="candidate execution receipt")
    )
    _require(
        execution.get("receipt_id") == CANDIDATE_EXECUTION_RECEIPT_ID
        and execution.get("candidate_panel_receipt_id") == CANDIDATE_PANEL_RECEIPT_ID
        and execution.get("workflow_run_id") == CANDIDATE_WORKFLOW_RUN_ID
        and execution.get("workflow_run_attempt") == CANDIDATE_WORKFLOW_RUN_ATTEMPT
        and _sha256_file(execution_path) == CANDIDATE_EXECUTION_RECEIPT_FILE_SHA256,
        "candidate execution receipt changed",
    )
    source_plan = _load_exact_source_plan(upstream_source_plan_path)
    revision = exact_revision(scorer_revision, name="scorer_revision")
    identity: dict[str, Any] = {
        "schema": SOURCE_SUFFIX_AUTHORIZATION_SCHEMA,
        "schema_version": SOURCE_SUFFIX_AUTHORIZATION_VERSION,
        "source_scoring_amendment_id": SOURCE_SCORING_AMENDMENT_ID,
        "scorer_revision": revision,
        "runner_name": _identifier(runner_name, name="runner name"),
        "workflow_run_id": genuine_integer(
            workflow_run_id, name="workflow run ID", minimum=1
        ),
        "workflow_run_attempt": genuine_integer(
            workflow_run_attempt, name="workflow run attempt", minimum=1
        ),
        "candidate_revision": CANDIDATE_REVISION,
        "candidate_workflow_run_id": CANDIDATE_WORKFLOW_RUN_ID,
        "candidate_workflow_run_attempt": CANDIDATE_WORKFLOW_RUN_ATTEMPT,
        "candidate_execution_receipt_id": execution["receipt_id"],
        "candidate_execution_receipt_file_sha256": _sha256_file(execution_path),
        "candidate_panel_receipt_id": panel["receipt_id"],
        "candidate_panel_receipt_file_sha256": _sha256_file(panel_path),
        "raw_prediction_batch_id": panel["raw_prediction_batch_id"],
        "raw_prediction_batch_file_sha256": _sha256_file(batch_path),
        "upstream_source_plan_id": source_plan["plan_id"],
        "upstream_source_plan_file_sha256": _sha256_file(upstream_source_plan_path),
        "prediction_record_count": 100,
        "technical_failure_record_count": 0,
        "development_suffix_access_authorized": True,
        "confirmation_payloads_opened": False,
        "human_approval_required": False,
        "information_boundary": dict(_AUTHORIZATION_BOUNDARY),
    }
    return {**identity, "authorization_id": content_id(identity)}


def validate_deform360_v61_source_suffix_authorization(
    value: object,
) -> dict[str, Any]:
    authorization = _mapping(value, name="source suffix authorization")
    require_exact_fields(
        authorization,
        expected=_AUTHORIZATION_FIELDS,
        name="source suffix authorization",
    )
    _content_identity(
        authorization,
        field="authorization_id",
        name="source suffix authorization",
    )
    _require(
        authorization.get("schema") == SOURCE_SUFFIX_AUTHORIZATION_SCHEMA
        and authorization.get("schema_version") == SOURCE_SUFFIX_AUTHORIZATION_VERSION
        and authorization.get("source_scoring_amendment_id")
        == SOURCE_SCORING_AMENDMENT_ID
        and authorization.get("candidate_revision") == CANDIDATE_REVISION
        and authorization.get("candidate_execution_receipt_id")
        == CANDIDATE_EXECUTION_RECEIPT_ID
        and authorization.get("candidate_panel_receipt_id")
        == CANDIDATE_PANEL_RECEIPT_ID
        and authorization.get("raw_prediction_batch_id") == RAW_PREDICTION_BATCH_ID
        and authorization.get("upstream_source_plan_id") == UPSTREAM_SOURCE_PLAN_ID
        and authorization.get("prediction_record_count") == 100
        and authorization.get("technical_failure_record_count") == 0
        and authorization.get("development_suffix_access_authorized") is True
        and authorization.get("confirmation_payloads_opened") is False
        and authorization.get("human_approval_required") is False
        and authorization.get("information_boundary") == _AUTHORIZATION_BOUNDARY,
        "source suffix authorization contract changed",
    )
    exact_revision(authorization.get("scorer_revision"), name="scorer_revision")
    _identifier(authorization.get("runner_name"), name="runner name")
    genuine_integer(
        authorization.get("workflow_run_id"), name="workflow run ID", minimum=1
    )
    genuine_integer(
        authorization.get("workflow_run_attempt"),
        name="workflow run attempt",
        minimum=1,
    )
    return cast(dict[str, Any], plain_json(authorization))


def publish_deform360_v61_source_suffix_authorization(
    value: Mapping[str, Any], path: str | Path
) -> dict[str, Any]:
    normalized = validate_deform360_v61_source_suffix_authorization(value)
    return _publish_or_validate_json(
        normalized,
        path,
        label="source suffix authorization",
    )


@dataclass(frozen=True, slots=True)
class Deform360V61SourceScoreConfig:
    """Frozen endpoint support and scoring thresholds."""

    minimum_depth_m: float = 0.05
    maximum_depth_m: float = 2.5
    maximum_target_points: int = 4096
    minimum_target_points: int = 32
    prediction_occlusion_tolerance_m: float = 0.020
    minimum_cells_per_reserved_view: int = 9
    minimum_views_per_evaluation_frame: int = 1
    distance_chunk_size: int = 512

    def __post_init__(self) -> None:
        _require(
            0.0 < self.minimum_depth_m < self.maximum_depth_m
            and self.maximum_target_points >= self.minimum_target_points >= 1
            and self.prediction_occlusion_tolerance_m > 0.0
            and self.minimum_cells_per_reserved_view == 9
            and self.minimum_views_per_evaluation_frame == 1
            and self.distance_chunk_size >= 1,
            "v6.1 source score configuration changed",
        )


@dataclass(frozen=True, slots=True)
class _ScoreCell:
    camera_id: str
    frame: int
    local_index: int
    target_points_m: np.ndarray
    common_node_indices: np.ndarray


def _target_pixel_indices(
    valid: np.ndarray,
    *,
    object_id: str,
    camera_id: str,
    frame: int,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.nonzero(valid)
    if len(rows) <= maximum:
        return rows, columns
    prefix = (
        object_id.encode("utf-8")
        + b"\0"
        + camera_id.encode("utf-8")
        + b"\0"
        + str(frame).encode("ascii")
        + b"\0"
    )
    ranked = sorted(
        range(len(rows)),
        key=lambda index: hashlib.sha256(
            prefix
            + str(int(rows[index])).encode("ascii")
            + b"\0"
            + str(int(columns[index])).encode("ascii")
            + b"\0"
            + TARGET_PIXEL_RANKING_DOMAIN
        ).digest(),
    )[:maximum]
    selected = np.asarray(sorted(ranked), dtype=np.int64)
    return rows[selected], columns[selected]


def _backproject(
    depth: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    z = np.asarray(depth[rows, columns], dtype=np.float64)
    x = (columns.astype(np.float64) - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (rows.astype(np.float64) - intrinsics[1, 2]) * z / intrinsics[1, 1]
    camera = np.column_stack((x, y, z))
    return camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]


def _visible_node_indices(
    points_world_m: np.ndarray,
    *,
    depth: np.ndarray,
    valid_target: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    config: Deform360V61SourceScoreConfig,
) -> np.ndarray:
    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]
    camera = (points_world_m - translation) @ rotation
    z = camera[:, 2]
    depth_valid = (
        np.all(np.isfinite(camera), axis=1)
        & (z >= config.minimum_depth_m)
        & (z <= config.maximum_depth_m)
    )
    safe_z = np.where(depth_valid, z, 1.0)
    columns = np.floor(
        intrinsics[0, 0] * camera[:, 0] / safe_z + intrinsics[0, 2] + 0.5
    ).astype(np.int64)
    rows = np.floor(
        intrinsics[1, 1] * camera[:, 1] / safe_z + intrinsics[1, 2] + 0.5
    ).astype(np.int64)
    height, width = depth.shape
    inside = (
        depth_valid & (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
    )
    indices = np.nonzero(inside)[0]
    if not len(indices):
        return np.zeros(0, dtype=np.int64)
    supported = valid_target[rows[indices], columns[indices]].copy()
    supported &= (
        z[indices]
        <= depth[rows[indices], columns[indices]]
        + config.prediction_occlusion_tolerance_m
    )
    return np.asarray(indices[supported], dtype=np.int64)


def _directed_distances(
    source: np.ndarray, target: np.ndarray, *, chunk_size: int
) -> np.ndarray:
    distances: list[np.ndarray] = []
    for start in range(0, len(source), chunk_size):
        block = source[start : start + chunk_size]
        squared = np.sum(np.square(block[:, None, :] - target[None, :, :]), axis=2)
        distances.append(np.sqrt(np.min(squared, axis=1)))
    return np.concatenate(distances)


def _nearest_target_points(
    source: np.ndarray, target: np.ndarray, *, chunk_size: int
) -> np.ndarray:
    nearest: list[np.ndarray] = []
    for start in range(0, len(source), chunk_size):
        block = source[start : start + chunk_size]
        squared = np.sum(np.square(block[:, None, :] - target[None, :, :]), axis=2)
        nearest.append(target[np.argmin(squared, axis=1)])
    return np.concatenate(nearest, axis=0)


def _symmetric_chamfer_mm(
    first: np.ndarray, second: np.ndarray, *, chunk_size: int
) -> float:
    return 500.0 * float(
        np.mean(_directed_distances(first, second, chunk_size=chunk_size))
        + np.mean(_directed_distances(second, first, chunk_size=chunk_size))
    )


def _prepare_score_cells(
    *,
    object_id: str,
    episode_id: int,
    b0_trajectory_m: np.ndarray,
    reserved_views: Sequence[Deform360ReservedViewGeometryV5],
    config: Deform360V61SourceScoreConfig,
) -> tuple[list[_ScoreCell], dict[str, Any]]:
    _require(len(reserved_views) == 2, "exactly two reserved views are required")
    expected_frames: np.ndarray = np.arange(*EVALUATION_RANGE, dtype=np.int64)
    cameras = [view.camera_id for view in reserved_views]
    _require(len(set(cameras)) == 2, "reserved camera IDs repeat")
    cells: list[_ScoreCell] = []
    target_counts: dict[str, int] = {camera: 0 for camera in cameras}
    for view in reserved_views:
        _require(
            view.object_id == object_id
            and view.episode_id == episode_id
            and np.array_equal(view.frame_indices, expected_frames),
            "reserved endpoint identity or frame roster changed",
        )
        for local_index, frame in enumerate(expected_frames):
            depth = np.asarray(view.depth_m[local_index], dtype=np.float64)
            valid = (
                np.asarray(view.object_mask[local_index], dtype=bool)
                & np.isfinite(depth)
                & (depth >= config.minimum_depth_m)
                & (depth <= config.maximum_depth_m)
            )
            rows, columns = _target_pixel_indices(
                valid,
                object_id=object_id,
                camera_id=view.camera_id,
                frame=int(frame),
                maximum=config.maximum_target_points,
            )
            if len(rows) < config.minimum_target_points:
                continue
            target = _backproject(
                depth,
                rows,
                columns,
                view.intrinsics,
                view.camera_to_world,
            )
            common = _visible_node_indices(
                np.asarray(b0_trajectory_m[int(frame)], dtype=np.float64),
                depth=depth,
                valid_target=valid,
                intrinsics=view.intrinsics,
                camera_to_world=view.camera_to_world,
                config=config,
            )
            if not len(common):
                continue
            cells.append(
                _ScoreCell(
                    camera_id=view.camera_id,
                    frame=int(frame),
                    local_index=local_index,
                    target_points_m=target,
                    common_node_indices=common,
                )
            )
            target_counts[view.camera_id] += 1
    frame_counts = Counter(cell.frame for cell in cells)
    if any(
        target_counts[camera] < config.minimum_cells_per_reserved_view
        for camera in cameras
    ) or any(
        frame_counts[int(frame)] < config.minimum_views_per_evaluation_frame
        for frame in expected_frames
    ):
        raise SourceEndpointSupportError(
            "source endpoint failed the frozen time-spanning support contract"
        )
    roster_id = content_id(
        {
            "schema": ("bayesian-phystwin.deform360-v6-source-common-query-roster"),
            "schema_version": 1,
            "object_id": object_id,
            "episode_id": episode_id,
            "cells": [
                {
                    "camera_id": cell.camera_id,
                    "frame": cell.frame,
                    "common_node_indices": cell.common_node_indices.tolist(),
                }
                for cell in cells
            ],
        }
    )
    return cells, {
        "admitted_cell_count": len(cells),
        "admitted_cell_count_by_reserved_view": dict(sorted(target_counts.items())),
        "minimum_admitted_views_per_frame": min(frame_counts.values()),
        "common_query_count": int(sum(len(cell.common_node_indices) for cell in cells)),
        "common_query_roster_id": roster_id,
    }


def _score_variant(
    *,
    trajectory_m: np.ndarray,
    covariance_m2: np.ndarray,
    cells: Sequence[_ScoreCell],
    config: Deform360V61SourceScoreConfig,
) -> dict[str, Any]:
    point_losses: list[float] = []
    mahalanobis_squared: list[np.ndarray] = []
    log_determinants: list[np.ndarray] = []
    radii: list[np.ndarray] = []
    query_count = 0
    for cell in cells:
        indices = cell.common_node_indices
        candidate = np.asarray(trajectory_m[cell.frame, indices], dtype=np.float64)
        target = cell.target_points_m
        point_losses.append(
            _symmetric_chamfer_mm(
                candidate,
                target,
                chunk_size=config.distance_chunk_size,
            )
        )
        nearest = _nearest_target_points(
            candidate,
            target,
            chunk_size=config.distance_chunk_size,
        )
        residual = candidate - nearest
        covariance = np.asarray(
            covariance_m2[cell.local_index, indices], dtype=np.float64
        )
        eigenvalues = np.linalg.eigvalsh(covariance)
        _require(
            np.all(np.isfinite(eigenvalues)) and np.min(eigenvalues) > 0.0,
            "candidate covariance is not positive definite",
        )
        solved = np.linalg.solve(covariance, residual[..., None])[..., 0]
        mahalanobis_squared.append(np.einsum("ni,ni->n", residual, solved))
        log_determinants.append(np.sum(np.log(eigenvalues), axis=1))
        radii.append(np.sqrt(np.max(eigenvalues, axis=1)))
        query_count += len(indices)
    mahalanobis = np.concatenate(mahalanobis_squared)
    logdet = np.concatenate(log_determinants)
    radius = np.concatenate(radii)
    return {
        "query_count": query_count,
        "point_loss": float(np.mean(point_losses)),
        "mean_raw_mahalanobis_squared": float(np.mean(mahalanobis)),
        "mean_log_determinant": float(np.mean(logdet)),
        "maximum_raw_mahalanobis_norm": float(np.sqrt(np.max(mahalanobis))),
        "mean_raw_radius": float(np.mean(radius)),
    }


def score_deform360_v61_candidate_artifact(
    *,
    prediction: Mapping[str, Any],
    candidate_directory: str | Path,
    reserved_views: Sequence[Deform360ReservedViewGeometryV5],
    config: Deform360V61SourceScoreConfig | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Score one sealed candidate on a common B0-defined query roster."""

    cfg = config or Deform360V61SourceScoreConfig()
    seal, candidate = load_deform360_v61_candidate_artifact(candidate_directory)
    object_id = _identifier(prediction.get("object_id"), name="object_id")
    episode_id_raw = prediction.get("episode_id")
    _require(
        type(episode_id_raw) is int
        and seal.get("object_id") == object_id
        and seal.get("episode_id") == episode_id_raw
        and seal.get("outer_held_out_object_id")
        == prediction.get("outer_held_out_object_id")
        and seal.get("candidate_revision") == CANDIDATE_REVISION,
        "candidate artifact differs from its raw prediction",
    )
    episode_id = cast(int, episode_id_raw)
    prediction_variants = _mapping(prediction.get("variants"), name="variants")
    seal_variants = _mapping(seal.get("variant_artifacts"), name="variants")
    for variant_id in VARIANT_IDS:
        _require(
            _mapping(prediction_variants[variant_id], name=variant_id).get(
                "prediction_artifact_id"
            )
            == _mapping(seal_variants[variant_id], name=variant_id).get(
                "prediction_artifact_id"
            ),
            "candidate prediction identity changed",
        )
    cells, support = _prepare_score_cells(
        object_id=object_id,
        episode_id=episode_id,
        b0_trajectory_m=candidate.arrays[f"trajectory__{B0}"],
        reserved_views=reserved_views,
        config=cfg,
    )
    scored: dict[str, dict[str, Any]] = {}
    for variant_id in CANDIDATE_IDS:
        raw = _score_variant(
            trajectory_m=candidate.arrays[f"trajectory__{variant_id}"],
            covariance_m2=candidate.arrays[f"covariance__{variant_id}"],
            cells=cells,
            config=cfg,
        )
        scored[variant_id] = {
            "available": True,
            "prediction_artifact_id": _mapping(
                prediction_variants[variant_id], name=variant_id
            )["prediction_artifact_id"],
            **raw,
        }
    fallback = {
        key: value
        for key, value in scored[B0].items()
        if key not in {"available", "prediction_artifact_id"}
    }
    for variant_id in (VT1_WORKING, VT1_OBSERVED, VT1_SANDWICH):
        prediction_row = _mapping(prediction_variants[variant_id], name=variant_id)
        _require(
            prediction_row.get("available") is False
            and prediction_row.get("prediction_artifact_id") is None,
            "public VT1 carrier unexpectedly became available",
        )
        scored[variant_id] = {
            "available": False,
            "prediction_artifact_id": None,
            **fallback,
        }
    return scored, support


def _file_record(value: object, *, name: str) -> dict[str, str]:
    record = _mapping(value, name=name)
    require_exact_fields(record, expected=_ENDPOINT_FILE_FIELDS, name=name)
    return {
        "path": canonical_relative_posix_path(record.get("path"), name=f"{name}.path"),
        "sha256": sha256_digest(record.get("sha256"), name=f"{name}.sha256"),
    }


def build_deform360_v61_source_endpoint_manifest(
    *,
    authorization: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    processor_revision: str,
    objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind the post-authorization public endpoint archives."""

    auth = validate_deform360_v61_source_suffix_authorization(authorization)
    plan = validate_deform360_v61_source_plan(source_plan)
    source_objects = {
        cast(str, row["object_id"]): row
        for row in cast(Sequence[Mapping[str, Any]], plan["objects"])
    }
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(objects):
        row = _mapping(raw, name=f"endpoint object {index}")
        require_exact_fields(
            row, expected=_ENDPOINT_OBJECT_FIELDS, name=f"endpoint object {index}"
        )
        object_id = _identifier(row.get("object_id"), name="object_id")
        _require(object_id in source_objects, "endpoint object is outside source plan")
        source = source_objects[object_id]
        prefix = cast(Sequence[int], source["raw_prefix_range_half_open"])
        expected_raw = [int(prefix[1]), int(prefix[1]) + 18]
        cameras = list(cast(Sequence[str], source["all_camera_ids"]))
        _require(
            row.get("episode_id") == source["episode_id"]
            and row.get("stratum") == source["stratum"]
            and row.get("all_camera_ids") == cameras
            and row.get("raw_endpoint_range_half_open") == expected_raw,
            "endpoint object identity changed",
        )
        views: list[dict[str, Any]] = []
        for raw_view in _sequence(row.get("reserved_views"), name="reserved_views"):
            view = _mapping(raw_view, name="reserved view")
            require_exact_fields(
                view, expected=_ENDPOINT_VIEW_FIELDS, name="reserved view"
            )
            views.append(
                {
                    "camera_id": _identifier(view.get("camera_id"), name="camera_id"),
                    "endpoint_archive": _file_record(
                        view.get("endpoint_archive"), name="endpoint archive"
                    ),
                }
            )
        views.sort(key=lambda item: cast(str, item["camera_id"]))
        _require(
            [item["camera_id"] for item in views]
            == sorted(source["reserved_endpoint_camera_ids"]),
            "endpoint reserved camera roster changed",
        )
        normalized.append(
            {
                "object_id": object_id,
                "episode_id": source["episode_id"],
                "stratum": source["stratum"],
                "all_camera_ids": cameras,
                "raw_endpoint_range_half_open": expected_raw,
                "reserved_views": views,
            }
        )
    normalized.sort(key=lambda item: cast(str, item["object_id"]))
    _require(
        [item["object_id"] for item in normalized] == sorted(source_objects),
        "endpoint manifest differs from the ten-unit source cohort",
    )
    identity: dict[str, Any] = {
        "schema": SOURCE_ENDPOINT_MANIFEST_SCHEMA,
        "schema_version": SOURCE_ENDPOINT_MANIFEST_VERSION,
        "source_scoring_amendment_id": SOURCE_SCORING_AMENDMENT_ID,
        "authorization_id": auth["authorization_id"],
        "upstream_source_plan_id": plan["plan_id"],
        "processor_revision": exact_revision(
            processor_revision, name="processor_revision"
        ),
        "objects": normalized,
        "information_boundary": dict(_ENDPOINT_BOUNDARY),
    }
    return {**identity, "manifest_id": content_id(identity)}


def validate_deform360_v61_source_endpoint_manifest(
    value: object,
    *,
    authorization: Mapping[str, Any],
    source_plan: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = _mapping(value, name="endpoint manifest")
    require_exact_fields(
        manifest,
        expected=_ENDPOINT_MANIFEST_FIELDS,
        name="endpoint manifest",
    )
    _require(
        manifest.get("schema") == SOURCE_ENDPOINT_MANIFEST_SCHEMA
        and manifest.get("schema_version") == SOURCE_ENDPOINT_MANIFEST_VERSION
        and manifest.get("source_scoring_amendment_id") == SOURCE_SCORING_AMENDMENT_ID
        and manifest.get("information_boundary") == _ENDPOINT_BOUNDARY,
        "endpoint manifest contract changed",
    )
    rebuilt = build_deform360_v61_source_endpoint_manifest(
        authorization=authorization,
        source_plan=source_plan,
        processor_revision=cast(str, manifest.get("processor_revision")),
        objects=cast(Sequence[Mapping[str, Any]], manifest.get("objects")),
    )
    _require(plain_json(manifest) == rebuilt, "endpoint manifest identity changed")
    return rebuilt


def _verified_file(root: Path, record: Mapping[str, Any], *, name: str) -> Path:
    relative = canonical_relative_posix_path(record.get("path"), name=f"{name}.path")
    canonical_root = root.resolve(strict=True)
    path = (canonical_root / relative).absolute()
    resolved = path.resolve(strict=True)
    _require(
        canonical_root in resolved.parents
        and path.is_file()
        and not path.is_symlink()
        and not any(
            parent.is_symlink()
            for parent in path.parents
            if parent != canonical_root and canonical_root in parent.parents
        )
        and _sha256_file(path) == record.get("sha256"),
        f"{name} bytes changed",
    )
    return resolved


def load_deform360_v61_source_endpoint_view(
    path: str | Path,
    *,
    object_id: str,
    episode_id: int,
    camera_id: str,
    raw_endpoint_range_half_open: tuple[int, int],
    source_artifact_ids: Mapping[str, str],
) -> Deform360ReservedViewGeometryV5:
    """Load one partial-support endpoint archive without filling empty frames."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            _require(
                set(archive.files) == ENDPOINT_ARCHIVE_MEMBERS,
                "endpoint archive member roster changed",
            )
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as error:
        raise ValueError("cannot load source endpoint archive") from error
    raw_start, raw_stop = raw_endpoint_range_half_open
    _require(
        np.array_equal(
            arrays["raw_frame_indices"],
            np.arange(raw_start, raw_stop, dtype=np.int64),
        ),
        "endpoint raw frame roster changed",
    )
    return Deform360ReservedViewGeometryV5(
        object_id=object_id,
        episode_id=episode_id,
        camera_id=camera_id,
        frame_indices=arrays["frame_indices"],
        depth_m=arrays["depth_m"],
        object_mask=arrays["object_mask"],
        intrinsics=arrays["intrinsics"],
        camera_to_world=arrays["camera_to_world"],
        source_artifact_ids=source_artifact_ids,
    )


def _endpoint_views_by_object(
    *,
    endpoint_manifest: Mapping[str, Any],
    endpoint_root: Path,
) -> tuple[
    dict[str, tuple[Deform360ReservedViewGeometryV5, ...]],
    dict[str, dict[str, str]],
]:
    views_by_object: dict[str, tuple[Deform360ReservedViewGeometryV5, ...]] = {}
    sources_by_object: dict[str, dict[str, str]] = {}
    for raw in cast(Sequence[Mapping[str, Any]], endpoint_manifest["objects"]):
        object_id = cast(str, raw["object_id"])
        episode_id = cast(int, raw["episode_id"])
        raw_range_values = cast(Sequence[int], raw["raw_endpoint_range_half_open"])
        raw_range = (int(raw_range_values[0]), int(raw_range_values[1]))
        views: list[Deform360ReservedViewGeometryV5] = []
        sources: dict[str, str] = {}
        for raw_view in cast(Sequence[Mapping[str, Any]], raw["reserved_views"]):
            camera_id = cast(str, raw_view["camera_id"])
            record = cast(Mapping[str, Any], raw_view["endpoint_archive"])
            path = _verified_file(
                endpoint_root,
                record,
                name=f"endpoint archive {object_id}/{camera_id}",
            )
            key = f"endpoint/{object_id}/{camera_id}.npz"
            sources[key] = _sha256_file(path)
            views.append(
                load_deform360_v61_source_endpoint_view(
                    path,
                    object_id=object_id,
                    episode_id=episode_id,
                    camera_id=camera_id,
                    raw_endpoint_range_half_open=raw_range,
                    source_artifact_ids={key: sources[key]},
                )
            )
        views_by_object[object_id] = tuple(views)
        sources_by_object[object_id] = dict(
            source_artifact_mapping(sources, name=f"endpoint sources for {object_id}")
        )
    return views_by_object, sources_by_object


def _candidate_directory(
    candidate_root: Path,
    *,
    ordered_ids: Sequence[str],
    outer_id: str,
    target_id: str,
) -> Path:
    return (
        candidate_root
        / "candidate-artifacts"
        / f"{ordered_ids.index(outer_id):02d}-{outer_id}"
        / f"{ordered_ids.index(target_id):02d}-{target_id}"
    )


def _support_report(
    *,
    prediction: Mapping[str, Any],
    support: Mapping[str, Any],
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "schema": "bayesian-phystwin.deform360-v6-source-score-support-report",
        "schema_version": 1,
        "source_scoring_amendment_id": SOURCE_SCORING_AMENDMENT_ID,
        "prediction_record_id": prediction["prediction_record_id"],
        "outer_held_out_object_id": prediction["outer_held_out_object_id"],
        "object_id": prediction["object_id"],
        "episode_id": prediction["episode_id"],
        "common_query_roster": dict(support),
        "candidate_dependent_missing_queries_allowed": False,
        "reserved_views_used_for_prediction": False,
    }
    return {**identity, "report_id": content_id(identity)}


def build_deform360_v61_source_scoring_receipt(
    *,
    scorer_revision: str,
    authorization: Mapping[str, Any],
    endpoint_manifest: Mapping[str, Any],
    result: Mapping[str, Any],
    outcome_count: int,
    artifacts: Mapping[str, str],
) -> dict[str, Any]:
    """Build a compact terminal decision without authorizing confirmation."""

    auth = validate_deform360_v61_source_suffix_authorization(authorization)
    _require(
        type(outcome_count) is int and outcome_count == 100,
        "source scoring must contain exactly 100 outcomes",
    )
    passed = result.get("source_gate_passed")
    _require(
        type(passed) is bool
        and result.get("source_continuation_authorized") is passed
        and result.get("claim_authorized") is False,
        "source result authorization changed",
    )
    identity: dict[str, Any] = {
        "schema": SOURCE_SCORING_RECEIPT_SCHEMA,
        "schema_version": SOURCE_SCORING_RECEIPT_VERSION,
        "status": "source-challenger-advanced"
        if passed
        else "source-reference-retained",
        "source_scoring_amendment_id": SOURCE_SCORING_AMENDMENT_ID,
        "scorer_revision": exact_revision(scorer_revision, name="scorer_revision"),
        "runner_name": auth["runner_name"],
        "workflow_run_id": auth["workflow_run_id"],
        "workflow_run_attempt": auth["workflow_run_attempt"],
        "authorization_id": auth["authorization_id"],
        "endpoint_manifest_id": endpoint_manifest["manifest_id"],
        "candidate_panel_receipt_id": CANDIDATE_PANEL_RECEIPT_ID,
        "raw_prediction_batch_id": RAW_PREDICTION_BATCH_ID,
        "prediction_record_count": 100,
        "outcome_count": outcome_count,
        "evidence_id": result["evidence_id"],
        "result_id": result["result_id"],
        "source_gate_passed": passed,
        "source_continuation_authorized": passed,
        "independent_confirmation_authorized": False,
        "claim_authorized": False,
        "artifacts": dict(
            source_artifact_mapping(artifacts, name="source scoring artifacts")
        ),
        "information_boundary": dict(_SCORING_BOUNDARY),
    }
    return {**identity, "receipt_id": content_id(identity)}


def validate_deform360_v61_source_scoring_receipt(
    value: object,
) -> dict[str, Any]:
    receipt = _mapping(value, name="source scoring receipt")
    require_exact_fields(
        receipt,
        expected=_SCORING_RECEIPT_FIELDS,
        name="source scoring receipt",
    )
    _content_identity(receipt, field="receipt_id", name="source scoring receipt")
    passed = receipt.get("source_gate_passed")
    _require(
        receipt.get("schema") == SOURCE_SCORING_RECEIPT_SCHEMA
        and receipt.get("schema_version") == SOURCE_SCORING_RECEIPT_VERSION
        and receipt.get("source_scoring_amendment_id") == SOURCE_SCORING_AMENDMENT_ID
        and receipt.get("candidate_panel_receipt_id") == CANDIDATE_PANEL_RECEIPT_ID
        and receipt.get("raw_prediction_batch_id") == RAW_PREDICTION_BATCH_ID
        and receipt.get("prediction_record_count") == 100
        and receipt.get("outcome_count") == 100
        and type(passed) is bool
        and receipt.get("source_continuation_authorized") is passed
        and receipt.get("independent_confirmation_authorized") is False
        and receipt.get("claim_authorized") is False
        and receipt.get("information_boundary") == _SCORING_BOUNDARY,
        "source scoring receipt contract changed",
    )
    exact_revision(receipt.get("scorer_revision"), name="scorer_revision")
    _identifier(receipt.get("runner_name"), name="runner name")
    genuine_integer(receipt.get("workflow_run_id"), name="workflow run ID", minimum=1)
    genuine_integer(
        receipt.get("workflow_run_attempt"),
        name="workflow run attempt",
        minimum=1,
    )
    for field in (
        "authorization_id",
        "endpoint_manifest_id",
        "evidence_id",
        "result_id",
    ):
        sha256_digest(receipt.get(field), name=field)
    source_artifact_mapping(
        cast(Mapping[str, str], receipt.get("artifacts")),
        name="source scoring artifacts",
    )
    return cast(dict[str, Any], plain_json(receipt))


def build_deform360_v61_source_scoring_technical_failure_receipt(
    *,
    scorer_revision: str,
    runner_name: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    authorization: Mapping[str, Any],
    terminal_stage: str,
    exit_code: int,
    source_suffix_opened: bool,
    retained_artifact_file_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Retain one post-barrier failure without evaluating the source gate."""

    auth = validate_deform360_v61_source_suffix_authorization(authorization)
    _require(
        type(source_suffix_opened) is bool,
        "source suffix opened flag must be boolean",
    )
    identity: dict[str, Any] = {
        "schema": SOURCE_SCORING_TECHNICAL_FAILURE_RECEIPT_SCHEMA,
        "schema_version": SOURCE_SCORING_TECHNICAL_FAILURE_RECEIPT_VERSION,
        "status": "source-scoring-technical-failure-retained",
        "source_scoring_amendment_id": SOURCE_SCORING_AMENDMENT_ID,
        "scorer_revision": exact_revision(scorer_revision, name="scorer revision"),
        "runner_name": _identifier(runner_name, name="runner name"),
        "workflow_run_id": genuine_integer(
            workflow_run_id, name="workflow run ID", minimum=1
        ),
        "workflow_run_attempt": genuine_integer(
            workflow_run_attempt, name="workflow run attempt", minimum=1
        ),
        "candidate_panel_receipt_id": CANDIDATE_PANEL_RECEIPT_ID,
        "raw_prediction_batch_id": RAW_PREDICTION_BATCH_ID,
        "authorization_id": auth["authorization_id"],
        "terminal_stage": _identifier(terminal_stage, name="terminal stage"),
        "exit_code": genuine_integer(exit_code, name="exit code", minimum=1),
        "source_suffix_access_authorized": True,
        "source_suffix_opened": source_suffix_opened,
        "source_gate_evaluated": False,
        "source_gate_passed": None,
        "source_continuation_authorized": False,
        "independent_confirmation_authorized": False,
        "claim_authorized": False,
        "retained_artifacts": dict(
            source_artifact_mapping(
                retained_artifact_file_sha256,
                name="retained source-scoring artifacts",
                allow_empty=True,
            )
        ),
        "information_boundary": _technical_failure_boundary(
            source_suffix_opened=source_suffix_opened
        ),
    }
    return validate_deform360_v61_source_scoring_technical_failure_receipt(
        {**identity, "receipt_id": content_id(identity)}
    )


def validate_deform360_v61_source_scoring_technical_failure_receipt(
    value: object,
) -> dict[str, Any]:
    """Validate a terminal source-scoring failure as distinct from a negative."""

    receipt = _mapping(value, name="source-scoring technical-failure receipt")
    require_exact_fields(
        receipt,
        expected=_SCORING_TECHNICAL_FAILURE_RECEIPT_FIELDS,
        name="source-scoring technical-failure receipt",
    )
    _content_identity(
        receipt,
        field="receipt_id",
        name="source-scoring technical-failure receipt",
    )
    suffix_opened = receipt.get("source_suffix_opened")
    _require(
        receipt.get("schema") == SOURCE_SCORING_TECHNICAL_FAILURE_RECEIPT_SCHEMA
        and receipt.get("schema_version")
        == SOURCE_SCORING_TECHNICAL_FAILURE_RECEIPT_VERSION
        and receipt.get("status") == "source-scoring-technical-failure-retained"
        and receipt.get("source_scoring_amendment_id") == SOURCE_SCORING_AMENDMENT_ID
        and receipt.get("candidate_panel_receipt_id") == CANDIDATE_PANEL_RECEIPT_ID
        and receipt.get("raw_prediction_batch_id") == RAW_PREDICTION_BATCH_ID
        and receipt.get("source_suffix_access_authorized") is True
        and type(suffix_opened) is bool
        and receipt.get("source_gate_evaluated") is False
        and receipt.get("source_gate_passed") is None
        and receipt.get("source_continuation_authorized") is False
        and receipt.get("independent_confirmation_authorized") is False
        and receipt.get("claim_authorized") is False
        and receipt.get("information_boundary")
        == _technical_failure_boundary(source_suffix_opened=bool(suffix_opened)),
        "source-scoring technical-failure contract changed",
    )
    exact_revision(receipt.get("scorer_revision"), name="scorer revision")
    _identifier(receipt.get("runner_name"), name="runner name")
    _identifier(receipt.get("terminal_stage"), name="terminal stage")
    genuine_integer(receipt.get("workflow_run_id"), name="workflow run ID", minimum=1)
    genuine_integer(
        receipt.get("workflow_run_attempt"),
        name="workflow run attempt",
        minimum=1,
    )
    genuine_integer(receipt.get("exit_code"), name="exit code", minimum=1)
    sha256_digest(receipt.get("authorization_id"), name="authorization ID")
    source_artifact_mapping(
        cast(Mapping[str, str], receipt.get("retained_artifacts")),
        name="retained source-scoring artifacts",
        allow_empty=True,
    )
    return cast(dict[str, Any], plain_json(receipt))


def retain_deform360_v61_source_scoring_failure(
    *,
    scorer_revision: str,
    runner_name: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    authorization_path: str | Path,
    terminal_stage: str,
    exit_code: int,
    source_suffix_opened: bool,
    artifact_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Hash and retain one bounded post-authorization technical failure."""

    authorization = validate_deform360_v61_source_suffix_authorization(
        load_strict_json_object(
            authorization_path,
            label="source suffix authorization",
        )
    )
    root = Path(artifact_root).resolve(strict=True)
    _require(
        root.is_dir()
        and not root.is_symlink()
        and not any(parent.is_symlink() for parent in root.parents),
        "failure artifact root is invalid",
    )
    destination = Path(output_path).absolute()
    retained: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        _require(
            not path.is_symlink()
            and not any(parent.is_symlink() for parent in path.parents),
            "retained source-scoring artifacts contain a symlink",
        )
        if path.is_file() and path.absolute() != destination:
            retained[path.relative_to(root).as_posix()] = _sha256_file(path)
    receipt = build_deform360_v61_source_scoring_technical_failure_receipt(
        scorer_revision=scorer_revision,
        runner_name=runner_name,
        workflow_run_id=workflow_run_id,
        workflow_run_attempt=workflow_run_attempt,
        authorization=authorization,
        terminal_stage=terminal_stage,
        exit_code=exit_code,
        source_suffix_opened=source_suffix_opened,
        retained_artifact_file_sha256=retained,
    )
    return _publish_or_validate_json(
        receipt,
        destination,
        label="source-scoring technical-failure receipt",
    )


def publish_deform360_v61_source_scores(
    *,
    source_scoring_amendment_path: str | Path,
    execution_lock_path: str | Path,
    candidate_root: str | Path,
    candidate_execution_receipt_path: str | Path,
    upstream_source_plan_path: str | Path,
    authorization_path: str | Path,
    endpoint_manifest_path: str | Path,
    endpoint_root: str | Path,
    output_root: str | Path,
    scorer_revision: str,
    runner_name: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
) -> dict[str, Any]:
    """Score exactly one authorized ten-by-ten public-source panel."""

    load_deform360_v61_source_scoring_amendment(source_scoring_amendment_path)
    lock = load_deform360_joint_sparse_source_execution_lock_v5(execution_lock_path)
    cohort = _cohort(lock)
    ordered_ids = tuple(sorted(cohort))
    _require(len(ordered_ids) == 10, "source scoring cohort changed")
    candidate = Path(candidate_root).resolve(strict=True)
    panel = validate_deform360_v61_candidate_panel(
        execution_lock_path=execution_lock_path,
        output_root=candidate,
    )
    _require(panel["receipt_id"] == CANDIDATE_PANEL_RECEIPT_ID, "panel changed")
    batch_path = candidate / CANDIDATE_BATCH_FILENAME
    batch = validate_deform360_v6_raw_nested_batch(
        load_strict_json_object(batch_path, label="candidate raw batch"),
        cohort=cohort,
    )
    _require(batch["prediction_batch_id"] == RAW_PREDICTION_BATCH_ID, "batch changed")
    expected_authorization = build_deform360_v61_source_suffix_authorization(
        source_scoring_amendment_path=source_scoring_amendment_path,
        execution_lock_path=execution_lock_path,
        candidate_root=candidate,
        candidate_execution_receipt_path=candidate_execution_receipt_path,
        upstream_source_plan_path=upstream_source_plan_path,
        scorer_revision=scorer_revision,
        runner_name=runner_name,
        workflow_run_id=workflow_run_id,
        workflow_run_attempt=workflow_run_attempt,
    )
    authorization_source = Path(authorization_path)
    authorization = validate_deform360_v61_source_suffix_authorization(
        load_strict_json_object(
            authorization_source,
            label="source suffix authorization",
        )
    )
    _require(
        authorization == expected_authorization,
        "published source suffix authorization differs from replay",
    )

    # The endpoint manifest is the first input that names suffix-derived files.
    source_plan = _load_exact_source_plan(upstream_source_plan_path)
    endpoint_manifest_source = Path(endpoint_manifest_path)
    endpoint_manifest = validate_deform360_v61_source_endpoint_manifest(
        load_strict_json_object(
            endpoint_manifest_source,
            label="source endpoint manifest",
        ),
        authorization=authorization,
        source_plan=source_plan,
    )
    views_by_object, endpoint_sources = _endpoint_views_by_object(
        endpoint_manifest=endpoint_manifest,
        endpoint_root=Path(endpoint_root),
    )
    output = Path(output_root).absolute()
    output.mkdir(parents=True, exist_ok=True)
    _require(
        output.is_dir()
        and not output.is_symlink()
        and not any(parent.is_symlink() for parent in output.parents),
        "source scoring output root is invalid",
    )
    outcome_root = output / "source-outcomes"
    support_root = output / "support-reports"
    outcome_root.mkdir(parents=True, exist_ok=True)
    support_root.mkdir(parents=True, exist_ok=True)
    outcomes: list[dict[str, Any]] = []
    for index, prediction in enumerate(
        cast(Sequence[Mapping[str, Any]], batch["records"])
    ):
        outer_id = cast(str, prediction["outer_held_out_object_id"])
        target_id = cast(str, prediction["object_id"])
        key = f"{ordered_ids.index(outer_id):02d}-{ordered_ids.index(target_id):02d}"
        directory = _candidate_directory(
            candidate,
            ordered_ids=ordered_ids,
            outer_id=outer_id,
            target_id=target_id,
        )
        variants, support = score_deform360_v61_candidate_artifact(
            prediction=prediction,
            candidate_directory=directory,
            reserved_views=views_by_object[target_id],
        )
        support_report = _support_report(prediction=prediction, support=support)
        support_path = support_root / f"{key}.json"
        _publish_or_validate_json(
            support_report,
            support_path,
            label=f"source support report {key}",
        )
        scoring_artifacts = {
            **endpoint_sources[target_id],
            f"candidate/{key}/{CANDIDATE_SEAL_FILENAME}": _sha256_file(
                directory / CANDIDATE_SEAL_FILENAME
            ),
            "source-suffix-opening-authorization.json": _sha256_file(
                authorization_source
            ),
            "source-endpoint-manifest.json": _sha256_file(endpoint_manifest_source),
            f"support-reports/{key}.json": _sha256_file(support_path),
        }
        outcome = build_deform360_v6_raw_nested_outcome(
            prediction_batch=batch,
            prediction_record_id=cast(str, prediction["prediction_record_id"]),
            variants=variants,
            scoring_artifacts=scoring_artifacts,
        )
        _publish_or_validate_json(
            outcome,
            outcome_root / f"{index:03d}.json",
            label=f"source outcome {index:03d}",
        )
        outcomes.append(outcome)
    evidence = assemble_deform360_v6_nested_evidence(
        prediction_batch=batch,
        outcomes=outcomes,
        cohort=cohort,
    )
    evidence_path = output / "source-evidence.json"
    if evidence_path.exists():
        _publish_or_validate_json(evidence, evidence_path, label="source evidence")
    else:
        publish_deform360_v6_nested_evidence(evidence, evidence_path, cohort=cohort)
    result = evaluate_deform360_v6_nested_source_gate(evidence, cohort=cohort)
    result_path = output / "source-result.json"
    if result_path.exists():
        _publish_or_validate_json(result, result_path, label="source result")
    else:
        publish_deform360_v6_nested_result(
            result,
            result_path,
            evidence=evidence,
            cohort=cohort,
        )
    receipt = build_deform360_v61_source_scoring_receipt(
        scorer_revision=scorer_revision,
        authorization=authorization,
        endpoint_manifest=endpoint_manifest,
        result=result,
        outcome_count=len(outcomes),
        artifacts={
            "source_scoring_amendment": _sha256_file(source_scoring_amendment_path),
            "candidate_execution_receipt": _sha256_file(
                candidate_execution_receipt_path
            ),
            "candidate_panel_receipt": _sha256_file(
                candidate / CANDIDATE_RECEIPT_FILENAME
            ),
            "candidate_raw_batch": _sha256_file(batch_path),
            "upstream_source_plan": _sha256_file(upstream_source_plan_path),
            "source_suffix_authorization": _sha256_file(authorization_source),
            "source_endpoint_manifest": _sha256_file(endpoint_manifest_source),
            "source_evidence": _sha256_file(evidence_path),
            "source_result": _sha256_file(result_path),
        },
    )
    validated_receipt = validate_deform360_v61_source_scoring_receipt(receipt)
    _publish_or_validate_json(
        validated_receipt,
        output / "source-scoring-receipt.json",
        label="source scoring receipt",
    )
    return validated_receipt


__all__ = [
    "CANDIDATE_EXECUTION_RECEIPT_ID",
    "CANDIDATE_PANEL_RECEIPT_ID",
    "RAW_PREDICTION_BATCH_ID",
    "SOURCE_ENDPOINT_MANIFEST_SCHEMA",
    "SOURCE_SCORING_AMENDMENT_ID",
    "SOURCE_SCORING_AMENDMENT_SCHEMA",
    "SOURCE_SCORING_TECHNICAL_FAILURE_RECEIPT_SCHEMA",
    "SOURCE_SUFFIX_AUTHORIZATION_SCHEMA",
    "Deform360V61SourceScoreConfig",
    "SourceEndpointSupportError",
    "build_deform360_v61_source_endpoint_manifest",
    "build_deform360_v61_source_scoring_receipt",
    "build_deform360_v61_source_scoring_technical_failure_receipt",
    "build_deform360_v61_source_suffix_authorization",
    "load_deform360_v61_source_endpoint_view",
    "load_deform360_v61_source_scoring_amendment",
    "publish_deform360_v61_source_suffix_authorization",
    "publish_deform360_v61_source_scores",
    "score_deform360_v61_candidate_artifact",
    "retain_deform360_v61_source_scoring_failure",
    "validate_deform360_v61_source_endpoint_manifest",
    "validate_deform360_v61_source_plan",
    "validate_deform360_v61_source_scoring_receipt",
    "validate_deform360_v61_source_scoring_technical_failure_receipt",
    "validate_deform360_v61_source_suffix_authorization",
]
