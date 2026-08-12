"""Publish the outcome-closed 10-by-10 Deform360 v6.1 candidate panel."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
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
from .deform360_fresh_object_session_candidate_v6_1 import (
    CANDIDATE_AMENDMENT_FILE_SHA256,
    CANDIDATE_AMENDMENT_ID,
    CANDIDATE_ARCHIVE_FILENAME,
    CANDIDATE_SEAL_FILENAME,
    EXECUTION_LOCK_FILE_SHA256,
    EXECUTION_LOCK_ID,
    UPSTREAM_EXECUTION_RECEIPT_FILE_SHA256,
    UPSTREAM_EXECUTION_RECEIPT_ID,
    UPSTREAM_PREDICTION_BATCH_FILE_SHA256,
    UPSTREAM_PREDICTION_RECEIPT_FILE_SHA256,
    UPSTREAM_PREDICTION_RECEIPT_ID,
    UPSTREAM_SOURCE_PLAN_FILE_SHA256,
    UPSTREAM_SOURCE_PLAN_ID,
    build_deform360_v61_candidate_arrays,
    build_deform360_v61_technical_fallback_arrays,
    load_deform360_v61_candidate_amendment,
    load_deform360_v61_candidate_artifact,
    publish_deform360_v61_candidate_artifact,
    raw_variants_from_deform360_v61_candidate_seal,
)
from .deform360_fresh_object_session_public_inputs_v6_1 import (
    prepare_deform360_disjoint_visual_window_v6_1,
)
from .deform360_fresh_object_session_source_v6_1 import (
    UPSTREAM_PREDICTION_BATCH_ID,
    UPSTREAM_REVISION,
    build_deform360_v6_raw_nested_batch,
    build_deform360_v6_raw_nested_prediction,
    publish_deform360_v6_raw_nested_batch,
    validate_deform360_v6_raw_nested_batch,
    validate_deform360_v6_raw_nested_prediction,
)
from .deform360_joint_sparse_materializer_v5 import (
    Deform360JointSparsePrefixFitV5,
)
from .deform360_joint_sparse_prediction_artifacts_v5 import (
    load_deform360_joint_sparse_prediction_v5,
)
from .deform360_joint_sparse_prediction_v5 import (
    B0_PHYSICAL_FALLBACK,
    B1_LAST_CAUSAL_RESIDUAL,
    RAW_METHOD_IDS,
)
from .deform360_joint_sparse_source_evidence_v5 import (
    validate_deform360_joint_sparse_source_prediction_batch_v5,
    validate_deform360_joint_sparse_source_prediction_seal_v5,
)
from .deform360_joint_sparse_source_gate_v5 import (
    load_deform360_joint_sparse_source_execution_lock_v5,
)
from .deform360_joint_sparse_source_runner_v5 import (
    _cohort,
    _load_physical_archive,
    _ordinary_root,
    _sha256_file,
    _verified_file,
)
from .deform360_joint_sparse_source_runner_v5_2 import (
    validate_deform360_joint_sparse_source_prediction_plan_v5_2,
    validate_deform360_joint_sparse_source_prediction_receipt_v5_2,
)
from .deform360_v6_source_camera_reuse import (
    validate_deform360_v6_source_camera_reuse_execution_receipt,
)

CANDIDATE_PANEL_RECEIPT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-v6-candidate-panel-receipt"
)
CANDIDATE_PANEL_RECEIPT_VERSION: Final = 1
CANDIDATE_BATCH_FILENAME: Final = "raw-nested-prediction-batch.json"
CANDIDATE_RECEIPT_FILENAME: Final = "candidate-panel-receipt.json"
SEALED_VISUAL_PRODUCT_FILENAME: Final = "baseline_disjoint.npz"
CANDIDATE_EXECUTION_RECEIPT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-v6-candidate-execution-receipt"
)
CANDIDATE_EXECUTION_RECEIPT_VERSION: Final = 1
CANDIDATE_TECHNICAL_FAILURE_RECEIPT_SCHEMA: Final = "bayesian-phystwin.deform360-fresh-object-session-v6-candidate-technical-failure-receipt"
CANDIDATE_TECHNICAL_FAILURE_RECEIPT_VERSION: Final = 1
_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "candidate_amendment_id",
        "candidate_revision",
        "upstream_prediction_batch_id",
        "upstream_revision",
        "upstream_prediction_receipt_id",
        "raw_prediction_batch_id",
        "raw_prediction_batch_file_sha256",
        "prediction_record_count",
        "technical_failure_record_count",
        "candidate_artifact_id_by_record",
        "candidate_seal_file_sha256_by_record",
        "raw_record_file_sha256_by_record",
        "information_boundary",
        "receipt_id",
    }
)
_INFORMATION_BOUNDARY: Final = {
    "all_100_raw_predictions_sealed": True,
    "source_suffix_opened": False,
    "future_object_observations_used_for_prediction": False,
    "v5_confirmation_payloads_used": False,
    "v5_confirmation_outcomes_used": False,
    "v6_target_payloads_used": False,
    "v6_target_outcomes_used": False,
    "existing_source_provider_products_reused": True,
    "prob4d_pipeline_artifacts_reused": True,
    "prob4d_decoded_uniform_fusion_used": False,
    "motioncrafter_disjoint_baseline_used": True,
    "new_prob4d_inference_run": False,
    "new_motioncrafter_inference_run": False,
    "human_selection_used": False,
    "replacement_allowed": False,
    "source_suffix_access_authorized": False,
    "independent_confirmation_authorized": False,
    "claim_authorized": False,
}
_EXECUTION_INFORMATION_BOUNDARY: Final = {
    **_INFORMATION_BOUNDARY,
    "confirmation_payloads_opened": False,
    "public_real_world_dataset": True,
    "new_measurements_collected": False,
    "source_prefix_opened": True,
}
_TECHNICAL_FAILURE_INFORMATION_BOUNDARY: Final = {
    **_EXECUTION_INFORMATION_BOUNDARY,
    "all_100_raw_predictions_sealed": False,
}
_EXECUTION_ARTIFACT_NAMES: Final = frozenset(
    {
        "candidate_amendment",
        "candidate_panel_receipt",
        "candidate_raw_batch",
        "execution_lock",
        "upstream_execution_receipt",
        "upstream_prediction_batch",
        "upstream_prediction_receipt",
        "upstream_source_plan",
    }
)
_EXECUTION_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        "candidate_amendment_id",
        "execution_lock_id",
        "candidate_revision",
        "runner_name",
        "workflow_run_id",
        "workflow_run_attempt",
        "upstream_execution_receipt_id",
        "candidate_panel_receipt_id",
        "raw_prediction_batch_id",
        "prediction_record_count",
        "technical_failure_record_count",
        "artifacts",
        "information_boundary",
        "source_suffix_access_authorized",
        "independent_confirmation_authorized",
        "claim_authorized",
        "receipt_id",
    }
)
_TECHNICAL_FAILURE_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        "candidate_amendment_id",
        "execution_lock_id",
        "candidate_revision",
        "runner_name",
        "workflow_run_id",
        "workflow_run_attempt",
        "upstream_execution_receipt_id",
        "terminal_stage",
        "exit_code",
        "retained_artifacts",
        "information_boundary",
        "source_suffix_access_authorized",
        "independent_confirmation_authorized",
        "claim_authorized",
        "receipt_id",
    }
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _validate_sealed_prediction_artifact(
    directory: Path,
    *,
    record: Mapping[str, Any],
    lock: Mapping[str, Any],
    implementation_revision: str,
) -> tuple[Mapping[str, Any], Any]:
    """Validate one prefix-sealed prediction without importing suffix tooling."""

    prediction_seal, result = load_deform360_joint_sparse_prediction_v5(directory)
    methods = cast(Mapping[str, Mapping[str, Any]], record["methods"])
    _require(
        prediction_seal["execution_lock_id"] == lock["execution_lock_id"]
        and prediction_seal["implementation_revision"] == implementation_revision
        and prediction_seal["prediction_fit_artifact_id"]
        == record["prediction_fit_artifact_id"]
        and prediction_seal["prediction_fit_object_ids"]
        == record["prediction_fit_object_ids"]
        and prediction_seal["factor_admitted"] == record["factor_admitted"]
        and prediction_seal["physical_mode"] == record["physical_mode"]
        and float(prediction_seal["risk_score"]) == float(record["risk_score"]),
        "published prediction differs from its source seal",
    )
    artifact_ids = cast(Mapping[str, str], prediction_seal["method_artifact_ids"])
    _require(
        all(
            artifact_ids[method_id] == methods[method_id]["artifact_id"]
            for method_id in RAW_METHOD_IDS
        ),
        "published method artifact differs from its source seal",
    )
    if record["technical_failure"]:
        baseline = result.trajectories_m[RAW_METHOD_IDS[0]]
        _require(
            not record["factor_admitted"]
            and all(
                np.array_equal(result.trajectories_m[method_id], baseline)
                for method_id in RAW_METHOD_IDS
            ),
            "technical prediction failure did not preserve exact fallback",
        )
    return prediction_seal, result


def _identifier(value: object, *, name: str) -> str:
    result = nonempty_string(value, name=name)
    _require(
        result == result.strip() and "\x00" not in result,
        f"{name} must be a canonical string",
    )
    return result


def _content_identity(value: Mapping[str, Any], *, field: str, name: str) -> None:
    declared = sha256_digest(value.get(field), name=field)
    identity = {key: item for key, item in value.items() if key != field}
    _require(declared == content_id(identity), f"{name} identity changed")


def _validate_upstream_execution_receipt(value: object) -> dict[str, Any]:
    receipt = validate_deform360_v6_source_camera_reuse_execution_receipt(value)
    _require(
        receipt["receipt_id"] == UPSTREAM_EXECUTION_RECEIPT_ID
        and receipt["prediction_batch_id"] == UPSTREAM_PREDICTION_BATCH_ID
        and receipt["source_prediction_receipt_id"] == UPSTREAM_PREDICTION_RECEIPT_ID
        and receipt["source_plan_id"] == UPSTREAM_SOURCE_PLAN_ID
        and receipt["source_revision"] == UPSTREAM_REVISION,
        "candidate runner binds another upstream execution",
    )
    return receipt


def build_deform360_v61_candidate_execution_receipt(
    *,
    candidate_revision: str,
    runner_name: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    upstream_execution_receipt: Mapping[str, Any],
    candidate_panel_receipt: Mapping[str, Any],
    artifact_file_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Bind one protected, fully validated 100-record prefix execution."""

    upstream = _validate_upstream_execution_receipt(upstream_execution_receipt)
    panel = validate_deform360_v61_candidate_panel_receipt(candidate_panel_receipt)
    revision = exact_revision(candidate_revision, name="candidate_revision")
    _require(
        panel["candidate_revision"] == revision,
        "candidate execution revision changed",
    )
    artifacts = source_artifact_mapping(
        artifact_file_sha256, name="candidate execution artifacts"
    )
    _require(
        set(artifacts) == _EXECUTION_ARTIFACT_NAMES
        and artifacts["candidate_amendment"] == CANDIDATE_AMENDMENT_FILE_SHA256
        and artifacts["execution_lock"] == EXECUTION_LOCK_FILE_SHA256
        and artifacts["upstream_source_plan"] == UPSTREAM_SOURCE_PLAN_FILE_SHA256
        and artifacts["upstream_prediction_batch"]
        == UPSTREAM_PREDICTION_BATCH_FILE_SHA256
        and artifacts["upstream_prediction_receipt"]
        == UPSTREAM_PREDICTION_RECEIPT_FILE_SHA256
        and artifacts["upstream_execution_receipt"]
        == UPSTREAM_EXECUTION_RECEIPT_FILE_SHA256
        and artifacts["candidate_raw_batch"]
        == panel["raw_prediction_batch_file_sha256"],
        "candidate execution artifacts changed",
    )
    identity: dict[str, Any] = {
        "schema": CANDIDATE_EXECUTION_RECEIPT_SCHEMA,
        "schema_version": CANDIDATE_EXECUTION_RECEIPT_VERSION,
        "status": "candidate-prefix-panel-sealed",
        "candidate_amendment_id": CANDIDATE_AMENDMENT_ID,
        "execution_lock_id": EXECUTION_LOCK_ID,
        "candidate_revision": revision,
        "runner_name": _identifier(runner_name, name="runner name"),
        "workflow_run_id": genuine_integer(
            workflow_run_id, name="workflow run ID", minimum=1
        ),
        "workflow_run_attempt": genuine_integer(
            workflow_run_attempt, name="workflow run attempt", minimum=1
        ),
        "upstream_execution_receipt_id": upstream["receipt_id"],
        "candidate_panel_receipt_id": panel["receipt_id"],
        "raw_prediction_batch_id": panel["raw_prediction_batch_id"],
        "prediction_record_count": panel["prediction_record_count"],
        "technical_failure_record_count": panel["technical_failure_record_count"],
        "artifacts": dict(artifacts),
        "information_boundary": dict(_EXECUTION_INFORMATION_BOUNDARY),
        "source_suffix_access_authorized": False,
        "independent_confirmation_authorized": False,
        "claim_authorized": False,
    }
    return validate_deform360_v61_candidate_execution_receipt(
        {**identity, "receipt_id": content_id(identity)}
    )


def validate_deform360_v61_candidate_execution_receipt(
    value: object,
) -> dict[str, Any]:
    """Validate one terminal prefix-only execution receipt."""

    _require(isinstance(value, Mapping), "execution receipt must be a JSON object")
    receipt = cast(Mapping[str, Any], value)
    require_exact_fields(
        receipt, expected=_EXECUTION_RECEIPT_FIELDS, name="candidate execution receipt"
    )
    _content_identity(receipt, field="receipt_id", name="candidate execution receipt")
    _require(
        receipt.get("schema") == CANDIDATE_EXECUTION_RECEIPT_SCHEMA
        and receipt.get("schema_version") == CANDIDATE_EXECUTION_RECEIPT_VERSION
        and receipt.get("status") == "candidate-prefix-panel-sealed"
        and receipt.get("candidate_amendment_id") == CANDIDATE_AMENDMENT_ID
        and receipt.get("execution_lock_id") == EXECUTION_LOCK_ID
        and receipt.get("upstream_execution_receipt_id")
        == UPSTREAM_EXECUTION_RECEIPT_ID
        and receipt.get("prediction_record_count") == 100
        and receipt.get("information_boundary") == _EXECUTION_INFORMATION_BOUNDARY
        and receipt.get("source_suffix_access_authorized") is False
        and receipt.get("independent_confirmation_authorized") is False
        and receipt.get("claim_authorized") is False,
        "candidate execution receipt contract changed",
    )
    exact_revision(receipt.get("candidate_revision"), name="candidate revision")
    _identifier(receipt.get("runner_name"), name="runner name")
    genuine_integer(receipt.get("workflow_run_id"), name="workflow run ID", minimum=1)
    genuine_integer(
        receipt.get("workflow_run_attempt"),
        name="workflow run attempt",
        minimum=1,
    )
    for field in (
        "candidate_panel_receipt_id",
        "raw_prediction_batch_id",
    ):
        sha256_digest(receipt.get(field), name=field)
    count = receipt.get("technical_failure_record_count")
    _require(type(count) is int and 0 <= count <= 100, "failure count changed")
    artifacts = source_artifact_mapping(
        cast(Mapping[str, str], receipt.get("artifacts")),
        name="candidate execution artifacts",
    )
    _require(
        set(artifacts) == _EXECUTION_ARTIFACT_NAMES
        and artifacts["candidate_amendment"] == CANDIDATE_AMENDMENT_FILE_SHA256
        and artifacts["execution_lock"] == EXECUTION_LOCK_FILE_SHA256
        and artifacts["upstream_source_plan"] == UPSTREAM_SOURCE_PLAN_FILE_SHA256
        and artifacts["upstream_prediction_batch"]
        == UPSTREAM_PREDICTION_BATCH_FILE_SHA256
        and artifacts["upstream_prediction_receipt"]
        == UPSTREAM_PREDICTION_RECEIPT_FILE_SHA256
        and artifacts["upstream_execution_receipt"]
        == UPSTREAM_EXECUTION_RECEIPT_FILE_SHA256,
        "artifact roster or immutable lineage changed",
    )
    return cast(dict[str, Any], plain_json(receipt))


def build_deform360_v61_candidate_technical_failure_receipt(
    *,
    candidate_revision: str,
    runner_name: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    terminal_stage: str,
    exit_code: int,
    retained_artifact_file_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Retain one failed prefix-only execution without authorizing replacement."""

    identity: dict[str, Any] = {
        "schema": CANDIDATE_TECHNICAL_FAILURE_RECEIPT_SCHEMA,
        "schema_version": CANDIDATE_TECHNICAL_FAILURE_RECEIPT_VERSION,
        "status": "candidate-prefix-technical-failure-retained",
        "candidate_amendment_id": CANDIDATE_AMENDMENT_ID,
        "execution_lock_id": EXECUTION_LOCK_ID,
        "candidate_revision": exact_revision(
            candidate_revision, name="candidate revision"
        ),
        "runner_name": _identifier(runner_name, name="runner name"),
        "workflow_run_id": genuine_integer(
            workflow_run_id, name="workflow run ID", minimum=1
        ),
        "workflow_run_attempt": genuine_integer(
            workflow_run_attempt, name="workflow run attempt", minimum=1
        ),
        "upstream_execution_receipt_id": UPSTREAM_EXECUTION_RECEIPT_ID,
        "terminal_stage": _identifier(terminal_stage, name="terminal stage"),
        "exit_code": genuine_integer(exit_code, name="exit code", minimum=1),
        "retained_artifacts": dict(
            source_artifact_mapping(
                retained_artifact_file_sha256,
                name="retained artifacts",
                allow_empty=True,
            )
        ),
        "information_boundary": dict(_TECHNICAL_FAILURE_INFORMATION_BOUNDARY),
        "source_suffix_access_authorized": False,
        "independent_confirmation_authorized": False,
        "claim_authorized": False,
    }
    return validate_deform360_v61_candidate_technical_failure_receipt(
        {**identity, "receipt_id": content_id(identity)}
    )


def validate_deform360_v61_candidate_technical_failure_receipt(
    value: object,
) -> dict[str, Any]:
    """Validate a retained technical failure without opening outcomes."""

    _require(isinstance(value, Mapping), "failure receipt must be a JSON object")
    receipt = cast(Mapping[str, Any], value)
    require_exact_fields(
        receipt,
        expected=_TECHNICAL_FAILURE_RECEIPT_FIELDS,
        name="candidate technical-failure receipt",
    )
    _content_identity(receipt, field="receipt_id", name="failure receipt")
    _require(
        receipt.get("schema") == CANDIDATE_TECHNICAL_FAILURE_RECEIPT_SCHEMA
        and receipt.get("schema_version") == CANDIDATE_TECHNICAL_FAILURE_RECEIPT_VERSION
        and receipt.get("status") == "candidate-prefix-technical-failure-retained"
        and receipt.get("candidate_amendment_id") == CANDIDATE_AMENDMENT_ID
        and receipt.get("execution_lock_id") == EXECUTION_LOCK_ID
        and receipt.get("upstream_execution_receipt_id")
        == UPSTREAM_EXECUTION_RECEIPT_ID
        and receipt.get("information_boundary")
        == _TECHNICAL_FAILURE_INFORMATION_BOUNDARY
        and receipt.get("source_suffix_access_authorized") is False
        and receipt.get("independent_confirmation_authorized") is False
        and receipt.get("claim_authorized") is False,
        "candidate technical-failure contract changed",
    )
    exact_revision(receipt.get("candidate_revision"), name="candidate revision")
    _identifier(receipt.get("runner_name"), name="runner name")
    _identifier(receipt.get("terminal_stage"), name="terminal stage")
    genuine_integer(receipt.get("workflow_run_id"), name="workflow run ID", minimum=1)
    genuine_integer(
        receipt.get("workflow_run_attempt"),
        name="workflow run attempt",
        minimum=1,
    )
    genuine_integer(receipt.get("exit_code"), name="exit code", minimum=1)
    source_artifact_mapping(
        cast(Mapping[str, str], receipt.get("retained_artifacts")),
        name="retained artifacts",
        allow_empty=True,
    )
    return cast(dict[str, Any], plain_json(receipt))


def _publish_or_validate_json(
    value: Mapping[str, Any], path: Path, *, label: str
) -> dict[str, Any]:
    normalized = cast(dict[str, Any], plain_json(value))
    if path.exists():
        _require(
            not path.is_symlink()
            and load_strict_json_object(path, label=label) == normalized,
            f"existing {label} differs",
        )
        return normalized
    write_atomic_json(normalized, path, overwrite=False)
    return normalized


def _prediction_directory(
    prediction_root: Path,
    *,
    ordered_ids: Sequence[str],
    outer_id: str,
    target_id: str,
) -> Path:
    return (
        prediction_root
        / "predictions"
        / f"{ordered_ids.index(outer_id):02d}-{outer_id}"
        / f"{ordered_ids.index(target_id):02d}-{target_id}"
    )


def _failure_id(
    *,
    object_id: str,
    episode_id: int,
    stage: str,
    error: Exception,
    candidate_revision: str,
) -> str:
    return content_id(
        {
            "schema": "bayesian-phystwin.deform360-v6-candidate-technical-failure",
            "schema_version": 1,
            "object_id": object_id,
            "episode_id": episode_id,
            "stage": stage,
            "exception_type": type(error).__name__,
            "exception_message_sha256": hashlib.sha256(
                str(error).encode("utf-8")
            ).hexdigest(),
            "candidate_revision": candidate_revision,
            "source_suffix_opened": False,
            "future_object_observations_used": False,
        }
    )


def build_deform360_v61_candidate_panel_receipt(
    *,
    candidate_revision: str,
    upstream_prediction_receipt_id: str,
    raw_prediction_batch: Mapping[str, Any],
    raw_prediction_batch_file_sha256: str,
    candidate_artifact_id_by_record: Mapping[str, str],
    candidate_seal_file_sha256_by_record: Mapping[str, str],
    raw_record_file_sha256_by_record: Mapping[str, str],
    technical_failure_record_count: int,
) -> dict[str, Any]:
    """Build the atomic pre-suffix barrier receipt for exactly 100 records."""

    revision = exact_revision(candidate_revision, name="candidate_revision")
    receipt_id = sha256_digest(
        upstream_prediction_receipt_id, name="upstream_prediction_receipt_id"
    )
    batch_id = sha256_digest(
        raw_prediction_batch.get("prediction_batch_id"),
        name="raw_prediction_batch_id",
    )
    _require(
        raw_prediction_batch.get("record_count") == 100,
        "raw candidate batch does not contain 100 records",
    )
    artifact_ids = source_artifact_mapping(
        candidate_artifact_id_by_record,
        name="candidate artifact IDs",
    )
    seal_digests = source_artifact_mapping(
        candidate_seal_file_sha256_by_record,
        name="candidate seal file digests",
    )
    record_digests = source_artifact_mapping(
        raw_record_file_sha256_by_record,
        name="raw record file digests",
    )
    expected = {
        f"{outer:02d}-{target:02d}" for outer in range(10) for target in range(10)
    }
    _require(
        set(artifact_ids) == set(seal_digests) == set(record_digests) == expected,
        "candidate receipt record roster changed",
    )
    _require(
        type(technical_failure_record_count) is int
        and 0 <= technical_failure_record_count <= 100,
        "technical failure record count changed",
    )
    identity: dict[str, Any] = {
        "schema": CANDIDATE_PANEL_RECEIPT_SCHEMA,
        "schema_version": CANDIDATE_PANEL_RECEIPT_VERSION,
        "candidate_amendment_id": CANDIDATE_AMENDMENT_ID,
        "candidate_revision": revision,
        "upstream_prediction_batch_id": UPSTREAM_PREDICTION_BATCH_ID,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_prediction_receipt_id": receipt_id,
        "raw_prediction_batch_id": batch_id,
        "raw_prediction_batch_file_sha256": sha256_digest(
            raw_prediction_batch_file_sha256,
            name="raw_prediction_batch_file_sha256",
        ),
        "prediction_record_count": 100,
        "technical_failure_record_count": technical_failure_record_count,
        "candidate_artifact_id_by_record": dict(artifact_ids),
        "candidate_seal_file_sha256_by_record": dict(seal_digests),
        "raw_record_file_sha256_by_record": dict(record_digests),
        "information_boundary": dict(_INFORMATION_BOUNDARY),
    }
    return {**identity, "receipt_id": content_id(identity)}


def validate_deform360_v61_candidate_panel_receipt(
    value: object,
    *,
    raw_prediction_batch: Mapping[str, Any] | None = None,
    raw_prediction_batch_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a complete source-prefix-only panel receipt."""

    _require(isinstance(value, Mapping), "candidate receipt must be a JSON object")
    receipt = cast(Mapping[str, Any], value)
    require_exact_fields(receipt, expected=_RECEIPT_FIELDS, name="candidate receipt")
    _require(
        receipt.get("schema") == CANDIDATE_PANEL_RECEIPT_SCHEMA
        and receipt.get("schema_version") == CANDIDATE_PANEL_RECEIPT_VERSION
        and receipt.get("candidate_amendment_id") == CANDIDATE_AMENDMENT_ID
        and receipt.get("upstream_prediction_batch_id") == UPSTREAM_PREDICTION_BATCH_ID
        and receipt.get("upstream_revision") == UPSTREAM_REVISION
        and receipt.get("prediction_record_count") == 100
        and receipt.get("information_boundary") == _INFORMATION_BOUNDARY,
        "candidate receipt contract changed",
    )
    exact_revision(receipt.get("candidate_revision"), name="candidate_revision")
    sha256_digest(
        receipt.get("upstream_prediction_receipt_id"),
        name="upstream_prediction_receipt_id",
    )
    for field in (
        "raw_prediction_batch_id",
        "raw_prediction_batch_file_sha256",
    ):
        sha256_digest(receipt.get(field), name=field)
    count = receipt.get("technical_failure_record_count")
    _require(
        type(count) is int and 0 <= count <= 100,
        "candidate receipt technical-failure count changed",
    )
    expected = {
        f"{outer:02d}-{target:02d}" for outer in range(10) for target in range(10)
    }
    for field in (
        "candidate_artifact_id_by_record",
        "candidate_seal_file_sha256_by_record",
        "raw_record_file_sha256_by_record",
    ):
        mapping = source_artifact_mapping(
            cast(Mapping[str, str], receipt.get(field)), name=field
        )
        _require(set(mapping) == expected, f"{field} roster changed")
    identity = {key: item for key, item in receipt.items() if key != "receipt_id"}
    _require(
        receipt.get("receipt_id") == content_id(identity),
        "candidate receipt identity changed",
    )
    if raw_prediction_batch is not None:
        _require(
            receipt.get("raw_prediction_batch_id")
            == raw_prediction_batch.get("prediction_batch_id"),
            "candidate receipt binds another raw batch",
        )
    if raw_prediction_batch_file_sha256 is not None:
        _require(
            receipt.get("raw_prediction_batch_file_sha256")
            == sha256_digest(
                raw_prediction_batch_file_sha256,
                name="raw_prediction_batch_file_sha256",
            ),
            "candidate receipt raw-batch digest changed",
        )
    return cast(dict[str, Any], plain_json(receipt))


def publish_deform360_v61_candidate_panel(
    *,
    candidate_amendment_path: str | Path,
    execution_lock_path: str | Path,
    source_plan_path: str | Path,
    upstream_prediction_batch_path: str | Path,
    upstream_prediction_receipt_path: str | Path,
    upstream_execution_receipt_path: str | Path,
    upstream_source_seal_root: str | Path,
    upstream_prediction_root: str | Path,
    input_root: str | Path,
    output_root: str | Path,
    candidate_revision: str,
) -> dict[str, Any]:
    """Publish exactly one 100-record candidate panel without suffix access."""

    revision = exact_revision(candidate_revision, name="candidate_revision")
    _require(revision != UPSTREAM_REVISION, "candidate revision equals upstream")
    amendment_path = Path(candidate_amendment_path).resolve(strict=True)
    load_deform360_v61_candidate_amendment(amendment_path)
    _require(
        _sha256_file(amendment_path) == CANDIDATE_AMENDMENT_FILE_SHA256,
        "candidate amendment bytes changed",
    )
    lock_path = Path(execution_lock_path).resolve(strict=True)
    lock = load_deform360_joint_sparse_source_execution_lock_v5(lock_path)
    _require(
        lock.get("execution_lock_id") == EXECUTION_LOCK_ID
        and _sha256_file(lock_path) == EXECUTION_LOCK_FILE_SHA256,
        "candidate execution lock changed",
    )
    plan_path = Path(source_plan_path).resolve(strict=True)
    plan = validate_deform360_joint_sparse_source_prediction_plan_v5_2(
        load_strict_json_object(plan_path, label="v5.2 source prediction plan"),
        lock=lock,
    )
    _require(
        plan.get("implementation_revision") == UPSTREAM_REVISION
        and plan.get("plan_id") == UPSTREAM_SOURCE_PLAN_ID
        and _sha256_file(plan_path) == UPSTREAM_SOURCE_PLAN_FILE_SHA256,
        "candidate plan binds another upstream revision",
    )
    batch_path = Path(upstream_prediction_batch_path).resolve(strict=True)
    batch = validate_deform360_joint_sparse_source_prediction_batch_v5(
        load_strict_json_object(batch_path, label="upstream prediction batch"),
        lock,
    )
    _require(
        batch.get("prediction_batch_id") == UPSTREAM_PREDICTION_BATCH_ID
        and batch.get("implementation_revision") == UPSTREAM_REVISION
        and _sha256_file(batch_path) == UPSTREAM_PREDICTION_BATCH_FILE_SHA256,
        "candidate runner binds another upstream prediction batch",
    )
    receipt_path = Path(upstream_prediction_receipt_path).resolve(strict=True)
    upstream_receipt = validate_deform360_joint_sparse_source_prediction_receipt_v5_2(
        load_strict_json_object(receipt_path, label="upstream prediction receipt"),
        lock=lock,
        plan=plan,
        prediction_batch=batch,
        prediction_batch_file_sha256=_sha256_file(batch_path),
    )
    _require(
        upstream_receipt["receipt_id"] == UPSTREAM_PREDICTION_RECEIPT_ID
        and _sha256_file(receipt_path) == UPSTREAM_PREDICTION_RECEIPT_FILE_SHA256,
        "candidate runner binds another upstream prediction receipt",
    )
    execution_receipt_path = Path(upstream_execution_receipt_path).resolve(strict=True)
    _require(
        _sha256_file(execution_receipt_path) == UPSTREAM_EXECUTION_RECEIPT_FILE_SHA256,
        "candidate runner binds another upstream execution receipt",
    )
    _validate_upstream_execution_receipt(
        load_strict_json_object(
            execution_receipt_path, label="upstream execution receipt"
        )
    )
    seal_root = Path(upstream_source_seal_root).resolve(strict=True)
    prediction_root = Path(upstream_prediction_root).resolve(strict=True)
    root = _ordinary_root(input_root)
    output = Path(output_root).absolute()
    output.mkdir(parents=True, exist_ok=True)
    _require(
        output.is_dir()
        and not output.is_symlink()
        and not any(parent.is_symlink() for parent in output.parents),
        "candidate output root is invalid",
    )
    cohort = _cohort(lock)
    ordered_ids = tuple(sorted(cohort))
    _require(len(ordered_ids) == 10, "candidate source cohort changed")
    objects = {
        cast(str, row["object_id"]): cast(Mapping[str, Any], row)
        for row in cast(Sequence[Mapping[str, Any]], plan["objects"])
    }
    _require(set(objects) == set(ordered_ids), "candidate source plan cohort changed")
    records = {
        (
            cast(str, row["outer_held_out_object_id"]),
            cast(str, row["object_id"]),
        ): cast(Mapping[str, Any], row)
        for row in cast(Sequence[Mapping[str, Any]], batch["records"])
    }
    _require(len(records) == 100, "upstream prediction record roster changed")
    common_sources = source_artifact_mapping(
        {
            "amendments/candidate-producer-v6-1.json": _sha256_file(amendment_path),
            "locks/source-execution-v5.json": _sha256_file(lock_path),
            "plans/source-prediction-plan-v5-2.json": _sha256_file(plan_path),
            "upstream/source-prediction-batch.json": _sha256_file(batch_path),
            "upstream/source-prediction-receipt.json": _sha256_file(receipt_path),
        },
        name="candidate common source artifacts",
    )
    base_fit = Deform360JointSparsePrefixFitV5(
        fit_object_ids=ordered_ids,
        source_artifact_ids=common_sources,
    )
    prepared: dict[
        str,
        tuple[
            np.ndarray, tuple[Any, ...], Mapping[str, str], tuple[str, Exception] | None
        ],
    ] = {}
    for target_id in ordered_ids:
        row = objects[target_id]
        physical_record = cast(Mapping[str, Any], row["physical"])
        physical_path = _verified_file(
            root, physical_record, name=f"physical archive for {target_id}"
        )
        physical, _persistence = _load_physical_archive(
            physical_path,
            physical_mode=cast(str, physical_record["physical_mode"]),
        )
        object_sources: dict[str, str] = {
            **dict(common_sources),
            f"physical/{target_id}.npz": _sha256_file(physical_path),
        }
        windows: list[Any] = []
        technical_failure: tuple[str, Exception] | None = None
        admission = cast(Mapping[str, Any], row["camera_admission"])
        if admission["exact_physical_fallback_required"]:
            technical_failure = (
                "camera_admission",
                ValueError("fewer than two passing public prefix cameras"),
            )
        else:
            try:
                prefix = cast(Sequence[int], row["raw_prefix_range_half_open"])
                raw_prefix = (int(prefix[0]), int(prefix[1]))
                for visual in cast(Sequence[Mapping[str, Any]], row["visual_windows"]):
                    camera_id = cast(str, visual["camera_id"])
                    visual_product = _verified_file(
                        root,
                        cast(Mapping[str, Any], visual["decoded_uniform"]),
                        name=(
                            "sealed MotionCrafter disjoint baseline for "
                            f"{target_id}/{camera_id}"
                        ),
                    )
                    _require(
                        visual_product.name == SEALED_VISUAL_PRODUCT_FILENAME,
                        "sealed visual product is not the MotionCrafter disjoint baseline",
                    )
                    metric = _verified_file(
                        root,
                        cast(Mapping[str, Any], visual["metric_prefix"]),
                        name=f"metric prefix for {target_id}/{camera_id}",
                    )
                    object_sources.update(
                        {
                            f"visual/{target_id}/{camera_id}/motioncrafter-disjoint-baseline.npz": _sha256_file(
                                visual_product
                            ),
                            f"visual/{target_id}/{camera_id}/metric-prefix.npz": _sha256_file(
                                metric
                            ),
                        }
                    )
                    visual_rows, _gauge = prepare_deform360_disjoint_visual_window_v6_1(
                        camera_id=camera_id,
                        disjoint_motioncrafter_path=visual_product,
                        metric_prefix_path=metric,
                        raw_prefix_range_half_open=raw_prefix,
                        fit=base_fit,
                        source_artifact_ids=object_sources,
                    )
                    windows.append(visual_rows)
            except (
                OSError,
                ValueError,
                ArithmeticError,
                np.linalg.LinAlgError,
            ) as error:
                technical_failure = ("prefix_provider", error)
                windows = []
        prepared[target_id] = (
            physical,
            tuple(windows),
            source_artifact_mapping(
                object_sources, name=f"candidate sources for {target_id}"
            ),
            technical_failure,
        )

    raw_records: list[dict[str, Any]] = []
    artifact_ids: dict[str, str] = {}
    artifact_seal_digests: dict[str, str] = {}
    raw_record_digests: dict[str, str] = {}
    technical_failure_records = 0
    receipt_seals = cast(
        Mapping[str, str], upstream_receipt["source_prediction_seal_file_sha256"]
    )
    artifact_root = output / "candidate-artifacts"
    record_root = output / "raw-records"
    artifact_root.mkdir(parents=True, exist_ok=True)
    record_root.mkdir(parents=True, exist_ok=True)
    for outer_index, outer_id in enumerate(ordered_ids):
        for target_index, target_id in enumerate(ordered_ids):
            key = f"{outer_index:02d}-{target_index:02d}"
            upstream_seal_path = seal_root / f"{key}.json"
            _require(
                upstream_seal_path.is_file()
                and not upstream_seal_path.is_symlink()
                and _sha256_file(upstream_seal_path)
                == receipt_seals[upstream_seal_path.name],
                "upstream source prediction seal bytes changed",
            )
            upstream_seal = validate_deform360_joint_sparse_source_prediction_seal_v5(
                load_strict_json_object(
                    upstream_seal_path, label="upstream source seal"
                ),
                lock,
            )
            upstream_record = records[(outer_id, target_id)]
            _require(
                upstream_seal == upstream_record, "upstream source seal order changed"
            )
            directory = _prediction_directory(
                prediction_root,
                ordered_ids=ordered_ids,
                outer_id=outer_id,
                target_id=target_id,
            )
            prediction_seal, result = _validate_sealed_prediction_artifact(
                directory,
                record=upstream_record,
                lock=lock,
                implementation_revision=UPSTREAM_REVISION,
            )
            prepared_physical, prepared_windows, prepared_sources, prepared_failure = (
                prepared[target_id]
            )
            if prepared_failure is None:
                arrays = build_deform360_v61_candidate_arrays(
                    physical_prediction_m=prepared_physical,
                    b0_trajectory_m=result.trajectories_m[B0_PHYSICAL_FALLBACK],
                    b1_trajectory_m=result.trajectories_m[B1_LAST_CAUSAL_RESIDUAL],
                    visual_windows=cast(Sequence[Any], prepared_windows),
                )
                failure_id = None
            else:
                failure_stage, failure_exception = prepared_failure
                failure_id = _failure_id(
                    object_id=target_id,
                    episode_id=cohort[target_id][0],
                    stage=failure_stage,
                    error=failure_exception,
                    candidate_revision=revision,
                )
                arrays = build_deform360_v61_technical_fallback_arrays(
                    physical_prediction_m=prepared_physical,
                    b0_trajectory_m=result.trajectories_m[B0_PHYSICAL_FALLBACK],
                    b1_trajectory_m=result.trajectories_m[B1_LAST_CAUSAL_RESIDUAL],
                )
                technical_failure_records += 1
            fit_ids = tuple(sorted(set(ordered_ids) - {outer_id, target_id}))
            candidate_directory = (
                artifact_root
                / f"{outer_index:02d}-{outer_id}"
                / f"{target_index:02d}-{target_id}"
            )
            candidate_seal = publish_deform360_v61_candidate_artifact(
                arrays,
                candidate_directory,
                candidate_revision=revision,
                outer_held_out_object_id=outer_id,
                object_id=target_id,
                episode_id=cohort[target_id][0],
                stratum=cohort[target_id][1],
                fit_object_ids=fit_ids,
                source_artifacts={
                    **dict(prepared_sources),
                    f"upstream/source-seals/{upstream_seal_path.name}": _sha256_file(
                        upstream_seal_path
                    ),
                    f"upstream/predictions/{outer_index:02d}-{outer_id}/{target_index:02d}-{target_id}/prediction-seal.json": _sha256_file(
                        directory / "prediction-seal.json"
                    ),
                    f"upstream/predictions/{outer_index:02d}-{outer_id}/{target_index:02d}-{target_id}/prediction-arrays.npz": _sha256_file(
                        directory / "prediction-arrays.npz"
                    ),
                    f"upstream/predictions/{outer_index:02d}-{outer_id}/{target_index:02d}-{target_id}/SHA256SUMS": _sha256_file(
                        directory / "SHA256SUMS"
                    ),
                    "upstream/prediction-seal-id": cast(
                        str, prediction_seal["prediction_seal_id"]
                    ),
                },
                technical_failure=prepared_failure is not None,
                technical_failure_id=failure_id,
            )
            candidate_seal_path = candidate_directory / CANDIDATE_SEAL_FILENAME
            raw_record = build_deform360_v6_raw_nested_prediction(
                cohort=cohort,
                upstream_prediction_batch_id=UPSTREAM_PREDICTION_BATCH_ID,
                upstream_revision=UPSTREAM_REVISION,
                candidate_revision=revision,
                outer_held_out_object_id=outer_id,
                object_id=target_id,
                variants=raw_variants_from_deform360_v61_candidate_seal(candidate_seal),
                source_artifacts={
                    **dict(prepared_sources),
                    f"candidate-artifacts/{key}/{CANDIDATE_SEAL_FILENAME}": _sha256_file(
                        candidate_seal_path
                    ),
                    f"candidate-artifacts/{key}/{CANDIDATE_ARCHIVE_FILENAME}": _sha256_file(
                        candidate_directory / CANDIDATE_ARCHIVE_FILENAME
                    ),
                },
            )
            raw_record_path = record_root / f"{key}.json"
            _publish_or_validate_json(raw_record, raw_record_path, label="raw record")
            raw_records.append(raw_record)
            artifact_ids[key] = cast(str, candidate_seal["candidate_artifact_id"])
            artifact_seal_digests[key] = _sha256_file(candidate_seal_path)
            raw_record_digests[key] = _sha256_file(raw_record_path)

    raw_batch = build_deform360_v6_raw_nested_batch(raw_records, cohort=cohort)
    raw_batch_path = output / CANDIDATE_BATCH_FILENAME
    publish_deform360_v6_raw_nested_batch(raw_batch, raw_batch_path, cohort=cohort)
    receipt = build_deform360_v61_candidate_panel_receipt(
        candidate_revision=revision,
        upstream_prediction_receipt_id=cast(str, upstream_receipt["receipt_id"]),
        raw_prediction_batch=raw_batch,
        raw_prediction_batch_file_sha256=_sha256_file(raw_batch_path),
        candidate_artifact_id_by_record=artifact_ids,
        candidate_seal_file_sha256_by_record=artifact_seal_digests,
        raw_record_file_sha256_by_record=raw_record_digests,
        technical_failure_record_count=technical_failure_records,
    )
    receipt_path = output / CANDIDATE_RECEIPT_FILENAME
    _publish_or_validate_json(receipt, receipt_path, label="candidate panel receipt")
    return receipt


def validate_deform360_v61_candidate_panel(
    *,
    execution_lock_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Rehash and validate all 100 records behind a candidate-panel receipt."""

    lock = load_deform360_joint_sparse_source_execution_lock_v5(execution_lock_path)
    cohort = _cohort(lock)
    ordered_ids = tuple(sorted(cohort))
    _require(len(ordered_ids) == 10, "candidate panel cohort changed")
    root = Path(output_root).absolute()
    _require(
        root.is_dir()
        and not root.is_symlink()
        and not any(parent.is_symlink() for parent in root.parents),
        "candidate panel root is invalid",
    )
    root = root.resolve(strict=True)
    batch_path = root / CANDIDATE_BATCH_FILENAME
    receipt_path = root / CANDIDATE_RECEIPT_FILENAME
    _require(
        batch_path.is_file()
        and receipt_path.is_file()
        and not batch_path.is_symlink()
        and not receipt_path.is_symlink(),
        "candidate panel barrier artifacts are incomplete",
    )
    batch = validate_deform360_v6_raw_nested_batch(
        load_strict_json_object(batch_path, label="raw nested candidate batch"),
        cohort=cohort,
    )
    receipt = validate_deform360_v61_candidate_panel_receipt(
        load_strict_json_object(receipt_path, label="candidate panel receipt"),
        raw_prediction_batch=batch,
        raw_prediction_batch_file_sha256=_sha256_file(batch_path),
    )
    records = {
        (
            cast(str, row["outer_held_out_object_id"]),
            cast(str, row["object_id"]),
        ): cast(Mapping[str, Any], row)
        for row in cast(Sequence[Mapping[str, Any]], batch["records"])
    }
    artifact_ids = cast(Mapping[str, str], receipt["candidate_artifact_id_by_record"])
    seal_digests = cast(
        Mapping[str, str], receipt["candidate_seal_file_sha256_by_record"]
    )
    record_digests = cast(
        Mapping[str, str], receipt["raw_record_file_sha256_by_record"]
    )
    technical_failures = 0
    for outer_index, outer_id in enumerate(ordered_ids):
        for target_index, target_id in enumerate(ordered_ids):
            key = f"{outer_index:02d}-{target_index:02d}"
            candidate_directory = (
                root
                / "candidate-artifacts"
                / f"{outer_index:02d}-{outer_id}"
                / f"{target_index:02d}-{target_id}"
            )
            candidate_seal, candidate_arrays = load_deform360_v61_candidate_artifact(
                candidate_directory
            )
            _require(
                candidate_seal["candidate_artifact_id"] == artifact_ids[key]
                and _sha256_file(candidate_directory / CANDIDATE_SEAL_FILENAME)
                == seal_digests[key],
                "candidate artifact differs from its panel receipt",
            )
            technical_failures += int(candidate_seal["technical_failure"])
            if candidate_seal["technical_failure"]:
                _require(
                    np.array_equal(
                        candidate_arrays.arrays["trajectory__d1_native_model_average"],
                        candidate_arrays.arrays["trajectory__b0_physical_fallback"],
                    ),
                    "technical candidate failure is not exact B0 fallback",
                )
            raw_record_path = root / "raw-records" / f"{key}.json"
            _require(
                raw_record_path.is_file()
                and not raw_record_path.is_symlink()
                and _sha256_file(raw_record_path) == record_digests[key],
                "raw candidate record differs from its panel receipt",
            )
            raw_record = validate_deform360_v6_raw_nested_prediction(
                load_strict_json_object(raw_record_path, label="raw candidate record"),
                cohort=cohort,
            )
            _require(
                raw_record == records[(outer_id, target_id)]
                and raw_record["candidate_revision"]
                == candidate_seal["candidate_revision"]
                == receipt["candidate_revision"],
                "candidate panel record lineage changed",
            )
    _require(
        technical_failures == receipt["technical_failure_record_count"],
        "candidate panel technical-failure accounting changed",
    )
    return receipt


def seal_deform360_v61_candidate_execution(
    *,
    candidate_amendment_path: str | Path,
    execution_lock_path: str | Path,
    upstream_source_plan_path: str | Path,
    upstream_prediction_batch_path: str | Path,
    upstream_prediction_receipt_path: str | Path,
    upstream_execution_receipt_path: str | Path,
    candidate_output_root: str | Path,
    candidate_revision: str,
    runner_name: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal one validated 100-record candidate execution receipt."""

    candidate_root = Path(candidate_output_root).resolve(strict=True)
    panel = validate_deform360_v61_candidate_panel(
        execution_lock_path=execution_lock_path,
        output_root=candidate_root,
    )
    paths = {
        "candidate_amendment": Path(candidate_amendment_path).resolve(strict=True),
        "execution_lock": Path(execution_lock_path).resolve(strict=True),
        "upstream_source_plan": Path(upstream_source_plan_path).resolve(strict=True),
        "upstream_prediction_batch": Path(upstream_prediction_batch_path).resolve(
            strict=True
        ),
        "upstream_prediction_receipt": Path(upstream_prediction_receipt_path).resolve(
            strict=True
        ),
        "upstream_execution_receipt": Path(upstream_execution_receipt_path).resolve(
            strict=True
        ),
        "candidate_raw_batch": candidate_root / CANDIDATE_BATCH_FILENAME,
        "candidate_panel_receipt": candidate_root / CANDIDATE_RECEIPT_FILENAME,
    }
    _require(
        all(path.is_file() and not path.is_symlink() for path in paths.values()),
        "candidate execution artifact is missing",
    )
    upstream_execution = _validate_upstream_execution_receipt(
        load_strict_json_object(
            paths["upstream_execution_receipt"],
            label="upstream execution receipt",
        )
    )
    receipt = build_deform360_v61_candidate_execution_receipt(
        candidate_revision=candidate_revision,
        runner_name=runner_name,
        workflow_run_id=workflow_run_id,
        workflow_run_attempt=workflow_run_attempt,
        upstream_execution_receipt=upstream_execution,
        candidate_panel_receipt=panel,
        artifact_file_sha256={name: _sha256_file(path) for name, path in paths.items()},
    )
    destination = Path(output_path).absolute()
    return _publish_or_validate_json(
        receipt, destination, label="candidate execution receipt"
    )


def retain_deform360_v61_candidate_execution_failure(
    *,
    candidate_revision: str,
    runner_name: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    terminal_stage: str,
    exit_code: int,
    artifact_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Retain one bounded source-prefix technical failure."""

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
            "retained failure artifacts contain a symlink",
        )
        if path.is_file() and path.absolute() != destination:
            retained[path.relative_to(root).as_posix()] = _sha256_file(path)
    receipt = build_deform360_v61_candidate_technical_failure_receipt(
        candidate_revision=candidate_revision,
        runner_name=runner_name,
        workflow_run_id=workflow_run_id,
        workflow_run_attempt=workflow_run_attempt,
        terminal_stage=terminal_stage,
        exit_code=exit_code,
        retained_artifact_file_sha256=retained,
    )
    return _publish_or_validate_json(
        receipt, destination, label="candidate technical-failure receipt"
    )


__all__ = [
    "CANDIDATE_BATCH_FILENAME",
    "CANDIDATE_EXECUTION_RECEIPT_SCHEMA",
    "CANDIDATE_PANEL_RECEIPT_SCHEMA",
    "CANDIDATE_PANEL_RECEIPT_VERSION",
    "CANDIDATE_RECEIPT_FILENAME",
    "CANDIDATE_TECHNICAL_FAILURE_RECEIPT_SCHEMA",
    "build_deform360_v61_candidate_execution_receipt",
    "build_deform360_v61_candidate_technical_failure_receipt",
    "build_deform360_v61_candidate_panel_receipt",
    "publish_deform360_v61_candidate_panel",
    "retain_deform360_v61_candidate_execution_failure",
    "seal_deform360_v61_candidate_execution",
    "validate_deform360_v61_candidate_execution_receipt",
    "validate_deform360_v61_candidate_panel",
    "validate_deform360_v61_candidate_panel_receipt",
    "validate_deform360_v61_candidate_technical_failure_receipt",
]
