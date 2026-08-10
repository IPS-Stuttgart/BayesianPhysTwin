"""Prediction-sealed evidence assembly for the public Deform360 v5 source gate.

The source-gate evaluator intentionally accepts one compact JSON artifact.  This
module closes the preceding custody gap: all 100 nested source forecasts are
sealed in an outcome-free batch before any development suffix score can be
attached.  Outcome records bind the exact batch and exact forecast artifact,
and assembly is deterministic and non-replacing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np

from ._canonical_contracts import genuine_boolean, genuine_integer, plain_json
from ._portable_contracts import (
    canonical_sorted_strings,
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
    write_atomic_json,
)
from .deform360_joint_sparse_source_gate_v5 import (
    RAW_METHOD_IDS,
    SOURCE_EVIDENCE_SCHEMA,
    SOURCE_EVIDENCE_SEMANTICS,
    SOURCE_EVIDENCE_VERSION,
    load_deform360_joint_sparse_source_execution_lock_v5,
    parse_deform360_joint_sparse_source_evidence_v5,
)

SOURCE_PREDICTION_SEAL_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-source-prediction-seal"
)
SOURCE_PREDICTION_SEAL_VERSION: Final = 1
SOURCE_PREDICTION_SEAL_SEMANTICS: Final = (
    "outcome-free-nested-development-source-forecast-v1"
)
SOURCE_PREDICTION_BATCH_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-source-prediction-batch"
)
SOURCE_PREDICTION_BATCH_VERSION: Final = 1
SOURCE_PREDICTION_BATCH_SEMANTICS: Final = (
    "complete-ten-fold-outcome-free-nested-source-forecast-batch-v1"
)
SOURCE_FOLD_PREDICTION_SEAL_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-source-fold-prediction-seal"
)
SOURCE_OUTCOME_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-source-outcome"
)
SOURCE_OUTCOME_VERSION: Final = 1
SOURCE_OUTCOME_SEMANTICS: Final = (
    "development-suffix-score-bound-to-sealed-source-forecast-v1"
)

RecordRole = Literal["held_out", "training"]

_PREDICTION_BOUNDARY = {
    "confirmation_outcomes_opened": False,
    "confirmation_payloads_opened": False,
    "development_suffix_opened": False,
    "future_object_observations_used_for_prediction": False,
    "human_approval_used": False,
    "new_measurements_collected": False,
    "public_released_measurements_used": True,
    "replacement_allowed": False,
    "target_outcomes_used": False,
}
_OUTCOME_BOUNDARY = {
    "confirmation_outcomes_opened": False,
    "confirmation_payloads_opened": False,
    "development_suffix_opened_after_prediction_batch": True,
    "future_object_observations_used_for_prediction": False,
    "human_approval_used": False,
    "new_measurements_collected": False,
    "public_released_measurements_used": True,
    "replacement_allowed": False,
    "target_outcomes_used": False,
}
_EVIDENCE_BOUNDARY = {
    "confirmation_outcomes_opened": False,
    "confirmation_payloads_opened": False,
    "development_suffix_opened_before_prediction_seal": False,
    "future_object_observations_used_for_prediction": False,
    "human_approval_used": False,
    "new_measurements_collected": False,
    "public_released_measurements_used": True,
    "replacement_allowed": False,
    "target_outcomes_used": False,
}

_METHOD_PREDICTION_FIELDS = frozenset({"artifact_id", "predicted_loss_mm"})
_METHOD_OUTCOME_FIELDS = frozenset({"artifact_id", "loss_mm"})
_SEAL_FIELDS = frozenset(
    {
        "episode_id",
        "execution_lock_id",
        "factor_admitted",
        "implementation_revision",
        "information_boundary",
        "methods",
        "object_id",
        "outer_held_out_object_id",
        "physical_mode",
        "prediction_fit_artifact_id",
        "prediction_fit_object_ids",
        "prospective_policy_id",
        "record_role",
        "risk_score",
        "schema",
        "schema_version",
        "seal_id",
        "selection_sha256",
        "semantics",
        "source_artifacts",
        "stratum",
        "technical_failure",
    }
)
_FOLD_FIELDS = frozenset(
    {
        "execution_lock_id",
        "fold_prediction_seal_id",
        "held_out_object_id",
        "held_out_prediction_seal_id",
        "implementation_revision",
        "information_boundary",
        "schema",
        "schema_version",
        "training_prediction_seal_ids",
    }
)
_BATCH_FIELDS = frozenset(
    {
        "execution_lock_id",
        "fold_count",
        "folds",
        "implementation_revision",
        "information_boundary",
        "prediction_batch_id",
        "prospective_policy_id",
        "record_count",
        "records",
        "schema",
        "schema_version",
        "selection_sha256",
        "semantics",
    }
)
_OUTCOME_FIELDS = frozenset(
    {
        "episode_id",
        "execution_lock_id",
        "implementation_revision",
        "information_boundary",
        "methods",
        "object_id",
        "outcome_id",
        "outer_held_out_object_id",
        "prediction_batch_id",
        "prediction_seal_id",
        "prospective_policy_id",
        "record_role",
        "schema",
        "schema_version",
        "scoring_artifacts",
        "selection_sha256",
        "semantics",
        "stratum",
    }
)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _canonical_id(value: object, *, name: str) -> str:
    result = nonempty_string(value, name=name)
    if result.strip() != result or "\x00" in result:
        raise ValueError(f"{name} must be a canonical string")
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _cohort(lock: Mapping[str, Any]) -> dict[str, tuple[int, str]]:
    cohort = _mapping(lock.get("cohort"), name="cohort")
    rows = _sequence(cohort.get("development_objects"), name="development_objects")
    result: dict[str, tuple[int, str]] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, name=f"development_objects[{index}]")
        require_exact_fields(
            row,
            expected=frozenset({"episode_id", "object_id", "stratum"}),
            name=f"development_objects[{index}]",
        )
        object_id = _canonical_id(row.get("object_id"), name="object_id")
        if object_id in result:
            raise ValueError("execution lock repeats a development object")
        episode_id = genuine_integer(
            row.get("episode_id"), name="episode_id", minimum=0
        )
        stratum = _canonical_id(row.get("stratum"), name="stratum")
        if stratum not in {"sheet", "volumetric"}:
            raise ValueError("development object stratum changed")
        result[object_id] = (episode_id, stratum)
    if len(result) != 10:
        raise ValueError("execution lock must bind exactly ten development objects")
    return result


def _lock_ids(lock: Mapping[str, Any]) -> tuple[str, str, str]:
    execution_lock_id = sha256_digest(
        lock.get("execution_lock_id"), name="execution_lock_id"
    )
    policy = _mapping(lock.get("prospective_policy"), name="prospective_policy")
    cohort = _mapping(lock.get("cohort"), name="cohort")
    prospective_policy_id = sha256_digest(
        policy.get("policy_id"), name="prospective_policy_id"
    )
    selection_sha256 = sha256_digest(
        cohort.get("selection_sha256"), name="selection_sha256"
    )
    return execution_lock_id, prospective_policy_id, selection_sha256


def _expected_fit_ids(
    cohort: Mapping[str, tuple[int, str]],
    *,
    outer_held_out_object_id: str,
    object_id: str,
    role: RecordRole,
) -> tuple[str, ...]:
    if outer_held_out_object_id not in cohort:
        raise ValueError("outer held-out object is outside the locked cohort")
    if object_id not in cohort:
        raise ValueError("prediction object is outside the locked cohort")
    if role == "held_out":
        if object_id != outer_held_out_object_id:
            raise ValueError("held-out prediction object differs from its outer fold")
        excluded = {outer_held_out_object_id}
    else:
        if object_id == outer_held_out_object_id:
            raise ValueError("training prediction cannot be the outer held-out object")
        excluded = {outer_held_out_object_id, object_id}
    return tuple(sorted(set(cohort) - excluded))


def _prediction_methods(value: object, *, name: str) -> dict[str, dict[str, Any]]:
    methods = _mapping(value, name=name)
    require_exact_fields(
        methods,
        expected=frozenset(RAW_METHOD_IDS),
        name=name,
    )
    result: dict[str, dict[str, Any]] = {}
    for method_id in RAW_METHOD_IDS:
        method = _mapping(methods[method_id], name=f"{name}.{method_id}")
        require_exact_fields(
            method,
            expected=_METHOD_PREDICTION_FIELDS,
            name=f"{name}.{method_id}",
        )
        result[method_id] = {
            "artifact_id": sha256_digest(
                method.get("artifact_id"),
                name=f"{name}.{method_id}.artifact_id",
            ),
            "predicted_loss_mm": _finite_nonnegative(
                method.get("predicted_loss_mm"),
                name=f"{name}.{method_id}.predicted_loss_mm",
            ),
        }
    return result


def _outcome_methods(value: object, *, name: str) -> dict[str, dict[str, Any]]:
    methods = _mapping(value, name=name)
    require_exact_fields(
        methods,
        expected=frozenset(RAW_METHOD_IDS),
        name=name,
    )
    result: dict[str, dict[str, Any]] = {}
    for method_id in RAW_METHOD_IDS:
        method = _mapping(methods[method_id], name=f"{name}.{method_id}")
        require_exact_fields(
            method,
            expected=_METHOD_OUTCOME_FIELDS,
            name=f"{name}.{method_id}",
        )
        result[method_id] = {
            "artifact_id": sha256_digest(
                method.get("artifact_id"),
                name=f"{name}.{method_id}.artifact_id",
            ),
            "loss_mm": _finite_nonnegative(
                method.get("loss_mm"), name=f"{name}.{method_id}.loss_mm"
            ),
        }
    return result


def build_deform360_joint_sparse_source_prediction_seal_v5(
    *,
    lock: Mapping[str, Any],
    implementation_revision: str,
    outer_held_out_object_id: str,
    record_role: RecordRole,
    object_id: str,
    factor_admitted: bool,
    technical_failure: bool,
    physical_mode: str,
    risk_score: float,
    prediction_fit_artifact_id: str,
    prediction_fit_object_ids: Sequence[str],
    methods: Mapping[str, Any],
    source_artifacts: Mapping[str, str],
) -> dict[str, Any]:
    """Build one outcome-free outer or inner source forecast seal."""

    cohort = _cohort(lock)
    execution_lock_id, policy_id, selection_sha256 = _lock_ids(lock)
    outer_id = _canonical_id(
        outer_held_out_object_id, name="outer_held_out_object_id"
    )
    target_id = _canonical_id(object_id, name="object_id")
    if record_role not in {"held_out", "training"}:
        raise ValueError("record_role changed")
    role = cast(RecordRole, record_role)
    expected_fit = _expected_fit_ids(
        cohort,
        outer_held_out_object_id=outer_id,
        object_id=target_id,
        role=role,
    )
    supplied_fit = canonical_sorted_strings(
        prediction_fit_object_ids,
        name="prediction_fit_object_ids",
    )
    if supplied_fit != expected_fit:
        raise ValueError("prediction fit roster differs from the frozen nested fold")
    admitted = genuine_boolean(factor_admitted, name="factor_admitted")
    failed = genuine_boolean(technical_failure, name="technical_failure")
    if admitted and failed:
        raise ValueError("a technical failure cannot be factor-admitted")
    mode = _canonical_id(physical_mode, name="physical_mode")
    if mode not in {"warp_twin", "persistence_fallback"}:
        raise ValueError("physical_mode changed")
    episode_id, stratum = cohort[target_id]
    identity: dict[str, Any] = {
        "schema": SOURCE_PREDICTION_SEAL_SCHEMA,
        "schema_version": SOURCE_PREDICTION_SEAL_VERSION,
        "semantics": SOURCE_PREDICTION_SEAL_SEMANTICS,
        "execution_lock_id": execution_lock_id,
        "prospective_policy_id": policy_id,
        "selection_sha256": selection_sha256,
        "implementation_revision": exact_revision(
            implementation_revision, name="implementation_revision"
        ),
        "outer_held_out_object_id": outer_id,
        "record_role": role,
        "object_id": target_id,
        "episode_id": episode_id,
        "stratum": stratum,
        "factor_admitted": admitted,
        "technical_failure": failed,
        "physical_mode": mode,
        "risk_score": _finite_nonnegative(risk_score, name="risk_score"),
        "prediction_fit_artifact_id": sha256_digest(
            prediction_fit_artifact_id, name="prediction_fit_artifact_id"
        ),
        "prediction_fit_object_ids": list(supplied_fit),
        "methods": _prediction_methods(methods, name="methods"),
        "source_artifacts": plain_json(
            source_artifact_mapping(
                source_artifacts,
                name="source_artifacts",
            )
        ),
        "information_boundary": dict(_PREDICTION_BOUNDARY),
    }
    return {**identity, "seal_id": content_id(identity)}


def validate_deform360_joint_sparse_source_prediction_seal_v5(
    value: object,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and canonically rebuild one source prediction seal."""

    payload = _mapping(value, name="prediction seal")
    require_exact_fields(payload, expected=_SEAL_FIELDS, name="prediction seal")
    if payload.get("schema") != SOURCE_PREDICTION_SEAL_SCHEMA:
        raise ValueError("prediction seal schema changed")
    if payload.get("schema_version") != SOURCE_PREDICTION_SEAL_VERSION:
        raise ValueError("prediction seal version changed")
    if payload.get("semantics") != SOURCE_PREDICTION_SEAL_SEMANTICS:
        raise ValueError("prediction seal semantics changed")
    if payload.get("information_boundary") != _PREDICTION_BOUNDARY:
        raise ValueError("prediction seal information boundary changed")
    rebuilt = build_deform360_joint_sparse_source_prediction_seal_v5(
        lock=lock,
        implementation_revision=cast(str, payload.get("implementation_revision")),
        outer_held_out_object_id=cast(
            str, payload.get("outer_held_out_object_id")
        ),
        record_role=cast(RecordRole, payload.get("record_role")),
        object_id=cast(str, payload.get("object_id")),
        factor_admitted=cast(bool, payload.get("factor_admitted")),
        technical_failure=cast(bool, payload.get("technical_failure")),
        physical_mode=cast(str, payload.get("physical_mode")),
        risk_score=cast(float, payload.get("risk_score")),
        prediction_fit_artifact_id=cast(
            str, payload.get("prediction_fit_artifact_id")
        ),
        prediction_fit_object_ids=cast(
            Sequence[str], payload.get("prediction_fit_object_ids")
        ),
        methods=cast(Mapping[str, Any], payload.get("methods")),
        source_artifacts=cast(Mapping[str, str], payload.get("source_artifacts")),
    )
    if plain_json(payload) != rebuilt:
        raise ValueError("prediction seal content identity changed")
    return rebuilt


def _fold_identity(
    *,
    execution_lock_id: str,
    implementation_revision: str,
    held_out_object_id: str,
    held_out_prediction_seal_id: str,
    training_prediction_seal_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": SOURCE_FOLD_PREDICTION_SEAL_SCHEMA,
        "schema_version": 1,
        "execution_lock_id": execution_lock_id,
        "implementation_revision": implementation_revision,
        "held_out_object_id": held_out_object_id,
        "held_out_prediction_seal_id": held_out_prediction_seal_id,
        "training_prediction_seal_ids": list(training_prediction_seal_ids),
        "information_boundary": dict(_PREDICTION_BOUNDARY),
    }


def build_deform360_joint_sparse_source_prediction_batch_v5(
    seals: Sequence[Mapping[str, Any]],
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the complete 10 outer plus 90 inner forecasts before scoring."""

    cohort = _cohort(lock)
    execution_lock_id, policy_id, selection_sha256 = _lock_ids(lock)
    records = [
        validate_deform360_joint_sparse_source_prediction_seal_v5(seal, lock)
        for seal in seals
    ]
    if len(records) != 100:
        raise ValueError("prediction batch must contain exactly 100 nested forecasts")
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    revisions: set[str] = set()
    comparator_methods: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        pair = (
            cast(str, record["outer_held_out_object_id"]),
            cast(str, record["object_id"]),
        )
        if pair in by_pair:
            raise ValueError(f"prediction batch repeats a nested forecast: {pair}")
        by_pair[pair] = record
        revisions.add(cast(str, record["implementation_revision"]))
        for method_id in ("B0_physical_fallback", "B1_last_causal_residual"):
            key = (pair[1], method_id)
            forecast = cast(Mapping[str, Any], record["methods"])[method_id]
            previous = comparator_methods.setdefault(key, forecast)
            if previous != forecast:
                raise ValueError(
                    f"unchanged comparator prediction differs across folds: {key}"
                )
    if len(revisions) != 1:
        raise ValueError("prediction batch mixes implementation revisions")
    expected_pairs = {(outer, object_id) for outer in cohort for object_id in cohort}
    if set(by_pair) != expected_pairs:
        raise ValueError("prediction batch differs from the exact nested-fold roster")
    revision = next(iter(revisions))
    sorted_records = [by_pair[pair] for pair in sorted(by_pair)]
    folds: list[dict[str, Any]] = []
    for held_out_id in sorted(cohort):
        held_out = by_pair[(held_out_id, held_out_id)]
        training = [
            by_pair[(held_out_id, object_id)]
            for object_id in sorted(set(cohort) - {held_out_id})
        ]
        training_ids = [cast(str, item["seal_id"]) for item in training]
        fold_identity = _fold_identity(
            execution_lock_id=execution_lock_id,
            implementation_revision=revision,
            held_out_object_id=held_out_id,
            held_out_prediction_seal_id=cast(str, held_out["seal_id"]),
            training_prediction_seal_ids=training_ids,
        )
        folds.append(
            {**fold_identity, "fold_prediction_seal_id": content_id(fold_identity)}
        )
    identity: dict[str, Any] = {
        "schema": SOURCE_PREDICTION_BATCH_SCHEMA,
        "schema_version": SOURCE_PREDICTION_BATCH_VERSION,
        "semantics": SOURCE_PREDICTION_BATCH_SEMANTICS,
        "execution_lock_id": execution_lock_id,
        "prospective_policy_id": policy_id,
        "selection_sha256": selection_sha256,
        "implementation_revision": revision,
        "record_count": len(sorted_records),
        "fold_count": len(folds),
        "records": sorted_records,
        "folds": folds,
        "information_boundary": dict(_PREDICTION_BOUNDARY),
    }
    return {**identity, "prediction_batch_id": content_id(identity)}


def validate_deform360_joint_sparse_source_prediction_batch_v5(
    value: object,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one complete, outcome-free nested prediction batch."""

    payload = _mapping(value, name="prediction batch")
    require_exact_fields(payload, expected=_BATCH_FIELDS, name="prediction batch")
    if payload.get("schema") != SOURCE_PREDICTION_BATCH_SCHEMA:
        raise ValueError("prediction batch schema changed")
    if payload.get("schema_version") != SOURCE_PREDICTION_BATCH_VERSION:
        raise ValueError("prediction batch version changed")
    if payload.get("semantics") != SOURCE_PREDICTION_BATCH_SEMANTICS:
        raise ValueError("prediction batch semantics changed")
    if payload.get("information_boundary") != _PREDICTION_BOUNDARY:
        raise ValueError("prediction batch information boundary changed")
    records = [
        _mapping(item, name=f"records[{index}]")
        for index, item in enumerate(_sequence(payload.get("records"), name="records"))
    ]
    rebuilt = build_deform360_joint_sparse_source_prediction_batch_v5(records, lock)
    if plain_json(payload) != rebuilt:
        raise ValueError("prediction batch content identity changed")
    for index, raw in enumerate(_sequence(payload.get("folds"), name="folds")):
        fold = _mapping(raw, name=f"folds[{index}]")
        require_exact_fields(fold, expected=_FOLD_FIELDS, name=f"folds[{index}]")
    return rebuilt


def _seal_by_id(batch: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        cast(str, record["seal_id"]): record
        for record in cast(Sequence[Mapping[str, Any]], batch["records"])
    }


def _build_source_outcome_from_validated_batch(
    *,
    batch: Mapping[str, Any],
    prediction_seal_id: str,
    methods: Mapping[str, Any],
    scoring_artifacts: Mapping[str, str],
) -> dict[str, Any]:
    seal_id = sha256_digest(prediction_seal_id, name="prediction_seal_id")
    seals = _seal_by_id(batch)
    if seal_id not in seals:
        raise ValueError("outcome refers to a seal outside the prediction batch")
    seal = seals[seal_id]
    outcome_methods = _outcome_methods(methods, name="methods")
    predictions = cast(Mapping[str, Mapping[str, Any]], seal["methods"])
    for method_id in RAW_METHOD_IDS:
        if outcome_methods[method_id]["artifact_id"] != predictions[method_id][
            "artifact_id"
        ]:
            raise ValueError(f"outcome method artifact differs from seal: {method_id}")
    identity: dict[str, Any] = {
        "schema": SOURCE_OUTCOME_SCHEMA,
        "schema_version": SOURCE_OUTCOME_VERSION,
        "semantics": SOURCE_OUTCOME_SEMANTICS,
        "execution_lock_id": batch["execution_lock_id"],
        "prospective_policy_id": batch["prospective_policy_id"],
        "selection_sha256": batch["selection_sha256"],
        "implementation_revision": batch["implementation_revision"],
        "prediction_batch_id": batch["prediction_batch_id"],
        "prediction_seal_id": seal_id,
        "outer_held_out_object_id": seal["outer_held_out_object_id"],
        "record_role": seal["record_role"],
        "object_id": seal["object_id"],
        "episode_id": seal["episode_id"],
        "stratum": seal["stratum"],
        "methods": outcome_methods,
        "scoring_artifacts": plain_json(
            source_artifact_mapping(
                scoring_artifacts,
                name="scoring_artifacts",
            )
        ),
        "information_boundary": dict(_OUTCOME_BOUNDARY),
    }
    return {**identity, "outcome_id": content_id(identity)}


def build_deform360_joint_sparse_source_outcome_v5(
    *,
    lock: Mapping[str, Any],
    prediction_batch: Mapping[str, Any],
    prediction_seal_id: str,
    methods: Mapping[str, Any],
    scoring_artifacts: Mapping[str, str],
) -> dict[str, Any]:
    """Bind development-suffix method losses to one pre-existing forecast seal."""

    batch = validate_deform360_joint_sparse_source_prediction_batch_v5(
        prediction_batch, lock
    )
    return _build_source_outcome_from_validated_batch(
        batch=batch,
        prediction_seal_id=prediction_seal_id,
        methods=methods,
        scoring_artifacts=scoring_artifacts,
    )


def build_deform360_joint_sparse_source_outcomes_v5(
    *,
    lock: Mapping[str, Any],
    prediction_batch: Mapping[str, Any],
    methods_by_prediction_seal_id: Mapping[str, Mapping[str, Any]],
    scoring_artifacts_by_prediction_seal_id: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Build the complete scored panel while validating the batch only once."""

    batch = validate_deform360_joint_sparse_source_prediction_batch_v5(
        prediction_batch, lock
    )
    seal_ids = set(_seal_by_id(batch))
    if set(methods_by_prediction_seal_id) != seal_ids:
        raise ValueError("scored method roster differs from the prediction batch")
    if set(scoring_artifacts_by_prediction_seal_id) != seal_ids:
        raise ValueError("scoring-artifact roster differs from the prediction batch")
    return [
        _build_source_outcome_from_validated_batch(
            batch=batch,
            prediction_seal_id=seal_id,
            methods=methods_by_prediction_seal_id[seal_id],
            scoring_artifacts=scoring_artifacts_by_prediction_seal_id[seal_id],
        )
        for seal_id in sorted(seal_ids)
    ]


def _validate_source_outcome_against_validated_batch(
    value: object,
    *,
    prediction_batch: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _mapping(value, name="source outcome")
    require_exact_fields(payload, expected=_OUTCOME_FIELDS, name="source outcome")
    if payload.get("schema") != SOURCE_OUTCOME_SCHEMA:
        raise ValueError("source outcome schema changed")
    if payload.get("schema_version") != SOURCE_OUTCOME_VERSION:
        raise ValueError("source outcome version changed")
    if payload.get("semantics") != SOURCE_OUTCOME_SEMANTICS:
        raise ValueError("source outcome semantics changed")
    if payload.get("information_boundary") != _OUTCOME_BOUNDARY:
        raise ValueError("source outcome information boundary changed")
    if payload.get("prediction_batch_id") != prediction_batch.get(
        "prediction_batch_id"
    ):
        raise ValueError("source outcome refers to another prediction batch")
    rebuilt = _build_source_outcome_from_validated_batch(
        batch=prediction_batch,
        prediction_seal_id=cast(str, payload.get("prediction_seal_id")),
        methods=cast(Mapping[str, Any], payload.get("methods")),
        scoring_artifacts=cast(
            Mapping[str, str], payload.get("scoring_artifacts")
        ),
    )
    if plain_json(payload) != rebuilt:
        raise ValueError("source outcome content identity changed")
    return rebuilt


def validate_deform360_joint_sparse_source_outcome_v5(
    value: object,
    *,
    lock: Mapping[str, Any],
    prediction_batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and canonically rebuild one source suffix outcome."""

    batch = validate_deform360_joint_sparse_source_prediction_batch_v5(
        prediction_batch, lock
    )
    return _validate_source_outcome_against_validated_batch(
        value,
        prediction_batch=batch,
    )


def _source_record(
    seal: Mapping[str, Any],
    outcome: Mapping[str, Any],
) -> dict[str, Any]:
    predictions = cast(Mapping[str, Mapping[str, Any]], seal["methods"])
    losses = cast(Mapping[str, Mapping[str, Any]], outcome["methods"])
    return {
        "object_id": seal["object_id"],
        "episode_id": seal["episode_id"],
        "stratum": seal["stratum"],
        "factor_admitted": seal["factor_admitted"],
        "technical_failure": seal["technical_failure"],
        "physical_mode": seal["physical_mode"],
        "risk_score": seal["risk_score"],
        "prediction_fit_artifact_id": seal["prediction_fit_artifact_id"],
        "prediction_fit_object_ids": seal["prediction_fit_object_ids"],
        "methods": {
            method_id: {
                "artifact_id": predictions[method_id]["artifact_id"],
                "predicted_loss_mm": predictions[method_id]["predicted_loss_mm"],
                "loss_mm": losses[method_id]["loss_mm"],
            }
            for method_id in RAW_METHOD_IDS
        },
    }


def assemble_deform360_joint_sparse_source_evidence_v5(
    *,
    lock: Mapping[str, Any],
    prediction_batch: Mapping[str, Any],
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble the evaluator input without permitting manual JSON substitution."""

    batch = validate_deform360_joint_sparse_source_prediction_batch_v5(
        prediction_batch, lock
    )
    validated = [
        _validate_source_outcome_against_validated_batch(
            outcome,
            prediction_batch=batch,
        )
        for outcome in outcomes
    ]
    if len(validated) != 100:
        raise ValueError("source evidence requires exactly 100 scored outcomes")
    by_seal: dict[str, dict[str, Any]] = {}
    for outcome in validated:
        seal_id = cast(str, outcome["prediction_seal_id"])
        if seal_id in by_seal:
            raise ValueError(f"source evidence repeats an outcome: {seal_id}")
        by_seal[seal_id] = outcome
    seals = _seal_by_id(batch)
    if set(by_seal) != set(seals):
        raise ValueError("source outcomes differ from the sealed prediction roster")

    folds: list[dict[str, Any]] = []
    for fold in cast(Sequence[Mapping[str, Any]], batch["folds"]):
        held_id = cast(str, fold["held_out_prediction_seal_id"])
        training_ids = cast(Sequence[str], fold["training_prediction_seal_ids"])
        held_seal = seals[held_id]
        folds.append(
            {
                "fold_prediction_seal_id": fold["fold_prediction_seal_id"],
                "held_out_object_id": fold["held_out_object_id"],
                "held_out_record": _source_record(
                    held_seal,
                    by_seal[held_id],
                ),
                "training_records": [
                    _source_record(seals[seal_id], by_seal[seal_id])
                    for seal_id in training_ids
                ],
            }
        )
    identity: dict[str, Any] = {
        "schema": SOURCE_EVIDENCE_SCHEMA,
        "schema_version": SOURCE_EVIDENCE_VERSION,
        "semantics": SOURCE_EVIDENCE_SEMANTICS,
        "execution_lock_id": batch["execution_lock_id"],
        "implementation_revision": batch["implementation_revision"],
        "information_boundary": dict(_EVIDENCE_BOUNDARY),
        "prediction_batch_id": batch["prediction_batch_id"],
        "prospective_policy_id": batch["prospective_policy_id"],
        "folds": folds,
        "selection_sha256": batch["selection_sha256"],
    }
    evidence = {"evidence_id": content_id(identity), **identity}
    parse_deform360_joint_sparse_source_evidence_v5(evidence, lock)
    return evidence


def publish_deform360_joint_sparse_source_prediction_batch_v5(
    batch: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Publish one validated prediction batch atomically without replacement."""

    validated = validate_deform360_joint_sparse_source_prediction_batch_v5(
        batch, lock
    )
    write_atomic_json(validated, output_path, overwrite=False)
    return validated


def publish_deform360_joint_sparse_source_evidence_v5(
    evidence: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Publish one evaluator-ready evidence artifact without replacement."""

    parse_deform360_joint_sparse_source_evidence_v5(evidence, lock)
    validated = cast(dict[str, Any], plain_json(evidence))
    write_atomic_json(validated, output_path, overwrite=False)
    return validated


def load_source_execution_lock_and_artifacts_v5(
    *,
    execution_lock_path: str | Path,
    artifact_paths: Sequence[str | Path],
    label: str,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    """Load a validated execution lock and a nonempty list of strict JSON files."""

    lock = load_deform360_joint_sparse_source_execution_lock_v5(
        execution_lock_path
    )
    if not artifact_paths:
        raise ValueError(f"{label} paths must not be empty")
    artifacts = [
        load_strict_json_object(path, label=f"{label} artifact")
        for path in artifact_paths
    ]
    return lock, artifacts


__all__ = [
    "SOURCE_OUTCOME_SCHEMA",
    "SOURCE_PREDICTION_BATCH_SCHEMA",
    "SOURCE_PREDICTION_SEAL_SCHEMA",
    "assemble_deform360_joint_sparse_source_evidence_v5",
    "build_deform360_joint_sparse_source_outcome_v5",
    "build_deform360_joint_sparse_source_outcomes_v5",
    "build_deform360_joint_sparse_source_prediction_batch_v5",
    "build_deform360_joint_sparse_source_prediction_seal_v5",
    "load_source_execution_lock_and_artifacts_v5",
    "publish_deform360_joint_sparse_source_evidence_v5",
    "publish_deform360_joint_sparse_source_prediction_batch_v5",
    "validate_deform360_joint_sparse_source_outcome_v5",
    "validate_deform360_joint_sparse_source_prediction_batch_v5",
    "validate_deform360_joint_sparse_source_prediction_seal_v5",
]
