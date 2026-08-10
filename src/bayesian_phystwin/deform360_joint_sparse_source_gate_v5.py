"""Frozen source gate for the public Deform360 joint-sparse v5 experiment.

The gate consumes predictions and risk scores sealed before development suffixes
were scored. Each outer fold selects its guard from nine inner cross-fitted
physical-object forecasts, preserves complete risk-score ties, and deploys the
exact physical fallback whenever the candidate is rejected. Confirmation data
are neither needed nor permitted here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import genuine_boolean, genuine_integer, plain_json
from ._portable_contracts import (
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)

EXECUTION_LOCK_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-source-execution-lock"
)
EXECUTION_LOCK_VERSION: Final = 1
EXECUTION_LOCK_SEMANTICS: Final = (
    "public-real-world-nested-source-gate-before-confirmation-v1"
)
SOURCE_EVIDENCE_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-source-evidence"
)
SOURCE_EVIDENCE_VERSION: Final = 1
SOURCE_EVIDENCE_SEMANTICS: Final = (
    "prediction-sealed-public-development-suffix-evidence-v1"
)
SOURCE_RESULT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-source-gate-result"
)
SOURCE_RESULT_VERSION: Final = 1
SOURCE_RESULT_SEMANTICS: Final = "nested-loo-public-source-gate-before-confirmation-v1"

PROSPECTIVE_POLICY_ID: Final = (
    "0f2af7bf30576833d3f9e82d3cc4238da007d772325766aa456b374a0a254749"
)
SELECTION_SHA256: Final = (
    "b28daf8477e214cb74a4d250ef5eea8f9f1a014aec10487699ac0ce063961222"
)

RAW_METHOD_IDS: Final = (
    "B0_physical_fallback",
    "B1_last_causal_residual",
    "V1_joint_sparse_visual_guarded",
    "T1_contact_anchor_only",
    "VT2_joint_sparse_visuotactile_unguarded",
    "VT3_joint_sparse_visuotactile_anchor_bias",
)
PRIMARY_RAW_METHOD: Final = "VT2_joint_sparse_visuotactile_unguarded"
PRIMARY_DEPLOYED_METHOD: Final = "VT1_joint_sparse_visuotactile_guarded"

_LOCK_FIELDS = frozenset(
    {
        "claim_boundary",
        "cohort",
        "execution_lock_id",
        "information_boundary",
        "physical_baseline",
        "prospective_policy",
        "public_measurements",
        "schema",
        "schema_version",
        "semantics",
        "source_gate",
    }
)
_EVIDENCE_FIELDS = frozenset(
    {
        "evidence_id",
        "execution_lock_id",
        "implementation_revision",
        "information_boundary",
        "prediction_batch_id",
        "prospective_policy_id",
        "folds",
        "schema",
        "schema_version",
        "selection_sha256",
        "semantics",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "episode_id",
        "factor_admitted",
        "methods",
        "object_id",
        "physical_mode",
        "prediction_fit_artifact_id",
        "prediction_fit_object_ids",
        "risk_score",
        "stratum",
        "technical_failure",
    }
)
_METHOD_FIELDS = frozenset({"artifact_id", "loss_mm", "predicted_loss_mm"})
_FOLD_FIELDS = frozenset(
    {
        "fold_prediction_seal_id",
        "held_out_object_id",
        "held_out_record",
        "training_records",
    }
)
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


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _open_probability(value: object, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie inside (0, 1)")
    return result


def _canonical_id(value: object, *, name: str) -> str:
    result = nonempty_string(value, name=name)
    if result.strip() != result or "\x00" in result:
        raise ValueError(f"{name} must be a canonical string")
    return result


@dataclass(frozen=True, slots=True)
class _MethodForecast:
    artifact_id: str
    loss_mm: float
    predicted_loss_mm: float


@dataclass(frozen=True, slots=True)
class _SourceRecord:
    object_id: str
    episode_id: int
    stratum: str
    factor_admitted: bool
    technical_failure: bool
    risk_score: float
    physical_mode: str
    prediction_fit_artifact_id: str
    prediction_fit_object_ids: tuple[str, ...]
    methods: Mapping[str, _MethodForecast]


@dataclass(frozen=True, slots=True)
class _SourceFold:
    held_out: _SourceRecord
    training: tuple[_SourceRecord, ...]
    fold_prediction_seal_id: str


def load_deform360_joint_sparse_source_execution_lock_v5(
    path: str | Path,
) -> Mapping[str, Any]:
    """Load and validate the additive v5 source-execution lock."""

    lock = load_strict_json_object(path, label="v5 source-execution lock")
    require_exact_fields(lock, expected=_LOCK_FIELDS, name="execution lock")
    if lock.get("schema") != EXECUTION_LOCK_SCHEMA:
        raise ValueError("execution lock schema changed")
    if lock.get("schema_version") != EXECUTION_LOCK_VERSION:
        raise ValueError("execution lock version changed")
    if lock.get("semantics") != EXECUTION_LOCK_SEMANTICS:
        raise ValueError("execution lock semantics changed")
    declared = sha256_digest(lock.get("execution_lock_id"), name="execution_lock_id")
    identity = {key: value for key, value in lock.items() if key != "execution_lock_id"}
    if declared != content_id(identity):
        raise ValueError("execution_lock_id does not match the lock content")

    measurements = _mapping(lock.get("public_measurements"), name="public_measurements")
    if measurements.get("released_real_world_recordings") is not True:
        raise ValueError("execution lock must use released real-world recordings")
    if measurements.get("new_measurements_required") is not False:
        raise ValueError("execution lock unexpectedly requires new measurements")
    if measurements.get("human_approval_required") is not False:
        raise ValueError("execution lock unexpectedly requires human approval")
    if measurements.get("prob4d_role") != (
        "used-as-the-frozen-probabilistic-visual-observation-feeder"
    ):
        raise ValueError("execution lock changed the Prob4D role")

    policy = _mapping(lock.get("prospective_policy"), name="prospective_policy")
    cohort = _mapping(lock.get("cohort"), name="cohort")
    if policy.get("policy_id") != PROSPECTIVE_POLICY_ID:
        raise ValueError("execution lock changed the prospective policy")
    if cohort.get("selection_sha256") != SELECTION_SHA256:
        raise ValueError("execution lock changed the selected cohort")
    baseline = _mapping(lock.get("physical_baseline"), name="physical_baseline")
    if baseline.get("generation_rule") != (
        "automatic-warp-twin-when-admissible-otherwise-exact-persistence-v1"
    ):
        raise ValueError("execution lock changed the physical baseline")
    gate = _mapping(lock.get("source_gate"), name="source_gate")
    expected_gate_values = {
        "harmful_update_relative_margin": 0.02,
        "maximum_risk_coverage": 0.98,
        "maximum_stratum_mean_regression": 0.02,
        "minimum_contact_increment_over_visual_only": 0.02,
        "minimum_passing_objects": 8,
        "minimum_passing_objects_per_stratum": 4,
        "minimum_relative_improvement_vs_last_causal_residual": 0.05,
        "minimum_relative_improvement_vs_physical_fallback": 0.1,
        "minimum_risk_coverage": 0.8,
        "nominal_conformal_coverage": 0.9,
        "risk_score_semantics": "lower-is-safer-inclusive-threshold-v1",
        "tie_policy": "accept-complete-tied-score-blocks-never-split-by-object-id",
    }
    if any(gate.get(key) != value for key, value in expected_gate_values.items()):
        raise ValueError("execution lock changed a source-gate decision value")

    boundary = _mapping(lock.get("information_boundary"), name="information_boundary")
    if boundary.get("confirmation_payloads_opened") is not False:
        raise ValueError("execution lock crosses the confirmation boundary")
    if boundary.get("target_outcomes_used") is not False:
        raise ValueError("execution lock uses target outcomes")
    return lock


def _parse_method(value: object, *, name: str) -> _MethodForecast:
    payload = _mapping(value, name=name)
    require_exact_fields(payload, expected=_METHOD_FIELDS, name=name)
    return _MethodForecast(
        artifact_id=sha256_digest(
            payload.get("artifact_id"), name=f"{name}.artifact_id"
        ),
        loss_mm=_finite_real(
            payload.get("loss_mm"), name=f"{name}.loss_mm", minimum=0.0
        ),
        predicted_loss_mm=_finite_real(
            payload.get("predicted_loss_mm"),
            name=f"{name}.predicted_loss_mm",
            minimum=0.0,
        ),
    )


def _expected_cohort(lock: Mapping[str, Any]) -> dict[str, tuple[int, str]]:
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
        episode = genuine_integer(row.get("episode_id"), name="episode_id", minimum=0)
        stratum = _canonical_id(row.get("stratum"), name="stratum")
        if stratum not in {"sheet", "volumetric"}:
            raise ValueError("development object stratum changed")
        result[object_id] = (episode, stratum)
    if len(result) != 10:
        raise ValueError("execution lock must bind exactly ten development objects")
    return result


def _parse_records(
    raw_records: object,
    *,
    expected: Mapping[str, tuple[int, str]],
    name: str,
) -> tuple[_SourceRecord, ...]:
    rows = _sequence(raw_records, name=name)
    records: list[_SourceRecord] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        record_name = f"{name}[{index}]"
        payload = _mapping(raw, name=record_name)
        require_exact_fields(payload, expected=_RECORD_FIELDS, name=record_name)
        object_id = _canonical_id(payload.get("object_id"), name="object_id")
        if object_id in seen:
            raise ValueError(f"duplicate source object: {object_id}")
        seen.add(object_id)
        if object_id not in expected:
            raise ValueError(f"unregistered source object: {object_id}")
        episode_id = genuine_integer(
            payload.get("episode_id"), name="episode_id", minimum=0
        )
        stratum = _canonical_id(payload.get("stratum"), name="stratum")
        if (episode_id, stratum) != expected[object_id]:
            raise ValueError(f"source identity changed: {object_id}")

        methods_payload = _mapping(payload.get("methods"), name="methods")
        require_exact_fields(
            methods_payload,
            expected=frozenset(RAW_METHOD_IDS),
            name=f"{object_id}.methods",
        )
        methods = {
            method_id: _parse_method(
                methods_payload[method_id], name=f"{object_id}.{method_id}"
            )
            for method_id in RAW_METHOD_IDS
        }
        factor_admitted = genuine_boolean(
            payload.get("factor_admitted"), name="factor_admitted"
        )
        technical_failure = genuine_boolean(
            payload.get("technical_failure"), name="technical_failure"
        )
        if technical_failure and factor_admitted:
            raise ValueError("a technical failure cannot be factor-admitted")
        physical_mode = _canonical_id(
            payload.get("physical_mode"), name="physical_mode"
        )
        if physical_mode not in {"warp_twin", "persistence_fallback"}:
            raise ValueError("physical_mode changed")
        prediction_fit_object_ids = tuple(
            sorted(
                _canonical_id(item, name="prediction_fit_object_id")
                for item in _sequence(
                    payload.get("prediction_fit_object_ids"),
                    name="prediction_fit_object_ids",
                )
            )
        )
        if len(set(prediction_fit_object_ids)) != len(prediction_fit_object_ids):
            raise ValueError("prediction_fit_object_ids contains duplicates")
        records.append(
            _SourceRecord(
                object_id=object_id,
                episode_id=episode_id,
                stratum=stratum,
                factor_admitted=factor_admitted,
                technical_failure=technical_failure,
                risk_score=_finite_real(
                    payload.get("risk_score"), name="risk_score", minimum=0.0
                ),
                physical_mode=physical_mode,
                prediction_fit_artifact_id=sha256_digest(
                    payload.get("prediction_fit_artifact_id"),
                    name="prediction_fit_artifact_id",
                ),
                prediction_fit_object_ids=prediction_fit_object_ids,
                methods=methods,
            )
        )
    return tuple(sorted(records, key=lambda record: record.object_id))


def _parse_folds(
    evidence: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> tuple[_SourceFold, ...]:
    expected = _expected_cohort(lock)
    raw_folds = _sequence(evidence.get("folds"), name="folds")
    folds: list[_SourceFold] = []
    seen_held_out: set[str] = set()
    comparator_forecasts: dict[tuple[str, str], _MethodForecast] = {}
    for index, raw in enumerate(raw_folds):
        fold_name = f"folds[{index}]"
        payload = _mapping(raw, name=fold_name)
        require_exact_fields(payload, expected=_FOLD_FIELDS, name=fold_name)
        held_out_id = _canonical_id(
            payload.get("held_out_object_id"), name="held_out_object_id"
        )
        if held_out_id in seen_held_out:
            raise ValueError(f"duplicate held-out source object: {held_out_id}")
        if held_out_id not in expected:
            raise ValueError(f"unregistered held-out source object: {held_out_id}")
        seen_held_out.add(held_out_id)

        held_rows = _parse_records(
            [payload.get("held_out_record")],
            expected=expected,
            name=f"{fold_name}.held_out_record",
        )
        held_out = held_rows[0]
        if held_out.object_id != held_out_id:
            raise ValueError("held-out record identity differs from its fold")
        training = _parse_records(
            payload.get("training_records"),
            expected=expected,
            name=f"{fold_name}.training_records",
        )
        expected_training = set(expected) - {held_out_id}
        if {record.object_id for record in training} != expected_training:
            raise ValueError("fold does not contain the exact nine training objects")
        if set(held_out.prediction_fit_object_ids) != expected_training:
            raise ValueError(
                "held-out prediction was not fit on the other nine objects"
            )
        for record in training:
            expected_inner_fit = expected_training - {record.object_id}
            if set(record.prediction_fit_object_ids) != expected_inner_fit:
                raise ValueError(
                    "training forecast is not inner cross-fitted within its outer fold"
                )

        for record in (held_out, *training):
            for method_id in (
                "B0_physical_fallback",
                "B1_last_causal_residual",
            ):
                key = (record.object_id, method_id)
                forecast = record.methods[method_id]
                previous = comparator_forecasts.setdefault(key, forecast)
                if previous != forecast:
                    raise ValueError(
                        f"unchanged comparator differs across folds: {key}"
                    )
        folds.append(
            _SourceFold(
                held_out=held_out,
                training=training,
                fold_prediction_seal_id=sha256_digest(
                    payload.get("fold_prediction_seal_id"),
                    name="fold_prediction_seal_id",
                ),
            )
        )
    if seen_held_out != set(expected):
        raise ValueError("source evidence does not contain the exact ten outer folds")
    return tuple(sorted(folds, key=lambda fold: fold.held_out.object_id))


def parse_deform360_joint_sparse_source_evidence_v5(
    payload: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> tuple[_SourceFold, ...]:
    """Validate one complete, prediction-sealed source evidence artifact."""

    require_exact_fields(payload, expected=_EVIDENCE_FIELDS, name="source evidence")
    if payload.get("schema") != SOURCE_EVIDENCE_SCHEMA:
        raise ValueError("source evidence schema changed")
    if payload.get("schema_version") != SOURCE_EVIDENCE_VERSION:
        raise ValueError("source evidence version changed")
    if payload.get("semantics") != SOURCE_EVIDENCE_SEMANTICS:
        raise ValueError("source evidence semantics changed")
    evidence_id = sha256_digest(payload.get("evidence_id"), name="evidence_id")
    identity = {key: value for key, value in payload.items() if key != "evidence_id"}
    if evidence_id != content_id(identity):
        raise ValueError("evidence_id does not match the evidence content")
    if payload.get("execution_lock_id") != lock.get("execution_lock_id"):
        raise ValueError("source evidence uses another execution lock")
    policy = _mapping(lock.get("prospective_policy"), name="prospective_policy")
    cohort = _mapping(lock.get("cohort"), name="cohort")
    if payload.get("prospective_policy_id") != policy.get("policy_id"):
        raise ValueError("source evidence uses another prospective policy")
    if payload.get("selection_sha256") != cohort.get("selection_sha256"):
        raise ValueError("source evidence uses another cohort selection")
    exact_revision(
        payload.get("implementation_revision"), name="implementation_revision"
    )
    sha256_digest(payload.get("prediction_batch_id"), name="prediction_batch_id")
    boundary = plain_json(
        _mapping(payload.get("information_boundary"), name="information_boundary")
    )
    if boundary != _EVIDENCE_BOUNDARY:
        raise ValueError("source evidence crossed its registered information boundary")
    return _parse_folds(payload, lock)


def _relative_improvement(candidate: float, comparator: float) -> float:
    if comparator <= 0.0:
        return 0.0 if candidate == comparator else -math.inf
    return 1.0 - candidate / comparator


def _is_harmful(record: _SourceRecord, *, margin: float) -> bool:
    candidate = record.methods[PRIMARY_RAW_METHOD].loss_mm
    fallback = record.methods["B0_physical_fallback"].loss_mm
    return candidate > (1.0 + margin) * fallback


def _fit_threshold(
    records: Sequence[_SourceRecord],
    *,
    minimum_coverage: float,
    maximum_coverage: float,
    harm_margin: float,
) -> tuple[float, int, float] | None:
    eligible = [
        record
        for record in records
        if record.factor_admitted and not record.technical_failure
    ]
    candidates = sorted({record.risk_score for record in eligible})
    valid: list[tuple[int, float]] = []
    for threshold in candidates:
        accepted = [record for record in eligible if record.risk_score <= threshold]
        coverage = len(accepted) / len(records)
        if not minimum_coverage <= coverage <= maximum_coverage:
            continue
        if any(_is_harmful(record, margin=harm_margin) for record in accepted):
            continue
        valid.append((len(accepted), threshold))
    if not valid:
        return None
    accepted_count, threshold = max(valid, key=lambda item: (item[0], item[1]))
    return threshold, accepted_count, accepted_count / len(records)


def _accepted(record: _SourceRecord, threshold: float | None) -> bool:
    return bool(
        threshold is not None
        and record.factor_admitted
        and not record.technical_failure
        and record.risk_score <= threshold
    )


def _deployed_method(record: _SourceRecord, accepted: bool) -> _MethodForecast:
    method_id = PRIMARY_RAW_METHOD if accepted else "B0_physical_fallback"
    return record.methods[method_id]


def _interval(
    training: Sequence[_SourceRecord],
    held_out: _SourceRecord,
    *,
    threshold: float | None,
) -> tuple[float, float, bool]:
    if threshold is None:
        deployed = _deployed_method(held_out, False)
        return deployed.predicted_loss_mm, deployed.predicted_loss_mm, False
    scores: list[float] = []
    for record in training:
        forecast = _deployed_method(record, _accepted(record, threshold))
        scores.append(forecast.loss_mm - forecast.predicted_loss_mm)
    quantile = max(scores)
    held_forecast = _deployed_method(held_out, _accepted(held_out, threshold))
    upper = max(0.0, held_forecast.predicted_loss_mm + quantile)
    return quantile, upper, held_forecast.loss_mm <= upper


def _fold_result(
    held_out: _SourceRecord,
    training: Sequence[_SourceRecord],
    gate: Mapping[str, Any],
    *,
    fold_prediction_seal_id: str,
) -> dict[str, Any]:
    minimum_coverage = _open_probability(
        gate.get("minimum_risk_coverage"), name="minimum_risk_coverage"
    )
    maximum_coverage = _open_probability(
        gate.get("maximum_risk_coverage"), name="maximum_risk_coverage"
    )
    harm_margin = _finite_real(
        gate.get("harmful_update_relative_margin"),
        name="harmful_update_relative_margin",
        minimum=0.0,
    )
    fitted = _fit_threshold(
        training,
        minimum_coverage=minimum_coverage,
        maximum_coverage=maximum_coverage,
        harm_margin=harm_margin,
    )
    threshold = None if fitted is None else fitted[0]
    accepted = _accepted(held_out, threshold)
    deployed = _deployed_method(held_out, accepted)
    fallback = held_out.methods["B0_physical_fallback"]
    last_residual = held_out.methods["B1_last_causal_residual"]
    visual = held_out.methods["V1_joint_sparse_visual_guarded"]
    quantile, interval_upper, interval_covered = _interval(
        training, held_out, threshold=threshold
    )
    checks = {
        "threshold_available": fitted is not None,
        "candidate_accepted": accepted,
        "minimum_gain_vs_physical": _relative_improvement(
            deployed.loss_mm, fallback.loss_mm
        )
        >= float(gate["minimum_relative_improvement_vs_physical_fallback"]),
        "minimum_gain_vs_last_residual": _relative_improvement(
            deployed.loss_mm, last_residual.loss_mm
        )
        >= float(gate["minimum_relative_improvement_vs_last_causal_residual"]),
        "minimum_contact_increment": _relative_improvement(
            deployed.loss_mm, visual.loss_mm
        )
        >= float(gate["minimum_contact_increment_over_visual_only"]),
        "interval_covered": interval_covered,
        "accepted_update_not_harmful": (
            not accepted or deployed.loss_mm <= (1.0 + harm_margin) * fallback.loss_mm
        ),
        "rejection_is_exact_fallback": (
            accepted or deployed.artifact_id == fallback.artifact_id
        ),
    }
    return {
        "accepted": accepted,
        "checks": checks,
        "deployed_artifact_id": deployed.artifact_id,
        "deployed_loss_mm": deployed.loss_mm,
        "factor_admitted": held_out.factor_admitted,
        "fold_prediction_seal_id": fold_prediction_seal_id,
        "fold_passed": all(checks.values()),
        "interval_covered": interval_covered,
        "interval_upper_mm": interval_upper,
        "object_id": held_out.object_id,
        "physical_mode": held_out.physical_mode,
        "prediction_fit_artifact_id": held_out.prediction_fit_artifact_id,
        "risk_score": held_out.risk_score,
        "stratum": held_out.stratum,
        "technical_failure": held_out.technical_failure,
        "threshold": threshold,
        "training_accepted_count": None if fitted is None else fitted[1],
        "training_conformal_quantile_mm": quantile,
        "training_risk_coverage": None if fitted is None else fitted[2],
    }


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def evaluate_deform360_joint_sparse_source_gate_v5(
    evidence: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen ten-object source gate without confirmation access."""

    source_folds = parse_deform360_joint_sparse_source_evidence_v5(evidence, lock)
    gate = _mapping(lock.get("source_gate"), name="source_gate")
    folds = [
        _fold_result(
            source_fold.held_out,
            source_fold.training,
            gate,
            fold_prediction_seal_id=source_fold.fold_prediction_seal_id,
        )
        for source_fold in source_folds
    ]
    records = tuple(source_fold.held_out for source_fold in source_folds)
    deployed_losses = [float(fold["deployed_loss_mm"]) for fold in folds]
    fallback_losses = [
        record.methods["B0_physical_fallback"].loss_mm for record in records
    ]
    residual_losses = [
        record.methods["B1_last_causal_residual"].loss_mm for record in records
    ]
    visual_losses = [
        record.methods["V1_joint_sparse_visual_guarded"].loss_mm for record in records
    ]
    fold_by_id = {str(fold["object_id"]): fold for fold in folds}
    passing_by_stratum = {
        stratum: sum(
            bool(fold_by_id[record.object_id]["fold_passed"])
            for record in records
            if record.stratum == stratum
        )
        for stratum in ("sheet", "volumetric")
    }
    stratum_regression = {
        stratum: -_relative_improvement(
            _mean(
                [
                    float(fold_by_id[record.object_id]["deployed_loss_mm"])
                    for record in records
                    if record.stratum == stratum
                ]
            ),
            _mean(
                [
                    record.methods["B0_physical_fallback"].loss_mm
                    for record in records
                    if record.stratum == stratum
                ]
            ),
        )
        for stratum in ("sheet", "volumetric")
    }
    full_fit = _fit_threshold(
        records,
        minimum_coverage=float(gate["minimum_risk_coverage"]),
        maximum_coverage=float(gate["maximum_risk_coverage"]),
        harm_margin=float(gate["harmful_update_relative_margin"]),
    )
    full_threshold = None if full_fit is None else full_fit[0]
    full_scores: list[float] = []
    if full_threshold is not None:
        for record in records:
            forecast = _deployed_method(record, _accepted(record, full_threshold))
            full_scores.append(forecast.loss_mm - forecast.predicted_loss_mm)

    passing_count = sum(bool(fold["fold_passed"]) for fold in folds)
    accepted_count = sum(bool(fold["accepted"]) for fold in folds)
    gain_physical = _relative_improvement(
        _mean(deployed_losses), _mean(fallback_losses)
    )
    gain_residual = _relative_improvement(
        _mean(deployed_losses), _mean(residual_losses)
    )
    contact_increment = _relative_improvement(
        _mean(deployed_losses), _mean(visual_losses)
    )
    checks = {
        "minimum_passing_objects": passing_count
        >= int(gate["minimum_passing_objects"]),
        "minimum_passing_objects_per_stratum": all(
            count >= int(gate["minimum_passing_objects_per_stratum"])
            for count in passing_by_stratum.values()
        ),
        "minimum_accepted_objects": accepted_count
        >= int(gate["minimum_passing_objects"]),
        "aggregate_gain_vs_physical": gain_physical
        >= float(gate["minimum_relative_improvement_vs_physical_fallback"]),
        "aggregate_gain_vs_last_residual": gain_residual
        >= float(gate["minimum_relative_improvement_vs_last_causal_residual"]),
        "aggregate_contact_increment": contact_increment
        >= float(gate["minimum_contact_increment_over_visual_only"]),
        "no_harmful_accepted_update": all(
            bool(fold["checks"]["accepted_update_not_harmful"]) for fold in folds
        ),
        "maximum_stratum_mean_regression": all(
            value <= float(gate["maximum_stratum_mean_regression"])
            for value in stratum_regression.values()
        ),
        "all_rejections_are_exact_fallback": all(
            bool(fold["checks"]["rejection_is_exact_fallback"]) for fold in folds
        ),
        "full_source_refit_available": full_fit is not None,
    }
    gate_passed = all(checks.values())
    authorization: dict[str, Any] | None = None
    if gate_passed:
        authorization_body = {
            "schema": "bayesian-phystwin.deform360-confirmation-opening-authorization",
            "schema_version": 1,
            "execution_lock_id": lock["execution_lock_id"],
            "source_evidence_id": evidence["evidence_id"],
            "confirmation_payloads_opened": False,
            "authorized": True,
        }
        authorization = {
            "authorization_id": content_id(authorization_body),
            **authorization_body,
        }
    descriptor: dict[str, Any] = {
        "schema": SOURCE_RESULT_SCHEMA,
        "schema_version": SOURCE_RESULT_VERSION,
        "semantics": SOURCE_RESULT_SEMANTICS,
        "execution_lock_id": lock["execution_lock_id"],
        "source_evidence_id": evidence["evidence_id"],
        "implementation_revision": evidence["implementation_revision"],
        "folds": folds,
        "aggregate": {
            "accepted_count": accepted_count,
            "contact_increment_over_visual_only": contact_increment,
            "mean_deployed_loss_mm": _mean(deployed_losses),
            "mean_last_causal_residual_loss_mm": _mean(residual_losses),
            "mean_physical_fallback_loss_mm": _mean(fallback_losses),
            "passing_count": passing_count,
            "passing_count_by_stratum": passing_by_stratum,
            "relative_improvement_vs_last_causal_residual": gain_residual,
            "relative_improvement_vs_physical_fallback": gain_physical,
            "stratum_mean_regression_vs_physical": stratum_regression,
        },
        "full_source_fit": {
            "accepted_count": None if full_fit is None else full_fit[1],
            "conformal_quantile_mm": None if not full_scores else max(full_scores),
            "risk_coverage": None if full_fit is None else full_fit[2],
            "threshold": full_threshold,
        },
        "checks": checks,
        "gate_passed": gate_passed,
        "confirmation_access_authorized": gate_passed,
        "confirmation_opening_authorization": authorization,
        "status": "source-gate-passed" if gate_passed else "source-gate-failed",
        "information_boundary": {
            "confirmation_outcomes_opened": False,
            "confirmation_payloads_opened": False,
            "human_approval_required": False,
            "new_measurements_required": False,
            "public_released_measurements_used": True,
            "target_outcomes_used": False,
        },
        "claim_boundary": lock["claim_boundary"],
    }
    return {"result_id": content_id(descriptor), **plain_json(descriptor)}


def publish_deform360_joint_sparse_source_gate_v5(
    evidence_path: str | Path,
    lock_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate, evaluate, and atomically publish one source-gate result."""

    lock = load_deform360_joint_sparse_source_execution_lock_v5(lock_path)
    evidence = load_strict_json_object(evidence_path, label="v5 source evidence")
    result = evaluate_deform360_joint_sparse_source_gate_v5(evidence, lock)
    write_atomic_json(result, output_path, overwrite=overwrite)
    return result


__all__ = [
    "EXECUTION_LOCK_SCHEMA",
    "EXECUTION_LOCK_SEMANTICS",
    "EXECUTION_LOCK_VERSION",
    "PRIMARY_DEPLOYED_METHOD",
    "PRIMARY_RAW_METHOD",
    "PROSPECTIVE_POLICY_ID",
    "RAW_METHOD_IDS",
    "SELECTION_SHA256",
    "SOURCE_EVIDENCE_SCHEMA",
    "SOURCE_EVIDENCE_SEMANTICS",
    "SOURCE_EVIDENCE_VERSION",
    "SOURCE_RESULT_SCHEMA",
    "SOURCE_RESULT_SEMANTICS",
    "SOURCE_RESULT_VERSION",
    "evaluate_deform360_joint_sparse_source_gate_v5",
    "load_deform360_joint_sparse_source_execution_lock_v5",
    "parse_deform360_joint_sparse_source_evidence_v5",
    "publish_deform360_joint_sparse_source_gate_v5",
]
