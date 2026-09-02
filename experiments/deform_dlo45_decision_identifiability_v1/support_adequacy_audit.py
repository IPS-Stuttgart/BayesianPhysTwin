"""Source-frozen outer support audit for the DEFORM decision certificate.

The existing inner certificate is exact over a registered finite source support.
This module asks a different question: can outcome-free source diagnostics identify
when that support is likely to understate held decision regret? All policy fitting
and threshold selection use the official source-test trajectories. Held evaluation
is retrospective and target-frozen; it is not a deployment guarantee.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._common import (
    ATOL,
    DLOS,
    canonical_sha256,
    load_protocol,
    read_json,
    sha256_file,
    trajectory_paths,
    write_json,
)
from ._evaluation import load_models
from .gate_audit import (
    AuditProtocol,
    WindowRecord,
    _augment_for_combine,
    _combine_method,
    _source_model,
    _window_records,
    load_audit_protocol,
    summarize_method,
)

CONTRACT = "deform-dlo45-support-adequacy-audit-v1"
REQUEST_CONTRACT = "deform-dlo45-support-adequacy-audit-request-v1"
PARENT_CONTRACT = "deform-dlo45-decision-identifiability-v1"
FALLBACK_ACTION = 0
FEATURE_NAMES = (
    "action_is_full",
    "source_regret_bound",
    "source_regret_bound_squared",
    "quotient_concentration",
    "maximum_quotient_mass",
    "maximum_kernel_weight",
    "expected_fallback_advantage",
    "expected_action_gap",
    "hypothesis_action_agreement",
    "negative_residual_disagreement",
    "negative_unsupported_specificity",
    "selected_feature_distance_min",
    "selected_feature_distance_median",
    "selected_feature_distance_max",
    "current_frame_fraction",
    "dlo_is_5",
)


def _json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


@dataclass(frozen=True)
class SupportProtocol:
    parent_workflow_run_id: int
    parent_source_artifact_digest: str
    dataset_repository: str
    dataset_commit: str
    ridge_lambdas: tuple[float, ...]
    minimum_retained_fraction: float
    strict_violation_fraction: float
    exploratory_violation_fraction: float
    maximum_harm_fraction: float
    tolerance: float
    claim_boundary: str


@dataclass(frozen=True)
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    regularization: float

    def record(self) -> dict[str, object]:
        return {
            "feature_names": list(FEATURE_NAMES),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "regularization": self.regularization,
        }

    @classmethod
    def from_record(cls, value: Mapping[str, object]) -> RidgeModel:
        if tuple(value.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("ridge feature order changed")
        return cls(
            mean=np.asarray(value["mean"], dtype=np.float64),
            scale=np.asarray(value["scale"], dtype=np.float64),
            coefficients=np.asarray(value["coefficients"], dtype=np.float64),
            regularization=float(value["regularization"]),
        )


def load_support_protocol(path: Path) -> SupportProtocol:
    value = read_json(path)
    if (
        value.get("contract") != CONTRACT
        or value.get("schema_version") != 1
        or value.get("parent_contract") != PARENT_CONTRACT
        or value.get("target_tuning") is not False
        or value.get("target_outcomes_used_for_policy") is not False
        or value.get("future_residual_distance_used_for_policy") is not False
        or value.get("policy_unit") != "complete_trajectory"
    ):
        raise ValueError("invalid support-adequacy protocol")
    model = value.get("model")
    gates = value.get("source_selection")
    if not isinstance(model, dict) or not isinstance(gates, dict):
        raise ValueError("support protocol model/gate records missing")
    lambdas = tuple(float(item) for item in model.get("ridge_lambdas", ()))
    if not lambdas or any(item <= 0.0 or not math.isfinite(item) for item in lambdas):
        raise ValueError("invalid ridge grid")
    fractions = (
        float(gates["minimum_retained_fraction"]),
        float(gates["strict_violation_fraction"]),
        float(gates["exploratory_violation_fraction"]),
        float(gates["maximum_harm_fraction"]),
    )
    if any(not 0.0 <= item <= 1.0 for item in fractions):
        raise ValueError("invalid source selection fractions")
    return SupportProtocol(
        parent_workflow_run_id=int(value["parent_workflow_run_id"]),
        parent_source_artifact_digest=str(value["parent_source_artifact_digest"]),
        dataset_repository=str(value["dataset_repository"]),
        dataset_commit=str(value["dataset_commit"]),
        ridge_lambdas=lambdas,
        minimum_retained_fraction=fractions[0],
        strict_violation_fraction=fractions[1],
        exploratory_violation_fraction=fractions[2],
        maximum_harm_fraction=fractions[3],
        tolerance=float(value["regret_tolerance"]),
        claim_boundary=str(value["claim_boundary"]),
    )


def validate_request(path: Path, protocol: SupportProtocol) -> dict[str, object]:
    value = read_json(path)
    if (
        value.get("contract") != REQUEST_CONTRACT
        or value.get("schema_version") != 1
        or value.get("status") != "authorized-retrospective-development"
        or value.get("parent_workflow_run_id") != protocol.parent_workflow_run_id
        or value.get("source_only_policy_selection") is not True
        or value.get("target_tuning") is not False
        or value.get("target_retries") is not False
        or value.get("paper_claim_authorized") is not False
        or not isinstance(value.get("run_key"), str)
        or not str(value["run_key"]).strip()
    ):
        raise ValueError("invalid support-adequacy request")
    return value


def _trajectory_key(record: WindowRecord) -> str:
    return f"{record.dlo}/{record.trajectory}"


def _feature_vector(record: WindowRecord) -> np.ndarray:
    decision = record.decision
    distances = np.asarray(decision.selected_distances, dtype=np.float64)
    if distances.ndim != 1 or not len(distances) or not np.all(np.isfinite(distances)):
        raise ValueError("invalid outcome-free selected feature distances")
    scores = decision.scores
    values = np.asarray(
        [
            float(decision.decision.certificate_action == 2),
            record.certificate_source_regret_bound,
            record.certificate_source_regret_bound**2,
            scores["quotient_concentration"],
            scores["maximum_quotient_mass"],
            scores["maximum_kernel_weight"],
            scores["expected_fallback_advantage"],
            scores["expected_action_gap"],
            scores["hypothesis_action_agreement"],
            scores["negative_residual_disagreement"],
            scores["negative_unsupported_specificity"],
            float(np.min(distances)),
            float(np.median(distances)),
            float(np.max(distances)),
            record.current_frame / 454.0,
            float(record.dlo == "DLO5"),
        ],
        dtype=np.float64,
    )
    if values.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(values)):
        raise ValueError("support feature vector is invalid")
    return values


def _eligible(records: Sequence[WindowRecord]) -> list[WindowRecord]:
    return [
        record
        for record in records
        if record.decision.decision.certificate_action != FALLBACK_ACTION
    ]


def _group_weights(groups: Sequence[str]) -> np.ndarray:
    counts = Counter(groups)
    if not counts:
        raise ValueError("empty group roster")
    total = len(groups)
    group_count = len(counts)
    return np.asarray(
        [total / (group_count * counts[group]) for group in groups],
        dtype=np.float64,
    )


def fit_ridge(
    features: np.ndarray,
    outcomes: np.ndarray,
    groups: Sequence[str],
    regularization: float,
) -> RidgeModel:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError("ridge design shape mismatch")
    if not len(x) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("ridge design is empty or nonfinite")
    weights = _group_weights(groups)
    weight_sum = float(np.sum(weights))
    mean = np.einsum("i,ij->j", weights, x) / weight_sum
    variance = (
        np.einsum(
            "i,ij,ij->j",
            weights,
            x - mean,
            x - mean,
        )
        / weight_sum
    )
    scale = np.sqrt(np.maximum(variance, 1e-12))
    standardized = (x - mean) / scale
    design = np.column_stack((np.ones(len(x)), standardized))
    gram = design.T @ (weights[:, None] * design)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(regularization)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        gram + penalty,
        design.T @ (weights * y),
    )
    return RidgeModel(
        mean=mean,
        scale=scale,
        coefficients=coefficients,
        regularization=float(regularization),
    )


def predict_ridge(model: RidgeModel, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError("ridge prediction shape mismatch")
    standardized = (x - model.mean) / model.scale
    design = np.column_stack((np.ones(len(x)), standardized))
    return design @ model.coefficients


def _source_arrays(
    records: Sequence[WindowRecord],
) -> tuple[list[WindowRecord], np.ndarray, np.ndarray, np.ndarray]:
    eligible = _eligible(records)
    if not eligible:
        raise ValueError("source certificate has no nonfallback decisions")
    features = np.vstack([_feature_vector(record) for record in eligible])
    excess = np.asarray(
        [record.certificate_regret_excess for record in eligible],
        dtype=np.float64,
    )
    groups = np.asarray(
        [_trajectory_key(record) for record in eligible],
        dtype=object,
    )
    return eligible, features, excess, groups


def cross_fitted_ridge_scores(
    records: Sequence[WindowRecord],
    regularization: float,
) -> tuple[np.ndarray, RidgeModel, float]:
    eligible, features, excess, groups = _source_arrays(records)
    predictions = np.empty(len(eligible), dtype=np.float64)
    unique_groups = sorted(set(str(item) for item in groups.tolist()))
    if len(unique_groups) < 4:
        raise ValueError("ridge cross-fitting requires at least four trajectories")
    group_max_residuals: list[float] = []
    for group in unique_groups:
        train = groups != group
        test = ~train
        model = fit_ridge(
            features[train],
            excess[train],
            groups[train].tolist(),
            regularization,
        )
        predictions[test] = predict_ridge(model, features[test])
        group_max_residuals.append(float(np.max(excess[test] - predictions[test])))
    final_model = fit_ridge(
        features,
        excess,
        groups.tolist(),
        regularization,
    )
    # With sixteen source-test trajectories, a one-sided 90% finite-group
    # order statistic is the maximum. This is a conservative development
    # envelope, not a fresh target-domain coverage theorem.
    envelope_offset = float(np.max(group_max_residuals))
    return predictions, final_model, envelope_offset


def _source_policy_metrics(
    records: Sequence[WindowRecord],
    risk_by_id: Mapping[str, float],
    threshold: float,
    tolerance: float,
) -> dict[str, object]:
    actions = [
        (
            record.decision.decision.certificate_action
            if (
                record.decision.decision.certificate_action != FALLBACK_ACTION
                and float(risk_by_id[record.stable_id]) <= threshold
            )
            else FALLBACK_ACTION
        )
        for record in records
    ]
    summary = summarize_method(records, actions)
    accepted = [
        (record, action)
        for record, action in zip(records, actions, strict=True)
        if action != FALLBACK_ACTION
    ]
    tolerance_violations = sum(
        record.normalized_regret[action] > tolerance + ATOL
        for record, action in accepted
    )
    bound_exceedances = sum(
        record.normalized_regret[action]
        > record.decision.decision.worst_case_regret[action] + ATOL
        for record, action in accepted
    )
    summary.update(
        {
            "realized_tolerance_violation_count": tolerance_violations,
            "realized_tolerance_violation_fraction": (
                tolerance_violations / len(accepted) if accepted else 0.0
            ),
            "inner_bound_exceedance_count": bound_exceedances,
            "inner_bound_exceedance_fraction": (
                bound_exceedances / len(accepted) if accepted else 0.0
            ),
            "threshold": float(threshold),
        }
    )
    return summary


def _per_dlo_no_regression(
    records: Sequence[WindowRecord],
    risk_by_id: Mapping[str, float],
    threshold: float,
) -> bool:
    for dlo in DLOS:
        subset = [record for record in records if record.dlo == dlo]
        actions = [
            (
                record.decision.decision.certificate_action
                if (
                    record.decision.decision.certificate_action != FALLBACK_ACTION
                    and float(risk_by_id[record.stable_id]) <= threshold
                )
                else FALLBACK_ACTION
            )
            for record in subset
        ]
        if summarize_method(subset, actions)["rmse_ratio_to_fallback"] > 1.0 + ATOL:
            return False
    return True


def _threshold_candidates(values: Sequence[float]) -> list[float]:
    unique = sorted(set(float(item) for item in values))
    if not unique:
        return []
    result = [math.nextafter(unique[0], -math.inf)]
    result.extend(unique)
    return result


def _policy_rank_key(item: Mapping[str, object]) -> tuple[float, ...]:
    metrics = item["source_cross_fitted"]
    assert isinstance(metrics, Mapping)
    return (
        float(metrics["nonfallback_count"]),
        -float(metrics["realized_tolerance_violation_fraction"]),
        -float(metrics["harmful_fraction_nonfallback"]),
        -float(metrics["rmse_ratio_to_fallback"]),
        -float(item["complexity_rank"]),
    )


def _fit_source_policies(
    records: Sequence[WindowRecord],
    protocol: SupportProtocol,
) -> dict[str, object]:
    eligible = _eligible(records)
    inner_count = len(eligible)
    minimum_count = max(
        1,
        math.ceil(protocol.minimum_retained_fraction * inner_count),
    )
    candidates: list[dict[str, object]] = []
    runtime_risks: dict[str, dict[str, float]] = {}
    ridge_records: dict[str, object] = {}

    def register(
        candidate: dict[str, object],
        risk_by_id: Mapping[str, float],
    ) -> None:
        metrics = _source_policy_metrics(
            records,
            risk_by_id,
            float(candidate["threshold"]),
            protocol.tolerance,
        )
        per_dlo_ok = _per_dlo_no_regression(
            records,
            risk_by_id,
            float(candidate["threshold"]),
        )
        strict_gate = bool(
            int(metrics["nonfallback_count"]) >= minimum_count
            and float(metrics["realized_tolerance_violation_fraction"])
            <= protocol.strict_violation_fraction + ATOL
            and float(metrics["harmful_fraction_nonfallback"])
            <= protocol.maximum_harm_fraction + ATOL
            and float(metrics["rmse_ratio_to_fallback"]) <= 1.0 + ATOL
            and per_dlo_ok
        )
        exploratory_gate = bool(
            int(metrics["nonfallback_count"]) >= minimum_count
            and float(metrics["realized_tolerance_violation_fraction"])
            <= protocol.exploratory_violation_fraction + ATOL
            and float(metrics["harmful_fraction_nonfallback"])
            <= protocol.maximum_harm_fraction + ATOL
            and float(metrics["rmse_ratio_to_fallback"]) <= 1.0 + ATOL
            and per_dlo_ok
        )
        record = {
            **candidate,
            "source_cross_fitted": metrics,
            "per_dlo_no_regression": per_dlo_ok,
            "strict_source_gate": strict_gate,
            "exploratory_source_gate": exploratory_gate,
        }
        name = str(record["candidate_id"])
        if name in runtime_risks:
            raise ValueError(f"duplicate source policy candidate: {name}")
        runtime_risks[name] = {
            str(key): float(value) for key, value in risk_by_id.items()
        }
        candidates.append(record)

    # Structural hypothesis: tolerance-certified full corrections may be more
    # transport-stable than compromise half corrections.
    structural_risk = {
        record.stable_id: float(record.decision.decision.certificate_action != 2)
        for record in eligible
    }
    register(
        {
            "candidate_id": "full_correction_only",
            "name": "full_correction_only",
            "kind": "structural",
            "complexity_rank": 0,
            "threshold": 0.5,
        },
        structural_risk,
    )

    # Generic current-observation support baseline. This uses distance in the
    # registered feature space, never future residual distance.
    distance_risk = {
        record.stable_id: float(np.median(record.decision.selected_distances))
        for record in eligible
    }
    for index, threshold in enumerate(_threshold_candidates(distance_risk.values())):
        register(
            {
                "candidate_id": f"feature_distance_{index}",
                "name": "feature_distance",
                "kind": "scalar",
                "complexity_rank": 1,
                "threshold": float(threshold),
            },
            distance_risk,
        )

    eligible_features = np.vstack([_feature_vector(record) for record in eligible])
    for regularization in protocol.ridge_lambdas:
        oof, final_model, envelope_offset = cross_fitted_ridge_scores(
            records,
            regularization,
        )
        ridge_key = f"ridge_{regularization:g}"
        oof_risk = {
            record.stable_id: float(value)
            for record, value in zip(eligible, oof, strict=True)
        }
        full_predictions = predict_ridge(final_model, eligible_features)
        ridge_records[ridge_key] = {
            "model": final_model.record(),
            "leave_trajectory_out_max_residual_offset": envelope_offset,
        }

        for index, threshold in enumerate(_threshold_candidates(oof_risk.values())):
            metrics = _source_policy_metrics(
                records,
                oof_risk,
                threshold,
                protocol.tolerance,
            )
            selected_count = int(metrics["nonfallback_count"])
            if selected_count:
                sorted_full = np.sort(full_predictions)
                full_threshold = float(
                    sorted_full[min(selected_count, len(sorted_full)) - 1]
                )
            else:
                full_threshold = math.nextafter(
                    float(np.min(full_predictions)),
                    -math.inf,
                )
            register(
                {
                    "candidate_id": f"{ridge_key}_{index}",
                    "name": ridge_key,
                    "kind": "ridge",
                    "complexity_rank": 2,
                    "regularization": regularization,
                    "threshold": float(threshold),
                    "full_fit_threshold": full_threshold,
                },
                oof_risk,
            )

        # Conservative composed score: represented-support regret plus predicted
        # regret excess plus the maximum leave-one-trajectory-out residual.
        envelope_risk = {
            record.stable_id: (
                record.certificate_source_regret_bound
                + float(predicted)
                + envelope_offset
            )
            for record, predicted in zip(eligible, oof, strict=True)
        }
        register(
            {
                "candidate_id": f"outer_envelope_{regularization:g}",
                "name": f"outer_envelope_{regularization:g}",
                "kind": "conformal",
                "complexity_rank": 3,
                "regularization": regularization,
                "threshold": protocol.tolerance,
            },
            envelope_risk,
        )

    strict = [item for item in candidates if bool(item["strict_source_gate"])]
    exploratory = [item for item in candidates if bool(item["exploratory_source_gate"])]
    pool = strict or exploratory
    if pool:
        selected = max(pool, key=_policy_rank_key)
        selection_class = (
            "strict-source-selected" if strict else "exploratory-source-selected"
        )
    else:
        selected = next(item for item in candidates if item["kind"] == "structural")
        selection_class = "no-source-gate-passed-structural-diagnostic"

    def deployable(item: Mapping[str, object]) -> dict[str, object]:
        deployed = {
            key: value
            for key, value in item.items()
            if key
            not in {
                "candidate_id",
                "source_cross_fitted",
                "strict_source_gate",
                "exploratory_source_gate",
                "per_dlo_no_regression",
            }
        }
        kind = str(deployed["kind"])
        if kind == "ridge":
            deployed["threshold"] = float(deployed.pop("full_fit_threshold"))
            deployed["model"] = ridge_records[str(deployed["name"])]["model"]
        elif kind == "conformal":
            ridge_key = f"ridge_{float(deployed['regularization']):g}"
            deployed["model"] = ridge_records[ridge_key]["model"]
            deployed["conformal_offset"] = ridge_records[ridge_key][
                "leave_trajectory_out_max_residual_offset"
            ]
        else:
            deployed.pop("full_fit_threshold", None)
        return deployed

    diagnostic_policies: dict[str, object] = {}
    for kind in ("scalar", "ridge", "conformal"):
        kind_pool = [
            item
            for item in candidates
            if item["kind"] == kind and bool(item["exploratory_source_gate"])
        ]
        if kind_pool:
            best = max(kind_pool, key=_policy_rank_key)
            diagnostic_policies[f"best_source_{kind}"] = deployable(best)

    return {
        "selection_class": selection_class,
        "strict_source_gate_passed": bool(strict),
        "minimum_retained_count": minimum_count,
        "inner_certificate_nonfallback_count": inner_count,
        "selected_policy": {
            **selected,
            **deployable(selected),
        },
        "diagnostic_policies": diagnostic_policies,
        "ridge_models": ridge_records,
        "candidate_count": len(candidates),
        "candidate_summaries": candidates,
    }


def _policy_risk(record: WindowRecord, policy: Mapping[str, object]) -> float:
    kind = str(policy["kind"])
    if kind == "structural":
        return float(record.decision.decision.certificate_action != 2)
    if kind == "scalar":
        return float(np.median(record.decision.selected_distances))
    if kind in {"ridge", "conformal"}:
        model_value = policy["model"]
        if not isinstance(model_value, Mapping):
            raise ValueError("ridge model record is malformed")
        model = RidgeModel.from_record(model_value)
        prediction = float(predict_ridge(model, _feature_vector(record)[None, :])[0])
        if kind == "conformal":
            return (
                record.certificate_source_regret_bound
                + prediction
                + float(policy["conformal_offset"])
            )
        return prediction
    raise ValueError(f"unknown support policy kind: {kind}")


def policy_actions(
    records: Sequence[WindowRecord],
    policy: Mapping[str, object],
) -> tuple[list[int], dict[str, float]]:
    risks: dict[str, float] = {}
    actions: list[int] = []
    threshold = float(policy["threshold"])
    for record in records:
        action = record.decision.decision.certificate_action
        if action == FALLBACK_ACTION:
            risks[record.stable_id] = math.inf
            actions.append(FALLBACK_ACTION)
            continue
        risk = _policy_risk(record, policy)
        risks[record.stable_id] = risk
        actions.append(action if risk <= threshold else FALLBACK_ACTION)
    return actions, risks


def _diagnostics(
    records: Sequence[WindowRecord],
    actions: Sequence[int],
    risks: Mapping[str, float],
    policy: Mapping[str, object],
    tolerance: float,
) -> dict[str, object]:
    accepted = [
        (record, action)
        for record, action in zip(records, actions, strict=True)
        if action != FALLBACK_ACTION
    ]
    inner_exceeded = sum(
        record.normalized_regret[action]
        > record.decision.decision.worst_case_regret[action] + ATOL
        for record, action in accepted
    )
    tolerance_exceeded = sum(
        record.normalized_regret[action] > tolerance + ATOL
        for record, action in accepted
    )
    exact_fallback_violations = sum(
        abs(float(record.physical_mse[action]) - record.fallback_mse) > ATOL
        for record, action in zip(records, actions, strict=True)
        if action == FALLBACK_ACTION
    )
    result: dict[str, object] = {
        "inner_bound_exceedance_count": inner_exceeded,
        "inner_bound_exceedance_fraction": (
            inner_exceeded / len(accepted) if accepted else 0.0
        ),
        "realized_tolerance_violation_count": tolerance_exceeded,
        "realized_tolerance_violation_fraction": (
            tolerance_exceeded / len(accepted) if accepted else 0.0
        ),
        "exact_fallback_violation_count": exact_fallback_violations,
        "accepted_risk_mean": (
            float(np.mean([risks[record.stable_id] for record, _ in accepted]))
            if accepted
            else None
        ),
    }
    if str(policy["kind"]) == "conformal":
        violations = sum(
            record.certificate_regret_excess
            > (float(risks[record.stable_id]) - record.certificate_source_regret_bound)
            + ATOL
            for record, _ in accepted
        )
        result["outer_envelope_violation_count"] = violations
        result["outer_envelope_violation_fraction"] = (
            violations / len(accepted) if accepted else 0.0
        )
    return result


def _summarize_policy(
    records_by_dlo: Mapping[str, Sequence[WindowRecord]],
    policy: Mapping[str, object],
    audit: AuditProtocol,
    tolerance: float,
    seed_offset: int,
) -> dict[str, object]:
    by_dlo: dict[str, dict[str, object]] = {}
    all_diagnostics: list[dict[str, object]] = []
    for dlo in DLOS:
        records = list(records_by_dlo[dlo])
        actions, risks = policy_actions(records, policy)
        summary = _augment_for_combine(
            summarize_method(records, actions),
            records,
            actions,
        )
        diagnostics = _diagnostics(
            records,
            actions,
            risks,
            policy,
            tolerance,
        )
        summary.update(diagnostics)
        by_dlo[dlo] = summary
        for record, action in zip(records, actions, strict=True):
            all_diagnostics.append(
                {
                    "stable_id": record.stable_id,
                    "dlo": record.dlo,
                    "trajectory": record.trajectory,
                    "current_frame": record.current_frame,
                    "inner_action": (record.decision.decision.certificate_action),
                    "deployed_action": action,
                    "risk_score": (
                        risks[record.stable_id]
                        if math.isfinite(risks[record.stable_id])
                        else None
                    ),
                    "source_regret_bound": (record.certificate_source_regret_bound),
                    "realized_regret": float(record.normalized_regret[action]),
                    "inner_regret_excess": (record.certificate_regret_excess),
                    "harmful_vs_fallback": bool(
                        record.physical_mse[action] > record.fallback_mse + ATOL
                    ),
                }
            )
    combined = _combine_method(by_dlo, audit, seed_offset)
    accepted = [item for item in all_diagnostics if item["deployed_action"] != 0]
    inner_exceeded = sum(
        item["realized_regret"] > item["source_regret_bound"] + ATOL
        for item in accepted
    )
    tolerance_exceeded = sum(
        item["realized_regret"] > tolerance + ATOL for item in accepted
    )
    combined.update(
        {
            "inner_bound_exceedance_count": inner_exceeded,
            "inner_bound_exceedance_fraction": (
                inner_exceeded / len(accepted) if accepted else 0.0
            ),
            "realized_tolerance_violation_count": tolerance_exceeded,
            "realized_tolerance_violation_fraction": (
                tolerance_exceeded / len(accepted) if accepted else 0.0
            ),
            "exact_fallback_violation_count": sum(
                item["deployed_action"] == 0 and item["harmful_vs_fallback"]
                for item in all_diagnostics
            ),
        }
    )
    return {
        "combined": combined,
        "dlos": by_dlo,
        "per_decision": all_diagnostics,
    }


def source_command(args: argparse.Namespace) -> int:
    parent_protocol = load_protocol(args.parent_protocol)
    load_audit_protocol(args.gate_audit_protocol)
    protocol = load_support_protocol(args.support_protocol)
    request = validate_request(args.request, protocol)
    source_result = read_json(args.parent_source_result)
    source_seal = read_json(args.parent_source_seal)
    if (
        source_result.get("contract") != PARENT_CONTRACT
        or source_result.get("stage") != "source"
        or source_seal.get("contract") != PARENT_CONTRACT
        or source_seal.get("stage") != "source-seal"
        or source_seal.get("source_result_sha256")
        != sha256_file(args.parent_source_result)
        or source_seal.get("source_model_sha256")
        != sha256_file(args.parent_source_model)
    ):
        raise ValueError("parent source artifact does not verify")

    records: list[WindowRecord] = []
    source_counts: dict[str, int] = {}
    for dlo in DLOS:
        paths = trajectory_paths(args.dataset_root, dlo, "train")
        model, source_test_paths = _source_model(
            paths,
            dlo,
            source_result,
            parent_protocol,
        )
        dlo_records = _window_records(
            source_test_paths,
            model,
            parent_protocol,
            dlo,
        )
        records.extend(dlo_records)
        source_counts[dlo] = len(dlo_records)

    policy = _fit_source_policies(records, protocol)
    result = {
        "contract": CONTRACT,
        "schema_version": 1,
        "stage": "source",
        "run_key": request["run_key"],
        "source_decision_counts": source_counts,
        "policy": policy,
        "parent_source_model_sha256": sha256_file(args.parent_source_model),
        "parent_source_result_sha256": sha256_file(args.parent_source_result),
        "parent_source_seal_sha256": sha256_file(args.parent_source_seal),
        "parent_protocol_sha256": sha256_file(args.parent_protocol),
        "support_protocol_sha256": sha256_file(args.support_protocol),
        "gate_audit_protocol_sha256": sha256_file(args.gate_audit_protocol),
        "request_sha256": sha256_file(args.request),
        "target_data_read": False,
        "target_outcomes_used_for_policy": False,
        "future_residual_distance_used_for_policy": False,
        "claim_boundary": protocol.claim_boundary,
    }
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "source_result.json", _json_ready(result))
    selected = policy["selected_policy"]
    diagnostics = policy["diagnostic_policies"]
    assert isinstance(selected, dict)
    assert isinstance(diagnostics, dict)
    deployment_policy = {
        key: value
        for key, value in selected.items()
        if key
        not in {
            "candidate_id",
            "source_cross_fitted",
            "strict_source_gate",
            "exploratory_source_gate",
            "per_dlo_no_regression",
            "full_fit_threshold",
        }
    }
    deployment = {
        "contract": CONTRACT,
        "schema_version": 1,
        "stage": "deployment-policy",
        "run_key": request["run_key"],
        "selection_class": policy["selection_class"],
        "strict_source_gate_passed": policy["strict_source_gate_passed"],
        "policy": deployment_policy,
        "diagnostic_policies": diagnostics,
        "feature_names": list(FEATURE_NAMES),
        "target_data_read": False,
        "target_outcomes_used_for_policy": False,
    }
    deployment["policy_id"] = canonical_sha256(deployment)
    write_json(output / "policy.json", _json_ready(deployment))
    seal = {
        "contract": CONTRACT,
        "schema_version": 1,
        "stage": "source-seal",
        "run_key": request["run_key"],
        "source_result_sha256": sha256_file(output / "source_result.json"),
        "policy_sha256": sha256_file(output / "policy.json"),
        "parent_source_model_sha256": sha256_file(args.parent_source_model),
    }
    seal["seal_id"] = canonical_sha256(seal)
    write_json(output / "source_seal.json", seal)
    return 0


def target_command(args: argparse.Namespace) -> int:
    parent_protocol = load_protocol(args.parent_protocol)
    audit = load_audit_protocol(args.gate_audit_protocol)
    protocol = load_support_protocol(args.support_protocol)
    request = validate_request(args.request, protocol)
    source = read_json(args.source_result)
    policy_record = read_json(args.policy)
    seal = read_json(args.source_seal)
    if (
        source.get("contract") != CONTRACT
        or source.get("stage") != "source"
        or policy_record.get("contract") != CONTRACT
        or policy_record.get("stage") != "deployment-policy"
        or seal.get("contract") != CONTRACT
        or seal.get("stage") != "source-seal"
        or seal.get("source_result_sha256") != sha256_file(args.source_result)
        or seal.get("policy_sha256") != sha256_file(args.policy)
        or seal.get("parent_source_model_sha256")
        != sha256_file(args.parent_source_model)
        or source.get("run_key") != request["run_key"]
        or policy_record.get("run_key") != request["run_key"]
    ):
        raise ValueError("source-frozen support policy does not verify")
    models = load_models(args.parent_source_model)
    records_by_dlo = {
        dlo: _window_records(
            trajectory_paths(args.dataset_root, dlo, "eval"),
            models[dlo],
            parent_protocol,
            dlo,
        )
        for dlo in DLOS
    }

    selected_policy = policy_record["policy"]
    diagnostic_policies = policy_record["diagnostic_policies"]
    assert isinstance(selected_policy, dict)
    assert isinstance(diagnostic_policies, dict)
    methods: dict[str, Mapping[str, object]] = {
        "source_selected_outer": selected_policy,
        "full_correction_only": {
            "name": "full_correction_only",
            "kind": "structural",
            "threshold": 0.5,
        },
    }
    for name, method in diagnostic_policies.items():
        if not isinstance(method, dict):
            raise ValueError("diagnostic policy record is malformed")
        methods[str(name)] = method

    # Refit-record diagnostics permit every conservative outer-envelope
    # regularization to be inspected without target-side model selection.
    source_policy = source["policy"]
    assert isinstance(source_policy, dict)
    ridge_models = source_policy["ridge_models"]
    assert isinstance(ridge_models, dict)
    for key, value in ridge_models.items():
        if not isinstance(value, dict):
            raise ValueError("ridge source record is malformed")
        methods[f"{key}_outer_envelope"] = {
            "name": f"{key}_outer_envelope",
            "kind": "conformal",
            "regularization": float(key.split("_", 1)[1]),
            "threshold": protocol.tolerance,
            "model": value["model"],
            "conformal_offset": value["leave_trajectory_out_max_residual_offset"],
        }

    # Existing inner certificate expressed as a pass-through structural policy.
    inner_policy = {
        "name": "inner_certificate",
        "kind": "structural",
        "threshold": 1.0,
    }
    methods = {"inner_certificate": inner_policy, **methods}

    summaries: dict[str, object] = {}
    per_decision: dict[str, list[dict[str, object]]] = {}
    for index, (name, method) in enumerate(methods.items()):
        summary = _summarize_policy(
            records_by_dlo,
            method,
            audit,
            protocol.tolerance,
            seed_offset=500 + index,
        )
        summaries[name] = {
            "combined": summary["combined"],
            "dlos": summary["dlos"],
            "policy": method,
        }
        per_decision[name] = summary["per_decision"]

    inner_record = summaries["inner_certificate"]
    selected_record = summaries["source_selected_outer"]
    full_record = summaries["full_correction_only"]
    assert isinstance(inner_record, dict)
    assert isinstance(selected_record, dict)
    assert isinstance(full_record, dict)
    inner = inner_record["combined"]
    selected = selected_record["combined"]
    full_only = full_record["combined"]
    assert isinstance(inner, dict)
    assert isinstance(selected, dict)
    assert isinstance(full_only, dict)
    big_gate = {
        "strict_source_gate_passed": bool(policy_record["strict_source_gate_passed"]),
        "positive_rmse_gain": float(selected["rmse_reduction"]) > 0.0,
        "at_least_30_nonfallback": int(selected["nonfallback_count"]) >= 30,
        "realized_tolerance_violation_at_most_10pct": (
            float(selected["realized_tolerance_violation_fraction"]) <= 0.10 + ATOL
        ),
        "fewer_tolerance_violations_than_inner": (
            int(selected["realized_tolerance_violation_count"])
            < int(inner["realized_tolerance_violation_count"])
        ),
        "no_dlo_regression": all(
            float(selected_record["dlos"][dlo]["rmse_ratio_to_fallback"]) <= 1.0 + ATOL
            for dlo in DLOS
        ),
        "zero_exact_fallback_violations": (
            int(selected["exact_fallback_violation_count"]) == 0
        ),
    }
    result = {
        "contract": CONTRACT,
        "schema_version": 1,
        "stage": "target",
        "status": "completed",
        "run_key": request["run_key"],
        "source_selection_class": policy_record["selection_class"],
        "strict_source_gate_passed": policy_record["strict_source_gate_passed"],
        "methods": summaries,
        "big_result_gate": {
            "criteria": big_gate,
            "passed": all(big_gate.values()),
        },
        "headline_comparison": {
            "inner_nonfallback": inner["nonfallback_count"],
            "inner_realized_tolerance_violation_fraction": inner[
                "realized_tolerance_violation_fraction"
            ],
            "inner_rmse_reduction": inner["rmse_reduction"],
            "selected_nonfallback": selected["nonfallback_count"],
            "selected_realized_tolerance_violation_fraction": selected[
                "realized_tolerance_violation_fraction"
            ],
            "selected_rmse_reduction": selected["rmse_reduction"],
            "full_only_nonfallback": full_only["nonfallback_count"],
            "full_only_realized_tolerance_violation_fraction": full_only[
                "realized_tolerance_violation_fraction"
            ],
            "full_only_rmse_reduction": full_only["rmse_reduction"],
        },
        "target_policy_tuning": False,
        "target_outcomes_used_for_policy": False,
        "target_retries": False,
        "paper_claim_authorized": False,
        "classification": ("retrospective-source-frozen-support-adequacy-development"),
        "claim_boundary": protocol.claim_boundary,
    }
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "result.json", _json_ready(result))
    with (output / "per_decision.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for method, items in per_decision.items():
            for item in items:
                payload = _json_ready({"method": method, **item})
                handle.write(
                    json.dumps(
                        payload,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
    report = [
        "# DEFORM support-aware outer certificate",
        "",
        f"Source selection: **{policy_record['selection_class']}**",
        (f"Strict source gate: **{policy_record['strict_source_gate_passed']}**"),
        (f"Big-result development gate: **{result['big_result_gate']['passed']}**"),
        "",
        "| Method | Nonfallback | RMSE gain | Tolerance violations | "
        "Inner-bound exceedances | Harm |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, value in summaries.items():
        if not isinstance(value, dict):
            raise ValueError("summary record is malformed")
        combined = value["combined"]
        if not isinstance(combined, dict):
            raise ValueError("combined summary is malformed")
        report.append(
            f"| `{name}` | {int(combined['nonfallback_count'])} | "
            f"{100.0 * float(combined['rmse_reduction']):.2f}% | "
            f"{int(combined['realized_tolerance_violation_count'])}/"
            f"{int(combined['nonfallback_count'])} | "
            f"{int(combined['inner_bound_exceedance_count'])}/"
            f"{int(combined['nonfallback_count'])} | "
            f"{int(combined['harmful_nonfallback_count'])}/"
            f"{int(combined['nonfallback_count'])} |"
        )
    report.extend(["", protocol.claim_boundary, ""])
    (output / "README.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    return 0


def self_test() -> None:
    x = np.asarray(
        [
            [0.0] * len(FEATURE_NAMES),
            [1.0] * len(FEATURE_NAMES),
            [2.0] * len(FEATURE_NAMES),
            [3.0] * len(FEATURE_NAMES),
        ],
        dtype=np.float64,
    )
    y = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    groups = ["a", "b", "c", "d"]
    model = fit_ridge(x, y, groups, 1.0)
    prediction = predict_ridge(model, x)
    if prediction.shape != y.shape or not np.all(np.isfinite(prediction)):
        raise RuntimeError("ridge self-test failed")
    record = model.record()
    restored = RidgeModel.from_record(record)
    if not np.allclose(predict_ridge(restored, x), prediction):
        raise RuntimeError("ridge serialization self-test failed")
    print("support adequacy audit self-test passed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("source", "target", "self-test"))
    parser.add_argument("--parent-protocol", type=Path)
    parser.add_argument("--gate-audit-protocol", type=Path)
    parser.add_argument("--support-protocol", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--parent-source-model", type=Path)
    parser.add_argument("--parent-source-result", type=Path)
    parser.add_argument("--parent-source-seal", type=Path)
    parser.add_argument("--source-result", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--source-seal", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "self-test":
        self_test()
        return 0
    required = (
        args.parent_protocol,
        args.gate_audit_protocol,
        args.support_protocol,
        args.request,
        args.dataset_root,
        args.output_root,
        args.parent_source_model,
        args.parent_source_seal,
    )
    if any(value is None for value in required):
        raise ValueError("missing required common argument")
    if args.command == "source":
        if args.parent_source_result is None:
            raise ValueError("source command requires parent source result")
        return source_command(args)
    if args.source_result is None or args.policy is None or args.source_seal is None:
        raise ValueError("target command requires source result, policy, and seal")
    return target_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
