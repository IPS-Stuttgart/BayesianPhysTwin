"""Source-frozen outer support-adequacy pilot for DEFORM DLO4/DLO5.

The existing finite-action certificate is unchanged. This module learns only an
outer support-misspecification model from official training trajectories, then
evaluates the composition on the already-open official evaluation trajectories.

The primary outer method is a group-conformal upper envelope on realized regret.
A secondary regularized logistic ranker is reported only as an operational signal
diagnostic. Neither method may use target-suffix distances or target outcomes as
features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from experiments.deform_dlo45_decision_identifiability_v1._common import (
    ATOL,
    DLOS,
    FRAME_COUNT,
    Protocol,
    canonical_sha256,
    load_protocol,
    read_json,
    sha256_file,
    trajectory_paths,
    window_starts,
    write_json,
)
from experiments.deform_dlo45_decision_identifiability_v1._evaluation import (
    load_models,
)
from experiments.deform_dlo45_decision_identifiability_v1._model import (
    build_pool,
    fit_model,
)
from experiments.deform_dlo45_decision_identifiability_v1.gate_audit import (
    WindowRecord,
    _window_records,
    summarize_method,
)

CONTRACT: Final = "deform-dlo45-support-adequacy-v1"
SOURCE_CONTRACT: Final = "deform-dlo45-support-adequacy-source-v1"
TARGET_CONTRACT: Final = "deform-dlo45-support-adequacy-target-v1"
MODEL_CONTRACT: Final = "deform-dlo45-support-adequacy-model-v1"
REQUEST_CONTRACT: Final = "deform-dlo45-support-adequacy-request-v1"
PARENT_CONTRACT: Final = "deform-dlo45-decision-identifiability-v1"

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

FEATURE_NAMES: Final = (
    "certificate_source_regret_bound",
    "expected_action_gap",
    "expected_fallback_advantage",
    "hypothesis_action_agreement",
    "maximum_kernel_weight",
    "maximum_quotient_mass",
    "negative_residual_disagreement",
    "negative_unsupported_specificity",
    "quotient_concentration",
    "worst_case_regret_fallback",
    "worst_case_regret_half",
    "worst_case_regret_full",
    "current_frame_fraction",
    "dlo5_indicator",
    "full_correction_indicator",
)

FORBIDDEN_FEATURE_TOKENS: Final = (
    "realized",
    "harmful",
    "nearest",
    "oracle",
    "future_internal",
)


@dataclass(frozen=True)
class LinearRiskModel:
    kind: str
    feature_names: tuple[str, ...]
    feature_mean: FloatArray
    feature_scale: FloatArray
    coefficients: FloatArray
    intercept: float
    ridge: float

    def _matrix(self, values: FloatArray) -> FloatArray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != len(self.feature_names):
            raise ValueError("risk-model feature dimension changed")
        if not np.all(np.isfinite(array)):
            raise ValueError("risk-model features are nonfinite")
        return (array - self.feature_mean[None, :]) / self.feature_scale[None, :]

    def predict_raw(self, values: FloatArray) -> FloatArray:
        return self.intercept + self._matrix(values) @ self.coefficients

    def predict_bad_probability(self, values: FloatArray) -> FloatArray:
        if self.kind != "logistic":
            raise ValueError("bad-probability prediction requires logistic model")
        raw = np.clip(self.predict_raw(values), -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-raw))

    def predict_excess(self, values: FloatArray) -> FloatArray:
        if self.kind != "ridge":
            raise ValueError("excess prediction requires ridge model")
        return self.predict_raw(values)

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "feature_names": list(self.feature_names),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "ridge": self.ridge,
        }

    @classmethod
    def from_record(cls, value: Mapping[str, object]) -> "LinearRiskModel":
        names = tuple(str(item) for item in value["feature_names"])
        if names != FEATURE_NAMES:
            raise ValueError("risk-model feature roster changed")
        return cls(
            kind=str(value["kind"]),
            feature_names=names,
            feature_mean=np.asarray(value["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(value["feature_scale"], dtype=np.float64),
            coefficients=np.asarray(value["coefficients"], dtype=np.float64),
            intercept=float(value["intercept"]),
            ridge=float(value["ridge"]),
        )


def _load_support_protocol(path: Path) -> dict[str, object]:
    value = read_json(path)
    parent = value.get("parent")
    crossfit = value.get("source_crossfit")
    support = value.get("support_misspecification")
    operational = value.get("operational_risk_gate")
    bootstrap = value.get("bootstrap")
    if (
        value.get("contract") != CONTRACT
        or value.get("schema_version") != 1
        or not isinstance(parent, dict)
        or not isinstance(crossfit, dict)
        or not isinstance(support, dict)
        or not isinstance(operational, dict)
        or not isinstance(bootstrap, dict)
        or tuple(value.get("outcome_free_features", ())) != FEATURE_NAMES
        or value.get("target_outcomes_used_for_model_or_threshold_selection")
        is not False
        or value.get("target_retries_authorized") is not False
        or value.get("paper_claim_authorized") is not False
    ):
        raise ValueError("support-adequacy protocol changed")
    if any(
        token in name for name in FEATURE_NAMES for token in FORBIDDEN_FEATURE_TOKENS
    ):
        raise ValueError("forbidden outcome token entered the feature roster")
    if (
        parent.get("decision_contract") != PARENT_CONTRACT
        or int(parent.get("workflow_run_id", -1)) != 33473378340
        or int(parent.get("source_artifact_id", -1)) != 9787311310
        or parent.get("upstream_repository") != "roahmlab/DEFORM"
        or parent.get("upstream_revision")
        != "b73b8b8ecc033caefa693fab7898741d4e6dbeff"
    ):
        raise ValueError("parent evidence binding changed")
    if (
        int(crossfit.get("fold_count", 0)) != 5
        or int(crossfit.get("fit_trajectories_per_dlo", 0)) != 32
        or int(crossfit.get("selection_trajectories_per_dlo", 0)) != 12
        or int(crossfit.get("calibration_trajectories_per_dlo", 0)) != 12
    ):
        raise ValueError("source cross-fit or split contract changed")
    if not math.isclose(
        float(support.get("registered_regret_tolerance", math.nan)),
        0.05,
        abs_tol=0.0,
    ):
        raise ValueError("registered regret tolerance changed")
    alpha_grid = tuple(float(item) for item in support["conformal_alpha_grid"])
    if (
        not alpha_grid
        or any(not 0.0 < alpha < 1.0 for alpha in alpha_grid)
        or float(support["primary_conformal_alpha"]) not in alpha_grid
    ):
        raise ValueError("invalid conformal alpha grid")
    return value


def validate_request(path: Path, protocol: Mapping[str, object]) -> dict[str, object]:
    value = read_json(path)
    parent = protocol["parent"]
    assert isinstance(parent, dict)
    expected = {
        "contract": REQUEST_CONTRACT,
        "schema_version": 1,
        "status": "authorized",
        "mode": "source-crossfit-and-retrospective-held-audit",
        "parent_workflow_run_id": int(parent["workflow_run_id"]),
        "parent_source_artifact_id": int(parent["source_artifact_id"]),
        "upstream_revision": str(parent["upstream_revision"]),
        "target_tuning": False,
        "target_retries": False,
        "paper_claim_authorized": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"request field changed: {key}")
    extra = {
        "run_key",
        "expected_source_revision",
        "support_protocol_sha256",
        "parent_protocol_sha256",
    }
    if set(value) != set(expected) | extra:
        raise ValueError("request field roster changed")
    if not str(value["run_key"]).strip():
        raise ValueError("empty run key")
    revision = str(value["expected_source_revision"])
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("invalid expected source revision")
    return value


def _hash_order(names: Sequence[str], domain: str, dlo: str) -> tuple[str, ...]:
    def key(name: str) -> tuple[bytes, str]:
        payload = f"{domain}\0{dlo}\0{name}".encode()
        return hashlib.sha256(payload).digest(), name

    return tuple(sorted(names, key=key))


def _selected_settings(
    source_result: Mapping[str, object], dlo: str
) -> dict[str, object]:
    dlos = source_result.get("dlos")
    if not isinstance(dlos, dict):
        raise ValueError("parent source result lacks DLO map")
    record = dlos.get(dlo)
    if not isinstance(record, dict):
        raise ValueError(f"parent source result lacks {dlo}")
    settings = record.get("selected_settings")
    if not isinstance(settings, dict):
        raise ValueError(f"parent source result lacks {dlo} settings")
    required = {
        "cluster_count",
        "neighbors",
        "temperature_scale",
        "regret_tolerance",
    }
    if set(settings) != required:
        raise ValueError(f"{dlo} setting roster changed")
    return settings


def _crossfit_records(
    dataset_root: Path,
    parent_source_result: Mapping[str, object],
    parent_protocol: Protocol,
    support_protocol: Mapping[str, object],
) -> list[WindowRecord]:
    crossfit = support_protocol["source_crossfit"]
    assert isinstance(crossfit, dict)
    fold_count = int(crossfit["fold_count"])
    domain = str(crossfit["fold_domain"])
    all_records: list[WindowRecord] = []
    for dlo in DLOS:
        paths = trajectory_paths(dataset_root, dlo, "train")
        by_name = {path.name: path for path in paths}
        ordered = _hash_order(tuple(by_name), domain, dlo)
        folds = tuple(ordered[index::fold_count] for index in range(fold_count))
        if sorted(name for fold in folds for name in fold) != sorted(by_name):
            raise ValueError(f"{dlo}: cross-fit folds lost source trajectories")
        settings = _selected_settings(parent_source_result, dlo)
        for fold in folds:
            held = set(fold)
            fit_names = tuple(name for name in ordered if name not in held)
            features, residuals, _ = build_pool(paths, fit_names, parent_protocol)
            model = fit_model(
                features,
                residuals,
                cluster_count=int(settings["cluster_count"]),
                neighbors=int(settings["neighbors"]),
                temperature_scale=float(settings["temperature_scale"]),
                regret_tolerance=float(settings["regret_tolerance"]),
                protocol=parent_protocol,
            )
            held_paths = tuple(by_name[name] for name in fold)
            all_records.extend(_window_records(held_paths, model, parent_protocol, dlo))
    expected = 2 * 56 * len(window_starts(parent_protocol))
    stable_ids = [record.stable_id for record in all_records]
    if len(all_records) != expected or len(set(stable_ids)) != expected:
        raise ValueError("cross-fitted source decision roster changed")
    return all_records


def _support_split(
    records: Sequence[WindowRecord],
    support_protocol: Mapping[str, object],
) -> dict[str, set[str]]:
    crossfit = support_protocol["source_crossfit"]
    assert isinstance(crossfit, dict)
    counts = (
        int(crossfit["fit_trajectories_per_dlo"]),
        int(crossfit["selection_trajectories_per_dlo"]),
        int(crossfit["calibration_trajectories_per_dlo"]),
    )
    if sum(counts) != 56:
        raise ValueError("support split must consume 56 trajectories per DLO")
    domain = str(crossfit["support_split_domain"])
    result = {"fit": set(), "selection": set(), "calibration": set()}
    for dlo in DLOS:
        names = sorted({record.trajectory for record in records if record.dlo == dlo})
        if len(names) != 56:
            raise ValueError(f"{dlo}: source trajectory roster changed")
        ordered = _hash_order(names, domain, dlo)
        first = counts[0]
        second = first + counts[1]
        for name in ordered[:first]:
            result["fit"].add(f"{dlo}/{name}")
        for name in ordered[first:second]:
            result["selection"].add(f"{dlo}/{name}")
        for name in ordered[second:]:
            result["calibration"].add(f"{dlo}/{name}")
    if (
        len(result["fit"]) != 64
        or len(result["selection"]) != 24
        or len(result["calibration"]) != 24
        or result["fit"] & result["selection"]
        or result["fit"] & result["calibration"]
        or result["selection"] & result["calibration"]
    ):
        raise ValueError("support split is not disjoint and complete")
    return result


def _group(record: WindowRecord) -> str:
    return f"{record.dlo}/{record.trajectory}"


def support_feature(record: WindowRecord) -> FloatArray:
    """Return only quantities available before the held internal-node suffix."""

    decision = record.decision
    exact = decision.decision
    scores = decision.scores
    values = np.asarray(
        (
            record.certificate_source_regret_bound,
            float(scores["expected_action_gap"]),
            float(scores["expected_fallback_advantage"]),
            float(scores["hypothesis_action_agreement"]),
            float(scores["maximum_kernel_weight"]),
            float(scores["maximum_quotient_mass"]),
            float(scores["negative_residual_disagreement"]),
            float(scores["negative_unsupported_specificity"]),
            float(scores["quotient_concentration"]),
            float(exact.worst_case_regret[0]),
            float(exact.worst_case_regret[1]),
            float(exact.worst_case_regret[2]),
            record.current_frame / float(FRAME_COUNT - 1),
            float(record.dlo == "DLO5"),
            float(exact.certificate_action == 2),
        ),
        dtype=np.float64,
    )
    if values.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(values)):
        raise ValueError("invalid outcome-free support feature")
    return values


def _eligible(records: Sequence[WindowRecord]) -> list[WindowRecord]:
    return [
        record
        for record in records
        if record.decision.decision.certificate_action != 0
    ]


def _features(records: Sequence[WindowRecord]) -> FloatArray:
    if not records:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.asarray([support_feature(record) for record in records], dtype=np.float64)


def _tolerance(
    parent_protocol: Protocol,
    support_protocol: Mapping[str, object],
) -> float:
    support = support_protocol["support_misspecification"]
    assert isinstance(support, dict)
    value = float(support["registered_regret_tolerance"])
    if value not in parent_protocol.regret_tolerance_grid:
        raise ValueError("support tolerance is outside the parent grid")
    return value


def _bad_label(record: WindowRecord, tolerance: float) -> bool:
    action = record.decision.decision.certificate_action
    if action == 0:
        raise ValueError("support label requested for fallback decision")
    return bool(record.normalized_regret[action] > tolerance + ATOL)


def _harm_label(record: WindowRecord) -> bool:
    action = record.decision.decision.certificate_action
    if action == 0:
        raise ValueError("harm label requested for fallback decision")
    return bool(record.physical_mse[action] > record.fallback_mse + ATOL)


def _realized_excess(record: WindowRecord) -> float:
    action = record.decision.decision.certificate_action
    if action == 0:
        raise ValueError("regret excess requested for fallback decision")
    return float(
        record.normalized_regret[action]
        - record.decision.decision.worst_case_regret[action]
    )


def _standardization(values: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not len(array):
        raise ValueError("model fit requires a nonempty matrix")
    mean = np.mean(array, axis=0)
    scale = np.maximum(np.std(array, axis=0), 1e-9)
    return (array - mean[None, :]) / scale[None, :], mean, scale


def fit_ridge(values: FloatArray, targets: FloatArray, ridge: float) -> LinearRiskModel:
    standardized, mean, scale = _standardization(values)
    response = np.asarray(targets, dtype=np.float64)
    if response.shape != (len(standardized),) or not np.all(np.isfinite(response)):
        raise ValueError("invalid ridge targets")
    design = np.column_stack((np.ones(len(standardized)), standardized))
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ response)
    return LinearRiskModel(
        kind="ridge",
        feature_names=FEATURE_NAMES,
        feature_mean=mean,
        feature_scale=scale,
        coefficients=beta[1:],
        intercept=float(beta[0]),
        ridge=float(ridge),
    )


def fit_logistic(
    values: FloatArray,
    targets: BoolArray,
    ridge: float,
    *,
    iterations: int = 100,
) -> LinearRiskModel:
    standardized, mean, scale = _standardization(values)
    response = np.asarray(targets, dtype=np.float64)
    if (
        response.shape != (len(standardized),)
        or not np.all((response == 0.0) | (response == 1.0))
        or len(np.unique(response)) != 2
    ):
        raise ValueError("logistic fit requires both binary classes")
    design = np.column_stack((np.ones(len(standardized)), standardized))
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    beta = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(iterations):
        linear = np.clip(design @ beta, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        weights = np.maximum(probability * (1.0 - probability), 1e-6)
        gradient = design.T @ (probability - response) + penalty @ beta
        hessian = design.T @ (weights[:, None] * design) + penalty
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if float(np.max(np.abs(step))) <= 1e-10:
            break
    return LinearRiskModel(
        kind="logistic",
        feature_names=FEATURE_NAMES,
        feature_mean=mean,
        feature_scale=scale,
        coefficients=beta[1:],
        intercept=float(beta[0]),
        ridge=float(ridge),
    )


def _log_loss(targets: BoolArray, probabilities: FloatArray) -> float:
    truth = np.asarray(targets, dtype=np.float64)
    probability = np.clip(
        np.asarray(probabilities, dtype=np.float64), 1e-9, 1.0 - 1e-9
    )
    return float(
        -np.mean(
            truth * np.log(probability)
            + (1.0 - truth) * np.log1p(-probability)
        )
    )


def _select_ridge_model(
    fit_records: Sequence[WindowRecord],
    selection_records: Sequence[WindowRecord],
    ridge_grid: Sequence[float],
    *,
    kind: str,
    tolerance: float,
) -> tuple[LinearRiskModel, list[dict[str, object]]]:
    fit_values = _features(fit_records)
    selection_values = _features(selection_records)
    bad_fit = np.asarray(
        [_bad_label(record, tolerance) for record in fit_records], dtype=bool
    )
    bad_selection = np.asarray(
        [_bad_label(record, tolerance) for record in selection_records], dtype=bool
    )
    excess_fit = np.asarray(
        [_realized_excess(record) for record in fit_records], dtype=np.float64
    )
    excess_selection = np.asarray(
        [_realized_excess(record) for record in selection_records], dtype=np.float64
    )
    candidates: list[
        tuple[tuple[float, float], LinearRiskModel, dict[str, object]]
    ] = []
    for raw_ridge in ridge_grid:
        ridge = float(raw_ridge)
        if not math.isfinite(ridge) or ridge <= 0.0:
            raise ValueError("ridge grid must be finite and positive")
        if kind == "logistic":
            model = fit_logistic(fit_values, bad_fit, ridge)
            probabilities = model.predict_bad_probability(selection_values)
            metric = _log_loss(bad_selection, probabilities)
            record = {"ridge": ridge, "selection_log_loss": metric}
        elif kind == "ridge":
            model = fit_ridge(fit_values, excess_fit, ridge)
            predicted = model.predict_excess(selection_values)
            metric = float(np.mean(np.square(predicted - excess_selection)))
            record = {"ridge": ridge, "selection_mse": metric}
        else:
            raise ValueError("unknown support model kind")
        candidates.append(((metric, -ridge), model, record))
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], [item[2] for item in candidates]


def _choose_operational_threshold(
    selection_records: Sequence[WindowRecord],
    model: LinearRiskModel,
    support_protocol: Mapping[str, object],
    tolerance: float,
) -> dict[str, object]:
    operational = support_protocol["operational_risk_gate"]
    assert isinstance(operational, dict)
    records = list(selection_records)
    probabilities = model.predict_bad_probability(_features(records))
    order = sorted(
        range(len(records)),
        key=lambda index: (float(probabilities[index]), records[index].stable_id),
    )
    candidates: list[dict[str, object]] = []
    for raw_fraction in operational["selection_coverage_grid"]:
        fraction = float(raw_fraction)
        count = max(1, int(round(fraction * len(records))))
        selected = order[:count]
        selected_records = [records[index] for index in selected]
        bad = sum(_bad_label(record, tolerance) for record in selected_records)
        harm = sum(_harm_label(record) for record in selected_records)
        dlos = sorted({record.dlo for record in selected_records})
        boundary = float(probabilities[selected[-1]])
        record = {
            "requested_fraction": fraction,
            "selected_count": count,
            "selected_fraction": count / len(records),
            "bad_count": int(bad),
            "bad_fraction": bad / count,
            "harmful_count": int(harm),
            "harmful_fraction": harm / count,
            "dlos": dlos,
            "probability_threshold": boundary,
        }
        record["eligible"] = bool(
            count >= int(operational["minimum_selection_nonfallback_count"])
            and bad / count
            <= float(operational["maximum_selection_tolerance_breach_fraction"])
            + ATOL
            and harm / count
            <= float(operational["maximum_selection_harm_fraction"]) + ATOL
            and (
                not bool(operational["require_both_dlos"])
                or tuple(dlos) == DLOS
            )
        )
        candidates.append(record)
    eligible = [record for record in candidates if bool(record["eligible"])]
    if not eligible:
        return {
            "status": "no-source-selection-threshold",
            "probability_threshold": -1.0,
            "selected_candidate": None,
            "candidates": candidates,
        }
    selected = max(
        eligible,
        key=lambda item: (
            int(item["selected_count"]),
            -float(item["bad_fraction"]),
            -float(item["harmful_fraction"]),
        ),
    )
    return {
        "status": "source-selection-threshold-frozen",
        "probability_threshold": float(selected["probability_threshold"]),
        "selected_candidate": selected,
        "candidates": candidates,
    }


def _split_records(
    records: Sequence[WindowRecord], groups: set[str]
) -> list[WindowRecord]:
    return [record for record in records if _group(record) in groups]


def _method_summary(
    records: Sequence[WindowRecord],
    actions: Sequence[int],
    tolerance: float,
) -> dict[str, object]:
    if len(records) != len(actions):
        raise ValueError("action roster length changed")
    base = dict(summarize_method(records, actions))
    base["rmse_reduction"] = 1.0 - float(base["rmse_ratio_to_fallback"])
    nonfallback_indices = [
        index for index, action in enumerate(actions) if int(action) != 0
    ]
    breach = sum(
        records[index].normalized_regret[int(actions[index])] > tolerance + ATOL
        for index in nonfallback_indices
    )
    realized = [
        float(records[index].normalized_regret[int(actions[index])])
        for index in nonfallback_indices
    ]
    excess = [
        float(
            records[index].normalized_regret[int(actions[index])]
            - records[index].decision.decision.worst_case_regret[int(actions[index])]
        )
        for index in nonfallback_indices
    ]
    base.update(
        {
            "realized_tolerance_breach_count_nonfallback": int(breach),
            "realized_tolerance_breach_fraction_nonfallback": (
                breach / len(nonfallback_indices) if nonfallback_indices else 0.0
            ),
            "realized_nonfallback_regret_mean": (
                float(np.mean(realized)) if realized else 0.0
            ),
            "realized_nonfallback_regret_p95": (
                float(np.quantile(realized, 0.95)) if realized else 0.0
            ),
            "realized_support_excess_mean": (
                float(np.mean(excess)) if excess else 0.0
            ),
            "exact_fallback_violation_count": 0,
        }
    )
    return base


def _actions_for_operational(
    records: Sequence[WindowRecord], model: LinearRiskModel, threshold: float
) -> tuple[list[int], list[float]]:
    actions: list[int] = []
    risks: list[float] = []
    for record in records:
        inner = int(record.decision.decision.certificate_action)
        if inner == 0:
            actions.append(0)
            risks.append(1.0)
            continue
        risk = float(model.predict_bad_probability(support_feature(record))[0])
        risks.append(risk)
        actions.append(inner if risk <= threshold + ATOL else 0)
    return actions, risks


def _finite_group_quantile(scores: Sequence[float], alpha: float) -> float:
    values = sorted(float(value) for value in scores)
    if not values:
        raise ValueError("conformal calibration requires groups")
    rank = int(math.ceil((len(values) + 1) * (1.0 - float(alpha))))
    if rank > len(values):
        return math.inf
    return values[rank - 1]


def _conformal_calibration(
    calibration_records: Sequence[WindowRecord],
    calibration_groups: set[str],
    model: LinearRiskModel,
    alpha_grid: Sequence[float],
) -> dict[str, object]:
    eligible = _eligible(calibration_records)
    predictions = model.predict_excess(_features(eligible))
    by_group: dict[str, list[float]] = {group: [] for group in calibration_groups}
    for record, predicted in zip(eligible, predictions, strict=True):
        by_group[_group(record)].append(_realized_excess(record) - float(predicted))
    scores = {
        group: max(0.0, max(values, default=0.0))
        for group, values in sorted(by_group.items())
    }
    quantiles = {
        f"{float(alpha):.6g}": _finite_group_quantile(
            tuple(scores.values()), float(alpha)
        )
        for alpha in alpha_grid
    }
    return {
        "group_count": len(scores),
        "group_max_positive_residuals": scores,
        "quantiles": quantiles,
        "score_minimum": float(min(scores.values())),
        "score_median": float(np.median(list(scores.values()))),
        "score_maximum": float(max(scores.values())),
    }


def _actions_for_conformal(
    records: Sequence[WindowRecord],
    model: LinearRiskModel,
    quantile: float,
    tolerance: float,
) -> tuple[list[int], list[float]]:
    actions: list[int] = []
    bounds: list[float] = []
    for record in records:
        inner = int(record.decision.decision.certificate_action)
        source_bound = float(record.decision.decision.worst_case_regret[inner])
        if inner == 0:
            actions.append(0)
            bounds.append(source_bound)
            continue
        predicted = float(model.predict_excess(support_feature(record))[0])
        outer = max(source_bound, source_bound + predicted + quantile)
        bounds.append(outer)
        actions.append(inner if outer <= tolerance + ATOL else 0)
    return actions, bounds


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=bool)
    positive = int(np.count_nonzero(truth))
    negative = len(truth) - positive
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    rank_sum = float(np.sum(ranks[truth]))
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def _summary_by_dlo(
    records: Sequence[WindowRecord], actions: Sequence[int], tolerance: float
) -> dict[str, object]:
    result: dict[str, object] = {
        "combined": _method_summary(records, actions, tolerance)
    }
    for dlo in DLOS:
        indices = [
            index for index, record in enumerate(records) if record.dlo == dlo
        ]
        result[dlo] = _method_summary(
            [records[index] for index in indices],
            [actions[index] for index in indices],
            tolerance,
        )
    return result


def _bootstrap_difference(
    records: Sequence[WindowRecord],
    candidate_actions: Sequence[int],
    reference_actions: Sequence[int],
    replicates: int,
    seed: int,
) -> list[float]:
    grouped: dict[str, list[tuple[float, float]]] = {dlo: [] for dlo in DLOS}
    keys = sorted({_group(record) for record in records})
    for key in keys:
        indices = [
            index for index, record in enumerate(records) if _group(record) == key
        ]
        dlo = records[indices[0]].dlo
        candidate = math.sqrt(
            float(
                np.mean(
                    [
                        records[index].physical_mse[int(candidate_actions[index])]
                        for index in indices
                    ]
                )
            )
        )
        reference = math.sqrt(
            float(
                np.mean(
                    [
                        records[index].physical_mse[int(reference_actions[index])]
                        for index in indices
                    ]
                )
            )
        )
        grouped[dlo].append((candidate, reference))
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        differences: list[float] = []
        for dlo in DLOS:
            rows = grouped[dlo]
            selected = rng.integers(0, len(rows), size=len(rows))
            differences.extend(
                rows[index][0] - rows[index][1] for index in selected
            )
        values[replicate] = float(np.mean(differences))
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def source_stage(args: argparse.Namespace) -> int:
    parent_protocol = load_protocol(args.parent_protocol)
    support_protocol = _load_support_protocol(args.support_protocol)
    request = validate_request(args.request, support_protocol)
    if sha256_file(args.support_protocol) != request["support_protocol_sha256"]:
        raise ValueError("support protocol bytes changed")
    if sha256_file(args.parent_protocol) != request["parent_protocol_sha256"]:
        raise ValueError("parent protocol bytes changed")
    parent_source = read_json(args.parent_source_result)
    if (
        parent_source.get("contract") != PARENT_CONTRACT
        or parent_source.get("stage") != "source"
        or parent_source.get("all_source_gates_passed") is not True
    ):
        raise ValueError("parent source result is not an admitted source artifact")

    records = _crossfit_records(
        args.dataset_root,
        parent_source,
        parent_protocol,
        support_protocol,
    )
    split = _support_split(records, support_protocol)
    tolerance = _tolerance(parent_protocol, support_protocol)
    eligible = _eligible(records)
    fit_records = _split_records(eligible, split["fit"])
    selection_records = _split_records(eligible, split["selection"])
    calibration_records = _split_records(eligible, split["calibration"])
    if (
        not fit_records
        or not selection_records
        or not calibration_records
        or len({_group(record) for record in fit_records}) < 2
    ):
        raise ValueError("support model received an empty source split")

    support = support_protocol["support_misspecification"]
    assert isinstance(support, dict)
    logistic, logistic_candidates = _select_ridge_model(
        fit_records,
        selection_records,
        support["classification_ridge_grid"],
        kind="logistic",
        tolerance=tolerance,
    )
    regression, regression_candidates = _select_ridge_model(
        fit_records,
        selection_records,
        support["regression_ridge_grid"],
        kind="ridge",
        tolerance=tolerance,
    )
    operational = _choose_operational_threshold(
        selection_records,
        logistic,
        support_protocol,
        tolerance,
    )
    calibration = _conformal_calibration(
        calibration_records,
        split["calibration"],
        regression,
        support["conformal_alpha_grid"],
    )

    operational_actions, calibration_risks = _actions_for_operational(
        calibration_records,
        logistic,
        float(operational["probability_threshold"]),
    )
    inner_calibration_actions = [
        int(record.decision.decision.certificate_action)
        for record in calibration_records
    ]
    conformal_calibration: dict[str, object] = {}
    for alpha_text, raw_quantile in calibration["quantiles"].items():
        actions, bounds = _actions_for_conformal(
            calibration_records,
            regression,
            float(raw_quantile),
            tolerance,
        )
        conformal_calibration[alpha_text] = {
            "quantile": raw_quantile,
            "summary": _summary_by_dlo(calibration_records, actions, tolerance),
            "maximum_outer_bound": float(max(bounds)) if bounds else 0.0,
        }

    calibration_labels = [
        _bad_label(record, tolerance) for record in calibration_records
    ]
    risk_auc = _auc(calibration_risks, calibration_labels)
    model_record = {
        "contract": MODEL_CONTRACT,
        "schema_version": 1,
        "run_key": request["run_key"],
        "source_revision": request["expected_source_revision"],
        "feature_names": list(FEATURE_NAMES),
        "logistic": logistic.to_record(),
        "regression": regression.to_record(),
        "operational_threshold": operational,
        "conformal_calibration": calibration,
        "target_outcomes_read": False,
        "target_outcomes_used_for_selection": False,
        "forbidden_gate_inputs": support_protocol["forbidden_gate_inputs"],
    }
    model_record["model_id"] = canonical_sha256(model_record)

    source_result = {
        "contract": SOURCE_CONTRACT,
        "schema_version": 1,
        "status": "completed",
        "run_key": request["run_key"],
        "support_protocol_sha256": sha256_file(args.support_protocol),
        "parent_protocol_sha256": sha256_file(args.parent_protocol),
        "parent_source_result_sha256": sha256_file(args.parent_source_result),
        "model_id": model_record["model_id"],
        "source_crossfit": {
            "decision_count": len(records),
            "trajectory_count": len({_group(record) for record in records}),
            "certificate_nonfallback_count": len(eligible),
            "split_group_counts": {
                name: len(groups) for name, groups in split.items()
            },
        },
        "ridge_selection": {
            "logistic": logistic_candidates,
            "regression": regression_candidates,
        },
        "selection_threshold": operational,
        "calibration": {
            "inner_certificate": _summary_by_dlo(
                calibration_records,
                inner_calibration_actions,
                tolerance,
            ),
            "operational_gate": _summary_by_dlo(
                calibration_records,
                operational_actions,
                tolerance,
            ),
            "operational_bad_probability_auc": risk_auc,
            "conformal": conformal_calibration,
        },
        "target_data_read": False,
        "target_outcomes_used": False,
        "paper_claim_authorized": False,
        "classification": support_protocol["classification"],
        "claim_boundary": support_protocol["claim_boundary"],
    }
    source_result["result_id"] = canonical_sha256(source_result)
    output = args.output_root
    output.mkdir(parents=True, exist_ok=False)
    (output / "support_protocol.json").write_bytes(args.support_protocol.read_bytes())
    write_json(output / "support_model.json", model_record)
    write_json(output / "source_result.json", source_result)
    _write_source_report(output / "source_report.md", source_result)
    return 0


def _verify_source_root(
    root: Path,
    support_protocol: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    model = read_json(root / "support_model.json")
    result = read_json(root / "source_result.json")
    unsigned_model = {key: value for key, value in model.items() if key != "model_id"}
    if (
        model.get("contract") != MODEL_CONTRACT
        or result.get("contract") != SOURCE_CONTRACT
        or model.get("model_id") != canonical_sha256(unsigned_model)
        or result.get("model_id") != model.get("model_id")
        or result.get("support_protocol_sha256")
        != sha256_file(root / "support_protocol.json")
        or result.get("target_data_read") is not False
        or result.get("target_outcomes_used") is not False
        or model.get("target_outcomes_used_for_selection") is not False
        or tuple(model.get("feature_names", ())) != FEATURE_NAMES
        or model.get("forbidden_gate_inputs")
        != support_protocol["forbidden_gate_inputs"]
    ):
        raise ValueError("source support artifact does not verify")
    return model, result


def target_stage(args: argparse.Namespace) -> int:
    parent_protocol = load_protocol(args.parent_protocol)
    support_protocol = _load_support_protocol(args.support_protocol)
    request = validate_request(args.request, support_protocol)
    if sha256_file(args.support_protocol) != request["support_protocol_sha256"]:
        raise ValueError("support protocol bytes changed")
    source_root = args.source_root
    model_record, source_result = _verify_source_root(source_root, support_protocol)
    parent_models = load_models(args.parent_source_model)
    tolerance = _tolerance(parent_protocol, support_protocol)
    records: list[WindowRecord] = []
    for dlo in DLOS:
        paths = trajectory_paths(args.dataset_root, dlo, "eval")
        records.extend(_window_records(paths, parent_models[dlo], parent_protocol, dlo))
    expected = 2 * 14 * len(window_starts(parent_protocol))
    if len(records) != expected:
        raise ValueError("target decision roster changed")

    inner_actions = [
        int(record.decision.decision.certificate_action) for record in records
    ]
    logistic_record = model_record["logistic"]
    regression_record = model_record["regression"]
    if not isinstance(logistic_record, dict) or not isinstance(regression_record, dict):
        raise ValueError("support model records changed")
    logistic = LinearRiskModel.from_record(logistic_record)
    regression = LinearRiskModel.from_record(regression_record)
    threshold_record = model_record["operational_threshold"]
    if not isinstance(threshold_record, dict):
        raise ValueError("operational threshold record changed")
    operational_actions, operational_risks = _actions_for_operational(
        records,
        logistic,
        float(threshold_record["probability_threshold"]),
    )
    calibration = model_record["conformal_calibration"]
    if not isinstance(calibration, dict):
        raise ValueError("conformal calibration record changed")
    quantiles = calibration["quantiles"]
    if not isinstance(quantiles, dict):
        raise ValueError("conformal quantiles changed")

    methods: dict[str, object] = {
        "inner_certificate": _summary_by_dlo(records, inner_actions, tolerance),
        "operational_support_gate": _summary_by_dlo(
            records,
            operational_actions,
            tolerance,
        ),
    }
    per_decision: list[dict[str, object]] = []
    conformal_actions_by_alpha: dict[str, list[int]] = {}
    conformal_bounds_by_alpha: dict[str, list[float]] = {}
    for alpha_text, raw_quantile in quantiles.items():
        actions, bounds = _actions_for_conformal(
            records,
            regression,
            float(raw_quantile),
            tolerance,
        )
        conformal_actions_by_alpha[alpha_text] = actions
        conformal_bounds_by_alpha[alpha_text] = bounds
        methods[f"group_conformal_alpha_{alpha_text}"] = _summary_by_dlo(
            records,
            actions,
            tolerance,
        )

    support = support_protocol["support_misspecification"]
    assert isinstance(support, dict)
    primary_alpha = float(support["primary_conformal_alpha"])
    primary_text = f"{primary_alpha:.6g}"
    primary_actions = conformal_actions_by_alpha[primary_text]
    bootstrap = support_protocol["bootstrap"]
    assert isinstance(bootstrap, dict)
    target_result = {
        "contract": TARGET_CONTRACT,
        "schema_version": 1,
        "status": "completed",
        "run_key": request["run_key"],
        "source_result_id": source_result["result_id"],
        "support_model_id": model_record["model_id"],
        "parent_source_model_sha256": sha256_file(args.parent_source_model),
        "decision_count": len(records),
        "trajectory_count": len({_group(record) for record in records}),
        "methods": methods,
        "operational_bad_probability_auc": _auc(
            [
                risk
                for risk, record in zip(operational_risks, records, strict=True)
                if record.decision.decision.certificate_action != 0
            ],
            [
                _bad_label(record, tolerance)
                for record in records
                if record.decision.decision.certificate_action != 0
            ],
        ),
        "primary_conformal_alpha": primary_alpha,
        "primary_conformal_bootstrap_rmse_difference_vs_inner_mm": [
            1000.0 * value
            for value in _bootstrap_difference(
                records,
                primary_actions,
                inner_actions,
                int(bootstrap["replicates"]),
                int(bootstrap["seed"]),
            )
        ],
        "target_model_or_threshold_selection": False,
        "target_retries": False,
        "paper_claim_authorized": False,
        "classification": support_protocol["classification"],
        "claim_boundary": support_protocol["claim_boundary"],
    }
    inner = methods["inner_certificate"]
    operational_result = methods["operational_support_gate"]
    conformal_result = methods[f"group_conformal_alpha_{primary_text}"]
    assert isinstance(inner, dict)
    assert isinstance(operational_result, dict)
    assert isinstance(conformal_result, dict)
    target_result["pilot_conclusion"] = _classify_pilot(
        inner["combined"],
        operational_result["combined"],
        conformal_result["combined"],
    )
    target_result["result_id"] = canonical_sha256(target_result)

    for index, record in enumerate(records):
        item: dict[str, object] = {
            "stable_id": record.stable_id,
            "dlo": record.dlo,
            "trajectory": record.trajectory,
            "current_frame": record.current_frame,
            "inner_action": inner_actions[index],
            "operational_action": operational_actions[index],
            "operational_bad_probability": operational_risks[index],
            "realized_inner_regret": float(
                record.normalized_regret[inner_actions[index]]
            ),
            "inner_source_regret_bound": float(
                record.decision.decision.worst_case_regret[inner_actions[index]]
            ),
        }
        for alpha_text in quantiles:
            item[f"conformal_action_alpha_{alpha_text}"] = (
                conformal_actions_by_alpha[alpha_text][index]
            )
            item[f"conformal_upper_regret_alpha_{alpha_text}"] = (
                conformal_bounds_by_alpha[alpha_text][index]
            )
        per_decision.append(item)

    output = args.output_root
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "target_result.json", target_result)
    with (output / "per_decision.jsonl").open("w", encoding="utf-8") as handle:
        for item in per_decision:
            handle.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
    _write_target_report(output / "target_report.md", target_result)
    return 0


def _classify_pilot(
    inner: Mapping[str, object],
    operational: Mapping[str, object],
    conformal: Mapping[str, object],
) -> str:
    inner_count = int(inner["nonfallback_count"])
    operational_count = int(operational["nonfallback_count"])
    conformal_count = int(conformal["nonfallback_count"])
    operational_breach = float(
        operational["realized_tolerance_breach_fraction_nonfallback"]
    )
    inner_breach = float(inner["realized_tolerance_breach_fraction_nonfallback"])
    operational_gain = float(operational["rmse_reduction"])
    conformal_breach = float(
        conformal["realized_tolerance_breach_fraction_nonfallback"]
    )
    if (
        conformal_count >= max(10, int(math.ceil(0.25 * inner_count)))
        and conformal_breach <= 0.10 + ATOL
        and float(conformal["rmse_reduction"]) > 0.0
    ):
        return "strict-group-envelope-promising"
    if (
        operational_count >= max(10, int(math.ceil(0.25 * inner_count)))
        and operational_breach <= min(0.20, inner_breach - 0.15)
        and operational_gain > 0.0
    ):
        return "outcome-free-support-signal-promising-strict-envelope-not-yet"
    return "support-adequacy-pilot-negative-or-insufficient"


def _write_source_report(path: Path, result: Mapping[str, object]) -> None:
    selection = result["selection_threshold"]
    calibration = result["calibration"]
    source_crossfit = result["source_crossfit"]
    assert isinstance(selection, dict)
    assert isinstance(calibration, dict)
    assert isinstance(source_crossfit, dict)
    operational = calibration["operational_gate"]
    inner = calibration["inner_certificate"]
    assert isinstance(operational, dict)
    assert isinstance(inner, dict)
    op_combined = operational["combined"]
    inner_combined = inner["combined"]
    assert isinstance(op_combined, dict)
    assert isinstance(inner_combined, dict)
    lines = [
        "# DEFORM support-adequacy source stage",
        "",
        f"- Cross-fitted source trajectories: **{source_crossfit['trajectory_count']}**",
        f"- Cross-fitted source decisions: **{source_crossfit['decision_count']}**",
        f"- Inner-certificate nonfallback decisions: **{source_crossfit['certificate_nonfallback_count']}**",
        f"- Operational threshold status: **{selection['status']}**",
        f"- Calibration bad-probability AUC: **{calibration['operational_bad_probability_auc']}**",
        "",
        "| Calibration policy | Nonfallback | Tolerance breaches | RMSE gain |",
        "|---|---:|---:|---:|",
        (
            f"| Inner certificate | {inner_combined['nonfallback_count']} | "
            f"{inner_combined['realized_tolerance_breach_count_nonfallback']} | "
            f"{100.0 * float(inner_combined['rmse_reduction']):.2f}% |"
        ),
        (
            f"| Operational outer gate | {op_combined['nonfallback_count']} | "
            f"{op_combined['realized_tolerance_breach_count_nonfallback']} | "
            f"{100.0 * float(op_combined['rmse_reduction']):.2f}% |"
        ),
        "",
        str(result["claim_boundary"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_target_report(path: Path, result: Mapping[str, object]) -> None:
    methods = result["methods"]
    assert isinstance(methods, dict)
    lines = [
        "# DEFORM outer support-adequacy pilot",
        "",
        f"Classification: **{result['pilot_conclusion']}**",
        f"Operational risk AUC on held certificate decisions: **{result['operational_bad_probability_auc']}**",
        "",
        "| Method | Nonfallback | RMSE gain | Regret > 0.05 | Harmful |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in methods.items():
        assert isinstance(item, dict)
        combined = item["combined"]
        assert isinstance(combined, dict)
        lines.append(
            f"| `{name}` | {combined['nonfallback_count']} | "
            f"{100.0 * float(combined['rmse_reduction']):.2f}% | "
            f"{combined['realized_tolerance_breach_count_nonfallback']} | "
            f"{combined['harmful_nonfallback_count']} |"
        )
    lines.extend(["", str(result["claim_boundary"]), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    values = np.asarray(
        (
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (2.0, 2.0),
            (2.0, 3.0),
        ),
        dtype=np.float64,
    )
    logistic_targets = np.asarray((False, False, False, True, True, True))
    logistic = fit_logistic(values, logistic_targets, 1.0)
    probabilities = logistic.predict_bad_probability(values)
    if not (
        probabilities.shape == (6,)
        and np.all(np.isfinite(probabilities))
        and float(np.mean(probabilities[3:])) > float(np.mean(probabilities[:3]))
    ):
        raise RuntimeError("logistic self-test failed")
    ridge_targets = np.asarray((0.0, 0.5, 0.5, 1.0, 2.0, 2.5))
    ridge = fit_ridge(values, ridge_targets, 1.0)
    if not np.all(np.isfinite(ridge.predict_excess(values))):
        raise RuntimeError("ridge self-test failed")
    if not math.isclose(
        _finite_group_quantile((0.0, 0.1, 0.2, 0.3), 0.25),
        0.3,
        abs_tol=0.0,
    ):
        raise RuntimeError("finite-group quantile self-test failed")
    print("DEFORM support-adequacy v1 self-test passed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("source", "target", "self-test"))
    parser.add_argument("--support-protocol", type=Path)
    parser.add_argument("--parent-protocol", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--parent-source-result", type=Path)
    parser.add_argument("--parent-source-model", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def _require(args: argparse.Namespace, names: Sequence[str]) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise ValueError(f"missing required arguments: {', '.join(missing)}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "self-test":
        self_test()
        return 0
    common = (
        "support_protocol",
        "parent_protocol",
        "request",
        "dataset_root",
        "output_root",
    )
    if args.stage == "source":
        _require(args, (*common, "parent_source_result"))
        return source_stage(args)
    _require(args, (*common, "parent_source_model", "source_root"))
    return target_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
