"""Source-calibrated trajectory-level regret envelope for DEFORM DLO4/DLO5.

This retrospective follow-up keeps the original DLO4/DLO5 source split and
selected hyperparameters fixed.  The model is rebuilt from the original fit and
calibration trajectories only.  The untouched source-test trajectories then
calibrate a split-conformal inflation of the registered finite-support regret
bound.  Evaluation trajectories are opened only by the target command.

The independent calibration unit is one complete trajectory.  The conformity
score is simultaneous over all 19 registered decisions and the two nonfallback
actions.  The resulting guarantee is trajectory-marginal under exchangeability
within the declared DLO stratum; it is not pointwise validity or an unseen-object
safety guarantee.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from bayesian_phystwin.conformal_regret_envelope_v1 import (
    CONFORMAL_REGRET_ENVELOPE_CLAIM_BOUNDARY,
    support_robust_decision,
    trajectory_conformal_regret_envelope,
)

from ._common import (
    ATOL,
    DLOS,
    INTERNAL,
    Model,
    Protocol,
    canonical_sha256,
    extract_observation,
    load_protocol,
    load_trajectory,
    partition_names,
    read_json,
    sha256_file,
    trajectory_paths,
    window_starts,
    write_json,
)
from ._evaluation import evaluate_paths, load_models, save_models
from ._model import build_pool, decide, fit_model

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

CONTRACT: Final = "deform-dlo45-conformal-regret-envelope-v1"
REQUEST_CONTRACT: Final = "deform-dlo45-conformal-regret-envelope-request-v1"
PARENT_CONTRACT: Final = "deform-dlo45-decision-identifiability-v1"
SOURCE_RESULT_CONTRACT: Final = "deform-dlo45-decision-identifiability-v1"
SOURCE_MODEL_NAME: Final = "calibration_models.npz"
SOURCE_ENVELOPE_NAME: Final = "source_envelope.json"
SOURCE_SEAL_NAME: Final = "source_seal.json"


@dataclass(frozen=True)
class EnvelopeProtocol:
    parent_workflow_run_id: int
    parent_source_result_sha256: str
    dataset_repository: str
    dataset_commit: str
    calibration_partition: str
    calibration_unit: str
    candidate_action_mask: tuple[bool, ...]
    miscoverage_levels: tuple[float, ...]
    primary_miscoverage: float
    regret_budget_grid: tuple[float, ...]
    primary_regret_budget: float
    bootstrap_replicates: int
    bootstrap_seed: int
    claim_boundary: str


@dataclass(frozen=True)
class WindowMeasurement:
    stable_id: str
    dlo: str
    trajectory: str
    current_frame: int
    registered_regret: FloatArray
    realized_regret: FloatArray
    physical_mse: FloatArray
    fallback_mse: float
    base_certificate_action: int


@dataclass(frozen=True)
class PolicySummary:
    decision_count: int
    nonfallback_count: int
    nonfallback_fraction: float
    harmful_nonfallback_count: int
    harmful_fraction_nonfallback: float
    budget_violation_count_nonfallback: int
    budget_violation_fraction_nonfallback: float
    trajectory_budget_violation_count: int
    trajectory_budget_violation_fraction: float
    rmse_mm: float
    fallback_rmse_mm: float
    rmse_ratio_to_fallback: float
    rmse_reduction: float
    mean_realized_regret: float
    p95_realized_regret: float
    action_counts: tuple[int, ...]
    trajectory_improvement_mean: float
    trajectory_improvement_ci95: tuple[float, float]
    maximum_trajectory_ratio: float


def _tuple_of(value: object, cast: type) -> tuple:
    if not isinstance(value, list):
        raise ValueError("protocol arrays must be JSON arrays")
    return tuple(cast(item) for item in value)


def load_envelope_protocol(path: Path) -> EnvelopeProtocol:
    value = read_json(path)
    calibration = value.get("calibration")
    decision = value.get("decision")
    bootstrap = value.get("bootstrap")
    if (
        value.get("contract") != CONTRACT
        or value.get("schema_version") != 1
        or value.get("parent_contract") != PARENT_CONTRACT
        or not isinstance(calibration, dict)
        or not isinstance(decision, dict)
        or not isinstance(bootstrap, dict)
    ):
        raise ValueError("invalid conformal-regret-envelope protocol")
    assert isinstance(calibration, dict)
    assert isinstance(decision, dict)
    assert isinstance(bootstrap, dict)
    levels = _tuple_of(calibration.get("miscoverage_levels"), float)
    budgets = _tuple_of(decision.get("regret_budget_grid"), float)
    mask = _tuple_of(calibration.get("candidate_action_mask"), bool)
    result = EnvelopeProtocol(
        parent_workflow_run_id=int(value["parent_workflow_run_id"]),
        parent_source_result_sha256=str(value["parent_source_result_sha256"]),
        dataset_repository=str(value["dataset_repository"]),
        dataset_commit=str(value["dataset_commit"]),
        calibration_partition=str(calibration["partition"]),
        calibration_unit=str(calibration["unit"]),
        candidate_action_mask=mask,
        miscoverage_levels=levels,
        primary_miscoverage=float(calibration["primary_miscoverage"]),
        regret_budget_grid=budgets,
        primary_regret_budget=float(decision["primary_regret_budget"]),
        bootstrap_replicates=int(bootstrap["replicates"]),
        bootstrap_seed=int(bootstrap["seed"]),
        claim_boundary=str(value["claim_boundary"]),
    )
    if (
        result.calibration_partition != "source_test"
        or result.calibration_unit != "complete_trajectory"
        or result.candidate_action_mask != (False, True, True)
        or not result.miscoverage_levels
        or result.primary_miscoverage not in result.miscoverage_levels
        or any(not 0.0 < item < 1.0 for item in result.miscoverage_levels)
        or not result.regret_budget_grid
        or result.primary_regret_budget not in result.regret_budget_grid
        or any(item < 0.0 for item in result.regret_budget_grid)
        or result.bootstrap_replicates < 1
    ):
        raise ValueError("invalid frozen calibration or decision settings")
    return result


def validate_request(
    path: Path, protocol: EnvelopeProtocol, protocol_sha256: str
) -> dict[str, object]:
    value = read_json(path)
    if (
        value.get("contract") != REQUEST_CONTRACT
        or value.get("schema_version") != 1
        or value.get("status") != "authorized"
        or value.get("parent_workflow_run_id") != protocol.parent_workflow_run_id
        or value.get("protocol_sha256") != protocol_sha256
        or tuple(value.get("dlos", ())) != DLOS
        or value.get("source_only_calibration") is not True
        or value.get("target_tuning") is not False
        or value.get("target_retries") is not False
        or value.get("report_complete_frontier") is not True
        or not isinstance(value.get("run_key"), str)
        or not str(value["run_key"]).strip()
    ):
        raise ValueError("invalid conformal-regret-envelope request")
    return value


def _source_record_for_dlo(
    parent_source_result: Mapping[str, object], dlo: str
) -> Mapping[str, object]:
    dlos = parent_source_result.get("dlos")
    if not isinstance(dlos, dict) or not isinstance(dlos.get(dlo), dict):
        raise ValueError(f"missing parent source record for {dlo}")
    return dlos[dlo]  # type: ignore[return-value]


def _rebuild_calibration_model(
    train_paths: tuple[Path, ...],
    dlo: str,
    protocol: Protocol,
    parent_record: Mapping[str, object],
) -> tuple[Model, tuple[Path, ...], dict[str, object]]:
    names = tuple(path.name for path in train_paths)
    split = partition_names(names, dlo, protocol)
    parent_partition = parent_record.get("partition")
    settings = parent_record.get("selected_settings")
    if not isinstance(parent_partition, dict) or not isinstance(settings, dict):
        raise ValueError(f"invalid parent model-selection record for {dlo}")
    expected_partition = {name: list(values) for name, values in split.items()}
    if parent_partition != expected_partition:
        raise ValueError(f"source partition drift for {dlo}")

    model_names = split["fit"] + split["calibration"]
    features, residuals, _ = build_pool(train_paths, model_names, protocol)
    model = fit_model(
        features,
        residuals,
        cluster_count=int(settings["cluster_count"]),
        neighbors=int(settings["neighbors"]),
        temperature_scale=float(settings["temperature_scale"]),
        regret_tolerance=float(settings["regret_tolerance"]),
        protocol=protocol,
    )
    source_test_names = set(split["source_test"])
    source_test_paths = tuple(
        path for path in train_paths if path.name in source_test_names
    )
    reconstructed = evaluate_paths(source_test_paths, model, protocol)
    parent_source_test = parent_record.get("source_test")
    if not isinstance(parent_source_test, dict):
        raise ValueError(f"missing parent source-test record for {dlo}")
    _assert_source_test_parity(reconstructed, parent_source_test, dlo)
    return (
        model,
        source_test_paths,
        {
            "partition": expected_partition,
            "selected_settings": dict(settings),
            "reconstructed_source_test": reconstructed,
        },
    )


def _assert_source_test_parity(
    reconstructed: Mapping[str, object],
    parent: Mapping[str, object],
    dlo: str,
) -> None:
    if int(reconstructed["decision_count"]) != int(parent["decision_count"]):
        raise RuntimeError(f"source-test decision-count parity failed for {dlo}")
    reconstructed_aggregate = reconstructed.get("aggregate")
    parent_aggregate = parent.get("aggregate")
    if not isinstance(reconstructed_aggregate, dict) or not isinstance(
        parent_aggregate, dict
    ):
        raise RuntimeError(f"source-test aggregate missing for {dlo}")
    for method in ("fallback", "certificate", "jeffrey_point"):
        first = reconstructed_aggregate.get(method)
        second = parent_aggregate.get(method)
        if not isinstance(first, dict) or not isinstance(second, dict):
            raise RuntimeError(f"source-test method record missing for {dlo}/{method}")
        if first.get("action_counts") != second.get("action_counts"):
            raise RuntimeError(f"source-test action parity failed for {dlo}/{method}")
        for key in ("rmse_mm", "rmse_ratio_to_fallback"):
            if not math.isclose(
                float(first[key]),
                float(second[key]),
                rel_tol=0.0,
                abs_tol=1e-11,
            ):
                raise RuntimeError(
                    f"source-test metric parity failed for {dlo}/{method}/{key}"
                )


def _measure_paths(
    paths: tuple[Path, ...], model: Model, protocol: Protocol, dlo: str
) -> list[WindowMeasurement]:
    records: list[WindowMeasurement] = []
    for path in paths:
        trajectory = load_trajectory(path)
        for current in window_starts(protocol):
            observation = extract_observation(trajectory, current, protocol)
            decision = decide(observation.feature, model, protocol)
            truth = trajectory[
                current + 1 : current + 1 + protocol.horizon_frames,
                INTERNAL,
                :,
            ].copy()
            actual_residual = (truth - observation.baseline).reshape(
                -1
            ) / observation.length_scale
            actions = model.action_scales[:, None] * decision.correction[None, :]
            normalized_mse = np.mean(
                np.square(actual_residual[None, :] - actions), axis=1
            )
            physical_mse = normalized_mse * observation.length_scale**2
            best = float(np.min(normalized_mse))
            denominator = max(float(normalized_mse[0]), model.loss_floor)
            realized_regret = (normalized_mse - best) / denominator
            records.append(
                WindowMeasurement(
                    stable_id=f"{dlo}/{path.name}/{current}",
                    dlo=dlo,
                    trajectory=path.name,
                    current_frame=current,
                    registered_regret=np.asarray(
                        decision.worst_case_regret, dtype=np.float64
                    ),
                    realized_regret=np.asarray(realized_regret, dtype=np.float64),
                    physical_mse=np.asarray(physical_mse, dtype=np.float64),
                    fallback_mse=float(physical_mse[0]),
                    base_certificate_action=int(decision.certificate_action),
                )
            )
    return records


def _trajectory_tensors(
    records: Sequence[WindowMeasurement], action_count: int
) -> tuple[tuple[str, ...], FloatArray, FloatArray, IntArray]:
    grouped: dict[str, list[WindowMeasurement]] = defaultdict(list)
    for record in records:
        grouped[record.trajectory].append(record)
    names = tuple(sorted(grouped))
    if not names:
        raise ValueError("no trajectory records")
    decision_counts = {len(grouped[name]) for name in names}
    if len(decision_counts) != 1:
        raise ValueError("trajectory decision counts differ")
    decision_count = decision_counts.pop()
    realized = np.empty((len(names), decision_count, action_count), dtype=np.float64)
    registered = np.empty_like(realized)
    base_actions = np.empty((len(names), decision_count), dtype=np.int64)
    for trajectory_index, name in enumerate(names):
        ordered = sorted(grouped[name], key=lambda item: item.current_frame)
        for decision_index, record in enumerate(ordered):
            if record.realized_regret.shape != (
                action_count,
            ) or record.registered_regret.shape != (action_count,):
                raise ValueError("action count changed within trajectory records")
            realized[trajectory_index, decision_index] = record.realized_regret
            registered[trajectory_index, decision_index] = record.registered_regret
            base_actions[trajectory_index, decision_index] = (
                record.base_certificate_action
            )
    return names, realized, registered, base_actions


def _selected_action_tensors(
    realized: FloatArray,
    registered: FloatArray,
    base_actions: IntArray,
    fallback_action: int,
) -> tuple[FloatArray, FloatArray]:
    selected_realized = np.take_along_axis(
        realized, base_actions[..., None], axis=2
    ).copy()
    selected_registered = np.take_along_axis(
        registered, base_actions[..., None], axis=2
    ).copy()
    fallback = base_actions == fallback_action
    selected_realized[fallback] = 0.0
    selected_registered[fallback] = 0.0
    return selected_realized, selected_registered


def _envelope_records(
    realized: FloatArray,
    registered: FloatArray,
    base_actions: IntArray,
    protocol: EnvelopeProtocol,
) -> dict[str, object]:
    selected_realized, selected_registered = _selected_action_tensors(
        realized, registered, base_actions, 0
    )
    result: dict[str, object] = {}
    for alpha in protocol.miscoverage_levels:
        all_actions = trajectory_conformal_regret_envelope(
            realized,
            registered,
            miscoverage=alpha,
            candidate_action_mask=np.asarray(protocol.candidate_action_mask),
        )
        selected = trajectory_conformal_regret_envelope(
            selected_realized,
            selected_registered,
            miscoverage=alpha,
        )
        result[f"{alpha:.6f}"] = {
            "all_nonfallback_actions": all_actions.summary()
            | {
                "trajectory_nonconformity_scores": (
                    all_actions.trajectory_nonconformity_scores.tolist()
                )
            },
            "base_certificate_selected_action": selected.summary()
            | {
                "trajectory_nonconformity_scores": (
                    selected.trajectory_nonconformity_scores.tolist()
                )
            },
        }
    return result


def run_source(
    *,
    parent_protocol_path: Path,
    envelope_protocol_path: Path,
    parent_source_result_path: Path,
    request_path: Path,
    dataset_root: Path,
    output_root: Path,
) -> dict[str, object]:
    parent_protocol = load_protocol(parent_protocol_path)
    envelope_protocol = load_envelope_protocol(envelope_protocol_path)
    request = validate_request(
        request_path,
        envelope_protocol,
        sha256_file(envelope_protocol_path),
    )
    parent_source_result = read_json(parent_source_result_path)
    if (
        parent_source_result.get("contract") != SOURCE_RESULT_CONTRACT
        or parent_source_result.get("stage") != "source"
        or sha256_file(parent_source_result_path)
        != envelope_protocol.parent_source_result_sha256
    ):
        raise ValueError("parent source result identity mismatch")

    models: dict[str, Model] = {}
    source_records: dict[str, object] = {}
    pooled_realized: list[FloatArray] = []
    pooled_registered: list[FloatArray] = []
    pooled_actions: list[IntArray] = []
    for dlo in DLOS:
        train_paths = trajectory_paths(dataset_root, dlo, "train")
        parent_record = _source_record_for_dlo(parent_source_result, dlo)
        model, source_test_paths, reconstruction = _rebuild_calibration_model(
            train_paths, dlo, parent_protocol, parent_record
        )
        models[dlo] = model
        measurements = _measure_paths(source_test_paths, model, parent_protocol, dlo)
        names, realized, registered, base_actions = _trajectory_tensors(
            measurements, len(model.action_scales)
        )
        pooled_realized.append(realized)
        pooled_registered.append(registered)
        pooled_actions.append(base_actions)
        source_records[dlo] = {
            **reconstruction,
            "calibration_trajectory_names": list(names),
            "calibration_trajectory_count": len(names),
            "decision_count_per_trajectory": int(realized.shape[1]),
            "envelopes": _envelope_records(
                realized,
                registered,
                base_actions,
                envelope_protocol,
            ),
        }

    pooled_realized_array = np.concatenate(pooled_realized, axis=0)
    pooled_registered_array = np.concatenate(pooled_registered, axis=0)
    pooled_actions_array = np.concatenate(pooled_actions, axis=0)
    source = {
        "contract": CONTRACT,
        "schema_version": 1,
        "stage": "source-calibration",
        "parent_contract": PARENT_CONTRACT,
        "parent_protocol_sha256": sha256_file(parent_protocol_path),
        "parent_source_result_sha256": sha256_file(parent_source_result_path),
        "envelope_protocol_sha256": sha256_file(envelope_protocol_path),
        "request_sha256": sha256_file(request_path),
        "run_key": request["run_key"],
        "target_data_read": False,
        "target_outcomes_used": False,
        "dlos": source_records,
        "pooled_normalized_regret": {
            "calibration_trajectory_count": int(pooled_realized_array.shape[0]),
            "envelopes": _envelope_records(
                pooled_realized_array,
                pooled_registered_array,
                pooled_actions_array,
                envelope_protocol,
            ),
            "assumption": (
                "Pooled validity requires exchangeability under the declared "
                "balanced DLO4/DLO5 trajectory mixture; per-DLO envelopes retain "
                "the narrower within-DLO interpretation."
            ),
        },
        "claim_boundary": envelope_protocol.claim_boundary,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = output_root / SOURCE_MODEL_NAME
    envelope_path = output_root / SOURCE_ENVELOPE_NAME
    save_models(model_path, models)
    write_json(envelope_path, source)
    seal = {
        "contract": CONTRACT,
        "schema_version": 1,
        "stage": "source-seal",
        "source_model_sha256": sha256_file(model_path),
        "source_envelope_sha256": sha256_file(envelope_path),
        "envelope_protocol_sha256": sha256_file(envelope_protocol_path),
        "request_sha256": sha256_file(request_path),
        "canonical_source_sha256": canonical_sha256(source),
        "target_data_read": False,
        "target_outcomes_used": False,
    }
    write_json(output_root / SOURCE_SEAL_NAME, seal)
    return source


def _bootstrap_trajectory_improvement(
    selected_by_trajectory: Sequence[FloatArray],
    fallback_by_trajectory: Sequence[FloatArray],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, tuple[float, float]]:
    improvements = np.asarray(
        [
            1.0
            - math.sqrt(float(np.mean(selected)))
            / max(math.sqrt(float(np.mean(fallback))), 1e-12)
            for selected, fallback in zip(
                selected_by_trajectory, fallback_by_trajectory, strict=True
            )
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, len(improvements), size=len(improvements))
        bootstrap[index] = float(np.mean(improvements[sample]))
    return float(np.mean(improvements)), (
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    )


def _summarize_policy(
    records: Sequence[WindowMeasurement],
    selected_actions: Sequence[int],
    *,
    regret_budget: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> PolicySummary:
    if len(records) != len(selected_actions) or not records:
        raise ValueError("policy actions must match nonempty records")
    action_count = len(records[0].physical_mse)
    action_counts = np.zeros(action_count, dtype=np.int64)
    selected_mse: list[float] = []
    fallback_mse: list[float] = []
    realized_regret: list[float] = []
    harmful: list[bool] = []
    budget_violation: list[bool] = []
    trajectory_selected: dict[str, list[float]] = defaultdict(list)
    trajectory_fallback: dict[str, list[float]] = defaultdict(list)
    trajectory_violation: dict[str, bool] = defaultdict(bool)
    for record, action in zip(records, selected_actions, strict=True):
        if not 0 <= action < action_count:
            raise ValueError("selected action is out of range")
        action_counts[action] += 1
        mse = float(record.physical_mse[action])
        selected_mse.append(mse)
        fallback_mse.append(record.fallback_mse)
        regret = float(record.realized_regret[action])
        realized_regret.append(regret)
        is_nonfallback = action != 0
        is_harmful = bool(is_nonfallback and mse > record.fallback_mse + ATOL)
        is_violation = bool(is_nonfallback and regret > regret_budget + ATOL)
        harmful.append(is_harmful)
        budget_violation.append(is_violation)
        trajectory_key = f"{record.dlo}/{record.trajectory}"
        trajectory_selected[trajectory_key].append(mse)
        trajectory_fallback[trajectory_key].append(record.fallback_mse)
        trajectory_violation[trajectory_key] |= is_violation

    selected_array = np.asarray(selected_mse, dtype=np.float64)
    fallback_array = np.asarray(fallback_mse, dtype=np.float64)
    regret_array = np.asarray(realized_regret, dtype=np.float64)
    nonfallback_count = int(len(records) - action_counts[0])
    harmful_count = int(np.count_nonzero(harmful))
    violation_count = int(np.count_nonzero(budget_violation))
    selected_grouped = [
        np.asarray(trajectory_selected[name], dtype=np.float64)
        for name in sorted(trajectory_selected)
    ]
    fallback_grouped = [
        np.asarray(trajectory_fallback[name], dtype=np.float64)
        for name in sorted(trajectory_fallback)
    ]
    trajectory_mean, trajectory_ci = _bootstrap_trajectory_improvement(
        selected_grouped,
        fallback_grouped,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    trajectory_ratios = [
        math.sqrt(float(np.mean(selected)))
        / max(math.sqrt(float(np.mean(fallback))), 1e-12)
        for selected, fallback in zip(selected_grouped, fallback_grouped, strict=True)
    ]
    selected_rmse = math.sqrt(float(np.mean(selected_array)))
    fallback_rmse_value = math.sqrt(float(np.mean(fallback_array)))
    ratio = selected_rmse / max(fallback_rmse_value, 1e-12)
    trajectory_violation_count = int(sum(trajectory_violation.values()))
    trajectory_count = len(trajectory_violation)
    return PolicySummary(
        decision_count=len(records),
        nonfallback_count=nonfallback_count,
        nonfallback_fraction=nonfallback_count / len(records),
        harmful_nonfallback_count=harmful_count,
        harmful_fraction_nonfallback=harmful_count / max(nonfallback_count, 1),
        budget_violation_count_nonfallback=violation_count,
        budget_violation_fraction_nonfallback=(
            violation_count / max(nonfallback_count, 1)
        ),
        trajectory_budget_violation_count=trajectory_violation_count,
        trajectory_budget_violation_fraction=(
            trajectory_violation_count / max(trajectory_count, 1)
        ),
        rmse_mm=1000.0 * selected_rmse,
        fallback_rmse_mm=1000.0 * fallback_rmse_value,
        rmse_ratio_to_fallback=ratio,
        rmse_reduction=1.0 - ratio,
        mean_realized_regret=float(np.mean(regret_array)),
        p95_realized_regret=float(np.quantile(regret_array, 0.95)),
        action_counts=tuple(int(item) for item in action_counts),
        trajectory_improvement_mean=trajectory_mean,
        trajectory_improvement_ci95=trajectory_ci,
        maximum_trajectory_ratio=max(trajectory_ratios),
    )


def _as_json(value: PolicySummary) -> dict[str, object]:
    return {
        "decision_count": value.decision_count,
        "nonfallback_count": value.nonfallback_count,
        "nonfallback_fraction": value.nonfallback_fraction,
        "harmful_nonfallback_count": value.harmful_nonfallback_count,
        "harmful_fraction_nonfallback": value.harmful_fraction_nonfallback,
        "budget_violation_count_nonfallback": (
            value.budget_violation_count_nonfallback
        ),
        "budget_violation_fraction_nonfallback": (
            value.budget_violation_fraction_nonfallback
        ),
        "trajectory_budget_violation_count": (value.trajectory_budget_violation_count),
        "trajectory_budget_violation_fraction": (
            value.trajectory_budget_violation_fraction
        ),
        "rmse_mm": value.rmse_mm,
        "fallback_rmse_mm": value.fallback_rmse_mm,
        "rmse_ratio_to_fallback": value.rmse_ratio_to_fallback,
        "rmse_reduction": value.rmse_reduction,
        "mean_realized_regret": value.mean_realized_regret,
        "p95_realized_regret": value.p95_realized_regret,
        "action_counts": list(value.action_counts),
        "trajectory_improvement_mean": value.trajectory_improvement_mean,
        "trajectory_improvement_ci95": list(value.trajectory_improvement_ci95),
        "maximum_trajectory_ratio": value.maximum_trajectory_ratio,
    }


def _radius(
    source: Mapping[str, object],
    *,
    dlo: str,
    grouping: str,
    alpha_key: str,
    envelope_kind: str,
) -> float:
    if grouping == "per_dlo":
        dlos = source.get("dlos")
        if not isinstance(dlos, dict) or not isinstance(dlos.get(dlo), dict):
            raise ValueError(f"missing source envelope for {dlo}")
        record = dlos[dlo]
    elif grouping == "pooled":
        record = source.get("pooled_normalized_regret")
    else:
        raise ValueError(f"unknown grouping {grouping!r}")
    if not isinstance(record, dict):
        raise ValueError("invalid source envelope record")
    envelopes = record.get("envelopes")
    if not isinstance(envelopes, dict) or not isinstance(
        envelopes.get(alpha_key), dict
    ):
        raise ValueError("missing source envelope level")
    level = envelopes[alpha_key]
    assert isinstance(level, dict)
    kind = level.get(envelope_kind)
    if not isinstance(kind, dict):
        raise ValueError("missing source envelope kind")
    return float(kind["radius"])


def _trajectory_coverage(
    realized: FloatArray,
    registered: FloatArray,
    base_actions: IntArray,
    *,
    radius: float,
    envelope_kind: str,
    candidate_action_mask: tuple[bool, ...],
) -> dict[str, object]:
    if envelope_kind == "all_nonfallback_actions":
        mask = np.asarray(candidate_action_mask, dtype=np.bool_)
        scores = np.max(realized[:, :, mask] - registered[:, :, mask], axis=(1, 2))
    elif envelope_kind == "base_certificate_selected_action":
        selected_realized, selected_registered = _selected_action_tensors(
            realized, registered, base_actions, 0
        )
        scores = np.max(selected_realized - selected_registered, axis=(1, 2))
    else:
        raise ValueError("unknown envelope kind")
    covered = scores <= radius + ATOL
    return {
        "trajectory_count": int(len(scores)),
        "covered_trajectory_count": int(np.count_nonzero(covered)),
        "empirical_trajectory_coverage": float(np.mean(covered)),
        "trajectory_scores": scores.tolist(),
        "covered_mask": covered.tolist(),
    }


def run_target(
    *,
    parent_protocol_path: Path,
    envelope_protocol_path: Path,
    request_path: Path,
    dataset_root: Path,
    source_root: Path,
    output_root: Path,
) -> dict[str, object]:
    parent_protocol = load_protocol(parent_protocol_path)
    envelope_protocol = load_envelope_protocol(envelope_protocol_path)
    request = validate_request(
        request_path,
        envelope_protocol,
        sha256_file(envelope_protocol_path),
    )
    source_path = source_root / SOURCE_ENVELOPE_NAME
    models_path = source_root / SOURCE_MODEL_NAME
    seal_path = source_root / SOURCE_SEAL_NAME
    source = read_json(source_path)
    seal = read_json(seal_path)
    if (
        source.get("contract") != CONTRACT
        or source.get("stage") != "source-calibration"
        or seal.get("contract") != CONTRACT
        or seal.get("stage") != "source-seal"
        or seal.get("source_model_sha256") != sha256_file(models_path)
        or seal.get("source_envelope_sha256") != sha256_file(source_path)
        or seal.get("envelope_protocol_sha256") != sha256_file(envelope_protocol_path)
        or seal.get("request_sha256") != sha256_file(request_path)
    ):
        raise ValueError("source envelope seal mismatch")
    models = load_models(models_path)

    result_by_dlo: dict[str, object] = {}
    per_decision: list[dict[str, object]] = []
    combined_records: list[WindowMeasurement] = []
    combined_actions_by_policy: dict[str, list[int]] = defaultdict(list)
    for dlo_index, dlo in enumerate(DLOS):
        eval_paths = trajectory_paths(dataset_root, dlo, "eval")
        records = _measure_paths(eval_paths, models[dlo], parent_protocol, dlo)
        combined_records.extend(records)
        names, realized, registered, base_actions = _trajectory_tensors(
            records, len(models[dlo].action_scales)
        )
        dlo_result: dict[str, object] = {
            "trajectory_names": list(names),
            "trajectory_count": len(names),
            "decision_count": len(records),
            "policies": {},
            "coverage": {},
        }
        policies = dlo_result["policies"]
        coverage = dlo_result["coverage"]
        assert isinstance(policies, dict)
        assert isinstance(coverage, dict)
        for grouping in ("per_dlo", "pooled"):
            grouping_policies: dict[str, object] = {}
            grouping_coverage: dict[str, object] = {}
            policies[grouping] = grouping_policies
            coverage[grouping] = grouping_coverage
            for alpha in envelope_protocol.miscoverage_levels:
                alpha_key = f"{alpha:.6f}"
                alpha_policies: dict[str, object] = {}
                alpha_coverage: dict[str, object] = {}
                grouping_policies[alpha_key] = alpha_policies
                grouping_coverage[alpha_key] = alpha_coverage
                for envelope_kind in (
                    "base_certificate_selected_action",
                    "all_nonfallback_actions",
                ):
                    radius = _radius(
                        source,
                        dlo=dlo,
                        grouping=grouping,
                        alpha_key=alpha_key,
                        envelope_kind=envelope_kind,
                    )
                    alpha_coverage[envelope_kind] = {
                        "radius": radius,
                        **_trajectory_coverage(
                            realized,
                            registered,
                            base_actions,
                            radius=radius,
                            envelope_kind=envelope_kind,
                            candidate_action_mask=(
                                envelope_protocol.candidate_action_mask
                            ),
                        ),
                    }
                    kind_policies: dict[str, object] = {}
                    alpha_policies[envelope_kind] = kind_policies
                    for budget_index, budget in enumerate(
                        envelope_protocol.regret_budget_grid
                    ):
                        selected_actions: list[int] = []
                        for record in records:
                            if envelope_kind == "base_certificate_selected_action":
                                action = record.base_certificate_action
                                bound = float(record.registered_regret[action])
                                selected = (
                                    action
                                    if action != 0
                                    and np.isfinite(radius)
                                    and bound + radius <= budget + ATOL
                                    else 0
                                )
                            else:
                                selected = support_robust_decision(
                                    record.registered_regret,
                                    conformal_radius=radius,
                                    regret_tolerance=budget,
                                    fallback_action_index=0,
                                ).selected_action_index
                            selected_actions.append(selected)
                        summary = _summarize_policy(
                            records,
                            selected_actions,
                            regret_budget=budget,
                            bootstrap_replicates=(
                                envelope_protocol.bootstrap_replicates
                            ),
                            bootstrap_seed=(
                                envelope_protocol.bootstrap_seed
                                + 10000 * dlo_index
                                + 1000 * int(round(alpha * 100))
                                + 100 * int(grouping == "pooled")
                                + 10 * int(envelope_kind == "all_nonfallback_actions")
                                + budget_index
                            ),
                        )
                        budget_key = f"{budget:.6f}"
                        kind_policies[budget_key] = _as_json(summary)
                        policy_key = "/".join(
                            (grouping, alpha_key, envelope_kind, budget_key)
                        )
                        combined_actions_by_policy[policy_key].extend(selected_actions)

        for record_index, record in enumerate(records):
            per_decision.append(
                {
                    "stable_id": record.stable_id,
                    "dlo": record.dlo,
                    "trajectory": record.trajectory,
                    "current_frame": record.current_frame,
                    "registered_regret": record.registered_regret.tolist(),
                    "realized_regret": record.realized_regret.tolist(),
                    "physical_mse": record.physical_mse.tolist(),
                    "fallback_mse": record.fallback_mse,
                    "base_certificate_action": (record.base_certificate_action),
                    "target_order_within_dlo": record_index,
                }
            )
        result_by_dlo[dlo] = dlo_result

    combined_policies: dict[str, object] = {}
    for policy_index, (key, actions) in enumerate(
        sorted(combined_actions_by_policy.items())
    ):
        budget = float(key.rsplit("/", 1)[-1])
        combined_policies[key] = _as_json(
            _summarize_policy(
                combined_records,
                actions,
                regret_budget=budget,
                bootstrap_replicates=envelope_protocol.bootstrap_replicates,
                bootstrap_seed=envelope_protocol.bootstrap_seed + 50000 + policy_index,
            )
        )

    primary_alpha = f"{envelope_protocol.primary_miscoverage:.6f}"
    primary_budget = f"{envelope_protocol.primary_regret_budget:.6f}"
    primary_key = "/".join(
        (
            "per_dlo",
            primary_alpha,
            "base_certificate_selected_action",
            primary_budget,
        )
    )
    target = {
        "contract": CONTRACT,
        "schema_version": 1,
        "stage": "target-evaluation",
        "status": "complete",
        "run_key": request["run_key"],
        "request_sha256": sha256_file(request_path),
        "parent_protocol_sha256": sha256_file(parent_protocol_path),
        "envelope_protocol_sha256": sha256_file(envelope_protocol_path),
        "source_model_sha256": sha256_file(models_path),
        "source_envelope_sha256": sha256_file(source_path),
        "source_seal_sha256": sha256_file(seal_path),
        "target_tuning": False,
        "target_retries": False,
        "dlos": result_by_dlo,
        "combined_policies": combined_policies,
        "primary_policy_key": primary_key,
        "primary_policy": combined_policies[primary_key],
        "guarantee_semantics": (
            "Under exchangeability within each DLO, each per-DLO envelope has "
            "trajectory-marginal simultaneous coverage at its nominal level for "
            "the registered decisions/actions. The empirical target audit is a "
            "retrospective check, not a new theorem or unseen-object guarantee."
        ),
        "claim_boundary": envelope_protocol.claim_boundary,
        "module_claim_boundary": CONFORMAL_REGRET_ENVELOPE_CLAIM_BOUNDARY,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "target_result.json", target)
    with (output_root / "per_decision.jsonl").open("w", encoding="utf-8") as handle:
        for record in per_decision:
            handle.write(
                __import__("json").dumps(
                    record, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
    write_json(
        output_root / "provenance.json",
        {
            "contract": CONTRACT,
            "schema_version": 1,
            "run_key": request["run_key"],
            "dataset_repository": envelope_protocol.dataset_repository,
            "dataset_commit": envelope_protocol.dataset_commit,
            "source_only_calibration": True,
            "target_tuning": False,
            "target_retries": False,
            "calibration_unit": envelope_protocol.calibration_unit,
            "target_unit": "complete_trajectory",
        },
    )
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    source = subparsers.add_parser("source")
    source.add_argument("--parent-protocol", type=Path, required=True)
    source.add_argument("--envelope-protocol", type=Path, required=True)
    source.add_argument("--parent-source-result", type=Path, required=True)
    source.add_argument("--request", type=Path, required=True)
    source.add_argument("--dataset-root", type=Path, required=True)
    source.add_argument("--output-root", type=Path, required=True)
    target = subparsers.add_parser("target")
    target.add_argument("--parent-protocol", type=Path, required=True)
    target.add_argument("--envelope-protocol", type=Path, required=True)
    target.add_argument("--request", type=Path, required=True)
    target.add_argument("--dataset-root", type=Path, required=True)
    target.add_argument("--source-root", type=Path, required=True)
    target.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stage == "source":
        result = run_source(
            parent_protocol_path=args.parent_protocol,
            envelope_protocol_path=args.envelope_protocol,
            parent_source_result_path=args.parent_source_result,
            request_path=args.request,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
        )
    else:
        result = run_target(
            parent_protocol_path=args.parent_protocol,
            envelope_protocol_path=args.envelope_protocol,
            request_path=args.request,
            dataset_root=args.dataset_root,
            source_root=args.source_root,
            output_root=args.output_root,
        )
    print(
        __import__("json").dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
