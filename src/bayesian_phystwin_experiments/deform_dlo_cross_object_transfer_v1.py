"""Outcome-blind transfer of DLO3 residual coefficients to DLO4 and DLO5.

The evaluator asks whether local-residual coefficients fitted only on DLO3
improve the fresh DLO4 and DLO5 physical predictions without any object-side
coefficient refit or outcome-dependent seed selection. It is a pre-score
secondary diagnostic: its complete arm, gate, and aggregation are frozen before
the protected DLO4/DLO5 target scores are opened, but after their target
prediction stage may have started.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from bayesian_phystwin_experiments.deform_dlo_cross_backend_transfer_v1 import (
    paired_point_summary,
)
from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    build_deform_local_residual_features,
)

SCHEMA_VERSION = 1
CONTRACT = "deform-dlo3-to-dlo45-no-refit-coefficient-transfer-v1"
RESULT_CONTRACT = "deform-dlo3-to-dlo45-no-refit-transfer-result-v1"
DLOS = ("DLO4", "DLO5")
SEEDS = (42, 43, 44)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _finite_array(value: object, *, ndim: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite {ndim}-D array")
    return array


def _identity(value: object, *, label: str) -> dict[str, object]:
    identity = _mapping(value, label=label)
    path = str(identity.get("path", identity.get("repository_path", "")))
    digest = str(identity.get("sha256", ""))
    size = int(cast(Any, identity.get("size_bytes", -1)))
    if not path or len(digest) != 64 or size <= 0:
        raise ValueError(f"{label} identity is invalid")
    return {"path": path, "sha256": digest, "size_bytes": size}


def load_cross_object_transfer_protocol(path: str | Path) -> dict[str, object]:
    """Load and strictly validate the frozen DLO3-to-DLO4/DLO5 protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("cross-object protocol must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported cross-object protocol schema")
    if payload.get("contract") != CONTRACT:
        raise ValueError("unsupported cross-object protocol contract")
    if payload.get("status") != "frozen-before-parent-target-scoring":
        raise ValueError("cross-object protocol is not frozen pre-score")

    registration = _mapping(
        payload.get("registration_boundary"),
        label="registration boundary",
    )
    if (
        int(cast(Any, registration.get("blocking_workflow_run_id", -1)))
        != 33361441865
        or registration.get("blocking_workflow_name")
        != "DEFORM DLO4/DLO5 staged timeout recovery v3"
        or registration.get("blocking_head_sha")
        != "0376ece871d7c3d9355788f812a3c4cc1c9165b0"
        or registration.get("blocking_head_branch")
        != "science/deform-dlo45-time-budget-recovery-v3"
        or registration.get("observed_parent_stage")
        != "target-prediction-in-progress"
        or registration.get("source_outcomes_previously_opened") is not True
        or registration.get("target_scores_opened") is not False
        or registration.get("target_predictions_used_for_design") is not False
        or registration.get("classification")
        != "outcome-blind-pre-score-secondary-diagnostic"
    ):
        raise ValueError("cross-object registration boundary changed")

    parent = _mapping(payload.get("parent_dlo45"), label="parent DLO45")
    if (
        parent.get("run_root")
        != (
            "/home/github-runner/.cache/workflows/"
            "deform-dlo45-time-budget-recovery-v3/runs/33361441865-1"
        )
        or parent.get("require_workflow_success") is not True
        or parent.get("result_contract")
        != "deform-dlo45-frozen-transfer-result-v1"
        or parent.get("joint_seal_contract")
        != "deform-dlo45-joint-prediction-seal-v1"
        or parent.get("prediction_seal_contract")
        != "deform-dlo45-target-prediction-seal-v1"
    ):
        raise ValueError("cross-object parent binding changed")
    _identity(parent.get("protocol"), label="parent DLO45 protocol")

    models = payload.get("dlo3_local_residual_models")
    if not isinstance(models, Sequence) or isinstance(models, (str, bytes)):
        raise ValueError("DLO3 residual models must be a sequence")
    model_records = [
        _mapping(value, label="DLO3 residual model") for value in models
    ]
    if [int(cast(Any, value.get("seed", -1))) for value in model_records] != list(
        SEEDS
    ):
        raise ValueError("DLO3 residual model seed set changed")
    for value in model_records:
        _identity(value, label="DLO3 residual model")

    data = _mapping(payload.get("data"), label="data")
    if (
        tuple(str(value) for value in cast(Sequence[object], data.get("dlos", ())))
        != DLOS
        or int(cast(Any, data.get("trajectory_count_per_dlo", -1))) != 14
        or int(cast(Any, data.get("frame_count", -1))) != 500
        or int(cast(Any, data.get("node_count", -1))) != 12
        or int(cast(Any, data.get("prediction_horizon", -1))) != 498
        or data.get("statistical_unit") != "complete-trajectory"
        or data.get("physical_backbone")
        != "matching-object-alltrain-update-6400"
    ):
        raise ValueError("cross-object data contract changed")

    evaluation = _mapping(payload.get("evaluation"), label="evaluation")
    if (
        evaluation.get("primary_arm")
        != "equal-seed-dlo3-residual-no-refit-to-dlo45"
        or evaluation.get("seed_aggregation") != "arithmetic-prediction-mean"
        or tuple(
            int(value)
            for value in cast(Sequence[object], evaluation.get("individual_seeds", ()))
        )
        != SEEDS
        or float(cast(Any, evaluation.get("shrinkage", math.nan))) != 0.25
        or evaluation.get("metric") != "mean-coordinate-l1-m-all-nodes"
        or evaluation.get("dlo_aggregation") != "equal-dlo-mean"
        or int(cast(Any, evaluation.get("bootstrap_repetitions", -1))) != 10000
        or int(cast(Any, evaluation.get("bootstrap_seed", -1))) != 20260902
        or evaluation.get("feature_support_diagnostic_affects_gate") is not False
    ):
        raise ValueError("cross-object evaluation contract changed")

    gate = _mapping(payload.get("promotion_gate"), label="promotion gate")
    if (
        float(cast(Any, gate.get("minimum_relative_improvement", math.nan)))
        != 0.01
        or int(cast(Any, gate.get("minimum_case_wins", -1))) != 8
        or float(cast(Any, gate.get("maximum_case_ratio", math.nan))) != 1.10
        or int(cast(Any, gate.get("minimum_improving_seed_models", -1))) != 2
        or gate.get("require_each_dlo") is not True
    ):
        raise ValueError("cross-object promotion gate changed")

    boundary = _mapping(payload.get("information_boundary"), label="boundary")
    if (
        boundary.get("dlo3_residual_refit") is not False
        or boundary.get("dlo4_or_dlo5_residual_refit") is not False
        or boundary.get("seed_or_weight_selection_from_target") is not False
        or boundary.get("shrinkage_selection_from_target") is not False
        or boundary.get("gate_selection_from_target") is not False
        or boundary.get("object_specific_physical_backbone_allowed") is not True
        or boundary.get("dlo3_official_evaluation_read") is not False
        or boundary.get("target_retry_authorized") is not False
        or boundary.get("paper_claim_authorized") is not False
    ):
        raise ValueError("cross-object information boundary changed")

    result = dict(payload)
    result["protocol_path"] = str(source)
    return result


def feature_support_summary(
    model: Mapping[str, object],
    initial_states: np.ndarray,
    clamped_action: np.ndarray,
    baseline_predictions: np.ndarray,
) -> dict[str, object]:
    """Summarize DLO3-standardized causal feature support on another DLO."""

    features, _ = build_deform_local_residual_features(
        initial_states,
        clamped_action,
        baseline_predictions,
    )
    location = _finite_array(
        model.get("feature_location"),
        ndim=2,
        label="feature location",
    )
    scale = _finite_array(
        model.get("feature_scale"),
        ndim=2,
        label="feature scale",
    )
    if location.shape != features.shape[2:] or scale.shape != location.shape:
        raise ValueError("feature support model arrays do not align")
    if np.any(scale <= 0.0):
        raise ValueError("feature support scale must be positive")
    absolute_z = np.abs(
        (features - location[None, None, :, :]) / scale[None, None, :, :]
    )
    quantile_levels = (0.50, 0.90, 0.95, 0.99, 0.999)
    quantiles = np.quantile(absolute_z, quantile_levels)
    return {
        "schema_version": 1,
        "contract": "deform-dlo-cross-object-feature-support-v1",
        "sample_count": int(absolute_z.size),
        "absolute_z_quantiles": {
            f"q{int(level * 1000):03d}": float(value)
            for level, value in zip(quantile_levels, quantiles, strict=True)
        },
        "fraction_absolute_z_gt_3": float(np.mean(absolute_z > 3.0)),
        "fraction_absolute_z_gt_5": float(np.mean(absolute_z > 5.0)),
        "fraction_absolute_z_gt_10": float(np.mean(absolute_z > 10.0)),
        "maximum_absolute_z": float(np.max(absolute_z)),
        "affects_promotion_gate": False,
    }


def _gate(
    summary: Mapping[str, object],
    contract: Mapping[str, object],
) -> dict[str, object]:
    minimum = float(cast(Any, contract["minimum_relative_improvement"]))
    minimum_wins = int(cast(Any, contract["minimum_case_wins"]))
    maximum_ratio = float(cast(Any, contract["maximum_case_ratio"]))
    passed = (
        float(cast(Any, summary["relative_improvement"])) >= minimum
        and int(cast(Any, summary["wins"])) >= minimum_wins
        and float(cast(Any, summary["maximum_case_ratio"])) <= maximum_ratio
    )
    return {
        "passed": passed,
        "minimum_relative_improvement": minimum,
        "minimum_case_wins": minimum_wins,
        "maximum_case_ratio": maximum_ratio,
    }


def _equal_dlo_summary(
    results: Mapping[str, Mapping[str, object]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    primary = {
        dlo: _mapping(
            results[dlo]["primary_vs_matching_physical"],
            label=f"{dlo} primary summary",
        )
        for dlo in DLOS
    }
    candidate = float(
        np.mean(
            [
                float(cast(Any, primary[dlo]["candidate_mean_l1_m"]))
                for dlo in DLOS
            ]
        )
    )
    baseline = float(
        np.mean(
            [
                float(cast(Any, primary[dlo]["baseline_mean_l1_m"]))
                for dlo in DLOS
            ]
        )
    )
    rng = np.random.default_rng(seed)
    dlo_draws = []
    for dlo in DLOS:
        cases = cast(Sequence[Mapping[str, object]], primary[dlo]["cases"])
        differences = np.asarray(
            [float(cast(Any, case["difference_m"])) for case in cases],
            dtype=np.float64,
        )
        indices = rng.integers(
            0,
            len(differences),
            size=(repetitions, len(differences)),
        )
        dlo_draws.append(differences[indices].mean(axis=1))
    pooled_draws = np.mean(np.stack(dlo_draws), axis=0)
    return {
        "aggregation": "equal-dlo-mean",
        "candidate_mean_l1_m": candidate,
        "baseline_mean_l1_m": baseline,
        "mean_difference_m": float(candidate - baseline),
        "relative_improvement": float(1.0 - candidate / baseline),
        "stratified_trajectory_bootstrap_95_interval_m": [
            float(value) for value in np.quantile(pooled_draws, (0.025, 0.975))
        ],
        "wins": sum(int(cast(Any, primary[dlo]["wins"])) for dlo in DLOS),
        "ties": sum(int(cast(Any, primary[dlo]["ties"])) for dlo in DLOS),
        "losses": sum(int(cast(Any, primary[dlo]["losses"])) for dlo in DLOS),
        "maximum_case_ratio": max(
            float(cast(Any, primary[dlo]["maximum_case_ratio"])) for dlo in DLOS
        ),
        "case_count": sum(int(cast(Any, primary[dlo]["case_count"])) for dlo in DLOS),
    }


def evaluate_cross_object_transfer(
    *,
    names_by_dlo: Mapping[str, Sequence[str]],
    truth_by_dlo: Mapping[str, np.ndarray],
    physical_by_dlo: Mapping[str, np.ndarray],
    object_specific_by_dlo: Mapping[str, np.ndarray],
    transferred_by_dlo: Mapping[str, Mapping[int, np.ndarray]],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate unchanged DLO3 residual coefficients on DLO4 and DLO5."""

    data = _mapping(protocol.get("data"), label="data")
    expected_count = int(cast(Any, data["trajectory_count_per_dlo"]))
    expected_horizon = int(cast(Any, data["prediction_horizon"]))
    expected_nodes = int(cast(Any, data["node_count"]))
    evaluation = _mapping(protocol.get("evaluation"), label="evaluation")
    repetitions = int(cast(Any, evaluation["bootstrap_repetitions"]))
    bootstrap_seed = int(cast(Any, evaluation["bootstrap_seed"]))
    gate_contract = _mapping(protocol.get("promotion_gate"), label="promotion gate")
    minimum_improving = int(cast(Any, gate_contract["minimum_improving_seed_models"]))

    if (
        set(names_by_dlo) != set(DLOS)
        or set(truth_by_dlo) != set(DLOS)
        or set(physical_by_dlo) != set(DLOS)
        or set(object_specific_by_dlo) != set(DLOS)
        or set(transferred_by_dlo) != set(DLOS)
    ):
        raise ValueError("cross-object DLO set differs")

    dlo_results: dict[str, dict[str, object]] = {}
    for dlo_index, dlo in enumerate(DLOS):
        names = tuple(str(name) for name in names_by_dlo[dlo])
        if (
            len(names) != expected_count
            or len(set(names)) != expected_count
            or any(not name for name in names)
        ):
            raise ValueError(f"{dlo} target roster differs")
        truth = _finite_array(truth_by_dlo[dlo], ndim=4, label=f"{dlo} truth")
        physical = _finite_array(
            physical_by_dlo[dlo],
            ndim=4,
            label=f"{dlo} physical",
        )
        object_specific = _finite_array(
            object_specific_by_dlo[dlo],
            ndim=4,
            label=f"{dlo} object-specific candidate",
        )
        seed_map = transferred_by_dlo[dlo]
        if tuple(sorted(seed_map)) != SEEDS:
            raise ValueError(f"{dlo} transferred seed set changed")
        seed_predictions = {
            seed: _finite_array(
                seed_map[seed],
                ndim=4,
                label=f"{dlo} seed-{seed} transfer",
            )
            for seed in SEEDS
        }
        expected_shape = (expected_count, expected_horizon, expected_nodes, 3)
        shapes = {
            truth.shape,
            physical.shape,
            object_specific.shape,
            *(value.shape for value in seed_predictions.values()),
        }
        if shapes != {expected_shape}:
            raise ValueError(f"{dlo} cross-object prediction shapes differ")

        ensemble = np.mean(
            np.stack([seed_predictions[seed] for seed in SEEDS]),
            axis=0,
        )
        primary = paired_point_summary(
            ensemble,
            physical,
            truth,
            names,
            repetitions=repetitions,
            seed=bootstrap_seed + 100 * dlo_index,
        )
        object_specific_summary = paired_point_summary(
            object_specific,
            physical,
            truth,
            names,
            repetitions=repetitions,
            seed=bootstrap_seed + 100 * dlo_index + 1,
        )
        direct_vs_specific = paired_point_summary(
            ensemble,
            object_specific,
            truth,
            names,
            repetitions=repetitions,
            seed=bootstrap_seed + 100 * dlo_index + 2,
        )
        seed_summaries = {
            str(seed): paired_point_summary(
                seed_predictions[seed],
                physical,
                truth,
                names,
                repetitions=repetitions,
                seed=bootstrap_seed + 100 * dlo_index + seed,
            )
            for seed in SEEDS
        }
        primary_gate = _gate(primary, gate_contract)
        improving_seeds = sum(
            float(cast(Any, summary["relative_improvement"])) > 0.0
            for summary in seed_summaries.values()
        )
        seed_stability_passed = improving_seeds >= minimum_improving
        supported = bool(primary_gate["passed"]) and seed_stability_passed

        baseline_mean = float(cast(Any, primary["baseline_mean_l1_m"]))
        transfer_mean = float(cast(Any, primary["candidate_mean_l1_m"]))
        specific_mean = float(
            cast(Any, object_specific_summary["candidate_mean_l1_m"])
        )
        specific_gain = baseline_mean - specific_mean
        retained = (
            (baseline_mean - transfer_mean) / specific_gain
            if specific_gain > 0.0
            else None
        )
        dlo_results[dlo] = {
            "decision": (
                "dlo3-residual-no-refit-transfer-supported"
                if supported
                else "dlo3-residual-no-refit-transfer-not-supported"
            ),
            "methods": {
                "matching_object_physical": baseline_mean,
                "matching_object_fitted_residual": specific_mean,
                "dlo3_equal_seed_no_refit_residual": transfer_mean,
                **{
                    f"dlo3_seed_{seed}_no_refit_residual": float(
                        cast(Any, seed_summaries[str(seed)]["candidate_mean_l1_m"])
                    )
                    for seed in SEEDS
                },
            },
            "primary_vs_matching_physical": primary,
            "matching_object_residual_vs_physical": object_specific_summary,
            "dlo3_transfer_vs_matching_object_residual": direct_vs_specific,
            "individual_seed_vs_matching_physical": seed_summaries,
            "promotion_gate": {
                **primary_gate,
                "improving_seed_models": improving_seeds,
                "minimum_improving_seed_models": minimum_improving,
                "seed_stability_passed": seed_stability_passed,
                "supported": supported,
            },
            "matching_object_gain_retained_fraction": (
                None if retained is None else float(retained)
            ),
        }

    both_supported = all(
        bool(
            _mapping(
                dlo_results[dlo]["promotion_gate"],
                label=f"{dlo} promotion gate",
            )["supported"]
        )
        for dlo in DLOS
    )
    equal_dlo = _equal_dlo_summary(
        dlo_results,
        repetitions=repetitions,
        seed=bootstrap_seed + 999,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": RESULT_CONTRACT,
        "decision": (
            "dlo3-residual-coefficients-transfer-to-both-fresh-dlos"
            if both_supported
            else "dlo3-residual-coefficients-do-not-transfer-to-both-fresh-dlos"
        ),
        "primary_arm": "equal-seed-dlo3-residual-no-refit-to-dlo45",
        "source_model_dlo": "DLO3",
        "target_dlos": list(DLOS),
        "source_model_seeds": list(SEEDS),
        "results": dlo_results,
        "equal_dlo_summary": equal_dlo,
        "both_dlos_supported": both_supported,
        "information_boundary": {
            "classification": "outcome-blind-pre-score-secondary-diagnostic",
            "frozen_before_parent_target_scoring": True,
            "parent_target_prediction_may_have_started": True,
            "dlo3_residual_refit": False,
            "dlo4_or_dlo5_residual_refit": False,
            "seed_or_weight_selection_from_target": False,
            "shrinkage_selection_from_target": False,
            "gate_selection_from_target": False,
            "object_specific_physical_backbone_allowed": True,
            "dlo3_official_evaluation_read": False,
            "target_retry_authorized": False,
            "paper_claim_authorized": False,
        },
        "claim_boundary": (
            "A positive decision supports unchanged DLO3 local-residual "
            "coefficient transfer to the exact released DLO4 and DLO5 target "
            "operators on top of separately fitted matching-object physical "
            "backbones. It does not establish transfer of the complete twin, "
            "arbitrary-object generalization, physical-parameter identification, "
            "deployment safety, or universal state of the art."
        ),
    }
