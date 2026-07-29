"""Open-27 evaluator for the pairwise bias-aware development candidate."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_bias_aware_belief_development import (
    Deform360BiasAwareDevelopmentConfig,
    predict_bias_aware_candidate_arrays,
)
from .deform360_online_belief_evaluation import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    EXPECTED_SOURCE_EPISODES,
    PRIMARY_METRICS,
    _load_pickle,
    _physical_object_cluster_bootstrap,
    _resolve_prediction_archive,
    _sha256,
    _validate_deform360_outcome_manifest,
    score_deform360_hidden_trajectory,
)
from .deform360_pairwise_bias_aware_development import (
    PROTOCOL_ID,
    PairwiseBiasAwareDevelopmentConfig,
    predict_pairwise_bias_aware_candidate_arrays,
)
from .deform360_selective_virtual_sensing_prediction import (
    predict_persistence_pairwise_rbf_arrays,
)

SELECTED_BASELINE_ARM = "selected_raw_baseline"
PAIRWISE_RBF_ARM = "pairwise_consensus_rbf"
BIAS_AWARE_V4_ARM = "bias_aware_v4"
PAIRWISE_BIAS_AWARE_ARM = "pairwise_bias_aware_v1"
ARMS = (
    SELECTED_BASELINE_ARM,
    PAIRWISE_RBF_ARM,
    BIAS_AWARE_V4_ARM,
    PAIRWISE_BIAS_AWARE_ARM,
)


@dataclass
class _SourceCase:
    case: str
    object_id: str
    center_ids: np.ndarray
    target: np.ndarray
    visibility: np.ndarray
    validity: np.ndarray
    trajectories: dict[str, np.ndarray]
    reports: dict[str, Any]
    scores: dict[str, dict[str, float]]
    input_sha256: dict[str, str]


def _expected_case_names() -> tuple[str, ...]:
    return tuple(
        f"{object_id}-ep{episode_id:04d}"
        for object_id, episode_ids in EXPECTED_SOURCE_EPISODES.items()
        for episode_id in episode_ids
    )


def _load_target_pickle(path: Path) -> Any:
    try:
        return _load_pickle(path)
    except ModuleNotFoundError as error:
        if error.name != "numpy._core.numeric":
            raise
        import numpy.core as numpy_core
        import numpy.core.numeric as numpy_core_numeric

        sys.modules.setdefault("numpy._core", numpy_core)
        sys.modules.setdefault("numpy._core.numeric", numpy_core_numeric)
        return _load_pickle(path)


def _scored_frames(
    frame_count: int,
    update_frames: tuple[int, ...],
) -> tuple[int, ...]:
    result: list[int] = []
    for index, update in enumerate(update_frames):
        stop = (
            update_frames[index + 1]
            if index + 1 < len(update_frames)
            else frame_count
        )
        result.extend(range(update + 1, stop))
    return tuple(result)


def _source_reliability_and_variance(
    measurement_arrays: Mapping[str, np.ndarray],
    uncertainty_arrays: Mapping[str, np.ndarray],
    *,
    center_ids: np.ndarray,
    config: PairwiseBiasAwareDevelopmentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    selected_camera_count = len(measurement_arrays["selected_cameras"])
    if selected_camera_count < 2:
        raise ValueError("source measurement has fewer than two cameras")
    inlier_count = np.asarray(
        measurement_arrays["triangulation_inlier_view_count"],
        dtype=np.float64,
    )
    reprojection = np.asarray(
        measurement_arrays["triangulation_median_reprojection_px"],
        dtype=np.float64,
    )
    expected = (len(config.update_frames), len(center_ids))
    if inlier_count.shape != expected or reprojection.shape != expected:
        raise ValueError("source triangulation diagnostic shape changed")
    redundancy = np.clip(
        (inlier_count - 1.0) / (selected_camera_count - 1.0),
        0.0,
        1.0,
    )
    geometry = np.exp(
        -0.5 * np.square(reprojection / config.reprojection_scale_px)
    )
    reliability = redundancy * geometry
    reliability[~np.isfinite(reliability)] = 0.0

    covariance = np.asarray(
        uncertainty_arrays["measurement_covariance_m2"],
        dtype=np.float64,
    )
    covariance_valid = np.asarray(
        uncertainty_arrays["measurement_covariance_valid"],
        dtype=bool,
    )
    if covariance.shape[-2:] != (3, 3):
        raise ValueError("source covariance must end in (3, 3)")
    variance = np.empty(expected, dtype=np.float64)
    for update_index, update in enumerate(config.update_frames):
        selected = covariance[update, center_ids]
        isotropic = np.trace(selected, axis1=1, axis2=2) / 3.0
        valid_covariance = covariance_valid[update, center_ids] & np.isfinite(
            isotropic
        )
        reliability[update_index, ~valid_covariance] = 0.0
        variance[update_index] = np.where(
            valid_covariance,
            np.maximum(isotropic, config.observation_variance_floor_m2),
            config.observation_variance_floor_m2,
        )
    return reliability, variance


def _v4_config(
    config: PairwiseBiasAwareDevelopmentConfig,
) -> Deform360BiasAwareDevelopmentConfig:
    return Deform360BiasAwareDevelopmentConfig(
        update_frames=config.update_frames,
        minimum_available_center_count=config.pairwise_gate.minimum_inlier_count,
        minimum_motion_center_count=config.minimum_motion_center_count,
        physical_response_rank=config.physical_response_rank,
        minimum_physical_response_m=config.minimum_physical_response_m,
        minimum_observed_motion_m=config.minimum_observed_motion_m,
        minimum_physical_agreement_gain=config.minimum_physical_agreement_gain,
        minimum_identifiable_fraction=config.minimum_identifiable_fraction,
        observation_variance_floor_m2=config.observation_variance_floor_m2,
        reprojection_scale_px=config.reprojection_scale_px,
        state_update=config.state_update,
    )


def _load_source_case(
    source_case_dir: Path,
    measurement_case_dir: Path,
    uncertainty_case_dir: Path,
    selected_baseline_path: Path,
    config: PairwiseBiasAwareDevelopmentConfig,
) -> _SourceCase:
    seal_path = source_case_dir / "prediction_seal.json"
    outcome_path = source_case_dir / "outcome.json"
    target_path = source_case_dir / "target_data.pkl"
    measurement_path = measurement_case_dir / "measurement.npz"
    uncertainty_path = uncertainty_case_dir / "measurement_cycle_uncertainty.npz"
    required_paths = (
        seal_path,
        outcome_path,
        target_path,
        measurement_path,
        uncertainty_path,
        selected_baseline_path,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    _validate_deform360_outcome_manifest(seal_path, target_path, seal, outcome)
    prediction_path = _resolve_prediction_archive(source_case_dir, seal)
    with np.load(prediction_path, allow_pickle=False) as stored:
        required = {
            "driven_readout_m",
            "zero_action_readout_m",
            "action_support",
            "frame_zero_points_m",
        }
        if not required.issubset(stored.files):
            raise ValueError("sealed source archive lacks physical-response inputs")
        physical_response = (
            np.asarray(stored["driven_readout_m"], dtype=np.float64)
            - np.asarray(stored["zero_action_readout_m"], dtype=np.float64)
        )
        action_support = np.asarray(stored["action_support"], dtype=np.float64)
        frame_zero = np.asarray(stored["frame_zero_points_m"], dtype=np.float64)
    with np.load(selected_baseline_path, allow_pickle=False) as stored:
        if "selected_raw_backbone" not in stored.files:
            raise ValueError("selected baseline archive lacks selected_raw_backbone")
        baseline = np.asarray(stored["selected_raw_backbone"]).copy()
    with np.load(measurement_path, allow_pickle=False) as stored:
        measurement_arrays = {name: np.asarray(stored[name]) for name in stored.files}
    with np.load(uncertainty_path, allow_pickle=False) as stored:
        uncertainty_arrays = {name: np.asarray(stored[name]) for name in stored.files}
    update_frames = tuple(int(value) for value in measurement_arrays["update_frames"])
    if update_frames != config.update_frames:
        raise ValueError("source measurement update frames changed")
    center_ids = np.asarray(measurement_arrays["center_ids"], dtype=np.int64)
    reliability, variance = _source_reliability_and_variance(
        measurement_arrays,
        uncertainty_arrays,
        center_ids=center_ids,
        config=config,
    )
    measurement = np.asarray(measurement_arrays["measurement_m"])
    measurement_visibility = np.asarray(
        measurement_arrays["measurement_visibility"],
        dtype=bool,
    )
    measurement_validity = np.asarray(
        measurement_arrays["measurement_validity"],
        dtype=bool,
    )

    pairwise_report, pairwise_candidate = predict_persistence_pairwise_rbf_arrays(
        baseline,
        measurement,
        measurement_visibility,
        measurement_validity,
        center_ids=center_ids,
        update_frames=config.update_frames,
        gate_config=config.pairwise_gate,
    )
    v4_report, v4_candidate = predict_bias_aware_candidate_arrays(
        baseline,
        physical_response,
        frame_zero,
        action_support,
        measurement,
        measurement_visibility,
        measurement_validity,
        center_ids=center_ids,
        prior_reliability=reliability,
        observation_variance_m2=variance,
        config=_v4_config(config),
    )
    candidate_report, candidate = predict_pairwise_bias_aware_candidate_arrays(
        baseline,
        physical_response,
        frame_zero,
        action_support,
        measurement,
        measurement_visibility,
        measurement_validity,
        center_ids=center_ids,
        prior_reliability=reliability,
        observation_variance_m2=variance,
        config=config,
    )

    target_data = _load_target_pickle(target_path)
    target = np.asarray(target_data["object_points"])
    visibility = np.asarray(target_data["object_visibilities"], dtype=bool)
    validity = np.asarray(target_data["object_motions_valid"], dtype=bool)
    if target.shape != baseline.shape:
        raise ValueError("source target and selected baseline shapes differ")
    if not np.array_equal(target[0].astype(np.float32), frame_zero.astype(np.float32)):
        raise ValueError("source frame-zero target differs from sealed prediction")
    trajectories = {
        SELECTED_BASELINE_ARM: baseline,
        PAIRWISE_RBF_ARM: pairwise_candidate,
        BIAS_AWARE_V4_ARM: v4_candidate,
        PAIRWISE_BIAS_AWARE_ARM: candidate,
    }
    scored_frames = _scored_frames(len(target), config.update_frames)
    scores = {
        arm: score_deform360_hidden_trajectory(
            trajectory,
            target,
            visibility,
            validity,
            center_ids=center_ids,
            scored_frames=scored_frames,
        )
        for arm, trajectory in trajectories.items()
    }
    return _SourceCase(
        case=source_case_dir.name,
        object_id=str(seal["object_id"]),
        center_ids=center_ids,
        target=target,
        visibility=visibility,
        validity=validity,
        trajectories=trajectories,
        reports={
            PAIRWISE_RBF_ARM: pairwise_report,
            BIAS_AWARE_V4_ARM: v4_report,
            PAIRWISE_BIAS_AWARE_ARM: candidate_report,
        },
        scores=scores,
        input_sha256={
            "prediction_seal": _sha256(seal_path),
            "prediction_archive": _sha256(prediction_path),
            "source_outcome": _sha256(outcome_path),
            "source_target": _sha256(target_path),
            "measurement": _sha256(measurement_path),
            "uncertainty": _sha256(uncertainty_path),
            "selected_baseline": _sha256(selected_baseline_path),
        },
    )


def _aggregate(
    cases: list[_SourceCase],
    arm: str,
) -> dict[str, dict[str, float]]:
    object_ids = sorted({case.object_id for case in cases})
    result: dict[str, dict[str, float]] = {}
    for metric in PRIMARY_METRICS:
        episode_values = [float(case.scores[arm][metric]) for case in cases]
        object_values = [
            float(
                np.mean(
                    [
                        case.scores[arm][metric]
                        for case in cases
                        if case.object_id == object_id
                    ]
                )
            )
            for object_id in object_ids
        ]
        result[metric] = {
            "episode_mean_m": float(np.mean(episode_values)),
            "object_balanced_mean_m": float(np.mean(object_values)),
        }
    return result


def _comparison(
    cases: list[_SourceCase],
    aggregate: Mapping[str, Mapping[str, Mapping[str, float]]],
    candidate: str,
    comparator: str,
) -> dict[str, Any]:
    groups = {case.case: case.object_id for case in cases}
    result: dict[str, Any] = {}
    for metric in PRIMARY_METRICS:
        differences = {
            case.case: float(
                case.scores[candidate][metric] - case.scores[comparator][metric]
            )
            for case in cases
        }
        comparison = _physical_object_cluster_bootstrap(
            differences,
            groups,
            draws=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED,
        )
        comparator_mean = aggregate[comparator][metric]["object_balanced_mean_m"]
        comparison["object_balanced_relative_change"] = (
            comparison["object_balanced_mean_difference_m"] / comparator_mean
        )
        comparison["episode_win_count"] = int(
            sum(value < 0.0 for value in differences.values())
        )
        comparison["episode_tie_count"] = int(
            sum(value == 0.0 for value in differences.values())
        )
        result[metric] = comparison
    return result


def _object_level_gate_summary(
    cases: list[_SourceCase],
    candidate: str,
    comparator: str,
) -> dict[str, Any]:
    object_ids = sorted({case.object_id for case in cases})
    joint_wins = 0
    maximum_regression = {metric: float("-inf") for metric in PRIMARY_METRICS}
    for object_id in object_ids:
        changes = {}
        for metric in PRIMARY_METRICS:
            candidate_mean = float(
                np.mean(
                    [
                        case.scores[candidate][metric]
                        for case in cases
                        if case.object_id == object_id
                    ]
                )
            )
            comparator_mean = float(
                np.mean(
                    [
                        case.scores[comparator][metric]
                        for case in cases
                        if case.object_id == object_id
                    ]
                )
            )
            changes[metric] = (candidate_mean - comparator_mean) / comparator_mean
            maximum_regression[metric] = max(
                maximum_regression[metric],
                changes[metric],
            )
        if all(changes[metric] < 0.0 for metric in PRIMARY_METRICS):
            joint_wins += 1
    return {
        "object_count": len(object_ids),
        "joint_object_win_count": joint_wins,
        "maximum_object_relative_regression": maximum_regression,
    }


def evaluate_pairwise_bias_aware_source(
    source_root: str | Path,
    measurement_root: str | Path,
    uncertainty_root: str | Path,
    selected_baseline_root: str | Path,
    output_dir: str | Path,
    *,
    config: PairwiseBiasAwareDevelopmentConfig | None = None,
    transfer_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen composition on exactly the already-open 27 cases."""

    if transfer_manifest_sha256 is not None and (
        len(transfer_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in transfer_manifest_sha256
        )
    ):
        raise ValueError("invalid transfer-manifest SHA-256")
    cfg = config or PairwiseBiasAwareDevelopmentConfig()
    source = Path(source_root).resolve()
    measurement = Path(measurement_root).resolve()
    uncertainty = Path(uncertainty_root).resolve()
    selected_baseline = Path(selected_baseline_root).resolve()
    output = Path(output_dir).resolve()
    cases = [
        _load_source_case(
            source / case_name,
            measurement / case_name,
            uncertainty / case_name,
            selected_baseline / f"{case_name}.npz",
            cfg,
        )
        for case_name in _expected_case_names()
    ]
    aggregate = {arm: _aggregate(cases, arm) for arm in ARMS}
    comparisons = {
        f"{PAIRWISE_BIAS_AWARE_ARM}_vs_{comparator}": _comparison(
            cases,
            aggregate,
            PAIRWISE_BIAS_AWARE_ARM,
            comparator,
        )
        for comparator in (
            SELECTED_BASELINE_ARM,
            PAIRWISE_RBF_ARM,
            BIAS_AWARE_V4_ARM,
        )
    }
    primary_comparison = comparisons[
        f"{PAIRWISE_BIAS_AWARE_ARM}_vs_{PAIRWISE_RBF_ARM}"
    ]
    object_gates = _object_level_gate_summary(
        cases,
        PAIRWISE_BIAS_AWARE_ARM,
        PAIRWISE_RBF_ARM,
    )
    candidate_updates = [
        update
        for case in cases
        for update in case.reports[PAIRWISE_BIAS_AWARE_ARM]["updates"]
    ]
    gates = {
        "both_metrics_improve_at_least_one_percent": all(
            primary_comparison[metric]["object_balanced_relative_change"] <= -0.01
            for metric in PRIMARY_METRICS
        ),
        "both_cluster_intervals_exclude_zero": all(
            primary_comparison[metric]["object_cluster_upper_95_m"] < 0.0
            for metric in PRIMARY_METRICS
        ),
        "at_least_four_of_five_joint_object_wins": (
            object_gates["joint_object_win_count"] >= 4
        ),
        "no_object_regresses_more_than_two_percent": all(
            object_gates["maximum_object_relative_regression"][metric] <= 0.02
            for metric in PRIMARY_METRICS
        ),
        "all_rejections_are_exact_fallback": all(
            bool(update["candidate_available"])
            or bool(update["bit_exact_baseline_fallback"])
            for update in candidate_updates
        ),
        "all_acceptances_pass_target_free_gates": all(
            not bool(update["candidate_available"])
            or (
                bool(update["pairwise_gate"]["accepted"])
                and bool(update["dynamic_window_selected"])
                and len(update["selected_center_ids"])
                >= cfg.pairwise_gate.minimum_inlier_count
            )
            for update in candidate_updates
        ),
    }

    output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for case in cases:
        report_path = output / f"{case.case}.json"
        arrays_path = output / f"{case.case}.npz"
        report_path.write_text(
            json.dumps(
                {
                    "protocol_id": PROTOCOL_ID,
                    "case": case.case,
                    "object_id": case.object_id,
                    "scores": case.scores,
                    "target_free_reports": case.reports,
                    "input_sha256": case.input_sha256,
                    "claim_boundary": "already-open source development only",
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(arrays_path, **case.trajectories)
        artifacts.append(
            {
                "case": case.case,
                "report_sha256": _sha256(report_path),
                "arrays_sha256": _sha256(arrays_path),
            }
        )
    summary: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "status": "already-open-source-development",
        "config": asdict(cfg),
        "case_count": len(cases),
        "object_count": len({case.object_id for case in cases}),
        "aggregate": aggregate,
        "comparisons": comparisons,
        "object_level_gates": object_gates,
        "advancement_gates": gates,
        "larger_preregistered_run_justified": all(gates.values()),
        "candidate_update_count": int(
            sum(bool(update["candidate_available"]) for update in candidate_updates)
        ),
        "exact_fallback_count": int(
            sum(
                not bool(update["candidate_available"])
                and bool(update["bit_exact_baseline_fallback"])
                for update in candidate_updates
            )
        ),
        "artifacts": artifacts,
        "input_roots": {
            "source": str(source),
            "measurement": str(measurement),
            "uncertainty": str(uncertainty),
            "selected_baseline": str(selected_baseline),
        },
        "transfer_manifest_sha256": transfer_manifest_sha256,
        "claim_boundary": (
            "The 27 cases and five objects were already outcome-open. This result "
            "can stop or lock a candidate for a genuinely fresh protocol; it "
            "cannot establish state of the art, calibration, or non-regression."
        ),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "ARMS",
    "BIAS_AWARE_V4_ARM",
    "PAIRWISE_BIAS_AWARE_ARM",
    "PAIRWISE_RBF_ARM",
    "SELECTED_BASELINE_ARM",
    "evaluate_pairwise_bias_aware_source",
]
