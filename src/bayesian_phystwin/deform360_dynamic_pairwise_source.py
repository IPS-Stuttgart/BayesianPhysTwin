"""Opened-source evaluation for dynamic guarded Deform360 observations.

The measurement builder is causal and target-free. This module is the only
stage that opens the already-public Open27 outcomes. Every identity in the
64-point observation pool is permanently hidden from every scored arm, so a
larger provider cannot buy apparent accuracy by scoring its own observations.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_dynamic_pairwise_belief import (
    DYNAMIC_PAIRWISE_ARM,
    DynamicPairwiseBeliefConfig,
    predict_dynamic_pairwise_belief_arrays,
)
from .deform360_dynamic_pairwise_belief import (
    SELECTED_BACKBONE_ARM as DYNAMIC_SELECTED_ARM,
)
from .deform360_online_belief_evaluation import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    PRIMARY_METRICS,
    _physical_object_cluster_bootstrap,
    _resolve_prediction_archive,
    _sha256,
    score_deform360_hidden_trajectory,
)
from .deform360_pairwise_regret_guard import (
    DUAL_BACKBONE_ARM,
    predict_dual_backbone_pairwise_rbf_arrays,
)
from .deform360_pairwise_regret_guard import (
    SELECTED_BACKBONE_ARM as FIXED_SELECTED_ARM,
)
from .deform360_raw_camera_observation import (
    _load_measurement_artifact,
    _load_open_case_for_evaluation,
    expected_open_case_names,
)

PROTOCOL_ID = "deform360-dynamic-pairwise-belief-open27-v1-development"
FIXED16_SELECTED_ARM = "fixed16_selected_backbone"
FIXED16_PAIRWISE_ARM = "fixed16_pairwise_consensus_rbf"
ARMS = (
    FIXED16_SELECTED_ARM,
    FIXED16_PAIRWISE_ARM,
    DYNAMIC_SELECTED_ARM,
    DYNAMIC_PAIRWISE_ARM,
)


@dataclass
class _SourceCase:
    case: str
    object_id: str
    pool_ids: np.ndarray
    trajectories: dict[str, np.ndarray]
    reports: dict[str, Any]
    scores: dict[str, dict[str, object]]
    input_sha256: dict[str, str]


def _scored_frames(
    frame_count: int,
    update_frames: tuple[int, ...],
) -> tuple[int, ...]:
    frames: list[int] = []
    for index, update in enumerate(update_frames):
        stop = (
            update_frames[index + 1]
            if index + 1 < len(update_frames)
            else frame_count
        )
        frames.extend(range(update + 1, stop))
    return tuple(frames)


def _observation_model(
    arrays: Mapping[str, np.ndarray],
    *,
    config: DynamicPairwiseBeliefConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build residual-independent reliability and metric variance."""

    pool = np.asarray(arrays["center_ids"], dtype=np.int64)
    selected_camera_count = len(arrays["selected_cameras"])
    if len(pool) != config.observation_pool_count:
        raise ValueError("measurement pool count differs from the frozen protocol")
    if selected_camera_count < config.minimum_inlier_view_count:
        raise ValueError("camera panel cannot meet the independent-view gate")
    view_count = np.asarray(
        arrays["triangulation_inlier_view_count"],
        dtype=np.int64,
    )
    reprojection = np.asarray(
        arrays["triangulation_median_reprojection_px"],
        dtype=np.float64,
    )
    expected = (len(config.update_frames), len(pool))
    if view_count.shape != expected or reprojection.shape != expected:
        raise ValueError("triangulation diagnostic shape changed")

    redundancy = np.clip(
        (view_count.astype(np.float64) - 1.0) / (selected_camera_count - 1.0),
        0.0,
        1.0,
    )
    geometry = np.exp(-0.5 * np.square(reprojection / 3.0))
    reliability = redundancy * geometry
    reliability[~np.isfinite(reliability)] = 0.0
    variance = np.full(
        expected,
        config.observation_variance_floor_m2,
        dtype=np.float64,
    )
    return reliability, variance, view_count


def _late_metrics(score: Mapping[str, object]) -> dict[str, float]:
    by_frame = score["by_frame"]
    if not isinstance(by_frame, Mapping):
        raise ValueError("hidden score lacks frame diagnostics")
    identity = np.asarray(by_frame["hidden_identity_rmse_m"], dtype=np.float64)
    chamfer = np.asarray(by_frame["hidden_symmetric_chamfer_m"], dtype=np.float64)
    start = 2 * len(identity) // 3
    return {
        "late_hidden_identity_rmse_m": float(np.mean(identity[start:])),
        "late_hidden_symmetric_chamfer_m": float(np.mean(chamfer[start:])),
    }


def _load_source_case(
    source_case_dir: Path,
    measurement_case_dir: Path,
    config: DynamicPairwiseBeliefConfig,
) -> _SourceCase:
    seal_path = source_case_dir / "prediction_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    manifest, measurement_arrays = _load_measurement_artifact(
        source_case_dir,
        measurement_case_dir,
        seal,
    )
    manifest_boundary = manifest.get("information_boundary", {})
    if (
        manifest_boundary.get("target_data_read") is not False
        or manifest_boundary.get("outcome_manifest_read") is not False
    ):
        raise ValueError("measurement artifact crossed the target boundary")
    update_frames = tuple(int(value) for value in measurement_arrays["update_frames"])
    if update_frames != config.update_frames:
        raise ValueError("measurement update frames differ from the frozen protocol")
    pool_ids = np.asarray(measurement_arrays["center_ids"], dtype=np.int64)
    reliability, variance, view_count = _observation_model(
        measurement_arrays,
        config=config,
    )

    prediction_path = _resolve_prediction_archive(source_case_dir, seal)
    with np.load(prediction_path, allow_pickle=False) as stored:
        required = {
            "driven_readout_m",
            "zero_action_readout_m",
            "frame_zero_points_m",
            "action_support",
        }
        if not required.issubset(stored.files):
            raise ValueError("sealed source archive lacks physical-response inputs")
        response = np.asarray(stored["driven_readout_m"], dtype=np.float64) - np.asarray(
            stored["zero_action_readout_m"], dtype=np.float64
        )
        frame_zero = np.asarray(stored["frame_zero_points_m"], dtype=np.float64)
        action_support = np.asarray(stored["action_support"], dtype=np.float64)

    open_seal, physical, persistence, target, visibility, validity = (
        _load_open_case_for_evaluation(source_case_dir)
    )
    if open_seal != seal:
        raise ValueError("prediction seal changed while opening the source outcome")
    measurement = np.asarray(measurement_arrays["measurement_m"], dtype=np.float64)
    measurement_visibility = np.asarray(
        measurement_arrays["measurement_visibility"], dtype=bool
    )
    measurement_validity = np.asarray(
        measurement_arrays["measurement_validity"], dtype=bool
    )

    fixed_ids = pool_ids[: config.active_center_count]
    fixed_report, fixed_arrays = predict_dual_backbone_pairwise_rbf_arrays(
        physical,
        persistence,
        measurement,
        measurement_visibility,
        measurement_validity,
        center_ids=fixed_ids,
        update_frames=config.update_frames,
        gate_config=config.pairwise_gate,
        belief_config=config.belief,
    )
    dynamic_report, dynamic_arrays = predict_dynamic_pairwise_belief_arrays(
        physical,
        persistence,
        response,
        frame_zero,
        action_support,
        measurement,
        measurement_visibility,
        measurement_validity,
        pool_ids=pool_ids,
        prior_reliability=reliability,
        observation_variance_m2=variance,
        inlier_view_count=view_count,
        config=config,
    )
    trajectories = {
        FIXED16_SELECTED_ARM: fixed_arrays[FIXED_SELECTED_ARM],
        FIXED16_PAIRWISE_ARM: fixed_arrays[DUAL_BACKBONE_ARM],
        DYNAMIC_SELECTED_ARM: dynamic_arrays[DYNAMIC_SELECTED_ARM],
        DYNAMIC_PAIRWISE_ARM: dynamic_arrays[DYNAMIC_PAIRWISE_ARM],
    }
    frames = _scored_frames(len(target), config.update_frames)
    scores: dict[str, dict[str, object]] = {}
    for arm, trajectory in trajectories.items():
        score = score_deform360_hidden_trajectory(
            trajectory,
            target,
            visibility,
            validity,
            center_ids=pool_ids,
            scored_frames=frames,
        )
        score.update(_late_metrics(score))
        scores[arm] = score
    return _SourceCase(
        case=source_case_dir.name,
        object_id=str(seal["object_id"]),
        pool_ids=pool_ids,
        trajectories=trajectories,
        reports={
            FIXED16_PAIRWISE_ARM: fixed_report,
            DYNAMIC_PAIRWISE_ARM: dynamic_report,
        },
        scores=scores,
        input_sha256={
            "prediction_seal": _sha256(seal_path),
            "prediction_archive": _sha256(prediction_path),
            "measurement_manifest": _sha256(
                measurement_case_dir / "measurement_manifest.json"
            ),
            "measurement_archive": _sha256(
                measurement_case_dir / "measurement.npz"
            ),
            "source_outcome": _sha256(source_case_dir / "outcome.json"),
            "source_target": _sha256(source_case_dir / "target_data.pkl"),
        },
    )


def _aggregate(
    cases: list[_SourceCase],
    arm: str,
) -> dict[str, dict[str, float]]:
    object_ids = sorted({case.object_id for case in cases})
    metrics = (*PRIMARY_METRICS, "late_hidden_identity_rmse_m")
    result: dict[str, dict[str, float]] = {}
    for metric in metrics:
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
    for metric in (*PRIMARY_METRICS, "late_hidden_identity_rmse_m"):
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


def _object_gate_summary(
    cases: list[_SourceCase],
    candidate: str,
    comparator: str,
) -> dict[str, Any]:
    object_ids = sorted({case.object_id for case in cases})
    joint_wins = 0
    maximum_regression = {metric: float("-inf") for metric in PRIMARY_METRICS}
    for object_id in object_ids:
        changes: dict[str, float] = {}
        object_cases = [case for case in cases if case.object_id == object_id]
        for metric in PRIMARY_METRICS:
            candidate_mean = float(
                np.mean([case.scores[candidate][metric] for case in object_cases])
            )
            comparator_mean = float(
                np.mean([case.scores[comparator][metric] for case in object_cases])
            )
            changes[metric] = (candidate_mean - comparator_mean) / comparator_mean
            maximum_regression[metric] = max(
                maximum_regression[metric], changes[metric]
            )
        joint_wins += int(all(changes[metric] < 0.0 for metric in PRIMARY_METRICS))
    return {
        "object_count": len(object_ids),
        "joint_object_win_count": joint_wins,
        "maximum_object_relative_regression": maximum_regression,
    }


def evaluate_dynamic_pairwise_source(
    source_root: str | Path,
    measurement_root: str | Path,
    output_dir: str | Path,
    *,
    config: DynamicPairwiseBeliefConfig | None = None,
    transfer_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen candidate on exactly the already-open Open27 panel."""

    if transfer_manifest_sha256 is not None and (
        len(transfer_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in transfer_manifest_sha256)
    ):
        raise ValueError("invalid transfer-manifest SHA-256")
    cfg = config or DynamicPairwiseBeliefConfig()
    source = Path(source_root).resolve()
    measurement = Path(measurement_root).resolve()
    output = Path(output_dir).resolve()
    cases = [
        _load_source_case(source / case, measurement / case, cfg)
        for case in expected_open_case_names()
    ]
    aggregate = {arm: _aggregate(cases, arm) for arm in ARMS}
    comparisons = {
        f"{DYNAMIC_PAIRWISE_ARM}_vs_{comparator}": _comparison(
            cases,
            aggregate,
            DYNAMIC_PAIRWISE_ARM,
            comparator,
        )
        for comparator in (
            DYNAMIC_SELECTED_ARM,
            FIXED16_PAIRWISE_ARM,
        )
    }
    primary = comparisons[f"{DYNAMIC_PAIRWISE_ARM}_vs_{FIXED16_PAIRWISE_ARM}"]
    object_gates = _object_gate_summary(
        cases,
        DYNAMIC_PAIRWISE_ARM,
        FIXED16_PAIRWISE_ARM,
    )
    updates = [
        update
        for case in cases
        for update in case.reports[DYNAMIC_PAIRWISE_ARM]["updates"]
    ]
    gates = {
        "both_primary_metrics_improve_at_least_one_percent": all(
            primary[metric]["object_balanced_relative_change"] <= -0.01
            for metric in PRIMARY_METRICS
        ),
        "both_primary_cluster_intervals_exclude_zero": all(
            primary[metric]["object_cluster_upper_95_m"] < 0.0
            for metric in PRIMARY_METRICS
        ),
        "late_identity_improves": (
            primary["late_hidden_identity_rmse_m"][
                "object_balanced_relative_change"
            ]
            < 0.0
        ),
        "at_least_four_of_five_joint_object_wins": (
            object_gates["joint_object_win_count"] >= 4
        ),
        "no_object_regresses_more_than_two_percent": all(
            object_gates["maximum_object_relative_regression"][metric] <= 0.02
            for metric in PRIMARY_METRICS
        ),
        "all_rejections_are_exact_fallback": all(
            bool(update["accepted"])
            or bool(update["bit_exact_selected_backbone_fallback"])
            for update in updates
        ),
        "all_acceptances_pass_physical_and_multiview_guards": all(
            not bool(update["accepted"])
            or (
                int(update["motion_center_count"]) >= cfg.minimum_motion_center_count
                and float(update["causal_physical_agreement_gain"])
                >= cfg.minimum_physical_agreement_gain
                and float(update["correction_physical_cosine"])
                >= cfg.minimum_correction_physical_cosine
                and float(update["correction_to_physical_motion_ratio"])
                <= cfg.maximum_correction_to_physical_motion_ratio
            )
            for update in updates
        ),
        "all_64_observed_identities_excluded_from_every_score": all(
            int(case.scores[arm]["permanently_excluded_center_count"])
            == cfg.observation_pool_count
            for case in cases
            for arm in ARMS
        ),
    }

    output.mkdir(parents=True, exist_ok=False)
    artifacts: list[dict[str, str]] = []
    for case in cases:
        report_path = output / f"{case.case}.json"
        arrays_path = output / f"{case.case}.npz"
        report_path.write_text(
            json.dumps(
                {
                    "protocol_id": PROTOCOL_ID,
                    "case": case.case,
                    "object_id": case.object_id,
                    "observation_pool_ids": case.pool_ids.tolist(),
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
        "accepted_update_count": int(sum(bool(update["accepted"]) for update in updates)),
        "exact_fallback_count": int(
            sum(
                not bool(update["accepted"])
                and bool(update["bit_exact_selected_backbone_fallback"])
                for update in updates
            )
        ),
        "artifacts": artifacts,
        "input_roots": {
            "source": str(source),
            "measurement": str(measurement),
        },
        "transfer_manifest_sha256": transfer_manifest_sha256,
        "claim_boundary": (
            "These 27 cases and five objects were already outcome-open. The result "
            "may stop or lock a candidate for a genuinely fresh protocol; it cannot "
            "establish state of the art, calibration, or non-regression."
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
    "FIXED16_PAIRWISE_ARM",
    "FIXED16_SELECTED_ARM",
    "PROTOCOL_ID",
    "evaluate_dynamic_pairwise_source",
]
