#!/usr/bin/env python3
"""Read-only causal residual-velocity diagnostic for two open cohorts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_online_belief_evaluation import (
    _validate_deform360_outcome_manifest,
    score_deform360_hidden_trajectory,
)
from bayesian_phystwin.phystwin_online_belief_evaluation import _score_trajectory


BPT = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1")
DEFORM_ROOT = Path(
    "/mnt/corsair/florianpfaff/deform360-dense-reusable-panel-v1/independent-source-v1"
)
DEFORM_RUN = BPT / "runs/deform360-online-belief-open27-v1"
PHYSTWIN_RUN = BPT / "runs/online-belief-original22-observation-gated-v3"
OUTPUT = Path("/tmp/velocity-belief-diagnostic-v1.json")

BETAS = (0.25, 0.5, 1.0)
LOCAL_BLEND = 0.25
LENGTH_SCALE_FRACTION = 0.1
MAXIMUM_CORRECTION_M = 0.1


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _object_scale(points: np.ndarray) -> float:
    finite = np.all(np.isfinite(points), axis=1)
    lower = np.quantile(points[finite], 0.05, axis=0)
    upper = np.quantile(points[finite], 0.95, axis=0)
    return max(float(np.linalg.norm(upper - lower)), 1.0e-4)


def _cap_norm(vectors: np.ndarray, maximum: float) -> np.ndarray:
    values = np.asarray(vectors, dtype=float).copy()
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    values *= np.minimum(1.0, maximum / np.maximum(norm, 1.0e-15))
    return values


def _residual_velocity_field(
    prior: np.ndarray,
    target: np.ndarray,
    visible: np.ndarray,
    valid: np.ndarray,
    centers: np.ndarray,
    previous: int,
    update: int,
    *,
    scale_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Median global velocity plus local Gaussian-RBF residual velocity."""

    supported = (
        visible[previous, centers]
        & valid[previous, centers]
        & visible[update, centers]
        & valid[update, centers]
        & np.all(np.isfinite(target[previous, centers]), axis=1)
        & np.all(np.isfinite(target[update, centers]), axis=1)
        & np.all(np.isfinite(prior[previous, centers]), axis=1)
        & np.all(np.isfinite(prior[update, centers]), axis=1)
    )
    selected = centers[supported]
    query = np.asarray(prior[update], dtype=float)
    if not len(selected):
        return np.zeros_like(query), {
            "matched_center_count": 0,
            "global_velocity_m_per_frame": [0.0, 0.0, 0.0],
            "maximum_velocity_norm_m_per_frame": 0.0,
        }
    current_residual = target[update, selected] - prior[update, selected]
    prior_residual = target[previous, selected] - prior[previous, selected]
    velocity = (current_residual - prior_residual) / float(update - previous)
    global_velocity = np.median(velocity, axis=0)
    local_velocity = velocity - global_velocity
    centre_positions = prior[update, selected]
    length_scale = max(scale_m * LENGTH_SCALE_FRACTION, 1.0e-4)
    distance = np.linalg.norm(query[:, None, :] - centre_positions[None, :, :], axis=2)
    weight = np.exp(-0.5 * np.square(distance / length_scale))
    weight_sum = np.sum(weight, axis=1, keepdims=True)
    normalized = weight / np.maximum(weight_sum, 1.0e-15)
    normalized[weight_sum[:, 0] < 1.0e-12] = 0.0
    field = global_velocity[None, :] + LOCAL_BLEND * (normalized @ local_velocity)
    return field, {
        "matched_center_count": int(len(selected)),
        "matched_center_ids": selected.tolist(),
        "global_velocity_m_per_frame": global_velocity.tolist(),
        "median_velocity_norm_m_per_frame": float(
            np.median(np.linalg.norm(velocity, axis=1))
        ),
        "maximum_velocity_norm_m_per_frame": float(
            np.max(np.linalg.norm(velocity, axis=1))
        ),
        "decoded_median_velocity_norm_m_per_frame": float(
            np.median(np.linalg.norm(field, axis=1))
        ),
        "decoded_maximum_velocity_norm_m_per_frame": float(
            np.max(np.linalg.norm(field, axis=1))
        ),
    }


def _build_arms(
    prior: np.ndarray,
    target: np.ndarray,
    visible: np.ndarray,
    valid: np.ndarray,
    centers: np.ndarray,
    updates: tuple[int, ...],
    accepted: tuple[bool, ...],
    beta0: np.ndarray,
    dispersion_thresholds: tuple[float, ...],
    *,
    scale_points: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Build fixed controls and velocity arms while preserving exact fallback."""

    arms: dict[str, np.ndarray] = {
        "physical_prior": prior.copy(),
        "beta0_field": beta0.copy(),
        "frozen_current_state": prior.copy(),
        "constant_baseline_velocity": prior.copy(),
    }
    for beta in BETAS:
        label = str(beta).replace(".", "p")
        arms[f"beta{label}"] = prior.copy()
        arms[f"beta{label}_history_clip"] = prior.copy()

    records: list[dict[str, Any]] = []
    scale = _object_scale(scale_points)
    for index, update in enumerate(updates):
        stop = updates[index + 1] if index + 1 < len(updates) else len(prior)
        previous = 0 if index == 0 else updates[index - 1]
        if not accepted[index]:
            records.append(
                {
                    "frame": update,
                    "previous_measurement_frame": previous,
                    "accepted": False,
                    "decision": "locked_gate_exact_physical_prior",
                }
            )
            continue

        # The saved beta-0 trajectory is the locked recursive position belief.
        # Its decoded mean is invariant over lead time, so recover it from the
        # first post-update frame without rerunning or changing the filter.
        position_correction = np.asarray(beta0[update + 1], dtype=float) - np.asarray(
            prior[update + 1], dtype=float
        )
        velocity_field, velocity_record = _residual_velocity_field(
            prior,
            target,
            visible,
            valid,
            centers,
            previous,
            update,
            scale_m=scale,
        )
        frozen_state = np.asarray(prior[update], dtype=float) + position_correction
        baseline_velocity = np.asarray(prior[update], dtype=float) - np.asarray(
            prior[update - 1], dtype=float
        )
        maximum_lead = max(1, stop - update - 1)
        dispersion_threshold = float(dispersion_thresholds[index])
        clipped_velocity = _cap_norm(
            velocity_field,
            dispersion_threshold / float(maximum_lead),
        )

        for frame in range(update + 1, stop):
            lead = frame - update
            arms["frozen_current_state"][frame] = frozen_state
            arms["constant_baseline_velocity"][frame] = (
                frozen_state + lead * baseline_velocity
            )
            for beta in BETAS:
                label = str(beta).replace(".", "p")
                correction = _cap_norm(
                    position_correction + beta * lead * velocity_field,
                    MAXIMUM_CORRECTION_M,
                )
                clipped_correction = _cap_norm(
                    position_correction + beta * lead * clipped_velocity,
                    MAXIMUM_CORRECTION_M,
                )
                arms[f"beta{label}"][frame] = prior[frame] + correction
                arms[f"beta{label}_history_clip"][frame] = (
                    prior[frame] + clipped_correction
                )
        records.append(
            {
                "frame": update,
                "previous_measurement_frame": previous,
                "interval_end_exclusive": stop,
                "accepted": True,
                "dispersion_threshold_m": dispersion_threshold,
                "history_clip_velocity_norm_m_per_frame": (
                    dispersion_threshold / float(maximum_lead)
                ),
                "position_correction_median_norm_m": float(
                    np.median(np.linalg.norm(position_correction, axis=1))
                ),
                "position_correction_maximum_norm_m": float(
                    np.max(np.linalg.norm(position_correction, axis=1))
                ),
                **velocity_record,
            }
        )
    return arms, records


def _aggregate(rows: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    arm_names = tuple(next(iter(rows.values())))
    metric_names = tuple(next(iter(next(iter(rows.values())).values())))
    aggregate: dict[str, Any] = {"arms": {}, "comparisons": {}}
    for arm in arm_names:
        aggregate["arms"][arm] = {
            metric: float(np.mean([row[arm][metric] for row in rows.values()]))
            for metric in metric_names
        }
    for reference in ("physical_prior", "beta0_field", "frozen_current_state"):
        comparisons: dict[str, Any] = {}
        for arm in arm_names:
            if arm == reference:
                continue
            comparisons[arm] = {}
            for metric in metric_names:
                arm_values = np.asarray([row[arm][metric] for row in rows.values()])
                ref_values = np.asarray(
                    [row[reference][metric] for row in rows.values()]
                )
                comparisons[arm][metric] = {
                    "mean_difference_m": float(np.mean(arm_values - ref_values)),
                    "relative_change_fraction": float(
                        (np.mean(arm_values) - np.mean(ref_values))
                        / np.mean(ref_values)
                    ),
                    "case_wins": int(np.sum(arm_values < ref_values)),
                    "case_ties": int(np.sum(arm_values == ref_values)),
                    "case_count": int(len(arm_values)),
                    "maximum_case_relative_regression": float(
                        np.max((arm_values - ref_values) / ref_values)
                    ),
                }
        aggregate["comparisons"][f"vs_{reference}"] = comparisons
    return aggregate


def _run_deform() -> dict[str, Any]:
    rows: dict[str, dict[str, dict[str, float]]] = {}
    details: dict[str, Any] = {}
    for report_path in sorted(DEFORM_RUN.glob("*.json")):
        if report_path.name == "summary.json":
            continue
        report = json.loads(report_path.read_text())
        arrays_path = report_path.with_suffix(".npz")
        with np.load(arrays_path) as stored:
            prior = stored["physical_prior_m"].copy()
            beta0 = stored["recursive_rbf_risk_limited_m"].copy()
            centers = stored["center_ids"].copy()
        episode = DEFORM_ROOT / report_path.stem
        seal_path = episode / "prediction_seal.json"
        target_path = episode / "target_data.pkl"
        outcome_path = episode / "outcome.json"
        seal = json.loads(seal_path.read_text())
        outcome = json.loads(outcome_path.read_text())
        _validate_deform360_outcome_manifest(
            seal_path,
            target_path,
            seal,
            outcome,
        )
        data = _load_pickle(target_path)
        target = np.asarray(data["object_points"], dtype=float)
        visible = np.asarray(data["object_visibilities"], dtype=bool)
        valid = np.asarray(data["object_motions_valid"], dtype=bool)
        updates = tuple(map(int, report["update_frames"]))
        accepted = tuple(bool(value["accepted"]) for value in report["updates"])
        thresholds = tuple(
            float(value["maximum_median_radial_residual_m"])
            for value in report["updates"]
        )
        arms, records = _build_arms(
            prior,
            target,
            visible,
            valid,
            centers,
            updates,
            accepted,
            beta0,
            thresholds,
            scale_points=target[0],
        )
        scored = tuple(map(int, report["scored_frames"]))
        case_scores = {}
        full_scores = {}
        for name, trajectory in arms.items():
            score = score_deform360_hidden_trajectory(
                trajectory,
                target,
                visible,
                valid,
                center_ids=centers,
                scored_frames=scored,
            )
            full_scores[name] = score
            case_scores[name] = {
                "hidden_identity_rmse_m": score["post_update_hidden_identity_rmse_m"],
                "hidden_symmetric_chamfer_m": score[
                    "post_update_hidden_symmetric_chamfer_m"
                ],
            }
        rows[report_path.stem] = case_scores
        details[report_path.stem] = {
            "object_id": report["object_id"],
            "episode_id": report["episode_id"],
            "center_ids": centers.tolist(),
            "updates": records,
            "scores": case_scores,
        }
    return {
        "case_count": len(rows),
        "aggregate": _aggregate(rows),
        "cases": details,
    }


def _run_phystwin() -> dict[str, Any]:
    rows: dict[str, dict[str, dict[str, float]]] = {}
    details: dict[str, Any] = {}
    for report_path in sorted(PHYSTWIN_RUN.glob("*.json")):
        if report_path.name == "summary.json":
            continue
        report = json.loads(report_path.read_text())
        arrays_path = report_path.with_suffix(".npz")
        with np.load(arrays_path) as stored:
            beta0 = stored["field_trajectory_m"].copy()
            centers = stored["center_ids"].copy()
        input_paths = report["inputs"]
        data = _load_pickle(Path(input_paths["final_data"]["path"]))
        prior = np.asarray(
            _load_pickle(Path(input_paths["baseline"]["path"])), dtype=float
        )
        target = np.asarray(data["object_points"], dtype=float)
        visible = np.asarray(data["object_visibilities"], dtype=bool)
        valid = np.asarray(data["object_motions_valid"], dtype=bool)
        test_end = int(report["split"]["test_end_frame"])
        prior = prior[:test_end]
        target = target[:test_end]
        visible = visible[:test_end]
        valid = valid[:test_end]
        beta0 = beta0[:test_end]
        updates = tuple(map(int, report["update_frames"]))
        accepted = tuple(bool(value["accepted"]) for value in report["updates"])
        thresholds = tuple(
            float(value["maximum_residual_dispersion_m"]) for value in report["updates"]
        )
        arms, records = _build_arms(
            prior,
            target,
            visible,
            valid,
            centers,
            updates,
            accepted,
            beta0,
            thresholds,
            scale_points=target[0],
        )
        manual_path = input_paths.get("manual_tracks")
        manual = (
            None
            if manual_path is None
            else np.asarray(_load_pickle(Path(manual_path["path"])), dtype=float)[
                :test_end
            ]
        )
        surface_count = target.shape[1] + len(np.asarray(data["surface_points"]))
        scored = tuple(map(int, report["scores"]["open_loop"]["scored_frames"]))
        case_scores = {}
        for name, trajectory in arms.items():
            score = _score_trajectory(
                trajectory,
                target,
                visible,
                valid,
                manual,
                surface_point_count=surface_count,
                center_ids=centers,
                scored_frames=scored,
            )
            case_scores[name] = {
                "hidden_identity_error_m": score["future_noncenter_point_error_m"],
                "one_sided_l1_chamfer_m": score["future_chamfer_distance_m"],
            }
        rows[report_path.stem] = case_scores
        details[report_path.stem] = {
            "center_ids": centers.tolist(),
            "updates": records,
            "scores": case_scores,
        }
    return {
        "case_count": len(rows),
        "aggregate": _aggregate(rows),
        "cases": details,
    }


def _cross_cohort_verdict(result: dict[str, Any]) -> dict[str, Any]:
    verdict: dict[str, Any] = {}
    candidate_names = [
        f"beta{str(beta).replace('.', 'p')}{suffix}"
        for beta in BETAS
        for suffix in ("", "_history_clip")
    ]
    for arm in candidate_names:
        cohorts = {}
        for cohort_name in ("phystwin22", "deform360_open27"):
            aggregate = result[cohort_name]["aggregate"]
            metrics = tuple(aggregate["arms"][arm])
            cohorts[cohort_name] = {
                "beats_beta0_each_metric": all(
                    aggregate["comparisons"]["vs_beta0_field"][arm][metric][
                        "mean_difference_m"
                    ]
                    < 0.0
                    for metric in metrics
                ),
                "beats_frozen_each_metric": all(
                    aggregate["comparisons"]["vs_frozen_current_state"][arm][metric][
                        "mean_difference_m"
                    ]
                    < 0.0
                    for metric in metrics
                ),
                "metrics": {
                    metric: {
                        "vs_beta0_relative_change_fraction": aggregate["comparisons"][
                            "vs_beta0_field"
                        ][arm][metric]["relative_change_fraction"],
                        "vs_frozen_relative_change_fraction": aggregate["comparisons"][
                            "vs_frozen_current_state"
                        ][arm][metric]["relative_change_fraction"],
                    }
                    for metric in metrics
                },
            }
        verdict[arm] = {
            "cohorts": cohorts,
            "improves_beta0_and_frozen_on_every_metric_both_cohorts": all(
                value["beats_beta0_each_metric"] and value["beats_frozen_each_metric"]
                for value in cohorts.values()
            ),
        }
    return verdict


def _cluster_bootstrap_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    """Paired intervals with physical-object resampling, matching the locked runs."""

    phystwin_config = json.loads(
        (
            BPT / "configs/sota/phystwin_online_belief_v3_original22_development.json"
        ).read_text()
    )
    groups_by_cohort = {
        "phystwin22": phystwin_config["confirmation_cohort"]["physical_object_groups"],
        "deform360_open27": {
            case: detail["object_id"]
            for case, detail in result["deform360_open27"]["cases"].items()
        },
    }
    output: dict[str, Any] = {}
    for cohort_name, groups in groups_by_cohort.items():
        rows = result[cohort_name]["cases"]
        group_names = tuple(sorted(set(groups.values())))
        cases_by_group = {
            group: tuple(case for case, value in groups.items() if value == group)
            for group in group_names
        }
        cohort: dict[str, Any] = {}
        for arm in ("beta0p25", "beta0p5", "beta1p0"):
            cohort[arm] = {}
            for reference in ("beta0_field", "frozen_current_state"):
                cohort[arm][f"vs_{reference}"] = {}
                for metric in rows[next(iter(rows))]["scores"][arm]:
                    differences = {
                        case: rows[case]["scores"][arm][metric]
                        - rows[case]["scores"][reference][metric]
                        for case in rows
                    }
                    rng = np.random.default_rng(20260719)
                    samples = np.empty(10_000, dtype=float)
                    for draw in range(len(samples)):
                        selected_groups = rng.choice(
                            group_names, size=len(group_names), replace=True
                        )
                        selected_cases = [
                            case
                            for group in selected_groups
                            for case in cases_by_group[str(group)]
                        ]
                        samples[draw] = np.mean(
                            [differences[case] for case in selected_cases]
                        )
                    cohort[arm][f"vs_{reference}"][metric] = {
                        "mean_difference_m": float(np.mean(list(differences.values()))),
                        "lower_95_m": float(np.quantile(samples, 0.025)),
                        "upper_95_m": float(np.quantile(samples, 0.975)),
                        "probability_improved": float(np.mean(samples < 0.0)),
                        "draws": int(len(samples)),
                        "seed": 20260719,
                        "cluster_count": int(len(group_names)),
                    }
        output[cohort_name] = cohort
    return output


def run_diagnostic(
    *,
    bpt_root: str | Path = BPT,
    deform_root: str | Path = DEFORM_ROOT,
    deform_run: str | Path = DEFORM_RUN,
    phystwin_run: str | Path = PHYSTWIN_RUN,
) -> dict[str, Any]:
    """Reproduce the read-only residual-velocity development diagnostic.

    Only sealed-input locations are configurable.  Candidate arms, causal
    information boundaries, scoring, bootstrap seeds, and output schema remain
    identical to the legacy diagnostic.
    """

    global BPT, DEFORM_ROOT, DEFORM_RUN, PHYSTWIN_RUN
    previous = (BPT, DEFORM_ROOT, DEFORM_RUN, PHYSTWIN_RUN)
    BPT = Path(bpt_root)
    DEFORM_ROOT = Path(deform_root)
    DEFORM_RUN = Path(deform_run)
    PHYSTWIN_RUN = Path(phystwin_run)
    try:
        result: dict[str, Any] = {
            "schema_version": 1,
            "status": "read-only post-hoc exploratory diagnostic",
            "information_boundary": {
                "phystwin22": (
                    "uses only locked v3 center IDs, accepted-update decisions, current "
                    "and previous scheduled sparse residual measurements; every selected "
                    "center is permanently excluded from the identity metric"
                ),
                "deform360_open27": (
                    "uses only already-open 27 source episodes, the physical prediction "
                    "sealed before source-future opening, deterministic frame-zero FPS "
                    "centers, the locked [0,19) dispersion gate, and measurements at "
                    "frames 0/19/38/57; every center is permanently excluded"
                ),
                "velocity": (
                    "residual change since frame zero for the first update and since the "
                    "previous scheduled update thereafter; no future observation, metric, "
                    "per-case beta, or outcome-dependent routing enters prediction"
                ),
                "history_clip": (
                    "fixed causal cap: maximum endpoint residual-velocity displacement "
                    "equals the already-frozen residual-dispersion threshold"
                ),
                "maximum_total_correction_m": MAXIMUM_CORRECTION_M,
            },
            "phystwin22": _run_phystwin(),
            "deform360_open27": _run_deform(),
        }
        result["cross_cohort_verdict"] = _cross_cohort_verdict(result)
        result["physical_object_cluster_bootstrap"] = _cluster_bootstrap_diagnostics(
            result
        )
        return result
    finally:
        BPT, DEFORM_ROOT, DEFORM_RUN, PHYSTWIN_RUN = previous


def main() -> None:
    result = run_diagnostic()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    compact = {
        "output": str(OUTPUT),
        "phystwin22": result["phystwin22"]["aggregate"],
        "deform360_open27": result["deform360_open27"]["aggregate"],
        "verdict": result["cross_cohort_verdict"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
