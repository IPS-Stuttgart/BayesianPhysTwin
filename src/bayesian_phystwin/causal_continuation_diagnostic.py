#!/usr/bin/env python3
"""Read-only causal continuation-vs-freeze diagnostic on saved belief runs."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from bayesian_phystwin.deform360_online_belief_evaluation import (
    _validate_deform360_outcome_manifest,
)


PHYS_RUN = Path(
    "/mnt/corsair/florianpfaff/bpt-online-belief-v1/runs/"
    "online-belief-original22-observation-gated-v3"
)
PHYS_CONFIG = Path(
    "/mnt/corsair/florianpfaff/bpt-online-belief-v1/configs/sota/"
    "phystwin_online_belief_v3_original22_development.json"
)
DEFORM_RUN = Path(
    "/mnt/corsair/florianpfaff/bpt-online-belief-v1/runs/"
    "deform360-online-belief-open27-v1"
)
OUTPUT = Path("/tmp/causal-physics-gain-v1/results.json")

FIXED_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
CAUSAL_METHODS = (
    "causal_last_huber",
    "causal_cumulative_huber",
    "causal_last_median",
    "causal_cumulative_median",
    "causal_last_huber_binary025",
    "causal_last_huber_binary050",
    "causal_last_median_binary025",
    "causal_last_median_binary050",
)
METRICS = ("hidden_identity_m", "hidden_chamfer_m")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cutoff = 0.5 * float(np.sum(weights))
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def _usable_projection_rows(prior_delta: np.ndarray, observed_delta: np.ndarray):
    p = np.asarray(prior_delta, dtype=float)
    y = np.asarray(observed_delta, dtype=float)
    finite = np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(y), axis=1)
    p = p[finite]
    y = y[finite]
    if len(p) < 3:
        return p[:0], y[:0]
    motion = np.linalg.norm(p, axis=1)
    positive = motion[motion > 1e-6]
    if len(positive) < 3:
        return p[:0], y[:0]
    # Avoid numerically explosive per-point ratios while keeping at least the
    # upper three quarters of the observed prior-motion support.
    threshold = max(1e-5, float(np.quantile(positive, 0.25)))
    keep = motion >= threshold
    return p[keep], y[keep]


def huber_projection(intervals: list[tuple[np.ndarray, np.ndarray]]) -> float:
    rows = [_usable_projection_rows(p, y) for p, y in intervals]
    rows = [(p, y) for p, y in rows if len(p)]
    if not rows:
        return 0.5
    p = np.concatenate([row[0] for row in rows], axis=0)
    y = np.concatenate([row[1] for row in rows], axis=0)
    denominator = float(np.sum(p * p))
    if denominator <= 1e-12:
        return 0.5
    alpha = float(np.sum(p * y) / denominator)
    for _ in range(20):
        radial = np.linalg.norm(y - alpha * p, axis=1)
        center = float(np.median(radial))
        mad = float(np.median(np.abs(radial - center)))
        scale = max(1e-4, 1.4826 * mad)
        cutoff = 1.345 * scale
        weights = np.minimum(1.0, cutoff / np.maximum(radial, 1e-12))
        new_denominator = float(np.sum(weights[:, None] * p * p))
        if new_denominator <= 1e-12:
            break
        updated = float(np.sum(weights[:, None] * p * y) / new_denominator)
        if abs(updated - alpha) <= 1e-8:
            alpha = updated
            break
        alpha = updated
    return float(np.clip(alpha, 0.0, 1.0))


def median_projection(intervals: list[tuple[np.ndarray, np.ndarray]]) -> float:
    values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for prior_delta, observed_delta in intervals:
        p, y = _usable_projection_rows(prior_delta, observed_delta)
        if not len(p):
            continue
        energy = np.sum(p * p, axis=1)
        values.append(np.sum(p * y, axis=1) / energy)
        # Linear rather than squared motion weighting limits domination by a
        # single fast centre while retaining information from meaningful moves.
        weights.append(np.sqrt(energy))
    if not values:
        return 0.5
    estimate = weighted_median(np.concatenate(values), np.concatenate(weights))
    return float(np.clip(estimate, 0.0, 1.0))


def estimate_alpha(
    method: str,
    intervals: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    selected = intervals[-1:] if "_last_" in method else intervals
    estimate = (
        huber_projection(selected)
        if "_huber" in method
        else median_projection(selected)
    )
    if method.endswith("_binary025"):
        return float(estimate > 0.25)
    if method.endswith("_binary050"):
        return float(estimate > 0.50)
    return estimate


def observation_interval(
    prior: np.ndarray,
    target: np.ndarray,
    visible: np.ndarray,
    valid: np.ndarray,
    centers: np.ndarray,
    previous: int,
    current: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    supported = (
        visible[previous, centers]
        & valid[previous, centers]
        & visible[current, centers]
        & valid[current, centers]
        & np.all(np.isfinite(target[previous, centers]), axis=1)
        & np.all(np.isfinite(target[current, centers]), axis=1)
        & np.all(np.isfinite(prior[previous, centers]), axis=1)
        & np.all(np.isfinite(prior[current, centers]), axis=1)
    )
    if int(np.sum(supported)) < 3:
        return None
    ids = centers[supported]
    return (
        prior[current, ids] - prior[previous, ids],
        target[current, ids] - target[previous, ids],
    )


def make_arms(
    prior: np.ndarray,
    beta0: np.ndarray,
    target: np.ndarray,
    visible: np.ndarray,
    valid: np.ndarray,
    centers: np.ndarray,
    updates: list[dict],
    test_end: int,
    first_anchor: int,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    arms = {
        "physical": prior.copy(),
        "beta0_alpha1": beta0.copy(),
    }
    for alpha in FIXED_ALPHAS:
        arms[f"fixed_alpha_{alpha:.2f}"] = prior.copy()
    for method in CAUSAL_METHODS:
        arms[method] = prior.copy()

    trusted_intervals: list[tuple[np.ndarray, np.ndarray]] = []
    last_trusted_frame = first_anchor
    alpha_records: list[dict] = []
    for index, record in enumerate(updates):
        update = int(record["frame"])
        stop = (
            int(updates[index + 1]["frame"]) if index + 1 < len(updates) else test_end
        )
        if not bool(record["accepted"]):
            alpha_records.append(
                {
                    "frame": update,
                    "accepted": False,
                    "interval_end_exclusive": stop,
                    "decision": "existing_gate_rejection_exact_physical_prior",
                }
            )
            continue

        pair = observation_interval(
            prior,
            target,
            visible,
            valid,
            centers,
            last_trusted_frame,
            update,
        )
        if pair is not None:
            trusted_intervals.append(pair)
        causal_alpha = {
            method: estimate_alpha(method, trusted_intervals)
            for method in CAUSAL_METHODS
        }
        alpha_records.append(
            {
                "frame": update,
                "accepted": True,
                "interval_end_exclusive": stop,
                "previous_trusted_observation_frame": last_trusted_frame,
                "projection_center_count": 0 if pair is None else int(len(pair[0])),
                "trusted_projection_interval_count": len(trusted_intervals),
                "causal_alpha": causal_alpha,
            }
        )

        for frame in range(update + 1, stop):
            prior_continuation = prior[frame] - prior[update]
            correction = beta0[frame] - prior[frame]
            corrected_current = prior[update] + correction
            for alpha in FIXED_ALPHAS:
                arms[f"fixed_alpha_{alpha:.2f}"][frame] = (
                    corrected_current + alpha * prior_continuation
                )
            for method, alpha in causal_alpha.items():
                arms[method][frame] = corrected_current + alpha * prior_continuation
        last_trusted_frame = update

    # These identities document the intended end points of the hybrid family.
    for record in updates:
        update = int(record["frame"])
        stop = int(record.get("interval_end_exclusive", test_end))
        if not bool(record["accepted"]):
            for name in arms:
                if name == "beta0_alpha1":
                    continue
                if not np.array_equal(
                    arms[name][update + 1 : stop], prior[update + 1 : stop]
                ):
                    raise AssertionError(f"{name} changed a rejected interval")
    arms["frozen_alpha0"] = arms.pop("fixed_alpha_0.00")
    # Floating-point reconstruction may differ at the ulp level, so retain the
    # saved beta-0 array as the authoritative alpha=1 endpoint.
    arms.pop("fixed_alpha_1.00")
    return arms, alpha_records


def score_phys(
    trajectory: np.ndarray,
    target: np.ndarray,
    visible: np.ndarray,
    valid: np.ndarray,
    centers: np.ndarray,
    scored: list[int],
    surface_count: int,
) -> dict[str, float]:
    hidden = np.ones(target.shape[1], dtype=bool)
    hidden[centers] = False
    prediction_indices = np.arange(surface_count)
    prediction_indices = prediction_indices[~np.isin(prediction_indices, centers)]
    identity: list[float] = []
    chamfer: list[float] = []
    for frame in scored:
        identity_mask = (
            hidden
            & visible[frame]
            & valid[frame]
            & np.all(np.isfinite(target[frame]), axis=1)
        )
        if np.any(identity_mask):
            residual = (
                trajectory[frame, : target.shape[1]][identity_mask]
                - target[frame, identity_mask]
            )
            identity.append(float(np.mean(np.linalg.norm(residual, axis=1))))
        chamfer_mask = (
            hidden & visible[frame] & np.all(np.isfinite(target[frame]), axis=1)
        )
        predicted_surface = trajectory[frame, prediction_indices]
        predicted_surface = predicted_surface[
            np.all(np.isfinite(predicted_surface), axis=1)
        ]
        observed_hidden = target[frame, chamfer_mask]
        chamfer.append(
            float(np.mean(cKDTree(predicted_surface).query(observed_hidden, p=1)[0]))
        )
    return {
        "hidden_identity_m": float(np.mean(identity)),
        "hidden_chamfer_m": float(np.mean(chamfer)),
    }


def score_deform(
    trajectory: np.ndarray,
    target: np.ndarray,
    visible: np.ndarray,
    valid: np.ndarray,
    centers: np.ndarray,
    scored: list[int],
) -> dict[str, float]:
    hidden = np.ones(target.shape[1], dtype=bool)
    hidden[centers] = False
    identity: list[float] = []
    chamfer: list[float] = []
    for frame in scored:
        mask = (
            hidden
            & visible[frame]
            & valid[frame]
            & np.all(np.isfinite(target[frame]), axis=1)
            & np.all(np.isfinite(trajectory[frame]), axis=1)
        )
        predicted = trajectory[frame, mask]
        observed = target[frame, mask]
        residual = predicted - observed
        identity.append(float(np.sqrt(np.mean(np.square(residual)))))
        forward = cKDTree(predicted).query(observed, p=2)[0]
        backward = cKDTree(observed).query(predicted, p=2)[0]
        chamfer.append(0.5 * float(np.mean(forward) + np.mean(backward)))
    return {
        "hidden_identity_m": float(np.mean(identity)),
        "hidden_chamfer_m": float(np.mean(chamfer)),
    }


def cluster_comparison(
    candidate: str,
    comparator: str,
    scores: dict[str, dict[str, dict[str, float]]],
    groups: dict[str, str],
    seed: int,
) -> dict:
    result: dict[str, dict] = {}
    cases = sorted(scores)
    group_names = sorted(set(groups.values()))
    rng = np.random.default_rng(seed)
    for metric in METRICS:
        differences = {
            case: scores[case][candidate][metric] - scores[case][comparator][metric]
            for case in cases
        }
        group_means = np.array(
            [
                np.mean([differences[case] for case in cases if groups[case] == group])
                for group in group_names
            ],
            dtype=float,
        )
        draws = np.mean(
            group_means[
                rng.integers(0, len(group_means), size=(10_000, len(group_means)))
            ],
            axis=1,
        )
        candidate_mean = float(
            np.mean([scores[case][candidate][metric] for case in cases])
        )
        comparator_mean = float(
            np.mean([scores[case][comparator][metric] for case in cases])
        )
        relative = candidate_mean / comparator_mean - 1.0
        regressions = {
            case: scores[case][candidate][metric] / scores[case][comparator][metric]
            - 1.0
            for case in cases
        }
        worst_case = max(regressions, key=regressions.get)
        result[metric] = {
            "candidate_case_mean_m": candidate_mean,
            "comparator_case_mean_m": comparator_mean,
            "case_mean_difference_m": float(np.mean(list(differences.values()))),
            "relative_change": float(relative),
            "case_wins": int(sum(value < 0.0 for value in differences.values())),
            "joint_case_count": len(cases),
            "object_balanced_difference_m": float(np.mean(group_means)),
            "object_cluster_lower_95_m": float(np.quantile(draws, 0.025)),
            "object_cluster_upper_95_m": float(np.quantile(draws, 0.975)),
            "object_cluster_probability_improved": float(np.mean(draws < 0.0)),
            "worst_relative_regression": float(regressions[worst_case]),
            "worst_regression_case": worst_case,
        }
    result["joint_two_metric_case_wins"] = int(
        sum(
            all(
                scores[case][candidate][metric] < scores[case][comparator][metric]
                for metric in METRICS
            )
            for case in cases
        )
    )
    return result


def aggregate_scores(scores: dict[str, dict[str, dict[str, float]]]) -> dict:
    cases = sorted(scores)
    arms = sorted(next(iter(scores.values())))
    return {
        arm: {
            metric: float(np.mean([scores[case][arm][metric] for case in cases]))
            for metric in METRICS
        }
        for arm in arms
    }


def evaluate_phys() -> dict:
    config = json.loads(PHYS_CONFIG.read_text())
    groups = dict(config["confirmation_cohort"]["physical_object_groups"])
    scores: dict[str, dict[str, dict[str, float]]] = {}
    alpha_records: dict[str, list[dict]] = {}
    inputs: list[dict] = []
    for case in sorted(groups):
        report_path = PHYS_RUN / f"{case}.json"
        arrays_path = PHYS_RUN / f"{case}.npz"
        report = json.loads(report_path.read_text())
        arrays = np.load(arrays_path)
        data_path = Path(report["inputs"]["final_data"]["path"])
        prior_path = Path(report["inputs"]["baseline"]["path"])
        data = load_pickle(data_path)
        prior = np.asarray(load_pickle(prior_path), dtype=float)
        target = np.asarray(data["object_points"], dtype=float)
        visible = np.asarray(data["object_visibilities"], dtype=bool)
        valid = np.asarray(data["object_motions_valid"], dtype=bool)
        test_end = int(report["split"]["test_end_frame"])
        prior = prior[:test_end]
        target = target[:test_end]
        visible = visible[:test_end]
        valid = valid[:test_end]
        beta0 = np.asarray(arrays["field_trajectory_m"], dtype=float)[:test_end]
        centers = np.asarray(arrays["center_ids"], dtype=int)
        updates = [dict(value) for value in report["updates"]]
        for index, update in enumerate(updates):
            update["interval_end_exclusive"] = (
                int(updates[index + 1]["frame"])
                if index + 1 < len(updates)
                else test_end
            )
        first_anchor = int(report["split"]["train_end_frame"]) - 1
        arms, records = make_arms(
            prior,
            beta0,
            target,
            visible,
            valid,
            centers,
            updates,
            test_end,
            first_anchor,
        )
        scored = [
            frame
            for index, update in enumerate(updates)
            for frame in range(
                int(update["frame"]) + 1,
                int(updates[index + 1]["frame"])
                if index + 1 < len(updates)
                else test_end,
            )
        ]
        surface_count = target.shape[1] + len(np.asarray(data["surface_points"]))
        scores[case] = {
            arm: score_phys(
                trajectory,
                target,
                visible,
                valid,
                centers,
                scored,
                surface_count,
            )
            for arm, trajectory in arms.items()
        }
        alpha_records[case] = records
        inputs.append(
            {
                "case": case,
                "report_sha256": sha256(report_path),
                "arrays_sha256": sha256(arrays_path),
                "target_sha256": sha256(data_path),
                "prior_sha256": sha256(prior_path),
            }
        )
    aggregate = aggregate_scores(scores)
    candidates = [
        name
        for name in aggregate
        if name not in {"physical", "beta0_alpha1", "frozen_alpha0"}
    ]
    comparisons = {
        candidate: {
            comparator: cluster_comparison(
                candidate, comparator, scores, groups, 20260719 + index
            )
            for index, comparator in enumerate(
                ("beta0_alpha1", "frozen_alpha0", "physical")
            )
        }
        for candidate in candidates
    }
    return {
        "dataset": "released PhysTwin 22 development cohort",
        "groups": groups,
        "case_count": len(scores),
        "aggregate": aggregate,
        "comparisons": comparisons,
        "scores_by_case": scores,
        "alpha_records": alpha_records,
        "inputs": inputs,
    }


def evaluate_deform() -> dict:
    summary = json.loads((DEFORM_RUN / "summary.json").read_text())
    groups = {
        f"{object_id}-ep{episode_id:04d}": object_id
        for object_id, episode_ids in summary["physical_objects"].items()
        for episode_id in episode_ids
    }
    scores: dict[str, dict[str, dict[str, float]]] = {}
    alpha_records: dict[str, list[dict]] = {}
    inputs: list[dict] = []
    for case in sorted(groups):
        report_path = DEFORM_RUN / f"{case}.json"
        arrays_path = DEFORM_RUN / f"{case}.npz"
        report = json.loads(report_path.read_text())
        arrays = np.load(arrays_path)
        target_path = Path(report["inputs"]["target_data"]["path"])
        seal_path = Path(report["inputs"]["prediction_seal"]["path"])
        outcome_path = Path(report["inputs"]["outcome"]["path"])
        seal = json.loads(seal_path.read_text())
        outcome = json.loads(outcome_path.read_text())
        _validate_deform360_outcome_manifest(
            seal_path,
            target_path,
            seal,
            outcome,
        )
        data = load_pickle(target_path)
        prior = np.asarray(arrays["physical_prior_m"], dtype=float)
        beta0 = np.asarray(arrays["recursive_rbf_risk_limited_m"], dtype=float)
        target = np.asarray(data["object_points"], dtype=float)
        visible = np.asarray(data["object_visibilities"], dtype=bool)
        valid = np.asarray(data["object_motions_valid"], dtype=bool)
        centers = np.asarray(arrays["center_ids"], dtype=int)
        updates = [dict(value) for value in report["updates"]]
        test_end = len(target)
        arms, records = make_arms(
            prior,
            beta0,
            target,
            visible,
            valid,
            centers,
            updates,
            test_end,
            0,
        )
        scored = list(map(int, report["scored_frames"]))
        scores[case] = {
            arm: score_deform(trajectory, target, visible, valid, centers, scored)
            for arm, trajectory in arms.items()
        }
        alpha_records[case] = records
        inputs.append(
            {
                "case": case,
                "report_sha256": sha256(report_path),
                "arrays_sha256": sha256(arrays_path),
                "target_sha256": sha256(target_path),
            }
        )
    aggregate = aggregate_scores(scores)
    candidates = [
        name
        for name in aggregate
        if name not in {"physical", "beta0_alpha1", "frozen_alpha0"}
    ]
    comparisons = {
        candidate: {
            comparator: cluster_comparison(
                candidate, comparator, scores, groups, 20260729 + index
            )
            for index, comparator in enumerate(
                ("beta0_alpha1", "frozen_alpha0", "physical")
            )
        }
        for candidate in candidates
    }
    return {
        "dataset": "open independent-source Deform360-27 panel",
        "groups": groups,
        "case_count": len(scores),
        "aggregate": aggregate,
        "comparisons": comparisons,
        "scores_by_case": scores,
        "alpha_records": alpha_records,
        "inputs": inputs,
    }


def run_diagnostic(
    *,
    phys_run: str | Path = PHYS_RUN,
    phys_config: str | Path = PHYS_CONFIG,
    deform_run: str | Path = DEFORM_RUN,
) -> dict:
    """Reproduce the read-only causal continuation diagnostic.

    The path parameters replace only the locations of the sealed input
    artifacts.  They do not alter the legacy protocol, deterministic seeds,
    scoring, or serialization schema.
    """

    global PHYS_RUN, PHYS_CONFIG, DEFORM_RUN
    previous = (PHYS_RUN, PHYS_CONFIG, DEFORM_RUN)
    PHYS_RUN = Path(phys_run)
    PHYS_CONFIG = Path(phys_config)
    DEFORM_RUN = Path(deform_run)
    try:
        return {
            "schema_version": 1,
            "protocol": {
                "name": "causal continuation-vs-freeze diagnostic v1",
                "information_boundary": (
                    "At accepted update u, estimate alpha only from sparse center observations "
                    "at or before u and the sealed physical prior. The first interval is last "
                    "pre-test frame to u for PhysTwin and frame 0 to u for Deform360; later "
                    "intervals begin at the prior accepted update. Rejected observations are "
                    "not used and rejected forecast intervals remain the exact physical prior."
                ),
                "hybrid_equation": (
                    "corrected_current(u) + alpha_u * "
                    "[physical_prior(future) - physical_prior(u)]"
                ),
                "robust_huber_projection": (
                    "point-vector IRLS Huber projection of observed displacement onto prior "
                    "displacement; lower prior-motion quartile removed; clipped to [0,1]"
                ),
                "robust_median_projection": (
                    "linearly motion-weighted median of per-center scalar projections after "
                    "lower prior-motion quartile removal; clipped to [0,1]"
                ),
                "fixed_alpha_ablations": list(FIXED_ALPHAS),
                "causal_arms": list(CAUSAL_METHODS),
                "binary_selector_ablations": (
                    "alpha=1 when the corresponding raw robust projection is strictly above "
                    "0.25 or 0.50, otherwise alpha=0; exploratory post-hoc ablations added "
                    "after inspecting the continuous-alpha diagnostic"
                ),
                "existing_gate_decisions_and_centers_reused": True,
                "assimilation_centers_permanently_excluded": True,
                "phys_metrics": {
                    "hidden_identity_m": "frame mean of mean Euclidean material-identity error",
                    "hidden_chamfer_m": (
                        "frame mean one-sided L1 target-to-predicted-surface Chamfer; centers "
                        "removed from tracked target and predicted surface identities"
                    ),
                },
                "deform_metrics": {
                    "hidden_identity_m": "frame mean coordinate RMSE over hidden identities",
                    "hidden_chamfer_m": "frame mean symmetric Euclidean hidden-identity Chamfer",
                },
                "aggregation": (
                    "equal case/episode mean; paired 10,000-draw equal-physical-object cluster "
                    "bootstrap, deterministic seeds"
                ),
                "claim_boundary": "retrospective read-only development diagnostic; not SOTA evidence",
            },
            "phystwin22": evaluate_phys(),
            "deform360_open27": evaluate_deform(),
        }
    finally:
        PHYS_RUN, PHYS_CONFIG, DEFORM_RUN = previous


def main() -> None:
    result = run_diagnostic()
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
