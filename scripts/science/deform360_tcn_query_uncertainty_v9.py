"""Grouped and causal online query-uncertainty calibration for the exact TCN mean."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from statistics import NormalDist
from typing import Any

import numpy as np

SCHEMA = "bayesian-phystwin/deform360-tcn-query-uncertainty-v9"
FOLD_SALT = "deform360-tcn-query-uncertainty-v9"
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
PRIOR_STRENGTHS = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
ONLINE_SCALE_CLIP = (0.1, 10.0)


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fold_for(group: str, fold_count: int) -> int:
    digest = hashlib.sha256((FOLD_SALT + "\0" + group).encode()).digest()
    return int.from_bytes(digest[:8], "big") % fold_count


def _groups(groups: np.ndarray) -> list[str]:
    return sorted(set(map(str, groups.tolist())))


def _require_shapes(
    residuals: np.ndarray,
    dropout_covariances: np.ndarray,
    groups: np.ndarray,
    window_indices: np.ndarray,
) -> None:
    if residuals.ndim != 2 or not np.all(np.isfinite(residuals)):
        raise ValueError("residuals must be a finite matrix")
    if dropout_covariances.shape != (
        residuals.shape[0],
        residuals.shape[1],
        residuals.shape[1],
    ) or not np.all(np.isfinite(dropout_covariances)):
        raise ValueError("dropout covariances do not align with residuals")
    if groups.shape != (residuals.shape[0],):
        raise ValueError("object identifiers do not align with residuals")
    if window_indices.shape != (residuals.shape[0],):
        raise ValueError("window indices do not align with residuals")
    if len(_groups(groups)) < 2:
        raise ValueError("at least two independent object groups are required")


def equal_object_mean(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    return np.mean(
        [np.mean(values[groups == group], axis=0) for group in _groups(groups)],
        axis=0,
    )


def homoscedastic_variance(residuals: np.ndarray, groups: np.ndarray) -> np.ndarray:
    value = equal_object_mean(residuals**2, groups)
    scale = max(float(np.max(value)), 1.0)
    return np.maximum(value, 1e-10 * scale)


def dropout_diagonal(covariances: np.ndarray) -> np.ndarray:
    value = np.diagonal(covariances, axis1=1, axis2=2).copy()
    scale = max(float(np.max(value)), 1.0)
    return np.maximum(value, 1e-10 * scale)


def normalize_feature_variance(
    feature: np.ndarray,
    source_groups: np.ndarray,
    target_variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = equal_object_mean(feature, source_groups)
    scale = target_variance / np.maximum(mean, 1e-12)
    return feature * scale[None, :], scale


def global_scale(
    residuals: np.ndarray,
    variances: np.ndarray,
    groups: np.ndarray,
) -> float:
    scores = np.mean(residuals**2 / np.maximum(variances, 1e-12), axis=1)
    object_scores = [
        float(np.mean(scores[groups == group])) for group in _groups(groups)
    ]
    return float(np.clip(np.mean(object_scores), 1e-4, 1e4))


def gaussian_nll_per_dimension(
    residuals: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    return 0.5 * np.mean(
        np.log(2.0 * math.pi * variances) + residuals**2 / variances,
        axis=1,
    )


def metrics(
    residuals: np.ndarray,
    variances: np.ndarray,
    groups: np.ndarray,
    probability: float = 0.9,
) -> dict[str, Any]:
    dimension = residuals.shape[1]
    z = NormalDist().inv_cdf(0.5 + probability / 2.0)
    chi = (
        dimension
        * (
            1.0
            - 2.0 / (9.0 * dimension)
            + NormalDist().inv_cdf(probability) * math.sqrt(2.0 / (9.0 * dimension))
        )
        ** 3
    )
    rows: list[dict[str, float]] = []
    nll = gaussian_nll_per_dimension(residuals, variances)
    distance = np.sum(residuals**2 / variances, axis=1)
    marginal_hits = np.abs(residuals) <= z * np.sqrt(variances)
    width = 2.0 * z * np.sqrt(variances)
    for group in _groups(groups):
        mask = groups == group
        rows.append(
            {
                "nll_per_dimension": float(np.mean(nll[mask])),
                "normalized_anees": float(np.mean(distance[mask] / dimension)),
                "ellipsoid_coverage": float(np.mean(distance[mask] <= chi)),
                "marginal_coverage": float(np.mean(marginal_hits[mask])),
                "mean_marginal_width": float(np.mean(width[mask])),
            }
        )
    return {
        "n_cases": int(residuals.shape[0]),
        "n_groups": len(rows),
        "query_dimension": dimension,
        "weighting": "equal-object-after-within-object-average",
        **{key: float(np.mean([row[key] for row in rows])) for key in rows[0]},
    }


def object_nll(
    residuals: np.ndarray,
    variances: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    values = gaussian_nll_per_dimension(residuals, variances)
    return {group: float(np.mean(values[groups == group])) for group in _groups(groups)}


def paired_bootstrap(
    first: Mapping[str, float],
    second: Mapping[str, float],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if set(first) != set(second):
        raise ValueError("paired object sets differ")
    names = sorted(first)
    differences = np.asarray([first[name] - second[name] for name in names])
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(names), size=(repetitions, len(names)))
    means = differences[indexes].mean(axis=1)
    return {
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "object_bootstrap_95_interval": [
            float(value) for value in np.quantile(means, [0.025, 0.975])
        ],
        "object_wins_ties_losses": [
            int(np.sum(differences < 0.0)),
            int(np.sum(differences == 0.0)),
            int(np.sum(differences > 0.0)),
        ],
        "worst_object_regret": float(np.max(differences)),
    }


def causal_online_scale(
    residuals: np.ndarray,
    static_variances: np.ndarray,
    groups: np.ndarray,
    window_indices: np.ndarray,
    *,
    prior_strength: float,
    maturity_lag_windows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if prior_strength <= 0.0 or maturity_lag_windows < 1:
        raise ValueError("invalid online calibration parameters")
    result = np.empty_like(static_variances)
    maximum_source_index = -1
    update_count = 0
    for group in _groups(groups):
        locations = np.flatnonzero(groups == group)
        order = locations[np.argsort(window_indices[locations], kind="stable")]
        if len(set(window_indices[order].tolist())) != len(order):
            raise ValueError(f"duplicate window index for object {group}")
        matured_sum = 0.0
        matured_count = 0
        for position, location in enumerate(order):
            matured_position = position - maturity_lag_windows
            if matured_position >= 0:
                matured_location = order[matured_position]
                if (
                    window_indices[matured_location] + maturity_lag_windows
                    > window_indices[location]
                ):
                    raise ValueError(
                        "online calibration attempted to use an immature outcome"
                    )
                score = float(
                    np.mean(
                        residuals[matured_location] ** 2
                        / static_variances[matured_location]
                    )
                )
                matured_sum += score
                matured_count += 1
                maximum_source_index = max(
                    maximum_source_index, int(window_indices[matured_location])
                )
                update_count += 1
            factor = (prior_strength + matured_sum) / (prior_strength + matured_count)
            factor = float(np.clip(factor, *ONLINE_SCALE_CLIP))
            result[location] = static_variances[location] * factor
    return result, {
        "maturity_lag_windows": maturity_lag_windows,
        "prior_strength": prior_strength,
        "online_scale_clip": list(ONLINE_SCALE_CLIP),
        "matured_update_count": update_count,
        "maximum_used_source_window_index": maximum_source_index,
        "future_or_current_outcome_used": False,
    }


def cyclic_shuffle_within_object(
    values: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    result = np.empty_like(values)
    offsets: dict[str, int] = {}
    for group in _groups(groups):
        locations = np.flatnonzero(groups == group)
        if len(locations) < 2:
            result[locations] = values[locations]
            offsets[group] = 0
            continue
        digest = hashlib.sha256((FOLD_SALT + "\0shuffle\0" + group).encode()).digest()
        offset = 1 + int.from_bytes(digest[:4], "big") % (len(locations) - 1)
        result[locations] = values[np.roll(locations, offset)]
        offsets[group] = offset
    return result, offsets


def blend_variance(
    homoscedastic: np.ndarray,
    normalized_dropout: np.ndarray,
    weight: float,
) -> np.ndarray:
    return (1.0 - weight) * homoscedastic[None, :] + weight * normalized_dropout


def fit_candidate(
    residuals: np.ndarray,
    dropout_variance: np.ndarray,
    groups: np.ndarray,
    window_indices: np.ndarray,
    *,
    maturity_lag_windows: int,
) -> dict[str, Any]:
    homo = homoscedastic_variance(residuals, groups)
    normalized_dropout, normalization = normalize_feature_variance(
        dropout_variance, groups, homo
    )
    candidates: list[dict[str, Any]] = []
    for weight in BLEND_WEIGHTS:
        raw = blend_variance(homo, normalized_dropout, weight)
        scale = global_scale(residuals, raw, groups)
        static = raw * scale
        static_nll = metrics(residuals, static, groups)["nll_per_dimension"]
        for prior_strength in PRIOR_STRENGTHS:
            online, online_audit = causal_online_scale(
                residuals,
                static,
                groups,
                window_indices,
                prior_strength=prior_strength,
                maturity_lag_windows=maturity_lag_windows,
            )
            online_nll = metrics(residuals, online, groups)["nll_per_dimension"]
            candidates.append(
                {
                    "dropout_blend_weight": weight,
                    "prior_strength": prior_strength,
                    "global_scale": scale,
                    "source_static_nll_per_dimension": static_nll,
                    "source_online_nll_per_dimension": online_nll,
                    "online_audit": online_audit,
                }
            )
    selected = min(
        candidates,
        key=lambda row: (
            row["source_online_nll_per_dimension"],
            row["dropout_blend_weight"],
            -row["prior_strength"],
        ),
    )
    return {
        "selected": selected,
        "homoscedastic_variance": homo,
        "dropout_normalization": normalization,
        "candidates": candidates,
    }


def apply_candidate(
    residuals: np.ndarray,
    dropout_variance: np.ndarray,
    groups: np.ndarray,
    window_indices: np.ndarray,
    fit_result: Mapping[str, Any],
    *,
    maturity_lag_windows: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    selected = fit_result["selected"]
    homo = np.asarray(fit_result["homoscedastic_variance"], dtype=np.float64)
    normalization = np.asarray(fit_result["dropout_normalization"], dtype=np.float64)
    normalized_dropout = dropout_variance * normalization[None, :]
    raw = blend_variance(
        homo, normalized_dropout, float(selected["dropout_blend_weight"])
    )
    static = raw * float(selected["global_scale"])
    online, audit = causal_online_scale(
        residuals,
        static,
        groups,
        window_indices,
        prior_strength=float(selected["prior_strength"]),
        maturity_lag_windows=maturity_lag_windows,
    )
    return static, online, audit


def study(
    residuals: np.ndarray,
    dropout_covariances: np.ndarray,
    groups: np.ndarray,
    window_indices: np.ndarray,
    *,
    fold_count: int,
    maturity_lag_windows: int,
    bootstrap_repetitions: int,
) -> dict[str, Any]:
    _require_shapes(residuals, dropout_covariances, groups, window_indices)
    unique = _groups(groups)
    group_fold = {group: fold_for(group, fold_count) for group in unique}
    if set(group_fold.values()) != set(range(fold_count)):
        raise ValueError("deterministic object folds are not all populated")
    dropout_variance = dropout_diagonal(dropout_covariances)
    shuffled_dropout, shuffle_offsets = cyclic_shuffle_within_object(
        dropout_variance, groups
    )
    arms: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "online_selected",
            "static_selected",
            "static_homoscedastic",
            "online_homoscedastic",
            "online_shuffled_dropout",
        )
    }
    arm_residuals: list[np.ndarray] = []
    arm_groups: list[np.ndarray] = []
    folds: list[dict[str, Any]] = []
    for fold in range(fold_count):
        target_mask = np.asarray([group_fold[str(group)] == fold for group in groups])
        source_mask = ~target_mask
        fit_result = fit_candidate(
            residuals[source_mask],
            dropout_variance[source_mask],
            groups[source_mask],
            window_indices[source_mask],
            maturity_lag_windows=maturity_lag_windows,
        )
        static, online, target_online_audit = apply_candidate(
            residuals[target_mask],
            dropout_variance[target_mask],
            groups[target_mask],
            window_indices[target_mask],
            fit_result,
            maturity_lag_windows=maturity_lag_windows,
        )
        homo_fit = dict(fit_result)
        homo_selected = dict(fit_result["selected"])
        homo_selected["dropout_blend_weight"] = 0.0
        homo_fit["selected"] = homo_selected
        homo_static, homo_online, homo_audit = apply_candidate(
            residuals[target_mask],
            dropout_variance[target_mask],
            groups[target_mask],
            window_indices[target_mask],
            homo_fit,
            maturity_lag_windows=maturity_lag_windows,
        )
        _, shuffled_online, shuffled_audit = apply_candidate(
            residuals[target_mask],
            shuffled_dropout[target_mask],
            groups[target_mask],
            window_indices[target_mask],
            fit_result,
            maturity_lag_windows=maturity_lag_windows,
        )
        arms["online_selected"].append(online)
        arms["static_selected"].append(static)
        arms["static_homoscedastic"].append(homo_static)
        arms["online_homoscedastic"].append(homo_online)
        arms["online_shuffled_dropout"].append(shuffled_online)
        arm_residuals.append(residuals[target_mask])
        arm_groups.append(groups[target_mask])
        folds.append(
            {
                "fold": fold,
                "source_object_count": len(_groups(groups[source_mask])),
                "target_object_count": len(_groups(groups[target_mask])),
                "target_case_count": int(np.count_nonzero(target_mask)),
                "selected": fit_result["selected"],
                "target_online_audit": target_online_audit,
                "target_homoscedastic_online_audit": homo_audit,
                "target_shuffled_online_audit": shuffled_audit,
            }
        )
    ordered_residuals = np.concatenate(arm_residuals)
    ordered_groups = np.concatenate(arm_groups)
    variances = {name: np.concatenate(values) for name, values in arms.items()}
    arm_metrics = {
        name: metrics(ordered_residuals, value, ordered_groups)
        for name, value in variances.items()
    }
    arm_object_nll = {
        name: object_nll(ordered_residuals, value, ordered_groups)
        for name, value in variances.items()
    }
    contrasts = {
        "online_selected_minus_static_homoscedastic": paired_bootstrap(
            arm_object_nll["online_selected"],
            arm_object_nll["static_homoscedastic"],
            repetitions=bootstrap_repetitions,
            seed=202609021,
        ),
        "online_selected_minus_static_selected": paired_bootstrap(
            arm_object_nll["online_selected"],
            arm_object_nll["static_selected"],
            repetitions=bootstrap_repetitions,
            seed=202609022,
        ),
        "online_selected_minus_online_homoscedastic": paired_bootstrap(
            arm_object_nll["online_selected"],
            arm_object_nll["online_homoscedastic"],
            repetitions=bootstrap_repetitions,
            seed=202609023,
        ),
        "online_selected_minus_online_shuffled_dropout": paired_bootstrap(
            arm_object_nll["online_selected"],
            arm_object_nll["online_shuffled_dropout"],
            repetitions=bootstrap_repetitions,
            seed=202609024,
        ),
    }
    primary = arm_metrics["online_selected"]
    gates = {
        "nll_better_than_static_homoscedastic": contrasts[
            "online_selected_minus_static_homoscedastic"
        ]["object_bootstrap_95_interval"][1]
        < 0.0,
        "nll_better_than_static_selected": contrasts[
            "online_selected_minus_static_selected"
        ]["object_bootstrap_95_interval"][1]
        < 0.0,
        "nll_better_than_shuffled_dropout": contrasts[
            "online_selected_minus_online_shuffled_dropout"
        ]["object_bootstrap_95_interval"][1]
        < 0.0,
        "normalized_anees_between_0p8_and_1p2": 0.8
        <= primary["normalized_anees"]
        <= 1.2,
        "marginal_coverage_between_0p87_and_0p93": 0.87
        <= primary["marginal_coverage"]
        <= 0.93,
        "zero_future_or_current_outcome_uses": all(
            not row["target_online_audit"]["future_or_current_outcome_used"]
            for row in folds
        ),
    }
    dropout_weights = [row["selected"]["dropout_blend_weight"] for row in folds]
    online_positive = all(
        gates[name]
        for name in (
            "nll_better_than_static_homoscedastic",
            "nll_better_than_static_selected",
            "normalized_anees_between_0p8_and_1p2",
            "marginal_coverage_between_0p87_and_0p93",
            "zero_future_or_current_outcome_uses",
        )
    )
    dropout_attributed = (
        all(weight > 0.0 for weight in dropout_weights)
        and gates["nll_better_than_shuffled_dropout"]
    )
    if online_positive and dropout_attributed:
        classification = "positive-online-and-dropout-attributed-query-uncertainty"
    elif online_positive:
        classification = (
            "positive-causal-online-query-calibration-without-dropout-attribution"
        )
    else:
        classification = "mixed-or-negative-query-calibration"
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 9,
        "status": "completed",
        "classification": classification,
        "case_count": int(residuals.shape[0]),
        "object_count": len(unique),
        "query_dimension": int(residuals.shape[1]),
        "fold_count": fold_count,
        "maturity_lag_windows": maturity_lag_windows,
        "candidate_contract": {
            "dropout_blend_weights": list(BLEND_WEIGHTS),
            "prior_strengths": list(PRIOR_STRENGTHS),
            "online_scale_clip": list(ONLINE_SCALE_CLIP),
            "selection_metric": "source-fold equal-object Gaussian NLL per query dimension",
            "target_object_influences_own_candidate": False,
            "online_update_uses_only_outcomes_matured_before_current_forecast": True,
        },
        "folds": folds,
        "metrics": arm_metrics,
        "contrasts": contrasts,
        "gates": gates,
        "online_positive": online_positive,
        "dropout_attributed": dropout_attributed,
        "shuffle_offsets": shuffle_offsets,
        "paper_claim_authorized": False,
        "globally_fresh_confirmation_authorized": False,
    }
    result["result_sha256"] = canonical_digest(result)
    return result


def self_test() -> None:
    rng = np.random.default_rng(20260902)
    groups = np.repeat(np.asarray([f"g{index}" for index in range(10)]), 20)
    windows = np.tile(np.arange(20), 10)
    latent_scale = np.repeat(np.linspace(0.5, 2.0, 10), 20)
    residuals = rng.normal(size=(200, 4)) * np.sqrt(latent_scale[:, None])
    dropout = np.stack([np.eye(4) * scale for scale in latent_scale], axis=0)
    result = study(
        residuals,
        dropout,
        groups,
        windows,
        fold_count=4,
        maturity_lag_windows=4,
        bootstrap_repetitions=500,
    )
    if result["object_count"] != 10 or result["case_count"] != 200:
        raise AssertionError("synthetic study shape changed")
    if not result["gates"]["zero_future_or_current_outcome_uses"]:
        raise AssertionError("online information boundary failed")
    print("Deform360 TCN query uncertainty v9 self-test passed")
