#!/usr/bin/env python3
"""Run a locked source-tuned tempered endpoint-model experiment.

The experiment uses only the frozen three-case development partition to select
an evidence temperature, a model-average-specific fallback guard, and
case-blocked horizon-wise covariance inflation. The nineteen-case confirmation
partition is not opened until those choices have been serialized and hashed.

This remains a retrospective, non-claim-bearing experiment because the released
22-case cohort and its development/confirmation split have already informed
BayesianPhysTwin development.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from scipy.stats import chi2

from bayesian_phystwin.endpoint_model_average import (
    ModelAveragedEndpointPosteriorV1,
    infer_model_averaged_endpoint,
    predict_model_averaged_endpoint,
)
from bayesian_phystwin.phystwin_bayesian_anchor import (
    BayesianResidualAnchorConfig,
    fit_bayesian_residual_anchor,
)
from bayesian_phystwin.phystwin_confirmatory import (
    DEVELOPMENT_CASES,
    _split_for_case,
)
from bayesian_phystwin.phystwin_residual_dynamics import (
    _lift_map,
    _load_pickle,
    _target_validity,
)


def _load_base_module() -> ModuleType:
    path = Path(__file__).with_name("run_full22_endpoint_model_average_diagnostic.py")
    spec = importlib.util.spec_from_file_location(
        "_bayesian_phystwin_full22_model_average_base",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the locked full-22 diagnostic helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base_module()
POINT_METHODS = (
    "released_phystwin",
    "selected_bayesian_anchor",
    "last_residual",
    "temperature_1_model_average",
    "tempered_model_average",
    "tempered_model_average_guard",
)
PREDICTIVE_POSTERIORS = (
    "selected_bayesian_anchor_raw",
    "temperature_1_model_average_raw",
    "tempered_model_average_raw",
    "tempered_model_average_group_conformal",
)
METRICS = ("chamfer_distance_m", "track_error_m")
HORIZON_LABELS = ("early", "middle", "late")


def _temperature_key(temperature: float) -> str:
    value = float(temperature)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("temperature must be finite and positive")
    return format(value, ".12g")


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_schema = "bayesian-phystwin-full22-tempered-model-average-experiment"
    if payload.get("schema") != expected_schema:
        raise ValueError("unexpected tempered experiment protocol schema")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported tempered experiment protocol version")
    if payload.get("status") != "retrospective-non-claim-bearing":
        raise ValueError("tempered experiment must retain its claim boundary")
    if tuple(payload["development_cases"]) != DEVELOPMENT_CASES:
        raise ValueError("tempered experiment development cases changed")
    if tuple(payload["point_methods"]) != POINT_METHODS:
        raise ValueError("tempered experiment point method ordering changed")
    if tuple(payload["predictive_posteriors"]) != PREDICTIVE_POSTERIORS:
        raise ValueError("tempered experiment posterior ordering changed")
    if tuple(payload["coverage_levels"]) != BASE.COVERAGE_LEVELS:
        raise ValueError("tempered experiment coverage levels changed")
    if tuple(payload["selective_fractions"]) != BASE.SELECTIVE_FRACTIONS:
        raise ValueError("tempered experiment selective fractions changed")
    temperatures = tuple(float(value) for value in payload["temperature_candidates"])
    if temperatures != tuple(sorted(set(temperatures))):
        raise ValueError("temperature candidates must be unique and increasing")
    if 1.0 not in temperatures or any(
        not math.isfinite(value) or value <= 0.0 for value in temperatures
    ):
        raise ValueError("temperature candidates must be positive and include 1")
    coverage = float(payload["group_conformal"]["target_coverage"])
    if not 0.0 < coverage < 1.0:
        raise ValueError("group conformal coverage must lie in (0, 1)")
    if tuple(payload["group_conformal"]["horizons"]) != HORIZON_LABELS:
        raise ValueError("group conformal horizon ordering changed")
    if float(payload["group_conformal"]["minimum_scale"]) < 1.0:
        raise ValueError("group conformal minimum scale must be at least one")
    guard_quantile = float(payload["regret_guard"]["score_quantile"])
    if not 0.0 < guard_quantile <= 1.0:
        raise ValueError("regret guard score quantile must lie in (0, 1]")
    return payload, BASE._canonical_sha256(payload)


@dataclass(frozen=True, slots=True)
class _TemperedEndpointPosterior:
    """Internal reweighted moments without inventing component probabilities."""

    mean_m: np.ndarray
    covariance_m2: np.ndarray
    update_count: np.ndarray
    component_weights: np.ndarray
    component_log_evidence: np.ndarray
    component_mean_m: np.ndarray
    component_variance_m2: np.ndarray
    component_process_variance_m2: np.ndarray
    config: Any
    end_frame: int

    @property
    def updated_mask(self) -> np.ndarray:
        return np.asarray(self.update_count > 0, dtype=bool)


def _tempered_posterior(
    posterior: ModelAveragedEndpointPosteriorV1,
    *,
    temperature: float,
) -> _TemperedEndpointPosterior:
    """Reweight a fitted component bank without reopening observations."""

    if not isinstance(posterior, ModelAveragedEndpointPosteriorV1):
        raise TypeError("posterior must be a ModelAveragedEndpointPosteriorV1")
    value = float(temperature)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("temperature must be finite and positive")
    prior = np.asarray(
        posterior.config.component_prior_probability,
        dtype=np.float64,
    )
    logits = np.log(prior)[None, :] + posterior.component_log_evidence / value
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.sum(weights, axis=1, keepdims=True)
    mean = np.einsum("nk,knc->nc", weights, posterior.component_mean_m)
    centered = posterior.component_mean_m - mean[None, :, :]
    outer = centered[:, :, :, None] * centered[:, :, None, :]
    within = posterior.component_variance_m2[:, :, None, None] * np.eye(3)
    covariance = np.einsum("nk,knij->nij", weights, within + outer)
    covariance = 0.5 * (covariance + covariance.transpose(0, 2, 1))
    return _TemperedEndpointPosterior(
        mean_m=mean,
        covariance_m2=covariance,
        update_count=posterior.update_count,
        component_weights=weights,
        component_log_evidence=posterior.component_log_evidence,
        component_mean_m=posterior.component_mean_m,
        component_variance_m2=posterior.component_variance_m2,
        component_process_variance_m2=posterior.component_process_variance_m2,
        config=posterior.config,
        end_frame=posterior.end_frame,
    )


def _predict_tempered_endpoint(
    posterior: _TemperedEndpointPosterior,
    *,
    horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(posterior, _TemperedEndpointPosterior):
        raise TypeError("posterior must be a _TemperedEndpointPosterior")
    if (
        isinstance(horizon_steps, bool)
        or int(horizon_steps) != horizon_steps
        or horizon_steps < 0
    ):
        raise ValueError("horizon_steps must be a nonnegative integer")
    horizon = int(horizon_steps)
    component_variance = (
        posterior.component_variance_m2
        + horizon * posterior.component_process_variance_m2[:, None]
    )
    centered = posterior.component_mean_m - posterior.mean_m[None, :, :]
    outer = centered[:, :, :, None] * centered[:, :, None, :]
    within = component_variance[:, :, None, None] * np.eye(3)
    covariance = np.einsum(
        "nk,knij->nij",
        posterior.component_weights,
        within + outer,
    )
    covariance = 0.5 * (covariance + covariance.transpose(0, 2, 1))
    return posterior.mean_m, covariance


def _horizon_lookup(future_count: int) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for label, indices in BASE._horizon_groups(future_count).items():
        for index in indices:
            lookup[int(index)] = label
    if len(lookup) != future_count:
        raise AssertionError("horizon groups do not cover the future interval")
    return lookup


def _finite_sample_higher_quantile(
    values: np.ndarray,
    *,
    coverage: float,
) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 1:
        raise ValueError("conformal quantile requires a nonempty vector")
    if not np.all(np.isfinite(array)):
        raise ValueError("conformal scores must be finite")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie in (0, 1)")
    ordered = np.sort(array)
    rank = int(math.ceil((len(ordered) + 1) * coverage)) - 1
    return float(ordered[min(max(rank, 0), len(ordered) - 1)])


def _guard_score(
    tempered_mean_m: np.ndarray,
    last_residual_m: np.ndarray,
    updated_mask: np.ndarray,
    *,
    quantile: float,
) -> float | None:
    mean = np.asarray(tempered_mean_m, dtype=np.float64)
    fallback = np.asarray(last_residual_m, dtype=np.float64)
    updated = np.asarray(updated_mask, dtype=bool)
    if mean.shape != fallback.shape or mean.ndim != 2 or mean.shape[1] != 3:
        raise ValueError("guard endpoint arrays must have shape (N, 3)")
    if updated.shape != (len(mean),):
        raise ValueError("guard updated mask must match the track count")
    values = np.linalg.norm(mean[updated] - fallback[updated], axis=1)
    if len(values) == 0:
        return None
    return _finite_sample_higher_quantile(values, coverage=quantile)


def _case_group_quantiles(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    train_end: int,
    updated_mask: np.ndarray,
    predictor: Any,
    coverage: float,
) -> dict[str, dict[str, float | int | None]]:
    future_count = len(residual) - train_end
    lookup = _horizon_lookup(future_count)
    collected: dict[str, list[np.ndarray]] = {label: [] for label in HORIZON_LABELS}
    for local_index, frame in enumerate(range(train_end, len(residual))):
        mean, covariance = predictor(local_index + 1)
        mask = np.asarray(valid[frame], dtype=bool) & np.asarray(
            updated_mask,
            dtype=bool,
        )
        if not np.any(mask):
            continue
        events = BASE._regularized_predictive_events(
            residual[frame, mask] - mean[mask],
            covariance[mask],
        )
        collected[lookup[local_index]].append(events["nees"])
    result: dict[str, dict[str, float | int | None]] = {}
    for label in HORIZON_LABELS:
        if not collected[label]:
            result[label] = {"count": 0, "quantile": None}
            continue
        values = np.concatenate(collected[label])
        result[label] = {
            "count": int(len(values)),
            "quantile": _finite_sample_higher_quantile(
                values,
                coverage=coverage,
            ),
        }
    return result


def _combined_relative_loss(
    candidate: Mapping[str, object],
    fallback: Mapping[str, object],
) -> float:
    ratios = []
    for metric in METRICS:
        candidate_value = float(candidate[metric])
        fallback_value = float(fallback[metric])
        if not math.isfinite(candidate_value) or not math.isfinite(fallback_value):
            raise ValueError("point metrics must be finite")
        if fallback_value <= 0.0:
            raise ValueError("fallback point metrics must be positive")
        ratios.append(candidate_value / fallback_value)
    return float(np.mean(ratios))


def _prepare_case(
    data_root: Path,
    scratch_root: Path,
    case: str,
    protocol: Mapping[str, Any],
    *,
    fit_anchor: bool,
) -> dict[str, Any]:
    case_dir = data_root / case
    case_scratch = scratch_root / case
    if case_scratch.exists():
        shutil.rmtree(case_scratch)
    case_scratch.mkdir(parents=True)

    fit_fraction = float(protocol["fit_fraction"])
    maximum_residual_m = float(protocol["maximum_residual_m"])
    interpolation_neighbors = int(protocol["interpolation_neighbors"])
    fit_end, train_end, frame_count = _split_for_case(case_dir, fit_fraction)

    anchor_summary: dict[str, Any] | None = None
    if fit_anchor:
        anchor_config = BayesianResidualAnchorConfig(
            fit_end_frame=fit_end,
            train_end_frame=train_end,
            interpolation_neighbors=interpolation_neighbors,
            maximum_residual_m=maximum_residual_m,
        )
        anchor_summary = fit_bayesian_residual_anchor(
            case_dir / "final_data.pkl",
            case_dir / "inference.pkl",
            case_dir / "gt_track_3d.pkl",
            case_scratch / "anchor",
            config=anchor_config,
        )

    data = _load_pickle(case_dir / "final_data.pkl")
    baseline = np.asarray(_load_pickle(case_dir / "inference.pkl"), dtype=float)
    gt_track = np.asarray(_load_pickle(case_dir / "gt_track_3d.pkl"), dtype=float)
    observed = np.asarray(data["object_points"], dtype=float)[:frame_count]
    visible = np.asarray(data["object_visibilities"], dtype=bool)[:frame_count]
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)[
        : frame_count - 1
    ]
    original_count = observed.shape[1]
    residual = observed - baseline[:frame_count, :original_count]
    valid = _target_validity(visible, motion_valid)
    lift_indices, lift_weights = _lift_map(
        baseline[0],
        original_count,
        interpolation_neighbors,
    )
    num_surface_points = original_count + len(np.asarray(data["surface_points"]))
    model_posterior = infer_model_averaged_endpoint(
        residual,
        valid,
        end_frame=train_end,
    )
    last_residual = BASE._last_valid_residual(
        residual,
        valid,
        end_frame=train_end,
    )
    return {
        "case": case,
        "case_scratch": case_scratch,
        "fit_end": fit_end,
        "train_end": train_end,
        "frame_count": frame_count,
        "maximum_residual_m": maximum_residual_m,
        "baseline": baseline,
        "gt_track": gt_track,
        "observed": observed,
        "visible": visible,
        "residual": residual,
        "valid": valid,
        "lift_indices": lift_indices,
        "lift_weights": lift_weights,
        "num_surface_points": num_surface_points,
        "model_posterior": model_posterior,
        "last_residual": last_residual,
        "anchor_summary": anchor_summary,
    }


def _trajectory_metrics(
    state: Mapping[str, Any],
    endpoint_mean_m: np.ndarray,
    *,
    accepted: bool = True,
) -> dict[str, object]:
    trajectory = BASE._constant_endpoint_trajectory(
        state["baseline"],
        endpoint_mean_m,
        start_frame=int(state["train_end"]),
        lift_indices=state["lift_indices"],
        lift_weights=state["lift_weights"],
        maximum_residual_m=float(state["maximum_residual_m"]),
        accepted=accepted,
    )
    return BASE._point_metrics(
        trajectory,
        state["observed"],
        state["visible"],
        state["gt_track"],
        num_surface_points=int(state["num_surface_points"]),
        start_frame=int(state["train_end"]),
        end_frame=int(state["frame_count"]),
    )


def _posterior_predictor(
    posterior: ModelAveragedEndpointPosteriorV1 | _TemperedEndpointPosterior,
) -> Any:
    def predictor(horizon_steps: int) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(posterior, _TemperedEndpointPosterior):
            return _predict_tempered_endpoint(
                posterior,
                horizon_steps=horizon_steps,
            )
        prediction = predict_model_averaged_endpoint(
            posterior,
            horizon_steps=horizon_steps,
        )
        return prediction.mean_m, prediction.covariance_m2

    return predictor


def _development_case(
    job: tuple[Path, Path, str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    data_root, scratch_root, case, protocol = job
    state = _prepare_case(
        data_root,
        scratch_root,
        case,
        protocol,
        fit_anchor=False,
    )
    raw_posterior = state["model_posterior"]
    last_point = _trajectory_metrics(state, state["last_residual"])
    guard_quantile = float(protocol["regret_guard"]["score_quantile"])
    target_coverage = float(protocol["group_conformal"]["target_coverage"])
    candidates: dict[str, object] = {}
    for temperature in protocol["temperature_candidates"]:
        value = float(temperature)
        key = _temperature_key(value)
        posterior = _tempered_posterior(raw_posterior, temperature=value)
        predictor = _posterior_predictor(posterior)
        point = _trajectory_metrics(state, posterior.mean_m)
        calibration = BASE._predictive_calibration(
            state["residual"],
            state["valid"],
            train_end=int(state["train_end"]),
            updated_mask=posterior.updated_mask,
            predictor=predictor,
        )
        candidates[key] = {
            "temperature": value,
            "point": point,
            "combined_point_loss_vs_last_residual": _combined_relative_loss(
                point,
                last_point,
            ),
            "predictive_calibration": calibration,
            "guard_score_m": _guard_score(
                posterior.mean_m,
                state["last_residual"],
                posterior.updated_mask,
                quantile=guard_quantile,
            ),
            "group_quantiles": _case_group_quantiles(
                state["residual"],
                state["valid"],
                train_end=int(state["train_end"]),
                updated_mask=posterior.updated_mask,
                predictor=predictor,
                coverage=target_coverage,
            ),
            "model_diagnostics": BASE._model_diagnostics(posterior),
        }
    shutil.rmtree(state["case_scratch"])
    return case, {
        "case": case,
        "split": {
            "fit_end_frame": int(state["fit_end"]),
            "train_end_frame": int(state["train_end"]),
            "frame_count": int(state["frame_count"]),
        },
        "last_residual_point": last_point,
        "temperature_candidates": candidates,
    }


def _select_temperature(
    development: Mapping[str, dict[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for temperature in protocol["temperature_candidates"]:
        value = float(temperature)
        key = _temperature_key(value)
        case_nll = []
        case_point_loss = []
        case_entropy = []
        for case in DEVELOPMENT_CASES:
            candidate = development[case]["temperature_candidates"][key]
            nll = candidate["predictive_calibration"]["updated_only"]["overall"].get(
                "mean_negative_log_likelihood"
            )
            if nll is None or not math.isfinite(float(nll)):
                raise ValueError(
                    f"temperature {value} has no finite development NLL for {case}"
                )
            case_nll.append(float(nll))
            case_point_loss.append(
                float(candidate["combined_point_loss_vs_last_residual"])
            )
            case_entropy.append(
                float(candidate["model_diagnostics"]["mean_component_entropy_nats"])
            )
        records.append(
            {
                "temperature": value,
                "equal_case_mean_updated_only_nll": float(np.mean(case_nll)),
                "equal_case_mean_point_loss_vs_last_residual": float(
                    np.mean(case_point_loss)
                ),
                "equal_case_mean_component_entropy_nats": float(np.mean(case_entropy)),
                "case_nll": dict(zip(DEVELOPMENT_CASES, case_nll, strict=True)),
                "case_point_loss_vs_last_residual": dict(
                    zip(DEVELOPMENT_CASES, case_point_loss, strict=True)
                ),
            }
        )
    selected = min(
        records,
        key=lambda record: (
            record["equal_case_mean_updated_only_nll"],
            record["equal_case_mean_point_loss_vs_last_residual"],
            record["temperature"],
        ),
    )
    return {
        "selected_temperature": float(selected["temperature"]),
        "objective": protocol["temperature_selection"]["objective"],
        "tie_break": protocol["temperature_selection"]["tie_break"],
        "candidate_records": records,
    }


def _select_guard(
    development: Mapping[str, dict[str, Any]],
    protocol: Mapping[str, Any],
    *,
    temperature: float,
) -> dict[str, Any]:
    key = _temperature_key(temperature)
    scores: dict[str, float] = {}
    candidate_losses: dict[str, float] = {}
    for case in DEVELOPMENT_CASES:
        candidate = development[case]["temperature_candidates"][key]
        raw_score = candidate["guard_score_m"]
        if raw_score is None or not math.isfinite(float(raw_score)):
            raise ValueError(f"development guard score is unavailable for {case}")
        scores[case] = float(raw_score)
        candidate_losses[case] = float(
            candidate["combined_point_loss_vs_last_residual"]
        )
    thresholds: list[float | None] = [None, *sorted(set(scores.values()))]
    maximum_regret = float(
        protocol["regret_guard"]["maximum_development_relative_regret"]
    )
    records: list[dict[str, Any]] = []
    for threshold in thresholds:
        decisions = {
            case: threshold is not None and scores[case] <= threshold
            for case in DEVELOPMENT_CASES
        }
        losses = {
            case: candidate_losses[case] if decisions[case] else 1.0
            for case in DEVELOPMENT_CASES
        }
        relative_regrets = {case: loss - 1.0 for case, loss in losses.items()}
        record = {
            "threshold_m": threshold,
            "accepted_case_count": int(sum(decisions.values())),
            "decisions": decisions,
            "equal_case_mean_guarded_loss_vs_fallback": float(
                np.mean(list(losses.values()))
            ),
            "maximum_case_relative_regret": float(max(relative_regrets.values())),
            "case_guarded_loss_vs_fallback": losses,
            "feasible": bool(max(relative_regrets.values()) <= maximum_regret + 1e-12),
        }
        records.append(record)
    feasible = [record for record in records if record["feasible"]]
    if not feasible:
        raise AssertionError("always-fallback guard candidate must be feasible")
    selected = min(
        feasible,
        key=lambda record: (
            record["equal_case_mean_guarded_loss_vs_fallback"],
            record["maximum_case_relative_regret"],
            -record["accepted_case_count"],
            -math.inf if record["threshold_m"] is None else record["threshold_m"],
        ),
    )
    return {
        "temperature": temperature,
        "selected_threshold_m": selected["threshold_m"],
        "fallback": protocol["regret_guard"]["fallback"],
        "score": protocol["regret_guard"]["score"],
        "maximum_development_relative_regret": maximum_regret,
        "selected_record": selected,
        "candidate_records": records,
        "development_scores_m": scores,
    }


def _fit_group_conformal(
    development: Mapping[str, dict[str, Any]],
    protocol: Mapping[str, Any],
    *,
    temperature: float,
) -> dict[str, Any]:
    key = _temperature_key(temperature)
    coverage = float(protocol["group_conformal"]["target_coverage"])
    reference = float(chi2.ppf(coverage, df=3))
    minimum_scale = float(protocol["group_conformal"]["minimum_scale"])
    scales: dict[str, float] = {}
    case_quantiles: dict[str, dict[str, float]] = {}
    case_counts: dict[str, dict[str, int]] = {}
    for label in HORIZON_LABELS:
        quantiles: dict[str, float] = {}
        counts: dict[str, int] = {}
        for case in DEVELOPMENT_CASES:
            record = development[case]["temperature_candidates"][key][
                "group_quantiles"
            ][label]
            count = int(record["count"])
            quantile = record["quantile"]
            if count < 1 or quantile is None:
                raise ValueError(f"no development conformal events for {case}/{label}")
            counts[case] = count
            quantiles[case] = float(quantile)
        group_quantile = max(quantiles.values())
        scales[label] = float(max(minimum_scale, group_quantile / reference))
        case_quantiles[label] = quantiles
        case_counts[label] = counts
    return {
        "temperature": temperature,
        "target_coverage": coverage,
        "chi_square_reference": reference,
        "scales_by_horizon": scales,
        "development_case_quantiles_by_horizon": case_quantiles,
        "development_event_counts_by_horizon": case_counts,
        "across_case_rule": protocol["group_conformal"]["across_case_rule"],
    }


def _guard_accepts(score: float | None, threshold: float | None) -> bool:
    return score is not None and threshold is not None and score <= threshold


def _final_case(
    job: tuple[Path, Path, str, dict[str, Any], dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    data_root, scratch_root, case, protocol, selection = job
    state = _prepare_case(
        data_root,
        scratch_root,
        case,
        protocol,
        fit_anchor=True,
    )
    raw_posterior = state["model_posterior"]
    temperature = float(selection["temperature"]["selected_temperature"])
    posterior = _tempered_posterior(
        raw_posterior,
        temperature=temperature,
    )
    guard_quantile = float(protocol["regret_guard"]["score_quantile"])
    guard_score = _guard_score(
        posterior.mean_m,
        state["last_residual"],
        posterior.updated_mask,
        quantile=guard_quantile,
    )
    threshold = selection["guard"]["selected_threshold_m"]
    accepted = _guard_accepts(guard_score, threshold)

    anchor_summary = state["anchor_summary"]
    if anchor_summary is None:
        raise AssertionError("final case is missing the selected anchor")
    anchor_trajectory = np.asarray(
        _load_pickle(state["case_scratch"] / "anchor" / "trajectory.pkl"),
        dtype=float,
    )
    trajectories = {
        "released_phystwin": state["baseline"],
        "selected_bayesian_anchor": anchor_trajectory,
        "last_residual": BASE._constant_endpoint_trajectory(
            state["baseline"],
            state["last_residual"],
            start_frame=int(state["train_end"]),
            lift_indices=state["lift_indices"],
            lift_weights=state["lift_weights"],
            maximum_residual_m=float(state["maximum_residual_m"]),
        ),
        "temperature_1_model_average": BASE._constant_endpoint_trajectory(
            state["baseline"],
            raw_posterior.mean_m,
            start_frame=int(state["train_end"]),
            lift_indices=state["lift_indices"],
            lift_weights=state["lift_weights"],
            maximum_residual_m=float(state["maximum_residual_m"]),
        ),
        "tempered_model_average": BASE._constant_endpoint_trajectory(
            state["baseline"],
            posterior.mean_m,
            start_frame=int(state["train_end"]),
            lift_indices=state["lift_indices"],
            lift_weights=state["lift_weights"],
            maximum_residual_m=float(state["maximum_residual_m"]),
        ),
        "tempered_model_average_guard": BASE._constant_endpoint_trajectory(
            state["baseline"],
            posterior.mean_m if accepted else state["last_residual"],
            start_frame=int(state["train_end"]),
            lift_indices=state["lift_indices"],
            lift_weights=state["lift_weights"],
            maximum_residual_m=float(state["maximum_residual_m"]),
        ),
    }
    point = {
        method: BASE._point_metrics(
            trajectory,
            state["observed"],
            state["visible"],
            state["gt_track"],
            num_surface_points=int(state["num_surface_points"]),
            start_frame=int(state["train_end"]),
            end_frame=int(state["frame_count"]),
        )
        for method, trajectory in trajectories.items()
    }

    with np.load(state["case_scratch"] / "anchor" / "posterior.npz") as anchor_npz:
        anchor_mean = np.asarray(anchor_npz["mean"], dtype=float)
        anchor_variance = np.asarray(anchor_npz["variance"], dtype=float)
        anchor_updated = np.asarray(anchor_npz["update_count"], dtype=np.int64) > 0
    anchor_selected = anchor_summary["selection"]["selected_candidate"]
    anchor_process_variance = float(anchor_selected["process_std_m"]) ** 2

    def anchor_predictor(horizon_steps: int) -> tuple[np.ndarray, np.ndarray]:
        variance = anchor_variance + horizon_steps * anchor_process_variance
        return anchor_mean, variance[:, None, None] * np.eye(3)[None]

    def raw_predictor(horizon_steps: int) -> tuple[np.ndarray, np.ndarray]:
        prediction = predict_model_averaged_endpoint(
            raw_posterior,
            horizon_steps=horizon_steps,
        )
        return prediction.mean_m, prediction.covariance_m2

    def tempered_predictor(horizon_steps: int) -> tuple[np.ndarray, np.ndarray]:
        return _predict_tempered_endpoint(
            posterior,
            horizon_steps=horizon_steps,
        )

    future_count = int(state["frame_count"]) - int(state["train_end"])
    horizon_lookup = _horizon_lookup(future_count)
    scales = selection["group_conformal"]["scales_by_horizon"]

    def conformal_predictor(horizon_steps: int) -> tuple[np.ndarray, np.ndarray]:
        mean, covariance = tempered_predictor(horizon_steps)
        label = horizon_lookup[horizon_steps - 1]
        return mean, covariance * float(scales[label])

    calibration = {
        "selected_bayesian_anchor_raw": BASE._predictive_calibration(
            state["residual"],
            state["valid"],
            train_end=int(state["train_end"]),
            updated_mask=anchor_updated,
            predictor=anchor_predictor,
        ),
        "temperature_1_model_average_raw": BASE._predictive_calibration(
            state["residual"],
            state["valid"],
            train_end=int(state["train_end"]),
            updated_mask=raw_posterior.updated_mask,
            predictor=raw_predictor,
        ),
        "tempered_model_average_raw": BASE._predictive_calibration(
            state["residual"],
            state["valid"],
            train_end=int(state["train_end"]),
            updated_mask=posterior.updated_mask,
            predictor=tempered_predictor,
        ),
        "tempered_model_average_group_conformal": (
            BASE._predictive_calibration(
                state["residual"],
                state["valid"],
                train_end=int(state["train_end"]),
                updated_mask=posterior.updated_mask,
                predictor=conformal_predictor,
            )
        ),
    }
    result = {
        "case": case,
        "cohort": "development" if case in DEVELOPMENT_CASES else "confirmation",
        "split": {
            "fit_end_frame": int(state["fit_end"]),
            "train_end_frame": int(state["train_end"]),
            "frame_count": int(state["frame_count"]),
            "future_frame_count": future_count,
        },
        "temperature": temperature,
        "guard": {
            "score_m": guard_score,
            "threshold_m": threshold,
            "accepted": accepted,
            "fallback": selection["guard"]["fallback"],
        },
        "anchor_validation": {
            "accepted": bool(anchor_summary["selection"]["accepted"]),
            "selected_process_std_m": float(anchor_selected["process_std_m"]),
            "selected_observation_std_m": float(anchor_selected["observation_std_m"]),
        },
        "point": point,
        "predictive_calibration": calibration,
        "model_diagnostics": {
            "temperature_1": BASE._model_diagnostics(raw_posterior),
            "tempered": BASE._model_diagnostics(posterior),
        },
    }
    shutil.rmtree(state["case_scratch"])
    return case, result


def _mean_or_none(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _aggregate_point(
    case_results: Mapping[str, dict[str, Any]],
    cases: Sequence[str],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    aggregate: dict[str, object] = {"methods": {}, "paired_comparisons": {}}
    methods = aggregate["methods"]
    assert isinstance(methods, dict)
    for method in POINT_METHODS:
        method_result: dict[str, object] = {}
        for metric in METRICS:
            case_values = [
                float(case_results[case]["point"][method][metric]) for case in cases
            ]
            frame_values = [
                value
                for case in cases
                for value in case_results[case]["point"][method]["by_frame"][metric]
            ]
            method_result[metric] = {
                "equal_case_mean_m": float(np.mean(case_values)),
                "frame_weighted_mean_m": float(np.mean(frame_values)),
            }
        by_horizon: dict[str, object] = {}
        for label in HORIZON_LABELS:
            by_horizon[label] = {
                metric: _mean_or_none(
                    [
                        float(
                            case_results[case]["point"][method]["by_horizon"][label][
                                metric
                            ]
                        )
                        for case in cases
                        if label in case_results[case]["point"][method]["by_horizon"]
                    ]
                )
                for metric in METRICS
            }
        method_result["equal_case_by_horizon"] = by_horizon
        methods[method] = method_result

    comparisons = aggregate["paired_comparisons"]
    assert isinstance(comparisons, dict)
    candidates = ("tempered_model_average", "tempered_model_average_guard")
    references = (
        "temperature_1_model_average",
        "selected_bayesian_anchor",
        "last_residual",
    )
    for candidate in candidates:
        for reference in references:
            comparisons[f"{candidate}_vs_{reference}"] = {
                metric: BASE._paired_bootstrap(
                    np.asarray(
                        [
                            case_results[case]["point"][candidate][metric]
                            for case in cases
                        ]
                    ),
                    np.asarray(
                        [
                            case_results[case]["point"][reference][metric]
                            for case in cases
                        ]
                    ),
                    samples=bootstrap_samples,
                    seed=bootstrap_seed,
                )
                for metric in METRICS
            }
    return aggregate


def _aggregate_calibration(
    case_results: Mapping[str, dict[str, Any]],
    cases: Sequence[str],
) -> dict[str, object]:
    scalar_keys = (
        "mean_error_norm_m",
        "rms_error_norm_m",
        "mean_predictive_std_m",
        "mean_nees",
        "mean_nees_over_dimension",
        "mean_negative_log_likelihood",
        "coverage_50",
        "coverage_90",
        "coverage_95",
    )
    result: dict[str, object] = {}
    for posterior in PREDICTIVE_POSTERIORS:
        posterior_result: dict[str, object] = {}
        for support in ("all_valid", "updated_only"):
            overall: dict[str, float | None] = {}
            for key in scalar_keys:
                values = [
                    case_results[case]["predictive_calibration"][posterior][support][
                        "overall"
                    ].get(key)
                    for case in cases
                ]
                numeric = [float(value) for value in values if value is not None]
                overall[key] = _mean_or_none(numeric)
            by_horizon: dict[str, object] = {}
            for label in HORIZON_LABELS:
                by_horizon[label] = {}
                for key in scalar_keys:
                    values = [
                        case_results[case]["predictive_calibration"][posterior][
                            support
                        ]["by_horizon"]
                        .get(label, {})
                        .get(key)
                        for case in cases
                    ]
                    numeric = [float(value) for value in values if value is not None]
                    by_horizon[label][key] = _mean_or_none(numeric)
            selective: dict[str, object] = {}
            for fraction in BASE.SELECTIVE_FRACTIONS:
                key = str(fraction)
                selective[key] = {}
                for metric in ("mean_error_norm_m", "rms_error_norm_m"):
                    values = [
                        case_results[case]["predictive_calibration"][posterior][
                            support
                        ]["overall"]
                        .get("selective_risk", {})
                        .get(key, {})
                        .get(metric)
                        for case in cases
                    ]
                    numeric = [float(value) for value in values if value is not None]
                    selective[key][metric] = _mean_or_none(numeric)
            posterior_result[support] = {
                "equal_case_overall": overall,
                "equal_case_by_horizon": by_horizon,
                "equal_case_selective_risk": selective,
            }
        result[posterior] = posterior_result
    return result


def _write_case_csv(
    path: Path,
    case_results: Mapping[str, dict[str, Any]],
) -> None:
    fieldnames = [
        "case",
        "cohort",
        "temperature",
        "guard_score_m",
        "guard_threshold_m",
        "guard_accepted",
    ]
    for method in POINT_METHODS:
        for metric in METRICS:
            fieldnames.append(f"{method}.{metric}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for case in sorted(case_results):
            result = case_results[case]
            row: dict[str, object] = {
                "case": case,
                "cohort": result["cohort"],
                "temperature": result["temperature"],
                "guard_score_m": result["guard"]["score_m"],
                "guard_threshold_m": result["guard"]["threshold_m"],
                "guard_accepted": result["guard"]["accepted"],
            }
            for method in POINT_METHODS:
                for metric in METRICS:
                    row[f"{method}.{metric}"] = result["point"][method][metric]
            writer.writerow(row)


def _run_jobs(jobs: Sequence[tuple[Any, ...]], workers: int, worker: Any) -> list[Any]:
    if workers == 1:
        return list(map(worker, jobs))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker, jobs))


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol, protocol_sha256 = _load_protocol(args.protocol)
    data_root = args.data_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"output already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    scratch = output / "_scratch"
    development_scratch = scratch / "development"
    final_scratch = scratch / "final"
    development_scratch.mkdir(parents=True)
    final_scratch.mkdir(parents=True)

    data_manifest = BASE._download_trajectory_subset(data_root)
    cases = tuple(str(case) for case in data_manifest["selected_cases"])
    if len(cases) != 22 or len(set(cases)) != 22:
        raise ValueError("the tempered experiment requires 22 unique cases")
    if not set(DEVELOPMENT_CASES) < set(cases):
        raise ValueError("the frozen development cases are not all present")

    development_jobs = [
        (data_root, development_scratch, case, protocol) for case in DEVELOPMENT_CASES
    ]
    development = dict(_run_jobs(development_jobs, args.workers, _development_case))
    temperature_selection = _select_temperature(development, protocol)
    selected_temperature = float(temperature_selection["selected_temperature"])
    guard_selection = _select_guard(
        development,
        protocol,
        temperature=selected_temperature,
    )
    conformal_selection = _fit_group_conformal(
        development,
        protocol,
        temperature=selected_temperature,
    )
    selection_core = {
        "schema": "bayesian-phystwin-tempered-model-average-selection",
        "schema_version": 1,
        "classification": "source-only-retrospective-selection",
        "protocol_sha256": protocol_sha256,
        "development_cases": list(DEVELOPMENT_CASES),
        "temperature": temperature_selection,
        "guard": guard_selection,
        "group_conformal": conformal_selection,
        "development_evidence": development,
        "confirmation_outcomes_opened": False,
    }
    selection_core["selection_id"] = BASE._canonical_sha256(selection_core)
    selection_path = output / "selection.json"
    BASE._write_json(selection_path, selection_core)
    selection_sha256 = BASE._sha256(selection_path)

    final_jobs = [
        (data_root, final_scratch, case, protocol, selection_core) for case in cases
    ]
    case_results = dict(_run_jobs(final_jobs, args.workers, _final_case))
    shutil.rmtree(scratch)

    cohorts = {
        "development_3": tuple(case for case in cases if case in DEVELOPMENT_CASES),
        "confirmation_19": tuple(
            case for case in cases if case not in DEVELOPMENT_CASES
        ),
        "all_22": cases,
    }
    bootstrap = protocol["bootstrap"]
    aggregate = {
        cohort: {
            "case_count": len(cohort_cases),
            "point": _aggregate_point(
                case_results,
                cohort_cases,
                bootstrap_samples=int(bootstrap["samples"]),
                bootstrap_seed=int(bootstrap["seed"]),
            ),
            "predictive_calibration": _aggregate_calibration(
                case_results,
                cohort_cases,
            ),
            "guard_acceptance_fraction": float(
                np.mean(
                    [case_results[case]["guard"]["accepted"] for case in cohort_cases]
                )
            ),
        }
        for cohort, cohort_cases in cohorts.items()
    }
    summary = {
        "schema": "bayesian-phystwin-full22-tempered-model-average-result",
        "schema_version": 1,
        "classification": "retrospective-non-claim-bearing-experiment",
        "protocol_sha256": protocol_sha256,
        "selection_sha256": selection_sha256,
        "selection_id": selection_core["selection_id"],
        "repository_revision": args.repository_revision,
        "claim_boundary": protocol["claim_boundary"],
        "data_manifest": {
            "path": data_manifest["manifest_path"],
            "sha256": data_manifest["manifest_sha256"],
            "selected_cases": list(cases),
        },
        "selection": {
            "selected_temperature": selected_temperature,
            "guard_threshold_m": guard_selection["selected_threshold_m"],
            "group_conformal_scales_by_horizon": conformal_selection[
                "scales_by_horizon"
            ],
            "locked_before_confirmation": True,
        },
        "case_results": case_results,
        "aggregate": aggregate,
    }
    BASE._write_json(output / "protocol.json", protocol)
    BASE._write_json(output / "summary.json", summary)
    _write_case_csv(output / "per_case.csv", case_results)
    confirmation = aggregate["confirmation_19"]
    readout = {
        "schema": "bayesian-phystwin-tempered-model-average-readout",
        "schema_version": 1,
        "classification": summary["classification"],
        "protocol_sha256": protocol_sha256,
        "selection_sha256": selection_sha256,
        "selection": summary["selection"],
        "confirmation_case_count": confirmation["case_count"],
        "confirmation_guard_acceptance_fraction": confirmation[
            "guard_acceptance_fraction"
        ],
        "confirmation_point": confirmation["point"],
        "confirmation_predictive_calibration": confirmation["predictive_calibration"],
        "claim_boundary": protocol["claim_boundary"],
    }
    BASE._write_json(output / "readout.json", readout)
    BASE._write_json(
        output / "artifact_manifest.json",
        {
            "schema": "bayesian-phystwin-tempered-model-average-artifact-manifest",
            "schema_version": 1,
            "protocol_sha256": protocol_sha256,
            "selection_sha256": selection_sha256,
            "summary_sha256": BASE._sha256(output / "summary.json"),
            "readout_sha256": BASE._sha256(output / "readout.json"),
            "per_case_csv_sha256": BASE._sha256(output / "per_case.csv"),
            "repository_revision": args.repository_revision,
        },
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repository-revision", required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    summary = run(args)
    print(
        json.dumps(
            {
                "selection": summary["selection"],
                "confirmation_19": summary["aggregate"]["confirmation_19"],
                "claim_boundary": summary["claim_boundary"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
