#!/usr/bin/env python3
"""Run a locked retrospective full-22 endpoint-model diagnostic.

The diagnostic compares the released PhysTwin trajectory, the frozen
winner-take-all Bayesian anchor, an evidence-weighted endpoint model average,
an anchor-validation-gated model average, and a last-residual baseline. It also
compares raw predictive calibration of the selected anchor and the model average.

This script deliberately does not promote a paper claim. The official 22-case
cohort has already informed method development, so all results are retrospective.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2

from bayesian_phystwin.endpoint_model_average import (
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
from bayesian_phystwin.phystwin_data import (
    DEFAULT_DATA_ARCHIVE,
    DEFAULT_EXPERIMENTS_ARCHIVE,
    EVALUATION_FILENAMES,
    _archive_factory,
    _available_cases,
    _retrieve_member,
)
from bayesian_phystwin.phystwin_official_evaluation import (
    official_phystwin_metrics_by_frame,
)
from bayesian_phystwin.phystwin_residual_dynamics import (
    _lift_map,
    _lift_residual,
    _load_pickle,
    _target_validity,
)

METHODS = (
    "released_phystwin",
    "selected_bayesian_anchor",
    "model_average",
    "model_average_anchor_guard",
    "last_residual",
)
POSTERIORS = ("selected_bayesian_anchor", "model_average")
HORIZON_LABELS = ("early", "middle", "late")
COVERAGE_LEVELS = (0.5, 0.9, 0.95)
SELECTIVE_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "bayesian-phystwin-full22-model-average-diagnostic":
        raise ValueError("unexpected diagnostic protocol schema")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported diagnostic protocol version")
    if payload.get("status") != "retrospective-non-claim-bearing":
        raise ValueError("diagnostic protocol must retain its claim boundary")
    declared_methods = tuple(payload["methods"])
    if declared_methods != METHODS:
        raise ValueError("diagnostic method ordering changed")
    if tuple(payload["coverage_levels"]) != COVERAGE_LEVELS:
        raise ValueError("diagnostic coverage levels changed")
    if tuple(payload["selective_fractions"]) != SELECTIVE_FRACTIONS:
        raise ValueError("diagnostic selective fractions changed")
    if tuple(payload["development_cases"]) != DEVELOPMENT_CASES:
        raise ValueError("diagnostic development cases changed")
    return payload, _canonical_sha256(payload)


def _download_trajectory_subset(data_root: Path) -> dict[str, Any]:
    """Retrieve only files needed for trajectory and residual evaluation."""

    records: dict[str, dict[str, object]] = {}
    with ExitStack() as stack:
        data_archive = stack.enter_context(_archive_factory(DEFAULT_DATA_ARCHIVE))
        experiments_archive = stack.enter_context(
            _archive_factory(DEFAULT_EXPERIMENTS_ARCHIVE)
        )
        available = _available_cases(data_archive, experiments_archive)
        if len(available) != 22:
            raise ValueError(
                "the released trajectory diagnostic requires exactly 22 "
                "complete cases; "
                f"found {len(available)}"
            )
        for case in available:
            case_dir = data_root / case
            files: dict[str, object] = {}
            for filename in EVALUATION_FILENAMES:
                member = f"data/different_types/{case}/{filename}"
                files[filename] = _retrieve_member(
                    data_archive,
                    member,
                    case_dir / filename,
                )
            member = f"experiments/{case}/inference.pkl"
            files["inference.pkl"] = _retrieve_member(
                experiments_archive,
                member,
                case_dir / "inference.pkl",
            )
            records[case] = {"files": files}

    manifest = {
        "schema": "bayesian-phystwin-trajectory-evaluation-subset",
        "schema_version": 1,
        "sources": {
            "data": DEFAULT_DATA_ARCHIVE,
            "experiments": DEFAULT_EXPERIMENTS_ARCHIVE,
        },
        "selected_cases": list(available),
        "cases": records,
    }
    manifest_path = data_root / "trajectory_evaluation_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path.resolve())
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest


def _last_valid_residual(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
) -> np.ndarray:
    values = np.asarray(residual, dtype=float)
    validity = np.asarray(valid, dtype=bool)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("residual must have shape (T, N, 3)")
    if validity.shape != values.shape[:2]:
        raise ValueError("valid must match residual")
    if not 0 < end_frame <= len(values):
        raise ValueError("end_frame lies outside residual")
    result = np.zeros((values.shape[1], 3), dtype=float)
    for track in range(values.shape[1]):
        support = np.flatnonzero(validity[:end_frame, track])
        if len(support):
            result[track] = values[support[-1], track]
    return result


def _constant_endpoint_trajectory(
    baseline: np.ndarray,
    endpoint_mean: np.ndarray,
    *,
    start_frame: int,
    lift_indices: np.ndarray,
    lift_weights: np.ndarray,
    maximum_residual_m: float,
    accepted: bool = True,
) -> np.ndarray:
    candidate = np.asarray(baseline, dtype=float).copy()
    future_count = len(candidate) - start_frame
    if future_count <= 0 or not accepted:
        return candidate
    tracked = np.repeat(
        np.asarray(endpoint_mean, dtype=float)[None],
        future_count,
        axis=0,
    )
    correction = _lift_residual(
        tracked,
        candidate.shape[1],
        lift_indices,
        lift_weights,
        maximum_norm=maximum_residual_m,
    )
    candidate[start_frame:] += correction
    return candidate


def _horizon_groups(frame_count: int) -> dict[str, np.ndarray]:
    if frame_count < 1:
        raise ValueError("future interval must contain at least one frame")
    chunks = np.array_split(np.arange(frame_count, dtype=np.int64), 3)
    return {
        label: chunk
        for label, chunk in zip(HORIZON_LABELS, chunks, strict=True)
        if len(chunk)
    }


def _point_metrics(
    trajectory: np.ndarray,
    observed: np.ndarray,
    visible: np.ndarray,
    gt_track: np.ndarray,
    *,
    num_surface_points: int,
    start_frame: int,
    end_frame: int,
) -> dict[str, object]:
    by_frame = official_phystwin_metrics_by_frame(
        trajectory,
        observed,
        visible,
        gt_track,
        num_surface_points=num_surface_points,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    result: dict[str, object] = {
        "frame_count": end_frame - start_frame,
        "chamfer_distance_m": float(np.mean(by_frame["chamfer_distance_m"])),
        "track_error_m": float(np.mean(by_frame["track_error_m"])),
        "by_frame": {
            metric: [float(value) for value in values]
            for metric, values in by_frame.items()
        },
        "by_horizon": {},
    }
    horizon = result["by_horizon"]
    assert isinstance(horizon, dict)
    for label, indices in _horizon_groups(end_frame - start_frame).items():
        horizon[label] = {
            metric: float(np.mean(values[indices]))
            for metric, values in by_frame.items()
        }
    return result


def _regularized_predictive_events(
    errors: np.ndarray,
    covariance: np.ndarray,
) -> dict[str, np.ndarray]:
    error = np.asarray(errors, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if error.ndim != 2 or error.shape[1] != 3:
        raise ValueError("errors must have shape (N, 3)")
    if cov.shape != (len(error), 3, 3):
        raise ValueError("covariance must have shape (N, 3, 3)")
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    projected = np.einsum("nji,nj->ni", eigenvectors, error)
    nees = np.sum(np.square(projected) / eigenvalues, axis=1)
    log_determinant = np.sum(np.log(eigenvalues), axis=1)
    negative_log_likelihood = 0.5 * (
        3.0 * math.log(2.0 * math.pi) + log_determinant + nees
    )
    return {
        "nees": nees,
        "negative_log_likelihood": negative_log_likelihood,
        "error_norm_m": np.linalg.norm(error, axis=1),
        "predictive_std_m": np.sqrt(np.trace(cov, axis1=1, axis2=2) / 3.0),
    }


def _summarize_event_arrays(events: Mapping[str, np.ndarray]) -> dict[str, object]:
    count = len(events["nees"])
    if count < 1:
        return {"count": 0}
    nees = events["nees"]
    nll = events["negative_log_likelihood"]
    error = events["error_norm_m"]
    std = events["predictive_std_m"]
    summary: dict[str, object] = {
        "count": count,
        "mean_error_norm_m": float(np.mean(error)),
        "rms_error_norm_m": float(np.sqrt(np.mean(np.square(error)))),
        "mean_predictive_std_m": float(np.mean(std)),
        "mean_nees": float(np.mean(nees)),
        "median_nees": float(np.median(nees)),
        "mean_nees_over_dimension": float(np.mean(nees) / 3.0),
        "mean_negative_log_likelihood": float(np.mean(nll)),
    }
    for level in COVERAGE_LEVELS:
        threshold = float(chi2.ppf(level, df=3))
        summary[f"coverage_{int(round(100 * level))}"] = float(
            np.mean(nees <= threshold)
        )
    order = np.argsort(std, kind="stable")
    selective: dict[str, object] = {}
    for fraction in SELECTIVE_FRACTIONS:
        selected_count = max(1, int(math.ceil(fraction * count)))
        selected = order[:selected_count]
        selective[str(fraction)] = {
            "selected_count": selected_count,
            "mean_error_norm_m": float(np.mean(error[selected])),
            "rms_error_norm_m": float(np.sqrt(np.mean(np.square(error[selected])))),
            "maximum_predictive_std_m": float(np.max(std[selected])),
        }
    summary["selective_risk"] = selective
    return summary


def _predictive_calibration(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    train_end: int,
    updated_mask: np.ndarray,
    predictor: Any,
) -> dict[str, object]:
    frame_count = len(residual)
    future_count = frame_count - train_end
    supports = {
        "all_valid": np.ones_like(updated_mask, dtype=bool),
        "updated_only": np.asarray(updated_mask, dtype=bool),
    }
    collected: dict[str, dict[str, list[np.ndarray]]] = {
        support_name: {
            "nees": [],
            "negative_log_likelihood": [],
            "error_norm_m": [],
            "predictive_std_m": [],
            "horizon_index": [],
        }
        for support_name in supports
    }
    for local_horizon, frame in enumerate(range(train_end, frame_count), start=1):
        mean, covariance = predictor(local_horizon)
        frame_valid = np.asarray(valid[frame], dtype=bool)
        for support_name, support in supports.items():
            mask = frame_valid & support
            if not np.any(mask):
                continue
            events = _regularized_predictive_events(
                residual[frame, mask] - mean[mask],
                covariance[mask],
            )
            for name, values in events.items():
                collected[support_name][name].append(values)
            collected[support_name]["horizon_index"].append(
                np.full(np.sum(mask), local_horizon - 1, dtype=np.int64)
            )

    result: dict[str, object] = {}
    horizon_groups = _horizon_groups(future_count)
    for support_name, arrays in collected.items():
        if not arrays["nees"]:
            result[support_name] = {"overall": {"count": 0}, "by_horizon": {}}
            continue
        concatenated = {
            name: np.concatenate(values)
            for name, values in arrays.items()
            if name != "horizon_index"
        }
        horizon_index = np.concatenate(arrays["horizon_index"])
        by_horizon: dict[str, object] = {}
        for label, indices in horizon_groups.items():
            mask = np.isin(horizon_index, indices)
            by_horizon[label] = _summarize_event_arrays(
                {name: values[mask] for name, values in concatenated.items()}
            )
        result[support_name] = {
            "overall": _summarize_event_arrays(concatenated),
            "by_horizon": by_horizon,
        }
    return result


def _model_diagnostics(posterior: Any) -> dict[str, object]:
    updated = posterior.updated_mask
    if not np.any(updated):
        return {"updated_track_count": 0}
    weights = posterior.component_weights[updated]
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-300)), axis=1)
    effective = 1.0 / np.sum(np.square(weights), axis=1)
    zero_process = np.array(
        [component.process_std_m == 0.0 for component in posterior.config.components],
        dtype=bool,
    )
    within_trace = 3.0 * np.einsum(
        "nk,kn->n",
        posterior.component_weights,
        posterior.component_variance_m2,
    )
    total_trace = np.trace(posterior.covariance_m2, axis1=1, axis2=2)
    between_fraction = np.maximum(
        0.0,
        1.0 - within_trace / np.maximum(total_trace, 1e-30),
    )
    return {
        "updated_track_count": int(np.sum(updated)),
        "mean_component_entropy_nats": float(np.mean(entropy)),
        "median_effective_component_count": float(np.median(effective)),
        "mean_zero_process_weight": float(
            np.mean(np.sum(weights[:, zero_process], axis=1))
        ),
        "median_between_model_covariance_fraction": float(
            np.median(between_fraction[updated])
        ),
    }


def _fit_case(
    job: tuple[Path, Path, str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    data_root, scratch_root, case, protocol = job
    case_dir = data_root / case
    case_scratch = scratch_root / case
    if case_scratch.exists():
        shutil.rmtree(case_scratch)
    case_scratch.mkdir(parents=True)

    fit_fraction = float(protocol["fit_fraction"])
    maximum_residual_m = float(protocol["maximum_residual_m"])
    interpolation_neighbors = int(protocol["interpolation_neighbors"])
    fit_end, train_end, frame_count = _split_for_case(case_dir, fit_fraction)
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
    last_residual = _last_valid_residual(residual, valid, end_frame=train_end)
    anchor_trajectory = np.asarray(
        _load_pickle(case_scratch / "anchor" / "trajectory.pkl"),
        dtype=float,
    )
    accepted = bool(anchor_summary["selection"]["accepted"])
    trajectories = {
        "released_phystwin": baseline,
        "selected_bayesian_anchor": anchor_trajectory,
        "model_average": _constant_endpoint_trajectory(
            baseline,
            model_posterior.mean_m,
            start_frame=train_end,
            lift_indices=lift_indices,
            lift_weights=lift_weights,
            maximum_residual_m=maximum_residual_m,
        ),
        "model_average_anchor_guard": _constant_endpoint_trajectory(
            baseline,
            model_posterior.mean_m,
            start_frame=train_end,
            lift_indices=lift_indices,
            lift_weights=lift_weights,
            maximum_residual_m=maximum_residual_m,
            accepted=accepted,
        ),
        "last_residual": _constant_endpoint_trajectory(
            baseline,
            last_residual,
            start_frame=train_end,
            lift_indices=lift_indices,
            lift_weights=lift_weights,
            maximum_residual_m=maximum_residual_m,
        ),
    }
    point = {
        method: _point_metrics(
            trajectories[method],
            observed,
            visible,
            gt_track,
            num_surface_points=num_surface_points,
            start_frame=train_end,
            end_frame=frame_count,
        )
        for method in METHODS
    }

    with np.load(case_scratch / "anchor" / "posterior.npz") as anchor_npz:
        anchor_mean = np.asarray(anchor_npz["mean"], dtype=float)
        anchor_variance = np.asarray(anchor_npz["variance"], dtype=float)
        anchor_updated = np.asarray(anchor_npz["update_count"], dtype=np.int64) > 0
    selected = anchor_summary["selection"]["selected_candidate"]
    anchor_process_variance = float(selected["process_std_m"]) ** 2

    def anchor_predictor(horizon_steps: int) -> tuple[np.ndarray, np.ndarray]:
        variance = anchor_variance + horizon_steps * anchor_process_variance
        covariance = variance[:, None, None] * np.eye(3)[None]
        return anchor_mean, covariance

    def model_predictor(horizon_steps: int) -> tuple[np.ndarray, np.ndarray]:
        prediction = predict_model_averaged_endpoint(
            model_posterior,
            horizon_steps=horizon_steps,
        )
        return prediction.mean_m, prediction.covariance_m2

    calibration = {
        "selected_bayesian_anchor": _predictive_calibration(
            residual,
            valid,
            train_end=train_end,
            updated_mask=anchor_updated,
            predictor=anchor_predictor,
        ),
        "model_average": _predictive_calibration(
            residual,
            valid,
            train_end=train_end,
            updated_mask=model_posterior.updated_mask,
            predictor=model_predictor,
        ),
    }
    result = {
        "case": case,
        "cohort": "development" if case in DEVELOPMENT_CASES else "confirmation",
        "split": {
            "fit_end_frame": fit_end,
            "train_end_frame": train_end,
            "frame_count": frame_count,
            "future_frame_count": frame_count - train_end,
        },
        "anchor_validation": {
            "accepted": accepted,
            "relative_improvement": float(
                anchor_summary["selection"]["relative_improvement"]
            ),
            "selected_process_std_m": float(selected["process_std_m"]),
            "selected_observation_std_m": float(selected["observation_std_m"]),
        },
        "point": point,
        "predictive_calibration": calibration,
        "model_average_diagnostics": _model_diagnostics(model_posterior),
    }
    shutil.rmtree(case_scratch)
    return case, result


def _mean_or_none(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _paired_bootstrap(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    candidate_values = np.asarray(candidate, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    if candidate_values.shape != reference_values.shape or candidate_values.ndim != 1:
        raise ValueError("paired bootstrap inputs must be aligned vectors")
    if len(candidate_values) < 1:
        raise ValueError("paired bootstrap requires at least one case")
    delta = candidate_values - reference_values
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    bootstrap = np.mean(delta[indices], axis=1)
    return {
        "case_count": len(delta),
        "mean_delta_m": float(np.mean(delta)),
        "median_delta_m": float(np.median(delta)),
        "lower_95_delta_m": float(np.quantile(bootstrap, 0.025)),
        "upper_95_delta_m": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_probability_mean_improvement": float(np.mean(bootstrap < 0.0)),
        "candidate_win_count": int(np.sum(delta < 0.0)),
        "tie_count": int(np.sum(delta == 0.0)),
    }


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
    for method in METHODS:
        method_result: dict[str, object] = {}
        for metric in ("chamfer_distance_m", "track_error_m"):
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
        horizon_result: dict[str, object] = {}
        for label in HORIZON_LABELS:
            horizon_result[label] = {
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
                for metric in ("chamfer_distance_m", "track_error_m")
            }
        method_result["equal_case_by_horizon"] = horizon_result
        methods[method] = method_result

    comparisons = aggregate["paired_comparisons"]
    assert isinstance(comparisons, dict)
    reference_methods = (
        "released_phystwin",
        "selected_bayesian_anchor",
        "last_residual",
    )
    candidate_methods = ("model_average", "model_average_anchor_guard")
    for candidate_method in candidate_methods:
        for reference_method in reference_methods:
            comparison_key = f"{candidate_method}_vs_{reference_method}"
            comparisons[comparison_key] = {
                metric: _paired_bootstrap(
                    np.array(
                        [
                            case_results[case]["point"][candidate_method][metric]
                            for case in cases
                        ]
                    ),
                    np.array(
                        [
                            case_results[case]["point"][reference_method][metric]
                            for case in cases
                        ]
                    ),
                    samples=bootstrap_samples,
                    seed=bootstrap_seed,
                )
                for metric in ("chamfer_distance_m", "track_error_m")
            }
    return aggregate


def _aggregate_calibration(
    case_results: Mapping[str, dict[str, Any]],
    cases: Sequence[str],
) -> dict[str, object]:
    result: dict[str, object] = {}
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
    for posterior in POSTERIORS:
        posterior_result: dict[str, object] = {}
        for support in ("all_valid", "updated_only"):
            support_result: dict[str, object] = {
                "equal_case_overall": {},
                "equal_case_by_horizon": {},
                "equal_case_selective_risk": {},
            }
            overall = support_result["equal_case_overall"]
            assert isinstance(overall, dict)
            for key in scalar_keys:
                values = [
                    case_results[case]["predictive_calibration"][posterior][support][
                        "overall"
                    ].get(key)
                    for case in cases
                ]
                numeric = [float(value) for value in values if value is not None]
                overall[key] = _mean_or_none(numeric)
            by_horizon = support_result["equal_case_by_horizon"]
            assert isinstance(by_horizon, dict)
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
            selective = support_result["equal_case_selective_risk"]
            assert isinstance(selective, dict)
            for fraction in SELECTIVE_FRACTIONS:
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
            posterior_result[support] = support_result
        result[posterior] = posterior_result
    return result


def _write_case_csv(
    path: Path,
    case_results: Mapping[str, dict[str, Any]],
) -> None:
    fieldnames = ["case", "cohort", "anchor_accepted"]
    for method in METHODS:
        for metric in ("chamfer_distance_m", "track_error_m"):
            fieldnames.append(f"{method}.{metric}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for case in sorted(case_results):
            result = case_results[case]
            row: dict[str, object] = {
                "case": case,
                "cohort": result["cohort"],
                "anchor_accepted": result["anchor_validation"]["accepted"],
            }
            for method in METHODS:
                for metric in ("chamfer_distance_m", "track_error_m"):
                    row[f"{method}.{metric}"] = result["point"][method][metric]
            writer.writerow(row)


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
    scratch.mkdir()

    data_manifest = _download_trajectory_subset(data_root)
    cases = tuple(str(case) for case in data_manifest["selected_cases"])
    if len(cases) != 22 or len(set(cases)) != 22:
        raise ValueError("the diagnostic cohort must contain 22 unique cases")
    if not set(DEVELOPMENT_CASES) < set(cases):
        raise ValueError("the frozen development cases are not all present")

    jobs = [(data_root, scratch, case, protocol) for case in cases]
    if args.workers == 1:
        fitted = list(map(_fit_case, jobs))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            fitted = list(executor.map(_fit_case, jobs))
    case_results = dict(fitted)
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
            "anchor_acceptance_fraction": float(
                np.mean(
                    [
                        case_results[case]["anchor_validation"]["accepted"]
                        for case in cohort_cases
                    ]
                )
            ),
        }
        for cohort, cohort_cases in cohorts.items()
    }
    summary = {
        "schema": "bayesian-phystwin-full22-model-average-result",
        "schema_version": 1,
        "classification": "retrospective-non-claim-bearing-diagnostic",
        "protocol_sha256": protocol_sha256,
        "repository_revision": args.repository_revision,
        "data_manifest": {
            "path": data_manifest["manifest_path"],
            "sha256": data_manifest["manifest_sha256"],
            "selected_cases": list(cases),
        },
        "claim_boundary": protocol["claim_boundary"],
        "case_results": case_results,
        "aggregate": aggregate,
    }
    _write_json(output / "protocol.json", protocol)
    _write_json(output / "summary.json", summary)
    _write_case_csv(output / "per_case.csv", case_results)
    _write_json(
        output / "artifact_manifest.json",
        {
            "schema_version": 1,
            "protocol_sha256": protocol_sha256,
            "summary_sha256": _sha256(output / "summary.json"),
            "per_case_csv_sha256": _sha256(output / "per_case.csv"),
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
    confirmation = summary["aggregate"]["confirmation_19"]
    print(json.dumps(confirmation, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
