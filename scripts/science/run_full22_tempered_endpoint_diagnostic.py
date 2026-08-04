#!/usr/bin/env python3
"""Run the frozen retrospective tempered-endpoint diagnostic on PhysTwin-22.

This experiment selects an effective evidence-count cap using only each case's
prefix validation interval.  Its deployment arm falls back exactly to the
last-supported-residual predictor when that validation does not show a locked
minimum gain.  The released cohort is already open, so the result can establish
mechanism headroom only; it is not independent validation or a SOTA claim.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from bayesian_phystwin.endpoint_model_average import (
    TemperedModelAveragedEndpointConfigV2,
    infer_model_averaged_endpoint,
    infer_tempered_model_averaged_endpoint,
    predict_tempered_model_averaged_endpoint,
)
from bayesian_phystwin.phystwin_confirmatory import DEVELOPMENT_CASES, _split_for_case
from bayesian_phystwin.phystwin_residual_dynamics import (
    _lift_map,
    _load_pickle,
    _target_validity,
)

SCHEMA = "bayesian-phystwin-full22-tempered-endpoint-diagnostic"
METHODS = (
    "released_phystwin",
    "last_residual",
    "historical_model_average",
    "tempered_selected",
    "tempered_guarded",
)
METRICS = ("chamfer_distance_m", "track_error_m")
HORIZON_LABELS = ("early", "middle", "late")


def _base_module() -> ModuleType:
    path = Path(__file__).with_name("run_full22_endpoint_model_average_diagnostic.py")
    spec = importlib.util.spec_from_file_location("full22_endpoint_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load frozen base diagnostic: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _base_module()


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema") != SCHEMA or protocol.get("schema_version") != 1:
        raise ValueError("unexpected tempered-endpoint protocol schema")
    if protocol.get("status") != "retrospective-non-claim-bearing":
        raise ValueError("tempered diagnostic must retain its claim boundary")
    if tuple(protocol.get("methods", ())) != METHODS:
        raise ValueError("tempered diagnostic method ordering changed")
    if tuple(protocol.get("development_cases", ())) != DEVELOPMENT_CASES:
        raise ValueError("development cases changed")
    caps = tuple(float(value) for value in protocol["effective_evidence_count_caps"])
    if not caps or not all(np.isfinite(caps)) or not all(value > 0.0 for value in caps):
        raise ValueError("effective evidence-count caps must be finite and positive")
    if len(set(caps)) != len(caps) or tuple(sorted(caps)) != caps:
        raise ValueError("effective evidence-count caps must be unique and sorted")
    guard = protocol["validation_guard"]
    if float(guard["minimum_absolute_improvement_m"]) < 0.0:
        raise ValueError("absolute validation improvement must be nonnegative")
    if float(guard["minimum_relative_improvement"]) < 0.0:
        raise ValueError("relative validation improvement must be nonnegative")
    return protocol, BASE._canonical_sha256(protocol)


def _endpoint_rmse(
    residual_m: np.ndarray,
    valid: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
    endpoint_mean_m: np.ndarray,
) -> float:
    residual = np.asarray(residual_m, dtype=np.float64)
    validity = np.asarray(valid, dtype=bool)
    mean = np.asarray(endpoint_mean_m, dtype=np.float64)
    if residual.ndim != 3 or residual.shape[2] != 3:
        raise ValueError("residual_m must have shape (T, N, 3)")
    if validity.shape != residual.shape[:2] or mean.shape != residual.shape[1:]:
        raise ValueError("validation arrays have incompatible shapes")
    if not 0 <= start_frame < end_frame <= len(residual):
        raise ValueError("validation interval is empty or out of bounds")
    mask = validity[start_frame:end_frame]
    if not np.any(mask):
        return math.inf
    error = residual[start_frame:end_frame] - mean[None, :, :]
    return float(np.sqrt(np.mean(np.sum(np.square(error[mask]), axis=1))))


def _select_tempered_cap(
    residual_m: np.ndarray,
    valid: np.ndarray,
    *,
    fit_end: int,
    train_end: int,
    caps: Sequence[float],
    minimum_absolute_improvement_m: float,
    minimum_relative_improvement: float,
) -> dict[str, object]:
    last_fit = BASE._last_valid_residual(residual_m, valid, end_frame=fit_end)
    fallback_rmse = _endpoint_rmse(
        residual_m,
        valid,
        start_frame=fit_end,
        end_frame=train_end,
        endpoint_mean_m=last_fit,
    )
    candidates: list[dict[str, float]] = []
    for cap in caps:
        posterior = infer_tempered_model_averaged_endpoint(
            residual_m,
            valid,
            end_frame=fit_end,
            config=TemperedModelAveragedEndpointConfigV2(
                effective_evidence_count_cap=float(cap)
            ),
        )
        validation_rmse = _endpoint_rmse(
            residual_m,
            valid,
            start_frame=fit_end,
            end_frame=train_end,
            endpoint_mean_m=posterior.mean_m,
        )
        candidates.append(
            {
                "effective_evidence_count_cap": float(cap),
                "validation_rmse_m": validation_rmse,
            }
        )
    selected = min(
        candidates,
        key=lambda item: (
            float(item["validation_rmse_m"]),
            float(item["effective_evidence_count_cap"]),
        ),
    )
    selected_rmse = float(selected["validation_rmse_m"])
    absolute = fallback_rmse - selected_rmse
    relative = absolute / fallback_rmse if fallback_rmse > 0.0 else -math.inf
    accepted = bool(
        np.isfinite(fallback_rmse)
        and np.isfinite(selected_rmse)
        and absolute >= minimum_absolute_improvement_m
        and relative >= minimum_relative_improvement
    )
    return {
        "fallback_validation_rmse_m": fallback_rmse,
        "selected_cap": float(selected["effective_evidence_count_cap"]),
        "selected_validation_rmse_m": selected_rmse,
        "absolute_improvement_m": absolute,
        "relative_improvement": relative,
        "accepted": accepted,
        "candidates": candidates,
    }


def _tempered_diagnostics(posterior: Any) -> dict[str, float]:
    updated = posterior.updated_mask
    if not np.any(updated):
        return {
            "updated_track_count": 0.0,
            "mean_component_entropy_nats": 0.0,
            "median_effective_component_count": 0.0,
            "mean_evidence_power": 1.0,
            "median_between_model_covariance_fraction": 0.0,
        }
    weights = posterior.component_weights[updated]
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-300)), axis=1)
    effective = 1.0 / np.sum(np.square(weights), axis=1)
    base = posterior.base_posterior
    within_trace = 3.0 * np.einsum(
        "nk,kn->n", posterior.component_weights, base.component_variance_m2
    )
    total_trace = np.trace(posterior.covariance_m2, axis1=1, axis2=2)
    between_fraction = np.maximum(
        0.0,
        1.0 - within_trace / np.maximum(total_trace, 1e-30),
    )
    return {
        "updated_track_count": float(np.sum(updated)),
        "mean_component_entropy_nats": float(np.mean(entropy)),
        "median_effective_component_count": float(np.median(effective)),
        "mean_evidence_power": float(np.mean(posterior.evidence_power[updated])),
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

    fit_end, train_end, frame_count = _split_for_case(
        case_dir, float(protocol["fit_fraction"])
    )
    maximum_residual_m = float(protocol["maximum_residual_m"])
    neighbors = int(protocol["interpolation_neighbors"])
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
    lift_indices, lift_weights = _lift_map(baseline[0], original_count, neighbors)
    surface_count = original_count + len(np.asarray(data["surface_points"]))

    guard = protocol["validation_guard"]
    selection = _select_tempered_cap(
        residual,
        valid,
        fit_end=fit_end,
        train_end=train_end,
        caps=protocol["effective_evidence_count_caps"],
        minimum_absolute_improvement_m=float(
            guard["minimum_absolute_improvement_m"]
        ),
        minimum_relative_improvement=float(guard["minimum_relative_improvement"]),
    )
    selected_cap = float(selection["selected_cap"])
    historical = infer_model_averaged_endpoint(residual, valid, end_frame=train_end)
    selected = infer_tempered_model_averaged_endpoint(
        residual,
        valid,
        end_frame=train_end,
        config=TemperedModelAveragedEndpointConfigV2(
            effective_evidence_count_cap=selected_cap
        ),
    )
    last_residual = BASE._last_valid_residual(residual, valid, end_frame=train_end)

    trajectories = {
        "released_phystwin": baseline,
        "last_residual": BASE._constant_endpoint_trajectory(
            baseline,
            last_residual,
            start_frame=train_end,
            lift_indices=lift_indices,
            lift_weights=lift_weights,
            maximum_residual_m=maximum_residual_m,
        ),
        "historical_model_average": BASE._constant_endpoint_trajectory(
            baseline,
            historical.mean_m,
            start_frame=train_end,
            lift_indices=lift_indices,
            lift_weights=lift_weights,
            maximum_residual_m=maximum_residual_m,
        ),
        "tempered_selected": BASE._constant_endpoint_trajectory(
            baseline,
            selected.mean_m,
            start_frame=train_end,
            lift_indices=lift_indices,
            lift_weights=lift_weights,
            maximum_residual_m=maximum_residual_m,
        ),
    }
    trajectories["tempered_guarded"] = (
        trajectories["tempered_selected"]
        if selection["accepted"]
        else trajectories["last_residual"]
    )
    point = {
        method: BASE._point_metrics(
            trajectories[method],
            observed,
            visible,
            gt_track,
            num_surface_points=surface_count,
            start_frame=train_end,
            end_frame=frame_count,
        )
        for method in METHODS
    }

    cap_point: dict[str, object] = {}
    for cap in protocol["effective_evidence_count_caps"]:
        posterior = infer_tempered_model_averaged_endpoint(
            residual,
            valid,
            end_frame=train_end,
            config=TemperedModelAveragedEndpointConfigV2(
                effective_evidence_count_cap=float(cap)
            ),
        )
        trajectory = BASE._constant_endpoint_trajectory(
            baseline,
            posterior.mean_m,
            start_frame=train_end,
            lift_indices=lift_indices,
            lift_weights=lift_weights,
            maximum_residual_m=maximum_residual_m,
        )
        cap_point[str(float(cap))] = BASE._point_metrics(
            trajectory,
            observed,
            visible,
            gt_track,
            num_surface_points=surface_count,
            start_frame=train_end,
            end_frame=frame_count,
        )

    def selected_predictor(horizon_steps: int) -> tuple[np.ndarray, np.ndarray]:
        prediction = predict_tempered_model_averaged_endpoint(
            selected, horizon_steps=horizon_steps
        )
        return prediction.mean_m, prediction.covariance_m2

    calibration = BASE._predictive_calibration(
        residual,
        valid,
        train_end=train_end,
        updated_mask=selected.updated_mask,
        predictor=selected_predictor,
    )
    shutil.rmtree(case_scratch)
    return case, {
        "case": case,
        "cohort": "development" if case in DEVELOPMENT_CASES else "confirmation",
        "split": {
            "fit_end_frame": fit_end,
            "train_end_frame": train_end,
            "frame_count": frame_count,
        },
        "selection": selection,
        "point": point,
        "cap_point": cap_point,
        "predictive_calibration": calibration,
        "tempered_diagnostics": _tempered_diagnostics(selected),
    }


def _aggregate(
    case_results: Mapping[str, dict[str, Any]],
    cases: Sequence[str],
    *,
    protocol: Mapping[str, Any],
) -> dict[str, object]:
    methods: dict[str, object] = {}
    for method in METHODS:
        methods[method] = {
            metric: {
                "equal_case_mean_m": float(
                    np.mean(
                        [case_results[case]["point"][method][metric] for case in cases]
                    )
                )
            }
            for metric in METRICS
        }
    caps: dict[str, object] = {}
    for cap in protocol["effective_evidence_count_caps"]:
        key = str(float(cap))
        caps[key] = {
            metric: float(
                np.mean(
                    [case_results[case]["cap_point"][key][metric] for case in cases]
                )
            )
            for metric in METRICS
        }
    bootstrap = protocol["bootstrap"]
    comparisons: dict[str, object] = {}
    for method in ("historical_model_average", "tempered_selected", "tempered_guarded"):
        comparisons[f"{method}_vs_last_residual"] = {
            metric: BASE._paired_bootstrap(
                np.asarray(
                    [case_results[case]["point"][method][metric] for case in cases]
                ),
                np.asarray(
                    [
                        case_results[case]["point"]["last_residual"][metric]
                        for case in cases
                    ]
                ),
                samples=int(bootstrap["samples"]),
                seed=int(bootstrap["seed"]),
            )
            for metric in METRICS
        }
    guard_accepts = [bool(case_results[case]["selection"]["accepted"]) for case in cases]
    selected_caps = [float(case_results[case]["selection"]["selected_cap"]) for case in cases]
    fallback = np.asarray(
        [
            [case_results[case]["point"]["last_residual"][metric] for metric in METRICS]
            for case in cases
        ]
    )
    selected = np.asarray(
        [
            [
                case_results[case]["point"]["tempered_selected"][metric]
                for metric in METRICS
            ]
            for case in cases
        ]
    )
    oracle = np.minimum(fallback, selected)
    return {
        "case_count": len(cases),
        "methods": methods,
        "fixed_caps": caps,
        "comparisons": comparisons,
        "guard_accept_count": int(np.sum(guard_accepts)),
        "selected_cap_counts": {
            str(cap): selected_caps.count(cap) for cap in sorted(set(selected_caps))
        },
        "selective_oracle": {
            metric: {
                "mean_m": float(np.mean(oracle[:, index])),
                "relative_improvement_over_last_residual": float(
                    1.0 - np.mean(oracle[:, index]) / np.mean(fallback[:, index])
                ),
            }
            for index, metric in enumerate(METRICS)
        },
    }


def _gate(summary: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, object]:
    confirmation = summary["aggregate"]["confirmation_19"]
    methods = confirmation["methods"]
    requirements = protocol["advancement_gate"]
    relative: dict[str, float] = {}
    for metric in METRICS:
        candidate = methods["tempered_guarded"][metric]["equal_case_mean_m"]
        fallback = methods["last_residual"][metric]["equal_case_mean_m"]
        relative[metric] = float(1.0 - candidate / fallback)
    case_results = summary["case_results"]
    confirmation_cases = [
        case for case, result in case_results.items() if result["cohort"] == "confirmation"
    ]
    joint_wins = 0
    maximum_regression = 0.0
    for case in confirmation_cases:
        candidate = case_results[case]["point"]["tempered_guarded"]
        fallback = case_results[case]["point"]["last_residual"]
        if all(candidate[metric] < fallback[metric] for metric in METRICS):
            joint_wins += 1
        maximum_regression = max(
            maximum_regression,
            *(candidate[metric] / fallback[metric] - 1.0 for metric in METRICS),
        )
    checks = {
        "minimum_cd_improvement": relative["chamfer_distance_m"]
        >= float(requirements["minimum_relative_improvement"]),
        "minimum_track_improvement": relative["track_error_m"]
        >= float(requirements["minimum_relative_improvement"]),
        "minimum_joint_wins": joint_wins >= int(requirements["minimum_joint_wins"]),
        "maximum_case_metric_regression": maximum_regression
        <= float(requirements["maximum_case_metric_regression"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "relative_improvement": relative,
        "joint_win_count": joint_wins,
        "maximum_case_metric_regression": maximum_regression,
        "decision": (
            "fresh grouped calibration may be designed"
            if all(checks.values())
            else "close this candidate without fresh evaluation"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol, protocol_sha256 = _load_protocol(args.protocol)
    output = args.output_dir.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"output already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    scratch = output / "_scratch"
    scratch.mkdir()
    data_manifest = BASE._download_trajectory_subset(args.data_root.resolve())
    cases = tuple(str(case) for case in data_manifest["selected_cases"])
    if len(cases) != 22 or len(set(cases)) != 22:
        raise ValueError("the retrospective diagnostic requires 22 unique cases")
    jobs = [(args.data_root.resolve(), scratch, case, protocol) for case in cases]
    if args.workers == 1:
        fitted = list(map(_fit_case, jobs))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            fitted = list(executor.map(_fit_case, jobs))
    case_results = dict(fitted)
    shutil.rmtree(scratch)
    cohorts = {
        "development_3": tuple(case for case in cases if case in DEVELOPMENT_CASES),
        "confirmation_19": tuple(case for case in cases if case not in DEVELOPMENT_CASES),
        "all_22": cases,
    }
    summary: dict[str, Any] = {
        "schema": "bayesian-phystwin-full22-tempered-endpoint-result",
        "schema_version": 1,
        "classification": "retrospective-non-claim-bearing-diagnostic",
        "claim_boundary": protocol["claim_boundary"],
        "protocol_sha256": protocol_sha256,
        "repository_revision": args.repository_revision,
        "data_manifest": {
            "path": data_manifest["manifest_path"],
            "sha256": data_manifest["manifest_sha256"],
            "selected_cases": list(cases),
        },
        "case_results": case_results,
        "aggregate": {
            name: _aggregate(case_results, cohort, protocol=protocol)
            for name, cohort in cohorts.items()
        },
    }
    summary["advancement_gate"] = _gate(summary, protocol)
    BASE._write_json(output / "protocol.json", protocol)
    BASE._write_json(output / "summary.json", summary)
    BASE._write_json(
        output / "artifact_manifest.json",
        {
            "schema_version": 1,
            "protocol_sha256": protocol_sha256,
            "summary_sha256": BASE._sha256(output / "summary.json"),
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
    result = run(args)
    print(json.dumps(result["advancement_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
