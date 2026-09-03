"""Evaluate post-hoc DEFORM adapter controls on the opened DLO4/DLO5 targets.

This study is deliberately retrospective. It reuses the immutable parent run,
recomputes the frozen physical rollout and adapter from the 56 public training
trajectories, and evaluates predeclared controls on the already opened 14-case
target partitions. Scientific outcomes never affect workflow success; only
identity, information-boundary, and numerical-parity failures do.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt

from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    _initial_action_frames,
    build_deform_local_residual_features,
    fit_deform_local_residual,
    predict_deform_local_residual,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

CONTRACT: Final = "deform-dlo45-adapter-controls-v1"
RESULT_CONTRACT: Final = "deform-dlo45-adapter-controls-result-v1"
COMBINED_CONTRACT: Final = "deform-dlo45-adapter-controls-combined-v1"
DLOS: Final = ("DLO4", "DLO5")
INTERNAL: Final = slice(2, -2)
EXPECTED_PARENT_RUN: Final = 33361441865
EXPECTED_PARENT_HEAD: Final = "0376ece871d7c3d9355788f812a3c4cc1c9165b0"
EXPECTED_UPSTREAM: Final = "b73b8b8ecc033caefa693fab7898741d4e6dbeff"
EXPECTED_FEATURE_COUNT: Final = 92

ALL_FEATURES: Final = np.arange(EXPECTED_FEATURE_COUNT, dtype=np.int64)
NO_EXPLICIT_ACTION_FEATURES: Final = np.concatenate(
    (
        np.arange(0, 24, dtype=np.int64),
        np.arange(66, 69, dtype=np.int64),
        np.arange(71, 80, dtype=np.int64),
    )
)
INITIAL_ACTION_ONLY_FEATURES: Final = np.concatenate(
    (
        np.arange(0, 12, dtype=np.int64),
        np.arange(24, 60, dtype=np.int64),
    )
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _finite_array(value: object, *, ndim: int, label: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite {ndim}-D array")
    return array


def _validate_identity(value: object, *, label: str) -> Path:
    identity = _mapping(value, label=label)
    path = Path(str(identity.get("path", ""))).resolve()
    expected_size = int(identity.get("size_bytes", -1))
    expected_hash = str(identity.get("sha256", ""))
    if (
        not path.is_file()
        or path.stat().st_size != expected_size
        or sha256_file(path) != expected_hash
    ):
        raise ValueError(f"{label} identity changed: {path}")
    return path


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_json(path.resolve())
    if (
        protocol.get("schema_version") != 1
        or protocol.get("contract") != CONTRACT
        or protocol.get("evidence_class") != "retrospective-post-open-control-study"
    ):
        raise ValueError("unsupported adapter-control protocol")

    parent = _mapping(protocol.get("parent"), label="parent")
    data = _mapping(protocol.get("data"), label="data")
    adapter = _mapping(protocol.get("adapter"), label="adapter")
    controls = _mapping(protocol.get("controls"), label="controls")
    evaluation = _mapping(protocol.get("evaluation"), label="evaluation")
    parent_files = _mapping(protocol.get("parent_files"), label="parent files")

    expected_methods = (
        "primary_full_adapter",
        "compute_matched_physical",
        "global_bias",
        "node_bias",
        "time_node_mean",
        "global_frame_linear",
        "no_explicit_action_linear",
        "initial_action_only_linear",
    )
    if (
        int(parent.get("workflow_run_id", -1)) != EXPECTED_PARENT_RUN
        or parent.get("head_sha") != EXPECTED_PARENT_HEAD
        or parent.get("target_outcomes_previously_opened") is not True
        or tuple(data.get("dlos", ())) != DLOS
        or data.get("upstream_commit") != EXPECTED_UPSTREAM
        or int(data.get("train_trajectory_count", -1)) != 56
        or int(data.get("evaluation_trajectory_count", -1)) != 14
        or int(data.get("frame_count", -1)) != 500
        or int(data.get("node_count", -1)) != 12
        or data.get("new_data_collection") is not False
        or data.get("dataset_mutation") is not False
        or data.get("target_selection") is not False
        or data.get("target_calibration") is not False
        or float(adapter.get("ridge", math.nan)) != 1.0
        or float(adapter.get("shrinkage", math.nan)) != 0.25
        or int(adapter.get("expected_feature_count", -1)) != EXPECTED_FEATURE_COUNT
        or int(adapter.get("expected_internal_node_count", -1)) != 8
        or tuple(controls.get("reported_methods", ())) != expected_methods
        or tuple(controls.get("trivial_shrinkage_grid", ()))
        != (0.0, 0.25, 0.5, 0.75, 1.0)
        or int(controls.get("trivial_cv_folds", -1)) != 7
        or tuple(controls.get("data_efficiency_sizes", ())) != (1, 2, 4, 8, 16, 32, 56)
        or int(controls.get("data_efficiency_replicates", -1)) != 8
        or controls.get("subset_target_selection") is not False
        or evaluation.get("statistical_unit") != "complete-trajectory"
        or int(evaluation.get("bootstrap_replicates", -1)) != 10000
        or evaluation.get("fail_on_scientific_control_loss") is not False
        or evaluation.get("fail_on_parent_or_parity_mismatch") is not True
        or set(parent_files) != set(DLOS)
    ):
        raise ValueError("adapter-control protocol changed")

    expected_keys = {
        "source_result",
        "source_manifest",
        "method_seal",
        "prediction_seal",
        "eval_manifest",
        "physical_checkpoint",
        "compute_matched_checkpoint",
        "full_covariance_model",
        "target_predictions",
    }
    for dlo in DLOS:
        dlo_files = _mapping(parent_files[dlo], label=f"{dlo} parent files")
        if set(dlo_files) != expected_keys:
            raise ValueError(f"{dlo} parent identity roster changed")
    return protocol


def _case_l1(prediction: FloatArray, target: FloatArray, nodes: slice) -> FloatArray:
    absolute = np.abs(prediction[:, :, nodes] - target[:, :, nodes])
    return np.mean(absolute, axis=(1, 2, 3))


def score_prediction(
    prediction: FloatArray,
    baseline: FloatArray,
    target: FloatArray,
    names: Sequence[str],
) -> dict[str, Any]:
    candidate = _finite_array(prediction, ndim=4, label="candidate prediction")
    reference = _finite_array(baseline, ndim=4, label="baseline prediction")
    truth = _finite_array(target, ndim=4, label="target")
    if candidate.shape != reference.shape or candidate.shape != truth.shape:
        raise ValueError("score arrays differ")
    if candidate.shape[0] != len(names) or len(set(names)) != len(names):
        raise ValueError("score names differ")

    candidate_case = _case_l1(candidate, truth, slice(None))
    baseline_case = _case_l1(reference, truth, slice(None))
    candidate_free = _case_l1(candidate, truth, INTERNAL)
    baseline_free = _case_l1(reference, truth, INTERNAL)
    if np.any(baseline_case <= 0.0) or np.any(baseline_free <= 0.0):
        raise ValueError("baseline error must be positive")

    ratios = candidate_case / baseline_case
    free_ratios = candidate_free / baseline_free
    return {
        "candidate_mean_l1_m": float(np.mean(candidate_case)),
        "baseline_mean_l1_m": float(np.mean(baseline_case)),
        "relative_improvement": float(
            1.0 - np.mean(candidate_case) / np.mean(baseline_case)
        ),
        "wins": int(np.count_nonzero(candidate_case < baseline_case)),
        "ties": int(np.count_nonzero(candidate_case == baseline_case)),
        "losses": int(np.count_nonzero(candidate_case > baseline_case)),
        "worst_candidate_to_baseline_ratio": float(np.max(ratios)),
        "case_names": list(names),
        "candidate_case_l1_m": candidate_case.tolist(),
        "baseline_case_l1_m": baseline_case.tolist(),
        "case_ratios": ratios.tolist(),
        "free_node": {
            "candidate_mean_l1_m": float(np.mean(candidate_free)),
            "baseline_mean_l1_m": float(np.mean(baseline_free)),
            "relative_improvement": float(
                1.0 - np.mean(candidate_free) / np.mean(baseline_free)
            ),
            "wins": int(np.count_nonzero(candidate_free < baseline_free)),
            "ties": int(np.count_nonzero(candidate_free == baseline_free)),
            "losses": int(np.count_nonzero(candidate_free > baseline_free)),
            "worst_candidate_to_baseline_ratio": float(np.max(free_ratios)),
            "candidate_case_l1_m": candidate_free.tolist(),
            "baseline_case_l1_m": baseline_free.tolist(),
            "case_ratios": free_ratios.tolist(),
        },
    }


def _hash_order(
    names: Sequence[str],
    *,
    domain: str,
    dlo: str,
    replicate: int,
) -> IntArray:
    decorated = []
    for index, name in enumerate(names):
        payload = f"{domain}\0{dlo}\0{replicate}\0{name}".encode()
        decorated.append((hashlib.sha256(payload).digest(), str(name), index))
    return np.asarray([row[2] for row in sorted(decorated)], dtype=np.int64)


def _balanced_folds(
    names: Sequence[str],
    *,
    domain: str,
    dlo: str,
    folds: int,
) -> IntArray:
    if folds < 2 or folds > len(names):
        raise ValueError("invalid fold count")
    order = _hash_order(names, domain=domain, dlo=dlo, replicate=0)
    fold_ids = np.empty(len(names), dtype=np.int64)
    for rank, index in enumerate(order):
        fold_ids[index] = rank % folds
    return fold_ids


def _collapse_duplicate_queries(
    initial: FloatArray,
    action: FloatArray,
    baseline: FloatArray,
    target: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    groups: dict[bytes, list[int]] = {}
    for index in range(initial.shape[0]):
        key = b"".join(
            (
                np.ascontiguousarray(initial[index]).tobytes(),
                np.ascontiguousarray(action[index]).tobytes(),
                np.ascontiguousarray(baseline[index]).tobytes(),
            )
        )
        groups.setdefault(key, []).append(index)
    if len(groups) == initial.shape[0]:
        return initial, action, baseline, target
    grouped_initial = []
    grouped_action = []
    grouped_baseline = []
    grouped_target = []
    for indices in groups.values():
        grouped_initial.append(initial[indices[0]])
        grouped_action.append(action[indices[0]])
        grouped_baseline.append(baseline[indices[0]])
        grouped_target.append(np.mean(target[indices], axis=0))
    return (
        np.stack(grouped_initial),
        np.stack(grouped_action),
        np.stack(grouped_baseline),
        np.stack(grouped_target),
    )


def _fit_masked_linear(
    initial: FloatArray,
    action: FloatArray,
    baseline: FloatArray,
    target: FloatArray,
    *,
    feature_indices: IntArray,
    coordinate_frame: str,
    ridge: float,
) -> dict[str, Any]:
    initial, action, baseline, target = _collapse_duplicate_queries(
        initial,
        action,
        baseline,
        target,
    )
    if feature_indices.ndim != 1 or feature_indices.size == 0:
        raise ValueError("feature mask is empty")
    if np.any(feature_indices < 0) or np.any(feature_indices >= EXPECTED_FEATURE_COUNT):
        raise ValueError("feature mask is outside the feature contract")
    if len(set(feature_indices.tolist())) != feature_indices.size:
        raise ValueError("feature mask has duplicates")

    features, frames = build_deform_local_residual_features(
        initial,
        action,
        baseline,
        coordinate_frame=coordinate_frame,
    )
    if features.shape[3] != EXPECTED_FEATURE_COUNT:
        raise ValueError("feature contract changed")
    residual_global = target - baseline
    residual_local = np.einsum("ntvi,nij->ntvj", residual_global, frames)
    residual_local = residual_local[:, :, INTERNAL]
    internal_count = residual_local.shape[2]
    feature_count = int(feature_indices.size)
    locations = np.zeros((internal_count, feature_count), dtype=np.float64)
    scales = np.ones_like(locations)
    coefficients = np.zeros((internal_count, feature_count + 1, 3), dtype=np.float64)
    penalty = np.eye(feature_count + 1, dtype=np.float64) * ridge
    penalty[0, 0] = 0.0

    for node in range(internal_count):
        selected = features[:, :, node][:, :, feature_indices]
        location = np.mean(selected, axis=(0, 1))
        scale = np.std(selected, axis=(0, 1))
        scale = np.where(scale > 1e-10, scale, 1.0)
        standardized = (selected - location) / scale
        design = np.concatenate(
            (
                np.ones((*standardized.shape[:2], 1), dtype=np.float64),
                standardized,
            ),
            axis=2,
        ).reshape(-1, feature_count + 1)
        response = residual_local[:, :, node].reshape(-1, 3)
        normal = design.T @ design + penalty
        coefficients[node] = np.linalg.solve(normal, design.T @ response)
        locations[node] = location
        scales[node] = scale

    return {
        "coordinate_frame": coordinate_frame,
        "feature_indices": feature_indices.copy(),
        "feature_location": locations,
        "feature_scale": scales,
        "coefficients": coefficients,
        "ridge": float(ridge),
        "point_parameter_count": int(coefficients.size),
    }


def _predict_masked_linear(
    model: Mapping[str, Any],
    initial: FloatArray,
    action: FloatArray,
    baseline: FloatArray,
    *,
    shrinkage: float,
) -> FloatArray:
    feature_indices = np.asarray(model["feature_indices"], dtype=np.int64)
    features, frames = build_deform_local_residual_features(
        initial,
        action,
        baseline,
        coordinate_frame=str(model["coordinate_frame"]),
    )
    location = np.asarray(model["feature_location"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    internal_count = baseline.shape[2] - 4
    local_means = []
    for node in range(internal_count):
        selected = features[:, :, node][:, :, feature_indices]
        standardized = (selected - location[node]) / scale[node]
        design = np.concatenate(
            (
                np.ones((*standardized.shape[:2], 1), dtype=np.float64),
                standardized,
            ),
            axis=2,
        )
        local_means.append(np.einsum("ntd,dc->ntc", design, coefficients[node]))
    correction_local = np.stack(local_means, axis=2)
    correction_global = np.einsum("ntvj,nij->ntvi", correction_local, frames)
    candidate = baseline.copy()
    candidate[:, :, INTERNAL] += shrinkage * correction_global
    return candidate


def _local_residual(
    initial: FloatArray,
    baseline: FloatArray,
    target: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    _, frames = _initial_action_frames(initial)
    residual = np.einsum("ntvi,nij->ntvj", target - baseline, frames)[:, :, INTERNAL]
    return residual, frames


def _fit_trivial_template(residual: FloatArray, kind: str) -> FloatArray:
    if kind == "global_bias":
        return np.mean(residual, axis=(0, 1, 2))
    if kind == "node_bias":
        return np.mean(residual, axis=(0, 1))
    if kind == "time_node_mean":
        return np.mean(residual, axis=0)
    raise ValueError(f"unknown trivial template: {kind}")


def _broadcast_trivial_template(
    template: FloatArray,
    *,
    kind: str,
    trajectory_count: int,
    horizon: int,
    internal_count: int,
) -> FloatArray:
    if kind == "global_bias":
        expected = (3,)
        shape = (trajectory_count, horizon, internal_count, 3)
        source = template[None, None, None, :]
    elif kind == "node_bias":
        expected = (internal_count, 3)
        shape = (trajectory_count, horizon, internal_count, 3)
        source = template[None, None, :, :]
    elif kind == "time_node_mean":
        expected = (horizon, internal_count, 3)
        shape = (trajectory_count, horizon, internal_count, 3)
        source = template[None, :, :, :]
    else:
        raise ValueError(f"unknown trivial template: {kind}")
    if template.shape != expected:
        raise ValueError(f"{kind} template shape changed: {template.shape}")
    return np.broadcast_to(source, shape)


def _apply_local_template(
    baseline: FloatArray,
    frames: FloatArray,
    template: FloatArray,
    *,
    kind: str,
    shrinkage: float,
) -> FloatArray:
    local = _broadcast_trivial_template(
        template,
        kind=kind,
        trajectory_count=baseline.shape[0],
        horizon=baseline.shape[1],
        internal_count=baseline.shape[2] - 4,
    )
    global_correction = np.einsum("ntvj,nij->ntvi", local, frames)
    candidate = baseline.copy()
    candidate[:, :, INTERNAL] += shrinkage * global_correction
    return candidate


def _select_trivial_shrinkage(
    residual: FloatArray,
    frames: FloatArray,
    baseline: FloatArray,
    target: FloatArray,
    names: Sequence[str],
    *,
    dlo: str,
    kind: str,
    grid: Sequence[float],
    folds: int,
) -> dict[str, Any]:
    fold_ids = _balanced_folds(
        names,
        domain=f"deform-dlo45-trivial-{kind}-cv-v1",
        dlo=dlo,
        folds=folds,
    )
    scores: dict[str, float] = {}
    for shrinkage in grid:
        case_errors = np.zeros(len(names), dtype=np.float64)
        for fold in range(folds):
            train_indices = np.flatnonzero(fold_ids != fold)
            held_indices = np.flatnonzero(fold_ids == fold)
            template = _fit_trivial_template(residual[train_indices], kind)
            candidate = _apply_local_template(
                baseline[held_indices],
                frames[held_indices],
                template,
                kind=kind,
                shrinkage=float(shrinkage),
            )
            case_errors[held_indices] = _case_l1(
                candidate,
                target[held_indices],
                slice(None),
            )
        scores[f"{float(shrinkage):.2f}"] = float(np.mean(case_errors))
    selected = min((score, float(key)) for key, score in scores.items())[1]
    return {
        "selected_shrinkage": selected,
        "fold_count": folds,
        "source_cv_mean_l1_m": scores,
    }


def _trajectory_rows(
    dlo: str,
    names: Sequence[str],
    methods: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, summary in methods.items():
        candidate = cast(Sequence[float], summary["candidate_case_l1_m"])
        baseline = cast(Sequence[float], summary["baseline_case_l1_m"])
        ratios = cast(Sequence[float], summary["case_ratios"])
        for index, name in enumerate(names):
            rows.append(
                {
                    "dlo": dlo,
                    "case_name": name,
                    "method": method,
                    "candidate_l1_m": float(candidate[index]),
                    "baseline_l1_m": float(baseline[index]),
                    "candidate_to_baseline_ratio": float(ratios[index]),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if list(row) != fieldnames:
                raise ValueError(f"CSV row fields differ: {path}")
            writer.writerow(dict(row))


def _render_dlo_summary(result: Mapping[str, Any]) -> str:
    lines = [
        f"# {result['dlo']} post-hoc adapter controls",
        "",
        "This is a retrospective analysis of the already opened parent target.",
        "Scientific outcomes do not control workflow success.",
        "",
        "| Method | L1 (mm) | Gain vs physical | W/T/L | Worst ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    method_results = cast(Mapping[str, Mapping[str, Any]], result["methods"])
    for method, summary in method_results.items():
        lines.append(
            "| {method} | {l1:.4f} | {gain:.2f}% | {wins}/{ties}/{losses} | "
            "{worst:.4f} |".format(
                method=method,
                l1=1000.0 * float(summary["candidate_mean_l1_m"]),
                gain=100.0 * float(summary["relative_improvement"]),
                wins=int(summary["wins"]),
                ties=int(summary["ties"]),
                losses=int(summary["losses"]),
                worst=float(summary["worst_candidate_to_baseline_ratio"]),
            )
        )
    lines.extend(
        [
            "",
            "## Numerical custody",
            "",
            "- physical parity max abs: "
            f"`{result['parity']['physical_max_abs_m']:.3e} m`",
            "- adapter parity max abs: "
            f"`{result['parity']['candidate_max_abs_m']:.3e} m`",
            "- generic linear parity max abs: "
            f"`{result['parity']['generic_max_abs_m']:.3e} m`",
            "",
            "## Source-data efficiency",
            "",
            "| Source trajectories | Mean target L1 (mm) | Min--max (mm) |",
            "|---:|---:|---:|",
        ]
    )
    for row in cast(Sequence[Mapping[str, Any]], result["data_efficiency_summary"]):
        lines.append(
            "| {size} | {mean:.4f} | {minimum:.4f}--{maximum:.4f} |".format(
                size=int(row["source_size"]),
                mean=1000.0 * float(row["mean_candidate_l1_m"]),
                minimum=1000.0 * float(row["minimum_candidate_l1_m"]),
                maximum=1000.0 * float(row["maximum_candidate_l1_m"]),
            )
        )
    return "\n".join(lines) + "\n"


def _load_parent(
    protocol: Mapping[str, Any],
    *,
    dlo: str,
) -> dict[str, Any]:
    parent_files = _mapping(protocol["parent_files"], label="parent files")
    identities = _mapping(parent_files[dlo], label=f"{dlo} parent files")
    paths = {
        key: _validate_identity(value, label=f"{dlo} {key}")
        for key, value in identities.items()
    }
    source_result = _read_json(paths["source_result"])
    method_seal = _read_json(paths["method_seal"])
    prediction_seal = _read_json(paths["prediction_seal"])
    source_manifest = _read_json(paths["source_manifest"])
    eval_manifest = _read_json(paths["eval_manifest"])

    if (
        source_result.get("dlo") != dlo
        or source_result.get("target_eval_read") is not False
        or method_seal.get("dlo") != dlo
        or method_seal.get("target_eval_read") is not False
        or prediction_seal.get("dlo") != dlo
        or prediction_seal.get("target_eval_read") is not True
        or prediction_seal.get("target_outcomes_scored") is not False
        or source_manifest.get("dlo") != dlo
        or eval_manifest.get("dlo") != dlo
        or eval_manifest.get("target_eval_read") is not True
        or eval_manifest.get("target_outcomes_scored") is not False
    ):
        raise ValueError(f"{dlo} parent semantic boundary changed")

    cross_checks = (
        (method_seal["source_result"], identities["source_result"]),
        (method_seal["physical_checkpoint"], identities["physical_checkpoint"]),
        (
            method_seal["compute_matched_checkpoint"],
            identities["compute_matched_checkpoint"],
        ),
        (method_seal["full_covariance_model"], identities["full_covariance_model"]),
        (prediction_seal["predictions"], identities["target_predictions"]),
        (source_result["source_manifest"], identities["source_manifest"]),
    )
    for observed, expected in cross_checks:
        observed_map = _mapping(observed, label="observed parent identity")
        expected_map = _mapping(expected, label="expected parent identity")
        if observed_map.get("sha256") != expected_map.get("sha256") or int(
            observed_map.get("size_bytes", -1)
        ) != int(expected_map.get("size_bytes", -2)):
            raise ValueError(f"{dlo} parent cross-identity changed")
    return {
        "paths": paths,
        "source_result": source_result,
        "method_seal": method_seal,
        "prediction_seal": prediction_seal,
        "source_manifest": source_manifest,
        "eval_manifest": eval_manifest,
    }


def run_dlo(args: argparse.Namespace) -> int:
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    data = _mapping(protocol["data"], label="data")
    adapter = _mapping(protocol["adapter"], label="adapter")
    controls = _mapping(protocol["controls"], label="controls")
    parent = _mapping(protocol["parent"], label="parent")
    dlo = str(args.dlo)
    if dlo not in DLOS:
        raise ValueError("unsupported DLO")
    if args.dataset_root.resolve() != Path(str(data["dataset_root"])).resolve():
        raise ValueError("dataset root differs from the protocol")
    if args.upstream_root.resolve() != Path(str(data["upstream_root"])).resolve():
        raise ValueError("upstream root differs from the protocol")
    if args.parent_run_root.resolve() != Path(str(parent["run_root"])).resolve():
        raise ValueError("parent run root differs from the protocol")

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    parent_data = _load_parent(protocol, dlo=dlo)
    paths = cast(Mapping[str, Path], parent_data["paths"])
    source_manifest = cast(Mapping[str, Any], parent_data["source_manifest"])
    eval_manifest = cast(Mapping[str, Any], parent_data["eval_manifest"])

    # Import the historical execution modules only on the self-hosted runner.
    from experiments.deform_dlo45_frozen_v1.core import (
        _assert_upstream_and_initialization,
        _load_named_from_manifest,
        _load_protocol,
        _setup_torch,
        posterior_runtime,
        source_runtime,
    )

    parent_protocol_path = Path(
        str(cast(Mapping[str, Any], parent_data["method_seal"])["protocol"]["path"])
    )
    frozen_protocol = _load_protocol(parent_protocol_path)
    if sha256_file(parent_protocol_path) != parent["frozen_protocol_sha256"]:
        raise ValueError("frozen parent protocol changed")
    _assert_upstream_and_initialization(frozen_protocol, args.upstream_root, dlo)

    torch = _setup_torch(frozen_protocol, args.device)
    modules = source_runtime._load_upstream(args.upstream_root.resolve())
    # Parent training enabled deterministic kernels before every saved replay.
    source_runtime._seed_everything(
        torch, int(frozen_protocol["physical_training"]["seed"])
    )
    checkpoint = torch.load(
        paths["physical_checkpoint"],
        map_location="cpu",
        weights_only=True,
    )
    state = dict(cast(Mapping[str, Any], checkpoint["model_state_dict"]))

    train_names = [str(value) for value in source_manifest["ordered_names"]]
    eval_names = [str(value) for value in eval_manifest["ordered_names"]]
    if len(train_names) != 56 or len(eval_names) != 14:
        raise ValueError("parent roster size changed")
    train = _load_named_from_manifest(
        source_manifest,
        train_names,
        frame_count=500,
        node_count=12,
    )
    evaluation = _load_named_from_manifest(
        eval_manifest,
        eval_names,
        frame_count=500,
        node_count=12,
    )

    rollout_started = time.perf_counter()
    train_rollout = posterior_runtime._evaluate_state(
        state,
        train,
        modules=modules,
        torch=torch,
        device=args.device,
        dlo_type=dlo,
        node_count=12,
    )
    eval_rollout = posterior_runtime._evaluate_state(
        state,
        evaluation,
        modules=modules,
        torch=torch,
        device=args.device,
        dlo_type=dlo,
        node_count=12,
    )
    rollout_seconds = time.perf_counter() - rollout_started

    with np.load(paths["target_predictions"], allow_pickle=False) as archive:
        cached_names = [str(value) for value in archive["names"].tolist()]
        cached_physical = np.asarray(archive["physical"], dtype=np.float64)
        cached_compute = np.asarray(
            archive["compute_matched_physical"], dtype=np.float64
        )
        cached_candidate = np.asarray(archive["candidate"], dtype=np.float64)
    if cached_names != eval_names:
        raise ValueError("cached target order changed")

    train_physical = np.asarray(train_rollout["predictions"], dtype=np.float64)
    train_target = np.asarray(train_rollout["targets"], dtype=np.float64)
    eval_physical = np.asarray(eval_rollout["predictions"], dtype=np.float64)
    eval_target = np.asarray(eval_rollout["targets"], dtype=np.float64)
    physical_parity = float(np.max(np.abs(eval_physical - cached_physical)))
    parity_limit = float(adapter["parity_max_abs_m"])
    if physical_parity > parity_limit:
        raise RuntimeError(f"physical replay parity failed: {physical_parity}")

    from experiments.deform_dlo45_frozen_v1.core import local_runtime

    train_initial, train_action = local_runtime._causal_inputs(train, train_names)
    eval_initial, eval_action = local_runtime._causal_inputs(evaluation, eval_names)
    train_initial = np.asarray(train_initial, dtype=np.float64)
    train_action = np.asarray(train_action, dtype=np.float64)
    eval_initial = np.asarray(eval_initial, dtype=np.float64)
    eval_action = np.asarray(eval_action, dtype=np.float64)

    full_started = time.perf_counter()
    full_model = fit_deform_local_residual(
        train_initial,
        train_action,
        train_physical,
        train_target,
        train_names,
        ridge=float(adapter["ridge"]),
        variance_floor_m2=float(adapter["coordinate_variance_floor_m2"]),
    )
    refitted = predict_deform_local_residual(
        full_model,
        eval_initial,
        eval_action,
        eval_physical,
        shrinkage=float(adapter["shrinkage"]),
    )["predictions"]
    full_fit_seconds = time.perf_counter() - full_started
    candidate_parity = float(np.max(np.abs(refitted - cached_candidate)))
    if candidate_parity > parity_limit:
        raise RuntimeError(f"adapter replay parity failed: {candidate_parity}")

    generic_started = time.perf_counter()
    generic_full = _fit_masked_linear(
        train_initial,
        train_action,
        train_physical,
        train_target,
        feature_indices=ALL_FEATURES,
        coordinate_frame="initial-action-local",
        ridge=float(adapter["ridge"]),
    )
    generic_prediction = _predict_masked_linear(
        generic_full,
        eval_initial,
        eval_action,
        eval_physical,
        shrinkage=float(adapter["shrinkage"]),
    )
    generic_fit_seconds = time.perf_counter() - generic_started
    generic_parity = float(np.max(np.abs(generic_prediction - cached_candidate)))
    if generic_parity > parity_limit:
        raise RuntimeError(f"generic ridge parity failed: {generic_parity}")

    methods: dict[str, dict[str, Any]] = {
        "primary_full_adapter": score_prediction(
            cached_candidate,
            cached_physical,
            eval_target,
            eval_names,
        ),
        "compute_matched_physical": score_prediction(
            cached_compute,
            cached_physical,
            eval_target,
            eval_names,
        ),
    }
    method_metadata: dict[str, dict[str, Any]] = {
        "primary_full_adapter": {
            "source_fit_trajectories": 56,
            "point_parameter_count": int(np.asarray(full_model["coefficients"]).size),
            "fit_seconds": full_fit_seconds,
            "coordinate_frame": "initial-action-local",
            "explicit_action_features": True,
            "baseline_rollout_features": True,
        },
        "compute_matched_physical": {
            "source_fit_trajectories": 56,
            "additional_optimizer_updates": int(
                cast(Mapping[str, Any], parent_data["method_seal"])["compute_match"][
                    "additional_updates"
                ]
            ),
            "fit_seconds": float(
                cast(Mapping[str, Any], parent_data["method_seal"])["compute_match"][
                    "local_residual_wall_seconds"
                ]
            ),
        },
    }

    train_residual_local, train_frames = _local_residual(
        train_initial,
        train_physical,
        train_target,
    )
    _, eval_frames = _local_residual(eval_initial, eval_physical, eval_target)
    trivial_grid = [float(value) for value in controls["trivial_shrinkage_grid"]]
    trivial_folds = int(controls["trivial_cv_folds"])
    for kind in ("global_bias", "node_bias", "time_node_mean"):
        selection = _select_trivial_shrinkage(
            train_residual_local,
            train_frames,
            train_physical,
            train_target,
            train_names,
            dlo=dlo,
            kind=kind,
            grid=trivial_grid,
            folds=trivial_folds,
        )
        fit_started = time.perf_counter()
        template = _fit_trivial_template(train_residual_local, kind)
        prediction = _apply_local_template(
            eval_physical,
            eval_frames,
            template,
            kind=kind,
            shrinkage=float(selection["selected_shrinkage"]),
        )
        methods[kind] = score_prediction(
            prediction,
            eval_physical,
            eval_target,
            eval_names,
        )
        method_metadata[kind] = {
            "source_fit_trajectories": 56,
            "point_parameter_count": int(template.size),
            "fit_seconds": time.perf_counter() - fit_started,
            "coordinate_frame": "initial-action-local",
            "source_only_shrinkage_selection": selection,
        }

    linear_specs = {
        "global_frame_linear": {
            "features": ALL_FEATURES,
            "frame": "action-centered-global",
            "description": "all features without rotational action-frame alignment",
        },
        "no_explicit_action_linear": {
            "features": NO_EXPLICIT_ACTION_FEATURES,
            "frame": "initial-action-local",
            "description": (
                "removes explicit action, endpoint-relative, and action-interaction "
                "features; the physical baseline remains action-conditioned"
            ),
        },
        "initial_action_only_linear": {
            "features": INITIAL_ACTION_ONLY_FEATURES,
            "frame": "initial-action-local",
            "description": (
                "uses time, node coordinate, initial state, and prescribed action; "
                "removes baseline forecast geometry and dynamics"
            ),
        },
    }
    for method, spec in linear_specs.items():
        fit_started = time.perf_counter()
        model = _fit_masked_linear(
            train_initial,
            train_action,
            train_physical,
            train_target,
            feature_indices=np.asarray(spec["features"], dtype=np.int64),
            coordinate_frame=str(spec["frame"]),
            ridge=float(adapter["ridge"]),
        )
        prediction = _predict_masked_linear(
            model,
            eval_initial,
            eval_action,
            eval_physical,
            shrinkage=float(adapter["shrinkage"]),
        )
        methods[method] = score_prediction(
            prediction,
            eval_physical,
            eval_target,
            eval_names,
        )
        method_metadata[method] = {
            "source_fit_trajectories": 56,
            "point_parameter_count": int(model["point_parameter_count"]),
            "feature_count": int(np.asarray(spec["features"]).size),
            "fit_seconds": time.perf_counter() - fit_started,
            "coordinate_frame": str(spec["frame"]),
            "description": str(spec["description"]),
            "shrinkage": float(adapter["shrinkage"]),
        }

    expected_methods = [str(value) for value in controls["reported_methods"]]
    if list(methods) != expected_methods:
        raise RuntimeError(f"reported method order changed: {list(methods)}")

    efficiency_rows: list[dict[str, Any]] = []
    sizes = [int(value) for value in controls["data_efficiency_sizes"]]
    replicate_count = int(controls["data_efficiency_replicates"])
    domain = str(controls["subset_order_domain"])
    for source_size in sizes:
        repetitions = 1 if source_size == len(train_names) else replicate_count
        for replicate in range(repetitions):
            order = _hash_order(
                train_names,
                domain=domain,
                dlo=dlo,
                replicate=replicate,
            )
            selected = order[:source_size]
            fit_started = time.perf_counter()
            model = _fit_masked_linear(
                train_initial[selected],
                train_action[selected],
                train_physical[selected],
                train_target[selected],
                feature_indices=ALL_FEATURES,
                coordinate_frame="initial-action-local",
                ridge=float(adapter["ridge"]),
            )
            prediction = _predict_masked_linear(
                model,
                eval_initial,
                eval_action,
                eval_physical,
                shrinkage=float(adapter["shrinkage"]),
            )
            summary = score_prediction(
                prediction,
                eval_physical,
                eval_target,
                eval_names,
            )
            efficiency_rows.append(
                {
                    "dlo": dlo,
                    "source_size": source_size,
                    "replicate": replicate,
                    "candidate_mean_l1_m": summary["candidate_mean_l1_m"],
                    "relative_improvement": summary["relative_improvement"],
                    "wins": summary["wins"],
                    "losses": summary["losses"],
                    "worst_candidate_to_baseline_ratio": summary[
                        "worst_candidate_to_baseline_ratio"
                    ],
                    "fit_seconds": time.perf_counter() - fit_started,
                    "selected_names_sha256": hashlib.sha256(
                        "\n".join(train_names[index] for index in selected).encode()
                    ).hexdigest(),
                }
            )

    efficiency_summary = []
    for source_size in sizes:
        selected_rows = [
            row for row in efficiency_rows if int(row["source_size"]) == source_size
        ]
        errors = np.asarray(
            [float(row["candidate_mean_l1_m"]) for row in selected_rows],
            dtype=np.float64,
        )
        gains = np.asarray(
            [float(row["relative_improvement"]) for row in selected_rows],
            dtype=np.float64,
        )
        efficiency_summary.append(
            {
                "source_size": source_size,
                "replicates": len(selected_rows),
                "mean_candidate_l1_m": float(np.mean(errors)),
                "minimum_candidate_l1_m": float(np.min(errors)),
                "maximum_candidate_l1_m": float(np.max(errors)),
                "mean_relative_improvement": float(np.mean(gains)),
                "minimum_relative_improvement": float(np.min(gains)),
                "maximum_relative_improvement": float(np.max(gains)),
            }
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "contract": RESULT_CONTRACT,
        "status": "completed-retrospective-post-open-control-study",
        "dlo": dlo,
        "source_revision": args.source_revision,
        "protocol_sha256": sha256_file(protocol_path),
        "protocol_id": _canonical_sha256(protocol),
        "parent": {
            "workflow_run_id": EXPECTED_PARENT_RUN,
            "head_sha": EXPECTED_PARENT_HEAD,
            "run_root": str(args.parent_run_root.resolve()),
            "target_outcomes_previously_opened": True,
        },
        "information_boundary": {
            "new_data_collected": False,
            "dataset_mutated": False,
            "target_selection": False,
            "target_calibration": False,
            "unused_dlo_payload_opened": False,
            "scientific_outcome_controls_workflow_success": False,
        },
        "runtime": {
            "runner_name": os.environ.get("RUNNER_NAME"),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
            "rollout_seconds": rollout_seconds,
            "full_adapter_fit_and_predict_seconds": full_fit_seconds,
            "generic_full_fit_and_predict_seconds": generic_fit_seconds,
        },
        "parity": {
            "threshold_m": parity_limit,
            "physical_max_abs_m": physical_parity,
            "candidate_max_abs_m": candidate_parity,
            "generic_max_abs_m": generic_parity,
        },
        "methods": methods,
        "method_metadata": method_metadata,
        "data_efficiency": efficiency_rows,
        "data_efficiency_summary": efficiency_summary,
        "train_roster_sha256": hashlib.sha256(
            "\n".join(train_names).encode()
        ).hexdigest(),
        "eval_roster_sha256": hashlib.sha256(
            "\n".join(eval_names).encode()
        ).hexdigest(),
        "parent_file_sha256": {key: sha256_file(path) for key, path in paths.items()},
    }
    result["result_id"] = _canonical_sha256(result)
    _write_json(output / "result.json", result)
    _write_csv(
        output / "trajectory_metrics.csv",
        _trajectory_rows(dlo, eval_names, methods),
    )
    _write_csv(output / "data_efficiency.csv", efficiency_rows)
    (output / "SUMMARY.md").write_text(
        _render_dlo_summary(result),
        encoding="utf-8",
    )
    print(json.dumps({"result_id": result["result_id"], "dlo": dlo}, indent=2))
    return 0


def _stratified_bootstrap(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    gains = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        candidate_means = []
        baseline_means = []
        for dlo in DLOS:
            summary = summaries[dlo]
            candidate = np.asarray(summary["candidate_case_l1_m"], dtype=np.float64)
            baseline = np.asarray(summary["baseline_case_l1_m"], dtype=np.float64)
            indices = rng.integers(0, candidate.size, size=candidate.size)
            candidate_means.append(float(np.mean(candidate[indices])))
            baseline_means.append(float(np.mean(baseline[indices])))
        gains[replicate] = 1.0 - np.mean(candidate_means) / np.mean(baseline_means)
    return {
        "replicates": replicates,
        "seed": seed,
        "relative_improvement_interval_95": [
            float(np.quantile(gains, 0.025)),
            float(np.quantile(gains, 0.975)),
        ],
        "relative_improvement_bootstrap_mean": float(np.mean(gains)),
    }


def _render_combined_summary(result: Mapping[str, Any]) -> str:
    lines = [
        "# Combined DEFORM DLO4/DLO5 adapter controls",
        "",
        "The target outcomes were already open before this control study was "
        "registered.",
        "Results are retrospective and cannot be described as fresh confirmation.",
        "",
        "| Method | Equal-DLO L1 (mm) | Gain | 95% stratified bootstrap | W/T/L |",
        "|---|---:|---:|---:|---:|",
    ]
    methods = cast(Mapping[str, Mapping[str, Any]], result["methods"])
    for method, summary in methods.items():
        bootstrap = cast(Mapping[str, Any], summary["bootstrap"])
        interval = cast(
            Sequence[float],
            bootstrap["relative_improvement_interval_95"],
        )
        lines.append(
            "| {method} | {l1:.4f} | {gain:.2f}% | [{low:.2f}, {high:.2f}]% | "
            "{wins}/{ties}/{losses} |".format(
                method=method,
                l1=1000.0 * float(summary["equal_dlo_candidate_mean_l1_m"]),
                gain=100.0 * float(summary["equal_dlo_relative_improvement"]),
                low=100.0 * float(interval[0]),
                high=100.0 * float(interval[1]),
                wins=int(summary["wins"]),
                ties=int(summary["ties"]),
                losses=int(summary["losses"]),
            )
        )
    lines.extend(
        [
            "",
            "Workflow success indicates custody and numerical parity only; it does not",
            "declare any scientific control favorable.",
        ]
    )
    return "\n".join(lines) + "\n"


def combine(args: argparse.Namespace) -> int:
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    evaluation = _mapping(protocol["evaluation"], label="evaluation")
    results = {
        "DLO4": _read_json(args.dlo4_result.resolve()),
        "DLO5": _read_json(args.dlo5_result.resolve()),
    }
    for dlo, result in results.items():
        if (
            result.get("contract") != RESULT_CONTRACT
            or result.get("dlo") != dlo
            or result.get("status") != "completed-retrospective-post-open-control-study"
            or result.get("protocol_sha256") != sha256_file(protocol_path)
            or result.get("source_revision") != args.source_revision
        ):
            raise ValueError(f"{dlo} result differs")
        boundary = _mapping(result.get("information_boundary"), label=f"{dlo} boundary")
        if (
            boundary.get("new_data_collected") is not False
            or boundary.get("dataset_mutated") is not False
            or boundary.get("target_selection") is not False
            or boundary.get("target_calibration") is not False
            or boundary.get("scientific_outcome_controls_workflow_success") is not False
        ):
            raise ValueError(f"{dlo} information boundary changed")

    method_names = [str(value) for value in protocol["controls"]["reported_methods"]]
    combined_methods: dict[str, dict[str, Any]] = {}
    for method in method_names:
        by_dlo = {
            dlo: cast(Mapping[str, Any], result["methods"])[method]
            for dlo, result in results.items()
        }
        candidate_means = [float(by_dlo[dlo]["candidate_mean_l1_m"]) for dlo in DLOS]
        baseline_means = [float(by_dlo[dlo]["baseline_mean_l1_m"]) for dlo in DLOS]
        equal_candidate = float(np.mean(candidate_means))
        equal_baseline = float(np.mean(baseline_means))
        combined_methods[method] = {
            "equal_dlo_candidate_mean_l1_m": equal_candidate,
            "equal_dlo_baseline_mean_l1_m": equal_baseline,
            "equal_dlo_relative_improvement": 1.0 - equal_candidate / equal_baseline,
            "wins": sum(int(by_dlo[dlo]["wins"]) for dlo in DLOS),
            "ties": sum(int(by_dlo[dlo]["ties"]) for dlo in DLOS),
            "losses": sum(int(by_dlo[dlo]["losses"]) for dlo in DLOS),
            "worst_candidate_to_baseline_ratio": max(
                float(by_dlo[dlo]["worst_candidate_to_baseline_ratio"]) for dlo in DLOS
            ),
            "per_dlo": {
                dlo: {
                    key: by_dlo[dlo][key]
                    for key in (
                        "candidate_mean_l1_m",
                        "baseline_mean_l1_m",
                        "relative_improvement",
                        "wins",
                        "ties",
                        "losses",
                        "worst_candidate_to_baseline_ratio",
                    )
                }
                for dlo in DLOS
            },
            "bootstrap": _stratified_bootstrap(
                by_dlo,
                replicates=int(evaluation["bootstrap_replicates"]),
                seed=int(evaluation["bootstrap_seed"]) + method_names.index(method),
            ),
        }

    combined_efficiency = []
    sizes = [int(value) for value in protocol["controls"]["data_efficiency_sizes"]]
    for source_size in sizes:
        by_dlo_rows = {
            dlo: [
                row
                for row in cast(
                    Sequence[Mapping[str, Any]],
                    results[dlo]["data_efficiency"],
                )
                if int(row["source_size"]) == source_size
            ]
            for dlo in DLOS
        }
        common_replicates = min(len(rows) for rows in by_dlo_rows.values())
        equal_errors = []
        equal_gains = []
        for replicate in range(common_replicates):
            errors = [
                float(by_dlo_rows[dlo][replicate]["candidate_mean_l1_m"])
                for dlo in DLOS
            ]
            gains = [
                float(by_dlo_rows[dlo][replicate]["relative_improvement"])
                for dlo in DLOS
            ]
            equal_errors.append(float(np.mean(errors)))
            equal_gains.append(float(np.mean(gains)))
        combined_efficiency.append(
            {
                "source_size": source_size,
                "paired_replicates": common_replicates,
                "mean_equal_dlo_candidate_l1_m": float(np.mean(equal_errors)),
                "minimum_equal_dlo_candidate_l1_m": float(np.min(equal_errors)),
                "maximum_equal_dlo_candidate_l1_m": float(np.max(equal_errors)),
                "mean_of_per_dlo_relative_improvements": float(np.mean(equal_gains)),
            }
        )

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1,
        "contract": COMBINED_CONTRACT,
        "status": "completed-retrospective-post-open-control-study",
        "source_revision": args.source_revision,
        "protocol_sha256": sha256_file(protocol_path),
        "protocol_id": _canonical_sha256(protocol),
        "parent_workflow_run_id": EXPECTED_PARENT_RUN,
        "target_outcomes_previously_opened": True,
        "scientific_outcome_controls_workflow_success": False,
        "methods": combined_methods,
        "data_efficiency": combined_efficiency,
        "dlo_result_ids": {dlo: results[dlo]["result_id"] for dlo in DLOS},
    }
    result["result_id"] = _canonical_sha256(result)
    _write_json(output / "result.json", result)
    (output / "SUMMARY.md").write_text(
        _render_combined_summary(result),
        encoding="utf-8",
    )
    print(json.dumps({"result_id": result["result_id"]}, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--dataset-root", type=Path, required=True)
    run.add_argument("--upstream-root", type=Path, required=True)
    run.add_argument("--parent-run-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--dlo", choices=DLOS, required=True)
    run.add_argument("--device", required=True)
    run.add_argument("--source-revision", required=True)

    merge = subparsers.add_parser("combine")
    merge.add_argument("--protocol", type=Path, required=True)
    merge.add_argument("--dlo4-result", type=Path, required=True)
    merge.add_argument("--dlo5-result", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.add_argument("--source-revision", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "run":
        return run_dlo(args)
    if args.command == "combine":
        return combine(args)
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
