"""Run retrospective controls for the DEFORM post-hoc residual adapter."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    build_deform_local_residual_features,
    deform_causal_inputs,
    deserialize_deform_local_residual_model,
    predict_deform_local_residual,
)
from experiments.deform_dlo45_frozen_v1 import core as frozen_core
from experiments.deform_dlo45_frozen_v1 import model as frozen_model
from experiments.deform_dlo_posthoc_controls_v1.model import (
    candidate_from_canonical,
    deterministic_subset_indices,
    equal_dlo_bootstrap,
    feature_indices,
    fit_linear_residual,
    node_constant_bias,
    predict_linear_residual,
    score_arm,
    summarize_repeated_curve,
    time_node_mean_residual,
    trajectory_rows,
)

CONTRACT = "deform-dlo-posthoc-controls-v1"
DLOS = ("DLO4", "DLO5")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_json(path)
    parent = _mapping(protocol.get("parent"), "parent")
    runtime = _mapping(protocol.get("runtime"), "runtime")
    data = _mapping(protocol.get("data"), "data")
    adapter = _mapping(protocol.get("adapter"), "adapter")
    arms = _mapping(protocol.get("arms"), "arms")
    evaluation = _mapping(protocol.get("evaluation"), "evaluation")
    if (
        protocol.get("schema_version") != 1
        or protocol.get("contract") != CONTRACT
        or protocol.get("evidence_class") != "retrospective-post-open-control-study"
        or int(parent.get("workflow_run_id", -1)) != 33361441865
        or parent.get("head_sha") != "0376ece871d7c3d9355788f812a3c4cc1c9165b0"
        or parent.get("joint_prediction_seal_required") is not True
        or parent.get("both_target_results_already_open") is not True
        or runtime.get("registered_runner_name") != "workstation1"
        or tuple(runtime.get("runner_labels", ()))
        != ("self-hosted", "Linux", "X64", "gpuserver4090")
        or tuple(data.get("dlos", ())) != DLOS
        or int(data.get("train_trajectories_per_dlo", -1)) != 56
        or int(data.get("evaluation_trajectories_per_dlo", -1)) != 14
        or int(data.get("frame_count", -1)) != 500
        or int(data.get("node_count", -1)) != 12
        or int(data.get("observed_prefix_frames", -1)) != 2
        or int(data.get("future_frames", -1)) != 498
        or data.get("new_data_collection") is not False
        or data.get("dataset_mutation") is not False
        or float(adapter.get("ridge", np.nan)) != 1.0
        or float(adapter.get("shrinkage", np.nan)) != 0.25
        or adapter.get("coordinate_frame") != "initial-action-local"
        or adapter.get("complete_trajectory_groups") is not True
        or adapter.get("clamped_nodes_baseline_exact") is not True
        or tuple(arms.get("source_data_sizes", ())) != (1, 2, 4, 8, 16, 32, 56)
        or int(arms.get("source_data_repeats", -1)) != 5
        or int(evaluation.get("bootstrap_replicates", -1)) != 10000
        or evaluation.get("statistical_unit") != "complete-trajectory"
        or evaluation.get(
            "trivial_and_structural_arms_are_falsification_controls_not_gate_tuning"
        )
        is not True
    ):
        raise ValueError("frozen post-hoc control protocol differs")
    return protocol


def _verified_identity(value: object, label: str) -> Path:
    identity = _mapping(value, label)
    path = Path(str(identity.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(identity.get("size_bytes", -1))
        or _sha256(path) != identity.get("sha256")
    ):
        raise ValueError(f"{label} identity changed")
    return path


def _load_parent_predictions(target_root: Path) -> dict[str, np.ndarray[Any, Any]]:
    seal_path = target_root / "prediction_seal.json"
    seal = _read_json(seal_path)
    if (
        seal.get("contract") != "deform-dlo45-target-prediction-seal-v1"
        or seal.get("target_case_count") != 14
        or seal.get("point_mean_count") != 1
        or seal.get("target_eval_read") is not True
        or seal.get("target_outcomes_scored") is not False
        or seal.get("retry_authorized") is not False
        or seal.get("case_replacement") is not False
    ):
        raise ValueError("parent target prediction seal differs")
    prediction_path = _verified_identity(seal.get("predictions"), "parent predictions")
    with np.load(prediction_path, allow_pickle=False) as archive:
        required = {
            "names",
            "physical",
            "compute_matched_physical",
            "persistence",
            "candidate",
        }
        if not required.issubset(archive.files):
            raise ValueError("parent prediction archive is incomplete")
        return {key: np.asarray(archive[key]) for key in required}


def _load_parent_model(target_root: Path) -> dict[str, object]:
    method_seal = _read_json(target_root / "method_seal.json")
    if (
        method_seal.get("contract") != "deform-dlo45-alltrain-method-seal-v1"
        or method_seal.get("target_selection") is not False
        or method_seal.get("target_calibration") is not False
        or method_seal.get("target_retries") is not False
        or float(method_seal.get("ridge", np.nan)) != 1.0
        or float(method_seal.get("shrinkage", np.nan)) != 0.25
    ):
        raise ValueError("parent method seal differs")
    model_path = _verified_identity(
        method_seal.get("full_covariance_model"), "parent residual model"
    )
    with np.load(model_path, allow_pickle=False) as archive:
        return deserialize_deform_local_residual_model(archive)


def _load_trajectories(
    dataset_root: Path,
    dlo: str,
    partition: str,
    *,
    names: Sequence[str] | None = None,
) -> tuple[list[str], dict[str, np.ndarray[Any, Any]]]:
    paths = frozen_core._paths(dataset_root, dlo, partition)
    by_name = {path.name: path for path in paths}
    ordered = list(by_name) if names is None else [str(name) for name in names]
    if len(ordered) != len(set(ordered)) or set(ordered) != set(by_name):
        raise ValueError(f"{dlo}/{partition} trajectory roster differs")
    trajectories = {
        name: frozen_core.source_runtime._load_trajectory(
            by_name[name], frame_count=500, node_count=12
        )
        for name in ordered
    }
    return ordered, trajectories


def _load_train_physical_rollout(
    protocol: Mapping[str, Any],
    *,
    dlo: str,
    target_root: Path,
    dataset_root: Path,
    upstream_root: Path,
    device: str,
) -> tuple[list[str], dict[str, np.ndarray[Any, Any]], np.ndarray[Any, Any], dict[str, Any]]:
    parent_protocol_path = Path("experiments/deform_dlo45_frozen_v1/protocol.json")
    parent_protocol = frozen_core._load_protocol(parent_protocol_path)
    frozen_core._assert_upstream_and_initialization(parent_protocol, upstream_root, dlo)
    train_names, train = _load_trajectories(dataset_root, dlo, "train")
    torch = frozen_core._setup_torch(parent_protocol, device)
    modules = frozen_core.source_runtime._load_upstream(upstream_root)
    checkpoint_path = target_root / "alltrain" / "physical_update_6400.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        bundle.get("dlo") != dlo
        or int(bundle.get("update", -1)) != 6400
        or bundle.get("target_eval_read") is not False
    ):
        raise ValueError("parent all-train checkpoint differs")
    started = time.perf_counter()
    rollout = frozen_core.posterior_runtime._evaluate_state(
        dict(bundle["model_state_dict"]),
        train,
        modules=modules,
        torch=torch,
        device=device,
        dlo_type=dlo,
        node_count=12,
    )
    elapsed = time.perf_counter() - started
    if list(rollout["names"]) != train_names:
        raise ValueError("train rollout order differs")
    predictions = np.asarray(rollout["predictions"], dtype=np.float64)
    targets = np.asarray(rollout["targets"], dtype=np.float64)
    expected_targets = np.stack([train[name][2:] for name in train_names])
    if not np.array_equal(targets, expected_targets):
        raise ValueError("parent train rollout target alignment differs")
    runtime = {
        "checkpoint": _identity(checkpoint_path),
        "rollout_seconds": elapsed,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(device),
    }
    del rollout, bundle
    torch.cuda.empty_cache()
    return train_names, train, predictions, runtime


def _score_and_record(
    candidate: np.ndarray[Any, Any],
    physical: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    names: Sequence[str],
    *,
    dlo: str,
    arm: str,
    all_rows: list[dict[str, Any]],
    repeat: int | None = None,
    source_count: int | None = None,
) -> dict[str, Any]:
    summary = score_arm(candidate, physical, target, names)
    all_rows.extend(
        trajectory_rows(
            summary,
            dlo=dlo,
            arm=arm,
            repeat=repeat,
            source_count=source_count,
        )
    )
    return summary


def _fit_predict(
    train_features: np.ndarray[Any, Any],
    train_residual_canonical: np.ndarray[Any, Any],
    eval_features: np.ndarray[Any, Any],
    eval_frames: np.ndarray[Any, Any],
    physical: np.ndarray[Any, Any],
    indices: np.ndarray[Any, Any],
    *,
    selected: np.ndarray[Any, Any],
    ridge: float,
    shrinkage: float,
) -> tuple[np.ndarray[Any, Any], float, int]:
    started = time.perf_counter()
    model = fit_linear_residual(
        train_features,
        train_residual_canonical,
        indices,
        selected_features=selected,
        ridge=ridge,
    )
    correction = predict_linear_residual(model, eval_features)
    candidate = candidate_from_canonical(
        physical, correction, eval_frames, shrinkage=shrinkage
    )
    elapsed = time.perf_counter() - started
    parameter_count = int(model.coefficients.size)
    return candidate, elapsed, parameter_count


def run_dlo(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.protocol)
    parent = _mapping(protocol["parent"], "parent")
    runtime = _mapping(protocol["runtime"], "runtime")
    adapter = _mapping(protocol["adapter"], "adapter")
    arms = _mapping(protocol["arms"], "arms")
    evaluation = _mapping(protocol["evaluation"], "evaluation")
    dlo = str(args.dlo)
    if dlo not in DLOS:
        raise ValueError(f"unsupported DLO: {dlo}")
    if os.environ.get("RUNNER_NAME") not in (None, runtime["registered_runner_name"]):
        raise RuntimeError("unexpected self-hosted runner")
    parent_root = Path(str(parent["run_root"])).resolve()
    target_root = parent_root / f"{dlo.lower()}-target"
    dataset_root = args.dataset_root.resolve()
    upstream_root = args.upstream_root.resolve()
    if not (parent_root / "joint" / "joint_prediction_seal.json").is_file():
        raise FileNotFoundError("parent joint prediction seal is unavailable")
    parent_predictions = _load_parent_predictions(target_root)
    names = [str(value) for value in parent_predictions["names"].tolist()]
    physical = np.asarray(parent_predictions["physical"], dtype=np.float64)
    compute_matched = np.asarray(
        parent_predictions["compute_matched_physical"], dtype=np.float64
    )
    registered = np.asarray(parent_predictions["candidate"], dtype=np.float64)
    eval_names, eval_trajectories = _load_trajectories(
        dataset_root, dlo, "eval", names=names
    )
    if eval_names != names:
        raise ValueError("parent target order differs")
    eval_full = np.stack([eval_trajectories[name] for name in names])
    target = np.asarray(eval_full[:, 2:], dtype=np.float64)
    eval_initial, eval_action = deform_causal_inputs(eval_full)
    parent_model = _load_parent_model(target_root)
    reproduced = predict_deform_local_residual(
        parent_model,
        eval_initial,
        eval_action,
        physical,
        shrinkage=float(adapter["shrinkage"]),
    )["predictions"]
    reproduction_max_abs = float(np.max(np.abs(reproduced - registered)))
    if reproduction_max_abs > float(
        evaluation["full_adapter_parent_prediction_tolerance_m"]
    ):
        raise RuntimeError(
            f"parent full-adapter prediction did not reproduce: {reproduction_max_abs}"
        )

    train_names, train_trajectories, train_physical, physical_runtime = (
        _load_train_physical_rollout(
            protocol,
            dlo=dlo,
            target_root=target_root,
            dataset_root=dataset_root,
            upstream_root=upstream_root,
            device=args.device,
        )
    )
    train_full = np.stack([train_trajectories[name] for name in train_names])
    train_target = np.asarray(train_full[:, 2:], dtype=np.float64)
    train_initial, train_action = deform_causal_inputs(train_full)

    all_rows: list[dict[str, Any]] = []
    reference = {
        "registered_full_adapter_vs_physical": _score_and_record(
            registered,
            physical,
            target,
            names,
            dlo=dlo,
            arm="registered_full_adapter",
            all_rows=all_rows,
        ),
        "compute_matched_vs_physical": _score_and_record(
            compute_matched,
            physical,
            target,
            names,
            dlo=dlo,
            arm="compute_matched_physical",
            all_rows=all_rows,
        ),
        "registered_full_adapter_vs_compute_matched": score_arm(
            registered, compute_matched, target, names
        ),
    }

    build_started = time.perf_counter()
    train_features, train_frames = build_deform_local_residual_features(
        train_initial,
        train_action,
        train_physical,
        coordinate_frame="initial-action-local",
    )
    eval_features, eval_frames = build_deform_local_residual_features(
        eval_initial,
        eval_action,
        physical,
        coordinate_frame="initial-action-local",
    )
    local_feature_seconds = time.perf_counter() - build_started
    residual_global = train_target - train_physical
    residual_canonical = np.einsum(
        "ntvi,nij->ntvj", residual_global, train_frames
    )[:, :, 2:-2]
    ridge = float(adapter["ridge"])
    shrinkage = float(adapter["shrinkage"])
    all_indices = np.arange(len(train_names), dtype=np.int64)

    trivial: dict[str, Any] = {}
    correction = node_constant_bias(
        residual_canonical, physical.shape[0], physical.shape[1]
    )
    candidate = candidate_from_canonical(
        physical, correction, eval_frames, shrinkage=shrinkage
    )
    trivial["node_constant_bias"] = _score_and_record(
        candidate,
        physical,
        target,
        names,
        dlo=dlo,
        arm="node_constant_bias",
        all_rows=all_rows,
    )
    correction = time_node_mean_residual(residual_canonical, physical.shape[0])
    candidate = candidate_from_canonical(
        physical, correction, eval_frames, shrinkage=shrinkage
    )
    trivial["time_node_mean_residual"] = _score_and_record(
        candidate,
        physical,
        target,
        names,
        dlo=dlo,
        arm="time_node_mean_residual",
        all_rows=all_rows,
    )
    candidate, elapsed, parameter_count = _fit_predict(
        train_features,
        residual_canonical,
        eval_features,
        eval_frames,
        physical,
        all_indices,
        selected=feature_indices("time_only_ridge"),
        ridge=ridge,
        shrinkage=shrinkage,
    )
    time_only_summary = _score_and_record(
        candidate,
        physical,
        target,
        names,
        dlo=dlo,
        arm="time_only_ridge",
        all_rows=all_rows,
    )
    time_only_summary["fit_predict_seconds"] = elapsed
    time_only_summary["parameter_count"] = parameter_count
    trivial["time_only_ridge"] = time_only_summary

    structural: dict[str, Any] = {}
    for arm in ("no_explicit_action_features", "no_baseline_dynamics_features"):
        candidate, elapsed, parameter_count = _fit_predict(
            train_features,
            residual_canonical,
            eval_features,
            eval_frames,
            physical,
            all_indices,
            selected=feature_indices(arm),
            ridge=ridge,
            shrinkage=shrinkage,
        )
        summary = _score_and_record(
            candidate,
            physical,
            target,
            names,
            dlo=dlo,
            arm=arm,
            all_rows=all_rows,
        )
        summary["fit_predict_seconds"] = elapsed
        summary["parameter_count"] = parameter_count
        structural[arm] = summary

    curve_records: list[dict[str, Any]] = []
    curve_fit_seconds = 0.0
    sizes = tuple(int(value) for value in arms["source_data_sizes"])
    repeats = int(arms["source_data_repeats"])
    domain = str(arms["source_subset_domain"])
    full_features = feature_indices("full")
    for size in sizes:
        repeat_values = (0,) if size == len(train_names) else range(repeats)
        for repeat in repeat_values:
            indices = deterministic_subset_indices(
                train_names,
                dlo=dlo,
                repeat=repeat,
                count=size,
                domain=domain,
            )
            candidate, elapsed, parameter_count = _fit_predict(
                train_features,
                residual_canonical,
                eval_features,
                eval_frames,
                physical,
                indices,
                selected=full_features,
                ridge=ridge,
                shrinkage=shrinkage,
            )
            curve_fit_seconds += elapsed
            summary = _score_and_record(
                candidate,
                physical,
                target,
                names,
                dlo=dlo,
                arm="source_data_curve",
                repeat=repeat,
                source_count=size,
                all_rows=all_rows,
            )
            curve_records.append(
                {
                    "source_count": size,
                    "repeat": repeat,
                    "source_names": [train_names[index] for index in indices],
                    "candidate_mean_l1_m": summary["candidate_mean_l1_m"],
                    "baseline_mean_l1_m": summary["baseline_mean_l1_m"],
                    "relative_improvement": summary["relative_improvement"],
                    "wins": summary["wins"],
                    "losses": summary["losses"],
                    "worst_candidate_to_baseline_ratio": summary[
                        "worst_candidate_to_baseline_ratio"
                    ],
                    "fit_predict_seconds": elapsed,
                    "parameter_count": parameter_count,
                }
            )

    del train_features, eval_features
    gc.collect()
    global_started = time.perf_counter()
    global_train_features, global_train_frames = build_deform_local_residual_features(
        train_initial,
        train_action,
        train_physical,
        coordinate_frame="action-centered-global",
    )
    global_eval_features, global_eval_frames = build_deform_local_residual_features(
        eval_initial,
        eval_action,
        physical,
        coordinate_frame="action-centered-global",
    )
    global_residual = np.einsum(
        "ntvi,nij->ntvj", residual_global, global_train_frames
    )[:, :, 2:-2]
    candidate, elapsed, parameter_count = _fit_predict(
        global_train_features,
        global_residual,
        global_eval_features,
        global_eval_frames,
        physical,
        all_indices,
        selected=full_features,
        ridge=ridge,
        shrinkage=shrinkage,
    )
    global_summary = _score_and_record(
        candidate,
        physical,
        target,
        names,
        dlo=dlo,
        arm="global_coordinate_frame",
        all_rows=all_rows,
    )
    global_summary["fit_predict_seconds"] = elapsed
    global_summary["feature_build_seconds"] = time.perf_counter() - global_started
    global_summary["parameter_count"] = parameter_count
    structural["global_coordinate_frame"] = global_summary

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "trajectory_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "dlo",
            "arm",
            "repeat",
            "source_count",
            "trajectory",
            "candidate_l1_mm",
            "physical_l1_mm",
            "candidate_to_physical_ratio",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    result = {
        "schema_version": 1,
        "contract": CONTRACT,
        "status": "completed-dlo",
        "dlo": dlo,
        "evidence_class": protocol["evidence_class"],
        "protocol": _identity(args.protocol),
        "parent": {
            "workflow_run_id": parent["workflow_run_id"],
            "head_sha": parent["head_sha"],
            "run_root": str(parent_root),
            "target_prediction_seal": _identity(target_root / "prediction_seal.json"),
            "joint_prediction_seal": _identity(
                parent_root / "joint" / "joint_prediction_seal.json"
            ),
            "parent_prediction_reproduction_max_abs_m": reproduction_max_abs,
        },
        "accounting": {
            "train_trajectories": len(train_names),
            "target_trajectories": len(names),
            "future_frames": target.shape[1],
            "new_data_collected": False,
            "backbone_training_updates": 0,
            "target_outcomes_already_open_before_registration": True,
            "target_outcomes_used_for_source_subset_selection": False,
            "raw_prediction_arrays_uploaded": False,
        },
        "reference": reference,
        "trivial_controls": trivial,
        "structural_ablations": structural,
        "source_data_curve": {
            "records": curve_records,
            "summary": summarize_repeated_curve(curve_records, sizes),
        },
        "runtime": {
            **physical_runtime,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "local_feature_build_seconds": local_feature_seconds,
            "source_curve_fit_predict_seconds": curve_fit_seconds,
            "registered_full_parameter_count": int(
                np.asarray(parent_model["coefficients"]).size
            ),
        },
        "primary_gate": {
            "parent_prediction_reproduced": reproduction_max_abs
            <= float(evaluation["full_adapter_parent_prediction_tolerance_m"]),
            "registered_full_adapter_beats_physical": reference[
                "registered_full_adapter_vs_physical"
            ]["relative_improvement"]
            > 0.0,
            "registered_full_adapter_beats_compute_matched": reference[
                "registered_full_adapter_vs_compute_matched"
            ]["relative_improvement"]
            > 0.0,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    result["primary_gate"]["passed"] = all(result["primary_gate"].values())
    _write_json(output / "result.json", result)
    print(
        json.dumps(
            {
                "dlo": dlo,
                "result": str(output / "result.json"),
                "primary_gate": result["primary_gate"],
                "registered_full_adapter": reference[
                    "registered_full_adapter_vs_physical"
                ],
                "compute_matched": reference["compute_matched_vs_physical"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _equal_dlo_arm(results: Mapping[str, Mapping[str, Any]], path: Sequence[str]) -> dict[str, Any]:
    summaries = []
    for dlo in DLOS:
        value: Any = results[dlo]
        for key in path:
            value = value[key]
        summaries.append(value)
    candidate = float(np.mean([row["candidate_mean_l1_m"] for row in summaries]))
    baseline = float(np.mean([row["baseline_mean_l1_m"] for row in summaries]))
    return {
        "candidate_mean_l1_m": candidate,
        "baseline_mean_l1_m": baseline,
        "relative_improvement": 1.0 - candidate / baseline,
        "wins": int(sum(int(row["wins"]) for row in summaries)),
        "ties": int(sum(int(row["ties"]) for row in summaries)),
        "losses": int(sum(int(row["losses"]) for row in summaries)),
    }


def aggregate(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.protocol)
    input_root = args.input_root.resolve()
    results = {
        dlo: _read_json(input_root / dlo.lower() / "result.json") for dlo in DLOS
    }
    for dlo, result in results.items():
        if (
            result.get("contract") != CONTRACT
            or result.get("status") != "completed-dlo"
            or result.get("dlo") != dlo
        ):
            raise ValueError(f"invalid {dlo} result")
    reference = {
        "registered_full_adapter_vs_physical": _equal_dlo_arm(
            results, ("reference", "registered_full_adapter_vs_physical")
        ),
        "compute_matched_vs_physical": _equal_dlo_arm(
            results, ("reference", "compute_matched_vs_physical")
        ),
        "registered_full_adapter_vs_compute_matched": _equal_dlo_arm(
            results, ("reference", "registered_full_adapter_vs_compute_matched")
        ),
    }
    controls: dict[str, Any] = {}
    for group, key in (
        ("trivial_controls", "trivial_controls"),
        ("structural_ablations", "structural_ablations"),
    ):
        arm_names = set(results["DLO4"][key])
        if arm_names != set(results["DLO5"][key]):
            raise ValueError(f"{group} arm sets differ")
        controls[group] = {
            arm: _equal_dlo_arm(results, (key, arm)) for arm in sorted(arm_names)
        }
    candidate_by_dlo = {
        dlo: results[dlo]["reference"]["registered_full_adapter_vs_physical"][
            "candidate_case_l1_m"
        ]
        for dlo in DLOS
    }
    physical_by_dlo = {
        dlo: results[dlo]["reference"]["registered_full_adapter_vs_physical"][
            "baseline_case_l1_m"
        ]
        for dlo in DLOS
    }
    compute_by_dlo = {
        dlo: results[dlo]["reference"]["registered_full_adapter_vs_compute_matched"][
            "baseline_case_l1_m"
        ]
        for dlo in DLOS
    }
    evaluation = _mapping(protocol["evaluation"], "evaluation")
    bootstrap = {
        "full_vs_physical": equal_dlo_bootstrap(
            candidate_by_dlo,
            physical_by_dlo,
            replicates=int(evaluation["bootstrap_replicates"]),
            seed=int(evaluation["bootstrap_seed"]),
        ),
        "full_vs_compute_matched": equal_dlo_bootstrap(
            candidate_by_dlo,
            compute_by_dlo,
            replicates=int(evaluation["bootstrap_replicates"]),
            seed=int(evaluation["bootstrap_seed"]) + 1,
        ),
    }
    source_curve: dict[str, Any] = {}
    sizes = tuple(int(value) for value in protocol["arms"]["source_data_sizes"])
    for size in sizes:
        records = [
            row
            for dlo in DLOS
            for row in results[dlo]["source_data_curve"]["records"]
            if int(row["source_count"]) == size
        ]
        by_repeat: dict[int, list[Mapping[str, Any]]] = {}
        for row in records:
            by_repeat.setdefault(int(row["repeat"]), []).append(row)
        equal_dlo = []
        for repeat, rows in sorted(by_repeat.items()):
            if len(rows) != 2:
                raise ValueError(f"size {size}, repeat {repeat} lacks one DLO")
            candidate = float(np.mean([row["candidate_mean_l1_m"] for row in rows]))
            baseline = float(np.mean([row["baseline_mean_l1_m"] for row in rows]))
            equal_dlo.append(
                {
                    "repeat": repeat,
                    "candidate_mean_l1_m": candidate,
                    "baseline_mean_l1_m": baseline,
                    "relative_improvement": 1.0 - candidate / baseline,
                }
            )
        source_curve[str(size)] = {
            "repeat_count": len(equal_dlo),
            "mean_candidate_l1_m": float(
                np.mean([row["candidate_mean_l1_m"] for row in equal_dlo])
            ),
            "mean_relative_improvement": float(
                np.mean([row["relative_improvement"] for row in equal_dlo])
            ),
            "minimum_relative_improvement": float(
                np.min([row["relative_improvement"] for row in equal_dlo])
            ),
            "maximum_relative_improvement": float(
                np.max([row["relative_improvement"] for row in equal_dlo])
            ),
            "improving_repeats": int(
                np.count_nonzero(
                    [row["relative_improvement"] > 0.0 for row in equal_dlo]
                )
            ),
            "records": equal_dlo,
        }
    per_dlo_gate = {
        dlo: results[dlo]["primary_gate"] for dlo in DLOS
    }
    primary_gate = {
        "parent_prediction_reproduced_each_dlo": all(
            bool(per_dlo_gate[dlo]["parent_prediction_reproduced"]) for dlo in DLOS
        ),
        "registered_full_adapter_beats_physical_each_dlo": all(
            bool(per_dlo_gate[dlo]["registered_full_adapter_beats_physical"])
            for dlo in DLOS
        ),
        "registered_full_adapter_beats_compute_matched_each_dlo": all(
            bool(per_dlo_gate[dlo]["registered_full_adapter_beats_compute_matched"])
            for dlo in DLOS
        ),
    }
    primary_gate["passed"] = all(primary_gate.values())
    best_trivial = min(
        controls["trivial_controls"].items(),
        key=lambda item: item[1]["candidate_mean_l1_m"],
    )
    best_structural = min(
        controls["structural_ablations"].items(),
        key=lambda item: item[1]["candidate_mean_l1_m"],
    )
    result = {
        "schema_version": 1,
        "contract": CONTRACT,
        "status": "completed-aggregate",
        "evidence_class": protocol["evidence_class"],
        "protocol": _identity(args.protocol),
        "dlo_results": {
            dlo: _identity(input_root / dlo.lower() / "result.json") for dlo in DLOS
        },
        "reference": reference,
        **controls,
        "source_data_curve": source_curve,
        "bootstrap": bootstrap,
        "primary_gate": primary_gate,
        "comparative_findings": {
            "best_trivial_arm": best_trivial[0],
            "best_trivial_equal_dlo_l1_m": best_trivial[1]["candidate_mean_l1_m"],
            "full_minus_best_trivial_l1_m": reference[
                "registered_full_adapter_vs_physical"
            ]["candidate_mean_l1_m"]
            - best_trivial[1]["candidate_mean_l1_m"],
            "best_structural_arm": best_structural[0],
            "best_structural_equal_dlo_l1_m": best_structural[1][
                "candidate_mean_l1_m"
            ],
            "full_minus_best_structural_l1_m": reference[
                "registered_full_adapter_vs_physical"
            ]["candidate_mean_l1_m"]
            - best_structural[1]["candidate_mean_l1_m"],
        },
        "accounting": {
            "dlos": 2,
            "target_trajectories": 28,
            "new_data_collected": False,
            "backbone_training_updates": 0,
            "retrospective_post_open": True,
            "target_selected_source_subsets": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "result.json", result)
    combined_csv = output / "trajectory_results.csv"
    header_written = False
    with combined_csv.open("w", newline="", encoding="utf-8") as destination:
        for dlo in DLOS:
            with (input_root / dlo.lower() / "trajectory_results.csv").open(
                newline="", encoding="utf-8"
            ) as source:
                reader = csv.DictReader(source)
                writer = csv.DictWriter(destination, fieldnames=cast(list[str], reader.fieldnames))
                if not header_written:
                    writer.writeheader()
                    header_written = True
                writer.writerows(reader)
    lines = [
        "# DEFORM post-hoc adapter controls v1",
        "",
        "This is a retrospective post-open control study on public DLO4/DLO5 data.",
        "No new data were collected and no DEFORM backbone update was executed.",
        "",
        "## Main comparison",
        "",
        "| Panel | Physical (mm) | Compute matched (mm) | Full adapter (mm) | Gain vs physical | Gain vs compute |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dlo in DLOS:
        full = results[dlo]["reference"]["registered_full_adapter_vs_physical"]
        compute = results[dlo]["reference"]["compute_matched_vs_physical"]
        full_compute = results[dlo]["reference"][
            "registered_full_adapter_vs_compute_matched"
        ]
        lines.append(
            f"| {dlo} | {1000.0 * full['baseline_mean_l1_m']:.4f} | "
            f"{1000.0 * compute['candidate_mean_l1_m']:.4f} | "
            f"{1000.0 * full['candidate_mean_l1_m']:.4f} | "
            f"{100.0 * full['relative_improvement']:.2f}% | "
            f"{100.0 * full_compute['relative_improvement']:.2f}% |"
        )
    equal_full = reference["registered_full_adapter_vs_physical"]
    equal_compute = reference["compute_matched_vs_physical"]
    equal_full_compute = reference["registered_full_adapter_vs_compute_matched"]
    lines.append(
        f"| Equal-DLO | {1000.0 * equal_full['baseline_mean_l1_m']:.4f} | "
        f"{1000.0 * equal_compute['candidate_mean_l1_m']:.4f} | "
        f"{1000.0 * equal_full['candidate_mean_l1_m']:.4f} | "
        f"{100.0 * equal_full['relative_improvement']:.2f}% | "
        f"{100.0 * equal_full_compute['relative_improvement']:.2f}% |"
    )
    lines.extend(
        [
            "",
            f"Primary gate: **{'PASS' if primary_gate['passed'] else 'FAIL'}**.",
            "",
            "## Trivial and structural controls",
            "",
            "| Arm | Equal-DLO L1 (mm) | Gain vs physical | W/T/L |",
            "|---|---:|---:|---:|",
        ]
    )
    for group in ("trivial_controls", "structural_ablations"):
        for arm, row in controls[group].items():
            lines.append(
                f"| `{arm}` | {1000.0 * row['candidate_mean_l1_m']:.4f} | "
                f"{100.0 * row['relative_improvement']:.2f}% | "
                f"{row['wins']}/{row['ties']}/{row['losses']} |"
            )
    lines.extend(
        [
            "",
            "## Source-data curve",
            "",
            "| Source trajectories per DLO | Mean equal-DLO L1 (mm) | Mean gain | Improving repeats |",
            "|---:|---:|---:|---:|",
        ]
    )
    for size in sizes:
        row = source_curve[str(size)]
        lines.append(
            f"| {size} | {1000.0 * row['mean_candidate_l1_m']:.4f} | "
            f"{100.0 * row['mean_relative_improvement']:.2f}% | "
            f"{row['improving_repeats']}/{row['repeat_count']} |"
        )
    ci = bootstrap["full_vs_physical"]
    lines.extend(
        [
            "",
            "## Complete-trajectory uncertainty",
            "",
            f"Equal-DLO gain versus physical: {100.0 * ci['relative_improvement']:.2f}% "
            f"(stratified trajectory bootstrap 95% interval "
            f"[{100.0 * ci['bootstrap_low']:.2f}%, {100.0 * ci['bootstrap_high']:.2f}%]).",
            "",
            "The controls were fixed before this rerun, but DLO4/DLO5 outcomes had already been opened by the parent study. These results cannot be relabeled as fresh confirmation.",
        ]
    )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"primary_gate": primary_gate, "output": str(output)}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-dlo")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--dlo", choices=DLOS, required=True)
    run.add_argument("--dataset-root", type=Path, required=True)
    run.add_argument("--upstream-root", type=Path, required=True)
    run.add_argument("--device", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    combine = subparsers.add_parser("aggregate")
    combine.add_argument("--protocol", type=Path, required=True)
    combine.add_argument("--input-root", type=Path, required=True)
    combine.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "run-dlo":
        return run_dlo(args)
    if args.command == "aggregate":
        return aggregate(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
