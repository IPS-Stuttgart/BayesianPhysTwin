#!/usr/bin/env python3
"""Run the frozen Cloth Sim2Real online-belief stages.

Simulation and prediction sealing never read future point clouds. Scoring is a
separate command that verifies the prediction seal before opening outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.cloth_sim2real_belief import (
    ClothReadoutBeliefConfig,
    apply_guarded_readout_correction,
    associate_dense_cloud,
    directed_l1_chamfer_m,
    fit_guarded_readout_correction,
    load_binary_little_endian_ply_xyz,
    sample_physical_rollout,
    symmetric_l1_chamfer_m,
    symmetric_l2_hausdorff_m,
)

BENCHMARK_COMMIT = "178a9b9722191c51cf0dcbc3cf0dc03701b09eb3"
METHOD_ID = "cloth-sim2real-guarded-readout-v1"
RELEASED_DYNAMIC_CUTOFF = {
    "chequered_rag_0": 95,
    "chequered_rag_1": 95,
    "chequered_rag_2": 99,
    "cotton_rag_0": 99,
    "cotton_rag_1": 98,
    "cotton_rag_2": 99,
    "linen_rag_0": 90,
    "linen_rag_1": 89,
    "linen_rag_2": 89,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    _require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        payload.get("artifact_kind") == "ClothSim2RealDatasetManifest",
        "dataset manifest kind changed",
    )
    return payload


def _case_descriptor(
    manifest: dict[str, Any],
    case_id: str,
    authorized_split: str,
) -> dict[str, Any]:
    matches = [case for case in manifest["cases"] if case["case_id"] == case_id]
    _require(len(matches) == 1, f"manifest does not contain exactly one {case_id}")
    case = matches[0]
    _require(
        case["split"] == authorized_split,
        f"{case_id} is {case['split']}, not authorized {authorized_split}",
    )
    return case


def _authorization(
    split: str,
    artifact_path: Path | None,
) -> tuple[str | None, str | None]:
    if split == "source":
        _require(
            artifact_path is None,
            "source predictions do not accept an authorization artifact",
        )
        return None, None
    _require(
        artifact_path is not None and artifact_path.is_file(),
        f"{split} predictions require an authorization artifact",
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if split == "calibration":
        _require(
            payload.get("artifact_kind") == "ClothSim2RealSourceGate"
            and payload.get("calibration_authorized") is True,
            "source gate does not authorize calibration",
        )
    else:
        _require(
            payload.get("artifact_kind") == "ClothSim2RealCalibrationGate"
            and payload.get("target_authorized") is True,
            "calibration gate does not authorize target",
        )
    return str(artifact_path.resolve()), _sha256(artifact_path)


def _cloud_dir(dataset_root: Path, case: dict[str, Any]) -> Path:
    root = dataset_root
    if (root / "Benchmarking_cloth").is_dir():
        root = root / "Benchmarking_cloth"
    path = root / case["case_id"] / "cloud"
    _require(path.is_dir(), f"point-cloud directory does not exist: {path}")
    return path


def _load_clouds(
    cloud_dir: Path,
    start: int,
    stop: int,
) -> list[np.ndarray]:
    _require(0 <= start < stop, "invalid cloud interval")
    return [
        load_binary_little_endian_ply_xyz(cloud_dir / f"{index:05d}.ply")
        for index in range(start, stop)
    ]


def _simulate(args: argparse.Namespace) -> int:
    benchmark_root = args.benchmark_code_root.resolve()
    _require((benchmark_root / "bcm").is_dir(), "benchmark code root has no bcm")
    git_head = subprocess.check_output(
        ["git", "-C", str(benchmark_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    _require(git_head == BENCHMARK_COMMIT, "benchmark checkout commit changed")
    sys.path.insert(0, str(benchmark_root))
    OmegaConf = importlib.import_module("omegaconf").OmegaConf
    get_env = importlib.import_module("bcm.envs").get_env
    generate_full_trajectory = importlib.import_module(
        "bcm.manipulation_utils"
    ).generate_full_trajectory

    cloth_sample = args.case_id.split("/", maxsplit=1)[0]
    task = args.case_id.split("/", maxsplit=1)[1]
    _require(task in {"dynamic", "quasi_static"}, "unknown task")
    full = OmegaConf.load(benchmark_root / "bcm/conf/envs/mujoco3.yaml")
    parameter_name = f"params_{task}_{cloth_sample}"
    _require(hasattr(full, parameter_name), f"missing parameters {parameter_name}")
    environment_config = OmegaConf.create(
        {
            "name": "mujoco3",
            "render_mode": "None",
            "depth": False,
            "width": 320,
            "height": 288,
            "params": OmegaConf.to_container(
                getattr(full, parameter_name),
                resolve=True,
            ),
        }
    )
    real_setup = {
        "table": {
            "xmin": -0.4,
            "xmax": 0.4,
            "ymin": -0.1,
            "ymax": 0.8,
            "zmax": 0.195,
        },
        "gripper_start": {
            "left": [0.0, 0.0, 1.0],
            "right": [0.5, 0.0, 1.0],
        },
    }
    environment = get_env(
        environment_config,
        real_setup=real_setup,
        target=None,
    )
    _, info = environment.reset()
    dt_s = float(environment.unwrapped.trajectory_dt)
    stabilization_steps = int(1.0 / dt_s)
    trajectory, pretrajectory_steps = generate_full_trajectory(
        dt_s,
        cloth_sample,
        "unused",
        stabilization_steps,
        task,
        "mujoco3",
    )
    vertices: list[np.ndarray] = []
    for index, action in enumerate(trajectory):
        _, _, _, _, info = environment.step(action)
        if index >= pretrajectory_steps:
            vertices.append(
                np.asarray(info["vertices"], dtype=np.float64).copy()
            )
    faces = np.asarray(info["faces"], dtype=np.int64).copy()
    environment.close()

    output = args.output.resolve()
    _require(not output.exists(), f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        vertices_m=np.stack(vertices),
        faces=faces,
        actions_m=np.asarray(
            trajectory[pretrajectory_steps:],
            dtype=np.float64,
        ),
        dt_s=np.asarray(dt_s),
    )
    metadata_path = output.with_suffix(".json")
    _write_json_once(
        metadata_path,
        {
            "schema_version": 1,
            "artifact_kind": "ClothSim2RealPhysicalBaseline",
            "case_id": args.case_id,
            "benchmark_commit": BENCHMARK_COMMIT,
            "mujoco_version": importlib.import_module("mujoco").__version__,
            "pretrajectory_steps": pretrajectory_steps,
            "simulator_frame_count": len(vertices),
            "node_count": int(vertices[0].shape[0]),
            "face_count": int(len(faces)),
            "dt_s": dt_s,
            "npz_sha256": _sha256(output),
            "point_cloud_coordinates_read": False,
            "future_outcomes_read": False,
        },
    )
    return 0


def _score_descriptor(score: Any) -> dict[str, Any]:
    return asdict(score)


def _seal(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    case = _case_descriptor(manifest, args.case_id, args.authorized_split)
    authorization_path, authorization_sha256 = _authorization(
        args.authorized_split,
        (
            None
            if args.authorization_artifact is None
            else args.authorization_artifact.resolve()
        ),
    )
    baseline_path = args.baseline.resolve()
    _require(baseline_path.is_file(), "physical baseline does not exist")
    with np.load(baseline_path, allow_pickle=False) as baseline:
        sampled, sampled_indices = sample_physical_rollout(
            baseline["vertices_m"],
            int(case["frame_count"]),
        )
        faces = np.asarray(baseline["faces"], dtype=np.int64)
    fit_stop = int(case["fit_stop_frame"])
    branch = int(case["branch_frame"])
    cloud_dir = _cloud_dir(args.dataset_root.resolve(), case)
    prefix_clouds = _load_clouds(cloud_dir, 0, branch + 1)
    config = ClothReadoutBeliefConfig()
    belief = fit_guarded_readout_correction(
        sampled[:fit_stop],
        prefix_clouds[:fit_stop],
        sampled[fit_stop : branch + 1],
        prefix_clouds[fit_stop : branch + 1],
        faces,
        config=config,
    )
    physical_future = sampled[branch + 1 :]
    candidate_future = apply_guarded_readout_correction(
        physical_future,
        belief,
    )

    output = args.output_dir.resolve()
    _require(not output.exists(), f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    physical_path = output / "physical_future.npy"
    candidate_path = output / "candidate_future.npy"
    belief_path = output / "readout_belief.npz"
    np.save(physical_path, physical_future, allow_pickle=False)
    np.save(candidate_path, candidate_future, allow_pickle=False)
    np.savez_compressed(
        belief_path,
        accepted=np.asarray(belief.accepted),
        selected_name=np.asarray(belief.selected_name),
        correction_m=belief.correction_m,
        variance_m2=belief.variance_m2,
        sampled_simulator_indices=sampled_indices,
    )
    exact_fallback = (
        not belief.accepted
        and np.array_equal(candidate_future, physical_future)
    )
    _require(
        belief.accepted or exact_fallback,
        "rejected candidate is not an exact physical fallback",
    )
    seal = {
        "schema_version": 1,
        "artifact_kind": "ClothSim2RealPredictionSeal",
        "method_id": METHOD_ID,
        "case_id": args.case_id,
        "authorized_split": args.authorized_split,
        "authorization_artifact": authorization_path,
        "authorization_artifact_sha256": authorization_sha256,
        "manifest_sha256": _sha256(manifest_path),
        "baseline_sha256": _sha256(baseline_path),
        "fit_frames": [0, fit_stop - 1],
        "validation_frames": [fit_stop, branch],
        "future_frames": [branch + 1, int(case["frame_count"]) - 1],
        "selected_name": belief.selected_name,
        "accepted": belief.accepted,
        "reason": belief.reason,
        "exact_fallback": exact_fallback,
        "config": asdict(config),
        "scores": [_score_descriptor(score) for score in belief.scores],
        "diagnostics": belief.diagnostics,
        "physical_future_sha256": _sha256(physical_path),
        "candidate_future_sha256": _sha256(candidate_path),
        "belief_sha256": _sha256(belief_path),
        "prefix_point_clouds_read": True,
        "future_point_clouds_read": False,
    }
    _write_json_once(output / "prediction_seal.json", seal)
    return 0


def _normal_energy_score(
    observation_m: np.ndarray,
    prediction_m: np.ndarray,
    variance_m2: np.ndarray,
    *,
    seed: int,
    sample_count: int = 64,
) -> float:
    rng = np.random.default_rng(seed)
    standard_deviation = np.sqrt(variance_m2)
    first = prediction_m[None] + rng.normal(
        size=(sample_count,) + prediction_m.shape
    ) * standard_deviation[None]
    second = prediction_m[None] + rng.normal(
        size=(sample_count,) + prediction_m.shape
    ) * standard_deviation[None]
    first_term = np.mean(
        np.linalg.norm(first - observation_m[None], axis=2)
    )
    second_term = np.mean(np.linalg.norm(first - second, axis=2))
    return float(first_term - 0.5 * second_term)


def _score(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    seal_path = output / "prediction_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _require(
        seal.get("artifact_kind") == "ClothSim2RealPredictionSeal",
        "prediction seal kind changed",
    )
    physical_path = output / "physical_future.npy"
    candidate_path = output / "candidate_future.npy"
    belief_path = output / "readout_belief.npz"
    _require(
        _sha256(physical_path) == seal["physical_future_sha256"],
        "physical prediction changed after sealing",
    )
    _require(
        _sha256(candidate_path) == seal["candidate_future_sha256"],
        "candidate prediction changed after sealing",
    )
    _require(
        _sha256(belief_path) == seal["belief_sha256"],
        "belief changed after sealing",
    )
    manifest = _load_manifest(args.manifest.resolve())
    case = _case_descriptor(manifest, seal["case_id"], seal["authorized_split"])
    calibration_std_multiplier = 1.0
    calibration_artifact_sha256 = None
    if seal["authorized_split"] == "target":
        _require(
            args.calibration_artifact is not None
            and args.calibration_artifact.is_file(),
            "target scoring requires the calibration artifact",
        )
        calibration = json.loads(
            args.calibration_artifact.read_text(encoding="utf-8")
        )
        calibration_artifact_sha256 = _sha256(args.calibration_artifact)
        _require(
            calibration.get("artifact_kind") == "ClothSim2RealCalibrationGate"
            and calibration.get("target_authorized") is True,
            "calibration artifact does not authorize target scoring",
        )
        _require(
            calibration_artifact_sha256
            == seal["authorization_artifact_sha256"],
            "target seal and calibration artifact differ",
        )
        calibration_std_multiplier = float(
            calibration["uncertainty_std_multiplier"]
        )
        _require(
            np.isfinite(calibration_std_multiplier)
            and calibration_std_multiplier >= 1.0,
            "calibration uncertainty multiplier is invalid",
        )
    branch = int(case["branch_frame"])
    cloud_dir = _cloud_dir(args.dataset_root.resolve(), case)
    observed_future = _load_clouds(
        cloud_dir,
        branch + 1,
        int(case["frame_count"]),
    )
    physical = np.load(physical_path, allow_pickle=False)
    candidate = np.load(candidate_path, allow_pickle=False)
    with np.load(belief_path, allow_pickle=False) as belief:
        variance = np.asarray(belief["variance_m2"], dtype=np.float64)
        correction = np.asarray(belief["correction_m"], dtype=np.float64)
    _require(
        len(physical) == len(candidate) == len(observed_future),
        "future lengths differ",
    )
    physical_symmetric = np.asarray(
        [
            symmetric_l1_chamfer_m(prediction, observation)
            for prediction, observation in zip(
                physical,
                observed_future,
                strict=True,
            )
        ]
    )
    candidate_symmetric = np.asarray(
        [
            symmetric_l1_chamfer_m(prediction, observation)
            for prediction, observation in zip(
                candidate,
                observed_future,
                strict=True,
            )
        ]
    )
    physical_directed = np.asarray(
        [
            directed_l1_chamfer_m(prediction, observation)
            for prediction, observation in zip(
                physical,
                observed_future,
                strict=True,
            )
        ]
    )
    candidate_directed = np.asarray(
        [
            directed_l1_chamfer_m(prediction, observation)
            for prediction, observation in zip(
                candidate,
                observed_future,
                strict=True,
            )
        ]
    )
    physical_hausdorff = np.asarray(
        [
            symmetric_l2_hausdorff_m(prediction, observation)
            for prediction, observation in zip(
                physical,
                observed_future,
                strict=True,
            )
        ]
    )
    candidate_hausdorff = np.asarray(
        [
            symmetric_l2_hausdorff_m(prediction, observation)
            for prediction, observation in zip(
                candidate,
                observed_future,
                strict=True,
            )
        ]
    )
    config = ClothReadoutBeliefConfig(**seal["config"])
    coverage_values: list[float] = []
    raw_coverage_values: list[float] = []
    interval_widths: list[float] = []
    energy_scores: list[float] = []
    standardized_absolute_residuals: list[np.ndarray] = []
    for horizon, (prediction, observation) in enumerate(
        zip(candidate, observed_future, strict=True),
        start=1,
    ):
        association = associate_dense_cloud(
            prediction,
            observation,
            candidate_count=config.candidate_count,
            sensor_std_m=config.sensor_std_m,
        )
        raw_total_variance = (
            variance
            + horizon * config.forecast_process_std_m_per_sqrt_frame**2
            + association.variance_m2[:, None]
        )
        total_variance = (
            calibration_std_multiplier**2 * raw_total_variance
        )
        residual = association.observed_points_m - prediction
        standardized_absolute_residuals.append(
            np.abs(residual) / np.sqrt(raw_total_variance)
        )
        half_width = 1.6448536269514722 * np.sqrt(total_variance)
        raw_half_width = 1.6448536269514722 * np.sqrt(raw_total_variance)
        coverage_values.append(float(np.mean(np.abs(residual) <= half_width)))
        raw_coverage_values.append(
            float(np.mean(np.abs(residual) <= raw_half_width))
        )
        interval_widths.append(float(np.mean(2.0 * half_width)))
        energy_scores.append(
            _normal_energy_score(
                association.observed_points_m,
                prediction,
                total_variance,
                seed=config.covariance_seed + horizon,
            )
        )
    horizon_indices = np.array_split(np.arange(len(physical_symmetric)), 3)
    standardized_q90 = float(
        np.quantile(
            np.concatenate(
                [
                    values.reshape(-1)
                    for values in standardized_absolute_residuals
                ]
            ),
            0.90,
        )
    )
    trial_id, task = seal["case_id"].split("/", maxsplit=1)
    absolute_future_frames = np.arange(branch + 1, int(case["frame_count"]))
    if task == "dynamic":
        released_cutoff = RELEASED_DYNAMIC_CUTOFF[trial_id]
        released_window = absolute_future_frames < released_cutoff
    else:
        released_cutoff = int(case["frame_count"])
        released_window = np.ones(len(absolute_future_frames), dtype=bool)
    _require(np.any(released_window), "released comparison window is empty")

    def relative_improvement(
        baseline: np.ndarray,
        corrected: np.ndarray,
    ) -> float:
        return float(
            (np.mean(baseline) - np.mean(corrected))
            / max(float(np.mean(baseline)), 1e-15)
        )

    result = {
        "schema_version": 1,
        "artifact_kind": "ClothSim2RealPredictionResult",
        "method_id": METHOD_ID,
        "case_id": seal["case_id"],
        "authorized_split": seal["authorized_split"],
        "calibration_artifact_sha256": calibration_artifact_sha256,
        "uncertainty_std_multiplier": calibration_std_multiplier,
        "prediction_seal_sha256": _sha256(seal_path),
        "selected_name": seal["selected_name"],
        "accepted": seal["accepted"],
        "future_frame_count": len(physical),
        "metrics": {
            "physical_symmetric_l1_chamfer_m": float(
                np.mean(physical_symmetric)
            ),
            "candidate_symmetric_l1_chamfer_m": float(
                np.mean(candidate_symmetric)
            ),
            "symmetric_relative_improvement": relative_improvement(
                physical_symmetric,
                candidate_symmetric,
            ),
            "symmetric_frame_wins": int(
                np.sum(candidate_symmetric < physical_symmetric)
            ),
            "physical_directed_l1_chamfer_m": float(
                np.mean(physical_directed)
            ),
            "candidate_directed_l1_chamfer_m": float(
                np.mean(candidate_directed)
            ),
            "directed_relative_improvement": relative_improvement(
                physical_directed,
                candidate_directed,
            ),
            "released_window_cutoff_frame_exclusive": released_cutoff,
            "released_window_physical_directed_l1_chamfer_m": float(
                np.mean(physical_directed[released_window])
            ),
            "released_window_candidate_directed_l1_chamfer_m": float(
                np.mean(candidate_directed[released_window])
            ),
            "released_window_directed_relative_improvement": (
                relative_improvement(
                    physical_directed[released_window],
                    candidate_directed[released_window],
                )
            ),
            "physical_symmetric_l2_hausdorff_m": float(
                np.mean(physical_hausdorff)
            ),
            "candidate_symmetric_l2_hausdorff_m": float(
                np.mean(candidate_hausdorff)
            ),
            "hausdorff_relative_improvement": relative_improvement(
                physical_hausdorff,
                candidate_hausdorff,
            ),
            "raw_90_coordinate_coverage": float(
                np.mean(raw_coverage_values)
            ),
            "reported_90_coordinate_coverage": float(
                np.mean(coverage_values)
            ),
            "trial_coordinate_abs_standardized_q90": standardized_q90,
            "mean_90_interval_width_m": float(np.mean(interval_widths)),
            "mean_energy_score_m": float(np.mean(energy_scores)),
            "mean_readout_correction_m": float(
                np.mean(np.linalg.norm(correction, axis=1))
            ),
        },
        "horizons": [
            {
                "name": name,
                "physical_symmetric_l1_chamfer_m": float(
                    np.mean(physical_symmetric[indices])
                ),
                "candidate_symmetric_l1_chamfer_m": float(
                    np.mean(candidate_symmetric[indices])
                ),
                "relative_improvement": relative_improvement(
                    physical_symmetric[indices],
                    candidate_symmetric[indices],
                ),
                "reported_90_coordinate_coverage": float(
                    np.mean(np.asarray(coverage_values)[indices])
                ),
                "raw_90_coordinate_coverage": float(
                    np.mean(np.asarray(raw_coverage_values)[indices])
                ),
            }
            for name, indices in zip(
                ("early", "middle", "late"),
                horizon_indices,
                strict=True,
            )
        ],
        "future_outcomes_read_only_after_prediction_seal": True,
        "claim_boundary": (
            "causal online continuation from a real prefix; not an identical-"
            "information open-loop benchmark comparison"
        ),
    }
    result_path = output / "result.json"
    _write_json_once(result_path, result)
    return 0


def _aggregate(args: argparse.Namespace) -> int:
    result_paths = tuple(sorted(args.results_root.glob("*/result.json")))
    _require(len(result_paths) == 6, "source aggregate requires exactly six cases")
    results = [
        json.loads(path.read_text(encoding="utf-8")) for path in result_paths
    ]
    _require(
        {result["authorized_split"] for result in results} == {"source"},
        "source aggregate contains another split",
    )
    dynamic = [
        result for result in results if result["case_id"].endswith("/dynamic")
    ]
    quasi_static = [
        result
        for result in results
        if result["case_id"].endswith("/quasi_static")
    ]
    _require(
        len(dynamic) == len(quasi_static) == 3,
        "source task counts changed",
    )
    dynamic_improvements = np.asarray(
        [
            result["metrics"]["symmetric_relative_improvement"]
            for result in dynamic
        ]
    )
    source_gate = bool(
        np.mean(dynamic_improvements) >= 0.05
        and np.all(dynamic_improvements >= 0.0)
    )
    payload = {
        "schema_version": 1,
        "artifact_kind": "ClothSim2RealSourceGate",
        "method_id": METHOD_ID,
        "result_sha256s": {
            path.parent.name: _sha256(path) for path in result_paths
        },
        "case_metrics": {
            result["case_id"]: result["metrics"] for result in results
        },
        "dynamic_object_balanced_relative_improvement": float(
            np.mean(dynamic_improvements)
        ),
        "dynamic_nonregressing_cloth_count": int(
            np.sum(dynamic_improvements >= 0.0)
        ),
        "quasi_static_object_balanced_relative_improvement": float(
            np.mean(
                [
                    result["metrics"]["symmetric_relative_improvement"]
                    for result in quasi_static
                ]
            )
        ),
        "source_gate_passed": source_gate,
        "calibration_authorized": source_gate,
        "target_authorized": False,
    }
    _write_json_once(args.output.resolve(), payload)
    return 0


def _aggregate_calibration(args: argparse.Namespace) -> int:
    source_gate_path = args.source_gate.resolve()
    source_gate = json.loads(source_gate_path.read_text(encoding="utf-8"))
    _require(
        source_gate.get("artifact_kind") == "ClothSim2RealSourceGate"
        and source_gate.get("calibration_authorized") is True,
        "source gate does not authorize calibration aggregation",
    )
    result_paths = tuple(sorted(args.results_root.glob("*/result.json")))
    _require(
        len(result_paths) == 6,
        "calibration aggregate requires exactly six cases",
    )
    results = [
        json.loads(path.read_text(encoding="utf-8")) for path in result_paths
    ]
    _require(
        {result["authorized_split"] for result in results} == {"calibration"},
        "calibration aggregate contains another split",
    )
    dynamic = [
        result for result in results if result["case_id"].endswith("/dynamic")
    ]
    _require(len(dynamic) == 3, "calibration dynamic case count changed")
    dynamic_improvements = np.asarray(
        [
            result["metrics"]["symmetric_relative_improvement"]
            for result in dynamic
        ],
        dtype=np.float64,
    )
    calibration_accuracy_gate = bool(
        np.mean(dynamic_improvements) >= 0.05
        and np.sum(dynamic_improvements > 0.0) >= 2
        and np.min(dynamic_improvements) >= -0.05
    )
    standard_normal_90 = 1.6448536269514722
    trial_scale_requirements = {
        result["case_id"]: max(
            1.0,
            float(
                result["metrics"]["trial_coordinate_abs_standardized_q90"]
                / standard_normal_90
            ),
        )
        for result in results
    }
    uncertainty_std_multiplier = float(
        max(trial_scale_requirements.values())
    )
    payload = {
        "schema_version": 1,
        "artifact_kind": "ClothSim2RealCalibrationGate",
        "method_id": METHOD_ID,
        "source_gate_sha256": _sha256(source_gate_path),
        "result_sha256s": {
            path.parent.name: _sha256(path) for path in result_paths
        },
        "case_metrics": {
            result["case_id"]: result["metrics"] for result in results
        },
        "dynamic_object_balanced_relative_improvement": float(
            np.mean(dynamic_improvements)
        ),
        "dynamic_win_count": int(np.sum(dynamic_improvements > 0.0)),
        "worst_dynamic_relative_improvement": float(
            np.min(dynamic_improvements)
        ),
        "calibration_accuracy_gate_passed": calibration_accuracy_gate,
        "trial_uncertainty_std_requirements": trial_scale_requirements,
        "uncertainty_std_multiplier": uncertainty_std_multiplier,
        "uncertainty_rule": (
            "maximum over six calibration-trial 90th percentiles of absolute "
            "coordinate residual divided by raw predictive standard deviation"
        ),
        "formal_90_split_conformal_claim": False,
        "finite_session_resolution": (
            "with six calibration sessions the maximum finite order statistic "
            "corresponds to rank 6/7, not formal 90% coverage"
        ),
        "target_authorized": calibration_accuracy_gate,
    }
    _write_json_once(args.output.resolve(), payload)
    return 0


def _target_task_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    _require(len(results) == 3, "target task requires exactly three cloths")

    def mean_metric(name: str) -> float:
        return float(np.mean([result["metrics"][name] for result in results]))

    horizon_names = ("early", "middle", "late")
    horizons = []
    for name in horizon_names:
        rows = [
            next(
                horizon
                for horizon in result["horizons"]
                if horizon["name"] == name
            )
            for result in results
        ]
        horizons.append(
            {
                "name": name,
                "physical_symmetric_l1_chamfer_m": float(
                    np.mean(
                        [
                            row["physical_symmetric_l1_chamfer_m"]
                            for row in rows
                        ]
                    )
                ),
                "candidate_symmetric_l1_chamfer_m": float(
                    np.mean(
                        [
                            row["candidate_symmetric_l1_chamfer_m"]
                            for row in rows
                        ]
                    )
                ),
                "object_balanced_relative_improvement": float(
                    np.mean([row["relative_improvement"] for row in rows])
                ),
                "raw_90_coordinate_coverage": float(
                    np.mean(
                        [row["raw_90_coordinate_coverage"] for row in rows]
                    )
                ),
                "reported_90_coordinate_coverage": float(
                    np.mean(
                        [
                            row["reported_90_coordinate_coverage"]
                            for row in rows
                        ]
                    )
                ),
            }
        )

    symmetric_improvements = np.asarray(
        [
            result["metrics"]["symmetric_relative_improvement"]
            for result in results
        ],
        dtype=np.float64,
    )
    return {
        "case_count": len(results),
        "accepted_case_count": int(
            np.sum([bool(result["accepted"]) for result in results])
        ),
        "symmetric_win_count": int(np.sum(symmetric_improvements > 0.0)),
        "physical_symmetric_l1_chamfer_m": mean_metric(
            "physical_symmetric_l1_chamfer_m"
        ),
        "candidate_symmetric_l1_chamfer_m": mean_metric(
            "candidate_symmetric_l1_chamfer_m"
        ),
        "object_balanced_symmetric_relative_improvement": float(
            np.mean(symmetric_improvements)
        ),
        "object_balanced_directed_relative_improvement": mean_metric(
            "directed_relative_improvement"
        ),
        "object_balanced_released_window_directed_relative_improvement": (
            mean_metric("released_window_directed_relative_improvement")
        ),
        "object_balanced_hausdorff_relative_improvement": mean_metric(
            "hausdorff_relative_improvement"
        ),
        "raw_90_coordinate_coverage": mean_metric(
            "raw_90_coordinate_coverage"
        ),
        "reported_90_coordinate_coverage": mean_metric(
            "reported_90_coordinate_coverage"
        ),
        "mean_90_interval_width_m": mean_metric("mean_90_interval_width_m"),
        "mean_energy_score_m": mean_metric("mean_energy_score_m"),
        "mean_readout_correction_m": mean_metric(
            "mean_readout_correction_m"
        ),
        "horizons": horizons,
    }


def _aggregate_target(args: argparse.Namespace) -> int:
    calibration_path = args.calibration_gate.resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration_sha256 = _sha256(calibration_path)
    _require(
        calibration.get("artifact_kind") == "ClothSim2RealCalibrationGate"
        and calibration.get("target_authorized") is True,
        "calibration gate does not authorize target aggregation",
    )
    target_lock_path = args.target_lock.resolve()
    target_lock = json.loads(target_lock_path.read_text(encoding="utf-8"))
    _require(
        target_lock.get("protocol_id") == "cloth-sim2real-online-belief-v1"
        and target_lock.get("method_id") == METHOD_ID
        and target_lock.get("status") == "pre_target_prefix_lock",
        "target lock identity changed",
    )
    _require(
        target_lock["calibration_evidence"]["calibration_gate_sha256"]
        == calibration_sha256
        and target_lock["calibration_evidence"]["target_authorized"] is True,
        "target lock and calibration gate differ",
    )
    _require(
        target_lock["target_scope"]["case_count"] == 6,
        "target lock case count changed",
    )

    result_paths = tuple(sorted(args.results_root.glob("*/result.json")))
    _require(len(result_paths) == 6, "target aggregate requires exactly six cases")
    results = [
        json.loads(path.read_text(encoding="utf-8")) for path in result_paths
    ]
    _require(
        {result["artifact_kind"] for result in results}
        == {"ClothSim2RealPredictionResult"}
        and {result["method_id"] for result in results} == {METHOD_ID}
        and {result["authorized_split"] for result in results} == {"target"},
        "target aggregate contains an incompatible result",
    )
    _require(
        all(
            result["calibration_artifact_sha256"] == calibration_sha256
            and result["future_outcomes_read_only_after_prediction_seal"]
            is True
            for result in results
        ),
        "target result violates its calibration or outcome boundary",
    )
    dynamic = [
        result for result in results if result["case_id"].endswith("/dynamic")
    ]
    quasi_static = [
        result
        for result in results
        if result["case_id"].endswith("/quasi_static")
    ]
    _require(
        len(dynamic) == len(quasi_static) == 3,
        "target task counts changed",
    )
    payload = {
        "schema_version": 1,
        "artifact_kind": "ClothSim2RealTargetResult",
        "method_id": METHOD_ID,
        "calibration_gate_sha256": calibration_sha256,
        "target_lock_sha256": _sha256(target_lock_path),
        "result_sha256s": {
            path.parent.name: _sha256(path) for path in result_paths
        },
        "prediction_seal_sha256s": {
            result["case_id"]: result["prediction_seal_sha256"]
            for result in results
        },
        "case_metrics": {
            result["case_id"]: result["metrics"] for result in results
        },
        "case_horizons": {
            result["case_id"]: result["horizons"] for result in results
        },
        "dynamic_primary": _target_task_summary(dynamic),
        "quasi_static_secondary": _target_task_summary(quasi_static),
        "uncertainty_std_multiplier": float(
            calibration["uncertainty_std_multiplier"]
        ),
        "formal_90_split_conformal_claim": False,
        "target_outcomes_opened_only_after_all_case_prediction_seals": True,
        "claim_boundary": (
            "independent repeat-2 evidence for causal online continuation; "
            "not an identical-information open-loop SOTA comparison"
        ),
    }
    _write_json_once(args.output.resolve(), payload)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--benchmark-code-root", type=Path, required=True)
    simulate.add_argument("--case-id", required=True)
    simulate.add_argument("--output", type=Path, required=True)
    simulate.set_defaults(function=_simulate)

    seal = subparsers.add_parser("seal")
    seal.add_argument("--manifest", type=Path, required=True)
    seal.add_argument("--dataset-root", type=Path, required=True)
    seal.add_argument("--baseline", type=Path, required=True)
    seal.add_argument("--case-id", required=True)
    seal.add_argument(
        "--authorized-split",
        choices=("source", "calibration", "target"),
        required=True,
    )
    seal.add_argument("--authorization-artifact", type=Path)
    seal.add_argument("--output-dir", type=Path, required=True)
    seal.set_defaults(function=_seal)

    score = subparsers.add_parser("score")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--dataset-root", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--calibration-artifact", type=Path)
    score.set_defaults(function=_score)

    aggregate = subparsers.add_parser("aggregate-source")
    aggregate.add_argument("--results-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.set_defaults(function=_aggregate)

    aggregate_calibration = subparsers.add_parser("aggregate-calibration")
    aggregate_calibration.add_argument(
        "--results-root",
        type=Path,
        required=True,
    )
    aggregate_calibration.add_argument(
        "--source-gate",
        type=Path,
        required=True,
    )
    aggregate_calibration.add_argument("--output", type=Path, required=True)
    aggregate_calibration.set_defaults(function=_aggregate_calibration)

    aggregate_target = subparsers.add_parser("aggregate-target")
    aggregate_target.add_argument(
        "--results-root",
        type=Path,
        required=True,
    )
    aggregate_target.add_argument(
        "--calibration-gate",
        type=Path,
        required=True,
    )
    aggregate_target.add_argument(
        "--target-lock",
        type=Path,
        required=True,
    )
    aggregate_target.add_argument("--output", type=Path, required=True)
    aggregate_target.set_defaults(function=_aggregate_target)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
