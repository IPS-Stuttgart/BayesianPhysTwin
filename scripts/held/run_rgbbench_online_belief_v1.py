#!/usr/bin/env python3
"""Run the prospective RGBench guarded online-belief protocol.

Simulation and prediction sealing read only the known physical inputs and the
permitted real prefix. Future point clouds are opened only by ``score`` after
the prediction seal and its hashes have been verified.
"""

from __future__ import annotations

import argparse
import csv
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
    symmetric_l1_chamfer_m,
    symmetric_l2_hausdorff_m,
)
from bayesian_phystwin.rgbench_online_belief import (
    evaluation_pcd_paths,
    load_obj_triangles,
    load_rgbbench_world_cloud,
    real_to_sim_l1_chamfer_m,
    sha256_file,
)
from bayesian_phystwin.rgbench_protocol import RGBENCH_COMMIT

METHOD_ID = "rgbbench-guarded-online-readout-v1"
SIMULATOR = "pybullet"
MODE = "fixed_point"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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
        payload.get("artifact_kind") == "RGBenchDatasetManifest",
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
            payload.get("artifact_kind") == "RGBenchSourceGate"
            and payload.get("calibration_authorized") is True,
            "source gate does not authorize calibration",
        )
    else:
        _require(
            payload.get("artifact_kind") == "RGBenchCalibrationGate"
            and payload.get("target_authorized") is True,
            "calibration gate does not authorize target",
        )
    return str(artifact_path.resolve()), sha256_file(artifact_path)


def _capture_root(dataset_root: Path, case: dict[str, Any]) -> Path:
    capture = dataset_root / case["data_subfolder"]
    _require(capture.is_dir(), f"capture does not exist: {capture}")
    return capture


def _case_pcd_paths(dataset_root: Path, case: dict[str, Any]) -> tuple[Path, ...]:
    return evaluation_pcd_paths(
        _capture_root(dataset_root, case),
        master_start_time_s=float(case["master_start_time_s"]),
        camera_delay_s=float(case["camera_delay_s"]),
        start_calculate_time_s=float(case["start_calculate_time_s"]),
        end_calculate_time_s=float(case["end_calculate_time_s"]),
        expected_count=int(case["evaluation_frame_count"]),
        expected_name_sha256=str(case["point_cloud_name_sha256"]),
    )


def _load_world_clouds(
    dataset_root: Path,
    case: dict[str, Any],
    start: int,
    stop: int,
) -> list[np.ndarray]:
    paths = _case_pcd_paths(dataset_root, case)
    _require(0 <= start < stop <= len(paths), "invalid point-cloud interval")
    transform = (
        _capture_root(dataset_root, case)
        / "calibration/world_to_camera_transform.json"
    )
    _require(
        sha256_file(transform) == case["calibration_sha256"],
        "camera calibration changed after locking",
    )
    return [load_rgbbench_world_cloud(path, transform) for path in paths[start:stop]]


def _compose_rgbbench_config(
    benchmark_root: Path,
    dataset_root: Path,
    case: dict[str, Any],
) -> object:
    sys.path.insert(0, str(benchmark_root))
    compose = importlib.import_module("hydra").compose
    initialize_config_dir = importlib.import_module(
        "hydra"
    ).initialize_config_dir
    OmegaConf = importlib.import_module("omegaconf").OmegaConf
    overrides = [
        f"params.cloth_name={case['garment']}",
        f"params.action_type={case['action']}",
        f"params.sample_index={case['sample']}",
        f"params.sim_environment={SIMULATOR}",
        f"params.sim_mode={MODE}",
        f"cloth_params={case['garment']}",
        f"env={SIMULATOR}",
        f"dataset_path={dataset_root}",
        f"cloth_model_path={dataset_root / 'meshes'}",
        f"project_root={benchmark_root}",
        f"output_path={benchmark_root / 'outputs'}",
        "active_run.visualization.vis_sim=false",
        "active_run.visualization.save_gifs=false",
        "active_run.visualization.save_sim_pcd=false",
        "active_run.visualization.save_target_pcd=false",
    ]
    with initialize_config_dir(
        version_base=None,
        config_dir=str((benchmark_root / "configs").resolve()),
    ):
        config = compose(config_name="main", overrides=overrides)
    OmegaConf.resolve(config)
    return config.active_run


def _simulate(args: argparse.Namespace) -> int:
    benchmark = args.benchmark_root.resolve()
    git_head = subprocess.check_output(
        ["git", "-C", str(benchmark), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    _require(git_head == RGBENCH_COMMIT, "RGBench checkout commit changed")
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    case = _case_descriptor(manifest, args.case_id, args.authorized_split)
    dataset = args.dataset_root.resolve()
    paths = _case_pcd_paths(dataset, case)
    config = _compose_rgbbench_config(benchmark, dataset, case)
    get_env = importlib.import_module("rgbench.envs").get_env
    environment = get_env(config)
    try:
        master_start = float(environment.get_master_start_time())
        _require(
            abs(master_start - float(case["master_start_time_s"])) <= 1e-6,
            "simulator and manifest master start times differ",
        )
        preparation_time = (
            float(config.action.fling_prepare_time)
            + float(config.action.fling_wait_time)
            if case["action"] == "fling"
            else 0.0
        )
        vertices: list[np.ndarray] = []
        target_times: list[float] = []
        for path in paths:
            absolute_time = float(
                path.name.removeprefix("pointcloud_").removesuffix(
                    "_segmented.pcd"
                )
            )
            target_time = absolute_time - master_start
            compensated_time = (
                target_time
                + float(case["camera_delay_s"])
                + preparation_time
            )
            environment.step_to_time(compensated_time)
            frame = np.asarray(environment.get_sim_vertices(), dtype=np.float64)
            _require(
                frame.ndim == 2
                and frame.shape[1] == 3
                and np.all(np.isfinite(frame)),
                "simulator returned invalid vertices",
            )
            vertices.append(frame.copy())
            target_times.append(target_time)
    finally:
        environment.close()

    mesh_path = dataset / "meshes" / case["mesh_relative_path"]
    _require(
        sha256_file(mesh_path) == case["mesh_sha256"],
        "physical mesh changed after locking",
    )
    _, faces = load_obj_triangles(mesh_path)
    node_count = len(vertices[0])
    _require(
        all(len(frame) == node_count for frame in vertices),
        "simulator node count changed during rollout",
    )
    _require(
        int(np.max(faces)) < node_count,
        "mesh faces do not index the simulator vertices",
    )
    output = args.output.resolve()
    _require(not output.exists(), f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        vertices_m=np.stack(vertices),
        faces=faces,
        target_times_s=np.asarray(target_times, dtype=np.float64),
    )
    _write_json_once(
        output.with_suffix(".json"),
        {
            "schema_version": 1,
            "artifact_kind": "RGBenchPhysicalBaseline",
            "method_id": METHOD_ID,
            "case_id": case["case_id"],
            "authorized_split": args.authorized_split,
            "rgbbench_commit": RGBENCH_COMMIT,
            "dataset_revision": manifest["dataset_revision"],
            "simulator": SIMULATOR,
            "mode": MODE,
            "node_count": node_count,
            "face_count": len(faces),
            "evaluation_frame_count": len(vertices),
            "manifest_sha256": sha256_file(manifest_path),
            "npz_sha256": sha256_file(output),
            "point_cloud_filenames_read": True,
            "point_cloud_coordinates_read": False,
            "known_future_actuator_trajectory_read": True,
            "future_object_outcomes_read": False,
        },
    )
    return 0


def _belief_config() -> ClothReadoutBeliefConfig:
    return ClothReadoutBeliefConfig(
        graph_prior_strengths=(0.1, 1.0, 10.0),
        correction_scales=(0.25, 0.5, 0.75, 1.0),
        maximum_correction_m=0.10,
        minimum_validation_improvement=0.02,
        minimum_validation_win_fraction=0.60,
        maximum_validation_worst_ratio=1.0,
        covariance_probes=16,
        covariance_seed=20260730,
    )


def _seal(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    case = _case_descriptor(manifest, args.case_id, args.authorized_split)
    authorization_path, authorization_sha256 = _authorization(
        args.authorized_split,
        None
        if args.authorization_artifact is None
        else args.authorization_artifact.resolve(),
    )
    baseline_path = args.baseline.resolve()
    baseline_metadata_path = baseline_path.with_suffix(".json")
    _require(
        baseline_path.is_file() and baseline_metadata_path.is_file(),
        "physical baseline is incomplete",
    )
    baseline_metadata = json.loads(
        baseline_metadata_path.read_text(encoding="utf-8")
    )
    _require(
        baseline_metadata.get("artifact_kind") == "RGBenchPhysicalBaseline"
        and baseline_metadata.get("case_id") == case["case_id"]
        and baseline_metadata.get("authorized_split") == args.authorized_split,
        "physical baseline metadata changed",
    )
    _require(
        baseline_metadata["npz_sha256"] == sha256_file(baseline_path),
        "physical baseline changed after simulation",
    )
    with np.load(baseline_path, allow_pickle=False) as baseline:
        physical = np.asarray(baseline["vertices_m"], dtype=np.float64)
        faces = np.asarray(baseline["faces"], dtype=np.int64)
    _require(
        physical.shape[0] == int(case["evaluation_frame_count"]),
        "physical and observed frame counts differ",
    )
    branch = int(case["branch_index"])
    fit_stop = int(case["fit_stop_index"])
    prefix_clouds = _load_world_clouds(
        args.dataset_root.resolve(),
        case,
        0,
        branch + 1,
    )
    config = _belief_config()
    belief = fit_guarded_readout_correction(
        physical[:fit_stop],
        prefix_clouds[:fit_stop],
        physical[fit_stop : branch + 1],
        prefix_clouds[fit_stop : branch + 1],
        faces,
        config=config,
        validation_metric="real_to_sim_l1_chamfer",
    )
    candidate = physical.copy()
    candidate[branch + 1 :] = apply_guarded_readout_correction(
        physical[branch + 1 :],
        belief,
    )
    exact_fallback = not belief.accepted and np.array_equal(candidate, physical)
    _require(
        belief.accepted or exact_fallback,
        "rejected candidate is not an exact physical fallback",
    )

    output = args.output_dir.resolve()
    _require(not output.exists(), f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    physical_path = output / "physical_evaluation.npy"
    candidate_path = output / "candidate_evaluation.npy"
    belief_path = output / "readout_belief.npz"
    np.save(physical_path, physical, allow_pickle=False)
    np.save(candidate_path, candidate, allow_pickle=False)
    np.savez_compressed(
        belief_path,
        accepted=np.asarray(belief.accepted),
        selected_name=np.asarray(belief.selected_name),
        correction_m=belief.correction_m,
        variance_m2=belief.variance_m2,
    )
    _write_json_once(
        output / "prediction_seal.json",
        {
            "schema_version": 1,
            "artifact_kind": "RGBenchPredictionSeal",
            "method_id": METHOD_ID,
            "case_id": case["case_id"],
            "authorized_split": args.authorized_split,
            "authorization_artifact": authorization_path,
            "authorization_artifact_sha256": authorization_sha256,
            "manifest_sha256": sha256_file(manifest_path),
            "baseline_sha256": sha256_file(baseline_path),
            "baseline_metadata_sha256": sha256_file(baseline_metadata_path),
            "fit_indices": [0, fit_stop - 1],
            "validation_indices": [fit_stop, branch],
            "future_indices": [
                branch + 1,
                int(case["evaluation_frame_count"]) - 1,
            ],
            "selected_name": belief.selected_name,
            "accepted": belief.accepted,
            "reason": belief.reason,
            "exact_fallback": exact_fallback,
            "config": asdict(config),
            "scores": [asdict(score) for score in belief.scores],
            "diagnostics": belief.diagnostics,
            "physical_evaluation_sha256": sha256_file(physical_path),
            "candidate_evaluation_sha256": sha256_file(candidate_path),
            "belief_sha256": sha256_file(belief_path),
            "prefix_point_cloud_coordinates_read": True,
            "future_point_cloud_coordinates_read": False,
            "candidate_prefix_is_exact_physical_baseline": bool(
                np.array_equal(candidate[: branch + 1], physical[: branch + 1])
            ),
        },
    )
    return 0


def _normal_energy_score(
    observation_m: np.ndarray,
    prediction_m: np.ndarray,
    variance_m2: np.ndarray,
    *,
    seed: int,
    sample_count: int = 32,
) -> float:
    rng = np.random.default_rng(seed)
    standard_deviation = np.sqrt(variance_m2)
    first = prediction_m[None] + rng.normal(
        size=(sample_count,) + prediction_m.shape
    ) * standard_deviation[None]
    second = prediction_m[None] + rng.normal(
        size=(sample_count,) + prediction_m.shape
    ) * standard_deviation[None]
    first_term = np.mean(np.linalg.norm(first - observation_m[None], axis=2))
    second_term = np.mean(np.linalg.norm(first - second, axis=2))
    return float(first_term - 0.5 * second_term)


def _paper_garment_dynamics(
    baseline_csv: Path,
    garment: str,
    action: str,
) -> float:
    with baseline_csv.open("r", encoding="utf-8", newline="") as stream:
        matches = [
            row
            for row in csv.DictReader(stream)
            if row["garment"] == garment
            and row["action"] == action
            and row["simulator"] == "mujoco_style3d"
            and row["mode"] == MODE
        ]
    _require(len(matches) == 1, "published GarmentDynamics cell is ambiguous")
    _require(int(matches[0]["n_samples"]) == 3, "published sample count changed")
    return float(matches[0]["cd_l1_r2s"])


def _relative_improvement(baseline: np.ndarray, candidate: np.ndarray) -> float:
    return float(
        (np.mean(baseline) - np.mean(candidate))
        / max(float(np.mean(baseline)), 1e-15)
    )


def _score(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    seal_path = output / "prediction_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _require(
        seal.get("artifact_kind") == "RGBenchPredictionSeal",
        "prediction seal kind changed",
    )
    manifest_path = args.manifest.resolve()
    _require(
        seal["manifest_sha256"] == sha256_file(manifest_path),
        "prediction seal and manifest differ",
    )
    physical_path = output / "physical_evaluation.npy"
    candidate_path = output / "candidate_evaluation.npy"
    belief_path = output / "readout_belief.npz"
    for path, key in (
        (physical_path, "physical_evaluation_sha256"),
        (candidate_path, "candidate_evaluation_sha256"),
        (belief_path, "belief_sha256"),
    ):
        _require(sha256_file(path) == seal[key], f"{path.name} changed after sealing")
    manifest = _load_manifest(manifest_path)
    case = _case_descriptor(
        manifest,
        seal["case_id"],
        seal["authorized_split"],
    )
    calibration_std_multiplier = 1.0
    calibration_artifact_sha256 = None
    if seal["authorized_split"] == "target":
        _require(
            args.calibration_artifact is not None
            and args.calibration_artifact.is_file(),
            "target scoring requires the calibration gate",
        )
        calibration = json.loads(
            args.calibration_artifact.read_text(encoding="utf-8")
        )
        calibration_artifact_sha256 = sha256_file(args.calibration_artifact)
        _require(
            calibration.get("artifact_kind") == "RGBenchCalibrationGate"
            and calibration.get("target_authorized") is True,
            "calibration gate does not authorize target scoring",
        )
        _require(
            calibration_artifact_sha256
            == seal["authorization_artifact_sha256"],
            "target seal and calibration gate differ",
        )
        calibration_std_multiplier = float(
            calibration["uncertainty_std_multiplier"]
        )
    physical = np.load(physical_path, allow_pickle=False)
    candidate = np.load(candidate_path, allow_pickle=False)
    with np.load(belief_path, allow_pickle=False) as belief:
        variance = np.asarray(belief["variance_m2"], dtype=np.float64)
        correction = np.asarray(belief["correction_m"], dtype=np.float64)
    clouds = _load_world_clouds(
        args.dataset_root.resolve(),
        case,
        0,
        int(case["evaluation_frame_count"]),
    )
    _require(len(physical) == len(candidate) == len(clouds), "score lengths differ")
    physical_primary = np.asarray(
        [
            real_to_sim_l1_chamfer_m(observed, prediction)
            for prediction, observed in zip(physical, clouds, strict=True)
        ],
        dtype=np.float64,
    )
    candidate_primary = np.asarray(
        [
            real_to_sim_l1_chamfer_m(observed, prediction)
            for prediction, observed in zip(candidate, clouds, strict=True)
        ],
        dtype=np.float64,
    )
    physical_secondary = np.asarray(
        [
            directed_l1_chamfer_m(prediction, observed)
            for prediction, observed in zip(physical, clouds, strict=True)
        ],
        dtype=np.float64,
    )
    candidate_secondary = np.asarray(
        [
            directed_l1_chamfer_m(prediction, observed)
            for prediction, observed in zip(candidate, clouds, strict=True)
        ],
        dtype=np.float64,
    )
    physical_symmetric = np.asarray(
        [
            symmetric_l1_chamfer_m(prediction, observed)
            for prediction, observed in zip(physical, clouds, strict=True)
        ],
        dtype=np.float64,
    )
    candidate_symmetric = np.asarray(
        [
            symmetric_l1_chamfer_m(prediction, observed)
            for prediction, observed in zip(candidate, clouds, strict=True)
        ],
        dtype=np.float64,
    )
    physical_hausdorff = np.asarray(
        [
            symmetric_l2_hausdorff_m(prediction, observed)
            for prediction, observed in zip(physical, clouds, strict=True)
        ],
        dtype=np.float64,
    )
    candidate_hausdorff = np.asarray(
        [
            symmetric_l2_hausdorff_m(prediction, observed)
            for prediction, observed in zip(candidate, clouds, strict=True)
        ],
        dtype=np.float64,
    )
    branch = int(case["branch_index"])
    future = np.arange(branch + 1, len(physical))
    horizons = np.array_split(future, 3)
    config = ClothReadoutBeliefConfig(**seal["config"])
    raw_coverage: list[float] = []
    calibrated_coverage: list[float] = []
    interval_widths: list[float] = []
    energy_scores: list[float] = []
    standardized: list[np.ndarray] = []
    for horizon, index in enumerate(future, start=1):
        association = associate_dense_cloud(
            candidate[index],
            clouds[index],
            candidate_count=config.candidate_count,
            sensor_std_m=config.sensor_std_m,
        )
        raw_variance = (
            variance
            + horizon * config.forecast_process_std_m_per_sqrt_frame**2
            + association.variance_m2[:, None]
        )
        total_variance = calibration_std_multiplier**2 * raw_variance
        residual = association.observed_points_m - candidate[index]
        standardized.append(np.abs(residual) / np.sqrt(raw_variance))
        raw_half_width = 1.6448536269514722 * np.sqrt(raw_variance)
        half_width = 1.6448536269514722 * np.sqrt(total_variance)
        raw_coverage.append(float(np.mean(np.abs(residual) <= raw_half_width)))
        calibrated_coverage.append(float(np.mean(np.abs(residual) <= half_width)))
        interval_widths.append(float(np.mean(2.0 * half_width)))
        energy_scores.append(
            _normal_energy_score(
                association.observed_points_m,
                candidate[index],
                total_variance,
                seed=config.covariance_seed + horizon,
            )
        )
    standardized_q90 = float(
        np.quantile(
            np.concatenate([values.reshape(-1) for values in standardized]),
            0.90,
        )
    )
    paper_csv = args.paper_baselines.resolve()
    _require(
        sha256_file(paper_csv) == manifest["paper_baselines_sha256"],
        "paper baseline CSV changed after locking",
    )
    paper_gd = _paper_garment_dynamics(
        paper_csv,
        case["garment"],
        case["action"],
    )
    result = {
        "schema_version": 1,
        "artifact_kind": "RGBenchPredictionResult",
        "method_id": METHOD_ID,
        "case_id": case["case_id"],
        "garment": case["garment"],
        "action": case["action"],
        "sample": case["sample"],
        "authorized_split": seal["authorized_split"],
        "prediction_seal_sha256": sha256_file(seal_path),
        "calibration_artifact_sha256": calibration_artifact_sha256,
        "uncertainty_std_multiplier": calibration_std_multiplier,
        "accepted": seal["accepted"],
        "selected_name": seal["selected_name"],
        "metrics": {
            "published_garment_dynamics_cell_real_to_sim_l1_m": paper_gd,
            "physical_full_real_to_sim_l1_m": float(np.mean(physical_primary)),
            "candidate_full_real_to_sim_l1_m": float(np.mean(candidate_primary)),
            "full_primary_relative_improvement": _relative_improvement(
                physical_primary,
                candidate_primary,
            ),
            "full_primary_frame_wins": int(
                np.sum(candidate_primary < physical_primary)
            ),
            "physical_future_real_to_sim_l1_m": float(
                np.mean(physical_primary[future])
            ),
            "candidate_future_real_to_sim_l1_m": float(
                np.mean(candidate_primary[future])
            ),
            "future_primary_relative_improvement": _relative_improvement(
                physical_primary[future],
                candidate_primary[future],
            ),
            "physical_full_sim_to_real_l1_m": float(
                np.mean(physical_secondary)
            ),
            "candidate_full_sim_to_real_l1_m": float(
                np.mean(candidate_secondary)
            ),
            "physical_full_symmetric_l1_m": float(
                np.mean(physical_symmetric)
            ),
            "candidate_full_symmetric_l1_m": float(
                np.mean(candidate_symmetric)
            ),
            "physical_full_symmetric_hausdorff_m": float(
                np.mean(physical_hausdorff)
            ),
            "candidate_full_symmetric_hausdorff_m": float(
                np.mean(candidate_hausdorff)
            ),
            "raw_90_coordinate_coverage": float(np.mean(raw_coverage)),
            "reported_90_coordinate_coverage": float(
                np.mean(calibrated_coverage)
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
                "physical_real_to_sim_l1_m": float(
                    np.mean(physical_primary[indices])
                ),
                "candidate_real_to_sim_l1_m": float(
                    np.mean(candidate_primary[indices])
                ),
                "relative_improvement": _relative_improvement(
                    physical_primary[indices],
                    candidate_primary[indices],
                ),
            }
            for name, indices in zip(
                ("early", "middle", "late"),
                horizons,
                strict=True,
            )
        ],
        "future_outcomes_read_only_after_prediction_seal": True,
        "claim_boundary": (
            "causal online continuation after a real prefix; the published "
            "GarmentDynamics comparator is open-loop and has less information"
        ),
    }
    _write_json_once(output / "result.json", result)
    return 0


def _cell_summaries(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    keys = sorted({(result["garment"], result["action"]) for result in results})
    for garment, action in keys:
        rows = [
            result
            for result in results
            if result["garment"] == garment and result["action"] == action
        ]
        _require(len(rows) == 3, f"{garment}/{action} does not have three samples")
        physical = float(
            np.mean(
                [
                    row["metrics"]["physical_full_real_to_sim_l1_m"]
                    for row in rows
                ]
            )
        )
        candidate = float(
            np.mean(
                [
                    row["metrics"]["candidate_full_real_to_sim_l1_m"]
                    for row in rows
                ]
            )
        )
        paper_values = {
            row["metrics"]["published_garment_dynamics_cell_real_to_sim_l1_m"]
            for row in rows
        }
        _require(len(paper_values) == 1, "published cell value changed by sample")
        paper = float(next(iter(paper_values)))
        cells.append(
            {
                "garment": garment,
                "action": action,
                "physical_real_to_sim_l1_m": physical,
                "candidate_real_to_sim_l1_m": candidate,
                "relative_improvement": (physical - candidate)
                / max(physical, 1e-15),
                "published_garment_dynamics_real_to_sim_l1_m": paper,
                "candidate_beats_published_garment_dynamics": candidate < paper,
            }
        )
    return cells


def _load_split_results(
    results_root: Path,
    expected_split: str,
    expected_count: int,
) -> tuple[list[Path], list[dict[str, Any]]]:
    paths = list(sorted(results_root.glob("*/*/*/result.json")))
    _require(
        len(paths) == expected_count,
        f"{expected_split} aggregate requires exactly {expected_count} cases",
    )
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    _require(
        {result["authorized_split"] for result in results} == {expected_split},
        f"{expected_split} aggregate contains another split",
    )
    return paths, results


def _garment_improvements(cells: list[dict[str, Any]]) -> dict[str, float]:
    garments = sorted({cell["garment"] for cell in cells})
    return {
        garment: float(
            np.mean(
                [
                    cell["relative_improvement"]
                    for cell in cells
                    if cell["garment"] == garment
                ]
            )
        )
        for garment in garments
    }


def _aggregate_source(args: argparse.Namespace) -> int:
    paths, results = _load_split_results(args.results_root, "source", 27)
    cells = _cell_summaries(results)
    cell_improvements = np.asarray(
        [cell["relative_improvement"] for cell in cells],
        dtype=np.float64,
    )
    garment_improvements = _garment_improvements(cells)
    candidate_mean = float(
        np.mean([cell["candidate_real_to_sim_l1_m"] for cell in cells])
    )
    paper_mean = float(
        np.mean(
            [
                cell["published_garment_dynamics_real_to_sim_l1_m"]
                for cell in cells
            ]
        )
    )
    gates = {
        "mean_relative_improvement_at_least_5pct": bool(
            np.mean(cell_improvements) >= 0.05
        ),
        "all_three_garments_nonregressing": bool(
            all(value >= 0.0 for value in garment_improvements.values())
        ),
        "at_least_six_of_nine_cells_improve": bool(
            np.sum(cell_improvements > 0.0) >= 6
        ),
        "aggregate_beats_published_garment_dynamics": candidate_mean < paper_mean,
        "at_least_six_of_nine_cells_beat_published_garment_dynamics": bool(
            sum(
                cell["candidate_beats_published_garment_dynamics"]
                for cell in cells
            )
            >= 6
        ),
    }
    passed = all(gates.values())
    _write_json_once(
        args.output.resolve(),
        {
            "schema_version": 1,
            "artifact_kind": "RGBenchSourceGate",
            "method_id": METHOD_ID,
            "result_sha256s": {
                str(path.relative_to(args.results_root)): sha256_file(path)
                for path in paths
            },
            "cells": cells,
            "garment_relative_improvements": garment_improvements,
            "object_action_balanced_relative_improvement": float(
                np.mean(cell_improvements)
            ),
            "candidate_object_action_balanced_real_to_sim_l1_m": candidate_mean,
            "published_garment_dynamics_object_action_balanced_real_to_sim_l1_m": (
                paper_mean
            ),
            "gates": gates,
            "source_gate_passed": passed,
            "calibration_authorized": passed,
            "target_authorized": False,
        },
    )
    return 0


def _aggregate_calibration(args: argparse.Namespace) -> int:
    source_gate_path = args.source_gate.resolve()
    source_gate = json.loads(source_gate_path.read_text(encoding="utf-8"))
    _require(
        source_gate.get("artifact_kind") == "RGBenchSourceGate"
        and source_gate.get("calibration_authorized") is True,
        "source gate does not authorize calibration",
    )
    paths, results = _load_split_results(args.results_root, "calibration", 18)
    cells = _cell_summaries(results)
    cell_improvements = np.asarray(
        [cell["relative_improvement"] for cell in cells],
        dtype=np.float64,
    )
    garment_improvements = _garment_improvements(cells)
    candidate_mean = float(
        np.mean([cell["candidate_real_to_sim_l1_m"] for cell in cells])
    )
    paper_mean = float(
        np.mean(
            [
                cell["published_garment_dynamics_real_to_sim_l1_m"]
                for cell in cells
            ]
        )
    )
    gates = {
        "mean_relative_improvement_at_least_3pct": bool(
            np.mean(cell_improvements) >= 0.03
        ),
        "both_garments_nonregressing": bool(
            all(value >= 0.0 for value in garment_improvements.values())
        ),
        "at_least_four_of_six_cells_improve": bool(
            np.sum(cell_improvements > 0.0) >= 4
        ),
        "aggregate_beats_published_garment_dynamics": candidate_mean < paper_mean,
        "at_least_four_of_six_cells_beat_published_garment_dynamics": bool(
            sum(
                cell["candidate_beats_published_garment_dynamics"]
                for cell in cells
            )
            >= 4
        ),
    }
    standard_normal_90 = 1.6448536269514722
    requirements = {
        result["case_id"]: max(
            1.0,
            float(
                result["metrics"]["trial_coordinate_abs_standardized_q90"]
                / standard_normal_90
            ),
        )
        for result in results
    }
    multiplier = float(max(requirements.values()))
    passed = all(gates.values())
    _write_json_once(
        args.output.resolve(),
        {
            "schema_version": 1,
            "artifact_kind": "RGBenchCalibrationGate",
            "method_id": METHOD_ID,
            "source_gate_sha256": sha256_file(source_gate_path),
            "result_sha256s": {
                str(path.relative_to(args.results_root)): sha256_file(path)
                for path in paths
            },
            "cells": cells,
            "garment_relative_improvements": garment_improvements,
            "object_action_balanced_relative_improvement": float(
                np.mean(cell_improvements)
            ),
            "candidate_object_action_balanced_real_to_sim_l1_m": candidate_mean,
            "published_garment_dynamics_object_action_balanced_real_to_sim_l1_m": (
                paper_mean
            ),
            "gates": gates,
            "trial_uncertainty_std_requirements": requirements,
            "uncertainty_std_multiplier": multiplier,
            "calibration_order_statistic_rank": 18,
            "calibration_session_count": 18,
            "formal_90_split_conformal_claim": True,
            "coverage_scope": (
                "marginal trial-level score coverage under exchangeability; "
                "not simultaneous coordinate or subgroup coverage"
            ),
            "calibration_gate_passed": passed,
            "target_authorized": passed,
        },
    )
    return 0


def _aggregate_target(args: argparse.Namespace) -> int:
    calibration_gate_path = args.calibration_gate.resolve()
    calibration_gate = json.loads(
        calibration_gate_path.read_text(encoding="utf-8")
    )
    _require(
        calibration_gate.get("artifact_kind") == "RGBenchCalibrationGate"
        and calibration_gate.get("target_authorized") is True,
        "calibration gate does not authorize target aggregation",
    )
    paths, results = _load_split_results(args.results_root, "target", 18)
    cells = _cell_summaries(results)
    cell_improvements = np.asarray(
        [cell["relative_improvement"] for cell in cells],
        dtype=np.float64,
    )
    garment_improvements = _garment_improvements(cells)
    candidate_mean = float(
        np.mean([cell["candidate_real_to_sim_l1_m"] for cell in cells])
    )
    paper_mean = float(
        np.mean(
            [
                cell["published_garment_dynamics_real_to_sim_l1_m"]
                for cell in cells
            ]
        )
    )
    _write_json_once(
        args.output.resolve(),
        {
            "schema_version": 1,
            "artifact_kind": "RGBenchTargetResult",
            "method_id": METHOD_ID,
            "calibration_gate_sha256": sha256_file(calibration_gate_path),
            "result_sha256s": {
                str(path.relative_to(args.results_root)): sha256_file(path)
                for path in paths
            },
            "cells": cells,
            "garment_relative_improvements": garment_improvements,
            "object_action_balanced_relative_improvement": float(
                np.mean(cell_improvements)
            ),
            "candidate_object_action_balanced_real_to_sim_l1_m": candidate_mean,
            "published_garment_dynamics_object_action_balanced_real_to_sim_l1_m": (
                paper_mean
            ),
            "improved_cell_count": int(np.sum(cell_improvements > 0.0)),
            "candidate_sota_cell_count": int(
                sum(
                    cell["candidate_beats_published_garment_dynamics"]
                    for cell in cells
                )
            ),
            "accepted_case_count": int(sum(result["accepted"] for result in results)),
            "reported_90_coordinate_coverage": float(
                np.mean(
                    [
                        result["metrics"]["reported_90_coordinate_coverage"]
                        for result in results
                    ]
                )
            ),
            "mean_90_interval_width_m": float(
                np.mean(
                    [
                        result["metrics"]["mean_90_interval_width_m"]
                        for result in results
                    ]
                )
            ),
            "claim_boundary": (
                "independent unseen-garment online continuation result; "
                "not an identical-information open-loop simulator comparison"
            ),
        },
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--manifest", type=Path, required=True)
    simulate.add_argument("--dataset-root", type=Path, required=True)
    simulate.add_argument("--benchmark-root", type=Path, required=True)
    simulate.add_argument("--case-id", required=True)
    simulate.add_argument(
        "--authorized-split",
        choices=("source", "calibration", "target"),
        required=True,
    )
    simulate.add_argument("--output", type=Path, required=True)
    simulate.set_defaults(func=_simulate)

    seal = subparsers.add_parser("seal")
    seal.add_argument("--manifest", type=Path, required=True)
    seal.add_argument("--dataset-root", type=Path, required=True)
    seal.add_argument("--case-id", required=True)
    seal.add_argument(
        "--authorized-split",
        choices=("source", "calibration", "target"),
        required=True,
    )
    seal.add_argument("--authorization-artifact", type=Path)
    seal.add_argument("--baseline", type=Path, required=True)
    seal.add_argument("--output-dir", type=Path, required=True)
    seal.set_defaults(func=_seal)

    score = subparsers.add_parser("score")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--dataset-root", type=Path, required=True)
    score.add_argument("--paper-baselines", type=Path, required=True)
    score.add_argument("--calibration-artifact", type=Path)
    score.add_argument("--output-dir", type=Path, required=True)
    score.set_defaults(func=_score)

    source = subparsers.add_parser("aggregate-source")
    source.add_argument("--results-root", type=Path, required=True)
    source.add_argument("--output", type=Path, required=True)
    source.set_defaults(func=_aggregate_source)

    calibration = subparsers.add_parser("aggregate-calibration")
    calibration.add_argument("--source-gate", type=Path, required=True)
    calibration.add_argument("--results-root", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.set_defaults(func=_aggregate_calibration)

    target = subparsers.add_parser("aggregate-target")
    target.add_argument("--calibration-gate", type=Path, required=True)
    target.add_argument("--results-root", type=Path, required=True)
    target.add_argument("--output", type=Path, required=True)
    target.set_defaults(func=_aggregate_target)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
