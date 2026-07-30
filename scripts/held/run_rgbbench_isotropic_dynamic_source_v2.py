#!/usr/bin/env python3
"""Run the source-only RGBench isotropic dynamic-discrepancy study.

Simulation reads known physics and actuator trajectories but no object outcome.
``seal-bank`` may read only the allowed object prefix.  ``score-bank`` is the
first command allowed to open future object point clouds.
"""

from __future__ import annotations

import argparse
import csv
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
    GuardedReadoutCorrection,
    associate_dense_cloud,
    fit_guarded_readout_correction,
)
from bayesian_phystwin.rgbench_dynamic_belief import (
    RGBenchDynamicSlope,
    build_rgbbench_dynamic_candidates,
    fit_rgbbench_dynamic_slope,
    select_leave_one_garment_out_shrinkages,
)
from bayesian_phystwin.rgbench_isotropic_mesh import (
    RGBenchIsotropicMeshArtifact,
    RGBenchIsotropicMeshManifest,
    load_isotropic_mesh_manifest,
    write_json_once,
)
from bayesian_phystwin.rgbench_online_belief import (
    evaluation_pcd_paths,
    force_pybullet_direct_connection,
    load_obj_triangles,
    load_rgbbench_world_cloud,
    real_to_sim_l1_chamfer_m,
    sha256_file,
)
from bayesian_phystwin.rgbench_protocol import (
    ACTIONS,
    RGBENCH_COMMIT,
    SOURCE_GARMENTS,
)

METHOD_ID = "rgbbench-isotropic-dynamic-v2"
SIMULATOR = "pybullet"
MODE = "fixed_point"
SHRINKAGES = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{path} is not a JSON object")
    return payload


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(b"rgbbench-isotropic-dynamic-array-v2\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_dataset_manifest(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    _require(
        payload.get("artifact_kind") == "RGBenchDatasetManifest"
        and payload.get("rgbbench_commit") == RGBENCH_COMMIT,
        "dataset manifest provenance changed",
    )
    return payload


def _case_descriptor(
    manifest: dict[str, Any],
    case_id: str,
) -> dict[str, Any]:
    matches = [case for case in manifest["cases"] if case["case_id"] == case_id]
    _require(len(matches) == 1, f"manifest does not contain exactly one {case_id}")
    case = matches[0]
    _require(case["split"] == "source", "source runner refuses a non-source case")
    _require(
        case["garment"] in SOURCE_GARMENTS,
        "source garment differs from the frozen partition",
    )
    return case


def _mesh_artifact(
    manifest: RGBenchIsotropicMeshManifest,
    garment: str,
) -> RGBenchIsotropicMeshArtifact:
    matches = [
        artifact for artifact in manifest.artifacts if artifact.garment == garment
    ]
    _require(len(matches) == 1, f"mesh manifest does not contain one {garment}")
    return matches[0]


def _capture_root(dataset_root: Path, case: dict[str, Any]) -> Path:
    capture = dataset_root / case["data_subfolder"]
    _require(capture.is_dir(), f"capture does not exist: {capture}")
    return capture


def _case_pcd_paths(
    dataset_root: Path,
    case: dict[str, Any],
) -> tuple[Path, ...]:
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


def _compose_config(
    benchmark_root: Path,
    dataset_root: Path,
    case: dict[str, Any],
    mesh_path: Path,
    mesh_artifact: RGBenchIsotropicMeshArtifact,
) -> object:
    sys.path.insert(0, str(benchmark_root))
    hydra = importlib.import_module("hydra")
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
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str((benchmark_root / "configs").resolve()),
    ):
        config = hydra.compose(config_name="main", overrides=overrides)
    OmegaConf.resolve(config)
    active = config.active_run
    OmegaConf.set_readonly(active, False)
    active.cloth.model_path = str(mesh_path)
    active.cloth_params.shoulder_index = list(
        mesh_artifact.derived_fling_pin_indices
    )
    return active


def _simulate(args: argparse.Namespace) -> int:
    benchmark = args.benchmark_root.resolve()
    git_head = subprocess.check_output(
        ["git", "-C", str(benchmark), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    _require(git_head == RGBENCH_COMMIT, "RGBench checkout commit changed")
    dataset_manifest_path = args.dataset_manifest.resolve()
    dataset_manifest = _load_dataset_manifest(dataset_manifest_path)
    case = _case_descriptor(dataset_manifest, args.case_id)
    mesh_manifest_path = args.mesh_manifest.resolve()
    mesh_manifest = load_isotropic_mesh_manifest(mesh_manifest_path)
    _require(
        mesh_manifest.rgbbench_commit == RGBENCH_COMMIT
        and mesh_manifest.dataset_manifest_file_sha256
        == sha256_file(dataset_manifest_path)
        and mesh_manifest.dataset_manifest_artifact_sha256
        == dataset_manifest["artifact_sha256"],
        "mesh and dataset manifests do not share one frozen source",
    )
    mesh_artifact = _mesh_artifact(mesh_manifest, str(case["garment"]))
    mesh_path = mesh_manifest_path.parent / mesh_artifact.derived_mesh_relative_path
    _require(
        mesh_path.is_file()
        and sha256_file(mesh_path) == mesh_artifact.derived_mesh_sha256,
        "derived physical mesh changed",
    )
    dataset = args.dataset_root.resolve()
    paths = _case_pcd_paths(dataset, case)
    config = _compose_config(
        benchmark,
        dataset,
        case,
        mesh_path,
        mesh_artifact,
    )
    get_env = importlib.import_module("rgbench.envs").get_env
    pybullet = importlib.import_module("pybullet")
    with force_pybullet_direct_connection(pybullet):
        environment = get_env(config)
    connection_info = pybullet.getConnectionInfo(
        physicsClientId=environment.physics_client
    )
    _require(
        connection_info["connectionMethod"] == pybullet.DIRECT,
        "PyBullet connection was not forced to DIRECT",
    )
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
            environment.step_to_time(
                target_time
                + float(case["camera_delay_s"])
                + preparation_time
            )
            frame = np.asarray(environment.get_sim_vertices(), dtype=np.float64)
            _require(
                frame.ndim == 2
                and frame.shape[1] == 3
                and np.all(np.isfinite(frame)),
                "simulator returned invalid vertices",
            )
            vertices.append(frame.copy())
            target_times.append(target_time)
        anchor_indices = (
            int(environment.left_anchor_vertex),
            int(environment.right_anchor_vertex),
        )
    finally:
        environment.close()

    _, faces = load_obj_triangles(mesh_path)
    node_count = len(vertices[0])
    _require(
        node_count == mesh_artifact.derived_vertex_count
        and all(len(frame) == node_count for frame in vertices),
        "simulator node count differs from the derived mesh",
    )
    _require(
        int(np.max(faces)) < node_count
        and len(faces) == mesh_artifact.derived_face_count,
        "derived faces do not index the simulator vertices",
    )
    if case["action"] == "fling":
        _require(
            anchor_indices == mesh_artifact.derived_fling_pin_indices,
            "simulator did not use the bound fling contacts",
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
    write_json_once(
        output.with_suffix(".json"),
        {
            "schema_version": 1,
            "artifact_kind": "RGBenchIsotropicPhysicalBaselineV2",
            "method_id": METHOD_ID,
            "case_id": case["case_id"],
            "authorized_split": "source",
            "rgbbench_commit": RGBENCH_COMMIT,
            "dataset_revision": dataset_manifest["dataset_revision"],
            "simulator": SIMULATOR,
            "mode": MODE,
            "pybullet_connection_mode": "DIRECT",
            "upstream_gui_request_overridden": True,
            "node_count": node_count,
            "face_count": len(faces),
            "anchor_indices": list(anchor_indices),
            "mesh_mode": mesh_artifact.mode,
            "mesh_artifact_sha256": mesh_artifact.artifact_sha256,
            "mesh_manifest_sha256": sha256_file(mesh_manifest_path),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "evaluation_frame_count": len(vertices),
            "npz_path": str(output),
            "npz_sha256": sha256_file(output),
            "information_boundary": {
                "point_cloud_filenames_read": True,
                "point_cloud_coordinates_read": False,
                "known_future_actuator_trajectory_read": True,
                "future_object_outcomes_read": False,
            },
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


def _load_baseline(
    baseline_path: Path,
    *,
    case: dict[str, Any],
    mesh_manifest_path: Path,
    dataset_manifest_path: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    metadata_path = baseline_path.with_suffix(".json")
    _require(
        baseline_path.is_file() and metadata_path.is_file(),
        "physical baseline is incomplete",
    )
    metadata = _load_json(metadata_path)
    _require(
        metadata.get("artifact_kind") == "RGBenchIsotropicPhysicalBaselineV2"
        and metadata.get("method_id") == METHOD_ID
        and metadata.get("case_id") == case["case_id"]
        and metadata.get("mesh_manifest_sha256") == sha256_file(mesh_manifest_path)
        and metadata.get("dataset_manifest_sha256")
        == sha256_file(dataset_manifest_path)
        and metadata.get("npz_sha256") == sha256_file(baseline_path),
        "physical baseline provenance changed",
    )
    with np.load(baseline_path, allow_pickle=False) as archive:
        physical = np.asarray(archive["vertices_m"], dtype=np.float64)
        faces = np.asarray(archive["faces"], dtype=np.int64)
        target_times = np.asarray(archive["target_times_s"], dtype=np.float64)
    _require(
        physical.shape[0] == int(case["evaluation_frame_count"])
        and target_times.shape == (len(physical),),
        "physical baseline frame count changed",
    )
    return metadata, physical, faces, target_times


def _save_belief(
    path: Path,
    belief: GuardedReadoutCorrection,
    slope: RGBenchDynamicSlope | None,
) -> None:
    node_count = len(belief.correction_m)
    np.savez_compressed(
        path,
        accepted=np.asarray(belief.accepted),
        selected_name=np.asarray(belief.selected_name),
        correction_m=belief.correction_m,
        variance_m2=belief.variance_m2,
        slope_m_per_s=(
            np.zeros((node_count, 3), dtype=np.float64)
            if slope is None
            else slope.slope_m_per_s
        ),
        slope_variance_m2_per_s2=(
            np.full((node_count, 3), 1e-15, dtype=np.float64)
            if slope is None
            else slope.variance_m2_per_s2
        ),
    )


def _seal_bank(args: argparse.Namespace) -> int:
    dataset_manifest_path = args.dataset_manifest.resolve()
    manifest = _load_dataset_manifest(dataset_manifest_path)
    case = _case_descriptor(manifest, args.case_id)
    baseline_path = args.baseline.resolve()
    metadata, physical, faces, target_times = _load_baseline(
        baseline_path,
        case=case,
        mesh_manifest_path=args.mesh_manifest.resolve(),
        dataset_manifest_path=dataset_manifest_path,
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
    slope = (
        fit_rgbbench_dynamic_slope(
            physical[: branch + 1],
            prefix_clouds,
            target_times[: branch + 1],
            faces,
            belief,
            config=config,
        )
        if belief.accepted
        else None
    )
    candidates = build_rgbbench_dynamic_candidates(
        physical,
        target_times,
        branch,
        belief,
        slope,
        shrinkages=SHRINKAGES,
        maximum_correction_m=config.maximum_correction_m,
    )
    output = args.output_dir.resolve()
    _require(not output.exists(), f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    belief_path = output / "dynamic_belief.npz"
    _save_belief(belief_path, belief, slope)
    candidate_hashes = {
        f"{value:g}": _array_sha256(candidate.trajectory_m)
        for value, candidate in candidates.items()
    }
    _require(
        all(
            np.array_equal(candidate.trajectory_m[: branch + 1], physical[: branch + 1])
            for candidate in candidates.values()
        ),
        "a candidate changed the observed prefix",
    )
    if not belief.accepted:
        _require(
            all(
                candidate.exact_physical_fallback
                and np.array_equal(candidate.trajectory_m, physical)
                for candidate in candidates.values()
            ),
            "rejected belief did not return exact physical fallback",
        )
    write_json_once(
        output / "prediction_seal.json",
        {
            "schema_version": 1,
            "artifact_kind": "RGBenchDynamicPredictionBankSealV2",
            "method_id": METHOD_ID,
            "case_id": case["case_id"],
            "authorized_split": "source",
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "mesh_manifest_sha256": sha256_file(args.mesh_manifest.resolve()),
            "baseline_sha256": sha256_file(baseline_path),
            "baseline_metadata_sha256": sha256_file(
                baseline_path.with_suffix(".json")
            ),
            "physical_array_sha256": _array_sha256(physical),
            "target_times_array_sha256": _array_sha256(target_times),
            "fit_indices": [0, fit_stop - 1],
            "validation_indices": [fit_stop, branch],
            "future_indices": [branch + 1, len(physical) - 1],
            "shrinkages": list(SHRINKAGES),
            "candidate_array_sha256s": candidate_hashes,
            "static_belief": {
                "accepted": belief.accepted,
                "reason": belief.reason,
                "selected_name": belief.selected_name,
                "config": asdict(config),
                "scores": [asdict(score) for score in belief.scores],
                "diagnostics": belief.diagnostics,
            },
            "dynamic_slope": None
            if slope is None
            else {
                "spatial_model": slope.spatial_model,
                "diagnostics": slope.diagnostics,
            },
            "belief_sha256": sha256_file(belief_path),
            "physical_metadata": {
                "node_count": metadata["node_count"],
                "mesh_mode": metadata["mesh_mode"],
                "mesh_artifact_sha256": metadata["mesh_artifact_sha256"],
            },
            "information_boundary": {
                "prefix_point_cloud_coordinates_read": True,
                "future_point_cloud_coordinates_read": False,
                "candidate_prefix_is_exact_physical_baseline": True,
                "shrinkage_selected_from_this_case_future": False,
            },
        },
    )
    return 0


def _load_sealed_bank(
    output_dir: Path,
    baseline_path: Path,
    *,
    case: dict[str, Any],
    mesh_manifest_path: Path,
    dataset_manifest_path: Path,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    dict[float, Any],
]:
    seal_path = output_dir / "prediction_seal.json"
    belief_path = output_dir / "dynamic_belief.npz"
    _require(seal_path.is_file() and belief_path.is_file(), "prediction is unsealed")
    seal = _load_json(seal_path)
    _require(
        seal.get("artifact_kind") == "RGBenchDynamicPredictionBankSealV2"
        and seal.get("method_id") == METHOD_ID
        and seal.get("case_id") == case["case_id"]
        and seal.get("dataset_manifest_sha256") == sha256_file(dataset_manifest_path)
        and seal.get("mesh_manifest_sha256") == sha256_file(mesh_manifest_path)
        and seal.get("baseline_sha256") == sha256_file(baseline_path)
        and seal.get("belief_sha256") == sha256_file(belief_path),
        "prediction seal provenance changed",
    )
    _, physical, _, target_times = _load_baseline(
        baseline_path,
        case=case,
        mesh_manifest_path=mesh_manifest_path,
        dataset_manifest_path=dataset_manifest_path,
    )
    _require(
        seal["physical_array_sha256"] == _array_sha256(physical)
        and seal["target_times_array_sha256"] == _array_sha256(target_times),
        "sealed physical arrays changed",
    )
    with np.load(belief_path, allow_pickle=False) as archive:
        accepted = bool(np.asarray(archive["accepted"]).item())
        selected_name = str(np.asarray(archive["selected_name"]).item())
        correction = np.asarray(archive["correction_m"], dtype=np.float64)
        variance = np.asarray(archive["variance_m2"], dtype=np.float64)
        slope_mean = np.asarray(archive["slope_m_per_s"], dtype=np.float64)
        slope_variance = np.asarray(
            archive["slope_variance_m2_per_s2"],
            dtype=np.float64,
        )
    belief = GuardedReadoutCorrection(
        accepted=accepted,
        reason=str(seal["static_belief"]["reason"]),
        selected_name=selected_name,
        correction_m=correction,
        variance_m2=variance,
        scores=(),
        diagnostics={},
    )
    slope = (
        RGBenchDynamicSlope(
            slope_m_per_s=slope_mean,
            variance_m2_per_s2=slope_variance,
            spatial_model=selected_name,
            diagnostics={},
        )
        if accepted
        else None
    )
    config = ClothReadoutBeliefConfig(**seal["static_belief"]["config"])
    candidates = build_rgbbench_dynamic_candidates(
        physical,
        target_times,
        int(case["branch_index"]),
        belief,
        slope,
        shrinkages=tuple(float(value) for value in seal["shrinkages"]),
        maximum_correction_m=config.maximum_correction_m,
    )
    _require(
        all(
            seal["candidate_array_sha256s"][f"{value:g}"]
            == _array_sha256(candidate.trajectory_m)
            for value, candidate in candidates.items()
        ),
        "candidate bank differs from its prediction seal",
    )
    return seal, physical, target_times, candidates


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


def _score_candidate(
    trajectory: np.ndarray,
    variance: np.ndarray,
    clouds: list[np.ndarray],
    *,
    physical_scores: np.ndarray,
    branch: int,
    config: ClothReadoutBeliefConfig,
) -> dict[str, Any]:
    primary = np.asarray(
        [
            real_to_sim_l1_chamfer_m(observed, prediction)
            for prediction, observed in zip(trajectory, clouds, strict=True)
        ],
        dtype=np.float64,
    )
    future = np.arange(branch + 1, len(trajectory))
    horizons = np.array_split(future, 3)
    coverage: list[float] = []
    nees: list[float] = []
    interval_width: list[float] = []
    for horizon, index in enumerate(future, start=1):
        association = associate_dense_cloud(
            trajectory[index],
            clouds[index],
            candidate_count=config.candidate_count,
            sensor_std_m=config.sensor_std_m,
        )
        total_variance = (
            variance[index]
            + horizon * config.forecast_process_std_m_per_sqrt_frame**2
            + association.variance_m2[:, None]
        )
        residual = association.observed_points_m - trajectory[index]
        half_width = 1.6448536269514722 * np.sqrt(total_variance)
        coverage.append(float(np.mean(np.abs(residual) <= half_width)))
        nees.append(float(np.mean(np.square(residual) / total_variance)))
        interval_width.append(float(np.mean(2.0 * half_width)))
    return {
        "full_real_to_sim_l1_m": float(np.mean(primary)),
        "future_real_to_sim_l1_m": float(np.mean(primary[future])),
        "full_relative_improvement_vs_physical": _relative_improvement(
            physical_scores,
            primary,
        ),
        "future_relative_improvement_vs_physical": _relative_improvement(
            physical_scores[future],
            primary[future],
        ),
        "full_frame_wins_vs_physical": int(np.sum(primary < physical_scores)),
        "raw_90_coordinate_coverage": float(np.mean(coverage)),
        "mean_coordinate_nees": float(np.mean(nees)),
        "mean_90_interval_width_m": float(np.mean(interval_width)),
        "horizons": [
            {
                "name": name,
                "real_to_sim_l1_m": float(np.mean(primary[indices])),
                "relative_improvement_vs_physical": _relative_improvement(
                    physical_scores[indices],
                    primary[indices],
                ),
            }
            for name, indices in zip(
                ("early", "middle", "late"),
                horizons,
                strict=True,
            )
        ],
    }


def _score_bank(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    result_path = output_dir / "bank_result.json"
    _require(not result_path.exists(), f"refusing to overwrite {result_path}")
    dataset_manifest_path = args.dataset_manifest.resolve()
    manifest = _load_dataset_manifest(dataset_manifest_path)
    case = _case_descriptor(manifest, args.case_id)
    baseline_path = args.baseline.resolve()
    seal, physical, _, candidates = _load_sealed_bank(
        output_dir,
        baseline_path,
        case=case,
        mesh_manifest_path=args.mesh_manifest.resolve(),
        dataset_manifest_path=dataset_manifest_path,
    )
    clouds = _load_world_clouds(
        args.dataset_root.resolve(),
        case,
        0,
        int(case["evaluation_frame_count"]),
    )
    physical_scores = np.asarray(
        [
            real_to_sim_l1_chamfer_m(observed, prediction)
            for prediction, observed in zip(physical, clouds, strict=True)
        ],
        dtype=np.float64,
    )
    branch = int(case["branch_index"])
    config = ClothReadoutBeliefConfig(**seal["static_belief"]["config"])
    arms = {
        f"{value:g}": _score_candidate(
            candidate.trajectory_m,
            candidate.variance_m2,
            clouds,
            physical_scores=physical_scores,
            branch=branch,
            config=config,
        )
        for value, candidate in candidates.items()
    }
    paper_csv = args.paper_baselines.resolve()
    _require(
        sha256_file(paper_csv) == manifest["paper_baselines_sha256"],
        "paper baseline CSV changed after locking",
    )
    paper = _paper_garment_dynamics(
        paper_csv,
        str(case["garment"]),
        str(case["action"]),
    )
    future = np.arange(branch + 1, len(physical))
    write_json_once(
        result_path,
        {
            "schema_version": 1,
            "artifact_kind": "RGBenchDynamicPredictionBankResultV2",
            "method_id": METHOD_ID,
            "case_id": case["case_id"],
            "garment": case["garment"],
            "action": case["action"],
            "sample": case["sample"],
            "authorized_split": "source",
            "prediction_seal_sha256": sha256_file(
                output_dir / "prediction_seal.json"
            ),
            "published_garment_dynamics_cell_real_to_sim_l1_m": paper,
            "physical_full_real_to_sim_l1_m": float(np.mean(physical_scores)),
            "physical_future_real_to_sim_l1_m": float(
                np.mean(physical_scores[future])
            ),
            "static_belief_accepted": seal["static_belief"]["accepted"],
            "arms": arms,
            "future_outcomes_read_only_after_prediction_bank_seal": True,
            "claim_boundary": (
                "source-only causal online continuation; published "
                "GarmentDynamics is an open-loop comparator with less information"
            ),
        },
    )
    return 0


def _load_source_results(
    root: Path,
) -> tuple[list[Path], list[dict[str, Any]]]:
    paths = sorted(root.glob("*/*/*/bank_result.json"))
    _require(len(paths) == 27, "source aggregate requires exactly 27 bank results")
    results = [_load_json(path) for path in paths]
    _require(
        {row.get("artifact_kind") for row in results}
        == {"RGBenchDynamicPredictionBankResultV2"}
        and {row.get("authorized_split") for row in results} == {"source"}
        and len({row["case_id"] for row in results}) == 27,
        "source result set is invalid",
    )
    return paths, results


def _aggregate_source(args: argparse.Namespace) -> int:
    paths, results = _load_source_results(args.results_root.resolve())
    score_rows = [
        {
            "garment": result["garment"],
            "action": result["action"],
            "sample": result["sample"],
            "shrinkage": shrinkage,
            "candidate_score_m": result["arms"][f"{shrinkage:g}"][
                "full_real_to_sim_l1_m"
            ],
        }
        for result in results
        for shrinkage in SHRINKAGES
    ]
    selections = select_leave_one_garment_out_shrinkages(
        score_rows,
        garments=SOURCE_GARMENTS,
        actions=ACTIONS,
        shrinkages=SHRINKAGES,
    )
    selected_cases: list[dict[str, Any]] = []
    for result in results:
        key = (str(result["garment"]), str(result["action"]))
        shrinkage = selections[key]
        selected = result["arms"][f"{shrinkage:g}"]
        selected_cases.append(
            {
                "case_id": result["case_id"],
                "garment": result["garment"],
                "action": result["action"],
                "sample": result["sample"],
                "selected_shrinkage": shrinkage,
                "physical_real_to_sim_l1_m": result[
                    "physical_full_real_to_sim_l1_m"
                ],
                "candidate_real_to_sim_l1_m": selected[
                    "full_real_to_sim_l1_m"
                ],
                "candidate_future_real_to_sim_l1_m": selected[
                    "future_real_to_sim_l1_m"
                ],
                "published_garment_dynamics_real_to_sim_l1_m": result[
                    "published_garment_dynamics_cell_real_to_sim_l1_m"
                ],
                "raw_90_coordinate_coverage": selected[
                    "raw_90_coordinate_coverage"
                ],
                "mean_coordinate_nees": selected["mean_coordinate_nees"],
                "horizons": selected["horizons"],
            }
        )

    cells: list[dict[str, Any]] = []
    for garment in SOURCE_GARMENTS:
        for action in ACTIONS:
            rows = [
                row
                for row in selected_cases
                if row["garment"] == garment and row["action"] == action
            ]
            _require(len(rows) == 3, f"{garment}/{action} lacks three samples")
            physical = float(
                np.mean([row["physical_real_to_sim_l1_m"] for row in rows])
            )
            candidate = float(
                np.mean([row["candidate_real_to_sim_l1_m"] for row in rows])
            )
            paper_values = {
                row["published_garment_dynamics_real_to_sim_l1_m"] for row in rows
            }
            _require(len(paper_values) == 1, "published cell score changed")
            paper = float(next(iter(paper_values)))
            cells.append(
                {
                    "garment": garment,
                    "action": action,
                    "selected_shrinkage": selections[(garment, action)],
                    "physical_real_to_sim_l1_m": physical,
                    "candidate_real_to_sim_l1_m": candidate,
                    "relative_improvement": (physical - candidate)
                    / max(physical, 1e-15),
                    "published_garment_dynamics_real_to_sim_l1_m": paper,
                    "candidate_beats_published_garment_dynamics": candidate < paper,
                }
            )
    garment_improvements = {
        garment: float(
            np.mean(
                [
                    cell["relative_improvement"]
                    for cell in cells
                    if cell["garment"] == garment
                ]
            )
        )
        for garment in SOURCE_GARMENTS
    }
    cell_improvements = np.asarray(
        [cell["relative_improvement"] for cell in cells],
        dtype=np.float64,
    )
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
        "object_action_balanced_improvement_at_least_5pct": bool(
            np.mean(cell_improvements) >= 0.05
        ),
        "all_three_source_garments_nonregressing": bool(
            all(value >= 0.0 for value in garment_improvements.values())
        ),
        "at_least_six_of_nine_cells_improve": bool(
            np.sum(cell_improvements > 0.0) >= 6
        ),
        "aggregate_below_published_garment_dynamics": candidate_mean < paper_mean,
        "at_least_six_of_nine_cells_beat_published_garment_dynamics": bool(
            sum(
                cell["candidate_beats_published_garment_dynamics"]
                for cell in cells
            )
            >= 6
        ),
    }
    deployment: dict[str, float] = {}
    for action in ACTIONS:
        means = [
            (
                float(
                    np.mean(
                        [
                            row["candidate_score_m"]
                            for row in score_rows
                            if row["action"] == action
                            and row["shrinkage"] == shrinkage
                        ]
                    )
                ),
                shrinkage,
            )
            for shrinkage in SHRINKAGES
        ]
        deployment[action] = min(means)[1]
    passed = all(gates.values())
    write_json_once(
        args.output.resolve(),
        {
            "schema_version": 1,
            "artifact_kind": "RGBenchDynamicSourceGateV2",
            "method_id": METHOD_ID,
            "result_sha256s": {
                str(path.relative_to(args.results_root)): sha256_file(path)
                for path in paths
            },
            "cross_fit": {
                "held_out_unit": "garment",
                "training_garment_count": 2,
                "selection_scope": "action-specific",
                "selections": {
                    f"{garment}/{action}": value
                    for (garment, action), value in sorted(selections.items())
                },
            },
            "deployment_shrinkages_by_action": deployment,
            "selected_cases": selected_cases,
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
            "gates": gates,
            "source_gate_passed": passed,
            "calibration_authorized": passed,
            "target_authorized": False,
            "claim_boundary": (
                "cross-fitted source online continuation; not an "
                "equal-information open-loop SOTA comparison"
            ),
        },
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--benchmark-root", type=Path, required=True)
    simulate.add_argument("--dataset-root", type=Path, required=True)
    simulate.add_argument("--dataset-manifest", type=Path, required=True)
    simulate.add_argument("--mesh-manifest", type=Path, required=True)
    simulate.add_argument("--case-id", required=True)
    simulate.add_argument("--output", type=Path, required=True)
    simulate.set_defaults(func=_simulate)

    seal = subparsers.add_parser("seal-bank")
    seal.add_argument("--dataset-root", type=Path, required=True)
    seal.add_argument("--dataset-manifest", type=Path, required=True)
    seal.add_argument("--mesh-manifest", type=Path, required=True)
    seal.add_argument("--case-id", required=True)
    seal.add_argument("--baseline", type=Path, required=True)
    seal.add_argument("--output-dir", type=Path, required=True)
    seal.set_defaults(func=_seal_bank)

    score = subparsers.add_parser("score-bank")
    score.add_argument("--dataset-root", type=Path, required=True)
    score.add_argument("--dataset-manifest", type=Path, required=True)
    score.add_argument("--mesh-manifest", type=Path, required=True)
    score.add_argument("--paper-baselines", type=Path, required=True)
    score.add_argument("--case-id", required=True)
    score.add_argument("--baseline", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.set_defaults(func=_score_bank)

    aggregate = subparsers.add_parser("aggregate-source")
    aggregate.add_argument("--results-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.set_defaults(func=_aggregate_source)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
