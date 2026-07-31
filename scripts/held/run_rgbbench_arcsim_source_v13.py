#!/usr/bin/env python3
"""Run the frozen one-case RGBench ARCSim source-accuracy screen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.rgbench_arcsim import load_arcsim_vertices, run_arcsim_fling
from bayesian_phystwin.rgbench_online_belief import (
    real_to_sim_l1_chamfer_m,
    sha256_file,
)
from scripts.held.run_rgbbench_arcsim_competence_v8 import (
    _git_head,
    _load_case,
    _parameters,
    _require,
    _verify_source,
    _write_json_once,
)
from scripts.held.run_rgbbench_isotropic_dynamic_source_v2 import (
    _case_descriptor,
    _case_pcd_paths,
    _load_dataset_manifest,
    _load_world_clouds,
)

PROTOCOL_ID = "rgbbench-arcsim-dirichlet-source-v13"
ARTIFACT_KIND = "RGBenchARCSimSourceAccuracyProtocol"
PREDICTION_KIND = "RGBenchARCSimSourcePredictionV13"
RESULT_KIND = "RGBenchARCSimSourceAccuracyResultV13"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--protocol", type=Path, required=True)
    simulate.add_argument("--rgbbench-root", type=Path, required=True)
    simulate.add_argument("--dataset-root", type=Path, required=True)
    simulate.add_argument("--arcsim-root", type=Path, required=True)
    simulate.add_argument("--arcsim-archive", type=Path, required=True)
    simulate.add_argument("--qualification-root", type=Path, required=True)
    simulate.add_argument("--output", type=Path, required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--protocol", type=Path, required=True)
    score.add_argument("--dataset-root", type=Path, required=True)
    score.add_argument("--prediction", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        isinstance(payload, dict)
        and payload.get("protocol_id") == PROTOCOL_ID
        and payload.get("artifact_kind") == ARTIFACT_KIND,
        "ARCSim source protocol identity changed",
    )
    case = payload.get("source_case")
    gate = payload.get("source_gate")
    _require(isinstance(case, dict), "source case is missing")
    _require(isinstance(gate, dict), "source gate is missing")
    _require(
        float(gate["minimum_relative_improvement_vs_published"]) == 0.05,
        "source advancement margin changed",
    )
    return payload


def _mapped_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    mapped = dict(protocol)
    mapped["competence_case"] = protocol["source_case"]
    mapped["competence_gate"] = protocol["source_gate"]
    return mapped


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dataset_case(
    protocol: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    case_spec = protocol["source_case"]
    manifest_path = (
        _repository_root() / case_spec["dataset_manifest_relative_path"]
    ).resolve()
    _require(
        manifest_path.is_file()
        and sha256_file(manifest_path) == case_spec["dataset_manifest_sha256"],
        "dataset manifest changed",
    )
    manifest = _load_dataset_manifest(manifest_path)
    case = _case_descriptor(manifest, str(case_spec["case_id"]))
    bound_fields = {
        "case_id": case_spec["case_id"],
        "split": "source",
        "garment": case_spec["garment"],
        "action": case_spec["action"],
        "sample": case_spec["sample"],
        "data_subfolder": case_spec["data_subfolder"],
        "evaluation_frame_count": case_spec["evaluation_frame_count"],
        "point_cloud_name_sha256": case_spec["point_cloud_name_sha256"],
        "master_start_time_s": case_spec["master_start_time_s"],
        "camera_delay_s": case_spec["camera_delay_s"],
    }
    _require(
        all(case.get(key) == value for key, value in bound_fields.items()),
        "dataset case changed after locking",
    )
    return manifest_path, case


def _verify_qualification(
    protocol: dict[str, Any],
    root: Path,
) -> np.ndarray:
    evidence = protocol["qualification_evidence"]
    gate_path = root / "gate.json"
    replay_path = root / "replay_1.npy"
    replay_metadata_path = root / "replay_1.json"
    _require(
        gate_path.is_file()
        and replay_path.is_file()
        and replay_metadata_path.is_file(),
        "full-horizon qualification is incomplete",
    )
    _require(
        sha256_file(gate_path) == evidence["gate_sha256"]
        and sha256_file(replay_path) == evidence["replay_sha256"]
        and sha256_file(replay_metadata_path) == evidence["replay_metadata_sha256"],
        "full-horizon qualification evidence changed",
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    metadata = json.loads(replay_metadata_path.read_text(encoding="utf-8"))
    _require(
        gate.get("qualification_gate_passed") is True
        and gate.get("source_accuracy_outcomes_read") is False
        and metadata.get("source_accuracy_outcomes_read") is False,
        "qualification did not preserve the source-outcome boundary",
    )
    final = np.load(replay_path, allow_pickle=False)
    _require(
        final.shape == (int(protocol["source_case"]["expected_vertex_count"]), 3)
        and np.all(np.isfinite(final)),
        "qualified final state is invalid",
    )
    return final


def _target_times_and_indices(
    protocol: dict[str, Any],
    dataset_root: Path,
    case: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    paths = _case_pcd_paths(dataset_root, case)
    target_times = np.asarray(
        [
            float(path.name.removeprefix("pointcloud_").removesuffix("_segmented.pcd"))
            - float(case["master_start_time_s"])
            for path in paths
        ],
        dtype=np.float64,
    )
    case_spec = protocol["source_case"]
    simulation_times = (
        target_times
        + float(case["camera_delay_s"])
        + float(case_spec["prepare_time_s"])
        + float(case_spec["wait_time_s"])
    )
    timestep_s = float(protocol["physics"]["timestep_s"])
    indices = np.rint(simulation_times / timestep_s).astype(np.int64)
    reconstructed = indices.astype(np.float64) * timestep_s
    _require(
        len(indices) == int(case_spec["evaluation_frame_count"])
        and np.all(indices >= 0)
        and np.all(indices <= int(protocol["source_gate"]["expected_step_count"]))
        and float(np.max(np.abs(reconstructed - simulation_times)))
        <= float(case_spec["maximum_alignment_error_s"]),
        "point-cloud timestamps do not map to the frozen solver grid",
    )
    return target_times, indices


def _simulate(args: argparse.Namespace) -> int:
    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path)
    mapped = _mapped_protocol(protocol)
    rgbbench_root = args.rgbbench_root.resolve()
    dataset_root = args.dataset_root.resolve()
    arcsim_root = args.arcsim_root.resolve()
    paths = _verify_source(
        mapped,
        rgbbench_root=rgbbench_root,
        dataset_root=dataset_root,
        arcsim_root=arcsim_root,
        arcsim_archive=args.arcsim_archive.resolve(),
    )
    qualified_final = _verify_qualification(
        protocol,
        args.qualification_root.resolve(),
    )
    manifest_path, case = _dataset_case(protocol)
    target_times, solver_indices = _target_times_and_indices(
        protocol,
        dataset_root,
        case,
    )
    vertices, _, controller = _load_case(mapped, paths)
    output = args.output.resolve()
    _require(output.suffix == ".npz", "prediction output must be .npz")
    _require(not output.exists(), f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    workspace = output.with_suffix(".workspace")
    rollout = run_arcsim_fling(
        source_mesh_path=paths["mesh"],
        initial_vertices_m=vertices,
        controller=controller,
        parameters=_parameters(mapped),
        duration_s=float(protocol["source_case"]["full_horizon_duration_s"]),
        initial_pose_xyz_wxyz=tuple(
            float(value) for value in protocol["source_case"]["initial_pose_xyz_wxyz"]
        ),
        workspace=workspace,
        arcsim_root=arcsim_root,
        timeout_s=float(protocol["source_gate"]["maximum_replay_elapsed_s"]),
    )
    _require(
        np.array_equal(rollout.final_vertices_m, qualified_final),
        "source replay differs from the qualified deterministic trajectory",
    )
    predictions = np.stack(
        [
            load_arcsim_vertices(workspace / "output" / f"{int(index):04d}_00.obj")
            for index in solver_indices
        ]
    )
    _require(
        predictions.shape
        == (
            int(protocol["source_case"]["evaluation_frame_count"]),
            int(protocol["source_case"]["expected_vertex_count"]),
            3,
        )
        and np.all(np.isfinite(predictions)),
        "sampled ARCSim prediction is invalid",
    )
    np.savez_compressed(
        output,
        vertices_m=predictions,
        target_times_s=target_times,
        solver_indices=solver_indices,
    )
    _write_json_once(
        output.with_suffix(".json"),
        {
            "schema_version": 1,
            "artifact_kind": PREDICTION_KIND,
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256_file(protocol_path),
            "implementation_commit": _git_head(_repository_root()),
            "case_id": case["case_id"],
            "authorized_split": "source",
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "prediction_sha256": sha256_file(output),
            "evaluation_frame_count": len(predictions),
            "solver_indices": solver_indices.tolist(),
            "maximum_timestamp_alignment_error_s": float(
                np.max(
                    np.abs(
                        solver_indices.astype(np.float64)
                        * float(protocol["physics"]["timestep_s"])
                        - (
                            target_times
                            + float(case["camera_delay_s"])
                            + float(protocol["source_case"]["prepare_time_s"])
                            + float(protocol["source_case"]["wait_time_s"])
                        )
                    )
                )
            ),
            "maximum_pin_target_error_m": rollout.maximum_pin_target_error_m,
            "qualified_final_state_reproduced_exactly": True,
            "information_boundary": {
                "point_cloud_filenames_read": True,
                "point_cloud_coordinates_read": False,
                "source_accuracy_outcomes_read": False,
                "known_future_actuator_trajectory_read": True,
            },
        },
    )
    return 0


def _load_prediction(
    protocol_path: Path,
    protocol: dict[str, Any],
    prediction_path: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    metadata_path = prediction_path.with_suffix(".json")
    _require(
        prediction_path.is_file() and metadata_path.is_file(),
        "source prediction is incomplete",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _require(
        metadata.get("artifact_kind") == PREDICTION_KIND
        and metadata.get("protocol_id") == protocol["protocol_id"]
        and metadata.get("protocol_sha256") == sha256_file(protocol_path)
        and metadata.get("prediction_sha256") == sha256_file(prediction_path)
        and metadata.get("information_boundary", {}).get("point_cloud_coordinates_read")
        is False,
        "prediction seal changed",
    )
    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction = np.asarray(archive["vertices_m"], dtype=np.float64)
    _require(
        prediction.shape
        == (
            int(protocol["source_case"]["evaluation_frame_count"]),
            int(protocol["source_case"]["expected_vertex_count"]),
            3,
        )
        and np.all(np.isfinite(prediction)),
        "prediction array changed",
    )
    return metadata, prediction


def _verify_comparators(protocol: dict[str, Any]) -> dict[str, float]:
    comparator = protocol["comparators"]
    source_gate_path = (
        _repository_root() / comparator["source_gate_relative_path"]
    ).resolve()
    _require(
        source_gate_path.is_file()
        and sha256_file(source_gate_path) == comparator["source_gate_sha256"],
        "frozen comparator artifact changed",
    )
    payload = json.loads(source_gate_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload["selected_cases"]
        if row["case_id"] == protocol["source_case"]["case_id"]
    ]
    _require(len(rows) == 1, "frozen source comparator is ambiguous")
    row = rows[0]
    values = {
        "physical_real_to_sim_l1_m": float(row["physical_real_to_sim_l1_m"]),
        "selected_dynamic_real_to_sim_l1_m": float(row["candidate_real_to_sim_l1_m"]),
        "published_garment_dynamics_real_to_sim_l1_m": float(
            row["published_garment_dynamics_real_to_sim_l1_m"]
        ),
    }
    _require(
        all(
            math.isclose(
                values[name],
                float(comparator[name]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for name in values
        ),
        "registered comparator values changed",
    )
    return values


def _relative_improvement(reference: float, candidate: float) -> float:
    return float((reference - candidate) / max(reference, 1e-15))


def _score(args: argparse.Namespace) -> int:
    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path)
    output = args.output.resolve()
    _require(not output.exists(), f"refusing to overwrite {output}")
    metadata, prediction = _load_prediction(
        protocol_path,
        protocol,
        args.prediction.resolve(),
    )
    _, case = _dataset_case(protocol)
    clouds = _load_world_clouds(
        args.dataset_root.resolve(),
        case,
        0,
        int(case["evaluation_frame_count"]),
    )
    per_frame = np.asarray(
        [
            real_to_sim_l1_chamfer_m(observed, simulated)
            for simulated, observed in zip(prediction, clouds, strict=True)
        ],
        dtype=np.float64,
    )
    mean_error = float(np.mean(per_frame))
    comparators = _verify_comparators(protocol)
    improvements = {
        name: _relative_improvement(value, mean_error)
        for name, value in comparators.items()
    }
    minimum_published = float(
        protocol["source_gate"]["minimum_relative_improvement_vs_published"]
    )
    gates = {
        "beats_remeshed_physical_baseline": (
            mean_error < comparators["physical_real_to_sim_l1_m"]
        ),
        "beats_selected_dynamic_baseline": (
            mean_error < comparators["selected_dynamic_real_to_sim_l1_m"]
        ),
        "published_improvement_at_least_5pct": (
            improvements["published_garment_dynamics_real_to_sim_l1_m"]
            >= minimum_published
        ),
    }
    gates["all_passed"] = all(gates.values())
    horizons = np.array_split(np.arange(len(per_frame)), 3)
    _write_json_once(
        output,
        {
            "schema_version": 1,
            "artifact_kind": RESULT_KIND,
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256_file(protocol_path),
            "implementation_commit": _git_head(_repository_root()),
            "case_id": case["case_id"],
            "authorized_split": "source",
            "prediction_metadata_sha256": sha256_file(
                args.prediction.resolve().with_suffix(".json")
            ),
            "prediction_sha256": metadata["prediction_sha256"],
            "arcsim_real_to_sim_l1_m": mean_error,
            "arcsim_endpoint_real_to_sim_l1_m": float(per_frame[-1]),
            "comparators": comparators,
            "relative_improvements": improvements,
            "horizons": [
                {
                    "name": name,
                    "real_to_sim_l1_m": float(np.mean(per_frame[indices])),
                }
                for name, indices in zip(
                    ("early", "middle", "late"),
                    horizons,
                    strict=True,
                )
            ],
            "gates": gates,
            "decision": (
                "advance-to-frozen-27-case-source-protocol"
                if gates["all_passed"]
                else "close-arcsim-source-route"
            ),
            "information_boundary": {
                "prediction_sealed_before_point_cloud_coordinates": True,
                "source_point_cloud_coordinates_read_for_scoring": True,
                "calibration_outcomes_read": False,
                "target_outcomes_read": False,
            },
            "claim_boundary": (
                "one already-declared source-case advancement screen; not a "
                "benchmark or state-of-the-art claim"
            ),
        },
    )
    return 0 if gates["all_passed"] else 2


def main() -> None:
    args = _parse_args()
    if args.command == "simulate":
        raise SystemExit(_simulate(args))
    raise SystemExit(_score(args))


if __name__ == "__main__":
    main()
