from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin.cli.newton_mpm_backend import build_parser
from bayesian_phystwin.newton_mpm_backend_v1 import file_sha256
from bayesian_phystwin.newton_mpm_source_gate_v1 import (
    FUTURE_OUTCOME_FILENAME,
    FUTURE_RESULT_SCHEMA,
    GRID_MANIFEST_FILENAME,
    GRID_SCHEMA,
    PREFIX_RESULT_FILENAME,
    SELECTED_PHYSICAL_FILENAME,
    SOURCE_CUSTODY_FILENAME,
    SOURCE_INPUT_FILENAME,
    _coordinate_rmse,
    load_grid_manifest,
    load_source_inputs,
    load_source_protocol,
    prepare_source_case,
    score_future_if_authorized,
    score_prefix_gate,
)
from bayesian_phystwin.physical_rollout_v1 import write_deterministic_npz


def _pickle(path: Path, value: object) -> Path:
    with path.open("wb") as stream:
        pickle.dump(value, stream)
    return path


def _physical_archive(
    path: Path,
    frame_zero: np.ndarray,
    *,
    step_m: float,
) -> Path:
    frame_count = 5
    persistence = np.repeat(frame_zero[None], frame_count, axis=0)
    prediction = persistence.copy()
    for frame in range(frame_count):
        prediction[frame, :, 0] += np.float32(step_m * frame)
    prediction[0] = frame_zero
    support = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    write_deterministic_npz(
        path,
        {
            "prediction_m": prediction,
            "persistence_m": persistence,
            "driven_readout_m": prediction,
            "zero_action_readout_m": persistence,
            "action_support": support,
            "frame_zero_points_m": frame_zero,
        },
    )
    return path


def _source_fixture(tmp_path: Path) -> dict[str, Path]:
    observed_zero = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
        dtype=np.float32,
    )
    surface = np.array([[0.0, 0.01, 0.0]], dtype=np.float32)
    interior = np.array([[0.0, 0.0, 0.01]], dtype=np.float32)
    frame_zero = np.concatenate((observed_zero, surface, interior), axis=0)
    object_points = np.repeat(observed_zero[None], 5, axis=0)
    controllers = np.repeat(observed_zero[None], 5, axis=0)
    for frame in range(5):
        object_points[frame, :, 0] += np.float32(0.001 * frame)
        controllers[frame, :, 0] += np.float32(0.001 * frame)
    final_data = {
        "object_points": object_points,
        "object_visibilities": np.ones((5, 2), dtype=bool),
        "object_motions_valid": np.ones((5, 2), dtype=bool),
        "controller_points": controllers,
        "surface_points": surface,
        "interior_points": interior,
    }
    final_data_path = _pickle(tmp_path / "final-data.pkl", final_data)
    optimal_path = _pickle(
        tmp_path / "optimal.pkl",
        {"controller_radius": 0.004, "controller_max_neighbours": 2},
    )
    incumbent_path = _physical_archive(
        tmp_path / "incumbent.npz",
        frame_zero,
        step_m=0.0008,
    )
    matphys_path = _physical_archive(
        tmp_path / "matphys.npz",
        frame_zero,
        step_m=0.0009,
    )
    replay_path = tmp_path / "replay.json"
    replay_path.write_text('{"source_only":true}\n', encoding="utf-8")
    protocol_value: dict[str, Any] = {
        "schema": "bayesian-phystwin.newton-mpm-source-gate-protocol",
        "schema_version": 1,
        "protocol_id": "newton-mpm-test-source-v1",
        "case_id": "synthetic-source",
        "cohort_role": "already-open-development-source",
        "claim_boundary": "unit test",
        "information_boundary": {
            "frame_zero_object_geometry_allowed": True,
            "known_full_controller_trajectory_allowed": True,
            "optimal_contact_neighbour_settings_allowed": True,
            "fit_object_frames_half_open": [1, 3],
            "validation_object_frames_half_open": [3, 4],
            "future_object_frames_half_open": [4, 5],
            "future_open_only_after_validation_gate": True,
            "target_or_held_out_artifact_access_allowed": False,
        },
        "source_files": {
            "final_data": {"sha256": file_sha256(final_data_path)},
            "optimal_params": {"sha256": file_sha256(optimal_path)},
            "incumbent_physical": {"sha256": file_sha256(incumbent_path)},
            "matphys_physical": {"sha256": file_sha256(matphys_path)},
            "matphys_replay_result": {"sha256": file_sha256(replay_path)},
        },
        "geometry": {
            "coordinate_frame": "test-world-v1",
            "position_units": "m",
            "frame_count": 5,
            "observed_identity_count": 2,
            "surface_point_count": 1,
            "interior_point_count": 1,
            "material_particle_count": 4,
            "controller_point_count": 2,
            "structure_order": [
                "object_points_frame_zero",
                "surface_points",
                "interior_points",
            ],
        },
        "contact_mapping": {
            "rule": "official-phystwin-radius-neighbours-v1",
            "controller_radius_m": 0.004,
            "controller_max_neighbours": 2,
            "expected_controller_edge_count": 2,
            "expected_attached_material_particle_count": 2,
            "multi_controller_reduction": (
                "inverse-distance-normalized-per-material-particle"
            ),
            "minimum_distance_m": 1.0e-6,
        },
        "simulation": {
            "engine": "newton-implicit-mpm",
            "engine_version": "1.5.0",
            "warp_version": "1.16.0",
            "numpy_version": np.__version__,
            "scipy_version": "1.18.0",
            "fps": 30.0,
            "substeps": 4,
            "voxel_size_m": 0.02,
            "particle_radius_m": 0.003,
            "density_kg_m3": 1000.0,
            "poisson_ratio": 0.35,
            "gravity_m_s2": [0.0, 0.0, 0.0],
            "max_iterations": 50,
            "tolerance": 1.0e-5,
            "solver": "cr",
            "integration_scheme": "pic",
            "strain_basis": "P1d",
            "velocity_basis": "Q1",
        },
        "parameter_grid": [
            {"young_modulus_pa": 25000.0, "damping": 0.002},
            {"young_modulus_pa": 100000.0, "damping": 0.02},
        ],
        "selection": {
            "fit_metric": "test",
            "tie_break": ["lower young_modulus_pa", "lower damping"],
            "validation_gates": {
                "maximum_balanced_ratio_vs_persistence": 0.95,
                "maximum_identity_rmse_ratio_vs_incumbent": 1.1,
                "maximum_chamfer_ratio_vs_incumbent": 1.1,
                "maximum_zero_action_drift_m": 0.002,
                "maximum_replay_coordinate_rmse_m": 1.0e-7,
                "minimum_final_ensemble_spread_m": 0.0001,
                "maximum_final_ensemble_spread_m": 0.1,
            },
            "failure_policy": "byte-exact incumbent physical archive",
            "required_successful_candidates": 2,
            "no_replacement": True,
        },
        "metrics": {
            "identity": "test",
            "chamfer": "test",
            "aggregation": "equal frame within split",
            "units": "m",
        },
        "pre_full_horizon_feasibility": {
            "frames_simulated": 2,
            "future_object_observations_read": False,
            "finite": True,
            "attachment_material_particle_count": 2,
            "purpose": "unit test",
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol_value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "protocol": protocol_path,
        "final_data": final_data_path,
        "optimal": optimal_path,
        "incumbent": incumbent_path,
        "matphys": matphys_path,
        "replay": replay_path,
    }


def _prepare(tmp_path: Path) -> tuple[dict[str, Path], Path]:
    paths = _source_fixture(tmp_path)
    bundle = tmp_path / "source-bundle"
    prepare_source_case(
        protocol_path=paths["protocol"],
        final_data_path=paths["final_data"],
        optimal_params_path=paths["optimal"],
        incumbent_physical_path=paths["incumbent"],
        matphys_physical_path=paths["matphys"],
        matphys_replay_result_path=paths["replay"],
        output_dir=bundle,
    )
    return paths, bundle


def _grid_manifest(
    root: Path,
    *,
    protocol_path: Path,
    source_inputs_path: Path,
    successful: bool,
) -> Path:
    protocol = load_source_protocol(protocol_path)
    candidates: list[dict[str, Any]] = []
    for index, parameters in enumerate(protocol.value["parameter_grid"]):
        base = {"candidate_index": index, **parameters}
        if successful:
            directory = root / f"candidate-{index:02d}"
            directory.mkdir(parents=True)
            physical_path = _physical_archive(
                directory / "physical-prediction.npz",
                load_source_inputs(
                    source_inputs_path,
                    protocol=protocol,
                )["frame_zero_points_m"],
                step_m=0.001 if index == 0 else 0.0006,
            )
            candidates.append(
                {
                    **base,
                    "status": "success",
                    "physical_archive": (
                        f"candidate-{index:02d}/physical-prediction.npz"
                    ),
                    "physical_archive_sha256": file_sha256(physical_path),
                    "replay_coordinate_rmse_m": 0.0,
                    "maximum_zero_action_drift_m": 0.0,
                    "maximum_action_response_m": 0.004,
                }
            )
        else:
            candidates.append(
                {
                    **base,
                    "status": "technical_failure",
                    "error_type": "SyntheticFailure",
                    "error_message": "retained denominator",
                }
            )
    identity: dict[str, Any] = {
        "schema": GRID_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "source_inputs_sha256": file_sha256(source_inputs_path),
        "runtime": {
            "engine_version": "1.5.0",
            "warp_version": "1.16.0",
            "numpy_version": np.__version__,
            "scipy_version": "1.18.0",
            "python_version": "3.12.0",
            "device": "cuda:0",
            "device_name": "synthetic-test-device",
        },
        "implementation": {
            "git_head": "b" * 40,
            "git_worktree_clean": True,
            "source_files": {
                "src/bayesian_phystwin/_newton_mpm_source_runtime.py": "a" * 64,
                "src/bayesian_phystwin/newton_mpm_source_gate_v1.py": "a" * 64,
                "src/bayesian_phystwin/cli/newton_mpm_backend.py": "a" * 64,
            },
        },
        "information_boundary": {
            "frame_zero_geometry_read": True,
            "known_full_controller_action_read": True,
            "object_outcome_artifact_read": False,
            "target_or_held_out_artifact_read": False,
        },
        "candidates": candidates,
        "successful_candidate_count": len(candidates) if successful else 0,
        "technical_failure_count": 0 if successful else len(candidates),
        "final_ensemble_spread_m": 0.0002 if successful else 0.0,
    }
    manifest = {**identity, "grid_id": content_id(identity)}
    path = root / GRID_MANIFEST_FILENAME
    write_atomic_json(manifest, path, overwrite=False)
    return path


def test_prepares_disjoint_source_artifacts_and_exact_geometry(tmp_path: Path) -> None:
    paths, bundle = _prepare(tmp_path)
    protocol = load_source_protocol(paths["protocol"])
    inputs = load_source_inputs(bundle / SOURCE_INPUT_FILENAME, protocol=protocol)

    assert inputs["frame_zero_points_m"].shape == (4, 3)
    np.testing.assert_array_equal(inputs["attachment_indices"], [0, 1])
    np.testing.assert_array_equal(inputs["attachment_weights"], np.eye(2))
    custody = json.loads((bundle / SOURCE_CUSTODY_FILENAME).read_text(encoding="utf-8"))
    assert custody["information_boundary"]["target_or_held_out_artifact_read"] is False
    assert custody["mapping"]["controller_edge_count"] == 2


def test_successful_prefix_gate_then_scores_source_future(tmp_path: Path) -> None:
    paths, bundle = _prepare(tmp_path)
    grid_root = tmp_path / "grid"
    grid_root.mkdir()
    grid_path = _grid_manifest(
        grid_root,
        protocol_path=paths["protocol"],
        source_inputs_path=bundle / SOURCE_INPUT_FILENAME,
        successful=True,
    )
    prefix_root = tmp_path / "prefix-result"
    prefix = score_prefix_gate(
        protocol_path=paths["protocol"],
        source_bundle_dir=bundle,
        grid_manifest_path=grid_path,
        incumbent_physical_path=paths["incumbent"],
        matphys_physical_path=paths["matphys"],
        output_dir=prefix_root,
    )

    assert prefix["validation_gate_passed"] is True
    assert prefix["selected_candidate_index"] == 0
    assert "resolved_physical_archive" not in prefix["candidates"][0]
    assert (prefix_root / SELECTED_PHYSICAL_FILENAME).read_bytes() == (
        grid_root / "candidate-00" / "physical-prediction.npz"
    ).read_bytes()

    future = score_future_if_authorized(
        protocol_path=paths["protocol"],
        source_bundle_dir=bundle,
        prefix_result_dir=prefix_root,
        grid_manifest_path=grid_path,
        incumbent_physical_path=paths["incumbent"],
        matphys_physical_path=paths["matphys"],
        output_path=tmp_path / "future-result.json",
    )
    assert future["schema"] == FUTURE_RESULT_SCHEMA
    assert future["future_outcomes_read"] is True
    assert future["metrics"]["selected"]["identity_coordinate_rmse_m"] == 0.0


def test_failed_gate_uses_exact_fallback_without_future_file(tmp_path: Path) -> None:
    paths, bundle = _prepare(tmp_path)
    grid_root = tmp_path / "grid"
    grid_root.mkdir()
    grid_path = _grid_manifest(
        grid_root,
        protocol_path=paths["protocol"],
        source_inputs_path=bundle / SOURCE_INPUT_FILENAME,
        successful=False,
    )
    prefix_root = tmp_path / "prefix-result"
    prefix = score_prefix_gate(
        protocol_path=paths["protocol"],
        source_bundle_dir=bundle,
        grid_manifest_path=grid_path,
        incumbent_physical_path=paths["incumbent"],
        matphys_physical_path=paths["matphys"],
        output_dir=prefix_root,
    )
    assert prefix["future_scoring_authorized"] is False
    assert (prefix_root / SELECTED_PHYSICAL_FILENAME).read_bytes() == paths[
        "incumbent"
    ].read_bytes()
    (bundle / FUTURE_OUTCOME_FILENAME).unlink()

    future = score_future_if_authorized(
        protocol_path=paths["protocol"],
        source_bundle_dir=bundle,
        prefix_result_dir=prefix_root,
        grid_manifest_path=grid_path,
        incumbent_physical_path=paths["incumbent"],
        matphys_physical_path=paths["matphys"],
        output_path=tmp_path / "future-result.json",
    )
    assert future["status"] == "future-not-opened-validation-gate-failed"
    assert future["future_outcomes_read"] is False


def test_rejects_rehashed_authorization_bit_on_failed_gate(tmp_path: Path) -> None:
    paths, bundle = _prepare(tmp_path)
    grid_root = tmp_path / "grid"
    grid_root.mkdir()
    grid_path = _grid_manifest(
        grid_root,
        protocol_path=paths["protocol"],
        source_inputs_path=bundle / SOURCE_INPUT_FILENAME,
        successful=False,
    )
    prefix_root = tmp_path / "prefix-result"
    score_prefix_gate(
        protocol_path=paths["protocol"],
        source_bundle_dir=bundle,
        grid_manifest_path=grid_path,
        incumbent_physical_path=paths["incumbent"],
        matphys_physical_path=paths["matphys"],
        output_dir=prefix_root,
    )
    prefix_path = prefix_root / PREFIX_RESULT_FILENAME
    prefix = json.loads(prefix_path.read_text(encoding="utf-8"))
    prefix["future_scoring_authorized"] = True
    identity = dict(prefix)
    identity.pop("result_id")
    prefix["result_id"] = content_id(identity)
    prefix_path.write_text(json.dumps(prefix, sort_keys=True), encoding="utf-8")
    (bundle / FUTURE_OUTCOME_FILENAME).unlink()

    with pytest.raises(ValueError, match="future authorization differs"):
        score_future_if_authorized(
            protocol_path=paths["protocol"],
            source_bundle_dir=bundle,
            prefix_result_dir=prefix_root,
            grid_manifest_path=grid_path,
            incumbent_physical_path=paths["incumbent"],
            matphys_physical_path=paths["matphys"],
            output_path=tmp_path / "future-result.json",
        )


def test_grid_parameters_are_bound_by_frozen_index(tmp_path: Path) -> None:
    paths, bundle = _prepare(tmp_path)
    grid_root = tmp_path / "grid"
    grid_root.mkdir()
    grid_path = _grid_manifest(
        grid_root,
        protocol_path=paths["protocol"],
        source_inputs_path=bundle / SOURCE_INPUT_FILENAME,
        successful=True,
    )
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    grid["candidates"][0]["young_modulus_pa"] = 50000.0
    identity = dict(grid)
    identity.pop("grid_id")
    grid["grid_id"] = content_id(identity)
    grid_path.write_text(json.dumps(grid, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen parameter grid"):
        load_grid_manifest(grid_path, protocol=load_source_protocol(paths["protocol"]))


def test_coordinate_rmse_uses_equal_frame_aggregation() -> None:
    outcome: np.ndarray = np.zeros((2, 2, 3), dtype=np.float32)
    prediction = outcome.copy()
    prediction[0, 0, 0] = 3.0
    valid = np.array([[True, False], [True, True]])

    assert _coordinate_rmse(prediction, outcome, valid) == pytest.approx(
        np.sqrt(3.0) / 2.0
    )


def test_existing_newton_command_exposes_source_gate_stages() -> None:
    parser = build_parser()
    for command in (
        "source-prepare",
        "source-run-grid",
        "source-score-prefix",
        "source-score-future",
    ):
        with pytest.raises(SystemExit) as exit_info:
            parser.parse_args([command, "--help"])
        assert exit_info.value.code == 0
