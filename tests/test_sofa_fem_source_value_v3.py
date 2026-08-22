from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

import bayesian_phystwin.sofa_fem_source_value_v3 as value_module
from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin.physical_rollout_v1 import write_deterministic_npz
from bayesian_phystwin.sofa_fem_source_qualification_v3 import file_sha256
from bayesian_phystwin.sofa_fem_source_value_v3 import (
    GRID_FILENAME,
    GRID_SCHEMA,
    SOURCE_FILES,
    finalize_sofa_fem_source_value_pre_prefix_v3,
    generate_sofa_fem_source_value_predictions_v3,
    load_sofa_fem_source_value_protocol_v3,
    marginal_energy_score_v1,
    score_sofa_fem_source_value_future_v3,
    score_sofa_fem_source_value_prefix_v3,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/sota/sofa_fem_zebra_source_value_v3.json"
RUNNER = ROOT / "scripts/remote/run_sofa_fem_source_value_v3.py"
INTERRUPTION = (
    ROOT
    / "results/sota/diagnostics/sofa_fem_zebra_source_value_v3"
    / "launch-interruption-v1.json"
)


def _trajectory(
    *,
    slope_m: float,
    frame_count: int = 8,
) -> npt.NDArray[np.float32]:
    frame_zero = np.asarray([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float32)
    result = np.repeat(frame_zero[None], frame_count, axis=0)
    result[:, :, 0] += np.arange(frame_count, dtype=np.float32)[:, None] * slope_m
    return np.ascontiguousarray(result)


def _physical_archive(
    path: Path,
    prediction_m: npt.NDArray[np.float32],
) -> Path:
    frame_zero = np.ascontiguousarray(prediction_m[0])
    persistence = np.repeat(frame_zero[None], len(prediction_m), axis=0)
    return cast(
        Path,
        write_deterministic_npz(
            path,
            {
                "action_support": np.ones(prediction_m.shape[1], dtype=np.float32),
                "driven_readout_m": prediction_m,
                "frame_zero_points_m": frame_zero,
                "persistence_m": persistence,
                "prediction_m": prediction_m,
                "zero_action_readout_m": persistence.copy(),
            },
        ),
    )


def _outcome_archive(
    path: Path,
    *,
    truth: npt.NDArray[np.float32],
    indices: npt.NDArray[np.int32],
) -> Path:
    return cast(
        Path,
        write_deterministic_npz(
            path,
            {
                "frame_indices": indices,
                "object_points_m": np.ascontiguousarray(truth[indices]),
                "valid_mask": np.ones((len(indices), truth.shape[1]), dtype=np.bool_),
            },
        ),
    )


def _synthetic_gate(
    tmp_path: Path,
    *,
    truth_slope_m: float,
) -> dict[str, Any]:
    protocol_value: dict[str, Any] = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    group_roots: dict[str, Path] = {}
    outcome_roots: dict[str, Path] = {}
    member_slopes = (0.0009, 0.0010, 0.0011)
    grid_records: list[dict[str, Any]] = []
    grid_root = tmp_path / "grid"
    grid_root.mkdir()
    truth = _trajectory(slope_m=truth_slope_m)
    prefix_indices = np.asarray([1, 2, 3, 4], dtype=np.int32)
    future_indices = np.asarray([5, 6, 7], dtype=np.int32)

    for raw_group in protocol_value["source_groups"]:
        group_id = str(raw_group["group_id"])
        root = tmp_path / "source" / group_id
        outcome_root = tmp_path / "outcomes" / group_id
        bundle = outcome_root / "source-bundle"
        root.mkdir(parents=True)
        bundle.mkdir(parents=True)
        group_roots[group_id] = root
        outcome_roots[group_id] = outcome_root
        source_inputs = root / "source-inputs.npz"
        prepared = root / "prepared.npz"
        source_inputs.write_bytes(b"synthetic-source-inputs")
        prepared.write_bytes(b"synthetic-prepared-source")
        prefix = _outcome_archive(
            bundle / "prefix-outcomes.npz",
            truth=truth,
            indices=prefix_indices,
        )
        future = _outcome_archive(
            bundle / "future-outcomes.npz",
            truth=truth,
            indices=future_indices,
        )
        incumbent = _physical_archive(
            root / "incumbent.npz",
            _trajectory(slope_m=0.0007),
        )
        raw_group.update(
            {
                "source_inputs_relative_path": source_inputs.name,
                "source_inputs_sha256": file_sha256(source_inputs),
                "prepared_archive_relative_path": prepared.name,
                "prepared_archive_sha256": file_sha256(prepared),
                "prefix_outcomes_relative_path": "source-bundle/prefix-outcomes.npz",
                "prefix_outcomes_sha256": file_sha256(prefix),
                "future_outcomes_relative_path": "source-bundle/future-outcomes.npz",
                "future_outcomes_sha256": file_sha256(future),
                "incumbent_relative_path": incumbent.name,
                "incumbent_sha256": file_sha256(incumbent),
                "frame_count": 8,
                "material_node_count": 2,
                "controller_point_count": 1,
                "attached_node_count": 1,
                "tetrahedron_count": 1,
                "contact_patch_sizes": [1],
            }
        )

        group_dir = grid_root / group_id
        group_dir.mkdir()
        member_records: list[dict[str, Any]] = []
        predictions: list[npt.NDArray[np.float32]] = []
        for index, slope in enumerate(member_slopes):
            prediction = _trajectory(slope_m=slope)
            member = _physical_archive(
                group_dir / f"member-{index:02d}.npz",
                prediction,
            )
            predictions.append(prediction)
            member_records.append(
                {
                    "candidate_index": index,
                    "young_modulus_pa": protocol_value["candidate"]["young_modulus_pa"][
                        index
                    ],
                    "poisson_ratio": protocol_value["candidate"]["poisson_ratio"],
                    "weight": protocol_value["candidate"]["weights"][index],
                    "physical_archive": f"{group_id}/{member.name}",
                    "physical_archive_sha256": file_sha256(member),
                    "minimum_deformation_determinant": 1.0,
                    "maximum_deformation_determinant": 1.0,
                    "maximum_node_displacement_m": float(
                        np.max(np.linalg.norm(prediction - prediction[0][None], axis=2))
                    ),
                    "maximum_native_attachment_error_m": 0.0,
                    "maximum_world_attachment_approximation_error_m": 1.0e-12,
                    "maximum_world_point_approximation_error_m": 1.0e-12,
                    "minimum_continuation_deformation_determinant": 1.0,
                    "native_step_count": 7 * 32,
                    "scene_sha256": "1" * 64,
                    "schedule_sha256": "2" * 64,
                    "gauge_sha256": "3" * 64,
                    "status": "success",
                }
            )
        stack = np.stack(predictions)
        mean = np.ascontiguousarray(
            np.tensordot(
                np.asarray(protocol_value["candidate"]["weights"]),
                stack.astype(np.float64),
                axes=(0, 0),
            ),
            dtype=np.float32,
        )
        mean_path = _physical_archive(group_dir / "ensemble-mean.npz", mean)
        grid_records.append(
            {
                "group_id": group_id,
                "source_inputs_sha256": raw_group["source_inputs_sha256"],
                "prepared_archive_sha256": raw_group["prepared_archive_sha256"],
                "incumbent_sha256": raw_group["incumbent_sha256"],
                "tetrahedron_count": 1,
                "contact_patch_sizes": [1],
                "maximum_contact_projection_error_m": 0.0,
                "members": member_records,
                "ensemble_mean_archive": f"{group_id}/{mean_path.name}",
                "ensemble_mean_sha256": file_sha256(mean_path),
                "final_ensemble_spread_m": float(
                    np.sqrt(
                        np.mean(np.var(stack[:, -1].astype(np.float64), axis=0, ddof=0))
                    )
                ),
            }
        )

    protocol_path = tmp_path / "protocol.json"
    write_atomic_json(protocol_value, protocol_path, overwrite=False)
    grid_identity: dict[str, Any] = {
        "schema": GRID_SCHEMA,
        "schema_version": 3,
        "protocol_sha256": file_sha256(protocol_path),
        "qualification_artifact_id": protocol_value["qualification"][
            "qualification_artifact_id"
        ],
        "implementation": {
            "git_head": "4" * 40,
            "git_worktree_clean": True,
            "source_files": {path: "5" * 64 for path in sorted(SOURCE_FILES)},
        },
        "groups": grid_records,
        "successful_candidate_count_per_group": 3,
        "information_boundary": {
            "source_inputs_read": True,
            "prepared_source_archives_read": True,
            "incumbent_bytes_read_for_hash_binding": True,
            "incumbent_prediction_arrays_read": False,
            "prefix_outcomes_read": False,
            "future_outcomes_read": False,
            "target_or_held_out_artifact_read": False,
        },
    }
    write_atomic_json(
        {**grid_identity, "grid_id": content_id(grid_identity)},
        grid_root / GRID_FILENAME,
        overwrite=False,
    )
    return {
        "protocol": protocol_path,
        "group_roots": group_roots,
        "outcome_roots": outcome_roots,
        "grid_root": grid_root,
    }


def test_frozen_protocol_binds_qualified_sofa_material_ensemble() -> None:
    protocol = load_sofa_fem_source_value_protocol_v3(PROTOCOL)

    assert protocol.runtime_id == (
        "f46e53707317bc652499e2f3af5b860a330f9dbf06014364acd19ffaf8acca8e"
    )
    assert protocol.qualification_artifact_id == (
        "78f61268a753d95e59071592a0793ecb11d5aeb2fc10f39032095902bedb9fc9"
    )
    assert [group.group_id for group in protocol.groups] == [
        "double_lift_zebra",
        "double_stretch_zebra",
    ]
    assert protocol.young_moduli_pa == (25000.0, 100000.0, 500000.0)
    assert protocol.poisson_ratio == 0.3
    assert np.isclose(sum(protocol.weights), 1.0)
    assert "matphys" not in PROTOCOL.read_text(encoding="utf-8").lower()
    assert protocol.value["information_boundary"]["no_replacement"] is True


def test_preinitialization_interruption_authorizes_one_managed_recovery() -> None:
    receipt = json.loads(INTERRUPTION.read_text(encoding="utf-8"))

    assert receipt["protocol_sha256"] == file_sha256(PROTOCOL)
    assert receipt["interrupted_implementation_revision"] == (
        "4f9528a8ccb91a88c6b38817a1e171ed70055a10"
    )
    assert receipt["classification"] == (
        "pre-initialization-orchestration-interruption-no-scientific-execution"
    )
    assert receipt["observations"] == {
        "launcher_process_present": False,
        "predictor_process_present": False,
        "output_root_created": False,
        "native_prediction_archive_count": 0,
        "source_group_input_read": False,
        "native_source_replay_started": False,
        "source_outcome_read": False,
        "target_or_held_out_artifact_read": False,
    }
    recovery = receipt["recovery"]
    assert recovery["protocol_changed"] is False
    assert recovery["scientific_method_changed"] is False
    assert recovery["threshold_changed"] is False
    assert recovery["source_roster_changed"] is False
    assert recovery["outcome_information_used"] is False
    assert recovery["preserve_interrupted_artifacts"] is True
    assert recovery["execution_mode"] == "managed-foreground-session"
    assert recovery["exactly_one_managed_recovery_authorized"] is True
    assert recovery["no_further_retry"] is True


def test_prefix_requires_passing_pre_prefix_receipt_before_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _synthetic_gate(tmp_path, truth_slope_m=0.001)
    monkeypatch.setattr(
        value_module,
        "_load_outcomes",
        lambda *_args, **_kwargs: pytest.fail("outcome opened before pre-prefix gate"),
    )

    with pytest.raises(ValueError, match="ordinary file"):
        score_sofa_fem_source_value_prefix_v3(
            protocol_path=paths["protocol"],
            group_roots=paths["group_roots"],
            outcome_roots=paths["outcome_roots"],
            grid_dir=paths["grid_root"],
            pre_prefix_dir=tmp_path / "missing-pre-prefix",
            output_dir=tmp_path / "prefix",
        )


def test_passing_physical_and_value_gates_score_future_once(tmp_path: Path) -> None:
    paths = _synthetic_gate(tmp_path, truth_slope_m=0.001)
    pre_prefix_root = tmp_path / "pre-prefix"
    pre_prefix = finalize_sofa_fem_source_value_pre_prefix_v3(
        protocol_path=paths["protocol"],
        group_roots=paths["group_roots"],
        grid_dir=paths["grid_root"],
        output_dir=pre_prefix_root,
    )
    assert pre_prefix["physical_gate_passed"] is True
    assert pre_prefix["prefix_scoring_authorized"] is True

    prefix_root = tmp_path / "prefix"
    prefix = score_sofa_fem_source_value_prefix_v3(
        protocol_path=paths["protocol"],
        group_roots=paths["group_roots"],
        outcome_roots=paths["outcome_roots"],
        grid_dir=paths["grid_root"],
        pre_prefix_dir=pre_prefix_root,
        output_dir=prefix_root,
    )
    assert prefix["validation_gate_passed"] is True
    assert all(
        record["selection"] == "sofa_fem_equal_ensemble_mean"
        for record in prefix["selected_predictions"]
    )

    future = score_sofa_fem_source_value_future_v3(
        protocol_path=paths["protocol"],
        group_roots=paths["group_roots"],
        outcome_roots=paths["outcome_roots"],
        prefix_dir=prefix_root,
        grid_dir=paths["grid_root"],
        pre_prefix_dir=pre_prefix_root,
        output_path=tmp_path / "future.json",
    )
    assert future["status"] == "source-future-scored-after-passing-gate"
    assert future["future_outcomes_read"] is True
    assert future["equal_group_ratios"]["balanced_point_ratio_vs_persistence"] < 0.05


def test_physical_failure_freezes_fallback_without_opening_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _synthetic_gate(tmp_path, truth_slope_m=0.001)
    grid_path = paths["grid_root"] / GRID_FILENAME
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    grid["groups"][1]["members"][2]["minimum_deformation_determinant"] = 0.4
    grid["groups"][1]["members"][2]["minimum_continuation_deformation_determinant"] = (
        0.4
    )
    identity = dict(grid)
    identity.pop("grid_id")
    grid["grid_id"] = content_id(identity)
    grid_path.write_text(json.dumps(grid, sort_keys=True), encoding="utf-8")

    for outcome_root in paths["outcome_roots"].values():
        for name in ("prefix-outcomes.npz", "future-outcomes.npz"):
            (outcome_root / "source-bundle" / name).unlink()
    output = tmp_path / "pre-prefix"
    result = finalize_sofa_fem_source_value_pre_prefix_v3(
        protocol_path=paths["protocol"],
        group_roots=paths["group_roots"],
        grid_dir=paths["grid_root"],
        output_dir=output,
    )
    assert result["physical_gate_passed"] is False
    assert result["prefix_scoring_authorized"] is False
    protocol = load_sofa_fem_source_value_protocol_v3(paths["protocol"])
    for group in protocol.groups:
        selected = output / group.group_id / "selected-physical-prediction.npz"
        incumbent = paths["group_roots"][group.group_id] / group.incumbent_relative_path
        assert selected.read_bytes() == incumbent.read_bytes()

    monkeypatch.setattr(
        value_module,
        "_load_outcomes",
        lambda *_args, **_kwargs: pytest.fail("outcome opened after failed gate"),
    )
    with pytest.raises(ValueError, match="not authorized"):
        score_sofa_fem_source_value_prefix_v3(
            protocol_path=paths["protocol"],
            group_roots=paths["group_roots"],
            outcome_roots=paths["outcome_roots"],
            grid_dir=paths["grid_root"],
            pre_prefix_dir=output,
            output_dir=tmp_path / "prefix",
        )


def test_marginal_energy_score_rewards_centered_spread() -> None:
    outcome = np.zeros((1, 1, 3), dtype=np.float64)
    valid = np.ones((1, 1), dtype=np.bool_)
    biased = np.full((1, 1, 1, 3), 0.01, dtype=np.float64)
    centered = np.zeros((3, 1, 1, 3), dtype=np.float64)
    centered[0, 0, 0, 0] = -0.001
    centered[2, 0, 0, 0] = 0.001

    assert marginal_energy_score_v1(
        centered, outcome, valid
    ) < marginal_energy_score_v1(biased, outcome, valid)


def test_prediction_generator_seals_three_members_without_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value: dict[str, Any] = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    frame_count = 8
    roots: dict[str, Path] = {}
    physics_groups: list[Any] = []
    points = np.asarray([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float64)
    controller = np.zeros((frame_count, 1, 3), dtype=np.float32)
    controller[:, 0, 0] = np.arange(frame_count, dtype=np.float32) * 0.001
    arrays = {
        "frame_zero_points_m": points.astype(np.float32),
        "controller_points_m": controller,
        "attachment_indices": np.asarray([0], dtype=np.int32),
        "attachment_weights": np.asarray([[1.0]], dtype=np.float32),
        "action_support": np.ones(2, dtype=np.float32),
    }
    contact = SimpleNamespace(
        projected_targets_m=np.asarray(controller[:2], dtype=np.float64),
        patch_local_indices=(np.asarray([0], dtype=np.int64),),
    )
    prepared_value = SimpleNamespace(
        points_m=points,
        cells=np.asarray([[0, 0, 0, 0]], dtype=np.int32),
        attachment_indices=np.asarray([0], dtype=np.int64),
        contact=contact,
    )

    for raw_group in value["source_groups"]:
        group_id = str(raw_group["group_id"])
        root = tmp_path / "source" / group_id
        root.mkdir(parents=True)
        roots[group_id] = root
        source = root / "source.npz"
        prepared = root / "prepared.npz"
        source.write_bytes(b"source")
        prepared.write_bytes(b"prepared")
        incumbent = _physical_archive(
            root / "incumbent.npz",
            _trajectory(slope_m=0.0007, frame_count=frame_count),
        )
        raw_group.update(
            {
                "source_inputs_relative_path": source.name,
                "source_inputs_sha256": file_sha256(source),
                "prepared_archive_relative_path": prepared.name,
                "prepared_archive_sha256": file_sha256(prepared),
                "incumbent_relative_path": incumbent.name,
                "incumbent_sha256": file_sha256(incumbent),
                "frame_count": frame_count,
                "material_node_count": 2,
                "controller_point_count": 1,
                "attached_node_count": 1,
                "tetrahedron_count": 1,
                "contact_patch_sizes": [1],
            }
        )
        physics_groups.append(
            SimpleNamespace(
                group_id=group_id,
                source_inputs_relative_path=PurePosixPath(source.name),
                source_inputs_sha256=raw_group["source_inputs_sha256"],
                prepared_archive_relative_path=PurePosixPath(prepared.name),
                prepared_archive_sha256=raw_group["prepared_archive_sha256"],
                incumbent_relative_path=PurePosixPath(incumbent.name),
                incumbent_sha256=raw_group["incumbent_sha256"],
                frame_count=frame_count,
                material_node_count=2,
                controller_point_count=1,
                attached_node_count=1,
                tetrahedron_count=1,
                expected_contact_patch_sizes=(1,),
            )
        )

    physics_protocol = tmp_path / "physics-protocol.json"
    physics_result = tmp_path / "physics-result.json"
    qualification = tmp_path / "qualification.json"
    physics_protocol.write_text("{}\n", encoding="utf-8")
    qualification.write_text("{}\n", encoding="utf-8")
    physics_result.write_text(
        json.dumps(
            {
                "result_id": value["qualification"]["source_physics_result_id"],
                "qualified": True,
                "source_value_scoring_authorized": True,
                "information_boundary": {
                    "source_object_outcomes_read": False,
                    "target_or_held_out_artifact_read": False,
                },
            }
        ),
        encoding="utf-8",
    )
    value["qualification"].update(
        {
            "source_physics_protocol_sha256": file_sha256(physics_protocol),
            "source_physics_result_sha256": file_sha256(physics_result),
            "qualification_artifact_sha256": file_sha256(qualification),
        }
    )
    protocol = tmp_path / "protocol.json"
    write_atomic_json(value, protocol, overwrite=False)
    fake_physics = SimpleNamespace(
        runtime_id=value["qualification"]["runtime_id"],
        source_groups=tuple(physics_groups),
        simulation={
            "young_modulus_probe_low_pa": 25000.0,
            "young_modulus_pa": 100000.0,
            "young_modulus_probe_high_pa": 500000.0,
            "poisson_ratio": 0.3,
            "fps": 30.0,
            "base_interval_substeps": 32,
            "hard_minimum_deformation_determinant": 0.35,
            "qualification_frame_count": 2,
            "density_kg_m3": 1000.0,
            "rayleigh_stiffness": 0.1,
            "rayleigh_mass": 0.1,
            "canonical_rounding_m": 1.0e-11,
            "minimum_relative_eigengap": 1.0e-6,
        },
    )

    monkeypatch.setattr(
        value_module,
        "load_material_backend_qualification_v1",
        lambda _path: SimpleNamespace(
            artifact_id=value["qualification"]["qualification_artifact_id"]
        ),
    )
    monkeypatch.setattr(
        value_module,
        "require_qualified_material_backend_runtime",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        value_module,
        "load_sofa_source_physics_protocol_v3",
        lambda _path: fake_physics,
    )
    monkeypatch.setattr(
        value_module,
        "load_native_sofa_fem_modules_v1",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        value_module,
        "load_sofa_source_inputs_v3",
        lambda *_args, **_kwargs: arrays,
    )
    monkeypatch.setattr(
        value_module,
        "load_prepared_sofa_source_v3",
        lambda *_args, **_kwargs: prepared_value,
    )
    monkeypatch.setattr(
        value_module,
        "rigid_contact_projection_v1",
        lambda _points, _indices, targets, patches: SimpleNamespace(
            projected_targets_m=np.asarray(targets),
            patch_local_indices=patches,
        ),
    )

    def fake_replay(**kwargs: Any) -> SimpleNamespace:
        young = float(kwargs["young_modulus_pa"])
        slope = {25000.0: 0.0009, 100000.0: 0.001, 500000.0: 0.0011}[young]
        positions = _trajectory(slope_m=slope, frame_count=frame_count).astype(
            np.float64
        )
        return SimpleNamespace(
            positions_m=positions,
            deformation_determinants=np.ones((frame_count, 1), dtype=np.float64),
            minimum_continuation_deformation_determinant=1.0,
            maximum_attachment_error_m=0.0,
            maximum_world_attachment_approximation_error_m=1.0e-12,
            native_step_count=(frame_count - 1) * 32,
            scene_sha256="1" * 64,
            schedule_sha256="2" * 64,
            gauge_sha256="3" * 64,
        )

    monkeypatch.setattr(
        value_module,
        "run_sofa_fem_canonical_source_replay_v3",
        fake_replay,
    )
    monkeypatch.setattr(
        value_module,
        "_git_provenance",
        lambda _root: {
            "git_head": "4" * 40,
            "git_worktree_clean": True,
            "source_files": {path: "5" * 64 for path in sorted(SOURCE_FILES)},
        },
    )

    output = tmp_path / "grid"
    grid = generate_sofa_fem_source_value_predictions_v3(
        protocol_path=protocol,
        physics_protocol_path=physics_protocol,
        physics_result_path=physics_result,
        qualification_path=qualification,
        group_roots=roots,
        output_dir=output,
        repo_root=tmp_path,
        distribution_archive=tmp_path / "sofa.zip",
        sofa_root=tmp_path / "sofa",
    )
    assert grid["successful_candidate_count_per_group"] == 3
    assert all(len(group["members"]) == 3 for group in grid["groups"])
    assert grid["information_boundary"]["prefix_outcomes_read"] is False
    assert grid["information_boundary"]["future_outcomes_read"] is False
    assert grid["information_boundary"]["incumbent_prediction_arrays_read"] is False


def test_runner_help_imports_without_native_sofa() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "finalize-pre-prefix" in completed.stdout
    assert "score-future" in completed.stdout
