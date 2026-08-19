from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

import bayesian_phystwin.jax_fem_source_value_v1 as value_module
from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin.jax_fem_source_qualification_v1 import file_sha256
from bayesian_phystwin.jax_fem_source_value_v1 import (
    GRID_FILENAME,
    PRE_PREFIX_FILENAME,
    PREFIX_FILENAME,
    finalize_jax_fem_source_value_pre_prefix_v1,
    generate_jax_fem_source_value_predictions_v1,
    load_jax_fem_source_value_protocol_v1,
    marginal_energy_score_v1,
    score_jax_fem_source_value_future_v1,
    score_jax_fem_source_value_prefix_v1,
)
from bayesian_phystwin.physical_rollout_v1 import write_deterministic_npz

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/sota/jax_fem_zebra_source_value_v1.json"
PHYSICS_EVIDENCE = ROOT / "results/sota/diagnostics/jax_fem_zebra_source_physics_v1"
VALUE_EVIDENCE = ROOT / "results/sota/diagnostics/jax_fem_zebra_source_value_v1"
PHYSICS_PROTOCOL = ROOT / "configs/sota/jax_fem_zebra_source_physics_v1.json"


def _physical_archive(
    path: Path,
    prediction_m: npt.NDArray[np.float32],
) -> Path:
    frame_zero = np.ascontiguousarray(prediction_m[0])
    persistence = np.repeat(frame_zero[None], len(prediction_m), axis=0)
    arrays = {
        "action_support": np.ones((prediction_m.shape[1],), dtype=np.float32),
        "driven_readout_m": prediction_m,
        "frame_zero_points_m": frame_zero,
        "persistence_m": persistence,
        "prediction_m": prediction_m,
        "zero_action_readout_m": persistence.copy(),
    }
    return cast(Path, write_deterministic_npz(path, arrays))


def _trajectory(
    *,
    slope_m: float,
    frame_count: int = 8,
) -> npt.NDArray[np.float32]:
    frame_zero = np.asarray([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float32)
    result = np.repeat(frame_zero[None], frame_count, axis=0)
    result[:, :, 0] += np.arange(frame_count, dtype=np.float32)[:, None] * slope_m
    return np.ascontiguousarray(result)


def _write_outcome(
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
                "valid_mask": np.ones(
                    (len(indices), truth.shape[1]),
                    dtype=np.bool_,
                ),
            },
        ),
    )


def _source_inputs_archive(
    path: Path,
    *,
    frame_count: int,
) -> Path:
    points = np.asarray([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float32)
    controller: npt.NDArray[np.float32] = np.zeros(
        (frame_count, 1, 3), dtype=np.float32
    )
    controller[:, 0, 0] = np.arange(frame_count, dtype=np.float32) * 0.001
    return cast(
        Path,
        write_deterministic_npz(
            path,
            {
                "frame_zero_points_m": points,
                "controller_points_m": controller,
                "attachment_indices": np.asarray([0], dtype=np.int32),
                "attachment_weights": np.asarray([[1.0]], dtype=np.float32),
                "action_support": np.ones(2, dtype=np.float32),
            },
        ),
    )


def _synthetic_generation_inputs(
    tmp_path: Path,
) -> tuple[Path, dict[str, Path], dict[str, Path]]:
    value: dict[str, Any] = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    roots: dict[str, Path] = {}
    matphys_paths: dict[str, Path] = {}
    frame_count = 8
    for raw_group in value["source_groups"]:
        group_id = str(raw_group["group_id"])
        root = tmp_path / "source" / group_id
        root.mkdir(parents=True)
        roots[group_id] = root
        source = _source_inputs_archive(
            root / "source-inputs.npz",
            frame_count=frame_count,
        )
        incumbent = _physical_archive(
            root / "incumbent.npz",
            _trajectory(slope_m=0.0007, frame_count=frame_count),
        )
        matphys = _physical_archive(
            root / "matphys.npz",
            _trajectory(slope_m=0.0008, frame_count=frame_count),
        )
        matphys_paths[group_id] = matphys
        raw_group.update(
            {
                "source_inputs_relative_path": source.name,
                "source_inputs_sha256": file_sha256(source),
                "incumbent_relative_path": incumbent.name,
                "incumbent_sha256": file_sha256(incumbent),
                "matphys_sha256": file_sha256(matphys),
                "frame_count": frame_count,
                "material_particle_count": 2,
                "controller_point_count": 1,
                "attached_particle_count": 1,
                "base_cell_count": 1,
                "contact_patch_sizes": [1],
            }
        )
    protocol = tmp_path / "generation-protocol.json"
    write_atomic_json(value, protocol, overwrite=False)
    return protocol, roots, matphys_paths


def _fake_value_replay(**kwargs: Any) -> SimpleNamespace:
    points = np.asarray(kwargs["points_m"], dtype=np.float64)
    frame_count = len(kwargs["frame_indices"])
    poisson = float(kwargs["poisson_ratio"])
    slope = {0.2: 0.0009, 0.35: 0.001, 0.45: 0.0011}[poisson]
    positions = np.repeat(points[None], frame_count, axis=0)
    positions[:, :, 0] += np.arange(frame_count, dtype=np.float64)[:, None] * slope
    return SimpleNamespace(
        positions_m=np.ascontiguousarray(positions),
        deformation_determinants=np.ones((frame_count, 1), dtype=np.float64),
    )


def _synthetic_gate(
    tmp_path: Path,
    *,
    truth_slope_m: float,
) -> dict[str, Any]:
    protocol_value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    group_roots: dict[str, Path] = {}
    matphys_paths: dict[str, Path] = {}
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
        root.mkdir(parents=True)
        group_roots[group_id] = root
        source_inputs = root / "source-inputs.npz"
        source_inputs.write_bytes(b"synthetic-source-inputs")
        prefix = _write_outcome(
            root / "prefix-outcomes.npz",
            truth=truth,
            indices=prefix_indices,
        )
        future = _write_outcome(
            root / "future-outcomes.npz",
            truth=truth,
            indices=future_indices,
        )
        incumbent = _physical_archive(
            root / "incumbent.npz", _trajectory(slope_m=0.0007)
        )
        matphys = _physical_archive(root / "matphys.npz", _trajectory(slope_m=0.0008))
        matphys_paths[group_id] = matphys
        raw_group.update(
            {
                "source_inputs_relative_path": source_inputs.name,
                "source_inputs_sha256": file_sha256(source_inputs),
                "prefix_outcomes_relative_path": prefix.name,
                "prefix_outcomes_sha256": file_sha256(prefix),
                "future_outcomes_relative_path": future.name,
                "future_outcomes_sha256": file_sha256(future),
                "incumbent_relative_path": incumbent.name,
                "incumbent_sha256": file_sha256(incumbent),
                "matphys_sha256": file_sha256(matphys),
                "frame_count": 8,
                "material_particle_count": 2,
                "controller_point_count": 1,
                "attached_particle_count": 1,
                "base_cell_count": 1,
                "contact_patch_sizes": [1],
            }
        )

        member_records: list[dict[str, Any]] = []
        predictions: list[npt.NDArray[np.float32]] = []
        group_grid = grid_root / group_id
        group_grid.mkdir()
        for index, slope in enumerate(member_slopes):
            prediction = _trajectory(slope_m=slope)
            member = _physical_archive(
                group_grid / f"member-{index:02d}.npz", prediction
            )
            predictions.append(prediction)
            member_records.append(
                {
                    "candidate_index": index,
                    "poisson_ratio": protocol_value["candidate"]["poisson_ratio"][
                        index
                    ],
                    "young_modulus_pa": protocol_value["candidate"]["young_modulus_pa"],
                    "weight": protocol_value["candidate"]["weights"][index],
                    "physical_archive": f"{group_id}/{member.name}",
                    "physical_archive_sha256": file_sha256(member),
                    "minimum_deformation_determinant": 1.0,
                    "maximum_deformation_determinant": 1.0,
                    "maximum_node_displacement_m": float(
                        np.max(np.linalg.norm(prediction - prediction[0][None], axis=2))
                    ),
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
        mean_path = _physical_archive(group_grid / "ensemble-mean.npz", mean)
        grid_records.append(
            {
                "group_id": group_id,
                "source_inputs_sha256": raw_group["source_inputs_sha256"],
                "incumbent_sha256": raw_group["incumbent_sha256"],
                "matphys_sha256": raw_group["matphys_sha256"],
                "base_cell_count": 1,
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
        "schema": "bayesian-phystwin.jax-fem-source-value-grid",
        "schema_version": 1,
        "protocol_sha256": file_sha256(protocol_path),
        "qualification_artifact_id": protocol_value["qualification"][
            "qualification_artifact_id"
        ],
        "implementation": {
            "git_head": "1" * 40,
            "git_worktree_clean": True,
            "source_files": {
                "src/bayesian_phystwin/jax_fem_source_qualification_v1.py": "2" * 64,
                "src/bayesian_phystwin/jax_fem_source_value_v1.py": "3" * 64,
                "scripts/remote/run_jax_fem_source_value_v1.py": "4" * 64,
            },
        },
        "groups": grid_records,
        "successful_candidate_count_per_group": 3,
        "information_boundary": {
            "source_inputs_read": True,
            "incumbent_and_matphys_predictions_read": True,
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
        "matphys_paths": matphys_paths,
        "grid_root": grid_root,
    }


def test_source_value_protocol_freezes_qualification_and_two_groups() -> None:
    protocol = load_jax_fem_source_value_protocol_v1(PROTOCOL)

    assert (
        protocol.runtime_id
        == "20c46dfa402712247416730e82289d4d4cd46096cab8c15b49ddb84a69d02a81"
    )
    assert (
        protocol.qualification_artifact_id
        == "9173b58fb3048b8f6466536e0df64e693345665fe93610c90a6677a5f5905dbc"
    )
    assert [group.group_id for group in protocol.groups] == [
        "double_lift_zebra",
        "double_stretch_zebra",
    ]
    assert protocol.poisson_ratios == (0.2, 0.35, 0.45)
    assert protocol.young_modulus_pa == 100000.0
    assert np.isclose(sum(protocol.weights), 1.0)
    assert protocol.value["information_boundary"]["no_replacement"] is True


def test_retained_evidence_qualifies_physics_and_rejects_value_pre_prefix() -> None:
    physics_path = PHYSICS_EVIDENCE / "result.json"
    qualification_path = PHYSICS_EVIDENCE / "material-backend-qualification.json"
    grid_path = VALUE_EVIDENCE / "grid.json"
    rejection_path = VALUE_EVIDENCE / "pre-prefix-result.json"

    assert file_sha256(physics_path) == (
        "ec8c7cb9b9e1a7f833d7857fc51ae3f86d83175bad9336d423d6d8856cacfbcf"
    )
    assert file_sha256(qualification_path) == (
        "68140e971e6758e5f1be015a0f0606d3dbfea8f97dd1541f5cea972659d9361c"
    )
    assert file_sha256(grid_path) == (
        "3bb6bf8afa878e7fd262344d8cb4ec3260fd16303e2776d8f884cc0d9d675414"
    )
    assert file_sha256(rejection_path) == (
        "39cd7fdda39673f8fb102e452d19e1f37ac0b6d786fbd16c5a5d66e52610a019"
    )

    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
    assert physics["qualified"] is True
    assert physics["source_value_scoring_authorized"] is True
    assert grid["information_boundary"]["prefix_outcomes_read"] is False
    assert grid["information_boundary"]["future_outcomes_read"] is False
    assert rejection["result_id"] == content_id(
        {key: value for key, value in rejection.items() if key != "result_id"}
    )
    assert rejection["physical_gate_passed"] is False
    assert rejection["prefix_scoring_authorized"] is False
    assert rejection["physical_checks"] == {
        "full_horizon_contact_projection": True,
        "full_horizon_deformation_determinants": False,
        "full_horizon_node_displacement": False,
    }
    assert rejection["information_boundary"] == {
        "prefix_outcomes_read": False,
        "future_outcomes_read": False,
        "target_or_held_out_artifact_read": False,
    }
    assert all(
        record["selection"] == "exact_incumbent_fallback"
        and record["byte_exact_source"] is True
        and record["selected_sha256"] == record["source_sha256"]
        for record in rejection["selected_predictions"]
    )


def test_prediction_generator_consumes_qualification_without_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, roots, matphys_paths = _synthetic_generation_inputs(tmp_path)
    protocol_value = json.loads(protocol.read_text(encoding="utf-8"))
    source_groups = tuple(
        SimpleNamespace(
            group_id=group["group_id"],
            source_inputs_relative_path=Path(group["source_inputs_relative_path"]),
            source_inputs_sha256=group["source_inputs_sha256"],
            incumbent_relative_path=Path(group["incumbent_relative_path"]),
            incumbent_sha256=group["incumbent_sha256"],
            frame_count=group["frame_count"],
            material_node_count=group["material_particle_count"],
            controller_point_count=group["controller_point_count"],
            attached_node_count=group["attached_particle_count"],
            expected_base_cell_count=group["base_cell_count"],
            expected_contact_patch_sizes=tuple(group["contact_patch_sizes"]),
        )
        for group in protocol_value["source_groups"]
    )
    fake_physics = SimpleNamespace(
        runtime_id=protocol_value["qualification"]["runtime_id"],
        source_groups=source_groups,
        simulation={
            "contact_cluster_radius_m": 0.015,
            "base_mesh_max_edge_m": 0.025,
            "minimum_tetrahedron_shape_ratio": 0.0001,
            "low_poisson_ratio": 0.2,
            "base_poisson_ratio": 0.35,
            "high_poisson_ratio": 0.45,
            "young_modulus_pa": 100000.0,
        },
    )
    monkeypatch.setattr(
        value_module,
        "load_jax_fem_source_physics_protocol_v1",
        lambda _path: fake_physics,
    )
    monkeypatch.setattr(
        value_module, "_load_native_modules", lambda _protocol: object()
    )
    monkeypatch.setattr(
        value_module,
        "contact_patch_local_indices_v1",
        lambda *_args, **_kwargs: (np.asarray([0], dtype=np.int64),),
    )
    monkeypatch.setattr(
        value_module,
        "rigid_contact_projection_v1",
        lambda _points, _indices, raw_targets, patches: SimpleNamespace(
            projected_targets_m=np.asarray(raw_targets),
            patch_local_indices=patches,
        ),
    )
    monkeypatch.setattr(
        value_module,
        "build_tetrahedral_cells_v1",
        lambda *_args, **_kwargs: np.asarray([[0, 0, 0, 0]], dtype=np.int32),
    )
    monkeypatch.setattr(value_module, "_run_native_replay", _fake_value_replay)
    monkeypatch.setattr(
        value_module,
        "_git_provenance",
        lambda *_args, **_kwargs: {
            "git_head": "1" * 40,
            "git_worktree_clean": True,
            "source_files": {"synthetic.py": "2" * 64},
        },
    )

    output = tmp_path / "grid"
    grid = generate_jax_fem_source_value_predictions_v1(
        protocol_path=protocol,
        physics_protocol_path=PHYSICS_PROTOCOL,
        physics_result_path=PHYSICS_EVIDENCE / "result.json",
        qualification_path=PHYSICS_EVIDENCE / "material-backend-qualification.json",
        group_roots=roots,
        matphys_paths=matphys_paths,
        output_dir=output,
        repo_root=tmp_path,
    )

    assert grid["successful_candidate_count_per_group"] == 3
    assert len(grid["groups"]) == 2
    assert grid["information_boundary"] == {
        "source_inputs_read": True,
        "incumbent_and_matphys_predictions_read": True,
        "prefix_outcomes_read": False,
        "future_outcomes_read": False,
        "target_or_held_out_artifact_read": False,
    }
    for group in grid["groups"]:
        assert len(group["members"]) == 3
        assert group["final_ensemble_spread_m"] > 0.0
        mean_path = output / group["ensemble_mean_archive"]
        assert file_sha256(mean_path) == group["ensemble_mean_sha256"]

    with pytest.raises(FileExistsError):
        generate_jax_fem_source_value_predictions_v1(
            protocol_path=protocol,
            physics_protocol_path=PHYSICS_PROTOCOL,
            physics_result_path=PHYSICS_EVIDENCE / "result.json",
            qualification_path=(
                PHYSICS_EVIDENCE / "material-backend-qualification.json"
            ),
            group_roots=roots,
            matphys_paths=matphys_paths,
            output_dir=output,
            repo_root=tmp_path,
        )


def test_marginal_energy_score_rewards_centered_spread() -> None:
    outcome: npt.NDArray[np.float64] = np.zeros((1, 1, 3), dtype=np.float64)
    valid: npt.NDArray[np.bool_] = np.ones((1, 1), dtype=np.bool_)
    biased: npt.NDArray[np.float64] = np.full(
        (1, 1, 1, 3),
        0.01,
        dtype=np.float64,
    )
    centered: npt.NDArray[np.float64] = np.zeros(
        (3, 1, 1, 3),
        dtype=np.float64,
    )
    centered[0, 0, 0, 0] = -0.001
    centered[2, 0, 0, 0] = 0.001

    assert marginal_energy_score_v1(
        centered, outcome, valid
    ) < marginal_energy_score_v1(biased, outcome, valid)


def test_energy_score_rejects_shape_or_empty_support() -> None:
    outcome: npt.NDArray[np.float64] = np.zeros((2, 3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="samples"):
        marginal_energy_score_v1(
            np.zeros((2, 3, 3), dtype=np.float64),
            outcome,
            np.ones((2, 3), dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="no supported"):
        marginal_energy_score_v1(
            np.zeros((2, 2, 3, 3), dtype=np.float64),
            outcome,
            np.zeros((2, 3), dtype=np.bool_),
        )


def test_passing_source_gate_scores_future_after_exact_selection(
    tmp_path: Path,
) -> None:
    paths = _synthetic_gate(tmp_path, truth_slope_m=0.001)
    prefix_root = tmp_path / "prefix-result"
    prefix = score_jax_fem_source_value_prefix_v1(
        protocol_path=paths["protocol"],
        group_roots=paths["group_roots"],
        matphys_paths=paths["matphys_paths"],
        grid_dir=paths["grid_root"],
        output_dir=prefix_root,
    )

    assert prefix["validation_gate_passed"] is True
    assert all(
        record["selection"] == "jax_fem_equal_ensemble_mean"
        for record in prefix["selected_predictions"]
    )
    future = score_jax_fem_source_value_future_v1(
        protocol_path=paths["protocol"],
        group_roots=paths["group_roots"],
        matphys_paths=paths["matphys_paths"],
        prefix_dir=prefix_root,
        grid_dir=paths["grid_root"],
        output_path=tmp_path / "future-result.json",
    )
    assert future["status"] == "source-future-scored-after-passing-gate"
    assert future["future_outcomes_read"] is True
    assert future["equal_group_ratios"]["balanced_point_ratio_vs_persistence"] < 0.05


def test_failed_source_gate_falls_back_without_future_file(tmp_path: Path) -> None:
    paths = _synthetic_gate(tmp_path, truth_slope_m=0.0001)
    prefix_root = tmp_path / "prefix-result"
    prefix = score_jax_fem_source_value_prefix_v1(
        protocol_path=paths["protocol"],
        group_roots=paths["group_roots"],
        matphys_paths=paths["matphys_paths"],
        grid_dir=paths["grid_root"],
        output_dir=prefix_root,
    )
    assert prefix["future_scoring_authorized"] is False
    protocol = load_jax_fem_source_value_protocol_v1(paths["protocol"])
    for group in protocol.groups:
        selected = prefix_root / group.group_id / "selected-physical-prediction.npz"
        incumbent = paths["group_roots"][group.group_id] / group.incumbent_relative_path
        assert selected.read_bytes() == incumbent.read_bytes()
        (
            paths["group_roots"][group.group_id] / group.future_outcomes_relative_path
        ).unlink()

    future = score_jax_fem_source_value_future_v1(
        protocol_path=paths["protocol"],
        group_roots=paths["group_roots"],
        matphys_paths=paths["matphys_paths"],
        prefix_dir=prefix_root,
        grid_dir=paths["grid_root"],
        output_path=tmp_path / "future-result.json",
    )
    assert future["status"] == "future-not-opened-validation-gate-failed"
    assert future["future_outcomes_read"] is False


def test_pre_prefix_physical_failure_falls_back_without_outcomes(
    tmp_path: Path,
) -> None:
    paths = _synthetic_gate(tmp_path, truth_slope_m=0.001)
    protocol = load_jax_fem_source_value_protocol_v1(paths["protocol"])
    for group in protocol.groups:
        (
            paths["group_roots"][group.group_id] / group.prefix_outcomes_relative_path
        ).unlink()
        (
            paths["group_roots"][group.group_id] / group.future_outcomes_relative_path
        ).unlink()
    grid_path = paths["grid_root"] / GRID_FILENAME
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    grid["groups"][1]["members"][2]["minimum_deformation_determinant"] = -1.0
    identity = dict(grid)
    identity.pop("grid_id")
    grid["grid_id"] = content_id(identity)
    grid_path.write_text(json.dumps(grid, sort_keys=True), encoding="utf-8")

    output = tmp_path / "pre-prefix"
    result = finalize_jax_fem_source_value_pre_prefix_v1(
        protocol_path=paths["protocol"],
        group_roots=paths["group_roots"],
        grid_dir=paths["grid_root"],
        output_dir=output,
    )

    assert result["physical_gate_passed"] is False
    assert result["prefix_scoring_authorized"] is False
    assert result["information_boundary"]["prefix_outcomes_read"] is False
    assert (output / PRE_PREFIX_FILENAME).is_file()
    for group in protocol.groups:
        selected = output / group.group_id / "selected-physical-prediction.npz"
        incumbent = paths["group_roots"][group.group_id] / group.incumbent_relative_path
        assert selected.read_bytes() == incumbent.read_bytes()


def test_rejects_rehashed_future_authorization_bit(tmp_path: Path) -> None:
    paths = _synthetic_gate(tmp_path, truth_slope_m=0.0001)
    prefix_root = tmp_path / "prefix-result"
    score_jax_fem_source_value_prefix_v1(
        protocol_path=paths["protocol"],
        group_roots=paths["group_roots"],
        matphys_paths=paths["matphys_paths"],
        grid_dir=paths["grid_root"],
        output_dir=prefix_root,
    )
    prefix_path = prefix_root / PREFIX_FILENAME
    prefix = json.loads(prefix_path.read_text(encoding="utf-8"))
    prefix["future_scoring_authorized"] = True
    identity = dict(prefix)
    identity.pop("result_id")
    prefix["result_id"] = content_id(identity)
    prefix_path.write_text(json.dumps(prefix, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="future authorization differs"):
        score_jax_fem_source_value_future_v1(
            protocol_path=paths["protocol"],
            group_roots=paths["group_roots"],
            matphys_paths=paths["matphys_paths"],
            prefix_dir=prefix_root,
            grid_dir=paths["grid_root"],
            output_path=tmp_path / "future-result.json",
        )


def test_grid_member_parameters_are_bound_by_frozen_index(tmp_path: Path) -> None:
    paths = _synthetic_gate(tmp_path, truth_slope_m=0.001)
    grid_path = paths["grid_root"] / GRID_FILENAME
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    grid["groups"][0]["members"][0]["young_modulus_pa"] = 50000.0
    identity = dict(grid)
    identity.pop("grid_id")
    grid["grid_id"] = content_id(identity)
    grid_path.write_text(json.dumps(grid, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen ensemble"):
        score_jax_fem_source_value_prefix_v1(
            protocol_path=paths["protocol"],
            group_roots=paths["group_roots"],
            matphys_paths=paths["matphys_paths"],
            grid_dir=paths["grid_root"],
            output_dir=tmp_path / "prefix-result",
        )
