from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin.genesis_mpm_source_qualification_v1 import file_sha256
from bayesian_phystwin.genesis_mpm_source_value_v1 import (
    GRID_FILENAME,
    PREFIX_FILENAME,
    load_genesis_source_value_protocol_v1,
    marginal_energy_score_v1,
    score_genesis_source_value_future_v1,
    score_genesis_source_value_prefix_v1,
)
from bayesian_phystwin.physical_rollout_v1 import write_deterministic_npz

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/sota/genesis_mpm_zebra_source_value_v1.json"
PHYSICS_EVIDENCE = (
    ROOT / "results/sota/diagnostics/genesis_mpm_zebra_source_physics_v1"
)
VALUE_EVIDENCE = ROOT / "results/sota/diagnostics/genesis_mpm_zebra_source_value_v1"


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
        incumbent = _physical_archive(root / "incumbent.npz", _trajectory(slope_m=0.0007))
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
            }
        )

        member_records: list[dict[str, Any]] = []
        predictions: list[npt.NDArray[np.float32]] = []
        group_grid = grid_root / group_id
        group_grid.mkdir()
        for index, slope in enumerate(member_slopes):
            prediction = _trajectory(slope_m=slope)
            member = _physical_archive(group_grid / f"member-{index:02d}.npz", prediction)
            predictions.append(prediction)
            member_records.append(
                {
                    "candidate_index": index,
                    "young_modulus_pa": protocol_value["candidate"]["young_modulus_pa"][index],
                    "weight": protocol_value["candidate"]["weights"][index],
                    "physical_archive": f"{group_id}/{member.name}",
                    "physical_archive_sha256": file_sha256(member),
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
                "members": member_records,
                "ensemble_mean_archive": f"{group_id}/{mean_path.name}",
                "ensemble_mean_sha256": file_sha256(mean_path),
                "final_ensemble_spread_m": float(
                    np.sqrt(
                        np.mean(
                            np.var(stack[:, -1].astype(np.float64), axis=0, ddof=0)
                        )
                    )
                ),
            }
        )

    protocol_path = tmp_path / "protocol.json"
    write_atomic_json(protocol_value, protocol_path, overwrite=False)
    grid_identity: dict[str, Any] = {
        "schema": "bayesian-phystwin.genesis-mpm-source-value-grid",
        "schema_version": 1,
        "protocol_sha256": file_sha256(protocol_path),
        "qualification_artifact_id": protocol_value["qualification"][
            "qualification_artifact_id"
        ],
        "implementation": {
            "git_head": "1" * 40,
            "git_worktree_clean": True,
            "source_files": {
                "src/bayesian_phystwin/genesis_mpm_source_qualification_v1.py": "2"
                * 64,
                "src/bayesian_phystwin/genesis_mpm_source_value_v1.py": "3" * 64,
                "scripts/remote/run_genesis_mpm_source_value_v1.py": "4" * 64,
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
    protocol = load_genesis_source_value_protocol_v1(PROTOCOL)

    assert protocol.runtime_id == "aecd2a170f974a166495da0c8692631acebf09d7b605c4ec0f9621f49434132a"
    assert protocol.qualification_artifact_id == "775fc43876318d7f5f01603d48cd26017a98bc69558d483e91ad58e523e38fa3"
    assert [group.group_id for group in protocol.groups] == [
        "double_lift_zebra",
        "double_stretch_zebra",
    ]
    assert protocol.young_moduli_pa == (25000.0, 100000.0, 500000.0)
    assert np.isclose(sum(protocol.weights), 1.0)
    assert protocol.value["information_boundary"]["no_replacement"] is True


def test_retained_evidence_qualifies_physics_and_rejects_source_value() -> None:
    physics_path = PHYSICS_EVIDENCE / "result.json"
    qualification_path = PHYSICS_EVIDENCE / "material-backend-qualification.json"
    grid_path = VALUE_EVIDENCE / "grid.json"
    prefix_path = VALUE_EVIDENCE / "prefix-result.json"
    future_path = VALUE_EVIDENCE / "future-result.json"

    assert file_sha256(physics_path) == (
        "e7e3a8172a4760a8ebc8f9cda16812c811037674dc08a9e2dc0b4810d826b0da"
    )
    assert file_sha256(qualification_path) == (
        "cc263bb7890af19c1f7bdae40f6c5f701d90f105a1e9c70bf386d2accb39561d"
    )
    assert file_sha256(grid_path) == (
        "caf35f48bd570ebcac836b5ccd37b9a22dd559ec810460710f567042afa3e2db"
    )
    assert file_sha256(prefix_path) == (
        "657a3c2d72395f33a33e6dacdff2e619db4a12959a533a6f425ec572b6cf58d9"
    )
    assert file_sha256(future_path) == (
        "3eacaa761b0ee4148f9600a4212c3e714e9d173fa7f3b55e0fcf9488dcbd8e0d"
    )
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    prefix = json.loads(prefix_path.read_text(encoding="utf-8"))
    future = json.loads(future_path.read_text(encoding="utf-8"))
    assert physics["qualified"] is True
    assert physics["source_value_scoring_authorized"] is True
    assert prefix["validation_gate_passed"] is False
    assert prefix["future_scoring_authorized"] is False
    assert all(
        record["selection"] == "exact_incumbent_fallback"
        and record["selected_sha256"] == record["source_sha256"]
        for record in prefix["selected_predictions"]
    )
    assert future["status"] == "future-not-opened-validation-gate-failed"
    assert future["future_outcomes_read"] is False


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

    assert marginal_energy_score_v1(centered, outcome, valid) < marginal_energy_score_v1(
        biased, outcome, valid
    )


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


def test_passing_source_gate_scores_future_after_exact_selection(tmp_path: Path) -> None:
    paths = _synthetic_gate(tmp_path, truth_slope_m=0.001)
    prefix_root = tmp_path / "prefix-result"
    prefix = score_genesis_source_value_prefix_v1(
        protocol_path=paths["protocol"],
        group_roots=paths["group_roots"],
        matphys_paths=paths["matphys_paths"],
        grid_dir=paths["grid_root"],
        output_dir=prefix_root,
    )

    assert prefix["validation_gate_passed"] is True
    assert all(
        record["selection"] == "genesis_equal_ensemble_mean"
        for record in prefix["selected_predictions"]
    )
    future = score_genesis_source_value_future_v1(
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
    prefix = score_genesis_source_value_prefix_v1(
        protocol_path=paths["protocol"],
        group_roots=paths["group_roots"],
        matphys_paths=paths["matphys_paths"],
        grid_dir=paths["grid_root"],
        output_dir=prefix_root,
    )
    assert prefix["future_scoring_authorized"] is False
    protocol = load_genesis_source_value_protocol_v1(paths["protocol"])
    for group in protocol.groups:
        selected = prefix_root / group.group_id / "selected-physical-prediction.npz"
        incumbent = paths["group_roots"][group.group_id] / group.incumbent_relative_path
        assert selected.read_bytes() == incumbent.read_bytes()
        (paths["group_roots"][group.group_id] / group.future_outcomes_relative_path).unlink()

    future = score_genesis_source_value_future_v1(
        protocol_path=paths["protocol"],
        group_roots=paths["group_roots"],
        matphys_paths=paths["matphys_paths"],
        prefix_dir=prefix_root,
        grid_dir=paths["grid_root"],
        output_path=tmp_path / "future-result.json",
    )
    assert future["status"] == "future-not-opened-validation-gate-failed"
    assert future["future_outcomes_read"] is False


def test_rejects_rehashed_future_authorization_bit(tmp_path: Path) -> None:
    paths = _synthetic_gate(tmp_path, truth_slope_m=0.0001)
    prefix_root = tmp_path / "prefix-result"
    score_genesis_source_value_prefix_v1(
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
        score_genesis_source_value_future_v1(
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
        score_genesis_source_value_prefix_v1(
            protocol_path=paths["protocol"],
            group_roots=paths["group_roots"],
            matphys_paths=paths["matphys_paths"],
            grid_dir=paths["grid_root"],
            output_dir=tmp_path / "prefix-result",
        )
