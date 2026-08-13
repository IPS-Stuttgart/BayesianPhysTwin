import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import write_atomic_json
from bayesian_phystwin.cli import matphys_backend as matphys_backend_cli
from bayesian_phystwin.cli.matphys_backend import main as matphys_backend_main
from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    PHYSICAL_ARRAY_NAMES,
)
from bayesian_phystwin.matphys_backend_v1 import (
    ARTIFACT_FILENAME,
    SELECTED_ARCHIVE_FILENAME,
    build_matphys_backend_gate,
    build_matphys_backend_proposal,
    materialize_matphys_backend,
    validate_matphys_backend_artifact,
    validate_matphys_backend_gate,
    validate_matphys_backend_proposal,
)


def _arrays(*, motion_m: float) -> dict[str, np.ndarray]:
    frame_zero = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        dtype=np.float32,
    )
    persistence = np.repeat(frame_zero[None], 4, axis=0)
    response = np.linspace(0.0, motion_m, 4, dtype=np.float32)[:, None, None]
    direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    prediction = persistence + response * direction
    prediction[0] = frame_zero
    return {
        "prediction_m": prediction,
        "persistence_m": persistence,
        "driven_readout_m": prediction.copy(),
        "zero_action_readout_m": persistence.copy(),
        "action_support": np.ones(len(frame_zero), dtype=np.float32),
        "frame_zero_points_m": frame_zero,
    }


def _archive(path: Path, *, motion_m: float) -> Path:
    arrays = _arrays(motion_m=motion_m)
    assert set(arrays) == PHYSICAL_ARRAY_NAMES
    np.savez_compressed(path, **arrays)
    return path


def _manifests(
    tmp_path: Path,
    *,
    candidate_metrics: dict[str, float] | None = None,
    identity_motion_m: float = 0.010,
    incumbent_archive: Path | None = None,
    candidate_archive: Path | None = None,
    identity_replay_archive: Path | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    incumbent_archive = incumbent_archive or _archive(
        tmp_path / "incumbent.npz", motion_m=0.010
    )
    candidate_archive = candidate_archive or _archive(
        tmp_path / "candidate.npz", motion_m=0.012
    )
    identity_replay_archive = identity_replay_archive or _archive(
        tmp_path / "identity.npz", motion_m=identity_motion_m
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"source-trained checkpoint")
    spring = tmp_path / "spring.npy"
    np.save(spring, np.array([10.0, 20.0, 30.0], dtype=np.float32))
    proposal = build_matphys_backend_proposal(
        source_revision="a" * 40,
        simulator_revision="b" * 40,
        target_object_id="target-object",
        training_object_ids=("source-a", "source-b"),
        target_evidence_end_frame_exclusive=2,
        proposal_strength=0.5,
        checkpoint_path=checkpoint,
        spring_field_path=spring,
        source_artifacts={"protocol/source.json": "c" * 64},
    )
    proposal_path = tmp_path / "proposal.json"
    write_atomic_json(proposal, proposal_path, overwrite=False)
    gate = build_matphys_backend_gate(
        proposal_id=proposal["proposal_id"],
        target_object_id="target-object",
        case_id="target-object-episode-1",
        validation_frame_range_half_open=(2, 3),
        future_frame_start=3,
        incumbent_archive_path=incumbent_archive,
        candidate_archive_path=candidate_archive,
        identity_replay_archive_path=identity_replay_archive,
        incumbent_metrics={
            "chamfer_distance_m": 0.010,
            "track_error_m": 0.020,
        },
        candidate_metrics=candidate_metrics
        or {"chamfer_distance_m": 0.008, "track_error_m": 0.018},
        minimum_relative_improvement=0.01,
        maximum_metric_regression=0.0,
        maximum_identity_replay_rmse_m=0.001,
        source_artifacts={"prefix/score.json": "d" * 64},
    )
    gate_path = tmp_path / "gate.json"
    write_atomic_json(gate, gate_path, overwrite=False)
    return (
        proposal_path,
        gate_path,
        incumbent_archive,
        candidate_archive,
        identity_replay_archive,
    )


def test_matphys_backend_accepts_safe_proposal_and_preserves_candidate_bytes(
    tmp_path: Path,
) -> None:
    proposal, gate, incumbent, candidate, identity = _manifests(tmp_path)

    artifact = materialize_matphys_backend(
        proposal_manifest_path=proposal,
        gate_manifest_path=gate,
        incumbent_archive_path=incumbent,
        candidate_archive_path=candidate,
        identity_replay_archive_path=identity,
        output_dir=tmp_path / "output",
    )

    assert artifact["candidate_accepted"] is True
    assert artifact["selected_backend"] == "matphys_warp_proposal"
    assert (tmp_path / "output" / SELECTED_ARCHIVE_FILENAME).read_bytes() == (
        candidate.read_bytes()
    )
    assert artifact["output"]["exact_incumbent_fallback_verified"] is False
    assert validate_matphys_backend_artifact(tmp_path / "output") == artifact


def test_unstable_identity_replay_forces_byte_exact_incumbent_fallback(
    tmp_path: Path,
) -> None:
    proposal, gate, incumbent, candidate, identity = _manifests(
        tmp_path, identity_motion_m=0.020
    )

    artifact = materialize_matphys_backend(
        proposal_manifest_path=proposal,
        gate_manifest_path=gate,
        incumbent_archive_path=incumbent,
        candidate_archive_path=candidate,
        identity_replay_archive_path=identity,
        output_dir=tmp_path / "output",
    )

    assert artifact["candidate_accepted"] is False
    assert artifact["selected_backend"] == "incumbent"
    assert artifact["selection"]["identity_replay_stable"] is False
    assert artifact["output"]["exact_incumbent_fallback_verified"] is True
    assert (tmp_path / "output" / SELECTED_ARCHIVE_FILENAME).read_bytes() == (
        incumbent.read_bytes()
    )


def test_metric_regression_forces_exact_fallback(tmp_path: Path) -> None:
    proposal_path, gate_path, incumbent, candidate, identity = _manifests(
        tmp_path,
        candidate_metrics={
            "chamfer_distance_m": 0.007,
            "track_error_m": 0.021,
        },
    )

    artifact = materialize_matphys_backend(
        proposal_manifest_path=proposal_path,
        gate_manifest_path=gate_path,
        incumbent_archive_path=incumbent,
        candidate_archive_path=candidate,
        identity_replay_archive_path=identity,
        output_dir=tmp_path / "output",
    )

    assert artifact["candidate_accepted"] is False
    assert (
        artifact["selection"]["decisions"]["matphys_warp_proposal"][
            "no_metric_regression"
        ]
        is False
    )
    assert (tmp_path / "output" / SELECTED_ARCHIVE_FILENAME).read_bytes() == (
        incumbent.read_bytes()
    )


def test_proposal_rejects_target_object_training_and_changed_source_bytes(
    tmp_path: Path,
) -> None:
    proposal_path, _, _, _, _ = _manifests(tmp_path)
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposal["training_object_ids"] = ["source-a", "target-object"]
    identity = {key: value for key, value in proposal.items() if key != "proposal_id"}
    from bayesian_phystwin._portable_contracts import content_id

    proposal["proposal_id"] = content_id(identity)
    with pytest.raises(ValueError, match="training includes the target"):
        validate_matphys_backend_proposal(proposal, verify_files=True)

    unchanged = json.loads(proposal_path.read_text(encoding="utf-8"))
    Path(unchanged["checkpoint"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint SHA-256 changed"):
        validate_matphys_backend_proposal(unchanged, verify_files=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("training_object_ids", "source-object", "JSON array"),
        ("target_evidence_end_frame_exclusive", -1, "nonnegative integer"),
        ("proposal_strength", True, "finite number"),
        ("proposal_strength", float("nan"), "finite number"),
        ("proposal_strength", 0.0, "must be >"),
        ("proposal_strength", 2.0, "must be <="),
    ],
)
def test_proposal_rejects_malformed_contract_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    proposal_path, _, _, _, _ = _manifests(tmp_path)
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposal[field] = value

    with pytest.raises(ValueError, match=message):
        validate_matphys_backend_proposal(proposal, verify_files=False)


def test_proposal_requires_a_json_object() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        validate_matphys_backend_proposal([], verify_files=False)


@pytest.mark.parametrize(
    "spring_values",
    [
        np.ones((2, 2), dtype=np.float32),
        np.ones(4, dtype=np.int64),
    ],
)
def test_proposal_rejects_invalid_spring_field(
    tmp_path: Path, spring_values: np.ndarray
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"source-trained checkpoint")
    spring = tmp_path / "spring.npy"
    np.save(spring, spring_values)

    with pytest.raises(ValueError, match="one-dimensional floating array"):
        build_matphys_backend_proposal(
            source_revision="a" * 40,
            simulator_revision="b" * 40,
            target_object_id="target-object",
            training_object_ids=("source-object",),
            target_evidence_end_frame_exclusive=2,
            proposal_strength=0.5,
            checkpoint_path=checkpoint,
            spring_field_path=spring,
            source_artifacts={"protocol/source.json": "c" * 64},
        )


def test_gate_rejects_overlap_and_future_opening(tmp_path: Path) -> None:
    proposal_path, gate_path, _, _, _ = _manifests(tmp_path)
    proposal = validate_matphys_backend_proposal(
        json.loads(proposal_path.read_text(encoding="utf-8")), verify_files=True
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["validation_frame_range_half_open"] = [1, 3]
    identity = {key: value for key, value in gate.items() if key != "gate_id"}
    from bayesian_phystwin._portable_contracts import content_id

    gate["gate_id"] = content_id(identity)
    with pytest.raises(ValueError, match="overlaps proposal fitting"):
        validate_matphys_backend_gate(gate, proposal=proposal)

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["future_outcomes_opened"] = True
    identity = {key: value for key, value in gate.items() if key != "gate_id"}
    gate["gate_id"] = content_id(identity)
    with pytest.raises(ValueError, match="after future outcomes opened"):
        validate_matphys_backend_gate(gate, proposal=proposal)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("validation_frame_range_half_open", [2], "exactly two"),
        ("validation_frame_range_half_open", [2, True], "integer frame"),
        ("validation_frame_range_half_open", [3, 2], "nonempty half-open"),
        ("future_frame_start", 0, "positive integer"),
    ],
)
def test_gate_rejects_malformed_frame_contracts(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    proposal_path, gate_path, _, _, _ = _manifests(tmp_path)
    proposal = validate_matphys_backend_proposal(
        json.loads(proposal_path.read_text(encoding="utf-8")), verify_files=False
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate[field] = value

    with pytest.raises(ValueError, match=message):
        validate_matphys_backend_gate(gate, proposal=proposal)


def test_cli_metric_loader_rejects_noncanonical_fields(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"chamfer_distance_m": 0.01}), encoding="utf-8")

    with pytest.raises(argparse.ArgumentTypeError, match="fields changed"):
        matphys_backend_cli._metrics(str(metrics))


def test_candidate_cannot_change_topology_support_contract(tmp_path: Path) -> None:
    incumbent = _archive(tmp_path / "incumbent.npz", motion_m=0.010)
    candidate_arrays = _arrays(motion_m=0.012)
    candidate_arrays["action_support"][0] = 0.5
    candidate = tmp_path / "candidate.npz"
    np.savez_compressed(candidate, **candidate_arrays)
    identity = _archive(tmp_path / "identity.npz", motion_m=0.010)
    proposal, gate, _, _, _ = _manifests(
        tmp_path,
        incumbent_archive=incumbent,
        candidate_archive=candidate,
        identity_replay_archive=identity,
    )

    with pytest.raises(ValueError, match="action_support contract"):
        materialize_matphys_backend(
            proposal_manifest_path=proposal,
            gate_manifest_path=gate,
            incumbent_archive_path=incumbent,
            candidate_archive_path=candidate,
            identity_replay_archive_path=identity,
            output_dir=tmp_path / "output",
        )


def test_artifact_mutation_is_detected(tmp_path: Path) -> None:
    proposal, gate, incumbent, candidate, identity = _manifests(tmp_path)
    output = tmp_path / "output"
    materialize_matphys_backend(
        proposal_manifest_path=proposal,
        gate_manifest_path=gate,
        incumbent_archive_path=incumbent,
        candidate_archive_path=candidate,
        identity_replay_archive_path=identity,
        output_dir=output,
    )
    artifact = json.loads((output / ARTIFACT_FILENAME).read_text(encoding="utf-8"))
    artifact["selected_backend"] = "incumbent"
    (output / ARTIFACT_FILENAME).write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="content identity changed"):
        validate_matphys_backend_artifact(output)


def test_checksum_manifest_mutation_is_detected(tmp_path: Path) -> None:
    proposal, gate, incumbent, candidate, identity = _manifests(tmp_path)
    output = tmp_path / "output"
    materialize_matphys_backend(
        proposal_manifest_path=proposal,
        gate_manifest_path=gate,
        incumbent_archive_path=incumbent,
        candidate_archive_path=candidate,
        identity_replay_archive_path=identity,
        output_dir=output,
    )
    (output / "SHA256SUMS").write_text("0" * 64 + "  wrong\n", encoding="ascii")

    with pytest.raises(ValueError, match="checksum manifest changed"):
        validate_matphys_backend_artifact(output)


def test_untracked_bundle_member_is_rejected(tmp_path: Path) -> None:
    proposal, gate, incumbent, candidate, identity = _manifests(tmp_path)
    output = tmp_path / "output"
    materialize_matphys_backend(
        proposal_manifest_path=proposal,
        gate_manifest_path=gate,
        incumbent_archive_path=incumbent,
        candidate_archive_path=candidate,
        identity_replay_archive_path=identity,
        output_dir=output,
    )
    (output / "untracked.txt").write_text("not part of the seal\n", encoding="utf-8")

    with pytest.raises(ValueError, match="root roster changed"):
        validate_matphys_backend_artifact(output)


def test_compact_provenance_symlink_is_rejected(tmp_path: Path) -> None:
    proposal, gate, incumbent, candidate, identity = _manifests(tmp_path)
    output = tmp_path / "output"
    materialize_matphys_backend(
        proposal_manifest_path=proposal,
        gate_manifest_path=gate,
        incumbent_archive_path=incumbent,
        candidate_archive_path=candidate,
        identity_replay_archive_path=identity,
        output_dir=output,
    )
    copied = output / "provenance" / "matphys-proposal.json"
    external = tmp_path / "external-proposal.json"
    external.write_bytes(copied.read_bytes())
    copied.unlink()
    copied.symlink_to(external)

    with pytest.raises(ValueError, match="ordinary non-symlink file"):
        validate_matphys_backend_artifact(output)


@pytest.mark.parametrize("name", [ARTIFACT_FILENAME, "SHA256SUMS"])
def test_top_level_metadata_symlink_is_rejected(tmp_path: Path, name: str) -> None:
    proposal, gate, incumbent, candidate, identity = _manifests(tmp_path)
    output = tmp_path / "output"
    materialize_matphys_backend(
        proposal_manifest_path=proposal,
        gate_manifest_path=gate,
        incumbent_archive_path=incumbent,
        candidate_archive_path=candidate,
        identity_replay_archive_path=identity,
        output_dir=output,
    )
    source = output / name
    external = tmp_path / f"external-{name}"
    external.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(external)

    with pytest.raises(ValueError, match="ordinary non-symlink file"):
        validate_matphys_backend_artifact(output)


def test_registered_cli_runs_proposal_gate_materialize_validate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    incumbent = _archive(tmp_path / "incumbent.npz", motion_m=0.010)
    candidate = _archive(tmp_path / "candidate.npz", motion_m=0.012)
    identity = _archive(tmp_path / "identity.npz", motion_m=0.010)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"source-trained checkpoint")
    spring = tmp_path / "spring.npy"
    np.save(spring, np.array([10.0, 20.0, 30.0], dtype=np.float32))
    proposal_sources = tmp_path / "proposal-sources.json"
    gate_sources = tmp_path / "gate-sources.json"
    incumbent_metrics = tmp_path / "incumbent-metrics.json"
    candidate_metrics = tmp_path / "candidate-metrics.json"
    proposal_sources.write_text(
        json.dumps({"protocol/source.json": "c" * 64}), encoding="utf-8"
    )
    gate_sources.write_text(
        json.dumps({"prefix/score.json": "d" * 64}), encoding="utf-8"
    )
    incumbent_metrics.write_text(
        json.dumps({"chamfer_distance_m": 0.010, "track_error_m": 0.020}),
        encoding="utf-8",
    )
    candidate_metrics.write_text(
        json.dumps({"chamfer_distance_m": 0.008, "track_error_m": 0.018}),
        encoding="utf-8",
    )
    proposal = tmp_path / "proposal.json"
    assert (
        matphys_backend_main(
            [
                "proposal",
                str(proposal),
                "--source-revision",
                "a" * 40,
                "--simulator-revision",
                "b" * 40,
                "--target-object-id",
                "target-object",
                "--training-object-id",
                "source-b",
                "--training-object-id",
                "source-a",
                "--target-evidence-end-frame-exclusive",
                "2",
                "--proposal-strength",
                "0.5",
                "--checkpoint",
                str(checkpoint),
                "--spring-field",
                str(spring),
                "--source-artifacts",
                str(proposal_sources),
            ]
        )
        == 0
    )
    proposal_id = json.loads(capsys.readouterr().out)["proposal_id"]
    gate = tmp_path / "gate.json"
    assert (
        matphys_backend_main(
            [
                "gate",
                str(gate),
                "--proposal-id",
                proposal_id,
                "--target-object-id",
                "target-object",
                "--case-id",
                "target-object-episode-1",
                "--validation-start",
                "2",
                "--validation-stop",
                "3",
                "--future-frame-start",
                "3",
                "--incumbent-archive",
                str(incumbent),
                "--candidate-archive",
                str(candidate),
                "--identity-replay-archive",
                str(identity),
                "--incumbent-metrics",
                str(incumbent_metrics),
                "--candidate-metrics",
                str(candidate_metrics),
                "--minimum-relative-improvement",
                "0.01",
                "--maximum-metric-regression",
                "0.0",
                "--maximum-identity-replay-rmse-m",
                "0.001",
                "--source-artifacts",
                str(gate_sources),
            ]
        )
        == 0
    )
    capsys.readouterr()
    output = tmp_path / "output"
    assert (
        matphys_backend_main(
            [
                "materialize",
                str(proposal),
                str(gate),
                str(incumbent),
                str(candidate),
                str(identity),
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["candidate_accepted"] is True
    assert matphys_backend_main(["validate", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["candidate_accepted"] is True
