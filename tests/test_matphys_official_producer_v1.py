from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.cli.matphys_official_producer import (
    _component_artifacts as parse_component_artifacts,
)
from bayesian_phystwin.cli.matphys_official_producer import main as producer_main
from bayesian_phystwin.matphys_backend_v1 import (
    validate_matphys_backend_proposal,
)
from bayesian_phystwin.matphys_official_producer_v1 import (
    ARTIFACT_FILENAME,
    CANDIDATE_ARCHIVE_FILENAME,
    CAUSAL_PROPOSAL_FILENAME,
    IDENTITY_ARCHIVE_FILENAME,
    MATPHYS_CAUSAL_PREFIX_MODE,
    MATPHYS_OFFICIAL_PIPELINE_COMPONENTS,
    MATPHYS_PUBLISHED_PARITY_MODE,
    load_matphys_official_replay_input,
    materialize_matphys_official_producer,
    validate_matphys_official_producer_artifact,
    validate_matphys_official_replay_arrays,
    write_matphys_official_replay_input,
)
from bayesian_phystwin.physical_rollout_v1 import load_physical_rollout_archive


def _replay_arrays() -> dict[str, np.ndarray]:
    frame_zero = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.01, 0.00, 0.00],
            [0.02, 0.00, 0.00],
            [0.03, 0.00, 0.00],
            [0.04, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    base = np.repeat(frame_zero[None], 4, axis=0)
    ramp = np.arange(4, dtype=np.float32)[:, None, None]
    y = np.array([0.0, 0.001, 0.0], dtype=np.float32)
    z = np.array([0.0, 0.0, 0.0002], dtype=np.float32)
    candidate_driven = base + ramp * y
    candidate_zero = base + ramp * z
    identity_driven = base + ramp * y * np.float32(0.8)
    identity_zero = base + ramp * z * np.float32(0.8)
    for value in (
        candidate_driven,
        candidate_zero,
        identity_driven,
        identity_zero,
    ):
        value[0] = frame_zero
    return {
        "candidate_driven_state_m": candidate_driven,
        "candidate_zero_action_state_m": candidate_zero,
        "identity_driven_state_m": identity_driven,
        "identity_zero_action_state_m": identity_zero,
        "material_query_indices": np.array([1, 3, 4], dtype=np.int64),
        "action_support": np.array([1.0, 0.8, 0.2], dtype=np.float32),
        "frame_indices": np.array([10, 11, 12, 13], dtype=np.int64),
    }


def _component_artifacts() -> dict[str, str]:
    return {
        name: f"{index + 1:x}" * 64
        for index, name in enumerate(MATPHYS_OFFICIAL_PIPELINE_COMPONENTS)
    }


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    replay = write_matphys_official_replay_input(
        tmp_path / "replay.npz", _replay_arrays()
    )
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"official MatPhys checkpoint")
    spring = tmp_path / "spring.npy"
    np.save(spring, np.array([100.0, 200.0, 300.0], dtype=np.float32))
    candidate_parameters = tmp_path / "candidate-parameters.pth"
    candidate_parameters.write_bytes(b"spring, contact, collision, damping")
    identity_parameters = tmp_path / "identity-parameters.json"
    identity_parameters.write_text(
        json.dumps({"proposal_strength": 0.0}), encoding="utf-8"
    )
    return replay, checkpoint, spring, candidate_parameters, identity_parameters


def _materialize(
    tmp_path: Path,
    *,
    mode: str = MATPHYS_CAUSAL_PREFIX_MODE,
    training: tuple[str, ...] = ("source-a", "source-b"),
    fit_range: tuple[int, int] = (10, 12),
    future_start: int = 13,
) -> dict[str, object]:
    replay, checkpoint, spring, candidate_parameters, identity_parameters = _inputs(
        tmp_path
    )
    return materialize_matphys_official_producer(
        replay_input_path=replay,
        checkpoint_path=checkpoint,
        spring_field_path=spring,
        candidate_parameter_path=candidate_parameters,
        identity_parameter_path=identity_parameters,
        output_dir=tmp_path / "output",
        mode=mode,  # type: ignore[arg-type]
        source_revision="a" * 40,
        simulator_revision="b" * 40,
        case_id="target-object-episode-1",
        target_object_id="target-object",
        checkpoint_training_object_ids=training,
        target_fit_frame_range_half_open=fit_range,
        future_frame_start=future_start,
        proposal_strength=1.0,
        pipeline_component_artifacts=_component_artifacts(),
        source_artifacts={"protocol/source.json": "f" * 64},
    )


def test_causal_mode_builds_guard_compatible_proposal_and_physical_archives(
    tmp_path: Path,
) -> None:
    artifact = _materialize(tmp_path)

    assert artifact["mode"] == MATPHYS_CAUSAL_PREFIX_MODE
    assert artifact["information_boundary"] == {
        "target_prefix_used_for_parameter_fit": True,
        "target_object_used_for_checkpoint_training": False,
        "target_future_observations_used": False,
        "future_outcomes_opened": False,
        "known_future_robot_action_used": True,
        "causal_backend_eligible": True,
        "published_benchmark_control_only": False,
    }
    output = tmp_path / "output"
    proposal = validate_matphys_backend_proposal(
        json.loads((output / CAUSAL_PROPOSAL_FILENAME).read_text(encoding="utf-8")),
        verify_files=True,
    )
    assert proposal["target_object_id"] == "target-object"
    assert proposal["training_object_ids"] == ["source-a", "source-b"]
    assert proposal["target_evidence_end_frame_exclusive"] == 12

    candidate = load_physical_rollout_archive(output / CANDIDATE_ARCHIVE_FILENAME)
    identity = load_physical_rollout_archive(output / IDENTITY_ARCHIVE_FILENAME)
    source = _replay_arrays()
    indices = source["material_query_indices"]
    assert np.array_equal(
        candidate["prediction_m"], source["candidate_driven_state_m"][:, indices]
    )
    assert np.array_equal(
        identity["prediction_m"], source["identity_driven_state_m"][:, indices]
    )
    assert (
        validate_matphys_official_producer_artifact(output, verify_sources=True)
        == artifact
    )


def test_published_parity_is_explicitly_noncausal_and_has_no_proposal(
    tmp_path: Path,
) -> None:
    artifact = _materialize(
        tmp_path,
        mode=MATPHYS_PUBLISHED_PARITY_MODE,
        training=("source-a", "target-object"),
    )

    assert artifact["outputs"]["causal_proposal"] is None
    assert artifact["information_boundary"]["causal_backend_eligible"] is False
    assert artifact["information_boundary"]["published_benchmark_control_only"] is True
    assert not (tmp_path / "output" / CAUSAL_PROPOSAL_FILENAME).exists()
    assert validate_matphys_official_producer_artifact(tmp_path / "output") == artifact


@pytest.mark.parametrize(
    ("mode", "training", "message"),
    [
        (
            MATPHYS_CAUSAL_PREFIX_MODE,
            ("source-a", "target-object"),
            "target-excluded",
        ),
    ],
)
def test_modes_cannot_silently_exchange_training_regimes(
    tmp_path: Path,
    mode: str,
    training: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _materialize(tmp_path, mode=mode, training=training)
    assert not (tmp_path / "output").exists()


def test_published_control_may_use_an_object_disjoint_checkpoint(
    tmp_path: Path,
) -> None:
    artifact = _materialize(
        tmp_path,
        mode=MATPHYS_PUBLISHED_PARITY_MODE,
        training=("source-a", "source-b"),
    )
    assert (
        artifact["information_boundary"]["target_object_used_for_checkpoint_training"]
        is False
    )
    assert artifact["information_boundary"]["causal_backend_eligible"] is False


def test_target_fit_must_remain_before_registered_future(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="crosses future boundary"):
        _materialize(tmp_path, fit_range=(10, 13), future_start=12)


def test_causal_mode_reserves_a_disjoint_gate_frame(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="disjoint validation prefix"):
        _materialize(tmp_path, fit_range=(10, 13), future_start=13)


def test_fit_and_future_boundaries_must_be_replay_frames(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not represented by replay frames"):
        _materialize(tmp_path, fit_range=(9, 12), future_start=13)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("frame-zero", "frame-zero state identity"),
        ("frame-order", "strictly increasing"),
        ("duplicate-query", "unique and in range"),
    ],
)
def test_replay_input_rejects_identity_and_ordering_failures(
    mutation: str, message: str
) -> None:
    arrays = _replay_arrays()
    if mutation == "frame-zero":
        arrays["identity_driven_state_m"][0, 0, 0] += 1.0
    elif mutation == "frame-order":
        arrays["frame_indices"] = np.array([10, 12, 11, 13], dtype=np.int64)
    else:
        arrays["material_query_indices"] = np.array([1, 1, 4], dtype=np.int64)
    with pytest.raises(ValueError, match=message):
        validate_matphys_official_replay_arrays(arrays)


def test_source_mutation_is_detected_only_when_source_rehash_is_requested(
    tmp_path: Path,
) -> None:
    artifact = _materialize(tmp_path)
    replay_path = Path(artifact["replay_input"]["path"])
    replay_path.write_bytes(b"mutated")

    assert validate_matphys_official_producer_artifact(tmp_path / "output") == artifact
    with pytest.raises(ValueError, match="replay input SHA-256 changed"):
        validate_matphys_official_producer_artifact(
            tmp_path / "output", verify_sources=True
        )


def test_output_mutation_breaks_bundle_custody(tmp_path: Path) -> None:
    _materialize(tmp_path)
    candidate = tmp_path / "output" / CANDIDATE_ARCHIVE_FILENAME
    candidate.write_bytes(candidate.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="candidate archive SHA-256 changed"):
        validate_matphys_official_producer_artifact(tmp_path / "output")


def test_cli_builds_and_validates_published_control(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    replay, checkpoint, spring, candidate_parameters, identity_parameters = _inputs(
        tmp_path
    )
    components = tmp_path / "components.json"
    sources = tmp_path / "sources.json"
    components.write_text(json.dumps(_component_artifacts()), encoding="utf-8")
    sources.write_text(json.dumps({"protocol/source.json": "f" * 64}), encoding="utf-8")
    output = tmp_path / "cli-output"
    assert (
        producer_main(
            [
                "build",
                str(replay),
                str(checkpoint),
                str(spring),
                str(candidate_parameters),
                str(identity_parameters),
                str(output),
                "--mode",
                MATPHYS_PUBLISHED_PARITY_MODE,
                "--source-revision",
                "a" * 40,
                "--simulator-revision",
                "b" * 40,
                "--case-id",
                "target-object-episode-1",
                "--target-object-id",
                "target-object",
                "--checkpoint-training-object-id",
                "target-object",
                "--target-fit-start",
                "10",
                "--target-fit-stop",
                "12",
                "--future-frame-start",
                "13",
                "--proposal-strength",
                "1",
                "--pipeline-component-artifacts",
                str(components),
                "--source-artifacts",
                str(sources),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["mode"] == MATPHYS_PUBLISHED_PARITY_MODE
    assert producer_main(["validate", str(output), "--verify-sources"]) == 0
    assert json.loads(capsys.readouterr().out)["artifact_id"]


def test_cli_rejects_an_incomplete_pipeline_component_roster(tmp_path: Path) -> None:
    components = tmp_path / "components.json"
    components.write_text(json.dumps({"checkpoint": "a" * 64}), encoding="utf-8")

    with pytest.raises(argparse.ArgumentTypeError, match="component roster changed"):
        parse_component_artifacts(str(components))


def test_replay_archive_is_deterministic_and_no_pickle(tmp_path: Path) -> None:
    first = write_matphys_official_replay_input(
        tmp_path / "first.npz", _replay_arrays()
    )
    second = write_matphys_official_replay_input(
        tmp_path / "second.npz", _replay_arrays()
    )
    assert first.read_bytes() == second.read_bytes()
    _, loaded = load_matphys_official_replay_input(first)
    assert set(loaded) == set(_replay_arrays())
    assert (tmp_path / "output" / ARTIFACT_FILENAME).exists() is False
