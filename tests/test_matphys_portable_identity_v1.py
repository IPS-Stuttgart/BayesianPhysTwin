from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.matphys_portable_identity_v1 as portable_identity
from bayesian_phystwin._artifact_custody import (
    checksum_manifest_text,
    copy_file_exact,
    publish_staging_directory,
)
from bayesian_phystwin.matphys_official_producer_v1 import (
    MATPHYS_CAUSAL_PREFIX_MODE,
    MATPHYS_OFFICIAL_PIPELINE_COMPONENTS,
    MATPHYS_PUBLISHED_PARITY_MODE,
    materialize_matphys_official_producer,
    write_matphys_official_replay_input,
)
from bayesian_phystwin.matphys_portable_identity_v1 import (
    PORTABLE_IDENTITY_FILENAME,
    SOURCE_VERIFICATION_FILENAME,
    materialize_matphys_portable_identity,
    validate_matphys_portable_identity,
)


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
    driven_delta = np.array([0.0, 0.001, 0.0], dtype=np.float32)
    zero_delta = np.array([0.0, 0.0, 0.0002], dtype=np.float32)
    candidate_driven = base + ramp * driven_delta
    candidate_zero = base + ramp * zero_delta
    identity_driven = base + ramp * driven_delta * np.float32(0.8)
    identity_zero = base + ramp * zero_delta * np.float32(0.8)
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


def _materialize_source(
    root: Path,
    *,
    mode: str = MATPHYS_CAUSAL_PREFIX_MODE,
) -> tuple[Path, dict[str, Path], dict[str, object]]:
    root.mkdir(parents=True)
    replay = write_matphys_official_replay_input(root / "replay.npz", _replay_arrays())
    checkpoint = root / "checkpoint.pth"
    checkpoint.write_bytes(b"official MatPhys checkpoint")
    spring = root / "spring.npy"
    np.save(spring, np.array([100.0, 200.0, 300.0], dtype=np.float32))
    candidate_parameters = root / "candidate-parameters.pth"
    candidate_parameters.write_bytes(b"spring, contact, collision, damping")
    identity_parameters = root / "identity-parameters.json"
    identity_parameters.write_text(
        json.dumps({"proposal_strength": 0.0}), encoding="utf-8"
    )
    output = root / "producer"
    training = (
        ("source-a", "source-b")
        if mode == MATPHYS_CAUSAL_PREFIX_MODE
        else ("source-a", "target-object")
    )
    artifact = materialize_matphys_official_producer(
        replay_input_path=replay,
        checkpoint_path=checkpoint,
        spring_field_path=spring,
        candidate_parameter_path=candidate_parameters,
        identity_parameter_path=identity_parameters,
        output_dir=output,
        mode=mode,  # type: ignore[arg-type]
        source_revision="a" * 40,
        simulator_revision="b" * 40,
        case_id="target-object-episode-1",
        target_object_id="target-object",
        checkpoint_training_object_ids=training,
        target_fit_frame_range_half_open=(10, 12),
        future_frame_start=13,
        proposal_strength=1.0,
        pipeline_component_artifacts=_component_artifacts(),
        source_artifacts={"protocol/source.json": "f" * 64},
    )
    paths = {
        "replay_input": replay,
        "checkpoint": checkpoint,
        "spring_field": spring,
        "candidate_parameters": candidate_parameters,
        "identity_parameters": identity_parameters,
    }
    return output, paths, artifact


def test_portable_identity_is_independent_of_host_paths(tmp_path: Path) -> None:
    first_source, first_paths, first_v1 = _materialize_source(tmp_path / "first")
    second_source, second_paths, second_v1 = _materialize_source(tmp_path / "second")

    assert first_v1["artifact_id"] != second_v1["artifact_id"]
    first = materialize_matphys_portable_identity(
        first_source, tmp_path / "first-portable"
    )
    second = materialize_matphys_portable_identity(
        second_source, tmp_path / "second-portable"
    )

    first_certificate = first["portable_artifact"]
    second_certificate = second["portable_artifact"]
    first_receipt = first["source_verification"]
    second_receipt = second["source_verification"]
    assert first_certificate == second_certificate
    assert (
        first_certificate["portable_artifact_id"]
        == second_certificate["portable_artifact_id"]
    )
    assert first_receipt["host_status_id"] != second_receipt["host_status_id"]

    certificate_text = (
        tmp_path / "first-portable" / PORTABLE_IDENTITY_FILENAME
    ).read_text(encoding="utf-8")
    assert str((tmp_path / "first").resolve()) not in certificate_text
    for source_path in first_paths.values():
        assert str(source_path.resolve()) not in certificate_text
    assert str((tmp_path / "second").resolve()) not in certificate_text
    for source_path in second_paths.values():
        assert str(source_path.resolve()) not in certificate_text


def test_portable_bundle_can_move_without_changing_identity(tmp_path: Path) -> None:
    source, _, _ = _materialize_source(tmp_path / "source")
    portable = tmp_path / "portable"
    result = materialize_matphys_portable_identity(source, portable)
    moved = tmp_path / "moved-portable"
    shutil.copytree(portable, moved)

    moved_result = validate_matphys_portable_identity(moved)
    verified_moved_result = validate_matphys_portable_identity(
        moved, verify_sources=True
    )

    assert moved_result == result
    assert verified_moved_result == result


def test_source_mutation_only_breaks_optional_host_verification(
    tmp_path: Path,
) -> None:
    source, paths, _ = _materialize_source(tmp_path / "source")
    portable = tmp_path / "portable"
    result = materialize_matphys_portable_identity(source, portable)
    paths["checkpoint"].write_bytes(b"mutated checkpoint")

    assert validate_matphys_portable_identity(portable) == result
    with pytest.raises(ValueError, match="checkpoint (byte count|SHA-256) changed"):
        validate_matphys_portable_identity(portable, verify_sources=True)


def test_certificate_mutation_breaks_portable_custody(tmp_path: Path) -> None:
    source, _, _ = _materialize_source(tmp_path / "source")
    portable = tmp_path / "portable"
    materialize_matphys_portable_identity(source, portable)
    certificate_path = portable / PORTABLE_IDENTITY_FILENAME
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    certificate["inputs"]["checkpoint"]["byte_count"] += 1
    certificate_path.write_text(
        json.dumps(certificate, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"portable (?:proposal input identity differs from producer identity|artifact identity changed)",
    ):
        validate_matphys_portable_identity(portable)


def test_source_receipt_mutation_breaks_host_status_identity(tmp_path: Path) -> None:
    source, _, _ = _materialize_source(tmp_path / "source")
    portable = tmp_path / "portable"
    materialize_matphys_portable_identity(source, portable)
    receipt_path = portable / SOURCE_VERIFICATION_FILENAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_bundle"]["path"] = str(tmp_path / "another-location")
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="host source status identity changed"):
        validate_matphys_portable_identity(portable)


def test_published_parity_certificate_has_no_causal_proposal(tmp_path: Path) -> None:
    source, _, _ = _materialize_source(
        tmp_path / "source", mode=MATPHYS_PUBLISHED_PARITY_MODE
    )
    result = materialize_matphys_portable_identity(source, tmp_path / "portable")
    certificate = result["portable_artifact"]

    assert certificate["mode"] == MATPHYS_PUBLISHED_PARITY_MODE
    assert certificate["outputs"]["causal_proposal"] is None
    assert certificate["information_boundary"] == {
        "target_prefix_used_for_parameter_fit": True,
        "target_object_used_for_checkpoint_training": True,
        "target_future_observations_used": False,
        "future_outcomes_opened": False,
        "known_future_robot_action_used": True,
        "causal_backend_eligible": False,
        "published_benchmark_control_only": True,
    }


def test_portable_materialization_is_no_overwrite(tmp_path: Path) -> None:
    source, _, _ = _materialize_source(tmp_path / "source")
    output = tmp_path / "portable"
    materialize_matphys_portable_identity(source, output)

    with pytest.raises(FileExistsError):
        materialize_matphys_portable_identity(source, output)


def test_shared_artifact_custody_copy_and_overwrite_guards(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"portable-custody-payload")
    destination = tmp_path / "nested" / "copy.bin"

    copied = copy_file_exact(source, destination)
    assert copied == destination.resolve()
    assert copied.read_bytes() == source.read_bytes()

    with pytest.raises(FileExistsError):
        copy_file_exact(source, destination)

    staging = tmp_path / "staging"
    staging.mkdir()
    existing_output = tmp_path / "published"
    existing_output.mkdir()
    with pytest.raises(FileExistsError):
        publish_staging_directory(staging, existing_output)


def test_shared_artifact_custody_rejects_bad_checksum_rosters(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checksum-root"
    root.mkdir()
    (root / "payload.bin").write_bytes(b"payload")

    with pytest.raises(ValueError, match="nonempty string"):
        checksum_manifest_text(root, [""])
    with pytest.raises(ValueError, match="unique"):
        checksum_manifest_text(root, ["payload.bin", "payload.bin"])


def test_portable_identity_primitive_validators_fail_closed() -> None:
    with pytest.raises(ValueError, match="JSON object with string keys"):
        portable_identity._mapping({1: "bad"}, name="value")
    with pytest.raises(ValueError, match="JSON array"):
        portable_identity._sequence("not-an-array", name="value")
    with pytest.raises(ValueError, match="positive integer"):
        portable_identity._positive_integer(0, name="value")
    with pytest.raises(ValueError, match="nonnegative integer"):
        portable_identity._nonnegative_integer(-1, name="value")
    with pytest.raises(ValueError, match="finite number"):
        portable_identity._finite_number(True, name="value")
    with pytest.raises(ValueError, match="finite number"):
        portable_identity._finite_number(float("nan"), name="value")
    with pytest.raises(ValueError, match="exactly two integer indices"):
        portable_identity._frame_range([1], name="value")
    with pytest.raises(ValueError, match="nonempty nonnegative half-open range"):
        portable_identity._frame_range([2, 2], name="value")
