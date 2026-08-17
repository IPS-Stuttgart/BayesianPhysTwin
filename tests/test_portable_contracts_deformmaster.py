from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.deformmaster_backend_v1 as backend_module
from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin.cli.deformmaster_backend import main as deformmaster_cli_main
from bayesian_phystwin.deformmaster_backend_v1 import (
    DEFORMMASTER_SOURCE_REPOSITORY,
    DEFORMMASTER_TRAINING_SCHEMA,
    PHYSICAL_ARCHIVE_FILENAME,
    file_sha256,
    load_deformmaster_surface_rollout,
    materialize_deformmaster_backend,
    seal_deformmaster_runtime_manifest,
    validate_deformmaster_backend,
    validate_deformmaster_runtime_manifest,
)
from bayesian_phystwin.physical_rollout_v1 import (
    load_physical_rollout_archive,
    write_deterministic_npz,
)

SOURCE_REVISION = "1" * 40
PRODUCER_REVISION = "2" * 40


def _inputs(
    tmp_path: Path,
    *,
    training_object_ids: list[str] | None = None,
) -> dict[str, Path]:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen-deformmaster-checkpoint")
    configuration = tmp_path / "config.yaml"
    configuration.write_text("mpm:\n  max_frames: 5\n", encoding="utf-8")
    training_ids = training_object_ids or ["source-a", "source-b"]
    training_identity = {
        "schema": DEFORMMASTER_TRAINING_SCHEMA,
        "schema_version": 1,
        "source_repository": DEFORMMASTER_SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "checkpoint_sha256": file_sha256(checkpoint),
        "training_object_ids": training_ids,
    }
    training = {
        **training_identity,
        "manifest_id": content_id(training_identity),
    }
    training_manifest = tmp_path / "training.json"
    write_atomic_json(training, training_manifest, overwrite=False)

    frame_zero = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        dtype=np.float32,
    )
    zero = np.repeat(frame_zero[None], 5, axis=0)
    driven = zero.copy()
    driven[:, 1, 2] = np.linspace(0.0, 0.02, 5, dtype=np.float32)
    driven[:, 2, 2] = np.linspace(0.0, 0.01, 5, dtype=np.float32)
    raw = tmp_path / "rollout.npz"
    write_deterministic_npz(
        raw,
        {
            "driven_surface_positions_m": driven,
            "zero_action_surface_positions_m": zero,
            "action_support": np.array([0.0, 1.0, 0.5], dtype=np.float32),
            "frame_zero_points_m": frame_zero,
        },
    )
    return {
        "checkpoint": checkpoint,
        "configuration": configuration,
        "training_manifest": training_manifest,
        "raw": raw,
    }


def _seal(tmp_path: Path, inputs: dict[str, Path]) -> Path:
    runtime = tmp_path / "runtime.json"
    seal_deformmaster_runtime_manifest(
        raw_rollout_path=inputs["raw"],
        checkpoint_path=inputs["checkpoint"],
        configuration_path=inputs["configuration"],
        training_manifest_path=inputs["training_manifest"],
        output_path=runtime,
        source_revision=SOURCE_REVISION,
        producer_repository="IPS-Stuttgart/BayesianPhysTwin",
        producer_revision=PRODUCER_REVISION,
        case_id="target-a-episode-1",
        target_object_id="target-a",
        prefix_end_frame_exclusive=2,
        time_step_s=1.0 / 30.0,
    )
    return runtime


def _materialize(
    inputs: dict[str, Path], runtime: Path, output: Path
) -> dict[str, object]:
    return materialize_deformmaster_backend(
        raw_rollout_path=inputs["raw"],
        runtime_manifest_path=runtime,
        checkpoint_path=inputs["checkpoint"],
        configuration_path=inputs["configuration"],
        training_manifest_path=inputs["training_manifest"],
        output_dir=output,
    )


def _mutated_runtime(runtime: Path, **updates: object) -> dict[str, object]:
    payload = json.loads(runtime.read_text(encoding="utf-8"))
    payload.update(updates)
    identity = {key: value for key, value in payload.items() if key != "runtime_id"}
    payload["runtime_id"] = content_id(identity)
    return payload


def test_deformmaster_causal_bundle_roundtrip(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    runtime = _seal(tmp_path, inputs)
    output = tmp_path / "bundle"
    artifact = _materialize(inputs, runtime, output)
    assert artifact["backend_kind"] == "deformmaster-mpm-neural-v1"
    assert artifact["information_boundary"]["future_object_tracks_read"] is False
    physical = load_physical_rollout_archive(output / PHYSICAL_ARCHIVE_FILENAME)
    _, raw = load_deformmaster_surface_rollout(inputs["raw"])
    assert np.array_equal(physical["prediction_m"], raw["driven_surface_positions_m"])
    assert validate_deformmaster_backend(output) == artifact


def test_deformmaster_cli_roundtrip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = _inputs(tmp_path)
    runtime = tmp_path / "runtime.json"
    assert (
        deformmaster_cli_main(
            [
                "seal-runtime",
                str(inputs["raw"]),
                str(inputs["checkpoint"]),
                str(inputs["configuration"]),
                str(inputs["training_manifest"]),
                str(runtime),
                "--source-revision",
                SOURCE_REVISION,
                "--producer-repository",
                "IPS-Stuttgart/BayesianPhysTwin",
                "--producer-revision",
                PRODUCER_REVISION,
                "--case-id",
                "target-a-episode-1",
                "--target-object-id",
                "target-a",
                "--prefix-end-frame-exclusive",
                "2",
                "--time-step-s",
                str(1.0 / 30.0),
            ]
        )
        == 0
    )
    capsys.readouterr()
    output = tmp_path / "bundle"
    assert (
        deformmaster_cli_main(
            [
                "materialize",
                str(inputs["raw"]),
                str(runtime),
                str(inputs["checkpoint"]),
                str(inputs["configuration"]),
                str(inputs["training_manifest"]),
                str(output),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert deformmaster_cli_main(["validate", str(output)]) == 0
    assert "deformmaster-mpm-neural-v1" in capsys.readouterr().out


def test_deformmaster_fixed_input_materialization_is_byte_identical(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    runtime = _seal(tmp_path, inputs)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_artifact = _materialize(inputs, runtime, first)
    second_artifact = _materialize(inputs, runtime, second)
    assert first_artifact == second_artifact
    for relative in (
        "deformmaster-backend.json",
        "physical-prediction.npz",
        "SHA256SUMS",
        "provenance/deformmaster-surface-rollout.npz",
        "provenance/deformmaster-runtime.json",
        "provenance/deformmaster-config.yaml",
        "provenance/deformmaster-training-data.json",
    ):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_deformmaster_runtime_rejects_all_frame_router_input(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    runtime = _seal(tmp_path, inputs)
    mutated = _mutated_runtime(runtime, router_input_frame_range_half_open=[0, 5])
    with pytest.raises(ValueError, match="router_input.*crosses"):
        validate_deformmaster_runtime_manifest(mutated)


def test_deformmaster_runtime_validates_without_external_files(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    runtime = _seal(tmp_path, inputs)
    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert validate_deformmaster_runtime_manifest(payload) == payload


def test_deformmaster_runtime_rejects_all_frame_offset_input(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    runtime = _seal(tmp_path, inputs)
    mutated = _mutated_runtime(runtime, offset_input_frame_range_half_open=[0, 5])
    with pytest.raises(ValueError, match="offset_input.*crosses"):
        validate_deformmaster_runtime_manifest(mutated)


def test_deformmaster_runtime_rejects_future_outcome_read(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    runtime = _seal(tmp_path, inputs)
    payload = json.loads(runtime.read_text(encoding="utf-8"))
    boundary = dict(payload["information_boundary"])
    boundary["future_outcomes_read"] = True
    mutated = _mutated_runtime(runtime, information_boundary=boundary)
    with pytest.raises(ValueError, match="information boundary"):
        validate_deformmaster_runtime_manifest(mutated)


def test_deformmaster_runtime_rejects_target_in_checkpoint_training_data(
    tmp_path: Path,
) -> None:
    inputs = _inputs(
        tmp_path,
        training_object_ids=["source-a", "target-a"],
    )
    with pytest.raises(ValueError, match="target object occurs"):
        _seal(tmp_path, inputs)


def test_deformmaster_runtime_rejects_changed_checkpoint(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    runtime = _seal(tmp_path, inputs)
    inputs["checkpoint"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint byte count changed"):
        _materialize(inputs, runtime, tmp_path / "bundle")


def test_deformmaster_raw_rollout_rejects_changed_frame_zero(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    with np.load(inputs["raw"], allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    arrays["driven_surface_positions_m"][0, 0, 0] = 1.0
    changed = tmp_path / "changed.npz"
    write_deterministic_npz(changed, arrays)
    with pytest.raises(ValueError, match="frame-zero material identity"):
        load_deformmaster_surface_rollout(changed)


def test_deformmaster_raw_rollout_rejects_unreadable_archive(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.npz"
    invalid.write_bytes(b"not-an-npz")
    with pytest.raises(ValueError, match="cannot load raw"):
        load_deformmaster_surface_rollout(invalid)


def test_deformmaster_scalar_and_range_validators_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        backend_module._mapping([], name="value")
    with pytest.raises(ValueError, match="string keys"):
        backend_module._mapping({1: "value"}, name="value")
    with pytest.raises(ValueError, match="positive integer"):
        backend_module._positive_integer(True, name="value")
    with pytest.raises(ValueError, match="positive integer"):
        backend_module._positive_integer(0, name="value")
    with pytest.raises(ValueError, match="positive number"):
        backend_module._finite_positive("1", name="value")
    with pytest.raises(ValueError, match="positive number"):
        backend_module._finite_positive(float("nan"), name="value")
    with pytest.raises(ValueError, match="two integer"):
        backend_module._half_open_range([0], name="value")
    with pytest.raises(ValueError, match="nonempty"):
        backend_module._half_open_range([2, 2], name="value")


def test_deformmaster_seal_rejects_empty_future(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    with pytest.raises(ValueError, match="leave a nonempty future"):
        seal_deformmaster_runtime_manifest(
            raw_rollout_path=inputs["raw"],
            checkpoint_path=inputs["checkpoint"],
            configuration_path=inputs["configuration"],
            training_manifest_path=inputs["training_manifest"],
            output_path=tmp_path / "runtime.json",
            source_revision=SOURCE_REVISION,
            producer_repository="IPS-Stuttgart/BayesianPhysTwin",
            producer_revision=PRODUCER_REVISION,
            case_id="target-a-episode-1",
            target_object_id="target-a",
            prefix_end_frame_exclusive=5,
            time_step_s=1.0 / 30.0,
        )


def test_deformmaster_bundle_detects_mutated_output(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    runtime = _seal(tmp_path, inputs)
    output = tmp_path / "bundle"
    _materialize(inputs, runtime, output)
    physical = output / PHYSICAL_ARCHIVE_FILENAME
    physical.write_bytes(physical.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="byte count changed"):
        validate_deformmaster_backend(output)
