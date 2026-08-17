from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.cli.learned_twin_backend import main
from bayesian_phystwin.learned_twin_backend_v1 import (
    ARTIFACT_FILENAME,
    CHECKSUMS_FILENAME,
    LEARNED_TWIN_CLAIM_BOUNDARY,
    LEARNED_TWIN_PROFILES,
    PHYSICAL_ARCHIVE_FILENAME,
    SOURCE_ARCHIVE_FILENAME,
    describe_learned_twin_profiles,
    materialize_learned_twin_backend,
    validate_learned_twin_backend,
)
from bayesian_phystwin.physical_rollout_v1 import (
    load_physical_rollout_archive,
    write_deterministic_npz,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arrays() -> dict[str, np.ndarray]:
    frame_zero = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float64)
    persistence = np.repeat(frame_zero[None], 3, axis=0)
    driven = persistence.copy()
    driven[1:, :, 2] += np.array([0.001, 0.002])[:, None]
    zero = persistence.copy()
    prediction = driven.copy()
    return {
        "prediction_m": prediction,
        "persistence_m": persistence,
        "driven_readout_m": driven,
        "zero_action_readout_m": zero,
        "action_support": np.array([1.0, 0.5], dtype=np.float64),
        "frame_zero_points_m": frame_zero,
    }


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source.npz"
    write_deterministic_npz(source, _arrays())
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen learned twin checkpoint\n")
    return source, checkpoint


def _build(
    tmp_path: Path,
    *,
    output_name: str = "bundle",
    profile_id: str = "neuspring-v1",
    mode: str = "causal-source-v1",
    training_object_ids: tuple[str, ...] = ("source-a", "source-b"),
    target_future_observations_used: bool = False,
) -> tuple[Path, Path, dict[str, object]]:
    source, checkpoint = _inputs(tmp_path)
    output = tmp_path / output_name
    result = materialize_learned_twin_backend(
        source_rollout_path=source,
        model_artifacts={"checkpoints/model.pt": checkpoint},
        output_dir=output,
        profile_id=profile_id,
        mode=mode,  # type: ignore[arg-type]
        producer_repository="example/portable-producer",
        producer_revision="b" * 40,
        producer_source_artifacts={"producer.py": "a" * 64},
        case_id="case-001",
        target_object_id="target-a",
        training_object_ids=training_object_ids,
        evidence_frame_range_half_open=(0, 4),
        rollout_frame_range_half_open=(4, 7),
        target_future_observations_used=target_future_observations_used,
    )
    return output, checkpoint, result


def _rewrite_checksums(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != CHECKSUMS_FILENAME
    )
    (root / CHECKSUMS_FILENAME).write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files
        ),
        encoding="ascii",
    )


def _rewrite_artifact(root: Path, update: dict[str, object]) -> None:
    path = root / ARTIFACT_FILENAME
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact.update(update)
    identity = {key: value for key, value in artifact.items() if key != "artifact_id"}
    canonical = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    artifact["artifact_id"] = hashlib.sha256(canonical).hexdigest()
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    _rewrite_checksums(root)


def test_profiles_report_support_without_inventing_native_releases() -> None:
    registry = describe_learned_twin_profiles()
    profiles = {item["profile_id"]: item for item in registry["profiles"]}
    assert set(profiles) == {
        "matphys-v1",
        "neuspring-v1",
        "physpring-v1",
        "physworld-v1",
        "egophys-v1",
    }
    assert profiles["matphys-v1"]["public_runtime_executable"] is True
    assert profiles["neuspring-v1"]["public_release_status"] == (
        "repository-metadata-only"
    )
    assert profiles["physpring-v1"]["upstream_repository"] is None
    assert profiles["physworld-v1"]["public_runtime_executable"] is False
    assert profiles["egophys-v1"]["upstream_snapshot_revision"] is None
    assert all(item["portable_intake_supported"] is True for item in profiles.values())
    assert all(
        item["evidence_stage"] == "registered-adapter" for item in profiles.values()
    )
    assert all(item["source_value_qualified"] is False for item in profiles.values())
    assert all(
        item["recommended_for_claim_bearing_evaluation"] is False
        for item in profiles.values()
    )
    assert len(registry["registry_id"]) == 64


def test_causal_bundle_round_trip_binds_exact_arrays_and_claim_boundary(
    tmp_path: Path,
) -> None:
    output, _, result = _build(tmp_path)
    validated = validate_learned_twin_backend(output, verify_sources=True)
    assert validated == result
    assert result["information_boundary"] == {
        "causal_forecast_eligible": True,
        "future_outcomes_opened_before_sealing": False,
        "known_future_controller_action_used": True,
        "official_method_reproduction_claimed": False,
        "prediction_hashed_before_future_scoring": True,
        "published_benchmark_parity_claimed": False,
        "target_future_observations_used": False,
        "target_object_excluded_from_training": True,
    }
    assert result["claim_boundary"] == LEARNED_TWIN_CLAIM_BOUNDARY
    source = load_physical_rollout_archive(
        output / "provenance" / SOURCE_ARCHIVE_FILENAME
    )
    physical = load_physical_rollout_archive(output / PHYSICAL_ARCHIVE_FILENAME)
    for name in source:
        assert source[name].dtype == physical[name].dtype
        assert source[name].tobytes() == physical[name].tobytes()


def test_build_is_deterministic_for_same_sources(tmp_path: Path) -> None:
    source, checkpoint = _inputs(tmp_path)

    def build(name: str) -> Path:
        output = tmp_path / name
        materialize_learned_twin_backend(
            source_rollout_path=source,
            model_artifacts={"checkpoints/model.pt": checkpoint},
            output_dir=output,
            profile_id="physworld-v1",
            mode="causal-source-v1",
            producer_repository="example/portable-producer",
            producer_revision="b" * 40,
            producer_source_artifacts={"producer.py": "a" * 64},
            case_id="case-001",
            target_object_id="target-a",
            training_object_ids=("source-a",),
            evidence_frame_range_half_open=(0, 4),
            rollout_frame_range_half_open=(4, 7),
        )
        return output

    first, second = build("first"), build("second")
    for relative in (
        ARTIFACT_FILENAME,
        CHECKSUMS_FILENAME,
        PHYSICAL_ARCHIVE_FILENAME,
        f"provenance/{SOURCE_ARCHIVE_FILENAME}",
    ):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_published_parity_is_explicitly_noncausal(tmp_path: Path) -> None:
    output, _, result = _build(
        tmp_path,
        mode="published-parity-v1",
        training_object_ids=("source-a", "target-a"),
        target_future_observations_used=True,
    )
    assert result["information_boundary"]["causal_forecast_eligible"] is False
    assert (
        result["information_boundary"]["target_object_excluded_from_training"] is False
    )
    assert result["information_boundary"]["target_future_observations_used"] is True
    assert result["information_boundary"]["published_benchmark_parity_claimed"] is False
    validate_learned_twin_backend(output)


@pytest.mark.parametrize(
    ("training", "future", "message"),
    [
        (("target-a",), False, "target-object exclusion"),
        (("source-a",), True, "forbids target future observations"),
    ],
)
def test_causal_mode_rejects_leakage(
    tmp_path: Path,
    training: tuple[str, ...],
    future: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _build(
            tmp_path,
            training_object_ids=training,
            target_future_observations_used=future,
        )


def test_causal_mode_rejects_overlapping_evidence(tmp_path: Path) -> None:
    source, checkpoint = _inputs(tmp_path)
    with pytest.raises(ValueError, match="evidence must end"):
        materialize_learned_twin_backend(
            source_rollout_path=source,
            model_artifacts={"model.pt": checkpoint},
            output_dir=tmp_path / "bundle",
            profile_id="matphys-v1",
            mode="causal-source-v1",
            producer_repository="example/producer",
            producer_revision="b" * 40,
            producer_source_artifacts={"producer.py": "a" * 64},
            case_id="case",
            target_object_id="target",
            training_object_ids=("source",),
            evidence_frame_range_half_open=(0, 5),
            rollout_frame_range_half_open=(4, 7),
        )


def test_model_source_mutation_is_detected_when_requested(tmp_path: Path) -> None:
    output, checkpoint, _ = _build(tmp_path)
    checkpoint.write_bytes(b"mutated learned twin checkpoint\n")
    validate_learned_twin_backend(output)
    with pytest.raises(ValueError, match="(byte count|digest) changed"):
        validate_learned_twin_backend(output, verify_sources=True)


def test_output_mutation_is_rejected(tmp_path: Path) -> None:
    output, _, _ = _build(tmp_path)
    physical = output / PHYSICAL_ARCHIVE_FILENAME
    arrays = load_physical_rollout_archive(physical)
    arrays["prediction_m"][1, 0, 1] += 0.001
    physical.unlink()
    write_deterministic_npz(physical, arrays)
    with pytest.raises(ValueError, match="output digest changed"):
        validate_learned_twin_backend(output)


def test_rehashed_artifact_cannot_claim_official_reproduction(tmp_path: Path) -> None:
    output, _, result = _build(tmp_path)
    boundary = dict(result["information_boundary"])
    boundary["official_method_reproduction_claimed"] = True
    _rewrite_artifact(output, {"information_boundary": boundary})
    with pytest.raises(ValueError, match="information boundary changed"):
        validate_learned_twin_backend(output)


def test_rehashed_profile_cannot_upgrade_unreleased_method(tmp_path: Path) -> None:
    output, _, result = _build(tmp_path)
    profile = dict(result["profile"])
    profile["public_runtime_executable"] = True
    _rewrite_artifact(output, {"profile": profile})
    with pytest.raises(ValueError, match="profile changed"):
        validate_learned_twin_backend(output)


def test_invalid_model_artifact_roster_and_rollout_length_fail(tmp_path: Path) -> None:
    source, _ = _inputs(tmp_path)
    kwargs = dict(
        source_rollout_path=source,
        output_dir=tmp_path / "bundle",
        profile_id="egophys-v1",
        mode="causal-source-v1",
        producer_repository="example/producer",
        producer_revision="b" * 40,
        producer_source_artifacts={"producer.py": "a" * 64},
        case_id="case",
        target_object_id="target",
        training_object_ids=(),
        evidence_frame_range_half_open=(0, 4),
        rollout_frame_range_half_open=(4, 7),
    )
    with pytest.raises(ValueError, match="model artifacts are empty"):
        materialize_learned_twin_backend(model_artifacts={}, **kwargs)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    with pytest.raises(ValueError, match="frame range"):
        materialize_learned_twin_backend(
            model_artifacts={"model.pt": checkpoint},
            **{**kwargs, "rollout_frame_range_half_open": (4, 8)},
        )


def test_cli_profiles_build_and_validate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["profiles"]) == 0
    profiles = json.loads(capsys.readouterr().out)
    assert len(profiles["profiles"]) == len(LEARNED_TWIN_PROFILES)

    source, checkpoint = _inputs(tmp_path)
    output = tmp_path / "cli-bundle"
    assert (
        main(
            [
                "build",
                str(source),
                str(output),
                "--profile",
                "matphys-v1",
                "--mode",
                "causal-source-v1",
                "--model-artifact",
                f"checkpoint.pt={checkpoint}",
                "--source-artifact",
                f"producer.py={'a' * 64}",
                "--producer-repository",
                "example/producer",
                "--producer-revision",
                "b" * 40,
                "--case-id",
                "case",
                "--target-object-id",
                "target",
                "--training-object-id",
                "source",
                "--evidence-start",
                "0",
                "--evidence-stop",
                "4",
                "--rollout-start",
                "4",
                "--rollout-stop",
                "7",
            ]
        )
        == 0
    )
    built = json.loads(capsys.readouterr().out)
    assert built["profile"]["profile_id"] == "matphys-v1"
    assert main(["validate", str(output), "--verify-sources"]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["artifact_id"] == built["artifact_id"]


def test_cli_rejects_duplicate_key_value_inputs(tmp_path: Path) -> None:
    source, checkpoint = _inputs(tmp_path)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "build",
                str(source),
                str(tmp_path / "bundle"),
                "--profile",
                "matphys-v1",
                "--mode",
                "causal-source-v1",
                "--model-artifact",
                f"model.pt={checkpoint}",
                "--model-artifact",
                f"model.pt={checkpoint}",
                "--source-artifact",
                f"producer.py={'a' * 64}",
                "--producer-repository",
                "example/producer",
                "--producer-revision",
                "b" * 40,
                "--case-id",
                "case",
                "--target-object-id",
                "target",
                "--evidence-start",
                "0",
                "--evidence-stop",
                "4",
                "--rollout-start",
                "4",
                "--rollout-stop",
                "7",
            ]
        )
    assert error.value.code == 2
