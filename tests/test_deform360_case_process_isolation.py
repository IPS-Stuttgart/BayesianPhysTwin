from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin import deform360_case_process_isolation as isolation
from bayesian_phystwin import deform360_held_v8_outcome_driver as driver

CASE_NAME = "083-blanket-cloth-ep0000"
BARRIER = "a" * 64
SMOKE_SHA256 = "b" * 64


def _write(path: Path, payload: bytes = b"fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _reconstruction(point_count: int = 5) -> dict[str, object]:
    points = np.arange(
        76 * point_count * 3, dtype=np.float32
    ).reshape(76, point_count, 3)
    return {
        "object_points": points,
        "object_visibilities": np.ones((76, point_count), dtype=bool),
        "object_motions_valid": np.ones((76, point_count), dtype=bool),
        "provenance": {
            "adapter_id": "fixture",
            "fresh_v8_reconstruction": True,
            "isolated_viser_process_churn_guard": _viser_guard(),
        },
    }


def _runtime_smoke() -> dict[str, object]:
    return {
        "artifact_sha256": SMOKE_SHA256,
        "extension_loaded_and_retained": True,
        "target_or_outcome_path_accessed": False,
    }


def _viser_guard(source: Path | None = None) -> dict[str, object]:
    source_record = (
        isolation._bound_file(source)
        if source is not None
        else {
            "path": "/code/viser_guard.py",
            "sha256": "d" * 64,
            "size_bytes": 1,
        }
    )
    value: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Deform360HeldViserProcessChurnGuardV1",
        "guard_source": source_record,
        "guard_installed_before_original_trainer_import": True,
        "target_or_outcome_path_accessed": False,
    }
    value["artifact_sha256"] = isolation._artifact_sha256(value)
    return value


def test_isolated_result_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    lock = _write(tmp_path / "lock.json")
    worker = _write(tmp_path / "run-worker.py")
    archive = tmp_path / "private" / "result.npz"
    manifest = tmp_path / "private" / "result.json"
    archive.parent.mkdir()

    written = isolation.write_isolated_reconstruction_result(
        archive,
        manifest,
        case_name=CASE_NAME,
        role="calibration",
        lock_path=lock,
        cohort_barrier_sha256=BARRIER,
        reconstruction=_reconstruction(),
        worker_source_path=worker,
    )

    assert written["case_name"] == CASE_NAME
    assert written["resource_lifecycle_policy"][
        "pinned_default_trainer"
    ] is True
    assert stat.S_IMODE(os.lstat(archive).st_mode) == 0o400
    assert stat.S_IMODE(os.lstat(manifest).st_mode) == 0o400
    loaded = isolation.load_isolated_reconstruction(
        manifest,
        expected_case_name=CASE_NAME,
        expected_role="calibration",
        expected_lock_path=lock,
        expected_cohort_barrier_sha256=BARRIER,
        expected_worker_source_path=worker,
    )
    assert np.array_equal(loaded["object_points"], _reconstruction()["object_points"])
    assert loaded["provenance"]["adapter_id"] == "fixture"

    os.chmod(archive, 0o600)
    with archive.open("ab") as stream:
        stream.write(b"tamper")
    os.chmod(archive, 0o400)
    with pytest.raises(ValueError, match="archive binding changed"):
        isolation.validate_isolated_reconstruction_result(manifest)


def test_isolated_result_is_write_once_and_rejects_wrong_dtypes(
    tmp_path: Path,
) -> None:
    lock = _write(tmp_path / "lock.json")
    worker = _write(tmp_path / "worker.py")
    parent = tmp_path / "private"
    parent.mkdir()
    archive = parent / "result.npz"
    manifest = parent / "result.json"
    arguments = {
        "case_name": CASE_NAME,
        "role": "calibration",
        "lock_path": lock,
        "cohort_barrier_sha256": BARRIER,
        "reconstruction": _reconstruction(),
        "worker_source_path": worker,
    }
    isolation.write_isolated_reconstruction_result(
        archive, manifest, **arguments
    )
    with pytest.raises(ValueError, match="already exists"):
        isolation.write_isolated_reconstruction_result(
            archive, manifest, **arguments
        )

    malformed = _reconstruction()
    malformed["object_points"] = np.asarray(
        malformed["object_points"], dtype=np.float64
    )
    with pytest.raises(ValueError, match="finite float32"):
        isolation.write_isolated_reconstruction_result(
            parent / "wrong.npz",
            parent / "wrong.json",
            **{**arguments, "reconstruction": malformed},
        )
    assert not (parent / "wrong.npz").exists()
    assert not (parent / "wrong.json").exists()


def test_isolated_result_requires_the_locked_child_runtime_smoke(
    tmp_path: Path,
) -> None:
    lock = _write(tmp_path / "lock.json")
    worker = _write(tmp_path / "worker.py")
    parent = tmp_path / "private"
    parent.mkdir()
    reconstruction = _reconstruction()
    reconstruction["provenance"] = {
        **dict(reconstruction["provenance"]),  # type: ignore[arg-type]
        "isolated_gsplat_runtime_smoke": _runtime_smoke(),
    }
    manifest = parent / "result.json"
    isolation.write_isolated_reconstruction_result(
        parent / "result.npz",
        manifest,
        case_name=CASE_NAME,
        role="calibration",
        lock_path=lock,
        cohort_barrier_sha256=BARRIER,
        reconstruction=reconstruction,
        worker_source_path=worker,
    )

    isolation.validate_isolated_reconstruction_result(
        manifest,
        expected_gsplat_runtime_smoke_artifact_sha256=SMOKE_SHA256,
    )
    with pytest.raises(ValueError, match="differs from the locked runtime"):
        isolation.validate_isolated_reconstruction_result(
            manifest,
            expected_gsplat_runtime_smoke_artifact_sha256="c" * 64,
        )


def test_isolated_worker_argv_has_no_query_score_or_gate_path(
    tmp_path: Path,
) -> None:
    code = tmp_path / ("code-" + "b" * 40)
    worker = _write(
        code / "scripts" / "held" / "run_deform360_isolated_reconstruction.py"
    )
    private = tmp_path / "calibration" / "private-targets" / CASE_NAME
    private.mkdir(parents=True)
    kwargs = {
        "python_executable": "/runtime/python",
        "deployed_code": code,
        "lock_path": tmp_path / "lock.json",
        "role": "calibration",
        "case_name": CASE_NAME,
        "online_prediction_seal_path": tmp_path / "online.json",
        "aligned_episode_dir": tmp_path / "aligned",
        "reconstruction_output_dir": private / "reconstruction",
        "result_archive_path": private / "isolated-result.npz",
        "result_manifest_path": private / "isolated-result.json",
        "cohort_barrier_sha256": BARRIER,
        "deform360_repo": tmp_path / "deform360",
        "sam2_repository": tmp_path / "sam2",
        "sam2_checkpoint": tmp_path / "sam2.pt",
        "cotracker_repo": tmp_path / "cotracker",
        "cotracker_checkpoint": tmp_path / "cotracker.pth",
        "device": "cuda:0",
        "ffmpeg": "ffmpeg",
        "pycache_prefix": driver.PYCACHE_PREFIX,
        "path_environment": driver.PINNED_PATH,
    }
    argv, environment, safe_cwd = (
        isolation.build_isolated_reconstruction_subprocess(**kwargs)
    )

    assert Path(argv[5]) == worker
    flags = {value for value in argv if value.startswith("--")}
    assert flags == {
        "--lock",
        "--role",
        "--case-name",
        "--online-prediction-seal",
        "--aligned-episode",
        "--reconstruction-output-dir",
        "--result-archive",
        "--result-manifest",
        "--cohort-barrier-sha256",
        "--deform360-repo",
        "--sam2-repository",
        "--sam2-checkpoint",
        "--cotracker-repo",
        "--cotracker-checkpoint",
        "--device",
        "--ffmpeg",
    }
    serialized = json.dumps(
        {"argv": argv, "environment": environment, "cwd": safe_cwd}
    )
    for forbidden in (
        "official-frame-zero-query",
        "queried-prediction",
        "future-score",
        "gate-decision",
    ):
        assert forbidden not in serialized
    assert environment == driver._normalized_environment()
    assert safe_cwd == str(private)


def test_runner_seals_logs_and_validates_child_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = tmp_path / ("code-" + "c" * 40)
    worker = _write(
        code / "scripts" / "held" / "run_deform360_isolated_reconstruction.py"
    )
    guard_source = _write(
        code
        / "src"
        / "bayesian_phystwin"
        / "deform360_held_v83_viser_guard.py"
    )
    lock = _write(tmp_path / "lock.json")
    private = tmp_path / "private"
    private.mkdir()
    archive = private / "result.npz"
    manifest = private / "result.json"
    build_kwargs = {
        "python_executable": "/runtime/python",
        "deployed_code": code,
        "lock_path": lock,
        "role": "calibration",
        "case_name": CASE_NAME,
        "online_prediction_seal_path": tmp_path / "online.json",
        "aligned_episode_dir": tmp_path / "aligned",
        "reconstruction_output_dir": private / "reconstruction",
        "result_archive_path": archive,
        "result_manifest_path": manifest,
        "cohort_barrier_sha256": BARRIER,
        "deform360_repo": tmp_path / "deform360",
        "sam2_repository": tmp_path / "sam2",
        "sam2_checkpoint": tmp_path / "sam2.pt",
        "cotracker_repo": tmp_path / "cotracker",
        "cotracker_checkpoint": tmp_path / "cotracker.pth",
        "device": "cuda:0",
        "ffmpeg": "ffmpeg",
        "pycache_prefix": driver.PYCACHE_PREFIX,
        "path_environment": driver.PINNED_PATH,
        "expected_gsplat_runtime_smoke_artifact_sha256": SMOKE_SHA256,
    }

    def fake_run(*_args: object, **kwargs: object) -> SimpleNamespace:
        kwargs["stdout"].write(b"worker output\n")  # type: ignore[union-attr]
        kwargs["stderr"].write(b"worker diagnostics\n")  # type: ignore[union-attr]
        isolation.write_isolated_reconstruction_result(
            archive,
            manifest,
            case_name=CASE_NAME,
            role="calibration",
            lock_path=lock,
            cohort_barrier_sha256=BARRIER,
            reconstruction={
                **_reconstruction(),
                "provenance": {
                    **dict(_reconstruction()["provenance"]),  # type: ignore[arg-type]
                    "isolated_gsplat_runtime_smoke": _runtime_smoke(),
                    "isolated_viser_process_churn_guard": _viser_guard(
                        guard_source
                    ),
                },
            },
            worker_source_path=worker,
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(isolation.subprocess, "run", fake_run)
    stdout = private / "worker.stdout.log"
    stderr = private / "worker.stderr.log"
    result = isolation.run_isolated_reconstruction_subprocess(
        stdout_log_path=stdout,
        stderr_log_path=stderr,
        **build_kwargs,
    )

    assert np.array_equal(result["object_points"], _reconstruction()["object_points"])
    assert stat.S_IMODE(os.lstat(stdout).st_mode) == 0o400
    assert stat.S_IMODE(os.lstat(stderr).st_mode) == 0o400
    assert stdout.read_text(encoding="utf-8") == "worker output\n"
    assert stderr.read_text(encoding="utf-8") == "worker diagnostics\n"
