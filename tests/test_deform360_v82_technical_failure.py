from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from bayesian_phystwin import deform360_held_v82_technical_failure as integrity


ROOT = Path(__file__).resolve().parents[1]
OPERATOR = ROOT / "scripts/held/seal_deform360_v82_technical_failure.py"


def _module():
    spec = importlib.util.spec_from_file_location("v82_failure_sealer", OPERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, payload: bytes, *, mode: int = 0o400) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)
    return path


def _git(code: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(code), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _fixture(
    module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    base = tmp_path / "formal"
    active = base / "held-v82"
    archive = base / "held-v82-attempt-1-technical-failure"
    pointer = base / "held-v82-attempt-1-technical-failure-pointer.json"
    completion = base / "held-v82-attempt-1-technical-failure-completion.json"
    active.mkdir(parents=True)

    monkeypatch.setattr(module, "BASE", base)
    monkeypatch.setattr(module, "ACTIVE", active)
    monkeypatch.setattr(module, "ARCHIVE", archive)
    monkeypatch.setattr(module, "POINTER", pointer)
    monkeypatch.setattr(module, "COMPLETION", completion)
    monkeypatch.setattr(module.socket, "gethostname", lambda: module.EXPECTED_HOST)
    monkeypatch.setattr(module, "_running_formal_processes", lambda: [])

    staging = active / "code-staging"
    staging.mkdir()
    _git(staging, "init", "--quiet")
    _git(staging, "config", "user.name", "Held Test")
    _git(staging, "config", "user.email", "held@example.invalid")
    (staging / "method.txt").write_text("frozen method\n", encoding="utf-8")
    _git(staging, "add", "method.txt")
    _git(staging, "commit", "--quiet", "-m", "fixture")
    head = _git(staging, "rev-parse", "HEAD")
    code = active / f"code-{head}"
    staging.rename(code)
    monkeypatch.setattr(module, "EXPECTED_DEPLOYED_HEAD", head)
    monkeypatch.setattr(module, "EXPECTED_DEPLOYED_CODE_NAME", code.name)

    calibration = active / "calibration"
    cases = calibration / "cases"
    private = calibration / "private-targets"
    query_inputs = calibration / "query-inputs"
    query_outputs = calibration / "query-outputs"
    for case in module.EXPECTED_CASES:
        for relative in (
            "frame-zero/frame_zero_bundle.manifest.json",
            "physical/physical_prior_seal.json",
            "prefix-authorization.json",
            "online/online_prediction_seal.json",
            "frozen-field/preoutcome-frozen-field-manifest.json",
        ):
            _write(cases / case / relative, b"sealed source metadata\n")
        (private / case).mkdir(parents=True)
        (query_inputs / case).mkdir(parents=True)
        (query_outputs / case).mkdir(parents=True)

    failed = private / module.FAILED_CASE
    reconstruction = failed / "fresh-official-reconstruction/staged-aligned"
    reconstruction.mkdir(parents=True)
    _write(reconstruction / "opaque-source.bin", b"source-only staging\n", mode=0o600)
    stdout_payload = b"isolated child stdout\n"
    stderr_payload = b"\n".join(module.ERROR_MARKERS) + b"\n"
    _write(
        failed / "isolated-official-reconstruction.stdout.log",
        stdout_payload,
    )
    _write(
        failed / "isolated-official-reconstruction.stderr.log",
        stderr_payload,
    )
    monkeypatch.setattr(module, "EXPECTED_STDOUT_SIZE", len(stdout_payload))
    monkeypatch.setattr(
        module,
        "EXPECTED_STDOUT_SHA256",
        hashlib.sha256(stdout_payload).hexdigest(),
    )
    monkeypatch.setattr(module, "EXPECTED_STDERR_SIZE", len(stderr_payload))
    monkeypatch.setattr(
        module,
        "EXPECTED_STDERR_SHA256",
        hashlib.sha256(stderr_payload).hexdigest(),
    )
    _write(active / "replacement-source/source.bin", b"source cache\n", mode=0o600)

    lock = integrity.signed(
        {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldOnlineBeliefLock",
            "protocol_id": integrity.PROTOCOL_ID,
            "execution_attempt": integrity.EXECUTION_ATTEMPT,
            "stage": "calibration",
            "held_root": str(active),
            "calibration_case_whitelist": list(module.EXPECTED_CASES),
            "immutable_bindings": {
                "method_head_text_sha256": hashlib.sha256(
                    head.encode("ascii")
                ).hexdigest()
            },
        }
    )
    lock_payload = (
        json.dumps(lock, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _write(active / "calibration-lock.json", lock_payload)
    monkeypatch.setattr(
        module,
        "EXPECTED_LOCK_FILE_SHA256",
        hashlib.sha256(lock_payload).hexdigest(),
    )
    monkeypatch.setattr(
        module,
        "EXPECTED_LOCK_ARTIFACT_SHA256",
        lock["artifact_sha256"],
    )
    return active, archive, pointer, completion


def test_seals_source_side_runtime_failure_and_is_restart_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    active, archive, pointer, completion = _fixture(module, monkeypatch, tmp_path)

    first = module.seal()
    second = module.seal()

    assert not active.exists()
    assert archive.is_dir()
    assert first == second
    assert first["v82_calibration_result"] == integrity.RESULT_STATUS
    assert first["v82_technical_failure_archive_integrity"]["regular_file_count"] > 0
    for path in (
        archive / integrity.REPORT_NAME,
        pointer,
        completion,
    ):
        assert path.stat().st_mode & 0o777 == 0o400


def test_rejects_any_protected_target_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    active, _archive, _pointer, _completion = _fixture(
        module,
        monkeypatch,
        tmp_path,
    )
    _write(
        active
        / "calibration/private-targets"
        / module.FAILED_CASE
        / "official-target.npz",
        b"must not exist\n",
    )

    with pytest.raises(RuntimeError, match="protected v8.2 outcome artifact"):
        module.seal()


def test_rejects_changed_child_failure_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    active, _archive, _pointer, _completion = _fixture(
        module,
        monkeypatch,
        tmp_path,
    )
    stderr = (
        active
        / "calibration/private-targets"
        / module.FAILED_CASE
        / "isolated-official-reconstruction.stderr.log"
    )
    stderr.chmod(0o600)
    stderr.write_bytes(b"different failure\n")
    stderr.chmod(0o400)

    with pytest.raises(RuntimeError, match="failure evidence changed"):
        module.seal()


def test_validator_detects_archive_content_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    _active, archive, pointer, completion = _fixture(module, monkeypatch, tmp_path)
    module.seal()
    payload = archive / "replacement-source/source.bin"
    payload.chmod(0o600)
    payload.write_bytes(b"mutated\n")
    payload.chmod(0o400)

    with pytest.raises(ValueError, match="archive content inventory changed"):
        integrity.validate_v82_technical_failure_lineage(
            archive_path=archive,
            report_path=archive / integrity.REPORT_NAME,
            pointer_path=pointer,
            completion_path=completion,
            verify_content_inventory=True,
        )
