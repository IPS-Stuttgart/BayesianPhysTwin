from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import io
import json
import shutil
import socket
import urllib.error
import urllib.request
import zipfile
from datetime import timedelta
from email.message import Message
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from test_poseit_checkpoint_acquisition import NOW, Source, _run, _spec
from test_poseit_checkpoint_acquisition import engine as engine
from test_poseit_remote_zip import _archive, _zip64_archive

from bayesian_phystwin._portable_contracts import write_atomic_json
from bayesian_phystwin_experiments import poseit_checkpoint_acquisition as acquisition
from bayesian_phystwin_experiments import poseit_checkpoint_structure as structure
from bayesian_phystwin_experiments.poseit_hash_checkpoint import RHashCheckpointEngine


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare(
    tmp_path: Path, engine: RHashCheckpointEngine, data: bytes | None = None
) -> tuple[acquisition.AcquisitionSpec, Source, dict[str, Any]]:
    payload = _archive() if data is None else data
    spec = _spec(tmp_path, engine, size=len(payload))
    source = Source(spec)
    source.data = payload
    receipt = _run(spec, engine, source)
    source.calls.clear()
    source.responses.clear()
    return spec, source, receipt


def _authorize(
    spec: acquisition.AcquisitionSpec, receipt: dict[str, Any], **changes: Any
) -> tuple[Path, str]:
    fields = {
        "scope": "central-directory-only",
        "spec_id": spec.spec_id,
        "amendment_sha256": spec.amendment_sha256,
        "acquisition_receipt_file_sha256": _digest(
            spec.root / "acquisition-receipt.json"
        ),
        "acquisition_receipt_record_id": receipt["record_id"],
        "archive_sha256": receipt["archive_sha256"],
        "implementation_revision": "1" * 40,
        "implementation_files": {
            path: _digest(structure.ROOT / path)
            for path in structure.IMPLEMENTATION_FILES
        },
        "hostname": socket.gethostname(),
        "output_root": str(spec.root.parent / "checkpoint-structure-v1"),
        "attempt_ledger_path": str(
            spec.root.parent / "checkpoint-structure-attempt-v1.json"
        ),
        "not_before_utc": NOW.isoformat(),
        "attempt_limit": 1,
        "central_directory_chunk_size_bytes": 97,
        "maximum_central_directory_size_bytes": 4096,
        "boundaries": dict(structure.BOUNDARIES),
        **changes,
    }
    authorization = acquisition._seal("checkpoint-structure-authorization", **fields)
    path = spec.root.parent / f"structure-auth-{authorization['record_id']}.json"
    write_atomic_json(authorization, path, overwrite=False)
    return path, _digest(path)


def _inventory(
    spec: acquisition.AcquisitionSpec,
    engine: RHashCheckpointEngine,
    source: Source,
    auth: tuple[Path, str],
    **changes: Any,
) -> dict[str, Any]:
    return structure.run_checkpointed_structure(
        spec,
        engine,
        auth[0],
        expected_authorization_sha256=auth[1],
        opener=source,
        clock=lambda: NOW,
        **changes,
    )


@pytest.mark.parametrize("zip64", [False, True])
def test_complete_native_chain_to_structure_without_opening_member_payloads(
    tmp_path: Path,
    engine: RHashCheckpointEngine,
    monkeypatch: pytest.MonkeyPatch,
    zip64: bool,
) -> None:
    spec, source, receipt = _prepare(
        tmp_path, engine, _zip64_archive() if zip64 else _archive()
    )
    auth = _authorize(spec, receipt)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("structure execution must not open or decompress member payloads")

    monkeypatch.setattr(zipfile.ZipFile, "open", forbidden)
    result = _inventory(spec, engine, source, auth)
    root = spec.root.parent / "checkpoint-structure-v1"
    private = json.loads((root / "private-member-manifest.json").read_text())
    terminal = json.loads((root / "terminal.json").read_text())
    assert result["archive_sha256"] == hashlib.sha256(source.data).hexdigest()
    assert result["acquisition_receipt_record_id"] == receipt["record_id"]
    assert result["private_member_manifest_file_sha256"] == _digest(
        root / "private-member-manifest.json"
    )
    assert result["structure"]["regular_member_count"] == 2
    assert result["source_and_confirmation_authorized"] is False
    assert all(value is False for value in result["boundaries"].values())
    assert terminal["status"] == "complete"
    assert terminal["result_record_id"] == result["record_id"]
    assert [entry["name"] for entry in private["members"]] == [
        "root/",
        "root/first.txt",
        "root/second.bin",
    ]
    assert "root/first.txt" not in json.dumps(result)
    assert all(
        start % 64 == 0 and end == min(start + 63, len(source.data) - 1)
        for start, end in source.calls
    )
    assert len(source.calls) == len(set(source.calls))
    audit = result["range_audit"]
    assert audit["network_attempt_count"] == len(source.calls)
    assert audit["parser_adapter_responses_are_http_observations"] is False
    assert audit["opaque_adjacent_bytes_may_be_buffered"] is True
    assert audit["raw_bytes_persisted"] is False
    assert all(event["body_identity_accepted"] for event in audit["network_attempts"])
    for event in audit["parser_ranges"]:
        assert (
            event["slice_sha256"]
            == hashlib.sha256(
                source.data[event["start"] : event["end"] + 1]
            ).hexdigest()
        )
    assert {p.name for p in root.iterdir()} == {
        "private-member-manifest.json",
        "structure-lock.json",
        "terminal.json",
    }
    source.calls.clear()
    assert (
        structure.verify_checkpointed_structure(
            spec,
            engine,
            auth[0],
            expected_authorization_sha256=auth[1],
            expected_result_sha256=_digest(root / "structure-lock.json"),
        )
        == result
    )
    assert source.calls == []


@pytest.mark.parametrize("remove_output", [False, True])
def test_separate_ledger_prevents_duplicate_even_if_output_removed(
    tmp_path: Path, engine: RHashCheckpointEngine, remove_output: bool
) -> None:
    spec, source, receipt = _prepare(tmp_path, engine)
    auth = _authorize(spec, receipt)
    _inventory(spec, engine, source, auth)
    if remove_output:
        shutil.rmtree(spec.root.parent / "checkpoint-structure-v1")
    source.calls.clear()
    with pytest.raises(ValueError, match="attempt consumed"):
        _inventory(spec, engine, source, auth)
    assert source.calls == []


@pytest.mark.parametrize(
    "changes",
    [
        {"scope": "member-payloads"},
        {"spec_id": "f" * 64},
        {"amendment_sha256": "f" * 64},
        {"archive_sha256": "f" * 64},
        {"acquisition_receipt_file_sha256": "f" * 64},
        {"acquisition_receipt_record_id": "f" * 64},
        {"implementation_revision": "main"},
        {"implementation_files": {}},
        {"hostname": "not-the-registered-host"},
        {"output_root": "/tmp/alternate-structure-root"},
        {"attempt_ledger_path": "/tmp/alternate-structure-ledger"},
        {"attempt_limit": True},
        {"attempt_limit": 2},
        {"not_before_utc": (NOW - timedelta(days=1)).isoformat()},
        {"not_before_utc": (NOW + timedelta(days=1)).isoformat()},
        {"central_directory_chunk_size_bytes": True},
        {"maximum_central_directory_size_bytes": 0},
        {"maximum_central_directory_size_bytes": 1},
        {"boundaries": {**structure.BOUNDARIES, "scientific_result": 0}},
        {"boundaries": {**structure.BOUNDARIES, "confirmation_opened": True}},
    ],
)
def test_invalid_authorization_never_consumes_attempt_or_contacts_provider(
    tmp_path: Path, engine: RHashCheckpointEngine, changes: dict[str, Any]
) -> None:
    spec, source, receipt = _prepare(tmp_path, engine)
    auth = _authorize(spec, receipt, **changes)
    with pytest.raises(ValueError):
        _inventory(spec, engine, source, auth)
    assert source.calls == []
    assert not (spec.root.parent / "checkpoint-structure-attempt-v1.json").exists()


@pytest.mark.parametrize(
    "target",
    [
        "acquisition-receipt.json",
        "chunks/000000.json",
        "attempts/000000/terminal.json",
    ],
)
def test_missing_or_modified_completion_custody_never_opens_structure(
    tmp_path: Path, engine: RHashCheckpointEngine, target: str
) -> None:
    spec, source, receipt = _prepare(tmp_path, engine)
    auth = _authorize(spec, receipt)
    (spec.root / target).unlink()
    with pytest.raises(ValueError):
        _inventory(spec, engine, source, auth)
    assert source.calls == []
    assert not (spec.root.parent / "checkpoint-structure-attempt-v1.json").exists()


def test_invalid_external_authorization_digest_and_code_binding_fail_closed(
    tmp_path: Path, engine: RHashCheckpointEngine
) -> None:
    spec, source, receipt = _prepare(tmp_path, engine)
    auth = _authorize(spec, receipt)
    with pytest.raises(ValueError, match="authorization changed"):
        _inventory(spec, engine, source, (auth[0], "0" * 64))
    fields = json.loads(auth[0].read_text())["implementation_files"]
    fields["src/bayesian_phystwin_experiments/poseit_checkpoint_structure.py"] = (
        "0" * 64
    )
    auth = _authorize(spec, receipt, implementation_files=fields)
    with pytest.raises(ValueError, match="structure code changed"):
        _inventory(spec, engine, source, auth)
    assert source.calls == []


def test_structure_shares_acquisition_lock_before_attempt_consumption(
    tmp_path: Path, engine: RHashCheckpointEngine
) -> None:
    spec, source, receipt = _prepare(tmp_path, engine)
    auth = _authorize(spec, receipt)
    with spec.lock_path.open("r+") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            _inventory(spec, engine, source, auth)
    assert source.calls == []
    assert not (spec.root.parent / "checkpoint-structure-attempt-v1.json").exists()


@pytest.mark.parametrize("problem", ["changed-bytes", "identity", "socket"])
def test_refetch_failure_preserved_without_parser_or_nested_retry(
    tmp_path: Path,
    engine: RHashCheckpointEngine,
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
) -> None:
    spec, source, receipt = _prepare(tmp_path, engine)
    auth = _authorize(spec, receipt)
    if problem == "changed-bytes":
        source.data = bytes(value ^ 0x80 for value in source.data)
    elif problem == "identity":
        source.header_changes = {"Last-Modified": "changed identity"}
    else:
        source.socket_failures = 99

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("unverified bytes reached the central-directory parser")

    monkeypatch.setattr(structure, "parse_remote_central_directory", forbidden)
    with pytest.raises(ValueError, match="no retry"):
        _inventory(spec, engine, source, auth)
    assert len(source.calls) == (3 if problem == "socket" else 1)
    if problem == "identity":
        assert all(
            response.read_calls == 0 and response.closed
            for response in source.responses
        )
    root = spec.root.parent / "checkpoint-structure-v1"
    terminal = json.loads((root / "terminal.json").read_text())
    assert terminal["status"] == "failed-preserved"
    assert terminal["archive_member_names_opened"] is False
    assert terminal["range_audit"]["network_attempt_count"] == len(source.calls)
    assert not (root / "structure-lock.json").exists()
    source.calls.clear()
    with pytest.raises(ValueError, match="attempt consumed"):
        _inventory(spec, engine, source, auth)
    assert source.calls == []


def test_http_error_rejected_once_without_reading_error_body(
    tmp_path: Path, engine: RHashCheckpointEngine
) -> None:
    spec, source, receipt = _prepare(tmp_path, engine)
    auth = _authorize(spec, receipt)
    calls: list[int] = []
    body = io.BytesIO(b"synthetic provider quota error")

    def reject(request: urllib.request.Request, timeout: float):
        calls.append(1)
        raise urllib.error.HTTPError(request.full_url, 429, "quota", Message(), body)

    with pytest.raises(ValueError, match="no retry"):
        structure.run_checkpointed_structure(
            spec,
            engine,
            auth[0],
            expected_authorization_sha256=auth[1],
            opener=reject,
            clock=lambda: NOW,
        )
    assert len(calls) == 1
    assert body.closed
    terminal = json.loads(
        (spec.root.parent / "checkpoint-structure-v1/terminal.json").read_text()
    )
    assert terminal["range_audit"]["network_attempts"] == [
        {
            "range": f"bytes={(len(source.data) - 22) // 64 * 64}-{len(source.data) - 1}",
            "http_status": 429,
            "body_identity_accepted": False,
        }
    ]


def test_bounded_socket_retry_success_counts_physical_requests_not_parser_calls(
    tmp_path: Path, engine: RHashCheckpointEngine
) -> None:
    spec, source, receipt = _prepare(tmp_path, engine)
    source.socket_failures = 2
    result = _inventory(spec, engine, source, _authorize(spec, receipt))
    audit = result["range_audit"]
    assert audit["network_attempt_count"] == len(set(source.calls)) + 2
    assert [row["http_status"] for row in audit["network_attempts"][:3]] == [
        None,
        None,
        206,
    ]


def test_unsafe_archive_preserves_terminal_without_publishing_names_in_failure(
    tmp_path: Path, engine: RHashCheckpointEngine
) -> None:
    spec, source, receipt = _prepare(
        tmp_path, engine, _archive(unsafe_name="../unsafe-synthetic.txt")
    )
    with pytest.raises(ValueError):
        _inventory(spec, engine, source, _authorize(spec, receipt))
    root = spec.root.parent / "checkpoint-structure-v1"
    text = (root / "terminal.json").read_text()
    assert "unsafe-synthetic" not in text
    assert json.loads(text)["archive_member_names_opened"] is True
    assert not (root / "structure-lock.json").exists()


@pytest.mark.parametrize("target", ["terminal.json", "private-member-manifest.json"])
def test_offline_verification_rejects_partial_publication(
    tmp_path: Path, engine: RHashCheckpointEngine, target: str
) -> None:
    spec, source, receipt = _prepare(tmp_path, engine)
    auth = _authorize(spec, receipt)
    _inventory(spec, engine, source, auth)
    root = spec.root.parent / "checkpoint-structure-v1"
    digest = _digest(root / "structure-lock.json")
    (root / target).unlink()
    source.calls.clear()
    with pytest.raises(ValueError, match="publication is incomplete"):
        structure.verify_checkpointed_structure(
            spec,
            engine,
            auth[0],
            expected_authorization_sha256=auth[1],
            expected_result_sha256=digest,
        )
    assert source.calls == []


def test_wrong_public_result_digest_does_not_authorize_downstream_consumption(
    tmp_path: Path, engine: RHashCheckpointEngine
) -> None:
    spec, source, receipt = _prepare(tmp_path, engine)
    auth = _authorize(spec, receipt)
    _inventory(spec, engine, source, auth)
    source.calls.clear()
    with pytest.raises(ValueError, match="published structure result changed"):
        structure.verify_checkpointed_structure(
            spec,
            engine,
            auth[0],
            expected_authorization_sha256=auth[1],
            expected_result_sha256="0" * 64,
        )
    assert source.calls == []


def test_publication_failure_keeps_attempt_consumed_and_failure_terminal(
    tmp_path: Path, engine: RHashCheckpointEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, source, receipt = _prepare(tmp_path, engine)
    auth = _authorize(spec, receipt)
    original = acquisition._write

    def fail_private(path: Path, value: dict[str, Any]) -> None:
        if path.name == "private-member-manifest.json":
            raise OSError("synthetic disk full")
        original(path, value)

    monkeypatch.setattr(acquisition, "_write", fail_private)
    with pytest.raises(OSError, match="synthetic disk full"):
        _inventory(spec, engine, source, auth)
    root = spec.root.parent / "checkpoint-structure-v1"
    terminal = json.loads((root / "terminal.json").read_text())
    assert terminal["status"] == "failed-preserved"
    assert terminal["archive_member_names_opened"] is True
    assert not (root / "structure-lock.json").exists()
    source.calls.clear()
    with pytest.raises(ValueError, match="attempt consumed"):
        _inventory(spec, engine, source, auth)
    assert source.calls == []


def test_linked_structure_output_fails_before_provider_or_ledger(
    tmp_path: Path, engine: RHashCheckpointEngine
) -> None:
    spec, source, receipt = _prepare(tmp_path, engine)
    auth = _authorize(spec, receipt)
    other = tmp_path / "other"
    other.mkdir()
    (tmp_path / "checkpoint-structure-v1").symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="path is linked"):
        _inventory(spec, engine, source, auth)
    assert source.calls == []
    assert not (tmp_path / "checkpoint-structure-attempt-v1.json").exists()


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = structure.ROOT / "scripts/science/build_poseit_checkpoint_structure_v1.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("checkpoint_structure_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["run", "verify"])
def test_cli_validates_frozen_acquisition_context_and_dispatches_exact_mode(
    tmp_path: Path,
    cli: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
) -> None:
    spec, engine = object(), object()
    calls: list[str] = []
    amendment = tmp_path / "amendment.json"
    authorization = tmp_path / "authorization.json"

    def load(path: Path, **kwargs: Any):
        assert path == amendment
        assert kwargs == {"expected_amendment_sha256": "a" * 64}
        calls.append("context")
        return spec, engine

    def dispatch(actual_spec: object, actual_engine: object, path: Path, **kwargs: Any):
        assert (actual_spec, actual_engine, path) == (spec, engine, authorization)
        assert kwargs["expected_authorization_sha256"] == "b" * 64
        if mode == "run":
            assert kwargs["opener"] is acquisition._default_open
            assert "expected_result_sha256" not in kwargs
        else:
            assert kwargs["expected_result_sha256"] == "c" * 64
            assert "opener" not in kwargs
        calls.append(mode)
        return {"synthetic_only": True}

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("the other mode must not execute")

    monkeypatch.setattr(cli.transport, "load_context", load)
    monkeypatch.setattr(
        cli.structure,
        "run_checkpointed_structure",
        dispatch if mode == "run" else forbidden,
    )
    monkeypatch.setattr(
        cli.structure,
        "verify_checkpointed_structure",
        dispatch if mode == "verify" else forbidden,
    )
    argv = [
        mode,
        "--amendment",
        str(amendment),
        "--expected-amendment-sha256",
        "a" * 64,
        "--authorization",
        str(authorization),
        "--expected-authorization-sha256",
        "b" * 64,
    ]
    if mode == "verify":
        argv += ["--expected-result-sha256", "c" * 64]
    assert cli.main(argv) == 0
    assert calls == ["context", mode]
    assert json.loads(capsys.readouterr().out) == {"synthetic_only": True}


@pytest.mark.parametrize(
    "mode,extra",
    [
        ("verify", []),
        ("run", ["--expected-result-sha256", "c" * 64]),
        ("run", ["--output-root", "/tmp/alternate"]),
    ],
)
def test_cli_rejects_incomplete_or_overridden_commands_before_loading_context(
    tmp_path: Path,
    cli: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    extra: list[str],
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid CLI cannot load execution context")

    monkeypatch.setattr(cli.transport, "load_context", forbidden)
    with pytest.raises(SystemExit):
        cli.main(
            [
                mode,
                "--amendment",
                str(tmp_path / "a.json"),
                "--expected-amendment-sha256",
                "a" * 64,
                "--authorization",
                str(tmp_path / "b.json"),
                "--expected-authorization-sha256",
                "b" * 64,
                *extra,
            ]
        )
