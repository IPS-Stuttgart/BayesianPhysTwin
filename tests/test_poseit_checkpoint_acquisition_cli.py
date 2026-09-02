from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from bayesian_phystwin._portable_contracts import write_atomic_json
from bayesian_phystwin_experiments import poseit_checkpoint_acquisition as acquisition

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/acquire_poseit_checkpointed_range_hash_v1.py"


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("poseit_checkpoint_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.socket, "gethostname", lambda: "synthetic-host")

    def native(path: Path, *, expected_library_sha256: str) -> SimpleNamespace:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_library_sha256
        return SimpleNamespace(library_sha256=expected_library_sha256)

    monkeypatch.setattr(module, "RHashCheckpointEngine", native)
    return module


def _plan(cli: ModuleType, tmp_path: Path) -> dict[str, Any]:
    library = tmp_path / "synthetic-library"
    library.write_bytes(b"not a real native library; constructor is mocked")
    return {
        "status": "frozen-acquisition-only",
        "implementation_revision": "1" * 40,
        "implementation_files": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in cli.IMPLEMENTATION_FILES
        },
        "parent_files": dict(cli.PARENT_FILES),
        "native_library": {
            "path": str(library),
            "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        },
        "execution": {
            "host_alias": "gpuserver4090",
            "hostname": "synthetic-host",
            "root": str(cli.REMOTE_ROOT / "checkpoint-range-hash-v1"),
            "lock_path": str(cli.REMOTE_ROOT / "range-hash.lock"),
            "first_request_not_before_utc": "2026-09-03T17:08:20.674819+00:00",
            "resume_delay_seconds": 86400,
        },
        "scientific_method_changed": False,
        "prior_partial_hashes_reused": False,
        "legacy_receipt_compatible": False,
        "resume_policy": "new-authorization-after-preserved-terminal-and-cooldown",
        "boundaries": dict(acquisition._BOUNDARIES),
    }


def _write_plan(tmp_path: Path, plan: dict[str, Any]) -> tuple[Path, str]:
    record = acquisition._seal("checkpoint-transport-amendment", **plan)
    path = tmp_path / "synthetic-amendment.json"
    write_atomic_json(record, path, overwrite=False)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_preflight_uses_frozen_transport_without_consuming_attempt_or_contacting_source(
    cli: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, digest = _write_plan(tmp_path, _plan(cli, tmp_path))

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail(
            "preflight must not open a provider, create a ledger, or publish a receipt"
        )

    monkeypatch.setattr(acquisition, "run_checkpointed_attempt", forbidden)
    monkeypatch.setattr(acquisition, "publish_completed_receipt", forbidden)
    monkeypatch.setattr(acquisition, "verify_completed_receipt", forbidden)
    monkeypatch.setattr(acquisition.urllib.request, "urlopen", forbidden)
    assert (
        cli.main(
            [
                "preflight",
                "--amendment",
                str(path),
                "--expected-amendment-sha256",
                digest,
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["archive_size_bytes"] == 905738058282
    assert result["chunk_count"] == 26994
    assert result["execution_authorized"] is False
    assert result["attempt_consumed"] is False and result["provider_contacted"] is False
    spec, _ = cli.load_context(path, expected_amendment_sha256=digest)
    assert spec.expectation.max_workers == 8
    assert spec.expectation.max_attempts_per_range == 3
    assert spec.expectation.chunk_size_bytes == 33554432
    assert spec.expectation.timeout_seconds == 120
    assert spec.parent_sha256 == cli.PARENT_FILES


@pytest.mark.parametrize(
    "case",
    [
        "status",
        "parent",
        "implementation",
        "canonical_helper",
        "missing_code",
        "extra_code",
        "cooldown",
        "resume_delay",
        "host",
        "root",
        "lock",
        "method",
        "partial_hash",
        "legacy",
        "resume_policy",
        "boundary",
        "unknown",
        "revision",
        "library_link",
    ],
)
def test_amendment_drift_is_rejected_before_native_loading(
    cli: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    plan = _plan(cli, tmp_path)
    if case == "status":
        plan["status"] = "draft"
    elif case == "parent":
        plan["parent_files"][f"{cli.PROTOCOL_PREFIX}.json"] = "f" * 64
    elif case == "implementation":
        plan["implementation_files"][next(iter(cli.IMPLEMENTATION_FILES))] = "f" * 64
    elif case == "canonical_helper":
        plan["implementation_files"][
            "src/bayesian_phystwin/_canonical_contracts.py"
        ] = "f" * 64
    elif case == "missing_code":
        plan["implementation_files"].pop(next(iter(cli.IMPLEMENTATION_FILES)))
    elif case == "extra_code":
        plan["implementation_files"]["another.py"] = "f" * 64
    elif case == "cooldown":
        plan["execution"]["first_request_not_before_utc"] = "2026-09-03T17:08:19+00:00"
    elif case == "resume_delay":
        plan["execution"]["resume_delay_seconds"] = 3600
    elif case == "host":
        plan["execution"]["hostname"] = "another-host"
    elif case == "root":
        plan["execution"]["root"] += "-second-copy"
    elif case == "lock":
        plan["execution"]["lock_path"] += "-second-copy"
    elif case == "method":
        plan["scientific_method_changed"] = True
    elif case == "partial_hash":
        plan["prior_partial_hashes_reused"] = True
    elif case == "legacy":
        plan["legacy_receipt_compatible"] = True
    elif case == "resume_policy":
        plan["resume_policy"] = "automatic"
    elif case == "boundary":
        plan["boundaries"]["confirmation_opened"] = 0
    elif case == "unknown":
        plan["unknown"] = True
    elif case == "revision":
        plan["implementation_revision"] = "main"
    else:
        link = tmp_path / "linked-library"
        link.symlink_to(plan["native_library"]["path"])
        plan["native_library"]["path"] = str(link)
    path, digest = _write_plan(tmp_path, plan)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid amendment reached native loading")

    monkeypatch.setattr(cli, "RHashCheckpointEngine", forbidden)
    with pytest.raises(ValueError):
        cli.load_context(path, expected_amendment_sha256=digest)


def test_import_origin_must_match_bound_source_tree(
    cli: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, digest = _write_plan(tmp_path, _plan(cli, tmp_path))
    monkeypatch.setattr(
        acquisition, "__file__", "/different/source/poseit_checkpoint_acquisition.py"
    )
    with pytest.raises(ValueError, match="import did not use"):
        cli.load_context(path, expected_amendment_sha256=digest)


@pytest.mark.parametrize("case", ["digest", "link"])
def test_amendment_external_hash_and_path_are_required(
    cli: ModuleType,
    tmp_path: Path,
    case: str,
) -> None:
    path, digest = _write_plan(tmp_path, _plan(cli, tmp_path))
    if case == "digest":
        digest = "f" * 64
    else:
        link = tmp_path / "linked-plan.json"
        link.symlink_to(path)
        path = link
    with pytest.raises(ValueError):
        cli.main(
            [
                "preflight",
                "--amendment",
                str(path),
                "--expected-amendment-sha256",
                digest,
            ]
        )


def test_run_requires_authorization_before_loading_amendment(cli: ModuleType) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(
            ["run", "--amendment", "/absent", "--expected-amendment-sha256", "f" * 64]
        )
    assert error.value.code == 2


@pytest.mark.parametrize(
    "mode,call",
    [
        ("run", "run_checkpointed_attempt"),
        ("verify", "verify_completed_receipt"),
        ("publish", "publish_completed_receipt"),
    ],
)
def test_modes_dispatch_only_to_the_registered_action(
    cli: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    call: str,
) -> None:
    path, digest = _write_plan(tmp_path, _plan(cli, tmp_path))
    called = []

    def selected(*args: Any, **kwargs: Any) -> dict[str, Any]:
        called.append((args, kwargs))
        return {"synthetic_dispatch_only": True}

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("unrequested action was dispatched")

    for name in (
        "run_checkpointed_attempt",
        "verify_completed_receipt",
        "publish_completed_receipt",
    ):
        monkeypatch.setattr(acquisition, name, selected if name == call else forbidden)
    argv = [mode, "--amendment", str(path), "--expected-amendment-sha256", digest]
    if mode == "run":
        argv += [
            "--authorization",
            str(tmp_path / "synthetic-authorization.json"),
            "--expected-authorization-sha256",
            "c" * 64,
        ]
    assert cli.main(argv) == 0
    assert len(called) == 1


@pytest.mark.parametrize("mode", ["preflight", "publish", "verify"])
def test_offline_modes_reject_attempt_arguments(cli: ModuleType, mode: str) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                mode,
                "--amendment",
                "/absent",
                "--expected-amendment-sha256",
                "f" * 64,
                "--authorization",
                "/absent-auth",
                "--expected-authorization-sha256",
                "f" * 64,
            ]
        )
    assert error.value.code == 2
