from __future__ import annotations

from types import SimpleNamespace

import pytest

from bayesian_phystwin import deform360_held_v83_viser_guard as guard


class _ExitedProcess:
    def __init__(self, error: type[BaseException]) -> None:
        self.error = error

    def cwd(self) -> str:
        raise self.error()

    def cmdline(self) -> list[str]:
        raise AssertionError("cmdline must not run after cwd failure")


class _LiveProcess:
    def __init__(self, cwd: str, command: list[str]) -> None:
        self._cwd = cwd
        self._command = command

    def cwd(self) -> str:
        return self._cwd

    def cmdline(self) -> list[str]:
        return list(self._command)


class _CmdlineExitedProcess:
    def __init__(self, error: type[BaseException]) -> None:
        self.error = error

    def cwd(self) -> str:
        return "/tmp/viser/client"

    def cmdline(self) -> list[str]:
        raise self.error()


def test_process_scan_ignores_churn_and_finds_live_yarn() -> None:
    class NoSuchProcess(Exception):
        pass

    class AccessDenied(Exception):
        pass

    class ZombieProcess(Exception):
        pass

    assert (
        guard._check_viser_yarn_running(
            [
                _ExitedProcess(NoSuchProcess),
                _CmdlineExitedProcess(NoSuchProcess),
                _ExitedProcess(AccessDenied),
                _ExitedProcess(ZombieProcess),
                _LiveProcess("/tmp/viser/client", ["/usr/bin/yarn", "start"]),
            ],
            ignored_exceptions=(AccessDenied, ZombieProcess, NoSuchProcess),
        )
        is True
    )


def test_process_scan_returns_false_without_matching_yarn() -> None:
    assert (
        guard._check_viser_yarn_running(
            [
                _LiveProcess("/tmp/viser/client", ["python", "worker.py"]),
                _LiveProcess("/tmp/other", ["/usr/bin/yarn.js", "start"]),
            ],
            ignored_exceptions=(RuntimeError,),
        )
        is False
    )


def test_installer_is_byte_bound_idempotent_and_pretrainer(
    monkeypatch,
    tmp_path,
) -> None:
    source = tmp_path / "_client_autobuild.py"
    source.write_bytes(b"pinned viser source")

    class AccessDenied(Exception):
        pass

    class ZombieProcess(Exception):
        pass

    class NoSuchProcess(Exception):
        pass

    def original() -> bool:
        return False

    original.__module__ = "viser._client_autobuild"
    original.__name__ = "_check_viser_yarn_running"
    autobuild = SimpleNamespace(
        __file__=str(source),
        __name__="viser._client_autobuild",
        _check_viser_yarn_running=original,
    )
    psutil = SimpleNamespace(
        AccessDenied=AccessDenied,
        ZombieProcess=ZombieProcess,
        NoSuchProcess=NoSuchProcess,
        process_iter=lambda: [_ExitedProcess(NoSuchProcess)],
    )

    def fake_import(name: str):
        return {
            "viser._client_autobuild": autobuild,
            "psutil": psutil,
        }[name]

    monkeypatch.setattr(guard.importlib, "import_module", fake_import)
    monkeypatch.setattr(
        guard,
        "UPSTREAM_CLIENT_AUTOBUILD_SHA256",
        guard._sha256_file(source),
    )
    monkeypatch.setattr(guard, "_INSTALLED_EVIDENCE", None)
    monkeypatch.setattr(guard, "_INSTALLED_HELPER", None)

    first = guard.install_viser_process_churn_guard()
    second = guard.install_viser_process_churn_guard()

    assert first == second
    unsigned = dict(first)
    artifact_sha256 = unsigned.pop("artifact_sha256")
    assert artifact_sha256 == guard._artifact(unsigned)["artifact_sha256"]
    assert first["guard_installed_before_original_trainer_import"] is True
    assert first["target_or_outcome_path_accessed"] is False
    assert autobuild._check_viser_yarn_running() is False


def test_installer_rejects_changed_upstream_source(
    monkeypatch,
    tmp_path,
) -> None:
    source = tmp_path / "_client_autobuild.py"
    source.write_bytes(b"changed viser source")
    autobuild = SimpleNamespace(
        __file__=str(source),
        __name__="viser._client_autobuild",
        _check_viser_yarn_running=lambda: False,
    )
    monkeypatch.setattr(
        guard.importlib,
        "import_module",
        lambda name: autobuild if name == "viser._client_autobuild" else None,
    )
    monkeypatch.setattr(guard, "_INSTALLED_EVIDENCE", None)
    monkeypatch.setattr(guard, "_INSTALLED_HELPER", None)

    with pytest.raises(
        RuntimeError,
        match="pinned Viser client-autobuild source changed",
    ):
        guard.install_viser_process_churn_guard()


def test_installer_rejects_posttrainer_installation(monkeypatch) -> None:
    monkeypatch.setattr(guard, "_INSTALLED_EVIDENCE", None)
    monkeypatch.setattr(guard, "_INSTALLED_HELPER", None)
    monkeypatch.setitem(
        guard.sys.modules,
        "deform360.processing.reconstruct_stage",
        SimpleNamespace(),
    )

    with pytest.raises(
        RuntimeError,
        match="must be installed before original trainer import",
    ):
        guard.install_viser_process_churn_guard()
