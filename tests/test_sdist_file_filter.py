from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup.py"
CONTRACT = ROOT / "release" / "stable_distribution_contract_v1.json"


class _FakeSdist:
    last_release_tree: tuple[str, tuple[str, ...]] | None = None

    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        type(self).last_release_tree = (base_dir, tuple(files))


def _load_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, Any]]:
    captured: dict[str, Any] = {}

    def capture_setup(**kwargs: Any) -> None:
        captured.update(kwargs)

    setuptools_module = ModuleType("setuptools")
    command_module = ModuleType("setuptools.command")
    sdist_module = ModuleType("setuptools.command.sdist")
    setattr(setuptools_module, "setup", capture_setup)
    setattr(setuptools_module, "command", command_module)
    setattr(command_module, "sdist", sdist_module)
    setattr(sdist_module, "sdist", _FakeSdist)
    monkeypatch.setitem(sys.modules, "setuptools", setuptools_module)
    monkeypatch.setitem(sys.modules, "setuptools.command", command_module)
    monkeypatch.setitem(sys.modules, "setuptools.command.sdist", sdist_module)

    namespace = runpy.run_path(str(SETUP))
    return namespace, captured


def test_sdist_filter_uses_exact_contract_self_test_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, captured = _load_setup(monkeypatch)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    supported = frozenset(contract["sdist"]["supported_self_test_files"])

    assert namespace["SUPPORTED_SDIST_SELF_TESTS"] == supported
    assert captured["cmdclass"]["sdist"] is namespace["StableSdist"]


def test_sdist_filter_prunes_repository_only_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, _ = _load_setup(monkeypatch)
    stable_sdist = namespace["StableSdist"]()
    _FakeSdist.last_release_tree = None
    stable_sdist.make_release_tree(
        "release-tree",
        [
            "src/bayesian_phystwin/__init__.py",
            "tests/test_stable_distribution_contract.py",
            r"tests\test_versioned_api_v1.py",
            "tests/test_ecosystem_current_actions.py",
            "tests/test_sdist_file_filter.py",
        ],
    )

    assert _FakeSdist.last_release_tree == (
        "release-tree",
        (
            "src/bayesian_phystwin/__init__.py",
            "tests/test_stable_distribution_contract.py",
            r"tests\test_versioned_api_v1.py",
        ),
    )
