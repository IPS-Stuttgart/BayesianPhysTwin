from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup.py"
CONTRACT = ROOT / "release" / "stable_distribution_contract_v1.json"


class _FakeSdist:
    last_release_tree: ClassVar[tuple[str, tuple[str, ...]] | None] = None

    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        _FakeSdist.last_release_tree = (base_dir, tuple(files))


def _load_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, Any]]:
    captured: dict[str, Any] = {}

    def capture_setup(**kwargs: Any) -> None:
        captured["setup"] = kwargs

    def capture_find_namespace_packages(**kwargs: Any) -> list[str]:
        captured["find_namespace_packages"] = kwargs
        return [
            "bayesian_phystwin",
            "bayesian_phystwin.cli",
            "bayesian_phystwin.inference",
        ]

    setuptools_module = ModuleType("setuptools")
    command_module = ModuleType("setuptools.command")
    sdist_module = ModuleType("setuptools.command.sdist")
    setuptools_module.__dict__.update(
        setup=capture_setup,
        find_namespace_packages=capture_find_namespace_packages,
        command=command_module,
    )
    command_module.__dict__["sdist"] = sdist_module
    sdist_module.__dict__["sdist"] = _FakeSdist
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
    setup_arguments = captured["setup"]

    assert namespace["SUPPORTED_SDIST_SELF_TESTS"] == supported
    assert setup_arguments["cmdclass"]["sdist"] is namespace["StableSdist"]


def test_setup_binds_explicit_src_package_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, captured = _load_setup(monkeypatch)
    setup_arguments = captured["setup"]

    assert captured["find_namespace_packages"] == {
        "where": "src",
        "include": namespace["PACKAGE_INCLUDE"],
        "exclude": namespace["PACKAGE_EXCLUDE"],
    }
    assert namespace["DISCOVERED_PACKAGES"] == (
        "bayesian_phystwin",
        "bayesian_phystwin.cli",
        "bayesian_phystwin.inference",
    )
    assert setup_arguments["package_dir"] == {"": "src"}
    assert setup_arguments["packages"] == namespace["DISCOVERED_PACKAGES"]
    assert all(
        package == "bayesian_phystwin" or package.startswith("bayesian_phystwin.")
        for package in setup_arguments["packages"]
    )


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
