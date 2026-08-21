from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import setuptools

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup.py"
CONTRACT = ROOT / "release" / "stable_distribution_contract_v1.json"


def _load_setup(monkeypatch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    captured: dict[str, Any] = {}

    def capture_setup(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(setuptools, "setup", capture_setup)
    namespace = runpy.run_path(str(SETUP))
    return namespace, captured


def test_sdist_filter_uses_exact_contract_self_test_boundary(monkeypatch: Any) -> None:
    namespace, captured = _load_setup(monkeypatch)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    supported = frozenset(contract["sdist"]["supported_self_test_files"])

    assert namespace["SUPPORTED_SDIST_SELF_TESTS"] == supported
    assert captured["cmdclass"]["sdist"] is namespace["StableSdist"]


def test_sdist_filter_prunes_repository_only_tests(monkeypatch: Any) -> None:
    namespace, _ = _load_setup(monkeypatch)
    retain = namespace["_retain_sdist_file"]

    assert retain("src/bayesian_phystwin/__init__.py")
    assert retain("tests/test_stable_distribution_contract.py")
    assert retain(r"tests\test_versioned_api_v1.py")
    assert not retain("tests/test_ecosystem_current_actions.py")
    assert not retain("tests/test_sdist_file_filter.py")
