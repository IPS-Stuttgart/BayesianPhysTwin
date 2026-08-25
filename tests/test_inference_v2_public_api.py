from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/quality/check_public_api.py"
MANIFEST_PATH = ROOT / "api/inference-session-public-api-v2.json"


def _tool() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "check_inference_session_public_api",
        TOOL_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


tool = _tool()


def _fake_module(symbols: list[str]) -> ModuleType:
    module = ModuleType("bayesian_phystwin.inference.v2")
    module.__all__ = symbols
    for symbol in symbols:
        setattr(module, symbol, object())
    return module


def test_provider_neutral_session_api_matches_snapshot() -> None:
    manifest = tool.load_manifest(MANIFEST_PATH)
    report = tool.validate_public_api(manifest, version="0.4.0")

    assert report == {
        "package": "bayesian_phystwin.inference.v2",
        "project_version": "0.4.0",
        "compatibility_line": "0.4",
        "policy": "exact-provider-neutral-session-export-surface",
        "symbol_count": 10,
        "status": "matched",
    }


def test_provider_neutral_session_api_reordering_requires_review() -> None:
    manifest = tool.load_manifest(MANIFEST_PATH)
    reordered = list(manifest["symbols"])
    reordered[0], reordered[1] = reordered[1], reordered[0]

    with pytest.raises(
        tool.PublicApiError,
        match="bayesian_phystwin.inference.v2 API order",
    ):
        tool.validate_public_api(
            manifest,
            module=_fake_module(reordered),
            version="0.4.1",
        )
