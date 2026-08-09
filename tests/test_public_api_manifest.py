from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/quality/check_public_api.py"
MANIFEST_PATH = ROOT / "api/root-public-api-v0.4.json"


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_public_api", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()


def _fake_module(symbols: list[str]) -> ModuleType:
    module = ModuleType("bayesian_phystwin")
    module.__all__ = symbols
    for symbol in symbols:
        setattr(module, symbol, object())
    return module


def test_repository_root_api_matches_versioned_snapshot() -> None:
    manifest = tool.load_manifest(MANIFEST_PATH)
    report = tool.validate_public_api(manifest, version="0.4.0")

    assert report == {
        "package": "bayesian_phystwin",
        "project_version": "0.4.0",
        "compatibility_line": "0.4",
        "policy": "exact-legacy-root-export-surface",
        "symbol_count": 184,
        "status": "matched",
    }


def test_root_api_set_drift_is_rejected() -> None:
    manifest = tool.load_manifest(MANIFEST_PATH)
    expected = list(manifest["symbols"])
    module = _fake_module([*expected, "UnexpectedExport"])

    with pytest.raises(tool.PublicApiError, match="root API set changed"):
        tool.validate_public_api(manifest, module=module, version="0.4.1")


def test_root_api_reordering_is_explicitly_reviewed() -> None:
    manifest = tool.load_manifest(MANIFEST_PATH)
    reordered = list(manifest["symbols"])
    reordered[0], reordered[1] = reordered[1], reordered[0]

    with pytest.raises(tool.PublicApiError, match="root API order changed"):
        tool.validate_public_api(
            manifest,
            module=_fake_module(reordered),
            version="0.4.1",
        )


def test_manifest_rejects_duplicates_and_unknown_fields(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["symbols"].append(payload["symbols"][0])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(tool.PublicApiError, match="duplicate symbols"):
        tool.load_manifest(path)

    payload["symbols"].pop()
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(tool.PublicApiError, match="fields changed"):
        tool.load_manifest(path)


def test_manifest_compatibility_line_tracks_project_version() -> None:
    manifest = tool.load_manifest(MANIFEST_PATH)

    with pytest.raises(tool.PublicApiError, match="outside the manifest"):
        tool.validate_public_api(manifest, version="0.5.0")
