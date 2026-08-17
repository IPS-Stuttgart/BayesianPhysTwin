from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/quality/check_public_api.py"
MIGRATION_TOOL_PATH = ROOT / "tools/quality/check_root_export_migration.py"
MANIFEST_PATH = ROOT / "api/root-public-api-v0.4.json"
VERSIONED_MANIFEST_PATH = ROOT / "api/versioned-public-api-v1.json"
INFERENCE_MANIFEST_PATH = ROOT / "api/inference-public-api-v1.json"
MIGRATION_MANIFEST_PATH = ROOT / "api/root-export-migration-v1.json"


def _tool(path: Path, *, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool(TOOL_PATH, name="check_public_api")
migration_tool = _tool(
    MIGRATION_TOOL_PATH,
    name="check_root_export_migration",
)


def _fake_module(symbols: list[str], *, name: str = "bayesian_phystwin") -> ModuleType:
    module = ModuleType(name)
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


def test_versioned_integration_api_matches_snapshot() -> None:
    manifest = tool.load_manifest(VERSIONED_MANIFEST_PATH)
    report = tool.validate_public_api(manifest, version="0.4.0")

    assert report == {
        "package": "bayesian_phystwin.v1",
        "project_version": "0.4.0",
        "compatibility_line": "0.4",
        "policy": "exact-versioned-integration-export-surface",
        "symbol_count": 38,
        "status": "matched",
    }


def test_guarded_inference_api_matches_snapshot() -> None:
    manifest = tool.load_manifest(INFERENCE_MANIFEST_PATH)
    report = tool.validate_public_api(manifest, version="0.4.0")

    assert report == {
        "package": "bayesian_phystwin.inference.v1",
        "project_version": "0.4.0",
        "compatibility_line": "0.4",
        "policy": "exact-guarded-inference-export-surface",
        "symbol_count": 12,
        "status": "matched",
    }


def test_root_export_migration_matches_runtime_owners() -> None:
    report = migration_tool.validate_root_export_migration(
        MIGRATION_MANIFEST_PATH,
        root_manifest_path=MANIFEST_PATH,
    )

    assert report == {
        "source_package": "bayesian_phystwin",
        "source_compatibility_line": "0.4",
        "target_compatibility_line": "0.5",
        "policy": "lazy-legacy-root-to-owning-module",
        "owner_count": 30,
        "symbol_count": 184,
        "resolved": True,
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


def test_versioned_api_reordering_is_explicitly_reviewed() -> None:
    manifest = tool.load_manifest(VERSIONED_MANIFEST_PATH)
    reordered = list(manifest["symbols"])
    reordered[0], reordered[1] = reordered[1], reordered[0]

    with pytest.raises(tool.PublicApiError, match="bayesian_phystwin.v1 API order"):
        tool.validate_public_api(
            manifest,
            module=_fake_module(reordered, name="bayesian_phystwin.v1"),
            version="0.4.1",
        )


def test_inference_api_reordering_is_explicitly_reviewed() -> None:
    manifest = tool.load_manifest(INFERENCE_MANIFEST_PATH)
    reordered = list(manifest["symbols"])
    reordered[0], reordered[1] = reordered[1], reordered[0]

    with pytest.raises(
        tool.PublicApiError,
        match="bayesian_phystwin.inference.v1 API order",
    ):
        tool.validate_public_api(
            manifest,
            module=_fake_module(
                reordered,
                name="bayesian_phystwin.inference.v1",
            ),
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


def test_manifest_rejects_contract_substitution(tmp_path: Path) -> None:
    payload = json.loads(VERSIONED_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["policy"] = "exact-legacy-root-export-surface"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(tool.PublicApiError, match="policy changed"):
        tool.load_manifest(path)

    payload["policy"] = "exact-versioned-integration-export-surface"
    payload["package"] = "bayesian_phystwin"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(tool.PublicApiError, match="package changed"):
        tool.load_manifest(path)

    payload["package"] = "bayesian_phystwin.v1"
    payload["schema"] = "bayesian-phystwin.unknown-api-snapshot"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(tool.PublicApiError, match="contract changed"):
        tool.load_manifest(path)


def test_migration_manifest_rejects_duplicate_symbols(tmp_path: Path) -> None:
    payload = json.loads(MIGRATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["owners"][0]["symbols"].append(payload["owners"][0]["symbols"][0])
    path = tmp_path / "migration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        migration_tool.RootExportMigrationError,
        match="duplicate export symbol",
    ):
        migration_tool.load_migration_manifest(path)


def test_migration_manifest_rejects_runtime_owner_drift(tmp_path: Path) -> None:
    payload = json.loads(MIGRATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["owners"][0]["module"] = "bayesian_phystwin.wrong_owner"
    path = tmp_path / "migration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        migration_tool.RootExportMigrationError,
        match="runtime migration mapping differs",
    ):
        migration_tool.validate_root_export_migration(
            path,
            root_manifest_path=MANIFEST_PATH,
            resolve=False,
        )


def test_migration_manifest_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = json.loads(MIGRATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path = tmp_path / "migration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        migration_tool.RootExportMigrationError,
        match="fields changed",
    ):
        migration_tool.load_migration_manifest(path)


def test_manifest_compatibility_line_tracks_project_version() -> None:
    manifest = tool.load_manifest(MANIFEST_PATH)

    with pytest.raises(tool.PublicApiError, match="outside the manifest"):
        tool.validate_public_api(manifest, version="0.5.0")


def test_project_version_parser_is_python_3_10_compatible(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[build-system]\n"
        'requires = ["setuptools"]\n\n'
        "[project]\n"
        'name = "example"\n'
        'version = "0.4.7"  # exact compatibility line\n\n'
        "[project.urls]\n"
        'Repository = "https://example.invalid"\n',
        encoding="utf-8",
    )

    assert tool.project_version(pyproject) == "0.4.7"

    pyproject.write_text(
        '[project]\nversion = "0.4.7"\nversion = "0.4.8"\n',
        encoding="utf-8",
    )
    with pytest.raises(tool.PublicApiError, match="one literal version"):
        tool.project_version(pyproject)


def test_migration_contract_is_part_of_the_source_distribution() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()

    assert "include api/root-export-migration-v1.json" in manifest
    assert "include tools/quality/check_root_export_migration.py" in manifest
