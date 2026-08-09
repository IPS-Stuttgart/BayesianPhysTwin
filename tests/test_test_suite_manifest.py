from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/quality/test_suite_manifest.py"
WORKFLOW_PATH = ROOT / ".github/workflows/tests.yml"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("test_suite_manifest", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _minimal_payload(suites: dict[str, list[str]]) -> dict[str, object]:
    return {
        "schema": "bayesian-phystwin.ci-test-suites",
        "schema_version": 1,
        "suites": suites,
        "subset_requirements": [],
    }


def test_repository_manifest_expands_deterministically() -> None:
    module = _module()
    suites = module.load_test_suites()

    assert set(suites) == {
        "stable-core-coverage",
        "core-contracts",
        "provider-contract",
    }
    assert set(suites["core-contracts"]) <= set(suites["stable-core-coverage"])
    assert set(suites["provider-contract"]) <= set(
        suites["stable-core-coverage"]
    )
    for files in suites.values():
        assert files
        assert len(files) == len(set(files))
        assert all(path.startswith("tests/") and path.endswith(".py") for path in files)


def test_workflow_consumes_manifest_without_collection_injection() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "test_suite_manifest.py list stable-core-coverage" in workflow
    assert "test_suite_manifest.py list core-contracts" in workflow
    assert "test_suite_manifest.py list provider-contract" in workflow
    assert "tests/test_causal4d_graph_provider_v1.py \\" not in workflow
    assert not (ROOT / "tests/conftest.py").exists()


def test_manifest_rejects_overlapping_patterns(tmp_path: Path) -> None:
    module = _module()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_a.py").write_text("\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        _minimal_payload(
            {
                "stable": [
                    "tests/test_a.py",
                    "tests/test_*.py",
                ]
            }
        ),
    )

    with pytest.raises(module.ManifestError, match="through both"):
        module.load_test_suites(manifest, repository_root=tmp_path)


def test_manifest_rejects_empty_globs_and_unknown_fields(tmp_path: Path) -> None:
    module = _module()
    (tmp_path / "tests").mkdir()
    manifest = tmp_path / "manifest.json"
    payload = _minimal_payload({"stable": ["tests/test_missing*.py"]})
    _write_manifest(manifest, payload)

    with pytest.raises(module.ManifestError, match="matched no files"):
        module.load_test_suites(manifest, repository_root=tmp_path)

    payload["unexpected"] = True
    _write_manifest(manifest, payload)
    with pytest.raises(module.ManifestError, match="fields changed"):
        module.load_test_suites(manifest, repository_root=tmp_path)


def test_manifest_enforces_declared_suite_subsets(tmp_path: Path) -> None:
    module = _module()
    (tmp_path / "tests").mkdir()
    for name in ("test_a.py", "test_b.py"):
        (tmp_path / "tests" / name).write_text("\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    payload = _minimal_payload(
        {
            "stable": ["tests/test_a.py"],
            "core": ["tests/test_b.py"],
        }
    )
    payload["subset_requirements"] = [
        {
            "subset": "core",
            "superset": "stable",
        }
    ]
    _write_manifest(manifest, payload)

    with pytest.raises(module.ManifestError, match="is not contained"):
        module.load_test_suites(manifest, repository_root=tmp_path)


def test_manifest_rejects_symlinked_tests(tmp_path: Path) -> None:
    module = _module()
    (tmp_path / "tests").mkdir()
    target = tmp_path / "tests/target.py"
    target.write_text("\n", encoding="utf-8")
    link = tmp_path / "tests/test_link.py"
    link.symlink_to(target)
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        _minimal_payload({"stable": ["tests/test_link.py"]}),
    )

    with pytest.raises(module.ManifestError, match="must not be a symlink"):
        module.load_test_suites(manifest, repository_root=tmp_path)
