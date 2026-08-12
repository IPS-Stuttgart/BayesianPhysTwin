from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ROOT_MANIFEST_PATH = ROOT / "api/root-public-api-v0.4.json"
MANIFEST_PATH = ROOT / "api/versioned-public-api-v1.json"
MIGRATION_PATH = ROOT / "api/root-public-api-migration-v0.5.json.gz"
MIGRATION_TOOL = ROOT / "tools/quality/generate_root_api_migration.py"


def _isolated(script: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_versioned_api_is_deliberately_small_and_frozen() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    v1 = __import__("bayesian_phystwin.v1", fromlist=["*"])

    assert manifest["schema"] == "bayesian-phystwin.versioned-public-api-snapshot"
    assert manifest["schema_version"] == 1
    assert manifest["package"] == "bayesian_phystwin.v1"
    assert manifest["compatibility_line"] == "0.4"
    assert manifest["policy"] == "exact-versioned-integration-export-surface"
    assert list(v1.__all__) == manifest["symbols"]
    assert len(v1.__all__) == len(set(v1.__all__))
    for name in manifest["symbols"]:
        assert getattr(v1, name) is not None


def test_package_root_import_is_lazy_and_preserves_the_frozen_surface() -> None:
    expected = json.loads(ROOT_MANIFEST_PATH.read_text(encoding="utf-8"))["symbols"]
    script = f"""
import json
import sys
sys.path.insert(0, {str(SRC)!r})
import bayesian_phystwin
print(json.dumps({{
    "exports": bayesian_phystwin.__all__,
    "submodules": sorted(
        name for name in sys.modules if name.startswith("bayesian_phystwin.")
    ),
}}))
"""
    result = _isolated(script)

    assert result["exports"] == expected
    assert result["submodules"] == []


def test_root_export_load_is_scoped_and_cached() -> None:
    script = f"""
import json
import sys
sys.path.insert(0, {str(SRC)!r})
import bayesian_phystwin
first = bayesian_phystwin.RobustLikelihoodConfig
second = bayesian_phystwin.RobustLikelihoodConfig
print(json.dumps({{
    "cached": first is second is bayesian_phystwin.__dict__["RobustLikelihoodConfig"],
    "module": first.__module__,
    "submodules": sorted(
        name for name in sys.modules if name.startswith("bayesian_phystwin.")
    ),
}}))
"""
    result = _isolated(script)

    assert result["cached"] is True
    assert result["module"] == "bayesian_phystwin.robust_likelihood"
    assert "bayesian_phystwin.robust_likelihood" in result["submodules"]
    assert not any(
        name.startswith("bayesian_phystwin.deform360")
        for name in result["submodules"]
    )


def test_unknown_root_attribute_uses_normal_module_error_semantics() -> None:
    package = __import__("bayesian_phystwin")

    try:
        getattr(package, "not_a_public_export")
    except AttributeError as error:
        assert "not_a_public_export" in str(error)
    else:  # pragma: no cover - regression guard
        raise AssertionError("unknown package-root attribute was accepted")


def test_versioned_api_import_skips_research_and_optional_modules() -> None:
    script = f"""
import json
import sys
sys.path.insert(0, {str(SRC)!r})
import bayesian_phystwin.v1

forbidden = [
    "bayesian_phystwin.claim_bearing_prob4d",
    "bayesian_phystwin.deform360_calibration_factor_materializer",
    "bayesian_phystwin.deform360_contact_anchor",
    "bayesian_phystwin.deform360_public_contact_prefix",
    "bayesian_phystwin.deform360_visual_provider_lock",
    "bayesian_phystwin.endpoint_model_average",
    "bayesian_phystwin.gauge_aware_belief",
    "bayesian_phystwin.material_identity_marginalization",
    "bayesian_phystwin.phystwin_adapter",
    "bayesian_phystwin.synthetic_benchmark",
    "cv2",
    "h5py",
    "remotezip",
    "scipy",
]
print(json.dumps({{"loaded": [name for name in forbidden if name in sys.modules]}}))
"""
    result = _isolated(script)

    assert result["loaded"] == []


def test_root_migration_map_is_complete_current_and_support_aware() -> None:
    root_symbols = json.loads(
        ROOT_MANIFEST_PATH.read_text(encoding="utf-8")
    )["symbols"]
    v1_symbols = set(json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["symbols"])
    payload = json.loads(gzip.decompress(MIGRATION_PATH.read_bytes()))
    entries = payload["symbols"]

    assert payload["schema"] == "bayesian-phystwin.root-api-migration"
    assert payload["source_compatibility_line"] == "0.4"
    assert payload["target_compatibility_line"] == "0.5"
    assert [entry["name"] for entry in entries] == root_symbols
    assert len(entries) == len({entry["name"] for entry in entries})
    for entry in entries:
        stable = entry["name"] in v1_symbols
        assert entry["support"] == (
            "stable-v1" if stable else "legacy-root-compatibility"
        )
        assert entry["preferred_import"] == (
            "bayesian_phystwin.v1" if stable else entry["defining_module"]
        )

    subprocess.run(
        [sys.executable, str(MIGRATION_TOOL), "--check"],
        cwd=ROOT,
        check=True,
    )


def test_versioned_api_assets_are_part_of_distributions() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    stub = (SRC / "bayesian_phystwin/__init__.pyi").read_text(encoding="utf-8")

    assert "include api/versioned-public-api-v1.json" in manifest
    assert "include api/root-public-api-migration-v0.5.json.gz" in manifest
    assert "include tools/quality/generate_root_api_migration.py" in manifest
    assert "include src/bayesian_phystwin/__init__.pyi" in manifest
    assert '"*.pyi"' in pyproject
    assert "from ._legacy_root_eager import *" in stub
