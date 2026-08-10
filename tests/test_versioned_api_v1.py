from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "api/versioned-public-api-v1.json"


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


def test_versioned_api_manifest_is_part_of_the_source_distribution() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()

    assert "include api/versioned-public-api-v1.json" in manifest
