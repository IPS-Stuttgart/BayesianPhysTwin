from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised by the Python 3.10 CI lane
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _project_metadata() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_evidence_decision_contract_is_in_wheel_and_sdist() -> None:
    metadata = _project_metadata()
    package_data = metadata["tool"]["setuptools"]["package-data"]["bayesian_phystwin"]
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "contract_data/evidence_decision_v1/*.json" in package_data
    assert "include docs/evidence_decision_v1.md" in manifest
    assert (
        "recursive-include "
        "src/bayesian_phystwin/contract_data/evidence_decision_v1 *.json" in manifest
    )
