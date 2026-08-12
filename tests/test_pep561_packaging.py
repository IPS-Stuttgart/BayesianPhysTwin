from __future__ import annotations

import sys
from importlib.resources import files
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised by the Python 3.10 CI lane
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
INSTALLED_TYPING_WORKFLOW = ROOT / ".github/workflows/installed-typing-contract.yml"


def _project_metadata() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_source_package_exposes_pep561_marker() -> None:
    marker = files("bayesian_phystwin").joinpath("py.typed")

    assert marker.is_file()
    assert "PEP 561" in marker.read_text(encoding="utf-8")


def test_project_metadata_declares_typed_package_data() -> None:
    metadata = _project_metadata()
    classifiers = metadata["project"]["classifiers"]
    package_data = metadata["tool"]["setuptools"]["package-data"]["bayesian_phystwin"]

    assert "Typing :: Typed" in classifiers
    assert "py.typed" in package_data
    assert "contract_data/observation_belief_v1/*.json" in package_data
    assert "contract_data/observation_belief_v1/vectors/*.json" in package_data


def test_installed_typing_workflow_tracks_public_type_surfaces() -> None:
    workflow = INSTALLED_TYPING_WORKFLOW.read_text(encoding="utf-8")
    tracked_paths = (
        "api/root-public-api-v0.4.json",
        "api/versioned-public-api-v1.json",
        "integration_tests/typing_consumer.py",
        "src/bayesian_phystwin/__init__.py",
        "src/bayesian_phystwin/claim_bundle_v1.py",
        "src/bayesian_phystwin/evidence_decision_v1.py",
        "src/bayesian_phystwin/gauge_aware_belief.py",
        "src/bayesian_phystwin/observation_belief.py",
        "src/bayesian_phystwin/physical_query_v1.py",
        "src/bayesian_phystwin/repository_provenance.py",
        "src/bayesian_phystwin/run_manifest.py",
        "src/bayesian_phystwin/run_manifest_v2.py",
        "src/bayesian_phystwin/v1/**",
    )

    for path in tracked_paths:
        assert workflow.count(f'- "{path}"') == 2
