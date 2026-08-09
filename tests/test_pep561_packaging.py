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
