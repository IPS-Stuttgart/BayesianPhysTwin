"""Regression checks for release, citation, typing, and licensing metadata."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import bayesian_phystwin


ROOT = Path(__file__).resolve().parents[1]
DISTRIBUTION_NAME = "bayesian-phystwin"
REPOSITORY_URL = "https://github.com/IPS-Stuttgart/BayesianPhysTwin"
_METADATA = importlib.import_module("importlib.metadata")


def _cff_scalar(key: str) -> str:
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(
        rf'^{re.escape(key)}:\s*(?:"([^"]*)"|([^#\n]+?))\s*$',
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise AssertionError(f"CITATION.cff has no scalar {key!r}")
    return (match.group(1) or match.group(2)).strip()


def test_release_metadata_files_are_present() -> None:
    required = {
        "CHANGELOG.md",
        "CITATION.cff",
        "LICENSE",
        "SUPPORT.md",
        "THIRD_PARTY_NOTICES.md",
    }
    assert {path.name for path in ROOT.iterdir()} >= required


def test_citation_matches_installed_distribution() -> None:
    assert _cff_scalar("version") == _METADATA.version(DISTRIBUTION_NAME)
    assert _cff_scalar("license") == "MIT"
    assert _cff_scalar("repository-code") == REPOSITORY_URL
    assert _cff_scalar("url") == REPOSITORY_URL


def test_distribution_uses_canonical_project_urls() -> None:
    package_metadata = _METADATA.metadata(DISTRIBUTION_NAME)
    project_urls = set(package_metadata.get_all("Project-URL") or ())
    assert f"Repository, {REPOSITORY_URL}" in project_urls
    assert f"Issues, {REPOSITORY_URL}/issues" in project_urls


def test_distribution_declares_spdx_license_expression() -> None:
    package_metadata = _METADATA.metadata(DISTRIBUTION_NAME)
    assert package_metadata["License-Expression"] == "MIT"
    license_files = set(package_metadata.get_all("License-File") or ())
    assert {"LICENSE", "THIRD_PARTY_NOTICES.md"} <= license_files


def test_distribution_contains_pep561_typing_marker() -> None:
    package_metadata = _METADATA.metadata(DISTRIBUTION_NAME)
    classifiers = set(package_metadata.get_all("Classifier") or ())
    assert "Typing :: Typed" in classifiers
    package_file = bayesian_phystwin.__file__
    assert package_file is not None
    marker = Path(package_file).with_name("py.typed")
    assert marker.is_file()


def test_third_party_notice_records_pinned_restrictions() -> None:
    notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    required = (
        "2b6630528141b9cba5a7677c8b88b2129b4a8390",
        "82e02e8029753ad4ef13cf06be7f4fc5facdda4d",
        "Creative Commons Attribution-NonCommercial 4.0",
        "1d6a8947ec6ebabbcf4fc1e0f6d06828fcf6f257",
        "academic purposes",
        "commercial or production use",
        "European Union",
    )
    for term in required:
        assert term in notice
