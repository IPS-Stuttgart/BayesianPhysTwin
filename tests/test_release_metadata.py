"""Regression checks for release, citation, and licensing metadata."""

from __future__ import annotations

import re
from importlib.metadata import metadata, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISTRIBUTION_NAME = "bayesian-phystwin"


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
    assert _cff_scalar("version") == version(DISTRIBUTION_NAME)
    assert _cff_scalar("license") == "MIT"
    assert _cff_scalar("repository-code") == (
        "https://github.com/IPS-Stuttgart/BayesianPhysTwin"
    )


def test_distribution_links_use_canonical_organization_repository() -> None:
    package_metadata = metadata(DISTRIBUTION_NAME)
    project_urls = {}
    for raw_value in package_metadata.get_all("Project-URL") or ():
        label, separator, url = raw_value.partition(",")
        assert separator
        project_urls[label.strip()] = url.strip()
    canonical = "https://github.com/IPS-Stuttgart/BayesianPhysTwin"
    assert project_urls["Repository"] == canonical
    assert project_urls["Documentation"] == f"{canonical}/tree/main/docs"
    assert project_urls["Issues"] == f"{canonical}/issues"
    assert _cff_scalar("url") == canonical
    assert _cff_scalar("repository-code") == canonical
    for path in (ROOT / "README.md", ROOT / "THIRD_PARTY_NOTICES.md"):
        text = path.read_text(encoding="utf-8")
        assert "github.com/FlorianPfaff/Bayesian-PhysTwin" not in text
        assert "github.com/FlorianPfaff/Prob4D" not in text
        assert "github.com/FlorianPfaff/Causal4D" not in text


def test_distribution_declares_spdx_license_expression() -> None:
    package_metadata = metadata(DISTRIBUTION_NAME)
    assert package_metadata["License-Expression"] == "MIT"
    license_files = set(package_metadata.get_all("License-File") or ())
    assert {"LICENSE", "THIRD_PARTY_NOTICES.md"} <= license_files


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
