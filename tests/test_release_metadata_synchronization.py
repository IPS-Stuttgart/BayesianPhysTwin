from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from tools.release.synchronize_release_metadata import (
    ReleaseSpecification,
    synchronize_changelog,
    synchronize_citation,
    synchronize_repository,
)

ROOT = Path(__file__).resolve().parents[1]
SPECIFICATION = ReleaseSpecification(
    version="0.4.0",
    release_date=date(2026, 8, 11),
)


def _write_fixture(root: Path) -> None:
    (root / "docs").mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nversion = "0.4.0"\n',
        encoding="utf-8",
    )
    (root / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n### Added\n\n- Evidence.\n",
        encoding="utf-8",
    )
    (root / "CITATION.cff").write_text(
        "cff-version: 1.2.0\ntitle: Bayesian PhysTwin\nversion: 0.3.0\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# Bayesian PhysTwin\n\n"
        "widths of approximately `38.87/42.68 mm` for CD/track.\n\n"
        "## Architecture\n",
        encoding="utf-8",
    )
    (root / "SUPPORT.md").write_text(
        "# Support\n\n## Reporting problems\n",
        encoding="utf-8",
    )
    (root / "docs/phystwin_release_claim_v1.md").write_text(
        "# Claim\n\n"
        "## Uncertainty claim boundary\n\n"
        "## Required release wording\n\n"
        "- raw posterior covariance is severely undercalibrated;\n"
        "- independent transfer remains unconfirmed.\n",
        encoding="utf-8",
    )


def test_release_specification_requires_canonical_version() -> None:
    with pytest.raises(ValueError, match="canonical X.Y.Z"):
        ReleaseSpecification(version="v0.4", release_date=date(2026, 8, 11))


def test_changelog_cut_is_idempotent_and_date_bound() -> None:
    source = "# Changelog\n\n## [Unreleased]\n\n### Added\n"

    released = synchronize_changelog(source, SPECIFICATION)
    assert "## [0.4.0] - 2026-08-11" in released
    assert synchronize_changelog(released, SPECIFICATION) == released

    changed_date = ReleaseSpecification(
        version="0.4.0",
        release_date=date(2026, 8, 12),
    )
    with pytest.raises(ValueError, match="release date contradicts"):
        synchronize_changelog(released, changed_date)


def test_citation_version_and_release_date_are_idempotent() -> None:
    source = "cff-version: 1.2.0\nversion: 0.3.0\n"

    released = synchronize_citation(source, SPECIFICATION)
    assert "version: 0.4.0" in released
    assert "date-released: 2026-08-11" in released
    assert synchronize_citation(released, SPECIFICATION) == released


def test_repository_synchronization_preserves_bounded_claim_wording(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)

    changed = synchronize_repository(tmp_path, SPECIFICATION, check=False)
    assert {path.relative_to(tmp_path).as_posix() for path in changed} == {
        "CHANGELOG.md",
        "CITATION.cff",
        "README.md",
        "SUPPORT.md",
        "docs/phystwin_release_claim_v1.md",
    }
    assert synchronize_repository(tmp_path, SPECIFICATION, check=True) == ()

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "trajectory unchanged in all `22/22` units" in readme
    assert "`3.10×`" in readme
    assert "not independent calibration" in readme

    claim = (tmp_path / "docs/phystwin_release_claim_v1.md").read_text(encoding="utf-8")
    assert "## Exact-mean covariance-only mechanism" in claim
    assert "`[-13.961, -4.312]`" in claim
    assert "the exact-mean covariance-only result is retrospective" in claim

    support = (tmp_path / "SUPPORT.md").read_text(encoding="utf-8")
    assert "## Release evidence and scientific wording" in support
    assert "exact `numpy==1.23.0`" in support


def test_repository_release_metadata_is_synchronized() -> None:
    assert synchronize_repository(ROOT, SPECIFICATION, check=True) == ()
