"""Synchronize one reviewed BayesianPhysTwin software release.

The tool deliberately edits only release metadata and bounded scientific wording.
It does not create a tag, publish a distribution, or promote a scientific claim.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_CHANGELOG_VERSION_PATTERN = re.compile(
    r"^## \[([0-9]+\.[0-9]+\.[0-9]+)\](?: - ([0-9]{4}-[0-9]{2}-[0-9]{2}))?$",
    re.MULTILINE,
)

_README_MARKER = (
    "widths of approximately `38.87/42.68 mm` for CD/track.\n\n"
)
_README_BLOCK = (
    "A retrospective exact-mean covariance-only analysis leaves the "
    "`last_residual`\n"
    "trajectory unchanged in all `22/22` units. Cross-fitted covariance "
    "changes\n"
    "Gaussian NLL by `-9.136` with simultaneous interval "
    "`[-13.961, -4.312]`,\n"
    "improves `17/22` units, and changes marginal 90% coverage from `70.6%` "
    "to\n"
    "`91.0%`. The mean full interval width increases from `16.45` to "
    "`50.94 mm`\n"
    "(`3.10×`), while Chamfer distance and track error remain exactly "
    "unchanged.\n"
    "This freezes one candidate for a fresh object/session study; it is not\n"
    "independent calibration or deployment authorization.\n\n"
)

_CLAIM_MARKER = "## Uncertainty claim boundary\n"
_CLAIM_SECTION = """## Exact-mean covariance-only mechanism

A separate retrospective analysis preserves the exact `last_residual` point
trajectory in every one of the `22/22` object/session units and changes only the
predictive covariance. Its primary cross-fitted result is:

| Diagnostic | Result |
| --- | ---: |
| Gaussian NLL difference | `-9.136` |
| Simultaneous interval | `[-13.961, -4.312]` |
| Object/session units improved | `17/22` |
| Marginal 90% coverage | `70.6%` to `91.0%` |
| Mean full interval width | `16.45 mm` to `50.94 mm` |
| Width ratio | `3.10×` |
| Chamfer-distance change | exactly `0` |
| Track-error change | exactly `0` |

The frozen future-study candidate uses the exact `last_residual` mean, the
`independent_endpoint_v1` covariance donor, and early/middle/late covariance
scales `[8, 16, 16]`. This result is retrospective mechanism evidence. It does
not establish independent calibration, fresh-object transfer, deployment
benefit, or permission to retune the donor or scales on a target cohort.

"""

_SUPPORT_MARKER = "## Reporting problems\n"
_SUPPORT_SECTION = """## Release evidence and scientific wording

A release candidate is supported only when its wheel and source distribution
build, install, and exercise the stable interface on Python `3.10`, `3.12`, and
`3.14`, and when the NumPy-only core contracts pass at exact `numpy==1.23.0`.
The release evidence must retain the checked-in resolver input, its
content-addressed `NumericalEnvironmentV1`, and the complete resolved inventory.
See [`docs/releasing.md`](docs/releasing.md).

Packaging and compatibility evidence does not promote a scientific result. Any
release that cites the full-22 PhysTwin improvement must also preserve the
last-residual near tie, the raw-covariance calibration failure, the
width-bearing exact-mean covariance-only result, and the still-open independent
object/session validation boundary from
[`docs/phystwin_release_claim_v1.md`](docs/phystwin_release_claim_v1.md).

"""

_REQUIRED_RELEASE_BULLET = (
    "- the exact-mean covariance-only result is retrospective and increases "
    "mean\n"
    "  full interval width by `3.10×`;\n"
)


@dataclass(frozen=True, slots=True)
class ReleaseSpecification:
    version: str
    release_date: date

    def __post_init__(self) -> None:
        if not _VERSION_PATTERN.fullmatch(self.version):
            raise ValueError("version must have canonical X.Y.Z form")

    @property
    def date_text(self) -> str:
        return self.release_date.isoformat()


@dataclass(frozen=True, slots=True)
class FileUpdate:
    path: Path
    before: str
    after: str

    @property
    def changed(self) -> bool:
        return self.before != self.after


def _replace_once(text: str, old: str, new: str, *, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{name} marker must occur exactly once, found {count}")
    return text.replace(old, new, 1)


def _insert_block_once(
    text: str,
    *,
    identity: str,
    marker: str,
    block: str,
    name: str,
) -> str:
    if identity in text:
        return text
    return _replace_once(text, marker, block + marker, name=name)


def synchronize_changelog(text: str, specification: ReleaseSpecification) -> str:
    release_heading = f"## [{specification.version}] - {specification.date_text}"
    versions = {
        version: recorded_date
        for version, recorded_date in _CHANGELOG_VERSION_PATTERN.findall(text)
    }
    existing = versions.get(specification.version)
    if existing:
        if existing != specification.date_text:
            raise ValueError("changelog release date contradicts requested release")
        return text
    marker = "## [Unreleased]\n"
    return _replace_once(
        text,
        marker,
        marker + "\n" + release_heading + "\n",
        name="CHANGELOG Unreleased",
    )


def synchronize_citation(text: str, specification: ReleaseSpecification) -> str:
    version_pattern = re.compile(r"^version: .+$", re.MULTILINE)
    matches = version_pattern.findall(text)
    if len(matches) != 1:
        raise ValueError("CITATION version must occur exactly once")
    updated = version_pattern.sub(f"version: {specification.version}", text, count=1)
    date_pattern = re.compile(r"^date-released: .+$", re.MULTILINE)
    date_matches = date_pattern.findall(updated)
    if len(date_matches) > 1:
        raise ValueError("CITATION date-released must occur at most once")
    if date_matches:
        updated = date_pattern.sub(
            f"date-released: {specification.date_text}",
            updated,
            count=1,
        )
    else:
        updated = _replace_once(
            updated,
            f"version: {specification.version}\n",
            (
                f"version: {specification.version}\n"
                f"date-released: {specification.date_text}\n"
            ),
            name="CITATION version insertion",
        )
    return updated


def synchronize_readme(text: str, _specification: ReleaseSpecification) -> str:
    if "A retrospective exact-mean covariance-only analysis" in text:
        return text
    return _replace_once(
        text,
        _README_MARKER,
        _README_MARKER + _README_BLOCK,
        name="README covariance evidence",
    )


def synchronize_claim_contract(
    text: str,
    _specification: ReleaseSpecification,
) -> str:
    updated = _insert_block_once(
        text,
        identity="## Exact-mean covariance-only mechanism",
        marker=_CLAIM_MARKER,
        block=_CLAIM_SECTION,
        name="release claim uncertainty section",
    )
    if _REQUIRED_RELEASE_BULLET not in updated:
        bullet_marker = (
            "- raw posterior covariance is severely undercalibrated;\n"
        )
        updated = _replace_once(
            updated,
            bullet_marker,
            bullet_marker + _REQUIRED_RELEASE_BULLET,
            name="release claim required wording",
        )
    return updated


def synchronize_support(text: str, _specification: ReleaseSpecification) -> str:
    return _insert_block_once(
        text,
        identity="## Release evidence and scientific wording",
        marker=_SUPPORT_MARKER,
        block=_SUPPORT_SECTION,
        name="SUPPORT reporting section",
    )


def _project_version(root: Path) -> str:
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    if match is None:
        raise ValueError("pyproject project version is missing")
    return match.group(1)


def planned_updates(
    root: Path,
    specification: ReleaseSpecification,
) -> tuple[FileUpdate, ...]:
    if _project_version(root) != specification.version:
        raise ValueError("requested release differs from pyproject version")
    transforms: Sequence[tuple[str, Callable[[str, ReleaseSpecification], str]]] = (
        ("CHANGELOG.md", synchronize_changelog),
        ("CITATION.cff", synchronize_citation),
        ("README.md", synchronize_readme),
        ("SUPPORT.md", synchronize_support),
        ("docs/phystwin_release_claim_v1.md", synchronize_claim_contract),
    )
    updates: list[FileUpdate] = []
    for relative, transform in transforms:
        path = root / relative
        before = path.read_text(encoding="utf-8")
        after = transform(before, specification)
        updates.append(FileUpdate(path=path, before=before, after=after))
    return tuple(updates)


def synchronize_repository(
    root: Path,
    specification: ReleaseSpecification,
    *,
    check: bool,
) -> tuple[Path, ...]:
    changed = tuple(update for update in planned_updates(root, specification) if update.changed)
    if check and changed:
        rendered = ", ".join(str(update.path.relative_to(root)) for update in changed)
        raise RuntimeError(f"release metadata is not synchronized: {rendered}")
    if not check:
        for update in changed:
            update.path.write_text(update.after, encoding="utf-8")
    return tuple(update.path for update in changed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--date", required=True, dest="release_date")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    specification = ReleaseSpecification(
        version=arguments.version,
        release_date=date.fromisoformat(arguments.release_date),
    )
    root = arguments.root.resolve()
    changed = synchronize_repository(root, specification, check=arguments.check)
    if not arguments.check:
        for path in changed:
            print(path.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
