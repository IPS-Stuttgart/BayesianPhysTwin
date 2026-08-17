#!/usr/bin/env python3
"""Verify that built distributions expose the installed typing contract."""

from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_MARKER_SUFFIX = PurePosixPath("bayesian_phystwin/py.typed")
_TYPED_CLASSIFIER = "Classifier: Typing :: Typed"


@dataclass(frozen=True)
class DistributionTypingEvidence:
    """Validated PEP 561 evidence extracted from one distribution archive."""

    archive: str
    marker_member: str
    metadata_members: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "archive": self.archive,
            "marker_member": self.marker_member,
            "metadata_members": list(self.metadata_members),
        }


def _normalized_members(names: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            name.replace("\\", "/").lstrip("./")
            for name in names
            if name and not name.endswith("/")
        )
    )


def _is_marker_member(name: str) -> bool:
    path = PurePosixPath(name)
    suffix_parts = _MARKER_SUFFIX.parts
    return (
        len(path.parts) >= len(suffix_parts)
        and path.parts[-len(suffix_parts) :] == suffix_parts
    )


def _metadata_candidates(names: Iterable[str]) -> tuple[str, ...]:
    result = []
    for name in names:
        path = PurePosixPath(name)
        if path.name == "METADATA" and path.parent.name.endswith(".dist-info"):
            result.append(name)
        elif path.name == "PKG-INFO":
            result.append(name)
    return tuple(sorted(result))


def _validate_members_and_metadata(
    *,
    archive: Path,
    members: tuple[str, ...],
    read_member: Callable[[str], bytes],
) -> DistributionTypingEvidence:
    marker_members = tuple(name for name in members if _is_marker_member(name))
    if len(marker_members) != 1:
        raise ValueError(
            f"{archive}: expected exactly one bayesian_phystwin/py.typed member, "
            f"found {len(marker_members)}"
        )

    metadata_members = _metadata_candidates(members)
    typed_metadata = tuple(
        name
        for name in metadata_members
        if _TYPED_CLASSIFIER in read_member(name).decode("utf-8", errors="strict")
    )
    if not typed_metadata:
        raise ValueError(
            f"{archive}: distribution metadata does not declare {_TYPED_CLASSIFIER!r}"
        )

    return DistributionTypingEvidence(
        archive=str(archive),
        marker_member=marker_members[0],
        metadata_members=typed_metadata,
    )


def verify_distribution(path: str | Path) -> DistributionTypingEvidence:
    """Validate the marker and classifier in one wheel or source distribution."""

    archive = Path(path)
    if not archive.is_file():
        raise ValueError(f"distribution archive does not exist: {archive}")

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            members = _normalized_members(handle.namelist())
            return _validate_members_and_metadata(
                archive=archive,
                members=members,
                read_member=handle.read,
            )

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, mode="r:*") as handle:
            members = _normalized_members(member.name for member in handle.getmembers())

            def read_member(name: str) -> bytes:
                extracted = handle.extractfile(name)
                if extracted is None:
                    raise ValueError(f"{archive}: cannot read archive member {name}")
                return extracted.read()

            return _validate_members_and_metadata(
                archive=archive,
                members=members,
                read_member=read_member,
            )

    raise ValueError(f"unsupported distribution archive: {archive}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archives",
        nargs="+",
        type=Path,
        help="wheel and/or source-distribution archives to verify",
    )
    return parser


def main() -> int:
    evidence = [
        verify_distribution(path).as_dict() for path in _parser().parse_args().archives
    ]
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
