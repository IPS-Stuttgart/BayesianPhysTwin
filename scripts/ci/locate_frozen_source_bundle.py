#!/usr/bin/env python3
"""Locate an exact historical source bundle without opening experiment data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from bayesian_phystwin.deform360_bias_aware_prospective_physical import (
    UPSTREAM_FILE_SHA256,
)

SCHEMA = "bayesian-phystwin.frozen-source-history-locator"
SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _git_text(
    repository: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=check,
        capture_output=True,
        text=True,
    )


def _git_bytes(
    repository: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=check,
        capture_output=True,
        text=False,
    )


def _canonical_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _canonical_path(value: object) -> str:
    text = _canonical_text(value, name="source-bundle path")
    pure = PurePosixPath(text)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"source-bundle path is not repository relative: {text}")
    if pure.as_posix() != text or "\\" in text or ":" in text:
        raise ValueError(f"source-bundle path is not canonical POSIX: {text}")
    return text


def _canonical_sha256(value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError("source-bundle digests must be lowercase SHA-256 strings")
    return value


def load_requirements(path: Path | None) -> dict[str, str]:
    """Load an exact relative-path to SHA-256 mapping."""

    if path is None:
        raw: object = UPSTREAM_FILE_SHA256
    else:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and "files" in loaded:
            raw = loaded["files"]
        else:
            raw = loaded
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("source-bundle requirements must be a nonempty JSON object")
    requirements: dict[str, str] = {}
    for raw_path, raw_digest in raw.items():
        canonical = _canonical_path(raw_path)
        requirements[canonical] = _canonical_sha256(raw_digest)
    return dict(sorted(requirements.items()))


def _blob(repository: Path, revision: str, relative_path: str) -> bytes | None:
    completed = _git_bytes(
        repository,
        "cat-file",
        "blob",
        f"{revision}:{relative_path}",
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout


def _lines(
    repository: Path,
    *arguments: str,
    check: bool = True,
) -> tuple[str, ...]:
    completed = _git_text(repository, *arguments, check=check)
    if completed.returncode != 0:
        return ()
    return tuple(line for line in completed.stdout.splitlines() if line)


def _candidate_commits(repository: Path, paths: Sequence[str]) -> tuple[str, ...]:
    values = _lines(repository, "rev-list", "--all", "--", *paths)
    return tuple(dict.fromkeys(values))


def _commit_record(repository: Path, revision: str) -> dict[str, object]:
    details = _lines(
        repository,
        "show",
        "-s",
        "--format=%H%n%cI%n%s",
        revision,
    )
    if len(details) < 3:
        raise RuntimeError(f"cannot describe candidate commit {revision}")
    refs = _lines(
        repository,
        "for-each-ref",
        "--format=%(refname)",
        f"--points-at={revision}",
        "refs/heads",
        "refs/remotes",
        "refs/tags",
    )
    containing_tags = _lines(repository, "tag", "--contains", revision)
    return {
        "revision": details[0],
        "committed_at": details[1],
        "subject": details[2],
        "refs_pointing_at": sorted(refs),
        "containing_tags": sorted(containing_tags),
    }


def _report_id(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def locate_frozen_source_bundle(
    repository: Path,
    requirements: Mapping[str, str],
    *,
    repository_id: str = "local-git-repository",
) -> dict[str, object]:
    """Search every fetched ref for a commit matching all required bytes."""

    root = repository.resolve()
    inside = _lines(root, "rev-parse", "--is-inside-work-tree", check=False)
    if inside != ("true",):
        raise ValueError("repository_root must be a Git working tree")
    shallow = _lines(root, "rev-parse", "--is-shallow-repository")
    if shallow != ("false",):
        raise ValueError("complete Git history is required")
    stable_repository_id = _canonical_text(repository_id, name="repository_id")
    normalized = {
        _canonical_path(path): _canonical_sha256(digest)
        for path, digest in requirements.items()
    }
    normalized = dict(sorted(normalized.items()))
    if not normalized:
        raise ValueError("source-bundle requirements must not be empty")
    anchor = (
        "scripts/remote/run_deform360_official_phystwin_smoke.py"
        if "scripts/remote/run_deform360_official_phystwin_smoke.py" in normalized
        else next(iter(normalized))
    )
    candidates = _candidate_commits(root, tuple(normalized))
    anchor_matches = 0
    exact_matches: list[dict[str, object]] = []
    for revision in candidates:
        anchor_blob = _blob(root, revision, anchor)
        if anchor_blob is None:
            continue
        if hashlib.sha256(anchor_blob).hexdigest() != normalized[anchor]:
            continue
        anchor_matches += 1
        observed: dict[str, str] = {anchor: normalized[anchor]}
        matches = True
        for relative_path, expected in normalized.items():
            if relative_path == anchor:
                continue
            blob = _blob(root, revision, relative_path)
            if blob is None:
                matches = False
                break
            digest = hashlib.sha256(blob).hexdigest()
            if digest != expected:
                matches = False
                break
            observed[relative_path] = digest
        if matches:
            exact_matches.append(
                {
                    **_commit_record(root, revision),
                    "observed_file_sha256": dict(sorted(observed.items())),
                }
            )
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "repository_id": stable_repository_id,
        "complete_history_searched": True,
        "required_file_sha256": normalized,
        "anchor_path": anchor,
        "candidate_commit_count": len(candidates),
        "anchor_match_count": anchor_matches,
        "exact_match_count": len(exact_matches),
        "exact_matches": exact_matches,
        "information_boundary": {
            "dataset_opened": False,
            "source_residual_opened": False,
            "development_suffix_opened": False,
            "target_payload_opened": False,
            "target_outcome_used": False,
        },
    }
    payload["report_id"] = _report_id(payload)
    return payload


def write_report(path: Path, report: Mapping[str, object]) -> None:
    """Publish atomically without following links or overwriting evidence."""

    destination = path.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or destination.exists():
        raise FileExistsError(f"refusing to overwrite locator report: {path}")
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--repository-id",
        default="IPS-Stuttgart/BayesianPhysTwin",
    )
    parser.add_argument("--requirements-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-match", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    requirements = load_requirements(args.requirements_json)
    report = locate_frozen_source_bundle(
        args.repository_root,
        requirements,
        repository_id=args.repository_id,
    )
    write_report(args.output, report)
    print(
        json.dumps(
            {
                "report_id": report["report_id"],
                "candidate_commit_count": report["candidate_commit_count"],
                "anchor_match_count": report["anchor_match_count"],
                "exact_match_count": report["exact_match_count"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return int(bool(args.require_match and report["exact_match_count"] == 0))


if __name__ == "__main__":
    sys.exit(main())
