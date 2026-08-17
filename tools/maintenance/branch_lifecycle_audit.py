#!/usr/bin/env python3
"""Inventory repository branches without deleting or rewriting any Git ref."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "bayesian-phystwin.branch-lifecycle-inventory"
SCHEMA_VERSION = 1
DEFAULT_STALE_DAYS = 60
DEFAULT_MAX_REFERENCE_BYTES = 5_000_000
DEFAULT_REFERENCE_GLOBS = (
    "README.md",
    "CHANGELOG.md",
    "STATUS.md",
    "CITATION.cff",
    "claims.json",
    "docs/**/*.md",
    "docs/**/*.json",
    "protocols/**/*.json",
    "protocols/**/*.md",
    "results/**/*.json",
    "results/**/*.md",
    "evidence/**/*",
    "configs/**/*.json",
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
)
_SHA40 = re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])")
_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")


class AuditError(ValueError):
    """Raised when branch inventory inputs are incomplete or inconsistent."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _parse_datetime(value: object, *, name: str) -> datetime:
    if type(value) is not str or not value:
        raise AuditError(f"{name} must be a nonempty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise AuditError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise AuditError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise AuditError("timestamps must include a timezone")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise AuditError(f"{name} must be nonempty canonical text")
    return value


def _require_sha(value: object, *, name: str) -> str:
    text = _require_text(value, name=name).lower()
    if len(text) != 40 or any(ch not in "0123456789abcdef" for ch in text):
        raise AuditError(f"{name} must be a 40-character Git commit SHA")
    return text


def _require_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise AuditError(f"{name} must be boolean")
    return value


def _repository_name(value: str) -> str:
    if _REPOSITORY.fullmatch(value) is None:
        raise AuditError("repository must use canonical owner/name form")
    return value


@dataclass(frozen=True)
class RemoteBranch:
    """One branch returned by GitHub's repository branches endpoint."""

    name: str
    revision: str
    protected: bool

    @classmethod
    def from_api(cls, payload: object, *, index: int) -> RemoteBranch:
        if not isinstance(payload, Mapping):
            raise AuditError(f"branches[{index}] must be a JSON object")
        commit = payload.get("commit")
        if not isinstance(commit, Mapping):
            raise AuditError(f"branches[{index}].commit must be a JSON object")
        return cls(
            name=_require_text(payload.get("name"), name=f"branches[{index}].name"),
            revision=_require_sha(
                commit.get("sha"), name=f"branches[{index}].commit.sha"
            ),
            protected=_require_bool(
                payload.get("protected"), name=f"branches[{index}].protected"
            ),
        )


@dataclass(frozen=True)
class PullRequest:
    """Branch-associated pull-request metadata used for retention decisions."""

    number: int
    state: str
    draft: bool
    merged_at: str | None
    updated_at: str
    html_url: str
    head_revision: str

    @classmethod
    def from_api(
        cls,
        payload: object,
        *,
        repository: str,
        index: int,
    ) -> tuple[str, PullRequest] | None:
        if not isinstance(payload, Mapping):
            raise AuditError(f"pulls[{index}] must be a JSON object")
        head = payload.get("head")
        if not isinstance(head, Mapping):
            raise AuditError(f"pulls[{index}].head must be a JSON object")
        head_repo = head.get("repo")
        if head_repo is None:
            return None
        if not isinstance(head_repo, Mapping):
            raise AuditError(f"pulls[{index}].head.repo must be a JSON object")
        if head_repo.get("full_name") != repository:
            return None
        number = payload.get("number")
        if type(number) is not int or number < 1:
            raise AuditError(f"pulls[{index}].number must be a positive integer")
        state = _require_text(payload.get("state"), name=f"pulls[{index}].state")
        if state not in {"open", "closed"}:
            raise AuditError(f"pulls[{index}].state is unsupported")
        merged_at = payload.get("merged_at")
        if merged_at is not None:
            _parse_datetime(merged_at, name=f"pulls[{index}].merged_at")
        updated_at = _require_text(
            payload.get("updated_at"), name=f"pulls[{index}].updated_at"
        )
        _parse_datetime(updated_at, name=f"pulls[{index}].updated_at")
        branch = _require_text(head.get("ref"), name=f"pulls[{index}].head.ref")
        return branch, cls(
            number=number,
            state=state,
            draft=_require_bool(payload.get("draft"), name=f"pulls[{index}].draft"),
            merged_at=merged_at,
            updated_at=updated_at,
            html_url=_require_text(
                payload.get("html_url"), name=f"pulls[{index}].html_url"
            ),
            head_revision=_require_sha(
                head.get("sha"), name=f"pulls[{index}].head.sha"
            ),
        )

    @property
    def is_open(self) -> bool:
        return self.state == "open"

    def as_dict(self) -> dict[str, object]:
        return {
            "number": self.number,
            "state": self.state,
            "draft": self.draft,
            "merged_at": self.merged_at,
            "updated_at": self.updated_at,
            "html_url": self.html_url,
            "head_revision": self.head_revision,
        }


@dataclass(frozen=True)
class ReferenceLocation:
    """Tracked evidence/document location that names an exact branch head."""

    path: str
    lines: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {"path": self.path, "lines": list(self.lines)}


@dataclass(frozen=True)
class BranchFacts:
    """Inputs used by the pure branch-lifecycle decision function."""

    name: str
    revision: str
    commit_time: datetime
    activity_time: datetime
    protected: bool
    reachable_from_default: bool
    tags: tuple[str, ...] = ()
    references: tuple[ReferenceLocation, ...] = ()
    pull_requests: tuple[PullRequest, ...] = ()


@dataclass(frozen=True)
class BranchDecision:
    """Non-destructive classification and recommended next action."""

    classification: str
    deletion_candidate: bool
    requires_tag_before_deletion: bool
    recommended_action: str
    reasons: tuple[str, ...]
    preservation_tag_suggestion: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "classification": self.classification,
            "deletion_candidate": self.deletion_candidate,
            "requires_tag_before_deletion": self.requires_tag_before_deletion,
            "recommended_action": self.recommended_action,
            "reasons": list(self.reasons),
            "preservation_tag_suggestion": self.preservation_tag_suggestion,
        }


def _tag_suggestion(branch: str, revision: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", branch).strip("-.") or "branch"
    slug = slug[:72].rstrip("-.")
    return f"archive/branch-head/{slug}-{revision[:12]}"


def classify_branch(
    facts: BranchFacts,
    *,
    default_branch: str,
    generated_at: datetime,
    stale_days: int,
) -> BranchDecision:
    """Classify one branch without ever authorizing automatic deletion."""

    if stale_days < 1:
        raise AuditError("stale_days must be positive")
    if generated_at.tzinfo is None or facts.activity_time.tzinfo is None:
        raise AuditError("branch classification timestamps must include a timezone")
    age_days = max(
        0,
        (
            generated_at.astimezone(timezone.utc)
            - facts.activity_time.astimezone(timezone.utc)
        ).days,
    )
    open_prs = tuple(pr for pr in facts.pull_requests if pr.is_open)
    if facts.name == default_branch:
        return BranchDecision(
            "retain-default", False, False, "retain", ("repository-default-branch",)
        )
    if facts.protected:
        return BranchDecision(
            "retain-protected", False, False, "retain", ("github-protected-branch",)
        )
    if open_prs:
        return BranchDecision(
            "retain-open-pr",
            False,
            False,
            "retain-until-pull-request-resolves",
            ("open-pull-request",),
        )
    if age_days < stale_days:
        return BranchDecision(
            "retain-recent",
            False,
            False,
            "retain-until-stale-threshold",
            ("recent-branch-activity",),
        )
    if facts.reachable_from_default:
        return BranchDecision(
            "deletion-candidate-merged",
            True,
            False,
            "review-then-delete-merged-branch",
            ("exact-head-reachable-from-default",),
        )
    if facts.tags:
        return BranchDecision(
            "deletion-candidate-tagged",
            True,
            False,
            "review-then-delete-tag-preserved-branch",
            ("exact-head-preserved-by-tag",),
        )
    suggestion = _tag_suggestion(facts.name, facts.revision)
    if facts.references:
        return BranchDecision(
            "retain-evidence-needs-tag",
            False,
            True,
            "create-immutable-tag-before-deletion-review",
            ("exact-head-referenced-by-tracked-evidence",),
            suggestion,
        )
    if facts.name.startswith("archive/"):
        return BranchDecision(
            "retain-archive-needs-tag",
            False,
            True,
            "create-immutable-tag-before-archive-branch-deletion",
            ("archive-branch-is-only-known-head-reference",),
            suggestion,
        )
    if facts.pull_requests:
        return BranchDecision(
            "review-closed-pr-unpreserved",
            False,
            True,
            "inspect-closed-pr-and-tag-before-deletion",
            ("closed-pull-request-head-not-reachable-or-tagged",),
            suggestion,
        )
    return BranchDecision(
        "review-unreferenced-unmerged",
        False,
        False,
        "manually-prove-unreferenced-before-deletion",
        ("unmerged-head-has-no-durable-reference-detected",),
    )


class GitHubClient:
    """Small read-only GitHub REST client with deterministic pagination."""

    def __init__(
        self,
        *,
        token: str | None,
        api_url: str = "https://api.github.com",
    ) -> None:
        self.token = token.strip() if token else None
        self.api_url = api_url.rstrip("/")

    def _get(self, path: str) -> object:
        request = urllib.request.Request(
            f"{self.api_url}{path}",
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "bayesian-phystwin-branch-lifecycle-audit/1",
                **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
        except (urllib.error.HTTPError, urllib.error.URLError) as error:
            raise AuditError(f"GitHub request failed for {path}: {error}") from error
        try:
            return json.loads(
                data.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AuditError(f"GitHub returned invalid JSON for {path}") from error

    def paginated_list(self, path: str, *, per_page: int = 100) -> list[object]:
        if not 1 <= per_page <= 100:
            raise AuditError("per_page must lie in [1, 100]")
        separator = "&" if "?" in path else "?"
        result: list[object] = []
        page = 1
        while True:
            payload = self._get(f"{path}{separator}per_page={per_page}&page={page}")
            if not isinstance(payload, list):
                raise AuditError(f"GitHub paginated response is not a list: {path}")
            result.extend(payload)
            if len(payload) < per_page:
                return result
            page += 1

    def branches(self, repository: str) -> tuple[RemoteBranch, ...]:
        payload = self.paginated_list(f"/repos/{repository}/branches")
        branches = tuple(
            RemoteBranch.from_api(item, index=index)
            for index, item in enumerate(payload)
        )
        names = [branch.name for branch in branches]
        if len(names) != len(set(names)):
            raise AuditError("GitHub returned duplicate branch names")
        return tuple(sorted(branches, key=lambda branch: branch.name))

    def pull_requests(self, repository: str) -> dict[str, tuple[PullRequest, ...]]:
        payload = self.paginated_list(
            f"/repos/{repository}/pulls?state=all&sort=updated&direction=desc"
        )
        grouped: defaultdict[str, list[PullRequest]] = defaultdict(list)
        for index, item in enumerate(payload):
            parsed = PullRequest.from_api(
                item,
                repository=repository,
                index=index,
            )
            if parsed is not None:
                branch, pull_request = parsed
                grouped[branch].append(pull_request)
        return {
            branch: tuple(sorted(items, key=lambda item: item.number))
            for branch, items in sorted(grouped.items())
        }

    def tags(self, repository: str) -> dict[str, tuple[str, ...]]:
        payload = self.paginated_list(f"/repos/{repository}/tags")
        grouped: defaultdict[str, set[str]] = defaultdict(set)
        for index, item in enumerate(payload):
            if not isinstance(item, Mapping):
                raise AuditError(f"tags[{index}] must be a JSON object")
            commit = item.get("commit")
            if not isinstance(commit, Mapping):
                raise AuditError(f"tags[{index}].commit must be a JSON object")
            grouped[
                _require_sha(commit.get("sha"), name=f"tags[{index}].commit.sha")
            ].add(_require_text(item.get("name"), name=f"tags[{index}].name"))
        return {
            revision: tuple(sorted(names))
            for revision, names in sorted(grouped.items())
        }


class GitInspector:
    """Read-only local Git operations used for exact ancestry checks."""

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root.resolve(strict=True)
        top_level = self._run("rev-parse", "--show-toplevel")
        if Path(top_level).resolve() != self.repository_root:
            raise AuditError("repository-root must be the Git checkout top level")

    def _run(
        self,
        *arguments: str,
        acceptable: tuple[int, ...] = (0,),
    ) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=self.repository_root,
            text=True,
            capture_output=True,
            timeout=30,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        if completed.returncode not in acceptable:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise AuditError(f"git {' '.join(arguments)} failed: {detail}")
        return completed.stdout.strip()

    def ensure_commit(self, revision: str) -> None:
        self._run("cat-file", "-e", f"{revision}^{{commit}}")

    def branch_revision(self, branch: str) -> str:
        for reference in (f"refs/remotes/origin/{branch}", f"refs/heads/{branch}"):
            completed = subprocess.run(
                ["git", "rev-parse", "--verify", f"{reference}^{{commit}}"],
                cwd=self.repository_root,
                text=True,
                capture_output=True,
                timeout=30,
            )
            if completed.returncode == 0:
                return _require_sha(completed.stdout.strip(), name=reference)
        raise AuditError(f"cannot resolve local branch revision: {branch}")

    def commit_time(self, revision: str) -> datetime:
        self.ensure_commit(revision)
        return _parse_datetime(
            self._run("show", "-s", "--format=%cI", revision),
            name=f"commit time for {revision}",
        )

    def is_ancestor(self, revision: str, default_revision: str) -> bool:
        self.ensure_commit(revision)
        self.ensure_commit(default_revision)
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", revision, default_revision],
            cwd=self.repository_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        if completed.returncode == 0:
            return True
        if completed.returncode == 1:
            return False
        raise AuditError("git ancestry check failed: " + completed.stderr.strip())

    def tracked_paths(self) -> frozenset[str]:
        output = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=self.repository_root,
            capture_output=True,
            timeout=30,
            check=True,
        ).stdout
        try:
            decoded = output.decode("utf-8")
        except UnicodeDecodeError as error:
            raise AuditError("tracked paths must be UTF-8") from error
        return frozenset(path for path in decoded.split("\0") if path)


def _canonical_glob(value: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise AuditError("reference globs must be nonempty canonical text")
    if "\\" in value:
        raise AuditError("reference globs must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise AuditError("reference globs must stay inside the repository")
    return value


def scan_exact_head_references(
    repository_root: Path,
    revisions: Iterable[str],
    *,
    globs: Sequence[str] = DEFAULT_REFERENCE_GLOBS,
    max_bytes: int = DEFAULT_MAX_REFERENCE_BYTES,
    tracked_paths: frozenset[str] | None = None,
) -> dict[str, tuple[ReferenceLocation, ...]]:
    """Find exact candidate head SHAs in tracked evidence-oriented text files."""

    root = repository_root.resolve(strict=True)
    if max_bytes < 1:
        raise AuditError("max_bytes must be positive")
    candidates = frozenset(
        _require_sha(revision, name="candidate revision") for revision in revisions
    )
    if not candidates:
        return {}
    selected: set[Path] = set()
    for pattern in globs:
        canonical = _canonical_glob(pattern)
        for path in root.glob(canonical):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError as error:
                raise AuditError("reference glob escaped repository") from error
            if tracked_paths is not None and relative not in tracked_paths:
                continue
            selected.add(path)

    found: defaultdict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for path in sorted(selected):
        if path.stat().st_size > max_bytes:
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(root).as_posix()
        for line_number, line in enumerate(text.splitlines(), start=1):
            for revision in _SHA40.findall(line.lower()):
                if revision in candidates:
                    found[revision][relative].add(line_number)
    return {
        revision: tuple(
            ReferenceLocation(path, tuple(sorted(lines)))
            for path, lines in sorted(paths.items())
        )
        for revision, paths in sorted(found.items())
    }


def build_inventory(
    *,
    repository: str,
    default_branch: str,
    branches: Sequence[RemoteBranch],
    pull_requests_by_branch: Mapping[str, Sequence[PullRequest]],
    tags_by_revision: Mapping[str, Sequence[str]],
    references_by_revision: Mapping[str, Sequence[ReferenceLocation]],
    inspector: GitInspector,
    generated_at: datetime,
    stale_days: int,
    reference_globs: Sequence[str],
) -> dict[str, object]:
    """Build a deterministic, content-addressed lifecycle inventory."""

    repository = _repository_name(repository)
    default_branch = _require_text(default_branch, name="default_branch")
    if generated_at.tzinfo is None:
        raise AuditError("generated_at must include a timezone")
    if stale_days < 1:
        raise AuditError("stale_days must be positive")
    generated = generated_at.astimezone(timezone.utc)
    ordered_branches = tuple(sorted(branches, key=lambda item: item.name))
    branch_names = [branch.name for branch in ordered_branches]
    if len(branch_names) != len(set(branch_names)):
        raise AuditError("branch inventory contains duplicate names")
    default_matches = [
        branch for branch in ordered_branches if branch.name == default_branch
    ]
    if len(default_matches) != 1:
        raise AuditError("default branch must occur exactly once")
    default_revision = default_matches[0].revision
    inspector.ensure_commit(default_revision)

    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for branch in ordered_branches:
        inspector.ensure_commit(branch.revision)
        commit_time = inspector.commit_time(branch.revision)
        pull_requests = tuple(
            sorted(
                pull_requests_by_branch.get(branch.name, ()),
                key=lambda item: item.number,
            )
        )
        numbers = [item.number for item in pull_requests]
        if len(numbers) != len(set(numbers)):
            raise AuditError(f"duplicate pull request for branch {branch.name}")
        pr_times = tuple(
            _parse_datetime(item.updated_at, name=f"PR #{item.number} updated_at")
            for item in pull_requests
        )
        activity_time = max((commit_time, *pr_times))
        tags = tuple(sorted(set(tags_by_revision.get(branch.revision, ()))))
        references = tuple(
            sorted(
                references_by_revision.get(branch.revision, ()),
                key=lambda item: item.path,
            )
        )
        facts = BranchFacts(
            name=branch.name,
            revision=branch.revision,
            commit_time=commit_time,
            activity_time=activity_time,
            protected=branch.protected,
            reachable_from_default=inspector.is_ancestor(
                branch.revision,
                default_revision,
            ),
            tags=tags,
            references=references,
            pull_requests=pull_requests,
        )
        decision = classify_branch(
            facts,
            default_branch=default_branch,
            generated_at=generated,
            stale_days=stale_days,
        )
        counts[decision.classification] += 1
        rows.append(
            {
                "branch": branch.name,
                "revision": branch.revision,
                "commit_time": _timestamp(commit_time),
                "activity_time": _timestamp(activity_time),
                "activity_age_days": max(0, (generated - activity_time).days),
                "protected": branch.protected,
                "reachable_from_default": facts.reachable_from_default,
                "tags": list(tags),
                "references": [item.as_dict() for item in references],
                "pull_requests": [item.as_dict() for item in pull_requests],
                **decision.as_dict(),
            }
        )

    summary = {
        "branch_count": len(rows),
        "non_default_branch_count": sum(
            row["branch"] != default_branch for row in rows
        ),
        "deletion_candidate_count": sum(
            bool(row["deletion_candidate"]) for row in rows
        ),
        "tag_required_count": sum(
            bool(row["requires_tag_before_deletion"]) for row in rows
        ),
        "manual_review_count": sum(
            str(row["classification"]).startswith("review-") for row in rows
        ),
        "classification_counts": dict(sorted(counts.items())),
    }
    descriptor: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "repository": repository,
        "default_branch": default_branch,
        "default_revision": default_revision,
        "stale_days": stale_days,
        "reference_globs": sorted(set(map(_canonical_glob, reference_globs))),
        "summary": summary,
        "branches": rows,
    }
    return {
        "inventory_id": hashlib.sha256(_canonical_json(descriptor)).hexdigest(),
        "generated_utc": _timestamp(generated),
        **descriptor,
    }


def render_markdown(inventory: Mapping[str, object]) -> str:
    """Render a compact operator-facing report from one inventory."""

    summary = inventory.get("summary")
    branches = inventory.get("branches")
    if not isinstance(summary, Mapping) or not isinstance(branches, list):
        raise AuditError("inventory is missing summary or branch rows")
    lines = [
        "# Branch lifecycle audit",
        "",
        f"- Repository: `{inventory['repository']}`",
        f"- Default: `{inventory['default_branch']}` at "
        f"`{str(inventory['default_revision'])[:12]}`",
        f"- Inventory: `{inventory['inventory_id']}`",
        f"- Generated: `{inventory['generated_utc']}`",
        f"- Stale threshold: `{inventory['stale_days']}` days",
        f"- Branches: `{summary['branch_count']}`; deletion candidates: "
        f"`{summary['deletion_candidate_count']}`; tag required: "
        f"`{summary['tag_required_count']}`; manual review: "
        f"`{summary['manual_review_count']}`",
        "",
        "This audit is read-only. A deletion candidate is a review queue entry, "
        "not an authorization to delete a branch automatically.",
        "",
    ]
    sections = (
        (
            "Deletion candidates already preserved",
            lambda row: bool(row.get("deletion_candidate")),
        ),
        (
            "Branches that need an immutable tag first",
            lambda row: bool(row.get("requires_tag_before_deletion")),
        ),
        (
            "Manual review for unmerged history",
            lambda row: str(row.get("classification", "")).startswith("review-"),
        ),
    )
    for title, predicate in sections:
        selected = [
            row for row in branches if isinstance(row, Mapping) and predicate(row)
        ]
        lines.extend((f"## {title}", ""))
        if not selected:
            lines.extend(("None.", ""))
            continue
        lines.extend(
            (
                "| Branch | Head | Age | Classification | Suggested action |",
                "| --- | --- | ---: | --- | --- |",
            )
        )
        for row in selected:
            action = str(row.get("recommended_action", ""))
            tag = row.get("preservation_tag_suggestion")
            if tag:
                action += f" (`{tag}`)"
            lines.append(
                f"| `{row['branch']}` | `{str(row['revision'])[:12]}` | "
                f"{row['activity_age_days']} d | `{row['classification']}` | "
                f"{action} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _atomic_write_text(path: Path, content: str) -> None:
    if path.is_symlink():
        raise AuditError(f"refusing to replace symlink output: {path}")
    destination = path.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_inventory(
    inventory: Mapping[str, object],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """Publish JSON and Markdown reports atomically."""

    if json_path.resolve() == markdown_path.resolve():
        raise AuditError("JSON and Markdown outputs must use different paths")
    json_text = (
        json.dumps(dict(inventory), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _atomic_write_text(json_path, json_text)
    _atomic_write_text(markdown_path, render_markdown(inventory))


def audit_repository(
    *,
    repository_root: Path,
    repository: str,
    default_branch: str,
    generated_at: datetime,
    stale_days: int,
    reference_globs: Sequence[str],
    max_reference_bytes: int,
    github_client: GitHubClient,
) -> dict[str, object]:
    """Collect GitHub and local-Git state and build one inventory."""

    inspector = GitInspector(repository_root)
    branches = github_client.branches(repository)
    return build_inventory(
        repository=repository,
        default_branch=default_branch,
        branches=branches,
        pull_requests_by_branch=github_client.pull_requests(repository),
        tags_by_revision=github_client.tags(repository),
        references_by_revision=scan_exact_head_references(
            inspector.repository_root,
            (branch.revision for branch in branches),
            globs=reference_globs,
            max_bytes=max_reference_bytes,
            tracked_paths=inspector.tracked_paths(),
        ),
        inspector=inspector,
        generated_at=generated_at,
        stale_days=stale_days,
        reference_globs=reference_globs,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="Top level of a complete Git checkout.",
    )
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="GitHub repository in owner/name form.",
    )
    parser.add_argument(
        "--default-branch",
        default="main",
        help="Default branch used for reachability checks.",
    )
    parser.add_argument(
        "--stale-days",
        type=int,
        default=DEFAULT_STALE_DAYS,
        help="Minimum inactivity age for cleanup review.",
    )
    parser.add_argument(
        "--reference-glob",
        action="append",
        dest="reference_globs",
        help="Tracked evidence/document glob; repeat to replace defaults.",
    )
    parser.add_argument(
        "--max-reference-bytes",
        type=int,
        default=DEFAULT_MAX_REFERENCE_BYTES,
        help="Maximum tracked text-file size scanned for exact head SHAs.",
    )
    parser.add_argument(
        "--generated-utc",
        help="Optional fixed ISO-8601 report time for deterministic replay.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        generated_at = (
            _parse_datetime(args.generated_utc, name="generated_utc")
            if args.generated_utc
            else datetime.now(timezone.utc)
        )
        inventory = audit_repository(
            repository_root=args.repository_root,
            repository=_repository_name(args.repository),
            default_branch=args.default_branch,
            generated_at=generated_at,
            stale_days=args.stale_days,
            reference_globs=tuple(args.reference_globs or DEFAULT_REFERENCE_GLOBS),
            max_reference_bytes=args.max_reference_bytes,
            github_client=GitHubClient(
                token=os.environ.get("GITHUB_TOKEN"),
                api_url=os.environ.get("GITHUB_API_URL", "https://api.github.com"),
            ),
        )
        write_inventory(
            inventory,
            json_path=args.output_json,
            markdown_path=args.output_markdown,
        )
        summary = inventory["summary"]
        assert isinstance(summary, Mapping)
        print(
            json.dumps(
                {
                    "inventory_id": inventory["inventory_id"],
                    "branch_count": summary["branch_count"],
                    "deletion_candidate_count": summary["deletion_candidate_count"],
                    "tag_required_count": summary["tag_required_count"],
                    "manual_review_count": summary["manual_review_count"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    except (AuditError, OSError, subprocess.SubprocessError) as error:
        print(f"branch lifecycle audit error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
