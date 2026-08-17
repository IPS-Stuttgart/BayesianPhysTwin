from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "maintenance"
    / "branch_lifecycle_audit.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "bayesian_phystwin_branch_lifecycle_audit",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

AuditError = _MODULE.AuditError
BranchFacts = _MODULE.BranchFacts
GitHubClient = _MODULE.GitHubClient
GitInspector = _MODULE.GitInspector
PullRequest = _MODULE.PullRequest
ReferenceLocation = _MODULE.ReferenceLocation
RemoteBranch = _MODULE.RemoteBranch
build_inventory = _MODULE.build_inventory
classify_branch = _MODULE.classify_branch
render_markdown = _MODULE.render_markdown
scan_exact_head_references = _MODULE.scan_exact_head_references
write_inventory = _MODULE.write_inventory

_NOW = datetime(2026, 8, 11, 8, 0, tzinfo=timezone.utc)
_MAIN_SHA = "1" * 40
_MERGED_SHA = "2" * 40
_UNMERGED_SHA = "3" * 40
_TAGGED_SHA = "4" * 40


def _pull_request(
    number: int,
    *,
    state: str = "closed",
    updated_at: datetime | None = None,
    merged: bool = False,
    head_revision: str = _UNMERGED_SHA,
) -> PullRequest:
    updated = updated_at or (_NOW - timedelta(days=100))
    return PullRequest(
        number=number,
        state=state,
        draft=False,
        merged_at=(updated.isoformat() if merged else None),
        updated_at=updated.isoformat(),
        html_url=f"https://github.com/example/project/pull/{number}",
        head_revision=head_revision,
    )


def _facts(
    *,
    name: str = "agent/example",
    revision: str = _UNMERGED_SHA,
    age_days: int = 100,
    protected: bool = False,
    reachable: bool = False,
    tags: tuple[str, ...] = (),
    references: tuple[ReferenceLocation, ...] = (),
    pull_requests: tuple[PullRequest, ...] = (),
) -> BranchFacts:
    activity = _NOW - timedelta(days=age_days)
    return BranchFacts(
        name=name,
        revision=revision,
        commit_time=activity,
        activity_time=activity,
        protected=protected,
        reachable_from_default=reachable,
        tags=tags,
        references=references,
        pull_requests=pull_requests,
    )


@pytest.mark.parametrize(
    ("facts", "classification"),
    (
        (_facts(name="main"), "retain-default"),
        (_facts(protected=True), "retain-protected"),
        (
            _facts(pull_requests=(_pull_request(7, state="open"),)),
            "retain-open-pr",
        ),
        (_facts(age_days=10), "retain-recent"),
    ),
)
def test_active_branches_are_never_cleanup_candidates(
    facts: BranchFacts,
    classification: str,
) -> None:
    decision = classify_branch(
        facts,
        default_branch="main",
        generated_at=_NOW,
        stale_days=60,
    )

    assert decision.classification == classification
    assert not decision.deletion_candidate
    assert not decision.requires_tag_before_deletion


def test_only_already_preserved_stale_heads_enter_deletion_queue() -> None:
    merged = classify_branch(
        _facts(reachable=True, revision=_MERGED_SHA),
        default_branch="main",
        generated_at=_NOW,
        stale_days=60,
    )
    tagged = classify_branch(
        _facts(tags=("evidence/example-v1",), revision=_TAGGED_SHA),
        default_branch="main",
        generated_at=_NOW,
        stale_days=60,
    )

    assert merged.classification == "deletion-candidate-merged"
    assert merged.deletion_candidate
    assert tagged.classification == "deletion-candidate-tagged"
    assert tagged.deletion_candidate
    assert "exact-head-preserved-by-tag" in tagged.reasons


def test_referenced_unmerged_head_requires_an_immutable_tag() -> None:
    decision = classify_branch(
        _facts(
            references=(ReferenceLocation("protocols/example.json", (12,)),),
        ),
        default_branch="main",
        generated_at=_NOW,
        stale_days=60,
    )

    assert decision.classification == "retain-evidence-needs-tag"
    assert decision.requires_tag_before_deletion
    assert not decision.deletion_candidate
    assert decision.preservation_tag_suggestion is not None
    assert decision.preservation_tag_suggestion.endswith(_UNMERGED_SHA[:12])


def test_archive_and_closed_pr_heads_fail_closed_without_preservation() -> None:
    archive = classify_branch(
        _facts(name="archive/pr123-pre-rebase"),
        default_branch="main",
        generated_at=_NOW,
        stale_days=60,
    )
    closed = classify_branch(
        _facts(pull_requests=(_pull_request(123),)),
        default_branch="main",
        generated_at=_NOW,
        stale_days=60,
    )

    assert archive.classification == "retain-archive-needs-tag"
    assert archive.requires_tag_before_deletion
    assert closed.classification == "review-closed-pr-unpreserved"
    assert closed.requires_tag_before_deletion


def test_unreferenced_unmerged_head_requires_manual_proof() -> None:
    decision = classify_branch(
        _facts(),
        default_branch="main",
        generated_at=_NOW,
        stale_days=60,
    )

    assert decision.classification == "review-unreferenced-unmerged"
    assert decision.recommended_action == (
        "manually-prove-unreferenced-before-deletion"
    )
    assert not decision.deletion_candidate


def test_exact_reference_scan_ignores_untracked_and_non_evidence_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "docs" / "evidence.md").write_text(
        f"source revision `{_UNMERGED_SHA}`\nduplicate `{_UNMERGED_SHA}`\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "untracked.md").write_text(
        _TAGGED_SHA,
        encoding="utf-8",
    )
    (tmp_path / "src" / "fixture.py").write_text(
        _TAGGED_SHA,
        encoding="utf-8",
    )

    references = scan_exact_head_references(
        tmp_path,
        (_UNMERGED_SHA, _TAGGED_SHA),
        tracked_paths=frozenset({"docs/evidence.md", "src/fixture.py"}),
    )

    assert references == {
        _UNMERGED_SHA: (ReferenceLocation("docs/evidence.md", (1, 2)),)
    }


def test_reference_scan_rejects_ambiguous_patterns(tmp_path: Path) -> None:
    with pytest.raises(AuditError, match="stay inside"):
        scan_exact_head_references(
            tmp_path,
            (_UNMERGED_SHA,),
            globs=("../outside.md",),
        )


class _Inspector:
    def __init__(self) -> None:
        self.times = {
            _MAIN_SHA: _NOW - timedelta(days=1),
            _MERGED_SHA: _NOW - timedelta(days=100),
            _UNMERGED_SHA: _NOW - timedelta(days=100),
            _TAGGED_SHA: _NOW - timedelta(days=100),
        }

    def ensure_commit(self, revision: str) -> None:
        if revision not in self.times:
            raise AssertionError(revision)

    def commit_time(self, revision: str) -> datetime:
        return self.times[revision]

    def is_ancestor(self, revision: str, default_revision: str) -> bool:
        assert default_revision == _MAIN_SHA
        return revision in {_MAIN_SHA, _MERGED_SHA}


def _inventory(*, reverse: bool) -> dict[str, object]:
    branches = [
        RemoteBranch("main", _MAIN_SHA, True),
        RemoteBranch("agent/merged", _MERGED_SHA, False),
        RemoteBranch("agent/evidence", _UNMERGED_SHA, False),
        RemoteBranch("agent/tagged", _TAGGED_SHA, False),
    ]
    pull_requests = {
        "agent/evidence": [
            _pull_request(12, head_revision=_UNMERGED_SHA),
            _pull_request(8, head_revision=_UNMERGED_SHA),
        ]
    }
    references = {
        _UNMERGED_SHA: [
            ReferenceLocation("protocols/z.json", (9,)),
            ReferenceLocation("docs/a.md", (4,)),
        ]
    }
    if reverse:
        branches.reverse()
        pull_requests["agent/evidence"].reverse()
        references[_UNMERGED_SHA].reverse()
    return build_inventory(
        repository="example/project",
        default_branch="main",
        branches=branches,
        pull_requests_by_branch=pull_requests,
        tags_by_revision={_TAGGED_SHA: ["z-tag", "a-tag", "z-tag"]},
        references_by_revision=references,
        inspector=_Inspector(),  # type: ignore[arg-type]
        generated_at=_NOW,
        stale_days=60,
        reference_globs=("protocols/**/*.json", "docs/**/*.md"),
    )


def test_inventory_identity_is_input_order_invariant() -> None:
    first = _inventory(reverse=False)
    second = _inventory(reverse=True)

    assert first == second
    assert first["summary"] == {
        "branch_count": 4,
        "non_default_branch_count": 3,
        "deletion_candidate_count": 2,
        "tag_required_count": 1,
        "manual_review_count": 0,
        "classification_counts": {
            "deletion-candidate-merged": 1,
            "deletion-candidate-tagged": 1,
            "retain-default": 1,
            "retain-evidence-needs-tag": 1,
        },
    }


def test_markdown_separates_preserved_tag_required_and_manual_rows() -> None:
    inventory = _inventory(reverse=False)
    markdown = render_markdown(inventory)

    assert "Deletion candidates already preserved" in markdown
    assert "Branches that need an immutable tag first" in markdown
    assert "`agent/merged`" in markdown
    assert "`agent/evidence`" in markdown
    assert "not an authorization to delete" in markdown


def test_atomic_output_round_trip_and_symlink_rejection(tmp_path: Path) -> None:
    inventory = _inventory(reverse=False)
    json_path = tmp_path / "inventory.json"
    markdown_path = tmp_path / "report.md"

    write_inventory(
        inventory,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    assert json.loads(json_path.read_text(encoding="utf-8")) == inventory
    assert markdown_path.read_text(encoding="utf-8").startswith(
        "# Branch lifecycle audit"
    )

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    symlink = tmp_path / "linked.json"
    symlink.symlink_to(target)
    with pytest.raises(AuditError, match="symlink"):
        write_inventory(
            inventory,
            json_path=symlink,
            markdown_path=markdown_path,
        )


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def test_git_inspector_uses_exact_commit_time_and_ancestry(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "user.name", "Branch Audit Test")
    _git(tmp_path, "config", "user.email", "branch-audit@example.invalid")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("main\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-m", "main")
    main_revision = _git(tmp_path, "rev-parse", "HEAD")

    _git(tmp_path, "switch", "-c", "agent/unmerged")
    tracked.write_text("unmerged\n", encoding="utf-8")
    _git(tmp_path, "commit", "-am", "unmerged")
    unmerged_revision = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "switch", "main")

    inspector = GitInspector(tmp_path)

    assert inspector.is_ancestor(main_revision, main_revision)
    assert not inspector.is_ancestor(unmerged_revision, main_revision)
    assert inspector.commit_time(main_revision).tzinfo is not None
    assert inspector.tracked_paths() == frozenset({"tracked.txt"})


def test_api_parsers_reject_coercion_and_ignore_foreign_prs() -> None:
    with pytest.raises(AuditError, match="protected must be boolean"):
        RemoteBranch.from_api(
            {"name": "agent/example", "commit": {"sha": _UNMERGED_SHA}, "protected": 0},
            index=0,
        )

    foreign = PullRequest.from_api(
        {
            "number": 3,
            "state": "open",
            "draft": False,
            "merged_at": None,
            "updated_at": _NOW.isoformat(),
            "html_url": "https://github.com/other/project/pull/3",
            "head": {
                "ref": "agent/example",
                "sha": _UNMERGED_SHA,
                "repo": {"full_name": "other/project"},
            },
        },
        repository="example/project",
        index=0,
    )
    assert foreign is None


class _PagingClient(GitHubClient):
    def __init__(self, pages: dict[int, list[object]]) -> None:
        super().__init__(token=None)
        self.pages = pages

    def _get(self, path: str) -> object:
        page = int(path.rsplit("page=", 1)[1])
        return self.pages.get(page, [])


def test_pagination_stops_after_short_page() -> None:
    client = _PagingClient({1: [1, 2], 2: [3]})
    assert client.paginated_list("/fixture", per_page=2) == [1, 2, 3]


def test_workflow_is_read_only_and_never_runs_on_pull_requests() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / ".github" / "workflows" / "branch-lifecycle-audit.yml"
    text = path.read_text(encoding="utf-8")
    workflow: dict[str, Any] = yaml.load(text, Loader=yaml.BaseLoader)

    assert set(workflow["on"]) == {"schedule", "workflow_dispatch"}
    assert workflow["permissions"] == {
        "actions": "read",
        "contents": "read",
        "pull-requests": "read",
    }
    assert "workflow_registry_audit.py" in text
    assert "workflow-registry.json" in text
    assert "refs/heads/*:refs/remotes/origin/*" in text
    assert "refs/tags/*:refs/tags/*" in text
    assert "git push" not in text
    assert "git branch -d" not in text
    assert "git branch -D" not in text
    assert "contents: write" not in text
    assert "pull_request:" not in text
