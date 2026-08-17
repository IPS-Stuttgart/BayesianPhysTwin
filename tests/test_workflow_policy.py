from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path

import pytest

from tools.quality.check_workflow_policy import inspect_workflow, validate_repository

_TODAY = date(2026, 8, 12)


def _permanent_workflow(*, action: str = "actions/checkout@" + "a" * 40) -> str:
    return f"""# workflow-lifecycle: permanent
# workflow-owner: IPS-Stuttgart maintainers
name: Permanent check

on:
  pull_request:

permissions:
  contents: read

concurrency:
  group: permanent-${{{{ github.ref }}}}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: {action}
"""


def _temporary_workflow(*, expiry: str = "2026-08-20") -> str:
    return f"""# workflow-lifecycle: temporary
# workflow-owner: IPS-Stuttgart maintainers
# workflow-issue: #999
# workflow-expiry: {expiry}
name: Temporary migration

on:
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: temporary-migration
  cancel-in-progress: false

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@{"b" * 40}
"""


def _legacy_workflow(*, name: str = "Historical workflow") -> str:
    return f"""name: {name}
on:
  workflow_dispatch:
permissions:
  contents: read
concurrency:
  group: historical
jobs: {{}}
"""


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _initialize_repository(repository: Path) -> None:
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "Workflow Policy Test")
    _git(repository, "config", "user.email", "workflow@example.invalid")
    (repository / ".github/workflows").mkdir(parents=True)


def _write_workflow(repository: Path, name: str, content: str) -> Path:
    path = repository / ".github/workflows" / name
    path.write_text(content, encoding="utf-8")
    return path


def _commit(repository: Path, message: str) -> str:
    _git(repository, "add", "-A")
    _git(repository, "commit", "-m", message)
    return _git(repository, "rev-parse", "HEAD")


def test_valid_permanent_workflow_passes() -> None:
    record = inspect_workflow(
        Path(".github/workflows/package.yml"),
        _permanent_workflow(),
        today=_TODAY,
        require_managed=True,
    )

    assert record.lifecycle == "permanent"
    assert record.violations == ()


def test_new_workflow_requires_lifecycle_metadata() -> None:
    text = _permanent_workflow().replace("# workflow-lifecycle: permanent\n", "")

    record = inspect_workflow(
        Path(".github/workflows/package.yml"),
        text,
        today=_TODAY,
        require_managed=True,
    )

    assert record.lifecycle == "legacy"
    assert any("workflow-lifecycle" in item for item in record.violations)


def test_permanent_workflow_rejects_temporary_filename() -> None:
    record = inspect_workflow(
        Path(".github/workflows/_fix-release-once.yml"),
        _permanent_workflow(),
        today=_TODAY,
        require_managed=True,
    )

    assert record.temporary_looking_name is True
    assert any("temporary-looking" in item for item in record.violations)


def test_external_actions_require_full_commit_sha() -> None:
    record = inspect_workflow(
        Path(".github/workflows/package.yml"),
        _permanent_workflow(action="actions/checkout@v7"),
        today=_TODAY,
        require_managed=True,
    )

    assert any("full commit SHAs" in item for item in record.violations)


def test_valid_temporary_workflow_passes_before_expiry() -> None:
    record = inspect_workflow(
        Path(".github/workflows/migration-temporary.yml"),
        _temporary_workflow(),
        today=_TODAY,
        require_managed=True,
    )

    assert record.lifecycle == "temporary"
    assert record.violations == ()


def test_temporary_workflow_fails_after_expiry() -> None:
    record = inspect_workflow(
        Path(".github/workflows/migration-temporary.yml"),
        _temporary_workflow(expiry="2026-08-11"),
        today=_TODAY,
    )

    assert any("expired" in item for item in record.violations)


def test_temporary_workflow_rejects_automatic_trigger() -> None:
    text = _temporary_workflow().replace(
        "  workflow_dispatch:\n", "  workflow_dispatch:\n  schedule:\n"
    )

    record = inspect_workflow(
        Path(".github/workflows/migration-temporary.yml"),
        text,
        today=_TODAY,
        require_managed=True,
    )

    assert any("may not use" in item for item in record.violations)


def test_legacy_workflow_is_inventoried_without_retroactive_failure() -> None:
    record = inspect_workflow(
        Path(".github/workflows/_one_shot_historical.yml"),
        _legacy_workflow(name="Historical one shot"),
        today=_TODAY,
    )

    assert record.lifecycle == "legacy"
    assert record.temporary_looking_name is True
    assert record.violations == ()


def test_duplicate_and_unknown_header_metadata_are_rejected() -> None:
    text = _permanent_workflow().replace(
        "# workflow-owner: IPS-Stuttgart maintainers\n",
        "# workflow-owner: IPS-Stuttgart maintainers\n"
        "# workflow-owner: Another owner\n"
        "# workflow-purpose: paper run\n",
    )

    record = inspect_workflow(
        Path(".github/workflows/package.yml"),
        text,
        today=_TODAY,
        require_managed=True,
    )

    assert any("duplicate workflow metadata" in item for item in record.violations)
    assert any("unknown workflow metadata" in item for item in record.violations)


def test_invalid_lifecycle_value_is_rejected() -> None:
    text = _permanent_workflow().replace(
        "# workflow-lifecycle: permanent", "# workflow-lifecycle: frozen-record"
    )

    record = inspect_workflow(
        Path(".github/workflows/package.yml"),
        text,
        today=_TODAY,
        require_managed=True,
    )

    assert record.lifecycle == "legacy"
    assert any("must be 'permanent' or 'temporary'" in item for item in record.violations)


def test_modified_legacy_workflow_enters_the_lifecycle_ratchet(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    _initialize_repository(repository)
    _write_workflow(repository, "historical.yml", _legacy_workflow())
    base = _commit(repository, "base")

    _write_workflow(
        repository,
        "historical.yml",
        _legacy_workflow(name="Modified historical workflow"),
    )
    head = _commit(repository, "modify workflow")

    records = validate_repository(
        repository,
        base=base,
        head=head,
        today=_TODAY,
    )

    assert len(records) == 1
    assert records[0].path == ".github/workflows/historical.yml"
    assert any("added or modified" in item for item in records[0].violations)


def test_untouched_legacy_workflow_remains_grandfathered(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _initialize_repository(repository)
    _write_workflow(repository, "historical.yml", _legacy_workflow())
    base = _commit(repository, "base")

    (repository / "notes.md").write_text("unrelated\n", encoding="utf-8")
    head = _commit(repository, "unrelated change")

    assert (
        validate_repository(
            repository,
            base=base,
            head=head,
            today=_TODAY,
        )
        == ()
    )


def test_deleted_workflow_is_not_reclassified(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _initialize_repository(repository)
    workflow = _write_workflow(repository, "historical.yml", _legacy_workflow())
    base = _commit(repository, "base")

    workflow.unlink()
    head = _commit(repository, "delete workflow")

    assert (
        validate_repository(
            repository,
            base=base,
            head=head,
            today=_TODAY,
        )
        == ()
    )


def test_requested_head_must_match_checkout(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _initialize_repository(repository)
    _write_workflow(repository, "managed.yml", _permanent_workflow())
    base = _commit(repository, "base")

    _write_workflow(
        repository,
        "managed.yml",
        _permanent_workflow().replace("Permanent check", "Updated check"),
    )
    head = _commit(repository, "head")
    _git(repository, "checkout", "--detach", base)

    with pytest.raises(ValueError, match="HEAD does not match"):
        validate_repository(
            repository,
            base=base,
            head=head,
            today=_TODAY,
        )


def test_changed_workflow_symlink_is_rejected(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _initialize_repository(repository)
    _write_workflow(repository, "managed.yml", _permanent_workflow())
    base = _commit(repository, "base")

    target = repository / "target.yml"
    target.write_text(_permanent_workflow(), encoding="utf-8")
    linked = repository / ".github/workflows/linked.yml"
    try:
        linked.symlink_to("../../target.yml")
    except OSError:
        pytest.skip("symlink creation is unavailable")
    head = _commit(repository, "add workflow symlink")

    with pytest.raises(ValueError, match="must not be a symlink"):
        validate_repository(
            repository,
            base=base,
            head=head,
            today=_TODAY,
        )
