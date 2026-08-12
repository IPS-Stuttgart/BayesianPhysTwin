from __future__ import annotations

from datetime import date
from pathlib import Path

from tools.quality.check_workflow_policy import inspect_workflow

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
    text = """name: Historical one shot
on:
  workflow_dispatch:
permissions:
  contents: read
concurrency:
  group: historical
jobs: {}
"""

    record = inspect_workflow(
        Path(".github/workflows/_one_shot_historical.yml"),
        text,
        today=_TODAY,
    )

    assert record.lifecycle == "legacy"
    assert record.temporary_looking_name is True
    assert record.violations == ()
