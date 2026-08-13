from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools.maintenance.workflow_registry_audit import (
    RegisteredWorkflow,
    build_inventory,
    checked_in_workflows,
    collect_workflow_pages,
    inventory_markdown,
    parse_registered_workflow,
)


def _raw_workflow(
    workflow_id: int,
    path: str,
    *,
    state: str = "active",
    name: str | None = None,
) -> dict[str, object]:
    return {
        "id": workflow_id,
        "name": name or Path(path).stem,
        "path": path,
        "state": state,
        "html_url": (
            f"https://github.com/example/project/actions/workflows/{workflow_id}"
        ),
        "created_at": "2026-08-01T00:00:00Z",
        "updated_at": "2026-08-02T00:00:00Z",
    }


def _workflow_root(tmp_path: Path) -> Path:
    directory = tmp_path / ".github" / "workflows"
    directory.mkdir(parents=True)
    return tmp_path


def test_inventory_distinguishes_registry_history_from_checked_in_files(
    tmp_path: Path,
) -> None:
    root = _workflow_root(tmp_path)
    (root / ".github/workflows/ci.yml").write_text("name: CI\n", encoding="utf-8")
    workflows = (
        parse_registered_workflow(_raw_workflow(1, ".github/workflows/ci.yml")),
        parse_registered_workflow(_raw_workflow(2, ".github/workflows/deleted.yml")),
        parse_registered_workflow(
            _raw_workflow(
                3,
                ".github/workflows/disabled.yml",
                state="disabled_manually",
            )
        ),
    )

    inventory = build_inventory(
        "example/project",
        root,
        workflows,
        generated_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
    )

    assert inventory["registry_workflow_count"] == 3
    assert inventory["checked_in_workflow_file_count"] == 1
    assert inventory["classification_counts"] == {
        "checked-in": 1,
        "orphaned-active": 1,
        "orphaned-disabled": 1,
        "orphaned-other-state": 0,
    }
    assert inventory["checked_in_unregistered_paths"] == []
    report = inventory_markdown(inventory)
    assert "Registry entries: **3**" in report
    assert "Orphaned active registry entries: **1**" in report
    assert "`.github/workflows/deleted.yml`" in report


def test_inventory_reports_checked_in_file_not_yet_registered(tmp_path: Path) -> None:
    root = _workflow_root(tmp_path)
    path = root / ".github/workflows/new.yml"
    path.write_text("name: New\n", encoding="utf-8")

    inventory = build_inventory(
        "example/project",
        root,
        (),
        generated_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
    )

    assert inventory["checked_in_unregistered_paths"] == [".github/workflows/new.yml"]
    assert len(inventory["inventory_id"]) == 64


def test_parse_registered_workflow_rejects_noncanonical_paths() -> None:
    with pytest.raises(ValueError, match="canonical"):
        parse_registered_workflow(_raw_workflow(1, ".github/workflows/../outside.yml"))
    with pytest.raises(ValueError, match="directly below"):
        parse_registered_workflow(_raw_workflow(2, ".github/workflows/nested/ci.yml"))
    with pytest.raises(ValueError, match="POSIX"):
        parse_registered_workflow(_raw_workflow(3, ".github\\workflows\\ci.yml"))


def test_collect_workflow_pages_validates_stable_pagination() -> None:
    pages = {
        1: {
            "total_count": 3,
            "workflows": [
                _raw_workflow(1, ".github/workflows/a.yml"),
                _raw_workflow(2, ".github/workflows/b.yml"),
            ],
        },
        2: {
            "total_count": 3,
            "workflows": [_raw_workflow(3, ".github/workflows/c.yml")],
        },
    }

    workflows = collect_workflow_pages(
        lambda page, _per_page: pages[page],
        per_page=2,
    )

    assert tuple(workflow.workflow_id for workflow in workflows) == (1, 2, 3)


def test_collect_workflow_pages_rejects_duplicate_ids() -> None:
    payload = {
        "total_count": 2,
        "workflows": [
            _raw_workflow(1, ".github/workflows/a.yml"),
            _raw_workflow(1, ".github/workflows/b.yml"),
        ],
    }

    with pytest.raises(ValueError, match="duplicate workflow id"):
        collect_workflow_pages(lambda _page, _per_page: payload)


def test_checked_in_workflows_rejects_symlinks(tmp_path: Path) -> None:
    root = _workflow_root(tmp_path)
    source = root / "source.yml"
    source.write_text("name: Source\n", encoding="utf-8")
    link = root / ".github/workflows/link.yml"
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError, match="ordinary file"):
        checked_in_workflows(root)


def test_inventory_identity_is_independent_of_generation_time(tmp_path: Path) -> None:
    root = _workflow_root(tmp_path)
    (root / ".github/workflows/ci.yml").write_text("name: CI\n", encoding="utf-8")
    workflow = RegisteredWorkflow(
        workflow_id=1,
        name="CI",
        path=".github/workflows/ci.yml",
        state="active",
        html_url="https://github.com/example/project/actions/workflows/1",
        created_at=None,
        updated_at=None,
    )

    first = build_inventory(
        "example/project",
        root,
        (workflow,),
        generated_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
    )
    second = build_inventory(
        "example/project",
        root,
        (workflow,),
        generated_at=datetime(2026, 8, 13, tzinfo=timezone.utc),
    )

    assert first["generated_at_utc"] != second["generated_at_utc"]
    assert first["inventory_id"] == second["inventory_id"]
