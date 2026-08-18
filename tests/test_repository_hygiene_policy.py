from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/quality/check_repository_hygiene.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_repository_hygiene", MODULE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_agent_placeholder_filename_is_rejected(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "docs/.agent-placeholder"
    path.parent.mkdir(parents=True)
    path.write_text("placeholder\n", encoding="utf-8")

    assert module.find_repository_hygiene_violations(
        tmp_path,
        ("docs/.agent-placeholder",),
    ) == ("forbidden placeholder filename: docs/.agent-placeholder",)


def test_placeholder_only_maintained_file_is_rejected(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "docs/status.md"
    path.parent.mkdir(parents=True)
    path.write_text("  Temporary Placeholder\n", encoding="utf-8")

    assert module.find_repository_hygiene_violations(
        tmp_path,
        ("docs/status.md",),
    ) == ("placeholder-only maintained file: docs/status.md",)


def test_test_fixtures_and_substantive_text_are_not_rejected(tmp_path: Path) -> None:
    module = _module()
    fixture = tmp_path / "tests/fixtures/placeholder.txt"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("placeholder\n", encoding="utf-8")
    document = tmp_path / "docs/status.md"
    document.parent.mkdir(parents=True)
    document.write_text("Placeholder values are rejected.\n", encoding="utf-8")

    assert (
        module.find_repository_hygiene_violations(
            tmp_path,
            ("tests/fixtures/placeholder.txt", "docs/status.md"),
        )
        == ()
    )


def test_noncanonical_paths_fail_closed(tmp_path: Path) -> None:
    module = _module()

    assert module.find_repository_hygiene_violations(
        tmp_path,
        ("../outside",),
    ) == ("noncanonical tracked path: ../outside",)
