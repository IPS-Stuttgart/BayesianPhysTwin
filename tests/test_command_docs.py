from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from bayesian_phystwin.cli.command_docs import (
    BEGIN_MARKER,
    END_MARKER,
    DOCUMENT_TARGETS,
    render_generated_section,
    replace_generated_section,
    stable_commands,
    synchronize_documents,
)

ROOT = Path(__file__).resolve().parents[1]


def test_stable_commands_own_existing_documentation() -> None:
    commands = stable_commands()
    assert len(commands) == 7
    assert len({command.documentation for command in commands}) == len(commands)
    for command in commands:
        assert command.documentation is not None
        assert (ROOT / command.documentation).is_file()
        assert command.to_dict()["documentation"] == command.documentation


def test_registry_derived_documentation_is_synchronized() -> None:
    assert synchronize_documents(ROOT, check=True) == ()


def test_each_generated_section_contains_every_stable_route_once() -> None:
    for target in DOCUMENT_TARGETS:
        generated = render_generated_section(target)
        for command in stable_commands():
            assert generated.count(command.grouped_command) == 1


def test_replace_generated_section_requires_exactly_one_marker_pair() -> None:
    generated = render_generated_section(PurePosixPath("README.md"))
    with pytest.raises(ValueError, match="exactly one"):
        replace_generated_section("missing markers", generated)
    with pytest.raises(ValueError, match="exactly one"):
        replace_generated_section(
            f"{BEGIN_MARKER}\n{BEGIN_MARKER}\n{END_MARKER}",
            generated,
        )


def test_documentation_sync_write_is_idempotent(tmp_path: Path) -> None:
    for command in stable_commands():
        assert command.documentation is not None
        path = tmp_path / command.documentation
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Guide\n", encoding="utf-8")
    for target in DOCUMENT_TARGETS:
        path = tmp_path / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"before\n{BEGIN_MARKER}\nstale\n{END_MARKER}\nafter\n",
            encoding="utf-8",
        )

    assert synchronize_documents(tmp_path, check=False) == DOCUMENT_TARGETS
    assert synchronize_documents(tmp_path, check=True) == ()


def test_registry_rejects_missing_or_noncanonical_stable_documentation() -> None:
    from dataclasses import replace

    from bayesian_phystwin.cli.command_registry import COMMANDS, validate_registry

    stable = COMMANDS[0]
    with pytest.raises(ValueError, match="missing documentation"):
        validate_registry((replace(stable, documentation=None),))
    for documentation in ("README.md", "/docs/guide.md", "docs/../guide.md"):
        with pytest.raises(ValueError, match="invalid documentation path"):
            validate_registry((replace(stable, documentation=documentation),))
