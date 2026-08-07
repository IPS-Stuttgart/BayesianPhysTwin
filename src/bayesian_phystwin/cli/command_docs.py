"""Render and verify registry-derived documentation for stable ``bpt`` routes."""

from __future__ import annotations

import argparse
import posixpath
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Final

from .command_registry import COMMANDS, CommandSpec, CommandStatus

BEGIN_MARKER: Final = "<!-- bpt-stable-commands:begin -->"
END_MARKER: Final = "<!-- bpt-stable-commands:end -->"
DOCUMENT_TARGETS: Final[tuple[PurePosixPath, ...]] = (
    PurePosixPath("README.md"),
    PurePosixPath("docs/command_line.md"),
    PurePosixPath("docs/experiment_index.md"),
)


def stable_commands() -> tuple[CommandSpec, ...]:
    """Return stable routes in their intentional public display order."""

    return tuple(
        command for command in COMMANDS if command.status is CommandStatus.STABLE
    )


def _sentence(text: str) -> str:
    rendered = text.strip()
    if not rendered:
        raise ValueError("stable command descriptions must be nonempty")
    rendered = rendered[0].upper() + rendered[1:]
    if rendered[-1] not in ".!?":
        rendered += "."
    return rendered


def _documentation_link(command: CommandSpec, target: PurePosixPath) -> str:
    documentation = command.documentation
    if documentation is None:
        raise ValueError(
            f"stable command {command.command_id!r} has no documentation path"
        )
    start = str(target.parent) if str(target.parent) != "." else "."
    return posixpath.relpath(documentation, start=start)


def render_stable_command_table(target: PurePosixPath) -> str:
    """Render the stable command table with links relative to ``target``."""

    lines = [
        "| Command | Purpose | Documentation |",
        "| --- | --- | --- |",
    ]
    for command in stable_commands():
        purpose = _sentence(command.description).replace("|", "\\|")
        documentation = _documentation_link(command, target)
        lines.append(
            f"| `{command.grouped_command}` | {purpose} | "
            f"[Guide]({documentation}) |"
        )
    return "\n".join(lines)


def render_stable_route_block() -> str:
    """Render the stable routes as a copyable text block."""

    routes = "\n".join(command.grouped_command for command in stable_commands())
    return f"```text\n{routes}\n```"


def render_generated_section(target: PurePosixPath) -> str:
    """Render the generated stable-command section for one tracked document."""

    if target == PurePosixPath("docs/command_line.md"):
        return render_stable_route_block()
    if target in {
        PurePosixPath("README.md"),
        PurePosixPath("docs/experiment_index.md"),
    }:
        return render_stable_command_table(target)
    raise ValueError(f"unsupported generated-command document: {target}")


def replace_generated_section(text: str, generated: str) -> str:
    """Replace exactly one generated section while preserving surrounding prose."""

    if text.count(BEGIN_MARKER) != 1 or text.count(END_MARKER) != 1:
        raise ValueError("document must contain exactly one stable-command marker pair")
    before, remainder = text.split(BEGIN_MARKER, 1)
    _, after = remainder.split(END_MARKER, 1)
    return f"{before}{BEGIN_MARKER}\n{generated.rstrip()}\n{END_MARKER}{after}"


def validate_stable_documentation(root: Path) -> None:
    """Require every stable command to own an existing Markdown guide."""

    for command in stable_commands():
        documentation = command.documentation
        if documentation is None:
            raise ValueError(
                f"stable command {command.command_id!r} has no documentation path"
            )
        path = root / documentation
        if not path.is_file():
            raise ValueError(
                f"documentation for {command.command_id!r} does not exist: "
                f"{documentation}"
            )


def synchronize_documents(root: Path, *, check: bool) -> tuple[PurePosixPath, ...]:
    """Check or rewrite all registry-derived stable-command sections."""

    validate_stable_documentation(root)
    changed: list[PurePosixPath] = []
    for target in DOCUMENT_TARGETS:
        path = root / target
        current = path.read_text(encoding="utf-8")
        expected = replace_generated_section(
            current,
            render_generated_section(target),
        )
        if expected == current:
            continue
        changed.append(target)
        if not check:
            path.write_text(expected, encoding="utf-8")
    return tuple(changed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="check or update registry-derived stable bpt command docs"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="fail when tracked documentation is out of date (default)",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="rewrite tracked generated sections in place",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="repository root containing README.md and docs/",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    changed = synchronize_documents(args.root.resolve(), check=not args.write)
    if not changed:
        print("stable command documentation is synchronized")
        return 0
    action = "out of date" if not args.write else "updated"
    for target in changed:
        print(f"{target}: {action}")
    return 1 if not args.write else 0


if __name__ == "__main__":
    raise SystemExit(main())
