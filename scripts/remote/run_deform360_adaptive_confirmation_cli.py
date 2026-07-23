#!/usr/bin/env python3
"""Source-only bootstrap for the H2 adaptive-confirmation CLI."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import sys


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _adapter_repository(arguments: list[str]) -> Path:
    values: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--adapter-repo":
            _require(index + 1 < len(arguments), "--adapter-repo has no value")
            values.append(arguments[index + 1])
            index += 2
            continue
        if argument.startswith("--adapter-repo="):
            values.append(argument.split("=", 1)[1])
        elif argument.startswith("--adapter") and "--adapter-repo".startswith(
            argument.split("=", 1)[0]
        ):
            raise ValueError("adapter repository option abbreviation is forbidden")
        index += 1
    _require(
        len(values) == 1 and bool(values[0]), "exactly one --adapter-repo is required"
    )
    return Path(values[0]).absolute()


def _reject_adapter_python_caches(adapter: Path) -> None:
    """Scan the adapter with stdlib only, before any adapter import."""

    _require(
        adapter.is_dir()
        and not adapter.is_symlink()
        and adapter.resolve(strict=True) == adapter,
        "adapter repository is invalid",
    )
    for root in (adapter / "src", adapter / "scripts"):
        _require(
            root.is_dir() and not root.is_symlink(),
            f"adapter Python source root is invalid: {root}",
        )
        pending = [root]
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    observed = entry.stat(follow_symlinks=False)
                    _require(
                        not stat.S_ISLNK(observed.st_mode),
                        f"adapter Python tree contains a symlink: {path}",
                    )
                    if stat.S_ISDIR(observed.st_mode):
                        _require(
                            entry.name != "__pycache__",
                            f"adapter Python bytecode cache is forbidden: {path}",
                        )
                        pending.append(path)
                    elif stat.S_ISREG(observed.st_mode):
                        _require(
                            path.suffix.lower() not in {".pyc", ".pyo"},
                            f"adapter Python bytecode is forbidden: {path}",
                        )
                    else:
                        _require(
                            False,
                            f"adapter Python tree contains a special file: {path}",
                        )


def main() -> None:
    adapter = _adapter_repository(sys.argv[1:])
    sys.dont_write_bytecode = True
    _reject_adapter_python_caches(adapter)
    sys.path.insert(0, str(adapter / "src"))

    from bayesian_phystwin.cli.deform360_adaptive_covariance_confirmation import (
        main as confirmation_main,
    )

    confirmation_main(source_bootstrap_file=__file__)


if __name__ == "__main__":
    main()
