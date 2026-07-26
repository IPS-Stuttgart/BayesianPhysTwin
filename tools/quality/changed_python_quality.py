#!/usr/bin/env python3
"""Apply incremental Python quality checks to changed and stable modules.

The repository contains a large historical experiment surface. This helper
ratchets quality without requiring an all-at-once cleanup:

* every added or modified Python file is linted and format-checked;
* every added or modified package module is type-checked;
* stable public contracts and scientific-core modules are always type-checked;
* a smaller, mature subset is checked with ``mypy --strict``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

_STABLE_TYPE_TARGETS = (
    "src/bayesian_phystwin/run_manifest.py",
    "src/bayesian_phystwin/repository_provenance.py",
    "src/bayesian_phystwin/run_manifest_v2.py",
    "src/bayesian_phystwin/cli/main.py",
    "src/bayesian_phystwin/cli/run_manifest.py",
    "src/bayesian_phystwin/observation_belief.py",
    "src/bayesian_phystwin/observation_belief_gauge_adapter.py",
    "src/bayesian_phystwin/_gauge_aware_contracts.py",
    "src/bayesian_phystwin/_gauge_aware_solver.py",
    "src/bayesian_phystwin/gauge_aware_belief.py",
    "src/bayesian_phystwin/causal4d_provider_v1.py",
    "src/bayesian_phystwin/prob4d_observation_contract.py",
    "src/bayesian_phystwin/prob4d_causal_lineage.py",
)

_STRICT_TYPE_TARGETS = (
    "src/bayesian_phystwin/run_manifest.py",
    "src/bayesian_phystwin/repository_provenance.py",
    "src/bayesian_phystwin/run_manifest_v2.py",
    "src/bayesian_phystwin/cli/main.py",
    "src/bayesian_phystwin/cli/run_manifest.py",
    "src/bayesian_phystwin/gauge_aware_belief.py",
)


def _run(command: Sequence[str], *, label: str) -> None:
    printable = " ".join(command)
    print(f"\n==> {label}\n$ {printable}", flush=True)
    subprocess.run(command, cwd=_REPOSITORY_ROOT, check=True)


def _git_text(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=_REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _git_bytes(*arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=_REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _commit_exists(revision: str) -> bool:
    if not revision or set(revision) == {"0"}:
        return False
    completed = subprocess.run(
        ("git", "rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}"),
        cwd=_REPOSITORY_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def _resolve_base(base: str | None, head: str) -> str | None:
    if base and _commit_exists(base):
        return base
    parent = f"{head}^"
    if _commit_exists(parent):
        return parent
    return None


def _changed_python_files(base: str | None, head: str) -> tuple[str, ...]:
    if base is None:
        print(
            "No comparison commit is available; changed-file checks are skipped. "
            "Stable targets remain enforced.",
            flush=True,
        )
        return ()
    output = _git_bytes(
        "diff",
        "--name-only",
        "--diff-filter=ACMR",
        "-z",
        f"{base}...{head}",
        "--",
        ":(glob)**/*.py",
        ":(glob)**/*.pyi",
    )
    paths = tuple(
        os.fsdecode(path)
        for path in output.split(b"\0")
        if path
    )
    return tuple(
        sorted(
            path
            for path in paths
            if (_REPOSITORY_ROOT / path).is_file()
        )
    )


def _existing_unique(paths: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            path for path in paths if (_REPOSITORY_ROOT / path).is_file()
        )
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run changed-file and stable-interface Python quality checks."
    )
    parser.add_argument(
        "--base",
        default=None,
        help="comparison commit; missing or all-zero values fall back to HEAD^",
    )
    parser.add_argument(
        "--head",
        default="HEAD",
        help="head commit to inspect (default: HEAD)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    head = _git_text("rev-parse", arguments.head)
    base = _resolve_base(arguments.base, head)
    changed_python = _changed_python_files(base, head)

    if changed_python:
        print("Changed Python files:", flush=True)
        for path in changed_python:
            print(f"  {path}", flush=True)
        _run(
            (sys.executable, "-m", "ruff", "check", "--", *changed_python),
            label="Ruff lint for changed Python files",
        )
        _run(
            (
                sys.executable,
                "-m",
                "ruff",
                "format",
                "--check",
                "--",
                *changed_python,
            ),
            label="Ruff format for changed Python files",
        )
    else:
        print("No added or modified Python files require Ruff checks.", flush=True)

    changed_package_modules = tuple(
        path
        for path in changed_python
        if path.startswith("src/bayesian_phystwin/")
    )
    type_targets = _existing_unique(
        (*_STABLE_TYPE_TARGETS, *changed_package_modules)
    )
    _run(
        (sys.executable, "-m", "mypy", *type_targets),
        label="Mypy for stable and changed package modules",
    )

    strict_targets = _existing_unique(_STRICT_TYPE_TARGETS)
    _run(
        (sys.executable, "-m", "mypy", "--strict", *strict_targets),
        label="Strict mypy for mature public interfaces",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
