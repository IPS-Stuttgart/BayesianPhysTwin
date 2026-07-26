"""Fail when repository-wide Ruff diagnostics exceed the checked-in baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import cast


class RuffDiagnostic(dict[str, object]):
    """Validated JSON object emitted by Ruff for one diagnostic."""


def _relative_path(repository_root: Path, filename: str) -> str:
    path = Path(filename)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(repository_root.resolve())
        except ValueError as error:
            raise ValueError(
                f"Ruff reported a path outside the repository: {path}"
            ) from error
    return path.as_posix()


def _load_baseline(path: Path) -> Counter[tuple[str, str]]:
    raw_payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError("Ruff baseline must be an object")
    payload = cast(dict[str, object], raw_payload)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported Ruff baseline schema")

    raw_diagnostics = payload.get("diagnostics")
    if not isinstance(raw_diagnostics, dict):
        raise ValueError("Ruff baseline diagnostics must be an object")
    diagnostics = cast(dict[object, object], raw_diagnostics)

    baseline: Counter[tuple[str, str]] = Counter()
    for path_value, raw_codes in diagnostics.items():
        if not isinstance(path_value, str) or not path_value:
            raise ValueError("Ruff baseline contains an invalid path")
        if not isinstance(raw_codes, dict):
            raise ValueError(f"Ruff baseline codes must be an object: {path_value}")
        codes = cast(dict[object, object], raw_codes)
        normalized_path = Path(path_value).as_posix()
        for code, count in codes.items():
            if not isinstance(code, str) or not code:
                raise ValueError(
                    f"Ruff baseline contains an invalid code: {path_value}"
                )
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError(
                    f"Ruff baseline contains an invalid count: {path_value} {code}"
                )
            baseline[(normalized_path, code)] = count
    return baseline


def _run_ruff(
    repository_root: Path,
    paths: Sequence[str],
) -> list[RuffDiagnostic]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            *paths,
            "--output-format=json",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            f"Ruff failed before producing diagnostics (exit {completed.returncode})"
        )

    raw_payload: object = json.loads(completed.stdout or "[]")
    if not isinstance(raw_payload, list) or any(
        not isinstance(item, dict) for item in raw_payload
    ):
        raise ValueError("Ruff JSON output has an unexpected shape")
    return [RuffDiagnostic(cast(dict[str, object], item)) for item in raw_payload]


def check_ruff_baseline(
    repository_root: Path,
    baseline_path: Path,
    paths: Sequence[str],
    *,
    report_path: Path | None = None,
) -> int:
    """Return zero when no path/code diagnostic count exceeds the baseline."""

    resolved_root = repository_root.resolve()
    resolved_baseline = (
        baseline_path
        if baseline_path.is_absolute()
        else resolved_root / baseline_path
    )
    baseline = _load_baseline(resolved_baseline)
    diagnostics = _run_ruff(resolved_root, paths)
    if report_path is not None:
        resolved_report = (
            report_path if report_path.is_absolute() else resolved_root / report_path
        )
        resolved_report.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    current: Counter[tuple[str, str]] = Counter()
    messages: dict[tuple[str, str], str] = {}
    for diagnostic in diagnostics:
        filename = diagnostic.get("filename")
        code = diagnostic.get("code")
        message = diagnostic.get("message")
        if not isinstance(filename, str) or not isinstance(code, str):
            raise ValueError("Ruff diagnostic is missing filename or code")
        key = (_relative_path(resolved_root, filename), code)
        current[key] += 1
        if isinstance(message, str):
            messages.setdefault(key, message)

    unexpected = current - baseline
    removed = baseline - current
    print(
        "Ruff baseline summary: "
        f"current={sum(current.values())}, "
        f"baseline={sum(baseline.values())}, "
        f"removed={sum(removed.values())}, "
        f"unexpected={sum(unexpected.values())}"
    )
    if unexpected:
        print("Unexpected Ruff diagnostics:", file=sys.stderr)
        for (path, code), count in sorted(unexpected.items()):
            message = messages.get((path, code), "")
            suffix = f": {message}" if message else ""
            print(f"  {path}: {code} x{count}{suffix}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="repository to lint",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("tools/ruff_baseline.json"),
        help="checked-in Ruff diagnostic baseline",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="optional path for the complete current Ruff JSON report",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["src", "tests", "tools"],
        help="paths passed to Ruff",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return check_ruff_baseline(
        cast(Path, args.repository_root),
        cast(Path, args.baseline),
        cast(list[str], args.paths),
        report_path=cast(Path | None, args.report),
    )


if __name__ == "__main__":
    raise SystemExit(main())
