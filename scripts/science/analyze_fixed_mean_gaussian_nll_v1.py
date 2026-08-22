#!/usr/bin/env python3
"""Run the retrospective fixed-mean Gaussian NLL decomposition."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from fixed_mean_gaussian_nll_v1 import analyze_fixed_mean_gaussian_nll


def _write_json(path: Path, payload: object, *, force: bool) -> None:
    target = path.resolve()
    if target.exists() and not force:
        raise FileExistsError(f"output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{os.getpid()}"
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        payload = json.loads(arguments.input.read_text(encoding="utf-8"))
        report = analyze_fixed_mean_gaussian_nll(payload)
        _write_json(arguments.output, report, force=arguments.force)
    except (OSError, ValueError, FloatingPointError, json.JSONDecodeError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
