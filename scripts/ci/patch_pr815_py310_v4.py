#!/usr/bin/env python3
"""Patch the one known Python 3.10-incompatible multiline f-string in PR 815."""

from __future__ import annotations

import argparse
from pathlib import Path


def patch(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    old = '''    if selected.size == 0:
        raise EvaluationError(
            f"episode {episode.descriptor.episode_id} has no windows for horizon {
                horizon
            }"
        )
'''
    new = '''    if selected.size == 0:
        message = (
            f"episode {episode.descriptor.episode_id} has no windows "
            f"for horizon {horizon}"
        )
        raise EvaluationError(message)
'''
    if old not in source:
        if new in source:
            return
        raise SystemExit("expected multiline f-string block was not found")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    patch(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
