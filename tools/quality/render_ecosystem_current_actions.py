#!/usr/bin/env python3
"""Render the generated ecosystem current-action snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from check_ecosystem_current_actions import (
    DEFAULT_RECORDS,
    DEFAULT_REGISTRY,
    EcosystemCurrentActionsError,
    render_registry_text,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=DEFAULT_RECORDS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REGISTRY,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of writing when the checked-in output differs.",
    )
    args = parser.parse_args()
    try:
        rendered = render_registry_text(args.records)
    except EcosystemCurrentActionsError as exc:
        parser.exit(1, f"cannot render ecosystem current actions: {exc}\n")

    if args.check:
        try:
            current = args.output.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            parser.exit(1, f"cannot read generated output: {exc}\n")
        if current != rendered:
            parser.exit(
                1,
                "ecosystem current-action snapshot is stale; "
                "run tools/quality/render_ecosystem_current_actions.py\n",
            )
        print(f"{args.output} is current")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
