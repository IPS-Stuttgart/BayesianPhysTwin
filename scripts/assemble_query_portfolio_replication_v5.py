#!/usr/bin/env python3
"""Assemble complete v5 component records into the joint certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from bayesian_phystwin._portable_contracts import write_atomic_json
from bayesian_phystwin.query_portfolio_evidence_v2 import assemble


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("component evidence must be a JSON object")
    return cast(dict[str, Any], value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wrapping", type=Path)
    parser.add_argument("slingshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = assemble(
        {
            "dlolab_wrapping_v9": _load(args.wrapping),
            "dlolab_slingshot_v4": _load(args.slingshot),
        }
    )
    write_atomic_json(result, args.output, overwrite=False)


if __name__ == "__main__":
    main()
