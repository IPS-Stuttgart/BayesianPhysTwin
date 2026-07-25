#!/usr/bin/env python3
"""Evaluate same-time PokeFlex D405 state-correction diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repository_root() / "src"))

from bayesian_phystwin.pokeflex_independent_depth_current_state_evaluation import (  # noqa: E402
    evaluate_current_state_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    payloads = [
        json.loads(path.resolve().read_text(encoding="utf-8"))
        for path in args.artifacts
    ]
    result = evaluate_current_state_artifacts(payloads)
    result["sources"] = [str(path.resolve()) for path in args.artifacts]
    if args.compact:
        for take in result["takes"]:
            take["competence"].pop("rows", None)
            take["selector"].pop("rows", None)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"existing current-state evaluation differs: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "competence": result["competence"],
                "object_balanced_selector": result["object_balanced_selector"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
