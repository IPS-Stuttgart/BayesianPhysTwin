#!/usr/bin/env python3
"""Nested cross-object evaluation of the PokeFlex force-depth regret guard."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repository_root() / "src"))

from bayesian_phystwin.pokeflex_force_depth_regret_guard import (  # noqa: E402
    evaluate_pokeflex_force_depth_cross_object,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [path.resolve() for path in args.artifacts]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    result = evaluate_pokeflex_force_depth_cross_object(payloads)
    result["source_artifacts"] = [
        {"path": str(path), "sha256": _sha256(path)} for path in paths
    ]
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = args.output.resolve()
    if output.exists() and output.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"existing evaluation differs: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "cross_object": result["cross_object"],
                "candidate_bank_oracle": result["candidate_bank_oracle"],
                "fixed_arm_controls": result["fixed_arm_controls"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
