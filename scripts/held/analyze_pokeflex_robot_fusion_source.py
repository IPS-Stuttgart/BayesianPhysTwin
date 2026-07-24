#!/usr/bin/env python3
"""Analyze the complete locked PokeFlex robot-fusion source cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repository_root() / "src"))

from bayesian_phystwin.pokeflex_robot_fusion_regret_guard import (  # noqa: E402
    evaluate_pokeflex_robot_fusion_cross_object,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    output = args.output.resolve()
    paths = sorted(
        path
        for path in args.source_root.resolve().glob("*.json")
        if path.resolve() != output
    )
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    result = evaluate_pokeflex_robot_fusion_cross_object(payloads)
    result["source_artifacts"] = [
        {
            "path": str(path),
            "sha256": _sha256(path),
        }
        for path in paths
    ]
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"existing source evaluation differs: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "source": result["source"],
                "cross_object": result["cross_object"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
