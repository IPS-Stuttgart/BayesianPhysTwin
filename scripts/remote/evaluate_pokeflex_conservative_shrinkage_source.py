#!/usr/bin/env python3
"""Evaluate the frozen PokeFlex conservative-shrinkage source panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repository_root() / "src"))

from bayesian_phystwin.pokeflex_conservative_shrinkage import (  # noqa: E402
    ConservativeShrinkageConfig,
    evaluate_pokeflex_conservative_shrinkage_source,
    load_pokeflex_conservative_shrinkage_protocol,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_pokeflex_conservative_shrinkage_protocol(args.protocol)
    selection = protocol["payload"]["selection"]
    paths = tuple(path.resolve() for path in args.artifacts)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    result = evaluate_pokeflex_conservative_shrinkage_source(
        payloads,
        expected_take_ids=protocol["opened_source_take_ids"],
        config=ConservativeShrinkageConfig(
            minimum_object_balanced_improvement=float(
                selection["minimum_object_balanced_relative_improvement"]
            ),
            maximum_object_regression=float(
                selection["maximum_per_object_relative_regression"]
            ),
        ),
    )
    result["protocol_sha256"] = protocol["protocol_sha256"]
    result["source_artifact_sha256s"] = {str(path): _sha256(path) for path in paths}
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = args.output.resolve()
    if output.exists() and output.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"existing source result differs: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "source_gate_passed": result["source_gate_passed"],
                "selected_arm": result["selected_arm"],
                "relative_improvement": result["selected_result"][
                    "object_balanced_relative_improvement"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
