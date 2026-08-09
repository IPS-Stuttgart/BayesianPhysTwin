#!/usr/bin/env python3
"""Aggregate the frozen missing-five PokeFlex source-scale bank."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _repository_root()
sys.path.insert(0, str(ROOT / "src"))

from bayesian_phystwin.pokeflex_missing5_scale import (  # noqa: E402
    build_source_result,
    file_sha256,
    validate_source_protocol,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to replace source result: {args.output}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validation = validate_source_protocol(protocol)
    paths = {
        take_id: args.artifact_root / f"{take_id}.json"
        for take_id in validation["source_take_ids"]
    }
    missing = [take_id for take_id, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"missing source artifacts: {missing}")
    artifacts = [
        json.loads(paths[take_id].read_text(encoding="utf-8"))
        for take_id in validation["source_take_ids"]
    ]
    digests = {take_id: file_sha256(path) for take_id, path in paths.items()}
    payload = build_source_result(
        artifacts,
        protocol,
        artifact_file_sha256s=digests,
        implementation_revision=protocol["implementation"]["revision"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "result_sha256": payload["result_sha256"],
                "source_gate": payload["source_gate"],
                "selected_multipliers": {
                    name: row["multiplier"] for name, row in payload["objects"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
