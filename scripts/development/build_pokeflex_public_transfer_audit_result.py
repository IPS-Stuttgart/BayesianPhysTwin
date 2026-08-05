#!/usr/bin/env python3
"""Aggregate the 78-action retrospective PokeFlex public transfer audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _repository_root()
sys.path.insert(0, str(ROOT / "src"))

from bayesian_phystwin.pokeflex_public_transfer_audit import (  # noqa: E402
    build_public_transfer_result,
    file_sha256,
    validate_public_transfer_protocol,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            ROOT
            / "configs"
            / "sota"
            / "pokeflex_action_robust_public78_retrospective_v6.json"
        ),
    )
    parser.add_argument(
        "--prospective-result",
        type=Path,
        default=(
            ROOT
            / "results"
            / "sota"
            / "pokeflex_action_robust_fresh2_v5"
            / "target_result.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validation = validate_public_transfer_protocol(protocol)
    paths = {
        take_id: args.artifact_root / f"{take_id}.json"
        for take_id in validation["retrospective_take_ids"]
    }
    missing = [take_id for take_id, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"missing retrospective artifacts: {missing}")
    artifacts = {
        take_id: json.loads(path.read_text(encoding="utf-8"))
        for take_id, path in paths.items()
    }
    digests = {take_id: file_sha256(path) for take_id, path in paths.items()}
    prospective = json.loads(args.prospective_result.read_text(encoding="utf-8"))
    payload = build_public_transfer_result(
        protocol,
        artifacts,
        smoke_artifact_file_sha256s=digests,
        prospective_target_result=prospective,
        prospective_target_result_file_sha256=file_sha256(args.prospective_result),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
