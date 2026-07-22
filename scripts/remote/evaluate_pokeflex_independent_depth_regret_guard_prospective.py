#!/usr/bin/env python3
"""Evaluate the frozen PokeFlex D405 regret guard on prospective takes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repository_root() / "src"))

from bayesian_phystwin.pokeflex_independent_depth_regret_guard import (  # noqa: E402
    evaluate_pokeflex_regret_guard_prospective,
)
from bayesian_phystwin.pokeflex_independent_depth_regret_guard_protocol import (  # noqa: E402
    load_pokeflex_regret_guard_prospective_protocol,
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
    parser.add_argument("--source-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            _repository_root()
            / "configs"
            / "sota"
            / "pokeflex_independent_depth_regret_guard_prospective_v1.json"
        ),
    )
    args = parser.parse_args()
    protocol = load_pokeflex_regret_guard_prospective_protocol(
        args.protocol.resolve()
    )
    source_path = args.source_evaluation.resolve()
    expected_source_hash = protocol["payload"]["source_evidence"]["result_sha256"]
    if _sha256(source_path) != expected_source_hash:
        raise ValueError("source evaluation checksum changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    paths = [path.resolve() for path in args.artifacts]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for payload in payloads:
        registration = payload.get("prospective_regret_guard", {})
        if registration.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise ValueError("candidate artifact prospective protocol changed")
    result = evaluate_pokeflex_regret_guard_prospective(
        payloads,
        source,
        expected_take_ids=protocol["take_ids"],
    )
    gates = protocol["payload"]["evaluation"]
    object_regression = max(
        0.0,
        max(-float(value["relative_improvement"]) for value in result["objects"]),
    )
    result["gate_checks"] = {
        "object_balanced_improvement": (
            result["object_balanced_relative_improvement"]
            >= float(gates["minimum_object_balanced_relative_improvement"])
        ),
        "object_wins": result["object_wins"] >= int(gates["minimum_object_wins"]),
        "maximum_object_regression": (
            object_regression <= float(gates["maximum_object_regression"])
        ),
    }
    result["gate_passed"] = all(result["gate_checks"].values())
    result["protocol_sha256"] = protocol["protocol_sha256"]
    result["source_evaluation_sha256"] = expected_source_hash
    result["candidate_artifacts"] = [
        {"path": str(path), "sha256": _sha256(path)} for path in paths
    ]
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = args.output.resolve()
    if output.exists() and output.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"existing prospective evaluation differs: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "gate_passed": result["gate_passed"],
                "relative_improvement": result[
                    "object_balanced_relative_improvement"
                ],
                "object_wins": result["object_wins"],
                "object_losses": result["object_losses"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
