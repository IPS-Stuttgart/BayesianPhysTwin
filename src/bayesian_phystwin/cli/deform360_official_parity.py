"""CLI for the fail-closed Deform360 official-parity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_official_parity import (
    audit_parity_contract,
    build_public_parity_audit,
    write_parity_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether a Deform360 3D evaluator contract supports an official "
            "comparison. Without a contract, emit the strongest public-evidence "
            "audit and candidate-convention sensitivity report."
        )
    )
    parser.add_argument("output_json")
    parser.add_argument("--contract")
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="exit with status 2 when the audited contract is not parity-ready",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.contract:
        contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
        result = audit_parity_contract(contract)
    else:
        result = build_public_parity_audit()
    write_parity_json(result, args.output_json)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_ready:
        ready = bool(result.get("parity_ready"))
        if result.get("artifact_kind") == "Deform360Official3DParityPublicAudit":
            ready = all(
                bool(audit["parity_ready"])
                for audit in result["audits"].values()
            )
        if not ready:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
