"""CLI for the pre-acquisition structural protocol amendment."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.structural_protocol import (
    audit_structural_protocol_readiness,
    scaffold_structural_protocol_amendment,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol_json")
    parser.add_argument("dataset_root")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    result = (
        audit_structural_protocol_readiness(args.protocol_json, args.dataset_root)
        if args.audit_only
        else scaffold_structural_protocol_amendment(
            args.protocol_json, args.dataset_root
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
