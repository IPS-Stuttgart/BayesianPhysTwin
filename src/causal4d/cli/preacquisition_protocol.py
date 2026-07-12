"""Generate and validate the Causal4D pre-acquisition amendment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from causal4d.preacquisition_analysis import audit_base_protocol_power
from causal4d.preacquisition_protocol import (
    build_preacquisition_amendment,
    load_preacquisition_amendment,
    validate_preacquisition_amendment,
    write_preacquisition_amendment,
)
from causal4d.real_protocol import load_protocol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("base_protocol_json")
    generate.add_argument("output_json")
    validate = subparsers.add_parser("validate")
    validate.add_argument("base_protocol_json")
    validate.add_argument("amendment_json")
    audit = subparsers.add_parser("audit-base")
    audit.add_argument("base_protocol_json")
    audit.add_argument("--output-json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = load_protocol(args.base_protocol_json)
        if args.command == "generate":
            amendment = build_preacquisition_amendment(protocol)
            output = write_preacquisition_amendment(args.output_json, amendment)
            result = {
                **validate_preacquisition_amendment(amendment, protocol),
                "output": str(output.resolve()),
            }
        elif args.command == "validate":
            amendment = load_preacquisition_amendment(args.amendment_json, protocol)
            result = validate_preacquisition_amendment(amendment, protocol)
        else:
            result = audit_base_protocol_power(protocol)
            if args.output_json:
                with open(args.output_json, "w", encoding="utf-8") as handle:
                    json.dump(result, handle, indent=2, sort_keys=True)
                    handle.write("\n")
    except (OSError, KeyError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
