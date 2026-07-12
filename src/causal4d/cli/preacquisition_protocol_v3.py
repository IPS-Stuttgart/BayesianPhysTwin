"""Generate and validate the superseding Causal4D pre-acquisition v3 lock."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from causal4d.preacquisition_protocol import load_preacquisition_amendment
from causal4d.preacquisition_protocol_v3 import (
    build_preacquisition_v3,
    load_preacquisition_v3,
    validate_preacquisition_v3,
    write_preacquisition_v3,
)
from causal4d.real_protocol import load_protocol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "validate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("base_protocol_json")
        subparser.add_argument("v2_json")
        subparser.add_argument("v3_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = load_protocol(args.base_protocol_json)
        v2 = load_preacquisition_amendment(args.v2_json, protocol)
        if args.command == "generate":
            v3 = build_preacquisition_v3(protocol, v2)
            output = write_preacquisition_v3(args.v3_json, v3)
            result = {
                **validate_preacquisition_v3(v3, protocol, v2),
                "output": str(output.resolve()),
            }
        else:
            v3 = load_preacquisition_v3(args.v3_json, protocol, v2)
            result = validate_preacquisition_v3(v3, protocol, v2)
    except (OSError, KeyError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
