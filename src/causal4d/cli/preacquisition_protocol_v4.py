"""Generate and validate the Causal4D pre-acquisition v4 addendum."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from causal4d.preacquisition_protocol import load_preacquisition_amendment
from causal4d.preacquisition_protocol_v3 import load_preacquisition_v3
from causal4d.preacquisition_protocol_v4 import (
    build_preacquisition_v4,
    load_preacquisition_v4,
    validate_preacquisition_v4,
    write_preacquisition_v4,
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
        subparser.add_argument("gate_control_json")
        subparser.add_argument("v4_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = load_protocol(args.base_protocol_json)
        v2 = load_preacquisition_amendment(args.v2_json, protocol)
        v3 = load_preacquisition_v3(args.v3_json, protocol, v2)
        gate_control = json.loads(
            Path(args.gate_control_json).read_text(encoding="utf-8")
        )
        if args.command == "generate":
            v4 = build_preacquisition_v4(protocol, v2, v3, gate_control)
            output = write_preacquisition_v4(args.v4_json, v4)
            result = {
                **validate_preacquisition_v4(v4, protocol, v2, v3, gate_control),
                "output": str(output.resolve()),
            }
        else:
            v4 = load_preacquisition_v4(args.v4_json, protocol, v2, v3, gate_control)
            result = validate_preacquisition_v4(v4, protocol, v2, v3, gate_control)
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
