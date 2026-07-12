"""Generate or validate the physical contact-registration artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from causal4d.contact_registration import (
    build_contact_registration_template,
    validate_contact_registration,
    write_contact_registration,
)
from causal4d.real_protocol import load_protocol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    template = subparsers.add_parser("template")
    template.add_argument("protocol_json")
    template.add_argument("output_json")
    template.add_argument("--camera-id", action="append", required=True)
    template.add_argument("--object-node-count", type=int, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("protocol_json")
    validate.add_argument("registration_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = load_protocol(args.protocol_json)
        if args.command == "template":
            artifact = build_contact_registration_template(
                protocol,
                camera_ids=args.camera_id,
                object_node_count=args.object_node_count,
            )
            output = write_contact_registration(args.output_json, artifact)
            result = {
                "passed": True,
                "status": "template",
                "output": str(output.resolve()),
            }
        else:
            with open(args.registration_json, encoding="utf-8") as handle:
                artifact = json.load(handle)
            result = validate_contact_registration(artifact, protocol)
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
