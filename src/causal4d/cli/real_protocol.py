"""Generate and validate the Causal4D multi-action real protocol."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from causal4d.real_protocol import (
    build_same_object_real_protocol,
    load_protocol,
    scaffold_dataset,
    validate_dataset,
    validate_protocol,
    write_acquisition_schedule,
    write_protocol,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="write the deterministic preregistered protocol",
    )
    generate.add_argument("output_json")
    generate.add_argument(
        "--schedule-csv",
        help="also write the locked operator acquisition order",
    )

    validate = subparsers.add_parser(
        "validate-protocol",
        help="validate design balance, hashes, and split boundaries",
    )
    validate.add_argument("protocol_json")

    scaffold = subparsers.add_parser(
        "scaffold",
        help="create non-overwriting session and execution templates",
    )
    scaffold.add_argument("protocol_json")
    scaffold.add_argument("output_root")

    dataset = subparsers.add_parser(
        "validate-dataset",
        help="validate a completed 36-execution acquisition tree",
    )
    dataset.add_argument("protocol_json")
    dataset.add_argument("dataset_root")
    dataset.add_argument(
        "--skip-file-hashes",
        action="store_true",
        help="validate descriptors without rehashing recorded files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            protocol = build_same_object_real_protocol()
            output = write_protocol(args.output_json, protocol)
            result = {**validate_protocol(protocol), "output": str(output.resolve())}
            if args.schedule_csv:
                schedule = write_acquisition_schedule(args.schedule_csv, protocol)
                result["schedule_csv"] = str(schedule.resolve())
        elif args.command == "validate-protocol":
            result = validate_protocol(load_protocol(args.protocol_json))
        elif args.command == "scaffold":
            result = scaffold_dataset(
                load_protocol(args.protocol_json), args.output_root
            )
        else:
            result = validate_dataset(
                load_protocol(args.protocol_json),
                args.dataset_root,
                verify_files=not args.skip_file_hashes,
            )
    except (OSError, KeyError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
