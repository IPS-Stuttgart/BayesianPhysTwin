"""CLI for the absolute 22-case PhysTwin benchmark comparison."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_sota_comparison import (
    aggregate_phystwin_sota_comparison,
)


def _method(value: str) -> tuple[str, str]:
    name, separator, template = value.partition("=")
    if not separator or not name or "{case}" not in template:
        raise argparse.ArgumentTypeError(
            "methods must use NAME=/path/containing/{case}/trajectory.pkl"
        )
    return name, template


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate absolute PhysTwin metrics over 22 and 19 cases."
    )
    parser.add_argument("data_root")
    parser.add_argument("output_json")
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        type=_method,
        metavar="NAME=PATH_TEMPLATE",
    )
    args = parser.parse_args()
    methods = dict(args.method)
    if len(methods) != len(args.method):
        parser.error("method names must be unique")
    result = aggregate_phystwin_sota_comparison(
        args.data_root,
        methods,
        args.output_json,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
