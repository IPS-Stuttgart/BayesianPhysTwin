"""CLI for aggregating all locked rest-geometry Warp transfer records."""

from __future__ import annotations

import argparse
import json

from causal4d.rest_geometry_cross_action import (
    write_rest_geometry_protocol_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and aggregate all 66 factual/same-grasp/new-contact "
            "rest-geometry Warp records under the frozen fold locks."
        )
    )
    parser.add_argument("protocol")
    parser.add_argument("selection_dir")
    parser.add_argument("result_record_dir")
    parser.add_argument("output")
    args = parser.parse_args()
    result = write_rest_geometry_protocol_result(
        args.protocol,
        args.selection_dir,
        args.result_record_dir,
        args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
