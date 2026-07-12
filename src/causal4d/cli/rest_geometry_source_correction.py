"""CLI for exporting a persistent source correction for target Warp transfer."""

from __future__ import annotations

import argparse
import json

from causal4d.real_protocol import load_protocol
from causal4d.rest_geometry_transfer import (
    write_source_rest_geometry_correction,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export the persistent frame/material correction from a locked "
            "source execution without carrying source holdout observations."
        )
    )
    parser.add_argument("protocol")
    parser.add_argument("source_execution_id")
    parser.add_argument("summary")
    parser.add_argument("archive")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    result = write_source_rest_geometry_correction(
        load_protocol(args.protocol),
        args.source_execution_id,
        args.summary,
        args.archive,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
