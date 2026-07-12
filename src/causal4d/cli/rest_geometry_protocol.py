"""CLI for the locked rest-geometry same-object validation boundary."""

from __future__ import annotations

import argparse
import json

from causal4d.rest_geometry_protocol import (
    write_rest_geometry_protocol_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit same-object acquisition readiness and write the locked "
            "rest-geometry fold plan."
        )
    )
    parser.add_argument("dataset_root")
    parser.add_argument("output_dir")
    parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args()
    result = write_rest_geometry_protocol_artifacts(
        args.dataset_root,
        args.output_dir,
        verify_files=args.verify_files,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
