"""CLI for opening futures after a MatPhys target-prefix family gate."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.matphys_part_family_gate import (
    open_matphys_part_family_future,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root")
    parser.add_argument("output_dir")
    parser.add_argument("candidate_manifest")
    parser.add_argument("gate_summary")
    args = parser.parse_args()
    result = open_matphys_part_family_future(
        args.data_root,
        args.output_dir,
        args.candidate_manifest,
        args.gate_summary,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
