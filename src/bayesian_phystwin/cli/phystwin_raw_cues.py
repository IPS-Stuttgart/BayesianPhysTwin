"""CLI for recovering independent PhysTwin camera reliability cues."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_raw_cues import (
    PhysTwinRawCueConfig,
    build_phystwin_raw_camera_cues,
    write_raw_cue_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover raw camera visibility and segmentation-boundary cues."
    )
    parser.add_argument("final_data")
    parser.add_argument("raw_case_dir")
    parser.add_argument("output_npz")
    parser.add_argument("--base-cues")
    parser.add_argument("--summary-json")
    parser.add_argument("--initial-match-tolerance-m", type=float, default=1e-6)
    args = parser.parse_args()
    summary = build_phystwin_raw_camera_cues(
        args.final_data,
        args.raw_case_dir,
        args.output_npz,
        config=PhysTwinRawCueConfig(
            initial_match_tolerance_m=args.initial_match_tolerance_m,
        ),
        base_cues_path=args.base_cues,
    )
    if args.summary_json:
        write_raw_cue_summary(summary, args.summary_json)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
