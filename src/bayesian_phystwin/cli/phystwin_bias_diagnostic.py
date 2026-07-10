"""CLI for the PhysTwin simulator-residual bias diagnostic."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_bias_diagnostic import (
    PhysTwinBiasDiagnosticConfig,
    diagnose_phystwin_bias_forecast,
    write_bias_diagnostic,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test whether simulator residual bias improves manual tracks."
    )
    parser.add_argument("final_data")
    parser.add_argument("baseline_trajectory")
    parser.add_argument("gt_track_3d")
    parser.add_argument("output_json")
    parser.add_argument("--fit-end-frame", type=int, required=True)
    parser.add_argument("--train-end-frame", type=int, required=True)
    args = parser.parse_args()
    summary = diagnose_phystwin_bias_forecast(
        args.final_data,
        args.baseline_trajectory,
        args.gt_track_3d,
        config=PhysTwinBiasDiagnosticConfig(
            fit_end_frame=args.fit_end_frame,
            train_end_frame=args.train_end_frame,
        ),
    )
    write_bias_diagnostic(summary, args.output_json)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
