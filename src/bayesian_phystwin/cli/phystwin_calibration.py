"""CLI for split-conformal and NEES evaluation of PhysTwin anchors."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_calibration import run_phystwin_calibration_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Bayesian PhysTwin anchor calibration on a locked cohort."
    )
    parser.add_argument("data_root")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--anchor-run-dir",
        help="Existing Bayesian-anchor run whose operational future posterior is audited.",
    )
    args = parser.parse_args()
    summary = run_phystwin_calibration_audit(
        args.data_root,
        args.output_dir,
        anchor_run_dir=args.anchor_run_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
