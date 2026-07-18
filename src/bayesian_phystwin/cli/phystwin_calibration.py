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
    parser.add_argument(
        "--external-backbone-manifest",
        help="Hash-validated external trajectory manifest to audit instead of inference.pkl.",
    )
    parser.add_argument(
        "--external-overlay-dir",
        help="Matching external-backbone overlay whose operational anchor is audited.",
    )
    args = parser.parse_args()
    summary = run_phystwin_calibration_audit(
        args.data_root,
        args.output_dir,
        anchor_run_dir=args.anchor_run_dir,
        external_backbone_manifest=args.external_backbone_manifest,
        external_overlay_dir=args.external_overlay_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
