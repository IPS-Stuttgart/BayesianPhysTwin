#!/usr/bin/env python3
"""Run the frozen TAPNext++ plus CoTracker3 source-provider diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.phystwin_tapnextpp_cotracker_complement import (
    EVALUATION_FILENAME,
    evaluate_complementary_prediction,
    seal_complementary_prediction,
    write_complementary_prediction,
)

QUERY_SHA256 = "8eb6f31c3908f65ddecd741eef32ad2f0fd4a3bac797fc91007f5610dd653039"
TAPNEXTPP_PREDICTION_SHA256 = (
    "d368814ecad07d425f43a612de2630b1a1be6ffc8e639ade5c13ee00bba678ce"
)
TAPNEXTPP_SEAL_SHA256 = (
    "a89dd9692719953996c800ac9cce90b1f974ea43de6a9270cb483a7def28c509"
)
COTRACKER_CUES_SHA256 = (
    "713fd1ac124c72f9835d848f8c2f6e3622667936c8b5626b42f744a8f2347d56"
)
WITHHELD_PREFIX_SHA256 = (
    "77f0a37b929bfc7e66020970a81cab1616078a566747ec511927e7841deaa143"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--query", type=Path, required=True)
    predict.add_argument("--tapnextpp-prediction", type=Path, required=True)
    predict.add_argument("--tapnextpp-seal", type=Path, required=True)
    predict.add_argument("--cotracker-cues", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)

    seal = subparsers.add_parser("seal")
    seal.add_argument("--prediction-dir", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--prediction-dir", type=Path, required=True)
    evaluate.add_argument("--withheld-prefix", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "predict":
        result = write_complementary_prediction(
            args.query,
            args.tapnextpp_prediction,
            args.tapnextpp_seal,
            args.cotracker_cues,
            args.output_dir,
            expected_query_sha256=QUERY_SHA256,
            expected_tapnextpp_prediction_sha256=TAPNEXTPP_PREDICTION_SHA256,
            expected_tapnextpp_seal_sha256=TAPNEXTPP_SEAL_SHA256,
            expected_cotracker_cues_sha256=COTRACKER_CUES_SHA256,
        )
    elif args.command == "seal":
        result = seal_complementary_prediction(args.prediction_dir)
    else:
        output = args.output or (
            args.prediction_dir.resolve() / EVALUATION_FILENAME
        )
        result = evaluate_complementary_prediction(
            args.prediction_dir,
            args.withheld_prefix,
            output,
            expected_withheld_sha256=WITHHELD_PREFIX_SHA256,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
