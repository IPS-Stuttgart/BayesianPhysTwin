"""Run the staged Deform360 001-rope contact experiment."""

from __future__ import annotations

import argparse
import json

from causal4d_public.deform360 import load_deform360_protocol_config
from causal4d_public.deform360_contact import (
    evaluate_target_contact_oracle,
    fit_contact_model,
    load_contact_artifact,
    seal_target_contact_predictions,
    validate_contact_artifact,
    write_contact_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit", help="Fit from source; validate on calibration.")
    fit.add_argument("raw_object_dir")
    fit.add_argument("processed_root")
    fit.add_argument("output_json")
    fit.add_argument("--config", required=True)

    seal = subparsers.add_parser(
        "seal", help="Seal target visual and tactile-prefix predictions."
    )
    seal.add_argument("processed_root")
    seal.add_argument("contact_model_json")
    seal.add_argument("output_json")
    seal.add_argument("--config", required=True)

    evaluate = subparsers.add_parser(
        "evaluate", help="Open the target tactile oracle after future predictions seal."
    )
    evaluate.add_argument("processed_root")
    evaluate.add_argument("contact_model_json")
    evaluate.add_argument("contact_prediction_seal_json")
    evaluate.add_argument("output_json")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--held-out-prediction-seal-sha256", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        config = load_deform360_protocol_config(args.config)
        if args.command == "fit":
            result = fit_contact_model(args.raw_object_dir, args.processed_root, config)
            kind = "Deform360ContactModel"
        elif args.command == "seal":
            model = load_contact_artifact(
                args.contact_model_json, expected_kind="Deform360ContactModel"
            )
            result = seal_target_contact_predictions(args.processed_root, config, model)
            kind = "Deform360TargetContactPredictionSeal"
        else:
            model = load_contact_artifact(
                args.contact_model_json, expected_kind="Deform360ContactModel"
            )
            prediction = load_contact_artifact(
                args.contact_prediction_seal_json,
                expected_kind="Deform360TargetContactPredictionSeal",
            )
            result = evaluate_target_contact_oracle(
                args.processed_root,
                config,
                model,
                prediction,
                held_out_prediction_seal_sha256=(args.held_out_prediction_seal_sha256),
            )
            kind = "Deform360TargetContactOracleEvaluation"
        write_contact_artifact(args.output_json, result)
        validation = validate_contact_artifact(result, expected_kind=kind)
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(
        json.dumps(
            {**validation, "output": args.output_json},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
