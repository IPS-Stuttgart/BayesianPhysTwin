"""Predict, validate, or score the locked released-particle Warp source audit."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from causal4d_public.deform360_released_warp_readout import (
    load_released_warp_readout_protocol,
)
from causal4d_public.deform360_released_warp_readout_execution import (
    load_released_warp_prediction_artifact,
    run_released_warp_readout_predictions,
    score_released_warp_readout_predictions,
    validate_released_warp_prediction_artifact,
    validate_released_warp_score_artifact,
    write_released_warp_prediction_artifact,
    write_released_warp_score_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--protocol", required=True)
    predict.add_argument("--official-repo", required=True)
    predict.add_argument("--source-observation-dir", required=True)
    predict.add_argument("--released-object-root", required=True)
    predict.add_argument("--output-json", required=True)
    predict.add_argument("--output-archive", required=True)
    predict.add_argument("--device", default="cuda:0")

    validate = subparsers.add_parser("validate-prediction")
    validate.add_argument("--protocol", required=True)
    validate.add_argument("--prediction-json", required=True)

    score = subparsers.add_parser("score")
    score.add_argument("--protocol", required=True)
    score.add_argument("--prediction-json", required=True)
    score.add_argument("--released-object-root", required=True)
    score.add_argument("--output-json", required=True)
    return parser


def _predict(args: argparse.Namespace) -> dict[str, object]:
    payload = run_released_warp_readout_predictions(
        args.protocol,
        official_repo=args.official_repo,
        source_observation_root=args.source_observation_dir,
        released_object_root=args.released_object_root,
        output_archive_path=args.output_archive,
        device=args.device,
    )
    output = write_released_warp_prediction_artifact(args.output_json, payload)
    protocol = load_released_warp_readout_protocol(args.protocol)
    stored = load_released_warp_prediction_artifact(
        output,
        protocol=protocol,
    )
    return {
        **validate_released_warp_prediction_artifact(
            stored,
            protocol=protocol,
            artifact_directory=output.parent,
        ),
        "output": str(output.resolve()),
    }


def _validate_prediction(args: argparse.Namespace) -> dict[str, object]:
    protocol = load_released_warp_readout_protocol(args.protocol)
    prediction_path = Path(args.prediction_json)
    payload = load_released_warp_prediction_artifact(
        prediction_path,
        protocol=protocol,
    )
    return validate_released_warp_prediction_artifact(
        payload,
        protocol=protocol,
        artifact_directory=prediction_path.parent,
    )


def _score(args: argparse.Namespace) -> dict[str, object]:
    payload = score_released_warp_readout_predictions(
        args.protocol,
        args.prediction_json,
        released_object_root=args.released_object_root,
    )
    output = write_released_warp_score_artifact(args.output_json, payload)
    stored = json.loads(output.read_text(encoding="utf-8"))
    protocol = load_released_warp_readout_protocol(args.protocol)
    return {
        **validate_released_warp_score_artifact(stored, protocol=protocol),
        "output": str(output.resolve()),
        "panel": stored["panel"],
        "transfer_gate": stored["transfer_gate"],
    }


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "predict":
            result = _predict(args)
        elif args.command == "validate-prediction":
            result = _validate_prediction(args)
        else:
            result = _score(args)
    except (
        OSError,
        RuntimeError,
        KeyError,
        TypeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "score" and not result["transfer_gate_passed"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
