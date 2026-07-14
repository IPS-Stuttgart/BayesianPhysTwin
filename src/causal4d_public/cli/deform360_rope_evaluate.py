"""Evaluate sealed Deform360 rope predictions on the post-seal target future."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causal4d_public.deform360_rope_evaluation import (
    evaluate_held_out_rope_predictions,
)
from causal4d_public.deform360_rope_future import (
    validate_target_future_rope_geometry,
)
from causal4d_public.deform360_rope_prefix import validate_target_prefix_rope_geometry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("held_out_prediction_seal_json")
    parser.add_argument("prefix_geometry_json")
    parser.add_argument("future_geometry_json")
    parser.add_argument("oracle_prediction_json")
    parser.add_argument("output_json")
    args = parser.parse_args()
    try:
        prediction_seal = json.loads(
            Path(args.held_out_prediction_seal_json).read_text(encoding="utf-8")
        )
        prefix_geometry = json.loads(
            Path(args.prefix_geometry_json).read_text(encoding="utf-8")
        )
        future_geometry = json.loads(
            Path(args.future_geometry_json).read_text(encoding="utf-8")
        )
        oracle_prediction = json.loads(
            Path(args.oracle_prediction_json).read_text(encoding="utf-8")
        )
        validate_target_prefix_rope_geometry(prefix_geometry)
        validate_target_future_rope_geometry(future_geometry)
        with np.load(future_geometry["archive"]["path"], allow_pickle=False) as stored:
            reference = np.asarray(stored["centerlines_m"], dtype=np.float64)
        with np.load(prefix_geometry["archive"]["path"], allow_pickle=False) as stored:
            prefix_endpoint = np.asarray(stored["centerlines_m"][-1], dtype=np.float64)
        persistence = np.repeat(prefix_endpoint[None], len(reference), axis=0)
        result = evaluate_held_out_rope_predictions(
            reference,
            held_out_prediction_seal=prediction_seal,
            target_future_geometry_sha256=future_geometry["result_sha256"],
            oracle_prediction=oracle_prediction,
            additional_predictions={"constant_persistence": persistence},
        )
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    summary = {
        name: {
            metric: values[metric]["mean_m"]
            for metric in ("chamfer_distance_m", "track_error_m")
        }
        for name, values in result["methods"].items()
    }
    print(
        json.dumps(
            {
                "passed": True,
                "result_sha256": result["result_sha256"],
                "methods": summary,
                "output": args.output_json,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
