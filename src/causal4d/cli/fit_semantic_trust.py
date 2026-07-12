"""Fit semantic beta and OOD thresholds on source validation cases."""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_residual_dynamics import _target_validity
from causal4d.contracts import PhysicalPosterior, load_contract
from causal4d.molmo_acceptance import (
    gate_beta_candidates,
    load_molmo_acceptance_result,
)
from causal4d.molmo_adapter import load_molmo_forecasts
from causal4d.semantic_posterior import molmo_task_evidence
from causal4d.semantic_trust import (
    SemanticValidationCase,
    fit_semantic_trust_calibration,
    save_semantic_trust_calibration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select semantic trust using source validation futures only."
    )
    parser.add_argument("source_manifest_json")
    parser.add_argument("output_calibration_json")
    parser.add_argument("--betas", default="0,1,3,6,12")
    parser.add_argument(
        "--molmo-acceptance-json",
        help="required to unlock positive beta candidates; rejection forces beta=0",
    )
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.0)
    parser.add_argument("--support-margin", type=float, default=1.5)
    return parser


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = Path(args.source_manifest_json)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("cases", [])
    if not entries:
        raise ValueError("source manifest must contain nonempty cases")
    source_cases = []
    for entry in entries:
        physical_artifact = load_contract(
            _resolve(manifest_path.parent, entry["physical_posterior"])
        )
        if not isinstance(physical_artifact, PhysicalPosterior):
            raise TypeError("source physical_posterior must be a PhysicalPosterior")
        bundle = load_molmo_forecasts(
            _resolve(manifest_path.parent, entry["molmo_forecast"])
        )
        evidence = molmo_task_evidence(
            bundle,
            str(entry["forecast_id"]),
            physical_artifact,
            scale_m=float(entry.get("scale_m", 0.10)),
            degrees_of_freedom=float(entry.get("degrees_of_freedom", 3.0)),
        )
        with _resolve(manifest_path.parent, entry["final_data_pickle"]).open(
            "rb"
        ) as handle:
            data = pickle.load(handle)
        observed = np.asarray(data["object_points"], dtype=float)
        valid = _target_validity(
            np.asarray(data["object_visibilities"], dtype=bool),
            np.asarray(data["object_motions_valid"], dtype=bool),
        )
        endpoint = physical_artifact.context.o_minus.frame_stop - 1
        state_count = physical_artifact.readout_trajectories_m.shape[2]
        source_cases.append(
            SemanticValidationCase(
                case_id=str(entry["case_id"]),
                physical=physical_artifact,
                evidence=evidence,
                truth_m=observed[endpoint:, :state_count],
                mask=valid[endpoint:, :state_count],
                start_frame=int(entry.get("start_frame", 1)),
            )
        )
    requested_betas = tuple(float(value) for value in args.betas.split(",") if value)
    acceptance = (
        load_molmo_acceptance_result(args.molmo_acceptance_json)
        if args.molmo_acceptance_json
        else None
    )
    betas = gate_beta_candidates(requested_betas, acceptance)
    calibration = fit_semantic_trust_calibration(
        source_cases,
        beta_candidates=betas,
        minimum_relative_improvement=args.minimum_relative_improvement,
        support_margin=args.support_margin,
    )
    save_semantic_trust_calibration(args.output_calibration_json, calibration)
    print(
        json.dumps(
            {
                "calibration_id": calibration.calibration_id,
                "output": str(Path(args.output_calibration_json).resolve()),
                "selected_beta": calibration.selected_beta,
                "requested_beta_candidates": list(requested_betas),
                "evaluated_beta_candidates": list(betas),
                "molmo_acceptance_passed": bool(
                    acceptance
                    and acceptance["decision"][
                        "accepted_for_semantic_reweighting"
                    ]
                ),
                "source_case_ids": list(calibration.source_case_ids),
                "source_mean_rmse_m": dict(
                    zip(
                        map(str, calibration.beta_candidates),
                        calibration.source_mean_rmse_m,
                        strict=True,
                    )
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
