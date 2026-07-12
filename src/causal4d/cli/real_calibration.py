"""Fit and evaluate source-only real Causal4D variance calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_residual_dynamics import _target_validity
from causal4d.contracts import PhysicalPosterior, load_contract
from causal4d.real_calibration import (
    RealCalibrationCase,
    case_from_physical_posterior,
    evaluate_real_calibration_case,
    fit_affine_variance_calibration,
    load_affine_variance_calibration,
    save_affine_variance_calibration,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_case(specification: dict) -> RealCalibrationCase:
    final_data_path = Path(specification["final_data"])
    with final_data_path.open("rb") as handle:
        data = pickle.load(handle)
    observed = np.asarray(data["object_points"], dtype=float)
    valid = _target_validity(
        np.asarray(data["object_visibilities"], dtype=bool),
        np.asarray(data["object_motions_valid"], dtype=bool),
    )
    labels = None
    if specification.get("node_group_labels"):
        labels = json.loads(
            Path(specification["node_group_labels"]).read_text(encoding="utf-8")
        )
    if specification.get("moments_npz"):
        method = str(specification.get("method", "graph_persistence"))
        with np.load(specification["moments_npz"], allow_pickle=False) as archive:
            descriptor = json.loads(str(archive["descriptor_json"]))
            if descriptor.get("schema_version") != 1:
                raise ValueError("unsupported real moments schema")
            if method not in descriptor["methods"]:
                raise ValueError(f"moments artifact has no method {method!r}")
            mean = np.asarray(archive[f"{method}_mean_m"], dtype=float)
            variance = np.asarray(archive[f"{method}_variance_m2"], dtype=float)
        endpoint = int(descriptor["endpoint_frame"])
        node_count = mean.shape[1]
        case = RealCalibrationCase(
            case_id=str(descriptor["case_id"]),
            action_id=str(specification.get("action_id", descriptor["action_id"])),
            contact_region_id=str(
                specification.get("contact_region_id", "unregistered")
            ),
            mean_m=mean,
            variance_m2=variance,
            truth_m=observed[endpoint:, :node_count],
            valid=valid[endpoint:, :node_count],
            start_frame=int(
                specification.get("start_frame", descriptor["start_frame"])
            ),
            node_group_labels=None if labels is None else tuple(labels),
        )
    else:
        posterior_path = Path(specification["physical_posterior"])
        artifact = load_contract(posterior_path)
        if not isinstance(artifact, PhysicalPosterior):
            raise TypeError("physical_posterior must contain a PhysicalPosterior")
        endpoint = artifact.context.o_minus.frame_stop - 1
        node_count = artifact.readout_trajectories_m.shape[2]
        case = case_from_physical_posterior(
            artifact,
            observed[endpoint:, :node_count],
            valid[endpoint:, :node_count],
            start_frame=int(specification.get("start_frame", 7)),
            action_id=specification.get("action_id"),
            contact_region_id=str(
                specification.get("contact_region_id", "unregistered")
            ),
            node_group_labels=labels,
        )
    expected_case = specification.get("case_id")
    if expected_case is not None and expected_case != case.case_id:
        raise ValueError("case_id differs from the PhysicalPosterior context")
    return case


def _load_manifest(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported real calibration manifest schema")
    return payload


def _aggregate_evaluations(evaluations: list[dict]) -> dict:
    groups: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)
    for evaluation in evaluations:
        groups[("action", evaluation["action_id"])].append(evaluation)
        groups[("contact", evaluation["contact_region_id"])].append(evaluation)
    by_factor = []
    for (factor, value), selected in sorted(groups.items()):
        by_factor.append(
            {
                "factor": factor,
                "value": value,
                "execution_count": len(selected),
                "raw_coverage": float(
                    np.mean([item["raw"]["all"]["coverage"] for item in selected])
                ),
                "calibrated_coverage": float(
                    np.mean(
                        [item["calibrated"]["all"]["coverage"] for item in selected]
                    )
                ),
                "raw_nll": float(
                    np.mean([item["raw"]["all"]["gaussian_nll"] for item in selected])
                ),
                "calibrated_nll": float(
                    np.mean(
                        [item["calibrated"]["all"]["gaussian_nll"] for item in selected]
                    )
                ),
            }
        )
    return {
        "by_action_and_contact": by_factor,
        "worst_group_coverage": {
            "raw": min(item["worst_group_coverage"]["raw"] for item in evaluations),
            "calibrated": min(
                item["worst_group_coverage"]["calibrated"] for item in evaluations
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("source_manifest_json")
    fit.add_argument("output_calibration_json")
    fit.add_argument("--confidence-level", type=float, default=0.90)
    fit.add_argument("--minimum-calibration-trials", type=int, default=10)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("calibration_json")
    evaluate.add_argument("target_manifest_json")
    evaluate.add_argument("output_evaluation_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fit":
        manifest = _load_manifest(args.source_manifest_json)
        fit_cases = tuple(_load_case(value) for value in manifest.get("fit", []))
        calibration_cases = tuple(
            _load_case(value) for value in manifest.get("calibration", [])
        )
        calibration, diagnostics = fit_affine_variance_calibration(
            fit_cases,
            calibration_cases,
            confidence_level=args.confidence_level,
            minimum_calibration_trials=args.minimum_calibration_trials,
        )
        diagnostics["source_manifest"] = {
            "path": str(Path(args.source_manifest_json).resolve()),
            "sha256": _sha256(args.source_manifest_json),
        }
        save_affine_variance_calibration(
            args.output_calibration_json,
            calibration,
            diagnostics,
        )
        result = {
            "calibration_id": calibration.calibration_id,
            "claim_ready": calibration.claim_ready,
            "output": str(Path(args.output_calibration_json).resolve()),
            **diagnostics,
        }
    else:
        calibration = load_affine_variance_calibration(args.calibration_json)
        manifest = _load_manifest(args.target_manifest_json)
        target_specs = manifest.get("target", manifest.get("cases", []))
        cases = [_load_case(value) for value in target_specs]
        if not cases:
            raise ValueError("target manifest contains no cases")
        overlap = {case.case_id for case in cases} & (
            set(calibration.fit_case_ids) | set(calibration.calibration_case_ids)
        )
        if overlap:
            raise ValueError(
                "target cases overlap source calibration: " + ", ".join(sorted(overlap))
            )
        evaluations = [
            evaluate_real_calibration_case(case, calibration) for case in cases
        ]
        result = {
            "schema_version": 1,
            "evaluation": "causal4d_source_only_affine_calibration_v1",
            "calibration_id": calibration.calibration_id,
            "calibration_claim_ready": calibration.claim_ready,
            "target_labels_used_for_calibration": False,
            "target_manifest": {
                "path": str(Path(args.target_manifest_json).resolve()),
                "sha256": _sha256(args.target_manifest_json),
            },
            "cases": evaluations,
            "aggregate": _aggregate_evaluations(evaluations),
        }
        output = Path(args.output_evaluation_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        result = {
            "calibration_id": calibration.calibration_id,
            "calibration_claim_ready": calibration.claim_ready,
            "case_count": len(cases),
            "output": str(output.resolve()),
            "worst_group_coverage": result["aggregate"]["worst_group_coverage"],
        }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
