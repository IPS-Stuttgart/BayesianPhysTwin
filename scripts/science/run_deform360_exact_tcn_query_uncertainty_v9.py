#!/usr/bin/env python3
"""Reproduce the best Deform360 TCN mean and evaluate causal query uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import deform360_exact_tcn_capture_v9 as capture
import deform360_tcn_query_uncertainty_v9 as uncertainty
import numpy as np


SCHEMA = "bayesian-phystwin/deform360-exact-tcn-query-uncertainty-result-v9"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def make_report(result: Mapping[str, Any]) -> str:
    study = result["uncertainty_study"]
    lines = [
        "# Exact action-TCN mean with causal query uncertainty v9",
        "",
        f"- Exact retained TCN reproduction: **{result['exact_tcn_reproduction']['passed']}**",
        f"- Action-conditioned TCN active-field RMSE: **{result['point_prediction']['active_field_rmse']:.8f}**",
        f"- Query cases / physical objects: **{study['case_count']} / {study['object_count']}**",
        f"- Classification: **{study['classification']}**",
        "",
        "| Uncertainty arm | NLL/dim | nANEES | Ellipsoid coverage | Marginal coverage | Width |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in study["metrics"].items():
        lines.append(
            f"| `{name}` | {metrics['nll_per_dimension']:.7f} | "
            f"{metrics['normalized_anees']:.7f} | "
            f"{metrics['ellipsoid_coverage']:.4f} | "
            f"{metrics['marginal_coverage']:.4f} | "
            f"{metrics['mean_marginal_width']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen gates",
            "",
        ]
    )
    for name, passed in study["gates"].items():
        lines.append(f"- `{name}`: **{'pass' if passed else 'fail'}**")
    lines.extend(
        [
            "",
            "Every point-prediction arm uses the exact retained action-conditioned TCN",
            "mean. Candidate uncertainty parameters are fitted on other physical objects,",
            "and online scale updates use only outcomes whose forecast horizon has already",
            "matured before the current forecast. This is retrospective grouped public-data",
            "evidence, not globally fresh confirmation or an authorized deployment claim.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    with np.load(args.query_source_npz, allow_pickle=False) as source:
        query_matrix = np.asarray(source["query_matrix"], dtype=np.float64)
    capture_manifest, arrays = capture.reproduce_and_capture(
        args.tcn_artifact_root,
        args.exact_tcn_root,
        args.data_root,
        query_matrix,
        args.dropout_draws,
    )
    capture_path = output / "exact-tcn-query-capture-v9.npz"
    np.savez_compressed(capture_path, **arrays)
    capture_manifest["statistics_file"] = capture_path.name
    capture_manifest["statistics_sha256"] = sha256_file(capture_path)
    capture_manifest["result_sha256"] = capture.canonical_digest(
        {
            key: value
            for key, value in capture_manifest.items()
            if key != "result_sha256"
        }
    )
    write_json(output / "capture-manifest.json", capture_manifest)

    study = uncertainty.study(
        arrays["query_residuals"],
        arrays["dropout_query_covariances"],
        arrays["object_ids"],
        arrays["window_indices"],
        fold_count=args.fold_count,
        maturity_lag_windows=args.maturity_lag_windows,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 9,
        "status": "completed",
        "classification": study["classification"],
        "exact_tcn_reproduction": {
            "passed": capture_manifest[
                "exact_scientific_result_reproduction"
            ],
            "maximum_absolute_numeric_difference": capture_manifest[
                "maximum_absolute_numeric_difference"
            ],
            "execution_revision": capture_manifest["tcn_execution_revision"],
            "retained_result_sha256": capture_manifest["tcn_result_sha256"],
            "carrier_audit": capture_manifest["carrier_audit"],
        },
        "point_prediction": {
            "method": "exact retained action-conditioned TCN",
            "active_field_rmse": capture_manifest[
                "action_conditioned_tcn_active_field_rmse"
            ],
            "predictive_mean_changed": False,
            "final_model_freeze": capture_manifest["final_model_freeze"],
        },
        "query_contract": {
            "family": "first twelve orthonormal DCT-II modes of the common 384-dimensional tactile field",
            "dimension": int(query_matrix.shape[0]),
            "field_dimension": int(query_matrix.shape[1]),
            "query_matrix_sha256": hashlib.sha256(
                np.ascontiguousarray(query_matrix).tobytes()
            ).hexdigest(),
        },
        "uncertainty_study": study,
        "information_boundary": {
            "public_data_only": True,
            "new_measurements_collected": False,
            "point_mean_retuned": False,
            "target_object_influences_own_crossfit_calibrator": False,
            "online_update_uses_current_or_future_outcome": False,
            "globally_fresh_confirmation": False,
            "post_confirmation_grouped_development": True,
        },
        "claim_boundary": (
            "Retrospective grouped public-data evidence for uncertainty around an exact "
            "retained action-conditioned TCN mean. A positive gate supports causal online "
            "query calibration on known objects, not zero-shot object transfer, strict "
            "counterfactual identification, calibrated dense 4-D geometry, deployment "
            "safety, or globally fresh confirmation."
        ),
        "paper_claim_authorized": False,
        "strict_counterfactual_claim_authorized": False,
        "globally_fresh_confirmation_authorized": False,
    }
    result["result_sha256"] = uncertainty.canonical_digest(result)
    write_json(output / "result.json", result)
    (output / "report.md").write_text(make_report(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcn-artifact-root", type=Path)
    parser.add_argument("--exact-tcn-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--query-source-npz", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dropout-draws", type=int, default=32)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--maturity-lag-windows", type=int, default=4)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        capture.self_test()
        uncertainty.self_test()
        return 0
    required = (
        args.tcn_artifact_root,
        args.exact_tcn_root,
        args.data_root,
        args.query_source_npz,
        args.output_root,
    )
    if any(value is None for value in required):
        parser.error("all data and output arguments are required unless --self-test is used")
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "classification": result["classification"],
                "result_sha256": result["result_sha256"],
                "gates": result["uncertainty_study"]["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
