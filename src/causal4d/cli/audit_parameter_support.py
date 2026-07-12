"""Audit predictive convergence under reduced physical-parameter support."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_residual_dynamics import _target_validity
from causal4d.contracts import TwinBelief, load_contract
from causal4d.parameter_support_audit import (
    ParameterSupportAuditConfig,
    audit_parameter_support,
)
from causal4d.phystwin_backend import load_rollout_bank


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flat_rows(result: dict) -> list[dict[str, object]]:
    rows = []
    for candidate in result["candidates"]:
        predictive = candidate["predictive"]
        row = {
            "method": candidate["method"],
            "requested_count": candidate["requested_count"],
            "count": candidate["count"],
            "directly_retained_probability_mass": candidate[
                "directly_retained_probability_mass"
            ],
            "represented_probability_mass": candidate["represented_probability_mass"],
            "prior_effective_support": candidate["effective_support"],
            "posterior_joint_effective_support": candidate[
                "posterior_joint_effective_support"
            ],
            "posterior_parameter_effective_support": candidate[
                "posterior_parameter_effective_support"
            ],
            "parameter_mean_error_l2": candidate["parameter_mean_error_l2"],
            "parameter_covariance_error_frobenius": candidate[
                "parameter_covariance_error_frobenius"
            ],
            "predictive_mean_rmse_vs_full_m": candidate[
                "predictive_mean_rmse_vs_full_m"
            ],
            "predictive_variance_relative_l2_vs_full": candidate[
                "predictive_variance_relative_l2_vs_full"
            ],
            "label_free_stable_vs_full": candidate["label_free_stable_vs_full"],
            "coordinate_rmse_m": predictive["coordinate_rmse_m"],
            "track_error_m": predictive["track_error_m"],
            "coverage": predictive["coverage"],
            "nees": predictive["nees"],
            "gaussian_nll": predictive["gaussian_nll"],
            "gaussian_energy_score_m": predictive["gaussian_energy_score_m"],
            "mean_interval_width_m": predictive["mean_interval_width_m"],
            "early_coverage": predictive["by_horizon"]["early"]["coverage"],
            "middle_coverage": predictive["by_horizon"]["middle"]["coverage"],
            "late_coverage": predictive["by_horizon"]["late"]["coverage"],
            "runtime_seconds": candidate["runtime_seconds"],
        }
        rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_rollout_bank_npz")
    parser.add_argument("full_twin_belief_npz")
    parser.add_argument("final_data_pickle")
    parser.add_argument("output_json")
    parser.add_argument("output_csv")
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=(4, 8, 16, 32, 81),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("top_mass", "weighted_coreset"),
        default=("top_mass", "weighted_coreset"),
    )
    parser.add_argument("--o-plus-prefix-frames", type=int, default=6)
    parser.add_argument("--confidence-level", type=float, default=0.90)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bank, manifest = load_rollout_bank(args.full_rollout_bank_npz)
    artifact = load_contract(args.full_twin_belief_npz)
    if not isinstance(artifact, TwinBelief):
        raise TypeError("full_twin_belief_npz must contain a TwinBelief")
    with Path(args.final_data_pickle).open("rb") as handle:
        data = pickle.load(handle)
    observed = np.asarray(data["object_points"], dtype=float)
    valid = _target_validity(
        np.asarray(data["object_visibilities"], dtype=bool),
        np.asarray(data["object_motions_valid"], dtype=bool),
    )
    endpoint = artifact.endpoint_frame
    result = audit_parameter_support(
        bank,
        artifact,
        observed[endpoint:, : bank.node_count],
        valid[endpoint:, : bank.node_count],
        config=ParameterSupportAuditConfig(
            counts=tuple(args.counts),
            methods=tuple(args.methods),
            prefix_frame_count=args.o_plus_prefix_frames + 1,
            confidence_level=args.confidence_level,
        ),
    )
    result.update(
        {
            "case": artifact.context.case_id,
            "inputs": {
                "full_rollout_bank": {
                    "path": str(Path(args.full_rollout_bank_npz).resolve()),
                    "sha256": _sha256(args.full_rollout_bank_npz),
                },
                "full_twin_belief": {
                    "path": str(Path(args.full_twin_belief_npz).resolve()),
                    "sha256": _sha256(args.full_twin_belief_npz),
                    "artifact_id": artifact.artifact_id,
                },
                "final_data": {
                    "path": str(Path(args.final_data_pickle).resolve()),
                    "sha256": _sha256(args.final_data_pickle),
                },
            },
            "rollout_bank_manifest": manifest,
        }
    )
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows = _flat_rows(result)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "case": result["case"],
                "full_support_count": result["full_support_count"],
                "stable_counts": result["stable_counts"],
                "output_json": str(output_json.resolve()),
                "output_csv": str(output_csv.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
