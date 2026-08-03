#!/usr/bin/env python3
"""Evaluate and freeze the PokeFlex baseline-relative regret certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _repository_root()
sys.path.insert(0, str(ROOT / "src"))

from bayesian_phystwin.bias_aware_belief import (  # noqa: E402
    SourceRegretCertificate,
)
from bayesian_phystwin.pokeflex_baseline_relative_guard import (  # noqa: E402
    FEATURE_NAMES,
    PokeFlexBaselineRelativeGuardConfig,
    baseline_relative_guard_decision,
    certificate_to_payload,
    decision_audit,
    fit_baseline_relative_guard_certificate,
    leave_one_physical_object_out_decisions,
    summarize_guard_decisions,
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked_json_sha256(path: Path) -> str:
    """Hash the canonical Git blob while checking checkout-equivalent JSON."""

    relative = path.resolve().relative_to(ROOT).as_posix()
    completed = subprocess.run(
        ["git", "cat-file", "blob", f"HEAD:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    if json.loads(completed.stdout) != _load(path):
        raise ValueError(f"tracked evidence differs semantically: {relative}")
    return hashlib.sha256(completed.stdout).hexdigest()


def _source_inventory(
    rows_artifact: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    counts = rows_artifact["take_target_frame_counts"]
    return {
        str(take_id): {
            "object": str(take_id).rpartition("_T")[0],
            "frame_count": int(frame_count),
        }
        for take_id, frame_count in counts.items()
    }


def _public_inventory(
    rows_artifact: Mapping[str, Any], target_result: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    inventory = {
        str(row["take_id"]): {
            "object": str(row["object_name"]),
            "frame_count": int(row["scored_frame_count"]),
        }
        for row in target_result["objects"]
    }
    for take_id, frame_count in rows_artifact["take_target_frame_counts"].items():
        if str(take_id) not in inventory:
            raise ValueError(f"raw-row take is missing from target result: {take_id}")
        if int(frame_count) != inventory[str(take_id)]["frame_count"]:
            raise ValueError(f"target-frame count changed for {take_id}")
    return inventory


def _deployment_decisions(
    rows: list[dict[str, Any]], certificate: SourceRegretCertificate
) -> list[dict[str, Any]]:
    decisions = []
    for row in rows:
        decision = baseline_relative_guard_decision(certificate, row["features"])
        decisions.append(
            {
                **{
                    name: row[name]
                    for name in (
                        "domain",
                        "object",
                        "take_id",
                        "target_frame",
                        "take_target_frame_count",
                        "regret_mm",
                    )
                },
                **decision,
                "selected_regret_mm": (
                    float(row["regret_mm"]) if decision["accepted"] else 0.0
                ),
            }
        )
    return decisions


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_rows_artifact = _load(args.source_rows)
    public_rows_artifact = _load(args.public_rows)
    alpha_control = _load(args.alpha_control)
    source_result = _load(args.source_result)
    public_result = _load(args.public_result)
    rows = list(source_rows_artifact["rows"]) + list(public_rows_artifact["rows"])
    config = PokeFlexBaselineRelativeGuardConfig()

    source_baselines = {
        str(name): float(value["baseline_mean_CD_UL1_mm"])
        for name, value in source_result["selected_result"]["objects"].items()
    }
    public_baselines = {
        str(row["object_name"]): float(row["baseline_mean_CD_UL1_mm"])
        for row in public_result["objects"]
    }
    source_inventory = _source_inventory(source_rows_artifact)
    public_inventory = _public_inventory(public_rows_artifact, public_result)

    cross_fitted = leave_one_physical_object_out_decisions(rows, config=config)
    certificate = fit_baseline_relative_guard_certificate(rows, config=config)
    deployed = _deployment_decisions(rows, certificate)

    loo_source = summarize_guard_decisions(
        cross_fitted,
        domain="source",
        object_baseline_mm=source_baselines,
        take_inventory=source_inventory,
    )
    loo_public = summarize_guard_decisions(
        cross_fitted,
        domain="public_paired_v1",
        object_baseline_mm=public_baselines,
        take_inventory=public_inventory,
    )
    deploy_source = summarize_guard_decisions(
        deployed,
        domain="source",
        object_baseline_mm=source_baselines,
        take_inventory=source_inventory,
    )
    deploy_public = summarize_guard_decisions(
        deployed,
        domain="public_paired_v1",
        object_baseline_mm=public_baselines,
        take_inventory=public_inventory,
    )
    loo_audit = decision_audit(cross_fitted)
    gates = {
        "leave_one_object_out_source_all_win": (
            loo_source["object_wins"] == loo_source["object_count"]
        ),
        "leave_one_object_out_public_no_loss": loo_public["object_losses"] == 0,
        "leave_one_object_out_public_minimum_ten_wins": (
            loo_public["object_wins"] >= 10
        ),
        "leave_one_object_out_false_safe_rate_at_most_ten_percent": (
            loo_audit["false_safe_rate"] <= 0.10
        ),
        "leave_one_object_out_upper_coverage_at_least_eighty_five_percent": (
            loo_audit["upper_coverage"] >= 0.85
        ),
        "deployment_public_minimum_twelve_wins": deploy_public["object_wins"] >= 12,
        "deployment_public_no_loss": deploy_public["object_losses"] == 0,
        "deployment_public_positive_improvement": (
            deploy_public["object_balanced_relative_improvement"] > 0.0
        ),
    }
    return {
        "schema_version": 1,
        "artifact_kind": "PokeFlexBaselineRelativeGuardDevelopmentEvaluation",
        "claim_status": (
            "post-open source plus public-paired-v1 development; fresh outcomes required"
        ),
        "feature_names": list(FEATURE_NAMES),
        "config": config.as_dict(),
        "evidence": {
            "source_rows_sha256": _sha256(args.source_rows),
            "public_paired_raw_rows_sha256": _sha256(args.public_rows),
            "source_result_sha256": _tracked_json_sha256(args.source_result),
            "public_paired_result_sha256": _tracked_json_sha256(
                args.public_result
            ),
            "alpha_scale_control_sha256": _sha256(args.alpha_control),
        },
        "global_scale_control": {
            "claim_status": alpha_control["claim_status"],
            "summary": alpha_control["summary"],
            "maximum_alpha_one_reproduction_error_mm": max(
                float(row["alpha1_reproduction_abs_mm"])
                for row in alpha_control["objects"]
            ),
            "interpretation": (
                "Every nonzero global scale retains one losing object; a global "
                "amplitude reduction cannot satisfy the no-regression gate."
            ),
        },
        "leave_one_physical_object_out": {
            "audit": loo_audit,
            "source": loo_source,
            "public_paired_v1": loo_public,
        },
        "deployment_fit": {
            "audit": decision_audit(deployed),
            "source": deploy_source,
            "public_paired_v1": deploy_public,
            "certificate": certificate_to_payload(certificate),
        },
        "development_gates": gates,
        "development_gate_passed": all(gates.values()),
        "claim_boundary": (
            "All outcomes are opened development evidence. The certificate is "
            "eligible only for a separately locked fresh-take evaluation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rows", type=Path, required=True)
    parser.add_argument("--public-rows", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--public-result", type=Path, required=True)
    parser.add_argument("--alpha-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(
        json.dumps(
            {
                "development_gate_passed": result["development_gate_passed"],
                "leave_one_physical_object_out": result[
                    "leave_one_physical_object_out"
                ],
                "deployment_fit": result["deployment_fit"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
