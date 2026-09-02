#!/usr/bin/env python3
"""Controlled study of target-directed versus full-cause intervention cost."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.target_directed_intervention_design_v1 import (
    InterventionDesignStatus,
    TargetDirectedInterventionDesignV1,
)

SHA = "a" * 64


def canonical_id(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def design(target: np.ndarray) -> TargetDirectedInterventionDesignV1:
    return TargetDirectedInterventionDesignV1(
        source_design_id=SHA,
        target_query_id=SHA,
        candidate_roster_id=SHA,
        source_design=np.asarray([[1.0, 1.0, 0.0]]),
        target_map=target,
        candidate_intervention_ids={
            "material-probe": SHA,
            "redundant-probe": SHA,
            "state-gauge-probe": SHA,
        },
        candidate_designs={
            "material-probe": np.asarray([[0.0, 0.0, 1.0]]),
            "redundant-probe": np.asarray([[2.0, 2.0, 0.0]]),
            "state-gauge-probe": np.asarray([[1.0, -1.0, 0.0]]),
        },
        intervention_costs={
            "material-probe": 1.0,
            "redundant-probe": 0.25,
            "state-gauge-probe": 1.0,
        },
    )


def run() -> dict[str, Any]:
    targets = {
        "source-visible-sum": np.asarray([[1.0, 1.0, 0.0]]),
        "state-gauge-difference": np.asarray([[1.0, -1.0, 0.0]]),
        "material-effect": np.asarray([[0.0, 0.0, 1.0]]),
    }
    records: dict[str, Any] = {}
    target_costs: list[float] = []
    full_costs: list[float] = []
    for target_id, target_map in targets.items():
        certificate = design(target_map)
        if certificate.selected_total_cost is None:
            raise RuntimeError(f"target {target_id} was not identified")
        if certificate.minimum_full_cause_identification_cost is None:
            raise RuntimeError("full cause identification must be feasible")
        target_costs.append(certificate.selected_total_cost)
        full_costs.append(certificate.minimum_full_cause_identification_cost)
        records[target_id] = {
            "status": certificate.status.value,
            "selected_interventions": list(certificate.selected_interventions),
            "selected_total_cost": certificate.selected_total_cost,
            "minimum_full_cause_interventions": list(
                certificate.minimum_full_cause_interventions or ()
            ),
            "minimum_full_cause_identification_cost": (
                certificate.minimum_full_cause_identification_cost
            ),
            "cost_saving_vs_full_cause_identification": (
                certificate.cost_saving_vs_full_cause_identification
            ),
            "source_target_identifiable_dimension": (
                certificate.source_target_identifiable_dimension
            ),
            "maximum_target_identifiable_dimension": (
                certificate.maximum_target_identifiable_dimension
            ),
            "artifact_id": certificate.artifact_id,
        }

    target_mean = float(np.mean(target_costs))
    full_mean = float(np.mean(full_costs))
    metrics = {
        "target_directed_mean_intervention_cost": target_mean,
        "full_cause_mean_intervention_cost": full_mean,
        "absolute_mean_cost_reduction": full_mean - target_mean,
        "relative_mean_cost_reduction": 1.0 - target_mean / full_mean,
        "targets_identified": sum(
            record["status"]
            in {
                InterventionDesignStatus.ALREADY_IDENTIFIABLE.value,
                InterventionDesignStatus.TARGET_IDENTIFIED.value,
            }
            for record in records.values()
        ),
        "targets_total": len(records),
        "zero_probe_targets": sum(
            not record["selected_interventions"]
            for record in records.values()
        ),
        "one_probe_targets": sum(
            len(record["selected_interventions"]) == 1
            for record in records.values()
        ),
    }
    checks = {
        "all_targets_identified": (
            metrics["targets_identified"] == metrics["targets_total"]
        ),
        "source_visible_query_requires_no_probe": (
            records["source-visible-sum"]["selected_interventions"] == []
        ),
        "difference_query_selects_only_state_gauge_probe": (
            records["state-gauge-difference"]["selected_interventions"]
            == ["state-gauge-probe"]
        ),
        "material_query_selects_only_material_probe": (
            records["material-effect"]["selected_interventions"]
            == ["material-probe"]
        ),
        "full_cause_identification_requires_both_informative_probes": all(
            set(record["minimum_full_cause_interventions"])
            == {"material-probe", "state-gauge-probe"}
            for record in records.values()
        ),
        "target_design_reduces_mean_cost_by_at_least_two_thirds": (
            metrics["relative_mean_cost_reduction"] >= 2.0 / 3.0 - 1e-12
        ),
        "redundant_probe_is_never_selected": all(
            "redundant-probe" not in record["selected_interventions"]
            for record in records.values()
        ),
    }
    result: dict[str, Any] = {
        "schema": "bayesian-phystwin.target-directed-intervention-controlled.v1",
        "source_design": [[1.0, 1.0, 0.0]],
        "cause_coordinates": ["state", "gauge", "material"],
        "targets": records,
        "metrics": metrics,
        "checks": checks,
        "decision": (
            "target-identification-cost-strictly-below-full-cause-identification"
            if all(checks.values())
            else "target-directed-intervention-advantage-not-established"
        ),
        "claim_boundary": (
            "Controlled exact finite-roster linear mechanism evidence only. The "
            "study does not establish that a modeled probe is physically valid, "
            "safe, or beneficial on real data, and it does not establish natural "
            "cause labels or nonlinear closure."
        ),
    }
    result["result_id"] = canonical_id(result)
    return result


def report(result: dict[str, Any]) -> str:
    lines = [
        "# Target-directed intervention controlled result",
        "",
        f"**Decision:** `{result['decision']}`",
        "",
        "| Target | Selected probes | Target cost | Full-cause cost | Saving |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for target, record in result["targets"].items():
        selected = ", ".join(record["selected_interventions"]) or "none"
        lines.append(
            f"| `{target}` | {selected} | "
            f"{record['selected_total_cost']:.3f} | "
            f"{record['minimum_full_cause_identification_cost']:.3f} | "
            f"{record['cost_saving_vs_full_cause_identification']:.3f} |"
        )
    lines += ["", "## Aggregate", ""]
    for key, value in result["metrics"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Frozen checks", ""]
    for key, value in result["checks"].items():
        lines.append(f"- `{key}`: **{'pass' if value else 'fail'}**")
    lines += ["", "## Claim boundary", "", result["claim_boundary"], ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(report(result), encoding="utf-8")
    print(
        json.dumps(
            {"decision": result["decision"], "result_id": result["result_id"]}
        )
    )
    return 0 if all(result["checks"].values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
