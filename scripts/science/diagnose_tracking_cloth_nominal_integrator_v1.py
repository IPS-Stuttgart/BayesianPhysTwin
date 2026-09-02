#!/usr/bin/env python3
"""Diagnose nominal contact-model stability using rep1 source trajectories only."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tracking_cloth_self_collision_selective_twin_v1.data import (
    audit_dataset,
    prediction_input,
    scoring_truth,
)
from experiments.tracking_cloth_self_collision_selective_twin_v1.model import (
    contact_rollout,
    trajectory_mse,
)

SUBSTEP_GRID = (8, 16, 32, 64, 128, 256)


def object_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(dataset_root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    if protocol.get("fit_repetition") != 1:
        raise ValueError("diagnostic requires rep1 as the source-fit repetition")
    if protocol.get("confirmation_repetition") != 3:
        raise ValueError("diagnostic requires rep3 to remain the confirmation repetition")
    if protocol["information_boundary"].get("paper_claim_authorized") is not False:
        raise ValueError("diagnostic protocol cannot authorize a paper claim")

    output.mkdir(parents=True, exist_ok=False)
    cases, inventory = audit_dataset(dataset_root, protocol)
    rep1 = [case for case in cases if case.repetition == 1]
    if len(rep1) != 12:
        raise ValueError("complete 12-case rep1 source roster is required")

    nominal = tuple(float(value) for value in protocol["nominal_parameters"])
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, int], np.ndarray] = {}
    for case in rep1:
        inputs = prediction_input(case, protocol)
        truth = scoring_truth(case, inputs)
        origin = inputs.cloth_prefix[0].mean(axis=0)
        for substeps in SUBSTEP_GRID:
            local = copy.deepcopy(protocol)
            local["integration_substeps"] = substeps
            row: dict[str, Any] = {
                "case_id": case.case_id,
                "material": case.material,
                "interaction": case.interaction,
                "repetition": case.repetition,
                "substeps": substeps,
                "stable": False,
                "failure": "",
                "rmse_mm": "",
                "maximum_radius_m": "",
            }
            try:
                prediction = contact_rollout(inputs, nominal, local)
            except ValueError as error:
                row["failure"] = str(error)
            else:
                predictions[(case.case_id, substeps)] = prediction
                row["stable"] = True
                row["rmse_mm"] = 1000.0 * float(
                    np.sqrt(trajectory_mse(prediction, truth, inputs))
                )
                row["maximum_radius_m"] = float(
                    np.max(np.linalg.norm(prediction - origin, axis=2))
                )
            rows.append(row)

    stable_by_step = {
        substeps: all(
            bool(row["stable"])
            for row in rows
            if int(row["substeps"]) == substeps
        )
        for substeps in SUBSTEP_GRID
    }
    common = [step for step in SUBSTEP_GRID if stable_by_step[step]]
    selected = common[0] if common else None
    registered = int(protocol["integration_substeps"])
    registered_failures = sum(
        not bool(row["stable"])
        for row in rows
        if int(row["substeps"]) == registered
    )

    convergence_rows: list[dict[str, Any]] = []
    if selected is not None:
        selected_index = SUBSTEP_GRID.index(selected)
        if selected_index + 1 < len(SUBSTEP_GRID):
            reference = SUBSTEP_GRID[selected_index + 1]
            for case in rep1:
                left = predictions[(case.case_id, selected)]
                right = predictions.get((case.case_id, reference))
                if right is None:
                    continue
                difference = np.linalg.norm(left - right, axis=2)
                convergence_rows.append(
                    {
                        "case_id": case.case_id,
                        "selected_substeps": selected,
                        "reference_substeps": reference,
                        "rms_prediction_difference_mm": 1000.0
                        * float(np.sqrt(np.mean(difference**2))),
                        "maximum_prediction_difference_mm": 1000.0
                        * float(np.max(difference)),
                    }
                )

    result: dict[str, Any] = {
        "schema": "bayesian-phystwin.tracking-cloth-nominal-integrator-diagnostic.v1",
        "schema_version": 1,
        "inventory_id": inventory["inventory_id"],
        "rep1_case_count": len(rep1),
        "rep3_numeric_outcomes_read": False,
        "nominal_parameters": list(nominal),
        "registered_integration_substeps": registered,
        "substep_grid": list(SUBSTEP_GRID),
        "stable_by_substeps": {
            str(key): value for key, value in stable_by_step.items()
        },
        "registered_substep_failure_count": registered_failures,
        "minimum_common_stable_substeps": selected,
        "all_rep1_cases_stabilizable": selected is not None,
        "convergence_case_count": len(convergence_rows),
        "maximum_selected_to_reference_rms_difference_mm": (
            max(
                row["rms_prediction_difference_mm"]
                for row in convergence_rows
            )
            if convergence_rows
            else None
        ),
        "decision": (
            "nominal-integrator-stabilizable-on-rep1"
            if selected is not None
            else "nominal-model-not-stabilized-on-rep1"
        ),
        "claim_boundary": (
            "Source-only numerical diagnostic on the twelve rep1 Tracking Cloth "
            "self-collision trajectories. It may distinguish an explicit-step "
            "instability from failure to stabilize over the registered grid. It "
            "does not validate model fidelity, authorize rep3 access, select a "
            "paper claim, or permit target-side numerical tuning."
        ),
    }
    result["result_id"] = object_digest(result)

    write_json(output / "diagnostic.json", result)
    write_json(output / "dataset_manifest.json", inventory)
    write_json(output / "protocol.json", protocol)
    write_csv(output / "case_substep_results.csv", rows)
    if convergence_rows:
        write_csv(output / "convergence_results.csv", convergence_rows)
    report = [
        "# Tracking Cloth nominal-integrator source diagnostic",
        "",
        f"Decision: **`{result['decision']}`**",
        "",
        f"- Rep1 cases: {len(rep1)}",
        f"- Registered substeps: {registered}",
        f"- Registered failures: {registered_failures}/{len(rep1)}",
        f"- Minimum common stable substeps: {selected}",
        f"- Rep3 numeric outcomes read: **false**",
        "",
        "## Stability grid",
        "",
    ]
    report.extend(
        f"- `{step}` substeps: **{'stable' if stable_by_step[step] else 'unstable'}**"
        for step in SUBSTEP_GRID
    )
    if convergence_rows:
        report.extend(
            [
                "",
                "## Numerical-convergence diagnostic",
                "",
                "Maximum selected-to-next-grid RMS trajectory difference: "
                f"{result['maximum_selected_to_reference_rms_difference_mm']:.6f} mm",
            ]
        )
    report.extend(["", "## Claim boundary", "", result["claim_boundary"]])
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.dataset_root, args.protocol, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
