"""Source-only feasibility run for the anisotropic Tracking Cloth probe bank.

This development study was registered after the first shake-to-twist target
scoring and therefore cannot create fresh twist evidence.  It may read all shake
outcomes and outcome-blind twist inputs (prefix plus prescribed driven corners),
but it never reads a twist free-marker outcome.  Its sole scientific purpose is
to decide whether the richer physical bank creates nontrivial query-specific
probe choices worth freezing for a future untouched query.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .active_probe_cli import fit_fold
from .active_probe_run import validate_protocol
from .anisotropic_model_v2 import parameter_bank, predict
from .data import (
    audit_dataset,
    digest,
    infer_source_scale,
    input_view,
    object_digest,
    read_prefix,
    scoring_view,
    write_json,
)

HERE = Path(__file__).resolve().parent
PROTOCOL_FILE = HERE / "active_probe_anisotropic_protocol_v2.json"
PRIOR_TARGET_RUN_ID = "33302686759"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_record(args):
    case, protocol, scale = args
    inputs = input_view(case, protocol, scale)
    prediction = predict(inputs, protocol)
    return case, prediction, scoring_view(case, inputs)


def _target_input(args):
    case, protocol, scale = args
    inputs = input_view(case, protocol, scale)
    return case, predict(inputs, protocol)


def _map(function, tasks: list[Any], workers: int) -> list[Any]:
    if workers == 1:
        return [function(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, tasks))


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing an empty source-only result table")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _selection_summary(specimens: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    rows = []
    task_counts = {action: 0 for action in protocol["probe_conditions"]}
    parameter_counts = {action: 0 for action in protocol["probe_conditions"]}
    fixed_counts = {action: 0 for action in protocol["probe_conditions"]}
    divergence = 0
    for specimen, state in sorted(specimens.items()):
        policies = state["policy_states"]
        task = policies["task_directed"]["1"]
        parameter = policies["parameter_information"]["1"]
        fixed = policies["fixed_order"]["1"]
        task_action = task["selected_actions"][0]
        parameter_action = parameter["selected_actions"][0]
        fixed_action = fixed["selected_actions"][0]
        task_counts[task_action] += 1
        parameter_counts[parameter_action] += 1
        fixed_counts[fixed_action] += 1
        differs = task_action != parameter_action
        divergence += int(differs)
        task_step = task["steps"][-1]
        parameter_step = parameter["steps"][-1]
        rows.append(
            {
                "specimen": specimen,
                "task_directed_k1": task_action,
                "parameter_information_k1": parameter_action,
                "fixed_order_k1": fixed_action,
                "task_vs_parameter_disagree": differs,
                "task_entropy_after": task_step["entropy_after"],
                "parameter_entropy_after": parameter_step["entropy_after"],
                "task_target_spread_after_m2": task_step[
                    "target_model_spread_after"
                ],
                "parameter_target_spread_after_m2": parameter_step[
                    "target_model_spread_after"
                ],
            }
        )
    count = len(rows)
    return {
        "rows": rows,
        "specimen_count": count,
        "task_vs_parameter_disagreement_count": divergence,
        "task_vs_parameter_disagreement_fraction": divergence / count,
        "task_choice_counts": task_counts,
        "parameter_choice_counts": parameter_counts,
        "fixed_choice_counts": fixed_counts,
        "mechanism_gate": {
            "policies_make_different_k1_choices": divergence > 0,
            "minimum_useful_divergence_reached": divergence
            >= int(protocol["minimum_policy_disagreements_for_followup"]),
        },
    }


def run(root: Path, output: Path, workers: int) -> None:
    protocol = json.loads(PROTOCOL_FILE.read_text())
    validate_protocol(protocol)
    if protocol["model_family"] != "anisotropic-spring-v2":
        raise ValueError("unexpected model family")
    bank = parameter_bank(protocol)
    root = root.resolve(strict=True)
    output = output.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("output and dataset must be disjoint directory trees")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    write_json(
        output / "run_manifest.json",
        {
            "created_at": now(),
            "protocol_id": object_digest(protocol),
            "implementation_sha256": {
                "anisotropic_model_v2.py": digest(HERE / "anisotropic_model_v2.py"),
                "active_probe_anisotropic_source_v2.py": digest(
                    HERE / "active_probe_anisotropic_source_v2.py"
                ),
                "active_probe.py": digest(HERE / "active_probe.py"),
                "active_probe_cli.py": digest(HERE / "active_probe_cli.py"),
                "active_probe_run.py": digest(HERE / "active_probe_run.py"),
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "github_sha": os.environ.get("GITHUB_SHA"),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "prior_target_run_id": PRIOR_TARGET_RUN_ID,
            "prior_twist_target_outcome_exposure": True,
            "twist_free_marker_outcomes_read": False,
            "paper_claim_authorized": False,
            "evidence_class": protocol["evidence_class"],
        },
    )
    cases, inventory = audit_dataset(root, protocol)
    write_json(output / "dataset_manifest.json", inventory)
    (output / "DATA_LICENSE.txt").write_text(inventory["included_license_text"])

    source_cases = [case for case in cases if case.motion == protocol["source_motion"]]
    target_cases = [case for case in cases if case.motion == protocol["target_motion"]]
    if len(source_cases) != 32 or len(target_cases) != 32:
        raise ValueError("complete 32-shake/32-twist-input roster is required")
    scales = [
        infer_source_scale(case, read_prefix(case, protocol["prefix_seconds"])[1])
        for case in source_cases
    ]
    if len(set(scales)) != 1:
        raise ValueError("source recordings disagree about metric coordinate units")
    scale = scales[0]
    source_records = _map(
        _source_record,
        [(case, protocol, scale) for case in source_cases],
        workers,
    )
    target_inputs = _map(
        _target_input,
        [(case, protocol, scale) for case in target_cases],
        workers,
    )

    folds: dict[str, Any] = {}
    specimens: dict[str, Any] = {}
    for held_material in protocol["materials"]:
        fold, held_specimens = fit_fold(
            held_material, source_records, target_inputs, protocol
        )
        folds[held_material] = fold
        if set(specimens) & set(held_specimens):
            raise ValueError("duplicate held specimen")
        specimens.update(held_specimens)
    if len(folds) != 4 or len(specimens) != 8:
        raise ValueError("incomplete leave-one-material-out source roster")

    selection = _selection_summary(specimens, protocol)
    source_fit = {
        "created_at": now(),
        "protocol_id": object_digest(protocol),
        "inventory_id": inventory["inventory_id"],
        "coordinate_scale_to_m": scale,
        "model_count": len(bank),
        "model_parameters": [member.as_tuple() for member in bank],
        "folds": folds,
        "specimens": specimens,
        "selection": selection,
        "shake_outcomes_read": 32,
        "twist_input_templates_read": 32,
        "twist_free_marker_outcomes_read": False,
        "target_scoring_authorized": False,
        "paper_claim_authorized": False,
    }
    write_json(output / "source_fit.json", source_fit)
    write_json(
        output / "source_summary.json",
        {
            "study_id": protocol["study_id"],
            "model_family": protocol["model_family"],
            "model_count": len(bank),
            **{key: value for key, value in selection.items() if key != "rows"},
            "decision_rule": (
                "freeze this bank for an untouched downstream query only if the "
                "registered minimum K=1 policy-divergence gate passes; otherwise "
                "retain this as a negative source-only development result"
            ),
            "twist_target_claim_authorized": False,
            "paper_claim_authorized": False,
        },
    )
    _save_csv(output / "probe_selections.csv", selection["rows"])

    report = [
        "# Tracking Cloth anisotropic active-probe source feasibility v2",
        "",
        f"Model family: `{protocol['model_family']}` with {len(bank)} frozen members.",
        "",
        "This run uses all 32 Shake outcomes and outcome-blind Twist inputs from",
        "the leave-one-material-out source folds. It does not read any Twist",
        "free-marker outcome and cannot alter or refresh the already exposed",
        "shake-to-twist result.",
        "",
        "## K=1 policy feasibility",
        "",
        (
            "Task-directed and parameter-information selections differ on "
            f"{selection['task_vs_parameter_disagreement_count']}/8 specimens "
            f"({100 * selection['task_vs_parameter_disagreement_fraction']:.1f}%)."
        ),
        "",
        f"Task choices: `{selection['task_choice_counts']}`.",
        f"Parameter-information choices: `{selection['parameter_choice_counts']}`.",
        "",
        "The follow-up gate is purely source-side: no target score is generated.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("source",), default="source")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    try:
        run(args.dataset_root, args.output, args.workers)
    except Exception as exc:
        if args.output.is_dir():
            write_json(
                args.output / "failure.json",
                {
                    "failed_at": now(),
                    "stage": "source",
                    "exception": type(exc).__name__,
                    "message": str(exc),
                    "twist_free_marker_outcomes_read": False,
                    "scientific_decision": "incomplete; no claim",
                },
            )
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
