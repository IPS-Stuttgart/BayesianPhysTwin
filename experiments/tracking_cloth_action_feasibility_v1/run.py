"""Source-only physical-action feasibility audit on Tracking Cloth.

The audit numerically opens repetitions 1 and 2 only. Repetition 3 remains
reserved. Every material/repetition block contains all three physically executed
self-collision release configurations, which permits a complete source action
loss vector without model-generated counterfactual outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tracking_cloth_action_feasibility_v1._data import source_rows
from experiments.tracking_cloth_action_feasibility_v1._decision import (
    _probe_binary_outcomes,
    decision_grid,
)
from experiments.tracking_cloth_action_feasibility_v1._metrics import (
    object_digest,
    read_protocol,
)

HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE / "protocol.json"


def _block_matrices(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> tuple[list[tuple[str, int]], np.ndarray, np.ndarray]:
    actions = list(protocol["interactions"])
    blocks = [
        (material, repetition)
        for material in protocol["materials"]
        for repetition in protocol["source_repetitions"]
    ]
    by_key = {
        (
            str(row["material"]),
            int(row["repetition"]),
            str(row["interaction"]),
        ): row
        for row in rows
    }
    losses = np.asarray(
        [
            [
                float(by_key[(material, repetition, action)]["task_loss"])
                for action in actions
            ]
            for material, repetition in blocks
        ],
        dtype=np.float64,
    )
    probe_features = np.asarray(
        [
            [
                float(by_key[(material, repetition, action)]["probe_feature"])
                for material, repetition in blocks
            ]
            for action in actions
        ],
        dtype=np.float64,
    )
    return blocks, losses, probe_features


def build_result(
    root: Path,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = read_protocol(protocol_path)
    rows, inventory = source_rows(root, protocol)
    blocks, losses, probe_features = _block_matrices(rows, protocol)
    probe_outcomes, thresholds = _probe_binary_outcomes(probe_features)
    grid, decision_summary = decision_grid(blocks, losses, probe_outcomes, protocol)

    actions = list(protocol["interactions"])
    best_action = np.argmin(losses, axis=1)
    best_action_names = [actions[int(index)] for index in best_action]
    by_material: dict[str, list[str]] = {}
    for (material, _), action in zip(blocks, best_action_names, strict=True):
        by_material.setdefault(material, []).append(action)
    stable_material_count = sum(
        len(set(values)) == 1 for values in by_material.values()
    )
    unique_best_actions = sorted(set(best_action_names))
    selected = decision_summary["selected_source_setting"]
    has_sensing_setting = (
        "mode_counts" in selected and selected["mode_counts"]["sense"] > 0
    )

    protocol_unsigned = dict(protocol)
    protocol_id = object_digest(protocol_unsigned)
    result: dict[str, Any] = {
        "schema": "bayesian-phystwin.tracking-cloth-action-feasibility-result.v1",
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "protocol_id": protocol_id,
        "dataset_inventory_id": inventory["inventory_id"],
        "dataset_record": protocol["dataset_record"],
        "source_block_count": len(blocks),
        "source_case_count": len(rows),
        "action_names": actions,
        "source_blocks": [
            {"material": material, "repetition": repetition}
            for material, repetition in blocks
        ],
        "action_loss_matrix": losses.tolist(),
        "best_action_by_block": best_action_names,
        "best_actions_by_material": by_material,
        "unique_best_actions": unique_best_actions,
        "stable_best_action_material_count": stable_material_count,
        "probe_feature": protocol["probe_feature"],
        "probe_threshold_by_interaction": {
            action: threshold
            for action, threshold in zip(actions, thresholds, strict=True)
        },
        "probe_outcome_by_interaction_and_block": probe_outcomes.tolist(),
        "decision_grid": grid,
        "decision_summary": decision_summary,
        "feasibility": {
            "nontrivial_action_choice": len(unique_best_actions) >= 2,
            "stable_source_action_for_at_least_two_materials": (
                stable_material_count >= 2
            ),
            "source_setting_uses_decision_probe": has_sensing_setting,
            "candidate_for_separately_reviewed_rep3_protocol": bool(
                len(unique_best_actions) >= 2
                and stable_material_count >= 2
                and has_sensing_setting
            ),
        },
        "information_boundary": protocol["information_boundary"],
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = object_digest(result)
    return result, rows


def write_outputs(
    output: Path,
    result: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output / "source_cases.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    feasibility = result["feasibility"]
    summary = [
        "# Tracking Cloth physical-action source feasibility",
        "",
        f"- source blocks: `{result['source_block_count']}`",
        f"- source recordings: `{result['source_case_count']}`",
        (
            "- unique source-optimal physical actions: "
            f"`{len(result['unique_best_actions'])}`"
        ),
        f"- stable materials: `{result['stable_best_action_material_count']}/4`",
        (
            "- source setting uses a decision probe: "
            f"`{feasibility['source_setting_uses_decision_probe']}`"
        ),
        (
            "- candidate for a separately reviewed rep3 protocol: "
            f"`{feasibility['candidate_for_separately_reviewed_rep3_protocol']}`"
        ),
        "- rep3 numerical outcomes read: `false`",
        "",
        (
            "This is source-only feasibility evidence, not a held-out "
            "action-choice result."
        ),
    ]
    (output / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    result, rows = build_result(args.dataset_root, args.protocol)
    write_outputs(args.output, result, rows)
    print(
        json.dumps(
            {
                "result_id": result["result_id"],
                "feasibility": result["feasibility"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
