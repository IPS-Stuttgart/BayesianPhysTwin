"""Run the source-only cost-aware Tracking Cloth action audit V2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tracking_cloth_action_feasibility_costed_v2._decision import (
    decision_grid_v2,
)
from experiments.tracking_cloth_action_feasibility_v1._decision import (
    _probe_binary_outcomes,
)
from experiments.tracking_cloth_action_feasibility_v1._metrics import (
    object_digest,
    read_protocol as read_source_protocol,
)
from experiments.tracking_cloth_action_feasibility_v1.run import (
    _block_matrices,
    source_rows,
)

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_PROTOCOL = HERE / "protocol.json"


def read_v2_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("V2 protocol must be a JSON object")
    if (
        value.get("schema")
        != "bayesian-phystwin.tracking-cloth-action-feasibility-costed.v2"
        or value.get("schema_version") != 2
    ):
        raise ValueError("unexpected cost-aware source protocol")
    if value.get("source_repetitions") != [1, 2]:
        raise ValueError("V2 source repetitions must remain [1, 2]")
    if value.get("reserved_target_repetition") != 3:
        raise ValueError("V2 target must remain repetition 3")
    objective = value.get("objective")
    robustness = value.get("support_robustness")
    boundary = value.get("information_boundary")
    if not all(isinstance(item, dict) for item in (objective, robustness, boundary)):
        raise ValueError("V2 protocol sections must be JSON objects")
    if objective["sensing_total_loss"] != (
        "terminal-task-loss-plus-probe-cost-times-source-loss-scale"
    ):
        raise ValueError("V2 probe-cost accounting changed")
    grid = [float(item) for item in robustness["support_miss_probability_grid"]]
    if not grid or sorted(set(grid)) != grid or any(not 0.0 <= item <= 1.0 for item in grid):
        raise ValueError("support-miss grid must be sorted and lie in [0, 1]")
    if float(robustness["primary_support_miss_probability"]) not in grid:
        raise ValueError("primary epsilon must occur in the support-miss grid")
    if robustness.get("bound_is_assumed_not_estimated") is not True:
        raise ValueError("the unknown-support bound must remain explicitly assumed")
    if robustness.get("target_tuning") is not False:
        raise ValueError("target tuning must remain disabled")
    if boundary.get("rep3_numeric_outcomes_read") is not False:
        raise ValueError("source audit cannot read rep3 numeric outcomes")
    if boundary.get("rep3_protocol_authorized") is not False:
        raise ValueError("source audit cannot authorize rep3")
    if boundary.get("paper_claim_authorized") is not False:
        raise ValueError("source audit cannot authorize a paper claim")
    return value


def source_protocol_path(v2_protocol: dict[str, Any]) -> Path:
    relative = Path(str(v2_protocol["base_source_protocol_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("base source protocol path must be repository-relative")
    path = REPOSITORY_ROOT / relative
    if not path.is_file():
        raise ValueError("base source protocol does not exist")
    return path


def build_result(
    dataset_root: Path,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    v2_protocol = read_v2_protocol(protocol_path)
    source_path = source_protocol_path(v2_protocol)
    source_protocol = read_source_protocol(source_path)
    if source_protocol["source_repetitions"] != [1, 2]:
        raise ValueError("base source protocol opened a non-source repetition")
    if source_protocol["reserved_target_repetition"] != 3:
        raise ValueError("base source protocol target changed")

    rows, inventory = source_rows(dataset_root, source_protocol)
    if any(int(row["repetition"]) not in {1, 2} for row in rows):
        raise ValueError("source rows contain a reserved target repetition")
    blocks, losses, probe_features = _block_matrices(rows, source_protocol)
    probe_outcomes, thresholds = _probe_binary_outcomes(probe_features)
    grid, decision_summary = decision_grid_v2(
        blocks,
        losses,
        probe_outcomes,
        source_protocol,
        v2_protocol,
    )

    actions = list(source_protocol["interactions"])
    best_action_index = np.argmin(losses, axis=1)
    best_action_names = [actions[int(index)] for index in best_action_index]
    best_by_material: dict[str, list[str]] = {}
    for (material, _), action in zip(blocks, best_action_names, strict=True):
        best_by_material.setdefault(material, []).append(action)
    stable_material_count = sum(
        len(set(values)) == 1 for values in best_by_material.values()
    )
    unique_best_actions = sorted(set(best_action_names))
    primary = decision_summary["selected_primary_source_setting"]
    primary_has_policy = "mode_counts" in primary
    primary_uses_sensing = bool(
        primary_has_policy and int(primary["mode_counts"]["sense"]) > 0
    )
    primary_gain = (
        float(primary["relative_objective_gain_vs_fallback"])
        if primary_has_policy
        else float("-inf")
    )
    gate_contract = v2_protocol["source_gate"]
    criteria = {
        "nontrivial_action_choice": len(unique_best_actions)
        >= int(gate_contract["minimum_distinct_source_optimal_actions"]),
        "repetition_stable_optimum": stable_material_count
        >= int(gate_contract["minimum_materials_with_repetition_stable_optimum"]),
        "primary_setting_uses_sensing": (
            primary_uses_sensing
            if gate_contract["require_primary_setting_to_use_sensing"]
            else True
        ),
        "positive_costed_source_gain": primary_gain
        > float(gate_contract["minimum_relative_objective_gain"]),
    }
    gate_pass = all(criteria.values())

    result: dict[str, Any] = {
        "schema": "bayesian-phystwin.tracking-cloth-action-feasibility-costed-result.v2",
        "schema_version": 2,
        "study_id": v2_protocol["study_id"],
        "protocol_id": object_digest(v2_protocol),
        "base_source_protocol_id": object_digest(source_protocol),
        "dataset_inventory_id": inventory["inventory_id"],
        "dataset_record": source_protocol["dataset_record"],
        "source_block_count": len(blocks),
        "source_case_count": len(rows),
        "action_names": actions,
        "source_blocks": [
            {"material": material, "repetition": repetition}
            for material, repetition in blocks
        ],
        "action_loss_matrix": losses.tolist(),
        "best_action_by_block": best_action_names,
        "best_actions_by_material": best_by_material,
        "unique_best_actions": unique_best_actions,
        "stable_best_action_material_count": stable_material_count,
        "probe_feature": source_protocol["probe_feature"],
        "probe_threshold_by_interaction": {
            action: float(threshold)
            for action, threshold in zip(actions, thresholds, strict=True)
        },
        "probe_outcome_by_interaction_and_block": probe_outcomes.tolist(),
        "decision_grid": grid,
        "decision_summary": decision_summary,
        "source_gate": {
            "pass": gate_pass,
            "criteria": criteria,
            "automatic_target_follow_on": False,
        },
        "rep3_numeric_outcomes_read": False,
        "rep3_protocol_authorized": False,
        "information_boundary": v2_protocol["information_boundary"],
        "claim_boundary": v2_protocol["claim_boundary"],
    }
    if primary_has_policy:
        result["primary_terminal_only_optimism"] = float(
            primary["relative_terminal_gain_vs_fallback"]
            - primary["relative_objective_gain_vs_fallback"]
        )
    else:
        result["primary_terminal_only_optimism"] = None
    result["result_id"] = object_digest(result)
    return result, rows


def write_outputs(
    output: Path,
    result: dict[str, Any],
    rows: list[dict[str, Any]],
    protocol_path: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "protocol.json").write_text(
        protocol_path.read_text(encoding="utf-8"),
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

    primary = result["decision_summary"]["selected_primary_source_setting"]
    lines = [
        "# Tracking Cloth cost-aware support-robust source audit V2",
        "",
        f"- source gate: `{'pass' if result['source_gate']['pass'] else 'fail'}`",
        f"- result ID: `{result['result_id']}`",
        f"- source blocks: `{result['source_block_count']}`",
        f"- source recordings: `{result['source_case_count']}`",
        f"- distinct source-optimal actions: `{len(result['unique_best_actions'])}`",
        (
            "- materials with a repetition-stable optimum: "
            f"`{result['stable_best_action_material_count']}`"
        ),
        (
            "- primary support-miss probability: "
            f"`{result['decision_summary']['primary_support_miss_probability']}`"
        ),
        "",
        "## Source-gate criteria",
        "",
    ]
    lines.extend(
        f"- `{name}`: `{'pass' if passed else 'fail'}`"
        for name, passed in result["source_gate"]["criteria"].items()
    )
    lines.extend(["", "## Primary source policy", ""])
    if "mode_counts" in primary:
        lines.extend(
            [
                f"- probe cost: `{primary['probe_cost']}`",
                f"- regret tolerance: `{primary['regret_tolerance']}`",
                f"- mode counts: `{primary['mode_counts']}`",
                (
                    "- objective gain versus fallback: "
                    f"`{100.0 * primary['relative_objective_gain_vs_fallback']:.3f}%`"
                ),
                (
                    "- terminal-only gain versus fallback: "
                    f"`{100.0 * primary['relative_terminal_gain_vs_fallback']:.3f}%`"
                ),
            ]
        )
    else:
        lines.append(f"- status: `{primary['status']}`")
    lines.extend(
        [
            "",
            "Repetition 3 was not numerically read and no target execution was",
            "automatically authorized.",
            "",
            result["claim_boundary"],
        ]
    )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    result, rows = build_result(args.dataset_root, args.protocol)
    write_outputs(args.output, result, rows, args.protocol)
    print(
        json.dumps(
            {
                "result_id": result["result_id"],
                "source_gate": result["source_gate"],
                "rep3_numeric_outcomes_read": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
