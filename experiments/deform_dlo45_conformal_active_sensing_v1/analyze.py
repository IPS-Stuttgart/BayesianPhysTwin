"""Trajectory-conformal wrapper for fixed DEFORM decision-directed sensing.

The parent sensing policy and finite-support certificate are immutable. One
split-conformal score is retained per complete trajectory: the maximum positive
realized-regret excess over every parent-policy nonfallback decision. The
terminal action is emitted only when its finite-support bound plus the
stratum-specific conformal radius is within a registered regret budget;
otherwise the exact physical fallback is restored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

CONTRACT = "deform-dlo45-conformal-active-sensing-v1"
DLOS = ("DLO4", "DLO5")
ATOL = 1e-12


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in {path}")
    return rows


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if value.get("contract") != CONTRACT or value.get("schema_version") != 1:
        raise ValueError("unsupported protocol")
    parent = value["parent"]
    calibration = value["calibration"]
    operational = value["operational"]
    evaluation = value["evaluation"]
    expected = {
        "workflow_run": 33613192892,
        "artifact_id": 9839951506,
        "artifact_digest": (
            "sha256:31ddbb20dc084514afb6b021ec4f6b0bdff72e22124c4734fb18b1952b347889"
        ),
        "outer_result_id": (
            "ba5f43a2ed1ca9c95a2a032e95679df06d3b6c6ec20cc3ad962ea6326f543fe0"
        ),
        "core_result_id": (
            "ac1626f7392c1de2d95ff4d5fa5e937d113f46c573c48b4dcef5fd3b38ef6ade"
        ),
        "policy": "decision_regret",
        "measurement_budget": 4,
    }
    if any(parent.get(key) != item for key, item in expected.items()):
        raise ValueError("parent identity changed")
    if (
        tuple(calibration["strata"]) != DLOS
        or calibration["unit"] != "complete_trajectory"
        or calibration["score"]
        != (
            "maximum_positive_realized_regret_minus_finite_support_bound_"
            "over_parent_nonfallback_decisions"
        )
        or operational["primary_miscoverage"] != 0.2
        or operational["primary_selection"]
        != (
            "smallest_registered_budget_retaining_every_parent_"
            "nonfallback_calibration_action"
        )
        or evaluation["official_evaluation_split_opened"] is not False
        or evaluation["new_data_collection"] is not False
        or evaluation["target_tuning"] is not False
    ):
        raise ValueError("frozen protocol changed")
    return value


def conformal_quantile(
    values: Sequence[float],
    miscoverage: float,
) -> dict[str, Any]:
    scores = np.asarray(values, dtype=float)
    if scores.ndim != 1 or not scores.size or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite and nonempty")
    rank = math.ceil((len(scores) + 1) * (1.0 - miscoverage))
    radius: float | str
    if rank > len(scores):
        radius = "infinite"
    else:
        radius = float(np.sort(scores)[rank - 1])
    return {
        "calibration_count": len(scores),
        "miscoverage": miscoverage,
        "finite_sample_rank": rank,
        "radius": radius,
        "minimum": float(np.min(scores)),
        "median": float(np.median(scores)),
        "maximum": float(np.max(scores)),
    }


def grouped_trajectory_scores(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["policy"] != "decision_regret" or int(row["budget"]) != 4:
            raise ValueError("unregistered calibration policy")
        grouped[(str(row["dlo"]), str(row["trajectory"]))].append(row)
    result = []
    for (dlo, trajectory), items in sorted(grouped.items()):
        if dlo not in DLOS or len(items) != 19:
            raise ValueError("each trajectory must contribute 19 decisions")
        selected = [row for row in items if bool(row["nonfallback"])]
        score = max(
            (
                max(
                    0.0,
                    float(row["normalized_realized_regret"])
                    - float(row["certificate_worst_case_regret"]),
                )
                for row in selected
            ),
            default=0.0,
        )
        result.append(
            {
                "dlo": dlo,
                "trajectory": trajectory,
                "decision_count": len(items),
                "parent_nonfallback_count": len(selected),
                "score": score,
            }
        )
    return result


def stratified_envelope(
    score_rows: Sequence[Mapping[str, Any]],
    miscoverage: float,
) -> dict[str, Any]:
    by_dlo = {}
    for dlo in DLOS:
        scores = [float(row["score"]) for row in score_rows if row["dlo"] == dlo]
        if len(scores) != 9:
            raise ValueError(f"{dlo}: expected nine calibration trajectories")
        by_dlo[dlo] = conformal_quantile(scores, miscoverage)
    return {
        "miscoverage": miscoverage,
        "nominal_coverage": 1.0 - miscoverage,
        "by_dlo": by_dlo,
    }


def radius_value(envelope: Mapping[str, Any], dlo: str) -> float:
    radius = envelope["by_dlo"][dlo]["radius"]
    return math.inf if radius == "infinite" else float(radius)


def choose_primary_budget(
    rows: Sequence[Mapping[str, Any]],
    envelope: Mapping[str, Any],
    budgets: Sequence[float],
) -> float | None:
    required = max(
        (
            float(row["certificate_worst_case_regret"])
            + radius_value(envelope, str(row["dlo"]))
            for row in rows
            if bool(row["nonfallback"])
        ),
        default=math.inf,
    )
    return next((float(value) for value in budgets if value + ATOL >= required), None)


def bootstrap_interval(
    values: Sequence[float],
    replicates: int,
    seed: int,
) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    means = np.mean(array[indices], axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def evaluate_frontier_point(
    rows: Sequence[Mapping[str, Any]],
    envelope: Mapping[str, Any],
    regret_budget: float,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    fixed = [
        row
        for row in rows
        if row["policy"] == "decision_regret" and int(row["budget"]) == 4
    ]
    if len(fixed) != 304:
        raise ValueError("expected 304 fixed-policy decisions")
    processed = []
    for row in fixed:
        radius = radius_value(envelope, str(row["dlo"]))
        inflated = float(row["certificate_worst_case_regret"]) + radius
        execute = bool(row["nonfallback"]) and inflated <= regret_budget + ATOL
        physical = float(row["physical_task_mse"])
        fallback = float(row["fallback_task_mse"])
        realized = float(row["normalized_realized_regret"])
        processed.append(
            {
                "dlo": str(row["dlo"]),
                "trajectory": str(row["trajectory"]),
                "execute": execute,
                "parent_nonfallback": bool(row["nonfallback"]),
                "mse": physical if execute else fallback,
                "fallback_mse": fallback,
                "harmful": execute and physical > fallback + ATOL,
                "envelope_exceed": execute and realized > inflated + ATOL,
                "budget_exceed": execute and realized > regret_budget + ATOL,
                "sensor_count": int(row["sensor_count"]),
                "effective_hypothesis_count": float(row["effective_hypothesis_count"]),
            }
        )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in processed:
        grouped[(row["dlo"], row["trajectory"])].append(row)
    if len(grouped) != 16 or any(len(items) != 19 for items in grouped.values()):
        raise ValueError("expected 16 trajectories with 19 decisions each")
    trajectory_rows = []
    for (dlo, trajectory), items in sorted(grouped.items()):
        rmse = math.sqrt(float(np.mean([row["mse"] for row in items])))
        fallback = math.sqrt(float(np.mean([row["fallback_mse"] for row in items])))
        trajectory_rows.append(
            {
                "dlo": dlo,
                "trajectory": trajectory,
                "improvement": 1.0 - rmse / fallback,
                "envelope_exceed": any(row["envelope_exceed"] for row in items),
                "budget_exceed": any(row["budget_exceed"] for row in items),
            }
        )
    selected = [row for row in processed if row["execute"]]
    task_rmse = 1000.0 * math.sqrt(float(np.mean([row["mse"] for row in processed])))
    fallback_rmse = 1000.0 * math.sqrt(
        float(np.mean([row["fallback_mse"] for row in processed]))
    )
    improvements = [float(row["improvement"]) for row in trajectory_rows]
    envelope_exceeds = sum(bool(row["envelope_exceed"]) for row in trajectory_rows)
    budget_exceeds = sum(bool(row["budget_exceed"]) for row in trajectory_rows)
    return {
        "regret_budget": regret_budget,
        "decision_count": len(processed),
        "parent_nonfallback_count": sum(
            bool(row["parent_nonfallback"]) for row in processed
        ),
        "nonfallback_count": len(selected),
        "nonfallback_fraction": len(selected) / len(processed),
        "pooled_task_rmse_mm": task_rmse,
        "pooled_fallback_rmse_mm": fallback_rmse,
        "pooled_rmse_reduction": 1.0 - task_rmse / fallback_rmse,
        "mean_trajectory_rmse_reduction": float(np.mean(improvements)),
        "trajectory_bootstrap_95_interval": bootstrap_interval(
            improvements,
            bootstrap_replicates,
            bootstrap_seed,
        ),
        "harmful_nonfallback_count": sum(bool(row["harmful"]) for row in selected),
        "envelope_exceed_decision_count": sum(
            bool(row["envelope_exceed"]) for row in selected
        ),
        "envelope_exceed_trajectory_count": envelope_exceeds,
        "empirical_simultaneous_trajectory_coverage": 1.0
        - envelope_exceeds / len(trajectory_rows),
        "regret_budget_exceed_decision_count": sum(
            bool(row["budget_exceed"]) for row in selected
        ),
        "regret_budget_exceed_trajectory_count": budget_exceeds,
        "mean_acquired_node_blocks_all_decisions": float(
            np.mean([row["sensor_count"] for row in processed])
        ),
        "decisions_acquiring_any_node_block": sum(
            int(row["sensor_count"]) > 0 for row in processed
        ),
        "mean_effective_hypotheses_when_acting": (
            float(np.mean([row["effective_hypothesis_count"] for row in selected]))
            if selected
            else None
        ),
        "state_ambiguous_fraction_when_acting": (
            float(
                np.mean([row["effective_hypothesis_count"] > 1.5 for row in selected])
            )
            if selected
            else None
        ),
    }


def render_summary(result: Mapping[str, Any]) -> str:
    point = result["primary_result"]
    selection = result["primary_selection"]
    radii = point["radii_by_dlo"]
    interval = point["trajectory_bootstrap_95_interval"]
    lines = [
        "# Trajectory-conformal active-sensing result",
        "",
        "- Parent policy: decision-regret acquisition, budget 4.",
        (
            "- Primary nominal trajectory coverage: "
            f"**{100 * (1 - selection['miscoverage']):.0f}%**."
        ),
        f"- Primary regret budget: **{selection['selected_regret_budget']:.2f}**.",
        (
            "- DLO4 / DLO5 conformal radii: "
            f"**{radii['DLO4']:.6f} / {radii['DLO5']:.6f}**."
        ),
        (
            "- Nonfallback decisions: "
            f"**{point['nonfallback_count']}/{point['decision_count']}**."
        ),
        (
            f"- Task RMSE: **{point['pooled_task_rmse_mm']:.3f} mm**, "
            f"versus **{point['pooled_fallback_rmse_mm']:.3f} mm** fallback."
        ),
        (
            "- Equal-trajectory RMSE reduction: "
            f"**{100 * point['mean_trajectory_rmse_reduction']:.2f}%** "
            f"(95% bootstrap **[{100 * interval[0]:.2f}%, "
            f"{100 * interval[1]:.2f}%]**)."
        ),
        (f"- Harmful nonfallback decisions: **{point['harmful_nonfallback_count']}**."),
        (
            "- Empirical simultaneous trajectory coverage: "
            f"**{100 * point['empirical_simultaneous_trajectory_coverage']:.2f}%**."
        ),
        (
            "- Regret-budget exceeds: "
            f"**{point['regret_budget_exceed_decision_count']} decisions on "
            f"{point['regret_budget_exceed_trajectory_count']} trajectories**."
        ),
        (
            "- State-ambiguous acting cases: "
            f"**{100 * point['state_ambiguous_fraction_when_acting']:.2f}%**."
        ),
        "",
        (
            "This is retrospective source-test evidence for a fixed "
            "virtual-sensing policy."
        ),
        (
            "The guarantee is trajectory-marginal under within-DLO "
            "exchangeability; it does"
        ),
        (
            "not validate sensors, avoid sensing cost before fallback, "
            "establish unseen-object"
        ),
        "transfer, or certify robot safety.",
        "",
    ]
    return "\n".join(lines)


def run(parent_dir: Path, protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    outer = read_json(parent_dir / "result.json")
    core = read_json(parent_dir / "core" / "result.json")
    if (
        outer["result_id"] != protocol["parent"]["outer_result_id"]
        or core["result_id"] != protocol["parent"]["core_result_id"]
        or outer["core_result_id"] != core["result_id"]
    ):
        raise ValueError("parent result identity mismatch")
    calibration_rows = read_jsonl(parent_dir / "core" / "calibration_cases.jsonl")
    calibration_rows = [
        row
        for row in calibration_rows
        if row["policy"] == "decision_regret" and int(row["budget"]) == 4
    ]
    test_rows = read_jsonl(parent_dir / "core" / "source_test_cases.jsonl")
    if len(calibration_rows) != 342:
        raise ValueError("expected 342 calibration decisions")
    scores = grouped_trajectory_scores(calibration_rows)
    alphas = [float(value) for value in protocol["calibration"]["miscoverage_levels"]]
    budgets = [float(value) for value in protocol["operational"]["regret_budgets"]]
    envelopes = {f"{alpha:.6g}": stratified_envelope(scores, alpha) for alpha in alphas}
    primary_alpha = float(protocol["operational"]["primary_miscoverage"])
    primary_envelope = envelopes[f"{primary_alpha:.6g}"]
    primary_budget = choose_primary_budget(
        calibration_rows,
        primary_envelope,
        budgets,
    )
    if primary_budget is None:
        raise ValueError("no finite primary operating point")
    frontier = []
    for alpha_index, alpha in enumerate(alphas):
        envelope = envelopes[f"{alpha:.6g}"]
        for budget_index, budget in enumerate(budgets):
            point = evaluate_frontier_point(
                test_rows,
                envelope,
                budget,
                bootstrap_replicates=int(
                    protocol["evaluation"]["bootstrap_replicates"]
                ),
                bootstrap_seed=(
                    int(protocol["evaluation"]["bootstrap_seed"])
                    + 100 * alpha_index
                    + budget_index
                ),
            )
            point.update(
                {
                    "miscoverage": alpha,
                    "nominal_trajectory_coverage": 1.0 - alpha,
                    "radii_by_dlo": {
                        dlo: envelope["by_dlo"][dlo]["radius"] for dlo in DLOS
                    },
                }
            )
            frontier.append(point)
    primary = next(
        point
        for point in frontier
        if abs(float(point["miscoverage"]) - primary_alpha) <= ATOL
        and abs(float(point["regret_budget"]) - primary_budget) <= ATOL
    )
    result: dict[str, Any] = {
        "contract": CONTRACT,
        "schema_version": 1,
        "status": "retrospective-fixed-policy-trajectory-conformal-evaluation",
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "parent": protocol["parent"],
        "theorem": {
            "unit": "complete_trajectory",
            "score": protocol["calibration"]["score"],
            "simultaneous_scope": (
                "all_parent_nonfallback_decisions_on_one_future_trajectory"
            ),
            "statement": (
                "For one future complete trajectory exchangeable within its known "
                "DLO stratum, with probability at least 1-alpha every nonfallback "
                "action emitted by the fixed parent policy has realized normalized "
                "regret no greater than its finite-support bound plus the stratum "
                "split-conformal radius."
            ),
        },
        "calibration_trajectory_scores": scores,
        "envelopes": envelopes,
        "primary_selection": {
            "miscoverage": primary_alpha,
            "rule": protocol["operational"]["primary_selection"],
            "selected_regret_budget": primary_budget,
        },
        "frontier": frontier,
        "primary_result": primary,
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = canonical_sha256(result)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    compact = {
        "contract": CONTRACT,
        "schema_version": 1,
        "status": result["status"],
        "result_id": result["result_id"],
        "protocol_sha256": result["protocol_sha256"],
        "parent": result["parent"],
        "primary_selection": result["primary_selection"],
        "primary_result": result["primary_result"],
        "claim_boundary": result["claim_boundary"],
    }
    (output_dir / "compact_result.json").write_text(
        json.dumps(compact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "SUMMARY.md").write_text(
        render_summary(result),
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.parent_dir, args.protocol, args.output_dir)
    print(render_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
