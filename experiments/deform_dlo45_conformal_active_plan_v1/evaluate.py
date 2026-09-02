"""Trajectory-conformal guard for a frozen decision-directed sensing policy.

The base policy, including its adaptive sensing path and terminal action, is
frozen before calibration. One calibration score is formed per complete
trajectory by taking the maximum positive amount by which realized normalized
regret exceeds the finite-support certificate over every nonfallback decision
emitted by that policy. A split-conformal order statistic then inflates the
certificate. The wrapper retains the already-selected action only when the
inflated bound lies within the registered operational tolerance; otherwise it
returns the exact base fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

CONTRACT = "deform-dlo45-conformal-active-plan-v1"
ATOL = 1.0e-12


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if value.get("contract") != CONTRACT or value.get("schema_version") != 1:
        raise ValueError("unsupported protocol")
    predecessor = value.get("predecessor")
    calibration = value.get("calibration")
    operation = value.get("operation")
    statistics = value.get("statistics")
    frontier = value.get("frontier")
    boundary = value.get("information_order")
    if not all(
        isinstance(section, dict)
        for section in (
            predecessor,
            calibration,
            operation,
            statistics,
            frontier,
            boundary,
        )
    ):
        raise ValueError("protocol sections must be objects")
    assert isinstance(predecessor, dict)
    assert isinstance(calibration, dict)
    assert isinstance(operation, dict)
    assert isinstance(statistics, dict)
    assert isinstance(frontier, dict)
    assert isinstance(boundary, dict)
    if (
        predecessor.get("experiment")
        != "deform-dlo45-decision-directed-sensing-v3"
        or predecessor.get("workflow_run_id") != 33613192892
        or predecessor.get("artifact_id") != 9839951506
        or predecessor.get("artifact_sha256")
        != "579972676099a536bc85033a07308df00a7fc121f450bfa99a692aab22d0759c"
        or predecessor.get("source_revision")
        != "56c31bc56b2fc3526f519f726ff7b922b909c65c"
        or predecessor.get("outer_result_id")
        != "d4aeb724d7a23bcd0137c4e16999737c33c654511e0a7a17adf9727862abbd0f"
        or predecessor.get("core_result_id")
        != "50a4dbdb7547393da7d16cfb2e491308f24437bca54909924df87e7406422ee0"
        or operation.get("base_policy") != "decision_regret"
        or operation.get("base_measurement_budget") != 4
        or operation.get("primary_operational_regret_tolerance") != 0.25
        or operation.get("no_post_probe_action_reoptimization") is not True
        or calibration.get("miscoverage") != 0.2
        or calibration.get("expected_trajectory_count") != 18
        or statistics.get("test_trajectory_count") != 16
        or statistics.get("bootstrap_replicates") != 20000
        or boundary.get("source_test_only") is not True
        or boundary.get("official_evaluation_access") is not False
        or boundary.get("target_tuning") is not False
        or boundary.get("new_data_collection") is not False
    ):
        raise ValueError("frozen protocol changed")
    alphas = [float(item) for item in frontier["miscoverage_levels"]]
    tolerances = [
        float(item) for item in frontier["operational_regret_tolerances"]
    ]
    if any(not 0.0 < item < 1.0 for item in alphas):
        raise ValueError("invalid miscoverage level")
    if any(item <= 0.0 for item in tolerances):
        raise ValueError("invalid regret tolerance")
    return value


def select_base_rows(
    rows: Iterable[dict[str, Any]],
    *,
    policy: str,
    budget: int,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("policy") == policy and int(row.get("budget", -1)) == budget
    ]
    required = {
        "dlo",
        "trajectory",
        "current_frame",
        "nonfallback",
        "certificate_worst_case_regret",
        "normalized_realized_regret",
        "physical_task_mse",
        "fallback_task_mse",
        "harmful_vs_fallback",
        "sensor_count",
        "effective_hypothesis_count",
    }
    for row in selected:
        missing = required - set(row)
        if missing:
            raise ValueError(f"row is missing required fields: {sorted(missing)}")
    return selected


def group_trajectories(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dlo"]), str(row["trajectory"]))].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["current_frame"]))
    return dict(sorted(grouped.items()))


def trajectory_max_excess(rows: Sequence[dict[str, Any]]) -> float:
    excess = [
        max(
            0.0,
            float(row["normalized_realized_regret"])
            - float(row["certificate_worst_case_regret"]),
        )
        for row in rows
        if bool(row["nonfallback"])
    ]
    return max(excess, default=0.0)


def split_conformal_quantile(
    scores: Sequence[float],
    *,
    miscoverage: float,
) -> dict[str, float | int | str]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a finite nonempty vector")
    if not 0.0 < miscoverage < 1.0:
        raise ValueError("miscoverage must lie in (0,1)")
    rank = int(math.ceil((values.size + 1) * (1.0 - miscoverage)))
    if rank > values.size:
        lower_bound = 1.0
        radius_json: float | str = "infinite"
    else:
        radius_json = float(np.sort(values)[rank - 1])
        lower_bound = rank / (values.size + 1)
    return {
        "calibration_trajectory_count": int(values.size),
        "miscoverage": miscoverage,
        "finite_sample_rank": rank,
        "finite_sample_coverage_lower_bound": lower_bound,
        "radius": radius_json,
        "minimum_score": float(np.min(values)),
        "median_score": float(np.median(values)),
        "maximum_score": float(np.max(values)),
    }


def numeric_radius(quantile: Mapping[str, object]) -> float:
    value = quantile["radius"]
    return math.inf if value == "infinite" else float(value)


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed: int,
    replicates: int,
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be finite and nonempty")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(replicates, array.size))
    means = np.mean(array[indices], axis=1)
    return {
        "mean": float(np.mean(array)),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "trajectory_count": int(array.size),
        "replicates": replicates,
    }


def evaluate_operating_point(
    grouped: Mapping[tuple[str, str], Sequence[dict[str, Any]]],
    *,
    radius: float,
    tolerance: float,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    per_trajectory: list[dict[str, Any]] = []
    total_mse: list[float] = []
    total_fallback_mse: list[float] = []
    emitted_rows: list[dict[str, Any]] = []
    for (dlo, trajectory), rows in grouped.items():
        selected_mse: list[float] = []
        fallback_mse: list[float] = []
        trajectory_emitted: list[dict[str, Any]] = []
        for row in rows:
            retain = bool(row["nonfallback"]) and (
                float(row["certificate_worst_case_regret"]) + radius
                <= tolerance + ATOL
            )
            selected_mse.append(
                float(row["physical_task_mse"])
                if retain
                else float(row["fallback_task_mse"])
            )
            fallback_mse.append(float(row["fallback_task_mse"]))
            if retain:
                trajectory_emitted.append(row)
                emitted_rows.append(row)
        rmse = float(np.sqrt(np.mean(selected_mse)))
        fallback_rmse = float(np.sqrt(np.mean(fallback_mse)))
        improvement = (fallback_rmse - rmse) / fallback_rmse
        violation_count = sum(
            float(row["normalized_realized_regret"]) > tolerance + ATOL
            for row in trajectory_emitted
        )
        per_trajectory.append(
            {
                "dlo": dlo,
                "trajectory": trajectory,
                "decision_count": len(rows),
                "nonfallback_count": len(trajectory_emitted),
                "harmful_nonfallback_count": sum(
                    bool(row["harmful_vs_fallback"])
                    for row in trajectory_emitted
                ),
                "regret_budget_exceed_count": violation_count,
                "covered": violation_count == 0,
                "rmse_mm": 1000.0 * rmse,
                "fallback_rmse_mm": 1000.0 * fallback_rmse,
                "relative_improvement": improvement,
            }
        )
        total_mse.extend(selected_mse)
        total_fallback_mse.extend(fallback_mse)

    improvement_values = [
        float(item["relative_improvement"]) for item in per_trajectory
    ]
    pooled_rmse = float(np.sqrt(np.mean(total_mse)))
    pooled_fallback_rmse = float(np.sqrt(np.mean(total_fallback_mse)))
    harmful = sum(bool(row["harmful_vs_fallback"]) for row in emitted_rows)
    regret_exceeds = sum(
        float(row["normalized_realized_regret"]) > tolerance + ATOL
        for row in emitted_rows
    )
    dlos: dict[str, dict[str, Any]] = {}
    for dlo in sorted({str(item["dlo"]) for item in per_trajectory}):
        dlo_items = [item for item in per_trajectory if item["dlo"] == dlo]
        dlo_rows = [
            row
            for (row_dlo, _), rows in grouped.items()
            if row_dlo == dlo
            for row in rows
        ]
        dlo_selected: list[float] = []
        dlo_fallback: list[float] = []
        dlo_emitted = 0
        for row in dlo_rows:
            retain = bool(row["nonfallback"]) and (
                float(row["certificate_worst_case_regret"]) + radius
                <= tolerance + ATOL
            )
            dlo_selected.append(
                float(row["physical_task_mse"])
                if retain
                else float(row["fallback_task_mse"])
            )
            dlo_fallback.append(float(row["fallback_task_mse"]))
            dlo_emitted += int(retain)
        dlo_rmse = float(np.sqrt(np.mean(dlo_selected)))
        dlo_fb = float(np.sqrt(np.mean(dlo_fallback)))
        dlos[dlo] = {
            "trajectory_count": len(dlo_items),
            "nonfallback_count": dlo_emitted,
            "rmse_mm": 1000.0 * dlo_rmse,
            "fallback_rmse_mm": 1000.0 * dlo_fb,
            "relative_improvement": (dlo_fb - dlo_rmse) / dlo_fb,
            "mean_trajectory_improvement": float(
                np.mean([item["relative_improvement"] for item in dlo_items])
            ),
            "wins_ties_losses": [
                sum(item["relative_improvement"] > ATOL for item in dlo_items),
                sum(abs(item["relative_improvement"]) <= ATOL for item in dlo_items),
                sum(item["relative_improvement"] < -ATOL for item in dlo_items),
            ],
        }

    return {
        "miscoverage_radius": "infinite" if not math.isfinite(radius) else radius,
        "operational_regret_tolerance": tolerance,
        "decision_count": sum(len(rows) for rows in grouped.values()),
        "nonfallback_count": len(emitted_rows),
        "nonfallback_fraction": len(emitted_rows)
        / sum(len(rows) for rows in grouped.values()),
        "harmful_nonfallback_count": harmful,
        "regret_budget_exceed_count": regret_exceeds,
        "trajectories_with_regret_budget_exceed": sum(
            not bool(item["covered"]) for item in per_trajectory
        ),
        "covered_trajectory_count": sum(
            bool(item["covered"]) for item in per_trajectory
        ),
        "trajectory_count": len(per_trajectory),
        "pooled_rmse_mm": 1000.0 * pooled_rmse,
        "fallback_pooled_rmse_mm": 1000.0 * pooled_fallback_rmse,
        "pooled_relative_improvement": (
            pooled_fallback_rmse - pooled_rmse
        )
        / pooled_fallback_rmse,
        "trajectory_improvement_bootstrap_95_interval": bootstrap_mean_interval(
            improvement_values,
            seed=seed,
            replicates=replicates,
        ),
        "mean_sensor_count_all_decisions": float(
            np.mean(
                [
                    float(row["sensor_count"])
                    for rows in grouped.values()
                    for row in rows
                ]
            )
        ),
        "mean_sensor_count_nonfallback": (
            float(np.mean([float(row["sensor_count"]) for row in emitted_rows]))
            if emitted_rows
            else 0.0
        ),
        "mean_effective_hypotheses_nonfallback": (
            float(
                np.mean(
                    [float(row["effective_hypothesis_count"]) for row in emitted_rows]
                )
            )
            if emitted_rows
            else 0.0
        ),
        "state_ambiguous_nonfallback_fraction": (
            float(
                np.mean(
                    [
                        float(row["effective_hypothesis_count"]) > 1.0 + ATOL
                        for row in emitted_rows
                    ]
                )
            )
            if emitted_rows
            else 0.0
        ),
        "by_dlo": dlos,
        "per_trajectory": per_trajectory,
    }


def run(
    *,
    predecessor_output: Path,
    protocol_path: Path,
    output_dir: Path,
    source_revision: str,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    if output_dir.exists():
        raise ValueError("output directory already exists")
    predecessor = protocol["predecessor"]
    if source_revision != predecessor["source_revision"]:
        raise ValueError("predecessor source revision mismatch")
    outer = read_json(predecessor_output / "result.json")
    core_root = predecessor_output / "core"
    core = read_json(core_root / "result.json")
    if (
        outer.get("result_id") != predecessor["outer_result_id"]
        or outer.get("core_result_id") != predecessor["core_result_id"]
        or core.get("result_id") != predecessor["core_result_id"]
        or outer.get("source_revision") != source_revision
        or int(outer.get("overlap_audit", {}).get("total_overlap_count", -1)) != 0
        or bool(core.get("evaluation", {}).get("official_evaluation_split_opened"))
    ):
        raise ValueError("predecessor evidence binding failed")

    operation = protocol["operation"]
    policy = str(operation["base_policy"])
    budget = int(operation["base_measurement_budget"])
    calibration_rows = select_base_rows(
        read_jsonl(core_root / "calibration_cases.jsonl"),
        policy=policy,
        budget=budget,
    )
    test_rows = select_base_rows(
        read_jsonl(core_root / "source_test_cases.jsonl"),
        policy=policy,
        budget=budget,
    )
    calibration_groups = group_trajectories(calibration_rows)
    test_groups = group_trajectories(test_rows)
    expected_calibration = int(protocol["calibration"]["expected_trajectory_count"])
    expected_test = int(protocol["statistics"]["test_trajectory_count"])
    if len(calibration_groups) != expected_calibration or len(test_groups) != expected_test:
        raise ValueError(
            "unexpected trajectory roster: "
            f"{len(calibration_groups)} calibration, {len(test_groups)} test"
        )
    if set(calibration_groups) & set(test_groups):
        raise ValueError("calibration and source-test trajectories overlap")
    if any(len(rows) != 19 for rows in calibration_groups.values()):
        raise ValueError("calibration trajectories must contain 19 decisions")
    if any(len(rows) != 19 for rows in test_groups.values()):
        raise ValueError("test trajectories must contain 19 decisions")

    scores = [trajectory_max_excess(rows) for rows in calibration_groups.values()]
    frontier: list[dict[str, Any]] = []
    quantiles: dict[str, dict[str, float | int | str]] = {}
    seed = int(protocol["statistics"]["seed"])
    replicates = int(protocol["statistics"]["bootstrap_replicates"])
    for alpha_index, alpha_value in enumerate(
        protocol["frontier"]["miscoverage_levels"]
    ):
        alpha = float(alpha_value)
        quantile = split_conformal_quantile(scores, miscoverage=alpha)
        quantiles[str(alpha)] = quantile
        radius = numeric_radius(quantile)
        for tolerance_index, tolerance_value in enumerate(
            protocol["frontier"]["operational_regret_tolerances"]
        ):
            tolerance = float(tolerance_value)
            operating = evaluate_operating_point(
                test_groups,
                radius=radius,
                tolerance=tolerance,
                seed=seed + 100 * alpha_index + tolerance_index,
                replicates=replicates,
            )
            frontier.append(
                {
                    "miscoverage": alpha,
                    "finite_sample_coverage_lower_bound": quantile[
                        "finite_sample_coverage_lower_bound"
                    ],
                    **operating,
                }
            )

    primary_alpha = float(protocol["calibration"]["miscoverage"])
    primary_tolerance = float(
        protocol["operation"]["primary_operational_regret_tolerance"]
    )
    primary = next(
        item
        for item in frontier
        if item["miscoverage"] == primary_alpha
        and item["operational_regret_tolerance"] == primary_tolerance
    )
    checks = {
        "predecessor_is_nonoverlapping_source_replication": (
            outer.get("status") == "source-test-only-nonoverlapping-replication"
        ),
        "primary_radius_is_finite": primary["miscoverage_radius"] != "infinite",
        "primary_has_nontrivial_nonfallback_coverage": primary["nonfallback_count"] > 0,
        "primary_improvement_interval_excludes_zero": float(
            primary["trajectory_improvement_bootstrap_95_interval"]["lower"]
        )
        > 0.0,
        "primary_has_zero_observed_harmful_nonfallback": (
            primary["harmful_nonfallback_count"] == 0
        ),
        "primary_has_zero_observed_regret_budget_exceeds": (
            primary["regret_budget_exceed_count"] == 0
            and primary["trajectories_with_regret_budget_exceed"] == 0
        ),
        "state_remains_ambiguous_when_acting": (
            primary["state_ambiguous_nonfallback_fraction"] == 1.0
        ),
        "official_evaluation_remains_closed": (
            protocol["information_order"]["official_evaluation_access"] is False
        ),
    }
    result: dict[str, Any] = {
        "contract": CONTRACT,
        "schema_version": 1,
        "status": "retrospective-source-test-active-policy-calibration",
        "classification": (
            "positive-trajectory-conformal-active-policy"
            if all(checks.values())
            else "mixed-or-negative-trajectory-conformal-active-policy"
        ),
        "protocol_sha256": sha256_file(protocol_path),
        "source_revision": source_revision,
        "predecessor": predecessor,
        "calibration": {
            "score_definition": protocol["calibration"]["score"],
            "trajectory_scores": [
                {
                    "dlo": key[0],
                    "trajectory": key[1],
                    "score": trajectory_max_excess(rows),
                    "base_nonfallback_count": sum(
                        bool(row["nonfallback"]) for row in rows
                    ),
                }
                for key, rows in calibration_groups.items()
            ],
            "quantiles": quantiles,
        },
        "primary": primary,
        "frontier": frontier,
        "checks": checks,
        "all_checks_passed": bool(all(checks.values())),
        "accounting": {
            "calibration_trajectory_count": len(calibration_groups),
            "source_test_trajectory_count": len(test_groups),
            "calibration_decision_count": len(calibration_rows),
            "source_test_decision_count": len(test_rows),
            "official_evaluation_files_opened": False,
            "new_data_collected": False,
        },
        "claim_boundary": protocol["claim_boundary"],
        "interpretation": (
            "A trajectory-level conformal radius is applied to the complete frozen "
            "decision-directed sensing policy before retaining any terminal action. "
            "The retained action is never re-optimized after the probe outcome."
        ),
    }
    result["result_id"] = canonical_sha256(result)
    output_dir.mkdir(parents=True)
    write_json(output_dir / "result.json", result)
    write_json(
        output_dir / "compact_result.json",
        {
            "contract": CONTRACT,
            "classification": result["classification"],
            "result_id": result["result_id"],
            "primary": {
                key: item
                for key, item in primary.items()
                if key not in {"per_trajectory"}
            },
            "checks": checks,
            "claim_boundary": result["claim_boundary"],
        },
    )
    (output_dir / "SUMMARY.md").write_text(
        render_summary(result), encoding="utf-8"
    )
    return result


def render_summary(result: Mapping[str, Any]) -> str:
    primary = result["primary"]
    interval = primary["trajectory_improvement_bootstrap_95_interval"]
    return (
        "# Trajectory-conformal active decision sensing\n\n"
        f"Classification: **{result['classification']}**.\n\n"
        f"- Calibration trajectories: **{result['accounting']['calibration_trajectory_count']}**\n"
        f"- Disjoint source-test trajectories: **{result['accounting']['source_test_trajectory_count']}**\n"
        f"- Primary miscoverage: **{primary['miscoverage']:.2f}**\n"
        f"- Finite-sample trajectory-coverage lower bound: **{primary['finite_sample_coverage_lower_bound']:.2%}**\n"
        f"- Conformal radius: **{primary['miscoverage_radius']:.6f}**\n"
        f"- Operational regret tolerance: **{primary['operational_regret_tolerance']:.2f}**\n"
        f"- Nonfallback decisions: **{primary['nonfallback_count']}/{primary['decision_count']}**\n"
        f"- Pooled RMSE reduction: **{100.0 * primary['pooled_relative_improvement']:.2f}%**\n"
        f"- Mean trajectory reduction: **{100.0 * interval['mean']:.2f}%** "
        f"[**{100.0 * interval['lower']:.2f}%**, **{100.0 * interval['upper']:.2f}%**]\n"
        f"- Harmful nonfallback decisions: **{primary['harmful_nonfallback_count']}**\n"
        f"- Observed regret-budget exceeds: **{primary['regret_budget_exceed_count']}** "
        f"across **{primary['trajectories_with_regret_budget_exceed']}** trajectories\n"
        f"- State ambiguous when acting: **{100.0 * primary['state_ambiguous_nonfallback_fraction']:.1f}%**\n\n"
        "This is trajectory-marginal source-test evidence for one fixed active policy, "
        "not pointwise conditional or unseen-object validation.\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor-output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(
        predecessor_output=args.predecessor_output.resolve(),
        protocol_path=args.protocol.resolve(),
        output_dir=args.output_dir.resolve(),
        source_revision=args.source_revision,
    )
    print(render_summary(result))
    return 0 if result["all_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
