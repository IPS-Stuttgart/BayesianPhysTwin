"""Non-overlapping replication with trajectory-level transport calibration.

The fixed v2 active-sensing operating point is rerun on a source-test roster
that has zero filename overlap with the preceding v2 source-test cohort.  The
v3 layer then calibrates an additive, trajectory-level slack between the exact
finite-support regret certificate and realized source-domain regret.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

import numpy as np

from experiments.deform_dlo45_decision_directed_sensing_v2 import (
    evaluate as core,
)

CONTRACT: Final = "deform-dlo45-decision-directed-sensing-v3"
DLOS: Final = ("DLO4", "DLO5")
ATOL: Final = 1e-12


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected one JSON object per line"
                )
            rows.append(value)
    return rows


def load_protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if value.get("contract") != CONTRACT or value.get("schema_version") != 1:
        raise ValueError("unsupported v3 protocol")
    predecessor = value.get("predecessor")
    transport = value.get("transport_calibration")
    acceptance = value.get("replication_acceptance")
    statistics = value.get("statistics")
    evaluation = value.get("evaluation")
    if not all(
        isinstance(item, dict)
        for item in (
            predecessor,
            transport,
            acceptance,
            statistics,
            evaluation,
        )
    ):
        raise ValueError("v3 protocol sections must be JSON objects")
    assert isinstance(predecessor, dict)
    assert isinstance(transport, dict)
    assert isinstance(acceptance, dict)
    assert isinstance(statistics, dict)
    assert isinstance(evaluation, dict)
    roster = predecessor.get("excluded_source_test_roster")
    if not isinstance(roster, dict):
        raise ValueError("predecessor test roster must be an object")
    if (
        predecessor.get("experiment")
        != "deform-dlo45-decision-directed-sensing-v2"
        or predecessor.get("result_id")
        != "a9b709398413e4518524023942790861801acc438e007179b2d9f6094d9bd035"
        or predecessor.get("fixed_likelihood_scale") != 2.0
        or predecessor.get("fixed_action_prototype_scale") != 1.0
        or predecessor.get("fixed_support_regret_tolerance") != 0.05
        or predecessor.get("fixed_measurement_budget") != 4
        or tuple(sorted(roster)) != DLOS
        or any(len(roster[dlo]) != 8 for dlo in DLOS)
        or transport.get("policy") != "decision_regret"
        or transport.get("measurement_budget") != 4
        or transport.get("miscoverage_level") != 0.1
        or transport.get("calibration_unit") != "complete_trajectory"
        or transport.get("trajectory_nonconformity")
        != "mean_positive_realized_regret_minus_finite_support_certificate"
        or acceptance.get("require_zero_predecessor_source_test_overlap")
        is not True
        or statistics.get("multiple_comparison_correction") != "holm"
        or evaluation.get("official_evaluation_split_opened") is not False
        or evaluation.get("new_data_collection") is not False
        or evaluation.get("target_tuning") is not False
    ):
        raise ValueError("frozen v3 contract changed")
    alpha = float(transport["miscoverage_level"])
    minimum_coverage = float(
        acceptance["minimum_transport_coverage_fraction"]
    )
    if not 0.0 < alpha < 1.0 or not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("invalid v3 risk levels")
    return value


def conformal_quantile(
    scores: list[float],
    miscoverage_level: float,
) -> dict[str, object]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("conformal scores must be a finite nonempty vector")
    if not 0.0 < miscoverage_level < 1.0:
        raise ValueError("miscoverage level must lie strictly between zero and one")
    rank = int(math.ceil((len(values) + 1) * (1.0 - miscoverage_level)))
    if rank > len(values):
        quantile = math.inf
    else:
        quantile = float(np.sort(values)[rank - 1])
    return {
        "calibration_count": len(values),
        "miscoverage_level": miscoverage_level,
        "finite_sample_rank": rank,
        "additive_slack": quantile,
        "minimum_score": float(np.min(values)),
        "median_score": float(np.median(values)),
        "maximum_score": float(np.max(values)),
    }


def trajectory_certificate_scores(
    rows: list[dict[str, Any]],
    *,
    policy: str,
    budget: int,
    minimum_certified: int,
) -> list[dict[str, object]]:
    selected = [
        row
        for row in rows
        if row.get("policy") == policy
        and int(row.get("budget", -1)) == budget
        and bool(row.get("certified"))
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in selected:
        key = (str(row["dlo"]), str(row["trajectory"]))
        grouped.setdefault(key, []).append(row)
    result: list[dict[str, object]] = []
    for (dlo, trajectory), items in sorted(grouped.items()):
        if len(items) < minimum_certified:
            continue
        realized = np.asarray(
            [float(item["normalized_realized_regret"]) for item in items]
        )
        certificate = np.asarray(
            [float(item["certificate_worst_case_regret"]) for item in items]
        )
        excess = realized - certificate
        result.append(
            {
                "dlo": dlo,
                "trajectory": trajectory,
                "certified_decision_count": len(items),
                "mean_realized_regret": float(np.mean(realized)),
                "mean_finite_support_certificate": float(
                    np.mean(certificate)
                ),
                "mean_excess": float(np.mean(excess)),
                "mean_positive_excess": float(
                    np.mean(np.maximum(excess, 0.0))
                ),
                "maximum_excess": float(np.max(excess)),
            }
        )
    return result


def transport_calibration(
    calibration_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, object]:
    settings = protocol["transport_calibration"]
    policy = str(settings["policy"])
    budget = int(settings["measurement_budget"])
    alpha = float(settings["miscoverage_level"])
    minimum = int(settings["minimum_certified_decisions_per_trajectory"])
    calibration = trajectory_certificate_scores(
        calibration_rows,
        policy=policy,
        budget=budget,
        minimum_certified=minimum,
    )
    source_test = trajectory_certificate_scores(
        test_rows,
        policy=policy,
        budget=budget,
        minimum_certified=minimum,
    )
    expected_calibration = 18
    expected_test = 16
    if len(calibration) != expected_calibration or len(source_test) != expected_test:
        raise ValueError(
            "transport calibration requires all complete trajectories: "
            f"{len(calibration)} calibration, {len(source_test)} source test"
        )
    quantile = conformal_quantile(
        [float(item["mean_positive_excess"]) for item in calibration],
        alpha,
    )
    slack = float(quantile["additive_slack"])
    covered = 0
    per_dlo: dict[str, dict[str, int]] = {
        dlo: {"covered": 0, "total": 0} for dlo in DLOS
    }
    enriched: list[dict[str, object]] = []
    for item in source_test:
        bound = float(item["mean_finite_support_certificate"]) + slack
        is_covered = bool(float(item["mean_realized_regret"]) <= bound + ATOL)
        covered += int(is_covered)
        dlo = str(item["dlo"])
        per_dlo[dlo]["covered"] += int(is_covered)
        per_dlo[dlo]["total"] += 1
        enriched.append(
            {
                **item,
                "transport_bound": bound,
                "covered": is_covered,
            }
        )
    for dlo in DLOS:
        total = per_dlo[dlo]["total"]
        per_dlo[dlo]["coverage_fraction"] = (
            per_dlo[dlo]["covered"] / total if total else 0.0
        )
    return {
        "policy": policy,
        "measurement_budget": budget,
        "score": settings["trajectory_nonconformity"],
        "target": settings["bound_target"],
        "quantile": quantile,
        "calibration_trajectory_scores": calibration,
        "source_test_trajectory_scores": enriched,
        "source_test_covered_count": covered,
        "source_test_trajectory_count": len(source_test),
        "source_test_coverage_fraction": covered / len(source_test),
        "source_test_coverage_by_dlo": per_dlo,
        "assumption": (
            "complete calibration and source-test trajectories are exchangeable "
            "for the registered trajectory-mean score"
        ),
    }


def one_sided_sign_test(wins: int, losses: int) -> float:
    count = wins + losses
    if count <= 0:
        return 1.0
    numerator = sum(math.comb(count, value) for value in range(wins, count + 1))
    return float(numerator / (2**count))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        candidate = min(1.0, (total - index) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def sign_test_summary(core_result: dict[str, Any], budget: int) -> dict[str, object]:
    decision = core_result["aggregate"]["decision_regret"][str(budget)]
    improvements = [
        float(item["relative_improvement"])
        for item in decision["per_trajectory"]
    ]
    fallback_wins = sum(value > ATOL for value in improvements)
    fallback_losses = sum(value < -ATOL for value in improvements)
    fallback_ties = len(improvements) - fallback_wins - fallback_losses
    raw: dict[str, float] = {}
    comparisons: dict[str, object] = {}
    for baseline, values in core_result["paired_comparisons"].items():
        row = values[str(budget)]
        wins, ties, losses = row["trajectory_wins_ties_losses"]
        p_value = one_sided_sign_test(int(wins), int(losses))
        raw[baseline] = p_value
        comparisons[baseline] = {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "one_sided_sign_test_p": p_value,
            "mean_improvement_advantage": row[
                "mean_trajectory_improvement_advantage"
            ],
            "bootstrap_95_interval": row[
                "improvement_advantage_bootstrap_95_interval"
            ],
        }
    adjusted = holm_adjust(raw)
    for baseline, value in adjusted.items():
        comparisons[baseline]["holm_adjusted_p"] = value
    return {
        "decision_vs_fallback": {
            "wins": fallback_wins,
            "ties": fallback_ties,
            "losses": fallback_losses,
            "one_sided_sign_test_p": one_sided_sign_test(
                fallback_wins, fallback_losses
            ),
        },
        "decision_vs_acquisition_baselines": comparisons,
        "multiple_comparison_correction": "holm",
    }


def overlap_audit(
    core_result: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, object]:
    excluded = protocol["predecessor"]["excluded_source_test_roster"]
    result: dict[str, object] = {}
    total = 0
    for dlo in DLOS:
        current = set(core_result["source_split"][dlo]["source_test"])
        previous = set(excluded[dlo])
        overlap = sorted(current & previous)
        total += len(overlap)
        result[dlo] = {
            "current_source_test_count": len(current),
            "predecessor_source_test_count": len(previous),
            "overlap_count": len(overlap),
            "overlap": overlap,
            "current_source_test_roster": sorted(current),
        }
    return {
        "total_overlap_count": total,
        "by_dlo": result,
        "selection_information": "filenames_only",
    }


def acceptance_summary(
    core_result: dict[str, Any],
    transport: dict[str, object],
    overlap: dict[str, object],
    protocol: dict[str, Any],
) -> dict[str, object]:
    settings = protocol["replication_acceptance"]
    budget = str(protocol["predecessor"]["fixed_measurement_budget"])
    decision = core_result["aggregate"]["decision_regret"][budget]
    comparisons = core_result["paired_comparisons"]
    checks = {
        "zero_predecessor_source_test_overlap": (
            int(overlap["total_overlap_count"]) == 0
        ),
        "core_calibration_gate": bool(
            core_result["selected_calibration"]["gate_passed"]
        ),
        "decision_improvement_bootstrap_lower_above_zero": (
            float(decision["trajectory_bootstrap_95_interval"][0]) > 0.0
        ),
        "decision_advantage_over_state_variance": (
            float(
                comparisons["state_variance"][budget][
                    "improvement_advantage_bootstrap_95_interval"
                ][0]
            )
            > 0.0
        ),
        "decision_advantage_over_posterior_entropy": (
            float(
                comparisons["posterior_entropy"][budget][
                    "improvement_advantage_bootstrap_95_interval"
                ][0]
            )
            > 0.0
        ),
        "transport_coverage": (
            float(transport["source_test_coverage_fraction"])
            >= float(settings["minimum_transport_coverage_fraction"])
        ),
        "nonfallback_harm": (
            float(decision["nonfallback_harmful_fraction"])
            <= float(settings["maximum_nonfallback_harmful_fraction"])
        ),
        "substantial_state_ambiguity_when_acting": (
            float(decision["nonfallback_effective_hypothesis_count"]) > 2.0
            and float(decision["nonfallback_state_ambiguous_fraction"]) > 0.5
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "minimum_transport_coverage_fraction": settings[
            "minimum_transport_coverage_fraction"
        ],
    }


def render_summary(result: dict[str, Any]) -> str:
    replication = result["replication_operating_point"]
    transport = result["transport_calibration"]
    acceptance = result["acceptance"]
    sign_tests = result["sign_tests"]
    quantile = transport["quantile"]
    lines = [
        "# DEFORM decision-directed sensing v3",
        "",
        f"Classification: **{result['classification']}**",
        "",
        "The v2 operating point was fixed and rerun on a source-test roster "
        "with zero filename overlap with the preceding v2 source-test cohort.",
        "",
        "## Replication operating point",
        "",
        f"- Task RMSE: {float(replication['pooled_task_rmse_mm']):.3f} mm",
        f"- Equal-trajectory improvement: "
        f"{100.0 * float(replication['mean_trajectory_improvement']):.2f}%",
        f"- 95% trajectory bootstrap interval: "
        f"[{100.0 * float(replication['trajectory_bootstrap_95_interval'][0]):.2f}, "
        f"{100.0 * float(replication['trajectory_bootstrap_95_interval'][1]):.2f}]%",
        f"- Nonfallback fraction: "
        f"{100.0 * float(replication['nonfallback_fraction']):.1f}%",
        f"- Mean measurements: {float(replication['mean_sensor_count']):.3f}",
        f"- Nonfallback harmful fraction: "
        f"{100.0 * float(replication['nonfallback_harmful_fraction']):.2f}%",
        f"- Effective hypotheses when acting: "
        f"{float(replication['nonfallback_effective_hypothesis_count']):.2f}",
        "",
        "## Trajectory-level transport calibration",
        "",
        f"- Calibration trajectories: {quantile['calibration_count']}",
        f"- Miscoverage level: {float(quantile['miscoverage_level']):.2f}",
        f"- Finite-sample rank: {quantile['finite_sample_rank']}",
        f"- Additive normalized-regret slack: "
        f"{float(quantile['additive_slack']):.6f}",
        f"- Replication coverage: {transport['source_test_covered_count']}/"
        f"{transport['source_test_trajectory_count']} "
        f"({100.0 * float(transport['source_test_coverage_fraction']):.1f}%)",
        "",
        "## Exact paired sign tests",
        "",
        f"- Decision vs fallback wins/ties/losses: "
        f"{sign_tests['decision_vs_fallback']['wins']}/"
        f"{sign_tests['decision_vs_fallback']['ties']}/"
        f"{sign_tests['decision_vs_fallback']['losses']}",
        f"- One-sided sign-test p-value: "
        f"{float(sign_tests['decision_vs_fallback']['one_sided_sign_test_p']):.6g}",
        "",
        f"Acceptance passed: **{acceptance['passed']}**",
        "",
        "## Boundary",
        "",
        str(result["claim_boundary"]),
        "",
    ]
    return "\n".join(lines)


def compact_result(result: dict[str, Any]) -> dict[str, object]:
    return {
        "contract": result["contract"],
        "schema_version": result["schema_version"],
        "status": result["status"],
        "classification": result["classification"],
        "result_id": result["result_id"],
        "core_result_id": result["core_result_id"],
        "overlap_audit": result["overlap_audit"],
        "replication_operating_point": result["replication_operating_point"],
        "transport_calibration": {
            key: result["transport_calibration"][key]
            for key in (
                "policy",
                "measurement_budget",
                "score",
                "target",
                "quantile",
                "source_test_covered_count",
                "source_test_trajectory_count",
                "source_test_coverage_fraction",
                "source_test_coverage_by_dlo",
                "assumption",
            )
        },
        "sign_tests": result["sign_tests"],
        "acceptance": result["acceptance"],
        "claim_boundary": result["claim_boundary"],
    }


def run(args: argparse.Namespace) -> int:
    protocol_path = Path(args.protocol).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError("output directory already exists")
    output.mkdir(parents=True)
    protocol = load_protocol(protocol_path)
    repository_root = protocol_path.parents[2]
    core_protocol = (
        repository_root / str(protocol["core_protocol_path"])
    ).resolve()
    if not core_protocol.is_file():
        raise ValueError(f"missing core protocol: {core_protocol}")
    core_output = output / "core"
    core_status = core.run(
        argparse.Namespace(
            dataset_root=args.dataset_root,
            protocol=str(core_protocol),
            output_dir=str(core_output),
            source_revision=args.source_revision,
        )
    )
    if core_status != 0:
        raise RuntimeError(f"core replication returned {core_status}")
    core_result = read_json(core_output / "result.json")
    selected = core_result["selected_calibration"]
    predecessor = protocol["predecessor"]
    if (
        float(selected["sensor_log_likelihood_scale"])
        != float(predecessor["fixed_likelihood_scale"])
        or float(selected["action_prototype_scale"])
        != float(predecessor["fixed_action_prototype_scale"])
        or float(selected["regret_tolerance"])
        != float(predecessor["fixed_support_regret_tolerance"])
    ):
        raise ValueError("core operating point differs from frozen predecessor")
    calibration_rows = read_jsonl(core_output / "calibration_cases.jsonl")
    test_rows = read_jsonl(core_output / "source_test_cases.jsonl")
    overlap = overlap_audit(core_result, protocol)
    transport = transport_calibration(
        calibration_rows,
        test_rows,
        protocol,
    )
    budget = int(predecessor["fixed_measurement_budget"])
    sign_tests = sign_test_summary(core_result, budget)
    acceptance = acceptance_summary(
        core_result,
        transport,
        overlap,
        protocol,
    )
    operating_point = core_result["aggregate"]["decision_regret"][str(budget)]
    classification = (
        "strong-nonoverlapping-source-replication"
        if acceptance["passed"]
        else "mixed-nonoverlapping-source-replication"
    )
    result: dict[str, Any] = {
        "contract": CONTRACT,
        "schema_version": 1,
        "status": "source-test-only-nonoverlapping-replication",
        "classification": classification,
        "protocol_sha256": sha256_file(protocol_path),
        "core_protocol_sha256": sha256_file(core_protocol),
        "source_revision": args.source_revision,
        "predecessor": predecessor,
        "core_result_id": core_result["result_id"],
        "core_classification": core_result["classification"],
        "core_selected_calibration": selected,
        "overlap_audit": overlap,
        "replication_operating_point": operating_point,
        "transport_calibration": transport,
        "sign_tests": sign_tests,
        "acceptance": acceptance,
        "accounting": {
            "calibration_trajectory_scores": len(
                transport["calibration_trajectory_scores"]
            ),
            "source_test_trajectory_scores": len(
                transport["source_test_trajectory_scores"]
            ),
            "predecessor_source_test_overlap": overlap[
                "total_overlap_count"
            ],
            "official_evaluation_files_opened": False,
            "new_data_collected": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = canonical_sha256(result)
    write_json(output / "result.json", result)
    write_json(output / "compact_result.json", compact_result(result))
    with (output / "transport_trajectory_scores.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for role in ("calibration", "source_test"):
            for row in transport[f"{role}_trajectory_scores"]:
                stream.write(
                    json.dumps(
                        {"role": role, **row},
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
    (output / "SUMMARY.md").write_text(
        render_summary(result),
        encoding="utf-8",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
