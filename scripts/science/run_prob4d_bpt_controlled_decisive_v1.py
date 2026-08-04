#!/usr/bin/env python3
"""Run the locked controlled Prob4D-to-BayesianPhysTwin decision study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin._prior_aware_gauge_math import PriorAwareGaugeConfigV1
from bayesian_phystwin.prior_aware_gauge_belief import (
    update_prior_aware_gauge_belief,
)

_SCIENCE_DIRECTORY = Path(__file__).resolve().parent
if str(_SCIENCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_SCIENCE_DIRECTORY))

from prob4d_bpt_controlled_decisive_core_v1 import (  # noqa: E402
    BASELINE_METHOD,
    CHI_SQUARE_3_90,
    FINITE_INFINITY,
    MARGINAL_METHOD,
    METHODS,
    PRIMARY_METHOD,
    REJECT_ALL_THRESHOLD,
    REPORT_SCHEMA,
    TRIAL_SCHEMA,
    Candidate,
    CandidateScore,
    GroupData,
    GuardCalibration,
    StudyConfig,
    TrialResult,
    _batch_for_method,
    _canonical_sha256,
    _parse_args,
    _query_covariance,
    _require,
    _risk_from_result,
    _sha256,
    _write_json,
    generate_group,
    load_protocol,
)
from prob4d_bpt_controlled_decisive_core_v1 import (  # noqa: E402
    _condition_gauge_prior as _condition_gauge_prior,
)


def _candidate_bpt(
    group: GroupData,
    method_id: str,
    config: StudyConfig,
) -> Candidate:
    batch = _batch_for_method(group, method_id, config)
    result = update_prior_aware_gauge_belief(
        batch,
        config=PriorAwareGaugeConfigV1(
            state_prior_std_m=config.state_prior_std,
            shared_bias_prior_std_m=0.012,
            view_bias_prior_std_m=0.010,
            effective_samples_per_correlation_group=12.0,
            degrees_of_freedom=5.0,
            outlier_covariance_multiplier=36.0,
            maximum_iterations=20,
            maximum_condition_number=1e13,
            minimum_conditional_information_fraction=1e-5,
            minimum_identifiable_fraction=0.02,
            minimum_query_sensitivity_fraction=1e-4,
            maximum_state_update_m=0.065,
            maximum_update_to_physical_response_ratio=4.0,
        ),
    )
    if result.inference_admissible:
        correction = np.einsum(
            "ncs,s->nc",
            group.query_state_jacobian,
            result.state_coefficients,
            optimize=True,
        )
    else:
        correction = np.zeros_like(group.true_query_correction_m)
    state_covariance = result.posterior_covariance[
        : config.state_count, : config.state_count
    ]
    covariance = _query_covariance(
        group.query_state_jacobian,
        state_covariance,
    )
    risk, nominal, identifiable, sensitivity, converged = _risk_from_result(
        group,
        result,
        covariance,
    )
    return Candidate(
        method_id=method_id,
        inference_admissible=bool(result.inference_admissible),
        reason=str(result.reason),
        correction_m=correction,
        covariance_m2=covariance,
        risk_score=risk,
        nominal_probability=nominal,
        identifiable_fraction=identifiable,
        query_sensitivity_fraction=sensitivity,
        fixed_point_converged=converged,
    )


def _candidate_last_frame(group: GroupData, config: StudyConfig) -> Candidate:
    stack = group.stack
    selected = stack.frame_indices == np.max(stack.frame_indices)
    innovation = (stack.world_mean_m - group.physical_prediction_m)[selected]
    design = group.state_jacobian[selected]
    matrix = design.reshape(-1, config.state_count)
    target = innovation.reshape(-1)
    ridge = 1e-4
    normal = matrix.T @ matrix + ridge * np.eye(config.state_count)
    right = matrix.T @ target
    condition_number = float(np.linalg.cond(normal))
    try:
        coefficients = np.linalg.solve(normal, right)
    except np.linalg.LinAlgError:
        coefficients = np.zeros(config.state_count)
    residual = target - matrix @ coefficients
    residual_variance = max(
        float(np.mean(np.square(residual))),
        config.conditional_noise_std_m**2,
    )
    state_covariance = residual_variance * np.linalg.inv(normal)
    correction = np.einsum(
        "ncs,s->nc",
        group.query_state_jacobian,
        coefficients,
        optimize=True,
    )
    covariance = _query_covariance(
        group.query_state_jacobian,
        state_covariance,
    )
    width = float(np.sqrt(np.mean(np.trace(covariance, axis1=1, axis2=2))))
    signal = float(np.sqrt(np.mean(np.square(target))))
    residual_rms = float(np.sqrt(np.mean(np.square(residual))))
    risk = (
        residual_rms / max(signal, 1e-12)
        + width / max(group.physical_response_scale_m, 1e-12)
        + math.log10(max(condition_number, 1.0)) / 20.0
    )
    admissible = bool(
        np.all(np.isfinite(coefficients))
        and condition_number < 1e12
        and np.max(np.linalg.norm(correction, axis=1)) <= 0.065
    )
    if not admissible:
        correction = np.zeros_like(correction)
    return Candidate(
        method_id="B1_naive_last_frame_state",
        inference_admissible=admissible,
        reason="naive-last-frame-admissible" if admissible else "naive-fallback",
        correction_m=correction,
        covariance_m2=covariance,
        risk_score=risk,
        nominal_probability=1.0,
        identifiable_fraction=1.0 if admissible else 0.0,
        query_sensitivity_fraction=1.0 if admissible else 0.0,
        fixed_point_converged=True,
    )


def make_candidate(
    group: GroupData,
    method_id: str,
    config: StudyConfig,
) -> Candidate:
    if method_id == BASELINE_METHOD:
        return Candidate(
            method_id=method_id,
            inference_admissible=False,
            reason="physical-fallback-reference",
            correction_m=np.zeros_like(group.true_query_correction_m),
            covariance_m2=np.zeros(
                (*group.true_query_correction_m.shape[:1], 3, 3),
                dtype=np.float64,
            ),
            risk_score=FINITE_INFINITY,
            nominal_probability=1.0,
            identifiable_fraction=0.0,
            query_sensitivity_fraction=0.0,
            fixed_point_converged=True,
        )
    if method_id == "B1_naive_last_frame_state":
        return _candidate_last_frame(group, config)
    return _candidate_bpt(group, method_id, config)


def _rmse(candidate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(candidate - truth))))


def _coverage_and_width(
    candidate: Candidate,
    truth: np.ndarray,
) -> tuple[float | None, float | None]:
    if not candidate.inference_admissible:
        return None, None
    covered: list[bool] = []
    widths: list[float] = []
    for prediction, actual, covariance in zip(
        candidate.correction_m,
        truth,
        candidate.covariance_m2,
        strict=True,
    ):
        covariance = 0.5 * (covariance + covariance.T)
        covariance = covariance + np.eye(3) * 1e-12
        residual = actual - prediction
        try:
            nees = float(residual @ np.linalg.solve(covariance, residual))
        except np.linalg.LinAlgError:
            nees = float("inf")
        covered.append(nees <= CHI_SQUARE_3_90)
        widths.append(float(np.sqrt(np.trace(covariance))))
    return float(np.mean(covered)), float(np.mean(widths))


def score_candidate(
    group: GroupData,
    candidate: Candidate,
    harmful_margin_m: float,
) -> CandidateScore:
    baseline = np.zeros_like(group.true_query_correction_m)
    baseline_rmse = _rmse(baseline, group.true_query_correction_m)
    raw_rmse = _rmse(candidate.correction_m, group.true_query_correction_m)
    coverage, width = _coverage_and_width(
        candidate,
        group.true_query_correction_m,
    )
    return CandidateScore(
        group_id=group.group_id,
        scenario=group.scenario,
        method_id=candidate.method_id,
        candidate=candidate,
        baseline_rmse_m=baseline_rmse,
        raw_rmse_m=raw_rmse,
        harmful_raw=raw_rmse > baseline_rmse + harmful_margin_m,
        coverage_90=coverage,
        predictive_width_rms_m=width,
    )


def calibrate_guard(
    scores: Sequence[CandidateScore],
    config: StudyConfig,
) -> GuardCalibration:
    method_id = scores[0].method_id
    if method_id == BASELINE_METHOD:
        baseline = float(np.mean([value.baseline_rmse_m for value in scores]))
        return GuardCalibration(
            method_id=method_id,
            risk_threshold=REJECT_ALL_THRESHOLD,
            accepted_group_count=0,
            harmful_accepted_count=0,
            harmful_accepted_rate=0.0,
            deployed_mean_rmse_m=baseline,
            baseline_mean_rmse_m=baseline,
            calibration_group_count=len(scores),
            fallback_only=True,
        )
    admissible_scores = sorted(
        {
            float(value.candidate.risk_score)
            for value in scores
            if value.candidate.inference_admissible
            and np.isfinite(value.candidate.risk_score)
        }
    )
    thresholds = [REJECT_ALL_THRESHOLD, *admissible_scores]
    best: tuple[float, int, int, float, float] | None = None
    baseline_mean = float(np.mean([value.baseline_rmse_m for value in scores]))
    for threshold in thresholds:
        accepted = [
            value
            for value in scores
            if value.candidate.inference_admissible
            and value.candidate.risk_score <= threshold
        ]
        accepted_count = len(accepted)
        harmful_count = sum(value.harmful_raw for value in accepted)
        harmful_rate = harmful_count / accepted_count if accepted_count else 0.0
        if accepted_count and accepted_count < config.guard_minimum_accepted_groups:
            continue
        if harmful_rate > config.guard_harmful_rate_at_most:
            continue
        deployed = [
            value.raw_rmse_m
            if value.candidate.inference_admissible
            and value.candidate.risk_score <= threshold
            else value.baseline_rmse_m
            for value in scores
        ]
        deployed_mean = float(np.mean(deployed))
        candidate_key = (
            deployed_mean,
            -accepted_count,
            harmful_count,
            float(threshold),
            harmful_rate,
        )
        if best is None or candidate_key < best:
            best = candidate_key
    if best is None:
        best = (baseline_mean, 0, 0, REJECT_ALL_THRESHOLD, 0.0)
    deployed_mean, negative_count, harmful_count, threshold, harmful_rate = best
    accepted_count = -negative_count
    return GuardCalibration(
        method_id=method_id,
        risk_threshold=threshold,
        accepted_group_count=accepted_count,
        harmful_accepted_count=harmful_count,
        harmful_accepted_rate=harmful_rate,
        deployed_mean_rmse_m=deployed_mean,
        baseline_mean_rmse_m=baseline_mean,
        calibration_group_count=len(scores),
        fallback_only=accepted_count == 0,
    )


def apply_guard(
    score: CandidateScore,
    calibration: GuardCalibration,
) -> TrialResult:
    candidate = score.candidate
    accepted = bool(
        candidate.inference_admissible
        and candidate.risk_score <= calibration.risk_threshold
    )
    baseline = np.zeros_like(candidate.correction_m)
    deployed = candidate.correction_m if accepted else baseline
    exact_fallback = bool(accepted or deployed.tobytes() == baseline.tobytes())
    _require(exact_fallback, "guard failed to return exact physical fallback")
    deployed_rmse = score.raw_rmse_m if accepted else score.baseline_rmse_m
    baseline = score.baseline_rmse_m
    raw_improvement = 1.0 - score.raw_rmse_m / baseline if baseline else 0.0
    deployed_improvement = 1.0 - deployed_rmse / baseline if baseline else 0.0
    return TrialResult(
        schema=TRIAL_SCHEMA,
        group_id=score.group_id,
        scenario=score.scenario,
        method_id=score.method_id,
        solver_admissible=candidate.inference_admissible,
        solver_reason=candidate.reason,
        risk_score=float(candidate.risk_score),
        guard_threshold=float(calibration.risk_threshold),
        guard_accepted=accepted,
        exact_fallback=exact_fallback,
        baseline_rmse_m=score.baseline_rmse_m,
        raw_rmse_m=score.raw_rmse_m,
        deployed_rmse_m=deployed_rmse,
        raw_harmful=score.harmful_raw,
        harmful_accepted=bool(accepted and score.harmful_raw),
        raw_improvement_fraction=raw_improvement,
        deployed_improvement_fraction=deployed_improvement,
        coverage_90=score.coverage_90,
        predictive_width_rms_m=score.predictive_width_rms_m,
        nominal_probability=candidate.nominal_probability,
        identifiable_fraction=candidate.identifiable_fraction,
        query_sensitivity_fraction=candidate.query_sensitivity_fraction,
        fixed_point_converged=candidate.fixed_point_converged,
    )


def _paired_interval(
    differences: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(differences),
        size=(resamples, len(differences)),
    )
    means = np.mean(differences[indices], axis=1)
    return tuple(map(float, np.quantile(means, [0.025, 0.975])))


def _risk_coverage(
    rows: Sequence[TrialResult],
) -> dict[str, float | None]:
    candidates = [value for value in rows if value.solver_admissible]
    candidates.sort(key=lambda value: value.risk_score)
    result: dict[str, float | None] = {}
    for fraction in (0.25, 0.50, 0.75, 1.0):
        count = max(1, int(math.ceil(fraction * len(candidates))))
        selected = candidates[:count]
        result[f"lowest_risk_{int(100 * fraction)}_percent_rmse_m"] = (
            float(np.mean([value.raw_rmse_m for value in selected]))
            if selected
            else None
        )
    return result


def aggregate_results(
    trials: Sequence[TrialResult],
    config: StudyConfig,
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for method_index, method_id in enumerate(METHODS):
        rows = [value for value in trials if value.method_id == method_id]
        deployed = np.asarray([value.deployed_rmse_m for value in rows])
        baseline = np.asarray([value.baseline_rmse_m for value in rows])
        differences = deployed - baseline
        interval = _paired_interval(
            differences,
            resamples=config.bootstrap_resamples,
            seed=config.bootstrap_seed + method_index,
        )
        accepted = [value for value in rows if value.guard_accepted]
        coverage = [
            value.coverage_90 for value in accepted if value.coverage_90 is not None
        ]
        width = [
            value.predictive_width_rms_m
            for value in accepted
            if value.predictive_width_rms_m is not None
        ]
        by_scenario: dict[str, Any] = {}
        for scenario in config.scenarios:
            selected = [value for value in rows if value.scenario == scenario]
            scenario_deployed = float(
                np.mean([value.deployed_rmse_m for value in selected])
            )
            scenario_baseline = float(
                np.mean([value.baseline_rmse_m for value in selected])
            )
            by_scenario[scenario] = {
                "group_count": len(selected),
                "deployed_mean_rmse_m": scenario_deployed,
                "baseline_mean_rmse_m": scenario_baseline,
                "deployed_improvement_fraction": (
                    1.0 - scenario_deployed / scenario_baseline
                ),
                "acceptance_fraction": float(
                    np.mean([value.guard_accepted for value in selected])
                ),
                "harmful_accepted_count": sum(
                    value.harmful_accepted for value in selected
                ),
            }
        aggregate[method_id] = {
            "group_count": len(rows),
            "raw_mean_rmse_m": float(np.mean([value.raw_rmse_m for value in rows])),
            "deployed_mean_rmse_m": float(np.mean(deployed)),
            "baseline_mean_rmse_m": float(np.mean(baseline)),
            "deployed_improvement_fraction": float(
                1.0 - np.mean(deployed) / np.mean(baseline)
            ),
            "paired_deployed_minus_baseline_95_m": list(interval),
            "solver_admissible_fraction": float(
                np.mean([value.solver_admissible for value in rows])
            ),
            "acceptance_fraction": float(
                np.mean([value.guard_accepted for value in rows])
            ),
            "accepted_group_count": len(accepted),
            "harmful_accepted_count": sum(value.harmful_accepted for value in rows),
            "harmful_accepted_rate": (
                sum(value.harmful_accepted for value in rows) / len(accepted)
                if accepted
                else 0.0
            ),
            "all_rejections_exact_fallback": all(
                value.exact_fallback for value in rows
            ),
            "accepted_coverage_90_mean": (
                float(np.mean(coverage)) if coverage else None
            ),
            "accepted_predictive_width_rms_m": (
                float(np.mean(width)) if width else None
            ),
            "risk_coverage": _risk_coverage(rows),
            "by_scenario": by_scenario,
        }
    return aggregate


def _decision(
    aggregate: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    primary = aggregate[PRIMARY_METHOD]
    marginal = aggregate[MARGINAL_METHOD]
    criteria_config = protocol["acceptance_criteria"]
    worst_scenario_regression = max(
        0.0,
        max(
            -float(value["deployed_improvement_fraction"])
            for value in primary["by_scenario"].values()
        ),
    )
    criteria = {
        "mean_improvement_at_least_registered": (
            primary["deployed_improvement_fraction"]
            >= criteria_config["mean_improvement_fraction_at_least"]
        ),
        "paired_upper_bound_below_zero": (
            primary["paired_deployed_minus_baseline_95_m"][1] < 0.0
        ),
        "harmful_accepted_rate_at_most_registered": (
            primary["harmful_accepted_rate"]
            <= criteria_config["harmful_accepted_rate_at_most"]
        ),
        "worst_scenario_regression_at_most_registered": (
            worst_scenario_regression
            <= criteria_config["worst_scenario_regression_fraction_at_most"]
        ),
        "all_rejections_exact_fallback": primary["all_rejections_exact_fallback"],
        "explicit_persistent_noninferior_to_marginal": (
            primary["deployed_mean_rmse_m"]
            <= (1.0 + criteria_config["explicit_vs_marginal_noninferiority_fraction"])
            * marginal["deployed_mean_rmse_m"]
        ),
    }
    return {
        "primary_method": PRIMARY_METHOD,
        "reference_method": BASELINE_METHOD,
        "criteria": criteria,
        "overall_passed": all(criteria.values()),
        "worst_scenario_regression_fraction": worst_scenario_regression,
        "method_decision": (
            "advance-to-fresh-physical-object/session-gate"
            if all(criteria.values())
            else "retain-as-controlled-negative-or-partial-result"
        ),
    }


def _generate_groups(
    config: StudyConfig,
    *,
    calibration: bool,
) -> list[GroupData]:
    groups_per_scenario = (
        config.calibration_groups_per_scenario
        if calibration
        else config.target_groups_per_scenario
    )
    seed_start = config.calibration_seed if calibration else config.target_seed
    prefix = "calibration" if calibration else "target"
    groups: list[GroupData] = []
    for scenario_index, scenario in enumerate(config.scenarios):
        for offset in range(groups_per_scenario):
            seed = seed_start + 100000 * scenario_index + offset
            groups.append(
                generate_group(
                    seed,
                    scenario,
                    config,
                    group_prefix=prefix,
                )
            )
    return groups


def _score_groups(
    groups: Sequence[GroupData],
    config: StudyConfig,
) -> dict[str, list[CandidateScore]]:
    output = {method_id: [] for method_id in METHODS}
    for group in groups:
        for method_id in METHODS:
            candidate = make_candidate(group, method_id, config)
            output[method_id].append(
                score_candidate(
                    group,
                    candidate,
                    config.harmful_margin_m,
                )
            )
    return output


def run_study(
    protocol: Mapping[str, Any],
    config: StudyConfig,
    *,
    repository_revision: str,
    prob4d_revision: str,
) -> tuple[dict[str, Any], list[TrialResult]]:
    _require(
        prob4d_revision == config.source_revision,
        "executing Prob4D revision differs from frozen protocol",
    )
    calibration_groups = _generate_groups(config, calibration=True)
    calibration_scores = _score_groups(calibration_groups, config)
    calibrations = {
        method_id: calibrate_guard(calibration_scores[method_id], config)
        for method_id in METHODS
    }

    target_groups = _generate_groups(config, calibration=False)
    target_scores = _score_groups(target_groups, config)
    trials = [
        apply_guard(score, calibrations[method_id])
        for method_id in METHODS
        for score in target_scores[method_id]
    ]
    aggregate = aggregate_results(trials, config)
    decision = _decision(aggregate, protocol)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _canonical_sha256(protocol),
        "repository_revision": repository_revision,
        "prob4d_revision": prob4d_revision,
        "configuration": asdict(config),
        "calibration": {
            method_id: asdict(calibration)
            for method_id, calibration in calibrations.items()
        },
        "aggregate": aggregate,
        "decision": decision,
        "claim_boundary": protocol["claim_boundary"],
    }
    report["report_id"] = _canonical_sha256(report)
    return report, trials


def _write_trials(path: Path, trials: Sequence[TrialResult]) -> None:
    fieldnames = list(asdict(trials[0]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            row = asdict(trial)
            writer.writerow(row)


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    aggregate = report["aggregate"]
    lines = [
        "# Controlled Prob4D-to-BayesianPhysTwin decision study",
        "",
        f"Decision: **{'PASS' if report['decision']['overall_passed'] else 'FAIL'}**",
        "",
        "| Method | Deployed RMSE | Improvement | Accept | Harmful accepted |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method_id in METHODS:
        row = aggregate[method_id]
        lines.append(
            "| "
            + method_id
            + f" | {1000 * row['deployed_mean_rmse_m']:.3f} mm"
            + f" | {100 * row['deployed_improvement_fraction']:+.2f}%"
            + f" | {100 * row['acceptance_fraction']:.1f}%"
            + f" | {row['harmful_accepted_count']} |"
        )
    lines.extend(["", "## Registered criteria", ""])
    for name, passed in report["decision"]["criteria"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(["", "## Claim boundary", "", report["claim_boundary"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_checksums(output_dir: Path) -> None:
    checksum_path = output_dir / "SHA256SUMS"
    files = [
        path
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != checksum_path.name
    ]
    checksum_path.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )


def execute(args: argparse.Namespace) -> int:
    protocol, config = load_protocol(args.protocol)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    report, trials = run_study(
        protocol,
        config,
        repository_revision=str(args.repository_revision),
        prob4d_revision=str(args.prob4d_revision),
    )
    _write_json(output_dir / "report.json", report)
    _write_trials(output_dir / "trials.csv", trials)
    _write_markdown(output_dir / "summary.md", report)
    _write_json(output_dir / "protocol.json", protocol)
    _write_checksums(output_dir)
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    return 0 if report["decision"]["overall_passed"] else 3


def main() -> None:
    raise SystemExit(execute(_parse_args()))


if __name__ == "__main__":
    main()
