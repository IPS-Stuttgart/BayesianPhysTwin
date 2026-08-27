"""Independent arithmetic replay of sealed synthetic action-choice evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta

from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    read_record,
    read_stage,
    write_record,
)


def independent_parts(
    observations: np.ndarray,
    goals: np.ndarray,
    prefix: np.ndarray,
    future: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    predicted = prefix[:, config["observation_times"]][
        :, :, config["observation_nodes"]
    ]
    difference = (observations[:, None] - predicted[None]).reshape(
        len(observations), 15, 12, 3
    )
    covariance = config["noise_std_m"] ** 2 * np.eye(12)
    covariance += config["shared_bias_std_m"] ** 2 * np.ones((12, 12))
    distance = np.einsum(
        "npic,ij,npjc->np", difference, np.linalg.inv(covariance), difference
    )
    log_weights = -0.5 * distance
    weights = np.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))
    iid_distance = np.sum(difference**2, axis=(2, 3)) / config["noise_std_m"] ** 2
    iid_log_weights = -0.5 * iid_distance
    iid_weights = np.exp(
        iid_log_weights - logsumexp(iid_log_weights, axis=1, keepdims=True)
    )
    offsets = np.asarray(config["action_offsets_m"])
    losses = np.sum((future[None, :, :, -1, -1] - goals[:, None, None]) ** 2, axis=-1)
    losses += config["effort_weight"] * np.sum(offsets**2, axis=1)[None, None]
    means = np.einsum("np,npa->na", weights, losses)
    raw_upper = np.zeros((len(observations), 3, 9))
    for case, w in enumerate(weights):
        for action in range(1, 9):
            for mode in range(2):
                if mode == 0:
                    values = losses[case, :, action] - losses[case, :, 0]
                    probabilities = w
                else:
                    values = (
                        losses[case, :, action, None] - losses[case, None, :, 0]
                    ).ravel()
                    probabilities = (w[:, None] * w[None]).ravel()
                pairs = sorted(
                    zip(values.tolist(), probabilities.tolist(), strict=True)
                )
                accumulated = 0.0
                selected = pairs[-1][0]
                for value, probability in pairs:
                    accumulated += probability
                    if accumulated >= config["posterior_quantile"]:
                        selected = value
                        break
                raw_upper[case, mode, action] = selected
        raw_upper[case, 2] = means[case] - means[case, 0]
    return {
        "weights": weights,
        "iid_weights": iid_weights,
        "expected_losses": means,
        "iid_expected_losses": np.einsum("np,npa->na", iid_weights, losses),
        "nominal_losses": losses[:, 7],
        "raw_upper": raw_upper,
    }


def independent_truth(
    future: np.ndarray, goals: np.ndarray, config: dict[str, Any]
) -> np.ndarray:
    result = np.empty((len(goals), 9))
    for case, goal in enumerate(goals):
        for action, displacement in enumerate(config["action_offsets_m"]):
            error = future[case, action, -1, -1] - goal
            result[case, action] = float(error @ error) + config[
                "effort_weight"
            ] * float(np.dot(displacement, displacement))
    return result


def verify_arithmetic(
    config: dict[str, Any],
    bank: dict[str, np.ndarray],
    calibration: dict[str, np.ndarray],
    prediction: dict[str, np.ndarray],
    outcome: dict[str, np.ndarray],
    calibrators: dict[str, Any],
    result: dict[str, Any],
) -> int:
    checked = 0

    def equal(a: Any, b: Any) -> None:
        nonlocal checked
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)
        checked += np.asarray(a).size

    inferred = []
    for partition in (calibration, prediction):
        parts = independent_parts(
            partition["observations"],
            partition["goals"],
            bank["prefix"],
            bank["future"],
            config,
        )
        for name, value in parts.items():
            equal(partition[name], value)
        inferred.append(parts)
    calibration_truth = independent_truth(
        calibration["future"], calibration["goals"], config
    )
    equal(calibration["losses"], calibration_truth)
    offsets = []
    for index, mode in enumerate(("joint", "independent", "mean")):
        value = calibrators[mode]
        if (value["count"], value["rank"], value["coverage"]) != (39, 36, 0.9):
            raise AssertionError("calibration partition changed")
        scores = np.max(
            calibration_truth[:, 1:]
            - calibration_truth[:, :1]
            - inferred[0]["raw_upper"][:, index, 1:],
            axis=1,
        )
        offset = max(0.0, float(sorted(scores.tolist())[35]))
        equal(value["offset"], offset)
        offsets.append(offset)
    parts = inferred[1]
    decisions = np.zeros((64, 7), dtype=np.int64)
    decisions[:, 1] = parts["nominal_losses"].argmin(axis=1)
    decisions[:, 2] = parts["iid_expected_losses"].argmin(axis=1)
    decisions[:, 3] = parts["expected_losses"].argmin(axis=1)
    for case in range(64):
        for arm, mode_index in ((4, 2), (5, 1), (6, 0)):
            allowed = [0] + [
                a
                for a in range(1, 9)
                if parts["raw_upper"][case, mode_index, a] + offsets[mode_index] < 0
            ]
            decisions[case, arm] = min(
                allowed, key=lambda a: (parts["expected_losses"][case, a], a)
            )
    np.testing.assert_array_equal(prediction["decisions"], decisions)
    checked += decisions.size
    truth = independent_truth(outcome["future"], prediction["goals"], config)
    equal(outcome["losses"], truth)
    deployed = truth[np.arange(64)[:, None], decisions]
    gain = truth[:, :1] - deployed
    indices = np.random.default_rng(config["bootstrap_seed"]).integers(
        0, 64, (config["bootstrap_replicates"], 64)
    )
    arms = config["arms"]
    for index, name in enumerate(arms):
        report = result["arms"][name]
        harm_count = int((deployed[:, index] > truth[:, 0] + 1e-12).sum())
        upper = (
            1.0
            if harm_count == 64
            else float(beta.ppf(0.95, harm_count + 1, 64 - harm_count))
        )
        equal(report["mean_task_loss_m2"], deployed[:, index].mean())
        equal(report["mean_gain_over_hold_m2"], gain[:, index].mean())
        equal(
            report["mean_loss_change_percent"],
            100 * (deployed[:, index].mean() / truth[:, 0].mean() - 1),
        )
        equal(report["nonhold_decisions"], np.count_nonzero(decisions[:, index]))
        equal(report["harmful_decisions"], harm_count)
        equal(report["harm_probability_upper_95"], upper)
        equal(
            report["mean_gain_ci95_m2"],
            np.percentile(gain[indices, index].mean(axis=1), [2.5, 97.5]),
        )
    for index, mode in enumerate(("joint", "independent", "mean")):
        covered = np.all(
            truth[:, 1:] - truth[:, :1]
            <= parts["raw_upper"][:, index, 1:] + offsets[index] + 1e-12,
            axis=1,
        )
        equal(result["simultaneous_action_bound_coverage"][mode], covered.mean())
    for index in (0, 4, 5):
        ci = np.percentile(
            (deployed[:, index] - deployed[:, 6])[indices].mean(axis=1), [2.5, 97.5]
        )
        equal(result["paired_gain_ci95_m2"][arms[index]], ci)
    oracle_count = len(set(truth.argmin(axis=1).tolist()))
    equal(result["distinct_oracle_actions"], oracle_count)
    equal(result["oracle_mean_task_loss_m2"], truth.min(axis=1).mean())
    primary = result["arms"][arms[6]]
    checks = {
        "complete_64_episode_denominator": True,
        "at_least_16_nonhold_decisions": int(np.count_nonzero(decisions[:, 6])) >= 16,
        "at_least_10pct_task_gain_over_hold": primary["mean_loss_change_percent"]
        <= -10,
        "harm_probability_upper_95_at_most_10pct": primary["harm_probability_upper_95"]
        <= 0.10,
        "simultaneous_action_coverage_at_least_85pct": result[
            "simultaneous_action_bound_coverage"
        ]["joint"]
        >= 0.85,
        "at_least_three_distinct_oracle_actions": oracle_count >= 3,
    }
    for control in (arms[4], arms[5]):
        checks[f"gain_10pct_larger_than_{control}"] = primary[
            "mean_gain_over_hold_m2"
        ] >= 1.1 * max(0.0, result["arms"][control]["mean_gain_over_hold_m2"])
    for control in (arms[0], arms[4], arms[5]):
        checks[f"paired_gain_lower_ci_positive_vs_{control}"] = (
            result["paired_gain_ci95_m2"][control][0] > 0
        )
    if result["checks"] != checks or result["source_gate_passed"] is not all(
        checks.values()
    ):
        raise AssertionError("primary gate was not recomputed faithfully")
    return checked + len(checks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = read_record(args.output / "lock.json")
    stages = {
        name: read_stage(args.output, name, lock)
        for name in ("bank", "calibrate", "predict", "score")
    }
    dependencies: dict[str, str] = {}
    for name, (seal, _) in stages.items():
        if seal["dependencies"] != dependencies:
            raise AssertionError("stage evidence order changed")
        if name != "score":
            dependencies[name] = seal["artifact_id"]
    score_seal, outcome = stages["score"]
    checked = verify_arithmetic(
        lock["protocol"],
        stages["bank"][1],
        stages["calibrate"][1],
        stages["predict"][1],
        outcome,
        stages["calibrate"][0]["calibrators"],
        score_seal["result"],
    )
    result = write_record(
        args.output / "verification.json",
        {
            "schema": "dlolab-regret-independent-arithmetic-verification-v1",
            "lock_id": lock["artifact_id"],
            "score_seal_id": score_seal["artifact_id"],
            "passed": True,
            "numeric_and_gate_checks": checked,
            "independent_human_review": False,
            "independent_implementation_replay": True,
            "protected_data_read": False,
            "new_native_execution": False,
            "source_gate_passed": score_seal["result"]["source_gate_passed"],
        },
    )
    print(f"independent arithmetic PASS; checks={checked}; id={result['artifact_id']}")


if __name__ == "__main__":
    main()
