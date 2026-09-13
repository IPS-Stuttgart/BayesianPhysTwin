"""Retrospective DEFORM active decision-probe-duration pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from bayesian_phystwin.active_decision_probe_v1 import (
    select_minimum_cost_decision_probe,
)
from experiments.deform_dlo45_active_decision_probe_v1._core import (
    OutcomeAssignment,
    PilotProtocol,
    assign_outcome,
    bootstrap_interval,
    build_probe_bundle,
    fit_source_model,
    local_support,
    method_summary,
    read_protocol,
    selected_action,
    terminal_mse,
)
from experiments.deform_dlo45_decision_identifiability_v1._common import (
    DLOS,
    INTERNAL,
    Protocol,
    extract_observation,
    load_protocol,
    load_trajectory,
    trajectory_paths,
    window_starts,
)


def evaluate_dlo(
    dataset_root: Path,
    dlo: str,
    passive_protocol: Protocol,
    pilot_protocol: PilotProtocol,
) -> dict[str, object]:
    train_paths = trajectory_paths(dataset_root, dlo, "train")
    eval_paths = trajectory_paths(dataset_root, dlo, "eval")
    model = fit_source_model(train_paths, passive_protocol, pilot_protocol)
    methods = (
        "fallback",
        "no_probe_certificate",
        "active_minimum_cost",
        *(
            f"fixed_probe_{frames}"
            for frames in pilot_protocol.probe_frames
            if frames > 0
        ),
        "max_outcome_entropy",
        "oracle_probe_action",
    )
    squared_error = {name: [] for name in methods}
    probe_cost = {name: [] for name in methods}
    actions = {name: [] for name in methods}
    per_trajectory: list[dict[str, object]] = []
    duration_counts = {str(frames): 0 for frames in pilot_protocol.probe_frames}
    duration_counts["fallback_no_certified_probe"] = 0
    no_probe_pass = 0
    selected_pass = 0
    selected_oos = 0
    probe_required = 0
    probed_state_ambiguous = 0
    before_regret: list[float] = []
    after_regret: list[float] = []
    selected_support_count: list[int] = []
    selected_probe_distance: list[float] = []
    decision_count = 0
    for path in eval_paths:
        trajectory = load_trajectory(path)
        local_error = {name: [] for name in methods}
        for current in window_starts(passive_protocol):
            observation = extract_observation(trajectory, current, passive_protocol)
            support = local_support(observation.feature, model)
            bundles = tuple(
                build_probe_bundle(
                    support, model, passive_protocol, pilot_protocol, frames
                )
                for frames in pilot_protocol.probe_frames
            )
            prior = np.full(
                len(support.selected), 1.0 / len(support.selected), dtype=np.float64
            )
            selection = select_minimum_cost_decision_probe(
                prior,
                support.quotient_weights,
                support.class_index,
                tuple(bundle.candidate for bundle in bundles),
                regret_tolerance=pilot_protocol.regret_tolerance,
            )
            certificates = selection.certificates
            before_regret.append(certificates[0].minimax_worst_case_regret)
            no_probe_pass += int(
                certificates[0].minimax_worst_case_regret
                <= pilot_protocol.regret_tolerance + 1e-12
            )
            if selection.selected_probe_index is None:
                active_index = 0
                active_is_certified = False
                duration_counts["fallback_no_certified_probe"] += 1
            else:
                active_index = selection.selected_probe_index
                active_is_certified = True
                selected_pass += 1
                duration_counts[str(bundles[active_index].frames)] += 1
            active_frames = bundles[active_index].frames
            probe_required += int(active_is_certified and active_frames > 0)
            max_probe = max(pilot_protocol.probe_frames)
            probe_truth = trajectory[
                current + 1 : current + 1 + max_probe, INTERNAL, :
            ].copy()
            probe_residual = (
                probe_truth - observation.baseline[:max_probe]
            ) / observation.length_scale
            assignments: list[OutcomeAssignment] = []
            for bundle in bundles:
                assignments.append(
                    assign_outcome(
                        probe_residual[: bundle.frames], bundle, pilot_protocol
                    )
                )
            chosen: dict[str, tuple[int, int]] = {"fallback": (0, 0)}
            no_assignment = assignments[0]
            no_action = selected_action(
                bundles[0],
                certificates[0],
                no_assignment,
                require_certificate=True,
                tolerance=pilot_protocol.regret_tolerance,
            )
            chosen["no_probe_certificate"] = (0, no_action)
            active_assignment = assignments[active_index]
            active_action = (
                selected_action(
                    bundles[active_index],
                    certificates[active_index],
                    active_assignment,
                    require_certificate=True,
                    tolerance=pilot_protocol.regret_tolerance,
                )
                if active_is_certified
                else 0
            )
            if not active_assignment.supported:
                selected_oos += int(active_frames > 0)
            chosen["active_minimum_cost"] = (active_index, active_action)
            selected_support_count.append(active_assignment.compatible_hypothesis_count)
            selected_probe_distance.append(active_assignment.squared_distance)
            if (
                active_is_certified
                and active_frames > 0
                and active_assignment.supported
                and (active_assignment.compatible_hypothesis_count > 1)
            ):
                probed_state_ambiguous += 1
            after_regret.append(
                certificates[active_index].minimax_worst_case_regret
                if active_is_certified
                else certificates[0].minimax_worst_case_regret
            )
            for index, bundle in enumerate(bundles):
                if bundle.frames == 0:
                    continue
                method = f"fixed_probe_{bundle.frames}"
                action = selected_action(
                    bundle,
                    certificates[index],
                    assignments[index],
                    require_certificate=True,
                    tolerance=pilot_protocol.regret_tolerance,
                )
                chosen[method] = (index, action)
            entropy_index = min(
                range(len(bundles)),
                key=lambda index: (
                    -bundles[index].nominal_outcome_entropy_bits,
                    bundles[index].frames,
                ),
            )
            entropy_action = selected_action(
                bundles[entropy_index],
                certificates[entropy_index],
                assignments[entropy_index],
                require_certificate=False,
                tolerance=pilot_protocol.regret_tolerance,
            )
            chosen["max_outcome_entropy"] = (entropy_index, entropy_action)
            terminal_truth = trajectory[
                current + passive_protocol.horizon_frames, INTERNAL, :
            ].copy()
            normalized_terminal = (terminal_truth - observation.baseline[-1]).reshape(
                -1
            ) / observation.length_scale
            oracle_best: tuple[float, int, int] | None = None
            for index, bundle in enumerate(bundles):
                assignment = assignments[index]
                if not assignment.supported:
                    continue
                for action in range(len(model.action_scales)):
                    value = terminal_mse(
                        normalized_terminal,
                        bundle,
                        assignment,
                        action,
                        model,
                        observation.length_scale,
                    )
                    candidate = (value, index, action)
                    if oracle_best is None or candidate < oracle_best:
                        oracle_best = candidate
            if oracle_best is None:
                oracle_best = (
                    terminal_mse(
                        normalized_terminal,
                        bundles[0],
                        assignments[0],
                        0,
                        model,
                        observation.length_scale,
                    ),
                    0,
                    0,
                )
            chosen["oracle_probe_action"] = (oracle_best[1], oracle_best[2])
            fallback_value = terminal_mse(
                normalized_terminal,
                bundles[0],
                assignments[0],
                0,
                model,
                observation.length_scale,
            )
            for method, (bundle_index, action) in chosen.items():
                value = terminal_mse(
                    normalized_terminal,
                    bundles[bundle_index],
                    assignments[bundle_index],
                    action,
                    model,
                    observation.length_scale,
                )
                squared_error[method].append(value)
                local_error[method].append(value)
                probe_cost[method].append(bundles[bundle_index].frames)
                actions[method].append(action)
            if not math.isclose(
                squared_error["fallback"][-1],
                fallback_value,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise RuntimeError("fallback computation changed")
            decision_count += 1
        fallback_rmse = math.sqrt(float(np.mean(local_error["fallback"])))
        record: dict[str, object] = {
            "trajectory": path.name,
            "decision_count": len(local_error["fallback"]),
            "fallback_terminal_rmse_mm": 1000.0 * fallback_rmse,
        }
        for method in methods[1:]:
            rmse = math.sqrt(float(np.mean(local_error[method])))
            record[f"{method}_terminal_rmse_mm"] = 1000.0 * rmse
            record[f"{method}_ratio"] = rmse / max(fallback_rmse, 1e-12)
        per_trajectory.append(record)
    aggregate = {
        method: method_summary(
            squared_error[method],
            squared_error["fallback"],
            probe_cost[method],
            actions[method],
            len(model.action_scales),
        )
        for method in methods
    }
    active_improvement = np.asarray(
        [
            100.0 * (1.0 - float(record["active_minimum_cost_ratio"]))
            for record in per_trajectory
        ],
        dtype=np.float64,
    )
    lower, upper = bootstrap_interval(
        active_improvement,
        pilot_protocol.bootstrap_replicates,
        pilot_protocol.bootstrap_seed + (4 if dlo == "DLO4" else 5),
    )
    return {
        "dlo": dlo,
        "decision_count": decision_count,
        "aggregate": aggregate,
        "active": {
            "duration_counts": duration_counts,
            "certified_probe_fraction": selected_pass / decision_count,
            "no_probe_certified_fraction": no_probe_pass / decision_count,
            "probe_required_fraction": probe_required / decision_count,
            "selected_probe_out_of_support_fraction": selected_oos / decision_count,
            "probed_decision_with_multiple_supported_states_fraction": probed_state_ambiguous
            / decision_count,
            "mean_supported_hypotheses_after_selected_probe": float(
                np.mean(selected_support_count)
            ),
            "mean_selected_probe_squared_distance": float(
                np.mean(selected_probe_distance)
            ),
            "mean_pre_probe_minimax_regret": float(np.mean(before_regret)),
            "mean_selected_minimax_regret": float(np.mean(after_regret)),
            "mean_trajectory_improvement_pct": float(np.mean(active_improvement)),
            "trajectory_bootstrap_95_pct": [lower, upper],
            "trajectory_wins_ties_losses": [
                int(np.count_nonzero(active_improvement > 1e-12)),
                int(np.count_nonzero(np.abs(active_improvement) <= 1e-12)),
                int(np.count_nonzero(active_improvement < -1e-12)),
            ],
        },
        "per_trajectory": per_trajectory,
        "_raw": {
            "squared_error": squared_error,
            "probe_cost": probe_cost,
            "actions": actions,
        },
    }


def pooled_result(dlo_results: dict[str, dict[str, object]]) -> dict[str, object]:
    methods = tuple(dlo_results["DLO4"]["aggregate"].keys())
    pooled: dict[str, object] = {}
    for method in methods:
        errors: list[float] = []
        fallback: list[float] = []
        probes: list[int] = []
        actions: list[int] = []
        for dlo in DLOS:
            raw = dlo_results[dlo]["_raw"]
            assert isinstance(raw, dict)
            dlo_error = raw["squared_error"]
            dlo_probe = raw["probe_cost"]
            dlo_action = raw["actions"]
            assert isinstance(dlo_error, dict)
            assert isinstance(dlo_probe, dict)
            assert isinstance(dlo_action, dict)
            errors.extend(dlo_error[method])
            fallback.extend(dlo_error["fallback"])
            probes.extend(dlo_probe[method])
            actions.extend(dlo_action[method])
        pooled[method] = method_summary(errors, fallback, probes, actions, 3)
    return pooled


def run(
    dataset_root: Path, passive_protocol_path: Path, pilot_protocol_path: Path
) -> dict[str, object]:
    passive_protocol = load_protocol(passive_protocol_path)
    pilot_protocol = read_protocol(pilot_protocol_path)
    if max(pilot_protocol.probe_frames) >= passive_protocol.horizon_frames:
        raise ValueError("probe must end before the terminal query horizon")
    dlo_results = {
        dlo: evaluate_dlo(dataset_root, dlo, passive_protocol, pilot_protocol)
        for dlo in DLOS
    }
    pooled = pooled_result(dlo_results)
    for result in dlo_results.values():
        result.pop("_raw")
    return {
        "schema": "bayesian-phystwin/deform-dlo45-active-probe-pilot-v1",
        "schema_version": 1,
        "status": "retrospective-development-mechanism-pilot",
        "dataset": {
            "name": "DEFORM",
            "dlos": list(DLOS),
            "evaluation_scope": "within-dlo-official-held-trajectories",
        },
        "probe_semantics": "Select a duration before target internal response is read; execute the registered endpoint-motion prefix; observe motion-capture internal response; quantize it with a source-local deterministic outcome model; choose a contingent residual-admission action; score only the original fixed terminal horizon.",
        "pilot_protocol": {
            "probe_frames": list(pilot_protocol.probe_frames),
            "outcome_count": pilot_protocol.outcome_count,
            "response_quotient_classes": pilot_protocol.cluster_count,
            "neighbors": pilot_protocol.neighbors,
            "temperature_scale": pilot_protocol.temperature_scale,
            "regret_tolerance": pilot_protocol.regret_tolerance,
            "target_support_multiplier": pilot_protocol.target_support_multiplier,
        },
        "dlos": dlo_results,
        "pooled": pooled,
        "claim_boundary": "This retrospective pilot uses oracle-quality motion-capture internal-node response during a prefix of the already recorded endpoint motion. It tests active probe duration and contingent belief-revision decisions, not alternative counterfactual probe directions, learned perception, unseen-object generalization, continuous control, target regret guarantees, deployment, or safety.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--passive-protocol",
        type=Path,
        default=Path(
            "experiments/deform_dlo45_decision_identifiability_v1/protocol.json"
        ),
    )
    parser.add_argument(
        "--pilot-protocol",
        type=Path,
        default=Path("experiments/deform_dlo45_active_decision_probe_v1/protocol.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.dataset_root, args.passive_protocol, args.pilot_protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pooled = result["pooled"]
    assert isinstance(pooled, dict)
    print(
        json.dumps(
            {
                "status": result["status"],
                "pooled": pooled,
                "active_by_dlo": {dlo: result["dlos"][dlo]["active"] for dlo in DLOS},
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
