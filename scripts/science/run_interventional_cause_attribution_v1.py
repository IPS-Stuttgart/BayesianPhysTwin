#!/usr/bin/env python3
"""Controlled falsification study for interventional cause attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.interventional_cause_identifiability_v1 import (
    CauseResponseSignatureV1,
    InterventionalCauseIdentifiabilityCertificateV1,
    InterventionResponseBlockV1,
)

SHA = "a" * 64
CAUSES = (
    "observation_bias",
    "physical_parameter",
    "physical_state",
    "realized_intervention",
    "source_local_discrepancy",
)
ACTIONS = ("action-0", "action-1", "action-2", "action-3")


def response_columns() -> dict[str, np.ndarray]:
    """Return four intervention blocks with identical source-action signatures."""
    return {
        "observation_bias": np.asarray(
            [[1.0, 0.0, 0.0]] * 4,
            dtype=np.float64,
        ),
        "physical_parameter": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.2],
                [0.2, 0.3, 1.0],
                [0.5, 1.0, -0.3],
            ],
            dtype=np.float64,
        ),
        "physical_state": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, -0.5, 0.5],
                [1.0, 0.2, -0.6],
            ],
            dtype=np.float64,
        ),
        "realized_intervention": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.5, -0.5, 1.0],
                [-0.5, 1.0, 0.1],
                [1.0, -1.0, 0.5],
            ],
            dtype=np.float64,
        ),
        "source_local_discrepancy": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    }


def _cause(cause_id: str, values: np.ndarray) -> CauseResponseSignatureV1:
    return CauseResponseSignatureV1(
        cause_id=cause_id,
        latent_coordinates_id=SHA,
        cause_query_id=SHA,
        intervention_blocks=tuple(
            InterventionResponseBlockV1(
                intervention_id=action_id,
                response_signature_id=SHA,
                whitened_response_signature=values[index, :, None],
            )
            for index, action_id in enumerate(ACTIONS)
        ),
        cause_query_map=np.eye(1),
    )


def certificate(
    *, nuisance: np.ndarray | None = None
) -> InterventionalCauseIdentifiabilityCertificateV1:
    columns = response_columns()
    if nuisance is None:
        nuisance = np.empty((12, 0), dtype=np.float64)
    return InterventionalCauseIdentifiabilityCertificateV1(
        observation_whitening_id=SHA,
        declared_nuisance_id=SHA,
        cause_family_id=SHA,
        cause_signatures=tuple(_cause(name, columns[name]) for name in CAUSES),
        joint_whitened_nuisance_design=nuisance,
        metadata={"study": "why-is-the-twin-wrong-controlled-v1"},
    )


def _fit_scalar(signature: np.ndarray, observation: np.ndarray) -> float:
    denominator = float(np.dot(signature, signature))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(signature, observation) / denominator)


def _trial(
    rng: np.random.Generator,
    true_cause: str,
    noise_standard_deviation: float,
) -> dict[str, Any]:
    columns = response_columns()
    amplitude = float(rng.choice((-1.0, 1.0)) * rng.uniform(0.5, 1.5))
    outcomes = {
        action: amplitude * columns[true_cause][index]
        + rng.normal(0.0, noise_standard_deviation, size=3)
        for index, action in enumerate(ACTIONS)
    }

    fits = {
        cause: _fit_scalar(columns[cause][0], outcomes["action-0"]) for cause in CAUSES
    }
    source_only_choice = CAUSES[0]
    selection_actions = ("action-1", "action-2")
    selection_losses = {
        cause: float(
            np.mean(
                [
                    np.sum(
                        (
                            fits[cause] * columns[cause][ACTIONS.index(action)]
                            - outcomes[action]
                        )
                        ** 2
                    )
                    for action in selection_actions
                ]
            )
        )
        for cause in CAUSES
    }
    multi_action_choice = min(
        CAUSES,
        key=lambda cause: (selection_losses[cause], cause),
    )
    confirmation_truth = outcomes["action-3"]

    def confirmation_rmse(cause: str) -> float:
        prediction = fits[cause] * columns[cause][3]
        return float(np.sqrt(np.mean((prediction - confirmation_truth) ** 2)))

    wrong_action_prediction = fits[true_cause] * columns[true_cause][2]
    wrong_action_rmse = float(
        np.sqrt(np.mean((wrong_action_prediction - confirmation_truth) ** 2))
    )
    return {
        "true_cause": true_cause,
        "source_only_choice": source_only_choice,
        "multi_action_choice": multi_action_choice,
        "source_only_correct": source_only_choice == true_cause,
        "multi_action_correct": multi_action_choice == true_cause,
        "source_only_confirmation_rmse": confirmation_rmse(source_only_choice),
        "multi_action_confirmation_rmse": confirmation_rmse(multi_action_choice),
        "oracle_confirmation_rmse": confirmation_rmse(true_cause),
        "wrong_action_confirmation_rmse": wrong_action_rmse,
    }


def run_study(
    *,
    trials_per_cause: int,
    seed: int,
    noise_standard_deviation: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rows = [
        _trial(rng, cause, noise_standard_deviation)
        for cause in CAUSES
        for _ in range(trials_per_cause)
    ]
    full_certificate = certificate()
    material_signature = response_columns()["physical_parameter"].reshape(-1, 1)
    nuisance_certificate = certificate(nuisance=material_signature)

    def mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    per_cause = {}
    for cause in CAUSES:
        subset = [row for row in rows if row["true_cause"] == cause]
        per_cause[cause] = {
            "source_only_accuracy": float(
                np.mean([bool(row["source_only_correct"]) for row in subset])
            ),
            "multi_action_accuracy": float(
                np.mean([bool(row["multi_action_correct"]) for row in subset])
            ),
            "source_only_confirmation_rmse": float(
                np.mean([row["source_only_confirmation_rmse"] for row in subset])
            ),
            "multi_action_confirmation_rmse": float(
                np.mean([row["multi_action_confirmation_rmse"] for row in subset])
            ),
            "oracle_confirmation_rmse": float(
                np.mean([row["oracle_confirmation_rmse"] for row in subset])
            ),
        }

    result = {
        "schema": "bayesian-phystwin.interventional-cause-controlled-result.v1",
        "study_id": "why-is-the-twin-wrong-controlled-v1",
        "seed": seed,
        "trials_per_cause": trials_per_cause,
        "total_trials": len(rows),
        "noise_standard_deviation": noise_standard_deviation,
        "causes": list(CAUSES),
        "interventions": list(ACTIONS),
        "certificate_id": full_certificate.artifact_id,
        "source_action_identifiable_cause_count": sum(
            dict(result.single_intervention_statuses)["action-0"].value
            == "identifiable"
            for result in full_certificate.cause_results
        ),
        "all_interventions_identifiable_cause_count": sum(
            result.status.value == "identifiable"
            for result in full_certificate.cause_results
        ),
        "minimum_identifying_intervention_counts": {
            result.cause_id: result.minimum_identifying_intervention_count
            for result in full_certificate.cause_results
        },
        "aggregate": {
            "source_only_cause_accuracy": mean("source_only_correct"),
            "multi_action_cause_accuracy": mean("multi_action_correct"),
            "source_only_confirmation_rmse": mean("source_only_confirmation_rmse"),
            "multi_action_confirmation_rmse": mean("multi_action_confirmation_rmse"),
            "oracle_confirmation_rmse": mean("oracle_confirmation_rmse"),
            "wrong_action_confirmation_rmse": mean("wrong_action_confirmation_rmse"),
        },
        "per_cause": per_cause,
        "undeclared_nuisance_control": {
            "naive_label": "physical_parameter",
            "declared_material_status": nuisance_certificate.result_for(
                "physical_parameter"
            ).status.value,
            "declared_material_residualized_rank": nuisance_certificate.result_for(
                "physical_parameter"
            ).residualized_cause_rank,
        },
        "claim_boundary": (
            "Controlled local-linear falsification only. The result shows that "
            "changed interventions can distinguish a registered finite cause "
            "family and that declaring an aligned nuisance revokes a false "
            "attribution. It does not establish unique physical causation or "
            "real-data transfer."
        ),
    }
    payload = json.dumps(result, allow_nan=False, sort_keys=True).encode("utf-8")
    result["result_id"] = hashlib.sha256(payload).hexdigest()
    return result


def _report(result: dict[str, Any]) -> str:
    aggregate = result["aggregate"]
    lines = [
        "# Why Is the Twin Wrong? Controlled attribution study",
        "",
        "## Registered result",
        "",
        f"- trials: {result['total_trials']}",
        f"- causes identifiable from source action alone: "
        f"{result['source_action_identifiable_cause_count']}/{len(CAUSES)}",
        f"- causes identifiable after intervention changes: "
        f"{result['all_interventions_identifiable_cause_count']}/{len(CAUSES)}",
        f"- source-only cause accuracy: "
        f"{100 * aggregate['source_only_cause_accuracy']:.2f}%",
        f"- multi-action cause accuracy: "
        f"{100 * aggregate['multi_action_cause_accuracy']:.2f}%",
        f"- source-only confirmation RMSE: "
        f"{aggregate['source_only_confirmation_rmse']:.4f}",
        f"- multi-action confirmation RMSE: "
        f"{aggregate['multi_action_confirmation_rmse']:.4f}",
        f"- oracle confirmation RMSE: {aggregate['oracle_confirmation_rmse']:.4f}",
        f"- wrong-action control RMSE: "
        f"{aggregate['wrong_action_confirmation_rmse']:.4f}",
        "",
        "## Minimum intervention count",
        "",
        "| Cause | Minimum interventions |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {cause} | {count} |"
        for cause, count in result["minimum_identifying_intervention_counts"].items()
    )
    control = result["undeclared_nuisance_control"]
    lines.extend(
        [
            "",
            "## Undeclared nuisance control",
            "",
            "An action-aligned nuisance is naively labelled as a material effect. "
            "Once that direction is declared, the material attribution becomes "
            f"`{control['declared_material_status']}` with residualized rank "
            f"{control['declared_material_residualized_rank']}.",
            "",
            "## Boundary",
            "",
            str(result["claim_boundary"]),
            "",
            f"Result ID: `{result['result_id']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials-per-cause", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--noise-standard-deviation", type=float, default=0.05)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    result = run_study(
        trials_per_cause=args.trials_per_cause,
        seed=args.seed,
        noise_standard_deviation=args.noise_standard_deviation,
    )
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output / "report.md").write_text(_report(result), encoding="utf-8")


if __name__ == "__main__":
    main()
