#!/usr/bin/env python3
"""Generate the controlled Explain--Transport--Probe--Abstain phase study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.explain_transport_probe_abstain_v1 import (
    DiagnosticDisposition,
    ExplainTransportProbeAbstainV1,
)
from bayesian_phystwin_experiments.interventional_cause_adequacy_v1 import (
    InterventionalCauseFamilyAdequacyV1,
)

SHA = "a" * 64


def _canonical_id(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _pipeline(
    *,
    signatures: dict[str, np.ndarray],
    residual: np.ndarray,
    targets: dict[str, np.ndarray],
    candidates: dict[str, np.ndarray],
    costs: dict[str, float],
    noise_radius: float = 0.05,
) -> ExplainTransportProbeAbstainV1:
    adequacy = InterventionalCauseFamilyAdequacyV1(
        residual_id=SHA,
        intervention_roster_id=SHA,
        whitening_id=SHA,
        cause_signature_ids={cause: SHA for cause in signatures},
        cause_signatures=signatures,
        whitened_residual=residual,
        noise_radius=noise_radius,
    )
    return ExplainTransportProbeAbstainV1(
        adequacy_certificate=adequacy,
        target_intervention_roster_id=SHA,
        target_transport_ids={target: SHA for target in targets},
        target_maps=targets,
        candidate_roster_id=SHA,
        candidate_intervention_ids={candidate: SHA for candidate in candidates},
        candidate_designs=candidates,
        intervention_costs=costs,
    )


def run() -> dict[str, Any]:
    # Sorted coefficient order is gauge, material, state.
    ambiguous_signatures = {
        "gauge": np.asarray([[1.0]]),
        "material": np.asarray([[0.0]]),
        "state": np.asarray([[1.0]]),
    }
    informative_candidates = {
        "material-probe": np.asarray([[0.0, 1.0, 0.0]]),
        "redundant-probe": np.asarray([[2.0, 0.0, 2.0]]),
        "state-gauge-probe": np.asarray([[1.0, 0.0, -1.0]]),
    }
    informative_costs = {
        "material-probe": 1.0,
        "redundant-probe": 0.1,
        "state-gauge-probe": 1.0,
    }

    phase = _pipeline(
        signatures=ambiguous_signatures,
        residual=np.asarray([2.0]),
        targets={
            "already-transportable-sum": np.asarray([[1.0, 0.0, 1.0]]),
            "material-effect": np.asarray([[0.0, 1.0, 0.0]]),
            "state-gauge-difference": np.asarray([[1.0, 0.0, -1.0]]),
        },
        candidates=informative_candidates,
        costs=informative_costs,
    )

    unique = _pipeline(
        signatures={
            "gauge": np.asarray([[0.0], [1.0]]),
            "state": np.asarray([[1.0], [0.0]]),
        },
        residual=np.asarray([2.0, -1.0]),
        targets={"state-effect": np.asarray([[0.0, 1.0]])},
        candidates={},
        costs={},
    )

    unmodeled = _pipeline(
        signatures={
            "gauge": np.asarray([[1.0], [0.0], [0.0]]),
            "state": np.asarray([[0.0], [1.0], [0.0]]),
        },
        residual=np.asarray([0.0, 0.0, 2.0]),
        targets={"state-effect": np.asarray([[0.0, 1.0]])},
        candidates={"state-gauge-probe": np.asarray([[1.0, -1.0]])},
        costs={"state-gauge-probe": 1.0},
        noise_radius=0.1,
    )

    unresolvable = _pipeline(
        signatures=ambiguous_signatures,
        residual=np.asarray([2.0]),
        targets={"state-gauge-difference": np.asarray([[1.0, 0.0, -1.0]])},
        candidates={
            "redundant-probe": np.asarray([[2.0, 0.0, 2.0]]),
        },
        costs={"redundant-probe": 0.1},
    )

    no_error = _pipeline(
        signatures={
            "gauge": np.asarray([[1.0]]),
            "state": np.asarray([[1.0]]),
        },
        residual=np.asarray([0.01]),
        targets={"sum": np.asarray([[1.0, 1.0]])},
        candidates={},
        costs={},
        noise_radius=0.05,
    )

    pipelines = {
        "ambiguous-phase": phase,
        "no-detectable-error": no_error,
        "unique-explanation": unique,
        "unmodeled-cause": unmodeled,
        "unresolvable-target": unresolvable,
    }
    records = {
        name: pipeline.to_record() for name, pipeline in sorted(pipelines.items())
    }

    phase_sum = phase.decision_for("already-transportable-sum")
    phase_difference = phase.decision_for("state-gauge-difference")
    phase_material = phase.decision_for("material-effect")
    unique_state = unique.decision_for("state-effect")
    unknown_state = unmodeled.decision_for("state-effect")
    unresolved_difference = unresolvable.decision_for("state-gauge-difference")
    no_error_sum = no_error.decision_for("sum")

    selected_target_costs = [
        float(phase_sum.selected_intervention_cost),
        float(phase_difference.selected_intervention_cost),
        float(phase_material.selected_intervention_cost),
    ]
    full_cause_costs = [
        float(phase_sum.minimum_full_cause_identification_cost),
        float(phase_difference.minimum_full_cause_identification_cost),
        float(phase_material.minimum_full_cause_identification_cost),
    ]
    target_mean_cost = float(np.mean(selected_target_costs))
    full_mean_cost = float(np.mean(full_cause_costs))
    metrics = {
        "registered_phase_targets": 3,
        "transport_without_unique_cause": 1,
        "target_directed_probe_targets": 2,
        "unique_explanation_transport_targets": 1,
        "none_of_the_above_targets": 1,
        "unresolvable_abstentions": 1,
        "no_detectable_error_targets": 1,
        "target_directed_mean_intervention_cost": target_mean_cost,
        "full_cause_mean_intervention_cost": full_mean_cost,
        "relative_mean_cost_reduction_vs_full_cause": (
            1.0 - target_mean_cost / full_mean_cost
        ),
    }
    checks = {
        "unique_registered_explanation_is_separate": (
            unique_state.disposition is DiagnosticDisposition.EXPLAIN_AND_TRANSPORT
        ),
        "cause_ambiguity_does_not_block_invariant_transport": (
            phase_sum.disposition is DiagnosticDisposition.TRANSPORT_WITHOUT_CAUSE
        ),
        "sensitive_target_selects_only_state_gauge_probe": (
            phase_difference.disposition is DiagnosticDisposition.PROBE_THEN_REASSESS
            and phase_difference.selected_interventions == ("state-gauge-probe",)
        ),
        "material_target_selects_only_material_probe": (
            phase_material.disposition is DiagnosticDisposition.PROBE_THEN_REASSESS
            and phase_material.selected_interventions == ("material-probe",)
        ),
        "target_directed_cost_is_two_thirds_lower_than_full_cause": (
            metrics["relative_mean_cost_reduction_vs_full_cause"] >= 2.0 / 3.0 - 1e-12
        ),
        "unmodeled_cause_never_reaches_probe_or_transport": (
            unknown_state.disposition is DiagnosticDisposition.NONE_OF_THE_ABOVE
            and not unknown_state.selected_interventions
            and unknown_state.fallback_required_now
        ),
        "unresolvable_target_abstains": (
            unresolved_difference.disposition is DiagnosticDisposition.ABSTAIN
        ),
        "no_detectable_error_does_not_create_correction": (
            no_error_sum.disposition is DiagnosticDisposition.NO_DETECTABLE_ERROR
        ),
    }

    result: dict[str, Any] = {
        "schema": "bayesian-phystwin.explain-transport-probe-abstain-controlled.v1",
        "pipelines": records,
        "metrics": metrics,
        "checks": checks,
        "decision": (
            "explain-transport-probe-abstain-strict-separation"
            if all(checks.values())
            else "controlled-check-failed"
        ),
        "claim_boundary": (
            "Deterministic local-linear mechanism evidence only. The result does "
            "not validate natural physical causes, nonlinear closure, real probe "
            "models, held-intervention transport, deployment, or safety."
        ),
    }
    result["result_id"] = _canonical_id(result)
    return result


def _report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    return "\n".join(
        [
            "# Explain--Transport--Probe--Abstain controlled result",
            "",
            f"Decision: **`{result['decision']}`**",
            "",
            "The deterministic phase study contains all operational outcomes:",
            "",
            "- a unique registered explanation that transports;",
            "- an ambiguous cause set with an invariant target that transports;",
            "- ambiguous targets resolved by one target-specific probe;",
            "- an omitted cause reported as `none_of_the_above`;",
            "- an unresolvable target that abstains; and",
            "- a residual below the detection threshold that creates no correction.",
            "",
            "For the three ambiguous-family targets, target-directed intervention "
            f"cost averages `{metrics['target_directed_mean_intervention_cost']:.6f}` "
            "versus "
            f"`{metrics['full_cause_mean_intervention_cost']:.6f}` for full cause "
            "identification, a relative reduction of "
            f"`{100.0 * metrics['relative_mean_cost_reduction_vs_full_cause']:.2f}%`.",
            "",
            f"Result ID: `{result['result_id']}`.",
            "",
            result["claim_boundary"],
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(_report(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "result_id": result["result_id"],
            },
            sort_keys=True,
        )
    )
    if result["decision"] != "explain-transport-probe-abstain-strict-separation":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
