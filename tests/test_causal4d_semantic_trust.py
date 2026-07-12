from pathlib import Path

import numpy as np

from causal4d.contracts import PhysicalPosterior, build_causal_context
from causal4d.semantic_posterior import SparseSemanticEvidence, task_posterior_mean
from causal4d.semantic_trust import (
    SemanticValidationCase,
    apply_adaptive_semantic_trust,
    fit_semantic_trust_calibration,
    load_semantic_trust_calibration,
    save_semantic_trust_calibration,
)


def _physical() -> PhysicalPosterior:
    observations = np.zeros((8, 1, 3), dtype=float)
    actions = np.zeros((8, 1, 3), dtype=float)
    context = build_causal_context(
        protocol_id="semantic_trust_unit",
        case_id="source",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=3,
    )
    trajectories = np.zeros((2, 6, 1, 3), dtype=float)
    trajectories[0, :, 0, 0] = -np.arange(6) * 0.01
    trajectories[1, :, 0, 0] = np.arange(6) * 0.01
    return PhysicalPosterior(
        context=context,
        component_ids=("left", "right"),
        state_trajectories_m=trajectories,
        readout_trajectories_m=trajectories,
        readout_variance_m2=np.full((2, 1, 3), 1e-5),
        weights=np.asarray([0.5, 0.5]),
        phi=np.ones((2, 1)),
        kappa_cf=np.asarray([[0.0], [1.0]]),
        hypothesis_indices=np.asarray([0, 1]),
        twin_particle_indices=np.asarray([0, 0]),
        phi_names=("gain",),
        kappa_names=("contact",),
        source_twin_belief_id="1" * 64,
        source_factual_intervention_id="2" * 64,
        source_query_id="3" * 64,
    )


def _evidence(
    positions: np.ndarray,
    *,
    source="MolmoMotion:source",
) -> SparseSemanticEvidence:
    return SparseSemanticEvidence(
        positions_m=positions,
        node_indices=np.asarray([0]),
        physical_frame_indices=np.arange(1, 6, dtype=float),
        scale_m=0.005,
        compare_displacements=True,
        anchor_positions_m=np.zeros((1, 3)),
        source=source,
    )


def _source_case():
    physical = _physical()
    evidence = _evidence(physical.readout_trajectories_m[1, 1:][:, [0]])
    truth = physical.readout_trajectories_m[1].copy()
    return SemanticValidationCase(
        case_id="held_out_source_case",
        physical=physical,
        evidence=evidence,
        truth_m=truth,
    )


def test_source_validation_selects_useful_beta_and_matching_target_passes() -> None:
    source = _source_case()
    calibration = fit_semantic_trust_calibration(
        [source],
        beta_candidates=(0.0, 5.0, 20.0),
        minimum_relative_improvement=0.01,
    )
    assert calibration.selected_beta > 0.0
    task, decision = apply_adaptive_semantic_trust(
        source.physical,
        source.evidence,
        calibration,
    )
    assert decision.accepted
    assert decision.applied_beta == calibration.selected_beta
    physical_mean = task_posterior_mean(source.physical, task)
    assert np.mean(np.square(physical_mean - source.truth_m)) < np.mean(
        np.square(
            np.mean(source.physical.readout_trajectories_m, axis=0)
            - source.truth_m
        )
    )


def test_static_semantic_forecast_falls_back_without_degradation() -> None:
    source = _source_case()
    calibration = fit_semantic_trust_calibration(
        [source],
        beta_candidates=(0.0, 5.0, 20.0),
    )
    static = _evidence(np.zeros((5, 1, 3)), source="MolmoMotion:static")
    task, decision = apply_adaptive_semantic_trust(
        source.physical,
        static,
        calibration,
    )
    assert not decision.accepted
    assert "static_semantic_forecast" in decision.reasons
    assert decision.applied_beta == 0.0
    assert task.task_weights.tobytes() == source.physical.weights.tobytes()
    expected = np.einsum(
        "k,ktnc->tnc",
        source.physical.weights,
        source.physical.readout_trajectories_m,
    )
    assert np.array_equal(task_posterior_mean(source.physical, task), expected)


def test_implausible_semantic_forecast_falls_back_without_degradation() -> None:
    source = _source_case()
    calibration = fit_semantic_trust_calibration(
        [source],
        beta_candidates=(0.0, 5.0, 20.0),
    )
    impossible_positions = np.zeros((5, 1, 3), dtype=float)
    impossible_positions[:, 0, 0] = np.arange(1, 6) * 1.0
    impossible = _evidence(impossible_positions, source="MolmoMotion:impossible")
    task, decision = apply_adaptive_semantic_trust(
        source.physical,
        impossible,
        calibration,
    )
    assert not decision.accepted
    assert {
        "semantic_motion_too_large",
        "outside_physical_support",
    } & set(decision.reasons)
    assert task.task_weights.tobytes() == source.physical.weights.tobytes()


def test_semantic_trust_calibration_round_trip_is_checksummed(tmp_path: Path) -> None:
    calibration = fit_semantic_trust_calibration(
        [_source_case()],
        beta_candidates=(0.0, 5.0),
    )
    path = tmp_path / "semantic_trust.json"
    save_semantic_trust_calibration(path, calibration)
    restored = load_semantic_trust_calibration(path)
    assert restored == calibration
    assert restored.calibration_id == calibration.calibration_id
