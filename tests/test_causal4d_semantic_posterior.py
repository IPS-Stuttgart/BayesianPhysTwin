import numpy as np

from causal4d.contracts import PhysicalPosterior, build_causal_context
from causal4d.semantic_posterior import (
    SparseSemanticEvidence,
    build_task_posterior,
    query_point_readout,
    semantic_component_log_scores,
    task_posterior_mean,
)


def _physical(*, unselected_offset=0.0) -> PhysicalPosterior:
    observations = np.zeros((7, 2, 3), dtype=float)
    actions = np.zeros((7, 1, 3), dtype=float)
    context = build_causal_context(
        protocol_id="semantic_unit",
        case_id="synthetic",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=3,
    )
    states = np.zeros((2, 5, 2, 3), dtype=float)
    states[0, :, 0, 0] = -np.arange(5) * 0.01
    states[1, :, 0, 0] = np.arange(5) * 0.01
    states[:, :, 1, 1] = unselected_offset
    return PhysicalPosterior(
        context=context,
        component_ids=("left", "right"),
        state_trajectories_m=states,
        readout_trajectories_m=states,
        readout_variance_m2=np.full((2, 2, 3), 1e-5),
        weights=np.asarray([0.6, 0.4]),
        phi=np.asarray([[1.0], [1.0]]),
        kappa_cf=np.asarray([[0.0], [1.0]]),
        hypothesis_indices=np.asarray([0, 1]),
        twin_particle_indices=np.asarray([0, 0]),
        phi_names=("gain",),
        kappa_names=("contact",),
        source_twin_belief_id="1" * 64,
        source_factual_intervention_id="2" * 64,
        source_query_id="3" * 64,
    )


def _evidence(physical: PhysicalPosterior) -> SparseSemanticEvidence:
    return SparseSemanticEvidence(
        positions_m=physical.readout_trajectories_m[1, 1:4][:, [0]],
        node_indices=np.asarray([0]),
        physical_frame_indices=np.asarray([1.0, 2.0, 3.0]),
        scale_m=0.002,
        compare_displacements=True,
        anchor_positions_m=np.zeros((1, 3)),
        source="MolmoMotion:instruction",
    )


def test_semantics_reweights_only_sparse_hq_readout() -> None:
    physical = _physical()
    evidence = _evidence(physical)
    task = build_task_posterior(physical, evidence, beta=20.0)
    assert task.task_weights[1] > 0.999
    assert task.physical_posterior_id == physical.artifact_id
    assert task.metadata["semantic_interface"] == "q_MM(H_Q(X) | I, language)"
    assert not task.metadata["physical_state_updated_by_semantics"]
    mean = task_posterior_mean(physical, task)
    assert np.all(mean[1:, 0, 0] > 0.0)


def test_beta_zero_is_bit_identical_to_physical_posterior() -> None:
    physical = _physical()
    before = physical.artifact_id
    task = build_task_posterior(physical, _evidence(physical), beta=0.0)
    assert np.array_equal(task.task_weights, physical.weights)
    assert task.task_weights.tobytes() == physical.weights.tobytes()
    assert physical.artifact_id == before


def test_unqueried_dense_nodes_cannot_change_semantic_scores() -> None:
    first = _physical(unselected_offset=0.0)
    second = _physical(unselected_offset=1000.0)
    scores_first = semantic_component_log_scores(first, _evidence(first))
    scores_second = semantic_component_log_scores(second, _evidence(second))
    assert np.array_equal(scores_first, scores_second)


def test_hq_interpolates_physical_frames_without_touching_dense_state() -> None:
    physical = _physical()
    selected = query_point_readout(
        physical,
        np.asarray([0]),
        np.asarray([1.5]),
    )
    expected = 0.5 * (
        physical.readout_trajectories_m[:, 1, [0]]
        + physical.readout_trajectories_m[:, 2, [0]]
    )
    assert np.allclose(selected[:, 0], expected)
