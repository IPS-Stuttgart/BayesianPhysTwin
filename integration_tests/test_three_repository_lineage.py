"""Independent Causal4D validation of the portable Prob4D observation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from causal4d.contracts import (
    TwinBelief,
    build_causal_context,
    load_contract,
    save_contract,
)
from causal4d.observation_lineage import (
    bind_twin_belief_observation_lineage,
    load_observation_lineage,
    validate_twin_belief_observation_lineage,
)
from prob4d.provider_v1 import save_observation_belief_export
from test_three_repository_golden_path import (
    EXPECTED_OBSERVATION_ARTIFACT_ID,
    _producer_artifact,
)


def test_causal4d_independently_validates_and_binds_prob4d_lineage(
    tmp_path: Path,
) -> None:
    observation_path = tmp_path / "prob4d-observation.npz"
    save_observation_belief_export(observation_path, _producer_artifact())

    lineage = load_observation_lineage(observation_path)
    assert lineage.artifact_id == EXPECTED_OBSERVATION_ARTIFACT_ID
    assert lineage.provider_validation["validated"] is True
    assert lineage.provider_validation["stream_contract_version"] == 2
    assert lineage.provider_validation["stream_contract_version_inferred"] is False
    assert lineage.provider_validation["cross_window_covariance_preserved"] is True

    frame_count = 8
    intervention_frame = 6
    observations = np.zeros((frame_count, 1, 3), dtype=np.float64)
    actions = np.zeros((frame_count, 1, 3), dtype=np.float64)
    context = build_causal_context(
        protocol_id="three-repository-installed-wheel-v1",
        case_id="three-repository-golden-path",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=intervention_frame,
    )
    unbound = TwinBelief(
        context=context,
        endpoint_frame=intervention_frame - 1,
        particle_ids=("particle-0",),
        theta_names=("spring_log_scale",),
        endpoint_position_m=np.zeros((1, 1, 3), dtype=np.float64),
        endpoint_velocity_mps=np.zeros((1, 1, 3), dtype=np.float64),
        theta=np.zeros((1, 1), dtype=np.float64),
        discrepancy_mean_m=np.zeros((1, 1, 3), dtype=np.float64),
        discrepancy_variance_m2=np.full((1, 1, 3), 1e-6, dtype=np.float64),
        weights=np.ones(1, dtype=np.float64),
    )

    unbound_result = validate_twin_belief_observation_lineage(
        unbound,
        lineage,
        require_bound=False,
    )
    assert unbound_result["status"] == "valid"
    assert unbound_result["lineage_bound"] is False

    bound = bind_twin_belief_observation_lineage(unbound, lineage)
    bound_result = validate_twin_belief_observation_lineage(bound, lineage)
    assert bound_result["status"] == "valid"
    assert bound_result["lineage_bound"] is True
    assert bound.metadata["source_observation_belief_id"] == lineage.artifact_id
    assert bound.metadata["source_observation_provider_validation"] == dict(
        lineage.provider_validation
    )

    bound_path = tmp_path / "lineage-bound-twin-belief.npz"
    save_contract(bound_path, bound)
    loaded = load_contract(bound_path)
    assert isinstance(loaded, TwinBelief)
    assert loaded.artifact_id == bound.artifact_id
    assert validate_twin_belief_observation_lineage(loaded, lineage)[
        "lineage_bound"
    ] is True
