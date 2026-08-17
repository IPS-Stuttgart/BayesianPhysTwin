from __future__ import annotations

import pytest

from bayesian_phystwin.material_backend_qualification_v1 import (
    MaterialBackendQualificationV1,
)


def test_qualification_rejects_candidate_as_its_own_incumbent() -> None:
    runtime_id = "a" * 64

    with pytest.raises(
        ValueError,
        match="incumbent_runtime_id must differ from runtime_id",
    ):
        MaterialBackendQualificationV1(
            canonical_profile_id="jax-fem-quasistatic-v1",
            producer_profile_id="jax-fem-quasistatic-v1",
            transport="lagrangian-export-v1",
            runtime_id=runtime_id,
            qualification_protocol_id="b" * 64,
            source_evidence_id="c" * 64,
            source_group_ids=("object-a", "object-b"),
            incumbent_runtime_id=runtime_id,
            units_coordinate_entity_order_valid=True,
            deterministic_replay_valid=True,
            maximum_zero_action_drift_m=0.0,
            allowed_zero_action_drift_m=0.001,
            maximum_rigid_equivariance_error_m=0.0,
            allowed_rigid_equivariance_error_m=0.001,
            time_step_refinement_relative_error=0.0,
            allowed_time_step_refinement_relative_error=0.05,
            topology_identity_preserved=True,
            physical_sanity_violations=0,
            gradient_claimed=False,
            maximum_jacobian_relative_error=None,
            allowed_jacobian_relative_error=None,
            source_query_parity_rmse_m=0.0,
            allowed_source_query_parity_rmse_m=0.005,
            exact_fallback_verified=True,
            protocol_frozen_before_source_outcomes=True,
            target_outcomes_used=False,
        )
