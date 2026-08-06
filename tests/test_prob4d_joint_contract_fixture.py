from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.observation_belief import ObservationBeliefV1
from bayesian_phystwin.observation_belief_gauge_adapter import (
    build_gauge_aware_batch_from_observation_belief,
)
from bayesian_phystwin.prob4d_causal_lineage import (
    PROB4D_JOINT_GAUGE_MODEL,
    validate_prob4d_causal_observation_belief,
)

FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "prob4d_joint_observation_v1.json"
)


def _belief() -> tuple[ObservationBeliefV1, str]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    descriptor = payload["descriptor"]
    arrays = {
        name: np.asarray(record["values"], dtype=np.dtype(record["dtype"]))
        for name, record in payload["arrays"].items()
    }
    belief = ObservationBeliefV1(
        case_id=descriptor["case_id"],
        stream_id=descriptor["stream_id"],
        causal_frame_stop=descriptor["causal_frame_stop"],
        view_names=tuple(descriptor["view_names"]),
        window_names=tuple(descriptor["window_names"]),
        factor_names=tuple(descriptor["factor_names"]),
        source_repository=descriptor["source_repository"],
        source_revision=descriptor["source_revision"],
        source_artifact_sha256=descriptor["source_artifact_sha256"],
        metadata=descriptor["metadata"],
        **arrays,
    )
    return belief, payload["expected_artifact_id"]


def _explicit_v2_belief() -> ObservationBeliefV1:
    belief, _ = _belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["prob4d_causal_stream_contract_version"] = 2
    metadata["metric_anchor_covariance_in_joint_factor"] = True
    metadata["metric_gauge_anchor"].update(
        {
            "schema_name": "prob4d.metric-gauge-anchor",
            "schema_version": 1,
            "case_id": belief.case_id,
            "coordinate_frame": "phystwin-world",
            "world_frame_id": "phystwin-world",
            "metric_units": "m",
            "calibration_artifact_sha256": "b" * 64,
            "covariance_treatment": "propagated_external_prior",
        }
    )
    return replace(belief, metadata=metadata)


def _adapt(belief: ObservationBeliefV1):
    state = np.zeros((belief.observation_count, 3, 2))
    state[:, 0, 0] = 1.0
    state[:, 1, 1] = 1.0
    return build_gauge_aware_batch_from_observation_belief(
        belief,
        physical_prediction_xyz_m=np.zeros_like(belief.mean_xyz_m),
        state_jacobian=state,
        query_state_jacobian=state[:1],
        physical_response_scale_m=0.05,
    )


def test_joint_gauge_fixture_has_identical_content_address_and_semantics() -> None:
    belief, expected_artifact_id = _belief()

    validation = validate_prob4d_causal_observation_belief(belief)
    adapted = _adapt(belief)

    assert belief.artifact_id == expected_artifact_id
    assert validation["covariance_semantics"] == PROB4D_JOINT_GAUGE_MODEL
    assert validation["cross_window_covariance_preserved"] is True
    assert validation["factor_group_count"] == 1
    assert validation["factor_rank"] == 5
    assert adapted.summary()["gauge_parameter_count"] == 5
    assert adapted.summary()["prob4d_causal_lineage_validated"] is True

    # The same latent column affects observations from different windows. The
    # adapter must therefore retain one shared nuisance vector, not duplicate it
    # once per window.
    cross_covariance = (
        belief.low_rank_factor_m[0] @ belief.low_rank_factor_m[2].T
    )
    assert cross_covariance[0, 0] != 0.0


def test_explicit_v2_binds_calibration_and_propagated_anchor_covariance() -> None:
    validation = validate_prob4d_causal_observation_belief(
        _explicit_v2_belief()
    )

    assert validation["stream_contract_version"] == 2
    assert validation["stream_contract_version_inferred"] is False
    assert validation["calibration_artifact_sha256"] == "b" * 64
    assert validation["metric_anchor_covariance_treatment"] == (
        "propagated_external_prior"
    )


def test_explicit_v2_rejects_missing_calibration_digest() -> None:
    belief = _explicit_v2_belief()
    metadata = deepcopy(dict(belief.metadata))
    del metadata["metric_gauge_anchor"]["calibration_artifact_sha256"]

    with pytest.raises(ValueError, match="calibration_artifact_sha256"):
        validate_prob4d_causal_observation_belief(
            replace(belief, metadata=metadata)
        )


def test_explicit_v2_rejects_untracked_anchor_covariance() -> None:
    belief = _explicit_v2_belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["metric_anchor_covariance_in_joint_factor"] = False

    with pytest.raises(ValueError, match="include metric-anchor covariance"):
        validate_prob4d_causal_observation_belief(
            replace(belief, metadata=metadata)
        )


def test_joint_gauge_fixture_rejects_per_window_factor_groups() -> None:
    belief, _ = _belief()

    with pytest.raises(ValueError, match="one shared factor group"):
        validate_prob4d_causal_observation_belief(
            replace(belief, factor_group_ids=belief.window_indices)
        )


def test_joint_gauge_fixture_rejects_rank_metadata_drift() -> None:
    belief, _ = _belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["gauge_posterior"]["exported_factor_rank"] = 4

    with pytest.raises(ValueError, match="rank differs"):
        validate_prob4d_causal_observation_belief(
            replace(belief, metadata=metadata)
        )


def test_joint_gauge_fixture_rejects_cross_window_claim_drift() -> None:
    belief, _ = _belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["joint_cross_window_gauge_covariance_represented"] = False

    with pytest.raises(ValueError, match="covariance flags differ"):
        validate_prob4d_causal_observation_belief(
            replace(belief, metadata=metadata)
        )


# The stable-core coverage job invokes this file explicitly. Import adjacent
# contract cases here so their new package code is covered by the same line and
# branch ratchets without weakening the thresholds or duplicating the workflow.
from prob4d_factor_stream_contract_cases import *  # noqa: E402,F403
from test_source_competence_linearization import *  # noqa: E402,F403
from test_source_competence_reliability import *  # noqa: E402,F403
