from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.query_identifiability_certificate_v2 import (
    QueryIdentifiabilityStatus,
)
from bayesian_phystwin_experiments.interventional_cause_identifiability_v1 import (
    INTERVENTIONAL_CAUSE_IDENTIFIABILITY_CLAIM_BOUNDARY,
    CauseAttributionStatus,
    CauseResponseSignatureV1,
    InterventionalCauseIdentifiabilityCertificateV1,
    InterventionResponseBlockV1,
)

SHA = "a" * 64
ACTIONS = ("action-0", "action-1", "action-2", "action-3")


def _columns() -> dict[str, np.ndarray]:
    return {
        "observation_bias": np.asarray(
            [[1.0, 0.0, 0.0]] * 4,
        ),
        "physical_state": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, -0.5, 0.5],
                [1.0, 0.2, -0.6],
            ]
        ),
        "physical_parameter": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.2],
                [0.2, 0.3, 1.0],
                [0.5, 1.0, -0.3],
            ]
        ),
        "realized_intervention": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.5, -0.5, 1.0],
                [-0.5, 1.0, 0.1],
                [1.0, -1.0, 0.5],
            ]
        ),
        "source_local_discrepancy": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    }


def _cause(cause_id: str, values: np.ndarray) -> CauseResponseSignatureV1:
    blocks = tuple(
        InterventionResponseBlockV1(
            intervention_id=action_id,
            response_signature_id=SHA,
            whitened_response_signature=values[index, :, None],
        )
        for index, action_id in enumerate(ACTIONS)
    )
    return CauseResponseSignatureV1(
        cause_id=cause_id,
        latent_coordinates_id=SHA,
        cause_query_id=SHA,
        intervention_blocks=blocks,
        cause_query_map=np.eye(1),
    )


def _certificate(
    *,
    nuisance: np.ndarray | None = None,
) -> InterventionalCauseIdentifiabilityCertificateV1:
    columns = _columns()
    causes = tuple(_cause(name, columns[name]) for name in sorted(columns))
    if nuisance is None:
        nuisance = np.empty((12, 0))
    return InterventionalCauseIdentifiabilityCertificateV1(
        observation_whitening_id=SHA,
        declared_nuisance_id=SHA,
        cause_family_id=SHA,
        cause_signatures=causes,
        joint_whitened_nuisance_design=nuisance,
        metadata={"protocol": "controlled-falsification"},
    )


def test_changed_interventions_separate_causes_confounding_on_source_action() -> None:
    certificate = _certificate()

    assert certificate.all_nontrivially_identifiable
    for result in certificate.cause_results:
        assert result.status is CauseAttributionStatus.IDENTIFIABLE
        assert result.full_cause_identifiable
        source_status = dict(result.single_intervention_statuses)["action-0"]
        assert source_status is QueryIdentifiabilityStatus.NONIDENTIFIABLE
        assert result.requires_multiple_interventions
        assert result.minimum_identifying_intervention_count in {2, 3}


def test_source_local_discrepancy_requires_source_plus_two_changed_actions() -> None:
    result = _certificate().result_for("source_local_discrepancy")

    assert result.minimum_identifying_intervention_count == 3
    assert all(
        "action-0" in subset for subset in result.minimal_identifying_intervention_sets
    )
    assert {subset for subset in result.minimal_identifying_intervention_sets} == {
        ("action-0", "action-1", "action-2"),
        ("action-0", "action-1", "action-3"),
        ("action-0", "action-2", "action-3"),
    }


def test_declared_action_aligned_nuisance_blocks_false_parameter_attribution() -> None:
    parameter = _columns()["physical_parameter"].reshape(-1, 1)
    certificate = _certificate(nuisance=parameter)
    result = certificate.result_for("physical_parameter")

    assert result.status is CauseAttributionStatus.CONFOUNDED
    assert result.residualized_cause_rank == 0
    assert result.minimum_identifying_intervention_count is None
    assert not result.full_cause_identifiable


def test_pairwise_coherence_exposes_same_direction_competition() -> None:
    certificate = _certificate()
    values = {
        frozenset((item.left_cause_id, item.right_cause_id)): item
        for item in certificate.pairwise_coherences
    }
    pair = values[frozenset(("observation_bias", "physical_state"))]

    assert 0.0 < pair.maximum_canonical_correlation < 1.0
    assert 0.0 < pair.minimum_principal_angle_degrees < 90.0


def test_content_identity_and_arrays_are_immutable() -> None:
    certificate = _certificate()
    artifact_id = certificate.artifact_id

    assert not certificate.joint_whitened_nuisance_design.flags.writeable
    with pytest.raises(ValueError):
        certificate.joint_whitened_nuisance_design.setflags(write=True)
    first_cause = certificate.cause_signatures[0]
    assert not first_cause.stacked_response_signature.flags.writeable
    with pytest.raises(ValueError):
        first_cause.stacked_response_signature.flat[0] = 0.0
    assert certificate.artifact_id == artifact_id
    assert certificate.to_record()["artifact_id"] == artifact_id
    assert certificate.summary()["claim_boundary"] == (
        INTERVENTIONAL_CAUSE_IDENTIFIABILITY_CLAIM_BOUNDARY
    )


def test_invalid_rosters_and_artifact_ids_fail_closed() -> None:
    columns = _columns()
    first = _cause("observation_bias", columns["observation_bias"])
    second = _cause("physical_state", columns["physical_state"])

    with pytest.raises(ValueError, match="sorted by cause_id"):
        InterventionalCauseIdentifiabilityCertificateV1(
            observation_whitening_id=SHA,
            declared_nuisance_id=SHA,
            cause_family_id=SHA,
            cause_signatures=(second, first),
            joint_whitened_nuisance_design=np.empty((12, 0)),
        )

    with pytest.raises(ValueError, match="artifact_id does not match content"):
        InterventionalCauseIdentifiabilityCertificateV1(
            observation_whitening_id=SHA,
            declared_nuisance_id=SHA,
            cause_family_id=SHA,
            cause_signatures=(first, second),
            joint_whitened_nuisance_design=np.empty((12, 0)),
            artifact_id="b" * 64,
        )


def test_query_specific_partial_attribution_is_reported() -> None:
    blocks = tuple(
        InterventionResponseBlockV1(
            intervention_id=action_id,
            response_signature_id=SHA,
            whitened_response_signature=np.asarray(
                [[1.0, float(index == 1)]],
            ),
        )
        for index, action_id in enumerate(ACTIONS)
    )
    target = CauseResponseSignatureV1(
        cause_id="a-target",
        latent_coordinates_id=SHA,
        cause_query_id=SHA,
        intervention_blocks=blocks,
        cause_query_map=np.asarray([[1.0, 1.0]]),
    )
    competitor = CauseResponseSignatureV1(
        cause_id="b-competitor",
        latent_coordinates_id=SHA,
        cause_query_id=SHA,
        intervention_blocks=tuple(
            InterventionResponseBlockV1(
                intervention_id=action_id,
                response_signature_id=SHA,
                whitened_response_signature=np.asarray([[1.0]]),
            )
            for action_id in ACTIONS
        ),
        cause_query_map=np.eye(1),
    )
    certificate = InterventionalCauseIdentifiabilityCertificateV1(
        observation_whitening_id=SHA,
        declared_nuisance_id=SHA,
        cause_family_id=SHA,
        cause_signatures=(target, competitor),
        joint_whitened_nuisance_design=np.empty((4, 0)),
    )
    result = certificate.result_for("a-target")

    assert result.status in {
        CauseAttributionStatus.IDENTIFIABLE,
        CauseAttributionStatus.PARTIALLY_IDENTIFIABLE,
    }
    assert result.identifiable_query_energy_fraction > 0.0
