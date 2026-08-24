from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.query_identifiability_certificate_v2 import (
    QueryIdentifiabilityStatus,
)
from bayesian_phystwin_experiments.multi_action_query_identifiability_v1 import (
    MULTI_ACTION_QUERY_IDENTIFIABILITY_CLAIM_BOUNDARY,
    ActionIdentifiabilityBlockV1,
    MultiActionQueryIdentifiabilityCertificateV1,
)

SHA = "a" * 64


def _block(
    action_id: str,
    physical: object,
    query: object,
    **kwargs: object,
) -> ActionIdentifiabilityBlockV1:
    return ActionIdentifiabilityBlockV1(
        action_id=action_id,
        physical_response_id=SHA,
        observation_mapping_id=SHA,
        query_transport_id=SHA,
        whitened_physical_design=np.asarray(physical),
        transported_query_map=np.asarray(query),
        **kwargs,
    )


def _certificate(
    blocks: tuple[ActionIdentifiabilityBlockV1, ...],
    nuisance: object | None = None,
    **kwargs: object,
) -> MultiActionQueryIdentifiabilityCertificateV1:
    row_count = sum(block.observation_dimension for block in blocks)
    if nuisance is None:
        nuisance = np.empty((row_count, 0))
    return MultiActionQueryIdentifiabilityCertificateV1(
        latent_coordinates_id=SHA,
        whitening_id=SHA,
        joint_nuisance_design_id=SHA,
        joint_query_id=SHA,
        action_blocks=blocks,
        joint_whitened_nuisance_design=np.asarray(nuisance),
        **kwargs,
    )


def test_two_actions_jointly_identify_query_hidden_from_each_action() -> None:
    certificate = _certificate(
        (
            _block("action-a", [[1.0, 0.0]], [[1.0, 1.0]]),
            _block("action-b", [[0.0, 1.0]], [[1.0, 1.0]]),
        )
    )

    assert certificate.status is QueryIdentifiabilityStatus.IDENTIFIABLE
    assert certificate.identifiable
    assert certificate.requires_multiple_actions
    assert certificate.joint_certificate.physical_rank == 2
    assert all(
        status is QueryIdentifiabilityStatus.NONIDENTIFIABLE
        for _, status in certificate.single_action_statuses
    )
    assert {
        contribution.without_action_status
        for contribution in certificate.action_contributions
    } == {QueryIdentifiabilityStatus.NONIDENTIFIABLE}
    assert all(
        contribution.energy_fraction_loss > 0.0
        for contribution in certificate.action_contributions
    )


def test_redundant_action_has_zero_leave_one_out_loss() -> None:
    certificate = _certificate(
        (
            _block("action-a", np.eye(2), [[1.0, 0.0]]),
            _block("action-b", np.eye(2), [[1.0, 0.0]]),
        )
    )

    assert certificate.status is QueryIdentifiabilityStatus.IDENTIFIABLE
    assert not certificate.requires_multiple_actions
    assert all(
        contribution.without_action_status is QueryIdentifiabilityStatus.IDENTIFIABLE
        for contribution in certificate.action_contributions
    )
    assert all(
        contribution.energy_fraction_loss == pytest.approx(0.0)
        for contribution in certificate.action_contributions
    )


def test_joint_nuisance_can_preserve_shared_cross_action_confounding() -> None:
    certificate = _certificate(
        (
            _block("action-a", [[1.0]], [[1.0]]),
            _block("action-b", [[1.0]], [[1.0]]),
        ),
        nuisance=[[1.0], [1.0]],
    )

    assert certificate.status is QueryIdentifiabilityStatus.NONIDENTIFIABLE
    assert certificate.joint_certificate.nuisance_rank == 1
    assert certificate.joint_certificate.physical_rank == 0


def test_action_that_breaks_shared_nuisance_enables_identifiability() -> None:
    certificate = _certificate(
        (
            _block("action-a", [[1.0]], [[1.0]]),
            _block("action-b", [[2.0]], [[1.0]]),
        ),
        nuisance=[[1.0], [1.0]],
    )

    assert certificate.status is QueryIdentifiabilityStatus.IDENTIFIABLE
    contributions = {item.action_id: item for item in certificate.action_contributions}
    assert (
        contributions["action-b"].without_action_status
        is QueryIdentifiabilityStatus.NONIDENTIFIABLE
    )


def test_arrays_are_copied_immutable_and_content_addressed() -> None:
    physical = np.array([[1.0, 0.0]])
    block = _block("action-a", physical, [[1.0, 0.0]])
    certificate = _certificate(
        (
            block,
            _block("action-b", [[0.0, 1.0]], [[0.0, 1.0]]),
        ),
        metadata={"protocol": "target-closed"},
    )
    artifact_id = certificate.artifact_id
    physical[0, 0] = 9.0

    assert block.whitened_physical_design[0, 0] == 1.0
    for array in certificate.arrays().values():
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0.0
    assert certificate.artifact_id == artifact_id
    assert certificate.to_record()["artifact_id"] == artifact_id

    roundtrip = _certificate(
        (
            _block("action-a", [[1.0, 0.0]], [[1.0, 0.0]]),
            _block("action-b", [[0.0, 1.0]], [[0.0, 1.0]]),
        ),
        metadata={"protocol": "target-closed"},
        artifact_id=artifact_id,
    )
    assert roundtrip.artifact_id == artifact_id
    with pytest.raises(ValueError, match="artifact_id does not match content"):
        _certificate(
            (
                _block("action-a", [[1.0, 0.0]], [[1.0, 0.0]]),
                _block("action-b", [[0.0, 1.0]], [[0.0, 1.0]]),
            ),
            artifact_id="b" * 64,
        )


def test_summary_states_bounded_claim() -> None:
    certificate = _certificate(
        (
            _block("action-a", [[1.0, 0.0]], [[1.0, 1.0]]),
            _block("action-b", [[0.0, 1.0]], [[1.0, 1.0]]),
        )
    )

    summary = certificate.summary()
    assert summary["status"] == "identifiable"
    assert summary["requires_multiple_actions"] is True
    assert summary["claim_boundary"] == (
        MULTI_ACTION_QUERY_IDENTIFIABILITY_CLAIM_BOUNDARY
    )
    assert "safe action execution" in str(summary["claim_boundary"])


@pytest.mark.parametrize(
    ("blocks", "nuisance", "match"),
    [
        (
            (
                _block("action-a", [[1.0]], [[1.0]]),
                _block("action-a", [[1.0]], [[1.0]]),
            ),
            np.empty((2, 0)),
            "unique action_id",
        ),
        (
            (
                _block("action-b", [[1.0]], [[1.0]]),
                _block("action-a", [[1.0]], [[1.0]]),
            ),
            np.empty((2, 0)),
            "sorted by action_id",
        ),
        (
            (
                _block("action-a", [[1.0]], [[1.0]]),
                _block("action-b", [[1.0, 0.0]], [[1.0, 0.0]]),
            ),
            np.empty((2, 0)),
            "latent dimension",
        ),
        (
            (
                _block("action-a", [[1.0]], [[1.0]]),
                _block("action-b", [[1.0]], [[1.0]]),
            ),
            np.empty((1, 0)),
            "stacked observation row count",
        ),
    ],
)
def test_invalid_multi_action_designs_fail_closed(
    blocks: tuple[ActionIdentifiabilityBlockV1, ...],
    nuisance: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _certificate(blocks, nuisance=nuisance)


def test_invalid_action_block_fails_closed() -> None:
    with pytest.raises(ValueError, match="nonempty literal string"):
        _block("", [[1.0]], [[1.0]])
    with pytest.raises(ValueError, match="one column per latent"):
        _block("action-a", [[1.0, 0.0]], [[1.0]])
