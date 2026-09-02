from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin_experiments.interventional_cause_adequacy_v1 import (
    CAUSE_FAMILY_ADEQUACY_CLAIM_BOUNDARY,
    CauseBlockStatus,
    CauseFamilyAdequacyStatus,
    InterventionalCauseFamilyAdequacyV1,
)

SHA = "a" * 64


def _certificate(
    signatures: dict[str, object],
    residual: object,
    *,
    noise_radius: float = 1e-9,
    **kwargs: object,
) -> InterventionalCauseFamilyAdequacyV1:
    return InterventionalCauseFamilyAdequacyV1(
        residual_id=SHA,
        intervention_roster_id=SHA,
        whitening_id=SHA,
        cause_signature_ids={cause: SHA for cause in signatures},
        cause_signatures={
            cause: np.asarray(value, dtype=float) for cause, value in signatures.items()
        },
        whitened_residual=np.asarray(residual, dtype=float),
        noise_radius=noise_radius,
        **kwargs,
    )


def test_complete_family_recovers_unique_coefficients() -> None:
    certificate = _certificate(
        {"contact": [[1.0], [0.0]], "material": [[0.0], [1.0]]},
        [2.0, -3.0],
    )

    assert certificate.status is CauseFamilyAdequacyStatus.ADEQUATE_UNIQUE
    assert certificate.family_adequate
    assert certificate.unique_coefficients
    assert certificate.solution_nullity == 0
    assert certificate.unexplained_norm == pytest.approx(0.0, abs=1e-12)
    assert certificate.minimum_norm_coefficients == pytest.approx([2.0, -3.0])
    assert all(
        block.status is CauseBlockStatus.IDENTIFIABLE
        for block in certificate.cause_blocks
    )


def test_unregistered_residual_direction_returns_none_of_the_above() -> None:
    certificate = _certificate(
        {
            "contact": [[1.0], [0.0], [0.0]],
            "material": [[0.0], [1.0], [0.0]],
        },
        [0.0, 0.0, 1.0],
        noise_radius=0.01,
    )

    assert certificate.status is CauseFamilyAdequacyStatus.UNMODELED_CAUSE
    assert not certificate.family_adequate
    assert not certificate.attribution_permitted
    assert certificate.unexplained_norm == pytest.approx(1.0)
    assert certificate.explained_energy_fraction == pytest.approx(0.0)


def test_confounded_causes_return_complete_set_not_forced_label() -> None:
    certificate = _certificate(
        {"state": [[1.0], [0.0]], "gauge": [[1.0], [0.0]]},
        [1.0, 0.0],
    )

    assert certificate.status is CauseFamilyAdequacyStatus.ADEQUATE_SET_VALUED
    assert certificate.family_adequate
    assert not certificate.unique_coefficients
    assert certificate.solution_nullity == 1
    assert certificate.coefficient_nullspace.shape == (2, 1)
    assert all(
        block.status is CauseBlockStatus.CONFOUNDED
        for block in certificate.cause_blocks
    )


def test_one_dimension_of_multivariate_cause_can_remain_identifiable() -> None:
    certificate = _certificate(
        {
            "material": [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            "discrepancy": [[0.0], [1.0], [0.0]],
        },
        [2.0, 3.0, 0.0],
    )
    blocks = {block.cause_id: block for block in certificate.cause_blocks}

    assert certificate.status is CauseFamilyAdequacyStatus.ADEQUATE_SET_VALUED
    assert blocks["material"].status is CauseBlockStatus.PARTIALLY_IDENTIFIABLE
    assert blocks["material"].identifiable_dimension == 1
    assert blocks["material"].unresolved_dimension == 1
    assert blocks["discrepancy"].status is CauseBlockStatus.CONFOUNDED


def test_noise_radius_prevents_interpreting_negligible_residual() -> None:
    certificate = _certificate(
        {"state": [[1.0], [0.0]]},
        [0.005, 0.0],
        noise_radius=0.01,
    )

    assert certificate.status is CauseFamilyAdequacyStatus.NO_DETECTABLE_ERROR
    assert not certificate.attribution_permitted


def test_status_is_invariant_to_invertible_within_cause_coordinates() -> None:
    original = _certificate(
        {
            "material": [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            "bias": [[0.0], [0.0], [1.0]],
        },
        [2.0, -1.0, 0.5],
    )
    transform = np.asarray([[2.0, 1.0], [0.0, 0.5]])
    transformed = _certificate(
        {
            "material": np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]) @ transform,
            "bias": [[0.0], [0.0], [1.0]],
        },
        [2.0, -1.0, 0.5],
    )

    assert transformed.status is original.status
    assert transformed.design_rank == original.design_rank
    assert transformed.solution_nullity == original.solution_nullity
    assert transformed.unexplained_norm == pytest.approx(original.unexplained_norm)


def test_arrays_are_immutable_and_content_addressed() -> None:
    source = np.eye(2)
    certificate = _certificate(
        {"state": source[:, :1], "material": source[:, 1:]},
        [1.0, 2.0],
        metadata={"protocol": "target-closed"},
    )
    artifact_id = certificate.artifact_id
    source[:] = 7.0

    for value in certificate.arrays().values():
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flat[0] = 0.0
    assert certificate.artifact_id == artifact_id
    assert certificate.to_record()["claim_boundary"] == (
        CAUSE_FAMILY_ADEQUACY_CLAIM_BOUNDARY
    )

    roundtrip = _certificate(
        {"state": [[1.0], [0.0]], "material": [[0.0], [1.0]]},
        [1.0, 2.0],
        metadata={"protocol": "target-closed"},
        artifact_id=artifact_id,
    )
    assert roundtrip.artifact_id == artifact_id
    with pytest.raises(ValueError, match="artifact_id"):
        _certificate(
            {"state": [[1.0], [0.0]], "material": [[0.0], [1.0]]},
            [1.0, 2.0],
            artifact_id="b" * 64,
        )


@pytest.mark.parametrize(
    ("signatures", "residual", "match"),
    [
        ({}, [1.0], "nonempty"),
        (
            {"state": [[1.0], [0.0]], "bias": [[1.0]]},
            [1.0, 0.0],
            "rows",
        ),
        ({"state": [[1.0], [0.0]]}, [1.0], "row count"),
        ({"state": [[np.nan], [0.0]]}, [1.0, 0.0], "finite"),
    ],
)
def test_invalid_designs_fail_closed(
    signatures: dict[str, object],
    residual: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _certificate(signatures, residual)


def test_signature_identity_roster_is_exact() -> None:
    with pytest.raises(ValueError, match="cover exactly"):
        InterventionalCauseFamilyAdequacyV1(
            residual_id=SHA,
            intervention_roster_id=SHA,
            whitening_id=SHA,
            cause_signature_ids={"other": SHA},
            cause_signatures={"state": np.asarray([[1.0]])},
            whitened_residual=np.asarray([1.0]),
            noise_radius=0.1,
        )
