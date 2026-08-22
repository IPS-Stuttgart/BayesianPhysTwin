from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.query_identifiability_certificate_v2 import (
    QUERY_IDENTIFIABILITY_CERTIFICATE_CLAIM_BOUNDARY,
    QUERY_IDENTIFIABILITY_CERTIFICATE_SCHEMA,
    QueryIdentifiabilityCertificateV2,
    QueryIdentifiabilityStatus,
)

SHA = "a" * 64


def _certificate(
    physical: object,
    query: object,
    nuisance: object | None = None,
    **kwargs: object,
) -> QueryIdentifiabilityCertificateV2:
    physical_array = np.asarray(physical)
    if nuisance is None:
        nuisance = np.empty((physical_array.shape[0], 0))
    return QueryIdentifiabilityCertificateV2(
        physical_response_id=SHA,
        observation_mapping_id=SHA,
        nuisance_design_id=SHA,
        query_id=SHA,
        whitened_physical_design=physical_array,
        whitened_nuisance_design=np.asarray(nuisance),
        query_map=np.asarray(query),
        **kwargs,
    )


def test_identifiable_query_factorizes_through_residualized_design() -> None:
    certificate = _certificate(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        [[1.0, 2.0], [0.0, -1.0]],
    )

    assert certificate.status is QueryIdentifiabilityStatus.IDENTIFIABLE
    assert certificate.identifiable
    assert certificate.nontrivially_identifiable
    assert certificate.physical_rank == 2
    assert certificate.physical_nullity == 0
    assert certificate.rank_increment == 0
    assert certificate.augmented_rank == 2
    assert certificate.normalized_factorization_residual == pytest.approx(0.0)
    assert certificate.identifiable_query_energy_fraction == pytest.approx(1.0)
    np.testing.assert_allclose(
        certificate.factor_operator @ certificate.residualized_physical_design,
        certificate.query_map,
        atol=1e-12,
    )


def test_hidden_latent_direction_is_nonidentifiable() -> None:
    certificate = _certificate(
        [[1.0, 0.0], [2.0, 0.0]],
        [[0.0, 1.0]],
    )

    assert certificate.status is QueryIdentifiabilityStatus.NONIDENTIFIABLE
    assert not certificate.identifiable
    assert not certificate.nontrivially_identifiable
    assert certificate.physical_rank == 1
    assert certificate.physical_nullity == 1
    assert certificate.rank_increment == 1
    assert certificate.augmented_rank == 2
    assert certificate.normalized_factorization_residual == pytest.approx(1.0)
    assert certificate.identifiable_query_energy_fraction == pytest.approx(0.0)


def test_query_that_ignores_hidden_direction_remains_identifiable() -> None:
    certificate = _certificate(
        [[1.0, 0.0], [2.0, 0.0]],
        [[3.0, 0.0]],
    )

    assert certificate.status is QueryIdentifiabilityStatus.IDENTIFIABLE
    assert certificate.physical_nullity == 1
    assert certificate.rank_increment == 0
    np.testing.assert_allclose(
        certificate.factor_operator @ certificate.residualized_physical_design,
        certificate.query_map,
        atol=1e-12,
    )


def test_declared_nuisance_can_remove_apparent_physical_support() -> None:
    certificate = _certificate(
        np.eye(2),
        [[1.0, 0.0]],
        [[1.0], [0.0]],
    )

    assert certificate.nuisance_rank == 1
    assert certificate.physical_rank == 1
    assert certificate.status is QueryIdentifiabilityStatus.NONIDENTIFIABLE
    np.testing.assert_allclose(
        certificate.residualized_physical_design,
        [[0.0, 0.0], [0.0, 1.0]],
        atol=1e-12,
    )


def test_invertible_nuisance_basis_change_preserves_projection_and_decision() -> None:
    physical = np.array([[1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])
    nuisance = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    query = np.array([[0.0, 1.0]])
    transform = np.array([[2.0, 1.0], [1.0, 1.0]])

    original = _certificate(physical, query, nuisance)
    reparameterized = _certificate(physical, query, nuisance @ transform)

    assert original.status is reparameterized.status
    np.testing.assert_allclose(
        original.nuisance_projector,
        reparameterized.nuisance_projector,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        original.residualized_physical_design,
        reparameterized.residualized_physical_design,
        atol=1e-12,
    )
    assert original.artifact_id != reparameterized.artifact_id


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ([[1.0, 0.0]], QueryIdentifiabilityStatus.IDENTIFIABLE),
        ([[0.0, 1.0]], QueryIdentifiabilityStatus.NONIDENTIFIABLE),
    ],
)
def test_invertible_latent_reparameterization_preserves_exact_status(
    query: list[list[float]],
    expected: QueryIdentifiabilityStatus,
) -> None:
    physical = np.array([[1.0, 0.0]])
    query_array = np.asarray(query)
    transform = np.array([[2.0, 1.0], [0.0, 1.0]])

    original = _certificate(physical, query_array)
    reparameterized = _certificate(
        physical @ transform,
        query_array @ transform,
    )

    assert original.status is expected
    assert reparameterized.status is expected


def test_zero_query_is_explicitly_trivial() -> None:
    certificate = _certificate(
        [[1.0, 0.0]],
        [[0.0, 0.0]],
    )

    assert certificate.status is QueryIdentifiabilityStatus.TRIVIAL_QUERY
    assert certificate.identifiable
    assert not certificate.nontrivially_identifiable
    assert certificate.query_rank == 0
    assert certificate.normalized_factorization_residual == pytest.approx(0.0)
    assert certificate.identifiable_query_energy_fraction == pytest.approx(1.0)


def test_arrays_are_copied_immutable_and_content_addressed() -> None:
    physical = np.eye(2)
    certificate = _certificate(
        physical,
        [[1.0, 0.0]],
        metadata={"protocol": "source-only"},
    )
    artifact_id = certificate.artifact_id
    physical[0, 0] = 9.0

    assert certificate.whitened_physical_design[0, 0] == 1.0
    for value in certificate.arrays().values():
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flat[0] = 0.0
    assert certificate.artifact_id == artifact_id
    assert certificate.to_record()["artifact_id"] == artifact_id

    roundtrip = _certificate(
        np.eye(2),
        [[1.0, 0.0]],
        metadata={"protocol": "source-only"},
        artifact_id=artifact_id,
    )
    assert roundtrip.artifact_id == artifact_id
    with pytest.raises(ValueError, match="artifact_id does not match content"):
        _certificate(np.eye(2), [[1.0, 0.0]], artifact_id="b" * 64)


def test_summary_states_the_local_claim_boundary() -> None:
    certificate = _certificate(np.eye(2), [[1.0, 0.0]])
    summary = certificate.summary()

    assert summary["schema"] == QUERY_IDENTIFIABILITY_CERTIFICATE_SCHEMA
    assert summary["status"] == "identifiable"
    assert summary["claim_boundary"] == (
        QUERY_IDENTIFIABILITY_CERTIFICATE_CLAIM_BOUNDARY
    )
    assert "global nonlinear identifiability" in str(summary["claim_boundary"])


@pytest.mark.parametrize(
    ("physical", "nuisance", "query", "match"),
    [
        ([[1.0, 0.0]], [[1.0], [0.0]], [[1.0, 0.0]], "row count"),
        ([[1.0, 0.0]], [[]], [[1.0]], "one column per latent"),
        (
            np.empty((0, 1)),
            np.empty((0, 0)),
            [[1.0]],
            "must have nonzero dimensions",
        ),
        ([[1.0, np.nan]], [[]], [[1.0, 0.0]], "must be finite"),
        ([[True, False]], [[]], [[1.0, 0.0]], "real numeric"),
    ],
)
def test_invalid_designs_fail_closed(
    physical: object,
    nuisance: object,
    query: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _certificate(physical, query, nuisance)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"relative_rank_tolerance": 0.0, "absolute_rank_tolerance": 0.0},
            "at least one rank tolerance",
        ),
        ({"identifiability_tolerance": 0.0}, "must be positive"),
        ({"relative_rank_tolerance": True}, "finite nonnegative"),
        ({"absolute_rank_tolerance": -1.0}, "finite nonnegative"),
    ],
)
def test_invalid_tolerances_fail_closed(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _certificate(np.eye(2), [[1.0, 0.0]], **kwargs)
