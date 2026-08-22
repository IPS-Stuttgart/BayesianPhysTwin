from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.query_estimability_certificate_v1 import (
    QUERY_ESTIMABILITY_CERTIFICATE_CLAIM_BOUNDARY,
    QUERY_ESTIMABILITY_CERTIFICATE_SCHEMA,
    QueryEstimabilityCertificateV1,
    QueryEstimabilityStatus,
)
from bayesian_phystwin.query_identifiability_certificate_v2 import (
    QueryIdentifiabilityCertificateV2,
)

SHA = "a" * 64
SCALE_SHA = "b" * 64


def _identifiability(
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


def _estimability(
    identifiability: QueryIdentifiabilityCertificateV2,
    scale: object,
    *,
    limit: float,
    **kwargs: object,
) -> QueryEstimabilityCertificateV1:
    return QueryEstimabilityCertificateV1(
        identifiability_certificate=identifiability,
        query_scale_id=SCALE_SHA,
        query_scale=np.asarray(scale),
        noise_gain_limit=limit,
        **kwargs,
    )


def test_well_conditioned_identifiable_query_is_stably_estimable() -> None:
    source = _identifiability(np.eye(2), np.eye(2))
    certificate = _estimability(source, np.eye(2), limit=1.01)

    assert certificate.status is QueryEstimabilityStatus.STABLY_ESTIMABLE
    assert certificate.identifiable
    assert certificate.stably_estimable
    assert certificate.passes_stability_gate
    assert certificate.maximum_normalized_noise_gain == pytest.approx(1.0)
    assert certificate.rms_normalized_noise_gain == pytest.approx(1.0)
    assert certificate.query_scale_condition_number == pytest.approx(1.0)
    assert certificate.noise_gain_margin == pytest.approx(0.01)
    np.testing.assert_allclose(certificate.query_noise_covariance, np.eye(2))
    np.testing.assert_allclose(
        certificate.normalized_query_noise_covariance,
        np.eye(2),
    )


def test_identifiable_weak_direction_is_rejected_as_unstable() -> None:
    source = _identifiability(
        np.diag([1.0, 1e-6]),
        np.eye(2),
    )
    certificate = _estimability(source, np.eye(2), limit=100.0)

    assert source.nontrivially_identifiable
    assert certificate.status is (
        QueryEstimabilityStatus.IDENTIFIABLE_BUT_UNSTABLE
    )
    assert certificate.identifiable
    assert not certificate.stably_estimable
    assert not certificate.passes_stability_gate
    assert certificate.maximum_normalized_noise_gain == pytest.approx(1e6)
    assert certificate.rms_normalized_noise_gain == pytest.approx(
        np.sqrt((1.0 + 1e12) / 2.0)
    )
    assert certificate.noise_gain_margin < 0.0


def test_query_ignoring_weak_latent_direction_remains_stable() -> None:
    source = _identifiability(
        np.diag([1.0, 1e-6]),
        [[1.0, 0.0]],
    )
    certificate = _estimability(source, [[1.0]], limit=1.1)

    assert certificate.status is QueryEstimabilityStatus.STABLY_ESTIMABLE
    assert certificate.maximum_normalized_noise_gain == pytest.approx(1.0)
    assert certificate.rms_normalized_noise_gain == pytest.approx(1.0)


def test_nuisance_projection_and_query_scale_define_dimensionless_gain() -> None:
    source = _identifiability(
        [[1.0], [1.0]],
        [[2.0]],
        [[1.0], [0.0]],
    )
    certificate = _estimability(source, [[4.0]], limit=1.1)

    assert certificate.status is QueryEstimabilityStatus.STABLY_ESTIMABLE
    assert certificate.nuisance_leakage_frobenius == pytest.approx(
        0.0,
        abs=1e-12,
    )
    np.testing.assert_allclose(
        certificate.effective_factor_operator,
        [[0.0, 2.0]],
        atol=1e-12,
    )
    np.testing.assert_allclose(certificate.query_noise_covariance, [[4.0]])
    np.testing.assert_allclose(
        certificate.normalized_query_noise_covariance,
        [[1.0]],
    )
    assert certificate.maximum_normalized_noise_gain == pytest.approx(1.0)


def test_consistent_query_coordinate_rescaling_preserves_noise_gains() -> None:
    physical = np.eye(2)
    query = np.eye(2)
    scale = np.diag([4.0, 9.0])
    transform = np.diag([10.0, 0.1])

    original = _estimability(
        _identifiability(physical, query),
        scale,
        limit=0.6,
    )
    transformed = _estimability(
        _identifiability(physical, transform @ query),
        transform @ scale @ transform.T,
        limit=0.6,
    )

    assert original.status is transformed.status
    assert transformed.maximum_normalized_noise_gain == pytest.approx(
        original.maximum_normalized_noise_gain
    )
    assert transformed.rms_normalized_noise_gain == pytest.approx(
        original.rms_normalized_noise_gain
    )
    assert original.artifact_id != transformed.artifact_id


def test_nonidentifiable_source_fails_closed_even_when_gain_is_zero() -> None:
    source = _identifiability(
        [[1.0, 0.0], [2.0, 0.0]],
        [[0.0, 1.0]],
    )
    certificate = _estimability(source, [[1.0]], limit=1e9)

    assert certificate.status is QueryEstimabilityStatus.NONIDENTIFIABLE
    assert not certificate.identifiable
    assert not certificate.stably_estimable
    assert certificate.maximum_normalized_noise_gain == pytest.approx(0.0)


def test_trivial_query_is_explicit_and_not_an_admitted_update() -> None:
    source = _identifiability([[1.0, 0.0]], [[0.0, 0.0]])
    certificate = _estimability(source, [[1.0]], limit=0.0)

    assert certificate.status is QueryEstimabilityStatus.TRIVIAL_QUERY
    assert certificate.identifiable
    assert not certificate.stably_estimable
    assert certificate.maximum_normalized_noise_gain == pytest.approx(0.0)


def test_arrays_are_copied_immutable_and_content_addressed() -> None:
    source = _identifiability(np.eye(2), np.eye(2))
    scale = np.eye(2)
    certificate = _estimability(
        source,
        scale,
        limit=1.1,
        metadata={"protocol": "source-only"},
    )
    artifact_id = certificate.artifact_id
    scale[0, 0] = 9.0

    assert certificate.query_scale[0, 0] == 1.0
    for value in certificate.arrays().values():
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flat[0] = 0.0
    assert certificate.artifact_id == artifact_id
    assert certificate.to_record()["artifact_id"] == artifact_id

    roundtrip = _estimability(
        source,
        np.eye(2),
        limit=1.1,
        metadata={"protocol": "source-only"},
        artifact_id=artifact_id,
    )
    assert roundtrip.artifact_id == artifact_id
    with pytest.raises(ValueError, match="artifact_id does not match content"):
        _estimability(source, np.eye(2), limit=1.1, artifact_id="c" * 64)


def test_summary_binds_source_and_states_local_claim_boundary() -> None:
    source = _identifiability(np.eye(2), np.eye(2))
    certificate = _estimability(source, np.eye(2), limit=1.1)
    summary = certificate.summary()

    assert summary["schema"] == QUERY_ESTIMABILITY_CERTIFICATE_SCHEMA
    assert summary["identifiability_certificate_id"] == source.artifact_id
    assert summary["status"] == "stably_estimable"
    assert summary["claim_boundary"] == (
        QUERY_ESTIMABILITY_CERTIFICATE_CLAIM_BOUNDARY
    )
    assert "uncertainty calibration" in str(summary["claim_boundary"])


@pytest.mark.parametrize(
    ("scale", "match"),
    [
        ([1.0, 0.0], "must be a matrix"),
        ([[1.0, 0.0]], "square"),
        ([[1.0, 0.1], [0.0, 1.0]], "exactly symmetric"),
        ([[1.0, 0.0], [0.0, 0.0]], "positive definite"),
        ([[1.0, np.nan], [np.nan, 1.0]], "must be finite"),
        ([[True, False], [False, True]], "real numeric"),
    ],
)
def test_invalid_query_scales_fail_closed(scale: object, match: str) -> None:
    source = _identifiability(np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match=match):
        _estimability(source, scale, limit=1.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"noise_gain_limit": -1.0}, "finite nonnegative"),
        ({"noise_gain_limit": np.inf}, "finite nonnegative"),
        (
            {
                "relative_scale_tolerance": 0.0,
                "absolute_scale_tolerance": 0.0,
            },
            "at least one query-scale tolerance",
        ),
        ({"relative_scale_tolerance": True}, "finite nonnegative"),
        ({"stability_tolerance": -1.0}, "finite nonnegative"),
    ],
)
def test_invalid_policy_values_fail_closed(
    kwargs: dict[str, object],
    match: str,
) -> None:
    source = _identifiability(np.eye(2), np.eye(2))
    limit = kwargs.get("noise_gain_limit", 1.0)
    remaining = {
        key: value for key, value in kwargs.items() if key != "noise_gain_limit"
    }
    with pytest.raises(ValueError, match=match):
        _estimability(source, np.eye(2), limit=limit, **remaining)


def test_invalid_source_type_and_identity_fail_closed() -> None:
    with pytest.raises(ValueError, match="QueryIdentifiabilityCertificateV2"):
        QueryEstimabilityCertificateV1(
            identifiability_certificate=object(),  # type: ignore[arg-type]
            query_scale_id=SCALE_SHA,
            query_scale=np.eye(1),
            noise_gain_limit=1.0,
        )

    source = _identifiability([[1.0]], [[1.0]])
    with pytest.raises(ValueError, match="query_scale_id"):
        QueryEstimabilityCertificateV1(
            identifiability_certificate=source,
            query_scale_id="not-a-sha",
            query_scale=np.eye(1),
            noise_gain_limit=1.0,
        )
