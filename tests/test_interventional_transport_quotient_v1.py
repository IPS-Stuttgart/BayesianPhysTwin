from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin_experiments.interventional_cause_adequacy_v1 import (
    CauseFamilyAdequacyStatus,
    InterventionalCauseFamilyAdequacyV1,
)
from bayesian_phystwin_experiments.interventional_transport_quotient_v1 import (
    TRANSPORT_QUOTIENT_CLAIM_BOUNDARY,
    InterventionalTransportQuotientV1,
    TransportQuotientStatus,
)

SHA = "a" * 64


def _adequacy(
    signatures: dict[str, object],
    residual: object,
    *,
    noise_radius: float = 1e-9,
) -> InterventionalCauseFamilyAdequacyV1:
    return InterventionalCauseFamilyAdequacyV1(
        residual_id=SHA,
        intervention_roster_id=SHA,
        whitening_id=SHA,
        cause_signature_ids={cause: SHA for cause in signatures},
        cause_signatures={
            cause: np.asarray(value, dtype=float)
            for cause, value in signatures.items()
        },
        whitened_residual=np.asarray(residual, dtype=float),
        noise_radius=noise_radius,
    )


def _quotient(
    adequacy: InterventionalCauseFamilyAdequacyV1,
    targets: dict[str, object],
    **kwargs: object,
) -> InterventionalTransportQuotientV1:
    return InterventionalTransportQuotientV1(
        adequacy_certificate=adequacy,
        target_intervention_roster_id=SHA,
        target_transport_ids={target: SHA for target in targets},
        target_maps={
            target: np.asarray(value, dtype=float)
            for target, value in targets.items()
        },
        **kwargs,
    )


def test_confounded_causes_can_have_fully_identifiable_transport() -> None:
    adequacy = _adequacy(
        {"state": [[1.0]], "gauge": [[1.0]]},
        [3.0],
    )
    quotient = _quotient(adequacy, {"held-action": [[1.0, 1.0]]})
    record = quotient.record_for("held-action")

    assert adequacy.status is CauseFamilyAdequacyStatus.ADEQUATE_SET_VALUED
    assert record.status is TransportQuotientStatus.FULLY_IDENTIFIABLE
    assert record.full_transport_permitted
    assert record.identifiable_dimension == 1
    assert record.ambiguity_dimension == 0
    assert record.identifiable_effect == pytest.approx([3.0])
    assert record.representative_invariance_residual == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_cause_sensitive_transport_remains_nonidentifiable() -> None:
    adequacy = _adequacy(
        {"state": [[1.0]], "gauge": [[1.0]]},
        [3.0],
    )
    quotient = _quotient(adequacy, {"held-action": [[1.0, 0.0]]})
    record = quotient.record_for("held-action")

    assert record.status is TransportQuotientStatus.NONIDENTIFIABLE
    assert not record.full_transport_permitted
    assert not record.partial_transport_available
    assert record.identifiable_dimension == 0
    assert record.ambiguity_dimension == 1
    assert record.identifiable_effect == pytest.approx([0.0])


def test_vector_target_can_be_partially_identifiable() -> None:
    adequacy = _adequacy(
        {"state": [[1.0]], "gauge": [[1.0]]},
        [2.0],
    )
    quotient = _quotient(adequacy, {"two-coordinate-query": np.eye(2)})
    record = quotient.record_for("two-coordinate-query")

    assert record.status is TransportQuotientStatus.PARTIALLY_IDENTIFIABLE
    assert not record.full_transport_permitted
    assert record.partial_transport_available
    assert record.identifiable_dimension == 1
    assert record.ambiguity_dimension == 1
    assert record.identifiable_energy_fraction == pytest.approx(0.5)
    assert record.identifiable_projector @ np.asarray([1.0, -1.0]) == (
        pytest.approx([0.0, 0.0], abs=1e-12)
    )


def test_identifiable_effect_is_invariant_to_affine_representative() -> None:
    adequacy = _adequacy(
        {"state": [[1.0]], "gauge": [[1.0]]},
        [4.0],
    )
    quotient = _quotient(adequacy, {"held-action": [[1.0, 1.0]]})
    record = quotient.record_for("held-action")
    alternative = (
        adequacy.minimum_norm_coefficients
        + 7.0 * adequacy.coefficient_nullspace[:, 0]
    )

    assert record.identifiable_effect == pytest.approx(
        record.identifiable_projector
        @ quotient.target_maps["held-action"]
        @ alternative
    )


def test_family_inadequacy_blocks_transport_even_for_invariant_map() -> None:
    adequacy = _adequacy(
        {
            "state": [[1.0], [0.0]],
            "gauge": [[1.0], [0.0]],
        },
        [1.0, 1.0],
        noise_radius=0.01,
    )
    quotient = _quotient(adequacy, {"held-action": [[1.0, 1.0]]})
    record = quotient.record_for("held-action")

    assert adequacy.status is CauseFamilyAdequacyStatus.UNMODELED_CAUSE
    assert record.status is TransportQuotientStatus.FAMILY_INADEQUATE
    assert not record.full_transport_permitted
    assert not record.partial_transport_available


def test_no_detectable_error_returns_exact_nontransport_status() -> None:
    adequacy = _adequacy(
        {"state": [[1.0], [0.0]]},
        [0.001, 0.0],
        noise_radius=0.01,
    )
    quotient = _quotient(adequacy, {"held-action": [[1.0]]})
    record = quotient.record_for("held-action")

    assert adequacy.status is CauseFamilyAdequacyStatus.NO_DETECTABLE_ERROR
    assert record.status is TransportQuotientStatus.NO_DETECTABLE_ERROR
    assert not record.full_transport_permitted
    assert record.identifiable_effect == pytest.approx([0.0])


def test_noise_bound_controls_identifiable_target_perturbation() -> None:
    base = _adequacy(
        {"state": [[2.0], [0.0]]},
        [4.0, 0.0],
        noise_radius=0.2,
    )
    quotient = _quotient(base, {"held-action": [[3.0]]})
    record = quotient.record_for("held-action")
    perturbation = np.asarray([0.2, 0.0])
    changed = _adequacy(
        {"state": [[2.0], [0.0]]},
        np.asarray([4.0, 0.0]) + perturbation,
        noise_radius=0.2,
    )
    changed_record = _quotient(
        changed,
        {"held-action": [[3.0]]},
    ).record_for("held-action")

    actual = float(
        np.linalg.norm(
            changed_record.identifiable_effect - record.identifiable_effect
        )
    )
    assert actual <= record.noise_error_bound + 1e-12
    assert record.stability_gain == pytest.approx(1.5)
    assert record.noise_error_bound == pytest.approx(0.3)


def test_arrays_are_immutable_and_content_addressed() -> None:
    target = np.asarray([[1.0, 1.0]])
    quotient = _quotient(
        _adequacy({"state": [[1.0]], "gauge": [[1.0]]}, [2.0]),
        {"held-action": target},
        metadata={"protocol": "target-closed"},
    )
    artifact_id = quotient.artifact_id
    target[:] = 9.0

    for value in quotient.arrays().values():
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flat[0] = 0.0
    assert quotient.target_maps["held-action"] == pytest.approx([[1.0, 1.0]])
    assert quotient.artifact_id == artifact_id
    assert quotient.to_record()["claim_boundary"] == (
        TRANSPORT_QUOTIENT_CLAIM_BOUNDARY
    )

    roundtrip = _quotient(
        _adequacy({"state": [[1.0]], "gauge": [[1.0]]}, [2.0]),
        {"held-action": [[1.0, 1.0]]},
        metadata={"protocol": "target-closed"},
        artifact_id=artifact_id,
    )
    assert roundtrip.artifact_id == artifact_id
    with pytest.raises(ValueError, match="artifact_id"):
        _quotient(
            _adequacy({"state": [[1.0]], "gauge": [[1.0]]}, [2.0]),
            {"held-action": [[1.0, 1.0]]},
            artifact_id="b" * 64,
        )


def test_invalid_target_rosters_fail_closed() -> None:
    adequacy = _adequacy({"state": [[1.0]]}, [1.0])
    with pytest.raises(ValueError, match="cover exactly"):
        InterventionalTransportQuotientV1(
            adequacy_certificate=adequacy,
            target_intervention_roster_id=SHA,
            target_transport_ids={"other": SHA},
            target_maps={"held-action": np.asarray([[1.0]])},
        )
    with pytest.raises(ValueError, match="one column"):
        _quotient(adequacy, {"held-action": [[1.0, 0.0]]})
    with pytest.raises(ValueError, match="nontrivial"):
        _quotient(adequacy, {"held-action": [[0.0]]})
