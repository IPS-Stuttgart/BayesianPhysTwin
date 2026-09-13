"""Fail-closed act-or-fallback gate for symmetry-complete physical twins.

This module composes three independently registered contributions to a finite
action regret bound:

1. a complete-orbit structural bound supplied by Prob4D;
2. a shared-gauge intervention-realization margin supplied by Causal4D; and
3. an optional target-transport margin supplied by a separate calibration step.

The gate never selects a latent gauge representative.  It validates the
content-addressed Causal4D receipt, binds all evidence identities, recomputes the
realization margin, and either returns a minimax action within the registered
regret tolerance or the exact caller-owned fallback.

The implementation validates an algebraic evidence contract.  It does not prove
that the registered symmetry is physically correct, that the source-side bound
covers a new target, that a calibration assumption is valid, or that an admitted
action is safe.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

GATE_VERSION: Final = 1
RECEIPT_SCHEMA_VERSION: Final = 1
STATUS_EXACT: Final = "verified-exact-optimal"
STATUS_BOUNDED: Final = "verified-bounded-regret"
STATUS_REJECT: Final = "verified-reject-exact-fallback"
STATUS_INVALID: Final = "invalid-fail-closed"
_ALLOWED_RADIUS_SCOPES: Final = frozenset(
    {
        "deterministic-complete",
        "registered-group-nodes-only",
        "externally-calibrated",
    }
)


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty, unpadded string")
    return value


def _hex_digest(value: object, *, name: str) -> str:
    result = _string(value, name=name)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _real(value: object, *, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _readonly_float(value: object, *, name: str, ndim: int) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _readonly_bool(value: object) -> BoolArray:
    result = np.array(value, dtype=np.bool_, copy=True)
    result.setflags(write=False)
    return result


def _strict_keys(
    value: Mapping[str, Any],
    *,
    name: str,
    expected: frozenset[str],
) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            f"{name} fields do not match schema; "
            f"missing={missing}, unknown={unknown}"
        )


def _string_sequence(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty list of strings")
    result = tuple(
        _string(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique identifiers")
    return result


def _action_vector(
    value: object,
    *,
    name: str,
    action_count: int,
) -> FloatArray:
    result = _readonly_float(value, name=name, ndim=1)
    if result.shape != (action_count,):
        raise ValueError(f"{name} must have shape ({action_count},)")
    if np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _pairwise_matrix(
    value: object,
    *,
    name: str,
    action_count: int,
    nonnegative: bool,
    symmetric: bool,
    zero_diagonal: bool,
    atol: float,
) -> FloatArray:
    result = _readonly_float(value, name=name, ndim=2)
    expected = (action_count, action_count)
    if result.shape != expected:
        raise ValueError(f"{name} must have shape {expected}")
    if nonnegative and np.any(result < -atol):
        raise ValueError(f"{name} must be nonnegative")
    if symmetric and not np.allclose(
        result,
        result.T,
        rtol=0.0,
        atol=atol,
    ):
        raise ValueError(f"{name} must be symmetric")
    if zero_diagonal and not np.allclose(
        np.diag(result),
        0.0,
        rtol=0.0,
        atol=atol,
    ):
        raise ValueError(f"{name} must have a zero diagonal")
    return result


@dataclass(frozen=True)
class VerifiedInterventionReceiptV1:
    """Strictly parsed and independently checked Causal4D receipt."""

    receipt_id: str
    contract_id: str
    transform_instance_id: str
    state_evidence_id: str
    action_template_id: str
    commanded_intervention_id: str
    realized_intervention_id: str
    loss_id: str
    fallback_id: str
    radius_provenance_id: str
    radius_scope: str
    group_element_ids: tuple[str, ...]
    action_count: int
    observed_realization_radius_by_action: FloatArray
    declared_realization_radius_by_action: FloatArray
    action_loss_lipschitz_by_action: FloatArray
    action_realization_loss_margin: FloatArray
    pairwise_realization_margin: FloatArray
    verification_tolerance: float


@dataclass(frozen=True)
class SymmetryCompleteActionDecisionV1:
    """Complete audit record returned by the act-or-fallback gate."""

    schema_version: int
    status: str
    valid_evidence: bool
    invalid_reasons: tuple[str, ...]
    action_names: tuple[str, ...]
    structural_certificate_id: str
    intervention_receipt_id: str | None
    transport_certificate_id: str | None
    structural_pairwise_upper_bound: FloatArray
    realization_pairwise_margin: FloatArray
    transport_pairwise_margin: FloatArray
    total_pairwise_upper_bound: FloatArray
    worst_case_regret_upper_bound: FloatArray
    robustly_optimal: BoolArray
    epsilon_admissible: BoolArray
    minimax_action: int | None
    fallback_action: int
    selected_action: int
    selected_action_name: str
    admitted: bool
    exact_fallback: bool
    regret_tolerance: float
    radius_scope: str | None
    decision_record_id: str
    claim_boundary: str


_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "receipt_id",
        "contract_id",
        "transform_instance_id",
        "state_evidence_id",
        "action_template_id",
        "commanded_intervention_id",
        "realized_intervention_id",
        "loss_id",
        "fallback_id",
        "radius_provenance_id",
        "radius_scope",
        "group_element_ids",
        "action_templates",
        "commanded_action_orbit",
        "realized_action_orbit",
        "observed_realization_radius_by_action",
        "declared_realization_radius_by_action",
        "action_loss_lipschitz_by_action",
        "action_realization_loss_margin",
        "pairwise_realization_margin",
        "verification_tolerance",
    }
)


def verify_causal4d_intervention_receipt_v1(
    receipt: Mapping[str, Any],
    *,
    atol: float = 1e-12,
) -> VerifiedInterventionReceiptV1:
    """Verify the portable Causal4D receipt without importing Causal4D.

    The receipt ID is recomputed from the exact schema used by
    ``causal4d.shared_gauge_intervention``.  The command and realization arrays
    remain in the hash even though this consumer only needs their action count
    and the exported margin.  Tampering with any bound provenance therefore
    invalidates the receipt.
    """

    tolerance = _real(atol, name="atol")
    if not isinstance(receipt, Mapping):
        raise ValueError("receipt must be a mapping")
    _strict_keys(receipt, name="receipt", expected=_RECEIPT_FIELDS)
    schema_version = _integer(receipt["schema_version"], name="schema_version")
    if schema_version != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported Causal4D receipt schema version")
    receipt_id = _hex_digest(receipt["receipt_id"], name="receipt_id")
    contract_id = _hex_digest(receipt["contract_id"], name="contract_id")
    identifiers = {
        field: _string(receipt[field], name=field)
        for field in (
            "transform_instance_id",
            "state_evidence_id",
            "action_template_id",
            "commanded_intervention_id",
            "realized_intervention_id",
            "loss_id",
            "fallback_id",
            "radius_provenance_id",
        )
    }
    radius_scope = _string(receipt["radius_scope"], name="radius_scope")
    if radius_scope not in _ALLOWED_RADIUS_SCOPES:
        raise ValueError(f"unsupported radius_scope: {radius_scope!r}")
    element_ids = _string_sequence(
        receipt["group_element_ids"],
        name="group_element_ids",
    )
    action_templates = _readonly_float(
        receipt["action_templates"],
        name="action_templates",
        ndim=2,
    )
    action_count, action_dimension = action_templates.shape
    if action_count < 2 or action_dimension < 1:
        raise ValueError("action_templates must have shape (A, D), with A >= 2")
    command = _readonly_float(
        receipt["commanded_action_orbit"],
        name="commanded_action_orbit",
        ndim=3,
    )
    realized = _readonly_float(
        receipt["realized_action_orbit"],
        name="realized_action_orbit",
        ndim=3,
    )
    expected_orbit_shape = (
        len(element_ids),
        action_count,
        action_dimension,
    )
    if command.shape != expected_orbit_shape or realized.shape != expected_orbit_shape:
        raise ValueError(
            "commanded and realized action orbits do not match the registered "
            f"shape {expected_orbit_shape}"
        )
    observed = _action_vector(
        receipt["observed_realization_radius_by_action"],
        name="observed_realization_radius_by_action",
        action_count=action_count,
    )
    declared = _action_vector(
        receipt["declared_realization_radius_by_action"],
        name="declared_realization_radius_by_action",
        action_count=action_count,
    )
    if np.any(observed > declared + tolerance):
        raise ValueError("observed realization radius exceeds the declared radius")
    lipschitz = _action_vector(
        receipt["action_loss_lipschitz_by_action"],
        name="action_loss_lipschitz_by_action",
        action_count=action_count,
    )
    one_action = _action_vector(
        receipt["action_realization_loss_margin"],
        name="action_realization_loss_margin",
        action_count=action_count,
    )
    expected_one_action = declared * lipschitz
    if not np.allclose(
        one_action,
        expected_one_action,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("action realization loss margin is inconsistent")
    pairwise = _pairwise_matrix(
        receipt["pairwise_realization_margin"],
        name="pairwise_realization_margin",
        action_count=action_count,
        nonnegative=True,
        symmetric=True,
        zero_diagonal=True,
        atol=tolerance,
    )
    expected_pairwise = expected_one_action[:, None] + expected_one_action[None, :]
    diagonal = np.arange(action_count)
    expected_pairwise[diagonal, diagonal] = 0.0
    if not np.allclose(
        pairwise,
        expected_pairwise,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("pairwise realization margin is inconsistent")
    receipt_tolerance = _real(
        receipt["verification_tolerance"],
        name="verification_tolerance",
    )

    canonical_payload = {
        key: receipt[key]
        for key in _RECEIPT_FIELDS
        if key != "receipt_id"
    }
    if _canonical_digest(canonical_payload) != receipt_id:
        raise ValueError("receipt_id does not match the Causal4D receipt content")
    return VerifiedInterventionReceiptV1(
        receipt_id=receipt_id,
        contract_id=contract_id,
        transform_instance_id=identifiers["transform_instance_id"],
        state_evidence_id=identifiers["state_evidence_id"],
        action_template_id=identifiers["action_template_id"],
        commanded_intervention_id=identifiers["commanded_intervention_id"],
        realized_intervention_id=identifiers["realized_intervention_id"],
        loss_id=identifiers["loss_id"],
        fallback_id=identifiers["fallback_id"],
        radius_provenance_id=identifiers["radius_provenance_id"],
        radius_scope=radius_scope,
        group_element_ids=element_ids,
        action_count=action_count,
        observed_realization_radius_by_action=observed,
        declared_realization_radius_by_action=declared,
        action_loss_lipschitz_by_action=lipschitz,
        action_realization_loss_margin=one_action,
        pairwise_realization_margin=pairwise,
        verification_tolerance=receipt_tolerance,
    )


def _invalid_decision(
    *,
    reason: str,
    action_names: tuple[str, ...],
    structural_certificate_id: str,
    fallback_action: int,
    fallback_id: str,
    regret_tolerance: float,
    action_count: int,
) -> SymmetryCompleteActionDecisionV1:
    zero = np.zeros((action_count, action_count), dtype=np.float64)
    regret = np.full(action_count, math.inf, dtype=np.float64)
    false = np.zeros(action_count, dtype=np.bool_)
    zero.setflags(write=False)
    regret.setflags(write=False)
    false.setflags(write=False)
    payload = {
        "schema_version": GATE_VERSION,
        "status": STATUS_INVALID,
        "invalid_reasons": [reason],
        "action_names": list(action_names),
        "structural_certificate_id": structural_certificate_id,
        "fallback_action": fallback_action,
        "fallback_id": fallback_id,
        "regret_tolerance": regret_tolerance,
    }
    return SymmetryCompleteActionDecisionV1(
        schema_version=GATE_VERSION,
        status=STATUS_INVALID,
        valid_evidence=False,
        invalid_reasons=(reason,),
        action_names=action_names,
        structural_certificate_id=structural_certificate_id,
        intervention_receipt_id=None,
        transport_certificate_id=None,
        structural_pairwise_upper_bound=zero,
        realization_pairwise_margin=zero,
        transport_pairwise_margin=zero,
        total_pairwise_upper_bound=zero,
        worst_case_regret_upper_bound=regret,
        robustly_optimal=false,
        epsilon_admissible=false,
        minimax_action=None,
        fallback_action=fallback_action,
        selected_action=fallback_action,
        selected_action_name=action_names[fallback_action],
        admitted=False,
        exact_fallback=True,
        regret_tolerance=regret_tolerance,
        radius_scope=None,
        decision_record_id=_canonical_digest(payload),
        claim_boundary=(
            "Invalid evidence caused exact fallback. No structural, target-risk, "
            "physical-symmetry, actuator-validity, or safety claim is made."
        ),
    )


def act_or_fallback_symmetry_complete_v1(
    *,
    action_names: Sequence[str],
    structural_pairwise_upper_bound: object,
    structural_certificate_id: str,
    state_evidence_id: str,
    action_template_id: str,
    loss_id: str,
    fallback_action: int,
    fallback_id: str,
    causal4d_receipt: Mapping[str, Any],
    regret_tolerance: float = 0.0,
    transport_pairwise_margin: object | None = None,
    transport_certificate_id: str | None = None,
    require_radius_scope: str | None = None,
    atol: float = 1e-12,
) -> SymmetryCompleteActionDecisionV1:
    """Compose registered evidence and return one action or exact fallback.

    Malformed, tampered, or cross-context evidence produces
    ``invalid-fail-closed`` and returns the supplied fallback action.  A valid
    receipt whose total regret exceeds the tolerance produces
    ``verified-reject-exact-fallback``.  These outcomes are intentionally
    distinct.
    """

    names = tuple(
        _string(value, name=f"action_names[{index}]")
        for index, value in enumerate(action_names)
    )
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("action_names must contain at least two unique names")
    action_count = len(names)
    fallback = _integer(fallback_action, name="fallback_action")
    if not 0 <= fallback < action_count:
        raise ValueError("fallback_action must index action_names")
    fallback_name = _string(fallback_id, name="fallback_id")
    tolerance = _real(regret_tolerance, name="regret_tolerance")
    numerical_atol = _real(atol, name="atol")
    structural_id = _string(
        structural_certificate_id,
        name="structural_certificate_id",
    )
    expected_state_id = _string(state_evidence_id, name="state_evidence_id")
    expected_action_id = _string(action_template_id, name="action_template_id")
    expected_loss_id = _string(loss_id, name="loss_id")

    try:
        structural = _pairwise_matrix(
            structural_pairwise_upper_bound,
            name="structural_pairwise_upper_bound",
            action_count=action_count,
            nonnegative=False,
            symmetric=False,
            zero_diagonal=True,
            atol=numerical_atol,
        )
        receipt = verify_causal4d_intervention_receipt_v1(
            causal4d_receipt,
            atol=numerical_atol,
        )
        if receipt.action_count != action_count:
            raise ValueError("receipt action count does not match action_names")
        bindings = {
            "state_evidence_id": (receipt.state_evidence_id, expected_state_id),
            "action_template_id": (receipt.action_template_id, expected_action_id),
            "loss_id": (receipt.loss_id, expected_loss_id),
            "fallback_id": (receipt.fallback_id, fallback_name),
        }
        mismatched = [
            name
            for name, (actual, expected) in bindings.items()
            if actual != expected
        ]
        if mismatched:
            raise ValueError(
                "receipt evidence binding mismatch: " + ", ".join(mismatched)
            )
        if require_radius_scope is not None:
            required_scope = _string(
                require_radius_scope,
                name="require_radius_scope",
            )
            if required_scope not in _ALLOWED_RADIUS_SCOPES:
                raise ValueError("require_radius_scope is unsupported")
            if receipt.radius_scope != required_scope:
                raise ValueError(
                    "receipt radius scope does not meet the registered requirement"
                )
        if transport_pairwise_margin is None:
            transport = np.zeros_like(structural)
            transport.setflags(write=False)
            transport_id = None
        else:
            transport = _pairwise_matrix(
                transport_pairwise_margin,
                name="transport_pairwise_margin",
                action_count=action_count,
                nonnegative=True,
                symmetric=True,
                zero_diagonal=True,
                atol=numerical_atol,
            )
            if transport_certificate_id is None:
                raise ValueError(
                    "transport_certificate_id is required with a transport margin"
                )
            transport_id = _string(
                transport_certificate_id,
                name="transport_certificate_id",
            )
    except (KeyError, TypeError, ValueError) as error:
        return _invalid_decision(
            reason=str(error),
            action_names=names,
            structural_certificate_id=structural_id,
            fallback_action=fallback,
            fallback_id=fallback_name,
            regret_tolerance=tolerance,
            action_count=action_count,
        )

    total = np.array(
        structural + receipt.pairwise_realization_margin + transport,
        dtype=np.float64,
        copy=True,
    )
    diagonal = np.arange(action_count)
    total[diagonal, diagonal] = 0.0
    regret = np.max(total, axis=1)
    robust = np.all(total <= numerical_atol, axis=1)
    admissible_actions = regret <= tolerance + numerical_atol
    minimax = int(np.argmin(regret))
    admitted = bool(admissible_actions[minimax])
    selected = minimax if admitted else fallback
    exact_fallback = selected == fallback and not admitted
    if bool(robust[minimax]):
        status = STATUS_EXACT
    elif admitted:
        status = STATUS_BOUNDED
    else:
        status = STATUS_REJECT

    arrays = (
        total,
        regret,
        robust,
        admissible_actions,
    )
    for array in arrays:
        array.setflags(write=False)
    payload = {
        "schema_version": GATE_VERSION,
        "status": status,
        "action_names": list(names),
        "structural_certificate_id": structural_id,
        "intervention_receipt_id": receipt.receipt_id,
        "transport_certificate_id": transport_id,
        "structural_pairwise_upper_bound": structural.tolist(),
        "realization_pairwise_margin": (
            receipt.pairwise_realization_margin.tolist()
        ),
        "transport_pairwise_margin": transport.tolist(),
        "total_pairwise_upper_bound": total.tolist(),
        "worst_case_regret_upper_bound": regret.tolist(),
        "minimax_action": minimax,
        "fallback_action": fallback,
        "selected_action": selected,
        "regret_tolerance": tolerance,
        "radius_scope": receipt.radius_scope,
    }
    return SymmetryCompleteActionDecisionV1(
        schema_version=GATE_VERSION,
        status=status,
        valid_evidence=True,
        invalid_reasons=(),
        action_names=names,
        structural_certificate_id=structural_id,
        intervention_receipt_id=receipt.receipt_id,
        transport_certificate_id=transport_id,
        structural_pairwise_upper_bound=structural,
        realization_pairwise_margin=receipt.pairwise_realization_margin,
        transport_pairwise_margin=transport,
        total_pairwise_upper_bound=total,
        worst_case_regret_upper_bound=regret,
        robustly_optimal=_readonly_bool(robust),
        epsilon_admissible=_readonly_bool(admissible_actions),
        minimax_action=minimax,
        fallback_action=fallback,
        selected_action=selected,
        selected_action_name=names[selected],
        admitted=admitted,
        exact_fallback=exact_fallback,
        regret_tolerance=tolerance,
        radius_scope=receipt.radius_scope,
        decision_record_id=_canonical_digest(payload),
        claim_boundary=(
            "The result composes supplied structural, intervention, and optional "
            "transport bounds. It does not validate the physical symmetry, "
            "source-to-target assumptions, learned provider, actuator, or safety."
        ),
    )


__all__ = [
    "GATE_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "STATUS_BOUNDED",
    "STATUS_EXACT",
    "STATUS_INVALID",
    "STATUS_REJECT",
    "SymmetryCompleteActionDecisionV1",
    "VerifiedInterventionReceiptV1",
    "act_or_fallback_symmetry_complete_v1",
    "verify_causal4d_intervention_receipt_v1",
]
