"""Shared definitions for the cross-action physicality v1 contract."""

from __future__ import annotations

from enum import Enum
from typing import Final, cast

import numpy as np

from bayesian_phystwin._canonical_contracts import genuine_integer, literal_lower_hex
from bayesian_phystwin._portable_contracts import content_id

CROSS_ACTION_PHYSICALITY_SCHEMA: Final = "bayesian_phystwin.cross_action_physicality"
CROSS_ACTION_PHYSICALITY_VERSION: Final = 1
CROSS_ACTION_PHYSICALITY_SEMANTICS: Final = (
    "target-blind-broken-mechanism-placebo-session-inference-v1"
)
FAMILYWISE_METHOD: Final = "paired-bonferroni-percentile-bootstrap-lower-v1"
FAMILYWISE_METHOD_ID: Final = cast(
    str,
    content_id({"cross_action_physicality_familywise_method": FAMILYWISE_METHOD}),
)
CROSS_ACTION_PHYSICALITY_CLAIM_BOUNDARY: Final = (
    "A positive result establishes bounded separation of the exact registered "
    "source-admitted guarded physical prediction from four target-closed broken-"
    "mechanism controls on the exact chronological physical-session roster. It "
    "does not establish a unique physical cause, unseen-object or arbitrary-action "
    "generalization, calibrated raw covariance, real Prob4D provider competence, "
    "Causal4D intervention benefit, deployment safety, or state of the art."
)


class BrokenMechanismPolicy(str, Enum):
    """Required target-closed placebo constructions."""

    WRONG_SOURCE_ACTION = "wrong_source_action"
    WRONG_OBJECT_SESSION = "wrong_object_session"
    PHASE_SHIFTED_SOURCE = "phase_shifted_source"
    IDENTITY_PERMUTED = "identity_permuted"


REQUIRED_PLACEBO_POLICIES: Final = tuple(
    sorted(BrokenMechanismPolicy, key=lambda policy: policy.value)
)


class PhysicalityDecision(str, Enum):
    """Registered physicality-certificate decision."""

    SUPPORTED = "physicality_supported"
    NOT_SUPPORTED = "physicality_not_supported"
    PARENT_NOT_SUPPORTED = "parent_transport_not_supported"
    INSUFFICIENT = "insufficient_physicality_evidence"


def _digest(value: object, *, name: str) -> str:
    return cast(str, literal_lower_hex(value, name=name, lengths={64}))


def _commit(value: object, *, name: str) -> str:
    return cast(str, literal_lower_hex(value, name=name, lengths={40, 64}))


def _literal(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


def _optional_literal(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _literal(value, name=name)


def _finite(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real number")
    result = float(raw.item())
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _probability(value: object, *, name: str) -> float:
    result = _finite(value, name=name, minimum=0.0, maximum=1.0)
    if result in {0.0, 1.0}:
        raise ValueError(f"{name} must be strictly between zero and one")
    return result


def _optional_nonzero_integer(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    result = genuine_integer(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be a nonzero integer")
    return result
