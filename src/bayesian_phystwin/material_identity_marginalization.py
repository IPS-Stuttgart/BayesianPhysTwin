"""Fail-closed Bayesian state marginalization over Prob4D identity hypotheses.

Prob4D owns the portable source-side material-identity mixture. This module
independently validates that JSON contract without importing Prob4D, binds every
candidate inference result to one common state domain, and applies the law of
total covariance only after all candidates pass the BayesianPhysTwin inference
boundary. Any inadmissible candidate or impossible likelihood set returns the
exact newest-window (null-hypothesis) state posterior.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)

FloatArray = NDArray[np.float64]

PROB4D_MATERIAL_IDENTITY_MIXTURE_SCHEMA = "prob4d.material-identity-mixture"
PROB4D_MATERIAL_IDENTITY_MIXTURE_VERSION = 1
PROB4D_MATERIAL_IDENTITY_HYPOTHESIS_SCHEMA = "prob4d.material-identity-hypothesis"
PROB4D_MATERIAL_IDENTITY_HYPOTHESIS_VERSION = 1
PROB4D_MATERIAL_IDENTITY_WEIGHT_SEMANTICS = "source-calibrated-log-weight-v1"
PROB4D_MATERIAL_IDENTITY_NULL_SEMANTICS = "newest-window-local-reference-v1"
PROB4D_MATERIAL_IDENTITY_CLAIM_BOUNDARY = (
    "Source-calibrated cross-window material-identity hypotheses only. Endpoints "
    "remain window-local, the null hypothesis preserves the newest-window "
    "reference exactly, and no physical-state update or Causal4D benefit is "
    "established by this artifact."
)

IDENTITY_LIKELIHOOD_EVIDENCE_SCHEMA = (
    "bayesian_phystwin.material_identity_likelihood_evidence"
)
IDENTITY_LIKELIHOOD_EVIDENCE_VERSION = 1
IDENTITY_LIKELIHOOD_SEMANTICS = "prefix-only-candidate-log-likelihood-v1"
IDENTITY_STATE_POSTERIOR_SCHEMA = "bayesian_phystwin.material_identity_state_posterior"
IDENTITY_STATE_POSTERIOR_VERSION = 1
IDENTITY_MARGINALIZATION_SEMANTICS = "source-prior-times-prefix-likelihood-v1"
IDENTITY_STATE_MOMENT_SEMANTICS = "common-state-law-of-total-covariance-v1"

_LINEAGE_MIXTURE_ID = "prob4d_material_identity_mixture_id"
_LINEAGE_CANDIDATE_ID = "prob4d_material_identity_candidate_id"
_LINEAGE_COMMON_STATE_DOMAIN_ID = "material_identity_common_state_domain_id"
_LINEAGE_CAUSAL_FRAME_STOP = "prob4d_material_identity_causal_frame_stop"
_LINEAGE_SOURCE_CALIBRATION_ID = "prob4d_material_identity_calibration_id"
_RESERVED_LINEAGE_FIELDS = frozenset(
    {
        _LINEAGE_MIXTURE_ID,
        _LINEAGE_CANDIDATE_ID,
        _LINEAGE_COMMON_STATE_DOMAIN_ID,
        _LINEAGE_CAUSAL_FRAME_STOP,
        _LINEAGE_SOURCE_CALIBRATION_ID,
    }
)

_MIXTURE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "mixture_id",
        "target_endpoint",
        "window_order",
        "causal_frame_stop",
        "association_rule_id",
        "calibration_id",
        "tracklet_producer_revision",
        "association_revision",
        "weight_semantics",
        "null_hypothesis_semantics",
        "candidates",
        "metadata",
        "claim_boundary",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "kind",
        "source_endpoint",
        "association_result_id",
        "source_score",
        "calibrated_log_weight",
        "metadata",
    }
)
_ENDPOINT_FIELDS = frozenset({"window_id", "track_id"})


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _array_id(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    header = json.dumps(
        {"dtype": "<f8", "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def _load_strict_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read material-identity mixture {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("material-identity mixture root must be a JSON object")
    return payload


def _require_exact_fields(
    value: Mapping[str, Any],
    *,
    expected: frozenset[str],
    name: str,
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{name} fields changed: missing={missing}, extra={extra}")


def _nonempty_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256(value: object, *, name: str) -> str:
    digest = _nonempty_string(value, name=name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _revision(value: object, *, name: str) -> str:
    revision = _nonempty_string(value, name=name)
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"{name} must be an exact lowercase Git revision")
    return revision


def _finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _readonly_float(value: np.ndarray) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _logsumexp(values: FloatArray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("logsumexp values must be a non-empty vector")
    if np.any(np.isnan(array)) or np.any(np.isposinf(array)):
        raise ValueError("logsumexp values may not contain NaN or positive infinity")
    maximum = float(np.max(array))
    if np.isneginf(maximum):
        return float("-inf")
    return float(maximum + np.log(np.sum(np.exp(array - maximum))))


def _validated_state_covariance(value: np.ndarray, *, name: str) -> FloatArray:
    covariance = np.asarray(value, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    if len(covariance):
        scale = max(1.0, float(np.linalg.norm(covariance, ord=2)))
        if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-10 * scale:
            raise ValueError(f"{name} must be positive semidefinite")
    return _readonly_float(covariance)


@dataclass(frozen=True, order=True)
class Prob4DLocalTrackEndpointV1:
    """Window-local Prob4D track endpoint; never a global material identity."""

    window_id: str
    track_id: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "window_id",
            _nonempty_string(self.window_id, name="window_id"),
        )
        object.__setattr__(
            self,
            "track_id",
            genuine_integer(self.track_id, name="track_id", minimum=0),
        )

    def to_record(self) -> dict[str, object]:
        return {"window_id": self.window_id, "track_id": self.track_id}

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str,
    ) -> Prob4DLocalTrackEndpointV1:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a JSON object")
        _require_exact_fields(value, expected=_ENDPOINT_FIELDS, name=name)
        return cls(window_id=value["window_id"], track_id=value["track_id"])


@dataclass(frozen=True)
class Prob4DMaterialIdentityCandidateV1:
    """One independently validated null or linked identity hypothesis."""

    candidate_id: str
    kind: Literal["null", "linked"]
    source_endpoint: Prob4DLocalTrackEndpointV1 | None
    association_result_id: str | None
    source_score: float | None
    calibrated_log_weight: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidate_id = _sha256(self.candidate_id, name="candidate_id")
        if self.kind not in {"null", "linked"}:
            raise ValueError("candidate kind is unsupported")
        is_null = self.kind == "null"
        if is_null != (self.source_endpoint is None):
            raise ValueError("candidate kind does not match source_endpoint")
        association_result_id = self.association_result_id
        source_score = self.source_score
        if is_null:
            if association_result_id is not None or source_score is not None:
                raise ValueError(
                    "the null hypothesis must not carry source association evidence"
                )
        else:
            if not isinstance(self.source_endpoint, Prob4DLocalTrackEndpointV1):
                raise ValueError("linked source_endpoint has invalid type")
            association_result_id = _sha256(
                association_result_id,
                name="association_result_id",
            )
            source_score = _finite_real(
                source_score,
                name="source_score",
                minimum=0.0,
                maximum=1.0,
            )
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "association_result_id", association_result_id)
        object.__setattr__(self, "source_score", source_score)
        object.__setattr__(
            self,
            "calibrated_log_weight",
            _finite_real(
                self.calibrated_log_weight,
                name="calibrated_log_weight",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="material-identity candidate metadata",
            ),
        )

    def ordering_key(self) -> tuple[object, ...]:
        if self.source_endpoint is None:
            return (0, "", -1, "")
        return (
            1,
            self.source_endpoint.window_id,
            self.source_endpoint.track_id,
            self.association_result_id,
        )

    def expected_candidate_id(
        self,
        *,
        target_endpoint: Prob4DLocalTrackEndpointV1,
    ) -> str:
        return _content_id(
            {
                "schema": PROB4D_MATERIAL_IDENTITY_HYPOTHESIS_SCHEMA,
                "schema_version": PROB4D_MATERIAL_IDENTITY_HYPOTHESIS_VERSION,
                "target_endpoint": target_endpoint.to_record(),
                "kind": self.kind,
                "source_endpoint": (
                    None
                    if self.source_endpoint is None
                    else self.source_endpoint.to_record()
                ),
                "association_result_id": self.association_result_id,
            }
        )

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "source_endpoint": (
                None
                if self.source_endpoint is None
                else self.source_endpoint.to_record()
            ),
            "association_result_id": self.association_result_id,
            "source_score": self.source_score,
            "calibrated_log_weight": self.calibrated_log_weight,
            "metadata": plain_json(self.metadata),
        }


@dataclass(frozen=True)
class Prob4DMaterialIdentityMixtureV1:
    """Independent BayesianPhysTwin view of a Prob4D identity mixture."""

    mixture_id: str
    target_endpoint: Prob4DLocalTrackEndpointV1
    window_order: tuple[str, ...]
    causal_frame_stop: int
    association_rule_id: str
    calibration_id: str
    tracklet_producer_revision: str
    association_revision: str
    candidates: tuple[Prob4DMaterialIdentityCandidateV1, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    weight_semantics: str = PROB4D_MATERIAL_IDENTITY_WEIGHT_SEMANTICS
    null_hypothesis_semantics: str = PROB4D_MATERIAL_IDENTITY_NULL_SEMANTICS

    def __post_init__(self) -> None:
        if not isinstance(self.target_endpoint, Prob4DLocalTrackEndpointV1):
            raise ValueError("target_endpoint has invalid type")
        if type(self.window_order) is not tuple or not self.window_order:
            raise ValueError("window_order must be a non-empty tuple")
        window_order = tuple(
            _nonempty_string(value, name=f"window_order[{index}]")
            for index, value in enumerate(self.window_order)
        )
        if len(set(window_order)) != len(window_order):
            raise ValueError("window_order must contain unique window IDs")
        if window_order[-1] != self.target_endpoint.window_id:
            raise ValueError("target_endpoint window must be last in window_order")
        if type(self.candidates) is not tuple or not self.candidates:
            raise ValueError("candidates must be a non-empty tuple")
        if any(
            not isinstance(candidate, Prob4DMaterialIdentityCandidateV1)
            for candidate in self.candidates
        ):
            raise ValueError("candidates contain an invalid value")
        candidates = self.candidates
        canonical_candidates = tuple(
            sorted(candidates, key=lambda item: item.ordering_key())
        )
        if candidates != canonical_candidates:
            raise ValueError("candidates are not in canonical Prob4D order")
        if sum(candidate.kind == "null" for candidate in candidates) != 1:
            raise ValueError("exactly one null identity hypothesis is required")
        if candidates[0].kind != "null":
            raise ValueError("the null identity hypothesis must be first")
        linked_endpoints = tuple(
            candidate.source_endpoint
            for candidate in candidates
            if candidate.source_endpoint is not None
        )
        if len(set(linked_endpoints)) != len(linked_endpoints):
            raise ValueError("linked source endpoints must be unique")
        source_windows = frozenset(window_order[:-1])
        if any(
            endpoint.window_id not in source_windows for endpoint in linked_endpoints
        ):
            raise ValueError("linked source endpoint windows must precede the target")
        for candidate in candidates:
            if (
                candidate.expected_candidate_id(target_endpoint=self.target_endpoint)
                != candidate.candidate_id
            ):
                raise ValueError("material-identity candidate ID mismatch")
        if self.weight_semantics != PROB4D_MATERIAL_IDENTITY_WEIGHT_SEMANTICS:
            raise ValueError("unsupported material-identity weight semantics")
        if self.null_hypothesis_semantics != PROB4D_MATERIAL_IDENTITY_NULL_SEMANTICS:
            raise ValueError("unsupported null-hypothesis semantics")

        object.__setattr__(self, "window_order", window_order)
        object.__setattr__(
            self,
            "causal_frame_stop",
            genuine_integer(
                self.causal_frame_stop,
                name="causal_frame_stop",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "association_rule_id",
            _sha256(self.association_rule_id, name="association_rule_id"),
        )
        object.__setattr__(
            self,
            "calibration_id",
            _sha256(self.calibration_id, name="calibration_id"),
        )
        object.__setattr__(
            self,
            "tracklet_producer_revision",
            _revision(
                self.tracklet_producer_revision,
                name="tracklet_producer_revision",
            ),
        )
        object.__setattr__(
            self,
            "association_revision",
            _revision(self.association_revision, name="association_revision"),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="material-identity mixture metadata",
            ),
        )
        mixture_id = _sha256(self.mixture_id, name="mixture_id")
        if mixture_id != _content_id(self.identity_record()):
            raise ValueError("material-identity mixture ID mismatch")
        object.__setattr__(self, "mixture_id", mixture_id)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)

    @property
    def normalized_log_weights(self) -> FloatArray:
        values = np.asarray(
            [candidate.calibrated_log_weight for candidate in self.candidates],
            dtype=np.float64,
        )
        return _readonly_float(values - _logsumexp(values))

    @property
    def source_probabilities(self) -> FloatArray:
        return _readonly_float(np.exp(self.normalized_log_weights))

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": PROB4D_MATERIAL_IDENTITY_MIXTURE_SCHEMA,
            "schema_version": PROB4D_MATERIAL_IDENTITY_MIXTURE_VERSION,
            "target_endpoint": self.target_endpoint.to_record(),
            "window_order": list(self.window_order),
            "causal_frame_stop": self.causal_frame_stop,
            "association_rule_id": self.association_rule_id,
            "calibration_id": self.calibration_id,
            "tracklet_producer_revision": self.tracklet_producer_revision,
            "association_revision": self.association_revision,
            "weight_semantics": self.weight_semantics,
            "null_hypothesis_semantics": self.null_hypothesis_semantics,
            "candidates": [candidate.to_record() for candidate in self.candidates],
            "metadata": plain_json(self.metadata),
            "claim_boundary": PROB4D_MATERIAL_IDENTITY_CLAIM_BOUNDARY,
        }


def _candidate_from_mapping(
    value: object,
    *,
    name: str,
) -> Prob4DMaterialIdentityCandidateV1:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    _require_exact_fields(value, expected=_CANDIDATE_FIELDS, name=name)
    kind = value["kind"]
    if kind not in {"null", "linked"}:
        raise ValueError(f"{name}.kind is unsupported")
    source_value = value["source_endpoint"]
    source_endpoint = (
        None
        if source_value is None
        else Prob4DLocalTrackEndpointV1.from_mapping(
            source_value,
            name=f"{name}.source_endpoint",
        )
    )
    metadata = value["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{name}.metadata must be a JSON object")
    return Prob4DMaterialIdentityCandidateV1(
        candidate_id=value["candidate_id"],
        kind=kind,
        source_endpoint=source_endpoint,
        association_result_id=value["association_result_id"],
        source_score=value["source_score"],
        calibrated_log_weight=value["calibrated_log_weight"],
        metadata=metadata,
    )


def validate_prob4d_material_identity_mixture(
    value: Mapping[str, Any],
) -> Prob4DMaterialIdentityMixtureV1:
    """Validate a portable Prob4D identity mixture without importing Prob4D."""

    _require_exact_fields(value, expected=_MIXTURE_FIELDS, name="mixture")
    if value["schema"] != PROB4D_MATERIAL_IDENTITY_MIXTURE_SCHEMA:
        raise ValueError("unsupported material-identity mixture schema")
    if genuine_integer(value["schema_version"], name="schema_version", minimum=1) != 1:
        raise ValueError("unsupported material-identity mixture schema version")
    if value["claim_boundary"] != PROB4D_MATERIAL_IDENTITY_CLAIM_BOUNDARY:
        raise ValueError("material-identity mixture claim boundary changed")
    target = Prob4DLocalTrackEndpointV1.from_mapping(
        value["target_endpoint"],
        name="target_endpoint",
    )
    raw_window_order = value["window_order"]
    if type(raw_window_order) is not list or not raw_window_order:
        raise ValueError("window_order must be a non-empty JSON array")
    raw_candidates = value["candidates"]
    if type(raw_candidates) is not list or not raw_candidates:
        raise ValueError("candidates must be a non-empty JSON array")
    metadata = value["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a JSON object")
    return Prob4DMaterialIdentityMixtureV1(
        mixture_id=value["mixture_id"],
        target_endpoint=target,
        window_order=tuple(raw_window_order),
        causal_frame_stop=value["causal_frame_stop"],
        association_rule_id=value["association_rule_id"],
        calibration_id=value["calibration_id"],
        tracklet_producer_revision=value["tracklet_producer_revision"],
        association_revision=value["association_revision"],
        candidates=tuple(
            _candidate_from_mapping(candidate, name=f"candidates[{index}]")
            for index, candidate in enumerate(raw_candidates)
        ),
        metadata=metadata,
        weight_semantics=value["weight_semantics"],
        null_hypothesis_semantics=value["null_hypothesis_semantics"],
    )


def load_prob4d_material_identity_mixture(
    path: str | Path,
) -> Prob4DMaterialIdentityMixtureV1:
    """Load and independently validate one Prob4D identity-mixture JSON file."""

    return validate_prob4d_material_identity_mixture(_load_strict_json(Path(path)))


@dataclass(frozen=True)
class MaterialIdentityLikelihoodEvidenceV1:
    """Target-blind candidate-aligned prefix likelihoods for one mixture."""

    mixture_id: str
    common_state_domain_id: str
    candidate_ids: tuple[str, ...]
    log_likelihoods: FloatArray
    calibration_id: str
    likelihood_power: float = 1.0
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    semantics: str = IDENTITY_LIKELIHOOD_SEMANTICS
    evidence_id: str | None = None

    def __post_init__(self) -> None:
        mixture_id = _sha256(self.mixture_id, name="mixture_id")
        domain_id = _sha256(
            self.common_state_domain_id,
            name="common_state_domain_id",
        )
        if type(self.candidate_ids) is not tuple or not self.candidate_ids:
            raise ValueError("candidate_ids must be a non-empty tuple")
        candidate_ids = tuple(
            _sha256(value, name=f"candidate_ids[{index}]")
            for index, value in enumerate(self.candidate_ids)
        )
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate_ids must be unique")
        log_likelihoods = np.asarray(self.log_likelihoods, dtype=np.float64)
        if log_likelihoods.shape != (len(candidate_ids),):
            raise ValueError("log_likelihoods must match candidate_ids")
        if np.any(np.isnan(log_likelihoods)) or np.any(np.isposinf(log_likelihoods)):
            raise ValueError("log_likelihoods may not contain NaN or positive infinity")
        if self.semantics != IDENTITY_LIKELIHOOD_SEMANTICS:
            raise ValueError("unsupported identity-likelihood semantics")
        target_outcomes_used = genuine_boolean(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        if target_outcomes_used:
            raise ValueError("identity likelihood evidence may not use target outcomes")
        object.__setattr__(self, "mixture_id", mixture_id)
        object.__setattr__(self, "common_state_domain_id", domain_id)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(
            self,
            "log_likelihoods",
            _readonly_float(log_likelihoods),
        )
        object.__setattr__(
            self,
            "calibration_id",
            _sha256(self.calibration_id, name="calibration_id"),
        )
        object.__setattr__(
            self,
            "likelihood_power",
            _finite_real(
                self.likelihood_power,
                name="likelihood_power",
                minimum=0.0,
            ),
        )
        object.__setattr__(self, "target_outcomes_used", target_outcomes_used)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="identity-likelihood metadata",
            ),
        )
        expected_id = _content_id(self.identity_record())
        supplied_id = self.evidence_id
        if (
            supplied_id is not None
            and _sha256(
                supplied_id,
                name="evidence_id",
            )
            != expected_id
        ):
            raise ValueError("identity-likelihood evidence ID mismatch")
        object.__setattr__(self, "evidence_id", expected_id)

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": IDENTITY_LIKELIHOOD_EVIDENCE_SCHEMA,
            "schema_version": IDENTITY_LIKELIHOOD_EVIDENCE_VERSION,
            "mixture_id": self.mixture_id,
            "common_state_domain_id": self.common_state_domain_id,
            "candidate_ids": list(self.candidate_ids),
            "log_likelihoods_sha256": _array_id(self.log_likelihoods),
            "calibration_id": self.calibration_id,
            "likelihood_power": self.likelihood_power,
            "target_outcomes_used": self.target_outcomes_used,
            "semantics": self.semantics,
            "metadata": plain_json(self.metadata),
        }


class GaugeAwareStateResult(Protocol):
    """Structural subset required from ``GaugeAwareBeliefResult``."""

    inference_admissible: bool
    reason: str
    state_coefficients: np.ndarray
    posterior_covariance: np.ndarray
    input_lineage: Mapping[str, Any]


@dataclass(frozen=True)
class _CandidateState:
    admissible: bool
    reason: str
    mean: FloatArray
    covariance: FloatArray


def material_identity_candidate_lineage(
    mixture: Prob4DMaterialIdentityMixtureV1,
    *,
    candidate_id: str,
    common_state_domain_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Bind one candidate batch/result to its mixture and common state domain."""

    if not isinstance(mixture, Prob4DMaterialIdentityMixtureV1):
        raise TypeError("mixture must be Prob4DMaterialIdentityMixtureV1")
    candidate = _sha256(candidate_id, name="candidate_id")
    if candidate not in mixture.candidate_ids:
        raise ValueError("candidate_id is not present in the material-identity mixture")
    domain = _sha256(common_state_domain_id, name="common_state_domain_id")
    base = dict(plain_json(metadata or {}))
    conflicts = sorted(_RESERVED_LINEAGE_FIELDS & set(base))
    if conflicts:
        raise ValueError(
            f"metadata already defines reserved lineage fields: {conflicts}"
        )
    base.update(
        {
            _LINEAGE_MIXTURE_ID: mixture.mixture_id,
            _LINEAGE_CANDIDATE_ID: candidate,
            _LINEAGE_COMMON_STATE_DOMAIN_ID: domain,
            _LINEAGE_CAUSAL_FRAME_STOP: mixture.causal_frame_stop,
            _LINEAGE_SOURCE_CALIBRATION_ID: mixture.calibration_id,
        }
    )
    return frozen_finite_json_mapping(base, name="material-identity candidate lineage")


def _extract_candidate_state(
    result: GaugeAwareStateResult,
    *,
    mixture: Prob4DMaterialIdentityMixtureV1,
    candidate_id: str,
    common_state_domain_id: str,
) -> _CandidateState:
    try:
        admissible = genuine_boolean(
            result.inference_admissible,
            name="result.inference_admissible",
        )
        reason = _nonempty_string(result.reason, name="result.reason")
        mean = np.asarray(result.state_coefficients, dtype=np.float64)
        full_covariance = np.asarray(result.posterior_covariance, dtype=np.float64)
        lineage = result.input_lineage
    except AttributeError as error:
        raise TypeError(
            "candidate result does not satisfy GaugeAwareStateResult"
        ) from error
    if mean.ndim != 1 or mean.size == 0 or not np.all(np.isfinite(mean)):
        raise ValueError(
            "candidate state_coefficients must be a finite non-empty vector"
        )
    if (
        full_covariance.ndim != 2
        or full_covariance.shape[0] != full_covariance.shape[1]
    ):
        raise ValueError("candidate posterior_covariance must be square")
    if len(full_covariance) < len(mean):
        raise ValueError("candidate posterior covariance is smaller than the state")
    covariance = _validated_state_covariance(
        full_covariance[: len(mean), : len(mean)],
        name="candidate state covariance",
    )
    if not isinstance(lineage, Mapping):
        raise ValueError("candidate result input_lineage must be a mapping")
    expected = material_identity_candidate_lineage(
        mixture,
        candidate_id=candidate_id,
        common_state_domain_id=common_state_domain_id,
    )
    for key in _RESERVED_LINEAGE_FIELDS:
        if lineage.get(key) != expected[key]:
            raise ValueError(f"candidate result lineage does not bind {key}")
    return _CandidateState(
        admissible=admissible,
        reason=reason,
        mean=_readonly_float(mean),
        covariance=covariance,
    )


@dataclass(frozen=True)
class MaterialIdentityStatePosteriorV1:
    """Common-state Gaussian moments after fail-closed identity marginalization."""

    mixture_id: str
    likelihood_evidence_id: str
    common_state_domain_id: str
    candidate_ids: tuple[str, ...]
    candidate_inference_admissible: tuple[bool, ...]
    identity_marginalization_admissible: bool
    deployed_reference_only: bool
    reason: str
    posterior_probabilities: FloatArray
    state_mean: FloatArray
    state_covariance: FloatArray
    within_identity_covariance: FloatArray
    between_identity_covariance: FloatArray
    identity_entropy_nats: float
    effective_hypothesis_count: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    marginalization_semantics: str = IDENTITY_MARGINALIZATION_SEMANTICS
    moment_semantics: str = IDENTITY_STATE_MOMENT_SEMANTICS
    posterior_id: str | None = None

    def __post_init__(self) -> None:
        mixture_id = _sha256(self.mixture_id, name="mixture_id")
        evidence_id = _sha256(
            self.likelihood_evidence_id,
            name="likelihood_evidence_id",
        )
        domain_id = _sha256(
            self.common_state_domain_id,
            name="common_state_domain_id",
        )
        if type(self.candidate_ids) is not tuple or not self.candidate_ids:
            raise ValueError("candidate_ids must be a non-empty tuple")
        candidate_ids = tuple(
            _sha256(value, name=f"candidate_ids[{index}]")
            for index, value in enumerate(self.candidate_ids)
        )
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate_ids must be unique")
        if type(self.candidate_inference_admissible) is not tuple:
            raise ValueError("candidate_inference_admissible must be a tuple")
        statuses = tuple(
            genuine_boolean(value, name=f"candidate_inference_admissible[{index}]")
            for index, value in enumerate(self.candidate_inference_admissible)
        )
        if len(statuses) != len(candidate_ids):
            raise ValueError("candidate inference status count changed")
        marginal_admissible = genuine_boolean(
            self.identity_marginalization_admissible,
            name="identity_marginalization_admissible",
        )
        reference_only = genuine_boolean(
            self.deployed_reference_only,
            name="deployed_reference_only",
        )
        if marginal_admissible and not all(statuses):
            raise ValueError(
                "admissible marginalization requires admissible candidates"
            )
        if not marginal_admissible and not reference_only:
            raise ValueError("inadmissible marginalization must deploy the reference")
        reason = _nonempty_string(self.reason, name="reason")
        probabilities = np.asarray(self.posterior_probabilities, dtype=np.float64)
        mean = np.asarray(self.state_mean, dtype=np.float64)
        if probabilities.shape != (len(candidate_ids),):
            raise ValueError("posterior_probabilities must match candidate_ids")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("posterior_probabilities must be finite and non-negative")
        if not np.isclose(float(np.sum(probabilities)), 1.0, atol=1e-12, rtol=1e-12):
            raise ValueError("posterior_probabilities must sum to one")
        if mean.ndim != 1 or mean.size == 0 or not np.all(np.isfinite(mean)):
            raise ValueError("state_mean must be a finite non-empty vector")
        covariance = _validated_state_covariance(
            self.state_covariance,
            name="state_covariance",
        )
        within = _validated_state_covariance(
            self.within_identity_covariance,
            name="within_identity_covariance",
        )
        between = _validated_state_covariance(
            self.between_identity_covariance,
            name="between_identity_covariance",
        )
        expected_shape = (len(mean), len(mean))
        if any(
            value.shape != expected_shape for value in (covariance, within, between)
        ):
            raise ValueError("state covariance dimensions do not match state_mean")
        if not np.allclose(covariance, within + between, atol=1e-12, rtol=1e-12):
            raise ValueError(
                "state covariance must equal within plus between covariance"
            )
        if reference_only:
            expected_probabilities = np.zeros(len(candidate_ids), dtype=np.float64)
            expected_probabilities[0] = 1.0
            if not np.array_equal(probabilities, expected_probabilities):
                raise ValueError(
                    "reference-only deployment must be one-hot on the null"
                )
            if np.any(between != 0.0):
                raise ValueError(
                    "reference-only deployment must have zero between covariance"
                )
        entropy = _finite_real(
            self.identity_entropy_nats,
            name="identity_entropy_nats",
            minimum=0.0,
        )
        effective = _finite_real(
            self.effective_hypothesis_count,
            name="effective_hypothesis_count",
            minimum=1.0,
        )
        active = probabilities > 0.0
        expected_entropy = float(
            -np.sum(probabilities[active] * np.log(probabilities[active]))
        )
        if not np.isclose(entropy, expected_entropy, atol=1e-12, rtol=1e-12):
            raise ValueError("identity entropy does not match posterior probabilities")
        if not np.isclose(effective, np.exp(entropy), atol=1e-12, rtol=1e-12):
            raise ValueError("effective hypothesis count does not match entropy")
        if self.marginalization_semantics != IDENTITY_MARGINALIZATION_SEMANTICS:
            raise ValueError("unsupported identity marginalization semantics")
        if self.moment_semantics != IDENTITY_STATE_MOMENT_SEMANTICS:
            raise ValueError("unsupported identity state-moment semantics")

        object.__setattr__(self, "mixture_id", mixture_id)
        object.__setattr__(self, "likelihood_evidence_id", evidence_id)
        object.__setattr__(self, "common_state_domain_id", domain_id)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "candidate_inference_admissible", statuses)
        object.__setattr__(
            self,
            "identity_marginalization_admissible",
            marginal_admissible,
        )
        object.__setattr__(self, "deployed_reference_only", reference_only)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self,
            "posterior_probabilities",
            _readonly_float(probabilities),
        )
        object.__setattr__(self, "state_mean", _readonly_float(mean))
        object.__setattr__(self, "state_covariance", covariance)
        object.__setattr__(self, "within_identity_covariance", within)
        object.__setattr__(self, "between_identity_covariance", between)
        object.__setattr__(self, "identity_entropy_nats", entropy)
        object.__setattr__(self, "effective_hypothesis_count", effective)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="material-identity posterior metadata",
            ),
        )
        expected_id = _content_id(self.identity_record())
        supplied_id = self.posterior_id
        if (
            supplied_id is not None
            and _sha256(
                supplied_id,
                name="posterior_id",
            )
            != expected_id
        ):
            raise ValueError("material-identity posterior ID mismatch")
        object.__setattr__(self, "posterior_id", expected_id)

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": IDENTITY_STATE_POSTERIOR_SCHEMA,
            "schema_version": IDENTITY_STATE_POSTERIOR_VERSION,
            "mixture_id": self.mixture_id,
            "likelihood_evidence_id": self.likelihood_evidence_id,
            "common_state_domain_id": self.common_state_domain_id,
            "candidate_ids": list(self.candidate_ids),
            "candidate_inference_admissible": list(self.candidate_inference_admissible),
            "identity_marginalization_admissible": (
                self.identity_marginalization_admissible
            ),
            "deployed_reference_only": self.deployed_reference_only,
            "reason": self.reason,
            "posterior_probabilities_sha256": _array_id(self.posterior_probabilities),
            "state_mean_sha256": _array_id(self.state_mean),
            "state_covariance_sha256": _array_id(self.state_covariance),
            "within_identity_covariance_sha256": _array_id(
                self.within_identity_covariance
            ),
            "between_identity_covariance_sha256": _array_id(
                self.between_identity_covariance
            ),
            "identity_entropy_nats": self.identity_entropy_nats,
            "effective_hypothesis_count": self.effective_hypothesis_count,
            "marginalization_semantics": self.marginalization_semantics,
            "moment_semantics": self.moment_semantics,
            "metadata": plain_json(self.metadata),
        }


def _reference_posterior(
    *,
    mixture: Prob4DMaterialIdentityMixtureV1,
    evidence: MaterialIdentityLikelihoodEvidenceV1,
    states: tuple[_CandidateState, ...],
    admissible: bool,
    reason: str,
    metadata: Mapping[str, Any] | None,
) -> MaterialIdentityStatePosteriorV1:
    reference = states[0]
    probabilities = np.zeros(len(states), dtype=np.float64)
    probabilities[0] = 1.0
    zero = np.zeros_like(reference.covariance)
    return MaterialIdentityStatePosteriorV1(
        mixture_id=mixture.mixture_id,
        likelihood_evidence_id=evidence.evidence_id or "",
        common_state_domain_id=evidence.common_state_domain_id,
        candidate_ids=mixture.candidate_ids,
        candidate_inference_admissible=tuple(state.admissible for state in states),
        identity_marginalization_admissible=admissible,
        deployed_reference_only=True,
        reason=reason,
        posterior_probabilities=probabilities,
        state_mean=reference.mean,
        state_covariance=reference.covariance,
        within_identity_covariance=reference.covariance,
        between_identity_covariance=zero,
        identity_entropy_nats=0.0,
        effective_hypothesis_count=1.0,
        metadata=metadata or {},
    )


def marginalize_material_identity_state(
    mixture: Prob4DMaterialIdentityMixtureV1,
    evidence: MaterialIdentityLikelihoodEvidenceV1,
    candidate_results: Mapping[str, GaugeAwareStateResult],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> MaterialIdentityStatePosteriorV1:
    """Marginalize common-state moments or return the exact null result.

    Contract or lineage mismatches raise immediately. Numerically inadmissible
    candidates and an all-impossible likelihood vector are scientific fallback
    events: they return the exact state mean and covariance of the null candidate.
    """

    if not isinstance(mixture, Prob4DMaterialIdentityMixtureV1):
        raise TypeError("mixture must be Prob4DMaterialIdentityMixtureV1")
    if not isinstance(evidence, MaterialIdentityLikelihoodEvidenceV1):
        raise TypeError("evidence must be MaterialIdentityLikelihoodEvidenceV1")
    if evidence.mixture_id != mixture.mixture_id:
        raise ValueError("identity likelihood evidence binds a different mixture")
    if evidence.candidate_ids != mixture.candidate_ids:
        raise ValueError("identity likelihood candidate order changed")
    if not isinstance(candidate_results, Mapping):
        raise TypeError("candidate_results must be a mapping")
    supplied_ids = set(candidate_results)
    expected_ids = set(mixture.candidate_ids)
    if supplied_ids != expected_ids:
        missing = sorted(expected_ids - supplied_ids)
        extra = sorted(supplied_ids - expected_ids)
        raise ValueError(
            f"candidate result IDs changed: missing={missing}, extra={extra}"
        )
    states = tuple(
        _extract_candidate_state(
            candidate_results[candidate_id],
            mixture=mixture,
            candidate_id=candidate_id,
            common_state_domain_id=evidence.common_state_domain_id,
        )
        for candidate_id in mixture.candidate_ids
    )
    state_dimension = len(states[0].mean)
    if any(len(state.mean) != state_dimension for state in states):
        raise ValueError("candidate state dimensions do not share one domain")
    inadmissible = next(
        (
            (candidate_id, state.reason)
            for candidate_id, state in zip(mixture.candidate_ids, states, strict=True)
            if not state.admissible
        ),
        None,
    )
    if inadmissible is not None:
        candidate_id, candidate_reason = inadmissible
        return _reference_posterior(
            mixture=mixture,
            evidence=evidence,
            states=states,
            admissible=False,
            reason=(
                f"candidate-inference-inadmissible:{candidate_id}:{candidate_reason}"
            ),
            metadata=metadata,
        )
    if len(states) == 1:
        return _reference_posterior(
            mixture=mixture,
            evidence=evidence,
            states=states,
            admissible=True,
            reason="null-only-mixture",
            metadata=metadata,
        )

    prior_log_weights = np.asarray(mixture.normalized_log_weights)
    if evidence.likelihood_power == 0.0:
        log_terms = prior_log_weights
    else:
        log_terms = prior_log_weights + evidence.likelihood_power * np.asarray(
            evidence.log_likelihoods
        )
    log_normalizer = _logsumexp(log_terms)
    if np.isneginf(log_normalizer):
        return _reference_posterior(
            mixture=mixture,
            evidence=evidence,
            states=states,
            admissible=False,
            reason="all-candidate-likelihoods-impossible",
            metadata=metadata,
        )
    probabilities = np.exp(log_terms - log_normalizer)
    if np.all(probabilities[1:] == 0.0):
        return _reference_posterior(
            mixture=mixture,
            evidence=evidence,
            states=states,
            admissible=True,
            reason="posterior-null-reference",
            metadata=metadata,
        )

    means = np.stack([state.mean for state in states])
    covariances = np.stack([state.covariance for state in states])
    marginal_mean = np.sum(probabilities[:, None] * means, axis=0)
    within = np.sum(probabilities[:, None, None] * covariances, axis=0)
    centered = means - marginal_mean
    between = np.einsum("i,ij,ik->jk", probabilities, centered, centered)
    within = 0.5 * (within + within.T)
    between = 0.5 * (between + between.T)
    covariance = within + between
    active = probabilities > 0.0
    entropy = float(-np.sum(probabilities[active] * np.log(probabilities[active])))
    return MaterialIdentityStatePosteriorV1(
        mixture_id=mixture.mixture_id,
        likelihood_evidence_id=evidence.evidence_id or "",
        common_state_domain_id=evidence.common_state_domain_id,
        candidate_ids=mixture.candidate_ids,
        candidate_inference_admissible=tuple(state.admissible for state in states),
        identity_marginalization_admissible=True,
        deployed_reference_only=False,
        reason="identity-marginalized",
        posterior_probabilities=probabilities,
        state_mean=marginal_mean,
        state_covariance=covariance,
        within_identity_covariance=within,
        between_identity_covariance=between,
        identity_entropy_nats=entropy,
        effective_hypothesis_count=float(np.exp(entropy)),
        metadata=metadata or {},
    )


__all__ = [
    "GaugeAwareStateResult",
    "IDENTITY_LIKELIHOOD_EVIDENCE_SCHEMA",
    "IDENTITY_LIKELIHOOD_EVIDENCE_VERSION",
    "IDENTITY_LIKELIHOOD_SEMANTICS",
    "IDENTITY_MARGINALIZATION_SEMANTICS",
    "IDENTITY_STATE_MOMENT_SEMANTICS",
    "IDENTITY_STATE_POSTERIOR_SCHEMA",
    "IDENTITY_STATE_POSTERIOR_VERSION",
    "MaterialIdentityLikelihoodEvidenceV1",
    "MaterialIdentityStatePosteriorV1",
    "PROB4D_MATERIAL_IDENTITY_CLAIM_BOUNDARY",
    "PROB4D_MATERIAL_IDENTITY_HYPOTHESIS_SCHEMA",
    "PROB4D_MATERIAL_IDENTITY_HYPOTHESIS_VERSION",
    "PROB4D_MATERIAL_IDENTITY_MIXTURE_SCHEMA",
    "PROB4D_MATERIAL_IDENTITY_MIXTURE_VERSION",
    "PROB4D_MATERIAL_IDENTITY_NULL_SEMANTICS",
    "PROB4D_MATERIAL_IDENTITY_WEIGHT_SEMANTICS",
    "Prob4DLocalTrackEndpointV1",
    "Prob4DMaterialIdentityCandidateV1",
    "Prob4DMaterialIdentityMixtureV1",
    "load_prob4d_material_identity_mixture",
    "marginalize_material_identity_state",
    "material_identity_candidate_lineage",
    "validate_prob4d_material_identity_mixture",
]
