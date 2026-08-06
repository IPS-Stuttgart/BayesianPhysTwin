"""Target-blind temporal source-competence reliability for observations.

The provider's source-calibrated ``prior_reliability`` remains the upper bound.
A content-addressed sidecar supplies target-blind competent/incompetent evidence,
and the existing Markov reliability model may only reduce the deployed row
reliability. Physical innovations, posterior responsibilities, association
labels, and confirmation outcomes are forbidden inputs to this contract.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from ._canonical_contracts import (
    canonical_string_tuple,
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import (
    content_id,
    load_strict_json_object,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)
from ._prob4d_stream_binding import prob4d_observation_identity_summary
from .observation_belief import ObservationBeliefV1
from .structured_reliability import (
    MARKOV_TIME_MODE_INTEGER_STEPS,
    MARKOV_TIME_MODE_ORDER_ONLY,
    MarkovReliabilityConfig,
    smooth_markov_reliability,
)

SOURCE_COMPETENCE_EVIDENCE_SCHEMA = "bayesian_phystwin.source_competence_evidence"
SOURCE_COMPETENCE_EVIDENCE_VERSION = 1
SOURCE_COMPETENCE_CONFIG_SCHEMA = "bayesian_phystwin.source_competence_markov_config"
SOURCE_COMPETENCE_CONFIG_VERSION = 1
SOURCE_COMPETENCE_UPDATE_SCHEMA = (
    "bayesian_phystwin.source_competence_reliability_update"
)
SOURCE_COMPETENCE_UPDATE_VERSION = 1
SOURCE_COMPETENCE_COMPOSITION = "posterior-capped-by-provider-prior-v1"
SOURCE_COMPETENCE_CLAIM_BOUNDARY = (
    "This contract temporally reduces a source-calibrated observation reliability "
    "using target-blind source-competence evidence. It does not establish provider "
    "competence, calibration, physical-state identifiability, guarded-query benefit, "
    "deployment safety, Causal4D benefit, or state of the art."
)

_EVIDENCE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "observation_artifact_id",
        "observation_identity_sha256",
        "source_feature_artifact_id",
        "source_reliability_model_id",
        "causal_frame_stop",
        "feature_names",
        "sequence_ids",
        "time_values",
        "log_competent_density",
        "log_incompetent_density",
        "uses_target_outcomes",
        "uses_physical_innovation",
        "uses_posterior_responsibility",
        "uses_association_probability_as_label",
        "metadata",
        "claim_boundary",
    }
)


def _finite_real(
    value: object,
    *,
    name: str,
    positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real number")
    result = float(raw.item())
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _immutable_array(value: object, *, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    if array.dtype.hasobject:
        raise TypeError("source-competence arrays must not contain Python objects")
    payload = array.tobytes(order="C")
    return np.frombuffer(payload, dtype=array.dtype).reshape(array.shape)


def _array_descriptor(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": digest.hexdigest(),
    }


def _sequence_ids(value: Sequence[str], *, count: int) -> tuple[str, ...]:
    result = canonical_string_tuple(
        value,
        name="sequence_ids",
        allow_empty=False,
    )
    if len(result) != count:
        raise ValueError(f"sequence_ids must have length {count}")
    return result


@dataclass(frozen=True, slots=True)
class SourceCompetenceMarkovConfigV1:
    """Content-addressed temporal persistence and conservative composition."""

    inlier_persistence: float = 0.98
    outlier_persistence: float = 0.90
    probability_floor: float = 1e-6
    time_delta_mode: str = MARKOV_TIME_MODE_ORDER_ONLY
    time_step: float = 1.0
    composition: str = SOURCE_COMPETENCE_COMPOSITION
    config_id: str | None = None

    def __post_init__(self) -> None:
        inlier = _finite_real(
            self.inlier_persistence,
            name="inlier_persistence",
        )
        outlier = _finite_real(
            self.outlier_persistence,
            name="outlier_persistence",
        )
        floor = _finite_real(self.probability_floor, name="probability_floor")
        step = _finite_real(self.time_step, name="time_step", positive=True)
        if not 0.0 < inlier < 1.0:
            raise ValueError("inlier_persistence must lie in (0, 1)")
        if not 0.0 < outlier < 1.0:
            raise ValueError("outlier_persistence must lie in (0, 1)")
        if not 0.0 < floor < 0.5:
            raise ValueError("probability_floor must lie in (0, 0.5)")
        if self.time_delta_mode not in {
            MARKOV_TIME_MODE_ORDER_ONLY,
            MARKOV_TIME_MODE_INTEGER_STEPS,
        }:
            raise ValueError("unsupported source-competence time_delta_mode")
        if self.composition != SOURCE_COMPETENCE_COMPOSITION:
            raise ValueError("unsupported source-competence composition")
        object.__setattr__(self, "inlier_persistence", inlier)
        object.__setattr__(self, "outlier_persistence", outlier)
        object.__setattr__(self, "probability_floor", floor)
        object.__setattr__(self, "time_step", step)
        expected = content_id(self.identity_record())
        supplied = self.config_id
        if supplied is not None:
            supplied = sha256_digest(supplied, name="config_id")
            if supplied != expected:
                raise ValueError("source-competence config ID mismatch")
        object.__setattr__(self, "config_id", expected)

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": SOURCE_COMPETENCE_CONFIG_SCHEMA,
            "schema_version": SOURCE_COMPETENCE_CONFIG_VERSION,
            "inlier_persistence": self.inlier_persistence,
            "outlier_persistence": self.outlier_persistence,
            "probability_floor": self.probability_floor,
            "time_delta_mode": self.time_delta_mode,
            "time_step": self.time_step,
            "composition": self.composition,
        }

    def as_markov_config(self) -> MarkovReliabilityConfig:
        return MarkovReliabilityConfig(
            inlier_persistence=self.inlier_persistence,
            outlier_persistence=self.outlier_persistence,
            probability_floor=self.probability_floor,
            time_delta_mode=self.time_delta_mode,
            time_step=self.time_step,
        )


@dataclass(frozen=True, slots=True)
class SourceCompetenceEvidenceV1:
    """Row-bound target-blind unary evidence for persistent source competence."""

    observation_artifact_id: str
    observation_identity_sha256: str
    source_feature_artifact_id: str
    source_reliability_model_id: str
    causal_frame_stop: int
    feature_names: tuple[str, ...]
    sequence_ids: tuple[str, ...]
    time_values: np.ndarray
    log_competent_density: np.ndarray
    log_incompetent_density: np.ndarray
    uses_target_outcomes: bool = False
    uses_physical_innovation: bool = False
    uses_posterior_responsibility: bool = False
    uses_association_probability_as_label: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "observation_artifact_id",
            "observation_identity_sha256",
            "source_feature_artifact_id",
            "source_reliability_model_id",
        ):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        cutoff = genuine_integer(
            self.causal_frame_stop,
            name="causal_frame_stop",
            minimum=1,
        )
        features = canonical_string_tuple(
            self.feature_names,
            name="feature_names",
            allow_empty=False,
        )
        if len(set(features)) != len(features):
            raise ValueError("feature_names must be unique")
        times = np.asarray(self.time_values)
        competent = np.asarray(self.log_competent_density)
        incompetent = np.asarray(self.log_incompetent_density)
        if times.dtype != np.dtype(np.float64) or times.ndim != 1:
            raise ValueError("time_values must be a one-dimensional float64 array")
        count = len(times)
        sequences = _sequence_ids(self.sequence_ids, count=count)
        for name, values in (
            ("time_values", times),
            ("log_competent_density", competent),
            ("log_incompetent_density", incompetent),
        ):
            if values.dtype != np.dtype(np.float64) or values.shape != (count,):
                raise ValueError(f"{name} must be float64 with shape ({count},)")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain finite values")
        if count == 0:
            raise ValueError("source-competence evidence must contain rows")
        forbidden = {
            "uses_target_outcomes": self.uses_target_outcomes,
            "uses_physical_innovation": self.uses_physical_innovation,
            "uses_posterior_responsibility": self.uses_posterior_responsibility,
            "uses_association_probability_as_label": (
                self.uses_association_probability_as_label
            ),
        }
        for name, raw in forbidden.items():
            value = genuine_boolean(raw, name=name)
            if value:
                raise ValueError(f"claim-bearing source competence forbids {name}")
            object.__setattr__(self, name, False)
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="source-competence evidence metadata",
        )
        object.__setattr__(self, "causal_frame_stop", cutoff)
        object.__setattr__(self, "feature_names", features)
        object.__setattr__(self, "sequence_ids", sequences)
        object.__setattr__(
            self,
            "time_values",
            _immutable_array(times, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(
            self,
            "log_competent_density",
            _immutable_array(competent, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(
            self,
            "log_incompetent_density",
            _immutable_array(incompetent, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(self, "metadata", metadata)
        expected = content_id(self.identity_record())
        supplied = self.artifact_id
        if supplied is not None:
            supplied = sha256_digest(supplied, name="artifact_id")
            if supplied != expected:
                raise ValueError("source-competence evidence ID mismatch")
        object.__setattr__(self, "artifact_id", expected)

    @property
    def observation_count(self) -> int:
        return len(self.time_values)

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": SOURCE_COMPETENCE_EVIDENCE_SCHEMA,
            "schema_version": SOURCE_COMPETENCE_EVIDENCE_VERSION,
            "observation_artifact_id": self.observation_artifact_id,
            "observation_identity_sha256": self.observation_identity_sha256,
            "source_feature_artifact_id": self.source_feature_artifact_id,
            "source_reliability_model_id": self.source_reliability_model_id,
            "causal_frame_stop": self.causal_frame_stop,
            "feature_names": list(self.feature_names),
            "sequence_ids": list(self.sequence_ids),
            "time_values": _array_descriptor(self.time_values),
            "log_competent_density": _array_descriptor(self.log_competent_density),
            "log_incompetent_density": _array_descriptor(self.log_incompetent_density),
            "uses_target_outcomes": False,
            "uses_physical_innovation": False,
            "uses_posterior_responsibility": False,
            "uses_association_probability_as_label": False,
            "metadata": plain_json(self.metadata),
            "claim_boundary": SOURCE_COMPETENCE_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self.identity_record(),
            "artifact_id": self.artifact_id,
            "time_values": self.time_values.tolist(),
            "log_competent_density": self.log_competent_density.tolist(),
            "log_incompetent_density": self.log_incompetent_density.tolist(),
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> SourceCompetenceEvidenceV1:
        require_exact_fields(
            value,
            expected=_EVIDENCE_FIELDS,
            name="source-competence evidence",
        )
        if value.get("schema") != SOURCE_COMPETENCE_EVIDENCE_SCHEMA:
            raise ValueError("source-competence evidence schema differs")
        if value.get("schema_version") != SOURCE_COMPETENCE_EVIDENCE_VERSION:
            raise ValueError("source-competence evidence version differs")
        if value.get("claim_boundary") != SOURCE_COMPETENCE_CLAIM_BOUNDARY:
            raise ValueError("source-competence claim boundary differs")
        feature_names = value.get("feature_names")
        sequence_ids = value.get("sequence_ids")
        metadata = value.get("metadata")
        if not isinstance(feature_names, list) or not isinstance(sequence_ids, list):
            raise ValueError("source-competence sequence fields must be lists")
        if not isinstance(metadata, Mapping):
            raise ValueError("source-competence metadata must be an object")
        return cls(
            observation_artifact_id=cast(str, value.get("observation_artifact_id")),
            observation_identity_sha256=cast(
                str, value.get("observation_identity_sha256")
            ),
            source_feature_artifact_id=cast(
                str, value.get("source_feature_artifact_id")
            ),
            source_reliability_model_id=cast(
                str, value.get("source_reliability_model_id")
            ),
            causal_frame_stop=cast(int, value.get("causal_frame_stop")),
            feature_names=tuple(feature_names),
            sequence_ids=tuple(sequence_ids),
            time_values=np.asarray(value.get("time_values"), dtype=np.float64),
            log_competent_density=np.asarray(
                value.get("log_competent_density"), dtype=np.float64
            ),
            log_incompetent_density=np.asarray(
                value.get("log_incompetent_density"), dtype=np.float64
            ),
            uses_target_outcomes=cast(bool, value.get("uses_target_outcomes")),
            uses_physical_innovation=cast(bool, value.get("uses_physical_innovation")),
            uses_posterior_responsibility=cast(
                bool, value.get("uses_posterior_responsibility")
            ),
            uses_association_probability_as_label=cast(
                bool, value.get("uses_association_probability_as_label")
            ),
            metadata=metadata,
            artifact_id=sha256_digest(value.get("artifact_id"), name="artifact_id"),
        )


@dataclass(frozen=True, slots=True)
class SourceCompetenceReliabilityUpdateV1:
    """A refined observation plus raw and deployed competence probabilities."""

    source_observation: ObservationBeliefV1
    refined_observation: ObservationBeliefV1
    evidence: SourceCompetenceEvidenceV1
    config: SourceCompetenceMarkovConfigV1
    posterior_competence_probability: np.ndarray
    deployed_prior_reliability: np.ndarray
    sequence_log_evidence: Mapping[str, float]
    update_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_observation, ObservationBeliefV1):
            raise TypeError("source_observation must be an ObservationBeliefV1")
        if not isinstance(self.refined_observation, ObservationBeliefV1):
            raise TypeError("refined_observation must be an ObservationBeliefV1")
        if not isinstance(self.evidence, SourceCompetenceEvidenceV1):
            raise TypeError("evidence must be a SourceCompetenceEvidenceV1")
        if not isinstance(self.config, SourceCompetenceMarkovConfigV1):
            raise TypeError("config must be a SourceCompetenceMarkovConfigV1")
        count = self.source_observation.observation_count
        posterior = np.asarray(self.posterior_competence_probability)
        deployed = np.asarray(self.deployed_prior_reliability)
        for name, values in (
            ("posterior_competence_probability", posterior),
            ("deployed_prior_reliability", deployed),
        ):
            if values.dtype != np.dtype(np.float64) or values.shape != (count,):
                raise ValueError(f"{name} must be float64 with shape ({count},)")
            if not np.all(np.isfinite(values)) or np.any(
                (values < 0.0) | (values > 1.0)
            ):
                raise ValueError(f"{name} must lie in [0, 1]")
        if np.any(deployed > self.source_observation.prior_reliability):
            raise ValueError("deployed reliability exceeds the provider prior")
        if not np.array_equal(
            deployed,
            self.refined_observation.prior_reliability,
        ):
            raise ValueError(
                "refined observation does not contain deployed reliability"
            )
        if self.evidence.observation_artifact_id != self.source_observation.artifact_id:
            raise ValueError("evidence identifies a different source observation")
        if not isinstance(self.sequence_log_evidence, Mapping):
            raise TypeError("sequence_log_evidence must be a mapping")
        evidence_mapping = frozen_finite_json_mapping(
            self.sequence_log_evidence,
            name="source-competence sequence log evidence",
        )
        object.__setattr__(
            self,
            "posterior_competence_probability",
            _immutable_array(posterior, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(
            self,
            "deployed_prior_reliability",
            _immutable_array(deployed, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(self, "sequence_log_evidence", evidence_mapping)
        expected = content_id(self.identity_record())
        supplied = self.update_id
        if supplied is not None:
            supplied = sha256_digest(supplied, name="update_id")
            if supplied != expected:
                raise ValueError("source-competence update ID mismatch")
        object.__setattr__(self, "update_id", expected)

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": SOURCE_COMPETENCE_UPDATE_SCHEMA,
            "schema_version": SOURCE_COMPETENCE_UPDATE_VERSION,
            "source_observation_artifact_id": self.source_observation.artifact_id,
            "refined_observation_artifact_id": self.refined_observation.artifact_id,
            "evidence_artifact_id": self.evidence.artifact_id,
            "config_id": self.config.config_id,
            "posterior_competence_probability": _array_descriptor(
                self.posterior_competence_probability
            ),
            "deployed_prior_reliability": _array_descriptor(
                self.deployed_prior_reliability
            ),
            "sequence_log_evidence": plain_json(self.sequence_log_evidence),
            "composition": SOURCE_COMPETENCE_COMPOSITION,
            "claim_boundary": SOURCE_COMPETENCE_CLAIM_BOUNDARY,
        }


def validate_source_competence_evidence(
    observation: ObservationBeliefV1,
    evidence: SourceCompetenceEvidenceV1,
) -> None:
    """Verify exact observation, row order, count, and causal-prefix identity."""

    if not isinstance(observation, ObservationBeliefV1):
        raise TypeError("observation must be an ObservationBeliefV1")
    if not isinstance(evidence, SourceCompetenceEvidenceV1):
        raise TypeError("evidence must be a SourceCompetenceEvidenceV1")
    if evidence.observation_artifact_id != observation.artifact_id:
        raise ValueError("source-competence evidence identifies another observation")
    _, count, identity_sha = prob4d_observation_identity_summary(observation)
    if evidence.observation_count != count:
        raise ValueError("source-competence evidence row count differs")
    if evidence.observation_identity_sha256 != identity_sha:
        raise ValueError("source-competence row identity digest differs")
    if evidence.causal_frame_stop != observation.causal_frame_stop:
        raise ValueError("source-competence causal frame cutoff differs")


def refine_observation_source_competence(
    observation: ObservationBeliefV1,
    evidence: SourceCompetenceEvidenceV1,
    *,
    config: SourceCompetenceMarkovConfigV1 | None = None,
) -> SourceCompetenceReliabilityUpdateV1:
    """Temporally reduce row reliability without changing means or covariance."""

    validate_source_competence_evidence(observation, evidence)
    cfg = SourceCompetenceMarkovConfigV1() if config is None else config
    if not isinstance(cfg, SourceCompetenceMarkovConfigV1):
        raise TypeError("config must be a SourceCompetenceMarkovConfigV1")
    smoothed = smooth_markov_reliability(
        observation.prior_reliability,
        evidence.log_competent_density,
        evidence.log_incompetent_density,
        evidence.sequence_ids,
        evidence.time_values,
        config=cfg.as_markov_config(),
    )
    posterior = np.asarray(
        smoothed.posterior_inlier_probability,
        dtype=np.float64,
    )
    deployed = np.minimum(observation.prior_reliability, posterior)
    metadata = dict(plain_json(observation.metadata))
    metadata.update(
        {
            "source_competence_evidence_id": evidence.artifact_id,
            "source_competence_feature_artifact_id": (
                evidence.source_feature_artifact_id
            ),
            "source_competence_reliability_model_id": (
                evidence.source_reliability_model_id
            ),
            "source_competence_markov_config_id": cfg.config_id,
            "source_competence_original_observation_id": observation.artifact_id,
            "source_competence_composition": SOURCE_COMPETENCE_COMPOSITION,
            "source_competence_covariance_changed": False,
            "source_competence_association_probability_changed": False,
            "source_competence_uses_target_outcomes": False,
            "source_competence_uses_physical_innovation": False,
            "source_competence_uses_posterior_responsibility": False,
            "source_competence_uses_association_probability_as_label": False,
            "source_competence_claim_boundary": SOURCE_COMPETENCE_CLAIM_BOUNDARY,
        }
    )
    refined = replace(
        observation,
        prior_reliability=deployed,
        metadata=metadata,
    )
    unchanged_arrays = (
        "declared_frame_ids",
        "mean_xyz_m",
        "frame_ids",
        "entity_ids",
        "view_indices",
        "window_indices",
        "correlation_group_ids",
        "factor_group_ids",
        "association_probability",
        "local_covariance_m2",
        "low_rank_factor_m",
        "group_ids",
        "group_prior_nominal_probability",
        "group_composite_weight",
    )
    for name in unchanged_arrays:
        if not np.array_equal(getattr(refined, name), getattr(observation, name)):
            raise AssertionError(f"source competence changed observation {name}")
    return SourceCompetenceReliabilityUpdateV1(
        source_observation=observation,
        refined_observation=refined,
        evidence=evidence,
        config=cfg,
        posterior_competence_probability=posterior,
        deployed_prior_reliability=deployed,
        sequence_log_evidence=smoothed.sequence_log_evidence,
    )


def write_source_competence_evidence(
    evidence: SourceCompetenceEvidenceV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(evidence, SourceCompetenceEvidenceV1):
        raise TypeError("evidence must be a SourceCompetenceEvidenceV1")
    write_atomic_json(evidence.to_record(), path, overwrite=overwrite)


def load_source_competence_evidence(
    path: str | Path,
) -> SourceCompetenceEvidenceV1:
    value = load_strict_json_object(path, label="source-competence evidence")
    return SourceCompetenceEvidenceV1.from_mapping(value)


__all__ = [
    "SOURCE_COMPETENCE_CLAIM_BOUNDARY",
    "SOURCE_COMPETENCE_COMPOSITION",
    "SOURCE_COMPETENCE_CONFIG_SCHEMA",
    "SOURCE_COMPETENCE_CONFIG_VERSION",
    "SOURCE_COMPETENCE_EVIDENCE_SCHEMA",
    "SOURCE_COMPETENCE_EVIDENCE_VERSION",
    "SOURCE_COMPETENCE_UPDATE_SCHEMA",
    "SOURCE_COMPETENCE_UPDATE_VERSION",
    "SourceCompetenceEvidenceV1",
    "SourceCompetenceMarkovConfigV1",
    "SourceCompetenceReliabilityUpdateV1",
    "load_source_competence_evidence",
    "refine_observation_source_competence",
    "validate_source_competence_evidence",
    "write_source_competence_evidence",
]
