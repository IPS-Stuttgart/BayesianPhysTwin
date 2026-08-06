"""Strict Prob4D timestamp lineage consumption without a Prob4D dependency.

The producer sidecar records one timestamp and one conditional jitter scale per
observation factor.  This module independently revalidates the sidecar and its
source factor bundle, binds selected BayesianPhysTwin rows back to those exact
factors, and keeps two timing effects separate:

* factor-local conditional jitter, represented as a low-rank factor; and
* one coherent clock-domain offset, represented by an explicit timing state.

The coherent offset is never folded into local point covariance or into the
factor-local jitter representation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_integer,
    integer_array,
    plain_json,
)
from ._portable_contracts import (
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
)
from ._prob4d_stream_common import PROB4D_SOURCE_REPOSITORY_ALIASES
from .observation_belief import ObservationBeliefV1
from .observation_timing_interchange import observation_timing_prior_from_payload
from .observation_timing_nuisance import (
    ObservationTimingPrior,
    build_timing_jacobian,
)

PROB4D_OBSERVATION_TIMESTAMP_LINEAGE_SCHEMA = "prob4d.observation-timestamp-lineage"
PROB4D_OBSERVATION_TIMESTAMP_LINEAGE_VERSION = 1
PROB4D_TIMESTAMP_UNCERTAINTY_SEMANTICS = (
    "conditional-jitter-excludes-shared-clock-offset"
)
PROB4D_OBSERVATION_FACTOR_BUNDLE_SCHEMA = "prob4d.observation-factor-bundle"
PROB4D_OBSERVATION_FACTOR_BUNDLE_VERSION = 4
PROB4D_OBSERVATION_TIMESTAMP_BINDING_SCHEMA = (
    "bayesian_phystwin.prob4d_observation_timestamp_binding"
)
PROB4D_OBSERVATION_TIMESTAMP_BINDING_VERSION = 1
PROB4D_CONDITIONAL_JITTER_FACTOR_SEMANTICS = (
    "one-factor-local-latent-per-recorded-factor-v1"
)

_LINEAGE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "sequence_id",
        "case_id",
        "stream_id",
        "source_revision",
        "source_artifact_sha256",
        "causal_frame_stop",
        "clock_domain",
        "time_scale",
        "timestamp_source",
        "factor_ids",
        "frame_indices",
        "timestamps_ns",
        "conditional_timestamp_std_ns",
        "timestamp_uncertainty_semantics",
        "shared_clock_offset_prior_artifact_id",
        "metadata",
    }
)


def _canonical_string(value: object, *, name: str) -> str:
    result = nonempty_string(value, name=name)
    if result != result.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace")
    return result


def _string_sequence(
    value: object,
    *,
    name: str,
    unique: bool,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(_canonical_string(item, name=f"{name} entry") for item in value)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _immutable_array(value: object, *, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    if array.dtype.hasobject:
        raise TypeError("contract arrays must not contain Python objects")
    payload = array.tobytes(order="C")
    return np.frombuffer(payload, dtype=array.dtype).reshape(array.shape)


def _immutable_integer_vector(value: object, *, name: str) -> np.ndarray:
    array = integer_array(value, name=name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    return _immutable_array(array, dtype=np.dtype(np.int64))


def _immutable_nonnegative_float_vector(
    value: object,
    *,
    name: str,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric vector")
    array = np.array(raw, dtype=np.float64, copy=True, order="C")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return _immutable_array(array, dtype=np.dtype(np.float64))


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _array_descriptor(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _array_sha256(array),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"cannot read {path}") from error
    return digest.hexdigest()


def _repository_equivalent(first: str, second: str) -> bool:
    if first == second:
        return True
    aliases = set(PROB4D_SOURCE_REPOSITORY_ALIASES)
    return first in aliases and second in aliases


@dataclass(frozen=True, slots=True)
class Prob4DObservationTimestampLineageV1:
    """Independent reconstruction of one Prob4D timestamp sidecar."""

    sequence_id: str
    case_id: str
    stream_id: str
    source_revision: str
    source_artifact_sha256: str
    causal_frame_stop: int
    clock_domain: str
    time_scale: str
    timestamp_source: str
    factor_ids: Sequence[str]
    frame_indices: np.ndarray
    timestamps_ns: np.ndarray
    conditional_timestamp_std_ns: np.ndarray
    shared_clock_offset_prior_artifact_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamp_uncertainty_semantics: str = PROB4D_TIMESTAMP_UNCERTAINTY_SEMANTICS
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        sequence_id = _canonical_string(self.sequence_id, name="sequence_id")
        case_id = _canonical_string(self.case_id, name="case_id")
        stream_id = _canonical_string(self.stream_id, name="stream_id")
        revision = exact_revision(self.source_revision, name="source_revision")
        source_artifact = sha256_digest(
            self.source_artifact_sha256,
            name="source_artifact_sha256",
        )
        causal_stop = genuine_integer(
            self.causal_frame_stop,
            name="causal_frame_stop",
            minimum=1,
        )
        clock_domain = _canonical_string(self.clock_domain, name="clock_domain")
        time_scale = _canonical_string(self.time_scale, name="time_scale")
        timestamp_source = _canonical_string(
            self.timestamp_source,
            name="timestamp_source",
        )
        factor_ids = _string_sequence(
            self.factor_ids,
            name="factor_ids",
            unique=True,
        )
        frames = _immutable_integer_vector(
            self.frame_indices,
            name="frame_indices",
        )
        timestamps = _immutable_integer_vector(
            self.timestamps_ns,
            name="timestamps_ns",
        )
        jitter = _immutable_nonnegative_float_vector(
            self.conditional_timestamp_std_ns,
            name="conditional_timestamp_std_ns",
        )
        expected_shape = (len(factor_ids),)
        for name, array in (
            ("frame_indices", frames),
            ("timestamps_ns", timestamps),
            ("conditional_timestamp_std_ns", jitter),
        ):
            if array.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}")
        if np.any(frames < 0) or np.any(frames >= causal_stop):
            raise ValueError("timestamp factor frames cross the causal boundary")
        if np.any(timestamps < 0):
            raise ValueError("timestamps_ns must be nonnegative")
        if self.timestamp_uncertainty_semantics != (
            PROB4D_TIMESTAMP_UNCERTAINTY_SEMANTICS
        ):
            raise ValueError("timestamp uncertainty semantics changed")
        shared_prior = self.shared_clock_offset_prior_artifact_id
        if shared_prior is not None:
            shared_prior = sha256_digest(
                shared_prior,
                name="shared_clock_offset_prior_artifact_id",
            )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="Prob4D timestamp metadata",
        )

        object.__setattr__(self, "sequence_id", sequence_id)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "stream_id", stream_id)
        object.__setattr__(self, "source_revision", revision)
        object.__setattr__(self, "source_artifact_sha256", source_artifact)
        object.__setattr__(self, "causal_frame_stop", causal_stop)
        object.__setattr__(self, "clock_domain", clock_domain)
        object.__setattr__(self, "time_scale", time_scale)
        object.__setattr__(self, "timestamp_source", timestamp_source)
        object.__setattr__(self, "factor_ids", factor_ids)
        object.__setattr__(self, "frame_indices", frames)
        object.__setattr__(self, "timestamps_ns", timestamps)
        object.__setattr__(self, "conditional_timestamp_std_ns", jitter)
        object.__setattr__(
            self,
            "shared_clock_offset_prior_artifact_id",
            shared_prior,
        )
        object.__setattr__(self, "metadata", metadata)

        expected_id = content_id(self.identity_record())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("Prob4D timestamp lineage artifact ID mismatch")
        object.__setattr__(self, "artifact_id", expected_id)

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": PROB4D_OBSERVATION_TIMESTAMP_LINEAGE_SCHEMA,
            "schema_version": PROB4D_OBSERVATION_TIMESTAMP_LINEAGE_VERSION,
            "sequence_id": self.sequence_id,
            "case_id": self.case_id,
            "stream_id": self.stream_id,
            "source_revision": self.source_revision,
            "source_artifact_sha256": self.source_artifact_sha256,
            "causal_frame_stop": self.causal_frame_stop,
            "clock_domain": self.clock_domain,
            "time_scale": self.time_scale,
            "timestamp_source": self.timestamp_source,
            "factor_ids": list(self.factor_ids),
            "frame_indices": self.frame_indices.tolist(),
            "timestamps_ns": self.timestamps_ns.tolist(),
            "conditional_timestamp_std_ns": (
                self.conditional_timestamp_std_ns.tolist()
            ),
            "timestamp_uncertainty_semantics": (self.timestamp_uncertainty_semantics),
            "shared_clock_offset_prior_artifact_id": (
                self.shared_clock_offset_prior_artifact_id
            ),
            "metadata": plain_json(self.metadata),
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> Prob4DObservationTimestampLineageV1:
        require_exact_fields(
            value,
            expected=_LINEAGE_FIELDS,
            name="Prob4D observation timestamp lineage",
        )
        if value["schema"] != PROB4D_OBSERVATION_TIMESTAMP_LINEAGE_SCHEMA:
            raise ValueError("unsupported Prob4D timestamp lineage schema")
        version = genuine_integer(
            value["schema_version"],
            name="schema_version",
            minimum=1,
        )
        if version != PROB4D_OBSERVATION_TIMESTAMP_LINEAGE_VERSION:
            raise ValueError("unsupported Prob4D timestamp lineage version")
        metadata = value["metadata"]
        if not isinstance(metadata, Mapping):
            raise ValueError("Prob4D timestamp metadata must be a mapping")
        return cls(
            sequence_id=cast(str, value["sequence_id"]),
            case_id=cast(str, value["case_id"]),
            stream_id=cast(str, value["stream_id"]),
            source_revision=cast(str, value["source_revision"]),
            source_artifact_sha256=cast(
                str,
                value["source_artifact_sha256"],
            ),
            causal_frame_stop=cast(int, value["causal_frame_stop"]),
            clock_domain=cast(str, value["clock_domain"]),
            time_scale=cast(str, value["time_scale"]),
            timestamp_source=cast(str, value["timestamp_source"]),
            factor_ids=cast(Sequence[str], value["factor_ids"]),
            frame_indices=np.asarray(value["frame_indices"]),
            timestamps_ns=np.asarray(value["timestamps_ns"]),
            conditional_timestamp_std_ns=np.asarray(
                value["conditional_timestamp_std_ns"]
            ),
            timestamp_uncertainty_semantics=cast(
                str,
                value["timestamp_uncertainty_semantics"],
            ),
            shared_clock_offset_prior_artifact_id=cast(
                str | None,
                value["shared_clock_offset_prior_artifact_id"],
            ),
            metadata=cast(Mapping[str, Any], metadata),
            artifact_id=cast(str, value["artifact_id"]),
        )


@dataclass(frozen=True, slots=True)
class _BundleFactorOrderV1:
    sequence_id: str
    case_id: str
    stream_id: str
    source_repository: str
    source_revision: str
    causal_frame_stop: int
    factor_ids: tuple[str, ...]
    frame_indices: np.ndarray
    manifest_sha256: str


def _load_bundle_factor_order(
    path: str | Path,
    *,
    expected_sha256: str,
) -> _BundleFactorOrderV1:
    bundle_path = Path(path)
    expected_digest = sha256_digest(
        expected_sha256,
        name="expected_bundle_manifest_sha256",
    )
    actual_digest = _file_sha256(bundle_path)
    if actual_digest != expected_digest:
        raise ValueError("Prob4D bundle manifest checksum mismatch")
    value = load_strict_json_object(
        bundle_path,
        label="Prob4D observation-factor bundle",
    )
    if value.get("schema") != PROB4D_OBSERVATION_FACTOR_BUNDLE_SCHEMA:
        raise ValueError("unsupported Prob4D observation-factor bundle schema")
    version = genuine_integer(
        value.get("schema_version"),
        name="bundle schema_version",
        minimum=1,
    )
    if version != PROB4D_OBSERVATION_FACTOR_BUNDLE_VERSION:
        raise ValueError("timestamp binding requires observation-factor schema v4")
    sequence_id = _canonical_string(value.get("sequence_id"), name="sequence_id")
    raw_case = value.get("case_id")
    case_id = (
        sequence_id
        if raw_case is None
        else _canonical_string(
            raw_case,
            name="case_id",
        )
    )
    raw_stream = value.get("stream_id")
    stream_id = (
        sequence_id
        if raw_stream is None
        else _canonical_string(
            raw_stream,
            name="stream_id",
        )
    )
    repository = _canonical_string(
        value.get("source_repository"),
        name="source_repository",
    )
    revision = exact_revision(value.get("source_revision"), name="source_revision")
    causal_stop = genuine_integer(
        value.get("causal_frame_stop"),
        name="causal_frame_stop",
        minimum=1,
    )
    raw_factors = value.get("factors")
    if not isinstance(raw_factors, list) or not raw_factors:
        raise ValueError("Prob4D bundle factors must be a nonempty array")
    factor_ids: list[str] = []
    frame_indices: list[int] = []
    for index, raw_factor in enumerate(raw_factors):
        if not isinstance(raw_factor, Mapping):
            raise ValueError(f"Prob4D factor {index} must be a mapping")
        factor_ids.append(
            _canonical_string(
                raw_factor.get("factor_id"),
                name=f"factor {index} factor_id",
            )
        )
        frame_index = genuine_integer(
            raw_factor.get("frame_index"),
            name=f"factor {index} frame_index",
            minimum=0,
        )
        if frame_index >= causal_stop:
            raise ValueError("Prob4D factor frame crosses the causal boundary")
        frame_indices.append(frame_index)
    factor_tuple = tuple(factor_ids)
    if len(set(factor_tuple)) != len(factor_tuple):
        raise ValueError("Prob4D bundle factor IDs must be unique")
    return _BundleFactorOrderV1(
        sequence_id=sequence_id,
        case_id=case_id,
        stream_id=stream_id,
        source_repository=repository,
        source_revision=revision,
        causal_frame_stop=causal_stop,
        factor_ids=factor_tuple,
        frame_indices=_immutable_array(
            np.asarray(frame_indices, dtype=np.int64),
            dtype=np.dtype(np.int64),
        ),
        manifest_sha256=actual_digest,
    )


def load_prob4d_observation_timestamp_lineage(
    path: str | Path,
) -> Prob4DObservationTimestampLineageV1:
    value = load_strict_json_object(
        path,
        label="Prob4D observation timestamp lineage",
    )
    return Prob4DObservationTimestampLineageV1.from_mapping(value)


def _validate_derivative_xyz(
    value: object,
    *,
    observation_count: int,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (observation_count, 3) or array.dtype.kind not in "iuf":
        raise ValueError(
            "observation_derivative_xyz_per_s must have numeric shape (N, 3)"
        )
    result = np.array(array, dtype=np.float64, copy=True, order="C")
    if not np.all(np.isfinite(result)):
        raise ValueError("observation derivative must be finite")
    return result


@dataclass(frozen=True, slots=True)
class Prob4DObservationTimestampBindingV1:
    """Exact factor-to-row timestamp binding for one BPT observation."""

    observation_artifact_id: str
    bundle_manifest_sha256: str
    timestamp_lineage_artifact_id: str
    clock_domain: str
    time_scale: str
    timestamp_source: str
    factor_ids: tuple[str, ...]
    factor_frame_indices: np.ndarray
    factor_timestamps_ns: np.ndarray
    conditional_timestamp_std_ns: np.ndarray
    row_factor_indices: np.ndarray
    row_timestamps_s: np.ndarray
    row_conditional_timestamp_std_s: np.ndarray
    shared_clock_offset_prior_artifact_id: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamp_uncertainty_semantics: str = PROB4D_TIMESTAMP_UNCERTAINTY_SEMANTICS
    conditional_jitter_factor_semantics: str = (
        PROB4D_CONDITIONAL_JITTER_FACTOR_SEMANTICS
    )
    binding_id: str | None = None

    def __post_init__(self) -> None:
        observation_id = sha256_digest(
            self.observation_artifact_id,
            name="observation_artifact_id",
        )
        manifest_sha = sha256_digest(
            self.bundle_manifest_sha256,
            name="bundle_manifest_sha256",
        )
        lineage_id = sha256_digest(
            self.timestamp_lineage_artifact_id,
            name="timestamp_lineage_artifact_id",
        )
        clock_domain = _canonical_string(self.clock_domain, name="clock_domain")
        time_scale = _canonical_string(self.time_scale, name="time_scale")
        timestamp_source = _canonical_string(
            self.timestamp_source,
            name="timestamp_source",
        )
        factor_ids = _string_sequence(
            self.factor_ids,
            name="factor_ids",
            unique=True,
        )
        factor_frames = _immutable_integer_vector(
            self.factor_frame_indices,
            name="factor_frame_indices",
        )
        factor_timestamps = _immutable_integer_vector(
            self.factor_timestamps_ns,
            name="factor_timestamps_ns",
        )
        factor_jitter = _immutable_nonnegative_float_vector(
            self.conditional_timestamp_std_ns,
            name="conditional_timestamp_std_ns",
        )
        row_factors = _immutable_integer_vector(
            self.row_factor_indices,
            name="row_factor_indices",
        )
        row_timestamps = _immutable_nonnegative_float_vector(
            self.row_timestamps_s,
            name="row_timestamps_s",
        )
        row_jitter = _immutable_nonnegative_float_vector(
            self.row_conditional_timestamp_std_s,
            name="row_conditional_timestamp_std_s",
        )
        factor_shape = (len(factor_ids),)
        for name, array in (
            ("factor_frame_indices", factor_frames),
            ("factor_timestamps_ns", factor_timestamps),
            ("conditional_timestamp_std_ns", factor_jitter),
        ):
            if array.shape != factor_shape:
                raise ValueError(f"{name} must have shape {factor_shape}")
        row_shape = row_factors.shape
        if row_factors.ndim != 1 or row_shape == (0,):
            raise ValueError("row_factor_indices must be a nonempty vector")
        if row_timestamps.shape != row_shape or row_jitter.shape != row_shape:
            raise ValueError("row timestamp arrays must match row_factor_indices")
        if np.any(row_factors < 0) or np.any(row_factors >= len(factor_ids)):
            raise ValueError("row_factor_indices reference an unknown factor")
        if self.timestamp_uncertainty_semantics != (
            PROB4D_TIMESTAMP_UNCERTAINTY_SEMANTICS
        ):
            raise ValueError("timestamp uncertainty semantics changed")
        if self.conditional_jitter_factor_semantics != (
            PROB4D_CONDITIONAL_JITTER_FACTOR_SEMANTICS
        ):
            raise ValueError("conditional jitter factor semantics changed")
        shared_prior = self.shared_clock_offset_prior_artifact_id
        if shared_prior is not None:
            shared_prior = sha256_digest(
                shared_prior,
                name="shared_clock_offset_prior_artifact_id",
            )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="Prob4D timestamp binding metadata",
        )

        object.__setattr__(self, "observation_artifact_id", observation_id)
        object.__setattr__(self, "bundle_manifest_sha256", manifest_sha)
        object.__setattr__(self, "timestamp_lineage_artifact_id", lineage_id)
        object.__setattr__(self, "clock_domain", clock_domain)
        object.__setattr__(self, "time_scale", time_scale)
        object.__setattr__(self, "timestamp_source", timestamp_source)
        object.__setattr__(self, "factor_ids", factor_ids)
        object.__setattr__(self, "factor_frame_indices", factor_frames)
        object.__setattr__(self, "factor_timestamps_ns", factor_timestamps)
        object.__setattr__(self, "conditional_timestamp_std_ns", factor_jitter)
        object.__setattr__(self, "row_factor_indices", row_factors)
        object.__setattr__(self, "row_timestamps_s", row_timestamps)
        object.__setattr__(
            self,
            "row_conditional_timestamp_std_s",
            row_jitter,
        )
        object.__setattr__(
            self,
            "shared_clock_offset_prior_artifact_id",
            shared_prior,
        )
        object.__setattr__(self, "metadata", metadata)

        expected_id = content_id(self.identity_record())
        supplied_id = self.binding_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="binding_id")
            if supplied_id != expected_id:
                raise ValueError("Prob4D timestamp binding ID mismatch")
        object.__setattr__(self, "binding_id", expected_id)

    @property
    def observation_count(self) -> int:
        return int(self.row_factor_indices.size)

    @property
    def factor_count(self) -> int:
        return len(self.factor_ids)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "factor_frame_indices": np.asarray(self.factor_frame_indices),
            "factor_timestamps_ns": np.asarray(self.factor_timestamps_ns),
            "conditional_timestamp_std_ns": np.asarray(
                self.conditional_timestamp_std_ns
            ),
            "row_factor_indices": np.asarray(self.row_factor_indices),
            "row_timestamps_s": np.asarray(self.row_timestamps_s),
            "row_conditional_timestamp_std_s": np.asarray(
                self.row_conditional_timestamp_std_s
            ),
        }

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": PROB4D_OBSERVATION_TIMESTAMP_BINDING_SCHEMA,
            "schema_version": PROB4D_OBSERVATION_TIMESTAMP_BINDING_VERSION,
            "observation_artifact_id": self.observation_artifact_id,
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "timestamp_lineage_artifact_id": self.timestamp_lineage_artifact_id,
            "clock_domain": self.clock_domain,
            "time_scale": self.time_scale,
            "timestamp_source": self.timestamp_source,
            "factor_ids": list(self.factor_ids),
            "arrays": {
                name: _array_descriptor(array) for name, array in self.arrays().items()
            },
            "shared_clock_offset_prior_artifact_id": (
                self.shared_clock_offset_prior_artifact_id
            ),
            "timestamp_uncertainty_semantics": (self.timestamp_uncertainty_semantics),
            "conditional_jitter_factor_semantics": (
                self.conditional_jitter_factor_semantics
            ),
            "metadata": plain_json(self.metadata),
        }

    def conditional_jitter_low_rank_factor(
        self,
        observation_derivative_xyz_per_s: object,
    ) -> np.ndarray:
        """Return factor-local timing jitter as ``(N, 3, factor_count)``."""

        derivative = _validate_derivative_xyz(
            observation_derivative_xyz_per_s,
            observation_count=self.observation_count,
        )
        result: np.ndarray = np.zeros(
            (self.observation_count, 3, self.factor_count),
            dtype=np.float64,
        )
        for row, factor_index in enumerate(self.row_factor_indices):
            result[row, :, int(factor_index)] = (
                derivative[row] * self.row_conditional_timestamp_std_s[row]
            )
        return _immutable_array(result, dtype=np.dtype(np.float64))

    def shared_clock_design(
        self,
        observation_derivative_xyz_per_s: object,
    ) -> np.ndarray:
        """Return the coherent clock-offset design with shape ``(3N, 1)``."""

        derivative = _validate_derivative_xyz(
            observation_derivative_xyz_per_s,
            observation_count=self.observation_count,
        )
        design = build_timing_jacobian(derivative.reshape(-1))
        return _immutable_array(design, dtype=np.dtype(np.float64))

    def exploratory_shared_clock_prior_from_payload(
        self,
        value: Mapping[str, object],
    ) -> ObservationTimingPrior:
        """Construct an exploratory compact prior; not claim-bearing."""

        expected_id = self.shared_clock_offset_prior_artifact_id
        if expected_id is None:
            raise ValueError("timestamp lineage declares no shared clock prior")
        prior = observation_timing_prior_from_payload(value)
        if prior.source_artifact_id != expected_id:
            raise ValueError("shared clock prior artifact ID differs from lineage")
        if prior.clock_domain != self.clock_domain:
            raise ValueError("shared clock prior domain differs from lineage")
        return prior


def load_prob4d_observation_timestamp_binding(
    observation: ObservationBeliefV1,
    *,
    timestamp_lineage_path: str | Path,
    bundle_manifest_path: str | Path,
    expected_bundle_manifest_sha256: str,
    row_factor_ids: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> Prob4DObservationTimestampBindingV1:
    """Load, revalidate, and bind one producer timestamp sidecar."""

    if not isinstance(observation, ObservationBeliefV1):
        raise TypeError("observation must be an ObservationBeliefV1")
    lineage = load_prob4d_observation_timestamp_lineage(timestamp_lineage_path)
    bundle = _load_bundle_factor_order(
        bundle_manifest_path,
        expected_sha256=expected_bundle_manifest_sha256,
    )
    expected_observation = {
        "case_id": bundle.case_id,
        "stream_id": bundle.stream_id,
        "source_revision": bundle.source_revision,
        "causal_frame_stop": bundle.causal_frame_stop,
    }
    for name, expected in expected_observation.items():
        if getattr(observation, name) != expected:
            raise ValueError(f"observation {name} differs from Prob4D bundle")
    if not _repository_equivalent(
        observation.source_repository,
        bundle.source_repository,
    ):
        raise ValueError("observation source repository differs from Prob4D bundle")
    expected_lineage = {
        "sequence_id": bundle.sequence_id,
        "case_id": bundle.case_id,
        "stream_id": bundle.stream_id,
        "source_revision": bundle.source_revision,
        "causal_frame_stop": bundle.causal_frame_stop,
    }
    for name, expected in expected_lineage.items():
        if getattr(lineage, name) != expected:
            raise ValueError(f"timestamp lineage {name} differs from bundle")
    if tuple(lineage.factor_ids) != bundle.factor_ids:
        raise ValueError("timestamp lineage factor order differs from bundle")
    if not np.array_equal(lineage.frame_indices, bundle.frame_indices):
        raise ValueError("timestamp lineage factor frames differ from bundle")

    row_ids = _string_sequence(
        row_factor_ids,
        name="row_factor_ids",
        unique=False,
    )
    if len(row_ids) != observation.observation_count:
        raise ValueError("row_factor_ids must identify every observation row")
    positions = {factor_id: index for index, factor_id in enumerate(bundle.factor_ids)}
    try:
        row_indices = np.asarray(
            [positions[factor_id] for factor_id in row_ids],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("row_factor_ids reference an unknown Prob4D factor") from error
    expected_frames = bundle.frame_indices[row_indices]
    if not np.array_equal(observation.frame_ids, expected_frames):
        raise ValueError("observation row frames differ from their Prob4D factors")

    row_timestamps = np.asarray(lineage.timestamps_ns[row_indices], dtype=np.float64)
    row_timestamps *= 1e-9
    row_jitter = np.asarray(
        lineage.conditional_timestamp_std_ns[row_indices],
        dtype=np.float64,
    )
    row_jitter *= 1e-9
    return Prob4DObservationTimestampBindingV1(
        observation_artifact_id=observation.artifact_id,
        bundle_manifest_sha256=bundle.manifest_sha256,
        timestamp_lineage_artifact_id=cast(str, lineage.artifact_id),
        clock_domain=lineage.clock_domain,
        time_scale=lineage.time_scale,
        timestamp_source=lineage.timestamp_source,
        factor_ids=bundle.factor_ids,
        factor_frame_indices=bundle.frame_indices,
        factor_timestamps_ns=lineage.timestamps_ns,
        conditional_timestamp_std_ns=lineage.conditional_timestamp_std_ns,
        row_factor_indices=row_indices,
        row_timestamps_s=row_timestamps,
        row_conditional_timestamp_std_s=row_jitter,
        shared_clock_offset_prior_artifact_id=(
            lineage.shared_clock_offset_prior_artifact_id
        ),
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "PROB4D_CONDITIONAL_JITTER_FACTOR_SEMANTICS",
    "PROB4D_OBSERVATION_TIMESTAMP_BINDING_SCHEMA",
    "PROB4D_OBSERVATION_TIMESTAMP_BINDING_VERSION",
    "PROB4D_OBSERVATION_TIMESTAMP_LINEAGE_SCHEMA",
    "PROB4D_OBSERVATION_TIMESTAMP_LINEAGE_VERSION",
    "PROB4D_TIMESTAMP_UNCERTAINTY_SEMANTICS",
    "Prob4DObservationTimestampBindingV1",
    "Prob4DObservationTimestampLineageV1",
    "load_prob4d_observation_timestamp_binding",
    "load_prob4d_observation_timestamp_lineage",
]
