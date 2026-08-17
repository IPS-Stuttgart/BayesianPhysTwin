"""Member-byte and observation-identity binding for Prob4D streams."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from ._canonical_contracts import genuine_boolean, genuine_integer
from ._portable_contracts import content_id, load_strict_json_object
from ._prob4d_stream_common import (
    PROB4D_OBSERVATION_FACTOR_BUNDLE_SCHEMA,
    PROB4D_OBSERVATION_FACTOR_BUNDLE_VERSION,
    PROB4D_STREAM_OBSERVATION_BINDING_SCHEMA,
    PROB4D_STREAM_OBSERVATION_BINDING_VERSION,
    _file_sha256,
    _ordinary_confined_file,
    _sha256,
    _write_atomic_json,
)
from ._prob4d_stream_manifest import (
    Prob4DObservationFactorStreamUpdateV1,
    Prob4DObservationFactorStreamV1,
)
from .observation_belief import ObservationBeliefV1


def _verify_stream_member(
    manifest_path: Path,
    update: Prob4DObservationFactorStreamUpdateV1,
) -> None:
    bundle_path = _ordinary_confined_file(
        manifest_path.parent,
        update.bundle_manifest_path,
        name="bundle_manifest_path",
    )
    if _file_sha256(bundle_path) != update.bundle_manifest_sha256:
        raise ValueError("bundle manifest checksum differs from stream update")
    bundle = load_strict_json_object(
        bundle_path,
        label="Prob4D observation-factor bundle",
    )
    if bundle.get("schema") != PROB4D_OBSERVATION_FACTOR_BUNDLE_SCHEMA:
        raise ValueError("stream member is not a Prob4D observation-factor bundle")
    if bundle.get("schema_version") != PROB4D_OBSERVATION_FACTOR_BUNDLE_VERSION:
        raise ValueError("stream member is not observation-factor schema v4")
    expected_fields = {
        "sequence_id": update.bundle_sequence_id,
        "case_id": update.case_id,
        "stream_id": update.stream_id,
        "source_repository": update.source_repository,
        "source_revision": update.source_revision,
        "causal_frame_stop": update.causal_frame_stop,
    }
    for name, expected in expected_fields.items():
        if bundle.get(name) != expected:
            raise ValueError(f"stream member {name} differs from stream update")
    factors = bundle.get("factors")
    if not isinstance(factors, list) or len(factors) != update.factor_count:
        raise ValueError("stream member factor_count differs from stream update")
    covariance = bundle.get("gauge_covariance")
    if not isinstance(covariance, Mapping):
        raise ValueError("stream member has no gauge covariance descriptor")
    if (
        covariance.get("semantics") != "joint-cross-window"
        or covariance.get("cross_window_covariance_preserved") is not True
    ):
        raise ValueError("stream member does not preserve joint gauge covariance")
    if covariance.get("ordered_gauge_ids") != list(update.gauge_ids):
        raise ValueError("stream member gauge order differs from stream update")
    payload = bundle.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("stream member has no payload descriptor")
    if payload.get("allow_pickle") is not False:
        raise ValueError("stream member payload must disable pickle")
    payload_path = _ordinary_confined_file(
        bundle_path.parent,
        cast(str, payload.get("path")),
        name="bundle payload path",
    )
    declared_payload_sha = _sha256(
        payload.get("sha256"),
        name="bundle payload sha256",
    )
    if declared_payload_sha != update.bundle_payload_sha256:
        raise ValueError("bundle payload digest differs from stream update")
    if _file_sha256(payload_path) != declared_payload_sha:
        raise ValueError("bundle payload checksum mismatch")


def load_prob4d_observation_factor_stream(
    path: str | Path,
    *,
    verify_member_files: bool = True,
) -> Prob4DObservationFactorStreamV1:
    """Load and independently validate a Prob4D factor-stream manifest."""

    manifest_path = Path(path)
    value = load_strict_json_object(
        manifest_path,
        label="Prob4D observation-factor stream",
    )
    stream = Prob4DObservationFactorStreamV1.from_mapping(value)
    verify = genuine_boolean(
        verify_member_files,
        name="verify_member_files",
    )
    if verify:
        for update in stream.updates:
            _verify_stream_member(manifest_path, update)
    return stream


def write_prob4d_observation_factor_stream(
    stream: Prob4DObservationFactorStreamV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a portable stream fixture or independently reconstructed manifest."""

    if not isinstance(stream, Prob4DObservationFactorStreamV1):
        raise TypeError("stream must be a Prob4DObservationFactorStreamV1")
    return _write_atomic_json(stream.to_record(), path, overwrite=overwrite)


def prob4d_observation_identity_summary(
    observation: ObservationBeliefV1,
) -> tuple[int, int, str]:
    """Recompute Prob4D's ordered observation and persistent-identity digest."""

    if not isinstance(observation, ObservationBeliefV1):
        raise TypeError("observation must be an ObservationBeliefV1")
    rows = zip(
        observation.frame_ids,
        observation.view_indices,
        observation.window_indices,
        observation.entity_ids,
        strict=True,
    )
    persistent: set[tuple[str, str, int]] = set()
    identities: list[tuple[int, str, str, int]] = []
    for frame_id, view_index, window_index, entity_id in rows:
        persistent_id = (
            observation.view_names[int(view_index)],
            observation.window_names[int(window_index)],
            int(entity_id),
        )
        identity = (int(frame_id), *persistent_id)
        persistent.add(persistent_id)
        identities.append(identity)
    if len(set(identities)) != len(identities):
        raise ValueError("observation contains duplicate Prob4D stream identities")
    digest = content_id({"observations": sorted(identities)})
    return len(persistent), len(identities), digest


@dataclass(frozen=True, slots=True)
class Prob4DStreamObservationBindingV1:
    """Content address binding one stream update to one BPT observation artifact."""

    stream_artifact_id: str
    stream_update_id: str
    update_index: int
    observation_artifact_id: str
    observation_identity_sha256: str
    persistent_identity_count: int
    observation_count: int
    causal_frame_stop: int
    binding_id: str | None = None

    def __post_init__(self) -> None:
        stream_id = _sha256(self.stream_artifact_id, name="stream_artifact_id")
        update_id = _sha256(self.stream_update_id, name="stream_update_id")
        update_index = genuine_integer(
            self.update_index,
            name="update_index",
            minimum=0,
        )
        observation_id = _sha256(
            self.observation_artifact_id,
            name="observation_artifact_id",
        )
        identity_sha = _sha256(
            self.observation_identity_sha256,
            name="observation_identity_sha256",
        )
        persistent_count = genuine_integer(
            self.persistent_identity_count,
            name="persistent_identity_count",
            minimum=1,
        )
        observation_count = genuine_integer(
            self.observation_count,
            name="observation_count",
            minimum=1,
        )
        causal_stop = genuine_integer(
            self.causal_frame_stop,
            name="causal_frame_stop",
            minimum=1,
        )
        object.__setattr__(self, "stream_artifact_id", stream_id)
        object.__setattr__(self, "stream_update_id", update_id)
        object.__setattr__(self, "update_index", update_index)
        object.__setattr__(self, "observation_artifact_id", observation_id)
        object.__setattr__(self, "observation_identity_sha256", identity_sha)
        object.__setattr__(self, "persistent_identity_count", persistent_count)
        object.__setattr__(self, "observation_count", observation_count)
        object.__setattr__(self, "causal_frame_stop", causal_stop)
        expected_id = content_id(self.descriptor())
        supplied_id = self.binding_id
        if supplied_id is not None:
            supplied_id = _sha256(supplied_id, name="binding_id")
            if supplied_id != expected_id:
                raise ValueError("observation binding_id does not match content")
        object.__setattr__(self, "binding_id", expected_id)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": PROB4D_STREAM_OBSERVATION_BINDING_SCHEMA,
            "schema_version": PROB4D_STREAM_OBSERVATION_BINDING_VERSION,
            "stream_artifact_id": self.stream_artifact_id,
            "stream_update_id": self.stream_update_id,
            "update_index": self.update_index,
            "observation_artifact_id": self.observation_artifact_id,
            "observation_identity_sha256": self.observation_identity_sha256,
            "persistent_identity_count": self.persistent_identity_count,
            "observation_count": self.observation_count,
            "causal_frame_stop": self.causal_frame_stop,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "binding_id": self.binding_id}


def bind_prob4d_stream_observation(
    stream: Prob4DObservationFactorStreamV1,
    update_index: int,
    observation: ObservationBeliefV1,
) -> Prob4DStreamObservationBindingV1:
    """Bind a BPT observation to the exact row identities of one stream update."""

    if not isinstance(stream, Prob4DObservationFactorStreamV1):
        raise TypeError("stream must be a Prob4DObservationFactorStreamV1")
    index = genuine_integer(update_index, name="update_index", minimum=0)
    if index >= len(stream.updates):
        raise ValueError("update_index is outside the Prob4D stream")
    if not isinstance(observation, ObservationBeliefV1):
        raise TypeError("observation must be an ObservationBeliefV1")
    update = stream.updates[index]
    expected = {
        "case_id": stream.case_id,
        "stream_id": stream.stream_id,
        "source_repository": stream.source_repository,
        "source_revision": stream.source_revision,
        "causal_frame_stop": update.causal_frame_stop,
    }
    for name, expected_value in expected.items():
        if getattr(observation, name) != expected_value:
            raise ValueError(f"observation {name} differs from stream update")
    frames = np.asarray(observation.frame_ids, dtype=np.int64)
    if np.any(frames < update.admitted_frame_start) or np.any(
        frames >= update.causal_frame_stop
    ):
        raise ValueError("observation rows cross the admitted stream interval")
    if tuple(observation.window_names) != tuple(update.gauge_ids):
        raise ValueError("observation window order differs from stream gauge order")
    persistent_count, observation_count, identity_sha = (
        prob4d_observation_identity_summary(observation)
    )
    if persistent_count != update.persistent_identity_count:
        raise ValueError("observation persistent-identity count differs from stream")
    if observation_count != update.observation_count:
        raise ValueError("observation row count differs from stream")
    if identity_sha != update.observation_identity_sha256:
        raise ValueError("observation identity digest differs from stream")
    return Prob4DStreamObservationBindingV1(
        stream_artifact_id=cast(str, stream.artifact_id),
        stream_update_id=cast(str, update.update_id),
        update_index=index,
        observation_artifact_id=observation.artifact_id,
        observation_identity_sha256=identity_sha,
        persistent_identity_count=persistent_count,
        observation_count=observation_count,
        causal_frame_stop=update.causal_frame_stop,
    )
