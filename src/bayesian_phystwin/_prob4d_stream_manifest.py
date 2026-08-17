"""Portable manifest contracts for append-only Prob4D factor streams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from ._canonical_contracts import (
    canonical_relative_posix_path,
    frozen_finite_json_mapping,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import content_id
from ._prob4d_stream_common import (
    PROB4D_OBSERVATION_FACTOR_STREAM_SCHEMA,
    PROB4D_OBSERVATION_FACTOR_STREAM_VERSION,
    _STREAM_FIELDS,
    _UPDATE_FIELDS,
    _nonempty_literal_string,
    _repository,
    _revision,
    _sha256,
    _string_tuple,
)


@dataclass(frozen=True, slots=True)
class Prob4DObservationFactorStreamUpdateV1:
    """Portable path-independent identity of one Prob4D factor-stream update."""

    update_index: int
    admitted_frame_start: int
    causal_frame_stop: int
    bundle_manifest_path: str
    bundle_manifest_sha256: str
    bundle_payload_sha256: str
    bundle_sequence_id: str
    case_id: str
    stream_id: str
    source_repository: str
    source_revision: str
    factor_count: int
    observation_count: int
    persistent_identity_count: int
    observation_identity_sha256: str
    gauge_ids: Sequence[str]
    previous_update_id: str | None = None
    update_id: str | None = None

    def __post_init__(self) -> None:
        update_index = genuine_integer(
            self.update_index,
            name="update_index",
            minimum=0,
        )
        admitted_start = genuine_integer(
            self.admitted_frame_start,
            name="admitted_frame_start",
            minimum=0,
        )
        causal_stop = genuine_integer(
            self.causal_frame_stop,
            name="causal_frame_stop",
            minimum=1,
        )
        if causal_stop <= admitted_start:
            raise ValueError("causal_frame_stop must exceed admitted_frame_start")
        counts = {
            name: genuine_integer(value, name=name, minimum=1)
            for name, value in (
                ("factor_count", self.factor_count),
                ("observation_count", self.observation_count),
                (
                    "persistent_identity_count",
                    self.persistent_identity_count,
                ),
            )
        }
        path = canonical_relative_posix_path(
            self.bundle_manifest_path,
            name="bundle_manifest_path",
        )
        manifest_sha = _sha256(
            self.bundle_manifest_sha256,
            name="bundle_manifest_sha256",
        )
        payload_sha = _sha256(
            self.bundle_payload_sha256,
            name="bundle_payload_sha256",
        )
        observation_sha = _sha256(
            self.observation_identity_sha256,
            name="observation_identity_sha256",
        )
        sequence_id = _nonempty_literal_string(
            self.bundle_sequence_id,
            name="bundle_sequence_id",
        )
        case_id = _nonempty_literal_string(self.case_id, name="case_id")
        stream_id = _nonempty_literal_string(self.stream_id, name="stream_id")
        repository = _repository(
            self.source_repository,
            name="source_repository",
        )
        revision = _revision(self.source_revision, name="source_revision")
        gauges = _string_tuple(self.gauge_ids, name="gauge_ids")
        previous = self.previous_update_id
        if previous is not None:
            previous = _sha256(previous, name="previous_update_id")
        if update_index == 0 and previous is not None:
            raise ValueError("the first stream update cannot have a predecessor")
        if update_index > 0 and previous is None:
            raise ValueError("a later stream update must bind its predecessor")

        object.__setattr__(self, "update_index", update_index)
        object.__setattr__(self, "admitted_frame_start", admitted_start)
        object.__setattr__(self, "causal_frame_stop", causal_stop)
        object.__setattr__(self, "bundle_manifest_path", path)
        object.__setattr__(self, "bundle_manifest_sha256", manifest_sha)
        object.__setattr__(self, "bundle_payload_sha256", payload_sha)
        object.__setattr__(self, "bundle_sequence_id", sequence_id)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "stream_id", stream_id)
        object.__setattr__(self, "source_repository", repository)
        object.__setattr__(self, "source_revision", revision)
        for name, value in counts.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "observation_identity_sha256", observation_sha)
        object.__setattr__(self, "gauge_ids", gauges)
        object.__setattr__(self, "previous_update_id", previous)

        expected_id = content_id(self.identity_record())
        supplied_id = self.update_id
        if supplied_id is not None:
            supplied_id = _sha256(supplied_id, name="update_id")
            if supplied_id != expected_id:
                raise ValueError("Prob4D stream update_id does not match content")
        object.__setattr__(self, "update_id", expected_id)

    def identity_record(self) -> dict[str, object]:
        return {
            "update_index": self.update_index,
            "admitted_frame_start": self.admitted_frame_start,
            "causal_frame_stop": self.causal_frame_stop,
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "bundle_payload_sha256": self.bundle_payload_sha256,
            "bundle_sequence_id": self.bundle_sequence_id,
            "case_id": self.case_id,
            "stream_id": self.stream_id,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "factor_count": self.factor_count,
            "observation_count": self.observation_count,
            "persistent_identity_count": self.persistent_identity_count,
            "observation_identity_sha256": self.observation_identity_sha256,
            "gauge_ids": list(self.gauge_ids),
            "previous_update_id": self.previous_update_id,
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self.identity_record(),
            "bundle_manifest_path": self.bundle_manifest_path,
            "update_id": self.update_id,
        }

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str = "Prob4D observation-factor stream update",
    ) -> Prob4DObservationFactorStreamUpdateV1:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping")
        if set(value) != _UPDATE_FIELDS:
            raise ValueError(f"{name} fields changed")
        return cls(
            update_index=cast(int, value["update_index"]),
            admitted_frame_start=cast(int, value["admitted_frame_start"]),
            causal_frame_stop=cast(int, value["causal_frame_stop"]),
            bundle_manifest_path=cast(str, value["bundle_manifest_path"]),
            bundle_manifest_sha256=cast(
                str,
                value["bundle_manifest_sha256"],
            ),
            bundle_payload_sha256=cast(
                str,
                value["bundle_payload_sha256"],
            ),
            bundle_sequence_id=cast(str, value["bundle_sequence_id"]),
            case_id=cast(str, value["case_id"]),
            stream_id=cast(str, value["stream_id"]),
            source_repository=cast(str, value["source_repository"]),
            source_revision=cast(str, value["source_revision"]),
            factor_count=cast(int, value["factor_count"]),
            observation_count=cast(int, value["observation_count"]),
            persistent_identity_count=cast(
                int,
                value["persistent_identity_count"],
            ),
            observation_identity_sha256=cast(
                str,
                value["observation_identity_sha256"],
            ),
            gauge_ids=cast(Sequence[str], value["gauge_ids"]),
            previous_update_id=cast(str | None, value["previous_update_id"]),
            update_id=cast(str, value["update_id"]),
        )


@dataclass(frozen=True, slots=True)
class Prob4DObservationFactorStreamV1:
    """Independently revalidated Prob4D append-only factor-stream manifest."""

    sequence_id: str
    case_id: str
    stream_id: str
    source_repository: str
    source_revision: str
    updates: Sequence[Prob4DObservationFactorStreamUpdateV1]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        sequence_id = _nonempty_literal_string(
            self.sequence_id,
            name="sequence_id",
        )
        case_id = _nonempty_literal_string(self.case_id, name="case_id")
        stream_id = _nonempty_literal_string(self.stream_id, name="stream_id")
        repository = _repository(
            self.source_repository,
            name="source_repository",
        )
        revision = _revision(self.source_revision, name="source_revision")
        updates = tuple(self.updates)
        if not updates or any(
            not isinstance(update, Prob4DObservationFactorStreamUpdateV1)
            for update in updates
        ):
            raise ValueError(
                "updates must contain Prob4DObservationFactorStreamUpdateV1 objects"
            )
        previous: Prob4DObservationFactorStreamUpdateV1 | None = None
        for index, update in enumerate(updates):
            if update.update_index != index:
                raise ValueError("stream update indices must be contiguous from zero")
            if update.bundle_sequence_id != sequence_id:
                raise ValueError("stream update sequence_id changed")
            expected_values = {
                "case_id": case_id,
                "stream_id": stream_id,
                "source_repository": repository,
                "source_revision": revision,
            }
            for name, expected in expected_values.items():
                if getattr(update, name) != expected:
                    raise ValueError(f"stream update {name} changed")
            expected_previous = None if previous is None else previous.update_id
            if update.previous_update_id != expected_previous:
                raise ValueError("stream update hash chain is broken")
            if previous is not None and (
                update.admitted_frame_start != previous.causal_frame_stop
            ):
                raise ValueError("stream frame intervals must be contiguous")
            previous = update

        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="Prob4D observation-factor stream metadata",
        )
        object.__setattr__(self, "sequence_id", sequence_id)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "stream_id", stream_id)
        object.__setattr__(self, "source_repository", repository)
        object.__setattr__(self, "source_revision", revision)
        object.__setattr__(self, "updates", updates)
        object.__setattr__(self, "metadata", metadata)

        expected_id = content_id(self.identity_record())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = _sha256(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("Prob4D stream artifact_id does not match content")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def admitted_frame_start(self) -> int:
        return self.updates[0].admitted_frame_start

    @property
    def causal_frame_stop(self) -> int:
        return self.updates[-1].causal_frame_stop

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": PROB4D_OBSERVATION_FACTOR_STREAM_SCHEMA,
            "schema_version": PROB4D_OBSERVATION_FACTOR_STREAM_VERSION,
            "sequence_id": self.sequence_id,
            "case_id": self.case_id,
            "stream_id": self.stream_id,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "metadata": plain_json(self.metadata),
            "updates": [
                update.identity_record() | {"update_id": update.update_id}
                for update in self.updates
            ],
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self.identity_record(),
            "updates": [update.to_record() for update in self.updates],
            "artifact_id": self.artifact_id,
        }

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str = "Prob4D observation-factor stream",
    ) -> Prob4DObservationFactorStreamV1:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping")
        if set(value) != _STREAM_FIELDS:
            raise ValueError(f"{name} fields changed")
        if value["schema"] != PROB4D_OBSERVATION_FACTOR_STREAM_SCHEMA:
            raise ValueError(f"{name} schema changed")
        version = genuine_integer(
            value["schema_version"],
            name=f"{name} schema_version",
            minimum=1,
        )
        if version != PROB4D_OBSERVATION_FACTOR_STREAM_VERSION:
            raise ValueError(f"{name} version changed")
        raw_updates = value["updates"]
        if not isinstance(raw_updates, list) or not raw_updates:
            raise ValueError(f"{name} updates must be a nonempty array")
        return cls(
            sequence_id=cast(str, value["sequence_id"]),
            case_id=cast(str, value["case_id"]),
            stream_id=cast(str, value["stream_id"]),
            source_repository=cast(str, value["source_repository"]),
            source_revision=cast(str, value["source_revision"]),
            updates=tuple(
                Prob4DObservationFactorStreamUpdateV1.from_mapping(
                    update,
                    name=f"{name} update {index}",
                )
                for index, update in enumerate(raw_updates)
            ),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            artifact_id=cast(str, value["artifact_id"]),
        )
