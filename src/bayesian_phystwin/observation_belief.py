"""Portable uncertainty-bearing observations for Bayesian physical twins.

The contract is intentionally independent of a particular perception feeder.
Dense or sparse 3-D observations are flattened into uniquely identified rows.
Local 3x3 covariance blocks capture pointwise uncertainty, while low-rank
factors with shared ``factor_group_ids`` retain coherent gauge or window error.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

OBSERVATION_BELIEF_SCHEMA = "phys4d.observation_belief"
OBSERVATION_BELIEF_VERSION = 1


def array_sha256(values: np.ndarray) -> str:
    """Hash an array including its dtype and shape."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it fully into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


class _FrozenDict(dict):
    __slots__ = ()
    _MUTATORS = frozenset({"clear", "pop", "popitem", "setdefault", "update"})

    def __getattribute__(self, name: str) -> Any:
        if name in type(self)._MUTATORS:
            return self._immutable
        return super().__getattribute__(name)

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> None:
        raise TypeError("metadata is immutable")

    def __setitem__(self, key, value):
        self._immutable(key, value)

    def __delitem__(self, key):
        self._immutable(key)

    def __ior__(self, other) -> _FrozenDict:  # type: ignore[misc]
        self._immutable(other)
        return self

    def __copy__(self) -> dict[str, Any]:
        return _plain_json(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        del memo
        return _plain_json(self)


class _FrozenList(list):
    __slots__ = ()
    _MUTATORS = frozenset(
        {"append", "clear", "extend", "insert", "pop", "remove", "reverse", "sort"}
    )

    def __getattribute__(self, name: str) -> Any:
        if name in type(self)._MUTATORS:
            return self._immutable
        return super().__getattribute__(name)

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> None:
        raise TypeError("metadata is immutable")

    def __setitem__(self, key, value):
        self._immutable(key, value)

    def __delitem__(self, key):
        self._immutable(key)

    def __iadd__(self, other) -> _FrozenList:  # type: ignore[misc]
        self._immutable(other)
        return self

    def __imul__(self, other) -> _FrozenList:  # type: ignore[misc]
        self._immutable(other)
        return self

    def __copy__(self) -> list[Any]:
        return _plain_json(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        del memo
        return _plain_json(self)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze_json(item) for item in value)
    return value


def _validated_metadata(values: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        normalized = json.loads(
            json.dumps(_plain_json(values), sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must be finite JSON data") from error
    return _freeze_json(normalized)


def _validate_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _readonly(
    values: np.ndarray,
    *,
    dtype: np.dtype[Any] | type | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _artifact_id(
    descriptor: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_json(descriptor))
    for name, values in sorted(arrays.items()):
        digest.update(name.encode("utf-8"))
        digest.update(array_sha256(values).encode("ascii"))
    return digest.hexdigest()


def _validate_rotation(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must have finite shape (3, 3)")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-9, rtol=1e-9):
        raise ValueError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-9, rtol=1e-9):
        raise ValueError("rotation must have determinant one")
    return matrix


@dataclass(frozen=True)
class ObservationBeliefV1:
    """Versioned 3-D observation belief with explicit dependence structure.

    ``local_covariance_m2`` is block diagonal across observation rows.
    ``low_rank_factor_m`` stores row blocks of coherent factors. Rows sharing a
    ``factor_group_id`` are driven by the same standard-normal latent vector;
    distinct factor groups are independent in the represented covariance.

    ``correlation_group_ids`` define the effective likelihood groups. Their
    prior nominal probabilities and composite-likelihood weights are provided
    separately so that dense pixels and overlapping windows are not silently
    treated as independent evidence.
    """

    case_id: str
    stream_id: str
    causal_frame_stop: int
    view_names: tuple[str, ...]
    window_names: tuple[str, ...]
    factor_names: tuple[str, ...]
    source_repository: str
    source_revision: str
    source_artifact_sha256: str

    declared_frame_ids: np.ndarray
    mean_xyz_m: np.ndarray
    frame_ids: np.ndarray
    entity_ids: np.ndarray
    view_indices: np.ndarray
    window_indices: np.ndarray
    correlation_group_ids: np.ndarray
    factor_group_ids: np.ndarray
    prior_reliability: np.ndarray
    association_probability: np.ndarray
    local_covariance_m2: np.ndarray
    low_rank_factor_m: np.ndarray
    group_ids: np.ndarray
    group_prior_nominal_probability: np.ndarray
    group_composite_weight: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id or not self.stream_id:
            raise ValueError("case_id and stream_id must be nonempty")
        if self.causal_frame_stop < 1:
            raise ValueError("causal_frame_stop must be positive")
        if not self.view_names or any(not name for name in self.view_names):
            raise ValueError("view_names must contain nonempty names")
        if not self.window_names or any(not name for name in self.window_names):
            raise ValueError("window_names must contain nonempty names")
        if any(not name for name in self.factor_names):
            raise ValueError("factor_names must be nonempty when present")
        if not self.source_repository or not self.source_revision:
            raise ValueError("source repository and revision must be nonempty")
        _validate_sha256(
            self.source_artifact_sha256,
            name="source_artifact_sha256",
        )

        declared_frames = _readonly(self.declared_frame_ids, dtype=np.int64)
        mean = _readonly(self.mean_xyz_m, dtype=np.float64)
        frame_ids = _readonly(self.frame_ids, dtype=np.int64)
        entity_ids = _readonly(self.entity_ids, dtype=np.int64)
        view_indices = _readonly(self.view_indices, dtype=np.int64)
        window_indices = _readonly(self.window_indices, dtype=np.int64)
        correlation_groups = _readonly(self.correlation_group_ids, dtype=np.int64)
        factor_groups = _readonly(self.factor_group_ids, dtype=np.int64)
        prior_reliability = _readonly(self.prior_reliability, dtype=np.float64)
        association_probability = _readonly(
            self.association_probability, dtype=np.float64
        )
        local_covariance = _readonly(self.local_covariance_m2, dtype=np.float64)
        factors = _readonly(self.low_rank_factor_m, dtype=np.float64)
        group_ids = _readonly(self.group_ids, dtype=np.int64)
        group_prior = _readonly(self.group_prior_nominal_probability, dtype=np.float64)
        group_weight = _readonly(self.group_composite_weight, dtype=np.float64)

        if (
            declared_frames.ndim != 1
            or len(declared_frames) == 0
            or np.any(declared_frames < 0)
            or np.any(np.diff(declared_frames) <= 0)
        ):
            raise ValueError(
                "declared_frame_ids must be nonempty, nonnegative, and "
                "strictly increasing"
            )
        if np.any(declared_frames >= self.causal_frame_stop):
            raise ValueError("declared frames must lie before causal_frame_stop")
        prior_only_fallback = (
            isinstance(self.metadata, Mapping)
            and self.metadata.get("prior_only_fallback") is True
        )
        if (
            mean.ndim != 2
            or mean.shape[1] != 3
            or (len(mean) == 0 and not prior_only_fallback)
        ):
            raise ValueError(
                "mean_xyz_m must have shape (N, 3); N may be zero only "
                "for an explicit prior-only fallback"
            )
        observation_count = len(mean)
        vectors = {
            "frame_ids": frame_ids,
            "entity_ids": entity_ids,
            "view_indices": view_indices,
            "window_indices": window_indices,
            "correlation_group_ids": correlation_groups,
            "factor_group_ids": factor_groups,
            "prior_reliability": prior_reliability,
            "association_probability": association_probability,
        }
        for name, values in vectors.items():
            if values.shape != (observation_count,):
                raise ValueError(f"{name} must have shape ({observation_count},)")
        if local_covariance.shape != (observation_count, 3, 3):
            raise ValueError("local_covariance_m2 must have shape (N, 3, 3)")
        factor_rank = len(self.factor_names)
        if factors.shape != (observation_count, 3, factor_rank):
            raise ValueError(f"low_rank_factor_m must have shape (N, 3, {factor_rank})")
        if not np.all(np.isfinite(mean)):
            raise ValueError("mean_xyz_m must be finite")
        if not np.all(np.isfinite(local_covariance)):
            raise ValueError("local covariance must be finite")
        if not np.all(np.isfinite(factors)):
            raise ValueError("low-rank factors must be finite")
        if np.any(entity_ids < 0):
            raise ValueError("entity_ids must be nonnegative")
        if np.any(frame_ids < 0) or np.any(frame_ids >= self.causal_frame_stop):
            raise ValueError("observation frames cross the causal boundary")
        if not np.all(np.isin(frame_ids, declared_frames)):
            raise ValueError("frame_ids must be contained in declared_frame_ids")
        if np.any(view_indices < 0) or np.any(view_indices >= len(self.view_names)):
            raise ValueError("view_indices reference unavailable views")
        if np.any(window_indices < 0) or np.any(
            window_indices >= len(self.window_names)
        ):
            raise ValueError("window_indices reference unavailable windows")
        if np.any(correlation_groups < 0) or np.any(factor_groups < 0):
            raise ValueError("group identifiers must be nonnegative")
        for name, values in (
            ("prior_reliability", prior_reliability),
            ("association_probability", association_probability),
        ):
            if not np.all(np.isfinite(values)) or np.any(
                (values < 0.0) | (values > 1.0)
            ):
                raise ValueError(f"{name} must lie in [0, 1]")

        symmetric = 0.5 * (local_covariance + np.swapaxes(local_covariance, 1, 2))
        if not np.allclose(local_covariance, symmetric, atol=1e-12, rtol=1e-10):
            raise ValueError("local covariance must be symmetric")
        if observation_count:
            minimum_eigenvalue = np.min(
                np.linalg.eigvalsh(symmetric),
                axis=1,
            )
            if np.any(minimum_eigenvalue <= 0.0):
                raise ValueError("local covariance must be positive definite")

        expected_groups = np.unique(correlation_groups)
        if group_ids.ndim != 1 or not np.array_equal(group_ids, expected_groups):
            raise ValueError("group_ids must equal sorted unique correlation_group_ids")
        group_count = len(group_ids)
        if group_prior.shape != (group_count,) or group_weight.shape != (group_count,):
            raise ValueError(
                "group prior and composite weight must identify every group"
            )
        if not np.all(np.isfinite(group_prior)) or np.any(
            (group_prior < 0.0) | (group_prior > 1.0)
        ):
            raise ValueError("group prior nominal probabilities must lie in [0, 1]")
        if not np.all(np.isfinite(group_weight)) or np.any(
            (group_weight <= 0.0) | (group_weight > 1.0)
        ):
            raise ValueError("group composite weights must lie in (0, 1]")

        order = np.lexsort((window_indices, view_indices, entity_ids, frame_ids))
        sorted_keys = np.column_stack(
            (
                frame_ids[order],
                entity_ids[order],
                view_indices[order],
                window_indices[order],
            )
        )
        if len(sorted_keys) > 1 and np.any(
            np.all(sorted_keys[1:] == sorted_keys[:-1], axis=1)
        ):
            raise ValueError(
                "observation identity (frame, entity, view, window) must be unique"
            )

        object.__setattr__(self, "declared_frame_ids", declared_frames)
        object.__setattr__(self, "mean_xyz_m", mean)
        object.__setattr__(self, "frame_ids", frame_ids)
        object.__setattr__(self, "entity_ids", entity_ids)
        object.__setattr__(self, "view_indices", view_indices)
        object.__setattr__(self, "window_indices", window_indices)
        object.__setattr__(self, "correlation_group_ids", correlation_groups)
        object.__setattr__(self, "factor_group_ids", factor_groups)
        object.__setattr__(self, "prior_reliability", prior_reliability)
        object.__setattr__(self, "association_probability", association_probability)
        object.__setattr__(self, "local_covariance_m2", local_covariance)
        object.__setattr__(self, "low_rank_factor_m", factors)
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(self, "group_prior_nominal_probability", group_prior)
        object.__setattr__(self, "group_composite_weight", group_weight)
        object.__setattr__(self, "metadata", _validated_metadata(self.metadata))

    def _descriptor(self) -> dict[str, Any]:
        return {
            "schema_name": OBSERVATION_BELIEF_SCHEMA,
            "schema_version": OBSERVATION_BELIEF_VERSION,
            "case_id": self.case_id,
            "stream_id": self.stream_id,
            "causal_frame_stop": self.causal_frame_stop,
            "view_names": list(self.view_names),
            "window_names": list(self.window_names),
            "factor_names": list(self.factor_names),
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "source_artifact_sha256": self.source_artifact_sha256,
            "metadata": _plain_json(self.metadata),
        }

    def _arrays(self) -> dict[str, np.ndarray]:
        return {
            "declared_frame_ids": self.declared_frame_ids,
            "mean_xyz_m": self.mean_xyz_m,
            "frame_ids": self.frame_ids,
            "entity_ids": self.entity_ids,
            "view_indices": self.view_indices,
            "window_indices": self.window_indices,
            "correlation_group_ids": self.correlation_group_ids,
            "factor_group_ids": self.factor_group_ids,
            "prior_reliability": self.prior_reliability,
            "association_probability": self.association_probability,
            "local_covariance_m2": self.local_covariance_m2,
            "low_rank_factor_m": self.low_rank_factor_m,
            "group_ids": self.group_ids,
            "group_prior_nominal_probability": (self.group_prior_nominal_probability),
            "group_composite_weight": self.group_composite_weight,
        }

    @property
    def artifact_id(self) -> str:
        """Content address of the descriptor and every numeric array."""

        return _artifact_id(self._descriptor(), self._arrays())

    @property
    def observation_count(self) -> int:
        return len(self.mean_xyz_m)

    @property
    def factor_rank(self) -> int:
        return len(self.factor_names)

    def group_position(self, group_id: int) -> int:
        position = int(np.searchsorted(self.group_ids, group_id))
        if position == len(self.group_ids) or self.group_ids[position] != group_id:
            raise KeyError(f"unknown correlation group {group_id}")
        return position

    def transformed(
        self,
        *,
        rotation: np.ndarray,
        translation_m: np.ndarray,
        scale: float = 1.0,
        stream_id: str | None = None,
    ) -> ObservationBeliefV1:
        """Return the same belief in another metric Sim(3) frame."""

        matrix = _validate_rotation(rotation)
        translation = np.asarray(translation_m, dtype=np.float64)
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError("translation_m must have finite shape (3,)")
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and positive")
        mean = scale * np.einsum("ij,nj->ni", matrix, self.mean_xyz_m)
        mean += translation
        covariance = scale**2 * np.einsum(
            "ij,njk,lk->nil",
            matrix,
            self.local_covariance_m2,
            matrix,
        )
        factors = scale * np.einsum("ij,njr->nir", matrix, self.low_rank_factor_m)
        metadata = _plain_json(self.metadata)
        metadata["metric_transform"] = {
            "rotation": matrix.tolist(),
            "translation_m": translation.tolist(),
            "scale": float(scale),
            "source_artifact_id": self.artifact_id,
        }
        return replace(
            self,
            stream_id=stream_id or self.stream_id,
            mean_xyz_m=mean,
            local_covariance_m2=covariance,
            low_rank_factor_m=factors,
            metadata=metadata,
        )

    def summary(self) -> dict[str, Any]:
        group_sizes = np.asarray(
            [
                np.sum(self.correlation_group_ids == group_id)
                for group_id in self.group_ids
            ],
            dtype=np.int64,
        )
        return {
            "artifact_id": self.artifact_id,
            "schema_name": OBSERVATION_BELIEF_SCHEMA,
            "schema_version": OBSERVATION_BELIEF_VERSION,
            "case_id": self.case_id,
            "stream_id": self.stream_id,
            "causal_frame_stop": self.causal_frame_stop,
            "frame_count": int(len(self.declared_frame_ids)),
            "observation_count": self.observation_count,
            "group_count": int(len(self.group_ids)),
            "factor_rank": self.factor_rank,
            "minimum_group_size": int(np.min(group_sizes)),
            "maximum_group_size": int(np.max(group_sizes)),
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "source_artifact_sha256": self.source_artifact_sha256,
        }


def save_observation_belief(path: str | Path, belief: ObservationBeliefV1) -> None:
    """Write a non-pickled, content-addressed observation artifact."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = belief._descriptor()
    descriptor["artifact_id"] = belief.artifact_id
    archive_payload: dict[str, Any] = {
        "descriptor_json": np.asarray(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    }
    archive_payload.update(belief._arrays())
    np.savez_compressed(target, **archive_payload)


def load_observation_belief(path: str | Path) -> ObservationBeliefV1:
    """Load and fully revalidate an ``ObservationBeliefV1`` artifact."""

    with np.load(path, allow_pickle=False) as archive:
        if "descriptor_json" not in archive:
            raise ValueError("observation artifact has no descriptor_json")
        descriptor = json.loads(str(archive["descriptor_json"]))
        if descriptor.get("schema_name") != OBSERVATION_BELIEF_SCHEMA:
            raise ValueError("unsupported observation-belief schema")
        if int(descriptor.get("schema_version", -1)) != (OBSERVATION_BELIEF_VERSION):
            raise ValueError("unsupported observation-belief version")
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "descriptor_json"
        }
    required_arrays = {
        "declared_frame_ids",
        "mean_xyz_m",
        "frame_ids",
        "entity_ids",
        "view_indices",
        "window_indices",
        "correlation_group_ids",
        "factor_group_ids",
        "prior_reliability",
        "association_probability",
        "local_covariance_m2",
        "low_rank_factor_m",
        "group_ids",
        "group_prior_nominal_probability",
        "group_composite_weight",
    }
    missing = required_arrays - arrays.keys()
    extra = arrays.keys() - required_arrays
    if missing or extra:
        raise ValueError(
            "observation artifact arrays changed; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    belief = ObservationBeliefV1(
        case_id=str(descriptor["case_id"]),
        stream_id=str(descriptor["stream_id"]),
        causal_frame_stop=int(descriptor["causal_frame_stop"]),
        view_names=tuple(map(str, descriptor["view_names"])),
        window_names=tuple(map(str, descriptor["window_names"])),
        factor_names=tuple(map(str, descriptor["factor_names"])),
        source_repository=str(descriptor["source_repository"]),
        source_revision=str(descriptor["source_revision"]),
        source_artifact_sha256=str(descriptor["source_artifact_sha256"]),
        metadata=descriptor["metadata"],
        **arrays,
    )
    expected = str(descriptor.get("artifact_id", ""))
    _validate_sha256(expected, name="artifact_id")
    if belief.artifact_id != expected:
        raise ValueError("observation artifact digest does not match its payload")
    return belief


__all__ = [
    "OBSERVATION_BELIEF_SCHEMA",
    "OBSERVATION_BELIEF_VERSION",
    "ObservationBeliefV1",
    "array_sha256",
    "file_sha256",
    "load_observation_belief",
    "save_observation_belief",
]
