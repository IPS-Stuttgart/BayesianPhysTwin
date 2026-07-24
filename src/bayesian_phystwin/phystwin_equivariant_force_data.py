"""Typed source episodes for equivariant generalized-force training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .phystwin_equivariant_force import (
    EQUIVARIANT_FORCE_CONTRACT,
    EquivariantForceConfig,
    canonicalize_force_edges,
)


FORCE_EPISODE_SCHEMA_VERSION = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(
        json.dumps(array.shape, separators=(",", ":")).encode("ascii")
    )
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _readonly(values: Any, *, dtype: Any) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _json_mapping(values: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        return json.loads(
            json.dumps(dict(values), sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON data") from error


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class EquivariantForceEpisode:
    """One physical rollout and source-only inverse-dynamics supervision."""

    case_id: str
    positions_m: np.ndarray
    velocities_mps: np.ndarray
    rest_positions_m: np.ndarray
    object_edges: np.ndarray
    rest_lengths_m: np.ndarray
    control_displacement_m: np.ndarray
    control_velocity_mps: np.ndarray
    action_support: np.ndarray
    external_support: np.ndarray
    gravity_mps2: np.ndarray
    action_activity: np.ndarray
    regime_probabilities: np.ndarray
    force_targets_sim: np.ndarray
    force_target_variance_sim2: np.ndarray
    force_target_weight: np.ndarray
    force_scale_sim: float
    fit_end_frame: int
    validation_end_frame: int
    frame_dt_s: float
    source_checksums: Mapping[str, str]
    information_boundary: Mapping[str, Any]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be nonempty")
        positions = _readonly(self.positions_m, dtype=np.float32)
        velocities = _readonly(self.velocities_mps, dtype=np.float32)
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("positions_m must have shape (T,N,3)")
        if velocities.shape != positions.shape:
            raise ValueError("velocities_mps must match positions_m")
        frames, nodes, _ = positions.shape
        rest = _readonly(self.rest_positions_m, dtype=np.float32)
        if rest.shape != (nodes, 3):
            raise ValueError("rest_positions_m must have shape (N,3)")
        edges = canonicalize_force_edges(self.object_edges, num_nodes=nodes)
        lengths = _readonly(self.rest_lengths_m, dtype=np.float32)
        if lengths.shape != (len(edges),) or np.any(lengths <= 0.0):
            raise ValueError("rest_lengths_m must be a positive edge vector")
        control_delta = _readonly(self.control_displacement_m, dtype=np.float32)
        control_velocity = _readonly(self.control_velocity_mps, dtype=np.float32)
        action_support = _readonly(self.action_support, dtype=np.float32)
        external_support = _readonly(self.external_support, dtype=np.float32)
        activity = _readonly(self.action_activity, dtype=np.float32)
        regimes = _readonly(self.regime_probabilities, dtype=np.float32)
        targets = _readonly(self.force_targets_sim, dtype=np.float32)
        variance = _readonly(self.force_target_variance_sim2, dtype=np.float32)
        weight = _readonly(self.force_target_weight, dtype=np.float32)
        if control_delta.shape != positions.shape or control_velocity.shape != positions.shape:
            raise ValueError("control fields must match positions_m")
        if action_support.shape == (nodes,):
            action_support = _readonly(
                np.broadcast_to(action_support, (frames, nodes)), dtype=np.float32
            )
        if action_support.shape != (frames, nodes):
            raise ValueError("action_support must have shape (N,) or (T,N)")
        if external_support.shape != (frames, nodes):
            raise ValueError("external_support must have shape (T,N)")
        if activity.shape != (frames,) or np.any((activity < 0.0) | (activity > 1.0)):
            raise ValueError("action_activity must have shape (T,) in [0,1]")
        if regimes.ndim != 2 or regimes.shape[0] != frames:
            raise ValueError("regime_probabilities must have shape (T,R)")
        if np.any(regimes < 0.0) or not np.allclose(
            np.sum(regimes, axis=1), 1.0, atol=1.0e-5, rtol=1.0e-5
        ):
            raise ValueError("regime probabilities must be simplex-valued")
        if targets.shape != positions.shape:
            raise ValueError("force_targets_sim must match positions_m")
        if variance.shape != (frames, nodes) or weight.shape != (frames, nodes):
            raise ValueError("force variance and weights must have shape (T,N)")
        if np.any(variance <= 0.0) or np.any(weight < 0.0):
            raise ValueError("force variance must be positive and weights nonnegative")
        if self.force_scale_sim <= 0.0 or not np.isfinite(
            self.force_scale_sim
        ):
            raise ValueError("force_scale_sim must be positive and finite")
        if not all(
            np.all(np.isfinite(values))
            for values in (
                positions,
                velocities,
                rest,
                lengths,
                control_delta,
                control_velocity,
                action_support,
                external_support,
                activity,
                regimes,
                targets,
                variance,
                weight,
            )
        ):
            raise ValueError("force episode arrays must be finite")
        if np.any((action_support < 0.0) | (action_support > 1.0)) or np.any(
            (external_support < 0.0) | (external_support > 1.0)
        ):
            raise ValueError("support arrays must lie in [0,1]")
        gravity = _readonly(self.gravity_mps2, dtype=np.float32)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError("gravity_mps2 must be a finite 3-vector")
        if not 3 <= self.fit_end_frame < self.validation_end_frame <= frames:
            raise ValueError("episode split must leave a validation suffix")
        if self.frame_dt_s <= 0.0 or not np.isfinite(self.frame_dt_s):
            raise ValueError("frame_dt_s must be positive")
        checksums = dict(sorted(dict(self.source_checksums).items()))
        if not checksums or any(
            not name or not _valid_sha256(digest)
            for name, digest in checksums.items()
        ):
            raise ValueError("source_checksums must contain SHA-256 values")
        boundary = _json_mapping(
            self.information_boundary, name="information_boundary"
        )
        required = {
            "target_future_used_for_episode_construction": False,
            "force_targets_use_state_innovation_once": True,
            "prior_reliability_uses_state_residual": False,
            "force_scale_uses_prefix_only": True,
            "force_values_are_claimed_as_newtons": False,
        }
        if any(boundary.get(key) != value for key, value in required.items()):
            raise ValueError("force episode violates its information boundary")
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "velocities_mps", velocities)
        object.__setattr__(self, "rest_positions_m", rest)
        object.__setattr__(self, "object_edges", edges)
        object.__setattr__(self, "rest_lengths_m", lengths)
        object.__setattr__(self, "control_displacement_m", control_delta)
        object.__setattr__(self, "control_velocity_mps", control_velocity)
        object.__setattr__(self, "action_support", action_support)
        object.__setattr__(self, "external_support", external_support)
        object.__setattr__(self, "gravity_mps2", gravity)
        object.__setattr__(self, "action_activity", activity)
        object.__setattr__(self, "regime_probabilities", regimes)
        object.__setattr__(self, "force_targets_sim", targets)
        object.__setattr__(self, "force_target_variance_sim2", variance)
        object.__setattr__(self, "force_target_weight", weight)
        object.__setattr__(self, "source_checksums", checksums)
        object.__setattr__(self, "information_boundary", boundary)
        object.__setattr__(
            self,
            "diagnostics",
            _json_mapping(self.diagnostics, name="diagnostics"),
        )

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "positions_m": self.positions_m,
            "velocities_mps": self.velocities_mps,
            "rest_positions_m": self.rest_positions_m,
            "object_edges": self.object_edges,
            "rest_lengths_m": self.rest_lengths_m,
            "control_displacement_m": self.control_displacement_m,
            "control_velocity_mps": self.control_velocity_mps,
            "action_support": self.action_support,
            "external_support": self.external_support,
            "gravity_mps2": self.gravity_mps2,
            "action_activity": self.action_activity,
            "regime_probabilities": self.regime_probabilities,
            "force_targets_sim": self.force_targets_sim,
            "force_target_variance_sim2": self.force_target_variance_sim2,
            "force_target_weight": self.force_target_weight,
        }

    def scalar_payload(self) -> dict[str, Any]:
        return {
            "schema_version": FORCE_EPISODE_SCHEMA_VERSION,
            "contract": EQUIVARIANT_FORCE_CONTRACT,
            "case_id": self.case_id,
            "fit_end_frame": self.fit_end_frame,
            "validation_end_frame": self.validation_end_frame,
            "frame_dt_s": self.frame_dt_s,
            "force_scale_sim": self.force_scale_sim,
            "source_checksums": self.source_checksums,
            "information_boundary": self.information_boundary,
            "diagnostics": self.diagnostics,
            "array_digests": {
                name: _array_digest(values)
                for name, values in sorted(self.arrays().items())
            },
        }

    @property
    def artifact_id(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.scalar_payload(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


def write_equivariant_force_episode(
    prefix: str | Path,
    episode: EquivariantForceEpisode,
) -> dict[str, str]:
    base = Path(prefix)
    if base.suffix in {".json", ".npz"}:
        base = base.with_suffix("")
    manifest = base.with_suffix(".json")
    arrays = base.with_suffix(".npz")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_arrays = arrays.with_name(arrays.name + ".tmp")
    temporary_manifest = manifest.with_name(manifest.name + ".tmp")
    with temporary_arrays.open("wb") as handle:
        np.savez_compressed(handle, **episode.arrays())
    payload = episode.scalar_payload()
    payload.update(
        {
            "artifact_id": episode.artifact_id,
            "arrays_file": arrays.name,
            "arrays_file_sha256": _sha256(temporary_arrays),
        }
    )
    temporary_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_arrays.replace(arrays)
    temporary_manifest.replace(manifest)
    return {
        "artifact_id": episode.artifact_id,
        "manifest_sha256": _sha256(manifest),
        "arrays_sha256": _sha256(arrays),
    }


def load_equivariant_force_episode(prefix: str | Path) -> EquivariantForceEpisode:
    base = Path(prefix)
    if base.suffix in {".json", ".npz"}:
        base = base.with_suffix("")
    manifest = base.with_suffix(".json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FORCE_EPISODE_SCHEMA_VERSION:
        raise ValueError("unsupported force episode schema")
    if payload.get("contract") != EQUIVARIANT_FORCE_CONTRACT:
        raise ValueError("force episode contract changed")
    arrays_path = manifest.parent / payload.get(
        "arrays_file", base.with_suffix(".npz").name
    )
    if _sha256(arrays_path) != payload.get("arrays_file_sha256"):
        raise ValueError("force episode array hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    episode = EquivariantForceEpisode(
        case_id=payload["case_id"],
        fit_end_frame=int(payload["fit_end_frame"]),
        validation_end_frame=int(payload["validation_end_frame"]),
        frame_dt_s=float(payload["frame_dt_s"]),
        force_scale_sim=float(payload["force_scale_sim"]),
        source_checksums=payload["source_checksums"],
        information_boundary=payload["information_boundary"],
        diagnostics=payload["diagnostics"],
        **arrays,
    )
    if episode.artifact_id != payload.get("artifact_id"):
        raise ValueError("force episode identity mismatch")
    if episode.scalar_payload()["array_digests"] != payload.get("array_digests"):
        raise ValueError("force episode array identity mismatch")
    return episode


def validate_force_episode_model_compatibility(
    episode: EquivariantForceEpisode,
    config: EquivariantForceConfig,
) -> None:
    """Reject source artifacts incompatible with one frozen model contract."""

    if episode.regime_probabilities.shape[1] != config.regime_dim:
        raise ValueError("episode regime dimension differs from the model")
