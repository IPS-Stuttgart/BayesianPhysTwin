"""Typed, prefix-only discrepancy localization for PhysTwin rollouts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._canonical_contracts import immutable_array

DYNAMIC_DISCREPANCY_SCHEMA_VERSION = 1
LOCALIZATION_GRAPH_RANK = 4


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _readonly(values: np.ndarray, *, dtype: Any = float) -> np.ndarray:
    return immutable_array(values, dtype=dtype)


def _json_data(values: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(values), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON data") from error


@dataclass(frozen=True)
class DynamicDiscrepancyCorrection:
    """One information-matched graph correction fitted at an O-plus prefix."""

    case_id: str
    graph_basis: np.ndarray
    graph_eigenvalues: np.ndarray
    position_coefficients_m: np.ndarray
    velocity_coefficients_mps: np.ndarray
    generalized_force_coefficients_n: np.ndarray
    structural_coefficients_m: np.ndarray
    prefix_frame_start: int
    prefix_frame_stop: int
    frame_dt_s: float
    information_boundary: Mapping[str, Any]
    regularization: Mapping[str, Any]
    source_checksums: Mapping[str, str]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be nonempty")
        basis = _readonly(self.graph_basis)
        eigenvalues = _readonly(self.graph_eigenvalues)
        if basis.ndim != 2 or basis.shape[1] != LOCALIZATION_GRAPH_RANK:
            raise ValueError(
                f"graph_basis must have shape (N, {LOCALIZATION_GRAPH_RANK})"
            )
        rank = basis.shape[1]
        if eigenvalues.shape != (rank,) or np.any(eigenvalues < -1e-10):
            raise ValueError("graph eigenvalues must be a nonnegative rank vector")
        if not np.all(np.isfinite(basis)) or not np.all(np.isfinite(eigenvalues)):
            raise ValueError("graph basis arrays must be finite")
        if not np.allclose(basis.T @ basis, np.eye(rank), atol=1e-7, rtol=1e-7):
            raise ValueError("graph_basis must be orthonormal")
        coefficient_names = (
            "position_coefficients_m",
            "velocity_coefficients_mps",
            "generalized_force_coefficients_n",
            "structural_coefficients_m",
        )
        for name in coefficient_names:
            values = _readonly(getattr(self, name))
            if values.shape != (rank, 3) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must have finite shape (rank, 3)")
            object.__setattr__(self, name, values)
        if self.prefix_frame_start < 0 or self.prefix_frame_stop <= self.prefix_frame_start:
            raise ValueError("prefix interval must be nonempty and nonnegative")
        if self.frame_dt_s <= 0.0 or not np.isfinite(self.frame_dt_s):
            raise ValueError("frame_dt_s must be positive and finite")
        boundary = _json_data(self.information_boundary, name="information_boundary")
        required = {
            "o_plus_prefix_frames": 6,
            "future_frames_used_for_fit_or_selection": False,
            "manual_tracks_used_for_fit_or_selection": False,
            "graph_rank": LOCALIZATION_GRAPH_RANK,
        }
        if any(boundary.get(key) != value for key, value in required.items()):
            raise ValueError("dynamic discrepancy artifact violates its information boundary")
        if self.prefix_frame_stop - self.prefix_frame_start != 7:
            raise ValueError("prefix must contain the endpoint and exactly six O-plus frames")
        checksums = dict(self.source_checksums)
        if not checksums or any(
            not key or not _is_sha256(value) for key, value in checksums.items()
        ):
            raise ValueError("source_checksums must map names to SHA-256 digests")
        object.__setattr__(self, "graph_basis", basis)
        object.__setattr__(self, "graph_eigenvalues", eigenvalues)
        object.__setattr__(self, "information_boundary", boundary)
        object.__setattr__(
            self,
            "regularization",
            _json_data(self.regularization, name="regularization"),
        )
        object.__setattr__(self, "source_checksums", dict(sorted(checksums.items())))
        object.__setattr__(
            self, "diagnostics", _json_data(self.diagnostics, name="diagnostics")
        )
        object.__setattr__(self, "metadata", _json_data(self.metadata, name="metadata"))

    @property
    def rank(self) -> int:
        return self.graph_basis.shape[1]

    def position_field_m(self) -> np.ndarray:
        return self.graph_basis @ self.position_coefficients_m

    def velocity_field_mps(self) -> np.ndarray:
        return self.graph_basis @ self.velocity_coefficients_mps

    def generalized_force_field_n(self) -> np.ndarray:
        return self.graph_basis @ self.generalized_force_coefficients_n

    def structural_field_m(self) -> np.ndarray:
        return self.graph_basis @ self.structural_coefficients_m

    def _scalar_payload(self) -> dict[str, Any]:
        return {
            "schema_version": DYNAMIC_DISCREPANCY_SCHEMA_VERSION,
            "artifact_kind": "DynamicDiscrepancyCorrection",
            "case_id": self.case_id,
            "rank": self.rank,
            "prefix_frame_start": self.prefix_frame_start,
            "prefix_frame_stop": self.prefix_frame_stop,
            "frame_dt_s": self.frame_dt_s,
            "information_boundary": self.information_boundary,
            "regularization": self.regularization,
            "source_checksums": self.source_checksums,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        }

    def _array_payload(self) -> dict[str, np.ndarray]:
        return {
            "graph_basis": self.graph_basis,
            "graph_eigenvalues": self.graph_eigenvalues,
            "position_coefficients_m": self.position_coefficients_m,
            "velocity_coefficients_mps": self.velocity_coefficients_mps,
            "generalized_force_coefficients_n": self.generalized_force_coefficients_n,
            "structural_coefficients_m": self.structural_coefficients_m,
        }

    @property
    def artifact_id(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                self._scalar_payload(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        for name, values in sorted(self._array_payload().items()):
            digest.update(name.encode("ascii"))
            digest.update(_array_sha256(values).encode("ascii"))
        return digest.hexdigest()


def project_prefix_graph_coefficients(
    residual_m: np.ndarray,
    valid: np.ndarray,
    graph_basis: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    """Project only the supplied prefix residual onto a fixed graph basis."""

    residual = np.asarray(residual_m, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    basis = np.asarray(graph_basis, dtype=float)
    if residual.ndim != 3 or residual.shape[2] != 3:
        raise ValueError("residual_m must have shape (T, observed_node, 3)")
    if mask.shape != residual.shape[:2]:
        raise ValueError("valid must have shape (T, observed_node)")
    if basis.ndim != 2 or basis.shape[1] != LOCALIZATION_GRAPH_RANK:
        raise ValueError("graph_basis must use the frozen localization rank")
    if residual.shape[1] > basis.shape[0] or ridge <= 0.0:
        raise ValueError("basis coverage and ridge must be valid")
    observed_basis = basis[: residual.shape[1]]
    identity = np.eye(basis.shape[1])
    coefficients = np.zeros((len(residual), basis.shape[1], 3), dtype=float)
    for frame in range(len(residual)):
        selected = mask[frame] & np.all(np.isfinite(residual[frame]), axis=1)
        if np.sum(selected) < basis.shape[1]:
            raise ValueError("too few valid prefix nodes for graph projection")
        design = observed_basis[selected]
        coefficients[frame] = np.linalg.solve(
            design.T @ design + ridge * identity,
            design.T @ residual[frame, selected],
        )
    return coefficients


def prefix_position_velocity_coefficients(
    residual_m: np.ndarray,
    valid: np.ndarray,
    graph_basis: np.ndarray,
    *,
    frame_dt_s: float,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate persistent position and local-linear velocity at prefix end."""

    if frame_dt_s <= 0.0 or not np.isfinite(frame_dt_s):
        raise ValueError("frame_dt_s must be positive and finite")
    coefficients = project_prefix_graph_coefficients(
        residual_m,
        valid,
        graph_basis,
        ridge=ridge,
    )
    if len(coefficients) < 3:
        raise ValueError("at least three prefix frames are required")
    times = frame_dt_s * np.arange(len(coefficients), dtype=float)
    centered = times - np.mean(times)
    velocity = np.tensordot(centered, coefficients, axes=(0, 0)) / float(
        centered @ centered
    )
    return coefficients[-1].copy(), velocity, coefficients


def fit_dimensionless_linearized_correction(
    residual_m: np.ndarray,
    valid: np.ndarray,
    response_at_step_m: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit dimensionless perturbation weights from prefix-only responses.

    ``response_at_step_m`` contains the nonlinear Warp displacement caused by
    one predeclared perturbation step for each parameter. The returned vector
    therefore multiplies those steps and is unitless.
    """

    residual = np.asarray(residual_m, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    response = np.asarray(response_at_step_m, dtype=float)
    if residual.ndim != 3 or residual.shape[2] != 3:
        raise ValueError("residual_m must have shape (T, N, 3)")
    if mask.shape != residual.shape[:2]:
        raise ValueError("valid must have shape (T, N)")
    if response.shape[:3] != residual.shape or response.ndim != 4:
        raise ValueError("response_at_step_m must have shape (T, N, 3, P)")
    if ridge <= 0.0 or not np.all(np.isfinite(response)):
        raise ValueError("linearized response and ridge must be valid")
    coordinate_mask = np.repeat(mask[:, :, None], 3, axis=2)
    coordinate_mask &= np.all(np.isfinite(residual), axis=2)[:, :, None]
    target = residual[coordinate_mask]
    design = response[coordinate_mask]
    if len(target) <= response.shape[3]:
        raise ValueError("prefix has too little support for linearized fitting")
    gram = design.T @ design
    coefficients = np.linalg.solve(
        gram + ridge * np.eye(response.shape[3]),
        design.T @ target,
    )
    prediction = design @ coefficients
    baseline_rmse = float(np.sqrt(np.mean(np.square(target))))
    fitted_rmse = float(np.sqrt(np.mean(np.square(prediction - target))))
    return coefficients, {
        "valid_coordinate_count": int(len(target)),
        "baseline_prefix_rmse_m": baseline_rmse,
        "linearized_prefix_rmse_m": fitted_rmse,
        "linearized_prefix_improvement_fraction": (
            1.0 - fitted_rmse / baseline_rmse if baseline_rmse > 0.0 else 0.0
        ),
        "dimensionless_coefficient_l2": float(np.linalg.norm(coefficients)),
        "dimensionless_coefficient_maximum": float(
            np.max(np.abs(coefficients), initial=0.0)
        ),
    }


def scale_coefficients_to_field_limit(
    graph_basis: np.ndarray,
    coefficients: np.ndarray,
    *,
    maximum_node_norm: float,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Apply one radial plausibility limit without changing field direction."""

    basis = np.asarray(graph_basis, dtype=float)
    values = np.asarray(coefficients, dtype=float)
    if values.shape != (basis.shape[1], 3) or maximum_node_norm <= 0.0:
        raise ValueError("coefficient shape and maximum_node_norm must be valid")
    field = basis @ values
    before = float(np.max(np.linalg.norm(field, axis=1), initial=0.0))
    scale = min(1.0, maximum_node_norm / max(before, np.finfo(float).tiny))
    limited = values * scale
    after = float(np.max(np.linalg.norm(basis @ limited, axis=1), initial=0.0))
    return limited, {
        "limit_applied": bool(scale < 1.0),
        "radial_scale": float(scale),
        "maximum_node_norm_before": before,
        "maximum_node_norm_after": after,
        "declared_maximum_node_norm": float(maximum_node_norm),
    }


def write_dynamic_discrepancy_correction(
    path: str | Path,
    correction: DynamicDiscrepancyCorrection,
) -> dict[str, Any]:
    """Write a checksummed JSON/NPZ pair without pickled payloads."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = target.with_suffix(".json")
    arrays_path = target.with_suffix(".npz")
    np.savez_compressed(arrays_path, **correction._array_payload())
    manifest = {
        **correction._scalar_payload(),
        "artifact_id": correction.artifact_id,
        "arrays_path": arrays_path.name,
        "arrays_sha256": _file_sha256(arrays_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "artifact_id": correction.artifact_id,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _file_sha256(manifest_path),
        "arrays_path": str(arrays_path.resolve()),
        "arrays_sha256": manifest["arrays_sha256"],
    }


def load_dynamic_discrepancy_correction(
    path: str | Path,
) -> DynamicDiscrepancyCorrection:
    """Load and revalidate a dynamic discrepancy artifact."""

    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_kind") != "DynamicDiscrepancyCorrection":
        raise ValueError("manifest is not a DynamicDiscrepancyCorrection")
    if int(manifest.get("schema_version", -1)) != DYNAMIC_DISCREPANCY_SCHEMA_VERSION:
        raise ValueError("unsupported dynamic discrepancy schema version")
    arrays_path = Path(manifest["arrays_path"])
    if not arrays_path.is_absolute():
        arrays_path = manifest_path.parent / arrays_path
    if _file_sha256(arrays_path) != manifest["arrays_sha256"]:
        raise ValueError("dynamic discrepancy arrays checksum mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    correction = DynamicDiscrepancyCorrection(
        case_id=str(manifest["case_id"]),
        prefix_frame_start=int(manifest["prefix_frame_start"]),
        prefix_frame_stop=int(manifest["prefix_frame_stop"]),
        frame_dt_s=float(manifest["frame_dt_s"]),
        information_boundary=manifest["information_boundary"],
        regularization=manifest["regularization"],
        source_checksums=manifest["source_checksums"],
        diagnostics=manifest["diagnostics"],
        metadata=manifest["metadata"],
        **arrays,
    )
    if correction.artifact_id != manifest["artifact_id"]:
        raise ValueError("dynamic discrepancy artifact digest mismatch")
    return correction
