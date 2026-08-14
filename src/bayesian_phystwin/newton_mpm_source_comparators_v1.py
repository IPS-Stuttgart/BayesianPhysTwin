"""Deterministically materialize source-only physical comparator archives."""

from __future__ import annotations

import pickle
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from ._portable_contracts import content_id, write_atomic_json
from .newton_mpm_backend_v1 import file_sha256
from .physical_rollout_v1 import (
    validate_physical_rollout_arrays,
    write_deterministic_npz,
)

COMPARATOR_MANIFEST_SCHEMA = "bayesian-phystwin.newton-source-comparators-v1"
COMPARATOR_MANIFEST_FILENAME = "comparator-materialization.json"
INCUMBENT_PHYSICAL_FILENAME = "incumbent-physical.npz"
MATPHYS_PHYSICAL_FILENAME = "matphys-physical.npz"
IMPLEMENTATION_SOURCE_PATHS = frozenset(
    {
        "src/bayesian_phystwin/newton_mpm_source_comparators_v1.py",
        "src/bayesian_phystwin/physical_rollout_v1.py",
        "src/bayesian_phystwin/cli/newton_mpm_backend.py",
    }
)


def _ordinary_file(path: str | Path, *, name: str) -> Path:
    source = Path(path).absolute()
    if (
        not source.is_file()
        or source.is_symlink()
        or any(parent.is_symlink() for parent in source.parents)
    ):
        raise ValueError(f"{name} must be an ordinary non-symlink file")
    return source.resolve(strict=True)


def _load_pickle(path: Path, *, name: str) -> object:
    try:
        with path.open("rb") as stream:
            return pickle.load(stream)
    except (OSError, pickle.PickleError) as error:
        raise ValueError(f"cannot load {name}") from error


def _float_array(value: object, *, name: str, ndim: int) -> npt.NDArray[np.float32]:
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype.kind not in "f":
        raise ValueError(f"{name} must be a floating {ndim}D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _frame_zero_structure(final_data: object) -> npt.NDArray[np.float32]:
    if not isinstance(final_data, Mapping):
        raise ValueError("final_data must contain a mapping")
    source = cast(Mapping[str, Any], final_data)
    required = {"object_points", "surface_points", "interior_points"}
    if not required <= set(source):
        raise ValueError("final_data geometry fields are incomplete")
    object_points = _float_array(
        source["object_points"],
        name="object_points",
        ndim=3,
    )
    surface = _float_array(source["surface_points"], name="surface_points", ndim=2)
    interior = _float_array(
        source["interior_points"],
        name="interior_points",
        ndim=2,
    )
    if object_points.shape[0] < 2 or object_points.shape[2:] != (3,):
        raise ValueError("object_points shape changed")
    if surface.shape[1:] != (3,) or interior.shape[1:] != (3,):
        raise ValueError("static geometry shape changed")
    return np.ascontiguousarray(
        np.concatenate((object_points[0], surface, interior), axis=0),
        dtype=np.float32,
    )


def _trajectory(
    value: object,
    *,
    name: str,
    frame_zero: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    trajectory = _float_array(value, name=name, ndim=3)
    if trajectory.shape[0] < 2 or trajectory.shape[1:] != frame_zero.shape:
        raise ValueError(f"{name} shape differs from frame-zero geometry")
    if not np.array_equal(trajectory[0], frame_zero):
        raise ValueError(f"{name} changed frame-zero material identities")
    return trajectory


def _normalized_action_support(
    trajectory: npt.NDArray[np.float32],
    frame_zero: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    response = np.linalg.norm(
        trajectory.astype(np.float64) - frame_zero.astype(np.float64)[None],
        axis=2,
    )
    maximum_response = np.max(response, axis=0)
    normalization = float(np.max(maximum_response))
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("incumbent comparator contains no action response")
    return np.ascontiguousarray(maximum_response / normalization, dtype=np.float32)


def _physical_arrays(
    trajectory: npt.NDArray[np.float32],
    frame_zero: npt.NDArray[np.float32],
    support: npt.NDArray[np.float32],
) -> dict[str, npt.NDArray[Any]]:
    persistence = np.repeat(frame_zero[None], len(trajectory), axis=0)
    arrays: dict[str, npt.NDArray[Any]] = {
        "prediction_m": trajectory,
        "persistence_m": persistence,
        "driven_readout_m": trajectory,
        "zero_action_readout_m": persistence,
        "action_support": support,
        "frame_zero_points_m": frame_zero,
    }
    return cast(
        dict[str, npt.NDArray[Any]],
        validate_physical_rollout_arrays(
            arrays,
            expected_frame_count=len(trajectory),
        ),
    )


def _implementation_provenance() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot bind comparator implementation") from error
    if status:
        raise RuntimeError("comparator materialization requires a clean Git worktree")
    return {
        "git_head": head,
        "git_worktree_clean": True,
        "source_files": {
            relative: file_sha256(repository / relative)
            for relative in sorted(IMPLEMENTATION_SOURCE_PATHS)
        },
    }


def materialize_source_comparators(
    *,
    final_data_path: str | Path,
    incumbent_trajectory_path: str | Path,
    matphys_trajectory_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create matched six-array comparators without reading metric outcomes."""

    sources = {
        "final_data": _ordinary_file(final_data_path, name="final_data"),
        "incumbent_trajectory": _ordinary_file(
            incumbent_trajectory_path,
            name="incumbent trajectory",
        ),
        "matphys_trajectory": _ordinary_file(
            matphys_trajectory_path,
            name="MatPhys trajectory",
        ),
    }
    output = Path(output_dir).absolute()
    if output.exists():
        raise FileExistsError(output)
    implementation = _implementation_provenance()
    frame_zero = _frame_zero_structure(
        _load_pickle(sources["final_data"], name="final_data")
    )
    incumbent = _trajectory(
        _load_pickle(sources["incumbent_trajectory"], name="incumbent trajectory"),
        name="incumbent trajectory",
        frame_zero=frame_zero,
    )
    matphys = _trajectory(
        _load_pickle(sources["matphys_trajectory"], name="MatPhys trajectory"),
        name="MatPhys trajectory",
        frame_zero=frame_zero,
    )
    if matphys.shape != incumbent.shape:
        raise ValueError("comparator trajectory shapes differ")
    support = _normalized_action_support(incumbent, frame_zero)
    output.mkdir(parents=True)
    incumbent_path = output / INCUMBENT_PHYSICAL_FILENAME
    matphys_path = output / MATPHYS_PHYSICAL_FILENAME
    write_deterministic_npz(
        incumbent_path,
        _physical_arrays(incumbent, frame_zero, support),
    )
    write_deterministic_npz(
        matphys_path,
        _physical_arrays(matphys, frame_zero, support),
    )
    identity: dict[str, Any] = {
        "schema": COMPARATOR_MANIFEST_SCHEMA,
        "schema_version": 1,
        "implementation": implementation,
        "information_boundary": {
            "source_role": "already-open-development-source",
            "final_data_pickle_deserialized": True,
            "object_geometry_frames_used": [0],
            "model_prediction_trajectories_read": True,
            "object_metric_outcomes_scored": False,
            "target_or_held_out_artifact_read": False,
        },
        "semantics": {
            "prediction": "byte-preserving float32 source trajectory",
            "zero_action": "exact persistence placeholder; not a measured zero-action comparator",
            "action_support": "incumbent maximum displacement normalized by its global maximum",
            "fixed_material_identity": True,
            "position_units": "m",
        },
        "geometry": {
            "frame_count": len(incumbent),
            "material_query_count": len(frame_zero),
        },
        "sources": {
            name: {
                "sha256": file_sha256(path),
                "byte_count": path.stat().st_size,
            }
            for name, path in sorted(sources.items())
        },
        "artifacts": {
            INCUMBENT_PHYSICAL_FILENAME: {
                "sha256": file_sha256(incumbent_path),
                "byte_count": incumbent_path.stat().st_size,
            },
            MATPHYS_PHYSICAL_FILENAME: {
                "sha256": file_sha256(matphys_path),
                "byte_count": matphys_path.stat().st_size,
            },
        },
    }
    manifest = {**identity, "materialization_id": content_id(identity)}
    write_atomic_json(
        manifest,
        output / COMPARATOR_MANIFEST_FILENAME,
        overwrite=False,
    )
    return manifest


__all__ = ["materialize_source_comparators"]
