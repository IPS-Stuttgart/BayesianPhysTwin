#!/usr/bin/env python3
"""Run the exact-runtime native smoke for pose-canonical SOFA FEM v3."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin.jax_fem_source_qualification_v1 import (
    RigidContactProjectionV1,
    rigid_contact_projection_v1,
    rigid_transform_v1,
)
from bayesian_phystwin.physical_rollout_v1 import write_deterministic_npz
from bayesian_phystwin.sofa_fem_canonical_source_v3 import (
    BACKEND_VARIANT,
    CANONICAL_ROUNDING_M,
    COORDINATE_POLICY,
    MINIMUM_RELATIVE_EIGENGAP,
    SofaCanonicalSourceReplayV3,
    run_sofa_fem_canonical_source_replay_v3,
)
from bayesian_phystwin.sofa_fem_source_qualification_v2 import file_sha256
from bayesian_phystwin.sofa_fem_source_v1 import (
    SOFA_ARCHIVE_SHA256,
    SOFA_REPOSITORY,
    SOFA_REVISION,
    SOFA_VERSION,
    NativeSofaFemModulesV1,
    load_native_sofa_fem_modules_v1,
)

RESULT_SCHEMA: Final = "bayesian-phystwin.sofa-fem-canonical-native-smoke-v3"
RESULT_FILENAME: Final = "sofa-fem-canonical-native-smoke-v3.json"
TRAJECTORY_FILENAME: Final = "sofa-fem-canonical-native-smoke-v3.npz"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    source_paths = (
        "src/bayesian_phystwin/_portable_contracts.py",
        "src/bayesian_phystwin/jax_fem_source_qualification_v1.py",
        "src/bayesian_phystwin/native_tet_fem_source_v1.py",
        "src/bayesian_phystwin/physical_rollout_v1.py",
        "src/bayesian_phystwin/sofa_fem_source_v1.py",
        "src/bayesian_phystwin/sofa_fem_kinematic_source_v2.py",
        "src/bayesian_phystwin/sofa_fem_canonical_source_v3.py",
        "src/bayesian_phystwin/sofa_fem_source_qualification_v2.py",
        "scripts/remote/run_sofa_fem_canonical_native_smoke_v3.py",
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(
        len(head) == 40 and all(character in "0123456789abcdef" for character in head),
        "Git revision is not canonical",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(status == "", "native smoke requires a clean Git worktree")
    return {
        "git_head": head,
        "git_worktree_clean": True,
        "source_files": {
            relative: file_sha256(repo_root / relative) for relative in source_paths
        },
    }


def _synthetic_source() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    RigidContactProjectionV1,
]:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.014, 0.0, 0.0],
            [0.0, 0.009, 0.0],
            [0.0, 0.0, 0.006],
            [0.002, 0.003, 0.001],
        ],
        dtype=np.float64,
    )
    cells = np.asarray([[0, 1, 2, 4], [0, 1, 4, 3]], dtype=np.int32)
    indices = np.arange(4, dtype=np.int64)
    local_rotation = rigid_transform_v1([2.0, -1.0, 3.0], 0.02)
    translation = np.asarray([0.0002, -0.0001, 0.00005], dtype=np.float64)
    targets = np.stack(
        (
            points[indices],
            points[indices] @ local_rotation.T + translation,
        )
    )
    contact = rigid_contact_projection_v1(
        points,
        indices,
        targets,
        (np.arange(4, dtype=np.int64),),
    )
    return points, cells, indices, contact


def _transformed_source(
    points_m: np.ndarray,
    attachment_indices: np.ndarray,
    contact: RigidContactProjectionV1,
) -> tuple[np.ndarray, RigidContactProjectionV1, np.ndarray, np.ndarray]:
    rotation = rigid_transform_v1([1.0, 2.0, 3.0], 0.37)
    translation = np.asarray([0.13, -0.08, 0.11], dtype=np.float64)
    points = np.ascontiguousarray(points_m @ rotation.T + translation)
    targets = np.ascontiguousarray(
        contact.projected_targets_m @ rotation.T + translation
    )
    transformed_contact = rigid_contact_projection_v1(
        points,
        attachment_indices,
        targets,
        contact.patch_local_indices,
    )
    return points, transformed_contact, rotation, translation


def _run(
    *,
    native: NativeSofaFemModulesV1,
    points_m: np.ndarray,
    cells: np.ndarray,
    attachment_indices: np.ndarray,
    contact: RigidContactProjectionV1,
) -> SofaCanonicalSourceReplayV3:
    return run_sofa_fem_canonical_source_replay_v3(
        native=native,
        points_m=points_m,
        cells=cells,
        attachment_indices=attachment_indices,
        contact=contact,
        driven=True,
        integrator_time_step_s=1.0 / 3000.0,
        interval_substeps=10,
        young_modulus_pa=100_000.0,
        poisson_ratio=0.3,
        density_kg_m3=1000.0,
        rayleigh_stiffness=0.1,
        rayleigh_mass=0.1,
        hard_minimum_deformation_determinant=0.35,
    )


def run_sofa_fem_canonical_native_smoke_v3(
    *,
    distribution_archive: str | Path,
    sofa_root: str | Path,
    repo_root: str | Path,
    output_dir: str | Path,
) -> Mapping[str, Any]:
    output = Path(output_dir).absolute()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    provenance = _git_provenance(Path(repo_root).absolute())
    native = load_native_sofa_fem_modules_v1(
        distribution_archive=distribution_archive,
        sofa_root=sofa_root,
    )
    points, cells, indices, contact = _synthetic_source()
    transformed_points, transformed_contact, rotation, translation = (
        _transformed_source(points, indices, contact)
    )
    base = _run(
        native=native,
        points_m=points,
        cells=cells,
        attachment_indices=indices,
        contact=contact,
    )
    repeat = _run(
        native=native,
        points_m=points,
        cells=cells,
        attachment_indices=indices,
        contact=contact,
    )
    transformed = _run(
        native=native,
        points_m=transformed_points,
        cells=cells,
        attachment_indices=indices,
        contact=transformed_contact,
    )
    transformed_in_source = np.ascontiguousarray(
        (transformed.positions_m - translation) @ rotation
    )
    equivariance = float(
        np.max(np.linalg.norm(transformed_in_source - base.positions_m, axis=2))
    )
    determinants = np.concatenate(
        (
            base.deformation_determinants.reshape(-1),
            repeat.deformation_determinants.reshape(-1),
            transformed.deformation_determinants.reshape(-1),
        )
    )
    deterministic = bool(
        np.array_equal(base.positions_m, repeat.positions_m)
        and np.array_equal(
            base.deformation_determinants,
            repeat.deformation_determinants,
        )
        and base.scene_sha256 == repeat.scene_sha256
        and base.schedule_sha256 == repeat.schedule_sha256
        and base.gauge_sha256 == repeat.gauge_sha256
    )
    maximum_native_attachment_error_m = max(
        base.maximum_attachment_error_m,
        repeat.maximum_attachment_error_m,
        transformed.maximum_attachment_error_m,
    )
    maximum_world_attachment_approximation_error_m = max(
        base.maximum_world_attachment_approximation_error_m,
        repeat.maximum_world_attachment_approximation_error_m,
        transformed.maximum_world_attachment_approximation_error_m,
    )
    minimum_deformation_determinant = float(np.min(determinants))
    maximum_deformation_determinant = float(np.max(determinants))
    checks = {
        "deterministic_replay": deterministic,
        "gauge_identity_under_rigid_pose": (
            base.gauge_sha256 == transformed.gauge_sha256
        ),
        "scene_identity_under_rigid_pose": (
            base.scene_sha256 == transformed.scene_sha256
        ),
        "maximum_rigid_equivariance_error_m": equivariance,
        "allowed_rigid_equivariance_error_m": 1.0e-12,
        "maximum_native_attachment_error_m": maximum_native_attachment_error_m,
        "allowed_native_attachment_error_m": 1.0e-12,
        "maximum_world_attachment_approximation_error_m": (
            maximum_world_attachment_approximation_error_m
        ),
        "allowed_world_attachment_approximation_error_m": 2.0e-11,
        "minimum_deformation_determinant": minimum_deformation_determinant,
        "maximum_deformation_determinant": maximum_deformation_determinant,
        "allowed_deformation_determinant_interval": [0.5, 2.0],
    }
    passed = bool(
        deterministic
        and checks["gauge_identity_under_rigid_pose"]
        and checks["scene_identity_under_rigid_pose"]
        and equivariance <= 1.0e-12
        and maximum_native_attachment_error_m <= 1.0e-12
        and maximum_world_attachment_approximation_error_m <= 2.0e-11
        and minimum_deformation_determinant >= 0.5
        and maximum_deformation_determinant <= 2.0
    )
    trajectory_path = write_deterministic_npz(
        output / TRAJECTORY_FILENAME,
        {
            "base_positions_m": base.positions_m,
            "repeat_positions_m": repeat.positions_m,
            "transformed_positions_m": transformed.positions_m,
            "base_deformation_determinant": base.deformation_determinants,
            "transformed_deformation_determinant": (
                transformed.deformation_determinants
            ),
        },
    )
    identity: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 3,
        "claim_boundary": (
            "Synthetic native-execution, pose-canonicalization, keyed-Dirichlet, "
            "and provenance smoke only; no source-value, source-outcome, target, "
            "held-out, or state-of-the-art claim."
        ),
        "backend_profile": "sofa-fem-v1",
        "backend_variant": BACKEND_VARIANT,
        "coordinate_policy": COORDINATE_POLICY,
        "canonical_rounding_m": CANONICAL_ROUNDING_M,
        "minimum_relative_eigengap": MINIMUM_RELATIVE_EIGENGAP,
        "engine": {
            "repository": SOFA_REPOSITORY,
            "revision": SOFA_REVISION,
            "version": SOFA_VERSION,
            "distribution_archive_sha256": SOFA_ARCHIVE_SHA256,
            "installed_records": native.installed_records,
        },
        "implementation": provenance,
        "runtime": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        "trajectory_archive_sha256": file_sha256(trajectory_path),
        "base_scene_sha256": base.scene_sha256,
        "base_schedule_sha256": base.schedule_sha256,
        "gauge_sha256": base.gauge_sha256,
        "checks": checks,
        "passed": passed,
        "information_boundary": {
            "dataset_payload_read": False,
            "source_object_outcomes_read": False,
            "target_or_held_out_artifact_read": False,
            "future_outcomes_read": False,
        },
    }
    result = {**identity, "smoke_id": content_id(identity)}
    write_atomic_json(result, output / RESULT_FILENAME, overwrite=False)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution-archive", type=Path, required=True)
    parser.add_argument("--sofa-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_sofa_fem_canonical_native_smoke_v3(
        distribution_archive=args.distribution_archive,
        sofa_root=args.sofa_root,
        repo_root=args.repo_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
