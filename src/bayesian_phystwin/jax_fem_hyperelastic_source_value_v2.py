"""Outcome-sealed source-value generator for the qualified JAX-FEM v2 arm."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from ._portable_contracts import content_id, write_atomic_json
from .jax_fem_hyperelastic_source_qualification_v2 import (
    load_jax_fem_hyperelastic_source_protocol_v2,
)
from .jax_fem_hyperelastic_v2 import (
    load_native_jax_fem_modules_v2,
    run_hyperelastic_replay_v2,
)
from .jax_fem_source_qualification_v1 import (
    _git_provenance,
    attachment_targets_m,
    build_tetrahedral_cells_v1,
    contact_patch_local_indices_v1,
    file_sha256,
    load_jax_fem_source_inputs_v1,
    rigid_contact_projection_v1,
)
from .jax_fem_source_value_v1 import (
    GRID_FILENAME,
    GRID_SCHEMA,
    _ordinary_file,
    _physical_arrays,
    _require,
    load_jax_fem_source_value_protocol_v1,
)
from .material_backend_qualification_v1 import (
    load_material_backend_qualification_v1,
    require_qualified_material_backend_runtime,
)
from .physical_rollout_v1 import write_deterministic_npz


def generate_jax_fem_hyperelastic_source_value_predictions_v2(
    *,
    protocol_path: str | Path,
    physics_protocol_path: str | Path,
    physics_result_path: str | Path,
    qualification_path: str | Path,
    group_roots: Mapping[str, str | Path],
    matphys_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Seal all finite-deformation predictions before any outcome is opened."""

    protocol = load_jax_fem_source_value_protocol_v1(protocol_path)
    _require(
        protocol.value["protocol_label"] == "jax-fem-zebra-source-value-v2",
        "v2 source-value protocol label changed",
    )
    _require(
        file_sha256(physics_protocol_path) == protocol.qualification_protocol_sha256,
        "source-physics protocol changed",
    )
    _require(
        file_sha256(physics_result_path) == protocol.qualification_result_sha256,
        "source-physics result changed",
    )
    _require(
        file_sha256(qualification_path) == protocol.qualification_artifact_sha256,
        "qualification artifact changed",
    )
    qualification = load_material_backend_qualification_v1(qualification_path)
    _require(
        qualification.artifact_id == protocol.qualification_artifact_id,
        "qualification artifact ID changed",
    )
    require_qualified_material_backend_runtime(
        profile_id="jax-fem-quasistatic-v1",
        producer_profile_id="jax-fem-quasistatic-v1",
        runtime_id=protocol.runtime_id,
        qualification=qualification,
    )
    expected_groups = {group.group_id for group in protocol.groups}
    _require(
        set(group_roots) == expected_groups and set(matphys_paths) == expected_groups,
        "complete source roots are required",
    )
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    provenance = _git_provenance(
        Path(repo_root).absolute(),
        source_paths=(
            "src/bayesian_phystwin/jax_fem_source_qualification_v1.py",
            "src/bayesian_phystwin/jax_fem_hyperelastic_v2.py",
            ("src/bayesian_phystwin/jax_fem_hyperelastic_source_qualification_v2.py"),
            "src/bayesian_phystwin/jax_fem_source_value_v1.py",
            "src/bayesian_phystwin/jax_fem_hyperelastic_source_value_v2.py",
            "scripts/remote/run_jax_fem_hyperelastic_source_value_v2.py",
        ),
    )

    physics_protocol = load_jax_fem_hyperelastic_source_protocol_v2(
        physics_protocol_path
    )
    _require(
        physics_protocol.runtime_id == protocol.runtime_id,
        "source-value runtime differs from source physics",
    )
    physics_groups = {
        group.group_id: group for group in physics_protocol.base_protocol.source_groups
    }
    _require(set(physics_groups) == expected_groups, "source group roster changed")
    simulation = physics_protocol.simulation
    base_simulation = physics_protocol.base_protocol.simulation
    _require(
        protocol.poisson_ratios
        == (
            float(simulation["low_poisson_ratio"]),
            float(simulation["base_poisson_ratio"]),
            float(simulation["high_poisson_ratio"]),
        )
        and protocol.young_modulus_pa == float(simulation["young_modulus_pa"]),
        "source-value ensemble differs from the qualified physics parameters",
    )
    native = load_native_jax_fem_modules_v2(
        runtime_versions=cast(
            Mapping[str, str], physics_protocol.backend["runtime_versions"]
        ),
        installed_source_sha256=cast(
            Mapping[str, str],
            physics_protocol.backend["installed_source_sha256"],
        ),
    )

    records: list[dict[str, Any]] = []
    for group in protocol.groups:
        root = Path(group_roots[group.group_id]).absolute()
        source_path = root / group.source_inputs_relative_path.as_posix()
        incumbent_path = root / group.incumbent_relative_path.as_posix()
        matphys_path = _ordinary_file(
            matphys_paths[group.group_id],
            name="MatPhys source comparator",
        )
        _require(
            file_sha256(incumbent_path) == group.incumbent_sha256,
            "incumbent changed",
        )
        _require(file_sha256(matphys_path) == group.matphys_sha256, "MatPhys changed")
        physics_group = physics_groups[group.group_id]
        _require(
            physics_group.source_inputs_relative_path
            == group.source_inputs_relative_path
            and physics_group.source_inputs_sha256 == group.source_inputs_sha256
            and physics_group.incumbent_relative_path == group.incumbent_relative_path
            and physics_group.incumbent_sha256 == group.incumbent_sha256
            and physics_group.frame_count == group.frame_count
            and physics_group.material_node_count == group.material_particle_count
            and physics_group.controller_point_count == group.controller_point_count
            and physics_group.attached_node_count == group.attached_particle_count
            and physics_group.expected_base_cell_count == group.base_cell_count
            and physics_group.expected_contact_patch_sizes == group.contact_patch_sizes,
            "source-value group differs from the qualified source-physics group",
        )
        arrays = load_jax_fem_source_inputs_v1(source_path, group=physics_group)
        points = np.asarray(arrays["frame_zero_points_m"], dtype=np.float64)
        controller = np.asarray(arrays["controller_points_m"], dtype=np.float64)
        indices = np.asarray(arrays["attachment_indices"], dtype=np.int64)
        raw_targets = attachment_targets_m(
            points,
            controller,
            indices,
            arrays["attachment_weights"],
        )
        patches = contact_patch_local_indices_v1(
            points,
            indices,
            radius_m=float(base_simulation["contact_cluster_radius_m"]),
        )
        _require(
            tuple(len(patch) for patch in patches)
            == physics_group.expected_contact_patch_sizes,
            "contact patch topology changed",
        )
        contact = rigid_contact_projection_v1(points, indices, raw_targets, patches)
        cells = build_tetrahedral_cells_v1(
            points,
            maximum_edge_m=float(base_simulation["base_mesh_max_edge_m"]),
            minimum_shape_ratio=float(
                base_simulation["minimum_tetrahedron_shape_ratio"]
            ),
        )
        _require(
            len(cells) == physics_group.expected_base_cell_count,
            "source-value tetrahedral topology changed",
        )
        maximum_contact_error = float(
            np.max(np.linalg.norm(contact.projected_targets_m - raw_targets, axis=2))
        )
        group_dir = output / group.group_id
        group_dir.mkdir()
        member_arrays: list[dict[str, npt.NDArray[Any]]] = []
        members: list[dict[str, Any]] = []
        for index, poisson_ratio in enumerate(protocol.poisson_ratios):
            replay = run_hyperelastic_replay_v2(
                native=native,
                points_m=points,
                cells=cells,
                attachment_indices=indices,
                contact=contact,
                young_modulus_pa=protocol.young_modulus_pa,
                poisson_ratio=poisson_ratio,
                interval_substeps=int(simulation["base_interval_substeps"]),
                driven=True,
                newton_absolute_tolerance=float(
                    simulation["newton_absolute_tolerance"]
                ),
                newton_relative_tolerance=float(
                    simulation["newton_relative_tolerance"]
                ),
                hard_minimum_deformation_determinant=float(
                    simulation["hard_minimum_deformation_determinant"]
                ),
            )
            physical = _physical_arrays(
                replay.positions_m,
                frame_zero_m=points,
                action_support=np.asarray(arrays["action_support"]),
            )
            member_path = write_deterministic_npz(
                group_dir / f"member-{index:02d}.npz",
                physical,
            )
            member_arrays.append(physical)
            members.append(
                {
                    "candidate_index": index,
                    "poisson_ratio": poisson_ratio,
                    "young_modulus_pa": protocol.young_modulus_pa,
                    "weight": protocol.weights[index],
                    "physical_archive": f"{group.group_id}/{member_path.name}",
                    "physical_archive_sha256": file_sha256(member_path),
                    "minimum_deformation_determinant": float(
                        np.min(replay.deformation_determinants)
                    ),
                    "maximum_deformation_determinant": float(
                        np.max(replay.deformation_determinants)
                    ),
                    "maximum_node_displacement_m": float(
                        np.max(
                            np.linalg.norm(
                                replay.positions_m - points[None],
                                axis=2,
                            )
                        )
                    ),
                    "status": "success",
                }
            )
        stack = np.stack(
            [
                np.asarray(value["prediction_m"], dtype=np.float64)
                for value in member_arrays
            ]
        )
        mean = np.tensordot(np.asarray(protocol.weights), stack, axes=(0, 0))
        mean_physical = _physical_arrays(
            cast(npt.NDArray[np.floating[Any]], mean),
            frame_zero_m=points,
            action_support=np.asarray(arrays["action_support"]),
        )
        mean_path = write_deterministic_npz(
            group_dir / "ensemble-mean.npz",
            mean_physical,
        )
        final_spread = float(np.sqrt(np.mean(np.var(stack[:, -1], axis=0, ddof=0))))
        records.append(
            {
                "group_id": group.group_id,
                "source_inputs_sha256": group.source_inputs_sha256,
                "incumbent_sha256": group.incumbent_sha256,
                "matphys_sha256": group.matphys_sha256,
                "base_cell_count": len(cells),
                "contact_patch_sizes": [len(patch) for patch in patches],
                "maximum_contact_projection_error_m": maximum_contact_error,
                "members": members,
                "ensemble_mean_archive": f"{group.group_id}/{mean_path.name}",
                "ensemble_mean_sha256": file_sha256(mean_path),
                "final_ensemble_spread_m": final_spread,
            }
        )
    identity: dict[str, Any] = {
        "schema": GRID_SCHEMA,
        "schema_version": 1,
        "protocol_sha256": protocol.sha256,
        "qualification_artifact_id": protocol.qualification_artifact_id,
        "implementation": provenance,
        "groups": records,
        "successful_candidate_count_per_group": len(protocol.poisson_ratios),
        "information_boundary": {
            "source_inputs_read": True,
            "incumbent_and_matphys_predictions_read": True,
            "prefix_outcomes_read": False,
            "future_outcomes_read": False,
            "target_or_held_out_artifact_read": False,
        },
    }
    grid = {**identity, "grid_id": content_id(identity)}
    write_atomic_json(grid, output / GRID_FILENAME, overwrite=False)
    return grid


__all__ = ["generate_jax_fem_hyperelastic_source_value_predictions_v2"]
