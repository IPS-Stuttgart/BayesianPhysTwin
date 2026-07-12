"""Build the immutable material graph shared by a real multi-action protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.phystwin_graph import (
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)
from bayesian_phystwin.phystwin_residual_dynamics import _load_pickle, _sha256
from causal4d.rest_geometry_transfer import write_canonical_material_graph


def build_canonical_material_graph_from_case(
    final_data_path: str | Path,
    optimal_params_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze one fitted twin's object vertices/topology before collection."""

    data = _load_pickle(final_data_path)
    optimal = _load_pickle(optimal_params_path)
    observed = np.asarray(data["object_points"], dtype=float)
    surface = np.asarray(data["surface_points"], dtype=float)
    interior = np.asarray(data["interior_points"], dtype=float)
    structure_points = np.concatenate((observed[0], surface, interior), axis=0)
    config = PhysTwinSpringGraphConfig(
        object_radius=float(optimal["object_radius"]),
        object_max_neighbours=int(optimal["object_max_neighbours"]),
        controller_radius=float(optimal["controller_radius"]),
        controller_max_neighbours=int(optimal["controller_max_neighbours"]),
    )
    graph = build_phystwin_spring_graph(
        structure_points,
        None,
        config=config,
    )
    artifact = write_canonical_material_graph(
        output_path,
        graph.vertices,
        graph.springs,
        graph.rest_lengths,
        graph.masses,
    )
    result = {
        "schema_version": 1,
        "artifact_kind": "canonical_phystwin_material_graph",
        **artifact,
        "inputs": {
            "final_data_sha256": _sha256(final_data_path),
            "optimal_params_sha256": _sha256(optimal_params_path),
        },
        "graph_config": {
            "object_radius": config.object_radius,
            "object_max_neighbours": config.object_max_neighbours,
            "controller_radius": config.controller_radius,
            "controller_max_neighbours": config.controller_max_neighbours,
        },
    }
    metadata_path = Path(output_path).with_suffix(".json")
    metadata_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**result, "metadata_path": str(metadata_path.resolve())}
