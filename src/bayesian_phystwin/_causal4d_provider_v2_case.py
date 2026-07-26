"""Case, graph, and controller operations for provider v2."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from . import causal4d_provider_v1 as _v1
from ._causal4d_provider_v2_types import (
    PhysTwinCase,
    PhysTwinControllerLayout,
    PhysTwinSpringGraph,
    PhysTwinSpringGraphConfig,
    _immutable_array,
)
from .legacy_artifacts import load_trusted_legacy_phystwin_pickle


def _public_spring_graph(graph: object) -> PhysTwinSpringGraph:
    return PhysTwinSpringGraph(
        vertices=np.asarray(getattr(graph, "vertices")),
        springs=np.asarray(getattr(graph, "springs")),
        rest_lengths=np.asarray(getattr(graph, "rest_lengths")),
        masses=np.asarray(getattr(graph, "masses")),
        num_object_springs=int(getattr(graph, "num_object_springs")),
        num_object_points=getattr(graph, "num_object_points", None),
    )


def _internal_spring_graph(graph: PhysTwinSpringGraph | object) -> object:
    public = (
        graph
        if isinstance(graph, PhysTwinSpringGraph)
        else _public_spring_graph(graph)
    )
    module = cast(Any, import_module("bayesian_phystwin.phystwin_graph"))
    return module.PhysTwinSpringGraph(
        vertices=np.asarray(public.vertices, dtype=np.float32).copy(),
        springs=np.asarray(public.springs, dtype=np.int32).copy(),
        rest_lengths=np.asarray(public.rest_lengths, dtype=np.float32).copy(),
        masses=np.asarray(public.masses, dtype=np.float32).copy(),
        num_object_springs=public.num_object_springs,
        num_object_points=public.num_object_points,
    )


def build_phystwin_spring_graph(
    structure_points_m: np.ndarray,
    controller_points_m: np.ndarray | None,
    *,
    config: PhysTwinSpringGraphConfig,
) -> PhysTwinSpringGraph:
    """Build the released graph and return a provider-owned immutable value."""

    module = cast(Any, import_module("bayesian_phystwin.phystwin_graph"))
    internal = module.build_phystwin_spring_graph(
        structure_points_m,
        controller_points_m,
        config=module.PhysTwinSpringGraphConfig(
            object_radius=config.object_radius,
            object_max_neighbours=config.object_max_neighbours,
            controller_radius=config.controller_radius,
            controller_max_neighbours=config.controller_max_neighbours,
        ),
    )
    return _public_spring_graph(internal)


def controller_hand_count(case_name: str) -> int:
    module = cast(
        Any, import_module("bayesian_phystwin.phystwin_controller_sensitivity")
    )
    return int(module.controller_hand_count(case_name))


def infer_controller_groups(
    initial_controller_points_m: np.ndarray, *, group_count: int
) -> np.ndarray:
    module = cast(
        Any, import_module("bayesian_phystwin.phystwin_controller_sensitivity")
    )
    values = module.infer_controller_groups(
        initial_controller_points_m, group_count=group_count
    )
    return _immutable_array(values, dtype=np.int32, name="controller_groups")


def released_controller_layout(
    case_name: str, initial_controller_points_m: np.ndarray
) -> PhysTwinControllerLayout:
    hand_count = controller_hand_count(case_name)
    return PhysTwinControllerLayout(
        hand_count=hand_count,
        group_ids=infer_controller_groups(
            initial_controller_points_m, group_count=hand_count
        ),
    )


def _load_legacy(
    path: str | Path,
    *,
    expected_sha256: str | None,
    artifact_kind: Literal["mapping", "sequence", "ndarray"],
    required_keys: tuple[str, ...] = (),
) -> Any:
    if expected_sha256 is None:
        return _v1.load_pickle(path)
    return load_trusted_legacy_phystwin_pickle(
        path,
        expected_sha256=expected_sha256,
        artifact_kind=artifact_kind,
        required_keys=required_keys,
    )


def load_official_phystwin_case(
    final_data_path: str | Path,
    optimal_params_path: str | Path,
    baseline_trajectory_path: str | Path | None = None,
    *,
    expected_sha256: Mapping[str, str] | None = None,
) -> PhysTwinCase:
    """Load and validate one released case through a schema-specific boundary.

    Supplying ``expected_sha256`` activates digest-bound deserialization. The
    mapping keys are ``final_data``, ``optimal_params``, and, when requested,
    ``baseline_trajectory``. Without a digest mapping this remains a trusted
    local compatibility reader for the released artifacts.
    """

    digests = dict(expected_sha256 or {})
    required_digest_keys = {"final_data", "optimal_params"}
    if baseline_trajectory_path is not None:
        required_digest_keys.add("baseline_trajectory")
    if expected_sha256 is not None:
        missing_digests = sorted(required_digest_keys - set(digests))
        if missing_digests:
            raise ValueError(
                "expected_sha256 is missing: " + ", ".join(missing_digests)
            )
    required_data = (
        "object_points",
        "object_visibilities",
        "object_motions_valid",
        "controller_points",
        "surface_points",
        "interior_points",
    )
    required_optimal = (
        "object_radius",
        "object_max_neighbours",
        "controller_radius",
        "controller_max_neighbours",
    )
    data = _load_legacy(
        final_data_path,
        expected_sha256=digests.get("final_data"),
        artifact_kind="mapping",
        required_keys=required_data,
    )
    optimal = _load_legacy(
        optimal_params_path,
        expected_sha256=digests.get("optimal_params"),
        artifact_kind="mapping",
        required_keys=required_optimal,
    )
    if not isinstance(data, Mapping) or not isinstance(optimal, Mapping):
        raise TypeError("released PhysTwin case and parameters must be mappings")
    missing_data = sorted(set(required_data) - set(data))
    missing_optimal = sorted(set(required_optimal) - set(optimal))
    if missing_data:
        raise ValueError(
            "released PhysTwin case is missing: " + ", ".join(missing_data)
        )
    if missing_optimal:
        raise ValueError(
            "released optimal parameters are missing: " + ", ".join(missing_optimal)
        )
    baseline = None
    if baseline_trajectory_path is not None:
        baseline = _load_legacy(
            baseline_trajectory_path,
            expected_sha256=digests.get("baseline_trajectory"),
            artifact_kind="ndarray",
        )
    return PhysTwinCase(
        case_name=Path(final_data_path).resolve().parent.name,
        object_points_m=np.asarray(data["object_points"]),
        object_visibilities=np.asarray(data["object_visibilities"]),
        object_motions_valid=np.asarray(data["object_motions_valid"]),
        controller_points_m=np.asarray(data["controller_points"]),
        surface_points_m=np.asarray(data["surface_points"]),
        interior_points_m=np.asarray(data["interior_points"]),
        graph_config=PhysTwinSpringGraphConfig(
            object_radius=float(optimal["object_radius"]),
            object_max_neighbours=int(optimal["object_max_neighbours"]),
            controller_radius=float(optimal["controller_radius"]),
            controller_max_neighbours=int(optimal["controller_max_neighbours"]),
        ),
        baseline_trajectory_m=None if baseline is None else np.asarray(baseline),
        _provider_data=data,
        _provider_optimal=optimal,
    )
