"""Replay and endpoint operations for provider v2."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import causal4d_provider_v1 as _v1
from ._causal4d_provider_v2_case import _internal_spring_graph
from ._causal4d_provider_v2_types import PhysTwinCase, PhysTwinSpringGraph

PhysTwinReplayProvider = _v1.PhysTwinReplayProvider
OfficialPhysTwinReplayProvider = _v1.OfficialPhysTwinReplayProvider
FIXED_PROCESS_STD_M = 0.005
FIXED_OBSERVATION_STD_M = 0.001
FIXED_INITIAL_STD_M = 0.01
FIXED_INLIER_PRIOR = 0.95
FIXED_OUTLIER_VARIANCE_MULTIPLIER = 100.0


def create_official_replay_provider(
    official_repo: str | Path,
    data: Mapping[str, object],
    optimal: Mapping[str, object],
    checkpoint_path: str | Path,
    graph: PhysTwinSpringGraph | object,
    *,
    num_surface_points: int,
    original_count: int,
    dt: float,
    num_substeps: int,
    self_collision: bool,
    deterministic_spring_forces: bool = False,
    spring_parameterization: str = "dense",
    device: str,
) -> OfficialPhysTwinReplayProvider:
    return _v1.create_official_replay_provider(
        official_repo,
        data,
        optimal,
        checkpoint_path,
        _internal_spring_graph(graph),
        num_surface_points=num_surface_points,
        original_count=original_count,
        dt=dt,
        num_substeps=num_substeps,
        self_collision=self_collision,
        deterministic_spring_forces=deterministic_spring_forces,
        spring_parameterization=spring_parameterization,
        device=device,
    )


def create_official_case_replay_provider(
    official_repo: str | Path,
    case: PhysTwinCase,
    checkpoint_path: str | Path,
    graph: PhysTwinSpringGraph | object,
    *,
    dt: float,
    num_substeps: int,
    self_collision: bool,
    deterministic_spring_forces: bool = False,
    spring_parameterization: str = "dense",
    device: str,
) -> OfficialPhysTwinReplayProvider:
    if not isinstance(case, PhysTwinCase):
        raise TypeError("case must be a PhysTwinCase")
    return create_official_replay_provider(
        official_repo,
        case._provider_data,
        case._provider_optimal,
        checkpoint_path,
        graph,
        num_surface_points=case.num_surface_points,
        original_count=case.original_count,
        dt=dt,
        num_substeps=num_substeps,
        self_collision=self_collision,
        deterministic_spring_forces=deterministic_spring_forces,
        spring_parameterization=spring_parameterization,
        device=device,
    )


def robust_random_walk_endpoint(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
    process_variance: float,
    observation_variance: float,
    initial_variance: float,
    inlier_prior: float,
    outlier_variance_multiplier: float,
) -> Any:
    module = import_module("bayesian_phystwin.phystwin_bayesian_anchor")
    return module.robust_random_walk_endpoint(
        residual,
        valid,
        end_frame=end_frame,
        process_variance=process_variance,
        observation_variance=observation_variance,
        initial_variance=initial_variance,
        inlier_prior=inlier_prior,
        outlier_variance_multiplier=outlier_variance_multiplier,
    )
