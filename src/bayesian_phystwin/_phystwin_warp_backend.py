"""Optional Torch/Warp backend for the official PhysTwin refit runner."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import warp as wp


@wp.func
def _smooth_l1_component(residual: float):
    distance = wp.abs(residual)
    result = float(0.0)  # noqa: UP018 - Warp requires a mutable typed scalar.
    if distance < 1.0:
        result = 0.5 * distance * distance
    else:
        result = distance - 0.5
    return result


@wp.kernel
def compute_reliability_weighted_track_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=wp.float32),
    normalizer: wp.array(dtype=wp.float32),
    loss_weight: float,
    track_loss: wp.array(dtype=wp.float32),
):
    track = wp.tid()
    weight = weights[track]
    if weight > 0.0:
        residual = pred[track] - gt[track]
        value = (
            _smooth_l1_component(residual[0])
            + _smooth_l1_component(residual[1])
            + _smooth_l1_component(residual[2])
        )
        denominator = wp.max(normalizer[0] * 3.0, 1.0)
        wp.atomic_add(track_loss, 0, loss_weight * weight * value / denominator)


@wp.kernel
def compute_reliability_mixture_track_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    support: wp.array(dtype=wp.int32),
    prior: wp.array(dtype=wp.float32),
    normalizer: wp.array(dtype=wp.float32),
    observation_variance: float,
    outlier_variance_multiplier: float,
    loss_weight: float,
    track_loss: wp.array(dtype=wp.float32),
):
    track = wp.tid()
    if support[track] == 1:
        residual = pred[track] - gt[track]
        squared_norm = wp.dot(residual, residual)
        inlier_prior = prior[track]
        outlier_prior = 1.0 - inlier_prior
        log_multiplier = wp.log(outlier_variance_multiplier)

        log_inlier = wp.log(inlier_prior) - 0.5 * squared_norm / observation_variance
        log_outlier = (
            wp.log(outlier_prior)
            - 1.5 * log_multiplier
            - 0.5
            * squared_norm
            / (observation_variance * outlier_variance_multiplier)
        )
        maximum = wp.max(log_inlier, log_outlier)
        log_mixture = maximum + wp.log(
            wp.exp(log_inlier - maximum) + wp.exp(log_outlier - maximum)
        )

        zero_log_inlier = wp.log(inlier_prior)
        zero_log_outlier = wp.log(outlier_prior) - 1.5 * log_multiplier
        zero_maximum = wp.max(zero_log_inlier, zero_log_outlier)
        zero_log_mixture = zero_maximum + wp.log(
            wp.exp(zero_log_inlier - zero_maximum)
            + wp.exp(zero_log_outlier - zero_maximum)
        )

        # Scale and zero-shift the NLL to match the local units of smooth L1.
        value = wp.max(
            -observation_variance * (log_mixture - zero_log_mixture),
            0.0,
        )
        denominator = wp.max(normalizer[0] * 3.0, 1.0)
        wp.atomic_add(track_loss, 0, loss_weight * value / denominator)


@wp.kernel
def expand_spring_group_log_y(
    reference_log_y: wp.array(dtype=wp.float32),
    group_log_scales: wp.array(dtype=wp.float32),
    spring_group_ids: wp.array(dtype=wp.int32),
    spring_log_y: wp.array(dtype=wp.float32),
):
    spring = wp.tid()
    group = spring_group_ids[spring]
    spring_log_y[spring] = reference_log_y[spring] + group_log_scales[group]


@wp.kernel
def expand_spring_basis_log_y(
    reference_log_y: wp.array(dtype=wp.float32),
    basis_weights: wp.array(dtype=wp.float32),
    basis_log_coefficients: wp.array(dtype=wp.float32),
    basis_parameter_count: int,
    spring_log_y: wp.array(dtype=wp.float32),
):
    spring = wp.tid()
    offset = spring * basis_parameter_count
    log_scale = float(0.0)  # noqa: UP018 - Warp dynamic-loop accumulator.
    for parameter in range(basis_parameter_count):
        log_scale += (
            basis_weights[offset + parameter]
            * basis_log_coefficients[parameter]
        )
    spring_log_y[spring] = reference_log_y[spring] + log_scale


@wp.kernel
def expand_sparse_spring_basis_log_y(
    reference_log_y: wp.array(dtype=wp.float32),
    basis_parameter_indices: wp.array(dtype=wp.int32),
    basis_weights: wp.array(dtype=wp.float32),
    basis_log_coefficients: wp.array(dtype=wp.float32),
    basis_support_count: int,
    spring_log_y: wp.array(dtype=wp.float32),
):
    spring = wp.tid()
    offset = spring * basis_support_count
    log_scale = float(0.0)  # noqa: UP018 - Warp dynamic-loop accumulator.
    for support in range(basis_support_count):
        parameter = basis_parameter_indices[offset + support]
        log_scale += basis_weights[offset + support] * basis_log_coefficients[parameter]
    spring_log_y[spring] = reference_log_y[spring] + log_scale


@wp.kernel
def eval_springs_deterministic(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    control_x: wp.array(dtype=wp.vec3),
    control_v: wp.array(dtype=wp.vec3),
    num_object_points: int,
    springs: wp.array(dtype=wp.vec2i),
    rest_lengths: wp.array(dtype=wp.float32),
    spring_y: wp.array(dtype=wp.float32),
    vertex_spring_offsets: wp.array(dtype=wp.int32),
    vertex_spring_ids: wp.array(dtype=wp.int32),
    vertex_spring_signs: wp.array(dtype=wp.int32),
    dashpot_damping: float,
    spring_y_min: float,
    spring_y_max: float,
    forces: wp.array(dtype=wp.vec3),
):
    """Sum incident springs in fixed index order without GPU atomics."""

    vertex = wp.tid()
    total = wp.vec3(0.0, 0.0, 0.0)
    start = vertex_spring_offsets[vertex]
    stop = vertex_spring_offsets[vertex + 1]
    for adjacency_index in range(start, stop):
        spring = vertex_spring_ids[adjacency_index]
        if wp.exp(spring_y[spring]) > spring_y_min:
            first = springs[spring][0]
            second = springs[spring][1]
            if first >= num_object_points:
                x1 = control_x[first - num_object_points]
                v1 = control_v[first - num_object_points]
            else:
                x1 = x[first]
                v1 = v[first]
            if second >= num_object_points:
                x2 = control_x[second - num_object_points]
                v2 = control_v[second - num_object_points]
            else:
                x2 = x[second]
                v2 = v[second]
            rest = rest_lengths[spring]
            displacement = x2 - x1
            length = wp.length(displacement)
            direction = displacement / wp.max(length, 1e-6)
            spring_force = (
                wp.clamp(
                    wp.exp(spring_y[spring]),
                    low=spring_y_min,
                    high=spring_y_max,
                )
                * (length / rest - 1.0)
                * direction
            )
            relative_speed = wp.dot(v2 - v1, direction)
            dashpot_force = dashpot_damping * relative_speed * direction
            sign = float(vertex_spring_signs[adjacency_index])
            total += sign * (spring_force + dashpot_force)
    forces[vertex] = total


@wp.kernel
def add_opt_in_external_forces(
    forces: wp.array(dtype=wp.vec3),
    external_forces: wp.array(dtype=wp.vec3),
    enabled: wp.array(dtype=wp.int32),
):
    """Add generalized forces while leaving the disabled path untouched."""

    vertex = wp.tid()
    if enabled[0] == 1:
        forces[vertex] = forces[vertex] + external_forces[vertex]


def deterministic_vertex_spring_adjacency(
    springs: np.ndarray,
    *,
    num_object_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return CSR incident-spring arrays in deterministic spring-index order."""

    edges = np.asarray(springs, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("springs must have shape (S, 2)")
    if num_object_points < 1 or np.any(edges < 0):
        raise ValueError("object point count and spring endpoints must be valid")
    incident: list[list[tuple[int, int]]] = [
        [] for _ in range(num_object_points)
    ]
    for spring, (first, second) in enumerate(edges):
        if first < num_object_points:
            incident[int(first)].append((spring, 1))
        if second < num_object_points:
            incident[int(second)].append((spring, -1))
    offsets = np.zeros(num_object_points + 1, dtype=np.int32)
    for vertex, values in enumerate(incident):
        values.sort(key=lambda value: value[0])
        offsets[vertex + 1] = offsets[vertex] + len(values)
    spring_ids = np.empty(int(offsets[-1]), dtype=np.int32)
    signs = np.empty_like(spring_ids)
    for vertex, values in enumerate(incident):
        start = int(offsets[vertex])
        for local, (spring, sign) in enumerate(values):
            spring_ids[start + local] = spring
            signs[start + local] = sign
    return offsets, spring_ids, signs


def load_official_spring_mass_module(
    official_repo: str | Path,
    *,
    runtime_config: SimpleNamespace,
) -> Any:
    """Load only PhysTwin's low-level simulator, bypassing rendering imports."""

    source = (
        Path(official_repo)
        / "qqtt"
        / "model"
        / "diff_simulator"
        / "spring_mass_warp.py"
    )
    if not source.is_file():
        raise FileNotFoundError(f"official simulator source not found: {source}")

    qqtt_module = types.ModuleType("qqtt")
    qqtt_module.__path__ = []
    utils_module = types.ModuleType("qqtt.utils")
    utils_module.logger = logging.getLogger("bayesian_phystwin.official")
    utils_module.cfg = runtime_config
    qqtt_module.utils = utils_module
    sys.modules["qqtt"] = qqtt_module
    sys.modules["qqtt.utils"] = utils_module

    module_name = "_bayesian_phystwin_official_spring_mass_warp"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load official simulator source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def make_reliability_simulator_class(official_module: Any):
    """Create an official simulator subclass with a normalized track likelihood."""

    class ReliabilitySpringMassSystemWarp(official_module.SpringMassSystemWarp):
        def __init__(
            self,
            *args: Any,
            objective: Any,
            observation_variance: float,
            outlier_variance_multiplier: float,
            spring_parameterization: str,
            num_object_springs: int,
            spring_group_ids: Any | None = None,
            spring_basis_weights: Any | None = None,
            spring_sparse_basis_indices: Any | None = None,
            spring_sparse_basis_weights: Any | None = None,
            spring_sparse_parameter_count: int | None = None,
            deterministic_spring_forces: bool = False,
            **kwargs: Any,
        ) -> None:
            self.loss_variant = objective.variant
            if spring_parameterization not in {
                "dense",
                "grouped",
                "regional",
                "part_pair",
                "canonical_basis",
                "canonical_triplane",
            }:
                raise ValueError(
                    "spring_parameterization must be 'dense', 'grouped', "
                    "'regional', 'part_pair', 'canonical_basis', or "
                    "'canonical_triplane'"
                )
            self.spring_parameterization = spring_parameterization
            self.deterministic_spring_forces = bool(deterministic_spring_forces)
            self.grouped_num_object_springs = int(num_object_springs)
            device = kwargs["gt_object_points"].device
            object_point_count = int(kwargs["num_object_points"])
            self.external_forces_tensor = torch.zeros(
                (object_point_count, 3),
                dtype=torch.float32,
                device=device,
            ).contiguous()
            self.external_force_enabled_tensor = torch.zeros(
                1,
                dtype=torch.int32,
                device=device,
            ).contiguous()
            self.wp_external_forces = wp.from_torch(
                self.external_forces_tensor,
                dtype=wp.vec3,
                requires_grad=False,
            )
            self.wp_external_force_enabled = wp.from_torch(
                self.external_force_enabled_tensor,
                dtype=wp.int32,
                requires_grad=False,
            )
            spring_count = int(args[1].shape[0])
            if not 0 <= self.grouped_num_object_springs <= spring_count:
                raise ValueError("num_object_springs is inconsistent with init_springs")
            if spring_group_ids is None:
                group_ids = torch.zeros(
                    spring_count,
                    dtype=torch.int32,
                    device=device,
                )
                group_ids[self.grouped_num_object_springs :] = 1
            else:
                group_ids = torch.as_tensor(
                    spring_group_ids,
                    dtype=torch.int32,
                    device=device,
                ).reshape(-1)
                if len(group_ids) != spring_count:
                    raise ValueError("spring_group_ids must match the spring count")
                if int(torch.min(group_ids).item()) < 0:
                    raise ValueError("spring_group_ids must be nonnegative")
            self.spring_group_ids_tensor = group_ids.contiguous()
            group_count = int(torch.max(group_ids).item()) + 1
            if self.spring_parameterization == "canonical_basis":
                if spring_basis_weights is None:
                    raise ValueError(
                        "canonical_basis requires spring_basis_weights"
                    )
                basis_weights = torch.as_tensor(
                    spring_basis_weights,
                    dtype=torch.float32,
                    device=device,
                )
                if basis_weights.ndim != 2 or basis_weights.shape[0] != spring_count:
                    raise ValueError(
                        "spring_basis_weights must have shape (S, P)"
                    )
                if basis_weights.shape[1] < 1:
                    raise ValueError("spring_basis_weights must have a parameter")
                if not bool(torch.all(torch.isfinite(basis_weights)).item()):
                    raise ValueError("spring_basis_weights must be finite")
                parameter_count = int(basis_weights.shape[1])
                self.spring_basis_weights_tensor = (
                    basis_weights.contiguous().reshape(-1)
                )
                self.spring_sparse_basis_indices_tensor = torch.zeros(
                    spring_count,
                    dtype=torch.int32,
                    device=device,
                )
                self.spring_sparse_basis_weights_tensor = torch.zeros(
                    spring_count,
                    dtype=torch.float32,
                    device=device,
                )
                self.spring_sparse_basis_support_count = 1
            elif self.spring_parameterization == "canonical_triplane":
                if (
                    spring_sparse_basis_indices is None
                    or spring_sparse_basis_weights is None
                    or spring_sparse_parameter_count is None
                ):
                    raise ValueError(
                        "canonical_triplane requires sparse basis indices, "
                        "weights, and parameter count"
                    )
                sparse_indices = torch.as_tensor(
                    spring_sparse_basis_indices,
                    dtype=torch.int32,
                    device=device,
                )
                sparse_weights = torch.as_tensor(
                    spring_sparse_basis_weights,
                    dtype=torch.float32,
                    device=device,
                )
                if (
                    sparse_indices.ndim != 2
                    or sparse_indices.shape[0] != spring_count
                    or sparse_indices.shape != sparse_weights.shape
                    or sparse_indices.shape[1] < 1
                ):
                    raise ValueError(
                        "sparse spring basis arrays must share shape (S, K)"
                    )
                parameter_count = int(spring_sparse_parameter_count)
                if parameter_count < 1:
                    raise ValueError("sparse spring basis needs parameters")
                if int(torch.min(sparse_indices).item()) < 0 or int(
                    torch.max(sparse_indices).item()
                ) >= parameter_count:
                    raise ValueError("sparse spring basis index is out of range")
                if not bool(torch.all(torch.isfinite(sparse_weights)).item()):
                    raise ValueError("sparse spring basis weights must be finite")
                self.spring_basis_weights_tensor = torch.zeros(
                    spring_count,
                    dtype=torch.float32,
                    device=device,
                )
                self.spring_sparse_basis_indices_tensor = (
                    sparse_indices.contiguous().reshape(-1)
                )
                self.spring_sparse_basis_weights_tensor = (
                    sparse_weights.contiguous().reshape(-1)
                )
                self.spring_sparse_basis_support_count = int(
                    sparse_indices.shape[1]
                )
            else:
                if spring_basis_weights is not None:
                    raise ValueError(
                        "spring_basis_weights require canonical_basis"
                    )
                if (
                    spring_sparse_basis_indices is not None
                    or spring_sparse_basis_weights is not None
                    or spring_sparse_parameter_count is not None
                ):
                    raise ValueError(
                        "sparse spring basis inputs require canonical_triplane"
                    )
                parameter_count = group_count
                self.spring_basis_weights_tensor = torch.zeros(
                    spring_count,
                    dtype=torch.float32,
                    device=device,
                )
                self.spring_sparse_basis_indices_tensor = torch.zeros(
                    spring_count,
                    dtype=torch.int32,
                    device=device,
                )
                self.spring_sparse_basis_weights_tensor = torch.zeros(
                    spring_count,
                    dtype=torch.float32,
                    device=device,
                )
                self.spring_sparse_basis_support_count = 1
            self.spring_basis_parameter_count = parameter_count
            self.reference_spring_log_y_tensor = torch.zeros(
                spring_count,
                dtype=torch.float32,
                device=device,
            )
            self.group_log_scale_tensor = torch.zeros(
                parameter_count,
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            self.wp_reference_spring_log_y = wp.from_torch(
                self.reference_spring_log_y_tensor,
                dtype=wp.float32,
                requires_grad=False,
            )
            self.wp_group_log_scales = wp.from_torch(
                self.group_log_scale_tensor,
                dtype=wp.float32,
                requires_grad=True,
            )
            self.wp_spring_group_ids = wp.from_torch(
                self.spring_group_ids_tensor,
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_spring_basis_weights = wp.from_torch(
                self.spring_basis_weights_tensor,
                dtype=wp.float32,
                requires_grad=False,
            )
            self.wp_spring_sparse_basis_indices = wp.from_torch(
                self.spring_sparse_basis_indices_tensor,
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_spring_sparse_basis_weights = wp.from_torch(
                self.spring_sparse_basis_weights_tensor,
                dtype=wp.float32,
                requires_grad=False,
            )
            if self.deterministic_spring_forces:
                offsets, spring_ids, signs = deterministic_vertex_spring_adjacency(
                    args[1].detach().cpu().numpy(),
                    num_object_points=int(kwargs["num_object_points"]),
                )
                self.vertex_spring_offsets_tensor = torch.as_tensor(
                    offsets, dtype=torch.int32, device=device
                ).contiguous()
                self.vertex_spring_ids_tensor = torch.as_tensor(
                    spring_ids, dtype=torch.int32, device=device
                ).contiguous()
                self.vertex_spring_signs_tensor = torch.as_tensor(
                    signs, dtype=torch.int32, device=device
                ).contiguous()
                self.wp_vertex_spring_offsets = wp.from_torch(
                    self.vertex_spring_offsets_tensor,
                    dtype=wp.int32,
                    requires_grad=False,
                )
                self.wp_vertex_spring_ids = wp.from_torch(
                    self.vertex_spring_ids_tensor,
                    dtype=wp.int32,
                    requires_grad=False,
                )
                self.wp_vertex_spring_signs = wp.from_torch(
                    self.vertex_spring_signs_tensor,
                    dtype=wp.int32,
                    requires_grad=False,
                )
            self.track_prior_frames = torch.as_tensor(
                objective.prior_inlier_probability,
                dtype=torch.float32,
                device=device,
            ).contiguous()
            self.track_support_frames = torch.as_tensor(
                objective.support,
                dtype=torch.int32,
                device=device,
            ).contiguous()
            self.track_weight_frames = torch.as_tensor(
                objective.weights,
                dtype=torch.float32,
                device=device,
            ).contiguous()
            self.track_normalizer_frames = torch.as_tensor(
                objective.normalizer,
                dtype=torch.float32,
                device=device,
            ).contiguous()
            self.wp_current_track_prior = wp.from_torch(
                self.track_prior_frames[1].clone(),
                dtype=wp.float32,
                requires_grad=False,
            )
            self.wp_current_track_support = wp.from_torch(
                self.track_support_frames[1].clone(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_current_track_weight = wp.from_torch(
                self.track_weight_frames[1].clone(),
                dtype=wp.float32,
                requires_grad=False,
            )
            self.wp_current_track_normalizer = wp.from_torch(
                self.track_normalizer_frames[1:2].clone(),
                dtype=wp.float32,
                requires_grad=False,
            )
            self.observation_variance = float(observation_variance)
            self.outlier_variance_multiplier = float(outlier_variance_multiplier)
            super().__init__(*args, **kwargs)

        def step(self):
            if self.spring_parameterization == "canonical_basis":
                wp.launch(
                    expand_spring_basis_log_y,
                    dim=self.n_springs,
                    inputs=[
                        self.wp_reference_spring_log_y,
                        self.wp_spring_basis_weights,
                        self.wp_group_log_scales,
                        self.spring_basis_parameter_count,
                    ],
                    outputs=[self.wp_spring_Y],
                )
            elif self.spring_parameterization == "canonical_triplane":
                wp.launch(
                    expand_sparse_spring_basis_log_y,
                    dim=self.n_springs,
                    inputs=[
                        self.wp_reference_spring_log_y,
                        self.wp_spring_sparse_basis_indices,
                        self.wp_spring_sparse_basis_weights,
                        self.wp_group_log_scales,
                        self.spring_sparse_basis_support_count,
                    ],
                    outputs=[self.wp_spring_Y],
                )
            elif self.spring_parameterization != "dense":
                wp.launch(
                    expand_spring_group_log_y,
                    dim=self.n_springs,
                    inputs=[
                        self.wp_reference_spring_log_y,
                        self.wp_group_log_scales,
                        self.wp_spring_group_ids,
                    ],
                    outputs=[self.wp_spring_Y],
                )
            if not self.deterministic_spring_forces:
                super().step()
                return
            for substep in range(self.num_substeps):
                if self.controller_points is not None:
                    wp.launch(
                        official_module.set_control_points,
                        dim=self.num_control_points,
                        inputs=[
                            self.num_substeps,
                            self.wp_original_control_point,
                            self.wp_target_control_point,
                            substep,
                        ],
                        outputs=[self.wp_states[substep].wp_control_x],
                    )
                wp.launch(
                    eval_springs_deterministic,
                    dim=self.num_object_points,
                    inputs=[
                        self.wp_states[substep].wp_x,
                        self.wp_states[substep].wp_v,
                        self.wp_states[substep].wp_control_x,
                        self.wp_states[substep].wp_control_v,
                        self.num_object_points,
                        self.wp_springs,
                        self.wp_rest_lengths,
                        self.wp_spring_Y,
                        self.wp_vertex_spring_offsets,
                        self.wp_vertex_spring_ids,
                        self.wp_vertex_spring_signs,
                        self.dashpot_damping,
                        self.spring_Y_min,
                        self.spring_Y_max,
                    ],
                    outputs=[self.wp_states[substep].wp_vertice_forces],
                )
                wp.launch(
                    add_opt_in_external_forces,
                    dim=self.num_object_points,
                    inputs=[
                        self.wp_states[substep].wp_vertice_forces,
                        self.wp_external_forces,
                        self.wp_external_force_enabled,
                    ],
                )
                if self.object_collision_flag:
                    output_velocity = self.wp_states[substep].wp_v_before_collision
                else:
                    output_velocity = self.wp_states[substep].wp_v_before_ground
                wp.launch(
                    official_module.update_vel_from_force,
                    dim=self.num_object_points,
                    inputs=[
                        self.wp_states[substep].wp_v,
                        self.wp_states[substep].wp_vertice_forces,
                        self.wp_masses,
                        self.dt,
                        self.drag_damping,
                        self.reverse_factor,
                    ],
                    outputs=[output_velocity],
                )
                if self.object_collision_flag:
                    wp.launch(
                        official_module.object_collision,
                        dim=self.num_object_points,
                        inputs=[
                            self.wp_states[substep].wp_x,
                            self.wp_states[substep].wp_v_before_collision,
                            self.wp_masses,
                            self.wp_masks,
                            self.wp_collide_object_elas,
                            self.wp_collide_object_fric,
                            self.collision_dist,
                            self.wp_collision_indices,
                            self.wp_collision_number,
                        ],
                        outputs=[self.wp_states[substep].wp_v_before_ground],
                    )
                wp.launch(
                    official_module.integrate_ground_collision,
                    dim=self.num_object_points,
                    inputs=[
                        self.wp_states[substep].wp_x,
                        self.wp_states[substep].wp_v_before_ground,
                        self.wp_collide_elas,
                        self.wp_collide_fric,
                        self.dt,
                        self.reverse_factor,
                    ],
                    outputs=[
                        self.wp_states[substep + 1].wp_x,
                        self.wp_states[substep + 1].wp_v,
                    ],
                )

        def set_reference_spring_y(self, spring_log_y: Any):
            if self.spring_parameterization == "dense":
                self.set_spring_Y(spring_log_y)
                return
            wp.launch(
                official_module.copy_float,
                dim=self.n_springs,
                inputs=[spring_log_y],
                outputs=[self.wp_reference_spring_log_y],
            )
            with torch.no_grad():
                self.group_log_scale_tensor.zero_()
            if self.spring_parameterization == "canonical_basis":
                wp.launch(
                    expand_spring_basis_log_y,
                    dim=self.n_springs,
                    inputs=[
                        self.wp_reference_spring_log_y,
                        self.wp_spring_basis_weights,
                        self.wp_group_log_scales,
                        self.spring_basis_parameter_count,
                    ],
                    outputs=[self.wp_spring_Y],
                )
            elif self.spring_parameterization == "canonical_triplane":
                wp.launch(
                    expand_sparse_spring_basis_log_y,
                    dim=self.n_springs,
                    inputs=[
                        self.wp_reference_spring_log_y,
                        self.wp_spring_sparse_basis_indices,
                        self.wp_spring_sparse_basis_weights,
                        self.wp_group_log_scales,
                        self.spring_sparse_basis_support_count,
                    ],
                    outputs=[self.wp_spring_Y],
                )
            else:
                wp.launch(
                    expand_spring_group_log_y,
                    dim=self.n_springs,
                    inputs=[
                        self.wp_reference_spring_log_y,
                        self.wp_group_log_scales,
                        self.wp_spring_group_ids,
                    ],
                    outputs=[self.wp_spring_Y],
                )

        def set_rest_lengths(self, rest_lengths: Any):
            """Replace spring rest lengths without rebuilding captured graphs."""

            if tuple(rest_lengths.shape) != (self.n_springs,):
                raise ValueError("rest_lengths must match the spring count")
            wp.launch(
                official_module.copy_float,
                dim=self.n_springs,
                inputs=[rest_lengths],
                outputs=[self.wp_rest_lengths],
            )

        def set_controller_trajectory(self, controller_points: Any):
            """Replace the recorded controls used by subsequent inference steps."""

            if self.controller_points is None:
                raise ValueError("simulator has no controller trajectory")
            if tuple(controller_points.shape) != tuple(self.controller_points.shape):
                raise ValueError("controller trajectory shape changed")
            self.controller_points = controller_points

        def set_external_forces(self, external_forces: Any):
            """Set a constant per-object-node force for captured inference steps."""

            values = torch.as_tensor(
                external_forces,
                dtype=torch.float32,
                device=self.external_forces_tensor.device,
            ).contiguous()
            if tuple(values.shape) != tuple(self.external_forces_tensor.shape):
                raise ValueError(
                    "external_forces must have shape (num_object_points, 3)"
                )
            if not bool(torch.all(torch.isfinite(values)).item()):
                raise ValueError("external_forces must be finite")
            enabled = bool(torch.any(values != 0.0).item())
            if enabled and not self.deterministic_spring_forces:
                raise ValueError(
                    "external forces require deterministic_spring_forces=True"
                )
            with torch.no_grad():
                self.external_forces_tensor.copy_(values)
                self.external_force_enabled_tensor.fill_(int(enabled))

        def clear_external_forces(self):
            """Disable and zero the opt-in generalized-force input."""

            with torch.no_grad():
                self.external_force_enabled_tensor.zero_()
                self.external_forces_tensor.zero_()

        def set_controller_target(self, frame_idx: int, pure_inference: bool = False):
            super().set_controller_target(frame_idx, pure_inference=pure_inference)
            if pure_inference:
                return
            wp.launch(
                official_module.copy_float,
                dim=self.num_original_points,
                inputs=[self.track_prior_frames[frame_idx]],
                outputs=[self.wp_current_track_prior],
            )
            wp.launch(
                official_module.copy_int,
                dim=self.num_original_points,
                inputs=[self.track_support_frames[frame_idx]],
                outputs=[self.wp_current_track_support],
            )
            wp.launch(
                official_module.copy_float,
                dim=self.num_original_points,
                inputs=[self.track_weight_frames[frame_idx]],
                outputs=[self.wp_current_track_weight],
            )
            wp.launch(
                official_module.copy_float,
                dim=1,
                inputs=[self.track_normalizer_frames[frame_idx : frame_idx + 1]],
                outputs=[self.wp_current_track_normalizer],
            )

        def calculate_loss(self):
            if self.loss_variant.endswith("mixture"):
                wp.launch(
                    compute_reliability_mixture_track_loss,
                    dim=self.num_original_points,
                    inputs=[
                        self.wp_states[-1].wp_x,
                        self.wp_current_object_points,
                        self.wp_current_track_support,
                        self.wp_current_track_prior,
                        self.wp_current_track_normalizer,
                        self.observation_variance,
                        self.outlier_variance_multiplier,
                        official_module.cfg.track_weight,
                    ],
                    outputs=[self.track_loss],
                )
            else:
                wp.launch(
                    compute_reliability_weighted_track_loss,
                    dim=self.num_original_points,
                    inputs=[
                        self.wp_states[-1].wp_x,
                        self.wp_current_object_points,
                        self.wp_current_track_weight,
                        self.wp_current_track_normalizer,
                        official_module.cfg.track_weight,
                    ],
                    outputs=[self.track_loss],
                )

            wp.launch(
                official_module.compute_acc_loss,
                dim=self.num_object_points,
                inputs=[
                    self.wp_states[0].wp_v,
                    self.wp_states[-1].wp_v,
                    self.prev_acc,
                    self.num_object_points,
                    self.acc_count,
                    official_module.cfg.acc_weight,
                ],
                outputs=[self.acc_loss],
            )
            wp.launch(
                official_module.compute_final_loss,
                dim=1,
                inputs=[self.chamfer_loss, self.track_loss, self.acc_loss],
                outputs=[self.loss],
            )

    return ReliabilitySpringMassSystemWarp
