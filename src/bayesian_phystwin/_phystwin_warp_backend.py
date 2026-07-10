"""Optional Torch/Warp backend for the official PhysTwin refit runner."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import warp as wp


@wp.func
def _smooth_l1_component(residual: float):
    distance = wp.abs(residual)
    result = float(0.0)
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
            -2.0 * observation_variance * (log_mixture - zero_log_mixture),
            0.0,
        )
        denominator = wp.max(normalizer[0] * 3.0, 1.0)
        wp.atomic_add(track_loss, 0, loss_weight * value / denominator)


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
            **kwargs: Any,
        ) -> None:
            self.loss_variant = objective.variant
            device = kwargs["gt_object_points"].device
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
            if self.loss_variant == "mixture":
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
