"""Causal, trust-gated PGRD residual inference for PhysTwin trajectories."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .phystwin_official_evaluation import evaluate_official_phystwin_interval
from .phystwin_residual_dynamics import (
    _lift_map,
    _lift_residual,
    _load_pickle,
    _sha256,
    _target_validity,
    _temporally_fill,
)


PGRD_UPSTREAM_COMMIT = "e294d96723054f77a1cfdd3c2c052de7b7cd9ce3"
PGRD_SLOTH_CHECKPOINT_SHA256 = (
    "79cc402835b73d6f7dc38a59ea37531f52ea3d2909d434ed9a2a8673509e073c"
)


@dataclass(frozen=True)
class PhysTwinPGRDAdapterConfig:
    """Causal split, transfer calibration, and safety settings."""

    fit_end_frame: int
    train_end_frame: int
    normalized_extent_candidates: tuple[float, ...] = (0.25, 0.5, 0.75)
    yaw_candidates_degrees: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    trust_candidates: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0)
    number_of_points: int = 512
    history_length: int = 2
    temporal_window: int = 5
    simulation_dt: float = 0.1
    model_frame_stride: int = 3
    interpolation_neighbors: int = 4
    maximum_residual_m: float = 0.01
    minimum_dynamic_improvement: float = 0.01
    maximum_metric_ratio: float = 1.02


@dataclass(frozen=True)
class MetricNormalizer:
    """Fixed affine map from metric PhysTwin coordinates to PGRD coordinates."""

    center_m: np.ndarray
    rotation_model_from_metric: np.ndarray
    scale_per_m: float
    normalized_extent: float
    yaw_degrees: float

    @classmethod
    def fit(
        cls,
        points_m: np.ndarray,
        normalized_extent: float,
        *,
        yaw_degrees: float = 0.0,
    ) -> "MetricNormalizer":
        points = np.asarray(points_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("points_m must have shape (N, 3) with N > 0")
        if not np.all(np.isfinite(points)):
            raise ValueError("points_m must be finite")
        if normalized_extent <= 0.0:
            raise ValueError("normalized_extent must be positive")
        if not np.isfinite(yaw_degrees):
            raise ValueError("yaw_degrees must be finite")
        span = float(np.max(np.ptp(points, axis=0), initial=0.0))
        if span <= 1e-8:
            raise ValueError("cannot normalize a degenerate point cloud")
        # PhysTwin gravity is -z; PGRD was trained with gravity on -y.
        gravity_alignment = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
        )
        angle = np.deg2rad(yaw_degrees)
        yaw = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        return cls(
            center_m=np.mean(points, axis=0),
            rotation_model_from_metric=yaw @ gravity_alignment,
            scale_per_m=float(normalized_extent / span),
            normalized_extent=float(normalized_extent),
            yaw_degrees=float(yaw_degrees),
        )

    def positions_to_model(self, points_m: np.ndarray) -> np.ndarray:
        centered = np.asarray(points_m, dtype=float) - self.center_m
        return (
            centered @ self.rotation_model_from_metric.T * self.scale_per_m
        )

    def positions_to_metric(self, points_model: np.ndarray) -> np.ndarray:
        metric = (
            np.asarray(points_model, dtype=float)
            / self.scale_per_m
            @ self.rotation_model_from_metric
        )
        return metric + self.center_m

    def velocities_to_model(self, velocity_m_per_s: np.ndarray) -> np.ndarray:
        return (
            np.asarray(velocity_m_per_s, dtype=float)
            @ self.rotation_model_from_metric.T
            * self.scale_per_m
        )

    def displacements_to_metric(self, displacement_model: np.ndarray) -> np.ndarray:
        return (
            np.asarray(displacement_model, dtype=float)
            / self.scale_per_m
            @ self.rotation_model_from_metric
        )


class PGRDResidualPredictor(Protocol):
    """Minimal stateful interface implemented by the official PGRD adapter."""

    def reset(self) -> None: ...

    def predict(
        self,
        x: np.ndarray,
        v: np.ndarray,
        x_history: np.ndarray,
        v_history: np.ndarray,
        x_sim: np.ndarray,
        v_sim: np.ndarray,
    ) -> np.ndarray: ...


class _AttributeConfig(dict[str, object]):
    """Small OmegaConf-compatible surface used by the released PGRD modules."""

    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def deterministic_farthest_point_sample(points: np.ndarray, count: int) -> np.ndarray:
    """Return a deterministic, geometry-spanning subset without a DGL dependency."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError("points must have shape (N, 3) with N > 0")
    if not 1 <= count <= len(values):
        raise ValueError("count must lie in [1, N]")
    centroid = np.mean(values, axis=0)
    first = int(np.argmax(np.sum(np.square(values - centroid), axis=1)))
    selected = np.empty(count, dtype=np.int64)
    selected[0] = first
    minimum_squared_distance = np.sum(np.square(values - values[first]), axis=1)
    minimum_squared_distance[first] = -1.0
    for position in range(1, count):
        index = int(np.argmax(minimum_squared_distance))
        selected[position] = index
        distance = np.sum(np.square(values - values[index]), axis=1)
        minimum_squared_distance = np.minimum(minimum_squared_distance, distance)
        minimum_squared_distance[selected[: position + 1]] = -1.0
    return selected


def _inverse_distance_map(
    reference: np.ndarray,
    query: np.ndarray,
    neighbors: int,
    *,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= neighbors <= len(reference):
        raise ValueError("neighbors exceeds the reference point count")
    indices = np.empty((len(query), neighbors), dtype=np.int64)
    weights = np.empty((len(query), neighbors), dtype=float)
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        delta = query[start:stop, None] - reference[None]
        squared = np.sum(np.square(delta), axis=2)
        local = np.argpartition(squared, neighbors - 1, axis=1)[:, :neighbors]
        local_squared = np.take_along_axis(squared, local, axis=1)
        order = np.argsort(local_squared, axis=1)
        local = np.take_along_axis(local, order, axis=1)
        local_squared = np.take_along_axis(local_squared, order, axis=1)
        local_weights = 1.0 / np.maximum(local_squared, 1e-16)
        exact = local_squared[:, 0] <= 1e-16
        local_weights /= np.sum(local_weights, axis=1, keepdims=True)
        local_weights[exact] = 0.0
        local_weights[exact, 0] = 1.0
        indices[start:stop] = local
        weights[start:stop] = local_weights
    return indices, weights


def _interpolate_sampled(
    sampled_values: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    return np.sum(sampled_values[:, indices] * weights[None, :, :, None], axis=2)


def compose_dense_endpoint_with_sampled_dynamics(
    sampled_correction: np.ndarray,
    dense_endpoint_correction: np.ndarray,
    sample_indices: np.ndarray,
    interpolation_indices: np.ndarray,
    interpolation_weights: np.ndarray,
) -> np.ndarray:
    """Preserve the dense endpoint exactly and interpolate only future change."""

    endpoint = np.asarray(dense_endpoint_correction, dtype=float)
    sampled_endpoint = endpoint[np.asarray(sample_indices, dtype=np.int64)]
    sampled_change = np.asarray(sampled_correction, dtype=float) - sampled_endpoint[None]
    dense_change = _interpolate_sampled(
        sampled_change, interpolation_indices, interpolation_weights
    )
    return endpoint[None] + dense_change


def _cap_vectors(values: np.ndarray, maximum_norm: float) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    norm = np.linalg.norm(result, axis=-1, keepdims=True)
    result *= np.minimum(1.0, maximum_norm / np.maximum(norm, 1e-12))
    return result


def _metric_ratios(
    metrics: dict[str, object], reference: dict[str, object]
) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for name in ("chamfer_distance_m", "track_error_m"):
        value = float(metrics[name])
        denominator = float(reference[name])
        ratios[name] = (
            value / denominator
            if denominator > 0.0
            else (1.0 if value == 0.0 else float("inf"))
        )
    return ratios


def _relative_score(
    metrics: dict[str, object], reference: dict[str, object]
) -> float:
    ratios = _metric_ratios(metrics, reference)
    return 0.5 * (ratios["chamfer_distance_m"] + ratios["track_error_m"])


def verify_pgrd_assets(
    checkout: str | Path,
    checkpoint: str | Path,
    *,
    expected_commit: str = PGRD_UPSTREAM_COMMIT,
    expected_checkpoint_sha256: str = PGRD_SLOTH_CHECKPOINT_SHA256,
) -> dict[str, str]:
    """Reject silent upstream or checkpoint drift before importing PGRD."""

    checkout_path = Path(checkout).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    if not checkout_path.is_dir():
        raise FileNotFoundError(f"PGRD checkout does not exist: {checkout_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"PGRD checkpoint does not exist: {checkpoint_path}")
    commit = subprocess.run(
        ["git", "-C", str(checkout_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    checkpoint_sha256 = _sha256(checkpoint_path)
    if commit != expected_commit:
        raise ValueError(f"PGRD commit mismatch: expected {expected_commit}, got {commit}")
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            "PGRD checkpoint mismatch: expected "
            f"{expected_checkpoint_sha256}, got {checkpoint_sha256}"
        )
    return {
        "checkout": str(checkout_path),
        "commit": commit,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
    }


def _install_torch_scatter_fallback(torch: object) -> None:
    try:
        __import__("torch_scatter")
        return
    except ImportError:
        pass

    module = types.ModuleType("torch_scatter")

    def segment_csr(src: object, indptr: object, reduce: str = "sum") -> object:
        chunks = []
        for start, stop in zip(indptr[:-1].tolist(), indptr[1:].tolist()):
            segment = src[int(start) : int(stop)]
            if reduce in {"mean", "avg"}:
                chunks.append(segment.mean(dim=0))
            elif reduce in {"max", "amax"}:
                chunks.append(segment.max(dim=0).values)
            elif reduce in {"min", "amin"}:
                chunks.append(segment.min(dim=0).values)
            elif reduce in {"sum", "add"}:
                chunks.append(segment.sum(dim=0))
            else:
                raise ValueError(f"unsupported segment_csr reduction: {reduce}")
        return torch.stack(chunks, dim=0)

    module.segment_csr = segment_csr  # type: ignore[attr-defined]
    sys.modules["torch_scatter"] = module


def _install_omegaconf_type_fallback() -> None:
    """Satisfy PGRD's type-only DictConfig imports in a minimal runtime."""

    try:
        __import__("omegaconf")
        return
    except ImportError:
        pass
    module = types.ModuleType("omegaconf")
    module.DictConfig = _AttributeConfig  # type: ignore[attr-defined]
    sys.modules["omegaconf"] = module


class OfficialPGRDResidualPredictor:
    """Lazy wrapper around a pinned official PGRD checkpoint."""

    def __init__(
        self,
        checkout: str | Path,
        checkpoint: str | Path,
        *,
        device: str = "cuda",
        history_length: int = 2,
        temporal_window: int = 5,
    ) -> None:
        self.provenance = verify_pgrd_assets(checkout, checkpoint)
        checkout_path = Path(checkout).resolve()
        if str(checkout_path) not in sys.path:
            sys.path.insert(0, str(checkout_path))

        import torch

        _install_torch_scatter_fallback(torch)
        _install_omegaconf_type_fallback()
        from meta_material.material.network import ptv3
        from meta_material.material import pbd_adaptive

        def no_flash_init(
            instance: object,
            global_feat: bool = True,
            feature_transform: bool = False,
            feature_dim: int = 1024,
            channel: int = 3,
        ) -> None:
            del feature_transform, feature_dim
            torch.nn.Module.__init__(instance)
            instance.global_feat = global_feat
            instance.point_transformer = ptv3.PointTransformerV3(
                in_channels=channel,
                enable_flash=False,
                upcast_attention=True,
                upcast_softmax=True,
            )

        pbd_adaptive.PTv3Encoder.__init__ = no_flash_init
        from experiments.train.temporal_transformer import TemporalResidualTransformer

        model_cfg = _AttributeConfig(
            cls="PointNetPBDAdaptiveMetaNP2G",
            radius=0.2,
            output_scale=1.0,
            input_scale=2.0,
            absolute_y=False,
            pe_num_func_res=0,
        )
        temporal_cfg = _AttributeConfig(
            model=_AttributeConfig(
                transformer_d_model=64,
                transformer_nhead=4,
                transformer_layers=2,
                transformer_dim_ff=128,
                transformer_window=temporal_window,
            )
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        residualnet = pbd_adaptive.PointNetPBDAdaptiveMetaNP2G(
            model_cfg, history_length
        )
        residualnet.set_params(50, num_grids_flexible=[50, 50, 50, 0.02])
        residualnet.load_state_dict(payload["residualnet"], strict=True)
        residualnet.encoder.point_transformer.shuffle_orders = False
        residualnet.to(device).eval()
        temporal = TemporalResidualTransformer(
            temporal_cfg, device=device, window_override=temporal_window
        )
        temporal.load_component_state_dicts(payload["transformer"])
        temporal.eval()
        self._torch = torch
        self._device = device
        self._residualnet = residualnet
        self._temporal = temporal
        self._history_length = history_length
        self._temporal_window = temporal_window

    def reset(self) -> None:
        self._temporal.reset_window()

    def predict(
        self,
        x: np.ndarray,
        v: np.ndarray,
        x_history: np.ndarray,
        v_history: np.ndarray,
        x_sim: np.ndarray,
        v_sim: np.ndarray,
    ) -> np.ndarray:
        torch = self._torch
        arrays = [x, v, x_history, v_history, x_sim, v_sim]
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("PGRD inputs must be finite")
        if x_history.shape != (len(x), self._history_length, 3):
            raise ValueError("x_history has an incompatible shape")
        tensors = [
            torch.as_tensor(value, dtype=torch.float32, device=self._device)
            for value in arrays
        ]
        x_t, v_t, x_his_t, v_his_t, x_sim_t, v_sim_t = tensors
        enabled = torch.ones((1, len(x), 3), device=self._device)
        with torch.inference_mode():
            features = self._residualnet(
                x_t[None],
                v_t[None],
                x_his_t.reshape(1, len(x), -1),
                v_his_t.reshape(1, len(x), -1),
                enabled,
                x_sim_t[None],
                v_sim_t[None],
            )
            residual_velocity = self._temporal(
                features, rollout_window_size=self._temporal_window
            )
        result = residual_velocity[0].detach().cpu().numpy().astype(float)
        if result.shape != x.shape or not np.all(np.isfinite(result)):
            raise RuntimeError("PGRD returned an invalid residual velocity")
        return result


def rollout_pgrd_correction(
    baseline_m: np.ndarray,
    observed_prefix_m: np.ndarray,
    sample_indices: np.ndarray,
    predictor: PGRDResidualPredictor,
    normalizer: MetricNormalizer,
    *,
    start_frame: int,
    end_frame: int,
    history_length: int,
    temporal_warmup_steps: int,
    simulation_dt: float,
    model_frame_stride: int,
    trust: float,
    maximum_residual_m: float,
) -> np.ndarray:
    """Roll out PGRD recursively, interpolating from persistence to PGRD."""

    baseline = np.asarray(baseline_m, dtype=float)[:, sample_indices]
    prefix = np.asarray(observed_prefix_m, dtype=float)[:, sample_indices]
    if not (
        (history_length - 1) * model_frame_stride
        < start_frame
        < end_frame
        <= len(baseline)
    ):
        raise ValueError("rollout interval leaves insufficient history")
    if prefix.shape[0] != start_frame or prefix.shape[1:] != baseline.shape[1:]:
        raise ValueError("observed_prefix_m must end exactly at start_frame")
    if not 0.0 <= trust <= 1.0:
        raise ValueError("trust must lie in [0, 1]")
    if simulation_dt <= 0.0 or maximum_residual_m <= 0.0:
        raise ValueError("simulation_dt and maximum_residual_m must be positive")
    if model_frame_stride < 1:
        raise ValueError("model_frame_stride must be positive")
    if temporal_warmup_steps < 1:
        raise ValueError("temporal_warmup_steps must be positive")

    persistent_residual = prefix[-1] - baseline[start_frame - 1]
    history_frames = [
        start_frame - 1 - offset * model_frame_stride
        for offset in reversed(range(history_length))
    ]
    states_m = [prefix[frame].copy() for frame in history_frames]
    velocities_model: list[np.ndarray] = [np.zeros_like(baseline[0])]
    for frame in range(1, len(states_m)):
        velocities_model.append(
            normalizer.velocities_to_model(
                (states_m[frame] - states_m[frame - 1]) / simulation_dt
            )
        )
    predictor.reset()
    minimum_target = (history_length + 1) * model_frame_stride
    warm_targets = list(
        range(
            start_frame - 1,
            minimum_target - 1,
            -model_frame_stride,
        )
    )[:temporal_warmup_steps]
    warm_targets.reverse()
    reference_pgrd_residual = np.zeros_like(persistent_residual)
    for target_frame in warm_targets:
        current_frame = target_frame - model_frame_stride
        current_m = prefix[current_frame]
        current_velocity_model = normalizer.velocities_to_model(
            (prefix[current_frame] - prefix[current_frame - model_frame_stride])
            / simulation_dt
        )
        warm_history_frames = [
            current_frame - offset * model_frame_stride
            for offset in reversed(range(history_length))
        ]
        warm_positions = prefix[warm_history_frames]
        warm_position_model = normalizer.positions_to_model(
            np.moveaxis(warm_positions, 0, 1)
        )
        warm_velocity_values = []
        for frame in warm_history_frames:
            warm_velocity_values.append(
                normalizer.velocities_to_model(
                    (prefix[frame] - prefix[frame - model_frame_stride])
                    / simulation_dt
                )
            )
        warm_velocity_model = np.stack(warm_velocity_values, axis=1)
        simulation_model = normalizer.positions_to_model(baseline[target_frame])
        simulation_velocity_model = normalizer.velocities_to_model(
            (baseline[target_frame] - baseline[current_frame]) / simulation_dt
        )
        reference_velocity = predictor.predict(
            normalizer.positions_to_model(current_m),
            current_velocity_model,
            warm_position_model,
            warm_velocity_model,
            simulation_model,
            simulation_velocity_model,
        )
        reference_pgrd_residual = normalizer.displacements_to_metric(
            reference_velocity * simulation_dt
        )
    corrections = np.empty(
        (end_frame - start_frame, len(sample_indices), 3), dtype=float
    )
    previous_frame = start_frame - 1
    previous_correction = persistent_residual.copy()
    while previous_frame < end_frame - 1:
        target_frame = min(previous_frame + model_frame_stride, len(baseline) - 1)
        current_model = normalizer.positions_to_model(states_m[-1])
        simulation_model = normalizer.positions_to_model(baseline[target_frame])
        simulation_velocity_model = normalizer.velocities_to_model(
            (baseline[target_frame] - baseline[previous_frame]) / simulation_dt
        )
        position_history = np.stack(states_m[-history_length:], axis=1)
        position_history_model = normalizer.positions_to_model(position_history)
        velocity_history_model = np.stack(
            velocities_model[-history_length:], axis=1
        )
        residual_velocity_model = predictor.predict(
            current_model,
            velocities_model[-1],
            position_history_model,
            velocity_history_model,
            simulation_model,
            simulation_velocity_model,
        )
        pgrd_residual_m = normalizer.displacements_to_metric(
            residual_velocity_model * simulation_dt
        )
        target_correction = persistent_residual + trust * (
            pgrd_residual_m - reference_pgrd_residual
        )
        target_correction = _cap_vectors(target_correction, maximum_residual_m)
        output_stop = min(target_frame, end_frame - 1)
        denominator = target_frame - previous_frame
        for frame in range(previous_frame + 1, output_stop + 1):
            fraction = (frame - previous_frame) / denominator
            correction_m = (
                (1.0 - fraction) * previous_correction
                + fraction * target_correction
            )
            corrections[frame - start_frame] = correction_m
        current_m = baseline[target_frame] + target_correction
        current_velocity_model = normalizer.velocities_to_model(
            (current_m - states_m[-1]) / simulation_dt
        )
        states_m.append(current_m)
        velocities_model.append(current_velocity_model)
        previous_frame = target_frame
        previous_correction = target_correction
    return corrections


def fit_pgrd_residual_adapter(
    final_data_path: str | Path,
    baseline_trajectory_path: str | Path,
    gt_track_path: str | Path,
    output_dir: str | Path,
    *,
    config: PhysTwinPGRDAdapterConfig,
    pgrd_checkout: str | Path | None = None,
    pgrd_checkpoint: str | Path | None = None,
    device: str = "cuda",
    predictor: PGRDResidualPredictor | None = None,
) -> dict[str, object]:
    """Select PGRD trust on validation frames and open futures only after its gate."""

    if not config.history_length + 2 < config.fit_end_frame < config.train_end_frame:
        raise ValueError("causal split leaves insufficient PGRD history")
    if not config.normalized_extent_candidates or any(
        value <= 0.0 for value in config.normalized_extent_candidates
    ):
        raise ValueError("normalized extents must be positive")
    if not config.yaw_candidates_degrees or any(
        not np.isfinite(value) for value in config.yaw_candidates_degrees
    ):
        raise ValueError("yaw candidates must be finite")
    if not config.trust_candidates or any(
        not 0.0 < value <= 1.0 for value in config.trust_candidates
    ):
        raise ValueError("trust candidates must lie in (0, 1]")
    if config.number_of_points < 1 or config.interpolation_neighbors < 1:
        raise ValueError("point counts must be positive")
    if config.minimum_dynamic_improvement < 0.0:
        raise ValueError("minimum_dynamic_improvement must be nonnegative")
    if config.maximum_metric_ratio < 1.0:
        raise ValueError("maximum_metric_ratio must be at least one")

    external_provenance: dict[str, str] | None = None
    if predictor is None:
        if pgrd_checkout is None or pgrd_checkpoint is None:
            raise ValueError("official inference requires a checkout and checkpoint")
        predictor = OfficialPGRDResidualPredictor(
            pgrd_checkout,
            pgrd_checkpoint,
            device=device,
            history_length=config.history_length,
            temporal_window=config.temporal_window,
        )
        external_provenance = predictor.provenance

    data = _load_pickle(final_data_path)
    baseline = np.asarray(_load_pickle(baseline_trajectory_path), dtype=float)
    gt_track = np.asarray(_load_pickle(gt_track_path), dtype=float)
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
    frame_count, original_count, _ = observed.shape
    if not config.train_end_frame < frame_count:
        raise ValueError("train_end_frame must be below frame count")
    if baseline.shape[0] < frame_count or baseline.shape[1] < original_count:
        raise ValueError("baseline trajectory does not cover observations")
    if config.number_of_points > original_count:
        raise ValueError("number_of_points exceeds tracked object points")
    baseline = baseline[:frame_count]
    valid = _target_validity(visible, motion_valid)
    residual = observed - baseline[:, :original_count]
    sample_indices = deterministic_farthest_point_sample(
        baseline[0, :original_count], config.number_of_points
    )
    sample_lift_indices, sample_lift_weights = _inverse_distance_map(
        baseline[0, sample_indices],
        baseline[0, :original_count],
        min(config.interpolation_neighbors, config.number_of_points),
    )
    lift_indices, lift_weights = _lift_map(
        baseline[0], original_count, config.interpolation_neighbors
    )
    num_surface_points = original_count + len(np.asarray(data["surface_points"]))

    def metrics_for_original(
        original_correction: np.ndarray,
        *,
        start_frame: int,
        end_frame: int,
    ) -> tuple[dict[str, object], np.ndarray]:
        lifted = _lift_residual(
            original_correction,
            baseline.shape[1],
            lift_indices,
            lift_weights,
            maximum_norm=config.maximum_residual_m,
        )
        candidate = baseline.copy()
        candidate[start_frame:end_frame] += lifted
        metrics = evaluate_official_phystwin_interval(
            candidate,
            observed,
            visible,
            gt_track,
            num_surface_points=num_surface_points,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        return metrics, lifted

    validation_baseline = evaluate_official_phystwin_interval(
        baseline,
        observed,
        visible,
        gt_track,
        num_surface_points=num_surface_points,
        start_frame=config.fit_end_frame,
        end_frame=config.train_end_frame,
    )
    fit_filled = _temporally_fill(residual, valid, config.fit_end_frame)
    validation_count = config.train_end_frame - config.fit_end_frame
    persistence_validation_original = np.repeat(
        fit_filled[-1][None], validation_count, axis=0
    )
    persistence_validation, _ = metrics_for_original(
        persistence_validation_original,
        start_frame=config.fit_end_frame,
        end_frame=config.train_end_frame,
    )

    candidates: list[dict[str, object]] = []
    best: tuple[tuple[float, float, float, float], dict[str, object]] | None = None
    observed_fit = baseline[: config.fit_end_frame, :original_count] + fit_filled
    for extent in sorted(set(config.normalized_extent_candidates)):
        for yaw_degrees in sorted(set(config.yaw_candidates_degrees)):
            normalizer = MetricNormalizer.fit(
                baseline[0, sample_indices], extent, yaw_degrees=yaw_degrees
            )
            for trust in sorted(set(config.trust_candidates)):
                sampled = rollout_pgrd_correction(
                    baseline[:, :original_count],
                    observed_fit,
                    sample_indices,
                    predictor,
                    normalizer,
                    start_frame=config.fit_end_frame,
                    end_frame=config.train_end_frame,
                    history_length=config.history_length,
                    temporal_warmup_steps=config.temporal_window,
                    simulation_dt=config.simulation_dt,
                    model_frame_stride=config.model_frame_stride,
                    trust=trust,
                    maximum_residual_m=config.maximum_residual_m,
                )
                original = compose_dense_endpoint_with_sampled_dynamics(
                    sampled,
                    fit_filled[-1],
                    sample_indices,
                    sample_lift_indices,
                    sample_lift_weights,
                )
                metrics, _ = metrics_for_original(
                    original,
                    start_frame=config.fit_end_frame,
                    end_frame=config.train_end_frame,
                )
                ratios = _metric_ratios(metrics, persistence_validation)
                candidate = {
                    "normalized_extent": extent,
                    "yaw_degrees": yaw_degrees,
                    "trust": trust,
                    "selection_score_relative_to_persistence": _relative_score(
                        metrics, persistence_validation
                    ),
                    "metric_ratios_relative_to_persistence": ratios,
                    "official_evaluation": metrics,
                }
                candidates.append(candidate)
                ranking = (
                    float(candidate["selection_score_relative_to_persistence"]),
                    trust,
                    extent,
                    yaw_degrees,
                )
                if best is None or ranking < best[0]:
                    best = (ranking, candidate)
    assert best is not None
    selected = best[1]
    selected_ratios = selected["metric_ratios_relative_to_persistence"]
    dynamic_accepted = (
        float(selected["selection_score_relative_to_persistence"])
        <= 1.0 - config.minimum_dynamic_improvement
        and max(float(value) for value in selected_ratios.values())
        <= config.maximum_metric_ratio
    )

    future_count = frame_count - config.train_end_frame
    train_filled = _temporally_fill(residual, valid, config.train_end_frame)
    if dynamic_accepted:
        extent = float(selected["normalized_extent"])
        yaw_degrees = float(selected["yaw_degrees"])
        trust = float(selected["trust"])
        normalizer = MetricNormalizer.fit(
            baseline[0, sample_indices], extent, yaw_degrees=yaw_degrees
        )
        observed_train = baseline[: config.train_end_frame, :original_count] + train_filled
        sampled_future = rollout_pgrd_correction(
            baseline[:, :original_count],
            observed_train,
            sample_indices,
            predictor,
            normalizer,
            start_frame=config.train_end_frame,
            end_frame=frame_count,
            history_length=config.history_length,
            temporal_warmup_steps=config.temporal_window,
            simulation_dt=config.simulation_dt,
            model_frame_stride=config.model_frame_stride,
            trust=trust,
            maximum_residual_m=config.maximum_residual_m,
        )
        original_future = compose_dense_endpoint_with_sampled_dynamics(
            sampled_future,
            train_filled[-1],
            sample_indices,
            sample_lift_indices,
            sample_lift_weights,
        )
        selected_method = "pgrd"
    else:
        original_future = np.repeat(train_filled[-1][None], future_count, axis=0)
        selected_method = "persistence"

    corrected_test, future_lifted = metrics_for_original(
        original_future,
        start_frame=config.train_end_frame,
        end_frame=frame_count,
    )
    baseline_test = evaluate_official_phystwin_interval(
        baseline,
        observed,
        visible,
        gt_track,
        num_surface_points=num_surface_points,
        start_frame=config.train_end_frame,
        end_frame=frame_count,
    )
    persistence_test, _ = metrics_for_original(
        np.repeat(train_filled[-1][None], future_count, axis=0),
        start_frame=config.train_end_frame,
        end_frame=frame_count,
    )
    corrected = baseline.copy()
    corrected[config.train_end_frame :] += future_lifted

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output / "trajectory.pkl"
    with trajectory_path.open("wb") as handle:
        pickle.dump(corrected.astype(np.float32), handle, protocol=pickle.HIGHEST_PROTOCOL)
    model_path = output / "pgrd_adapter_model.npz"
    np.savez_compressed(
        model_path,
        sample_indices=sample_indices,
        sample_lift_indices=sample_lift_indices,
        sample_lift_weights=sample_lift_weights,
        lift_indices=lift_indices,
        lift_weights=lift_weights,
    )
    correction_norm = np.linalg.norm(future_lifted, axis=2)
    summary: dict[str, object] = {
        "schema_version": 1,
        "config": asdict(config),
        "contract": {
            "selection_interval": [config.fit_end_frame, config.train_end_frame],
            "final_refit_interval": [0, config.train_end_frame],
            "future_observations_used": False,
            "candidate": (
                "prefix-warmed PGRD residual-field change added to endpoint persistence"
            ),
            "fallback": "exact temporally filled endpoint persistence",
            "physical_injection_claim": False,
            "released_case_status": "development-only; cannot confirm transfer",
        },
        "pgrd": external_provenance or {"backend": "injected test predictor"},
        "inputs": {
            name: {
                "path": str(Path(path).resolve()),
                "sha256": _sha256(path),
            }
            for name, path in {
                "final_data": final_data_path,
                "baseline_trajectory": baseline_trajectory_path,
                "gt_track_3d": gt_track_path,
            }.items()
        },
        "selection": {
            "selected_method": selected_method,
            "dynamic_accepted": dynamic_accepted,
            "baseline_official_evaluation": validation_baseline,
            "persistence_official_evaluation": persistence_validation,
            "selected_candidate": selected,
            "candidates": candidates,
        },
        "test": {
            "future_metrics_opened": dynamic_accepted,
            "baseline_official_evaluation": baseline_test,
            "persistence_official_evaluation": persistence_test,
            "corrected_official_evaluation": corrected_test,
            "selection_score_relative_to_persistence": _relative_score(
                corrected_test, persistence_test
            ),
            "metric_ratios_relative_to_persistence": _metric_ratios(
                corrected_test, persistence_test
            ),
        },
        "correction": {
            "rms_m": float(np.sqrt(np.mean(np.square(correction_norm)))),
            "maximum_m": float(np.max(correction_norm, initial=0.0)),
        },
        "outputs": {
            "trajectory": str(trajectory_path.resolve()),
            "model": str(model_path.resolve()),
        },
    }
    summary_path = output / "summary.json"
    summary["outputs"]["summary"] = str(summary_path.resolve())
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
