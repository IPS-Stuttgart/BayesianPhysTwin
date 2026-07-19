"""Canonical triplane residual dynamics with prefix-only case adaptation."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .phystwin_shared_nonlinear_residual import (
    _episode_specs,
    _sha256,
    _write_json_with_digest,
    blend_with_persistence,
)
from .phystwin_shared_residual_velocity import (
    SharedResidualVelocityConfig,
    _exogenous_features,
    _interval_metrics,
    _load_episode,
    _ratios,
    _temporally_fill,
    _training_indices,
)


CANONICAL_TRIPLANE_RESIDUAL_CONTRACT = (
    "source-meta-canonical-triplane-residual-v1"
)


@dataclass(frozen=True)
class CanonicalTriplaneResidualConfig:
    """Frozen architecture, optimization, adaptation, and gate settings."""

    grid_resolution: int = 16
    plane_channels: int = 24
    point_hidden_dim: int = 64
    decoder_hidden_dim: int = 96
    latent_dim: int = 16
    rollout_horizon: int = 12
    meta_training_steps: int = 1500
    adaptation_steps: int = 250
    maximum_training_points: int = 192
    learning_rate: float = 3.0e-4
    adaptation_learning_rate: float = 1.0e-2
    weight_decay: float = 1.0e-5
    latent_regularization: float = 1.0e-3
    adaptation_regularization: float = 1.0e-2
    gradient_clip: float = 1.0
    seeds: tuple[int, ...] = (17, 43, 101)
    blend_candidates: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    interpolation_neighbors: int = 4
    controller_kernel_fraction: float = 0.25
    maximum_velocity_multiplier: float = 2.0
    maximum_residual_m: float = 0.01
    minimum_balanced_improvement: float = 0.03
    minimum_both_win_folds: int = 2
    maximum_case_metric_ratio: float = 1.05
    device: str = "cuda:0"


@dataclass
class _CanonicalEpisode:
    loaded: Any
    indices: np.ndarray
    state: np.ndarray
    velocity: np.ndarray
    exogenous: np.ndarray
    coordinates: np.ndarray
    plane_indices: np.ndarray
    plane_weights: np.ndarray
    valid: np.ndarray
    basis: np.ndarray
    residual_cap: float
    velocity_cap: float


def _validate_config(config: CanonicalTriplaneResidualConfig) -> None:
    if config.grid_resolution < 4 or config.plane_channels < 1:
        raise ValueError("triplane resolution and channel count are invalid")
    if min(
        config.point_hidden_dim,
        config.decoder_hidden_dim,
        config.latent_dim,
    ) < 1:
        raise ValueError("network widths and latent dimension must be positive")
    if config.rollout_horizon < 2:
        raise ValueError("rollout_horizon must be at least two")
    if config.meta_training_steps < 1 or config.adaptation_steps < 1:
        raise ValueError("training and adaptation steps must be positive")
    if config.maximum_training_points < 4:
        raise ValueError("maximum_training_points must be at least four")
    if min(config.learning_rate, config.adaptation_learning_rate) <= 0.0:
        raise ValueError("learning rates must be positive")
    if min(config.weight_decay, config.latent_regularization) < 0.0:
        raise ValueError("regularization values must be nonnegative")
    if config.adaptation_regularization < 0.0 or config.gradient_clip <= 0.0:
        raise ValueError("adaptation regularization or gradient clipping is invalid")
    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be nonempty and unique")
    if not config.blend_candidates or any(
        not 0.0 <= value <= 1.0 for value in config.blend_candidates
    ):
        raise ValueError("blend candidates must lie in [0, 1]")
    if config.maximum_velocity_multiplier <= 0.0:
        raise ValueError("maximum_velocity_multiplier must be positive")
    if config.maximum_residual_m <= 0.0:
        raise ValueError("maximum_residual_m must be positive")
    if not 0.0 <= config.minimum_balanced_improvement < 1.0:
        raise ValueError("minimum_balanced_improvement must lie in [0, 1)")
    if config.minimum_both_win_folds < 1:
        raise ValueError("minimum_both_win_folds must be positive")
    if config.maximum_case_metric_ratio < 1.0:
        raise ValueError("maximum_case_metric_ratio must be at least one")


def _canonical_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a deterministic right-handed PCA frame and grid scale."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 4:
        raise ValueError("canonical frame requires at least four 3D points")
    if not np.isfinite(values).all():
        raise ValueError("canonical-frame points must be finite")
    center = np.mean(values, axis=0)
    centered = values - center
    covariance = centered.T @ centered / float(len(centered))
    eigenvalues, basis = np.linalg.eigh(covariance)
    basis = basis[:, np.argsort(eigenvalues)[::-1]]
    projections = centered @ basis
    for axis in range(3):
        pivot = int(np.argmax(np.abs(projections[:, axis])))
        if projections[pivot, axis] < 0.0:
            basis[:, axis] *= -1.0
            projections[:, axis] *= -1.0
    if np.linalg.det(basis) < 0.0:
        basis[:, 2] *= -1.0
        projections[:, 2] *= -1.0
    extent = float(np.max(np.abs(projections)))
    if not np.isfinite(extent) or extent <= 1.0e-8:
        raise ValueError("canonical-frame extent is degenerate")
    return center, basis, extent / 0.95


_VECTOR_FEATURE_STARTS = (0, 3, 6, 9, 12, 15, 19, 22)


def _rotate_exogenous(features: np.ndarray, basis: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=float).copy()
    if values.shape[-1] != 25:
        raise ValueError("expected the 25-dimensional controller feature contract")
    for start in _VECTOR_FEATURE_STARTS:
        values[..., start : start + 3] = values[..., start : start + 3] @ basis
    return values


def _triplane_stencil(
    coordinates: np.ndarray, resolution: int
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute bilinear raster/sample stencils for XY, XZ, and YZ."""

    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")
    if resolution < 2:
        raise ValueError("resolution must be at least two")
    clipped = np.clip(values, -1.0, 1.0)
    scaled = (clipped + 1.0) * (0.5 * (resolution - 1))
    plane_axes = ((0, 1), (0, 2), (1, 2))
    indices = np.empty((3, len(values), 4), dtype=np.int64)
    weights = np.empty((3, len(values), 4), dtype=np.float32)
    for plane, (first_axis, second_axis) in enumerate(plane_axes):
        first = scaled[:, first_axis]
        second = scaled[:, second_axis]
        first_low = np.floor(first).astype(np.int64)
        second_low = np.floor(second).astype(np.int64)
        first_high = np.minimum(first_low + 1, resolution - 1)
        second_high = np.minimum(second_low + 1, resolution - 1)
        first_fraction = first - first_low
        second_fraction = second - second_low
        indices[plane] = np.stack(
            (
                second_low * resolution + first_low,
                second_low * resolution + first_high,
                second_high * resolution + first_low,
                second_high * resolution + first_high,
            ),
            axis=1,
        )
        weights[plane] = np.stack(
            (
                (1.0 - first_fraction) * (1.0 - second_fraction),
                first_fraction * (1.0 - second_fraction),
                (1.0 - first_fraction) * second_fraction,
                first_fraction * second_fraction,
            ),
            axis=1,
        )
    return indices, weights


def _prepare_episode(
    loaded: Any,
    config: CanonicalTriplaneResidualConfig,
    *,
    maximum_points: int | None,
    evidence_end: int,
) -> _CanonicalEpisode:
    original_count = loaded.observed.shape[1]
    if not 2 < evidence_end <= len(loaded.observed):
        raise ValueError("episode evidence boundary is invalid")
    if maximum_points is None:
        indices = np.arange(original_count, dtype=np.int64)
    else:
        indices = _training_indices(
            loaded.valid,
            end_frame=evidence_end,
            maximum_points=maximum_points,
        )
    reference = loaded.baseline[0, :original_count]
    center, basis, grid_scale = _canonical_frame(reference)
    coordinates = ((reference[indices] - center) @ basis) / grid_scale
    filled_prefix = _temporally_fill(loaded.residual, loaded.valid, evidence_end)
    filled = np.empty_like(loaded.residual, dtype=float)
    filled[:evidence_end] = filled_prefix
    filled[evidence_end:] = filled_prefix[-1]
    state = (filled[:, indices] / loaded.object_scale) @ basis
    velocity = np.zeros_like(state)
    velocity[1:evidence_end] = np.diff(state[:evidence_end], axis=0)
    exogenous = np.empty((len(state), len(indices), 25), dtype=np.float32)
    exogenous[0] = 0.0
    for frame in range(1, len(state)):
        exogenous[frame] = _rotate_exogenous(
            _exogenous_features(loaded, frame, indices), basis
        )
    velocity_norm = np.linalg.norm(velocity[1:evidence_end], axis=2)
    velocity_cap = max(
        config.maximum_velocity_multiplier
        * float(np.quantile(velocity_norm, 0.99)),
        1.0e-6,
    )
    plane_indices, plane_weights = _triplane_stencil(
        coordinates, config.grid_resolution
    )
    return _CanonicalEpisode(
        loaded=loaded,
        indices=indices,
        state=state.astype(np.float32),
        velocity=velocity.astype(np.float32),
        exogenous=exogenous,
        coordinates=coordinates.astype(np.float32),
        plane_indices=plane_indices,
        plane_weights=plane_weights,
        valid=loaded.valid[:, indices],
        basis=basis.astype(np.float32),
        residual_cap=float(config.maximum_residual_m / loaded.object_scale),
        velocity_cap=velocity_cap,
    )


def _point_feature_dimension() -> int:
    return 3 + 3 + 25


def _build_model(torch: Any, config: CanonicalTriplaneResidualConfig) -> Any:
    nn = torch.nn

    class CanonicalTriplaneResidual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.resolution = int(config.grid_resolution)
            self.point_encoder = nn.Sequential(
                nn.Linear(_point_feature_dimension(), config.point_hidden_dim),
                nn.SiLU(),
                nn.Linear(config.point_hidden_dim, config.plane_channels),
                nn.SiLU(),
            )
            self.plane_tower = nn.Sequential(
                nn.Conv2d(
                    config.plane_channels,
                    config.plane_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.SiLU(),
                nn.Conv2d(
                    config.plane_channels,
                    config.plane_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.SiLU(),
            )
            decoder_input = (
                config.plane_channels * 4
                + config.latent_dim
                + _point_feature_dimension()
            )
            self.decoder = nn.Sequential(
                nn.Linear(decoder_input, config.decoder_hidden_dim),
                nn.SiLU(),
                nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim),
                nn.SiLU(),
                nn.Linear(config.decoder_hidden_dim, 3),
            )
            nn.init.zeros_(self.decoder[-1].weight)
            nn.init.zeros_(self.decoder[-1].bias)

        def _rasterize(
            self, point_features: Any, indices: Any, weights: Any
        ) -> Any:
            channels = point_features.shape[1]
            flattened = point_features.new_zeros(
                (channels, self.resolution * self.resolution)
            )
            normalizer = point_features.new_zeros(
                (1, self.resolution * self.resolution)
            )
            for corner in range(4):
                corner_indices = indices[:, corner]
                corner_weights = weights[:, corner]
                flattened.scatter_add_(
                    1,
                    corner_indices.unsqueeze(0).expand(channels, -1),
                    (point_features * corner_weights.unsqueeze(1)).T,
                )
                normalizer.scatter_add_(
                    1,
                    corner_indices.unsqueeze(0),
                    corner_weights.unsqueeze(0),
                )
            flattened = flattened / torch.clamp(normalizer, min=1.0e-6)
            return flattened.reshape(
                1, channels, self.resolution, self.resolution
            )

        @staticmethod
        def _sample(grid: Any, indices: Any, weights: Any) -> Any:
            flattened = grid.reshape(grid.shape[1], -1)
            sampled = grid.new_zeros((indices.shape[0], grid.shape[1]))
            for corner in range(4):
                sampled = sampled + (
                    flattened[:, indices[:, corner]].T
                    * weights[:, corner].unsqueeze(1)
                )
            return sampled

        def forward(
            self,
            state: Any,
            velocity: Any,
            exogenous: Any,
            plane_indices: Any,
            plane_weights: Any,
            latent: Any,
        ) -> Any:
            raw = torch.cat((state, velocity, exogenous), dim=1)
            point_features = self.point_encoder(raw)
            sampled_planes = []
            for plane in range(3):
                grid = self._rasterize(
                    point_features,
                    plane_indices[plane],
                    plane_weights[plane],
                )
                grid = self.plane_tower(grid)
                sampled_planes.append(
                    self._sample(
                        grid,
                        plane_indices[plane],
                        plane_weights[plane],
                    )
                )
            repeated_latent = latent.reshape(1, -1).expand(len(state), -1)
            return self.decoder(
                torch.cat(
                    (raw, point_features, *sampled_planes, repeated_latent), dim=1
                )
            )

    return CanonicalTriplaneResidual()


def _tensor_episode(torch: Any, episode: _CanonicalEpisode, device: Any) -> dict[str, Any]:
    return {
        "state": torch.as_tensor(episode.state, device=device),
        "velocity": torch.as_tensor(episode.velocity, device=device),
        "exogenous": torch.as_tensor(episode.exogenous, device=device),
        "valid": torch.as_tensor(episode.valid, dtype=torch.float32, device=device),
        "plane_indices": torch.as_tensor(
            episode.plane_indices, dtype=torch.long, device=device
        ),
        "plane_weights": torch.as_tensor(episode.plane_weights, device=device),
    }


def _step_state(
    torch: Any,
    model: Any,
    state: Any,
    velocity: Any,
    exogenous: Any,
    plane_indices: Any,
    plane_weights: Any,
    latent: Any,
    *,
    velocity_cap: float,
    residual_cap: float,
) -> tuple[Any, Any]:
    next_velocity = float(velocity_cap) * torch.tanh(
        model(
            state,
            velocity,
            exogenous,
            plane_indices,
            plane_weights,
            latent,
        )
    )
    next_state = state + next_velocity
    norm = torch.linalg.vector_norm(next_state, dim=1, keepdim=True)
    scale = torch.clamp(
        float(residual_cap) / torch.clamp(norm, min=1.0e-12), max=1.0
    )
    return next_state * scale, next_velocity


def _recursive_loss(
    torch: Any,
    model: Any,
    tensors: Mapping[str, Any],
    latent: Any,
    episode: _CanonicalEpisode,
    *,
    start: int,
    horizon: int,
) -> Any:
    state = tensors["state"][start]
    velocity = tensors["velocity"][start]
    loss = torch.zeros((), device=state.device)
    weight_sum = torch.zeros((), device=state.device)
    for offset in range(1, horizon + 1):
        frame = start + offset
        state, velocity = _step_state(
            torch,
            model,
            state,
            velocity,
            tensors["exogenous"][frame],
            tensors["plane_indices"],
            tensors["plane_weights"],
            latent,
            velocity_cap=episode.velocity_cap,
            residual_cap=episode.residual_cap,
        )
        point_loss = torch.nn.functional.smooth_l1_loss(
            state,
            tensors["state"][frame],
            reduction="none",
            beta=0.05,
        ).mean(dim=1)
        horizon_weight = float(offset) / float(horizon)
        loss = loss + horizon_weight * torch.sum(
            tensors["valid"][frame] * point_loss
        )
        weight_sum = weight_sum + horizon_weight * torch.sum(
            tensors["valid"][frame]
        )
    return loss / torch.clamp(weight_sum, min=1.0)


def _train_meta_model(
    prepared: Sequence[_CanonicalEpisode],
    config: CanonicalTriplaneResidualConfig,
    *,
    seed: int,
    output_path: Path,
) -> tuple[Any, dict[str, object]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("canonical triplane training requires PyTorch") from exc
    if not prepared:
        raise ValueError("meta training requires source episodes")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    rng = np.random.default_rng(int(seed))
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = _build_model(torch, config).to(device)
    case_latents = torch.nn.ParameterDict(
        {
            episode.loaded.spec.case: torch.nn.Parameter(
                torch.zeros(config.latent_dim, device=device)
            )
            for episode in prepared
        }
    )
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(case_latents.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    tensor_cache = {
        episode.loaded.spec.case: _tensor_episode(torch, episode, device)
        for episode in prepared
    }
    losses: list[float] = []
    model.train()
    for step in range(config.meta_training_steps):
        episode = prepared[step % len(prepared)]
        case = episode.loaded.spec.case
        horizon = min(config.rollout_horizon, len(episode.state) - 2)
        maximum_start = len(episode.state) - horizon - 1
        start = int(rng.integers(1, maximum_start + 1))
        optimizer.zero_grad(set_to_none=True)
        loss = _recursive_loss(
            torch,
            model,
            tensor_cache[case],
            case_latents[case],
            episode,
            start=start,
            horizon=horizon,
        )
        loss = loss + config.latent_regularization * torch.mean(
            torch.square(case_latents[case])
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("meta training produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(case_latents.parameters()),
            config.gradient_clip,
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "contract": CANONICAL_TRIPLANE_RESIDUAL_CONTRACT,
            "seed": int(seed),
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "source_case_latents": {
                case: latent.detach().cpu()
                for case, latent in case_latents.items()
            },
        },
        output_path,
    )
    return model, {
        "seed": int(seed),
        "checkpoint": {"path": str(output_path.resolve()), "sha256": _sha256(output_path)},
        "initial_loss": losses[0],
        "terminal_loss": losses[-1],
        "minimum_loss": min(losses),
        "step_count": len(losses),
    }


def _adapt_case_latent(
    model: Any,
    episode: _CanonicalEpisode,
    config: CanonicalTriplaneResidualConfig,
    *,
    seed: int,
    evidence_end: int,
) -> tuple[Any, dict[str, float]]:
    import torch

    if evidence_end <= config.rollout_horizon + 2:
        raise ValueError("adaptation prefix is too short")
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    local_model = copy.deepcopy(model).to(device)
    local_model.eval()
    for parameter in local_model.parameters():
        parameter.requires_grad_(False)
    latent = torch.nn.Parameter(torch.zeros(config.latent_dim, device=device))
    optimizer = torch.optim.Adam([latent], lr=config.adaptation_learning_rate)
    tensors = _tensor_episode(torch, episode, device)
    rng = np.random.default_rng(int(seed) + 1_000_003)
    losses: list[float] = []
    horizon = min(config.rollout_horizon, evidence_end - 2)
    maximum_start = evidence_end - horizon - 1
    for _ in range(config.adaptation_steps):
        start = int(rng.integers(1, maximum_start + 1))
        optimizer.zero_grad(set_to_none=True)
        loss = _recursive_loss(
            torch,
            local_model,
            tensors,
            latent,
            episode,
            start=start,
            horizon=horizon,
        )
        loss = loss + config.adaptation_regularization * torch.mean(
            torch.square(latent)
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("case adaptation produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([latent], config.gradient_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return latent.detach(), {
        "initial_loss": losses[0],
        "terminal_loss": losses[-1],
        "minimum_loss": min(losses),
        "latent_norm": float(torch.linalg.vector_norm(latent).detach().cpu()),
    }


def _rollout_model(
    model: Any,
    latent: Any,
    episode: _CanonicalEpisode,
    *,
    start_frame: int,
    end_frame: int,
    device_name: str,
) -> np.ndarray:
    import torch

    if not 1 <= start_frame < end_frame <= len(episode.state):
        raise ValueError("rollout interval is invalid")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    tensors = _tensor_episode(torch, episode, device)
    state = tensors["state"][start_frame - 1]
    velocity = tensors["velocity"][start_frame - 1]
    latent = latent.to(device)
    result = []
    with torch.no_grad():
        for frame in range(start_frame, end_frame):
            state, velocity = _step_state(
                torch,
                model,
                state,
                velocity,
                tensors["exogenous"][frame],
                tensors["plane_indices"],
                tensors["plane_weights"],
                latent,
                velocity_cap=episode.velocity_cap,
                residual_cap=episode.residual_cap,
            )
            result.append(state.detach().cpu().numpy())
    canonical = np.asarray(result, dtype=np.float32)
    return (canonical @ episode.basis.T) * episode.loaded.object_scale


def _load_protocol(
    path: str | Path,
) -> tuple[dict[str, object], CanonicalTriplaneResidualConfig]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("contract") != CANONICAL_TRIPLANE_RESIDUAL_CONTRACT:
        raise ValueError("source protocol uses an unsupported contract")
    source_cases = payload.get("source_cases")
    target_cases = payload.get("target_cases")
    folds = payload.get("source_folds")
    if not isinstance(source_cases, list) or len(source_cases) < 3:
        raise ValueError("source protocol requires at least three cases")
    if len(set(source_cases)) != len(source_cases):
        raise ValueError("source cases must be unique")
    if not isinstance(target_cases, list) or set(source_cases) & set(target_cases):
        raise ValueError("source and target case names must be disjoint")
    if not isinstance(folds, list) or len(folds) < 2:
        raise ValueError("source protocol requires at least two folds")
    held_out: list[str] = []
    for fold in folds:
        if not isinstance(fold, Mapping) or not isinstance(
            fold.get("held_out_cases"), list
        ):
            raise ValueError("every source fold must declare held_out_cases")
        held_out.extend(str(case) for case in fold["held_out_cases"])
    if sorted(held_out) != sorted(str(case) for case in source_cases):
        raise ValueError("source folds must hold out every source case exactly once")
    raw_model = payload.get("model")
    if not isinstance(raw_model, Mapping):
        raise ValueError("source protocol omits its model configuration")
    config = CanonicalTriplaneResidualConfig(
        **{
            key: tuple(value) if key in {"seeds", "blend_candidates"} else value
            for key, value in raw_model.items()
        }
    )
    _validate_config(config)
    return payload, config


def fit_canonical_triplane_residual_source_gate(
    data_root: str | Path,
    protocol_path: str | Path,
    output_dir: str | Path,
    *,
    device: str | None = None,
) -> dict[str, object]:
    """Cross-fit, prefix-adapt, and gate without reading target trajectories."""

    protocol, config = _load_protocol(protocol_path)
    if device is not None:
        config = CanonicalTriplaneResidualConfig(
            **{**asdict(config), "device": device}
        )
    root = Path(data_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fit_fraction = float(protocol.get("fit_fraction", 0.75))
    if not 0.5 <= fit_fraction < 1.0:
        raise ValueError("fit_fraction must lie in [0.5, 1)")
    source_cases = tuple(str(case) for case in protocol["source_cases"])
    specs = _episode_specs(root, source_cases, fit_fraction)
    metric_config = SharedResidualVelocityConfig(
        maximum_training_points=config.maximum_training_points,
        interpolation_neighbors=config.interpolation_neighbors,
        controller_kernel_fraction=config.controller_kernel_fraction,
        maximum_velocity_multiplier=config.maximum_velocity_multiplier,
        maximum_residual_m=config.maximum_residual_m,
    )
    loaded = {spec.case: _load_episode(spec, metric_config) for spec in specs}
    complete_training = {
        case: _prepare_episode(
            episode,
            config,
            maximum_points=config.maximum_training_points,
            evidence_end=len(episode.observed),
        )
        for case, episode in loaded.items()
    }
    heldout_adaptation = {
        case: _prepare_episode(
            episode,
            config,
            maximum_points=config.maximum_training_points,
            evidence_end=episode.spec.fit_end_frame,
        )
        for case, episode in loaded.items()
    }
    heldout_evaluation = {
        case: _prepare_episode(
            episode,
            config,
            maximum_points=None,
            evidence_end=episode.spec.fit_end_frame,
        )
        for case, episode in loaded.items()
    }

    fold_predictions: dict[str, dict[str, np.ndarray]] = {}
    fold_records: list[dict[str, object]] = []
    for fold_index, raw_fold in enumerate(protocol["source_folds"]):
        held_out = tuple(str(case) for case in raw_fold["held_out_cases"])
        training_cases = tuple(case for case in source_cases if case not in held_out)
        seed_models = []
        seed_summaries = []
        for seed in config.seeds:
            checkpoint = output / f"fold_{fold_index:02d}" / f"model_seed_{seed}.pt"
            model, training_summary = _train_meta_model(
                [complete_training[case] for case in training_cases],
                config,
                seed=seed,
                output_path=checkpoint,
            )
            seed_models.append((int(seed), model))
            seed_summaries.append(training_summary)
        case_predictions: dict[str, np.ndarray] = {}
        case_adaptation: dict[str, object] = {}
        for case in held_out:
            members = []
            adaptations = []
            for seed, model in seed_models:
                latent, adaptation = _adapt_case_latent(
                    model,
                    heldout_adaptation[case],
                    config,
                    seed=seed,
                    evidence_end=loaded[case].spec.fit_end_frame,
                )
                members.append(
                    _rollout_model(
                        model,
                        latent,
                        heldout_evaluation[case],
                        start_frame=loaded[case].spec.fit_end_frame,
                        end_frame=loaded[case].spec.train_end_frame,
                        device_name=config.device,
                    )
                )
                adaptations.append({"seed": seed, **adaptation})
            case_predictions[case] = np.mean(members, axis=0)
            case_adaptation[case] = adaptations
        fold_predictions[str(fold_index)] = case_predictions
        fold_records.append(
            {
                "name": str(raw_fold.get("name", f"fold_{fold_index}")),
                "held_out_cases": list(held_out),
                "training_cases": list(training_cases),
                "seed_models": seed_summaries,
                "heldout_prefix_adaptation": case_adaptation,
            }
        )

    candidates = []
    for blend in config.blend_candidates:
        case_results = []
        fold_results = []
        for fold_index, raw_fold in enumerate(protocol["source_folds"]):
            fold_case_results = []
            for raw_case in raw_fold["held_out_cases"]:
                case = str(raw_case)
                episode = loaded[case]
                prepared = heldout_evaluation[case]
                endpoint = (
                    prepared.state[episode.spec.fit_end_frame - 1]
                    @ prepared.basis.T
                ) * episode.object_scale
                dynamic = blend_with_persistence(
                    fold_predictions[str(fold_index)][case], endpoint, float(blend)
                )
                dynamic_metrics, _ = _interval_metrics(
                    episode,
                    dynamic,
                    start_frame=episode.spec.fit_end_frame,
                    end_frame=episode.spec.train_end_frame,
                    config=metric_config,
                )
                persistence = np.broadcast_to(endpoint, dynamic.shape)
                persistence_metrics, _ = _interval_metrics(
                    episode,
                    persistence,
                    start_frame=episode.spec.fit_end_frame,
                    end_frame=episode.spec.train_end_frame,
                    config=metric_config,
                )
                record = {
                    "case": case,
                    "ratios_relative_to_persistence": _ratios(
                        dynamic_metrics, persistence_metrics
                    ),
                    "persistence_official_evaluation": persistence_metrics,
                    "dynamic_official_evaluation": dynamic_metrics,
                }
                case_results.append(record)
                fold_case_results.append(record)
            fold_ratios = {
                metric: float(
                    np.mean(
                        [
                            item["ratios_relative_to_persistence"][metric]
                            for item in fold_case_results
                        ]
                    )
                )
                for metric in ("chamfer_distance_m", "track_error_m")
            }
            fold_results.append(
                {
                    "name": str(raw_fold.get("name", f"fold_{fold_index}")),
                    "aggregate_ratios_relative_to_persistence": fold_ratios,
                    "both_metrics_improve": max(fold_ratios.values()) < 1.0,
                }
            )
        aggregate_ratios = {
            metric: float(
                np.mean(
                    [
                        item["ratios_relative_to_persistence"][metric]
                        for item in case_results
                    ]
                )
            )
            for metric in ("chamfer_distance_m", "track_error_m")
        }
        candidates.append(
            {
                "blend": float(blend),
                "balanced_improvement": 1.0
                - 0.5 * sum(aggregate_ratios.values()),
                "aggregate_ratios_relative_to_persistence": aggregate_ratios,
                "both_win_fold_count": int(
                    sum(bool(item["both_metrics_improve"]) for item in fold_results)
                ),
                "maximum_case_metric_ratio": max(
                    max(item["ratios_relative_to_persistence"].values())
                    for item in case_results
                ),
                "fold_results": fold_results,
                "case_results": case_results,
            }
        )
    selected = min(
        candidates,
        key=lambda item: (-float(item["balanced_improvement"]), float(item["blend"])),
    )
    gate_passed = (
        float(selected["balanced_improvement"])
        >= config.minimum_balanced_improvement
        and int(selected["both_win_fold_count"]) >= config.minimum_both_win_folds
        and max(selected["aggregate_ratios_relative_to_persistence"].values()) < 1.0
        and float(selected["maximum_case_metric_ratio"])
        <= config.maximum_case_metric_ratio
    )
    summary = {
        "schema_version": 1,
        "contract": CANONICAL_TRIPLANE_RESIDUAL_CONTRACT,
        "source_gate_passed": gate_passed,
        "target_future_opened": False,
        "protocol": {"path": str(Path(protocol_path).resolve()), "sha256": _sha256(protocol_path)},
        "config": asdict(config),
        "source_inputs": [
            {
                "case": spec.case,
                "fit_end_frame": spec.fit_end_frame,
                "train_end_frame": spec.train_end_frame,
                "final_data_sha256": _sha256(spec.final_data),
                "baseline_trajectory_sha256": _sha256(spec.baseline_trajectory),
                "gt_track_3d_sha256": _sha256(spec.gt_track_3d),
            }
            for spec in specs
        ],
        "fold_training": fold_records,
        "selection": {"selected_candidate": selected, "candidates": candidates},
        "information_boundary": (
            "Complete outcomes supervise only non-held-out source interactions. "
            "Each held-out source model adapts only a case latent on [0, fit_end), "
            "and is scored on [fit_end, train_end). No target artifact is read."
        ),
    }
    summary_path = output / "source_gate_summary.json"
    digest = _write_json_with_digest(summary_path, summary)
    return {
        **summary,
        "summary_artifact": {"path": str(summary_path), "sha256": digest},
    }
