#!/usr/bin/env python3
"""Competitive two-stream TCN baseline for Deform360 tactile forecasting.

The baseline is intentionally separate from the frozen v3 Bayesian action ensemble.
A standard temporal convolutional network consumes a tactile-history sequence and,
for the action-conditioned arm, the known future robot trajectory. Architecture and
training-budget selection use the original 14-object development roster. The final
models are then trained from source episodes of the 92-object confirmation roster,
content-hashed, and only afterwards are target tactile outcomes opened for scoring.

This is a post-confirmation competitive-baseline audit on released measurements. It
is not a new untouched confirmation and does not alter the frozen v3 method.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCHEMA = "bayesian-phystwin/deform360-tcn-baseline-audit-v6"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
v3 = load_module(
    HERE / "run_deform360_action_kernel_v3.py",
    "deform360_action_kernel_v3_v6",
)
base = v3.base


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise base.EvaluationError(f"expected JSON object: {path}")
    return value


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def exact_one_sided_sign_pvalue(wins: int, losses: int) -> float:
    count = wins + losses
    if count == 0:
        return 1.0
    return float(sum(math.comb(count, k) for k in range(wins, count + 1)) / 2**count)


def bootstrap_interval(
    values: np.ndarray,
    repetitions: int,
    seed: int,
) -> list[float]:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("bootstrap values must be a nonempty vector")
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass(frozen=True)
class TemporalSamples:
    tactile: np.ndarray
    action: np.ndarray
    static: np.ndarray
    target: np.ndarray
    current: np.ndarray
    truth: np.ndarray
    active: np.ndarray

    def __len__(self) -> int:
        return int(self.tactile.shape[0])


@dataclass(frozen=True)
class ObjectSource:
    object_id: str
    descriptors: tuple[Any, ...]
    source_descriptors: tuple[Any, ...]
    target_descriptor: Any
    source: tuple[Any, ...]
    feature_scale: np.ndarray
    source_samples: TemporalSamples


@dataclass(frozen=True)
class Scaler:
    tactile_mean: np.ndarray
    tactile_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "tactile_mean": self.tactile_mean.tolist(),
            "tactile_std": self.tactile_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
        }


def concatenate_samples(samples: Sequence[TemporalSamples]) -> TemporalSamples:
    if not samples:
        raise base.EvaluationError("cannot concatenate an empty sample collection")
    feature_dimensions = {item.tactile.shape[-1] for item in samples}
    target_dimensions = {item.target.shape[-1] for item in samples}
    if len(feature_dimensions) != 1 or len(target_dimensions) != 1:
        raise base.EvaluationError("TCN carrier dimensions differ between objects")
    return TemporalSamples(
        tactile=np.concatenate([item.tactile for item in samples]),
        action=np.concatenate([item.action for item in samples]),
        static=np.concatenate([item.static for item in samples]),
        target=np.concatenate([item.target for item in samples]),
        current=np.concatenate([item.current for item in samples]),
        truth=np.concatenate([item.truth for item in samples]),
        active=np.concatenate([item.active for item in samples]),
    )


def action_sequence(
    robot: np.ndarray,
    starts: np.ndarray,
    horizon: int,
    sample_count: int,
) -> np.ndarray:
    offsets = np.rint(np.linspace(0, horizon, sample_count)).astype(int)
    rows: list[np.ndarray] = []
    for start in starts:
        segment = robot[start + offsets]
        current = robot[start]
        relative = segment - current[None, ...]
        increments = np.diff(segment, axis=0, prepend=segment[:1])
        rows.append(
            np.concatenate(
                (
                    relative.reshape(sample_count, -1),
                    increments.reshape(sample_count, -1),
                ),
                axis=1,
            )
        )
    return np.asarray(rows, dtype=np.float64)


def temporal_samples(
    episode: Any,
    feature_scale: np.ndarray,
    base_protocol: Mapping[str, Any],
    horizon: int,
    history_frames: int,
    action_samples: int,
) -> TemporalSamples:
    model = base_protocol["model"]
    clip = float(model["normalized_feature_clip"])
    lag = int(model["trend_lag_frames"])
    stride = int(model["window_stride_frames"])
    threshold = float(model["active_threshold"])
    values = base.normalize_tactile(episode.tactile, feature_scale, clip)
    first = max(history_frames - 1, lag)
    starts = np.arange(first, len(values) - horizon, stride, dtype=int)
    if len(starts) == 0:
        raise base.EvaluationError(
            f"episode {episode.descriptor.episode_id} has no temporal windows"
        )
    history_offsets = np.arange(history_frames - 1, -1, -1, dtype=int)
    histories = values[starts[:, None] - history_offsets[None, :]]
    current = values[starts]
    truth = values[starts + horizon]
    target = truth - current
    action = action_sequence(
        episode.robot_actions,
        starts,
        horizon,
        action_samples,
    )
    static_row = np.concatenate(
        (
            base.action_one_hot(episode.descriptor.action),
            np.asarray([float(episode.bimanual)], dtype=np.float64),
        )
    )
    static = np.repeat(static_row[None, :], len(starts), axis=0)
    active = (truth > threshold) | (current > threshold)
    return TemporalSamples(
        tactile=np.asarray(histories, dtype=np.float32),
        action=np.asarray(action, dtype=np.float32),
        static=np.asarray(static, dtype=np.float32),
        target=np.asarray(target, dtype=np.float32),
        current=np.asarray(current, dtype=np.float64),
        truth=np.asarray(truth, dtype=np.float64),
        active=np.asarray(active, dtype=bool),
    )


def discover_source_object(
    root: Path,
    object_id: str,
    minimum_episodes: int,
    base_protocol: Mapping[str, Any],
    horizon: int,
    history_frames: int,
    action_samples: int,
) -> ObjectSource:
    descriptors = base.discover_object(root, object_id, minimum_episodes)
    if len(descriptors) < minimum_episodes:
        raise base.EvaluationError(f"object {object_id} lacks required carriers")
    target_descriptor = max(descriptors, key=lambda item: item.episode_id)
    source_descriptors = tuple(
        descriptor for descriptor in descriptors if descriptor is not target_descriptor
    )
    source = tuple(base.load_episode(descriptor) for descriptor in source_descriptors)
    model = base_protocol["model"]
    feature_scale = base.feature_scale(
        source,
        float(model["source_feature_scale_quantile"]),
    )
    source_samples = concatenate_samples(
        [
            temporal_samples(
                episode,
                feature_scale,
                base_protocol,
                horizon,
                history_frames,
                action_samples,
            )
            for episode in source
        ]
    )
    return ObjectSource(
        object_id=object_id,
        descriptors=tuple(descriptors),
        source_descriptors=source_descriptors,
        target_descriptor=target_descriptor,
        source=source,
        feature_scale=feature_scale,
        source_samples=source_samples,
    )


def load_target_samples(
    source_object: ObjectSource,
    base_protocol: Mapping[str, Any],
    horizon: int,
    history_frames: int,
    action_samples: int,
) -> TemporalSamples:
    target = base.load_episode(source_object.target_descriptor)
    return temporal_samples(
        target,
        source_object.feature_scale,
        base_protocol,
        horizon,
        history_frames,
        action_samples,
    )


def fit_scaler(samples: TemporalSamples) -> Scaler:
    tactile_flat = samples.tactile.reshape(-1, samples.tactile.shape[-1]).astype(
        np.float64
    )
    action_flat = samples.action.reshape(-1, samples.action.shape[-1]).astype(
        np.float64
    )
    target = samples.target.astype(np.float64)
    return Scaler(
        tactile_mean=tactile_flat.mean(axis=0),
        tactile_std=np.maximum(tactile_flat.std(axis=0), 1e-6),
        action_mean=action_flat.mean(axis=0),
        action_std=np.maximum(action_flat.std(axis=0), 1e-6),
        target_mean=target.mean(axis=0),
        target_std=np.maximum(target.std(axis=0), 1e-6),
    )


def standardized_arrays(
    samples: TemporalSamples,
    scaler: Scaler,
    use_action: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tactile = (
        samples.tactile - scaler.tactile_mean[None, None, :]
    ) / scaler.tactile_std[None, None, :]
    if use_action:
        action = (
            samples.action - scaler.action_mean[None, None, :]
        ) / scaler.action_std[None, None, :]
        static = samples.static
    else:
        action = np.zeros_like(samples.action)
        static = np.zeros_like(samples.static)
    target = (samples.target - scaler.target_mean[None, :]) / scaler.target_std[None, :]
    return (
        np.asarray(tactile, dtype=np.float32),
        np.asarray(action, dtype=np.float32),
        np.asarray(static, dtype=np.float32),
        np.asarray(target, dtype=np.float32),
        np.asarray(samples.active, dtype=np.float32),
    )


class TemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                width,
                width,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=width,
            ),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.normalization = nn.LayerNorm(width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        updated = values + self.network(values)
        return self.normalization(updated.transpose(1, 2)).transpose(1, 2)


class TwoStreamTCN(nn.Module):
    def __init__(
        self,
        tactile_width: int,
        action_width: int,
        static_width: int,
        target_width: int,
        hidden_width: int,
        action_hidden_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.tactile_projection = nn.Linear(tactile_width, hidden_width)
        self.tactile_blocks = nn.Sequential(
            TemporalBlock(hidden_width, dilation=1, dropout=dropout),
            TemporalBlock(hidden_width, dilation=2, dropout=dropout),
        )
        self.action_projection = nn.Linear(action_width, action_hidden_width)
        self.action_blocks = nn.Sequential(
            TemporalBlock(action_hidden_width, dilation=1, dropout=dropout),
            TemporalBlock(action_hidden_width, dilation=2, dropout=dropout),
        )
        static_hidden = max(action_hidden_width // 2, 8)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_width, static_hidden),
            nn.GELU(),
        )
        combined = 2 * hidden_width + 2 * action_hidden_width + static_hidden
        self.head = nn.Sequential(
            nn.Linear(combined, 2 * hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_width, target_width),
        )

    @staticmethod
    def summarize(encoded: torch.Tensor) -> torch.Tensor:
        return torch.cat((encoded[:, :, -1], encoded.mean(dim=2)), dim=1)

    def forward(
        self,
        tactile: torch.Tensor,
        action: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        tactile_encoded = self.tactile_projection(tactile).transpose(1, 2)
        tactile_encoded = self.tactile_blocks(tactile_encoded)
        action_encoded = self.action_projection(action).transpose(1, 2)
        action_encoded = self.action_blocks(action_encoded)
        return self.head(
            torch.cat(
                (
                    self.summarize(tactile_encoded),
                    self.summarize(action_encoded),
                    self.static_encoder(static),
                ),
                dim=1,
            )
        )


def make_model(samples: TemporalSamples, config: Mapping[str, Any]) -> TwoStreamTCN:
    return TwoStreamTCN(
        tactile_width=int(samples.tactile.shape[-1]),
        action_width=int(samples.action.shape[-1]),
        static_width=int(samples.static.shape[-1]),
        target_width=int(samples.target.shape[-1]),
        hidden_width=int(config["hidden_width"]),
        action_hidden_width=int(config["action_hidden_width"]),
        dropout=float(config["dropout"]),
    )


def state_dict_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def make_loader(
    samples: TemporalSamples,
    scaler: Scaler,
    use_action: bool,
    batch_size: int,
    seed: int,
) -> DataLoader:
    tactile, action, static, target, active = standardized_arrays(
        samples,
        scaler,
        use_action,
    )
    dataset = TensorDataset(
        torch.from_numpy(tactile),
        torch.from_numpy(action),
        torch.from_numpy(static),
        torch.from_numpy(target),
        torch.from_numpy(active),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def weighted_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    elementwise = nn.functional.smooth_l1_loss(
        prediction,
        target,
        beta=float(config["smooth_l1_beta"]),
        reduction="none",
    )
    weights = 1.0 + float(config["active_coordinate_weight"]) * active
    return torch.sum(elementwise * weights) / torch.sum(weights)


def train_epochs(
    model: TwoStreamTCN,
    samples: TemporalSamples,
    scaler: Scaler,
    use_action: bool,
    config: Mapping[str, Any],
    epochs: int,
    seed: int,
    device: torch.device,
) -> None:
    loader = make_loader(
        samples,
        scaler,
        use_action,
        int(config["batch_size"]),
        seed,
    )
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=float(config["minimum_learning_rate"]),
    )
    for _ in range(epochs):
        for tactile, action, static, target, active in loader:
            tactile = tactile.to(device)
            action = action.to(device)
            static = static.to(device)
            target = target.to(device)
            active = active.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(tactile, action, static)
            loss = weighted_loss(prediction, target, active, config)
            if not torch.isfinite(loss):
                raise base.EvaluationError("TCN training produced nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["gradient_clip_norm"]),
            )
            optimizer.step()
        scheduler.step()


def predict_delta(
    model: TwoStreamTCN,
    samples: TemporalSamples,
    scaler: Scaler,
    use_action: bool,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    tactile, action, static, _, _ = standardized_arrays(
        samples,
        scaler,
        use_action,
    )
    dataset = TensorDataset(
        torch.from_numpy(tactile),
        torch.from_numpy(action),
        torch.from_numpy(static),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for tactile_batch, action_batch, static_batch in loader:
            prediction = model(
                tactile_batch.to(device),
                action_batch.to(device),
                static_batch.to(device),
            )
            rows.append(prediction.cpu().numpy())
    standardized = np.concatenate(rows)
    return scaler.target_mean[None, :] + standardized * scaler.target_std[None, :]


def evaluate_model(
    model: TwoStreamTCN,
    evaluation: Sequence[tuple[ObjectSource, TemporalSamples]],
    scaler: Scaler,
    use_action: bool,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    object_metrics: dict[str, float] = {}
    clip = float(config["normalized_feature_clip"])
    for source_object, samples in evaluation:
        delta = predict_delta(
            model,
            samples,
            scaler,
            use_action,
            int(config["evaluation_batch_size"]),
            device,
        )
        prediction = np.clip(samples.current + delta, 0.0, clip)
        object_metrics[source_object.object_id] = base.rmse(
            prediction,
            samples.truth,
            samples.active,
        )[1]
    return float(np.mean(list(object_metrics.values()))), object_metrics


def select_epoch_on_development(
    source_samples: TemporalSamples,
    evaluation: Sequence[tuple[ObjectSource, TemporalSamples]],
    use_action: bool,
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[int, float, list[float]]:
    scaler = fit_scaler(source_samples)
    seed_everything(seed)
    model = make_model(source_samples, config).to(device)
    loader = make_loader(
        source_samples,
        scaler,
        use_action,
        int(config["batch_size"]),
        seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    maximum_epochs = int(config["maximum_development_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(maximum_epochs, 1),
        eta_min=float(config["minimum_learning_rate"]),
    )
    best_metric = math.inf
    best_epoch = 1
    metrics: list[float] = []
    patience = int(config["early_stopping_patience"])
    minimum_epochs = int(config["minimum_development_epochs"])
    stale = 0
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        for tactile, action, static, target, active in loader:
            tactile = tactile.to(device)
            action = action.to(device)
            static = static.to(device)
            target = target.to(device)
            active = active.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(tactile, action, static)
            loss = weighted_loss(prediction, target, active, config)
            if not torch.isfinite(loss):
                raise base.EvaluationError("development TCN produced nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["gradient_clip_norm"]),
            )
            optimizer.step()
        scheduler.step()
        metric, _ = evaluate_model(
            model,
            evaluation,
            scaler,
            use_action,
            config,
            device,
        )
        metrics.append(metric)
        if metric < best_metric - float(config["minimum_improvement"]):
            best_metric = metric
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch >= minimum_epochs and stale >= patience:
            break
    return best_epoch, float(best_metric), metrics


def train_final_model(
    samples: TemporalSamples,
    use_action: bool,
    config: Mapping[str, Any],
    selected_epoch: int,
    seed: int,
    device: torch.device,
) -> tuple[TwoStreamTCN, Scaler]:
    scaler = fit_scaler(samples)
    seed_everything(seed)
    model = make_model(samples, config)
    train_epochs(
        model,
        samples,
        scaler,
        use_action,
        config,
        selected_epoch,
        seed,
        device,
    )
    return model, scaler


def comparison(
    rows: Sequence[dict[str, Any]],
    first: str,
    second: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    differences = np.asarray(
        [row[first] - row[second] for row in rows],
        dtype=np.float64,
    )
    wins = int(np.sum(differences < 0.0))
    ties = int(np.sum(differences == 0.0))
    losses = int(np.sum(differences > 0.0))
    denominator = float(np.mean([row[second] for row in rows]))
    return {
        "first": first,
        "second": second,
        "mean_difference": float(np.mean(differences)),
        "relative_change": float(np.mean(differences) / denominator),
        "object_bootstrap_95_interval": bootstrap_interval(
            differences,
            repetitions,
            seed,
        ),
        "object_wins_ties_losses": [wins, ties, losses],
        "exact_one_sided_sign_test_pvalue": exact_one_sided_sign_pvalue(
            wins,
            losses,
        ),
        "worst_object_regret": float(np.max(differences)),
    }


def held_out_action_family(row: Mapping[str, Any]) -> bool:
    source_families = {base.action_family(action) for action in row["source_actions"]}
    return str(row["target_action_family"]) not in source_families


def validate_protocol(
    protocol: Mapping[str, Any],
    root: Path,
    v3_protocol: Mapping[str, Any],
    v5_protocol: Mapping[str, Any],
) -> None:
    if protocol.get("schema") != (
        "bayesian-phystwin/deform360-tcn-baseline-protocol-v6"
    ):
        raise base.EvaluationError("unexpected temporal-baseline protocol schema")
    if Path(str(protocol["dataset_root"])) != root:
        raise base.EvaluationError("temporal-baseline dataset root changed")
    if list(protocol["development_object_ids"]) != list(
        v3_protocol["development_object_ids"]
    ):
        raise base.EvaluationError(
            "development roster does not match frozen v3 protocol"
        )
    if list(protocol["evaluation_object_ids"]) != list(
        v5_protocol["eligible_object_ids"]
    ):
        raise base.EvaluationError("92-object evaluation roster changed")
    if set(protocol["development_object_ids"]) & set(protocol["evaluation_object_ids"]):
        raise base.EvaluationError("development and evaluation rosters overlap")
    if protocol.get("changes_to_frozen_v3_method_allowed") is not False:
        raise base.EvaluationError("protocol permits changes to frozen v3 method")
    if protocol.get("paper_claim_authorized") is not False:
        raise base.EvaluationError("protocol self-authorized a paper claim")


def run(protocol_path: Path, root: Path) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    v3_protocol = read_json(Path(str(protocol["v3_protocol_path"])))
    v5_protocol = read_json(Path(str(protocol["v5_protocol_path"])))
    base_protocol = read_json(
        Path(str(v3_protocol["shared_preprocessing"]["base_protocol_path"]))
    )
    root = root.resolve(strict=True)
    validate_protocol(protocol, root, v3_protocol, v5_protocol)

    training = dict(protocol["training"])
    training["normalized_feature_clip"] = float(
        base_protocol["model"]["normalized_feature_clip"]
    )
    horizon = int(v3_protocol["shared_preprocessing"]["forecast_horizon_frames"])
    history_frames = int(training["history_frames"])
    action_samples = int(training["future_action_samples"])
    minimum_episodes = int(
        v5_protocol["selection"]["minimum_complete_episodes_per_object"]
    )
    seed = int(protocol["statistics"]["random_seed"])
    bootstrap_repetitions = int(protocol["statistics"]["bootstrap_repetitions"])
    torch.set_num_threads(int(training["torch_threads"]))
    torch.set_num_interop_threads(1)
    seed_everything(seed)
    device = torch.device(str(training["device"]))

    development_sources = [
        discover_source_object(
            root,
            str(object_id),
            minimum_episodes,
            base_protocol,
            horizon,
            history_frames,
            action_samples,
        )
        for object_id in protocol["development_object_ids"]
    ]
    development_source_samples = concatenate_samples(
        [item.source_samples for item in development_sources]
    )
    development_targets = [
        (
            item,
            load_target_samples(
                item,
                base_protocol,
                horizon,
                history_frames,
                action_samples,
            ),
        )
        for item in development_sources
    ]

    selected_epochs: dict[str, int] = {}
    development_metrics: dict[str, Any] = {}
    for index, (name, use_action) in enumerate(
        (("state_only_tcn", False), ("action_conditioned_tcn", True))
    ):
        selected_epoch, best_metric, trace = select_epoch_on_development(
            development_source_samples,
            development_targets,
            use_action,
            training,
            seed + 101 * index,
            device,
        )
        selected_epochs[name] = selected_epoch
        development_metrics[name] = {
            "selected_epoch": selected_epoch,
            "best_object_balanced_active_field_rmse": best_metric,
            "epoch_metric_trace": trace,
        }

    evaluation_sources = [
        discover_source_object(
            root,
            str(object_id),
            minimum_episodes,
            base_protocol,
            horizon,
            history_frames,
            action_samples,
        )
        for object_id in protocol["evaluation_object_ids"]
    ]
    evaluation_source_samples = concatenate_samples(
        [item.source_samples for item in evaluation_sources]
    )

    trained: dict[str, tuple[TwoStreamTCN, Scaler]] = {}
    freeze_records: dict[str, Any] = {}
    for index, (name, use_action) in enumerate(
        (("state_only_tcn", False), ("action_conditioned_tcn", True))
    ):
        model, scaler = train_final_model(
            evaluation_source_samples,
            use_action,
            training,
            selected_epochs[name],
            seed + 1009 + 101 * index,
            device,
        )
        trained[name] = (model, scaler)
        freeze_records[name] = {
            "selected_epoch": selected_epochs[name],
            "state_dict_sha256": state_dict_sha256(model),
            "scaler_sha256": canonical_digest(scaler.as_dict()),
            "parameter_count": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
        }
    final_model_freeze_sha256 = canonical_digest(
        {
            "training_config": training,
            "selected_epochs": selected_epochs,
            "models": freeze_records,
            "evaluation_source_object_ids": [
                item.object_id for item in evaluation_sources
            ],
            "evaluation_source_episode_ids": {
                item.object_id: [
                    descriptor.episode_id for descriptor in item.source_descriptors
                ]
                for item in evaluation_sources
            },
            "evaluation_target_episode_ids": {
                item.object_id: item.target_descriptor.episode_id
                for item in evaluation_sources
            },
        }
    )

    evaluation_targets = [
        (
            item,
            load_target_samples(
                item,
                base_protocol,
                horizon,
                history_frames,
                action_samples,
            ),
        )
        for item in evaluation_sources
    ]
    tcn_metrics: dict[str, dict[str, float]] = {}
    for name, use_action in (
        ("state_only_tcn", False),
        ("action_conditioned_tcn", True),
    ):
        model, scaler = trained[name]
        _, per_object = evaluate_model(
            model,
            evaluation_targets,
            scaler,
            use_action,
            training,
            device,
        )
        tcn_metrics[name] = per_object

    rng = np.random.default_rng(int(v3_protocol["statistics"]["random_seed"]))
    v3_rows: dict[str, dict[str, Any]] = {}
    for source_object in evaluation_sources:
        row = v3.evaluate_object(
            list(source_object.descriptors),
            v3_protocol,
            base_protocol,
            rng,
        )
        v3_rows[source_object.object_id] = row

    rows: list[dict[str, Any]] = []
    for source_object in evaluation_sources:
        object_id = source_object.object_id
        v3_row = v3_rows[object_id]
        rows.append(
            {
                "object_id": object_id,
                "source_episode_ids": [
                    descriptor.episode_id
                    for descriptor in source_object.source_descriptors
                ],
                "source_actions": [
                    descriptor.action for descriptor in source_object.source_descriptors
                ],
                "target_episode_id": source_object.target_descriptor.episode_id,
                "target_action": source_object.target_descriptor.action,
                "target_action_family": base.action_family(
                    source_object.target_descriptor.action
                ),
                "held_out_action_family": held_out_action_family(v3_row),
                "persistence": float(
                    v3_row["metrics"]["persistence"]["active_field_rmse"]
                ),
                "v3_state_kernel": float(
                    v3_row["metrics"]["state_kernel"]["active_field_rmse"]
                ),
                "v3_shuffled_action": float(
                    v3_row["metrics"]["shuffled_action_control"]["active_field_rmse"]
                ),
                "v3_bayesian_action_ensemble": float(
                    v3_row["metrics"]["bayesian_action_ensemble"]["active_field_rmse"]
                ),
                "state_only_tcn": float(tcn_metrics["state_only_tcn"][object_id]),
                "action_conditioned_tcn": float(
                    tcn_metrics["action_conditioned_tcn"][object_id]
                ),
            }
        )

    methods = (
        "persistence",
        "v3_state_kernel",
        "v3_shuffled_action",
        "v3_bayesian_action_ensemble",
        "state_only_tcn",
        "action_conditioned_tcn",
    )
    aggregate = {
        method: float(np.mean([row[method] for row in rows])) for method in methods
    }
    comparisons = {
        "v3_ensemble_vs_persistence": comparison(
            rows,
            "v3_bayesian_action_ensemble",
            "persistence",
            bootstrap_repetitions,
            seed,
        ),
        "v3_ensemble_vs_state_only_tcn": comparison(
            rows,
            "v3_bayesian_action_ensemble",
            "state_only_tcn",
            bootstrap_repetitions,
            seed + 1,
        ),
        "v3_ensemble_vs_action_conditioned_tcn": comparison(
            rows,
            "v3_bayesian_action_ensemble",
            "action_conditioned_tcn",
            bootstrap_repetitions,
            seed + 2,
        ),
        "action_tcn_vs_state_tcn": comparison(
            rows,
            "action_conditioned_tcn",
            "state_only_tcn",
            bootstrap_repetitions,
            seed + 3,
        ),
        "action_tcn_vs_persistence": comparison(
            rows,
            "action_conditioned_tcn",
            "persistence",
            bootstrap_repetitions,
            seed + 4,
        ),
    }
    held_out_rows = [row for row in rows if row["held_out_action_family"]]
    if len(held_out_rows) != int(protocol["statistics"]["expected_held_out_count"]):
        raise base.EvaluationError(
            f"held-out action-family subset changed: {len(held_out_rows)}"
        )
    held_out = {
        "object_count": len(held_out_rows),
        "methods": {
            method: float(np.mean([row[method] for row in held_out_rows]))
            for method in methods
        },
        "v3_ensemble_vs_persistence": comparison(
            held_out_rows,
            "v3_bayesian_action_ensemble",
            "persistence",
            bootstrap_repetitions,
            seed + 11,
        ),
        "v3_ensemble_vs_state_kernel": comparison(
            held_out_rows,
            "v3_bayesian_action_ensemble",
            "v3_state_kernel",
            bootstrap_repetitions,
            seed + 12,
        ),
        "v3_ensemble_vs_action_conditioned_tcn": comparison(
            held_out_rows,
            "v3_bayesian_action_ensemble",
            "action_conditioned_tcn",
            bootstrap_repetitions,
            seed + 13,
        ),
        "action_tcn_vs_state_tcn": comparison(
            held_out_rows,
            "action_conditioned_tcn",
            "state_only_tcn",
            bootstrap_repetitions,
            seed + 14,
        ),
    }

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 6,
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "dataset_root": str(root),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "torch_version": torch.__version__,
        "device": str(device),
        "development_epoch_selection": development_metrics,
        "final_model_freeze": {
            "freeze_sha256": final_model_freeze_sha256,
            "models": freeze_records,
            "target_tactile_opened_after_freeze": True,
        },
        "summary": {
            "object_count": len(rows),
            "horizon_frames": horizon,
            "history_frames": history_frames,
            "methods": aggregate,
            "comparisons": comparisons,
            "held_out_action_family": held_out,
        },
        "objects": rows,
        "information_boundary": {
            "v3_method_changed": False,
            "tcn_architecture_selected_on_original_14_object_development_roster": True,
            "tcn_final_models_trained_on_92_object_source_episodes_only": True,
            "tcn_final_models_content_hashed_before_92_target_tactile_open": True,
            "evaluation_target_robot_trajectory_used_as_known_input": True,
            "evaluation_target_tactile_used_for_scoring_only": True,
            "camera_pixels_opened": False,
            "geometry_or_point_cloud_opened": False,
            "new_measurements_collected": False,
            "post_confirmation_competitive_baseline_audit": True,
            "globally_fresh_confirmation": False,
        },
        "protocol": protocol,
        "paper_claim_authorized": False,
        "strict_counterfactual_claim_authorized": False,
    }
    result["result_sha256"] = canonical_digest(result)
    return result


def make_report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    methods = summary["methods"]
    lines = [
        "# Deform360 competitive temporal-baseline audit v6",
        "",
        f"- Status: **{result['status']}**",
        f"- Objects: **{summary['object_count']}**",
        f"- Horizon/history: **{summary['horizon_frames']}/{summary['history_frames']} frames**",
        f"- Torch/device: `{result['torch_version']}` / `{result['device']}`",
        "- Final TCN models frozen before evaluation target tactile loading: **yes**",
        "",
        "## Object-balanced active-field RMSE",
        "",
        "| Method | RMSE |",
        "|---|---:|",
    ]
    for name, value in methods.items():
        lines.append(f"| `{name}` | {value:.8g} |")
    lines.extend(
        [
            "",
            "## Primary comparisons",
            "",
            "Negative differences favor the first method.",
            "",
            "| Comparison | Difference | Relative | 95% bootstrap | W/T/L | sign-test p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, value in summary["comparisons"].items():
        interval = value["object_bootstrap_95_interval"]
        wtl = value["object_wins_ties_losses"]
        lines.append(
            f"| `{name}` | {value['mean_difference']:.8g} | "
            f"{value['relative_change']:+.2%} | "
            f"[{interval[0]:.8g}, {interval[1]:.8g}] | "
            f"{wtl[0]}/{wtl[1]}/{wtl[2]} | "
            f"{value['exact_one_sided_sign_test_pvalue']:.4g} |"
        )
    held = summary["held_out_action_family"]
    lines.extend(
        [
            "",
            "## Target action family absent from same-object source episodes",
            "",
            f"Objects: **{held['object_count']}**",
            "",
            "| Comparison | Difference | Relative | 95% bootstrap | W/T/L |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in (
        "v3_ensemble_vs_persistence",
        "v3_ensemble_vs_state_kernel",
        "v3_ensemble_vs_action_conditioned_tcn",
        "action_tcn_vs_state_tcn",
    ):
        value = held[name]
        interval = value["object_bootstrap_95_interval"]
        wtl = value["object_wins_ties_losses"]
        lines.append(
            f"| `{name}` | {value['mean_difference']:.8g} | "
            f"{value['relative_change']:+.2%} | "
            f"[{interval[0]:.8g}, {interval[1]:.8g}] | "
            f"{wtl[0]}/{wtl[1]}/{wtl[2]} |"
        )
    lines.extend(
        [
            "",
            "This is a post-confirmation competitive-baseline audit. The frozen v3",
            "Bayesian action ensemble is unchanged. The neural architecture and training",
            "budget were selected on the original 14-object development roster; final TCN",
            "weights and source-only scalers were hashed before the 92 evaluation targets",
            "were loaded by this execution. The 92 outcomes had been opened previously by",
            "the v5 confirmation and are therefore not globally fresh.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write empty object table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(materialized[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            encoded = dict(row)
            encoded["source_episode_ids"] = json.dumps(encoded["source_episode_ids"])
            encoded["source_actions"] = json.dumps(encoded["source_actions"])
            writer.writerow(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.data_root)
    write_json(args.output_json, result)
    args.output_report.write_text(make_report(result), encoding="utf-8")
    write_csv(args.output_csv, result["objects"])
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
