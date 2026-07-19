"""Source-trained nonlinear residual dynamics with a persistence fallback."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .phystwin_shared_residual_velocity import (
    SharedResidualVelocityConfig,
    SharedResidualVelocityEpisode,
    _exogenous_features,
    _interval_metrics,
    _load_episode,
    _ratios,
    _temporally_fill,
    _training_indices,
)


SHARED_NONLINEAR_RESIDUAL_CONTRACT = (
    "source-trained-graph-local-multistep-residual-v1"
)


@dataclass(frozen=True)
class SharedNonlinearResidualConfig:
    """Fixed architecture, optimization, and source-gate settings."""

    hidden_dim: int = 96
    hidden_layers: int = 3
    rollout_horizon: int = 12
    training_steps: int = 2000
    maximum_training_points: int = 384
    neighbor_count: int = 8
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
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
class _PreparedEpisode:
    loaded: Any
    indices: np.ndarray
    state: np.ndarray
    velocity: np.ndarray
    exogenous: np.ndarray
    neighbors: np.ndarray
    valid: np.ndarray
    residual_cap: float
    velocity_cap: float


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_with_digest(path: Path, payload: Mapping[str, object]) -> str:
    """Write canonical evidence JSON and an external SHA-256 sidecar."""

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = _sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def _validate_config(config: SharedNonlinearResidualConfig) -> None:
    if config.hidden_dim < 1 or config.hidden_layers < 1:
        raise ValueError("the nonlinear model must contain hidden units and layers")
    if config.rollout_horizon < 2 or config.training_steps < 1:
        raise ValueError("rollout_horizon and training_steps must be positive")
    if config.maximum_training_points < 2 or config.neighbor_count < 1:
        raise ValueError("point and neighbor counts must be positive")
    if config.learning_rate <= 0.0 or config.weight_decay < 0.0:
        raise ValueError("optimizer scales are invalid")
    if config.gradient_clip <= 0.0:
        raise ValueError("gradient_clip must be positive")
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


def _knn_indices(points: np.ndarray, neighbor_count: int) -> np.ndarray:
    """Return deterministic non-self nearest neighbors for each point."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("points must have shape (N, 3) with N >= 2")
    if not np.isfinite(values).all():
        raise ValueError("points must be finite")
    count = min(int(neighbor_count), len(values) - 1)
    if count < 1:
        raise ValueError("neighbor_count must be positive")
    try:
        from scipy.spatial import cKDTree

        _, indices = cKDTree(values).query(values, k=count + 1)
        result = np.asarray(indices, dtype=np.int64)[:, 1:]
    except (ImportError, OSError, ValueError):
        squared = np.sum(np.square(values[:, None] - values[None, :]), axis=2)
        np.fill_diagonal(squared, np.inf)
        result = np.argsort(squared, axis=1, kind="stable")[:, :count]
    if result.shape != (len(values), count):
        raise RuntimeError("nearest-neighbor construction returned an invalid shape")
    if np.any(result == np.arange(len(values))[:, None]):
        raise RuntimeError("nearest-neighbor construction retained self edges")
    return result


def _prepare_episode(
    loaded: Any,
    config: SharedNonlinearResidualConfig,
    *,
    training_subset: bool,
) -> _PreparedEpisode:
    original_count = loaded.observed.shape[1]
    indices = (
        _training_indices(
            loaded.valid,
            end_frame=len(loaded.observed),
            maximum_points=config.maximum_training_points,
        )
        if training_subset
        else np.arange(original_count, dtype=np.int64)
    )
    filled = _temporally_fill(loaded.residual, loaded.valid, len(loaded.observed))
    state = filled[:, indices] / loaded.object_scale
    velocity = np.zeros_like(state)
    velocity[1:] = np.diff(state, axis=0)
    exogenous = np.zeros((len(state), len(indices), 25), dtype=np.float32)
    for frame in range(1, len(state)):
        exogenous[frame] = _exogenous_features(loaded, frame, indices)
    velocity_norm = np.linalg.norm(velocity[1:], axis=2)
    velocity_cap = max(
        config.maximum_velocity_multiplier
        * float(np.quantile(velocity_norm, 0.99)),
        1.0e-6,
    )
    return _PreparedEpisode(
        loaded=loaded,
        indices=indices,
        state=state.astype(np.float32),
        velocity=velocity.astype(np.float32),
        exogenous=exogenous,
        neighbors=_knn_indices(
            loaded.baseline[0, indices], config.neighbor_count
        ),
        valid=loaded.valid[:, indices],
        residual_cap=float(config.maximum_residual_m / loaded.object_scale),
        velocity_cap=velocity_cap,
    )


def _feature_dimension() -> int:
    return 3 + 3 + 3 + 3 + 25


def _build_model(torch: Any, config: SharedNonlinearResidualConfig) -> Any:
    nn = torch.nn
    layers: list[Any] = []
    width = _feature_dimension()
    for _ in range(config.hidden_layers):
        layers.extend((nn.Linear(width, config.hidden_dim), nn.SiLU()))
        width = config.hidden_dim
    output = nn.Linear(width, 3)
    nn.init.zeros_(output.weight)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


def _graph_features(
    torch: Any,
    state: Any,
    velocity: Any,
    exogenous: Any,
    neighbors: Any,
) -> Any:
    neighbor_state = state[neighbors].mean(dim=1)
    neighbor_velocity = velocity[neighbors].mean(dim=1)
    return torch.cat(
        (
            state,
            velocity,
            neighbor_state - state,
            neighbor_velocity - velocity,
            exogenous,
        ),
        dim=1,
    )


def _step_state(
    torch: Any,
    model: Any,
    state: Any,
    velocity: Any,
    exogenous: Any,
    neighbors: Any,
    *,
    velocity_cap: float,
    residual_cap: float,
) -> tuple[Any, Any]:
    features = _graph_features(torch, state, velocity, exogenous, neighbors)
    next_velocity = float(velocity_cap) * torch.tanh(model(features))
    next_state = state + next_velocity
    norm = torch.linalg.vector_norm(next_state, dim=1, keepdim=True)
    scale = torch.clamp(float(residual_cap) / torch.clamp(norm, min=1.0e-12), max=1.0)
    return next_state * scale, next_velocity


def _train_one_model(
    prepared: Sequence[_PreparedEpisode],
    config: SharedNonlinearResidualConfig,
    *,
    seed: int,
    output_path: Path,
) -> tuple[Any, dict[str, object]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("nonlinear residual training requires PyTorch") from exc

    if not prepared:
        raise ValueError("training requires at least one episode")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np_rng = np.random.default_rng(int(seed))
    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu"
    )
    model = _build_model(torch, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    losses: list[float] = []
    model.train()
    for step in range(config.training_steps):
        episode = prepared[step % len(prepared)]
        horizon = min(config.rollout_horizon, len(episode.state) - 2)
        maximum_start = len(episode.state) - horizon - 1
        start = int(np_rng.integers(1, maximum_start + 1))
        state = torch.as_tensor(episode.state[start], device=device)
        velocity = torch.as_tensor(episode.velocity[start], device=device)
        neighbors = torch.as_tensor(
            episode.neighbors, dtype=torch.long, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device)
        weight_sum = torch.zeros((), device=device)
        for offset in range(1, horizon + 1):
            frame = start + offset
            exogenous = torch.as_tensor(episode.exogenous[frame], device=device)
            state, velocity = _step_state(
                torch,
                model,
                state,
                velocity,
                exogenous,
                neighbors,
                velocity_cap=episode.velocity_cap,
                residual_cap=episode.residual_cap,
            )
            target = torch.as_tensor(episode.state[frame], device=device)
            valid = torch.as_tensor(
                episode.valid[frame], dtype=state.dtype, device=device
            )
            point_loss = torch.nn.functional.smooth_l1_loss(
                state,
                target,
                reduction="none",
                beta=0.05,
            ).mean(dim=1)
            horizon_weight = float(offset) / float(horizon)
            loss = loss + horizon_weight * torch.sum(valid * point_loss)
            weight_sum = weight_sum + horizon_weight * torch.sum(valid)
        loss = loss / torch.clamp(weight_sum, min=1.0)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonlinear residual training produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "contract": SHARED_NONLINEAR_RESIDUAL_CONTRACT,
            "seed": int(seed),
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
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


def _rollout_model(
    model: Any,
    prepared: _PreparedEpisode,
    *,
    start_frame: int,
    end_frame: int,
    device_name: str,
) -> np.ndarray:
    import torch

    if not 1 <= start_frame < end_frame <= len(prepared.state):
        raise ValueError("rollout interval is invalid")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model.eval()
    state = torch.as_tensor(prepared.state[start_frame - 1], device=device)
    velocity = torch.as_tensor(prepared.velocity[start_frame - 1], device=device)
    neighbors = torch.as_tensor(
        prepared.neighbors, dtype=torch.long, device=device
    )
    result = []
    with torch.no_grad():
        for frame in range(start_frame, end_frame):
            exogenous = torch.as_tensor(prepared.exogenous[frame], device=device)
            state, velocity = _step_state(
                torch,
                model,
                state,
                velocity,
                exogenous,
                neighbors,
                velocity_cap=prepared.velocity_cap,
                residual_cap=prepared.residual_cap,
            )
            result.append(state.detach().cpu().numpy())
    return np.asarray(result, dtype=np.float32) * prepared.loaded.object_scale


def blend_with_persistence(
    dynamic: np.ndarray,
    endpoint: np.ndarray,
    coefficient: float,
) -> np.ndarray:
    """Shrink a dynamic rollout toward exact endpoint persistence."""

    values = np.asarray(dynamic, dtype=float)
    anchor = np.asarray(endpoint, dtype=float)
    if values.ndim != 3 or values.shape[1:] != anchor.shape:
        raise ValueError("dynamic and endpoint residual shapes disagree")
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError("coefficient must lie in [0, 1]")
    persistence = np.broadcast_to(anchor, values.shape)
    return persistence + float(coefficient) * (values - persistence)


def _episode_specs(
    data_root: Path,
    cases: Sequence[str],
    fit_fraction: float,
) -> list[SharedResidualVelocityEpisode]:
    result = []
    for case in cases:
        case_root = data_root / case
        split = json.loads((case_root / "split.json").read_text(encoding="utf-8"))
        train_start, train_end = (int(value) for value in split["train"])
        if train_start != 0:
            raise ValueError(f"{case}: expected a zero-based training split")
        fit_end = int(train_end * fit_fraction)
        result.append(
            SharedResidualVelocityEpisode(
                case=case,
                final_data=str((case_root / "final_data.pkl").resolve()),
                baseline_trajectory=str((case_root / "inference.pkl").resolve()),
                gt_track_3d=str((case_root / "gt_track_3d.pkl").resolve()),
                fit_end_frame=fit_end,
                train_end_frame=train_end,
            )
        )
    return result


def _load_protocol(
    path: str | Path,
) -> tuple[dict[str, object], SharedNonlinearResidualConfig]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("contract") != SHARED_NONLINEAR_RESIDUAL_CONTRACT:
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
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("source protocol omits its model configuration")
    config = SharedNonlinearResidualConfig(
        **{key: tuple(value) if key in {"seeds", "blend_candidates"} else value for key, value in model.items()}
    )
    _validate_config(config)
    return payload, config


def fit_shared_nonlinear_residual_source_gate(
    data_root: str | Path,
    protocol_path: str | Path,
    output_dir: str | Path,
    *,
    device: str | None = None,
) -> dict[str, object]:
    """Cross-fit source episodes and stop before any target trajectory is read."""

    protocol, config = _load_protocol(protocol_path)
    if device is not None:
        config = SharedNonlinearResidualConfig(**{**asdict(config), "device": device})
    root = Path(data_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fit_fraction = float(protocol.get("fit_fraction", 0.75))
    if not 0.5 <= fit_fraction < 1.0:
        raise ValueError("fit_fraction must lie in [0.5, 1)")
    source_cases = tuple(str(case) for case in protocol["source_cases"])
    specs = _episode_specs(root, source_cases, fit_fraction)
    base_config = SharedResidualVelocityConfig(
        maximum_training_points=config.maximum_training_points,
        interpolation_neighbors=config.interpolation_neighbors,
        controller_kernel_fraction=config.controller_kernel_fraction,
        maximum_velocity_multiplier=config.maximum_velocity_multiplier,
        maximum_residual_m=config.maximum_residual_m,
    )
    loaded = {spec.case: _load_episode(spec, base_config) for spec in specs}
    training_prepared = {
        case: _prepare_episode(value, config, training_subset=True)
        for case, value in loaded.items()
    }
    evaluation_prepared = {
        case: _prepare_episode(value, config, training_subset=False)
        for case, value in loaded.items()
    }

    fold_predictions: dict[str, dict[str, np.ndarray]] = {}
    fold_training: list[dict[str, object]] = []
    for fold_index, raw_fold in enumerate(protocol["source_folds"]):
        held_out = tuple(str(case) for case in raw_fold["held_out_cases"])
        training_cases = tuple(case for case in source_cases if case not in held_out)
        seed_models = []
        seed_summaries = []
        for seed in config.seeds:
            checkpoint = output / f"fold_{fold_index:02d}" / f"model_seed_{seed}.pt"
            model, training_summary = _train_one_model(
                [training_prepared[case] for case in training_cases],
                config,
                seed=seed,
                output_path=checkpoint,
            )
            seed_models.append(model)
            seed_summaries.append(training_summary)
        predictions: dict[str, np.ndarray] = {}
        for case in held_out:
            prepared = evaluation_prepared[case]
            members = [
                _rollout_model(
                    model,
                    prepared,
                    start_frame=prepared.loaded.spec.fit_end_frame,
                    end_frame=prepared.loaded.spec.train_end_frame,
                    device_name=config.device,
                )
                for model in seed_models
            ]
            predictions[case] = np.mean(members, axis=0)
        fold_predictions[str(fold_index)] = predictions
        fold_training.append(
            {
                "name": str(raw_fold.get("name", f"fold_{fold_index}")),
                "held_out_cases": list(held_out),
                "training_cases": list(training_cases),
                "seed_models": seed_summaries,
            }
        )

    candidates = []
    for blend in config.blend_candidates:
        case_results = []
        fold_both_wins = []
        for fold_index, raw_fold in enumerate(protocol["source_folds"]):
            fold_case_results = []
            for case in raw_fold["held_out_cases"]:
                episode = loaded[str(case)]
                prepared = evaluation_prepared[str(case)]
                endpoint = prepared.state[episode.spec.fit_end_frame - 1]
                endpoint = endpoint * episode.object_scale
                tracked = blend_with_persistence(
                    fold_predictions[str(fold_index)][str(case)],
                    endpoint,
                    float(blend),
                )
                dynamic_metrics, _ = _interval_metrics(
                    episode,
                    tracked,
                    start_frame=episode.spec.fit_end_frame,
                    end_frame=episode.spec.train_end_frame,
                    config=base_config,
                )
                persistence = np.broadcast_to(endpoint, tracked.shape)
                persistence_metrics, _ = _interval_metrics(
                    episode,
                    persistence,
                    start_frame=episode.spec.fit_end_frame,
                    end_frame=episode.spec.train_end_frame,
                    config=base_config,
                )
                ratios = _ratios(dynamic_metrics, persistence_metrics)
                record = {
                    "case": str(case),
                    "ratios_relative_to_persistence": ratios,
                    "persistence_official_evaluation": persistence_metrics,
                    "dynamic_official_evaluation": dynamic_metrics,
                }
                case_results.append(record)
                fold_case_results.append(record)
            fold_both_wins.append(
                all(
                    max(result["ratios_relative_to_persistence"].values()) < 1.0
                    for result in fold_case_results
                )
            )
        aggregate_ratios = {
            metric: float(
                np.mean(
                    [result["ratios_relative_to_persistence"][metric] for result in case_results]
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
                "both_win_fold_count": int(sum(fold_both_wins)),
                "maximum_case_metric_ratio": max(
                    max(result["ratios_relative_to_persistence"].values())
                    for result in case_results
                ),
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
        "contract": SHARED_NONLINEAR_RESIDUAL_CONTRACT,
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
        "fold_training": fold_training,
        "selection": {"selected_candidate": selected, "candidates": candidates},
        "claim_boundary": (
            "Complete outcomes are used only for registered source interactions. "
            "No target trajectory, target metric, or target future artifact is read."
        ),
    }
    summary_path = output / "source_gate_summary.json"
    digest = _write_json_with_digest(summary_path, summary)
    return {
        **summary,
        "summary_artifact": {"path": str(summary_path), "sha256": digest},
    }
