"""Source-only smoke experiment for the Bayesian Deform360 residual model."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .deform360_bayesian_residual import (
    BayesianResidualModelConfig,
    EquivariantBayesianResidual,
    clustered_student_t_nll,
    load_bayesian_residual_config,
)
from .deform360_bayesian_residual_data import (
    ControllerSurfaceProvider,
    Deform360ResidualSourceEpisode,
    load_deform360_residual_source_episode,
)
from .deform360_independent_source import EXPECTED_INDEPENDENT_SOURCE_EPISODES


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class ResidualTrainingConfig:
    steps: int = 2000
    rollout_steps: int = 5
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-6
    physics_velocity_retention: float = 0.85
    position_loss_scale: float = 2500.0
    utility_loss_scale: float = 0.1
    gradient_norm: float = 1.0
    seed: int = 0


@dataclass
class _TorchEpisode:
    source: Deform360ResidualSourceEpisode
    positions: torch.Tensor
    physics_positions: torch.Tensor
    reliability: torch.Tensor
    controllers: torch.Tensor
    controller_velocities: torch.Tensor
    closure: torch.Tensor
    edges: torch.Tensor
    clusters: torch.Tensor


def _torch_episode(
    episode: Deform360ResidualSourceEpisode, device: torch.device
) -> _TorchEpisode:
    return _TorchEpisode(
        source=episode,
        positions=torch.tensor(episode.positions_m, device=device),
        physics_positions=torch.tensor(episode.physics_positions_m, device=device),
        reliability=torch.tensor(episode.prior_reliability, device=device),
        controllers=torch.tensor(episode.controller_positions_m, device=device),
        controller_velocities=torch.tensor(
            episode.controller_velocities_mps, device=device
        ),
        closure=torch.tensor(episode.closure_probability, device=device),
        edges=torch.tensor(episode.edge_index, device=device),
        clusters=torch.tensor(episode.cluster_ids, device=device),
    )


def _contact_probability_torch(
    positions: torch.Tensor,
    controllers: torch.Tensor,
    closure: torch.Tensor,
    proximity_scale_m: float = 0.03,
    *,
    relative_to_nearest: bool = True,
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(
        positions[:, :, None] - controllers[:, None], dim=-1
    )
    if relative_to_nearest:
        distance = distance - torch.amin(distance, dim=1, keepdim=True)
    proximity = torch.exp(-0.5 * torch.square(distance / proximity_scale_m))
    return proximity * closure[:, None]


def _causal_velocity(
    positions: torch.Tensor, frame_index: int, frame_interval_s: float
) -> torch.Tensor:
    if frame_index <= 0:
        return torch.zeros_like(positions[frame_index])
    return (positions[frame_index] - positions[frame_index - 1]) / frame_interval_s


def _result_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cross_fitted_trust_scales(
    trust_diagnosis_path: str | Path,
    closure_diagnosis_path: str | Path,
) -> dict[str, float]:
    """Recover the frozen closure-AND-self-diagnostic source decisions."""

    trust = json.loads(Path(trust_diagnosis_path).read_text(encoding="utf-8"))
    closure = json.loads(Path(closure_diagnosis_path).read_text(encoding="utf-8"))
    _require(
        trust.get("artifact_kind") == "Deform360SameObjectTrustDiagnosis",
        "unexpected reusable-trust diagnosis kind",
    )
    _require(
        closure.get("artifact_kind")
        == "Deform360IndependentSourceFailureDiagnosis",
        "unexpected closure diagnosis kind",
    )
    _require(
        trust.get("result_sha256") == _result_sha256(trust),
        "reusable-trust diagnosis checksum differs",
    )
    _require(
        closure.get("result_sha256") == _result_sha256(closure),
        "closure diagnosis checksum differs",
    )
    self_diagnostic = trust["feature_sets"]["simulation_self_diagnostic"][
        "selected_alpha_by_episode"
    ]
    closure_scale = closure["exploratory_group_cross_fit"][
        "selected_alpha_by_episode"
    ]
    expected = {
        f"{object_id}/{episode_id}"
        for object_id, episode_ids in EXPECTED_INDEPENDENT_SOURCE_EPISODES.items()
        for episode_id in episode_ids
    }
    _require(
        set(self_diagnostic) == expected and set(closure_scale) == expected,
        "cross-fitted trust decisions do not cover the frozen source panel",
    )
    return {
        episode_key: (
            float(self_diagnostic[episode_key])
            if float(closure_scale[episode_key]) > 0.0
            else 0.0
        )
        for episode_key in sorted(expected)
    }


def load_source_residual_panel(
    source_root: str | Path,
    *,
    maximum_node_count: int,
    neighbor_count: int,
    controller_surface_provider: ControllerSurfaceProvider | None = None,
    controller_points_per_gripper: int = 32,
    physics_root: str | Path | None = None,
    physics_response_scale_by_episode: Mapping[str, float] | None = None,
    physics_reference_response_scale: float = 0.9,
) -> list[Deform360ResidualSourceEpisode]:
    """Load exactly the 27 already-open development episodes."""

    root = Path(source_root).resolve()
    expected_keys = {
        f"{object_id}/{episode_id}"
        for object_id, episode_ids in EXPECTED_INDEPENDENT_SOURCE_EPISODES.items()
        for episode_id in episode_ids
    }
    if physics_response_scale_by_episode is not None:
        _require(
            physics_root is not None,
            "trusted physical response scales require a physical prediction root",
        )
        _require(
            set(physics_response_scale_by_episode) == expected_keys,
            "trusted physical response scales do not cover the source panel",
        )
    episodes = []
    for object_id, episode_ids in EXPECTED_INDEPENDENT_SOURCE_EPISODES.items():
        for episode_id in episode_ids:
            episode_key = f"{object_id}/{episode_id}"
            directory = (
                root / f"{object_id}-ep{episode_id:04d}" / "episode_0000"
            )
            physics_path = None
            if physics_root is not None:
                physics_directory = (
                    Path(physics_root).resolve()
                    / f"{object_id}-ep{episode_id:04d}"
                )
                candidates = (
                    physics_directory / "prediction.npz",
                    physics_directory / "sealed_prediction.npz",
                )
                physics_path = next(
                    (candidate for candidate in candidates if candidate.is_file()),
                    candidates[0],
                )
            episodes.append(
                load_deform360_residual_source_episode(
                    directory,
                    object_id=object_id,
                    episode_id=episode_id,
                    maximum_node_count=maximum_node_count,
                    neighbor_count=neighbor_count,
                    controller_surface_provider=controller_surface_provider,
                    controller_points_per_gripper=controller_points_per_gripper,
                    physics_prediction_path=physics_path,
                    physics_response_scale=(
                        None
                        if physics_response_scale_by_episode is None
                        else float(physics_response_scale_by_episode[episode_key])
                    ),
                    physics_reference_response_scale=physics_reference_response_scale,
                )
            )
    _require(len(episodes) == 27, "source residual panel is incomplete")
    return episodes


def train_residual_model(
    episodes: Sequence[Deform360ResidualSourceEpisode],
    *,
    model_config: BayesianResidualModelConfig,
    training_config: ResidualTrainingConfig,
    device: str | torch.device,
) -> tuple[EquivariantBayesianResidual, dict[str, Any]]:
    """Train a rollout-aware residual on source episodes only."""

    _require(bool(episodes), "residual training panel is empty")
    _require(training_config.steps >= 1, "training steps must be positive")
    _require(training_config.rollout_steps >= 1, "rollout steps must be positive")
    torch_device = torch.device(device)
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(training_config.seed)
    model = EquivariantBayesianResidual(model_config).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    torch_episodes = [_torch_episode(episode, torch_device) for episode in episodes]
    generator = random.Random(training_config.seed)
    loss_history: list[float] = []
    for step in range(training_config.steps):
        episode = torch_episodes[generator.randrange(len(torch_episodes))]
        maximum_start = len(episode.positions) - training_config.rollout_steps - 1
        start = generator.randrange(maximum_start + 1)
        dt = float(episode.source.frame_interval_s)
        current = episode.positions[start : start + 1]
        velocity = _causal_velocity(episode.positions, start, dt)[None]
        static_reliability = episode.reliability[0:1]
        losses = []
        for offset in range(training_config.rollout_steps):
            frame_index = start + offset
            target = episode.positions[frame_index + 1 : frame_index + 2]
            physics_delta = (
                episode.physics_positions[frame_index + 1 : frame_index + 2]
                - episode.physics_positions[frame_index : frame_index + 1]
            )
            physics_velocity = physics_delta / dt
            physics_position = current + physics_delta
            contact = _contact_probability_torch(
                current,
                episode.controllers[frame_index : frame_index + 1],
                episode.closure[frame_index : frame_index + 1],
                relative_to_nearest=(
                    episode.source.controller_geometry == "end_effector_origins"
                ),
            )
            prediction = model(
                positions_m=current,
                velocities_mps=velocity,
                physics_positions_m=physics_position,
                physics_velocities_mps=physics_velocity,
                controller_positions_m=episode.controllers[
                    frame_index : frame_index + 1
                ],
                controller_velocities_mps=episode.controller_velocities[
                    frame_index : frame_index + 1
                ],
                contact_probabilities=contact,
                prior_reliability=static_reliability,
                edge_index=episode.edges,
            )
            target_residual = (target - physics_position) / dt
            nll = clustered_student_t_nll(
                target_residual,
                prediction,
                episode.reliability[frame_index : frame_index + 1],
                episode.clusters[None],
            )
            corrected_velocity = physics_velocity + prediction.mean_mps
            corrected_position = physics_position + prediction.mean_mps * dt
            position_loss = torch.mean(torch.square(corrected_position - target))
            baseline_error = torch.sum(torch.square(target_residual), dim=-1)
            corrected_error = torch.sum(
                torch.square(target_residual - prediction.mean_mps), dim=-1
            )
            utility_target = (corrected_error < baseline_error).to(
                prediction.utility_probability.dtype
            )
            utility_loss = nn.functional.binary_cross_entropy(
                prediction.utility_probability,
                utility_target.detach(),
            )
            losses.append(
                nll
                + training_config.position_loss_scale * position_loss
                + training_config.utility_loss_scale * utility_loss
            )
            current = corrected_position
            velocity = corrected_velocity
        loss = torch.mean(torch.stack(losses))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_norm)
        optimizer.step()
        loss_history.append(float(loss.detach().cpu()))
    model.eval()
    tail = loss_history[-min(100, len(loss_history)) :]
    return model, {
        "training_episode_count": len(episodes),
        "steps": training_config.steps,
        "rollout_steps": training_config.rollout_steps,
        "seed": training_config.seed,
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "mean_last_100_loss": float(np.mean(tail)),
    }


def _chamfer_m(prediction: np.ndarray, target: np.ndarray) -> float:
    distance = np.linalg.norm(
        prediction[:, None] - target[None], axis=-1
    )
    return float(0.5 * (np.mean(np.min(distance, axis=1)) + np.mean(np.min(distance, axis=0))))


def _trajectory_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    _require(prediction.shape == target.shape, "trajectory shapes differ")
    _require(len(prediction) >= 4, "trajectory is too short")
    frame_track = np.mean(np.linalg.norm(prediction - target, axis=-1), axis=1)
    frame_chamfer = np.asarray(
        [_chamfer_m(predicted, observed) for predicted, observed in zip(prediction, target, strict=True)]
    )
    late_start = max(1, 2 * len(prediction) // 3)
    return {
        "future_track_error_m": float(np.mean(frame_track[1:])),
        "future_chamfer_m": float(np.mean(frame_chamfer[1:])),
        "late_track_error_m": float(np.mean(frame_track[late_start:])),
        "late_chamfer_m": float(np.mean(frame_chamfer[late_start:])),
    }


@torch.no_grad()
def rollout_residual_model(
    model: EquivariantBayesianResidual,
    episode: Deform360ResidualSourceEpisode,
    *,
    physics_velocity_retention: float,
    device: str | torch.device,
    utility_threshold: float = 0.0,
    maximum_variance_m2ps2: float = float("inf"),
) -> dict[str, np.ndarray]:
    """Roll out without reading any post-initial object state as model input."""

    torch_device = torch.device(device)
    data = _torch_episode(episode, torch_device)
    dt = float(episode.frame_interval_s)
    current = data.positions[0:1]
    velocity = torch.zeros_like(current)
    static_reliability = data.reliability[0:1]
    prediction_frames = [current[0].detach().cpu().numpy()]
    physics_frames = [data.physics_positions[0].detach().cpu().numpy()]
    persistence_frames = [prediction_frames[0].copy()]
    inertial_frames = [prediction_frames[0].copy()]
    inertial_current = current.clone()
    inertial_velocity = velocity.clone()
    variance_frames = [np.zeros(current.shape[1], dtype=np.float32)]
    accepted_frames = [np.zeros(current.shape[1], dtype=bool)]
    accumulated_variance = torch.zeros(current.shape[:2], device=torch_device)
    for frame_index in range(len(data.positions) - 1):
        physics_delta = (
            data.physics_positions[frame_index + 1 : frame_index + 2]
            - data.physics_positions[frame_index : frame_index + 1]
        )
        physics_velocity = physics_delta / dt
        physics_position = current + physics_delta
        contact = _contact_probability_torch(
            current,
            data.controllers[frame_index : frame_index + 1],
            data.closure[frame_index : frame_index + 1],
            relative_to_nearest=(
                episode.controller_geometry == "end_effector_origins"
            ),
        )
        distribution = model(
            positions_m=current,
            velocities_mps=velocity,
            physics_positions_m=physics_position,
            physics_velocities_mps=physics_velocity,
            controller_positions_m=data.controllers[frame_index : frame_index + 1],
            controller_velocities_mps=data.controller_velocities[
                frame_index : frame_index + 1
            ],
            contact_probabilities=contact,
            prior_reliability=static_reliability,
            edge_index=data.edges,
        )
        accepted = (
            (distribution.utility_probability >= utility_threshold)
            & (distribution.aleatoric_variance_m2ps2 <= maximum_variance_m2ps2)
        )
        residual = torch.where(
            accepted[..., None],
            distribution.mean_mps,
            torch.zeros_like(distribution.mean_mps),
        )
        if bool(torch.any(accepted)):
            velocity = physics_velocity + residual
            current = physics_position + residual * dt
        else:
            current = data.physics_positions[frame_index + 1 : frame_index + 2]
            velocity = physics_velocity
        accumulated_variance = accumulated_variance + (
            distribution.aleatoric_variance_m2ps2 * dt**2
        )
        prediction_frames.append(current[0].detach().cpu().numpy())
        variance_frames.append(accumulated_variance[0].detach().cpu().numpy())
        accepted_frames.append(accepted[0].detach().cpu().numpy())
        physics_frames.append(
            data.physics_positions[frame_index + 1].detach().cpu().numpy()
        )
        persistence_frames.append(persistence_frames[0].copy())
        inertial_velocity = physics_velocity_retention * inertial_velocity
        inertial_current = inertial_current + inertial_velocity * dt
        inertial_frames.append(inertial_current[0].detach().cpu().numpy())
    return {
        "residual_m": np.stack(prediction_frames),
        "physics_m": np.stack(physics_frames),
        "persistence_m": np.stack(persistence_frames),
        "inertial_m": np.stack(inertial_frames),
        "position_variance_m2": np.stack(variance_frames),
        "accepted": np.stack(accepted_frames),
    }


def _select_source_utility_threshold(
    model: EquivariantBayesianResidual,
    episodes: Sequence[Deform360ResidualSourceEpisode],
    *,
    physics_velocity_retention: float,
    device: str | torch.device,
    forced_fallback_arm: str | None = None,
    thresholds: Sequence[float] = (0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.1),
) -> dict[str, Any]:
    """Select abstention on inner episodes, including exact fallback at 1.1."""

    _require(bool(episodes), "utility calibration panel is empty")
    episode_rollouts = {
        episode.episode_key: rollout_residual_model(
            model,
            episode,
            physics_velocity_retention=physics_velocity_retention,
            device=device,
        )
        for episode in episodes
    }
    baseline_scores: dict[str, list[float]] = {"persistence": [], "physics": []}
    for episode in episodes:
        rollouts = episode_rollouts[episode.episode_key]
        target = episode.positions_m.astype(np.float64)
        persistence = _trajectory_metrics(
            rollouts["persistence_m"].astype(np.float64), target
        )
        for arm, key in (("persistence", "persistence_m"), ("physics", "physics_m")):
            metrics = _trajectory_metrics(rollouts[key].astype(np.float64), target)
            baseline_scores[arm].append(
                0.5
                * (
                    metrics["future_track_error_m"]
                    / persistence["future_track_error_m"]
                    + metrics["future_chamfer_m"]
                    / persistence["future_chamfer_m"]
                )
            )
    if forced_fallback_arm is None:
        fallback_arm = min(
            baseline_scores,
            key=lambda arm: float(np.mean(baseline_scores[arm])),
        )
        fallback_selection_kind = "selected on inner source episodes"
    else:
        _require(
            forced_fallback_arm in baseline_scores,
            "forced residual fallback arm is unsupported",
        )
        fallback_arm = forced_fallback_arm
        fallback_selection_kind = "inherited from episode-level trust policy"
    table = []
    for threshold in thresholds:
        rows = []
        for episode in episodes:
            rollouts = rollout_residual_model(
                model,
                episode,
                physics_velocity_retention=physics_velocity_retention,
                device=device,
                utility_threshold=float(threshold),
            )
            target = episode.positions_m.astype(np.float64)
            baseline = _trajectory_metrics(
                rollouts["physics_m"].astype(np.float64), target
            )
            candidate = _trajectory_metrics(
                rollouts["residual_m"].astype(np.float64), target
            )
            track_ratio = (
                candidate["future_track_error_m"]
                / baseline["future_track_error_m"]
            )
            chamfer_ratio = (
                candidate["future_chamfer_m"] / baseline["future_chamfer_m"]
            )
            rows.append(
                {
                    "episode_key": episode.episode_key,
                    "track_ratio": float(track_ratio),
                    "chamfer_ratio": float(chamfer_ratio),
                    "accepted_fraction": float(np.mean(rollouts["accepted"][1:])),
                }
            )
        maximum_degradation = max(
            max(row["track_ratio"], row["chamfer_ratio"]) - 1.0 for row in rows
        )
        table.append(
            {
                "utility_threshold": float(threshold),
                "combined_relative_score": float(
                    np.mean(
                        [
                            0.5 * (row["track_ratio"] + row["chamfer_ratio"])
                            for row in rows
                        ]
                    )
                ),
                "maximum_episode_degradation_fraction": float(maximum_degradation),
                "maximum_degradation_gate_passed": bool(maximum_degradation <= 0.10),
                "mean_accepted_fraction": float(
                    np.mean([row["accepted_fraction"] for row in rows])
                ),
                "episodes": rows,
            }
        )
    admissible = [row for row in table if row["maximum_degradation_gate_passed"]]
    _require(bool(admissible), "exact fallback did not pass the utility gate")
    selected = min(
        admissible,
        key=lambda row: (
            row["combined_relative_score"],
            -row["utility_threshold"],
        ),
    )
    return {
        "selection_kind": "one reserved episode per outer-training object",
        "fallback_selection_kind": fallback_selection_kind,
        "selected_fallback_arm": fallback_arm,
        "threshold_reference_arm": "physics",
        "fallback_relative_score_vs_persistence": float(
            np.mean(baseline_scores[fallback_arm])
        ),
        "selected_utility_threshold": selected["utility_threshold"],
        "selected_combined_relative_score": selected["combined_relative_score"],
        "threshold_table": table,
    }


def run_leave_one_object_out_smoke(
    episodes: Sequence[Deform360ResidualSourceEpisode],
    *,
    held_object_id: str,
    model_config: BayesianResidualModelConfig,
    training_config: ResidualTrainingConfig,
    device: str | torch.device,
) -> dict[str, Any]:
    outer_training = [
        episode for episode in episodes if episode.object_id != held_object_id
    ]
    held = [episode for episode in episodes if episode.object_id == held_object_id]
    _require(bool(outer_training) and bool(held), "leave-one-object-out split is empty")
    calibration_keys = {
        max(
            episode.episode_id
            for episode in outer_training
            if episode.object_id == object_id
        )
        for object_id in {episode.object_id for episode in outer_training}
    }
    calibration = [
        episode
        for episode in outer_training
        if episode.episode_id in calibration_keys
        and episode.episode_id
        == max(
            candidate.episode_id
            for candidate in outer_training
            if candidate.object_id == episode.object_id
        )
    ]
    calibration_episode_keys = {episode.episode_key for episode in calibration}
    training = [
        episode
        for episode in outer_training
        if episode.episode_key not in calibration_episode_keys
    ]
    _require(
        len(calibration) == len({episode.object_id for episode in outer_training}),
        "inner utility calibration did not reserve one episode per object",
    )
    model, training_summary = train_residual_model(
        training,
        model_config=model_config,
        training_config=training_config,
        device=device,
    )
    physics_prior_kinds = {episode.physics_prior_kind for episode in episodes}
    episode_trust_already_applied = physics_prior_kinds == {
        "trusted_sealed_graph_action_support"
    }
    utility_selection = _select_source_utility_threshold(
        model,
        calibration,
        physics_velocity_retention=training_config.physics_velocity_retention,
        device=device,
        forced_fallback_arm=("physics" if episode_trust_already_applied else None),
    )
    selected_threshold = float(utility_selection["selected_utility_threshold"])
    fallback_arm = str(utility_selection["selected_fallback_arm"])
    physics_branch_admitted = fallback_arm == "physics"
    per_episode = []
    for episode in held:
        deterministic_rollouts = rollout_residual_model(
            model,
            episode,
            physics_velocity_retention=training_config.physics_velocity_retention,
            device=device,
        )
        gated_rollouts = rollout_residual_model(
            model,
            episode,
            physics_velocity_retention=training_config.physics_velocity_retention,
            device=device,
            utility_threshold=selected_threshold,
        )
        if not physics_branch_admitted:
            gated_rollouts["residual_m"] = gated_rollouts["persistence_m"].copy()
            gated_rollouts["accepted"] = np.zeros_like(
                gated_rollouts["accepted"], dtype=bool
            )
        target = episode.positions_m.astype(np.float64)
        arms = {
            name: _trajectory_metrics(values.astype(np.float64), target)
            for name, values in (
                ("persistence", deterministic_rollouts["persistence_m"]),
                ("inertial", deterministic_rollouts["inertial_m"]),
                ("physics", deterministic_rollouts["physics_m"]),
                ("deterministic_residual", deterministic_rollouts["residual_m"]),
                ("gated_residual", gated_rollouts["residual_m"]),
            )
        }
        per_episode.append(
            {
                "episode_key": episode.episode_key,
                "arms": arms,
                "deterministic_accepted_fraction": float(
                    np.mean(deterministic_rollouts["accepted"][1:])
                ),
                "gated_accepted_fraction": float(
                    np.mean(gated_rollouts["accepted"][1:])
                ),
            }
        )
    aggregate: dict[str, dict[str, float]] = {}
    for arm in (
        "persistence",
        "inertial",
        "physics",
        "deterministic_residual",
        "gated_residual",
    ):
        aggregate[arm] = {
            metric: float(np.mean([row["arms"][arm][metric] for row in per_episode]))
            for metric in (
                "future_track_error_m",
                "future_chamfer_m",
                "late_track_error_m",
                "late_chamfer_m",
            )
        }
    improvements = {
        arm: {
            metric: float(
                1.0 - aggregate[arm][metric] / aggregate[fallback_arm][metric]
            )
            for metric in aggregate[fallback_arm]
        }
        for arm in ("deterministic_residual", "gated_residual")
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360BayesianResidualSourceSmoke",
        "protocol_id": "deform360-bayesian-residual-source-v1",
        "held_object_id": held_object_id,
        "training_object_ids": sorted({episode.object_id for episode in training}),
        "training_episode_keys": sorted(episode.episode_key for episode in training),
        "utility_calibration_episode_keys": sorted(calibration_episode_keys),
        "training": training_summary,
        "utility_selection": utility_selection,
        "model": {
            "hidden_dim": model_config.hidden_dim,
            "message_steps": model_config.message_steps,
            "maximum_residual_speed_mps": model_config.maximum_residual_speed_mps,
            "controller_geometry": sorted(
                {episode.controller_geometry for episode in episodes}
            ),
            "physics_prior_kind": sorted(
                physics_prior_kinds
            ),
            "episode_trust_already_applied": episode_trust_already_applied,
            "physics_response_scale_by_episode": {
                episode.episode_key: episode.physics_response_scale
                for episode in episodes
            },
            "physics_reference_response_scale": sorted(
                {episode.physics_reference_response_scale for episode in episodes}
            ),
        },
        "fallback_arm": fallback_arm,
        "physics_branch_admitted": physics_branch_admitted,
        "aggregate": aggregate,
        "residual_improvement_fraction_vs_fallback": improvements,
        "episodes": per_episode,
        "information_boundary": {
            "opened_source_episode_count": 27,
            "penguin_held_media_or_outcomes_read": False,
            "pokeflex_target_read": False,
            "state_of_the_art_claim_supported": False,
        },
    }
    payload["result_sha256"] = _result_sha256(payload)
    return payload


def write_source_smoke_result(path: str | Path, payload: Mapping[str, Any]) -> None:
    _require(payload.get("result_sha256") == _result_sha256(payload), "result checksum differs")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_source_protocol_before_experiment(path: str | Path) -> dict[str, Any]:
    """Expose the lock check to the CLI without broad package imports."""

    return load_bayesian_residual_config(path)


__all__ = [
    "ResidualTrainingConfig",
    "load_cross_fitted_trust_scales",
    "load_source_residual_panel",
    "rollout_residual_model",
    "run_leave_one_object_out_smoke",
    "train_residual_model",
    "validate_source_protocol_before_experiment",
    "write_source_smoke_result",
]
