"""Source-only temporal residual experiment for reusable Deform360 twins."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .deform360_bayesian_residual import (
    TemporalBayesianResidualModelConfig,
    TemporalEquivariantBayesianResidual,
    TemporalResidualState,
    clustered_student_t_nll,
)
from .deform360_bayesian_residual_data import Deform360ResidualSourceEpisode
from .deform360_bayesian_residual_experiment import (
    _causal_velocity,
    _contact_probability_torch,
    _result_sha256,
    _torch_episode,
    _trajectory_metrics,
    write_source_smoke_result,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class TemporalResidualTrainingConfig:
    steps: int = 5000
    context_steps: int = 8
    rollout_steps: int = 8
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-6
    physics_velocity_retention: float = 0.85
    position_loss_scale: float = 2500.0
    utility_loss_scale: float = 0.1
    gradient_norm: float = 1.0
    seed: int = 0


def _temporal_model_step(
    model: TemporalEquivariantBayesianResidual,
    episode: Any,
    *,
    current: torch.Tensor,
    velocity: torch.Tensor,
    frame_index: int,
    static_reliability: torch.Tensor,
    state: TemporalResidualState | None,
) -> tuple[Any, TemporalResidualState, torch.Tensor, torch.Tensor]:
    dt = float(episode.source.frame_interval_s)
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
    prediction, next_state = model(
        positions_m=current,
        velocities_mps=velocity,
        physics_positions_m=physics_position,
        physics_velocities_mps=physics_velocity,
        controller_positions_m=episode.controllers[frame_index : frame_index + 1],
        controller_velocities_mps=episode.controller_velocities[
            frame_index : frame_index + 1
        ],
        contact_probabilities=contact,
        prior_reliability=static_reliability,
        edge_index=episode.edges,
        temporal_state=state,
    )
    return prediction, next_state, physics_velocity, physics_position


def train_temporal_residual_model(
    episodes: Sequence[Deform360ResidualSourceEpisode],
    *,
    model_config: TemporalBayesianResidualModelConfig,
    training_config: TemporalResidualTrainingConfig,
    device: str | torch.device,
) -> tuple[TemporalEquivariantBayesianResidual, dict[str, Any]]:
    """Train with a causal context window and open-loop multi-step loss."""

    _require(bool(episodes), "temporal residual training panel is empty")
    _require(training_config.steps >= 1, "training steps must be positive")
    _require(training_config.context_steps >= 1, "context steps must be positive")
    _require(training_config.rollout_steps >= 2, "rollout steps must exceed one")
    torch_device = torch.device(device)
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(training_config.seed)
    model = TemporalEquivariantBayesianResidual(model_config).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    torch_episodes = [_torch_episode(episode, torch_device) for episode in episodes]
    minimum_length = training_config.context_steps + training_config.rollout_steps + 1
    _require(
        all(len(episode.positions) >= minimum_length for episode in torch_episodes),
        "an episode is too short for temporal training",
    )
    generator = random.Random(training_config.seed)
    loss_history: list[float] = []
    for _ in range(training_config.steps):
        episode = torch_episodes[generator.randrange(len(torch_episodes))]
        maximum_start = len(episode.positions) - training_config.rollout_steps - 1
        start = generator.randrange(training_config.context_steps, maximum_start + 1)
        context_start = start - training_config.context_steps
        dt = float(episode.source.frame_interval_s)
        static_reliability = episode.reliability[0:1]
        state: TemporalResidualState | None = None

        for frame_index in range(context_start, start):
            observed = episode.positions[frame_index : frame_index + 1]
            observed_velocity = _causal_velocity(
                episode.positions, frame_index, dt
            )[None]
            _, state, _, _ = _temporal_model_step(
                model,
                episode,
                current=observed,
                velocity=observed_velocity,
                frame_index=frame_index,
                static_reliability=static_reliability,
                state=state,
            )

        current = episode.positions[start : start + 1]
        velocity = _causal_velocity(episode.positions, start, dt)[None]
        losses = []
        for offset in range(training_config.rollout_steps):
            frame_index = start + offset
            target = episode.positions[frame_index + 1 : frame_index + 2]
            prediction, state, physics_velocity, physics_position = (
                _temporal_model_step(
                    model,
                    episode,
                    current=current,
                    velocity=velocity,
                    frame_index=frame_index,
                    static_reliability=static_reliability,
                    state=state,
                )
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
        "context_steps": training_config.context_steps,
        "rollout_steps": training_config.rollout_steps,
        "seed": training_config.seed,
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "mean_last_100_loss": float(np.mean(tail)),
    }


@torch.no_grad()
def rollout_temporal_residual_model(
    model: TemporalEquivariantBayesianResidual,
    episode: Deform360ResidualSourceEpisode,
    *,
    physics_velocity_retention: float,
    device: str | torch.device,
    utility_threshold: float = 0.0,
    maximum_variance_m2ps2: float = float("inf"),
) -> dict[str, np.ndarray]:
    """Roll out causally while retaining exact per-node physical fallback."""

    torch_device = torch.device(device)
    data = _torch_episode(episode, torch_device)
    dt = float(episode.frame_interval_s)
    current = data.positions[0:1]
    velocity = torch.zeros_like(current)
    state: TemporalResidualState | None = None
    static_reliability = data.reliability[0:1]
    initial = current[0].detach().cpu().numpy()
    prediction_frames = [initial]
    physics_frames = [data.physics_positions[0].detach().cpu().numpy()]
    persistence_frames = [initial.copy()]
    inertial_frames = [initial.copy()]
    inertial_current = current.clone()
    inertial_velocity = velocity.clone()
    variance_frames = [np.zeros(current.shape[1], dtype=np.float32)]
    accepted_frames = [np.zeros(current.shape[1], dtype=bool)]
    accumulated_variance = torch.zeros(current.shape[:2], device=torch_device)
    for frame_index in range(len(data.positions) - 1):
        distribution, state, physics_velocity, physics_position = (
            _temporal_model_step(
                model,
                data,
                current=current,
                velocity=velocity,
                frame_index=frame_index,
                static_reliability=static_reliability,
                state=state,
            )
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


def _select_utility_threshold(
    model: TemporalEquivariantBayesianResidual,
    episodes: Sequence[Deform360ResidualSourceEpisode],
    *,
    physics_velocity_retention: float,
    device: str | torch.device,
    forced_fallback_arm: str | None,
    thresholds: Sequence[float] = (
        0.0,
        0.25,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        0.99,
        1.1,
    ),
) -> dict[str, Any]:
    _require(bool(episodes), "utility calibration panel is empty")
    base_rollouts = {
        episode.episode_key: rollout_temporal_residual_model(
            model,
            episode,
            physics_velocity_retention=physics_velocity_retention,
            device=device,
        )
        for episode in episodes
    }
    baseline_scores: dict[str, list[float]] = {"persistence": [], "physics": []}
    for episode in episodes:
        rollouts = base_rollouts[episode.episode_key]
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
    fallback_arm = (
        min(baseline_scores, key=lambda arm: float(np.mean(baseline_scores[arm])))
        if forced_fallback_arm is None
        else forced_fallback_arm
    )
    _require(fallback_arm in baseline_scores, "unsupported temporal fallback arm")
    table = []
    for threshold in thresholds:
        rows = []
        for episode in episodes:
            rollouts = rollout_temporal_residual_model(
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
            rows.append(
                {
                    "episode_key": episode.episode_key,
                    "track_ratio": float(
                        candidate["future_track_error_m"]
                        / baseline["future_track_error_m"]
                    ),
                    "chamfer_ratio": float(
                        candidate["future_chamfer_m"]
                        / baseline["future_chamfer_m"]
                    ),
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
                "maximum_degradation_gate_passed": bool(
                    maximum_degradation <= 0.10
                ),
                "mean_accepted_fraction": float(
                    np.mean([row["accepted_fraction"] for row in rows])
                ),
                "episodes": rows,
            }
        )
    admissible = [row for row in table if row["maximum_degradation_gate_passed"]]
    _require(bool(admissible), "exact temporal fallback did not pass")
    selected = min(
        admissible,
        key=lambda row: (row["combined_relative_score"], -row["utility_threshold"]),
    )
    return {
        "selection_kind": "one reserved episode per outer-training object",
        "selected_fallback_arm": fallback_arm,
        "threshold_reference_arm": "physics",
        "fallback_relative_score_vs_persistence": float(
            np.mean(baseline_scores[fallback_arm])
        ),
        "selected_utility_threshold": selected["utility_threshold"],
        "selected_combined_relative_score": selected["combined_relative_score"],
        "threshold_table": table,
    }


def run_leave_one_object_out_temporal_smoke(
    episodes: Sequence[Deform360ResidualSourceEpisode],
    *,
    held_object_id: str,
    model_config: TemporalBayesianResidualModelConfig,
    training_config: TemporalResidualTrainingConfig,
    device: str | torch.device,
) -> dict[str, Any]:
    """Train, calibrate, and score one source-only held-object fold."""

    outer_training = [
        episode for episode in episodes if episode.object_id != held_object_id
    ]
    held = [episode for episode in episodes if episode.object_id == held_object_id]
    _require(bool(outer_training) and bool(held), "temporal outer split is empty")
    calibration = [
        max(
            (episode for episode in outer_training if episode.object_id == object_id),
            key=lambda episode: episode.episode_id,
        )
        for object_id in sorted({episode.object_id for episode in outer_training})
    ]
    calibration_keys = {episode.episode_key for episode in calibration}
    training = [
        episode
        for episode in outer_training
        if episode.episode_key not in calibration_keys
    ]
    model, training_summary = train_temporal_residual_model(
        training,
        model_config=model_config,
        training_config=training_config,
        device=device,
    )
    physics_prior_kinds = {episode.physics_prior_kind for episode in episodes}
    episode_trust_already_applied = physics_prior_kinds == {
        "trusted_sealed_graph_action_support"
    }
    utility_selection = _select_utility_threshold(
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
        deterministic = rollout_temporal_residual_model(
            model,
            episode,
            physics_velocity_retention=training_config.physics_velocity_retention,
            device=device,
        )
        gated = rollout_temporal_residual_model(
            model,
            episode,
            physics_velocity_retention=training_config.physics_velocity_retention,
            device=device,
            utility_threshold=selected_threshold,
        )
        if not physics_branch_admitted:
            gated["residual_m"] = gated["persistence_m"].copy()
            gated["accepted"] = np.zeros_like(gated["accepted"], dtype=bool)
        target = episode.positions_m.astype(np.float64)
        arms = {
            name: _trajectory_metrics(values.astype(np.float64), target)
            for name, values in (
                ("persistence", deterministic["persistence_m"]),
                ("inertial", deterministic["inertial_m"]),
                ("physics", deterministic["physics_m"]),
                ("deterministic_temporal", deterministic["residual_m"]),
                ("gated_temporal", gated["residual_m"]),
            )
        }
        per_episode.append(
            {
                "episode_key": episode.episode_key,
                "arms": arms,
                "deterministic_accepted_fraction": float(
                    np.mean(deterministic["accepted"][1:])
                ),
                "gated_accepted_fraction": float(np.mean(gated["accepted"][1:])),
            }
        )
    aggregate: dict[str, dict[str, float]] = {}
    for arm in (
        "persistence",
        "inertial",
        "physics",
        "deterministic_temporal",
        "gated_temporal",
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
        for arm in ("deterministic_temporal", "gated_temporal")
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360TemporalBayesianResidualSourceSmoke",
        "protocol_id": "deform360-bayesian-residual-source-v1",
        "held_object_id": held_object_id,
        "training_object_ids": sorted({episode.object_id for episode in training}),
        "training_episode_keys": sorted(episode.episode_key for episode in training),
        "utility_calibration_episode_keys": sorted(calibration_keys),
        "training": training_summary,
        "utility_selection": utility_selection,
        "model": {
            "hidden_dim": model_config.hidden_dim,
            "message_steps": model_config.message_steps,
            "temporal_hidden_dim": model_config.temporal_hidden_dim,
            "maximum_residual_speed_mps": (
                model_config.maximum_residual_speed_mps
            ),
            "maximum_temporal_correction_mps": (
                model_config.maximum_temporal_correction_mps
            ),
            "controller_geometry": sorted(
                {episode.controller_geometry for episode in episodes}
            ),
            "physics_prior_kind": sorted(physics_prior_kinds),
            "episode_trust_already_applied": episode_trust_already_applied,
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


__all__ = [
    "TemporalResidualTrainingConfig",
    "rollout_temporal_residual_model",
    "run_leave_one_object_out_temporal_smoke",
    "train_temporal_residual_model",
    "write_source_smoke_result",
]
