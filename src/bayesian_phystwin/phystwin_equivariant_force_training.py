"""Source-only meta-training and prefix latent adaptation for force models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .phystwin_equivariant_force import (
    EQUIVARIANT_FORCE_CONTRACT,
    EquivariantForceConfig,
    build_equivariant_force_model,
)
from .phystwin_equivariant_force_artifact import (
    EquivariantForceArtifact,
    write_equivariant_force_artifact,
)
from .phystwin_equivariant_force_data import (
    EquivariantForceEpisode,
    validate_force_episode_model_compatibility,
)


@dataclass(frozen=True)
class EquivariantForceTrainingConfig:
    """Frozen optimization and source-competence settings."""

    training_steps: int = 2000
    adaptation_steps: int = 250
    learning_rate: float = 3.0e-4
    adaptation_learning_rate: float = 1.0e-2
    weight_decay: float = 1.0e-5
    latent_regularization: float = 1.0e-3
    adaptation_regularization: float = 1.0e-2
    gradient_clip: float = 1.0
    huber_delta_normalized: float = 0.05
    seeds: tuple[int, ...] = (17, 43, 101)
    minimum_force_target_improvement: float = 0.10
    minimum_both_win_folds: int = 2
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        if self.training_steps < 1 or self.adaptation_steps < 1:
            raise ValueError("training and adaptation steps must be positive")
        positive = (
            self.learning_rate,
            self.adaptation_learning_rate,
            self.gradient_clip,
            self.huber_delta_normalized,
        )
        if any(value <= 0.0 or not np.isfinite(value) for value in positive):
            raise ValueError("optimizer and robust-loss scales must be positive")
        if self.weight_decay < 0.0 or self.latent_regularization < 0.0:
            raise ValueError("regularization must be nonnegative")
        if self.adaptation_regularization < 0.0:
            raise ValueError("adaptation_regularization must be nonnegative")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("training seeds must be nonempty and unique")
        if not 0.0 <= self.minimum_force_target_improvement < 1.0:
            raise ValueError("minimum improvement must lie in [0,1)")
        if self.minimum_both_win_folds < 1:
            raise ValueError("minimum_both_win_folds must be positive")


def _episode_frames(
    episode: EquivariantForceEpisode,
    *,
    start: int,
    stop: int,
) -> np.ndarray:
    if not 0 <= start < stop <= len(episode.positions_m):
        raise ValueError("training interval lies outside the force episode")
    supported = np.flatnonzero(
        np.sum(episode.force_target_weight[start:stop], axis=1) > 0.0
    )
    if not len(supported):
        raise ValueError(f"{episode.case_id} has no supported force targets")
    return supported + start


def _episode_tensors(
    episode: EquivariantForceEpisode,
    torch: Any,
    *,
    device: str,
) -> dict[str, Any]:
    """Copy one immutable episode to its training device exactly once."""

    float_names = (
        "positions_m",
        "velocities_mps",
        "rest_positions_m",
        "rest_lengths_m",
        "control_displacement_m",
        "control_velocity_mps",
        "action_support",
        "external_support",
        "gravity_mps2",
        "action_activity",
        "regime_probabilities",
        "force_targets_sim",
        "force_target_weight",
    )
    tensors = {
        name: torch.tensor(
            getattr(episode, name),
            dtype=torch.float32,
            device=device,
        )
        for name in float_names
    }
    tensors["object_edges"] = torch.tensor(
        episode.object_edges,
        dtype=torch.long,
        device=device,
    )
    tensors["force_scale_sim"] = torch.tensor(
        episode.force_scale_sim,
        dtype=torch.float32,
        device=device,
    )
    return tensors


def _frame_prediction(
    model: Any,
    torch: Any,
    episode: EquivariantForceEpisode,
    frame: int,
    latent: Any,
    *,
    device: str,
    tensors: Mapping[str, Any] | None = None,
):
    values = tensors or _episode_tensors(episode, torch, device=device)
    return model(
        positions_m=values["positions_m"][frame],
        velocities_mps=values["velocities_mps"][frame],
        rest_positions_m=values["rest_positions_m"],
        edges=values["object_edges"],
        rest_lengths_m=values["rest_lengths_m"],
        control_displacement_m=values["control_displacement_m"][frame],
        control_velocity_mps=values["control_velocity_mps"][frame],
        action_support=values["action_support"][frame],
        external_support=values["external_support"][frame],
        gravity_mps2=values["gravity_mps2"],
        force_scale_sim=values["force_scale_sim"],
        action_activity=values["action_activity"][frame],
        regime_probabilities=values["regime_probabilities"][frame],
        latent=latent,
        admission_weight=1.0,
    )


def _frame_loss(
    model: Any,
    torch: Any,
    episode: EquivariantForceEpisode,
    frame: int,
    latent: Any,
    *,
    config: EquivariantForceTrainingConfig,
    tensors: Mapping[str, Any] | None = None,
):
    values = tensors or _episode_tensors(
        episode,
        torch,
        device=config.device,
    )
    prediction = _frame_prediction(
        model,
        torch,
        episode,
        frame,
        latent,
        device=config.device,
        tensors=values,
    )
    scale = values["force_scale_sim"]
    target = values["force_targets_sim"][frame] / scale
    prediction = prediction / scale
    weight = values["force_target_weight"][frame]
    element = torch.nn.functional.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
        beta=config.huber_delta_normalized,
    )
    denominator = torch.clamp(3.0 * torch.sum(weight), min=1.0)
    return torch.sum(element * weight[:, None]) / denominator


def fit_shared_equivariant_force_model(
    episodes: Sequence[EquivariantForceEpisode],
    torch: Any,
    *,
    model_config: EquivariantForceConfig,
    training_config: EquivariantForceTrainingConfig,
    seed: int,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Fit global weights and per-source nuisance latents at a terminal step."""

    if not episodes:
        raise ValueError("at least one source episode is required")
    for episode in episodes:
        validate_force_episode_model_compatibility(episode, model_config)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_equivariant_force_model(torch, model_config).to(
        training_config.device
    )
    source_latents = torch.nn.Parameter(
        torch.zeros(
            (len(episodes), model_config.latent_dim),
            dtype=torch.float32,
            device=training_config.device,
        )
    )
    optimizer = torch.optim.AdamW(
        [*model.parameters(), source_latents],
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    frame_sets = [
        _episode_frames(
            episode, start=0, stop=episode.validation_end_frame
        )
        for episode in episodes
    ]
    tensor_sets = [
        _episode_tensors(episode, torch, device=training_config.device)
        for episode in episodes
    ]
    trace: list[dict[str, float | int]] = []
    accepted_steps = 0
    model.train()
    for step in range(training_config.training_steps):
        episode_index = step % len(episodes)
        frames = frame_sets[episode_index]
        frame = int(frames[rng.integers(0, len(frames))])
        optimizer.zero_grad(set_to_none=True)
        loss = _frame_loss(
            model,
            torch,
            episodes[episode_index],
            frame,
            source_latents[episode_index],
            config=training_config,
            tensors=tensor_sets[episode_index],
        )
        loss = loss + training_config.latent_regularization * torch.mean(
            torch.square(source_latents[episode_index])
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite equivariant-force training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            [*model.parameters(), source_latents],
            training_config.gradient_clip,
        )
        if not bool(torch.isfinite(norm).item()):
            raise RuntimeError("non-finite equivariant-force gradient")
        optimizer.step()
        accepted_steps += 1
        if step == 0 or (step + 1) % max(
            training_config.training_steps // 10, 1
        ) == 0:
            trace.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm": float(
                        torch.as_tensor(norm).detach().cpu()
                    ),
                }
            )
    return (
        model,
        source_latents.detach().cpu().numpy(),
        {
            "seed": seed,
            "accepted_steps": accepted_steps,
            "terminal_step_selected": True,
            "trace": trace,
        },
    )


def adapt_equivariant_force_latent(
    model: Any,
    episode: EquivariantForceEpisode,
    torch: Any,
    *,
    model_config: EquivariantForceConfig,
    training_config: EquivariantForceTrainingConfig,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Adapt only one held-out latent using its permitted prefix."""

    validate_force_episode_model_compatibility(episode, model_config)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    latent = torch.nn.Parameter(
        torch.zeros(
            model_config.latent_dim,
            dtype=torch.float32,
            device=training_config.device,
        )
    )
    optimizer = torch.optim.Adam(
        [latent], lr=training_config.adaptation_learning_rate
    )
    frames = _episode_frames(episode, start=0, stop=episode.fit_end_frame)
    tensors = _episode_tensors(
        episode,
        torch,
        device=training_config.device,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trace: list[dict[str, float | int]] = []
    try:
        for step in range(training_config.adaptation_steps):
            frame = int(frames[rng.integers(0, len(frames))])
            optimizer.zero_grad(set_to_none=True)
            loss = _frame_loss(
                model,
                torch,
                episode,
                frame,
                latent,
                config=training_config,
                tensors=tensors,
            )
            loss = loss + training_config.adaptation_regularization * torch.mean(
                torch.square(latent)
            )
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("non-finite held-out latent loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [latent], training_config.gradient_clip
            )
            optimizer.step()
            if step == 0 or (step + 1) % max(
                training_config.adaptation_steps // 5, 1
            ) == 0:
                trace.append(
                    {
                        "step": step + 1,
                        "loss": float(loss.detach().cpu()),
                    }
                )
    finally:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    return latent.detach().cpu().numpy(), {
        "seed": seed,
        "terminal_step_selected": True,
        "prefix_stop": episode.fit_end_frame,
        "future_frames_used": False,
        "trace": trace,
    }


def force_target_metrics(
    model: Any,
    episode: EquivariantForceEpisode,
    latent: np.ndarray,
    torch: Any,
    *,
    start_frame: int,
    stop_frame: int,
    device: str,
) -> dict[str, float | int]:
    """Score held-out force targets as a competence diagnostic, not a paper metric."""

    frames = _episode_frames(episode, start=start_frame, stop=stop_frame)
    squared_candidate = 0.0
    squared_zero = 0.0
    weight_total = 0.0
    model.eval()
    latent_tensor = torch.as_tensor(
        latent, dtype=torch.float32, device=device
    )
    tensors = _episode_tensors(episode, torch, device=device)
    with torch.no_grad():
        for frame in frames:
            prediction = _frame_prediction(
                model,
                torch,
                episode,
                int(frame),
                latent_tensor,
                device=device,
                tensors=tensors,
            ).detach().cpu().numpy()
            scale = episode.force_scale_sim
            prediction = prediction / scale
            target = episode.force_targets_sim[frame] / scale
            weight = episode.force_target_weight[frame]
            squared_candidate += float(
                np.sum(weight[:, None] * np.square(prediction - target))
            )
            squared_zero += float(
                np.sum(weight[:, None] * np.square(target))
            )
            weight_total += float(3.0 * np.sum(weight))
    candidate_rmse = float(np.sqrt(squared_candidate / max(weight_total, 1.0)))
    zero_rmse = float(np.sqrt(squared_zero / max(weight_total, 1.0)))
    return {
        "candidate_normalized_force_rmse": candidate_rmse,
        "zero_normalized_force_rmse": zero_rmse,
        "relative_rmse": candidate_rmse / max(zero_rmse, 1.0e-12),
        "improvement": 1.0 - candidate_rmse / max(zero_rmse, 1.0e-12),
        "frame_count": int(len(frames)),
    }


def _validate_folds(
    episodes: Sequence[EquivariantForceEpisode],
    folds: Sequence[Mapping[str, Any]],
) -> list[tuple[str, list[str]]]:
    cases = {episode.case_id for episode in episodes}
    if len(cases) != len(episodes):
        raise ValueError("force episode case IDs must be unique")
    parsed: list[tuple[str, list[str]]] = []
    held: list[str] = []
    for fold in folds:
        name = str(fold["name"])
        values = [str(value) for value in fold["held_out_cases"]]
        if not name or not values or len(set(values)) != len(values):
            raise ValueError("fold names and held-out cases must be unique")
        parsed.append((name, values))
        held.extend(values)
    if len(held) != len(set(held)) or set(held) != cases:
        raise ValueError("folds must provide disjoint complete case coverage")
    return parsed


def crossfit_equivariant_force_competence(
    episodes: Sequence[EquivariantForceEpisode],
    folds: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    torch: Any,
    *,
    model_config: EquivariantForceConfig,
    training_config: EquivariantForceTrainingConfig,
) -> dict[str, Any]:
    """Cross-fit source force targets before any official-Warp promotion gate."""

    parsed_folds = _validate_folds(episodes, folds)
    by_case = {episode.case_id: episode for episode in episodes}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fold_results = []
    for fold_name, held_cases in parsed_folds:
        source = [
            episode for episode in episodes if episode.case_id not in held_cases
        ]
        seed_results = []
        for seed in training_config.seeds:
            model, _, training = fit_shared_equivariant_force_model(
                source,
                torch,
                model_config=model_config,
                training_config=training_config,
                seed=seed,
            )
            source_checksums = {
                f"source_episode_{episode.case_id}": episode.artifact_id
                for episode in source
            }
            artifact = EquivariantForceArtifact.from_model(
                model,
                config=model_config,
                source_checksums=source_checksums,
                information_boundary={
                    "target_future_used_for_fit_or_selection": False,
                    "exact_zero_force_fallback": True,
                    "force_location": "inside_official_warp",
                    "force_unit_contract": (
                        "warp_simulator_generalized_force_not_newtons"
                    ),
                    "complete_source_outcomes_supervise_shared_weights": True,
                    "heldout_prefix_adapts_latent_only": True,
                },
                training_summary={
                    **training,
                    "fold": fold_name,
                    "held_out_cases": held_cases,
                },
                admission_policy={
                    "force_target_competence_is_not_the_promotion_gate": True,
                    "official_warp_metrics_required_for_promotion": True,
                    "fallback": "unchanged_bayesian_phystwin",
                },
            )
            artifact_record = write_equivariant_force_artifact(
                output / fold_name / f"seed_{seed}" / "model",
                artifact,
            )
            held_results = []
            for case in held_cases:
                latent, adaptation = adapt_equivariant_force_latent(
                    model,
                    by_case[case],
                    torch,
                    model_config=model_config,
                    training_config=training_config,
                    seed=seed,
                )
                metrics = force_target_metrics(
                    model,
                    by_case[case],
                    latent,
                    torch,
                    start_frame=by_case[case].fit_end_frame,
                    stop_frame=by_case[case].validation_end_frame,
                    device=training_config.device,
                )
                latent_path = output / fold_name / f"seed_{seed}" / f"{case}.npz"
                latent_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(latent_path, latent=latent)
                held_results.append(
                    {
                        "case": case,
                        "adaptation": adaptation,
                        "force_target_metrics": metrics,
                        "latent_path": str(latent_path.resolve()),
                        "latent_sha256": _sha256(latent_path),
                    }
                )
            seed_results.append(
                {
                    "seed": seed,
                    "model_artifact": artifact_record,
                    "held_out": held_results,
                }
            )
        fold_results.append(
            {
                "fold": fold_name,
                "held_out_cases": held_cases,
                "seeds": seed_results,
            }
        )

    case_metrics: dict[str, list[float]] = {}
    for fold in fold_results:
        for seed in fold["seeds"]:
            for held in seed["held_out"]:
                case_metrics.setdefault(held["case"], []).append(
                    held["force_target_metrics"]["improvement"]
                )
    case_mean = {
        case: float(np.mean(values)) for case, values in sorted(case_metrics.items())
    }
    fold_wins = []
    for fold in fold_results:
        values = [case_mean[case] for case in fold["held_out_cases"]]
        fold_wins.append(
            bool(
                values
                and float(np.mean(values))
                >= training_config.minimum_force_target_improvement
            )
        )
    competence_passed = (
        sum(fold_wins) >= training_config.minimum_both_win_folds
        and float(np.mean(list(case_mean.values())))
        >= training_config.minimum_force_target_improvement
    )
    summary = {
        "schema_version": 2,
        "contract": EQUIVARIANT_FORCE_CONTRACT,
        "model_config": model_config.to_dict(),
        "training_config": asdict(training_config),
        "folds": fold_results,
        "case_mean_force_target_improvement": case_mean,
        "fold_competence_wins": fold_wins,
        "force_target_competence_passed": competence_passed,
        "official_warp_promotion_authorized": False,
        "claim_boundary": (
            "Source inverse-dynamics competence only. Passing does not establish "
            "trajectory improvement, calibration, or state of the art."
        ),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary["summary_sha256"] = _sha256(summary_path)
    return summary


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
