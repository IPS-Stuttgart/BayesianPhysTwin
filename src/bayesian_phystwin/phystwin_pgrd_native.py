"""Native temporal-head training for PGRD features on PhysTwin prefixes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .phystwin_official_evaluation import evaluate_official_phystwin_interval
from .phystwin_pgrd_adapter import (
    MetricNormalizer,
    OfficialPGRDResidualPredictor,
    _cap_vectors,
    _inverse_distance_map,
    _metric_ratios,
    _relative_score,
    compose_dense_endpoint_with_sampled_dynamics,
    deterministic_farthest_point_sample,
    rollout_pgrd_correction,
)
from .phystwin_residual_dynamics import (
    _lift_map,
    _lift_residual,
    _load_pickle,
    _sha256,
    _target_validity,
    _temporally_fill,
)


@dataclass(frozen=True)
class NativePGRDTrainingConfig:
    """Frozen low-capacity adaptation and validation settings."""

    number_of_points: int = 512
    normalized_extent: float = 0.5
    yaw_degrees: float = 180.0
    history_length: int = 2
    temporal_window: int = 5
    model_frame_stride: int = 3
    simulation_dt: float = 0.1
    interpolation_neighbors: int = 4
    maximum_residual_m: float = 0.01
    residual_scale: float = 0.2
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    l2_starting_point_weight: float = 1e-3
    huber_delta: float = 0.01
    optimization_chunk_frames: int = 4
    random_seed: int = 20260718
    minimum_balanced_improvement: float = 0.01
    minimum_both_win_count: int = 2
    maximum_metric_ratio: float = 1.02


@dataclass(frozen=True)
class PGRDFeatureSequence:
    """One cadence-aligned teacher-forced sequence for temporal-head training."""

    case: str
    target_frames: np.ndarray
    sample_indices: np.ndarray
    spatial_features: np.ndarray
    target_residual_velocity: np.ndarray
    valid: np.ndarray


class PGRDSpatialFeaturePredictor(Protocol):
    """Feature-only surface needed by the causal source extractor."""

    def spatial_features(
        self,
        x: np.ndarray,
        v: np.ndarray,
        x_history: np.ndarray,
        v_history: np.ndarray,
        x_sim: np.ndarray,
        v_sim: np.ndarray,
    ) -> np.ndarray: ...


def _validate_config(config: NativePGRDTrainingConfig) -> None:
    if config.number_of_points < 1 or config.interpolation_neighbors < 1:
        raise ValueError("point counts must be positive")
    if config.normalized_extent <= 0.0 or config.simulation_dt <= 0.0:
        raise ValueError("normalization extent and simulation_dt must be positive")
    if config.history_length < 1 or config.temporal_window < 1:
        raise ValueError("history and temporal windows must be positive")
    if config.model_frame_stride < 1 or config.epochs < 1:
        raise ValueError("stride and epochs must be positive")
    if config.maximum_residual_m <= 0.0 or config.residual_scale <= 0.0:
        raise ValueError("residual scales must be positive")
    if config.learning_rate <= 0.0 or config.huber_delta <= 0.0:
        raise ValueError("optimizer scales must be positive")
    if config.weight_decay < 0.0 or config.l2_starting_point_weight < 0.0:
        raise ValueError("regularization weights must be nonnegative")
    if config.optimization_chunk_frames < 1:
        raise ValueError("optimization_chunk_frames must be positive")
    if not 0 <= config.minimum_both_win_count <= 3:
        raise ValueError("minimum_both_win_count must lie in [0, 3]")
    if config.maximum_metric_ratio < 1.0:
        raise ValueError("maximum_metric_ratio must be at least one")


def _cadence_frames(end_frame: int, config: NativePGRDTrainingConfig) -> np.ndarray:
    minimum_target = (config.history_length + 1) * config.model_frame_stride
    last = end_frame - 1
    if last < minimum_target:
        raise ValueError("episode leaves insufficient cadence-aligned history")
    frames = list(range(last, minimum_target - 1, -config.model_frame_stride))
    return np.asarray(frames[::-1], dtype=np.int64)


def build_teacher_forced_pgrd_sequence(
    case: str,
    baseline_m: np.ndarray,
    observed_m: np.ndarray,
    valid: np.ndarray,
    sample_indices: np.ndarray,
    predictor: PGRDSpatialFeaturePredictor,
    normalizer: MetricNormalizer,
    *,
    end_frame: int,
    config: NativePGRDTrainingConfig,
) -> PGRDFeatureSequence:
    """Extract causal PGRD features and metric residual targets from a prefix."""

    _validate_config(config)
    baseline = np.asarray(baseline_m, dtype=float)
    observed = np.asarray(observed_m, dtype=float)
    validity = np.asarray(valid, dtype=bool)
    indices = np.asarray(sample_indices, dtype=np.int64)
    if observed.ndim != 3 or observed.shape[2] != 3:
        raise ValueError("observed_m must have shape (T, N, 3)")
    if baseline.ndim != 3 or baseline.shape[2] != 3:
        raise ValueError("baseline_m must have shape (T, M, 3)")
    if baseline.shape[0] < end_frame or observed.shape[0] < end_frame:
        raise ValueError("end_frame exceeds a trajectory")
    if baseline.shape[1] < observed.shape[1]:
        raise ValueError("baseline_m omits observed material points")
    if validity.shape != observed.shape[:2]:
        raise ValueError("valid must match observed frames and points")
    if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= observed.shape[1]):
        raise ValueError("sample_indices are outside observed material points")

    residual = observed - baseline[:, : observed.shape[1]]
    filled_residual = _temporally_fill(residual, validity, end_frame)
    filled_observed = baseline[:end_frame, : observed.shape[1]] + filled_residual
    target_frames = _cadence_frames(end_frame, config)
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    stride = config.model_frame_stride
    dt = config.simulation_dt

    for target_frame in target_frames:
        current_frame = int(target_frame - stride)
        history_frames = [
            current_frame - offset * stride
            for offset in reversed(range(config.history_length))
        ]
        current_m = filled_observed[current_frame, indices]
        current_velocity_m = (
            filled_observed[current_frame, indices]
            - filled_observed[current_frame - stride, indices]
        ) / dt
        position_history_m = np.moveaxis(
            filled_observed[history_frames][:, indices], 0, 1
        )
        velocity_history_m = np.stack(
            [
                (
                    filled_observed[frame, indices]
                    - filled_observed[frame - stride, indices]
                )
                / dt
                for frame in history_frames
            ],
            axis=1,
        )
        simulation_m = baseline[target_frame, indices]
        simulation_velocity_m = (
            baseline[target_frame, indices] - baseline[current_frame, indices]
        ) / dt
        features.append(
            predictor.spatial_features(
                normalizer.positions_to_model(current_m),
                normalizer.velocities_to_model(current_velocity_m),
                normalizer.positions_to_model(position_history_m),
                normalizer.velocities_to_model(velocity_history_m),
                normalizer.positions_to_model(simulation_m),
                normalizer.velocities_to_model(simulation_velocity_m),
            )
        )
        target_correction_m = _cap_vectors(
            filled_residual[target_frame, indices], config.maximum_residual_m
        )
        targets.append(
            normalizer.velocities_to_model(target_correction_m) / dt
        )
        masks.append(validity[target_frame, indices])

    return PGRDFeatureSequence(
        case=case,
        target_frames=target_frames,
        sample_indices=indices,
        spatial_features=np.stack(features),
        target_residual_velocity=np.stack(targets),
        valid=np.stack(masks),
    )


def save_feature_sequence(sequence: PGRDFeatureSequence, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        case=np.asarray(sequence.case),
        target_frames=sequence.target_frames,
        sample_indices=sequence.sample_indices,
        spatial_features=sequence.spatial_features.astype(np.float32),
        target_residual_velocity=sequence.target_residual_velocity.astype(np.float32),
        valid=sequence.valid,
    )


def load_feature_sequence(path: str | Path) -> PGRDFeatureSequence:
    with np.load(path, allow_pickle=False) as archive:
        return PGRDFeatureSequence(
            case=str(archive["case"].item()),
            target_frames=np.asarray(archive["target_frames"], dtype=np.int64),
            sample_indices=np.asarray(archive["sample_indices"], dtype=np.int64),
            spatial_features=np.asarray(archive["spatial_features"], dtype=float),
            target_residual_velocity=np.asarray(
                archive["target_residual_velocity"], dtype=float
            ),
            valid=np.asarray(archive["valid"], dtype=bool),
        )


def fit_native_temporal_head(
    predictor: OfficialPGRDResidualPredictor,
    sequences: list[PGRDFeatureSequence],
    output_checkpoint: str | Path,
    *,
    config: NativePGRDTrainingConfig,
) -> dict[str, object]:
    """Fine-tune only PGRD's temporal head with frame-balanced robust losses."""

    _validate_config(config)
    if not sequences:
        raise ValueError("at least one source sequence is required")
    torch = predictor._torch
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    temporal = predictor._temporal
    temporal.train()
    predictor._residualnet.eval()
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in temporal.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        temporal.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.random_seed)
    epoch_losses: list[float] = []

    for _epoch in range(config.epochs):
        order = rng.permutation(len(sequences))
        total_loss = 0.0
        frame_count = 0
        for sequence_index in order:
            sequence = sequences[int(sequence_index)]
            temporal.reset_window()
            optimizer.zero_grad(set_to_none=True)
            pending_losses = []
            for frame_index in range(len(sequence.target_frames)):
                feature = torch.as_tensor(
                    sequence.spatial_features[frame_index],
                    dtype=torch.float32,
                    device=predictor._device,
                )[None]
                target = torch.as_tensor(
                    sequence.target_residual_velocity[frame_index],
                    dtype=torch.float32,
                    device=predictor._device,
                )
                mask = torch.as_tensor(
                    sequence.valid[frame_index],
                    dtype=torch.bool,
                    device=predictor._device,
                )
                output = temporal(
                    feature,
                    rollout_window_size=config.temporal_window,
                    residual_scale=config.residual_scale,
                )[0]
                if bool(mask.any()):
                    bounded_target = target.clamp(
                        min=-config.residual_scale,
                        max=config.residual_scale,
                    )
                    loss = torch.nn.functional.smooth_l1_loss(
                        output[mask],
                        bounded_target[mask],
                        beta=config.huber_delta,
                        reduction="mean",
                    )
                    pending_losses.append(loss)
                    total_loss += float(loss.detach().cpu())
                    frame_count += 1
                flush = (
                    len(pending_losses) >= config.optimization_chunk_frames
                    or frame_index + 1 == len(sequence.target_frames)
                )
                if flush and pending_losses:
                    objective = torch.stack(pending_losses).mean()
                    if config.l2_starting_point_weight > 0.0:
                        squared = torch.zeros((), device=predictor._device)
                        parameter_count = 0
                        for name, parameter in temporal.named_parameters():
                            squared = squared + torch.sum(
                                torch.square(parameter - initial_parameters[name])
                            )
                            parameter_count += parameter.numel()
                        objective = objective + config.l2_starting_point_weight * (
                            squared / max(parameter_count, 1)
                        )
                    objective.backward()
                    torch.nn.utils.clip_grad_norm_(temporal.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    pending_losses = []
        epoch_losses.append(total_loss / max(frame_count, 1))

    temporal.eval()
    checkpoint = Path(output_checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "residualnet": predictor._residualnet.state_dict(),
            "transformer": temporal.export_component_state_dicts(),
            "native_training": {
                "config": asdict(config),
                "source_cases": [sequence.case for sequence in sequences],
                "epoch_losses": epoch_losses,
            },
        },
        checkpoint,
    )
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_case_count": len(sequences),
        "source_frame_count": int(
            sum(len(sequence.target_frames) for sequence in sequences)
        ),
        "initial_loss": epoch_losses[0],
        "final_loss": epoch_losses[-1],
        "epoch_losses": epoch_losses,
    }


def evaluate_native_pgrd_validation(
    final_data_path: str | Path,
    baseline_trajectory_path: str | Path,
    gt_track_path: str | Path,
    predictor: OfficialPGRDResidualPredictor,
    *,
    fit_end_frame: int,
    train_end_frame: int,
    config: NativePGRDTrainingConfig,
) -> dict[str, object]:
    """Evaluate one untouched training-tail interval without opening its future."""

    data = _load_pickle(final_data_path)
    baseline = np.asarray(_load_pickle(baseline_trajectory_path), dtype=float)
    gt_track = np.asarray(_load_pickle(gt_track_path), dtype=float)
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
    frame_count, original_count, _ = observed.shape
    if not config.history_length + 2 < fit_end_frame < train_end_frame <= frame_count:
        raise ValueError("invalid validation boundaries")
    if baseline.shape[0] < frame_count or baseline.shape[1] < original_count:
        raise ValueError("baseline trajectory does not cover observations")
    if config.number_of_points > original_count:
        raise ValueError("number_of_points exceeds tracked object points")
    baseline = baseline[:frame_count]
    valid = _target_validity(visible, motion_valid)
    residual = observed - baseline[:, :original_count]
    fit_filled = _temporally_fill(residual, valid, fit_end_frame)
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
    normalizer = MetricNormalizer.fit(
        baseline[0, sample_indices],
        config.normalized_extent,
        yaw_degrees=config.yaw_degrees,
    )
    observed_fit = baseline[:fit_end_frame, :original_count] + fit_filled
    sampled = rollout_pgrd_correction(
        baseline[:, :original_count],
        observed_fit,
        sample_indices,
        predictor,
        normalizer,
        start_frame=fit_end_frame,
        end_frame=train_end_frame,
        history_length=config.history_length,
        temporal_warmup_steps=config.temporal_window,
        simulation_dt=config.simulation_dt,
        model_frame_stride=config.model_frame_stride,
        trust=1.0,
        maximum_residual_m=config.maximum_residual_m,
    )
    original = compose_dense_endpoint_with_sampled_dynamics(
        sampled,
        fit_filled[-1],
        sample_indices,
        sample_lift_indices,
        sample_lift_weights,
    )
    persistence = np.repeat(
        fit_filled[-1][None], train_end_frame - fit_end_frame, axis=0
    )
    num_surface_points = original_count + len(np.asarray(data["surface_points"]))

    def metrics(correction: np.ndarray | None) -> dict[str, object]:
        candidate = baseline.copy()
        if correction is not None:
            lifted = _lift_residual(
                correction,
                baseline.shape[1],
                lift_indices,
                lift_weights,
                maximum_norm=config.maximum_residual_m,
            )
            candidate[fit_end_frame:train_end_frame] += lifted
        return evaluate_official_phystwin_interval(
            candidate,
            observed,
            visible,
            gt_track,
            num_surface_points=num_surface_points,
            start_frame=fit_end_frame,
            end_frame=train_end_frame,
        )

    baseline_metrics = metrics(None)
    persistence_metrics = metrics(persistence)
    native_metrics = metrics(original)
    ratios = _metric_ratios(native_metrics, persistence_metrics)
    return {
        "baseline_official_evaluation": baseline_metrics,
        "persistence_official_evaluation": persistence_metrics,
        "native_pgrd_official_evaluation": native_metrics,
        "metric_ratios_relative_to_persistence": ratios,
        "balanced_improvement": 1.0 - _relative_score(
            native_metrics, persistence_metrics
        ),
        "future_metrics_opened": False,
    }


def aggregate_native_pgrd_gate(
    validation_results: dict[str, dict[str, object]],
    *,
    config: NativePGRDTrainingConfig,
) -> dict[str, object]:
    """Apply the frozen cross-action gate without target-specific fallback."""

    if len(validation_results) != 3:
        raise ValueError("the development gate requires exactly three cases")
    metrics = ("chamfer_distance_m", "track_error_m")
    aggregate_ratios = {
        metric: float(
            np.mean(
                [
                    result["metric_ratios_relative_to_persistence"][metric]
                    for result in validation_results.values()
                ]
            )
        )
        for metric in metrics
    }
    both_win_count = sum(
        max(result["metric_ratios_relative_to_persistence"].values()) < 1.0
        for result in validation_results.values()
    )
    balanced_improvement = 1.0 - 0.5 * sum(aggregate_ratios.values())
    passed = (
        balanced_improvement >= config.minimum_balanced_improvement
        and both_win_count >= config.minimum_both_win_count
        and max(aggregate_ratios.values()) < 1.0
        and all(
            max(result["metric_ratios_relative_to_persistence"].values())
            <= config.maximum_metric_ratio
            for result in validation_results.values()
        )
    )
    return {
        "passed": passed,
        "balanced_improvement": balanced_improvement,
        "aggregate_metric_ratios_relative_to_persistence": aggregate_ratios,
        "both_win_count": both_win_count,
        "exploratory_19_case_future_authorized": passed,
    }


def _load_protocol(path: str | Path) -> dict[str, object]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported native PGRD protocol schema")
    source_cases = protocol.get("source_cases")
    development = protocol.get("development_cases")
    if not isinstance(source_cases, list) or not source_cases:
        raise ValueError("protocol must declare source_cases")
    if not isinstance(development, list) or len(development) != 3:
        raise ValueError("protocol must declare three development_cases")
    names = [str(value) for value in source_cases]
    development_names = [str(value["case"]) for value in development]
    if len(names) != len(set(names)) or len(development_names) != len(
        set(development_names)
    ):
        raise ValueError("protocol case names must be unique")
    if set(names) & set(development_names):
        raise ValueError("source and development cases must be disjoint")
    return protocol


def run_native_pgrd_development(
    protocol_path: str | Path,
    output_dir: str | Path,
    *,
    pgrd_checkout: str | Path,
    pgrd_checkpoint: str | Path,
    device: str = "cuda",
    config: NativePGRDTrainingConfig = NativePGRDTrainingConfig(),
    source_case_limit: int | None = None,
) -> dict[str, object]:
    """Train on source prefixes and evaluate only the three rejection cases."""

    _validate_config(config)
    protocol = _load_protocol(protocol_path)
    data_root = Path(str(protocol["data_root"]))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if source_case_limit is not None and source_case_limit < 1:
        raise ValueError("source_case_limit must be positive")
    source_cases = list(protocol["source_cases"])
    smoke_mode = source_case_limit is not None
    if source_case_limit is not None:
        source_cases = source_cases[:source_case_limit]
    predictor = OfficialPGRDResidualPredictor(
        pgrd_checkout,
        pgrd_checkpoint,
        device=device,
        history_length=config.history_length,
        temporal_window=config.temporal_window,
        residual_scale=config.residual_scale,
    )

    sequences: list[PGRDFeatureSequence] = []
    source_inputs: list[dict[str, object]] = []
    for raw_case in source_cases:
        case = str(raw_case)
        case_root = data_root / case
        split_path = case_root / "split.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        end_frame = int(split["train"][1])
        final_path = case_root / "final_data.pkl"
        baseline_path = case_root / "inference.pkl"
        data = _load_pickle(final_path)
        baseline = np.asarray(_load_pickle(baseline_path), dtype=float)
        observed = np.asarray(data["object_points"], dtype=float)
        valid = _target_validity(
            np.asarray(data["object_visibilities"], dtype=bool),
            np.asarray(data["object_motions_valid"], dtype=bool),
        )
        if config.number_of_points > observed.shape[1]:
            raise ValueError(f"{case}: number_of_points exceeds observations")
        sample_indices = deterministic_farthest_point_sample(
            baseline[0, : observed.shape[1]], config.number_of_points
        )
        normalizer = MetricNormalizer.fit(
            baseline[0, sample_indices],
            config.normalized_extent,
            yaw_degrees=config.yaw_degrees,
        )
        input_hashes = {
            "final_data": _sha256(final_path),
            "baseline_trajectory": _sha256(baseline_path),
            "split": _sha256(split_path),
        }
        cache_contract = json.dumps(
            {
                "case": case,
                "end_frame": end_frame,
                "config": asdict(config),
                "inputs": input_hashes,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        cache_key = hashlib.sha256(cache_contract).hexdigest()
        cache_path = output / "feature_cache" / f"{case}-{cache_key[:16]}.npz"
        if cache_path.is_file():
            sequence = load_feature_sequence(cache_path)
        else:
            sequence = build_teacher_forced_pgrd_sequence(
                case,
                baseline,
                observed,
                valid,
                sample_indices,
                predictor,
                normalizer,
                end_frame=end_frame,
                config=config,
            )
            save_feature_sequence(sequence, cache_path)
        sequences.append(sequence)
        source_inputs.append(
            {
                "case": case,
                "train_end_frame_exclusive": end_frame,
                "final_data_sha256": input_hashes["final_data"],
                "baseline_trajectory_sha256": input_hashes[
                    "baseline_trajectory"
                ],
                "split_sha256": input_hashes["split"],
                "feature_cache_contract_sha256": cache_key,
                "feature_cache": str(cache_path.resolve()),
                "feature_cache_sha256": _sha256(cache_path),
            }
        )

    trained_checkpoint = output / "native_pgrd_temporal.pt"
    training = fit_native_temporal_head(
        predictor, sequences, trained_checkpoint, config=config
    )
    validation: dict[str, dict[str, object]] = {}
    for entry in protocol["development_cases"]:
        case = str(entry["case"])
        case_root = data_root / case
        validation[case] = evaluate_native_pgrd_validation(
            case_root / "final_data.pkl",
            case_root / "inference.pkl",
            case_root / "gt_track_3d.pkl",
            predictor,
            fit_end_frame=int(entry["fit_end_frame"]),
            train_end_frame=int(entry["train_end_frame"]),
            config=config,
        )
    gate = aggregate_native_pgrd_gate(validation, config=config)
    if smoke_mode:
        gate = {
            **gate,
            "passed": False,
            "exploratory_19_case_future_authorized": False,
            "smoke_override": "A source-limited run cannot authorize evaluation.",
        }
    summary: dict[str, object] = {
        "schema_version": 1,
        "config": asdict(config),
        "contract": {
            "spatial_encoder": "pinned PGRD checkpoint, frozen",
            "trainable_parameters": "PGRD temporal residual head only",
            "training_evidence": "released source-case training prefixes only",
            "development_evidence": "three untouched sloth training tails",
            "dense_endpoint_anchor_preserved": True,
            "future_observations_used": False,
            "development_future_metrics_opened": False,
            "exploratory_future_opened_only_after_gate": True,
            "physical_injection_claim": False,
            "source_limited_smoke": smoke_mode,
        },
        "protocol": {
            "path": str(Path(protocol_path).resolve()),
            "sha256": _sha256(protocol_path),
        },
        "pgrd": predictor.provenance,
        "source_inputs": source_inputs,
        "training": training,
        "validation": validation,
        "gate": gate,
    }
    summary_path = output / "summary.json"
    summary["outputs"] = {
        "summary": str(summary_path.resolve()),
        "checkpoint": str(trained_checkpoint.resolve()),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
