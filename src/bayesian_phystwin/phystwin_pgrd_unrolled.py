"""Source-only unrolled adaptation of PGRD to PhysTwin trajectories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .phystwin_pgrd_adapter import (
    MetricNormalizer,
    OfficialPGRDResidualPredictor,
    deterministic_farthest_point_sample,
)
from .phystwin_pgrd_native import (
    NativePGRDTrainingConfig,
    _cadence_frames,
    _load_protocol,
    _validate_config,
    aggregate_native_pgrd_gate,
    evaluate_native_pgrd_validation,
)
from .phystwin_residual_dynamics import (
    _load_pickle,
    _sha256,
    _target_validity,
    _temporally_fill,
)


@dataclass(frozen=True)
class UnrolledPGRDTrainingConfig:
    """Frozen rollout-adaptation and rejection-gate settings."""

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
    rollout_steps: int = 5
    windows_per_case_per_epoch: int = 4
    epochs: int = 20
    temporal_learning_rate: float = 5e-5
    spatial_decoder_learning_rate: float = 1e-5
    weight_decay: float = 0.0
    l2_starting_point_weight: float = 1e-4
    huber_delta_m: float = 0.002
    random_seed: int = 20260719
    minimum_balanced_improvement: float = 0.01
    minimum_both_win_count: int = 2
    maximum_metric_ratio: float = 1.02

    def evaluation_config(self) -> NativePGRDTrainingConfig:
        """Return the exactly matched anchored-rollout evaluation contract."""

        return NativePGRDTrainingConfig(
            number_of_points=self.number_of_points,
            normalized_extent=self.normalized_extent,
            yaw_degrees=self.yaw_degrees,
            history_length=self.history_length,
            temporal_window=self.temporal_window,
            model_frame_stride=self.model_frame_stride,
            simulation_dt=self.simulation_dt,
            interpolation_neighbors=self.interpolation_neighbors,
            maximum_residual_m=self.maximum_residual_m,
            residual_scale=self.residual_scale,
            epochs=max(self.epochs, 1),
            random_seed=self.random_seed,
            minimum_balanced_improvement=self.minimum_balanced_improvement,
            minimum_both_win_count=self.minimum_both_win_count,
            maximum_metric_ratio=self.maximum_metric_ratio,
        )


@dataclass(frozen=True)
class PGRDUnrolledSequence:
    """Sampled metric trajectory and fixed normalization for one source case."""

    case: str
    target_frames: np.ndarray
    sample_indices: np.ndarray
    baseline_m: np.ndarray
    observed_m: np.ndarray
    valid: np.ndarray
    center_m: np.ndarray
    rotation_model_from_metric: np.ndarray
    scale_per_m: float

    @property
    def normalizer(self) -> MetricNormalizer:
        return MetricNormalizer(
            center_m=np.asarray(self.center_m, dtype=float),
            rotation_model_from_metric=np.asarray(
                self.rotation_model_from_metric, dtype=float
            ),
            scale_per_m=float(self.scale_per_m),
            normalized_extent=float("nan"),
            yaw_degrees=float("nan"),
        )


def _validate_unrolled_config(config: UnrolledPGRDTrainingConfig) -> None:
    _validate_config(config.evaluation_config())
    if config.rollout_steps < 2:
        raise ValueError("rollout_steps must be at least two")
    if config.windows_per_case_per_epoch < 1:
        raise ValueError("windows_per_case_per_epoch must be positive")
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.temporal_learning_rate <= 0.0:
        raise ValueError("temporal_learning_rate must be positive")
    if config.spatial_decoder_learning_rate <= 0.0:
        raise ValueError("spatial_decoder_learning_rate must be positive")
    if config.weight_decay < 0.0 or config.l2_starting_point_weight < 0.0:
        raise ValueError("regularization weights must be nonnegative")
    if config.huber_delta_m <= 0.0:
        raise ValueError("huber_delta_m must be positive")


def build_unrolled_pgrd_sequence(
    case: str,
    baseline_m: np.ndarray,
    observed_m: np.ndarray,
    valid: np.ndarray,
    sample_indices: np.ndarray,
    normalizer: MetricNormalizer,
    *,
    end_frame: int,
    config: UnrolledPGRDTrainingConfig,
) -> PGRDUnrolledSequence:
    """Create a future-blind sampled source trajectory for recurrent training."""

    _validate_unrolled_config(config)
    baseline = np.asarray(baseline_m, dtype=float)
    observed = np.asarray(observed_m, dtype=float)
    validity = np.asarray(valid, dtype=bool)
    indices = np.asarray(sample_indices, dtype=np.int64)
    if baseline.ndim != 3 or baseline.shape[2] != 3:
        raise ValueError("baseline_m must have shape (T, M, 3)")
    if observed.ndim != 3 or observed.shape[2] != 3:
        raise ValueError("observed_m must have shape (T, N, 3)")
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
    target_frames = _cadence_frames(end_frame, config.evaluation_config())
    if len(target_frames) < config.rollout_steps:
        raise ValueError("source sequence is shorter than the unrolled window")
    return PGRDUnrolledSequence(
        case=case,
        target_frames=target_frames,
        sample_indices=indices,
        baseline_m=baseline[:end_frame, indices],
        observed_m=filled_observed[:, indices],
        valid=validity[:end_frame, indices],
        center_m=np.asarray(normalizer.center_m, dtype=float),
        rotation_model_from_metric=np.asarray(
            normalizer.rotation_model_from_metric, dtype=float
        ),
        scale_per_m=float(normalizer.scale_per_m),
    )


def save_unrolled_sequence(sequence: PGRDUnrolledSequence, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        case=np.asarray(sequence.case),
        target_frames=sequence.target_frames,
        sample_indices=sequence.sample_indices,
        baseline_m=sequence.baseline_m.astype(np.float32),
        observed_m=sequence.observed_m.astype(np.float32),
        valid=sequence.valid,
        center_m=sequence.center_m,
        rotation_model_from_metric=sequence.rotation_model_from_metric,
        scale_per_m=np.asarray(sequence.scale_per_m),
    )


def load_unrolled_sequence(path: str | Path) -> PGRDUnrolledSequence:
    with np.load(path, allow_pickle=False) as archive:
        return PGRDUnrolledSequence(
            case=str(archive["case"].item()),
            target_frames=np.asarray(archive["target_frames"], dtype=np.int64),
            sample_indices=np.asarray(archive["sample_indices"], dtype=np.int64),
            baseline_m=np.asarray(archive["baseline_m"], dtype=float),
            observed_m=np.asarray(archive["observed_m"], dtype=float),
            valid=np.asarray(archive["valid"], dtype=bool),
            center_m=np.asarray(archive["center_m"], dtype=float),
            rotation_model_from_metric=np.asarray(
                archive["rotation_model_from_metric"], dtype=float
            ),
            scale_per_m=float(archive["scale_per_m"].item()),
        )


def available_window_starts(
    sequence: PGRDUnrolledSequence, rollout_steps: int
) -> np.ndarray:
    """Return target-frame indexes that admit a complete recursive window."""

    count = len(sequence.target_frames) - rollout_steps + 1
    if rollout_steps < 1 or count < 1:
        raise ValueError("sequence does not admit the requested rollout window")
    return np.arange(count, dtype=np.int64)


def _sample_window_starts(
    sequence: PGRDUnrolledSequence,
    rollout_steps: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    available = available_window_starts(sequence, rollout_steps)
    if len(available) <= count:
        return available
    return np.sort(rng.choice(available, size=count, replace=False))


def _metric_positions_to_model(torch: object, values: object, sequence: PGRDUnrolledSequence) -> object:
    center = torch.as_tensor(
        sequence.center_m, dtype=values.dtype, device=values.device
    )
    rotation = torch.as_tensor(
        sequence.rotation_model_from_metric,
        dtype=values.dtype,
        device=values.device,
    )
    return (values - center) @ rotation.T * sequence.scale_per_m


def _metric_velocities_to_model(torch: object, values: object, sequence: PGRDUnrolledSequence) -> object:
    rotation = torch.as_tensor(
        sequence.rotation_model_from_metric,
        dtype=values.dtype,
        device=values.device,
    )
    return values @ rotation.T * sequence.scale_per_m


def _model_displacements_to_metric(torch: object, values: object, sequence: PGRDUnrolledSequence) -> object:
    rotation = torch.as_tensor(
        sequence.rotation_model_from_metric,
        dtype=values.dtype,
        device=values.device,
    )
    return values / sequence.scale_per_m @ rotation


def _spatial_features_tensor(
    predictor: OfficialPGRDResidualPredictor,
    sequence: PGRDUnrolledSequence,
    current_m: object,
    current_velocity_m: object,
    history_m: list[object],
    history_velocity_m: list[object],
    simulation_m: object,
    simulation_velocity_m: object,
) -> object:
    torch = predictor._torch
    count = current_m.shape[0]
    enabled = torch.ones((1, count, 3), device=predictor._device)
    return predictor._residualnet(
        _metric_positions_to_model(torch, current_m, sequence)[None],
        _metric_velocities_to_model(torch, current_velocity_m, sequence)[None],
        _metric_positions_to_model(
            torch, torch.stack(history_m, dim=1), sequence
        ).reshape(1, count, -1),
        _metric_velocities_to_model(
            torch, torch.stack(history_velocity_m, dim=1), sequence
        ).reshape(1, count, -1),
        enabled,
        _metric_positions_to_model(torch, simulation_m, sequence)[None],
        _metric_velocities_to_model(
            torch, simulation_velocity_m, sequence
        )[None],
    )


def _observed_history(
    predictor: OfficialPGRDResidualPredictor,
    sequence: PGRDUnrolledSequence,
    current_frame: int,
    config: UnrolledPGRDTrainingConfig,
) -> tuple[list[object], list[object]]:
    torch = predictor._torch
    stride = config.model_frame_stride
    frames = [
        current_frame - offset * stride
        for offset in reversed(range(config.history_length))
    ]
    history = [
        torch.as_tensor(
            sequence.observed_m[frame],
            dtype=torch.float32,
            device=predictor._device,
        )
        for frame in frames
    ]
    velocities = [
        torch.as_tensor(
            (
                sequence.observed_m[frame]
                - sequence.observed_m[frame - stride]
            )
            / config.simulation_dt,
            dtype=torch.float32,
            device=predictor._device,
        )
        for frame in frames
    ]
    return history, velocities


def _warm_temporal_window(
    predictor: OfficialPGRDResidualPredictor,
    sequence: PGRDUnrolledSequence,
    start_index: int,
    config: UnrolledPGRDTrainingConfig,
) -> None:
    torch = predictor._torch
    earliest = max(0, start_index - config.temporal_window)
    with torch.no_grad():
        for index in range(earliest, start_index):
            target_frame = int(sequence.target_frames[index])
            current_frame = target_frame - config.model_frame_stride
            history, history_velocities = _observed_history(
                predictor, sequence, current_frame, config
            )
            current = history[-1]
            current_velocity = history_velocities[-1]
            simulation = torch.as_tensor(
                sequence.baseline_m[target_frame],
                dtype=torch.float32,
                device=predictor._device,
            )
            simulation_velocity = (
                simulation
                - torch.as_tensor(
                    sequence.baseline_m[current_frame],
                    dtype=torch.float32,
                    device=predictor._device,
                )
            ) / config.simulation_dt
            features = _spatial_features_tensor(
                predictor,
                sequence,
                current,
                current_velocity,
                history,
                history_velocities,
                simulation,
                simulation_velocity,
            )
            predictor._temporal(
                features,
                rollout_window_size=config.temporal_window,
                residual_scale=config.residual_scale,
            )


def _unrolled_window_loss(
    predictor: OfficialPGRDResidualPredictor,
    sequence: PGRDUnrolledSequence,
    start_index: int,
    config: UnrolledPGRDTrainingConfig,
) -> object:
    torch = predictor._torch
    predictor._temporal.reset_window()
    _warm_temporal_window(predictor, sequence, start_index, config)
    first_target = int(sequence.target_frames[start_index])
    current_frame = first_target - config.model_frame_stride
    history, history_velocities = _observed_history(
        predictor, sequence, current_frame, config
    )
    current = history[-1]
    current_velocity = history_velocities[-1]
    losses: list[object] = []

    for index in range(start_index, start_index + config.rollout_steps):
        target_frame = int(sequence.target_frames[index])
        simulation = torch.as_tensor(
            sequence.baseline_m[target_frame],
            dtype=torch.float32,
            device=predictor._device,
        )
        previous_baseline = torch.as_tensor(
            sequence.baseline_m[target_frame - config.model_frame_stride],
            dtype=torch.float32,
            device=predictor._device,
        )
        simulation_velocity = (
            simulation - previous_baseline
        ) / config.simulation_dt
        features = _spatial_features_tensor(
            predictor,
            sequence,
            current,
            current_velocity,
            history,
            history_velocities,
            simulation,
            simulation_velocity,
        )
        residual_velocity_model = predictor._temporal(
            features,
            rollout_window_size=config.temporal_window,
            residual_scale=config.residual_scale,
        )[0]
        residual_m = _model_displacements_to_metric(
            torch,
            residual_velocity_model * config.simulation_dt,
            sequence,
        )
        norm = torch.linalg.vector_norm(residual_m, dim=1, keepdim=True)
        residual_m = residual_m * torch.clamp(
            config.maximum_residual_m / torch.clamp(norm, min=1e-12),
            max=1.0,
        )
        predicted = simulation + residual_m
        target = torch.as_tensor(
            sequence.observed_m[target_frame],
            dtype=torch.float32,
            device=predictor._device,
        )
        valid = torch.as_tensor(
            sequence.valid[target_frame],
            dtype=torch.bool,
            device=predictor._device,
        )
        if bool(valid.any()):
            losses.append(
                torch.nn.functional.smooth_l1_loss(
                    predicted[valid],
                    target[valid],
                    beta=config.huber_delta_m,
                    reduction="mean",
                )
            )
        next_velocity = (predicted - current) / config.simulation_dt
        history = [*history[1:], predicted]
        history_velocities = [*history_velocities[1:], next_velocity]
        current = predicted
        current_velocity = next_velocity
    if not losses:
        raise ValueError(f"{sequence.case}: unrolled window contains no valid targets")
    return torch.stack(losses).mean()


def fit_unrolled_pgrd(
    predictor: OfficialPGRDResidualPredictor,
    sequences: list[PGRDUnrolledSequence],
    output_checkpoint: str | Path,
    *,
    config: UnrolledPGRDTrainingConfig,
) -> dict[str, object]:
    """Adapt PGRD's spatial decoder and temporal head through short rollouts."""

    _validate_unrolled_config(config)
    if not sequences:
        raise ValueError("at least one source sequence is required")
    torch = predictor._torch
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    for parameter in predictor._residualnet.parameters():
        parameter.requires_grad_(False)
    for parameter in predictor._residualnet.decoder.parameters():
        parameter.requires_grad_(True)
    for parameter in predictor._temporal.parameters():
        parameter.requires_grad_(True)
    predictor._residualnet.eval()
    predictor._temporal.train()

    spatial_parameters = list(predictor._residualnet.decoder.parameters())
    temporal_parameters = list(predictor._temporal.parameters())
    trainable = [*spatial_parameters, *temporal_parameters]
    initial = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": spatial_parameters,
                "lr": config.spatial_decoder_learning_rate,
            },
            {
                "params": temporal_parameters,
                "lr": config.temporal_learning_rate,
            },
        ],
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.random_seed)
    epoch_losses: list[float] = []
    window_count = 0

    for _epoch in range(config.epochs):
        work: list[tuple[int, int]] = []
        for sequence_index, sequence in enumerate(sequences):
            starts = _sample_window_starts(
                sequence,
                config.rollout_steps,
                config.windows_per_case_per_epoch,
                rng,
            )
            work.extend((sequence_index, int(start)) for start in starts)
        rng.shuffle(work)
        total = 0.0
        for sequence_index, start in work:
            optimizer.zero_grad(set_to_none=True)
            objective = _unrolled_window_loss(
                predictor, sequences[sequence_index], start, config
            )
            if config.l2_starting_point_weight > 0.0:
                squared = torch.zeros((), device=predictor._device)
                parameter_count = 0
                for parameter, starting in zip(trainable, initial):
                    squared = squared + torch.sum(torch.square(parameter - starting))
                    parameter_count += parameter.numel()
                objective = objective + config.l2_starting_point_weight * (
                    squared / max(parameter_count, 1)
                )
            objective.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            total += float(objective.detach().cpu())
        window_count = len(work)
        epoch_losses.append(total / max(window_count, 1))

    predictor._temporal.eval()
    checkpoint = Path(output_checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "residualnet": predictor._residualnet.state_dict(),
            "transformer": predictor._temporal.export_component_state_dicts(),
            "unrolled_training": {
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
        "windows_per_epoch": window_count,
        "initial_loss_m": epoch_losses[0],
        "final_loss_m": epoch_losses[-1],
        "epoch_losses_m": epoch_losses,
        "trainable_spatial_parameter_count": sum(
            parameter.numel() for parameter in spatial_parameters
        ),
        "trainable_temporal_parameter_count": sum(
            parameter.numel() for parameter in temporal_parameters
        ),
    }


def run_unrolled_pgrd_development(
    protocol_path: str | Path,
    output_dir: str | Path,
    *,
    pgrd_checkout: str | Path,
    pgrd_checkpoint: str | Path,
    device: str = "cuda",
    config: UnrolledPGRDTrainingConfig = UnrolledPGRDTrainingConfig(),
    source_case_limit: int | None = None,
) -> dict[str, object]:
    """Train recursively on source prefixes and run the sealed rejection gate."""

    _validate_unrolled_config(config)
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

    sequences: list[PGRDUnrolledSequence] = []
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
        cache_path = output / "sequence_cache" / f"{case}-{cache_key[:16]}.npz"
        if cache_path.is_file():
            sequence = load_unrolled_sequence(cache_path)
        else:
            sequence = build_unrolled_pgrd_sequence(
                case,
                baseline,
                observed,
                valid,
                sample_indices,
                normalizer,
                end_frame=end_frame,
                config=config,
            )
            save_unrolled_sequence(sequence, cache_path)
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
                "sequence_cache_contract_sha256": cache_key,
                "sequence_cache": str(cache_path.resolve()),
                "sequence_cache_sha256": _sha256(cache_path),
            }
        )

    trained_checkpoint = output / "unrolled_pgrd.pt"
    training = fit_unrolled_pgrd(
        predictor, sequences, trained_checkpoint, config=config
    )
    evaluation_config = config.evaluation_config()
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
            config=evaluation_config,
        )
    gate = aggregate_native_pgrd_gate(validation, config=evaluation_config)
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
            "spatial_encoder": "pinned PGRD point transformer, frozen",
            "trainable_parameters": "PGRD spatial decoder and temporal head",
            "training_objective": "five-step recursive source-prefix rollout",
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
