"""Leave-one-topology-out evaluation for latent realized-contact inference."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from causal4d.baselines import FittedBaselines, PredictiveDistribution, fit_baselines
from causal4d.benchmark import (
    CounterfactualBenchmarkConfig,
    Episode,
    ObjectProtocol,
    build_protocol,
    generate_episodes,
    make_parameter_grid,
    protocol_manifest,
)
from causal4d.contact_inference import (
    ContactRolloutBank,
    ContactState,
    GraphContactHypothesisModel,
    LatentContactConfig,
    build_rollout_bank,
    fit_contact_prior,
    posterior_predictive_for_state,
    true_contact_state,
    true_parameter_predictive_for_state,
)
from causal4d.contact_metrics import (
    aggregate_contact_recovery,
    contact_recovery_metrics,
)
from causal4d.metrics import intervention_metrics


@dataclass(frozen=True)
class _FittedObject:
    protocol: ObjectProtocol
    training: tuple[Episode, ...]
    validation: Episode
    held_out: tuple[Episode, ...]
    baselines: FittedBaselines


@dataclass(frozen=True)
class FoldCalibration:
    likelihood_scale_m: float
    likelihood_power: float
    dynamic_likelihood_weight: float
    posterior_temperature: float
    matched_pre_variance_multiplier: float
    shifted_pre_variance_multiplier: float
    matched_online_variance_multiplier: float
    shifted_online_variance_multiplier: float
    source_calibration_rmse_m: float

    def as_dict(self) -> dict[str, float]:
        return {
            "likelihood_scale_m": self.likelihood_scale_m,
            "likelihood_power": self.likelihood_power,
            "dynamic_likelihood_weight": self.dynamic_likelihood_weight,
            "posterior_temperature": self.posterior_temperature,
            "matched_pre_variance_multiplier": self.matched_pre_variance_multiplier,
            "shifted_pre_variance_multiplier": self.shifted_pre_variance_multiplier,
            "matched_online_variance_multiplier": self.matched_online_variance_multiplier,
            "shifted_online_variance_multiplier": self.shifted_online_variance_multiplier,
            "source_calibration_rmse_m": self.source_calibration_rmse_m,
        }

    def variance_multiplier(self, setting: str, shift_probability: float) -> float:
        if setting == "pre_intervention":
            matched = self.matched_pre_variance_multiplier
            shifted = self.shifted_pre_variance_multiplier
        elif setting == "online_adaptation":
            matched = self.matched_online_variance_multiplier
            shifted = self.shifted_online_variance_multiplier
        else:
            raise ValueError("unknown inference setting")
        return float((1.0 - shift_probability) * matched + shift_probability * shifted)


_INTERVENTION_METRICS = (
    "trajectory_rmse_m",
    "relative_intervention_rmse",
    "ade_m",
    "fde_m",
    "early_rmse_m",
    "middle_rmse_m",
    "late_rmse_m",
    "direction_error_deg",
    "gross_failure",
    "coverage",
    "coverage_error",
    "mean_interval_width_m",
    "nees",
    "gaussian_nll",
)


def _prediction_window(
    prediction: PredictiveDistribution,
    truth: np.ndarray,
    start_frame: int,
) -> tuple[PredictiveDistribution, np.ndarray]:
    if not 0 <= start_frame < truth.shape[0] - 1:
        raise ValueError("forecast window must contain at least two frames")
    return (
        PredictiveDistribution(
            method=prediction.method,
            mean=prediction.mean[start_frame:],
            variance=prediction.variance[start_frame:],
            interval_lower=(
                prediction.interval_lower[start_frame:]
                if prediction.interval_lower is not None
                else None
            ),
            interval_upper=(
                prediction.interval_upper[start_frame:]
                if prediction.interval_upper is not None
                else None
            ),
        ),
        truth[start_frame:],
    )


def _interval_variance_multiplier(
    predictions: Sequence[PredictiveDistribution],
    truths: Sequence[np.ndarray],
    *,
    confidence_level: float,
    minimum: float,
    maximum: float,
) -> float:
    """Calibrate scale against empirical coverage of mixture-quantile intervals."""

    if any(
        prediction.interval_lower is None or prediction.interval_upper is None
        for prediction in predictions
    ):
        raise ValueError("mixture interval calibration requires explicit bounds")
    candidates = np.geomspace(minimum, maximum, 121)
    scores: list[tuple[float, float, float]] = []
    for scale in candidates:
        square_root = np.sqrt(scale)
        covered: list[np.ndarray] = []
        widths: list[np.ndarray] = []
        for prediction, truth in zip(predictions, truths, strict=True):
            lower = prediction.mean + square_root * (
                prediction.interval_lower - prediction.mean
            )
            upper = prediction.mean + square_root * (
                prediction.interval_upper - prediction.mean
            )
            covered.append((truth >= lower) & (truth <= upper))
            widths.append(upper - lower)
        coverage = float(
            np.mean(np.concatenate([item.reshape(-1) for item in covered]))
        )
        width = float(np.mean(np.concatenate([item.reshape(-1) for item in widths])))
        scores.append((abs(coverage - confidence_level), width, float(scale)))
    return min(scores)[2]


def _temper_joint_weights(weights: np.ndarray, temperature: float) -> np.ndarray:
    log_weights = np.log(np.maximum(np.asarray(weights, dtype=float), 1e-300))
    log_weights *= temperature
    log_weights -= float(np.max(log_weights))
    tempered = np.exp(log_weights)
    tempered /= np.sum(tempered)
    return tempered


def _contact_shift_probability(
    bank: ContactRolloutBank, contact_weights: np.ndarray
) -> float:
    return float(
        sum(
            weight
            for state, weight in zip(bank.contact_states, contact_weights, strict=True)
            if state.contact_nodes != bank.action.contact_nodes
        )
    )


def _calibrate_fold(
    source_objects: Sequence[_FittedObject],
    model: GraphContactHypothesisModel,
    benchmark_config: CounterfactualBenchmarkConfig,
    contact_config: LatentContactConfig,
    *,
    calibration_seed: int,
) -> FoldCalibration:
    """Select likelihood and predictive scales on labelled source topologies."""

    banks: list[tuple[ContactRolloutBank, Episode, np.ndarray]] = []
    for source_index, source in enumerate(source_objects):
        bank = build_rollout_bank(
            source.protocol.graph_object,
            source.protocol.test_action,
            source.baselines.physics.posterior,
            model,
            simulator_config=benchmark_config.simulator,
            parameter_particle_count=contact_config.parameter_particle_count,
            variance_floor_m2=benchmark_config.predictive_variance_floor_m2,
            confidence_level=contact_config.confidence_level,
        )
        for condition_index, episode in enumerate(source.held_out):
            rng = np.random.default_rng(
                calibration_seed + source_index * 10_007 + condition_index * 101
            )
            observations = episode.truth + rng.normal(
                scale=contact_config.observation_noise_std_m,
                size=episode.truth.shape,
            )
            banks.append((bank, episode, observations))

    prefix = contact_config.prefix_frame_count(benchmark_config.frame_count)
    errors_by_scale: list[tuple[float, float, float, float]] = []
    for scale in contact_config.likelihood_scales_m:
        for likelihood_power in contact_config.likelihood_powers:
            for dynamic_weight in contact_config.dynamic_likelihood_weights:
                errors: list[float] = []
                for bank, calibration_episode, observations in banks:
                    joint_weights = bank.update_weights(
                        observations,
                        prefix_frame_count=prefix,
                        likelihood_scale_m=scale,
                        likelihood_power=likelihood_power,
                        dynamic_likelihood_weight=dynamic_weight,
                    )
                    prediction = bank.predictive_distribution(
                        joint_weights,
                        method="latent_contact",
                        include_intervals=False,
                    )
                    errors.append(
                        float(
                            np.sqrt(
                                np.mean(
                                    np.square(
                                        prediction.mean[prefix:]
                                        - calibration_episode.truth[prefix:]
                                    )
                                )
                            )
                        )
                    )
                errors_by_scale.append(
                    (
                        scale,
                        likelihood_power,
                        dynamic_weight,
                        float(np.mean(errors)),
                    )
                )
    selected_scale, selected_power, selected_dynamic_weight, selected_rmse = min(
        errors_by_scale, key=lambda item: item[3]
    )

    temperature_scores: list[tuple[float, float]] = []
    for temperature in contact_config.posterior_temperatures:
        brier_scores: list[float] = []
        for bank, calibration_episode, observations in banks:
            raw_weights = bank.update_weights(
                observations,
                prefix_frame_count=prefix,
                likelihood_scale_m=selected_scale,
                likelihood_power=selected_power,
                dynamic_likelihood_weight=selected_dynamic_weight,
            )
            joint_weights = _temper_joint_weights(raw_weights, temperature)
            recovery = contact_recovery_metrics(
                bank.contact_states,
                bank.contact_marginal(joint_weights),
                true_contact_state(
                    bank.graph_object,
                    calibration_episode.action,
                    calibration_episode.condition,
                ),
                confidence_level=contact_config.confidence_level,
            )
            brier_scores.append(float(recovery["node_brier"]))
        temperature_scores.append((temperature, float(np.mean(brier_scores))))
    selected_temperature = min(temperature_scores, key=lambda item: item[1])[0]

    pre_predictions: dict[bool, list[PredictiveDistribution]] = {
        False: [],
        True: [],
    }
    pre_truths: dict[bool, list[np.ndarray]] = {False: [], True: []}
    online_predictions: dict[bool, list[PredictiveDistribution]] = {
        False: [],
        True: [],
    }
    online_truths: dict[bool, list[np.ndarray]] = {False: [], True: []}
    for bank, calibration_episode, observations in banks:
        shifted = calibration_episode.condition.shift_contact_nodes
        pre_predictions[shifted].append(
            bank.predictive_distribution(method="latent_contact")
        )
        pre_truths[shifted].append(calibration_episode.truth)
        raw_weights = bank.update_weights(
            observations,
            prefix_frame_count=prefix,
            likelihood_scale_m=selected_scale,
            likelihood_power=selected_power,
            dynamic_likelihood_weight=selected_dynamic_weight,
        )
        joint_weights = _temper_joint_weights(raw_weights, selected_temperature)
        prediction, truth = _prediction_window(
            bank.predictive_distribution(
                joint_weights,
                method="latent_contact",
            ),
            calibration_episode.truth,
            prefix,
        )
        online_predictions[shifted].append(prediction)
        online_truths[shifted].append(truth)

    return FoldCalibration(
        likelihood_scale_m=float(selected_scale),
        likelihood_power=float(selected_power),
        dynamic_likelihood_weight=float(selected_dynamic_weight),
        posterior_temperature=float(selected_temperature),
        matched_pre_variance_multiplier=_interval_variance_multiplier(
            pre_predictions[False],
            pre_truths[False],
            confidence_level=contact_config.confidence_level,
            minimum=contact_config.variance_scale_min,
            maximum=contact_config.variance_scale_max,
        ),
        shifted_pre_variance_multiplier=_interval_variance_multiplier(
            pre_predictions[True],
            pre_truths[True],
            confidence_level=contact_config.confidence_level,
            minimum=contact_config.variance_scale_min,
            maximum=contact_config.variance_scale_max,
        ),
        matched_online_variance_multiplier=_interval_variance_multiplier(
            online_predictions[False],
            online_truths[False],
            confidence_level=contact_config.confidence_level,
            minimum=contact_config.variance_scale_min,
            maximum=contact_config.variance_scale_max,
        ),
        shifted_online_variance_multiplier=_interval_variance_multiplier(
            online_predictions[True],
            online_truths[True],
            confidence_level=contact_config.confidence_level,
            minimum=contact_config.variance_scale_min,
            maximum=contact_config.variance_scale_max,
        ),
        source_calibration_rmse_m=float(selected_rmse),
    )


def _joint_parameter_metrics(
    bank: ContactRolloutBank,
    joint_weights: np.ndarray,
) -> dict[str, float]:
    marginal = bank.parameter_marginal(joint_weights)
    mean = np.sum(marginal[:, None] * bank.parameter_particles, axis=0)
    truth = bank.graph_object.true_parameters.as_array()
    return {
        "parameter_effective_sample_size": float(1.0 / np.sum(np.square(marginal))),
        "stiffness_posterior_mean": float(mean[0]),
        "stiffness_absolute_error": float(abs(mean[0] - truth[0])),
        "damping_posterior_mean": float(mean[1]),
        "damping_absolute_error": float(abs(mean[1] - truth[1])),
        "contact_gain_posterior_mean": float(mean[2]),
        "contact_gain_absolute_error": float(abs(mean[2] - truth[2])),
    }


def _append_intervention_row(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    target: _FittedObject,
    episode: Episode,
    setting: str,
    prediction: PredictiveDistribution,
    start_frame: int,
    source_objects: tuple[str, ...],
    calibration: FoldCalibration,
    contact_config: LatentContactConfig,
    benchmark_config: CounterfactualBenchmarkConfig,
) -> None:
    window_prediction, window_truth = _prediction_window(
        prediction, episode.truth, start_frame
    )
    rows.append(
        {
            "seed": seed,
            "object": target.protocol.graph_object.name,
            "held_out_topology": True,
            "source_objects": ";".join(source_objects),
            "action": episode.action.action_id,
            "world_condition": episode.condition.name,
            "setting": setting,
            "method": prediction.method,
            "observation_fraction": (
                contact_config.observation_fraction
                if setting == "online_adaptation"
                else 0.0
            ),
            "forecast_start_frame": start_frame,
            "selected_likelihood_scale_m": calibration.likelihood_scale_m,
            **intervention_metrics(
                window_prediction,
                window_truth,
                confidence_level=contact_config.confidence_level,
                gross_failure_threshold_m=benchmark_config.gross_failure_threshold_m,
            ),
        }
    )


def aggregate_latent_interventions(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (str(row["setting"]), str(row["method"]), str(row["world_condition"]))
        ].append(row)
    output: list[dict[str, Any]] = []
    for (setting, method, world), selected in sorted(groups.items()):
        aggregate: dict[str, Any] = {
            "setting": setting,
            "method": method,
            "world_condition": world,
            "case_count": len(selected),
            "object_count": len({row["object"] for row in selected}),
        }
        for metric in _INTERVENTION_METRICS:
            aggregate[f"mean_{metric}"] = float(
                np.mean([float(row[metric]) for row in selected])
            )
        output.append(aggregate)
    return output


def _aggregate_object_gap_closure(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    objects = sorted({str(row["object"]) for row in rows})
    for object_name in objects:
        errors: dict[str, float] = {}
        for method in (
            "nominal_physics",
            "latent_contact",
            "oracle_contact_theta",
        ):
            selected = [
                row["trajectory_rmse_m"]
                for row in rows
                if row["setting"] == "online_adaptation"
                and row["world_condition"] == "shifted_contact"
                and row["object"] == object_name
                and row["method"] == method
            ]
            errors[method] = float(np.mean(selected))
        oracle_gap = errors["nominal_physics"] - errors["oracle_contact_theta"]
        closure = (
            (errors["nominal_physics"] - errors["latent_contact"]) / oracle_gap
            if oracle_gap > 1e-12
            else 0.0
        )
        output.append(
            {
                "object": object_name,
                "nominal_rmse_m": errors["nominal_physics"],
                "latent_rmse_m": errors["latent_contact"],
                "oracle_rmse_m": errors["oracle_contact_theta"],
                "oracle_gap_m": oracle_gap,
                "oracle_gap_closure": float(closure),
            }
        )
    return output


def _find_aggregate(
    rows: list[dict[str, Any]],
    *,
    setting: str,
    method: str,
    world: str,
) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["setting"] == setting
        and row["method"] == method
        and row["world_condition"] == world
    )


def _find_contact_aggregate(
    rows: list[dict[str, Any]], *, setting: str, world: str
) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["setting"] == setting and row["world_condition"] == world
    )


def evaluate_success_gates(
    intervention_aggregate: list[dict[str, Any]],
    contact_aggregate: list[dict[str, Any]],
    object_gap_closure: list[dict[str, Any]],
    config: LatentContactConfig,
) -> dict[str, Any]:
    """Evaluate every pre-registered milestone gate from aggregate evidence."""

    setting = "online_adaptation"
    shifted_nominal = _find_aggregate(
        intervention_aggregate,
        setting=setting,
        method="nominal_physics",
        world="shifted_contact",
    )
    shifted_latent = _find_aggregate(
        intervention_aggregate,
        setting=setting,
        method="latent_contact",
        world="shifted_contact",
    )
    shifted_oracle = _find_aggregate(
        intervention_aggregate,
        setting=setting,
        method="oracle_contact_theta",
        world="shifted_contact",
    )
    matched_nominal = _find_aggregate(
        intervention_aggregate,
        setting=setting,
        method="nominal_physics",
        world="matched_contact",
    )
    matched_latent = _find_aggregate(
        intervention_aggregate,
        setting=setting,
        method="latent_contact",
        world="matched_contact",
    )
    shifted_recovery = _find_contact_aggregate(
        contact_aggregate, setting=setting, world="shifted_contact"
    )

    nominal_error = shifted_nominal["mean_trajectory_rmse_m"]
    latent_error = shifted_latent["mean_trajectory_rmse_m"]
    oracle_error = shifted_oracle["mean_trajectory_rmse_m"]
    oracle_gap = nominal_error - oracle_error
    gap_closure = (
        (nominal_error - latent_error) / oracle_gap if oracle_gap > 1e-12 else 0.0
    )
    matched_degradation = (
        matched_latent["mean_trajectory_rmse_m"]
        - matched_nominal["mean_trajectory_rmse_m"]
    ) / matched_nominal["mean_trajectory_rmse_m"]
    latent_coverages = [
        _find_aggregate(
            intervention_aggregate,
            setting=setting,
            method="latent_contact",
            world=world,
        )["mean_coverage"]
        for world in ("matched_contact", "shifted_contact")
    ]
    maximum_coverage_error = max(
        abs(coverage - config.confidence_level) for coverage in latent_coverages
    )
    minimum_topology_closure = min(
        row["oracle_gap_closure"] for row in object_gap_closure
    )

    def gate(
        name: str,
        value: float | int,
        threshold: float | int,
        comparison: str,
    ) -> dict[str, Any]:
        passed = value >= threshold if comparison == ">=" else value <= threshold
        return {
            "name": name,
            "value": value,
            "comparison": comparison,
            "threshold": threshold,
            "passed": bool(passed),
        }

    gates = [
        gate("shifted_oracle_gap_closure", gap_closure, config.gate_gap_closure, ">="),
        gate(
            "matched_contact_relative_degradation",
            matched_degradation,
            config.gate_matched_degradation,
            "<=",
        ),
        gate(
            "maximum_online_coverage_error",
            maximum_coverage_error,
            config.gate_coverage_tolerance,
            "<=",
        ),
        gate(
            "shifted_node_accuracy",
            shifted_recovery["node_accuracy"],
            config.gate_node_accuracy,
            ">=",
        ),
        gate(
            "shifted_node_credible_coverage",
            shifted_recovery["node_credible_coverage"],
            config.gate_node_credible_coverage,
            ">=",
        ),
        gate(
            "shifted_node_calibration_error",
            shifted_recovery["node_calibration_error"],
            config.gate_node_calibration_error,
            "<=",
        ),
        gate(
            "shifted_gain_mae",
            shifted_recovery["mean_gain_absolute_error"],
            config.gate_gain_mae,
            "<=",
        ),
        gate(
            "shifted_gain_coverage",
            shifted_recovery["gain_coverage"],
            config.gate_gain_coverage,
            ">=",
        ),
        gate(
            "shifted_delay_mae_steps",
            shifted_recovery["mean_delay_absolute_error"],
            config.gate_delay_mae_steps,
            "<=",
        ),
        gate(
            "shifted_delay_map_accuracy",
            shifted_recovery["delay_map_accuracy"],
            config.gate_delay_map_accuracy,
            ">=",
        ),
        gate(
            "shifted_delay_coverage",
            shifted_recovery["delay_coverage"],
            config.gate_delay_coverage,
            ">=",
        ),
        gate("held_out_topology_count", len(object_gap_closure), 3, ">="),
        gate(
            "minimum_topology_oracle_gap_closure",
            minimum_topology_closure,
            config.gate_minimum_topology_gap_closure,
            ">=",
        ),
    ]
    return {
        "overall_passed": all(item["passed"] for item in gates),
        "gates": gates,
        "derived": {
            "shifted_nominal_rmse_m": nominal_error,
            "shifted_latent_rmse_m": latent_error,
            "shifted_oracle_rmse_m": oracle_error,
            "shifted_oracle_gap_m": oracle_gap,
            "shifted_oracle_gap_closure": gap_closure,
            "matched_relative_degradation": matched_degradation,
            "online_latent_coverages": latent_coverages,
            "minimum_topology_oracle_gap_closure": minimum_topology_closure,
        },
    }


def run_latent_contact_benchmark(
    *,
    seeds: Sequence[int],
    benchmark_config: CounterfactualBenchmarkConfig | None = None,
    contact_config: LatentContactConfig | None = None,
) -> dict[str, Any]:
    """Run pre-intervention and online inference with held-out topology priors."""

    cfg = benchmark_config or CounterfactualBenchmarkConfig()
    latent_cfg = contact_config or LatentContactConfig(
        confidence_level=cfg.confidence_level,
        observation_noise_std_m=cfg.observation_noise_std_m,
    )
    normalized_seeds = [int(seed) for seed in seeds]
    if not normalized_seeds:
        raise ValueError("at least one seed is required")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("seeds must be unique")

    protocols = build_protocol(cfg)
    intervention_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    for seed in normalized_seeds:
        fitted: list[_FittedObject] = []
        for object_index, protocol in enumerate(protocols):
            training, validation, held_out = generate_episodes(
                protocol,
                cfg,
                seed=seed * 10_000 + object_index * 101,
            )
            baselines = fit_baselines(
                training,
                validation,
                make_parameter_grid(protocol.graph_object, cfg),
                cfg,
            )
            fitted.append(
                _FittedObject(
                    protocol=protocol,
                    training=training,
                    validation=validation,
                    held_out=held_out,
                    baselines=baselines,
                )
            )

        for target_index, target in enumerate(fitted):
            sources = tuple(
                item for index, item in enumerate(fitted) if index != target_index
            )
            source_protocols = tuple(item.protocol for item in sources)
            prior = fit_contact_prior(
                source_protocols,
                latent_cfg,
                action_split="test",
            )
            model = GraphContactHypothesisModel(prior=prior, config=latent_cfg)
            calibration = _calibrate_fold(
                sources,
                model,
                cfg,
                latent_cfg,
                calibration_seed=seed * 1_000_003 + target_index * 100_003 + 17,
            )
            source_names = tuple(item.protocol.graph_object.name for item in sources)
            calibration_rows.append(
                {
                    "seed": seed,
                    "held_out_object": target.protocol.graph_object.name,
                    "source_objects": ";".join(source_names),
                    "source_excludes_target": target.protocol.graph_object.name
                    not in source_names,
                    "contact_hypothesis_count": len(
                        model.hypotheses(
                            target.protocol.graph_object, target.protocol.test_action
                        )[0]
                    ),
                    **calibration.as_dict(),
                    "prior": json.dumps(prior.as_dict(latent_cfg), sort_keys=True),
                }
            )

            bank = build_rollout_bank(
                target.protocol.graph_object,
                target.protocol.test_action,
                target.baselines.physics.posterior,
                model,
                simulator_config=cfg.simulator,
                parameter_particle_count=latent_cfg.parameter_particle_count,
                variance_floor_m2=cfg.predictive_variance_floor_m2,
                confidence_level=latent_cfg.confidence_level,
            )
            pre_weights = bank.prior_joint_weights
            pre_contact_weights = bank.contact_marginal(pre_weights)
            pre_shift_probability = _contact_shift_probability(
                bank, pre_contact_weights
            )
            pre_prediction = bank.predictive_distribution(
                method="latent_contact",
                variance_multiplier=calibration.variance_multiplier(
                    "pre_intervention", pre_shift_probability
                ),
            )
            prefix = latent_cfg.prefix_frame_count(cfg.frame_count)

            for condition_index, episode in enumerate(target.held_out):
                observation_rng = np.random.default_rng(
                    seed * 1_000_003 + target_index * 10_007 + condition_index * 97
                )
                online_observations = episode.truth + observation_rng.normal(
                    scale=latent_cfg.observation_noise_std_m,
                    size=episode.truth.shape,
                )
                raw_online_weights = bank.update_weights(
                    online_observations,
                    prefix_frame_count=prefix,
                    likelihood_scale_m=calibration.likelihood_scale_m,
                    likelihood_power=calibration.likelihood_power,
                    dynamic_likelihood_weight=calibration.dynamic_likelihood_weight,
                )
                online_weights = _temper_joint_weights(
                    raw_online_weights, calibration.posterior_temperature
                )
                online_contact_weights = bank.contact_marginal(online_weights)
                online_shift_probability = _contact_shift_probability(
                    bank, online_contact_weights
                )
                online_prediction = bank.predictive_distribution(
                    online_weights,
                    method="latent_contact",
                    variance_multiplier=calibration.variance_multiplier(
                        "online_adaptation", online_shift_probability
                    ),
                )
                nominal_pre_prediction = target.baselines.physics.predict(episode)
                nominal_state = ContactState(
                    contact_nodes=episode.action.contact_nodes,
                    gain_multiplier=1.0,
                    delay_steps=0,
                    slip_fraction=0.0,
                    rotation_radians=0.0,
                )
                nominal_online_prediction = posterior_predictive_for_state(
                    target.protocol.graph_object,
                    episode.action,
                    nominal_state,
                    target.baselines.physics.posterior,
                    simulator_config=cfg.simulator,
                    variance_floor_m2=cfg.predictive_variance_floor_m2,
                    method="nominal_physics",
                    observations=online_observations,
                    prefix_frame_count=prefix,
                    likelihood_scale_m=calibration.likelihood_scale_m,
                    likelihood_power=calibration.likelihood_power,
                    dynamic_likelihood_weight=calibration.dynamic_likelihood_weight,
                    posterior_temperature=calibration.posterior_temperature,
                )
                true_state = true_contact_state(
                    target.protocol.graph_object,
                    episode.action,
                    episode.condition,
                )
                oracle_pre_prediction = posterior_predictive_for_state(
                    target.protocol.graph_object,
                    episode.action,
                    true_state,
                    target.baselines.physics.posterior,
                    simulator_config=cfg.simulator,
                    variance_floor_m2=cfg.predictive_variance_floor_m2,
                    method="oracle_contact",
                )
                oracle_online_prediction = posterior_predictive_for_state(
                    target.protocol.graph_object,
                    episode.action,
                    true_state,
                    target.baselines.physics.posterior,
                    simulator_config=cfg.simulator,
                    variance_floor_m2=cfg.predictive_variance_floor_m2,
                    method="oracle_contact",
                    observations=online_observations,
                    prefix_frame_count=prefix,
                    likelihood_scale_m=calibration.likelihood_scale_m,
                    likelihood_power=calibration.likelihood_power,
                    dynamic_likelihood_weight=calibration.dynamic_likelihood_weight,
                    posterior_temperature=calibration.posterior_temperature,
                )
                oracle_theta_prediction = true_parameter_predictive_for_state(
                    target.protocol.graph_object,
                    episode.action,
                    true_state,
                    simulator_config=cfg.simulator,
                    variance_floor_m2=cfg.predictive_variance_floor_m2,
                )
                for setting, start_frame, predictions in (
                    (
                        "pre_intervention",
                        0,
                        (
                            nominal_pre_prediction,
                            pre_prediction,
                            oracle_pre_prediction,
                            oracle_theta_prediction,
                        ),
                    ),
                    (
                        "online_adaptation",
                        prefix,
                        (
                            nominal_online_prediction,
                            online_prediction,
                            oracle_online_prediction,
                            oracle_theta_prediction,
                        ),
                    ),
                ):
                    for prediction in predictions:
                        prediction_method = (
                            "nominal_physics"
                            if prediction.method == "physics_only"
                            else prediction.method
                        )
                        normalized_prediction = PredictiveDistribution(
                            method=prediction_method,
                            mean=prediction.mean,
                            variance=prediction.variance,
                            interval_lower=prediction.interval_lower,
                            interval_upper=prediction.interval_upper,
                        )
                        _append_intervention_row(
                            intervention_rows,
                            seed=seed,
                            target=target,
                            episode=episode,
                            setting=setting,
                            prediction=normalized_prediction,
                            start_frame=start_frame,
                            source_objects=source_names,
                            calibration=calibration,
                            contact_config=latent_cfg,
                            benchmark_config=cfg,
                        )

                for setting, joint_weights, contact_weights in (
                    ("pre_intervention", pre_weights, pre_contact_weights),
                    ("online_adaptation", online_weights, online_contact_weights),
                ):
                    recovery_rows.append(
                        {
                            "seed": seed,
                            "object": target.protocol.graph_object.name,
                            "held_out_topology": True,
                            "source_objects": ";".join(source_names),
                            "world_condition": episode.condition.name,
                            "setting": setting,
                            "observation_fraction": (
                                latent_cfg.observation_fraction
                                if setting == "online_adaptation"
                                else 0.0
                            ),
                            **contact_recovery_metrics(
                                bank.contact_states,
                                contact_weights,
                                true_state,
                                confidence_level=latent_cfg.confidence_level,
                            ),
                            **_joint_parameter_metrics(bank, joint_weights),
                        }
                    )

    intervention_aggregate = aggregate_latent_interventions(intervention_rows)
    contact_aggregate = aggregate_contact_recovery(recovery_rows)
    object_gap_closure = _aggregate_object_gap_closure(intervention_rows)
    success_gates = evaluate_success_gates(
        intervention_aggregate,
        contact_aggregate,
        object_gap_closure,
        latent_cfg,
    )
    return {
        "schema_version": 1,
        "benchmark": "causal4d-latent-contact-v1",
        "seeds": normalized_seeds,
        "benchmark_config": cfg.as_dict(),
        "contact_config": latent_cfg.as_dict(),
        "protocol": {
            "base": protocol_manifest(protocols, cfg),
            "contact_model": {
                "transfer": "leave-one-topology-out",
                "source_supervision": (
                    "matched and shifted test-action contacts from the two source "
                    "topologies fit the action-conditioned prior, likelihood, "
                    "temperature, and interval scales"
                ),
                "target_exclusion": (
                    "the target object, contact labels, prefixes, and future "
                    "trajectories are excluded from all fitting and calibration"
                ),
                "model_inputs": [
                    "object graph",
                    "rest geometry",
                    "commanded forces",
                    "nominal contact nodes",
                    "early state prefix in online setting",
                ],
                "evaluator_only": [
                    "realized contact nodes",
                    "gain multiplier",
                    "delay",
                    "slip spread",
                    "control-frame rotation",
                ],
                "controls": {
                    "nominal_physics": (
                        "fix nominal contact; update theta from the same prefix online"
                    ),
                    "oracle_contact": (
                        "expose true contact; update theta from the same prefix online"
                    ),
                    "oracle_contact_theta": (
                        "expose true contact and true simulated theta as the strict ceiling"
                    ),
                },
            },
        },
        "interventions": intervention_rows,
        "contact_recovery": recovery_rows,
        "fold_calibration": calibration_rows,
        "aggregate": {
            "interventions": intervention_aggregate,
            "contact_recovery": contact_aggregate,
            "held_out_topology": object_gap_closure,
        },
        "success_gates": success_gates,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty artifact: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_latent_contact_artifacts(
    result: dict[str, Any], output_directory: str | Path
) -> dict[str, str]:
    """Write deterministic predictions, recovery, calibration, and gate evidence."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output / "summary.json",
        "protocol": output / "protocol.json",
        "interventions": output / "interventions.csv",
        "contact_recovery": output / "contact_recovery.csv",
        "fold_calibration": output / "fold_calibration.csv",
        "success_gates": output / "success_gates.json",
    }
    _write_json(
        paths["summary"],
        {
            "schema_version": result["schema_version"],
            "benchmark": result["benchmark"],
            "seeds": result["seeds"],
            "benchmark_config": result["benchmark_config"],
            "contact_config": result["contact_config"],
            "aggregate": result["aggregate"],
            "success_gates": result["success_gates"],
        },
    )
    _write_json(paths["protocol"], result["protocol"])
    _write_csv(paths["interventions"], result["interventions"])
    _write_csv(paths["contact_recovery"], result["contact_recovery"])
    _write_csv(paths["fold_calibration"], result["fold_calibration"])
    _write_json(paths["success_gates"], result["success_gates"])

    manifest_path = output / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "benchmark": result["benchmark"],
            "artifacts": {
                path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
                for path in paths.values()
            },
        },
    )
    return {
        **{name: str(path) for name, path in paths.items()},
        "manifest": str(manifest_path),
    }
