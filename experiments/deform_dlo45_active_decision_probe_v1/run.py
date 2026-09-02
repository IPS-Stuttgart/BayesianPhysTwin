"""Retrospective DEFORM active decision-probe-duration pilot.

A probe is the first ``p`` frames of the already registered endpoint-motion
carrier.  Its duration is selected before target internal-node response is read.
After executing that prefix, the observed internal response is quantized by a
source-only local outcome model and a precomputed contingent terminal action is
used to predict the internal nodes at the original 25-frame terminal horizon.

This is a mechanism pilot with motion-capture probe observations.  It is not a
counterfactual comparison of alternative physical probe directions.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from bayesian_phystwin.active_decision_probe_v1 import (
    ActiveDecisionProbeCertificateV1,
    DecisionProbeCandidateV1,
    decision_probe_candidate,
    select_minimum_cost_decision_probe,
)
from experiments.deform_dlo45_decision_identifiability_v1._common import (
    DLOS,
    INTERNAL,
    FloatArray,
    Model,
    Protocol,
    extract_observation,
    load_protocol,
    load_trajectory,
    trajectory_paths,
    window_starts,
)
from experiments.deform_dlo45_decision_identifiability_v1._model import (
    build_pool,
    deterministic_kmeans,
    fit_model,
)

IntArray = npt.NDArray[np.int64]

CONTRACT = "deform-dlo45-active-decision-probe-v1"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PilotProtocol:
    probe_frames: tuple[int, ...]
    outcome_count: int
    cluster_count: int
    neighbors: int
    temperature_scale: float
    regret_tolerance: float
    target_support_multiplier: float
    bootstrap_replicates: int
    bootstrap_seed: int


class LocalSupport(NamedTuple):
    selected: IntArray
    kernel_weights: FloatArray
    class_index: IntArray
    quotient_weights: FloatArray
    jeffrey_weights: FloatArray
    residuals: FloatArray


class ProbeBundle(NamedTuple):
    frames: int
    candidate: DecisionProbeCandidateV1
    corrections: FloatArray
    outcome_labels: IntArray
    signature_mean: FloatArray
    signature_scale: FloatArray
    standardized_centers: FloatArray
    support_radius: FloatArray
    nominal_outcome_entropy_bits: float


class OutcomeAssignment(NamedTuple):
    outcome: int
    squared_distance: float
    supported: bool
    compatible_hypothesis_count: int


def read_protocol(path: Path) -> PilotProtocol:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("contract") != CONTRACT
        or value.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("unsupported active-probe pilot protocol")
    model = value["source_model"]
    probe = value["probe"]
    bootstrap = value["bootstrap"]
    frames = tuple(int(item) for item in probe["frames"])
    protocol = PilotProtocol(
        probe_frames=frames,
        outcome_count=int(probe["outcome_count"]),
        cluster_count=int(model["response_quotient_classes"]),
        neighbors=int(model["neighbors"]),
        temperature_scale=float(model["temperature_scale"]),
        regret_tolerance=float(model["regret_tolerance"]),
        target_support_multiplier=float(probe["target_support_multiplier"]),
        bootstrap_replicates=int(bootstrap["replicates"]),
        bootstrap_seed=int(bootstrap["seed"]),
    )
    if (
        not frames
        or frames[0] != 0
        or tuple(sorted(set(frames))) != frames
        or protocol.outcome_count < 2
        or protocol.cluster_count < 1
        or protocol.neighbors < 1
        or protocol.temperature_scale <= 0.0
        or protocol.regret_tolerance < 0.0
        or protocol.target_support_multiplier < 1.0
    ):
        raise ValueError("invalid active-probe pilot protocol")
    return protocol


def local_support(feature: FloatArray, model: Model) -> LocalSupport:
    query = (feature - model.feature_mean) / model.feature_scale
    pool = (model.features - model.feature_mean) / model.feature_scale
    distance = np.mean(np.square(pool - query[None, :]), axis=1)
    neighbor_count = min(model.neighbors, len(distance))
    selected = np.argpartition(distance, neighbor_count - 1)[:neighbor_count]
    selected = selected[np.lexsort((selected, distance[selected]))]
    selected_distance = distance[selected]
    positive = selected_distance[selected_distance > 0.0]
    base_bandwidth = (
        float(np.median(positive))
        if len(positive)
        else max(float(np.mean(selected_distance)), 1e-12)
    )
    bandwidth = max(base_bandwidth * model.temperature_scale, 1e-12)
    logits = -(selected_distance - float(np.min(selected_distance))) / bandwidth
    kernel_weights = np.exp(logits)
    kernel_weights /= np.sum(kernel_weights)

    global_classes = model.class_labels[selected]
    unique_classes = np.unique(global_classes)
    remap = {int(value): index for index, value in enumerate(unique_classes)}
    classes = np.asarray(
        [remap[int(value)] for value in global_classes],
        dtype=np.int64,
    )
    class_count = len(unique_classes)
    quotient = np.bincount(
        classes,
        weights=kernel_weights,
        minlength=class_count,
    ).astype(np.float64)
    class_sizes = np.bincount(
        classes,
        minlength=class_count,
    ).astype(np.float64)
    jeffrey = quotient[classes] / class_sizes[classes]
    return LocalSupport(
        selected=np.asarray(selected, dtype=np.int64),
        kernel_weights=np.asarray(kernel_weights, dtype=np.float64),
        class_index=classes,
        quotient_weights=quotient,
        jeffrey_weights=jeffrey,
        residuals=np.asarray(model.residuals[selected], dtype=np.float64),
    )


def probe_signature(residual: FloatArray) -> FloatArray:
    if residual.ndim == 3:
        residual = residual[None, ...]
        squeeze = True
    elif residual.ndim == 4:
        squeeze = False
    else:
        raise ValueError("probe residual must have shape (T,N,3) or (H,T,N,3)")
    if residual.shape[1] < 1 or residual.shape[-1] != 3:
        raise ValueError("probe residual has invalid shape")
    mean_all = np.mean(residual, axis=(1, 2))
    final_mean = np.mean(residual[:, -1], axis=1)
    rms = np.sqrt(np.mean(np.square(residual), axis=(1, 2, 3)))[:, None]
    signature = np.concatenate((mean_all, final_mean, rms), axis=1)
    return signature[0] if squeeze else signature


def fit_probe_quantizer(
    signatures: FloatArray,
    outcome_count: int,
    iterations: int,
) -> tuple[IntArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    mean = np.mean(signatures, axis=0)
    scale = np.maximum(np.std(signatures, axis=0), 1e-9)
    standardized = (signatures - mean) / scale
    cluster_count = min(outcome_count, len(signatures))
    labels = deterministic_kmeans(signatures, cluster_count, iterations)
    centers = np.asarray(
        [
            np.mean(standardized[labels == cluster], axis=0)
            for cluster in range(cluster_count)
        ],
        dtype=np.float64,
    )
    assigned = centers[labels]
    squared_distance = np.mean(np.square(standardized - assigned), axis=1)
    radii = np.asarray(
        [
            float(np.max(squared_distance[labels == cluster]))
            for cluster in range(cluster_count)
        ],
        dtype=np.float64,
    )
    return labels, mean, scale, centers, radii


def entropy_bits(probabilities: FloatArray) -> float:
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log2(positive)))


def build_probe_bundle(
    support: LocalSupport,
    model: Model,
    passive_protocol: Protocol,
    pilot_protocol: PilotProtocol,
    frames: int,
) -> ProbeBundle:
    hypothesis_count = len(support.selected)
    shaped = support.residuals.reshape(
        hypothesis_count,
        passive_protocol.horizon_frames,
        -1,
        3,
    )
    terminal = shaped[:, -1].reshape(hypothesis_count, -1)
    if frames == 0:
        labels = np.zeros(hypothesis_count, dtype=np.int64)
        signature_mean = np.zeros(7, dtype=np.float64)
        signature_scale = np.ones(7, dtype=np.float64)
        centers = np.zeros((1, 7), dtype=np.float64)
        radii = np.asarray([0.0], dtype=np.float64)
    else:
        signatures = probe_signature(shaped[:, :frames])
        labels, signature_mean, signature_scale, centers, radii = (
            fit_probe_quantizer(
                signatures,
                pilot_protocol.outcome_count,
                passive_protocol.kmeans_iterations,
            )
        )
    outcome_count = int(np.max(labels)) + 1
    likelihood = np.zeros(
        (hypothesis_count, outcome_count),
        dtype=np.float64,
    )
    likelihood[np.arange(hypothesis_count), labels] = 1.0

    corrections = np.zeros(
        (outcome_count, terminal.shape[1]),
        dtype=np.float64,
    )
    for outcome in range(outcome_count):
        nominal = support.jeffrey_weights * (labels == outcome)
        mass = float(np.sum(nominal))
        if mass <= 0.0:
            raise RuntimeError("occupied probe outcome has zero Jeffrey mass")
        nominal /= mass
        corrections[outcome] = np.einsum("i,id->d", nominal, terminal)

    action_scales = model.action_scales
    losses = np.empty(
        (hypothesis_count, outcome_count, len(action_scales)),
        dtype=np.float64,
    )
    fallback_mse = np.mean(np.square(terminal), axis=1)
    denominator = np.maximum(fallback_mse, model.loss_floor)
    for outcome in range(outcome_count):
        actions = action_scales[:, None] * corrections[outcome][None, :]
        losses[:, outcome, :] = (
            np.mean(
                np.square(terminal[:, None, :] - actions[None, :, :]),
                axis=2,
            )
            / denominator[:, None]
        )

    nominal_outcome = support.jeffrey_weights @ likelihood
    candidate = decision_probe_candidate(
        f"probe_{frames}",
        float(frames),
        likelihood,
        losses,
     )
    return ProbeBundle(
        frames=frames,
        candidate=candidate,
        corrections=corrections,
        outcome_labels=labels,
        signature_mean=signature_mean,
        signature_scale=signature_scale,
        standardized_centers=centers,
        support_radius=radii,
        nominal_outcome_entropy_bits=entropy_bits(nominal_outcome),
    )


def assign_outcome(
    actual_probe_residual: FloatArray,
    bundle: ProbeBundle,
    pilot_protocol: PilotProtocol,
) -> OutcomeAssignment:
    if bundle.frames == 0:
        return OutcomeAssignment(
            outcome=0,
            squared_distance=0.0,
            supported=True,
            compatible_hypothesis_count=len(bundle.outcome_labels),
        )
    signature = probe_signature(actual_probe_residual)
    standardized = (signature - bundle.signature_mean) / bundle.signature_scale
    distances = np.mean(
        np.square(bundle.standardized_centers - standardized[None, :]),
        axis=1,
    )
    outcome = int(np.argmin(distances))
    distance = float(distances[outcome])
    radius = float(bundle.support_radius[outcome])
    supported = bool(
        distance
        <= pilot_protocol.target_support_multiplier * max(radius, 1e-12)
    )
    return OutcomeAssignment(
        outcome=outcome,
        squared_distance=distance,
        supported=supported,
        compatible_hypothesis_count=int(
            np.count_nonzero(bundle.outcome_labels == outcome)
        ),
    )


def selected_action(
    bundle: ProbeBundle,
    certificate: ActiveDecisionProbeCertificateV1,
    assignment: OutcomeAssignment,
    *,
    require_certificate: bool,
    tolerance: float,
) -> int:
    if not assignment.supported:
        return 0
    if (
        require_certificate
        and certificate.minimax_worst_case_regret > tolerance + 1e-12
    ):
        return 0
    return int(certificate.minimax_terminal_policy[assignment.outcome])


def terminal_mse(
    normalized_terminal_residual: FloatArray,
    bundle: ProbeBundle,
    assignment: OutcomeAssignment,
    action: int,
    model: Model,
    length_scale: float,
) -> float:
    correction = bundle.corrections[assignment.outcome]
    predicted = model.action_scales[action] * correction
    return float(
        np.mean(np.square(normalized_terminal_residual - predicted))
        * length_scale**2
    )


def fit_source_model(
    train_paths: tuple[Path, ...],
    passive_protocol: Protocol,
    pilot_protocol: PilotProtocol,
) -> Model:
    names = tuple(path.name for path in train_paths)
    features, residuals, _ = build_pool(
        train_paths,
        names,
        passive_protocol,
    )
    return fit_model(
        features,
        residuals,
        cluster_count=pilot_protocol.cluster_count,
        neighbors=pilot_protocol.neighbors,
        temperature_scale=pilot_protocol.temperature_scale,
        regret_tolerance=pilot_protocol.regret_tolerance,
        protocol=passive_protocol,
    )


def bootstrap_interval(
    values: FloatArray,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, len(values), size=len(values))
        estimates[index] = float(np.mean(values[sample]))
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def method_summary(
    squared_error: list[float],
    fallback_error: list[float],
    probe_frames: list[int],
    actions: list[int],
    action_count: int,
) -> dict[str, object]:
    values = np.asarray(squared_error, dtype=np.float64)
    fallback = np.asarray(fallback_error, dtype=np.float64)
    action_array = np.asarray(actions, dtype=np.int64)
    rmse = math.sqrt(float(np.mean(values)))
    fallback_rmse = math.sqrt(float(np.mean(fallback)))
    return {
        "terminal_rmse_mm": 1000.0 * rmse,
        "rmse_ratio_to_fallback": rmse / max(fallback_rmse, 1e-12),
        "harm_fraction_vs_fallback": float(
            np.mean(values > fallback + 1e-12)
        ),
        "mean_probe_frames": float(np.mean(probe_frames)),
        "action_counts": np.bincount(
            action_array,
            minlength=action_count,
        ).tolist(),
        "fallback_action_fraction": float(np.mean(action_array == 0)),
    }


def evaluate_dlo(
    dataset_root: Path,
    dlo: str,
    passive_protocol: Protocol,
    pilot_protocol: PilotProtocol,
) -> dict[str, object]:
    train_paths = trajectory_paths(dataset_root, dlo, "train")
    eval_paths = trajectory_paths(dataset_root, dlo, "eval")
    model = fit_source_model(train_paths, passive_protocol, pilot_protocol)
    methods = (
        "fallback",
        "no_probe_certificate",
        "active_minimum_cost",
        *(
            f"fixed_probe_{frames}"
            for frames in pilot_protocol.probe_frames
            if frames > 0
        ),
        "max_outcome_entropy",
        "oracle_probe_action",
    )
    squared_error = {name: [] for name in methods}
    probe_cost = {name: [] for name in methods}
    actions = {name: [] for name in methods}
    per_trajectory: list[dict[str, object]] = []
    duration_counts = {
        str(frames): 0 for frames in pilot_protocol.probe_frames
    }
    duration_counts["fallback_no_certified_probe"] = 0
    no_probe_pass = 0
    selected_pass = 0
    selected_oos = 0
    probe_required = 0
    probed_state_ambiguous = 0
    before_regret: list[float] = []
    after_regret: list[float] = []
    selected_support_count: list[int] = []
    selected_probe_distance: list[float] = []
    decision_count = 0

    for path in eval_paths:
        trajectory = load_trajectory(path)
        local_error = {name: [] for name in methods}
        for current in window_starts(passive_protocol):
            observation = extract_observation(
                trajectory,
                current,
                passive_protocol,
            )
            support = local_support(observation.feature, model)
            bundles = tuple(
                build_probe_bundle(
                    support,
                    model,
                    passive_protocol,
                    pilot_protocol,
                    frames,
                )
                for frames in pilot_protocol.probe_frames
            )
            prior = np.full(
                len(support.selected),
                1.0 / len(support.selected),
                dtype=np.float64,
            )
            selection = select_minimum_cost_decision_probe(
                prior,
                support.quotient_weights,
                support.class_index,
                tuple(bundle.candidate for bundle in bundles),
                regret_tolerance=pilot_protocol.regret_tolerance,
            )
            certificates = selection.certificates
            before_regret.append(
                certificates[0].minimax_worst_case_regret
            )
            no_probe_pass += int(
                certificates[0].minimax_worst_case_regret
                <= pilot_protocol.regret_tolerance + 1e-12
            )

            # The active duration is selected before any target internal-node
            # response from the probe or terminal horizon is sliced.
            if selection.selected_probe_index is None:
                active_index = 0
                active_is_certified = False
                duration_counts["fallback_no_certified_probe"] += 1
            else:
                active_index = selection.selected_probe_index
                active_is_certified = True
                selected_pass += 1
                duration_counts[str(bundles[active_index].frames)] += 1
            active_frames = bundles[active_index].frames
            probe_required += int(active_is_certified and active_frames > 0)

            max_probe = max(pilot_protocol.probe_frames)
            probe_truth = trajectory[
                current + 1 : current + 1 + max_probe,
                INTERNAL,
                :,
            ].copy()
            probe_residual = (
                probe_truth - observation.baseline[:max_probe]
            ) / observation.length_scale

            assignments: list[OutcomeAssignment] = []
            for bundle in bundles:
                assignments.append(
                    assign_outcome(
                        probe_residual[: bundle.frames],
                        bundle,
                        pilot_protocol,
                    )
                )

            chosen: dict[str, tuple[int, int]] = {
                "fallback": (0, 0),
            }
            no_assignment = assignments[0]
            no_action = selected_action(
                bundles[0],
                certificates[0],
                no_assignment,
                require_certificate=True,
                tolerance=pilot_protocol.regret_tolerance,
            )
            chosen["no_probe_certificate"] = (0, no_action)

            active_assignment = assignments[active_index]
            active_action = (
                selected_action(
                    bundles[active_index],
                    certificates[active_index],
                    active_assignment,
                    require_certificate=True,
                    tolerance=pilot_protocol.regret_tolerance,
                )
                if active_is_certified
                else 0
            )
            if not active_assignment.supported:
                selected_oos += int(active_frames > 0)
            chosen["active_minimum_cost"] = (
                active_index,
                active_action,
            )
            selected_support_count.append(
                active_assignment.compatible_hypothesis_count
            )
            selected_probe_distance.append(
                active_assignment.squared_distance
            )
            if (
                active_is_certified
                and active_frames > 0
                and active_assignment.supported
                and active_assignment.compatible_hypothesis_count > 1
            ):
                probed_state_ambiguous += 1
            after_regret.append(
                certificates[active_index].minimax_worst_case_regret
            )

            for index, bundle in enumerate(bundles):
                if bundle.frames == 0:
                  continue
                method = f"fixed_probe_{bundle.frames}"
                action = selected_action(
                    bundle,
                    certificates[index],
                    assignments[index],
                    require_certificate=True,
                    tolerance=pilot_protocol.regret_tolerance,
                )
                chosen[method] = (index, action)

            entrop