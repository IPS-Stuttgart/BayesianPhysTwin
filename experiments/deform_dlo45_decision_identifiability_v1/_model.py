"""Source quotient construction and exact finite-action decisions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bayesian_phystwin.query_decision_certificate_v1 import (
    query_decision_certificate,
)

from ._common import (
    ATOL,
    INTERNAL,
    NODE_COUNT,
    Decision,
    FloatArray,
    IntArray,
    Model,
    Protocol,
    load_trajectory,
    source_window,
    window_starts,
)


def build_pool(
    paths: tuple[Path, ...],
    names: tuple[str, ...],
    protocol: Protocol,
) -> tuple[FloatArray, FloatArray, IntArray]:
    wanted = set(names)
    starts = window_starts(protocol)
    features: list[FloatArray] = []
    residuals: list[FloatArray] = []
    groups: list[int] = []
    group_id = 0
    for path in paths:
        if path.name not in wanted:
            continue
        trajectory = load_trajectory(path)
        for current in starts:
            feature, residual = source_window(trajectory, current, protocol)
            features.append(feature)
            residuals.append(residual)
            groups.append(group_id)
        group_id += 1
    if group_id != len(names):
        raise ValueError("source pool did not contain every requested trajectory")
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(residuals, dtype=np.float64),
        np.asarray(groups, dtype=np.int64),
    )


def response_signature(residuals: FloatArray, protocol: Protocol) -> FloatArray:
    reshaped = residuals.reshape(
        residuals.shape[0],
        protocol.horizon_frames,
        NODE_COUNT - 4,
        3,
    )
    mean_all = np.mean(reshaped, axis=(1, 2))
    final_mean = np.mean(reshaped[:, -1], axis=1)
    rms = np.sqrt(np.mean(np.square(reshaped), axis=(1, 2, 3)))[:, None]
    return np.concatenate((mean_all, final_mean, rms), axis=1)


def deterministic_kmeans(
    values: FloatArray,
    cluster_count: int,
    iterations: int,
) -> IntArray:
    if values.ndim != 2 or len(values) < cluster_count or cluster_count < 1:
        raise ValueError("invalid k-means inputs")
    mean = np.mean(values, axis=0)
    scale = np.maximum(np.std(values, axis=0), 1e-9)
    standardized = (values - mean) / scale
    first = int(np.argmax(np.sum(np.square(standardized), axis=1)))
    centers = [standardized[first]]
    minimum_distance = np.sum(
        np.square(standardized - centers[0][None, :]), axis=1
    )
    for _ in range(1, cluster_count):
        index = int(np.argmax(minimum_distance))
        centers.append(standardized[index])
        distance = np.sum(
            np.square(standardized - centers[-1][None, :]), axis=1
        )
        minimum_distance = np.minimum(minimum_distance, distance)
    center_array = np.asarray(centers)
    labels = np.zeros(len(values), dtype=np.int64)
    for _ in range(iterations):
        distances = np.mean(
            np.square(standardized[:, None, :] - center_array[None, :, :]),
            axis=2,
        )
        updated = np.argmin(distances, axis=1).astype(np.int64)
        if np.array_equal(updated, labels):
            break
        labels = updated
        for cluster in range(cluster_count):
            members = standardized[labels == cluster]
            if len(members):
                center_array[cluster] = np.mean(members, axis=0)
    occupied = np.unique(labels)
    remap = {int(label): index for index, label in enumerate(occupied.tolist())}
    return np.asarray([remap[int(label)] for label in labels], dtype=np.int64)


def fit_model(
    features: FloatArray,
    residuals: FloatArray,
    *,
    cluster_count: int,
    neighbors: int,
    temperature_scale: float,
    regret_tolerance: float,
    protocol: Protocol,
) -> Model:
    feature_mean = np.mean(features, axis=0)
    feature_scale = np.maximum(np.std(features, axis=0), 1e-9)
    classes = deterministic_kmeans(
        response_signature(residuals, protocol),
        cluster_count,
        protocol.kmeans_iterations,
    )
    fallback_losses = np.mean(np.square(residuals), axis=1)
    loss_floor = max(float(np.quantile(fallback_losses, 0.05)) * 0.1, 1e-12)
    return Model(
        features=features,
        residuals=residuals,
        class_labels=classes,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        loss_floor=loss_floor,
        neighbors=neighbors,
        temperature_scale=temperature_scale,
        regret_tolerance=regret_tolerance,
        action_scales=np.asarray(protocol.action_scales, dtype=np.float64),
    )


def class_ambiguity(
    residuals: FloatArray,
    class_index: IntArray,
    quotient: FloatArray,
    protocol: Protocol,
) -> float:
    shaped = residuals.reshape(
        len(residuals),
        protocol.horizon_frames,
        NODE_COUNT - 4,
        3,
    )
    endpoint = np.mean(shaped[:, -1], axis=1)
    total = 0.0
    for class_id, weight in enumerate(quotient):
        members = endpoint[class_index == class_id]
        if len(members):
            width = np.linalg.norm(
                np.max(members, axis=0) - np.min(members, axis=0)
            )
            total += float(weight) * float(width)
    return total


def unsupported_specificity(
    weights: FloatArray,
    class_index: IntArray,
    quotient: FloatArray,
) -> float:
    result = 0.0
    for class_id, class_mass in enumerate(quotient):
        members = np.flatnonzero(class_index == class_id)
        if class_mass <= 0.0 or not len(members):
            continue
        conditional = weights[members] / class_mass
        prior_conditional = 1.0 / len(members)
        positive = conditional > 0.0
        result += float(class_mass) * float(
            np.sum(
                conditional[positive]
                * np.log(conditional[positive] / prior_conditional)
            )
        )
    return result


def decide(feature: FloatArray, model: Model, protocol: Protocol) -> Decision:
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
        [remap[int(value)] for value in global_classes], dtype=np.int64
    )
    class_count = len(unique_classes)
    quotient = np.bincount(
        classes,
        weights=kernel_weights,
        minlength=class_count,
    ).astype(np.float64)
    class_sizes = np.bincount(classes, minlength=class_count).astype(np.float64)
    jeffrey_weights = quotient[classes] / class_sizes[classes]
    selected_residuals = model.residuals[selected]
    correction = np.einsum("i,id->d", jeffrey_weights, selected_residuals)
    actions = model.action_scales[:, None] * correction[None, :]
    raw_losses = np.mean(
        np.square(selected_residuals[:, None, :] - actions[None, :, :]),
        axis=2,
    )
    relative_losses = raw_losses / (raw_losses[:, :1] + model.loss_floor)
    prior = np.full(neighbor_count, 1.0 / neighbor_count)
    certificate = query_decision_certificate(
        prior,
        quotient,
        classes,
        relative_losses,
        regret_tolerance=model.regret_tolerance,
    )
    certificate_action = (
        certificate.minimax_action_index
        if certificate.minimax_worst_case_regret
        <= model.regret_tolerance + ATOL
        else 0
    )
    jeffrey_action = int(
        np.argmin(np.einsum("i,ia->a", jeffrey_weights, relative_losses))
    )
    kernel_action = int(
        np.argmin(np.einsum("i,ia->a", kernel_weights, relative_losses))
    )
    map_action = int(np.argmin(relative_losses[0]))
    return Decision(
        certificate_action=certificate_action,
        jeffrey_action=jeffrey_action,
        kernel_action=kernel_action,
        map_action=map_action,
        correction=correction,
        worst_case_regret=certificate.worst_case_regret,
        minimax_regret=certificate.minimax_worst_case_regret,
        robust_mask=certificate.robustly_optimal_action_mask,
        tolerance_mask=certificate.tolerance_admissible_action_mask,
        ambiguity_width=class_ambiguity(
            selected_residuals,
            classes,
            quotient,
            protocol,
        ),
        unsupported_specificity_nats=unsupported_specificity(
            kernel_weights,
            classes,
            quotient,
        ),
        neighbor_count=neighbor_count,
        quotient_class_count=class_count,
    )
