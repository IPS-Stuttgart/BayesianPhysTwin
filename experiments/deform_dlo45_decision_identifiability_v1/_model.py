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
            width = np.linalg.norm(np.max(members, axis=0) - np.min(members, axis=0))
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
    selected = selected[
        np.lexsort((selected, distance[selected]))
    ]
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
        weiYÚÏZÙ\›™[İÙZYÚËˆZ[›[™İXÛ\Ü×ØÛİ[ˆ
K˜\İ\Jœ™›Ø]
BˆÛ\Ü×ÜÚ^™\ÈHœ˜š[˜Ûİ[
Û\ÜÙ\ËZ[›[™İXÛ\Ü×ØÛİ[
K˜\İ\Jœ™›Ø]
Bˆ™Y™œ™^WİÙZYÚÈH][İY[ØÛ\ÜÙ\×HÈÛ\Ü×ÜÚ^™\ÖØÛ\ÜÙ\×BˆÙ[XİYÜ™\ÚYX[ÈH[Ù[œ™\ÚYX[ÖÜÙ[XİYBˆÛÜœ™Xİ[ÛˆHœ™Z[œİ[JšKYO™‹™Y™œ™^WİÙZYÚËÙ[XİYÜ™\ÚYX[ÊBˆXİ[ÛœÈH[Ù[˜Xİ[Û—ÜØØ[\ÖÎ‹›Û™WH
ˆÛÜœ™Xİ[Û–Ó›Û™K—Bˆ˜]×ÛÜÜÙ\ÈHœ›YX[ŠˆœœÜ]X\™JÙ[XİYÜ™\ÚYX[ÖÎ‹›Û™K—HHXİ[ÛœÖÓ›Û™K‹—JKˆ^\ÏL‹ˆ
Bˆ™[]]™WÛÜÜÙ\ÈH˜]×ÛÜÜÙ\ÈÈ
ˆ˜]×ÛÜÜÙ\ÖÎ‹ŒWH
È[Ù[›ÜÜ×Ù›ÛÜ‚ˆ
Bˆš[ÜˆHœ™[
™ZYÚ›Ü—ØÛİ[KŒÈ™ZYÚ›Ü—ØÛİ[
BˆÙ\YšXØ]HH]Y\WÙXÚ\Ú[Û—ØÙ\YšXØ]Jˆš[Ü‹ˆ][İY[ˆÛ\ÜÙ\Ëˆ™[]]™WÛÜÜÙ\Ëˆ™YÜ™]İÛ\˜[˜ÙO[[Ù[œ™YÜ™]İÛ\˜[˜ÙKˆ
BˆÙ\YšXØ]WØXİ[ÛˆH
ˆÙ\YšXØ]K›Z[š[X^ØXİ[Û—Ú[™^ˆYˆÙ\YšXØ]K›Z[š[X^İÛÜœİØØ\ÙWÜ™YÜ™]ˆH[Ù[œ™YÜ™]İÛ\˜[˜ÙH
ÈUÓˆ[ÙHˆ
Bˆ™Y™œ™^WØXİ[ÛˆH[
ˆœ˜\™ÛZ[Šœ™Z[œİ[JšKXKO˜H‹™Y™œ™^WİÙZYÚË™[]]™WÛÜÜÙ\ÊJBˆ
BˆÙ\›™[ØXİ[ÛˆH[
ˆœ˜\™ÛZ[Šœ™Z[œİ[JšKXKO˜H‹Ù\›™[İÙZYÚË™[]]™WÛÜÜÙ\ÊJBˆ
BˆX\ØXİ[ÛˆH[
œ˜\™ÛZ[Š™[]]™WÛÜÜÙ\ÖÌJJBˆ™]\›ˆXÚ\Ú[ÛŠˆÙ\YšXØ]WØXİ[ÛXÙ\YšXØ]WØXİ[Û‹ˆ™Y™œ™^WØXİ[ÛZ™Y™œ™^WØXİ[Û‹ˆÙ\›™[ØXİ[ÛZÙ\›™[ØXİ[Û‹ˆX\ØXİ[Û[X\ØXİ[Û‹ˆÛÜœ™Xİ[ÛXÛÜœ™Xİ[Û‹ˆÛÜœİØØ\ÙWÜ™YÜ™]XÙ\YšXØ]KÛÜœİØØ\ÙWÜ™YÜ™]ˆZ[š[X^Ü™YÜ™]XÙ\YšXØ]K›Z[š[X^İÛÜœİØØ\ÙWÜ™YÜ™]ˆ›Ø\İÛX\ÚÏXÙ\YšXØ]Kœ›Ø\İWÛÜ[X[ØXİ[Û—ÛX\ÚËˆÛ\˜[˜ÙWÛX\ÚÏXÙ\YšXØ]KÛ\˜[˜ÙWØYZ\ÜÚX›WØXİ[Û—ÛX\ÚËˆ[XšYİZ]WİÚYXÛ\Ü×Ø[XšYİZ]JˆÙ[XİYÜ™\ÚYX[ËˆÛ\ÜÙ\Ëˆ][İY[ˆ›İØÛÛˆ
Kˆ[œİ\ÜYÜÜXÚYšXÚ]WÛ˜]Ï][œİ\ÜYÜÜXÚYšXÚ]JˆÙ\›™[İÙZYÚËˆÛ\ÜÙ\Ëˆ][İY[ˆ
Kˆ™ZYÚ›Ü—ØÛİ[[™ZYÚ›Ü—ØÛİ[ˆ][İY[ØÛ\Ü×ØÛİ[XÛ\Ü×ØÛİ[ˆ
B‚‚