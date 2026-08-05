"""Deterministic plotting primitives for the PokeFlex paper video."""

from __future__ import annotations

from typing import Any

import numpy as np


def canonical_projection(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(vertices, axis=0)
    _, _, right = np.linalg.svd(vertices - center, full_matrices=False)
    basis = right[:2].T
    for column in range(2):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    return center, basis


def _subsample(array: np.ndarray, maximum_count: int) -> np.ndarray:
    if len(array) <= maximum_count:
        return array
    indices = np.linspace(0, len(array) - 1, maximum_count, dtype=np.int64)
    return array[indices]


def _project(vertices: np.ndarray, center: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (vertices - center) @ basis


def _distance_mm(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(target, dtype=np.float64))
    distance = tree.query(np.asarray(prediction, dtype=np.float64), k=1)[0]
    return 1000.0 * np.asarray(distance, dtype=np.float64)


def render_rgb_frame(
    *,
    take_id: str,
    frame_index: int,
    frame_count: int,
    target_frame: int,
    target_vertices: np.ndarray,
    baseline_vertices: np.ndarray,
    guarded_vertices: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    limits: tuple[float, float, float, float],
    distance_limit_mm: float,
    decisions: list[dict[str, Any]],
) -> np.ndarray:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(16.0, 9.0), dpi=120, facecolor="white")
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=(0.78, 0.22),
        left=0.045,
        right=0.965,
        top=0.90,
        bottom=0.09,
        hspace=0.17,
        wspace=0.08,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    timeline = figure.add_subplot(grid[1, :])

    target_sample = _subsample(target_vertices, 4500)
    baseline_sample = _subsample(baseline_vertices, 3500)
    guarded_sample = _subsample(guarded_vertices, 3500)
    target_projected = _project(target_sample, center, basis)
    baseline_projected = _project(baseline_sample, center, basis)
    guarded_projected = _project(guarded_sample, center, basis)
    baseline_distance = _distance_mm(baseline_sample, target_vertices)
    guarded_distance = _distance_mm(guarded_sample, target_vertices)
    x_min, x_max, y_min, y_max = limits

    axes[0].scatter(
        target_projected[:, 0],
        target_projected[:, 1],
        s=2.2,
        color="#444444",
        alpha=0.82,
        linewidths=0.0,
    )
    axes[0].set_title("Volumetric target mesh", fontsize=12.0, fontweight="bold")
    latest_scatter = None
    for axis, projected, distance, title in (
        (axes[1], baseline_projected, baseline_distance, "Released checkpoint"),
        (axes[2], guarded_projected, guarded_distance, "Frozen regret guard"),
    ):
        axis.scatter(
            target_projected[:, 0],
            target_projected[:, 1],
            s=1.7,
            color="#c8c8c8",
            alpha=0.38,
            linewidths=0.0,
        )
        latest_scatter = axis.scatter(
            projected[:, 0],
            projected[:, 1],
            c=distance,
            s=3.2,
            cmap="magma",
            vmin=0.0,
            vmax=distance_limit_mm,
            alpha=0.90,
            linewidths=0.0,
        )
        axis.set_title(title, fontsize=12.0, fontweight="bold")
    for axis in axes:
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#d4d4d4")
    if latest_scatter is not None:
        colorbar = figure.colorbar(
            latest_scatter,
            ax=axes,
            fraction=0.018,
            pad=0.012,
            aspect=35,
        )
        colorbar.set_label("Nearest target distance [mm]", fontsize=9.0)

    current = decisions[frame_index]
    baseline_error = np.asarray(
        [value["baseline_error_mm"] for value in decisions], dtype=np.float64
    )
    guarded_error = np.asarray(
        [value["selected_error_mm"] for value in decisions], dtype=np.float64
    )
    source_frames = np.asarray(
        [value["target_frame"] for value in decisions], dtype=np.int64
    )
    timeline.plot(
        source_frames,
        baseline_error,
        color="#a1a7b1",
        linewidth=1.25,
        label="Released checkpoint",
    )
    timeline.plot(
        source_frames,
        guarded_error,
        color="#087e8b",
        linewidth=1.65,
        label="Regret guarded",
    )
    timeline.scatter(
        source_frames[
            np.asarray([value["accepted"] for value in decisions], dtype=bool)
        ],
        guarded_error[
            np.asarray([value["accepted"] for value in decisions], dtype=bool)
        ],
        s=12,
        color="#087e8b",
        zorder=3,
        label="Accepted update",
    )
    timeline.axvline(target_frame, color="#222222", linewidth=1.15)
    timeline.set_xlabel("PokeFlex target frame")
    timeline.set_ylabel("CD$_{UL1}$ [mm]")
    timeline.grid(alpha=0.18, linewidth=0.6)
    timeline.legend(loc="upper left", ncol=3, fontsize=8.3)

    accepted = bool(current["accepted"])
    delta = float(current["baseline_error_mm"] - current["selected_error_mm"])
    status = "GUARDED UPDATE" if accepted else "EXACT CHECKPOINT FALLBACK"
    status_color = "#087e8b" if accepted else "#555555"
    result_color = "#087e8b" if delta >= 0.0 else "#c0362c"
    figure.suptitle(
        f"PokeFlex {take_id} — prospective frame {target_frame}",
        fontsize=17.0,
        fontweight="bold",
    )
    figure.text(
        0.50,
        0.935,
        status,
        ha="center",
        va="center",
        fontsize=11.0,
        color=status_color,
        fontweight="bold",
    )
    bound = current["selector_adjusted_upper_regret_mm"]
    bound_text = "unsupported" if bound is None else f"{float(bound):+.3f} mm"
    figure.text(
        0.50,
        0.045,
        (
            f"Released {float(current['baseline_error_mm']):.3f} mm  |  "
            f"Guarded {float(current['selected_error_mm']):.3f} mm  |  "
            f"Improvement {delta:+.3f} mm  |  Frozen upper regret {bound_text}"
        ),
        ha="center",
        va="center",
        fontsize=10.2,
        color=result_color,
        fontweight="bold",
    )
    figure.text(
        0.985,
        0.013,
        f"visual frame {frame_index + 1}/{frame_count}",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#777777",
    )
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()
    rgba = np.asarray(figure.canvas.buffer_rgba()).reshape(height, width, 4)
    rgb = np.asarray(rgba[:, :, :3], dtype=np.uint8).copy()
    plt.close(figure)
    return rgb


def video_limits(
    captured: dict[int, dict[str, np.ndarray]],
    center: np.ndarray,
    basis: np.ndarray,
) -> tuple[tuple[float, float, float, float], float]:
    projected = []
    sampled_distances = []
    frames = sorted(captured)
    for index, frame in enumerate(frames):
        current = captured[frame]
        for key in (
            "target_vertices_m",
            "baseline_vertices_m",
            "guarded_vertices_m",
        ):
            sample = _subsample(current[key], 1200)
            projected.append(_project(sample, center, basis))
        if index % 6 == 0 or index == len(frames) - 1:
            target = current["target_vertices_m"]
            for key in ("baseline_vertices_m", "guarded_vertices_m"):
                sample = _subsample(current[key], 800)
                sampled_distances.extend(_distance_mm(sample, target).tolist())
    merged = np.concatenate(projected, axis=0)
    x_min, y_min = np.min(merged, axis=0)
    x_max, y_max = np.max(merged, axis=0)
    span = max(float(x_max - x_min), float(y_max - y_min), 1e-4)
    padding = 0.075 * span
    limits = (
        float(x_min - padding),
        float(x_max + padding),
        float(y_min - padding),
        float(y_max + padding),
    )
    distance_limit = float(np.quantile(sampled_distances, 0.96))
    return limits, max(4.0, min(distance_limit, 20.0))
