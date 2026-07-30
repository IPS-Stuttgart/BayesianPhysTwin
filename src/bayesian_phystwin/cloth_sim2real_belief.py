"""Guarded online readout correction for the Cloth Sim2Real benchmark.

The module deliberately keeps the real point clouds outside the physical
state. Prefix observations estimate an observation-space correction over the
released simulator mesh. A disjoint prefix interval decides whether that
correction is admitted; rejection returns the physical rollout unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .phystwin_graph_discrepancy import (
    graph_smoothed_discrepancy_posterior,
    normalized_spring_laplacian,
)
from .pseudo_measurements import PseudoMeasurementBatch
from .robust_likelihood import RobustLikelihoodConfig, robust_mixture_likelihood


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_points(value: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    _require(
        points.ndim == 2 and points.shape[1] == 3,
        f"{name} must have shape (N, 3)",
    )
    _require(len(points) >= 1, f"{name} is empty")
    _require(np.all(np.isfinite(points)), f"{name} contains non-finite values")
    return points


_PLY_SCALAR_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int16": "<i2",
    "uint16": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "int32": "<i4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def load_binary_little_endian_ply_xyz(path: str | Path) -> np.ndarray:
    """Load XYZ coordinates from the benchmark's Open3D binary PLY files."""

    source = Path(path)
    _require(source.is_file(), f"PLY file does not exist: {source}")
    with source.open("rb") as stream:
        first = stream.readline()
        _require(first == b"ply\n", f"{source} is not a PLY file")
        vertex_count: int | None = None
        vertex_properties: list[tuple[str, str]] = []
        current_element: str | None = None
        binary_little_endian = False
        while True:
            line = stream.readline()
            _require(bool(line), f"{source} has no end_header")
            try:
                fields = line.decode("ascii").strip().split()
            except UnicodeDecodeError as error:
                raise ValueError(f"{source} has a non-ASCII PLY header") from error
            if fields[:1] == ["format"]:
                binary_little_endian = fields[1:3] == [
                    "binary_little_endian",
                    "1.0",
                ]
            elif fields[:1] == ["element"]:
                _require(len(fields) == 3, f"{source} has an invalid element")
                current_element = fields[1]
                if current_element == "vertex":
                    vertex_count = int(fields[2])
            elif fields[:1] == ["property"] and current_element == "vertex":
                _require(
                    len(fields) == 3 and fields[1] != "list",
                    f"{source} has an unsupported vertex property",
                )
                scalar_type = _PLY_SCALAR_TYPES.get(fields[1])
                _require(
                    scalar_type is not None,
                    f"{source} has unsupported PLY type {fields[1]}",
                )
                vertex_properties.append((fields[2], scalar_type))
            elif fields == ["end_header"]:
                break
        _require(binary_little_endian, f"{source} is not binary little endian")
        _require(
            vertex_count is not None and vertex_count >= 1,
            f"{source} has no vertices",
        )
        names = {name for name, _ in vertex_properties}
        _require({"x", "y", "z"} <= names, f"{source} has no XYZ properties")
        values = np.fromfile(
            stream,
            dtype=np.dtype(vertex_properties),
            count=vertex_count,
        )
    _require(
        len(values) == vertex_count,
        f"{source} ended before all vertices were read",
    )
    points = np.column_stack((values["x"], values["y"], values["z"])).astype(
        np.float64,
        copy=False,
    )
    _require(np.all(np.isfinite(points)), f"{source} has non-finite XYZ values")
    points.setflags(write=False)
    return points


def sample_physical_rollout(
    simulator_vertices_m: np.ndarray,
    observed_frame_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a simulator rollout at the benchmark point-cloud cadence."""

    vertices = np.asarray(simulator_vertices_m, dtype=np.float64)
    _require(
        vertices.ndim == 3 and vertices.shape[2] == 3,
        "simulator_vertices_m must have shape (T, N, 3)",
    )
    _require(np.all(np.isfinite(vertices)), "simulator rollout is not finite")
    _require(observed_frame_count >= 2, "observed_frame_count must be at least two")
    _require(
        len(vertices) >= observed_frame_count,
        "simulator rollout is shorter than the observed sequence",
    )
    indices = np.linspace(
        0,
        len(vertices) - 1,
        observed_frame_count,
        dtype=np.int64,
    )
    sampled = vertices[indices].copy()
    sampled.setflags(write=False)
    indices.setflags(write=False)
    return sampled, indices


def mesh_edges_from_faces(faces: np.ndarray, node_count: int) -> np.ndarray:
    """Return deterministic unique undirected edges from triangular faces."""

    triangles = np.asarray(faces, dtype=np.int64)
    _require(
        triangles.ndim == 2 and triangles.shape[1] == 3 and len(triangles) >= 1,
        "faces must have nonempty shape (F, 3)",
    )
    _require(
        np.all((triangles >= 0) & (triangles < node_count)),
        "face index exceeds node_count",
    )
    edges = np.concatenate(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [0, 2]],
        ),
        axis=0,
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    _require(np.all(edges[:, 0] != edges[:, 1]), "mesh contains a self edge")
    edges.setflags(write=False)
    return edges


@dataclass(frozen=True)
class DenseCloudAssociation:
    """Per-vertex point-cloud pseudo-measurements and assignment uncertainty."""

    observed_points_m: np.ndarray
    variance_m2: np.ndarray
    prior_reliability: np.ndarray
    assignment_entropy: np.ndarray

    def __post_init__(self) -> None:
        observed = _finite_points(self.observed_points_m, "observed_points_m").copy()
        count = len(observed)
        variance = np.asarray(self.variance_m2, dtype=np.float64).copy()
        reliability = np.asarray(self.prior_reliability, dtype=np.float64).copy()
        entropy = np.asarray(self.assignment_entropy, dtype=np.float64).copy()
        _require(variance.shape == (count,), "variance_m2 shape changed")
        _require(reliability.shape == (count,), "prior_reliability shape changed")
        _require(entropy.shape == (count,), "assignment_entropy shape changed")
        _require(
            np.all(np.isfinite(variance)) and np.all(variance > 0.0),
            "association variance must be finite and positive",
        )
        _require(
            np.all(np.isfinite(reliability))
            and np.all((reliability > 0.0) & (reliability <= 1.0)),
            "prior reliability must lie in (0, 1]",
        )
        _require(
            np.all(np.isfinite(entropy))
            and np.all((entropy >= 0.0) & (entropy <= 1.0 + 1e-12)),
            "assignment entropy must lie in [0, 1]",
        )
        for name, value in (
            ("observed_points_m", observed),
            ("variance_m2", variance),
            ("prior_reliability", reliability),
            ("assignment_entropy", entropy),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def associate_dense_cloud(
    physical_points_m: np.ndarray,
    observed_cloud_m: np.ndarray,
    *,
    candidate_count: int = 4,
    sensor_std_m: float = 0.003,
    minimum_softmax_scale_m: float = 0.001,
    source_confidence: np.ndarray | None = None,
) -> DenseCloudAssociation:
    """Associate each mesh vertex with a local mixture in one observed cloud.

    Geometry determines assignment probabilities, but not prior reliability.
    The latter is copied only from an optional residual-independent source cue.
    Assignment ambiguity enters metric covariance through mixture spread.
    """

    try:
        from scipy.spatial import cKDTree
    except (ImportError, OSError) as error:
        raise RuntimeError("cloth point-cloud association requires scipy") from error

    physical = _finite_points(physical_points_m, "physical_points_m")
    observed = _finite_points(observed_cloud_m, "observed_cloud_m")
    _require(candidate_count >= 1, "candidate_count must be positive")
    _require(
        np.isfinite(sensor_std_m) and sensor_std_m > 0.0,
        "sensor_std_m must be positive",
    )
    _require(
        np.isfinite(minimum_softmax_scale_m) and minimum_softmax_scale_m > 0.0,
        "minimum_softmax_scale_m must be positive",
    )
    count = min(candidate_count, len(observed))
    distances, indices = cKDTree(observed).query(
        physical,
        k=count,
        p=2,
        workers=-1,
    )
    if count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    candidates = observed[indices]
    scale = np.maximum(np.median(distances, axis=1), minimum_softmax_scale_m)
    logits = -0.5 * np.square(distances / scale[:, None])
    logits -= np.max(logits, axis=1, keepdims=True)
    assignment = np.exp(logits)
    assignment /= np.sum(assignment, axis=1, keepdims=True)
    mixture_mean = np.sum(assignment[:, :, None] * candidates, axis=1)
    coordinate_spread = np.sum(
        assignment[:, :, None]
        * np.square(candidates - mixture_mean[:, None, :]),
        axis=1,
    )
    variance = sensor_std_m**2 + np.mean(coordinate_spread, axis=1)
    if count == 1:
        entropy = np.zeros(len(physical), dtype=np.float64)
    else:
        entropy = -np.sum(
            assignment * np.log(np.maximum(assignment, 1e-15)),
            axis=1,
        ) / np.log(count)
    if source_confidence is None:
        reliability = np.ones(len(physical), dtype=np.float64)
    else:
        reliability = np.asarray(source_confidence, dtype=np.float64).copy()
        _require(
            reliability.shape == (len(physical),),
            "source_confidence shape changed",
        )
        _require(
            np.all(np.isfinite(reliability))
            and np.all((reliability > 0.0) & (reliability <= 1.0)),
            "source_confidence must lie in (0, 1]",
        )
    return DenseCloudAssociation(
        observed_points_m=mixture_mean,
        variance_m2=variance,
        prior_reliability=reliability,
        assignment_entropy=entropy,
    )


@dataclass(frozen=True)
class ClothReadoutBeliefConfig:
    """Frozen settings for robust prefix fitting and admission."""

    candidate_count: int = 4
    sensor_std_m: float = 0.003
    shared_bias_std_m: float = 0.005
    model_discrepancy_std_m: float = 0.003
    forecast_process_std_m_per_sqrt_frame: float = 0.001
    outlier_variance_multiplier: float = 100.0
    effective_fit_frames: float = 4.0
    graph_prior_strengths: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    correction_scales: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    maximum_correction_m: float = 0.10
    minimum_validation_improvement: float = 0.02
    minimum_validation_win_fraction: float = 0.60
    maximum_validation_worst_ratio: float = 1.0
    covariance_probes: int = 64
    covariance_seed: int = 20260730

    def __post_init__(self) -> None:
        positive = (
            self.candidate_count,
            self.sensor_std_m,
            self.shared_bias_std_m,
            self.model_discrepancy_std_m,
            self.forecast_process_std_m_per_sqrt_frame,
            self.outlier_variance_multiplier,
            self.effective_fit_frames,
            self.maximum_correction_m,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "belief scales must be positive",
        )
        _require(
            self.outlier_variance_multiplier > 1.0,
            "outlier variance multiplier must exceed one",
        )
        _require(
            len(self.graph_prior_strengths) >= 1
            and all(
                np.isfinite(value) and value > 0.0
                for value in self.graph_prior_strengths
            ),
            "graph prior strengths must be positive",
        )
        _require(
            len(self.correction_scales) >= 1
            and all(
                np.isfinite(value) and 0.0 < value <= 1.0
                for value in self.correction_scales
            ),
            "correction scales must lie in (0, 1]",
        )
        _require(
            0.0 <= self.minimum_validation_improvement < 1.0,
            "minimum validation improvement must lie in [0, 1)",
        )
        _require(
            0.0 <= self.minimum_validation_win_fraction <= 1.0,
            "minimum validation win fraction must lie in [0, 1]",
        )
        _require(
            self.maximum_validation_worst_ratio >= 1.0,
            "maximum validation worst ratio must be at least one",
        )
        _require(self.covariance_probes >= 0, "covariance probes must be nonnegative")


@dataclass(frozen=True)
class ReadoutCandidateScore:
    """Disjoint-prefix validation score for one correction arm."""

    name: str
    mean_symmetric_l1_chamfer_m: float
    maximum_symmetric_l1_chamfer_m: float
    win_fraction: float


@dataclass(frozen=True)
class GuardedReadoutCorrection:
    """Selected prefix belief and exact-fallback decision."""

    accepted: bool
    reason: str
    selected_name: str
    correction_m: np.ndarray
    variance_m2: np.ndarray
    scores: tuple[ReadoutCandidateScore, ...]
    diagnostics: dict[str, object]

    def __post_init__(self) -> None:
        correction = _finite_points(self.correction_m, "correction_m").copy()
        variance = np.asarray(self.variance_m2, dtype=np.float64).copy()
        _require(variance.shape == correction.shape, "variance_m2 shape changed")
        _require(
            np.all(np.isfinite(variance)) and np.all(variance > 0.0),
            "readout variance must be finite and positive",
        )
        if not self.accepted:
            _require(
                np.array_equal(correction, np.zeros_like(correction)),
                "rejected correction must be exact zero",
            )
            _require(
                self.selected_name == "baseline",
                "rejected correction must select baseline",
            )
        correction.setflags(write=False)
        variance.setflags(write=False)
        object.__setattr__(self, "correction_m", correction)
        object.__setattr__(self, "variance_m2", variance)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


def symmetric_l1_chamfer_m(first_m: np.ndarray, second_m: np.ndarray) -> float:
    """Return the mean of both directed nearest-neighbour L1 distances."""

    try:
        from scipy.spatial import cKDTree
    except (ImportError, OSError) as error:
        raise RuntimeError("cloth Chamfer evaluation requires scipy") from error
    first = _finite_points(first_m, "first_m")
    second = _finite_points(second_m, "second_m")
    first_to_second = cKDTree(second).query(first, k=1, p=1, workers=-1)[0]
    second_to_first = cKDTree(first).query(second, k=1, p=1, workers=-1)[0]
    return float(0.5 * (np.mean(first_to_second) + np.mean(second_to_first)))


def directed_l1_chamfer_m(simulated_m: np.ndarray, observed_m: np.ndarray) -> float:
    """Return the benchmark's simulator-to-observation L1 distance."""

    try:
        from scipy.spatial import cKDTree
    except (ImportError, OSError) as error:
        raise RuntimeError("cloth Chamfer evaluation requires scipy") from error
    simulated = _finite_points(simulated_m, "simulated_m")
    observed = _finite_points(observed_m, "observed_m")
    distances = cKDTree(observed).query(
        simulated,
        k=1,
        p=1,
        workers=-1,
    )[0]
    return float(np.mean(distances))


def symmetric_l2_hausdorff_m(first_m: np.ndarray, second_m: np.ndarray) -> float:
    """Return the symmetric Euclidean Hausdorff distance."""

    try:
        from scipy.spatial import cKDTree
    except (ImportError, OSError) as error:
        raise RuntimeError("cloth Hausdorff evaluation requires scipy") from error
    first = _finite_points(first_m, "first_m")
    second = _finite_points(second_m, "second_m")
    first_to_second = cKDTree(second).query(first, k=1, p=2, workers=-1)[0]
    second_to_first = cKDTree(first).query(second, k=1, p=2, workers=-1)[0]
    return float(max(np.max(first_to_second), np.max(second_to_first)))


def _cap_correction(correction_m: np.ndarray, maximum_m: float) -> np.ndarray:
    correction = np.asarray(correction_m, dtype=np.float64).copy()
    norm = np.linalg.norm(correction, axis=1, keepdims=True)
    correction *= np.minimum(1.0, maximum_m / np.maximum(norm, 1e-15))
    return correction


def _fit_robust_vertex_observations(
    physical_fit_m: np.ndarray,
    observed_fit_clouds_m: Sequence[np.ndarray],
    config: ClothReadoutBeliefConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    fit = np.asarray(physical_fit_m, dtype=np.float64)
    _require(
        fit.ndim == 3 and fit.shape[2] == 3,
        "physical_fit_m must have shape (T, N, 3)",
    )
    _require(
        len(observed_fit_clouds_m) == len(fit) and len(fit) >= 2,
        "fit cloud count must match at least two physical frames",
    )
    residuals: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    reliabilities: list[np.ndarray] = []
    entropies: list[np.ndarray] = []
    for physical, cloud in zip(fit, observed_fit_clouds_m, strict=True):
        association = associate_dense_cloud(
            physical,
            cloud,
            candidate_count=config.candidate_count,
            sensor_std_m=config.sensor_std_m,
        )
        residuals.append(association.observed_points_m - physical)
        variances.append(association.variance_m2)
        reliabilities.append(association.prior_reliability)
        entropies.append(association.assignment_entropy)
    residual = np.stack(residuals)
    variance = np.stack(variances)
    prior_reliability = np.stack(reliabilities)
    temporal_center = np.median(residual, axis=0)
    batch = PseudoMeasurementBatch(
        observed=residual.reshape(-1, 3),
        predicted=np.broadcast_to(temporal_center, residual.shape).reshape(-1, 3),
        variance=variance.reshape(-1),
    )
    robust = robust_mixture_likelihood(
        batch,
        prior_reliability=prior_reliability.reshape(-1),
        config=RobustLikelihoodConfig(
            outlier_variance_multiplier=config.outlier_variance_multiplier,
            model_discrepancy_variance=config.model_discrepancy_std_m**2,
        ),
    ).posterior_inlier_probability.reshape(residual.shape[:2])
    weight = robust * prior_reliability / variance
    weight_sum = np.sum(weight, axis=0)
    mean = np.sum(weight[:, :, None] * residual, axis=0) / np.maximum(
        weight_sum[:, None],
        1e-15,
    )
    raw_effective_count = np.square(weight_sum) / np.maximum(
        np.sum(np.square(weight), axis=0),
        1e-15,
    )
    effective_count = np.minimum(raw_effective_count, config.effective_fit_frames)
    temporal_scatter = np.sum(
        weight * np.mean(np.square(residual - mean[None]), axis=2),
        axis=0,
    ) / np.maximum(weight_sum, 1e-15)
    assignment_variance = np.sum(weight * variance, axis=0) / np.maximum(
        weight_sum,
        1e-15,
    )
    posterior_variance = (
        config.shared_bias_std_m**2
        + (temporal_scatter + assignment_variance)
        / np.maximum(effective_count, 1.0)
    )
    diagnostics: dict[str, object] = {
        "fit_frame_count": len(fit),
        "node_count": fit.shape[1],
        "mean_assignment_entropy": float(np.mean(entropies)),
        "mean_posterior_inlier_probability": float(np.mean(robust)),
        "minimum_posterior_inlier_probability": float(np.min(robust)),
        "median_raw_effective_frame_count": float(
            np.median(raw_effective_count)
        ),
        "effective_frame_count_cap": config.effective_fit_frames,
        "prior_reliability_uses_state_innovation": False,
        "shared_bias_floor_preserved": True,
    }
    return mean, posterior_variance, diagnostics


def _candidate_corrections(
    observed_mean_m: np.ndarray,
    observed_variance_m2: np.ndarray,
    faces: np.ndarray,
    config: ClothReadoutBeliefConfig,
) -> tuple[dict[str, np.ndarray], object]:
    node_count = len(observed_mean_m)
    edges = mesh_edges_from_faces(faces, node_count)
    laplacian = normalized_spring_laplacian(node_count, edges)
    global_mean = np.median(observed_mean_m, axis=0)
    candidates: dict[str, np.ndarray] = {
        "baseline": np.zeros_like(observed_mean_m),
        "global": _cap_correction(
            np.broadcast_to(global_mean, observed_mean_m.shape),
            config.maximum_correction_m,
        ),
    }
    observed = np.ones(node_count, dtype=bool)
    for strength in config.graph_prior_strengths:
        posterior = graph_smoothed_discrepancy_posterior(
            observed_mean_m,
            observed_variance_m2,
            observed,
            laplacian,
            prior_strength=strength,
        )
        for scale in config.correction_scales:
            name = f"graph_l{strength:g}_s{scale:g}"
            candidates[name] = _cap_correction(
                scale * posterior.mean,
                config.maximum_correction_m,
            )
    return candidates, laplacian


def fit_guarded_readout_correction(
    physical_fit_m: np.ndarray,
    observed_fit_clouds_m: Sequence[np.ndarray],
    physical_validation_m: np.ndarray,
    observed_validation_clouds_m: Sequence[np.ndarray],
    faces: np.ndarray,
    *,
    config: ClothReadoutBeliefConfig | None = None,
) -> GuardedReadoutCorrection:
    """Fit on one prefix block and admit only on a disjoint prefix block."""

    cfg = config or ClothReadoutBeliefConfig()
    validation = np.asarray(physical_validation_m, dtype=np.float64)
    _require(
        validation.ndim == 3 and validation.shape[2] == 3,
        "physical_validation_m must have shape (T, N, 3)",
    )
    _require(
        len(observed_validation_clouds_m) == len(validation)
        and len(validation) >= 1,
        "validation cloud count must match physical frames",
    )
    observed_mean, observed_variance, diagnostics = (
        _fit_robust_vertex_observations(
            physical_fit_m,
            observed_fit_clouds_m,
            cfg,
        )
    )
    _require(
        validation.shape[1] == len(observed_mean),
        "fit and validation node counts differ",
    )
    candidates, laplacian = _candidate_corrections(
        observed_mean,
        observed_variance,
        faces,
        cfg,
    )
    baseline_frame_scores = np.asarray(
        [
            symmetric_l1_chamfer_m(physical, observed)
            for physical, observed in zip(
                validation,
                observed_validation_clouds_m,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    scores: list[ReadoutCandidateScore] = []
    frame_scores_by_name: dict[str, np.ndarray] = {}
    for name, correction in candidates.items():
        frame_scores = np.asarray(
            [
                symmetric_l1_chamfer_m(physical + correction, observed)
                for physical, observed in zip(
                    validation,
                    observed_validation_clouds_m,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
        frame_scores_by_name[name] = frame_scores
        scores.append(
            ReadoutCandidateScore(
                name=name,
                mean_symmetric_l1_chamfer_m=float(np.mean(frame_scores)),
                maximum_symmetric_l1_chamfer_m=float(np.max(frame_scores)),
                win_fraction=float(np.mean(frame_scores < baseline_frame_scores)),
            )
        )
    scores.sort(key=lambda score: (score.mean_symmetric_l1_chamfer_m, score.name))
    best = scores[0]
    baseline_mean = float(np.mean(baseline_frame_scores))
    improvement = (
        baseline_mean - best.mean_symmetric_l1_chamfer_m
    ) / max(baseline_mean, 1e-15)
    worst_ratio = best.maximum_symmetric_l1_chamfer_m / max(
        float(np.max(baseline_frame_scores)),
        1e-15,
    )
    accepted = (
        best.name != "baseline"
        and improvement >= cfg.minimum_validation_improvement
        and best.win_fraction >= cfg.minimum_validation_win_fraction
        and worst_ratio <= cfg.maximum_validation_worst_ratio
    )
    diagnostics.update(
        {
            "validation_frame_count": len(validation),
            "validation_baseline_mean_m": baseline_mean,
            "selected_validation_improvement": float(improvement),
            "selected_validation_worst_ratio": float(worst_ratio),
            "selected_validation_win_fraction": best.win_fraction,
            "candidate_count": len(candidates),
        }
    )
    if not accepted:
        return GuardedReadoutCorrection(
            accepted=False,
            reason="prefix-validation-gate-failed",
            selected_name="baseline",
            correction_m=np.zeros_like(observed_mean),
            variance_m2=np.full(
                observed_mean.shape,
                cfg.shared_bias_std_m**2,
                dtype=np.float64,
            ),
            scores=tuple(scores),
            diagnostics=diagnostics,
        )

    selected = candidates[best.name]
    if best.name.startswith("graph_l"):
        name_parts = best.name.split("_")
        strength_text = name_parts[1][1:]
        scale_text = name_parts[2][1:]
        strength = float(strength_text)
        selected_scale = float(scale_text)
        posterior = graph_smoothed_discrepancy_posterior(
            observed_mean,
            observed_variance,
            np.ones(len(observed_mean), dtype=bool),
            laplacian,
            prior_strength=strength,
            covariance_probes=cfg.covariance_probes,
            covariance_seed=cfg.covariance_seed,
        )
        graph_variance = (
            np.zeros(len(observed_mean), dtype=np.float64)
            if posterior.marginal_variance is None
            else posterior.marginal_variance
        )
        marginal_variance = (
            selected_scale**2 * graph_variance
            + cfg.shared_bias_std_m**2
        )
    else:
        global_variance = max(
            float(np.median(observed_variance)),
            cfg.shared_bias_std_m**2,
        )
        marginal_variance = np.full(
            len(observed_mean),
            global_variance,
            dtype=np.float64,
        )
    diagnostics["mean_correction_m"] = float(
        np.mean(np.linalg.norm(selected, axis=1))
    )
    diagnostics["maximum_correction_m"] = float(
        np.max(np.linalg.norm(selected, axis=1))
    )
    return GuardedReadoutCorrection(
        accepted=True,
        reason="accepted",
        selected_name=best.name,
        correction_m=selected,
        variance_m2=np.repeat(marginal_variance[:, None], 3, axis=1),
        scores=tuple(scores),
        diagnostics=diagnostics,
    )


def apply_guarded_readout_correction(
    physical_future_m: np.ndarray,
    belief: GuardedReadoutCorrection,
) -> np.ndarray:
    """Apply an admitted readout field or return the physical values exactly."""

    physical = np.asarray(physical_future_m)
    _require(
        physical.ndim == 3
        and physical.shape[1:] == belief.correction_m.shape,
        "physical_future_m must have shape (T, N, 3)",
    )
    _require(np.all(np.isfinite(physical)), "physical future is not finite")
    if not belief.accepted:
        return physical
    return physical + belief.correction_m[None]
