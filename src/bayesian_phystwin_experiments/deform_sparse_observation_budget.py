"""Matched-budget sparse observations of one already-open DEFORM trajectory.

The frozen DEFORM/local-residual point forecast is never retrained. A small
Gaussian readout-discrepancy field supplies cross-node covariance for an
exploratory sensing-policy comparison; it is not a latent physical-state update.
Selection is independent of observed values. Future truth is read only after
the complete prediction archive has been sealed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any
from zipfile import ZipFile

import numpy as np

from bayesian_phystwin.nuisance_aware_information import (
    NuisanceAwareInformationState,
    greedy_nuisance_aware_selection,
)
from bayesian_phystwin.numerical_linear_algebra_v1 import solve_spd
from bayesian_phystwin.query_aware_anchor_planning import greedy_query_aware_selection

POLICIES = (
    "random",
    "spatial",
    "maximum_variance",
    "global_information",
    "future_query",
)
CONDITIONS = ("native_annotations", "simulated_shared_bias")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape), "order": "C"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


@dataclass(frozen=True)
class BudgetConfig:
    source_archive_sha256: str
    case_name: str
    dataset_frame_offset: int
    prefix_end_exclusive: int
    forecast_end_exclusive: int
    observation_frames: tuple[int, ...]
    candidate_nodes: tuple[int, ...]
    hidden_nodes: tuple[int, ...]
    budgets: tuple[int, ...]
    graph_rank: int
    measurement_std_m: float
    shared_bias_std_m: float
    random_policy_repetitions: int
    bias_repetitions: int
    seed: int

    def __post_init__(self) -> None:
        if len(self.source_archive_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_archive_sha256
        ):
            raise ValueError("source archive needs a SHA-256 digest")
        for name in (
            "dataset_frame_offset",
            "prefix_end_exclusive",
            "forecast_end_exclusive",
            "graph_rank",
            "random_policy_repetitions",
            "bias_repetitions",
            "seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in (
            "observation_frames",
            "candidate_nodes",
            "hidden_nodes",
            "budgets",
        ):
            values = getattr(self, name)
            if (
                not values
                or any(
                    isinstance(v, bool) or not isinstance(v, int) or v < 0
                    for v in values
                )
                or tuple(sorted(set(values))) != values
            ):
                raise ValueError(f"{name} must contain sorted unique integers")
        if not self.case_name or self.case_name != Path(self.case_name).name:
            raise ValueError("invalid case name")
        if not self.dataset_frame_offset <= self.observation_frames[0]:
            raise ValueError("observation precedes the archived frame interval")
        if not self.observation_frames[-1] < self.prefix_end_exclusive:
            raise ValueError("observation reaches the future")
        if self.forecast_end_exclusive - self.prefix_end_exclusive < 3:
            raise ValueError("forecast must contain at least three frames")
        if set(self.hidden_nodes) & set(self.candidate_nodes):
            raise ValueError("observed and hidden identities must be disjoint")
        if self.budgets[0] != 0 or self.budgets[-1] > (
            len(self.observation_frames) * len(self.candidate_nodes)
        ):
            raise ValueError("budget exceeds distinct candidate measurements")
        if (
            min(self.graph_rank, self.random_policy_repetitions, self.bias_repetitions)
            < 1
        ):
            raise ValueError("rank and repetition counts must be positive")
        if (
            isinstance(self.measurement_std_m, bool)
            or not np.isfinite(self.measurement_std_m)
            or self.measurement_std_m <= 0
        ):
            raise ValueError("measurement scale must be positive and finite")
        if (
            isinstance(self.shared_bias_std_m, bool)
            or not np.isfinite(self.shared_bias_std_m)
            or self.shared_bias_std_m <= 0
        ):
            raise ValueError("bias scale must be positive and finite")


def load_config(path: Path) -> BudgetConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (
        raw.pop("schema", None) != "deform-sparse-observation-budget-dev-v1"
        or raw.pop("case_selection", None)
        != "lexicographically-first-already-open-dlo2-v7-case"
        or raw.pop("scope", None) != "one-already-open-trajectory-exploratory-only"
        or raw.pop("fresh_confirmation_authorized", None) is not False
        or raw.pop("primary_metric", None) != "hidden-future-mean-coordinate-l1-mm"
        or tuple(raw.pop("conditions", ())) != CONDITIONS
        or tuple(raw.pop("policies", ())) != POLICIES
    ):
        raise ValueError("development experiment boundary changed")
    for key in ("observation_frames", "candidate_nodes", "hidden_nodes", "budgets"):
        raw[key] = tuple(raw[key])
    return BudgetConfig(**raw)


def read_case_window(
    archive: Path, member: str, case_index: int, start: int, stop: int
) -> np.ndarray:
    """Read only one contiguous window from a C-order numeric NPZ member.

    The prefix path does not materialize the remainder of the truth array.
    Hashing compressed bytes for custody is separate from reading their values.
    """

    with ZipFile(archive) as zipped, zipped.open(member + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("unsupported array header version")
        if (
            fortran
            or len(shape) != 4
            or shape[-1] != 3
            or dtype.kind not in "f"
            or dtype.hasobject
            or not 0 <= case_index < shape[0]
            or not 0 <= start < stop <= shape[1]
        ):
            raise ValueError("invalid archived trajectory layout or slice")
        frame_bytes = shape[2] * 3 * dtype.itemsize
        remaining = (case_index * shape[1] + start) * frame_bytes
        while remaining:
            chunk = stream.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("truncated array before requested window")
            remaining -= len(chunk)
        payload = stream.read((stop - start) * frame_bytes)
        if len(payload) != (stop - start) * frame_bytes:
            raise ValueError("truncated requested window")
    return np.frombuffer(payload, dtype=dtype).reshape(stop - start, shape[2], 3)


@dataclass(frozen=True)
class BudgetProblem:
    reference_mean: np.ndarray
    field_design: np.ndarray
    candidate_frames: np.ndarray
    candidate_nodes: np.ndarray
    candidate_means: np.ndarray
    candidate_designs: np.ndarray
    candidate_geometry: np.ndarray
    query_design: np.ndarray
    forecast_design: np.ndarray
    forecast_frames: np.ndarray
    hidden_nodes: np.ndarray


def build_problem(
    reference_mean: np.ndarray, marginal_variance: np.ndarray, config: BudgetConfig
) -> BudgetProblem:
    mean = np.asarray(reference_mean)
    variance = np.asarray(marginal_variance, dtype=np.float64)
    if (
        mean.ndim != 3
        or mean.shape[-1] != 3
        or mean.shape != variance.shape
        or mean.dtype.kind != "f"
        or not np.isfinite(mean).all()
        or not np.isfinite(variance).all()
        or np.any(variance < 0)
    ):
        raise ValueError("reference mean and metric marginal variance must align")
    node_count = mean.shape[1]
    if mean.shape[0] != config.forecast_end_exclusive - config.dataset_frame_offset:
        raise ValueError("archived predictor interval changed")
    if max((*config.candidate_nodes, *config.hidden_nodes)) >= node_count:
        raise ValueError("identity is outside the archived graph")
    if not 1 <= config.graph_rank <= node_count - 4:
        raise ValueError("rank exceeds the free-node graph")
    if any(node not in range(2, node_count - 2) for node in config.hidden_nodes):
        raise ValueError("hidden score identities must be free nodes")

    # Dirichlet chain modes vanish at controlled endpoints. Row normalization
    # preserves the checkpoint's marginal variances; cross-node correlation is
    # an explicit new diagnostic assumption, not a source-fitted covariance.
    interior: np.ndarray = np.arange(1, node_count - 3, dtype=np.float64)
    modes: np.ndarray = np.arange(1, config.graph_rank + 1, dtype=np.float64)
    basis = np.sin(np.pi * interior[:, None] * modes / (node_count - 3)) / modes
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    full_basis = np.zeros((node_count, config.graph_rank))
    full_basis[2:-2] = basis
    design = np.zeros((*mean.shape, 3 * config.graph_rank))
    for coordinate in range(3):
        design[:, :, coordinate, coordinate::3] = (
            np.sqrt(variance[:, :, coordinate, None]) * full_basis[None]
        )
    candidate_frames = np.repeat(config.observation_frames, len(config.candidate_nodes))
    candidate_nodes = np.tile(config.candidate_nodes, len(config.observation_frames))
    array_frames = candidate_frames - config.dataset_frame_offset
    future_start = config.prefix_end_exclusive - config.dataset_frame_offset
    forecast_design = design[future_start:]
    hidden = np.array(config.hidden_nodes, dtype=np.int64)
    query = forecast_design[:, hidden].reshape(-1, design.shape[-1]).copy()
    query /= np.sqrt(len(query))
    # QR preserves J.T @ J and hence trace(J Sigma J.T), without constructing
    # a covariance matrix over thousands of query coordinates in the planner.
    query = np.linalg.qr(query, mode="reduced")[1]
    return BudgetProblem(
        reference_mean=mean,
        field_design=design,
        candidate_frames=candidate_frames,
        candidate_nodes=candidate_nodes,
        candidate_means=mean[array_frames, candidate_nodes],
        candidate_designs=design[array_frames, candidate_nodes],
        candidate_geometry=mean[array_frames, candidate_nodes],
        query_design=query,
        forecast_design=forecast_design,
        forecast_frames=np.arange(
            config.prefix_end_exclusive, config.forecast_end_exclusive
        ),
        hidden_nodes=hidden,
    )


def information_prior(
    problem: BudgetProblem, bias_std_m: float
) -> NuisanceAwareInformationState:
    if not np.isfinite(bias_std_m) or bias_std_m < 0:
        raise ValueError("bias scale must be finite and nonnegative")
    return NuisanceAwareInformationState.from_independent_priors(
        np.eye(problem.field_design.shape[-1]),
        np.eye(3) / bias_std_m**2 if bias_std_m > 0 else None,
    )


def _observation_parts(
    problem: BudgetProblem, config: BudgetConfig, bias_std_m: float
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    state = list(problem.candidate_designs)
    nuisance = [np.eye(3) if bias_std_m > 0 else np.empty((3, 0)) for _ in state]
    noise = [np.eye(3) * config.measurement_std_m**2 for _ in state]
    return state, nuisance, noise


def selection_order(
    problem: BudgetProblem,
    config: BudgetConfig,
    policy: str,
    *,
    bias_std_m: float,
    seed: int,
) -> np.ndarray:
    """Return a nested plan without receiving any observation or score values."""

    if policy not in POLICIES:
        raise ValueError("unknown measurement-selection policy")
    count = config.budgets[-1]
    candidate_count = len(problem.candidate_nodes)
    if policy == "random":
        return np.random.default_rng(seed).permutation(candidate_count)[:count]
    state, nuisance, noise = _observation_parts(problem, config, bias_std_m)
    prior = information_prior(problem, bias_std_m)
    if policy == "future_query":
        return greedy_query_aware_selection(
            prior, problem.query_design, state, nuisance, noise, count=count
        ).selected_indices
    if policy == "global_information":
        return greedy_nuisance_aware_selection(
            prior, state, nuisance, noise, count=count
        ).selected_indices
    if policy == "maximum_variance":
        selected: list[int] = []
        for _ in range(count):
            marginal = solve_spd(
                prior.marginal_state_precision(), np.eye(prior.state_dimension)
            ).solution
            scores = [
                float(np.trace(jac @ marginal @ jac.T))
                if i not in selected
                else -np.inf
                for i, jac in enumerate(state)
            ]
            index = int(np.argmax(scores))
            selected.append(index)
            prior = prior.add_observation(state[index], nuisance[index], noise[index])
        return np.array(selected, dtype=np.int64)

    # Farthest sampling in predicted geometry and time uses no target positions.
    xyz = problem.candidate_geometry
    geometry_scale = max(float(np.linalg.norm(np.ptp(xyz, axis=0))), 1e-9)
    time_scale = max(float(np.ptp(problem.candidate_frames)), 1.0)
    features = np.column_stack(
        (
            xyz / geometry_scale,
            (problem.candidate_frames - problem.candidate_frames.min()) / time_scale,
        )
    )
    selected = [0] if count else []
    while len(selected) < count:
        distance = np.min(
            np.sum((features[:, None] - features[selected]) ** 2, axis=-1), axis=1
        )
        distance[selected] = -np.inf
        selected.append(int(np.argmax(distance)))
    return np.array(selected, dtype=np.int64)


def condition_forecast(
    problem: BudgetProblem,
    config: BudgetConfig,
    selected: np.ndarray,
    observations: np.ndarray,
    *,
    bias_std_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the identical joint Gaussian update for every selection policy."""

    indices = np.asarray(selected)
    values = np.asarray(observations, dtype=np.float64)
    if (
        indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or len(np.unique(indices)) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= len(problem.candidate_nodes))
        or values.shape != (len(indices), 3)
        or not np.isfinite(values).all()
    ):
        raise ValueError("selected observations must be finite, unique, and aligned")
    reference = problem.reference_mean[-len(problem.forecast_frames) :]
    prior = information_prior(problem, bias_std_m)
    dimension = prior.state_dimension
    if len(indices) == 0:
        variance = np.sum(problem.forecast_design**2, axis=-1)
        return reference, variance
    joint = np.block(
        [
            [prior.state_precision, prior.state_nuisance_precision],
            [prior.state_nuisance_precision.T, prior.nuisance_precision],
        ]
    )
    eta = np.zeros(joint.shape[0])
    for index, observation in zip(indices, values, strict=True):
        jac = problem.candidate_designs[index]
        if bias_std_m > 0:
            jac = np.column_stack((jac, np.eye(3)))
        innovation = observation - problem.candidate_means[index]
        joint += jac.T @ jac / config.measurement_std_m**2
        eta += jac.T @ innovation / config.measurement_std_m**2
    solution = solve_spd(joint, np.column_stack((eta, np.eye(len(joint))))).solution
    latent_mean = solution[:dimension, 0]
    latent_covariance = solution[:dimension, 1 : dimension + 1]
    forecast = reference + np.einsum(
        "tncd,d->tnc", problem.forecast_design, latent_mean
    )
    variance = np.einsum(
        "tncd,de,tnce->tnc",
        problem.forecast_design,
        latent_covariance,
        problem.forecast_design,
    )
    if not np.isfinite(forecast).all() or np.min(variance) < -1e-12:
        raise ValueError("invalid conditional forecast")
    return forecast.astype(reference.dtype, copy=False), np.maximum(variance, 0)


def _metrics(
    prediction: np.ndarray, truth: np.ndarray, variance: np.ndarray, floor: float
) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - truth
    total = variance + floor
    z = NormalDist().inv_cdf(0.95)
    return {
        "coordinate_l1_mm": float(np.mean(np.abs(error)) * 1000),
        "point_rmse_mm": float(np.sqrt(np.mean(np.sum(error**2, axis=-1))) * 1000),
        "coordinate_coverage_90": float(np.mean(np.abs(error) <= z * np.sqrt(total))),
        "coordinate_interval_width_mm": float(np.mean(2 * z * np.sqrt(total)) * 1000),
        "coordinate_nees": float(np.mean(error**2 / total)),
        "gaussian_nll_per_point": float(
            np.mean(
                np.sum(0.5 * (np.log(2 * np.pi * total) + error**2 / total), axis=-1)
            )
        ),
    }


def _plot(
    output: Path, rows: Sequence[Mapping[str, Any]], config: BudgetConfig
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "random": "#6b7280",
        "spatial": "#b45309",
        "maximum_variance": "#be185d",
        "global_information": "#2563eb",
        "future_query": "#047857",
    }
    labels = {
        "random": f"Random ({config.random_policy_repetitions} orders)",
        "spatial": "Spatial / temporal spread",
        "maximum_variance": "Maximum variance",
        "global_information": "Global information",
        "future_query": "Future-query information",
    }
    markers = {
        "random": "o",
        "spatial": "s",
        "maximum_variance": "^",
        "global_information": "D",
        "future_query": "x",
    }
    styles = {
        "random": "--",
        "spatial": "-.",
        "maximum_variance": "--",
        "global_information": ":",
        "future_query": "-",
    }
    fig, axes = plt.subplots(
        1, 2, figsize=(11.0, 4.7), sharey=True, constrained_layout=True
    )
    for ax, condition, title in zip(
        axes,
        CONDITIONS,
        ("Released coordinates", "Simulated shared measurement bias"),
        strict=True,
    ):
        for policy in POLICIES:
            group = [
                row
                for row in rows
                if row["condition"] == condition and row["policy"] == policy
            ]
            ax.plot(
                [row["budget"] for row in group],
                [row["coordinate_l1_mm"] for row in group],
                marker=markers[policy],
                linestyle=styles[policy],
                markersize=5,
                linewidth=1.8 if policy != "future_query" else 2.5,
                color=colors[policy],
                label=labels[policy],
            )
        ax.set(title=title, xlabel="3D prefix measurements", xticks=config.budgets)
        ax.set_ylabel("Hidden future coordinate L1 (mm)")
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("DEFORM DLO2: one already-open development trajectory", fontsize=13)
    fig.savefig(output / "error-versus-budget.png", dpi=180)
    fig.savefig(output / "error-versus-budget.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def _source_identity(config_path: Path, *, require_clean: bool) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if require_clean and status:
        raise ValueError(
            "commit the experiment before reading development observations"
        )
    sources = (
        "src/bayesian_phystwin_experiments/deform_sparse_observation_budget.py",
        "src/bayesian_phystwin/nuisance_aware_information.py",
        "src/bayesian_phystwin/numerical_linear_algebra_v1.py",
        "src/bayesian_phystwin/query_aware_anchor_planning.py",
        "scripts/run_deform_sparse_observation_budget.py",
    )
    return {
        "git_revision": revision,
        "git_clean": not bool(status),
        "source_sha256s": {name: file_sha256(root / name) for name in sources},
        "config_sha256": file_sha256(config_path),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def _report(
    output: Path, rows: Sequence[Mapping[str, Any]], config: BudgetConfig
) -> None:
    lines = [
        "# DEFORM Sparse Observation Budget: Development Result",
        "",
        f"One already-open DLO2 trajectory: `{config.case_name}`. This is exploratory,",
        "not a fresh confirmation, official benchmark score, or robot experiment.",
        "",
        f"Prediction frames: [{config.prefix_end_exclusive}, {config.forecast_end_exclusive}).",
        f"Identities excluded from added measurements: {list(config.hidden_nodes)}.",
        "Each budget unit is one three-coordinate point measurement in the prefix.",
        "All policies use the same readout update; the DEFORM predictor is unchanged.",
        "",
        "## Hidden-Future Coordinate L1 (mm)",
        "",
        "Lower is better. Random orders and synthetic noise draws are not independent",
        "physical executions and supply no confidence interval for generalization.",
        "",
    ]
    for condition in CONDITIONS:
        lines.extend([f"### {condition}", ""])
        lines.append("| Policy | " + " | ".join(str(b) for b in config.budgets) + " |")
        lines.append("|---|" + "---:|" * len(config.budgets))
        for policy in POLICIES:
            values = [
                next(
                    row
                    for row in rows
                    if row["condition"] == condition
                    and row["policy"] == policy
                    and row["budget"] == budget
                )
                for budget in config.budgets
            ]
            lines.append(
                "| "
                + policy
                + " | "
                + " | ".join(f"{row['coordinate_l1_mm']:.3f}" for row in values)
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Boundaries",
            "",
            "- The zero-budget forecast preserves the original mean dtype and C-order bytes.",
            "- Selection uses predicted geometry and covariance, never observation values or future truth.",
            "- All predictions were sealed before the hidden future was supplied to the scorer.",
            "- The graph correlation and persistent latent field are diagnostic assumptions, not a learned physical transition.",
            "- Native coordinates are released annotations. The 1 mm likelihood scale is a fixed modeling assumption, not verified annotation accuracy.",
            "- The synthetic condition adds a shared 5 mm bias and independent 1 mm noise; it is not evidence of real sensor-bias removal.",
            "- Coverage, width, coordinate NEES and diagonal-Gaussian NLL in `curves.csv` are descriptive marginal diagnostics, not calibrated joint uncertainty.",
            "- No original model, protected target, held-v8 artifact, DLO4 or DLO5 data was modified or used.",
            "",
            "![Error versus budget](error-versus-budget.png)",
            "",
        ]
    )
    with (output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def run_study(
    archive: Path,
    config_path: Path,
    output: Path,
    *,
    require_clean_source: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    if file_sha256(archive) != config.source_archive_sha256:
        raise ValueError("archived baseline digest mismatch")
    if output.exists():
        raise FileExistsError("use a new output directory; do not overwrite results")
    source_identity = _source_identity(config_path, require_clean=require_clean_source)
    with np.load(archive, allow_pickle=False) as data:
        names = data["names"].tolist()
    if len(names) != len(set(names)) or config.case_name != min(names):
        raise ValueError("case must be the lexicographically first archived trajectory")
    case_index = names.index(config.case_name)
    stop = config.forecast_end_exclusive - config.dataset_frame_offset
    reference = read_case_window(archive, "candidate_predictions", case_index, 0, stop)
    marginal = read_case_window(archive, "coordinate_variance_m2", case_index, 0, stop)
    problem = build_problem(reference, marginal, config)
    output.mkdir(parents=True)
    write_json(
        output / "input-manifest.json",
        {
            "schema": "deform-sparse-budget-input-v1",
            "config": asdict(config),
            "config_sha256": file_sha256(config_path),
            "source_archive_sha256": file_sha256(archive),
            "case_index": case_index,
            "case_count": 1,
            "reference_mean_sha256": array_sha256(reference),
            "saved_marginal_variance_sha256": array_sha256(marginal),
            "baseline_retrained": False,
            "fresh_targets_accessed": False,
            "held_v8_accessed": False,
            "dlo4_dlo5_accessed": False,
            "future_truth_used_for_selection_or_update": False,
            "implementation": source_identity,
        },
    )

    schedules: dict[tuple[str, str, int], np.ndarray] = {}
    for condition in CONDITIONS:
        bias = config.shared_bias_std_m if condition == "simulated_shared_bias" else 0.0
        for policy in POLICIES:
            repetitions = config.random_policy_repetitions if policy == "random" else 1
            for repetition in range(repetitions):
                order = selection_order(
                    problem,
                    config,
                    policy,
                    bias_std_m=bias,
                    seed=config.seed + repetition,
                )
                if len(order) != config.budgets[-1]:
                    raise ValueError(
                        "policy cannot supply the matched measurement budget"
                    )
                schedules[(condition, policy, repetition)] = order
    write_json(
        output / "selection-seal.json",
        {
            "schema": "deform-sparse-budget-selection-v1",
            "input_manifest_sha256": file_sha256(output / "input-manifest.json"),
            "observation_values_read": False,
            "future_truth_read": False,
            "candidate_frames": problem.candidate_frames.tolist(),
            "candidate_nodes": problem.candidate_nodes.tolist(),
            "orders": [
                {
                    "condition": key[0],
                    "policy": key[1],
                    "repetition": key[2],
                    "indices": order.tolist(),
                }
                for key, order in schedules.items()
            ],
        },
    )

    prefix = read_case_window(
        archive,
        "targets",
        case_index,
        0,
        config.prefix_end_exclusive - config.dataset_frame_offset,
    )
    native = prefix[
        problem.candidate_frames - config.dataset_frame_offset, problem.candidate_nodes
    ].astype(np.float64)
    records: list[dict[str, Any]] = []
    forecasts: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    for condition in CONDITIONS:
        bias = config.shared_bias_std_m if condition == "simulated_shared_bias" else 0.0
        trials = config.bias_repetitions if bias > 0 else 1
        for trial in range(trials):
            observed = native.copy()
            if bias > 0:
                rng = np.random.default_rng(config.seed + 10000 + trial)
                observed += rng.normal(0, bias, size=(1, 3))
                observed += rng.normal(0, config.measurement_std_m, size=observed.shape)
            for policy in POLICIES:
                repetitions = (
                    config.random_policy_repetitions if policy == "random" else 1
                )
                for repetition in range(repetitions):
                    order = schedules[(condition, policy, repetition)]
                    for budget in config.budgets:
                        selected = order[:budget]
                        prediction, variance = condition_forecast(
                            problem,
                            config,
                            selected,
                            observed[selected],
                            bias_std_m=bias,
                        )
                        if budget == 0 and array_sha256(prediction) != array_sha256(
                            reference[-len(prediction) :]
                        ):
                            raise RuntimeError(
                                "zero-budget mean lost exact baseline identity"
                            )
                        forecasts.append(prediction)
                        variances.append(variance)
                        records.append(
                            {
                                "condition": condition,
                                "trial": trial,
                                "policy": policy,
                                "repetition": repetition,
                                "budget": budget,
                                "actual_count": len(selected),
                                "selected_indices": selected.tolist(),
                            }
                        )
    prediction_array = np.stack(forecasts)
    variance_array = np.stack(variances)
    reference_forecast = reference[-len(problem.forecast_frames) :]
    if any(
        array_sha256(prediction_array[i]) != array_sha256(reference_forecast)
        for i, record in enumerate(records)
        if record["budget"] == 0
    ):
        raise RuntimeError("saved zero-budget means lost byte identity")
    np.savez_compressed(
        output / "predictions.npz",
        predictions=prediction_array,
        variance_m2=variance_array,
        forecast_frames=problem.forecast_frames,
        hidden_nodes=problem.hidden_nodes,
    )
    write_json(
        output / "prediction-seal.json",
        {
            "schema": "deform-sparse-budget-prediction-v1",
            "selection_sha256": file_sha256(output / "selection-seal.json"),
            "prediction_file_sha256": file_sha256(output / "predictions.npz"),
            "reference_forecast_sha256": array_sha256(
                reference[-len(problem.forecast_frames) :]
            ),
            "future_truth_read": False,
            "record_count": len(records),
            "records": records,
        },
    )

    # Only the scorer receives the hidden future. All policies and all budgets
    # above have already been selected, materialized, and hash-bound.
    if (
        file_sha256(output / "predictions.npz")
        != json.loads((output / "prediction-seal.json").read_text())[
            "prediction_file_sha256"
        ]
    ):
        raise RuntimeError("prediction archive changed before scoring")
    if file_sha256(archive) != config.source_archive_sha256:
        raise RuntimeError("input archive changed before scoring")
    truth = read_case_window(
        archive,
        "targets",
        case_index,
        config.prefix_end_exclusive - config.dataset_frame_offset,
        stop,
    )[:, problem.hidden_nodes]
    scored: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        prediction = prediction_array[index][:, problem.hidden_nodes]
        variance = variance_array[index][:, problem.hidden_nodes]
        metrics = _metrics(prediction, truth, variance, config.measurement_std_m**2)
        for name, positions in zip(
            ("early", "middle", "late"),
            np.array_split(np.arange(len(truth)), 3),
            strict=True,
        ):
            metrics[name + "_coordinate_l1_mm"] = float(
                np.mean(np.abs(prediction[positions] - truth[positions])) * 1000
            )
        scored.append({**record, **metrics})
    rows: list[dict[str, Any]] = []
    metric_names = tuple(key for key in scored[0] if key not in records[0])
    for condition in CONDITIONS:
        for policy in POLICIES:
            for budget in config.budgets:
                group = [
                    r
                    for r in scored
                    if r["condition"] == condition
                    and r["policy"] == policy
                    and r["budget"] == budget
                ]
                rows.append(
                    {
                        "condition": condition,
                        "policy": policy,
                        "budget": budget,
                        "replicate_count": len(group),
                        **{
                            metric: float(np.mean([r[metric] for r in group]))
                            for metric in metric_names
                        },
                        "min_coordinate_l1_mm": min(
                            r["coordinate_l1_mm"] for r in group
                        ),
                        "max_coordinate_l1_mm": max(
                            r["coordinate_l1_mm"] for r in group
                        ),
                        "mean_actual_count": float(
                            np.mean([r["actual_count"] for r in group])
                        ),
                    }
                )
    with (output / "curves.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "schema": "deform-sparse-budget-result-v1",
        "case_name": config.case_name,
        "scope": "one-already-open-trajectory-exploratory-only",
        "case_count": 1,
        "config_sha256": file_sha256(config_path),
        "prediction_seal_sha256": file_sha256(output / "prediction-seal.json"),
        "source_archive_sha256": file_sha256(archive),
        "forecast_frames": [
            int(problem.forecast_frames[0]),
            int(problem.forecast_frames[-1]),
        ],
        "hidden_nodes": problem.hidden_nodes.tolist(),
        "curves": rows,
        "records": scored,
        "future_truth_used_for_selection_or_update": False,
        "zero_budget_mean_byte_identical": True,
        "baseline_retrained": False,
        "fresh_targets_accessed": False,
        "held_v8_accessed": False,
        "dlo4_dlo5_accessed": False,
        "independent_physical_executions": 1,
        "uncertainty_calibration_claim": False,
        "synthetic_bias_is_real_sensor_evidence": False,
    }
    write_json(output / "results.json", result)
    _plot(output, rows, config)
    _report(output, rows, config)
    write_json(
        output / "run-complete.json",
        {
            "schema": "deform-sparse-budget-complete-v1",
            "files_sha256": {
                path.name: file_sha256(path)
                for path in sorted(output.iterdir())
                if path.is_file()
            },
            "status": "complete-exploratory",
            "fresh_confirmation_authorized": False,
        },
    )
    return result
