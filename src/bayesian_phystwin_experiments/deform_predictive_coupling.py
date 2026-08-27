"""Source-learned measurement-to-future coupling on already-open trajectories.

Every outer forecast receives other trajectories as training data, its own
reference prediction, and selected prefix observations, never its own future.
This is exploratory cross-validation, not a sealed fresh-object evaluation.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.nuisance_aware_information import (
    NuisanceAwareInformationState,
    greedy_nuisance_aware_selection,
)
from bayesian_phystwin.numerical_linear_algebra_v1 import solve_spd
from bayesian_phystwin.query_aware_anchor_planning import greedy_query_aware_selection
from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    _initial_action_frames,
)
from bayesian_phystwin_experiments.deform_sparse_observation_budget import (
    BudgetConfig,
    array_sha256,
    build_problem,
    condition_forecast,
    file_sha256,
    load_config,
    read_case_window,
    selection_order,
    write_json,
)

POLICIES = (
    "random",
    "spatial",
    "maximum_variance",
    "global_information",
    "future_query",
    "latest_uniform",
)
METHODS = (
    "graph_persistence",
    "empirical_no_floor",
    "empirical_floor",
    "permuted_floor",
    "source_guarded_floor",
)
ELLIPSOID_90_CHI2_3 = 6.251388631170325


@dataclass(frozen=True)
class CouplingConfig:
    budget: BudgetConfig
    design_case: str = "103.pkl"
    expected_trajectory_count: int = 14
    floor_fraction: float = 0.5
    guard_blends: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    guard_minimum_mean_improvement: float = 0.01
    guard_minimum_joint_win_fraction: float = 2 / 3
    guard_maximum_case_ratio: float = 1.1
    guard_random_repetitions: int = 8
    bootstrap_replicates: int = 10000
    bootstrap_seed: int = 260828

    def __post_init__(self) -> None:
        for name in (
            "expected_trajectory_count",
            "guard_random_repetitions",
            "bootstrap_replicates",
            "bootstrap_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for value in (
            self.floor_fraction,
            self.guard_minimum_mean_improvement,
            self.guard_minimum_joint_win_fraction,
            self.guard_maximum_case_ratio,
            *self.guard_blends,
        ):
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError("coupling and guard parameters must be finite numbers")
        if not 0 < self.floor_fraction < 1:
            raise ValueError("floor fraction must be strictly between zero and one")
        if self.design_case != self.budget.case_name:
            raise ValueError(
                "the previous design case must remain excluded from scoring"
            )
        if self.expected_trajectory_count < 5:
            raise ValueError("nested trajectory validation needs at least five cases")
        if (
            not self.guard_blends
            or tuple(sorted(set(self.guard_blends))) != self.guard_blends
            or self.guard_blends[0] != 0
            or self.guard_blends[-1] != 1
        ):
            raise ValueError(
                "guard blends must be sorted, unique, and span zero to one"
            )
        if not 0 <= self.guard_minimum_mean_improvement < 1:
            raise ValueError("invalid guard improvement threshold")
        if (
            not 0 < self.guard_minimum_joint_win_fraction <= 1
            or self.guard_maximum_case_ratio < 1
        ):
            raise ValueError("invalid guard win or harm threshold")
        if (
            not 1
            <= self.guard_random_repetitions
            <= self.budget.random_policy_repetitions
        ):
            raise ValueError("invalid inner random-order count")
        if self.bootstrap_replicates < 1 or self.bootstrap_seed < 0:
            raise ValueError("invalid descriptive bootstrap configuration")


def load_coupling_config(path: Path, repo: Path) -> CouplingConfig:
    raw = json.loads(path.read_text())
    if (
        raw.pop("schema", None) != "deform-predictive-coupling-dev-v1"
        or raw.pop("scope", None)
        != "exploratory-whole-trajectory-crossfit-already-open-dlo2"
        or raw.pop("fresh_confirmation_authorized", None) is not False
    ):
        raise ValueError("development scope changed")
    if (
        tuple(raw.pop("policies", ())) != POLICIES
        or tuple(raw.pop("methods", ())) != METHODS
    ):
        raise ValueError("registered comparison changed")
    base = load_config(repo / raw.pop("base_budget_config"))
    if raw.pop("source_archive_sha256") != base.source_archive_sha256:
        raise ValueError("archive binding changed")
    if raw.pop("reported_holdout_count") != raw["expected_trajectory_count"] - 1:
        raise ValueError("holdout denominator changed")
    raw["guard_blends"] = tuple(raw["guard_blends"])
    return CouplingConfig(budget=base, **raw)


@dataclass(frozen=True)
class Case:
    name: str
    reference: np.ndarray
    variance: np.ndarray
    prefix: np.ndarray

    def validate(self, config: CouplingConfig) -> None:
        b = config.budget
        if not self.name or self.name != Path(self.name).name:
            raise ValueError("invalid trajectory name")
        if (
            self.reference.shape != self.variance.shape
            or self.reference.ndim != 3
            or self.reference.shape[1:] != (12, 3)
            or any(
                a.dtype.kind != "f"
                for a in (self.reference, self.variance, self.prefix)
            )
        ):
            raise ValueError("reference and covariance do not align")
        if max((*b.candidate_nodes, *b.hidden_nodes)) >= 12 or any(
            node not in range(2, 10) for node in b.hidden_nodes
        ):
            raise ValueError(
                "measurement or hidden identity is outside the registered graph"
            )
        if not any(node in range(2, 10) for node in b.candidate_nodes):
            raise ValueError("measurement pool needs a free identity")
        if self.reference.shape[0] != b.forecast_end_exclusive - b.dataset_frame_offset:
            raise ValueError("case frame interval changed")
        if self.prefix.shape != (
            b.prefix_end_exclusive - b.dataset_frame_offset,
            *self.reference.shape[1:],
        ):
            raise ValueError("case prefix is not the allowed interval")
        if not all(
            np.isfinite(a).all() for a in (self.reference, self.variance, self.prefix)
        ) or np.any(self.variance < 0):
            raise ValueError("nonfinite case or negative variance")


@dataclass(frozen=True)
class TrainingCase:
    case: Case
    truth: np.ndarray

    def residual(self, config: CouplingConfig) -> np.ndarray:
        self.case.validate(config)
        if (
            self.truth.shape != self.case.reference.shape
            or not np.isfinite(self.truth).all()
        ):
            raise ValueError("training outcomes do not align")
        if not np.array_equal(self.truth[: len(self.case.prefix)], self.case.prefix):
            raise ValueError("training prefix disagrees with its source trajectory")
        frame = action_frame(self.case.reference)
        error = np.einsum(
            "tnc,cd->tnd",
            self.truth.astype(np.float64) - self.case.reference.astype(np.float64),
            frame,
        )
        error[:, [0, 1, error.shape[1] - 2, error.shape[1] - 1]] = 0
        return error


def action_frame(reference: np.ndarray) -> np.ndarray:
    # Only the archived reference, not ground truth, defines the local axes.
    return _initial_action_frames(reference[None, :2].astype(np.float64))[1][0]


def diagonal_covariance(variance: np.ndarray) -> np.ndarray:
    return variance[..., :, None] * np.eye(3)


@dataclass(frozen=True)
class Coupling:
    source_names: tuple[str, ...]
    observation_factors: np.ndarray
    future_factors: np.ndarray
    observation_floor: np.ndarray
    future_floor: np.ndarray
    observation_means: np.ndarray
    reference_future: np.ndarray
    baseline_covariance: np.ndarray
    frames: np.ndarray
    nodes: np.ndarray


def fit_coupling(
    sources: Sequence[TrainingCase],
    held: Case,
    config: CouplingConfig,
    *,
    floor_fraction: float,
    permute: bool = False,
) -> Coupling:
    held.validate(config)
    names = tuple(source.case.name for source in sources)
    if len(names) < 3 or len(set(names)) != len(names) or held.name in names:
        raise ValueError(
            "training must contain distinct whole trajectories excluding the holdout"
        )
    if not 0 <= floor_fraction < 1:
        raise ValueError("invalid covariance floor fraction")
    ordered = sorted(sources, key=lambda source: source.case.name)
    errors = np.stack([source.residual(config) for source in ordered])
    b = config.budget
    frames = np.repeat(b.observation_frames, len(b.candidate_nodes))
    nodes = np.tile(b.candidate_nodes, len(b.observation_frames))
    offsets = frames - b.dataset_frame_offset
    start = b.prefix_end_exclusive - b.dataset_frame_offset
    observations = errors[:, offsets, nodes]
    future = errors[:, start:, b.hidden_nodes]
    if permute:
        future = np.roll(future, 1, axis=0)
    frame = action_frame(held.reference)
    observations = np.einsum("nqc,dc->nqd", observations, frame)
    future = np.einsum("ntqc,dc->ntqd", future, frame)
    scale = np.sqrt((1 - floor_fraction) / len(ordered))
    observation_factors = np.moveaxis(observations, 0, -1) * scale
    future_factors = np.moveaxis(future, 0, -1) * scale
    observation_floor = (
        floor_fraction
        * np.einsum("nqc,nqd->qcd", observations, observations)
        / len(ordered)
    )
    future_floor = (
        floor_fraction * np.einsum("ntqc,ntqd->tqcd", future, future) / len(ordered)
    )
    return Coupling(
        source_names=tuple(source.case.name for source in ordered),
        observation_factors=observation_factors,
        future_factors=future_factors,
        observation_floor=observation_floor,
        future_floor=future_floor,
        observation_means=held.reference[offsets, nodes],
        reference_future=held.reference[start:, b.hidden_nodes],
        baseline_covariance=diagonal_covariance(held.variance[start:, b.hidden_nodes]),
        frames=frames,
        nodes=nodes,
    )


def latest_uniform_order(config: CouplingConfig) -> np.ndarray:
    b = config.budget
    free = [node for node in b.candidate_nodes if node not in (0, 1, 10, 11)]
    ranked = [min(free)]
    while len(ranked) < len(free):
        ranked.append(
            max(
                (node for node in free if node not in ranked),
                key=lambda node: (min(abs(node - old) for old in ranked), -node),
            )
        )
    ranked.extend(node for node in b.candidate_nodes if node not in ranked)
    order = [
        time * len(b.candidate_nodes) + b.candidate_nodes.index(node)
        for time in reversed(range(len(b.observation_frames)))
        for node in ranked
    ]
    return np.asarray(order[: b.budgets[-1]], dtype=np.int64)


def choose_order(
    model: Coupling,
    held: Case,
    config: CouplingConfig,
    policy: str,
    *,
    seed: int,
) -> np.ndarray:
    b = config.budget
    if policy == "latest_uniform":
        return latest_uniform_order(config)
    if policy in ("random", "spatial"):
        problem = build_problem(held.reference, held.variance, b)
        return selection_order(problem, b, policy, bias_std_m=0, seed=seed)
    state = list(model.observation_factors)
    covariance = list(model.observation_floor + b.measurement_std_m**2 * np.eye(3))
    dimension = model.observation_factors.shape[-1]
    prior = NuisanceAwareInformationState.from_independent_priors(np.eye(dimension))
    if policy == "global_information":
        selected = greedy_nuisance_aware_selection(
            prior, state, [None] * len(state), covariance, count=b.budgets[-1]
        ).selected_indices
    elif policy == "future_query":
        query = model.future_factors.reshape(-1, dimension)
        query = np.linalg.qr(query / np.sqrt(len(query)), mode="reduced")[1]
        selected = (
            greedy_query_aware_selection(
                prior,
                query,
                state,
                [None] * len(state),
                covariance,
                count=b.budgets[-1],
            ).selected_indices
            if np.any(query)
            else np.array([], dtype=np.int64)
        )
    elif policy == "maximum_variance":
        chosen: list[int] = []
        for _ in range(b.budgets[-1]):
            sigma = solve_spd(
                prior.marginal_state_precision(), np.eye(dimension)
            ).solution
            scores = [
                float(np.trace(jac @ sigma @ jac.T + model.observation_floor[i]))
                if i not in chosen
                else -np.inf
                for i, jac in enumerate(state)
            ]
            index = int(np.argmax(scores))
            chosen.append(index)
            prior = prior.add_observation(state[index], None, covariance[index])
        selected = np.asarray(chosen, dtype=np.int64)
    else:
        raise ValueError("unknown measurement policy")
    if len(selected) < b.budgets[-1]:
        # Zero-utility ties still consume the same declared budget. They cannot
        # turn a zero predictive cross-covariance into an empirical update.
        remaining = [i for i in range(len(state)) if i not in selected]
        selected = np.concatenate(
            (selected, remaining[: b.budgets[-1] - len(selected)])
        ).astype(np.int64)
    return selected


def condition(
    model: Coupling,
    config: CouplingConfig,
    selected: np.ndarray,
    observations: np.ndarray,
    *,
    with_covariance: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    indices = np.asarray(selected)
    values = np.asarray(observations)
    if (
        indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or len(set(indices.tolist())) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= len(model.nodes))
        or values.shape != (len(indices), 3)
        or not np.isfinite(values).all()
    ):
        raise ValueError("invalid selected observations")
    dimension = model.observation_factors.shape[-1]
    precision = np.eye(dimension)
    eta = np.zeros(dimension)
    for i, value in zip(indices, values, strict=True):
        jac = model.observation_factors[i]
        noise = model.observation_floor[
            i
        ] + config.budget.measurement_std_m**2 * np.eye(3)
        solved = solve_spd(
            noise, np.column_stack((value - model.observation_means[i], jac))
        ).solution
        precision += jac.T @ solved[:, 1:]
        eta += jac.T @ solved[:, 0]
    rhs = np.column_stack((eta, np.eye(dimension))) if with_covariance else eta
    solved = solve_spd(precision, rhs).solution
    latent = solved[:, 0] if with_covariance else solved
    mean = model.reference_future + np.einsum(
        "tqcd,d->tqc", model.future_factors, latent
    )
    if not len(indices):
        mean = model.reference_future
    covariance = None
    if with_covariance:
        covariance = (
            np.einsum(
                "tqci,ij,tqdj->tqcd",
                model.future_factors,
                solved[:, 1:],
                model.future_factors,
            )
            + model.future_floor
        )
        covariance = (covariance + np.swapaxes(covariance, -1, -2)) * 0.5
        if (
            not np.isfinite(covariance).all()
            or np.min(np.linalg.eigvalsh(covariance)) < -1e-12
        ):
            raise ValueError("conditional covariance is not finite PSD")
    return mean.astype(model.reference_future.dtype, copy=False), covariance


def mixture_shrinkage(
    reference: np.ndarray,
    baseline_covariance: np.ndarray,
    candidate: np.ndarray,
    candidate_covariance: np.ndarray,
    blend: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= blend <= 1:
        raise ValueError("invalid shrinkage weight")
    if blend == 0:
        return reference, baseline_covariance
    difference = candidate - reference
    mean = (reference + blend * difference).astype(reference.dtype, copy=False)
    covariance = (
        (1 - blend) * baseline_covariance
        + blend * candidate_covariance
        + blend * (1 - blend) * difference[..., :, None] * difference[..., None, :]
    )
    return mean, covariance


def last_residual(
    held: Case,
    config: CouplingConfig,
    selected: np.ndarray,
) -> np.ndarray:
    b = config.budget
    start = b.prefix_end_exclusive - b.dataset_frame_offset
    reference = held.reference[start:, b.hidden_nodes]
    if not len(selected):
        return reference
    frames = np.repeat(b.observation_frames, len(b.candidate_nodes))
    nodes = np.tile(b.candidate_nodes, len(b.observation_frames))
    latest: dict[int, int] = {}
    for index in selected:
        node = int(nodes[index])
        if node not in (0, 1, 10, 11) and (
            node not in latest or frames[index] > frames[latest[node]]
        ):
            latest[node] = int(index)
    values = {1: np.zeros(3), 10: np.zeros(3)}
    for node, index in latest.items():
        offset = frames[index] - b.dataset_frame_offset
        values[node] = held.prefix[offset, node] - held.reference[offset, node]
    ordered = sorted(values)
    correction = np.column_stack(
        [
            np.interp(b.hidden_nodes, ordered, [values[node][axis] for node in ordered])
            for axis in range(3)
        ]
    )
    return (reference + correction[None]).astype(reference.dtype, copy=False)


def point_errors(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    error = prediction - truth
    return float(np.mean(np.abs(error))), float(
        np.sqrt(np.mean(np.sum(error**2, axis=-1)))
    )


def score(
    prediction: np.ndarray, covariance: np.ndarray, truth: np.ndarray, noise_std: float
) -> dict[str, float]:
    if (
        prediction.ndim != 3
        or prediction.shape != truth.shape
        or prediction.shape[-1] != 3
        or len(prediction) < 3
        or prediction.shape[1] == 0
        or covariance.shape != (*prediction.shape, 3)
        or any(not np.isfinite(a).all() for a in (prediction, covariance, truth))
        or isinstance(noise_std, bool)
        or not np.isfinite(noise_std)
        or noise_std < 0
        or not np.allclose(
            covariance, np.swapaxes(covariance, -1, -2), rtol=1e-10, atol=1e-12
        )
        or np.min(np.linalg.eigvalsh(covariance)) < -1e-12
    ):
        raise ValueError(
            "metric inputs must align with finite means and PSD covariance"
        )
    error = prediction - truth
    total = covariance + noise_std**2 * np.eye(3)
    try:
        np.linalg.cholesky(total)
    except np.linalg.LinAlgError as exc:
        raise ValueError("scored covariance must be positive definite") from exc
    sign, logdet = np.linalg.slogdet(total)
    if not np.isfinite(total).all() or np.any(sign <= 0):
        raise ValueError("scored covariance must be positive definite")
    nees = np.einsum(
        "tqc,tqc->tq", error, np.linalg.solve(total, error[..., None])[..., 0]
    )
    l1, rmse = point_errors(prediction, truth)
    out = {
        "coordinate_l1_mm": l1 * 1000,
        "point_rmse_mm": rmse * 1000,
        "point_nees": float(np.mean(nees)),
        "point_coverage_90": float(np.mean(nees <= ELLIPSOID_90_CHI2_3)),
        "gaussian_nll_per_point": float(
            np.mean(0.5 * (3 * np.log(2 * np.pi) + logdet + nees))
        ),
        "ellipsoid_volume_mm3": float(
            np.mean((4 * np.pi / 3) * ELLIPSOID_90_CHI2_3**1.5 * np.exp(logdet / 2))
            * 1e9
        ),
    }
    for label, positions in zip(
        ("early", "middle", "late"),
        np.array_split(np.arange(len(truth)), 3),
        strict=True,
    ):
        out[label + "_coordinate_l1_mm"] = (
            point_errors(prediction[positions], truth[positions])[0] * 1000
        )
    return out


def fit_guard(
    sources: Sequence[TrainingCase],
    config: CouplingConfig,
) -> tuple[dict[tuple[str, int], float], list[dict[str, Any]]]:
    """Nested source-only mean guard; no calibration or safety guarantee."""
    b = config.budget
    sources = sorted(sources, key=lambda source: source.case.name)
    validations = [
        source for source in sources if source.case.name != config.design_case
    ]
    if not validations:
        raise ValueError("guard has no eligible validation trajectories")
    groups: dict[tuple[str, int, float], list[tuple[float, float]]] = {}
    for validation in validations:
        inner = [
            source for source in sources if source.case.name != validation.case.name
        ]
        model = fit_coupling(
            inner, validation.case, config, floor_fraction=config.floor_fraction
        )
        truth = validation.truth[
            b.prefix_end_exclusive - b.dataset_frame_offset :, b.hidden_nodes
        ]
        base_l1, base_rmse = point_errors(model.reference_future, truth)
        if min(base_l1, base_rmse) <= 0:
            raise ValueError("guard ratios require a nonzero source baseline error")
        pool = validation.case.prefix[
            model.frames - b.dataset_frame_offset, model.nodes
        ]
        for policy in POLICIES:
            repetitions = config.guard_random_repetitions if policy == "random" else 1
            by_budget: dict[tuple[int, float], list[tuple[float, float]]] = {}
            for repetition in range(repetitions):
                order = choose_order(
                    model, validation.case, config, policy, seed=b.seed + repetition
                )
                for count in b.budgets:
                    mean, _ = condition(
                        model,
                        config,
                        order[:count],
                        pool[order[:count]],
                        with_covariance=False,
                    )
                    for blend in config.guard_blends:
                        mixed = model.reference_future + blend * (
                            mean - model.reference_future
                        )
                        l1, rmse = point_errors(mixed, truth)
                        by_budget.setdefault((count, blend), []).append(
                            (l1 / base_l1, rmse / base_rmse)
                        )
            for (count, blend), values in by_budget.items():
                average = np.mean(values, axis=0)
                groups.setdefault((policy, count, blend), []).append(
                    (float(average[0]), float(average[1]))
                )
    chosen: dict[tuple[str, int], float] = {}
    diagnostics: list[dict[str, Any]] = []
    for policy in POLICIES:
        for count in b.budgets:
            eligible: list[tuple[float, float]] = []
            for blend in config.guard_blends:
                values = np.array(groups[(policy, count, blend)])
                average = values.mean(axis=0)
                joint_wins = int(np.count_nonzero(np.all(values < 1, axis=1)))
                passed = bool(
                    blend > 0
                    and count > 0
                    and np.all(average <= 1 - config.guard_minimum_mean_improvement)
                    and joint_wins / len(values)
                    >= config.guard_minimum_joint_win_fraction
                    and np.max(values) <= config.guard_maximum_case_ratio
                )
                diagnostics.append(
                    {
                        "policy": policy,
                        "budget": count,
                        "blend": blend,
                        "validation_count": len(values),
                        "validation_names": [v.case.name for v in validations],
                        "mean_l1_ratio": float(average[0]),
                        "mean_rmse_ratio": float(average[1]),
                        "joint_wins": joint_wins,
                        "worst_case_ratio": float(np.max(values)),
                        "eligible": passed,
                    }
                )
                if passed:
                    eligible.append((float(np.mean(average)), blend))
            chosen[(policy, count)] = min(eligible)[1] if eligible else 0.0
    return chosen, diagnostics


def predict_fold(
    sources: Sequence[TrainingCase],
    held: Case,
    config: CouplingConfig,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    if held.name == config.design_case or held.name in {s.case.name for s in sources}:
        raise ValueError("design case or training overlap cannot be an outer holdout")
    if len(
        sources
    ) != config.expected_trajectory_count - 1 or config.design_case not in {
        s.case.name for s in sources
    }:
        raise ValueError(
            "outer source denominator or design-case training role changed"
        )
    b = config.budget
    held.validate(config)
    guard, guard_diagnostics = fit_guard(sources, config)
    models = {
        "empirical_no_floor": fit_coupling(sources, held, config, floor_fraction=0),
        "empirical_floor": fit_coupling(
            sources, held, config, floor_fraction=config.floor_fraction
        ),
        "permuted_floor": fit_coupling(
            sources, held, config, floor_fraction=config.floor_fraction, permute=True
        ),
    }
    prototype = models["empirical_floor"]
    problem = build_problem(held.reference, held.variance, b)
    pool = held.prefix[prototype.frames - b.dataset_frame_offset, prototype.nodes]
    records: list[dict[str, Any]] = []
    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []

    def append(
        method, policy, count, repetition, selected, mean, covariance, blend=1.0
    ):
        if len(selected) != count:
            raise ValueError("measurement budget mismatch")
        if count == 0 and array_sha256(mean) != array_sha256(
            prototype.reference_future
        ):
            raise ValueError("zero-budget mean changed")
        records.append(
            {
                "method": method,
                "policy": policy,
                "budget": count,
                "repetition": repetition,
                "selected_indices": selected.tolist(),
                "blend": blend,
            }
        )
        means.append(mean)
        covariances.append(covariance)

    for policy in POLICIES:
        repetitions = b.random_policy_repetitions if policy == "random" else 1
        for repetition in range(repetitions):
            seed = b.seed + repetition
            graph_order = (
                latest_uniform_order(config)
                if policy == "latest_uniform"
                else selection_order(problem, b, policy, bias_std_m=0, seed=seed)
            )
            for count in b.budgets:
                graph_mean, graph_variance = condition_forecast(
                    problem,
                    b,
                    graph_order[:count],
                    pool[graph_order[:count]],
                    bias_std_m=0,
                )
                append(
                    "graph_persistence",
                    policy,
                    count,
                    repetition,
                    graph_order[:count],
                    graph_mean[:, b.hidden_nodes],
                    diagonal_covariance(graph_variance[:, b.hidden_nodes]),
                )
            for method, model in models.items():
                order = choose_order(model, held, config, policy, seed=seed)
                for count in b.budgets:
                    mean, covariance = condition(
                        model, config, order[:count], pool[order[:count]]
                    )
                    assert covariance is not None
                    append(
                        method,
                        policy,
                        count,
                        repetition,
                        order[:count],
                        mean,
                        covariance,
                    )
                    if method == "empirical_floor":
                        blend = guard[(policy, count)]
                        mixed_mean, mixed_covariance = mixture_shrinkage(
                            model.reference_future,
                            model.baseline_covariance,
                            mean,
                            covariance,
                            blend,
                        )
                        if blend == 0 and (
                            array_sha256(mixed_mean)
                            != array_sha256(model.reference_future)
                            or array_sha256(mixed_covariance)
                            != array_sha256(model.baseline_covariance)
                        ):
                            raise ValueError("guard fallback is not byte-exact")
                        append(
                            "source_guarded_floor",
                            policy,
                            count,
                            repetition,
                            order[:count],
                            mixed_mean,
                            mixed_covariance,
                            blend,
                        )
    order = latest_uniform_order(config)
    for count in b.budgets:
        append(
            "last_residual",
            "latest_uniform",
            count,
            0,
            order[:count],
            last_residual(held, config, order[:count]),
            prototype.baseline_covariance,
        )
    append(
        "unchanged_baseline",
        "none",
        0,
        0,
        np.array([], dtype=int),
        prototype.reference_future,
        prototype.baseline_covariance,
    )
    return (
        records,
        np.stack(means),
        np.stack(covariances),
        {
            "held_name": held.name,
            "source_names": list(prototype.source_names),
            "held_future_received": False,
            "guard": guard_diagnostics,
            "reference_future_sha256": array_sha256(prototype.reference_future),
            "baseline_covariance_sha256": array_sha256(prototype.baseline_covariance),
        },
    )


def aggregate_rows(
    rows: Sequence[dict[str, Any]], config: CouplingConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_names = tuple(
        score(
            np.zeros((3, 1, 3)),
            np.broadcast_to(np.eye(3), (3, 1, 3, 3)),
            np.zeros((3, 1, 3)),
            0.001,
        )
    )
    case_rows: list[dict[str, Any]] = []
    keys = sorted({(r["case"], r["method"], r["policy"], r["budget"]) for r in rows})
    for name, method, policy, count in keys:
        group = [
            r
            for r in rows
            if (r["case"], r["method"], r["policy"], r["budget"])
            == (name, method, policy, count)
        ]
        case_rows.append(
            {
                "case": name,
                "method": method,
                "policy": policy,
                "budget": count,
                "mean_blend": float(np.mean([r["blend"] for r in group])),
                **{
                    key: float(np.mean([r[key] for r in group])) for key in metric_names
                },
            }
        )
    baseline = {r["case"]: r for r in case_rows if r["method"] == "unchanged_baseline"}
    rng = np.random.default_rng(config.bootstrap_seed)
    names = sorted(baseline)
    if (
        len(names) != config.expected_trajectory_count - 1
        or config.design_case in names
    ):
        raise ValueError(
            "reported trajectory denominator or design-case exclusion changed"
        )
    samples = rng.integers(0, len(names), (config.bootstrap_replicates, len(names)))
    summaries: list[dict[str, Any]] = []
    for method, policy, count in sorted(
        {(r["method"], r["policy"], r["budget"]) for r in case_rows}
    ):
        group = sorted(
            (
                r
                for r in case_rows
                if (r["method"], r["policy"], r["budget"]) == (method, policy, count)
            ),
            key=lambda r: r["case"],
        )
        if [r["case"] for r in group] != names:
            raise ValueError("case denominator differs across methods")
        l1_delta = np.array(
            [
                r["coordinate_l1_mm"] - baseline[r["case"]]["coordinate_l1_mm"]
                for r in group
            ]
        )
        rmse_delta = np.array(
            [r["point_rmse_mm"] - baseline[r["case"]]["point_rmse_mm"] for r in group]
        )
        summaries.append(
            {
                "method": method,
                "policy": policy,
                "budget": count,
                "case_count": len(group),
                **{
                    key: float(np.mean([r[key] for r in group])) for key in metric_names
                },
                "mean_blend": float(np.mean([r["mean_blend"] for r in group])),
                "guard_accepted_cases": (
                    sum(r["mean_blend"] > 0 for r in group)
                    if method == "source_guarded_floor"
                    else None
                ),
                "joint_wins": int(
                    np.count_nonzero((l1_delta < -1e-10) & (rmse_delta < -1e-10))
                ),
                "l1_delta_mm": float(l1_delta.mean()),
                "rmse_delta_mm": float(rmse_delta.mean()),
                "l1_delta_ci95_mm": np.quantile(
                    l1_delta[samples].mean(axis=1), [0.025, 0.975]
                ).tolist(),
                "rmse_delta_ci95_mm": np.quantile(
                    rmse_delta[samples].mean(axis=1), [0.025, 0.975]
                ).tolist(),
            }
        )
    return case_rows, summaries


def plot_summary(root: Path, summaries: Sequence[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    colors = {
        "graph_persistence": "#b45309",
        "empirical_no_floor": "#b91c1c",
        "empirical_floor": "#2563eb",
        "permuted_floor": "#6b7280",
        "source_guarded_floor": "#047857",
    }
    for method in METHODS:
        group = sorted(
            (
                r
                for r in summaries
                if r["method"] == method and r["policy"] == "future_query"
            ),
            key=lambda r: r["budget"],
        )
        for ax, key in zip(
            axes,
            ("coordinate_l1_mm", "point_rmse_mm", "point_coverage_90"),
            strict=True,
        ):
            ax.plot(
                [r["budget"] for r in group],
                [r[key] for r in group],
                marker="o",
                color=colors[method],
                label=method.replace("_", " "),
            )
    control = sorted(
        (r for r in summaries if r["method"] == "last_residual"),
        key=lambda r: r["budget"],
    )
    baseline = next(r for r in summaries if r["method"] == "unchanged_baseline")
    for ax, key, title in zip(
        axes,
        ("coordinate_l1_mm", "point_rmse_mm", "point_coverage_90"),
        (
            "Hidden-future coordinate L1 (mm)",
            "Hidden-future point RMSE (mm)",
            "90% point-ellipsoid coverage",
        ),
        strict=True,
    ):
        ax.plot(
            [r["budget"] for r in control],
            [r[key] for r in control],
            "--s",
            color="#9333a5",
            label="last residual / latest uniform",
        )
        ax.axhline(
            baseline[key], color="black", linestyle=":", label="unchanged baseline"
        )
        ax.set(
            title=title,
            xlabel="3D prefix measurements",
            xticks=[r["budget"] for r in control],
        )
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].axhline(0.9, color="black", alpha=0.3, linewidth=0.8)
    axes[2].set_ylim(0, 1.04)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8
    )
    fig.suptitle(
        "DEFORM DLO2: exploratory trajectory cross-validation (not fresh-object evidence)",
        fontsize=12,
    )
    fig.savefig(root / "predictive-coupling.png", dpi=180)
    fig.savefig(root / "predictive-coupling.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def run_study(
    archive: Path, config_path: Path, root: Path, *, require_clean: bool = True
) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    config = load_coupling_config(config_path, repo)
    if file_sha256(archive) != config.budget.source_archive_sha256:
        raise ValueError("input archive changed")
    if root.exists():
        raise FileExistsError("do not overwrite an experiment")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if require_clean and status:
        raise ValueError("commit source and configuration before empirical execution")
    with np.load(archive, allow_pickle=False) as data:
        names = tuple(str(name) for name in data["names"])
    if (
        len(names) != config.expected_trajectory_count
        or len(set(names)) != len(names)
        or min(names) != config.design_case
    ):
        raise ValueError("archive trajectory denominator or design case changed")
    b = config.budget
    stop = b.forecast_end_exclusive - b.dataset_frame_offset
    prefix_stop = b.prefix_end_exclusive - b.dataset_frame_offset
    cases: list[TrainingCase] = []
    for index, name in enumerate(names):
        mean = read_case_window(archive, "candidate_predictions", index, 0, stop)
        variance = read_case_window(archive, "coordinate_variance_m2", index, 0, stop)
        truth = read_case_window(archive, "targets", index, 0, stop)
        case = Case(name, mean, variance, truth[:prefix_stop])
        case.validate(config)
        cases.append(TrainingCase(case, truth))
    root.mkdir(parents=True)
    write_json(
        root / "input-manifest.json",
        {
            "schema": "deform-predictive-coupling-input-v1",
            "source_revision": revision,
            "source_clean": not bool(status),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__},
            "source_archive_sha256": file_sha256(archive),
            "config_sha256": file_sha256(config_path),
            "module_sha256": file_sha256(Path(__file__)),
            "design_case": config.design_case,
            "outer_holdout_names": sorted(set(names) - {config.design_case}),
            "crossfit_source_future_outcomes_used": True,
            "held_own_future_input_to_predictor": False,
            "fresh_targets_accessed": False,
            "held_v8_accessed": False,
            "dlo4_dlo5_accessed": False,
        },
    )
    seals: list[dict[str, Any]] = []
    for held in sorted(cases, key=lambda item: item.case.name):
        if held.case.name == config.design_case:
            continue
        sources = [source for source in cases if source.case.name != held.case.name]
        records, means, covariances, metadata = predict_fold(sources, held.case, config)
        case_root = root / held.case.name.removesuffix(".pkl")
        case_root.mkdir()
        np.savez_compressed(
            case_root / "predictions.npz", means=means, covariance_m2=covariances
        )
        write_json(
            case_root / "prediction-seal.json",
            {
                **metadata,
                "records": records,
                "prediction_sha256": file_sha256(case_root / "predictions.npz"),
                "outer_scoring_started": False,
            },
        )
        seals.append(
            {
                "case": held.case.name,
                "seal_sha256": file_sha256(case_root / "prediction-seal.json"),
            }
        )
        print(f"Sealed {len(seals)}/{len(cases) - 1} trajectory holdouts", flush=True)
    write_json(
        root / "prediction-barrier.json",
        {"case_count": len(seals), "seals": seals, "outer_scoring_started": False},
    )
    if file_sha256(archive) != b.source_archive_sha256:
        raise ValueError("input archive changed before outer scoring")
    scored: list[dict[str, Any]] = []
    for held in cases:
        if held.case.name == config.design_case:
            continue
        case_root = root / held.case.name.removesuffix(".pkl")
        bound_seal = next(s for s in seals if s["case"] == held.case.name)
        if file_sha256(case_root / "prediction-seal.json") != bound_seal["seal_sha256"]:
            raise ValueError("prediction seal changed before scoring")
        seal = json.loads((case_root / "prediction-seal.json").read_text())
        if file_sha256(case_root / "predictions.npz") != seal["prediction_sha256"]:
            raise ValueError("prediction changed before scoring")
        with np.load(case_root / "predictions.npz", allow_pickle=False) as data:
            means, covariances = data["means"], data["covariance_m2"]
        truth = held.truth[prefix_stop:, b.hidden_nodes]
        for metadata, mean, covariance in zip(
            seal["records"], means, covariances, strict=True
        ):
            scored.append(
                {
                    "case": held.case.name,
                    **metadata,
                    **score(mean, covariance, truth, b.measurement_std_m),
                }
            )
    case_rows, summaries = aggregate_rows(scored, config)
    for filename, rows in (("case-results.csv", case_rows), ("summary.csv", summaries)):
        with (root / filename).open("x", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    result = {
        "schema": "deform-predictive-coupling-result-v1",
        "scope": "exploratory-whole-trajectory-crossfit-already-open-dlo2",
        "case_count": len(seals),
        "physical_object_count": 1,
        "source_revision": revision,
        "prediction_barrier_sha256": file_sha256(root / "prediction-barrier.json"),
        "summaries": summaries,
        "case_results": case_rows,
        "point_sota_claim": False,
        "fresh_confirmation_authorized": False,
    }
    write_json(root / "results.json", result)
    plot_summary(root, summaries)
    write_json(
        root / "run-complete.json",
        {
            "status": "complete-exploratory",
            "results_sha256": file_sha256(root / "results.json"),
            "plot_sha256": file_sha256(root / "predictive-coupling.png"),
            "prediction_barrier_sha256": file_sha256(root / "prediction-barrier.json"),
        },
    )
    return result
