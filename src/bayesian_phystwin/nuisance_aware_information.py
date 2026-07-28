"""Nuisance-aware information gain for active Bayesian observations.

The module represents a Gaussian information state in block form for physical
state coefficients and nuisance coefficients. Candidate observation blocks are
added without explicit covariance inverses, and their value is measured after
marginalizing nuisance variables with a Schur complement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


_NUMERICAL_TOLERANCE = 1e-10


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _finite_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64).copy()
    _require(matrix.ndim == 2, f"{name} must be a matrix")
    _require(np.all(np.isfinite(matrix)), f"{name} contains non-finite values")
    return matrix


def _symmetric_positive_definite(value: np.ndarray, *, name: str) -> np.ndarray:
    matrix = _finite_matrix(value, name=name)
    _require(matrix.shape[0] == matrix.shape[1], f"{name} must be square")
    _require(np.allclose(matrix, matrix.T), f"{name} must be symmetric")
    matrix = 0.5 * (matrix + matrix.T)
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return matrix


def _log_determinant_spd(value: np.ndarray, *, name: str) -> float:
    try:
        cholesky = np.linalg.cholesky(value)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return float(2.0 * np.sum(np.log(np.diag(cholesky))))


def _reliability_vector(value: float | np.ndarray, row_count: int) -> np.ndarray:
    reliability = np.asarray(value, dtype=np.float64)
    if reliability.ndim == 0:
        reliability = np.full(row_count, float(reliability), dtype=np.float64)
    _require(
        reliability.shape == (row_count,),
        "reliability must be a scalar or one value per observation row",
    )
    _require(
        np.all(np.isfinite(reliability))
        and np.all((reliability >= 0.0) & (reliability <= 1.0)),
        "reliability must lie in [0, 1]",
    )
    return reliability


@dataclass(frozen=True)
class NuisanceAwareInformationState:
    """Joint Gaussian precision for state and nuisance coefficients.

    ``state_precision`` is the physical-state block, ``nuisance_precision`` is
    the nuisance block, and ``state_nuisance_precision`` is their cross block.
    The marginal state precision is the Schur complement after nuisance
    marginalization.
    """

    state_precision: np.ndarray
    nuisance_precision: np.ndarray
    state_nuisance_precision: np.ndarray

    def __post_init__(self) -> None:
        state = _symmetric_positive_definite(
            self.state_precision,
            name="state_precision",
        )
        nuisance = _finite_matrix(
            self.nuisance_precision,
            name="nuisance_precision",
        )
        _require(
            nuisance.shape[0] == nuisance.shape[1],
            "nuisance_precision must be square",
        )
        cross = _finite_matrix(
            self.state_nuisance_precision,
            name="state_nuisance_precision",
        )
        _require(
            cross.shape == (state.shape[0], nuisance.shape[0]),
            "state_nuisance_precision shape changed",
        )
        if nuisance.shape[0]:
            nuisance = _symmetric_positive_definite(
                nuisance,
                name="nuisance_precision",
            )
            joint = np.block([[state, cross], [cross.T, nuisance]])
            _symmetric_positive_definite(joint, name="joint precision")
        for name, array in (
            ("state_precision", state),
            ("nuisance_precision", nuisance),
            ("state_nuisance_precision", cross),
        ):
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @classmethod
    def from_independent_priors(
        cls,
        state_precision: np.ndarray,
        nuisance_precision: np.ndarray | None = None,
    ) -> NuisanceAwareInformationState:
        """Build a state with independent state and nuisance priors."""

        state = _finite_matrix(state_precision, name="state_precision")
        if nuisance_precision is None:
            nuisance = np.empty((0, 0), dtype=np.float64)
        else:
            nuisance = _finite_matrix(
                nuisance_precision,
                name="nuisance_precision",
            )
        return cls(
            state_precision=state,
            nuisance_precision=nuisance,
            state_nuisance_precision=np.zeros(
                (state.shape[0], nuisance.shape[0]),
                dtype=np.float64,
            ),
        )

    @property
    def state_dimension(self) -> int:
        """Number of physical-state coefficients."""

        return int(self.state_precision.shape[0])

    @property
    def nuisance_dimension(self) -> int:
        """Number of nuisance coefficients."""

        return int(self.nuisance_precision.shape[0])

    def marginal_state_precision(self) -> np.ndarray:
        """Return state precision after marginalizing nuisance coefficients."""

        if not self.nuisance_dimension:
            marginal = self.state_precision.copy()
        else:
            nuisance_solution = np.linalg.solve(
                self.nuisance_precision,
                self.state_nuisance_precision.T,
            )
            marginal = (
                self.state_precision
                - self.state_nuisance_precision @ nuisance_solution
            )
            marginal = 0.5 * (marginal + marginal.T)
        _symmetric_positive_definite(
            marginal,
            name="marginal state precision",
        )
        marginal.setflags(write=False)
        return marginal

    def marginal_log_determinant(self) -> float:
        """Return the log determinant of marginalized state precision."""

        return _log_determinant_spd(
            self.marginal_state_precision(),
            name="marginal state precision",
        )

    def add_observation(
        self,
        state_jacobian: np.ndarray,
        nuisance_jacobian: np.ndarray | None,
        observation_covariance: np.ndarray,
        *,
        reliability: float | np.ndarray = 1.0,
    ) -> NuisanceAwareInformationState:
        """Return the information state after one observation block.

        Reliability is an expected-information weight. For correlated rows it is
        applied after covariance whitening, which preserves a positive
        semidefinite information contribution without pretending that dependent
        rows are independent.
        """

        state_jacobian_array = _finite_matrix(
            state_jacobian,
            name="state_jacobian",
        )
        row_count = state_jacobian_array.shape[0]
        _require(
            state_jacobian_array.shape[1] == self.state_dimension,
            "state_jacobian state dimension changed",
        )
        if nuisance_jacobian is None:
            nuisance_jacobian_array = np.zeros(
                (row_count, self.nuisance_dimension),
                dtype=np.float64,
            )
        else:
            nuisance_jacobian_array = _finite_matrix(
                nuisance_jacobian,
                name="nuisance_jacobian",
            )
        _require(
            nuisance_jacobian_array.shape
            == (row_count, self.nuisance_dimension),
            "nuisance_jacobian nuisance dimension changed",
        )
        covariance = _symmetric_positive_definite(
            observation_covariance,
            name="observation_covariance",
        )
        _require(
            covariance.shape == (row_count, row_count),
            "observation_covariance row count changed",
        )
        reliability_array = _reliability_vector(reliability, row_count)

        cholesky = np.linalg.cholesky(covariance)
        state_whitened = np.linalg.solve(cholesky, state_jacobian_array)
        nuisance_whitened = np.linalg.solve(cholesky, nuisance_jacobian_array)
        scale = np.sqrt(reliability_array)[:, None]
        state_whitened *= scale
        nuisance_whitened *= scale

        state_increment = state_whitened.T @ state_whitened
        nuisance_increment = nuisance_whitened.T @ nuisance_whitened
        cross_increment = state_whitened.T @ nuisance_whitened
        return NuisanceAwareInformationState(
            state_precision=self.state_precision + state_increment,
            nuisance_precision=self.nuisance_precision + nuisance_increment,
            state_nuisance_precision=(
                self.state_nuisance_precision + cross_increment
            ),
        )

    def observation_information_gain(
        self,
        state_jacobian: np.ndarray,
        nuisance_jacobian: np.ndarray | None,
        observation_covariance: np.ndarray,
        *,
        reliability: float | np.ndarray = 1.0,
    ) -> NuisanceAwareInformationUpdate:
        """Evaluate one candidate and return its nuisance-marginalized gain."""

        updated = self.add_observation(
            state_jacobian,
            nuisance_jacobian,
            observation_covariance,
            reliability=reliability,
        )
        log_gain = (
            updated.marginal_log_determinant()
            - self.marginal_log_determinant()
        )
        if log_gain < -_NUMERICAL_TOLERANCE:
            raise RuntimeError(
                "marginal state information decreased after an observation"
            )
        log_gain = max(log_gain, 0.0)
        return NuisanceAwareInformationUpdate(
            updated_state=updated,
            marginal_log_determinant_gain=log_gain,
            mutual_information_nats=0.5 * log_gain,
        )


@dataclass(frozen=True)
class NuisanceAwareInformationUpdate:
    """One candidate observation and its marginalized information gain."""

    updated_state: NuisanceAwareInformationState
    marginal_log_determinant_gain: float
    mutual_information_nats: float

    def __post_init__(self) -> None:
        _require(
            np.isfinite(self.marginal_log_determinant_gain)
            and self.marginal_log_determinant_gain >= 0.0,
            "marginal_log_determinant_gain must be finite and nonnegative",
        )
        _require(
            np.isfinite(self.mutual_information_nats)
            and self.mutual_information_nats >= 0.0,
            "mutual_information_nats must be finite and nonnegative",
        )
        _require(
            np.isclose(
                self.mutual_information_nats,
                0.5 * self.marginal_log_determinant_gain,
            ),
            "mutual-information and log-determinant gains are inconsistent",
        )


@dataclass(frozen=True)
class GreedyNuisanceAwareSelection:
    """Deterministic greedy selection under nuisance-marginalized information gain."""

    selected_indices: np.ndarray
    mutual_information_nats: np.ndarray
    final_state: NuisanceAwareInformationState

    def __post_init__(self) -> None:
        indices = np.asarray(self.selected_indices, dtype=np.int64).copy()
        gains = np.asarray(self.mutual_information_nats, dtype=np.float64).copy()
        _require(indices.ndim == 1, "selected_indices must be a vector")
        _require(gains.shape == indices.shape, "selection gain count changed")
        _require(np.all(indices >= 0), "selected_indices must be nonnegative")
        _require(
            len(np.unique(indices)) == len(indices),
            "selected_indices must be unique",
        )
        _require(
            np.all(np.isfinite(gains)) and np.all(gains >= 0.0),
            "selection gains must be finite and nonnegative",
        )
        indices.setflags(write=False)
        gains.setflags(write=False)
        object.__setattr__(self, "selected_indices", indices)
        object.__setattr__(self, "mutual_information_nats", gains)


def greedy_nuisance_aware_selection(
    prior: NuisanceAwareInformationState,
    state_jacobians: Sequence[np.ndarray],
    nuisance_jacobians: Sequence[np.ndarray | None],
    observation_covariances: Sequence[np.ndarray],
    *,
    reliabilities: Sequence[float | np.ndarray] | None = None,
    count: int,
    minimum_gain_nats: float = 0.0,
) -> GreedyNuisanceAwareSelection:
    """Greedily choose candidates by nuisance-marginalized mutual information.

    Ties are broken by the lowest original candidate index. Each selected block
    updates the joint state before the next candidate is scored, so redundant
    observations exhibit diminishing returns.
    """

    candidate_count = len(state_jacobians)
    _require(
        len(nuisance_jacobians) == candidate_count
        and len(observation_covariances) == candidate_count,
        "candidate input counts differ",
    )
    _require(count >= 0, "count must be nonnegative")
    _require(
        np.isfinite(minimum_gain_nats) and minimum_gain_nats >= 0.0,
        "minimum_gain_nats must be finite and nonnegative",
    )
    if reliabilities is None:
        reliability_values: Sequence[float | np.ndarray] = [1.0] * candidate_count
    else:
        _require(
            len(reliabilities) == candidate_count,
            "reliability candidate count differs",
        )
        reliability_values = reliabilities

    state = prior
    remaining = set(range(candidate_count))
    selected: list[int] = []
    gains: list[float] = []
    while remaining and len(selected) < count:
        evaluations: dict[int, NuisanceAwareInformationUpdate] = {}
        for candidate in sorted(remaining):
            evaluations[candidate] = state.observation_information_gain(
                state_jacobians[candidate],
                nuisance_jacobians[candidate],
                observation_covariances[candidate],
                reliability=reliability_values[candidate],
            )
        best_gain = max(
            evaluation.mutual_information_nats
            for evaluation in evaluations.values()
        )
        tied = [
            candidate
            for candidate, evaluation in evaluations.items()
            if np.isclose(
                evaluation.mutual_information_nats,
                best_gain,
                rtol=0.0,
                atol=1e-12,
            )
        ]
        chosen = min(tied)
        chosen_evaluation = evaluations[chosen]
        if chosen_evaluation.mutual_information_nats <= minimum_gain_nats:
            break
        selected.append(chosen)
        gains.append(chosen_evaluation.mutual_information_nats)
        state = chosen_evaluation.updated_state
        remaining.remove(chosen)

    return GreedyNuisanceAwareSelection(
        selected_indices=np.asarray(selected, dtype=np.int64),
        mutual_information_nats=np.asarray(gains, dtype=np.float64),
        final_state=state,
    )
