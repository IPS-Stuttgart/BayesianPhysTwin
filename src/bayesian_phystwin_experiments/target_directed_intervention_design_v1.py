"""Exact finite intervention design for a pending transported query.

The selector minimizes intervention cost subject to identifying the registered
target query. It does not require full cause-coefficient identification when the
target is invariant over the remaining coefficient nullspace.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, Final

import numpy as np

TARGET_DIRECTED_INTERVENTION_SCHEMA: Final = (
    "bayesian_phystwin.target_directed_intervention_design"
)
TARGET_DIRECTED_INTERVENTION_VERSION: Final = 1
TARGET_DIRECTED_INTERVENTION_SEMANTICS: Final = (
    "minimum-cost-finite-intervention-subset-for-target-kernel-inclusion-v1"
)
TARGET_DIRECTED_INTERVENTION_CLAIM_BOUNDARY: Final = (
    "The selected subset is exactly optimal only inside the supplied finite "
    "candidate roster, additive cost model, local linear designs, target map, "
    "coordinates, and numerical tolerances. Target identifiability does not "
    "imply unique cause identification. The result does not validate the "
    "intervention-response models, prove nonlinear closure, establish real-data "
    "transport, guarantee execution safety, or establish state of the art."
)
MAXIMUM_CANDIDATE_INTERVENTIONS: Final = 12


class InterventionDesignStatus(str, Enum):
    """Result of exact target-directed finite intervention search."""

    ALREADY_IDENTIFIABLE = "already_identifiable"
    TARGET_IDENTIFIED = "target_identified"
    PARTIAL_IMPROVEMENT = "partial_improvement"
    UNRESOLVABLE = "unresolvable"


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character lowercase hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a 64-character lowercase hex digest")
    return value


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _matrix(
    value: object,
    *,
    name: str,
    allow_zero_rows: bool = False,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    matrix = np.ascontiguousarray(raw, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a matrix with nonzero columns")
    if not allow_zero_rows and matrix.shape[0] == 0:
        raise ValueError(f"{name} must have nonzero rows")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return matrix


def _immutable(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64).reshape(
        contiguous.shape
    )


def _array_record(value: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _canonical_id(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rank_tolerance(
    singular_values: np.ndarray,
    *,
    relative: float,
    absolute: float,
) -> float:
    scale = float(singular_values[0]) if singular_values.size else 0.0
    return max(absolute, relative * scale)


def _rank_and_nullspace(
    design: np.ndarray,
    *,
    relative: float,
    absolute: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    _, singular_values, right = np.linalg.svd(design, full_matrices=True)
    tolerance = _rank_tolerance(
        singular_values,
        relative=relative,
        absolute=absolute,
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    nullspace = right[rank:, :].T
    return rank, singular_values, nullspace


def _stable_pseudoinverse(
    design: np.ndarray,
    *,
    rank: int,
) -> np.ndarray:
    left, singular_values, right = np.linalg.svd(design, full_matrices=False)
    if rank == 0:
        return np.zeros((design.shape[1], design.shape[0]), dtype=np.float64)
    return right[:rank, :].T @ np.diag(1.0 / singular_values[:rank]) @ left[:, :rank].T


@dataclass(frozen=True, slots=True)
class InterventionSubsetRecordV1:
    """One exact candidate subset under the frozen target objective."""

    intervention_ids: tuple[str, ...]
    total_cost: float
    row_count: int
    physical_rank: int
    full_cause_identifiable: bool
    target_dimension: int
    target_identifiable_dimension: int
    target_ambiguity_dimension: int
    target_fully_identifiable: bool
    target_ambiguity_spectral_norm: float
    target_stability_gain: float | None

    def to_record(self) -> dict[str, object]:
        return {
            "intervention_ids": list(self.intervention_ids),
            "total_cost": self.total_cost,
            "row_count": self.row_count,
            "physical_rank": self.physical_rank,
            "full_cause_identifiable": self.full_cause_identifiable,
            "target_dimension": self.target_dimension,
            "target_identifiable_dimension": self.target_identifiable_dimension,
            "target_ambiguity_dimension": self.target_ambiguity_dimension,
            "target_fully_identifiable": self.target_fully_identifiable,
            "target_ambiguity_spectral_norm": (self.target_ambiguity_spectral_norm),
            "target_stability_gain": self.target_stability_gain,
        }


@dataclass(frozen=True, slots=True)
class TargetDirectedInterventionDesignV1:
    """Enumerate a finite roster and minimize target-identification cost exactly."""

    source_design_id: str
    target_query_id: str
    candidate_roster_id: str
    source_design: np.ndarray
    target_map: np.ndarray
    candidate_intervention_ids: Mapping[str, str]
    candidate_designs: Mapping[str, np.ndarray]
    intervention_costs: Mapping[str, float]
    relative_rank_tolerance: float = 1e-10
    absolute_rank_tolerance: float = 1e-12
    cost_tolerance: float = 1e-12
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    candidate_order: tuple[str, ...] = field(init=False)
    status: InterventionDesignStatus = field(init=False)
    source_physical_rank: int = field(init=False)
    source_target_identifiable_dimension: int = field(init=False)
    maximum_target_identifiable_dimension: int = field(init=False)
    selected_interventions: tuple[str, ...] = field(init=False)
    selected_total_cost: float | None = field(init=False)
    selected_record: InterventionSubsetRecordV1 = field(init=False)
    equally_optimal_subsets: tuple[tuple[str, ...], ...] = field(init=False)
    minimum_full_cause_identification_cost: float | None = field(init=False)
    minimum_full_cause_interventions: tuple[str, ...] | None = field(init=False)
    cost_saving_vs_full_cause_identification: float | None = field(init=False)
    subset_records: tuple[InterventionSubsetRecordV1, ...] = field(init=False)

    def __post_init__(self) -> None:
        for name in ("source_design_id", "target_query_id", "candidate_roster_id"):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name=name),
            )
        if not isinstance(self.candidate_designs, Mapping):
            raise TypeError("candidate_designs must be a mapping")
        if not isinstance(self.candidate_intervention_ids, Mapping):
            raise TypeError("candidate_intervention_ids must be a mapping")
        if not isinstance(self.intervention_costs, Mapping):
            raise TypeError("intervention_costs must be a mapping")
        candidates = tuple(sorted(self.candidate_designs))
        if any(type(item) is not str or not item for item in candidates):
            raise ValueError("candidate IDs must be nonempty literal strings")
        if len(candidates) > MAXIMUM_CANDIDATE_INTERVENTIONS:
            raise ValueError(
                "candidate roster exceeds the exact-search intervention limit"
            )
        if set(self.candidate_intervention_ids) != set(candidates):
            raise ValueError(
                "candidate_intervention_ids must cover exactly the candidate roster"
            )
        if set(self.intervention_costs) != set(candidates):
            raise ValueError(
                "intervention_costs must cover exactly the candidate roster"
            )

        source = _matrix(
            self.source_design,
            name="source_design",
            allow_zero_rows=True,
        )
        target = _matrix(self.target_map, name="target_map")
        if target.shape[1] != source.shape[1]:
            raise ValueError("target_map must share the latent coefficient dimension")
        if float(np.linalg.norm(target, ord="fro")) == 0.0:
            raise ValueError("target_map must contain a nontrivial query")

        candidate_designs: dict[str, np.ndarray] = {}
        candidate_ids: dict[str, str] = {}
        costs: dict[str, float] = {}
        for candidate in candidates:
            design = _matrix(
                self.candidate_designs[candidate],
                name=f"candidate_designs[{candidate!r}]",
            )
            if design.shape[1] != source.shape[1]:
                raise ValueError(
                    "every candidate design must share the latent dimension"
                )
            candidate_designs[candidate] = design
            candidate_ids[candidate] = _digest(
                self.candidate_intervention_ids[candidate],
                name=f"candidate_intervention_ids[{candidate!r}]",
            )
            costs[candidate] = _finite_nonnegative(
                self.intervention_costs[candidate],
                name=f"intervention_costs[{candidate!r}]",
            )

        relative = _finite_nonnegative(
            self.relative_rank_tolerance,
            name="relative_rank_tolerance",
        )
        absolute = _finite_nonnegative(
            self.absolute_rank_tolerance,
            name="absolute_rank_tolerance",
        )
        cost_tolerance = _finite_nonnegative(
            self.cost_tolerance,
            name="cost_tolerance",
        )
        if relative == 0.0 and absolute == 0.0:
            raise ValueError("at least one rank tolerance must be positive")

        records: list[InterventionSubsetRecordV1] = []
        for count in range(len(candidates) + 1):
            for subset in itertools.combinations(candidates, count):
                if subset:
                    design = np.vstack(
                        [source, *(candidate_designs[item] for item in subset)]
                    )
                else:
                    design = source
                rank, _, nullspace = _rank_and_nullspace(
                    design,
                    relative=relative,
                    absolute=absolute,
                )
                ambiguity = target @ nullspace
                ambiguity_singular = np.linalg.svd(
                    ambiguity,
                    compute_uv=False,
                )
                ambiguity_tolerance = _rank_tolerance(
                    ambiguity_singular,
                    relative=relative,
                    absolute=absolute,
                )
                ambiguity_rank = int(
                    np.count_nonzero(ambiguity_singular > ambiguity_tolerance)
                )
                target_dimension = target.shape[0]
                identifiable_dimension = target_dimension - ambiguity_rank
                fully_identifiable = ambiguity_rank == 0
                if fully_identifiable:
                    pseudoinverse = _stable_pseudoinverse(design, rank=rank)
                    stability_gain: float | None = float(
                        np.linalg.norm(target @ pseudoinverse, ord=2)
                    )
                else:
                    stability_gain = None
                records.append(
                    InterventionSubsetRecordV1(
                        intervention_ids=subset,
                        total_cost=float(sum(costs[item] for item in subset)),
                        row_count=int(design.shape[0]),
                        physical_rank=rank,
                        full_cause_identifiable=rank == source.shape[1],
                        target_dimension=target_dimension,
                        target_identifiable_dimension=identifiable_dimension,
                        target_ambiguity_dimension=ambiguity_rank,
                        target_fully_identifiable=fully_identifiable,
                        target_ambiguity_spectral_norm=(
                            float(ambiguity_singular[0])
                            if ambiguity_singular.size
                            else 0.0
                        ),
                        target_stability_gain=stability_gain,
                    )
                )

        source_record = next(
            record for record in records if not record.intervention_ids
        )
        full_target = [record for record in records if record.target_fully_identifiable]
        max_dimension = max(record.target_identifiable_dimension for record in records)
        if full_target:
            minimum_cost = min(record.total_cost for record in full_target)
            cost_best = [
                record
                for record in full_target
                if abs(record.total_cost - minimum_cost) <= cost_tolerance
            ]
            minimum_count = min(len(record.intervention_ids) for record in cost_best)
            count_best = [
                record
                for record in cost_best
                if len(record.intervention_ids) == minimum_count
            ]
            minimum_gain = min(
                float(record.target_stability_gain)
                for record in count_best
                if record.target_stability_gain is not None
            )
            optimal = [
                record
                for record in count_best
                if record.target_stability_gain is not None
                and abs(record.target_stability_gain - minimum_gain) <= cost_tolerance
            ]
            optimal = sorted(optimal, key=lambda item: item.intervention_ids)
            selected = optimal[0]
            status = (
                InterventionDesignStatus.ALREADY_IDENTIFIABLE
                if not selected.intervention_ids
                else InterventionDesignStatus.TARGET_IDENTIFIED
            )
        else:
            improved = [
                record
                for record in records
                if record.target_identifiable_dimension == max_dimension
            ]
            minimum_cost = min(record.total_cost for record in improved)
            cost_best = [
                record
                for record in improved
                if abs(record.total_cost - minimum_cost) <= cost_tolerance
            ]
            minimum_count = min(len(record.intervention_ids) for record in cost_best)
            optimal = sorted(
                [
                    record
                    for record in cost_best
                    if len(record.intervention_ids) == minimum_count
                ],
                key=lambda item: item.intervention_ids,
            )
            selected = optimal[0]
            status = (
                InterventionDesignStatus.PARTIAL_IMPROVEMENT
                if max_dimension > source_record.target_identifiable_dimension
                else InterventionDesignStatus.UNRESOLVABLE
            )

        full_cause = [record for record in records if record.full_cause_identifiable]
        if full_cause:
            minimum_full_cost = min(record.total_cost for record in full_cause)
            minimum_full_records = sorted(
                [
                    record
                    for record in full_cause
                    if abs(record.total_cost - minimum_full_cost) <= cost_tolerance
                ],
                key=lambda item: (
                    len(item.intervention_ids),
                    item.intervention_ids,
                ),
            )
            minimum_full_subset: tuple[str, ...] | None = minimum_full_records[
                0
            ].intervention_ids
        else:
            minimum_full_cost = None
            minimum_full_subset = None
        selected_cost: float | None = (
            selected.total_cost
            if status
            in {
                InterventionDesignStatus.ALREADY_IDENTIFIABLE,
                InterventionDesignStatus.TARGET_IDENTIFIED,
            }
            else None
        )
        saving = (
            minimum_full_cost - selected_cost
            if minimum_full_cost is not None and selected_cost is not None
            else None
        )

        metadata = json.loads(
            json.dumps(self.metadata, sort_keys=True, allow_nan=False)
        )
        object.__setattr__(self, "source_design", _immutable(source))
        object.__setattr__(self, "target_map", _immutable(target))
        object.__setattr__(
            self,
            "candidate_designs",
            {item: _immutable(candidate_designs[item]) for item in candidates},
        )
        object.__setattr__(self, "candidate_intervention_ids", candidate_ids)
        object.__setattr__(self, "intervention_costs", costs)
        object.__setattr__(self, "relative_rank_tolerance", relative)
        object.__setattr__(self, "absolute_rank_tolerance", absolute)
        object.__setattr__(self, "cost_tolerance", cost_tolerance)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "candidate_order", candidates)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "source_physical_rank", source_record.physical_rank)
        object.__setattr__(
            self,
            "source_target_identifiable_dimension",
            source_record.target_identifiable_dimension,
        )
        object.__setattr__(
            self,
            "maximum_target_identifiable_dimension",
            max_dimension,
        )
        object.__setattr__(self, "selected_interventions", selected.intervention_ids)
        object.__setattr__(self, "selected_total_cost", selected_cost)
        object.__setattr__(self, "selected_record", selected)
        object.__setattr__(
            self,
            "equally_optimal_subsets",
            tuple(record.intervention_ids for record in optimal),
        )
        object.__setattr__(
            self,
            "minimum_full_cause_identification_cost",
            minimum_full_cost,
        )
        object.__setattr__(
            self,
            "minimum_full_cause_interventions",
            minimum_full_subset,
        )
        object.__setattr__(
            self,
            "cost_saving_vs_full_cause_identification",
            saving,
        )
        object.__setattr__(self, "subset_records", tuple(records))

        expected = _canonical_id(self.descriptor())
        supplied = self.artifact_id
        if supplied is not None:
            supplied = _digest(supplied, name="artifact_id")
            if supplied != expected:
                raise ValueError("artifact_id does not match design content")
        object.__setattr__(self, "artifact_id", expected)

    @property
    def target_identified(self) -> bool:
        return self.status in {
            InterventionDesignStatus.ALREADY_IDENTIFIABLE,
            InterventionDesignStatus.TARGET_IDENTIFIED,
        }

    def arrays(self) -> Mapping[str, np.ndarray]:
        return {
            "source_design": self.source_design,
            "target_map": self.target_map,
            **{
                f"candidate_design::{item}": self.candidate_designs[item]
                for item in self.candidate_order
            },
        }

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": TARGET_DIRECTED_INTERVENTION_SCHEMA,
            "schema_version": TARGET_DIRECTED_INTERVENTION_VERSION,
            "semantics": TARGET_DIRECTED_INTERVENTION_SEMANTICS,
            "source_design_id": self.source_design_id,
            "target_query_id": self.target_query_id,
            "candidate_roster_id": self.candidate_roster_id,
            "source_design": _array_record(self.source_design),
            "target_map": _array_record(self.target_map),
            "candidate_order": list(self.candidate_order),
            "candidate_intervention_ids": dict(self.candidate_intervention_ids),
            "candidate_designs": {
                item: _array_record(self.candidate_designs[item])
                for item in self.candidate_order
            },
            "intervention_costs": dict(self.intervention_costs),
            "relative_rank_tolerance": self.relative_rank_tolerance,
            "absolute_rank_tolerance": self.absolute_rank_tolerance,
            "cost_tolerance": self.cost_tolerance,
            "status": self.status.value,
            "source_physical_rank": self.source_physical_rank,
            "source_target_identifiable_dimension": (
                self.source_target_identifiable_dimension
            ),
            "maximum_target_identifiable_dimension": (
                self.maximum_target_identifiable_dimension
            ),
            "selected_interventions": list(self.selected_interventions),
            "selected_total_cost": self.selected_total_cost,
            "selected_record": self.selected_record.to_record(),
            "equally_optimal_subsets": [
                list(subset) for subset in self.equally_optimal_subsets
            ],
            "minimum_full_cause_identification_cost": (
                self.minimum_full_cause_identification_cost
            ),
            "minimum_full_cause_interventions": (
                None
                if self.minimum_full_cause_interventions is None
                else list(self.minimum_full_cause_interventions)
            ),
            "cost_saving_vs_full_cause_identification": (
                self.cost_saving_vs_full_cause_identification
            ),
            "subset_records": [record.to_record() for record in self.subset_records],
            "metadata": self.metadata,
            "claim_boundary": TARGET_DIRECTED_INTERVENTION_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}
