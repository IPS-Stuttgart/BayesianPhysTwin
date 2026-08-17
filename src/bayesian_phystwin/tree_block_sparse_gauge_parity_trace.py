"""Opt-in shadow-parity traces for the production tree-block IRLS update.

The historical update remains authoritative and unchanged for ordinary callers.
This module observes only normal systems that have passed the production
factorization gate, requires the independent tree-separator solver to agree, and
binds those reports to the exact unchanged production result.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Final

import numpy as np

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from ._gauge_aware_contracts import GaugeAwareObservationBatch
from ._prior_aware_gauge_math import PriorAwareGaugeConfigV1
from .sparse_prior_aware_gauge_belief import TreeSparseGaugeDesignV1
from .tree_block_gaussian import TreeBlockNormalSystemV1
from .tree_block_sparse_gauge_belief import (
    TreeBlockGaugeAwareBeliefResultV1,
    _update_tree_block_sparse_prior_aware_gauge_belief,
)
from .tree_separator_gaussian_parity import (
    TREE_SEPARATOR_GAUSSIAN_PARITY_IMPLEMENTATION,
    TreeSeparatorGaussianParityV1,
    require_tree_separator_gaussian_parity,
)

TREE_BLOCK_SPARSE_GAUGE_PARITY_TRACE_SCHEMA: Final = (
    "bayesian_phystwin.tree_block_sparse_gauge_parity_trace"
)
TREE_BLOCK_SPARSE_GAUGE_PARITY_TRACE_VERSION: Final = 1
TREE_BLOCK_SPARSE_GAUGE_PARITY_TRACE_IMPLEMENTATION: Final = (
    "production-tree-block-irls-shadow-trace-v1"
)
TREE_BLOCK_SPARSE_GAUGE_PARITY_TRACE_BOUNDARY: Final = (
    "Numerical shadow evidence for production-admitted IRLS systems only. It "
    "does not establish provider competence, calibrated uncertainty, physical-"
    "query benefit, intervention benefit, deployment safety, or state of the art."
)

_STEP_SCHEMA: Final = "bayesian_phystwin.tree_block_sparse_gauge_parity_step"
_STEP_VERSION: Final = 1
_PHASE_ORDER = {"irls-solve": 0, "irls-final": 1}


def _canonical_id(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            plain_json(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _positive_condition_limit(value: object) -> float:
    result = _finite_nonnegative(value, name="maximum_condition_number")
    if result <= 0.0:
        raise ValueError("maximum_condition_number must be positive")
    return result


@dataclass(frozen=True, slots=True)
class TreeBlockSparseGaugeParityStepV1:
    """One production-admitted IRLS normal system and its parity report."""

    phase: str
    iteration: int
    parity: TreeSeparatorGaussianParityV1
    _step_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.phase) is not str or self.phase not in _PHASE_ORDER:
            raise ValueError("phase must be 'irls-solve' or 'irls-final'")
        if type(self.iteration) is not int or self.iteration < 1:
            raise ValueError("iteration must be a positive integer")
        if not isinstance(self.parity, TreeSeparatorGaussianParityV1):
            raise TypeError("parity must be a TreeSeparatorGaussianParityV1")
        if not self.parity.passed:
            raise ValueError("a parity trace cannot contain a failed report")
        object.__setattr__(self, "_step_id", _canonical_id(self.descriptor()))

    @property
    def step_id(self) -> str:
        return self._step_id

    def descriptor(self) -> Mapping[str, object]:
        return frozen_finite_json_mapping(
            {
                "schema": _STEP_SCHEMA,
                "schema_version": _STEP_VERSION,
                "phase": self.phase,
                "iteration": self.iteration,
                "parity": self.parity.to_dict(),
            },
            name="tree-block sparse-gauge parity step",
        )

    def to_dict(self) -> dict[str, object]:
        return {**dict(self.descriptor()), "step_id": self.step_id}


@dataclass(frozen=True, slots=True)
class TreeBlockSparseGaugeParityTraceV1:
    """Content-addressed parity trace bound to one unchanged production result."""

    result: TreeBlockGaugeAwareBeliefResultV1
    steps: tuple[TreeBlockSparseGaugeParityStepV1, ...]
    maximum_condition_number: float
    relative_tolerance: float
    absolute_tolerance: float
    _trace_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.result, TreeBlockGaugeAwareBeliefResultV1):
            raise TypeError("result must be a TreeBlockGaugeAwareBeliefResultV1")
        if not isinstance(self.steps, tuple):
            raise TypeError("steps must be a tuple")
        for position, step in enumerate(self.steps):
            if not isinstance(step, TreeBlockSparseGaugeParityStepV1):
                raise TypeError(
                    f"steps[{position}] must be a TreeBlockSparseGaugeParityStepV1"
                )
        condition_limit = _positive_condition_limit(self.maximum_condition_number)
        relative = _finite_nonnegative(
            self.relative_tolerance,
            name="relative_tolerance",
        )
        absolute = _finite_nonnegative(
            self.absolute_tolerance,
            name="absolute_tolerance",
        )
        object.__setattr__(self, "maximum_condition_number", condition_limit)
        object.__setattr__(self, "relative_tolerance", relative)
        object.__setattr__(self, "absolute_tolerance", absolute)

        expected_iteration = 1
        previous_phase_order = -1
        for step in self.steps:
            if step.iteration == expected_iteration + 1:
                if previous_phase_order != _PHASE_ORDER["irls-final"]:
                    raise ValueError(
                        "a new parity-trace iteration must follow an irls-final step"
                    )
                expected_iteration += 1
                previous_phase_order = -1
            if step.iteration != expected_iteration:
                raise ValueError("parity-trace iterations must be contiguous from one")
            phase_order = _PHASE_ORDER[step.phase]
            if phase_order <= previous_phase_order:
                raise ValueError("parity-trace phases are duplicated or out of order")
            if phase_order == _PHASE_ORDER["irls-final"] and previous_phase_order != 0:
                raise ValueError("an irls-final step must follow irls-solve")
            previous_phase_order = phase_order
            report = step.parity
            if report.maximum_condition_number != condition_limit:
                raise ValueError("step condition limit differs from the trace")
            if report.relative_tolerance != relative:
                raise ValueError("step relative tolerance differs from the trace")
            if report.absolute_tolerance != absolute:
                raise ValueError("step absolute tolerance differs from the trace")
        if self.result.inference_admissible:
            if not self.steps or self.steps[-1].phase != "irls-final":
                raise ValueError(
                    "an admissible result requires a complete final parity step"
                )
        object.__setattr__(self, "_trace_id", _canonical_id(self.descriptor()))

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def observed_iteration_count(self) -> int:
        return 0 if not self.steps else self.steps[-1].iteration

    @property
    def dense_precision_avoided_bytes(self) -> int:
        return sum(step.parity.dense_precision_avoided_bytes for step in self.steps)

    def descriptor(self) -> Mapping[str, object]:
        return frozen_finite_json_mapping(
            {
                "schema": TREE_BLOCK_SPARSE_GAUGE_PARITY_TRACE_SCHEMA,
                "schema_version": TREE_BLOCK_SPARSE_GAUGE_PARITY_TRACE_VERSION,
                "implementation": (TREE_BLOCK_SPARSE_GAUGE_PARITY_TRACE_IMPLEMENTATION),
                "parity_implementation": (
                    TREE_SEPARATOR_GAUSSIAN_PARITY_IMPLEMENTATION
                ),
                "result_id": self.result.result_id,
                "result_inference_admissible": self.result.inference_admissible,
                "result_reason": self.result.reason,
                "maximum_condition_number": self.maximum_condition_number,
                "relative_tolerance": self.relative_tolerance,
                "absolute_tolerance": self.absolute_tolerance,
                "observed_iteration_count": self.observed_iteration_count,
                "step_count": len(self.steps),
                "dense_precision_avoided_bytes": (self.dense_precision_avoided_bytes),
                "steps": [step.to_dict() for step in self.steps],
                "claim_boundary": TREE_BLOCK_SPARSE_GAUGE_PARITY_TRACE_BOUNDARY,
            },
            name="tree-block sparse-gauge parity trace",
        )

    def to_dict(self) -> dict[str, object]:
        return {**dict(self.descriptor()), "trace_id": self.trace_id}


def update_tree_block_sparse_prior_aware_gauge_belief_with_parity_trace(
    batch: GaugeAwareObservationBatch,
    gauge: TreeSparseGaugeDesignV1,
    *,
    config: PriorAwareGaugeConfigV1 | None = None,
    node_indices: Sequence[int] | None = None,
    relative_tolerance: float = 3.0e-11,
    absolute_tolerance: float = 3.0e-12,
) -> TreeBlockSparseGaugeParityTraceV1:
    """Run the production update and require parity for every admitted IRLS system."""

    cfg = config or PriorAwareGaugeConfigV1()
    steps: list[TreeBlockSparseGaugeParityStepV1] = []

    def observe(
        phase: str,
        iteration: int,
        system: TreeBlockNormalSystemV1,
    ) -> None:
        report = require_tree_separator_gaussian_parity(
            system,
            maximum_condition_number=cfg.maximum_condition_number,
            node_indices=node_indices,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        steps.append(
            TreeBlockSparseGaugeParityStepV1(
                phase=phase,
                iteration=iteration,
                parity=report,
            )
        )

    result = _update_tree_block_sparse_prior_aware_gauge_belief(
        batch,
        gauge,
        config=cfg,
        system_observer=observe,
    )
    return TreeBlockSparseGaugeParityTraceV1(
        result=result,
        steps=tuple(steps),
        maximum_condition_number=cfg.maximum_condition_number,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )


__all__ = [
    "TREE_BLOCK_SPARSE_GAUGE_PARITY_TRACE_BOUNDARY",
    "TREE_BLOCK_SPARSE_GAUGE_PARITY_TRACE_IMPLEMENTATION",
    "TREE_BLOCK_SPARSE_GAUGE_PARITY_TRACE_SCHEMA",
    "TREE_BLOCK_SPARSE_GAUGE_PARITY_TRACE_VERSION",
    "TreeBlockSparseGaugeParityStepV1",
    "TreeBlockSparseGaugeParityTraceV1",
    "update_tree_block_sparse_prior_aware_gauge_belief_with_parity_trace",
]
