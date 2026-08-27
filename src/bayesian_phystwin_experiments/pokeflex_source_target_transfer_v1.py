"""Source-bound PokeFlex source-to-later-interaction transfer evidence.

The analysis asks whether the strength of a two-source-action maximin signal
ranks benefit on later interactions of the same physical object. Complete
physical objects, not interactions or frames, are the independent units.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from bayesian_phystwin._canonical_contracts import (
    frozen_finite_json_mapping,
    plain_json,
)
from bayesian_phystwin._portable_contracts import (
    content_id,
    load_strict_json_object,
    sha256_digest,
)

POKEFLEX_TRANSFER_SCHEMA: Final = (
    "bayesian_phystwin.pokeflex_source_target_transfer"
)
POKEFLEX_TRANSFER_VERSION: Final = 1
SOURCE_CALIBRATION_ID: Final = (
    "78d3c74e4246ec6b69cbcfe113ed04324bf1a9f49d543194df8a7a87d7f09157"
)
SOURCE_CALIBRATION_FILE_SHA256: Final = (
    "96fe0046d15dfdd150b3f2f695b678a5b2b8a6acd790978624b120f6fa6408b0"
)
FRESH6_SUMMARY_ID: Final = (
    "3bdf93d6f939d87a4ad971d64589095553fd3df59b4016767479b664ba0fb945"
)
FRESH6_TARGET_RESULT_FILE_SHA256: Final = (
    "1e8fcae19d618d52a05762ebd039e92098b52725459bf8d320124fffcaead204"
)
OFFICIAL13_SUMMARY_ID: Final = (
    "a7797cb5b318cb54e84c3cee16f33206a39fed81d5de0090409e2dc4ca00e6cf"
)
OFFICIAL13_TARGET_RESULT_FILE_SHA256: Final = (
    "619c46726aab0f7e81d2e943bd44820e521c9fe6285906add28af87203c15ebd"
)
CLAIM_BOUNDARY: Final = (
    "This is a retrospective cross-panel mechanism analysis over already "
    "opened source interactions, six prospectively locked later interactions, "
    "and the retrospective public official13 subset. It establishes only that "
    "source-action evidence strength ranks later benefit on previously studied "
    "physical objects under the frozen scale rule. It does not establish "
    "unseen-object transfer, prospective confirmation of the rank statistic, "
    "a unique physical-state interpretation, full official-split reproduction, "
    "deployment safety, or state of the art."
)


def _literal(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _finite(value: object, *, name: str) -> float:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real number")
    result = float(raw.item())
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _midranks(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(tuple(values), dtype=np.float64)
    if vector.ndim != 1 or not len(vector) or not np.all(np.isfinite(vector)):
        raise ValueError("rank values must be a nonempty finite vector")
    order = np.argsort(vector, kind="mergesort")
    ranks = np.empty(len(vector), dtype=np.float64)
    start = 0
    while start < len(vector):
        stop = start + 1
        while stop < len(vector) and vector[order[stop]] == vector[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def spearman_midrank(
    values_x: Sequence[float],
    values_y: Sequence[float],
) -> float:
    """Return Spearman correlation with deterministic average ranks."""

    x = _midranks(values_x)
    y = _midranks(values_y)
    if len(x) != len(y):
        raise ValueError("Spearman vectors must have equal length")
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denominator == 0.0:
        raise ValueError("Spearman correlation requires nonconstant vectors")
    return float(np.dot(x, y) / denominator)


def exact_spearman_permutation_tail(
    values_x: Sequence[float],
    values_y: Sequence[float],
) -> tuple[float, int, int]:
    """Return rho and an exact one-sided labelled-permutation tail.

    A subset dynamic program counts all ``n!`` assignments without materializing
    them. Midranks are doubled after centering, making every dot product integral
    even when ties produce half-ranks.
    """

    x_ranks = _midranks(values_x)
    y_ranks = _midranks(values_y)
    if len(x_ranks) != len(y_ranks):
        raise ValueError("permutation vectors must have equal length")
    count = len(x_ranks)
    if count > 14:
        raise ValueError("exact permutation analysis is limited to 14 units")
    x_scaled = np.rint(2.0 * (x_ranks - np.mean(x_ranks))).astype(np.int64)
    y_scaled = np.rint(2.0 * (y_ranks - np.mean(y_ranks))).astype(np.int64)
    observed_dot = int(np.dot(x_scaled, y_scaled))

    states: dict[int, dict[int, int]] = {0: {0: 1}}
    for position, x_value in enumerate(x_scaled):
        next_states: dict[int, dict[int, int]] = {}
        for mask, distribution in states.items():
            for target_index, y_value in enumerate(y_scaled):
                bit = 1 << target_index
                if mask & bit:
                    continue
                next_mask = mask | bit
                next_distribution = next_states.setdefault(
                    next_mask,
                    defaultdict(int),
                )
                increment = int(x_value * y_value)
                for dot_product, multiplicity in distribution.items():
                    next_distribution[dot_product + increment] += multiplicity
        states = next_states
        if len(states) != math.comb(count, position + 1):
            raise RuntimeError("permutation state count changed")

    distribution = states[(1 << count) - 1]
    denominator = math.factorial(count)
    if sum(distribution.values()) != denominator:
        raise RuntimeError("permutation multiplicity does not equal n!")
    numerator = sum(
        multiplicity
        for dot_product, multiplicity in distribution.items()
        if dot_product >= observed_dot
    )
    return spearman_midrank(values_x, values_y), numerator, denominator


def _one_sided_sign_tail(
    positive_count: int,
    negative_count: int,
) -> tuple[int, int]:
    non_ties = positive_count + negative_count
    if non_ties == 0:
        return 1, 1
    numerator = sum(
        math.comb(non_ties, successes)
        for successes in range(positive_count, non_ties + 1)
    )
    return numerator, 2**non_ties


@dataclass(frozen=True, slots=True)
class TargetInteractionGainV1:
    """One later-interaction gain over the unchanged global correction."""

    panel_id: str
    prospective: bool
    take_id: str
    relative_gain_over_global: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "panel_id", _literal(self.panel_id, name="panel_id"))
        object.__setattr__(self, "take_id", _literal(self.take_id, name="take_id"))
        if type(self.prospective) is not bool:
            raise ValueError("prospective must be a literal boolean")
        object.__setattr__(
            self,
            "relative_gain_over_global",
            _finite(
                self.relative_gain_over_global,
                name="relative_gain_over_global",
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "panel_id": self.panel_id,
            "prospective": self.prospective,
            "take_id": self.take_id,
            "relative_gain_over_global": self.relative_gain_over_global,
        }


@dataclass(frozen=True, slots=True)
class PokeFlexTransferObjectV1:
    """Source evidence and all retained later interactions for one object."""

    object_name: str
    source_take_ids: tuple[str, str]
    source_mean_relative_improvement: float
    source_minimum_relative_improvement: float
    selected_multiplier: float
    target_interactions: tuple[TargetInteractionGainV1, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "object_name",
            _literal(self.object_name, name="object_name"),
        )
        source_ids = tuple(
            _literal(value, name="source_take_id") for value in self.source_take_ids
        )
        if len(source_ids) != 2 or len(set(source_ids)) != 2:
            raise ValueError("each object requires two distinct source takes")
        object.__setattr__(self, "source_take_ids", cast(tuple[str, str], source_ids))
        object.__setattr__(
            self,
            "source_mean_relative_improvement",
            _finite(
                self.source_mean_relative_improvement,
                name="source_mean_relative_improvement",
            ),
        )
        object.__setattr__(
            self,
            "source_minimum_relative_improvement",
            _finite(
                self.source_minimum_relative_improvement,
                name="source_minimum_relative_improvement",
            ),
        )
        object.__setattr__(
            self,
            "selected_multiplier",
            _finite(self.selected_multiplier, name="selected_multiplier"),
        )
        targets = tuple(
            sorted(
                self.target_interactions,
                key=lambda value: (value.panel_id, value.take_id),
            )
        )
        if not targets or any(
            not isinstance(value, TargetInteractionGainV1) for value in targets
        ):
            raise TypeError("target_interactions must contain gain records")
        target_ids = tuple((value.panel_id, value.take_id) for value in targets)
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("target interactions must be unique")
        object.__setattr__(self, "target_interactions", targets)

    @property
    def target_mean_relative_gain(self) -> float:
        return float(
            np.mean(
                [
                    interaction.relative_gain_over_global
                    for interaction in self.target_interactions
                ]
            )
        )

    @property
    def target_minimum_relative_gain(self) -> float:
        return min(
            interaction.relative_gain_over_global
            for interaction in self.target_interactions
        )

    def to_record(self) -> dict[str, object]:
        return {
            "object_name": self.object_name,
            "source_take_ids": list(self.source_take_ids),
            "source_mean_relative_improvement": (
                self.source_mean_relative_improvement
            ),
            "source_minimum_relative_improvement": (
                self.source_minimum_relative_improvement
            ),
            "selected_multiplier": self.selected_multiplier,
            "target_interactions": [
                interaction.to_record() for interaction in self.target_interactions
            ],
            "target_mean_relative_gain": self.target_mean_relative_gain,
            "target_minimum_relative_gain": self.target_minimum_relative_gain,
        }


@dataclass(frozen=True, slots=True)
class PokeFlexSourceTargetTransferResultV1:
    """Exact object-level cross-panel transfer-calibration artifact."""

    objects: tuple[PokeFlexTransferObjectV1, ...]
    source_artifacts: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    summary_record: Mapping[str, Any] = field(init=False)
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        objects = tuple(sorted(self.objects, key=lambda value: value.object_name))
        if len(objects) < 3 or any(
            not isinstance(value, PokeFlexTransferObjectV1) for value in objects
        ):
            raise TypeError("objects must contain transfer-object records")
        names = tuple(value.object_name for value in objects)
        if len(names) != len(set(names)):
            raise ValueError("transfer objects must be unique")
        object.__setattr__(self, "objects", objects)
        object.__setattr__(
            self,
            "source_artifacts",
            frozen_finite_json_mapping(
                self.source_artifacts,
                name="source artifacts",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="result metadata"),
        )
        object.__setattr__(
            self,
            "summary_record",
            frozen_finite_json_mapping(self._build_summary(), name="summary"),
        )
        object.__setattr__(self, "artifact_id", content_id(self.descriptor()))

    @property
    def source_means(self) -> tuple[float, ...]:
        return tuple(value.source_mean_relative_improvement for value in self.objects)

    @property
    def target_means(self) -> tuple[float, ...]:
        return tuple(value.target_mean_relative_gain for value in self.objects)

    @property
    def target_minima(self) -> tuple[float, ...]:
        return tuple(value.target_minimum_relative_gain for value in self.objects)

    def _panel_spearman(self, *, prospective: bool) -> tuple[float, int, int]:
        source: list[float] = []
        target: list[float] = []
        for value in self.objects:
            panel_values = [
                item.relative_gain_over_global
                for item in value.target_interactions
                if item.prospective is prospective
            ]
            if panel_values:
                source.append(value.source_mean_relative_improvement)
                target.append(float(np.mean(panel_values)))
        return exact_spearman_permutation_tail(source, target)

    def _build_summary(self) -> dict[str, object]:
        primary_rho, primary_num, primary_den = exact_spearman_permutation_tail(
            self.source_means,
            self.target_means,
        )
        minimum_rho, minimum_num, minimum_den = (
            exact_spearman_permutation_tail(
                self.source_means,
                self.target_minima,
            )
        )
        prospective_rho, prospective_num, prospective_den = self._panel_spearman(
            prospective=True
        )
        retrospective_rho, retrospective_num, retrospective_den = (
            self._panel_spearman(prospective=False)
        )
        leave_one_out = []
        for omitted in range(len(self.objects)):
            source = tuple(
                value
                for index, value in enumerate(self.source_means)
                if index != omitted
            )
            target = tuple(
                value
                for index, value in enumerate(self.target_means)
                if index != omitted
            )
            leave_one_out.append(spearman_midrank(source, target))
        positive = sum(value > 0.0 for value in self.target_means)
        negative = sum(value < 0.0 for value in self.target_means)
        ties = len(self.objects) - positive - negative
        sign_num, sign_den = _one_sided_sign_tail(positive, negative)
        return {
            "object_count": len(self.objects),
            "target_interaction_count": sum(
                len(value.target_interactions) for value in self.objects
            ),
            "prospective_target_interaction_count": sum(
                item.prospective
                for value in self.objects
                for item in value.target_interactions
            ),
            "retrospective_target_interaction_count": sum(
                not item.prospective
                for value in self.objects
                for item in value.target_interactions
            ),
            "primary_source_mean_vs_target_mean_spearman": primary_rho,
            "primary_exact_one_sided_permutation_numerator": primary_num,
            "primary_exact_one_sided_permutation_denominator": primary_den,
            "primary_exact_one_sided_permutation_p": primary_num / primary_den,
            "minimum_target_gain_spearman": minimum_rho,
            "minimum_target_gain_exact_one_sided_permutation_numerator": (
                minimum_num
            ),
            "minimum_target_gain_exact_one_sided_permutation_denominator": (
                minimum_den
            ),
            "minimum_target_gain_exact_one_sided_permutation_p": (
                minimum_num / minimum_den
            ),
            "leave_one_object_out_spearman_minimum": min(leave_one_out),
            "leave_one_object_out_spearman_maximum": max(leave_one_out),
            "prospective_fresh6_spearman": prospective_rho,
            "prospective_fresh6_exact_one_sided_permutation_numerator": (
                prospective_num
            ),
            "prospective_fresh6_exact_one_sided_permutation_denominator": (
                prospective_den
            ),
            "prospective_fresh6_exact_one_sided_permutation_p": (
                prospective_num / prospective_den
            ),
            "retrospective_official13_overlap_spearman": retrospective_rho,
            "retrospective_official13_exact_one_sided_permutation_numerator": (
                retrospective_num
            ),
            "retrospective_official13_exact_one_sided_permutation_denominator": (
                retrospective_den
            ),
            "retrospective_official13_exact_one_sided_permutation_p": (
                retrospective_num / retrospective_den
            ),
            "positive_object_count": positive,
            "negative_object_count": negative,
            "tied_object_count": ties,
            "positive_object_exact_sign_numerator": sign_num,
            "positive_object_exact_sign_denominator": sign_den,
            "positive_object_exact_sign_p": sign_num / sign_den,
        }

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": POKEFLEX_TRANSFER_SCHEMA,
            "schema_version": POKEFLEX_TRANSFER_VERSION,
            "artifact_kind": "PokeFlexSourceTargetTransferResultV1",
            "analysis_role": "retrospective-cross-panel-mechanism-diagnostic",
            "statistical_unit": "physical-object-v1",
            "source_artifacts": plain_json(self.source_artifacts),
            "objects": [value.to_record() for value in self.objects],
            "summary": plain_json(self.summary_record),
            "claim_boundary": CLAIM_BOUNDARY,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


def build_pokeflex_source_target_transfer(
    calibration: Mapping[str, Any],
    fresh6: Mapping[str, Any],
    official13: Mapping[str, Any],
) -> PokeFlexSourceTargetTransferResultV1:
    """Build the frozen result from the three retained public artifacts."""

    if calibration.get("artifact_kind") != "PokeFlexActionRobustScaleCalibration":
        raise ValueError("unexpected source calibration artifact")
    if calibration.get("calibration_sha256") != SOURCE_CALIBRATION_ID:
        raise ValueError("source calibration identity changed")
    if fresh6.get("artifact_kind") != "PokeFlexActionRobustFresh6V3Summary":
        raise ValueError("unexpected fresh6 summary artifact")
    if fresh6.get("summary_sha256") != FRESH6_SUMMARY_ID:
        raise ValueError("fresh6 summary identity changed")
    if official13.get("artifact_kind") != (
        "PokeFlexActionRobustOfficial13PublicV1Summary"
    ):
        raise ValueError("unexpected official13 summary artifact")
    if official13.get("summary_sha256") != OFFICIAL13_SUMMARY_ID:
        raise ValueError("official13 summary identity changed")

    calibration_objects = _mapping(calibration.get("objects"), name="source objects")
    targets: dict[str, list[TargetInteractionGainV1]] = defaultdict(list)
    panel_specs = (
        ("prospective-fresh6-v3", True, fresh6),
        ("retrospective-official13-public-v1", False, official13),
    )
    for panel_id, prospective, payload in panel_specs:
        rows = _sequence(payload.get("objects"), name=f"{panel_id} objects")
        for raw_row in rows:
            row = _mapping(raw_row, name=f"{panel_id} object row")
            object_name = _literal(row.get("object_name"), name="object_name")
            if object_name not in calibration_objects:
                continue
            targets[object_name].append(
                TargetInteractionGainV1(
                    panel_id=panel_id,
                    prospective=prospective,
                    take_id=_literal(row.get("take_id"), name="take_id"),
                    relative_gain_over_global=_finite(
                        row.get("action_robust_vs_global_relative_improvement"),
                        name="action_robust_vs_global_relative_improvement",
                    ),
                )
            )

    objects = []
    for object_name, target_rows in targets.items():
        source = _mapping(
            calibration_objects[object_name],
            name=f"source object {object_name}",
        )
        source_take_ids = tuple(
            _literal(value, name="source_take_id")
            for value in _sequence(
                source.get("source_take_ids"),
                name="source_take_ids",
            )
        )
        if len(source_take_ids) != 2:
            raise ValueError("source_take_ids must contain exactly two takes")
        objects.append(
            PokeFlexTransferObjectV1(
                object_name=object_name,
                source_take_ids=cast(tuple[str, str], source_take_ids),
                source_mean_relative_improvement=_finite(
                    source.get("mean_source_relative_improvement"),
                    name="mean_source_relative_improvement",
                ),
                source_minimum_relative_improvement=_finite(
                    source.get("minimum_source_relative_improvement"),
                    name="minimum_source_relative_improvement",
                ),
                selected_multiplier=_finite(
                    source.get("multiplier"),
                    name="multiplier",
                ),
                target_interactions=tuple(target_rows),
            )
        )

    source_artifacts = {
        "calibration": {
            "path": "configs/sota/pokeflex_action_robust_scale_v3.json",
            "canonical_id": SOURCE_CALIBRATION_ID,
            "file_sha256": SOURCE_CALIBRATION_FILE_SHA256,
        },
        "prospective_fresh6": {
            "path": "results/sota/pokeflex_action_robust_fresh6_v3/summary.json",
            "canonical_id": FRESH6_SUMMARY_ID,
            "target_result_file_sha256": FRESH6_TARGET_RESULT_FILE_SHA256,
        },
        "retrospective_official13": {
            "path": (
                "results/sota/pokeflex_action_robust_official13_public_v1/"
                "summary.json"
            ),
            "canonical_id": OFFICIAL13_SUMMARY_ID,
            "target_result_file_sha256": OFFICIAL13_TARGET_RESULT_FILE_SHA256,
        },
    }
    return PokeFlexSourceTargetTransferResultV1(
        objects=tuple(objects),
        source_artifacts=source_artifacts,
        metadata={
            "object_aggregation": (
                "equal mean over retained target interactions per physical object"
            ),
            "minimum_gain_sensitivity": (
                "minimum retained target gain per physical object"
            ),
            "permutation_method": (
                "exact labelled midrank-Spearman permutation via subset dynamic "
                "program"
            ),
        },
    )


def build_from_repository_root(
    repository_root: str | Path,
) -> PokeFlexSourceTargetTransferResultV1:
    """Load the three retained artifacts below one repository checkout."""

    root = Path(repository_root)
    calibration = load_strict_json_object(
        root / "configs/sota/pokeflex_action_robust_scale_v3.json",
        label="PokeFlex source calibration",
    )
    fresh6 = load_strict_json_object(
        root / "results/sota/pokeflex_action_robust_fresh6_v3/summary.json",
        label="PokeFlex fresh6 summary",
    )
    official13 = load_strict_json_object(
        root
        / "results/sota/pokeflex_action_robust_official13_public_v1/summary.json",
        label="PokeFlex official13 summary",
    )
    return build_pokeflex_source_target_transfer(
        calibration,
        fresh6,
        official13,
    )


def validate_artifact_identity(record: Mapping[str, Any]) -> str:
    """Validate an externally loaded result record's content identity."""

    artifact_id = sha256_digest(record.get("artifact_id"), name="artifact_id")
    descriptor = dict(record)
    descriptor.pop("artifact_id")
    if content_id(descriptor) != artifact_id:
        raise ValueError("artifact_id does not match result content")
    return artifact_id


__all__ = [
    "CLAIM_BOUNDARY",
    "PokeFlexSourceTargetTransferResultV1",
    "PokeFlexTransferObjectV1",
    "TargetInteractionGainV1",
    "build_from_repository_root",
    "build_pokeflex_source_target_transfer",
    "exact_spearman_permutation_tail",
    "spearman_midrank",
    "validate_artifact_identity",
]
