#!/usr/bin/env python3
"""Develop a source-only cross-window identity rule before target locking."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from prob4d.causal_tracklets import CausalTrackletSet
from prob4d.cross_window_tracklets import (
    CrossWindowAssociationConfig,
    CrossWindowAssociationResult,
    associate_cross_window_tracklets,
)
from prob4d.sim3 import Sim3

from bayesian_phystwin._gauge_aware_contracts import (
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
    GaugeAwareObservationBatch,
)
from bayesian_phystwin._prior_aware_gauge_math import PriorAwareGaugeConfigV1
from bayesian_phystwin.prior_aware_gauge_belief import (
    update_prior_aware_gauge_belief,
)

_SCIENCE_DIRECTORY = Path(__file__).resolve().parent
if str(_SCIENCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_SCIENCE_DIRECTORY))

from prob4d_bpt_controlled_decisive_core_v1 import (  # noqa: E402
    BASELINE_METHOD,
    FINITE_INFINITY,
    Candidate,
    CandidateScore,
    GroupData,
    StudyConfig,
    TrialResult,
    _canonical_sha256,
    _query_covariance,
    _require,
    _risk_from_result,
    _sha256,
    _write_json,
    generate_group,
)
from run_prob4d_bpt_controlled_decisive_v1 import (  # noqa: E402
    _paired_interval,
    apply_guard,
    calibrate_guard,
    score_candidate,
)

PROTOCOL_SCHEMA = "bayesian-phystwin-prob4d-cross-window-identity-development"
REPORT_SCHEMA = "bayesian-phystwin-prob4d-cross-window-identity-development-report"
BASELINE = BASELINE_METHOD
FRAMEWISE = "B1_newest_frame_explicit_gauge"
NEWEST_WINDOW = "P0_newest_window_persistent"
NAIVE_MERGE = "P1_naive_local_id_cross_window_merge"
SOURCE_LINKED = "P2_source_linked_cross_window_identity"
ORACLE_LINKED = "P3_oracle_cross_window_identity"
METHODS = (
    BASELINE,
    FRAMEWISE,
    NEWEST_WINDOW,
    NAIVE_MERGE,
    SOURCE_LINKED,
    ORACLE_LINKED,
)
PRIMARY_METHOD = SOURCE_LINKED
REFERENCE_METHOD = NEWEST_WINDOW


@dataclass(frozen=True, slots=True)
class AssociationCandidateConfig:
    """One source-only association configuration in the development grid."""

    use_covariance: bool
    configuration: CrossWindowAssociationConfig

    def descriptor(self) -> dict[str, object]:
        return {
            "use_covariance": self.use_covariance,
            "configuration": self.configuration.to_dict(),
        }

    @property
    def configuration_id(self) -> str:
        return _canonical_sha256(self.descriptor())


@dataclass(frozen=True, slots=True)
class AssociationContext:
    """Synthetic overlapping window tracklets and their hidden identity labels."""

    left: CausalTrackletSet
    right: CausalTrackletSet
    left_global_from_local: Sim3
    right_global_from_local: Sim3
    left_covariance_m2: np.ndarray
    right_covariance_m2: np.ndarray
    right_local_to_true: tuple[int, ...]
    true_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class AssociationCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    accepted: int = 0
    truth: int = 0

    def add(self, other: AssociationCounts) -> AssociationCounts:
        return AssociationCounts(
            true_positive=self.true_positive + other.true_positive,
            false_positive=self.false_positive + other.false_positive,
            false_negative=self.false_negative + other.false_negative,
            accepted=self.accepted + other.accepted,
            truth=self.truth + other.truth,
        )

    def metrics(self) -> dict[str, float | int]:
        precision = self.true_positive / self.accepted if self.accepted else 1.0
        recall = self.true_positive / self.truth if self.truth else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        return {
            **asdict(self),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


@dataclass(frozen=True, slots=True)
class GroupAssociation:
    """One association result and target-hidden evaluation counts."""

    context: AssociationContext
    result: CrossWindowAssociationResult
    counts: AssociationCounts


@dataclass(frozen=True, slots=True)
class Partition:
    groups_per_scenario: int
    seed_start: int


@dataclass(frozen=True, slots=True)
class DevelopmentProtocol:
    raw: dict[str, Any]
    base_config: StudyConfig
    association_partition: Partition
    pilot_guard_partition: Partition
    pilot_evaluation_partition: Partition
    minimum_precision: float
    minimum_recall: float
    minimum_pilot_improvement: float
    maximum_harmful_rate: float
    scenarios: tuple[str, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-revision", required=True)
    parser.add_argument("--prob4d-revision", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _number(value: object, *, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _partition(value: Mapping[str, Any], *, name: str) -> Partition:
    return Partition(
        groups_per_scenario=_integer(
            value.get("groups_per_scenario"),
            name=f"{name}.groups_per_scenario",
            minimum=1,
        ),
        seed_start=_integer(
            value.get("seed_start"),
            name=f"{name}.seed_start",
            minimum=0,
        ),
    )


def load_protocol(path: Path) -> DevelopmentProtocol:
    raw = json.loads(path.read_text(encoding="utf-8"))
    _require(raw.get("schema") == PROTOCOL_SCHEMA, "unexpected protocol schema")
    _require(raw.get("schema_version") == 1, "unexpected protocol version")
    _require(
        raw.get("status") == "development-only-before-confirmatory-target-lock",
        "development protocol status changed",
    )
    _require(tuple(raw.get("methods", ())) == METHODS, "method registry changed")
    target = raw.get("confirmatory_target")
    _require(isinstance(target, Mapping), "confirmatory_target must be an object")
    _require(target.get("seeds_committed") is False, "target seeds opened too early")
    _require("seed_start" not in target, "development protocol contains target seeds")

    geometry = raw["geometry"]
    generator = raw["generator"]
    scenarios = tuple(map(str, raw["scenarios"]))
    partitions = raw["development_partitions"]
    guard = raw["guard_calibration"]
    pins = raw["repository_pins"]
    base = StudyConfig(
        point_count=_integer(geometry["point_count"], name="point_count", minimum=8),
        frame_count=_integer(geometry["frame_count"], name="frame_count", minimum=4),
        state_count=_integer(geometry["state_count"], name="state_count", minimum=1),
        bias_mode_count=_integer(
            geometry["bias_mode_count"], name="bias_mode_count", minimum=1
        ),
        window_count=2,
        calibration_groups_per_scenario=1,
        target_groups_per_scenario=1,
        calibration_seed=0,
        target_seed=1,
        bootstrap_resamples=_integer(
            raw["bootstrap"]["resamples"],
            name="bootstrap.resamples",
            minimum=100,
        ),
        bootstrap_seed=_integer(
            raw["bootstrap"]["seed"], name="bootstrap.seed", minimum=0
        ),
        harmful_margin_m=_number(
            raw["endpoints"]["harmful_margin_m"],
            name="endpoints.harmful_margin_m",
        ),
        guard_harmful_rate_at_most=_number(
            guard["harmful_accepted_rate_at_most"],
            name="guard.harmful_accepted_rate_at_most",
        ),
        guard_minimum_accepted_groups=_integer(
            guard["minimum_accepted_groups"],
            name="guard.minimum_accepted_groups",
            minimum=1,
        ),
        conditional_noise_std_m=_number(
            generator["conditional_noise_std_m"],
            name="generator.conditional_noise_std_m",
            minimum=1e-12,
        ),
        state_mode_maximum_m=_number(
            generator["state_mode_maximum_m"],
            name="generator.state_mode_maximum_m",
            minimum=1e-12,
        ),
        query_progress=_number(
            generator["query_progress"],
            name="generator.query_progress",
            minimum=1e-12,
        ),
        state_prior_std=_number(
            generator["state_prior_std"],
            name="generator.state_prior_std",
            minimum=1e-12,
        ),
        source_revision=str(pins["prob4d"]),
        scenarios=scenarios,
    )
    gates = raw["development_gates"]
    protocol = DevelopmentProtocol(
        raw=raw,
        base_config=base,
        association_partition=_partition(
            partitions["association_configuration"],
            name="association_configuration",
        ),
        pilot_guard_partition=_partition(
            partitions["pilot_guard_calibration"],
            name="pilot_guard_calibration",
        ),
        pilot_evaluation_partition=_partition(
            partitions["pilot_evaluation"],
            name="pilot_evaluation",
        ),
        minimum_precision=_number(
            gates["association_precision_at_least"],
            name="association_precision_at_least",
        ),
        minimum_recall=_number(
            gates["association_recall_at_least"],
            name="association_recall_at_least",
        ),
        minimum_pilot_improvement=_number(
            gates["source_vs_newest_improvement_fraction_at_least"],
            name="source_vs_newest_improvement_fraction_at_least",
        ),
        maximum_harmful_rate=_number(
            gates["harmful_accepted_rate_at_most"],
            name="harmful_accepted_rate_at_most",
        ),
        scenarios=scenarios,
    )
    starts = {
        protocol.association_partition.seed_start,
        protocol.pilot_guard_partition.seed_start,
        protocol.pilot_evaluation_partition.seed_start,
    }
    _require(len(starts) == 3, "development partition seed starts overlap")
    return protocol


def _scenario_association_parameters(scenario: str) -> dict[str, float]:
    table = {
        "nominal_correlated": {
            "point_noise_m": 0.0010,
            "gauge_error_m": 0.0015,
            "retain_fraction": 1.00,
            "outlier_fraction": 0.00,
            "outlier_shift_m": 0.000,
            "ambiguity_fraction": 0.00,
            "ambiguity_own_fraction": 1.00,
            "link_probability": 0.97,
        },
        "common_mode_bias": {
            "point_noise_m": 0.0012,
            "gauge_error_m": 0.0035,
            "retain_fraction": 0.92,
            "outlier_fraction": 0.00,
            "outlier_shift_m": 0.000,
            "ambiguity_fraction": 0.00,
            "ambiguity_own_fraction": 1.00,
            "link_probability": 0.95,
        },
        "outlier_groups": {
            "point_noise_m": 0.0015,
            "gauge_error_m": 0.0025,
            "retain_fraction": 0.90,
            "outlier_fraction": 0.18,
            "outlier_shift_m": 0.018,
            "ambiguity_fraction": 0.00,
            "ambiguity_own_fraction": 1.00,
            "link_probability": 0.91,
        },
        "weak_identifiability": {
            "point_noise_m": 0.0014,
            "gauge_error_m": 0.0025,
            "retain_fraction": 0.88,
            "outlier_fraction": 0.00,
            "outlier_shift_m": 0.000,
            "ambiguity_fraction": 0.28,
            "ambiguity_own_fraction": 0.54,
            "link_probability": 0.89,
        },
        "large_gauge_uncertainty": {
            "point_noise_m": 0.0018,
            "gauge_error_m": 0.0075,
            "retain_fraction": 0.88,
            "outlier_fraction": 0.00,
            "outlier_shift_m": 0.000,
            "ambiguity_fraction": 0.08,
            "ambiguity_own_fraction": 0.65,
            "link_probability": 0.90,
        },
        "mixed_stress": {
            "point_noise_m": 0.0022,
            "gauge_error_m": 0.0060,
            "retain_fraction": 0.72,
            "outlier_fraction": 0.18,
            "outlier_shift_m": 0.016,
            "ambiguity_fraction": 0.22,
            "ambiguity_own_fraction": 0.52,
            "link_probability": 0.84,
        },
    }
    if scenario not in table:
        raise ValueError(f"unknown association scenario: {scenario}")
    return table[scenario]


def _group_seed(group: GroupData) -> int:
    return int(group.group_id.rsplit("-", 1)[1])


def _random_sim3(rng: np.random.Generator) -> Sim3:
    vector = np.concatenate(
        (
            rng.normal(scale=0.012, size=1),
            rng.normal(scale=0.035, size=3),
            rng.normal(scale=0.025, size=3),
        )
    )
    return Sim3.from_vector(vector)


def _gauge_error_transform(
    rng: np.random.Generator,
    error_m: float,
) -> Sim3:
    radius = 0.9
    vector = np.concatenate(
        (
            rng.normal(scale=error_m / radius, size=1),
            rng.normal(scale=error_m / radius, size=3),
            rng.normal(scale=error_m, size=3),
        )
    )
    return Sim3.from_vector(vector)


def _tracklet_set(
    *,
    window_id: str,
    absolute_frames: tuple[int, ...],
    global_points: np.ndarray,
    true_global_from_local: Sim3,
    link_probability: float,
) -> CausalTrackletSet:
    track_count = global_points.shape[1]
    local_points = true_global_from_local.inverse().transform_points(global_points)
    track_ids: list[int] = []
    frame_indices: list[int] = []
    local_indices: list[int] = []
    rows: list[int] = []
    columns: list[int] = []
    points: list[np.ndarray] = []
    links: list[float] = []
    associations: list[float] = []
    for track_id in range(track_count):
        cumulative = 1.0
        for local_index, frame in enumerate(absolute_frames):
            link = 1.0 if local_index == 0 else link_probability
            cumulative *= link
            track_ids.append(track_id)
            frame_indices.append(frame)
            local_indices.append(local_index)
            rows.append(track_id)
            columns.append(0)
            points.append(local_points[local_index, track_id])
            links.append(link)
            associations.append(cumulative)
    return CausalTrackletSet(
        window_id=window_id,
        causal_frame_stop=max(absolute_frames) + 1,
        source_shape=(len(absolute_frames), track_count, 1),
        seed_frame_index=absolute_frames[0],
        track_ids=np.asarray(track_ids, dtype=np.int64),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        local_frame_indices=np.asarray(local_indices, dtype=np.int64),
        rows=np.asarray(rows, dtype=np.int64),
        columns=np.asarray(columns, dtype=np.int64),
        points_local=np.asarray(points, dtype=np.float64),
        link_probability=np.asarray(links, dtype=np.float64),
        association_probability=np.asarray(associations, dtype=np.float64),
        metadata={
            "study": "prob4d-cross-window-identity-development-v1",
            "source_only": True,
        },
    )


def build_association_context(group: GroupData) -> AssociationContext:
    """Create two overlapping, independently gauged windows from one group."""

    rng = np.random.default_rng(_group_seed(group) + 31_415_927)
    config = group.stack
    point_count = len(np.unique(config.point_ids))
    frame_count = len(np.unique(config.frame_indices))
    _require(frame_count == 4, "development study requires four factor frames")
    state_grid = group.state_jacobian.reshape(frame_count, point_count, 3, -1)
    physical_grid = group.physical_prediction_m.reshape(frame_count, point_count, 3)
    state_signal = np.einsum("tncs,s->tnc", state_grid, group.true_state, optimize=True)
    trajectory = physical_grid + state_signal
    parameters = _scenario_association_parameters(group.scenario)

    retain_count = max(
        8,
        int(np.ceil(parameters["retain_fraction"] * point_count)),
    )
    retain_count = min(retain_count, point_count)
    retained_true = np.sort(rng.choice(point_count, size=retain_count, replace=False))
    right_local_to_true = tuple(int(value) for value in rng.permutation(retained_true))

    left_frames = (0, 1, 2)
    right_frames = (1, 2, 3)
    left_global = trajectory[np.asarray(left_frames)].copy()
    right_global = trajectory[
        np.asarray(right_frames)[:, None],
        np.asarray(right_local_to_true)[None, :],
    ].copy()
    point_noise = parameters["point_noise_m"]
    left_global += rng.normal(scale=point_noise, size=left_global.shape)
    right_global += rng.normal(scale=point_noise, size=right_global.shape)

    ambiguous_count = int(
        np.floor(parameters["ambiguity_fraction"] * retain_count / 2.0)
    )
    if ambiguous_count:
        shuffled = rng.permutation(retain_count)[: 2 * ambiguous_count]
        own_fraction = parameters["ambiguity_own_fraction"]
        for first, second in shuffled.reshape(-1, 2):
            first_points = right_global[:2, first].copy()
            second_points = right_global[:2, second].copy()
            right_global[:2, first] = (
                own_fraction * first_points + (1.0 - own_fraction) * second_points
            )
            right_global[:2, second] = (
                own_fraction * second_points + (1.0 - own_fraction) * first_points
            )

    outlier_count = int(np.ceil(parameters["outlier_fraction"] * retain_count))
    if outlier_count:
        for local_id in rng.choice(
            retain_count,
            size=min(outlier_count, retain_count),
            replace=False,
        ):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            overlap_frame = int(rng.integers(0, 2))
            right_global[overlap_frame, int(local_id)] += (
                parameters["outlier_shift_m"] * direction
            )

    left_true_gauge = _random_sim3(rng)
    right_true_gauge = _random_sim3(rng)
    left_estimated_gauge = _gauge_error_transform(
        rng, parameters["gauge_error_m"]
    ).compose(left_true_gauge)
    right_estimated_gauge = _gauge_error_transform(
        rng, parameters["gauge_error_m"]
    ).compose(right_true_gauge)
    left = _tracklet_set(
        window_id=f"{group.group_id}:left",
        absolute_frames=left_frames,
        global_points=left_global,
        true_global_from_local=left_true_gauge,
        link_probability=parameters["link_probability"],
    )
    right = _tracklet_set(
        window_id=f"{group.group_id}:right",
        absolute_frames=right_frames,
        global_points=right_global,
        true_global_from_local=right_true_gauge,
        link_probability=parameters["link_probability"],
    )
    marginal_sigma = np.hypot(point_noise, parameters["gauge_error_m"])
    left_covariance = np.repeat(
        (np.eye(3) * marginal_sigma**2)[None],
        left.observation_count,
        axis=0,
    )
    right_covariance = np.repeat(
        (np.eye(3) * marginal_sigma**2)[None],
        right.observation_count,
        axis=0,
    )
    true_pairs = tuple(
        sorted(
            (true_id, local_id) for local_id, true_id in enumerate(right_local_to_true)
        )
    )
    return AssociationContext(
        left=left,
        right=right,
        left_global_from_local=left_estimated_gauge,
        right_global_from_local=right_estimated_gauge,
        left_covariance_m2=left_covariance,
        right_covariance_m2=right_covariance,
        right_local_to_true=right_local_to_true,
        true_pairs=true_pairs,
    )


def association_configurations(
    protocol: Mapping[str, Any],
) -> tuple[AssociationCandidateConfig, ...]:
    grid = protocol["association_grid"]
    fixed = grid["fixed"]
    configurations: list[AssociationCandidateConfig] = []
    for (
        use_covariance,
        scale,
        maximum_rms,
        minimum_score,
        minimum_margin,
    ) in itertools.product(
        grid["use_covariance"],
        grid["isotropic_distance_scale_m"],
        grid["maximum_weighted_rms_m"],
        grid["minimum_compatibility_score"],
        grid["minimum_score_margin"],
    ):
        maximum_distance = max(
            float(fixed["minimum_spatial_gate_m"]),
            float(fixed["spatial_gate_multiplier"]) * float(maximum_rms),
        )
        configurations.append(
            AssociationCandidateConfig(
                use_covariance=bool(use_covariance),
                configuration=CrossWindowAssociationConfig(
                    minimum_shared_frames=int(fixed["minimum_shared_frames"]),
                    minimum_effective_support=float(fixed["minimum_effective_support"]),
                    isotropic_distance_scale_m=float(scale),
                    covariance_floor_m2=float(fixed["covariance_floor_m2"]),
                    maximum_weighted_rms_m=float(maximum_rms),
                    maximum_shared_frame_distance_m=maximum_distance,
                    maximum_spatial_candidate_pairs=int(
                        fixed["maximum_spatial_candidate_pairs"]
                    ),
                    minimum_compatibility_score=float(minimum_score),
                    minimum_score_margin=float(minimum_margin),
                ),
            )
        )
    return tuple(sorted(configurations, key=lambda item: item.configuration_id))


def run_association(
    context: AssociationContext,
    candidate: AssociationCandidateConfig,
) -> CrossWindowAssociationResult:
    kwargs: dict[str, Any] = {}
    if candidate.use_covariance:
        kwargs = {
            "left_global_covariance_m2": context.left_covariance_m2,
            "right_global_covariance_m2": context.right_covariance_m2,
        }
    return associate_cross_window_tracklets(
        context.left,
        context.right,
        left_global_from_local=context.left_global_from_local,
        right_global_from_local=context.right_global_from_local,
        configuration=candidate.configuration,
        candidate_chunk_size=64,
        **kwargs,
    )


def association_counts(
    context: AssociationContext,
    result: CrossWindowAssociationResult,
) -> AssociationCounts:
    accepted = set(result.accepted_pairs)
    truth = set(context.true_pairs)
    true_positive = len(accepted & truth)
    return AssociationCounts(
        true_positive=true_positive,
        false_positive=len(accepted - truth),
        false_negative=len(truth - accepted),
        accepted=len(accepted),
        truth=len(truth),
    )


def select_association_configuration(
    groups: Sequence[GroupData],
    protocol: DevelopmentProtocol,
) -> tuple[
    AssociationCandidateConfig,
    dict[str, Any],
    dict[str, AssociationContext],
]:
    contexts = {group.group_id: build_association_context(group) for group in groups}
    summaries: list[dict[str, Any]] = []
    selected_candidate: AssociationCandidateConfig | None = None
    selected_key: tuple[Any, ...] | None = None
    for candidate in association_configurations(protocol.raw):
        total = AssociationCounts()
        by_scenario: dict[str, AssociationCounts] = {
            scenario: AssociationCounts() for scenario in protocol.scenarios
        }
        for group in groups:
            context = contexts[group.group_id]
            counts = association_counts(context, run_association(context, candidate))
            total = total.add(counts)
            by_scenario[group.scenario] = by_scenario[group.scenario].add(counts)
        metrics = total.metrics()
        scenario_metrics = {
            scenario: counts.metrics() for scenario, counts in by_scenario.items()
        }
        eligible = bool(
            metrics["precision"] >= protocol.minimum_precision
            and metrics["recall"] >= protocol.minimum_recall
        )
        summary = {
            "configuration_id": candidate.configuration_id,
            **candidate.descriptor(),
            "aggregate": metrics,
            "by_scenario": scenario_metrics,
            "development_gate_eligible": eligible,
        }
        summaries.append(summary)
        key = (
            0 if eligible else 1,
            -float(metrics["f1"]),
            -float(metrics["recall"]),
            -float(metrics["precision"]),
            int(metrics["false_positive"]),
            candidate.configuration_id,
        )
        if selected_key is None or key < selected_key:
            selected_key = key
            selected_candidate = candidate
    assert selected_candidate is not None
    selected = next(
        item
        for item in summaries
        if item["configuration_id"] == selected_candidate.configuration_id
    )
    return (
        selected_candidate,
        {
            "selection_rule": (
                "eligible precision/recall gate first, then maximum micro F1, "
                "recall, precision, minimum false links, and configuration ID"
            ),
            "candidate_count": len(summaries),
            "selected": selected,
            "candidates": summaries,
        },
        contexts,
    )


def _selected_rows(
    group: GroupData,
    association: GroupAssociation,
    method_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    point_count = len(np.unique(group.stack.point_ids))
    specifications: list[tuple[int, int, int]] = []
    right_true = association.context.right_local_to_true

    def add_newest_window() -> None:
        for frame in (2, 3):
            specifications.extend((frame, true_id, true_id) for true_id in right_true)

    if method_id == FRAMEWISE:
        specifications.extend((3, true_id, true_id) for true_id in right_true)
    elif method_id == NEWEST_WINDOW:
        add_newest_window()
    elif method_id == NAIVE_MERGE:
        add_newest_window()
        for frame in (0, 1):
            specifications.extend(
                (frame, local_id, true_id)
                for local_id, true_id in enumerate(right_true)
            )
    elif method_id == SOURCE_LINKED:
        add_newest_window()
        for left_id, right_id in association.result.accepted_pairs:
            true_id = right_true[right_id]
            specifications.extend((frame, left_id, true_id) for frame in (0, 1))
    elif method_id == ORACLE_LINKED:
        add_newest_window()
        for true_id in right_true:
            specifications.extend((frame, true_id, true_id) for frame in (0, 1))
    else:
        raise ValueError(f"unknown observation method: {method_id}")

    row_indices = np.asarray(
        [frame * point_count + true_id for frame, true_id, _ in specifications],
        dtype=np.int64,
    )
    assigned_ids = np.asarray(
        [assigned for _, _, assigned in specifications],
        dtype=np.int64,
    )
    return row_indices, assigned_ids


def _batch_for_method(
    group: GroupData,
    association: GroupAssociation,
    method_id: str,
    config: StudyConfig,
) -> GaugeAwareObservationBatch:
    row_indices, assigned_ids = _selected_rows(group, association, method_id)
    _require(len(row_indices) > 0, "method selected no observation rows")
    point_count = len(np.unique(group.stack.point_ids))
    frame_count = len(np.unique(group.stack.frame_indices))
    state_grid = group.state_jacobian.reshape(
        frame_count,
        point_count,
        3,
        config.state_count,
    )
    selected_frames = group.stack.frame_indices[row_indices]
    state = state_grid[selected_frames, assigned_ids]
    innovation = (
        group.stack.world_mean_m[row_indices] - group.physical_prediction_m[row_indices]
    )
    gauge = group.stack.dense_gauge_jacobian()[row_indices]
    return GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=(
            group.stack.conditional_world_covariance_m2[row_indices]
        ),
        state_jacobian=state,
        gauge_jacobian=gauge,
        shared_bias_jacobian=group.shared_bias_jacobian[row_indices],
        view_bias_jacobian=np.zeros((len(row_indices), 3, 0), dtype=np.float64),
        query_state_jacobian=group.query_state_jacobian,
        gauge_prior_covariance=group.stack.gauge_prior_covariance,
        correlation_group_ids=tuple(
            group.stack.correlation_group_ids[index] for index in row_indices
        ),
        prior_reliability=group.stack.prior_reliability[row_indices],
        prior_nominal_probability=(group.stack.prior_nominal_probability[row_indices]),
        composite_weight=group.stack.composite_weight[row_indices],
        state_prior_covariance_m2=(
            np.eye(config.state_count, dtype=np.float64) * config.state_prior_std**2
        ),
        physical_response_scale_m=group.physical_response_scale_m,
        composite_weight_mode=COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
        metadata={
            "study": "prob4d-cross-window-identity-development-v1",
            "group_id": group.group_id,
            "scenario": group.scenario,
            "method_id": method_id,
            "association_result_id": association.result.result_id,
            "accepted_cross_window_links": len(association.result.links),
            "target_identity_labels_used_for_batch": method_id == ORACLE_LINKED,
        },
    )


def make_candidate(
    group: GroupData,
    association: GroupAssociation,
    method_id: str,
    config: StudyConfig,
) -> Candidate:
    if method_id == BASELINE:
        return Candidate(
            method_id=method_id,
            inference_admissible=False,
            reason="physical-fallback-reference",
            correction_m=np.zeros_like(group.true_query_correction_m),
            covariance_m2=np.zeros(
                (len(group.true_query_correction_m), 3, 3), dtype=np.float64
            ),
            risk_score=FINITE_INFINITY,
            nominal_probability=1.0,
            identifiable_fraction=0.0,
            query_sensitivity_fraction=0.0,
            fixed_point_converged=True,
        )
    batch = _batch_for_method(group, association, method_id, config)
    result = update_prior_aware_gauge_belief(
        batch,
        config=PriorAwareGaugeConfigV1(
            state_prior_std_m=config.state_prior_std,
            shared_bias_prior_std_m=0.012,
            view_bias_prior_std_m=0.010,
            effective_samples_per_correlation_group=12.0,
            degrees_of_freedom=5.0,
            outlier_covariance_multiplier=36.0,
            maximum_iterations=20,
            maximum_condition_number=1e13,
            minimum_conditional_information_fraction=1e-5,
            minimum_identifiable_fraction=0.02,
            minimum_query_sensitivity_fraction=1e-4,
            maximum_state_update_m=0.065,
            maximum_update_to_physical_response_ratio=4.0,
        ),
    )
    correction = (
        np.einsum(
            "ncs,s->nc",
            group.query_state_jacobian,
            result.state_coefficients,
            optimize=True,
        )
        if result.inference_admissible
        else np.zeros_like(group.true_query_correction_m)
    )
    state_covariance = result.posterior_covariance[
        : config.state_count, : config.state_count
    ]
    covariance = _query_covariance(group.query_state_jacobian, state_covariance)
    risk, nominal, identifiable, sensitivity, converged = _risk_from_result(
        group, result, covariance
    )
    return Candidate(
        method_id=method_id,
        inference_admissible=bool(result.inference_admissible),
        reason=str(result.reason),
        correction_m=correction,
        covariance_m2=covariance,
        risk_score=risk,
        nominal_probability=nominal,
        identifiable_fraction=identifiable,
        query_sensitivity_fraction=sensitivity,
        fixed_point_converged=converged,
    )


def _groups(
    protocol: DevelopmentProtocol,
    partition: Partition,
    *,
    prefix: str,
) -> list[GroupData]:
    groups: list[GroupData] = []
    for scenario_index, scenario in enumerate(protocol.scenarios):
        for offset in range(partition.groups_per_scenario):
            seed = partition.seed_start + 100_000 * scenario_index + offset
            groups.append(
                generate_group(
                    seed,
                    scenario,
                    protocol.base_config,
                    group_prefix=prefix,
                )
            )
    return groups


def _group_association(
    group: GroupData,
    selected: AssociationCandidateConfig,
    context: AssociationContext | None = None,
) -> GroupAssociation:
    resolved = context or build_association_context(group)
    result = run_association(resolved, selected)
    return GroupAssociation(
        context=resolved,
        result=result,
        counts=association_counts(resolved, result),
    )


def _score_partition(
    groups: Sequence[GroupData],
    selected: AssociationCandidateConfig,
    config: StudyConfig,
) -> tuple[dict[str, list[CandidateScore]], AssociationCounts]:
    scores = {method_id: [] for method_id in METHODS}
    association_total = AssociationCounts()
    for group in groups:
        association = _group_association(group, selected)
        association_total = association_total.add(association.counts)
        for method_id in METHODS:
            candidate = make_candidate(group, association, method_id, config)
            scores[method_id].append(
                score_candidate(group, candidate, config.harmful_margin_m)
            )
    return scores, association_total


def _aggregate_trials(
    trials: Sequence[TrialResult],
    protocol: DevelopmentProtocol,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    by_method = {
        method_id: [trial for trial in trials if trial.method_id == method_id]
        for method_id in METHODS
    }
    newest = by_method[REFERENCE_METHOD]
    newest_rmse = np.asarray([trial.deployed_rmse_m for trial in newest])
    for method_index, method_id in enumerate(METHODS):
        rows = by_method[method_id]
        deployed = np.asarray([trial.deployed_rmse_m for trial in rows])
        baseline = np.asarray([trial.baseline_rmse_m for trial in rows])
        accepted = [trial for trial in rows if trial.guard_accepted]
        difference_newest = deployed - newest_rmse
        interval_newest = _paired_interval(
            difference_newest,
            resamples=protocol.base_config.bootstrap_resamples,
            seed=protocol.base_config.bootstrap_seed + 100 + method_index,
        )
        by_scenario: dict[str, Any] = {}
        for scenario in protocol.scenarios:
            selected = [trial for trial in rows if trial.scenario == scenario]
            selected_newest = [trial for trial in newest if trial.scenario == scenario]
            method_mean = float(np.mean([trial.deployed_rmse_m for trial in selected]))
            newest_mean = float(
                np.mean([trial.deployed_rmse_m for trial in selected_newest])
            )
            by_scenario[scenario] = {
                "group_count": len(selected),
                "deployed_mean_rmse_m": method_mean,
                "newest_window_mean_rmse_m": newest_mean,
                "improvement_vs_newest_fraction": 1.0 - method_mean / newest_mean,
                "acceptance_fraction": float(
                    np.mean([trial.guard_accepted for trial in selected])
                ),
                "harmful_accepted_count": sum(
                    trial.harmful_accepted for trial in selected
                ),
            }
        output[method_id] = {
            "group_count": len(rows),
            "raw_mean_rmse_m": float(np.mean([trial.raw_rmse_m for trial in rows])),
            "deployed_mean_rmse_m": float(np.mean(deployed)),
            "physical_baseline_mean_rmse_m": float(np.mean(baseline)),
            "newest_window_mean_rmse_m": float(np.mean(newest_rmse)),
            "improvement_vs_physical_fraction": float(
                1.0 - np.mean(deployed) / np.mean(baseline)
            ),
            "improvement_vs_newest_fraction": float(
                1.0 - np.mean(deployed) / np.mean(newest_rmse)
            ),
            "paired_deployed_minus_newest_95_m": list(interval_newest),
            "solver_admissible_fraction": float(
                np.mean([trial.solver_admissible for trial in rows])
            ),
            "acceptance_fraction": float(
                np.mean([trial.guard_accepted for trial in rows])
            ),
            "accepted_group_count": len(accepted),
            "harmful_accepted_count": sum(trial.harmful_accepted for trial in rows),
            "harmful_accepted_rate": (
                sum(trial.harmful_accepted for trial in rows) / len(accepted)
                if accepted
                else 0.0
            ),
            "all_rejections_exact_fallback": all(
                trial.exact_fallback for trial in rows
            ),
            "by_scenario": by_scenario,
        }
    return output


def run_development(
    protocol: DevelopmentProtocol,
    *,
    repository_revision: str,
    prob4d_revision: str,
) -> tuple[dict[str, Any], list[TrialResult]]:
    _require(
        prob4d_revision == protocol.base_config.source_revision,
        "executing Prob4D revision differs from development protocol",
    )
    association_groups = _groups(
        protocol,
        protocol.association_partition,
        prefix="association-development",
    )
    selected, selection_report, _ = select_association_configuration(
        association_groups,
        protocol,
    )

    guard_groups = _groups(
        protocol,
        protocol.pilot_guard_partition,
        prefix="pilot-guard",
    )
    guard_scores, guard_association = _score_partition(
        guard_groups,
        selected,
        protocol.base_config,
    )
    calibrations = {
        method_id: calibrate_guard(guard_scores[method_id], protocol.base_config)
        for method_id in METHODS
    }

    evaluation_groups = _groups(
        protocol,
        protocol.pilot_evaluation_partition,
        prefix="pilot-evaluation",
    )
    evaluation_scores, evaluation_association = _score_partition(
        evaluation_groups,
        selected,
        protocol.base_config,
    )
    trials = [
        apply_guard(score, calibrations[method_id])
        for method_id in METHODS
        for score in evaluation_scores[method_id]
    ]
    aggregate = _aggregate_trials(trials, protocol)
    selected_metrics = selection_report["selected"]["aggregate"]
    primary = aggregate[PRIMARY_METHOD]
    worst_scenario_regression = max(
        0.0,
        max(
            -float(value["improvement_vs_newest_fraction"])
            for value in primary["by_scenario"].values()
        ),
    )
    criteria = {
        "development_association_precision": (
            selected_metrics["precision"] >= protocol.minimum_precision
        ),
        "development_association_recall": (
            selected_metrics["recall"] >= protocol.minimum_recall
        ),
        "pilot_source_improves_newest": (
            primary["improvement_vs_newest_fraction"]
            >= protocol.minimum_pilot_improvement
        ),
        "pilot_paired_upper_bound_below_zero": (
            primary["paired_deployed_minus_newest_95_m"][1] < 0.0
        ),
        "pilot_harmful_rate_bounded": (
            primary["harmful_accepted_rate"] <= protocol.maximum_harmful_rate
        ),
        "pilot_exact_fallback": primary["all_rejections_exact_fallback"],
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol.raw["protocol_id"],
        "protocol_sha256": _canonical_sha256(protocol.raw),
        "repository_revision": repository_revision,
        "prob4d_revision": prob4d_revision,
        "partition_semantics": {
            "association_configuration": asdict(protocol.association_partition),
            "pilot_guard_calibration": asdict(protocol.pilot_guard_partition),
            "pilot_evaluation": asdict(protocol.pilot_evaluation_partition),
            "confirmatory_target_seeds_committed": False,
        },
        "association_configuration_selection": selection_report,
        "pilot_guard_calibration": {
            method_id: asdict(calibration)
            for method_id, calibration in calibrations.items()
        },
        "pilot_association": {
            "guard_partition": guard_association.metrics(),
            "evaluation_partition": evaluation_association.metrics(),
        },
        "pilot_aggregate": aggregate,
        "development_decision": {
            "criteria": criteria,
            "overall_passed": all(criteria.values()),
            "worst_scenario_regression_fraction": worst_scenario_regression,
            "next_action": (
                "freeze-disjoint-calibration-and-target-protocol"
                if all(criteria.values())
                else "retain-development-result-and-do-not-open-target-seeds"
            ),
        },
        "claim_boundary": protocol.raw["claim_boundary"],
    }
    report["report_id"] = _canonical_sha256(report)
    return report, trials


def _write_trials(path: Path, trials: Sequence[TrialResult]) -> None:
    fieldnames = list(asdict(trials[0]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            writer.writerow(asdict(trial))


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    selected = report["association_configuration_selection"]["selected"]
    aggregate = report["pilot_aggregate"]
    decision = report["development_decision"]
    lines = [
        "# Prob4D cross-window identity development study",
        "",
        f"Development decision: **{'PASS' if decision['overall_passed'] else 'FAIL'}**",
        "",
        "The confirmatory target seeds are not committed or executed in this report.",
        "",
        "## Selected source-only association rule",
        "",
        f"- Configuration: `{selected['configuration_id']}`",
        f"- Precision: {100 * selected['aggregate']['precision']:.2f}%",
        f"- Recall: {100 * selected['aggregate']['recall']:.2f}%",
        f"- F1: {100 * selected['aggregate']['f1']:.2f}%",
        "",
        "## Disjoint pilot evaluation",
        "",
        "| Method | Deployed RMSE | vs newest | Accept | Harmful accepted |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method_id in METHODS:
        row = aggregate[method_id]
        lines.append(
            "| "
            + method_id
            + f" | {1000 * row['deployed_mean_rmse_m']:.3f} mm"
            + f" | {100 * row['improvement_vs_newest_fraction']:+.2f}%"
            + f" | {100 * row['acceptance_fraction']:.1f}%"
            + f" | {row['harmful_accepted_count']} |"
        )
    lines.extend(["", "## Development criteria", ""])
    for name, passed in decision["criteria"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(["", "## Claim boundary", "", report["claim_boundary"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_checksums(output_dir: Path) -> None:
    checksum_path = output_dir / "SHA256SUMS"
    files = [
        path
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != checksum_path.name
    ]
    checksum_path.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )


def execute(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.protocol.resolve())
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    report, trials = run_development(
        protocol,
        repository_revision=str(args.repository_revision),
        prob4d_revision=str(args.prob4d_revision),
    )
    _write_json(output_dir / "report.json", report)
    _write_json(output_dir / "protocol.json", protocol.raw)
    _write_json(
        output_dir / "selected_association_configuration.json",
        report["association_configuration_selection"]["selected"],
    )
    _write_trials(output_dir / "pilot_trials.csv", trials)
    _write_markdown(output_dir / "summary.md", report)
    _write_checksums(output_dir)
    print(json.dumps(report["development_decision"], indent=2, sort_keys=True))
    return 0 if report["development_decision"]["overall_passed"] else 3


def main() -> None:
    raise SystemExit(execute(_parse_args()))


if __name__ == "__main__":
    main()
