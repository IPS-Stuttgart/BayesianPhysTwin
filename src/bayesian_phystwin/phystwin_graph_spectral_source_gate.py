"""Whole-case source gate for graph-spectral PhysTwin discrepancy dynamics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .phystwin_graph_spectral_residual import (
    GraphSpectralResidualConfig,
    GraphSpectralSeries,
    blend_with_endpoint_persistence,
    compose_dense_endpoint_with_anchor_dynamics,
    deterministic_farthest_point_sample,
    fit_graph_spectral_transition,
    inverse_distance_map,
    prepare_graph_spectral_series,
    rollout_graph_spectral_transition,
)
from .phystwin_residual_dynamics import _temporally_fill
from .phystwin_shared_nonlinear_residual import _episode_specs
from .phystwin_shared_residual_velocity import (
    SharedResidualVelocityConfig,
    _interval_metrics,
    _load_episode,
    _ratios,
)

GRAPH_SPECTRAL_SOURCE_CONTRACT = (
    "source-trained-graph-spectral-discrepancy-v1"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_with_digest(path: Path, payload: Mapping[str, object]) -> str:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = _sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="ascii",
    )
    return digest


@dataclass(frozen=True)
class GraphSpectralSourceConfig:
    """Frozen low-capacity family and source-transfer gates."""

    rank_candidates: tuple[int, ...] = (8, 16)
    temporal_smoothing_candidates: tuple[float, ...] = (0.25, 0.5)
    ridge_fraction: float = 0.01
    local_prior_strength_candidates: tuple[float, ...] = (10.0, 100.0)
    blend_candidates: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    maximum_spectral_points: int = 128
    graph_neighbor_count: int = 8
    mode_group_count: int = 4
    interpolation_neighbors: int = 4
    controller_kernel_fraction: float = 0.25
    maximum_residual_m: float = 0.01
    minimum_group_samples: int = 24
    minimum_balanced_improvement: float = 0.03
    minimum_both_win_folds: int = 2
    maximum_case_metric_ratio: float = 1.05

    def __post_init__(self) -> None:
        _require(
            bool(self.rank_candidates)
            and all(rank >= 2 for rank in self.rank_candidates),
            "rank candidates must be at least two",
        )
        _require(
            bool(self.temporal_smoothing_candidates)
            and all(
                0.0 < value <= 1.0
                for value in self.temporal_smoothing_candidates
            ),
            "temporal smoothing candidates must lie in (0, 1]",
        )
        _require(
            np.isfinite(self.ridge_fraction) and self.ridge_fraction > 0.0,
            "ridge fraction must be positive",
        )
        _require(
            bool(self.local_prior_strength_candidates)
            and all(
                np.isfinite(value) and value > 0.0
                for value in self.local_prior_strength_candidates
            ),
            "local prior strengths must be positive",
        )
        _require(
            bool(self.blend_candidates)
            and 0.0 in self.blend_candidates
            and all(0.0 <= value <= 1.0 for value in self.blend_candidates),
            "blend candidates must include exact fallback and lie in [0, 1]",
        )
        _require(
            self.maximum_spectral_points > max(self.rank_candidates),
            "spectral point count must exceed every rank",
        )
        _require(
            self.graph_neighbor_count >= 1
            and self.mode_group_count >= 2
            and self.interpolation_neighbors >= 1,
            "graph and interpolation counts must be positive",
        )
        _require(
            self.mode_group_count <= min(self.rank_candidates),
            "mode group count exceeds a candidate rank",
        )
        _require(
            self.controller_kernel_fraction > 0.0
            and self.maximum_residual_m > 0.0,
            "physical scales must be positive",
        )
        _require(
            self.minimum_group_samples >= 3,
            "minimum group samples must be at least three",
        )
        _require(
            0.0 <= self.minimum_balanced_improvement < 1.0
            and self.minimum_both_win_folds >= 1
            and self.maximum_case_metric_ratio >= 1.0,
            "source gates are invalid",
        )


@dataclass
class _PreparedCase:
    loaded: Any
    anchor_indices: np.ndarray
    interpolation_indices: np.ndarray
    interpolation_weights: np.ndarray
    source_series: GraphSpectralSeries
    prefix_series: GraphSpectralSeries
    dense_endpoint_m: np.ndarray
    persistence_metrics: dict[str, object]
    baseline_metrics: dict[str, object]


def _load_protocol(
    path: str | Path,
) -> tuple[dict[str, object], GraphSpectralSourceConfig]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        payload.get("contract") == GRAPH_SPECTRAL_SOURCE_CONTRACT,
        "source protocol uses an unsupported contract",
    )
    source_cases = payload.get("source_cases")
    target_cases = payload.get("target_cases")
    folds = payload.get("source_folds")
    _require(
        isinstance(source_cases, list)
        and len(source_cases) >= 3
        and len(set(source_cases)) == len(source_cases),
        "source cases must be a unique list of at least three cases",
    )
    _require(
        isinstance(target_cases, list)
        and not set(source_cases) & set(target_cases),
        "source and target cases must be disjoint",
    )
    _require(
        isinstance(folds, list) and len(folds) >= 2,
        "source protocol requires at least two folds",
    )
    held_out: list[str] = []
    for fold in folds:
        _require(
            isinstance(fold, Mapping)
            and isinstance(fold.get("held_out_cases"), list),
            "every fold must declare held-out cases",
        )
        held_out.extend(str(case) for case in fold["held_out_cases"])
    _require(
        sorted(held_out) == sorted(str(case) for case in source_cases),
        "source folds must hold out every source case exactly once",
    )
    model = payload.get("model")
    _require(isinstance(model, Mapping), "source protocol omits its model")
    tuple_fields = {
        "rank_candidates",
        "temporal_smoothing_candidates",
        "local_prior_strength_candidates",
        "blend_candidates",
    }
    config = GraphSpectralSourceConfig(
        **{
            key: tuple(value) if key in tuple_fields else value
            for key, value in model.items()
        }
    )
    return payload, config


def _core_config(
    source_config: GraphSpectralSourceConfig,
    *,
    rank: int,
    temporal_smoothing: float,
) -> GraphSpectralResidualConfig:
    return GraphSpectralResidualConfig(
        rank=int(rank),
        neighbor_count=source_config.graph_neighbor_count,
        mode_group_count=source_config.mode_group_count,
        temporal_smoothing=float(temporal_smoothing),
        controller_kernel_fraction=source_config.controller_kernel_fraction,
        ridge_fraction=source_config.ridge_fraction,
        minimum_group_samples=source_config.minimum_group_samples,
        maximum_residual_m=source_config.maximum_residual_m,
    )


def _prepare_case(
    loaded: Any,
    source_config: GraphSpectralSourceConfig,
    core_config: GraphSpectralResidualConfig,
) -> _PreparedCase:
    original_count = loaded.observed.shape[1]
    initial = loaded.baseline[0, :original_count]
    prefix_supported = np.any(
        loaded.valid[: loaded.spec.fit_end_frame],
        axis=0,
    )
    eligible = np.flatnonzero(
        prefix_supported & np.all(np.isfinite(initial), axis=1)
    )
    sample_count = min(source_config.maximum_spectral_points, len(eligible))
    _require(
        sample_count > core_config.rank,
        f"{loaded.spec.case}: insufficient prefix-supported spectral anchors",
    )
    anchors = deterministic_farthest_point_sample(
        initial,
        eligible,
        sample_count,
    )
    interpolation_indices, interpolation_weights = inverse_distance_map(
        initial[anchors],
        initial,
        neighbor_count=min(
            source_config.interpolation_neighbors,
            len(anchors),
        ),
    )
    source_series = prepare_graph_spectral_series(
        initial[anchors],
        loaded.residual[:, anchors],
        loaded.valid[:, anchors],
        loaded.baseline[:, anchors],
        loaded.controllers,
        end_frame=len(loaded.observed),
        config=core_config,
    )
    prefix_series = prepare_graph_spectral_series(
        initial[anchors],
        loaded.residual[:, anchors],
        loaded.valid[:, anchors],
        loaded.baseline[:, anchors],
        loaded.controllers,
        end_frame=loaded.spec.fit_end_frame,
        action_end_frame=loaded.spec.train_end_frame,
        config=core_config,
    )
    dense_endpoint = _temporally_fill(
        loaded.residual,
        loaded.valid,
        loaded.spec.fit_end_frame,
    )[-1]
    future_count = loaded.spec.train_end_frame - loaded.spec.fit_end_frame
    persistence = np.broadcast_to(
        dense_endpoint,
        (future_count, *dense_endpoint.shape),
    )
    metric_config = SharedResidualVelocityConfig(
        interpolation_neighbors=source_config.interpolation_neighbors,
        controller_kernel_fraction=source_config.controller_kernel_fraction,
        maximum_residual_m=source_config.maximum_residual_m,
    )
    persistence_metrics, _ = _interval_metrics(
        loaded,
        persistence,
        start_frame=loaded.spec.fit_end_frame,
        end_frame=loaded.spec.train_end_frame,
        config=metric_config,
    )
    baseline_metrics, _ = _interval_metrics(
        loaded,
        None,
        start_frame=loaded.spec.fit_end_frame,
        end_frame=loaded.spec.train_end_frame,
        config=metric_config,
    )
    return _PreparedCase(
        loaded=loaded,
        anchor_indices=anchors,
        interpolation_indices=interpolation_indices,
        interpolation_weights=interpolation_weights,
        source_series=source_series,
        prefix_series=prefix_series,
        dense_endpoint_m=dense_endpoint,
        persistence_metrics=persistence_metrics,
        baseline_metrics=baseline_metrics,
    )


def _candidate_summary(
    case_results: Sequence[Mapping[str, object]],
    folds: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    aggregate_ratios = {
        metric: float(
            np.mean(
                [
                    float(result["ratios_relative_to_persistence"][metric])
                    for result in case_results
                ]
            )
        )
        for metric in ("chamfer_distance_m", "track_error_m")
    }
    by_case = {str(result["case"]): result for result in case_results}
    both_win_folds = 0
    for fold in folds:
        if all(
            max(
                float(value)
                for value in by_case[str(case)][
                    "ratios_relative_to_persistence"
                ].values()
            )
            < 1.0
            for case in fold["held_out_cases"]
        ):
            both_win_folds += 1
    return {
        "aggregate_ratios_relative_to_persistence": aggregate_ratios,
        "balanced_improvement": 1.0
        - 0.5 * sum(aggregate_ratios.values()),
        "both_win_fold_count": both_win_folds,
        "maximum_case_metric_ratio": max(
            max(
                float(value)
                for value in result[
                    "ratios_relative_to_persistence"
                ].values()
            )
            for result in case_results
        ),
    }


def _candidate_passes(
    candidate: Mapping[str, object],
    config: GraphSpectralSourceConfig,
) -> bool:
    ratios = candidate["aggregate_ratios_relative_to_persistence"]
    return bool(
        float(candidate["balanced_improvement"])
        >= config.minimum_balanced_improvement
        and int(candidate["both_win_fold_count"])
        >= config.minimum_both_win_folds
        and max(float(value) for value in ratios.values()) < 1.0
        and float(candidate["maximum_case_metric_ratio"])
        <= config.maximum_case_metric_ratio
    )


def run_graph_spectral_source_gate(
    data_root: str | Path,
    protocol_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Cross-fit the source family without reading any target-case artifact."""

    protocol, source_config = _load_protocol(protocol_path)
    fit_fraction = float(protocol.get("fit_fraction", 0.75))
    _require(
        0.5 <= fit_fraction < 1.0,
        "fit fraction must lie in [0.5, 1)",
    )
    root = Path(data_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_cases = tuple(str(case) for case in protocol["source_cases"])
    specs = _episode_specs(root, source_cases, fit_fraction)
    metric_config = SharedResidualVelocityConfig(
        interpolation_neighbors=source_config.interpolation_neighbors,
        controller_kernel_fraction=source_config.controller_kernel_fraction,
        maximum_residual_m=source_config.maximum_residual_m,
    )
    loaded = {
        spec.case: _load_episode(spec, metric_config)
        for spec in specs
    }

    candidates: list[dict[str, object]] = []
    preparation_records: list[dict[str, object]] = []
    for rank in source_config.rank_candidates:
        for smoothing in source_config.temporal_smoothing_candidates:
            core_config = _core_config(
                source_config,
                rank=rank,
                temporal_smoothing=smoothing,
            )
            prepared = {
                case: _prepare_case(value, source_config, core_config)
                for case, value in loaded.items()
            }
            preparation_records.append(
                {
                    "rank": int(rank),
                    "temporal_smoothing": float(smoothing),
                    "anchor_counts": {
                        case: len(item.anchor_indices)
                        for case, item in prepared.items()
                    },
                }
            )
            for prior_strength in source_config.local_prior_strength_candidates:
                dynamic_by_case: dict[str, np.ndarray] = {}
                transition_records = []
                for fold_index, fold in enumerate(protocol["source_folds"]):
                    held_out = tuple(
                        str(case) for case in fold["held_out_cases"]
                    )
                    training_cases = tuple(
                        case for case in source_cases if case not in held_out
                    )
                    source_prior = fit_graph_spectral_transition(
                        [
                            prepared[case].source_series
                            for case in training_cases
                        ],
                        config=core_config,
                    )
                    local_records = {}
                    for case in held_out:
                        item = prepared[case]
                        local = fit_graph_spectral_transition(
                            [item.prefix_series],
                            config=core_config,
                            prior=source_prior,
                            prior_strength=float(prior_strength),
                        )
                        anchor_dynamic = rollout_graph_spectral_transition(
                            item.prefix_series,
                            local,
                            start_frame=item.loaded.spec.fit_end_frame,
                            end_frame=item.loaded.spec.train_end_frame,
                            config=core_config,
                        )
                        dynamic_by_case[case] = (
                            compose_dense_endpoint_with_anchor_dynamics(
                                anchor_dynamic,
                                item.dense_endpoint_m,
                                item.anchor_indices,
                                item.interpolation_indices,
                                item.interpolation_weights,
                                maximum_residual_m=(
                                    source_config.maximum_residual_m
                                ),
                            )
                        )
                        local_records[case] = {
                            "velocity_retention": (
                                local.velocity_retention.tolist()
                            ),
                            "action_current": local.action_current.tolist(),
                            "action_change": local.action_change.tolist(),
                            "sample_count": local.sample_count.tolist(),
                        }
                    transition_records.append(
                        {
                            "fold_index": fold_index,
                            "training_cases": list(training_cases),
                            "held_out_cases": list(held_out),
                            "source_prior": {
                                "velocity_retention": (
                                    source_prior.velocity_retention.tolist()
                                ),
                                "action_current": (
                                    source_prior.action_current.tolist()
                                ),
                                "action_change": (
                                    source_prior.action_change.tolist()
                                ),
                                "sample_count": (
                                    source_prior.sample_count.tolist()
                                ),
                            },
                            "local_posteriors": local_records,
                        }
                    )
                for blend in source_config.blend_candidates:
                    case_results = []
                    for case in source_cases:
                        item = prepared[case]
                        count = (
                            item.loaded.spec.train_end_frame
                            - item.loaded.spec.fit_end_frame
                        )
                        persistence = np.broadcast_to(
                            item.dense_endpoint_m,
                            (count, *item.dense_endpoint_m.shape),
                        )
                        tracked = blend_with_endpoint_persistence(
                            dynamic_by_case[case],
                            persistence,
                            float(blend),
                        )
                        dynamic_metrics, _ = _interval_metrics(
                            item.loaded,
                            tracked,
                            start_frame=item.loaded.spec.fit_end_frame,
                            end_frame=item.loaded.spec.train_end_frame,
                            config=metric_config,
                        )
                        case_results.append(
                            {
                                "case": case,
                                "ratios_relative_to_persistence": _ratios(
                                    dynamic_metrics,
                                    item.persistence_metrics,
                                ),
                                "baseline_official_evaluation": (
                                    item.baseline_metrics
                                ),
                                "persistence_official_evaluation": (
                                    item.persistence_metrics
                                ),
                                "dynamic_official_evaluation": dynamic_metrics,
                            }
                        )
                    aggregate = _candidate_summary(
                        case_results,
                        protocol["source_folds"],
                    )
                    candidates.append(
                        {
                            "rank": int(rank),
                            "temporal_smoothing": float(smoothing),
                            "local_prior_strength": float(prior_strength),
                            "blend": float(blend),
                            **aggregate,
                            "case_results": case_results,
                            "transition_records": transition_records,
                        }
                    )

    selected = min(
        candidates,
        key=lambda item: (
            -float(item["balanced_improvement"]),
            int(item["rank"]),
            float(item["blend"]),
            float(item["local_prior_strength"]),
            float(item["temporal_smoothing"]),
        ),
    )
    gate_passed = _candidate_passes(selected, source_config)
    summary: dict[str, object] = {
        "schema_version": 1,
        "contract": GRAPH_SPECTRAL_SOURCE_CONTRACT,
        "source_gate_passed": gate_passed,
        "target_future_opened": False,
        "protocol": {
            "path": str(Path(protocol_path).resolve()),
            "sha256": _sha256(protocol_path),
        },
        "data_root": str(root),
        "config": asdict(source_config),
        "source_inputs": [
            {
                "case": spec.case,
                "fit_end_frame": spec.fit_end_frame,
                "train_end_frame": spec.train_end_frame,
                "final_data_sha256": _sha256(spec.final_data),
                "baseline_trajectory_sha256": _sha256(
                    spec.baseline_trajectory
                ),
                "gt_track_3d_sha256": _sha256(spec.gt_track_3d),
            }
            for spec in specs
        ],
        "preparations": preparation_records,
        "selection": {
            "selected_candidate": selected,
            "candidates": candidates,
        },
        "claim_boundary": (
            "Complete outcomes supervise registered source interactions only. "
            "Each scored case is excluded from its source-prior fit; its local "
            "adaptation uses only the allowed prefix. No target artifact is read."
        ),
    }
    summary_path = output / "source_gate_summary.json"
    digest = _write_json_with_digest(summary_path, summary)
    return {
        **summary,
        "summary_artifact": {
            "path": str(summary_path),
            "sha256": digest,
        },
    }


__all__ = [
    "GRAPH_SPECTRAL_SOURCE_CONTRACT",
    "GraphSpectralSourceConfig",
    "run_graph_spectral_source_gate",
]
