"""Source-calibrated model-form trust for reusable Deform360 PhysTwin rollouts."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree


PHYSTWIN_TRUST_SCHEMA_VERSION = 1
CARDINALITY_TRUST_PROTOCOL_ID = "deform360-cardinality-trust-002-rope-silk-v1"
CANONICAL_CARDINALITY_TRUST_CONFIG_SHA256 = (
    "ab1177d24f87281c9dffc80d68666844efc939e96852eabf00526acbceed588d"
)
CARDINALITY_SOURCE_EXECUTION_PROTOCOL_ID = (
    "deform360-cardinality-source-execution-002-rope-silk-v1"
)
CANONICAL_CARDINALITY_SOURCE_EXECUTION_CONFIG_SHA256 = (
    "5175235d0409368e6e69ab708eb958255cc85eaa82c5260098f3363238bfc8b7"
)
CONTACT_ANCHORED_CAUSAL_TRUST_PROTOCOL_ID = (
    "deform360-contact-anchored-causal-trust-002-rope-silk-v1"
)
CANONICAL_CONTACT_ANCHORED_CAUSAL_TRUST_CONFIG_SHA256 = (
    "1ce4dabdc54683c73ae3b93a0dcec2e1d87542fb245fbfe2338abdfe9dc3341e"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _result_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _config_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, order=True)
class CausalTrustWeights:
    """Trust assigned to intervention response and autonomous simulator drift."""

    action_response: float
    autonomous_drift: float

    def __post_init__(self) -> None:
        for value in (self.action_response, self.autonomous_drift):
            _require(np.isfinite(value), "causal trust weight is non-finite")
            _require(0.0 <= value <= 1.0, "causal trust weight is outside [0, 1]")

    def as_dict(self) -> dict[str, float]:
        return {
            "action_response": float(self.action_response),
            "autonomous_drift": float(self.autonomous_drift),
        }


@dataclass(frozen=True, order=True)
class PhysicalTrustParameters:
    """One preregistered PhysTwin parameter tuple in the source grid."""

    init_spring_y: float
    drag_damping: float
    dashpot_damping: float

    def __post_init__(self) -> None:
        values = (self.init_spring_y, self.drag_damping, self.dashpot_damping)
        _require(
            all(np.isfinite(value) for value in values), "physical grid is non-finite"
        )
        _require(self.init_spring_y > 0.0, "spring stiffness must be positive")
        _require(
            self.drag_damping >= 0.0 and self.dashpot_damping >= 0.0,
            "physical damping must be non-negative",
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "init_spring_y": float(self.init_spring_y),
            "drag_damping": float(self.drag_damping),
            "dashpot_damping": float(self.dashpot_damping),
        }


@dataclass(frozen=True)
class CausalTrustEpisode:
    """One source execution with driven and matched zero-action Warp rollouts."""

    episode_id: str
    target_m: np.ndarray
    visibility: np.ndarray
    validity: np.ndarray
    driven_m: np.ndarray
    zero_action_m: np.ndarray
    train_stop_frame: int
    source_data_sha256: str
    driven_trajectory_sha256: str
    zero_action_trajectory_sha256: str
    controller_count: int = 1

    def __post_init__(self) -> None:
        target = np.asarray(self.target_m, dtype=np.float64)
        driven = np.asarray(self.driven_m, dtype=np.float64)
        zero = np.asarray(self.zero_action_m, dtype=np.float64)
        visibility = np.asarray(self.visibility, dtype=bool)
        validity = np.asarray(self.validity, dtype=bool)
        _require(bool(self.episode_id), "trust episode needs an identity")
        _require(self.controller_count >= 1, "controller count must be positive")
        _require(
            target.ndim == 3 and target.shape[-1] == 3 and target.shape == driven.shape,
            "target and driven trajectories differ",
        )
        _require(zero.shape == target.shape, "zero-action trajectory differs")
        _require(
            visibility.shape == target.shape[:2] and validity.shape == target.shape[:2],
            "trust episode masks differ from trajectories",
        )
        _require(
            np.all(np.isfinite(target))
            and np.all(np.isfinite(driven))
            and np.all(np.isfinite(zero)),
            "trust episode trajectory is non-finite",
        )
        _require(
            2 <= self.train_stop_frame < len(target),
            "trust episode has no train/tail split",
        )
        _require(
            np.allclose(driven[0], zero[0], atol=1e-7, rtol=0.0),
            "driven and zero-action initial states differ",
        )
        _require(
            np.allclose(target[0], zero[0], atol=1e-5, rtol=0.0),
            "observed and simulated initial states are not registered",
        )
        hashes = (
            self.source_data_sha256,
            self.driven_trajectory_sha256,
            self.zero_action_trajectory_sha256,
        )
        _require(
            all(_valid_sha256(value) for value in hashes), "source hash is invalid"
        )
        for name, value in (
            ("target_m", target),
            ("visibility", visibility),
            ("validity", validity),
            ("driven_m", driven),
            ("zero_action_m", zero),
        ):
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)


def load_official_phystwin_trust_episode(
    episode_id: str,
    data_path: str | Path,
    driven_result_path: str | Path,
    zero_action_result_path: str | Path,
    split_path: str | Path,
) -> CausalTrustEpisode:
    """Load a matched official-Warp source pair with checksum validation."""

    data_file = Path(data_path)
    driven_result_file = Path(driven_result_path)
    zero_result_file = Path(zero_action_result_path)
    split_file = Path(split_path)
    driven_result = json.loads(driven_result_file.read_text(encoding="utf-8"))
    zero_result = json.loads(zero_result_file.read_text(encoding="utf-8"))
    for label, result in (("driven", driven_result), ("zero-action", zero_result)):
        _require(result.get("passed") is True, f"{label} Warp rollout failed")
        _require(
            result.get("source_only_smoke") is True,
            f"{label} Warp rollout is not source-only",
        )
        _require(
            result.get("data_sha256") == _sha256_file(data_file),
            f"{label} source data checksum changed",
        )
        _require(
            result.get("split_sha256") == _sha256_file(split_file),
            f"{label} split checksum changed",
        )
    for key in (
        "official_phystwin_revision",
        "data_sha256",
        "config_sha256",
        "split_sha256",
        "config_overrides",
        "support_dynamics",
        "effective_inertia",
        "contact_transmission",
        "frame_count",
        "num_controller_points",
        "num_original_points",
    ):
        _require(
            driven_result.get(key) == zero_result.get(key),
            f"matched Warp pair differs in {key}",
        )
    _require(
        np.isclose(
            float(
                driven_result.get("realized_actuation", {}).get(
                    "controller_displacement_scale", np.nan
                )
            ),
            1.0,
        ),
        "driven rollout does not use the full source action",
    )
    _require(
        np.isclose(
            float(
                zero_result.get("realized_actuation", {}).get(
                    "controller_displacement_scale", np.nan
                )
            ),
            0.0,
        ),
        "zero-action rollout moves the controller",
    )
    driven_trajectory_path = driven_result_file.with_name(
        "official_phystwin_trajectory.npz"
    )
    zero_trajectory_path = zero_result_file.with_name(
        "official_phystwin_trajectory.npz"
    )
    _require(
        driven_result.get("trajectory_sha256") == _sha256_file(driven_trajectory_path),
        "driven trajectory checksum changed",
    )
    _require(
        zero_result.get("trajectory_sha256") == _sha256_file(zero_trajectory_path),
        "zero-action trajectory checksum changed",
    )
    with data_file.open("rb") as stream:
        data = pickle.load(stream)
    _require(isinstance(data, Mapping), "PhysTwin source bundle is not a mapping")
    target = np.asarray(data["object_points"])
    visibility = np.asarray(data["object_visibilities"])
    validity = np.asarray(data["object_motions_valid"])
    driven_vertices = np.load(driven_trajectory_path)["vertices"]
    zero_vertices = np.load(zero_trajectory_path)["vertices"]
    point_count = target.shape[1]
    driven = driven_vertices[:, :point_count]
    zero = zero_vertices[:, :point_count]
    split = json.loads(split_file.read_text(encoding="utf-8"))
    _require(
        int(split.get("frame_len", -1)) == len(target),
        "trust split frame count differs",
    )
    train = split.get("train")
    tail = split.get("test")
    _require(
        isinstance(train, list)
        and train[0] == 0
        and isinstance(tail, list)
        and tail[0] == train[1]
        and tail[1] == len(target),
        "trust split is not a contiguous source train/tail partition",
    )
    return CausalTrustEpisode(
        episode_id=episode_id,
        target_m=target,
        visibility=visibility,
        validity=validity,
        driven_m=driven,
        zero_action_m=zero,
        train_stop_frame=int(train[1]),
        source_data_sha256=_sha256_file(data_file),
        driven_trajectory_sha256=_sha256_file(driven_trajectory_path),
        zero_action_trajectory_sha256=_sha256_file(zero_trajectory_path),
        controller_count=int(driven_result["num_controller_points"]),
    )


def load_cardinality_trust_protocol(path: str | Path) -> dict[str, Any]:
    """Load the immutable independent-object hypothesis lock."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        payload.get("schema_version") == PHYSTWIN_TRUST_SCHEMA_VERSION,
        "cardinality trust protocol schema changed",
    )
    observed = _config_sha256(payload)
    _require(
        payload.get("config_sha256") == observed,
        "cardinality trust protocol checksum mismatch",
    )
    _require(
        observed == CANONICAL_CARDINALITY_TRUST_CONFIG_SHA256,
        "cardinality trust protocol differs from canonical lock",
    )
    config = payload.get("config", {})
    _require(
        config.get("protocol_id") == CARDINALITY_TRUST_PROTOCOL_ID,
        "cardinality trust protocol id changed",
    )
    _require(
        config.get("validation_object_id") == "002-rope-silk"
        and config.get("source_episode_ids") == [0, 2, 5, 6, 7, 9]
        and config.get("calibration_episode_ids") == [3, 4, 8]
        and config.get("sealed_target_episode_id") == 1,
        "cardinality trust evidence split changed",
    )
    boundary = config.get("information_boundary", {})
    _require(
        boundary.get("source_tails_select_parameters") is False
        and boundary.get("calibration_episodes_read_before_source_gate") is False
        and boundary.get("target_episode_read_before_source_and_calibration_gates")
        is False
        and boundary.get("future_object_observations_used_at_prediction_time") is False,
        "cardinality trust information boundary changed",
    )
    return payload


def load_cardinality_source_execution_protocol(
    path: str | Path,
) -> dict[str, Any]:
    """Load the immutable source execution addendum for the independent test."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        payload.get("schema_version") == PHYSTWIN_TRUST_SCHEMA_VERSION,
        "cardinality source execution schema changed",
    )
    observed = _config_sha256(payload)
    _require(
        payload.get("config_sha256") == observed,
        "cardinality source execution checksum mismatch",
    )
    _require(
        observed == CANONICAL_CARDINALITY_SOURCE_EXECUTION_CONFIG_SHA256,
        "cardinality source execution differs from canonical lock",
    )
    config = payload.get("config", {})
    _require(
        config.get("protocol_id") == CARDINALITY_SOURCE_EXECUTION_PROTOCOL_ID,
        "cardinality source execution protocol id changed",
    )
    _require(
        config.get("parent_config_sha256") == CANONICAL_CARDINALITY_TRUST_CONFIG_SHA256,
        "cardinality source execution parent changed",
    )
    frame_slice = config.get("frame_slice", {})
    _require(
        frame_slice.get("frame_count") == 81
        and frame_slice.get("train_frame_range") == [0, 64]
        and frame_slice.get("untouched_tail_frame_range") == [64, 81],
        "cardinality source train/tail boundary changed",
    )
    roles = config.get("physical_arm_roles", {})
    _require(
        roles.get("primary_gate_arm") == "source_pooled_grid"
        and roles.get("transfer_control_arm") == "inherited_081_control"
        and roles.get("transfer_control_can_open_calibration_or_target") is False,
        "cardinality source physical arm roles changed",
    )
    boundary = config.get("information_boundary", {})
    _require(
        boundary.get("source_object_trajectories_read_before_this_lock") is False
        and boundary.get("source_tail_outcomes_read_before_this_lock") is False
        and boundary.get("calibration_episode_read") is False
        and boundary.get("sealed_target_episode_read") is False,
        "cardinality source information boundary changed",
    )
    return payload


def load_contact_anchored_causal_trust_protocol(
    path: str | Path,
) -> dict[str, Any]:
    """Load the post-hoc source discovery frozen before calibration and target."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        payload.get("schema_version") == PHYSTWIN_TRUST_SCHEMA_VERSION,
        "contact-anchored trust schema changed",
    )
    observed = _config_sha256(payload)
    _require(
        payload.get("config_sha256") == observed,
        "contact-anchored trust checksum mismatch",
    )
    _require(
        observed == CANONICAL_CONTACT_ANCHORED_CAUSAL_TRUST_CONFIG_SHA256,
        "contact-anchored trust differs from canonical lock",
    )
    config = payload.get("config", {})
    _require(
        config.get("protocol_id") == CONTACT_ANCHORED_CAUSAL_TRUST_PROTOCOL_ID,
        "contact-anchored trust protocol id changed",
    )
    _require(
        config.get("status")
        == "post-hoc-source-method-freeze-before-calibration-and-target",
        "contact-anchored trust status changed",
    )
    _require(
        config.get("source_episode_ids") == [0, 2, 5, 6, 7, 9]
        and config.get("calibration_episode_ids") == [3, 4, 8]
        and config.get("sealed_target_episode_id") == 1,
        "contact-anchored trust evidence split changed",
    )
    association = config.get("material_association", {})
    _require(
        association.get("observation_frame_count") == 6
        and association.get("node_count") == 21
        and association.get("state_innovation_used_for_prior_reliability") is False,
        "contact-anchored association changed",
    )
    candidate = config.get("physical_candidate", {})
    trust = config.get("causal_trust", {})
    _require(
        candidate.get("candidate_index") == 157
        and trust.get("prehensile_action_response_weight") == "1 / controller_count"
        and trust.get("autonomous_drift_weight") == 0.0
        and trust.get("support_tangential_policy") == "exact-persistence-fallback",
        "contact-anchored causal policy changed",
    )
    boundary = config.get("information_boundary", {})
    _require(
        boundary.get("all_source_outcomes_read_before_this_freeze") is True
        and boundary.get("calibration_episode_outcomes_read_before_this_freeze")
        is False
        and boundary.get("target_episode_outcome_read_before_this_freeze") is False
        and boundary.get("future_object_observations_used_at_prediction_time") is False,
        "contact-anchored trust information boundary changed",
    )
    return payload


def causal_control_variate_prediction(
    target_initial_m: np.ndarray,
    driven_m: np.ndarray,
    zero_action_m: np.ndarray,
    weights: CausalTrustWeights,
) -> np.ndarray:
    """Combine intervention response and autonomous drift with separate trust.

    The decomposition is

    ``driven = initial + (driven - zero_action) + (zero_action - initial)``.

    The matched zero-action rollout acts as a causal control variate. It removes
    free-settling drift that is unrelated to the commanded intervention without
    reading future observations.
    """

    initial = np.asarray(target_initial_m, dtype=np.float64)
    driven = np.asarray(driven_m, dtype=np.float64)
    zero = np.asarray(zero_action_m, dtype=np.float64)
    _require(driven.shape == zero.shape, "driven and zero-action rollouts differ")
    _require(
        driven.ndim == 3 and driven.shape[-1] == 3,
        "control-variate rollout must have shape (T,N,3)",
    )
    _require(
        initial.shape in ((1, *driven.shape[1:]), driven.shape[1:]),
        "initial state differs",
    )
    initial = initial.reshape((1, *driven.shape[1:]))
    _require(
        np.all(np.isfinite(initial))
        and np.all(np.isfinite(driven))
        and np.all(np.isfinite(zero)),
        "control-variate input is non-finite",
    )
    action_response = driven - zero
    autonomous_drift = zero - zero[:1]
    return (
        initial
        + weights.action_response * action_response
        + weights.autonomous_drift * autonomous_drift
    )


def _symmetric_chamfer_m(predicted: np.ndarray, target: np.ndarray) -> float:
    # Candidate grids call this thousands of times; per-query thread pools are
    # slower than serial KD-tree queries at Deform360's point counts.
    pred_to_target = cKDTree(target).query(predicted)[0]
    target_to_pred = cKDTree(predicted).query(target)[0]
    return float(0.5 * (pred_to_target.mean() + target_to_pred.mean()))


def _score_interval(
    episode: CausalTrustEpisode,
    predicted_m: np.ndarray,
    start: int,
    stop: int,
    *,
    persistence_metrics: Mapping[str, float] | None = None,
) -> dict[str, float | int]:
    _require(0 <= start < stop <= len(episode.target_m), "score interval is invalid")
    track = []
    chamfer = []
    persistence_track = [] if persistence_metrics is None else None
    persistence_chamfer = [] if persistence_metrics is None else None
    for frame in range(max(1, start), stop):
        mask = episode.visibility[frame] & episode.validity[frame]
        if not np.any(mask):
            mask = np.ones(episode.target_m.shape[1], dtype=bool)
        predicted = predicted_m[frame, mask]
        target = episode.target_m[frame, mask]
        track.append(float(np.sqrt(np.mean((predicted - target) ** 2))))
        chamfer.append(_symmetric_chamfer_m(predicted, target))
        if persistence_metrics is None:
            static = episode.target_m[0, mask]
            persistence_track.append(float(np.sqrt(np.mean((static - target) ** 2))))
            persistence_chamfer.append(_symmetric_chamfer_m(static, target))
    _require(bool(track), "score interval has no future frame")
    if persistence_metrics is None:
        persistence_track_rmse_m = float(np.mean(persistence_track))
        persistence_chamfer_m = float(np.mean(persistence_chamfer))
    else:
        persistence_track_rmse_m = float(
            persistence_metrics["persistence_track_rmse_m"]
        )
        persistence_chamfer_m = float(persistence_metrics["persistence_chamfer_m"])
    return {
        "frame_count": len(track),
        "track_rmse_m": float(np.mean(track)),
        "chamfer_m": float(np.mean(chamfer)),
        "persistence_track_rmse_m": persistence_track_rmse_m,
        "persistence_chamfer_m": persistence_chamfer_m,
    }


def _relative_score(metrics: Mapping[str, float | int]) -> float:
    persistence_track = float(metrics["persistence_track_rmse_m"])
    persistence_chamfer = float(metrics["persistence_chamfer_m"])
    _require(
        persistence_track > 0.0 and persistence_chamfer > 0.0,
        "persistence denominator is zero",
    )
    return float(
        0.5
        * (
            float(metrics["track_rmse_m"]) / persistence_track
            + float(metrics["chamfer_m"]) / persistence_chamfer
        )
    )


def _candidate_metrics(
    episode: CausalTrustEpisode,
    weights: CausalTrustWeights,
    *,
    interval: str,
    persistence_metrics: Mapping[str, float] | None = None,
) -> dict[str, float | int]:
    predicted = causal_control_variate_prediction(
        episode.target_m[:1], episode.driven_m, episode.zero_action_m, weights
    )
    if interval == "train":
        start, stop = 0, episode.train_stop_frame
    elif interval == "tail":
        start, stop = episode.train_stop_frame, len(episode.target_m)
    else:
        raise ValueError(f"unknown trust interval: {interval!r}")
    metrics = _score_interval(
        episode,
        predicted,
        start,
        stop,
        persistence_metrics=persistence_metrics,
    )
    metrics["relative_score_vs_persistence"] = _relative_score(metrics)
    return metrics


def _candidate_grid(
    action_response_grid: Sequence[float],
    autonomous_drift_grid: Sequence[float],
) -> tuple[CausalTrustWeights, ...]:
    candidates = tuple(
        CausalTrustWeights(float(action), float(drift))
        for action in action_response_grid
        for drift in autonomous_drift_grid
    )
    _require(bool(candidates), "causal trust grid is empty")
    _require(len(set(candidates)) == len(candidates), "causal trust grid is repeated")
    _require(
        CausalTrustWeights(0.0, 0.0) in candidates
        and CausalTrustWeights(1.0, 1.0) in candidates,
        "causal trust grid must contain persistence and raw PhysTwin controls",
    )
    return candidates


def _evaluate_candidate_table(
    episodes: Sequence[CausalTrustEpisode],
    candidates: Sequence[CausalTrustWeights],
) -> list[dict[str, Any]]:
    persistence_by_episode = {
        episode.episode_id: _candidate_metrics(
            episode,
            CausalTrustWeights(0.0, 0.0),
            interval="train",
        )
        for episode in episodes
    }
    table = []
    for weights in candidates:
        by_episode = {
            episode.episode_id: _candidate_metrics(
                episode,
                weights,
                interval="train",
                persistence_metrics=persistence_by_episode[episode.episode_id],
            )
            for episode in episodes
        }
        aggregate = float(
            np.mean(
                [
                    float(metrics["relative_score_vs_persistence"])
                    for metrics in by_episode.values()
                ]
            )
        )
        table.append(
            {
                "weights": weights.as_dict(),
                "pooled_train_relative_score_vs_persistence": aggregate,
                "train_by_episode": by_episode,
            }
        )
    return table


def _select_candidate(
    episode_ids: Sequence[str],
    candidates: Sequence[CausalTrustWeights],
    table: Sequence[Mapping[str, Any]],
) -> CausalTrustWeights:
    selected_episode_ids = tuple(episode_ids)
    _require(bool(selected_episode_ids), "causal trust selection has no source")
    _require(len(table) == len(candidates), "causal trust candidate table differs")

    def subset_score(index: int) -> float:
        by_episode = table[index]["train_by_episode"]
        return float(
            np.mean(
                [
                    float(by_episode[episode_id]["relative_score_vs_persistence"])
                    for episode_id in selected_episode_ids
                ]
            )
        )

    selected_index = min(
        range(len(candidates)),
        key=lambda index: (
            subset_score(index),
            candidates[index].action_response + candidates[index].autonomous_drift,
            candidates[index].action_response,
            candidates[index].autonomous_drift,
        ),
    )
    return candidates[selected_index]


def fit_source_causal_trust(
    episodes: Sequence[CausalTrustEpisode],
    *,
    action_response_grid: Sequence[float] = tuple(np.linspace(0.0, 1.0, 11)),
    autonomous_drift_grid: Sequence[float] = tuple(np.linspace(0.0, 1.0, 11)),
) -> dict[str, Any]:
    """Select trust on source train frames and evaluate untouched source tails."""

    source = tuple(episodes)
    _require(len(source) >= 2, "causal trust fit needs at least two source actions")
    _require(
        len({episode.episode_id for episode in source}) == len(source),
        "causal trust source episode is repeated",
    )
    candidates = _candidate_grid(action_response_grid, autonomous_drift_grid)
    candidate_table = _evaluate_candidate_table(source, candidates)
    selected = _select_candidate(
        [episode.episode_id for episode in source], candidates, candidate_table
    )
    selected_train = {
        episode.episode_id: _candidate_metrics(episode, selected, interval="train")
        for episode in source
    }
    selected_tail = {
        episode.episode_id: _candidate_metrics(episode, selected, interval="tail")
        for episode in source
    }
    leave_one_action_out = []
    if len(source) >= 3:
        for held_out in source:
            fitting_ids = [
                episode.episode_id
                for episode in source
                if episode.episode_id != held_out.episode_id
            ]
            fold_weights = _select_candidate(fitting_ids, candidates, candidate_table)
            held_metrics = _candidate_metrics(held_out, fold_weights, interval="tail")
            leave_one_action_out.append(
                {
                    "held_out_episode_id": held_out.episode_id,
                    "selected_weights": fold_weights.as_dict(),
                    "tail_metrics": held_metrics,
                    "beats_persistence_track": float(held_metrics["track_rmse_m"])
                    < float(held_metrics["persistence_track_rmse_m"]),
                    "beats_persistence_chamfer": float(held_metrics["chamfer_m"])
                    < float(held_metrics["persistence_chamfer_m"]),
                }
            )
    pooled_tail = {
        name: float(
            np.mean([float(metrics[name]) for metrics in selected_tail.values()])
        )
        for name in (
            "track_rmse_m",
            "chamfer_m",
            "persistence_track_rmse_m",
            "persistence_chamfer_m",
        )
    }
    pooled_tail["track_improvement_fraction_vs_persistence"] = float(
        (pooled_tail["persistence_track_rmse_m"] - pooled_tail["track_rmse_m"])
        / pooled_tail["persistence_track_rmse_m"]
    )
    pooled_tail["chamfer_improvement_fraction_vs_persistence"] = float(
        (pooled_tail["persistence_chamfer_m"] - pooled_tail["chamfer_m"])
        / pooled_tail["persistence_chamfer_m"]
    )
    payload: dict[str, Any] = {
        "schema_version": PHYSTWIN_TRUST_SCHEMA_VERSION,
        "artifact_kind": "Deform360PhysTwinCausalTrustFit",
        "source_episode_ids": [episode.episode_id for episode in source],
        "source_inputs": {
            episode.episode_id: {
                "data_sha256": episode.source_data_sha256,
                "driven_trajectory_sha256": episode.driven_trajectory_sha256,
                "zero_action_trajectory_sha256": episode.zero_action_trajectory_sha256,
                "train_frame_range": [1, episode.train_stop_frame],
                "untouched_tail_frame_range": [
                    episode.train_stop_frame,
                    len(episode.target_m),
                ],
            }
            for episode in source
        },
        "method": {
            "decomposition": (
                "target_initial + action_response_trust * (driven - zero_action) "
                "+ autonomous_drift_trust * (zero_action - zero_action_initial)"
            ),
            "selection_score": (
                "execution-balanced mean of normalized track RMSE and symmetric "
                "Chamfer on source train frames"
            ),
            "candidate_count": len(candidates),
            "action_response_grid": sorted(
                {weights.action_response for weights in candidates}
            ),
            "autonomous_drift_grid": sorted(
                {weights.autonomous_drift for weights in candidates}
            ),
        },
        "selected_weights": selected.as_dict(),
        "selected_train_by_episode": selected_train,
        "selected_tail_by_episode": selected_tail,
        "pooled_source_tail": pooled_tail,
        "leave_one_action_out": leave_one_action_out,
        "candidate_table": candidate_table,
        "information_boundary": {
            "selection_uses_source_train_frames_only": True,
            "source_tails_used_for_selection": False,
            "source_tails_used_for_exploratory_transfer_evaluation": True,
            "calibration_episode_read": False,
            "target_episode_read": False,
            "future_observation_required_at_prediction_time": False,
        },
        "claim_boundary": (
            "source-only model-form trust diagnostic; not a Deform360 SOTA claim "
            "until the locked multi-action and sealed evaluation gates pass"
        ),
    }
    payload["result_sha256"] = _result_sha256(payload)
    return payload


def fit_cardinality_normalized_source_causal_trust(
    episodes: Sequence[CausalTrustEpisode],
    *,
    action_response_grid: Sequence[float] = tuple(np.linspace(0.0, 1.0, 11)),
    autonomous_drift_grid: Sequence[float] = tuple(np.linspace(0.0, 1.0, 11)),
) -> dict[str, Any]:
    """Fit trust after normalizing aggregate action response by controller count.

    The registered driven-minus-zero trajectory is an aggregate response to all
    active controllers. Dividing it by controller cardinality prevents a
    bimanual action from receiving twice the trusted response solely because it
    contains two virtual attachments. This is a source hypothesis, not a claim
    that deformable responses are generally additive.
    """

    source = tuple(episodes)
    _require(
        len(source) >= 3,
        "cardinality-normalized trust needs at least three source actions",
    )
    normalized = tuple(
        replace(
            episode,
            driven_m=episode.zero_action_m
            + (episode.driven_m - episode.zero_action_m)
            / float(episode.controller_count),
        )
        for episode in source
    )
    payload = fit_source_causal_trust(
        normalized,
        action_response_grid=action_response_grid,
        autonomous_drift_grid=autonomous_drift_grid,
    )
    payload.pop("result_sha256")
    payload["artifact_kind"] = "Deform360PhysTwinCardinalityNormalizedCausalTrustFit"
    payload["controller_counts"] = {
        episode.episode_id: int(episode.controller_count) for episode in source
    }
    for episode in source:
        payload["source_inputs"][episode.episode_id]["controller_count"] = int(
            episode.controller_count
        )
    payload["method"]["decomposition"] = (
        "target_initial + base_action_response_trust / controller_count * "
        "(driven - zero_action) + autonomous_drift_trust * "
        "(zero_action - zero_action_initial)"
    )
    payload["method"]["controller_cardinality_normalization"] = (
        "registered controller count available before the object outcome"
    )
    selected_action = float(payload["selected_weights"]["action_response"])
    payload["effective_selected_action_response_by_episode"] = {
        episode.episode_id: selected_action / float(episode.controller_count)
        for episode in source
    }
    controller_counts = payload["controller_counts"]
    for fold in payload["leave_one_action_out"]:
        episode_id = fold["held_out_episode_id"]
        fold["controller_count"] = controller_counts[episode_id]
        fold["effective_action_response"] = float(
            fold["selected_weights"]["action_response"]
        ) / float(controller_counts[episode_id])
    payload["information_boundary"]["controller_count_available_before_outcome"] = True
    payload["claim_boundary"] = (
        "exploratory source-only controller-cardinality hypothesis discovered "
        "after the first bimanual prehensile result; independent-object "
        "preregistration is required"
    )
    payload["result_sha256"] = _result_sha256(payload)
    return payload


def _cardinality_normalized_episode(
    episode: CausalTrustEpisode,
) -> CausalTrustEpisode:
    return replace(
        episode,
        driven_m=episode.zero_action_m
        + (episode.driven_m - episode.zero_action_m) / float(episode.controller_count),
    )


def _validate_physical_candidate_sources(
    episodes_by_candidate: Mapping[
        PhysicalTrustParameters, Sequence[CausalTrustEpisode]
    ],
) -> tuple[tuple[PhysicalTrustParameters, ...], tuple[str, ...]]:
    physical_candidates = tuple(sorted(episodes_by_candidate))
    _require(
        len(physical_candidates) >= 2,
        "physical-grid trust needs at least two parameter candidates",
    )
    reference = tuple(episodes_by_candidate[physical_candidates[0]])
    _require(
        len(reference) >= 3,
        "physical-grid trust needs at least three source actions",
    )
    episode_ids = tuple(episode.episode_id for episode in reference)
    _require(
        len(set(episode_ids)) == len(episode_ids),
        "physical-grid source episode is repeated",
    )
    reference_by_id = {episode.episode_id: episode for episode in reference}
    for physical in physical_candidates:
        episodes = tuple(episodes_by_candidate[physical])
        _require(
            tuple(episode.episode_id for episode in episodes) == episode_ids,
            "physical candidates use different source episode ordering",
        )
        for episode in episodes:
            expected = reference_by_id[episode.episode_id]
            _require(
                episode.source_data_sha256 == expected.source_data_sha256
                and episode.train_stop_frame == expected.train_stop_frame
                and episode.controller_count == expected.controller_count,
                "physical candidates use different source provenance",
            )
            _require(
                np.array_equal(episode.target_m, expected.target_m)
                and np.array_equal(episode.visibility, expected.visibility)
                and np.array_equal(episode.validity, expected.validity),
                "physical candidates use different source observations",
            )
    return physical_candidates, episode_ids


def _candidate_subset_score(
    episode_ids: Sequence[str],
    weights: CausalTrustWeights,
    candidates: Sequence[CausalTrustWeights],
    table: Sequence[Mapping[str, Any]],
) -> float:
    index = candidates.index(weights)
    by_episode = table[index]["train_by_episode"]
    return float(
        np.mean(
            [
                float(by_episode[episode_id]["relative_score_vs_persistence"])
                for episode_id in episode_ids
            ]
        )
    )


def fit_cardinality_normalized_physical_grid_source_trust(
    episodes_by_candidate: Mapping[
        PhysicalTrustParameters, Sequence[CausalTrustEpisode]
    ],
    *,
    action_response_grid: Sequence[float] = tuple(np.linspace(0.0, 1.0, 11)),
    autonomous_drift_grid: Sequence[float] = tuple(np.linspace(0.0, 1.0, 11)),
) -> dict[str, Any]:
    """Cross-fit physical parameters and trust without reading source tails."""

    physical_candidates, episode_ids = _validate_physical_candidate_sources(
        episodes_by_candidate
    )
    trust_candidates = _candidate_grid(action_response_grid, autonomous_drift_grid)
    normalized_by_candidate = {
        physical: tuple(
            _cardinality_normalized_episode(episode)
            for episode in episodes_by_candidate[physical]
        )
        for physical in physical_candidates
    }
    episode_lookup = {
        physical: {episode.episode_id: episode for episode in episodes}
        for physical, episodes in normalized_by_candidate.items()
    }
    tables = {
        physical: _evaluate_candidate_table(episodes, trust_candidates)
        for physical, episodes in normalized_by_candidate.items()
    }

    def select_joint(
        fitting_ids: Sequence[str],
    ) -> tuple[
        PhysicalTrustParameters,
        CausalTrustWeights,
        list[dict[str, Any]],
    ]:
        rows = []
        for physical in physical_candidates:
            weights = _select_candidate(fitting_ids, trust_candidates, tables[physical])
            rows.append(
                {
                    "physical_parameters": physical.as_dict(),
                    "weights": weights.as_dict(),
                    "train_relative_score_vs_persistence": (
                        _candidate_subset_score(
                            fitting_ids,
                            weights,
                            trust_candidates,
                            tables[physical],
                        )
                    ),
                }
            )
        selected_index = min(
            range(len(rows)),
            key=lambda index: (
                rows[index]["train_relative_score_vs_persistence"],
                rows[index]["weights"]["action_response"]
                + rows[index]["weights"]["autonomous_drift"],
                rows[index]["weights"]["action_response"],
                rows[index]["weights"]["autonomous_drift"],
                physical_candidates[index].init_spring_y,
                physical_candidates[index].drag_damping,
                physical_candidates[index].dashpot_damping,
            ),
        )
        return (
            physical_candidates[selected_index],
            CausalTrustWeights(**rows[selected_index]["weights"]),
            rows,
        )

    selected_physical, selected_weights, all_source_table = select_joint(episode_ids)
    leave_one_action_out = []
    fold_selection_tables: dict[str, list[dict[str, Any]]] = {}
    for held_out_id in episode_ids:
        fitting_ids = [
            episode_id for episode_id in episode_ids if episode_id != held_out_id
        ]
        fold_physical, fold_weights, fold_table = select_joint(fitting_ids)
        fold_selection_tables[held_out_id] = fold_table
        held_metrics = _candidate_metrics(
            episode_lookup[fold_physical][held_out_id],
            fold_weights,
            interval="tail",
        )
        controller_count = episode_lookup[fold_physical][held_out_id].controller_count
        leave_one_action_out.append(
            {
                "held_out_episode_id": held_out_id,
                "selected_physical_parameters": fold_physical.as_dict(),
                "selected_weights": fold_weights.as_dict(),
                "controller_count": int(controller_count),
                "effective_action_response": (
                    fold_weights.action_response / float(controller_count)
                ),
                "tail_metrics": held_metrics,
                "beats_persistence_track": float(held_metrics["track_rmse_m"])
                < float(held_metrics["persistence_track_rmse_m"]),
                "beats_persistence_chamfer": float(held_metrics["chamfer_m"])
                < float(held_metrics["persistence_chamfer_m"]),
            }
        )
    pooled_tail = {
        name: float(
            np.mean(
                [float(fold["tail_metrics"][name]) for fold in leave_one_action_out]
            )
        )
        for name in (
            "track_rmse_m",
            "chamfer_m",
            "persistence_track_rmse_m",
            "persistence_chamfer_m",
        )
    }
    pooled_tail["track_improvement_fraction_vs_persistence"] = float(
        (pooled_tail["persistence_track_rmse_m"] - pooled_tail["track_rmse_m"])
        / pooled_tail["persistence_track_rmse_m"]
    )
    pooled_tail["chamfer_improvement_fraction_vs_persistence"] = float(
        (pooled_tail["persistence_chamfer_m"] - pooled_tail["chamfer_m"])
        / pooled_tail["persistence_chamfer_m"]
    )
    payload: dict[str, Any] = {
        "schema_version": PHYSTWIN_TRUST_SCHEMA_VERSION,
        "artifact_kind": (
            "Deform360PhysTwinCardinalityNormalizedPhysicalGridSourceFit"
        ),
        "source_episode_ids": list(episode_ids),
        "physical_candidates": [physical.as_dict() for physical in physical_candidates],
        "source_inputs_by_candidate": {
            json.dumps(physical.as_dict(), sort_keys=True): {
                episode.episode_id: {
                    "data_sha256": episode.source_data_sha256,
                    "driven_trajectory_sha256": episode.driven_trajectory_sha256,
                    "zero_action_trajectory_sha256": (
                        episode.zero_action_trajectory_sha256
                    ),
                    "controller_count": int(episode.controller_count),
                    "train_frame_range": [1, episode.train_stop_frame],
                    "untouched_tail_frame_range": [
                        episode.train_stop_frame,
                        len(episode.target_m),
                    ],
                }
                for episode in episodes_by_candidate[physical]
            }
            for physical in physical_candidates
        },
        "method": {
            "selection": (
                "joint physical-parameter and cardinality-normalized trust "
                "selection on fitting episodes' train frames only"
            ),
            "selection_score": (
                "execution-balanced mean normalized track RMSE and symmetric Chamfer"
            ),
            "outer_evaluation": "leave-one-action-out untouched source tails",
            "trust_candidate_count": len(trust_candidates),
        },
        "selected_physical_parameters": selected_physical.as_dict(),
        "selected_weights": selected_weights.as_dict(),
        "all_source_selection_table": all_source_table,
        "fold_selection_tables": fold_selection_tables,
        "leave_one_action_out": leave_one_action_out,
        "pooled_leave_one_action_out_tail": pooled_tail,
        "information_boundary": {
            "physical_and_trust_selection_use_source_train_frames_only": True,
            "source_tails_used_for_selection": False,
            "source_tails_used_for_outer_evaluation": True,
            "calibration_episode_read": False,
            "target_episode_read": False,
            "future_observation_required_at_prediction_time": False,
        },
        "claim_boundary": (
            "independent source-only competence test; calibration and target "
            "remain sealed until the registered source gate passes"
        ),
    }
    payload["result_sha256"] = _result_sha256(payload)
    return payload


def apply_cardinality_physical_grid_source_gate(
    fit: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    source_execution_protocol: Mapping[str, Any],
    *,
    registered_qa_by_episode: Mapping[str, bool],
    tail_mutation_invariant: bool,
) -> dict[str, Any]:
    """Apply the canonical independent source gate to outer held-out tails."""

    validate_cardinality_physical_grid_source_trust_artifact(fit)
    source_ids = [
        str(value) for value in parent_protocol["config"]["source_episode_ids"]
    ]
    _require(
        list(fit["source_episode_ids"]) == source_ids,
        "source gate episode ordering changed",
    )
    _require(
        source_execution_protocol["config"]["physical_arm_roles"]["primary_gate_arm"]
        == "source_pooled_grid",
        "source gate is not applied to the primary arm",
    )
    _require(
        set(registered_qa_by_episode) == set(source_ids),
        "registered QA does not cover every source episode",
    )
    folds = {fold["held_out_episode_id"]: fold for fold in fit["leave_one_action_out"]}
    bimanual = {
        str(value)
        for value in source_execution_protocol["config"]["source_gate_scope"][
            "bimanual_episode_ids"
        ]
    }

    def joint_win(fold: Mapping[str, Any]) -> bool:
        return bool(fold["beats_persistence_track"]) and bool(
            fold["beats_persistence_chamfer"]
        )

    joint_win_count = sum(joint_win(folds[episode_id]) for episode_id in source_ids)
    bimanual_joint_win_count = sum(
        joint_win(folds[episode_id])
        for episode_id in source_ids
        if episode_id in bimanual
    )
    maximum_degradation = max(
        max(
            float(folds[episode_id]["tail_metrics"][metric])
            / float(folds[episode_id]["tail_metrics"][f"persistence_{metric}"])
            - 1.0
            for metric in ("track_rmse_m", "chamfer_m")
        )
        for episode_id in source_ids
    )
    thresholds = parent_protocol["config"]["source_gate"]
    pooled = fit["pooled_leave_one_action_out_tail"]
    checks = {
        "joint_track_and_chamfer_win_count": joint_win_count
        >= int(thresholds["minimum_joint_track_and_chamfer_win_count"]),
        "bimanual_joint_win_count": bimanual_joint_win_count
        >= int(thresholds["minimum_bimanual_joint_win_count"]),
        "pooled_track_improvement": float(
            pooled["track_improvement_fraction_vs_persistence"]
        )
        >= float(thresholds["minimum_pooled_track_improvement_fraction"]),
        "pooled_chamfer_improvement": float(
            pooled["chamfer_improvement_fraction_vs_persistence"]
        )
        >= float(thresholds["minimum_pooled_chamfer_improvement_fraction"]),
        "maximum_any_metric_degradation": maximum_degradation
        <= float(thresholds["maximum_any_metric_degradation_fraction"]),
        "tail_mutation_invariance": bool(tail_mutation_invariant),
        "all_registered_qa": all(
            bool(registered_qa_by_episode[episode_id]) for episode_id in source_ids
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": PHYSTWIN_TRUST_SCHEMA_VERSION,
        "artifact_kind": "Deform360PhysTwinCardinalityPhysicalGridSourceGate",
        "fit_result_sha256": fit["result_sha256"],
        "parent_config_sha256": parent_protocol["config_sha256"],
        "source_execution_config_sha256": source_execution_protocol["config_sha256"],
        "joint_win_count": joint_win_count,
        "bimanual_joint_win_count": bimanual_joint_win_count,
        "maximum_any_metric_degradation_fraction": maximum_degradation,
        "pooled_leave_one_action_out_tail": pooled,
        "registered_qa_by_episode": dict(registered_qa_by_episode),
        "checks": checks,
        "passed": all(checks.values()),
        "decision": (
            "open_declared_calibration_only"
            if all(checks.values())
            else "freeze_negative_and_keep_calibration_and_target_sealed"
        ),
    }
    payload["result_sha256"] = _result_sha256(payload)
    return payload


def fit_regime_gated_source_causal_trust(
    episodes: Sequence[CausalTrustEpisode],
    regimes: Mapping[str, str],
    *,
    action_response_grid: Sequence[float] = tuple(np.linspace(0.0, 1.0, 11)),
    autonomous_drift_grid: Sequence[float] = tuple(np.linspace(0.0, 1.0, 11)),
) -> dict[str, Any]:
    """Fit prehensile trust and force nonprehensile actions to persistence."""

    source = tuple(episodes)
    episode_ids = [episode.episode_id for episode in source]
    _require(set(regimes) == set(episode_ids), "contact regimes differ from sources")
    _require(
        set(regimes.values()) <= {"prehensile", "nonprehensile"},
        "unknown contact regime",
    )
    prehensile = tuple(
        episode for episode in source if regimes[episode.episode_id] == "prehensile"
    )
    _require(
        len(prehensile) >= 3,
        "regime-gated trust needs three prehensile source actions",
    )
    prehensile_fit = fit_source_causal_trust(
        prehensile,
        action_response_grid=action_response_grid,
        autonomous_drift_grid=autonomous_drift_grid,
    )
    selected = CausalTrustWeights(**prehensile_fit["selected_weights"])
    persistence = CausalTrustWeights(0.0, 0.0)
    selected_train = {}
    selected_tail = {}
    for episode in source:
        weights = (
            selected if regimes[episode.episode_id] == "prehensile" else persistence
        )
        selected_train[episode.episode_id] = _candidate_metrics(
            episode, weights, interval="train"
        )
        selected_tail[episode.episode_id] = _candidate_metrics(
            episode, weights, interval="tail"
        )
    prehensile_folds = {
        fold["held_out_episode_id"]: fold
        for fold in prehensile_fit["leave_one_action_out"]
    }
    leave_one_action_out = []
    for episode in source:
        regime = regimes[episode.episode_id]
        if regime == "prehensile":
            base_fold = prehensile_folds[episode.episode_id]
            fold_weights = CausalTrustWeights(**base_fold["selected_weights"])
        else:
            fold_weights = persistence
        tail_metrics = _candidate_metrics(episode, fold_weights, interval="tail")
        track = float(tail_metrics["track_rmse_m"])
        chamfer = float(tail_metrics["chamfer_m"])
        persistence_track = float(tail_metrics["persistence_track_rmse_m"])
        persistence_chamfer = float(tail_metrics["persistence_chamfer_m"])
        leave_one_action_out.append(
            {
                "held_out_episode_id": episode.episode_id,
                "contact_regime": regime,
                "selected_weights": fold_weights.as_dict(),
                "tail_metrics": tail_metrics,
                "beats_persistence_track": track < persistence_track,
                "beats_persistence_chamfer": chamfer < persistence_chamfer,
                "non_degraded_track": track <= persistence_track * (1.0 + 1e-12),
                "non_degraded_chamfer": chamfer <= persistence_chamfer * (1.0 + 1e-12),
                "exact_persistence_fallback": bool(
                    regime == "nonprehensile"
                    and np.isclose(track, persistence_track, atol=1e-15, rtol=0.0)
                    and np.isclose(chamfer, persistence_chamfer, atol=1e-15, rtol=0.0)
                ),
            }
        )
    pooled_tail = {
        name: float(
            np.mean(
                [float(fold["tail_metrics"][name]) for fold in leave_one_action_out]
            )
        )
        for name in (
            "track_rmse_m",
            "chamfer_m",
            "persistence_track_rmse_m",
            "persistence_chamfer_m",
        )
    }
    pooled_tail["track_improvement_fraction_vs_persistence"] = float(
        (pooled_tail["persistence_track_rmse_m"] - pooled_tail["track_rmse_m"])
        / pooled_tail["persistence_track_rmse_m"]
    )
    pooled_tail["chamfer_improvement_fraction_vs_persistence"] = float(
        (pooled_tail["persistence_chamfer_m"] - pooled_tail["chamfer_m"])
        / pooled_tail["persistence_chamfer_m"]
    )
    prehensile_loo = [
        fold for fold in leave_one_action_out if fold["contact_regime"] == "prehensile"
    ]
    nonprehensile_loo = [
        fold
        for fold in leave_one_action_out
        if fold["contact_regime"] == "nonprehensile"
    ]
    prehensile_joint_wins = sum(
        bool(fold["beats_persistence_track"])
        and bool(fold["beats_persistence_chamfer"])
        for fold in prehensile_loo
    )
    maximum_prehensile_degradation = max(
        max(
            float(fold["tail_metrics"][metric])
            / float(fold["tail_metrics"][f"persistence_{metric}"])
            - 1.0
            for metric in ("track_rmse_m", "chamfer_m")
        )
        for fold in prehensile_loo
    )
    gate_checks = {
        "all_nonprehensile_folds_are_exact_persistence": all(
            bool(fold["exact_persistence_fallback"]) for fold in nonprehensile_loo
        ),
        "prehensile_joint_wins_at_least_two_of_three": (prehensile_joint_wins >= 2),
        "pooled_track_improvement_at_least_three_percent": (
            pooled_tail["track_improvement_fraction_vs_persistence"] >= 0.03
        ),
        "pooled_chamfer_improvement_at_least_one_percent": (
            pooled_tail["chamfer_improvement_fraction_vs_persistence"] >= 0.01
        ),
        "maximum_prehensile_metric_degradation_at_most_ten_percent": (
            maximum_prehensile_degradation <= 0.10
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": PHYSTWIN_TRUST_SCHEMA_VERSION,
        "artifact_kind": "Deform360PhysTwinRegimeGatedCausalTrustFit",
        "source_episode_ids": episode_ids,
        "contact_regimes": dict(regimes),
        "source_inputs": {
            episode.episode_id: {
                "data_sha256": episode.source_data_sha256,
                "driven_trajectory_sha256": episode.driven_trajectory_sha256,
                "zero_action_trajectory_sha256": (
                    episode.zero_action_trajectory_sha256
                ),
                "train_frame_range": [1, episode.train_stop_frame],
                "untouched_tail_frame_range": [
                    episode.train_stop_frame,
                    len(episode.target_m),
                ],
            }
            for episode in source
        },
        "policy": {
            "prehensile": {
                "method": "source-cross-fitted-causal-trust",
                "selected_weights": selected.as_dict(),
                "fit_result_sha256": prehensile_fit["result_sha256"],
            },
            "nonprehensile": {
                "method": "exact-persistence-fallback",
                "selected_weights": persistence.as_dict(),
                "reason": (
                    "the current fixed virtual spring is bilateral and is not "
                    "a valid model of unilateral pushing contact"
                ),
            },
        },
        "prehensile_fit": prehensile_fit,
        "selected_train_by_episode": selected_train,
        "selected_tail_by_episode": selected_tail,
        "leave_one_action_out": leave_one_action_out,
        "pooled_leave_one_action_out_tail": pooled_tail,
        "prospective_source_gate": {
            "criteria_frozen_before_episode_0006_outcome": True,
            "prehensile_joint_win_count": prehensile_joint_wins,
            "maximum_prehensile_metric_degradation_fraction": (
                maximum_prehensile_degradation
            ),
            "checks": gate_checks,
            "passed": all(gate_checks.values()),
        },
        "information_boundary": {
            "regime_available_before_outcome": True,
            "selection_uses_prehensile_source_train_frames_only": True,
            "source_tails_used_for_selection": False,
            "source_tails_used_for_exploratory_transfer_evaluation": True,
            "calibration_episode_read": False,
            "target_episode_read": False,
            "future_observation_required_at_prediction_time": False,
        },
        "claim_boundary": (
            "source-only contact-regime trust policy; nonprehensile performance "
            "is an exact safe fallback, not a physics improvement"
        ),
    }
    payload["result_sha256"] = _result_sha256(payload)
    return payload


def validate_source_causal_trust_artifact(payload: Mapping[str, Any]) -> None:
    _require(
        payload.get("schema_version") == PHYSTWIN_TRUST_SCHEMA_VERSION,
        "causal trust schema changed",
    )
    _require(
        payload.get("artifact_kind") == "Deform360PhysTwinCausalTrustFit",
        "causal trust artifact kind changed",
    )
    _require(
        payload.get("result_sha256") == _result_sha256(payload),
        "causal trust checksum mismatch",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("selection_uses_source_train_frames_only") is True
        and boundary.get("source_tails_used_for_selection") is False
        and boundary.get("calibration_episode_read") is False
        and boundary.get("target_episode_read") is False,
        "causal trust information boundary changed",
    )


def validate_cardinality_normalized_source_causal_trust_artifact(
    payload: Mapping[str, Any],
) -> None:
    _require(
        payload.get("schema_version") == PHYSTWIN_TRUST_SCHEMA_VERSION,
        "cardinality-normalized trust schema changed",
    )
    _require(
        payload.get("artifact_kind")
        == "Deform360PhysTwinCardinalityNormalizedCausalTrustFit",
        "cardinality-normalized trust artifact kind changed",
    )
    _require(
        payload.get("result_sha256") == _result_sha256(payload),
        "cardinality-normalized trust checksum mismatch",
    )
    source_ids = payload.get("source_episode_ids", [])
    controller_counts = payload.get("controller_counts", {})
    _require(
        isinstance(source_ids, list)
        and set(source_ids) == set(controller_counts)
        and all(
            isinstance(controller_counts[episode_id], int)
            and controller_counts[episode_id] >= 1
            for episode_id in source_ids
        ),
        "cardinality-normalized controller provenance changed",
    )
    effective = payload.get("effective_selected_action_response_by_episode", {})
    selected_action = float(payload.get("selected_weights", {}).get("action_response"))
    _require(
        set(effective) == set(source_ids)
        and all(
            np.isclose(
                float(effective[episode_id]),
                selected_action / float(controller_counts[episode_id]),
                atol=1e-15,
                rtol=0.0,
            )
            for episode_id in source_ids
        ),
        "cardinality-normalized effective trust changed",
    )
    for fold in payload.get("leave_one_action_out", []):
        episode_id = fold.get("held_out_episode_id")
        _require(
            fold.get("controller_count") == controller_counts.get(episode_id)
            and np.isclose(
                float(fold.get("effective_action_response")),
                float(fold.get("selected_weights", {}).get("action_response"))
                / float(controller_counts[episode_id]),
                atol=1e-15,
                rtol=0.0,
            ),
            "cardinality-normalized fold trust changed",
        )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("selection_uses_source_train_frames_only") is True
        and boundary.get("source_tails_used_for_selection") is False
        and boundary.get("controller_count_available_before_outcome") is True
        and boundary.get("calibration_episode_read") is False
        and boundary.get("target_episode_read") is False,
        "cardinality-normalized information boundary changed",
    )


def validate_cardinality_physical_grid_source_trust_artifact(
    payload: Mapping[str, Any],
) -> None:
    _require(
        payload.get("schema_version") == PHYSTWIN_TRUST_SCHEMA_VERSION,
        "physical-grid trust schema changed",
    )
    _require(
        payload.get("artifact_kind")
        == "Deform360PhysTwinCardinalityNormalizedPhysicalGridSourceFit",
        "physical-grid trust artifact kind changed",
    )
    _require(
        payload.get("result_sha256") == _result_sha256(payload),
        "physical-grid trust checksum mismatch",
    )
    source_ids = payload.get("source_episode_ids", [])
    folds = payload.get("leave_one_action_out", [])
    _require(
        isinstance(source_ids, list)
        and len(source_ids) >= 3
        and [fold.get("held_out_episode_id") for fold in folds] == source_ids,
        "physical-grid outer folds changed",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("physical_and_trust_selection_use_source_train_frames_only")
        is True
        and boundary.get("source_tails_used_for_selection") is False
        and boundary.get("source_tails_used_for_outer_evaluation") is True
        and boundary.get("calibration_episode_read") is False
        and boundary.get("target_episode_read") is False,
        "physical-grid trust information boundary changed",
    )


def validate_regime_gated_source_causal_trust_artifact(
    payload: Mapping[str, Any],
) -> None:
    _require(
        payload.get("schema_version") == PHYSTWIN_TRUST_SCHEMA_VERSION,
        "regime-gated trust schema changed",
    )
    _require(
        payload.get("artifact_kind") == "Deform360PhysTwinRegimeGatedCausalTrustFit",
        "regime-gated trust artifact kind changed",
    )
    _require(
        payload.get("result_sha256") == _result_sha256(payload),
        "regime-gated trust checksum mismatch",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("regime_available_before_outcome") is True
        and boundary.get("selection_uses_prehensile_source_train_frames_only") is True
        and boundary.get("source_tails_used_for_selection") is False
        and boundary.get("calibration_episode_read") is False
        and boundary.get("target_episode_read") is False,
        "regime-gated trust information boundary changed",
    )
    nonprehensile = payload.get("policy", {}).get("nonprehensile", {})
    _require(
        nonprehensile.get("method") == "exact-persistence-fallback"
        and nonprehensile.get("selected_weights")
        == {"action_response": 0.0, "autonomous_drift": 0.0},
        "nonprehensile fallback changed",
    )
    source_ids = payload.get("source_episode_ids", [])
    regimes = payload.get("contact_regimes", {})
    _require(
        isinstance(source_ids, list)
        and set(source_ids) == set(regimes)
        and set(source_ids) == set(payload.get("source_inputs", {})),
        "regime-gated source provenance changed",
    )
    prehensile_fit = payload.get("prehensile_fit", {})
    validate_source_causal_trust_artifact(prehensile_fit)
    _require(
        payload.get("policy", {}).get("prehensile", {}).get("fit_result_sha256")
        == prehensile_fit.get("result_sha256"),
        "embedded prehensile fit changed",
    )
    folds = payload.get("leave_one_action_out", [])
    _require(
        isinstance(folds, list) and len(folds) == len(source_ids),
        "regime-gated folds changed",
    )
    for fold in folds:
        episode_id = fold.get("held_out_episode_id")
        _require(
            fold.get("contact_regime") == regimes.get(episode_id),
            "fold contact regime changed",
        )
        if regimes.get(episode_id) == "nonprehensile":
            _require(
                fold.get("selected_weights")
                == {"action_response": 0.0, "autonomous_drift": 0.0}
                and fold.get("exact_persistence_fallback") is True,
                "nonprehensile fold is not exact persistence",
            )
    gate = payload.get("prospective_source_gate", {})
    checks = gate.get("checks", {})
    _require(
        gate.get("criteria_frozen_before_episode_0006_outcome") is True
        and isinstance(checks, Mapping)
        and bool(checks)
        and gate.get("passed") is all(bool(value) for value in checks.values()),
        "regime-gated source gate changed",
    )


def write_source_causal_trust_artifact(
    path: str | Path, payload: Mapping[str, Any]
) -> Path:
    validate_source_causal_trust_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def write_cardinality_normalized_source_causal_trust_artifact(
    path: str | Path, payload: Mapping[str, Any]
) -> Path:
    validate_cardinality_normalized_source_causal_trust_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def write_regime_gated_source_causal_trust_artifact(
    path: str | Path, payload: Mapping[str, Any]
) -> Path:
    validate_regime_gated_source_causal_trust_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "CausalTrustEpisode",
    "CausalTrustWeights",
    "causal_control_variate_prediction",
    "fit_cardinality_normalized_source_causal_trust",
    "fit_regime_gated_source_causal_trust",
    "fit_source_causal_trust",
    "load_cardinality_trust_protocol",
    "load_contact_anchored_causal_trust_protocol",
    "load_official_phystwin_trust_episode",
    "validate_cardinality_normalized_source_causal_trust_artifact",
    "validate_regime_gated_source_causal_trust_artifact",
    "validate_source_causal_trust_artifact",
    "write_cardinality_normalized_source_causal_trust_artifact",
    "write_regime_gated_source_causal_trust_artifact",
    "write_source_causal_trust_artifact",
]
