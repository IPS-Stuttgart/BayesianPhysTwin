"""Source-only admission gate for the official PhysTwin Warp simulator."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from causal4d_public.deform360_replication import (
    PINNED_OFFICIAL_PHYSTWIN_COMMIT,
    load_deform360_replication_protocol,
)
from causal4d_public.deform360_rope_dynamics import RopeDynamicsObservation
from causal4d_public.deform360_rope_observations import (
    load_source_rope_dynamics_observation,
)


PHYSTWIN_FEASIBILITY_SCHEMA_VERSION = 1
OFFICIAL_SIMULATOR_RELATIVE_PATH = (
    Path("qqtt") / "model" / "diff_simulator" / "spring_mass_warp.py"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phystwin_feasibility_artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


@dataclass(frozen=True)
class WarpRopeFeasibilityConfig:
    """Source-only numerical choices for the official-Warp admission test."""

    prefix_frame_count: int = 6
    substeps: int = 128
    initial_ground_clearance_m: float = 0.001
    stretch_spring_y_grid: tuple[float, ...] = (
        100.0,
        300.0,
        1000.0,
        3000.0,
        10000.0,
    )
    bend_spring_y_grid: tuple[float, ...] = (1e-6, 10.0, 100.0, 1000.0)
    controller_spring_y_grid: tuple[float, ...] = (
        100.0,
        300.0,
        1000.0,
        3000.0,
        10000.0,
    )
    ground_friction_grid: tuple[float, ...] = (0.0, 0.3)
    dashpot_damping: float = 0.0
    drag_damping: float = 3.0
    ground_elasticity: float = 0.0
    inactive_controller_spring_y: float = 1e-12
    maximum_repeat_rollout_rmse_m: float = 1e-4
    maximum_p99_relative_edge_strain: float = 0.5
    minimum_source_chamfer_improvement_fraction: float = 0.05
    minimum_leave_one_source_win_fraction: float = 0.6

    def __post_init__(self) -> None:
        _require(self.prefix_frame_count >= 2, "prefix must contain two frames")
        _require(self.substeps >= 1, "substeps must be positive")
        _require(
            self.initial_ground_clearance_m >= 0.0,
            "ground clearance must be nonnegative",
        )
        for name in (
            "stretch_spring_y_grid",
            "bend_spring_y_grid",
            "controller_spring_y_grid",
            "ground_friction_grid",
        ):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            _require(len(values) >= 1, f"{name} must be nonempty")
            _require(
                np.all(np.isfinite(values)) and np.all(values >= 0.0),
                f"{name} must contain finite nonnegative values",
            )
        _require(
            self.inactive_controller_spring_y > 0.0,
            "inactive controller stiffness must be positive",
        )


@dataclass(frozen=True)
class WarpRopeCandidate:
    stretch_spring_y: float
    bend_spring_y: float
    controller_spring_y: float
    ground_friction: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class WarpRopeForecastCase:
    episode_index: int
    episode_id: str
    positions_m: np.ndarray
    controller_positions_m: np.ndarray
    contact_active: np.ndarray
    contact_node_indices: tuple[int, ...]
    contact_offsets_m: np.ndarray
    dt_seconds: float
    prefix_start_index: int
    prefix_end_index_exclusive: int


def warp_rope_candidates(
    config: WarpRopeFeasibilityConfig,
) -> tuple[WarpRopeCandidate, ...]:
    return tuple(
        WarpRopeCandidate(
            stretch_spring_y=float(stretch),
            bend_spring_y=float(bend),
            controller_spring_y=float(controller),
            ground_friction=float(friction),
        )
        for stretch, bend, controller, friction in product(
            config.stretch_spring_y_grid,
            config.bend_spring_y_grid,
            config.controller_spring_y_grid,
            config.ground_friction_grid,
        )
    )


def deform360_xyz_to_warp_xzy(
    values: np.ndarray,
    *,
    initial_support_height_m: float,
    clearance_m: float,
) -> np.ndarray:
    """Map Deform360 coordinates to Warp's z-up ground convention."""

    points = np.asarray(values, dtype=np.float64)
    _require(points.shape[-1] == 3, "coordinates must end in dimension three")
    _require(np.all(np.isfinite(points)), "coordinates must be finite")
    transformed = np.empty_like(points)
    transformed[..., 0] = points[..., 0]
    transformed[..., 1] = points[..., 2]
    transformed[..., 2] = (
        points[..., 1] - float(initial_support_height_m) + float(clearance_m)
    )
    return transformed


def _source_forecast_case(
    episode_index: int,
    observation: RopeDynamicsObservation,
    config: WarpRopeFeasibilityConfig,
) -> WarpRopeForecastCase:
    contact_frames = np.flatnonzero(np.any(observation.contact_active, axis=1))
    _require(len(contact_frames) > 0, "source observation contains no active contact")
    prefix_start = int(contact_frames[0])
    prefix_end = prefix_start + config.prefix_frame_count
    _require(prefix_end < len(observation.positions_m), "source prefix has no future")
    return WarpRopeForecastCase(
        episode_index=episode_index,
        episode_id=observation.episode_id,
        positions_m=np.asarray(
            observation.positions_m[prefix_end - 1 :], dtype=np.float64
        ),
        controller_positions_m=np.asarray(
            observation.controller_positions_m[prefix_end - 1 :], dtype=np.float64
        ),
        contact_active=np.asarray(
            observation.contact_active[prefix_end - 1 :], dtype=bool
        ),
        contact_node_indices=observation.contact_node_indices,
        contact_offsets_m=np.asarray(observation.contact_offsets_m, dtype=np.float64),
        dt_seconds=float(observation.dt_seconds),
        prefix_start_index=prefix_start,
        prefix_end_index_exclusive=prefix_end,
    )


def load_locked_warp_source_cases(
    protocol_path: str | Path,
    observation_json_paths: Sequence[str | Path],
    *,
    config: WarpRopeFeasibilityConfig,
) -> tuple[dict[str, Any], tuple[WarpRopeForecastCase, ...], list[dict[str, Any]]]:
    protocol = load_deform360_replication_protocol(protocol_path)
    gate = protocol["config"]["gates"]["official_warp_feasibility"]
    allowed = tuple(map(int, gate["allowed_source_episode_ids"]))
    forbidden = set(map(int, gate["forbidden_episode_ids"]))
    _require(not set(allowed) & forbidden, "allowed and forbidden source sets overlap")
    records: dict[int, tuple[dict[str, Any], Path]] = {}
    for value in observation_json_paths:
        path = Path(value).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        episode_index = int(payload["episode_index"])
        _require(episode_index not in forbidden, "forbidden episode was provided")
        _require(episode_index in allowed, "unregistered source episode was provided")
        _require(episode_index not in records, "source episode was provided twice")
        records[episode_index] = (payload, path)
    _require(
        tuple(sorted(records)) == tuple(sorted(allowed)),
        "source observations do not match the locked Warp gate",
    )
    cases = []
    inputs = []
    for episode_index in allowed:
        payload, path = records[episode_index]
        observation = load_source_rope_dynamics_observation(payload)
        cases.append(_source_forecast_case(episode_index, observation, config))
        inputs.append(
            {
                "episode_index": episode_index,
                "episode_id": observation.episode_id,
                "observation_json": str(path),
                "observation_json_sha256": _sha256_file(path),
                "observation_result_sha256": payload["result_sha256"],
                "archive_path": payload["archive"]["path"],
                "archive_sha256": payload["archive"]["sha256"],
            }
        )
    return protocol, tuple(cases), inputs


def _mean_chamfer_m(reference: np.ndarray, prediction: np.ndarray) -> float:
    _require(reference.shape == prediction.shape, "trajectory shapes disagree")
    difference = reference[:, :, None, :] - prediction[:, None, :, :]
    distances = np.linalg.norm(difference, axis=3)
    per_frame = 0.5 * (
        np.mean(np.min(distances, axis=1), axis=1)
        + np.mean(np.min(distances, axis=2), axis=1)
    )
    return float(np.mean(per_frame))


def _trajectory_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    rest_lengths: np.ndarray,
) -> dict[str, float | int]:
    finite_count = int(np.count_nonzero(~np.isfinite(prediction)))
    if finite_count:
        return {
            "chamfer_distance_m": float("inf"),
            "track_error_m": float("inf"),
            "p99_relative_edge_strain": float("inf"),
            "nonfinite_state_count": finite_count,
        }
    edge_lengths = np.linalg.norm(np.diff(prediction, axis=1), axis=2)
    relative_strain = np.abs(edge_lengths / rest_lengths[None] - 1.0)
    return {
        "chamfer_distance_m": _mean_chamfer_m(reference, prediction),
        "track_error_m": float(np.mean(np.linalg.norm(reference - prediction, axis=2))),
        "p99_relative_edge_strain": float(np.quantile(relative_strain, 0.99)),
        "nonfinite_state_count": 0,
    }


def _finite_metric(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _git_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class _OfficialWarpRopeRunner:
    def __init__(
        self,
        official_repo: Path,
        case: WarpRopeForecastCase,
        config: WarpRopeFeasibilityConfig,
        *,
        device: str,
    ) -> None:
        try:
            import torch
            import warp as wp
        except ImportError as error:  # pragma: no cover - GPU integration
            raise RuntimeError("official-Warp gate requires torch and warp") from error
        from bayesian_phystwin._phystwin_warp_backend import (
            load_official_spring_mass_module,
            make_reliability_simulator_class,
        )
        from bayesian_phystwin.phystwin_refit import build_phystwin_track_objective

        self.torch = torch
        self.wp = wp
        self.config = config
        self.case = case
        self.device = device
        initial = case.positions_m[0]
        support_height = float(np.min(initial[:, 1]))
        self.reference = deform360_xyz_to_warp_xzy(
            case.positions_m,
            initial_support_height_m=support_height,
            clearance_m=config.initial_ground_clearance_m,
        )
        self.controllers = deform360_xyz_to_warp_xzy(
            case.controller_positions_m,
            initial_support_height_m=support_height,
            clearance_m=config.initial_ground_clearance_m,
        )
        self.initial_positions = self.reference[0].astype(np.float32)
        self.rest_lengths = np.linalg.norm(
            np.diff(self.initial_positions.astype(np.float64), axis=0), axis=1
        )
        _require(
            np.all(self.rest_lengths > 1e-5),
            "rope chain contains a degenerate edge",
        )
        (
            vertices,
            springs,
            all_rest_lengths,
            masses,
            num_object_springs,
            stretch_count,
            bend_count,
        ) = self._graph_arrays()
        self.springs = springs
        self.all_rest_lengths = all_rest_lengths
        self.num_object_springs = num_object_springs
        self.stretch_count = stretch_count
        self.bend_count = bend_count

        runtime_config = SimpleNamespace(
            device=device,
            use_graph=True,
            data_type="real",
            collision_learn=False,
            chamfer_weight=0.0,
            track_weight=0.0,
            acc_weight=0.0,
        )
        official = load_official_spring_mass_module(
            official_repo, runtime_config=runtime_config
        )
        simulator_class = make_reliability_simulator_class(official)
        visible = np.ones(self.reference.shape[:2], dtype=bool)
        motion_valid = np.ones(
            (len(self.reference) - 1, self.reference.shape[1]), dtype=bool
        )
        objective = build_phystwin_track_objective(
            visible, motion_valid, variant="hard"
        )

        def tensor(values: np.ndarray, dtype):
            return torch.as_tensor(values, dtype=dtype, device=device).contiguous()

        gt = tensor(self.reference.astype(np.float32), torch.float32)
        self.simulator = simulator_class(
            tensor(vertices, torch.float32),
            tensor(springs, torch.int32),
            tensor(all_rest_lengths, torch.float32),
            tensor(masses, torch.float32),
            dt=case.dt_seconds / config.substeps,
            num_substeps=config.substeps,
            spring_Y=1000.0,
            collide_elas=config.ground_elasticity,
            collide_fric=0.3,
            dashpot_damping=config.dashpot_damping,
            drag_damping=config.drag_damping,
            collide_object_elas=0.0,
            collide_object_fric=0.0,
            collision_dist=0.01,
            num_object_points=len(self.initial_positions),
            num_surface_points=len(self.initial_positions),
            num_original_points=len(self.initial_positions),
            controller_points=tensor(
                self.controllers.astype(np.float32), torch.float32
            ),
            reverse_z=False,
            spring_Y_min=0.0,
            spring_Y_max=1e5,
            gt_object_points=gt,
            gt_object_visibilities=tensor(visible.astype(np.int32), torch.int32),
            gt_object_motions_valid=tensor(motion_valid.astype(np.int32), torch.int32),
            self_collision=False,
            disable_backward=True,
            objective=objective,
            observation_variance=1e-4,
            outlier_variance_multiplier=100.0,
            spring_parameterization="dense",
            num_object_springs=num_object_springs,
            deterministic_spring_forces=True,
        )
        self.initial_tensor = tensor(
            self.initial_positions.astype(np.float32), torch.float32
        )
        self.zero_velocity_tensor = torch.zeros_like(self.initial_tensor)
        self.wp_initial = wp.from_torch(
            self.initial_tensor, dtype=wp.vec3, requires_grad=False
        )
        self.wp_zero_velocity = wp.from_torch(
            self.zero_velocity_tensor, dtype=wp.vec3, requires_grad=False
        )
        wp.synchronize()

    def _graph_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
        node_count = len(self.initial_positions)
        stretch = [(index, index + 1) for index in range(node_count - 1)]
        bend = [(index, index + 2) for index in range(node_count - 2)]
        control = [
            (node_count + controller, int(node))
            for controller, node in enumerate(self.case.contact_node_indices)
        ]
        springs = np.asarray(stretch + bend + control, dtype=np.int32)
        object_rest = np.linalg.norm(
            self.initial_positions[springs[: len(stretch) + len(bend), 1]]
            - self.initial_positions[springs[: len(stretch) + len(bend), 0]],
            axis=1,
        )
        control_rest = np.maximum(
            np.linalg.norm(self.case.contact_offsets_m, axis=1), 1e-3
        )
        rest = np.concatenate((object_rest, control_rest)).astype(np.float32)
        vertices = np.concatenate(
            (self.initial_positions, self.controllers[0].astype(np.float32)), axis=0
        )
        masses = np.ones(len(vertices), dtype=np.float32)
        return (
            vertices,
            springs,
            rest,
            masses,
            len(stretch) + len(bend),
            len(stretch),
            len(bend),
        )

    def _spring_log_y(
        self,
        candidate: WarpRopeCandidate,
        active: tuple[bool, ...],
    ):
        values = np.empty(len(self.springs), dtype=np.float32)
        values[: self.stretch_count] = candidate.stretch_spring_y
        values[self.stretch_count : self.stretch_count + self.bend_count] = (
            candidate.bend_spring_y
        )
        for controller, enabled in enumerate(active):
            values[self.num_object_springs + controller] = (
                candidate.controller_spring_y
                if enabled
                else self.config.inactive_controller_spring_y
            )
        return self.torch.log(
            self.torch.as_tensor(values, dtype=self.torch.float32, device=self.device)
        ).contiguous()

    def rollout(self, candidate: WarpRopeCandidate) -> np.ndarray:
        torch = self.torch
        wp = self.wp
        friction = torch.as_tensor(
            [candidate.ground_friction],
            dtype=torch.float32,
            device=self.device,
        ).contiguous()
        elasticity = torch.as_tensor(
            [self.config.ground_elasticity],
            dtype=torch.float32,
            device=self.device,
        ).contiguous()
        self.simulator.set_collide(elasticity, friction)
        self.simulator.set_init_state(
            self.wp_initial, self.wp_zero_velocity, pure_inference=True
        )
        wp.synchronize()
        trajectory = [self.initial_positions.astype(np.float64)]
        previous_active: tuple[bool, ...] | None = None
        for frame in range(1, len(self.reference)):
            active = tuple(map(bool, self.case.contact_active[frame]))
            if active != previous_active:
                self.simulator.set_reference_spring_y(
                    self._spring_log_y(candidate, active)
                )
                previous_active = active
            self.simulator.set_controller_target(frame, pure_inference=True)
            wp.capture_launch(self.simulator.forward_graph)
            wp.synchronize()
            position = (
                wp.to_torch(self.simulator.wp_states[-1].wp_x)
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
            trajectory.append(position.astype(np.float64))
            if not np.all(np.isfinite(position)):
                missing = len(self.reference) - len(trajectory)
                trajectory.extend(
                    [
                        np.full_like(self.initial_positions, np.nan, dtype=np.float64)
                        for _ in range(missing)
                    ]
                )
                break
            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
                pure_inference=True,
            )
        return np.stack(trajectory)


def _summarize_candidate_scores(
    candidates: Sequence[WarpRopeCandidate],
    episode_ids: Sequence[str],
    scores: np.ndarray,
    track_scores: np.ndarray,
    persistence_scores: np.ndarray,
    *,
    config: WarpRopeFeasibilityConfig,
) -> dict[str, Any]:
    _require(
        scores.shape == (len(candidates), len(episode_ids)),
        "candidate score matrix has the wrong shape",
    )
    valid = np.all(np.isfinite(scores), axis=1)
    _require(np.any(valid), "every official-Warp candidate was nonfinite")
    valid_indices = np.flatnonzero(valid)
    selected_index = int(
        min(
            valid_indices,
            key=lambda index: (float(np.mean(scores[index])), int(index)),
        )
    )
    leave_one_out = []
    for held_index, episode_id in enumerate(episode_ids):
        training = [index for index in range(len(episode_ids)) if index != held_index]
        fold_index = int(
            min(
                valid_indices,
                key=lambda index: (
                    float(np.mean(scores[index, training])),
                    int(index),
                ),
            )
        )
        held_score = float(scores[fold_index, held_index])
        baseline = float(persistence_scores[held_index])
        leave_one_out.append(
            {
                "held_out_episode_id": episode_id,
                "selected_candidate_index": fold_index,
                "held_out_chamfer_distance_m": held_score,
                "persistence_chamfer_distance_m": baseline,
                "held_out_track_error_m": float(track_scores[fold_index, held_index]),
                "chamfer_better_than_persistence": bool(held_score < baseline),
            }
        )
    selected_mean = float(np.mean(scores[selected_index]))
    persistence_mean = float(np.mean(persistence_scores))
    improvement = (persistence_mean - selected_mean) / persistence_mean
    loo_win_fraction = float(
        np.mean([row["chamfer_better_than_persistence"] for row in leave_one_out])
    )
    loo_mean = float(
        np.mean([row["held_out_chamfer_distance_m"] for row in leave_one_out])
    )
    return {
        "selected_candidate_index": selected_index,
        "selected_parameters": candidates[selected_index].as_dict(),
        "selected_mean_chamfer_distance_m": selected_mean,
        "selected_mean_track_error_m": float(np.mean(track_scores[selected_index])),
        "persistence_mean_chamfer_distance_m": persistence_mean,
        "observed_source_chamfer_improvement_fraction": improvement,
        "observed_leave_one_source_win_fraction": loo_win_fraction,
        "leave_one_source_out_mean_chamfer_distance_m": loo_mean,
        "leave_one_source_out": leave_one_out,
        "competence_passed": bool(
            improvement >= config.minimum_source_chamfer_improvement_fraction
            and loo_win_fraction >= config.minimum_leave_one_source_win_fraction
            and loo_mean < persistence_mean
        ),
    }


def run_official_warp_feasibility_gate(
    protocol_path: str | Path,
    official_repo: str | Path,
    observation_json_paths: Sequence[str | Path],
    output_archive_path: str | Path,
    *,
    config: WarpRopeFeasibilityConfig | None = None,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Fit and audit official Warp using only the locked 001-rope sources."""

    cfg = config or WarpRopeFeasibilityConfig()
    protocol, cases, source_inputs = load_locked_warp_source_cases(
        protocol_path, observation_json_paths, config=cfg
    )
    official_path = Path(official_repo).resolve()
    _require(official_path.is_dir(), "official PhysTwin repository is missing")
    official_head = _git_head(official_path)
    _require(
        official_head == PINNED_OFFICIAL_PHYSTWIN_COMMIT,
        "official PhysTwin checkout differs from the preregistered commit",
    )
    simulator_source = official_path / OFFICIAL_SIMULATOR_RELATIVE_PATH
    _require(simulator_source.is_file(), "official Warp simulator source is missing")
    candidates = warp_rope_candidates(cfg)
    episode_ids = [case.episode_id for case in cases]
    scores = np.full((len(candidates), len(cases)), np.inf, dtype=np.float64)
    track_scores = np.full_like(scores, np.inf)
    strain_scores = np.full_like(scores, np.inf)
    nonfinite_counts = np.zeros_like(scores, dtype=np.int64)
    persistence_scores = np.empty(len(cases), dtype=np.float64)
    runners = []
    for episode_index, case in enumerate(cases):
        runner = _OfficialWarpRopeRunner(official_path, case, cfg, device=device)
        runners.append(runner)
        reference = runner.reference[1:]
        persistence = np.repeat(
            runner.initial_positions[None].astype(np.float64),
            len(reference),
            axis=0,
        )
        persistence_scores[episode_index] = _mean_chamfer_m(reference, persistence)
        for candidate_index, candidate in enumerate(candidates):
            prediction = runner.rollout(candidate)[1:]
            metrics = _trajectory_metrics(reference, prediction, runner.rest_lengths)
            scores[candidate_index, episode_index] = float(
                metrics["chamfer_distance_m"]
            )
            track_scores[candidate_index, episode_index] = float(
                metrics["track_error_m"]
            )
            strain_scores[candidate_index, episode_index] = float(
                metrics["p99_relative_edge_strain"]
            )
            nonfinite_counts[candidate_index, episode_index] = int(
                metrics["nonfinite_state_count"]
            )
    summary = _summarize_candidate_scores(
        candidates,
        episode_ids,
        scores,
        track_scores,
        persistence_scores,
        config=cfg,
    )
    selected_index = int(summary["selected_candidate_index"])
    selected = candidates[selected_index]
    selected_predictions = {}
    repeat_rmse = []
    selected_metrics = []
    for runner in runners:
        first = runner.rollout(selected)
        second = runner.rollout(selected)
        rmse = float(np.sqrt(np.mean(np.square(first - second))))
        repeat_rmse.append(rmse)
        selected_predictions[
            f"episode_{runner.case.episode_index:04d}_prediction_m"
        ] = first
        selected_predictions[f"episode_{runner.case.episode_index:04d}_repeat_m"] = (
            second
        )
        selected_metrics.append(
            {
                "episode_index": runner.case.episode_index,
                "episode_id": runner.case.episode_id,
                "chamfer_distance_m": float(
                    scores[selected_index, len(selected_metrics)]
                ),
                "track_error_m": float(
                    track_scores[selected_index, len(selected_metrics)]
                ),
                "persistence_chamfer_distance_m": float(
                    persistence_scores[len(selected_metrics)]
                ),
                "p99_relative_edge_strain": float(
                    strain_scores[selected_index, len(selected_metrics)]
                ),
                "nonfinite_state_count": int(
                    nonfinite_counts[selected_index, len(selected_metrics)]
                ),
                "repeat_rollout_rmse_m": rmse,
            }
        )
    archive = Path(output_archive_path).resolve()
    _require(archive.suffix == ".npz", "prediction archive must end in .npz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive, **selected_predictions)
    maximum_repeat = float(np.max(repeat_rmse))
    maximum_strain = float(np.max(strain_scores[selected_index]))
    maximum_nonfinite = int(np.max(nonfinite_counts[selected_index]))
    numerical_passed = bool(
        maximum_repeat <= cfg.maximum_repeat_rollout_rmse_m
        and maximum_strain <= cfg.maximum_p99_relative_edge_strain
        and maximum_nonfinite == 0
    )
    gate_passed = bool(summary["competence_passed"] and numerical_passed)
    candidate_rows = []
    for candidate_index, candidate in enumerate(candidates):
        per_episode = []
        for episode_index, episode_id in enumerate(episode_ids):
            per_episode.append(
                {
                    "episode_id": episode_id,
                    "chamfer_distance_m": _finite_metric(
                        scores[candidate_index, episode_index]
                    ),
                    "track_error_m": _finite_metric(
                        track_scores[candidate_index, episode_index]
                    ),
                    "p99_relative_edge_strain": _finite_metric(
                        strain_scores[candidate_index, episode_index]
                    ),
                    "nonfinite_state_count": int(
                        nonfinite_counts[candidate_index, episode_index]
                    ),
                }
            )
        candidate_rows.append(
            {
                "candidate_index": candidate_index,
                "parameters": candidate.as_dict(),
                "per_episode": per_episode,
            }
        )
    return {
        "schema_version": PHYSTWIN_FEASIBILITY_SCHEMA_VERSION,
        "artifact_kind": "Deform360OfficialWarpSourceFeasibilityGate",
        "protocol_id": protocol["config"]["protocol_id"],
        "replication_config_sha256": protocol["config_sha256"],
        "official_phystwin": {
            "repository": str(official_path),
            "commit": official_head,
            "simulator_source": str(simulator_source),
            "simulator_source_sha256": _sha256_file(simulator_source),
        },
        "config": asdict(cfg),
        "candidate_count": len(candidates),
        "source_inputs": source_inputs,
        "candidate_scores": candidate_rows,
        "selected_source_metrics": selected_metrics,
        "source_competence": summary,
        "numerical_audit": {
            "maximum_repeat_rollout_rmse_m": maximum_repeat,
            "maximum_allowed_repeat_rollout_rmse_m": (
                cfg.maximum_repeat_rollout_rmse_m
            ),
            "maximum_selected_p99_relative_edge_strain": maximum_strain,
            "maximum_allowed_p99_relative_edge_strain": (
                cfg.maximum_p99_relative_edge_strain
            ),
            "maximum_selected_nonfinite_state_count": maximum_nonfinite,
            "passed": numerical_passed,
        },
        "prediction_archive": {
            "path": str(archive),
            "sha256": _sha256_file(archive),
            "bytes": archive.stat().st_size,
        },
        "information_boundary": {
            "source_episode_ids": [case.episode_index for case in cases],
            "forbidden_episode_ids": protocol["config"]["gates"][
                "official_warp_feasibility"
            ]["forbidden_episode_ids"],
            "target_episode_6_read": False,
            "selected_replication_object_media_read": False,
            "target_outcomes_read": False,
        },
        "claim_boundary": (
            "Official PhysTwin Warp simulator feasibility on a source-only "
            "21-node public rope graph; not a dense reconstructed PhysTwin."
        ),
        "gate_passed": gate_passed,
        "admitted_backend": "official_warp" if gate_passed else None,
    }


def write_official_warp_feasibility_artifact(
    path: str | Path, payload: Mapping[str, Any]
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    artifact["result_sha256"] = phystwin_feasibility_artifact_sha256(artifact)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def validate_official_warp_feasibility_artifact(
    payload: Mapping[str, Any], *, verify_archive: bool = True
) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == PHYSTWIN_FEASIBILITY_SCHEMA_VERSION,
        "unsupported official-Warp feasibility schema",
    )
    _require(
        payload.get("artifact_kind") == "Deform360OfficialWarpSourceFeasibilityGate",
        "unexpected official-Warp artifact kind",
    )
    _require(
        payload.get("result_sha256") == phystwin_feasibility_artifact_sha256(payload),
        "official-Warp artifact checksum mismatch",
    )
    _require(
        payload.get("information_boundary", {}).get("target_episode_6_read") is False,
        "official-Warp artifact read the exhausted target",
    )
    if verify_archive:
        archive = Path(payload["prediction_archive"]["path"])
        _require(archive.is_file(), "official-Warp prediction archive is missing")
        _require(
            _sha256_file(archive) == payload["prediction_archive"]["sha256"],
            "official-Warp prediction archive checksum mismatch",
        )
    return {
        "passed": True,
        "gate_passed": bool(payload["gate_passed"]),
        "result_sha256": payload["result_sha256"],
    }
