"""Released-case preparation and source-only gate for equivariant forces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .phystwin_equivariant_force import EquivariantForceConfig
from .phystwin_equivariant_force_data import (
    EquivariantForceEpisode,
    load_equivariant_force_episode,
    write_equivariant_force_episode,
)
from .phystwin_equivariant_force_training import (
    EquivariantForceTrainingConfig,
    crossfit_equivariant_force_competence,
)
from .phystwin_equivariant_force_warp import (
    controller_attachment_matrix,
    controller_conditioning_fields,
)
from .phystwin_force_targets import (
    acceleration_to_force_targets,
    estimate_residual_acceleration,
    graph_smooth_residual_acceleration,
    robust_prefix_force_scale,
)
from .phystwin_graph import (
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)
from .phystwin_graph_discrepancy import normalized_spring_laplacian
from .phystwin_residual_dynamics import (
    _load_pickle,
    _sha256,
    _target_validity,
)


EQUIVARIANT_FORCE_SOURCE_CONTRACT = (
    "phystwin-equivariant-generalized-force-source-v2"
)


@dataclass(frozen=True)
class ForceTargetBuildConfig:
    """Frozen robust target and graph-lifting settings."""

    window_radius: int = 3
    huber_delta_m: float = 0.003
    robust_iterations: int = 4
    variance_floor_m2ps4: float = 1.0e-6
    graph_prior_strength: float = 0.5
    graph_ridge: float = 1.0e-8
    graph_covariance_probes: int = 8
    force_scale_node_quantile: float = 0.95
    force_scale_temporal_quantile: float = 0.90
    minimum_force_scale_sim: float = 0.10
    maximum_force_scale_sim: float = 50.0
    minimum_training_weight: float = 1.0e-4

    def __post_init__(self) -> None:
        if self.window_radius < 2 or self.robust_iterations < 1:
            raise ValueError("force target windows and iterations are invalid")
        positive = (
            self.huber_delta_m,
            self.variance_floor_m2ps4,
            self.graph_prior_strength,
            self.graph_ridge,
            self.minimum_force_scale_sim,
            self.maximum_force_scale_sim,
            self.minimum_training_weight,
        )
        if any(value <= 0.0 or not np.isfinite(value) for value in positive):
            raise ValueError("force target scales must be positive and finite")
        if self.graph_covariance_probes < 1:
            raise ValueError("graph_covariance_probes must be positive")
        if (
            not 0.0 < self.force_scale_node_quantile < 1.0
            or not 0.0 < self.force_scale_temporal_quantile < 1.0
        ):
            raise ValueError("force-scale quantiles must lie in (0,1)")
        if self.maximum_force_scale_sim < self.minimum_force_scale_sim:
            raise ValueError("force-scale bounds must be ordered")


@dataclass(frozen=True)
class EquivariantForceSourceProtocol:
    """Validated protocol plus the typed implementation configurations."""

    payload: Mapping[str, Any]
    model: EquivariantForceConfig
    training: EquivariantForceTrainingConfig
    target: ForceTargetBuildConfig


def _backward_velocity(positions_m: np.ndarray, frame_dt_s: float) -> np.ndarray:
    positions = np.asarray(positions_m, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions_m must have shape (T,N,3)")
    if frame_dt_s <= 0.0:
        raise ValueError("frame_dt_s must be positive")
    velocity = np.zeros_like(positions)
    velocity[1:] = np.diff(positions, axis=0) / frame_dt_s
    return velocity


def _regime_probabilities(activity: np.ndarray) -> np.ndarray:
    """Return a conservative free/sticking mixture without residual cues."""

    values = np.asarray(activity, dtype=float)
    if values.ndim != 1 or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("activity must be a vector in [0,1]")
    regimes = np.zeros((len(values), 5), dtype=float)
    regimes[:, 0] = 1.0 - values
    regimes[:, 2] = values
    return regimes


def build_released_equivariant_force_episode(
    case_id: str,
    final_data_path: str | Path,
    baseline_trajectory_path: str | Path,
    optimal_params_path: str | Path,
    *,
    fit_end_frame: int,
    validation_end_frame: int,
    frame_dt_s: float,
    target_config: ForceTargetBuildConfig,
    activity_speed_mps: float,
    gravity_mps2: Sequence[float] = (0.0, 0.0, -9.81),
    prior_reliability: np.ndarray | None = None,
    support_prior: np.ndarray | None = None,
) -> EquivariantForceEpisode:
    """Build one causal inverse-dynamics episode from released source data."""

    data = _load_pickle(final_data_path)
    optimal = _load_pickle(optimal_params_path)
    released = np.asarray(
        _load_pickle(baseline_trajectory_path),
        dtype=float,
    )
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
    controllers = np.asarray(data["controller_points"], dtype=float)
    surface = np.asarray(data["surface_points"], dtype=float)
    interior = np.asarray(data["interior_points"], dtype=float)
    if not 3 <= fit_end_frame < validation_end_frame <= len(observed):
        raise ValueError("case split must leave a held-out source suffix")
    if released.shape[0] < validation_end_frame:
        raise ValueError("released trajectory does not cover the source interval")
    if controllers.shape[0] < validation_end_frame:
        raise ValueError("controller trajectory does not cover the source interval")

    original_count = observed.shape[1]
    structure = np.concatenate((observed[0], surface, interior), axis=0)
    graph = build_phystwin_spring_graph(
        structure,
        controllers[0],
        config=PhysTwinSpringGraphConfig(
            object_radius=float(optimal["object_radius"]),
            object_max_neighbours=int(optimal["object_max_neighbours"]),
            controller_radius=float(optimal["controller_radius"]),
            controller_max_neighbours=int(
                optimal["controller_max_neighbours"]
            ),
        ),
    )
    object_count = graph.num_object_points
    if object_count is None:
        raise RuntimeError("released graph does not declare its object boundary")
    if released.shape[1:] != (object_count, 3):
        raise ValueError("released state and reconstructed graph disagree")
    positions = released[:validation_end_frame]
    velocities = _backward_velocity(positions, frame_dt_s)
    valid = _target_validity(visible, motion_valid)[:validation_end_frame]
    if prior_reliability is None:
        reliability = valid.astype(float)
    else:
        reliability = np.asarray(prior_reliability, dtype=float)
        if reliability.shape[0] < validation_end_frame:
            raise ValueError("prior reliability does not cover the source interval")
        reliability = reliability[:validation_end_frame]
        if reliability.shape != valid.shape:
            raise ValueError("prior reliability must match released tracks")
        reliability = reliability * valid

    acceleration = estimate_residual_acceleration(
        observed[:validation_end_frame],
        positions[:, :original_count],
        valid,
        frame_dt_s=frame_dt_s,
        end_frame=validation_end_frame,
        prior_reliability=reliability,
        window_radius=target_config.window_radius,
        huber_delta_m=target_config.huber_delta_m,
        robust_iterations=target_config.robust_iterations,
        variance_floor_m2ps4=target_config.variance_floor_m2ps4,
        causal_window=True,
    )
    object_edges = graph.springs[: graph.num_object_springs]
    laplacian = normalized_spring_laplacian(object_count, object_edges)
    acceleration = graph_smooth_residual_acceleration(
        acceleration,
        laplacian,
        prior_strength=target_config.graph_prior_strength,
        ridge=target_config.graph_ridge,
        covariance_probes=target_config.graph_covariance_probes,
    )
    force_scale = robust_prefix_force_scale(
        acceleration,
        graph.masses[:object_count],
        prefix_end_frame=fit_end_frame,
        node_quantile=target_config.force_scale_node_quantile,
        temporal_quantile=target_config.force_scale_temporal_quantile,
        minimum_scale_sim=target_config.minimum_force_scale_sim,
        maximum_scale_sim=target_config.maximum_force_scale_sim,
    )
    targets = acceleration_to_force_targets(
        acceleration,
        graph.masses[:object_count],
        maximum_force_sim=force_scale.value_sim,
        minimum_training_weight=target_config.minimum_training_weight,
    )

    attachment, attachment_support = controller_attachment_matrix(
        graph.springs,
        num_object_nodes=object_count,
        num_control_nodes=controllers.shape[1],
    )
    prior = (
        np.zeros(object_count, dtype=float)
        if support_prior is None
        else np.asarray(support_prior, dtype=float)
    )
    if prior.shape != (object_count,) or np.any(
        (prior < 0.0) | (prior > 1.0)
    ):
        raise ValueError("support_prior must be an object-node vector in [0,1]")
    controls = np.zeros_like(positions)
    control_velocity = np.zeros_like(positions)
    action_support = np.zeros((validation_end_frame, object_count), dtype=float)
    external_support = np.zeros_like(action_support)
    activity = np.zeros(validation_end_frame, dtype=float)
    for frame in range(validation_end_frame):
        previous = max(frame - 1, 0)
        conditioning = controller_conditioning_fields(
            positions[frame],
            controllers[previous],
            controllers[frame],
            attachment,
            frame_dt_s=frame_dt_s,
            support_prior=np.maximum(prior, attachment_support),
            activity_speed_mps=activity_speed_mps,
        )
        controls[frame] = conditioning["control_displacement_m"]
        control_velocity[frame] = conditioning["control_velocity_mps"]
        action_support[frame] = conditioning["action_support"]
        external_support[frame] = conditioning["external_support"]
        activity[frame] = conditioning["action_activity"]

    source_checksums = {
        "baseline_trajectory": _sha256(baseline_trajectory_path),
        "final_data": _sha256(final_data_path),
        "optimal_params": _sha256(optimal_params_path),
    }
    diagnostics = {
        "target": targets.diagnostics,
        "force_scale": force_scale.diagnostics,
        "target_config": asdict(target_config),
        "object_node_count": object_count,
        "object_spring_count": graph.num_object_springs,
        "controller_spring_count": (
            len(graph.springs) - graph.num_object_springs
        ),
        "prior_reliability_source": (
            "released_validity" if prior_reliability is None else "supplied_cues"
        ),
        "support_prior_source": (
            "controller_attachment_only"
            if support_prior is None
            else "supplied_geometry"
        ),
        "mass_unit_contract": (
            "released_unit_simulation_masses_not_kilograms"
        ),
        "unique_object_masses_sim": np.unique(
            graph.masses[:object_count]
        ).astype(float).tolist(),
    }
    return EquivariantForceEpisode(
        case_id=case_id,
        positions_m=positions,
        velocities_mps=velocities,
        rest_positions_m=graph.vertices[:object_count],
        object_edges=object_edges,
        rest_lengths_m=graph.rest_lengths[: graph.num_object_springs],
        control_displacement_m=controls,
        control_velocity_mps=control_velocity,
        action_support=action_support,
        external_support=external_support,
        gravity_mps2=np.asarray(gravity_mps2, dtype=float),
        action_activity=activity,
        regime_probabilities=_regime_probabilities(activity),
        force_targets_sim=targets.mean_sim,
        force_target_variance_sim2=targets.variance_sim2,
        force_target_weight=targets.training_weight,
        force_scale_sim=force_scale.value_sim,
        fit_end_frame=fit_end_frame,
        validation_end_frame=validation_end_frame,
        frame_dt_s=frame_dt_s,
        source_checksums=source_checksums,
        information_boundary={
            "target_future_used_for_episode_construction": False,
            "force_targets_use_state_innovation_once": True,
            "prior_reliability_uses_state_residual": False,
            "force_targets_are_causal_per_frame": True,
            "force_scale_uses_prefix_only": True,
            "force_values_are_claimed_as_newtons": False,
            "target_cohort_accessed": False,
        },
        diagnostics=diagnostics,
    )


def load_equivariant_force_source_protocol(
    path: str | Path,
    *,
    device: str | None = None,
) -> EquivariantForceSourceProtocol:
    """Validate a locked source protocol without resolving any target path."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("unsupported equivariant-force source schema")
    if payload.get("contract") != EQUIVARIANT_FORCE_SOURCE_CONTRACT:
        raise ValueError("unsupported equivariant-force source contract")
    source = payload.get("source_cases")
    target = payload.get("target_cases")
    folds = payload.get("source_folds")
    if not isinstance(source, list) or len(source) < 3:
        raise ValueError("source protocol requires at least three cases")
    if len(set(source)) != len(source):
        raise ValueError("source cases must be unique")
    if not isinstance(target, list) or set(source) & set(target):
        raise ValueError("source and target case names must be disjoint")
    if not isinstance(folds, list) or len(folds) < 2:
        raise ValueError("source protocol requires at least two folds")
    held = []
    for fold in folds:
        if not isinstance(fold, Mapping) or not isinstance(
            fold.get("held_out_cases"),
            list,
        ):
            raise ValueError("every source fold must declare held_out_cases")
        held.extend(str(case) for case in fold["held_out_cases"])
    if len(held) != len(set(held)) or set(held) != set(source):
        raise ValueError("source folds must provide disjoint complete coverage")

    raw_model = payload.get("model")
    raw_training = payload.get("training")
    raw_target = payload.get("force_target")
    if not all(
        isinstance(value, Mapping)
        for value in (raw_model, raw_training, raw_target)
    ):
        raise ValueError("source protocol omits typed model settings")
    model = EquivariantForceConfig(**dict(raw_model))
    training_values = dict(raw_training)
    training_values["seeds"] = tuple(training_values["seeds"])
    if device is not None:
        training_values["device"] = device
    training = EquivariantForceTrainingConfig(**training_values)
    target_config = ForceTargetBuildConfig(**dict(raw_target))
    gate = payload.get("source_gate")
    warp = payload.get("official_warp")
    target_qa = payload.get("source_target_qa")
    if not all(
        isinstance(value, Mapping)
        for value in (gate, warp, target_qa)
    ):
        raise ValueError("source protocol omits promotion settings")
    if warp.get("required_zero_force_bitwise_parity") is not True:
        raise ValueError("source protocol must require exact zero-force parity")
    if gate.get("target_access_on_failure") is not False:
        raise ValueError("source protocol must keep targets closed on failure")
    if target_qa.get("failure_blocks_stage_1") is not True:
        raise ValueError("source target QA must block Stage 1 on failure")
    if not 0.0 <= float(
        target_qa.get("maximum_prefix_cap_fraction", -1.0)
    ) <= 1.0:
        raise ValueError("source target QA cap threshold is invalid")
    return EquivariantForceSourceProtocol(
        payload=payload,
        model=model,
        training=training,
        target=target_config,
    )


def build_equivariant_force_source_episodes(
    data_root: str | Path,
    protocol_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build source episodes only; target names are never resolved."""

    protocol = load_equivariant_force_source_protocol(protocol_path)
    root = Path(data_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fit_fraction = float(protocol.payload.get("fit_fraction", 0.75))
    warp = protocol.payload["official_warp"]
    frame_dt_s = float(warp["dt"]) * int(warp["num_substeps"])
    records = {}
    qa_records = {}
    target_qa = protocol.payload["source_target_qa"]
    for case in protocol.payload["source_cases"]:
        case_root = root / str(case)
        split = json.loads(
            (case_root / "split.json").read_text(encoding="utf-8")
        )
        train_start, train_end = (int(value) for value in split["train"])
        if train_start != 0:
            raise ValueError(f"{case}: expected a zero-based training split")
        fit_end = int(train_end * fit_fraction)
        episode = build_released_equivariant_force_episode(
            str(case),
            case_root / "final_data.pkl",
            case_root / "inference.pkl",
            case_root / "optimal_params.pkl",
            fit_end_frame=fit_end,
            validation_end_frame=train_end,
            frame_dt_s=frame_dt_s,
            target_config=protocol.target,
            activity_speed_mps=float(warp["activity_speed_mps"]),
        )
        records[str(case)] = write_equivariant_force_episode(
            output / str(case) / "force_episode",
            episode,
        )
        unique_masses = episode.diagnostics["unique_object_masses_sim"]
        cap_fraction = float(
            episode.diagnostics["force_scale"]["prefix_cap_fraction"]
        )
        unit_contract = episode.diagnostics["target"][
            "force_unit_contract"
        ]
        mass_passed = (
            not bool(target_qa["require_released_unit_simulation_masses"])
            or unique_masses == [1.0]
        )
        unit_passed = (
            unit_contract == target_qa["required_force_unit_contract"]
        )
        cap_passed = cap_fraction <= float(
            target_qa["maximum_prefix_cap_fraction"]
        )
        qa_records[str(case)] = {
            "prefix_cap_fraction": cap_fraction,
            "force_scale_sim": episode.force_scale_sim,
            "unique_object_masses_sim": unique_masses,
            "unit_contract": unit_contract,
            "mass_contract_passed": mass_passed,
            "unit_contract_passed": unit_passed,
            "prefix_cap_fraction_passed": cap_passed,
            "passed": mass_passed and unit_passed and cap_passed,
        }
    qa_passed = all(record["passed"] for record in qa_records.values())
    summary = {
        "schema_version": 2,
        "contract": EQUIVARIANT_FORCE_SOURCE_CONTRACT,
        "protocol_sha256": _sha256(protocol_path),
        "source_episode_count": len(records),
        "source_episodes": records,
        "source_target_qa": qa_records,
        "source_target_qa_passed": qa_passed,
        "target_artifacts_opened": False,
        "stage": "episode_construction_only",
    }
    summary_path = output / "episode_build_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def run_equivariant_force_source_competence(
    episode_root: str | Path,
    protocol_path: str | Path,
    output_dir: str | Path,
    torch: Any,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    """Run Stage 1 only and leave official-Warp promotion unauthorized."""

    protocol = load_equivariant_force_source_protocol(
        protocol_path,
        device=device,
    )
    root = Path(episode_root).resolve()
    validate_equivariant_force_episode_build(
        root,
        protocol_path,
    )
    episodes = [
        load_equivariant_force_episode(root / str(case) / "force_episode")
        for case in protocol.payload["source_cases"]
    ]
    result = crossfit_equivariant_force_competence(
        episodes,
        protocol.payload["source_folds"],
        output_dir,
        torch,
        model_config=protocol.model,
        training_config=protocol.training,
    )
    result["protocol_sha256"] = _sha256(protocol_path)
    result["target_artifacts_opened"] = False
    result["stage_2_official_warp_required"] = True
    result["official_warp_promotion_authorized"] = False
    record_path = Path(output_dir).resolve() / "source_competence_record.json"
    record_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result["source_competence_record"] = str(record_path)
    result["source_competence_record_sha256"] = _sha256(record_path)
    return result


def validate_equivariant_force_episode_build(
    episode_root: str | Path,
    protocol_path: str | Path,
) -> dict[str, Any]:
    """Verify that target QA passed before source training can start."""

    root = Path(episode_root).resolve()
    build_summary_path = root / "episode_build_summary.json"
    build_summary = json.loads(
        build_summary_path.read_text(encoding="utf-8")
    )
    if build_summary.get("schema_version") != 2:
        raise ValueError("source episode build schema changed")
    if build_summary.get("contract") != EQUIVARIANT_FORCE_SOURCE_CONTRACT:
        raise ValueError("source episode build contract changed")
    if build_summary.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("source episode build used a different protocol")
    if build_summary.get("target_artifacts_opened") is not False:
        raise ValueError("source episode build crossed the target boundary")
    if build_summary.get("source_target_qa_passed") is not True:
        raise ValueError("source target QA failed; Stage 1 is blocked")
    if build_summary.get("source_episode_count") != len(
        build_summary.get("source_episodes", {})
    ):
        raise ValueError("source episode build count is inconsistent")
    return build_summary
