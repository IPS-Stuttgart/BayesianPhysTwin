#!/usr/bin/env python3
"""Run and score a sealed sparse-identity PhysTwin state-update source smoke."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from bayesian_phystwin.phystwin_comparison import (  # noqa: E402
    official_metrics_by_frame,
)
from bayesian_phystwin.phystwin_graph import (  # noqa: E402
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)
from bayesian_phystwin.phystwin_sparse_identity_split import (  # noqa: E402
    split_sparse_identity_tracks,
)
from bayesian_phystwin.phystwin_sparse_state_update import (  # noqa: E402
    closure_gate_passed,
    fixed_identity_node_association,
    low_frequency_scalar_graph_basis,
    nonlinear_closure_diagnostics,
    prefix_persistence_correction,
)
from bayesian_phystwin.phystwin_state_injection import (  # noqa: E402
    _initialize_simulator,
    _released_self_collision_for_case,
    _rollout_initial,
    _rollout_restart,
    _trajectory_error,
)
from bayesian_phystwin.propagated_state_belief import (  # noqa: E402
    PropagatedStateBeliefConfig,
)
from bayesian_phystwin.propagated_state_correction import (  # noqa: E402
    PropagatedStateCorrection,
    PropagatedStateSelectionConfig,
    modal_state_parameter_fields,
    scale_posterior_covariance_for_state_limits,
    select_propagated_state_update,
    write_propagated_state_correction,
)


_NUMPY_PICKLE_MODULE_ALIASES = {
    "numpy._core.multiarray": "numpy.core.multiarray",
    "numpy._core.numeric": "numpy.core.numeric",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _load_pickle(path: str | Path) -> Any:
    class NumpyCompatibilityUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            compatible = _NUMPY_PICKLE_MODULE_ALIASES.get(module)
            if compatible is not None:
                try:
                    importlib.import_module(module)
                except ModuleNotFoundError:
                    module = compatible
            return super().find_class(module, name)

    with Path(path).open("rb") as handle:
        return NumpyCompatibilityUnpickler(handle).load()


def _git_commit(path: str | Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    _require(
        protocol.get("protocol_id")
        == "phystwin-prior-aware-sparse-identity-source-v1",
        "unexpected protocol id",
    )
    _require(int(protocol.get("schema_version", -1)) == 1, "unsupported protocol")
    return protocol


def _input_paths(protocol: dict[str, Any]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name, record in protocol["inputs"].items():
        path = Path(record["path"]).resolve()
        _require(path.is_file(), f"missing input {name}: {path}")
        _require(_sha256(path) == record["sha256"], f"input hash changed: {name}")
        result[name] = path
    return result


def _verify_implementation(protocol: dict[str, Any]) -> None:
    implementation = protocol["implementation"]
    runner = Path(__file__).resolve()
    _require(_sha256(runner) == implementation["runner_sha256"], "runner changed")
    module = REPO_ROOT / "src" / "bayesian_phystwin" / "phystwin_sparse_state_update.py"
    _require(
        _sha256(module) == implementation["state_update_module_sha256"],
        "state-update module changed",
    )


def _verify_official_repo(protocol: dict[str, Any]) -> Path:
    record = protocol["official_simulator"]
    repository = Path(record["repository"]).resolve()
    source = repository / "qqtt" / "model" / "diff_simulator" / "spring_mass_warp.py"
    _require(source.is_file(), "official simulator source is missing")
    _require(_git_commit(repository) == record["commit"], "official simulator commit changed")
    _require(_sha256(source) == record["source_sha256"], "official simulator changed")
    return repository


def _sanitize_simulator_observations(
    data: dict[str, Any],
    *,
    future_start_frame: int,
) -> dict[str, Any]:
    """Remove forecast object observations while retaining the given action."""

    sanitized = copy.copy(data)
    points = np.asarray(data["object_points"]).copy()
    visible = np.asarray(data["object_visibilities"]).copy()
    motion_valid = np.asarray(data["object_motions_valid"]).copy()
    points[future_start_frame:] = points[future_start_frame - 1]
    visible[future_start_frame:] = False
    motion_valid[future_start_frame:] = False
    sanitized["object_points"] = points
    sanitized["object_visibilities"] = visible
    sanitized["object_motions_valid"] = motion_valid
    return sanitized


def _response_tensor(
    simulator: Any,
    torch: Any,
    wp: Any,
    nominal_positions: np.ndarray,
    nominal_velocities: np.ndarray,
    position_fields: np.ndarray,
    velocity_fields: np.ndarray,
    *,
    prefix_start: int,
    prefix_stop: int,
    device: str,
) -> np.ndarray:
    frame_count = prefix_stop - prefix_start
    parameter_count = position_fields.shape[2] + velocity_fields.shape[2]
    response = np.empty(
        (
            frame_count,
            nominal_positions.shape[1],
            3,
            parameter_count,
        ),
        dtype=np.float32,
    )
    zero_position = np.zeros_like(position_fields[:, :, 0])
    zero_velocity = np.zeros_like(velocity_fields[:, :, 0])
    for parameter in range(parameter_count):
        position_delta = (
            position_fields[:, :, parameter]
            if parameter < position_fields.shape[2]
            else zero_position
        )
        velocity_delta = (
            velocity_fields[:, :, parameter - position_fields.shape[2]]
            if parameter >= position_fields.shape[2]
            else zero_velocity
        )
        initial_position = nominal_positions[prefix_start] + position_delta
        initial_velocity = nominal_velocities[prefix_start] + velocity_delta
        continuation = _rollout_restart(
            simulator,
            torch,
            wp,
            initial_position,
            initial_velocity,
            start_frame=prefix_start + 1,
            stop_frame=prefix_stop,
            device=device,
        )
        perturbed = np.concatenate((initial_position[None], continuation), axis=0)
        response[..., parameter] = (
            perturbed - nominal_positions[prefix_start:prefix_stop]
        )
    return response


def _posterior_covariance(
    selection: Any,
    *,
    graph_rank: int,
) -> np.ndarray:
    if not selection.accepted:
        return np.zeros((9 * graph_rank, 9 * graph_rank), dtype=np.float64)
    state_limits = selection.diagnostics["full_state_limits"]
    bias_limit = selection.diagnostics["full_shared_bias_limit"]
    return scale_posterior_covariance_for_state_limits(
        selection.full_belief.posterior_covariance,
        graph_rank=graph_rank,
        position_scale=float(state_limits["position"]["radial_scale"]),
        velocity_scale=float(state_limits["velocity"]["radial_scale"]),
        shared_bias_scale=float(bias_limit["radial_scale"]),
    )


def _predict(protocol_path: Path, output: Path) -> None:
    protocol = _load_protocol(protocol_path)
    _verify_implementation(protocol)
    official_repo = _verify_official_repo(protocol)
    paths = _input_paths(protocol)
    _require(not output.exists(), "prediction output already exists")
    output.mkdir(parents=True)

    case_id = str(protocol["case_id"])
    settings = protocol["state_update"]
    simulator_settings = protocol["simulator"]
    split = json.loads(paths["split"].read_text(encoding="utf-8"))
    train_end = int(split["train"][1])
    future_end = int(split["test"][1])
    _require(
        train_end == int(protocol["frames"]["future_start_exclusive"]),
        "training endpoint changed",
    )
    _require(
        future_end == int(protocol["frames"]["future_end_exclusive"]),
        "future endpoint changed",
    )
    response_frame_count = int(settings["response_frame_count"])
    prefix_start = train_end - response_frame_count
    _require(prefix_start >= 1, "response prefix starts before frame one")

    data = _load_pickle(paths["final_data"])
    optimal = _load_pickle(paths["optimal_params"])
    baseline = np.asarray(_load_pickle(paths["baseline_trajectory"]), dtype=np.float64)
    baseline = baseline[:future_end]
    tracks_raw = np.asarray(_load_pickle(paths["manual_tracks"]), dtype=np.float64)
    tracks_raw = tracks_raw[:future_end]
    identity_split = split_sparse_identity_tracks(
        tracks_raw,
        observed_count=int(settings["observed_identity_count"]),
        future_start_frame=train_end,
    )
    observation_tracks = identity_split.observation_tracks_m
    observed_ids = identity_split.observed_indices
    del tracks_raw

    object_points = np.asarray(data["object_points"], dtype=np.float64)[:future_end]
    controller_points = np.asarray(data["controller_points"], dtype=np.float64)[
        :future_end
    ]
    surface_points = np.asarray(data["surface_points"], dtype=np.float64)
    interior_points = np.asarray(data["interior_points"], dtype=np.float64)
    original_count = object_points.shape[1]
    structure = np.concatenate(
        (object_points[0], surface_points, interior_points),
        axis=0,
    )
    _require(
        baseline.shape == (future_end, len(structure), 3),
        "baseline trajectory shape changed",
    )
    graph = build_phystwin_spring_graph(
        structure,
        controller_points[0],
        config=PhysTwinSpringGraphConfig(
            object_radius=float(optimal["object_radius"]),
            object_max_neighbours=int(optimal["object_max_neighbours"]),
            controller_radius=float(optimal["controller_radius"]),
            controller_max_neighbours=int(optimal["controller_max_neighbours"]),
        ),
    )
    object_springs = graph.springs[: graph.num_object_springs]
    graph_basis, graph_eigenvalues, graph_diagnostics = (
        low_frequency_scalar_graph_basis(
            len(structure),
            object_springs,
            rank=int(settings["graph_rank"]),
        )
    )
    association_nodes, association_distance, association_diagnostics = (
        fixed_identity_node_association(
            baseline[0],
            observation_tracks[0],
            observed_ids,
        )
    )
    _require(
        float(np.max(association_distance, initial=0.0))
        <= float(settings["maximum_initial_association_distance_m"]),
        "frame-zero identity association exceeds the lock",
    )

    simulator_data = _sanitize_simulator_observations(
        data,
        future_start_frame=train_end,
    )
    simulator, torch, wp, _ = _initialize_simulator(
        official_repo,
        simulator_data,
        optimal,
        paths["checkpoint"],
        graph,
        num_surface_points=original_count + len(surface_points),
        original_count=original_count,
        dt=float(simulator_settings["dt_s"]),
        num_substeps=int(simulator_settings["num_substeps"]),
        self_collision=_released_self_collision_for_case(case_id),
        deterministic_spring_forces=bool(
            simulator_settings["deterministic_spring_forces"]
        ),
        device=str(simulator_settings["device"]),
    )
    replay_positions, replay_velocities = _rollout_initial(
        simulator,
        wp,
        frame_count=future_end,
    )
    parity = {
        "all_frames": _trajectory_error(baseline, replay_positions),
        "prefix_endpoint": _trajectory_error(
            baseline[train_end - 1 : train_end],
            replay_positions[train_end - 1 : train_end],
        ),
        "future": _trajectory_error(
            baseline[train_end:future_end],
            replay_positions[train_end:future_end],
        ),
    }
    parity["passed"] = bool(
        parity["all_frames"]["vector_rmse_m"]
        <= float(simulator_settings["maximum_replay_vector_rmse_m"])
        and parity["future"]["maximum_norm_m"]
        <= float(simulator_settings["maximum_replay_norm_m"])
    )
    _require(parity["passed"], "official Warp replay parity gate failed")

    position_fields, velocity_fields, position_steps, velocity_steps = (
        modal_state_parameter_fields(
            graph_basis,
            position_step_m=float(settings["position_step_m"]),
            velocity_step_mps=float(settings["velocity_step_mps"]),
        )
    )
    response = _response_tensor(
        simulator,
        torch,
        wp,
        replay_positions,
        replay_velocities,
        position_fields,
        velocity_fields,
        prefix_start=prefix_start,
        prefix_stop=train_end,
        device=str(simulator_settings["device"]),
    )
    observed_response = response[:, association_nodes]
    observed_basis = graph_basis[association_nodes]
    observed_values = observation_tracks[
        prefix_start:train_end,
        observed_ids,
    ]
    available = np.all(np.isfinite(observed_values), axis=2)
    _require(np.all(available), "locked smoke requires all four prefix identities")
    innovation = observed_values - replay_positions[
        prefix_start:train_end,
        association_nodes,
    ]
    observation_variance = np.full(
        available.shape,
        float(settings["observation_std_m"]) ** 2,
        dtype=np.float64,
    )
    selection_config = PropagatedStateSelectionConfig(
        fit_frame_count=int(settings["fit_frame_count"]),
        minimum_validation_improvement_fraction=float(
            settings["minimum_validation_improvement_fraction"]
        ),
        minimum_validation_improvement_m=float(
            settings["minimum_validation_improvement_m"]
        ),
        projection_ridge=float(settings["projection_ridge"]),
        maximum_position_update_m=float(settings["maximum_position_update_m"]),
        maximum_velocity_update_mps=float(
            settings["maximum_velocity_update_mps"]
        ),
        maximum_shared_bias_m=float(settings["maximum_shared_bias_m"]),
    )
    belief_config = PropagatedStateBeliefConfig(
        observation_std_m=float(settings["observation_std_m"]),
        state_weight_prior_std=float(settings["state_weight_prior_std"]),
        shared_bias_prior_std_m=float(settings["shared_bias_prior_std_m"]),
        effective_samples_per_frame=float(settings["effective_samples_per_frame"]),
        effective_frame_count=float(settings["effective_frame_count"]),
        degrees_of_freedom=float(settings["degrees_of_freedom"]),
        minimum_robust_weight=float(settings["minimum_robust_weight"]),
        maximum_iterations=int(settings["maximum_iterations"]),
        convergence_tolerance=float(settings["convergence_tolerance"]),
        maximum_condition_number=float(settings["maximum_condition_number"]),
        ambiguous_subspace_cosine=float(settings["ambiguous_subspace_cosine"]),
        reject_unidentifiable_state=bool(settings["reject_unidentifiable_state"]),
    )
    selection = select_propagated_state_update(
        innovation,
        available,
        observed_response,
        observed_basis,
        graph_basis,
        position_steps,
        velocity_steps,
        prior_reliability=np.ones(available.shape, dtype=np.float64),
        observation_variance_m2=observation_variance,
        belief_config=belief_config,
        selection_config=selection_config,
    )
    persistence_field, persistence_coefficients, persistence_diagnostics = (
        prefix_persistence_correction(
            innovation,
            available,
            observed_basis,
            graph_basis,
            fit_frame_count=response_frame_count,
            ridge=float(settings["projection_ridge"]),
            maximum_node_norm_m=float(settings["maximum_shared_bias_m"]),
        )
    )
    persistence = baseline.copy()
    persistence[train_end:future_end] += persistence_field[None]

    closure = None
    closure_passed = False
    nonlinear = None
    if selection.accepted:
        corrected_position = (
            replay_positions[prefix_start] + selection.position_update_m
        )
        corrected_velocity = (
            replay_velocities[prefix_start] + selection.velocity_update_mps
        )
        continuation = _rollout_restart(
            simulator,
            torch,
            wp,
            corrected_position,
            corrected_velocity,
            start_frame=prefix_start + 1,
            stop_frame=future_end,
            device=str(simulator_settings["device"]),
        )
        nonlinear = np.concatenate((corrected_position[None], continuation), axis=0)
        nonlinear_prefix_displacement = (
            nonlinear[:response_frame_count, association_nodes]
            - replay_positions[prefix_start:train_end, association_nodes]
        )
        closure = nonlinear_closure_diagnostics(
            observed_response,
            selection.state_weights,
            nonlinear_prefix_displacement,
            available,
        )
        closure_passed = closure_gate_passed(
            closure,
            maximum_vector_rmse_m=float(settings["maximum_closure_vector_rmse_m"]),
            maximum_relative_vector_rmse=float(
                settings["maximum_closure_relative_vector_rmse"]
            ),
        )

    accepted = bool(selection.accepted and closure_passed)
    state_only = persistence.copy()
    candidate = persistence.copy()
    if accepted:
        state_only = baseline.copy()
        state_only[train_end:future_end] = nonlinear[
            train_end - prefix_start :
        ]
        candidate = state_only.copy()
        shared_bias_field = (
            graph_basis @ selection.shared_bias_coefficients_m
        )
        candidate[train_end:future_end] += shared_bias_field[None]
    else:
        shared_bias_field = persistence_field.copy()
        _require(
            np.array_equal(candidate, persistence),
            "rejected state update changed persistence fallback bytes",
        )

    state_weights = (
        selection.state_weights
        if accepted
        else np.zeros_like(selection.state_weights)
    )
    shared_bias_coefficients = (
        selection.shared_bias_coefficients_m
        if accepted
        else np.zeros_like(selection.shared_bias_coefficients_m)
    )
    correction = PropagatedStateCorrection(
        case_id=case_id,
        graph_basis=graph_basis,
        graph_eigenvalues=graph_eigenvalues,
        position_coefficient_steps_m=position_steps,
        velocity_coefficient_steps_mps=velocity_steps,
        state_weights=state_weights,
        shared_bias_coefficients_m=shared_bias_coefficients,
        posterior_covariance=(
            _posterior_covariance(selection, graph_rank=graph_basis.shape[1])
            if accepted
            else np.zeros((9 * graph_basis.shape[1],) * 2, dtype=np.float64)
        ),
        accepted_state_update=accepted,
        selection_reason=(
            selection.reason
            if accepted or not selection.accepted
            else "nonlinear-closure-gate"
        ),
        prefix_frame_start=prefix_start,
        fit_frame_stop=prefix_start + int(settings["fit_frame_count"]),
        prefix_frame_stop=train_end,
        information_boundary={
            "released_case_role": "opened_source_development_smoke_only",
            "forecast_frames_used_for_fit_or_selection": False,
            "future_manual_identities_used_for_fit_or_selection": False,
            "future_object_observations_removed_before_simulator_initialization": True,
            "future_controller_action_is_given": True,
            "observed_identity_count": int(len(observed_ids)),
            "hidden_identity_count": int(len(identity_split.hidden_indices)),
        },
        source_checksums={
            name: str(protocol["inputs"][name]["sha256"])
            for name in sorted(protocol["inputs"])
        },
        diagnostics={
            "selection": selection.diagnostics,
            "selection_accepted_before_closure": selection.accepted,
            "nonlinear_closure": closure,
            "nonlinear_closure_passed": closure_passed,
            "graph": graph_diagnostics,
            "identity_association": association_diagnostics,
            "replay_parity": parity,
            "persistence": persistence_diagnostics,
        },
    )
    correction_record = write_propagated_state_correction(
        output / "propagated_state_correction",
        correction,
    )
    prediction_path = output / "prediction.npz"
    _atomic_npz(
        prediction_path,
        baseline_trajectory=baseline,
        persistence_trajectory=persistence,
        state_only_trajectory=state_only,
        candidate_trajectory=candidate,
        observed_identity_indices=observed_ids,
        hidden_identity_indices=identity_split.hidden_indices,
        association_node_indices=association_nodes,
        persistence_coefficients_m=persistence_coefficients,
    )
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "status": "prediction_sealed",
        "accepted_state_update": accepted,
        "selection_reason": correction.selection_reason,
        "prediction_path": prediction_path.name,
        "prediction_sha256": _sha256(prediction_path),
        "prediction_array_sha256": {
            "baseline_trajectory": _array_sha256(baseline),
            "persistence_trajectory": _array_sha256(persistence),
            "state_only_trajectory": _array_sha256(state_only),
            "candidate_trajectory": _array_sha256(candidate),
        },
        "correction_artifact": correction_record,
        "implementation": {
            "repository_commit": _git_commit(REPO_ROOT),
            "runner_sha256": _sha256(Path(__file__).resolve()),
        },
        "information_boundary": correction.information_boundary,
        "diagnostics": correction.diagnostics,
        "outcomes": None,
    }
    manifest_path = output / "prediction_manifest.json"
    _atomic_json(manifest_path, manifest)
    (output / "PREDICTION_COMPLETE").write_text(
        _sha256(manifest_path) + "\n",
        encoding="ascii",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _metric_summary(values: dict[str, np.ndarray]) -> dict[str, Any]:
    frame_count = len(next(iter(values.values())))
    boundaries = np.linspace(0, frame_count, 4, dtype=int)
    result: dict[str, Any] = {}
    for name, frames in values.items():
        frame_values = np.asarray(frames, dtype=np.float64)
        result[name] = {
            "mean_m": float(np.mean(frame_values)),
            "early_mean_m": float(np.mean(frame_values[boundaries[0] : boundaries[1]])),
            "middle_mean_m": float(np.mean(frame_values[boundaries[1] : boundaries[2]])),
            "late_mean_m": float(np.mean(frame_values[boundaries[2] : boundaries[3]])),
            "by_frame_m": frame_values.tolist(),
        }
    return result


def _score(protocol_path: Path, output: Path) -> None:
    protocol = _load_protocol(protocol_path)
    _verify_implementation(protocol)
    paths = _input_paths(protocol)
    manifest_path = output / "prediction_manifest.json"
    marker = output / "PREDICTION_COMPLETE"
    _require(manifest_path.is_file() and marker.is_file(), "prediction is not sealed")
    _require(not (output / "score.json").exists(), "score already exists")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        marker.read_text(encoding="ascii").strip() == _sha256(manifest_path),
        "prediction completion marker changed",
    )
    _require(
        manifest["protocol_sha256"] == _sha256(protocol_path),
        "prediction protocol changed",
    )
    prediction_path = output / manifest["prediction_path"]
    _require(
        _sha256(prediction_path) == manifest["prediction_sha256"],
        "sealed prediction changed",
    )
    with np.load(prediction_path, allow_pickle=False) as archive:
        trajectories = {
            name.removesuffix("_trajectory"): np.asarray(archive[name])
            for name in (
                "baseline_trajectory",
                "persistence_trajectory",
                "state_only_trajectory",
                "candidate_trajectory",
            )
        }
        observed_ids = np.asarray(archive["observed_identity_indices"])
        hidden_ids = np.asarray(archive["hidden_identity_indices"])

    split = json.loads(paths["split"].read_text(encoding="utf-8"))
    train_end = int(split["train"][1])
    future_end = int(split["test"][1])
    tracks = np.asarray(_load_pickle(paths["manual_tracks"]), dtype=np.float64)[
        :future_end
    ]
    identity_split = split_sparse_identity_tracks(
        tracks,
        observed_count=int(protocol["state_update"]["observed_identity_count"]),
        future_start_frame=train_end,
    )
    _require(
        np.array_equal(observed_ids, identity_split.observed_indices)
        and np.array_equal(hidden_ids, identity_split.hidden_indices),
        "sealed identity split changed",
    )
    data = _load_pickle(paths["final_data"])
    observed = np.asarray(data["object_points"], dtype=np.float64)[:future_end]
    visible = np.asarray(data["object_visibilities"], dtype=bool)[:future_end]
    surface_count = len(np.asarray(data["surface_points"]))
    num_surface_points = observed.shape[1] + surface_count
    scores = {
        name: _metric_summary(
            official_metrics_by_frame(
                trajectory,
                observed,
                visible,
                identity_split.scoring_tracks_m,
                num_surface_points=num_surface_points,
                start_frame=train_end,
                end_frame=future_end,
            )
        )
        for name, trajectory in trajectories.items()
    }
    comparisons: dict[str, Any] = {}
    for candidate_name in ("persistence", "state_only", "candidate"):
        comparisons[candidate_name] = {}
        for metric in ("chamfer_distance_m", "track_error_m"):
            baseline_value = scores["baseline"][metric]["mean_m"]
            candidate_value = scores[candidate_name][metric]["mean_m"]
            comparisons[candidate_name][metric] = {
                "baseline_mean_m": baseline_value,
                "candidate_mean_m": candidate_value,
                "difference_m": candidate_value - baseline_value,
                "improvement_fraction": (
                    1.0 - candidate_value / baseline_value
                    if baseline_value > 0.0
                    else 0.0
                ),
            }
    result = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "opened_source_development_result_not_sota_evidence",
        "case_id": protocol["case_id"],
        "prediction_manifest_sha256": _sha256(manifest_path),
        "accepted_state_update": bool(manifest["accepted_state_update"]),
        "selection_reason": manifest["selection_reason"],
        "identity_boundary": {
            "observed_prefix_identity_count": int(len(observed_ids)),
            "hidden_future_identity_count": int(len(hidden_ids)),
            "observed_and_hidden_are_disjoint": True,
        },
        "scores": scores,
        "comparisons_to_unchanged_baseline": comparisons,
        "claim_boundary": (
            "One previously opened source interaction with manual sparse prefix "
            "identities; this is an online-supervised development smoke, not an "
            "open-loop, independent, confirmatory, or SOTA result."
        ),
    }
    score_path = output / "score.json"
    _atomic_json(score_path, result)
    (output / "SCORE_COMPLETE").write_text(
        _sha256(score_path) + "\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("predict", "score"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "predict":
        _predict(args.protocol.resolve(), args.output.resolve())
    else:
        _score(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
