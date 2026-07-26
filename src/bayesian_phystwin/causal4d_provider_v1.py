"""Stable Bayesian-PhysTwin compatibility surface for Causal4D.

The module owns the dependency on implementation-private PhysTwin helpers.  Downstream
projects should import artifacts and execution primitives from here rather than from
underscore-prefixed implementation modules.
"""

from __future__ import annotations

import gc
import json
import os
from importlib import import_module
from importlib.metadata import (
    PackageNotFoundError,
    distribution,
    version as distribution_version,
)
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

CAUSAL4D_PROVIDER_API_VERSION = 1
CAUSAL4D_PROVIDER_PACKAGE_VERSION = "0.4.0"
CAUSAL4D_PROVIDER_CAPABILITIES = (
    "artifact_checksums",
    "diagnostic_compatibility",
    "particle_endpoint_position",
    "particle_endpoint_velocity",
    "physical_parameter_particles",
    "phystwin_replay",
    "residual_lifting",
    "target_validity",
)
CAUSAL4D_ARTIFACT_SCHEMA_VERSIONS = {
    "GraphBelief": 1,
    "TwinBelief": 1,
}


def _installed_provider_version() -> str:
    try:
        return distribution_version("bayesian-phystwin")
    except PackageNotFoundError:
        return CAUSAL4D_PROVIDER_PACKAGE_VERSION


def _installed_provider_revision() -> str | None:
    try:
        direct_url = distribution("bayesian-phystwin").read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    if not direct_url:
        return None
    try:
        payload = json.loads(direct_url)
    except (TypeError, json.JSONDecodeError):
        return None
    commit_id = payload.get("vcs_info", {}).get("commit_id")
    return str(commit_id) if commit_id else None


def causal4d_provider_manifest(
    *,
    provider_revision: str | None = None,
) -> dict[str, object]:
    """Return the versioned provider descriptor consumed by Causal4D.

    ``provider_revision`` should be the exact Git revision for frozen experiments.
    Normal editable or package-index installs may omit it; their compatibility is
    decided from ``provider_version`` and the API/schema versions instead.
    """

    revision = (
        provider_revision
        or os.environ.get("BAYESIAN_PHYSTWIN_REVISION")
        or _installed_provider_revision()
        or "unversioned-install"
    )
    return {
        "provider_name": "bayesian-phystwin",
        "provider_version": _installed_provider_version(),
        "provider_revision": revision,
        "schema_version": CAUSAL4D_PROVIDER_API_VERSION,
        "capabilities": list(CAUSAL4D_PROVIDER_CAPABILITIES),
        "artifact_schema_versions": dict(CAUSAL4D_ARTIFACT_SCHEMA_VERSIONS),
        "metadata": {
            "provider_api": "bayesian_phystwin.causal4d_provider_v1",
            "provider_api_version": CAUSAL4D_PROVIDER_API_VERSION,
        },
    }


@runtime_checkable
class PhysTwinReplayProvider(Protocol):
    """Execution boundary required by Causal4D's official PhysTwin backend."""

    @property
    def device(self) -> str:
        """Torch device used by the provider."""

    def set_group_log_scales(self, values: np.ndarray) -> None:
        """Set the grouped object/controller spring log-scales."""

    def set_controller_points(self, values: np.ndarray) -> None:
        """Replace the controller trajectory used by subsequent replays."""

    def replay_initial(self, *, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
        """Replay from the released initial state and return position/velocity histories."""

    def replay_restart(
        self,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        *,
        start_frame: int,
        stop_frame: int,
    ) -> np.ndarray:
        """Replay from an explicit endpoint state and return future positions."""

    def close(self) -> None:
        """Release simulator and accelerator resources."""


class OfficialPhysTwinReplayProvider:
    """Adapter around the released Warp simulator implementing the public protocol."""

    def __init__(self, simulator: Any, torch: Any, wp: Any, *, device: str) -> None:
        self._simulator = simulator
        self._torch = torch
        self._wp = wp
        self._device = str(device)
        self._closed = False

    @property
    def device(self) -> str:
        return self._device

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("PhysTwin replay provider is closed")

    def set_group_log_scales(self, values: np.ndarray) -> None:
        self._require_open()
        array = np.asarray(values, dtype=np.float32)
        target = self._simulator.group_log_scale_tensor
        expected = tuple(int(value) for value in target.shape)
        if array.shape != expected:
            raise ValueError(f"group log-scales must have shape {expected}")
        if not np.all(np.isfinite(array)):
            raise ValueError("group log-scales must be finite")
        with self._torch.no_grad():
            target.copy_(
                self._torch.as_tensor(
                    array,
                    dtype=self._torch.float32,
                    device=self._device,
                )
            )
        self._wp.synchronize()

    def set_controller_points(self, values: np.ndarray) -> None:
        self._require_open()
        array = np.asarray(values, dtype=np.float32)
        current = self._simulator.controller_points
        expected = tuple(int(value) for value in current.shape)
        if array.shape != expected:
            raise ValueError(f"controller points must have shape {expected}")
        if not np.all(np.isfinite(array)):
            raise ValueError("controller points must be finite")
        self._simulator.controller_points = self._torch.as_tensor(
            array,
            dtype=self._torch.float32,
            device=self._device,
        ).contiguous()
        self._wp.synchronize()

    def replay_initial(self, *, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
        self._require_open()
        if frame_count < 1:
            raise ValueError("frame_count must be positive")
        positions, velocities = _rollout_initial(
            self._simulator,
            self._wp,
            frame_count=frame_count,
        )
        return np.asarray(positions), np.asarray(velocities)

    def replay_restart(
        self,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        *,
        start_frame: int,
        stop_frame: int,
    ) -> np.ndarray:
        self._require_open()
        position = np.asarray(position_m, dtype=np.float32)
        velocity = np.asarray(velocity_mps, dtype=np.float32)
        if position.ndim != 2 or position.shape[1] != 3 or velocity.shape != position.shape:
            raise ValueError("restart position and velocity must have shape (N, 3)")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError("restart state must be finite")
        if not 0 <= start_frame < stop_frame:
            raise ValueError("restart frame interval must be nonempty")
        return np.asarray(
            rollout_restart(
                self._simulator,
                self._torch,
                self._wp,
                position,
                velocity,
                start_frame=start_frame,
                stop_frame=stop_frame,
                device=self._device,
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._simulator = None
        gc.collect()
        cuda = getattr(self._torch, "cuda", None)
        if cuda is not None and hasattr(cuda, "empty_cache"):
            cuda.empty_cache()

    def __enter__(self) -> "OfficialPhysTwinReplayProvider":
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def create_official_replay_provider(
    official_repo: str | Path,
    data: Mapping[str, object],
    optimal: Mapping[str, object],
    checkpoint_path: str | Path,
    graph: Any,
    *,
    num_surface_points: int,
    original_count: int,
    dt: float,
    num_substeps: int,
    self_collision: bool,
    deterministic_spring_forces: bool = False,
    spring_parameterization: str = "dense",
    device: str,
) -> OfficialPhysTwinReplayProvider:
    """Construct the official replay adapter without exposing simulator internals."""

    simulator, torch, wp, _ = initialize_simulator(
        official_repo,
        dict(data),
        dict(optimal),
        checkpoint_path,
        graph,
        num_surface_points=num_surface_points,
        original_count=original_count,
        dt=dt,
        num_substeps=num_substeps,
        self_collision=self_collision,
        deterministic_spring_forces=deterministic_spring_forces,
        spring_parameterization=spring_parameterization,
        device=device,
    )
    return OfficialPhysTwinReplayProvider(simulator, torch, wp, device=device)


def _delegate(module: str, name: str, *args: Any, **kwargs: Any) -> Any:
    function = getattr(import_module(f"bayesian_phystwin.{module}"), name)
    return function(*args, **kwargs)


# Artifact and geometry primitives.
def load_pickle(path: str | Path) -> Any:
    return _delegate("phystwin_residual_dynamics", "_load_pickle", path)


def sha256_file(path: str | Path) -> str:
    return str(_delegate("phystwin_residual_dynamics", "_sha256", path))


def target_validity(visible: np.ndarray, motion_valid: np.ndarray) -> np.ndarray:
    return np.asarray(
        _delegate("phystwin_residual_dynamics", "_target_validity", visible, motion_valid),
        dtype=bool,
    )


def build_lift_map(
    initial_vertices: np.ndarray,
    original_count: int,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices, weights = _delegate(
        "phystwin_residual_dynamics",
        "_lift_map",
        initial_vertices,
        original_count,
        neighbors,
    )
    return np.asarray(indices, dtype=np.int64), np.asarray(weights, dtype=float)


def lift_residual(
    tracked_residual: np.ndarray,
    state_count: int,
    indices: np.ndarray,
    weights: np.ndarray,
    *,
    maximum_norm: float,
) -> np.ndarray:
    return np.asarray(
        _delegate(
            "phystwin_residual_dynamics",
            "_lift_residual",
            tracked_residual,
            state_count,
            indices,
            weights,
            maximum_norm=maximum_norm,
        )
    )


# Publicly named compatibility operations for advanced Causal4D diagnostics.
def chamfer_by_frame(*args: Any, **kwargs: Any) -> Any:
    return _delegate("phystwin_additional_confirmation", "_chamfer_by_frame", *args, **kwargs)


def lock_protocol(*args: Any, **kwargs: Any) -> Any:
    return _delegate("phystwin_confirmatory", "_lock_protocol", *args, **kwargs)


def git_commit(*args: Any, **kwargs: Any) -> Any:
    return _delegate("phystwin_state_injection", "_git_commit", *args, **kwargs)


def initialize_simulator(*args: Any, **kwargs: Any) -> Any:
    return _delegate("phystwin_state_injection", "_initialize_simulator", *args, **kwargs)


def metric_summary(*args: Any, **kwargs: Any) -> Any:
    return _delegate("phystwin_state_injection", "_metric_summary", *args, **kwargs)


def released_self_collision_for_case(*args: Any, **kwargs: Any) -> Any:
    return _delegate(
        "phystwin_state_injection",
        "_released_self_collision_for_case",
        *args,
        **kwargs,
    )


def _rollout_initial(*args: Any, **kwargs: Any) -> Any:
    return _delegate("phystwin_state_injection", "_rollout_initial", *args, **kwargs)


def rollout_restart(*args: Any, **kwargs: Any) -> Any:
    return _delegate("phystwin_state_injection", "_rollout_restart", *args, **kwargs)


def simulator_runtime(*args: Any, **kwargs: Any) -> Any:
    return _delegate("phystwin_state_injection", "_simulator_runtime", *args, **kwargs)


def state_numpy(*args: Any, **kwargs: Any) -> Any:
    return _delegate("phystwin_state_injection", "_state_numpy", *args, **kwargs)


def load_official_spring_mass_module(*args: Any, **kwargs: Any) -> Any:
    return _delegate(
        "_phystwin_warp_backend",
        "load_official_spring_mass_module",
        *args,
        **kwargs,
    )


def make_reliability_simulator_class(*args: Any, **kwargs: Any) -> Any:
    return _delegate(
        "_phystwin_warp_backend",
        "make_reliability_simulator_class",
        *args,
        **kwargs,
    )


def measurement_target_audit(*args: Any, **kwargs: Any) -> Any:
    return _delegate(
        "deform360_selective_virtual_sensing_evaluation",
        "_measurement_target_audit",
        *args,
        **kwargs,
    )


def attachment_support_nodes(*args: Any, **kwargs: Any) -> Any:
    return _delegate(
        "phystwin_structural_diagnostic", "_attachment_support_nodes", *args, **kwargs
    )


def far_graph_observation_error(*args: Any, **kwargs: Any) -> Any:
    return _delegate(
        "phystwin_structural_diagnostic",
        "_far_graph_observation_error",
        *args,
        **kwargs,
    )


def graph_distance(*args: Any, **kwargs: Any) -> Any:
    return _delegate("phystwin_structural_diagnostic", "_graph_distance", *args, **kwargs)


def horizon_summary(*args: Any, **kwargs: Any) -> Any:
    return _delegate("phystwin_structural_diagnostic", "_horizon_summary", *args, **kwargs)


def object_rest_lengths(*args: Any, **kwargs: Any) -> Any:
    return _delegate(
        "phystwin_structural_diagnostic", "_object_rest_lengths", *args, **kwargs
    )


def set_simulator_arrays(*args: Any, **kwargs: Any) -> Any:
    return _delegate(
        "phystwin_structural_diagnostic", "_set_simulator_arrays", *args, **kwargs
    )


__all__ = [
    "CAUSAL4D_ARTIFACT_SCHEMA_VERSIONS",
    "CAUSAL4D_PROVIDER_API_VERSION",
    "CAUSAL4D_PROVIDER_CAPABILITIES",
    "CAUSAL4D_PROVIDER_PACKAGE_VERSION",
    "OfficialPhysTwinReplayProvider",
    "PhysTwinReplayProvider",
    "attachment_support_nodes",
    "build_lift_map",
    "causal4d_provider_manifest",
    "chamfer_by_frame",
    "create_official_replay_provider",
    "far_graph_observation_error",
    "git_commit",
    "graph_distance",
    "horizon_summary",
    "initialize_simulator",
    "lift_residual",
    "load_official_spring_mass_module",
    "load_pickle",
    "lock_protocol",
    "make_reliability_simulator_class",
    "measurement_target_audit",
    "metric_summary",
    "object_rest_lengths",
    "released_self_collision_for_case",
    "rollout_restart",
    "set_simulator_arrays",
    "sha256_file",
    "simulator_runtime",
    "state_numpy",
    "target_validity",
]
