"""Stable Bayesian-PhysTwin compatibility surface for Causal4D.

The module preserves the provider-v1 import path while forwarding owned replay,
geometry, hashing, and metadata operations to stable core modules.  Historical
diagnostic helpers remain isolated here for frozen Causal4D consumers.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from .contracts.provider import (
    installed_distribution_revision,
    installed_distribution_version,
)
from .contracts.replay import PhysTwinReplayProviderV1 as PhysTwinReplayProvider
from .phystwin.artifacts import sha256_file
from .phystwin.geometry import build_lift_map, lift_residual, target_validity
from .phystwin.replay import (
    _rollout_initial_trajectory,
    _rollout_restart_trajectory,
)
from .phystwin.replay import (
    _state_numpy as _owned_state_numpy,
)

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
        or installed_distribution_revision("bayesian-phystwin")
        or "unversioned-install"
    )
    return {
        "provider_name": "bayesian-phystwin",
        "provider_version": installed_distribution_version(
            "bayesian-phystwin",
            fallback=CAUSAL4D_PROVIDER_PACKAGE_VERSION,
        ),
        "provider_revision": revision,
        "schema_version": CAUSAL4D_PROVIDER_API_VERSION,
        "capabilities": list(CAUSAL4D_PROVIDER_CAPABILITIES),
        "artifact_schema_versions": dict(CAUSAL4D_ARTIFACT_SCHEMA_VERSIONS),
        "metadata": {
            "provider_api": "bayesian_phystwin.causal4d_provider_v1",
            "provider_api_version": CAUSAL4D_PROVIDER_API_VERSION,
        },
    }


class OfficialPhysTwinReplayProvider:
    """Adapter around the released Warp simulator implementing provider API v1."""

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
            raise ValueError(