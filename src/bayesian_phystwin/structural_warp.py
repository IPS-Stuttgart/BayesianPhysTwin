"""Opt-in structural configuration boundary for official PhysTwin Warp runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .structural_artifact import (
    StructuralTwinCorrection,
    corrected_rest_geometry,
    structural_displacement,
)


FROZEN_CAUSAL4D_TAG = "v0.3.0-causal4d-aip"
FROZEN_CAUSAL4D_COMMIT = "2f4652657cad6c6a6a7cba76a0b2afe1f0cd37a8"


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class StructuralWarpConfiguration:
    """Exact arrays applied to Warp before one structural rollout."""

    structural_artifact_id: str
    session_id: str
    corrected_rest_positions_m: np.ndarray
    corrected_rest_lengths_m: np.ndarray
    initial_position_m: np.ndarray
    initial_velocity_mps: np.ndarray
    controller_points_m: np.ndarray
    gravity_mps2: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        positions = np.asarray(self.corrected_rest_positions_m).copy()
        lengths = np.asarray(self.corrected_rest_lengths_m).copy()
        initial = np.asarray(self.initial_position_m).copy()
        velocity = np.asarray(self.initial_velocity_mps).copy()
        controls = np.asarray(self.controller_points_m).copy()
        gravity = np.asarray(self.gravity_mps2).copy()
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("corrected rest positions must have shape (N, 3)")
        if lengths.ndim != 1 or np.any(lengths <= 0.0):
            raise ValueError("corrected rest lengths must be a positive vector")
        if initial.shape != positions.shape or velocity.shape != positions.shape:
            raise ValueError("initial position and velocity must match object nodes")
        if controls.ndim != 3 or controls.shape[2] != 3:
            raise ValueError("controller_points_m must have shape (T, C, 3)")
        if gravity.shape != (3,):
            raise ValueError("gravity_mps2 must be a 3-vector")
        arrays = (positions, lengths, initial, velocity, controls, gravity)
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("Warp structural arrays must be finite")
        try:
            diagnostics = json.loads(
                json.dumps(dict(self.diagnostics), sort_keys=True, allow_nan=False)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Warp diagnostics must be finite JSON data") from error
        for name, values in (
            ("corrected_rest_positions_m", positions),
            ("corrected_rest_lengths_m", lengths),
            ("initial_position_m", initial),
            ("initial_velocity_mps", velocity),
            ("controller_points_m", controls),
            ("gravity_mps2", gravity),
        ):
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def configuration_id(self) -> str:
        digest = hashlib.sha256()
        descriptor = {
            "structural_artifact_id": self.structural_artifact_id,
            "session_id": self.session_id,
            "diagnostics": self.diagnostics,
            "frozen_backend_tag": FROZEN_CAUSAL4D_TAG,
            "frozen_backend_commit": FROZEN_CAUSAL4D_COMMIT,
        }
        digest.update(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        for name, values in sorted(self.array_payload().items()):
            digest.update(name.encode("ascii"))
            digest.update(_array_sha256(values).encode("ascii"))
        return digest.hexdigest()

    def array_payload(self) -> dict[str, np.ndarray]:
        return {
            "corrected_rest_positions_m": self.corrected_rest_positions_m,
            "corrected_rest_lengths_m": self.corrected_rest_lengths_m,
            "initial_position_m": self.initial_position_m,
            "initial_velocity_mps": self.initial_velocity_mps,
            "controller_points_m": self.controller_points_m,
            "gravity_mps2": self.gravity_mps2,
        }


def _identity_session(correction: StructuralTwinCorrection, session_id: str) -> bool:
    session = correction.session(session_id)
    return bool(
        np.array_equal(
            correction.persistent_rest_coefficients,
            np.zeros_like(correction.persistent_rest_coefficients),
        )
        and np.array_equal(
            session.settled_state_coefficients,
            np.zeros_like(session.settled_state_coefficients),
        )
        and np.array_equal(session.frame_linear, np.eye(3))
        and np.array_equal(session.frame_translation_m, np.zeros(3))
        and np.array_equal(session.gravity_correction_mps2, np.zeros(3))
    )


def prepare_structural_warp_configuration(
    correction: StructuralTwinCorrection,
    nominal_rest_positions_m: np.ndarray,
    springs: np.ndarray,
    nominal_rest_lengths_m: np.ndarray,
    *,
    num_object_springs: int,
    session_id: str,
    nominal_initial_position_m: np.ndarray,
    nominal_initial_velocity_mps: np.ndarray,
    controller_points_m: np.ndarray,
    nominal_gravity_mps2: np.ndarray = np.array([0.0, 0.0, -9.81]),
    corrected_equilibrium_position_m: np.ndarray | None = None,
) -> StructuralWarpConfiguration:
    """Prepare an artifact before simulation without changing the frozen path.

    A nonzero persistent rest correction requires an equilibrium produced with
    the corrected rest lengths. This prevents treating a gravity-deformed
    observation as material rest geometry and applying gravity twice.
    """

    geometry = corrected_rest_geometry(
        correction,
        nominal_rest_positions_m,
        springs,
        nominal_rest_lengths_m,
        num_object_springs=num_object_springs,
    )
    session = correction.session(session_id)
    nominal_initial = np.asarray(nominal_initial_position_m)
    nominal_velocity = np.asarray(nominal_initial_velocity_mps)
    controls = np.asarray(controller_points_m)
    nominal_gravity = np.asarray(nominal_gravity_mps2)
    if nominal_initial.shape != geometry.rest_positions.shape:
        raise ValueError("nominal initial state differs from the structural graph")
    if nominal_velocity.shape != nominal_initial.shape:
        raise ValueError("nominal initial velocity differs from the structural graph")
    if controls.ndim != 3 or controls.shape[2] != 3:
        raise ValueError("controller_points_m must have shape (T, C, 3)")
    if nominal_gravity.shape != (3,):
        raise ValueError("nominal_gravity_mps2 must be a 3-vector")
    rest_is_nonzero = not np.array_equal(
        correction.persistent_rest_coefficients,
        np.zeros_like(correction.persistent_rest_coefficients),
    )
    if rest_is_nonzero and corrected_equilibrium_position_m is None:
        raise ValueError(
            "a nonzero rest correction requires a corrected simulated equilibrium"
        )
    identity = _identity_session(correction, session_id)
    if identity and corrected_equilibrium_position_m is None:
        initial = nominal_initial.copy()
        velocity = nominal_velocity.copy()
        corrected_controls = controls.copy()
        gravity = nominal_gravity.copy()
    else:
        equilibrium = (
            nominal_initial
            if corrected_equilibrium_position_m is None
            else np.asarray(corrected_equilibrium_position_m)
        )
        if equilibrium.shape != nominal_initial.shape or not np.all(np.isfinite(equilibrium)):
            raise ValueError("corrected equilibrium must match the object state")
        settled = structural_displacement(
            correction.graph_basis, session.settled_state_coefficients
        )
        initial = equilibrium @ session.frame_linear + session.frame_translation_m
        initial = initial + settled
        velocity = nominal_velocity @ session.frame_linear
        corrected_controls = (
            controls @ session.frame_linear + session.frame_translation_m
        )
        gravity = nominal_gravity + session.gravity_correction_mps2
    diagnostics = {
        **dict(geometry.diagnostics),
        "identity_session": identity,
        "equilibrium_required": rest_is_nonzero,
        "equilibrium_supplied": corrected_equilibrium_position_m is not None,
        "settled_state_rms_m": float(
            np.sqrt(
                np.mean(
                    np.square(
                        structural_displacement(
                            correction.graph_basis,
                            session.settled_state_coefficients,
                        )
                    )
                )
            )
        ),
        "frame_translation_norm_m": float(
            np.linalg.norm(session.frame_translation_m)
        ),
        "gravity_correction_norm_mps2": float(
            np.linalg.norm(session.gravity_correction_mps2)
        ),
        "frozen_backend_tag": FROZEN_CAUSAL4D_TAG,
        "frozen_backend_commit": FROZEN_CAUSAL4D_COMMIT,
        "legacy_backend_files_modified_by_adapter": False,
    }
    return StructuralWarpConfiguration(
        structural_artifact_id=correction.artifact_id,
        session_id=session_id,
        corrected_rest_positions_m=geometry.rest_positions,
        corrected_rest_lengths_m=geometry.rest_lengths,
        initial_position_m=initial,
        initial_velocity_mps=velocity,
        controller_points_m=corrected_controls,
        gravity_mps2=gravity,
        diagnostics=diagnostics,
    )


def assert_zero_configuration_parity(
    configuration: StructuralWarpConfiguration,
    *,
    nominal_rest_positions_m: np.ndarray,
    nominal_rest_lengths_m: np.ndarray,
    nominal_initial_position_m: np.ndarray,
    nominal_initial_velocity_mps: np.ndarray,
    controller_points_m: np.ndarray,
    nominal_gravity_mps2: np.ndarray = np.array([0.0, 0.0, -9.81]),
) -> dict[str, Any]:
    """Require every zero-correction backend input to be byte-identical."""

    comparisons = {
        "rest_positions": (
            configuration.corrected_rest_positions_m,
            nominal_rest_positions_m,
        ),
        "rest_lengths": (
            configuration.corrected_rest_lengths_m,
            nominal_rest_lengths_m,
        ),
        "initial_position": (
            configuration.initial_position_m,
            nominal_initial_position_m,
        ),
        "initial_velocity": (
            configuration.initial_velocity_mps,
            nominal_initial_velocity_mps,
        ),
        "controller_points": (
            configuration.controller_points_m,
            controller_points_m,
        ),
        "gravity": (configuration.gravity_mps2, nominal_gravity_mps2),
    }
    results = {}
    for name, (candidate, nominal) in comparisons.items():
        candidate_array = np.asarray(candidate)
        nominal_array = np.asarray(nominal)
        identical = bool(
            candidate_array.dtype == nominal_array.dtype
            and candidate_array.shape == nominal_array.shape
            and candidate_array.tobytes() == nominal_array.tobytes()
        )
        results[name] = {
            "byte_identical": identical,
            "candidate_sha256": _array_sha256(candidate_array),
            "nominal_sha256": _array_sha256(nominal_array),
        }
        if not identical:
            raise AssertionError(f"zero structural correction changed {name}")
    return {
        "passed": True,
        "frozen_backend_tag": FROZEN_CAUSAL4D_TAG,
        "frozen_backend_commit": FROZEN_CAUSAL4D_COMMIT,
        "arrays": results,
    }


def apply_structural_configuration_to_simulator(
    simulator: Any,
    torch: Any,
    wp: Any,
    configuration: StructuralWarpConfiguration,
    *,
    device: str,
    nominal_gravity_mps2: np.ndarray = np.array([0.0, 0.0, -9.81]),
) -> tuple[np.ndarray, np.ndarray]:
    """Apply rest lengths, controls, and gravity before a Warp restart."""

    rest_lengths = torch.as_tensor(
        np.asarray(configuration.corrected_rest_lengths_m).copy(),
        dtype=torch.float32,
        device=device,
    ).contiguous()
    controls = torch.as_tensor(
        np.asarray(configuration.controller_points_m).copy(),
        dtype=torch.float32,
        device=device,
    ).contiguous()
    if not hasattr(simulator, "set_rest_lengths") or not hasattr(
        simulator, "set_controller_trajectory"
    ):
        raise TypeError("simulator does not expose the structural configuration API")
    simulator.set_rest_lengths(rest_lengths)
    simulator.set_controller_trajectory(controls)
    gravity_changed = not np.array_equal(
        configuration.gravity_mps2, np.asarray(nominal_gravity_mps2)
    )
    if gravity_changed:
        if not hasattr(simulator, "set_gravity"):
            raise TypeError(
                "nonzero gravity correction requires an official backend gravity setter"
            )
        simulator.set_gravity(
            torch.as_tensor(
                np.asarray(configuration.gravity_mps2).copy(),
                dtype=torch.float32,
                device=device,
            ).contiguous()
        )
    wp.synchronize()
    return (
        np.asarray(configuration.initial_position_m).copy(),
        np.asarray(configuration.initial_velocity_mps).copy(),
    )


def write_structural_warp_configuration(
    output_dir: str | Path,
    configuration: StructuralWarpConfiguration,
) -> dict[str, Any]:
    """Export the exact corrected positions and rest lengths consumed by Warp."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "structural_warp_configuration.npz"
    manifest_path = output / "structural_warp_configuration.json"
    np.savez_compressed(archive_path, **configuration.array_payload())
    manifest = {
        "schema_version": 1,
        "artifact_kind": "StructuralWarpConfiguration",
        "configuration_id": configuration.configuration_id,
        "structural_artifact_id": configuration.structural_artifact_id,
        "session_id": configuration.session_id,
        "frozen_backend": {
            "tag": FROZEN_CAUSAL4D_TAG,
            "commit": FROZEN_CAUSAL4D_COMMIT,
            "legacy_path_modified": False,
        },
        "diagnostics": configuration.diagnostics,
        "array_archive": {
            "path": archive_path.name,
            "sha256": _file_sha256(archive_path),
            "arrays": {
                name: {
                    "shape": list(values.shape),
                    "dtype": values.dtype.str,
                    "sha256": _array_sha256(values),
                }
                for name, values in sorted(configuration.array_payload().items())
            },
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "configuration_id": configuration.configuration_id,
        "manifest_path": str(manifest_path.resolve()),
        "archive_path": str(archive_path.resolve()),
        "manifest_sha256": _file_sha256(manifest_path),
        "archive_sha256": manifest["array_archive"]["sha256"],
    }
