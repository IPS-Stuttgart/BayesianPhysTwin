"""Honest post-acquisition lock for the Deform360 visual provider.

The original provider-lock contract requires that selected calibration payloads
are unopened. Stage 1 acquisition preceded that contract, so it must remain
invalid for this campaign. This module provides a narrower recovery contract:
the exact provider and causal window are frozen after acquisition but before any
calibration score, provider comparison, or confirmation-payload access.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    literal_lower_hex,
    plain_json,
)

DEFORM360_VISUAL_PROVIDER_RECOVERY_LOCK_SCHEMA = (
    "bayesian-phystwin.deform360-visual-provider-recovery-lock"
)
DEFORM360_VISUAL_PROVIDER_RECOVERY_LOCK_VERSION = 1
DEFORM360_VISUAL_PROVIDER_RECOVERY_LOCK_SEMANTICS = (
    "post-payload-pre-score-prob4d-motioncrafter-lock-v1"
)
DEFORM360_VISUOTACTILE_PROTOCOL_ID = "deform360-official-hub-visuotactile-v1"
DEFORM360_STAGE1_PROVENANCE_ID = (
    "cc92a2a4297ef4d3a25e6c54f233469f7eef419262e5ea0d3d3904da088909b9"
)
DEFORM360_SELECTION_ARTIFACT_SHA256 = (
    "dc1c2d192fbb841d2f0e290d77f21d697983b3f8bfbcae476e71fe902309cd82"
)
DEFORM360_PROB4D_REPOSITORY = "IPS-Stuttgart/Prob4D"
DEFORM360_MOTIONCRAFTER_REPOSITORY = "TencentARC/MotionCrafter"

DEFORM360_CONTACT_TAXEL_ROWS = 12
DEFORM360_CONTACT_ACTIVE_THRESHOLD = 0.0
DEFORM360_CONTACT_MINIMUM_ACTIVE_TAXELS_PER_GRIPPER = 2
DEFORM360_PROVIDER_WINDOW_SIZE = 25
DEFORM360_PROVIDER_OVERLAP = 8
DEFORM360_PROVIDER_WINDOW_COUNT = 2
DEFORM360_OBSERVED_HISTORY_FRAMES = DEFORM360_PROVIDER_WINDOW_SIZE + (
    DEFORM360_PROVIDER_WINDOW_COUNT - 1
) * (DEFORM360_PROVIDER_WINDOW_SIZE - DEFORM360_PROVIDER_OVERLAP)
DEFORM360_OBSERVED_CONTACT_FRAMES = 6
DEFORM360_FUTURE_FRAMES = 24
DEFORM360_CAMERA_PANEL_SIZE = 3

_RECOVERY_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "artifact_id",
        "protocol_id",
        "stage1_provenance_id",
        "selection_artifact_sha256",
        "provider_repository",
        "provider_revision",
        "provider_manifest_id",
        "provider_manifest_sha256",
        "provider_api_version",
        "motioncrafter_repository",
        "motioncrafter_revision",
        "motioncrafter_model_set_id",
        "motioncrafter_model_set_manifest_sha256",
        "motioncrafter_model_type",
        "root_seed",
        "seed_policy",
        "num_inference_steps",
        "guidance_scale",
        "decode_chunk_size",
        "low_memory_usage",
        "frame_stride",
        "window_size",
        "overlap",
        "window_count",
        "height",
        "width",
        "storage_dtype",
        "observed_history_frames",
        "observed_contact_frames",
        "future_frames",
        "causal_cutoff_convention",
        "event_clock",
        "contact_taxel_rows",
        "contact_active_threshold",
        "contact_minimum_active_taxels_per_gripper",
        "insufficient_context_policy",
        "additional_metric_anchor_policy",
        "initial_metric_frame_prior_policy",
        "initial_metric_frame_prior_policy_id",
        "fusion_rule",
        "pixel_stride",
        "sampling_mode",
        "effective_samples_per_group",
        "minimum_prior_reliability",
        "gauge_mode",
        "covariance_root_mode",
        "composition_jacobian_mode",
        "allow_pointwise_covariance_fallback",
        "max_gauge_rank",
        "minimum_retained_gauge_trace",
        "full_joint_gauge_covariance",
        "persistent_material_identities",
        "provider_attestation_policy",
        "selected_calibration_payloads_opened",
        "calibration_values_used_for_provider_selection",
        "calibration_scores_opened",
        "calibration_policy_fit",
        "confirmation_payloads_opened",
        "target_outcomes_used",
        "claim_boundary",
        "metadata",
    }
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _literal_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


def _sha256(value: object, *, name: str) -> str:
    try:
        return literal_lower_hex(value, name=name, lengths={64})
    except ValueError as error:
        raise ValueError(
            f"{name} must be a literal lowercase SHA-256 digest"
        ) from error


def _revision(value: object, *, name: str) -> str:
    try:
        return literal_lower_hex(value, name=name, lengths={40})
    except ValueError as error:
        raise ValueError(
            f"{name} must be an exact literal lowercase Git commit"
        ) from error


def _finite_probability(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real number in (0, 1]")
    result = float(value)
    if not np.isfinite(result) or not (0.0 < result <= 1.0):
        raise ValueError(f"{name} must be a real number in (0, 1]")
    return result


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def _load_strict_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read visual-provider recovery lock {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError("visual-provider recovery lock root must be a JSON object")
    return value


@dataclass(frozen=True)
class Deform360CausalWindowV1:
    """One target-free contiguous processing and evaluation window."""

    contact_start_frame: int
    source_start_frame: int
    causal_cutoff_frame: int
    future_stop_frame: int
    total_episode_frames: int

    @property
    def observed_frame_count(self) -> int:
        return self.causal_cutoff_frame - self.source_start_frame

    @property
    def future_frame_count(self) -> int:
        return self.future_stop_frame - self.causal_cutoff_frame

    @property
    def processing_frame_count(self) -> int:
        return self.future_stop_frame - self.source_start_frame

    def to_record(self) -> dict[str, int]:
        return {
            "contact_start_frame": self.contact_start_frame,
            "source_start_frame": self.source_start_frame,
            "causal_cutoff_frame": self.causal_cutoff_frame,
            "future_stop_frame": self.future_stop_frame,
            "total_episode_frames": self.total_episode_frames,
            "observed_frame_count": self.observed_frame_count,
            "future_frame_count": self.future_frame_count,
            "processing_frame_count": self.processing_frame_count,
        }


def _gripper_group(sensor_name: str) -> str:
    for suffix in ("_left", "_right"):
        if sensor_name.endswith(suffix):
            return sensor_name[: -len(suffix)]
    return sensor_name


def first_deform360_contact_frame(
    tactile_by_sensor: Mapping[str, np.ndarray],
    *,
    total_episode_frames: int,
) -> int:
    """Return the first official-rule tactile contact without reading past it."""

    frame_count = genuine_integer(
        total_episode_frames,
        name="total_episode_frames",
        minimum=1,
    )
    if not tactile_by_sensor:
        raise ValueError("at least one tactile stream is required")

    streams: dict[str, np.ndarray] = {}
    for sensor_name, values in tactile_by_sensor.items():
        name = _literal_string(sensor_name, name="tactile sensor name")
        array = np.asarray(values)
        if array.ndim != 3 or array.shape[1:] != (16, 32):
            raise ValueError(f"tactile stream {name!r} must have shape (T, 16, 32)")
        if array.shape[0] < frame_count:
            raise ValueError(
                f"tactile stream {name!r} is shorter than the episode timeline"
            )
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"tactile stream {name!r} must be numeric")
        streams[name] = array

    for frame in range(frame_count):
        active_by_gripper: dict[str, int] = {}
        for sensor_name, array in streams.items():
            frame_values = array[frame, :DEFORM360_CONTACT_TAXEL_ROWS]
            if not np.isfinite(frame_values).all():
                raise ValueError(
                    f"tactile stream {sensor_name!r} has non-finite values at "
                    f"frame {frame}"
                )
            active_count = int(
                np.count_nonzero(frame_values > DEFORM360_CONTACT_ACTIVE_THRESHOLD)
            )
            group = _gripper_group(sensor_name)
            active_by_gripper[group] = active_by_gripper.get(group, 0) + active_count
        if any(
            count >= DEFORM360_CONTACT_MINIMUM_ACTIVE_TAXELS_PER_GRIPPER
            for count in active_by_gripper.values()
        ):
            return frame
    raise ValueError("no tactile contact detected in the episode")


def derive_deform360_causal_window(
    tactile_by_sensor: Mapping[str, np.ndarray],
    *,
    total_episode_frames: int,
) -> Deform360CausalWindowV1:
    """Derive the frozen two-window observed prefix and untouched future."""

    contact_start = first_deform360_contact_frame(
        tactile_by_sensor,
        total_episode_frames=total_episode_frames,
    )
    cutoff = contact_start + DEFORM360_OBSERVED_CONTACT_FRAMES
    source_start = cutoff - DEFORM360_OBSERVED_HISTORY_FRAMES
    future_stop = cutoff + DEFORM360_FUTURE_FRAMES
    if source_start < 0:
        raise ValueError("insufficient pre-contact history for the frozen provider")
    if future_stop > total_episode_frames:
        raise ValueError("insufficient untouched future for the frozen evaluation")
    return Deform360CausalWindowV1(
        contact_start_frame=contact_start,
        source_start_frame=source_start,
        causal_cutoff_frame=cutoff,
        future_stop_frame=future_stop,
        total_episode_frames=total_episode_frames,
    )


def select_deform360_camera_panel(
    camera_to_world: Mapping[str, np.ndarray],
) -> tuple[str, ...]:
    """Select the locked three-view panel from camera pose geometry only."""

    if len(camera_to_world) < DEFORM360_CAMERA_PANEL_SIZE:
        raise ValueError("insufficient calibrated cameras for the frozen panel")
    centers: dict[str, np.ndarray] = {}
    for camera_name, value in camera_to_world.items():
        name = _literal_string(camera_name, name="camera name")
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(
                f"camera-to-world for {name!r} must be a finite 4x4 matrix"
            )
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
            raise ValueError(f"camera-to-world for {name!r} is not homogeneous")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError(f"camera-to-world for {name!r} has a non-rigid rotation")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError(f"camera-to-world for {name!r} has invalid handedness")
        centers[name] = matrix[:3, 3]

    names = sorted(centers)
    center_array = np.stack([centers[name] for name in names])
    centered = center_array - center_array.mean(axis=0, keepdims=True)
    radii = np.linalg.norm(centered, axis=1)
    if np.any(radii <= 1e-9):
        raise ValueError("camera-center geometry cannot define spherical directions")
    directions = {
        name: centered[index] / radii[index] for index, name in enumerate(names)
    }

    best_names: tuple[str, ...] | None = None
    best_score: tuple[float, float] | None = None
    for candidate in itertools.combinations(names, DEFORM360_CAMERA_PANEL_SIZE):
        distances = [
            float(np.linalg.norm(directions[left] - directions[right]))
            for left, right in itertools.combinations(candidate, 2)
        ]
        score = (round(min(distances), 12), round(sum(distances), 12))
        if best_score is None or score > best_score:
            best_names = candidate
            best_score = score
    if best_names is None:
        raise AssertionError("camera-panel enumeration unexpectedly produced no result")
    return best_names


@dataclass(frozen=True)
class Deform360VisualProviderRecoveryLockV1:
    """Exact provider/window lock under the corrected information order."""

    provider_revision: str
    provider_manifest_id: str
    provider_manifest_sha256: str
    motioncrafter_revision: str
    motioncrafter_model_set_id: str
    motioncrafter_model_set_manifest_sha256: str
    initial_metric_frame_prior_policy_id: str
    root_seed: int = 20260805
    seed_policy: str = "derived-per-call"
    height: int = 320
    width: int = 640
    storage_dtype: Literal["float32"] = "float32"
    additional_metric_anchor_policy: Literal["none"] = "none"
    max_gauge_rank: int = 64
    minimum_retained_gauge_trace: float = 0.999
    metadata: Mapping[str, Any] = field(default_factory=dict)
    protocol_id: str = DEFORM360_VISUOTACTILE_PROTOCOL_ID
    stage1_provenance_id: str = DEFORM360_STAGE1_PROVENANCE_ID
    selection_artifact_sha256: str = DEFORM360_SELECTION_ARTIFACT_SHA256
    provider_repository: str = DEFORM360_PROB4D_REPOSITORY
    motioncrafter_repository: str = DEFORM360_MOTIONCRAFTER_REPOSITORY
    motioncrafter_model_type: Literal["determ"] = "determ"
    selected_calibration_payloads_opened: bool = True
    calibration_values_used_for_provider_selection: bool = False
    calibration_scores_opened: bool = False
    calibration_policy_fit: bool = False
    confirmation_payloads_opened: bool = False
    target_outcomes_used: bool = False

    def __post_init__(self) -> None:
        protocol_id = _literal_string(self.protocol_id, name="protocol_id")
        if protocol_id != DEFORM360_VISUOTACTILE_PROTOCOL_ID:
            raise ValueError("recovery lock changed protocol_id")
        stage1_id = _sha256(
            self.stage1_provenance_id,
            name="stage1_provenance_id",
        )
        if stage1_id != DEFORM360_STAGE1_PROVENANCE_ID:
            raise ValueError("recovery lock changed Stage 1 provenance")
        selection_id = _sha256(
            self.selection_artifact_sha256,
            name="selection_artifact_sha256",
        )
        if selection_id != DEFORM360_SELECTION_ARTIFACT_SHA256:
            raise ValueError("recovery lock changed the selected cohort")
        if self.provider_repository != DEFORM360_PROB4D_REPOSITORY:
            raise ValueError("recovery lock changed the Prob4D repository")
        if self.motioncrafter_repository != DEFORM360_MOTIONCRAFTER_REPOSITORY:
            raise ValueError("recovery lock changed the MotionCrafter repository")
        if self.motioncrafter_model_type != "determ":
            raise ValueError("recovery lock requires deterministic MotionCrafter")
        if self.storage_dtype != "float32":
            raise ValueError("recovery lock requires float32 storage")
        if self.additional_metric_anchor_policy != "none":
            raise ValueError("recovery lock forbids additional metric anchors")

        provider_revision = _revision(
            self.provider_revision,
            name="provider_revision",
        )
        provider_manifest_id = _sha256(
            self.provider_manifest_id,
            name="provider_manifest_id",
        )
        provider_manifest_sha256 = _sha256(
            self.provider_manifest_sha256,
            name="provider_manifest_sha256",
        )
        motioncrafter_revision = _revision(
            self.motioncrafter_revision,
            name="motioncrafter_revision",
        )
        model_set_id = _sha256(
            self.motioncrafter_model_set_id,
            name="motioncrafter_model_set_id",
        )
        model_set_manifest_sha256 = _sha256(
            self.motioncrafter_model_set_manifest_sha256,
            name="motioncrafter_model_set_manifest_sha256",
        )
        metric_prior_policy_id = _sha256(
            self.initial_metric_frame_prior_policy_id,
            name="initial_metric_frame_prior_policy_id",
        )
        root_seed = genuine_integer(self.root_seed, name="root_seed", minimum=0)
        seed_policy = _literal_string(self.seed_policy, name="seed_policy")
        height = genuine_integer(self.height, name="height", minimum=1)
        width = genuine_integer(self.width, name="width", minimum=1)
        max_gauge_rank = genuine_integer(
            self.max_gauge_rank,
            name="max_gauge_rank",
            minimum=1,
        )
        retained_trace = _finite_probability(
            self.minimum_retained_gauge_trace,
            name="minimum_retained_gauge_trace",
        )

        selected_opened = genuine_boolean(
            self.selected_calibration_payloads_opened,
            name="selected_calibration_payloads_opened",
        )
        calibration_values_used = genuine_boolean(
            self.calibration_values_used_for_provider_selection,
            name="calibration_values_used_for_provider_selection",
        )
        calibration_scores_opened = genuine_boolean(
            self.calibration_scores_opened,
            name="calibration_scores_opened",
        )
        calibration_policy_fit = genuine_boolean(
            self.calibration_policy_fit,
            name="calibration_policy_fit",
        )
        confirmation_opened = genuine_boolean(
            self.confirmation_payloads_opened,
            name="confirmation_payloads_opened",
        )
        target_used = genuine_boolean(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        if not selected_opened:
            raise ValueError("recovery lock must record opened calibration payloads")
        if calibration_values_used:
            raise ValueError("provider selection must not use calibration values")
        if calibration_scores_opened or calibration_policy_fit:
            raise ValueError(
                "recovery lock must precede calibration scoring and fitting"
            )
        if confirmation_opened or target_used:
            raise ValueError(
                "recovery lock must precede confirmation and target access"
            )

        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "stage1_provenance_id", stage1_id)
        object.__setattr__(self, "selection_artifact_sha256", selection_id)
        object.__setattr__(self, "provider_revision", provider_revision)
        object.__setattr__(self, "provider_manifest_id", provider_manifest_id)
        object.__setattr__(self, "provider_manifest_sha256", provider_manifest_sha256)
        object.__setattr__(self, "motioncrafter_revision", motioncrafter_revision)
        object.__setattr__(self, "motioncrafter_model_set_id", model_set_id)
        object.__setattr__(
            self,
            "motioncrafter_model_set_manifest_sha256",
            model_set_manifest_sha256,
        )
        object.__setattr__(
            self,
            "initial_metric_frame_prior_policy_id",
            metric_prior_policy_id,
        )
        object.__setattr__(self, "root_seed", root_seed)
        object.__setattr__(self, "seed_policy", seed_policy)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "max_gauge_rank", max_gauge_rank)
        object.__setattr__(self, "minimum_retained_gauge_trace", retained_trace)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="visual-provider recovery-lock metadata",
            ),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": DEFORM360_VISUAL_PROVIDER_RECOVERY_LOCK_SCHEMA,
            "schema_version": DEFORM360_VISUAL_PROVIDER_RECOVERY_LOCK_VERSION,
            "semantics": DEFORM360_VISUAL_PROVIDER_RECOVERY_LOCK_SEMANTICS,
            "protocol_id": self.protocol_id,
            "stage1_provenance_id": self.stage1_provenance_id,
            "selection_artifact_sha256": self.selection_artifact_sha256,
            "provider_repository": self.provider_repository,
            "provider_revision": self.provider_revision,
            "provider_manifest_id": self.provider_manifest_id,
            "provider_manifest_sha256": self.provider_manifest_sha256,
            "provider_api_version": 2,
            "motioncrafter_repository": self.motioncrafter_repository,
            "motioncrafter_revision": self.motioncrafter_revision,
            "motioncrafter_model_set_id": self.motioncrafter_model_set_id,
            "motioncrafter_model_set_manifest_sha256": (
                self.motioncrafter_model_set_manifest_sha256
            ),
            "motioncrafter_model_type": self.motioncrafter_model_type,
            "root_seed": self.root_seed,
            "seed_policy": self.seed_policy,
            "num_inference_steps": 5,
            "guidance_scale": 1.0,
            "decode_chunk_size": 25,
            "low_memory_usage": True,
            "frame_stride": 1,
            "window_size": DEFORM360_PROVIDER_WINDOW_SIZE,
            "overlap": DEFORM360_PROVIDER_OVERLAP,
            "window_count": DEFORM360_PROVIDER_WINDOW_COUNT,
            "height": self.height,
            "width": self.width,
            "storage_dtype": self.storage_dtype,
            "observed_history_frames": DEFORM360_OBSERVED_HISTORY_FRAMES,
            "observed_contact_frames": DEFORM360_OBSERVED_CONTACT_FRAMES,
            "future_frames": DEFORM360_FUTURE_FRAMES,
            "causal_cutoff_convention": "exclusive",
            "event_clock": "official-processed-tactile-first-contact-v1",
            "contact_taxel_rows": DEFORM360_CONTACT_TAXEL_ROWS,
            "contact_active_threshold": DEFORM360_CONTACT_ACTIVE_THRESHOLD,
            "contact_minimum_active_taxels_per_gripper": (
                DEFORM360_CONTACT_MINIMUM_ACTIVE_TAXELS_PER_GRIPPER
            ),
            "insufficient_context_policy": "retained-technical-failure-no-replacement",
            "additional_metric_anchor_policy": self.additional_metric_anchor_policy,
            "initial_metric_frame_prior_policy": (
                "first-observed-frame-official-multiview-depth-sim3-v1"
            ),
            "initial_metric_frame_prior_policy_id": (
                self.initial_metric_frame_prior_policy_id
            ),
            "fusion_rule": "decoded-uniform",
            "pixel_stride": 4,
            "sampling_mode": "fixed_grid",
            "effective_samples_per_group": 64.0,
            "minimum_prior_reliability": 0.05,
            "gauge_mode": "sequential",
            "covariance_root_mode": "canonical_eigenspaces",
            "composition_jacobian_mode": "analytic",
            "allow_pointwise_covariance_fallback": False,
            "max_gauge_rank": self.max_gauge_rank,
            "minimum_retained_gauge_trace": self.minimum_retained_gauge_trace,
            "full_joint_gauge_covariance": True,
            "persistent_material_identities": True,
            "provider_attestation_policy": (
                "claim-bearing-calibrated-attestation-bound-by-calibration-lock-v1"
            ),
            "selected_calibration_payloads_opened": (
                self.selected_calibration_payloads_opened
            ),
            "calibration_values_used_for_provider_selection": (
                self.calibration_values_used_for_provider_selection
            ),
            "calibration_scores_opened": self.calibration_scores_opened,
            "calibration_policy_fit": self.calibration_policy_fit,
            "confirmation_payloads_opened": self.confirmation_payloads_opened,
            "target_outcomes_used": self.target_outcomes_used,
            "claim_boundary": (
                "Post-acquisition, pre-score recovery lock only. It establishes the "
                "provider and causal processing window for a still-sealed confirmation "
                "study; it is not pre-payload preregistration, calibration evidence, "
                "provider-competence evidence, physical improvement, or SOTA evidence."
            ),
            "metadata": plain_json(self.metadata),
        }

    @property
    def artifact_id(self) -> str:
        return _content_id(self.descriptor())

    def to_record(self) -> dict[str, object]:
        return {"artifact_id": self.artifact_id, **self.descriptor()}

    @classmethod
    def from_mapping(
        cls,
        value: object,
    ) -> Deform360VisualProviderRecoveryLockV1:
        if not isinstance(value, Mapping):
            raise ValueError("visual-provider recovery lock must be a JSON object")
        missing = sorted(_RECOVERY_FIELDS - set(value))
        extra = sorted(set(value) - _RECOVERY_FIELDS)
        if missing or extra:
            raise ValueError(
                "visual-provider recovery-lock fields changed: "
                f"missing={missing}, extra={extra}"
            )
        if value["schema"] != DEFORM360_VISUAL_PROVIDER_RECOVERY_LOCK_SCHEMA:
            raise ValueError("unsupported visual-provider recovery-lock schema")
        if value["schema_version"] != DEFORM360_VISUAL_PROVIDER_RECOVERY_LOCK_VERSION:
            raise ValueError("unsupported visual-provider recovery-lock version")
        if value["semantics"] != DEFORM360_VISUAL_PROVIDER_RECOVERY_LOCK_SEMANTICS:
            raise ValueError("visual-provider recovery-lock semantics changed")

        fixed_fields = {
            "provider_api_version": 2,
            "num_inference_steps": 5,
            "guidance_scale": 1.0,
            "decode_chunk_size": 25,
            "low_memory_usage": True,
            "frame_stride": 1,
            "window_size": DEFORM360_PROVIDER_WINDOW_SIZE,
            "overlap": DEFORM360_PROVIDER_OVERLAP,
            "window_count": DEFORM360_PROVIDER_WINDOW_COUNT,
            "observed_history_frames": DEFORM360_OBSERVED_HISTORY_FRAMES,
            "observed_contact_frames": DEFORM360_OBSERVED_CONTACT_FRAMES,
            "future_frames": DEFORM360_FUTURE_FRAMES,
            "causal_cutoff_convention": "exclusive",
            "event_clock": "official-processed-tactile-first-contact-v1",
            "contact_taxel_rows": DEFORM360_CONTACT_TAXEL_ROWS,
            "contact_active_threshold": DEFORM360_CONTACT_ACTIVE_THRESHOLD,
            "contact_minimum_active_taxels_per_gripper": (
                DEFORM360_CONTACT_MINIMUM_ACTIVE_TAXELS_PER_GRIPPER
            ),
            "insufficient_context_policy": "retained-technical-failure-no-replacement",
            "initial_metric_frame_prior_policy": (
                "first-observed-frame-official-multiview-depth-sim3-v1"
            ),
            "fusion_rule": "decoded-uniform",
            "pixel_stride": 4,
            "sampling_mode": "fixed_grid",
            "effective_samples_per_group": 64.0,
            "minimum_prior_reliability": 0.05,
            "gauge_mode": "sequential",
            "covariance_root_mode": "canonical_eigenspaces",
            "composition_jacobian_mode": "analytic",
            "allow_pointwise_covariance_fallback": False,
            "full_joint_gauge_covariance": True,
            "persistent_material_identities": True,
            "provider_attestation_policy": (
                "claim-bearing-calibrated-attestation-bound-by-calibration-lock-v1"
            ),
        }
        for field_name, expected in fixed_fields.items():
            if value[field_name] != expected or type(value[field_name]) is not type(
                expected
            ):
                raise ValueError(f"visual-provider recovery lock changed {field_name}")

        lock = cls(
            provider_revision=cast(str, value["provider_revision"]),
            provider_manifest_id=cast(str, value["provider_manifest_id"]),
            provider_manifest_sha256=cast(str, value["provider_manifest_sha256"]),
            motioncrafter_revision=cast(str, value["motioncrafter_revision"]),
            motioncrafter_model_set_id=cast(
                str,
                value["motioncrafter_model_set_id"],
            ),
            motioncrafter_model_set_manifest_sha256=cast(
                str,
                value["motioncrafter_model_set_manifest_sha256"],
            ),
            initial_metric_frame_prior_policy_id=cast(
                str,
                value["initial_metric_frame_prior_policy_id"],
            ),
            root_seed=cast(int, value["root_seed"]),
            seed_policy=cast(str, value["seed_policy"]),
            height=cast(int, value["height"]),
            width=cast(int, value["width"]),
            storage_dtype=cast(Literal["float32"], value["storage_dtype"]),
            additional_metric_anchor_policy=cast(
                Literal["none"],
                value["additional_metric_anchor_policy"],
            ),
            max_gauge_rank=cast(int, value["max_gauge_rank"]),
            minimum_retained_gauge_trace=cast(
                float,
                value["minimum_retained_gauge_trace"],
            ),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            protocol_id=cast(str, value["protocol_id"]),
            stage1_provenance_id=cast(str, value["stage1_provenance_id"]),
            selection_artifact_sha256=cast(
                str,
                value["selection_artifact_sha256"],
            ),
            provider_repository=cast(str, value["provider_repository"]),
            motioncrafter_repository=cast(
                str,
                value["motioncrafter_repository"],
            ),
            motioncrafter_model_type=cast(
                Literal["determ"],
                value["motioncrafter_model_type"],
            ),
            selected_calibration_payloads_opened=cast(
                bool,
                value["selected_calibration_payloads_opened"],
            ),
            calibration_values_used_for_provider_selection=cast(
                bool,
                value["calibration_values_used_for_provider_selection"],
            ),
            calibration_scores_opened=cast(
                bool,
                value["calibration_scores_opened"],
            ),
            calibration_policy_fit=cast(bool, value["calibration_policy_fit"]),
            confirmation_payloads_opened=cast(
                bool,
                value["confirmation_payloads_opened"],
            ),
            target_outcomes_used=cast(bool, value["target_outcomes_used"]),
        )
        if value["claim_boundary"] != lock.descriptor()["claim_boundary"]:
            raise ValueError("visual-provider recovery lock changed claim_boundary")
        declared_id = _sha256(value["artifact_id"], name="artifact_id")
        if declared_id != lock.artifact_id:
            raise ValueError("visual-provider recovery-lock artifact_id changed")
        return lock


def save_deform360_visual_provider_recovery_lock(
    path: str | Path,
    lock: Deform360VisualProviderRecoveryLockV1,
) -> None:
    Path(path).write_text(
        json.dumps(lock.to_record(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_deform360_visual_provider_recovery_lock(
    path: str | Path,
) -> Deform360VisualProviderRecoveryLockV1:
    return Deform360VisualProviderRecoveryLockV1.from_mapping(
        _load_strict_json(Path(path))
    )


__all__ = [
    "DEFORM360_FUTURE_FRAMES",
    "DEFORM360_CAMERA_PANEL_SIZE",
    "DEFORM360_OBSERVED_CONTACT_FRAMES",
    "DEFORM360_OBSERVED_HISTORY_FRAMES",
    "DEFORM360_PROVIDER_OVERLAP",
    "DEFORM360_PROVIDER_WINDOW_COUNT",
    "DEFORM360_PROVIDER_WINDOW_SIZE",
    "DEFORM360_VISUAL_PROVIDER_RECOVERY_LOCK_SCHEMA",
    "DEFORM360_VISUAL_PROVIDER_RECOVERY_LOCK_SEMANTICS",
    "DEFORM360_VISUAL_PROVIDER_RECOVERY_LOCK_VERSION",
    "Deform360CausalWindowV1",
    "Deform360VisualProviderRecoveryLockV1",
    "derive_deform360_causal_window",
    "first_deform360_contact_frame",
    "load_deform360_visual_provider_recovery_lock",
    "save_deform360_visual_provider_recovery_lock",
    "select_deform360_camera_panel",
]
