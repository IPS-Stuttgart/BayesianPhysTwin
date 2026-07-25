"""Future-blind frame-zero assets for held Deform360 online belief.

This module deliberately has no HDF5 reader and no outcome argument.  It decodes
exactly one RGB frame from each selected camera, segments those materialized
frames with the pinned SAM 2.1 image model, and derives a multiview visual hull,
surface colors, depth maps, and projection matrices from immutable calibration.
The aligned realized robot kinematics are bound as an exogenous known input.
Only the fourth, reference-optional fallback may use its current row to render
and subtract the pinned official robot silhouette; the first three geometry
paths remain robot-independent.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .deform360_object_sam2 import (
    DeformableObjectSam2MaskConfig,
    deformable_object_mask_candidate_diagnostics,
    mask_appearance_descriptor,
    mask_appearance_similarity,
)
from .deform360_sam2 import (
    PINNED_SAM2_CHECKPOINT_SHA256,
    PINNED_SAM2_COMMIT,
    PINNED_SAM2_MODEL_CONFIG,
    PINNED_SAM2_REPOSITORY,
)
from .deform360_visual_hull import (
    carve_candidate_points,
    regular_grid_in_bounds,
)

from .deform360_robot_kinematics import (
    Deform360RobotKinematics,
    ROBOT_KINEMATICS_WINDOW_CONTRACT,
    ROBOT_KINEMATICS_WINDOW_POLICY_ID,
    load_robot_kinematics_archive,
    select_robot_kinematics_window,
    slice_robot_kinematics,
    validate_robot_kinematics_selection_audit,
    validate_selected_robot_kinematics_bundle,
)
from .deform360_dataset_containment import (
    AlignedEpisodeLayout,
    layout_from_dataset_file,
    validate_aligned_episode,
    validate_regular_file_nofollow,
)
from .deform360_frame_zero_semantic_gate import (
    FRAME_ZERO_REFERENCE_OPTIONAL_ASSIGNMENT_STRATEGY,
    FRAME_ZERO_REFERENCE_OPTIONAL_FALLBACK_POLICY_ID,
    FRAME_ZERO_REFERENCE_OPTIONAL_GEOMETRY_STRATEGY,
    FRAME_ZERO_SEMANTIC_GATE_CONTRACT,
    FRAME_ZERO_SEMANTIC_GATE_CONTRACT_SHA256,
    semantic_label_for_object_id,
    validate_semantic_gate_audit,
)

FRAME_ZERO_BUNDLE_SCHEMA_VERSION = 1
HELD_PROTOCOL_ID = "deform360-held-online-belief-v7"
_HELD_V8_PROTOCOL_ID = "deform360-held-online-belief-v8.3"
HELD_LOCK_ARTIFACT_KIND = "Deform360HeldOnlineBeliefLock"
FRAME_ZERO_BUNDLE_ARTIFACT_KIND = "Deform360HeldFrameZeroBundle"
FRAME_ZERO_CAMERA_SELECTION_POLICY_ID = (
    "deform360-frame-zero-reference-anchored-inlier-abstention-v3"
)
FRAME_ZERO_CAMERA_SELECTION_RULE = (
    "process every aligned calibrated camera; keep the fixed reference and "
    "frozen-threshold-eligible views, except that the audited common-geometry "
    "fallback retains the fixed reference plus seven deterministic "
    "maximum-consensus inliers"
)
FRAME_ZERO_GEOMETRY_FALLBACK_POLICY_ID = (
    "deform360-frame-zero-reference-anchored-exact-eight-v2"
)
FRAME_ZERO_REFERENCE_OPTIONAL_CAMERA_SELECTION_POLICY_ID = (
    "deform360-frame-zero-reference-conditioned-exact-eight-abstention-v1"
)
FRAME_ZERO_REFERENCE_OPTIONAL_CAMERA_SELECTION_RULE = (
    "process every aligned calibrated camera; after the three reference-anchored "
    "strategies fail, condition proposal eligibility on the frozen fixed-reference "
    "seeds but exhaustively select exactly eight common-geometry cameras, allowing "
    "the fixed reference itself to abstain"
)

# These are source-bound policy constants rather than user-facing configuration.
# The held lock binds this module byte-for-byte, and keeping them out of
# ``FrameZeroAssetConfig`` preserves the already frozen legacy configuration.
_FALLBACK_STRICT_CONSENSUS_VOTES = 8
_FALLBACK_COMMON_GRID_AXIS_COUNT = 64
_FALLBACK_COMMON_LOCAL_REQUESTED_VOXEL_SIZE_M = 0.004
_FALLBACK_REFERENCE_SEED_POLICY = "top-eight-frozen-local-mask-score"
_FALLBACK_REFERENCE_SEED_COUNT = 8
_FALLBACK_MINIMUM_COARSE_COMPONENT_POINT_COUNT = 64
_FALLBACK_LOCAL_REQUESTED_VOXEL_SIZE_M = 0.002
_FALLBACK_STABILITY_REQUESTED_VOXEL_SIZE_M = 0.0025
_FALLBACK_MAXIMUM_LOCAL_GRID_POINT_COUNT = 8_000_000
_FALLBACK_LOCAL_MARGIN_COARSE_CELLS = 3
_FALLBACK_MINIMUM_REFINED_SURFACE_POINT_COUNT = 128
_FALLBACK_MINIMUM_SCALE_STABILITY = 0.70
_FALLBACK_ACCEPTANCE_GATE_NAMES = (
    "camera_count",
    "coarse_strict_consensus_vote_count",
    "refined_strict_consensus_vote_count",
    "stability_strict_consensus_vote_count",
    "coarse_connected_core_point_count",
    "coarse_largest_component_fraction",
    "refined_surface_point_count",
    "refined_largest_component_fraction",
    "refined_grid_not_coarsened",
    "local_scale_stability",
    "stability_largest_component_fraction",
    "stability_grid_not_coarsened",
    "raw_median_hull_mask_containment",
    "median_depth_mask_coverage",
    "all_refined_surface_points_have_footprint_support",
    "all_sampled_surface_points_colored",
)

_NO_AUTOMATIC_CANDIDATES = "no-automatic-mask-candidates"
_NO_MASK_THRESHOLD_CANDIDATES = "no-candidate-met-frozen-mask-thresholds"
_NO_REFERENCE_CONSISTENT_CANDIDATES = (
    "no-candidate-met-frozen-reference-appearance-threshold"
)
_COMMON_INLIER_ABSTENTION_REASON = (
    "excluded-by-deterministic-reference-anchored-max-consensus-selection"
)
_COMMON_INLIER_SELECTION_RULE = (
    "top-eight-local-mask-reference-seeds-anchored-global-local-exact-eight"
)
_REFERENCE_OPTIONAL_INLIER_SELECTION_RULE = (
    "reference-conditioned-candidates-exhaustive-reference-optional-exact8"
)
EXACT_EIGHT_SUBSET_BOUNDED_AUDIT_SCHEMA_ID = (
    "deform360-exhaustive-exact-eight-canonical-json-array-sha256-v1"
)
_EXACT_EIGHT_SUBSET_RECORD_FIELDS = (
    "cameras",
    "largest_exact_component_voxel_count",
    "exact_common_voxel_count",
    "exact_component_count",
    "raw_component_coverage_sum",
    "semantic_score_sum",
    "exact_common_mask_sha256",
)
_EXACT_EIGHT_SUBSET_INTEGER_METRICS = (
    "largest_exact_component_voxel_count",
    "exact_common_voxel_count",
    "exact_component_count",
)
_EXACT_EIGHT_SUBSET_FLOAT_METRICS = (
    "raw_component_coverage_sum",
    "semantic_score_sum",
)

HELD_TARGET_CASES_V1 = (
    "002-rope-silk-ep0001",
    "081-stripe-rope-ep0005",
    "085-scarf-cloth-ep0002",
    "083-blanket-cloth-ep0007",
    "092-squirrel-ep0001",
    "170-spider-ep0006",
)
ABSOLUTELY_FORBIDDEN_OBJECT_PREFIXES = frozenset({"003", "004", "086", "171"})
APPROVED_CALIBRATION_SMOKE_CASE = "083-blanket-cloth-ep0000"

FRAME_ZERO_INFORMATION_BOUNDARY: dict[str, Any] = {
    "maximum_object_rgb_frame_read": 0,
    "object_observation_frames_used": [0],
    "known_aligned_realized_robot_kinematics_read": True,
    "known_robot_trajectory_semantics": ROBOT_KINEMATICS_WINDOW_CONTRACT[
        "trajectory_semantics"
    ],
    "robot_delta_command_read": False,
    "commanded_control_read": False,
    # Compatibility key retained for the v2 held-protocol validator.  The
    # truthful fields above state what the public Deform360 archive contains.
    "known_future_robot_action_read": True,
    "future_object_rgb_read": False,
    "future_object_geometry_read": False,
    "future_depth_or_mask_read": False,
    "future_tactile_read": False,
    "outcome_created": False,
    "outcome_read": False,
    "whole_future_container_hashed_or_read": False,
}

_CASE_PATTERN = re.compile(r"^(?P<object>\d{3}-.+)-ep(?P<episode>\d{4})$")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_INPUT_SUFFIXES = frozenset({".h5", ".hdf5", ".hdf"})
_FORBIDDEN_DERIVATION_TOKENS = (
    "mask_refined",
    "rendered_depth",
    "future_mask",
    "future_depth",
    "propagated_mask",
    "sam2_propagat",
    "target_data",
    "ground_truth",
    "outcome",
)
_FRAME_ZERO_CAMERA_ACCESS_FIELDS = frozenset(
    {
        "camera",
        "path",
        "decoded_frame_count",
        "maximum_rgb_frame_read",
        "action_window_frame_index",
        "source_aligned_frame_index",
        "decoded_rgb_sha256",
        "whole_file_hashed_or_read",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class FrameZeroGeometryQAError(ValueError):
    """A source-only geometry gate failure eligible for audited fallback."""


def _require_geometry(condition: bool, message: str) -> None:
    if not condition:
        raise FrameZeroGeometryQAError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    path = validate_regular_file_nofollow(path, label="SHA-256 input")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    return _sha256_file(Path(path))


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = _canonical_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = validate_regular_file_nofollow(path, label="bound input")
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def artifact_sha256(payload: Mapping[str, Any]) -> str:
    """Canonical JSON hash used by both the generic lock and bundle manifest."""

    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


EXACT_EIGHT_SUBSET_BOUNDED_AUDIT_CONTRACT: Mapping[str, Any] = {
    "contract_id": EXACT_EIGHT_SUBSET_BOUNDED_AUDIT_SCHEMA_ID,
    "held_protocol_id": _HELD_V8_PROTOCOL_ID,
    "enumeration_and_selection": (
        "unchanged exhaustive itertools.combinations order and unchanged "
        "five-term lexicographic winner"
    ),
    "record_fields": list(_EXACT_EIGHT_SUBSET_RECORD_FIELDS),
    "canonical_stream": (
        "sha256 of the compact sorted-key allow_nan=false JSON array of every "
        "record in enumeration order"
    ),
    "retained_records": ["first_record", "last_record", "selected_record"],
    "retained_summaries": ["record_count", "feasibility_counts", "metric_extrema"],
    "sidecar": None,
    "legacy_v7_representation": "full record list unchanged",
}
EXACT_EIGHT_SUBSET_BOUNDED_AUDIT_CONTRACT_SHA256 = hashlib.sha256(
    _canonical_bytes(EXACT_EIGHT_SUBSET_BOUNDED_AUDIT_CONTRACT)
).hexdigest()


class _ExactEightSubsetAuditAccumulator:
    """Accumulate the exhaustive audit without retaining it under held v8.

    The streaming digest is exactly the SHA-256 of the compact, sorted-key JSON
    array that the legacy path materializes.  Thus a small fixture can compare
    the two representations byte-for-byte while a large v8 run keeps only a
    constant-size commitment and summary.
    """

    def __init__(self, *, bounded: bool) -> None:
        self._records: list[dict[str, Any]] | None = [] if not bounded else None
        self._digest = hashlib.sha256()
        if bounded:
            self._digest.update(b"[")
        self._record_count = 0
        self._first_record: dict[str, Any] | None = None
        self._last_record: dict[str, Any] | None = None
        self._positive_exact_common_count = 0
        self._positive_largest_component_count = 0
        self._minimum: dict[str, int | float] = {}
        self._maximum: dict[str, int | float] = {}

    @property
    def record_count(self) -> int:
        return self._record_count

    def add(self, record: dict[str, Any]) -> None:
        if self._records is not None:
            self._records.append(record)
        else:
            if self._record_count:
                self._digest.update(b",")
            self._digest.update(_canonical_bytes(record))
            if self._first_record is None:
                self._first_record = deepcopy(record)
            # Every enumeration iteration creates a new immutable audit record,
            # so retaining only the current tail avoids millions of deep copies.
            self._last_record = record
            if int(record["exact_common_voxel_count"]) > 0:
                self._positive_exact_common_count += 1
            if int(record["largest_exact_component_voxel_count"]) > 0:
                self._positive_largest_component_count += 1
            for key in (
                *_EXACT_EIGHT_SUBSET_INTEGER_METRICS,
                *_EXACT_EIGHT_SUBSET_FLOAT_METRICS,
            ):
                value = record[key]
                if key not in self._minimum:
                    self._minimum[key] = value
                    self._maximum[key] = value
                else:
                    self._minimum[key] = min(self._minimum[key], value)
                    self._maximum[key] = max(self._maximum[key], value)
        self._record_count += 1

    def materialize(self, *, selected_record: Mapping[str, Any]) -> object:
        if self._records is not None:
            return self._records
        _require_geometry(
            self._record_count > 0
            and self._first_record is not None
            and self._last_record is not None,
            "exact-eight subset audit is empty",
        )
        digest = self._digest.copy()
        digest.update(b"]")
        selected = {
            key: deepcopy(selected_record[key])
            for key in _EXACT_EIGHT_SUBSET_RECORD_FIELDS
        }
        return {
            "schema_id": EXACT_EIGHT_SUBSET_BOUNDED_AUDIT_SCHEMA_ID,
            "contract_sha256": EXACT_EIGHT_SUBSET_BOUNDED_AUDIT_CONTRACT_SHA256,
            "record_count": self._record_count,
            "canonical_json_array_sha256": digest.hexdigest(),
            "first_record": self._first_record,
            "last_record": self._last_record,
            "selected_record": selected,
            "feasibility_counts": {
                "positive_exact_common_voxel_record_count": (
                    self._positive_exact_common_count
                ),
                "zero_exact_common_voxel_record_count": (
                    self._record_count - self._positive_exact_common_count
                ),
                "positive_largest_exact_component_record_count": (
                    self._positive_largest_component_count
                ),
                "zero_largest_exact_component_record_count": (
                    self._record_count - self._positive_largest_component_count
                ),
            },
            "metric_extrema": {
                key: {
                    "minimum": self._minimum[key],
                    "maximum": self._maximum[key],
                }
                for key in (
                    *_EXACT_EIGHT_SUBSET_INTEGER_METRICS,
                    *_EXACT_EIGHT_SUBSET_FLOAT_METRICS,
                )
            },
        }


def _valid_exact_eight_subset_record(
    record: object,
    *,
    candidate_cameras: Sequence[str],
    fixed_first_camera: str | None,
) -> bool:
    if not isinstance(record, Mapping) or set(record) != set(
        _EXACT_EIGHT_SUBSET_RECORD_FIELDS
    ):
        return False
    cameras = record.get("cameras")
    return bool(
        isinstance(cameras, list)
        and len(cameras) == _FALLBACK_STRICT_CONSENSUS_VOTES
        and len(set(cameras)) == _FALLBACK_STRICT_CONSENSUS_VOTES
        and all(
            isinstance(camera, str) and camera in candidate_cameras
            for camera in cameras
        )
        and (fixed_first_camera is None or cameras[0] == fixed_first_camera)
        and _valid_sha256(record.get("exact_common_mask_sha256"))
        and all(
            isinstance(record.get(key), int)
            and not isinstance(record.get(key), bool)
            and int(record[key]) >= 0
            for key in _EXACT_EIGHT_SUBSET_INTEGER_METRICS
        )
        and all(
            isinstance(record.get(key), (int, float))
            and not isinstance(record.get(key), bool)
            and math.isfinite(float(record[key]))
            for key in _EXACT_EIGHT_SUBSET_FLOAT_METRICS
        )
    )


def _validate_bounded_exact_eight_subset_audit(
    value: object,
    *,
    expected_record_count: int,
    expected_first_cameras: Sequence[str],
    expected_last_cameras: Sequence[str],
    selected_cameras: Sequence[str],
    candidate_cameras: Sequence[str],
    fixed_first_camera: str | None,
) -> Mapping[str, Any]:
    """Validate the constant-size v8 commitment to exhaustive enumeration."""

    keys = {
        "schema_id",
        "contract_sha256",
        "record_count",
        "canonical_json_array_sha256",
        "first_record",
        "last_record",
        "selected_record",
        "feasibility_counts",
        "metric_extrema",
    }
    _require(
        isinstance(value, Mapping)
        and set(value) == keys
        and value.get("schema_id") == EXACT_EIGHT_SUBSET_BOUNDED_AUDIT_SCHEMA_ID
        and value.get("contract_sha256")
        == EXACT_EIGHT_SUBSET_BOUNDED_AUDIT_CONTRACT_SHA256
        and value.get("record_count") == expected_record_count > 0
        and _valid_sha256(value.get("canonical_json_array_sha256")),
        "invalid bounded exact-eight subset audit header",
    )
    first = value["first_record"]
    last = value["last_record"]
    selected = value["selected_record"]
    _require(
        all(
            _valid_exact_eight_subset_record(
                record,
                candidate_cameras=candidate_cameras,
                fixed_first_camera=fixed_first_camera,
            )
            for record in (first, last, selected)
        )
        and first["cameras"] == list(expected_first_cameras)
        and last["cameras"] == list(expected_last_cameras)
        and sorted(selected["cameras"]) == list(selected_cameras)
        and (
            expected_record_count != 1
            or (
                first == last == selected
                and value["canonical_json_array_sha256"]
                == hashlib.sha256(_canonical_bytes([first])).hexdigest()
            )
        ),
        "invalid bounded exact-eight subset endpoint/selection audit",
    )
    feasibility = value["feasibility_counts"]
    feasibility_keys = {
        "positive_exact_common_voxel_record_count",
        "zero_exact_common_voxel_record_count",
        "positive_largest_exact_component_record_count",
        "zero_largest_exact_component_record_count",
    }
    _require(
        isinstance(feasibility, Mapping)
        and set(feasibility) == feasibility_keys
        and all(
            isinstance(feasibility.get(key), int)
            and not isinstance(feasibility.get(key), bool)
            and 0 <= feasibility[key] <= expected_record_count
            for key in feasibility_keys
        )
        and feasibility["positive_exact_common_voxel_record_count"]
        + feasibility["zero_exact_common_voxel_record_count"]
        == expected_record_count
        and feasibility["positive_largest_exact_component_record_count"]
        + feasibility["zero_largest_exact_component_record_count"]
        == expected_record_count
        and feasibility["positive_largest_exact_component_record_count"]
        <= feasibility["positive_exact_common_voxel_record_count"],
        "invalid bounded exact-eight subset feasibility summary",
    )
    extrema = value["metric_extrema"]
    metric_keys = {
        *_EXACT_EIGHT_SUBSET_INTEGER_METRICS,
        *_EXACT_EIGHT_SUBSET_FLOAT_METRICS,
    }
    _require(
        isinstance(extrema, Mapping) and set(extrema) == metric_keys,
        "invalid bounded exact-eight subset extrema",
    )
    for key in metric_keys:
        bounds = extrema[key]
        integer_metric = key in _EXACT_EIGHT_SUBSET_INTEGER_METRICS
        _require(
            isinstance(bounds, Mapping)
            and set(bounds) == {"minimum", "maximum"}
            and all(
                (isinstance(bound, int) and not isinstance(bound, bool) and bound >= 0)
                if integer_metric
                else (
                    isinstance(bound, (int, float))
                    and not isinstance(bound, bool)
                    and math.isfinite(float(bound))
                )
                for bound in bounds.values()
            )
            and bounds["minimum"] <= bounds["maximum"]
            and all(
                bounds["minimum"] <= record[key] <= bounds["maximum"]
                for record in (first, last, selected)
            ),
            "invalid bounded exact-eight subset metric extrema",
        )
    _require(
        selected["largest_exact_component_voxel_count"]
        == extrema["largest_exact_component_voxel_count"]["maximum"]
        and (feasibility["positive_exact_common_voxel_record_count"] > 0)
        is (extrema["exact_common_voxel_count"]["maximum"] > 0)
        and (feasibility["positive_largest_exact_component_record_count"] > 0)
        is (extrema["largest_exact_component_voxel_count"]["maximum"] > 0),
        "bounded exact-eight subset winner/extrema changed",
    )
    return value


def frame_zero_view_diagnostics_sha256(
    diagnostics: Sequence[Mapping[str, Any]],
    *,
    policy_id: str = FRAME_ZERO_CAMERA_SELECTION_POLICY_ID,
) -> str:
    """Bind all attempted views to the source-only v2 selection policy."""

    return hashlib.sha256(
        _canonical_bytes(
            {
                "policy_id": policy_id,
                "view_diagnostics": list(diagnostics),
            }
        )
    ).hexdigest()


def _case_parts(case_name: str) -> tuple[str, int]:
    match = _CASE_PATTERN.fullmatch(case_name)
    _require(match is not None, "invalid Deform360 case name")
    return match.group("object"), int(match.group("episode"))


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _HEX_SHA256.fullmatch(value) is not None


def validate_generic_held_lock(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only the generic fields shared with the held protocol agent.

    The builder intentionally does not import the held agent's lock/seal module.
    Additional lock fields remain checksum-bound but are not interpreted here.
    """

    _require(payload.get("schema_version") == 1, "unsupported held lock schema")
    _require(
        payload.get("artifact_kind") == HELD_LOCK_ARTIFACT_KIND,
        "unexpected held lock artifact kind",
    )
    _require(payload.get("protocol_id") == HELD_PROTOCOL_ID, "held protocol changed")
    observed = artifact_sha256(payload)
    _require(payload.get("artifact_sha256") == observed, "held lock checksum mismatch")
    whitelist = payload.get("case_whitelist")
    _require(isinstance(whitelist, list), "held lock lacks case_whitelist")
    _require(tuple(whitelist) == HELD_TARGET_CASES_V1, "held target whitelist changed")
    cohort = payload.get("cohort")
    _require(isinstance(cohort, list) and len(cohort) == 6, "held cohort changed")
    cohort_names = [
        record.get("case_name") for record in cohort if isinstance(record, Mapping)
    ]
    _require(cohort_names == list(HELD_TARGET_CASES_V1), "held cohort order changed")
    _require(payload.get("update_frames") == [19, 38, 57], "held update frames changed")
    _require(payload.get("frame_count") == 76, "held frame count changed")
    stage = payload.get("stage")
    _require(stage in {"calibration", "confirmation"}, "unknown held lock stage")
    if stage == "calibration":
        _require(
            payload.get("confirmation_access_authorized") is False,
            "calibration lock authorized confirmation",
        )
        _require(
            payload.get("parent_calibration_lock") is None,
            "calibration lock has a parent",
        )
        _require(
            payload.get("calibration_gate_evidence") is None,
            "calibration lock already contains gate evidence",
        )
    else:
        _require(
            payload.get("confirmation_access_authorized") is True,
            "promoted lock does not authorize confirmation",
        )
        _require(
            isinstance(payload.get("parent_calibration_lock"), Mapping),
            "promoted lock lacks its calibration parent",
        )
        _require(
            isinstance(payload.get("calibration_gate_evidence"), Mapping),
            "promoted lock lacks calibration gate evidence",
        )
    calibration = payload.get("calibration_case_whitelist")
    _require(
        isinstance(calibration, list)
        and APPROVED_CALIBRATION_SMOKE_CASE in calibration,
        "held lock does not authorize the calibration smoke case",
    )
    _require(
        not set(whitelist) & set(calibration), "held/calibration whitelists overlap"
    )
    for case_name in [*whitelist, *calibration]:
        object_id, _ = _case_parts(str(case_name))
        _require(
            object_id.split("-", 1)[0] not in ABSOLUTELY_FORBIDDEN_OBJECT_PREFIXES,
            "lock names an absolutely forbidden object",
        )
    bindings = payload.get("immutable_bindings")
    _require(isinstance(bindings, Mapping), "held lock lacks immutable bindings")
    _require(
        all(
            isinstance(key, str) and _valid_sha256(value)
            for key, value in bindings.items()
        ),
        "held lock contains an invalid immutable binding",
    )
    _require(
        _valid_sha256(bindings.get("frame_zero_default_config")),
        "held lock lacks the frame-zero configuration binding",
    )
    return {
        "passed": True,
        "artifact_sha256": observed,
        "target_case_count": len(whitelist),
        "calibration_case_count": len(calibration),
    }


def load_generic_held_lock(path: str | Path) -> dict[str, Any]:
    """Load a frame-zero lock through the complete held-protocol validator.

    In particular, a confirmation lock is not trusted merely because it is a
    self-consistent JSON object with non-empty parent/evidence mappings.  The
    full validator recursively replays the calibration GO decision before any
    confirmation payload can be opened.
    """

    # The held protocol does not import this module, so this local import is
    # cycle-free while keeping the future-blind builder usable in isolation.
    from .deform360_held_protocol import load_held_protocol_lock

    payload = load_held_protocol_lock(path)
    validate_generic_held_lock(payload)
    return payload


def _git_stdout(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
    ).stdout


def _literal_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _verify_clean_pinned_git_runtime(
    repository: Path,
    bindings: Mapping[str, str],
    *,
    prefix: str,
    expected_revision: str,
) -> dict[str, str]:
    """Verify the checked-out bytes, not only the name of ``HEAD``."""

    try:
        revision = _git_stdout(repository, "rev-parse", "HEAD").decode().strip()
        commit_object = _git_stdout(repository, "cat-file", "commit", "HEAD")
        tree_lines = _git_stdout(
            repository, "ls-tree", "-r", "--full-tree", "HEAD"
        ).splitlines()
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "diff",
                "--quiet",
                "--ignore-submodules",
                "--",
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "diff",
                "--cached",
                "--quiet",
                "--ignore-submodules",
                "--",
            ],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            f"{prefix} repository is unavailable or has dirty tracked files"
        ) from error
    tree_manifest = b"".join(line + b"\n" for line in sorted(tree_lines))
    observed = {
        "revision_literal": _literal_sha256(revision),
        "commit_object": hashlib.sha256(commit_object).hexdigest(),
        "git_tree_manifest": hashlib.sha256(tree_manifest).hexdigest(),
    }
    expected = {
        "revision_literal": bindings[f"{prefix}_revision_literal"],
        "commit_object": bindings[f"{prefix}_commit_object"],
        "git_tree_manifest": bindings[f"{prefix}_git_tree_manifest"],
    }
    _require(revision == expected_revision, f"{prefix} repository revision mismatch")
    _require(observed == expected, f"{prefix} runtime differs from the held lock")
    return {"revision": revision, **observed}


def authorize_frame_zero_case(
    lock: Mapping[str, Any], case_name: str, *, role: str
) -> dict[str, Any]:
    validate_generic_held_lock(lock)
    object_id, episode_id = _case_parts(case_name)
    prefix = object_id.split("-", 1)[0]
    _require(prefix not in ABSOLUTELY_FORBIDDEN_OBJECT_PREFIXES, "object is forbidden")
    _require(role in {"calibration", "confirmation"}, "unknown frame-zero role")
    if role == "calibration":
        _require(
            lock.get("stage") == "calibration", "calibration requires the initial lock"
        )
        _require(
            case_name in lock["calibration_case_whitelist"],
            "case is not authorized for calibration",
        )
        _require(
            case_name not in lock["case_whitelist"], "confirmation used as calibration"
        )
    else:
        _require(
            lock.get("stage") == "confirmation"
            and lock.get("confirmation_access_authorized") is True,
            "confirmation requires a promoted held lock",
        )
        _require(case_name in lock["case_whitelist"], "case is not a held confirmation")
        _require(
            case_name not in lock["calibration_case_whitelist"],
            "calibration used as confirmation",
        )
    return {
        "case_name": case_name,
        "object_id": object_id,
        "episode_id": episode_id,
        "role": role,
        "lock_artifact_sha256": lock["artifact_sha256"],
    }


def reject_future_derived_input(path: str | Path, *, purpose: str) -> Path:
    """Fail closed on HDF5 and conventional future-derived asset names."""

    candidate = Path(path)
    lowered = candidate.as_posix().lower()
    _require(
        candidate.suffix.lower() not in _FORBIDDEN_INPUT_SUFFIXES,
        f"{purpose} may not use an HDF5 input",
    )
    _require(
        not any(token in lowered for token in _FORBIDDEN_DERIVATION_TOKENS),
        f"{purpose} appears future-derived",
    )
    return candidate


@dataclass(frozen=True)
class FrameZeroAssetConfig:
    """Outcome-free geometry and QA choices fixed before a case is decoded."""

    reference_camera: str = "brics-odroid-001_cam0"
    minimum_camera_count: int = 8
    cube_half_extent_m: float = 0.5
    requested_voxel_size_m: float = 0.008
    maximum_grid_point_count: int = 1_500_000
    consensus_fraction_of_peak: float = 0.55
    minimum_consensus_votes: int = 8
    minimum_hull_point_count: int = 1_000
    minimum_largest_component_fraction: float = 0.50
    object_point_count: int = 10_000
    depth_splat_radius_pixels: int = 2
    minimum_median_depth_mask_coverage: float = 0.10
    minimum_median_hull_mask_containment: float = 0.60
    action_window_length_frames: int = 81
    prediction_frame_count: int = 76
    action_candidate_first_frame: int = 8
    action_candidate_stride_frames: int = 6
    rng_seed: int = 0
    sam2: DeformableObjectSam2MaskConfig = field(
        default_factory=DeformableObjectSam2MaskConfig
    )

    def __post_init__(self) -> None:
        _require(self.minimum_camera_count >= 3, "too few frame-zero cameras")
        _require(self.cube_half_extent_m > 0.0, "cube extent must be positive")
        _require(self.requested_voxel_size_m > 0.0, "voxel size must be positive")
        _require(self.maximum_grid_point_count >= 1_000, "grid cap is too small")
        _require(
            0.0 < self.consensus_fraction_of_peak <= 1.0,
            "invalid consensus fraction",
        )
        _require(self.minimum_consensus_votes >= 2, "consensus needs two cameras")
        _require(self.minimum_hull_point_count >= 8, "minimum hull is too small")
        _require(
            0.0 < self.minimum_largest_component_fraction <= 1.0,
            "invalid component fraction",
        )
        _require(self.object_point_count >= 128, "object point count is too small")
        _require(self.depth_splat_radius_pixels >= 0, "negative depth splat radius")
        _require(
            self.action_window_length_frames > self.prediction_frame_count >= 2,
            "invalid action/prediction frame counts",
        )
        _require(
            self.action_window_length_frames - self.prediction_frame_count == 5,
            "held protocol must skip exactly five tracking-tail frames",
        )
        _require(self.action_candidate_first_frame >= 0, "negative action candidate")
        _require(self.action_candidate_stride_frames >= 1, "invalid action stride")


class FrameZeroMaskRuntime(Protocol):
    model_id: str

    def generate(self, rgb: np.ndarray) -> list[Mapping[str, Any]]: ...


class FrameZeroSemanticGateRuntime(Protocol):
    """Lazy seam used only after all three reference-anchored paths fail."""

    model_id: str

    def prepare_proposals(
        self,
        proposals_by_camera: Mapping[str, Sequence[Mapping[str, Any]]],
        rgb_by_camera: Mapping[str, np.ndarray],
        intrinsics_by_camera: Mapping[str, np.ndarray],
        camera_to_world_by_camera: Mapping[str, np.ndarray],
        selected_action_arrays: Mapping[str, np.ndarray],
    ) -> tuple[
        dict[str, list[dict[str, Any]]],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, Any],
    ]: ...

    def subtract_robot(
        self,
        selected_masks: Mapping[str, np.ndarray],
        exact_robot_masks: Mapping[str, np.ndarray],
        dilated_robot_masks: Mapping[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]: ...

    def evaluate(
        self,
        rgb_by_camera: Mapping[str, np.ndarray],
        selected_masks: Mapping[str, np.ndarray],
        *,
        object_id: str,
        selected_proposals: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]: ...


class PinnedFrameZeroSam2Runtime:
    """Pinned SAM 2.1 image-only runtime with no video propagation model."""

    def __init__(
        self,
        repository: str | Path,
        checkpoint: str | Path,
        *,
        config: DeformableObjectSam2MaskConfig,
        immutable_bindings: Mapping[str, str],
        device: str = "cuda",
    ) -> None:
        self.repository = Path(repository).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.config = config
        self.device = device
        _require(self.repository.is_dir(), "SAM2 repository does not exist")
        _require(self.checkpoint.is_file(), "SAM2 checkpoint does not exist")
        self.git_binding = _verify_clean_pinned_git_runtime(
            self.repository,
            immutable_bindings,
            prefix="sam2",
            expected_revision=PINNED_SAM2_COMMIT,
        )
        _require(
            _sha256_file(self.checkpoint)
            == PINNED_SAM2_CHECKPOINT_SHA256
            == immutable_bindings["sam2_checkpoint"],
            "SAM2 checkpoint checksum mismatch",
        )
        # ``PINNED_SAM2_MODEL_CONFIG`` is the Hydra identifier expected by
        # build_sam2.  The actual tracked file lives below the Python package.
        self.model_config_path = (
            self.repository / "sam2" / PINNED_SAM2_MODEL_CONFIG
        ).resolve()
        _require(
            self.model_config_path.is_file()
            and not self.model_config_path.is_symlink()
            and self.model_config_path.is_relative_to(self.repository),
            "SAM2 model config file is missing from the pinned repository",
        )
        _require(
            _sha256_file(self.model_config_path)
            == immutable_bindings["sam2_model_config"],
            "SAM2 model config differs from the held lock",
        )
        if str(self.repository) not in sys.path:
            sys.path.insert(0, str(self.repository))
        try:
            import torch

            automatic_mask_generator = importlib.import_module(
                "sam2.automatic_mask_generator"
            )
            build_sam_module = importlib.import_module("sam2.build_sam")
        except ImportError as error:  # pragma: no cover - GPU integration
            raise RuntimeError(
                "pinned SAM2 runtime dependencies are unavailable"
            ) from error
        for module in (automatic_mask_generator, build_sam_module):
            module_path = Path(str(module.__file__)).resolve()
            _require(
                module_path.is_relative_to(self.repository),
                "imported SAM2 runtime comes from another repository",
            )
        SAM2AutomaticMaskGenerator = automatic_mask_generator.SAM2AutomaticMaskGenerator
        build_sam2 = build_sam_module.build_sam2
        self._torch = torch
        model = build_sam2(
            PINNED_SAM2_MODEL_CONFIG,
            str(self.checkpoint),
            device=device,
            apply_postprocessing=False,
        )
        self._generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=config.points_per_side,
            points_per_batch=config.points_per_batch,
            pred_iou_thresh=config.predicted_iou_threshold,
            stability_score_thresh=config.stability_score_threshold,
            min_mask_region_area=config.minimum_mask_region_area,
        )
        self.model_id = (
            "bayesian_phystwin/frame-zero-sam2.1-small-automatic-v1@"
            f"{PINNED_SAM2_COMMIT[:12]}"
        )

    def generate(self, rgb: np.ndarray) -> list[Mapping[str, Any]]:
        with (
            self._torch.inference_mode(),
            self._torch.autocast(
                "cuda",
                dtype=self._torch.bfloat16,
                enabled=self.device.startswith("cuda"),
            ),
        ):
            return self._generator.generate(np.asarray(rgb, dtype=np.uint8))

    def close(self) -> None:
        self._generator = None
        if self.device.startswith("cuda"):
            self._torch.cuda.empty_cache()


def decode_exact_frame_zero(
    video_path: str | Path, *, source_aligned_frame_index: int = 0
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode exactly the selected robot-window frame zero and audit alignment.

    ``source_aligned_frame_index`` is selected from known aligned realized robot
    kinematics only.  The returned RGB material has selected-window index zero;
    no later object frame is decoded, hashed, or passed to the segmentation
    model.
    """

    path = reject_future_derived_input(video_path, purpose="camera video")
    _require(path.suffix.lower() in {".mp4", ".mov", ".mkv"}, "unsupported video input")
    path = validate_regular_file_nofollow(path, label="camera video")
    _require(source_aligned_frame_index >= 0, "source frame index is negative")
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - server integration
        raise RuntimeError("OpenCV is required for frame-zero decoding") from error
    capture = cv2.VideoCapture(str(path))
    try:
        if source_aligned_frame_index:
            positioned = capture.set(
                cv2.CAP_PROP_POS_FRAMES, source_aligned_frame_index
            )
            _require(bool(positioned), "cannot seek to action-window frame zero")
        ok, bgr = capture.read()
    finally:
        capture.release()
    _require(bool(ok) and bgr is not None, f"cannot decode frame zero: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, {
        "path": str(path),
        "decoded_frame_count": 1,
        "maximum_rgb_frame_read": 0,
        "action_window_frame_index": 0,
        "source_aligned_frame_index": source_aligned_frame_index,
        "decoded_rgb_sha256": _sha256_array(rgb),
        "whole_file_hashed_or_read": False,
    }


def _select_reference_mask(
    rgb: np.ndarray,
    annotations: Sequence[Mapping[str, Any]],
    config: DeformableObjectSam2MaskConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for index, annotation in enumerate(annotations):
        mask = np.asarray(annotation["segmentation"], dtype=bool)
        diagnostic = deformable_object_mask_candidate_diagnostics(rgb, mask, config)
        predicted_iou = float(annotation.get("predicted_iou", 1.0))
        stability = float(annotation.get("stability_score", 1.0))
        eligible = bool(
            diagnostic["eligible"]
            and predicted_iou >= config.predicted_iou_threshold
            and stability >= config.stability_score_threshold
        )
        score = (
            float(diagnostic["score"]) * math.sqrt(max(0.0, predicted_iou * stability))
            if eligible
            else -1.0
        )
        diagnostic.update(
            {
                "eligible": eligible,
                "score": score,
                "candidate_index": index,
                "predicted_iou": predicted_iou,
                "stability_score": stability,
            }
        )
        candidates.append((float(diagnostic["score"]), index, diagnostic))
    eligible = [candidate for candidate in candidates if candidate[0] >= 0.0]
    _require(bool(eligible), "SAM2 found no eligible reference mask")
    _, selected_index, selected = max(eligible, key=lambda item: (item[0], -item[1]))
    return np.asarray(annotations[selected_index]["segmentation"], dtype=bool), {
        "automatic_candidate_count": len(annotations),
        "eligible_candidate_count": len(eligible),
        "rejected_candidate_count": len(annotations) - len(eligible),
        "rejection_counts": {
            "mask_threshold": len(annotations) - len(eligible),
            "reference_appearance_threshold": 0,
            "total": len(annotations) - len(eligible),
        },
        "maximum_reference_appearance_similarity": 1.0,
        "view_selected": True,
        "abstained": False,
        "abstention_reason": None,
        "selected": selected,
    }


def _select_reference_consistent_mask(
    rgb: np.ndarray,
    annotations: Sequence[Mapping[str, Any]],
    reference_descriptor: Mapping[str, Any],
    config: DeformableObjectSam2MaskConfig,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    mask_threshold_rejections = 0
    appearance_threshold_rejections = 0
    for index, annotation in enumerate(annotations):
        mask = np.asarray(annotation["segmentation"], dtype=bool)
        diagnostic = deformable_object_mask_candidate_diagnostics(rgb, mask, config)
        predicted_iou = float(annotation.get("predicted_iou", 1.0))
        stability = float(annotation.get("stability_score", 1.0))
        basic = bool(
            diagnostic["eligible"]
            and predicted_iou >= config.predicted_iou_threshold
            and stability >= config.stability_score_threshold
        )
        similarity = (
            mask_appearance_similarity(
                reference_descriptor, mask_appearance_descriptor(rgb, mask)
            )
            if basic
            else {
                "combined": 0.0,
                "hs_histogram_intersection": 0.0,
                "lab_similarity": 0.0,
                "shape_similarity": 0.0,
            }
        )
        eligible = bool(
            basic
            and similarity["combined"] >= config.minimum_reference_appearance_similarity
        )
        mask_threshold_rejections += int(not basic)
        appearance_threshold_rejections += int(
            basic
            and similarity["combined"] < config.minimum_reference_appearance_similarity
        )
        score = (
            similarity["combined"] ** 3
            * similarity["shape_similarity"] ** 4
            * math.sqrt(float(diagnostic["area_fraction"]))
            * math.sqrt(max(0.0, predicted_iou * stability))
            if eligible
            else -1.0
        )
        diagnostic.update(
            {
                "eligible": eligible,
                "score": float(score),
                "candidate_index": index,
                "predicted_iou": predicted_iou,
                "stability_score": stability,
                "reference_appearance": similarity,
            }
        )
        candidates.append((float(score), index, diagnostic))
    eligible_candidates = [candidate for candidate in candidates if candidate[0] >= 0.0]
    maximum_similarity = max(
        (
            float(candidate[2]["reference_appearance"]["combined"])
            for candidate in candidates
        ),
        default=0.0,
    )
    summary: dict[str, Any] = {
        "automatic_candidate_count": len(annotations),
        "eligible_candidate_count": len(eligible_candidates),
        "rejected_candidate_count": len(annotations) - len(eligible_candidates),
        "rejection_counts": {
            "mask_threshold": mask_threshold_rejections,
            "reference_appearance_threshold": appearance_threshold_rejections,
            "total": len(annotations) - len(eligible_candidates),
        },
        "maximum_reference_appearance_similarity": maximum_similarity,
    }
    if not eligible_candidates:
        if not annotations:
            abstention_reason = _NO_AUTOMATIC_CANDIDATES
        elif mask_threshold_rejections == len(annotations):
            abstention_reason = _NO_MASK_THRESHOLD_CANDIDATES
        else:
            abstention_reason = _NO_REFERENCE_CONSISTENT_CANDIDATES
        summary.update(
            {
                "view_selected": False,
                "abstained": True,
                "abstention_reason": abstention_reason,
                "selected": None,
            }
        )
        return None, summary
    _, selected_index, selected = max(
        eligible_candidates, key=lambda item: (item[0], -item[1])
    )
    summary.update(
        {
            "view_selected": True,
            "abstained": False,
            "abstention_reason": None,
            "selected": selected,
        }
    )
    return np.asarray(annotations[selected_index]["segmentation"], dtype=bool), summary


def segment_frame_zero_views(
    rgb_by_camera: Mapping[str, np.ndarray],
    runtime: FrameZeroMaskRuntime,
    *,
    reference_camera: str,
    config: DeformableObjectSam2MaskConfig,
    proposal_sink: dict[str, list[Mapping[str, Any]]] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    cameras = tuple(sorted(rgb_by_camera))
    _require(reference_camera in cameras, "reference camera is unavailable")
    reference_rgb = np.asarray(rgb_by_camera[reference_camera], dtype=np.uint8)
    reference_annotations = list(runtime.generate(reference_rgb))
    if proposal_sink is not None:
        _require(not proposal_sink, "proposal sink must initially be empty")
        proposal_sink[reference_camera] = reference_annotations
    reference_mask, reference_diagnostic = _select_reference_mask(
        reference_rgb, reference_annotations, config
    )
    reference_descriptor = mask_appearance_descriptor(reference_rgb, reference_mask)
    masks = {reference_camera: reference_mask}
    diagnostics = [
        {
            "camera": reference_camera,
            "initialization": "automatic-reference-frame-zero",
            **reference_diagnostic,
        }
    ]
    for camera in cameras:
        if camera == reference_camera:
            continue
        rgb = np.asarray(rgb_by_camera[camera], dtype=np.uint8)
        annotations = list(runtime.generate(rgb))
        if proposal_sink is not None:
            proposal_sink[camera] = annotations
        mask, diagnostic = _select_reference_consistent_mask(
            rgb, annotations, reference_descriptor, config
        )
        if mask is not None:
            masks[camera] = mask
        diagnostics.append(
            {
                "camera": camera,
                "initialization": "reference-appearance-frame-zero",
                "reference_camera": reference_camera,
                **diagnostic,
            }
        )
    diagnostics.sort(key=lambda item: str(item["camera"]))
    return masks, diagnostics


def _mask_set_sha256(masks_by_camera: Mapping[str, np.ndarray]) -> str:
    return hashlib.sha256(
        _canonical_bytes(
            {
                camera: _sha256_array(np.asarray(masks_by_camera[camera], dtype=bool))
                for camera in sorted(masks_by_camera)
            }
        )
    ).hexdigest()


def _basic_mask_candidate_records(
    rgb: np.ndarray,
    annotations: Sequence[Mapping[str, Any]],
    config: DeformableObjectSam2MaskConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, annotation in enumerate(annotations):
        mask = np.asarray(annotation["segmentation"], dtype=bool)
        diagnostic = deformable_object_mask_candidate_diagnostics(rgb, mask, config)
        predicted_iou = float(annotation.get("predicted_iou", 1.0))
        stability = float(annotation.get("stability_score", 1.0))
        basic = bool(
            diagnostic["eligible"]
            and predicted_iou >= config.predicted_iou_threshold
            and stability >= config.stability_score_threshold
        )
        local_score = (
            float(diagnostic["score"]) * math.sqrt(max(0.0, predicted_iou * stability))
            if basic
            else -1.0
        )
        diagnostic.update(
            {
                "eligible": basic,
                "score": local_score,
                "candidate_index": index,
                "predicted_iou": predicted_iou,
                "stability_score": stability,
            }
        )
        records.append(
            {
                "candidate_index": index,
                "mask": mask,
                "mask_sha256": _sha256_array(mask),
                "basic_eligible": basic,
                "local_score": local_score,
                "descriptor": (
                    mask_appearance_descriptor(rgb, mask) if basic else None
                ),
                "diagnostic": diagnostic,
            }
        )
    return records


def _reference_consistent_candidate_records(
    basic_records: Sequence[Mapping[str, Any]],
    reference_descriptor: Mapping[str, Any],
    config: DeformableObjectSam2MaskConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in basic_records:
        basic = bool(source["basic_eligible"])
        similarity = (
            mask_appearance_similarity(reference_descriptor, source["descriptor"])
            if basic
            else {
                "combined": 0.0,
                "hs_histogram_intersection": 0.0,
                "lab_similarity": 0.0,
                "shape_similarity": 0.0,
            }
        )
        eligible = bool(
            basic
            and similarity["combined"] >= config.minimum_reference_appearance_similarity
        )
        diagnostic = dict(source["diagnostic"])
        score = (
            similarity["combined"] ** 3
            * similarity["shape_similarity"] ** 4
            * math.sqrt(float(diagnostic["area_fraction"]))
            * math.sqrt(
                max(
                    0.0,
                    float(diagnostic["predicted_iou"])
                    * float(diagnostic["stability_score"]),
                )
            )
            if eligible
            else -1.0
        )
        diagnostic.update(
            {
                "eligible": eligible,
                "score": float(score),
                "reference_appearance": similarity,
            }
        )
        records.append(
            {
                **source,
                "eligible": eligible,
                "appearance_score": float(score),
                "reference_appearance": similarity,
                "diagnostic": diagnostic,
            }
        )
    return records


def _proposal_inventory_sha256(
    basic_by_camera: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    payload = {
        camera: [
            {
                "candidate_index": int(record["candidate_index"]),
                "mask_sha256": str(record["mask_sha256"]),
                "basic_eligible": bool(record["basic_eligible"]),
                "local_score": float(record["local_score"]),
                "predicted_iou": float(record["diagnostic"]["predicted_iou"]),
                "stability_score": float(record["diagnostic"]["stability_score"]),
            }
            for record in records
        ]
        for camera, records in sorted(basic_by_camera.items())
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _selection_view_diagnostic(
    camera: str,
    records: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any] | None,
    *,
    reference_camera: str,
    geometry_inlier_selection: Mapping[str, Any],
) -> dict[str, Any]:
    reference = camera == reference_camera
    automatic_count = len(records)
    if reference:
        eligible_records = [record for record in records if record["basic_eligible"]]
        mask_rejections = automatic_count - len(eligible_records)
        appearance_rejections = 0
        maximum_similarity = 1.0
    else:
        eligible_records = [record for record in records if record["eligible"]]
        mask_rejections = sum(not record["basic_eligible"] for record in records)
        appearance_rejections = sum(
            bool(record["basic_eligible"]) and not bool(record["eligible"])
            for record in records
        )
        maximum_similarity = max(
            (float(record["reference_appearance"]["combined"]) for record in records),
            default=0.0,
        )
    rejected_count = automatic_count - len(eligible_records)
    diagnostic: dict[str, Any] = {
        "camera": camera,
        "initialization": (
            "automatic-reference-frame-zero"
            if reference
            else "reference-appearance-frame-zero"
        ),
        "automatic_candidate_count": automatic_count,
        "eligible_candidate_count": len(eligible_records),
        "rejected_candidate_count": rejected_count,
        "rejection_counts": {
            "mask_threshold": int(mask_rejections),
            "reference_appearance_threshold": int(appearance_rejections),
            "total": rejected_count,
        },
        "maximum_reference_appearance_similarity": maximum_similarity,
        "geometry_inlier_selection": dict(geometry_inlier_selection),
    }
    if not reference:
        diagnostic["reference_camera"] = reference_camera
    if selected is not None:
        diagnostic.update(
            {
                "view_selected": True,
                "abstained": False,
                "abstention_reason": None,
                "selected": dict(selected["diagnostic"]),
            }
        )
        return diagnostic
    if (
        geometry_inlier_selection.get("candidate_index") is not None
        and geometry_inlier_selection.get("retained") is False
    ):
        reason = _COMMON_INLIER_ABSTENTION_REASON
    elif not records:
        reason = _NO_AUTOMATIC_CANDIDATES
    elif mask_rejections == automatic_count:
        reason = _NO_MASK_THRESHOLD_CANDIDATES
    else:
        reason = _NO_REFERENCE_CONSISTENT_CANDIDATES
    diagnostic.update(
        {
            "view_selected": False,
            "abstained": True,
            "abstention_reason": reason,
            "selected": None,
        }
    )
    return diagnostic


def _common_voxel_mask_assignment(
    rgb_by_camera: Mapping[str, np.ndarray],
    proposals_by_camera: Mapping[str, Sequence[Mapping[str, Any]]],
    intrinsics_by_camera: Mapping[str, np.ndarray],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    *,
    reference_camera: str,
    config: FrameZeroAssetConfig,
    reference_optional: bool = False,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    """Select one coherent proposal per camera using a strict common voxel.

    The default path is the original fixed-reference contract.  The optional
    mode changes only the three reference-hit intersections and the exhaustive
    exact-eight subset constraint; the reference still supplies the frozen
    appearance-conditioning seeds.
    """

    cameras = tuple(sorted(rgb_by_camera))
    _require(
        cameras == tuple(sorted(proposals_by_camera)),
        "proposal/RGB camera sets differ",
    )
    _require(reference_camera in cameras, "common search reference is unavailable")
    _require(
        set(cameras) <= set(intrinsics_by_camera)
        and set(cameras) <= set(camera_to_world_by_camera),
        "common search calibration is incomplete",
    )
    basic_by_camera = {
        camera: _basic_mask_candidate_records(
            np.asarray(rgb_by_camera[camera], dtype=np.uint8),
            proposals_by_camera[camera],
            config.sam2,
        )
        for camera in cameras
    }
    ranked_reference_candidates = sorted(
        (
            record
            for record in basic_by_camera[reference_camera]
            if record["basic_eligible"]
        ),
        key=lambda record: (-float(record["local_score"]), record["candidate_index"]),
    )
    reference_candidates = ranked_reference_candidates[:_FALLBACK_REFERENCE_SEED_COUNT]
    _require_geometry(
        bool(reference_candidates),
        "common search has no eligible semantic reference seed",
    )

    centers = np.stack(
        [np.asarray(camera_to_world_by_camera[camera])[:3, 3] for camera in cameras]
    )
    grid_center = np.mean(centers, axis=0)
    minimum = grid_center - config.cube_half_extent_m
    maximum = grid_center + config.cube_half_extent_m
    axes = [
        np.linspace(minimum[axis], maximum[axis], _FALLBACK_COMMON_GRID_AXIS_COUNT)
        for axis in range(3)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    grid_shape = [_FALLBACK_COMMON_GRID_AXIS_COUNT] * 3
    grid_spacing = [float(axis[1] - axis[0]) for axis in axes]
    hit_cache: dict[tuple[str, int], np.ndarray] = {}
    for camera in cameras:
        for record in basic_by_camera[camera]:
            hits, _in_bounds = _point_mask_hits(
                grid,
                record["mask"],
                intrinsics_by_camera[camera],
                camera_to_world_by_camera[camera],
            )
            hit_cache[(camera, int(record["candidate_index"]))] = hits

    seed_evaluations: list[dict[str, Any]] = []
    seed_audit: list[dict[str, Any]] = []
    for rank, reference_record in enumerate(reference_candidates):
        reference_descriptor = reference_record["descriptor"]
        records_by_camera: dict[str, list[dict[str, Any]]] = {
            reference_camera: [reference_record]
        }
        for camera in cameras:
            if camera == reference_camera:
                continue
            records_by_camera[camera] = _reference_consistent_candidate_records(
                basic_by_camera[camera], reference_descriptor, config.sam2
            )
        support = np.zeros(len(grid), dtype=np.uint16)
        eligible_camera_count = 0
        for camera in cameras:
            eligible = (
                records_by_camera[camera]
                if camera == reference_camera
                else [
                    record for record in records_by_camera[camera] if record["eligible"]
                ]
            )
            if not eligible:
                continue
            eligible_camera_count += 1
            camera_union = np.zeros(len(grid), dtype=bool)
            for record in eligible:
                camera_union |= hit_cache[(camera, int(record["candidate_index"]))]
            support += camera_union.astype(np.uint16)
        reference_hits = hit_cache[
            (reference_camera, int(reference_record["candidate_index"]))
        ]
        feasible = support >= _FALLBACK_STRICT_CONSENSUS_VOTES
        if not reference_optional:
            feasible &= reference_hits
        feasible_ids = np.flatnonzero(feasible)
        component_ids = np.empty(0, dtype=np.int64)
        largest_count = 0
        component_count = 0
        local_feasible_count = 0
        local_largest_count = 0
        local_component_count = 0
        local_peak = 0
        local_grid_shape: list[int] | None = None
        local_grid_point_count = 0
        local_grid_coarsened = False
        local_feasible_sha256: str | None = None
        if len(feasible_ids):
            keep, _surface, components = _largest_grid_component(
                grid[feasible],
                bounds_minimum=minimum,
                bounds_maximum=maximum,
                grid_shape=grid_shape,
            )
            component_ids = feasible_ids[keep]
            largest_count = int(components["largest_component_point_count"])
            component_count = int(components["component_count"])
            seed_local_minimum, seed_local_maximum, _seed_local_bounds = (
                _local_refinement_bounds(
                    grid[component_ids],
                    global_minimum_world_m=minimum,
                    global_maximum_world_m=maximum,
                    coarse_axis_spacing_m=grid_spacing,
                )
            )
            seed_local_grid, seed_local_grid_diagnostics = regular_grid_in_bounds(
                seed_local_minimum,
                seed_local_maximum,
                requested_voxel_size_m=(_FALLBACK_COMMON_LOCAL_REQUESTED_VOXEL_SIZE_M),
                maximum_point_count=_FALLBACK_MAXIMUM_LOCAL_GRID_POINT_COUNT,
            )
            local_grid_shape = [
                int(value) for value in seed_local_grid_diagnostics["grid_shape"]
            ]
            local_grid_point_count = len(seed_local_grid)
            local_grid_coarsened = bool(
                seed_local_grid_diagnostics["coarsened_for_grid_cap"]
            )
            if not local_grid_coarsened:
                seed_local_support = np.zeros(len(seed_local_grid), dtype=np.uint16)
                seed_local_reference_hits, _in_bounds = _point_mask_hits(
                    seed_local_grid,
                    reference_record["mask"],
                    intrinsics_by_camera[reference_camera],
                    camera_to_world_by_camera[reference_camera],
                )
                for camera in cameras:
                    eligible = (
                        records_by_camera[camera]
                        if camera == reference_camera
                        else [
                            record
                            for record in records_by_camera[camera]
                            if record["eligible"]
                        ]
                    )
                    if not eligible:
                        continue
                    camera_union = np.zeros(len(seed_local_grid), dtype=bool)
                    for record in eligible:
                        hits, _in_bounds = _point_mask_hits(
                            seed_local_grid,
                            record["mask"],
                            intrinsics_by_camera[camera],
                            camera_to_world_by_camera[camera],
                        )
                        camera_union |= hits
                    seed_local_support += camera_union.astype(np.uint16)
                seed_local_feasible = (
                    seed_local_support >= _FALLBACK_STRICT_CONSENSUS_VOTES
                )
                if not reference_optional:
                    seed_local_feasible &= seed_local_reference_hits
                seed_local_feasible_ids = np.flatnonzero(seed_local_feasible)
                local_feasible_count = len(seed_local_feasible_ids)
                local_feasible_sha256 = _sha256_array(seed_local_feasible)
                if local_feasible_count:
                    local_keep, _local_surface, local_components = (
                        _largest_grid_component(
                            seed_local_grid[seed_local_feasible],
                            bounds_minimum=seed_local_minimum,
                            bounds_maximum=seed_local_maximum,
                            grid_shape=local_grid_shape,
                        )
                    )
                    local_component_ids = seed_local_feasible_ids[local_keep]
                    local_largest_count = int(
                        local_components["largest_component_point_count"]
                    )
                    local_component_count = int(local_components["component_count"])
                    local_peak = int(
                        np.max(seed_local_support[local_component_ids], initial=0)
                    )
        peak = int(support.max(initial=0))
        objective = (
            local_largest_count,
            local_feasible_count,
            local_peak,
            largest_count,
            int(len(feasible_ids)),
            float(reference_record["local_score"]),
            -int(reference_record["candidate_index"]),
        )
        evaluation = {
            "rank": rank,
            "reference_record": reference_record,
            "records_by_camera": records_by_camera,
            "support": support,
            "feasible_ids": feasible_ids,
            "component_ids": component_ids,
            "local_largest_component_point_count": local_largest_count,
            "objective": objective,
        }
        seed_evaluations.append(evaluation)
        seed_audit.append(
            {
                "reference_seed_rank": rank,
                "reference_candidate_index": int(reference_record["candidate_index"]),
                "reference_mask_sha256": reference_record["mask_sha256"],
                "reference_local_score": float(reference_record["local_score"]),
                "eligible_camera_count": eligible_camera_count,
                "strict_feasible_voxel_count": int(len(feasible_ids)),
                "strict_feasible_component_count": component_count,
                "largest_strict_component_voxel_count": largest_count,
                "maximum_union_support_count": peak,
                "reference_anchor_hit_count": int(np.count_nonzero(reference_hits)),
                "local_grid_shape": local_grid_shape,
                "local_grid_point_count": local_grid_point_count,
                "local_grid_coarsened_for_cap": local_grid_coarsened,
                "local_strict_feasible_voxel_count": local_feasible_count,
                "local_strict_feasible_component_count": local_component_count,
                "local_largest_strict_component_voxel_count": local_largest_count,
                "local_maximum_union_support_count": local_peak,
                "local_strict_feasible_mask_sha256": local_feasible_sha256,
                "lexicographic_objective": list(objective),
                "strict_feasible_mask_sha256": _sha256_array(feasible),
            }
        )
    selected_evaluation = max(
        seed_evaluations, key=lambda evaluation: evaluation["objective"]
    )
    _require_geometry(
        len(selected_evaluation["component_ids"]) > 0
        and selected_evaluation["local_largest_component_point_count"] > 0,
        "common search found no reference-anchored strict-eight component",
    )
    component_ids = np.asarray(selected_evaluation["component_ids"], dtype=np.int64)
    support = np.asarray(selected_evaluation["support"], dtype=np.uint16)
    global_selected_seed_id = max(
        component_ids.tolist(), key=lambda index: (int(support[index]), -int(index))
    )
    component_mask = np.zeros(len(grid), dtype=bool)
    component_mask[component_ids] = True

    local_minimum, local_maximum, local_bounds = _local_refinement_bounds(
        grid[component_ids],
        global_minimum_world_m=minimum,
        global_maximum_world_m=maximum,
        coarse_axis_spacing_m=grid_spacing,
    )
    local_grid, local_grid_diagnostics = regular_grid_in_bounds(
        local_minimum,
        local_maximum,
        requested_voxel_size_m=_FALLBACK_COMMON_LOCAL_REQUESTED_VOXEL_SIZE_M,
        maximum_point_count=_FALLBACK_MAXIMUM_LOCAL_GRID_POINT_COUNT,
    )
    _require_geometry(
        not bool(local_grid_diagnostics["coarsened_for_grid_cap"]),
        "selected common local grid exceeds the exact-grid cap",
    )
    local_hit_cache: dict[tuple[str, int], np.ndarray] = {}
    local_support = np.zeros(len(local_grid), dtype=np.uint16)
    for camera in cameras:
        records = selected_evaluation["records_by_camera"][camera]
        eligible_records = (
            records
            if camera == reference_camera
            else [record for record in records if record["eligible"]]
        )
        if not eligible_records:
            continue
        camera_union = np.zeros(len(local_grid), dtype=bool)
        for record in eligible_records:
            cache_key = (camera, int(record["candidate_index"]))
            hits, _in_bounds = _point_mask_hits(
                local_grid,
                record["mask"],
                intrinsics_by_camera[camera],
                camera_to_world_by_camera[camera],
            )
            local_hit_cache[cache_key] = hits
            camera_union |= hits
        local_support += camera_union.astype(np.uint16)
    local_reference_hits = local_hit_cache[
        (
            reference_camera,
            int(selected_evaluation["reference_record"]["candidate_index"]),
        )
    ]
    local_feasible = local_support >= _FALLBACK_STRICT_CONSENSUS_VOTES
    if not reference_optional:
        local_feasible &= local_reference_hits
    local_feasible_ids = np.flatnonzero(local_feasible)
    _require_geometry(
        bool(len(local_feasible_ids)),
        "local common search found no strict-eight common voxel",
    )
    local_keep, _local_surface, local_components = _largest_grid_component(
        local_grid[local_feasible],
        bounds_minimum=local_minimum,
        bounds_maximum=local_maximum,
        grid_shape=local_grid_diagnostics["grid_shape"],
    )
    local_component_ids = local_feasible_ids[local_keep]
    local_selected_seed_id = max(
        local_component_ids.tolist(),
        key=lambda index: (int(local_support[index]), -int(index)),
    )
    local_component_mask = np.zeros(len(local_grid), dtype=bool)
    local_component_mask[local_component_ids] = True
    selected_candidate_by_camera: dict[str, Mapping[str, Any]] = {}
    candidate_metrics_by_camera: dict[str, dict[str, Any]] = {}
    for camera in cameras:
        records = selected_evaluation["records_by_camera"][camera]
        eligible_records = (
            records
            if camera == reference_camera
            else [record for record in records if record["eligible"]]
        )
        if not eligible_records:
            continue

        def candidate_objective(record: Mapping[str, Any]) -> tuple[Any, ...]:
            hits = local_hit_cache[(camera, int(record["candidate_index"]))]
            overlap = int(np.count_nonzero(hits & local_component_mask))
            hit_count = int(np.count_nonzero(hits))
            coverage = overlap / len(local_component_ids)
            precision = overlap / hit_count if hit_count else 0.0
            score = (
                float(record["local_score"])
                if camera == reference_camera
                else float(record["appearance_score"])
            )
            return (
                coverage,
                precision,
                int(hits[local_selected_seed_id]),
                score,
                -int(record["candidate_index"]),
            )

        selected = max(eligible_records, key=candidate_objective)
        objective = candidate_objective(selected)
        selected_candidate_by_camera[camera] = selected
        candidate_metrics_by_camera[camera] = {
            "seed_conditioned_candidate_count": len(eligible_records),
            "raw_component_hit_count": int(
                np.count_nonzero(
                    local_hit_cache[(camera, int(selected["candidate_index"]))]
                    & local_component_mask
                )
            ),
            "raw_component_coverage": float(objective[0]),
            "raw_component_precision": float(objective[1]),
            "hits_local_union_peak": bool(objective[2]),
            "semantic_score": float(objective[3]),
            "candidate_lexicographic_objective": list(objective),
        }
    _require_geometry(
        len(selected_candidate_by_camera) >= _FALLBACK_STRICT_CONSENSUS_VOTES
        and (reference_optional or reference_camera in selected_candidate_by_camera),
        (
            "reference-optional common search found too few eligible exact-eight "
            "candidates"
            if reference_optional
            else "common search found too few eligible exact-eight candidates"
        ),
    )

    nonreference_candidates = tuple(
        camera
        for camera in cameras
        if camera in selected_candidate_by_camera and camera != reference_camera
    )
    subset_audit = _ExactEightSubsetAuditAccumulator(
        bounded=HELD_PROTOCOL_ID == _HELD_V8_PROTOCOL_ID
    )
    best_subset: dict[str, Any] | None = None
    best_subset_key: tuple[Any, ...] | None = None
    subset_candidates = (
        tuple(sorted(selected_candidate_by_camera))
        if reference_optional
        else nonreference_candidates
    )
    subset_size = (
        _FALLBACK_STRICT_CONSENSUS_VOTES
        if reference_optional
        else _FALLBACK_STRICT_CONSENSUS_VOTES - 1
    )
    for candidate_subset in itertools.combinations(subset_candidates, subset_size):
        subset = (
            candidate_subset
            if reference_optional
            else (reference_camera, *candidate_subset)
        )
        exact_mask = np.logical_and.reduce(
            [
                local_hit_cache[
                    (
                        camera,
                        int(selected_candidate_by_camera[camera]["candidate_index"]),
                    )
                ]
                for camera in subset
            ]
        )
        exact_ids = np.flatnonzero(exact_mask)
        exact_component_ids = np.empty(0, dtype=np.int64)
        exact_component_count = 0
        exact_largest_count = 0
        if len(exact_ids):
            exact_keep, _exact_surface, exact_components = _largest_grid_component(
                local_grid[exact_mask],
                bounds_minimum=local_minimum,
                bounds_maximum=local_maximum,
                grid_shape=local_grid_diagnostics["grid_shape"],
            )
            exact_component_ids = exact_ids[exact_keep]
            exact_component_count = int(exact_components["component_count"])
            exact_largest_count = int(exact_components["largest_component_point_count"])
        coverage_sum = float(
            sum(
                candidate_metrics_by_camera[camera]["raw_component_coverage"]
                for camera in subset
            )
        )
        semantic_score_sum = float(
            sum(
                candidate_metrics_by_camera[camera]["semantic_score"]
                for camera in subset
            )
        )
        subset_record = {
            "cameras": list(subset),
            "largest_exact_component_voxel_count": exact_largest_count,
            "exact_common_voxel_count": len(exact_ids),
            "exact_component_count": exact_component_count,
            "raw_component_coverage_sum": coverage_sum,
            "semantic_score_sum": semantic_score_sum,
            "exact_common_mask_sha256": _sha256_array(exact_mask),
        }
        subset_audit.add(subset_record)
        subset_key = (
            -exact_largest_count,
            -len(exact_ids),
            -coverage_sum,
            -semantic_score_sum,
            subset,
        )
        if best_subset_key is None or subset_key < best_subset_key:
            best_subset_key = subset_key
            best_subset = {
                **subset_record,
                "exact_mask": exact_mask,
                "exact_component_ids": exact_component_ids,
            }
    _require_geometry(
        best_subset is not None
        and best_subset["largest_exact_component_voxel_count"] > 0,
        (
            "reference-optional common search found no exact-eight component"
            if reference_optional
            else "common search found no reference-anchored exact-eight component"
        ),
    )
    selected_cameras = tuple(sorted(best_subset["cameras"]))
    selected_by_camera = {
        camera: selected_candidate_by_camera[camera] for camera in selected_cameras
    }
    masks = {
        camera: np.asarray(record["mask"], dtype=bool)
        for camera, record in sorted(selected_by_camera.items())
    }
    _require_geometry(
        len(masks) == _FALLBACK_STRICT_CONSENSUS_VOTES
        and (reference_optional or reference_camera in masks),
        (
            "reference-optional common search did not retain exactly eight cameras"
            if reference_optional
            else "common search did not retain fixed-reference exact-eight cameras"
        ),
    )

    ranked_nonreference = sorted(
        nonreference_candidates,
        key=lambda camera: (
            -candidate_metrics_by_camera[camera]["raw_component_coverage"],
            camera,
        ),
    )
    coverage_rank = {
        camera: rank for rank, camera in enumerate(ranked_nonreference, start=1)
    }
    camera_inlier_records: list[dict[str, Any]] = []
    for camera in cameras:
        selected = selected_candidate_by_camera.get(camera)
        metrics = candidate_metrics_by_camera.get(camera)
        retained = camera in masks
        camera_inlier_records.append(
            {
                "camera": camera,
                "candidate_index": (
                    int(selected["candidate_index"]) if selected is not None else None
                ),
                "mask_sha256": (
                    str(selected["mask_sha256"]) if selected is not None else None
                ),
                "seed_conditioned_candidate_count": (
                    int(metrics["seed_conditioned_candidate_count"])
                    if metrics is not None
                    else 0
                ),
                "fixed_reference": camera == reference_camera,
                "nonreference_coverage_rank": (
                    0 if camera == reference_camera else coverage_rank.get(camera)
                ),
                "retained": retained,
                "raw_component_hit_count": (
                    int(metrics["raw_component_hit_count"])
                    if metrics is not None
                    else 0
                ),
                "raw_component_coverage": (
                    float(metrics["raw_component_coverage"])
                    if metrics is not None
                    else 0.0
                ),
                "raw_component_precision": (
                    float(metrics["raw_component_precision"])
                    if metrics is not None
                    else 0.0
                ),
                "hits_local_union_peak": (
                    bool(metrics["hits_local_union_peak"])
                    if metrics is not None
                    else False
                ),
                "semantic_score": (
                    float(metrics["semantic_score"]) if metrics is not None else None
                ),
                "candidate_lexicographic_objective": (
                    list(metrics["candidate_lexicographic_objective"])
                    if metrics is not None
                    else None
                ),
            }
        )
    inlier_record_by_camera = {
        record["camera"]: record for record in camera_inlier_records
    }
    selection_records = [
        dict(inlier_record_by_camera[camera]) for camera in selected_cameras
    ]
    selected_reference = selected_evaluation["reference_record"]
    selected_reference_descriptor = selected_reference["descriptor"]
    final_records_by_camera: dict[str, list[dict[str, Any]]] = {}
    for camera in cameras:
        if camera == reference_camera:
            final_records_by_camera[camera] = basic_by_camera[camera]
        else:
            final_records_by_camera[camera] = _reference_consistent_candidate_records(
                basic_by_camera[camera],
                selected_reference_descriptor,
                config.sam2,
            )
    diagnostics = [
        _selection_view_diagnostic(
            camera,
            final_records_by_camera[camera],
            selected_by_camera.get(camera),
            reference_camera=reference_camera,
            geometry_inlier_selection=inlier_record_by_camera[camera],
        )
        for camera in cameras
    ]
    selected_rank = int(selected_evaluation["rank"])
    selected_exact_component_ids = np.asarray(
        best_subset["exact_component_ids"], dtype=np.int64
    )
    selected_exact_component_mask = np.zeros(len(local_grid), dtype=bool)
    selected_exact_component_mask[selected_exact_component_ids] = True
    selected_exact_seed_id = int(np.min(selected_exact_component_ids))
    audit: dict[str, Any] = {
        "policy_id": (
            FRAME_ZERO_REFERENCE_OPTIONAL_FALLBACK_POLICY_ID
            if reference_optional
            else FRAME_ZERO_GEOMETRY_FALLBACK_POLICY_ID
        ),
        "strategy": (
            "reference-conditioned-reference-optional-exhaustive-exact-eight-assignment"
            if reference_optional
            else "reference-anchored-exhaustive-exact-eight-assignment"
        ),
        "reference_seed_policy": _FALLBACK_REFERENCE_SEED_POLICY,
        "reference_seed_limit": _FALLBACK_REFERENCE_SEED_COUNT,
        "inlier_selection_rule": (
            _REFERENCE_OPTIONAL_INLIER_SELECTION_RULE
            if reference_optional
            else _COMMON_INLIER_SELECTION_RULE
        ),
        "search_hit_representation": "nearest-pixel-raw-center",
        "evaluated_reference_seed_count": len(reference_candidates),
        "strict_consensus_vote_count": _FALLBACK_STRICT_CONSENSUS_VOTES,
        "grid": {
            "bounds_minimum_world_m": minimum.tolist(),
            "bounds_maximum_world_m": maximum.tolist(),
            "grid_shape": grid_shape,
            "effective_axis_spacing_m": grid_spacing,
            "grid_point_count": len(grid),
            "grid_points_sha256": _sha256_array(grid),
        },
        "proposal_count_by_camera": {
            camera: len(basic_by_camera[camera]) for camera in cameras
        },
        "proposal_inventory_sha256": _proposal_inventory_sha256(basic_by_camera),
        "reference_candidate_ranking": [
            {
                "rank": rank,
                "candidate_index": int(record["candidate_index"]),
                "mask_sha256": record["mask_sha256"],
                "local_mask_score": float(record["local_score"]),
                "selected_for_seed_evaluation": rank < _FALLBACK_REFERENCE_SEED_COUNT,
            }
            for rank, record in enumerate(ranked_reference_candidates)
        ],
        "reference_seed_evaluations": seed_audit,
        "reference_seed_objective_order": [
            "local_largest_component_voxel_count_desc",
            "local_strict_feasible_voxel_count_desc",
            "local_peak_support_desc",
            "global_largest_component_voxel_count_desc",
            "global_strict_feasible_voxel_count_desc",
            "reference_semantic_score_desc",
            "reference_candidate_index_asc",
        ],
        "selected_reference_seed_rank": selected_rank,
        "selected_reference_candidate_index": int(
            selected_reference["candidate_index"]
        ),
        "selected_reference_mask_sha256": selected_reference["mask_sha256"],
        "global_selected_common_voxel_flat_index": int(global_selected_seed_id),
        "global_selected_common_voxel_world_m": grid[global_selected_seed_id].tolist(),
        "global_selected_common_voxel_support_count": int(
            support[global_selected_seed_id]
        ),
        "global_selected_strict_component_voxel_count": len(component_ids),
        "global_selected_strict_component_sha256": _sha256_array(component_mask),
        "local_refinement": {
            "requested_voxel_size_m": (_FALLBACK_COMMON_LOCAL_REQUESTED_VOXEL_SIZE_M),
            "bounds": local_bounds,
            "grid": local_grid_diagnostics,
            "grid_points_sha256": _sha256_array(local_grid),
            "strict_feasible_voxel_count": len(local_feasible_ids),
            "strict_feasible_mask_sha256": _sha256_array(local_feasible),
            "components": local_components,
        },
        "local_union_peak_voxel_flat_index": int(local_selected_seed_id),
        "local_union_peak_voxel_world_m": local_grid[local_selected_seed_id].tolist(),
        "local_union_peak_support_count": int(local_support[local_selected_seed_id]),
        "local_union_strict_component_voxel_count": len(local_component_ids),
        "local_union_strict_component_sha256": _sha256_array(local_component_mask),
        "candidate_objective_order": [
            "local_component_coverage_desc",
            "local_component_precision_desc",
            "hits_local_union_peak_desc",
            "semantic_score_desc",
            "candidate_index_asc",
        ],
        "camera_inlier_ranking": camera_inlier_records,
        "evaluated_exact_eight_subset_count": subset_audit.record_count,
        "exact_eight_subset_objective_order": [
            "largest_exact_component_voxel_count_desc",
            "exact_common_voxel_count_desc",
            "raw_component_coverage_sum_desc",
            "semantic_score_sum_desc",
            "camera_tuple_asc",
        ],
        "exact_eight_subset_evaluations": subset_audit.materialize(
            selected_record=best_subset
        ),
        "selected_exact_eight_cameras": list(selected_cameras),
        "selected_common_voxel_flat_index": selected_exact_seed_id,
        "selected_common_voxel_world_m": local_grid[selected_exact_seed_id].tolist(),
        "selected_common_voxel_support_count": _FALLBACK_STRICT_CONSENSUS_VOTES,
        "selected_exact_common_voxel_count": int(
            best_subset["exact_common_voxel_count"]
        ),
        "selected_exact_common_mask_sha256": best_subset["exact_common_mask_sha256"],
        "selected_strict_component_voxel_count": len(selected_exact_component_ids),
        "selected_strict_component_sha256": _sha256_array(
            selected_exact_component_mask
        ),
        "selected_proposals": selection_records,
        "selected_mask_set_sha256": _mask_set_sha256(masks),
    }
    audit["artifact_sha256"] = artifact_sha256(audit)
    return masks, diagnostics, audit


def _projection_matrix(
    intrinsics: np.ndarray, camera_to_world: np.ndarray
) -> np.ndarray:
    return (
        np.asarray(intrinsics, dtype=np.float64)
        @ np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))[:3]
    )


def _project_points(
    points_world_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world_to_camera = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))
    points = np.asarray(points_world_m, dtype=np.float64)
    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    depth = camera[:, 2]
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    front = depth > 1e-8
    pixels[front, 0] = (
        camera[front, 0] / depth[front] * intrinsics[0, 0] + intrinsics[0, 2]
    )
    pixels[front, 1] = (
        camera[front, 1] / depth[front] * intrinsics[1, 1] + intrinsics[1, 2]
    )
    return pixels, depth


def _projected_half_voxel_radii(
    points_world_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    axis_spacing_m: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project an axis-aligned voxel's half extents into pixel coordinates.

    The first-order footprint is deliberately evaluated independently at every
    candidate point.  A half-pixel term covers nearest-pixel quantisation; the
    remaining term is the L1 support radius of the world-axis-aligned voxel
    under the local projection Jacobian.
    """

    points = np.asarray(points_world_m, dtype=np.float64)
    intrinsics_array = np.asarray(intrinsics, dtype=np.float64)
    camera_to_world_array = np.asarray(camera_to_world, dtype=np.float64)
    spacing = np.asarray(axis_spacing_m, dtype=np.float64)
    _require(
        points.ndim == 2 and points.shape[1] == 3,
        "footprint points must have shape (N,3)",
    )
    _require(np.all(np.isfinite(points)), "footprint points are non-finite")
    _require(intrinsics_array.shape == (3, 3), "invalid footprint intrinsics")
    _require(
        camera_to_world_array.shape == (4, 4),
        "invalid footprint camera transform",
    )
    _require(
        spacing.shape == (3,)
        and np.all(np.isfinite(spacing))
        and np.all(spacing > 0.0),
        "invalid footprint axis spacing",
    )
    world_to_camera = np.linalg.inv(camera_to_world_array)
    rotation = world_to_camera[:3, :3]
    camera = points @ rotation.T + world_to_camera[:3, 3]
    depth = camera[:, 2]
    front = depth > 1e-8
    safe_depth = np.where(front, depth, 1.0)
    x = camera[:, 0]
    y = camera[:, 1]
    fx = float(intrinsics_array[0, 0])
    fy = float(intrinsics_array[1, 1])
    pixels = np.empty((len(points), 2), dtype=np.float64)
    pixels[:, 0] = x / safe_depth * fx + intrinsics_array[0, 2]
    pixels[:, 1] = y / safe_depth * fy + intrinsics_array[1, 2]

    du_camera = np.stack(
        (
            np.full(len(points), fx, dtype=np.float64) / safe_depth,
            np.zeros(len(points), dtype=np.float64),
            -fx * x / np.square(safe_depth),
        ),
        axis=1,
    )
    dv_camera = np.stack(
        (
            np.zeros(len(points), dtype=np.float64),
            np.full(len(points), fy, dtype=np.float64) / safe_depth,
            -fy * y / np.square(safe_depth),
        ),
        axis=1,
    )
    du_world = du_camera @ rotation
    dv_world = dv_camera @ rotation
    radius_u = 0.5 * np.sum(np.abs(du_world) * spacing[None, :], axis=1) + 0.5
    radius_v = 0.5 * np.sum(np.abs(dv_world) * spacing[None, :], axis=1) + 0.5
    return pixels, depth, radius_u, radius_v


def _projected_footprint_hits(
    points_world_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    axis_spacing_m: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Test whether each projected voxel footprint overlaps a source mask."""

    source_mask = np.asarray(mask, dtype=bool)
    _require(source_mask.ndim == 2, "footprint mask must be two-dimensional")
    pixels, depth, radius_u, radius_v = _projected_half_voxel_radii(
        points_world_m,
        intrinsics,
        camera_to_world,
        axis_spacing_m=axis_spacing_m,
    )
    height, width = source_mask.shape
    finite = (
        (depth > 1e-8)
        & np.all(np.isfinite(pixels), axis=1)
        & np.isfinite(radius_u)
        & np.isfinite(radius_v)
    )
    safe_u = np.where(finite, pixels[:, 0], 0.0)
    safe_v = np.where(finite, pixels[:, 1], 0.0)
    safe_radius_u = np.where(finite, radius_u, 0.5)
    safe_radius_v = np.where(finite, radius_v, 0.5)
    left = np.ceil(safe_u - safe_radius_u).astype(np.int64)
    right = np.floor(safe_u + safe_radius_u).astype(np.int64)
    top = np.ceil(safe_v - safe_radius_v).astype(np.int64)
    bottom = np.floor(safe_v + safe_radius_v).astype(np.int64)
    overlaps_image = (
        finite
        & (left <= right)
        & (top <= bottom)
        & (right >= 0)
        & (left < width)
        & (bottom >= 0)
        & (top < height)
    )
    left_clipped = np.clip(left, 0, width - 1)
    right_clipped = np.clip(right, 0, width - 1)
    top_clipped = np.clip(top, 0, height - 1)
    bottom_clipped = np.clip(bottom, 0, height - 1)
    integral = (
        np.pad(source_mask.astype(np.int64), ((1, 0), (1, 0)), mode="constant")
        .cumsum(axis=0)
        .cumsum(axis=1)
    )
    sums = (
        integral[bottom_clipped + 1, right_clipped + 1]
        - integral[top_clipped, right_clipped + 1]
        - integral[bottom_clipped + 1, left_clipped]
        + integral[top_clipped, left_clipped]
    )
    hits = overlaps_image & (sums > 0)
    visible_ids = np.flatnonzero(overlaps_image)

    def radius_summary(values: np.ndarray) -> dict[str, float | None]:
        selected = values[visible_ids]
        if not len(selected):
            return {"minimum": None, "median": None, "maximum": None}
        return {
            "minimum": float(np.min(selected)),
            "median": float(np.median(selected)),
            "maximum": float(np.max(selected)),
        }

    return (
        hits,
        overlaps_image,
        {
            "in_bounds_footprint_count": int(np.count_nonzero(overlaps_image)),
            "mask_overlap_footprint_count": int(np.count_nonzero(hits)),
            "projected_half_voxel_radius_u_pixels": radius_summary(radius_u),
            "projected_half_voxel_radius_v_pixels": radius_summary(radius_v),
        },
    )


def _nearest_projected_footprint_mask_pixels(
    points_world_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    axis_spacing_m: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Choose one deterministic RGB pixel inside each overlapping footprint.

    Distance from the continuous projection is the primary key.  Image row and
    then column are explicit tie-breaks, so symmetric footprints remain stable
    across platforms and NumPy versions.
    """

    source_mask = np.asarray(mask, dtype=bool)
    _require(source_mask.ndim == 2, "footprint mask must be two-dimensional")
    pixels, depth, radius_u, radius_v = _projected_half_voxel_radii(
        points_world_m,
        intrinsics,
        camera_to_world,
        axis_spacing_m=axis_spacing_m,
    )
    height, width = source_mask.shape
    finite = (
        (depth > 1e-8)
        & np.all(np.isfinite(pixels), axis=1)
        & np.isfinite(radius_u)
        & np.isfinite(radius_v)
    )
    safe_u = np.where(finite, pixels[:, 0], 0.0)
    safe_v = np.where(finite, pixels[:, 1], 0.0)
    left = np.ceil(safe_u - np.where(finite, radius_u, 0.5)).astype(np.int64)
    right = np.floor(safe_u + np.where(finite, radius_u, 0.5)).astype(np.int64)
    top = np.ceil(safe_v - np.where(finite, radius_v, 0.5)).astype(np.int64)
    bottom = np.floor(safe_v + np.where(finite, radius_v, 0.5)).astype(np.int64)
    overlaps_image = (
        finite
        & (left <= right)
        & (top <= bottom)
        & (right >= 0)
        & (left < width)
        & (bottom >= 0)
        & (top < height)
    )
    selected_pixels = np.full((len(pixels), 2), -1, dtype=np.int64)
    found = np.zeros(len(pixels), dtype=bool)
    for point_index in np.flatnonzero(overlaps_image):
        left_value = max(0, int(left[point_index]))
        right_value = min(width - 1, int(right[point_index]))
        top_value = max(0, int(top[point_index]))
        bottom_value = min(height - 1, int(bottom[point_index]))
        candidates = np.argwhere(
            source_mask[
                top_value : bottom_value + 1,
                left_value : right_value + 1,
            ]
        )
        if not len(candidates):
            continue
        rows = candidates[:, 0] + top_value
        columns = candidates[:, 1] + left_value
        squared_distance = np.square(columns - pixels[point_index, 0]) + np.square(
            rows - pixels[point_index, 1]
        )
        order = np.lexsort((columns, rows, squared_distance))
        selected = int(order[0])
        selected_pixels[point_index] = [columns[selected], rows[selected]]
        found[point_index] = True
    return found, selected_pixels


def _point_mask_hits(
    points_world_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-pixel point/mask membership used by the legacy-compatible search."""

    source_mask = np.asarray(mask, dtype=bool)
    pixels, depth = _project_points(points_world_m, intrinsics, camera_to_world)
    finite = (depth > 1e-8) & np.all(np.isfinite(pixels), axis=1)
    safe = np.where(np.isfinite(pixels), pixels, 0.0)
    rounded = np.rint(safe).astype(np.int64)
    height, width = source_mask.shape
    in_bounds = (
        finite
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    hits = np.zeros(len(rounded), dtype=bool)
    ids = np.flatnonzero(in_bounds)
    if len(ids):
        hits[ids] = source_mask[rounded[ids, 1], rounded[ids, 0]]
    return hits, in_bounds


def _carve_candidate_points_with_footprints(
    candidate_points_world_m: np.ndarray,
    masks_by_camera: Mapping[str, np.ndarray],
    intrinsics_by_camera: Mapping[str, np.ndarray],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    *,
    axis_spacing_m: Sequence[float],
    required_vote_count: int = _FALLBACK_STRICT_CONSENSUS_VOTES,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Carve with source-mask overlap by projected voxel footprints.

    Unlike the legacy carver, the quorum is absolute.  A low peak therefore
    produces an empty hull rather than silently relaxing the physical support
    requirement.
    """

    points = np.asarray(candidate_points_world_m, dtype=np.float64)
    _require(
        points.ndim == 2 and points.shape[1] == 3 and len(points) > 0,
        "candidate points must have shape (N,3)",
    )
    cameras = tuple(sorted(masks_by_camera))
    _require(len(cameras) >= required_vote_count, "too few cameras for strict quorum")
    _require(
        set(cameras) <= set(intrinsics_by_camera)
        and set(cameras) <= set(camera_to_world_by_camera),
        "footprint calibration is incomplete",
    )
    votes = np.zeros(len(points), dtype=np.uint16)
    per_camera: list[dict[str, Any]] = []
    for camera in cameras:
        hits, _in_bounds, footprint = _projected_footprint_hits(
            points,
            masks_by_camera[camera],
            intrinsics_by_camera[camera],
            camera_to_world_by_camera[camera],
            axis_spacing_m=axis_spacing_m,
        )
        votes += hits.astype(np.uint16)
        per_camera.append({"camera": camera, **footprint})
    accepted = votes >= required_vote_count
    peak = int(votes.max(initial=0))
    return (
        points[accepted],
        accepted,
        {
            "candidate_point_count": len(points),
            "camera_count": len(cameras),
            "peak_vote_count": peak,
            "required_vote_count": required_vote_count,
            "hull_point_count": int(np.count_nonzero(accepted)),
            "accepted_candidate_fraction": float(np.mean(accepted)),
            "axis_spacing_m": np.asarray(axis_spacing_m, dtype=np.float64).tolist(),
            "vote_count_sha256": _sha256_array(votes),
            "accepted_mask_sha256": _sha256_array(accepted),
            "per_camera": per_camera,
        },
    )


def _local_refinement_bounds(
    coarse_component_points_world_m: np.ndarray,
    *,
    global_minimum_world_m: np.ndarray,
    global_maximum_world_m: np.ndarray,
    coarse_axis_spacing_m: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    points = np.asarray(coarse_component_points_world_m, dtype=np.float64)
    spacing = np.asarray(coarse_axis_spacing_m, dtype=np.float64)
    global_minimum = np.asarray(global_minimum_world_m, dtype=np.float64)
    global_maximum = np.asarray(global_maximum_world_m, dtype=np.float64)
    _require(
        points.ndim == 2 and points.shape[1] == 3 and len(points) > 0,
        "coarse component is empty",
    )
    _require(
        spacing.shape == (3,) and np.all(spacing > 0.0),
        "invalid coarse spacing",
    )
    margin = _FALLBACK_LOCAL_MARGIN_COARSE_CELLS * spacing
    minimum = np.maximum(global_minimum, np.min(points, axis=0) - margin)
    maximum = np.minimum(global_maximum, np.max(points, axis=0) + margin)
    _require(np.all(maximum > minimum), "local refinement bounds collapsed")
    return (
        minimum,
        maximum,
        {
            "coarse_margin_cell_count": _FALLBACK_LOCAL_MARGIN_COARSE_CELLS,
            "margin_m": margin.tolist(),
            "bounds_minimum_world_m": minimum.tolist(),
            "bounds_maximum_world_m": maximum.tolist(),
        },
    )


def _resolution_aware_depth_radius(
    hull_points_world_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    axis_spacing_m: Sequence[float],
    image_shape: Sequence[int],
) -> tuple[int, dict[str, Any]]:
    """Choose a deterministic per-camera splat radius from projected voxels."""

    pixels, depth, radius_u, radius_v = _projected_half_voxel_radii(
        hull_points_world_m,
        intrinsics,
        camera_to_world,
        axis_spacing_m=axis_spacing_m,
    )
    height, width = (int(image_shape[0]), int(image_shape[1]))
    visible = (
        (depth > 1e-8)
        & np.all(np.isfinite(pixels), axis=1)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < height)
    )
    projected = np.maximum(radius_u[visible], radius_v[visible])
    radius = (
        max(1, int(math.ceil(float(np.median(projected))))) if len(projected) else 1
    )
    return radius, {
        "policy": "ceil-median-projected-half-voxel-radius",
        "visible_hull_point_count": int(np.count_nonzero(visible)),
        "radius_pixels": radius,
        "projected_radius_pixels": {
            "minimum": float(np.min(projected)) if len(projected) else None,
            "median": float(np.median(projected)) if len(projected) else None,
            "maximum": float(np.max(projected)) if len(projected) else None,
        },
    }


def _largest_grid_component(
    points: np.ndarray,
    *,
    bounds_minimum: np.ndarray,
    bounds_maximum: np.ndarray,
    grid_shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    shape = tuple(int(value) for value in grid_shape)
    _require(
        len(shape) == 3 and all(value >= 2 for value in shape), "invalid grid shape"
    )
    scale = (np.asarray(shape, dtype=np.float64) - 1.0) / (
        np.asarray(bounds_maximum) - np.asarray(bounds_minimum)
    )
    indices = np.rint((points - bounds_minimum) * scale).astype(np.int64)
    _require(
        np.all(indices >= 0) and np.all(indices < np.asarray(shape)),
        "hull point lies outside its grid",
    )
    occupancy = np.zeros(shape, dtype=bool)
    occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    visited = np.zeros(shape, dtype=bool)
    components: list[list[tuple[int, int, int]]] = []
    for seed_array in np.argwhere(occupancy):
        seed = tuple(int(value) for value in seed_array)
        if visited[seed]:
            continue
        visited[seed] = True
        queue = deque([seed])
        component: list[tuple[int, int, int]] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for axis in range(3):
                for direction in (-1, 1):
                    neighbor = list(current)
                    neighbor[axis] += direction
                    if not 0 <= neighbor[axis] < shape[axis]:
                        continue
                    key = tuple(neighbor)
                    if occupancy[key] and not visited[key]:
                        visited[key] = True
                        queue.append(key)
        components.append(component)
    _require(bool(components), "visual hull has no connected component")
    largest = max(components, key=len)
    keep_occupancy = np.zeros(shape, dtype=bool)
    component_indices = np.asarray(largest, dtype=np.int64)
    keep_occupancy[
        component_indices[:, 0], component_indices[:, 1], component_indices[:, 2]
    ] = True
    keep = keep_occupancy[indices[:, 0], indices[:, 1], indices[:, 2]]
    interior = keep_occupancy.copy()
    for axis in range(3):
        positive = np.zeros_like(keep_occupancy)
        negative = np.zeros_like(keep_occupancy)
        positive_slice = [slice(None)] * 3
        positive_source = [slice(None)] * 3
        positive_slice[axis] = slice(0, -1)
        positive_source[axis] = slice(1, None)
        positive[tuple(positive_slice)] = keep_occupancy[tuple(positive_source)]
        negative_slice = [slice(None)] * 3
        negative_source = [slice(None)] * 3
        negative_slice[axis] = slice(1, None)
        negative_source[axis] = slice(0, -1)
        negative[tuple(negative_slice)] = keep_occupancy[tuple(negative_source)]
        interior &= positive & negative
    surface_occupancy = keep_occupancy & ~interior
    surface = surface_occupancy[indices[:, 0], indices[:, 1], indices[:, 2]]
    sizes = sorted((len(component) for component in components), reverse=True)
    return (
        keep,
        surface,
        {
            "component_count": len(components),
            "component_point_counts_descending": sizes[:20],
            "largest_component_point_count": int(np.count_nonzero(keep)),
            "largest_component_fraction": float(np.mean(keep)),
            "surface_point_count": int(np.count_nonzero(surface)),
        },
    )


def _sample_surface_points(
    surface_points: np.ndarray, *, count: int, rng_seed: int
) -> np.ndarray:
    if len(surface_points) <= count:
        return np.asarray(surface_points, dtype=np.float64)
    rng = np.random.default_rng(rng_seed)
    indices = np.sort(rng.choice(len(surface_points), size=count, replace=False))
    return np.asarray(surface_points[indices], dtype=np.float64)


def _render_depth(
    hull_points: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    height, width = mask.shape
    pixels, depth = _project_points(hull_points, intrinsics, camera_to_world)
    rounded = np.rint(pixels).astype(np.int64)
    base = (
        (depth > 0.0)
        & np.all(np.isfinite(pixels), axis=1)
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    flat = np.full(height * width, np.inf, dtype=np.float64)
    for row_offset in range(-radius, radius + 1):
        for column_offset in range(-radius, radius + 1):
            if row_offset**2 + column_offset**2 > radius**2:
                continue
            rows = rounded[:, 1] + row_offset
            columns = rounded[:, 0] + column_offset
            valid = (
                base
                & (rows >= 0)
                & (rows < height)
                & (columns >= 0)
                & (columns < width)
            )
            ids = np.flatnonzero(valid)
            if len(ids):
                np.minimum.at(flat, rows[ids] * width + columns[ids], depth[ids])
    depth_map = flat.reshape(height, width)
    valid = np.isfinite(depth_map) & mask
    result = np.zeros((height, width), dtype=np.float32)
    result[valid] = depth_map[valid].astype(np.float32)
    mask_count = int(np.count_nonzero(mask))
    return (
        result,
        valid,
        {
            "mask_pixel_count": mask_count,
            "valid_depth_pixel_count": int(np.count_nonzero(valid)),
            "depth_mask_coverage": (
                float(np.count_nonzero(valid) / mask_count) if mask_count else 0.0
            ),
            "minimum_depth_m": float(np.min(result[valid])) if np.any(valid) else None,
            "median_depth_m": (
                float(np.median(result[valid])) if np.any(valid) else None
            ),
            "maximum_depth_m": float(np.max(result[valid])) if np.any(valid) else None,
        },
    )


def build_frame_zero_geometry(
    rgb_by_camera: Mapping[str, np.ndarray],
    masks_by_camera: Mapping[str, np.ndarray],
    intrinsics_by_camera: Mapping[str, np.ndarray],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    *,
    config: FrameZeroAssetConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Carve and color an object-only visual hull from frame-zero material."""

    cameras = tuple(sorted(masks_by_camera))
    _require(len(cameras) >= config.minimum_camera_count, "too few selected cameras")
    _require(
        config.minimum_consensus_votes <= len(cameras),
        "consensus votes exceed selected camera count",
    )
    _require(set(cameras) == set(rgb_by_camera), "RGB/mask cameras differ")
    _require(
        set(cameras) <= set(intrinsics_by_camera)
        and set(cameras) <= set(camera_to_world_by_camera),
        "calibration is incomplete",
    )
    shapes = {np.asarray(rgb_by_camera[camera]).shape for camera in cameras}
    _require(len(shapes) == 1, "selected RGB frame shapes differ")
    image_shape = next(iter(shapes))
    _require(len(image_shape) == 3 and image_shape[2] == 3, "invalid RGB frame shape")
    for camera in cameras:
        _require(
            np.asarray(masks_by_camera[camera]).shape == image_shape[:2],
            f"mask/RGB shape mismatch for {camera}",
        )
    centers = np.stack(
        [np.asarray(camera_to_world_by_camera[camera])[:3, 3] for camera in cameras]
    )
    grid_center = np.mean(centers, axis=0)
    minimum = grid_center - config.cube_half_extent_m
    maximum = grid_center + config.cube_half_extent_m
    grid, grid_diagnostics = regular_grid_in_bounds(
        minimum,
        maximum,
        requested_voxel_size_m=config.requested_voxel_size_m,
        maximum_point_count=config.maximum_grid_point_count,
    )
    hull, carving = carve_candidate_points(
        grid,
        masks_by_camera,
        intrinsics_by_camera,
        camera_to_world_by_camera,
        consensus_fraction_of_peak=config.consensus_fraction_of_peak,
        minimum_consensus_votes=config.minimum_consensus_votes,
    )
    _require_geometry(
        len(hull) >= config.minimum_hull_point_count,
        "frame-zero visual hull is too small",
    )
    keep, surface, components = _largest_grid_component(
        hull,
        bounds_minimum=minimum,
        bounds_maximum=maximum,
        grid_shape=grid_diagnostics["grid_shape"],
    )
    _require_geometry(
        components["largest_component_fraction"]
        >= config.minimum_largest_component_fraction,
        "visual hull is dominated by disconnected clutter",
    )
    hull = hull[keep]
    surface_points = _sample_surface_points(
        np.asarray(hull[surface[keep]], dtype=np.float64),
        count=config.object_point_count,
        rng_seed=config.rng_seed,
    )
    _require_geometry(len(surface_points) >= 128, "too few object surface points")

    rgb_stack = np.stack(
        [np.asarray(rgb_by_camera[camera], dtype=np.uint8) for camera in cameras]
    )
    mask_stack = np.stack(
        [np.asarray(masks_by_camera[camera], dtype=bool) for camera in cameras]
    )
    intrinsics_stack = np.stack(
        [
            np.asarray(intrinsics_by_camera[camera], dtype=np.float64)
            for camera in cameras
        ]
    )
    extrinsics_stack = np.stack(
        [
            np.asarray(camera_to_world_by_camera[camera], dtype=np.float64)
            for camera in cameras
        ]
    )
    projection_stack = np.stack(
        [
            _projection_matrix(
                intrinsics_by_camera[camera], camera_to_world_by_camera[camera]
            )
            for camera in cameras
        ]
    )
    sampled_colors = np.full((len(cameras), len(surface_points), 3), np.nan)
    depth_maps = []
    depth_valid = []
    camera_qa = []
    for camera_index, camera in enumerate(cameras):
        mask = mask_stack[camera_index]
        pixels, point_depth = _project_points(
            surface_points,
            intrinsics_stack[camera_index],
            extrinsics_stack[camera_index],
        )
        rounded = np.rint(pixels).astype(np.int64)
        height, width = mask.shape
        visible = (
            (point_depth > 0.0)
            & np.all(np.isfinite(pixels), axis=1)
            & (rounded[:, 0] >= 0)
            & (rounded[:, 0] < width)
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < height)
        )
        ids = np.flatnonzero(visible)
        inside = np.zeros(len(surface_points), dtype=bool)
        if len(ids):
            inside[ids] = mask[rounded[ids, 1], rounded[ids, 0]]
            color_ids = np.flatnonzero(inside)
            sampled_colors[camera_index, color_ids] = rgb_stack[
                camera_index, rounded[color_ids, 1], rounded[color_ids, 0]
            ]
        depth_map, valid_map, depth_qa = _render_depth(
            hull,
            mask,
            intrinsics_stack[camera_index],
            extrinsics_stack[camera_index],
            radius=config.depth_splat_radius_pixels,
        )
        depth_maps.append(depth_map)
        depth_valid.append(valid_map)
        camera_qa.append(
            {
                "camera": camera,
                "mask_area_pixels": int(np.count_nonzero(mask)),
                "mask_area_fraction": float(np.mean(mask)),
                "visible_surface_point_count": int(np.count_nonzero(visible)),
                "inside_mask_surface_point_count": int(np.count_nonzero(inside)),
                "hull_mask_containment": (
                    float(np.count_nonzero(inside) / np.count_nonzero(visible))
                    if np.any(visible)
                    else 0.0
                ),
                **depth_qa,
            }
        )
    support_count = np.sum(np.all(np.isfinite(sampled_colors), axis=2), axis=0)
    _require_geometry(
        np.all(support_count > 0), "one or more surface points have no RGB support"
    )
    object_colors = np.nanmedian(sampled_colors, axis=0).astype(np.uint8)
    containment = np.asarray(
        [record["hull_mask_containment"] for record in camera_qa], dtype=np.float64
    )
    coverage = np.asarray(
        [record["depth_mask_coverage"] for record in camera_qa], dtype=np.float64
    )
    quantiles = np.percentile(surface_points, [1.0, 50.0, 99.0], axis=0)
    gates = {
        "camera_count": len(cameras) >= config.minimum_camera_count,
        "hull_point_count": len(hull) >= config.minimum_hull_point_count,
        "largest_component_fraction": (
            components["largest_component_fraction"]
            >= config.minimum_largest_component_fraction
        ),
        "median_depth_mask_coverage": (
            float(np.median(coverage)) >= config.minimum_median_depth_mask_coverage
        ),
        "median_hull_mask_containment": (
            float(np.median(containment)) >= config.minimum_median_hull_mask_containment
        ),
        "all_surface_points_colored": bool(np.all(support_count > 0)),
    }
    arrays = {
        "frame_indices": np.asarray([0], dtype=np.int64),
        "camera_names": np.asarray(cameras),
        "rgb_frame0": rgb_stack,
        "mask_frame0": mask_stack,
        "depth_frame0_m": np.stack(depth_maps).astype(np.float32),
        "depth_valid_frame0": np.stack(depth_valid),
        "intrinsics": intrinsics_stack,
        "camera_to_world": extrinsics_stack,
        "projection_world_to_pixel": projection_stack,
        "object_points_world_m": surface_points.astype(np.float32),
        "object_colors_rgb": object_colors,
        "object_color_support_count": support_count.astype(np.uint8),
        "visual_hull_points_world_m": hull.astype(np.float32),
    }
    diagnostics = {
        "grid": grid_diagnostics,
        "carving": carving,
        "components": components,
        "visual_hull_point_count": len(hull),
        "object_point_count": len(surface_points),
        "object_world_m": {
            "q01": quantiles[0].tolist(),
            "median": quantiles[1].tolist(),
            "q99": quantiles[2].tolist(),
            "q01_to_q99_span": (quantiles[2] - quantiles[0]).tolist(),
        },
        "object_color_support_count": {
            "minimum": int(np.min(support_count)),
            "median": float(np.median(support_count)),
            "maximum": int(np.max(support_count)),
        },
        "depth_mask_coverage": {
            "minimum": float(np.min(coverage)),
            "median": float(np.median(coverage)),
            "maximum": float(np.max(coverage)),
        },
        "hull_mask_containment": {
            "minimum": float(np.min(containment)),
            "median": float(np.median(containment)),
            "maximum": float(np.max(containment)),
        },
        "per_camera": camera_qa,
        "acceptance_gates": gates,
        "geometry_qa_passed": all(gates.values()),
    }
    failed_gates = sorted(name for name, passed in gates.items() if not passed)
    _require_geometry(
        diagnostics["geometry_qa_passed"],
        "frame-zero geometry failed QA: " + ",".join(failed_gates),
    )
    return arrays, diagnostics


def _build_frame_zero_fallback_geometry(
    rgb_by_camera: Mapping[str, np.ndarray],
    masks_by_camera: Mapping[str, np.ndarray],
    intrinsics_by_camera: Mapping[str, np.ndarray],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    *,
    config: FrameZeroAssetConfig,
    strategy: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build strict-quorum footprint geometry after an audited legacy failure."""

    _require(
        strategy
        in {
            "same-masks-projected-footprint",
            "common-voxel-assignment-projected-footprint",
            FRAME_ZERO_REFERENCE_OPTIONAL_GEOMETRY_STRATEGY,
        },
        "unknown geometry fallback strategy",
    )
    cameras = tuple(sorted(masks_by_camera))
    _require(
        len(cameras) >= _FALLBACK_STRICT_CONSENSUS_VOTES,
        "too few selected cameras for geometry fallback",
    )
    _require(set(cameras) == set(rgb_by_camera), "fallback RGB/mask cameras differ")
    _require(
        set(cameras) <= set(intrinsics_by_camera)
        and set(cameras) <= set(camera_to_world_by_camera),
        "fallback calibration is incomplete",
    )
    image_shapes = {np.asarray(rgb_by_camera[camera]).shape for camera in cameras}
    _require(len(image_shapes) == 1, "fallback RGB frame shapes differ")
    image_shape = next(iter(image_shapes))
    _require(
        len(image_shape) == 3 and image_shape[2] == 3,
        "invalid fallback RGB frame shape",
    )
    for camera in cameras:
        _require(
            np.asarray(masks_by_camera[camera]).shape == image_shape[:2],
            f"fallback mask/RGB shape mismatch for {camera}",
        )

    centers = np.stack(
        [np.asarray(camera_to_world_by_camera[camera])[:3, 3] for camera in cameras]
    )
    grid_center = np.mean(centers, axis=0)
    global_minimum = grid_center - config.cube_half_extent_m
    global_maximum = grid_center + config.cube_half_extent_m
    coarse_grid, coarse_grid_diagnostics = regular_grid_in_bounds(
        global_minimum,
        global_maximum,
        requested_voxel_size_m=config.requested_voxel_size_m,
        maximum_point_count=config.maximum_grid_point_count,
    )
    coarse_spacing = coarse_grid_diagnostics["effective_axis_spacing_m"]
    coarse_hull, _coarse_accepted, coarse_carving = (
        _carve_candidate_points_with_footprints(
            coarse_grid,
            masks_by_camera,
            intrinsics_by_camera,
            camera_to_world_by_camera,
            axis_spacing_m=coarse_spacing,
        )
    )
    _require_geometry(
        len(coarse_hull) >= _FALLBACK_MINIMUM_COARSE_COMPONENT_POINT_COUNT,
        "projected-footprint coarse hull is too small",
    )
    coarse_keep, _coarse_surface, coarse_components = _largest_grid_component(
        coarse_hull,
        bounds_minimum=global_minimum,
        bounds_maximum=global_maximum,
        grid_shape=coarse_grid_diagnostics["grid_shape"],
    )
    _require_geometry(
        coarse_components["largest_component_point_count"]
        >= _FALLBACK_MINIMUM_COARSE_COMPONENT_POINT_COUNT,
        "projected-footprint coarse connected core is too small",
    )
    _require_geometry(
        coarse_components["largest_component_fraction"]
        >= config.minimum_largest_component_fraction,
        "projected-footprint coarse hull is dominated by disconnected clutter",
    )
    coarse_component = np.asarray(coarse_hull[coarse_keep], dtype=np.float64)
    local_minimum, local_maximum, local_bounds = _local_refinement_bounds(
        coarse_component,
        global_minimum_world_m=global_minimum,
        global_maximum_world_m=global_maximum,
        coarse_axis_spacing_m=coarse_spacing,
    )

    refined_grid, refined_grid_diagnostics = regular_grid_in_bounds(
        local_minimum,
        local_maximum,
        requested_voxel_size_m=_FALLBACK_LOCAL_REQUESTED_VOXEL_SIZE_M,
        maximum_point_count=_FALLBACK_MAXIMUM_LOCAL_GRID_POINT_COUNT,
    )
    refined_spacing = refined_grid_diagnostics["effective_axis_spacing_m"]
    refined_hull, _refined_accepted, refined_carving = (
        _carve_candidate_points_with_footprints(
            refined_grid,
            masks_by_camera,
            intrinsics_by_camera,
            camera_to_world_by_camera,
            axis_spacing_m=refined_spacing,
        )
    )
    _require_geometry(
        len(refined_hull) >= _FALLBACK_MINIMUM_REFINED_SURFACE_POINT_COUNT,
        "projected-footprint refined hull is too small",
    )
    refined_keep, refined_surface, refined_components = _largest_grid_component(
        refined_hull,
        bounds_minimum=local_minimum,
        bounds_maximum=local_maximum,
        grid_shape=refined_grid_diagnostics["grid_shape"],
    )
    _require_geometry(
        refined_components["largest_component_fraction"]
        >= config.minimum_largest_component_fraction,
        "projected-footprint refined hull is dominated by disconnected clutter",
    )
    hull = np.asarray(refined_hull[refined_keep], dtype=np.float64)
    all_surface_points = np.asarray(
        refined_hull[refined_keep][refined_surface[refined_keep]], dtype=np.float64
    )
    _require_geometry(
        len(all_surface_points) >= _FALLBACK_MINIMUM_REFINED_SURFACE_POINT_COUNT,
        "too few projected-footprint refined surface points",
    )

    stability_grid, stability_grid_diagnostics = regular_grid_in_bounds(
        local_minimum,
        local_maximum,
        requested_voxel_size_m=_FALLBACK_STABILITY_REQUESTED_VOXEL_SIZE_M,
        maximum_point_count=_FALLBACK_MAXIMUM_LOCAL_GRID_POINT_COUNT,
    )
    stability_spacing = stability_grid_diagnostics["effective_axis_spacing_m"]
    stability_hull, _stability_accepted, stability_carving = (
        _carve_candidate_points_with_footprints(
            stability_grid,
            masks_by_camera,
            intrinsics_by_camera,
            camera_to_world_by_camera,
            axis_spacing_m=stability_spacing,
        )
    )
    _require_geometry(
        len(stability_hull) >= _FALLBACK_MINIMUM_REFINED_SURFACE_POINT_COUNT,
        "projected-footprint stability hull is too small",
    )
    stability_keep, _stability_surface, stability_components = _largest_grid_component(
        stability_hull,
        bounds_minimum=local_minimum,
        bounds_maximum=local_maximum,
        grid_shape=stability_grid_diagnostics["grid_shape"],
    )
    stability_component = np.asarray(stability_hull[stability_keep], dtype=np.float64)
    _require_geometry(
        stability_components["largest_component_fraction"]
        >= config.minimum_largest_component_fraction,
        "projected-footprint stability hull is dominated by disconnected clutter",
    )
    refined_voxel_volume = float(np.prod(np.asarray(refined_spacing)))
    stability_voxel_volume = float(np.prod(np.asarray(stability_spacing)))
    refined_physical_volume = float(len(hull) * refined_voxel_volume)
    stability_physical_volume = float(len(stability_component) * stability_voxel_volume)
    scale_stability = (
        min(refined_physical_volume, stability_physical_volume)
        / max(refined_physical_volume, stability_physical_volume)
        if max(refined_physical_volume, stability_physical_volume) > 0.0
        else 0.0
    )
    _require_geometry(
        scale_stability >= _FALLBACK_MINIMUM_SCALE_STABILITY,
        "projected-footprint local hull is not resolution stable",
    )

    rgb_stack = np.stack(
        [np.asarray(rgb_by_camera[camera], dtype=np.uint8) for camera in cameras]
    )
    mask_stack = np.stack(
        [np.asarray(masks_by_camera[camera], dtype=bool) for camera in cameras]
    )
    intrinsics_stack = np.stack(
        [
            np.asarray(intrinsics_by_camera[camera], dtype=np.float64)
            for camera in cameras
        ]
    )
    extrinsics_stack = np.stack(
        [
            np.asarray(camera_to_world_by_camera[camera], dtype=np.float64)
            for camera in cameras
        ]
    )
    projection_stack = np.stack(
        [
            _projection_matrix(
                intrinsics_by_camera[camera], camera_to_world_by_camera[camera]
            )
            for camera in cameras
        ]
    )

    footprint_surface_hits: list[np.ndarray] = []
    surface_camera_qa: list[dict[str, Any]] = []
    raw_containment_values = []
    footprint_containment_values = []
    for camera_index, camera in enumerate(cameras):
        raw_hits, raw_visible = _point_mask_hits(
            all_surface_points,
            mask_stack[camera_index],
            intrinsics_stack[camera_index],
            extrinsics_stack[camera_index],
        )
        footprint_hits, footprint_visible, footprint_qa = _projected_footprint_hits(
            all_surface_points,
            mask_stack[camera_index],
            intrinsics_stack[camera_index],
            extrinsics_stack[camera_index],
            axis_spacing_m=refined_spacing,
        )
        raw_containment = (
            float(np.count_nonzero(raw_hits) / np.count_nonzero(raw_visible))
            if np.any(raw_visible)
            else 0.0
        )
        footprint_containment = (
            float(
                np.count_nonzero(footprint_hits) / np.count_nonzero(footprint_visible)
            )
            if np.any(footprint_visible)
            else 0.0
        )
        footprint_surface_hits.append(footprint_hits)
        raw_containment_values.append(raw_containment)
        footprint_containment_values.append(footprint_containment)
        surface_camera_qa.append(
            {
                "camera": camera,
                "visible_refined_surface_point_count": int(
                    np.count_nonzero(raw_visible)
                ),
                "inside_raw_mask_refined_surface_point_count": int(
                    np.count_nonzero(raw_hits)
                ),
                "raw_hull_mask_containment": raw_containment,
                "footprint_hull_mask_containment": footprint_containment,
                "refined_surface_footprint": footprint_qa,
            }
        )
    footprint_hit_stack = np.stack(footprint_surface_hits)
    _require_geometry(
        np.all(np.any(footprint_hit_stack, axis=0)),
        "one or more refined surface points have no footprint color support",
    )
    surface_points = _sample_surface_points(
        all_surface_points,
        count=config.object_point_count,
        rng_seed=config.rng_seed,
    )
    sampled_colors = np.full((len(cameras), len(surface_points), 3), np.nan)
    sampled_raw_center_support = np.zeros(
        (len(cameras), len(surface_points)), dtype=bool
    )
    sampled_footprint_support = np.zeros(
        (len(cameras), len(surface_points)), dtype=bool
    )
    depth_maps = []
    depth_valid = []
    depth_coverage_values = []
    camera_qa = []
    for camera_index, camera in enumerate(cameras):
        mask = mask_stack[camera_index]
        raw_inside, visible = _point_mask_hits(
            surface_points,
            mask,
            intrinsics_stack[camera_index],
            extrinsics_stack[camera_index],
        )
        footprint_inside, footprint_pixels = _nearest_projected_footprint_mask_pixels(
            surface_points,
            mask,
            intrinsics_stack[camera_index],
            extrinsics_stack[camera_index],
            axis_spacing_m=refined_spacing,
        )
        sampled_raw_center_support[camera_index] = raw_inside
        sampled_footprint_support[camera_index] = footprint_inside
        color_ids = np.flatnonzero(footprint_inside)
        if len(color_ids):
            sampled_colors[camera_index, color_ids] = rgb_stack[
                camera_index,
                footprint_pixels[color_ids, 1],
                footprint_pixels[color_ids, 0],
            ]
        depth_radius, depth_radius_qa = _resolution_aware_depth_radius(
            hull,
            intrinsics_stack[camera_index],
            extrinsics_stack[camera_index],
            axis_spacing_m=refined_spacing,
            image_shape=mask.shape,
        )
        depth_map, valid_map, depth_qa = _render_depth(
            hull,
            mask,
            intrinsics_stack[camera_index],
            extrinsics_stack[camera_index],
            radius=depth_radius,
        )
        depth_maps.append(depth_map)
        depth_valid.append(valid_map)
        depth_coverage_values.append(float(depth_qa["depth_mask_coverage"]))
        camera_qa.append(
            {
                **surface_camera_qa[camera_index],
                "mask_area_pixels": int(np.count_nonzero(mask)),
                "mask_area_fraction": float(np.mean(mask)),
                "visible_sampled_surface_point_count": int(np.count_nonzero(visible)),
                "inside_raw_mask_sampled_surface_point_count": int(
                    np.count_nonzero(raw_inside)
                ),
                "inside_footprint_sampled_surface_point_count": int(
                    np.count_nonzero(footprint_inside)
                ),
                "depth_radius": depth_radius_qa,
                **depth_qa,
            }
        )
    raw_center_support_count = np.sum(sampled_raw_center_support, axis=0)
    support_count = np.sum(sampled_footprint_support, axis=0)
    _require_geometry(
        np.all(support_count > 0)
        and np.all(np.all(np.isfinite(sampled_colors), axis=2).sum(axis=0) > 0),
        "one or more fallback surface points have no footprint RGB support",
    )
    object_colors = np.nanmedian(sampled_colors, axis=0).astype(np.uint8)
    raw_containment = np.asarray(raw_containment_values, dtype=np.float64)
    footprint_containment = np.asarray(footprint_containment_values, dtype=np.float64)
    coverage = np.asarray(depth_coverage_values, dtype=np.float64)
    quantiles = np.percentile(surface_points, [1.0, 50.0, 99.0], axis=0)
    surface_area_proxy = float(
        len(all_surface_points) * refined_voxel_volume ** (2.0 / 3.0)
    )
    gates = {
        "camera_count": len(cameras) >= config.minimum_camera_count,
        "coarse_strict_consensus_vote_count": (
            coarse_carving["required_vote_count"] == _FALLBACK_STRICT_CONSENSUS_VOTES
        ),
        "refined_strict_consensus_vote_count": (
            refined_carving["required_vote_count"] == _FALLBACK_STRICT_CONSENSUS_VOTES
        ),
        "stability_strict_consensus_vote_count": (
            stability_carving["required_vote_count"] == _FALLBACK_STRICT_CONSENSUS_VOTES
        ),
        "coarse_connected_core_point_count": (
            coarse_components["largest_component_point_count"]
            >= _FALLBACK_MINIMUM_COARSE_COMPONENT_POINT_COUNT
        ),
        "coarse_largest_component_fraction": (
            coarse_components["largest_component_fraction"]
            >= config.minimum_largest_component_fraction
        ),
        "refined_surface_point_count": (
            len(all_surface_points) >= _FALLBACK_MINIMUM_REFINED_SURFACE_POINT_COUNT
        ),
        "refined_largest_component_fraction": (
            refined_components["largest_component_fraction"]
            >= config.minimum_largest_component_fraction
        ),
        "refined_grid_not_coarsened": not bool(
            refined_grid_diagnostics["coarsened_for_grid_cap"]
        ),
        "local_scale_stability": (scale_stability >= _FALLBACK_MINIMUM_SCALE_STABILITY),
        "stability_largest_component_fraction": (
            stability_components["largest_component_fraction"]
            >= config.minimum_largest_component_fraction
        ),
        "stability_grid_not_coarsened": not bool(
            stability_grid_diagnostics["coarsened_for_grid_cap"]
        ),
        "raw_median_hull_mask_containment": (
            float(np.median(raw_containment))
            >= config.minimum_median_hull_mask_containment
        ),
        "median_depth_mask_coverage": (
            float(np.median(coverage)) >= config.minimum_median_depth_mask_coverage
        ),
        "all_refined_surface_points_have_footprint_support": bool(
            np.all(np.any(footprint_hit_stack, axis=0))
        ),
        "all_sampled_surface_points_colored": bool(np.all(support_count > 0)),
    }
    _require(
        set(gates) == set(_FALLBACK_ACCEPTANCE_GATE_NAMES),
        "frame-zero geometry fallback acceptance-gate contract changed",
    )
    arrays = {
        "frame_indices": np.asarray([0], dtype=np.int64),
        "camera_names": np.asarray(cameras),
        "rgb_frame0": rgb_stack,
        "mask_frame0": mask_stack,
        "depth_frame0_m": np.stack(depth_maps).astype(np.float32),
        "depth_valid_frame0": np.stack(depth_valid),
        "intrinsics": intrinsics_stack,
        "camera_to_world": extrinsics_stack,
        "projection_world_to_pixel": projection_stack,
        "object_points_world_m": surface_points.astype(np.float32),
        "object_colors_rgb": object_colors,
        "object_color_support_count": support_count.astype(np.uint8),
        "visual_hull_points_world_m": hull.astype(np.float32),
    }
    diagnostics = {
        "strategy": strategy,
        "fallback_policy_id": (
            FRAME_ZERO_REFERENCE_OPTIONAL_FALLBACK_POLICY_ID
            if strategy == FRAME_ZERO_REFERENCE_OPTIONAL_GEOMETRY_STRATEGY
            else FRAME_ZERO_GEOMETRY_FALLBACK_POLICY_ID
        ),
        "coarse_grid": coarse_grid_diagnostics,
        "coarse_carving": coarse_carving,
        "coarse_components": coarse_components,
        "local_bounds": local_bounds,
        "refined_grid": refined_grid_diagnostics,
        "refined_carving": refined_carving,
        "refined_components": refined_components,
        "stability_grid": stability_grid_diagnostics,
        "stability_carving": stability_carving,
        "stability_components": stability_components,
        "local_resolution_stability": {
            "refined_requested_voxel_size_m": (_FALLBACK_LOCAL_REQUESTED_VOXEL_SIZE_M),
            "stability_requested_voxel_size_m": (
                _FALLBACK_STABILITY_REQUESTED_VOXEL_SIZE_M
            ),
            "maximum_local_grid_point_count": (
                _FALLBACK_MAXIMUM_LOCAL_GRID_POINT_COUNT
            ),
            "refined_component_physical_volume_m3": refined_physical_volume,
            "stability_component_physical_volume_m3": stability_physical_volume,
            "symmetric_volume_ratio": scale_stability,
            "minimum_symmetric_volume_ratio": _FALLBACK_MINIMUM_SCALE_STABILITY,
        },
        "visual_hull_point_count": len(hull),
        "visual_hull_points_sha256": _sha256_array(hull.astype(np.float32)),
        "refined_surface_point_count": len(all_surface_points),
        "refined_component_physical_volume_m3": refined_physical_volume,
        "refined_surface_area_proxy_m2": surface_area_proxy,
        "object_point_count": len(surface_points),
        "object_world_m": {
            "q01": quantiles[0].tolist(),
            "median": quantiles[1].tolist(),
            "q99": quantiles[2].tolist(),
            "q01_to_q99_span": (quantiles[2] - quantiles[0]).tolist(),
        },
        "object_color_support_count": {
            "minimum": int(np.min(support_count)),
            "median": float(np.median(support_count)),
            "maximum": int(np.max(support_count)),
        },
        "raw_center_object_color_support_count": {
            "minimum": int(np.min(raw_center_support_count)),
            "median": float(np.median(raw_center_support_count)),
            "maximum": int(np.max(raw_center_support_count)),
        },
        "depth_mask_coverage": {
            "minimum": float(np.min(coverage)),
            "median": float(np.median(coverage)),
            "maximum": float(np.max(coverage)),
        },
        "raw_hull_mask_containment": {
            "minimum": float(np.min(raw_containment)),
            "median": float(np.median(raw_containment)),
            "maximum": float(np.max(raw_containment)),
        },
        "footprint_hull_mask_containment": {
            "minimum": float(np.min(footprint_containment)),
            "median": float(np.median(footprint_containment)),
            "maximum": float(np.max(footprint_containment)),
        },
        "per_camera": camera_qa,
        "acceptance_gates": gates,
        "geometry_qa_passed": all(gates.values()),
    }
    failed_gates = sorted(name for name, passed in gates.items() if not passed)
    _require_geometry(
        diagnostics["geometry_qa_passed"],
        "projected-footprint frame-zero geometry failed QA: " + ",".join(failed_gates),
    )
    return arrays, diagnostics


def _selected_proposal_audit(
    diagnostics: Sequence[Mapping[str, Any]],
    masks_by_camera: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    by_camera = {str(record["camera"]): record for record in diagnostics}
    records = []
    for camera in sorted(masks_by_camera):
        diagnostic = by_camera[camera]
        selected = diagnostic["selected"]
        records.append(
            {
                "camera": camera,
                "candidate_index": int(selected["candidate_index"]),
                "automatic_candidate_count": int(
                    diagnostic["automatic_candidate_count"]
                ),
                "eligible_candidate_count": int(diagnostic["eligible_candidate_count"]),
                "mask_sha256": _sha256_array(
                    np.asarray(masks_by_camera[camera], dtype=bool)
                ),
            }
        )
    return records


def _fallback_attempt_pass_record(
    strategy: str,
    geometry_qa: Mapping[str, Any],
    masks_by_camera: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    footprint_payload = {
        stage: geometry_qa[stage]["per_camera"]
        for stage in ("coarse_carving", "refined_carving", "stability_carving")
    }
    return {
        "strategy": strategy,
        "status": "passed",
        "selected_camera_count": len(masks_by_camera),
        "selected_mask_set_sha256": _mask_set_sha256(masks_by_camera),
        "coarse_peak_vote_count": int(geometry_qa["coarse_carving"]["peak_vote_count"]),
        "coarse_required_vote_count": int(
            geometry_qa["coarse_carving"]["required_vote_count"]
        ),
        "coarse_connected_core_point_count": int(
            geometry_qa["coarse_components"]["largest_component_point_count"]
        ),
        "refined_surface_point_count": int(geometry_qa["refined_surface_point_count"]),
        "refined_required_vote_count": int(
            geometry_qa["refined_carving"]["required_vote_count"]
        ),
        "stability_required_vote_count": int(
            geometry_qa["stability_carving"]["required_vote_count"]
        ),
        "refined_grid_coarsened_for_cap": bool(
            geometry_qa["refined_grid"]["coarsened_for_grid_cap"]
        ),
        "stability_grid_coarsened_for_cap": bool(
            geometry_qa["stability_grid"]["coarsened_for_grid_cap"]
        ),
        "refined_effective_axis_spacing_m": list(
            geometry_qa["refined_grid"]["effective_axis_spacing_m"]
        ),
        "stability_effective_axis_spacing_m": list(
            geometry_qa["stability_grid"]["effective_axis_spacing_m"]
        ),
        "raw_median_hull_mask_containment": float(
            geometry_qa["raw_hull_mask_containment"]["median"]
        ),
        "footprint_median_hull_mask_containment": float(
            geometry_qa["footprint_hull_mask_containment"]["median"]
        ),
        "median_depth_mask_coverage": float(
            geometry_qa["depth_mask_coverage"]["median"]
        ),
        "local_scale_stability": float(
            geometry_qa["local_resolution_stability"]["symmetric_volume_ratio"]
        ),
        "stability_component_count": int(
            geometry_qa["stability_components"]["component_count"]
        ),
        "stability_largest_component_fraction": float(
            geometry_qa["stability_components"]["largest_component_fraction"]
        ),
        "projected_footprint_diagnostics_sha256": hashlib.sha256(
            _canonical_bytes(footprint_payload)
        ).hexdigest(),
        "geometry_qa_sha256": hashlib.sha256(_canonical_bytes(geometry_qa)).hexdigest(),
    }


def _fallback_attempt_failure_record(
    strategy: str, error: FrameZeroGeometryQAError
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "status": "failed",
        "error_type": type(error).__name__,
        "reason": str(error),
    }


def _load_calibration(
    layout: AlignedEpisodeLayout,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    intrinsic_path = layout.file(
        "undistorted_intrinsics.npy", label="camera intrinsics"
    )
    extrinsic_path = layout.file("extrinsics.npy", label="camera extrinsics")
    intrinsics_raw = np.load(intrinsic_path, allow_pickle=True).item()
    extrinsics_raw = np.load(extrinsic_path, allow_pickle=True).item()
    _require(isinstance(intrinsics_raw, Mapping), "intrinsics archive is not a mapping")
    _require(isinstance(extrinsics_raw, Mapping), "extrinsics archive is not a mapping")
    _require(
        set(intrinsics_raw) == set(extrinsics_raw), "calibration camera sets differ"
    )
    intrinsics = {
        str(camera): np.asarray(value, dtype=np.float64)
        for camera, value in intrinsics_raw.items()
    }
    extrinsics = {
        str(camera): np.asarray(value, dtype=np.float64)
        for camera, value in extrinsics_raw.items()
    }
    for camera in intrinsics:
        _require(intrinsics[camera].shape == (3, 3), f"invalid intrinsics for {camera}")
        _require(extrinsics[camera].shape == (4, 4), f"invalid extrinsics for {camera}")
        _require(
            np.isfinite(intrinsics[camera]).all()
            and np.isfinite(extrinsics[camera]).all(),
            f"non-finite calibration for {camera}",
        )
    return (
        intrinsics,
        extrinsics,
        {
            "intrinsics": _file_record(intrinsic_path),
            "extrinsics": _file_record(extrinsic_path),
        },
    )


def select_action_only_window(
    state: Deform360RobotKinematics,
    *,
    window_length_frames: int,
    prediction_frame_count: int,
    candidate_first_frame: int,
    candidate_stride_frames: int,
) -> dict[str, Any]:
    """Compatibility name for the shared, validated robot-state selector.

    The old two-array helper treated all five heterogeneous ``actions`` rows as
    points and averaged them.  Requiring a complete validated robot state makes
    that failure mode impossible: selection uses only ``T_worlds`` translation
    and ``openings`` through the shared contract.
    """

    _require(
        isinstance(state, Deform360RobotKinematics),
        "window selection requires a complete validated robot state",
    )
    return select_robot_kinematics_window(
        state,
        window_length_frames=window_length_frames,
        prediction_frame_count=prediction_frame_count,
        candidate_first_frame=candidate_first_frame,
        candidate_stride_frames=candidate_stride_frames,
    )


def _slice_known_action(
    robot_path: Path,
    output_path: Path,
    *,
    config: FrameZeroAssetConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_state = load_robot_kinematics_archive(robot_path)
    selection_audit = select_action_only_window(
        source_state,
        window_length_frames=config.action_window_length_frames,
        prediction_frame_count=config.prediction_frame_count,
        candidate_first_frame=config.action_candidate_first_frame,
        candidate_stride_frames=config.action_candidate_stride_frames,
    )
    start, stop = selection_audit["prediction_raw_frame_range_half_open"]
    selected_state = slice_robot_kinematics(
        source_state,
        start_frame=int(start),
        frame_count=int(stop) - int(start),
    )
    arrays = selected_state.archive_arrays()
    np.savez_compressed(output_path, **arrays)
    selected_bundle = _file_record(output_path)
    exact_slice_audit = validate_selected_robot_kinematics_bundle(
        output_path,
        source_state=source_state,
        prediction_start_frame=int(start),
        prediction_frame_count=config.prediction_frame_count,
    )
    alignment: dict[str, Any] = {
        "policy_id": ROBOT_KINEMATICS_WINDOW_POLICY_ID,
        "trajectory_semantics": ROBOT_KINEMATICS_WINDOW_CONTRACT[
            "trajectory_semantics"
        ],
        "selection_audit": selection_audit,
        # Compatibility mirrors used by the v2 held consumers.  They are
        # checked against the nested shared audit by the local validator.
        "selected_raw_frame_range_half_open": list(
            selection_audit["selected_raw_frame_range_half_open"]
        ),
        "prediction_raw_frame_range_half_open": list(
            selection_audit["prediction_raw_frame_range_half_open"]
        ),
        "tracking_tail_frame_count": selection_audit["tracking_tail_frame_count"],
        "source_robot_frame_count": source_state.frame_count,
        "prediction_frame_count": config.prediction_frame_count,
        "selected_robot_kinematics_bundle": selected_bundle,
        "selected_action_bundle": selected_bundle,
        "selected_action_bundle_is_compatibility_alias": True,
        "selected_action_arrays": _bundle_array_records(arrays),
        "selected_bundle_exact_slice_audit": exact_slice_audit,
    }
    return arrays, alignment


def _action_inputs(
    layout: AlignedEpisodeLayout,
) -> tuple[dict[str, dict[str, Any]], Path]:
    robot = layout.file("robot", "robot.npz", label="realized robot kinematics")
    metadata = layout.file(
        "robot",
        "robot.meta.json",
        label="realized robot kinematics metadata",
    )
    return {
        "robot_trajectory": _file_record(robot),
        "robot_metadata": _file_record(metadata),
    }, robot


def _bundle_array_records(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": np.asarray(value).dtype.str,
            "sha256": _sha256_array(np.asarray(value)),
        }
        for name, value in sorted(arrays.items())
    }


def _validate_camera_selection_contract(payload: Mapping[str, Any]) -> None:
    config = payload.get("config")
    _require(isinstance(config, Mapping), "frame-zero config is missing")
    minimum_camera_count = config.get("minimum_camera_count")
    _require(
        isinstance(minimum_camera_count, int)
        and not isinstance(minimum_camera_count, bool)
        and minimum_camera_count >= 3,
        "invalid minimum selected camera count",
    )
    policy = payload.get("camera_policy")
    policy_keys = {
        "policy_id",
        "rule",
        "reference_camera",
        "minimum_selected_camera_count",
        "candidate_cameras",
        "candidate_camera_count",
        "selected_cameras",
        "selected_camera_count",
        "abstained_cameras",
        "abstained_camera_count",
    }
    _require(
        isinstance(policy, Mapping) and set(policy) == policy_keys,
        "frame-zero camera selection policy changed",
    )
    reference_optional = (
        policy.get("policy_id")
        == FRAME_ZERO_REFERENCE_OPTIONAL_CAMERA_SELECTION_POLICY_ID
    )
    _require(
        (
            reference_optional
            and policy.get("rule")
            == FRAME_ZERO_REFERENCE_OPTIONAL_CAMERA_SELECTION_RULE
        )
        or (
            not reference_optional
            and policy.get("policy_id") == FRAME_ZERO_CAMERA_SELECTION_POLICY_ID
            and policy.get("rule") == FRAME_ZERO_CAMERA_SELECTION_RULE
        ),
        "frame-zero camera selection policy changed",
    )
    candidate_cameras = policy.get("candidate_cameras")
    selected_cameras = policy.get("selected_cameras")
    abstained_cameras = policy.get("abstained_cameras")
    _require(
        isinstance(candidate_cameras, list)
        and isinstance(selected_cameras, list)
        and isinstance(abstained_cameras, list)
        and all(isinstance(camera, str) and camera for camera in candidate_cameras)
        and all(isinstance(camera, str) and camera for camera in selected_cameras)
        and all(isinstance(camera, str) and camera for camera in abstained_cameras)
        and candidate_cameras == sorted(set(candidate_cameras))
        and selected_cameras == sorted(set(selected_cameras))
        and abstained_cameras == sorted(set(abstained_cameras))
        and set(candidate_cameras) == set(selected_cameras) | set(abstained_cameras)
        and not set(selected_cameras) & set(abstained_cameras),
        "frame-zero camera selection sets are invalid",
    )
    reference_camera = policy.get("reference_camera")
    _require(
        isinstance(reference_camera, str)
        and reference_camera == config.get("reference_camera")
        and (reference_optional or reference_camera in selected_cameras)
        and policy.get("minimum_selected_camera_count") == minimum_camera_count
        and policy.get("candidate_camera_count") == len(candidate_cameras)
        and policy.get("selected_camera_count") == len(selected_cameras)
        and policy.get("abstained_camera_count") == len(abstained_cameras)
        and len(selected_cameras) >= minimum_camera_count,
        "frame-zero selected camera requirement failed",
    )
    sam2 = payload.get("sam2")
    diagnostics = sam2.get("view_diagnostics") if isinstance(sam2, Mapping) else None
    _require(
        isinstance(diagnostics, list)
        and sam2.get("view_diagnostics_sha256")
        == frame_zero_view_diagnostics_sha256(
            diagnostics, policy_id=str(policy["policy_id"])
        ),
        "frame-zero view diagnostics checksum changed",
    )
    _require(
        len(diagnostics) == len(candidate_cameras)
        and all(isinstance(record, Mapping) for record in diagnostics),
        "frame-zero per-camera diagnostics are incomplete",
    )
    _require(
        [record.get("camera") for record in diagnostics] == candidate_cameras,
        "frame-zero per-camera diagnostic order changed",
    )
    diagnostic_selected: list[str] = []
    diagnostic_abstained: list[str] = []
    allowed_abstention_reasons = {
        _NO_AUTOMATIC_CANDIDATES,
        _NO_MASK_THRESHOLD_CANDIDATES,
        _NO_REFERENCE_CONSISTENT_CANDIDATES,
        _COMMON_INLIER_ABSTENTION_REASON,
    }
    for record in diagnostics:
        automatic_count = record.get("automatic_candidate_count")
        eligible_count = record.get("eligible_candidate_count")
        rejected_count = record.get("rejected_candidate_count")
        rejection_counts = record.get("rejection_counts")
        maximum_similarity = record.get("maximum_reference_appearance_similarity")
        _require(
            all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (automatic_count, eligible_count, rejected_count)
            )
            and automatic_count == eligible_count + rejected_count
            and isinstance(rejection_counts, Mapping)
            and set(rejection_counts)
            == {"mask_threshold", "reference_appearance_threshold", "total"}
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in rejection_counts.values()
            )
            and rejection_counts["mask_threshold"]
            + rejection_counts["reference_appearance_threshold"]
            == rejection_counts["total"]
            == rejected_count,
            "invalid per-camera candidate rejection counts",
        )
        _require(
            isinstance(maximum_similarity, (int, float))
            and not isinstance(maximum_similarity, bool)
            and math.isfinite(float(maximum_similarity))
            and 0.0 <= float(maximum_similarity) <= 1.0,
            "invalid per-camera maximum appearance similarity",
        )
        camera = str(record["camera"])
        geometry_inlier = record.get("geometry_inlier_selection")
        if record.get("view_selected") is True:
            _require(
                record.get("abstained") is False
                and record.get("abstention_reason") is None
                and eligible_count >= 1
                and isinstance(record.get("selected"), Mapping),
                "selected camera diagnostics are inconsistent",
            )
            if geometry_inlier is not None:
                _require(
                    isinstance(geometry_inlier, Mapping)
                    and geometry_inlier.get("retained") is True,
                    "selected geometry-inlier diagnostics are inconsistent",
                )
            diagnostic_selected.append(camera)
        else:
            inlier_exclusion = (
                record.get("abstention_reason") == _COMMON_INLIER_ABSTENTION_REASON
            )
            _require(
                record.get("view_selected") is False
                and record.get("abstained") is True
                and record.get("abstention_reason") in allowed_abstention_reasons
                and (
                    (
                        inlier_exclusion
                        and eligible_count >= 1
                        and isinstance(geometry_inlier, Mapping)
                        and geometry_inlier.get("retained") is False
                        and geometry_inlier.get("candidate_index") is not None
                    )
                    or (not inlier_exclusion and eligible_count == 0)
                )
                and record.get("selected") is None,
                "abstained camera diagnostics are inconsistent",
            )
            diagnostic_abstained.append(camera)
    _require(
        diagnostic_selected == selected_cameras
        and diagnostic_abstained == abstained_cameras,
        "frame-zero camera policy differs from its diagnostics",
    )


def _proposal_audit_cameras_and_hash(
    value: object,
    *,
    expected_cameras: Sequence[str] | None,
) -> tuple[list[str], str]:
    _require(isinstance(value, list) and bool(value), "proposal audit is missing")
    proposal_keys = {
        "camera",
        "candidate_index",
        "automatic_candidate_count",
        "eligible_candidate_count",
        "mask_sha256",
    }
    cameras: list[str] = []
    mask_bindings: dict[str, str] = {}
    for record in value:
        _require(
            isinstance(record, Mapping)
            and set(record) == proposal_keys
            and isinstance(record.get("camera"), str)
            and isinstance(record.get("candidate_index"), int)
            and not isinstance(record.get("candidate_index"), bool)
            and int(record["candidate_index"]) >= 0
            and isinstance(record.get("automatic_candidate_count"), int)
            and not isinstance(record.get("automatic_candidate_count"), bool)
            and isinstance(record.get("eligible_candidate_count"), int)
            and not isinstance(record.get("eligible_candidate_count"), bool)
            and 1
            <= int(record["eligible_candidate_count"])
            <= int(record["automatic_candidate_count"])
            and _valid_sha256(record.get("mask_sha256")),
            "invalid selected proposal audit",
        )
        camera = str(record["camera"])
        cameras.append(camera)
        mask_bindings[camera] = str(record["mask_sha256"])
    _require(cameras == sorted(set(cameras)), "proposal camera order changed")
    if expected_cameras is not None:
        _require(cameras == list(expected_cameras), "proposal cameras changed")
    return cameras, hashlib.sha256(_canonical_bytes(mask_bindings)).hexdigest()


def _validate_reference_optional_assignment_audit(
    assignment: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    semantic_proposals: Sequence[Mapping[str, Any]],
    require_bounded_subset_audit: bool,
) -> None:
    """Validate exhaustive all-camera exact-eight selection and its cross-links."""

    common_keys = {
        "policy_id",
        "strategy",
        "reference_seed_policy",
        "reference_seed_limit",
        "inlier_selection_rule",
        "search_hit_representation",
        "evaluated_reference_seed_count",
        "strict_consensus_vote_count",
        "grid",
        "proposal_count_by_camera",
        "proposal_inventory_sha256",
        "reference_candidate_ranking",
        "reference_seed_evaluations",
        "reference_seed_objective_order",
        "selected_reference_seed_rank",
        "selected_reference_candidate_index",
        "selected_reference_mask_sha256",
        "global_selected_common_voxel_flat_index",
        "global_selected_common_voxel_world_m",
        "global_selected_common_voxel_support_count",
        "global_selected_strict_component_voxel_count",
        "global_selected_strict_component_sha256",
        "local_refinement",
        "local_union_peak_voxel_flat_index",
        "local_union_peak_voxel_world_m",
        "local_union_peak_support_count",
        "local_union_strict_component_voxel_count",
        "local_union_strict_component_sha256",
        "candidate_objective_order",
        "camera_inlier_ranking",
        "evaluated_exact_eight_subset_count",
        "exact_eight_subset_objective_order",
        "exact_eight_subset_evaluations",
        "selected_exact_eight_cameras",
        "selected_common_voxel_flat_index",
        "selected_common_voxel_world_m",
        "selected_common_voxel_support_count",
        "selected_exact_common_voxel_count",
        "selected_exact_common_mask_sha256",
        "selected_strict_component_voxel_count",
        "selected_strict_component_sha256",
        "selected_proposals",
        "selected_mask_set_sha256",
        "artifact_sha256",
    }
    _require(
        isinstance(assignment, Mapping)
        and set(assignment) == common_keys
        and assignment.get("artifact_sha256") == artifact_sha256(assignment)
        and assignment.get("policy_id")
        == FRAME_ZERO_REFERENCE_OPTIONAL_FALLBACK_POLICY_ID
        and assignment.get("strategy")
        == FRAME_ZERO_REFERENCE_OPTIONAL_ASSIGNMENT_STRATEGY
        and assignment.get("reference_seed_policy") == _FALLBACK_REFERENCE_SEED_POLICY
        and assignment.get("reference_seed_limit") == _FALLBACK_REFERENCE_SEED_COUNT
        and assignment.get("inlier_selection_rule")
        == _REFERENCE_OPTIONAL_INLIER_SELECTION_RULE
        and assignment.get("search_hit_representation") == "nearest-pixel-raw-center"
        and assignment.get("strict_consensus_vote_count")
        == _FALLBACK_STRICT_CONSENSUS_VOTES,
        "invalid reference-optional assignment header",
    )
    candidate_cameras = payload["camera_policy"]["candidate_cameras"]
    selected_cameras = payload["camera_policy"]["selected_cameras"]
    reference_camera = payload["camera_policy"]["reference_camera"]
    _require(
        assignment.get("selected_exact_eight_cameras") == selected_cameras
        and len(selected_cameras) == _FALLBACK_STRICT_CONSENSUS_VOTES,
        "reference-optional selected cameras changed",
    )
    grid = assignment.get("grid")
    local = assignment.get("local_refinement")
    _require(
        isinstance(grid, Mapping)
        and grid.get("grid_shape") == [_FALLBACK_COMMON_GRID_AXIS_COUNT] * 3
        and grid.get("grid_point_count") == _FALLBACK_COMMON_GRID_AXIS_COUNT**3
        and _valid_sha256(grid.get("grid_points_sha256"))
        and isinstance(local, Mapping)
        and local.get("requested_voxel_size_m")
        == _FALLBACK_COMMON_LOCAL_REQUESTED_VOXEL_SIZE_M
        and isinstance(local.get("grid"), Mapping)
        and local["grid"].get("coarsened_for_grid_cap") is False
        and _valid_sha256(assignment.get("proposal_inventory_sha256"))
        and _valid_sha256(assignment.get("selected_exact_common_mask_sha256"))
        and _valid_sha256(assignment.get("selected_strict_component_sha256"))
        and assignment.get("selected_common_voxel_support_count")
        == _FALLBACK_STRICT_CONSENSUS_VOTES
        and isinstance(assignment.get("selected_exact_common_voxel_count"), int)
        and assignment["selected_exact_common_voxel_count"] >= 1
        and isinstance(assignment.get("selected_strict_component_voxel_count"), int)
        and assignment["selected_strict_component_voxel_count"] >= 1,
        "reference-optional geometry search binding changed",
    )
    inliers = assignment.get("camera_inlier_ranking")
    inlier_keys = {
        "camera",
        "candidate_index",
        "mask_sha256",
        "seed_conditioned_candidate_count",
        "fixed_reference",
        "nonreference_coverage_rank",
        "retained",
        "raw_component_hit_count",
        "raw_component_coverage",
        "raw_component_precision",
        "hits_local_union_peak",
        "semantic_score",
        "candidate_lexicographic_objective",
    }
    _require(
        isinstance(inliers, list)
        and len(inliers) == len(candidate_cameras)
        and all(
            isinstance(record, Mapping)
            and set(record) == inlier_keys
            and record.get("camera") == camera
            and isinstance(record.get("retained"), bool)
            and record.get("fixed_reference") == (camera == reference_camera)
            and isinstance(record.get("seed_conditioned_candidate_count"), int)
            and not isinstance(record.get("seed_conditioned_candidate_count"), bool)
            and record["seed_conditioned_candidate_count"] >= 0
            and isinstance(record.get("raw_component_hit_count"), int)
            and not isinstance(record.get("raw_component_hit_count"), bool)
            and record["raw_component_hit_count"] >= 0
            and isinstance(record.get("hits_local_union_peak"), bool)
            and all(
                isinstance(record.get(key), (int, float))
                and not isinstance(record.get(key), bool)
                and math.isfinite(float(record[key]))
                and 0.0 <= float(record[key]) <= 1.0
                for key in ("raw_component_coverage", "raw_component_precision")
            )
            for camera, record in zip(candidate_cameras, inliers, strict=True)
        ),
        "reference-optional camera-inlier audit changed",
    )
    eligible = []
    for record in inliers:
        if record["seed_conditioned_candidate_count"] > 0:
            _require(
                isinstance(record.get("candidate_index"), int)
                and not isinstance(record.get("candidate_index"), bool)
                and record["candidate_index"] >= 0
                and _valid_sha256(record.get("mask_sha256"))
                and isinstance(record.get("semantic_score"), (int, float))
                and not isinstance(record.get("semantic_score"), bool)
                and math.isfinite(float(record["semantic_score"]))
                and record.get("candidate_lexicographic_objective")
                == [
                    float(record["raw_component_coverage"]),
                    float(record["raw_component_precision"]),
                    int(bool(record["hits_local_union_peak"])),
                    float(record["semantic_score"]),
                    -int(record["candidate_index"]),
                ],
                "reference-optional candidate objective changed",
            )
            eligible.append(record)
        else:
            _require(
                record.get("candidate_index") is None
                and record.get("mask_sha256") is None
                and record.get("semantic_score") is None
                and record.get("candidate_lexicographic_objective") is None
                and record.get("retained") is False,
                "ineligible reference-optional candidate changed",
            )
    retained = [record for record in inliers if record["retained"]]
    selected_proposals = assignment.get("selected_proposals")
    _require(
        retained == selected_proposals
        and [record["camera"] for record in retained] == selected_cameras
        and [
            {
                "camera": record["camera"],
                "candidate_index": record["candidate_index"],
                "mask_sha256": record["mask_sha256"],
            }
            for record in retained
        ]
        == [
            {
                "camera": record["camera"],
                "candidate_index": record["candidate_index"],
                "mask_sha256": record["mask_sha256"],
            }
            for record in semantic_proposals
        ],
        "reference-optional selected proposal binding changed",
    )
    semantic_mask_set = hashlib.sha256(
        _canonical_bytes(
            {
                str(record["camera"]): str(record["mask_sha256"])
                for record in semantic_proposals
            }
        )
    ).hexdigest()
    _require(
        assignment.get("selected_mask_set_sha256") == semantic_mask_set,
        "reference-optional selected mask-set checksum changed",
    )
    subsets = assignment.get("exact_eight_subset_evaluations")
    eligible_cameras = [str(record["camera"]) for record in eligible]
    expected_subset_count = math.comb(
        len(eligible_cameras), _FALLBACK_STRICT_CONSENSUS_VOTES
    )
    if require_bounded_subset_audit:
        _require(
            expected_subset_count > 0,
            "reference-optional exhaustive subset audit is empty",
        )
        bounded_subsets = _validate_bounded_exact_eight_subset_audit(
            subsets,
            expected_record_count=expected_subset_count,
            expected_first_cameras=(
                eligible_cameras[:_FALLBACK_STRICT_CONSENSUS_VOTES]
            ),
            expected_last_cameras=(
                eligible_cameras[-_FALLBACK_STRICT_CONSENSUS_VOTES:]
            ),
            selected_cameras=selected_cameras,
            candidate_cameras=eligible_cameras,
            fixed_first_camera=None,
        )
        winner = bounded_subsets["selected_record"]
        valid_subsets = True
    else:
        expected_subsets = [
            list(combination)
            for combination in itertools.combinations(
                eligible_cameras, _FALLBACK_STRICT_CONSENSUS_VOTES
            )
        ]
        valid_subsets = bool(
            isinstance(subsets, list)
            and len(subsets) == len(expected_subsets)
            and [record.get("cameras") for record in subsets] == expected_subsets
            and all(
                _valid_exact_eight_subset_record(
                    record,
                    candidate_cameras=eligible_cameras,
                    fixed_first_camera=None,
                )
                for record in subsets
            )
        )
        winner = (
            min(
                subsets,
                key=lambda record: (
                    -record["largest_exact_component_voxel_count"],
                    -record["exact_common_voxel_count"],
                    -record["raw_component_coverage_sum"],
                    -record["semantic_score_sum"],
                    tuple(record["cameras"]),
                ),
            )
            if valid_subsets
            else None
        )
    _require(
        valid_subsets
        and assignment.get("evaluated_exact_eight_subset_count")
        == expected_subset_count
        and assignment.get("exact_eight_subset_objective_order")
        == [
            "largest_exact_component_voxel_count_desc",
            "exact_common_voxel_count_desc",
            "raw_component_coverage_sum_desc",
            "semantic_score_sum_desc",
            "camera_tuple_asc",
        ]
        and winner is not None,
        "reference-optional exhaustive subset audit changed",
    )
    _require(
        winner["cameras"] == selected_cameras
        and assignment.get("selected_exact_common_voxel_count")
        == winner["exact_common_voxel_count"]
        and assignment.get("selected_exact_common_mask_sha256")
        == winner["exact_common_mask_sha256"]
        and assignment.get("selected_strict_component_voxel_count")
        == winner["largest_exact_component_voxel_count"],
        "reference-optional winning subset changed",
    )
    diagnostics = payload["sam2"]["view_diagnostics"]
    _require(
        [record.get("geometry_inlier_selection") for record in diagnostics] == inliers,
        "reference-optional diagnostics/inlier cross-link changed",
    )
    ranking = assignment.get("reference_candidate_ranking")
    seeds = assignment.get("reference_seed_evaluations")
    selected_seed_rank = assignment.get("selected_reference_seed_rank")
    _require(
        isinstance(ranking, list)
        and isinstance(seeds, list)
        and 1 <= len(seeds) <= _FALLBACK_REFERENCE_SEED_COUNT
        and assignment.get("evaluated_reference_seed_count") == len(seeds)
        and isinstance(selected_seed_rank, int)
        and not isinstance(selected_seed_rank, bool)
        and 0 <= selected_seed_rank < len(seeds)
        and all(_valid_sha256(record.get("reference_mask_sha256")) for record in seeds)
        and seeds[selected_seed_rank].get("reference_candidate_index")
        == assignment.get("selected_reference_candidate_index")
        and seeds[selected_seed_rank].get("reference_mask_sha256")
        == assignment.get("selected_reference_mask_sha256"),
        "reference-conditioning seed audit changed",
    )


def _validate_reference_optional_fallback_contract(
    payload: Mapping[str, Any],
    fallback: Mapping[str, Any],
    *,
    require_bounded_subset_audit: bool,
) -> None:
    keys = {
        "policy_id",
        "ordered_strategies",
        "strict_consensus_vote_count",
        "reference_seed_policy",
        "reference_seed_limit",
        "common_grid_axis_count",
        "common_local_requested_voxel_size_m",
        "minimum_coarse_component_point_count",
        "local_requested_voxel_size_m",
        "stability_requested_voxel_size_m",
        "maximum_local_grid_point_count",
        "minimum_scale_stability",
        "selected_strategy",
        "attempts",
        "legacy_selected_proposals",
        "legacy_selected_mask_set_sha256",
        "common_assignment",
        "final_selected_proposals",
        "final_selected_mask_set_sha256",
        "reference_optional_safeguard",
        "artifact_sha256",
    }
    _require(
        set(fallback) == keys
        and fallback.get("artifact_sha256") == artifact_sha256(fallback)
        and fallback.get("policy_id")
        == FRAME_ZERO_REFERENCE_OPTIONAL_FALLBACK_POLICY_ID
        and fallback.get("ordered_strategies")
        == list(FRAME_ZERO_SEMANTIC_GATE_CONTRACT["application_order"])
        and fallback.get("strict_consensus_vote_count")
        == _FALLBACK_STRICT_CONSENSUS_VOTES
        and fallback.get("reference_seed_policy") == _FALLBACK_REFERENCE_SEED_POLICY
        and fallback.get("reference_seed_limit") == _FALLBACK_REFERENCE_SEED_COUNT
        and fallback.get("common_grid_axis_count") == _FALLBACK_COMMON_GRID_AXIS_COUNT
        and fallback.get("common_local_requested_voxel_size_m")
        == _FALLBACK_COMMON_LOCAL_REQUESTED_VOXEL_SIZE_M
        and fallback.get("minimum_coarse_component_point_count")
        == _FALLBACK_MINIMUM_COARSE_COMPONENT_POINT_COUNT
        and fallback.get("local_requested_voxel_size_m")
        == _FALLBACK_LOCAL_REQUESTED_VOXEL_SIZE_M
        and fallback.get("stability_requested_voxel_size_m")
        == _FALLBACK_STABILITY_REQUESTED_VOXEL_SIZE_M
        and fallback.get("maximum_local_grid_point_count")
        == _FALLBACK_MAXIMUM_LOCAL_GRID_POINT_COUNT
        and fallback.get("minimum_scale_stability") == _FALLBACK_MINIMUM_SCALE_STABILITY
        and fallback.get("selected_strategy")
        == FRAME_ZERO_REFERENCE_OPTIONAL_GEOMETRY_STRATEGY,
        "reference-optional fallback policy changed",
    )
    selected_cameras = payload["camera_policy"]["selected_cameras"]
    legacy_cameras, legacy_hash = _proposal_audit_cameras_and_hash(
        fallback.get("legacy_selected_proposals"), expected_cameras=None
    )
    final_cameras, final_hash = _proposal_audit_cameras_and_hash(
        fallback.get("final_selected_proposals"),
        expected_cameras=selected_cameras,
    )
    _require(
        legacy_cameras
        and fallback.get("legacy_selected_mask_set_sha256") == legacy_hash
        and fallback.get("final_selected_mask_set_sha256") == final_hash,
        "reference-optional outer mask-set binding changed",
    )
    attempts = fallback.get("attempts")
    _require(
        isinstance(attempts, list)
        and len(attempts) == 4
        and [record.get("strategy") for record in attempts]
        == list(FRAME_ZERO_SEMANTIC_GATE_CONTRACT["application_order"])
        and all(
            isinstance(record, Mapping)
            and set(record) == {"strategy", "status", "error_type", "reason"}
            and record.get("status") == "failed"
            and record.get("error_type") == "FrameZeroGeometryQAError"
            and isinstance(record.get("reason"), str)
            and bool(record["reason"])
            for record in attempts[:3]
        ),
        "reference-optional attempt order changed",
    )
    passed = attempts[3]
    geometry_qa = payload.get("geometry_qa")
    pass_keys = {
        "strategy",
        "status",
        "selected_camera_count",
        "selected_mask_set_sha256",
        "coarse_peak_vote_count",
        "coarse_required_vote_count",
        "coarse_connected_core_point_count",
        "refined_surface_point_count",
        "refined_required_vote_count",
        "stability_required_vote_count",
        "refined_grid_coarsened_for_cap",
        "stability_grid_coarsened_for_cap",
        "refined_effective_axis_spacing_m",
        "stability_effective_axis_spacing_m",
        "raw_median_hull_mask_containment",
        "footprint_median_hull_mask_containment",
        "median_depth_mask_coverage",
        "local_scale_stability",
        "stability_component_count",
        "stability_largest_component_fraction",
        "projected_footprint_diagnostics_sha256",
        "geometry_qa_sha256",
    }
    _require(
        isinstance(passed, Mapping)
        and set(passed) == pass_keys
        and passed.get("status") == "passed"
        and passed.get("strategy") == FRAME_ZERO_REFERENCE_OPTIONAL_GEOMETRY_STRATEGY
        and passed.get("selected_camera_count") == len(selected_cameras) == 8
        and passed.get("selected_mask_set_sha256") == final_hash
        and passed.get("geometry_qa_sha256")
        == hashlib.sha256(_canonical_bytes(geometry_qa)).hexdigest()
        and isinstance(geometry_qa, Mapping)
        and geometry_qa.get("strategy")
        == FRAME_ZERO_REFERENCE_OPTIONAL_GEOMETRY_STRATEGY
        and geometry_qa.get("fallback_policy_id")
        == FRAME_ZERO_REFERENCE_OPTIONAL_FALLBACK_POLICY_ID
        and geometry_qa.get("geometry_qa_passed") is True
        and set(geometry_qa.get("acceptance_gates", {}))
        == set(_FALLBACK_ACCEPTANCE_GATE_NAMES)
        and all(geometry_qa["acceptance_gates"].values())
        and passed.get("coarse_required_vote_count") == 8
        and passed.get("refined_required_vote_count") == 8
        and passed.get("stability_required_vote_count") == 8
        and passed.get("refined_grid_coarsened_for_cap") is False
        and passed.get("stability_grid_coarsened_for_cap") is False
        and passed.get("coarse_connected_core_point_count")
        >= _FALLBACK_MINIMUM_COARSE_COMPONENT_POINT_COUNT
        and passed.get("refined_surface_point_count")
        >= _FALLBACK_MINIMUM_REFINED_SURFACE_POINT_COUNT
        and passed.get("local_scale_stability") >= _FALLBACK_MINIMUM_SCALE_STABILITY,
        "reference-optional geometry pass audit changed",
    )
    safeguard = fallback.get("reference_optional_safeguard")
    safeguard_keys = {
        "contract_sha256",
        "assignment",
        "official_urdf",
        "semantic_selected_proposals",
        "semantic_selected_mask_set_sha256",
        "semantic_gate",
        "robot_subtraction",
        "final_selected_geometry_masks",
        "final_selected_geometry_mask_set_sha256",
        "artifact_sha256",
    }
    _require(
        isinstance(safeguard, Mapping)
        and set(safeguard) == safeguard_keys
        and safeguard.get("artifact_sha256") == artifact_sha256(safeguard)
        and safeguard.get("contract_sha256") == FRAME_ZERO_SEMANTIC_GATE_CONTRACT_SHA256
        and fallback.get("common_assignment") == safeguard.get("assignment"),
        "reference-optional safeguard checksum/cross-link changed",
    )
    semantic_proposals = safeguard["semantic_selected_proposals"]
    semantic_cameras, semantic_hash = _proposal_audit_cameras_and_hash(
        semantic_proposals, expected_cameras=selected_cameras
    )
    safeguard_final_cameras, safeguard_final_hash = _proposal_audit_cameras_and_hash(
        safeguard["final_selected_geometry_masks"],
        expected_cameras=selected_cameras,
    )
    _require(
        semantic_cameras == final_cameras == safeguard_final_cameras
        and safeguard.get("semantic_selected_mask_set_sha256") == semantic_hash
        and safeguard.get("final_selected_geometry_mask_set_sha256")
        == safeguard_final_hash
        == final_hash
        and safeguard["final_selected_geometry_masks"]
        == fallback["final_selected_proposals"],
        "reference-optional raw/final mask cross-link changed",
    )
    assignment = safeguard["assignment"]
    _validate_reference_optional_assignment_audit(
        assignment,
        payload=payload,
        semantic_proposals=semantic_proposals,
        require_bounded_subset_audit=require_bounded_subset_audit,
    )
    official = safeguard["official_urdf"]
    _require(
        isinstance(official, Mapping)
        and set(official)
        == {
            "bindings",
            "selected_action_frame_index",
            "selected_action",
            "render",
            "proposal_exclusion",
            "artifact_sha256",
        }
        and official.get("artifact_sha256") == artifact_sha256(official)
        and official.get("selected_action_frame_index") == 0
        and official.get("bindings", {}).get("commit")
        == FRAME_ZERO_SEMANTIC_GATE_CONTRACT["official_urdf"]["repository_commit"]
        and official.get("bindings", {}).get("source_and_asset_sha256")
        == FRAME_ZERO_SEMANTIC_GATE_CONTRACT["official_urdf"]["source_and_asset_sha256"]
        and _valid_sha256(official.get("selected_action", {}).get("T_worlds_sha256"))
        and _valid_sha256(official.get("selected_action", {}).get("openings_sha256")),
        "official URDF render audit changed",
    )
    selected_robot_bundle = Path(
        str(payload["action_alignment"]["selected_robot_kinematics_bundle"]["path"])
    )
    selected_robot_state = load_robot_kinematics_archive(selected_robot_bundle)
    selected_transforms = np.asarray(selected_robot_state.T_worlds[0])
    selected_openings = np.asarray(selected_robot_state.openings[0])
    if not selected_robot_state.bimanual:
        selected_transforms = selected_transforms[None]
        selected_openings = np.asarray([selected_openings])
    _require(
        official["selected_action"].get("bimanual") is selected_robot_state.bimanual
        and official["selected_action"].get("gripper_count")
        == selected_robot_state.gripper_count
        and official["selected_action"].get("T_worlds_sha256")
        == _sha256_array(selected_transforms)
        and official["selected_action"].get("openings_sha256")
        == _sha256_array(selected_openings)
        and official.get("render")
        == {
            "implementation": (
                "official deform360.processing.urdf_render.PyrenderGripperRenderer"
            ),
            "camera_pose": "invert_transform(T_worlds[0,g]) @ camera_to_world",
            "multi_gripper_union": "boolean union per camera",
            "image_shape": payload["arrays"]["rgb_frame0"]["shape"][1:3],
        },
        "official URDF selected action/render cross-link changed",
    )
    exclusion = official["proposal_exclusion"]
    _require(
        isinstance(exclusion, Mapping)
        and exclusion.get("artifact_sha256") == artifact_sha256(exclusion)
        and exclusion.get("policy_id")
        == FRAME_ZERO_SEMANTIC_GATE_CONTRACT["official_urdf"]["policy_id"]
        and exclusion.get("dilation_shape") == "square/Chebyshev"
        and exclusion.get("dilation_radius_pixels") == 5
        and exclusion.get("kernel_shape_pixels") == [11, 11]
        and exclusion.get("reject_if_overlap_fraction_greater_than_or_equal") == 0.5
        and exclusion.get("candidate_indices_preserved") is True
        and isinstance(exclusion.get("per_camera"), list)
        and [record.get("camera") for record in exclusion["per_camera"]]
        == payload["camera_policy"]["candidate_cameras"],
        "official URDF proposal-exclusion policy changed",
    )
    urdf_by_camera = {}
    for camera_record in exclusion["per_camera"]:
        candidates = camera_record.get("candidates")
        _require(
            isinstance(candidates, list)
            and camera_record.get("automatic_candidate_count") == len(candidates)
            and [record.get("candidate_index") for record in candidates]
            == list(range(len(candidates)))
            and camera_record.get("rejected_candidate_count")
            == sum(
                record.get("rejected_as_robot_dominated") is True
                for record in candidates
            )
            and _valid_sha256(camera_record.get("exact_robot_mask_sha256"))
            and _valid_sha256(camera_record.get("dilated_robot_mask_sha256")),
            "official URDF per-camera inventory changed",
        )
        for record in candidates:
            count = record.get("mask_pixel_count")
            exact = record.get("exact_robot_intersection_pixel_count")
            broad = record.get("dilated_robot_intersection_pixel_count")
            _require(
                isinstance(count, int)
                and not isinstance(count, bool)
                and count > 0
                and isinstance(exact, int)
                and 0 <= exact <= broad
                and isinstance(broad, int)
                and broad <= count
                and record.get("exact_robot_overlap_fraction") == exact / count
                and record.get("dilated_robot_overlap_fraction") == broad / count
                and record.get("rejected_as_robot_dominated") is (broad / count >= 0.5)
                and _valid_sha256(record.get("mask_sha256")),
                "official URDF proposal overlap arithmetic changed",
            )
        urdf_by_camera[str(camera_record["camera"])] = camera_record
    _require(
        assignment.get("proposal_count_by_camera")
        == {
            camera: int(urdf_by_camera[camera]["automatic_candidate_count"])
            for camera in payload["camera_policy"]["candidate_cameras"]
        },
        "URDF/assignment proposal inventory count changed",
    )
    for record in semantic_proposals:
        urdf_candidate = urdf_by_camera[str(record["camera"])]["candidates"][
            int(record["candidate_index"])
        ]
        _require(
            urdf_candidate["mask_sha256"] == record["mask_sha256"]
            and urdf_candidate["rejected_as_robot_dominated"] is False,
            "semantic assignment retained a robot-dominated or different proposal",
        )
    subtraction = safeguard["robot_subtraction"]
    _require(
        isinstance(subtraction, Mapping)
        and subtraction.get("artifact_sha256") == artifact_sha256(subtraction)
        and subtraction.get("policy_id") == exclusion.get("policy_id")
        and subtraction.get("operation")
        == "selected_mask AND NOT dilated_official_URDF"
        and isinstance(subtraction.get("per_camera"), list)
        and [record.get("camera") for record in subtraction["per_camera"]]
        == selected_cameras,
        "official URDF subtraction audit changed",
    )
    raw_by_camera = {record["camera"]: record for record in semantic_proposals}
    final_by_camera = {
        record["camera"]: record for record in fallback["final_selected_proposals"]
    }
    for record in subtraction["per_camera"]:
        camera = record["camera"]
        before = record.get("selected_pixel_count_before_subtraction")
        removed = record.get("removed_pixel_count")
        after = record.get("selected_pixel_count_after_subtraction")
        _require(
            record.get("selected_mask_sha256_before_subtraction")
            == raw_by_camera[camera]["mask_sha256"]
            and record.get("selected_mask_sha256_after_subtraction")
            == final_by_camera[camera]["mask_sha256"]
            and record.get("exact_robot_mask_sha256")
            == urdf_by_camera[camera]["exact_robot_mask_sha256"]
            and record.get("dilated_robot_mask_sha256")
            == urdf_by_camera[camera]["dilated_robot_mask_sha256"]
            and isinstance(before, int)
            and isinstance(removed, int)
            and isinstance(after, int)
            and before > 0
            and 0 <= removed < before
            and after == before - removed > 0,
            "official URDF subtraction cross-link changed",
        )
    semantic_result = validate_semantic_gate_audit(safeguard["semantic_gate"])
    _require(
        semantic_result["selected_cameras"] == selected_cameras
        and semantic_result["true_label"]
        == semantic_label_for_object_id(str(payload["object_id"]))
        and [
            {
                "camera": record["camera"],
                "candidate_index": record["candidate_index"],
                "selected_mask_sha256": record["selected_mask_sha256"],
            }
            for record in safeguard["semantic_gate"]["selected_exact8"]
        ]
        == [
            {
                "camera": record["camera"],
                "candidate_index": record["candidate_index"],
                "selected_mask_sha256": record["mask_sha256"],
            }
            for record in semantic_proposals
        ],
        "semantic gate differs from locked object/assignment",
    )


def _validate_geometry_fallback_contract(
    payload: Mapping[str, Any],
    *,
    require_bounded_subset_audit: bool,
) -> None:
    fallback = payload.get("geometry_fallback")
    if fallback is None:
        return
    _require(isinstance(fallback, Mapping), "invalid geometry fallback record")
    if fallback.get("policy_id") == FRAME_ZERO_REFERENCE_OPTIONAL_FALLBACK_POLICY_ID:
        _validate_reference_optional_fallback_contract(
            payload,
            fallback,
            require_bounded_subset_audit=require_bounded_subset_audit,
        )
        return
    keys = {
        "policy_id",
        "ordered_strategies",
        "strict_consensus_vote_count",
        "reference_seed_policy",
        "reference_seed_limit",
        "common_grid_axis_count",
        "common_local_requested_voxel_size_m",
        "minimum_coarse_component_point_count",
        "local_requested_voxel_size_m",
        "stability_requested_voxel_size_m",
        "maximum_local_grid_point_count",
        "minimum_scale_stability",
        "selected_strategy",
        "attempts",
        "legacy_selected_proposals",
        "legacy_selected_mask_set_sha256",
        "common_assignment",
        "final_selected_proposals",
        "final_selected_mask_set_sha256",
        "artifact_sha256",
    }
    _require(
        isinstance(fallback, Mapping) and set(fallback) == keys,
        "invalid frame-zero geometry fallback record",
    )
    _require(
        fallback.get("artifact_sha256") == artifact_sha256(fallback),
        "frame-zero geometry fallback checksum mismatch",
    )
    ordered = [
        "legacy",
        "same-masks-projected-footprint",
        "common-voxel-assignment-projected-footprint",
    ]
    _require(
        fallback.get("policy_id") == FRAME_ZERO_GEOMETRY_FALLBACK_POLICY_ID
        and fallback.get("ordered_strategies") == ordered
        and fallback.get("strict_consensus_vote_count")
        == _FALLBACK_STRICT_CONSENSUS_VOTES
        and fallback.get("reference_seed_policy") == _FALLBACK_REFERENCE_SEED_POLICY
        and fallback.get("reference_seed_limit") == _FALLBACK_REFERENCE_SEED_COUNT
        and fallback.get("common_grid_axis_count") == _FALLBACK_COMMON_GRID_AXIS_COUNT
        and fallback.get("common_local_requested_voxel_size_m")
        == _FALLBACK_COMMON_LOCAL_REQUESTED_VOXEL_SIZE_M
        and fallback.get("minimum_coarse_component_point_count")
        == _FALLBACK_MINIMUM_COARSE_COMPONENT_POINT_COUNT
        and fallback.get("local_requested_voxel_size_m")
        == _FALLBACK_LOCAL_REQUESTED_VOXEL_SIZE_M
        and fallback.get("stability_requested_voxel_size_m")
        == _FALLBACK_STABILITY_REQUESTED_VOXEL_SIZE_M
        and fallback.get("maximum_local_grid_point_count")
        == _FALLBACK_MAXIMUM_LOCAL_GRID_POINT_COUNT
        and fallback.get("minimum_scale_stability")
        == _FALLBACK_MINIMUM_SCALE_STABILITY,
        "frame-zero geometry fallback policy changed",
    )
    selected_strategy = fallback.get("selected_strategy")
    _require(
        selected_strategy in set(ordered[1:]),
        "invalid selected geometry fallback strategy",
    )
    attempts = fallback.get("attempts")
    expected_attempt_strategies = (
        ordered[:2]
        if selected_strategy == "same-masks-projected-footprint"
        else ordered
    )
    _require(
        isinstance(attempts, list)
        and all(isinstance(attempt, Mapping) for attempt in attempts)
        and [attempt.get("strategy") for attempt in attempts]
        == expected_attempt_strategies,
        "geometry fallback attempt order changed",
    )
    failure_keys = {"strategy", "status", "error_type", "reason"}
    pass_keys = {
        "strategy",
        "status",
        "selected_camera_count",
        "selected_mask_set_sha256",
        "coarse_peak_vote_count",
        "coarse_required_vote_count",
        "coarse_connected_core_point_count",
        "refined_surface_point_count",
        "refined_required_vote_count",
        "stability_required_vote_count",
        "refined_grid_coarsened_for_cap",
        "stability_grid_coarsened_for_cap",
        "refined_effective_axis_spacing_m",
        "stability_effective_axis_spacing_m",
        "raw_median_hull_mask_containment",
        "footprint_median_hull_mask_containment",
        "median_depth_mask_coverage",
        "local_scale_stability",
        "stability_component_count",
        "stability_largest_component_fraction",
        "projected_footprint_diagnostics_sha256",
        "geometry_qa_sha256",
    }
    for attempt in attempts[:-1]:
        _require(
            isinstance(attempt, Mapping)
            and set(attempt) == failure_keys
            and attempt.get("status") == "failed"
            and attempt.get("error_type") == "FrameZeroGeometryQAError"
            and isinstance(attempt.get("reason"), str)
            and bool(attempt["reason"]),
            "invalid failed geometry fallback attempt",
        )
    passed = attempts[-1]

    def integer_at_least(record: Mapping[str, Any], key: str, minimum: int) -> bool:
        value = record.get(key)
        return (
            isinstance(value, int) and not isinstance(value, bool) and value >= minimum
        )

    def number_at_least(record: Mapping[str, Any], key: str, minimum: float) -> bool:
        value = record.get(key)
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= minimum
        )

    _require(
        isinstance(passed, Mapping)
        and set(passed) == pass_keys
        and passed.get("status") == "passed"
        and passed.get("strategy") == selected_strategy
        and passed.get("selected_camera_count")
        == payload["camera_policy"]["selected_camera_count"]
        and passed.get("selected_mask_set_sha256")
        == fallback.get("final_selected_mask_set_sha256")
        and all(
            _valid_sha256(passed.get(key))
            for key in (
                "selected_mask_set_sha256",
                "projected_footprint_diagnostics_sha256",
                "geometry_qa_sha256",
            )
        )
        and integer_at_least(
            passed,
            "coarse_peak_vote_count",
            _FALLBACK_STRICT_CONSENSUS_VOTES,
        )
        and passed.get("coarse_required_vote_count") == _FALLBACK_STRICT_CONSENSUS_VOTES
        and passed.get("refined_required_vote_count")
        == _FALLBACK_STRICT_CONSENSUS_VOTES
        and passed.get("stability_required_vote_count")
        == _FALLBACK_STRICT_CONSENSUS_VOTES
        and passed.get("refined_grid_coarsened_for_cap") is False
        and passed.get("stability_grid_coarsened_for_cap") is False
        and all(
            isinstance(spacing, list)
            and len(spacing) == 3
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) > 0.0
                for value in spacing
            )
            for spacing in (
                passed.get("refined_effective_axis_spacing_m"),
                passed.get("stability_effective_axis_spacing_m"),
            )
        )
        and integer_at_least(
            passed,
            "coarse_connected_core_point_count",
            _FALLBACK_MINIMUM_COARSE_COMPONENT_POINT_COUNT,
        )
        and integer_at_least(
            passed,
            "refined_surface_point_count",
            _FALLBACK_MINIMUM_REFINED_SURFACE_POINT_COUNT,
        )
        and number_at_least(
            passed,
            "raw_median_hull_mask_containment",
            float(payload["config"]["minimum_median_hull_mask_containment"]),
        )
        and number_at_least(
            passed,
            "median_depth_mask_coverage",
            float(payload["config"]["minimum_median_depth_mask_coverage"]),
        )
        and number_at_least(
            passed,
            "local_scale_stability",
            _FALLBACK_MINIMUM_SCALE_STABILITY,
        )
        and number_at_least(
            passed,
            "stability_largest_component_fraction",
            float(payload["config"]["minimum_largest_component_fraction"]),
        ),
        "invalid passed geometry fallback attempt",
    )

    def validate_proposals(
        value: object, *, expected_cameras: Sequence[str] | None
    ) -> list[str]:
        _require(isinstance(value, list) and bool(value), "proposal audit is missing")
        cameras = []
        proposal_keys = {
            "camera",
            "candidate_index",
            "automatic_candidate_count",
            "eligible_candidate_count",
            "mask_sha256",
        }
        for record in value:
            _require(
                isinstance(record, Mapping)
                and set(record) == proposal_keys
                and isinstance(record.get("camera"), str)
                and isinstance(record.get("candidate_index"), int)
                and not isinstance(record.get("candidate_index"), bool)
                and int(record["candidate_index"]) >= 0
                and isinstance(record.get("automatic_candidate_count"), int)
                and not isinstance(record.get("automatic_candidate_count"), bool)
                and isinstance(record.get("eligible_candidate_count"), int)
                and not isinstance(record.get("eligible_candidate_count"), bool)
                and 1
                <= int(record["eligible_candidate_count"])
                <= int(record["automatic_candidate_count"])
                and _valid_sha256(record.get("mask_sha256")),
                "invalid selected proposal audit",
            )
            cameras.append(str(record["camera"]))
        _require(cameras == sorted(set(cameras)), "proposal camera order changed")
        if expected_cameras is not None:
            _require(cameras == list(expected_cameras), "proposal cameras changed")
        return cameras

    legacy_proposals = fallback.get("legacy_selected_proposals")
    final_proposals = fallback.get("final_selected_proposals")
    validate_proposals(legacy_proposals, expected_cameras=None)
    final_cameras = payload["camera_policy"]["selected_cameras"]
    validate_proposals(final_proposals, expected_cameras=final_cameras)

    def proposal_mask_set_sha256(value: Sequence[Mapping[str, Any]]) -> str:
        return hashlib.sha256(
            _canonical_bytes(
                {str(record["camera"]): record["mask_sha256"] for record in value}
            )
        ).hexdigest()

    _require(
        fallback.get("legacy_selected_mask_set_sha256")
        == proposal_mask_set_sha256(legacy_proposals)
        and fallback.get("final_selected_mask_set_sha256")
        == proposal_mask_set_sha256(final_proposals),
        "invalid geometry fallback mask checksum",
    )
    geometry_qa = payload.get("geometry_qa")
    acceptance_gates = (
        geometry_qa.get("acceptance_gates")
        if isinstance(geometry_qa, Mapping)
        else None
    )
    _require(
        isinstance(geometry_qa, Mapping)
        and geometry_qa.get("geometry_qa_passed") is True
        and isinstance(acceptance_gates, Mapping)
        and set(acceptance_gates) == set(_FALLBACK_ACCEPTANCE_GATE_NAMES)
        and all(value is True for value in acceptance_gates.values())
        and geometry_qa.get("fallback_policy_id")
        == FRAME_ZERO_GEOMETRY_FALLBACK_POLICY_ID
        and geometry_qa.get("strategy") == selected_strategy
        and passed.get("geometry_qa_sha256")
        == hashlib.sha256(_canonical_bytes(geometry_qa)).hexdigest(),
        "geometry fallback differs from geometry QA",
    )
    common = fallback.get("common_assignment")
    if selected_strategy == "same-masks-projected-footprint":
        _require(
            common is None
            and legacy_proposals == final_proposals
            and fallback.get("legacy_selected_mask_set_sha256")
            == fallback.get("final_selected_mask_set_sha256"),
            "same-mask fallback unexpectedly changed proposals",
        )
        return
    common_grid = common.get("grid") if isinstance(common, Mapping) else None
    common_local = (
        common.get("local_refinement") if isinstance(common, Mapping) else None
    )
    common_proposals = (
        common.get("selected_proposals") if isinstance(common, Mapping) else None
    )
    common_proposal_bindings = (
        [
            {
                "camera": record.get("camera"),
                "candidate_index": record.get("candidate_index"),
                "mask_sha256": record.get("mask_sha256"),
            }
            for record in common_proposals
        ]
        if isinstance(common_proposals, list)
        and all(isinstance(record, Mapping) for record in common_proposals)
        else None
    )
    final_proposal_bindings = [
        {
            "camera": record["camera"],
            "candidate_index": record["candidate_index"],
            "mask_sha256": record["mask_sha256"],
        }
        for record in final_proposals
    ]
    camera_inliers = (
        common.get("camera_inlier_ranking") if isinstance(common, Mapping) else None
    )
    candidate_cameras = payload["camera_policy"]["candidate_cameras"]
    reference_camera = payload["camera_policy"]["reference_camera"]
    inlier_keys = {
        "camera",
        "candidate_index",
        "mask_sha256",
        "seed_conditioned_candidate_count",
        "fixed_reference",
        "nonreference_coverage_rank",
        "retained",
        "raw_component_hit_count",
        "raw_component_coverage",
        "raw_component_precision",
        "hits_local_union_peak",
        "semantic_score",
        "candidate_lexicographic_objective",
    }
    valid_inliers = isinstance(camera_inliers, list) and len(camera_inliers) == len(
        candidate_cameras
    )
    if valid_inliers:
        valid_inliers = all(
            isinstance(record, Mapping)
            and set(record) == inlier_keys
            and record.get("camera") == camera
            and isinstance(record.get("seed_conditioned_candidate_count"), int)
            and not isinstance(record.get("seed_conditioned_candidate_count"), bool)
            and int(record["seed_conditioned_candidate_count"]) >= 0
            and isinstance(record.get("retained"), bool)
            and isinstance(record.get("fixed_reference"), bool)
            and record.get("fixed_reference") == (camera == reference_camera)
            and isinstance(record.get("raw_component_hit_count"), int)
            and not isinstance(record.get("raw_component_hit_count"), bool)
            and int(record["raw_component_hit_count"]) >= 0
            and all(
                isinstance(record.get(key), (int, float))
                and not isinstance(record.get(key), bool)
                and math.isfinite(float(record[key]))
                and 0.0 <= float(record[key]) <= 1.0
                for key in ("raw_component_coverage", "raw_component_precision")
            )
            and isinstance(record.get("hits_local_union_peak"), bool)
            for camera, record in zip(candidate_cameras, camera_inliers, strict=True)
        )
    retained_inliers = (
        [record for record in camera_inliers if record.get("retained") is True]
        if valid_inliers
        else []
    )
    chosen_nonreference = (
        [
            record
            for record in camera_inliers
            if record.get("candidate_index") is not None
            and record.get("camera") != reference_camera
        ]
        if valid_inliers
        else []
    )
    expected_coverage_order = sorted(
        chosen_nonreference,
        key=lambda record: (-float(record["raw_component_coverage"]), record["camera"]),
    )
    valid_inlier_details = valid_inliers and all(
        (
            (
                isinstance(record.get("candidate_index"), int)
                and not isinstance(record.get("candidate_index"), bool)
                and int(record["candidate_index"]) >= 0
                and _valid_sha256(record.get("mask_sha256"))
                and isinstance(record.get("semantic_score"), (int, float))
                and not isinstance(record.get("semantic_score"), bool)
                and math.isfinite(float(record["semantic_score"]))
                and isinstance(record.get("candidate_lexicographic_objective"), list)
                and len(record["candidate_lexicographic_objective"]) == 5
                and record["candidate_lexicographic_objective"]
                == [
                    float(record["raw_component_coverage"]),
                    float(record["raw_component_precision"]),
                    int(bool(record["hits_local_union_peak"])),
                    float(record["semantic_score"]),
                    -int(record["candidate_index"]),
                ]
            )
            if record.get("seed_conditioned_candidate_count", 0) > 0
            else (
                record.get("candidate_index") is None
                and record.get("mask_sha256") is None
                and record.get("semantic_score") is None
                and record.get("candidate_lexicographic_objective") is None
                and record.get("retained") is False
                and record.get("nonreference_coverage_rank") is None
            )
        )
        for record in camera_inliers
    )
    valid_inlier_details = valid_inlier_details and all(
        record.get("nonreference_coverage_rank") == rank
        for rank, record in enumerate(expected_coverage_order, start=1)
    )
    reference_inlier = (
        next(
            (
                record
                for record in camera_inliers
                if record.get("camera") == reference_camera
            ),
            None,
        )
        if valid_inliers
        else None
    )
    seed_evaluations = (
        common.get("reference_seed_evaluations")
        if isinstance(common, Mapping)
        else None
    )
    selected_seed_rank = common.get("selected_reference_seed_rank")
    seed_keys = {
        "reference_seed_rank",
        "reference_candidate_index",
        "reference_mask_sha256",
        "reference_local_score",
        "eligible_camera_count",
        "strict_feasible_voxel_count",
        "strict_feasible_component_count",
        "largest_strict_component_voxel_count",
        "maximum_union_support_count",
        "reference_anchor_hit_count",
        "local_grid_shape",
        "local_grid_point_count",
        "local_grid_coarsened_for_cap",
        "local_strict_feasible_voxel_count",
        "local_strict_feasible_component_count",
        "local_largest_strict_component_voxel_count",
        "local_maximum_union_support_count",
        "local_strict_feasible_mask_sha256",
        "lexicographic_objective",
        "strict_feasible_mask_sha256",
    }
    seed_objective_order = [
        "local_largest_component_voxel_count_desc",
        "local_strict_feasible_voxel_count_desc",
        "local_peak_support_desc",
        "global_largest_component_voxel_count_desc",
        "global_strict_feasible_voxel_count_desc",
        "reference_semantic_score_desc",
        "reference_candidate_index_asc",
    ]
    reference_diagnostic = next(
        (
            record
            for record in payload["sam2"]["view_diagnostics"]
            if record.get("camera") == reference_camera
        ),
        None,
    )
    expected_seed_count = (
        min(
            int(reference_diagnostic["eligible_candidate_count"]),
            _FALLBACK_REFERENCE_SEED_COUNT,
        )
        if isinstance(reference_diagnostic, Mapping)
        and isinstance(reference_diagnostic.get("eligible_candidate_count"), int)
        and not isinstance(reference_diagnostic.get("eligible_candidate_count"), bool)
        else -1
    )
    reference_ranking = (
        common.get("reference_candidate_ranking")
        if isinstance(common, Mapping)
        else None
    )
    reference_ranking_keys = {
        "rank",
        "candidate_index",
        "mask_sha256",
        "local_mask_score",
        "selected_for_seed_evaluation",
    }
    valid_reference_ranking = (
        isinstance(reference_ranking, list)
        and isinstance(reference_diagnostic, Mapping)
        and len(reference_ranking)
        == int(reference_diagnostic["eligible_candidate_count"])
        and all(
            isinstance(record, Mapping)
            and set(record) == reference_ranking_keys
            and record.get("rank") == rank
            and isinstance(record.get("candidate_index"), int)
            and not isinstance(record.get("candidate_index"), bool)
            and int(record["candidate_index"]) >= 0
            and _valid_sha256(record.get("mask_sha256"))
            and isinstance(record.get("local_mask_score"), (int, float))
            and not isinstance(record.get("local_mask_score"), bool)
            and math.isfinite(float(record["local_mask_score"]))
            and record.get("selected_for_seed_evaluation")
            is (rank < _FALLBACK_REFERENCE_SEED_COUNT)
            for rank, record in enumerate(reference_ranking)
        )
        and [
            int(record["candidate_index"])
            for record in sorted(
                reference_ranking,
                key=lambda record: (
                    -float(record["local_mask_score"]),
                    int(record["candidate_index"]),
                ),
            )
        ]
        == [int(record["candidate_index"]) for record in reference_ranking]
    )

    def valid_seed_record(record: object) -> bool:
        if not isinstance(record, Mapping) or set(record) != seed_keys:
            return False
        integer_keys = (
            "reference_seed_rank",
            "reference_candidate_index",
            "eligible_camera_count",
            "strict_feasible_voxel_count",
            "strict_feasible_component_count",
            "largest_strict_component_voxel_count",
            "maximum_union_support_count",
            "reference_anchor_hit_count",
            "local_grid_point_count",
            "local_strict_feasible_voxel_count",
            "local_strict_feasible_component_count",
            "local_largest_strict_component_voxel_count",
            "local_maximum_union_support_count",
        )
        if not all(
            isinstance(record.get(key), int)
            and not isinstance(record.get(key), bool)
            and int(record[key]) >= 0
            for key in integer_keys
        ):
            return False
        score = record.get("reference_local_score")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or not _valid_sha256(record.get("reference_mask_sha256"))
            or not _valid_sha256(record.get("strict_feasible_mask_sha256"))
            or not isinstance(record.get("local_grid_coarsened_for_cap"), bool)
        ):
            return False
        local_shape = record.get("local_grid_shape")
        local_hash = record.get("local_strict_feasible_mask_sha256")
        local_zero_metrics = all(
            record[key] == 0
            for key in (
                "local_strict_feasible_voxel_count",
                "local_strict_feasible_component_count",
                "local_largest_strict_component_voxel_count",
                "local_maximum_union_support_count",
            )
        )
        if record["local_grid_point_count"] == 0:
            if (
                local_shape is not None
                or local_hash is not None
                or record["local_grid_coarsened_for_cap"] is True
                or not local_zero_metrics
            ):
                return False
        elif (
            not isinstance(local_shape, list)
            or len(local_shape) != 3
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1
                for value in local_shape
            )
            or math.prod(local_shape) != record["local_grid_point_count"]
            or (
                record["local_grid_coarsened_for_cap"] is False
                and not _valid_sha256(local_hash)
            )
            or (
                record["local_grid_coarsened_for_cap"] is True
                and (local_hash is not None or not local_zero_metrics)
            )
        ):
            return False
        expected_objective = [
            record["local_largest_strict_component_voxel_count"],
            record["local_strict_feasible_voxel_count"],
            record["local_maximum_union_support_count"],
            record["largest_strict_component_voxel_count"],
            record["strict_feasible_voxel_count"],
            float(score),
            -int(record["reference_candidate_index"]),
        ]
        return record.get("lexicographic_objective") == expected_objective

    valid_seed_audit = (
        isinstance(seed_evaluations, list)
        and valid_reference_ranking
        and len(seed_evaluations) == expected_seed_count
        and common.get("evaluated_reference_seed_count") == len(seed_evaluations)
        and common.get("reference_seed_objective_order") == seed_objective_order
        and all(valid_seed_record(record) for record in seed_evaluations)
        and [record.get("reference_seed_rank") for record in seed_evaluations]
        == list(range(len(seed_evaluations)))
        and [
            (
                record["reference_candidate_index"],
                record["reference_mask_sha256"],
                float(record["reference_local_score"]),
            )
            for record in seed_evaluations
        ]
        == [
            (
                record["candidate_index"],
                record["mask_sha256"],
                float(record["local_mask_score"]),
            )
            for record in reference_ranking[:expected_seed_count]
        ]
        and len(
            {record.get("reference_candidate_index") for record in seed_evaluations}
        )
        == len(seed_evaluations)
        and [
            int(record["reference_candidate_index"])
            for record in sorted(
                seed_evaluations,
                key=lambda record: (
                    -float(record["reference_local_score"]),
                    int(record["reference_candidate_index"]),
                ),
            )
        ]
        == [int(record["reference_candidate_index"]) for record in seed_evaluations]
        and isinstance(selected_seed_rank, int)
        and not isinstance(selected_seed_rank, bool)
        and 0 <= selected_seed_rank < len(seed_evaluations)
        and selected_seed_rank
        == max(
            range(len(seed_evaluations)),
            key=lambda index: tuple(seed_evaluations[index]["lexicographic_objective"]),
        )
    )
    subset_evaluations = (
        common.get("exact_eight_subset_evaluations")
        if isinstance(common, Mapping)
        else None
    )
    chosen_nonreference_cameras = [
        str(record["camera"]) for record in chosen_nonreference
    ]
    expected_subset_count = (
        math.comb(len(chosen_nonreference), _FALLBACK_STRICT_CONSENSUS_VOTES - 1)
        if len(chosen_nonreference) >= _FALLBACK_STRICT_CONSENSUS_VOTES - 1
        else 0
    )
    subset_objective_order = [
        "largest_exact_component_voxel_count_desc",
        "exact_common_voxel_count_desc",
        "raw_component_coverage_sum_desc",
        "semantic_score_sum_desc",
        "camera_tuple_asc",
    ]
    if require_bounded_subset_audit:
        _require(
            expected_subset_count > 0,
            "fixed-reference exhaustive subset audit is empty",
        )
        bounded_subsets = _validate_bounded_exact_eight_subset_audit(
            subset_evaluations,
            expected_record_count=expected_subset_count,
            expected_first_cameras=[
                reference_camera,
                *chosen_nonreference_cameras[: _FALLBACK_STRICT_CONSENSUS_VOTES - 1],
            ],
            expected_last_cameras=[
                reference_camera,
                *chosen_nonreference_cameras[-(_FALLBACK_STRICT_CONSENSUS_VOTES - 1) :],
            ],
            selected_cameras=final_cameras,
            candidate_cameras=candidate_cameras,
            fixed_first_camera=reference_camera,
        )
        valid_subsets = True
        observed_subset_count = int(bounded_subsets["record_count"])
        selected_subset = bounded_subsets["selected_record"]
    else:
        expected_subset_cameras = [
            [reference_camera, *combination]
            for combination in itertools.combinations(
                chosen_nonreference_cameras,
                _FALLBACK_STRICT_CONSENSUS_VOTES - 1,
            )
        ]
        valid_subsets = bool(
            isinstance(subset_evaluations, list)
            and all(
                _valid_exact_eight_subset_record(
                    record,
                    candidate_cameras=candidate_cameras,
                    fixed_first_camera=reference_camera,
                )
                for record in subset_evaluations
            )
            and [record["cameras"] for record in subset_evaluations]
            == expected_subset_cameras
            and len({tuple(record["cameras"]) for record in subset_evaluations})
            == len(subset_evaluations)
        )
        observed_subset_count = (
            len(subset_evaluations) if isinstance(subset_evaluations, list) else -1
        )
        selected_subset = (
            min(
                subset_evaluations,
                key=lambda record: (
                    -int(record["largest_exact_component_voxel_count"]),
                    -int(record["exact_common_voxel_count"]),
                    -float(record["raw_component_coverage_sum"]),
                    -float(record["semantic_score_sum"]),
                    tuple(record["cameras"]),
                ),
            )
            if valid_subsets and subset_evaluations
            else None
        )
    valid_subsets = (
        valid_subsets
        and common.get("exact_eight_subset_objective_order") == subset_objective_order
    )
    diagnostic_inliers = [
        record.get("geometry_inlier_selection")
        for record in payload["sam2"]["view_diagnostics"]
    ]
    common_keys = {
        "policy_id",
        "strategy",
        "reference_seed_policy",
        "reference_seed_limit",
        "inlier_selection_rule",
        "search_hit_representation",
        "evaluated_reference_seed_count",
        "strict_consensus_vote_count",
        "grid",
        "proposal_count_by_camera",
        "proposal_inventory_sha256",
        "reference_candidate_ranking",
        "reference_seed_evaluations",
        "reference_seed_objective_order",
        "selected_reference_seed_rank",
        "selected_reference_candidate_index",
        "selected_reference_mask_sha256",
        "global_selected_common_voxel_flat_index",
        "global_selected_common_voxel_world_m",
        "global_selected_common_voxel_support_count",
        "global_selected_strict_component_voxel_count",
        "global_selected_strict_component_sha256",
        "local_refinement",
        "local_union_peak_voxel_flat_index",
        "local_union_peak_voxel_world_m",
        "local_union_peak_support_count",
        "local_union_strict_component_voxel_count",
        "local_union_strict_component_sha256",
        "candidate_objective_order",
        "camera_inlier_ranking",
        "evaluated_exact_eight_subset_count",
        "exact_eight_subset_objective_order",
        "exact_eight_subset_evaluations",
        "selected_exact_eight_cameras",
        "selected_common_voxel_flat_index",
        "selected_common_voxel_world_m",
        "selected_common_voxel_support_count",
        "selected_exact_common_voxel_count",
        "selected_exact_common_mask_sha256",
        "selected_strict_component_voxel_count",
        "selected_strict_component_sha256",
        "selected_proposals",
        "selected_mask_set_sha256",
        "artifact_sha256",
    }
    _require(
        isinstance(common, Mapping)
        and set(common) == common_keys
        and common.get("artifact_sha256") == artifact_sha256(common)
        and common.get("policy_id") == FRAME_ZERO_GEOMETRY_FALLBACK_POLICY_ID
        and common.get("strategy")
        == "reference-anchored-exhaustive-exact-eight-assignment"
        and common.get("reference_seed_policy") == _FALLBACK_REFERENCE_SEED_POLICY
        and common.get("reference_seed_limit") == _FALLBACK_REFERENCE_SEED_COUNT
        and common.get("inlier_selection_rule") == _COMMON_INLIER_SELECTION_RULE
        and common.get("search_hit_representation") == "nearest-pixel-raw-center"
        and common.get("strict_consensus_vote_count")
        == _FALLBACK_STRICT_CONSENSUS_VOTES
        and isinstance(common_grid, Mapping)
        and common_grid.get("grid_shape") == [_FALLBACK_COMMON_GRID_AXIS_COUNT] * 3
        and common_grid.get("grid_point_count") == _FALLBACK_COMMON_GRID_AXIS_COUNT**3
        and isinstance(common_local, Mapping)
        and common_local.get("requested_voxel_size_m")
        == _FALLBACK_COMMON_LOCAL_REQUESTED_VOXEL_SIZE_M
        and isinstance(common_local.get("grid"), Mapping)
        and common_local["grid"].get("requested_voxel_size_m")
        == _FALLBACK_COMMON_LOCAL_REQUESTED_VOXEL_SIZE_M
        and common_local["grid"].get("coarsened_for_grid_cap") is False
        and common.get("selected_common_voxel_support_count")
        == _FALLBACK_STRICT_CONSENSUS_VOTES
        and integer_at_least(common, "selected_exact_common_voxel_count", 1)
        and integer_at_least(common, "selected_strict_component_voxel_count", 1)
        and _valid_sha256(common.get("selected_exact_common_mask_sha256"))
        and _valid_sha256(common.get("selected_strict_component_sha256"))
        and common.get("selected_mask_set_sha256")
        == fallback.get("final_selected_mask_set_sha256")
        and common_proposal_bindings == final_proposal_bindings
        and common_proposals == retained_inliers
        and isinstance(common_proposals, list)
        and all(isinstance(record, Mapping) for record in common_proposals)
        and [
            record.get("camera")
            for record in common_proposals
            if isinstance(record, Mapping)
        ]
        == final_cameras
        and len(final_cameras) == _FALLBACK_STRICT_CONSENSUS_VOTES
        and reference_camera in final_cameras
        and valid_inlier_details
        and [record["camera"] for record in retained_inliers] == final_cameras
        and reference_inlier is not None
        and reference_inlier.get("retained") is True
        and reference_inlier.get("nonreference_coverage_rank") == 0
        and reference_inlier.get("candidate_index")
        == common.get("selected_reference_candidate_index")
        and reference_inlier.get("mask_sha256")
        == common.get("selected_reference_mask_sha256")
        and diagnostic_inliers == camera_inliers
        and valid_seed_audit
        and seed_evaluations[selected_seed_rank].get("reference_candidate_index")
        == common.get("selected_reference_candidate_index")
        and seed_evaluations[selected_seed_rank].get("reference_mask_sha256")
        == common.get("selected_reference_mask_sha256")
        and seed_evaluations[selected_seed_rank].get("local_grid_coarsened_for_cap")
        is False
        and valid_subsets
        and observed_subset_count == expected_subset_count
        and common.get("evaluated_exact_eight_subset_count") == expected_subset_count
        and selected_subset is not None
        and sorted(selected_subset["cameras"])
        == common.get("selected_exact_eight_cameras")
        == final_cameras
        and common.get("selected_exact_common_voxel_count")
        == selected_subset["exact_common_voxel_count"]
        and common.get("selected_exact_common_mask_sha256")
        == selected_subset["exact_common_mask_sha256"]
        and common.get("selected_strict_component_voxel_count")
        == selected_subset["largest_exact_component_voxel_count"],
        "invalid common-voxel assignment audit",
    )


def _validate_local_bound_file_record(
    record: object,
    *,
    label: str,
) -> Path:
    """Validate a manifest file binding against the materialized local file."""

    _require(
        isinstance(record, Mapping) and set(record) == {"path", "sha256", "size_bytes"},
        f"invalid {label} file record",
    )
    resolved = validate_regular_file_nofollow(str(record.get("path")), label=label)
    _require(
        record.get("sha256") == _sha256_file(resolved)
        and record.get("size_bytes") == resolved.stat().st_size,
        f"{label} binding changed",
    )
    return resolved


def validate_frame_zero_bundle_manifest(
    payload: Mapping[str, Any],
    *,
    require_bounded_subset_audit: bool = False,
) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == 1, "unsupported frame-zero manifest schema"
    )
    _require(
        payload.get("artifact_kind") == FRAME_ZERO_BUNDLE_ARTIFACT_KIND,
        "unexpected frame-zero manifest kind",
    )
    _require(
        payload.get("protocol_id") == HELD_PROTOCOL_ID, "frame-zero protocol changed"
    )
    object_id, episode_id = _case_parts(str(payload.get("case_name")))
    _require(payload.get("object_id") == object_id, "manifest object/case mismatch")
    _require(payload.get("episode_id") == episode_id, "manifest episode/case mismatch")
    _require(
        _valid_sha256(payload.get("lock_sha256")), "invalid held lock file checksum"
    )
    _require(
        _valid_sha256(payload.get("lock_artifact_sha256")),
        "invalid held lock artifact checksum",
    )
    _require(payload.get("frame_indices") == [0], "frame-zero bundle is multi-frame")
    _require(isinstance(payload.get("config"), Mapping), "frame-zero config is missing")
    _require(
        payload.get("information_boundary") == FRAME_ZERO_INFORMATION_BOUNDARY,
        "frame-zero information boundary changed",
    )
    bundle = payload.get("bundle")
    _require(isinstance(bundle, Mapping), "frame-zero manifest lacks bundle")
    _require(Path(str(bundle.get("path"))).is_absolute(), "bundle path is not absolute")
    _require(_valid_sha256(bundle.get("sha256")), "invalid bundle checksum")
    _require(isinstance(bundle.get("size_bytes"), int), "invalid bundle size")
    action_inputs = payload.get("action_inputs")
    _require(
        isinstance(action_inputs, Mapping)
        and set(action_inputs) == {"robot_trajectory", "robot_metadata"},
        "frame-zero manifest lacks exact action inputs",
    )
    for record in action_inputs.values():
        _require(isinstance(record, Mapping), "invalid robot input record")
    raw_robot_record = action_inputs["robot_trajectory"]
    raw_robot_record_path = Path(str(raw_robot_record.get("path")))
    dataset_layout = layout_from_dataset_file(
        raw_robot_record_path,
        object_id=object_id,
        episode_id=episode_id,
        relative_parts=("robot", "robot.npz"),
        label="source realized robot kinematics",
    )
    raw_robot_path = _validate_local_bound_file_record(
        raw_robot_record,
        label="source realized robot kinematics",
    )
    robot_metadata_path = _validate_local_bound_file_record(
        action_inputs["robot_metadata"],
        label="source realized robot kinematics metadata",
    )
    dataset_layout.validate_file(
        robot_metadata_path,
        "robot",
        "robot.meta.json",
        label="source realized robot kinematics metadata",
    )
    calibration_inputs = payload.get("calibration_inputs")
    _require(
        isinstance(calibration_inputs, Mapping)
        and set(calibration_inputs) == {"intrinsics", "extrinsics"},
        "frame-zero manifest lacks exact calibration inputs",
    )
    for key, filename in (
        ("intrinsics", "undistorted_intrinsics.npy"),
        ("extrinsics", "extrinsics.npy"),
    ):
        calibration_path = _validate_local_bound_file_record(
            calibration_inputs[key], label=f"source {key}"
        )
        dataset_layout.validate_file(calibration_path, filename, label=f"source {key}")
    action_alignment = payload.get("action_alignment")
    _require(
        isinstance(action_alignment, Mapping),
        "frame-zero manifest lacks robot kinematics alignment",
    )
    config = payload["config"]
    selection_audit = action_alignment.get("selection_audit")
    _require(
        isinstance(selection_audit, Mapping),
        "frame-zero manifest lacks the shared robot selection audit",
    )
    source_state = load_robot_kinematics_archive(raw_robot_path)
    validated_selection = validate_robot_kinematics_selection_audit(
        selection_audit,
        source_state,
        window_length_frames=int(config["action_window_length_frames"]),
        prediction_frame_count=int(config["prediction_frame_count"]),
        candidate_first_frame=int(config["action_candidate_first_frame"]),
        candidate_stride_frames=int(config["action_candidate_stride_frames"]),
    )
    raw_range = action_alignment.get("selected_raw_frame_range_half_open")
    prediction_range = action_alignment.get("prediction_raw_frame_range_half_open")
    _require(
        raw_range == validated_selection["selected_raw_frame_range_half_open"]
        and int(raw_range[1]) - int(raw_range[0])
        == int(config["action_window_length_frames"]),
        "action selection is not an 81-frame window",
    )
    _require(
        prediction_range == validated_selection["prediction_raw_frame_range_half_open"]
        and int(prediction_range[0]) == int(raw_range[0])
        and int(prediction_range[1]) - int(prediction_range[0])
        == int(config["prediction_frame_count"]),
        "known robot kinematics bundle is not the 76-frame prediction window",
    )
    _require(
        action_alignment.get("policy_id") == ROBOT_KINEMATICS_WINDOW_POLICY_ID
        and action_alignment.get("trajectory_semantics")
        == ROBOT_KINEMATICS_WINDOW_CONTRACT["trajectory_semantics"]
        and action_alignment.get("tracking_tail_frame_count")
        == validated_selection["tracking_tail_frame_count"]
        and action_alignment.get("source_robot_frame_count") == source_state.frame_count
        and action_alignment.get("prediction_frame_count")
        == int(config["prediction_frame_count"]),
        "robot kinematics alignment compatibility fields changed",
    )
    selected_action = action_alignment.get("selected_action_bundle")
    _require(
        selected_action == action_alignment.get("selected_robot_kinematics_bundle")
        and action_alignment.get("selected_action_bundle_is_compatibility_alias")
        is True,
        "selected action compatibility alias changed",
    )
    selected_robot_path = _validate_local_bound_file_record(
        selected_action,
        label="selected realized robot kinematics",
    )
    selected_state = load_robot_kinematics_archive(
        selected_robot_path,
        expected_frame_count=int(config["prediction_frame_count"]),
    )
    _require(
        action_alignment.get("selected_action_arrays")
        == _bundle_array_records(selected_state.archive_arrays()),
        "selected robot kinematics array bindings changed",
    )
    exact_slice_audit = validate_selected_robot_kinematics_bundle(
        selected_state,
        source_state=source_state,
        prediction_start_frame=int(prediction_range[0]),
        prediction_frame_count=int(config["prediction_frame_count"]),
    )
    _require(
        action_alignment.get("selected_bundle_exact_slice_audit") == exact_slice_audit,
        "selected robot kinematics exact-slice proof changed",
    )
    camera_access = payload.get("camera_frame_zero_access")
    candidate_cameras = payload.get("camera_policy", {}).get("candidate_cameras")
    selected_raw_start = int(raw_range[0])
    _require(
        isinstance(camera_access, list)
        and isinstance(candidate_cameras, list)
        and len(camera_access) == len(candidate_cameras)
        and all(
            isinstance(record, Mapping)
            and set(record) == set(_FRAME_ZERO_CAMERA_ACCESS_FIELDS)
            and Path(str(record.get("path"))).is_absolute()
            and _valid_sha256(record.get("decoded_rgb_sha256"))
            for record in camera_access
        )
        and [record.get("camera") for record in camera_access] == candidate_cameras
        and all(
            record.get("source_aligned_frame_index") == selected_raw_start
            and record.get("action_window_frame_index") == 0
            and record.get("decoded_frame_count") == 1
            and record.get("maximum_rgb_frame_read") == 0
            and record.get("whole_file_hashed_or_read") is False
            for record in camera_access
        ),
        "camera frame-zero access differs from the selected raw start",
    )
    for record in camera_access:
        camera = str(record["camera"])
        dataset_layout.validate_file(
            str(record["path"]),
            camera,
            "undistorted.mp4",
            label=f"camera video {camera}",
        )
    _validate_camera_selection_contract(payload)
    _validate_geometry_fallback_contract(
        payload,
        require_bounded_subset_audit=(
            require_bounded_subset_audit
            or (
                HELD_PROTOCOL_ID == _HELD_V8_PROTOCOL_ID
                and payload.get("protocol_id") == _HELD_V8_PROTOCOL_ID
            )
        ),
    )
    _require(
        payload.get("artifact_sha256") == artifact_sha256(payload),
        "frame-zero manifest checksum mismatch",
    )
    return {
        "passed": True,
        "artifact_sha256": payload["artifact_sha256"],
        "bundle_sha256": bundle["sha256"],
    }


def run_frame_zero_asset_builder(
    episode_dir: str | Path,
    case_name: str,
    lock_path: str | Path,
    output_dir: str | Path,
    runtime: FrameZeroMaskRuntime,
    *,
    role: str,
    config: FrameZeroAssetConfig | None = None,
    semantic_runtime: FrameZeroSemanticGateRuntime | None = None,
) -> dict[str, Any]:
    """Build and seal one held-compatible, outcome-blind frame-zero bundle."""

    cfg = config or FrameZeroAssetConfig()
    lock_source = Path(lock_path).resolve()
    lock = load_generic_held_lock(lock_source)
    lock_file_sha256 = sha256_file(lock_source)
    authorization = authorize_frame_zero_case(lock, case_name, role=role)
    expected_config_sha256 = lock["immutable_bindings"].get("frame_zero_default_config")
    _require(
        expected_config_sha256 == artifact_sha256(asdict(cfg)),
        "effective frame-zero configuration differs from the immutable lock",
    )
    reject_future_derived_input(episode_dir, purpose="aligned episode directory")
    dataset_layout = validate_aligned_episode(
        episode_dir,
        object_id=str(authorization["object_id"]),
        episode_id=int(authorization["episode_id"]),
    )
    episode = dataset_layout.episode_dir
    output = Path(output_dir).resolve()
    _require(not output.exists(), f"frame-zero output already exists: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    try:
        intrinsics, extrinsics, calibration_inputs = _load_calibration(dataset_layout)
        action_inputs, robot_path = _action_inputs(dataset_layout)
        selected_action_arrays, action_alignment = _slice_known_action(
            robot_path,
            output / "known_action_76.npz",
            config=cfg,
        )
        action_frame_zero = int(
            action_alignment["selected_raw_frame_range_half_open"][0]
        )
        available_cameras: list[str] = []
        for camera in sorted(set(intrinsics) & set(extrinsics)):
            video = dataset_layout.optional_file(
                camera, "undistorted.mp4", label=f"camera video {camera}"
            )
            if video is None:
                continue
            available_cameras.append(camera)
        cameras = tuple(available_cameras)
        _require(
            len(cameras) >= cfg.minimum_camera_count, "too few aligned camera videos"
        )
        _require(cfg.reference_camera in cameras, "reference camera is not aligned")
        rgb_by_camera: dict[str, np.ndarray] = {}
        access_records: list[dict[str, Any]] = []
        for camera in cameras:
            rgb, access = decode_exact_frame_zero(
                episode / camera / "undistorted.mp4",
                source_aligned_frame_index=action_frame_zero,
            )
            rgb_by_camera[camera] = rgb
            access_records.append({"camera": camera, **access})
        proposals_by_camera: dict[str, list[Mapping[str, Any]]] = {}
        masks, mask_diagnostics = segment_frame_zero_views(
            rgb_by_camera,
            runtime,
            reference_camera=cfg.reference_camera,
            config=cfg.sam2,
            proposal_sink=proposals_by_camera,
        )
        legacy_masks = dict(masks)
        legacy_mask_diagnostics = list(mask_diagnostics)
        selected_cameras = tuple(sorted(masks))
        abstained_cameras = tuple(
            camera for camera in cameras if camera not in selected_cameras
        )
        _require(
            cfg.reference_camera in selected_cameras,
            "fixed frame-zero reference camera did not produce a mask",
        )
        selected_rgb = {camera: rgb_by_camera[camera] for camera in selected_cameras}
        selected_intrinsics = {
            camera: intrinsics[camera] for camera in selected_cameras
        }
        selected_extrinsics = {
            camera: extrinsics[camera] for camera in selected_cameras
        }
        geometry_fallback: dict[str, Any] | None = None
        common_assignment: dict[str, Any] | None = None
        reference_optional_safeguard: dict[str, Any] | None = None
        used_reference_optional = False
        fallback_attempts: list[dict[str, Any]] | None = None
        needs_common_assignment = False
        if len(selected_cameras) < cfg.minimum_camera_count:
            legacy_inapplicable = FrameZeroGeometryQAError(
                "legacy segmentation retained too few cameras for its frozen "
                f"geometry quorum: {len(selected_cameras)} < "
                f"{cfg.minimum_camera_count}"
            )
            same_mask_inapplicable = FrameZeroGeometryQAError(
                "same-mask projected-footprint geometry is inapplicable because "
                "legacy segmentation retained too few cameras for its frozen "
                "quorum"
            )
            fallback_attempts = [
                _fallback_attempt_failure_record("legacy", legacy_inapplicable),
                _fallback_attempt_failure_record(
                    "same-masks-projected-footprint", same_mask_inapplicable
                ),
            ]
            needs_common_assignment = True
        else:
            try:
                arrays, geometry_qa = build_frame_zero_geometry(
                    selected_rgb,
                    masks,
                    selected_intrinsics,
                    selected_extrinsics,
                    config=cfg,
                )
            except FrameZeroGeometryQAError as legacy_error:
                fallback_attempts = [
                    _fallback_attempt_failure_record("legacy", legacy_error)
                ]
                try:
                    arrays, geometry_qa = _build_frame_zero_fallback_geometry(
                        selected_rgb,
                        masks,
                        selected_intrinsics,
                        selected_extrinsics,
                        config=cfg,
                        strategy="same-masks-projected-footprint",
                    )
                    fallback_attempts.append(
                        _fallback_attempt_pass_record(
                            "same-masks-projected-footprint", geometry_qa, masks
                        )
                    )
                    selected_strategy = "same-masks-projected-footprint"
                except FrameZeroGeometryQAError as same_mask_error:
                    fallback_attempts.append(
                        _fallback_attempt_failure_record(
                            "same-masks-projected-footprint", same_mask_error
                        )
                    )
                    needs_common_assignment = True

        if needs_common_assignment:
            _require(fallback_attempts is not None, "missing fallback attempt audit")
            try:
                masks, mask_diagnostics, common_assignment = (
                    _common_voxel_mask_assignment(
                        rgb_by_camera,
                        proposals_by_camera,
                        intrinsics,
                        extrinsics,
                        reference_camera=cfg.reference_camera,
                        config=cfg,
                    )
                )
                selected_cameras = tuple(sorted(masks))
                abstained_cameras = tuple(
                    camera for camera in cameras if camera not in selected_cameras
                )
                _require_geometry(
                    len(selected_cameras) >= cfg.minimum_camera_count,
                    "common assignment retained too few cameras for the frozen quorum",
                )
                selected_rgb = {
                    camera: rgb_by_camera[camera] for camera in selected_cameras
                }
                selected_intrinsics = {
                    camera: intrinsics[camera] for camera in selected_cameras
                }
                selected_extrinsics = {
                    camera: extrinsics[camera] for camera in selected_cameras
                }
                arrays, geometry_qa = _build_frame_zero_fallback_geometry(
                    selected_rgb,
                    masks,
                    selected_intrinsics,
                    selected_extrinsics,
                    config=cfg,
                    strategy="common-voxel-assignment-projected-footprint",
                )
            except FrameZeroGeometryQAError as common_error:
                fallback_attempts.append(
                    _fallback_attempt_failure_record(
                        "common-voxel-assignment-projected-footprint", common_error
                    )
                )
                _require(
                    semantic_runtime is not None,
                    "the fourth frame-zero fallback requires the pinned official "
                    "URDF and SigLIP2 runtime",
                )
                (
                    filtered_proposals,
                    exact_robot_masks,
                    dilated_robot_masks,
                    official_urdf_audit,
                ) = semantic_runtime.prepare_proposals(
                    proposals_by_camera,
                    rgb_by_camera,
                    intrinsics,
                    extrinsics,
                    selected_action_arrays,
                )
                (
                    raw_optional_masks,
                    mask_diagnostics,
                    optional_assignment,
                ) = _common_voxel_mask_assignment(
                    rgb_by_camera,
                    filtered_proposals,
                    intrinsics,
                    extrinsics,
                    reference_camera=cfg.reference_camera,
                    config=cfg,
                    reference_optional=True,
                )
                # The outer cross-link remains populated for held/online
                # consumers; the safeguard nests the exact same immutable
                # assignment beside the URDF and semantic evidence.
                common_assignment = optional_assignment
                selected_cameras = tuple(sorted(raw_optional_masks))
                abstained_cameras = tuple(
                    camera for camera in cameras if camera not in selected_cameras
                )
                _require_geometry(
                    len(selected_cameras) == _FALLBACK_STRICT_CONSENSUS_VOTES,
                    "reference-optional assignment did not retain exactly eight cameras",
                )
                semantic_selected_proposals = _selected_proposal_audit(
                    mask_diagnostics, raw_optional_masks
                )
                semantic_audit = semantic_runtime.evaluate(
                    {camera: rgb_by_camera[camera] for camera in selected_cameras},
                    raw_optional_masks,
                    object_id=str(authorization["object_id"]),
                    selected_proposals=semantic_selected_proposals,
                )
                masks, subtraction_audit = semantic_runtime.subtract_robot(
                    raw_optional_masks,
                    exact_robot_masks,
                    dilated_robot_masks,
                )
                selected_rgb = {
                    camera: rgb_by_camera[camera] for camera in selected_cameras
                }
                selected_intrinsics = {
                    camera: intrinsics[camera] for camera in selected_cameras
                }
                selected_extrinsics = {
                    camera: extrinsics[camera] for camera in selected_cameras
                }
                arrays, geometry_qa = _build_frame_zero_fallback_geometry(
                    selected_rgb,
                    masks,
                    selected_intrinsics,
                    selected_extrinsics,
                    config=cfg,
                    strategy=FRAME_ZERO_REFERENCE_OPTIONAL_GEOMETRY_STRATEGY,
                )
                fallback_attempts.append(
                    _fallback_attempt_pass_record(
                        FRAME_ZERO_REFERENCE_OPTIONAL_GEOMETRY_STRATEGY,
                        geometry_qa,
                        masks,
                    )
                )
                reference_optional_safeguard = {
                    "contract_sha256": FRAME_ZERO_SEMANTIC_GATE_CONTRACT_SHA256,
                    "assignment": optional_assignment,
                    "official_urdf": official_urdf_audit,
                    "semantic_selected_proposals": semantic_selected_proposals,
                    "semantic_selected_mask_set_sha256": _mask_set_sha256(
                        raw_optional_masks
                    ),
                    "semantic_gate": semantic_audit,
                    "robot_subtraction": subtraction_audit,
                    "final_selected_geometry_masks": _selected_proposal_audit(
                        mask_diagnostics, masks
                    ),
                    "final_selected_geometry_mask_set_sha256": _mask_set_sha256(masks),
                }
                reference_optional_safeguard["artifact_sha256"] = artifact_sha256(
                    reference_optional_safeguard
                )
                selected_strategy = FRAME_ZERO_REFERENCE_OPTIONAL_GEOMETRY_STRATEGY
                used_reference_optional = True
            else:
                fallback_attempts.append(
                    _fallback_attempt_pass_record(
                        "common-voxel-assignment-projected-footprint",
                        geometry_qa,
                        masks,
                    )
                )
                selected_strategy = "common-voxel-assignment-projected-footprint"

        if fallback_attempts is not None:
            geometry_fallback = {
                "policy_id": (
                    FRAME_ZERO_REFERENCE_OPTIONAL_FALLBACK_POLICY_ID
                    if used_reference_optional
                    else FRAME_ZERO_GEOMETRY_FALLBACK_POLICY_ID
                ),
                "ordered_strategies": (
                    list(FRAME_ZERO_SEMANTIC_GATE_CONTRACT["application_order"])
                    if used_reference_optional
                    else [
                        "legacy",
                        "same-masks-projected-footprint",
                        "common-voxel-assignment-projected-footprint",
                    ]
                ),
                "strict_consensus_vote_count": (_FALLBACK_STRICT_CONSENSUS_VOTES),
                "reference_seed_policy": _FALLBACK_REFERENCE_SEED_POLICY,
                "reference_seed_limit": _FALLBACK_REFERENCE_SEED_COUNT,
                "common_grid_axis_count": _FALLBACK_COMMON_GRID_AXIS_COUNT,
                "common_local_requested_voxel_size_m": (
                    _FALLBACK_COMMON_LOCAL_REQUESTED_VOXEL_SIZE_M
                ),
                "minimum_coarse_component_point_count": (
                    _FALLBACK_MINIMUM_COARSE_COMPONENT_POINT_COUNT
                ),
                "local_requested_voxel_size_m": (
                    _FALLBACK_LOCAL_REQUESTED_VOXEL_SIZE_M
                ),
                "stability_requested_voxel_size_m": (
                    _FALLBACK_STABILITY_REQUESTED_VOXEL_SIZE_M
                ),
                "maximum_local_grid_point_count": (
                    _FALLBACK_MAXIMUM_LOCAL_GRID_POINT_COUNT
                ),
                "minimum_scale_stability": _FALLBACK_MINIMUM_SCALE_STABILITY,
                "selected_strategy": selected_strategy,
                "attempts": fallback_attempts,
                "legacy_selected_proposals": _selected_proposal_audit(
                    legacy_mask_diagnostics, legacy_masks
                ),
                "legacy_selected_mask_set_sha256": _mask_set_sha256(legacy_masks),
                "common_assignment": common_assignment,
                "final_selected_proposals": _selected_proposal_audit(
                    mask_diagnostics, masks
                ),
                "final_selected_mask_set_sha256": _mask_set_sha256(masks),
            }
            if used_reference_optional:
                _require(
                    reference_optional_safeguard is not None,
                    "missing reference-optional safeguard audit",
                )
                geometry_fallback["reference_optional_safeguard"] = (
                    reference_optional_safeguard
                )
            geometry_fallback["artifact_sha256"] = artifact_sha256(geometry_fallback)
        bundle_path = output / "frame_zero_bundle.npz"
        np.savez_compressed(bundle_path, **arrays)
        bundle_record = _file_record(bundle_path)
        elapsed = time.perf_counter() - started
        manifest: dict[str, Any] = {
            "schema_version": FRAME_ZERO_BUNDLE_SCHEMA_VERSION,
            "artifact_kind": FRAME_ZERO_BUNDLE_ARTIFACT_KIND,
            "protocol_id": HELD_PROTOCOL_ID,
            "case_name": case_name,
            "object_id": authorization["object_id"],
            "episode_id": authorization["episode_id"],
            "role": role,
            "frame_indices": [0],
            "lock_sha256": lock_file_sha256,
            "lock_artifact_sha256": lock["artifact_sha256"],
            "authorization": authorization,
            "bundle": bundle_record,
            "arrays": _bundle_array_records(arrays),
            "action_inputs": action_inputs,
            "action_alignment": action_alignment,
            "calibration_inputs": calibration_inputs,
            "camera_policy": {
                "policy_id": (
                    FRAME_ZERO_REFERENCE_OPTIONAL_CAMERA_SELECTION_POLICY_ID
                    if used_reference_optional
                    else FRAME_ZERO_CAMERA_SELECTION_POLICY_ID
                ),
                "rule": (
                    FRAME_ZERO_REFERENCE_OPTIONAL_CAMERA_SELECTION_RULE
                    if used_reference_optional
                    else FRAME_ZERO_CAMERA_SELECTION_RULE
                ),
                "reference_camera": cfg.reference_camera,
                "minimum_selected_camera_count": cfg.minimum_camera_count,
                "candidate_cameras": list(cameras),
                "candidate_camera_count": len(cameras),
                "selected_cameras": list(selected_cameras),
                "selected_camera_count": len(selected_cameras),
                "abstained_cameras": list(abstained_cameras),
                "abstained_camera_count": len(abstained_cameras),
            },
            "camera_frame_zero_access": access_records,
            "sam2": {
                "repository": PINNED_SAM2_REPOSITORY,
                "commit": PINNED_SAM2_COMMIT,
                "checkpoint_sha256": PINNED_SAM2_CHECKPOINT_SHA256,
                "model_config": PINNED_SAM2_MODEL_CONFIG,
                "model_id": runtime.model_id,
                "parameters": asdict(cfg.sam2),
                "image_model_only": True,
                "video_propagator_constructed": False,
                "view_diagnostics": mask_diagnostics,
                "view_diagnostics_sha256": frame_zero_view_diagnostics_sha256(
                    mask_diagnostics,
                    policy_id=(
                        FRAME_ZERO_REFERENCE_OPTIONAL_CAMERA_SELECTION_POLICY_ID
                        if used_reference_optional
                        else FRAME_ZERO_CAMERA_SELECTION_POLICY_ID
                    ),
                ),
            },
            "config": asdict(cfg),
            "geometry_qa": geometry_qa,
            "runtime": {
                "elapsed_seconds": elapsed,
                "camera_count": len(selected_cameras),
                "candidate_camera_count": len(cameras),
                "abstained_camera_count": len(abstained_cameras),
                "maximum_object_rgb_frame_read": 0,
                "decoded_rgb_frame_count_per_camera": 1,
            },
            "information_boundary": dict(FRAME_ZERO_INFORMATION_BOUNDARY),
        }
        if geometry_fallback is not None:
            manifest["geometry_fallback"] = geometry_fallback
        manifest["artifact_sha256"] = artifact_sha256(manifest)
        validate_frame_zero_bundle_manifest(manifest)
        manifest_path = output / "frame_zero_bundle.manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return manifest
    except Exception:
        # A failed run never leaves a seal.  Intermediate files are retained for
        # debugging, but consumers require the absent manifest and therefore fail.
        raise


__all__ = [
    "ABSOLUTELY_FORBIDDEN_OBJECT_PREFIXES",
    "APPROVED_CALIBRATION_SMOKE_CASE",
    "EXACT_EIGHT_SUBSET_BOUNDED_AUDIT_CONTRACT",
    "EXACT_EIGHT_SUBSET_BOUNDED_AUDIT_CONTRACT_SHA256",
    "EXACT_EIGHT_SUBSET_BOUNDED_AUDIT_SCHEMA_ID",
    "FRAME_ZERO_CAMERA_SELECTION_POLICY_ID",
    "FRAME_ZERO_CAMERA_SELECTION_RULE",
    "FRAME_ZERO_INFORMATION_BOUNDARY",
    "FRAME_ZERO_REFERENCE_OPTIONAL_CAMERA_SELECTION_POLICY_ID",
    "FRAME_ZERO_REFERENCE_OPTIONAL_CAMERA_SELECTION_RULE",
    "FRAME_ZERO_REFERENCE_OPTIONAL_FALLBACK_POLICY_ID",
    "FRAME_ZERO_SEMANTIC_GATE_CONTRACT",
    "FRAME_ZERO_SEMANTIC_GATE_CONTRACT_SHA256",
    "FrameZeroAssetConfig",
    "FrameZeroSemanticGateRuntime",
    "HELD_TARGET_CASES_V1",
    "PinnedFrameZeroSam2Runtime",
    "artifact_sha256",
    "authorize_frame_zero_case",
    "build_frame_zero_geometry",
    "decode_exact_frame_zero",
    "frame_zero_view_diagnostics_sha256",
    "load_generic_held_lock",
    "reject_future_derived_input",
    "run_frame_zero_asset_builder",
    "segment_frame_zero_views",
    "sha256_file",
    "validate_frame_zero_bundle_manifest",
    "validate_generic_held_lock",
]
