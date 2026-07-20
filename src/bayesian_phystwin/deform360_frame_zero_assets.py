"""Future-blind frame-zero assets for held Deform360 online belief.

This module deliberately has no HDF5 reader and no outcome argument.  It decodes
exactly one RGB frame from each selected camera, segments those materialized
frames with the pinned SAM 2.1 image model, and derives a multiview visual hull,
surface colors, depth maps, and projection matrices from immutable calibration.
The known robot trajectory is bound as an exogenous action input; it is never
used to construct object geometry.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from causal4d_public.deform360_object_sam2 import (
    DeformableObjectSam2MaskConfig,
    deformable_object_mask_candidate_diagnostics,
    mask_appearance_descriptor,
    mask_appearance_similarity,
)
from causal4d_public.deform360_sam2 import (
    PINNED_SAM2_CHECKPOINT_SHA256,
    PINNED_SAM2_COMMIT,
    PINNED_SAM2_MODEL_CONFIG,
    PINNED_SAM2_REPOSITORY,
)
from causal4d_public.deform360_visual_hull import (
    carve_candidate_points,
    regular_grid_in_bounds,
)


FRAME_ZERO_BUNDLE_SCHEMA_VERSION = 1
HELD_PROTOCOL_ID = "deform360-held-online-belief-v1"
HELD_LOCK_ARTIFACT_KIND = "Deform360HeldOnlineBeliefLock"
FRAME_ZERO_BUNDLE_ARTIFACT_KIND = "Deform360HeldFrameZeroBundle"
FRAME_ZERO_CAMERA_SELECTION_POLICY_ID = (
    "deform360-frame-zero-reference-consistent-camera-abstention-v2"
)
FRAME_ZERO_CAMERA_SELECTION_RULE = (
    "process every aligned calibrated camera; keep the fixed reference and "
    "only non-reference views with a frozen-threshold-eligible mask"
)

_NO_AUTOMATIC_CANDIDATES = "no-automatic-mask-candidates"
_NO_MASK_THRESHOLD_CANDIDATES = "no-candidate-met-frozen-mask-thresholds"
_NO_REFERENCE_CONSISTENT_CANDIDATES = (
    "no-candidate-met-frozen-reference-appearance-threshold"
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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
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
    resolved = path.resolve()
    _require(resolved.is_file(), f"required input is missing: {resolved}")
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


def frame_zero_view_diagnostics_sha256(
    diagnostics: Sequence[Mapping[str, Any]],
) -> str:
    """Bind all attempted views to the source-only v2 selection policy."""

    return hashlib.sha256(
        _canonical_bytes(
            {
                "policy_id": FRAME_ZERO_CAMERA_SELECTION_POLICY_ID,
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

    resolved = Path(path).resolve()
    lowered = resolved.as_posix().lower()
    _require(
        resolved.suffix.lower() not in _FORBIDDEN_INPUT_SUFFIXES,
        f"{purpose} may not use an HDF5 input",
    )
    _require(
        not any(token in lowered for token in _FORBIDDEN_DERIVATION_TOKENS),
        f"{purpose} appears future-derived",
    )
    return resolved


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
    """Decode exactly action-window frame zero and audit its source alignment.

    ``source_aligned_frame_index`` is selected from the known robot action only.
    The returned RGB material has action-window index zero; no later object frame
    is decoded, hashed, or passed to the segmentation model.
    """

    path = reject_future_derived_input(video_path, purpose="camera video")
    _require(path.suffix.lower() in {".mp4", ".mov", ".mkv"}, "unsupported video input")
    _require(path.is_file(), f"camera video is missing: {path}")
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
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    cameras = tuple(sorted(rgb_by_camera))
    _require(reference_camera in cameras, "reference camera is unavailable")
    reference_rgb = np.asarray(rgb_by_camera[reference_camera], dtype=np.uint8)
    reference_mask, reference_diagnostic = _select_reference_mask(
        reference_rgb, runtime.generate(reference_rgb), config
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
        mask, diagnostic = _select_reference_consistent_mask(
            rgb, runtime.generate(rgb), reference_descriptor, config
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
            "median_depth_m": float(np.median(result[valid]))
            if np.any(valid)
            else None,
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
    _require(
        len(hull) >= config.minimum_hull_point_count,
        "frame-zero visual hull is too small",
    )
    keep, surface, components = _largest_grid_component(
        hull,
        bounds_minimum=minimum,
        bounds_maximum=maximum,
        grid_shape=grid_diagnostics["grid_shape"],
    )
    _require(
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
    _require(len(surface_points) >= 128, "too few object surface points")

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
    _require(
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
    _require(diagnostics["geometry_qa_passed"], "frame-zero geometry failed QA")
    return arrays, diagnostics


def _load_calibration(
    episode_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    intrinsic_path = reject_future_derived_input(
        episode_dir / "undistorted_intrinsics.npy", purpose="camera intrinsics"
    )
    extrinsic_path = reject_future_derived_input(
        episode_dir / "extrinsics.npy", purpose="camera extrinsics"
    )
    _require(intrinsic_path.is_file(), "undistorted intrinsics are missing")
    _require(extrinsic_path.is_file(), "camera extrinsics are missing")
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


def _controller_centres(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    _require(np.all(np.isfinite(values)), "robot actions are non-finite")
    if values.ndim == 3:
        _require(values.shape[-1] == 3, "robot actions must end in xyz")
        values = values[:, None, :, :]
    _require(
        values.ndim == 4 and values.shape[-1] == 3,
        "robot actions must have shape (T,P,3) or (T,G,P,3)",
    )
    _require(len(values) >= 2, "robot action is too short")
    return np.mean(values, axis=2)


def _closure_confidence(openings: np.ndarray, gripper_count: int) -> np.ndarray:
    aperture = np.asarray(openings, dtype=np.float64)
    if aperture.ndim == 1:
        aperture = aperture[:, None]
    _require(
        aperture.ndim == 2 and aperture.shape[1] == gripper_count,
        "robot openings do not match action groups",
    )
    _require(np.all(np.isfinite(aperture)), "robot openings are non-finite")
    low = np.quantile(aperture, 0.1, axis=0)
    high = np.quantile(aperture, 0.9, axis=0)
    span = high - low
    confidence = np.ones_like(aperture)
    varying = span > 1e-9
    confidence[:, varying] = np.clip(
        (high[varying] - aperture[:, varying]) / span[varying], 0.0, 1.0
    )
    return confidence


def select_action_only_window(
    actions: np.ndarray,
    openings: np.ndarray,
    *,
    window_length_frames: int,
    prediction_frame_count: int,
    candidate_first_frame: int,
    candidate_stride_frames: int,
) -> dict[str, Any]:
    """Apply the frozen action-only Deform360 window rule.

    The score is the mean per-gripper centre path, with each step weighted by
    the minimum endpoint closure confidence.  Candidate order supplies the
    frozen earliest-start tie break.
    """

    centres = _controller_centres(actions)
    closed = _closure_confidence(openings, centres.shape[1])
    _require(len(closed) == len(centres), "action/opening frame counts differ")
    _require(
        2 <= prediction_frame_count < window_length_frames <= len(centres),
        "action episode is shorter than its frozen window",
    )
    starts = np.arange(
        candidate_first_frame,
        len(centres) - window_length_frames + 1,
        candidate_stride_frames,
        dtype=np.int64,
    )
    _require(len(starts) > 0, "action window has no complete candidate")
    candidates = []
    for start_value in starts:
        start = int(start_value)
        stop = start + window_length_frames
        selected = centres[start:stop]
        step = np.linalg.norm(np.diff(selected, axis=0), axis=-1)
        weighted = step * np.minimum(closed[start : stop - 1], closed[start + 1 : stop])
        per_gripper_path = np.sum(weighted, axis=0)
        candidates.append(
            {
                "frame_range_half_open": [start, stop],
                "mean_closed_weighted_path_length_m": float(np.mean(per_gripper_path)),
                "maximum_closed_weighted_path_length_m": float(
                    np.max(per_gripper_path)
                ),
            }
        )
    selected = max(
        candidates,
        key=lambda record: float(record["mean_closed_weighted_path_length_m"]),
    )
    start, stop = selected["frame_range_half_open"]
    return {
        "selection_rule": (
            "maximize mean per-gripper centre path weighted per step by minimum "
            "endpoint closure confidence; earliest candidate breaks ties"
        ),
        "selection_inputs": ["robot.npz:actions", "robot.npz:openings"],
        "candidate_first_frame": candidate_first_frame,
        "candidate_stride_frames": candidate_stride_frames,
        "candidate_count": len(candidates),
        "selected_raw_frame_range_half_open": [start, stop],
        "prediction_raw_frame_range_half_open": [start, start + prediction_frame_count],
        "tracking_tail_frame_count": stop - (start + prediction_frame_count),
        "selected_score": selected,
        "object_geometry_used_for_selection": False,
        "tactile_used_for_selection": False,
    }


def _slice_known_action(
    robot_path: Path,
    output_path: Path,
    *,
    config: FrameZeroAssetConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with np.load(robot_path, allow_pickle=False) as stored:
        _require(
            "actions" in stored and "openings" in stored,
            "robot action fields are missing",
        )
        alignment = select_action_only_window(
            stored["actions"],
            stored["openings"],
            window_length_frames=config.action_window_length_frames,
            prediction_frame_count=config.prediction_frame_count,
            candidate_first_frame=config.action_candidate_first_frame,
            candidate_stride_frames=config.action_candidate_stride_frames,
        )
        start, stop = alignment["prediction_raw_frame_range_half_open"]
        source_frame_count = len(stored["actions"])
        arrays: dict[str, np.ndarray] = {}
        for name in stored.files:
            value = np.asarray(stored[name])
            if value.ndim >= 1 and len(value) == source_frame_count:
                arrays[name] = value[start:stop]
            else:
                arrays[name] = value
    _require(
        len(arrays["actions"])
        == len(arrays["openings"])
        == config.prediction_frame_count,
        "sliced action has the wrong prediction length",
    )
    np.savez_compressed(output_path, **arrays)
    alignment.update(
        {
            "source_robot_frame_count": source_frame_count,
            "prediction_frame_count": config.prediction_frame_count,
            "selected_action_bundle": _file_record(output_path),
            "selected_action_arrays": _bundle_array_records(arrays),
        }
    )
    return arrays, alignment


def _action_inputs(episode_dir: Path) -> tuple[dict[str, dict[str, Any]], Path]:
    robot = reject_future_derived_input(
        episode_dir / "robot" / "robot.npz", purpose="robot action"
    )
    metadata = reject_future_derived_input(
        episode_dir / "robot" / "robot.meta.json", purpose="robot action metadata"
    )
    return {
        "robot_trajectory": _file_record(robot),
        "robot_metadata": _file_record(metadata),
    }, robot


def _validate_case_directory(
    episode_dir: Path, authorization: Mapping[str, Any]
) -> None:
    expected_episode = f"episode_{int(authorization['episode_id']):04d}"
    _require(episode_dir.name == expected_episode, "episode directory/case mismatch")
    _require(
        episode_dir.parent.name == authorization["object_id"],
        "object directory/case mismatch",
    )


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
        isinstance(policy, Mapping)
        and set(policy) == policy_keys
        and policy.get("policy_id") == FRAME_ZERO_CAMERA_SELECTION_POLICY_ID
        and policy.get("rule") == FRAME_ZERO_CAMERA_SELECTION_RULE,
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
        and reference_camera in selected_cameras
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
        == frame_zero_view_diagnostics_sha256(diagnostics),
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
        if record.get("view_selected") is True:
            _require(
                record.get("abstained") is False
                and record.get("abstention_reason") is None
                and eligible_count >= 1
                and isinstance(record.get("selected"), Mapping),
                "selected camera diagnostics are inconsistent",
            )
            diagnostic_selected.append(camera)
        else:
            _require(
                record.get("view_selected") is False
                and record.get("abstained") is True
                and record.get("abstention_reason") in allowed_abstention_reasons
                and eligible_count == 0
                and record.get("selected") is None,
                "abstained camera diagnostics are inconsistent",
            )
            diagnostic_abstained.append(camera)
    _require(
        diagnostic_selected == selected_cameras
        and diagnostic_abstained == abstained_cameras,
        "frame-zero camera policy differs from its diagnostics",
    )


def validate_frame_zero_bundle_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
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
        _require(isinstance(record, Mapping), "invalid action input record")
        _require(
            Path(str(record.get("path"))).is_absolute(), "action path is not absolute"
        )
        _require(_valid_sha256(record.get("sha256")), "invalid action checksum")
        _require(isinstance(record.get("size_bytes"), int), "invalid action size")
    action_alignment = payload.get("action_alignment")
    _require(
        isinstance(action_alignment, Mapping),
        "frame-zero manifest lacks action alignment",
    )
    raw_range = action_alignment.get("selected_raw_frame_range_half_open")
    prediction_range = action_alignment.get("prediction_raw_frame_range_half_open")
    _require(
        isinstance(raw_range, list)
        and len(raw_range) == 2
        and int(raw_range[1]) - int(raw_range[0]) == 81,
        "action selection is not an 81-frame window",
    )
    _require(
        isinstance(prediction_range, list)
        and len(prediction_range) == 2
        and int(prediction_range[0]) == int(raw_range[0])
        and int(prediction_range[1]) - int(prediction_range[0]) == 76,
        "known action bundle is not the 76-frame prediction window",
    )
    selected_action = action_alignment.get("selected_action_bundle")
    _require(isinstance(selected_action, Mapping), "selected action bundle is missing")
    _require(
        Path(str(selected_action.get("path"))).is_absolute(),
        "selected action bundle path is not absolute",
    )
    _require(
        _valid_sha256(selected_action.get("sha256")), "invalid selected action checksum"
    )
    _require(
        isinstance(selected_action.get("size_bytes"), int),
        "invalid selected action size",
    )
    _validate_camera_selection_contract(payload)
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
    episode = reject_future_derived_input(
        episode_dir, purpose="aligned episode directory"
    )
    _require(episode.is_dir(), f"aligned episode directory is missing: {episode}")
    _validate_case_directory(episode, authorization)
    output = Path(output_dir).resolve()
    _require(not output.exists(), f"frame-zero output already exists: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    try:
        intrinsics, extrinsics, calibration_inputs = _load_calibration(episode)
        action_inputs, robot_path = _action_inputs(episode)
        _, action_alignment = _slice_known_action(
            robot_path,
            output / "known_action_76.npz",
            config=cfg,
        )
        action_frame_zero = int(
            action_alignment["selected_raw_frame_range_half_open"][0]
        )
        cameras = tuple(
            sorted(
                camera
                for camera in set(intrinsics) & set(extrinsics)
                if (episode / camera / "undistorted.mp4").is_file()
            )
        )
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
        masks, mask_diagnostics = segment_frame_zero_views(
            rgb_by_camera,
            runtime,
            reference_camera=cfg.reference_camera,
            config=cfg.sam2,
        )
        selected_cameras = tuple(sorted(masks))
        abstained_cameras = tuple(
            camera for camera in cameras if camera not in selected_cameras
        )
        _require(
            cfg.reference_camera in selected_cameras,
            "fixed frame-zero reference camera did not produce a mask",
        )
        _require(
            len(selected_cameras) >= cfg.minimum_camera_count,
            "too few non-abstaining frame-zero cameras",
        )
        selected_rgb = {camera: rgb_by_camera[camera] for camera in selected_cameras}
        selected_intrinsics = {
            camera: intrinsics[camera] for camera in selected_cameras
        }
        selected_extrinsics = {
            camera: extrinsics[camera] for camera in selected_cameras
        }
        arrays, geometry_qa = build_frame_zero_geometry(
            selected_rgb,
            masks,
            selected_intrinsics,
            selected_extrinsics,
            config=cfg,
        )
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
                "policy_id": FRAME_ZERO_CAMERA_SELECTION_POLICY_ID,
                "rule": FRAME_ZERO_CAMERA_SELECTION_RULE,
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
                    mask_diagnostics
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
    "FRAME_ZERO_CAMERA_SELECTION_POLICY_ID",
    "FRAME_ZERO_CAMERA_SELECTION_RULE",
    "FRAME_ZERO_INFORMATION_BOUNDARY",
    "FrameZeroAssetConfig",
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
