"""Pinned SAM 2.1 fallback masks for the Deform360 ``001-rope`` pilot."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

import numpy as np


PINNED_SAM2_REPOSITORY = "https://github.com/facebookresearch/sam2"
PINNED_SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
PINNED_SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
)
PINNED_SAM2_CHECKPOINT_SHA256 = (
    "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
)
PINNED_SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
DEFORM360_SAM2_MASK_SCHEMA_VERSION = 1


class _EpisodeAccessConfig(Protocol):
    expected_episode_count: int
    source_episode_ids: tuple[int, ...]
    target_episode_ids: tuple[int, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sam2_mask_artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def validate_sam2_mask_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == DEFORM360_SAM2_MASK_SCHEMA_VERSION,
        "unsupported SAM2 mask artifact schema",
    )
    _require(
        payload.get("artifact_kind") == "Deform360RopeSam2MaskAudit",
        "unexpected SAM2 mask artifact kind",
    )
    _require(
        payload.get("result_sha256") == sam2_mask_artifact_sha256(payload),
        "SAM2 mask artifact checksum mismatch",
    )
    return {"passed": True, "result_sha256": payload["result_sha256"]}


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_sam2_episode_access(
    episode_index: int,
    config: _EpisodeAccessConfig,
    *,
    held_out_prediction_seal_sha256: str | None,
) -> dict[str, Any]:
    _require(
        0 <= episode_index < config.expected_episode_count,
        "episode index is outside the locked cohort",
    )
    is_target = episode_index in config.target_episode_ids
    if is_target:
        _require(
            _valid_sha256(held_out_prediction_seal_sha256),
            "full target masks require a valid held-out prediction seal",
        )
    else:
        _require(
            held_out_prediction_seal_sha256 is None,
            "a held-out prediction seal is only valid for a target episode",
        )
    return {
        "episode_index": episode_index,
        "split": "target"
        if is_target
        else "source"
        if episode_index in config.source_episode_ids
        else "calibration",
        "target_future_annotation_unlocked": is_target,
        "held_out_prediction_seal_sha256": held_out_prediction_seal_sha256,
    }


@dataclass(frozen=True)
class RopeSam2MaskConfig:
    points_per_side: int = 16
    points_per_batch: int = 64
    predicted_iou_threshold: float = 0.7
    stability_score_threshold: float = 0.8
    minimum_mask_region_area: int = 100
    minimum_colored_pixels_per_family: int = 30
    pink_hue_low: int = 145
    pink_hue_high_wrap: int = 5
    green_hue_low: int = 35
    green_hue_high: int = 100
    saturation_minimum: int = 70
    value_minimum: int = 45

    def __post_init__(self) -> None:
        _require(self.points_per_side >= 4, "points_per_side is too small")
        _require(self.points_per_batch >= 1, "points_per_batch must be positive")
        _require(
            0.0 <= self.predicted_iou_threshold <= 1.0,
            "invalid predicted-IoU threshold",
        )
        _require(
            0.0 <= self.stability_score_threshold <= 1.0,
            "invalid stability threshold",
        )
        _require(
            self.minimum_colored_pixels_per_family >= 1,
            "colored-pixel threshold must be positive",
        )


def _color_families(
    rgb: np.ndarray, config: RopeSam2MaskConfig
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - exercised on the GPU host
        raise RuntimeError(
            "OpenCV is required for the SAM2 rope-mask adapter"
        ) from error
    image = np.asarray(rgb, dtype=np.uint8)
    _require(image.ndim == 3 and image.shape[2] == 3, "RGB image must be HxWx3")
    hue, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    vivid = (saturation >= config.saturation_minimum) & (value >= config.value_minimum)
    pink = vivid & ((hue >= config.pink_hue_low) | (hue <= config.pink_hue_high_wrap))
    green = vivid & (hue >= config.green_hue_low) & (hue <= config.green_hue_high)
    return pink, green


def rope_mask_candidate_diagnostics(
    rgb: np.ndarray,
    mask: np.ndarray,
    config: RopeSam2MaskConfig,
) -> dict[str, Any]:
    candidate = np.asarray(mask, dtype=bool)
    _require(
        candidate.shape == np.asarray(rgb).shape[:2], "candidate mask shape mismatch"
    )
    pink, green = _color_families(rgb, config)
    rows, columns = np.nonzero(candidate)
    area = int(len(columns))
    pink_count = int(np.count_nonzero(candidate & pink))
    green_count = int(np.count_nonzero(candidate & green))
    eligible = bool(
        area >= config.minimum_mask_region_area
        and pink_count >= config.minimum_colored_pixels_per_family
        and green_count >= config.minimum_colored_pixels_per_family
    )
    if area >= 3:
        coordinates = np.column_stack((columns, rows)).astype(np.float64)
        if len(coordinates) > 20_000:
            stride = int(np.ceil(len(coordinates) / 20_000))
            coordinates = coordinates[::stride]
        eigenvalues = np.linalg.eigvalsh(np.cov(coordinates.T))
        elongation = float(
            np.sqrt((float(eigenvalues[-1]) + 1.0) / (float(eigenvalues[0]) + 1.0))
        )
    else:
        elongation = 1.0
    color_purity = (pink_count + green_count) / area if area else 0.0
    score = (
        float(np.sqrt(pink_count * green_count) * elongation * color_purity)
        if eligible
        else -1.0
    )
    return {
        "eligible": eligible,
        "score": score,
        "area_pixels": area,
        "pink_pixels": pink_count,
        "green_pixels": green_count,
        "elongation": elongation,
        "colored_pixel_fraction": color_purity,
    }


class RopeSam2VideoPredictor:
    """Automatic first-frame rope selection followed by SAM 2.1 propagation."""

    def __init__(
        self,
        sam2_repository: str | Path,
        checkpoint: str | Path,
        *,
        device: str = "cuda",
        config: RopeSam2MaskConfig | None = None,
    ) -> None:
        self.repository = Path(sam2_repository).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.device = device
        self.config = config or RopeSam2MaskConfig()
        _require(self.repository.is_dir(), "SAM2 repository does not exist")
        _require(self.checkpoint.is_file(), "SAM2 checkpoint does not exist")
        try:
            commit = subprocess.run(
                ["git", "-C", str(self.repository), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError("cannot verify the SAM2 repository revision") from error
        _require(commit == PINNED_SAM2_COMMIT, "SAM2 repository revision mismatch")
        _require(
            _sha256_file(self.checkpoint) == PINNED_SAM2_CHECKPOINT_SHA256,
            "SAM2 checkpoint checksum mismatch",
        )
        if str(self.repository) not in sys.path:
            sys.path.insert(0, str(self.repository))
        try:
            import torch
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2, build_sam2_video_predictor
        except ImportError as error:  # pragma: no cover - GPU-host integration
            raise RuntimeError(
                "Pinned SAM2 runtime dependencies are unavailable"
            ) from error
        self._torch = torch
        image_model = build_sam2(
            PINNED_SAM2_MODEL_CONFIG,
            str(self.checkpoint),
            device=device,
            apply_postprocessing=False,
        )
        self._mask_generator = SAM2AutomaticMaskGenerator(
            image_model,
            points_per_side=self.config.points_per_side,
            points_per_batch=self.config.points_per_batch,
            pred_iou_thresh=self.config.predicted_iou_threshold,
            stability_score_thresh=self.config.stability_score_threshold,
            min_mask_region_area=self.config.minimum_mask_region_area,
        )
        self._video_predictor = build_sam2_video_predictor(
            PINNED_SAM2_MODEL_CONFIG,
            str(self.checkpoint),
            device=device,
            apply_postprocessing=False,
        )
        self.model_id = (
            f"causal4d_public/rope-sam2.1-small-automatic-v1@{PINNED_SAM2_COMMIT[:12]}"
        )
        self.diagnostics: list[dict[str, Any]] = []

    def _first_frame_rgb(self, video_path: Path) -> np.ndarray:
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - GPU-host integration
            raise RuntimeError("OpenCV is required for SAM2 video decoding") from error
        if video_path.is_dir():
            images = sorted(
                path
                for path in video_path.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg"}
            )
            _require(images, f"frame directory contains no JPEG images: {video_path}")
            bgr = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
            ok = bgr is not None
        else:
            capture = cv2.VideoCapture(str(video_path))
            try:
                ok, bgr = capture.read()
            finally:
                capture.release()
        _require(ok, f"cannot read first video frame: {video_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def select_initial_mask(
        self, video_path: Path
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Select one first-frame rope mask without propagating the video."""

        rgb = self._first_frame_rgb(video_path)
        with (
            self._torch.inference_mode(),
            self._torch.autocast(
                "cuda",
                dtype=self._torch.bfloat16,
                enabled=self.device.startswith("cuda"),
            ),
        ):
            annotations = self._mask_generator.generate(rgb)
        candidates = []
        for index, annotation in enumerate(annotations):
            diagnostics = rope_mask_candidate_diagnostics(
                rgb, annotation["segmentation"], self.config
            )
            diagnostics.update(
                {
                    "candidate_index": index,
                    "predicted_iou": float(annotation["predicted_iou"]),
                    "stability_score": float(annotation["stability_score"]),
                }
            )
            candidates.append((diagnostics["score"], index, diagnostics))
        eligible = [candidate for candidate in candidates if candidate[0] >= 0.0]
        _require(eligible, f"SAM2 found no rope-like first-frame mask for {video_path}")
        _, selected_index, selected = max(
            eligible, key=lambda item: (item[0], -item[1])
        )
        mask = np.asarray(annotations[selected_index]["segmentation"], dtype=bool)
        return mask, {
            "camera": video_path.parent.name,
            "video": video_path.name,
            "automatic_candidate_count": len(annotations),
            "eligible_candidate_count": len(eligible),
            "selected": selected,
        }

    def segment(
        self,
        video_path: Path,
        text_prompt: str,
        box_xywh: object | None = None,
    ) -> Iterator[tuple[int, np.ndarray]]:
        del text_prompt
        _require(box_xywh is None, "external boxes are not used by this pinned adapter")
        initial_mask, diagnostic = self.select_initial_mask(video_path)
        yield from self._propagate_from_initial_mask(
            video_path, initial_mask, diagnostic=diagnostic
        )

    def segment_from_initial_mask(
        self,
        video_path: Path,
        initial_mask: np.ndarray,
        *,
        initialization: Mapping[str, Any],
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Propagate a previously sealed first-frame material mask."""

        rgb = self._first_frame_rgb(video_path)
        mask = np.asarray(initial_mask, dtype=bool)
        _require(mask.shape == rgb.shape[:2], "sealed initial mask shape mismatch")
        diagnostic = {
            "camera": video_path.parent.name,
            "video": video_path.name,
            "initialization": "sealed-prefix-mask",
            "sealed_initial_mask": dict(initialization),
            "selected": rope_mask_candidate_diagnostics(rgb, mask, self.config),
        }
        yield from self._propagate_from_initial_mask(
            video_path, mask, diagnostic=diagnostic
        )

    def _propagate_from_initial_mask(
        self,
        video_path: Path,
        initial_mask: np.ndarray,
        *,
        diagnostic: dict[str, Any],
    ) -> Iterator[tuple[int, np.ndarray]]:
        state = self._video_predictor.init_state(
            video_path=str(video_path),
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
        self._video_predictor.add_new_mask(state, 0, 1, initial_mask)
        areas = []
        with (
            self._torch.inference_mode(),
            self._torch.autocast(
                "cuda",
                dtype=self._torch.bfloat16,
                enabled=self.device.startswith("cuda"),
            ),
        ):
            for frame_index, _, logits in self._video_predictor.propagate_in_video(
                state
            ):
                mask = (logits[0, 0] > 0.0).cpu().numpy()
                areas.append(int(np.count_nonzero(mask)))
                yield int(frame_index), mask
        diagnostic["propagation"] = {
            "frame_count": len(areas),
            "empty_frame_count": sum(area == 0 for area in areas),
            "area_pixels": {
                "minimum": min(areas) if areas else None,
                "median": float(np.median(areas)) if areas else None,
                "maximum": max(areas) if areas else None,
            },
        }
        self.diagnostics.append(diagnostic)

    def close(self) -> None:
        self._mask_generator = None
        self._video_predictor = None
        if self.device.startswith("cuda"):
            self._torch.cuda.empty_cache()


def build_sam2_mask_audit(
    *,
    protocol_id: str,
    episode_access: Mapping[str, Any],
    predictor: RopeSam2VideoPredictor,
    output_paths: Mapping[str, str | Path],
    view_audit_result_sha256: str | None = None,
) -> dict[str, Any]:
    _require(
        view_audit_result_sha256 is None or _valid_sha256(view_audit_result_sha256),
        "invalid SAM2 view-audit checksum",
    )
    outputs = {
        camera: {
            "path": Path(path).name,
            "sha256": _sha256_file(Path(path)),
            "bytes": Path(path).stat().st_size,
        }
        for camera, path in sorted(output_paths.items())
    }
    payload: dict[str, Any] = {
        "schema_version": DEFORM360_SAM2_MASK_SCHEMA_VERSION,
        "artifact_kind": "Deform360RopeSam2MaskAudit",
        "protocol_id": protocol_id,
        "episode_access": dict(episode_access),
        "upstream": {
            "repository": PINNED_SAM2_REPOSITORY,
            "commit": PINNED_SAM2_COMMIT,
            "checkpoint_url": PINNED_SAM2_CHECKPOINT_URL,
            "checkpoint_sha256": PINNED_SAM2_CHECKPOINT_SHA256,
            "model_config": PINNED_SAM2_MODEL_CONFIG,
        },
        "parameters": asdict(predictor.config),
        "model_id": predictor.model_id,
        "view_selection": {
            "view_audit_result_sha256": view_audit_result_sha256,
            "cross_view_gate_applied": view_audit_result_sha256 is not None,
        },
        "camera_diagnostics": predictor.diagnostics,
        "outputs": outputs,
        "claim_boundary": (
            "Public SAM2 fallback, not the gated SAM3 mask stage used by Deform360; "
            "mask quality must pass source multiview geometry QA before physics use."
        ),
    }
    payload["result_sha256"] = sam2_mask_artifact_sha256(payload)
    return payload


def write_sam2_mask_audit(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "DEFORM360_SAM2_MASK_SCHEMA_VERSION",
    "PINNED_SAM2_CHECKPOINT_SHA256",
    "PINNED_SAM2_COMMIT",
    "RopeSam2MaskConfig",
    "RopeSam2VideoPredictor",
    "build_sam2_mask_audit",
    "rope_mask_candidate_diagnostics",
    "sam2_mask_artifact_sha256",
    "validate_sam2_episode_access",
    "validate_sam2_mask_artifact",
    "write_sam2_mask_audit",
]
