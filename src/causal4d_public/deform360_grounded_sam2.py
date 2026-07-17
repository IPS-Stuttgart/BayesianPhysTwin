"""Text-grounded, box-prompted SAM2 masks for Deform360 frame zero."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .deform360_object_sam2 import (
    DeformableObjectSam2MaskConfig,
    deformable_object_mask_candidate_diagnostics,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class GroundedSam2MaskConfig:
    model_id: str
    model_revision: str
    transformers_version: str
    box_threshold: float = 0.15
    text_threshold: float = 0.15
    maximum_boxes_per_camera: int = 4
    maximum_candidates_per_camera: int = 8
    minimum_sam2_score: float = 0.0
    duplicate_iou_threshold: float = 0.98

    def __post_init__(self) -> None:
        _require(bool(self.model_id), "grounding model identity is empty")
        _require(
            len(self.model_revision) == 40
            and all(value in "0123456789abcdef" for value in self.model_revision),
            "grounding model revision must be a full Git SHA",
        )
        _require(bool(self.transformers_version), "transformers version is empty")
        _require(0.0 < self.box_threshold < 1.0, "invalid box threshold")
        _require(0.0 < self.text_threshold < 1.0, "invalid text threshold")
        _require(self.maximum_boxes_per_camera >= 1, "box budget must be positive")
        _require(
            self.maximum_candidates_per_camera >= 1,
            "candidate budget must be positive",
        )
        _require(-1.0 <= self.minimum_sam2_score <= 1.0, "invalid SAM2 threshold")
        _require(
            0.0 < self.duplicate_iou_threshold <= 1.0,
            "invalid duplicate-IoU threshold",
        )


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return float(intersection / union) if union else 1.0


def rank_grounded_sam2_candidates(
    rgb: np.ndarray,
    *,
    boxes_xyxy: np.ndarray,
    box_scores: np.ndarray,
    masks_by_box: list[np.ndarray],
    mask_scores_by_box: list[np.ndarray],
    object_config: DeformableObjectSam2MaskConfig,
    grounded_config: GroundedSam2MaskConfig,
) -> list[dict[str, Any]]:
    """Rank detector/SAM2 candidates without simulator-state information."""

    image = np.asarray(rgb, dtype=np.uint8)
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    scores = np.asarray(box_scores, dtype=np.float64)
    _require(image.ndim == 3 and image.shape[2] == 3, "RGB image must be HxWx3")
    _require(boxes.ndim == 2 and boxes.shape[1] == 4, "boxes must have shape Bx4")
    _require(scores.shape == (len(boxes),), "box scores have the wrong shape")
    _require(
        len(masks_by_box) == len(boxes) == len(mask_scores_by_box),
        "grounding and SAM2 result counts differ",
    )

    candidates: list[dict[str, Any]] = []
    for box_index, (box, box_score, raw_masks, raw_mask_scores) in enumerate(
        zip(boxes, scores, masks_by_box, mask_scores_by_box, strict=True)
    ):
        masks = np.asarray(raw_masks)
        mask_scores = np.asarray(raw_mask_scores, dtype=np.float64)
        _require(
            masks.ndim == 3 and masks.shape[1:] == image.shape[:2],
            "SAM2 masks have the wrong shape",
        )
        _require(
            mask_scores.shape == (len(masks),),
            "SAM2 scores have the wrong shape",
        )
        for mask_index, (raw_mask, mask_score) in enumerate(
            zip(masks, mask_scores, strict=True)
        ):
            mask = np.asarray(raw_mask, dtype=bool)
            diagnostics = deformable_object_mask_candidate_diagnostics(
                image, mask, object_config
            )
            if (
                not diagnostics["eligible"]
                or mask_score < grounded_config.minimum_sam2_score
            ):
                continue
            prior_score = float(
                max(0.0, box_score)
                * max(0.0, mask_score)
                * np.sqrt(max(0.0, diagnostics["foreground_contrast"]))
            )
            diagnostics.update(
                {
                    "grounding_box_index": box_index,
                    "grounding_box_xyxy": box.tolist(),
                    "grounding_score": float(box_score),
                    "sam2_mask_index": mask_index,
                    "sam2_score": float(mask_score),
                    "joint_selection_score": prior_score,
                }
            )
            candidates.append(
                {
                    "mask": mask,
                    "prior_score": prior_score,
                    "candidate_index": len(candidates),
                    "diagnostic": diagnostics,
                }
            )

    ranked = sorted(
        candidates,
        key=lambda record: (-float(record["prior_score"]), record["candidate_index"]),
    )
    distinct: list[dict[str, Any]] = []
    for record in ranked:
        if any(
            _mask_iou(record["mask"], retained["mask"])
            >= grounded_config.duplicate_iou_threshold
            for retained in distinct
        ):
            continue
        distinct.append(record)
        if len(distinct) == grounded_config.maximum_candidates_per_camera:
            break
    _require(distinct, "text grounding produced no eligible SAM2 masks")
    for index, record in enumerate(distinct):
        record["candidate_index"] = index
    return distinct


class GroundedSam2ImagePredictor:
    """Grounding DINO boxes followed by pinned SAM2 image masks."""

    def __init__(
        self,
        sam2_repository: str | Path,
        checkpoint: str | Path,
        *,
        device: str,
        object_config: DeformableObjectSam2MaskConfig,
        grounded_config: GroundedSam2MaskConfig,
        cache_dir: str | Path | None = None,
    ) -> None:
        repository = str(Path(sam2_repository).resolve())
        if repository not in sys.path:
            sys.path.insert(0, repository)
        try:
            import torch
            import transformers
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
        except ImportError as error:  # pragma: no cover - GPU integration
            raise RuntimeError(
                "Grounding DINO and pinned SAM2 dependencies are unavailable"
            ) from error
        _require(
            transformers.__version__ == grounded_config.transformers_version,
            "transformers version differs from the frozen observation model",
        )
        self._torch = torch
        self.device = device
        self.object_config = object_config
        self.config = grounded_config
        common = {
            "revision": grounded_config.model_revision,
            "cache_dir": None if cache_dir is None else str(cache_dir),
        }
        self._processor = AutoProcessor.from_pretrained(
            grounded_config.model_id, **common
        )
        self._grounder = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounded_config.model_id, **common
        ).to(device)
        self._grounder.eval()
        sam2 = build_sam2(
            "configs/sam2.1/sam2.1_hiera_s.yaml",
            str(Path(checkpoint).resolve()),
            device=device,
            apply_postprocessing=False,
        )
        self._sam2 = SAM2ImagePredictor(sam2)

    def candidates_from_rgb(
        self,
        rgb: np.ndarray,
        *,
        prompt: str,
        camera: str,
        video_name: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        image = np.asarray(rgb, dtype=np.uint8)
        _require(image.ndim == 3 and image.shape[2] == 3, "RGB image must be HxWx3")
        _require(bool(prompt.strip()), "grounding prompt is empty")
        try:
            from PIL import Image
        except ImportError as error:  # pragma: no cover - GPU integration
            raise RuntimeError("Pillow is required for text grounding") from error
        pil_image = Image.fromarray(image)
        inputs = self._processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)
        with self._torch.inference_mode():
            outputs = self._grounder(**inputs)
        result = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=[image.shape[:2]],
        )[0]
        boxes = result["boxes"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy()
        order = np.argsort(-scores, kind="stable")[
            : self.config.maximum_boxes_per_camera
        ]
        boxes = boxes[order]
        scores = scores[order]
        _require(len(boxes) > 0, f"text grounding found no box in {camera}")

        self._sam2.set_image(image.copy())
        masks_by_box = []
        mask_scores_by_box = []
        for box in boxes:
            masks, mask_scores, _ = self._sam2.predict(
                box=box,
                multimask_output=True,
            )
            masks_by_box.append(np.asarray(masks) > 0)
            mask_scores_by_box.append(np.asarray(mask_scores, dtype=np.float64))
        candidates = rank_grounded_sam2_candidates(
            image,
            boxes_xyxy=boxes,
            box_scores=scores,
            masks_by_box=masks_by_box,
            mask_scores_by_box=mask_scores_by_box,
            object_config=self.object_config,
            grounded_config=self.config,
        )
        return candidates, {
            "camera": camera,
            "video": video_name,
            "initialization": "text-grounded-box-prompted-sam2",
            "text_prompt": prompt,
            "grounding_box_count": len(boxes),
            "eligible_candidate_count": len(candidates),
            "grounding_model_id": self.config.model_id,
            "grounding_model_revision": self.config.model_revision,
            "transformers_version": self.config.transformers_version,
        }

    def close(self) -> None:
        grounder, self._grounder = self._grounder, None
        sam2, self._sam2 = self._sam2, None
        del grounder, sam2
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


__all__ = [
    "GroundedSam2ImagePredictor",
    "GroundedSam2MaskConfig",
    "rank_grounded_sam2_candidates",
]
