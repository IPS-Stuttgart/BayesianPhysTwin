"""Generic first-frame SAM2 selection for the Deform360 replication cohort."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from .deform360_sam2 import RopeSam2VideoPredictor, _require


@dataclass(frozen=True)
class DeformableObjectSam2MaskConfig:
    """Source-locked automatic-mask settings shared across object strata."""

    points_per_side: int = 24
    points_per_batch: int = 64
    predicted_iou_threshold: float = 0.70
    stability_score_threshold: float = 0.80
    minimum_mask_region_area: int = 100
    minimum_area_fraction: float = 0.0005
    maximum_area_fraction: float = 0.65
    minimum_foreground_contrast: float = 0.06
    maximum_normalized_center_distance: float = 0.90
    border_side_penalty: float = 0.55
    minimum_reference_appearance_similarity: float = 0.35

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
        _require(self.minimum_mask_region_area >= 1, "mask area must be positive")
        _require(
            0.0 < self.minimum_area_fraction < self.maximum_area_fraction <= 1.0,
            "invalid area-fraction interval",
        )
        _require(
            0.0 <= self.minimum_foreground_contrast <= 1.0,
            "invalid foreground-contrast threshold",
        )
        _require(
            0.0 < self.maximum_normalized_center_distance <= 1.0,
            "invalid center-distance threshold",
        )
        _require(
            0.0 < self.border_side_penalty <= 1.0,
            "invalid border-side penalty",
        )
        _require(
            0.0 <= self.minimum_reference_appearance_similarity <= 1.0,
            "invalid reference-appearance threshold",
        )


def deformable_object_mask_candidate_diagnostics(
    rgb: np.ndarray,
    mask: np.ndarray,
    config: DeformableObjectSam2MaskConfig,
) -> dict[str, Any]:
    """Score a SAM2 candidate without object-specific colors or target outcomes."""

    image = np.asarray(rgb, dtype=np.uint8)
    candidate = np.asarray(mask, dtype=bool)
    _require(image.ndim == 3 and image.shape[2] == 3, "RGB image must be HxWx3")
    _require(candidate.shape == image.shape[:2], "candidate mask shape mismatch")

    height, width = candidate.shape
    rows, columns = np.nonzero(candidate)
    area = int(len(columns))
    area_fraction = area / float(height * width)
    if area:
        centroid_x = float(np.mean(columns))
        centroid_y = float(np.mean(rows))
        half_diagonal = 0.5 * float(np.hypot(width, height))
        center_distance = float(
            np.hypot(centroid_x - 0.5 * (width - 1), centroid_y - 0.5 * (height - 1))
            / half_diagonal
        )
        bounding_width = int(columns.max() - columns.min() + 1)
        bounding_height = int(rows.max() - rows.min() + 1)
        bounding_fill = area / float(bounding_width * bounding_height)
    else:
        centroid_x = centroid_y = center_distance = 0.0
        bounding_width = bounding_height = 0
        bounding_fill = 0.0

    border_pixels = np.concatenate(
        (image[0], image[-1], image[1:-1, 0], image[1:-1, -1]), axis=0
    )
    background_rgb = np.median(border_pixels.astype(np.float64), axis=0)
    if area:
        color_distance = np.linalg.norm(
            image[candidate].astype(np.float64) - background_rgb[None, :], axis=1
        )
        foreground_contrast = float(np.median(color_distance) / np.sqrt(3 * 255**2))
    else:
        foreground_contrast = 0.0

    border_sides = int(np.any(candidate[0]))
    border_sides += int(np.any(candidate[-1]))
    border_sides += int(np.any(candidate[:, 0]))
    border_sides += int(np.any(candidate[:, -1]))
    eligible = bool(
        area >= config.minimum_mask_region_area
        and config.minimum_area_fraction
        <= area_fraction
        <= config.maximum_area_fraction
        and foreground_contrast >= config.minimum_foreground_contrast
        and center_distance <= config.maximum_normalized_center_distance
    )
    centrality = float(np.exp(-2.0 * center_distance**2))
    border_factor = float(config.border_side_penalty**border_sides)
    score = (
        1000.0
        * np.sqrt(area_fraction)
        * foreground_contrast
        * centrality
        * border_factor
        if eligible
        else -1.0
    )
    return {
        "eligible": eligible,
        "score": float(score),
        "area_pixels": area,
        "area_fraction": area_fraction,
        "centroid_xy": [centroid_x, centroid_y],
        "normalized_center_distance": center_distance,
        "bounding_box_wh": [bounding_width, bounding_height],
        "bounding_box_fill_fraction": bounding_fill,
        "border_side_count": border_sides,
        "foreground_contrast": foreground_contrast,
        "border_background_rgb": background_rgb.tolist(),
    }


def mask_appearance_descriptor(rgb: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Describe masked appearance with illumination-tolerant color statistics."""

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - GPU-host integration
        raise RuntimeError("OpenCV is required for mask appearance") from error
    image = np.asarray(rgb, dtype=np.uint8)
    candidate = np.asarray(mask, dtype=bool)
    _require(image.ndim == 3 and image.shape[2] == 3, "RGB image must be HxWx3")
    _require(candidate.shape == image.shape[:2], "candidate mask shape mismatch")
    _require(np.any(candidate), "appearance mask is empty")

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        candidate.astype(np.uint8),
        [18, 8],
        [0, 180, 0, 256],
    ).astype(np.float64)
    histogram /= max(float(np.sum(histogram)), 1.0)
    lab_values = lab[candidate].astype(np.float64)
    rows, columns = np.nonzero(candidate)
    bounding_area = float(
        (rows.max() - rows.min() + 1) * (columns.max() - columns.min() + 1)
    )
    coordinates = np.column_stack((columns, rows)).astype(np.float64)
    if len(coordinates) > 20_000:
        stride = int(np.ceil(len(coordinates) / 20_000))
        coordinates = coordinates[::stride]
    if len(coordinates) >= 3:
        eigenvalues = np.linalg.eigvalsh(np.cov(coordinates.T))
        elongation = float(
            np.sqrt((float(eigenvalues[-1]) + 1.0) / (float(eigenvalues[0]) + 1.0))
        )
    else:
        elongation = 1.0
    return {
        "hs_histogram": histogram.ravel().tolist(),
        "lab_median": np.median(lab_values, axis=0).tolist(),
        "lab_iqr": (
            np.quantile(lab_values, 0.75, axis=0)
            - np.quantile(lab_values, 0.25, axis=0)
        ).tolist(),
        "bounding_box_fill_fraction": float(
            np.count_nonzero(candidate) / bounding_area
        ),
        "area_fraction": float(np.mean(candidate)),
        "elongation": elongation,
    }


def mask_appearance_similarity(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    """Compare two descriptors without assuming a view-specific object position."""

    reference_histogram = np.asarray(reference["hs_histogram"], dtype=np.float64)
    candidate_histogram = np.asarray(candidate["hs_histogram"], dtype=np.float64)
    _require(
        reference_histogram.shape == candidate_histogram.shape,
        "appearance histograms have different shapes",
    )
    histogram_intersection = float(
        np.sum(np.minimum(reference_histogram, candidate_histogram))
    )
    reference_lab = np.asarray(reference["lab_median"], dtype=np.float64)
    candidate_lab = np.asarray(candidate["lab_median"], dtype=np.float64)
    lab_distance = float(
        np.linalg.norm(reference_lab - candidate_lab) / np.sqrt(3 * 255**2)
    )
    lab_similarity = float(np.exp(-4.0 * lab_distance))
    reference_fill = max(float(reference["bounding_box_fill_fraction"]), 1e-6)
    candidate_fill = max(float(candidate["bounding_box_fill_fraction"]), 1e-6)
    fill_similarity = float(np.exp(-abs(np.log(candidate_fill / reference_fill))))
    reference_area = max(float(reference["area_fraction"]), 1e-6)
    candidate_area = max(float(candidate["area_fraction"]), 1e-6)
    area_similarity = float(np.exp(-0.5 * abs(np.log(candidate_area / reference_area))))
    reference_elongation = max(float(reference["elongation"]), 1.0)
    candidate_elongation = max(float(candidate["elongation"]), 1.0)
    elongation_similarity = float(
        np.exp(-abs(np.log(candidate_elongation / reference_elongation)))
    )
    shape_similarity = float(
        (fill_similarity * area_similarity * elongation_similarity) ** (1.0 / 3.0)
    )
    combined = float(
        0.65 * histogram_intersection + 0.25 * lab_similarity + 0.10 * shape_similarity
    )
    return {
        "combined": combined,
        "hs_histogram_intersection": histogram_intersection,
        "lab_similarity": lab_similarity,
        "shape_similarity": shape_similarity,
        "fill_similarity": fill_similarity,
        "area_similarity": area_similarity,
        "elongation_similarity": elongation_similarity,
    }


class DeformableObjectSam2VideoPredictor(RopeSam2VideoPredictor):
    """Object-agnostic first-frame selection followed by pinned SAM2 propagation."""

    def __init__(
        self,
        sam2_repository: str | Path,
        checkpoint: str | Path,
        *,
        device: str = "cuda",
        config: DeformableObjectSam2MaskConfig | None = None,
    ) -> None:
        object_config = config or DeformableObjectSam2MaskConfig()
        super().__init__(
            sam2_repository,
            checkpoint,
            device=device,
            config=object_config,  # type: ignore[arg-type]
        )
        self.model_id = (
            "causal4d_public/deformable-object-sam2.1-small-automatic-v1@"
            f"{self.model_id.rsplit('@', 1)[-1]}"
        )

    def _automatic_annotations(self, rgb: np.ndarray) -> list[Mapping[str, Any]]:
        with (
            self._torch.inference_mode(),
            self._torch.autocast(
                "cuda",
                dtype=self._torch.bfloat16,
                enabled=self.device.startswith("cuda"),
            ),
        ):
            return self._mask_generator.generate(rgb)

    def select_initial_mask(
        self, video_path: Path
    ) -> tuple[np.ndarray, dict[str, Any]]:
        rgb = self._first_frame_rgb(video_path)
        annotations = self._automatic_annotations(rgb)

        candidates = []
        for index, annotation in enumerate(annotations):
            diagnostics = deformable_object_mask_candidate_diagnostics(
                rgb,
                annotation["segmentation"],
                self.config,
            )
            predicted_iou = float(annotation["predicted_iou"])
            stability = float(annotation["stability_score"])
            if diagnostics["eligible"]:
                diagnostics["score"] *= float(
                    np.sqrt(max(0.0, predicted_iou * stability))
                )
            diagnostics.update(
                {
                    "candidate_index": index,
                    "predicted_iou": predicted_iou,
                    "stability_score": stability,
                }
            )
            candidates.append((diagnostics["score"], index, diagnostics))
        eligible = [candidate for candidate in candidates if candidate[0] >= 0.0]
        _require(
            eligible,
            f"SAM2 found no deformable-object first-frame mask for {video_path}",
        )
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

    def select_initial_mask_with_reference(
        self,
        video_path: Path,
        reference_rgb: np.ndarray,
        reference_mask: np.ndarray,
        *,
        reference_camera: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Select a source-view mask using appearance from a fixed source view."""

        ranked, summary = self.initial_mask_candidates_with_reference(
            video_path,
            reference_rgb,
            reference_mask,
            reference_camera=reference_camera,
            maximum_candidates=1,
        )
        selected = ranked[0]
        return np.asarray(selected["mask"], dtype=bool), {
            **summary,
            "selected": selected["diagnostic"],
        }

    def initial_mask_candidates_with_reference(
        self,
        video_path: Path,
        reference_rgb: np.ndarray,
        reference_mask: np.ndarray,
        *,
        reference_camera: str,
        maximum_candidates: int = 4,
        include_below_appearance_threshold: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Return ranked appearance candidates for calibrated joint selection.

        Candidate generation remains independent of the physical-state residual.
        The caller may use camera calibration to choose one candidate per view.
        """

        rgb = self._first_frame_rgb(video_path)
        return self.initial_mask_candidates_from_rgb_with_reference(
            rgb,
            camera=video_path.parent.name,
            video_name=video_path.name,
            reference_rgb=reference_rgb,
            reference_mask=reference_mask,
            reference_camera=reference_camera,
            maximum_candidates=maximum_candidates,
            include_below_appearance_threshold=include_below_appearance_threshold,
        )

    def initial_mask_candidates_from_rgb_with_reference(
        self,
        rgb: np.ndarray,
        *,
        camera: str,
        video_name: str,
        reference_rgb: np.ndarray,
        reference_mask: np.ndarray,
        reference_camera: str,
        maximum_candidates: int = 4,
        include_below_appearance_threshold: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Return ranked candidates for an explicitly supplied observation frame."""

        _require(maximum_candidates >= 1, "maximum_candidates must be positive")

        image = np.asarray(rgb, dtype=np.uint8)
        _require(image.ndim == 3 and image.shape[2] == 3, "RGB image must be HxWx3")
        reference_descriptor = mask_appearance_descriptor(reference_rgb, reference_mask)
        annotations = self._automatic_annotations(image)
        candidates = []
        for index, annotation in enumerate(annotations):
            mask = np.asarray(annotation["segmentation"], dtype=bool)
            diagnostics = deformable_object_mask_candidate_diagnostics(
                image, mask, self.config
            )
            basic_eligible = bool(
                diagnostics["area_pixels"] >= self.config.minimum_mask_region_area
                and self.config.minimum_area_fraction
                <= diagnostics["area_fraction"]
                <= self.config.maximum_area_fraction
                and diagnostics["foreground_contrast"]
                >= self.config.minimum_foreground_contrast
            )
            if basic_eligible:
                descriptor = mask_appearance_descriptor(image, mask)
                similarity = mask_appearance_similarity(
                    reference_descriptor, descriptor
                )
            else:
                descriptor = None
                similarity = {
                    "combined": 0.0,
                    "hs_histogram_intersection": 0.0,
                    "lab_similarity": 0.0,
                    "shape_similarity": 0.0,
                }
            appearance_eligible = bool(
                basic_eligible
                and similarity["combined"]
                >= self.config.minimum_reference_appearance_similarity
            )
            predicted_iou = float(annotation["predicted_iou"])
            stability = float(annotation["stability_score"])
            raw_score = (
                similarity["combined"] ** 3
                * similarity["shape_similarity"] ** 4
                * np.sqrt(diagnostics["area_fraction"])
                * np.sqrt(max(0.0, predicted_iou * stability))
                if basic_eligible
                else -1.0
            )
            joint_eligible = bool(
                appearance_eligible
                or (include_below_appearance_threshold and basic_eligible)
            )
            score = raw_score if joint_eligible else -1.0
            diagnostics.update(
                {
                    "eligible": appearance_eligible,
                    "score": float(score if appearance_eligible else -1.0),
                    "candidate_index": index,
                    "predicted_iou": predicted_iou,
                    "stability_score": stability,
                    "reference_appearance": similarity,
                    "appearance_descriptor": descriptor,
                }
            )
            if include_below_appearance_threshold:
                diagnostics.update(
                    {
                        "joint_selection_eligible": joint_eligible,
                        "joint_selection_score": float(score),
                    }
                )
            candidates.append((score, index, diagnostics))
        eligible = [candidate for candidate in candidates if candidate[0] >= 0.0]
        _require(
            eligible,
            f"SAM2 found no reference-consistent mask for {camera}/{video_name}",
        )
        ranked = sorted(eligible, key=lambda item: (-item[0], item[1]))
        ranked_records = [
            {
                "mask": np.asarray(
                    annotations[candidate_index]["segmentation"], dtype=bool
                ),
                "prior_score": float(score),
                "candidate_index": int(candidate_index),
                "diagnostic": diagnostic,
            }
            for score, candidate_index, diagnostic in ranked[:maximum_candidates]
        ]
        summary = {
            "camera": camera,
            "video": video_name,
            "initialization": "source-reference-appearance",
            "reference_camera": reference_camera,
            "reference_descriptor": reference_descriptor,
            "automatic_candidate_count": len(annotations),
            "eligible_candidate_count": len(eligible),
        }
        if include_below_appearance_threshold:
            summary["appearance_eligible_candidate_count"] = sum(
                int(diagnostic["eligible"]) for _, _, diagnostic in candidates
            )
        return ranked_records, summary

    def segment_from_initial_mask(
        self,
        video_path: Path,
        initial_mask: np.ndarray,
        *,
        initialization: Mapping[str, Any],
    ) -> Iterator[tuple[int, np.ndarray]]:
        rgb = self._first_frame_rgb(video_path)
        mask = np.asarray(initial_mask, dtype=bool)
        _require(mask.shape == rgb.shape[:2], "sealed initial mask shape mismatch")
        diagnostic = {
            "camera": video_path.parent.name,
            "video": video_path.name,
            "initialization": "sealed-prefix-mask",
            "sealed_initial_mask": dict(initialization),
            "selected": deformable_object_mask_candidate_diagnostics(
                rgb, mask, self.config
            ),
        }
        yield from self._propagate_from_initial_mask(
            video_path, mask, diagnostic=diagnostic
        )


__all__ = [
    "DeformableObjectSam2MaskConfig",
    "DeformableObjectSam2VideoPredictor",
    "deformable_object_mask_candidate_diagnostics",
    "mask_appearance_descriptor",
    "mask_appearance_similarity",
]
