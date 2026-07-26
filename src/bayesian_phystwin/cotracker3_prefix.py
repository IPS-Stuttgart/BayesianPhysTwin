"""Exact-prefix sparse CoTracker3 runtime.

The runtime deliberately exposes the same small ``track_prefix`` boundary used
by the Deform360 camera adapters.  A call for update frame ``u`` decodes only
RGB frames ``[0, u]`` and queries material points from frame zero.  No dense
future tracking product is accepted as a shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@dataclass(frozen=True)
class CoTracker3PrefixConfig:
    """Frozen numerical choices for sparse causal-prefix tracking."""

    maximum_image_side: int = 512
    model_window_length: int = 60

    def __post_init__(self) -> None:
        _require(self.maximum_image_side >= 64, "maximum image side is too small")
        _require(self.model_window_length >= 2, "model window must exceed one frame")


class CoTracker3PrefixRuntime:
    """Load the released offline CoTracker3 predictor and query exact prefixes."""

    def __init__(
        self,
        source_root: str | Path,
        checkpoint: str | Path,
        *,
        device: str = "cuda:0",
        config: CoTracker3PrefixConfig | None = None,
    ) -> None:
        source = Path(source_root).resolve()
        checkpoint_path = Path(checkpoint).resolve()
        _require((source / "cotracker" / "predictor.py").is_file(), "invalid source")
        _require(checkpoint_path.is_file(), "CoTracker3 checkpoint is missing")
        self.config = config or CoTracker3PrefixConfig()
        self.source_root = source
        self.checkpoint = checkpoint_path
        self.source_revision = _git_revision(source)
        self.predictor_source_sha256 = _sha256_file(
            source / "cotracker" / "predictor.py"
        )
        self.checkpoint_sha256 = _sha256_file(checkpoint_path)
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        torch = importlib.import_module("torch")
        _require(torch.cuda.is_available(), "CoTracker3 requires CUDA")
        predictor_module = importlib.import_module("cotracker.predictor")
        self._torch = torch
        self._device = torch.device(device)
        self.device_name = str(self._device)
        self._model = predictor_module.CoTrackerPredictor(
            checkpoint=str(checkpoint_path),
            v2=False,
            offline=True,
            window_len=self.config.model_window_length,
        ).to(self._device)

    @staticmethod
    def _decode_prefix(
        video_path: Path, update_frame: int
    ) -> tuple[np.ndarray, str]:
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - server integration
            raise RuntimeError("OpenCV is required for CoTracker3 input") from error
        _require(update_frame >= 0, "update frame is negative")
        capture = cv2.VideoCapture(str(video_path))
        frames: list[np.ndarray] = []
        digest = hashlib.sha256()
        try:
            for frame_index in range(update_frame + 1):
                okay, bgr = capture.read()
                _require(
                    okay,
                    f"cannot read causal frame {frame_index} from {video_path}",
                )
                rgb = np.ascontiguousarray(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                )
                frames.append(rgb)
                digest.update(str(rgb.dtype).encode("ascii"))
                digest.update(np.asarray(rgb.shape, dtype=np.int64).tobytes())
                digest.update(rgb.tobytes())
        finally:
            capture.release()
        return np.stack(frames), digest.hexdigest()

    @staticmethod
    def _resize_prefix(
        rgb: np.ndarray, maximum_side: int
    ) -> tuple[np.ndarray, tuple[int, int]]:
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - server integration
            raise RuntimeError("OpenCV is required for CoTracker3 input") from error
        original_height, original_width = rgb.shape[1:3]
        scale = min(1.0, maximum_side / max(original_height, original_width))
        height = max(8, int(original_height * scale) // 8 * 8)
        width = max(8, int(original_width * scale) // 8 * 8)
        if (height, width) == (original_height, original_width):
            return rgb, (original_height, original_width)
        resized = np.stack(
            [
                cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                for frame in rgb
            ]
        )
        return resized, (original_height, original_width)

    def track_prefix(
        self,
        video_path: str | Path,
        query_pixels_xy: np.ndarray,
        update_frame: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Track frame-zero queries through exactly frames ``[0, update]``."""

        queries = np.asarray(query_pixels_xy, dtype=np.float64)
        _require(
            queries.ndim == 2
            and queries.shape[1] == 2
            and np.all(np.isfinite(queries)),
            "query pixels must have finite shape (N, 2)",
        )
        path = Path(video_path).resolve()
        rgb, prefix_sha256 = self._decode_prefix(path, update_frame)
        resized, original_shape = self._resize_prefix(
            rgb, self.config.maximum_image_side
        )
        original_height, original_width = original_shape
        height, width = resized.shape[1:3]
        scaled_queries = queries.copy()
        scaled_queries[:, 0] *= (width - 1) / max(1, original_width - 1)
        scaled_queries[:, 1] *= (height - 1) / max(1, original_height - 1)
        query_tensor = np.column_stack(
            (
                np.zeros(len(scaled_queries), dtype=np.float32),
                scaled_queries.astype(np.float32),
            )
        )
        torch = self._torch
        video = (
            torch.from_numpy(np.ascontiguousarray(resized))
            .permute(0, 3, 1, 2)[None]
            .float()
            .to(self._device)
        )
        query = torch.from_numpy(query_tensor[None]).float().to(self._device)
        start = time.perf_counter()
        with torch.no_grad():
            tracks, visibility = self._model(video, queries=query)
        runtime_seconds = time.perf_counter() - start
        endpoint = tracks[0, update_frame].detach().cpu().numpy().astype(np.float32)
        visible = (
            visibility[0, update_frame]
            .detach()
            .cpu()
            .numpy()
            .astype(bool)
        )
        endpoint[:, 0] *= (original_width - 1) / max(1, width - 1)
        endpoint[:, 1] *= (original_height - 1) / max(1, height - 1)
        finite = np.all(np.isfinite(endpoint), axis=1)
        visible &= finite
        del video, query, tracks, visibility
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (
            endpoint,
            visible,
            {
                "tracker": "CoTracker3",
                "direction": "forward_exact_prefix",
                "source_prefix_frame_range_half_open": [0, update_frame + 1],
                "maximum_video_frame_read": update_frame,
                "decoded_frame_count": update_frame + 1,
                "decoded_rgb_prefix_sha256": prefix_sha256,
                "original_image_shape": [original_height, original_width],
                "inference_image_shape": [height, width],
                "query_count": len(queries),
                "visible_query_count": int(np.sum(visible)),
                "runtime_seconds": runtime_seconds,
            },
        )

    def close(self) -> None:
        """Release the model and cached CUDA allocations."""

        self._model = None
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


__all__ = ["CoTracker3PrefixConfig", "CoTracker3PrefixRuntime"]
