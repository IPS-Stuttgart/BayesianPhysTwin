from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from bayesian_phystwin.deform360_fresh_bias_guard_diagnostic import (
    apply_frozen_fresh_bias_guard_arrays,
)


REPO = Path(__file__).resolve().parents[1]
SOURCE_LOCK = (
    REPO
    / "results/sota/deform360_bias_aware_guarded_belief_v4"
    / "prospective_lock.json"
)


def test_fresh_guard_signature_has_no_target_or_outcome() -> None:
    parameters = inspect.signature(apply_frozen_fresh_bias_guard_arrays).parameters

    assert "target" not in parameters
    assert "outcome" not in parameters


def test_zero_physical_response_preserves_baseline_bit_exactly() -> None:
    rng = np.random.default_rng(4)
    baseline = rng.normal(0.0, 0.1, size=(76, 32, 3)).astype(np.float32)
    measurement = np.full_like(baseline, np.nan)
    visibility = np.zeros(baseline.shape[:2], dtype=bool)
    centers = np.arange(16, dtype=np.int64)
    for update_index, frame in enumerate((19, 38, 57)):
        measurement[frame, centers] = baseline[frame, centers]
        visibility[frame, centers] = True
    source_lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))

    report, selected = apply_frozen_fresh_bias_guard_arrays(
        baseline,
        baseline,
        baseline,
        measurement,
        visibility,
        visibility,
        center_ids=centers,
        update_frames=np.asarray([19, 38, 57]),
        selected_camera_count=8,
        triangulation_inlier_view_count=np.full((3, 16), 4),
        triangulation_median_reprojection_px=np.ones((3, 16)),
        source_lock=source_lock,
    )

    np.testing.assert_array_equal(selected, baseline)
    assert report["candidate_available_count"] == 0
    assert report["accepted_count"] == 0
    assert report["exact_fallback_interval_count"] == 3
