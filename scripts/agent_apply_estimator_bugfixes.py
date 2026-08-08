from pathlib import Path


def replace_exact(path: str, old: str, new: str, *, expected: int = 1) -> None:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected:
        raise SystemExit(
            f"{path}: expected {expected} occurrences, found {count}: {old!r}"
        )
    source.write_text(text.replace(old, new), encoding="utf-8")


Path("src/bayesian_phystwin/mask_distance.py").write_text(
    '''"""Canonical Euclidean distance from mask interior to background."""

from __future__ import annotations

import numpy as np


def _squared_distance_transform_1d(cost: np.ndarray) -> np.ndarray:
    """Return the exact squared 1-D Euclidean distance transform."""

    values = np.asarray(cost, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("cost must be a nonempty vector")
    size = values.size
    locations = np.empty(size, dtype=np.int64)
    boundaries = np.empty(size + 1, dtype=np.float64)
    output = np.empty(size, dtype=np.float64)
    envelope = 0
    locations[0] = 0
    boundaries[0] = -np.inf
    boundaries[1] = np.inf

    for query in range(1, size):
        while True:
            previous = int(locations[envelope])
            separation = (
                values[query]
                + float(query * query)
                - values[previous]
                - float(previous * previous)
            ) / (2.0 * (query - previous))
            if separation > boundaries[envelope]:
                break
            envelope -= 1
        envelope += 1
        locations[envelope] = query
        boundaries[envelope] = separation
        boundaries[envelope + 1] = np.inf

    envelope = 0
    for query in range(size):
        while boundaries[envelope + 1] < query:
            envelope += 1
        previous = int(locations[envelope])
        delta = query - previous
        output[query] = float(delta * delta) + values[previous]
    return output


def _interior_mask_distance_fallback(mask: np.ndarray) -> np.ndarray:
    """Pure-NumPy exact EDT used when SciPy is unavailable."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or min(values.shape, default=0) == 0:
        raise ValueError("mask must be a nonempty 2-D array")
    padded = np.pad(values, 1, constant_values=False)
    maximum_squared_distance = float(sum(padded.shape) ** 2)
    squared = np.where(padded, maximum_squared_distance, 0.0)
    row_pass = np.empty_like(squared)
    for row in range(squared.shape[0]):
        row_pass[row] = _squared_distance_transform_1d(squared[row])
    column_pass = np.empty_like(row_pass)
    for column in range(row_pass.shape[1]):
        column_pass[:, column] = _squared_distance_transform_1d(
            row_pass[:, column]
        )
    return np.sqrt(column_pass)[1:-1, 1:-1]


def interior_mask_distance(mask: np.ndarray) -> np.ndarray:
    """Return border-aware Euclidean distance for pixels inside ``mask``.

    The image exterior is background. This makes a foreground pixel on an
    image edge exactly one pixel from background and gives identical
    semantics with and without SciPy.
    """

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or min(values.shape, default=0) == 0:
        raise ValueError("mask must be a nonempty 2-D array")
    padded = np.pad(values, 1, constant_values=False)
    try:
        from scipy.ndimage import distance_transform_edt
    except (ImportError, OSError):
        return _interior_mask_distance_fallback(values)
    distance = np.asarray(distance_transform_edt(padded), dtype=np.float64)
    return distance[1:-1, 1:-1]


__all__ = ["interior_mask_distance"]
''',
    encoding="utf-8",
)

replace_exact(
    "src/bayesian_phystwin/phystwin_raw_cues.py",
    "import numpy as np\n\n\n@dataclass",
    "import numpy as np\n\nfrom .mask_distance import interior_mask_distance\n\n\n@dataclass",
)
replace_exact(
    "src/bayesian_phystwin/phystwin_raw_cues.py",
    '''    try:\n        from scipy.ndimage import distance_transform_edt\n    except ImportError as error:\n        raise RuntimeError("raw camera cues require scipy") from error\n''',
    "",
)
replace_exact(
    "src/bayesian_phystwin/phystwin_raw_cues.py",
    "distance_transform_edt(object_mask)",
    "interior_mask_distance(object_mask)",
)

replace_exact(
    "src/bayesian_phystwin/phystwin_cotracker3_cues.py",
    "import numpy as np\n\nfrom .phystwin_raw_cues import",
    "import numpy as np\n\nfrom .mask_distance import interior_mask_distance\nfrom .phystwin_raw_cues import",
)
replace_exact(
    "src/bayesian_phystwin/phystwin_cotracker3_cues.py",
    '''    try:\n        from scipy.ndimage import distance_transform_edt\n    except ImportError as error:\n        raise RuntimeError("rich cue extraction requires scipy") from error\n''',
    "",
)
replace_exact(
    "src/bayesian_phystwin/phystwin_cotracker3_cues.py",
    "distance_transform_edt(object_mask)",
    "interior_mask_distance(object_mask)",
)
replace_exact(
    "src/bayesian_phystwin/phystwin_cotracker3_cues.py",
    "    camera_counts = np.sum(validity, axis=0).astype(np.int16)\n",
    "    candidate_camera_counts = np.sum(validity, axis=0).astype(np.int16)\n    camera_counts = np.zeros_like(candidate_camera_counts)\n",
)
replace_exact(
    "src/bayesian_phystwin/phystwin_cotracker3_cues.py",
    "        selected = (camera_counts[frame] >= 2) & (\n",
    "        selected = (candidate_camera_counts[frame] >= 2) & (\n",
)
replace_exact(
    "src/bayesian_phystwin/phystwin_cotracker3_cues.py",
    "        points[frame, selected] = solved\n        squared_error = np.zeros(np.sum(selected), dtype=float)\n        weight_sum = np.zeros(np.sum(selected), dtype=float)\n",
    "        squared_error = np.zeros(np.sum(selected), dtype=float)\n        weight_sum = np.zeros(np.sum(selected), dtype=float)\n        positive_depth_count = np.zeros(np.sum(selected), dtype=np.int16)\n",
)
replace_exact(
    "src/bayesian_phystwin/phystwin_cotracker3_cues.py",
    "            camera_valid = validity[camera, frame, selected] & (depth > 0.0)\n            camera_weight = effective_weight[camera, selected] * camera_valid\n",
    "            camera_valid = (\n                validity[camera, frame, selected]\n                & (effective_weight[camera, selected] > 0.0)\n                & (depth > 0.0)\n            )\n            positive_depth_count += camera_valid.astype(np.int16)\n            camera_weight = effective_weight[camera, selected] * camera_valid\n",
)
replace_exact(
    "src/bayesian_phystwin/phystwin_cotracker3_cues.py",
    "        usable = weight_sum > 0.0\n        frame_error = np.full(np.sum(selected), np.nan, dtype=float)\n        frame_error[usable] = np.sqrt(squared_error[usable] / weight_sum[usable])\n        error[frame, selected] = frame_error\n",
    "        camera_counts[frame, selected] = positive_depth_count\n        usable = (positive_depth_count >= 2) & (weight_sum > 0.0)\n        selected_indices = np.flatnonzero(selected)\n        points[frame, selected_indices[usable]] = solved[usable]\n        frame_error = np.full(np.sum(selected), np.nan, dtype=float)\n        frame_error[usable] = np.sqrt(squared_error[usable] / weight_sum[usable])\n        error[frame, selected] = frame_error\n",
)

replace_exact(
    "src/bayesian_phystwin/phystwin_alltracker_cues.py",
    "import numpy as np\n\nfrom .deform360_raw_camera_observation import",
    "import numpy as np\n\nfrom .mask_distance import interior_mask_distance\nfrom .deform360_raw_camera_observation import",
)
replace_exact(
    "src/bayesian_phystwin/phystwin_alltracker_cues.py",
    '''    try:\n        from scipy.ndimage import distance_transform_edt\n    except ImportError as exc:\n        raise RuntimeError("AllTracker cue extraction requires scipy") from exc\n''',
    "",
)
replace_exact(
    "src/bayesian_phystwin/phystwin_alltracker_cues.py",
    "distance_transform_edt(object_mask)",
    "interior_mask_distance(object_mask)",
)

replace_exact(
    "src/bayesian_phystwin/phystwin_motioncrafter_assimilation.py",
    "import numpy as np\n\nfrom .phystwin_graph import",
    "import numpy as np\n\nfrom .mask_distance import interior_mask_distance\nfrom .phystwin_graph import",
)
replace_exact(
    "src/bayesian_phystwin/phystwin_motioncrafter_assimilation.py",
    '''def _mask_boundary_distance(mask: np.ndarray) -> np.ndarray:\n    """Return an interior pixel distance without making SciPy a base dependency."""\n\n    values = np.asarray(mask, dtype=bool)\n    try:\n        from scipy.ndimage import distance_transform_edt\n    except (ImportError, OSError):\n        padded = np.pad(values, 1, constant_values=False)\n        interior = values.copy()\n        distance = np.zeros(values.shape, dtype=float)\n        for step in range(1, max(values.shape) + 1):\n            if not np.any(interior):\n                break\n            distance[interior] = step\n            neighbors = (\n                padded[:-2, 1:-1]\n                & padded[2:, 1:-1]\n                & padded[1:-1, :-2]\n                & padded[1:-1, 2:]\n            )\n            interior &= neighbors\n            padded = np.pad(interior, 1, constant_values=False)\n        return distance\n    return np.asarray(distance_transform_edt(values), dtype=float)\n''',
    '''def _mask_boundary_distance(mask: np.ndarray) -> np.ndarray:\n    """Return the canonical border-aware Euclidean interior distance."""\n\n    return interior_mask_distance(mask)\n''',
)

Path("tests/test_mask_distance_and_multiview_geometry_regressions.py").write_text(
    '''import numpy as np

from bayesian_phystwin.mask_distance import (
    _interior_mask_distance_fallback,
    interior_mask_distance,
)
from bayesian_phystwin.phystwin_cotracker3_cues import (
    project_world_points,
    triangulate_multiview_tracks,
)
from bayesian_phystwin.phystwin_motioncrafter_assimilation import (
    _mask_boundary_distance,
)


def test_boundary_distance_treats_image_exterior_as_background() -> None:
    mask = np.ones((3, 4), dtype=bool)
    expected = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    np.testing.assert_allclose(interior_mask_distance(mask), expected)
    np.testing.assert_allclose(_mask_boundary_distance(mask), expected)


def test_numpy_boundary_fallback_is_exact_euclidean() -> None:
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False
    expected = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0), 1.0],
            [1.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0), 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    fallback = _interior_mask_distance_fallback(mask)
    np.testing.assert_allclose(fallback, expected)
    np.testing.assert_allclose(interior_mask_distance(mask), fallback)


def test_triangulation_rejects_behind_camera_support() -> None:
    intrinsics = np.repeat(np.eye(3)[None], 2, axis=0)
    camera_to_world = np.repeat(np.eye(4)[None], 2, axis=0)
    camera_to_world[1, 0, 3] = 1.0
    camera_to_world[1, 2, 3] = 1.0
    point = np.array([[0.0, 0.0, 0.5]])
    tracks = np.empty((2, 1, 1, 2), dtype=float)
    depths = []
    for camera in range(2):
        tracks[camera, 0], depth = project_world_points(
            point, intrinsics[camera], camera_to_world[camera]
        )
        depths.append(float(depth[0]))

    assert depths[0] > 0.0
    assert depths[1] < 0.0
    reconstructed, error, count = triangulate_multiview_tracks(
        tracks,
        np.ones((2, 1, 1), dtype=bool),
        np.ones((2, 1, 1), dtype=float),
        intrinsics,
        camera_to_world,
    )

    np.testing.assert_array_equal(count, [[1]])
    assert np.all(np.isnan(reconstructed[0, 0]))
    assert np.isnan(error[0, 0])
''',
    encoding="utf-8",
)
