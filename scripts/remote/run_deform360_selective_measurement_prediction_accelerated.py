#!/usr/bin/env python3
"""Run the frozen selective study with an exact camera-search accelerator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import bayesian_phystwin.deform360_raw_camera_observation as raw_camera
from bayesian_phystwin.deform360_exact_camera_subset import (
    FROZEN_RAW_CAMERA_BUILDER_SHA256,
    frozen_builder_path,
    select_frame_zero_observation_plan_exact_accelerated,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_runner(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "deform360_selective_measurement_prediction_frozen_runner",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    builder = frozen_builder_path()
    builder_sha256 = _sha256(builder)
    if builder_sha256 != FROZEN_RAW_CAMERA_BUILDER_SHA256:
        raise ValueError(
            "camera accelerator base differs from the frozen raw-camera builder"
        )
    raw_camera.select_frame_zero_observation_plan = (
        select_frame_zero_observation_plan_exact_accelerated
    )
    frozen_runner = Path(__file__).with_name(
        "run_deform360_selective_measurement_prediction.py"
    )
    print(
        json.dumps(
            {
                "artifact_kind": "Deform360ExactCameraSubsetOperationalAmendment",
                "frozen_raw_camera_builder_sha256": builder_sha256,
                "accelerator_source_sha256": _sha256(
                    Path(
                        select_frame_zero_observation_plan_exact_accelerated.__code__.co_filename
                    )
                ),
                "wrapper_source_sha256": _sha256(Path(__file__).resolve()),
                "scientific_score_changed": False,
                "target_data_read": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return int(_load_frozen_runner(frozen_runner).main())


if __name__ == "__main__":
    raise SystemExit(main())
