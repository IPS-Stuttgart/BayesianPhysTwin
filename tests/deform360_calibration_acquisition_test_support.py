from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_calibration_acquisition import (
    Deform360CalibrationAcquisitionCaseV1,
    Deform360CalibrationAcquisitionPlanV1,
    build_calibration_acquisition_plan,
    build_calibration_acquisition_result,
    build_calibration_evidence_ledger,
    select_calibration_object_paths,
    validate_calibration_acquisition_result,
    validate_calibration_download_root,
)
from bayesian_phystwin.deform360_calibration_bundle import Deform360CohortUnitV1

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "run_deform360_calibration_acquisition.py"
STAGE0 = (
    ROOT
    / "protocols"
    / "locks"
    / "deform360_official_hub_visuotactile_v1_selection.json"
)
PROTOCOL = ROOT / "protocols" / "deform360_official_hub_visuotactile_v1.json"
VISUAL_LOCK = (
    ROOT
    / "protocols"
    / "locks"
    / "deform360_official_hub_visuotactile_v1_visual_provider"
    / "visual-provider-lock.json"
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _units() -> tuple[Deform360CohortUnitV1, ...]:
    result = []
    for stratum, prefix in (("sheet", "sheet"), ("volumetric", "volume")):
        for index in range(5):
            object_id = f"{prefix}-{index}"
            result.append(
                Deform360CohortUnitV1(
                    object_id=object_id,
                    episode_id=index,
                    stratum=stratum,
                    metadata_path=f"raw/{object_id}/metadata.json",
                    metadata_sha256=_digest(f"metadata:{object_id}"),
                )
            )
    return tuple(result)


def _plan() -> Deform360CalibrationAcquisitionPlanV1:
    return Deform360CalibrationAcquisitionPlanV1(
        selection_artifact_sha256=_digest("selection-artifact"),
        content_selection_sha256=_digest("selection-content"),
        visual_provider_lock_id=_digest("visual-provider"),
        dataset_revision="a" * 40,
        processing_revision="b" * 40,
        implementation_revision="c" * 40,
        calibration_units=_units(),
        forbidden_confirmation_object_ids=tuple(
            f"confirmation-{index}" for index in range(12)
        ),
        metadata={"source": "test"},
    )


def _prepared_case(
    plan: Deform360CalibrationAcquisitionPlanV1,
    unit: Deform360CohortUnitV1,
) -> Deform360CalibrationAcquisitionCaseV1:
    return Deform360CalibrationAcquisitionCaseV1(
        plan_id=plan.plan_id,
        object_id=unit.object_id,
        episode_id=unit.episode_id,
        stratum=unit.stratum,
        status="prepared",
        raw_factor_artifacts={unit.metadata_path: unit.metadata_sha256},
        output_artifacts={
            (
                f"processed/{unit.object_id}/"
                f"episode_{unit.episode_id:04d}/alignment.json"
            ): _digest(f"output:{unit.object_id}")
        },
        aligned_frame_count=81 + unit.episode_id,
        camera_count=8,
        tactile_sensor_count=2,
        bimanual=False,
        metadata={"timeline_sha256": _digest(f"timeline:{unit.object_id}")},
    )


def _failed_case(
    plan: Deform360CalibrationAcquisitionPlanV1,
    unit: Deform360CohortUnitV1,
) -> Deform360CalibrationAcquisitionCaseV1:
    return Deform360CalibrationAcquisitionCaseV1(
        plan_id=plan.plan_id,
        object_id=unit.object_id,
        episode_id=unit.episode_id,
        stratum=unit.stratum,
        status="technical_failure",
        raw_factor_artifacts={unit.metadata_path: unit.metadata_sha256},
        output_artifacts={},
        failure_stage="tactile",
        failure_type="FileNotFoundError",
        failure_message_sha256=_digest("missing exact tactile sidecar"),
        metadata={"technical_failure_retained_without_replacement": True},
    )


def _script_module() -> Any:
    specification = importlib.util.spec_from_file_location(
        "deform360_calibration_acquisition_script",
        SCRIPT,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


__all__ = [name for name in globals() if not name.startswith("__")]
