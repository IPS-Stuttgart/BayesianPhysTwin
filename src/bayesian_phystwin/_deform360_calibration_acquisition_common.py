# ruff: noqa: E402, F401
"""Calibration-only Deform360 payload acquisition and source-preparation records.

Stage 0 selects exact object/episode units from names and ``metadata.json`` only.
The visual-provider lock then freezes the executable Prob4D/MotionCrafter
producer before any selected raw payload is opened. This module defines the next
boundary: open and prepare only the ten calibration units, keep all confirmation
payloads closed, retain technical failures without replacement, and publish a
calibration-only evidence ledger.

The contracts in this module do not download data or score outcomes. The runtime
script in ``scripts/science/run_deform360_calibration_acquisition.py`` performs
the exact selective download and pinned official source preparation.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import (
    canonical_sorted_strings,
    content_id,
    exact_revision,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
    write_atomic_json,
)
from .deform360_calibration_bundle import (
    DEFORM360_CALIBRATION_OBJECTS_PER_STRATUM,
    DEFORM360_CALIBRATION_PROTOCOL_ID,
    DEFORM360_CONFIRMATION_OBJECTS_PER_STRATUM,
    DEFORM360_DATASET_REPOSITORY,
    DEFORM360_PROCESSING_REPOSITORY,
    Deform360CohortUnitV1,
)
from .deform360_calibration_execution import (
    DEFORM360_CALIBRATION_LEDGER_CASE_ID,
    load_deform360_stage0_selection,
)
from .deform360_visual_provider_lock import (
    load_deform360_visual_provider_lock,
)
from .evidence_use_ledger import EvidenceUseLedgerV1, EvidenceUseV1

DEFORM360_CALIBRATION_ACQUISITION_PLAN_SCHEMA = (
    "bayesian-phystwin.deform360-calibration-acquisition-plan"
)
DEFORM360_CALIBRATION_ACQUISITION_CASE_SCHEMA = (
    "bayesian-phystwin.deform360-calibration-acquisition-case"
)
DEFORM360_CALIBRATION_ACQUISITION_RESULT_SCHEMA = (
    "bayesian-phystwin.deform360-calibration-acquisition-result"
)
DEFORM360_CALIBRATION_ACQUISITION_VERSION = 1
DEFORM360_CALIBRATION_ACQUISITION_SEMANTICS = (
    "calibration-payload-open-confirmation-closed-source-preparation-v1"
)
DEFORM360_CALIBRATION_ACQUISITION_CLAIM_BOUNDARY = (
    "Calibration source acquisition and information-order evidence only. A valid "
    "result does not establish provider competence, physical-query accuracy, "
    "tactile benefit, predictive calibration, material identification, Causal4D "
    "benefit, safety, or state of the art."
)

Deform360CalibrationAcquisitionStatus = Literal[
    "prepared",
    "technical_failure",
]

_CAMERA_RE = re.compile(r"^brics-odroid-\d+_cam\d+$")
_TACTILE_RE = re.compile(r"^brics-odroid_tactile[^/]+$")
_AUDIO_SUFFIXES = frozenset({".wav", ".flac"})
_CALIBRATION_FILES = (
    "calibration_refined/dist.npy",
    "calibration_refined/extrinsics.npy",
    "calibration_refined/intrinsics.npy",
)

_PLAN_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "plan_id",
        "protocol_id",
        "selection_artifact_sha256",
        "content_selection_sha256",
        "visual_provider_lock_id",
        "dataset_repository",
        "dataset_revision",
        "processing_repository",
        "processing_revision",
        "implementation_revision",
        "calibration_units",
        "forbidden_confirmation_object_ids",
        "camera_policy",
        "tactile_policy",
        "audio_policy",
        "calibration_payloads_opened",
        "confirmation_payloads_opened",
        "target_outcomes_used",
        "metadata",
        "claim_boundary",
    }
)
_CASE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "case_id",
        "plan_id",
        "object_id",
        "episode_id",
        "stratum",
        "status",
        "aligned_frame_count",
        "camera_count",
        "tactile_sensor_count",
        "bimanual",
        "raw_factor_artifacts",
        "output_artifacts",
        "failure_stage",
        "failure_type",
        "failure_message_sha256",
        "calibration_payloads_opened",
        "confirmation_payloads_opened",
        "target_outcomes_used",
        "metadata",
        "claim_boundary",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "result_id",
        "protocol_id",
        "plan_id",
        "implementation_revision",
        "case_ids",
        "evidence_use_ledger_id",
        "prepared_object_count",
        "technical_failure_count",
        "status",
        "source_artifacts",
        "calibration_payloads_opened",
        "confirmation_payloads_opened",
        "target_outcomes_used",
        "replacement_allowed",
        "metadata",
        "claim_boundary",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of one ordinary file without following symlinks."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"source must be an ordinary file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _calibration_units(
    values: Sequence[Deform360CohortUnitV1],
) -> tuple[Deform360CohortUnitV1, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("calibration_units must be a sequence")
    units = tuple(values)
    if any(not isinstance(unit, Deform360CohortUnitV1) for unit in units):
        raise ValueError("calibration_units must contain Deform360CohortUnitV1")
    units = tuple(sorted(units, key=lambda unit: (unit.stratum, unit.object_id)))
    object_ids = [unit.object_id for unit in units]
    _require(len(set(object_ids)) == len(object_ids), "calibration object repeated")
    for stratum in ("sheet", "volumetric"):
        count = sum(unit.stratum == stratum for unit in units)
        _require(
            count == DEFORM360_CALIBRATION_OBJECTS_PER_STRATUM,
            f"calibration stratum {stratum} must contain exactly "
            f"{DEFORM360_CALIBRATION_OBJECTS_PER_STRATUM} objects",
        )
    return units


def _unit_from_record(value: object, *, name: str) -> Deform360CohortUnitV1:
    return Deform360CohortUnitV1.from_mapping(value, name=name)



__all__ = [name for name in globals() if not name.startswith("__")]
