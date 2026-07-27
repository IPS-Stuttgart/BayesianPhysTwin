"""Versioned Bayesian-PhysTwin surface for Causal4D public-data studies."""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any

from .contracts.provider import (
    installed_distribution_revision,
    installed_distribution_version,
)

CAUSAL4D_PUBLIC_PROVIDER_API_VERSION = 1
CAUSAL4D_PUBLIC_PROVIDER_PACKAGE_VERSION = "0.4.0"
CAUSAL4D_PUBLIC_PROVIDER_CAPABILITIES = (
    "deform360_selective_virtual_sensing",
    "phystwin_track_objective",
)

_LAZY_PUBLIC_EXPORTS: dict[str, tuple[str, str]] = {
    "ARM_TO_ARCHIVE_KEY": (
        "deform360_selective_virtual_sensing_evaluation",
        "ARM_TO_ARCHIVE_KEY",
    ),
    "MANIFEST_FILENAME": ("deform360_raw_camera_observation", "MANIFEST_FILENAME"),
    "MEASUREMENT_FILENAME": (
        "deform360_raw_camera_observation",
        "MEASUREMENT_FILENAME",
    ),
    "PROTOCOL_ID": ("deform360_selective_virtual_sensing_protocol", "PROTOCOL_ID"),
    "SCORED_FRAMES": (
        "deform360_selective_virtual_sensing_evaluation",
        "SCORED_FRAMES",
    ),
    "VIRTUAL_SENSING_ARCHIVE_FILENAME": (
        "deform360_selective_virtual_sensing_artifacts",
        "VIRTUAL_SENSING_ARCHIVE_FILENAME",
    ),
    "VIRTUAL_SENSING_REPORT_FILENAME": (
        "deform360_selective_virtual_sensing_artifacts",
        "VIRTUAL_SENSING_REPORT_FILENAME",
    ),
    "VIRTUAL_SENSING_SEAL_FILENAME": (
        "deform360_selective_virtual_sensing_artifacts",
        "VIRTUAL_SENSING_SEAL_FILENAME",
    ),
    "build_phystwin_track_objective": (
        "phystwin_refit",
        "build_phystwin_track_objective",
    ),
    "closure_confidence": (
        "deform360_selective_virtual_sensing_staging",
        "closure_confidence",
    ),
    "dynamic_window_source_case": (
        "deform360_selective_virtual_sensing_staging",
        "dynamic_window_source_case",
    ),
    "end_effector_origins": (
        "deform360_selective_virtual_sensing_staging",
        "end_effector_origins",
    ),
    "load_selective_virtual_sensing_protocol": (
        "deform360_selective_virtual_sensing_protocol",
        "load_selective_virtual_sensing_protocol",
    ),
    "score_selective_virtual_sensing_arrays": (
        "deform360_selective_virtual_sensing_evaluation",
        "score_selective_virtual_sensing_arrays",
    ),
    "select_action_only_window": (
        "deform360_selective_virtual_sensing_staging",
        "select_action_only_window",
    ),
    "select_translation_contact_window": (
        "deform360_selective_virtual_sensing_staging",
        "select_translation_contact_window",
    ),
    "selective_case_records": (
        "deform360_selective_virtual_sensing_artifacts",
        "selective_case_records",
    ),
    "validate_selective_prediction_seal": (
        "deform360_selective_virtual_sensing_artifacts",
        "validate_selective_prediction_seal",
    ),
}


def causal4d_public_provider_manifest(
    *, provider_revision: str | None = None
) -> dict[str, object]:
    """Return the public-data provider descriptor consumed by Causal4D."""

    revision = (
        provider_revision
        or os.environ.get("BAYESIAN_PHYSTWIN_REVISION")
        or installed_distribution_revision("bayesian-phystwin")
        or "unversioned-install"
    )
    return {
        "provider_name": "bayesian-phystwin",
        "provider_version": installed_distribution_version(
            "bayesian-phystwin",
            fallback=CAUSAL4D_PUBLIC_PROVIDER_PACKAGE_VERSION,
        ),
        "provider_revision": revision,
        "schema_version": CAUSAL4D_PUBLIC_PROVIDER_API_VERSION,
        "capabilities": list(CAUSAL4D_PUBLIC_PROVIDER_CAPABILITIES),
        "metadata": {
            "provider_api": "bayesian_phystwin.causal4d_public_provider_v1",
            "provider_api_version": CAUSAL4D_PUBLIC_PROVIDER_API_VERSION,
        },
    }


def __getattr__(name: str) -> Any:
    target = _LAZY_PUBLIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(f"bayesian_phystwin.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_PUBLIC_EXPORTS))


__all__ = [
    "CAUSAL4D_PUBLIC_PROVIDER_API_VERSION",
    "CAUSAL4D_PUBLIC_PROVIDER_CAPABILITIES",
    "CAUSAL4D_PUBLIC_PROVIDER_PACKAGE_VERSION",
    "causal4d_public_provider_manifest",
    *sorted(_LAZY_PUBLIC_EXPORTS),
]
