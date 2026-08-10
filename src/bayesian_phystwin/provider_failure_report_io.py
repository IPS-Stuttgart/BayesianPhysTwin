"""Strict input and atomic publication for provider-failure reports."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .strict_json_report_io import (
    DEFAULT_MAXIMUM_INPUT_BYTES,
    canonical_json_sha256,
    load_strict_json_mapping,
    publish_json_report,
)


def load_provider_failure_input(
    path: str | Path,
    *,
    maximum_input_bytes: int = DEFAULT_MAXIMUM_INPUT_BYTES,
) -> tuple[Mapping[str, object], dict[str, object]]:
    """Read one unchanged ordinary UTF-8 provider-failure JSON object."""

    return load_strict_json_mapping(
        path,
        artifact_label="provider-failure",
        maximum_input_bytes=maximum_input_bytes,
    )


def publish_provider_failure_report(
    path: str | Path,
    report: Mapping[str, Any],
    *,
    input_artifact: Mapping[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Publish one verified provider-failure report atomically."""

    return publish_json_report(
        path,
        report,
        input_artifact=input_artifact,
        overwrite=overwrite,
    )


__all__ = [
    "DEFAULT_MAXIMUM_INPUT_BYTES",
    "canonical_json_sha256",
    "load_provider_failure_input",
    "publish_provider_failure_report",
]
