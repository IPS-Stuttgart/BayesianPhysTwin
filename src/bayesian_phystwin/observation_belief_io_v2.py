"""Bounded, atomic I/O for portable observation-belief artifacts.

The underlying ``ObservationBeliefV1`` payload and content identity are unchanged.
This additive execution boundary preflights ZIP and NPY containers before NumPy
can allocate their arrays and publishes through same-filesystem atomic replace.
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from ._observation_belief_io_v2 import (
    _ARRAY_DTYPES,
    _DESCRIPTOR_FIELDS,
    ObservationBeliefIOLimitsV2,
    ordinary_input_stream,
    preflight_archive,
    snapshot_unchanged,
    strict_descriptor_json,
    validate_sha256,
)
from .observation_belief import (
    OBSERVATION_BELIEF_SCHEMA,
    OBSERVATION_BELIEF_VERSION,
    ObservationBeliefV1,
)

OBSERVATION_BELIEF_IO_V2_SCHEMA: Final = "bayesian-phystwin.observation-belief-io"
OBSERVATION_BELIEF_IO_V2_VERSION: Final = 2
OBSERVATION_BELIEF_IO_V2_SEMANTICS: Final = (
    "bounded-strict-npz-read-and-atomic-replace-v2"
)
OBSERVATION_BELIEF_IO_V2_CLAIM_BOUNDARY: Final = (
    "This I/O boundary validates container structure, resource budgets, exact "
    "array dtypes, descriptor identity, and atomic publication. It does not "
    "establish observation accuracy, covariance calibration, physical-state "
    "identifiability, downstream benefit, deployment safety, or state of the art."
)


def load_observation_belief_bounded_v2(
    path: str | Path,
    *,
    limits: ObservationBeliefIOLimitsV2 | None = None,
) -> ObservationBeliefV1:
    """Load one strict observation belief after bounded container preflight."""

    settings = limits or ObservationBeliefIOLimitsV2()
    if not isinstance(settings, ObservationBeliefIOLimitsV2):
        raise TypeError("limits must be an ObservationBeliefIOLimitsV2")
    stream, before = ordinary_input_stream(Path(path))
    with stream:
        if before.st_size < 1 or before.st_size > settings.maximum_archive_bytes:
            raise ValueError("observation artifact exceeds the archive byte budget")
        headers = preflight_archive(stream, limits=settings)
        stream.seek(0)
        try:
            with np.load(stream, allow_pickle=False) as archive:
                descriptor_member = np.asarray(archive["descriptor_json"])
                if descriptor_member.shape != ():
                    raise ValueError("descriptor_json must be a scalar array")
                raw_descriptor = descriptor_member.item()
                if isinstance(raw_descriptor, bytes):
                    try:
                        raw_descriptor = raw_descriptor.decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise ValueError(
                            "descriptor_json is not valid UTF-8"
                        ) from error
                if type(raw_descriptor) is not str:
                    raise ValueError("descriptor_json must contain a string")
                if (
                    len(raw_descriptor.encode("utf-8"))
                    > settings.maximum_descriptor_bytes
                ):
                    raise ValueError("descriptor_json exceeds its UTF-8 byte budget")
                descriptor = strict_descriptor_json(raw_descriptor)
                if set(descriptor) != _DESCRIPTOR_FIELDS:
                    raise ValueError("observation descriptor fields changed")
                if descriptor.get("schema_name") != OBSERVATION_BELIEF_SCHEMA:
                    raise ValueError("unsupported observation-belief schema")
                version = descriptor.get("schema_version")
                if isinstance(version, bool) or not isinstance(version, int):
                    raise ValueError("observation-belief schema_version changed type")
                if version != OBSERVATION_BELIEF_VERSION:
                    raise ValueError("unsupported observation-belief version")
                arrays: dict[str, np.ndarray] = {}
                for name, expected_dtype in _ARRAY_DTYPES.items():
                    values = np.asarray(archive[name])
                    if values.dtype != expected_dtype:
                        raise ValueError(
                            f"{name} must have exact dtype {expected_dtype.name}"
                        )
                    if values.shape != headers[f"{name}.npy"].shape:
                        raise ValueError(f"{name} shape changed after preflight")
                    arrays[name] = values
        except (
            OSError,
            EOFError,
            KeyError,
            TypeError,
            ValueError,
            zipfile.BadZipFile,
        ) as error:
            if isinstance(error, ValueError):
                raise
            raise ValueError("cannot decode observation artifact") from error
        after = os.fstat(stream.fileno())
        if not snapshot_unchanged(before, after):
            raise ValueError("observation artifact changed while it was being read")

    belief = ObservationBeliefV1(
        case_id=cast(str, descriptor["case_id"]),
        stream_id=cast(str, descriptor["stream_id"]),
        causal_frame_stop=cast(int, descriptor["causal_frame_stop"]),
        view_names=tuple(cast(Sequence[str], descriptor["view_names"])),
        window_names=tuple(cast(Sequence[str], descriptor["window_names"])),
        factor_names=tuple(cast(Sequence[str], descriptor["factor_names"])),
        source_repository=cast(str, descriptor["source_repository"]),
        source_revision=cast(str, descriptor["source_revision"]),
        source_artifact_sha256=cast(str, descriptor["source_artifact_sha256"]),
        metadata=cast(Mapping[str, Any], descriptor["metadata"]),
        **arrays,
    )
    expected = validate_sha256(descriptor["artifact_id"], name="artifact_id")
    if belief.artifact_id != expected:
        raise ValueError("observation artifact digest does not match its payload")
    return belief


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            return
    finally:
        os.close(descriptor)


def save_observation_belief_atomic_v2(
    path: str | Path,
    belief: ObservationBeliefV1,
    *,
    limits: ObservationBeliefIOLimitsV2 | None = None,
) -> None:
    """Validate and atomically replace one observation-belief NPZ archive."""

    if not isinstance(belief, ObservationBeliefV1):
        raise TypeError("belief must be an ObservationBeliefV1")
    settings = limits or ObservationBeliefIOLimitsV2()
    if not isinstance(settings, ObservationBeliefIOLimitsV2):
        raise TypeError("limits must be an ObservationBeliefIOLimitsV2")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.is_dir():
        raise ValueError("observation artifact target must not be a directory")

    descriptor = belief._descriptor()
    descriptor["artifact_id"] = belief.artifact_id
    archive_payload: dict[str, Any] = {
        "descriptor_json": np.asarray(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    }
    archive_payload.update(belief._arrays())

    descriptor_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".partial",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor_fd, "w+b") as stream:
            np.savez_compressed(stream, **archive_payload)
            stream.flush()
            os.fsync(stream.fileno())
        verified = load_observation_belief_bounded_v2(temporary, limits=settings)
        if verified.artifact_id != belief.artifact_id:
            raise ValueError("serialized observation belief changed identity")
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "OBSERVATION_BELIEF_IO_V2_CLAIM_BOUNDARY",
    "OBSERVATION_BELIEF_IO_V2_SCHEMA",
    "OBSERVATION_BELIEF_IO_V2_SEMANTICS",
    "OBSERVATION_BELIEF_IO_V2_VERSION",
    "ObservationBeliefIOLimitsV2",
    "load_observation_belief_bounded_v2",
    "save_observation_belief_atomic_v2",
]
