"""Digest-bound loading for trusted legacy PhysTwin pickle artifacts.

Pickle deserialization can execute code. This module does not sandbox pickle;
it copies the source into a private snapshot while hashing it, verifies that
snapshot against an externally trusted SHA-256 digest, and deserializes exactly
the verified bytes before validating the caller's top-level representation.
New Bayesian-PhysTwin artifacts should continue to use JSON/NPZ contracts.
"""

from __future__ import annotations

import hashlib
import hmac
import pickle
import tempfile
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

LegacyPhysTwinArtifactKind = Literal["mapping", "sequence", "ndarray"]
_VALID_ARTIFACT_KINDS = frozenset({"mapping", "sequence", "ndarray"})
_COPY_CHUNK_SIZE_BYTES = 1024 * 1024
_SNAPSHOT_MEMORY_LIMIT_BYTES = 16 * 1024 * 1024


def _validate_sha256(value: str, *, name: str) -> str:
    normalized = str(value)
    if (
        normalized != normalized.lower()
        or len(normalized) != 64
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _sha256_file(
    path: Path,
    *,
    chunk_size: int = _COPY_CHUNK_SIZE_BYTES,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _load_verified_pickle_snapshot(
    source: Path,
    *,
    expected_sha256: str,
) -> Any:
    """Deserialize exactly the bytes whose trusted digest was verified."""

    digest = hashlib.sha256()
    with source.open("rb") as stream, tempfile.SpooledTemporaryFile(
        max_size=_SNAPSHOT_MEMORY_LIMIT_BYTES,
        mode="w+b",
    ) as snapshot:
        while block := stream.read(_COPY_CHUNK_SIZE_BYTES):
            digest.update(block)
            snapshot.write(block)

        actual = digest.hexdigest()
        if not hmac.compare_digest(actual, expected_sha256):
            raise ValueError(
                "legacy PhysTwin artifact SHA-256 mismatch; refusing to deserialize"
            )

        snapshot.seek(0)
        return pickle.load(snapshot)


def load_trusted_legacy_phystwin_pickle(
    path: str | Path,
    *,
    expected_sha256: str,
    artifact_kind: LegacyPhysTwinArtifactKind,
    required_keys: Sequence[str] = (),
) -> Any:
    """Load one hash-locked legacy pickle and validate its top-level contract.

    ``expected_sha256`` must come from an independently trusted manifest or
    protocol lock. A matching digest establishes byte identity, not general
    pickle safety; callers must never accept a digest supplied alongside an
    otherwise untrusted pickle.
    """

    expected = _validate_sha256(expected_sha256, name="expected_sha256")
    kind = str(artifact_kind)
    if kind not in _VALID_ARTIFACT_KINDS:
        raise ValueError(f"unsupported legacy artifact kind: {kind!r}")

    keys = tuple(map(str, required_keys))
    if any(not key for key in keys) or len(keys) != len(set(keys)):
        raise ValueError("required_keys must contain unique nonempty names")
    if keys and kind != "mapping":
        raise ValueError("required_keys are supported only for mapping artifacts")

    value = _load_verified_pickle_snapshot(
        Path(path),
        expected_sha256=expected,
    )

    if kind == "mapping":
        if not isinstance(value, Mapping):
            raise TypeError("legacy PhysTwin artifact must contain a mapping")
        missing = sorted(set(keys) - set(map(str, value.keys())))
        if missing:
            raise ValueError(
                "legacy PhysTwin mapping is missing required keys: "
                + ", ".join(missing)
            )
    elif kind == "sequence":
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(
            value, Sequence
        ):
            raise TypeError("legacy PhysTwin artifact must contain a sequence")
    else:
        numpy_module = import_module("numpy")
        if not isinstance(value, numpy_module.ndarray):
            raise TypeError("legacy PhysTwin artifact must contain a NumPy array")

    return value


__all__ = [
    "LegacyPhysTwinArtifactKind",
    "load_trusted_legacy_phystwin_pickle",
]
