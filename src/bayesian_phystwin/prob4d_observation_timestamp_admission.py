"""Claim-bearing admission for Prob4D observation timestamp lineages.

The portable timestamp loader proves that a sidecar is internally
content-addressed. Claim-bearing use additionally needs an independently known
identity for the raw timestamp source. Otherwise a forged sidecar and a forged
content ID could remain mutually self-consistent.

This module therefore snapshots the exact sidecar bytes, checks the producer's
``source_artifact_sha256`` against an independently supplied digest, binds both
identities into the returned timestamp binding metadata, and rejects replacement
of the sidecar during admission.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._canonical_contracts import plain_json
from ._portable_contracts import sha256_digest
from .observation_belief import ObservationBeliefV1
from .prob4d_observation_timestamps import (
    Prob4DObservationTimestampBindingV1,
    load_prob4d_observation_timestamp_binding,
    load_prob4d_observation_timestamp_lineage,
)

_CLAIM_METADATA_FIELDS = frozenset(
    {
        "prob4d_timestamp_source_sha256",
        "prob4d_timestamp_source_independently_verified",
        "prob4d_timestamp_lineage_artifact_id",
        "prob4d_timestamp_lineage_file_sha256",
    }
)


def _ordinary_snapshot(path: str | Path) -> tuple[Path, str]:
    artifact_path = Path(path)
    if artifact_path.is_symlink():
        raise ValueError("Prob4D timestamp lineage path must not be a symlink")
    try:
        payload = artifact_path.read_bytes()
    except OSError as error:
        raise ValueError("Prob4D timestamp lineage is unreadable") from error
    if not artifact_path.is_file():
        raise ValueError("Prob4D timestamp lineage must be an ordinary file")
    return artifact_path, hashlib.sha256(payload).hexdigest()


def load_claim_bearing_prob4d_observation_timestamp_binding(
    observation: ObservationBeliefV1,
    *,
    timestamp_lineage_path: str | Path,
    expected_timestamp_source_sha256: str,
    bundle_manifest_path: str | Path,
    expected_bundle_manifest_sha256: str,
    row_factor_ids: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> Prob4DObservationTimestampBindingV1:
    """Admit one timestamp sidecar with independent source-byte evidence.

    ``expected_timestamp_source_sha256`` must come from an independently frozen
    source/calibration manifest, not from the timestamp sidecar being admitted.
    The sidecar is hashed before and after the lower-level factor/row binding so
    a concurrent replacement cannot change the admitted evidence silently.
    """

    expected_source = sha256_digest(
        expected_timestamp_source_sha256,
        name="expected_timestamp_source_sha256",
    )
    artifact_path, file_sha_before = _ordinary_snapshot(timestamp_lineage_path)
    lineage = load_prob4d_observation_timestamp_lineage(artifact_path)
    if lineage.source_artifact_sha256 != expected_source:
        raise ValueError(
            "Prob4D timestamp source artifact differs from independent evidence"
        )
    lineage_id = lineage.artifact_id
    if lineage_id is None:
        raise AssertionError("validated Prob4D timestamp lineage lacks an artifact ID")

    caller_metadata = {} if metadata is None else plain_json(metadata)
    if not isinstance(caller_metadata, dict):
        raise ValueError("claim-bearing timestamp metadata must be a mapping")
    overlap = _CLAIM_METADATA_FIELDS.intersection(caller_metadata)
    if overlap:
        raise ValueError(
            "claim-bearing timestamp metadata reserves fields "
            f"{sorted(overlap)}"
        )
    admitted_metadata = {
        **caller_metadata,
        "prob4d_timestamp_source_sha256": expected_source,
        "prob4d_timestamp_source_independently_verified": True,
        "prob4d_timestamp_lineage_artifact_id": lineage_id,
        "prob4d_timestamp_lineage_file_sha256": file_sha_before,
    }
    binding = load_prob4d_observation_timestamp_binding(
        observation,
        timestamp_lineage_path=artifact_path,
        bundle_manifest_path=bundle_manifest_path,
        expected_bundle_manifest_sha256=expected_bundle_manifest_sha256,
        row_factor_ids=row_factor_ids,
        metadata=admitted_metadata,
    )

    _, file_sha_after = _ordinary_snapshot(artifact_path)
    if file_sha_after != file_sha_before:
        raise ValueError("Prob4D timestamp lineage changed during admission")
    if binding.timestamp_lineage_artifact_id != lineage_id:
        raise ValueError("Prob4D timestamp lineage identity changed during admission")
    if dict(binding.metadata) != admitted_metadata:
        raise ValueError("claim-bearing timestamp evidence was not retained exactly")
    return binding


__all__ = ["load_claim_bearing_prob4d_observation_timestamp_binding"]
