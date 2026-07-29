from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_causal_response_direct_depth_cohort import (
    build_v14_staging_queue,
    validate_v14_staging_queue,
)
from bayesian_phystwin.deform360_fresh_source_download import (
    fresh_source_download_plan,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "sota" / (
    "deform360_causal_response_direct_depth_v14.json"
)
CATALOG = ROOT / "results" / "sota" / "deform360_fresh_source_lock_v1" / (
    "public_object_catalog_2026-07-26.json"
)
METADATA = ROOT / "results" / "sota" / (
    "deform360_dynamic_tapnextpp_provider_v1"
) / "metadata_preflight_v1.json"
EXCLUSION = ROOT / "configs" / "sota" / (
    "deform360_fresh_object_exclusion_v14.json"
)


def _object_hash(object_id: str) -> str:
    return hashlib.sha256(
        b"deform360-fresh-object-exclusion-v1\0"
        + object_id.encode("utf-8")
    ).hexdigest()


def test_repository_artifacts_build_frozen_v14_queue(tmp_path: Path) -> None:
    output = tmp_path / "queue.json"
    queue = build_v14_staging_queue(
        output,
        protocol_path=PROTOCOL,
        catalog_path=CATALOG,
        metadata_preflight_path=METADATA,
        exclusion_path=EXCLUSION,
    )

    assert output.is_file()
    assert queue["queue_contract"]["required_admitted_source_count"] == 12
    assert queue["queue_contract"]["prediction_or_outcome_triggers_replacement"] is False
    assert queue["metadata_dispositions"] == {
        "fresh_catalog_count": 54,
        "accepted_candidate_count": 53,
        "rejected_count": 1,
        "rejected_hash_only": queue["metadata_dispositions"]["rejected_hash_only"],
    }
    assert queue["stratum_counts"] == {
        "sheet": 43,
        "compact": 4,
        "complex": 6,
    }
    assert [row["category"] for row in queue["candidates"][:12]] == [
        "sheet",
        "compact",
        "complex",
    ] * 4
    exclusion_hashes = set(
        json.loads(EXCLUSION.read_text(encoding="utf-8"))["object_hashes"]
    )
    assert not {
        _object_hash(row["object_id"]) for row in queue["candidates"]
    }.intersection(exclusion_hashes)
    assert validate_v14_staging_queue(output) == queue
    assert len(fresh_source_download_plan(output).candidates) == 53


def test_v14_queue_rejects_boundary_tampering(tmp_path: Path) -> None:
    output = tmp_path / "queue.json"
    queue = build_v14_staging_queue(
        output,
        protocol_path=PROTOCOL,
        catalog_path=CATALOG,
        metadata_preflight_path=METADATA,
        exclusion_path=EXCLUSION,
    )
    queue["information_boundary"]["outcome_or_metric_read"] = True

    with pytest.raises(ValueError, match="information boundary"):
        validate_v14_staging_queue(queue)
