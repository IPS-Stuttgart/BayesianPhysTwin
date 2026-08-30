from __future__ import annotations

import hashlib
from pathlib import Path

from bayesian_phystwin.query_competence_atlas_v2 import (
    load_query_competence_atlas,
)
from scripts.build_dlolab_query_competence_atlas_v4 import build_atlas

ROOT = Path(__file__).resolve().parents[1]
ATLAS_V3 = ROOT / "results/source/dlolab_query_competence_atlas_v3/atlas.json"
ATLAS_V4 = ROOT / "results/source/dlolab_query_competence_atlas_v4/atlas.json"


def _by_task():
    atlas = load_query_competence_atlas(ATLAS_V4)
    return atlas, {entry.query_scope.metadata["task"]: entry for entry in atlas.entries}


def test_committed_v4_atlas_is_exact_builder_output() -> None:
    committed = load_query_competence_atlas(ATLAS_V4)
    rebuilt = build_atlas()
    assert committed.to_record() == rebuilt.to_record()
    assert committed.metadata["atlas_release"] == 4
    assert committed.metadata["prior_atlas_v3_id"] == (
        "b81af983d4f52f4673f8c9d43a45183ad78f49a93d727ce47520481fa5dfbe35"
    )


def test_v4_preserves_every_v3_entry_byte_semantically() -> None:
    prior = load_query_competence_atlas(ATLAS_V3)
    current = load_query_competence_atlas(ATLAS_V4)
    current_by_id = {entry.query_scope.query_id: entry for entry in current.entries}
    for entry in prior.entries:
        assert (
            current_by_id[entry.query_scope.query_id].to_record() == entry.to_record()
        )


def test_unknotting_stops_at_native_qualification_with_exact_fallback() -> None:
    atlas, tasks = _by_task()
    assert set(tasks) == {
        "wrapping",
        "slingshot",
        "coiling",
        "separation",
        "unknotting",
    }
    unknotting = tasks["unknotting"]
    assert unknotting.decision == "rejected"
    assert unknotting.first_failed_stage == "native_qualification"
    assert unknotting.furthest_evaluated_stage == "native_qualification"
    assert unknotting.action_headroom == "not_evaluated"
    assert unknotting.source_transfer == "not_evaluated"
    assert unknotting.prospective_risk == "not_evaluated"
    assert unknotting.independent_group_count == 1
    assert unknotting.exact_fallback_retained
    assert (
        unknotting.metadata["reported_maximum_segment_relative_error"]
        > (unknotting.metadata["segment_relative_error_threshold"])
    )
    assert (
        unknotting.metadata["independent_final_segment_relative_error"]
        > (unknotting.metadata["segment_relative_error_threshold"])
    )
    assert not unknotting.metadata["backend_wide_conclusion"]
    assert len(atlas.certified_query_ids) == 1
    assert len(atlas.rejected_query_ids) == 4


def test_atlas_v4_file_digest_is_frozen() -> None:
    assert hashlib.sha256(ATLAS_V4.read_bytes()).hexdigest() == (
        "45890333ac292c0cd2bb5620b1e2bb572e297bddb923d5e570a4cb098adfd94b"
    )
    assert load_query_competence_atlas(ATLAS_V4).artifact_id == (
        "842941a296a055c78d17278e671de796a8f413bcb7ea30fb3d4ef4b232c460c2"
    )
