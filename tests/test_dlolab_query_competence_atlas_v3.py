from __future__ import annotations

import hashlib
from pathlib import Path

from bayesian_phystwin.query_competence_atlas_v2 import (
    load_query_competence_atlas,
)
from scripts.build_dlolab_query_competence_atlas_v3 import build_atlas

ROOT = Path(__file__).resolve().parents[1]
ATLAS_V2 = ROOT / "results/source/dlolab_query_competence_atlas_v2/atlas.json"
ATLAS_V3 = ROOT / "results/source/dlolab_query_competence_atlas_v3/atlas.json"


def _by_task():
    atlas = load_query_competence_atlas(ATLAS_V3)
    return atlas, {entry.query_scope.metadata["task"]: entry for entry in atlas.entries}


def test_committed_v3_atlas_is_exact_builder_output() -> None:
    committed = load_query_competence_atlas(ATLAS_V3)
    rebuilt = build_atlas()
    assert committed.to_record() == rebuilt.to_record()
    assert committed.metadata["atlas_release"] == 3
    assert committed.metadata["prior_atlas_v2_id"] == (
        "69bc63be221614750496fa1437fde462ad30f80dc0d37adb4a3c56638539252c"
    )


def test_v3_preserves_every_v2_entry_byte_semantically() -> None:
    prior = load_query_competence_atlas(ATLAS_V2)
    current = load_query_competence_atlas(ATLAS_V3)
    current_by_id = {entry.query_scope.query_id: entry for entry in current.entries}
    for entry in prior.entries:
        assert current_by_id[entry.query_scope.query_id].to_record() == entry.to_record()


def test_separation_stops_at_native_qualification_with_exact_fallback() -> None:
    atlas, tasks = _by_task()
    assert set(tasks) == {"wrapping", "slingshot", "coiling", "separation"}
    separation = tasks["separation"]
    assert separation.decision == "rejected"
    assert separation.first_failed_stage == "native_qualification"
    assert separation.furthest_evaluated_stage == "native_qualification"
    assert separation.action_headroom == "not_evaluated"
    assert separation.source_transfer == "not_evaluated"
    assert separation.prospective_risk == "not_evaluated"
    assert separation.independent_group_count == 1
    assert separation.exact_fallback_retained
    assert separation.metadata["attachment_distance_m"] > (
        separation.metadata["attachment_threshold_m"]
    )
    assert not separation.metadata["backend_wide_conclusion"]
    assert len(atlas.certified_query_ids) == 1
    assert len(atlas.rejected_query_ids) == 3


def test_atlas_v3_file_digest_is_frozen() -> None:
    assert hashlib.sha256(ATLAS_V3.read_bytes()).hexdigest() == (
        "4b4fddb146f13688dbca7ce40d8cd44feeb04616f8a36626f8975d43d5b1e07b"
    )
    assert load_query_competence_atlas(ATLAS_V3).artifact_id == (
        "b81af983d4f52f4673f8c9d43a45183ad78f49a93d727ce47520481fa5dfbe35"
    )
