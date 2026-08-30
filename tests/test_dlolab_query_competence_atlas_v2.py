from __future__ import annotations

import hashlib
from pathlib import Path

from bayesian_phystwin.query_competence_atlas_v2 import (
    load_query_competence_atlas,
)
from scripts.build_dlolab_query_competence_atlas_v2 import build_atlas

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "results/source/dlolab_query_competence_atlas_v2/atlas.json"


def _by_task():
    atlas = load_query_competence_atlas(ATLAS)
    return atlas, {entry.query_scope.metadata["task"]: entry for entry in atlas.entries}


def test_committed_atlas_is_exact_builder_output() -> None:
    committed = load_query_competence_atlas(ATLAS)
    rebuilt = build_atlas()
    assert committed.artifact_id == (
        "69bc63be221614750496fa1437fde462ad30f80dc0d37adb4a3c56638539252c"
    )
    assert committed.to_record() == rebuilt.to_record()
    assert hashlib.sha256(ATLAS.read_bytes()).hexdigest() == (
        "6438b04e766c04e8f25b4a42123655724b5cd32f3b4c5bc88ac15f10ee0b6fa6"
    )


def test_atlas_distinguishes_three_stage_outcomes_without_pooling() -> None:
    atlas, tasks = _by_task()
    assert set(tasks) == {"wrapping", "slingshot", "coiling"}

    wrapping = tasks["wrapping"]
    assert wrapping.decision == "certified"
    assert wrapping.first_failed_stage is None
    assert wrapping.furthest_evaluated_stage == "prospective_risk"

    slingshot = tasks["slingshot"]
    assert slingshot.decision == "rejected"
    assert slingshot.first_failed_stage == "action_headroom"
    assert slingshot.prospective_risk == "failed"

    coiling = tasks["coiling"]
    assert coiling.decision == "rejected"
    assert coiling.native_qualification == "passed"
    assert coiling.action_headroom == "failed"
    assert coiling.source_transfer == "failed"
    assert coiling.prospective_risk == "not_evaluated"
    assert coiling.furthest_evaluated_stage == "source_transfer"
    assert coiling.independent_group_count == 12

    assert atlas.certified_query_ids == (wrapping.query_scope.query_id,)
    assert set(atlas.rejected_query_ids) == {
        slingshot.query_scope.query_id,
        coiling.query_scope.query_id,
    }
    assert not atlas.to_record()["backend_wide_competence_claim"]
    assert not atlas.metadata["independent_human_review"]


def test_bound_coiling_metrics_remain_source_only() -> None:
    _, tasks = _by_task()
    coiling = tasks["coiling"]
    assert coiling.evidence_role == "source_screen"
    assert coiling.metadata["oracle_headroom"] == 0.0016061967989121628
    assert coiling.metadata["crossfit_guarded_mean_gain"] == (-0.0004303340638860791)
    assert coiling.metadata["mean_observation_draw_harm_probability"] == (
        0.08333333333333333
    )
    assert coiling.exact_fallback_retained
    assert coiling.protocol_frozen_before_outcomes
    assert not coiling.outcomes_used_for_selection
    assert not coiling.protected_data_read
