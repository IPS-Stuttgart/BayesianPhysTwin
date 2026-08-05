import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (
    barrier_sha256,
    canonical_payload_sha256,
    prediction_seal_sha256,
)
from bayesian_phystwin.pokeflex_fresh12_staging import stage_manifest_sha256

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "sota" / "pokeflex_action_robust_fresh2_v5"
SUMMARY_FILE_SHA256 = "06e872cd68753bea003ea2df41baf89d715d5a8d2ead55a8df9f76d68300bdd4"


def _load(name: str) -> dict[str, object]:
    return json.loads((RESULT / name).read_text(encoding="utf-8"))


def test_final_two_result_preserves_the_failed_advancement_gate() -> None:
    summary = _load("summary.json")
    aggregate = summary["aggregate"]

    assert summary["decision"]["beats_released_checkpoint"] is True
    assert summary["decision"]["beats_global_scale"] is False
    assert summary["decision"]["all_preregistered_gates_passed"] is False
    assert summary["decision"]["retuning_from_final_two_outcomes_authorized"] is False
    assert aggregate["checkpoint_pairing"]["win_count"] == 2
    assert aggregate["global_scale_advancement"]["win_count"] == 1
    assert aggregate["global_scale_advancement"][
        "minimum_per_object_relative_improvement"
    ] == pytest.approx(-0.0004598515706082617)


def test_final_two_barrier_and_seals_are_canonical_and_precede_outcomes() -> None:
    barrier = _load("prediction_barrier.json")
    target = _load("target_result.json")

    assert barrier["prediction_count"] == 2
    assert barrier["barrier_sha256"] == barrier_sha256(barrier)
    assert target["barrier_sha256"] == barrier["barrier_sha256"]
    assert target["target_meshes_opened_after_complete_barrier"] is True
    for take_id in ("Pillow_T4", "PlushDice_T3"):
        seal = _load(f"prediction_seals/{take_id}/seal.json")
        assert seal["seal_sha256"] == prediction_seal_sha256(seal)
        assert seal["future_mesh_read"] is False
        assert seal["future_mesh_read_count"] == 0
        assert seal["implementation_revision"] == barrier["implementation_revision"]


def test_final_two_stage_manifests_bind_archives_without_decoding() -> None:
    expected = {
        "Pillow_T4": (
            "1d5cf1c344f0515faf1f3e1dd33d3bb995c9502689bfbab15e6b5c3142fe7049"
        ),
        "PlushDice_T3": (
            "3dfb9b174a5c306026d0e532ad4c1c9c6ebb4be876998b3b1e3a42c504bfc8a0"
        ),
    }
    for take_id, archive_sha256 in expected.items():
        stage = _load(f"stage_manifests/{take_id}.source_stage_manifest.json")
        assert stage["archive_sha256"] == archive_sha256
        assert stage["stage_manifest_sha256"] == stage_manifest_sha256(stage)
        assert stage["target_mesh_geometry_decoded"] is False
        assert stage["outcome_metric_computed"] is False


def test_final_two_summary_is_canonical_and_byte_bound() -> None:
    path = RESULT / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))

    assert summary["summary_sha256"] == canonical_payload_sha256(
        summary,
        digest_field="summary_sha256",
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SUMMARY_FILE_SHA256
    for relative, digest in summary["files"].items():
        assert hashlib.sha256((RESULT / relative).read_bytes()).hexdigest() == digest
