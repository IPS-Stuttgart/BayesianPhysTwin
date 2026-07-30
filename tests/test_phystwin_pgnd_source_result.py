import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "sota" / "phystwin_pgnd_source_smoke_v1"
EXPECTED_HASHES = {
    "evaluation.json": (
        "ab7cadd1951943b151e617a2aad80c684ec6977befe39de7440f087524d3c7ab"
    ),
    "prediction_input_summary.json": (
        "719a8813475b61065262eb01532045669e815e4fc2b2ab323bfdff500fa1b768"
    ),
    "prediction_seal.json": (
        "5dd77f354370a489a1d8a73886c5483c8af8c9536f0196b87811a63d9701f75c"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pgnd_source_smoke_result_is_hash_bound_and_failed() -> None:
    for name, expected in EXPECTED_HASHES.items():
        assert _sha256(RESULT_ROOT / name) == expected

    seal = json.loads(
        (RESULT_ROOT / "prediction_seal.json").read_text(encoding="utf-8")
    )
    assert seal["status"] == "prediction_sealed"
    assert seal["replay_max_abs_position_m"] == 0.0
    assert (
        seal["implementation"]["commit"] == "32b1a2af5a03a23eea453e1c5316cf2d3e9ff097"
    )

    result = json.loads((RESULT_ROOT / "evaluation.json").read_text(encoding="utf-8"))
    gate = result["evaluation"]["gate"]
    assert gate["passed"] is False
    assert gate["candidate_to_full_physical_ratio"][
        "chamfer_distance_m"
    ] == pytest.approx(1.3069794870384857)
    assert gate["candidate_to_full_physical_ratio"]["track_error_m"] == pytest.approx(
        1.6852643414651787
    )
    assert gate["next_step"] == "Close raw PGND replacement without a wider run."
