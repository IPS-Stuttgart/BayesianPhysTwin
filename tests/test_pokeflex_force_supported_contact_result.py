import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    REPOSITORY_ROOT
    / "results"
    / "sota"
    / "pokeflex_force_supported_contact_v1"
    / "summary.json"
)


def test_force_supported_result_preserves_failed_gate() -> None:
    result = json.loads(SUMMARY.read_text(encoding="utf-8"))
    evaluation = result["development_evaluation"]

    assert evaluation["take_count"] == 20
    assert evaluation["frame_count"] == 1418
    assert evaluation["object_wins"] == 5
    assert evaluation["gates"]["minimum_relative_improvement"]["passed"] is False
    assert evaluation["gates"]["all_passed"] is False
    assert evaluation["decision"].startswith("FAILED")


def test_force_supported_result_keeps_unopened_boundaries() -> None:
    result = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert result["unopened_boundaries"] == {
        "calibration_objects": True,
        "development_take_T2": True,
        "development_takes": ["T7", "T8"],
        "target_objects": True,
    }
    assert len(result["source_artifacts"]) == 20
    assert all(len(item["sha256"]) == 64 for item in result["source_artifacts"])
    assert result["claim_boundary"]["sota_claim_permitted"] is False


def test_force_supported_result_locks_implementation_hashes() -> None:
    result = json.loads(SUMMARY.read_text(encoding="utf-8"))
    implementation = result["implementation"]
    paths = {
        "registration_module_sha256": (
            REPOSITORY_ROOT
            / "src"
            / "bayesian_phystwin"
            / "pokeflex_bayesian_registration.py"
        ),
        "runner_sha256": (
            REPOSITORY_ROOT
            / "scripts"
            / "remote"
            / "run_pokeflex_checkpoint_registration_smoke.py"
        ),
        "test_module_sha256": (
            REPOSITORY_ROOT / "tests" / "test_pokeflex_bayesian_registration.py"
        ),
    }

    for key, path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == implementation[key]
