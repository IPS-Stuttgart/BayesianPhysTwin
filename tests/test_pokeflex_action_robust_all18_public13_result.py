import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    ROOT
    / "results"
    / "sota"
    / "pokeflex_action_robust_all18_v4_public13_retrospective"
    / "result.json"
)
CALIBRATION_PATH = (
    ROOT / "configs" / "sota" / "pokeflex_action_robust_scale_all18_v4.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public13_retrospective_result_is_frozen_and_source_bound() -> None:
    result = _load(RESULT_PATH)
    calibration = _load(CALIBRATION_PATH)

    assert _file_sha256(RESULT_PATH) == (
        "9d6a3ce6e4d606485dcecfb12418199dc4bd3bbf43236e2d42f3f25f94a98a0e"
    )
    assert result["artifact_kind"] == (
        "PokeFlexActionRobustAll18V4Public13RetrospectiveDiagnostic"
    )
    assert result["status"] == (
        "post-open retrospective diagnostic; not confirmation"
    )
    assert result["implementation_revision"] == (
        "6af891b882782e5c5f099dd1610c49f772866445"
    )
    assert result["calibration_sha256"] == calibration["calibration_sha256"]
    assert result["calibration_sha256"] == (
        "e94eeb9bdd2cc69e245b0bd48d843e5f64cb039e1eb02841e4a784cbe4dbc880"
    )
    assert result["calibration_file_sha256"] == _file_sha256(CALIBRATION_PATH)
    assert result["calibration_file_sha256"] == (
        "00cdf5732f5dbf7eb0f899ebbb536260d9e66c0a151b41eec81ffaaef4aaf110"
    )
    assert result["archived_v3_result_file_sha256"] == (
        "619c46726aab0f7e81d2e943bd44820e521c9fe6285906add28af87203c15ebd"
    )


def test_public13_retrospective_result_preserves_access_boundary() -> None:
    result = _load(RESULT_PATH)

    assert result["parameter_selection_from_this_cohort"] is False
    assert result["future_or_missing_official_takes_accessed"] is False
    assert result["held_v8_accessed"] is False
    assert len(result["objects"]) == 13
    assert sum(row["scored_frame_count"] for row in result["objects"]) == 970
    assert sum(row["supported_frame_count"] for row in result["objects"]) == 835
    boundary = result["claim_boundary"].lower()
    assert "retrospective" in boundary
    assert "cannot confirm" in boundary
    assert "official 18-take" in boundary


def test_public13_retrospective_result_improves_registered_references() -> None:
    result = _load(RESULT_PATH)
    aggregate = result["aggregate"]

    assert aggregate["baseline_frame_balanced_CD_UL1_mm"] == pytest.approx(
        6.5694225302682865
    )
    assert aggregate["baseline_object_balanced_CD_UL1_mm"] == pytest.approx(
        6.79037059694753
    )
    assert aggregate["global_frame_balanced_CD_UL1_mm"] == pytest.approx(
        6.4993172114797195
    )
    assert aggregate["global_object_balanced_CD_UL1_mm"] == pytest.approx(
        6.719721545535675
    )
    assert aggregate["v3_robust_frame_balanced_CD_UL1_mm"] == pytest.approx(
        6.447848304173961
    )
    assert aggregate["v3_robust_object_balanced_CD_UL1_mm"] == pytest.approx(
        6.66389783963011
    )
    assert aggregate["v4_all18_frame_balanced_CD_UL1_mm"] == pytest.approx(
        6.4034362412801356
    )
    assert aggregate["v4_all18_object_balanced_CD_UL1_mm"] == pytest.approx(
        6.624635527710045
    )
    assert aggregate["v4_relative_improvement_vs_baseline"] == pytest.approx(
        0.024407367296269158
    )
    assert aggregate["v4_relative_improvement_vs_global"] == pytest.approx(
        0.014150291374618281
    )
    assert aggregate["v4_relative_improvement_vs_v3"] == pytest.approx(
        0.005891793791701362
    )
    assert aggregate["v4_wins_ties_losses_vs_baseline"] == [12, 1, 0]
    assert aggregate["bootstrap_97_5_upper_v4_minus_baseline_mm"] < 0
    assert aggregate["bootstrap_97_5_upper_v4_minus_global_mm"] < 0


def test_public13_retrospective_context_is_numerical_not_confirmatory() -> None:
    result = _load(RESULT_PATH)
    aggregate = result["aggregate"]

    assert aggregate["max_baseline_reproduction_drift_mm"] == 0
    assert aggregate["max_global_reproduction_drift_mm"] == 0
    assert aggregate["v4_frame_balanced_below_published_reference"] is True
    assert aggregate["v4_all18_frame_balanced_CD_UL1_mm"] < 6.498

    regressions = [
        row
        for row in result["objects"]
        if row["v4_all18_mean_CD_UL1_mm"] > row["global_mean_CD_UL1_mm"]
    ]
    assert [row["take_id"] for row in regressions] == ["3dPrintedBunny_T1"]
    assert (
        regressions[0]["v4_all18_mean_CD_UL1_mm"]
        < regressions[0]["baseline_mean_CD_UL1_mm"]
    )
