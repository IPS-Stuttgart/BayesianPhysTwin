"""Tests for the bounded PokeFlex same-object paper reporting path."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin import pokeflex_same_object_reporting as reporting

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_RESULT_PATH = (
    _REPOSITORY_ROOT
    / "results"
    / "sota"
    / "pokeflex_independent_depth_regret_guard_prospective_v1"
    / "prospective_evaluation.json"
)


def _synthetic_result() -> dict[str, object]:
    decisions = [
        {
            "frame_id": "SeenA_T7:f00001",
            "take_id": "SeenA_T7",
            "object": "SeenA",
            "take": "T7",
            "target_frame": 1,
            "baseline_error_mm": 2.0,
            "selected_error_mm": 1.0,
            "hidden_regret_mm": -1.0,
            "candidate_upper_regret_mm": -0.3,
            "selector_adjusted_upper_regret_mm": -0.2,
            "selected_arm": "candidate_a",
            "accepted": True,
        },
        {
            "frame_id": "SeenA_T8:f00001",
            "take_id": "SeenA_T8",
            "object": "SeenA",
            "take": "T8",
            "target_frame": 1,
            "baseline_error_mm": 2.0,
            "selected_error_mm": 2.0,
            "hidden_regret_mm": 0.0,
            "candidate_upper_regret_mm": 0.2,
            "selector_adjusted_upper_regret_mm": 0.3,
            "selected_arm": "released_checkpoint",
            "accepted": False,
        },
        {
            "frame_id": "SeenB_T7:f00001",
            "take_id": "SeenB_T7",
            "object": "SeenB",
            "take": "T7",
            "target_frame": 1,
            "baseline_error_mm": 2.0,
            "selected_error_mm": 2.0,
            "hidden_regret_mm": 0.0,
            "candidate_upper_regret_mm": 0.1,
            "selector_adjusted_upper_regret_mm": 0.2,
            "selected_arm": "released_checkpoint",
            "accepted": False,
        },
    ]
    takes = [
        {
            "take_id": "SeenA_T7",
            "object": "SeenA",
            "target_frame_count": 1,
            "baseline_mean_CD_UL1_mm": 2.0,
            "selected_mean_CD_UL1_mm": 1.0,
            "relative_improvement": 0.5,
        },
        {
            "take_id": "SeenA_T8",
            "object": "SeenA",
            "target_frame_count": 1,
            "baseline_mean_CD_UL1_mm": 2.0,
            "selected_mean_CD_UL1_mm": 2.0,
            "relative_improvement": 0.0,
        },
        {
            "take_id": "SeenB_T7",
            "object": "SeenB",
            "target_frame_count": 1,
            "baseline_mean_CD_UL1_mm": 2.0,
            "selected_mean_CD_UL1_mm": 2.0,
            "relative_improvement": 0.0,
        },
    ]
    objects = [
        {
            "object": "SeenA",
            "take_count": 2,
            "baseline_mean_CD_UL1_mm": 2.0,
            "selected_mean_CD_UL1_mm": 1.5,
            "relative_improvement": 0.25,
        },
        {
            "object": "SeenB",
            "take_count": 1,
            "baseline_mean_CD_UL1_mm": 2.0,
            "selected_mean_CD_UL1_mm": 2.0,
            "relative_improvement": 0.0,
        },
    ]
    return {
        "artifact_kind": reporting.EXPECTED_ARTIFACT_KIND,
        "claim_status": reporting.EXPECTED_CLAIM_STATUS,
        "gate_passed": True,
        "aggregation": "equal frames, takes, then objects",
        "take_ids": ["SeenA_T7", "SeenA_T8", "SeenB_T7"],
        "take_count": 3,
        "object_count": 2,
        "object_wins": 2,
        "object_losses": 0,
        "baseline_object_mean_CD_UL1_mm": 2.0,
        "selected_object_mean_CD_UL1_mm": 1.75,
        "object_balanced_relative_improvement": 0.125,
        "accepted_frame_count": 1,
        "accepted_frame_wins": 1,
        "accepted_frame_losses": 0,
        "exact_fallback_frame_count": 2,
        "takes": takes,
        "objects": objects,
        "decisions": decisions,
        "deployment_artifact": {
            "candidate_certificate": {},
            "selector_correction_bound": {},
        },
    }


def test_committed_same_object_result_retains_bounded_claim() -> None:
    result = reporting.load_json_object(_RESULT_PATH)
    summary = reporting.validate_bounded_result(result)

    assert summary["take_count"] == 3
    assert summary["object_count"] == 2
    assert summary["frame_count"] == 241
    assert summary["accepted_frame_count"] == 87
    assert summary["exact_fallback_frame_count"] == 154
    assert summary["accepted_frame_wins"] == 77
    assert summary["accepted_frame_losses"] == 10
    assert summary["object_wins"] == 2
    assert summary["object_losses"] == 0
    assert summary["object_balanced_relative_improvement"] == pytest.approx(
        0.030601394040032672
    )
    assert "independent-object generalization" in summary["excluded_claims"]


def test_post_outcome_diagnostic_reconstructs_frozen_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _synthetic_result()
    frames = [
        {
            "frame_id": value["frame_id"],
            "take_id": value["take_id"],
            "object": value["object"],
            "take": value["take"],
            "target_frame": value["target_frame"],
            "baseline_error_mm": value["baseline_error_mm"],
        }
        for value in result["decisions"]
    ]
    rows = [
        {
            **frames[0],
            "candidate": "candidate_a",
            "features": np.asarray([-0.3]),
            "candidate_error_mm": 1.0,
        },
        {
            **frames[1],
            "candidate": "candidate_b",
            "features": np.asarray([0.2]),
            "candidate_error_mm": 3.0,
        },
        {
            **frames[2],
            "candidate": "candidate_c",
            "features": np.asarray([0.1]),
            "candidate_error_mm": 1.9,
        },
    ]
    certificate = SimpleNamespace(
        minimum_improvement=0.0,
        nominal_coverage=0.9,
        finite_sample_coverage=0.92,
        upper_regret=lambda features: float(features[0]),
    )
    bound = SimpleNamespace(
        upper_regret_m=0.1,
        nominal_coverage=0.9,
        finite_sample_coverage=0.91,
    )
    monkeypatch.setattr(
        reporting,
        "extract_pokeflex_regret_guard_rows",
        lambda payloads: (rows, frames),
    )
    monkeypatch.setattr(reporting, "_certificate_from_dict", lambda value: certificate)
    monkeypatch.setattr(reporting, "_bound_from_dict", lambda value: bound)

    diagnostic = reporting.build_candidate_diagnostics([{}], result)
    summary = diagnostic["candidate_diagnostic"]
    assert summary["candidate_supported_frame_count"] == 3
    assert summary["accepted_frame_count"] == 1
    assert summary["safe_accepted_frame_count"] == 1
    assert summary["harmful_accepted_frame_count"] == 0
    assert summary["harmful_candidate_fallback_count"] == 1
    assert summary["beneficial_candidate_fallback_count"] == 1
    assert diagnostic["analysis_role"].startswith("post-outcome visualization")


def test_calibration_figure_renders_from_diagnostic(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    path = (
        _REPOSITORY_ROOT / "scripts" / "paper" / "make_pokeflex_same_object_figure.py"
    )
    spec = importlib.util.spec_from_file_location("pokeflex_same_object_figure", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    diagnostic = {
        "bounded_result": reporting.validate_bounded_result(_synthetic_result()),
        "candidate_diagnostic": {
            "adjusted_upper_bound_coverage": 2.0 / 3.0,
            "accepted_harmful_fraction": 0.0,
        },
        "rows": [
            {
                "candidate_supported": True,
                "accepted": True,
                "candidate_regret_mm": -1.0,
                "selector_adjusted_upper_regret_mm": -0.2,
            },
            {
                "candidate_supported": True,
                "accepted": False,
                "candidate_regret_mm": 1.0,
                "selector_adjusted_upper_regret_mm": 0.3,
            },
            {
                "candidate_supported": True,
                "accepted": False,
                "candidate_regret_mm": -0.1,
                "selector_adjusted_upper_regret_mm": 0.2,
            },
        ],
    }
    png_path = tmp_path / "figure.png"
    pdf_path = tmp_path / "figure.pdf"
    module.render_figure(diagnostic, png_path, pdf_path)
    assert png_path.stat().st_size > 10_000
    assert pdf_path.stat().st_size > 5_000


def test_reporting_helpers_fail_closed_and_write_deterministically(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must be an array"):
        reporting._mapping_list({}, name="records")
    with pytest.raises(ValueError, match="exactly 2 entries"):
        reporting._mapping_list([{}], name="records", expected_length=2)
    with pytest.raises(ValueError, match="must not be empty"):
        reporting._mapping_list([], name="records", nonempty=True)
    with pytest.raises(ValueError, match="entries must be objects"):
        reporting._mapping_list([{}, 1], name="records")
    with pytest.raises(ValueError, match="nonempty strings"):
        reporting._string_list({}, name="names")
    with pytest.raises(ValueError, match="nonempty strings"):
        reporting._string_list([""], name="names")
    with pytest.raises(ValueError, match="must be an integer"):
        reporting._integer(True, name="count")
    with pytest.raises(ValueError, match="must be an integer"):
        reporting._integer(1.0, name="count")
    with pytest.raises(ValueError, match="must be numeric"):
        reporting._number(False, name="score")
    with pytest.raises(ValueError, match="must be numeric"):
        reporting._number("1", name="score")
    with pytest.raises(ValueError, match="must be finite"):
        reporting._number(float("inf"), name="score")

    assert reporting._close(None, None)
    assert not reporting._close(None, 0.0)
    assert reporting._close(1.0, 1.0 + 1e-11)
    assert not reporting._close(1.0, 1.1)

    payload_path = tmp_path / "payload.bin"
    payload_path.write_bytes(b"bounded-pokeflex")
    assert reporting.sha256_file(payload_path) == (
        "0d8f4d895824c350563388434ab0c7b3bb6b184c1f075bbf3d3ef2ec6cc0856e"
    )

    array_path = tmp_path / "array.json"
    array_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON root is not an object"):
        reporting.load_json_object(array_path)

    output_path = tmp_path / "nested" / "result.json"
    reporting.write_json(output_path, {"finite": 1.25, "status": "bounded"})
    assert reporting.load_json_object(output_path) == {
        "finite": 1.25,
        "status": "bounded",
    }


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("kind", "unexpected prospective result kind"),
        ("claim", "prospective claim status changed"),
        ("gate", "prospective gate did not pass"),
        ("takes_type", "takes must be an array"),
        ("takes_length", "takes must contain exactly 3 entries"),
        ("decisions_empty", "decisions must not be empty"),
        ("decisions_entry", "decisions entries must be objects"),
        ("take_count_type", "take_count must be an integer"),
        ("take_count", "take count changed"),
        ("object_count", "object count changed"),
        ("object_wins", "object-win result changed"),
        ("object_losses", "object-loss result changed"),
        ("frame_accounting", "take/frame accounting changed"),
        ("fallback_accounting", "accept/fallback accounting changed"),
        ("accepted_accounting", "accepted-frame accounting changed"),
        ("take_regression", "a prospective take now regresses"),
        ("baseline_type", "baseline_object_mean_CD_UL1_mm must be numeric"),
        ("baseline_finite", "baseline_object_mean_CD_UL1_mm must be finite"),
        ("relative", "object-balanced improvement is inconsistent"),
    ],
)
def test_bounded_result_rejects_tampering(case: str, match: str) -> None:
    result = _synthetic_result()
    takes = result["takes"]
    decisions = result["decisions"]
    assert isinstance(takes, list)
    assert isinstance(decisions, list)

    if case == "kind":
        result["artifact_kind"] = "other"
    elif case == "claim":
        result["claim_status"] = "retuned"
    elif case == "gate":
        result["gate_passed"] = False
    elif case == "takes_type":
        result["takes"] = {}
    elif case == "takes_length":
        result["takes"] = takes[:-1]
    elif case == "decisions_empty":
        result["decisions"] = []
    elif case == "decisions_entry":
        result["decisions"] = [*decisions, 1]
    elif case == "take_count_type":
        result["take_count"] = True
    elif case == "take_count":
        result["take_count"] = 4
    elif case == "object_count":
        result["object_count"] = 3
    elif case == "object_wins":
        result["object_wins"] = 1
    elif case == "object_losses":
        result["object_losses"] = 1
    elif case == "frame_accounting":
        first_take = takes[0]
        assert isinstance(first_take, dict)
        first_take["target_frame_count"] = 2
    elif case == "fallback_accounting":
        result["accepted_frame_count"] = 2
    elif case == "accepted_accounting":
        result["accepted_frame_wins"] = 0
    elif case == "take_regression":
        first_take = takes[0]
        assert isinstance(first_take, dict)
        first_take["selected_mean_CD_UL1_mm"] = 3.0
    elif case == "baseline_type":
        result["baseline_object_mean_CD_UL1_mm"] = False
    elif case == "baseline_finite":
        result["baseline_object_mean_CD_UL1_mm"] = float("nan")
    elif case == "relative":
        result["object_balanced_relative_improvement"] = 0.5
    else:  # pragma: no cover - the parameter table is exhaustive.
        raise AssertionError(case)

    with pytest.raises(ValueError, match=match):
        reporting.validate_bounded_result(result)


def _diagnostic_stubs(
    monkeypatch: pytest.MonkeyPatch,
    result: dict[str, object],
    *,
    candidate_errors: tuple[float, ...] = (1.0, 3.0, 1.9),
    selected_rows: dict[str, tuple[float, int]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    decisions = result["decisions"]
    assert isinstance(decisions, list)
    frames = [
        {
            "frame_id": value["frame_id"],
            "take_id": value["take_id"],
            "object": value["object"],
            "take": value["take"],
            "target_frame": value["target_frame"],
            "baseline_error_mm": value["baseline_error_mm"],
        }
        for value in decisions
        if isinstance(value, dict)
    ]
    features = (-0.3, 0.2, 0.1)
    rows = [
        {
            **frame,
            "candidate": f"candidate_{index}",
            "features": np.asarray([features[index]]),
            "candidate_error_mm": candidate_errors[index],
        }
        for index, frame in enumerate(frames)
    ]
    certificate = SimpleNamespace(
        minimum_improvement=0.0,
        nominal_coverage=0.9,
        finite_sample_coverage=0.92,
        upper_regret=lambda feature: float(feature[0]),
    )
    bound = SimpleNamespace(
        upper_regret_m=0.1,
        nominal_coverage=0.9,
        finite_sample_coverage=0.91,
    )
    monkeypatch.setattr(
        reporting,
        "extract_pokeflex_regret_guard_rows",
        lambda payloads: (rows, frames),
    )
    monkeypatch.setattr(reporting, "_certificate_from_dict", lambda value: certificate)
    monkeypatch.setattr(reporting, "_bound_from_dict", lambda value: bound)
    if selected_rows is None:
        selected_rows = {
            str(frame["frame_id"]): (features[index], index)
            for index, frame in enumerate(frames)
        }
    monkeypatch.setattr(
        reporting,
        "_select_candidates",
        lambda candidate_rows, upper_by_index: selected_rows,
    )
    return rows, frames


def test_candidate_diagnostic_handles_unsupported_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _synthetic_result()
    decisions = result["decisions"]
    assert isinstance(decisions, list)
    unsupported = decisions[2]
    assert isinstance(unsupported, dict)
    unsupported["candidate_upper_regret_mm"] = None
    unsupported["selector_adjusted_upper_regret_mm"] = None
    selected_rows = {
        "SeenA_T7:f00001": (-0.3, 0),
        "SeenA_T8:f00001": (0.2, 1),
    }
    _diagnostic_stubs(monkeypatch, result, selected_rows=selected_rows)

    diagnostic = reporting.build_candidate_diagnostics([{}], result)
    summary = diagnostic["candidate_diagnostic"]
    assert summary["candidate_supported_frame_count"] == 2
    assert summary["candidate_unsupported_frame_count"] == 1
    unsupported_rows = [
        row for row in diagnostic["rows"] if not row["candidate_supported"]
    ]
    assert unsupported_rows == [
        {
            "frame_id": "SeenB_T7:f00001",
            "take_id": "SeenB_T7",
            "object": "SeenB",
            "take": "T7",
            "target_frame": 1,
            "candidate_supported": False,
            "accepted": False,
            "selected_arm": "released_checkpoint",
            "baseline_error_mm": 2.0,
            "deployed_error_mm": 2.0,
        }
    ]


def test_candidate_diagnostic_handles_no_supported_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _synthetic_result()
    decisions = result["decisions"]
    takes = result["takes"]
    assert isinstance(decisions, list)
    assert isinstance(takes, list)
    for decision in decisions:
        assert isinstance(decision, dict)
        decision.update(
            {
                "accepted": False,
                "candidate_upper_regret_mm": None,
                "selector_adjusted_upper_regret_mm": None,
                "selected_arm": "released_checkpoint",
                "selected_error_mm": decision["baseline_error_mm"],
            }
        )
    for take in takes:
        assert isinstance(take, dict)
        take["selected_mean_CD_UL1_mm"] = take["baseline_mean_CD_UL1_mm"]
    result.update(
        {
            "accepted_frame_count": 0,
            "accepted_frame_wins": 0,
            "accepted_frame_losses": 0,
            "exact_fallback_frame_count": 3,
            "selected_object_mean_CD_UL1_mm": 2.0,
            "object_balanced_relative_improvement": 0.0,
        }
    )
    _diagnostic_stubs(monkeypatch, result, selected_rows={})

    diagnostic = reporting.build_candidate_diagnostics([{}], result)
    summary = diagnostic["candidate_diagnostic"]
    assert summary["candidate_supported_frame_count"] == 0
    assert summary["adjusted_upper_bound_coverage"] is None
    assert summary["accepted_harmful_fraction"] == 0.0


def test_candidate_diagnostic_counts_harmful_accepted_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _synthetic_result()
    decisions = result["decisions"]
    assert isinstance(decisions, list)
    accepted = decisions[0]
    assert isinstance(accepted, dict)
    accepted["selected_error_mm"] = 3.0
    result["accepted_frame_wins"] = 0
    result["accepted_frame_losses"] = 1
    _diagnostic_stubs(
        monkeypatch,
        result,
        candidate_errors=(3.0, 3.0, 1.9),
    )

    diagnostic = reporting.build_candidate_diagnostics([{}], result)
    summary = diagnostic["candidate_diagnostic"]
    assert summary["safe_accepted_frame_count"] == 0
    assert summary["harmful_accepted_frame_count"] == 1
    assert summary["accepted_harmful_fraction"] == 1.0


def test_candidate_diagnostic_rejects_inventory_and_decision_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _synthetic_result()
    _, frames = _diagnostic_stubs(monkeypatch, result)
    frames[0]["take_id"] = "Other_T7"
    with pytest.raises(ValueError, match="candidate take inventory changed"):
        reporting.build_candidate_diagnostics([{}], result)

    result = _synthetic_result()
    result["deployment_artifact"] = None
    _diagnostic_stubs(monkeypatch, result)
    with pytest.raises(ValueError, match="deployment artifact is missing"):
        reporting.build_candidate_diagnostics([{}], result)

    result = _synthetic_result()
    decisions = result["decisions"]
    assert isinstance(decisions, list)
    first = decisions[0]
    second = decisions[1]
    assert isinstance(first, dict)
    assert isinstance(second, dict)
    second["frame_id"] = first["frame_id"]
    _diagnostic_stubs(monkeypatch, result)
    with pytest.raises(ValueError, match="committed decision inventory changed"):
        reporting.build_candidate_diagnostics([{}], result)


def test_candidate_diagnostic_rejects_committed_numeric_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = (
        ("candidate_upper_regret_mm", -0.25, "candidate bound changed"),
        (
            "selector_adjusted_upper_regret_mm",
            -0.15,
            "selector-adjusted bound changed",
        ),
        ("accepted", False, "decision changed"),
        ("selected_arm", "released_checkpoint", "selected arm changed"),
        ("selected_error_mm", 1.5, "deployed error changed"),
    )
    for key, value, match in cases:
        result = _synthetic_result()
        decisions = result["decisions"]
        assert isinstance(decisions, list)
        first = decisions[0]
        assert isinstance(first, dict)
        first[key] = value
        _diagnostic_stubs(monkeypatch, result)
        with pytest.raises(ValueError, match=match):
            reporting.build_candidate_diagnostics([{}], result)
