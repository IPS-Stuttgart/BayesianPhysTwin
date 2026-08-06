#!/usr/bin/env python3
"""Finish the temporary deterministic repair for PokeFlex paper-artifact CI."""

from __future__ import annotations

from pathlib import Path

_TEST_MARKER = "def test_reporting_helpers_fail_closed_and_write_deterministically"
_TEST_APPENDIX = r'''

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
        reporting._mapping_list([{} , 1], name="records")
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
        ("selector_adjusted_upper_regret_mm", -0.15, "selector-adjusted bound changed"),
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
'''


def validate_isolated_artifact_runtime() -> None:
    path = Path(".github/workflows/pokeflex-same-object-paper-artifacts.yml")
    text = path.read_text(encoding="utf-8")
    marker = "\n  artifacts:\n"
    if text.count(marker) != 1:
        raise SystemExit("unexpected paper-artifact job boundary")
    artifact_job = text.split(marker, 1)[1]
    if "actions/setup-python" in artifact_job:
        raise SystemExit("self-hosted artifact job still uses setup-python")
    required = (
        "Initialize isolated artifact paths and Python",
        "Validate frozen input custody before runtime installation",
        "Create isolated released-checkpoint runtime",
        '"${POKEFLEX_BOOTSTRAP_PYTHON}" -m venv --clear "${POKEFLEX_VENV}"',
        '"${POKEFLEX_VENV}/bin/python" -m pip check',
        "--no-cache-dir",
    )
    missing = [term for term in required if term not in artifact_job]
    if missing:
        raise SystemExit(f"isolated artifact runtime is incomplete: {missing}")


def repair_core_coverage() -> None:
    path = Path(".github/workflows/tests.yml")
    lines = path.read_text(encoding="utf-8").splitlines()
    target = "tests/test_pokeflex_same_object_reporting.py"
    target_count = sum(line.strip().rstrip(" \\") == target for line in lines)
    if target_count == 0:
        inserted = 0
        index = 0
        while index < len(lines):
            if lines[index].strip().rstrip(" \\") == "tests/test_quality_invariants.py":
                indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
                if not lines[index].rstrip().endswith("\\"):
                    lines[index] = lines[index].rstrip() + " \\"
                lines.insert(index + 1, indent + target)
                inserted += 1
                index += 1
            index += 1
        if inserted != 2:
            raise SystemExit(
                f"expected two stable/core reporting-test insertions, found {inserted}"
            )
    final_count = sum(line.strip().rstrip(" \\") == target for line in lines)
    if final_count != 2:
        raise SystemExit(
            f"PokeFlex reporting test must run in both core lists, found {final_count}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extend_reporting_tests() -> None:
    path = Path("tests/test_pokeflex_same_object_reporting.py")
    text = path.read_text(encoding="utf-8")
    if _TEST_MARKER not in text:
        path.write_text(text.rstrip() + "\n" + _TEST_APPENDIX + "\n", encoding="utf-8")


def main() -> int:
    validate_isolated_artifact_runtime()
    repair_core_coverage()
    extend_reporting_tests()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
