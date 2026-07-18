import json
import pickle

import numpy as np
import pytest

from bayesian_phystwin.phystwin_sota_comparison import (
    PHYSTWIN_TABLE1_CASES,
    aggregate_phystwin_sota_comparison,
)


def _write_case(root, name: str, offset: float, candidate_scale: float) -> None:
    case = root / name
    case.mkdir()
    observed = np.zeros((3, 2, 3), dtype=float)
    visible = np.ones((3, 2), dtype=bool)
    tracks = observed[:, :1].copy()
    released = observed.copy()
    released[1:, :, 0] = offset
    candidate = observed.copy()
    candidate[1:, :, 0] = candidate_scale * offset
    with (case / "final_data.pkl").open("wb") as handle:
        pickle.dump(
            {
                "object_points": observed,
                "object_visibilities": visible,
                "surface_points": np.empty((0, 3)),
            },
            handle,
        )
    with (case / "gt_track_3d.pkl").open("wb") as handle:
        pickle.dump(tracks, handle)
    with (case / "released.pkl").open("wb") as handle:
        pickle.dump(released, handle)
    with (case / "candidate.pkl").open("wb") as handle:
        pickle.dump(candidate, handle)
    (case / "split.json").write_text(
        json.dumps({"train": [0, 1], "test": [1, 3], "frame_len": 3}),
        encoding="utf-8",
    )


def test_absolute_comparison_separates_all_and_confirmation_cohorts(tmp_path) -> None:
    cases = PHYSTWIN_TABLE1_CASES
    for index, case in enumerate(cases, start=1):
        _write_case(tmp_path, case, 0.01 * index, 0.5)
    (tmp_path / "evaluation_subset_manifest.json").write_text(
        json.dumps({"available_cases": list(cases), "selected_cases": list(cases)}),
        encoding="utf-8",
    )

    result = aggregate_phystwin_sota_comparison(
        tmp_path,
        {
            "released": str(tmp_path / "{case}" / "released.pkl"),
            "candidate": str(tmp_path / "{case}" / "candidate.pkl"),
        },
        tmp_path / "comparison.json",
    )

    released = result["methods"]["released"]["cohorts"]
    candidate = result["methods"]["candidate"]["cohorts"]
    assert released["all_22_table_compatible"]["case_count"] == 22
    assert released["confirmation_19"]["case_count"] == 19
    assert released["all_22_table_compatible"]["chamfer_distance_m"][
        "equal_case_mean_m"
    ] == pytest.approx(0.115)
    assert candidate["all_22_table_compatible"]["track_error_m"][
        "frame_weighted_mean_m"
    ] == pytest.approx(0.0575)
    first_case = cases[0]
    released_inputs = result["methods"]["released"]["per_case"][first_case]["inputs"]
    candidate_inputs = result["methods"]["candidate"]["per_case"][first_case]["inputs"]
    for shared_name in ("final_data", "gt_track_3d", "split"):
        assert released_inputs[shared_name] == candidate_inputs[shared_name]
        assert len(released_inputs[shared_name]["sha256"]) == 64
    assert result["schema_version"] == 2
    assert len(result["evaluator_identity"]["package_implementation_sha256"]) == 64


def test_comparison_rejects_a_template_without_case_placeholder(tmp_path) -> None:
    with pytest.raises(ValueError, match="must contain"):
        aggregate_phystwin_sota_comparison(
            tmp_path,
            {"released": str(tmp_path / "released.pkl")},
            tmp_path / "comparison.json",
        )


def test_comparison_rejects_a_substituted_22_case_cohort(tmp_path) -> None:
    substituted = (*PHYSTWIN_TABLE1_CASES[:-1], "substitute")
    (tmp_path / "evaluation_subset_manifest.json").write_text(
        json.dumps(
            {"available_cases": list(substituted), "selected_cases": list(substituted)}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Table-1 cohort"):
        aggregate_phystwin_sota_comparison(
            tmp_path,
            {"released": str(tmp_path / "{case}" / "released.pkl")},
            tmp_path / "comparison.json",
        )


@pytest.mark.parametrize(
    ("bad_trajectory", "message"),
    [
        (np.zeros((2, 2, 3)), "complete 3-frame sequence"),
        (np.zeros((3, 1, 3)), "num_surface_points exceeds"),
        (np.full((3, 2, 3), np.nan), "non-finite"),
    ],
)
def test_comparison_rejects_invalid_full_state_trajectories(
    tmp_path, bad_trajectory: np.ndarray, message: str
) -> None:
    for case in PHYSTWIN_TABLE1_CASES:
        _write_case(tmp_path, case, 0.01, 0.5)
    with (tmp_path / PHYSTWIN_TABLE1_CASES[0] / "candidate.pkl").open("wb") as handle:
        pickle.dump(bad_trajectory, handle)
    (tmp_path / "evaluation_subset_manifest.json").write_text(
        json.dumps(
            {
                "available_cases": list(PHYSTWIN_TABLE1_CASES),
                "selected_cases": list(PHYSTWIN_TABLE1_CASES),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        aggregate_phystwin_sota_comparison(
            tmp_path,
            {"candidate": str(tmp_path / "{case}" / "candidate.pkl")},
            tmp_path / "comparison.json",
        )


def test_comparison_rejects_disabling_input_identity_hashing(tmp_path) -> None:
    with pytest.raises(ValueError, match="identity hashing is mandatory"):
        aggregate_phystwin_sota_comparison(
            tmp_path,
            {"released": str(tmp_path / "{case}" / "released.pkl")},
            tmp_path / "comparison.json",
            hash_inputs=False,
        )


def test_comparison_rejects_an_input_changed_during_evaluation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for case in PHYSTWIN_TABLE1_CASES:
        _write_case(tmp_path, case, 0.01, 0.5)
    (tmp_path / "evaluation_subset_manifest.json").write_text(
        json.dumps(
            {
                "available_cases": list(PHYSTWIN_TABLE1_CASES),
                "selected_cases": list(PHYSTWIN_TABLE1_CASES),
            }
        ),
        encoding="utf-8",
    )
    import bayesian_phystwin.phystwin_sota_comparison as comparison_module

    original = comparison_module.official_phystwin_metrics_by_frame
    calls = 0

    def mutating_evaluator(*args, **kwargs):
        nonlocal calls
        metrics = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            changed = tmp_path / PHYSTWIN_TABLE1_CASES[0] / "final_data.pkl"
            changed.write_bytes(b"changed after its immutable snapshot was loaded")
        return metrics

    monkeypatch.setattr(
        comparison_module,
        "official_phystwin_metrics_by_frame",
        mutating_evaluator,
    )

    with pytest.raises(RuntimeError, match="input changed during aggregation"):
        aggregate_phystwin_sota_comparison(
            tmp_path,
            {
                "released": str(tmp_path / "{case}" / "released.pkl"),
                "candidate": str(tmp_path / "{case}" / "candidate.pkl"),
            },
            tmp_path / "comparison.json",
        )
