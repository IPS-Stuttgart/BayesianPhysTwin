from __future__ import annotations

import numpy as np

import bayesian_phystwin.deform360_pairwise_bias_aware_source as source
from bayesian_phystwin.deform360_online_belief_evaluation import PRIMARY_METRICS


def _score(value: float) -> dict[str, float]:
    return {metric: value for metric in PRIMARY_METRICS}


def _case(index: int) -> source._SourceCase:
    case_name = f"object-{index}-ep0000"
    trajectory = np.zeros((2, 1, 3), dtype=np.float32)
    candidate_report = {
        "updates": [
            {
                "candidate_available": True,
                "bit_exact_baseline_fallback": False,
                "pairwise_gate": {"accepted": True},
                "dynamic_window_selected": True,
                "selected_center_ids": list(range(9)),
            }
        ]
    }
    return source._SourceCase(
        case=case_name,
        object_id=f"object-{index}",
        center_ids=np.asarray([0]),
        target=trajectory.copy(),
        visibility=np.ones((2, 1), dtype=bool),
        validity=np.ones((2, 1), dtype=bool),
        trajectories={arm: trajectory.copy() for arm in source.ARMS},
        reports={
            source.PAIRWISE_RBF_ARM: {"updates": []},
            source.BIAS_AWARE_V4_ARM: {"updates": []},
            source.PAIRWISE_BIAS_AWARE_ARM: candidate_report,
        },
        scores={
            source.SELECTED_BASELINE_ARM: _score(1.0),
            source.PAIRWISE_RBF_ARM: _score(0.8),
            source.BIAS_AWARE_V4_ARM: _score(0.9),
            source.PAIRWISE_BIAS_AWARE_ARM: _score(0.75),
        },
        input_sha256={"synthetic": f"{index:064x}"},
    )


def test_source_evaluator_applies_frozen_object_level_gates(
    monkeypatch,
    tmp_path,
) -> None:
    cases = {_case(index).case: _case(index) for index in range(5)}
    monkeypatch.setattr(source, "_expected_case_names", lambda: tuple(cases))
    monkeypatch.setattr(
        source,
        "_load_source_case",
        lambda source_case_dir, *args, **kwargs: cases[source_case_dir.name],
    )

    result = source.evaluate_pairwise_bias_aware_source(
        tmp_path / "source",
        tmp_path / "measurement",
        tmp_path / "uncertainty",
        tmp_path / "baseline",
        tmp_path / "output",
    )

    assert result["larger_preregistered_run_justified"] is True
    assert all(result["advancement_gates"].values())
    assert result["object_level_gates"]["joint_object_win_count"] == 5
    assert result["candidate_update_count"] == 5
    assert result["exact_fallback_count"] == 0
    assert (tmp_path / "output" / "summary.json").is_file()
    assert len(result["artifacts"]) == 5
