from __future__ import annotations

from pathlib import Path

import numpy as np

from bayesian_phystwin.deform360_raw_pairwise_correspondence_diagnostic import (
    CPD_ARM,
    PERSISTENCE_CLIQUE_RBF_ARM,
    UNGATED_RBF_ARM,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_artifacts import (
    selective_case_records,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_evaluation import (
    PRIMARY_METRICS,
    aggregate_selective_case_reports,
    score_selective_virtual_sensing_arrays,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_protocol import (
    PROTOCOL_ID,
)


PROTOCOL = (
    Path(__file__).parents[1]
    / "configs"
    / "sota"
    / "deform360_selective_virtual_sensing_v1.json"
)


def test_scoring_permanently_excludes_assimilation_centers() -> None:
    frame_zero = np.linspace(-0.2, 0.2, 24 * 3, dtype=np.float32).reshape(24, 3)
    target = np.repeat(frame_zero[None], 76, axis=0)
    target[:, :, 0] += np.arange(76, dtype=np.float32)[:, None] * 0.001
    persistence = np.repeat(frame_zero[None], 76, axis=0)
    primary = target.copy()
    primary[:, :16] += 10.0
    trajectories = {
        PERSISTENCE_CLIQUE_RBF_ARM: primary,
        "persistence": persistence,
        UNGATED_RBF_ARM: persistence.copy(),
        CPD_ARM: persistence.copy(),
    }
    valid = np.ones(target.shape[:2], dtype=bool)

    scores = score_selective_virtual_sensing_arrays(
        trajectories,
        target,
        valid,
        valid,
        center_ids=np.arange(16, dtype=np.int64),
    )

    assert scores[PERSISTENCE_CLIQUE_RBF_ARM][
        "post_update_hidden_identity_rmse_m"
    ] == 0.0
    assert scores[PERSISTENCE_CLIQUE_RBF_ARM][
        "post_update_hidden_symmetric_chamfer_m"
    ] == 0.0
    assert scores["persistence"]["post_update_hidden_identity_rmse_m"] > 0.0


def _case_reports(*, regressing_stratum: str | None = None) -> list[dict[str, object]]:
    chosen_objects: dict[str, list[str]] = {}
    reports = []
    for record in selective_case_records(PROTOCOL):
        stratum = str(record["stratum"])
        object_id = str(record["object_id"])
        selected = chosen_objects.setdefault(stratum, [])
        if object_id in selected or len(selected) == 3:
            continue
        selected.append(object_id)
        primary = 0.011 if stratum == regressing_stratum else 0.008
        scores = {
            PERSISTENCE_CLIQUE_RBF_ARM: {
                metric: primary for metric in PRIMARY_METRICS
            },
            "persistence": {metric: 0.010 for metric in PRIMARY_METRICS},
            UNGATED_RBF_ARM: {metric: 0.009 for metric in PRIMARY_METRICS},
            CPD_ARM: {metric: 0.0095 for metric in PRIMARY_METRICS},
        }
        reports.append(
            {
                "artifact_kind": "Deform360SelectiveVirtualSensingEvaluation",
                "protocol_id": PROTOCOL_ID,
                **record,
                "scores": scores,
            }
        )
    return reports


def test_object_balanced_aggregate_passes_every_locked_gate() -> None:
    result = aggregate_selective_case_reports(
        _case_reports(), protocol_path=PROTOCOL
    )

    assert result["object_count"] == 9
    assert result["object_count_by_stratum"] == {
        "filament": 3,
        "sheet": 3,
        "volumetric": 3,
    }
    assert result["paper_threshold_passed"] is True
    for metric in PRIMARY_METRICS:
        comparison = result["primary_vs_persistence"][metric]
        assert np.isclose(comparison["relative_change"], -0.2)
        assert comparison["exact_sign_test"]["one_sided_exact_p"] < 0.05
        assert all(comparison["gates"].values())


def test_any_stratum_regression_blocks_the_paper_threshold() -> None:
    result = aggregate_selective_case_reports(
        _case_reports(regressing_stratum="sheet"), protocol_path=PROTOCOL
    )

    assert result["paper_threshold_passed"] is False
    for metric in PRIMARY_METRICS:
        comparison = result["primary_vs_persistence"][metric]
        assert comparison["gates"]["no_stratum_mean_regression"] is False
