from __future__ import annotations

import numpy as np

from bayesian_phystwin.cli.deform360_dynamic_pairwise_source import build_parser
from bayesian_phystwin.deform360_dynamic_pairwise_belief import (
    DynamicPairwiseBeliefConfig,
)
from bayesian_phystwin.deform360_dynamic_pairwise_source import (
    _late_metrics,
    _observation_model,
)


def test_observation_model_uses_only_multiview_diagnostics() -> None:
    config = DynamicPairwiseBeliefConfig()
    arrays = {
        "center_ids": np.arange(64),
        "selected_cameras": np.asarray(["a", "b", "c", "d"]),
        "triangulation_inlier_view_count": np.full((3, 64), 3),
        "triangulation_median_reprojection_px": np.zeros((3, 64)),
    }

    reliability, variance, view_count = _observation_model(arrays, config=config)

    np.testing.assert_allclose(reliability, 2.0 / 3.0)
    np.testing.assert_array_equal(view_count, 3)
    np.testing.assert_allclose(variance, config.observation_variance_floor_m2)


def test_observation_model_rejects_wrong_pool_without_outcome_input() -> None:
    config = DynamicPairwiseBeliefConfig()
    arrays = {
        "center_ids": np.arange(63),
        "selected_cameras": np.asarray(["a", "b", "c"]),
        "triangulation_inlier_view_count": np.full((3, 63), 3),
        "triangulation_median_reprojection_px": np.zeros((3, 63)),
    }

    try:
        _observation_model(arrays, config=config)
    except ValueError as error:
        assert "pool count" in str(error)
    else:
        raise AssertionError("wrong observation pool was accepted")


def test_late_metrics_use_final_third_of_scored_frames() -> None:
    score = {
        "by_frame": {
            "hidden_identity_rmse_m": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "hidden_symmetric_chamfer_m": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        }
    }

    late = _late_metrics(score)

    assert late["late_hidden_identity_rmse_m"] == 5.5
    assert late["late_hidden_symmetric_chamfer_m"] == 11.0


def test_source_cli_keeps_transfer_digest_optional() -> None:
    args = build_parser().parse_args(["source", "measurement", "output"])

    assert args.source_root == "source"
    assert args.transfer_manifest_sha256 is None
