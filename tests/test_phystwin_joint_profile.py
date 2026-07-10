import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.phystwin_joint_profile import combine_joint_profile_files


def _profile(path: Path, object_center: float, controller_center: float) -> None:
    object_grid = np.array([-0.2, 0.0, 0.2])
    controller_grid = np.array([-0.4, 0.0, 0.4])
    object_mesh, controller_mesh = np.meshgrid(
        object_grid,
        controller_grid,
        indexing="ij",
    )
    likelihood = -20.0 * np.square(object_mesh - object_center)
    likelihood -= 5.0 * np.square(controller_mesh - controller_center)
    np.savez(
        path,
        object_log_scales=object_grid,
        controller_log_scales=controller_grid,
        log_likelihood=likelihood,
    )


def test_joint_profiles_share_object_marginal_and_keep_trial_controllers(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _profile(first, -0.2, -0.4)
    _profile(second, 0.2, 0.4)

    summary = combine_joint_profile_files(
        {"first": first, "second": second},
        tmp_path / "joint",
        object_prior_std=0.3,
        controller_prior_std=0.5,
    )

    with np.load(summary["outputs"]["first"]) as first_joint:
        first_weights = first_joint["posterior_weights"]
        shared = first_joint["shared_object_weights"]
    with np.load(summary["outputs"]["second"]) as second_joint:
        second_weights = second_joint["posterior_weights"]

    np.testing.assert_allclose(first_weights.sum(axis=1), shared)
    np.testing.assert_allclose(second_weights.sum(axis=1), shared)
    assert np.sum(first_weights[:, 0]) > np.sum(first_weights[:, -1])
    assert np.sum(second_weights[:, -1]) > np.sum(second_weights[:, 0])
    assert json.loads((tmp_path / "joint" / "summary.json").read_text())[
        "contract"
    ].startswith("one shared")


def test_joint_profiles_require_matching_grids(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _profile(first, 0.0, 0.0)
    _profile(second, 0.0, 0.0)
    with np.load(second) as archive:
        np.savez(
            second,
            object_log_scales=np.array([-0.1, 0.0, 0.1]),
            controller_log_scales=archive["controller_log_scales"],
            log_likelihood=archive["log_likelihood"],
        )

    with pytest.raises(ValueError, match="identical profile grids"):
        combine_joint_profile_files(
            {"first": first, "second": second},
            tmp_path / "joint",
        )


def test_hierarchical_profiles_partially_pool_conflicting_object_scales(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _profile(first, -0.2, -0.4)
    _profile(second, 0.2, 0.4)

    summary = combine_joint_profile_files(
        {"first": first, "second": second},
        tmp_path / "hierarchical",
        object_prior_std=0.3,
        controller_prior_std=0.5,
        object_deviation_stds=(0.05, 0.2, 0.5),
        object_deviation_prior_scale=0.2,
    )

    with np.load(summary["outputs"]["first"]) as first_profile:
        first_weights = first_profile["posterior_weights"]
        assert "population_hyper_weights" in first_profile.files
    with np.load(summary["outputs"]["second"]) as second_profile:
        second_weights = second_profile["posterior_weights"]

    object_grid = np.array([-0.2, 0.0, 0.2])
    first_mean = float(np.sum(first_weights.sum(axis=1) * object_grid))
    second_mean = float(np.sum(second_weights.sum(axis=1) * object_grid))
    assert first_mean < 0.0 < second_mean
    assert summary["pooling"] == "hierarchical"
    assert summary["object_deviation_std"]["mean"] > 0.0
