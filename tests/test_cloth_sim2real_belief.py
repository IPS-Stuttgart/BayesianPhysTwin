from __future__ import annotations

from pathlib import Path

import numpy as np

from bayesian_phystwin.cloth_sim2real_belief import (
    ClothReadoutBeliefConfig,
    apply_guarded_readout_correction,
    associate_dense_cloud,
    fit_guarded_readout_correction,
    load_binary_little_endian_ply_xyz,
    mesh_edges_from_faces,
    sample_physical_rollout,
    symmetric_l1_chamfer_m,
)


def _write_open3d_ply(path: Path, points: np.ndarray) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Created by Open3D\n"
        f"element vertex {len(points)}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    values = np.empty(
        len(points),
        dtype=np.dtype(
            [
                ("x", "<f8"),
                ("y", "<f8"),
                ("z", "<f8"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        ),
    )
    values["x"], values["y"], values["z"] = points.T
    values["red"] = 1
    values["green"] = 2
    values["blue"] = 3
    with path.open("wb") as stream:
        stream.write(header)
        values.tofile(stream)


def _grid() -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    return points, faces


def test_open3d_binary_ply_loader_preserves_xyz(tmp_path: Path) -> None:
    expected, _ = _grid()
    path = tmp_path / "cloud.ply"
    _write_open3d_ply(path, expected)

    actual = load_binary_little_endian_ply_xyz(path)

    assert np.array_equal(actual, expected)
    assert not actual.flags.writeable


def test_rollout_sampling_and_edges_are_deterministic() -> None:
    points, faces = _grid()
    rollout = np.stack([points + index for index in range(9)])

    sampled, indices = sample_physical_rollout(rollout, 5)
    edges = mesh_edges_from_faces(faces, len(points))

    assert np.array_equal(indices, np.asarray([0, 2, 4, 6, 8]))
    assert np.array_equal(sampled, rollout[indices])
    assert np.array_equal(
        edges,
        np.asarray([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]),
    )


def test_prior_reliability_does_not_use_state_innovation() -> None:
    points, _ = _grid()
    cloud = np.repeat(points, 4, axis=0)

    near = associate_dense_cloud(points, cloud)
    far = associate_dense_cloud(points + np.asarray([10.0, -2.0, 3.0]), cloud)

    assert np.array_equal(near.prior_reliability, np.ones(len(points)))
    assert np.array_equal(far.prior_reliability, near.prior_reliability)


def test_duplicate_correlated_cloud_does_not_remove_bias_floor() -> None:
    points, faces = _grid()
    translation = np.asarray([0.02, -0.01, 0.005])
    physical_fit = np.stack([points, points, points])
    observed = [
        np.repeat(points + translation, repeats, axis=0)
        for repeats in (1, 4, 16)
    ]
    validation = np.stack([points, points])
    validation_clouds = [points + translation, points + translation]
    config = ClothReadoutBeliefConfig(
        graph_prior_strengths=(1.0,),
        correction_scales=(1.0,),
        shared_bias_std_m=0.005,
        covariance_probes=4,
        minimum_validation_improvement=0.0,
    )

    belief = fit_guarded_readout_correction(
        physical_fit,
        observed,
        validation,
        validation_clouds,
        faces,
        config=config,
    )

    assert belief.accepted
    assert np.all(belief.variance_m2 >= config.shared_bias_std_m**2)
    assert belief.diagnostics["shared_bias_floor_preserved"] is True


def test_guard_admits_transferable_correction() -> None:
    points, faces = _grid()
    translation = np.asarray([0.02, -0.01, 0.005])
    physical_fit = np.stack([points, points, points])
    observed_fit = [points + translation] * 3
    physical_validation = np.stack([points, points])
    observed_validation = [points + translation] * 2
    config = ClothReadoutBeliefConfig(
        graph_prior_strengths=(1.0,),
        correction_scales=(1.0,),
        covariance_probes=4,
        minimum_validation_improvement=0.01,
    )

    belief = fit_guarded_readout_correction(
        physical_fit,
        observed_fit,
        physical_validation,
        observed_validation,
        faces,
        config=config,
    )
    corrected = apply_guarded_readout_correction(
        physical_validation,
        belief,
    )

    assert belief.accepted
    assert belief.selected_name != "baseline"
    assert symmetric_l1_chamfer_m(corrected[0], observed_validation[0]) < (
        symmetric_l1_chamfer_m(points, observed_validation[0])
    )


def test_robust_innovation_rejects_one_gross_fit_outlier() -> None:
    points, faces = _grid()
    translation = np.asarray([0.02, -0.01, 0.005])
    physical_fit = np.stack([points] * 5)
    observed_fit = [np.repeat(points + translation, 4, axis=0)] * 4 + [
        np.repeat(points + np.asarray([0.4, -0.3, 0.2]), 4, axis=0)
    ]
    physical_validation = np.stack([points, points])
    observed_validation = [np.repeat(points + translation, 4, axis=0)] * 2
    config = ClothReadoutBeliefConfig(
        graph_prior_strengths=(1.0,),
        correction_scales=(1.0,),
        covariance_probes=4,
        minimum_validation_improvement=0.01,
    )

    belief = fit_guarded_readout_correction(
        physical_fit,
        observed_fit,
        physical_validation,
        observed_validation,
        faces,
        config=config,
    )

    assert belief.accepted
    assert belief.diagnostics["minimum_posterior_inlier_probability"] < 0.01
    assert belief.diagnostics["prior_reliability_uses_state_innovation"] is False
    assert np.linalg.norm(
        np.median(belief.correction_m, axis=0) - translation
    ) < 0.01


def test_failed_guard_is_exact_physical_fallback() -> None:
    points, faces = _grid()
    physical_fit = np.stack([points, points, points])
    observed_fit = [points + np.asarray([0.04, 0.0, 0.0])] * 3
    physical_validation = np.stack([points, points])
    observed_validation = [points, points]
    config = ClothReadoutBeliefConfig(
        graph_prior_strengths=(1.0,),
        correction_scales=(1.0,),
        covariance_probes=4,
    )

    belief = fit_guarded_readout_correction(
        physical_fit,
        observed_fit,
        physical_validation,
        observed_validation,
        faces,
        config=config,
    )
    corrected = apply_guarded_readout_correction(
        physical_validation,
        belief,
    )

    assert not belief.accepted
    assert belief.selected_name == "baseline"
    assert corrected is physical_validation
    assert np.array_equal(corrected, physical_validation)
