import numpy as np

from bayesian_phystwin.phystwin_planar_discrepancy import (
    fit_canonical_planar_discrepancy,
)


def _planar_grid() -> np.ndarray:
    u, v = np.meshgrid(
        np.linspace(-0.4, 0.4, 7),
        np.linspace(-0.3, 0.3, 6),
    )
    return np.column_stack((u.reshape(-1), v.reshape(-1), np.zeros(u.size)))


def test_canonical_planar_affine_field_recovers_synthetic_discrepancy() -> None:
    points = _planar_grid()
    target = np.column_stack(
        (
            0.01 + 0.2 * points[:, 0] - 0.1 * points[:, 1],
            -0.02 + 0.05 * points[:, 0],
            0.03 - 0.15 * points[:, 1],
        )
    )

    model = fit_canonical_planar_discrepancy(
        points,
        np.arange(len(points)),
        target,
        np.full(len(points), 1e-6),
        np.ones(len(points), dtype=bool),
        degree=1,
        ridge_strength=0.0,
    )

    np.testing.assert_allclose(model.predict(points), target, atol=1e-10)
    assert model.fit_count == len(points)


def test_canonical_planar_quadratic_field_decodes_unobserved_points() -> None:
    points = _planar_grid()
    observed_ids = np.flatnonzero(
        ~(
            np.isclose(points[:, 0], 0.0, atol=0.08)
            & np.isclose(points[:, 1], 0.0, atol=0.08)
        )
    )
    target = np.column_stack(
        (
            np.square(points[:, 0]),
            points[:, 0] * points[:, 1],
            np.square(points[:, 1]),
        )
    )

    model = fit_canonical_planar_discrepancy(
        points,
        observed_ids,
        target[observed_ids],
        np.full(len(observed_ids), 1e-6),
        np.ones(len(observed_ids), dtype=bool),
        degree=2,
        ridge_strength=0.0,
    )

    np.testing.assert_allclose(model.predict(points), target, atol=1e-10)


def test_canonical_planar_fit_downweights_large_outlier() -> None:
    points = _planar_grid()
    target = np.repeat(np.array([[0.01, -0.02, 0.03]]), len(points), axis=0)
    corrupted = target.copy()
    corrupted[-1] = [2.0, -3.0, 4.0]

    model = fit_canonical_planar_discrepancy(
        points,
        np.arange(len(points)),
        corrupted,
        np.full(len(points), 1e-6),
        np.ones(len(points), dtype=bool),
        degree=0,
    )

    np.testing.assert_allclose(
        np.median(model.predict(points), axis=0),
        target[0],
        atol=1e-4,
    )


def test_canonical_planar_fit_respects_observation_variance() -> None:
    points = _planar_grid()[:8]
    target = np.zeros((len(points), 3), dtype=float)
    target[-1, 0] = 0.5
    variance = np.full(len(points), 1e-6)
    variance[-1] = 1.0

    model = fit_canonical_planar_discrepancy(
        points,
        np.arange(len(points)),
        target,
        variance,
        np.ones(len(points), dtype=bool),
        degree=0,
    )

    assert abs(float(model.predict(points)[0, 0])) < 1e-3
