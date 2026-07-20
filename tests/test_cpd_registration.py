import numpy as np
import pytest

from bayesian_phystwin.cpd_registration import (
    NonrigidCpdConfig,
    fit_nonrigid_cpd,
)
from bayesian_phystwin.deform360_cpd_diagnostic import _symmetric_set_chamfer_m


def _irregular_points() -> np.ndarray:
    rng = np.random.default_rng(20260719)
    return rng.normal(size=(16, 3)) * np.array([0.06, 0.04, 0.02])


def test_nonrigid_cpd_recovers_translation_without_correspondence_order() -> None:
    source = _irregular_points()
    translation = np.array([0.012, -0.007, 0.004])
    permutation = np.random.default_rng(4).permutation(len(source))

    fitted = fit_nonrigid_cpd(source, (source + translation)[permutation])
    predicted = fitted.transform(source)

    assert fitted.converged
    np.testing.assert_allclose(predicted, source + translation, atol=1e-6)


def test_nonrigid_cpd_fit_is_permutation_invariant() -> None:
    source = _irregular_points()
    target = source.copy()
    target[:, 0] += 0.008 * np.tanh(source[:, 1] / 0.02)
    source_order = np.random.default_rng(1).permutation(len(source))
    target_order = np.random.default_rng(2).permutation(len(target))

    canonical = fit_nonrigid_cpd(source, target)
    permuted = fit_nonrigid_cpd(source[source_order], target[target_order])
    queries = np.vstack((source, np.mean(source, axis=0, keepdims=True)))

    np.testing.assert_allclose(
        canonical.transform(queries),
        permuted.transform(queries),
        atol=1e-10,
    )


def test_nonrigid_cpd_rejects_too_few_or_nonfinite_points() -> None:
    with pytest.raises(ValueError, match="M >= 3"):
        fit_nonrigid_cpd(np.zeros((2, 3)), np.zeros((3, 3)))
    invalid = np.zeros((3, 3))
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        fit_nonrigid_cpd(invalid, np.zeros((3, 3)))


def test_nonrigid_cpd_config_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="outlier_weight"):
        NonrigidCpdConfig(outlier_weight=1.0)
    with pytest.raises(ValueError, match="maximum_iterations"):
        NonrigidCpdConfig(maximum_iterations=0)


def test_small_set_chamfer_is_symmetric_and_permutation_invariant() -> None:
    first = _irregular_points()[:5]
    second = first + np.array([0.01, 0.0, 0.0])
    expected = _symmetric_set_chamfer_m(first, second)

    assert expected == pytest.approx(_symmetric_set_chamfer_m(second, first))
    assert expected == pytest.approx(
        _symmetric_set_chamfer_m(first[::-1], second[[2, 4, 1, 0, 3]])
    )
