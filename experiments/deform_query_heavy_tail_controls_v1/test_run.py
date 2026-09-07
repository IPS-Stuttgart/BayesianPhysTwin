"""Synthetic numerical fixtures; not scientific real-data evidence."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm, t

spec = importlib.util.spec_from_file_location("tail_followup", Path(__file__).with_name("run.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
parent_path = Path(os.environ.get("PARENT_SCRIPT", "experiments/deform_conditional_query_posterior_v1/run.py"))
p = m.load_parent_module(parent_path)


def fixture():
    rng = np.random.default_rng(13)
    f = rng.normal(size=(8, 54))
    y = .015*(f[:, :3] @ rng.normal(size=(3, 24)) + rng.normal(size=(8, 24)))
    return p.fit_model(f, y), rng.normal(size=(12, 54)), rng.normal(size=(12, 24)) * .03


@pytest.mark.parametrize("df", [3., 4., 11., 35., 128., None])
def test_distribution_variance_not_student_scale(df):
    variance = np.array([.01, .03, .07])
    d = m.distribution(variance, df, 4.)
    actual = t.var(df, scale=d["scale"]) if df else norm.var(scale=d["scale"])
    assert np.allclose(actual, 4*variance)


def test_vectorized_grid_matches_independent_scipy():
    rng = np.random.default_rng(31)
    variance = rng.uniform(.01, .1, (5, 12, 6))
    errors = rng.normal(size=(12, 6))*.1
    losses, specs = m.candidate_losses(variance, errors)
    for value, (df, i, temp) in zip(losses, specs, strict=True):
        d = m.distribution(variance[i], df, temp)
        expected = -(t.logpdf(errors, df, scale=d["scale"]) if df else norm.logpdf(errors, scale=d["scale"])).mean()
        assert abs(expected-value) < 1e-12


def test_empirical_covariance_psd_and_marginal_parity():
    rng = np.random.default_rng(12)
    residual = rng.normal(size=(8, 24))
    diagonal = None
    for s in m.CONFIG["shrinkage_grid"]:
        c = m.empirical_covariance(residual, s)
        assert np.linalg.eigvalsh(c).min() > 0
        if diagonal is None:
            diagonal = c.diagonal()
        assert np.allclose(c.diagonal(), diagonal)


def test_sandwich_not_posterior_covariance():
    model, features, _ = fixture()
    for f in features:
        x, noise = p.transform(model, f)
        h_bayes = x@model["v"]@x
        h_freq = m.multiplier(p, model, f, "sandwich_student_conditional")-noise
        assert 0 <= h_freq <= h_bayes + 1e-12
        # Ridge penalty explains precisely the posterior/frequentist difference.
        assert np.isclose(h_bayes-h_freq, model["ridge"]*(x@model["v"]@model["v"]@x))


@pytest.mark.parametrize("arm", m.NEW_ARMS)
def test_calibration_only_selects_registered_parameters(arm):
    model, features, errors = fixture()
    dev, held = p.queries()
    settings = m.calibrate(p, model, features, errors, dev, arm)
    assert settings["df"] in m.CONFIG["df_grid"]
    assert settings["temperature"] in m.CONFIG["temperature_grid"]
    assert settings["shrinkage"] in m.CONFIG["shrinkage_grid"]
    score = []
    for f, error in zip(features, errors, strict=True):
        score.append(p.nll(m.predict(p, model, f, dev, arm, settings), error@dev.T).mean())
        d = m.predict(p, model, f, held, arm, settings)
        assert np.allclose(p.cdf(d, 0), .5)
        assert all(np.isfinite(v).all() for v in p.scores(d, error@held.T).values())
    assert abs(np.mean(score)-settings["calibration_nll"]) < 1e-12


def test_future_outcome_poisoning_cannot_change_control_prediction():
    model, f, errors = fixture()
    dev, held = p.queries()
    rng = np.random.default_rng(122)
    trajectory = rng.normal(size=(500, 12, 3))
    altered = trajectory.copy()
    altered[101:, 2:10] *= 1e8
    a = p.inputs(trajectory, 100, 20)[0]
    b = p.inputs(altered, 100, 20)[0]
    assert np.array_equal(a, b)
    for arm in m.NEW_ARMS:
        setting = m.calibrate(p, model, f, errors, dev, arm)
        da = m.predict(p, model, a, held, arm, setting)
        db = m.predict(p, model, b, held, arm, setting)
        assert np.array_equal(da["scale"], db["scale"])


def test_nonfinite_and_invalid_variance_fail_closed():
    for variance in (np.array([0.]), np.array([-1.]), np.array([np.nan])):
        with pytest.raises(ValueError):
            m.distribution(variance, 5., 1.)
    with pytest.raises(ValueError):
        m.distribution(np.array([1.]), 2., 1.)


def test_archive_integrity_required(tmp_path):
    archive = tmp_path/"invalid.zip"
    archive.write_bytes(b"bad bytes")
    with pytest.raises(ValueError, match="hash"):
        m.unpack(archive, tmp_path/"unpacked")


def test_official_eval_partition_is_not_an_option(tmp_path):
    with pytest.raises(ValueError, match="authorized"):
        m.checked_records(p, tmp_path, {}, "DLO4", "eval")


def test_aggregates_use_trajectories_not_windows():
    rows = []
    for size in (8, 16, 32):
        for dlo in ("DLO4", "DLO5"):
            for record in range(12):
                for panel in range(15):
                    for arm in p.ARMS + m.NEW_ARMS:
                        v = 1. if arm == "posterior_student" else 2.
                        rows.append({"source_size": size, "dlo": dlo, "trajectory": str(record), "arm": arm,
                                     **dict.fromkeys(m.METRICS, v)})
    result = m.summarize(rows, p.ARMS)
    assert all(result["superiority_condition_by_source_size"].values())
    assert result["posterior_minus_comparator"]["8"]["empirical_student_conditional"]["nll"]["difference"] == -1
    with pytest.raises(ValueError, match="Missing or duplicate"):
        m.summarize(rows[:-1], p.ARMS)
