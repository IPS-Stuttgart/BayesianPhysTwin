"""Numerical and information-boundary tests; synthetic fixtures are not evidence."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm, t

spec = importlib.util.spec_from_file_location("query_capsule", Path(__file__).with_name("run.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(n=32):
    rng = np.random.default_rng(75)
    f = rng.normal(size=(n, 54))
    y = .01*(f[:, :4]@rng.normal(size=(4, 24)) + rng.normal(size=(n, 24)))
    return m.fit_model(f, y), f, y


@pytest.mark.parametrize("df", [5., 35.])
def test_student_crps_matches_integrated_cdf(df):
    d = dict(kind="student", scale=np.array([.7]), df=df)
    error = np.array([1.2])
    expected = quad(lambda x: t.cdf(x/.7, df)**2, -np.inf, 1.2)[0]
    expected += quad(lambda x: (1-t.cdf(x/.7, df))**2, 1.2, np.inf)[0]
    assert np.allclose(m.crps(d, error), expected, atol=1e-8)


def test_normal_nll_and_crps():
    d = dict(kind="normal", scale=np.array([.3]))
    assert np.allclose(m.nll(d, np.array([.2])), -norm.logpdf(.2, scale=.3))
    assert np.allclose(m.crps(d, np.array([0.])), .3*(np.sqrt(2)-1)/np.sqrt(np.pi))


def test_mixture_crps_matches_integrated_cdf():
    d = dict(kind="mixture", locations=np.array([[-.8, .8]]), scale=np.array([.2]))
    error = np.array([.6])
    expected = quad(lambda x: m.cdf(d, x)[0]**2, -3, .6)[0]
    expected += quad(lambda x: (1-m.cdf(d, x)[0])**2, .6, 3)[0]
    assert np.allclose(m.crps(d, error), expected, atol=1e-8)
    assert np.allclose(m.cdf(d, 0), .5)


def test_distribution_mean_and_covariance_parity():
    model, f, _ = fixture()
    _, q = m.queries()
    a = m.make_distribution(model, f[0], q, "posterior_student", temperature=2.)
    b = m.make_distribution(model, f[0], q, "same_covariance_gaussian", temperature=2.)
    assert np.allclose(a["scale"]**2*a["df"]/(a["df"]-2), b["scale"]**2)
    for arm in ("global_residual_bootstrap", "local_residual_bootstrap"):
        d = m.make_distribution(model, f[0], q, arm)
        assert np.max(np.abs(d["locations"].mean(1))) < 1e-14
        assert np.allclose(m.cdf(d, 0), .5)


def test_future_internal_replacement_cannot_change_inputs():
    rng = np.random.default_rng(42)
    trajectory = rng.normal(size=(500, 12, 3))
    altered = trajectory.copy()
    altered[101:, 2:10] = rng.normal(size=altered[101:, 2:10].shape)*1e6
    for a, b in zip(m.inputs(trajectory, 100, 20), m.inputs(altered, 100, 20), strict=True):
        assert np.array_equal(a, b)


def test_held_queries_not_development_duplicates():
    dev, held = m.queries()
    assert dev.shape == (6, 24) and held.shape == (12, 24)
    assert not any(np.array_equal(a, b) for a in dev for b in held)


def test_source_only_partition(tmp_path):
    for dlo in m.CONFIG["dlos"]:
        train = tmp_path/dlo/"train"
        train.mkdir(parents=True)
        for i in range(56):
            (train/f"case{i:02}.pkl").write_bytes(b"fixture")
        fit, cal, held = m.split_paths(tmp_path, dlo)
        assert [len(fit), len(cal), len(held)] == [32, 12, 12]
        assert len(set(fit+cal+held)) == 56
        assert (fit, cal, held) == m.split_paths(tmp_path, dlo)
    with pytest.raises(ValueError, match="outside"):
        m.load(tmp_path/"DLO4"/"eval"/"case.pkl")


def test_calibration_matches_moment_comparator_temperature():
    model, f, y = fixture(8)
    dev, held = m.queries()
    errors = np.stack([yy-m.transform(model, ff)[0]@model["w"] for ff, yy in zip(f, y, strict=True)])
    setting = m.calibrate(model, f, errors, dev)
    assert setting["posterior_student"] == setting["same_covariance_gaussian"]
    for arm in m.ARMS:
        d = m.make_distribution(model, f[0], held, arm, **setting[arm])
        result = m.scores(d, errors[0]@held.T)
        assert all(np.isfinite(v).all() for v in result.values())
        assert (result["crps_m"] >= -1e-10).all()
        assert (result["width90_m"] > 0).all()


def test_conjugate_posterior_matches_precision_update():
    model, f, _ = fixture()
    x, s2 = m.transform(model, f[0])
    assert s2 > 0 and x@model["v"]@x > 0
    assert model["df"] == 35
    assert np.linalg.eigvalsh(model["psi"]).min() > 0
