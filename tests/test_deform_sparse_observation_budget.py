from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from bayesian_phystwin_experiments import deform_sparse_observation_budget as budget


def _config() -> budget.BudgetConfig:
    return budget.BudgetConfig(
        source_archive_sha256="0" * 64,
        case_name="103.pkl",
        dataset_frame_offset=2,
        prefix_end_exclusive=8,
        forecast_end_exclusive=20,
        observation_frames=(3, 5, 7),
        candidate_nodes=(0, 2, 4, 6, 8, 11),
        hidden_nodes=(3, 5, 7, 9),
        budgets=(0, 1, 2, 4, 8),
        graph_rank=4,
        measurement_std_m=0.001,
        shared_bias_std_m=0.005,
        random_policy_repetitions=2,
        bias_repetitions=2,
        seed=260827,
    )


def _arrays(
    config: budget.BudgetConfig, dtype: type = np.float64
) -> tuple[np.ndarray, np.ndarray]:
    count = config.forecast_end_exclusive - config.dataset_frame_offset
    time = np.arange(count)[:, None]
    node = np.arange(12)[None, :]
    mean = np.zeros((count, 12, 3), dtype=dtype)
    mean[..., 0] = node * 0.012
    mean[..., 1] = np.sin(node / 3) * 0.04 + time * 0.0003
    mean[..., 2] = np.cos(node / 4 + time / 10) * 0.005
    variance = np.broadcast_to(
        (0.003 + np.arange(count) * 0.0001)[:, None, None] ** 2,
        mean.shape,
    ).copy()
    variance[:, [0, 1, 10, 11]] = 0
    return mean, variance


def _write_config(path: Path, config: budget.BudgetConfig) -> None:
    budget.write_json(
        path,
        {
            **asdict(config),
            "schema": "deform-sparse-observation-budget-dev-v1",
            "case_selection": "lexicographically-first-already-open-dlo2-v7-case",
            "scope": "one-already-open-trajectory-exploratory-only",
            "fresh_confirmation_authorized": False,
            "primary_metric": "hidden-future-mean-coordinate-l1-mm",
            "conditions": budget.CONDITIONS,
            "policies": budget.POLICIES,
        },
    )


def _experiment(
    root: Path,
    *,
    future_shift: float = 0,
    hidden_prefix_shift: float = 0,
    dtype: type = np.float64,
) -> tuple[Path, Path, budget.BudgetConfig]:
    root.mkdir()
    config = _config()
    mean, variance = _arrays(config, dtype)
    problem = budget.build_problem(mean, variance, config)
    latent = np.linspace(-0.7, 0.7, problem.field_design.shape[-1])
    truth = mean + np.einsum("tncd,d->tnc", problem.field_design, latent)
    start = config.prefix_end_exclusive - config.dataset_frame_offset
    truth[start:, config.hidden_nodes] += future_shift
    truth[:start, config.hidden_nodes] += hidden_prefix_shift
    archive = root / "predictions.npz"
    np.savez_compressed(
        archive,
        names=np.array([config.case_name]),
        candidate_predictions=mean[None],
        coordinate_variance_m2=variance[None],
        targets=truth[None],
    )
    config = replace(config, source_archive_sha256=budget.file_sha256(archive))
    path = root / "config.json"
    _write_config(path, config)
    return archive, path, config


@pytest.mark.parametrize(
    "change",
    [
        {"observation_frames": (3, 5, 8)},
        {"hidden_nodes": (2, 5, 7, 9)},
        {"observation_frames": (3, 3, 7)},
        {"candidate_nodes": (True, 2, 4, 6)},
        {"budgets": (1, 2, 4)},
        {"budgets": (0, 100)},
        {"measurement_std_m": True},
        {"shared_bias_std_m": np.nan},
        {"source_archive_sha256": "not-a-digest"},
    ],
)
def test_config_rejects_changed_information_boundary(change: dict) -> None:
    with pytest.raises(ValueError):
        replace(_config(), **change)


def test_config_roundtrip_and_boundary_label(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write_config(path, _config())
    assert budget.load_config(path) == _config()
    payload = json.loads(path.read_text())
    payload["fresh_confirmation_authorized"] = True
    other = tmp_path / "changed.json"
    budget.write_json(other, payload)
    with pytest.raises(ValueError, match="boundary"):
        budget.load_config(other)


def test_reader_limits_materialized_truth_to_requested_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, _, config = _experiment(tmp_path / "input")
    read_lengths: list[int] = []

    class CountingZip(ZipFile):
        def open(self, *args, **kwargs):
            stream = super().open(*args, **kwargs)
            original_read = stream.read

            def read(count=-1):
                value = original_read(count)
                read_lengths.append(len(value))
                return value

            stream.read = read
            return stream

    with ZipFile(archive) as zipped, zipped.open("targets.npy") as stream:
        np.lib.format.read_magic(stream)
        np.lib.format.read_array_header_1_0(stream)
        header_bytes = stream.tell()
    monkeypatch.setattr(budget, "ZipFile", CountingZip)
    result = budget.read_case_window(archive, "targets", 0, 0, 3)
    assert result.shape == (3, 12, 3)
    assert sum(read_lengths) == header_bytes + result.nbytes
    assert not result.flags.writeable
    with pytest.raises(ValueError, match="layout"):
        budget.read_case_window(archive, "targets", 0, 0, config.forecast_end_exclusive)


def test_field_preserves_marginals_and_compressed_query_objective() -> None:
    config = _config()
    mean, variance = _arrays(config)
    before = budget.array_sha256(mean)
    problem = budget.build_problem(mean, variance, config)
    np.testing.assert_allclose(np.sum(problem.field_design**2, axis=-1), variance)
    full_query = problem.forecast_design[:, config.hidden_nodes].reshape(-1, 12)
    full_query /= np.sqrt(len(full_query))
    np.testing.assert_allclose(
        problem.query_design.T @ problem.query_design,
        full_query.T @ full_query,
        rtol=1e-12,
        atol=1e-18,
    )
    assert problem.query_design.shape == (12, 12)
    assert not set(problem.candidate_nodes) & set(problem.hidden_nodes)
    assert budget.array_sha256(mean) == before
    assert np.all(problem.field_design[:, [0, 1, 10, 11]] == 0)


@pytest.mark.parametrize("policy", budget.POLICIES)
@pytest.mark.parametrize("bias_std_m", [0.0, 0.005])
def test_policies_have_equal_nested_unique_budgets(
    policy: str, bias_std_m: float
) -> None:
    config = _config()
    problem = budget.build_problem(*_arrays(config), config)
    first = budget.selection_order(
        problem, config, policy, bias_std_m=bias_std_m, seed=42
    )
    second = budget.selection_order(
        problem, config, policy, bias_std_m=bias_std_m, seed=42
    )
    assert np.array_equal(first, second)
    assert len(first) == len(set(first)) == config.budgets[-1]
    assert (problem.candidate_frames[first] < config.prefix_end_exclusive).all()
    for smaller, larger in zip(config.budgets[:-1], config.budgets[1:], strict=True):
        assert set(first[:smaller]) <= set(first[:larger])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_budget_is_exact_and_does_not_mutate_reference(dtype: type) -> None:
    config = _config()
    mean, variance = _arrays(config, dtype)
    before = budget.array_sha256(mean)
    problem = budget.build_problem(mean, variance, config)
    result, covariance = budget.condition_forecast(
        problem,
        config,
        np.array([], dtype=int),
        np.empty((0, 3)),
        bias_std_m=0,
    )
    reference = mean[-len(result) :]
    assert result.dtype == reference.dtype
    assert budget.array_sha256(result) == budget.array_sha256(reference)
    assert np.shares_memory(result, mean)
    np.testing.assert_allclose(covariance, variance[-len(result) :])
    assert budget.array_sha256(mean) == before


def test_shared_bias_is_marginalized_and_not_counted_as_independent_noise() -> None:
    config = _config()
    problem = budget.build_problem(*_arrays(config), config)
    selected = np.array([1, 7, 13])
    observed = problem.candidate_means[selected] + np.array([0.005, 0, 0])
    naive, naive_variance = budget.condition_forecast(
        problem, config, selected, observed, bias_std_m=0
    )
    aware, aware_variance = budget.condition_forecast(
        problem, config, selected, observed, bias_std_m=0.005
    )
    reference = problem.reference_mean[-len(aware) :]
    assert np.linalg.norm(aware - reference) < np.linalg.norm(naive - reference)
    assert np.all(aware_variance >= naive_variance - 1e-16)
    assert np.all(aware_variance >= 0)


def test_known_control_anchor_helps_separate_bias_from_field() -> None:
    config = _config()
    problem = budget.build_problem(*_arrays(config), config)
    field_only = np.array([1, 7, 13])
    with_control = np.array([0, 1, 7, 13])
    bias = np.array([0.005, 0, 0])
    first, _ = budget.condition_forecast(
        problem,
        config,
        field_only,
        problem.candidate_means[field_only] + bias,
        bias_std_m=0.005,
    )
    second, _ = budget.condition_forecast(
        problem,
        config,
        with_control,
        problem.candidate_means[with_control] + bias,
        bias_std_m=0.005,
    )
    reference = problem.reference_mean[-len(second) :]
    assert np.linalg.norm(second - reference) < np.linalg.norm(first - reference)


def test_identical_update_reduces_gaussian_variance_without_touching_clamps() -> None:
    config = _config()
    problem = budget.build_problem(*_arrays(config), config)
    order = budget.selection_order(
        problem, config, "future_query", bias_std_m=0, seed=1
    )
    old = np.sum(problem.forecast_design**2, axis=-1)
    for count in config.budgets:
        selected = order[:count]
        forecast, variance = budget.condition_forecast(
            problem,
            config,
            selected,
            problem.candidate_means[selected] + 0.003,
            bias_std_m=0,
        )
        assert np.isfinite(variance).all()
        assert np.all(variance <= old + 1e-16)
        assert np.all(variance >= 0)
        np.testing.assert_array_equal(
            forecast[:, [0, 1, 10, 11]],
            problem.reference_mean[-len(forecast) :, [0, 1, 10, 11]],
        )
        old = variance


def test_duplicate_measurements_and_nonfinite_values_fail() -> None:
    config = _config()
    problem = budget.build_problem(*_arrays(config), config)
    with pytest.raises(ValueError, match="unique"):
        budget.condition_forecast(
            problem, config, np.array([1, 1]), np.zeros((2, 3)), bias_std_m=0
        )
    with pytest.raises(ValueError, match="finite"):
        budget.condition_forecast(
            problem, config, np.array([1]), np.full((1, 3), np.nan), bias_std_m=0
        )


def test_marginal_metrics_have_declared_units() -> None:
    prediction = np.ones((3, 2, 3)) * 0.002
    variance = np.ones_like(prediction) * 4e-6
    metrics = budget._metrics(prediction, np.zeros_like(prediction), variance, 0)
    assert metrics["coordinate_l1_mm"] == pytest.approx(2)
    assert metrics["point_rmse_mm"] == pytest.approx(np.sqrt(12))
    assert metrics["coordinate_nees"] == pytest.approx(1)
    assert metrics["coordinate_coverage_90"] == 1


def test_source_hash_failure_creates_no_output(tmp_path: Path) -> None:
    archive, config_path, config = _experiment(tmp_path / "input")
    bad = tmp_path / "bad.json"
    _write_config(bad, replace(config, source_archive_sha256="1" * 64))
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="digest"):
        budget.run_study(archive, bad, output)
    assert not output.exists()
    assert config_path.is_file()


def test_end_to_end_seals_before_future_and_preserves_float32_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, config_path, config = _experiment(tmp_path / "input", dtype=np.float32)
    output = tmp_path / "run"
    original_reader = budget.read_case_window
    truth_stages: list[str] = []

    def guarded_reader(path, member, case_index, start, stop):
        if member == "targets":
            assert (output / "selection-seal.json").exists()
            if start == 0:
                assert stop == config.prefix_end_exclusive - config.dataset_frame_offset
                assert not (output / "prediction-seal.json").exists()
                truth_stages.append("prefix")
            else:
                assert (output / "prediction-seal.json").exists()
                truth_stages.append("future-score")
        return original_reader(path, member, case_index, start, stop)

    monkeypatch.setattr(budget, "read_case_window", guarded_reader)
    result = budget.run_study(archive, config_path, output)
    assert truth_stages == ["prefix", "future-score"]
    assert result["case_count"] == 1
    assert result["zero_budget_mean_byte_identical"]
    assert not result["future_truth_used_for_selection_or_update"]
    assert not result["fresh_targets_accessed"]
    assert len(result["curves"]) == len(budget.CONDITIONS) * len(budget.POLICIES) * len(
        config.budgets
    )
    assert all(row["budget"] == row["mean_actual_count"] for row in result["curves"])
    with np.load(output / "predictions.npz", allow_pickle=False) as saved:
        reference = original_reader(archive, "candidate_predictions", 0, 6, 18)
        for i, row in enumerate(result["records"]):
            if row["budget"] == 0:
                assert budget.array_sha256(
                    saved["predictions"][i]
                ) == budget.array_sha256(reference)
    complete = json.loads((output / "run-complete.json").read_text())
    for name, digest in complete["files_sha256"].items():
        assert budget.file_sha256(output / name) == digest
    from matplotlib import image

    pixels = image.imread(output / "error-versus-budget.png")
    assert pixels.shape[0] > 200 and pixels.std() > 0.05
    assert (output / "error-versus-budget.pdf").stat().st_size > 1000
    assert (output / "report.md").is_file()
    assert budget.file_sha256(archive) == config.source_archive_sha256
    with pytest.raises(FileExistsError):
        budget.run_study(archive, config_path, output)


def test_hidden_prefix_and_future_truth_cannot_change_plans_or_predictions(
    tmp_path: Path,
) -> None:
    archive_a, config_a, _ = _experiment(tmp_path / "input-a")
    archive_b, config_b, _ = _experiment(
        tmp_path / "input-b", future_shift=1.0, hidden_prefix_shift=2.0
    )
    first = budget.run_study(archive_a, config_a, tmp_path / "run-a")
    second = budget.run_study(archive_b, config_b, tmp_path / "run-b")
    plans = [
        json.loads((tmp_path / name / "selection-seal.json").read_text())["orders"]
        for name in ("run-a", "run-b")
    ]
    assert plans[0] == plans[1]
    with (
        np.load(tmp_path / "run-a/predictions.npz") as a,
        np.load(tmp_path / "run-b/predictions.npz") as b,
    ):
        assert a.files == b.files
        for member in a.files:
            np.testing.assert_array_equal(a[member], b[member])
    assert (
        first["curves"][0]["coordinate_l1_mm"] < second["curves"][0]["coordinate_l1_mm"]
    )
