from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments import deform_predictive_coupling as coupling
from bayesian_phystwin_experiments import deform_sparse_observation_budget as budget


def _config() -> coupling.CouplingConfig:
    base = budget.BudgetConfig(
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
    return coupling.CouplingConfig(
        budget=base,
        expected_trajectory_count=6,
        guard_random_repetitions=1,
        bootstrap_replicates=128,
    )


def _cases(
    config: coupling.CouplingConfig,
    *,
    zero_prefix: bool = False,
    dtype: type = np.float64,
) -> list[coupling.TrainingCase]:
    b = config.budget
    count = b.forecast_end_exclusive - b.dataset_frame_offset
    start = b.prefix_end_exclusive - b.dataset_frame_offset
    times = np.arange(count)[:, None]
    nodes = np.arange(12)[None]
    reference = np.zeros((count, 12, 3), dtype=dtype)
    reference[..., 0] = nodes * 0.012
    reference[..., 1] = np.sin(nodes / 3) * 0.04 + times * 0.0003
    reference[..., 2] = np.cos(nodes / 4 + times / 10) * 0.005
    variance = np.full(reference.shape, 0.008**2)
    variance[:, [0, 1, 10, 11]] = 0
    spatial = np.zeros(12)
    spatial[2:10] = np.sin(np.pi * np.arange(1, 9) / 9)
    local = (
        np.linspace(0.7, 1.4, count)[:, None, None]
        * spatial[None, :, None]
        * np.array([0.003, -0.004, 0.002])
    )
    if zero_prefix:
        local[:start] = 0
    world = np.einsum("tnc,dc->tnd", local, coupling.action_frame(reference))
    out = []
    for index, amplitude in enumerate((-2.0, -1.0, 1.0, 2.0, -0.5, 0.75)):
        truth = (reference + amplitude * world).astype(dtype)
        case = coupling.Case(
            f"{103 + index}.pkl",
            reference.copy(),
            variance.copy(),
            truth[:start].copy(),
        )
        out.append(coupling.TrainingCase(case, truth))
    return out


def _write_configs(root: Path, config: coupling.CouplingConfig) -> Path:
    base_path = root / "base.json"
    budget.write_json(
        base_path,
        {
            **asdict(config.budget),
            "schema": "deform-sparse-observation-budget-dev-v1",
            "case_selection": "lexicographically-first-already-open-dlo2-v7-case",
            "scope": "one-already-open-trajectory-exploratory-only",
            "fresh_confirmation_authorized": False,
            "primary_metric": "hidden-future-mean-coordinate-l1-mm",
            "conditions": budget.CONDITIONS,
            "policies": budget.POLICIES,
        },
    )
    raw = asdict(config)
    raw.pop("budget")
    path = root / "coupling.json"
    budget.write_json(
        path,
        {
            **raw,
            "schema": "deform-predictive-coupling-dev-v1",
            "scope": "exploratory-whole-trajectory-crossfit-already-open-dlo2",
            "fresh_confirmation_authorized": False,
            "base_budget_config": str(base_path),
            "source_archive_sha256": config.budget.source_archive_sha256,
            "reported_holdout_count": config.expected_trajectory_count - 1,
            "methods": coupling.METHODS,
            "policies": coupling.POLICIES,
        },
    )
    return path


@pytest.mark.parametrize(
    "change",
    [
        {"floor_fraction": 0.0},
        {"floor_fraction": 1.0},
        {"floor_fraction": np.nan},
        {"design_case": "104.pkl"},
        {"expected_trajectory_count": 4},
        {"expected_trajectory_count": True},
        {"guard_blends": ()},
        {"guard_blends": (0.0, 1.0, 0.5)},
        {"guard_blends": (0.0, np.nan, 1.0)},
        {"guard_random_repetitions": 3},
        {"guard_minimum_mean_improvement": -0.1},
        {"guard_minimum_joint_win_fraction": 0.0},
        {"guard_maximum_case_ratio": np.nan},
        {"bootstrap_replicates": 0},
        {"bootstrap_seed": -1},
    ],
)
def test_config_rejects_invalid_scope_or_guard(change: dict) -> None:
    with pytest.raises(ValueError):
        replace(_config(), **change)


def test_config_roundtrip_and_frozen_comparison(tmp_path: Path) -> None:
    config = _config()
    path = _write_configs(tmp_path, config)
    assert coupling.load_coupling_config(path, tmp_path) == config
    for key, value in (
        ("fresh_confirmation_authorized", True),
        ("reported_holdout_count", 6),
        ("source_archive_sha256", "1" * 64),
        ("methods", ["empirical_floor"]),
        ("policies", ["future_query"]),
    ):
        raw = json.loads(path.read_text())
        raw[key] = value
        changed = tmp_path / (key + ".json")
        budget.write_json(changed, raw)
        with pytest.raises(ValueError):
            coupling.load_coupling_config(changed, tmp_path)


def test_case_and_source_alignment_are_required() -> None:
    config = _config()
    source = _cases(config)[0]
    for case in (
        replace(source.case, name="../103.pkl"),
        replace(source.case, prefix=source.truth),
        replace(source.case, reference=source.case.reference.astype(int)),
        replace(source.case, variance=-source.case.variance),
        replace(source.case, reference=source.case.reference[:, :11]),
    ):
        with pytest.raises(ValueError):
            case.validate(config)
    with pytest.raises(ValueError, match="prefix disagrees"):
        replace(source, truth=source.truth + 1).residual(config)


def test_whole_trajectory_exclusion_and_sorted_source_identity() -> None:
    config = _config()
    cases = _cases(config)
    sources, held = cases[:-1], cases[-1].case
    first = coupling.fit_coupling(sources, held, config, floor_fraction=0.5)
    second = coupling.fit_coupling(sources[::-1], held, config, floor_fraction=0.5)
    assert first.source_names == tuple(sorted(s.case.name for s in sources))
    np.testing.assert_array_equal(first.future_factors, second.future_factors)
    np.testing.assert_array_equal(first.observation_factors, second.observation_factors)
    with pytest.raises(ValueError, match="excluding the holdout"):
        coupling.fit_coupling(cases, held, config, floor_fraction=0.5)
    with pytest.raises(ValueError, match="distinct"):
        coupling.fit_coupling(sources + sources[:1], held, config, floor_fraction=0.5)
    with pytest.raises(ValueError, match="design case"):
        coupling.predict_fold(cases[1:], cases[0].case, config)
    with pytest.raises(ValueError, match="denominator"):
        coupling.predict_fold(cases[:3], held, config)


@pytest.mark.parametrize("floor", [0.0, 0.5])
def test_known_prefix_future_coupling_reduces_hidden_future_error(floor: float) -> None:
    config = _config()
    cases = _cases(config)
    held = cases[-1]
    before = [budget.array_sha256(c.truth) for c in cases]
    model = coupling.fit_coupling(cases[:-1], held.case, config, floor_fraction=floor)
    order = coupling.latest_uniform_order(config)[:4]
    pool = held.case.prefix[
        model.frames - config.budget.dataset_frame_offset, model.nodes
    ]
    prediction, covariance = coupling.condition(model, config, order, pool[order])
    truth = held.truth[-len(prediction) :, config.budget.hidden_nodes]
    base = coupling.point_errors(model.reference_future, truth)
    candidate = coupling.point_errors(prediction, truth)
    assert np.max(np.array(candidate) / base) < 0.3
    assert covariance is not None
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-12
    assert np.min(np.linalg.eigvalsh(covariance - model.future_floor)) >= -1e-12
    if floor:
        assert np.max(np.trace(model.future_floor, axis1=-2, axis2=-1)) > 0
    assert [budget.array_sha256(c.truth) for c in cases] == before


def test_covariance_floor_preserves_prior_but_cannot_be_conditioned_away() -> None:
    config = _config()
    cases = _cases(config)
    raw = coupling.fit_coupling(cases[:-1], cases[-1].case, config, floor_fraction=0)
    floor = coupling.fit_coupling(
        cases[:-1], cases[-1].case, config, floor_fraction=0.5
    )
    empty = np.array([], dtype=int)
    _, raw_prior = coupling.condition(raw, config, empty, np.empty((0, 3)))
    _, floor_prior = coupling.condition(floor, config, empty, np.empty((0, 3)))
    np.testing.assert_allclose(raw_prior, floor_prior, atol=1e-18)
    selected = coupling.latest_uniform_order(config)
    pool = cases[-1].case.prefix[
        floor.frames - config.budget.dataset_frame_offset, floor.nodes
    ]
    _, raw_after = coupling.condition(raw, config, selected, pool[selected])
    _, floor_after = coupling.condition(floor, config, selected, pool[selected])
    assert floor_after is not None and raw_after is not None
    assert np.min(np.linalg.eigvalsh(floor_after - floor.future_floor)) >= -1e-12
    assert np.mean(np.trace(floor_after, axis1=-2, axis2=-1)) > 10 * np.mean(
        np.trace(raw_after, axis1=-2, axis2=-1)
    )


def test_permuted_control_preserves_marginals_and_destroys_known_coupling() -> None:
    config = _config()
    cases = _cases(config)
    regular = coupling.fit_coupling(
        cases[:-1], cases[-1].case, config, floor_fraction=0.5
    )
    placebo = coupling.fit_coupling(
        cases[:-1], cases[-1].case, config, floor_fraction=0.5, permute=True
    )
    np.testing.assert_allclose(regular.future_floor, placebo.future_floor, atol=1e-18)
    np.testing.assert_array_equal(
        regular.observation_factors, placebo.observation_factors
    )
    np.testing.assert_allclose(
        np.sum(regular.future_factors**2, axis=-1),
        np.sum(placebo.future_factors**2, axis=-1),
        atol=1e-18,
    )
    selected = coupling.latest_uniform_order(config)[:4]
    pool = cases[-1].case.prefix[
        regular.frames - config.budget.dataset_frame_offset, regular.nodes
    ]
    first, _ = coupling.condition(regular, config, selected, pool[selected])
    second, _ = coupling.condition(placebo, config, selected, pool[selected])
    truth = cases[-1].truth[-len(first) :, config.budget.hidden_nodes]
    assert (
        coupling.point_errors(first, truth)[1]
        < coupling.point_errors(second, truth)[1] / 2
    )


@pytest.mark.parametrize("policy", coupling.POLICIES)
@pytest.mark.parametrize("zero_prefix", [False, True])
def test_all_policies_are_unique_nested_deterministic_and_prefix_only(
    policy: str, zero_prefix: bool
) -> None:
    config = _config()
    cases = _cases(config, zero_prefix=zero_prefix)
    model = coupling.fit_coupling(
        cases[:-1], cases[-1].case, config, floor_fraction=0.5
    )
    order = coupling.choose_order(model, cases[-1].case, config, policy, seed=7)
    again = coupling.choose_order(model, cases[-1].case, config, policy, seed=7)
    np.testing.assert_array_equal(order, again)
    assert len(order) == len(set(order)) == config.budget.budgets[-1]
    assert np.all(model.frames[order] < config.budget.prefix_end_exclusive)
    assert not set(model.nodes[order]) & set(config.budget.hidden_nodes)
    for smaller, larger in zip(
        config.budget.budgets[:-1], config.budget.budgets[1:], strict=True
    ):
        assert set(order[:smaller]) <= set(order[:larger])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_budget_and_declined_update_are_byte_exact(dtype: type) -> None:
    config = _config()
    cases = _cases(config, dtype=dtype)
    model = coupling.fit_coupling(
        cases[:-1], cases[-1].case, config, floor_fraction=0.5
    )
    prediction, _ = coupling.condition(
        model, config, np.array([], dtype=int), np.empty((0, 3))
    )
    assert prediction is model.reference_future
    mean, covariance = coupling.mixture_shrinkage(
        model.reference_future,
        model.baseline_covariance,
        prediction + 1,
        model.baseline_covariance * 0.1,
        0,
    )
    assert mean is model.reference_future and covariance is model.baseline_covariance
    assert mean.dtype == dtype
    assert budget.array_sha256(mean) == budget.array_sha256(model.reference_future)
    assert budget.array_sha256(covariance) == budget.array_sha256(
        model.baseline_covariance
    )


def test_partial_shrinkage_keeps_between_mean_covariance() -> None:
    reference = np.zeros((3, 4, 3))
    candidate = np.broadcast_to([0.001, 0.002, 0.003], reference.shape)
    covariance = np.broadcast_to(np.eye(3) * 1e-6, (3, 4, 3, 3))
    mean, mixed = coupling.mixture_shrinkage(
        reference, covariance, candidate, covariance, 0.5
    )
    np.testing.assert_array_equal(mean, candidate / 2)
    np.testing.assert_allclose(
        mixed - covariance, 0.25 * candidate[..., :, None] * candidate[..., None, :]
    )
    assert np.min(np.linalg.eigvalsh(mixed)) > 0


def test_latest_residual_uses_each_identitys_last_selected_causal_frame() -> None:
    config = _config()
    held = _cases(config)[-1].case
    prefix = held.reference[: len(held.prefix)].copy()
    prefix[1, 2, 0] += 0.001
    prefix[3, 2, 0] += 0.004
    prefix[5, 4, 0] += 0.008
    held = replace(held, prefix=prefix)
    prediction = coupling.last_residual(held, config, np.array([1, 7, 14]))
    reference = held.reference[-len(prediction) :, config.budget.hidden_nodes]
    expected = np.interp(
        config.budget.hidden_nodes, [1, 2, 4, 10], [0, 0.004, 0.008, 0]
    )
    np.testing.assert_allclose(
        prediction[..., 0] - reference[..., 0],
        np.broadcast_to(expected, prediction.shape[:2]),
    )
    np.testing.assert_array_equal(prediction[..., 1:], reference[..., 1:])


def test_nested_guard_leaves_out_whole_validation_and_excludes_design_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    cases = _cases(config)
    original = coupling.fit_coupling
    calls = []

    def checked(sources, held, *args, **kwargs):
        names = {s.case.name for s in sources}
        assert held.name not in names
        assert held.name != config.design_case
        assert config.design_case in names
        assert cases[-1].case.name not in names
        calls.append(held.name)
        return original(sources, held, *args, **kwargs)

    monkeypatch.setattr(coupling, "fit_coupling", checked)
    guard, diagnostics = coupling.fit_guard(cases[:-1], config)
    assert calls == [c.case.name for c in cases[1:-1]]
    assert guard[("latest_uniform", 4)] > 0
    assert all(config.design_case not in r["validation_names"] for r in diagnostics)
    assert all(r["validation_count"] == 4 for r in diagnostics)


def test_zero_predictive_coupling_declines_every_guarded_update() -> None:
    config = _config()
    cases = _cases(config, zero_prefix=True)
    records, means, covariances, _ = coupling.predict_fold(
        cases[:-1], cases[-1].case, config
    )
    reference = cases[-1].case.reference[-12:, config.budget.hidden_nodes]
    original_covariance = coupling.diagonal_covariance(
        cases[-1].case.variance[-12:, config.budget.hidden_nodes]
    )
    guarded = 0
    for record, mean, covariance in zip(records, means, covariances, strict=True):
        if record["method"] == "source_guarded_floor":
            assert record["blend"] == 0
            assert budget.array_sha256(mean) == budget.array_sha256(reference)
            assert budget.array_sha256(covariance) == budget.array_sha256(
                original_covariance
            )
            guarded += 1
    assert guarded == (
        config.budget.random_policy_repetitions + len(coupling.POLICIES) - 1
    ) * len(config.budget.budgets)


def test_predictions_ignore_held_future_and_hidden_prefix_and_preserve_inputs() -> None:
    config = _config()
    cases = _cases(config)
    held = cases[-1]
    hashes = [
        (budget.array_sha256(s.truth), budget.array_sha256(s.case.reference))
        for s in cases
    ]
    first = coupling.predict_fold(cases[:-1], held.case, config)
    changed_truth = held.truth.copy()
    changed_truth[len(held.case.prefix) :] += 100
    changed_prefix = held.case.prefix.copy()
    changed_prefix[:, config.budget.hidden_nodes] -= 100
    changed = replace(
        held, truth=changed_truth, case=replace(held.case, prefix=changed_prefix)
    )
    second = coupling.predict_fold(cases[:-1], changed.case, config)
    assert first[0] == second[0] and first[3] == second[3]
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], second[2])
    assert hashes == [
        (budget.array_sha256(s.truth), budget.array_sha256(s.case.reference))
        for s in cases
    ]


def test_action_frame_coupling_is_equivariant_under_rigid_transform() -> None:
    config = _config()
    cases = _cases(config)
    held = cases[-1].case
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    shift = np.array([1.0, -0.5, 0.3])
    moved = replace(
        held,
        reference=held.reference @ rotation.T + shift,
        prefix=held.prefix @ rotation.T + shift,
    )
    original = coupling.fit_coupling(cases[:-1], held, config, floor_fraction=0.5)
    transformed = coupling.fit_coupling(cases[:-1], moved, config, floor_fraction=0.5)
    indices = coupling.latest_uniform_order(config)[:4]
    offsets = original.frames - config.budget.dataset_frame_offset
    first, cov1 = coupling.condition(
        original,
        config,
        indices,
        held.prefix[offsets[indices], original.nodes[indices]],
    )
    second, cov2 = coupling.condition(
        transformed,
        config,
        indices,
        moved.prefix[offsets[indices], original.nodes[indices]],
    )
    np.testing.assert_allclose(second, first @ rotation.T + shift, atol=1e-12)
    np.testing.assert_allclose(cov2, rotation @ cov1 @ rotation.T, atol=1e-12)


def test_metric_formulas_are_point_marginals_in_metric_units() -> None:
    predicted = np.zeros((3, 1, 3))
    truth = np.ones_like(predicted) * 0.002
    covariance = np.broadcast_to(np.eye(3) * 3e-6, (3, 1, 3, 3))
    result = coupling.score(predicted, covariance, truth, 0.001)
    assert result["coordinate_l1_mm"] == pytest.approx(2)
    assert result["point_rmse_mm"] == pytest.approx(np.sqrt(12))
    assert result["point_nees"] == pytest.approx(3)
    assert result["point_coverage_90"] == 1
    assert result["gaussian_nll_per_point"] == pytest.approx(
        0.5 * (3 * np.log(2 * np.pi * 4e-6) + 3)
    )
    assert result["ellipsoid_volume_mm3"] == pytest.approx(
        4 * np.pi / 3 * coupling.ELLIPSOID_90_CHI2_3**1.5 * 8
    )
    for horizon in ("early", "middle", "late"):
        assert result[horizon + "_coordinate_l1_mm"] == pytest.approx(2)


def test_random_schedules_are_averaged_before_trajectory_bootstrap() -> None:
    config = _config()
    truth = np.zeros((3, 1, 3))
    covariance = np.broadcast_to(np.eye(3) * 1e-4, (3, 1, 3, 3))
    rows = []
    for index in range(5):
        name = f"{104 + index}.pkl"
        base = coupling.score(truth + 0.002, covariance, truth, 0.001)
        rows.append(
            {
                "case": name,
                "method": "unchanged_baseline",
                "policy": "none",
                "budget": 0,
                "blend": 1.0,
                **base,
            }
        )
        for repetition in range(index + 1):
            candidate = coupling.score(
                truth + (index + 1) * 0.001, covariance, truth, 0.001
            )
            rows.append(
                {
                    "case": name,
                    "method": "empirical_floor",
                    "policy": "random",
                    "budget": 1,
                    "repetition": repetition,
                    "blend": 1.0,
                    **candidate,
                }
            )
    case_rows, summary = coupling.aggregate_rows(rows, config)
    result = next(row for row in summary if row["method"] == "empirical_floor")
    assert len(case_rows) == 10
    assert result["case_count"] == 5
    assert result["coordinate_l1_mm"] == pytest.approx(3)
    assert result["l1_delta_mm"] == pytest.approx(1)
    assert result["guard_accepted_cases"] is None
    assert coupling.aggregate_rows(rows[::-1], config)[1] == summary
    with pytest.raises(ValueError, match="denominator"):
        coupling.aggregate_rows(
            [row for row in rows if row["case"] != "104.pkl"], config
        )


@pytest.mark.parametrize(
    "selected,values",
    [
        ([-1], [[0.0, 0.0, 0.0]]),
        ([1, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ([18], [[0.0, 0.0, 0.0]]),
        ([1], [[np.nan, 0.0, 0.0]]),
        ([1], [[0.0, 0.0]]),
    ],
)
def test_invalid_or_duplicate_measurements_fail_closed(
    selected: list, values: list
) -> None:
    config = _config()
    cases = _cases(config)
    model = coupling.fit_coupling(
        cases[:-1], cases[-1].case, config, floor_fraction=0.5
    )
    with pytest.raises(ValueError, match="selected observations"):
        coupling.condition(model, config, np.array(selected), np.array(values))


@pytest.mark.parametrize(
    "diagonal", [[-1.0, -1.0, 1.0], [0.0, 0.0, 0.0], [np.nan, 1.0, 1.0]]
)
def test_scorer_rejects_invalid_covariance_even_when_determinant_positive(
    diagonal: list,
) -> None:
    means = np.zeros((3, 1, 3))
    covariance = np.broadcast_to(np.diag(diagonal), (3, 1, 3, 3))
    with pytest.raises(ValueError):
        coupling.score(means, covariance, means, 0)


def test_complete_synthetic_study_seals_all_predictions_before_outer_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    cases = _cases(config)
    archive = tmp_path / "input.npz"
    np.savez_compressed(
        archive,
        names=np.array([c.case.name for c in cases]),
        candidate_predictions=np.stack([c.case.reference for c in cases]),
        coordinate_variance_m2=np.stack([c.case.variance for c in cases]),
        targets=np.stack([c.truth for c in cases]),
    )
    config = replace(
        config,
        budget=replace(
            config.budget, source_archive_sha256=budget.file_sha256(archive)
        ),
    )
    path = _write_configs(tmp_path, config)
    output = tmp_path / "run"
    original = coupling.score
    calls = []

    def checked(*args, **kwargs):
        barrier = json.loads((output / "prediction-barrier.json").read_text())
        assert barrier["case_count"] == 5
        assert barrier["outer_scoring_started"] is False
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(coupling, "score", checked)
    result = coupling.run_study(archive, path, output, require_clean=False)
    assert calls and result["case_count"] == 5
    assert result["physical_object_count"] == 1
    assert result["fresh_confirmation_authorized"] is False
    assert result["point_sota_claim"] is False
    assert len(list(output.glob("*/prediction-seal.json"))) == 5
    assert not (output / "103").exists()
    receipt = json.loads((output / "run-complete.json").read_text())
    assert receipt["results_sha256"] == budget.file_sha256(output / "results.json")
    assert (output / "predictive-coupling.png").stat().st_size > 10000
    manifest = json.loads((output / "input-manifest.json").read_text())
    assert manifest["crossfit_source_future_outcomes_used"] is True
    assert manifest["held_own_future_input_to_predictor"] is False
    assert manifest["held_v8_accessed"] is False
    assert manifest["dlo4_dlo5_accessed"] is False
    for row in result["summaries"]:
        assert row["case_count"] == 5
    with pytest.raises(FileExistsError):
        coupling.run_study(archive, path, output, require_clean=False)
