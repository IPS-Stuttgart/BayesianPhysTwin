import json
import math
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    deform_causal_inputs,
    fit_deform_local_residual,
    predict_deform_local_residual,
)
from bayesian_phystwin_experiments.deform_dlo_pyelastica import (
    deform_pyelastica_directors,
    deform_pyelastica_kinematic_sample,
    deform_pyelastica_parameter_bank,
)
from bayesian_phystwin_experiments.deform_dlo_robustness import (
    DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS,
    assign_deform_dlo3_source_partitions,
    augment_deform_local_residual_full_covariance,
    build_deform_bayesian_covariance_ablation_v1,
    build_deform_dlo3_source_manifest,
    calibrate_deform_full_covariance,
    deform_bayesian_covariance_archive_key,
    deform_local_feature_indices,
    evaluate_deform_backend_source_gate,
    evaluate_deform_dlo3_source_gate,
    evaluate_deform_dlo3_stability_gate,
    evaluate_deform_dlo3_target_gate,
    evaluate_deform_predictive_distribution,
    fit_deform_local_residual_variant,
    load_deform_dlo_robustness_v1_protocol,
    predict_deform_local_residual_full_covariance,
    predict_deform_local_residual_variant,
    scale_deform_coordinate_covariance,
    validate_deform_dlo3_source_manifest,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "sota" / "deform_dlo_robustness_v1.json"
SEED_RUNNER = ROOT / "scripts" / "remote" / "run_deform_dlo3_robustness_seed_v1.py"
STABILITY_RUNNER = (
    ROOT / "scripts" / "remote" / "evaluate_deform_dlo3_stability_gate_v1.py"
)
SENSITIVITY_RUNNER = ROOT / "scripts" / "remote" / "run_deform_dlo3_sensitivity_v1.py"
PYELASTICA_RUNNER = (
    ROOT / "scripts" / "remote" / "run_deform_dlo3_pyelastica_source_v1.py"
)
ALLTRAIN_RUNNER = (
    ROOT / "scripts" / "remote" / "run_deform_dlo3_robustness_alltrain_v1.py"
)
EVALUATOR_RUNNER = (
    ROOT / "scripts" / "remote" / "run_deform_dlo3_robustness_evaluator_v1.py"
)
READINESS_RUNNER = (
    ROOT / "scripts" / "remote" / "attest_deform_dlo3_robustness_readiness_v1.py"
)


def _payload() -> dict[str, object]:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def _residual_problem(count: int = 12) -> tuple[np.ndarray, ...]:
    frames = 14
    nodes = 7
    trajectories = np.zeros((count, frames, nodes, 3), dtype=np.float64)
    arc = np.linspace(-1.0, 1.0, nodes)
    for case in range(count):
        amplitude = 0.01 + 0.001 * case
        for frame in range(frames):
            phase = frame / (frames - 1)
            trajectories[case, frame, :, 0] = arc
            trajectories[case, frame, :, 1] = amplitude * phase * (1.0 - arc**2)
            trajectories[case, frame, :, 2] = 0.002 * case * phase
            trajectories[case, frame, :2, 1] += amplitude * phase
            trajectories[case, frame, -2:, 1] -= 0.5 * amplitude * phase
    initial, action = deform_causal_inputs(trajectories)
    targets = trajectories[:, 2:].copy()
    baseline = targets.copy()
    time = np.linspace(0.0, 1.0, targets.shape[1])
    bias = (0.004 + 0.0005 * np.arange(count))[:, None, None]
    baseline[:, :, 2:-2, 1] -= bias * time[None, :, None]
    baseline[:, :, 2:-2, 2] += 0.25 * bias * np.square(time)[None, :, None]
    names = np.asarray([f"case-{index:02d}" for index in range(count)])
    return initial, action, baseline, targets, names


def _seed_result(
    seed: int, *, ratio: float = 0.90, passed: bool = True
) -> dict[str, object]:
    cases = [
        {
            "name": f"case-{index}",
            "candidate_to_baseline_ratio": ratio,
        }
        for index in range(8)
    ]
    return {
        "contract": "deform-dlo3-robustness-seed-result-v1",
        "seed": seed,
        "protocol": {"sha256": "a" * 64},
        "source_manifest": {"sha256": "b" * 64},
        "primary_source_gate": {
            "contract": "deform-dlo3-robustness-source-gate-v1",
            "case_count": 8,
            "candidate_mean_l1_m": ratio * 0.008,
            "baseline_mean_l1_m": 0.008,
            "maximum_case_ratio": ratio,
            "passed": passed,
            "cases": cases,
        },
        "source_test_opened": True,
        "primary_eval_enumerated": False,
        "primary_eval_read": False,
        "target_authorized": False,
        "retry_authorized": False,
        "prob4d_used": False,
        "held_v8_access": False,
    }


def test_loads_locked_dlo_robustness_protocol() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)

    assert protocol["prob4d_used"] is False
    assert protocol["freshness"]["primary_dlo"] == "DLO3"
    assert protocol["custody"]["held_v8_access"] is False


def test_pyelastica_bank_and_geometry_are_frozen() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    bank = deform_pyelastica_parameter_bank(protocol)
    assert len(bank) == 36
    assert bank[0].to_record() == {
        "youngs_modulus_pa": 1e5,
        "density_kg_m3": 900.0,
        "damping_constant": 0.1,
        "integration_substeps": 2,
    }
    assert bank[-1].integration_substeps == 8

    parameter = np.linspace(0.0, 1.0, 12)
    positions = np.column_stack((parameter, 0.1 * parameter**2, 0.05 * parameter))
    directors = deform_pyelastica_directors(positions)
    assert directors.shape == (3, 3, 11)
    assert np.allclose(
        np.einsum("ain,bin->abn", directors, directors),
        np.repeat(np.eye(3)[:, :, None], 11, axis=2),
        atol=1e-12,
    )


def test_pyelastica_kinematic_interpolation_is_causal_and_metric() -> None:
    series = np.zeros((3, 4, 3), dtype=np.float64)
    series[1, :, 0] = 0.01
    series[2, :, 0] = 0.03

    position, velocity = deform_pyelastica_kinematic_sample(series, 0.015)

    assert np.allclose(position[:, 0], 0.02)
    assert np.allclose(velocity[:, 0], 2.0)
    assert np.allclose(position[:, 1:], 0.0)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("freshness", "primary_dlo"), "DLO4", "data boundary"),
        (("physical_training", "primary_seed"), 43, "fixed recipe"),
        (("local_residual", "shrinkage"), 0.5, "fixed recipe"),
        (("source_gate", "minimum_case_wins"), 5, "source gates"),
        (("backend_portability", "version"), "latest", "backend contract"),
        (("target_evaluation", "target_retries"), True, "Bayesian or target"),
        (("custody", "held_v8_access"), True, "Bayesian or target"),
    ],
)
def test_rejects_protocol_mutation(
    tmp_path: Path,
    path: tuple[str, str],
    value: object,
    match: str,
) -> None:
    payload = _payload()
    payload[path[0]][path[1]] = value
    mutated = tmp_path / "protocol.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_deform_dlo_robustness_v1_protocol(mutated)


def test_source_assignment_is_order_independent_and_disjoint() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    names = [f"trajectory_{index:02d}.pkl" for index in range(56)]

    forward = assign_deform_dlo3_source_partitions(names, protocol)
    reverse = assign_deform_dlo3_source_partitions(list(reversed(names)), protocol)

    assert forward == reverse
    assert forward["payload_read"] is False
    fit = set(forward["fit"])
    calibration = set(forward["calibration"])
    source_test = set(forward["source_test"])
    assert (len(fit), len(calibration), len(source_test)) == (39, 9, 8)
    assert fit.isdisjoint(calibration | source_test)
    assert calibration.isdisjoint(source_test)
    assert fit | calibration | source_test == set(names)


def test_source_assignment_rejects_non_basename_or_wrong_count() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    names = [f"trajectory_{index:02d}.pkl" for index in range(56)]

    with pytest.raises(ValueError, match="incomplete"):
        assign_deform_dlo3_source_partitions(names[:-1], protocol)
    names[0] = "nested/trajectory_00.pkl"
    with pytest.raises(ValueError, match="basename"):
        assign_deform_dlo3_source_partitions(names, protocol)


def test_builds_and_revalidates_source_manifest_without_deserialization(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data_set"
    train_root = data_root / "DLO3" / "train"
    train_root.mkdir(parents=True)
    for index in range(56):
        (train_root / f"trajectory_{index:02d}.pkl").write_bytes(
            f"opaque-{index}".encode()
        )

    manifest = build_deform_dlo3_source_manifest(PROTOCOL, data_root)
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    partitions = validate_deform_dlo3_source_manifest(
        manifest,
        protocol,
        protocol_sha256=sha256_file(PROTOCOL),
        verify_files=True,
    )

    assert tuple(len(partitions[name]) for name in partitions) == (39, 9, 8)
    assert manifest["trajectory_deserialized"] is False
    assert manifest["primary_eval_enumerated"] is False
    assert not (data_root / "DLO3" / "eval").exists()


def test_source_manifest_detects_byte_or_partition_change(tmp_path: Path) -> None:
    data_root = tmp_path / "data_set"
    train_root = data_root / "DLO3" / "train"
    train_root.mkdir(parents=True)
    for index in range(56):
        (train_root / f"trajectory_{index:02d}.pkl").write_bytes(
            f"opaque-{index}".encode()
        )
    manifest = build_deform_dlo3_source_manifest(PROTOCOL, data_root)
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    protocol_sha256 = sha256_file(PROTOCOL)

    manifest["split"]["fit"], manifest["split"]["source_test"] = (
        manifest["split"]["source_test"],
        manifest["split"]["fit"],
    )
    with pytest.raises(ValueError, match="partition differs"):
        validate_deform_dlo3_source_manifest(
            manifest,
            protocol,
            protocol_sha256=protocol_sha256,
            verify_files=False,
        )

    manifest = build_deform_dlo3_source_manifest(PROTOCOL, data_root)
    changed = Path(next(iter(manifest["trajectories"].values()))["path"])
    changed.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        validate_deform_dlo3_source_manifest(
            manifest,
            protocol,
            protocol_sha256=protocol_sha256,
            verify_files=True,
        )


def test_mechanism_feature_subsets_are_fixed() -> None:
    assert len(deform_local_feature_indices("full-local")) == 92
    assert deform_local_feature_indices("intercept-only") == ()
    no_action = deform_local_feature_indices("full-no-action")
    assert len(no_action) == 36
    assert set(no_action).isdisjoint(set(range(24, 66)) | {69, 70} | set(range(80, 92)))

    with pytest.raises(ValueError, match="mechanism arm"):
        deform_local_feature_indices("selected-from-target")


def test_full_variant_preserves_frozen_point_operator() -> None:
    initial, action, baseline, targets, names = _residual_problem()
    frozen = fit_deform_local_residual(
        initial[:9],
        action[:9],
        baseline[:9],
        targets[:9],
        names[:9].tolist(),
        ridge=1.0,
        variance_floor_m2=1e-6,
    )
    variant = fit_deform_local_residual_variant(
        initial[:9],
        action[:9],
        baseline[:9],
        targets[:9],
        names[:9].tolist(),
        ridge=1.0,
        arm="full-local",
    )

    expected = predict_deform_local_residual(
        frozen,
        initial[9:],
        action[9:],
        baseline[9:],
        shrinkage=0.25,
    )
    actual = predict_deform_local_residual_variant(
        variant,
        initial[9:],
        action[9:],
        baseline[9:],
        shrinkage=0.25,
    )

    assert np.allclose(
        actual["predictions"], expected["predictions"], rtol=0.0, atol=1e-14
    )
    assert np.array_equal(
        actual["predictions"][:, :, (0, 1, -2, -1)],
        baseline[9:, :, (0, 1, -2, -1)],
    )


def test_full_covariance_preserves_mean_and_supports_calibration() -> None:
    initial, action, baseline, targets, names = _residual_problem(count=9)
    diagonal = fit_deform_local_residual(
        initial,
        action,
        baseline,
        targets,
        names.tolist(),
        ridge=1.0,
        variance_floor_m2=1e-6,
    )
    full_model = augment_deform_local_residual_full_covariance(
        diagonal,
        initial,
        action,
        baseline,
        targets,
        names.tolist(),
    )
    expected = predict_deform_local_residual(
        diagonal,
        initial,
        action,
        baseline,
        shrinkage=0.25,
    )
    full = predict_deform_local_residual_full_covariance(
        full_model,
        initial,
        action,
        baseline,
        shrinkage=0.25,
    )

    assert np.array_equal(full["predictions"], expected["predictions"])
    covariance = full["coordinate_covariance_m2"][:, :, 2:-2]
    assert np.allclose(covariance, covariance.swapaxes(-1, -2), atol=1e-12)
    assert np.min(np.linalg.eigvalsh(covariance)) > 0.0
    assert np.allclose(
        np.diagonal(covariance, axis1=-2, axis2=-1),
        expected["coordinate_variance_m2"][:, :, 2:-2],
        rtol=1e-7,
        atol=1e-12,
    )

    calibration = calibrate_deform_full_covariance(
        full["predictions"],
        targets,
        full["coordinate_covariance_m2"],
    )
    assert calibration["rank"] == 9
    assert calibration["variance_scale"] >= 1.0
    scaled = scale_deform_coordinate_covariance(
        full["coordinate_covariance_m2"],
        float(calibration["variance_scale"]),
    )
    raw_metrics = evaluate_deform_predictive_distribution(
        full["predictions"],
        targets,
        full["coordinate_covariance_m2"],
        sample_count=4,
    )
    scaled_metrics = evaluate_deform_predictive_distribution(
        full["predictions"],
        targets,
        scaled,
        sample_count=4,
    )
    assert all(
        math.isfinite(float(raw_metrics[key]))
        for key in (
            "gaussian_nll",
            "coordinate_nees",
            "multivariate_nees",
            "energy_score",
        )
    )
    assert (
        scaled_metrics["coordinate_coverage_90"]
        >= raw_metrics["coordinate_coverage_90"]
    )
    assert np.array_equal(full["predictions"], expected["predictions"])


def test_bayesian_covariance_ablation_is_complete_and_mean_preserving() -> None:
    initial, action, baseline, targets, names = _residual_problem(count=9)
    diagonal = fit_deform_local_residual(
        initial,
        action,
        baseline,
        targets,
        names.tolist(),
        ridge=1.0,
        variance_floor_m2=1e-6,
    )
    full_model = augment_deform_local_residual_full_covariance(
        diagonal,
        initial,
        action,
        baseline,
        targets,
        names.tolist(),
    )
    point = predict_deform_local_residual(
        diagonal,
        initial,
        action,
        baseline,
        shrinkage=0.25,
    )["predictions"]

    arms = build_deform_bayesian_covariance_ablation_v1(
        full_model,
        initial,
        action,
        baseline,
        shrinkage=0.25,
        variance_scale=4.0,
    )

    assert tuple(arms) == DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
    assert len(
        {
            deform_bayesian_covariance_archive_key(label)
            for label in DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
        }
    ) == len(arms)
    for values in arms.values():
        assert np.array_equal(values["predictions"], point)
        covariance = values["coordinate_covariance_m2"]
        assert covariance.shape == (*point.shape, 3)
        assert np.count_nonzero(covariance[:, :, :2]) == 0
        assert np.count_nonzero(covariance[:, :, -2:]) == 0
        assert np.min(np.linalg.eigvalsh(covariance[:, :, 2:-2])) > 0.0

    current = arms["current-diagonal-conservative-v1"]["coordinate_covariance_m2"]
    propagated = arms["shrinkage-propagated-diagonal"]["coordinate_covariance_m2"]
    current_variance = np.diagonal(current[:, :, 2:-2], axis1=-2, axis2=-1)
    propagated_variance = np.diagonal(propagated[:, :, 2:-2], axis1=-2, axis2=-1)
    assert np.all(propagated_variance <= current_variance + 1e-15)
    assert np.any(propagated_variance < current_variance)

    coefficient = arms["coefficient-only"]["coordinate_covariance_m2"]
    residual = arms["residual-only"]["coordinate_covariance_m2"]
    assert not np.array_equal(coefficient, residual)

    pooled = arms["pooled-isotropic"]["coordinate_covariance_m2"][:, :, 2:-2]
    pooled_diagonal = np.diagonal(pooled, axis1=-2, axis2=-1)
    assert np.allclose(pooled_diagonal, pooled_diagonal.reshape(-1)[0])
    assert (
        np.count_nonzero(
            pooled - np.eye(3, dtype=np.float64) * pooled_diagonal[..., None]
        )
        == 0
    )

    raw = arms["trajectory-clustered-full-coordinate-covariance-v1"][
        "coordinate_covariance_m2"
    ]
    calibrated = arms["calibrated-full-coordinate-covariance-v1"][
        "coordinate_covariance_m2"
    ]
    assert np.array_equal(calibrated, raw * 4.0)


def test_bayesian_covariance_ablation_rejects_unknown_archive_label() -> None:
    with pytest.raises(ValueError, match="covariance arm"):
        deform_bayesian_covariance_archive_key("selected-from-target")


def test_source_gate_uses_fixed_casewise_arithmetic() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    targets = np.zeros((8, 2, 5, 3), dtype=np.float64)
    baseline = np.full_like(targets, 0.007)
    candidate = np.full_like(targets, 0.006)
    names = [f"case-{index}" for index in range(8)]

    gate = evaluate_deform_dlo3_source_gate(
        candidate, baseline, targets, names, protocol
    )

    assert gate["passed"] is True
    assert gate["wins"] == 8
    assert gate["candidate_mean_l1_m"] == pytest.approx(0.006)
    assert gate["relative_improvement"] == pytest.approx(1.0 - 6.0 / 7.0)

    candidate[0] = 0.008
    changed = evaluate_deform_dlo3_source_gate(
        candidate, baseline, targets, names, protocol
    )
    assert changed["maximum_case_ratio"] == pytest.approx(8.0 / 7.0)
    assert changed["maximum_case_ratio_passed"] is False
    assert changed["passed"] is False


def test_backend_source_gate_has_no_published_reference_shortcut() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    targets = np.zeros((8, 2, 5, 3), dtype=np.float64)
    backend = np.full_like(targets, 0.020)
    candidate = np.full_like(targets, 0.018)
    names = [f"case-{index}" for index in range(8)]

    gate = evaluate_deform_backend_source_gate(
        candidate, backend, targets, names, protocol
    )

    assert gate["passed"] is True
    assert gate["relative_improvement"] == pytest.approx(0.10)
    assert "published_reference_passed" not in gate


def test_target_gate_reports_unique_and_canonical_reference_operators() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    targets = np.zeros((14, 2, 5, 3), dtype=np.float64)
    baseline = np.full_like(targets, 0.008)
    candidate = np.full_like(targets, 0.007)
    names = [f"case-{index}" for index in range(14)]

    gate = evaluate_deform_dlo3_target_gate(
        candidate, baseline, targets, names, protocol
    )

    assert gate["passed"] is True
    assert gate["candidate_mean_l1_m"] == pytest.approx(0.007)
    assert gate["canonical_reference_draw_mean_l1_m"] == pytest.approx(0.007)
    assert gate["all_unique_below_published_reference"] is True
    assert gate["canonical_draw_below_published_reference"] is True


def test_stability_gate_requires_primary_and_two_of_three_seeds() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    results = [_seed_result(seed) for seed in (42, 43, 44)]

    gate = evaluate_deform_dlo3_stability_gate(results, protocol)

    assert gate["passed"] is True
    assert gate["alltrain_fit_authorized"] is True
    assert gate["target_authorized"] is False
    assert gate["seed_source_passes"] == 3
    assert gate["seed_selection"] is False

    primary_failed = [_seed_result(seed, passed=seed != 42) for seed in (42, 43, 44)]
    rejected = evaluate_deform_dlo3_stability_gate(primary_failed, protocol)
    assert rejected["seed_source_passes"] == 2
    assert rejected["primary_seed_passed"] is False
    assert rejected["passed"] is False


def test_stability_gate_rejects_instability_or_custody_change() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    unstable = [
        _seed_result(42),
        _seed_result(43),
        _seed_result(44, ratio=1.20, passed=False),
    ]

    gate = evaluate_deform_dlo3_stability_gate(unstable, protocol)

    assert gate["maximum_seed_mean_ratio"] == pytest.approx(1.20)
    assert gate["seed_mean_ratio_requirement"] is False
    assert gate["passed"] is False

    changed = [_seed_result(seed) for seed in (42, 43, 44)]
    changed[2]["primary_eval_read"] = True
    with pytest.raises(ValueError, match="custody"):
        evaluate_deform_dlo3_stability_gate(changed, protocol)


def test_stability_runner_cannot_authorize_target() -> None:
    source = STABILITY_RUNNER.read_text(encoding="utf-8")

    assert "evaluate_deform_dlo3_stability_gate" in source
    assert "DLO3" not in source or "/eval" not in source
    assert "target_authorized" not in source


def test_sensitivity_runner_seals_before_scoring_and_never_selects() -> None:
    source = SENSITIVITY_RUNNER.read_text(encoding="utf-8")

    source_open = source.index("trajectories = source_runtime._load_named_trajectories")
    prediction_seal = source.index('seal_path = output_root / "prediction_seal.json"')
    scoring = source.index("scores = {")
    assert source_open < prediction_seal < scoring
    assert '"selection_effect": "none"' in source
    assert '"target_authorized": False' in source
    assert (
        'source_runtime._install_eval_read_guard(data_root / "DLO3" / "eval")' in source
    )


def test_pyelastica_runner_seals_fit_and_predictions_before_source_scoring() -> None:
    source = PYELASTICA_RUNNER.read_text(encoding="utf-8")

    method_seal = source.index('method_seal_path = output_root / "method_seal.json"')
    source_open = source.index("source_panel = source_runtime._load_named_trajectories")
    prediction_seal = source.index(
        'prediction_seal_path = output_root / "prediction_seal.json"'
    )
    scoring = source.index("gate = evaluate_deform_backend_source_gate")
    assert method_seal < source_open < prediction_seal < scoring
    assert '"primary_target_authorized": False' in source
    assert '"retry_authorized": False' in source


def test_alltrain_runner_requires_every_source_audit_and_guards_eval() -> None:
    source = ALLTRAIN_RUNNER.read_text(encoding="utf-8")

    assert '"deform-dlo3-training-stability-gate-v1"' in source
    assert '"deform-dlo3-physics-solver-sensitivity-result-v1"' in source
    assert '"deform-dlo3-pyelastica-source-result-v1"' in source
    assert '"deform-dlo3-count-only-custody-deviation-v1"' in source
    assert (
        'source_runtime._install_eval_read_guard(data_root / "DLO3" / "eval")' in source
    )
    assert '"target_authorized": False' in source


def test_evaluator_authorizes_before_target_manifest_and_seals_before_score() -> None:
    source = EVALUATOR_RUNNER.read_text(encoding="utf-8")

    authorization = source.index(
        'authorization_path = output_root / "authorization.json"'
    )
    target_manifest = source.index('stage = "target-manifest"')
    bayesian_construction = source.index(
        "bayesian_predictions = build_deform_bayesian_covariance_ablation_v1"
    )
    covariance_archive = source.index("deform_bayesian_covariance_archive_key(label):")
    prediction_seal = source.index(
        'prediction_seal_path = output_root / "prediction_seal.json"'
    )
    distribution_scoring = source.index("bayesian_distributions = {")
    target_score = source.index("gate = evaluate_deform_dlo3_target_gate")
    assert (
        authorization
        < target_manifest
        < bayesian_construction
        < covariance_archive
        < prediction_seal
        < distribution_scoring
        < target_score
    )
    assert '"distribution_selection": "none"' in source
    assert '"target_outcomes_used_for_distribution_selection": False' in source
    assert '"retry_authorized": False' in source
    assert '"case_replacement": False' in source


def test_readiness_requires_dry_run_and_discloses_count_deviation() -> None:
    source = READINESS_RUNNER.read_text(encoding="utf-8")

    assert '"deform-dlo3-robustness-evaluator-dry-run-v1"' in source
    assert '"deform-dlo3-count-only-custody-deviation-v1"' in source
    assert '"count_only_custody_deviation_acknowledged": True' in source
    assert '"target_authorized": True' in source


def test_seed_runner_seals_models_and_predictions_before_scoring() -> None:
    source = SEED_RUNNER.read_text(encoding="utf-8")

    method_seal = source.index('method_seal_path = output_root / "method_seal.json"')
    source_open = source.index("source_test_trajectories =")
    bayesian_construction = source.index(
        "bayesian_predictions = build_deform_bayesian_covariance_ablation_v1"
    )
    covariance_archive = source.index("deform_bayesian_covariance_archive_key(label):")
    prediction_seal = source.index(
        'prediction_seal_path = output_root / "prediction_seal.json"'
    )
    distribution_scoring = source.index("bayesian_distributions = {")
    scoring = source.index("primary_gate = evaluate_deform_dlo3_source_gate")
    assert (
        method_seal
        < source_open
        < bayesian_construction
        < covariance_archive
        < prediction_seal
        < scoring
        < distribution_scoring
    )
    assert '"distribution_selection": "none"' in source
    assert '"source_test_outcomes_used_for_covariance_construction": False' in source
    assert (
        'source_runtime._install_eval_read_guard(data_root / "DLO3" / "eval")' in source
    )
    assert 'source_runtime._install_eval_read_guard(data_root / "DLO4")' in source
    assert 'source_runtime._install_eval_read_guard(data_root / "DLO5")' in source
    assert '"target_authorized": False' in source
