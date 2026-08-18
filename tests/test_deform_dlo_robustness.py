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
from bayesian_phystwin_experiments.deform_dlo_robustness import (
    assign_deform_dlo3_source_partitions,
    augment_deform_local_residual_full_covariance,
    build_deform_dlo3_source_manifest,
    calibrate_deform_full_covariance,
    deform_local_feature_indices,
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


def test_loads_locked_dlo_robustness_protocol() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)

    assert protocol["prob4d_used"] is False
    assert protocol["freshness"]["primary_dlo"] == "DLO3"
    assert protocol["custody"]["held_v8_access"] is False


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
