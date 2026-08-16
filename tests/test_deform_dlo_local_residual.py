import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform_dlo_local_residual import (
    build_deform_local_residual_features,
    deform_causal_inputs,
    fit_deform_local_residual,
    load_deform_dlo2_local_residual_protocol,
    load_deform_dlo2_local_residual_v6_protocol,
    load_deform_local_residual_protocol,
    predict_deform_local_residual,
    serialize_deform_local_residual_model,
    validate_deform_dlo2_local_residual_parent,
    validate_deform_dlo2_local_residual_v6_parents,
)
from bayesian_phystwin.deform_dlo_source import (
    sha256_file,
    validate_deform_dlo2_stage_authorization,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "configs" / "sota" / "deform_dlo_local_residual_v4.json"
RUNNER = REPOSITORY_ROOT / "scripts" / "remote" / "run_deform_dlo_local_residual.py"
DLO2_PROTOCOL = (
    REPOSITORY_ROOT / "configs" / "sota" / "deform_dlo2_local_residual_v5.json"
)
DLO2_RUNNER = (
    REPOSITORY_ROOT / "scripts" / "remote" / "run_deform_dlo2_local_residual.py"
)
DLO2_V6_DEVELOPMENT_RUNNER = (
    REPOSITORY_ROOT
    / "scripts"
    / "remote"
    / "analyze_deform_dlo2_local_residual_v6.py"
)
DLO2_V6_PROTOCOL = (
    REPOSITORY_ROOT / "configs" / "sota" / "deform_dlo2_local_residual_v6.json"
)
DLO2_V6_RUNNER = (
    REPOSITORY_ROOT / "scripts" / "remote" / "run_deform_dlo2_local_residual_v6.py"
)
SOURCE_RUNNER = REPOSITORY_ROOT / "scripts" / "remote" / "run_deform_dlo_source.py"
DLO1_RESULT = (
    REPOSITORY_ROOT
    / "results"
    / "sota"
    / "deform_dlo_local_residual_v4"
    / "result.json"
)
DLO2_V5_RESULT = (
    REPOSITORY_ROOT
    / "results"
    / "sota"
    / "deform_dlo2_local_residual_v5"
    / "result.json"
)
DLO2_V6_DEVELOPMENT = (
    REPOSITORY_ROOT
    / "results"
    / "sota"
    / "deform_dlo2_local_residual_v6"
    / "development_selection.json"
)


def _trajectories(count: int = 6, frames: int = 14, nodes: int = 7) -> np.ndarray:
    result = np.zeros((count, frames, nodes, 3), dtype=float)
    arc = np.linspace(-1.0, 1.0, nodes)
    for case in range(count):
        amplitude = 0.02 + 0.01 * case
        for frame in range(frames):
            phase = frame / (frames - 1)
            result[case, frame, :, 0] = arc
            result[case, frame, :, 1] = amplitude * phase * (1.0 - arc**2)
            result[case, frame, :, 2] = 0.01 * case * phase
            result[case, frame, :2, 1] += amplitude * phase
            result[case, frame, -2:, 1] -= 0.5 * amplitude * phase
    return result


def _problem(duplicate: bool = False) -> tuple[np.ndarray, ...]:
    trajectories = _trajectories()
    initial, action = deform_causal_inputs(trajectories)
    targets = trajectories[:, 2:].copy()
    baseline = targets.copy()
    time = np.linspace(0.0, 1.0, targets.shape[1])
    baseline[:, :, 2:-2, 1] -= (0.005 + 0.002 * np.arange(trajectories.shape[0]))[
        :, None, None
    ] * time[None, :, None]
    names = np.asarray([f"case-{index}" for index in range(trajectories.shape[0])])
    if duplicate:
        initial = np.concatenate((initial, initial[:1]))
        action = np.concatenate((action, action[:1]))
        targets = np.concatenate((targets, targets[:1]))
        baseline = np.concatenate((baseline, baseline[:1]))
        names = np.concatenate((names, np.asarray(("case-0-copy",))))
    return initial, action, baseline, targets, names


def test_protocol_seals_dlo2_and_source_until_validation(tmp_path: Path) -> None:
    protocol = load_deform_local_residual_protocol(PROTOCOL)

    assert protocol["information_boundary"]["dlo2_training_read"] is False
    assert (
        protocol["information_boundary"][
            "dlo1_source_test_read_only_after_validation_seal"
        ]
        is True
    )

    changed_payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    changed_payload["information_boundary"]["dlo2_training_read"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(changed_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="seal future data"):
        load_deform_local_residual_protocol(changed)


def test_runner_seals_validation_before_source_and_guards_future_data() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    validation_seal = source.index(
        'selection_path = output_root / "validation_selection_seal.json"'
    )
    fallback_gate = source.index('if bool(selection["fallback_used"]):')
    source_load = source.index("source_trajectories =")
    assert validation_seal < fallback_gate < source_load
    assert '_install_eval_read_guard(data_root / "DLO1" / "eval")' in source
    assert '_install_eval_read_guard(data_root / "DLO2")' in source


def test_dlo2_protocol_fixes_dlo1_arm_and_validates_parent(tmp_path: Path) -> None:
    protocol = load_deform_dlo2_local_residual_protocol(DLO2_PROTOCOL)
    parent = json.loads(DLO1_RESULT.read_text(encoding="utf-8"))
    validated = validate_deform_dlo2_local_residual_parent(protocol, parent)

    assert protocol["local_residual"]["fixed_arm"] == {
        "name": "r1_s0p5",
        "ridge": 1.0,
        "selection_source": "frozen-dlo1-v4",
        "shrinkage": 0.5,
    }
    assert validated["source_gate_passed"] is True
    assert validated["official_eval_read"] is False

    changed_payload = json.loads(DLO2_PROTOCOL.read_text(encoding="utf-8"))
    changed_payload["local_residual"]["fixed_arm"]["shrinkage"] = 0.75
    changed = tmp_path / "changed-dlo2.json"
    changed.write_text(json.dumps(changed_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed arm differs"):
        load_deform_dlo2_local_residual_protocol(changed)


def test_dlo2_stage_authorization_accepts_only_bound_v4_result() -> None:
    protocol = load_deform_dlo2_local_residual_protocol(DLO2_PROTOCOL)
    parent = json.loads(DLO1_RESULT.read_text(encoding="utf-8"))
    validated_parent = validate_deform_dlo2_local_residual_parent(protocol, parent)
    authorization = {
        "contract": "deform-dlo2-local-residual-authorization-v1",
        "official_eval_read": False,
        "source_test_opened": False,
        "protocol": {"sha256": sha256_file(DLO2_PROTOCOL)},
        "parent_local_residual_result": {
            "sha256": sha256_file(DLO1_RESULT),
            **validated_parent,
        },
    }

    accepted = validate_deform_dlo2_stage_authorization(
        protocol,
        authorization,
        protocol_sha256=sha256_file(DLO2_PROTOCOL),
    )
    assert accepted["selected_arm"] == "r1_s0p5"

    authorization["parent_local_residual_result"]["source_gate_passed"] = False
    with pytest.raises(ValueError, match="authorization differs"):
        validate_deform_dlo2_stage_authorization(
            protocol,
            authorization,
            protocol_sha256=sha256_file(DLO2_PROTOCOL),
        )


def test_dlo2_training_and_transfer_seal_source_before_loading() -> None:
    generic = SOURCE_RUNNER.read_text(encoding="utf-8")
    transfer = DLO2_RUNNER.read_text(encoding="utf-8")

    training_stop = generic.index('if args.mode == "train-validation":')
    generic_source_load = generic.index("source_test_trajectories =")
    transfer_seal = transfer.index(
        'selection_path = output_root / "validation_transfer_seal.json"'
    )
    transfer_fallback = transfer.index('if not bool(validation_gate["passed"]):')
    transfer_source_load = transfer.index("source_trajectories =")
    assert training_stop < generic_source_load
    assert transfer_seal < transfer_fallback < transfer_source_load
    assert '_install_eval_read_guard(data_root / "DLO2" / "eval")' in transfer


def test_dlo2_v6_development_runner_cannot_open_source_or_official_eval() -> None:
    source = DLO2_V6_DEVELOPMENT_RUNNER.read_text(encoding="utf-8")

    assert 'manifest["split"]["source_test"]' not in source
    assert "source_test_opened\": False" in source
    assert "official_eval_read\": False" in source
    assert '_install_eval_read_guard(data_root / "DLO2" / "eval")' in source
    assert '_install_eval_read_guard(data_root / "DLO1" / "eval")' in source


def test_dlo2_v6_protocol_binds_validation_selection_and_closed_v5() -> None:
    protocol = load_deform_dlo2_local_residual_v6_protocol(DLO2_V6_PROTOCOL)
    parent = json.loads(DLO2_V5_RESULT.read_text(encoding="utf-8"))
    development = json.loads(DLO2_V6_DEVELOPMENT.read_text(encoding="utf-8"))

    authorized = validate_deform_dlo2_local_residual_v6_parents(
        protocol,
        parent,
        development,
    )

    assert protocol["local_residual"]["fixed_arm"]["shrinkage"] == 0.25
    assert authorized["selected_arm"] == "r1_s0p25"
    assert authorized["source_test_opened"] is False


def test_dlo2_v6_parent_validation_rejects_opened_source() -> None:
    protocol = load_deform_dlo2_local_residual_v6_protocol(DLO2_V6_PROTOCOL)
    parent = json.loads(DLO2_V5_RESULT.read_text(encoding="utf-8"))
    development = json.loads(DLO2_V6_DEVELOPMENT.read_text(encoding="utf-8"))
    parent["source_test_opened"] = True

    with pytest.raises(ValueError, match="do not authorize source"):
        validate_deform_dlo2_local_residual_v6_parents(
            protocol,
            parent,
            development,
        )


def test_dlo2_v6_seals_source_opening_before_loading_outcomes() -> None:
    source = DLO2_V6_RUNNER.read_text(encoding="utf-8")

    opening_seal = source.index(
        'source_opening_path = output_root / "source_opening_seal.json"'
    )
    preflight_stop = source.index('if args.mode == "preflight":')
    source_names = source.index('source_names = list(manifest["split"]["source_test"])')
    source_load = source.index("source_trajectories =")
    assert opening_seal < preflight_stop < source_names < source_load
    assert '_install_eval_read_guard(data_root / "DLO2" / "eval")' in source
    assert '_install_eval_read_guard(data_root / "DLO1" / "eval")' in source
    assert '"official_eval_read": False' in source


def test_causal_input_extractor_omits_future_free_nodes() -> None:
    trajectories = _trajectories()
    changed = trajectories.copy()
    changed[:, 2:, 2:-2] += 1000.0

    initial, action = deform_causal_inputs(trajectories)
    changed_initial, changed_action = deform_causal_inputs(changed)

    assert np.array_equal(initial, changed_initial)
    assert np.array_equal(action, changed_action)


def test_features_and_predictions_are_yaw_equivariant() -> None:
    initial, action, baseline, targets, names = _problem()
    model = fit_deform_local_residual(
        initial[:4],
        action[:4],
        baseline[:4],
        targets[:4],
        names[:4].tolist(),
        ridge=1e-2,
        variance_floor_m2=1e-6,
    )
    original = predict_deform_local_residual(
        model,
        initial[4:],
        action[4:],
        baseline[4:],
        shrinkage=0.5,
    )
    angle = 0.6
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    translation = np.asarray((3.0, -1.0, 0.5))
    moved = predict_deform_local_residual(
        model,
        initial[4:] @ rotation.T + translation,
        action[4:] @ rotation.T + translation,
        baseline[4:] @ rotation.T + translation,
        shrinkage=0.5,
    )

    expected = original["predictions"] @ rotation.T + translation
    assert np.allclose(moved["predictions"], expected, atol=1e-9)


def test_local_residual_recovers_synthetic_bias_and_preserves_clamps() -> None:
    initial, action, baseline, targets, names = _problem()
    model = fit_deform_local_residual(
        initial[:4],
        action[:4],
        baseline[:4],
        targets[:4],
        names[:4].tolist(),
        ridge=1e-3,
        variance_floor_m2=1e-6,
    )
    result = predict_deform_local_residual(
        model,
        initial[4:],
        action[4:],
        baseline[4:],
        shrinkage=1.0,
    )

    baseline_error = np.mean(np.abs(baseline[4:] - targets[4:]))
    candidate_error = np.mean(np.abs(result["predictions"] - targets[4:]))
    assert candidate_error < 0.25 * baseline_error
    assert np.array_equal(
        result["predictions"][:, :, (0, 1, -2, -1)],
        baseline[4:, :, (0, 1, -2, -1)],
    )
    assert np.all(result["coordinate_variance_m2"] >= 0.0)


def test_duplicate_query_is_one_covariance_cluster() -> None:
    initial, action, baseline, targets, names = _problem(duplicate=True)
    model = fit_deform_local_residual(
        initial,
        action,
        baseline,
        targets,
        names.tolist(),
        ridge=1e-2,
        variance_floor_m2=1e-6,
    )

    assert len(model["trajectory_clusters"]) == 6
    assert model["trajectory_clusters"][0] == ("case-0", "case-0-copy")


def test_query_features_do_not_accept_target_or_innovation() -> None:
    initial, action, baseline, _, _ = _problem()

    features, frames = build_deform_local_residual_features(
        initial,
        action,
        baseline,
    )

    assert features.shape[:3] == (6, 12, 3)
    assert frames.shape == (6, 3, 3)


def test_shrinkage_leaves_unresolved_correction_in_variance() -> None:
    initial, action, baseline, targets, names = _problem()
    model = fit_deform_local_residual(
        initial[:4],
        action[:4],
        baseline[:4],
        targets[:4],
        names[:4].tolist(),
        ridge=1e-2,
        variance_floor_m2=1e-6,
    )
    full = predict_deform_local_residual(
        model,
        initial[4:],
        action[4:],
        baseline[4:],
        shrinkage=1.0,
    )
    partial = predict_deform_local_residual(
        model,
        initial[4:],
        action[4:],
        baseline[4:],
        shrinkage=0.25,
    )

    internal = (slice(None), slice(None), slice(2, -2), slice(None))
    assert np.mean(partial["coordinate_variance_m2"][internal]) > np.mean(
        full["coordinate_variance_m2"][internal]
    )


def test_model_serialization_is_pickle_free() -> None:
    initial, action, baseline, targets, names = _problem()
    model = fit_deform_local_residual(
        initial,
        action,
        baseline,
        targets,
        names.tolist(),
        ridge=1e-2,
        variance_floor_m2=1e-6,
    )
    serialized = serialize_deform_local_residual_model(model)

    assert serialized
    assert all(np.asarray(value).dtype != object for value in serialized.values())


def test_invalid_ridge_is_rejected() -> None:
    initial, action, baseline, targets, names = _problem()

    with pytest.raises(ValueError, match="ridge"):
        fit_deform_local_residual(
            initial,
            action,
            baseline,
            targets,
            names.tolist(),
            ridge=0.0,
            variance_floor_m2=1e-6,
        )
