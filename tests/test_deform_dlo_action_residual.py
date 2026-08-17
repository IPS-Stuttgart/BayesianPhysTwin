import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.experiments.deform_dlo_action_residual import (
    build_deform_action_descriptors,
    deform_action_residual_records,
    fit_deform_action_residual,
    load_deform_action_residual_protocol,
    predict_deform_action_residual,
    select_deform_action_residual_arm,
    serialize_deform_action_residual_model,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "configs" / "sota" / "deform_dlo_action_residual_v3.json"


def _trajectories(count: int = 3, frames: int = 8, nodes: int = 7) -> np.ndarray:
    result = np.zeros((count, frames, nodes, 3), dtype=float)
    arc = np.linspace(-1.0, 1.0, nodes)
    for case in range(count):
        for frame in range(frames):
            result[case, frame, :, 0] = arc + 0.02 * case
            result[case, frame, :, 1] = 0.03 * frame * arc**2
            result[case, frame, :, 2] = 0.01 * case * frame
            result[case, frame, (0, 1), 1] += 0.02 * case * frame
            result[case, frame, (-2, -1), 1] -= 0.01 * case * frame
    return result


def _fit_model(*, duplicate: bool = False) -> tuple[dict[str, object], np.ndarray]:
    trajectories = _trajectories()
    targets = trajectories[:, 2:].copy()
    baseline = targets.copy()
    baseline[:, :, 2:-2, 1] -= np.asarray((0.01, 0.02, 0.03))[:, None, None]
    names = ["a", "b", "c"]
    if duplicate:
        trajectories = np.concatenate((trajectories, trajectories[:1]))
        targets = np.concatenate((targets, targets[:1]))
        baseline = np.concatenate((baseline, baseline[:1]))
        names.append("a-copy")
    model = fit_deform_action_residual(
        trajectories,
        baseline,
        targets,
        names,
        sample_count=4,
        variance_floor_m2=1e-6,
    )
    return model, trajectories


def test_action_residual_protocol_seals_dlo2_and_official_eval(tmp_path: Path) -> None:
    protocol = load_deform_action_residual_protocol(PROTOCOL)

    assert protocol["information_boundary"]["dlo2_training_read"] is False
    assert protocol["information_boundary"]["official_eval_read"] is False

    changed_payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    changed_payload["information_boundary"]["dlo2_training_read"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(changed_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="seal future data"):
        load_deform_action_residual_protocol(changed)


def test_action_descriptor_is_rigid_invariant() -> None:
    trajectories = _trajectories()
    angle = 0.4
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    transformed = trajectories @ rotation.T + np.asarray((2.0, -3.0, 0.4))

    original, _ = build_deform_action_descriptors(trajectories, sample_count=4)
    moved, _ = build_deform_action_descriptors(transformed, sample_count=4)

    assert np.allclose(original, moved, atol=1e-10)


def test_action_residual_prediction_is_equivariant_and_preserves_clamps() -> None:
    model, trajectories = _fit_model()
    query = trajectories[1:2]
    target = query[:, 2:].copy()
    baseline = target.copy()
    baseline[:, :, 2:-2, 1] -= 0.02
    result = predict_deform_action_residual(
        model,
        query,
        baseline,
        neighbor_count=1,
        length_scale_multiplier=1.0,
        shrinkage=1.0,
    )

    assert np.array_equal(
        result["predictions"][:, :, (0, 1, -2, -1)], baseline[:, :, (0, 1, -2, -1)]
    )
    assert np.mean(np.abs(result["predictions"] - target)) < np.mean(
        np.abs(baseline - target)
    )

    angle = -0.7
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    translation = np.asarray((-1.0, 0.4, 2.0))
    moved_query = query @ rotation.T + translation
    moved_baseline = baseline @ rotation.T + translation
    moved = predict_deform_action_residual(
        model,
        moved_query,
        moved_baseline,
        neighbor_count=1,
        length_scale_multiplier=1.0,
        shrinkage=1.0,
    )

    expected = result["predictions"] @ rotation.T + translation
    assert np.allclose(moved["predictions"], expected, atol=1e-10)


def test_duplicate_donor_is_collapsed_before_weighting() -> None:
    ordinary, trajectories = _fit_model()
    duplicated, _ = _fit_model(duplicate=True)
    baseline = trajectories[1:2, 2:].copy()

    first = predict_deform_action_residual(
        ordinary,
        trajectories[1:2],
        baseline,
        neighbor_count=3,
        length_scale_multiplier=1.0,
        shrinkage=0.5,
    )
    second = predict_deform_action_residual(
        duplicated,
        trajectories[1:2],
        baseline,
        neighbor_count=3,
        length_scale_multiplier=1.0,
        shrinkage=0.5,
    )

    assert duplicated["donor_descriptors"].shape[0] == 3
    assert np.array_equal(duplicated["donor_cluster_sizes"], np.asarray((2, 1, 1)))
    assert np.allclose(first["weights"], second["weights"])
    assert np.allclose(
        first["coordinate_variance_m2"], second["coordinate_variance_m2"]
    )


def test_weights_do_not_depend_on_query_innovation() -> None:
    model, trajectories = _fit_model()
    query = trajectories[2:3]
    baseline_a = query[:, 2:].copy()
    baseline_b = baseline_a + 100.0

    first = predict_deform_action_residual(
        model,
        query,
        baseline_a,
        neighbor_count=2,
        length_scale_multiplier=1.0,
        shrinkage=0.25,
    )
    second = predict_deform_action_residual(
        model,
        query,
        baseline_b,
        neighbor_count=2,
        length_scale_multiplier=1.0,
        shrinkage=0.25,
    )

    assert np.array_equal(first["neighbor_indices"], second["neighbor_indices"])
    assert np.array_equal(first["weights"], second["weights"])
    assert np.array_equal(
        first["coordinate_variance_m2"], second["coordinate_variance_m2"]
    )


def test_mixture_spread_survives_dense_prediction() -> None:
    model, trajectories = _fit_model()
    baseline = trajectories[:1, 2:].copy()
    one = predict_deform_action_residual(
        model,
        trajectories[:1],
        baseline,
        neighbor_count=1,
        length_scale_multiplier=1.0,
        shrinkage=1.0,
    )
    three = predict_deform_action_residual(
        model,
        trajectories[:1],
        baseline,
        neighbor_count=3,
        length_scale_multiplier=10.0,
        shrinkage=1.0,
    )

    internal = (slice(None), slice(None), slice(2, -2), slice(None))
    assert np.all(one["coordinate_variance_m2"][internal] == pytest.approx(1e-6))
    assert np.mean(three["coordinate_variance_m2"][internal]) > 1e-6


def test_selector_returns_exact_fallback_when_gate_fails() -> None:
    target = np.zeros((2, 3, 5, 3), dtype=float)
    baseline = np.ones_like(target)
    candidate = baseline.copy()
    candidate[0] *= 0.9
    candidate[1] *= 1.1
    records = deform_action_residual_records(
        candidate,
        target,
        baseline,
        ("a", "b"),
    )

    selected = select_deform_action_residual_arm(
        {"candidate": records},
        minimum_relative_improvement=0.01,
        minimum_case_wins=2,
        maximum_case_ratio=1.05,
    )

    assert selected["selected_arm"] == "baseline_exact"
    assert selected["fallback_used"] is True


def test_synthetic_action_conditioned_residual_is_recovered() -> None:
    trajectories = _trajectories(count=4)
    targets = trajectories[:, 2:].copy()
    baseline = targets.copy()
    amplitudes = np.asarray((0.01, 0.02, 0.03, 0.04))
    baseline[:, :, 2:-2, 1] -= amplitudes[:, None, None]
    model = fit_deform_action_residual(
        trajectories[:3],
        baseline[:3],
        targets[:3],
        ("a", "b", "c"),
        sample_count=4,
        variance_floor_m2=1e-6,
    )
    result = predict_deform_action_residual(
        model,
        trajectories[3:],
        baseline[3:],
        neighbor_count=1,
        length_scale_multiplier=1.0,
        shrinkage=1.0,
    )

    baseline_error = np.mean(np.abs(baseline[3:] - targets[3:]))
    candidate_error = np.mean(np.abs(result["predictions"] - targets[3:]))
    assert candidate_error < baseline_error


def test_model_serialization_contains_no_object_arrays() -> None:
    model, _ = _fit_model()
    serialized = serialize_deform_action_residual_model(model)

    assert serialized
    assert all(np.asarray(value).dtype != object for value in serialized.values())
