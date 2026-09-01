from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.deform_dlo_cross_backend_scalar_v1 import (
    evaluate_cross_backend_scalar_transport,
    leave_one_trajectory_out_scalar_transport,
    load_cross_backend_scalar_protocol,
    trajectory_alignment,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "protocols" / "deform_dlo3_cross_backend_scalar_v1.json"
RUNNER = (
    REPOSITORY_ROOT
    / "scripts"
    / "remote"
    / "run_deform_dlo3_cross_backend_scalar_v1.py"
)


def _problem() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    names = [f"case-{index}" for index in range(8)]
    truth = np.zeros((8, 5, 6, 3), dtype=float)
    baseline = np.full_like(truth, 0.10)
    specific = np.full_like(truth, 0.02)
    return names, truth, baseline, specific


def test_protocol_freezes_one_scalar_complete_trajectory_folds() -> None:
    protocol = load_cross_backend_scalar_protocol(PROTOCOL)

    assert protocol["parent"]["seed_models"] == [42, 43, 44]
    assert protocol["transport"]["minimum_scalar"] == 0.0
    assert protocol["transport"]["maximum_scalar"] == 4.0
    assert protocol["source_panel"]["trajectory_count"] == 8
    assert (
        protocol["information_boundary"][
            "same_trajectory_label_used_for_its_scalar"
        ]
        is False
    )
    assert protocol["information_boundary"]["dlo3_official_evaluation_read"] is False


def test_scalar_transport_recovers_shared_direction_when_direct_amplitude_fails() -> None:
    protocol = load_cross_backend_scalar_protocol(PROTOCOL)
    names, truth, baseline, specific = _problem()
    direct = np.full_like(truth, -0.10)

    result = evaluate_cross_backend_scalar_transport(
        names=names,
        truth=truth,
        baseline=baseline,
        direct_prediction=direct,
        pyelastica_specific_candidate=specific,
        protocol=protocol,
    )

    assert result["claim_ladder"]["exact_no_refit_point_transfer_supported"] is False
    assert (
        result["claim_ladder"][
            "one_scalar_cross_validated_point_transfer_supported"
        ]
        is True
    )
    assert result["claim_ladder"]["directional_alignment_supported"] is True
    assert result["claim_ladder"]["shared_residual_geometry_supported"] is True
    assert result["decision"] == "cross-backend-shared-residual-geometry-supported"
    assert result["fold_scalars"]["values"] == pytest.approx([0.5] * 8)
    assert result["scalar_vs_raw_pyelastica"]["wins"] == 8
    assert result["scalar_vs_raw_pyelastica"]["candidate_mean_l1_m"] == pytest.approx(
        0.0
    )


def test_opposite_residual_direction_fails_closed_at_zero_scalar() -> None:
    protocol = load_cross_backend_scalar_protocol(PROTOCOL)
    names, truth, baseline, specific = _problem()
    direct = np.full_like(truth, 0.20)

    result = evaluate_cross_backend_scalar_transport(
        names=names,
        truth=truth,
        baseline=baseline,
        direct_prediction=direct,
        pyelastica_specific_candidate=specific,
        protocol=protocol,
    )

    assert result["fold_scalars"]["values"] == pytest.approx([0.0] * 8)
    assert result["directional_alignment"]["positive_cases"] == 0
    assert result["scalar_vs_raw_pyelastica"]["wins"] == 0
    assert result["promotion_gate"]["supported"] is False
    assert result["decision"] == "cross-backend-shared-residual-geometry-not-supported"


def test_held_out_truth_never_changes_its_own_fold_scalar() -> None:
    baseline = np.zeros((8, 2, 3, 1), dtype=float)
    direct = np.ones_like(baseline)
    truth = np.full_like(baseline, 0.5)

    _, first = leave_one_trajectory_out_scalar_transport(
        direct,
        baseline,
        truth,
        minimum_scalar=0.0,
        maximum_scalar=4.0,
    )
    changed_truth = truth.copy()
    changed_truth[0] = 100.0
    _, second = leave_one_trajectory_out_scalar_transport(
        direct,
        baseline,
        changed_truth,
        minimum_scalar=0.0,
        maximum_scalar=4.0,
    )

    assert first[0] == pytest.approx(0.5)
    assert second[0] == pytest.approx(first[0])
    assert second[1] != pytest.approx(first[1])


def test_alignment_is_scale_invariant_and_rejects_shape_mismatch() -> None:
    baseline = np.zeros((3, 2, 2, 1), dtype=float)
    truth = np.ones_like(baseline)
    direct = 7.0 * truth

    assert trajectory_alignment(direct, baseline, truth) == pytest.approx(
        np.ones(3)
    )
    with pytest.raises(ValueError, match="differ in shape"):
        trajectory_alignment(direct[:, :, :1], baseline, truth)


def test_scalar_transport_rejects_invalid_bounds() -> None:
    _, truth, baseline, _ = _problem()
    with pytest.raises(ValueError, match="bounds"):
        leave_one_trajectory_out_scalar_transport(
            baseline,
            baseline,
            truth,
            minimum_scalar=1.0,
            maximum_scalar=1.0,
        )


def test_runner_writes_method_seal_before_source_payload_loading() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    seal = source.index('method_seal_path = output_root / "method_seal.json"')
    source_load = source.index("trajectories = direct_runtime._load_source_trajectories(")
    assert seal < source_load
    assert "same_trajectory_label_used_for_its_scalar" in source
    assert "dlo3_official_evaluation_read" in source
    assert "dlo4_or_dlo5_read" in source
    assert "allow_pickle=False" in source
