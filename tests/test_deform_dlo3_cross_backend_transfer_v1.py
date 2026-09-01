import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.deform_dlo_cross_backend_transfer_v1 import (
    evaluate_cross_backend_transfer,
    load_cross_backend_transfer_protocol,
    paired_point_summary,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "protocols" / "deform_dlo3_cross_backend_transfer_v1.json"
RUNNER = (
    REPOSITORY_ROOT
    / "scripts"
    / "remote"
    / "run_deform_dlo3_cross_backend_transfer_v1.py"
)


def _problem() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    names = [f"case-{index}" for index in range(8)]
    truth = np.zeros((8, 6, 7, 3), dtype=float)
    baseline = np.full_like(truth, 0.10)
    backend_specific = np.full_like(truth, 0.04)
    return names, truth, baseline, backend_specific


def test_protocol_freezes_no_refit_three_seed_transfer(tmp_path: Path) -> None:
    protocol = load_cross_backend_transfer_protocol(PROTOCOL)

    models = protocol["artifacts"]["deform_local_residual_models"]
    assert [model["seed"] for model in models] == [42, 43, 44]
    assert protocol["evaluation"]["shrinkage"] == 0.25
    assert protocol["source_panel"]["official_evaluation_read"] is False
    assert (
        protocol["execution_priority"]["required_blocking_run_terminal_before_dispatch"]
        is True
    )
    assert protocol["execution_priority"]["request_or_workflow_trigger_added"] is False

    changed_payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    changed_payload["information_boundary"]["dlo3_official_evaluation_read"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(changed_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="information boundary"):
        load_cross_backend_transfer_protocol(changed)


def test_positive_equal_seed_no_refit_transfer_passes() -> None:
    protocol = load_cross_backend_transfer_protocol(PROTOCOL)
    names, truth, baseline, backend_specific = _problem()
    transferred = {
        42: np.full_like(truth, 0.06),
        43: np.full_like(truth, 0.07),
        44: np.full_like(truth, 0.08),
    }

    result = evaluate_cross_backend_transfer(
        names=names,
        truth=truth,
        pyelastica_backend=baseline,
        pyelastica_specific_candidate=backend_specific,
        transferred_predictions=transferred,
        protocol=protocol,
    )

    assert result["decision"] == "no-refit-cross-backend-transfer-supported"
    assert result["promotion_gate"]["supported"] is True
    assert result["promotion_gate"]["improving_seed_models"] == 3
    assert result["primary_vs_raw_pyelastica"]["wins"] == 8
    assert result["primary_vs_raw_pyelastica"]["losses"] == 0
    assert result["methods"]["deform_no_refit_equal_seed_transfer"] == pytest.approx(
        0.07
    )
    assert result["backend_specific_gain_retained_fraction"] == pytest.approx(0.5)


def test_transfer_requires_seed_stability() -> None:
    protocol = load_cross_backend_transfer_protocol(PROTOCOL)
    names, truth, baseline, backend_specific = _problem()
    transferred = {
        42: np.full_like(truth, 0.01),
        43: np.full_like(truth, 0.11),
        44: np.full_like(truth, 0.11),
    }

    result = evaluate_cross_backend_transfer(
        names=names,
        truth=truth,
        pyelastica_backend=baseline,
        pyelastica_specific_candidate=backend_specific,
        transferred_predictions=transferred,
        protocol=protocol,
    )

    assert result["promotion_gate"]["passed"] is True
    assert result["promotion_gate"]["improving_seed_models"] == 1
    assert result["promotion_gate"]["seed_stability_passed"] is False
    assert result["decision"] == "no-refit-cross-backend-transfer-not-supported"


def test_negative_transfer_fails_primary_gate() -> None:
    protocol = load_cross_backend_transfer_protocol(PROTOCOL)
    names, truth, baseline, backend_specific = _problem()
    transferred = {seed: np.full_like(truth, 0.12) for seed in (42, 43, 44)}

    result = evaluate_cross_backend_transfer(
        names=names,
        truth=truth,
        pyelastica_backend=baseline,
        pyelastica_specific_candidate=backend_specific,
        transferred_predictions=transferred,
        protocol=protocol,
    )

    assert result["promotion_gate"]["passed"] is False
    assert result["primary_vs_raw_pyelastica"]["wins"] == 0
    assert result["decision"] == "no-refit-cross-backend-transfer-not-supported"


def test_paired_summary_rejects_duplicate_names() -> None:
    names, truth, baseline, _ = _problem()
    names[-1] = names[0]
    with pytest.raises(ValueError, match="arrays or names"):
        paired_point_summary(
            baseline,
            baseline,
            truth,
            names,
            repetitions=10,
            seed=1,
        )


def test_runner_seals_method_before_source_payload_loading() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    seal = source.index('method_seal_path = output_root / "method_seal.json"')
    source_load = source.index("trajectories = _load_source_trajectories(")
    assert seal < source_load
    assert "dlo3_official_evaluation_read" in source
    assert "dlo4_or_dlo5_read" in source
    assert "allow_pickle=False" in source
