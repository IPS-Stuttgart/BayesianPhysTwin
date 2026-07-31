import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform_dlo_source import (
    DEFORM_DLO_SOURCE_CONTRACT,
    build_deform_dlo_source_manifest,
    choose_deform_validation_checkpoint,
    deform_mean_coordinate_l1_m,
    evaluate_deform_source_gate,
    load_deform_dlo_source_protocol,
    partition_deform_source_names,
    sha256_file,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "configs" / "sota" / "deform_dlo_source_v1.json"


def test_registered_deform_source_protocol_is_source_only() -> None:
    protocol = load_deform_dlo_source_protocol(PROTOCOL)

    assert protocol["contract"] == DEFORM_DLO_SOURCE_CONTRACT
    assert protocol["data"]["development_partition"] == "train"
    assert protocol["data"]["official_eval_metrics_opened"] is False
    assert protocol["data"]["forbid_eval_reads_during_source_stage"] is True
    assert protocol["training"]["unroll_horizon_frames"] == 50
    assert protocol["training"]["checkpoint_updates"][-1] == 280
    assert protocol["training"]["cublas_workspace_config"] == ":4096:8"


def test_deform_source_protocol_rejects_eval_development(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["data"]["development_partition"] = "eval"
    changed = tmp_path / "protocol.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="official train"):
        load_deform_dlo_source_protocol(changed)


def test_deform_source_partition_is_stable_exhaustive_and_disjoint() -> None:
    names = [f"{index}.pkl" for index in range(56)]
    first = partition_deform_source_names(
        names,
        seed="fixed",
        fit_count=40,
        validation_count=8,
        source_test_count=8,
    )
    second = partition_deform_source_names(
        list(reversed(names)),
        seed="fixed",
        fit_count=40,
        validation_count=8,
        source_test_count=8,
    )

    assert first == second
    assert tuple(map(len, first.values())) == (40, 8, 8)
    assert not set(first["fit"]) & set(first["validation"])
    assert not set(first["fit"]) & set(first["source_test"])
    assert not set(first["validation"]) & set(first["source_test"])
    assert set().union(*map(set, first.values())) == set(names)


def test_deform_source_manifest_binds_only_train_bytes(tmp_path: Path) -> None:
    data_root = tmp_path / "data_set"
    train_root = data_root / "DLO1" / "train"
    eval_root = data_root / "DLO1" / "eval"
    train_root.mkdir(parents=True)
    eval_root.mkdir(parents=True)
    for index in range(56):
        (train_root / f"{index}.pkl").write_bytes(f"train-{index}".encode())
    forbidden = eval_root / "target.pkl"
    forbidden.write_bytes(b"must-not-be-bound")

    manifest = build_deform_dlo_source_manifest(
        PROTOCOL,
        data_root,
        dlo_type="DLO1",
    )

    assert manifest["partition"] == "train"
    assert manifest["official_eval_read"] is False
    assert len(manifest["trajectories"]) == 56
    assert all(
        "/eval/" not in entry["path"].replace("\\", "/")
        for entry in manifest["trajectories"].values()
    )
    assert sha256_file(forbidden) not in {
        entry["sha256"] for entry in manifest["trajectories"].values()
    }


def test_deform_metric_is_coordinate_mean_l1() -> None:
    target = np.zeros((2, 3, 3))
    prediction = target.copy()
    prediction[0, 0, 0] = 0.018

    assert deform_mean_coordinate_l1_m(prediction, target) == pytest.approx(0.001)


def test_deform_validation_checkpoint_uses_validation_only_and_tie_earliest() -> None:
    selected = choose_deform_validation_checkpoint(
        (
            {"update": 0, "validation_l1_m": 0.03},
            {"update": 40, "validation_l1_m": 0.011},
            {"update": 80, "validation_l1_m": 0.011},
        )
    )

    assert selected["update"] == 40


def test_deform_source_gate_requires_parity_and_case_wins() -> None:
    passing = [
        {
            "name": f"case-{index}",
            "model_l1_m": 0.010,
            "persistence_l1_m": 0.02,
        }
        for index in range(8)
    ]
    gate = evaluate_deform_source_gate(
        passing,
        published_reference_l1_m=0.0101,
        published_error_multiplier_max=1.1,
        minimum_persistence_wins=6,
    )
    assert gate["passed"] is True

    weak_wins = [
        {
            "name": f"case-{index}",
            "model_l1_m": 0.010,
            "persistence_l1_m": 0.009 if index < 3 else 0.02,
        }
        for index in range(8)
    ]
    gate = evaluate_deform_source_gate(
        weak_wins,
        published_reference_l1_m=0.0101,
        published_error_multiplier_max=1.1,
        minimum_persistence_wins=6,
    )
    assert gate["parity_passed"] is True
    assert gate["persistence_gate_passed"] is False
    assert gate["passed"] is False
