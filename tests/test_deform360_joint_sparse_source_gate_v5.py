from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_joint_sparse_source_gate_v5 import (
    RAW_METHOD_IDS,
    evaluate_deform360_joint_sparse_source_gate_v5,
    load_deform360_joint_sparse_source_execution_lock_v5,
    parse_deform360_joint_sparse_source_evidence_v5,
    publish_deform360_joint_sparse_source_gate_v5,
)

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / (
    "protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json"
)
POLICY_PATH = ROOT / (
    "protocols/locks/deform360_official_hub_joint_sparse_prospective_v5.json"
)
SELECTION_PATH = ROOT / (
    "protocols/locks/deform360_official_hub_visuotactile_v1_selection.json"
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _method(
    object_id: str,
    method_id: str,
    loss_mm: float,
) -> dict[str, object]:
    return {
        "artifact_id": _digest(f"{object_id}\0{method_id}"),
        "loss_mm": loss_mm,
        "predicted_loss_mm": loss_mm,
    }


def _evidence(
    *,
    equal_risk_scores: bool = False,
    harmful_object_index: int | None = None,
) -> dict[str, object]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    records = []
    for index, row in enumerate(lock["cohort"]["development_objects"]):
        object_id = row["object_id"]
        candidate_loss = 11.0 if index == harmful_object_index else 7.5
        losses = {
            "B0_physical_fallback": 10.0,
            "B1_last_causal_residual": 9.0,
            "V1_joint_sparse_visual_guarded": 8.5,
            "T1_contact_anchor_only": 9.25,
            "VT2_joint_sparse_visuotactile_unguarded": candidate_loss,
            "VT3_joint_sparse_visuotactile_anchor_bias": 7.6,
        }
        assert set(losses) == set(RAW_METHOD_IDS)
        records.append(
            {
                "episode_id": row["episode_id"],
                "factor_admitted": True,
                "methods": {
                    method_id: _method(object_id, method_id, loss)
                    for method_id, loss in losses.items()
                },
                "object_id": object_id,
                "physical_mode": "warp_twin",
                "risk_score": 1.0 if equal_risk_scores else float(index + 1),
                "stratum": row["stratum"],
                "technical_failure": False,
            }
        )
    folds = []
    object_ids = [str(record["object_id"]) for record in records]
    for held_out_index, held_out in enumerate(records):
        held_out_id = str(held_out["object_id"])
        held_out_record = copy.deepcopy(held_out)
        held_out_record["prediction_fit_artifact_id"] = _digest(
            f"outer-fit\0{held_out_id}"
        )
        held_out_record["prediction_fit_object_ids"] = sorted(
            set(object_ids) - {held_out_id}
        )
        training_records = []
        for index, record in enumerate(records):
            if index == held_out_index:
                continue
            training_record = copy.deepcopy(record)
            training_id = str(training_record["object_id"])
            training_record["prediction_fit_artifact_id"] = _digest(
                f"inner-fit\0{held_out_id}\0{training_id}"
            )
            training_record["prediction_fit_object_ids"] = sorted(
                set(object_ids) - {held_out_id, training_id}
            )
            training_records.append(training_record)
        folds.append(
            {
                "fold_prediction_seal_id": _digest(f"fold-seal\0{held_out_id}"),
                "held_out_object_id": held_out_id,
                "held_out_record": held_out_record,
                "training_records": training_records,
            }
        )
    descriptor: dict[str, object] = {
        "schema": "bayesian-phystwin.deform360-joint-sparse-source-evidence",
        "schema_version": 1,
        "semantics": "prediction-sealed-public-development-suffix-evidence-v1",
        "execution_lock_id": lock["execution_lock_id"],
        "implementation_revision": "a" * 40,
        "information_boundary": {
            "confirmation_outcomes_opened": False,
            "confirmation_payloads_opened": False,
            "development_suffix_opened_before_prediction_seal": False,
            "future_object_observations_used_for_prediction": False,
            "human_approval_used": False,
            "new_measurements_collected": False,
            "public_released_measurements_used": True,
            "replacement_allowed": False,
            "target_outcomes_used": False,
        },
        "prediction_batch_id": _digest("prediction-batch"),
        "prospective_policy_id": lock["prospective_policy"]["policy_id"],
        "folds": folds,
        "selection_sha256": lock["cohort"]["selection_sha256"],
    }
    return {"evidence_id": content_id(descriptor), **descriptor}


def _reidentify(evidence: dict[str, object]) -> dict[str, object]:
    descriptor = {key: value for key, value in evidence.items() if key != "evidence_id"}
    evidence["evidence_id"] = content_id(descriptor)
    return evidence


def test_execution_lock_binds_public_data_and_machine_gate() -> None:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(LOCK_PATH)

    assert lock["execution_lock_id"] == (
        "30a216f42e46ff92df3ce49f49a635d57b41b5f9019b27d4166127650719aced"
    )
    assert lock["public_measurements"] == {
        "dataset_repository": "brownu/deform360",
        "dataset_revision": "f804696d7a133908c7497ffdab43819d879b5cbc",
        "human_approval_required": False,
        "measurement_modalities": [
            "released-calibrated-rgb",
            "released-tactile-arrays",
            "released-robot-state-and-action",
            "released-camera-calibration",
            "released-held-out-view-geometry",
        ],
        "new_measurements_required": False,
        "prob4d_role": "used-as-the-frozen-probabilistic-visual-observation-feeder",
        "released_real_world_recordings": True,
    }
    assert lock["physical_baseline"]["generation_rule"] == (
        "automatic-warp-twin-when-admissible-otherwise-exact-persistence-v1"
    )
    assert lock["source_gate"]["minimum_passing_objects"] == 8
    assert lock["source_gate"]["minimum_passing_objects_per_stratum"] == 4
    assert lock["source_gate"]["tie_policy"] == (
        "accept-complete-tied-score-blocks-never-split-by-object-id"
    )


def test_execution_lock_binds_unchanged_policy_selection_and_code() -> None:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(LOCK_PATH)

    assert (
        hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()
        == (lock["prospective_policy"]["file_sha256"])
    )
    assert (
        hashlib.sha256(SELECTION_PATH.read_bytes()).hexdigest()
        == (lock["cohort"]["selection_artifact_file_sha256"])
    )
    for relative, expected in lock["physical_baseline"]["source_files_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    for path_key, digest_key in (
        ("source_evaluator_path", "source_evaluator_file_sha256"),
        ("source_runner_path", "source_runner_file_sha256"),
    ):
        assert (
            hashlib.sha256(
                (ROOT / lock["source_gate"][path_key]).read_bytes()
            ).hexdigest()
            == lock["source_gate"][digest_key]
        )


def test_transferable_source_evidence_passes_without_human_approval() -> None:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(LOCK_PATH)
    result = evaluate_deform360_joint_sparse_source_gate_v5(_evidence(), lock)

    assert result["gate_passed"] is True
    assert result["confirmation_access_authorized"] is True
    assert result["aggregate"]["passing_count"] == 8
    assert result["aggregate"]["passing_count_by_stratum"] == {
        "sheet": 4,
        "volumetric": 4,
    }
    assert result["aggregate"]["accepted_count"] == 8
    assert result["full_source_fit"]["accepted_count"] == 9
    assert result["information_boundary"]["human_approval_required"] is False
    assert result["information_boundary"]["new_measurements_required"] is False
    authorization = result["confirmation_opening_authorization"]
    assert authorization["authorized"] is True
    assert authorization["confirmation_payloads_opened"] is False


def test_rejected_outer_folds_deploy_exact_physical_artifact() -> None:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(LOCK_PATH)
    evidence = _evidence()
    result = evaluate_deform360_joint_sparse_source_gate_v5(evidence, lock)
    records = {
        fold["held_out_object_id"]: fold["held_out_record"]
        for fold in evidence["folds"]
    }

    rejected = [fold for fold in result["folds"] if not fold["accepted"]]
    assert len(rejected) == 2
    for fold in rejected:
        fallback = records[fold["object_id"]]["methods"]["B0_physical_fallback"]
        assert fold["deployed_artifact_id"] == fallback["artifact_id"]
        assert fold["deployed_loss_mm"] == fallback["loss_mm"]


def test_tied_risk_scores_cannot_be_split_by_object_identity() -> None:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(LOCK_PATH)
    result = evaluate_deform360_joint_sparse_source_gate_v5(
        _evidence(equal_risk_scores=True), lock
    )

    assert result["gate_passed"] is False
    assert result["confirmation_access_authorized"] is False
    assert result["aggregate"]["accepted_count"] == 0
    assert result["full_source_fit"]["threshold"] is None
    assert all(fold["threshold"] is None for fold in result["folds"])


def test_low_risk_harmful_update_fails_source_gate() -> None:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(LOCK_PATH)
    result = evaluate_deform360_joint_sparse_source_gate_v5(
        _evidence(harmful_object_index=0), lock
    )

    assert result["gate_passed"] is False
    assert result["confirmation_opening_authorization"] is None
    assert result["checks"]["minimum_passing_objects"] is False


def test_source_evidence_rejects_future_input_and_duplicate_objects() -> None:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(LOCK_PATH)
    future = _evidence()
    future["information_boundary"]["future_object_observations_used_for_prediction"] = (
        True
    )
    _reidentify(future)
    with pytest.raises(ValueError, match="information boundary"):
        parse_deform360_joint_sparse_source_evidence_v5(future, lock)

    duplicate = _evidence()
    duplicate["folds"][0]["training_records"][1] = copy.deepcopy(
        duplicate["folds"][0]["training_records"][0]
    )
    _reidentify(duplicate)
    with pytest.raises(ValueError, match="duplicate source object"):
        parse_deform360_joint_sparse_source_evidence_v5(duplicate, lock)

    leaky = _evidence()
    outer = leaky["folds"][0]
    training = outer["training_records"][0]
    fit_ids = list(training["prediction_fit_object_ids"])
    fit_ids[0] = outer["held_out_object_id"]
    training["prediction_fit_object_ids"] = sorted(set(fit_ids))
    _reidentify(leaky)
    with pytest.raises(ValueError, match="not inner cross-fitted"):
        parse_deform360_joint_sparse_source_evidence_v5(leaky, lock)


def test_publication_is_atomic_and_refuses_silent_overwrite(tmp_path: Path) -> None:
    evidence_path = tmp_path / "source-evidence.json"
    evidence_path.write_text(
        json.dumps(_evidence(), sort_keys=True) + "\n", encoding="utf-8"
    )
    output = tmp_path / "source-gate-result.json"

    first = publish_deform360_joint_sparse_source_gate_v5(
        evidence_path, LOCK_PATH, output
    )
    assert json.loads(output.read_text(encoding="utf-8")) == first
    with pytest.raises(FileExistsError):
        publish_deform360_joint_sparse_source_gate_v5(evidence_path, LOCK_PATH, output)
