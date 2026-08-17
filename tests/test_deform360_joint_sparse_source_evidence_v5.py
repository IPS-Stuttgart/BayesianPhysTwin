from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from bayesian_phystwin.deform360_joint_sparse_source_evidence_v5 import (
    assemble_deform360_joint_sparse_source_evidence_v5,
    build_deform360_joint_sparse_source_outcome_v5,
    build_deform360_joint_sparse_source_outcomes_v5,
    build_deform360_joint_sparse_source_prediction_batch_v5,
    build_deform360_joint_sparse_source_prediction_seal_v5,
    publish_deform360_joint_sparse_source_evidence_v5,
    publish_deform360_joint_sparse_source_prediction_batch_v5,
    validate_deform360_joint_sparse_source_outcome_v5,
    validate_deform360_joint_sparse_source_prediction_batch_v5,
    validate_deform360_joint_sparse_source_prediction_seal_v5,
)
from bayesian_phystwin.deform360_joint_sparse_source_gate_v5 import (
    RAW_METHOD_IDS,
    evaluate_deform360_joint_sparse_source_gate_v5,
    load_deform360_joint_sparse_source_execution_lock_v5,
)
from scripts.science.materialize_deform360_joint_sparse_source_evidence_v5 import (
    main,
)

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / (
    "protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json"
)
REVISION = "a" * 40


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _cohort(lock: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(row["object_id"])
            for row in cast(
                Sequence[Mapping[str, Any]],
                cast(Mapping[str, Any], lock["cohort"])["development_objects"],
            )
        )
    )


def _losses() -> dict[str, float]:
    result = {
        "B0_physical_fallback": 10.0,
        "B1_last_causal_residual": 9.0,
        "V1_joint_sparse_visual_guarded": 7.5,
        "T1_contact_anchor_only": 9.25,
        "VT2_joint_sparse_visuotactile_unguarded": 7.5,
        "VT3_joint_sparse_visuotactile_anchor_bias": 7.6,
    }
    assert set(result) == set(RAW_METHOD_IDS)
    return result


def _prediction_methods(outer_id: str, object_id: str) -> dict[str, object]:
    return {
        method_id: {
            "artifact_id": _digest(
                f"{object_id}\0{method_id}"
                if method_id in {"B0_physical_fallback", "B1_last_causal_residual"}
                else f"{outer_id}\0{object_id}\0{method_id}"
            ),
            "predicted_loss_mm": loss,
        }
        for method_id, loss in _losses().items()
    }


def _seals(lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    object_ids = _cohort(lock)
    index = {object_id: position for position, object_id in enumerate(object_ids)}
    result = []
    for outer_id in object_ids:
        for object_id in object_ids:
            role = "held_out" if outer_id == object_id else "training"
            fit_ids = sorted(
                set(object_ids)
                - ({outer_id} if role == "held_out" else {outer_id, object_id})
            )
            result.append(
                build_deform360_joint_sparse_source_prediction_seal_v5(
                    lock=lock,
                    implementation_revision=REVISION,
                    outer_held_out_object_id=outer_id,
                    record_role=role,
                    object_id=object_id,
                    factor_admitted=True,
                    technical_failure=False,
                    physical_mode="warp_twin",
                    risk_score=float(index[object_id] + 1),
                    prediction_fit_artifact_id=_digest(f"fit\0{outer_id}\0{object_id}"),
                    prediction_fit_object_ids=fit_ids,
                    methods=_prediction_methods(outer_id, object_id),
                    source_artifacts={
                        f"forecasts/{outer_id}/{object_id}.json": _digest(
                            f"source\0{outer_id}\0{object_id}"
                        )
                    },
                )
            )
    return result


def _outcomes(
    lock: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = cast(Sequence[Mapping[str, Any]], batch["records"])
    methods_by_seal = {
        str(seal["seal_id"]): {
            method_id: {
                "artifact_id": method["artifact_id"],
                "loss_mm": _losses()[method_id],
            }
            for method_id, method in cast(
                Mapping[str, Mapping[str, Any]], seal["methods"]
            ).items()
        }
        for seal in records
    }
    artifacts_by_seal = {
        str(seal["seal_id"]): {
            f"scores/{seal['seal_id']}.json": _digest(f"score\0{seal['seal_id']}")
        }
        for seal in records
    }
    return build_deform360_joint_sparse_source_outcomes_v5(
        lock=lock,
        prediction_batch=batch,
        methods_by_prediction_seal_id=methods_by_seal,
        scoring_artifacts_by_prediction_seal_id=artifacts_by_seal,
    )


def _fixture() -> tuple[
    Mapping[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(LOCK_PATH)
    seals = _seals(lock)
    batch = build_deform360_joint_sparse_source_prediction_batch_v5(seals, lock)
    outcomes = _outcomes(lock, batch)
    return lock, seals, batch, outcomes


def _reidentify(payload: dict[str, Any], *, id_field: str) -> None:
    from bayesian_phystwin._portable_contracts import content_id

    identity = {key: value for key, value in payload.items() if key != id_field}
    payload[id_field] = content_id(identity)


def test_complete_batch_and_evidence_are_order_invariant_and_gate_ready() -> None:
    lock, seals, batch, outcomes = _fixture()
    reversed_batch = build_deform360_joint_sparse_source_prediction_batch_v5(
        list(reversed(seals)), lock
    )
    assert batch == reversed_batch
    assert batch["record_count"] == 100
    assert batch["fold_count"] == 10

    evidence = assemble_deform360_joint_sparse_source_evidence_v5(
        lock=lock,
        prediction_batch=batch,
        outcomes=outcomes,
    )
    reversed_evidence = assemble_deform360_joint_sparse_source_evidence_v5(
        lock=lock,
        prediction_batch=batch,
        outcomes=list(reversed(outcomes)),
    )
    assert evidence == reversed_evidence
    assert evidence["prediction_batch_id"] == batch["prediction_batch_id"]
    result = evaluate_deform360_joint_sparse_source_gate_v5(evidence, lock)
    assert result["gate_passed"] is True
    assert result["confirmation_access_authorized"] is True
    assert result["aggregate"]["passing_count"] == 8


def test_prediction_seal_rejects_leakage_invalid_state_and_tampering() -> None:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(LOCK_PATH)
    object_ids = _cohort(lock)
    outer_id = object_ids[0]
    other_id = object_ids[1]
    methods = _prediction_methods(outer_id, outer_id)
    base = {
        "lock": lock,
        "implementation_revision": REVISION,
        "outer_held_out_object_id": outer_id,
        "record_role": "held_out",
        "object_id": outer_id,
        "factor_admitted": True,
        "technical_failure": False,
        "physical_mode": "warp_twin",
        "risk_score": 1.0,
        "prediction_fit_artifact_id": _digest("fit"),
        "prediction_fit_object_ids": sorted(set(object_ids) - {outer_id}),
        "methods": methods,
        "source_artifacts": {"source.json": _digest("source")},
    }
    seal = build_deform360_joint_sparse_source_prediction_seal_v5(**base)
    assert validate_deform360_joint_sparse_source_prediction_seal_v5(seal, lock) == seal

    leaked = dict(base)
    leaked["prediction_fit_object_ids"] = list(object_ids)
    with pytest.raises(ValueError, match="fit roster"):
        build_deform360_joint_sparse_source_prediction_seal_v5(**leaked)

    wrong_role = dict(base)
    wrong_role.update(record_role="training", object_id=outer_id)
    with pytest.raises(ValueError, match="cannot be the outer held-out"):
        build_deform360_joint_sparse_source_prediction_seal_v5(**wrong_role)

    wrong_identity = dict(base)
    wrong_identity.update(object_id=other_id)
    with pytest.raises(ValueError, match="differs from its outer fold"):
        build_deform360_joint_sparse_source_prediction_seal_v5(**wrong_identity)

    failed = dict(base)
    failed["technical_failure"] = True
    with pytest.raises(ValueError, match="cannot be factor-admitted"):
        build_deform360_joint_sparse_source_prediction_seal_v5(**failed)

    changed = copy.deepcopy(seal)
    changed["information_boundary"]["development_suffix_opened"] = True
    _reidentify(changed, id_field="seal_id")
    with pytest.raises(ValueError, match="information boundary"):
        validate_deform360_joint_sparse_source_prediction_seal_v5(changed, lock)


def test_prediction_batch_rejects_roster_gaps_and_comparator_drift() -> None:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(LOCK_PATH)
    seals = _seals(lock)
    with pytest.raises(ValueError, match="exactly 100"):
        build_deform360_joint_sparse_source_prediction_batch_v5(seals[:-1], lock)
    with pytest.raises(ValueError, match="repeats a nested forecast"):
        build_deform360_joint_sparse_source_prediction_batch_v5(
            [*seals[:-1], seals[0]], lock
        )

    drifted = copy.deepcopy(seals)
    methods = cast(dict[str, dict[str, Any]], drifted[1]["methods"])
    methods["B0_physical_fallback"]["predicted_loss_mm"] = 10.5
    _reidentify(drifted[1], id_field="seal_id")
    with pytest.raises(ValueError, match="comparator prediction differs"):
        build_deform360_joint_sparse_source_prediction_batch_v5(drifted, lock)

    batch = build_deform360_joint_sparse_source_prediction_batch_v5(seals, lock)
    changed = copy.deepcopy(batch)
    changed["record_count"] = 99
    _reidentify(changed, id_field="prediction_batch_id")
    with pytest.raises(ValueError, match="content identity"):
        validate_deform360_joint_sparse_source_prediction_batch_v5(changed, lock)


def test_outcome_binds_exact_batch_seal_artifacts_and_boundary() -> None:
    lock, _seals_value, batch, outcomes = _fixture()
    outcome = outcomes[0]
    assert (
        validate_deform360_joint_sparse_source_outcome_v5(
            outcome,
            lock=lock,
            prediction_batch=batch,
        )
        == outcome
    )
    seal = cast(Sequence[Mapping[str, Any]], batch["records"])[0]
    methods = {
        method_id: {
            "artifact_id": method["artifact_id"],
            "loss_mm": _losses()[method_id],
        }
        for method_id, method in cast(
            Mapping[str, Mapping[str, Any]], seal["methods"]
        ).items()
    }
    cast(dict[str, Any], methods["VT2_joint_sparse_visuotactile_unguarded"])[
        "artifact_id"
    ] = "0" * 64
    with pytest.raises(ValueError, match="differs from seal"):
        build_deform360_joint_sparse_source_outcome_v5(
            lock=lock,
            prediction_batch=batch,
            prediction_seal_id=str(seal["seal_id"]),
            methods=methods,
            scoring_artifacts={"score.json": _digest("score")},
        )

    changed = copy.deepcopy(outcome)
    changed["information_boundary"][
        "development_suffix_opened_after_prediction_batch"
    ] = False
    _reidentify(changed, id_field="outcome_id")
    with pytest.raises(ValueError, match="information boundary"):
        validate_deform360_joint_sparse_source_outcome_v5(
            changed,
            lock=lock,
            prediction_batch=batch,
        )


def test_assembly_rejects_missing_duplicate_and_foreign_outcomes() -> None:
    lock, seals, batch, outcomes = _fixture()
    with pytest.raises(ValueError, match="exactly 100"):
        assemble_deform360_joint_sparse_source_evidence_v5(
            lock=lock,
            prediction_batch=batch,
            outcomes=outcomes[:-1],
        )
    with pytest.raises(ValueError, match="repeats an outcome"):
        assemble_deform360_joint_sparse_source_evidence_v5(
            lock=lock,
            prediction_batch=batch,
            outcomes=[*outcomes[:-1], outcomes[0]],
        )

    changed_seals = copy.deepcopy(seals)
    methods = cast(dict[str, dict[str, Any]], changed_seals[0]["methods"])
    methods["VT2_joint_sparse_visuotactile_unguarded"]["predicted_loss_mm"] = 7.4
    _reidentify(changed_seals[0], id_field="seal_id")
    foreign_batch = build_deform360_joint_sparse_source_prediction_batch_v5(
        changed_seals, lock
    )
    foreign_outcome = _outcomes(lock, foreign_batch)[0]
    with pytest.raises(ValueError, match="another prediction batch"):
        assemble_deform360_joint_sparse_source_evidence_v5(
            lock=lock,
            prediction_batch=batch,
            outcomes=[foreign_outcome, *outcomes[1:]],
        )


def test_atomic_publication_and_cli_complete_the_two_stage_barrier(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lock, seals, batch, outcomes = _fixture()
    batch_path = tmp_path / "prediction-batch.json"
    evidence_path = tmp_path / "source-evidence.json"
    publish_deform360_joint_sparse_source_prediction_batch_v5(
        batch,
        lock=lock,
        output_path=batch_path,
    )
    with pytest.raises(FileExistsError):
        publish_deform360_joint_sparse_source_prediction_batch_v5(
            batch,
            lock=lock,
            output_path=batch_path,
        )
    evidence = assemble_deform360_joint_sparse_source_evidence_v5(
        lock=lock,
        prediction_batch=batch,
        outcomes=outcomes,
    )
    publish_deform360_joint_sparse_source_evidence_v5(
        evidence,
        lock=lock,
        output_path=evidence_path,
    )
    with pytest.raises(FileExistsError):
        publish_deform360_joint_sparse_source_evidence_v5(
            evidence,
            lock=lock,
            output_path=evidence_path,
        )

    seal_paths = []
    for index, seal in enumerate(seals):
        path = tmp_path / f"seal-{index:03d}.json"
        path.write_text(json.dumps(seal), encoding="utf-8")
        seal_paths.append(path)
    cli_batch = tmp_path / "cli-prediction-batch.json"
    seal_arguments = [
        "seal-batch",
        "--execution-lock",
        str(LOCK_PATH),
        "--output",
        str(cli_batch),
    ]
    for path in seal_paths:
        seal_arguments.extend(["--prediction-seal", str(path)])
    assert main(seal_arguments) == 0
    capsys.readouterr()

    outcome_paths = []
    for index, outcome in enumerate(outcomes):
        path = tmp_path / f"outcome-{index:03d}.json"
        path.write_text(json.dumps(outcome), encoding="utf-8")
        outcome_paths.append(path)
    cli_evidence = tmp_path / "cli-source-evidence.json"
    outcome_arguments = [
        "assemble",
        "--execution-lock",
        str(LOCK_PATH),
        "--prediction-batch",
        str(cli_batch),
        "--output",
        str(cli_evidence),
    ]
    for path in outcome_paths:
        outcome_arguments.extend(["--outcome", str(path)])
    assert main(outcome_arguments) == 0
    capsys.readouterr()
    assert json.loads(cli_batch.read_text(encoding="utf-8")) == batch
    assert json.loads(cli_evidence.read_text(encoding="utf-8")) == evidence


def test_malformed_contracts_fail_closed_across_all_public_layers(
    tmp_path: Path,
) -> None:
    lock, seals, batch, outcomes = _fixture()
    object_ids = _cohort(lock)
    base = seals[0]

    with pytest.raises(ValueError, match="JSON object"):
        validate_deform360_joint_sparse_source_prediction_seal_v5([], lock)

    for field, value, message in (
        ("schema", "wrong", "schema changed"),
        ("schema_version", 2, "version changed"),
        ("semantics", "wrong", "semantics changed"),
    ):
        changed = copy.deepcopy(base)
        changed[field] = value
        with pytest.raises(ValueError, match=message):
            validate_deform360_joint_sparse_source_prediction_seal_v5(changed, lock)

    changed = copy.deepcopy(base)
    changed["risk_score"] = 2.0
    with pytest.raises(ValueError, match="content identity"):
        validate_deform360_joint_sparse_source_prediction_seal_v5(changed, lock)

    builder = {
        "lock": lock,
        "implementation_revision": REVISION,
        "outer_held_out_object_id": object_ids[0],
        "record_role": "held_out",
        "object_id": object_ids[0],
        "factor_admitted": True,
        "technical_failure": False,
        "physical_mode": "warp_twin",
        "risk_score": 1.0,
        "prediction_fit_artifact_id": _digest("fit-edge"),
        "prediction_fit_object_ids": sorted(set(object_ids) - {object_ids[0]}),
        "methods": _prediction_methods(object_ids[0], object_ids[0]),
        "source_artifacts": {"source.json": _digest("source-edge")},
    }
    for update, message in (
        ({"outer_held_out_object_id": " outside"}, "canonical string"),
        ({"outer_held_out_object_id": "outside"}, "outside the locked cohort"),
        ({"object_id": "outside"}, "outside the locked cohort"),
        ({"record_role": "invalid"}, "record_role changed"),
        ({"physical_mode": "invalid"}, "physical_mode changed"),
        ({"risk_score": True}, "finite nonnegative"),
        ({"risk_score": float("nan")}, "finite nonnegative"),
        ({"risk_score": -1.0}, "finite nonnegative"),
    ):
        arguments = dict(builder)
        arguments.update(update)
        with pytest.raises(ValueError, match=message):
            build_deform360_joint_sparse_source_prediction_seal_v5(**arguments)

    methods = copy.deepcopy(builder["methods"])
    cast(dict[str, Any], methods).pop("T1_contact_anchor_only")
    arguments = dict(builder)
    arguments["methods"] = methods
    with pytest.raises(ValueError, match="fields changed"):
        build_deform360_joint_sparse_source_prediction_seal_v5(**arguments)

    malformed_lock = copy.deepcopy(lock)
    cohort = cast(dict[str, Any], malformed_lock["cohort"])
    rows = cast(list[dict[str, Any]], cohort["development_objects"])
    rows[1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError, match="repeats a development object"):
        build_deform360_joint_sparse_source_prediction_seal_v5(
            **builder | {"lock": malformed_lock}
        )

    malformed_lock = copy.deepcopy(lock)
    rows = cast(
        list[dict[str, Any]],
        cast(dict[str, Any], malformed_lock["cohort"])["development_objects"],
    )
    rows[0]["stratum"] = "other"
    with pytest.raises(ValueError, match="stratum changed"):
        build_deform360_joint_sparse_source_prediction_seal_v5(
            **builder | {"lock": malformed_lock}
        )

    malformed_lock = copy.deepcopy(lock)
    cast(dict[str, Any], malformed_lock["cohort"])["development_objects"] = cast(
        list[Any],
        cast(dict[str, Any], malformed_lock["cohort"])["development_objects"],
    )[:-1]
    with pytest.raises(ValueError, match="exactly ten"):
        build_deform360_joint_sparse_source_prediction_seal_v5(
            **builder | {"lock": malformed_lock}
        )

    mixed = copy.deepcopy(seals)
    mixed[0]["implementation_revision"] = "b" * 40
    _reidentify(mixed[0], id_field="seal_id")
    with pytest.raises(ValueError, match="mixes implementation revisions"):
        build_deform360_joint_sparse_source_prediction_batch_v5(mixed, lock)

    for field, value, message in (
        ("schema", "wrong", "schema changed"),
        ("schema_version", 2, "version changed"),
        ("semantics", "wrong", "semantics changed"),
    ):
        changed_batch = copy.deepcopy(batch)
        changed_batch[field] = value
        with pytest.raises(ValueError, match=message):
            validate_deform360_joint_sparse_source_prediction_batch_v5(
                changed_batch, lock
            )
    changed_batch = copy.deepcopy(batch)
    changed_batch["information_boundary"]["development_suffix_opened"] = True
    with pytest.raises(ValueError, match="information boundary"):
        validate_deform360_joint_sparse_source_prediction_batch_v5(changed_batch, lock)
    changed_batch = copy.deepcopy(batch)
    changed_batch["records"] = "not-an-array"
    with pytest.raises(ValueError, match="JSON array"):
        validate_deform360_joint_sparse_source_prediction_batch_v5(changed_batch, lock)

    with pytest.raises(ValueError, match="outside the prediction batch"):
        build_deform360_joint_sparse_source_outcome_v5(
            lock=lock,
            prediction_batch=batch,
            prediction_seal_id="0" * 64,
            methods=cast(Mapping[str, Any], outcomes[0]["methods"]),
            scoring_artifacts={"score.json": _digest("score-edge")},
        )

    records = cast(Sequence[Mapping[str, Any]], batch["records"])
    methods_by_seal = {
        str(record["seal_id"]): cast(Mapping[str, Any], outcomes[index]["methods"])
        for index, record in enumerate(records)
    }
    artifacts_by_seal = {
        str(record["seal_id"]): {"score.json": _digest(str(record["seal_id"]))}
        for record in records
    }
    methods_by_seal.pop(next(iter(methods_by_seal)))
    with pytest.raises(ValueError, match="scored method roster"):
        build_deform360_joint_sparse_source_outcomes_v5(
            lock=lock,
            prediction_batch=batch,
            methods_by_prediction_seal_id=methods_by_seal,
            scoring_artifacts_by_prediction_seal_id=artifacts_by_seal,
        )
    methods_by_seal = {
        str(record["seal_id"]): cast(Mapping[str, Any], outcomes[index]["methods"])
        for index, record in enumerate(records)
    }
    artifacts_by_seal.pop(next(iter(artifacts_by_seal)))
    with pytest.raises(ValueError, match="scoring-artifact roster"):
        build_deform360_joint_sparse_source_outcomes_v5(
            lock=lock,
            prediction_batch=batch,
            methods_by_prediction_seal_id=methods_by_seal,
            scoring_artifacts_by_prediction_seal_id=artifacts_by_seal,
        )

    for field, value, message in (
        ("schema", "wrong", "schema changed"),
        ("schema_version", 2, "version changed"),
        ("semantics", "wrong", "semantics changed"),
    ):
        changed_outcome = copy.deepcopy(outcomes[0])
        changed_outcome[field] = value
        with pytest.raises(ValueError, match=message):
            validate_deform360_joint_sparse_source_outcome_v5(
                changed_outcome,
                lock=lock,
                prediction_batch=batch,
            )
    changed_outcome = copy.deepcopy(outcomes[0])
    scoring = cast(dict[str, str], changed_outcome["scoring_artifacts"])
    scoring[next(iter(scoring))] = "f" * 64
    with pytest.raises(ValueError, match="content identity"):
        validate_deform360_joint_sparse_source_outcome_v5(
            changed_outcome,
            lock=lock,
            prediction_batch=batch,
        )

    from bayesian_phystwin.deform360_joint_sparse_source_evidence_v5 import (
        load_source_execution_lock_and_artifacts_v5,
    )

    with pytest.raises(ValueError, match="must not be empty"):
        load_source_execution_lock_and_artifacts_v5(
            execution_lock_path=LOCK_PATH,
            artifact_paths=[],
            label="test",
        )
    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a JSON object"):
        load_source_execution_lock_and_artifacts_v5(
            execution_lock_path=LOCK_PATH,
            artifact_paths=[malformed],
            label="test",
        )


class _DiscardCapture:
    def readouterr(self) -> None:
        return None


def exercise_source_evidence_contracts_v5(tmp_path: Path) -> None:
    """Exercise this contract from the registered stable-core source-gate suite."""

    test_complete_batch_and_evidence_are_order_invariant_and_gate_ready()
    test_prediction_seal_rejects_leakage_invalid_state_and_tampering()
    test_prediction_batch_rejects_roster_gaps_and_comparator_drift()
    test_outcome_binds_exact_batch_seal_artifacts_and_boundary()
    test_assembly_rejects_missing_duplicate_and_foreign_outcomes()
    atomic = tmp_path / "atomic"
    atomic.mkdir()
    test_atomic_publication_and_cli_complete_the_two_stage_barrier(
        atomic,
        cast(pytest.CaptureFixture[str], _DiscardCapture()),
    )
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    test_malformed_contracts_fail_closed_across_all_public_layers(malformed)
