from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import bayesian_phystwin.deform360_fresh_object_session_source_v6 as v6_legacy
import bayesian_phystwin.deform360_fresh_object_session_source_v6_1 as v61
from bayesian_phystwin.deform360_fresh_object_session_source_v6 import (
    B0,
    B1,
    CHALLENGER_VARIANTS,
    D1_NATIVE,
    VARIANT_IDS,
    VT1_OBSERVED,
    VT1_SANDWICH,
    VT1_WORKING,
)

ROOT = Path(__file__).resolve().parents[1]
REPAIR_PATH = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_nested_source_contract_repair.json"
)
UPSTREAM_REVISION = v61.UPSTREAM_REVISION
CANDIDATE_REVISION = "2" * 40
UPSTREAM_BATCH_ID = v61.UPSTREAM_PREDICTION_BATCH_ID


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _cohort() -> dict[str, tuple[int, str]]:
    return {
        f"object-{index}": (
            index,
            "sheet" if index < 5 else "volumetric",
        )
        for index in range(10)
    }


def _variant(
    variant_id: str,
    *,
    outer_id: str,
    object_id: str,
    unavailable: set[str],
) -> dict[str, Any]:
    challenger = variant_id in CHALLENGER_VARIANTS
    fit_ids = sorted(set(_cohort()) - {outer_id, object_id}) if challenger else []
    if variant_id in unavailable:
        return {
            "available": False,
            "prediction_artifact_id": None,
            "fit_artifact_id": None,
            "fit_object_ids": fit_ids,
            "covariance_artifact_id": None,
            "risk_score": None,
            "unavailable_reason": "registered-covariance-unavailable",
        }
    point_family = (
        "vt1" if variant_id in {VT1_WORKING, VT1_OBSERVED, VT1_SANDWICH} else variant_id
    )
    return {
        "available": True,
        "prediction_artifact_id": _digest(
            f"prediction/{outer_id}/{object_id}/{point_family}"
        ),
        "fit_artifact_id": _digest(f"fit/{outer_id}/{point_family}"),
        "fit_object_ids": fit_ids,
        "covariance_artifact_id": _digest(
            f"covariance/{outer_id}/{object_id}/{variant_id}"
        ),
        "risk_score": 0.1 if challenger else None,
        "unavailable_reason": None,
    }


def _prediction(
    outer_id: str,
    object_id: str,
    *,
    unavailable: set[str] | None = None,
) -> dict[str, Any]:
    missing = set() if unavailable is None else unavailable
    return v61.build_deform360_v6_raw_nested_prediction(
        cohort=_cohort(),
        upstream_prediction_batch_id=UPSTREAM_BATCH_ID,
        upstream_revision=UPSTREAM_REVISION,
        candidate_revision=CANDIDATE_REVISION,
        outer_held_out_object_id=outer_id,
        object_id=object_id,
        variants={
            variant_id: _variant(
                variant_id,
                outer_id=outer_id,
                object_id=object_id,
                unavailable=missing,
            )
            for variant_id in VARIANT_IDS
        },
        source_artifacts={
            f"prefix/{outer_id}/{object_id}.json": _digest(
                f"source/{outer_id}/{object_id}"
            )
        },
    )


def _batch(*, unavailable: set[str] | None = None) -> dict[str, Any]:
    records = [
        _prediction(outer_id, object_id, unavailable=unavailable)
        for outer_id in sorted(_cohort())
        for object_id in sorted(_cohort())
    ]
    return v61.build_deform360_v6_raw_nested_batch(records, cohort=_cohort())


def _score_values(variant_id: str, object_id: str) -> tuple[float, float, float, float]:
    index = int(object_id.rsplit("-", 1)[1])
    maximum = 2.0 if index == 0 else 1.0
    if variant_id == B0:
        point = 10.0
    elif variant_id == B1:
        point = 9.0
    elif variant_id == D1_NATIVE:
        point = 7.0
    else:
        point = 8.0
    return point, 3.0, 0.0, maximum


def _outcome_variants(prediction: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for variant_id in VARIANT_IDS:
        predicted = prediction["variants"][variant_id]
        score_variant = variant_id if predicted["available"] else B0
        point, mean_squared, logdet, maximum = _score_values(
            score_variant, prediction["object_id"]
        )
        result[variant_id] = {
            "available": predicted["available"],
            "prediction_artifact_id": predicted["prediction_artifact_id"],
            "query_count": 18,
            "point_loss": point,
            "mean_raw_mahalanobis_squared": mean_squared,
            "mean_log_determinant": logdet,
            "maximum_raw_mahalanobis_norm": maximum,
            "mean_raw_radius": 1.0,
        }
    return result


def _evidence(
    *, unavailable: set[str] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    batch = _batch(unavailable=unavailable)
    outcomes = [
        v61.build_deform360_v6_raw_nested_outcome(
            prediction_batch=batch,
            prediction_record_id=prediction["prediction_record_id"],
            variants=_outcome_variants(prediction),
            scoring_artifacts={
                f"score/{prediction['outer_held_out_object_id']}/{prediction['object_id']}.json": _digest(
                    f"score/{prediction['prediction_record_id']}"
                )
            },
        )
        for prediction in batch["records"]
    ]
    return (
        batch,
        outcomes,
        v61.assemble_deform360_v6_nested_evidence(
            prediction_batch=batch,
            outcomes=outcomes,
            cohort=_cohort(),
        ),
    )


def test_repair_is_content_addressed_and_source_outcome_closed() -> None:
    repair = v61.load_deform360_v6_nested_source_repair(REPAIR_PATH)

    assert repair["amendment_id"] == v61.NESTED_REPAIR_ID
    assert repair["correction"]["legacy_precomputed_acceptance_not_admissible"]
    assert repair["correction"]["guard_threshold_fitted_inside_each_outer_fold"]
    assert (
        repair["covariance_calibration"][
            "finite_sample_split_conformal_coverage_claimed"
        ]
        is False
    )
    assert (
        repair["information_boundary"]["source_outcomes_used_to_design_this_repair"]
        is False
    )


def test_legacy_public_evaluator_is_nonprogressing() -> None:
    with pytest.raises(
        RuntimeError, match="legacy Deform360 v6 source assembler is retired"
    ):
        v6_legacy.assemble_deform360_v6_source_evidence(
            prediction_batch={},
            outcomes=[],
        )
    with pytest.raises(
        RuntimeError, match="legacy Deform360 v6 source evaluator is retired"
    ):
        v6_legacy.evaluate_deform360_v6_source_gate({}, {})


def test_scalar_contract_helpers_reject_noncanonical_values() -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        v61._mapping([], name="value")  # noqa: SLF001
    with pytest.raises(ValueError, match="must be a JSON array"):
        v61._sequence("not-an-array", name="value")  # noqa: SLF001
    with pytest.raises(ValueError, match="must be a canonical string"):
        v61._identifier(" padded ", name="value")  # noqa: SLF001
    with pytest.raises(ValueError, match="must be a finite real number"):
        v61._finite(True, name="value")  # noqa: SLF001
    with pytest.raises(ValueError, match="must be finite and at least"):
        v61._finite(-1.0, name="value", minimum=0.0)  # noqa: SLF001
    with pytest.raises(ValueError, match="content identity changed"):
        v61._content_identity(  # noqa: SLF001
            {"artifact_id": "0" * 64, "value": 1},
            field="artifact_id",
            name="artifact",
        )


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("base_policy", "policy_id"), "0" * 64, "binds another policy"),
        (
            (
                "correction",
                "guard_threshold_fitted_inside_each_outer_fold",
            ),
            False,
            "correction changed",
        ),
        (
            ("covariance_calibration", "calibration_method"),
            "split-conformal",
            "covariance calibration changed",
        ),
        (
            ("nested_selection", "variant_tie_break"),
            [],
            "selection order changed",
        ),
        (
            ("information_boundary", "source_outcomes_used_to_design_this_repair"),
            True,
            "crossed its information boundary",
        ),
        (
            ("observation_feeder", "new_prob4d_inference_run"),
            True,
            "observation feeder changed",
        ),
    ],
)
def test_repair_loader_rejects_bound_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, str],
    replacement: object,
    message: str,
) -> None:
    repair = json.loads(REPAIR_PATH.read_text(encoding="utf-8"))
    repair[path[0]][path[1]] = replacement
    body = {key: value for key, value in repair.items() if key != "amendment_id"}
    amendment_id = v61.content_id(body)
    repair["amendment_id"] = amendment_id
    monkeypatch.setattr(v61, "NESTED_REPAIR_ID", amendment_id)
    changed = tmp_path / "repair.json"
    changed.write_text(json.dumps(repair), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        v61.load_deform360_v6_nested_source_repair(changed)


def test_repair_loader_rejects_schema_and_identity_drift(tmp_path: Path) -> None:
    repair = json.loads(REPAIR_PATH.read_text(encoding="utf-8"))
    repair["schema"] = "changed"
    changed = tmp_path / "schema.json"
    changed.write_text(json.dumps(repair), encoding="utf-8")
    with pytest.raises(ValueError, match="schema changed"):
        v61.load_deform360_v6_nested_source_repair(changed)

    repair = json.loads(REPAIR_PATH.read_text(encoding="utf-8"))
    repair["amendment_id"] = "0" * 64
    changed = tmp_path / "identity.json"
    changed.write_text(json.dumps(repair), encoding="utf-8")
    with pytest.raises(ValueError, match="content identity changed"):
        v61.load_deform360_v6_nested_source_repair(changed)


def test_cohort_contract_rejects_shape_stratum_and_balance() -> None:
    malformed = _cohort()
    malformed["object-0"] = (0,)  # type: ignore[assignment]
    with pytest.raises(ValueError, match="identity must be"):
        v61._cohort(malformed)  # noqa: SLF001

    wrong_stratum = _cohort()
    wrong_stratum["object-0"] = (0, "rope")
    with pytest.raises(ValueError, match="stratum changed"):
        v61._cohort(wrong_stratum)  # noqa: SLF001

    incomplete = _cohort()
    incomplete.pop("object-0")
    with pytest.raises(ValueError, match="five units per stratum"):
        v61._cohort(incomplete)  # noqa: SLF001


def test_raw_variant_contract_rejects_roster_and_availability_drift() -> None:
    variants = {
        variant_id: _variant(
            variant_id,
            outer_id="object-0",
            object_id="object-1",
            unavailable=set(),
        )
        for variant_id in VARIANT_IDS
    }

    changed = copy.deepcopy(variants)
    changed[D1_NATIVE]["fit_object_ids"] = []
    with pytest.raises(ValueError, match="fit roster changed"):
        _build_prediction_with_variants(changed)

    changed = copy.deepcopy(variants)
    changed[B0].update(
        available=False,
        prediction_artifact_id=None,
        fit_artifact_id=None,
        covariance_artifact_id=None,
        unavailable_reason="missing",
    )
    with pytest.raises(ValueError, match="must be available"):
        _build_prediction_with_variants(changed)

    changed = copy.deepcopy(variants)
    changed[B0]["risk_score"] = 0.1
    with pytest.raises(ValueError, match="must not carry a risk score"):
        _build_prediction_with_variants(changed)

    changed = copy.deepcopy(variants)
    changed[B0]["unavailable_reason"] = "contradiction"
    with pytest.raises(ValueError, match="has an unavailable reason"):
        _build_prediction_with_variants(changed)

    changed = copy.deepcopy(variants)
    changed[VT1_OBSERVED].update(
        available=False,
        unavailable_reason="missing",
        risk_score=None,
    )
    with pytest.raises(ValueError, match="has an artifact"):
        _build_prediction_with_variants(changed)

    changed = copy.deepcopy(variants)
    changed[VT1_OBSERVED].update(
        available=False,
        prediction_artifact_id=None,
        fit_artifact_id=None,
        covariance_artifact_id=None,
        unavailable_reason="missing",
    )
    changed[VT1_OBSERVED]["risk_score"] = 0.1
    with pytest.raises(ValueError, match="has a risk score"):
        _build_prediction_with_variants(changed)

    changed = copy.deepcopy(variants)
    changed[VT1_OBSERVED]["fit_artifact_id"] = _digest("different-fit")
    with pytest.raises(ValueError, match="share one raw mean and fit"):
        _build_prediction_with_variants(changed)


def _build_prediction_with_variants(variants: Mapping[str, Any]) -> dict[str, Any]:
    return v61.build_deform360_v6_raw_nested_prediction(
        cohort=_cohort(),
        upstream_prediction_batch_id=UPSTREAM_BATCH_ID,
        upstream_revision=UPSTREAM_REVISION,
        candidate_revision=CANDIDATE_REVISION,
        outer_held_out_object_id="object-0",
        object_id="object-1",
        variants=variants,
        source_artifacts={"source.json": _digest("source")},
    )


def test_raw_prediction_separates_upstream_and_candidate_revisions() -> None:
    record = _prediction("object-0", "object-0")

    assert record["upstream_revision"] == UPSTREAM_REVISION
    assert record["candidate_revision"] == CANDIDATE_REVISION
    assert record["upstream_revision"] != record["candidate_revision"]
    assert (
        record["source_selection_artifact_sha256"] == v61.SOURCE_SELECTION_ARTIFACT_ID
    )
    assert record["information_boundary"]["prob4d_used"] is True
    assert record["information_boundary"]["new_prob4d_inference_run"] is False
    assert len(record["variants"][D1_NATIVE]["fit_object_ids"]) == 9

    inner = _prediction("object-0", "object-1")
    assert len(inner["variants"][D1_NATIVE]["fit_object_ids"]) == 8

    with pytest.raises(ValueError, match="must differ from the upstream"):
        v61.build_deform360_v6_raw_nested_prediction(
            cohort=_cohort(),
            upstream_prediction_batch_id=UPSTREAM_BATCH_ID,
            upstream_revision=UPSTREAM_REVISION,
            candidate_revision=UPSTREAM_REVISION,
            outer_held_out_object_id="object-0",
            object_id="object-0",
            variants={
                variant_id: _variant(
                    variant_id,
                    outer_id="object-0",
                    object_id="object-0",
                    unavailable=set(),
                )
                for variant_id in VARIANT_IDS
            },
            source_artifacts={"source.json": _digest("source")},
        )


@pytest.mark.parametrize("field", ["accepted", "guard_threshold", "interval_covered"])
def test_precomputed_decision_or_interval_fields_are_rejected(field: str) -> None:
    variants = {
        variant_id: _variant(
            variant_id,
            outer_id="object-0",
            object_id="object-0",
            unavailable=set(),
        )
        for variant_id in VARIANT_IDS
    }
    variants[D1_NATIVE][field] = False

    with pytest.raises(ValueError, match="fields changed"):
        v61.build_deform360_v6_raw_nested_prediction(
            cohort=_cohort(),
            upstream_prediction_batch_id=UPSTREAM_BATCH_ID,
            upstream_revision=UPSTREAM_REVISION,
            candidate_revision=CANDIDATE_REVISION,
            outer_held_out_object_id="object-0",
            object_id="object-0",
            variants=variants,
            source_artifacts={"source.json": _digest("source")},
        )


def test_nested_batch_is_complete_order_invariant_and_fail_closed() -> None:
    batch = _batch()

    assert batch["record_count"] == 100
    assert (
        v61.build_deform360_v6_raw_nested_batch(
            list(reversed(batch["records"])), cohort=_cohort()
        )
        == batch
    )
    with pytest.raises(ValueError, match="100 records"):
        v61.build_deform360_v6_raw_nested_batch(batch["records"][:-1], cohort=_cohort())
    with pytest.raises(ValueError, match="repeats"):
        v61.build_deform360_v6_raw_nested_batch(
            [*batch["records"][:-1], batch["records"][0]], cohort=_cohort()
        )


def test_atomic_publication_revalidates_the_cohort(tmp_path: Path) -> None:
    batch, _, evidence = _evidence()
    batch_path = tmp_path / "batch.json"
    evidence_path = tmp_path / "evidence.json"

    v61.publish_deform360_v6_raw_nested_batch(
        batch,
        batch_path,
        cohort=_cohort(),
    )
    v61.publish_deform360_v6_nested_evidence(
        evidence,
        evidence_path,
        cohort=_cohort(),
    )
    with pytest.raises(FileExistsError):
        v61.publish_deform360_v6_raw_nested_batch(
            batch,
            batch_path,
            cohort=_cohort(),
        )
    with pytest.raises(FileExistsError):
        v61.publish_deform360_v6_nested_evidence(
            evidence,
            evidence_path,
            cohort=_cohort(),
        )


def test_result_publication_replays_exact_evidence(tmp_path: Path) -> None:
    _, _, evidence = _evidence()
    result = v61.evaluate_deform360_v6_nested_source_gate(
        evidence,
        cohort=_cohort(),
    )
    result_path = tmp_path / "result.json"

    v61.publish_deform360_v6_nested_result(
        result,
        result_path,
        evidence=evidence,
        cohort=_cohort(),
    )
    changed = copy.deepcopy(result)
    changed["source_continuation_authorized"] = False
    changed["result_id"] = v61.content_id(
        {key: value for key, value in changed.items() if key != "result_id"}
    )
    with pytest.raises(ValueError, match="differs from replayed source evidence"):
        v61.publish_deform360_v6_nested_result(
            changed,
            tmp_path / "changed-result.json",
            evidence=evidence,
            cohort=_cohort(),
        )


def test_nested_gate_fits_rank_nine_calibration_and_advances_d1() -> None:
    _, _, evidence = _evidence()

    result = v61.evaluate_deform360_v6_nested_source_gate(evidence, cohort=_cohort())

    assert result["source_gate_passed"] is True
    assert result["selected_variant"] == D1_NATIVE
    assert result["outer_fold_selected_count"] == 10
    assert result["aggregate"]["coverage"] == 0.9
    assert result["aggregate"]["accepted_count"] == 10
    assert result["aggregate"]["accepted_count_by_stratum"] == {
        "sheet": 5,
        "volumetric": 5,
    }
    assert all(result["checks"].values())
    assert all(
        fold["deployed"]["calibration_artifact_id"]
        != result["full_source_fit"]["calibration"]["calibration_artifact_id"]
        for fold in result["folds"]
    )
    assert all(
        fold["deployed"]["guard_artifact_id"] is not None for fold in result["folds"]
    )
    assert all(fold["deployed"]["guard_threshold"] == 0.1 for fold in result["folds"])
    assert all(
        fold["deployed"]["calibration_artifact_id"]
        and fold["deployed"]["interval_width"] > 0.0
        for fold in result["folds"]
    )
    assert result["full_source_fit"]["calibration"]["grouped_residual_rank"] == 10
    assert result["full_source_fit"]["calibration"]["nominal_coverage_target"] == 0.9
    assert result["fresh_target_selection_authorized"] is False
    assert result["claim_authorized"] is False


def test_held_out_extreme_is_not_used_to_fit_its_fold_calibration() -> None:
    _, outcomes, evidence = _evidence()
    changed = copy.deepcopy(outcomes)
    held = next(
        row
        for row in changed
        if next(
            prediction
            for prediction in evidence["records"]
            if prediction["outcome"]["outcome_id"] == row["outcome_id"]
        )["prediction"]["outer_held_out_object_id"]
        == next(
            prediction
            for prediction in evidence["records"]
            if prediction["outcome"]["outcome_id"] == row["outcome_id"]
        )["prediction"]["object_id"]
        == "object-0"
    )
    original_id = held["outcome_id"]
    held["variants"][D1_NATIVE]["maximum_raw_mahalanobis_norm"] = 100.0
    held["outcome_id"] = v61.content_id(
        {key: value for key, value in held.items() if key != "outcome_id"}
    )
    assert held["outcome_id"] != original_id
    # Reassembly is intentionally impossible without rebuilding against the exact
    # batch API, so use the public builder to preserve all bindings.
    batch = _batch()
    rebuilt = []
    for prediction in batch["records"]:
        variants = _outcome_variants(prediction)
        if (
            prediction["outer_held_out_object_id"]
            == prediction["object_id"]
            == "object-0"
        ):
            variants[D1_NATIVE]["maximum_raw_mahalanobis_norm"] = 100.0
        rebuilt.append(
            v61.build_deform360_v6_raw_nested_outcome(
                prediction_batch=batch,
                prediction_record_id=prediction["prediction_record_id"],
                variants=variants,
                scoring_artifacts={
                    "score.json": _digest(prediction["prediction_record_id"])
                },
            )
        )
    changed_evidence = v61.assemble_deform360_v6_nested_evidence(
        prediction_batch=batch,
        outcomes=rebuilt,
        cohort=_cohort(),
    )
    result = v61.evaluate_deform360_v6_nested_source_gate(
        changed_evidence, cohort=_cohort()
    )
    fold = next(
        row for row in result["folds"] if row["outer_held_out_object_id"] == "object-0"
    )

    assert fold["deployed"]["guard_threshold"] == 0.1
    assert fold["deployed"]["interval_covered"] is False


def test_unavailable_covariance_is_explicit_and_does_not_block_d1() -> None:
    _, _, evidence = _evidence(unavailable={VT1_OBSERVED})

    result = v61.evaluate_deform360_v6_nested_source_gate(evidence, cohort=_cohort())

    assert result["source_gate_passed"] is True
    assert result["selected_variant"] == D1_NATIVE


def test_later_covariance_amendment_ranks_proper_score_before_point_loss() -> None:
    batch = _batch()
    outcomes = []
    for prediction in batch["records"]:
        variants = _outcome_variants(prediction)
        variants[VT1_WORKING]["mean_log_determinant"] = -4.0
        outcomes.append(
            v61.build_deform360_v6_raw_nested_outcome(
                prediction_batch=batch,
                prediction_record_id=prediction["prediction_record_id"],
                variants=variants,
                scoring_artifacts={
                    "score.json": _digest(prediction["prediction_record_id"])
                },
            )
        )
    evidence = v61.assemble_deform360_v6_nested_evidence(
        prediction_batch=batch,
        outcomes=outcomes,
        cohort=_cohort(),
    )

    result = v61.evaluate_deform360_v6_nested_source_gate(
        evidence,
        cohort=_cohort(),
    )

    assert result["selected_variant"] == VT1_WORKING
    assert all(fold["selected_variant"] == VT1_WORKING for fold in result["folds"])


def test_vt1_covariance_variants_cannot_change_point_loss() -> None:
    batch = _batch()
    prediction = batch["records"][0]
    variants = _outcome_variants(prediction)
    variants[VT1_OBSERVED]["point_loss"] += 1.0

    with pytest.raises(ValueError, match="share one scored point prediction"):
        v61.build_deform360_v6_raw_nested_outcome(
            prediction_batch=batch,
            prediction_record_id=prediction["prediction_record_id"],
            variants=variants,
            scoring_artifacts={"score.json": _digest("score")},
        )


def test_rejected_candidate_uses_exact_physical_point_and_interval() -> None:
    batch, _, _ = _evidence()
    records = copy.deepcopy(batch["records"])
    # Make object-9 risky only when it is the held-out unit. Training folds still
    # fit the inclusive 0.1 threshold from their own nine records.
    for record in records:
        if record["object_id"] == "object-9":
            for variant_id in CHALLENGER_VARIANTS:
                record["variants"][variant_id]["risk_score"] = 9.0
            record["prediction_record_id"] = v61.content_id(
                {
                    key: value
                    for key, value in record.items()
                    if key != "prediction_record_id"
                }
            )
    risky_batch = v61.build_deform360_v6_raw_nested_batch(records, cohort=_cohort())
    outcomes = [
        v61.build_deform360_v6_raw_nested_outcome(
            prediction_batch=risky_batch,
            prediction_record_id=prediction["prediction_record_id"],
            variants=_outcome_variants(prediction),
            scoring_artifacts={
                "score.json": _digest(prediction["prediction_record_id"])
            },
        )
        for prediction in risky_batch["records"]
    ]
    evidence = v61.assemble_deform360_v6_nested_evidence(
        prediction_batch=risky_batch,
        outcomes=outcomes,
        cohort=_cohort(),
    )
    result = v61.evaluate_deform360_v6_nested_source_gate(evidence, cohort=_cohort())
    fold = next(
        row for row in result["folds"] if row["outer_held_out_object_id"] == "object-9"
    )

    assert fold["accepted"] is False
    assert fold["exact_fallback"] is True
    assert (
        fold["deployed"]["prediction_artifact_id"]
        == fold["fallback"]["prediction_artifact_id"]
    )
    assert fold["deployed"]["proper_score"] == fold["fallback"]["proper_score"]
    assert fold["deployed"]["interval_width"] == fold["fallback"]["interval_width"]


def test_tampering_with_evidence_identity_fails_closed() -> None:
    _, _, evidence = _evidence()
    changed = copy.deepcopy(evidence)
    changed["records"][0]["outcome"]["prediction_record_id"] = "0" * 64
    changed["evidence_id"] = v61.content_id(
        {key: value for key, value in changed.items() if key != "evidence_id"}
    )

    with pytest.raises(ValueError, match="does not bind one prediction record"):
        v61.validate_deform360_v6_nested_evidence(changed, cohort=_cohort())
