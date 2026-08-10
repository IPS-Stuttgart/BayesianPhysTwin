from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import bayesian_phystwin.deform360_fresh_object_session_source_v6 as v6

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    ROOT / "protocols/locks/deform360_official_hub_fresh_object_session_v6.json"
)
AMENDMENT_PATH = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_source_covariance.json"
)
SELECTION_PATH = ROOT / (
    "protocols/locks/deform360_official_hub_visuotactile_v1_selection.json"
)
REVISION = "a" * 40


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _policy() -> dict[str, Any]:
    return {
        "policy_id": v6.POLICY_ID,
        "predecessor_boundary": {
            "v5_selection_artifact_sha256": "b" * 64,
        },
        "source_tournament": {
            "source_unit_count": 10,
            "final_winner_must_match_at_least_outer_folds": 8,
            "minimum_nonregressing_held_out_units": 8,
            "minimum_nonregressing_held_out_units_per_stratum": 4,
        },
        "guard_calibration": {
            "minimum_accepted_source_units": 8,
            "minimum_accepted_source_units_per_stratum": 4,
        },
        "covariance_calibration": {
            "minimum_source_object_balanced_coverage": 0.8,
            "maximum_source_object_balanced_coverage": 0.98,
            "maximum_mean_full_interval_width_ratio_vs_reference": 1.25,
        },
    }


def _selection() -> dict[str, Any]:
    rows = [
        {
            "object_id": f"object-{index}",
            "episode_id": index,
            "stratum": "sheet" if index < 5 else "volumetric",
        }
        for index in range(10)
    ]
    return {
        "selection_artifact_sha256": "b" * 64,
        "selection": {"calibration": rows},
    }


def _amendment() -> dict[str, Any]:
    return {"amendment_id": v6.AMENDMENT_ID}


def _source_ids(selection: Mapping[str, Any]) -> list[str]:
    return sorted(row["object_id"] for row in selection["selection"]["calibration"])


def _variant_prediction(
    variant_id: str,
    *,
    object_id: str,
    source_ids: list[str],
    accepted: bool,
    available: bool = True,
) -> dict[str, Any]:
    challenger = variant_id in v6.CHALLENGER_VARIANTS
    if not available:
        return {
            "available": False,
            "accepted": False,
            "prediction_artifact_id": None,
            "fit_artifact_id": None,
            "fit_object_ids": sorted(set(source_ids) - {object_id}),
            "guard_artifact_id": None,
            "guard_threshold": None,
            "covariance_artifact_id": None,
            "interval_artifact_id": None,
            "risk_score": None,
            "unavailable_reason": "technical-failure-retained",
        }
    return {
        "available": True,
        "accepted": (
            False
            if variant_id == v6.B0
            else True if variant_id == v6.B1 else accepted
        ),
        "prediction_artifact_id": _digest(f"{object_id}/{variant_id}/prediction"),
        "fit_artifact_id": _digest(f"{object_id}/{variant_id}/fit"),
        "fit_object_ids": (
            sorted(set(source_ids) - {object_id}) if challenger else []
        ),
        "guard_artifact_id": _digest(f"{object_id}/{variant_id}/guard"),
        "guard_threshold": 0.2 if challenger else None,
        "covariance_artifact_id": _digest(f"{object_id}/{variant_id}/covariance"),
        "interval_artifact_id": _digest(f"{object_id}/{variant_id}/interval"),
        "risk_score": (0.1 if accepted else 0.3) if challenger else None,
        "unavailable_reason": None,
    }


def _prediction_variants(
    object_id: str,
    source_ids: list[str],
    rejected: set[str],
    *,
    unavailable_variant: str | None = None,
) -> dict[str, dict[str, Any]]:
    result = {}
    for variant_id in v6.VARIANT_IDS:
        result[variant_id] = _variant_prediction(
            variant_id,
            object_id=object_id,
            source_ids=source_ids,
            accepted=object_id not in rejected,
            available=variant_id != unavailable_variant,
        )
    return result


def _score_values(
    variant_id: str,
    *,
    object_id: str,
    positive: bool,
) -> tuple[float, float, bool, float]:
    if variant_id == v6.B0:
        return 10.0, 5.0, True, 2.0
    if variant_id == v6.B1:
        return 9.0, 4.0, True, 2.0
    if not positive:
        return 9.5, 4.5, True, 2.1
    if variant_id == v6.D1_NATIVE:
        return 7.0, 3.0, object_id != "object-0", 1.8
    return 8.0, 3.5, True, 1.9


def _outcome_variants(
    seal: Mapping[str, Any],
    *,
    positive: bool,
) -> dict[str, dict[str, Any]]:
    result = {}
    fallback = _score_values(v6.B0, object_id=seal["object_id"], positive=positive)
    for variant_id in v6.VARIANT_IDS:
        prediction = seal["variants"][variant_id]
        values = _score_values(
            variant_id,
            object_id=seal["object_id"],
            positive=positive,
        )
        if not prediction["available"]:
            values = fallback
        result[variant_id] = {
            "available": prediction["available"],
            "prediction_artifact_id": prediction["prediction_artifact_id"],
            "point_loss": values[0],
            "proper_score": values[1],
            "interval_covered": values[2],
            "interval_width": values[3],
        }
    return result


def _bundle(
    *,
    positive: bool = True,
    unavailable_variant: str | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
]:
    policy = _policy()
    amendment = _amendment()
    selection = _selection()
    source_ids = _source_ids(selection)
    rejected = {"object-4", "object-9"}
    seals = [
        v6.build_deform360_v6_source_prediction_seal(
            policy=policy,
            amendment=amendment,
            selection=selection,
            implementation_revision=REVISION,
            object_id=object_id,
            variants=_prediction_variants(
                object_id,
                source_ids,
                rejected,
                unavailable_variant=unavailable_variant,
            ),
            source_artifacts={
                f"predictions/{object_id}.json": _digest(f"source/{object_id}")
            },
        )
        for object_id in source_ids
    ]
    batch = v6.build_deform360_v6_source_prediction_batch(
        seals,
        policy=policy,
        amendment=amendment,
        selection=selection,
    )
    outcomes = [
        v6.build_deform360_v6_source_outcome(
            prediction_batch=batch,
            prediction_seal_id=seal["seal_id"],
            variants=_outcome_variants(seal, positive=positive),
            scoring_artifacts={
                f"scores/{seal['object_id']}.json": _digest(
                    f"score/{seal['object_id']}"
                )
            },
        )
        for seal in batch["records"]
    ]
    evidence = v6.assemble_deform360_v6_source_evidence(
        prediction_batch=batch,
        outcomes=outcomes,
    )
    return policy, amendment, selection, seals, batch, outcomes, evidence


def _reidentify(payload: dict[str, Any], id_field: str) -> None:
    identity = {key: value for key, value in payload.items() if key != id_field}
    payload[id_field] = v6.content_id(identity)


def test_positive_source_gate_is_prediction_first_and_target_closed() -> None:
    policy, amendment, selection, seals, batch, outcomes, evidence = _bundle()

    assert v6.build_deform360_v6_source_prediction_batch(
        list(reversed(seals)),
        policy=policy,
        amendment=amendment,
        selection=selection,
    ) == batch
    assert v6.assemble_deform360_v6_source_evidence(
        prediction_batch=batch,
        outcomes=list(reversed(outcomes)),
    ) == evidence
    assert v6.validate_deform360_v6_source_prediction_batch(
        batch,
        policy=policy,
        amendment=amendment,
        selection=selection,
    ) == batch
    assert v6.validate_deform360_v6_source_evidence(evidence) == evidence

    result = v6.evaluate_deform360_v6_source_gate(evidence, policy)

    assert result["source_gate_passed"] is True
    assert result["selected_variant"] == v6.D1_NATIVE
    assert result["selected_candidate_family"] == (
        "D1_dynamic_endpoint_model_average_v2"
    )
    assert result["selected_covariance"] == "native-model-average"
    assert result["accepted_count"] == 8
    assert result["accepted_count_by_stratum"] == {"sheet": 4, "volumetric": 4}
    assert result["nonregressing_count"] == 8
    assert all(result["checks"].values())
    assert result["source_continuation_authorized"] is True
    assert result["fresh_target_selection_authorized"] is False
    assert result["fresh_target_payload_access_authorized"] is False
    assert result["claim_authorized"] is False
    assert v6.validate_deform360_v6_source_result(result) == result


def test_source_negative_retains_reference_without_target_access() -> None:
    policy, _, _, _, _, _, evidence = _bundle(positive=False)

    result = v6.evaluate_deform360_v6_source_gate(evidence, policy)

    assert result["source_gate_passed"] is False
    assert result["selected_variant"] == v6.B1
    assert result["status"] == "source-reference-retained"
    assert result["next_stage"] == "terminate-v6-before-fresh-cohort-selection"
    assert result["source_continuation_authorized"] is False
    assert result["fresh_target_selection_authorized"] is False
    assert result["fresh_target_payload_access_authorized"] is False


def test_unavailable_covariance_variant_is_retained_as_exact_fallback() -> None:
    policy, _, _, _, _, _, evidence = _bundle(unavailable_variant=v6.VT1_OBSERVED)
    tournament = evidence["tournament_input"]
    rows = [
        row
        for row in tournament["records"]
        if row["candidate_id"] == v6.VT1_OBSERVED
    ]

    assert len(rows) == 10
    assert all(row["accepted"] is False for row in rows)
    assert all(row["point_loss"] == row["fallback_point_loss"] for row in rows)
    assert all(
        row["deployed_proper_score"] == row["fallback_proper_score"] for row in rows
    )
    result = v6.evaluate_deform360_v6_source_gate(evidence, policy)
    assert result["selected_variant"] == v6.D1_NATIVE


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda variants, ids: variants[v6.B0].__setitem__("accepted", True),
            "physical fallback",
        ),
        (
            lambda variants, ids: variants[v6.B1].__setitem__("accepted", False),
            "last residual",
        ),
        (
            lambda variants, ids: variants[v6.D1_NATIVE].__setitem__(
                "fit_object_ids", ids[:-2]
            ),
            "fit roster",
        ),
        (
            lambda variants, ids: variants[v6.D1_NATIVE].__setitem__(
                "guard_threshold", 0.05
            ),
            "guard decision",
        ),
        (
            lambda variants, ids: variants[v6.D1_NATIVE].__setitem__(
                "unavailable_reason", "unexpected"
            ),
            "available source variant",
        ),
        (
            lambda variants, ids: variants[v6.B0].__setitem__("risk_score", 0.1),
            "baseline",
        ),
    ],
)
def test_prediction_contracts_fail_closed(mutate, message: str) -> None:
    policy = _policy()
    amendment = _amendment()
    selection = _selection()
    source_ids = _source_ids(selection)
    variants = _prediction_variants(source_ids[0], source_ids, {"object-4", "object-9"})
    mutate(variants, source_ids)

    with pytest.raises(ValueError, match=message):
        v6.build_deform360_v6_source_prediction_seal(
            policy=policy,
            amendment=amendment,
            selection=selection,
            implementation_revision=REVISION,
            object_id=source_ids[0],
            variants=variants,
            source_artifacts={"source.json": _digest("source")},
        )


def test_unavailable_prediction_contract_rejects_artifacts_and_acceptance() -> None:
    policy = _policy()
    amendment = _amendment()
    selection = _selection()
    source_ids = _source_ids(selection)
    variants = _prediction_variants(
        source_ids[0],
        source_ids,
        {"object-4", "object-9"},
        unavailable_variant=v6.VT1_OBSERVED,
    )
    variants[v6.VT1_OBSERVED]["prediction_artifact_id"] = _digest("forbidden")
    with pytest.raises(ValueError, match="unavailable source variant has an artifact"):
        v6.build_deform360_v6_source_prediction_seal(
            policy=policy,
            amendment=amendment,
            selection=selection,
            implementation_revision=REVISION,
            object_id=source_ids[0],
            variants=variants,
            source_artifacts={"source.json": _digest("source")},
        )

    variants[v6.VT1_OBSERVED]["prediction_artifact_id"] = None
    variants[v6.VT1_OBSERVED]["accepted"] = True
    with pytest.raises(ValueError, match="unavailable.*accepted"):
        v6.build_deform360_v6_source_prediction_seal(
            policy=policy,
            amendment=amendment,
            selection=selection,
            implementation_revision=REVISION,
            object_id=source_ids[0],
            variants=variants,
            source_artifacts={"source.json": _digest("source")},
        )


def test_prediction_batch_and_seal_tampering_fail_closed() -> None:
    policy, amendment, selection, seals, batch, _, _ = _bundle()

    with pytest.raises(ValueError, match="ten seals"):
        v6.build_deform360_v6_source_prediction_batch(
            seals[:-1],
            policy=policy,
            amendment=amendment,
            selection=selection,
        )
    with pytest.raises(ValueError, match="repeats a unit"):
        v6.build_deform360_v6_source_prediction_batch(
            [*seals[:-1], seals[0]],
            policy=policy,
            amendment=amendment,
            selection=selection,
        )

    changed = copy.deepcopy(seals[0])
    changed["information_boundary"]["source_suffix_opened"] = True
    _reidentify(changed, "seal_id")
    with pytest.raises(ValueError, match="crossed its boundary"):
        v6.validate_deform360_v6_source_prediction_seal(
            changed,
            policy=policy,
            amendment=amendment,
            selection=selection,
        )

    changed_batch = copy.deepcopy(batch)
    changed_batch["record_count"] = 9
    _reidentify(changed_batch, "prediction_batch_id")
    with pytest.raises(ValueError, match="content changed"):
        v6.validate_deform360_v6_source_prediction_batch(
            changed_batch,
            policy=policy,
            amendment=amendment,
            selection=selection,
        )


def test_outcome_binding_and_exact_unavailable_fallback_fail_closed() -> None:
    _, _, _, _, batch, outcomes, _ = _bundle(unavailable_variant=v6.VT1_OBSERVED)
    with pytest.raises(ValueError, match="does not bind one prediction seal"):
        v6.build_deform360_v6_source_outcome(
            prediction_batch=batch,
            prediction_seal_id="0" * 64,
            variants=outcomes[0]["variants"],
            scoring_artifacts={"score.json": _digest("score")},
        )

    variants = copy.deepcopy(outcomes[0]["variants"])
    variants[v6.D1_NATIVE]["prediction_artifact_id"] = "0" * 64
    with pytest.raises(ValueError, match="artifact differs"):
        v6.build_deform360_v6_source_outcome(
            prediction_batch=batch,
            prediction_seal_id=outcomes[0]["prediction_seal_id"],
            variants=variants,
            scoring_artifacts={"score.json": _digest("score")},
        )

    variants = copy.deepcopy(outcomes[0]["variants"])
    variants[v6.VT1_OBSERVED]["point_loss"] = 1.0
    with pytest.raises(ValueError, match="exact fallback"):
        v6.build_deform360_v6_source_outcome(
            prediction_batch=batch,
            prediction_seal_id=outcomes[0]["prediction_seal_id"],
            variants=variants,
            scoring_artifacts={"score.json": _digest("score")},
        )

    changed = copy.deepcopy(outcomes[0])
    changed["information_boundary"][
        "source_suffix_opened_after_prediction_batch"
    ] = False
    _reidentify(changed, "outcome_id")
    with pytest.raises(ValueError, match="crossed its information boundary"):
        v6.validate_deform360_v6_source_outcome(changed, prediction_batch=batch)


def test_evidence_roster_and_tournament_identity_fail_closed() -> None:
    _, _, _, _, batch, outcomes, evidence = _bundle()

    with pytest.raises(ValueError, match="ten outcomes"):
        v6.assemble_deform360_v6_source_evidence(
            prediction_batch=batch,
            outcomes=outcomes[:-1],
        )
    with pytest.raises(ValueError, match="repeats an outcome"):
        v6.assemble_deform360_v6_source_evidence(
            prediction_batch=batch,
            outcomes=[*outcomes[:-1], outcomes[0]],
        )

    changed = copy.deepcopy(evidence)
    changed["tournament_input"]["selection"][
        "minimum_relative_point_improvement"
    ] = 0.0
    _reidentify(changed, "evidence_id")
    with pytest.raises(ValueError, match="differs from sealed evidence"):
        v6.validate_deform360_v6_source_evidence(changed)

    changed = copy.deepcopy(evidence)
    changed["records"][1]["object_id"] = changed["records"][0]["object_id"]
    changed["tournament_input"] = changed["tournament_input"]
    _reidentify(changed, "evidence_id")
    with pytest.raises(ValueError, match="repeats a source unit"):
        v6.validate_deform360_v6_source_evidence(changed)


def test_adapter_enforces_eight_fold_stability(monkeypatch) -> None:
    policy, _, _, _, _, _, evidence = _bundle()
    original = v6.analyze_discrepancy_candidate_tournament

    def unstable(payload):
        report = original(payload)
        for fold in report["cross_fitted"]["folds"][:3]:
            fold["selected_candidate"] = v6.VT1_WORKING
        return report

    monkeypatch.setattr(v6, "analyze_discrepancy_candidate_tournament", unstable)
    result = v6.evaluate_deform360_v6_source_gate(evidence, policy)

    assert result["checks"]["stable_variant_selection"] is False
    assert result["source_gate_passed"] is False
    assert result["selected_variant"] == v6.B1


def test_result_validation_rejects_target_or_claim_authorization() -> None:
    policy, _, _, _, _, _, evidence = _bundle()
    result = v6.evaluate_deform360_v6_source_gate(evidence, policy)

    for key in (
        "fresh_target_selection_authorized",
        "fresh_target_payload_access_authorized",
        "claim_authorized",
    ):
        changed = copy.deepcopy(result)
        changed[key] = True
        _reidentify(changed, "result_id")
        with pytest.raises(ValueError, match="cannot authorize"):
            v6.validate_deform360_v6_source_result(changed)

    changed = copy.deepcopy(result)
    changed["result_id"] = "0" * 64
    with pytest.raises(ValueError, match="content identity"):
        v6.validate_deform360_v6_source_result(changed)


def test_atomic_publication_is_non_replacing(tmp_path: Path) -> None:
    policy, _, _, _, batch, _, evidence = _bundle()
    result = v6.evaluate_deform360_v6_source_gate(evidence, policy)
    batch_path = tmp_path / "batch.json"
    evidence_path = tmp_path / "evidence.json"
    result_path = tmp_path / "result.json"

    v6.publish_deform360_v6_source_prediction_batch(batch, batch_path)
    v6.publish_deform360_v6_source_evidence(evidence, evidence_path)
    v6.publish_deform360_v6_source_result(result, result_path)
    with pytest.raises(FileExistsError):
        v6.publish_deform360_v6_source_prediction_batch(batch, batch_path)
    with pytest.raises(FileExistsError):
        v6.publish_deform360_v6_source_evidence(evidence, evidence_path)
    with pytest.raises(FileExistsError):
        v6.publish_deform360_v6_source_result(result, result_path)


def test_selection_validation_rejects_identity_roster_and_strata() -> None:
    policy = _policy()
    selection = _selection()

    changed = copy.deepcopy(selection)
    changed["selection_artifact_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="selection identity"):
        v6.validate_deform360_v6_source_selection(changed, policy)

    changed = copy.deepcopy(selection)
    changed["selection"]["calibration"][1]["object_id"] = "object-0"
    with pytest.raises(ValueError, match="repeats an object"):
        v6.validate_deform360_v6_source_selection(changed, policy)

    changed = copy.deepcopy(selection)
    changed["selection"]["calibration"].pop()
    with pytest.raises(ValueError, match="ten units"):
        v6.validate_deform360_v6_source_selection(changed, policy)

    changed = copy.deepcopy(selection)
    changed["selection"]["calibration"][5]["stratum"] = "sheet"
    with pytest.raises(ValueError, match="stratum counts"):
        v6.validate_deform360_v6_source_selection(changed, policy)

    changed = copy.deepcopy(selection)
    changed["selection"]["calibration"][0]["stratum"] = "filament"
    with pytest.raises(ValueError, match="stratum changed"):
        v6.validate_deform360_v6_source_selection(changed, policy)


def test_real_locks_load_and_bind_candidate_specific_covariances() -> None:
    if not POLICY_PATH.exists():
        pytest.skip("repository lock files are unavailable in the isolated harness")
    policy = v6.load_deform360_fresh_object_session_v6_policy(POLICY_PATH)
    amendment = v6.load_deform360_fresh_object_session_v6_covariance_amendment(
        AMENDMENT_PATH, policy
    )
    selection, cohort = v6.load_deform360_v6_source_selection(SELECTION_PATH, policy)

    assert policy["policy_id"] == v6.POLICY_ID
    assert amendment["amendment_id"] == v6.AMENDMENT_ID
    assert selection["selection_artifact_sha256"] == (
        policy["predecessor_boundary"]["v5_selection_artifact_sha256"]
    )
    assert len(cohort) == 10
    assert v6.VARIANT_COVARIANCE[v6.D1_NATIVE] == "native-model-average"
    assert {
        v6.VARIANT_COVARIANCE[v6.VT1_WORKING],
        v6.VARIANT_COVARIANCE[v6.VT1_OBSERVED],
        v6.VARIANT_COVARIANCE[v6.VT1_SANDWICH],
    } == {"working-irls", "observed-information", "group-sandwich"}


def test_low_level_type_identifier_and_numeric_validation() -> None:
    policy = _policy()
    amendment = _amendment()
    selection = _selection()
    source_ids = _source_ids(selection)
    variants = _prediction_variants(source_ids[0], source_ids, {"object-4", "object-9"})

    with pytest.raises(ValueError, match="JSON object"):
        v6.build_deform360_v6_source_prediction_seal(
            policy=policy,
            amendment=amendment,
            selection=selection,
            implementation_revision=REVISION,
            object_id=source_ids[0],
            variants=None,  # type: ignore[arg-type]
            source_artifacts={"source.json": _digest("source")},
        )

    changed_selection = copy.deepcopy(selection)
    changed_selection["selection"]["calibration"] = "not-an-array"
    with pytest.raises(ValueError, match="JSON array"):
        v6.validate_deform360_v6_source_selection(changed_selection, policy)

    with pytest.raises(ValueError, match="canonical string"):
        v6.build_deform360_v6_source_prediction_seal(
            policy=policy,
            amendment=amendment,
            selection=selection,
            implementation_revision=REVISION,
            object_id=f" {source_ids[0]}",
            variants=variants,
            source_artifacts={"source.json": _digest("source")},
        )

    for invalid in (True, float("nan"), -1.0):
        changed = copy.deepcopy(variants)
        changed[v6.D1_NATIVE]["risk_score"] = invalid
        with pytest.raises(ValueError, match="finite real|at least"):
            v6.build_deform360_v6_source_prediction_seal(
                policy=policy,
                amendment=amendment,
                selection=selection,
                implementation_revision=REVISION,
                object_id=source_ids[0],
                variants=changed,
                source_artifacts={"source.json": _digest("source")},
            )


def test_seal_rejects_wrong_amendment_unknown_object_and_schema() -> None:
    policy, amendment, selection, seals, _, _, _ = _bundle()
    source_ids = _source_ids(selection)
    variants = _prediction_variants(source_ids[0], source_ids, {"object-4", "object-9"})

    with pytest.raises(ValueError, match="another covariance amendment"):
        v6.build_deform360_v6_source_prediction_seal(
            policy=policy,
            amendment={"amendment_id": "0" * 64},
            selection=selection,
            implementation_revision=REVISION,
            object_id=source_ids[0],
            variants=variants,
            source_artifacts={"source.json": _digest("source")},
        )
    with pytest.raises(ValueError, match="unregistered object"):
        v6.build_deform360_v6_source_prediction_seal(
            policy=policy,
            amendment=amendment,
            selection=selection,
            implementation_revision=REVISION,
            object_id="unregistered",
            variants=variants,
            source_artifacts={"source.json": _digest("source")},
        )

    changed = copy.deepcopy(seals[0])
    changed["schema"] = "changed"
    _reidentify(changed, "seal_id")
    with pytest.raises(ValueError, match="schema changed"):
        v6.validate_deform360_v6_source_prediction_seal(
            changed,
            policy=policy,
            amendment=amendment,
            selection=selection,
        )

    changed = copy.deepcopy(seals[0])
    changed["seal_id"] = "0" * 64
    with pytest.raises(ValueError, match="content changed"):
        v6.validate_deform360_v6_source_prediction_seal(
            changed,
            policy=policy,
            amendment=amendment,
            selection=selection,
        )


def test_batch_rejects_incomplete_roster_mixed_revision_and_schema() -> None:
    policy, amendment, selection, seals, batch, _, _ = _bundle()

    changed = copy.deepcopy(seals)
    changed[-1]["object_id"] = "unknown"
    _reidentify(changed[-1], "seal_id")
    with pytest.raises(ValueError):
        v6.build_deform360_v6_source_prediction_batch(
            changed,
            policy=policy,
            amendment=amendment,
            selection=selection,
        )

    changed = copy.deepcopy(seals)
    changed[-1]["implementation_revision"] = "c" * 40
    _reidentify(changed[-1], "seal_id")
    with pytest.raises(ValueError, match="mixes implementation revisions"):
        v6.build_deform360_v6_source_prediction_batch(
            changed,
            policy=policy,
            amendment=amendment,
            selection=selection,
        )

    changed_batch = copy.deepcopy(batch)
    changed_batch["schema"] = "changed"
    _reidentify(changed_batch, "prediction_batch_id")
    with pytest.raises(ValueError, match="schema changed"):
        v6.validate_deform360_v6_source_prediction_batch(
            changed_batch,
            policy=policy,
            amendment=amendment,
            selection=selection,
        )


def test_outcome_rejects_schema_availability_and_content_tampering() -> None:
    _, _, _, _, batch, outcomes, _ = _bundle()

    variants = copy.deepcopy(outcomes[0]["variants"])
    variants[v6.D1_NATIVE]["available"] = False
    with pytest.raises(ValueError, match="availability differs"):
        v6.build_deform360_v6_source_outcome(
            prediction_batch=batch,
            prediction_seal_id=outcomes[0]["prediction_seal_id"],
            variants=variants,
            scoring_artifacts={"score.json": _digest("score")},
        )

    changed = copy.deepcopy(outcomes[0])
    changed["schema"] = "changed"
    _reidentify(changed, "outcome_id")
    with pytest.raises(ValueError, match="schema changed"):
        v6.validate_deform360_v6_source_outcome(changed, prediction_batch=batch)

    changed = copy.deepcopy(outcomes[0])
    changed["outcome_id"] = "0" * 64
    with pytest.raises(ValueError, match="content changed"):
        v6.validate_deform360_v6_source_outcome(changed, prediction_batch=batch)


def test_evidence_rejects_schema_policy_boundary_count_and_strata() -> None:
    _, _, _, _, _, _, evidence = _bundle()

    mutations = [
        ("schema", "changed", "schema changed"),
        ("policy_id", "0" * 64, "another policy"),
        ("record_count", 9, "record count"),
    ]
    for key, value, message in mutations:
        changed = copy.deepcopy(evidence)
        changed[key] = value
        _reidentify(changed, "evidence_id")
        with pytest.raises(ValueError, match=message):
            v6.validate_deform360_v6_source_evidence(changed)

    changed = copy.deepcopy(evidence)
    changed["information_boundary"]["v6_target_payloads_used"] = True
    _reidentify(changed, "evidence_id")
    with pytest.raises(ValueError, match="crossed its boundary"):
        v6.validate_deform360_v6_source_evidence(changed)

    changed = copy.deepcopy(evidence)
    changed["records"].pop()
    changed["record_count"] = 9
    _reidentify(changed, "evidence_id")
    with pytest.raises(ValueError, match="record count"):
        v6.validate_deform360_v6_source_evidence(changed)

    changed = copy.deepcopy(evidence)
    changed["records"][0]["stratum"] = "filament"
    _reidentify(changed, "evidence_id")
    with pytest.raises(ValueError, match="stratum changed"):
        v6.validate_deform360_v6_source_evidence(changed)


def test_real_lock_loaders_fail_closed_on_tampering(tmp_path: Path) -> None:
    if not POLICY_PATH.exists():
        pytest.skip("repository lock files are unavailable in the isolated harness")

    policy_payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    amendment_payload = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))

    def write(name: str, value: Mapping[str, Any]) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    changed = copy.deepcopy(policy_payload)
    changed["schema"] = "changed"
    _reidentify(changed, "policy_id")
    with pytest.raises(ValueError, match="schema changed"):
        v6.load_deform360_fresh_object_session_v6_policy(
            write("policy-schema.json", changed)
        )

    changed = copy.deepcopy(policy_payload)
    changed["source_tournament"]["source_unit_count"] = 9
    _reidentify(changed, "policy_id")
    with pytest.raises(ValueError, match="policy identity changed|source unit count"):
        v6.load_deform360_fresh_object_session_v6_policy(
            write("policy-count.json", changed)
        )

    changed = copy.deepcopy(policy_payload)
    changed["policy_id"] = "0" * 64
    with pytest.raises(ValueError, match="content identity"):
        v6.load_deform360_fresh_object_session_v6_policy(
            write("policy-id.json", changed)
        )

    policy = v6.load_deform360_fresh_object_session_v6_policy(POLICY_PATH)
    changed = copy.deepcopy(amendment_payload)
    changed["schema"] = "changed"
    _reidentify(changed, "amendment_id")
    with pytest.raises(ValueError, match="schema changed"):
        v6.load_deform360_fresh_object_session_v6_covariance_amendment(
            write("amendment-schema.json", changed), policy
        )

    changed = copy.deepcopy(amendment_payload)
    changed["base_policy"]["policy_id"] = "0" * 64
    _reidentify(changed, "amendment_id")
    with pytest.raises(ValueError, match="identity changed|binds another policy"):
        v6.load_deform360_fresh_object_session_v6_covariance_amendment(
            write("amendment-policy.json", changed), policy
        )

    changed = copy.deepcopy(amendment_payload)
    changed["information_boundary"]["source_outcomes_used"] = True
    _reidentify(changed, "amendment_id")
    with pytest.raises(ValueError, match="used source outcomes"):
        v6.load_deform360_fresh_object_session_v6_covariance_amendment(
            write("amendment-boundary.json", changed), policy
        )

    changed = copy.deepcopy(amendment_payload)
    changed["candidate_covariance_rosters"][
        "D1_dynamic_endpoint_model_average_v2"
    ]["required_methods"] = ["working-irls"]
    _reidentify(changed, "amendment_id")
    with pytest.raises(ValueError, match="roster changed"):
        v6.load_deform360_fresh_object_session_v6_covariance_amendment(
            write("amendment-roster.json", changed), policy
        )
