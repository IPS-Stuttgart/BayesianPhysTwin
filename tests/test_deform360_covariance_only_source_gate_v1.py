from __future__ import annotations

import copy
import hashlib

import pytest

from bayesian_phystwin.deform360_covariance_only_source_gate_v1 import (
    BATCH_SCHEMA,
    COVARIANCE_DONOR_ID,
    COVARIANCE_SCALES,
    OBSERVATION_STD_M,
    PAPER_PROTOCOL_ID,
    REFERENCE_PREDICTOR_ID,
    SCHEMA_VERSION,
    SCORES_SCHEMA,
    SOFTWARE_PROTOCOL_ID,
    SOURCE_ROSTER,
    evaluate_source_gate,
    seal_prediction_batch,
    seal_source_scores,
    validate_prediction_batch,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _batch() -> dict[str, object]:
    units = [
        {"object_id": object_id, "episode": episode, "stratum": stratum}
        for object_id, episode, stratum in SOURCE_ROSTER
    ]
    records: list[dict[str, object]] = []
    selected: dict[str, str] = {}
    for unit_index, (object_id, episode, stratum) in enumerate(SOURCE_ROSTER):
        for fold_index in range(10):
            prediction_id = _sha(
                f"prediction:{object_id}:{episode}:{fold_index}"
            )
            mean_id = _sha(f"mean:{object_id}:{episode}:{fold_index}")
            records.append(
                {
                    "prediction_id": prediction_id,
                    "object_id": object_id,
                    "episode": episode,
                    "stratum": stratum,
                    "outer_fold_index": fold_index,
                    "mean_sha256": mean_id,
                    "reference_mean_sha256": mean_id,
                    "disposition": "candidate",
                    "exact_fallback": False,
                    "source_suffix_used": False,
                    "confirmation_outcomes_used": False,
                }
            )
            if fold_index == unit_index:
                selected[f"{object_id}#{episode}"] = prediction_id
    return seal_prediction_batch(
        {
            "schema": BATCH_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "software_protocol_id": SOFTWARE_PROTOCOL_ID,
            "paper_protocol_id": PAPER_PROTOCOL_ID,
            "candidate": {
                "reference_predictor_id": REFERENCE_PREDICTOR_ID,
                "covariance_donor_id": COVARIANCE_DONOR_ID,
                "early_middle_late_covariance_scales": list(
                    COVARIANCE_SCALES
                ),
                "observation_std_m": OBSERVATION_STD_M,
                "point_prediction_change_allowed": False,
            },
            "information_boundary": {
                "sealed_before_source_suffix": True,
                "source_suffix_used": False,
                "confirmation_payloads_opened": False,
                "confirmation_predictions_run": False,
                "confirmation_outcomes_used": False,
                "replacement_used": False,
                "target_informed_selection_used": False,
            },
            "source_units": units,
            "records": records,
            "scoring_prediction_by_source_unit": selected,
        }
    )


def _scores(
    batch: dict[str, object],
    *,
    candidate_minus_reference: float = -1.0,
) -> dict[str, object]:
    selected = batch["scoring_prediction_by_source_unit"]
    assert isinstance(selected, dict)
    rows = []
    for object_id, episode, stratum in SOURCE_ROSTER:
        rows.append(
            {
                "object_id": object_id,
                "episode": episode,
                "stratum": stratum,
                "prediction_id": selected[f"{object_id}#{episode}"],
                "disposition": "candidate",
                "point_mean_identity": True,
                "point_metric_difference_m": 0.0,
                "supported_or_exact_fallback": True,
                "exact_fallback": False,
                "candidate_nll": candidate_minus_reference,
                "reference_nll": 0.0,
            }
        )
    return seal_source_scores(
        {
            "schema": SCORES_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "batch_id": batch["batch_id"],
            "information_boundary": {
                "source_suffix_opened": True,
                "confirmation_payloads_opened": False,
                "confirmation_predictions_run": False,
                "confirmation_outcomes_used": False,
                "candidate_retuned": False,
                "replacement_used": False,
            },
            "rows": rows,
        }
    )


def test_positive_source_gate_authorizes_predictions_not_outcomes() -> None:
    batch = _batch()
    decision = evaluate_source_gate(batch, _scores(batch))
    assert decision["status"] == "source-positive"
    assert decision["confirmation_prediction_authorized"] is True
    assert decision["confirmation_payload_opening_authorized"] is False
    assert decision["confirmation_outcome_opening_authorized"] is False
    assert decision["claim_authorized"] is False
    assert decision["mean_candidate_minus_reference_nll"] == -1.0


def test_nonnegative_mean_is_complete_source_negative() -> None:
    batch = _batch()
    decision = evaluate_source_gate(
        batch,
        _scores(batch, candidate_minus_reference=0.0),
    )
    assert decision["status"] == "source-negative"
    assert decision["confirmation_prediction_authorized"] is False
    assert "overall-mean-nll-difference-not-negative" in decision["reasons"]


def test_technical_failure_is_retained_and_keeps_target_closed() -> None:
    batch = _batch()
    scores = _scores(batch)
    scores.pop("score_set_id")
    row = scores["rows"][0]
    assert isinstance(row, dict)
    row.update(
        {
            "disposition": "technical_failure",
            "candidate_nll": None,
            "reference_nll": None,
            "supported_or_exact_fallback": False,
            "technical_failure_reason": "retained-source-processing-failure",
        }
    )
    scores = seal_source_scores(scores)
    decision = evaluate_source_gate(batch, scores)
    assert decision["status"] == "source-technical-negative"
    assert decision["technical_failure_count"] == 1
    assert decision["confirmation_prediction_authorized"] is False


def test_batch_rejects_target_use_and_incomplete_barrier() -> None:
    batch = _batch()
    changed = copy.deepcopy(batch)
    changed.pop("batch_id")
    boundary = changed["information_boundary"]
    assert isinstance(boundary, dict)
    boundary["confirmation_outcomes_used"] = True
    with pytest.raises(ValueError, match="information boundary"):
        seal_prediction_batch(changed)

    changed = copy.deepcopy(batch)
    changed.pop("batch_id")
    records = changed["records"]
    assert isinstance(records, list)
    records.pop()
    with pytest.raises(ValueError, match="exactly 100"):
        seal_prediction_batch(changed)


def test_batch_rejects_mean_change_and_non_diagonal_scoring_record() -> None:
    batch = _batch()
    changed = copy.deepcopy(batch)
    changed.pop("batch_id")
    records = changed["records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    record["mean_sha256"] = _sha("changed-mean")
    with pytest.raises(ValueError, match="changed the registered mean"):
        seal_prediction_batch(changed)

    changed = copy.deepcopy(batch)
    changed.pop("batch_id")
    selected = changed["scoring_prediction_by_source_unit"]
    records = changed["records"]
    assert isinstance(selected, dict)
    assert isinstance(records, list)
    first_object, first_episode, _ = SOURCE_ROSTER[0]
    non_diagonal = next(
        item
        for item in records
        if item["object_id"] == first_object
        and item["episode"] == first_episode
        and item["outer_fold_index"] == 1
    )
    selected[f"{first_object}#{first_episode}"] = non_diagonal[
        "prediction_id"
    ]
    with pytest.raises(ValueError, match="frozen diagonal"):
        seal_prediction_batch(changed)
