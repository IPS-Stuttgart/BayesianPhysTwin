from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from bayesian_phystwin_experiments.deform360_covariance_only_source_gate_v1 import (
    BATCH_SCHEMA,
    COVARIANCE_DONOR_ID,
    COVARIANCE_EIGENVALUE_FLOOR_M2,
    COVARIANCE_SCALES,
    CROSSREPO_BINDING_ID,
    FUTURE_RANGE,
    OBSERVATION_STD_M,
    PAPER_PROTOCOL_ID,
    PREFIX_RANGE,
    REFERENCE_PREDICTOR_ID,
    SCHEMA_VERSION,
    SCORES_SCHEMA,
    SELECTION_SHA256,
    SOFTWARE_PROTOCOL_ID,
    SOURCE_ROSTER,
    evaluate_source_gate,
    seal_prediction_batch,
    seal_source_scores,
)

ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "scripts/science/run_deform360_covariance_only_source_gate_v1.py"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _content_id(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _record(*, fold_index: int, unit_index: int) -> dict[str, object]:
    object_id, episode, stratum = SOURCE_ROSTER[unit_index]
    outer_object, outer_episode, outer_stratum = SOURCE_ROSTER[fold_index]
    mean_id = _sha(f"mean:{object_id}:{episode}:{fold_index}")
    covariance_id = _sha(f"covariance:{object_id}:{episode}:{fold_index}")
    payload: dict[str, object] = {
        "software_protocol_id": SOFTWARE_PROTOCOL_ID,
        "paper_protocol_id": PAPER_PROTOCOL_ID,
        "crossrepo_binding_id": CROSSREPO_BINDING_ID,
        "selection_sha256": SELECTION_SHA256,
        "runtime_id": _sha("runtime"),
        "implementation_revision": "1" * 40,
        "distribution": {"name": "bayesian-phystwin", "version": "test"},
        "environment": {
            "byteorder": "little",
            "machine": "test",
            "python_implementation": "CPython",
            "python_version": "3.12.0",
            "system": "Linux",
        },
        "numerical_runtime": {
            "float64_epsilon": 2.220446049250313e-16,
            "numpy_version": "2.0.0",
        },
        "outer_fold_index": fold_index,
        "outer_fold_context": {
            "object_id": outer_object,
            "episode": outer_episode,
            "stratum": outer_stratum,
        },
        "source_unit_index": unit_index,
        "object_id": object_id,
        "episode": episode,
        "stratum": stratum,
        "prefix_range_half_open": list(PREFIX_RANGE),
        "future_range_half_open": list(FUTURE_RANGE),
        "future_horizon_steps": list(range(1, 19)),
        "future_horizon_bins": ["early"] * 6 + ["middle"] * 6 + ["late"] * 6,
        "reference_predictor_id": REFERENCE_PREDICTOR_ID,
        "covariance_donor_id": COVARIANCE_DONOR_ID,
        "early_middle_late_covariance_scales": list(COVARIANCE_SCALES),
        "observation_std_m": OBSERVATION_STD_M,
        "covariance_eigenvalue_floor_m2": COVARIANCE_EIGENVALUE_FLOOR_M2,
        "mean_sha256": mean_id,
        "reference_mean_sha256": mean_id,
        "mean_dtype": "<f8",
        "mean_shape": [18, 128, 3],
        "mean_c_contiguous": True,
        "mean_bytes_identical": True,
        "covariance_sha256": covariance_id,
        "covariance_shape": [18, 128, 3, 3],
        "covariance_dtype": "<f8",
        "covariance_units": "m^2",
        "covariance_diagnostics": {
            "finite": True,
            "minimum_eigenvalue_m2": COVARIANCE_EIGENVALUE_FLOOR_M2,
            "positive_semidefinite": True,
            "symmetric": True,
        },
        "disposition": "candidate",
        "exact_fallback": False,
        "exact_fallback_reference_identity": None,
        "technical_failure": False,
        "technical_failure_code": None,
        "diagnostic_code": "accepted",
        "input_files": [
            {
                "logical_name": "prefix/input.npz",
                "path": "source/input.npz",
                "sha256": _sha("input"),
                "size_bytes": 1,
            }
        ],
        "unit_manifest_id": _sha(f"manifest:{object_id}:{episode}"),
        "unit_manifest_file_sha256": _sha(f"manifest-file:{object_id}:{episode}"),
        "prediction_payload_sha256": _sha(f"payload:{object_id}:{episode}"),
        "source_suffix_used": False,
        "confirmation_outcomes_used": False,
    }
    return {**payload, "prediction_id": _content_id(payload)}


def _reseal_record(record: dict[str, object]) -> None:
    record.pop("prediction_id", None)
    record["prediction_id"] = _content_id(record)


def _batch() -> dict[str, object]:
    units = [
        {"object_id": object_id, "episode": episode, "stratum": stratum}
        for object_id, episode, stratum in SOURCE_ROSTER
    ]
    records: list[dict[str, object]] = []
    selected: dict[str, str] = {}
    for fold_index in range(10):
        for unit_index, (object_id, episode, _stratum) in enumerate(SOURCE_ROSTER):
            record = _record(fold_index=fold_index, unit_index=unit_index)
            records.append(record)
            if fold_index == unit_index:
                selected[f"{object_id}#{episode}"] = str(record["prediction_id"])
    return seal_prediction_batch(
        {
            "schema": BATCH_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "software_protocol_id": SOFTWARE_PROTOCOL_ID,
            "paper_protocol_id": PAPER_PROTOCOL_ID,
            "candidate": {
                "reference_predictor_id": REFERENCE_PREDICTOR_ID,
                "covariance_donor_id": COVARIANCE_DONOR_ID,
                "early_middle_late_covariance_scales": list(COVARIANCE_SCALES),
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


def _reseal_scores(scores: dict[str, object]) -> dict[str, object]:
    changed = copy.deepcopy(scores)
    changed.pop("score_set_id", None)
    return seal_source_scores(changed)


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "deform360_covariance_source_gate_cli",
        CLI_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    decision = evaluate_source_gate(batch, _reseal_scores(scores))
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
    _reseal_record(record)
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
    selected[f"{first_object}#{first_episode}"] = non_diagonal["prediction_id"]
    with pytest.raises(ValueError, match="frozen diagonal"):
        seal_prediction_batch(changed)


def test_batch_rejects_malformed_runtime_covariance_and_forbidden_input() -> None:
    batch = _batch()
    changed = copy.deepcopy(batch)
    changed.pop("batch_id")
    record = changed["records"][0]
    record["numerical_runtime"]["float64_epsilon"] = 1e-7
    _reseal_record(record)
    with pytest.raises(ValueError, match="numerical runtime changed"):
        seal_prediction_batch(changed)

    changed = copy.deepcopy(batch)
    changed.pop("batch_id")
    record = changed["records"][0]
    record["covariance_diagnostics"]["positive_semidefinite"] = False
    _reseal_record(record)
    with pytest.raises(ValueError, match="covariance diagnostics failed"):
        seal_prediction_batch(changed)

    changed = copy.deepcopy(batch)
    changed.pop("batch_id")
    record = changed["records"][0]
    record["input_files"][0]["path"] = "confirmation/forbidden.npz"
    _reseal_record(record)
    with pytest.raises(ValueError, match="forbidden suffix or target path"):
        seal_prediction_batch(changed)


def test_score_disposition_must_match_sealed_prediction() -> None:
    batch = _batch()
    changed = copy.deepcopy(batch)
    changed.pop("batch_id")
    selected = changed["scoring_prediction_by_source_unit"]
    records = changed["records"]
    assert isinstance(selected, dict)
    assert isinstance(records, list)
    object_id, episode, _ = SOURCE_ROSTER[0]
    selected_id = selected[f"{object_id}#{episode}"]
    record = next(item for item in records if item["prediction_id"] == selected_id)
    record["disposition"] = "exact_fallback"
    record["exact_fallback"] = True
    record["exact_fallback_reference_identity"] = record["covariance_sha256"]
    record["diagnostic_code"] = "insufficient-per-track-support"
    _reseal_record(record)
    selected[f"{object_id}#{episode}"] = record["prediction_id"]
    batch = seal_prediction_batch(changed)
    with pytest.raises(ValueError, match="does not match the selected prediction"):
        evaluate_source_gate(batch, _scores(batch))


def test_nontechnical_unsupported_row_is_rejected() -> None:
    batch = _batch()
    scores = _scores(batch)
    row = scores["rows"][0]
    assert isinstance(row, dict)
    row["supported_or_exact_fallback"] = False
    with pytest.raises(ValueError, match="supported or exact fallback"):
        evaluate_source_gate(batch, _reseal_scores(scores))


def test_technical_failure_cannot_claim_support_or_fallback() -> None:
    batch = _batch()
    scores = _scores(batch)
    row = scores["rows"][0]
    assert isinstance(row, dict)
    row.update(
        {
            "disposition": "technical_failure",
            "candidate_nll": None,
            "reference_nll": None,
            "technical_failure_reason": "retained-source-processing-failure",
        }
    )
    with pytest.raises(ValueError, match="cannot claim support"):
        evaluate_source_gate(batch, _reseal_scores(scores))


def test_cli_atomic_publication_is_no_clobber(tmp_path: Path) -> None:
    module = _load_cli_module()
    output = tmp_path / "decision.json"
    module._atomic_create(output, {"value": 1})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module._atomic_create(output, {"value": 2})
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}
