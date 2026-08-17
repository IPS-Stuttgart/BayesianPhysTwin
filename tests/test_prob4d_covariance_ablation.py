from __future__ import annotations

import json
import runpy
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from bayesian_phystwin.cli.command_registry import COMMANDS_BY_ID
from bayesian_phystwin.cli.prob4d_covariance_ablation import main
from bayesian_phystwin.prob4d_covariance_ablation import (
    ALLOWED_VARIANT_DIFFERENCE,
    COVARIANCE_TREATMENTS,
    INVARIANT_DIGEST_FIELDS,
    PROB4D_COVARIANCE_ABLATION_REPORT_SCHEMA,
    PROB4D_COVARIANCE_ABLATION_SCHEMA,
    CovarianceVariantV1,
    Prob4DCovarianceAblationV1,
    analyze_prob4d_covariance_ablation,
)
from bayesian_phystwin.strict_json_report_io import (
    canonical_json_sha256,
    load_strict_json_mapping,
    publish_json_report,
)

_METHODS = {
    treatment: "prob4d-" + treatment.replace("_", "-")
    for treatment in COVARIANCE_TREATMENTS
}
_PARAMETERS = {
    "full_joint": (1.0, True),
    "block_diagonal": (1.0, False),
    "independent_rows": (0.0, False),
    "shared_uncertainty_removed": (0.0, False),
    "shared_uncertainty_underreported": (0.5, True),
}
_INVARIANT_DIGESTS = {
    name: format(index + 100, "064x")
    for index, name in enumerate(INVARIANT_DIGEST_FIELDS)
}


def _variant(treatment: str, index: int) -> dict[str, object]:
    scale, gauge = _PARAMETERS[treatment]
    return {
        "method": _METHODS[treatment],
        "treatment": treatment,
        "shared_uncertainty_scale": scale,
        "gauge_factors_enabled": gauge,
        "run_manifest_sha256": format(index + 200, "064x"),
        "covariance_artifact_sha256": format(index + 300, "064x"),
        **_INVARIANT_DIGESTS,
    }


def _record(
    unit_id: str,
    treatment: str,
    *,
    loss: float,
    fallback_loss: float,
) -> dict[str, object]:
    return {
        "unit_id": unit_id,
        "group_id": unit_id,
        "metric": "track_error",
        "method": _METHODS[treatment],
        "loss": loss,
        "fallback_loss": fallback_loss,
        "risk_score": 0.1 + 0.01 * COVARIANCE_TREATMENTS.index(treatment),
        "accepted": True,
        "deployed_loss": loss,
        "horizon": "future-1",
        "reliability": 0.8,
        "identifiable_rank": 3,
        "intervals": [
            {
                "nominal_coverage": 0.9,
                "covered": loss <= fallback_loss,
                "width": 0.2,
            }
        ],
    }


def _payload() -> dict[str, object]:
    losses = {
        "full_joint": (0.80, 0.90),
        "block_diagonal": (0.95, 1.00),
        "independent_rows": (1.00, 1.10),
        "shared_uncertainty_removed": (1.05, 1.15),
        "shared_uncertainty_underreported": (0.90, 0.95),
    }
    records: list[dict[str, object]] = []
    for unit_index, unit_id in enumerate(("object-01", "object-02")):
        fallback_loss = 1.2 + 0.1 * unit_index
        for treatment in COVARIANCE_TREATMENTS:
            records.append(
                _record(
                    unit_id,
                    treatment,
                    loss=losses[treatment][unit_index],
                    fallback_loss=fallback_loss,
                )
            )
    return {
        "schema": PROB4D_COVARIANCE_ABLATION_SCHEMA,
        "schema_version": 1,
        "ablation_id": "fresh-object-prob4d-covariance-v1",
        "reference_treatment": "independent_rows",
        "locked_factors": {
            "dataset_id": "fresh-object-panel-v1",
            "split_id": "confirmation-v1",
            "registered_statistical_unit": "physical object",
            "source_or_calibration_policy_frozen": True,
            "allowed_variant_difference": ALLOWED_VARIANT_DIFFERENCE,
            "provider_revision": "prob4d@0123456",
        },
        "variants": [
            _variant(treatment, index)
            for index, treatment in enumerate(COVARIANCE_TREATMENTS)
        ],
        "evidence": {
            "contract": "bayesian-phystwin-decisive-evidence-v1",
            "schema_version": 1,
            "protocol_id": "fresh-object-prob4d-covariance-v1",
            "statistical_unit": "physical object",
            "claim_boundary": "controlled covariance attribution only",
            "reference_method": _METHODS["independent_rows"],
            "records": records,
        },
        "metadata": {"target_outcomes_opened": True, "analysis": "paired"},
    }


def _variants(payload: dict[str, object]) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], payload["variants"])


def _records(payload: dict[str, object]) -> list[dict[str, Any]]:
    evidence = cast(dict[str, Any], payload["evidence"])
    return cast(list[dict[str, Any]], evidence["records"])


def test_report_binds_invariants_and_extracts_full_joint_attribution() -> None:
    report = cast(dict[str, Any], analyze_prob4d_covariance_ablation(_payload()))

    assert report["schema"] == PROB4D_COVARIANCE_ABLATION_REPORT_SCHEMA
    assert report["complete_five_way_ablation"] is True
    assert report["only_covariance_treatment_varied"] is True
    assert report["claim_authorized"] is False
    assert report["reference_method"] == _METHODS["independent_rows"]
    assert len(report["report_id"]) == 64
    assert report["invariant_digests"] == _INVARIANT_DIGESTS
    assert set(report["variants"]) == set(COVARIANCE_TREATMENTS)

    attribution = report["full_joint_attribution"]
    comparisons = attribution["comparisons"]
    assert attribution["full_joint_method"] == _METHODS["full_joint"]
    assert set(comparisons) == set(COVARIANCE_TREATMENTS) - {"full_joint"}
    independent = comparisons["independent_rows"]["metrics"]["track_error"]
    assert independent["raw"]["relative_change_of_means"] < 0.0
    assert independent["operational"]["relative_change_of_means"] < 0.0


def test_report_identity_is_canonical_over_variant_and_mapping_order() -> None:
    first = _payload()
    second = deepcopy(first)
    second["variants"] = list(reversed(_variants(second)))
    second["locked_factors"] = dict(
        reversed(list(cast(dict[str, Any], second["locked_factors"]).items()))
    )
    second["metadata"] = dict(
        reversed(list(cast(dict[str, Any], second["metadata"]).items()))
    )

    first_report = analyze_prob4d_covariance_ablation(first)
    second_report = analyze_prob4d_covariance_ablation(second)
    assert first_report["input_content_sha256"] == second_report["input_content_sha256"]
    assert first_report["report_id"] == second_report["report_id"]


def test_variant_contract_rejects_invalid_treatment_parameters_and_digest() -> None:
    valid = _variant("full_joint", 0)
    with pytest.raises(ValueError, match="full_joint requires"):
        CovarianceVariantV1.from_mapping(
            {**valid, "shared_uncertainty_scale": 0.9},
            index=0,
        )
    underreported = _variant("shared_uncertainty_underreported", 1)
    with pytest.raises(ValueError, match="requires a scale inside"):
        CovarianceVariantV1.from_mapping(
            {**underreported, "shared_uncertainty_scale": 1.0},
            index=0,
        )
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        CovarianceVariantV1.from_mapping(
            {**valid, "run_manifest_sha256": "ABC"},
            index=0,
        )
    with pytest.raises(ValueError, match="unknown fields"):
        CovarianceVariantV1.from_mapping({**valid, "extra": True}, index=0)


def test_ablation_rejects_incomplete_or_nonisolated_variants() -> None:
    incomplete = _payload()
    incomplete["variants"] = _variants(incomplete)[:-1]
    with pytest.raises(ValueError, match="complete five-way"):
        analyze_prob4d_covariance_ablation(incomplete)

    changed = _payload()
    _variants(changed)[2]["physical_linearization_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="changed invariant digest"):
        analyze_prob4d_covariance_ablation(changed)

    duplicate_manifest = _payload()
    variants = _variants(duplicate_manifest)
    variants[1]["run_manifest_sha256"] = variants[0]["run_manifest_sha256"]
    with pytest.raises(ValueError, match="distinct run manifest"):
        analyze_prob4d_covariance_ablation(duplicate_manifest)

    duplicate_covariance = _payload()
    variants = _variants(duplicate_covariance)
    variants[1]["covariance_artifact_sha256"] = variants[0][
        "covariance_artifact_sha256"
    ]
    with pytest.raises(ValueError, match="distinct covariance artifact"):
        analyze_prob4d_covariance_ablation(duplicate_covariance)


def test_ablation_rejects_unfrozen_or_mismatched_evidence() -> None:
    unfrozen = _payload()
    cast(dict[str, Any], unfrozen["locked_factors"])[
        "source_or_calibration_policy_frozen"
    ] = False
    with pytest.raises(ValueError, match="must be true"):
        analyze_prob4d_covariance_ablation(unfrozen)

    protocol = _payload()
    cast(dict[str, Any], protocol["evidence"])["protocol_id"] = "other"
    with pytest.raises(ValueError, match="must equal ablation_id"):
        analyze_prob4d_covariance_ablation(protocol)

    reference = _payload()
    cast(dict[str, Any], reference["evidence"])["reference_method"] = _METHODS[
        "block_diagonal"
    ]
    with pytest.raises(ValueError, match="differs from reference_treatment"):
        analyze_prob4d_covariance_ablation(reference)

    methods = _payload()
    for record in _records(methods):
        if record["method"] == _METHODS["shared_uncertainty_removed"]:
            record["method"] = "unexpected-method"
    with pytest.raises(ValueError, match="exactly match"):
        analyze_prob4d_covariance_ablation(methods)


def test_low_level_contract_boundaries_are_rejected() -> None:
    with pytest.raises(ValueError, match="literal string keys"):
        Prob4DCovarianceAblationV1.from_mapping(cast(Any, {1: "value"}))

    invalid_sequence = _payload()
    invalid_sequence["variants"] = "full_joint"
    with pytest.raises(ValueError, match="must be a sequence"):
        analyze_prob4d_covariance_ablation(invalid_sequence)

    invalid_text = _payload()
    invalid_text["ablation_id"] = " untrimmed"
    with pytest.raises(ValueError, match="nonempty trimmed string"):
        analyze_prob4d_covariance_ablation(invalid_text)

    valid = _variant("full_joint", 0)
    with pytest.raises(ValueError, match="finite number"):
        CovarianceVariantV1.from_mapping(
            {**valid, "shared_uncertainty_scale": True},
            index=0,
        )
    with pytest.raises(ValueError, match="finite number"):
        CovarianceVariantV1.from_mapping(
            {**valid, "shared_uncertainty_scale": float("nan")},
            index=0,
        )
    with pytest.raises(ValueError, match="at least"):
        CovarianceVariantV1.from_mapping(
            {**valid, "shared_uncertainty_scale": -0.1},
            index=0,
        )
    with pytest.raises(ValueError, match="at most"):
        CovarianceVariantV1.from_mapping(
            {**valid, "shared_uncertainty_scale": 1.1},
            index=0,
        )
    with pytest.raises(ValueError, match="must be a bool"):
        CovarianceVariantV1.from_mapping(
            {**valid, "gauge_factors_enabled": 1},
            index=0,
        )

    missing_variant_field = dict(valid)
    missing_variant_field.pop("method")
    with pytest.raises(ValueError, match="missing fields"):
        CovarianceVariantV1.from_mapping(missing_variant_field, index=0)
    with pytest.raises(ValueError, match="unsupported covariance treatment"):
        CovarianceVariantV1.from_mapping(
            {**valid, "treatment": "unsupported"},
            index=0,
        )

    block_diagonal = _variant("block_diagonal", 1)
    with pytest.raises(ValueError, match="block_diagonal requires"):
        CovarianceVariantV1.from_mapping(
            {**block_diagonal, "gauge_factors_enabled": True},
            index=0,
        )
    independent = _variant("independent_rows", 2)
    with pytest.raises(ValueError, match="independent_rows requires"):
        CovarianceVariantV1.from_mapping(
            {**independent, "shared_uncertainty_scale": 0.1},
            index=0,
        )

    unsupported_reference = _payload()
    unsupported_reference["reference_treatment"] = "unsupported"
    with pytest.raises(ValueError, match="reference_treatment is unsupported"):
        analyze_prob4d_covariance_ablation(unsupported_reference)

    missing_lock = _payload()
    cast(dict[str, Any], missing_lock["locked_factors"]).pop("dataset_id")
    with pytest.raises(ValueError, match="locked_factors is missing fields"):
        analyze_prob4d_covariance_ablation(missing_lock)

    duplicate_methods = _payload()
    variants = _variants(duplicate_methods)
    variants[1]["method"] = variants[0]["method"]
    with pytest.raises(ValueError, match="method names must be unique"):
        analyze_prob4d_covariance_ablation(duplicate_methods)


def test_top_level_contract_fails_closed() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        Prob4DCovarianceAblationV1.from_mapping(cast(Any, []))
    with pytest.raises(ValueError, match="unsupported schema"):
        Prob4DCovarianceAblationV1.from_mapping({**_payload(), "schema": "other"})
    with pytest.raises(ValueError, match="schema_version"):
        Prob4DCovarianceAblationV1.from_mapping({**_payload(), "schema_version": True})
    with pytest.raises(ValueError, match="ablated comparator"):
        Prob4DCovarianceAblationV1.from_mapping(
            {**_payload(), "reference_treatment": "full_joint"}
        )
    with pytest.raises(ValueError, match="unknown fields"):
        Prob4DCovarianceAblationV1.from_mapping({**_payload(), "extra": True})


def test_cli_writes_content_bound_report_and_requires_overwrite(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "report.json"
    input_path.write_text(json.dumps(_payload()), encoding="utf-8")

    assert main([str(input_path), str(output_path)]) == 0
    report = cast(
        dict[str, Any],
        json.loads(output_path.read_text(encoding="utf-8")),
    )
    assert report["input_artifact"]["sha256"]
    status_sha256 = report.pop("status_sha256")
    assert status_sha256 == canonical_json_sha256(report)

    with pytest.raises(FileExistsError):
        main([str(input_path), str(output_path)])
    assert main([str(input_path), str(output_path), "--overwrite"]) == 0


def test_strict_json_io_rejects_ambiguous_inputs_and_owned_fields(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        load_strict_json_mapping(duplicate, artifact_label="ablation")

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite constant"):
        load_strict_json_mapping(nonfinite, artifact_label="ablation")

    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        load_strict_json_mapping(array, artifact_label="ablation")

    with pytest.raises(ValueError, match="artifact_label"):
        load_strict_json_mapping(array, artifact_label=" ablation")
    with pytest.raises(TypeError, match="genuine integer"):
        load_strict_json_mapping(
            array,
            artifact_label="ablation",
            maximum_input_bytes=cast(Any, True),
        )
    with pytest.raises(ValueError, match="positive"):
        load_strict_json_mapping(
            array,
            artifact_label="ablation",
            maximum_input_bytes=0,
        )
    with pytest.raises(ValueError, match="publication-owned fields"):
        publish_json_report(
            output,
            {"status_sha256": "forged"},
            input_artifact={},
        )
    with pytest.raises(TypeError, match="overwrite must be a bool"):
        publish_json_report(
            output,
            {},
            input_artifact={},
            overwrite=cast(Any, 1),
        )


def test_cli_module_main_and_registry_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "report.json"
    input_path.write_text(json.dumps(_payload()), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bpt diagnostic run audit-prob4d-covariance-ablation",
            str(input_path),
            str(output_path),
        ],
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_module(
            "bayesian_phystwin.cli.prob4d_covariance_ablation",
            run_name="__main__",
        )
    assert error.value.code == 0
    spec = COMMANDS_BY_ID["audit-prob4d-covariance-ablation"]
    assert spec.status.value == "diagnostic"
    assert spec.owner == "prob4d-covariance-ablation-v1"
    assert spec.optional_dependencies == ()
