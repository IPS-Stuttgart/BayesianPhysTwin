from __future__ import annotations

import copy
import math

import numpy as np
import pytest

import bayesian_phystwin.probabilistic_scoring as scoring_module
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.decisive_evidence import parse_decisive_evidence
from bayesian_phystwin.probabilistic_scoring import (
    ENERGY_SCORE,
    GAUSSIAN_NLL_PER_DIMENSION,
    PROBABILISTIC_SCORE_INPUT_CONTRACT,
    VARIOGRAM_SCORE,
    WEIGHTED_INTERVAL_SCORE,
    build_decisive_evidence_from_score_report,
    energy_score,
    gaussian_nll_per_dimension,
    interval_score,
    parse_probabilistic_score_bundle,
    score_probabilistic_bundle,
    variogram_score,
    weighted_interval_score,
)

SCORE_NAMES_FOR_TEST = (
    ENERGY_SCORE,
    VARIOGRAM_SCORE,
    GAUSSIAN_NLL_PER_DIMENSION,
    WEIGHTED_INTERVAL_SCORE,
)


def _arm(
    method: str,
    *,
    accepted: bool,
    offset: float,
) -> dict[str, object]:
    mean = [offset, -0.5 * offset]
    return {
        "method": method,
        "accepted": accepted,
        "risk_score": abs(offset),
        "reliability": max(0.0, 1.0 - abs(offset)),
        "identifiable_rank": 2,
        "samples": [
            [mean[0] - 0.05, mean[1] + 0.02],
            [mean[0] + 0.05, mean[1] - 0.02],
        ],
        "sample_weights": [0.4, 0.6],
        "gaussian_mean": mean,
        "gaussian_covariance": [[0.04, 0.01], [0.01, 0.03]],
        "median": mean,
        "intervals": [
            {
                "nominal_coverage": 0.5,
                "lower": [mean[0] - 0.2, mean[1] - 0.2],
                "upper": [mean[0] + 0.2, mean[1] + 0.2],
            },
            {
                "nominal_coverage": 0.9,
                "lower": [mean[0] - 0.5, mean[1] - 0.5],
                "upper": [mean[0] + 0.5, mean[1] + 0.5],
            },
        ],
    }


def _payload() -> dict[str, object]:
    return {
        "contract": PROBABILISTIC_SCORE_INPUT_CONTRACT,
        "schema_version": 1,
        "protocol_id": "bayesian-value-decomposition-test-v1",
        "statistical_unit": "physical-object-or-session",
        "claim_boundary": "synthetic test fixture; no scientific claim",
        "fallback_method": "physical_fallback",
        "reference_method": "last_residual",
        "comparison_pairs": [
            {
                "comparison_id": "bayesian_full_vs_last_residual",
                "candidate_method": "bayesian_full_guarded",
                "reference_method": "last_residual",
            },
            {
                "comparison_id": "last_residual_vs_physical_fallback",
                "candidate_method": "last_residual",
                "reference_method": "physical_fallback",
            },
        ],
        "score_configuration": {
            "score_names": [
                ENERGY_SCORE,
                VARIOGRAM_SCORE,
                GAUSSIAN_NLL_PER_DIMENSION,
                WEIGHTED_INTERVAL_SCORE,
            ],
            "energy_beta": 1.0,
            "variogram_order": 0.5,
            "variogram_pairs": [[0, 1]],
            "variogram_pair_weights": [1.0],
            "gaussian_maximum_condition_number": 1e8,
        },
        "units": [
            {
                "unit_id": "unit-a",
                "group_id": "group-a",
                "horizon": 1,
                "observation": [0.0, 0.0],
                "predictions": [
                    _arm("bayesian_full_guarded", accepted=True, offset=0.02),
                    _arm("last_residual", accepted=True, offset=0.08),
                    _arm("physical_fallback", accepted=True, offset=0.5),
                ],
            },
            {
                "unit_id": "unit-b",
                "group_id": "group-b",
                "horizon": 2,
                "observation": [0.1, -0.1],
                "predictions": [
                    _arm("bayesian_full_guarded", accepted=False, offset=0.1),
                    _arm("last_residual", accepted=True, offset=0.12),
                    _arm("physical_fallback", accepted=True, offset=0.6),
                ],
            },
        ],
    }


def test_energy_score_matches_manual_weighted_empirical_value() -> None:
    samples = np.asarray([[0.0], [2.0]])
    assert energy_score(samples, [1.0]) == pytest.approx(0.5)
    assert energy_score(samples[::-1], [1.0]) == pytest.approx(0.5)
    assert energy_score(
        samples,
        [1.0],
        sample_weights=[0.25, 0.75],
        block_size=1,
    ) == pytest.approx(0.625)


def test_variogram_score_uses_registered_pairs_and_normalized_weights() -> None:
    samples = np.asarray(
        [
            [0.0, 1.0, 4.0],
            [0.0, 3.0, 2.0],
        ]
    )
    score = variogram_score(
        samples,
        [0.0, 2.0, 4.0],
        pair_indices=[[0, 1], [1, 2]],
        order=1.0,
        pair_weights=[1.0, 3.0],
    )
    assert score == pytest.approx(0.0)
    with pytest.raises(ValueError, match="left < right"):
        variogram_score(samples, [0.0, 2.0, 4.0], pair_indices=[[1, 1]])


def test_gaussian_negative_log_score_is_exact_and_fail_closed() -> None:
    expected = 0.5 * math.log(2.0 * math.pi)
    assert gaussian_nll_per_dimension([0.0, 0.0], np.eye(2), [0.0, 0.0]) == (
        pytest.approx(expected)
    )
    with pytest.raises(ValueError, match="positive definite"):
        gaussian_nll_per_dimension(
            [0.0, 0.0],
            [[1.0, 1.0], [1.0, 1.0]],
            [0.0, 0.0],
        )
    with pytest.raises(ValueError, match="condition-number"):
        gaussian_nll_per_dimension(
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1e-12]],
            [0.0, 0.0],
            maximum_condition_number=1e6,
        )


def test_interval_and_weighted_interval_scores_penalize_width_and_misses() -> None:
    assert interval_score([0.0], [2.0], [1.0], nominal_coverage=0.9) == 2.0
    missed = interval_score([0.0], [1.0], [2.0], nominal_coverage=0.5)
    assert missed == pytest.approx(5.0)
    assert (
        weighted_interval_score(
            [0.0],
            [[0.0], [0.0]],
            [[0.0], [0.0]],
            [0.0],
            nominal_coverages=(0.5, 0.9),
        )
        == 0.0
    )


def test_bundle_is_matched_deterministic_and_decisive_evidence_compatible() -> None:
    payload = _payload()
    bundle = parse_probabilistic_score_bundle(payload)
    assert bundle.units[0].observation.flags.writeable is False
    report = score_probabilistic_bundle(bundle)
    repeated = score_probabilistic_bundle(payload)
    assert report == repeated
    assert len(report["report_id"]) == 64
    assert report["claim_authorized"] is False
    assert set(report["aggregate"]) == {
        ENERGY_SCORE,
        VARIOGRAM_SCORE,
        GAUSSIAN_NLL_PER_DIMENSION,
        WEIGHTED_INTERVAL_SCORE,
    }
    attribution = report["pairwise_attribution"]["bayesian_full_vs_last_residual"][
        "scores"
    ][ENERGY_SCORE]
    assert attribution["equal_group_mean_raw_score_difference"] < 0.0
    assert attribution["equal_group_mean_deployed_score_difference"] > 0.0
    assert attribution["candidate_better_unit_count"] == 1
    assert attribution["candidate_worse_unit_count"] == 1

    evidence = build_decisive_evidence_from_score_report(report)
    gaussian_offsets = [
        row["common_additive_offset"]
        for row in evidence["loss_offsets"]
        if row["score_name"] == GAUSSIAN_NLL_PER_DIMENSION
    ]
    assert gaussian_offsets and max(gaussian_offsets) > 0.0
    parsed = parse_decisive_evidence(evidence)
    assert parsed.protocol_id == "bayesian-value-decomposition-test-v1"
    assert len(parsed.records) == 2 * 3 * 4

    rejected_energy = next(
        record
        for record in parsed.records
        if record.unit_id == "unit-b"
        and record.method == "bayesian_full_guarded"
        and record.metric == f"probabilistic/{ENERGY_SCORE}"
    )
    assert rejected_energy.accepted is False
    assert rejected_energy.deployed_loss == rejected_energy.fallback_loss
    assert rejected_energy.intervals == ()

    rejected_wis = next(
        record
        for record in parsed.records
        if record.unit_id == "unit-b"
        and record.method == "bayesian_full_guarded"
        and record.metric == f"probabilistic/{WEIGHTED_INTERVAL_SCORE}"
    )
    assert rejected_wis.intervals
    assert rejected_wis.intervals[0].width == pytest.approx(0.4)


def test_bundle_rejects_method_and_interval_mismatch() -> None:
    payload = _payload()
    invalid = copy.deepcopy(payload)
    invalid["units"][0]["predictions"].reverse()  # type: ignore[index]
    with pytest.raises(ValueError, match="unique and sorted"):
        parse_probabilistic_score_bundle(invalid)

    invalid = copy.deepcopy(payload)
    invalid["units"][1]["predictions"][0]["intervals"][0][  # type: ignore[index]
        "nominal_coverage"
    ] = 0.6
    with pytest.raises(ValueError, match="same interval coverages"):
        parse_probabilistic_score_bundle(invalid)

    invalid = copy.deepcopy(payload)
    invalid["comparison_pairs"][0][  # type: ignore[index]
        "candidate_method"
    ] = "unknown"
    with pytest.raises(ValueError, match="unknown methods"):
        parse_probabilistic_score_bundle(invalid)

    invalid = copy.deepcopy(payload)
    invalid["comparison_pairs"].reverse()  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="unique and sorted"):
        parse_probabilistic_score_bundle(invalid)


def test_configuration_rejects_hidden_or_unused_scoring_fields() -> None:
    payload = _payload()
    invalid = copy.deepcopy(payload)
    invalid["score_configuration"][  # type: ignore[index]
        "score_names"
    ] = [ENERGY_SCORE]
    with pytest.raises(ValueError, match="variogram fields are unused"):
        parse_probabilistic_score_bundle(invalid)

    invalid = copy.deepcopy(payload)
    invalid["score_configuration"]["score_names"] = [  # type: ignore[index]
        WEIGHTED_INTERVAL_SCORE,
        ENERGY_SCORE,
    ]
    with pytest.raises(ValueError, match="canonical score order"):
        parse_probabilistic_score_bundle(invalid)


def test_bayesian_value_profile_example_is_executable_and_fail_closed() -> None:
    import json
    from pathlib import Path

    from bayesian_phystwin.probabilistic_scoring import (
        BAYESIAN_VALUE_DECOMPOSITION_PROFILE,
        validate_bayesian_value_decomposition_bundle,
    )

    root = Path(__file__).resolve().parents[1]
    example = json.loads(
        (root / "examples/bayesian_value_decomposition_score_input.json").read_text(
            encoding="utf-8"
        )
    )
    bundle = validate_bayesian_value_decomposition_bundle(example)
    assert bundle.analysis_profile == BAYESIAN_VALUE_DECOMPOSITION_PROFILE
    assert len(bundle.comparison_pairs) == 4
    report = score_probabilistic_bundle(bundle)
    guard_attribution = report["pairwise_attribution"][
        "guarded-last-residual-vs-last-residual"
    ]
    for score_name in SCORE_NAMES_FOR_TEST:
        assert guard_attribution["scores"][score_name][
            "mean_raw_score_difference"
        ] == pytest.approx(0.0)

    invalid = copy.deepcopy(example)
    invalid["units"][0]["predictions"][3]["samples"][0][0] += 0.001
    with pytest.raises(ValueError, match="raw prediction must equal"):
        validate_bayesian_value_decomposition_bundle(invalid)

    protocol = json.loads(
        (root / "protocols/templates/bayesian_value_decomposition_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["protocol_id"] == BAYESIAN_VALUE_DECOMPOSITION_PROFILE
    assert protocol["claim_authorized"] is False


def test_public_score_functions_fail_closed_on_malformed_inputs() -> None:
    malformed = [
        (
            lambda: energy_score([[0.0]], [0.0], beta=2.0),
            ValueError,
            "strictly inside",
        ),
        (
            lambda: energy_score([[0.0]], [0.0], block_size=True),
            TypeError,
            "genuine integer",
        ),
        (
            lambda: energy_score([[0.0]], [0.0], block_size=0),
            ValueError,
            "positive",
        ),
        (
            lambda: energy_score([0.0], [0.0]),
            ValueError,
            "at least 2 dimensions",
        ),
        (
            lambda: energy_score(
                [[0.0], [1.0]],
                [0.0],
                sample_weights=[1.0],
            ),
            ValueError,
            "shape",
        ),
        (
            lambda: energy_score(
                [[0.0], [1.0]],
                [0.0],
                sample_weights=[1.0, -1.0],
            ),
            ValueError,
            "nonnegative",
        ),
        (
            lambda: energy_score(
                [[0.0], [1.0]],
                [0.0],
                sample_weights=[0.0, 0.0],
            ),
            ValueError,
            "positive finite mass",
        ),
        (
            lambda: variogram_score(
                [[0.0, 1.0]],
                [0.0, 1.0],
                pair_indices=[[0.0, 1.0]],
            ),
            ValueError,
            "integer array",
        ),
        (
            lambda: variogram_score(
                [[0.0, 1.0]],
                [0.0, 1.0],
                pair_indices=np.empty((0, 2), dtype=np.int64),
            ),
            ValueError,
            "must not be empty",
        ),
        (
            lambda: variogram_score(
                [[0.0, 1.0]],
                [0.0, 1.0],
                pair_indices=[[0, 2]],
            ),
            ValueError,
            "out-of-range",
        ),
        (
            lambda: variogram_score(
                [[0.0, 1.0]],
                [0.0, 1.0],
                pair_indices=[[0, 1], [0, 1]],
            ),
            ValueError,
            "duplicates",
        ),
        (
            lambda: gaussian_nll_per_dimension(
                [0.0],
                [[1.0]],
                [0.0, 1.0],
            ),
            ValueError,
            "mean shape",
        ),
        (
            lambda: gaussian_nll_per_dimension(
                [0.0, 0.0],
                [[1.0]],
                [0.0, 0.0],
            ),
            ValueError,
            "must have shape",
        ),
        (
            lambda: gaussian_nll_per_dimension(
                [0.0, 0.0],
                [[1.0, 0.1], [0.2, 1.0]],
                [0.0, 0.0],
            ),
            ValueError,
            "symmetric",
        ),
        (
            lambda: interval_score(
                [0.0],
                [1.0, 2.0],
                [0.0],
                nominal_coverage=0.9,
            ),
            ValueError,
            "must match",
        ),
        (
            lambda: interval_score(
                [1.0],
                [0.0],
                [0.5],
                nominal_coverage=0.9,
            ),
            ValueError,
            "lower bounds exceed",
        ),
        (
            lambda: interval_score(
                [0.0],
                [1.0],
                [0.5],
                nominal_coverage=1.0,
            ),
            ValueError,
            "strictly inside",
        ),
        (
            lambda: weighted_interval_score(
                [0.0, 0.0],
                [[0.0]],
                [[1.0]],
                [0.0],
                nominal_coverages=(0.5,),
            ),
            ValueError,
            "median shape",
        ),
        (
            lambda: weighted_interval_score(
                [0.0],
                [[0.0]],
                [[1.0], [2.0]],
                [0.0],
                nominal_coverages=(0.5,),
            ),
            ValueError,
            "interval arrays",
        ),
        (
            lambda: weighted_interval_score(
                [0.0],
                [[0.0]],
                [[1.0]],
                [0.0],
                nominal_coverages=(0.5, 0.9),
            ),
            ValueError,
            "length differs",
        ),
        (
            lambda: weighted_interval_score(
                [0.0],
                [[0.0], [0.0]],
                [[1.0], [1.0]],
                [0.0],
                nominal_coverages=(0.9, 0.5),
            ),
            ValueError,
            "strictly increasing",
        ),
        (
            lambda: weighted_interval_score(
                [0.0],
                [[0.0]],
                [[1.0]],
                [0.0],
                nominal_coverages=(1.0,),
            ),
            ValueError,
            "strictly inside",
        ),
        (
            lambda: weighted_interval_score(
                [0.0],
                [[1.0]],
                [[0.0]],
                [0.5],
                nominal_coverages=(0.5,),
            ),
            ValueError,
            "lower bounds exceed",
        ),
    ]
    for call, error, match in malformed:
        with pytest.raises(error, match=match):
            call()


def test_score_configuration_contract_rejects_ambiguous_fields() -> None:
    invalid_configurations = [
        ({"score_names": []}, "unique and nonempty"),
        (
            {"score_names": [ENERGY_SCORE, ENERGY_SCORE], "energy_beta": 1.0},
            "unique and nonempty",
        ),
        ({"score_names": ["unknown"]}, "unknown probabilistic scores"),
        ({"score_names": [ENERGY_SCORE]}, "energy_beta is required"),
        (
            {
                "score_names": [GAUSSIAN_NLL_PER_DIMENSION],
                "energy_beta": 1.0,
            },
            "energy_beta is unused",
        ),
        (
            {"score_names": [ENERGY_SCORE], "energy_beta": 2.0},
            "strictly inside",
        ),
        (
            {"score_names": [VARIOGRAM_SCORE], "variogram_order": 0.5},
            "configuration missing",
        ),
        (
            {
                "score_names": [VARIOGRAM_SCORE],
                "variogram_order": 0.5,
                "variogram_pairs": [0, 1],
            },
            "integer shape",
        ),
        (
            {
                "score_names": [VARIOGRAM_SCORE],
                "variogram_order": 0.5,
                "variogram_pairs": [],
            },
            "integer shape",
        ),
        (
            {
                "score_names": [VARIOGRAM_SCORE],
                "variogram_order": 0.5,
                "variogram_pairs": [[-1, 1]],
            },
            "nonnegative",
        ),
        (
            {
                "score_names": [VARIOGRAM_SCORE],
                "variogram_order": 0.5,
                "variogram_pairs": [[1, 1]],
            },
            "left < right",
        ),
        (
            {
                "score_names": [VARIOGRAM_SCORE],
                "variogram_order": 0.5,
                "variogram_pairs": [[0, 1], [0, 1]],
            },
            "duplicates",
        ),
        (
            {
                "score_names": [VARIOGRAM_SCORE],
                "variogram_order": 0.5,
                "variogram_pairs": [[0, 1]],
                "variogram_pair_weights": [1.0, 1.0],
            },
            "shape",
        ),
        (
            {
                "score_names": [ENERGY_SCORE],
                "energy_beta": 1.0,
                "gaussian_maximum_condition_number": 1e8,
            },
            "unused without Gaussian NLL",
        ),
    ]
    for configuration, match in invalid_configurations:
        payload = _payload()
        payload["score_configuration"] = configuration
        with pytest.raises(ValueError, match=match):
            parse_probabilistic_score_bundle(payload)


def test_bundle_contract_rejects_malformed_registered_units() -> None:
    with pytest.raises(ValueError, match="input must be a JSON object"):
        parse_probabilistic_score_bundle([])

    invalid_cases: list[tuple[dict[str, object], str]] = []

    payload = _payload()
    del payload["protocol_id"]
    invalid_cases.append((payload, "fields changed"))

    payload = _payload()
    payload["unexpected"] = True
    invalid_cases.append((payload, "fields changed"))

    payload = _payload()
    payload["contract"] = "wrong"
    invalid_cases.append((payload, "contract must be"))

    payload = _payload()
    payload["schema_version"] = True
    invalid_cases.append((payload, "integer 1"))

    payload = _payload()
    payload["units"] = []
    invalid_cases.append((payload, "units must not be empty"))

    payload = _payload()
    payload["fallback_method"] = "last_residual"
    invalid_cases.append((payload, "must differ"))

    payload = _payload()
    payload["analysis_profile"] = "unknown"
    invalid_cases.append((payload, "unknown analysis_profile"))

    payload = _payload()
    payload["units"][1]["unit_id"] = "unit-a"  # type: ignore[index]
    invalid_cases.append((payload, "duplicate unit_id"))

    payload = _payload()
    payload["units"][0]["predictions"] = []  # type: ignore[index]
    invalid_cases.append((payload, "predictions must not be empty"))

    payload = _payload()
    payload["units"][1]["predictions"].pop()  # type: ignore[index]
    invalid_cases.append((payload, "same sorted methods"))

    payload = _payload()
    payload["fallback_method"] = "missing"
    invalid_cases.append((payload, "lacks fallback_method"))

    payload = _payload()
    payload["reference_method"] = "missing"
    invalid_cases.append((payload, "lacks reference_method"))

    payload = _payload()
    payload["units"][0]["predictions"][2]["accepted"] = False  # type: ignore[index]
    invalid_cases.append((payload, "fallback_method must be marked accepted"))

    payload = _payload()
    payload["units"][0]["horizon"] = True  # type: ignore[index]
    invalid_cases.append((payload, "label or nonnegative number"))

    payload = _payload()
    payload["units"][0]["predictions"][0]["reliability"] = 1.1  # type: ignore[index]
    invalid_cases.append((payload, "at most 1.0"))

    payload = _payload()
    predictions = payload["units"][0]["predictions"]  # type: ignore[index]
    predictions[0]["identifiable_rank"] = True
    invalid_cases.append((payload, "nonnegative integer or null"))

    payload = _payload()
    payload["score_configuration"]["variogram_pairs"] = [[0, 2]]  # type: ignore[index]
    invalid_cases.append((payload, "exceed the registered query dimension"))

    for invalid, match in invalid_cases:
        with pytest.raises(ValueError, match=match):
            parse_probabilistic_score_bundle(invalid)


def test_comparison_contract_rejects_self_and_duplicate_attribution() -> None:
    payload = _payload()
    payload["comparison_pairs"] = [
        {
            "comparison_id": "self",
            "candidate_method": "last_residual",
            "reference_method": "last_residual",
        }
    ]
    with pytest.raises(ValueError, match="distinct methods"):
        parse_probabilistic_score_bundle(payload)

    payload = _payload()
    pair = copy.deepcopy(payload["comparison_pairs"][0])  # type: ignore[index]
    pair["comparison_id"] = "duplicate-method-pair"  # type: ignore[index]
    payload["comparison_pairs"].append(pair)  # type: ignore[union-attr]
    payload["comparison_pairs"].sort(  # type: ignore[union-attr]
        key=lambda value: value["comparison_id"]
    )
    with pytest.raises(ValueError, match="repeats a candidate/reference pair"):
        parse_probabilistic_score_bundle(payload)


def test_bayesian_value_profile_rejects_role_drift() -> None:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    example = json.loads(
        (root / "examples/bayesian_value_decomposition_score_input.json").read_text(
            encoding="utf-8"
        )
    )

    mutations = []
    value = copy.deepcopy(example)
    value["analysis_profile"] = "general-probabilistic-scoring-v1"
    mutations.append((value, "analysis_profile must select"))

    value = copy.deepcopy(example)
    value["fallback_method"] = "last_residual"
    value["reference_method"] = "bayesian_mean_guarded"
    mutations.append((value, "physical_fallback as fallback_method"))

    value = copy.deepcopy(example)
    value["reference_method"] = "bayesian_mean_guarded"
    mutations.append((value, "last_residual as reference_method"))

    value = copy.deepcopy(example)
    value["units"][0]["predictions"][2]["accepted"] = False
    mutations.append((value, "last_residual to be unguarded"))

    value = copy.deepcopy(example)
    value["comparison_pairs"] = value["comparison_pairs"][:-1]
    mutations.append((value, "four registered attribution pairs"))

    for invalid, match in mutations:
        with pytest.raises(ValueError, match=match):
            from bayesian_phystwin.probabilistic_scoring import (
                validate_bayesian_value_decomposition_bundle,
            )

            validate_bayesian_value_decomposition_bundle(invalid)


def test_decisive_evidence_adapter_rejects_corrupted_reports() -> None:
    report = score_probabilistic_bundle(_payload())
    with pytest.raises(ValueError, match="report must be a JSON object"):
        build_decisive_evidence_from_score_report([])

    corrupted = copy.deepcopy(report)
    corrupted["contract"] = "wrong"
    with pytest.raises(ValueError, match="report contract"):
        build_decisive_evidence_from_score_report(corrupted)

    corrupted = copy.deepcopy(report)
    corrupted["unit_score_rows"].append(  # type: ignore[union-attr]
        copy.deepcopy(corrupted["unit_score_rows"][0])  # type: ignore[index]
    )
    with pytest.raises(ValueError, match="duplicate score-report row"):
        build_decisive_evidence_from_score_report(corrupted)

    corrupted = copy.deepcopy(report)
    corrupted["unit_score_rows"] = [  # type: ignore[index]
        row
        for row in corrupted["unit_score_rows"]  # type: ignore[union-attr]
        if not (row["unit_id"] == "unit-a" and row["method"] == "physical_fallback")
    ]
    with pytest.raises(ValueError, match="lacks the fallback method"):
        build_decisive_evidence_from_score_report(corrupted)

    corrupted = copy.deepcopy(report)
    row = next(
        value
        for value in corrupted["unit_score_rows"]  # type: ignore[index]
        if value["unit_id"] == "unit-b" and value["method"] == "bayesian_full_guarded"
    )
    row["deployed_scores"][ENERGY_SCORE] += 0.1
    with pytest.raises(ValueError, match="exact fallback semantics"):
        build_decisive_evidence_from_score_report(corrupted)


def test_decisive_evidence_adapter_verifies_report_id_and_publication() -> None:
    report = score_probabilistic_bundle(_payload())

    corrupted = copy.deepcopy(report)
    corrupted["aggregate"][ENERGY_SCORE]["last_residual"][  # type: ignore[index]
        "mean_raw_score"
    ] += 0.1
    with pytest.raises(ValueError, match="report_id"):
        build_decisive_evidence_from_score_report(corrupted)

    corrupted = copy.deepcopy(report)
    corrupted["unexpected"] = True
    with pytest.raises(ValueError, match="report fields changed"):
        build_decisive_evidence_from_score_report(corrupted)

    published = {
        **copy.deepcopy(report),
        "input_artifact": {
            "path": "/registered/input.json",
            "sha256": "a" * 64,
            "bytes": 123,
        },
    }
    published["status_sha256"] = content_id(published)
    evidence = build_decisive_evidence_from_score_report(published)
    assert len(evidence["records"]) == 24

    published["status_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="status_sha256"):
        build_decisive_evidence_from_score_report(published)


def test_bundle_scalar_and_prediction_shape_edges() -> None:
    invalid_cases: list[tuple[dict[str, object], str]] = []

    payload = _payload()
    payload["units"] = "not-an-array"
    invalid_cases.append((payload, "must be a JSON array"))

    payload = _payload()
    payload["protocol_id"] = " padded "
    invalid_cases.append((payload, "nonempty literal string"))

    payload = _payload()
    payload["units"][0]["predictions"][0]["accepted"] = 1  # type: ignore[index]
    invalid_cases.append((payload, "must be a bool"))

    payload = _payload()
    payload["units"][0]["predictions"][0]["risk_score"] = "zero"  # type: ignore[index]
    invalid_cases.append((payload, "finite number"))

    payload = _payload()
    predictions = payload["units"][0]["predictions"]  # type: ignore[index]
    predictions[0]["risk_score"] = float("nan")
    invalid_cases.append((payload, "finite number"))

    payload = _payload()
    predictions = payload["units"][0]["predictions"]  # type: ignore[index]
    predictions[0]["identifiable_rank"] = -1
    invalid_cases.append((payload, "nonnegative integer or null"))

    payload = _payload()
    payload["units"][0]["horizon"] = {}  # type: ignore[index]
    invalid_cases.append((payload, "label or nonnegative number"))

    payload = _payload()
    payload["units"][0]["observation"] = [float("inf"), 0.0]  # type: ignore[index]
    invalid_cases.append((payload, "must be finite"))

    payload = _payload()
    payload["units"][0]["observation"] = []  # type: ignore[index]
    invalid_cases.append((payload, "must not be empty"))

    payload = _payload()
    payload["units"][0]["predictions"][0]["samples"] = [[0.0]]  # type: ignore[index]
    invalid_cases.append((payload, "must have shape"))

    payload = _payload()
    predictions = payload["units"][0]["predictions"]  # type: ignore[index]
    predictions[0]["gaussian_mean"] = [0.0]
    invalid_cases.append((payload, "gaussian_mean shape changed"))

    payload = _payload()
    payload["units"][0]["predictions"][0]["median"] = [0.0]  # type: ignore[index]
    invalid_cases.append((payload, "median shape changed"))

    payload = _payload()
    payload["units"][0]["predictions"][0]["intervals"] = []  # type: ignore[index]
    invalid_cases.append((payload, "increasing coverages"))

    payload = _payload()
    payload["units"][0]["predictions"][0]["intervals"][0][  # type: ignore[index]
        "lower"
    ] = [0.0]
    invalid_cases.append((payload, "bounds are inconsistent"))

    for invalid, match in invalid_cases:
        with pytest.raises(ValueError, match=match):
            parse_probabilistic_score_bundle(invalid)

    payload = _payload()
    payload["units"][0]["horizon"] = "short"  # type: ignore[index]
    parsed = parse_probabilistic_score_bundle(payload)
    assert parsed.units[0].horizon == "short"


def test_registered_intervals_and_variogram_are_globally_matched() -> None:
    payload = _payload()
    for arm in payload["units"][1]["predictions"]:  # type: ignore[index]
        arm["intervals"][0]["nominal_coverage"] = 0.6
    with pytest.raises(ValueError, match="same interval coverages"):
        parse_probabilistic_score_bundle(payload)

    payload = _payload()
    payload["score_configuration"]["variogram_pairs"] = np.empty(  # type: ignore[index]
        (0, 2),
        dtype=np.int64,
    )
    with pytest.raises(ValueError, match="must not be empty"):
        parse_probabilistic_score_bundle(payload)


def test_predictive_interval_constructor_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        scoring_module.PredictiveIntervalV1(
            nominal_coverage=1.0,
            lower=np.asarray([0.0]),
            upper=np.asarray([1.0]),
        )
    with pytest.raises(ValueError, match="bounds are inconsistent"):
        scoring_module.PredictiveIntervalV1(
            nominal_coverage=0.9,
            lower=np.asarray([1.0]),
            upper=np.asarray([0.0]),
        )


def test_optional_array_and_interval_identity_helpers() -> None:
    assert scoring_module._same_optional_array(None, None)
    assert not scoring_module._same_optional_array(None, np.asarray([1.0]))
    first = scoring_module.PredictiveIntervalV1(
        nominal_coverage=0.9,
        lower=np.asarray([0.0]),
        upper=np.asarray([1.0]),
    )
    second = scoring_module.PredictiveIntervalV1(
        nominal_coverage=0.5,
        lower=np.asarray([0.0]),
        upper=np.asarray([1.0]),
    )
    assert not scoring_module._same_intervals((first,), ())
    assert not scoring_module._same_intervals((first,), (second,))


def test_nonnegative_score_roundoff_and_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        scoring_module._nonnegative_score(
            -np.finfo(np.float64).eps,
            offset=0.0,
            name="roundoff",
        )
        == 0.0
    )
    with pytest.raises(ValueError, match="could not be shifted"):
        scoring_module._nonnegative_score(-1.0, offset=0.0, name="invalid")
    with pytest.raises(ValueError, match="could not be shifted"):
        scoring_module._nonnegative_score(float("inf"), offset=0.0, name="invalid")

    original_norm = scoring_module.np.linalg.norm

    def invalid_norm(*args: object, **kwargs: object) -> np.ndarray:
        result = np.asarray(original_norm(*args, **kwargs), dtype=np.float64)
        return np.full_like(result, np.nan)

    monkeypatch.setattr(scoring_module.np.linalg, "norm", invalid_norm)
    with pytest.raises(FloatingPointError, match="energy score became invalid"):
        energy_score([[0.0]], [0.0])
