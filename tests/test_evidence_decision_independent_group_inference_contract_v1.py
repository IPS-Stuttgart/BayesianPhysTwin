from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.independent_group_inference_v1 import (
    BOOTSTRAP_RNG,
    EFFECT_DIRECTION,
    GROUP_WEIGHTING,
    INDEPENDENT_GROUP_INFERENCE_SCHEMA,
    INDEPENDENT_GROUP_INFERENCE_VERSION,
    MAXIMUM_EXACT_GROUPS,
    RESAMPLING_UNIT,
    SIGN_FLIP_ASSUMPTION,
    IndependentGroupInferenceV1,
    analyze_independent_group_inference_v1,
    load_independent_group_inference_v1,
    save_independent_group_inference_v1,
)


def _result(
    effects: np.ndarray | None = None,
    *,
    group_ids: tuple[str, ...] = ("g3", "g1", "g2"),
    estimand_ids: tuple[str, ...] = ("candidate-vs-last", "candidate-vs-physics"),
    seed: int = 17,
    replicates: int = 256,
    metadata: dict[str, object] | None = None,
) -> IndependentGroupInferenceV1:
    if effects is None:
        effects = np.asarray(
            [
                [-1.0, -0.4],
                [-0.2, 0.3],
                [-0.8, -0.1],
            ]
        )
    return analyze_independent_group_inference_v1(
        protocol_id="protocol-sha256",
        family_id="two-contrast-family-v1",
        statistical_unit="complete physical object-session",
        within_group_aggregation="equal-horizon mean before group inference",
        group_ids=group_ids,
        estimand_ids=estimand_ids,
        group_effects=effects,
        bootstrap_replicates=replicates,
        bootstrap_seed=seed,
        metadata={} if metadata is None else metadata,
    )


def test_json_round_trip_replays_every_result(tmp_path: Path) -> None:
    result = _result(replicates=128)
    target = tmp_path / "inference.json"

    save_independent_group_inference_v1(result, target)
    loaded = load_independent_group_inference_v1(target)

    assert loaded.artifact_id == result.artifact_id
    assert loaded.to_payload() == result.to_payload()
    with pytest.raises(FileExistsError):
        save_independent_group_inference_v1(result, target)
    save_independent_group_inference_v1(result, target, overwrite=True)


def test_payload_tampering_fails_even_when_result_field_is_only_summary() -> None:
    result = _result(replicates=64)
    payload = result.to_payload()
    payload["observed_mean"][0] += 0.5

    with pytest.raises(ValueError, match="artifact_id|replay"):
        IndependentGroupInferenceV1.from_payload(payload)

    payload["artifact_id"] = content_id(
        {key: value for key, value in payload.items() if key != "artifact_id"}
    )
    with pytest.raises(ValueError, match="artifact_id|replay"):
        IndependentGroupInferenceV1.from_payload(payload)


def test_payload_requires_exact_schema_types() -> None:
    payload = _result(replicates=16).to_payload()
    payload["schema"] = 1
    with pytest.raises(ValueError, match="schema changed"):
        IndependentGroupInferenceV1.from_payload(payload)

    payload = _result(replicates=16).to_payload()
    payload["schema_version"] = True
    with pytest.raises(ValueError, match="schema version changed"):
        IndependentGroupInferenceV1.from_payload(payload)


def test_payload_rejects_unknown_and_changed_constant_fields() -> None:
    payload = _result(replicates=32).to_payload()
    payload["unknown"] = True
    with pytest.raises(ValueError, match="fields changed"):
        IndependentGroupInferenceV1.from_payload(payload)

    payload = _result(replicates=32).to_payload()
    payload["effect_direction"] = "positive_is_better"
    with pytest.raises(ValueError, match="effect_direction changed"):
        IndependentGroupInferenceV1.from_payload(payload)


def test_loader_rejects_duplicate_keys_and_nonfinite_json(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_independent_group_inference_v1(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_independent_group_inference_v1(nonfinite)


def test_contract_constants_are_persisted() -> None:
    payload = _result(replicates=32).to_payload()

    assert payload["schema"] == INDEPENDENT_GROUP_INFERENCE_SCHEMA
    assert payload["schema_version"] == INDEPENDENT_GROUP_INFERENCE_VERSION
    assert payload["effect_direction"] == EFFECT_DIRECTION
    assert payload["resampling_unit"] == RESAMPLING_UNIT
    assert payload["group_weighting"] == GROUP_WEIGHTING
    assert payload["sign_flip_assumption"] == SIGN_FLIP_ASSUMPTION
    assert payload["bootstrap_rng"] == BOOTSTRAP_RNG
    assert payload["maximum_exact_groups"] == MAXIMUM_EXACT_GROUPS


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"group_ids": ("g1",)}, "at least 2"),
        ({"group_ids": "g1,g2,g3"}, "sequence of exact strings"),
        ({"group_ids": ("g1", "g1", "g2")}, "duplicates"),
        ({"estimand_ids": ("same", "same")}, "duplicates"),
        ({"group_effects": [[True, False]] * 3}, "real numeric"),
        ({"group_effects": [[1.0, np.nan]] * 3}, "finite"),
        ({"group_effects": [[1.0], [2.0], [3.0]]}, "shape"),
        ({"confidence": 1.0}, "strictly inside"),
        ({"confidence": float("nan")}, "finite"),
        ({"confidence": True}, "finite real"),
        ({"bootstrap_replicates": 1.5}, "integer"),
        ({"bootstrap_replicates": True}, "integer"),
        ({"bootstrap_seed": -1}, "integer"),
        ({"metadata": {"bad": float("inf")}}, "finite JSON"),
    ],
)
def test_invalid_inputs_fail_before_resampling(
    kwargs: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "protocol_id": "protocol",
        "family_id": "family",
        "statistical_unit": "object-session",
        "within_group_aggregation": "mean",
        "group_ids": ("g1", "g2", "g3"),
        "estimand_ids": ("a", "b"),
        "group_effects": [[-1.0, -0.2], [-0.5, 0.1], [-0.3, -0.1]],
        "bootstrap_replicates": 8,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        IndependentGroupInferenceV1(**arguments)


def test_exact_enumeration_and_bootstrap_workload_limits_fail_closed() -> None:
    with pytest.raises(ValueError, match="at most"):
        IndependentGroupInferenceV1(
            protocol_id="protocol",
            family_id="family",
            statistical_unit="object-session",
            within_group_aggregation="mean",
            group_ids=tuple(f"g{index:02d}" for index in range(21)),
            estimand_ids=("a",),
            group_effects=np.zeros((21, 1)),
            bootstrap_replicates=1,
        )

    with pytest.raises(ValueError, match="result-value budget"):
        IndependentGroupInferenceV1(
            protocol_id="protocol",
            family_id="family",
            statistical_unit="object-session",
            within_group_aggregation="mean",
            group_ids=("g1", "g2"),
            estimand_ids=tuple(f"e{index:02d}" for index in range(64)),
            group_effects=np.zeros((2, 64)),
            bootstrap_replicates=200_000,
        )


def test_group_draw_budget_is_enforced_separately() -> None:
    with pytest.raises(ValueError, match="group-draw budget"):
        IndependentGroupInferenceV1(
            protocol_id="protocol",
            family_id="family",
            statistical_unit="object-session",
            within_group_aggregation="mean",
            group_ids=tuple(f"g{index:02d}" for index in range(20)),
            estimand_ids=("a",),
            group_effects=np.zeros((20, 1)),
            bootstrap_replicates=1_000_000,
        )


def test_save_rejects_the_wrong_result_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="IndependentGroupInferenceV1"):
        save_independent_group_inference_v1(  # type: ignore[arg-type]
            {"not": "an inference"},
            tmp_path / "result.json",
        )


def test_custom_string_subclasses_are_not_silently_admitted() -> None:
    class StringSubclass(str):
        pass

    with pytest.raises(ValueError, match="canonical string"):
        _result(group_ids=(StringSubclass("g1"), "g2", "g3"))


def test_save_requires_literal_overwrite_boolean(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    with pytest.raises(ValueError, match="literal Boolean"):
        save_independent_group_inference_v1(
            _result(replicates=16),
            target,
            overwrite=np.bool_(False),
        )


def test_artifact_id_is_lowercase_sha256_and_payload_is_json_finite() -> None:
    result = _result(replicates=32)
    artifact_id = result.artifact_id

    assert artifact_id is not None
    assert len(artifact_id) == 64
    assert set(artifact_id) <= set("0123456789abcdef")
    json.dumps(result.to_payload(), allow_nan=False)
