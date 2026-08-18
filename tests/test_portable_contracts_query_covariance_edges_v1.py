from __future__ import annotations

import hashlib
import runpy
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import bayesian_phystwin.bound_query_covariance_decision_v1 as bound_module
import bayesian_phystwin.query_jacobian_binding_v1 as binding_module
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.bound_query_covariance_decision_v1 import (
    BOUND_QUERY_COVARIANCE_PROJECTION_CLAIM_BOUNDARY,
    BOUND_QUERY_COVARIANCE_PROJECTION_SCHEMA,
    BOUND_QUERY_COVARIANCE_PROJECTION_VERSION,
    compose_bound_query_covariance_treatment,
    validate_bound_query_covariance_projection,
)
from bayesian_phystwin.query_covariance_decision_v1 import (
    PROB4D_QUERY_COVARIANCE_SCHEMA,
)
from bayesian_phystwin.query_jacobian_binding_v1 import (
    QUERY_JACOBIAN_BINDING_CLAIM_BOUNDARY,
    QUERY_JACOBIAN_BINDING_SCHEMA,
    QUERY_JACOBIAN_BINDING_VERSION,
    QueryJacobianBindingV1,
    build_query_jacobian_binding,
    write_query_jacobian_binding,
)

_HELPERS = runpy.run_path(
    str(
        Path(__file__).with_name(
            "test_portable_contracts_bound_query_covariance_decision_v1.py"
        )
    )
)
_binding = cast(Callable[[], Any], _HELPERS["_binding"])
_query = cast(Callable[[Any], Any], _HELPERS["_query"])
_certificate = cast(Callable[[Any], Any], _HELPERS["_certificate"])
_bound_projection = cast(
    Callable[[Any], dict[str, object]], _HELPERS["_bound_projection"]
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _minimal_binding() -> QueryJacobianBindingV1:
    return build_query_jacobian_binding(
        query_name="endpoint",
        component_order=("x",),
        physical_unit="m",
        coordinate_frame="world",
        source_observation_artifact_id=_sha("observation"),
        provider_manifest_id=_sha("provider"),
        causal_frame_stop=1,
        query_jacobian=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        row_ids=("row-0",),
    )


def test_query_binding_private_validators_reject_malformed_values() -> None:
    with pytest.raises(ValueError):
        binding_module._mapping([], name="value")
    with pytest.raises(ValueError):
        binding_module._mapping({1: "value"}, name="value")
    with pytest.raises(ValueError):
        binding_module._text(" ", name="value")
    with pytest.raises(ValueError):
        binding_module._integer(True, name="value")
    with pytest.raises(ValueError):
        binding_module._component_order("x")
    with pytest.raises(ValueError):
        binding_module._component_order([])
    with pytest.raises(ValueError):
        binding_module._component_order(["x", "x"])


def test_query_jacobian_canonicalization_covers_scalar_and_invalid_shapes() -> None:
    scalar = binding_module._canonical_jacobian(
        np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    )
    assert scalar.shape == (1, 1, 3)
    assert not scalar.flags.writeable

    with pytest.raises(ValueError):
        binding_module._canonical_jacobian(np.ones((2, 2), dtype=np.float64))
    with pytest.raises(ValueError):
        binding_module._canonical_jacobian(np.ones((1, 1, 2), dtype=np.float64))
    with pytest.raises(ValueError):
        binding_module._canonical_jacobian(np.ones((1, 0, 3), dtype=np.float64))
    nonfinite = np.ones((1, 1, 3), dtype=np.float64)
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        binding_module._canonical_jacobian(nonfinite)


def test_query_row_and_descriptor_validation_fail_closed() -> None:
    with pytest.raises(ValueError):
        binding_module._row_ids("row", expected_count=1)
    with pytest.raises(ValueError):
        binding_module._row_ids(("row",), expected_count=2)
    with pytest.raises(ValueError):
        binding_module._row_ids(("row", "row"), expected_count=2)

    descriptor: dict[str, object] = {
        "dtype": "<f8",
        "shape": [1, 1, 3],
        "sha256": _sha("jacobian"),
    }
    assert binding_module._validated_array_descriptor(descriptor)[0] == (1, 1, 3)
    with pytest.raises(ValueError):
        binding_module._validated_array_descriptor({**descriptor, "dtype": ">f8"})
    with pytest.raises(ValueError):
        binding_module._validated_array_descriptor({**descriptor, "shape": "1,1,3"})
    with pytest.raises(ValueError):
        binding_module._validated_array_descriptor({**descriptor, "shape": [1, 1, 2]})
    with pytest.raises(ValueError):
        binding_module._validated_array_descriptor({**descriptor, "shape": [1, 0, 3]})

    rows: dict[str, object] = {
        "schema": binding_module.OBSERVATION_ROW_BINDING_SCHEMA,
        "schema_version": binding_module.OBSERVATION_ROW_BINDING_VERSION,
        "count": 1,
        "sha256": _sha("rows"),
    }
    assert binding_module._validated_row_binding(rows)[0] == 1
    with pytest.raises(ValueError):
        binding_module._validated_row_binding({**rows, "schema": "wrong"})
    with pytest.raises(ValueError):
        binding_module._validated_row_binding({**rows, "schema_version": 2})


def test_query_binding_constructor_and_payload_edge_contracts(tmp_path: Path) -> None:
    binding = _minimal_binding()

    with pytest.raises(ValueError):
        replace(binding, query_name="", artifact_id=None)
    with pytest.raises(ValueError):
        replace(binding, query_jacobian_shape=[1, 1, 3], artifact_id=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        replace(binding, query_jacobian_shape=(1, 1, 2), artifact_id=None)
    with pytest.raises(ValueError):
        replace(binding, observation_count=2, artifact_id=None)
    with pytest.raises(ValueError):
        replace(binding, causal_frame_stop=0, artifact_id=None)
    with pytest.raises(ValueError):
        binding.validate_payload(
            np.ones((1, 2, 3), dtype=np.float64), ("row-0", "row-1")
        )
    with pytest.raises(TypeError):
        write_query_jacobian_binding(object(), tmp_path / "invalid.json")  # type: ignore[arg-type]


def test_query_binding_record_schema_and_policy_tampering_fail_closed() -> None:
    binding = _minimal_binding()
    record = binding.to_record()

    cases = [
        {**record, "schema": "wrong"},
        {**record, "schema_version": QUERY_JACOBIAN_BINDING_VERSION + 1},
        {**record, "future_frames_used": True},
        {**record, "claim_boundary": "wrong"},
    ]
    for tampered in cases:
        with pytest.raises(ValueError):
            QueryJacobianBindingV1.from_mapping(tampered)

    missing = dict(record)
    missing.pop("metadata")
    with pytest.raises(ValueError):
        QueryJacobianBindingV1.from_mapping(missing)
    assert record["schema"] == QUERY_JACOBIAN_BINDING_SCHEMA
    assert record["claim_boundary"] == QUERY_JACOBIAN_BINDING_CLAIM_BOUNDARY


def test_bound_projection_private_descriptor_validation_edges() -> None:
    with pytest.raises(ValueError):
        bound_module._mapping([], name="value")
    with pytest.raises(ValueError):
        bound_module._mapping({1: "value"}, name="value")
    with pytest.raises(ValueError):
        bound_module._integer(True, name="value")

    descriptor: dict[str, object] = {
        "dtype": "<f8",
        "shape": [2, 3, 3],
        "sha256": _sha("array"),
    }
    local = bound_module._array_descriptor(
        descriptor,
        name="local",
        observation_count=2,
        covariance=True,
    )
    assert local["shape"] == [2, 3, 3]
    factor = bound_module._array_descriptor(
        {**descriptor, "shape": [2, 3, 0]},
        name="factor",
        observation_count=2,
        covariance=False,
    )
    assert factor["shape"] == [2, 3, 0]

    invalid_descriptors = [
        {**descriptor, "dtype": ">f8"},
        {**descriptor, "shape": "2,3,3"},
        {**descriptor, "shape": [3, 3, 3]},
        {**descriptor, "shape": [2, 3, 2]},
    ]
    for invalid in invalid_descriptors:
        with pytest.raises(ValueError):
            bound_module._array_descriptor(
                invalid,
                name="local",
                observation_count=2,
                covariance=True,
            )


def test_bound_projection_rejects_schema_lineage_and_summary_substitutions() -> None:
    binding = _binding()
    projection = _bound_projection(binding)

    with pytest.raises(TypeError):
        validate_bound_query_covariance_projection(projection, binding=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validate_bound_query_covariance_projection([], binding=binding)

    cases: list[dict[str, object]] = [
        {**projection, "schema": "wrong"},
        {**projection, "schema_version": BOUND_QUERY_COVARIANCE_PROJECTION_VERSION + 1},
        {**projection, "claim_boundary": "wrong"},
        {**projection, "query_jacobian_binding_id": _sha("other-binding")},
        {**projection, "source_observation_artifact_id": _sha("other-observation")},
        {**projection, "provider_manifest_id": _sha("other-provider")},
        {**projection, "query_jacobian_sha256": _sha("other-jacobian")},
    ]
    for tampered in cases:
        with pytest.raises(ValueError):
            validate_bound_query_covariance_projection(tampered, binding=binding)

    wrong_summary = {**cast(dict[str, object], projection["projection_summary"])}
    wrong_summary["schema"] = "wrong"
    with pytest.raises(ValueError):
        validate_bound_query_covariance_projection(
            {**projection, "projection_summary": wrong_summary},
            binding=binding,
        )

    wrong_dimension = {**cast(dict[str, object], projection["projection_summary"])}
    wrong_dimension["query_dimension"] = binding.query_dimension + 1
    with pytest.raises(ValueError):
        validate_bound_query_covariance_projection(
            {**projection, "projection_summary": wrong_dimension},
            binding=binding,
        )

    assert projection["schema"] == BOUND_QUERY_COVARIANCE_PROJECTION_SCHEMA
    assert (
        projection["claim_boundary"] == BOUND_QUERY_COVARIANCE_PROJECTION_CLAIM_BOUNDARY
    )


def test_bound_projection_rejects_covariance_descriptor_tampering() -> None:
    binding = _binding()
    projection = _bound_projection(binding)
    local = cast(dict[str, object], projection["local_covariance_m2"])
    factor = cast(dict[str, object], projection["low_rank_factor_m"])

    with pytest.raises(ValueError):
        validate_bound_query_covariance_projection(
            {**projection, "local_covariance_m2": {**local, "shape": [2, 3, 2]}},
            binding=binding,
        )
    with pytest.raises(ValueError):
        validate_bound_query_covariance_projection(
            {**projection, "low_rank_factor_m": {**factor, "shape": [3, 3, 1]}},
            binding=binding,
        )


def test_bound_composition_rejects_all_physical_query_semantic_substitutions() -> None:
    binding = _binding()
    query = _query(binding)
    certificate = _certificate(query)
    projection = _bound_projection(binding)

    with pytest.raises(TypeError):
        compose_bound_query_covariance_treatment(
            object(),  # type: ignore[arg-type]
            binding,
            projection,
            certificate,
        )
    with pytest.raises(TypeError):
        compose_bound_query_covariance_treatment(
            query,
            object(),  # type: ignore[arg-type]
            projection,
            certificate,
        )

    substitutions = [
        {"query_name": "other-query"},
        {"component_order": ("other-a", "other-b")},
        {"coordinate_frame": "other-frame"},
        {"provider_manifest_id": _sha("other-provider")},
    ]
    for changes in substitutions:
        altered = replace(query, **changes, query_id=None)
        with pytest.raises(ValueError):
            compose_bound_query_covariance_treatment(
                altered,
                binding,
                projection,
                certificate,
            )

    altered_unit = replace(
        query,
        physical_unit="cm",
        decision_margins=replace(query.decision_margins, width_unit="cm"),
        query_id=None,
    )
    with pytest.raises(ValueError):
        compose_bound_query_covariance_treatment(
            altered_unit,
            binding,
            projection,
            certificate,
        )


def test_bound_projection_validated_record_remains_content_addressed() -> None:
    binding = _binding()
    projection = _bound_projection(binding)
    validated = validate_bound_query_covariance_projection(projection, binding=binding)

    unsigned = dict(projection)
    unsigned.pop("artifact_id")
    assert validated.artifact_id == content_id(unsigned)
    assert validated.projection_summary["schema"] == PROB4D_QUERY_COVARIANCE_SCHEMA
