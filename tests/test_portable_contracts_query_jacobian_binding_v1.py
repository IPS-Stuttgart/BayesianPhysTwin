from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.query_jacobian_binding_v1 import (
    QUERY_JACOBIAN_BINDING_CLAIM_BOUNDARY,
    QueryJacobianBindingV1,
    build_query_jacobian_binding,
    load_query_jacobian_binding,
    write_query_jacobian_binding,
)


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _binding() -> tuple[QueryJacobianBindingV1, np.ndarray, tuple[str, ...]]:
    jacobian = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, -1.0, 0.0]],
        ],
        dtype=np.float64,
    )
    rows = ("factor-a/point-0", "factor-b/point-4")
    binding = build_query_jacobian_binding(
        query_name="endpoint-displacement",
        component_order=("x", "z"),
        physical_unit="m",
        coordinate_frame="registered-world",
        source_observation_artifact_id=_sha256("observation"),
        provider_manifest_id=_sha256("provider"),
        causal_frame_stop=18,
        query_jacobian=jacobian,
        row_ids=rows,
        metadata={"target_opened": False},
    )
    return binding, jacobian, rows


def test_binding_validates_exact_jacobian_bytes_and_row_order() -> None:
    binding, jacobian, rows = _binding()

    validated = binding.validate_payload(jacobian, rows)

    np.testing.assert_array_equal(validated, jacobian)
    assert not validated.flags.writeable
    assert binding.query_dimension == 2
    assert binding.observation_count == 2
    assert binding.descriptor()["target_outcomes_used"] is False
    assert binding.descriptor()["future_frames_used"] is False
    assert binding.descriptor()["claim_boundary"] == (
        QUERY_JACOBIAN_BINDING_CLAIM_BOUNDARY
    )


def test_binding_rejects_changed_bytes_and_reordered_rows() -> None:
    binding, jacobian, rows = _binding()
    changed = jacobian.copy()
    changed[0, 0, 0] += 1e-12

    with pytest.raises(ValueError, match="bytes differ"):
        binding.validate_payload(changed, rows)
    with pytest.raises(ValueError, match="row_ids differ"):
        binding.validate_payload(jacobian, tuple(reversed(rows)))


def test_binding_identity_changes_with_jacobian_or_row_roster() -> None:
    binding, jacobian, rows = _binding()
    changed = jacobian.copy()
    changed[0, 0, 0] = 2.0

    changed_jacobian = build_query_jacobian_binding(
        query_name=binding.query_name,
        component_order=binding.component_order,
        physical_unit=binding.physical_unit,
        coordinate_frame=binding.coordinate_frame,
        source_observation_artifact_id=binding.source_observation_artifact_id,
        provider_manifest_id=binding.provider_manifest_id,
        causal_frame_stop=binding.causal_frame_stop,
        query_jacobian=changed,
        row_ids=rows,
    )
    changed_rows = build_query_jacobian_binding(
        query_name=binding.query_name,
        component_order=binding.component_order,
        physical_unit=binding.physical_unit,
        coordinate_frame=binding.coordinate_frame,
        source_observation_artifact_id=binding.source_observation_artifact_id,
        provider_manifest_id=binding.provider_manifest_id,
        causal_frame_stop=binding.causal_frame_stop,
        query_jacobian=jacobian,
        row_ids=("factor-a/point-0", "factor-b/point-5"),
    )

    assert changed_jacobian.artifact_id != binding.artifact_id
    assert changed_rows.artifact_id != binding.artifact_id


def test_binding_roundtrip_and_atomic_no_clobber(tmp_path: Path) -> None:
    binding, _, _ = _binding()
    path = tmp_path / "query-jacobian-binding.json"

    write_query_jacobian_binding(binding, path)
    loaded = load_query_jacobian_binding(path)

    assert loaded.to_record() == binding.to_record()
    with pytest.raises(FileExistsError):
        write_query_jacobian_binding(binding, path)


def test_binding_rejects_tampering_and_coercive_boolean_aliases() -> None:
    binding, _, _ = _binding()
    record = binding.to_record()
    record["target_outcomes_used"] = 0
    with pytest.raises(ValueError, match="target blind"):
        QueryJacobianBindingV1.from_mapping(record)

    record = binding.to_record()
    record["artifact_id"] = _sha256("different")
    with pytest.raises(ValueError, match="artifact_id"):
        QueryJacobianBindingV1.from_mapping(record)

    with pytest.raises(ValueError, match="unique"):
        build_query_jacobian_binding(
            query_name="endpoint-displacement",
            component_order=("x", "z"),
            physical_unit="m",
            coordinate_frame="registered-world",
            source_observation_artifact_id=_sha256("observation"),
            provider_manifest_id=_sha256("provider"),
            causal_frame_stop=18,
            query_jacobian=np.ones((2, 2, 3)),
            row_ids=("same", "same"),
        )


def test_direct_construction_revalidates_shape_and_content_identity() -> None:
    binding, _, _ = _binding()

    with pytest.raises(ValueError, match="component_order length"):
        replace(binding, component_order=("x",), artifact_id=None)
    with pytest.raises(ValueError, match="artifact_id"):
        replace(binding, artifact_id=_sha256("tampered"))
