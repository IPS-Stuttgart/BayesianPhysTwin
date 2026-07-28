from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_dynamic_tapnextpp_physical import (
    GRAPH_BASIS_RANK,
    PHYSICAL_ARCHIVE_FILENAME,
    build_readout_graph_basis,
    validate_dynamic_physical_artifacts,
    write_dynamic_physical_artifacts,
)


def _graph(point_count: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.column_stack(
        (
            np.linspace(0.0, 0.08, point_count),
            np.zeros(point_count),
            np.zeros(point_count),
        )
    )
    springs = np.column_stack(
        (
            np.arange(point_count - 1),
            np.arange(1, point_count),
        )
    )
    weights = np.eye(point_count)
    return vertices, springs, weights


def _arrays() -> dict[str, np.ndarray]:
    vertices, springs, weights = _graph()
    frame_zero = vertices.astype(np.float32)
    persistence = np.repeat(frame_zero[None], 76, axis=0)
    displacement = np.linspace(0.0, 0.02, 76, dtype=np.float32)
    driven = persistence.copy()
    driven[:, :, 1] += displacement[:, None]
    physical = persistence.copy()
    physical[:, :, 1] += 0.9 * displacement[:, None]
    return {
        "action_support": np.ones(len(frame_zero), dtype=np.float32),
        "driven_readout_m": driven,
        "frame_zero_points_m": frame_zero,
        "graph_basis": build_readout_graph_basis(
            vertices,
            springs,
            weights,
        ).astype(np.float32),
        "persistence_prediction_m": persistence,
        "physical_prediction_m": physical,
        "zero_action_readout_m": persistence.copy(),
    }


def test_readout_graph_basis_is_deterministic_orthonormal_and_vector_valued() -> None:
    vertices, springs, weights = _graph()

    first = build_readout_graph_basis(vertices, springs, weights)
    second = build_readout_graph_basis(vertices, springs, weights)

    assert first.shape == (len(vertices), 3, GRAPH_BASIS_RANK)
    np.testing.assert_array_equal(first, second)
    flat = first.reshape(-1, GRAPH_BASIS_RANK)
    np.testing.assert_allclose(flat.T @ flat, np.eye(GRAPH_BASIS_RANK), atol=1e-10)
    assert np.count_nonzero(first[:, 1:, 0]) == 0
    assert np.count_nonzero(first[:, (0, 2), 1]) == 0
    assert np.count_nonzero(first[:, :2, 2]) == 0


def test_dynamic_physical_artifact_round_trip_and_tamper_rejection(
    tmp_path: Path,
) -> None:
    protocol = tmp_path / "protocol.json"
    cohort = tmp_path / "cohort.json"
    admission = tmp_path / "admission.json"
    protocol.write_text('{"protocol_id":"test"}\n', encoding="utf-8")
    cohort.write_text('{"cohort":"test"}\n', encoding="utf-8")
    admission.write_text('{"admission":"test"}\n', encoding="utf-8")
    output = tmp_path / "physical"

    manifest = write_dynamic_physical_artifacts(
        output,
        _arrays(),
        protocol_path=protocol,
        cohort_lock_path=cohort,
        case_record={
            "queue_rank": 1,
            "case": "001-test-ep0000",
            "object_id": "001-test",
            "episode_id": 0,
            "category": "compact",
            "object_hash": "b" * 64,
            "case_hash": "c" * 64,
            "admission_sha256": "d" * 64,
        },
        partition="source",
        physical_mode="warp_twin",
        code_revision="a" * 40,
        input_files={"source_admission": admission},
        runtime_provenance={"runtime": "test"},
    )
    loaded_manifest, loaded_arrays = validate_dynamic_physical_artifacts(output)

    assert loaded_manifest == manifest
    assert loaded_manifest["physical_admitted"] is True
    for name, values in _arrays().items():
        np.testing.assert_array_equal(loaded_arrays[name], values)

    archive = output / PHYSICAL_ARCHIVE_FILENAME
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="archive checksum changed"):
        validate_dynamic_physical_artifacts(output)


def test_dynamic_physical_artifact_rejects_frame_zero_drift(tmp_path: Path) -> None:
    arrays = _arrays()
    arrays["driven_readout_m"] = arrays["driven_readout_m"].copy()
    arrays["driven_readout_m"][0, 0, 0] += 0.001
    protocol = tmp_path / "protocol.json"
    cohort = tmp_path / "cohort.json"
    protocol.write_text("{}\n", encoding="utf-8")
    cohort.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frame-zero material identities"):
        write_dynamic_physical_artifacts(
            tmp_path / "physical",
            arrays,
            protocol_path=protocol,
            cohort_lock_path=cohort,
            case_record={},
            partition="source",
            physical_mode="warp_twin",
            code_revision="a" * 40,
            input_files={},
            runtime_provenance={},
        )
