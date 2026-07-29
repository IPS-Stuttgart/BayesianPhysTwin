from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_causal_response_adaptive_query import (
    AdaptiveCausalResponseQueryConfig,
    build_adaptive_causal_response_query_schedule,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_preflight import (
    evaluate_adaptive_direct_depth_source_preflight_v14,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_source_lock import (
    AdaptiveDirectDepthSourceCaseV14,
    build_adaptive_direct_depth_source_lock_v14,
    validate_adaptive_direct_depth_source_lock_v14,
    write_adaptive_direct_depth_source_lock_v14,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_synthetic import (
    run_adaptive_direct_depth_synthetic_v14,
    write_adaptive_direct_depth_synthetic_v14,
)
from bayesian_phystwin.deform360_causal_response_preflight import (
    REGISTERED_CAMERA_IDS,
    CausalResponseSourceCameraRecord,
)


def _carrier():
    camera_count = 8
    node_count = 20
    height = width = 96
    coordinate = np.linspace(-1.0, 1.0, node_count)
    frame_zero = np.column_stack(
        (
            0.18 * coordinate,
            0.04 * np.sin(np.pi * coordinate),
            np.full(node_count, 2.0),
        )
    )
    graph_basis = np.zeros((node_count, 3, 8))
    for mode in range(8):
        graph_basis[:, mode % 3, mode] = coordinate ** (mode % 4 + 1)
    intrinsics = np.repeat(np.eye(3)[None], camera_count, axis=0)
    intrinsics[:, 0, 0] = 60.0
    intrinsics[:, 1, 1] = 60.0
    intrinsics[:, 0, 2] = width / 2
    intrinsics[:, 1, 2] = height / 2
    return build_adaptive_causal_response_query_schedule(
        frame_zero,
        graph_basis,
        np.ones(node_count),
        intrinsics,
        np.repeat(np.eye(4)[None], camera_count, axis=0),
        np.full((camera_count, height, width), 2.0),
        np.ones((camera_count, height, width), dtype=bool),
        camera_ids=REGISTERED_CAMERA_IDS[:camera_count],
        config=AdaptiveCausalResponseQueryConfig(
            query_count=8,
            graph_basis_rank=8,
        ),
    )


def _records() -> tuple[CausalResponseSourceCameraRecord, ...]:
    return tuple(
        CausalResponseSourceCameraRecord(
            camera_id=camera,
            depth_frame_count=76 if index < 8 else 0,
            mask_frame_count=76 if index < 8 else 0,
            calibration_valid=index < 8,
            frame_zero_projected_support_count=20 if index < 8 else 0,
        )
        for index, camera in enumerate(REGISTERED_CAMERA_IDS)
    )


def _sources() -> dict[str, str]:
    sources = {
        "metadata": "1" * 64,
        "robot": "2" * 64,
        "physical_geometry": "3" * 64,
        "tactile": "4" * 64,
    }
    for camera in REGISTERED_CAMERA_IDS[:8]:
        sources[f"depth/{camera}"] = "5" * 64
        sources[f"mask/{camera}"] = "6" * 64
        sources[f"calibration/{camera}"] = "7" * 64
    return sources


def _panel():
    carrier = _carrier()
    cases = []
    preflights = []
    for index in range(12):
        object_id = f"unit-fresh-{index:02d}"
        preflight = evaluate_adaptive_direct_depth_source_preflight_v14(
            object_id=object_id,
            episode_id=0,
            category="cloth",
            bimanual_value="no",
            episode_frame_count=76,
            robot_frame_count=76,
            tactile_frame_count=76,
            physical_node_count=256,
            camera_records=_records(),
            carrier=carrier,
            source_sha256=_sources(),
        )
        preflights.append(preflight)
        cases.append(
            AdaptiveDirectDepthSourceCaseV14(
                case_id=f"{object_id}-ep0000",
                case_hash=preflight.case_hash,
                object_hash=preflight.object_hash,
                metadata_sha256=hashlib.sha256(object_id.encode()).hexdigest(),
                source_preflight_sha256=preflight.artifact_sha256,
                carrier_artifact_sha256=preflight.carrier_artifact_sha256,
                carrier_arm=preflight.carrier_arm,
                fold=index % 3,
            )
        )
    return tuple(cases), tuple(preflights)


def test_v14_source_lock_binds_exclusion_preflights_and_folds(
    tmp_path: Path,
) -> None:
    cases, preflights = _panel()
    root = Path(__file__).resolve().parents[1]
    exclusion = (
        root / "configs" / "sota" / "deform360_fresh_object_exclusion_v14.json"
    )
    synthetic = tmp_path / "synthetic.json"
    write_adaptive_direct_depth_synthetic_v14(
        synthetic,
        run_adaptive_direct_depth_synthetic_v14(),
    )
    lock = build_adaptive_direct_depth_source_lock_v14(
        cases,
        repository_revision="a" * 40,
        method_config_sha256="b" * 64,
        exclusion_manifest_path=exclusion,
        synthetic_control_result_path=synthetic,
        selection_metadata_sha256="c" * 64,
        source_preflights=preflights,
    )
    output = tmp_path / "source_lock.json"
    write_adaptive_direct_depth_source_lock_v14(output, lock)
    loaded = validate_adaptive_direct_depth_source_lock_v14(output)

    assert loaded == lock
    assert len(loaded.cases) == 12
    assert {case.fold for case in loaded.cases} == {0, 1, 2}
    assert loaded.descriptor()["information_boundary"][
        "identity_or_metric_outcome_read"
    ] is False


def test_v14_source_lock_rejects_an_excluded_object(tmp_path: Path) -> None:
    cases, preflights = _panel()
    root = Path(__file__).resolve().parents[1]
    exclusion = (
        root / "configs" / "sota" / "deform360_fresh_object_exclusion_v14.json"
    )
    synthetic = tmp_path / "synthetic.json"
    write_adaptive_direct_depth_synthetic_v14(
        synthetic,
        run_adaptive_direct_depth_synthetic_v14(),
    )
    excluded = json.loads(exclusion.read_text())["object_hashes"][0]
    replaced = list(cases)
    replaced[0] = AdaptiveDirectDepthSourceCaseV14(
        **{**cases[0].__dict__, "object_hash": excluded}
    )

    with pytest.raises(ValueError, match="excluded"):
        build_adaptive_direct_depth_source_lock_v14(
            replaced,
            repository_revision="a" * 40,
            method_config_sha256="b" * 64,
            exclusion_manifest_path=exclusion,
            synthetic_control_result_path=synthetic,
            selection_metadata_sha256="c" * 64,
            source_preflights=preflights,
        )
