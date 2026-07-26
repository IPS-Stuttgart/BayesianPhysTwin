from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from causal4d_public.deform360_sota_evaluation import (
    aggregate_deform360_panel,
    authorize_deform360_table4_claim,
    build_development_evaluator_contract,
    build_released_processed_evaluator_contract,
    deform360_evaluator_contract_sha256,
    inspect_deform360_released_processed_episode,
    score_deform360_episode,
    score_deform360_released_processed_persistence,
    validate_deform360_evaluator_contract,
)
from causal4d_public.deform360_reusable_sota_protocol import (
    load_reusable_sota_config,
)
from causal4d_public.deform360_sota_processing import (
    DEVELOPMENT_OBSERVATIONS_KIND,
    PINNED_COTRACKER_CHECKPOINT_SHA256,
    PINNED_COTRACKER_REVISION,
    PINNED_DEFORM360_PROCESSING_REVISION,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/causal4d_public/deform360_reusable_sota_v1.json"


def _seal(payload: dict[str, object]) -> dict[str, object]:
    payload["result_sha256"] = deform360_evaluator_contract_sha256(payload)
    return payload


def _contract(
    *,
    status: str = "independent-protocol",
    panel: str = "object_balanced_mean",
    track: str = "mean_euclidean_m",
) -> dict[str, object]:
    held = {"object-a": [0, 1], "object-b": [0]}
    identities = {
        "object-a/0": "a" * 64,
        "object-a/1": "b" * 64,
        "object-b/0": "c" * 64,
    }
    payload: dict[str, object] = {
        "schema_version": 2,
        "artifact_kind": "Deform360EvaluatorContract",
        "contract_id": "fixture",
        "status": status,
        "dataset": {
            "repository": "fixture/deform360",
            "revision": "d" * 40,
            "coordinate_unit": "m",
        },
        "split": {
            "object_ids": ["object-a", "object-b"],
            "fit_episode_ids_by_object": {"object-a": [2], "object-b": [2]},
            "held_episode_ids_by_object": held,
        },
        "input_boundary": {
            "initial_object_frame_count": 1,
            "known_future_robot_action": True,
            "future_object_observation": False,
        },
        "temporal": {
            "evaluation_start_frame": 1,
            "evaluation_stop_frame_exclusive": 3,
            "frame_stride": 1,
        },
        "particles": {
            "identity_policy": "fixed_material_correspondence",
            "identity_sha256_by_episode": identities,
        },
        "metrics": {
            "chamfer": {
                "definition": "symmetric_mean_euclidean_m",
                "visibility_policy": "all_finite_material_points",
            },
            "track": {
                "definition": track,
                "visibility_policy": "all_finite_material_points",
            },
        },
        "aggregation": {
            "frame": "mean",
            "episode": "mean",
            "object": "mean",
            "panel": panel,
        },
        "published_reference": {
            "method": "ParticleFormer",
            "future_chamfer_m": 0.051,
            "future_track_error_m": 0.079,
        },
        "evaluator_provenance": {
            "released_by_deform360_authors": False,
        },
        "unresolved_fields": [],
    }
    return _seal(payload)


def _episode_score(
    contract: dict[str, object], object_id: str, episode_id: int, error: float
) -> dict[str, object]:
    target = np.zeros((3, 1, 3), dtype=np.float64)
    prediction = target.copy()
    prediction[1:, 0, 0] = error
    key = f"{object_id}/{episode_id}"
    identity = contract["particles"]["identity_sha256_by_episode"][key]
    return score_deform360_episode(
        contract,
        object_id=object_id,
        episode_id=episode_id,
        particle_identity_sha256=identity,
        target_m=target,
        prediction_m=prediction,
    )


def _write_released_processed_episode(root: Path) -> Path:
    episode = root / "episode_0"
    pcd = episode / "pcd_clean"
    pcd.mkdir(parents=True)
    metadata = {
        "fps": 30,
        "frame_num": 10,
        "start_frame": 2,
        "end_frame": 11,
        "cameras": ["camera-0", "camera-1"],
    }
    split = {
        "frame_len": 10,
        "train": [2, 10],
        "test": [10, 12],
    }
    (episode / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    (episode / "split.json").write_text(
        json.dumps(split),
        encoding="utf-8",
    )
    frame_zero = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    velocity = np.repeat(
        np.array([[0.0, 0.3, 0.0]], dtype=np.float32),
        len(frame_zero),
        axis=0,
    )
    for frame_index in range(12):
        points = frame_zero + frame_index * velocity / 30.0
        np.savez_compressed(
            pcd / f"{frame_index:06d}.npz",
            pts=points.astype(np.float32),
            colors=np.zeros_like(points, dtype=np.float32),
            vels=velocity.astype(np.float32),
            camera_indices=np.zeros(len(points), dtype=np.int32),
            visibility_matrix=np.ones((len(points), 2), dtype=np.uint8),
        )
    return episode


def test_episode_score_obeys_explicit_future_horizon_and_metric() -> None:
    contract = _contract()
    target = np.zeros((3, 1, 3), dtype=np.float64)
    prediction = target.copy()
    prediction[0, 0, 0] = 100.0
    prediction[1, 0, 0] = 0.1
    prediction[2, 0, 0] = 0.2

    result = score_deform360_episode(
        contract,
        object_id="object-a",
        episode_id=0,
        particle_identity_sha256="a" * 64,
        target_m=target,
        prediction_m=prediction,
    )

    assert result["evaluated_frame_indices"] == [1, 2]
    assert result["metrics"]["future_chamfer"] == pytest.approx(0.15)
    assert result["metrics"]["future_track_error"] == pytest.approx(0.15)


@pytest.mark.parametrize(
    ("definition", "take_square_root"),
    [
        ("symmetric_mean_euclidean_m", True),
        ("symmetric_mean_squared_euclidean_m2", False),
    ],
)
def test_chunked_chamfer_matches_explicit_pairwise_reference(
    definition: str,
    take_square_root: bool,
) -> None:
    contract = _contract()
    contract["metrics"]["chamfer"]["definition"] = definition
    _seal(contract)
    target_points = np.array(
        [[0.0, 0.0, 0.0], [0.8, -0.2, 0.1], [1.4, 0.4, -0.3]],
        dtype=np.float64,
    )
    prediction_points = np.array(
        [[0.1, 0.2, 0.0], [0.7, -0.4, 0.2], [1.8, 0.1, -0.1]],
        dtype=np.float64,
    )
    target = np.repeat(target_points[None, :, :], 3, axis=0)
    prediction = np.repeat(prediction_points[None, :, :], 3, axis=0)
    difference = target_points[:, None, :] - prediction_points[None, :, :]
    pairwise = np.sum(difference * difference, axis=2)
    if take_square_root:
        pairwise = np.sqrt(pairwise)
    expected = 0.5 * (
        float(np.mean(np.min(pairwise, axis=0)))
        + float(np.mean(np.min(pairwise, axis=1)))
    )

    score = score_deform360_episode(
        contract,
        object_id="object-a",
        episode_id=0,
        particle_identity_sha256="a" * 64,
        target_m=target,
        prediction_m=prediction,
    )

    assert score["metrics"]["future_chamfer"] == pytest.approx(expected)


def test_track_semantics_are_not_interchangeable() -> None:
    target = np.zeros((3, 2, 3), dtype=np.float64)
    prediction = target.copy()
    prediction[1:, 0, 0] = 3.0
    prediction[1:, 0, 1] = 4.0

    means = _contract(track="mean_euclidean_m")
    rms = _contract(track="root_mean_squared_euclidean_m")
    squared = _contract(track="mean_squared_euclidean_m2")

    values = []
    for contract in (means, rms, squared):
        values.append(
            score_deform360_episode(
                contract,
                object_id="object-a",
                episode_id=0,
                particle_identity_sha256="a" * 64,
                target_m=target,
                prediction_m=prediction,
            )["metrics"]["future_track_error"]
        )

    assert values == pytest.approx([2.5, np.sqrt(12.5), 12.5])


def test_chamfer_and_track_visibility_policies_are_independent() -> None:
    contract = _contract()
    contract["metrics"]["track"]["visibility_policy"] = (
        "visible_and_finite_material_points"
    )
    _seal(contract)
    target = np.zeros((3, 2, 3), dtype=np.float64)
    prediction = target.copy()
    prediction[1:, 1, 0] = 1.0
    visibility = np.ones((3, 2), dtype=bool)
    visibility[1:, 1] = False

    result = score_deform360_episode(
        contract,
        object_id="object-a",
        episode_id=0,
        particle_identity_sha256="a" * 64,
        target_m=target,
        prediction_m=prediction,
        visibility=visibility,
    )

    assert result["metrics"]["future_chamfer"] > 0.0
    assert result["metrics"]["future_track_error"] == 0.0
    assert result["valid_chamfer_particle_count_by_frame"] == [2, 2]
    assert result["valid_track_particle_count_by_frame"] == [1, 1]


def test_particle_identity_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="particle identity differs"):
        score_deform360_episode(
            _contract(),
            object_id="object-a",
            episode_id=0,
            particle_identity_sha256="f" * 64,
            target_m=np.zeros((3, 1, 3)),
            prediction_m=np.zeros((3, 1, 3)),
        )


def test_panel_aggregation_is_object_balanced() -> None:
    contract = _contract(panel="object_balanced_mean")
    scores = [
        _episode_score(contract, "object-a", 0, 1.0),
        _episode_score(contract, "object-a", 1, 3.0),
        _episode_score(contract, "object-b", 0, 10.0),
    ]

    result = aggregate_deform360_panel(contract, scores)

    assert result["metrics"]["by_object"]["object-a"]["future_track_error"] == 2.0
    assert result["metrics"]["future_track_error"] == 6.0


def test_unresolved_contract_refuses_direct_table4_claim() -> None:
    contract = _contract(status="unresolved-non-authorizing")
    contract["unresolved_fields"] = ["metrics.track.definition"]
    contract["metrics"]["track"]["definition"] = "unresolved"
    _seal(contract)
    validation = validate_deform360_evaluator_contract(contract)

    assert validation["official_table4_authorizing"] is False
    with pytest.raises(ValueError, match="evaluator parity is not established"):
        authorize_deform360_table4_claim(
            contract,
            {
                "artifact_kind": "Deform360PanelScore",
                "result_sha256": "0" * 64,
            },
        )


def test_official_contract_requires_reference_reproduction() -> None:
    contract = _contract(status="official-parity")
    contract["evaluator_provenance"] = {
        "released_by_deform360_authors": True,
        "source_revision_sha256": "1" * 64,
        "entrypoint_sha256": "2" * 64,
        "particleformer_table4_reproduction": {
            "passed": False,
            "future_chamfer_m": 0.051,
            "future_track_error_m": 0.079,
        },
    }
    _seal(contract)

    with pytest.raises(ValueError, match="has not reproduced"):
        validate_deform360_evaluator_contract(contract)


def test_authorization_needs_both_metrics_below_reference() -> None:
    contract = _contract(status="official-parity")
    contract["evaluator_provenance"] = {
        "released_by_deform360_authors": True,
        "source_revision_sha256": "1" * 64,
        "entrypoint_sha256": "2" * 64,
        "particleformer_table4_reproduction": {
            "passed": True,
            "future_chamfer_m": 0.051,
            "future_track_error_m": 0.079,
        },
    }
    _seal(contract)
    scores = [
        _episode_score(contract, "object-a", 0, 0.040),
        _episode_score(contract, "object-a", 1, 0.040),
        _episode_score(contract, "object-b", 0, 0.040),
    ]
    panel = aggregate_deform360_panel(contract, scores)

    decision = authorize_deform360_table4_claim(contract, panel)

    assert decision["authorized"] is True
    worse = deepcopy(panel)
    worse["metrics"]["future_track_error"] = 0.080
    worse["result_sha256"] = deform360_evaluator_contract_sha256(worse)
    decision = authorize_deform360_table4_claim(contract, worse)
    assert decision["authorized"] is False
    assert decision["gates"]["future_chamfer_below_particleformer"] is True
    assert decision["gates"]["future_track_below_particleformer"] is False


def test_development_contract_is_explicit_but_never_table4_authorizing() -> None:
    protocol = load_reusable_sota_config(PROTOCOL)
    manifest = {
        "schema_version": 1,
        "artifact_kind": DEVELOPMENT_OBSERVATIONS_KIND,
        "authorization": {
            "protocol_id": "deform360-reusable-sota-v1",
            "protocol_config_sha256": protocol["config_sha256"],
            "object_id": "004-rubber-band",
            "episode_id": 0,
            "role": "held-development",
            "development_only": True,
            "confirmatory_object_opened": False,
        },
        "object_id": "004-rubber-band",
        "episode_id": 0,
        "role": "held-development",
        "camera_count": 3,
        "frame_count": 20,
        "point_frame_count": 15,
        "material_point_count": 2,
        "material_identity_sha256": "f" * 64,
        "implementation_revision": {
            "deform360_processing": PINNED_DEFORM360_PROCESSING_REVISION,
            "cotracker": PINNED_COTRACKER_REVISION,
        },
        "input_sha256": {
            "cotracker_checkpoint": PINNED_COTRACKER_CHECKPOINT_SHA256,
        },
        "information_boundary": {
            "development_only": True,
            "prediction_metric_computed": False,
            "confirmatory_object_opened": False,
            "pokeflex_target_opened": False,
        },
    }
    _seal(manifest)

    contract = build_development_evaluator_contract(
        protocol,
        [manifest],
        evaluation_start_frame=1,
        evaluation_stop_frame_exclusive=12,
    )

    assert contract["status"] == "independent-protocol"
    assert contract["split"]["held_episode_ids_by_object"] == {"004-rubber-band": [0]}
    assert contract["particles"]["identity_sha256_by_episode"] == {
        "004-rubber-band/0": "f" * 64
    }
    validation = validate_deform360_evaluator_contract(contract)
    assert validation["official_table4_authorizing"] is False
    with pytest.raises(ValueError, match="evaluator parity is not established"):
        authorize_deform360_table4_claim(contract, {})

    altered = deepcopy(manifest)
    altered["authorization"]["role"] = "fit"
    _seal(altered)
    with pytest.raises(ValueError, match="held-development"):
        build_development_evaluator_contract(
            protocol,
            [altered],
            evaluation_start_frame=1,
            evaluation_stop_frame_exclusive=12,
        )


def test_author_released_processed_episode_preserves_identity_and_split(
    tmp_path: Path,
) -> None:
    episode = _write_released_processed_episode(tmp_path)

    manifest = inspect_deform360_released_processed_episode(
        episode,
        object_id="001-rope",
        episode_id=0,
        dataset_revision="9" * 40,
    )
    contract = build_released_processed_evaluator_contract([manifest])
    score = score_deform360_released_processed_persistence(
        contract,
        episode_dir=episode,
        episode_manifest=manifest,
    )

    assert manifest["split"]["train_source_frames"] == [2, 10]
    assert manifest["split"]["test_source_frames"] == [10, 12]
    assert manifest["particles"]["released_trajectory_source_frames"] == [0, 12]
    assert manifest["particles"]["ordered_advection_check"]["passed"] is True
    assert contract["status"] == "independent-protocol"
    assert contract["temporal"]["evaluation_frame_indices_by_episode"] == {
        "001-rope/0": [0, 1]
    }
    assert score["evaluated_frame_indices"] == [0, 1]
    assert score["evaluated_source_frame_indices"] == [10, 11]
    assert score["metrics"]["future_track_error"] == pytest.approx(0.015)
    assert score["metrics"]["future_chamfer"] == pytest.approx(0.015)
    with pytest.raises(ValueError, match="evaluator parity is not established"):
        authorize_deform360_table4_claim(contract, {})


def test_author_released_processed_episode_rejects_identity_reordering(
    tmp_path: Path,
) -> None:
    episode = _write_released_processed_episode(tmp_path)
    path = episode / "pcd_clean" / "000011.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["pts"] = arrays["pts"][::-1].copy()
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="ordered velocity advection"):
        inspect_deform360_released_processed_episode(
            episode,
            object_id="001-rope",
            episode_id=0,
            dataset_revision="9" * 40,
        )
