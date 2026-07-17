from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from causal4d_public.deform360_sota_evaluation import (
    aggregate_deform360_panel,
    authorize_deform360_table4_claim,
    deform360_evaluator_contract_sha256,
    score_deform360_episode,
    validate_deform360_evaluator_contract,
)


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
        "schema_version": 1,
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
            "chamfer": {"definition": "symmetric_mean_euclidean_m"},
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
