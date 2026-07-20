import json
from pathlib import Path
import pickle

import numpy as np
import pytest

import bayesian_phystwin.deform360_query_field_development as development
from bayesian_phystwin.deform360_online_belief_evaluation import (
    PROTOCOL_ID as SOURCE_PROTOCOL_ID,
)


def _synthetic_trajectories(
    *, frames: int = 76, identities: int = 280
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    frame_zero = rng.uniform(-0.1, 0.1, size=(identities, 3)).astype(np.float32)
    comparator = np.repeat(frame_zero[None], frames, axis=0)
    primary = comparator.copy()
    target = comparator.copy()
    local = frame_zero[:, [1, 2, 0]]
    for frame in range(1, frames):
        global_delta = np.asarray(
            [0.0007 * frame, -0.0003 * frame, 0.0001 * frame],
            dtype=np.float32,
        )
        comparator[frame] += global_delta
        primary[frame] += global_delta + np.float32(0.0002 * frame) * local
        target[frame] = primary[frame]
    visible = np.ones((frames, identities), dtype=bool)
    valid = np.ones_like(visible)
    return primary, comparator, target, visible, valid


def test_candidate_grid_is_exact_and_nearest_wins_a_tolerance_tie() -> None:
    candidates = development._candidate_grid()
    assert len(candidates) == 10
    assert candidates[0].descriptor() == {
        "candidate_id": "nearest-v1",
        "operator_id": "nearest-v1",
        "neighbor_count": 1,
        "length_scale_fraction": 0.0,
        "support_radius_fraction": 0.5,
    }
    assert {
        (value.neighbor_count, value.length_scale_fraction) for value in candidates[1:]
    } == {(count, fraction) for count in (4, 8, 12) for fraction in (0.05, 0.10, 0.20)}
    by_id = {value.candidate_id: value for value in candidates}
    rows = [
        {
            "candidate_id": candidates[1].candidate_id,
            "selection_objective_m": 1.0,
        },
        {
            "candidate_id": candidates[0].candidate_id,
            "selection_objective_m": 1.0 + 0.5e-12,
        },
    ]

    ranking = development._rank_with_tolerance(
        rows,
        value_key="selection_objective_m",
        candidates=by_id,
    )

    assert ranking[0]["candidate_id"] == "nearest-v1"


@pytest.mark.parametrize("anchor_count", development.ANCHOR_COUNTS)
def test_case_evaluator_permanently_hides_centers_and_anchors(
    anchor_count: int,
) -> None:
    primary, comparator, target, visible, valid = _synthetic_trajectories()
    centers = np.arange(16, dtype=np.int64)
    candidate = development._candidate_grid()[4]

    report, arrays = development.evaluate_query_field_case_arrays(
        primary,
        comparator,
        target,
        visible,
        valid,
        centers,
        anchor_count=anchor_count,
        candidate=candidate,
        scored_frames=tuple(range(20, 76)),
    )

    assert len(arrays["anchor_ids"]) == anchor_count
    assert len(arrays["query_ids"]) == len(target[0]) - len(centers) - anchor_count
    assert not np.any(np.isin(arrays["anchor_ids"], centers))
    assert not np.any(np.isin(arrays["query_ids"], centers))
    assert not np.any(np.isin(arrays["query_ids"], arrays["anchor_ids"]))
    assert report["target_scores"]["primary"]["identity_rmse_m"] < 2e-3
    assert report["geometry"]["supported_query_fraction"] == 1.0
    assert report["field_native_fidelity"]["shared_mask"].endswith(
        "no future target value or mask"
    )


def test_case_evaluator_uses_one_shared_future_target_mask() -> None:
    primary, comparator, target, visible, valid = _synthetic_trajectories()
    centers = np.arange(16, dtype=np.int64)
    visible[20, 20:40] = False
    valid[21, 40:60] = False

    report, _ = development.evaluate_query_field_case_arrays(
        primary,
        comparator,
        target,
        visible,
        valid,
        centers,
        anchor_count=64,
        candidate=development._candidate_grid()[0],
        scored_frames=(20, 21),
    )

    assert (
        report["target_scores"]["primary"]["scored_identity_count_per_frame"]
        == report["target_scores"]["comparator"]["scored_identity_count_per_frame"]
    )


def _write(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return development._sha256(path)


@pytest.mark.parametrize("archive_name", ["prediction.npz", "sealed_prediction.npz"])
def test_audited_case_loader_revalidates_source_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archive_name: str
) -> None:
    case = "002-rope-silk-ep0002"
    source = tmp_path / "independent-source-v1"
    episode = source / case
    run = tmp_path / "audited-run"
    episode.mkdir(parents=True)
    run.mkdir()
    primary, comparator, target, visibility, validity = _synthetic_trajectories()
    centers = np.arange(16, dtype=np.int64)
    roles = {
        "prediction_seal": episode / "prediction_seal.json",
        "prediction_archive": episode / archive_name,
        "target_data": episode / "target_data.pkl",
        "outcome": episode / "outcome.json",
    }
    _write(roles["prediction_seal"], b"{}\n")
    _write(roles["prediction_archive"], b"sealed prediction placeholder")
    with roles["target_data"].open("wb") as handle:
        pickle.dump(
            {
                "object_points": target,
                "object_visibilities": visibility,
                "object_motions_valid": validity,
            },
            handle,
        )
    _write(roles["outcome"], b"{}\n")
    inputs = {
        role: {"path": str(path), "sha256": development._sha256(path)}
        for role, path in roles.items()
    }
    report = {
        "protocol_id": SOURCE_PROTOCOL_ID,
        "case": case,
        "object_id": "002-rope-silk",
        "episode_id": 2,
        "center_ids": centers.tolist(),
        "scored_frames": list(development._post_update_scored_frames(76)),
        "inputs": inputs,
    }
    report_path = run / f"{case}.json"
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")
    archive_path = run / f"{case}.npz"
    np.savez_compressed(
        archive_path,
        center_ids=centers,
        physical_prior_m=comparator,
        recursive_rbf_risk_limited_m=primary,
    )
    validated_arrays = {
        "center_ids": centers,
        "physical_prior_m": comparator,
        "recursive_rbf_risk_limited_m": primary,
    }
    monkeypatch.setattr(
        development,
        "evaluate_deform360_online_belief_case",
        lambda _: (report, validated_arrays),
    )
    artifact = {
        "case": case,
        "report_sha256": development._sha256(report_path),
        "arrays_sha256": development._sha256(archive_path),
    }

    metadata, loaded = development._load_audited_case(source, run, case, artifact)

    assert metadata["case"] == case
    np.testing.assert_array_equal(loaded["target"], target)
    roles["target_data"].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="target_data binding or checksum changed"):
        development._load_audited_case(source, run, case, artifact)


def test_run_inventory_rejects_missing_or_nonwhitelisted_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(development, "_expected_case_names", lambda: ("a", "b"))
    for case in ("a", "b"):
        (tmp_path / f"{case}.json").write_text("{}")
        (tmp_path / f"{case}.npz").write_bytes(b"npz")
    assert development._validate_exact_run_inventory(tmp_path) == ("a", "b")

    (tmp_path / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="not exactly"):
        development._validate_exact_run_inventory(tmp_path)
