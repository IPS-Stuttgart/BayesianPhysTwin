from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "science"
    / "run_deform360_cross_object_normalized_evidence.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_deform360_cross_object_normalized_evidence",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

EpisodeSpec = _MODULE.EpisodeSpec
RollingEvent = _MODULE.RollingEvent
load_episode = _MODULE.load_episode
load_protocol = _MODULE.load_protocol
normalized_weights = _MODULE.normalized_weights
run_experiment = _MODULE.run_experiment


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _protocol(
    objects: list[str], episodes: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "schema": "bayesian-phystwin/deform360-cross-object-normalized-evidence-v1",
        "schema_version": 1,
        "protocol_id": "synthetic-cross-object",
        "status": "retrospective-non-fresh-cross-fitted-diagnostic",
        "claim_boundary": "Synthetic non-fresh cross-fitted diagnostic only.",
        "provenance_locks": {
            "allowed_relative_root": "data/observations",
            "metadata_content_inventory_sha256": "1" * 64,
            "metadata_preflight_commit": "2" * 40,
            "npz_header_inventory_sha256": "3" * 64,
        },
        "information_boundary": {
            "all_objects_previously_used_by_repository": True,
            "array_payloads_opened_when_protocol_locked": False,
            "fold_local_target_exclusion": True,
            "globally_sealed_target_cohort": False,
            "npz_headers_previously_inspected": True,
            "official_deform360_task": False,
        },
        "objects": objects,
        "episodes": episodes,
        "minimum_prefix_displacements": 2,
        "max_points_per_hull": 16,
        "normalized_evidence": {
            "weight_rule": "normalized",
            "kappa_candidates": [0.5, 1.0, 2.0],
            "selection_unit": "equal source-object mean",
            "selection_objective": "one-step Gaussian negative log likelihood",
            "tie_break": ["point", "one", "lower"],
            "boundary_selection_is_inconclusive": True,
        },
        "cross_fitting": {
            "folds": "one fold per target object",
            "source_objects": "all others",
            "target_outcome_used_for_own_selection": False,
            "same_object_used_as_source_in_other_folds": True,
        },
        "methods": [
            "persistence",
            "last_residual",
            "cumulative_model_average",
            "normalized_kappa_1",
            "cross_fitted_normalized",
        ],
        "metrics": {},
        "bootstrap": {"samples": 100, "seed": 7, "unit": "target object"},
        "diagnostic_readout": {"no_automatic_claim_promotion": True},
    }


def _packed_episode(path: Path, velocity: np.ndarray, *, frames: int = 8) -> None:
    base = np.array(
        [
            [-0.01, -0.01, 0.0],
            [0.01, -0.01, 0.0],
            [-0.01, 0.01, 0.0],
            [0.01, 0.01, 0.0],
        ],
        dtype=np.float64,
    )
    hulls = []
    for frame in range(frames):
        acceleration = 0.00005 * frame * frame * np.array([1.0, -0.5, 0.25])
        hulls.append(base + frame * velocity + acceleration)
    points = np.concatenate(hulls, axis=0)
    offsets = np.arange(0, len(points) + 1, len(base), dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        frame_indices=np.arange(frames, dtype=np.int32),
        point_offsets=offsets,
        points_world_m=points,
    )


def _event() -> object:
    component_mean = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    component_variance = np.array([1e-4, 1e-4])
    weights = np.array([1.0 - 1e-12, 1e-12])
    prior = np.array([0.5, 0.5])
    hull = np.zeros((4, 3))
    return RollingEvent(
        object_id="object",
        episode_path="episode",
        step_index=4,
        target_delta_m=np.zeros(3),
        last_delta_m=np.zeros(3),
        current_hull_m=hull,
        target_hull_m=hull,
        component_log_evidence=np.array([0.0, -100.0]),
        component_mean_m=component_mean,
        component_variance_m2=component_variance,
        cumulative_weights=weights,
        prior_probability=prior,
        update_count=100,
    )


def test_update_count_normalization_prevents_cumulative_collapse() -> None:
    event = _event()
    weights = normalized_weights(event, 1.0)

    assert 0.70 < weights[0] < 0.75
    assert 0.25 < weights[1] < 0.30
    assert weights[1] > event.cumulative_weights[1] * 1e10
    assert not weights.flags.writeable


def test_protocol_rejects_fresh_or_official_claim(tmp_path: Path) -> None:
    protocol = _protocol(
        ["a", "b", "c"],
        [
            {
                "object_id": object_id,
                "path": f"data/observations/{object_id}/episode/sampled_hulls.npz",
                "expected_frame_count": 8,
            }
            for object_id in ("a", "b", "c")
        ],
    )
    protocol["information_boundary"]["globally_sealed_target_cohort"] = True
    path = tmp_path / "protocol.json"
    _write_json(path, protocol)

    with pytest.raises(ValueError, match="information boundary"):
        load_protocol(path)


def test_episode_loader_validates_locked_path_and_contract(tmp_path: Path) -> None:
    relative = "data/observations/a/episode/sampled_hulls.npz"
    path = tmp_path / relative
    _packed_episode(path, np.array([0.001, 0.0, 0.0]))
    episode = load_episode(
        tmp_path,
        EpisodeSpec("a", relative, 8),
        max_points=4,
    )

    assert episode.centroids_m.shape == (8, 3)
    assert len(episode.hulls_m) == 8
    assert all(hull.shape == (4, 3) for hull in episode.hulls_m)
    assert len(episode.file_sha256) == 64

    with pytest.raises(ValueError, match="escaped"):
        load_episode(
            tmp_path,
            EpisodeSpec("a", "../outside.npz", 8),
            max_points=4,
        )


def test_end_to_end_leave_one_object_out_diagnostic(tmp_path: Path) -> None:
    objects = ["a", "b", "c"]
    velocities = {
        "a": np.array([0.0010, 0.0001, 0.0]),
        "b": np.array([0.0012, -0.0001, 0.0]),
        "c": np.array([0.0008, 0.0002, 0.0]),
    }
    episodes = []
    for object_id in objects:
        relative = f"data/observations/{object_id}/episode/sampled_hulls.npz"
        _packed_episode(tmp_path / relative, velocities[object_id])
        episodes.append(
            {
                "object_id": object_id,
                "path": relative,
                "expected_frame_count": 8,
            }
        )
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, _protocol(objects, episodes))
    output = tmp_path / "output"

    result = run_experiment(
        tmp_path,
        protocol_path,
        output,
        workers=1,
        bootstrap_samples=100,
        revision="revision-test",
    )

    assert len(result["folds"]) == 3
    for fold in result["folds"]:
        assert fold["target_object"] not in fold["selection"]["source_objects"]
        assert len(fold["selection"]["source_objects"]) == 2
        assert fold["target_metrics"]["cross_fitted_normalized"]["event_count"] > 0
    assert result["diagnosis"]["no_automatic_claim_promotion"] is True
    assert (output / "summary.json").is_file()
    assert (output / "readout.json").is_file()
    assert (output / "per_object.csv").is_file()
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    assert len(manifest["inputs"]) == 3
    assert len(result["result_sha256"]) == 64
