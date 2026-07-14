from __future__ import annotations

import json
from pathlib import Path

from causal4d_public.pokeflex import (
    PINNED_POKEFLEX_COMMIT,
    PokeFlexEpisode,
    PokeFlexReadinessConfig,
    discover_pokeflex_episodes,
    load_readiness_config,
    preflight_pokeflex_dataset,
    validate_preflight_result,
    write_synthetic_pokeflex_fixture,
)


def _config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "causal4d_public"
        / "pokeflex_readiness_v1.json"
    )


def test_readiness_config_is_checksum_locked_to_public_upstream() -> None:
    config = load_readiness_config(_config_path())

    assert config.upstream_commit == PINNED_POKEFLEX_COMMIT
    assert config.minimum_takes_per_object_for_cross_take == 5
    assert config.prefix_frame_count == 6


def test_synthetic_fixture_passes_without_reading_outcomes(tmp_path: Path) -> None:
    root = tmp_path / "pokeflex"
    write_synthetic_pokeflex_fixture(
        root,
        takes_per_object=5,
        frame_count=16,
        include_material_identity=True,
    )

    episodes = discover_pokeflex_episodes(root)
    result = preflight_pokeflex_dataset(root, PokeFlexReadinessConfig())
    validation = validate_preflight_result(result)

    assert len(episodes) == 5
    assert all(isinstance(episode, PokeFlexEpisode) for episode in episodes)
    assert episodes[0].episode_id == "fixture_plush_00/poke_000"
    assert episodes[0].layout == "raw"
    assert validation["passed"] is True
    assert result["preflight_passed"] is True
    assert result["dataset_inventory"]["take_count"] == 5
    assert result["metadata_only_split"]["split_counts"] == {
        "development": 3,
        "calibration": 1,
        "target": 1,
    }
    assert result["capability_gates"]["cross_take_interventional_evaluation_ready"]
    assert result["capability_gates"]["identity_dependent_track_metric_ready"]
    assert result["information_boundary"]["prediction_metrics_computed"] is False
    assert result["information_boundary"]["model_parameters_fitted"] is False
    assert str(tmp_path) not in json.dumps(result)


def test_public_like_fixture_keeps_geometry_and_identity_gates_separate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pokeflex"
    write_synthetic_pokeflex_fixture(root, takes_per_object=5, frame_count=16)

    result = preflight_pokeflex_dataset(root)

    assert result["preflight_passed"] is True
    assert result["capability_gates"]["factual_geometry_continuation_ready"]
    assert result["capability_gates"]["identity_dependent_track_metric_ready"] is False
    assert all(
        take["mesh"]["material_identity"]["status"] == "unverified_by_public_code"
        for take in result["takes"]
    )


def test_topology_mutation_is_reported_without_disabling_geometry_metrics(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pokeflex"
    write_synthetic_pokeflex_fixture(
        root,
        takes_per_object=5,
        frame_count=16,
        mutate_last_mesh_topology=True,
    )

    result = preflight_pokeflex_dataset(root)
    mutated = next(take for take in result["takes"] if take["take_id"] == "poke_004")

    assert mutated["capabilities"]["sampled_topology_consistent"] is False
    assert mutated["capabilities"]["geometry_observation_ready"] is True
    assert result["metric_contract"]["geometry_metrics_allowed"]
    assert result["capability_gates"]["identity_dependent_track_metric_ready"] is False


def test_missing_timestamps_blocks_delay_but_not_frame_aligned_continuation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pokeflex"
    write_synthetic_pokeflex_fixture(root, takes_per_object=5, frame_count=16)
    for robot_path in root.glob("*/*/robot_data.json"):
        records = json.loads(robot_path.read_text(encoding="utf-8"))
        for record in records:
            record.pop("timestamp_s")
        robot_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    result = preflight_pokeflex_dataset(root)

    assert result["preflight_passed"] is True
    assert result["capability_gates"]["delay_inference_ready"] is False
    assert result["capability_gates"]["factual_geometry_continuation_ready"]


def test_zero_padded_public_frame_ids_are_accepted(tmp_path: Path) -> None:
    root = tmp_path / "pokeflex"
    write_synthetic_pokeflex_fixture(root, takes_per_object=5, frame_count=16)
    for robot_path in root.glob("*/*/robot_data.json"):
        records = json.loads(robot_path.read_text(encoding="utf-8"))
        for record in records:
            record["frame"] = f"{record['frame']:05d}"
        robot_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    result = preflight_pokeflex_dataset(root)

    assert result["preflight_passed"] is True
    assert all(take["robot"]["frames_unique"] for take in result["takes"])
    assert all(take["robot"]["first_frame"] == 1 for take in result["takes"])
    assert all(take["robot"]["last_frame"] == 16 for take in result["takes"])


def test_fixture_and_preflight_are_location_independent(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_synthetic_pokeflex_fixture(first, takes_per_object=5, frame_count=16)
    write_synthetic_pokeflex_fixture(second, takes_per_object=5, frame_count=16)

    first_result = preflight_pokeflex_dataset(first)
    second_result = preflight_pokeflex_dataset(second)

    assert first_result == second_result


def test_malformed_take_is_reported_without_hiding_other_eligible_takes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pokeflex"
    write_synthetic_pokeflex_fixture(root, takes_per_object=5, frame_count=16)
    malformed = root / "fixture_plush_00" / "poke_004" / "robot_data.json"
    malformed.write_text("{not-json}\n", encoding="utf-8")

    result = preflight_pokeflex_dataset(root)
    failed = next(take for take in result["takes"] if take["take_id"] == "poke_004")

    assert result["preflight_passed"] is True
    assert result["dataset_inventory"]["eligible_take_count"] == 4
    assert failed["eligible_for_metadata_split"] is False
    assert "robot audit failed" in failed["robot"]["reason"]
    assert (
        result["capability_gates"]["cross_take_interventional_evaluation_ready"]
        is False
    )
