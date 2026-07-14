import copy

from causal4d_public.deform360_replication_fit import (
    pool_source_warp_candidate_grids,
)


def _grid(episode: str, best: int) -> dict:
    episode_id = f"fixture/episode_{episode}"
    rows = []
    for index in range(200):
        value = 0.8 if index == best else 1.2 + index / 1000
        rows.append(
            {
                "candidate_index": index,
                "parameters": {
                    "stretch_spring_y": float(index + 1),
                    "bend_spring_y": 1.0,
                    "controller_spring_y": 1.0,
                    "ground_friction": 0.0,
                },
                "mean_chamfer_m": value,
                "p99_relative_edge_strain": 0.2,
            }
        )
    payload = {
        "schema_version": 3,
        "artifact_kind": "Deform360ReplicationSourceWarpCandidateGrid",
        "episode_id": episode_id,
        "reference_geometry_result_sha256": "0" * 64,
        "reference_geometry_total_frame_count": 10,
        "reference_geometry_available_frame_count": 9,
        "config": {"maximum_p99_relative_edge_strain": 0.5},
        "raw_hull_frame_indices": list(range(9)),
        "persistence": {"mean_m": 1.0},
        "candidate_scores": rows,
        "information_boundary": {"target_future_read": False},
    }
    from causal4d_public.deform360_replication_fit import _artifact_sha256

    payload["result_sha256"] = _artifact_sha256(payload)
    return payload


def test_pooling_seals_direct_single_source_control() -> None:
    grids = [_grid("a", 4), _grid("b", 4), _grid("c", 7)]
    result = pool_source_warp_candidate_grids(grids)
    assert result["selection"]["pooled_candidate_index"] == 4
    assert result["selection"]["single_source_candidate_indices"] == (4, 4, 7)
    assert result["pooled_source_relative_improvement_vs_persistence"] > 0.0
    assert result["information_boundary"]["target_future_read"] is False


def test_pooling_rejects_checksum_mutation() -> None:
    grids = [_grid("a", 4), _grid("b", 4)]
    changed = copy.deepcopy(grids[0])
    changed["candidate_scores"][0]["mean_chamfer_m"] = 0.0
    try:
        pool_source_warp_candidate_grids([changed, grids[1]])
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("mutated source grid was accepted")


def test_pooling_excludes_candidates_that_violate_source_strain_gate() -> None:
    first = _grid("a", 4)
    second = _grid("b", 4)
    first["candidate_scores"][4]["p99_relative_edge_strain"] = 0.8
    second["candidate_scores"][4]["p99_relative_edge_strain"] = 0.8
    from causal4d_public.deform360_replication_fit import _artifact_sha256

    first["result_sha256"] = _artifact_sha256(first)
    second["result_sha256"] = _artifact_sha256(second)
    result = pool_source_warp_candidate_grids([first, second])
    assert result["selection"]["pooled_candidate_index"] != 4
    assert result["candidate_quality_filter"][
        "maximum_p99_relative_edge_strain"
    ] == 0.5
