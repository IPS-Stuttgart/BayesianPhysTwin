import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.matphys_causal_bridge import sha256_file
from bayesian_phystwin.matphys_graph_parts import (
    GRAPH_PART_COMPACT_PROXY_CONTRACT,
    GRAPH_PART_PROXY_CONTRACT,
    compact_graph_part_proxy,
    graph_semantic_parts,
    materialize_compact_graph_proxy_subset,
)


def _chain(length: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.column_stack(
        (np.arange(length, dtype=float), np.zeros(length), np.zeros(length))
    )
    edges = np.column_stack((np.arange(length - 1), np.arange(1, length)))
    return points, edges


def test_graph_semantic_parts_are_deterministic_connected_and_nonempty() -> None:
    points, edges = _chain(12)
    features = np.zeros((12, 4), dtype=float)
    features[:4, 0] = 1.0
    features[4:8, 1] = 1.0
    features[8:, 2] = 1.0

    first = graph_semantic_parts(points, edges, features, part_count=3)
    second = graph_semantic_parts(points, edges, features, part_count=3)

    np.testing.assert_array_equal(first.assignments, second.assignments)
    np.testing.assert_array_equal(first.seeds, second.seeds)
    np.testing.assert_array_equal(first.part_counts, second.part_counts)
    assert np.all(first.part_counts > 0)
    for part in range(3):
        members = np.flatnonzero(first.assignments == part)
        assert np.all(np.diff(members) == 1)
    assert 0.0 < first.boundary_edge_fraction < 1.0


def test_semantic_boundary_changes_graph_geodesic_partition() -> None:
    points, edges = _chain(10)
    uniform = np.ones((10, 2), dtype=float)
    semantic = np.zeros((10, 2), dtype=float)
    semantic[:7, 0] = 1.0
    semantic[7:, 1] = 1.0

    geometric = graph_semantic_parts(
        points,
        edges,
        uniform,
        part_count=2,
        semantic_edge_weight=4.0,
    )
    informed = graph_semantic_parts(
        points,
        edges,
        semantic,
        part_count=2,
        semantic_edge_weight=20.0,
    )

    assert not np.array_equal(geometric.assignments, informed.assignments)
    assert informed.assignments[6] != informed.assignments[7]


def test_graph_semantic_parts_reject_zero_features() -> None:
    points, edges = _chain(4)

    with pytest.raises(ValueError, match="nonzero semantic feature"):
        graph_semantic_parts(points, edges, np.zeros((4, 2)), part_count=2)


def test_compact_proxy_preserves_part_features_and_replaces_only_dead_semantics(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    source_root = tmp_path / "source"
    node_sem = source_root / "semantic_cache" / "case_a_node_sem.npz"
    train_ready = source_root / "results" / "case_a" / "train" / "train_ready.pt"
    mapping = source_root / "case_to_material.json"
    node_sem.parent.mkdir(parents=True)
    train_ready.parent.mkdir(parents=True)
    mapping.write_text(json.dumps({"case_to_material": {"case_a": 0}}) + "\n")
    features = np.arange(24, dtype=np.float32).reshape(3, 8) + 1.0
    np.savez_compressed(node_sem, node_sem=features)
    part_features = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    torch.save(
        {
            "xyz": torch.zeros((3, 3)),
            "part_assignments": torch.tensor([0, 0, 1]),
            "part_features": part_features,
            "material_distributions": torch.ones((2, 10)) / 10,
        },
        train_ready,
    )
    summary_path = source_root / "proxy_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": GRAPH_PART_PROXY_CONTRACT,
                "claim_boundary": "test full graph proxy",
                "mapping": {"path": str(mapping), "sha256": sha256_file(mapping)},
                "semantic_cache_dir": str(node_sem.parent),
                "results_dir": str(source_root / "results"),
                "cases": [
                    {
                        "name": "case_a",
                        "structure_point_count": 3,
                        "semantic_dimension": 8,
                        "node_sem": {
                            "path": str(node_sem),
                            "sha256": sha256_file(node_sem),
                        },
                        "train_ready": {
                            "path": str(train_ready),
                            "sha256": sha256_file(train_ready),
                        },
                    }
                ],
            }
        )
        + "\n"
    )

    compact = compact_graph_part_proxy(summary_path, tmp_path / "compact")

    assert compact["contract"] == GRAPH_PART_COMPACT_PROXY_CONTRACT
    record = compact["cases"][0]
    compact_semantics = np.load(record["node_sem"]["path"])["node_sem"]
    assert compact_semantics.shape == (3, 1)
    np.testing.assert_array_equal(compact_semantics, 1.0)
    assert record["train_ready"]["sha256"] == sha256_file(train_ready)
    compact_ready = torch.load(
        record["train_ready"]["path"], map_location="cpu", weights_only=False
    )
    torch.testing.assert_close(compact_ready["part_features"], part_features)


def _compact_proxy_source(
    root: Path,
    source_mapping: Path,
    cases: tuple[str, ...],
) -> Path:
    class_to_id = {"cloth": 0, "rope": 1}
    mapping = root / "case_to_material.json"
    mapping.parent.mkdir(parents=True)
    mapping.write_text(
        json.dumps(
            {
                "case_to_material": {
                    case: "cloth" if "cloth" in case else "rope" for case in cases
                },
                "class_to_id": class_to_id,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = []
    for index, case in enumerate(cases):
        node_sem = root / "semantic_cache" / f"{case}_node_sem.npz"
        train_ready = root / "results" / case / "train" / "train_ready.pt"
        node_sem.parent.mkdir(parents=True, exist_ok=True)
        train_ready.parent.mkdir(parents=True, exist_ok=True)
        node_sem.write_bytes(f"node-{case}-{index}".encode())
        train_ready.write_bytes(f"ready-{case}-{index}".encode())
        label = "cloth" if "cloth" in case else "rope"
        records.append(
            {
                "name": case,
                "material_label": label,
                "material_id": class_to_id[label],
                "structure_point_count": index + 2,
                "node_sem": {"path": str(node_sem), "sha256": sha256_file(node_sem)},
                "train_ready": {
                    "path": str(train_ready),
                    "sha256": sha256_file(train_ready),
                },
            }
        )
    summary = root / "proxy_summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": GRAPH_PART_COMPACT_PROXY_CONTRACT,
                "part_count": 5,
                "semantic_edge_weight": 4.0,
                "source_mapping": {
                    "path": str(source_mapping),
                    "sha256": sha256_file(source_mapping),
                },
                "mapping": {"path": str(mapping), "sha256": sha256_file(mapping)},
                "cases": records,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def test_materialize_compact_proxy_subset_combines_byte_bound_sources(
    tmp_path: Path,
) -> None:
    source_mapping_a = tmp_path / "source_mapping_a.json"
    source_mapping_b = tmp_path / "source_mapping_b.json"
    mapping_bytes = json.dumps({"cloth": 0, "rope": 1}) + "\n"
    source_mapping_a.write_text(mapping_bytes, encoding="utf-8")
    source_mapping_b.write_text(mapping_bytes, encoding="utf-8")
    first = _compact_proxy_source(
        tmp_path / "first", source_mapping_a, ("single_cloth", "single_rope")
    )
    second = _compact_proxy_source(
        tmp_path / "second", source_mapping_b, ("double_cloth",)
    )

    result = materialize_compact_graph_proxy_subset(
        (first, second),
        tmp_path / "subset",
        ("double_cloth", "single_rope"),
    )

    assert [record["name"] for record in result["cases"]] == [
        "double_cloth",
        "single_rope",
    ]
    assert result["source_mapping"]["sha256"] == sha256_file(source_mapping_a)
    for record in result["cases"]:
        source_record = next(
            item
            for summary_path in (first, second)
            for item in json.loads(summary_path.read_text())["cases"]
            if item["name"] == record["name"]
        )
        assert sha256_file(record["node_sem"]["path"]) == source_record["node_sem"][
            "sha256"
        ]
        assert sha256_file(record["train_ready"]["path"]) == source_record[
            "train_ready"
        ]["sha256"]
    repeated = materialize_compact_graph_proxy_subset(
        (first, second),
        tmp_path / "subset",
        ("double_cloth", "single_rope"),
    )
    assert repeated["mapping"]["sha256"] == result["mapping"]["sha256"]


def test_materialize_compact_proxy_subset_rejects_missing_and_duplicate_cases(
    tmp_path: Path,
) -> None:
    source_mapping = tmp_path / "source_mapping.json"
    source_mapping.write_text("{}\n", encoding="utf-8")
    summary = _compact_proxy_source(tmp_path / "source", source_mapping, ("rope",))

    with pytest.raises(ValueError, match="nonempty and unique"):
        materialize_compact_graph_proxy_subset(
            (summary,), tmp_path / "duplicate", ("rope", "rope")
        )
    with pytest.raises(ValueError, match="omit requested"):
        materialize_compact_graph_proxy_subset(
            (summary,), tmp_path / "missing", ("cloth",)
        )
