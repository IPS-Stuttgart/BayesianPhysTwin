import json
from pathlib import Path

import pytest

from bayesian_phystwin.matphys_causal_bridge import sha256_file
from bayesian_phystwin.matphys_graph_parts import GRAPH_PART_COMPACT_PROXY_CONTRACT
from bayesian_phystwin.matphys_loo_protocol import (
    load_matphys_loo_protocol,
    prepare_matphys_loo_workspace,
)
from bayesian_phystwin.phystwin_sota_comparison import PHYSTWIN_TABLE1_CASES


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPOSITORY_ROOT
    / "configs"
    / "sota"
    / "matphys_guarded_bayesian_loo22_v1.json"
)


def test_registered_loo_protocol_is_exhaustive_and_object_disjoint() -> None:
    protocol = load_matphys_loo_protocol(PROTOCOL)

    targets = [case for fold in protocol["folds"] for case in fold["targets"]]
    assert sorted(targets) == sorted(PHYSTWIN_TABLE1_CASES)
    assert len(targets) == len(set(targets)) == 22
    assert len(protocol["folds"]) == 11
    for fold in protocol["folds"]:
        assert not set(fold["targets"]) & set(fold["sources"])


def test_loo_protocol_rejects_interaction_leakage(tmp_path: Path) -> None:
    payload = PROTOCOL.read_text(encoding="utf-8").replace(
        '"targets": ["single_lift_cloth"]',
        '"targets": ["single_lift_cloth", "single_lift_cloth_1"]',
    )
    changed = tmp_path / "protocol.json"
    changed.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="appears twice|wrong object fold"):
        load_matphys_loo_protocol(changed)


def test_prepare_loo_workspace_materializes_every_partition(tmp_path: Path) -> None:
    source_root = tmp_path / "compact"
    source_mapping = source_root / "source_mapping.json"
    mapping = source_root / "case_to_material.json"
    source_mapping.parent.mkdir(parents=True)
    source_mapping.write_text('{"generic": 0}\n', encoding="utf-8")
    mapping.write_text(
        json.dumps(
            {
                "case_to_material": {
                    case: "generic" for case in PHYSTWIN_TABLE1_CASES
                },
                "class_to_id": {"generic": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = []
    for case in PHYSTWIN_TABLE1_CASES:
        node_sem = source_root / "semantic_cache" / f"{case}_node_sem.npz"
        train_ready = source_root / "results" / case / "train" / "train_ready.pt"
        node_sem.parent.mkdir(parents=True, exist_ok=True)
        train_ready.parent.mkdir(parents=True, exist_ok=True)
        node_sem.write_bytes(f"node-{case}".encode())
        train_ready.write_bytes(f"ready-{case}".encode())
        records.append(
            {
                "name": case,
                "material_label": "generic",
                "material_id": 0,
                "semantic_dimension": 1024,
                "node_sem": {
                    "path": str(node_sem),
                    "sha256": sha256_file(node_sem),
                },
                "train_ready": {
                    "path": str(train_ready),
                    "sha256": sha256_file(train_ready),
                },
            }
        )
    summary = source_root / "proxy_summary.json"
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

    workspace = prepare_matphys_loo_workspace(
        PROTOCOL, (summary,), tmp_path / "workspace"
    )

    assert workspace["future_opened"] is False
    assert len(workspace["folds"]) == 11
    for fold in workspace["folds"]:
        registration = json.loads(Path(fold["registration"]["path"]).read_text())
        assert set(registration["source_cases"]) == set(fold["source_cases"])
        assert set(registration["target_cases"]) == set(fold["target_cases"])
        assert not set(fold["source_cases"]) & set(fold["target_cases"])
        for proxy_name, expected_cases in (
            ("source_proxy", fold["source_cases"]),
            ("target_proxy", fold["target_cases"]),
        ):
            proxy = json.loads(Path(fold[proxy_name]["path"]).read_text())
            assert [record["name"] for record in proxy["cases"]] == expected_cases
