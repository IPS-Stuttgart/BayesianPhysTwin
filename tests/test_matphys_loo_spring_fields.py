import json
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.matphys_loo_spring_fields as spring_fields_module
from bayesian_phystwin.matphys_causal_bridge import (
    matphys_fresh_fold_initialization,
    sha256_file,
)
from bayesian_phystwin.matphys_loo_spring_fields import (
    collect_loo_spring_fields,
    merge_loo_spring_field_bundles,
    validate_loo_spring_fields,
)


def _identity(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _workspace(tmp_path: Path, *, fresh: bool = False) -> Path:
    protocol = tmp_path / "protocol.json"
    protocol_payload = {
        "protocol_id": "loo-test",
        "case_order": ["case_a", "case_b"],
    }
    if fresh:
        protocol_payload["source_training"] = {
            "initialization": matphys_fresh_fold_initialization(42)
        }
    protocol.write_text(json.dumps(protocol_payload), encoding="utf-8")
    folds = []
    for index, case in enumerate(("case_a", "case_b")):
        root = tmp_path / f"fold_{index:02d}"
        export_root = root / "matphys_export"
        case_root = export_root / "cases" / case
        case_root.mkdir(parents=True)
        field = case_root / "candidate_spring_y.npy"
        np.save(field, np.full(3, index + 1.0), allow_pickle=False)
        checkpoint = root / "last_checkpoint.pth"
        checkpoint.write_bytes(f"checkpoint-{index}".encode())
        audit = root / "audit.json"
        audit.write_text("{}\n", encoding="utf-8")
        export = {
            "schema_version": 1,
            "backbone": {
                "name": "MatPhys test",
                "source_repository": "https://example.test/matphys",
                "source_commit": "a" * 40,
                "future_observations_used": False,
                "coordinate_frame": "phystwin-world-metres-v1",
                "vertex_contract": "phystwin-observed-prefix-then-surface-v1",
                "proxy_contract": "test-proxy",
                "claim_boundary": "test",
                "checkpoint": _identity(checkpoint),
                "causal_training_audit": _identity(audit),
            },
            "cases": [
                {
                    "name": case,
                    "evidence_end_frame_exclusive": 4,
                    "spring_field_summary": {
                        "complete_spring_y": {
                            **_identity(field),
                            "count": 3,
                        }
                    },
                }
            ],
        }
        (export_root / "external_backbone_manifest.json").write_text(
            json.dumps(export), encoding="utf-8"
        )
        folds.append(
            {
                "fold_index": index,
                "held_out_object": f"object_{index}",
                "root": str(root),
                "target_cases": [case],
            }
        )
    workspace = tmp_path / "workspace.json"
    workspace.write_text(
        json.dumps(
            {
                "contract": "matphys-object-disjoint-loo-workspace-v1",
                "future_opened": False,
                "protocol": _identity(protocol),
                "folds": folds,
            }
        ),
        encoding="utf-8",
    )
    return workspace


def test_collect_bundle_remains_valid_after_directory_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        spring_fields_module,
        "validate_source_supervised_training_audit",
        lambda audit, checkpoint: {"audit": str(audit), "checkpoint": str(checkpoint)},
    )
    workspace = _workspace(tmp_path)
    original = tmp_path / "partial"
    result = collect_loo_spring_fields(workspace, original, fold_indices=(0,))

    assert result["complete_cohort"] is False
    moved = tmp_path / "moved"
    original.rename(moved)
    validated = validate_loo_spring_fields(
        moved / "loo_spring_fields.json", require_complete=False
    )

    assert validated["case_order"] == ["case_a"]
    assert Path(validated["cases"][0]["candidate_spring_y_path"]).is_file()


def test_merge_disjoint_host_bundles_in_canonical_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        spring_fields_module,
        "validate_source_supervised_training_audit",
        lambda audit, checkpoint: {"audit": str(audit), "checkpoint": str(checkpoint)},
    )
    workspace = _workspace(tmp_path)
    first = collect_loo_spring_fields(workspace, tmp_path / "first", fold_indices=(0,))
    second = collect_loo_spring_fields(
        workspace, tmp_path / "second", fold_indices=(1,)
    )

    merged = merge_loo_spring_field_bundles(
        [second["manifest_path"], first["manifest_path"]],
        tmp_path / "merged",
    )

    assert merged["complete_cohort"] is True
    assert merged["case_order"] == ["case_a", "case_b"]
    assert merged["fold_indices"] == [0, 1]
    validated = validate_loo_spring_fields(merged["manifest_path"])
    assert [entry["name"] for entry in validated["cases"]] == [
        "case_a",
        "case_b",
    ]


def test_merge_rejects_duplicate_case_bundles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        spring_fields_module,
        "validate_source_supervised_training_audit",
        lambda audit, checkpoint: {"audit": str(audit), "checkpoint": str(checkpoint)},
    )
    workspace = _workspace(tmp_path)
    first = collect_loo_spring_fields(workspace, tmp_path / "first", fold_indices=(0,))

    with pytest.raises(ValueError, match="duplicate spring field"):
        merge_loo_spring_field_bundles(
            [first["manifest_path"], first["manifest_path"]],
            tmp_path / "merged",
        )


def test_collect_rejects_fold_with_different_fresh_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        spring_fields_module,
        "validate_source_supervised_training_audit",
        lambda audit, checkpoint: {
            "parameterization": {
                "initialization": matphys_fresh_fold_initialization(43)
            }
        },
    )
    workspace = _workspace(tmp_path, fresh=True)

    with pytest.raises(ValueError, match="training initialization differs"):
        collect_loo_spring_fields(workspace, tmp_path / "bundle", fold_indices=(0,))
