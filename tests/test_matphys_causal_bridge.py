import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.matphys_causal_bridge import (
    causal_uniform_frame_indices,
    merge_matphys_external_manifests,
    sha256_file,
    validate_causal_training_audit,
    write_causal_training_audit,
)


def test_causal_uniform_frames_never_reach_future() -> None:
    indices = causal_uniform_frame_indices(100, 60, 16)

    assert len(indices) == 16
    assert indices[0] == 0
    assert indices[-1] == 59
    assert np.all(indices < 60)


def test_causal_uniform_frames_reject_invalid_boundary() -> None:
    with pytest.raises(ValueError, match="evidence end"):
        causal_uniform_frame_indices(10, 11, 4)


def test_training_audit_binds_checkpoint_and_rejects_future_access(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    proxy = tmp_path / "proxy.json"
    proxy.write_text("{}\n")
    audit_path = tmp_path / "audit.json"
    audit = write_causal_training_audit(
        [checkpoint],
        audit_path,
        source_repository="https://example.test/matphys",
        source_commit="a" * 40,
        data_root=tmp_path,
        accessed_frame_indices={"case_a": [0, 2, 4]},
        objective_end_frames_exclusive={"case_a": 5},
        split_by_case={"case_a": {"train": [0, 5], "test": [5, 8]}},
        proxy_summary_path=proxy,
    )

    validated = validate_causal_training_audit(audit_path, checkpoint)
    assert validated["future_observations_used"] is False
    assert validated["checkpoints"][0]["sha256"] == sha256_file(checkpoint)
    assert audit["audit_sha256"] == sha256_file(audit_path)

    with pytest.raises(ValueError, match="future frame"):
        write_causal_training_audit(
            [checkpoint],
            tmp_path / "bad.json",
            source_repository="https://example.test/matphys",
            source_commit="a" * 40,
            data_root=tmp_path,
            accessed_frame_indices={"case_a": [0, 5]},
            objective_end_frames_exclusive={"case_a": 5},
            split_by_case={"case_a": {"train": [0, 5], "test": [5, 8]}},
            proxy_summary_path=proxy,
        )


def test_training_audit_rejects_changed_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    proxy = tmp_path / "proxy.json"
    proxy.write_text("{}\n")
    audit_path = tmp_path / "audit.json"
    write_causal_training_audit(
        [checkpoint],
        audit_path,
        source_repository="https://example.test/matphys",
        source_commit="a" * 40,
        data_root=tmp_path,
        accessed_frame_indices={"case_a": [0]},
        objective_end_frames_exclusive={"case_a": 2},
        split_by_case={"case_a": {"train": [0, 2], "test": [2, 3]}},
        proxy_summary_path=proxy,
    )
    checkpoint.write_bytes(b"changed")

    with pytest.raises(ValueError, match="checkpoint bytes"):
        validate_causal_training_audit(audit_path, checkpoint)


def test_audit_revalidates_serialized_frame_boundary(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "future_observations_used": False,
                "fit_all_frames": False,
                "optimization_and_checkpoint_selection": "released-prefix-only-v1",
                "checkpoint_policy": "fixed-terminal-epoch-v1",
                "checkpoints": [
                    {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)}
                ],
                "cases": [
                    {
                        "name": "case_a",
                        "train_end_frame_exclusive": 2,
                        "accessed_frame_indices": [0, 2],
                        "objective_frame_interval": [1, 2],
                        "maximum_objective_frame": 1,
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="future video access"):
        validate_causal_training_audit(audit_path, checkpoint)


def test_training_audit_rejects_future_objective_access(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    proxy = tmp_path / "proxy.json"
    proxy.write_text("{}\n")

    with pytest.raises(ValueError, match="objective accessed a future frame"):
        write_causal_training_audit(
            [checkpoint],
            tmp_path / "bad_objective.json",
            source_repository="https://example.test/matphys",
            source_commit="a" * 40,
            data_root=tmp_path,
            accessed_frame_indices={"case_a": [0, 1]},
            objective_end_frames_exclusive={"case_a": 3},
            split_by_case={"case_a": {"train": [0, 2], "test": [2, 3]}},
            proxy_summary_path=proxy,
        )


def _write_external_manifest(
    path: Path,
    case: str,
    trajectory: Path,
    checkpoint: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backbone": {
                    "name": "causal MatPhys",
                    "source_repository": "https://example.test/matphys",
                    "source_commit": "a" * 40,
                    "future_observations_used": False,
                    "coordinate_frame": "phystwin-world-metres-v1",
                    "vertex_contract": "phystwin-observed-prefix-then-surface-v1",
                    "proxy_contract": "test-proxy",
                    "claim_boundary": "test boundary",
                    "checkpoint": {"sha256": checkpoint},
                    "causal_training_audit": {"sha256": f"audit-{checkpoint}"},
                },
                "cases": [
                    {
                        "name": case,
                        "trajectory": str(trajectory),
                        "sha256": sha256_file(trajectory),
                        "evidence_end_frame_exclusive": 2,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_merge_matphys_manifests_keeps_per_case_provenance(tmp_path: Path) -> None:
    first_trajectory = tmp_path / "first.pkl"
    second_trajectory = tmp_path / "second.pkl"
    first_trajectory.write_bytes(b"first")
    second_trajectory.write_bytes(b"second")
    first_manifest = tmp_path / "first.json"
    second_manifest = tmp_path / "second.json"
    _write_external_manifest(first_manifest, "case_a", first_trajectory, "one")
    _write_external_manifest(second_manifest, "case_b", second_trajectory, "two")

    merged = merge_matphys_external_manifests(
        [second_manifest, first_manifest],
        tmp_path / "merged.json",
        case_order=("case_a", "case_b"),
    )

    assert [case["name"] for case in merged["cases"]] == ["case_a", "case_b"]
    assert merged["backbone"]["training_scope"] == (
        "independent-per-case-fixed-terminal-v1"
    )
    components = merged["backbone"]["component_manifests"]
    assert [component["checkpoint"]["sha256"] for component in components] == [
        "two",
        "one",
    ]
    assert Path(merged["manifest_path"]).is_file()


def test_merge_matphys_manifests_rejects_incompatible_backbones(
    tmp_path: Path,
) -> None:
    trajectory = tmp_path / "trajectory.pkl"
    trajectory.write_bytes(b"trajectory")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_external_manifest(first, "case_a", trajectory, "one")
    _write_external_manifest(second, "case_b", trajectory, "two")
    payload = json.loads(second.read_text(encoding="utf-8"))
    payload["backbone"]["source_commit"] = "b" * 40
    second.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible backbones"):
        merge_matphys_external_manifests([first, second], tmp_path / "merged.json")
