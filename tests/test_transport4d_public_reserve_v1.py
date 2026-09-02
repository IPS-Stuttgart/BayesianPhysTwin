from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin_experiments.transport4d_public_reserve_v1 import (
    audit_deform360_transport_reserve,
    canonical_id,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def metadata(action: str) -> dict[str, object]:
    return {
        "object": action,
        "sequences": {
            "0": {"action": f"touch-{action}"},
            "1": {"action": f"move-{action}"},
        },
    }


def fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "deform360"
    raw = root / "raw-repository" / "raw"
    for object_id in (
        "opened-development",
        "protected-upstream-reserve",
        "opened-confirmation",
        "cross-repo-protected",
        "new-object-a",
        "new-object-b",
    ):
        object_root = raw / object_id
        write_json(object_root / "metadata.json", metadata(object_id))
        # The audit must never inspect this synthetic numeric payload.
        (object_root / "numeric-outcome.npy").write_bytes(b"not-a-numpy-array")

    action_path = tmp_path / "action.json"
    action = {
        "schema": "bayesian-phystwin/deform360-action-kernel-protocol-v3",
        "protocol_id": "action-test-v3",
        "development_object_ids": ["opened-development"],
        "reserved_object_ids": ["protected-upstream-reserve"],
    }
    write_json(action_path, action)

    untouched_path = tmp_path / "untouched.json"
    untouched = {
        "schema": "bayesian-phystwin/deform360-untouched-confirmation-protocol-v5",
        "protocol_id": "untouched-test-v5",
        "eligible_object_ids": ["opened-confirmation"],
    }
    write_json(untouched_path, untouched)

    protocol_path = tmp_path / "reserve.json"
    protocol = {
        "schema": "bayesian-phystwin.transport4d_deform360_reserve_protocol",
        "schema_version": 1,
        "status": "frozen-before-reserve-metadata-access",
        "dataset_root": str(root.resolve()),
        "upstream_bindings": {
            "action_kernel_v3": {
                "path": str(action_path),
                "protocol_id": "action-test-v3",
            },
            "untouched_confirmation_v5": {
                "path": str(untouched_path),
                "protocol_id": "untouched-test-v5",
                "eligible_object_count": 1,
            },
            "causal4d_deform360_holdings_v1": {
                "repository": "IPS-Stuttgart/Causal4D",
                "revision": "a" * 40,
                "path": (
                    "configs/causal4d_public/deform360_gpuserver6000_holdings_v1.json"
                ),
                "git_blob_sha1": "b" * 40,
                "additional_protected_object_ids": ["cross-repo-protected"],
            },
        },
        "additional_protected_object_ids": ["cross-repo-protected"],
        "reservation": {
            "include_every_remaining_metadata_object": True,
            "split_rule": "sha256-ranked-first-calibration-remainder-confirmation-v1",
            "split_salt": "test-reserve",
            "preferred_calibration_fraction": 0.5,
            "minimum_calibration_objects": 1,
            "minimum_confirmation_objects": 1,
            "maximum_metadata_bytes": 100000,
            "replacement_allowed": False,
            "split_changes_after_metadata_access_allowed": False,
        },
        "future_carrier_qualification": {
            "metadata_split_must_remain_fixed": True,
            "numeric_payload_may_not_select_support": True,
            "carrier_identity_and_file_structure_only": True,
            "support_negative_objects_retained": True,
            "replacement_allowed": False,
            "confirmation_numeric_access_requires_separate_reviewed_protocol": True,
        },
        "information_boundary": {
            "metadata_json_may_open": True,
            "directory_names_may_open": True,
            "robot_numeric_payload_opened": False,
            "tactile_numeric_payload_opened": False,
            "camera_pixel_opened": False,
            "geometry_or_point_cloud_opened": False,
            "target_outcome_opened": False,
            "confirmation_authorized": False,
            "paper_claim_authorized": False,
        },
        "claim_boundary": "test-only metadata reservation",
    }
    protocol["protocol_id"] = canonical_id(protocol)
    write_json(protocol_path, protocol)
    return root, protocol_path, action_path, untouched_path


def run_fixture(tmp_path: Path) -> dict[str, object]:
    root, protocol, action, untouched = fixture(tmp_path)
    return audit_deform360_transport_reserve(
        data_root=root,
        reserve_protocol_path=protocol,
        action_kernel_protocol_path=action,
        untouched_protocol_path=untouched,
    )


def test_reserves_every_unprotected_metadata_object_before_numeric_access(
    tmp_path: Path,
) -> None:
    result = run_fixture(tmp_path)

    assert result["status"] == "metadata-reserve-ready"
    assert result["remaining_metadata_object_count"] == 2
    assigned = set(result["calibration_object_ids"]) | set(
        result["confirmation_object_ids"]
    )
    assert assigned == {"new-object-a", "new-object-b"}
    assert set(result["calibration_object_ids"]).isdisjoint(
        result["confirmation_object_ids"]
    )
    assert result["reservation_ready"] is True
    assert "cross-repo-protected" in result["protected_object_ids"]
    assert all(row["numeric_payload_opened"] is False for row in result["objects"])
    boundary = result["information_boundary"]
    assert boundary["target_outcome_opened"] is False
    assert boundary["confirmation_authorized"] is False


def test_reservation_is_deterministic(tmp_path: Path) -> None:
    first = run_fixture(tmp_path / "first")
    second = run_fixture(tmp_path / "second")

    assert first["calibration_object_ids"] == second["calibration_object_ids"]
    assert first["confirmation_object_ids"] == second["confirmation_object_ids"]
    first_rows = [(row["object_id"], row["split"]) for row in first["objects"]]
    second_rows = [(row["object_id"], row["split"]) for row in second["objects"]]
    assert first_rows == second_rows


def test_protocol_tampering_fails_closed(tmp_path: Path) -> None:
    root, protocol_path, action, untouched = fixture(tmp_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["reservation"]["replacement_allowed"] = True
    write_json(protocol_path, protocol)

    with pytest.raises(ValueError, match="protocol_id"):
        audit_deform360_transport_reserve(
            data_root=root,
            reserve_protocol_path=protocol_path,
            action_kernel_protocol_path=action,
            untouched_protocol_path=untouched,
        )


def test_upstream_roster_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    root, protocol, action_path, untouched = fixture(tmp_path)
    action = json.loads(action_path.read_text(encoding="utf-8"))
    action["protocol_id"] = "another-protocol"
    write_json(action_path, action)

    with pytest.raises(ValueError, match="action-kernel protocol changed"):
        audit_deform360_transport_reserve(
            data_root=root,
            reserve_protocol_path=protocol,
            action_kernel_protocol_path=action_path,
            untouched_protocol_path=untouched,
        )


def test_cross_repository_roster_mismatch_fails_closed(tmp_path: Path) -> None:
    root, protocol_path, action, untouched = fixture(tmp_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["additional_protected_object_ids"] = []
    unsigned = {key: value for key, value in protocol.items() if key != "protocol_id"}
    protocol["protocol_id"] = canonical_id(unsigned)
    write_json(protocol_path, protocol)

    with pytest.raises(ValueError, match="cross-repository protected object roster"):
        audit_deform360_transport_reserve(
            data_root=root,
            reserve_protocol_path=protocol_path,
            action_kernel_protocol_path=action,
            untouched_protocol_path=untouched,
        )
