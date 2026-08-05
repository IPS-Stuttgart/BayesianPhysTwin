import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

import bayesian_phystwin.pokeflex_public_transfer_audit as audit_module
from bayesian_phystwin.pokeflex_action_robust_all18 import SOURCE_FIELD
from bayesian_phystwin.pokeflex_public_transfer_audit import (
    BASE_EFFECTIVE_SCALE,
    LEGACY_RUNNER_FILE_SHA256,
    PROTOCOL_SHA256,
    build_public_transfer_protocol,
    protocol_sha256,
    public_transfer_partitions,
    summarize_rows,
    take_row_from_smoke,
    validate_public_transfer_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = (
    ROOT / "configs" / "sota" / "pokeflex_action_robust_scale_all18_v4.json"
)
FRESHNESS = (
    ROOT
    / "configs"
    / "sota"
    / "pokeflex_action_robust_fresh2_exclusion_audit_v5.json"
)
FROZEN_PROTOCOL = (
    ROOT
    / "configs"
    / "sota"
    / "pokeflex_action_robust_public78_retrospective_v6.json"
)
ARCHIVE_INVENTORY = (
    ROOT / "configs" / "sota" / "pokeflex_public78_archive_inventory_v6.json"
)


def _inputs() -> tuple[dict[str, object], dict[str, object]]:
    return (
        json.loads(CALIBRATION.read_text(encoding="utf-8")),
        json.loads(FRESHNESS.read_text(encoding="utf-8")),
    )


def _built_protocol() -> dict[str, object]:
    calibration, freshness = _inputs()
    partitions = public_transfer_partitions(calibration, freshness)
    inventory = {
        "root": "/registered/poking",
        "zip_count": len(partitions["retrospective"]),
        "takes": {
            take_id: {
                "relative_path": f"{take_id.rpartition('_T')[0]}/{take_id}.zip",
                "sha256": "a" * 64,
                "bytes": 123,
            }
            for take_id in partitions["retrospective"]
        },
    }
    return build_public_transfer_protocol(
        calibration,
        freshness,
        inventory,
        archive_inventory_file_sha256="b" * 64,
        locked_at_utc="2026-08-05T12:00:00Z",
    )


def test_public_transfer_partition_is_exact_and_disjoint() -> None:
    calibration, freshness = _inputs()
    partitions = public_transfer_partitions(calibration, freshness)

    assert len(partitions["public"]) == 116
    assert len(partitions["source"]) == 36
    assert len(partitions["prospective"]) == 2
    assert len(partitions["retrospective"]) == 78
    assert set(partitions["prospective"]) == {"Pillow_T4", "PlushDice_T3"}
    assert not (set(partitions["source"]) & set(partitions["retrospective"]))


def test_public_transfer_protocol_rejects_resigned_cohort_change() -> None:
    protocol = _built_protocol()
    assert validate_public_transfer_protocol(
        protocol,
        bind_registered_digest=False,
    )["passed"] is True

    changed = deepcopy(protocol)
    changed["cohort"]["retrospective_take_ids"][0] = "Pillow_T4"
    changed["protocol_sha256"] = protocol_sha256(changed)
    with pytest.raises(ValueError, match="partition|cohort"):
        validate_public_transfer_protocol(changed, bind_registered_digest=False)


def test_frozen_public_transfer_protocol_and_archive_inventory_are_exact() -> None:
    protocol = json.loads(FROZEN_PROTOCOL.read_text(encoding="utf-8"))
    validation = validate_public_transfer_protocol(protocol)

    assert protocol["protocol_sha256"] == PROTOCOL_SHA256
    assert hashlib.sha256(FROZEN_PROTOCOL.read_bytes()).hexdigest() == (
        "b78fd58294656e548c7459e87889ce93f1447ef2633367d6b07852c1caaa4218"
    )
    assert hashlib.sha256(ARCHIVE_INVENTORY.read_bytes()).hexdigest() == (
        "428257e0915f27f09074d06b8871c2a739ce3413a7069cc29c2e080e12c7c057"
    )
    assert len(validation["retrospective_take_ids"]) == 78


def test_smoke_extraction_uses_fixed_scale_and_rejects_future_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _built_protocol()
    monkeypatch.setattr(
        audit_module,
        "PROTOCOL_SHA256",
        protocol["protocol_sha256"],
        raising=False,
    )
    take_id = protocol["cohort"]["retrospective_take_ids"][0]
    object_name = take_id.rpartition("_T")[0]
    scale = protocol["source_calibration"]["effective_scales"][object_name]
    global_key = f"checkpoint_{SOURCE_FIELD}_residual_scale_{BASE_EFFECTIVE_SCALE:g}"
    candidate_key = f"checkpoint_{SOURCE_FIELD}_residual_scale_{scale:g}"
    payload = {
        "artifact_kind": "PokeFlexCheckpointBayesianRegistrationDevelopmentSmoke",
        "public_transfer_protocol_sha256": protocol["protocol_sha256"],
        "legacy_runner_file_sha256": LEGACY_RUNNER_FILE_SHA256,
        "future_observation_used": False,
        "correction_fields": [SOURCE_FIELD],
        "take": {"id": take_id},
        "upstream": {"git_commit": audit_module.UPSTREAM_COMMIT},
        "aggregates": {
            "released_checkpoint": {"mean_CD_UL1_mm": 2.0},
            global_key: {"mean_CD_UL1_mm": 1.8},
            candidate_key: {"mean_CD_UL1_mm": 1.6},
        },
        "targets": [
            {
                "target_frame": 6,
                "released_checkpoint_CD_UL1_mm": 1.0,
                global_key: 0.8,
                candidate_key: 0.6,
            },
            {
                "target_frame": 7,
                "released_checkpoint_CD_UL1_mm": 3.0,
                global_key: 2.8,
                candidate_key: 2.6,
            },
        ],
    }
    row = take_row_from_smoke(payload, protocol)
    assert row["take_id"] == take_id
    assert row["candidate_CD_UL1_mm"] == pytest.approx(1.6)
    assert row["global_CD_UL1_mm"] == pytest.approx(1.8)

    payload["future_observation_used"] = True
    with pytest.raises(ValueError, match="future observation"):
        take_row_from_smoke(payload, protocol)


def test_summary_clusters_by_object_and_reports_losses() -> None:
    rows = [
        {
            "take_id": "A_T1",
            "object_name": "A",
            "checkpoint_CD_UL1_mm": 2.0,
            "global_CD_UL1_mm": 1.5,
            "candidate_CD_UL1_mm": 1.0,
            "frames": [
                {
                    "target_frame": 6,
                    "checkpoint_CD_UL1_mm": 2.0,
                    "global_CD_UL1_mm": 1.5,
                    "candidate_CD_UL1_mm": 1.0,
                }
            ],
        },
        {
            "take_id": "A_T2",
            "object_name": "A",
            "checkpoint_CD_UL1_mm": 4.0,
            "global_CD_UL1_mm": 3.0,
            "candidate_CD_UL1_mm": 2.0,
            "frames": [
                {
                    "target_frame": 6,
                    "checkpoint_CD_UL1_mm": 4.0,
                    "global_CD_UL1_mm": 3.0,
                    "candidate_CD_UL1_mm": 2.0,
                }
            ],
        },
        {
            "take_id": "B_T1",
            "object_name": "B",
            "checkpoint_CD_UL1_mm": 1.0,
            "global_CD_UL1_mm": 0.8,
            "candidate_CD_UL1_mm": 0.9,
            "frames": [
                {
                    "target_frame": 6,
                    "checkpoint_CD_UL1_mm": 1.0,
                    "global_CD_UL1_mm": 0.8,
                    "candidate_CD_UL1_mm": 0.9,
                }
            ],
        },
    ]
    summary = summarize_rows(rows)

    assert summary["object_count"] == 2
    assert summary["take_count"] == 3
    assert summary["candidate_vs_checkpoint"]["object_win_count"] == 2
    assert summary["candidate_vs_global"]["object_win_count"] == 1
    assert summary["candidate_vs_global"]["object_loss_count"] == 1
