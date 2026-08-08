#!/usr/bin/env python3
"""Apply the reviewed Deform360 result file/semantic identity repair."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    source = target.read_text(encoding="utf-8")
    count = source.count(old)
    if count != 1:
        raise SystemExit(
            f"expected exactly one source fragment in {path}, observed {count}"
        )
    target.write_text(source.replace(old, new), encoding="utf-8")


def main() -> None:
    replace_once(
        "scripts/science/inventory_deform360_calibration_prepared_source.py",
        '''INVENTORY_VERSION = 1
INVENTORY_SEMANTICS = "exact-retained-calibration-rgb-tactile-robot-inventory-v1"
''',
        '''INVENTORY_VERSION = 2
INVENTORY_SEMANTICS = "exact-retained-calibration-rgb-tactile-robot-inventory-v2"
''',
    )
    replace_once(
        "scripts/science/inventory_deform360_calibration_prepared_source.py",
        '''    _compare_summary(result, record, keys=_RESULT_KEYS, name="result")
    result_value, result_file_sha256 = load_json_object(result_path)
''',
        '''    _compare_summary(result, record, keys=_RESULT_KEYS, name="result")
    result_sha256 = sha256_digest(
        result.get("result_sha256"),
        name="calibration source result semantic SHA-256",
    )
    result_value, result_file_sha256 = load_json_object(result_path)
''',
    )
    replace_once(
        "scripts/science/inventory_deform360_calibration_prepared_source.py",
        '''        "calibration_source_run_record_sha256": sha256_digest(
            record.get("record_sha256"),
            name="calibration source record SHA-256",
        ),
        "object_count": len(objects),
''',
        '''        "calibration_source_run_record_sha256": sha256_digest(
            record.get("record_sha256"),
            name="calibration source record SHA-256",
        ),
        "calibration_source_result_sha256": result_sha256,
        "calibration_source_result_file_sha256": result_file_sha256,
        "object_count": len(objects),
''',
    )

    replace_once(
        "src/bayesian_phystwin/deform360_calibration_visual_execution_admission.py",
        '''DEFORM360_PREPARED_SOURCE_INVENTORY_VERSION: Final = 1
DEFORM360_PREPARED_SOURCE_INVENTORY_SEMANTICS: Final = (
    "exact-retained-calibration-rgb-tactile-robot-inventory-v1"
)
''',
        '''DEFORM360_PREPARED_SOURCE_INVENTORY_VERSION: Final = 2
DEFORM360_PREPARED_SOURCE_INVENTORY_SEMANTICS: Final = (
    "exact-retained-calibration-rgb-tactile-robot-inventory-v2"
)
''',
    )
    replace_once(
        "src/bayesian_phystwin/deform360_calibration_visual_execution_admission.py",
        '''        "visual_provider_lock_id",
        "calibration_source_run_record_sha256",
        "object_count",
''',
        '''        "visual_provider_lock_id",
        "calibration_source_run_record_sha256",
        "calibration_source_result_sha256",
        "calibration_source_result_file_sha256",
        "object_count",
''',
    )
    replace_once(
        "src/bayesian_phystwin/deform360_calibration_visual_execution_admission.py",
        '''        "selection_artifact_sha256",
        "visual_provider_lock_id",
        "calibration_source_run_record_sha256",
    ):
        sha256_digest(inventory[field], name=field)
''',
        '''        "selection_artifact_sha256",
        "visual_provider_lock_id",
        "calibration_source_run_record_sha256",
        "calibration_source_result_sha256",
        "calibration_source_result_file_sha256",
    ):
        sha256_digest(inventory[field], name=field)
''',
    )
    replace_once(
        "src/bayesian_phystwin/deform360_calibration_visual_execution_admission.py",
        '''    required_result_path = "sources/calibration-source/result.json"
    if required_result_path not in source_artifacts:
        raise ValueError("inventory does not bind the calibration-source result")

    objects = _sequence(inventory["objects"], name="inventory objects")
''',
        '''    required_result_path = "sources/calibration-source/result.json"
    if required_result_path not in source_artifacts:
        raise ValueError("inventory does not bind the calibration-source result")
    result_file_sha256 = sha256_digest(
        source_artifacts[required_result_path],
        name="inventory calibration-source result file",
    )
    if inventory["calibration_source_result_file_sha256"] != result_file_sha256:
        raise ValueError(
            "prepared-source inventory calibration-source result file differs"
        )

    objects = _sequence(inventory["objects"], name="inventory objects")
''',
    )
    replace_once(
        "src/bayesian_phystwin/deform360_calibration_visual_execution_admission.py",
        '''    inventory_sources = cast(Mapping[str, Any], inventory["source_artifacts"])
    result_sha256 = sha256_digest(
        inventory_sources["sources/calibration-source/result.json"],
        name="inventory calibration-source result",
    )
    if plan["calibration_source_result_sha256"] != result_sha256:
        raise ValueError("plan and inventory differ: calibration-source result")
''',
        '''    inventory_sources = cast(Mapping[str, Any], inventory["source_artifacts"])
    result_file_sha256 = sha256_digest(
        inventory["calibration_source_result_file_sha256"],
        name="inventory calibration-source result file identity",
    )
    if (
        inventory_sources["sources/calibration-source/result.json"]
        != result_file_sha256
    ):
        raise ValueError("plan and inventory differ: calibration-source result file")
    result_sha256 = sha256_digest(
        inventory["calibration_source_result_sha256"],
        name="inventory calibration-source result semantic identity",
    )
    if plan["calibration_source_result_sha256"] != result_sha256:
        raise ValueError("plan and inventory differ: calibration-source result")
''',
    )

    replace_once(
        "tests/test_deform360_calibration_visual_execution_admission.py",
        '''        "schema_version": 1,
        "semantics": "exact-retained-calibration-rgb-tactile-robot-inventory-v1",
''',
        '''        "schema_version": 2,
        "semantics": "exact-retained-calibration-rgb-tactile-robot-inventory-v2",
''',
    )
    replace_once(
        "tests/test_deform360_calibration_visual_execution_admission.py",
        '''        "calibration_source_run_record_sha256": plan[
            "calibration_source_run_record_sha256"
        ],
        "object_count": 10,
''',
        '''        "calibration_source_run_record_sha256": plan[
            "calibration_source_run_record_sha256"
        ],
        "calibration_source_result_sha256": plan[
            "calibration_source_result_sha256"
        ],
        "calibration_source_result_file_sha256": _digest("result-file"),
        "object_count": 10,
''',
    )
    replace_once(
        "tests/test_deform360_calibration_visual_execution_admission.py",
        '''        "source_artifacts": {
            "sources/calibration-source/result.json": plan[
                "calibration_source_result_sha256"
            ],
            "sources/stage0/selection.json": plan["selection_artifact_sha256"],
        },
''',
        '''        "source_artifacts": {
            "sources/calibration-source/result.json": _digest("result-file"),
            "sources/stage0/selection.json": plan["selection_artifact_sha256"],
        },
''',
    )
    replace_once(
        "tests/test_deform360_calibration_visual_execution_admission.py",
        '''

def test_admission_round_trip_is_deterministic_and_non_replacing(
''',
        '''

def test_admission_binds_semantic_and_file_result_identities_separately(
    tmp_path: Path,
) -> None:
    plan_path, inventory_path, plan = _inputs(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    result_file_sha256 = inventory["source_artifacts"][
        "sources/calibration-source/result.json"
    ]
    assert result_file_sha256 == inventory[
        "calibration_source_result_file_sha256"
    ]
    assert result_file_sha256 != plan["calibration_source_result_sha256"]

    admission = build_deform360_calibration_visual_execution_admission(
        visual_production_plan_path=plan_path,
        prepared_source_inventory_path=inventory_path,
        implementation_revision=IMPLEMENTATION_REVISION,
    )

    assert admission["calibration_source_result_sha256"] == plan[
        "calibration_source_result_sha256"
    ]


def test_admission_round_trip_is_deterministic_and_non_replacing(
''',
    )
    replace_once(
        "tests/test_deform360_calibration_visual_execution_admission.py",
        '''

def test_duplicate_keys_boolean_versions_and_tampering_are_rejected(
''',
        '''

    plan_path, inventory_path, _plan = _inputs(tmp_path / "result")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["calibration_source_result_sha256"] = "d" * 64
    _rewrite_inventory(inventory_path, inventory)
    with pytest.raises(ValueError, match="calibration-source result"):
        build_deform360_calibration_visual_execution_admission(
            visual_production_plan_path=plan_path,
            prepared_source_inventory_path=inventory_path,
            implementation_revision=IMPLEMENTATION_REVISION,
        )


def test_duplicate_keys_boolean_versions_and_tampering_are_rejected(
''',
    )

    replace_once(
        "tests/test_deform360_calibration_prepared_inventory.py",
        '''    value = json.loads(output.read_text(encoding="utf-8"))

    assert value["object_count"] == 10
''',
        '''    value = json.loads(output.read_text(encoding="utf-8"))

    result_file_sha256 = value["source_artifacts"][
        "sources/calibration-source/result.json"
    ]
    assert value["schema_version"] == 2
    assert value["semantics"] == (
        "exact-retained-calibration-rgb-tactile-robot-inventory-v2"
    )
    assert value["calibration_source_result_sha256"] == inputs.chain.result[
        "result_sha256"
    ]
    assert value["calibration_source_result_file_sha256"] == _sha256(
        inputs.chain.result_path
    )
    assert result_file_sha256 == value["calibration_source_result_file_sha256"]
    assert result_file_sha256 != value["calibration_source_result_sha256"]
    assert value["object_count"] == 10
''',
    )

    replace_once(
        "docs/deform360_calibration_prepared_inventory.md",
        '''The content-addressed inventory contains one row per exact calibration object,
the exact source artifact digests, the action-selected window, all camera media
contracts, tactile array contracts, robot array contracts, and the closed
information boundary.
''',
        '''The content-addressed inventory contains one row per exact calibration object,
the exact source-file digests, the semantic calibration-source result identity,
the action-selected window, all camera media contracts, tactile array contracts,
robot array contracts, and the closed information boundary. Inventory schema
version 2 deliberately carries explicit `calibration_source_result_file_sha256`
and `calibration_source_result_sha256` fields; they are different identities and
neither substitutes for the other.
''',
    )
    replace_once(
        ".github/workflows/launch-deform360-calibration-retained-source-once.yml",
        "# Reviewed retained-source admission request: 2026-08-09-setup-python-v3\n",
        "# Reviewed retained-source admission request: 2026-08-09-result-identity-v4\n",
    )


if __name__ == "__main__":
    main()
