from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
SCIENCE_RUNNER = ROOT / (
    "scripts/ci/archive/run_deform360_v6_source_prediction_evidence_v2.sh"
)
LOCK = (
    ROOT
    / "protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json"
)

AUTHORITATIVE_ADMISSION_RUN_ID = "31272512658"
AUTHORITATIVE_IMPLEMENTATION_REVISION = "e190c94014e6024e324d860618662526af6ea682"
PREPARED_INVENTORY_ID = (
    "6994aa621b38dc8fb21cd38e43363bde3ea12dd644532addeecfc07a30f84e7b"
)
PREPARED_INVENTORY_FILE_SHA256 = (
    "4da96c4f636d195f7aea5d971fbd83bd3b0f35b1c66a77af68007bbd08a69007"
)


def test_runtime_reuses_the_frozen_prepared_inventory_identity() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    assert lock["physical_baseline"]["prepared_source_inventory"] == {
        "file_sha256": PREPARED_INVENTORY_FILE_SHA256,
        "inventory_id": PREPARED_INVENTORY_ID,
    }

    runner = RUNNER.read_text(encoding="utf-8")
    assert (
        f'PREPARED_INVENTORY_IMPLEMENTATION_REVISION="'
        f'{AUTHORITATIVE_IMPLEMENTATION_REVISION}"'
    ) in runner
    assert f'PREPARED_INVENTORY_ID="{PREPARED_INVENTORY_ID}"' in runner
    assert (
        f'PREPARED_INVENTORY_FILE_SHA256="{PREPARED_INVENTORY_FILE_SHA256}"' in runner
    )
    assert (
        f'PREPARED_INVENTORY_ADMISSION_RUN_ID="{AUTHORITATIVE_ADMISSION_RUN_ID}"'
        in runner
    )


def test_runtime_rewrites_only_the_inventory_generator_revision() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    science = SCIENCE_RUNNER.read_text(encoding="utf-8")
    inventory_pattern = re.compile(
        r"scripts/science/inventory_deform360_calibration_prepared_source\.py"
        r"[\s\S]*?--implementation-revision \"\$\{BPT_SOURCE_SHA\}\""
        r"[\s\S]*?--output \"\$\{RUN_ROOT\}/prepared-source-inventory\.json\""
    )

    assert science.count('--implementation-revision "${BPT_SOURCE_SHA}"') == 2
    assert len(inventory_pattern.findall(science)) == 1
    assert (
        'target="scripts/science/inventory_deform360_calibration_prepared_source.py"'
        in runner
    )
    assert "inventory_pattern = re.compile(" in runner
    assert "len(inventory_pattern.findall(runner)) != 1" in runner
    assert 'rewritten+=("$1" "${PREPARED_INVENTORY_IMPLEMENTATION_REVISION}")' in runner
    assert 'if [[ "${replacements}" -ne 1 ]]' in runner
    assert 'exec "${REAL_BPT_PYTHON}" "$@"' in runner
    assert 'BPT_PYTHON="${PYTHON_SHIM}"' in runner
    assert '"runtime_prepared_inventory_identity"' in runner
