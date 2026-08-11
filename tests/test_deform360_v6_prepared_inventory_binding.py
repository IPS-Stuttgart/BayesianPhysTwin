from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
ARCHIVED_RUNNER = ROOT / (
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
REVISION_ARGUMENT = '--implementation-revision "${BPT_SOURCE_SHA}"'


def _stage_block(text: str, start_marker: str, end_marker: str) -> str:
    assert text.count(start_marker) == 1
    assert text.count(end_marker) == 1
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def _python_shim(text: str) -> str:
    start_marker = 'cat > "${PYTHON_SHIM}" <<\'SH\'\n'
    end_marker = '\nSH\nchmod 700 "${PYTHON_SHIM}"'
    assert text.count(start_marker) == 1
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker, start)
    return text[start:end] + "\n"


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


def test_archived_revision_bindings_are_validated_per_stage() -> None:
    archived = ARCHIVED_RUNNER.read_text(encoding="utf-8")
    inventory = _stage_block(
        archived,
        'set_stage "materialize-prepared-source-inventory"\n',
        'set_stage "locate-frozen-sam2-checkpoint"\n',
    )
    source_plan = _stage_block(
        archived,
        'set_stage "materialize-source-plan"\n',
        'set_stage "generate-nested-source-predictions"\n',
    )

    assert archived.count(REVISION_ARGUMENT) == 2
    assert inventory.count(
        "scripts/science/inventory_deform360_calibration_prepared_source.py"
    ) == 1
    assert inventory.count(REVISION_ARGUMENT) == 1
    assert source_plan.count(
        "scripts/science/materialize_deform360_v6_source_plan_inputs.py"
    ) == 1
    assert source_plan.count(REVISION_ARGUMENT) == 1

    runner = RUNNER.read_text(encoding="utf-8")
    assert "inventory_start_marker =" in runner
    assert "source_plan_start_marker =" in runner
    assert "runner.count(needle) != 2" in runner
    assert "archived source revision binding roster changed" in runner
    assert "archived prepared-inventory implementation binding changed" in runner
    assert "archived source-plan implementation binding changed" in runner


def test_runtime_shim_rewrites_only_the_inventory_generator_revision(
    tmp_path: Path,
) -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    shim = tmp_path / "python-shim"
    shim.write_text(_python_shim(runner), encoding="utf-8")
    shim.chmod(0o700)

    capture = tmp_path / "arguments.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\nprintf \'%s\\n\' "$@" > "${CAPTURE}"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    env = {
        **os.environ,
        "CAPTURE": str(capture),
        "REAL_BPT_PYTHON": str(fake_python),
        "PREPARED_INVENTORY_IMPLEMENTATION_REVISION": (
            AUTHORITATIVE_IMPLEMENTATION_REVISION
        ),
    }

    inventory_target = (
        "scripts/science/inventory_deform360_calibration_prepared_source.py"
    )
    subprocess.run(
        [
            str(shim),
            inventory_target,
            "--implementation-revision",
            "runtime-head",
            "--output",
            "inventory.json",
        ],
        check=True,
        env=env,
    )
    assert capture.read_text(encoding="utf-8").splitlines() == [
        inventory_target,
        "--implementation-revision",
        AUTHORITATIVE_IMPLEMENTATION_REVISION,
        "--output",
        "inventory.json",
    ]

    source_plan_target = "scripts/science/materialize_deform360_v6_source_plan_inputs.py"
    subprocess.run(
        [
            str(shim),
            source_plan_target,
            "--implementation-revision",
            "runtime-head",
            "--output",
            "source-plan.json",
        ],
        check=True,
        env=env,
    )
    assert capture.read_text(encoding="utf-8").splitlines() == [
        source_plan_target,
        "--implementation-revision",
        "runtime-head",
        "--output",
        "source-plan.json",
    ]

    assert (
        'target="scripts/science/inventory_deform360_calibration_prepared_source.py"'
        in runner
    )
    assert 'rewritten+=("$1" "${PREPARED_INVENTORY_IMPLEMENTATION_REVISION}")' in runner
    assert 'if [[ "${replacements}" -ne 1 ]]' in runner
    assert 'exec "${REAL_BPT_PYTHON}" "$@"' in runner
    assert 'BPT_PYTHON="${PYTHON_SHIM}"' in runner
    assert '"runtime_prepared_inventory_identity"' in runner
