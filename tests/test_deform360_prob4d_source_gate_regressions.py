from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPERS_PATH = ROOT / "tests/test_deform360_prob4d_source_gate.py"
SPEC = importlib.util.spec_from_file_location(
    "deform360_prob4d_source_gate_test_helpers", HELPERS_PATH
)
assert SPEC is not None and SPEC.loader is not None
helpers = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = helpers
SPEC.loader.exec_module(helpers)

source_gate = helpers.source_gate


def _rewrite_result_integrity(output: Path, result: dict[str, object]) -> None:
    result_path = output / source_gate.SOURCE_GATE_RESULT_FILENAME
    identity = {key: value for key, value in result.items() if key != "result_id"}
    result["result_id"] = source_gate.content_id(identity)
    source_gate._write_json(result_path, result)
    paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{source_gate._sha256_file(path)}  {path.relative_to(output).as_posix()}\n"
            for path in paths
        ),
        encoding="ascii",
    )


def test_validator_rejects_rehashed_false_gate_relabelled_as_passed(
    tmp_path: Path,
) -> None:
    samples = helpers._synthetic_samples(adversarial_objects=(0, 1, 2))
    source_root = tmp_path / "source-calibration"
    source_result = helpers._source_calibration_result(source_root, samples)
    output = tmp_path / "gate"
    published = source_gate.publish_source_gate_result(
        samples=samples,
        source_calibration_result_path=source_result,
        source_calibration_root=source_root,
        gate_lock_path=helpers.LOCK,
        implementation_revision="c" * 40,
        output_directory=output,
    )
    assert published["gate_passed"] is False

    result_path = output / source_gate.SOURCE_GATE_RESULT_FILENAME
    forged = json.loads(result_path.read_text(encoding="utf-8"))
    forged["checks"] = {name: True for name in forged["checks"]}
    forged["passed_check_count"] = len(forged["checks"])
    forged["total_check_count"] = len(forged["checks"])
    forged["gate_passed"] = True
    forged["confirmation_access_authorized"] = True
    forged["status"] = "source-gate-passed"
    _rewrite_result_integrity(output, forged)

    with pytest.raises(
        ValueError, match="checks differ from stored decision evidence"
    ):
        source_gate.validate_source_gate_result(output)
