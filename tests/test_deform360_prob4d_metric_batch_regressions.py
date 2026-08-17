from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPERS_PATH = ROOT / "tests/test_deform360_prob4d_metric_batch.py"
SPEC = importlib.util.spec_from_file_location(
    "deform360_prob4d_metric_batch_test_helpers", HELPERS_PATH
)
assert SPEC is not None and SPEC.loader is not None
helpers = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = helpers
SPEC.loader.exec_module(helpers)

metric_batch = helpers.metric_batch


def _rewrite_batch_integrity(output: Path) -> None:
    result_path = output / helpers.METRIC_BATCH_RESULT_FILENAME
    plan_path = output / helpers.METRIC_PREFIX_PLAN_FILENAME
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["plan_file"] = metric_batch._file_record(plan_path, root=output)
    identity = {key: value for key, value in result.items() if key != "result_id"}
    result["result_id"] = metric_batch.content_id(identity)
    metric_batch._write_json(result_path, result)
    metric_batch._write_checksums(output)


def test_metric_batch_validator_binds_plan_to_declared_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = helpers._fixture(tmp_path)
    helpers._install_contract_stubs(monkeypatch)
    helpers._install_metric_stub(monkeypatch)
    helpers.materialize_deform360_prob4d_metric_batch(**arguments)

    output = Path(arguments["output_directory"])
    plan_path = output / helpers.METRIC_PREFIX_PLAN_FILENAME
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["visual_production_result_id"] = "a" * 64
    identity = {key: value for key, value in plan.items() if key != "plan_id"}
    plan["plan_id"] = metric_batch.content_id(identity)
    metric_batch._write_json(plan_path, plan)
    _rewrite_batch_integrity(output)

    with pytest.raises(ValueError, match="different production result"):
        helpers.validate_deform360_prob4d_metric_batch(output)


def test_metric_batch_does_not_relabel_plan_contract_error_as_multiview_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = helpers._fixture(tmp_path)
    helpers._install_contract_stubs(monkeypatch)
    helpers._install_metric_stub(monkeypatch)
    selection_path = Path(arguments["selection_path"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["dataset"]["resolved_revision"] = "not-a-revision"
    helpers._write_json(selection_path, selection)

    with pytest.raises(ValueError):
        helpers.materialize_deform360_prob4d_metric_batch(**arguments)

    assert not Path(arguments["output_directory"]).exists()
