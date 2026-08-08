"""Information-order and publication boundaries for observability batches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import test_deform360_calibration_observability_batch as batch_cases


def test_query_bytes_follow_first_case_lineage_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = batch_cases._batch_inputs(tmp_path / "inputs")
    spec = batch_cases._write_spec(tmp_path / "batch-spec.json")
    output = tmp_path / "published"
    args = batch_cases.CLI.build_parser().parse_args(
        batch_cases._arguments(inputs, spec=spec, output=output)
    )
    query = inputs.query_path.resolve()
    original_read = Path.read_bytes

    def guarded_read(path: Path) -> bytes:
        if path.resolve() == query:
            raise AssertionError("query bytes opened before source-lineage validation")
        return original_read(path)

    def reject_lineage(**_kwargs: Any) -> None:
        raise ValueError("source lineage rejected")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", guarded_read)
        patch.setattr(
            batch_cases.CLI,
            "build_evaluated_case_from_paths",
            reject_lineage,
        )
        with pytest.raises(ValueError, match="source lineage rejected"):
            batch_cases.CLI._run(args)
    assert not output.exists()


def test_dangling_destination_symlink_is_not_replaced(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    output = tmp_path / "published"
    target = tmp_path / "missing-target"
    output.symlink_to(target, target_is_directory=True)

    with pytest.raises(FileExistsError):
        batch_cases.CLI._publish(staged, output)

    assert output.is_symlink()
    assert staged.is_dir()
    assert not target.exists()
    assert not (tmp_path / ".published.publish.lock").exists()
