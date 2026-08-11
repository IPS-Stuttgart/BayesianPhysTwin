from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

import scripts.remote.run_deform360_joint_sparse_physical_source_v5_selector_repair as repair

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json"


def _arguments(selector: Path) -> list[str]:
    return [
        str(repair.__file__),
        "--execution-repo",
        str(ROOT),
        "--execution-lock",
        str(LOCK),
        "--stage",
        "stage-prefix",
        "--protocol",
        str(LOCK),
        "--generic-selector-source",
        str(selector),
    ]


def test_selector_digest_is_repaired_only_after_original_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = tmp_path / "deform360_object_sam2.py"
    selector.write_bytes(b"x" * repair.CORRECTED_SELECTOR_BYTE_COUNT)
    events: list[str] = []
    module = SimpleNamespace(GENERIC_SELECTOR_SHA256=repair.PREVIOUS_SELECTOR_SHA256)

    def validate(*args: object, **kwargs: object) -> None:
        assert args[0] == LOCK
        assert kwargs["repository"] == ROOT
        events.append("validate")

    def load_stage(path: Path) -> SimpleNamespace:
        assert path == (ROOT / repair.STAGE_RELATIVE_PATH).resolve()
        events.append("load")
        return module

    def patch_stage(*args: object, **kwargs: object) -> None:
        assert args[0] is module
        assert kwargs["stage"] == "stage-prefix"
        events.append("patch")

    @contextmanager
    def active() -> Iterator[None]:
        events.append("activate")
        yield
        events.append("deactivate")

    def stage_main() -> int:
        assert module.GENERIC_SELECTOR_SHA256 == repair.CORRECTED_SELECTOR_SHA256
        assert sys.argv[0] == str((ROOT / repair.STAGE_RELATIVE_PATH).resolve())
        events.append("main")
        return 0

    module.main = stage_main
    monkeypatch.setattr(repair, "validate_joint_sparse_physical_execution_v5", validate)
    monkeypatch.setattr(repair, "_git_blob_sha1", lambda *args: repair.STAGE_GIT_BLOB_SHA1)
    monkeypatch.setattr(repair, "_load_stage", load_stage)
    monkeypatch.setattr(repair, "patch_joint_sparse_physical_stage_v5", patch_stage)
    monkeypatch.setattr(repair, "activate_joint_sparse_physical_runtime_v5", active)
    monkeypatch.setattr(repair, "file_sha256", lambda path: repair.CORRECTED_SELECTOR_SHA256)
    monkeypatch.setattr(sys, "argv", _arguments(selector))

    assert repair.main() == 0
    assert events == ["validate", "load", "activate", "patch", "main", "deactivate"]


def test_selector_repair_fails_closed_when_historical_stage_identity_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = tmp_path / "deform360_object_sam2.py"
    selector.write_bytes(b"x" * repair.CORRECTED_SELECTOR_BYTE_COUNT)
    monkeypatch.setattr(
        repair,
        "validate_joint_sparse_physical_execution_v5",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(repair, "_git_blob_sha1", lambda *args: "0" * 40)
    monkeypatch.setattr(repair, "file_sha256", lambda path: repair.CORRECTED_SELECTOR_SHA256)
    monkeypatch.setattr(sys, "argv", _arguments(selector))

    with pytest.raises(ValueError, match="checksum-bound prefix stage changed"):
        repair.main()


def test_selector_repair_fails_closed_before_loading_wrong_selector_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = tmp_path / "deform360_object_sam2.py"
    selector.write_bytes(b"x" * repair.CORRECTED_SELECTOR_BYTE_COUNT)
    loaded = False

    def load_stage(path: Path) -> SimpleNamespace:
        nonlocal loaded
        loaded = True
        return SimpleNamespace()

    monkeypatch.setattr(
        repair,
        "validate_joint_sparse_physical_execution_v5",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(repair, "file_sha256", lambda path: "f" * 64)
    monkeypatch.setattr(repair, "_load_stage", load_stage)
    monkeypatch.setattr(sys, "argv", _arguments(selector))

    with pytest.raises(ValueError, match="corrected selector bytes changed"):
        repair.main()
    assert loaded is False


def test_selector_repair_runner_has_no_target_or_future_interface() -> None:
    parser_source = Path(repair.__file__).read_text(encoding="utf-8")

    assert "--target" not in parser_source
    assert "--future" not in parser_source
    assert "--confirmation" not in parser_source
    assert "target_payload" not in parser_source
