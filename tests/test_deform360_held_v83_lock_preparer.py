from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


SOURCE = (
    Path(__file__).parents[1]
    / "scripts"
    / "held"
    / "prepare_deform360_v83_lock.py"
)
SPEC = importlib.util.spec_from_file_location("deform360_v83_lock_preparer", SOURCE)
assert SPEC is not None and SPEC.loader is not None
preparer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preparer)


def _lineage() -> dict[str, Any]:
    return {
        "process_isolation_qualification_attempt": {
            "path": "/qualified/attempt.json",
            "sha256": "1" * 64,
            "size_bytes": 1,
            "artifact_sha256": "2" * 64,
        },
        "process_isolation_qualification_evidence": {
            "path": "/qualified/evidence.json",
            "sha256": "3" * 64,
            "size_bytes": 1,
            "artifact_sha256": "4" * 64,
        },
        "process_isolation_qualification_integrity_completion": {
            "path": "/qualified/completion.json",
            "sha256": "5" * 64,
            "size_bytes": 1,
            "artifact_sha256": "6" * 64,
        },
        "process_isolation_qualification_integrity": {
            "source_head": "a" * 40,
            "source_tree": "b" * 40,
            "terminal_outcome": "qualified",
            "admission_eligible": True,
            "inventory_sha256": "7" * 64,
            "metadata_inventory_sha256": "8" * 64,
            "qualification_source_sha256": "9" * 64,
            "numerical_adapter_source_sha256": "a" * 64,
            "isolation_source_sha256": "b" * 64,
            "worker_source_sha256": "c" * 64,
            "worker_runtime_source_sha256": "f" * 64,
            "outcome_driver_source_sha256": "d" * 64,
            "sealer_source_sha256": "e" * 64,
        },
    }


def test_qualification_paths_are_revision_specific() -> None:
    head = "a" * 40
    root, evidence, completion = preparer.qualification_paths(head)

    assert root == Path(
        f"/mnt/corsair/florianpfaff/bpt-process-isolation-qualification-{head}"
    )
    assert evidence == root / "process-isolation-qualification.json"
    assert completion == Path(f"{root}-integrity-completion.json")
    with pytest.raises(ValueError, match="revision is invalid"):
        preparer.qualification_paths("not-a-revision")


def test_external_bindings_exclude_attempt5_placeholders() -> None:
    observed: list[str] = []
    support = SimpleNamespace(
        _EXPECTED_EXTERNAL_FILES={
            "stable": (Path("/stable"), "a" * 64, 0o400),
            "v8_external_admission_metadata_only_replay": (
                Path("/never"),
                None,
                0o400,
            ),
            "v8_external_admission_replay_code_binding": (
                Path("/never-either"),
                None,
                0o400,
            ),
        },
        _EXPECTED_EXTERNAL_ARTIFACT_SHA256={},
        _valid_sha256=lambda value: isinstance(value, str) and len(value) == 64,
        _sha256_file=lambda path, **_kwargs: (
            observed.append(str(path)) or "a" * 64
        ),
        _validate_pinned_python=lambda: "b" * 64,
    )

    bindings = preparer._external_bindings(support)

    assert bindings == {
        "stable": "a" * 64,
        "pinned_python_executable_target": "b" * 64,
    }
    assert observed == ["/stable"]


def test_qualification_must_bind_exact_deployed_worker_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = tmp_path / "code"
    source = code / preparer.SOURCE_RELATIVE
    source.parent.mkdir(parents=True)
    source.write_text("# preparer\n", encoding="utf-8")
    monkeypatch.setattr(preparer, "__file__", str(source))
    lineage = _lineage()
    qualification = SimpleNamespace(
        validate_process_isolation_qualification_lineage=lambda **_kwargs: lineage
    )
    local = {
        binding_name: lineage[
            "process_isolation_qualification_integrity"
        ][integrity_name]
        for integrity_name, binding_name in preparer.QUALIFIED_SOURCE_MAP.items()
    }

    accepted, _root, _evidence, _completion = preparer._validate_qualification(
        code=code,
        head="a" * 40,
        qualification=qualification,
        local_bindings=local,
    )
    assert accepted == lineage

    local["held_v83_process_isolation_worker_source"] = "f" * 64
    with pytest.raises(
        ValueError,
        match="held_v83_process_isolation_worker_source differs",
    ):
        preparer._validate_qualification(
            code=code,
            head="a" * 40,
            qualification=qualification,
            local_bindings=local,
        )
