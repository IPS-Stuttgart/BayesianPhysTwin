from __future__ import annotations

import argparse
import json
import runpy
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bayesian_phystwin import deform360_calibration_execution as execution
from bayesian_phystwin.cli import deform360_calibration_execution as cli
from bayesian_phystwin.deform360_calibration_bundle import (
    DEFORM360_CALIBRATION_ROLES,
    Deform360CalibrationArtifactRefV1,
)
from bayesian_phystwin.deform360_visual_provider_lock import (
    Deform360VisualProviderLockV1,
)
from bayesian_phystwin.evidence_use_ledger import (
    EvidenceUseLedgerV1,
    EvidenceUseV1,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _stage0_path() -> Path:
    return (
        _repository_root()
        / "protocols"
        / "locks"
        / "deform360_official_hub_visuotactile_v1_selection.json"
    )


def _stage0() -> execution.Deform360Stage0SelectionV1:
    return execution.load_deform360_stage0_selection(_stage0_path())


def _provider() -> Deform360VisualProviderLockV1:
    return Deform360VisualProviderLockV1(
        provider_revision="1" * 40,
        provider_manifest_id="2" * 64,
        provider_attestation_sha256="3" * 64,
        motioncrafter_revision="4" * 40,
        model_set_id="5" * 64,
        root_seed=20260805,
        seed_policy="per-object-derived-seed-v1",
        window_size=25,
        overlap=8,
        height=320,
        width=640,
        storage_dtype="float32",
        initial_metric_frame_prior_id="6" * 64,
        additional_metric_anchor_policy="none",
        max_gauge_rank=64,
        minimum_retained_gauge_trace=0.999,
    )


def _artifacts(
    stage0: execution.Deform360Stage0SelectionV1,
    *,
    implementation_revision: str = "c" * 40,
) -> tuple[Deform360CalibrationArtifactRefV1, ...]:
    groups = tuple(unit.object_id for unit in stage0.calibration_units)
    return tuple(
        Deform360CalibrationArtifactRefV1(
            role=role,
            artifact_id=f"{index + 1:064x}",
            implementation_revision=implementation_revision,
            selection_evidence_id=f"{index + 101:064x}",
            selected_candidate_id=f"candidate-{index}",
            candidate_count=index + 2,
            calibration_group_ids=groups,
            source_artifacts={f"calibration/{role}.json": f"{index + 201:064x}"},
        )
        for index, role in enumerate(DEFORM360_CALIBRATION_ROLES)
    )


def _entry(
    unit: Any,
    *,
    index: int,
    metadata: dict[str, Any] | None = None,
) -> EvidenceUseV1:
    return EvidenceUseV1(
        evidence_artifact_id=f"{index + 301:064x}",
        raw_factor_id=f"{index + 401:064x}",
        raw_factor_sha256=f"{index + 501:064x}",
        source_repository="brownu/deform360",
        source_revision="b" * 40,
        source_artifacts={unit.metadata_path: unit.metadata_sha256},
        sensor_family="deform360-calibration-source",
        stream_id=f"{unit.object_id}:episode-{unit.episode_id}",
        clock_id="deform360-frame-clock",
        causal_frame_start=0,
        causal_frame_stop=10,
        correlation_group_ids=(f"group-{index}",),
        inference_role="calibration_only",
        metadata=metadata or {"object_id": unit.object_id},
    )


def _ledger(
    stage0: execution.Deform360Stage0SelectionV1,
    *,
    protocol_id: str | None = None,
    entries: tuple[EvidenceUseV1, ...] | None = None,
) -> EvidenceUseLedgerV1:
    selected = entries
    if selected is None:
        selected = tuple(
            _entry(unit, index=index)
            for index, unit in enumerate(stage0.calibration_units)
        )
    return EvidenceUseLedgerV1(
        protocol_id=protocol_id or stage0.protocol_id,
        case_id=execution.DEFORM360_CALIBRATION_LEDGER_CASE_ID,
        causal_frame_stop=10,
        entries=selected,
    )


def _sources(stage0: execution.Deform360Stage0SelectionV1) -> dict[str, str]:
    result = {
        "sources/stage0/selection.json": stage0.source_sha256,
        "sources/locks/visual-provider-lock.json": "1" * 64,
        "sources/calibration/evidence-use-ledger.json": "2" * 64,
    }
    for index, role in enumerate(DEFORM360_CALIBRATION_ROLES):
        result[f"sources/calibration/artifacts/{role}.json"] = f"{index + 10:064x}"
    return result


def _products() -> execution.Deform360CalibrationExecutionArtifactsV1:
    stage0 = _stage0()
    return execution.build_deform360_calibration_execution_seal(
        stage0_selection=stage0,
        visual_provider_lock=_provider(),
        evidence_use_ledger=_ledger(stage0),
        calibration_artifacts=_artifacts(stage0),
        implementation_revision="c" * 40,
        source_artifacts=_sources(stage0),
    )


def test_file_and_cohort_helpers_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ordinary file"):
        execution.file_sha256(tmp_path / "missing")

    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="ordinary file"):
        execution.file_sha256(link)

    stage0 = _stage0()
    with pytest.raises(ValueError, match="must be a sequence"):
        replace(stage0, calibration_units="not-a-sequence")
    with pytest.raises(ValueError, match="must contain"):
        replace(
            stage0,
            calibration_units=(object(), *stage0.calibration_units[1:]),
        )
    duplicate = (*stage0.calibration_units[:-1], stage0.calibration_units[0])
    with pytest.raises(ValueError, match="repeats an object"):
        replace(stage0, calibration_units=duplicate)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unit", "must be a JSON object"),
        ("boundary", "information_boundary must be an object"),
        ("dataset", "dataset must be an object"),
        ("dataset_repository", "dataset repository changed"),
        ("raw_prefix", "raw prefix changed"),
        ("processing", "official_processing must be an object"),
        ("processing_repository", "processing repository changed"),
        ("roles", "selection roles changed"),
        ("calibration_array", "calibration selection must be an array"),
        ("confirmation_array", "confirmation selection must be an array"),
    ],
)
def test_stage0_nested_contract_rejections(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    payload = json.loads(_stage0_path().read_text(encoding="utf-8"))
    if case == "unit":
        payload["selection"]["calibration"][0] = 1
    elif case == "boundary":
        payload["information_boundary"] = []
    elif case == "dataset":
        payload["dataset"] = []
    elif case == "dataset_repository":
        payload["dataset"]["repo_id"] = "changed/repository"
    elif case == "raw_prefix":
        payload["dataset"]["raw_prefix"] = "changed"
    elif case == "processing":
        payload["official_processing"] = []
    elif case == "processing_repository":
        payload["official_processing"]["repository"] = "changed/repository"
    elif case == "roles":
        payload["selection"]["extra"] = []
    elif case == "calibration_array":
        payload["selection"]["calibration"] = {}
    elif case == "confirmation_array":
        payload["selection"]["confirmation"] = {}
    path = tmp_path / f"{case}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        execution.load_deform360_stage0_selection(path)


def test_component_and_ledger_private_boundaries() -> None:
    with pytest.raises(ValueError, match="must be a sequence"):
        execution.deform360_calibration_component_ids("bad")

    with pytest.raises(ValueError, match="object_id and object_ids"):
        execution._entry_object_ids(  # noqa: SLF001
            {"object_id": "a", "object_ids": ["a"]}
        )
    with pytest.raises(ValueError, match="must identify"):
        execution._entry_object_ids({})  # noqa: SLF001
    with pytest.raises(ValueError, match="must identify"):
        execution._entry_object_ids({"object_ids": "a"})  # noqa: SLF001
    with pytest.raises(ValueError, match="must be a sequence"):
        execution._entry_object_ids({"object_ids": 1})  # noqa: SLF001
    assert execution._entry_object_ids(  # noqa: SLF001
        {"object_ids": ["b", "a"]}
    ) == {"a", "b"}

    stage0 = _stage0()
    with pytest.raises(TypeError, match="EvidenceUseLedgerV1"):
        execution._validate_calibration_ledger(object(), stage0)  # type: ignore[arg-type]  # noqa: SLF001
    with pytest.raises(ValueError, match="protocol changed"):
        execution._validate_calibration_ledger(  # noqa: SLF001
            _ledger(stage0, protocol_id="changed-protocol"),
            stage0,
        )
    with pytest.raises(ValueError, match="must contain evidence"):
        execution._validate_calibration_ledger(  # noqa: SLF001
            _ledger(stage0, entries=()),
            stage0,
        )

    foreign = _entry(
        stage0.calibration_units[0],
        index=99,
        metadata={"object_id": "foreign-object"},
    )
    remaining = tuple(
        _entry(unit, index=index)
        for index, unit in enumerate(stage0.calibration_units[1:], start=1)
    )
    with pytest.raises(ValueError, match="outside the Stage-0"):
        execution._validate_calibration_ledger(  # noqa: SLF001
            _ledger(stage0, entries=(foreign, *remaining)),
            stage0,
        )

    all_objects = [unit.object_id for unit in stage0.calibration_units]
    pooled = _entry(
        stage0.calibration_units[0],
        index=100,
        metadata={"object_ids": all_objects},
    )
    execution._validate_calibration_ledger(  # noqa: SLF001
        _ledger(stage0, entries=(pooled,)),
        stage0,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "changed", "schema changed"),
        ("schema_version", 2, "schema_version changed"),
        ("semantics", "changed", "semantics changed"),
        ("claim_boundary", "changed", "claim boundary changed"),
    ],
)
def test_execution_record_header_rejections(
    field: str,
    value: object,
    message: str,
) -> None:
    record = _products().execution_seal.to_record()
    record[field] = value
    with pytest.raises(ValueError, match=message):
        execution.Deform360CalibrationExecutionSealV1.from_mapping(record)


def test_execution_record_wrong_types_and_verifier_guards(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        execution.Deform360CalibrationExecutionSealV1.from_mapping("bad")

    stage0 = _stage0()
    with pytest.raises(TypeError, match="products must be"):
        execution.verify_deform360_calibration_execution_artifacts(
            object(),  # type: ignore[arg-type]
            stage0_selection=stage0,
            visual_provider_lock=_provider(),
            evidence_use_ledger=_ledger(stage0),
        )
    with pytest.raises(TypeError, match="seal must be"):
        execution.save_deform360_calibration_execution_seal(
            object(),  # type: ignore[arg-type]
            tmp_path / "seal.json",
        )


@pytest.mark.parametrize("value", ["", "name", "=path", "name="])
def test_cli_named_path_rejects_malformed_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="NAME=PATH"):
        cli._named_path(value)  # noqa: SLF001


def test_cli_argument_and_logical_path_rejections() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="ROLE must be"):
        cli._artifact_path("unknown=path")  # noqa: SLF001
    for value in (r"windows\path", "/absolute", "../parent", "./dot"):
        with pytest.raises(ValueError, match="relative POSIX path|POSIX separators"):
            cli._logical_name(value)  # noqa: SLF001


def test_cli_ordinary_file_and_git_guards(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        cli._ordinary_file(tmp_path / "missing", name="source")  # noqa: SLF001
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="ordinary file"):
        cli._ordinary_file(directory, name="source")  # noqa: SLF001

    source = directory / "source.txt"
    source.write_text("source", encoding="utf-8")
    file_link = tmp_path / "source-link.txt"
    file_link.symlink_to(source)
    with pytest.raises(ValueError, match="symlinks"):
        cli._ordinary_file(file_link, name="source")  # noqa: SLF001

    directory_link = tmp_path / "directory-link"
    directory_link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        cli._ordinary_file(directory_link / "source.txt", name="source")  # noqa: SLF001

    with pytest.raises(ValueError, match="cannot verify Git repository"):
        cli._git_output(tmp_path / "missing-repository", "rev-parse", "HEAD")  # noqa: SLF001


def test_cli_repository_revision_and_dirty_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_root = tmp_path / "file-root"
    file_root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        cli._verify_repository(  # noqa: SLF001
            file_root,
            expected_revision="a" * 40,
        )

    root = tmp_path / "repository"
    root.mkdir()

    def mismatch(_repository: Path, *arguments: str) -> str:
        assert arguments == ("rev-parse", "HEAD")
        return "b" * 40

    monkeypatch.setattr(cli, "_git_output", mismatch)
    with pytest.raises(ValueError, match="differs from repository HEAD"):
        cli._verify_repository(root, expected_revision="a" * 40)  # noqa: SLF001

    def dirty(_repository: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return "a" * 40
        return "?? untracked"

    monkeypatch.setattr(cli, "_git_output", dirty)
    with pytest.raises(ValueError, match="must be clean"):
        cli._verify_repository(root, expected_revision="a" * 40)  # noqa: SLF001


def test_cli_copy_source_detects_changed_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")
    digests = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(cli, "file_sha256", lambda _path: next(digests))
    with pytest.raises(ValueError, match="copied source bytes changed"):
        cli._copy_source(source, tmp_path / "copy.txt")  # noqa: SLF001


def test_cli_copy_inputs_additional_and_duplicate_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_copy_source", lambda _source, _destination: "a" * 64)
    artifacts = {role: tmp_path / role for role in DEFORM360_CALIBRATION_ROLES}
    common = {
        "temporary": tmp_path / "temporary",
        "repository_root": tmp_path / "repository",
        "selection_lock": tmp_path / "selection.json",
        "visual_provider_lock": tmp_path / "provider.json",
        "evidence_ledger": tmp_path / "ledger.json",
        "artifacts": artifacts,
    }
    sources = cli._copy_inputs(  # noqa: SLF001
        **common,
        additional_sources=(("extra.json", tmp_path / "extra.json"),),
    )
    assert "sources/additional/extra.json" in sources

    with pytest.raises(ValueError, match="duplicate source path"):
        cli._copy_inputs(  # noqa: SLF001
            **common,
            additional_sources=(
                ("duplicate.json", tmp_path / "first.json"),
                ("duplicate.json", tmp_path / "second.json"),
            ),
        )


def test_cli_artifact_mapping_and_metadata_boundaries(tmp_path: Path) -> None:
    role = DEFORM360_CALIBRATION_ROLES[0]
    with pytest.raises(ValueError, match="duplicate calibration artifact role"):
        cli._artifact_mapping(  # noqa: SLF001
            ((role, tmp_path / "a"), (role, tmp_path / "b"))
        )
    with pytest.raises(ValueError, match="roles changed"):
        cli._artifact_mapping(())  # noqa: SLF001
    complete = tuple((item, tmp_path / item) for item in DEFORM360_CALIBRATION_ROLES)
    with pytest.raises(ValueError, match="roles changed"):
        cli._artifact_mapping((*complete, ("unexpected", tmp_path / "extra")))  # noqa: SLF001

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text('{"operator":"test"}', encoding="utf-8")
    assert cli._metadata(metadata_path) == {"operator": "test"}  # noqa: SLF001


def _run_args(**updates: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "calibration_payloads_opened": True,
        "repository_root": Path("/repository"),
        "implementation_revision": "a" * 40,
        "output_dir": Path("/output/sealed"),
        "artifact": (),
        "selection_lock": Path("selection.json"),
        "visual_provider_lock": Path("provider.json"),
        "evidence_ledger": Path("ledger.json"),
        "additional_source": (),
        "metadata_json": None,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_cli_run_rejects_unacknowledged_and_in_checkout_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="calibration-payloads-opened"):
        cli._run(_run_args(calibration_payloads_opened=False))  # noqa: SLF001

    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(cli, "_verify_repository", lambda *_args, **_kwargs: "a" * 40)
    monkeypatch.setattr(cli, "_verify_runtime_sources", lambda _repository: None)
    monkeypatch.setattr(
        cli,
        "_verify_committed_selection_lock",
        lambda _repository, _selection: None,
    )
    with pytest.raises(ValueError, match="outside the Git checkout"):
        cli._run(  # noqa: SLF001
            _run_args(
                repository_root=repository,
                output_dir=repository / "sealed",
            )
        )


def test_cli_run_cleans_temporary_output_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    output = tmp_path / "outputs" / "sealed"
    monkeypatch.setattr(cli, "_verify_repository", lambda *_args, **_kwargs: "a" * 40)
    monkeypatch.setattr(cli, "_verify_runtime_sources", lambda _repository: None)
    monkeypatch.setattr(
        cli,
        "_verify_committed_selection_lock",
        lambda _repository, _selection: None,
    )
    monkeypatch.setattr(cli, "_artifact_mapping", lambda _values: {})

    def fail_copy(**_kwargs: Any) -> dict[str, str]:
        raise ValueError("copy failed")

    monkeypatch.setattr(cli, "_copy_inputs", fail_copy)
    with pytest.raises(ValueError, match="copy failed"):
        cli._run(  # noqa: SLF001
            _run_args(repository_root=repository, output_dir=output)
        )
    assert not output.exists()
    assert not list(output.parent.glob(".sealed.*.tmp"))


def test_cli_module_main_help_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["bayesian_phystwin.cli.deform360_calibration_execution", "--help"],
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_module(
            "bayesian_phystwin.cli.deform360_calibration_execution",
            run_name="__main__",
        )
    assert error.value.code == 0


def test_cli_rejects_unreviewed_selection_and_runtime_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository_root()
    committed = _stage0_path()
    substituted = tmp_path / "selection.json"
    substituted.write_bytes(committed.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="committed lock"):
        cli._verify_committed_selection_lock(  # noqa: SLF001
            root,
            substituted,
        )

    cli._verify_runtime_sources(root)  # noqa: SLF001
    module_name = next(iter(cli._RUNTIME_MODULE_SOURCES))  # noqa: SLF001
    monkeypatch.setitem(
        cli._RUNTIME_MODULE_SOURCES,  # noqa: SLF001
        module_name,
        "pyproject.toml",
    )
    with pytest.raises(ValueError, match="runtime source bytes"):
        cli._verify_runtime_sources(root)  # noqa: SLF001
