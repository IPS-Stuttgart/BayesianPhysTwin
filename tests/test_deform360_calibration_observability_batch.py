"""Atomic ten-object Deform360 observability batch contracts."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import test_deform360_calibration_observability_case_builder as case_inputs

from bayesian_phystwin.deform360_calibration_observability_report import (
    load_deform360_calibration_observability_report,
)
from bayesian_phystwin.deform360_visual_provider_lock import (
    Deform360VisualProviderLockV1,
)

ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "scripts/science/build_deform360_calibration_observability_batch.py"
SPEC = importlib.util.spec_from_file_location("deform360_batch_cli", CLI_PATH)
assert SPEC is not None and SPEC.loader is not None
CLI = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CLI
SPEC.loader.exec_module(CLI)

OBJECT_IDS = tuple(
    [f"cal-sheet-{index}" for index in range(5)]
    + [f"cal-volumetric-{index}" for index in range(5)]
)


def _batch_inputs(tmp_path: Path) -> case_inputs.Inputs:
    """Upgrade the shared synthetic chain to the strict Stage-0 schema."""

    inputs = case_inputs._inputs(tmp_path)
    source = case_inputs.source_run_cases

    provider = Deform360VisualProviderLockV1(
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
        metadata={"selection_role": "calibration-and-confirmation"},
    ).to_record()
    provider_file_sha256 = source._write(inputs.chain.provider_path, provider)

    source_protocol = json.loads(
        inputs.chain.source_protocol_path.read_text(encoding="utf-8")
    )
    source_protocol["locks"]["visual_provider_lock_id"] = provider["artifact_id"]
    source_protocol["protocol_sha256"] = source.canonical_sha256(
        source_protocol,
        digest_key="protocol_sha256",
    )
    source._write(inputs.chain.source_protocol_path, source_protocol)

    selection = json.loads(inputs.chain.selection_path.read_text(encoding="utf-8"))
    calibration = selection["selection"]["calibration"]
    confirmation = selection["selection"]["confirmation"]
    selection.update(
        {
            "available_raw_object_count": len(calibration) + len(confirmation),
            "cache_preflight": {},
            "excluded_object_count": 0,
            "information_boundary": {
                "camera_media_opened": False,
                "geometry_annotations_opened": False,
                "object_directory_names_opened": True,
                "object_metadata_json_opened": True,
                "robot_arrays_opened": False,
                "tactile_arrays_opened": False,
                "target_outcomes_opened": False,
            },
            "next_gate": "synthetic calibration-source execution",
            "prior_protocols": {},
        }
    )
    selection["dataset"]["raw_prefix"] = "raw"
    selection["selection_sha256"] = source.content_sha256(selection["selection"])
    content_payload = dict(selection)
    content_payload.pop("content_selection_sha256")
    content_payload.pop("implementation_revision")
    content_payload.pop("selection_artifact_sha256")
    selection["content_selection_sha256"] = source.content_sha256(content_payload)
    artifact_payload = dict(selection)
    artifact_payload.pop("selection_artifact_sha256")
    selection["selection_artifact_sha256"] = source.content_sha256(artifact_payload)
    selection_file_sha256 = source._write(inputs.chain.selection_path, selection)

    plan = json.loads(inputs.chain.plan_path.read_text(encoding="utf-8"))
    plan["protocol_sha256"] = source_protocol["protocol_sha256"]
    plan["selection_source_sha256"] = selection_file_sha256
    plan["visual_provider_lock_id"] = provider["artifact_id"]
    plan["visual_provider_source_sha256"] = provider_file_sha256
    plan["plan_sha256"] = source.canonical_sha256(plan, digest_key="plan_sha256")
    source._write(inputs.chain.plan_path, plan)

    download = json.loads(inputs.chain.download_path.read_text(encoding="utf-8"))
    download["plan_sha256"] = plan["plan_sha256"]
    download["download_sha256"] = source.canonical_sha256(
        download,
        digest_key="download_sha256",
    )
    source._write(inputs.chain.download_path, download)

    result = json.loads(inputs.chain.result_path.read_text(encoding="utf-8"))
    result["plan_sha256"] = plan["plan_sha256"]
    result["download_sha256"] = download["download_sha256"]
    result["result_sha256"] = source.canonical_sha256(
        result,
        digest_key="result_sha256",
    )
    source._write(inputs.chain.result_path, result)

    inputs.run_record_path.unlink()
    case_inputs.save_deform360_calibration_source_run_record(
        source._record(inputs.chain),
        inputs.run_record_path,
    )
    return inputs


def _rows(
    *,
    failures: frozenset[str] = frozenset(),
    candidate_overrides: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    overrides = {} if candidate_overrides is None else candidate_overrides
    rows: list[dict[str, Any]] = []
    for object_id in OBJECT_IDS:
        if object_id in failures:
            rows.append(
                {
                    "mode": "technical-failure",
                    "object_id": object_id,
                    "failure_evidence": "failure.txt",
                    "failure_reason": "registered observability failure",
                }
            )
        else:
            rows.append(
                {
                    "mode": "evaluated",
                    "object_id": object_id,
                    "reference_marginal_precision": "reference.npy",
                    "candidate_marginal_precision": overrides.get(
                        object_id,
                        "candidate.npy",
                    ),
                    "contact_anchor_artifact": "contact-anchor.json",
                }
            )
    return rows


def _write_spec(
    path: Path,
    *,
    rows: list[dict[str, Any]] | None = None,
    indent: int | None = 2,
) -> Path:
    value = {
        "schema": CLI.SPEC_SCHEMA,
        "schema_version": CLI.SCHEMA_VERSION,
        "cases": _rows() if rows is None else rows,
    }
    path.write_text(
        json.dumps(value, indent=indent, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _arguments(
    inputs: case_inputs.Inputs,
    *,
    spec: Path,
    output: Path,
) -> list[str]:
    return [
        "--batch-spec",
        str(spec),
        "--input-root",
        str(inputs.reference_path.parent),
        "--source-protocol",
        str(inputs.chain.source_protocol_path),
        "--stage0-protocol",
        str(inputs.chain.stage0_protocol_path),
        "--selection-lock",
        str(inputs.chain.selection_path),
        "--visual-provider-lock",
        str(inputs.chain.provider_path),
        "--calibration-source-plan",
        str(inputs.chain.plan_path),
        "--calibration-source-download",
        str(inputs.chain.download_path),
        "--calibration-source-run-record",
        str(inputs.run_record_path),
        "--calibration-source-result",
        str(inputs.chain.result_path),
        "--query-jacobian",
        str(inputs.query_path),
        "--implementation-revision",
        case_inputs.IMPLEMENTATION_REVISION,
        "--output-dir",
        str(output),
    ]


def _run(
    tmp_path: Path,
    *,
    failures: frozenset[str] = frozenset(),
) -> tuple[case_inputs.Inputs, Path, dict[str, object]]:
    inputs = _batch_inputs(tmp_path / "inputs")
    spec = _write_spec(tmp_path / "batch-spec.json", rows=_rows(failures=failures))
    output = tmp_path / "published"
    args = CLI.build_parser().parse_args(_arguments(inputs, spec=spec, output=output))
    manifest = CLI._run(args)
    return inputs, output, manifest


def test_batch_builds_all_cases_report_and_portable_manifest(tmp_path: Path) -> None:
    _inputs, output, manifest = _run(tmp_path)

    assert manifest["report"]["status"] == (
        "completed-supported-calibration-observability"
    )
    assert manifest["report"]["support_gate"]["evaluated_object_count"] == 10
    assert manifest["report"]["support_gate"]["support_passed"] is True
    assert len(manifest["case_files"]) == 10
    assert CLI.validate_directory(output)["batch_id"] == manifest["batch_id"]

    report = load_deform360_calibration_observability_report(
        output / "calibration-observability-report.json"
    )
    assert report.report_id == manifest["report"]["report_id"]
    assert report.metadata["batch_spec_id"] == manifest["spec_id"]
    assert all(case.comparison is not None for case in report.cases)

    serialized = (output / "batch-manifest.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert "confirm-sheet" not in serialized
    assert sorted(path.name for path in (output / "cases").iterdir()) == sorted(
        f"{entry['case_id']}.json" for entry in manifest["case_files"]
    )


def test_two_retained_failures_preserve_supported_eight_object_gate(
    tmp_path: Path,
) -> None:
    failures = frozenset({"cal-sheet-4", "cal-volumetric-4"})
    _inputs, output, manifest = _run(tmp_path, failures=failures)

    gate = manifest["report"]["support_gate"]
    assert gate["evaluated_object_count"] == 8
    assert gate["technical_failure_count"] == 2
    assert gate["evaluated_by_stratum"] == {"sheet": 4, "volumetric": 4}
    assert gate["support_passed"] is True
    report = load_deform360_calibration_observability_report(
        output / "calibration-observability-report.json"
    )
    assert (
        sum(
            case.status == "technical_failure_without_replacement"
            for case in report.cases
        )
        == 2
    )
    assert CLI.validate_directory(output)["batch_id"] == manifest["batch_id"]


def test_insufficient_support_is_published_as_valid_negative(tmp_path: Path) -> None:
    failures = frozenset({"cal-sheet-0", "cal-sheet-1", "cal-volumetric-0"})
    _inputs, output, manifest = _run(tmp_path, failures=failures)

    assert manifest["report"]["status"] == "completed-insufficient-calibration-support"
    assert manifest["report"]["support_gate"]["evaluated_object_count"] == 7
    assert manifest["report"]["support_gate"]["support_passed"] is False
    assert output.is_dir()
    assert CLI.validate_directory(output)["batch_id"] == manifest["batch_id"]


def test_spec_semantics_ignore_order_and_format_but_bind_exact_bytes(
    tmp_path: Path,
) -> None:
    rows = _rows()
    first = _write_spec(tmp_path / "first.json", rows=rows, indent=2)
    second = _write_spec(
        tmp_path / "second.json",
        rows=list(reversed(rows)),
        indent=None,
    )

    first_cases, first_id, first_digest = CLI._load_spec(first)
    second_cases, second_id, second_digest = CLI._load_spec(second)

    assert [case.object_id for case in first_cases] == [
        case.object_id for case in second_cases
    ]
    assert first_id == second_id
    assert first_digest != second_digest


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda rows: rows[:-1], "exactly ten"),
        (
            lambda rows: [*rows[:-1], dict(rows[0])],
            "repeats a physical object",
        ),
        (
            lambda rows: [
                *rows[:-1],
                {**rows[-1], "object_id": "confirm-sheet-0"},
            ],
            "calibration cohort",
        ),
        (
            lambda rows: [
                {**rows[0], "reference_marginal_precision": "../escape.npy"},
                *rows[1:],
            ],
            "canonical relative POSIX path",
        ),
    ),
)
def test_spec_rejects_incomplete_duplicate_confirmation_and_escape(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    inputs = _batch_inputs(tmp_path / "inputs")
    spec = _write_spec(tmp_path / "batch-spec.json", rows=mutation(_rows()))
    output = tmp_path / "published"
    args = CLI.build_parser().parse_args(_arguments(inputs, spec=spec, output=output))

    with pytest.raises(ValueError, match=message):
        CLI._run(args)
    assert not output.exists()


def test_spec_rejects_duplicate_keys_nonfinite_and_unknown_fields(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema":"x","schema":"y","schema_version":1,"cases":[]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        CLI._load_spec(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text(
        '{"schema":"x","schema_version":NaN,"cases":[]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite"):
        CLI._load_spec(nonfinite)

    unknown_rows = _rows()
    unknown_rows[0]["unknown"] = True
    unknown = _write_spec(tmp_path / "unknown.json", rows=unknown_rows)
    with pytest.raises(ValueError, match="fields changed"):
        CLI._load_spec(unknown)


def test_symlinked_input_and_late_case_failure_leave_no_partial_output(
    tmp_path: Path,
) -> None:
    inputs = _batch_inputs(tmp_path / "inputs")
    link = inputs.reference_path.parent / "reference-link.npy"
    link.symlink_to(inputs.reference_path)
    rows = _rows()
    rows[0]["reference_marginal_precision"] = "reference-link.npy"
    spec = _write_spec(tmp_path / "symlink-spec.json", rows=rows)
    output = tmp_path / "symlink-output"
    args = CLI.build_parser().parse_args(_arguments(inputs, spec=spec, output=output))
    with pytest.raises(ValueError, match="must not contain symlinks"):
        CLI._run(args)
    assert not output.exists()

    bad = inputs.reference_path.parent / "bad-candidate.npy"
    np.save(bad, np.eye(3) * 0.5, allow_pickle=False)
    rows = _rows(candidate_overrides={"cal-volumetric-4": "bad-candidate.npy"})
    spec = _write_spec(tmp_path / "late-failure-spec.json", rows=rows)
    output = tmp_path / "late-failure-output"
    args = CLI.build_parser().parse_args(_arguments(inputs, spec=spec, output=output))
    with pytest.raises(ValueError, match="candidate"):
        CLI._run(args)
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.batch.*"))


@pytest.mark.parametrize("target", ("case", "report", "checksums", "extra"))
def test_directory_validation_detects_every_published_file_mutation(
    tmp_path: Path,
    target: str,
) -> None:
    _inputs, original, manifest = _run(tmp_path / "source")
    changed = tmp_path / "changed"
    shutil.copytree(original, changed)

    if target == "case":
        case_path = changed / manifest["case_files"][0]["path"]
        case_path.write_text(
            case_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
    elif target == "report":
        report = changed / "calibration-observability-report.json"
        report.write_text(
            report.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
    elif target == "checksums":
        checksums = changed / "SHA256SUMS"
        checksums.write_text("0" * 64 + "  batch-manifest.json\n", encoding="utf-8")
    else:
        (changed / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(ValueError):
        CLI.validate_directory(changed)


def test_manifest_contract_rejects_accounting_and_identity_tampering(
    tmp_path: Path,
) -> None:
    _inputs, _output, manifest = _run(tmp_path)

    changed_gate = json.loads(json.dumps(manifest))
    changed_gate["report"]["support_gate"]["evaluated_object_count"] = 9
    with pytest.raises(ValueError, match="identity"):
        CLI._validated_manifest(changed_gate)

    changed_id = json.loads(json.dumps(manifest))
    changed_id["batch_id"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        CLI._validated_manifest(changed_id)

    changed_order = json.loads(json.dumps(manifest))
    changed_order["case_files"].reverse()
    with pytest.raises(ValueError, match="canonical order"):
        CLI._validated_manifest(changed_order)

    changed_source = json.loads(json.dumps(manifest))
    changed_source["source_artifacts"][CLI._SPEC_SOURCE_KEY] = "1" * 64
    with pytest.raises(ValueError, match="specification digest"):
        CLI._validated_manifest(changed_source)


def test_publish_lock_is_not_removed_or_overwritten(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    output = tmp_path / "published"
    lock = tmp_path / ".published.publish.lock"
    lock.write_text("owned elsewhere\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        CLI._publish(staged, output)
    assert lock.read_text(encoding="utf-8") == "owned elsewhere\n"
    assert staged.is_dir()
    assert not output.exists()


def test_cli_exit_codes_distinguish_supported_negative_and_contract_failure(
    tmp_path: Path,
) -> None:
    supported_inputs = _batch_inputs(tmp_path / "supported-inputs")
    supported_spec = _write_spec(tmp_path / "supported-spec.json")
    supported_output = tmp_path / "supported"
    supported_args = _arguments(
        supported_inputs,
        spec=supported_spec,
        output=supported_output,
    )
    assert CLI.main(supported_args) == 0
    assert CLI.main(supported_args) == CLI.CONTRACT_FAILURE_EXIT_CODE

    failures = frozenset({"cal-sheet-0", "cal-sheet-1", "cal-volumetric-0"})
    negative_inputs = _batch_inputs(tmp_path / "negative-inputs")
    negative_spec = _write_spec(
        tmp_path / "negative-spec.json",
        rows=_rows(failures=failures),
    )
    assert (
        CLI.main(
            _arguments(
                negative_inputs,
                spec=negative_spec,
                output=tmp_path / "negative",
            )
        )
        == CLI.INSUFFICIENT_SUPPORT_EXIT_CODE
    )

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}\n", encoding="utf-8")
    assert (
        CLI.main(
            _arguments(
                negative_inputs,
                spec=invalid,
                output=tmp_path / "invalid-output",
            )
        )
        == CLI.CONTRACT_FAILURE_EXIT_CODE
    )
