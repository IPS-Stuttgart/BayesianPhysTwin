#!/usr/bin/env python3
"""Execute registered target-free studies and retain one auditable evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

STUDIES: Final[tuple[str, ...]] = (
    "simulation-based-calibration",
    "synthetic-benchmark-sbc",
    "recursive-corruption",
)
STUDY_CHOICES: Final[tuple[str, ...]] = (*STUDIES, "all-controlled")
EXPECTED_SIMULATION_DECISION: Final = (
    "exact-model-calibration-not-rejected-and-misspecification-detected"
)
EXPECTED_RECURSIVE_CONDITIONS: Final[tuple[str, ...]] = (
    "clean",
    "missing_burst",
    "outlier_burst",
    "coherent_drift",
    "identity_switch",
    "delayed_observation",
    "density_drop",
)
EXPECTED_RECURSIVE_METHODS: Final[tuple[str, ...]] = (
    "physical_baseline",
    "last_residual",
    "exponential_residual",
    "recursive_gaussian",
    "guarded_recursive",
)
EXPECTED_SELECTIVITY_GRID: Final[tuple[float, ...]] = (
    1.0,
    2.0,
    4.0,
    9.0,
    16.0,
    36.0,
    1_000_000.0,
)
CLAIM_BOUNDARIES: Final[Mapping[str, str]] = {
    "simulation-based-calibration": (
        "Controlled synthetic calibration evidence only. A passing decision does "
        "not establish real-data calibration, physical identification, provider "
        "competence, Causal4D intervention benefit, deployment safety, or state "
        "of the art."
    ),
    "synthetic-benchmark-sbc": (
        "End-to-end controlled synthetic posterior evidence only. It does not "
        "establish simulator adequacy, real calibration, physical identifiability, "
        "provider competence, intervention benefit, deployment safety, or state "
        "of the art."
    ),
    "recursive-corruption": (
        "Controlled synthetic mechanism and retrospective selectivity evidence "
        "only. It does not establish real-provider competence, physical-object "
        "transfer, real covariance calibration, Causal4D intervention benefit, "
        "deployment safety, or state of the art."
    ),
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=STUDY_CHOICES, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--workflow-run-id", type=int, required=True)
    parser.add_argument("--workflow-run-attempt", type=int, required=True)
    parser.add_argument("--verify-replay", action="store_true")
    return parser.parse_args(argv)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _validate_finite(value: object, *, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite numeric value at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite(child, path=f"{path}[{index}]")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    _validate_finite(value)
    return value


def _write_json_no_clobber(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_id(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_command(command: Sequence[str], *, cwd: Path, log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=False)
    _write_json_no_clobber(
        log_dir / "command.json",
        {
            "argv": list(command),
            "cwd": str(cwd.resolve()),
            "display": shlex.join(command),
        },
    )
    with (log_dir / "run.log").open(
        "x", encoding="utf-8", newline=""
    ) as log:
        process = subprocess.Popen(  # noqa: S603
            list(command),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONHASHSEED": "0"},
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return_code = process.wait()
        log.flush()
        os.fsync(log.fileno())
    _write_json_no_clobber(
        log_dir / "command-status.json",
        {"exit_code": return_code},
    )
    return return_code


def _simulation_command(output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/science/run_simulation_based_calibration_v1.py",
        "--protocol",
        "protocols/locks/simulation_based_calibration_v1.json",
        "--output",
        str(output_dir / "result.json"),
        "--summary-output",
        str(output_dir / "summary.json"),
        "--require-registered-decision",
    ]


def _synthetic_sbc_command(output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/science/run_synthetic_benchmark_sbc_v1.py",
        "--replicates",
        "512",
        "--seed",
        "20260824",
        "--bins",
        "10",
        "--likelihood-scale-multipliers",
        "1,0.5,2",
        "--output",
        str(output_dir / "result.json"),
    ]


def _recursive_commands(output_dir: Path) -> tuple[list[str], list[str]]:
    benchmark = [
        sys.executable,
        "-m",
        "bayesian_phystwin.cli.recursive_corruption_benchmark",
        "--seeds",
        "0:50",
        "--output-json",
        str(output_dir / "result.json"),
        "--output-csv",
        str(output_dir / "records.csv"),
    ]
    selectivity = [
        sys.executable,
        "scripts/science/analyze_recursive_corruption_selectivity_v1.py",
        "--seeds",
        "0:50",
        "--conditions",
        (
            "missing_burst,outlier_burst,coherent_drift,identity_switch,"
            "delayed_observation,density_drop"
        ),
        "--maximum-nis-grid",
        "1,2,4,9,16,36,1000000",
        "--output",
        str(output_dir / "selectivity.json"),
    ]
    return benchmark, selectivity


def _validate_simulation(output_dir: Path, *, exit_code: int) -> dict[str, object]:
    result = _load_json(output_dir / "result.json")
    summary = _load_json(output_dir / "summary.json")
    decision = summary.get("decision")
    passed = exit_code == 0 and decision == EXPECTED_SIMULATION_DECISION
    return {
        "passed": passed,
        "exit_code": exit_code,
        "decision": decision,
        "expected_decision": EXPECTED_SIMULATION_DECISION,
        "protocol_id": summary.get("protocol_id"),
        "result_id": summary.get("result_id"),
        "summary_id": summary.get("summary_id"),
        "replicate_row_count": summary.get("replicate_row_count"),
        "correlated_failed_test_fraction": summary.get(
            "correlated_failed_test_fraction"
        ),
        "full_result_id": result.get("result_id"),
    }


def _validate_synthetic_sbc(
    output_dir: Path,
    *,
    exit_code: int,
) -> dict[str, object]:
    result = _load_json(output_dir / "result.json")
    separation = result.get("normative_control_separation")
    if not isinstance(separation, Mapping):
        raise ValueError("synthetic SBC normative_control_separation is missing")
    smallest_ks = separation.get("matched_has_smallest_mean_ks") is True
    smallest_coverage_error = (
        separation.get("matched_has_smallest_90_coverage_error") is True
    )
    passed = exit_code == 0 and smallest_ks and smallest_coverage_error
    return {
        "passed": passed,
        "exit_code": exit_code,
        "decision": (
            "matched-posterior-separated-from-dispersion-controls"
            if passed
            else "normative-control-separation-not-established"
        ),
        "result_id": result.get("result_id"),
        "replicate_count": result.get("replicate_count"),
        "parameter_grid_size": result.get("parameter_grid_size"),
        "matched_has_smallest_mean_ks": smallest_ks,
        "matched_has_smallest_90_coverage_error": smallest_coverage_error,
    }


def _validate_recursive(
    output_dir: Path,
    *,
    benchmark_exit_code: int,
    selectivity_exit_code: int,
) -> dict[str, object]:
    result = _load_json(output_dir / "result.json")
    selectivity = _load_json(output_dir / "selectivity.json")
    expected_seeds = list(range(50))
    expected_record_count = (
        50 * len(EXPECTED_RECURSIVE_CONDITIONS) * len(EXPECTED_RECURSIVE_METHODS)
    )
    expected_corrupted_sequences = 50 * (len(EXPECTED_RECURSIVE_CONDITIONS) - 1)

    if result.get("seeds") != expected_seeds:
        raise ValueError("recursive benchmark seed roster drifted")
    if result.get("conditions") != list(EXPECTED_RECURSIVE_CONDITIONS):
        raise ValueError("recursive benchmark condition roster drifted")
    if result.get("methods") != list(EXPECTED_RECURSIVE_METHODS):
        raise ValueError("recursive benchmark method roster drifted")
    records = result.get("records")
    if not isinstance(records, list) or len(records) != expected_record_count:
        raise ValueError("recursive benchmark record count drifted")
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("recursive benchmark summary is missing")
    if (
        summary.get("corrupted_sequence_count_per_method")
        != expected_corrupted_sequences
    ):
        raise ValueError("recursive benchmark corrupted-sequence count drifted")
    fallback_violations = summary.get("guarded_exact_fallback_violation_count")
    if fallback_violations != 0:
        raise ValueError("recursive benchmark reported exact-fallback violations")

    with (output_dir / "records.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        csv_rows = sum(1 for _ in csv.DictReader(stream))
    if csv_rows != expected_record_count:
        raise ValueError("recursive benchmark CSV row count drifted")

    if selectivity.get("selection_authorized") is not False:
        raise ValueError("selectivity report must not authorize threshold selection")
    if selectivity.get("seeds") != expected_seeds:
        raise ValueError("selectivity seed roster drifted")
    if selectivity.get("conditions") != list(EXPECTED_RECURSIVE_CONDITIONS[1:]):
        raise ValueError("selectivity condition roster drifted")
    if selectivity.get("maximum_nis_grid") != list(EXPECTED_SELECTIVITY_GRID):
        raise ValueError("selectivity threshold grid drifted")
    curve = selectivity.get("curve")
    if not isinstance(curve, list) or len(curve) != len(EXPECTED_SELECTIVITY_GRID):
        raise ValueError("selectivity curve length drifted")
    for row in curve:
        if not isinstance(row, Mapping):
            raise ValueError("selectivity curve row must be a mapping")
        if row.get("sequence_count") != expected_corrupted_sequences:
            raise ValueError("selectivity sequence count drifted")
        if row.get("exact_fallback_violation_count") != 0:
            raise ValueError("selectivity curve reported exact-fallback violations")

    passed = benchmark_exit_code == 0 and selectivity_exit_code == 0
    return {
        "passed": passed,
        "decision": (
            "complete-finite-controlled-evidence"
            if passed
            else "controlled-evidence-command-failed"
        ),
        "benchmark_exit_code": benchmark_exit_code,
        "selectivity_exit_code": selectivity_exit_code,
        "record_count": expected_record_count,
        "corrupted_sequence_count_per_method": expected_corrupted_sequences,
        "guarded_exact_fallback_violation_count": fallback_violations,
        "selectivity_report_id": selectivity.get("report_id"),
        "selection_authorized": False,
    }


def _run_once(
    study: str,
    *,
    repository_root: Path,
    output_dir: Path,
    label: str,
) -> tuple[dict[str, object], tuple[str, ...]]:
    output_dir.mkdir(parents=True, exist_ok=False)
    if study == "simulation-based-calibration":
        status = _run_command(
            _simulation_command(output_dir),
            cwd=repository_root,
            log_dir=output_dir.parent / f"{label}-command",
        )
        return (
            _validate_simulation(output_dir, exit_code=status),
            ("result.json", "summary.json"),
        )
    if study == "synthetic-benchmark-sbc":
        status = _run_command(
            _synthetic_sbc_command(output_dir),
            cwd=repository_root,
            log_dir=output_dir.parent / f"{label}-command",
        )
        return (
            _validate_synthetic_sbc(output_dir, exit_code=status),
            ("result.json",),
        )
    if study == "recursive-corruption":
        benchmark, selectivity = _recursive_commands(output_dir)
        benchmark_status = _run_command(
            benchmark,
            cwd=repository_root,
            log_dir=output_dir / "benchmark-command",
        )
        selectivity_status = _run_command(
            selectivity,
            cwd=repository_root,
            log_dir=output_dir / "selectivity-command",
        )
        return (
            _validate_recursive(
                output_dir,
                benchmark_exit_code=benchmark_status,
                selectivity_exit_code=selectivity_status,
            ),
            ("result.json", "records.csv", "selectivity.json"),
        )
    raise ValueError(f"unknown controlled study {study!r}")


def _compare_files(
    primary_dir: Path,
    replay_dir: Path,
    filenames: Sequence[str],
) -> dict[str, object]:
    files: list[dict[str, object]] = []
    byte_identical = True
    for filename in filenames:
        primary = primary_dir / filename
        replay = replay_dir / filename
        same = primary.read_bytes() == replay.read_bytes()
        files.append(
            {
                "path": filename,
                "primary_sha256": _sha256(primary),
                "replay_sha256": _sha256(replay),
                "byte_identical": same,
            }
        )
        byte_identical = byte_identical and same
    return {"byte_identical": byte_identical, "files": files}


def _run_study(
    study: str,
    *,
    repository_root: Path,
    output_root: Path,
    verify_replay: bool,
) -> dict[str, object]:
    study_root = output_root / study
    study_root.mkdir(parents=True, exist_ok=False)
    primary_dir = study_root / "primary"
    primary, filenames = _run_once(
        study,
        repository_root=repository_root,
        output_dir=primary_dir,
        label="primary",
    )

    replay: dict[str, object] | None = None
    if verify_replay:
        replay_dir = study_root / "replay"
        replay_validation, replay_filenames = _run_once(
            study,
            repository_root=repository_root,
            output_dir=replay_dir,
            label="replay",
        )
        if replay_filenames != filenames:
            raise ValueError("primary and replay file rosters differ")
        replay = {
            "validation": replay_validation,
            **_compare_files(primary_dir, replay_dir, filenames),
        }
    replay_passed = replay is None or (
        replay.get("byte_identical") is True
        and isinstance(replay.get("validation"), Mapping)
        and replay["validation"].get("passed") is True
    )
    outcome: dict[str, object] = {
        "study": study,
        "primary": primary,
        "verify_replay": verify_replay,
        "replay": replay,
        "passed": primary.get("passed") is True and replay_passed,
        "target_outcomes_used": False,
        "deform360_confirmation_opened": False,
        "causal4d_physical_outcome_used": False,
        "claim_boundary": CLAIM_BOUNDARIES[study],
    }
    _write_json_no_clobber(study_root / "decision.json", outcome)
    return outcome


def _failure_outcome(
    study: str,
    error: Exception,
    *,
    verify_replay: bool,
) -> dict[str, object]:
    return {
        "study": study,
        "primary": None,
        "verify_replay": verify_replay,
        "replay": None,
        "passed": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "target_outcomes_used": False,
        "deform360_confirmation_opened": False,
        "causal4d_physical_outcome_used": False,
        "claim_boundary": CLAIM_BOUNDARIES[study],
    }


def _write_manifest(
    output_root: Path,
    *,
    summary: Mapping[str, object],
    repository: str,
    commit: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
) -> dict[str, object]:
    summary_path = output_root / "bundle-summary.json"
    _write_json_no_clobber(summary_path, summary)
    files: list[dict[str, object]] = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "ControlledScientificEvidenceManifest",
        "repository": repository,
        "commit": commit,
        "workflow_run_id": workflow_run_id,
        "workflow_run_attempt": workflow_run_attempt,
        "bundle_summary_sha256": _sha256(summary_path),
        "target_outcomes_used": False,
        "deform360_confirmation_opened": False,
        "causal4d_physical_outcome_used": False,
        "files": files,
        "claim_boundary": (
            "This bundle contains controlled target-free evidence only. It cannot "
            "authorize real-provider, independent-object, Causal4D physical, "
            "deployment-safety, or state-of-the-art claims."
        ),
    }
    manifest["manifest_id"] = _canonical_id(manifest)
    _write_json_no_clobber(output_root / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", args.commit):
        raise ValueError("commit must be a literal lowercase Git identity")
    if args.repository.count("/") != 1 or any(
        not part for part in args.repository.split("/")
    ):
        raise ValueError("repository must use owner/name form")
    if args.workflow_run_id < 1 or args.workflow_run_attempt < 1:
        raise ValueError("workflow run identity values must be positive")

    repository_root = Path.cwd().resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for reserved in ("bundle-summary.json", "manifest.json"):
        if (output_root / reserved).exists():
            raise FileExistsError(f"refusing to replace {output_root / reserved}")

    selected = STUDIES if args.study == "all-controlled" else (args.study,)
    outcomes: dict[str, object] = {}
    for study in selected:
        try:
            outcomes[study] = _run_study(
                study,
                repository_root=repository_root,
                output_root=output_root,
                verify_replay=args.verify_replay,
            )
        except Exception as error:
            outcome = _failure_outcome(
                study,
                error,
                verify_replay=args.verify_replay,
            )
            outcomes[study] = outcome
            study_root = output_root / study
            study_root.mkdir(parents=True, exist_ok=True)
            if not (study_root / "decision.json").exists():
                _write_json_no_clobber(study_root / "decision.json", outcome)

    all_passed = all(
        isinstance(outcome, Mapping) and outcome.get("passed") is True
        for outcome in outcomes.values()
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "ControlledScientificEvidenceBundleSummary",
        "selected_studies": list(selected),
        "verify_replay": args.verify_replay,
        "all_passed": all_passed,
        "outcomes": outcomes,
        "target_outcomes_used": False,
        "deform360_confirmation_opened": False,
        "causal4d_physical_outcome_used": False,
    }
    manifest = _write_manifest(
        output_root,
        summary=summary,
        repository=args.repository,
        commit=args.commit,
        workflow_run_id=args.workflow_run_id,
        workflow_run_attempt=args.workflow_run_attempt,
    )
    print(
        json.dumps(
            {
                "all_passed": all_passed,
                "manifest_id": manifest["manifest_id"],
                "selected_studies": list(selected),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if all_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
