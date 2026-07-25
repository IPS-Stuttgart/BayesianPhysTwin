#!/usr/bin/env python3
"""Qualify one-original-trainer-process-per-case resource isolation.

This operator is source-only. It never accepts a formal held root, target
query, outcome, gate, or score path. The canonical run launches four fresh
children, each of which exercises the original pinned Deform360 trainer for
the 81-fit lifecycle of one official reconstruction case.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import stat
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _load_support_module() -> Any:
    source = Path(__file__).resolve().with_name(
        "qualify_deform360_resource_lifecycle.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_deform360_resource_lifecycle_support", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the resource-lifecycle support module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


support = _load_support_module()

QUALIFICATION_ID = "deform360-original-trainer-process-isolation-v1"
QUALIFICATION_KIND = "Deform360ProcessIsolationQualificationEvidenceV1"
ATTEMPT_KIND = "Deform360ProcessIsolationQualificationAttemptV1"
CHILD_KIND = "Deform360ProcessIsolationCaseChildEvidenceV1"
RELATIVE_SOURCE = Path(
    "scripts/development/qualify_deform360_process_isolation.py"
)
RELATIVE_NUMERICAL_SOURCE = Path(
    "src/bayesian_phystwin/deform360_held_outcome_reconstruction.py"
)
CANONICAL_CASE_COUNT = 4
CANONICAL_FITS_PER_CASE = 81
CANONICAL_ITERATIONS = 1
CANONICAL_SEED = 0
PHYSICAL_GPU_INDEX = 1
EXPECTED_SOFT_NOFILE_LIMIT = 1024
CHILD_MINIMUM_UNUSED_FILE_DESCRIPTORS = 256
CHILD_TASK_GROWTH_LIMIT = 192
CHILD_START_SPREAD_LIMIT = 4
PARENT_RESOURCE_GROWTH_LIMIT = 2
CASE_TIMEOUT_SECONDS = 28_800
QUALIFICATION_BASE = Path("/mnt/corsair/florianpfaff")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _global_counts(writer: Any, profiler: Any) -> dict[str, int]:
    return {
        "event_writer_count": len(writer.EVENT_WRITERS),
        "event_storage_count": len(writer.EVENT_STORAGE),
        "global_buffer_key_count": len(writer.GLOBAL_BUFFER),
        "profiler_count": len(profiler.PROFILER),
        "pytorch_profiler_present": int(profiler.PYTORCH_PROFILER is not None),
    }


def _evaluate_child(
    *,
    before: Mapping[str, int],
    fits: Sequence[Mapping[str, Any]],
    final_globals: Mapping[str, int],
    expected_fit_count: int,
) -> dict[str, Any]:
    boundaries = [
        dict(record.get("resource_boundary", {})) for record in fits
    ]
    fd_values = [
        int(boundary.get("file_descriptor_count", -1))
        for boundary in boundaries
    ]
    task_values = [
        int(boundary.get("task_count", -1)) for boundary in boundaries
    ]
    rlimits = [
        (
            int(boundary.get("rlimit_nofile_soft", -1)),
            int(boundary.get("rlimit_nofile_hard", -1)),
        )
        for boundary in boundaries
    ]
    initial_rlimit = (
        int(before["rlimit_nofile_soft"]),
        int(before["rlimit_nofile_hard"]),
    )
    maximum_fd = max(fd_values) if fd_values else None
    maximum_task = max(task_values) if task_values else None
    predicates = {
        "fit_count_exact": len(fits) == expected_fit_count,
        "all_outputs_created": bool(fits)
        and all(record.get("output_created") is True for record in fits),
        "all_outputs_removed": bool(fits)
        and all(record.get("output_absent_after_cleanup") is True for record in fits),
        "all_generated_trees_removed": bool(fits)
        and all(
            record.get("generated_outputs_absent_after_cleanup") is True
            for record in fits
        ),
        "resource_boundary_recorded_after_every_fit": bool(fits)
        and all(
            record.get("resource_boundary_stage") == "after_cleanup"
            for record in fits
        ),
        "soft_nofile_limit_is_pinned": (
            initial_rlimit[0] == EXPECTED_SOFT_NOFILE_LIMIT
        ),
        "nofile_limit_unchanged": bool(fits)
        and all(value == initial_rlimit for value in rlimits),
        "child_retains_fd_safety_margin": (
            maximum_fd is not None
            and maximum_fd
            <= initial_rlimit[0] - CHILD_MINIMUM_UNUSED_FILE_DESCRIPTORS
        ),
        "child_task_growth_within_limit": (
            maximum_task is not None
            and maximum_task
            <= int(before["task_count"]) + CHILD_TASK_GROWTH_LIMIT
        ),
        "original_writer_path_exercised": (
            final_globals.get("event_writer_count") == expected_fit_count
        ),
        "original_profiler_path_exercised": (
            final_globals.get("profiler_count") == expected_fit_count
        ),
    }
    return {
        "passed": all(predicates.values()),
        "predicates": predicates,
        "limits": {
            "expected_soft_nofile": EXPECTED_SOFT_NOFILE_LIMIT,
            "minimum_unused_file_descriptors": (
                CHILD_MINIMUM_UNUSED_FILE_DESCRIPTORS
            ),
            "task_growth": CHILD_TASK_GROWTH_LIMIT,
        },
        "observed": {
            "initial": dict(before),
            "minimum_file_descriptor_count": min(fd_values) if fd_values else None,
            "maximum_file_descriptor_count": maximum_fd,
            "unused_file_descriptors_at_peak": (
                initial_rlimit[0] - maximum_fd
                if maximum_fd is not None
                else None
            ),
            "minimum_task_count": min(task_values) if task_values else None,
            "maximum_task_count": maximum_task,
            "maximum_task_growth": (
                maximum_task - int(before["task_count"])
                if maximum_task is not None
                else None
            ),
            "final_globals": dict(final_globals),
        },
    }


def _validate_child_evidence(
    value: Mapping[str, Any],
    *,
    expected_case_index: int,
    expected_fit_count: int,
    expected_iterations: int,
    expected_seed: int,
) -> None:
    _require(
        value.get("artifact_sha256") == support._artifact_sha256(value),
        "case-child evidence signature changed",
    )
    _require(
        value.get("artifact_kind") == CHILD_KIND
        and value.get("qualification_id") == QUALIFICATION_ID
        and value.get("case_index") == expected_case_index
        and value.get("passed") is True,
        "case-child identity or decision changed",
    )
    _require(
        value.get("parameters")
        == {
            "fit_count": expected_fit_count,
            "iterations_per_fit": expected_iterations,
            "seed": expected_seed,
            "trainer_instance_count": 1,
            "trainer_variant": "original-pinned-default",
        },
        "case-child parameters changed",
    )
    _require(
        value.get("information_boundary")
        == {
            "formal_held_path_supplied": False,
            "target_query_path_received": False,
            "outcome_path_received": False,
            "gate_path_received": False,
            "score_path_received": False,
        },
        "case-child information boundary changed",
    )
    evaluation = value.get("evaluation")
    _require(
        isinstance(evaluation, Mapping)
        and evaluation.get("passed") is True
        and all(evaluation.get("predicates", {}).values()),
        "case-child lifecycle evaluation failed",
    )


def _evaluate_parent(
    *,
    initial: Mapping[str, int],
    cases: Sequence[Mapping[str, Any]],
    expected_case_count: int,
) -> dict[str, Any]:
    child_values = [
        record.get("child_evidence", {}) for record in cases
    ]
    child_pids = [int(value.get("process_id", -1)) for value in child_values]
    child_initial_fd = [
        int(value.get("evaluation", {}).get("observed", {}).get("initial", {}).get(
            "file_descriptor_count", -1
        ))
        for value in child_values
    ]
    child_initial_tasks = [
        int(value.get("evaluation", {}).get("observed", {}).get("initial", {}).get(
            "task_count", -1
        ))
        for value in child_values
    ]
    parent_boundaries = [
        dict(record.get("parent_after_child_exit", {})) for record in cases
    ]
    parent_fd = [
        int(value.get("file_descriptor_count", -1)) for value in parent_boundaries
    ]
    parent_tasks = [
        int(value.get("task_count", -1)) for value in parent_boundaries
    ]
    initial_rlimit = (
        int(initial["rlimit_nofile_soft"]),
        int(initial["rlimit_nofile_hard"]),
    )
    parent_rlimits = [
        (
            int(value.get("rlimit_nofile_soft", -1)),
            int(value.get("rlimit_nofile_hard", -1)),
        )
        for value in parent_boundaries
    ]
    predicates = {
        "case_count_exact": len(cases) == expected_case_count,
        "all_child_invocations_succeeded": bool(cases)
        and all(
            record.get("invocation", {}).get("return_code") == 0
            and record.get("invocation", {}).get("timed_out") is False
            for record in cases
        ),
        "all_child_evidence_passed": bool(cases)
        and all(value.get("passed") is True for value in child_values),
        "all_child_contracts_valid": bool(cases)
        and all(record.get("child_contract_valid") is True for record in cases),
        "all_child_processes_distinct": (
            len(child_pids) == expected_case_count
            and len(set(child_pids)) == expected_case_count
            and min(child_pids, default=-1) > 0
        ),
        "child_start_fd_spread_within_limit": bool(child_initial_fd)
        and min(child_initial_fd) > 0
        and max(child_initial_fd) - min(child_initial_fd)
        <= CHILD_START_SPREAD_LIMIT,
        "child_start_task_spread_within_limit": bool(child_initial_tasks)
        and min(child_initial_tasks) > 0
        and max(child_initial_tasks) - min(child_initial_tasks)
        <= CHILD_START_SPREAD_LIMIT,
        "parent_fd_growth_within_limit": bool(parent_fd)
        and max(parent_fd)
        <= int(initial["file_descriptor_count"]) + PARENT_RESOURCE_GROWTH_LIMIT,
        "parent_task_growth_within_limit": bool(parent_tasks)
        and max(parent_tasks)
        <= int(initial["task_count"]) + PARENT_RESOURCE_GROWTH_LIMIT,
        "parent_nofile_limit_unchanged": bool(parent_rlimits)
        and all(value == initial_rlimit for value in parent_rlimits),
        "all_materialized_inputs_stable": bool(cases)
        and all(record.get("materialized_inputs_stable") is True for record in cases),
        "all_source_inputs_stable": bool(cases)
        and all(record.get("source_inputs_stable") is True for record in cases),
        "no_generated_dataset_outputs_remain": bool(cases)
        and all(
            record.get("generated_dataset_outputs_absent") is True
            for record in cases
        ),
    }
    return {
        "passed": all(predicates.values()),
        "predicates": predicates,
        "limits": {
            "child_start_spread": CHILD_START_SPREAD_LIMIT,
            "parent_fd_growth": PARENT_RESOURCE_GROWTH_LIMIT,
            "parent_task_growth": PARENT_RESOURCE_GROWTH_LIMIT,
        },
        "observed": {
            "initial_parent_boundary": dict(initial),
            "child_process_ids": child_pids,
            "child_initial_file_descriptor_counts": child_initial_fd,
            "child_initial_task_counts": child_initial_tasks,
            "parent_post_child_file_descriptor_counts": parent_fd,
            "parent_post_child_task_counts": parent_tasks,
        },
    }


def _case_child(arguments: argparse.Namespace) -> int:
    result_path = support._assert_nonheld_path(
        arguments.result, label="case-child result", must_exist=False
    )
    try:
        code = support._assert_nonheld_path(
            arguments.code_root, label="case-child code root", must_exist=True
        )
        deform360 = support._assert_nonheld_path(
            arguments.deform360_repo,
            label="case-child Deform360 root",
            must_exist=True,
        )
        dataset = support._assert_nonheld_path(
            arguments.dataset, label="case-child dataset", must_exist=True
        )
        output = support._assert_nonheld_path(
            arguments.output_dir, label="case-child output", must_exist=True
        )
        _require(
            Path(__file__).resolve(strict=True)
            == (code / RELATIVE_SOURCE).resolve(strict=True),
            "case-child operator escaped the bound code root",
        )
        runtime = support._seed_runtime(arguments.seed)
        gsplat_runtime = support._load_and_smoke_gsplat_runtime(code)
        trainer_type, _, writer, profiler = support._import_trainers(code, deform360)
        numerical_source = (
            code / RELATIVE_NUMERICAL_SOURCE
        ).resolve(strict=True)
        trainer_source = Path(
            sys.modules[trainer_type.__module__].__file__
        ).resolve(strict=True)
        expected_trainer_source = (
            deform360 / "deform360/processing/reconstruct_stage.py"
        ).resolve(strict=True)
        _require(
            trainer_source == expected_trainer_source,
            "case-child trainer came from another Deform360 checkout",
        )
        trainer = trainer_type()
        initial_globals = _global_counts(writer, profiler)
        _require(
            initial_globals["event_writer_count"] == 0
            and initial_globals["profiler_count"] == 0,
            "case-child inherited Nerfstudio instrumentation",
        )
        before = support._process_boundary()
        fits: list[dict[str, Any]] = []
        for index in range(arguments.fit_count):
            output_filename = f"splat-{index:04d}.ply"
            produced = support._absolute(
                Path(
                    trainer.train(
                        dataset,
                        output,
                        output_filename,
                        arguments.iterations,
                    )
                )
            )
            expected_output = support._absolute(output / output_filename)
            _require(produced == expected_output, "case-child output escaped")
            observed = os.lstat(produced)
            output_created = bool(
                stat.S_ISREG(observed.st_mode)
                and not stat.S_ISLNK(observed.st_mode)
                and observed.st_nlink == 1
                and observed.st_size > 0
            )
            _require(output_created, "case-child output is not a regular file")
            output_size = observed.st_size
            support._remove_owned_file(
                produced,
                parent=output,
                label=f"case-child fit {index} PLY",
            )
            generated = dataset / "outputs"
            _require(
                generated.is_dir() and not generated.is_symlink(),
                "case-child Nerfstudio output tree is absent or linked",
            )
            support._remove_owned_tree(
                generated,
                parent=dataset,
                label=f"case-child fit {index} Nerfstudio outputs",
            )
            boundary = support._process_boundary()
            fits.append(
                {
                    "fit_index": index,
                    "output_created": output_created,
                    "output_size_bytes": output_size,
                    "output_absent_after_cleanup": not os.path.lexists(produced),
                    "generated_outputs_absent_after_cleanup": (
                        not os.path.lexists(generated)
                    ),
                    "resource_boundary_stage": "after_cleanup",
                    "resource_boundary": boundary,
                    "global_counts": _global_counts(writer, profiler),
                }
            )
        final_globals = _global_counts(writer, profiler)
        evaluation = _evaluate_child(
            before=before,
            fits=fits,
            final_globals=final_globals,
            expected_fit_count=arguments.fit_count,
        )
        evidence = support._signed(
            {
                "schema_version": 1,
                "artifact_kind": CHILD_KIND,
                "qualification_id": QUALIFICATION_ID,
                "case_index": arguments.case_index,
                "process_id": os.getpid(),
                "passed": evaluation["passed"],
                "parameters": {
                    "fit_count": arguments.fit_count,
                    "iterations_per_fit": arguments.iterations,
                    "seed": arguments.seed,
                    "trainer_instance_count": 1,
                    "trainer_variant": "original-pinned-default",
                },
                "runtime": runtime,
                "gsplat_runtime_smoke": gsplat_runtime,
                "trainer_source": support._bound_file(
                    expected_trainer_source, label="original trainer source"
                ),
                "numerical_adapter_source": support._bound_file(
                    numerical_source, label="process-isolation numerical adapter"
                ),
                "dataset": os.fspath(dataset),
                "output": os.fspath(output),
                "initial_globals": initial_globals,
                "fits": fits,
                "evaluation": evaluation,
                "information_boundary": {
                    "formal_held_path_supplied": False,
                    "target_query_path_received": False,
                    "outcome_path_received": False,
                    "gate_path_received": False,
                    "score_path_received": False,
                },
            }
        )
        support._write_new_json(result_path, evidence)
        return 0 if evaluation["passed"] else 2
    except BaseException as error:
        failure = support._signed(
            {
                "schema_version": 1,
                "artifact_kind": CHILD_KIND,
                "qualification_id": QUALIFICATION_ID,
                "case_index": getattr(arguments, "case_index", -1),
                "process_id": os.getpid(),
                "passed": False,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                "information_boundary": {
                    "formal_held_path_supplied": False,
                    "target_query_path_received": False,
                    "outcome_path_received": False,
                    "gate_path_received": False,
                    "score_path_received": False,
                },
            }
        )
        if not os.path.lexists(result_path):
            support._write_new_json(result_path, failure)
        return 2


def _canonical_parameters(arguments: argparse.Namespace, dataset: Path) -> dict[str, Any]:
    expected = {
        "case_count": CANONICAL_CASE_COUNT,
        "fit_count": CANONICAL_FITS_PER_CASE,
        "iterations": CANONICAL_ITERATIONS,
        "seed": CANONICAL_SEED,
        "cuda_device": PHYSICAL_GPU_INDEX,
        "case_timeout_seconds": CASE_TIMEOUT_SECONDS,
    }
    for name, expected_value in expected.items():
        _require(
            getattr(arguments, name) == expected_value,
            f"canonical qualification requires {name}={expected_value!r}",
        )
    canonical_dataset = support._assert_nonheld_path(
        support.DEFAULT_PUBLIC_DEV_DATASET,
        label="canonical public development dataset",
        must_exist=True,
    )
    _require(dataset == canonical_dataset, "canonical source dataset changed")
    return {"dataset": os.fspath(canonical_dataset), **expected}


def _run(arguments: argparse.Namespace) -> int:
    code = support._assert_nonheld_path(
        arguments.code_root, label="qualification code root", must_exist=True
    )
    deform360 = support._assert_nonheld_path(
        arguments.deform360_repo,
        label="qualification Deform360 root",
        must_exist=True,
    )
    dataset = support._assert_nonheld_path(
        arguments.dataset, label="qualification source dataset", must_exist=True
    )
    output = support._assert_nonheld_path(
        arguments.output_dir, label="qualification output", must_exist=False
    )
    canonical = _canonical_parameters(arguments, dataset)
    _require(
        socket.gethostname() == "workstation2",
        "process-isolation qualification host is not pinned",
    )
    python = support._absolute(arguments.python)
    _require(python == support.PINNED_PYTHON, "qualification Python changed")
    _require(
        deform360 == support.PINNED_DEFORM360,
        "qualification Deform360 runtime changed",
    )
    parent_python = support._current_python_process_binding()
    python_binding = support._python_runtime_binding(python)
    code_binding = support._git_binding(code)
    deform360_binding = support._git_binding(
        deform360, expected_head=support.PINNED_DEFORM360_REVISION
    )
    script = (code / RELATIVE_SOURCE).resolve(strict=True)
    _require(
        script == Path(__file__).resolve(strict=True),
        "qualification operator escaped the clean code root",
    )
    expected_output = QUALIFICATION_BASE / (
        f"bpt-process-isolation-qualification-{code_binding['head']}"
    )
    _require(output == expected_output, "qualification output root is not canonical")
    _require(not os.path.lexists(output), "qualification output root is consumed")
    for protected in (code, deform360, dataset):
        _require(
            protected not in output.parents,
            "qualification output is nested under a protected input",
        )

    output.mkdir(parents=True, exist_ok=False)
    attempt = support._write_new_json(
        output / "qualification-attempt.json",
        support._signed(
            {
                "schema_version": 1,
                "artifact_kind": ATTEMPT_KIND,
                "qualification_id": QUALIFICATION_ID,
                "state": "canonical-root-consumed-at-creation",
                "output_root": os.fspath(output),
                "code_revision": code_binding["head"],
                "physical_gpu_index": PHYSICAL_GPU_INDEX,
                "canonical_parameters": canonical,
                "root_consumption_policy": {
                    "same_root_retry_permitted": False,
                    "same_revision_retry_permitted": False,
                    "in_place_reuse_permitted": False,
                    "later_fix_requires_new_revision_and_root": True,
                },
                "information_boundary": {
                    "formal_held_path_supplied": False,
                    "target_query_path_received": False,
                    "outcome_path_received": False,
                    "gate_path_received": False,
                    "score_path_received": False,
                },
            }
        ),
    )
    source_binding = support._bound_file(script, label="qualification source")
    numerical_binding = support._bound_file(
        code / RELATIVE_NUMERICAL_SOURCE,
        label="process-isolation numerical adapter source",
    )
    initial_parent = support._process_boundary()
    cases: list[dict[str, Any]] = []
    for case_index in range(arguments.case_count):
        case_root = output / f"case-{case_index:03d}"
        case_root.mkdir()
        materialized = case_root / "dataset"
        export = case_root / "export"
        export.mkdir()
        temporary = case_root / "tmp"
        temporary.mkdir(mode=0o700)
        dataset_audit = support._materialize_dataset(dataset, materialized)
        result_path = case_root / "case-child-evidence.json"
        command = [
            *support._child_python_argv_prefix(python, script),
            "_case-child",
            "--code-root",
            os.fspath(code),
            "--deform360-repo",
            os.fspath(deform360),
            "--dataset",
            os.fspath(materialized),
            "--output-dir",
            os.fspath(export),
            "--result",
            os.fspath(result_path),
            "--case-index",
            str(case_index),
            "--fit-count",
            str(arguments.fit_count),
            "--iterations",
            str(arguments.iterations),
            "--seed",
            str(arguments.seed),
        ]
        before_child = support._process_boundary()
        invocation = support._invoke_child(
            command,
            environment=support._child_environment(
                arguments.cuda_device, temporary
            ),
            log_path=case_root / "case-child.log",
            timeout_seconds=arguments.case_timeout_seconds,
        )
        after_child = support._process_boundary()
        invocation_succeeded = bool(
            invocation["return_code"] == 0
            and invocation["timed_out"] is False
            and invocation["timeout_error"] is None
        )
        child: dict[str, Any] = {}
        child_contract_valid = False
        child_validation_error: dict[str, str] | None = None
        child_evidence_file: dict[str, Any] | None = None
        if os.path.lexists(result_path):
            try:
                child = support._load_signed_json(
                    result_path, label=f"case-child {case_index} evidence"
                )
                if invocation_succeeded:
                    _validate_child_evidence(
                        child,
                        expected_case_index=case_index,
                        expected_fit_count=arguments.fit_count,
                        expected_iterations=arguments.iterations,
                        expected_seed=arguments.seed,
                    )
                    child_contract_valid = True
                child_evidence_file = support._bound_file(
                    result_path, label=f"case-child {case_index} evidence file"
                )
            except (OSError, ValueError) as error:
                child_validation_error = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
        materialized_stable = support._materialized_inputs_stable(dataset_audit)
        source_stable = support._source_inputs_stable(dataset_audit)
        generated_absent = not os.path.lexists(materialized / "outputs")
        _require(
            materialized_stable and source_stable and generated_absent,
            f"process-isolated source case {case_index} changed inputs",
        )
        cases.append(
            {
                "case_index": case_index,
                "dataset": dataset_audit,
                "parent_before_child_launch": before_child,
                "invocation": invocation,
                "child_evidence": child,
                "child_contract_valid": child_contract_valid,
                "child_validation_error": child_validation_error,
                "child_evidence_file": child_evidence_file,
                "parent_after_child_exit": after_child,
                "materialized_inputs_stable": materialized_stable,
                "source_inputs_stable": source_stable,
                "generated_dataset_outputs_absent": generated_absent,
            }
        )
        if not invocation_succeeded or not child_contract_valid:
            break
    evaluation = _evaluate_parent(
        initial=initial_parent,
        cases=cases,
        expected_case_count=arguments.case_count,
    )
    evidence = support._signed(
        {
            "schema_version": 1,
            "artifact_kind": QUALIFICATION_KIND,
            "qualification_id": QUALIFICATION_ID,
            "passed": evaluation["passed"],
            "canonical_parameters": canonical,
            "attempt_marker": support._bound_file(
                attempt, label="qualification attempt marker"
            ),
            "host": {
                "hostname": socket.gethostname(),
                "physical_gpu_index": PHYSICAL_GPU_INDEX,
            },
            "runtime_bindings": {
                "parent_python": parent_python,
                "python": python_binding,
                "code": code_binding,
                "deform360": deform360_binding,
                "qualification_source": source_binding,
                "numerical_adapter_source": numerical_binding,
            },
            "process_boundary": {
                "one_original_trainer_per_child": True,
                "one_official_case_lifecycle_per_child": True,
                "fits_per_case": CANONICAL_FITS_PER_CASE,
                "trainer_configuration_overridden": False,
                "process_exit_reclaims_case_resources": True,
                "parent_process_imports_nerfstudio": False,
            },
            "cases": cases,
            "evaluation": evaluation,
            "information_boundary": {
                "formal_held_path_supplied": False,
                "target_query_path_received": False,
                "outcome_path_received": False,
                "gate_path_received": False,
                "score_path_received": False,
            },
        }
    )
    support._write_new_json(
        output / "process-isolation-qualification.json", evidence
    )
    return 0 if evaluation["passed"] else 3


def _add_child_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--deform360-repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--fit-count", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--code-root", type=Path, required=True)
    run.add_argument("--python", type=Path, default=support.PINNED_PYTHON)
    run.add_argument(
        "--deform360-repo", type=Path, default=support.PINNED_DEFORM360
    )
    run.add_argument(
        "--dataset", type=Path, default=support.DEFAULT_PUBLIC_DEV_DATASET
    )
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--case-count", type=int, default=CANONICAL_CASE_COUNT)
    run.add_argument("--fit-count", type=int, default=CANONICAL_FITS_PER_CASE)
    run.add_argument("--iterations", type=int, default=CANONICAL_ITERATIONS)
    run.add_argument("--seed", type=int, default=CANONICAL_SEED)
    run.add_argument(
        "--cuda-device", type=int, choices=(0, 1), default=PHYSICAL_GPU_INDEX
    )
    run.add_argument(
        "--case-timeout-seconds", type=int, default=CASE_TIMEOUT_SECONDS
    )
    child = commands.add_parser("_case-child")
    _add_child_arguments(child)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "run":
        return _run(arguments)
    if arguments.command == "_case-child":
        return _case_child(arguments)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
