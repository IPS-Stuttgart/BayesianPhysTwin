from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_operator() -> ModuleType:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "development"
        / "qualify_deform360_process_isolation.py"
    )
    name = "_test_deform360_process_isolation_qualification"
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


operator = _load_operator()


def _boundary(fd: int, tasks: int) -> dict[str, int]:
    return {
        "file_descriptor_count": fd,
        "task_count": tasks,
        "rss_kib": 1000,
        "rlimit_nofile_soft": 1024,
        "rlimit_nofile_hard": 1048576,
    }


def _fits(count: int = 81) -> list[dict[str, object]]:
    return [
        {
            "fit_index": index,
            "output_created": True,
            "output_absent_after_cleanup": True,
            "generated_outputs_absent_after_cleanup": True,
            "resource_boundary_stage": "after_cleanup",
            "resource_boundary": _boundary(52 + 4 * index, 133 + 2 * index),
        }
        for index in range(count)
    ]


def test_child_gate_accepts_one_original_81_fit_lifecycle() -> None:
    evaluation = operator._evaluate_child(
        before=_boundary(48, 131),
        fits=_fits(),
        final_globals={
            "event_writer_count": 81,
            "event_storage_count": 0,
            "global_buffer_key_count": 5,
            "profiler_count": 81,
            "pytorch_profiler_present": 0,
        },
        expected_fit_count=81,
    )

    assert evaluation["passed"] is True
    assert evaluation["observed"]["maximum_file_descriptor_count"] == 372
    assert evaluation["observed"]["unused_file_descriptors_at_peak"] == 652
    assert evaluation["observed"]["maximum_task_growth"] == 162


def test_child_gate_rejects_exhausted_fd_margin_or_wrong_trainer_path() -> None:
    fits = _fits()
    fits[-1]["resource_boundary"] = _boundary(800, 293)
    evaluation = operator._evaluate_child(
        before=_boundary(48, 131),
        fits=fits,
        final_globals={
            "event_writer_count": 80,
            "event_storage_count": 0,
            "global_buffer_key_count": 5,
            "profiler_count": 81,
            "pytorch_profiler_present": 0,
        },
        expected_fit_count=81,
    )

    assert evaluation["passed"] is False
    assert (
        evaluation["predicates"]["child_retains_fd_safety_margin"] is False
    )
    assert evaluation["predicates"]["original_writer_path_exercised"] is False


def _case(index: int, *, parent_fd: int = 5) -> dict[str, object]:
    return {
        "case_index": index,
        "invocation": {
            "return_code": 0,
            "timed_out": False,
        },
        "child_evidence": {
            "process_id": 2000 + index,
            "passed": True,
            "evaluation": {
                "observed": {
                    "initial": _boundary(48, 131),
                }
            },
        },
        "child_contract_valid": True,
        "parent_after_child_exit": _boundary(parent_fd, 1),
        "materialized_inputs_stable": True,
        "source_inputs_stable": True,
        "generated_dataset_outputs_absent": True,
    }


def test_parent_gate_requires_distinct_children_and_no_resource_growth() -> None:
    cases = [_case(index) for index in range(4)]
    evaluation = operator._evaluate_parent(
        initial=_boundary(4, 1),
        cases=cases,
        expected_case_count=4,
    )
    assert evaluation["passed"] is True

    cases[3]["child_evidence"]["process_id"] = 2000
    cases[2]["parent_after_child_exit"] = _boundary(7, 1)
    rejected = operator._evaluate_parent(
        initial=_boundary(4, 1),
        cases=cases,
        expected_case_count=4,
    )
    assert rejected["passed"] is False
    assert rejected["predicates"]["all_child_processes_distinct"] is False
    assert rejected["predicates"]["parent_fd_growth_within_limit"] is False


def test_child_evidence_contract_is_signed_and_exact() -> None:
    value = operator.support._signed(
        {
            "artifact_kind": operator.CHILD_KIND,
            "qualification_id": operator.QUALIFICATION_ID,
            "case_index": 2,
            "passed": True,
            "parameters": {
                "fit_count": 81,
                "iterations_per_fit": 1,
                "seed": 0,
                "trainer_instance_count": 1,
                "trainer_variant": "original-pinned-default",
            },
            "information_boundary": {
                "formal_held_path_supplied": False,
                "target_query_path_received": False,
                "outcome_path_received": False,
                "gate_path_received": False,
                "score_path_received": False,
            },
            "evaluation": {
                "passed": True,
                "predicates": {"all": True},
            },
        }
    )
    operator._validate_child_evidence(
        value,
        expected_case_index=2,
        expected_fit_count=81,
        expected_iterations=1,
        expected_seed=0,
    )

    value["parameters"]["trainer_variant"] = "resource-bounded"
    value = operator.support._signed(value)
    with pytest.raises(ValueError, match="parameters changed"):
        operator._validate_child_evidence(
            value,
            expected_case_index=2,
            expected_fit_count=81,
            expected_iterations=1,
            expected_seed=0,
        )


def test_operator_surface_accepts_no_target_or_score_paths() -> None:
    parser = operator._parser()
    run = parser.parse_args(
        [
            "run",
            "--code-root",
            "/code",
            "--output-dir",
            "/evidence",
        ]
    )
    assert run.case_count == 4
    assert run.fit_count == 81
    assert run.iterations == 1
    assert {
        action.dest
        for action in parser._subparsers._group_actions[0]
        .choices["run"]
        ._actions
    }.isdisjoint(
        {
            "target_query",
            "outcome",
            "gate",
            "score",
            "held_root",
        }
    )
