from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from bayesian_phystwin import deform360_held_v8_outcome_driver as driver
from bayesian_phystwin import deform360_held_v8_protocol as protocol


EXPECTED_CALIBRATION = (
    "072-cotton-clohesline-ep0003",
    "002-rope-silk-ep0004",
    "002-rope-silk-ep0008",
    "083-blanket-cloth-ep0000",
    "083-blanket-cloth-ep0003",
    "083-blanket-cloth-ep0006",
    "085-scarf-cloth-ep0000",
    "085-scarf-cloth-ep0005",
    "085-scarf-cloth-ep0007",
    "092-squirrel-ep0002",
    "092-squirrel-ep0003",
    "092-squirrel-ep0006",
    "170-spider-ep0002",
    "170-spider-ep0004",
    "170-spider-ep0007",
)
EXPECTED_CONFIRMATION = (
    "002-rope-silk-ep0001",
    "081-stripe-rope-ep0005",
    "085-scarf-cloth-ep0002",
    "083-blanket-cloth-ep0007",
    "092-squirrel-ep0001",
    "170-spider-ep0006",
)


def test_driver_uses_the_exact_frozen_v8_cohorts() -> None:
    assert protocol.CALIBRATION_CASE_NAMES == EXPECTED_CALIBRATION
    assert protocol.CONFIRMATION_CASE_NAMES == EXPECTED_CONFIRMATION
    assert "002-rope-silk-ep0003" not in protocol.CALIBRATION_CASE_NAMES
    assert protocol.CALIBRATION_CASE_NAMES.count("072-cotton-clohesline-ep0003") == 1


def test_query_subprocess_has_only_x0_field_lock_and_outputs_as_data_paths(
    tmp_path: Path,
) -> None:
    code = tmp_path / ("code-" + "a" * 40)
    lock = tmp_path / "calibration-lock.json"
    x0 = tmp_path / "query-inputs" / "case" / "official-frame-zero-query.json"
    field = tmp_path / "cases" / "case" / "preoutcome-frozen-field.json"
    output_archive = tmp_path / "query-outputs" / "case" / "prediction.npz"
    output_seal = tmp_path / "query-outputs" / "case" / "prediction.json"
    target = tmp_path / "private-targets" / "case" / "official-target.json"
    argv, environment, safe_cwd = driver.build_query_subprocess(
        deployed_code=code,
        lock_path=lock,
        official_query_manifest_path=x0,
        frozen_field_manifest_path=field,
        output_archive_path=output_archive,
        output_seal_path=output_seal,
    )
    serialized = json.dumps({"argv": argv, "environment": environment, "cwd": safe_cwd})
    assert str(target) not in serialized
    assert "official-target" not in serialized
    assert "visibility" not in serialized
    assert "validity" not in serialized
    assert "score" not in serialized
    flags = {value for value in argv if value.startswith("--")}
    assert flags == {
        "--lock",
        "--official-query-manifest",
        "--frozen-field-manifest",
        "--output-archive",
        "--output-seal",
    }
    assert set(environment.values()).isdisjoint(
        {str(lock), str(x0), str(field), str(output_archive), str(output_seal)}
    )
    assert safe_cwd == str(tmp_path / "query-outputs" / "case")
    assert str(target.parent) not in serialized
    assert "private-targets" not in serialized


def test_query_paths_are_lexically_disjoint_from_protected_future_paths(
    tmp_path: Path,
) -> None:
    layout = driver.build_layout(
        root=tmp_path,
        role="calibration",
        cases=("001-object-ep0001",),
    )
    paths = layout.cases["001-object-ep0001"]
    assert layout.protected_future_root not in paths.official_query_manifest.parents
    assert layout.protected_future_root not in paths.queried_seal.parents
    assert layout.x0_query_root not in paths.target_manifest.parents
    assert layout.queried_prediction_root not in paths.target_manifest.parents


def test_deployed_git_tree_is_detached_clean_bound_and_tamper_evident(
    tmp_path: Path,
) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(seed), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Held Test")
    tracked = seed / "tracked.txt"
    tracked.write_text("sealed\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-q", "-m", "sealed tree")
    head = git("rev-parse", "HEAD")
    git("checkout", "-q", "--detach", head)
    code = tmp_path / f"code-{head}"
    seed.rename(code)
    evidence = driver._deployed_repository_evidence(code)
    bindings = {
        "method_head_text_sha256": evidence["head_text_sha256"],
        "method_deployed_snapshot_tree": evidence["tree_sha256"],
    }
    assert driver._validate_deployed_repository(code, bindings) == evidence
    (code / "tracked.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="worktree is not completely clean"):
        driver._validate_deployed_repository(code, bindings)


@dataclass
class _Inputs:
    case_name: str
    permit_evidence: Mapping[str, Any]

    def scoring_kwargs(self) -> dict[str, Any]:
        return {"case_name": self.case_name}


class _FakeBackend:
    def __init__(self, *, events: list[str], **_kwargs: Any) -> None:
        self.events = events
        events.append("backend")


class _FakeProtocol:
    PROTOCOL_ID = driver.PROTOCOL_ID
    TARGET_RECONSTRUCTION_OPERATION = "create-official-target-v1"
    FUTURE_SCORE_OPERATION = "read-official-target-for-score-v1"

    def __init__(
        self,
        cases: tuple[str, ...],
        events: list[str],
        *,
        role: str = "calibration",
        reject_lock: bool = False,
    ) -> None:
        self.cases = cases
        self.events = events
        self.role = role
        self.reject_lock = reject_lock
        self.FRESH_REPLACEMENT_CASE_NAME = (
            cases[0] if role == "calibration" else "not-in-confirmation-cohort"
        )

    def validate_protocol_lock(self, _path: str) -> dict[str, Any]:
        self.events.append("lock")
        if self.reject_lock:
            raise ValueError("confirmation remains inaccessible before GO")
        return {"stage": self.role}

    def locked_case_names(self, _path: str, *, role: str) -> tuple[str, ...]:
        assert role == self.role
        return self.cases

    def validate_first_cohort_barrier(self, _path: str, **_kwargs: Any) -> Any:
        self.events.append("barrier1")
        return SimpleNamespace(barrier_sha256="1" * 64)

    def authorize_target_reconstruction_capabilities(
        self, _path: str, **_kwargs: Any
    ) -> dict[str, object]:
        self.events.append("target-caps")
        return {case: object() for case in self.cases}

    def consume_case_capability(
        self, _permit: object, *, case_name: str, operation: str
    ) -> dict[str, Any]:
        self.events.append(f"consume:{operation}:{case_name}")
        return {
            "protocol_id": self.PROTOCOL_ID,
            "case_name": case_name,
            "operation": operation,
            "single_use_consumed": True,
        }

    def replacement_source_permit_evidence(self, _path: str) -> dict[str, Any]:
        return {"operation": "source"}

    def confirmation_source_permit_evidence(self, _path: str) -> dict[str, Any]:
        return {"operation": "confirmation-source"}

    def validate_second_cohort_barrier(self, _path: str, **_kwargs: Any) -> Any:
        self.events.append("barrier2")
        return SimpleNamespace(barrier_sha256="2" * 64)

    def authorize_future_score_capabilities(
        self, _path: str, **_kwargs: Any
    ) -> dict[str, object]:
        self.events.append("score-caps")
        return {case: object() for case in self.cases}


def _fake_post(
    root: Path,
    events: list[str],
    *,
    decision_label: str,
) -> driver.PostBarrierApi:
    def reconstruct(**kwargs: Any) -> dict[str, Any]:
        case = kwargs["case_name"]
        events.append(f"reconstruct:{case}")
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True)
        return {"case_name": case}

    def write_target_and_query(
        target_archive: Path,
        target_manifest: Path,
        query_archive: Path,
        query_manifest: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        case = kwargs["case_name"]
        kwargs["consume_target_reconstruction_permit"](
            kwargs["target_reconstruction_permit"],
            case_name=case,
            operation="create-official-target-v1",
        )
        kwargs["reconstruction_loader"]()
        for path in (target_archive, target_manifest, query_archive, query_manifest):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(case, encoding="utf-8")
        return {"case_name": case}

    def validate_replacement_source(_path: str, **_kwargs: Any) -> dict[str, Any]:
        aligned = root / "replacement-aligned"
        aligned.mkdir(exist_ok=True)
        return {"aligned_episode_dir": str(aligned)}

    def validate_confirmation_source(path: str, **_kwargs: Any) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def validate_frozen_field(_path: Path, **_kwargs: Any) -> dict[str, Any]:
        return {"source_array_records": {"frame_zero_points_m": {"shape": [24, 3]}}}

    def load_scoring_inputs(**kwargs: Any) -> _Inputs:
        case = kwargs["case_name"]
        evidence = kwargs["consume_future_score_permit"](
            kwargs["future_score_permit"],
            case_name=case,
            operation="read-official-target-for-score-v1",
        )
        events.append(f"score-load:{case}")
        return _Inputs(case_name=case, permit_evidence=evidence)

    def score_case(*, case_name: str) -> dict[str, Any]:
        events.append(f"score:{case_name}")
        return {"case_name": case_name}

    def create_score(evidence_path: Path, decision_path: Path, **_kwargs: Any):
        gate_result = {"passed": decision_label in {"GO", "CONFIRMED"}}
        evidence = {"gate_result": gate_result}
        evidence["artifact_sha256"] = driver._execution_completion_artifact_sha256(
            evidence
        )
        decision = {"decision": decision_label, "gate_result": gate_result}
        decision["artifact_sha256"] = driver._execution_completion_artifact_sha256(
            decision
        )
        evidence_path.write_text(
            json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8"
        )
        evidence_path.chmod(0o400)
        decision_path.write_text(
            json.dumps(decision, sort_keys=True) + "\n", encoding="utf-8"
        )
        decision_path.chmod(0o400)
        events.append("decision")
        return evidence, decision

    return driver.PostBarrierApi(
        backend_type=lambda **kwargs: _FakeBackend(events=events, **kwargs),
        reconstruct=reconstruct,
        write_target_and_query=write_target_and_query,
        validate_target=lambda *_args, **_kwargs: {},
        validate_frozen_field=validate_frozen_field,
        validate_queried_prediction=lambda *_args, **_kwargs: {},
        load_scoring_inputs=load_scoring_inputs,
        score_case=score_case,
        create_score_evidence_and_decision=create_score,
        validate_replacement_source=validate_replacement_source,
        validate_confirmation_source=validate_confirmation_source,
    )


def _fake_outcome_execution(
    tmp_path: Path,
    *,
    role: str,
    decision_label: str,
) -> tuple[
    driver.DriverArguments,
    _FakeProtocol,
    driver.PostBarrierApi,
    list[str],
    tuple[str, ...],
    Any,
]:
    events: list[str] = []
    count = 15 if role == "calibration" else 6
    cases = tuple(f"{index:03d}-object-ep{index:04d}" for index in range(count))
    lock = tmp_path / f"{role}-lock.json"
    lock.write_text("lock", encoding="utf-8")
    lock.chmod(0o400)
    (tmp_path / role).mkdir()
    source_manifest = driver.canonical_role_source_manifest_path(tmp_path, role)
    source_manifest.parent.mkdir(parents=True)
    source_value: dict[str, Any] = {
        "protocol_id": driver.PROTOCOL_ID,
        "role": role,
    }
    if role == "confirmation":
        source_value.update(
            {
                "source_root": str(source_manifest.parent.parent),
                "cases": [
                    {
                        "case_name": case,
                        "aligned_episode_relative_path": (
                            f"aligned/{case.rpartition('-ep')[0]}/"
                            f"episode_{case.rpartition('-ep')[2]}"
                        ),
                    }
                    for case in cases
                ],
            }
        )
    source_value["artifact_sha256"] = driver._execution_completion_artifact_sha256(
        source_value
    )
    source_manifest.write_text(
        json.dumps(source_value, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_manifest.chmod(0o400)
    replacement: Path | None = source_manifest if role == "calibration" else None
    confirmation: Path | None = source_manifest if role == "confirmation" else None
    arguments = driver.DriverArguments(
        role=role,
        deployed_code=str(tmp_path / ("code-" + "a" * 40)),
        lock_path=str(lock),
        replacement_source_manifest_path=(str(replacement) if replacement else None),
        dry_run_barrier_only=False,
        confirmation_source_manifest_path=(str(confirmation) if confirmation else None),
        aligned_root=str(tmp_path / "aligned"),
        deform360_repo="d",
        sam2_repository="s",
        sam2_checkpoint="sc",
        cotracker_repo="c",
        cotracker_checkpoint="cc",
    )
    fake_protocol = _FakeProtocol(cases, events, role=role)
    post = _fake_post(tmp_path, events, decision_label=decision_label)

    def query_runner(**kwargs: Any) -> None:
        case = Path(kwargs["official_query_manifest_path"]).parent.name
        Path(kwargs["output_archive_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_archive_path"]).write_text(case, encoding="utf-8")
        Path(kwargs["output_seal_path"]).write_text(case, encoding="utf-8")
        events.append(f"query:{case}")

    return arguments, fake_protocol, post, events, cases, query_runner


@pytest.mark.parametrize(
    ("role", "decision_label", "expected_return_code"),
    (
        ("calibration", "GO", 0),
        ("calibration", "NO-GO", driver.NO_GO_EXIT_CODE),
        ("confirmation", "CONFIRMED", 0),
        ("confirmation", "NOT-CONFIRMED", driver.NOT_CONFIRMED_EXIT_CODE),
    ),
)
def test_every_semantic_outcome_publishes_then_seals_and_preserves_return_code(
    tmp_path: Path,
    role: str,
    decision_label: str,
    expected_return_code: int,
) -> None:
    arguments, fake_protocol, post, events, cases, query_runner = (
        _fake_outcome_execution(tmp_path, role=role, decision_label=decision_label)
    )
    result = driver.execute_outcomes(
        arguments,
        protocol=fake_protocol,
        deployment_verifier=lambda _arguments: events.append("verify"),
        smoke_gsplat_runtime=lambda: {"artifact_sha256": "a" * 64},
        load_post_barrier_api=lambda: post,
        query_runner=query_runner,
        validate_runtime=lambda _arguments: None,
        rlimit_nofile_getter=lambda: (1024, 4096),
        role_sealer=lambda sealed: events.append(f"seal:{sealed.role}"),
        formal_paths=False,
    )
    assert result == expected_return_code
    assert events.count(f"seal:{role}") == 1
    layout = driver.build_layout(root=tmp_path, role=role, cases=cases)
    completion = driver.validate_role_execution_completion(
        layout.execution_completion_path,
        lock_path=arguments.lock_path,
        expected_role=role,
        expected_ordered_case_names=cases,
    )
    assert completion["semantic_decision"]["semantic_outcome"] == decision_label
    assert (
        completion["semantic_decision"]["semantic_return_code"] == expected_return_code
    )
    resource_boundary = completion["resource_boundary"]
    assert len(resource_boundary["post_cases"]) == len(cases)
    assert resource_boundary["publication"]["open_descriptor_delta_from_end"] == 2
    assert resource_boundary["publication"]["file_descriptor_count"] == (
        resource_boundary["end_outcome"]["file_descriptor_count"] + 2
    )
    assert resource_boundary["publication"]["marker_fd_open"] is True
    assert resource_boundary["publication"]["parent_directory_fd_open"] is True


@pytest.mark.parametrize("drift_call", (9, 10))
def test_final_or_open_descriptor_boundary_drift_leaves_no_marker_or_seal(
    tmp_path: Path,
    drift_call: int,
) -> None:
    arguments, fake_protocol, post, events, cases, query_runner = (
        _fake_outcome_execution(
            tmp_path, role="confirmation", decision_label="CONFIRMED"
        )
    )
    calls = 0

    def limits() -> tuple[int, int]:
        nonlocal calls
        calls += 1
        return (1024, 8192) if calls == drift_call else (1024, 4096)

    with pytest.raises(ValueError, match="RLIMIT_NOFILE soft/hard pair changed"):
        driver.execute_outcomes(
            arguments,
            protocol=fake_protocol,
            deployment_verifier=lambda _arguments: events.append("verify"),
            smoke_gsplat_runtime=lambda: {"artifact_sha256": "a" * 64},
            load_post_barrier_api=lambda: post,
            query_runner=query_runner,
            validate_runtime=lambda _arguments: None,
            rlimit_nofile_getter=limits,
            role_sealer=lambda sealed: events.append(f"seal:{sealed.role}"),
            formal_paths=False,
        )
    layout = driver.build_layout(root=tmp_path, role="confirmation", cases=cases)
    assert not layout.execution_completion_path.exists()
    assert not any(event.startswith("seal:") for event in events)


@pytest.mark.parametrize("publication_count", (101, 103))
def test_publication_fd_delta_must_be_exactly_two(
    tmp_path: Path,
    publication_count: int,
) -> None:
    arguments, fake_protocol, post, events, cases, query_runner = (
        _fake_outcome_execution(
            tmp_path, role="confirmation", decision_label="CONFIRMED"
        )
    )
    fd_counts = iter([100] * 8 + [publication_count])

    with pytest.raises(
        ValueError,
        match="execution completion descriptors violate the final FD boundary",
    ):
        driver.execute_outcomes(
            arguments,
            protocol=fake_protocol,
            deployment_verifier=lambda _arguments: events.append("verify"),
            smoke_gsplat_runtime=lambda: {"artifact_sha256": "a" * 64},
            load_post_barrier_api=lambda: post,
            query_runner=query_runner,
            validate_runtime=lambda _arguments: None,
            fd_counter=lambda: next(fd_counts),
            rlimit_nofile_getter=lambda: (1024, 4096),
            role_sealer=lambda sealed: events.append(f"seal:{sealed.role}"),
            formal_paths=False,
        )

    layout = driver.build_layout(root=tmp_path, role="confirmation", cases=cases)
    assert not layout.execution_completion_path.exists()
    assert not any(event.startswith("seal:") for event in events)


def test_sealer_failure_does_not_return_the_semantic_result(tmp_path: Path) -> None:
    arguments, fake_protocol, post, events, cases, query_runner = (
        _fake_outcome_execution(
            tmp_path, role="confirmation", decision_label="NOT-CONFIRMED"
        )
    )

    def fail_sealer(_arguments: driver.DriverArguments) -> None:
        events.append("seal-attempt")
        raise RuntimeError("integrity sealer failed")

    with pytest.raises(RuntimeError, match="integrity sealer failed"):
        driver.execute_outcomes(
            arguments,
            protocol=fake_protocol,
            deployment_verifier=lambda _arguments: events.append("verify"),
            smoke_gsplat_runtime=lambda: {"artifact_sha256": "a" * 64},
            load_post_barrier_api=lambda: post,
            query_runner=query_runner,
            validate_runtime=lambda _arguments: None,
            rlimit_nofile_getter=lambda: (1024, 4096),
            role_sealer=fail_sealer,
            formal_paths=False,
        )
    layout = driver.build_layout(root=tmp_path, role="confirmation", cases=cases)
    assert layout.execution_completion_path.is_file()
    assert "seal-attempt" in events
    assert not (
        tmp_path / "confirmation" / "confirmation-outcome-integrity-completion.json"
    ).exists()


def test_open_writable_role_log_blocks_marker_and_sealer(tmp_path: Path) -> None:
    arguments, fake_protocol, post, events, cases, query_runner = (
        _fake_outcome_execution(
            tmp_path, role="confirmation", decision_label="CONFIRMED"
        )
    )
    log_path = tmp_path / "confirmation" / "driver.log"
    with log_path.open("w", encoding="utf-8") as log_stream:
        log_stream.write("open during outcome\n")
        log_stream.flush()
        with pytest.raises(ValueError, match="writable descriptor into the held root"):
            driver.execute_outcomes(
                arguments,
                protocol=fake_protocol,
                deployment_verifier=lambda _arguments: events.append("verify"),
                smoke_gsplat_runtime=lambda: {"artifact_sha256": "a" * 64},
                load_post_barrier_api=lambda: post,
                query_runner=query_runner,
                validate_runtime=lambda _arguments: None,
                rlimit_nofile_getter=lambda: (1024, 4096),
                role_sealer=lambda sealed: events.append(f"seal:{sealed.role}"),
                formal_paths=False,
            )
    layout = driver.build_layout(root=tmp_path, role="confirmation", cases=cases)
    assert not layout.execution_completion_path.exists()
    assert not any(event.startswith("seal:") for event in events)


def test_dry_run_never_publishes_or_invokes_the_role_sealer(tmp_path: Path) -> None:
    events: list[str] = []
    cases = tuple(f"{index:03d}-object-ep{index:04d}" for index in range(15))
    lock = tmp_path / "calibration-lock.json"
    lock.write_text("lock", encoding="utf-8")
    (tmp_path / "calibration").mkdir()
    replacement = driver.canonical_role_source_manifest_path(tmp_path, "calibration")
    replacement.parent.mkdir(parents=True)
    replacement_value = {
        "protocol_id": driver.PROTOCOL_ID,
        "role": "calibration",
    }
    replacement_value["artifact_sha256"] = driver._execution_completion_artifact_sha256(
        replacement_value
    )
    replacement.write_text(
        json.dumps(replacement_value, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    replacement.chmod(0o400)
    arguments = driver.DriverArguments(
        role="calibration",
        deployed_code=str(tmp_path / ("code-" + "a" * 40)),
        lock_path=str(lock),
        replacement_source_manifest_path=str(replacement),
        dry_run_barrier_only=True,
    )
    result = driver.execute_outcomes(
        arguments,
        protocol=_FakeProtocol(cases, events),
        deployment_verifier=lambda _arguments: events.append("verify"),
        smoke_gsplat_runtime=lambda: {"artifact_sha256": "a" * 64},
        load_post_barrier_api=lambda: pytest.fail("post-barrier API loaded"),
        rlimit_nofile_getter=lambda: pytest.fail("NOFILE inspected in dry-run"),
        role_sealer=lambda _arguments: pytest.fail("dry-run invoked sealer"),
        formal_paths=False,
    )
    assert result == 0
    layout = driver.build_layout(root=tmp_path, role="calibration", cases=cases)
    assert not layout.execution_completion_path.exists()
    assert "barrier1" in events
    assert "target-caps" in events


def test_fresh_output_preflight_rejects_a_preexisting_execution_completion(
    tmp_path: Path,
) -> None:
    role_root = tmp_path / "calibration"
    role_root.mkdir()
    layout = driver.build_layout(
        root=tmp_path, role="calibration", cases=("001-object-ep0001",)
    )
    layout.execution_completion_path.write_text("stale\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fresh role execution completion"):
        driver._prepare_fresh_outputs(layout)
    assert not (role_root / ".v8-outcome-phase.claim").exists()


def test_role_sealer_subprocess_is_exact_pinned_and_isolated(tmp_path: Path) -> None:
    code = tmp_path / ("code-" + "a" * 40)
    lock = tmp_path / "confirmation-lock.json"
    arguments = driver.DriverArguments(
        role="confirmation",
        deployed_code=str(code),
        lock_path=str(lock),
        replacement_source_manifest_path=None,
        dry_run_barrier_only=False,
    )
    argv, environment, cwd = driver.build_role_outcome_sealer_subprocess(arguments)
    assert argv == (
        str(driver.PINNED_PYTHON),
        "-I",
        "-B",
        "-X",
        f"pycache_prefix={driver.PYCACHE_PREFIX}",
        str(code / "scripts" / "held" / "seal_deform360_v8_role_outcome.py"),
        "--role",
        "confirmation",
        "--lock",
        str(lock),
        "--deployed-code",
        str(code),
    )
    assert environment == driver._normalized_environment()
    assert cwd == str(code)


def test_role_sealer_runner_checks_live_limit_and_captures_terminal_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "held-v8"
    code = root / ("code-" + "a" * 40)
    lock = root / "confirmation-lock.json"
    arguments = driver.DriverArguments(
        role="confirmation",
        deployed_code=str(code),
        lock_path=str(lock),
        replacement_source_manifest_path=None,
        dry_run_barrier_only=False,
    )
    marker = driver.canonical_role_execution_completion_path(root, "confirmation")
    completion = {
        "resource_boundary": {
            "initial_nofile": {
                "rlimit_nofile_soft": 1024,
                "rlimit_nofile_hard": 4096,
            }
        },
        "semantic_decision": {"semantic_outcome": "NOT-CONFIRMED"},
    }
    monkeypatch.setattr(
        driver,
        "validate_role_execution_completion",
        lambda path, **_kwargs: completion if Path(path) == marker else pytest.fail(),
    )
    monkeypatch.setattr(driver, "_rlimit_nofile_pair", lambda: (1024, 4096))
    monkeypatch.setattr(
        driver,
        "_validate_no_writable_held_descriptors",
        lambda held_root: {"held_root": str(held_root)},
    )
    monkeypatch.setattr(
        driver,
        "build_role_outcome_sealer_subprocess",
        lambda _arguments: (("sealed-python",), {"SEALED": "1"}, "/sealed"),
    )
    observed: dict[str, Any] = {}

    def run(argv: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        observed.update({"argv": argv, **kwargs})
        payload = {
            "event": "DEFORM360_V8_ROLE_OUTCOME_INTEGRITY_COMPLETE",
            "role": "confirmation",
            "terminal_outcome": "NOT-CONFIRMED",
            "role_completion_path": str(
                root / "confirmation" / "confirmation-outcome-integrity-completion.json"
            ),
            "role_completion_artifact_sha256": "a" * 64,
            "terminal_root_finalized": True,
        }
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(json.dumps(payload) + "\n").encode("utf-8"),
            stderr=b"",
        )

    monkeypatch.setattr(driver.subprocess, "run", run)
    driver.run_role_outcome_sealer(arguments)
    assert observed["argv"] == ("sealed-python",)
    assert observed["stdout"] is subprocess.PIPE
    assert observed["stderr"] is subprocess.PIPE
    assert observed["close_fds"] is True

    monkeypatch.setattr(driver, "_rlimit_nofile_pair", lambda: (1024, 8192))
    with pytest.raises(ValueError, match="soft/hard pair changed"):
        driver.run_role_outcome_sealer(arguments)


def test_source_only_main_path_never_reaches_normalization_or_sealing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    namespace = SimpleNamespace(
        source_only_verifier=True,
        promote_only=False,
        deployed_code="/sealed/code",
        lock="/sealed/calibration-lock.json",
    )
    protocol_stub = object()
    monkeypatch.setattr(driver, "_parse_args", lambda _role: namespace)
    monkeypatch.setattr(
        driver,
        "_load_protocol",
        lambda _code: (protocol_stub, Path("/sealed/code/src")),
    )
    monkeypatch.setattr(
        driver,
        "verify_source_only_deployment",
        lambda **_kwargs: events.append("source-only"),
    )
    monkeypatch.setattr(
        driver,
        "_normalize_or_reexec",
        lambda: pytest.fail("source-only normalized"),
    )
    monkeypatch.setattr(
        driver,
        "run_role_outcome_sealer",
        lambda _arguments: pytest.fail("source-only sealed"),
    )
    assert driver.main_for_role("calibration") == 0
    assert events == ["source-only"]


def test_promote_only_main_path_never_invokes_the_role_sealer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    namespace = SimpleNamespace(
        source_only_verifier=False,
        promote_only=True,
        deployed_code="/sealed/code",
    )

    class _PromotionProtocol:
        @staticmethod
        def create_confirmation_protocol_lock(*_args: object) -> None:
            events.append("promote")

    monkeypatch.setattr(driver, "_parse_args", lambda _role: namespace)
    monkeypatch.setattr(
        driver,
        "_load_protocol",
        lambda _code: (_PromotionProtocol(), Path("/sealed/code/src")),
    )
    monkeypatch.setattr(driver, "_normalize_or_reexec", lambda: None)
    monkeypatch.setattr(
        driver,
        "run_source_only_deployment_verifier",
        lambda _arguments: events.append("verify"),
    )
    monkeypatch.setattr(
        driver,
        "run_role_outcome_sealer",
        lambda _arguments: pytest.fail("promotion invoked outcome sealer"),
    )
    assert driver.main_for_role("confirmation") == 0
    assert events == ["verify", "promote"]


def test_two_barrier_order_and_no_go_never_promotes_confirmation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []
    cases = tuple(f"{index:03d}-object-ep{index:04d}" for index in range(15))
    lock = tmp_path / "calibration-lock.json"
    lock.write_text("lock", encoding="utf-8")
    lock.chmod(0o400)
    (tmp_path / "calibration").mkdir()
    replacement = driver.canonical_role_source_manifest_path(tmp_path, "calibration")
    replacement.parent.mkdir(parents=True)
    replacement_value = {
        "protocol_id": driver.PROTOCOL_ID,
        "role": "calibration",
    }
    replacement_value["artifact_sha256"] = driver._execution_completion_artifact_sha256(
        replacement_value
    )
    replacement.write_text(
        json.dumps(replacement_value, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    replacement.chmod(0o400)
    fake_protocol = _FakeProtocol(cases, events)
    post = _fake_post(tmp_path, events, decision_label="NO-GO")
    arguments = driver.DriverArguments(
        role="calibration",
        deployed_code=str(tmp_path / ("code-" + "a" * 40)),
        lock_path=str(lock),
        replacement_source_manifest_path=str(replacement),
        dry_run_barrier_only=False,
        aligned_root=str(tmp_path / "aligned"),
        deform360_repo="d",
        sam2_repository="s",
        sam2_checkpoint="sc",
        cotracker_repo="c",
        cotracker_checkpoint="cc",
    )

    def query_runner(**kwargs: Any) -> None:
        case = Path(kwargs["official_query_manifest_path"]).parents[0].name
        target_paths = [
            value
            for key, value in kwargs.items()
            if "target" in key or "outcome" in key
        ]
        assert target_paths == []
        Path(kwargs["output_archive_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_archive_path"]).write_text(case, encoding="utf-8")
        Path(kwargs["output_seal_path"]).write_text(case, encoding="utf-8")
        events.append(f"query:{case}")

    result = driver.execute_outcomes(
        arguments,
        protocol=fake_protocol,
        deployment_verifier=lambda _arguments: events.append("verify"),
        smoke_gsplat_runtime=lambda: (
            events.append("smoke") or {"artifact_sha256": "a" * 64}
        ),
        load_post_barrier_api=lambda: events.append("post-load") or post,
        query_runner=query_runner,
        validate_runtime=lambda _arguments: events.append("runtime"),
        rlimit_nofile_getter=lambda: (1024, 4096),
        role_sealer=lambda sealed: events.append(f"seal:{sealed.role}"),
        formal_paths=False,
    )
    assert result == driver.NO_GO_EXIT_CODE
    assert events[:5] == ["verify", "lock", "smoke", "barrier1", "target-caps"]
    assert events.index("runtime") > events.index("target-caps")
    assert events.index("post-load") > events.index("barrier1")
    for case in cases:
        assert events.index(f"reconstruct:{case}") < events.index(f"query:{case}")
    first_query = min(
        index for index, event in enumerate(events) if event.startswith("query:")
    )
    last_reconstruction = max(
        index for index, event in enumerate(events) if event.startswith("reconstruct:")
    )
    last_query = max(
        index for index, event in enumerate(events) if event.startswith("query:")
    )
    assert first_query < last_reconstruction < last_query < events.index("barrier2")
    assert events.index("barrier2") < events.index("score-caps")
    assert events[-1] == "seal:calibration"
    assert not (tmp_path / "confirmation-lock.json").exists()
    emitted = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    initial_limit = next(
        row for row in emitted if row["event"] == "QUALIFIED_RLIMIT_NOFILE_CAPTURED"
    )
    assert initial_limit == {
        "event": "QUALIFIED_RLIMIT_NOFILE_CAPTURED",
        "soft_limit": 1024,
        "hard_limit": 4096,
        "qualified_soft_limit": 1024,
    }
    post_case_limits = [
        row
        for row in emitted
        if row["event"] == "POST_CASE_RESOURCE_BOUNDARY_VALIDATED"
    ]
    assert len(post_case_limits) == len(cases)
    assert all(
        row["rlimit_nofile_soft"] == 1024 and row["rlimit_nofile_hard"] == 4096
        for row in post_case_limits
    )
    end_limit = next(
        row
        for row in emitted
        if row["event"] == "END_OUTCOME_RESOURCE_BOUNDARY_VALIDATED"
    )
    assert end_limit["rlimit_nofile_soft"] == 1024
    assert end_limit["rlimit_nofile_hard"] == 4096
    layout = driver.build_layout(root=tmp_path, role="calibration", cases=cases)
    completion = driver.validate_role_execution_completion(
        layout.execution_completion_path,
        lock_path=lock,
        expected_role="calibration",
        expected_ordered_case_names=cases,
    )
    assert completion["semantic_decision"]["semantic_outcome"] == "NO-GO"
    assert completion["semantic_decision"]["semantic_return_code"] == 3
    for paths in layout.cases.values():
        for directory in (
            paths.target_manifest.parent,
            paths.official_query_manifest.parent,
            paths.queried_seal.parent,
        ):
            assert directory.stat().st_mode & 0o777 == 0o700


def test_qualified_rlimit_nofile_rejects_wrong_soft_and_pair_drift() -> None:
    reference = driver._validate_qualified_rlimit_nofile(
        (1024, 4096), phase="test baseline"
    )
    assert reference == (1024, 4096)
    with pytest.raises(ValueError, match="soft limit differs from qualified value"):
        driver._validate_qualified_rlimit_nofile((2048, 4096), phase="test baseline")
    with pytest.raises(ValueError, match="soft/hard pair changed"):
        driver._validate_qualified_rlimit_nofile(
            (1024, 8192), reference=reference, phase="test post-case boundary"
        )


def test_fd_growth_guard_stops_before_the_next_target_and_second_barrier(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    cases = tuple(f"{index:03d}-object-ep{index:04d}" for index in range(15))
    lock = tmp_path / "calibration-lock.json"
    lock.write_text("lock", encoding="utf-8")
    lock.chmod(0o400)
    (tmp_path / "calibration").mkdir()
    replacement = tmp_path / "replacement-source.json"
    replacement.write_text("source", encoding="utf-8")
    fake_protocol = _FakeProtocol(cases, events)
    post = _fake_post(tmp_path, events, decision_label="GO")
    arguments = driver.DriverArguments(
        role="calibration",
        deployed_code=str(tmp_path / ("code-" + "a" * 40)),
        lock_path=str(lock),
        replacement_source_manifest_path=str(replacement),
        dry_run_barrier_only=False,
        aligned_root=str(tmp_path / "aligned"),
        deform360_repo="d",
        sam2_repository="s",
        sam2_checkpoint="sc",
        cotracker_repo="c",
        cotracker_checkpoint="cc",
    )

    def query_runner(**kwargs: Any) -> None:
        case = Path(kwargs["official_query_manifest_path"]).parent.name
        Path(kwargs["output_archive_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_archive_path"]).write_text(case, encoding="utf-8")
        Path(kwargs["output_seal_path"]).write_text(case, encoding="utf-8")
        events.append(f"query:{case}")

    counts = iter((100, 100, 100 + driver.POST_CASE_FD_GROWTH_LIMIT + 1))
    with pytest.raises(
        ValueError,
        match="post-case file-descriptor growth exceeded the frozen safety limit",
    ):
        driver.execute_outcomes(
            arguments,
            protocol=fake_protocol,
            deployment_verifier=lambda _arguments: events.append("verify"),
            smoke_gsplat_runtime=lambda: {"artifact_sha256": "a" * 64},
            load_post_barrier_api=lambda: post,
            query_runner=query_runner,
            validate_runtime=lambda _arguments: None,
            fd_counter=lambda: next(counts),
            rlimit_nofile_getter=lambda: (1024, 4096),
            formal_paths=False,
        )

    assert f"reconstruct:{cases[0]}" in events
    assert f"query:{cases[0]}" in events
    assert f"reconstruct:{cases[1]}" in events
    assert f"query:{cases[1]}" in events
    assert f"reconstruct:{cases[2]}" not in events
    assert "barrier2" not in events
    assert "score-caps" not in events
    assert "decision" not in events


def test_confirmation_source_verifier_stops_before_smoke_without_go_lock(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    cases = tuple(f"{index:03d}-object-ep{index:04d}" for index in range(6))
    lock = tmp_path / "confirmation-lock.json"
    arguments = driver.DriverArguments(
        role="confirmation",
        deployed_code=str(tmp_path / ("code-" + "a" * 40)),
        lock_path=str(lock),
        replacement_source_manifest_path=None,
        dry_run_barrier_only=True,
    )
    fake_protocol = _FakeProtocol(cases, events, role="confirmation", reject_lock=True)

    def reject_in_source_verifier(_arguments: driver.DriverArguments) -> None:
        events.append("verify")
        fake_protocol.validate_protocol_lock(str(lock))

    with pytest.raises(ValueError, match="inaccessible before GO"):
        driver.execute_outcomes(
            arguments,
            protocol=fake_protocol,
            deployment_verifier=reject_in_source_verifier,
            smoke_gsplat_runtime=lambda: events.append("smoke") or {},
            load_post_barrier_api=lambda: pytest.fail("post-barrier API loaded"),
            formal_paths=False,
        )
    assert events == ["verify", "lock"]
