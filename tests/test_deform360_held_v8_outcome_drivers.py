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
        self.FRESH_REPLACEMENT_CASE_NAME = cases[0]

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
        evidence_path.write_text("evidence", encoding="utf-8")
        decision_path.write_text(decision_label, encoding="utf-8")
        events.append("decision")
        return (
            {"artifact_sha256": "e" * 64},
            {"artifact_sha256": "d" * 64, "decision": decision_label},
        )

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
    )


def test_two_barrier_order_and_no_go_never_promotes_confirmation(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    cases = tuple(f"{index:03d}-object-ep{index:04d}" for index in range(15))
    lock = tmp_path / "calibration-lock.json"
    lock.write_text("lock", encoding="utf-8")
    (tmp_path / "calibration").mkdir()
    replacement = tmp_path / "replacement-source.json"
    replacement.write_text("source", encoding="utf-8")
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
    assert not (tmp_path / "confirmation-lock.json").exists()
    layout = driver.build_layout(root=tmp_path, role="calibration", cases=cases)
    for paths in layout.cases.values():
        for directory in (
            paths.target_manifest.parent,
            paths.official_query_manifest.parent,
            paths.queried_seal.parent,
        ):
            assert directory.stat().st_mode & 0o777 == 0o700


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
