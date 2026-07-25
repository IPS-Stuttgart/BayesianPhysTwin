"""Integrity closure for the source-only Deform360 process-isolation gate."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any


QUALIFICATION_ID = "deform360-original-trainer-process-isolation-v1"
QUALIFICATION_KIND = "Deform360ProcessIsolationQualificationEvidenceV1"
ATTEMPT_KIND = "Deform360ProcessIsolationQualificationAttemptV1"
CHILD_KIND = "Deform360ProcessIsolationCaseChildEvidenceV1"
COMPLETION_KIND = "Deform360ProcessIsolationQualificationIntegrityCompletionV1"
QUALIFICATION_BASE = Path("/mnt/corsair/florianpfaff")
QUALIFICATION_ROOT_PREFIX = "bpt-process-isolation-qualification-"
EVIDENCE_NAME = "process-isolation-qualification.json"
ATTEMPT_NAME = "qualification-attempt.json"
EXPECTED_HOST = "workstation2"
EXPECTED_CASE_COUNT = 4
EXPECTED_FITS_PER_CASE = 81
EXPECTED_ITERATIONS_PER_FIT = 1
EXPECTED_SEED = 0
EXPECTED_PHYSICAL_GPU_INDEX = 1
EXPECTED_SOFT_NOFILE_LIMIT = 1024
EXPECTED_INFORMATION_BOUNDARY = {
    "formal_held_path_supplied": False,
    "target_query_path_received": False,
    "outcome_path_received": False,
    "gate_path_received": False,
    "score_path_received": False,
}
RELATIVE_QUALIFICATION_SOURCE = Path(
    "scripts/development/qualify_deform360_process_isolation.py"
)
RELATIVE_NUMERICAL_SOURCE = Path(
    "src/bayesian_phystwin/deform360_held_outcome_reconstruction.py"
)
RELATIVE_ISOLATION_SOURCE = Path(
    "src/bayesian_phystwin/deform360_case_process_isolation.py"
)
RELATIVE_WORKER_SOURCE = Path(
    "scripts/held/run_deform360_isolated_reconstruction.py"
)
RELATIVE_OUTCOME_DRIVER_SOURCE = Path(
    "src/bayesian_phystwin/deform360_held_v8_outcome_driver.py"
)
RUNTIME_SOURCE_BINDINGS = {
    "qualification_source": RELATIVE_QUALIFICATION_SOURCE,
    "numerical_adapter_source": RELATIVE_NUMERICAL_SOURCE,
    "isolation_source": RELATIVE_ISOLATION_SOURCE,
    "worker_source": RELATIVE_WORKER_SOURCE,
    "outcome_driver_source": RELATIVE_OUTCOME_DRIVER_SOURCE,
}
ROOT_CONSUMPTION_POLICY = {
    "same_root_retry_permitted": False,
    "same_revision_retry_permitted": False,
    "in_place_reuse_permitted": False,
    "later_fix_requires_new_revision_and_root": True,
}
_SHA256_LENGTH = 64


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def artifact_sha256(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_git_oid(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _stable_state(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular(path: str | Path, *, role: str) -> tuple[bytes, os.stat_result]:
    source = _absolute(path)
    before = os.lstat(source)
    _require(
        stat.S_ISREG(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and before.st_nlink == 1
        and source.resolve(strict=True) == source,
        f"{role} is absent, linked, or non-canonical",
    )
    descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    chunks: list[bytes] = []
    try:
        opened = os.fstat(descriptor)
        _require(_stable_state(opened) == _stable_state(before), f"{role} changed")
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(source)
    _require(
        _stable_state(before) == _stable_state(after) == _stable_state(current),
        f"{role} changed while reading",
    )
    return b"".join(chunks), before


def _file_record(path: str | Path, *, role: str) -> dict[str, Any]:
    payload, observed = _read_regular(path, role=role)
    return {
        "path": os.fspath(_absolute(path)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": observed.st_size,
        "mode_octal": f"{stat.S_IMODE(observed.st_mode):04o}",
    }


def _load_signed(path: str | Path, *, role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, _ = _read_regular(path, role=role)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} is not UTF-8 JSON") from error
    _require(isinstance(value, dict), f"{role} is not a JSON object")
    _require(
        value.get("artifact_sha256") == artifact_sha256(value),
        f"{role} signature changed",
    )
    return value, _file_record(path, role=role)


def _record_matches(
    declared: object,
    observed: Mapping[str, Any],
    *,
    role: str,
    require_artifact_sha256: str | None = None,
) -> None:
    _require(isinstance(declared, Mapping), f"{role} binding is absent")
    for key in ("path", "sha256", "size_bytes"):
        _require(declared.get(key) == observed.get(key), f"{role} binding changed")
    if require_artifact_sha256 is not None:
        _require(
            declared.get("artifact_sha256") == require_artifact_sha256,
            f"{role} artifact binding changed",
        )


def _validate_git_binding(value: object, *, role: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{role} is absent")
    _require(
        _valid_git_oid(value.get("head"))
        and _valid_git_oid(value.get("tree"))
        and value.get("clean") is True
        and value.get("ordinary_untracked_file_count") == 0
        and value.get("ignored_untracked_file_count") == 0
        and isinstance(value.get("path"), str),
        f"{role} identity changed",
    )
    return dict(value)


def _validate_runtime_source(
    value: object,
    *,
    code_root: Path,
    relative: Path,
    role: str,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{role} binding is absent")
    expected = (code_root / relative).resolve(strict=True)
    observed = _file_record(expected, role=role)
    _record_matches(value, observed, role=role)
    _require(
        _absolute(str(value.get("path"))) == expected,
        f"{role} escaped the qualified code root",
    )
    return dict(value)


def _validate_child_evidence(
    value: object,
    *,
    expected_case_index: int,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "case-child evidence is absent")
    child = dict(value)
    _require(
        child.get("artifact_sha256") == artifact_sha256(child)
        and child.get("schema_version") == 1
        and child.get("artifact_kind") == CHILD_KIND
        and child.get("qualification_id") == QUALIFICATION_ID
        and child.get("case_index") == expected_case_index
        and child.get("passed") is True,
        "case-child identity or decision changed",
    )
    _require(
        child.get("parameters")
        == {
            "fit_count": EXPECTED_FITS_PER_CASE,
            "iterations_per_fit": EXPECTED_ITERATIONS_PER_FIT,
            "seed": EXPECTED_SEED,
            "trainer_instance_count": 1,
            "trainer_variant": "original-pinned-default",
        },
        "case-child parameters changed",
    )
    _require(
        child.get("information_boundary") == EXPECTED_INFORMATION_BOUNDARY,
        "case-child information boundary changed",
    )
    fits = child.get("fits")
    evaluation = child.get("evaluation")
    _require(
        isinstance(fits, list)
        and len(fits) == EXPECTED_FITS_PER_CASE
        and isinstance(evaluation, Mapping)
        and evaluation.get("passed") is True
        and isinstance(evaluation.get("predicates"), Mapping)
        and evaluation["predicates"]
        and all(value is True for value in evaluation["predicates"].values()),
        "case-child lifecycle evidence failed",
    )
    _require(
        [record.get("fit_index") for record in fits]
        == list(range(EXPECTED_FITS_PER_CASE))
        and all(
            record.get("output_created") is True
            and record.get("output_absent_after_cleanup") is True
            and record.get("generated_outputs_absent_after_cleanup") is True
            and record.get("resource_boundary_stage") == "after_cleanup"
            for record in fits
        ),
        "case-child fit sequence or cleanup changed",
    )
    return child


def _validate_evidence(
    evidence_path: str | Path,
    *,
    expected_source_head: str | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    evidence_source = _absolute(evidence_path)
    root = evidence_source.parent
    _require(
        evidence_source.name == EVIDENCE_NAME
        and root.parent == QUALIFICATION_BASE
        and root.name.startswith(QUALIFICATION_ROOT_PREFIX),
        "process-isolation qualification path changed",
    )
    root_state = os.lstat(root)
    _require(
        stat.S_ISDIR(root_state.st_mode)
        and not stat.S_ISLNK(root_state.st_mode)
        and root.resolve(strict=True) == root,
        "process-isolation qualification root is not canonical",
    )
    evidence, evidence_record = _load_signed(
        evidence_source, role="process-isolation qualification evidence"
    )
    _require(
        evidence.get("schema_version") == 1
        and evidence.get("artifact_kind") == QUALIFICATION_KIND
        and evidence.get("qualification_id") == QUALIFICATION_ID
        and evidence.get("passed") is True
        and evidence.get("information_boundary") == EXPECTED_INFORMATION_BOUNDARY,
        "process-isolation qualification identity or decision changed",
    )
    _require(
        evidence.get("canonical_parameters")
        == {
            "dataset": evidence.get("canonical_parameters", {}).get("dataset"),
            "case_count": EXPECTED_CASE_COUNT,
            "fit_count": EXPECTED_FITS_PER_CASE,
            "iterations": EXPECTED_ITERATIONS_PER_FIT,
            "seed": EXPECTED_SEED,
            "cuda_device": EXPECTED_PHYSICAL_GPU_INDEX,
            "case_timeout_seconds": 28_800,
        }
        and isinstance(evidence["canonical_parameters"]["dataset"], str)
        and evidence["canonical_parameters"]["dataset"],
        "process-isolation canonical parameters changed",
    )
    _require(
        evidence.get("host")
        == {
            "hostname": EXPECTED_HOST,
            "physical_gpu_index": EXPECTED_PHYSICAL_GPU_INDEX,
        },
        "process-isolation host binding changed",
    )
    _require(
        evidence.get("process_boundary")
        == {
            "one_original_trainer_per_child": True,
            "one_official_case_lifecycle_per_child": True,
            "fits_per_case": EXPECTED_FITS_PER_CASE,
            "trainer_configuration_overridden": False,
            "process_exit_reclaims_case_resources": True,
            "parent_process_imports_nerfstudio": False,
        },
        "process-isolation boundary changed",
    )
    runtime = evidence.get("runtime_bindings")
    _require(isinstance(runtime, Mapping), "qualification runtime bindings are absent")
    code = _validate_git_binding(runtime.get("code"), role="qualified code")
    code_root = _absolute(code["path"])
    _require(
        code_root.is_dir()
        and not code_root.is_symlink()
        and code_root.resolve(strict=True) == code_root
        and root.name == f"{QUALIFICATION_ROOT_PREFIX}{code['head']}",
        "qualification root and source revision differ",
    )
    if expected_source_head is not None:
        _require(code["head"] == expected_source_head, "qualified source head changed")
    for name, relative in RUNTIME_SOURCE_BINDINGS.items():
        _validate_runtime_source(
            runtime.get(name),
            code_root=code_root,
            relative=relative,
            role=name.replace("_", " "),
        )
    attempt_path = root / ATTEMPT_NAME
    attempt, attempt_record = _load_signed(
        attempt_path, role="process-isolation qualification attempt"
    )
    _require(
        attempt.get("schema_version") == 1
        and attempt.get("artifact_kind") == ATTEMPT_KIND
        and attempt.get("qualification_id") == QUALIFICATION_ID
        and attempt.get("state") == "canonical-root-consumed-at-creation"
        and attempt.get("output_root") == os.fspath(root)
        and attempt.get("code_revision") == code["head"]
        and attempt.get("physical_gpu_index") == EXPECTED_PHYSICAL_GPU_INDEX
        and attempt.get("canonical_parameters") == evidence["canonical_parameters"]
        and attempt.get("root_consumption_policy") == ROOT_CONSUMPTION_POLICY
        and attempt.get("information_boundary") == EXPECTED_INFORMATION_BOUNDARY,
        "process-isolation attempt marker changed",
    )
    _record_matches(
        evidence.get("attempt_marker"),
        attempt_record,
        role="qualification attempt marker",
    )
    cases = evidence.get("cases")
    _require(
        isinstance(cases, list) and len(cases) == EXPECTED_CASE_COUNT,
        "process-isolation case count changed",
    )
    child_process_ids: list[int] = []
    for case_index, record in enumerate(cases):
        _require(isinstance(record, Mapping), "qualification case record is invalid")
        invocation = record.get("invocation")
        _require(
            record.get("case_index") == case_index
            and isinstance(invocation, Mapping)
            and invocation.get("return_code") == 0
            and invocation.get("timed_out") is False
            and invocation.get("timeout_error") is None
            and record.get("child_contract_valid") is True
            and record.get("child_validation_error") is None
            and record.get("materialized_inputs_stable") is True
            and record.get("source_inputs_stable") is True
            and record.get("generated_dataset_outputs_absent") is True,
            "qualification case invocation or input boundary changed",
        )
        child = _validate_child_evidence(
            record.get("child_evidence"), expected_case_index=case_index
        )
        child_process_ids.append(int(child.get("process_id", -1)))
        child_path = root / f"case-{case_index:03d}" / "case-child-evidence.json"
        child_artifact, child_record = _load_signed(
            child_path, role=f"case-child {case_index} retained evidence"
        )
        _require(
            child_artifact == child,
            "embedded and retained case-child evidence differ",
        )
        _record_matches(
            record.get("child_evidence_file"),
            child_record,
            role=f"case-child {case_index} evidence file",
        )
    _require(
        len(set(child_process_ids)) == EXPECTED_CASE_COUNT
        and min(child_process_ids) > 0,
        "qualification child process identities changed",
    )
    evaluation = evidence.get("evaluation")
    _require(
        isinstance(evaluation, Mapping)
        and evaluation.get("passed") is True
        and isinstance(evaluation.get("predicates"), Mapping)
        and evaluation["predicates"]
        and all(value is True for value in evaluation["predicates"].values())
        and evaluation.get("limits")
        == {
            "child_start_spread": 4,
            "parent_fd_growth": 2,
            "parent_task_growth": 2,
        },
        "process-isolation aggregate gate changed",
    )
    return root, evidence, evidence_record, attempt_record


def _tree_inventory(root: Path, *, require_sealed: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    states: dict[Path, tuple[int, ...]] = {}
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        current_state = os.lstat(current_path)
        _require(
            stat.S_ISDIR(current_state.st_mode)
            and not stat.S_ISLNK(current_state.st_mode),
            f"qualification directory is invalid: {current_path}",
        )
        if require_sealed:
            _require(
                stat.S_IMODE(current_state.st_mode) == 0o500,
                f"qualification directory is not sealed: {current_path}",
            )
        states[current_path] = _stable_state(current_state)
        directories[:] = sorted(directories)
        for name in directories:
            child = current_path / name
            observed = os.lstat(child)
            _require(
                stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode),
                f"qualification child directory is invalid: {child}",
            )
        for name in sorted(files):
            child = current_path / name
            observed = os.lstat(child)
            _require(
                stat.S_ISREG(observed.st_mode)
                and not stat.S_ISLNK(observed.st_mode)
                and observed.st_nlink == 1,
                f"qualification file is linked or special: {child}",
            )
            if require_sealed:
                _require(
                    stat.S_IMODE(observed.st_mode) == 0o400,
                    f"qualification file is not sealed: {child}",
                )
            states[child] = _stable_state(observed)
            record = _file_record(child, role="qualification inventory file")
            relative = child.relative_to(root).as_posix()
            rows.append(
                {
                    "path": relative,
                    "type": "file",
                    "size_bytes": record["size_bytes"],
                    "sha256": record["sha256"],
                }
            )
            metadata_rows.append(
                {
                    "path": relative,
                    "type": "file",
                    "mode_octal": (
                        "0400"
                        if require_sealed
                        else f"{stat.S_IMODE(observed.st_mode):04o}"
                    ),
                    "size_bytes": observed.st_size,
                }
            )
        if current_path != root:
            relative = current_path.relative_to(root).as_posix()
            rows.append({"path": relative, "type": "directory"})
            metadata_rows.append(
                {
                    "path": relative,
                    "type": "directory",
                    "mode_octal": (
                        "0500"
                        if require_sealed
                        else f"{stat.S_IMODE(current_state.st_mode):04o}"
                    ),
                }
            )
    for path, expected in states.items():
        _require(
            _stable_state(os.lstat(path)) == expected,
            f"qualification tree changed while inventorying: {path}",
        )
    rows.sort(key=lambda row: str(row["path"]))
    metadata_rows.sort(key=lambda row: str(row["path"]))
    return {
        "entry_count": len(rows),
        "content_inventory_sha256": hashlib.sha256(
            _canonical_bytes({"rows": rows})
        ).hexdigest(),
        "metadata_inventory_sha256": hashlib.sha256(
            _canonical_bytes({"rows": metadata_rows})
        ).hexdigest(),
    }


def _seal_tree(root: Path) -> None:
    paths: list[Path] = []
    for current, directories, files in os.walk(root, topdown=False, followlinks=False):
        current_path = Path(current)
        paths.extend(current_path / name for name in files)
        paths.extend(current_path / name for name in directories)
    paths.append(root)
    for path in paths:
        observed = os.lstat(path)
        if stat.S_ISREG(observed.st_mode):
            _require(
                not stat.S_ISLNK(observed.st_mode) and observed.st_nlink == 1,
                f"qualification file is linked: {path}",
            )
            os.chmod(path, 0o400, follow_symlinks=False)
        else:
            _require(
                stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode),
                f"qualification tree contains a special entry: {path}",
            )
            os.chmod(path, 0o500, follow_symlinks=False)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o400, follow_symlinks=False)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def seal_process_isolation_qualification(
    root_path: str | Path,
    completion_path: str | Path,
    *,
    sealer_source_path: str | Path,
) -> dict[str, Any]:
    """Seal one consumed qualification root and publish its integrity closure."""

    root = _absolute(root_path)
    completion = _absolute(completion_path)
    evidence_path = root / EVIDENCE_NAME
    _require(
        completion == Path(f"{root}-integrity-completion.json"),
        "qualification completion path changed",
    )
    _require(
        not os.path.lexists(completion),
        "qualification completion already exists",
    )
    validated_root, evidence, evidence_record, attempt_record = _validate_evidence(
        evidence_path
    )
    _require(validated_root == root, "qualification root and evidence differ")
    before = _tree_inventory(root, require_sealed=False)
    _seal_tree(root)
    after = _tree_inventory(root, require_sealed=True)
    _require(
        before["entry_count"] == after["entry_count"]
        and before["content_inventory_sha256"]
        == after["content_inventory_sha256"],
        "qualification content changed while sealing",
    )
    runtime = evidence["runtime_bindings"]
    code = runtime["code"]
    sealer_source = _file_record(
        sealer_source_path, role="process-isolation qualification sealer source"
    )
    completion_value: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": COMPLETION_KIND,
        "qualification_id": QUALIFICATION_ID,
        "status": "qualification-integrity-complete",
        "passed": True,
        "admission_eligible": True,
        "qualification_root": os.fspath(root),
        "qualification_root_mode_octal": "0500",
        "qualification_tree_fully_nonwritable": True,
        "root_consumption_policy": dict(ROOT_CONSUMPTION_POLICY),
        "qualification_attempt": {
            **{key: attempt_record[key] for key in ("path", "sha256", "size_bytes")},
            "artifact_sha256": _load_signed(
                root / ATTEMPT_NAME,
                role="sealed process-isolation qualification attempt",
            )[0]["artifact_sha256"],
        },
        "qualification_evidence": {
            **{key: evidence_record[key] for key in ("path", "sha256", "size_bytes")},
            "artifact_sha256": evidence["artifact_sha256"],
        },
        "sealed_content_inventory": after,
        "source_code": {
            "path": code["path"],
            "head": code["head"],
            "tree": code["tree"],
        },
        "executed_integrity_sealer_source": sealer_source,
        "information_boundary": {
            **EXPECTED_INFORMATION_BOUNDARY,
            "scientific_method_selected_from_qualification": False,
            "formal_target_query_prediction_or_score_deserialized": False,
        },
    }
    completion_value["artifact_sha256"] = artifact_sha256(completion_value)
    _write_new_json(completion, completion_value)
    return validate_process_isolation_qualification_lineage(
        evidence_path=evidence_path,
        completion_path=completion,
        expected_source_head=str(code["head"]),
        verify_content_inventory=True,
    )


def validate_process_isolation_qualification_lineage(
    *,
    evidence_path: str | Path,
    completion_path: str | Path,
    expected_source_head: str | None = None,
    verify_content_inventory: bool = False,
) -> dict[str, Any]:
    """Validate one sealed, admitted process-isolation qualification."""

    root, evidence, evidence_record, attempt_record = _validate_evidence(
        evidence_path,
        expected_source_head=expected_source_head,
    )
    root_state = os.lstat(root)
    _require(
        stat.S_IMODE(root_state.st_mode) == 0o500,
        "process-isolation qualification root is not sealed",
    )
    completion = _absolute(completion_path)
    _require(
        completion == Path(f"{root}-integrity-completion.json"),
        "process-isolation completion path changed",
    )
    completion_value, completion_record = _load_signed(
        completion, role="process-isolation qualification completion"
    )
    runtime = evidence["runtime_bindings"]
    code = runtime["code"]
    _require(
        completion_value.get("schema_version") == 1
        and completion_value.get("artifact_kind") == COMPLETION_KIND
        and completion_value.get("qualification_id") == QUALIFICATION_ID
        and completion_value.get("status") == "qualification-integrity-complete"
        and completion_value.get("passed") is True
        and completion_value.get("admission_eligible") is True
        and completion_value.get("qualification_root") == os.fspath(root)
        and completion_value.get("qualification_root_mode_octal") == "0500"
        and completion_value.get("qualification_tree_fully_nonwritable") is True
        and completion_value.get("root_consumption_policy")
        == ROOT_CONSUMPTION_POLICY
        and completion_value.get("source_code")
        == {
            "path": code["path"],
            "head": code["head"],
            "tree": code["tree"],
        }
        and completion_value.get("information_boundary")
        == {
            **EXPECTED_INFORMATION_BOUNDARY,
            "scientific_method_selected_from_qualification": False,
            "formal_target_query_prediction_or_score_deserialized": False,
        },
        "process-isolation completion identity changed",
    )
    _record_matches(
        completion_value.get("qualification_attempt"),
        attempt_record,
        role="completion qualification attempt",
        require_artifact_sha256=_load_signed(
            root / ATTEMPT_NAME,
            role="sealed process-isolation qualification attempt",
        )[0]["artifact_sha256"],
    )
    _record_matches(
        completion_value.get("qualification_evidence"),
        evidence_record,
        role="completion qualification evidence",
        require_artifact_sha256=evidence["artifact_sha256"],
    )
    observed_inventory = _tree_inventory(root, require_sealed=True)
    declared_inventory = completion_value.get("sealed_content_inventory")
    _require(
        isinstance(declared_inventory, Mapping)
        and declared_inventory.get("entry_count")
        == observed_inventory["entry_count"]
        and declared_inventory.get("metadata_inventory_sha256")
        == observed_inventory["metadata_inventory_sha256"],
        "process-isolation sealed metadata inventory changed",
    )
    if verify_content_inventory:
        _require(
            declared_inventory.get("content_inventory_sha256")
            == observed_inventory["content_inventory_sha256"],
            "process-isolation sealed content inventory changed",
        )
    sealer_source = completion_value.get("executed_integrity_sealer_source")
    _require(
        isinstance(sealer_source, Mapping)
        and _valid_sha256(sealer_source.get("sha256"))
        and isinstance(sealer_source.get("path"), str),
        "process-isolation sealer source binding is absent",
    )
    observed_sealer = _file_record(
        str(sealer_source["path"]),
        role="process-isolation qualification sealer source",
    )
    _record_matches(
        sealer_source,
        observed_sealer,
        role="process-isolation qualification sealer source",
    )
    return {
        "process_isolation_qualification_attempt": {
            **{key: attempt_record[key] for key in ("path", "sha256", "size_bytes")},
            "artifact_sha256": _load_signed(
                root / ATTEMPT_NAME,
                role="sealed process-isolation qualification attempt",
            )[0]["artifact_sha256"],
        },
        "process_isolation_qualification_evidence": {
            **{key: evidence_record[key] for key in ("path", "sha256", "size_bytes")},
            "artifact_sha256": evidence["artifact_sha256"],
        },
        "process_isolation_qualification_integrity_completion": {
            **{
                key: completion_record[key]
                for key in ("path", "sha256", "size_bytes")
            },
            "artifact_sha256": completion_value["artifact_sha256"],
        },
        "process_isolation_qualification_integrity": {
            "qualification_id": QUALIFICATION_ID,
            "source_head": code["head"],
            "source_tree": code["tree"],
            "terminal_outcome": "qualified",
            "admission_eligible": True,
            "inventory_sha256": observed_inventory["content_inventory_sha256"],
            "metadata_inventory_sha256": observed_inventory[
                "metadata_inventory_sha256"
            ],
            "entry_count": observed_inventory["entry_count"],
            "qualification_source_sha256": runtime["qualification_source"]["sha256"],
            "numerical_adapter_source_sha256": runtime[
                "numerical_adapter_source"
            ]["sha256"],
            "isolation_source_sha256": runtime["isolation_source"]["sha256"],
            "worker_source_sha256": runtime["worker_source"]["sha256"],
            "outcome_driver_source_sha256": runtime["outcome_driver_source"][
                "sha256"
            ],
            "sealer_source_sha256": sealer_source["sha256"],
        },
    }


__all__ = [
    "ATTEMPT_KIND",
    "COMPLETION_KIND",
    "EVIDENCE_NAME",
    "QUALIFICATION_ID",
    "QUALIFICATION_KIND",
    "ROOT_CONSUMPTION_POLICY",
    "artifact_sha256",
    "seal_process_isolation_qualification",
    "validate_process_isolation_qualification_lineage",
]
