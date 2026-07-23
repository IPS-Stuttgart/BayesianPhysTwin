from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
OPERATOR = ROOT / "scripts" / "held" / "seal_deform360_v8_post_withdrawal_disclosure.py"


def _module():
    spec = importlib.util.spec_from_file_location("v8_disclosure", OPERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sealed(path: Path, payload: bytes) -> tuple[Path, int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o400)
    return path, len(payload), hashlib.sha256(payload).hexdigest()


def _git(code: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(code), *arguments],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _seal_tree(root: Path) -> None:
    paths: list[Path] = []
    for current, directories, files in os.walk(root, topdown=False):
        current_path = Path(current)
        paths.extend(current_path / name for name in files)
        paths.extend(current_path / name for name in directories)
    paths.append(root)
    for path in paths:
        path.chmod(0o500 if path.is_dir() else 0o400)


def _sealed_test_repository(
    root: Path, module: object
) -> tuple[Path, dict[str, object]]:
    stage = root / "code-stage"
    stage.mkdir()
    _git(stage, "init", "--quiet")
    _git(stage, "config", "user.email", "held-test@example.invalid")
    _git(stage, "config", "user.name", "Held Test")
    (stage / ".gitignore").write_text("ignored-runtime.py\n", encoding="utf-8")
    (stage / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(stage, "add", ".gitignore", "tracked.py")
    _git(stage, "commit", "--quiet", "-m", "test fixture")
    _git(stage, "checkout", "--quiet", "--detach", "HEAD")
    head = _git(stage, "rev-parse", "HEAD").decode("ascii").strip()
    deployed_code = root / f"code-{head}"
    stage.rename(deployed_code)
    _seal_tree(deployed_code)
    binding = module._attempt3_repository_binding(deployed_code)
    return deployed_code, binding


def test_disclosure_is_conservative_and_bars_all_v7_execution_reuse() -> None:
    module = _module()
    bindings = {
        name: {
            "path": str(path),
            "sha256": sha256,
            "size_bytes": size,
            "mode_octal": "0400",
        }
        for name, (path, size, sha256) in module._EXPECTED_FILES.items()
    }
    archive_integrity = dict(module._ATTEMPT3_ARCHIVE_INTEGRITY)
    qualification_files = {
        "resource_lifecycle_qualification_attempt": {
            "path": "/qualification/qualification-attempt.json",
            "sha256": "0" * 64,
            "size_bytes": 1,
            "mode_octal": "0400",
            "artifact_sha256": "1" * 64,
        },
        "resource_lifecycle_qualification_evidence": {
            "path": "/qualification/resource-lifecycle-qualification.json",
            "sha256": "a" * 64,
            "size_bytes": 1,
            "mode_octal": "0400",
            "artifact_sha256": "2" * 64,
        },
        "resource_lifecycle_qualification_repeat_manifest": {
            "path": "/qualification/equivalence/repeat-manifest.json",
            "sha256": "3" * 64,
            "size_bytes": 1,
            "mode_octal": "0400",
            "artifact_sha256": "4" * 64,
        },
        "resource_lifecycle_qualification_equivalence_result": {
            "path": "/qualification/equivalence/analysis-result.json",
            "sha256": "5" * 64,
            "size_bytes": 1,
            "mode_octal": "0400",
            "artifact_sha256": "6" * 64,
        },
        "resource_lifecycle_qualification_integrity_completion": {
            "path": "/qualification-integrity-completion.json",
            "sha256": "b" * 64,
            "size_bytes": 2,
            "mode_octal": "0400",
            "artifact_sha256": "7" * 64,
        },
    }
    qualification_integrity = {
        "root": "/qualification",
        "root_mode_octal": "0500",
        "fully_nonwritable": True,
        "entry_count": 5,
        "inventory_sha256": "c" * 64,
        "source_head": "d" * 40,
        "source_tree": "e" * 40,
        "terminal_outcome": "qualified",
        "admission_eligible": True,
        "generator_profile": "same-as-analyzer",
        "physical_gpu_index": 1,
        "equivalence_acceptance_basis": "secondary-distributional-envelope",
        "analyzer_source_sha256": module._QUALIFICATION_ANALYZER_SHA256,
    }
    report = module.expected_unsigned_report(
        bindings,
        archive_integrity,
        attempt4_launcher={"path": str(module._ATTEMPT4_LAUNCHER)},
        attempt4_execution={"score_evidence_count": 0},
        attempt4_information={"score_created_or_read": False},
        qualification_files=qualification_files,
        qualification_integrity=qualification_integrity,
    )

    assert report["protocol_id"] == "deform360-held-online-belief-v8.1"
    assert set(report["disclosed_v7_files"]) == module._V7_FILE_NAMES
    assert set(report["disclosed_v8_attempt3_files"]) == (module._ATTEMPT3_FILE_NAMES)
    assert set(report["disclosed_v8_attempt4_files"]) == (module._ATTEMPT4_FILE_NAMES)
    assert report["v8_attempt3_archive_integrity"] == archive_integrity
    assert report["v8_attempt4_archive_integrity"] == module._ATTEMPT4_ARCHIVE_INTEGRITY
    assert report["v8_attempt4_execution_boundary"]["calibration_result"] == (
        "NO_CALIBRATION_RESULT"
    )
    assert report["resource_lifecycle_qualification_files"] == qualification_files
    assert report["resource_lifecycle_qualification_integrity"] == (
        qualification_integrity
    )
    assert report["v8_attempt3_revision_basis"] == {
        "official_x0_geometry_used_to_diagnose_exclusion_liveness": True,
        "future_target_coordinates_masks_or_scores_used_for_revision": False,
        "queried_prediction_score_or_gate_existed": False,
        "revision": (
            "replace exact-one-per-center matching with the inclusive 15 mm "
            "x0-only radius union"
        ),
    }
    assert report["retirement"] == {
        "exact_episode": "002-rope-silk-ep0003",
        "replacement_episode": "072-cotton-clohesline-ep0003",
        "replacement_search_excluded_entire_002_rope_silk_object": True,
        "reason": (
            "the exact held-v7 episode was exposed after formal withdrawal; "
            "the replacement was selected outside that object's episodes"
        ),
    }
    development = report["post_withdrawal_development"]
    assert development["future_coordinates_or_masks_may_have_been_read"] is True
    assert development["derived_metrics_may_have_been_computed"] is True
    assert (
        development[
            "field_hypothesis_was_subsequently_reselected_on_independent_open27"
        ]
        is True
    )
    reuse = report["v8_1_reuse_boundary"]
    assert reuse["v7_withdrawal_report_used_only_as_immutable_lineage"] is True
    assert (
        reuse["all_v8_1_attempt5_predictions_targets_queries_and_scores_fresh"] is True
    )
    assert reuse["full_15_case_fresh_rerun_required"] is True
    assert all(
        value is False for key, value in reuse.items() if key.endswith("_reused")
    )


def test_operator_hashes_only_exact_sealed_files_and_writes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    temporary = tempfile.TemporaryDirectory(
        prefix="bpt-v8-disclosure-test-", dir="/tmp"
    )
    root = Path(temporary.name)
    held_v8 = root / "held-v8"
    held_v8.mkdir()
    attempt3_archive = root / "held-v8-attempt-3-withdrawn-postbarrier"
    attempt3_archive.mkdir()
    deployed_code, deployed_binding = _sealed_test_repository(attempt3_archive, module)
    evidence = attempt3_archive / "sealed-evidence"
    evidence.mkdir()
    evidence_payload = b"attempt-3-sealed-payload\n"
    evidence_file = evidence / "payload.bin"
    evidence_file.write_bytes(evidence_payload)
    evidence_file.chmod(0o400)
    evidence.chmod(0o500)
    expected = {
        "v7_outcome_withdrawal_report": _sealed(root / "withdrawal.json", b"w\n"),
        "retired_case_official_target": _sealed(root / "target.npz", b"t\n"),
        "retired_case_online_prediction": _sealed(root / "online.npz", b"p\n"),
        "retired_case_online_prediction_seal": _sealed(root / "seal.json", b"s\n"),
        "v8_attempt3_withdrawal_report": _sealed(
            attempt3_archive / "execution-withdrawal-postbarrier-attempt3.json",
            b"r\n",
        ),
        "v8_attempt3_withdrawal_pointer": _sealed(root / "pointer.json", b"q\n"),
        "v8_attempt3_withdrawal_integrity_completion": _sealed(
            root / "completion.json", b"c\n"
        ),
        "v8_attempt4_withdrawal_report": _sealed(
            root / "attempt4-report.json", b"r4\n"
        ),
        "v8_attempt4_withdrawal_pointer": _sealed(
            root / "attempt4-pointer.json", b"p4\n"
        ),
        "v8_attempt4_withdrawal_integrity_completion": _sealed(
            root / "attempt4-completion.json", b"c4\n"
        ),
    }
    attempt3_archive.chmod(0o500)
    inventory_rows = [
        {
            "path": "sealed-evidence",
            "type": "directory",
            "mode_octal": "0500",
        },
        {
            "path": "sealed-evidence/payload.bin",
            "type": "file",
            "mode_octal": "0400",
            "size_bytes": len(evidence_payload),
            "sha256": hashlib.sha256(evidence_payload).hexdigest(),
        },
    ]
    inventory_sha256 = hashlib.sha256(
        json.dumps(
            {"rows": inventory_rows},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    archive_integrity = {
        "path": str(attempt3_archive),
        "root_mode_octal": "0500",
        "fully_nonwritable": True,
        "postseal_noncode_inventory_sha256": inventory_sha256,
        "postseal_noncode_entry_count": len(inventory_rows),
    }
    output = held_v8 / "post-withdrawal-development-use-disclosure.json"
    monkeypatch.setattr(module, "_EXPECTED_FILES", expected)
    monkeypatch.setattr(
        module,
        "_V7_FILE_NAMES",
        frozenset(
            {
                "v7_outcome_withdrawal_report",
                "retired_case_official_target",
                "retired_case_online_prediction",
                "retired_case_online_prediction_seal",
            }
        ),
    )
    monkeypatch.setattr(
        module,
        "_ATTEMPT3_FILE_NAMES",
        frozenset(
            {
                "v8_attempt3_withdrawal_report",
                "v8_attempt3_withdrawal_pointer",
                "v8_attempt3_withdrawal_integrity_completion",
            }
        ),
    )
    monkeypatch.setattr(
        module,
        "_ATTEMPT4_FILE_NAMES",
        frozenset(
            {
                "v8_attempt4_withdrawal_report",
                "v8_attempt4_withdrawal_pointer",
                "v8_attempt4_withdrawal_integrity_completion",
            }
        ),
    )
    monkeypatch.setattr(module, "_ATTEMPT3_ARCHIVE", attempt3_archive)
    monkeypatch.setattr(module, "_ATTEMPT3_ARCHIVE_INTEGRITY", archive_integrity)
    monkeypatch.setattr(module, "_ATTEMPT3_DEPLOYED_CODE_BINDING", deployed_binding)
    monkeypatch.setattr(module, "_V8_ROOT", held_v8)
    monkeypatch.setattr(module, "_OUTPUT", output)
    attempt4_launcher = {"path": "/attempt4-launcher"}
    attempt4_execution = {"score_evidence_count": 0}
    attempt4_information = {"score_created_or_read": False}
    monkeypatch.setattr(
        module,
        "_bind_attempt4_lineage",
        lambda _bindings: (
            attempt4_launcher,
            attempt4_execution,
            attempt4_information,
        ),
    )
    qualification_evidence = Path(
        _sealed(root / "qualification/resource-lifecycle-qualification.json", b"qe\n")[
            0
        ]
    )
    qualification_completion = Path(
        _sealed(root / "qualification-integrity-completion.json", b"qc\n")[0]
    )
    qualification_files = {
        "resource_lifecycle_qualification_attempt": {
            "path": str(qualification_evidence.parent / "qualification-attempt.json"),
            "sha256": "0" * 64,
            "size_bytes": 3,
            "mode_octal": "0400",
            "artifact_sha256": "1" * 64,
        },
        "resource_lifecycle_qualification_evidence": {
            "path": str(qualification_evidence),
            "sha256": hashlib.sha256(b"qe\n").hexdigest(),
            "size_bytes": 3,
            "mode_octal": "0400",
            "artifact_sha256": "2" * 64,
        },
        "resource_lifecycle_qualification_repeat_manifest": {
            "path": str(
                qualification_evidence.parent / "equivalence/repeat-manifest.json"
            ),
            "sha256": "3" * 64,
            "size_bytes": 3,
            "mode_octal": "0400",
            "artifact_sha256": "4" * 64,
        },
        "resource_lifecycle_qualification_equivalence_result": {
            "path": str(
                qualification_evidence.parent / "equivalence/analysis-result.json"
            ),
            "sha256": "5" * 64,
            "size_bytes": 3,
            "mode_octal": "0400",
            "artifact_sha256": "6" * 64,
        },
        "resource_lifecycle_qualification_integrity_completion": {
            "path": str(qualification_completion),
            "sha256": hashlib.sha256(b"qc\n").hexdigest(),
            "size_bytes": 3,
            "mode_octal": "0400",
            "artifact_sha256": "7" * 64,
        },
    }
    qualification_integrity = {
        "root": str(qualification_evidence.parent),
        "root_mode_octal": "0500",
        "fully_nonwritable": True,
        "entry_count": 5,
        "inventory_sha256": "f" * 64,
        "source_head": "a" * 40,
        "source_tree": "b" * 40,
        "terminal_outcome": "qualified",
        "admission_eligible": True,
        "generator_profile": "same-as-analyzer",
        "physical_gpu_index": 1,
        "equivalence_acceptance_basis": "secondary-distributional-envelope",
        "analyzer_source_sha256": module._QUALIFICATION_ANALYZER_SHA256,
    }
    monkeypatch.setattr(
        module,
        "_bind_resource_qualification",
        lambda *_args, **_kwargs: (qualification_files, qualification_integrity),
    )

    def build():
        return module.build_report(
            qualification_evidence=qualification_evidence,
            qualification_completion=qualification_completion,
            qualification_evidence_sha256=qualification_files[
                "resource_lifecycle_qualification_evidence"
            ]["sha256"],
            qualification_completion_sha256=qualification_files[
                "resource_lifecycle_qualification_integrity_completion"
            ]["sha256"],
        )

    try:
        signed, payload = build()
        bindings = {
            **signed["disclosed_v7_files"],
            **signed["disclosed_v8_attempt3_files"],
            **signed["disclosed_v8_attempt4_files"],
        }
        assert (
            signed["artifact_sha256"]
            == module._artifact(
                module.expected_unsigned_report(
                    bindings,
                    signed["v8_attempt3_archive_integrity"],
                    attempt4_launcher=attempt4_launcher,
                    attempt4_execution=attempt4_execution,
                    attempt4_information=attempt4_information,
                    qualification_files=qualification_files,
                    qualification_integrity=qualification_integrity,
                )
            )[0]["artifact_sha256"]
        )

        wrong_binding = dict(deployed_binding)
        wrong_binding["git_tree_manifest_sha256"] = "0" * 64
        monkeypatch.setattr(module, "_ATTEMPT3_DEPLOYED_CODE_BINDING", wrong_binding)
        with pytest.raises(RuntimeError, match="repository binding changed"):
            build()
        monkeypatch.setattr(module, "_ATTEMPT3_DEPLOYED_CODE_BINDING", deployed_binding)

        tracked = deployed_code / "tracked.py"
        tracked.chmod(0o600)
        tracked.write_text("VALUE = 2\n", encoding="utf-8")
        tracked.chmod(0o400)
        with pytest.raises(RuntimeError, match="worktree content changed"):
            build()
        tracked.chmod(0o600)
        tracked.write_text("VALUE = 1\n", encoding="utf-8")
        tracked.chmod(0o400)

        deployed_code.chmod(0o700)
        ignored = deployed_code / "ignored-runtime.py"
        ignored.write_text("raise RuntimeError('must not execute')\n", encoding="utf-8")
        ignored.chmod(0o400)
        deployed_code.chmod(0o500)
        with pytest.raises(RuntimeError, match="untracked or ignored"):
            build()
        deployed_code.chmod(0o700)
        ignored.unlink()
        deployed_code.chmod(0o500)

        attempt3_archive.chmod(0o700)
        unexpected = attempt3_archive / "unexpected.bin"
        unexpected.write_bytes(b"unexpected\n")
        unexpected.chmod(0o400)
        attempt3_archive.chmod(0o500)
        with pytest.raises(RuntimeError, match="archive inventory changed"):
            build()
        attempt3_archive.chmod(0o700)
        unexpected.unlink()
        attempt3_archive.chmod(0o500)

        evidence.chmod(0o700)
        evidence_file.unlink()
        evidence.chmod(0o500)
        with pytest.raises(RuntimeError, match="archive inventory changed"):
            build()
        evidence.chmod(0o700)
        evidence_file.write_bytes(evidence_payload)
        evidence_file.chmod(0o400)
        evidence.chmod(0o500)

        evidence_file.chmod(0o600)
        evidence_file.write_bytes(b"changed\n")
        evidence_file.chmod(0o400)
        with pytest.raises(RuntimeError, match="archive inventory changed"):
            build()
        evidence_file.chmod(0o600)
        evidence_file.write_bytes(evidence_payload)
        evidence_file.chmod(0o400)

        module._write_once(output, payload)
        assert stat.S_IMODE(output.stat().st_mode) == 0o400
        assert output.read_bytes() == payload
        module._write_once(output, payload)
        alias = root / "disclosure-alias.json"
        os.link(output, alias)
        with pytest.raises(RuntimeError, match="sealed regular file"):
            module._write_once(output, payload)
        alias.unlink()
        output.chmod(0o600)
        output.write_bytes(payload + b"tamper")
        output.chmod(0o400)
        with pytest.raises(RuntimeError, match="payload changed"):
            module._write_once(output, payload)
    finally:
        for path in root.rglob("*"):
            if path.is_file():
                os.chmod(path, 0o600)
        for path in sorted(
            (candidate for candidate in root.rglob("*") if candidate.is_dir()),
            key=lambda candidate: len(candidate.parts),
            reverse=True,
        ):
            os.chmod(path, 0o700)
        temporary.cleanup()


def test_disclosure_source_cannot_deserialize_protected_payloads() -> None:
    source = OPERATOR.read_text(encoding="utf-8")
    for forbidden in (
        "np.load(",
        "numpy.load",
        "pickle.load",
        "torch.load",
        "h5py",
        "cv2",
        "PlyData.read",
    ):
        assert forbidden not in source
