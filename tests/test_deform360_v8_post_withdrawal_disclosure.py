from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import stat
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
    report = module.expected_unsigned_report(bindings, archive_integrity)

    assert report["protocol_id"] == "deform360-held-online-belief-v8.1"
    assert set(report["disclosed_v7_files"]) == module._V7_FILE_NAMES
    assert set(report["disclosed_v8_attempt3_files"]) == (
        module._ATTEMPT3_FILE_NAMES
    )
    assert report["v8_attempt3_archive_integrity"] == archive_integrity
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
    assert reuse[
        "all_v8_1_attempt4_predictions_targets_queries_and_scores_fresh"
    ] is True
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
    }
    attempt3_archive.chmod(0o500)
    archive_integrity = {
        "path": str(attempt3_archive),
        "root_mode_octal": "0500",
        "fully_nonwritable": True,
        "postseal_noncode_inventory_sha256": "d" * 64,
        "postseal_noncode_entry_count": 1,
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
    monkeypatch.setattr(module, "_ATTEMPT3_ARCHIVE", attempt3_archive)
    monkeypatch.setattr(
        module, "_ATTEMPT3_ARCHIVE_INTEGRITY", archive_integrity
    )
    monkeypatch.setattr(module, "_V8_ROOT", held_v8)
    monkeypatch.setattr(module, "_OUTPUT", output)

    try:
        signed, payload = module.build_report()
        bindings = {
            **signed["disclosed_v7_files"],
            **signed["disclosed_v8_attempt3_files"],
        }
        assert (
            signed["artifact_sha256"]
            == module._artifact(
                module.expected_unsigned_report(
                    bindings, signed["v8_attempt3_archive_integrity"]
                )
            )[0]["artifact_sha256"]
        )
        module._write_once(output, payload)
        assert stat.S_IMODE(output.stat().st_mode) == 0o400
        assert output.read_bytes() == payload
        module._write_once(output, payload)
        output.chmod(0o600)
        output.write_bytes(payload + b"tamper")
        output.chmod(0o400)
        with pytest.raises(RuntimeError, match="payload changed"):
            module._write_once(output, payload)
    finally:
        for path in root.rglob("*"):
            if path.is_file():
                os.chmod(path, 0o600)
        attempt3_archive.chmod(0o700)
        temporary.cleanup()


def test_disclosure_source_cannot_deserialize_protected_payloads() -> None:
    source = OPERATOR.read_text(encoding="utf-8")
    for forbidden in (
        "json.loads(",
        "np.load(",
        "numpy",
        "h5py",
        "cv2",
        "read_text(",
        "read_bytes(",
    ):
        assert forbidden not in source
