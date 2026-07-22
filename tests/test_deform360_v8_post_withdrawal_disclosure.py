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
    report = module.expected_unsigned_report(bindings)

    assert report["protocol_id"].endswith("v8")
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
    reuse = report["v8_reuse_boundary"]
    assert reuse["v7_withdrawal_report_used_only_as_immutable_lineage"] is True
    assert reuse["all_v8_predictions_targets_queries_and_scores_must_be_fresh"] is True
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
    expected = {
        "v7_outcome_withdrawal_report": _sealed(root / "withdrawal.json", b"w\n"),
        "retired_case_official_target": _sealed(root / "target.npz", b"t\n"),
        "retired_case_online_prediction": _sealed(root / "online.npz", b"p\n"),
        "retired_case_online_prediction_seal": _sealed(root / "seal.json", b"s\n"),
    }
    output = held_v8 / "post-withdrawal-development-use-disclosure.json"
    monkeypatch.setattr(module, "_EXPECTED_FILES", expected)
    monkeypatch.setattr(module, "_V8_ROOT", held_v8)
    monkeypatch.setattr(module, "_OUTPUT", output)

    try:
        signed, payload = module.build_report()
        assert (
            signed["artifact_sha256"]
            == module._artifact(
                module.expected_unsigned_report(signed["disclosed_v7_files"])
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
