import hashlib
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPOSITORY_ROOT / "results" / "sota" / "deform_dlo_source_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_frozen_deform_source_result_fails_only_parity_gate() -> None:
    path = RESULT_ROOT / "source_result.json"
    result = json.loads(path.read_text(encoding="utf-8"))

    assert (
        _sha256(path)
        == "9722a7bf4800e18677daa15cb220a39f9a72c73ae2f7ca7d100bb4cba25e8f65"
    )
    assert result["contract"] == "deform-dlo-source-reproduction-result-v1"
    assert result["official_eval_read"] is False
    assert result["selected_checkpoint"]["update"] == 280
    assert result["source_gate"]["model_mean_l1_m"] == pytest.approx(
        0.014032386883627623
    )
    assert result["source_gate"]["persistence_wins"] == 8
    assert result["source_gate"]["parity_passed"] is False
    assert result["source_gate"]["persistence_gate_passed"] is True
    assert result["source_gate"]["passed"] is False
    assert result["advancement_authorized"] is False


def test_frozen_deform_source_artifact_identities() -> None:
    expected = {
        "preflight.json": (
            "9b322b9aeabf3b1f70123e017b256aeee9d9d8d45418d21969be8f23d5bb3798"
        ),
        "source_manifest.json": (
            "570e8f65a4a9c5b5bbfeb923fcb8714885896ec041d862548b91ef6ff1599a8a"
        ),
        "window_schedule.npz": (
            "9d8e565e357f3adeca7392b2cf7b6f8c8ce86381329aa891214ae8265523d7b3"
        ),
    }

    for name, digest in expected.items():
        assert _sha256(RESULT_ROOT / name) == digest

    manifest_text = (RESULT_ROOT / "source_manifest.json").read_text(encoding="utf-8")
    assert "/DLO1/eval/" not in manifest_text
