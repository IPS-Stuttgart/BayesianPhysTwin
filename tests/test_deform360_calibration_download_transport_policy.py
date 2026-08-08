"""Static transport-policy contracts for frozen Deform360 acquisition."""

from __future__ import annotations

from pathlib import Path

from scripts.science.deform360_calibration_source import cli, download

ROOT = Path(__file__).resolve().parents[1]
REUSABLE = (
    ROOT / ".github/workflows/deform360-official-hub-calibration-source-reusable.yml"
)
DISPATCHER = (
    ROOT / ".github/workflows/dispatch-deform360-calibration-source-pr-target.yml"
)
DIRECT_SCRIPT = ROOT / "scripts/ci/run_deform360_calibration_source_direct.sh"


def test_download_lane_disables_xet_and_accepts_only_optional_hf_token() -> None:
    reusable = REUSABLE.read_text(encoding="utf-8")
    dispatcher = DISPATCHER.read_text(encoding="utf-8")

    assert "hf_token:" in reusable
    assert "required: false" in reusable
    assert 'HF_HUB_DISABLE_XET: "1"' in reusable
    assert "HF_TOKEN: ${{ secrets.hf_token }}" in reusable
    assert "hf_token: ${{ secrets.HF_TOKEN }}" in dispatcher
    assert "secrets: inherit" not in dispatcher


def test_download_lane_never_prints_or_persists_token_value() -> None:
    reusable = REUSABLE.read_text(encoding="utf-8")
    direct = DIRECT_SCRIPT.read_text(encoding="utf-8")

    forbidden = (
        'echo "${HF_TOKEN}',
        "echo ${HF_TOKEN}",
        'printf "%s" "${HF_TOKEN}',
        "set -x",
    )
    assert all(pattern not in reusable for pattern in forbidden)
    assert all(pattern not in direct for pattern in forbidden)
    assert "hf_token_configured" not in reusable


def test_download_concurrency_is_hard_capped_and_cli_defaults_to_cap() -> None:
    assert download.DOWNLOAD_MAX_WORKERS == 2
    parser = cli._parser()
    args = parser.parse_args(
        [
            "download",
            "--protocol",
            "protocol.json",
            "--selection-lock",
            "selection.json",
            "--visual-provider-lock",
            "provider.json",
            "--plan",
            "plan.json",
            "--data-root",
            "data",
            "--output",
            "download.json",
        ]
    )
    assert args.workers == download.DOWNLOAD_MAX_WORKERS


def test_scientific_and_security_boundaries_remain_unchanged() -> None:
    reusable = REUSABLE.read_text(encoding="utf-8")
    dispatcher = DISPATCHER.read_text(encoding="utf-8")
    direct = DIRECT_SCRIPT.read_text(encoding="utf-8")

    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in reusable
    assert "persist-credentials: false" in reusable
    assert "confirmation_payloads_opened=false" in direct
    assert "head.repo.full_name == github.repository" in dispatcher
    assert "head.ref == 'agent/calibration-dispatch-trigger-v1'" in dispatcher
    assert "changed_files == 1" in dispatcher
    assert "additions == 1" in dispatcher
    assert "deletions == 0" in dispatcher
