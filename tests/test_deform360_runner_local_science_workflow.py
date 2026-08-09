"""Static safety contracts for runner-local Deform360 scientific execution."""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW = Path(".github/workflows/deform360-runner-local-science.yml")
CONTRACT_WORKFLOW = Path(".github/workflows/deform360-runner-local-contracts.yml")
RUNTIME_LOCK = Path("requirements/locks/deform360-runner-local-science-py312.txt")
GUIDE = Path("docs/deform360_runner_local_bootstrap.md")
OFFICIAL_ROOT = "/mnt/lexar4tb/datasets/deform360/data-7fea8e2"
ADAPTIVE_ROOT = (
    "/mnt/lexar4tb/datasets/deform360/"
    "adaptive-confirmation-download-5a9c56d593462486bdd0953dcaf6f9c643bf8370"
)


def _block(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[begin:finish]


def test_workflow_keeps_pull_request_validation_hosted_and_data_free() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    contracts = _block(text, "  contracts:", "  science:")
    science = text[text.index("  science:") :]
    reusable = CONTRACT_WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" in text
    assert "workflow_dispatch:" in text
    assert "pull_request_target:" not in text
    assert "permissions:\n  contents: read" in text
    assert "github.workflow" in text
    assert "github.ref" in text
    assert "cancel-in-progress: ${{ github.event_name == 'pull_request' }}" in text
    assert "uses: ./.github/workflows/deform360-runner-local-contracts.yml" in contracts
    assert (
        "source_sha: ${{ github.event.pull_request.head.sha || github.sha }}"
        in contracts
    )
    assert "workflow_call:" in reusable
    assert "source_sha:" in reusable
    assert "runs-on: ubuntu-latest" in reusable
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in science
    assert "github.event_name == 'workflow_dispatch'" in science
    assert "github.ref == 'refs/heads/main'" in science
    assert "github.repository == 'IPS-Stuttgart/BayesianPhysTwin'" in science
    assert text.count("persist-credentials: false") >= 2
    assert "persist-credentials: false" in reusable
    assert "contents: write" not in text + reusable
    assert "git push" not in text + reusable


def test_workflow_binds_exact_runner_roots_revisions_and_identity() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    science = text[text.index("  science:") :]

    assert f"OFFICIAL_RAW_ROOT: {OFFICIAL_ROOT}" in text
    assert f"ADAPTIVE_CONFIRMATION_ROOT: {ADAPTIVE_ROOT}" in text
    assert (
        "OFFICIAL_RAW_SOURCE_REVISION: 7fea8e20231a47641d1d2bc8791920ec4e62ec5e"
    ) in text
    assert "DATASET_REVISION: f804696d7a133908c7497ffdab43819d879b5cbc" in text
    assert "PROCESSING_REVISION: d8522a4403b766aeb387510c04e89032a56fdf35" in text
    assert "AUTHORIZED_RUNNER_NAME: workstation2" in text
    assert (
        "LOCAL_SCIENCE_ROOT: "
        "/mnt/lexar4tb/datasets/deform360/bpt-runner-local-science-f804696d7a13"
    ) in text
    admission = _block(
        science,
        "      - name: Admit the sole Deform360 runner before checkout",
        "      - name: Check out exact trusted BayesianPhysTwin source",
    )
    assert 'test "${RUNNER_NAME}" = "${AUTHORIZED_RUNNER_NAME}"' in admission
    assert 'test "${RUNNER_OS}" = "Linux"' in admission
    assert 'test "${RUNNER_ARCH}" = "X64"' in admission
    assert "command -v nvidia-smi" in admission
    assert science.index("Admit the sole Deform360 runner") < science.index(
        "Check out exact trusted BayesianPhysTwin source"
    )


def test_runtime_is_exactly_locked_and_verified() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    reusable = CONTRACT_WORKFLOW.read_text(encoding="utf-8")
    lock_lines = [
        line.strip()
        for line in RUNTIME_LOCK.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert lock_lines
    assert all(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*==[^\s;]+", line)
        for line in lock_lines
    )
    assert len({line.split("==", 1)[0].lower() for line in lock_lines}) == len(
        lock_lines
    )
    for required in (
        "pip==25.2",
        "setuptools==80.9.0",
        "wheel==0.45.1",
        "numpy==2.2.6",
        "huggingface_hub==1.27.0",
        "opencv-contrib-python==5.0.0.93",
        "scipy==1.18.0",
    ):
        assert required in lock_lines
    assert text.count('--constraint "${RUNTIME_LOCK}"') >= 2
    assert "--no-build-isolation" in text
    assert "verify_deform360_runtime_lock.py" in text
    assert "--require-complete" in text
    assert "runtime-lock-validation.json" in text
    assert '--constraint "${RUNTIME_LOCK}"' in reusable
    assert "verify_deform360_runtime_lock.py" in reusable


def test_hugging_face_token_is_scoped_only_to_exact_download_step() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    download = _block(
        text,
        "      - name: Download and hash only missing frozen calibration files",
        "      - name: Prepare synchronized calibration RGB tactile and robot source",
    )
    before_download = text[
        : text.index(
            "      - name: Download and hash only missing frozen calibration files"
        )
    ]
    prepare_marker = (
        "      - name: Prepare synchronized calibration RGB tactile and robot source"
    )
    after_download = text[text.index(prepare_marker) :]

    assert text.count("${{ secrets.HF_TOKEN }}") == 1
    assert "${{ secrets.HF_TOKEN }}" in download
    assert 'HF_HUB_DISABLE_XET: "1"' in download
    assert "${{ secrets.HF_TOKEN }}" not in before_download
    assert "${{ secrets.HF_TOKEN }}" not in after_download


def test_adaptive_confirmation_root_is_names_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    inventory = _block(
        text,
        "      - name: Build names-only inventories of both runner-resident roots",
        "      - name: Compare runner-resident roots without opening dataset payloads",
    )
    reuse = _block(
        text,
        (
            "      - name: Reuse only exact calibration bytes from the official "
            "raw snapshot"
        ),
        "      - name: Admit plan-derived writable storage before download",
    )

    assert "${RESOLVED_OFFICIAL_RAW_ROOT}" in inventory
    assert "${RESOLVED_ADAPTIVE_CONFIRMATION_ROOT}" in inventory
    assert "stage_deform360_local_calibration_cache.py" in reuse
    assert '--source-root "${RESOLVED_OFFICIAL_RAW_ROOT}"' in reuse
    assert "RESOLVED_ADAPTIVE_CONFIRMATION_ROOT" not in reuse
    assert 'boundary["adaptive_confirmation_root_accessed"] is not False' in reuse
    assert 'boundary["confirmation_payloads_opened"] is not False' in reuse


def test_storage_capacity_is_derived_after_reuse_and_before_download() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    science = text[text.index("  science:") :]
    capacity = _block(
        science,
        "      - name: Admit plan-derived writable storage before download",
        "      - name: Download and hash only missing frozen calibration files",
    )

    assert science.index("Reuse only exact calibration bytes") < science.index(
        "Admit plan-derived writable storage"
    )
    assert science.index("Admit plan-derived writable storage") < science.index(
        "Download and hash only missing frozen calibration files"
    )
    assert "check_deform360_runner_capacity.py" in capacity
    assert '--plan "${EVIDENCE_ROOT}/calibration-source-plan.json"' in capacity
    assert '--data-root "${DATA_ROOT}"' in capacity
    assert '--processed-root "${PROCESSED_ROOT}"' in capacity
    assert '--cache-root "${CACHE_ROOT}"' in capacity
    assert '--reserve-bytes "${CAPACITY_RESERVE_BYTES}"' in capacity
    assert "storage-capacity.json" in capacity


def test_science_path_uses_existing_frozen_calibration_contracts() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    science = text[text.index("  science:") :]

    assert "run_deform360_official_hub_calibration_source.py plan" in science
    assert "run_deform360_official_hub_calibration_source.py download" in science
    assert "run_deform360_official_hub_calibration_source.py prepare" in science
    assert "protocols/deform360_official_hub_calibration_source_v1.json" in science
    assert (
        "protocols/locks/deform360_official_hub_visuotactile_v1_selection.json"
        in science
    )
    assert '--data-root "${DATA_ROOT}"' in science
    assert '--processing-repository "${PROCESSING_REPO}"' in science
    assert "--workers 2" in science
    assert "confirmation_payloads_opened=false" in science
    assert "adaptive_confirmation_payloads_opened=false" in science
    assert "target_outcomes_used=false" in science
    assert "nvidia-smi" in science


def test_raw_snapshot_reuse_is_copy_on_write_and_artifacts_are_compact() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    upload = _block(
        text,
        "      - name: Upload compact runner-local science evidence",
        "      - name: Remove isolated runtime and processing checkout",
    )

    assert "local-cache-staging.json" in text
    assert 'boundary["hardlink_allowed"] is not False' in text
    assert "${{ env.EVIDENCE_ROOT }}" in upload
    assert "${{ env.DATA_ROOT }}" not in upload
    assert "${{ env.PROCESSED_ROOT }}" not in upload
    assert "runtime-lock-validation.json" in upload
    assert "storage-capacity.json" in upload
    assert "retention-days: 30" in upload


def test_documentation_marks_bootstrap_as_non_claim_bearing() -> None:
    guide = GUIDE.read_text(encoding="utf-8")

    assert "not a claim-bearing experiment" in guide
    assert "workstation2" in guide
    assert "31236230283" in guide
    assert "`HF_TOKEN` is not a job environment variable" in guide
    assert "confirmation-opening" in guide
