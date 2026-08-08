from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW = Path(".github/workflows/deform360-calibration-visual-production.yml")
LAUNCHER = Path(
    ".github/workflows/launch-deform360-calibration-visual-production-once.yml"
)
SCRIPT = Path("scripts/science/execute_deform360_calibration_visual_production.py")


def _workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_visual_production_workflow_is_valid_main_only_and_resumable() -> None:
    text = _workflow()
    parsed = yaml.load(text, Loader=yaml.BaseLoader)

    assert isinstance(parsed, dict)
    assert "workflow_call:" in text
    assert "workflow_dispatch:" not in text
    assert "inputs.execute_authorized == true" in text
    assert "github.event_name == 'push'" in text
    assert (
        "IPS-Stuttgart/BayesianPhysTwin/.github/workflows/"
        "launch-deform360-calibration-visual-production-once.yml@refs/heads/main"
        in text
    )
    assert "github.ref == 'refs/heads/main'" in text
    assert "github.repository == 'IPS-Stuttgart/BayesianPhysTwin'" in text
    assert "runs-on: self-hosted" in text
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" not in text
    assert "timeout-minutes: 1320" in text
    assert "cancel-in-progress: false" in text
    assert "--resume" in text
    assert "--attempt-id" in text


def test_one_shot_launcher_calls_only_the_reviewed_reusable_lane() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    parsed = yaml.load(text, Loader=yaml.BaseLoader)

    assert isinstance(parsed, dict)
    assert "workflow_dispatch:" not in text
    assert "branches: [main]" in text
    assert (
        text.count(
            '      - ".github/workflows/'
            'launch-deform360-calibration-visual-production-once.yml"'
        )
        == 1
    )
    assert (
        "uses: ./.github/workflows/deform360-calibration-visual-production.yml" in text
    )
    assert "execute_authorized: true" in text
    assert "resume: true" in text
    assert "secrets: inherit" in text
    assert "cancel-in-progress: false" in text
    assert r"official raw payload opened: \`false\`" in text
    assert r"adaptive-confirmation payloads opened: \`false\`" in text
    assert r"confirmation payloads opened: \`false\`" in text
    assert r"target outcomes used: \`false\`" in text
    assert r"replacement allowed: \`false\`" in text
    assert r"pre-payload predecessor: run \`31274946936\`" in text
    assert r"environment-bootstrap predecessor: run \`31275886113\`" in text
    assert "2026-08-09-sole-self-hosted-layout-v4" in text


def test_visual_production_excludes_nested_checkouts_and_seals_early_failures() -> None:
    text = _workflow()
    production = text[text.index("  production:") :]

    exclude = "printf '%s\\n' '/_prob4d/' '/_motioncrafter/' >> .git/info/exclude"
    assert production.count(exclude) == 1
    assert production.index(exclude) < production.index(
        'test -z "$(git status --porcelain=v1)"'
    )
    assert production.index(
        'echo "COMPACT_ROOT=${evidence}/compact"'
    ) < production.index("Check out exact reviewed BayesianPhysTwin main")
    assert 'if [[ ! -d "${processed}/aligned" ]]' in production
    assert "The frozen calibration-processed root is unavailable." in production


def test_visual_production_uses_uv_for_the_unseeded_producer_environment() -> None:
    text = _workflow()
    bootstrap = text[
        text.index("Bootstrap exact GPU producer environment") : text.index(
            "Bootstrap exact immutable model snapshots"
        )
    ]

    assert "if command -v uv >/dev/null 2>&1" in bootstrap
    assert 'uv_bin="${HOME}/.local/bin/uv"' in bootstrap
    assert '"${uv_bin}" pip install \\\n' in bootstrap
    assert '--python "${env_root}/bin/python"' in bootstrap
    assert '"${uv_bin}" pip check --python "${env_root}/bin/python"' in bootstrap
    assert '"${env_root}/bin/python" -m pip' not in bootstrap


def test_visual_production_consumes_exact_frozen_admission_artifact() -> None:
    text = _workflow()

    assert 'AUTHORITATIVE_ADMISSION_RUN_ID: "31272512658"' in text
    assert 'AUTHORITATIVE_ADMISSION_ARTIFACT_ID: "9026043628"' in text
    assert "deform360-calibration-retained-source-admission-31272512658-1" in text
    assert (
        "sha256:d0041af0ba0cfe6e5c5bd4008c47adb3ed4cf0cf0f6754eff67a238e746c7a86"
        in text
    )
    assert "715ab8479bad4d97eba766cdba1a161f1f6e83e3fd597bb09a2bf8ab8dc91e15" in text
    assert "run-id: ${{ env.AUTHORITATIVE_ADMISSION_RUN_ID }}" in text
    assert "repository: ${{ github.repository }}" in text
    assert "github-token: ${{ github.token }}" in text
    assert "sha256sum -c SHA256SUMS" in text
    assert (
        "uses: ./.github/workflows/deform360-calibration-prepared-inventory.yml"
        not in text
    )
    assert "needs.retained-source" not in text


def test_visual_production_binds_the_sole_runner_and_exact_raw_roots() -> None:
    text = _workflow()

    assert "name: Admitted all-camera production / sole Deform360 runner" in text
    assert "runs-on: self-hosted" in text
    assert "runner_label_contract=self-hosted-only" in text
    assert "command -v nvidia-smi" in text
    assert "nvidia-smi -L" in text
    assert "DEFORM360_STORAGE_ROOT: /mnt/lexar4tb/datasets/deform360" in text
    assert (
        "DEFORM360_OFFICIAL_RAW_ROOT: "
        "/mnt/lexar4tb/datasets/deform360/data-7fea8e2" in text
    )
    assert (
        "DEFORM360_ADAPTIVE_CONFIRMATION_RAW_ROOT: "
        "/mnt/lexar4tb/datasets/deform360/"
        "adaptive-confirmation-download-5a9c56d593462486bdd0953dcaf6f9c643bf8370"
        in text
    )
    assert 'storage="$(realpath -e "${DEFORM360_STORAGE_ROOT}")"' in text
    assert 'official_raw="$(realpath -e "${DEFORM360_OFFICIAL_RAW_ROOT}")"' in text
    assert (
        'adaptive_raw="$(realpath -e '
        '"${DEFORM360_ADAPTIVE_CONFIRMATION_RAW_ROOT}")"' in text
    )
    assert "adaptive_confirmation_directory_stat_only" in text
    assert "adaptive_confirmation_payloads_opened=false" in text


def test_visual_production_keeps_outputs_and_cache_on_the_dataset_volume() -> None:
    text = _workflow()

    assert "${storage}/results/bayesian-phystwin/calibration-visual-production" in text
    assert "${storage}/caches/huggingface/hub" in text
    assert (
        "Production output and model cache must stay on the Deform360 volume." in text
    )
    assert "Processed, raw, output, and cache roots must be disjoint." in text
    assert "runner-storage-preflight.json" in text
    assert "storage_total_bytes" in text
    assert "storage_free_bytes" in text
    assert 'find "${storage}" -mindepth 2 -maxdepth 6' in text
    assert '-path "${official_raw}" -o -path "${adaptive_raw}"' in text


def test_visual_production_has_no_caller_selected_host_paths() -> None:
    text = _workflow()
    call_contract = text[text.index("  workflow_call:") : text.index("\npermissions:")]

    assert "processed_root" not in call_contract
    assert "output_root" not in call_contract
    assert "hf_cache_dir" not in call_contract
    assert "INPUT_PROCESSED_ROOT" not in text
    assert "INPUT_OUTPUT_ROOT" not in text
    assert "INPUT_HF_CACHE_DIR" not in text
    assert "VAR_PROCESSED_ROOT" in text
    assert "VAR_OUTPUT_ROOT" in text
    assert "VAR_HF_CACHE_DIR" in text


def test_visual_production_pins_external_sources_and_single_model_session() -> None:
    text = _workflow()
    script = SCRIPT.read_text(encoding="utf-8")

    assert "25d90ef7f78ba4307f4555cb636d666004e1bf66" in text
    assert "9cb4e9679f5f34e249945544052464ef46324bc2" in text
    assert "model_loading_policy=single-session-shared-adapter-v1" in text
    assert "_SharedAdapterFactory" in script
    assert "SafeMotionCrafterRunner" in script
    assert "PinnedMotionCrafterModelSet" in script
    assert "produced = _run(command)" not in script
    assert "--prob4d-motioncrafter" not in text
    assert "--prob4d-motioncrafter" not in script


def test_visual_production_artifact_excludes_large_predictions_and_targets() -> None:
    text = _workflow()
    execution = text[
        text.index("Execute or resume every admitted causal-prefix job") : text.index(
            "Collect compact seals and accounting evidence"
        )
    ]
    compact = text[text.index("Collect compact seals and accounting evidence") :]

    assert "*.npz" not in compact
    assert "predictions.json" not in compact
    assert "confirmation-processed" not in text
    assert "ADAPTIVE_CONFIRMATION" not in execution
    assert "official_raw_payload_opened=false" in text
    assert "reserved_evaluation_frames_opened=false" in text
    assert "adaptive_confirmation_payloads_opened=false" in text
    assert "confirmation_payloads_opened=false" in text
    assert "target_outcomes_used=false" in text
    assert 'echo "job_status=${{ job.status }}"' in compact
    assert "replacement_allowed=true" not in text


def test_hugging_face_token_is_not_workflow_wide() -> None:
    text = _workflow()
    global_env = text[text.index("env:") : text.index("jobs:")]

    assert "HF_TOKEN" not in global_env
    assert (
        "secrets:"
        in text[text.index("  workflow_call:") : text.index("\npermissions:")]
    )
    assert text.count("HF_TOKEN: ${{ secrets.HF_TOKEN }}") == 2
