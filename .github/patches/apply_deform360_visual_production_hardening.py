#!/usr/bin/env python3
"""Apply frozen-admission and single-session Deform360 production hardening."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

WORKFLOW = r'''name: Deform360 calibration visual production

on:
  pull_request:
    branches: [main]
    paths:
      - ".github/workflows/deform360-calibration-visual-production.yml"
      - ".github/workflows/launch-deform360-calibration-visual-production-once.yml"
      - "MANIFEST.in"
      - "docs/deform360_calibration_visual_production.md"
      - "scripts/science/execute_deform360_calibration_visual_production.py"
      - "src/bayesian_phystwin/deform360_calibration_visual_production.py"
      - "tests/test_deform360_calibration_visual_production.py"
      - "tests/test_deform360_calibration_visual_production_workflow.py"
  workflow_call:
    inputs:
      resume:
        description: Revalidate exact receipts and resume matching journals
        required: false
        default: true
        type: boolean
  workflow_dispatch:
    inputs:
      resume:
        description: Revalidate exact receipts and resume matching journals
        required: true
        default: true
        type: boolean

permissions:
  actions: read
  contents: read

concurrency:
  group: deform360-calibration-visual-production-${{ github.event_name == 'pull_request' && github.ref || 'protected-main' }}
  cancel-in-progress: false

env:
  BPT_HEAD_SHA: ${{ github.event.pull_request.head.sha || github.sha }}
  PROB4D_REVISION: 25d90ef7f78ba4307f4555cb636d666004e1bf66
  MOTIONCRAFTER_REVISION: 9cb4e9679f5f34e249945544052464ef46324bc2
  AUTHORITATIVE_ADMISSION_RUN_ID: "31272512658"
  AUTHORITATIVE_ADMISSION_ARTIFACT_ID: "9026043628"
  AUTHORITATIVE_ADMISSION_ARTIFACT_NAME: deform360-calibration-retained-source-admission-31272512658-1
  AUTHORITATIVE_ADMISSION_ARTIFACT_DIGEST: sha256:d0041af0ba0cfe6e5c5bd4008c47adb3ed4cf0cf0f6754eff67a238e746c7a86
  AUTHORITATIVE_INVENTORY_ID: 6994aa621b38dc8fb21cd38e43363bde3ea12dd644532addeecfc07a30f84e7b
  AUTHORITATIVE_PLAN_ID: 7743763cc61e905bebd264902102bf6c93bd064b466d395a15e8d552b4c9351b
  AUTHORITATIVE_ADMISSION_ID: 715ab8479bad4d97eba766cdba1a161f1f6e83e3fd597bb09a2bf8ab8dc91e15
  AUTHORITATIVE_RESULT_FILE_SHA256: 4a45f5ccd10c4f2d7c3d606aa37d6e1159ff5e5100bfca5f40d445e33ffd951a
  AUTHORITATIVE_CAMERA_VIEW_COUNT: "324"
  VAR_PROCESSED_ROOT: ${{ vars.DEFORM360_OFFICIAL_HUB_CALIBRATION_PROCESSED_ROOT }}
  VAR_OUTPUT_ROOT: ${{ vars.DEFORM360_CALIBRATION_VISUAL_OUTPUT_ROOT }}
  VAR_HF_CACHE_DIR: ${{ vars.MOTIONCRAFTER_HF_CACHE_DIR }}
  PYTHONHASHSEED: "0"
  PYTHONUNBUFFERED: "1"
  PYTHONDONTWRITEBYTECODE: "1"

jobs:
  contracts:
    name: Visual-production contracts / Python ${{ matrix.python-version }}
    runs-on: ubuntu-latest
    timeout-minutes: 30
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.12"]
    steps:
      - name: Check out exact proposed source
        uses: actions/checkout@v7
        with:
          ref: ${{ env.BPT_HEAD_SHA }}
          fetch-depth: 0
          persist-credentials: false
          clean: true
      - uses: actions/setup-python@v6
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
          cache-dependency-path: pyproject.toml
      - name: Validate contracts and distribution membership
        shell: bash
        run: |
          set -euo pipefail
          python -m pip install -e ".[dev,graph]"
          python -m pip install "build>=1.2"
          files=(
            src/bayesian_phystwin/deform360_calibration_visual_production.py
            scripts/science/execute_deform360_calibration_visual_production.py
            tests/test_deform360_calibration_visual_production.py
            tests/test_deform360_calibration_visual_production_workflow.py
          )
          python -m ruff check "${files[@]}"
          python -m ruff format --check "${files[@]}"
          python -m mypy \
            src/bayesian_phystwin/deform360_calibration_visual_production.py \
            scripts/science/execute_deform360_calibration_visual_production.py
          python -m pytest -q -p no:cacheprovider \
            tests/test_deform360_calibration_visual_production.py \
            tests/test_deform360_calibration_visual_production_workflow.py \
            tests/test_deform360_calibration_visual_execution_admission.py \
            tests/test_deform360_calibration_visual_execution_admission_edges.py
          rm -rf build dist ./*.egg-info src/*.egg-info
          python -m build --sdist
          archive="$(find dist -maxdepth 1 -name '*.tar.gz' -print -quit)"
          test -n "${archive}"
          listing="$(tar -tzf "${archive}")"
          grep -Fq '/docs/deform360_calibration_visual_production.md' <<<"${listing}"
          grep -Fq '/scripts/science/execute_deform360_calibration_visual_production.py' <<<"${listing}"
          grep -Fq '/src/bayesian_phystwin/deform360_calibration_visual_production.py' <<<"${listing}"
          rm -rf build dist ./*.egg-info src/*.egg-info
          git diff --exit-code
          test -z "$(git status --porcelain=v1)"

  production:
    name: Admitted all-camera production / workstation2
    if: >-
      github.event_name != 'pull_request' &&
      github.ref == 'refs/heads/main' &&
      github.repository == 'IPS-Stuttgart/BayesianPhysTwin'
    needs: contracts
    permissions:
      actions: read
      contents: read
    runs-on: [self-hosted, Linux, X64, nvidia-smi]
    timeout-minutes: 1320
    steps:
      - name: Publish frozen calibration-only beacon
        run: |
          echo "retained_source_artifact_id=${AUTHORITATIVE_ADMISSION_ARTIFACT_ID}"
          echo "retained_source_artifact_digest=${AUTHORITATIVE_ADMISSION_ARTIFACT_DIGEST}"
          echo "retained_source_admission_id=${AUTHORITATIVE_ADMISSION_ID}"
          echo "admitted_camera_jobs=${AUTHORITATIVE_CAMERA_VIEW_COUNT}"
          echo "model_loading_policy=single-session-shared-adapter-v1"
          echo "reserved_evaluation_frames_opened=false"
          echo "confirmation_payloads_opened=false"
          echo "target_outcomes_used=false"
      - name: Check out exact reviewed BayesianPhysTwin main
        uses: actions/checkout@v7
        with:
          ref: ${{ github.sha }}
          fetch-depth: 0
          persist-credentials: false
          clean: true
      - name: Check out exact Prob4D provider
        uses: actions/checkout@v7
        with:
          repository: IPS-Stuttgart/Prob4D
          ref: ${{ env.PROB4D_REVISION }}
          path: _prob4d
          fetch-depth: 1
          persist-credentials: false
          clean: true
      - name: Check out exact MotionCrafter provider
        uses: actions/checkout@v7
        with:
          repository: TencentARC/MotionCrafter
          ref: ${{ env.MOTIONCRAFTER_REVISION }}
          path: _motioncrafter
          fetch-depth: 1
          persist-credentials: false
          clean: true
      - name: Provision complete Python 3.12 runtime
        uses: actions/setup-python@v6
        with:
          python-version: "3.12"
          check-latest: false
      - name: Resolve protected roots and verify exact repository state
        shell: bash
        run: |
          set -euo pipefail
          test "${GITHUB_REF}" = "refs/heads/main"
          test "$(git rev-parse HEAD)" = "${GITHUB_SHA}"
          test "$(git -C _prob4d rev-parse HEAD)" = "${PROB4D_REVISION}"
          test "$(git -C _motioncrafter rev-parse HEAD)" = "${MOTIONCRAFTER_REVISION}"
          test -z "$(git status --porcelain=v1)"
          test -z "$(git -C _prob4d status --porcelain=v1)"
          test -z "$(git -C _motioncrafter status --porcelain=v1)"

          processed="${VAR_PROCESSED_ROOT:-/mnt/lexar4tb/datasets/deform360_official_hub_visuotactile_v1/calibration-processed}"
          output="${VAR_OUTPUT_ROOT:-/mnt/lexar4tb/datasets/deform360_official_hub_visuotactile_v1/calibration-visual-production}"
          cache="${VAR_HF_CACHE_DIR:-${HOME}/.cache/huggingface/hub}"
          for value in "${processed}" "${output}" "${cache}"; do
            [[ "${value}" = /* ]]
          done
          processed="$(realpath -m "${processed}")"
          output="$(realpath -m "${output}")"
          cache="$(realpath -m "${cache}")"
          test "${processed}" != "${output}"
          test "${processed}" != "${cache}"
          test "${output}" != "${cache}"
          test -d "${processed}/aligned"
          mkdir -p "${output}" "${cache}"

          evidence="${RUNNER_TEMP}/deform360-visual-production-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
          mkdir -p "${evidence}/admission" "${evidence}/compact"
          {
            echo "PROCESSED_ROOT=${processed}"
            echo "VISUAL_OUTPUT_ROOT=${output}"
            echo "MODEL_CACHE=${cache}"
            echo "EVIDENCE_ROOT=${evidence}"
            echo "ADMISSION_ROOT=${evidence}/admission"
            echo "COMPACT_ROOT=${evidence}/compact"
          } >> "${GITHUB_ENV}"
      - name: Verify authoritative admission artifact metadata
        env:
          GH_TOKEN: ${{ github.token }}
        shell: bash
        run: |
          set -euo pipefail
          python - <<'PY'
          import json
          import os
          import urllib.request

          request = urllib.request.Request(
              f"{os.environ['GITHUB_API_URL']}/repos/{os.environ['GITHUB_REPOSITORY']}"
              f"/actions/artifacts/{os.environ['AUTHORITATIVE_ADMISSION_ARTIFACT_ID']}",
              headers={
                  "Accept": "application/vnd.github+json",
                  "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                  "X-GitHub-Api-Version": "2022-11-28",
              },
          )
          with urllib.request.urlopen(request, timeout=60) as response:
              artifact = json.load(response)
          expected = {
              "id": int(os.environ["AUTHORITATIVE_ADMISSION_ARTIFACT_ID"]),
              "name": os.environ["AUTHORITATIVE_ADMISSION_ARTIFACT_NAME"],
              "digest": os.environ["AUTHORITATIVE_ADMISSION_ARTIFACT_DIGEST"],
              "expired": False,
          }
          observed = {key: artifact.get(key) for key in expected}
          if observed != expected:
              raise SystemExit(
                  f"authoritative retained-source artifact changed: {observed!r}"
              )
          PY
      - name: Download frozen authoritative retained-source admission
        uses: actions/download-artifact@v8
        with:
          name: ${{ env.AUTHORITATIVE_ADMISSION_ARTIFACT_NAME }}
          path: ${{ env.ADMISSION_ROOT }}
          github-token: ${{ github.token }}
          repository: ${{ github.repository }}
          run-id: ${{ env.AUTHORITATIVE_ADMISSION_RUN_ID }}
      - name: Verify frozen admission members and identities
        shell: bash
        run: |
          set -euo pipefail
          (
            cd "${ADMISSION_ROOT}"
            sha256sum -c SHA256SUMS
          )
          python - <<'PY'
          import json
          import os
          from pathlib import Path

          root = Path(os.environ["ADMISSION_ROOT"])
          receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
          inventory = json.loads(
              (root / "prepared-source-inventory.json").read_text(encoding="utf-8")
          )
          plan = json.loads(
              (root / "calibration-visual-production-plan.json").read_text(
                  encoding="utf-8"
              )
          )
          admission = json.loads(
              (root / "calibration-visual-execution-admission.json").read_text(
                  encoding="utf-8"
              )
          )
          expected = {
              "inventory_id": os.environ["AUTHORITATIVE_INVENTORY_ID"],
              "plan_id": os.environ["AUTHORITATIVE_PLAN_ID"],
              "admission_id": os.environ["AUTHORITATIVE_ADMISSION_ID"],
          }
          for field, value in expected.items():
              if receipt.get(field) != value:
                  raise SystemExit(f"receipt {field} changed")
          if inventory.get("inventory_id") != expected["inventory_id"]:
              raise SystemExit("inventory identity changed")
          if plan.get("plan_id") != expected["plan_id"]:
              raise SystemExit("visual-production plan identity changed")
          if admission.get("admission_id") != expected["admission_id"]:
              raise SystemExit("visual-execution admission identity changed")
          if plan.get("calibration_source_result_sha256") != os.environ[
              "AUTHORITATIVE_RESULT_FILE_SHA256"
          ]:
              raise SystemExit("calibration-source result file identity changed")
          count = int(os.environ["AUTHORITATIVE_CAMERA_VIEW_COUNT"])
          if (
              receipt.get("camera_view_count") != count
              or admission.get("camera_view_count") != count
              or len(admission.get("jobs", [])) != count
              or admission.get("object_count") != 10
          ):
              raise SystemExit("authoritative admitted roster changed")
          if receipt.get("confirmation_payloads_opened") is not False:
              raise SystemExit("confirmation boundary changed")
          if receipt.get("target_outcomes_used") is not False:
              raise SystemExit("target-outcome boundary changed")
          PY
          echo "ADMISSION_ID=${AUTHORITATIVE_ADMISSION_ID}" >> "${GITHUB_ENV}"
          echo "PRODUCTION_RUN_ROOT=${VISUAL_OUTPUT_ROOT}/${AUTHORITATIVE_ADMISSION_ID}/${GITHUB_SHA}" >> "${GITHUB_ENV}"
      - name: Bootstrap exact GPU producer environment
        shell: bash
        run: |
          set -euo pipefail
          env_root="${RUNNER_TEMP}/prob4d-motioncrafter-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
          bash _prob4d/scripts/bootstrap_motioncrafter_env.sh "${GITHUB_WORKSPACE}/_motioncrafter" "${env_root}"
          "${env_root}/bin/python" -m pip install --no-deps .
          "${env_root}/bin/python" -m pip check
          echo "PRODUCTION_PYTHON=${env_root}/bin/python" >> "${GITHUB_ENV}"
      - name: Bootstrap exact immutable model snapshots
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          "${PRODUCTION_PYTHON}" scripts/science/bootstrap_deform360_visual_provider_models.py \
            --spec protocols/locks/deform360_official_hub_visuotactile_v1_visual_provider_spec.json \
            --cache-dir "${MODEL_CACHE}" \
            --output "${ADMISSION_ROOT}/model-bootstrap.json"
      - name: Execute or resume every admitted causal-prefix job
        id: production
        shell: bash
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          set -euo pipefail
          resume=()
          if [[ "${{ inputs.resume }}" == "true" ]]; then resume=(--resume); fi
          set +e
          "${PRODUCTION_PYTHON}" scripts/science/execute_deform360_calibration_visual_production.py run \
            --admission "${ADMISSION_ROOT}/calibration-visual-execution-admission.json" \
            --visual-provider-lock protocols/locks/deform360_official_hub_visuotactile_v1_visual_provider/visual-provider-lock.json \
            --model-set-binding protocols/locks/deform360_official_hub_visuotactile_v1_visual_provider/motioncrafter-model-set.json \
            --retained-root "${PROCESSED_ROOT}/aligned" \
            --output-root "${VISUAL_OUTPUT_ROOT}" \
            --prob4d-root "${GITHUB_WORKSPACE}/_prob4d" \
            --motioncrafter-root "${GITHUB_WORKSPACE}/_motioncrafter" \
            --cache-dir "${MODEL_CACHE}" \
            --implementation-revision "${GITHUB_SHA}" \
            --attempt-id "${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}" \
            "${resume[@]}" | tee "${ADMISSION_ROOT}/production-console.json"
          code=${PIPESTATUS[0]}
          set -e
          if [[ "${code}" != 0 && "${code}" != 3 ]]; then exit "${code}"; fi
          echo "terminal_code=${code}" >> "${GITHUB_OUTPUT}"
      - name: Collect compact seals and accounting evidence
        if: always()
        shell: bash
        run: |
          set -euo pipefail
          mkdir -p "${COMPACT_ROOT}/admission"
          cp -a "${ADMISSION_ROOT}/." "${COMPACT_ROOT}/admission/"
          if [[ -n "${PRODUCTION_RUN_ROOT:-}" && -d "${PRODUCTION_RUN_ROOT}" ]]; then
            mkdir -p "${COMPACT_ROOT}/production"
            (
              cd "${PRODUCTION_RUN_ROOT}"
              find . -type f \
                \( -name 'visual-production-result.json' \
                   -o -name 'prediction-seal.json' \
                   -o -path './failures/*.json' \) \
                -print0 | sort -z | xargs -0 -r cp --parents -t "${COMPACT_ROOT}/production"
            )
          fi
          {
            echo "repository=${GITHUB_REPOSITORY}"
            echo "bpt_revision=${GITHUB_SHA}"
            echo "prob4d_revision=${PROB4D_REVISION}"
            echo "motioncrafter_revision=${MOTIONCRAFTER_REVISION}"
            echo "retained_source_artifact_id=${AUTHORITATIVE_ADMISSION_ARTIFACT_ID}"
            echo "retained_source_artifact_digest=${AUTHORITATIVE_ADMISSION_ARTIFACT_DIGEST}"
            echo "retained_source_inventory_id=${AUTHORITATIVE_INVENTORY_ID}"
            echo "retained_source_plan_id=${AUTHORITATIVE_PLAN_ID}"
            echo "retained_source_admission_id=${AUTHORITATIVE_ADMISSION_ID}"
            echo "admitted_camera_jobs=${AUTHORITATIVE_CAMERA_VIEW_COUNT}"
            echo "model_loading_policy=single-session-shared-adapter-v1"
            echo "reserved_evaluation_frames_opened=false"
            echo "confirmation_payloads_opened=false"
            echo "target_outcomes_used=false"
            nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
          } > "${COMPACT_ROOT}/environment.txt"
          (
            cd "${COMPACT_ROOT}"
            find . -type f ! -name SHA256SUMS -print0 \
              | sort -z | xargs -0 sha256sum > SHA256SUMS
          )
      - name: Upload compact calibration-only production evidence
        if: always()
        uses: actions/upload-artifact@v7
        with:
          name: deform360-calibration-visual-production-${{ github.run_id }}-${{ github.run_attempt }}
          path: ${{ env.COMPACT_ROOT }}
          if-no-files-found: error
          retention-days: 90
      - name: Verify closed information boundary and clean tracked checkouts
        if: always()
        run: |
          set -euo pipefail
          test -z "$(git status --porcelain=v1 --untracked-files=no)"
          test -z "$(git -C _prob4d status --porcelain=v1 --untracked-files=no)"
          test -z "$(git -C _motioncrafter status --porcelain=v1 --untracked-files=no)"
          echo "reserved_evaluation_frames_opened=false"
          echo "confirmation_payloads_opened=false"
          echo "target_outcomes_used=false"
'''

WORKFLOW_TEST = r'''from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW = Path(".github/workflows/deform360-calibration-visual-production.yml")
SCRIPT = Path("scripts/science/execute_deform360_calibration_visual_production.py")


def _workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_visual_production_workflow_is_valid_main_only_and_resumable() -> None:
    text = _workflow()
    parsed = yaml.load(text, Loader=yaml.BaseLoader)

    assert isinstance(parsed, dict)
    assert "workflow_call:" in text
    assert "workflow_dispatch:" in text
    assert "github.event_name != 'pull_request'" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert "github.repository == 'IPS-Stuttgart/BayesianPhysTwin'" in text
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in text
    assert "timeout-minutes: 1320" in text
    assert "cancel-in-progress: false" in text
    assert "--resume" in text
    assert "--attempt-id" in text


def test_visual_production_consumes_exact_frozen_admission_artifact() -> None:
    text = _workflow()

    assert 'AUTHORITATIVE_ADMISSION_RUN_ID: "31272512658"' in text
    assert 'AUTHORITATIVE_ADMISSION_ARTIFACT_ID: "9026043628"' in text
    assert (
        "deform360-calibration-retained-source-admission-31272512658-1" in text
    )
    assert (
        "sha256:d0041af0ba0cfe6e5c5bd4008c47adb3ed4cf0cf0f6754eff67a238e746c7a86"
        in text
    )
    assert (
        "715ab8479bad4d97eba766cdba1a161f1f6e83e3fd597bb09a2bf8ab8dc91e15"
        in text
    )
    assert "run-id: ${{ env.AUTHORITATIVE_ADMISSION_RUN_ID }}" in text
    assert "repository: ${{ github.repository }}" in text
    assert "github-token: ${{ github.token }}" in text
    assert "sha256sum -c SHA256SUMS" in text
    assert "uses: ./.github/workflows/deform360-calibration-prepared-inventory.yml" not in text
    assert "needs.retained-source" not in text


def test_visual_production_has_no_caller_selected_host_paths() -> None:
    text = _workflow()
    dispatch = text[text.index("  workflow_dispatch:") : text.index("\npermissions:")]

    assert "processed_root" not in dispatch
    assert "output_root" not in dispatch
    assert "hf_cache_dir" not in dispatch
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
    upload = text[text.index("Upload compact calibration-only production evidence") :]

    assert "*.npz" not in upload
    assert "predictions.json" not in upload
    assert "confirmation-processed" not in text
    assert "reserved_evaluation_frames_opened=false" in text
    assert "confirmation_payloads_opened=false" in text
    assert "target_outcomes_used=false" in text
    assert "replacement_allowed=true" not in text


def test_hugging_face_token_is_not_workflow_wide() -> None:
    text = _workflow()
    global_env = text[text.index("env:") : text.index("jobs:")]

    assert "HF_TOKEN" not in global_env
    assert text.count("HF_TOKEN: ${{ secrets.HF_TOKEN }}") == 2
'''


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    source = target.read_text(encoding="utf-8")
    count = source.count(old)
    if count != 1:
        raise SystemExit(
            f"expected exactly one source fragment in {path}, observed {count}"
        )
    target.write_text(source.replace(old, new), encoding="utf-8")


def main() -> None:
    (ROOT / ".github/workflows/deform360-calibration-visual-production.yml").write_text(
        WORKFLOW,
        encoding="utf-8",
    )
    (ROOT / "tests/test_deform360_calibration_visual_production_workflow.py").write_text(
        WORKFLOW_TEST,
        encoding="utf-8",
    )

    script = "scripts/science/execute_deform360_calibration_visual_production.py"
    replace_once(
        script,
        '''import argparse
import fcntl
import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
''',
        '''import argparse
import fcntl
import gc
import hashlib
import json
import os
import stat
import subprocess
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
''',
    )
    replace_once(
        script,
        '''    build_deform360_calibration_visual_command,
    build_deform360_calibration_visual_prediction_seal,
''',
        '''    build_deform360_calibration_visual_prediction_seal,
''',
    )
    replace_once(
        script,
        '''@dataclass(frozen=True)
class ProcessOutcome:
    return_code: int
    stdout: bytes
    stderr: bytes


''',
        '''@dataclass(frozen=True)
class ProcessOutcome:
    return_code: int
    stdout: bytes
    stderr: bytes


_VARIABLE_CONFIG_FIELDS = frozenset(
    {"video_path", "output_directory", "seed", "frame_start", "frame_stop"}
)


class _SharedAdapterFactory:
    """Load the pinned model set once and rebind only per-job run fields."""

    def __init__(self, factory: Callable[[Any], Any]) -> None:
        self._factory = factory
        self._adapter: Any | None = None
        self._fixed_config: dict[str, object] | None = None
        self.creation_count = 0

    @staticmethod
    def _fixed_record(config: Any) -> dict[str, object]:
        record = asdict(config)
        for field in _VARIABLE_CONFIG_FIELDS:
            record.pop(field, None)
        return {
            key: str(value) if isinstance(value, Path) else cast(object, value)
            for key, value in record.items()
        }

    @property
    def adapter(self) -> Any | None:
        return self._adapter

    def __call__(self, config: Any) -> Any:
        fixed = self._fixed_record(config)
        if self._adapter is None:
            self._adapter = self._factory(config)
            self._fixed_config = fixed
            self.creation_count += 1
        else:
            if fixed != self._fixed_config:
                raise ValueError("per-job MotionCrafter config changed fixed fields")
            self._adapter.config = config
        return self._adapter


class _SharedMotionCrafterProducer:
    """Run integrity-bound per-view journals through one loaded adapter."""

    def __init__(
        self,
        *,
        model_binding: Mapping[str, Any],
        motioncrafter_root: Path,
        cache_directory: Path,
        provider_lock: Deform360VisualProviderLockV1,
    ) -> None:
        from prob4d.motioncrafter_integrity import (
            verify_motioncrafter_prediction_manifest,
        )
        from prob4d.motioncrafter_models import PinnedMotionCrafterModelSet
        from prob4d.motioncrafter_runner import SafeMotionCrafterRunner

        sources = cast(Mapping[str, Mapping[str, str]], model_binding["sources"])
        model_set = PinnedMotionCrafterModelSet.inspect(
            model_type="determ",
            unet_reference=sources["unet"]["repository"],
            unet_revision=sources["unet"]["revision"],
            vae_reference=sources["vae"]["repository"],
            vae_revision=sources["vae"]["revision"],
            image_vae_reference=sources["image_vae"]["repository"],
            image_vae_revision=sources["image_vae"]["revision"],
            base_pipeline_reference=sources["base_pipeline"]["repository"],
            base_pipeline_revision=sources["base_pipeline"]["revision"],
        )
        if model_set.set_sha256 != model_binding["model_set_id"]:
            raise ValueError("installed Prob4D model-set identity changed")
        if model_set.manifest() != model_binding["manifest"]:
            raise ValueError("installed Prob4D model-set manifest changed")
        self._model_set = model_set
        self._motioncrafter_root = motioncrafter_root
        self._cache_directory = cache_directory
        self._provider_lock = provider_lock
        self._runner_class = SafeMotionCrafterRunner
        self._verifier = verify_motioncrafter_prediction_manifest
        self._shared_factory = _SharedAdapterFactory(model_set.adapter_factory())

    @property
    def model_load_count(self) -> int:
        return self._shared_factory.creation_count

    def produce(
        self,
        *,
        job: Mapping[str, Any],
        source_video_path: Path,
        output_directory: Path,
        resume: bool,
    ) -> Path:
        prefix = cast(Sequence[int], job["prefix_source_frame_range_half_open"])
        config = self._model_set.build_config(
            upstream_root=self._motioncrafter_root,
            video_path=source_video_path,
            output_directory=output_directory,
            cache_directory=str(self._cache_directory),
            height=self._provider_lock.height,
            width=self._provider_lock.width,
            window_size=self._provider_lock.window_size,
            overlap=self._provider_lock.overlap,
            num_inference_steps=5,
            guidance_scale=1.0,
            decode_chunk_size=25,
            seed=cast(int, job["view_root_seed"]),
            seed_policy="derived-per-call",
            low_memory_usage=False,
            frame_start=prefix[0],
            frame_stop=prefix[1],
            frame_stride=1,
        )
        manifest = self._runner_class(
            config,
            adapter_factory=self._shared_factory,
        ).run(resume=resume)
        if self.model_load_count != 1:
            raise RuntimeError("MotionCrafter adapter was not loaded exactly once")
        return cast(Path, manifest)

    def verify(self, manifest_path: Path) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self._verifier(manifest_path, verify_hashes=True),
        )

    def release_temporaries(self) -> None:
        gc.collect()
        adapter = self._shared_factory.adapter
        torch = getattr(adapter, "torch", None)
        cuda = getattr(torch, "cuda", None)
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()


''',
    )
    replace_once(
        script,
        '''    model_binding: Mapping[str, Any],
    prob4d_executable: Path,
    resume: bool,
''',
        '''    model_binding: Mapping[str, Any],
    verify_prediction: Callable[[Path], Mapping[str, Any]],
    resume: bool,
''',
    )
    replace_once(
        script,
        '''        verified = _run(
            (
                str(prob4d_executable),
                "--output-dir",
                str(output),
                "--verify-only",
            )
        )
        verification = _parse_stdout_json(
            verified,
            label="Prob4D sealed prediction verifier",
        )
''',
        '''        verification = verify_prediction(manifest_path)
''',
    )
    replace_once(
        script,
        '''    output_root: Path,
    prob4d_executable: Path,
    prob4d_root: Path,
''',
        '''    output_root: Path,
    prob4d_root: Path,
''',
    )
    replace_once(
        script,
        '''    if not prob4d_executable.is_file() or not os.access(prob4d_executable, os.X_OK):
        raise ValueError("prob4d-motioncrafter executable is missing or not executable")
    admission = load_deform360_calibration_visual_execution_admission(admission_path)
''',
        '''    admission = load_deform360_calibration_visual_execution_admission(admission_path)
''',
    )
    replace_once(
        script,
        '''    run_root = output_root / cast(str, admission["admission_id"])
''',
        '''    run_root = (
        output_root / cast(str, admission["admission_id"]) / implementation
    )
''',
    )
    replace_once(
        script,
        '''    lock_path = run_root / ".production.lock"
''',
        '''    producer = _SharedMotionCrafterProducer(
        model_binding=binding,
        motioncrafter_root=motioncrafter_root,
        cache_directory=cache_directory,
        provider_lock=lock,
    )

    lock_path = run_root / ".production.lock"
''',
    )
    replace_once(
        script,
        '''                model_binding=binding,
                prob4d_executable=prob4d_executable,
                resume=resume,
''',
        '''                model_binding=binding,
                verify_prediction=producer.verify,
                resume=resume,
''',
    )
    replace_once(
        script,
        '''            command = build_deform360_calibration_visual_command(
                executable=prob4d_executable,
                source_video_path=sources[cast(str, job["job_id"])],
                output_directory=output,
                motioncrafter_root=motioncrafter_root,
                cache_directory=cache_directory,
                job=job,
                provider_lock=lock,
                model_binding=binding,
                resume=resume,
            )
            try:
                produced = _run(command)
            except Exception as error:  # pragma: no cover - process adapter boundary
                produced = ProcessOutcome(1, b"", repr(error).encode())
            if produced.return_code != 0:
                receipt = _failure(
                    run_root=run_root,
                    attempt_id=attempt_id,
                    implementation_revision=implementation,
                    admission=admission,
                    lock=lock,
                    job=job,
                    command_id=command_id,
                    stage="motioncrafter-production",
                    outcome=produced,
                    detail=b"producer returned a nonzero exit status",
                )
                rows.append(
                    _result_row(
                        run_root=run_root,
                        job=job,
                        status="technical-failure",
                        receipt=receipt,
                    )
                )
                continue

            verified = ProcessOutcome(1, b"", b"")
            try:
                verified = _run(
                    (
                        str(prob4d_executable),
                        "--output-dir",
                        str(output),
                        "--verify-only",
                    )
                )
                verification = _parse_stdout_json(
                    verified,
                    label="Prob4D prediction verifier",
                )
                manifest_path = output / "predictions.json"
''',
        '''            try:
                manifest_path = producer.produce(
                    job=job,
                    source_video_path=sources[cast(str, job["job_id"])],
                    output_directory=output,
                    resume=resume,
                )
            except Exception:
                outcome = ProcessOutcome(
                    1,
                    b"",
                    traceback.format_exc().encode("utf-8", errors="replace"),
                )
                receipt = _failure(
                    run_root=run_root,
                    attempt_id=attempt_id,
                    implementation_revision=implementation,
                    admission=admission,
                    lock=lock,
                    job=job,
                    command_id=command_id,
                    stage="motioncrafter-production",
                    outcome=outcome,
                    detail=b"single-session producer raised an exception",
                )
                rows.append(
                    _result_row(
                        run_root=run_root,
                        job=job,
                        status="technical-failure",
                        receipt=receipt,
                    )
                )
                producer.release_temporaries()
                continue

            verification_outcome = ProcessOutcome(1, b"", b"")
            try:
                verification = producer.verify(manifest_path)
''',
    )
    replace_once(
        script,
        '''            except Exception as error:
                detail = repr(error).encode("utf-8", errors="replace")
                receipt = _failure(
''',
        '''            except Exception:
                detail = traceback.format_exc().encode("utf-8", errors="replace")
                verification_outcome = ProcessOutcome(1, b"", detail)
                receipt = _failure(
''',
    )
    replace_once(
        script,
        '''                    outcome=verified,
                    detail=detail,
''',
        '''                    outcome=verification_outcome,
                    detail=detail,
''',
    )
    replace_once(
        script,
        '''                )
                continue
            seal_path = output / "prediction-seal.json"
''',
        '''                )
                producer.release_temporaries()
                continue
            seal_path = output / "prediction-seal.json"
''',
    )
    replace_once(
        script,
        '''            rows.append(
                _result_row(
                    run_root=run_root,
                    job=job,
                    status="succeeded",
                    receipt=seal_path,
                )
            )
        result = build_deform360_calibration_visual_production_result(
''',
        '''            rows.append(
                _result_row(
                    run_root=run_root,
                    job=job,
                    status="succeeded",
                    receipt=seal_path,
                )
            )
            producer.release_temporaries()
        if producer.model_load_count > 1:
            raise RuntimeError("more than one MotionCrafter adapter was loaded")
        result = build_deform360_calibration_visual_production_result(
''',
    )
    replace_once(
        script,
        '''    run.add_argument("--prob4d-motioncrafter", type=Path, required=True)
''',
        '''''',
    )
    replace_once(
        script,
        '''                output_root=arguments.output_root,
                prob4d_executable=arguments.prob4d_motioncrafter,
                prob4d_root=arguments.prob4d_root,
''',
        '''                output_root=arguments.output_root,
                prob4d_root=arguments.prob4d_root,
''',
    )

    tests = ROOT / "tests/test_deform360_calibration_visual_production.py"
    source = tests.read_text(encoding="utf-8")
    if "test_shared_adapter_factory_reuses_one_loaded_adapter" not in source:
        source += r'''


def test_shared_adapter_factory_reuses_one_loaded_adapter() -> None:
    import runpy
    from dataclasses import dataclass
    from pathlib import Path

    namespace = runpy.run_path(
        "scripts/science/execute_deform360_calibration_visual_production.py"
    )
    shared_factory_type = namespace["_SharedAdapterFactory"]

    @dataclass(frozen=True)
    class Config:
        upstream_root: Path
        video_path: Path
        output_directory: Path
        cache_directory: str
        height: int
        seed: int
        frame_start: int
        frame_stop: int

    class Adapter:
        def __init__(self, config: Config) -> None:
            self.config = config

    created: list[Adapter] = []

    def factory(config: Config) -> Adapter:
        adapter = Adapter(config)
        created.append(adapter)
        return adapter

    shared = shared_factory_type(factory)
    first_config = Config(
        upstream_root=Path("MotionCrafter"),
        video_path=Path("a.mp4"),
        output_directory=Path("a"),
        cache_directory="cache",
        height=320,
        seed=11,
        frame_start=0,
        frame_stop=58,
    )
    second_config = Config(
        upstream_root=Path("MotionCrafter"),
        video_path=Path("b.mp4"),
        output_directory=Path("b"),
        cache_directory="cache",
        height=320,
        seed=12,
        frame_start=20,
        frame_stop=78,
    )

    first = shared(first_config)
    second = shared(second_config)

    assert first is second
    assert len(created) == 1
    assert shared.creation_count == 1
    assert second.config == second_config

    changed_fixed = Config(
        upstream_root=Path("different"),
        video_path=Path("c.mp4"),
        output_directory=Path("c"),
        cache_directory="cache",
        height=320,
        seed=13,
        frame_start=40,
        frame_stop=98,
    )
    with pytest.raises(ValueError, match="fixed fields"):
        shared(changed_fixed)
'''
    tests.write_text(source, encoding="utf-8")

    docs = "docs/deform360_calibration_visual_production.md"
    replace_once(
        docs,
        '''At runtime the workflow:

1. invokes the reviewed reusable retained-source admission workflow already
   merged on `main`;
2. consumes its uploaded inventory, plan, admission, content identities, and
   artifact digest without rebuilding a parallel custody path;
3. verifies clean, exact BayesianPhysTwin, Prob4D, and MotionCrafter revisions;
4. bootstraps the exact model snapshots frozen by the provider lock;
5. executes each admitted camera job through the pinned
   `prob4d-motioncrafter` entry point; and
6. uploads only compact admission metadata, per-job seals or retained failure
   receipts, complete accounting, and environment evidence.
''',
        '''At runtime the workflow:

1. downloads the frozen successful retained-source artifact from run
   `31272512658` by exact artifact ID, name, and digest;
2. verifies its internal `SHA256SUMS`, inventory ID, plan ID, admission ID,
   ten-object roster, and all 324 admitted camera jobs;
3. verifies clean, exact BayesianPhysTwin, Prob4D, and MotionCrafter revisions;
4. bootstraps the exact model snapshots frozen by the provider lock;
5. loads the pinned MotionCrafter model set once, then executes every unfinished
   camera through a separate crash-safe Prob4D runner and progress journal; and
6. uploads only compact admission metadata, per-job seals or retained failure
   receipts, complete accounting, and environment evidence.
''',
    )
    replace_once(
        docs,
        '''The generated Prob4D command binds:
''',
        '''Every generated Prob4D run configuration binds:
''',
    )
    replace_once(
        docs,
        '''The persistent run directory is keyed by the execution-admission ID. Re-running
with `resume=true`:
''',
        '''The persistent run directory is keyed by the execution-admission ID and exact
BayesianPhysTwin implementation revision. Re-running the same revision with
`resume=true`:
''',
    )
    replace_once(
        docs,
        '''  --retained-root /protected/calibration-processed/aligned \
  --output-root /protected/calibration-visual-production \
  --prob4d-motioncrafter /exact/env/bin/prob4d-motioncrafter \
  --prob4d-root /exact/Prob4D \
''',
        '''  --retained-root /protected/calibration-processed/aligned \
  --output-root /protected/calibration-visual-production \
  --prob4d-root /exact/Prob4D \
''',
    )
    replace_once(
        docs,
        '''A process-level file lock prevents concurrent writers while allowing automatic
release after cancellation or runner failure.
''',
        '''A process-level file lock prevents concurrent writers while allowing automatic
release after cancellation or runner failure. The shared adapter changes only
per-job video, output, seed, and frame-boundary fields; any change to a fixed
model or inference setting fails closed before reuse.
''',
    )


if __name__ == "__main__":
    main()
