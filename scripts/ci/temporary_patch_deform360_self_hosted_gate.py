"""One-shot exact-source patch for the Deform360 calibration workflow.

This file is intentionally temporary. The publishing workflow validates and
commits only the three permanent files changed below; this script is deleted
before the pull request is reviewed or merged.
"""

from __future__ import annotations

from pathlib import Path


WORKFLOW_PATH = Path(
    ".github/workflows/deform360-official-hub-calibration-source.yml"
)
TEST_PATH = Path("tests/test_deform360_calibration_source_workflow.py")
DOC_PATH = Path("docs/deform360_official_hub_calibration_source_v1.md")


CONTRACT_JOB = '''  contracts:
    name: Validate reviewed calibration-source contracts / workstation2
    if: >-
      github.event_name != 'pull_request' ||
      github.event.pull_request.head.repo.full_name == github.repository
    runs-on: [self-hosted, Linux, X64, nvidia-smi]
    timeout-minutes: 45
    steps:
      - name: Check out exact reviewed source
        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          ref: ${{ env.BPT_SOURCE_SHA }}
          fetch-depth: 0
          persist-credentials: false
          clean: true

      - name: Initialize an isolated contract target site
        shell: bash
        run: |
          set -euo pipefail
          contract_root="${RUNNER_TEMP}/deform360-calibration-contracts-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
          contract_site="${contract_root}/site"
          rm -rf "${contract_root}"
          mkdir -p "${contract_site}"

          base_python=""
          for candidate in \
            /usr/bin/python3.12 \
            /usr/local/bin/python3.12 \
            /usr/bin/python3 \
            /usr/local/bin/python3
          do
            if [[ -x "${candidate}" ]]; then
              base_python="${candidate}"
              break
            fi
          done
          if [[ -z "${base_python}" ]]; then
            base_python="$(command -v python3 || true)"
          fi
          if [[ -z "${base_python}" ]]; then
            echo "No system Python is available on the self-hosted runner." >&2
            exit 1
          fi

          {
            echo "CONTRACT_ROOT=${contract_root}"
            echo "CONTRACT_SITE=${contract_site}"
            echo "CONTRACT_BASE_PYTHON=${base_python}"
            echo "PYTHONPATH=${GITHUB_WORKSPACE}/src:${contract_site}${PYTHONPATH:+:${PYTHONPATH}}"
          } >> "${GITHUB_ENV}"
          "${base_python}" --version | tee "${contract_root}/base-python.txt"
          "${base_python}" -m pip --version | tee "${contract_root}/base-pip.txt"

      - name: Populate the fresh contract target site
        shell: bash
        run: |
          set -euo pipefail
          "${CONTRACT_BASE_PYTHON}" -m pip install \
            --break-system-packages \
            --no-cache-dir \
            --target "${CONTRACT_SITE}" \
            ".[dev,graph]" \
            "huggingface_hub>=0.24"
          "${CONTRACT_BASE_PYTHON}" -m pip check
          {
            echo "repository=${GITHUB_REPOSITORY}"
            echo "event_name=${GITHUB_EVENT_NAME}"
            echo "event_revision=${GITHUB_SHA}"
            echo "checked_out_revision=$(git rev-parse HEAD)"
            echo "runner_name=${RUNNER_NAME}"
            echo "runner_os=${RUNNER_OS}"
            echo "runner_arch=${RUNNER_ARCH}"
            "${CONTRACT_BASE_PYTHON}" --version
            "${CONTRACT_BASE_PYTHON}" -m pip --version
            "${CONTRACT_BASE_PYTHON}" -m pip freeze --path "${CONTRACT_SITE}"
          } > "${CONTRACT_ROOT}/environment.txt"

      - name: Validate source without opening dataset payloads
        shell: bash
        run: |
          set -euo pipefail
          python="${CONTRACT_BASE_PYTHON}"
          files=(
            scripts/science/deform360_calibration_source/__init__.py
            scripts/science/deform360_calibration_source/contracts.py
            scripts/science/deform360_calibration_source/planning.py
            scripts/science/deform360_calibration_source/download.py
            scripts/science/deform360_calibration_source/prepare.py
            scripts/science/deform360_calibration_source/cli.py
            scripts/science/run_deform360_official_hub_calibration_source.py
            tests/test_deform360_official_hub_calibration_source.py
            tests/test_deform360_calibration_source_workflow.py
          )
          "${python}" -m ruff check "${files[@]}"
          "${python}" -m ruff format --check "${files[@]}"
          "${python}" -m pytest -q -p no:cacheprovider \
            tests/test_deform360_official_hub_calibration_source.py \
            tests/test_deform360_calibration_source_workflow.py \
            tests/test_deform360_calibration_execution.py \
            tests/test_deform360_visual_provider_freeze.py \
            tests/test_pull_request_workflow_integrity.py \
            | tee "${CONTRACT_ROOT}/pytest.txt"
          "${python}" -m compileall -q \
            scripts/science/deform360_calibration_source \
            scripts/science/run_deform360_official_hub_calibration_source.py
          git diff --exit-code
          test -z "$(git status --porcelain --untracked-files=all)"

      - name: Upload compact contract evidence
        if: always()
        uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7
        with:
          name: deform360-calibration-contracts-${{ github.run_id }}-${{ github.run_attempt }}
          path: |
            ${{ env.CONTRACT_ROOT }}/base-python.txt
            ${{ env.CONTRACT_ROOT }}/base-pip.txt
            ${{ env.CONTRACT_ROOT }}/environment.txt
            ${{ env.CONTRACT_ROOT }}/pytest.txt
          if-no-files-found: warn
          retention-days: 30

      - name: Remove the isolated contract target site
        if: always()
        shell: bash
        run: |
          set -euo pipefail
          if [[ -n "${CONTRACT_ROOT:-}" ]]; then
            rm -rf -- "${CONTRACT_ROOT}"
          fi

'''


EMPIRICAL_INITIALIZATION = '''      - name: Initialize isolated runtime and evidence paths
        shell: bash
        run: |
          set -euo pipefail
          evidence_root="${RUNNER_TEMP}/deform360-calibration-source-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
          site_root="${RUNNER_TEMP}/deform360-calibration-site-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
          staged_root="${PROCESSED_ROOT}/staged-raw"
          {
            echo "EVIDENCE_ROOT=${evidence_root}"
            echo "PYTHON_SITE=${site_root}"
            echo "STAGED_ROOT=${staged_root}"
            echo "PYTHONPATH=${GITHUB_WORKSPACE}/src:${PROCESSING_REPO}:${site_root}${PYTHONPATH:+:${PYTHONPATH}}"
          } >> "${GITHUB_ENV}"
          rm -rf "${evidence_root}" "${site_root}"
          mkdir -p \
            "${DATA_ROOT}" \
            "${PROCESSED_ROOT}" \
            "${evidence_root}" \
            "${site_root}"
          printf '/_deform360_processing/\n' >> .git/info/exclude

          base_python=""
          for candidate in \
            /usr/bin/python3.12 \
            /usr/local/bin/python3.12 \
            /usr/bin/python3 \
            /usr/local/bin/python3
          do
            if [[ -x "${candidate}" ]]; then
              base_python="${candidate}"
              break
            fi
          done
          if [[ -z "${base_python}" ]]; then
            base_python="$(command -v python3 || true)"
          fi
          if [[ -z "${base_python}" ]]; then
            echo "No system Python is available on the self-hosted runner." >&2
            exit 1
          fi

          echo "BASE_PYTHON=${base_python}" >> "${GITHUB_ENV}"
          "${base_python}" --version | tee "${evidence_root}/base-python.txt"
          "${base_python}" -m pip --version | tee "${evidence_root}/base-pip.txt"

'''


EMPIRICAL_TARGET_SITE = '''      - name: Populate a fresh calibration-source target site
        shell: bash
        run: |
          set -euo pipefail
          "${BASE_PYTHON}" -m pip install \
            --break-system-packages \
            --no-cache-dir \
            --target "${PYTHON_SITE}" \
            ".[dev,graph]" \
            build \
            scipy \
            matplotlib \
            "${PROCESSING_REPO}[all]"
          "${BASE_PYTHON}" -m pip check
          "${BASE_PYTHON}" -m pip freeze --path "${PYTHON_SITE}" \
            > "${EVIDENCE_ROOT}/pip-freeze.txt"
          test -z "$(git -C "${PROCESSING_REPO}" status --porcelain)"

'''


TEST_SOURCE = '''"""Workflow contracts for the locked Deform360 calibration-source stage."""

from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(".github/workflows/deform360-official-hub-calibration-source.yml")


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_trusted_contracts_use_isolated_self_hosted_execution() -> None:
    text = _workflow_text()
    contracts = text.index("  contracts:")
    empirical = text.index("  prepare-calibration-source:")
    contract_block = text[contracts:empirical]
    empirical_block = text[empirical:]

    assert contracts < empirical
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in contract_block
    assert (
        "github.event.pull_request.head.repo.full_name == github.repository"
        in contract_block
    )
    assert "actions/setup-python" not in contract_block
    assert "${RUNNER_TEMP}/deform360-calibration-contracts-" in contract_block
    assert 'echo "CONTRACT_SITE=${contract_site}"' in contract_block
    assert '--target "${CONTRACT_SITE}"' in contract_block
    assert "--break-system-packages" in contract_block
    assert "--no-cache-dir" in contract_block
    assert " -m venv" not in contract_block
    assert "ensurepip" not in contract_block
    assert "DATA_ROOT:" not in contract_block
    assert "PROCESSED_ROOT:" not in contract_block
    assert "HF_TOKEN" not in contract_block
    assert (
        " scripts/science/run_deform360_official_hub_calibration_source.py plan "
        not in contract_block
    )
    assert "if: github.event_name != 'pull_request'" in empirical_block
    assert "needs: contracts" in empirical_block


def test_empirical_job_uses_a_fresh_runner_temp_target_site() -> None:
    text = _workflow_text()
    empirical = text[text.index("  prepare-calibration-source:") :]
    job_header = empirical[: empirical.index("    steps:")]

    assert "actions/setup-python" not in empirical
    assert "${{ runner.temp }}" not in job_header
    assert "${RUNNER_TEMP}/deform360-calibration-site-" in empirical
    assert 'echo "PYTHON_SITE=${site_root}"' in empirical
    assert '--target "${PYTHON_SITE}"' in empirical
    assert "--break-system-packages" in empirical
    assert "--no-cache-dir" in empirical
    assert " -m venv" not in empirical
    assert "ensurepip" not in empirical
    assert "VENV_ROOT" not in empirical


def test_failure_reporting_does_not_require_the_target_site() -> None:
    text = _workflow_text()
    confirmation = text[text.index("      - name: Verify the confirmation cohort") :]
    summary = confirmation[confirmation.index("      - name: Publish job summary") :]

    assert "${PYTHON_SITE}/" not in confirmation
    assert "${PYTHON_SITE}/" not in summary
    assert "if-no-files-found: warn" in confirmation
    assert (
        'for path in "${PYTHON_SITE:-}" "${PROCESSING_REPO:-}"'
        in summary
    )
'''


DOC_SECTION = '''## Workflow trust and runner-capacity boundary

The source-contract gate no longer depends on GitHub-hosted runner capacity. It
runs on `workstation2` for trusted pushes, manual dispatches, and pull requests
whose head branch belongs to this repository. Pull requests from forks are not
admitted to the self-hosted runner. The contract job checks out the exact
reviewed revision with read-only credentials and installs into a fresh isolated
`RUNNER_TEMP` target site without a package cache. It opens no dataset root or
payload.

The empirical job uses a separate fresh target site rather than the runner's
Python toolcache or `venv` support. Both target sites are removed after the job,
while raw and processed calibration data remain only in their registered
persistent roots.

The empirical preparation job still has the explicit
`github.event_name != 'pull_request'` guard. Therefore no pull request can run
the names-only planner, download calibration bytes, open camera or robot data,
or inspect a confirmation-object subtree. A merge that changes this registered
lane runs the self-hosted contract gate first and only then starts the locked
calibration-source preparation.

'''


def _replace_between(
    text: str,
    *,
    start_marker: str,
    end_marker: str,
    replacement: str,
) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[:start] + replacement + text[end:]


def patch_workflow() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    workflow = _replace_between(
        workflow,
        start_marker="  contracts:\n",
        end_marker="  prepare-calibration-source:\n",
        replacement=CONTRACT_JOB,
    )
    workflow = _replace_between(
        workflow,
        start_marker="      - name: Initialize isolated runtime and evidence paths\n",
        end_marker="      - name: Record runner and source identity\n",
        replacement=EMPIRICAL_INITIALIZATION,
    )
    workflow = _replace_between(
        workflow,
        start_marker="      - name: Create a fresh calibration-source environment\n",
        end_marker="      - name: Validate the exact source before dataset access\n",
        replacement=EMPIRICAL_TARGET_SITE,
    )
    workflow = workflow.replace(
        '"${VENV_ROOT}/bin/python"',
        '"${BASE_PYTHON}"',
    )
    cleanup_before = (
        '          for path in "${VENV_ROOT:-}" '
        '"${PROCESSING_REPO:-}"; do\n'
    )
    cleanup_after = (
        '          for path in "${PYTHON_SITE:-}" '
        '"${PROCESSING_REPO:-}"; do\n'
    )
    if cleanup_before not in workflow:
        raise RuntimeError("empirical cleanup block changed unexpectedly")
    workflow = workflow.replace(cleanup_before, cleanup_after, 1)
    if "VENV_ROOT" in workflow:
        raise RuntimeError("obsolete VENV_ROOT remains in workflow")
    WORKFLOW_PATH.write_text(workflow, encoding="utf-8")


def patch_test() -> None:
    TEST_PATH.write_text(TEST_SOURCE, encoding="utf-8")


def patch_documentation() -> None:
    documentation = DOC_PATH.read_text(encoding="utf-8")
    marker = "## Persistent and uploaded data\n"
    if marker not in documentation:
        raise RuntimeError("documentation insertion marker is missing")
    if "## Workflow trust and runner-capacity boundary" not in documentation:
        documentation = documentation.replace(marker, DOC_SECTION + marker, 1)
    DOC_PATH.write_text(documentation, encoding="utf-8")


def main() -> None:
    patch_workflow()
    patch_test()
    patch_documentation()


if __name__ == "__main__":
    main()
