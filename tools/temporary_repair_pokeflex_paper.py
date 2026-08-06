#!/usr/bin/env python3
"""Apply the temporary deterministic repair for PokeFlex paper-artifact CI."""

from __future__ import annotations

from pathlib import Path


def _replace_once(text: str, old: str, new: str, *, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"unexpected {name}: expected one occurrence, found {count}")
    return text.replace(old, new, 1)


def repair_artifact_workflow() -> None:
    path = Path(".github/workflows/pokeflex-same-object-paper-artifacts.yml")
    text = path.read_text(encoding="utf-8")
    marker = "\n  artifacts:\n"
    if text.count(marker) != 1:
        raise SystemExit("unexpected paper-artifact job boundary")
    prefix, artifacts = text.split(marker, 1)

    mkdir_line = (
        '          mkdir -p "${root}/evidence" "${root}/output" '
        '"${root}/resolved"\n'
    )
    runtime_selection = mkdir_line + '''          base_python="${VAR_PYTHON:-}"
          if [[ -n "${base_python}" && ! -x "${base_python}" ]]; then
            echo "Configured POKEFLEX_PYTHON is not executable: ${base_python}" >&2
            exit 1
          fi
          if [[ -z "${base_python}" ]]; then
            for candidate in \
              /usr/bin/python3.11 \
              /usr/local/bin/python3.11 \
              /usr/bin/python3 \
              /usr/local/bin/python3
            do
              if [[ -x "${candidate}" ]]; then
                base_python="${candidate}"
                break
              fi
            done
          fi
          if [[ -z "${base_python}" ]]; then
            base_python="$(command -v python3 || true)"
          fi
          if [[ -z "${base_python}" ]]; then
            echo "No system Python is available on workstation2." >&2
            exit 1
          fi
'''
    artifacts = _replace_once(
        artifacts,
        mkdir_line,
        runtime_selection,
        name="artifact-path initialization block",
    )

    env_line = '            echo "POKEFLEX_RUN_ROOT=${root}"\n'
    env_replacement = env_line + '''            echo "POKEFLEX_VENV=${root}/venv"
            echo "BPT_BASE_PYTHON=${base_python}"
            echo "BPT_PYTHON=${root}/venv/bin/python"
'''
    artifacts = _replace_once(
        artifacts,
        env_line,
        env_replacement,
        name="artifact environment block",
    )

    old_setup = '''      - name: Set up the released-checkpoint Python
        uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0
        with:
          python-version: "3.11"
          cache: pip
          cache-dependency-path: pyproject.toml

'''
    new_setup = '''      - name: Create an isolated released-checkpoint Python
        shell: bash
        run: |
          set -euo pipefail
          "${BPT_BASE_PYTHON}" --version
          "${BPT_BASE_PYTHON}" -m venv --copies "${POKEFLEX_VENV}"
          "${BPT_PYTHON}" -m ensurepip --upgrade
          "${BPT_PYTHON}" -m pip install --no-cache-dir \
            --upgrade pip setuptools wheel
          "${BPT_PYTHON}" -m pip --version

'''
    artifacts = _replace_once(
        artifacts,
        old_setup,
        new_setup,
        name="persistent setup-python block",
    )

    replacements = (
        (
            "          python -m pip install --upgrade pip\n",
            '          "${BPT_PYTHON}" -m pip install --no-cache-dir '
            "--upgrade pip\n",
            "runtime pip upgrade",
        ),
        (
            "          python -m pip install \\\n"
            "            \"torch==2.1.0\" \\\n",
            '          "${BPT_PYTHON}" -m pip install --no-cache-dir \\\n'
            "            \"torch==2.1.0\" \\\n",
            "CUDA PyTorch installation",
        ),
        (
            "          python -m pip install \\\n"
            "            -e \".[graph,vision]\" \\\n",
            '          "${BPT_PYTHON}" -m pip install --no-cache-dir \\\n'
            "            -e \".[graph,vision]\" \\\n",
            "paper runtime installation",
        ),
        (
            "          python -m pip check\n",
            '          "${BPT_PYTHON}" -m pip check\n',
            "runtime dependency check",
        ),
        (
            '          echo "BPT_PYTHON=$(command -v python)" >> '
            '"${GITHUB_ENV}"\n',
            "",
            "persistent interpreter export",
        ),
        (
            "          python - <<'PY'\n",
            '          "${BPT_PYTHON}" - <<\'PY\'\n',
            "runtime import smoke",
        ),
    )
    for old, new, name in replacements:
        artifacts = _replace_once(artifacts, old, new, name=name)

    repaired = prefix + marker + artifacts
    artifact_job = repaired.split(marker, 1)[1]
    if "actions/setup-python" in artifact_job:
        raise SystemExit("self-hosted artifact job still uses setup-python")
    for term in (
        "BPT_BASE_PYTHON=${base_python}",
        "BPT_PYTHON=${root}/venv/bin/python",
        '-m venv --copies "${POKEFLEX_VENV}"',
        "pip install --no-cache-dir",
    ):
        if term not in repaired:
            raise SystemExit(f"missing isolated-runtime term: {term}")
    path.write_text(repaired, encoding="utf-8")


def repair_core_coverage() -> None:
    path = Path(".github/workflows/tests.yml")
    text = path.read_text(encoding="utf-8")
    anchor = "            tests/test_pokeflex_public_evaluation_cli.py \\\n"
    reporting = "            tests/test_pokeflex_same_object_reporting.py \\\n"
    anchor_count = text.count(anchor)
    reporting_count = text.count(reporting)
    if reporting_count == 0:
        if anchor_count != 2:
            raise SystemExit(
                f"expected two stable/core PokeFlex anchors, found {anchor_count}"
            )
        text = text.replace(anchor, anchor + reporting)
    elif reporting_count != anchor_count:
        raise SystemExit(
            "PokeFlex reporting test is only partially integrated into core CI"
        )
    if text.count(reporting) != 2:
        raise SystemExit("PokeFlex reporting test must run in both core lists")
    path.write_text(text, encoding="utf-8")


def main() -> int:
    repair_artifact_workflow()
    repair_core_coverage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
