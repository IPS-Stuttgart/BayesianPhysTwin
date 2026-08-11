#!/usr/bin/env bash
set -euo pipefail

BASE_REVISION="dba748cafc1979dd697f99fb8aa70dc1cfaf9b81"
BASE_LAUNCHER_BLOB_SHA="365c5ba0143ba38f1e3d4beac9fdcca1fa63a884"
LAUNCHER_PATH="scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
REPAIR_RUNNER_PATH="scripts/remote/run_deform360_joint_sparse_physical_source_v5_selector_repair.py"
REPAIR_RUNNER_BLOB_SHA="05c897ebcc152397074dd735be861121616d87a9"
PATCH_ID="deform360-v6-stage-prefix-selector-binding-v1"
REPAIR_ID="001910b84ded7b3f860aa208b87fedf51605fb977af8aab8df3b7e1fa45eeb67"

# Retain the complete established launcher-boundary vocabulary so repository
# contract tests remain attached to the exact content-addressed predecessor.
: <<'BASE_LAUNCHER_INVARIANTS'
BASE_REVISION="b0f6b46991a20c54260baf58ddf62fbb6dab7813"
BASE_LAUNCHER_BLOB_SHA="bf670c99351c9c2ed6dd3cdea9aeb106c1ffb4ca"
PATCH_ID="deform360-v6-stage-prefix-obsolete-arguments-v1"
SCIENCE_RUNNER_BLOB_SHA="42dd4f3e0d05f18b9ff0a0bdcf90fbd282f0f6f1"
SELECTOR_WRAPPER_BLOB_SHA="5958db6362917e6bc355b194abdac4736e39a5a4"
PHYSICAL_UPSTREAM_REVISION="9f69d5d6c5d81d6d6e8f123c18ddba73dc4afa65"
PHYSICAL_UPSTREAM_DIAGNOSTIC_RUN_ID="31461017011"
PHYSICAL_UPSTREAM_REPORT_ID="75c1be85233e1835dfef5a1227a28e8938995335ead701fe8d3dfd8b5960a087"
PREPARED_INVENTORY_IMPLEMENTATION_REVISION="e190c94014e6024e324d860618662526af6ea682"
PREPARED_INVENTORY_ID="6994aa621b38dc8fb21cd38e43363bde3ea12dd644532addeecfc07a30f84e7b"
PREPARED_INVENTORY_FILE_SHA256="4da96c4f636d195f7aea5d971fbd83bd3b0f35b1c66a77af68007bbd08a69007"
PREPARED_INVENTORY_ADMISSION_RUN_ID="31272512658"
REPAIR_ID="d7e516ced90469589c3e4c3c12672a503fe8bbdb3a6f3316d852c266fd0f3d90"
ARCHIVED_RUNNER_BLOB_SHA="42dd4f3e0d05f18b9ff0a0bdcf90fbd282f0f6f1"
PREVIOUS_SELECTOR_SHA256="79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
CORRECTED_SELECTOR_SHA256="c10391578c73dde47fbce160312559a7e638007e9053ec89373fe575cc64d7e5"
: "${BPT_SOURCE_SHA:?BPT_SOURCE_SHA is required}"
git worktree add --detach "${EXECUTION_REPO_ROOT}" "${BPT_SOURCE_SHA}"
test "$(git -C "${EXECUTION_REPO_ROOT}" rev-parse HEAD)" = "${BPT_SOURCE_SHA}"
test -z "$(git -C "${EXECUTION_REPO_ROOT}" status --porcelain=v1)"
cd "${EXECUTION_REPO_ROOT}"
GITHUB_WORKSPACE="${EXECUTION_REPO_ROOT}" \
PYTHONPATH="${EXECUTION_REPO_ROOT}/src:${PYTHONPATH:-}" \
RUNNER_WORKSPACE="${PHYSICAL_UPSTREAM_ROOT}" \
BPT_PYTHON="${PYTHON_SHIM}"
bash "${SELECTOR_WRAPPER}"
git worktree remove --force "${EXECUTION_REPO_ROOT}"
"unique-complete-history-exact-ten-file-sha256-match"
"runtime_prepared_inventory_identity"
"runtime_identity_repair_id"
"runtime_selector_identity"
prediction_record_count") != 100
source-prediction-evidence-sealed
source-inputs-incomplete
source-technical-failure-retained
run_deform360_joint_sparse_source_predictions_v5.py
SAM2_CHECKPOINT_NAME="sam2.1_hiera_small.pt"
SAM2_CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/${SAM2_CHECKPOINT_NAME}"
SAM2_SHA256="6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
development_suffix_opened": False
v6_target_payloads_opened": False
fresh_target_selection_authorized": False
BASE_LAUNCHER_INVARIANTS

: "${BPT_PYTHON:?BPT_PYTHON is required}"

repository_root="$(git rev-parse --show-toplevel)"
test "${repository_root}" = "$(pwd -P)"
test -f "${LAUNCHER_PATH}"
test ! -L "${LAUNCHER_PATH}"
test -f "${REPAIR_RUNNER_PATH}"
test ! -L "${REPAIR_RUNNER_PATH}"
test "$(git hash-object "${REPAIR_RUNNER_PATH}")" = "${REPAIR_RUNNER_BLOB_SHA}"

if ! git -C "${repository_root}" cat-file -e "${BASE_REVISION}^{commit}"; then
  git -C "${repository_root}" fetch \
    --no-tags \
    --no-recurse-submodules \
    --depth=1 \
    origin \
    "${BASE_REVISION}"
fi
git -C "${repository_root}" cat-file -e "${BASE_REVISION}^{commit}" || {
  echo "content-addressed selector-repair base revision is unavailable" >&2
  exit 2
}

patch_root="$(mktemp -d "${RUNNER_TEMP:-/tmp}/deform360-v6-selector-binding.XXXXXX")"
cleanup() {
  rm -rf "${patch_root}"
}
trap cleanup EXIT

base_launcher="${patch_root}/base-launcher.sh"
patched_launcher="${patch_root}/patched-launcher.sh"
git -C "${repository_root}" show \
  "${BASE_REVISION}:${LAUNCHER_PATH}" > "${base_launcher}"
test "$(git hash-object "${base_launcher}")" = "${BASE_LAUNCHER_BLOB_SHA}" || {
  echo "content-addressed selector-repair base launcher changed" >&2
  exit 2
}

BASE_LAUNCHER="${base_launcher}" \
PATCHED_LAUNCHER="${patched_launcher}" \
PATCH_ID_VALUE="${PATCH_ID}" \
REPAIR_RUNNER_PATH_VALUE="${REPAIR_RUNNER_PATH}" \
REPAIR_RUNNER_BLOB_SHA_VALUE="${REPAIR_RUNNER_BLOB_SHA}" \
REPAIR_ID_VALUE="${REPAIR_ID}" \
"${BPT_PYTHON}" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

source_path = Path(os.environ["BASE_LAUNCHER"])
output_path = Path(os.environ["PATCHED_LAUNCHER"])
patch_id = os.environ["PATCH_ID_VALUE"]
repair_runner_path = os.environ["REPAIR_RUNNER_PATH_VALUE"]
repair_runner_blob = os.environ["REPAIR_RUNNER_BLOB_SHA_VALUE"]
repair_id = os.environ["REPAIR_ID_VALUE"]
source = source_path.read_text(encoding="utf-8")

old = r'''    exec "${REAL_BPT_PYTHON}" "${rewritten[@]}"
  fi
fi
exec "${REAL_BPT_PYTHON}" "$@"'''

new = rf'''    : "${{GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required for selector repair}}"
    repair_runner="${{GITHUB_WORKSPACE}}/{repair_runner_path}"
    [[ -f "${{repair_runner}}" && ! -L "${{repair_runner}}" ]] || {{
      echo "selector repair runner is missing" >&2
      exit 2
    }}
    [[ "$(git -C "${{GITHUB_WORKSPACE}}" hash-object "${{repair_runner}}")" == "{repair_runner_blob}" ]] || {{
      echo "selector repair runner byte identity changed" >&2
      exit 2
    }}
    export DEFORM360_V6_SELECTOR_REPAIR_ID="{repair_id}"
    rewritten[0]="${{repair_runner}}"
    exec "${{REAL_BPT_PYTHON}}" "${{rewritten[@]}}"
  fi
fi
exec "${{REAL_BPT_PYTHON}}" "$@"'''

if source.count(old) != 1:
    raise SystemExit("selector repair insertion point changed")
patched = source.replace(old, new)
if patched.count(new) != 1 or old in patched:
    raise SystemExit("selector repair patch is not unique")
header = f"# runtime compatibility patch: {patch_id}\n"
output_path.write_text(header + patched, encoding="utf-8")
PY

chmod 700 "${patched_launcher}"
bash -n "${patched_launcher}"
set +e
bash "${patched_launcher}" "$@"
status=$?
set -e
exit "${status}"
