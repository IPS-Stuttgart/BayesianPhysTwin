#!/usr/bin/env bash
set -euo pipefail

BASE_REVISION="dba748cafc1979dd697f99fb8aa70dc1cfaf9b81"
BASE_LAUNCHER_BLOB_SHA="365c5ba0143ba38f1e3d4beac9fdcca1fa63a884"
LAUNCHER_PATH="scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
PATCH_ID="deform360-v6-stage-selector-consumer-identity-v1"
STAGE_SELECTOR_REPAIR_ID="aea2506a8c648fcbaad460ae6eb0311801466015268271c5492bac9a6e1d2bae"
STAGE_SELECTOR_REPAIR_PATH="protocols/amendments/deform360_official_hub_fresh_object_session_v6_stage_selector_identity_repair.json"
STAGE_SELECTOR_REPAIR_SHA256="02ef431ee5900c2dfa06e9a5031aeb5ea88659345dbd1829b22ed6c065289134"
STAGE_SELECTOR_HELPER_PATH="scripts/remote/run_deform360_v6_stage_selector_identity_repair.py"
STAGE_SELECTOR_HELPER_SHA256="ac8a1db996c83cc219e3bb5321b045179769edf6afc4c3460c2e278924a20fe1"
STAGE_SELECTOR_CONSUMER_PATH="scripts/remote/stage_deform360_bias_aware_prediction_prefix.py"
STAGE_SELECTOR_CONSUMER_SHA256="a90578e8a83e5a72388b86f25c6b7b9dee872b75e2919c352e3a3a3ea431e5d6"
STAGE_SELECTOR_PREVIOUS_SHA256="79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
STAGE_SELECTOR_CORRECTED_SHA256="c10391578c73dde47fbce160312559a7e638007e9053ec89373fe575cc64d7e5"

# Preserve the complete reviewed launcher and its already-validated repairs by
# exact Git blob identity. The executable path below changes one uniquely
# matched runtime-shim literal and leaves all checksum-bound scientific files
# untouched.
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
["git", "-C", str(repository), "show"
"fetch",
"--depth=1",
"origin",
            revision,
target="scripts/science/inventory_deform360_calibration_prepared_source.py"
inventory_pattern = re.compile(
len(inventory_pattern.findall(runner)) != 1
rewritten+=("$1" "${PREPARED_INVENTORY_IMPLEMENTATION_REVISION}")
if [[ "${replacements}" -ne 1 ]]
exec "${REAL_BPT_PYTHON}" "$@"
"runtime_prepared_inventory_identity"
text.count(old) != 1
patched.count(new) != 1
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
if [[ "${1:-}" != "--materialize-physical-upstream" ]]; then
  : "${EVIDENCE_ROOT:?EVIDENCE_ROOT is required}"
fi

repository_root="$(git rev-parse --show-toplevel)"
test "${repository_root}" = "$(pwd -P)"
for path in \
  "${LAUNCHER_PATH}" \
  "${STAGE_SELECTOR_REPAIR_PATH}" \
  "${STAGE_SELECTOR_HELPER_PATH}" \
  "${STAGE_SELECTOR_CONSUMER_PATH}"
do
  test -f "${path}"
  test ! -L "${path}"
done

test "$(sha256sum "${STAGE_SELECTOR_REPAIR_PATH}" | awk '{print $1}')" \
  = "${STAGE_SELECTOR_REPAIR_SHA256}" || {
  echo "stage selector repair bytes changed" >&2
  exit 2
}
test "$(sha256sum "${STAGE_SELECTOR_HELPER_PATH}" | awk '{print $1}')" \
  = "${STAGE_SELECTOR_HELPER_SHA256}" || {
  echo "stage selector helper bytes changed" >&2
  exit 2
}
test "$(sha256sum "${STAGE_SELECTOR_CONSUMER_PATH}" | awk '{print $1}')" \
  = "${STAGE_SELECTOR_CONSUMER_SHA256}" || {
  echo "stage selector consumer bytes changed" >&2
  exit 2
}

if ! git -C "${repository_root}" cat-file -e "${BASE_REVISION}^{commit}"; then
  git -C "${repository_root}" fetch \
    --no-tags \
    --no-recurse-submodules \
    --depth=1 \
    origin \
    "${BASE_REVISION}"
fi
git -C "${repository_root}" cat-file -e "${BASE_REVISION}^{commit}" || {
  echo "content-addressed predecessor launcher revision is unavailable" >&2
  exit 2
}

patch_root="$(mktemp -d "${RUNNER_TEMP:-/tmp}/deform360-v6-selector-consumer-patch.XXXXXX")"
activation_marker="${patch_root}/stage-selector-activation.json"
cleanup() {
  rm -rf "${patch_root}"
}
trap cleanup EXIT

base_launcher="${patch_root}/base-launcher.sh"
patched_launcher="${patch_root}/patched-launcher.sh"
git -C "${repository_root}" show \
  "${BASE_REVISION}:${LAUNCHER_PATH}" > "${base_launcher}"
test "$(git hash-object "${base_launcher}")" = "${BASE_LAUNCHER_BLOB_SHA}" || {
  echo "content-addressed predecessor launcher byte identity changed" >&2
  exit 2
}

BASE_LAUNCHER="${base_launcher}" \
PATCHED_LAUNCHER="${patched_launcher}" \
PATCH_ID_VALUE="${PATCH_ID}" \
"${BPT_PYTHON}" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

source_path = Path(os.environ["BASE_LAUNCHER"])
output_path = Path(os.environ["PATCHED_LAUNCHER"])
patch_id = os.environ["PATCH_ID_VALUE"]
source = source_path.read_text(encoding="utf-8")

old = r'''inventory_target="scripts/science/inventory_deform360_calibration_prepared_source.py"
physical_target="scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"
if [[ "${1:-}" == "${inventory_target}" ]]; then
  rewritten=()
  replacements=0
  while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == "--implementation-revision" ]]; then
      [[ "$#" -ge 2 ]] || {
        echo "prepared inventory implementation revision lacks a value" >&2
        exit 2
      }
      rewritten+=("$1" "${PREPARED_INVENTORY_IMPLEMENTATION_REVISION}")
      replacements=$((replacements + 1))
      shift 2
    else
      rewritten+=("$1")
      shift
    fi
  done
  if [[ "${replacements}" -ne 1 ]]; then
    echo "prepared inventory implementation binding is not unique" >&2
    exit 2
  fi
  exec "${REAL_BPT_PYTHON}" "${rewritten[@]}"
fi
if [[ "${1:-}" == "${physical_target}" ]]; then
  arguments=("$@")
  stage_count=0
  stage_value=""
  for ((index = 0; index < ${#arguments[@]}; index++)); do
    if [[ "${arguments[index]}" == "--stage" ]]; then
      ((index + 1 < ${#arguments[@]})) || {
        echo "physical source stage lacks a value" >&2
        exit 2
      }
      stage_count=$((stage_count + 1))
      stage_value="${arguments[index + 1]}"
    fi
  done
  if [[ "${stage_count}" -ne 1 ]]; then
    echo "physical source stage binding is not unique" >&2
    exit 2
  fi
  if [[ "${stage_value}" == "stage-prefix" ]]; then
    : "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required for stage-prefix repair}"
    rewritten=()
    repo_replacements=0
    role_replacements=0
    while [[ "$#" -gt 0 ]]; do
      case "$1" in
        --repo)
          [[ "$#" -ge 2 ]] || {
            echo "stage-prefix compatibility repo lacks a value" >&2
            exit 2
          }
          [[ "$2" == "${GITHUB_WORKSPACE}" ]] || {
            echo "stage-prefix compatibility repo differs from the exact worktree" >&2
            exit 2
          }
          repo_replacements=$((repo_replacements + 1))
          shift 2
          ;;
        --role)
          [[ "$#" -ge 2 ]] || {
            echo "stage-prefix compatibility role lacks a value" >&2
            exit 2
          }
          [[ "$2" == "calibration" ]] || {
            echo "stage-prefix compatibility role must remain calibration" >&2
            exit 2
          }
          role_replacements=$((role_replacements + 1))
          shift 2
          ;;
        *)
          rewritten+=("$1")
          shift
          ;;
      esac
    done
    if [[ "${repo_replacements}" -ne 1 || "${role_replacements}" -ne 1 ]]; then
      echo "stage-prefix compatibility bindings are not unique" >&2
      exit 2
    fi
    exec "${REAL_BPT_PYTHON}" "${rewritten[@]}"
  fi
fi
exec "${REAL_BPT_PYTHON}" "$@"'''

new = r'''inventory_target="scripts/science/inventory_deform360_calibration_prepared_source.py"
physical_target="scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"
if [[ "${1:-}" == "${inventory_target}" ]]; then
  rewritten=()
  replacements=0
  while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == "--implementation-revision" ]]; then
      [[ "$#" -ge 2 ]] || {
        echo "prepared inventory implementation revision lacks a value" >&2
        exit 2
      }
      rewritten+=("$1" "${PREPARED_INVENTORY_IMPLEMENTATION_REVISION}")
      replacements=$((replacements + 1))
      shift 2
    else
      rewritten+=("$1")
      shift
    fi
  done
  if [[ "${replacements}" -ne 1 ]]; then
    echo "prepared inventory implementation binding is not unique" >&2
    exit 2
  fi
  exec "${REAL_BPT_PYTHON}" "${rewritten[@]}"
fi
if [[ "${1:-}" == "${physical_target}" ]]; then
  arguments=("$@")
  stage_count=0
  stage_value=""
  for ((index = 0; index < ${#arguments[@]}; index++)); do
    if [[ "${arguments[index]}" == "--stage" ]]; then
      ((index + 1 < ${#arguments[@]})) || {
        echo "physical source stage lacks a value" >&2
        exit 2
      }
      stage_count=$((stage_count + 1))
      stage_value="${arguments[index + 1]}"
    fi
  done
  if [[ "${stage_count}" -ne 1 ]]; then
    echo "physical source stage binding is not unique" >&2
    exit 2
  fi
  if [[ "${stage_value}" == "stage-prefix" ]]; then
    : "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required for stage-prefix repair}"
    rewritten=()
    repo_replacements=0
    role_replacements=0
    while [[ "$#" -gt 0 ]]; do
      case "$1" in
        --repo)
          [[ "$#" -ge 2 ]] || {
            echo "stage-prefix compatibility repo lacks a value" >&2
            exit 2
          }
          [[ "$2" == "${GITHUB_WORKSPACE}" ]] || {
            echo "stage-prefix compatibility repo differs from the exact worktree" >&2
            exit 2
          }
          repo_replacements=$((repo_replacements + 1))
          shift 2
          ;;
        --role)
          [[ "$#" -ge 2 ]] || {
            echo "stage-prefix compatibility role lacks a value" >&2
            exit 2
          }
          [[ "$2" == "calibration" ]] || {
            echo "stage-prefix compatibility role must remain calibration" >&2
            exit 2
          }
          role_replacements=$((role_replacements + 1))
          shift 2
          ;;
        *)
          rewritten+=("$1")
          shift
          ;;
      esac
    done
    if [[ "${repo_replacements}" -ne 1 || "${role_replacements}" -ne 1 ]]; then
      echo "stage-prefix compatibility bindings are not unique" >&2
      exit 2
    fi
    : "${DEFORM360_V6_STAGE_SELECTOR_REPAIR_PATH:?stage selector repair path is required}"
    : "${DEFORM360_V6_STAGE_SELECTOR_HELPER_PATH:?stage selector helper path is required}"
    : "${DEFORM360_V6_STAGE_SELECTOR_ACTIVATION_MARKER:?stage selector marker is required}"
    helper=(
      "${DEFORM360_V6_STAGE_SELECTOR_HELPER_PATH}"
      "--runtime-repair"
      "${DEFORM360_V6_STAGE_SELECTOR_REPAIR_PATH}"
      "--activation-marker"
      "${DEFORM360_V6_STAGE_SELECTOR_ACTIVATION_MARKER}"
    )
    helper+=("${rewritten[@]:1}")
    exec "${REAL_BPT_PYTHON}" "${helper[@]}"
  fi
fi
exec "${REAL_BPT_PYTHON}" "$@"'''

if source.count(old) != 1:
    raise SystemExit("stage selector consumer patch source changed")
patched = source.replace(old, new)
if patched.count(new) != 1 or old in patched:
    raise SystemExit("stage selector consumer patch is not unique")
header = f"# runtime compatibility patch: {patch_id}\n"
output_path.write_text(header + patched, encoding="utf-8")
PY

chmod 700 "${patched_launcher}"
bash -n "${patched_launcher}"
set +e
DEFORM360_V6_STAGE_SELECTOR_REPAIR_PATH="${STAGE_SELECTOR_REPAIR_PATH}" \
DEFORM360_V6_STAGE_SELECTOR_HELPER_PATH="${STAGE_SELECTOR_HELPER_PATH}" \
DEFORM360_V6_STAGE_SELECTOR_ACTIVATION_MARKER="${activation_marker}" \
  bash "${patched_launcher}" "$@"
status=$?
set -e

if [[ "${1:-}" == "--materialize-physical-upstream" ]]; then
  exit "${status}"
fi

receipt="${EVIDENCE_ROOT}/deform360-v6-source-prediction-evidence/execution-receipt.json"
if [[ -f "${receipt}" && ! -L "${receipt}" ]]; then
  export RECEIPT_PATH="${receipt}"
  export STAGE_SELECTOR_REPAIR_ID STAGE_SELECTOR_REPAIR_PATH
  export STAGE_SELECTOR_REPAIR_SHA256 STAGE_SELECTOR_HELPER_PATH
  export STAGE_SELECTOR_HELPER_SHA256 STAGE_SELECTOR_CONSUMER_PATH
  export STAGE_SELECTOR_CONSUMER_SHA256 STAGE_SELECTOR_PREVIOUS_SHA256
  export STAGE_SELECTOR_CORRECTED_SHA256
  export STAGE_SELECTOR_ACTIVATION_MARKER="${activation_marker}"
  receipt_python="${BPT_PYTHON}"
  "${receipt_python}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

path = Path(os.environ["RECEIPT_PATH"])
receipt = json.loads(path.read_text(encoding="utf-8"))
receipt.pop("receipt_id", None)
marker_path = Path(os.environ["STAGE_SELECTOR_ACTIVATION_MARKER"])
activated = marker_path.is_file() and not marker_path.is_symlink()
if activated:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected_marker = {
        "application": "process-local-loaded-module-constant-only",
        "consumer_file_sha256": os.environ["STAGE_SELECTOR_CONSUMER_SHA256"],
        "corrected_expected_sha256": os.environ[
            "STAGE_SELECTOR_CORRECTED_SHA256"
        ],
        "previous_expected_sha256": os.environ[
            "STAGE_SELECTOR_PREVIOUS_SHA256"
        ],
        "repair_id": os.environ["STAGE_SELECTOR_REPAIR_ID"],
        "selector_file_sha256": os.environ[
            "STAGE_SELECTOR_CORRECTED_SHA256"
        ],
    }
    if marker != expected_marker:
        raise SystemExit("stage selector activation marker changed")
receipt["runtime_stage_selector_consumer_identity_repair"] = {
    "activated": activated,
    "application": "process-local-loaded-module-constant-only",
    "consumer_file_sha256": os.environ["STAGE_SELECTOR_CONSUMER_SHA256"],
    "consumer_path": os.environ["STAGE_SELECTOR_CONSUMER_PATH"],
    "corrected_expected_sha256": os.environ[
        "STAGE_SELECTOR_CORRECTED_SHA256"
    ],
    "helper_file_sha256": os.environ["STAGE_SELECTOR_HELPER_SHA256"],
    "helper_path": os.environ["STAGE_SELECTOR_HELPER_PATH"],
    "previous_expected_sha256": os.environ[
        "STAGE_SELECTOR_PREVIOUS_SHA256"
    ],
    "repair_file_sha256": os.environ["STAGE_SELECTOR_REPAIR_SHA256"],
    "repair_id": os.environ["STAGE_SELECTOR_REPAIR_ID"],
    "repair_path": os.environ["STAGE_SELECTOR_REPAIR_PATH"],
    "selector_file_sha256": os.environ["STAGE_SELECTOR_CORRECTED_SHA256"],
}
canonical = json.dumps(
    receipt,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
receipt["receipt_id"] = hashlib.sha256(canonical).hexdigest()
path.write_text(
    json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
PY
  compact="$(dirname "${receipt}")"
  (
    cd "${compact}"
    rm -f SHA256SUMS
    find . -type f ! -name SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 sha256sum > SHA256SUMS
    sha256sum --check SHA256SUMS >/dev/null
  )
fi

exit "${status}"
