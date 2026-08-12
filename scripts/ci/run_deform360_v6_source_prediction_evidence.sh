#!/usr/bin/env bash
set -euo pipefail

readonly BASE_REVISION="812da43f993b4fc5e1f6a96bcc308756b131fc4c"
readonly BASE_LAUNCHER_BLOB_SHA="b2b2307a2f89f3983cce349e1220033bf7f8f50c"
readonly LAUNCHER_PATH="scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
readonly SOURCE_PLAN_ENVIRONMENT_REPAIR_ID=\
"65096d1d4e8903eeacef0fc50816e47752a61e0d1bb4b6601f291bfcffb9ac4e"
readonly SOURCE_PLAN_ENVIRONMENT_REPAIR_PATH="protocols/amendments/"\
"deform360_official_hub_fresh_object_session_v6_source_plan_environment.json"
readonly SOURCE_PLAN_ENVIRONMENT_REPAIR_SHA256=\
"1eda9a28e17e46756f9f4bf4fc341b920a8c6f6de8d3e492be1b035a6368651d"

: "${BPT_SOURCE_SHA:?BPT_SOURCE_SHA is required}"
if [[ "${1:-}" != "--materialize-physical-upstream" ]]; then
  : "${RESULTS_ROOT:?RESULTS_ROOT is required}"
  : "${AMENDMENT_ID:?AMENDMENT_ID is required}"
  : "${EVIDENCE_ROOT:?EVIDENCE_ROOT is required}"
fi

repository_root="$(git rev-parse --show-toplevel)"
[[ "${repository_root}" == "$(pwd -P)" ]] || {
  echo "source-plan environment repair must run from the repository root" >&2
  exit 2
}
for path in "${LAUNCHER_PATH}" "${SOURCE_PLAN_ENVIRONMENT_REPAIR_PATH}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || {
    echo "source-plan environment repair input is unavailable: ${path}" >&2
    exit 2
  }
done
[[ "$(sha256sum "${SOURCE_PLAN_ENVIRONMENT_REPAIR_PATH}" | awk '{print $1}')" \
  == "${SOURCE_PLAN_ENVIRONMENT_REPAIR_SHA256}" ]] || {
  echo "source-plan environment repair bytes changed" >&2
  exit 2
}

if ! git cat-file -e "${BASE_REVISION}^{commit}"; then
  git fetch --no-tags --no-recurse-submodules --depth=1 origin "${BASE_REVISION}"
fi
git cat-file -e "${BASE_REVISION}^{commit}" || {
  echo "source-plan predecessor revision is unavailable" >&2
  exit 2
}

repair_root="$(
  mktemp -d "${RUNNER_TEMP:-/tmp}/deform360-v6-source-plan-env.XXXXXX"
)"
cleanup() {
  rm -rf "${repair_root}"
}
trap cleanup EXIT

base_launcher="${repair_root}/predecessor-launcher.sh"
git show "${BASE_REVISION}:${LAUNCHER_PATH}" > "${base_launcher}"
[[ "$(git hash-object "${base_launcher}")" == "${BASE_LAUNCHER_BLOB_SHA}" ]] || {
  echo "source-plan predecessor launcher byte identity changed" >&2
  exit 2
}
chmod 700 "${base_launcher}"
bash -n "${base_launcher}"

run_root_defaulted=false
legacy_defaults=()
export_default() {
  local name="$1"
  local value="$2"
  if declare -p "${name}" >/dev/null 2>&1; then
    export "${name}"
    return
  fi
  printf -v "${name}" '%s' "${value}"
  export "${name}"
  legacy_defaults+=("${name}")
}

if [[ "${1:-}" != "--materialize-physical-upstream" ]]; then
  [[ "${RESULTS_ROOT}" == /* && "${RESULTS_ROOT}" != */ ]] || {
    echo "RESULTS_ROOT must be an absolute canonical directory path" >&2
    exit 2
  }
  [[ "${AMENDMENT_ID}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "AMENDMENT_ID must be a lowercase SHA-256 digest" >&2
    exit 2
  }
  [[ "${BPT_SOURCE_SHA}" =~ ^[0-9a-f]{40}$ ]] || {
    echo "BPT_SOURCE_SHA must be a lowercase Git commit SHA" >&2
    exit 2
  }
  expected_run_root="${RESULTS_ROOT}/bayesian-phystwin/"\
"deform360-v6-source-prediction/${AMENDMENT_ID}/${BPT_SOURCE_SHA}"
  if declare -p RUN_ROOT >/dev/null 2>&1; then
    [[ "${RUN_ROOT}" == "${expected_run_root}" ]] || {
      echo "RUN_ROOT differs from the deterministic source-plan path" >&2
      exit 2
    }
  else
    RUN_ROOT="${expected_run_root}"
    run_root_defaulted=true
  fi
  export RUN_ROOT

  # The predecessor's generic receipt enrichment predates the dual-runtime
  # workflow. Supply explicit non-observed sentinels only when those legacy
  # variables are absent, then remove the legacy runtime fields before the
  # final receipt is content-addressed.
  export_default CUDA_HOST_COMPILER_REPAIR_ID "not-observed"
  export_default CUDA_HOST_COMPILER_REPAIR_PATH "not-observed"
  export_default CUDA_HOST_COMPILER_REPAIR_SHA256 "not-observed"
  export_default CUDA_HOST_COMPILER_VERSION "not-observed"
  export_default CUDA_HOST_CC_PACKAGE "not-observed"
  export_default CUDA_HOST_CXX_PACKAGE "not-observed"
  export_default CUDA_HOST_CC_RESOLVED "not-observed"
  export_default CUDA_HOST_CXX_RESOLVED "not-observed"
  export_default CUDA_HOST_CC_SHA256 "not-observed"
  export_default CUDA_HOST_CXX_SHA256 "not-observed"
  export_default CUDA_HOST_COMPILER_PROBE_PASSED "false"
  export_default NINJA_BUILD_TOOL_REPAIR_ID "not-observed"
  export_default NINJA_BUILD_TOOL_REPAIR_PATH "not-observed"
  export_default NINJA_BUILD_TOOL_REPAIR_SHA256 "not-observed"
  export_default NINJA_DISTRIBUTION_VERSION "not-observed"
  export_default NINJA_EXECUTABLE_PATH "not-observed"
  export_default NINJA_EXECUTABLE_SHA256 "not-observed"
  export_default NINJA_EXECUTABLE_VERSION "not-observed"
  export_default NINJA_PYTORCH_PROBE_PASSED "false"

  export BPT_SOURCE_PLAN_RUN_ROOT_DEFAULTED="${run_root_defaulted}"
  if ((${#legacy_defaults[@]})); then
    IFS=,
    export BPT_LEGACY_RECEIPT_DEFAULTS="${legacy_defaults[*]}"
    unset IFS
  else
    export BPT_LEGACY_RECEIPT_DEFAULTS=""
  fi
fi

set +e
bash "${base_launcher}" "$@"
status=$?
set -e

if [[ "${1:-}" != "--materialize-physical-upstream" ]]; then
  receipt="${EVIDENCE_ROOT}/deform360-v6-source-prediction-evidence/"\
"execution-receipt.json"
  if [[ -f "${receipt}" && ! -L "${receipt}" ]]; then
    export RECEIPT_PATH="${receipt}"
    export SOURCE_PLAN_ENVIRONMENT_REPAIR_ID
    export SOURCE_PLAN_ENVIRONMENT_REPAIR_PATH
    export SOURCE_PLAN_ENVIRONMENT_REPAIR_SHA256
    export BASE_REVISION BASE_LAUNCHER_BLOB_SHA LAUNCHER_PATH
    receipt_python="${BPT_PRIMARY_PYTHON:-python}"
    "${receipt_python}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

path = Path(os.environ["RECEIPT_PATH"])
receipt = json.loads(path.read_text(encoding="utf-8"))
receipt.pop("receipt_id", None)
if "runtime_source_plan_environment_repair" in receipt:
    raise SystemExit("source-plan environment repair receipt field already exists")

relative_run_root = (
    "bayesian-phystwin/deform360-v6-source-prediction/"
    f"{os.environ['AMENDMENT_ID']}/{os.environ['BPT_SOURCE_SHA']}"
)
expected_run_root = str(Path(os.environ["RESULTS_ROOT"]) / relative_run_root)
if os.environ["RUN_ROOT"] != expected_run_root:
    raise SystemExit("bound RUN_ROOT changed before receipt serialization")

legacy_defaults = tuple(
    value
    for value in os.environ.get("BPT_LEGACY_RECEIPT_DEFAULTS", "").split(",")
    if value
)
if legacy_defaults:
    receipt.pop("runtime_cuda_host_compiler_repair", None)
    receipt.pop("runtime_ninja_build_tool_repair", None)

receipt["runtime_source_plan_environment_repair"] = {
    "activated": True,
    "existing_run_root_preserved": (
        os.environ["BPT_SOURCE_PLAN_RUN_ROOT_DEFAULTED"] == "false"
    ),
    "failed_execution_receipt_id": (
        "79bd32e1af16b3529aeb190494c892cfdb927d526a0f1ef0202aafc99c9188cb"
    ),
    "launcher_path": os.environ["LAUNCHER_PATH"],
    "legacy_receipt_defaults_removed": list(legacy_defaults),
    "predecessor_launcher_blob_sha": os.environ["BASE_LAUNCHER_BLOB_SHA"],
    "predecessor_revision": os.environ["BASE_REVISION"],
    "repair_file_sha256": os.environ["SOURCE_PLAN_ENVIRONMENT_REPAIR_SHA256"],
    "repair_id": os.environ["SOURCE_PLAN_ENVIRONMENT_REPAIR_ID"],
    "repair_path": os.environ["SOURCE_PLAN_ENVIRONMENT_REPAIR_PATH"],
    "run_root_environment_bound": True,
    "run_root_relative_path": relative_run_root,
    "run_root_sha256": hashlib.sha256(
        os.environ["RUN_ROOT"].encode("utf-8")
    ).hexdigest(),
    "source_plan_algorithm_changed": False,
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
fi

exit "${status}"
