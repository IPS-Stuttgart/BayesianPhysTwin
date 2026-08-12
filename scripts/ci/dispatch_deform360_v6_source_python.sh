#!/usr/bin/env bash
set -euo pipefail

: "${BPT_PRIMARY_PYTHON:?BPT_PRIMARY_PYTHON is required}"
: "${BPT_FRAME_ZERO_PYTHON:?BPT_FRAME_ZERO_PYTHON is required}"
: "${BPT_FRAME_ZERO_RUNTIME_MARKER:?BPT_FRAME_ZERO_RUNTIME_MARKER is required}"
: "${BPT_FRAME_ZERO_FALLBACK_CONFIG_REPAIR_MARKER:?BPT_FRAME_ZERO_FALLBACK_CONFIG_REPAIR_MARKER is required}"
: "${BPT_OFFICIAL_PHYSTWIN_RUNTIME_MARKER:?BPT_OFFICIAL_PHYSTWIN_RUNTIME_MARKER is required}"
: "${BPT_CASE_STDIN_ISOLATION_MARKER:?BPT_CASE_STDIN_ISOLATION_MARKER is required}"

readonly FRAME_ZERO_DISPATCH_REPAIR_ID="6524b544bb59d06fee3388906d680b8f1436a0c6a36555cd8f3de0c76074deb8"
readonly FALLBACK_CONFIG_ROUTE_REPAIR_ID="df4fd52c65acc25c70c4cde650dd021f704e799dceda3323f3aa28af6fd99e0e"
readonly OFFICIAL_PHYSTWIN_RUNTIME_REPAIR_ID="72db4752194340a4e8122332ec7483e7d397240c714b3aeec771b1e043369deb"
readonly CASE_STDIN_ISOLATION_REPAIR_ID="08df8831e0261b482c9e682b30b0a7fdf37b0924de23a58484db9ce2546b625e"
readonly PHYSICAL_TARGET="scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"
readonly STAGE_SELECTOR_HELPER_TARGET="scripts/remote/run_deform360_v6_stage_selector_identity_repair.py"
readonly MATERIALIZER_TARGET="scripts/science/materialize_deform360_joint_sparse_physical_source_v5.py"
readonly FRAME_ZERO_STAGE="frame-zero"
readonly PHYSICAL_PRIOR_STAGE="physical-prior"
readonly FALLBACK_CONFIG_FLAG="--persistence-fallback-source-config"
readonly PREVIOUS_FALLBACK_CONFIG="configs/sota/deform360_reconstruction_failure_persistence_fallback_v1.json"
readonly PREVIOUS_FALLBACK_CONFIG_FILE_SHA256="240554ed41986cf5b330225d759c8df24de99d8642f9c1dcd6114185ee16fc0d"
readonly CORRECTED_FALLBACK_CONFIG="configs/sota/deform360_frame_zero_initializer_source_v1.json"
readonly CORRECTED_FALLBACK_CONFIG_FILE_SHA256="60e9887836ea0ba3410066dbb5668988e35763ea6161826752b24b257cf9fc66"
readonly EXPECTED_MARKER="{\"repair_id\":\"${FRAME_ZERO_DISPATCH_REPAIR_ID}\",\"stage\":\"${FRAME_ZERO_STAGE}\"}"
readonly EXPECTED_FALLBACK_CONFIG_REPAIR_MARKER="{\"corrected_config\":\"${CORRECTED_FALLBACK_CONFIG}\",\"previous_config\":\"${PREVIOUS_FALLBACK_CONFIG}\",\"repair_id\":\"${FALLBACK_CONFIG_ROUTE_REPAIR_ID}\",\"stage\":\"${FRAME_ZERO_STAGE}\"}"
readonly EXPECTED_OFFICIAL_PHYSTWIN_RUNTIME_MARKER="{\"repair_id\":\"${OFFICIAL_PHYSTWIN_RUNTIME_REPAIR_ID}\",\"stage\":\"${PHYSICAL_PRIOR_STAGE}\"}"
readonly EXPECTED_CASE_STDIN_ISOLATION_MARKER="{\"repair_id\":\"${CASE_STDIN_ISOLATION_REPAIR_ID}\",\"stdin\":\"/dev/null\"}"

mark_case_stdin_isolation() {
  local marker="${BPT_CASE_STDIN_ISOLATION_MARKER}"
  local parent
  local temporary
  parent="$(dirname "${marker}")"
  mkdir -p "${parent}"
  if [[ -L "${marker}" ]]; then
    echo "refusing symlinked case stdin isolation marker" >&2
    exit 2
  fi
  if [[ -e "${marker}" ]]; then
    [[ -f "${marker}" ]] || {
      echo "case stdin isolation marker is not a regular file" >&2
      exit 2
    }
    [[ "$(cat "${marker}")" == "${EXPECTED_CASE_STDIN_ISOLATION_MARKER}" ]] || {
      echo "case stdin isolation marker changed" >&2
      exit 2
    }
    return
  fi
  temporary="${marker}.tmp.$$"
  umask 077
  printf '%s\n' "${EXPECTED_CASE_STDIN_ISOLATION_MARKER}" > "${temporary}"
  [[ ! -L "${temporary}" && -f "${temporary}" ]] || {
    echo "case stdin isolation marker temporary path is unsafe" >&2
    exit 2
  }
  mv "${temporary}" "${marker}"
}

mark_frame_zero_runtime() {
  local marker="${BPT_FRAME_ZERO_RUNTIME_MARKER}"
  local parent
  local temporary
  parent="$(dirname "${marker}")"
  mkdir -p "${parent}"
  if [[ -L "${marker}" ]]; then
    echo "refusing symlinked frame-zero runtime marker" >&2
    exit 2
  fi
  if [[ -e "${marker}" ]]; then
    [[ -f "${marker}" ]] || {
      echo "frame-zero runtime marker is not a regular file" >&2
      exit 2
    }
    [[ "$(cat "${marker}")" == "${EXPECTED_MARKER}" ]] || {
      echo "frame-zero runtime marker changed" >&2
      exit 2
    }
    return
  fi
  temporary="${marker}.tmp.$$"
  umask 077
  printf '%s\n' "${EXPECTED_MARKER}" > "${temporary}"
  [[ ! -L "${temporary}" && -f "${temporary}" ]] || {
    echo "frame-zero runtime marker temporary path is unsafe" >&2
    exit 2
  }
  mv "${temporary}" "${marker}"
}

mark_fallback_config_repair() {
  local marker="${BPT_FRAME_ZERO_FALLBACK_CONFIG_REPAIR_MARKER}"
  local parent
  local temporary
  parent="$(dirname "${marker}")"
  mkdir -p "${parent}"
  if [[ -L "${marker}" ]]; then
    echo "refusing symlinked fallback-config repair marker" >&2
    exit 2
  fi
  if [[ -e "${marker}" ]]; then
    [[ -f "${marker}" ]] || {
      echo "fallback-config repair marker is not a regular file" >&2
      exit 2
    }
    [[ "$(cat "${marker}")" == "${EXPECTED_FALLBACK_CONFIG_REPAIR_MARKER}" ]] || {
      echo "fallback-config repair marker changed" >&2
      exit 2
    }
    return
  fi
  temporary="${marker}.tmp.$$"
  umask 077
  printf '%s\n' "${EXPECTED_FALLBACK_CONFIG_REPAIR_MARKER}" > "${temporary}"
  [[ ! -L "${temporary}" && -f "${temporary}" ]] || {
    echo "fallback-config repair marker temporary path is unsafe" >&2
    exit 2
  }
  mv "${temporary}" "${marker}"
}

mark_official_phystwin_runtime() {
  local marker="${BPT_OFFICIAL_PHYSTWIN_RUNTIME_MARKER}"
  local parent
  local temporary
  parent="$(dirname "${marker}")"
  mkdir -p "${parent}"
  if [[ -L "${marker}" ]]; then
    echo "refusing symlinked official PhysTwin runtime marker" >&2
    exit 2
  fi
  if [[ -e "${marker}" ]]; then
    [[ -f "${marker}" ]] || {
      echo "official PhysTwin runtime marker is not a regular file" >&2
      exit 2
    }
    [[ "$(cat "${marker}")" == "${EXPECTED_OFFICIAL_PHYSTWIN_RUNTIME_MARKER}" ]] || {
      echo "official PhysTwin runtime marker changed" >&2
      exit 2
    }
    return
  fi
  temporary="${marker}.tmp.$$"
  umask 077
  printf '%s\n' "${EXPECTED_OFFICIAL_PHYSTWIN_RUNTIME_MARKER}" > "${temporary}"
  [[ ! -L "${temporary}" && -f "${temporary}" ]] || {
    echo "official PhysTwin runtime marker temporary path is unsafe" >&2
    exit 2
  }
  mv "${temporary}" "${marker}"
}

set_legacy_receipt_default() {
  local family="$1"
  local name="$2"
  local value="$3"
  if declare -p "${name}" >/dev/null 2>&1; then
    export "${name}"
    return
  fi
  printf -v "${name}" '%s' "${value}"
  export "${name}"
  if [[ "${family}" == "cuda" ]]; then
    BPT_CUDA_LEGACY_RECEIPT_DEFAULTED=true
  else
    BPT_NINJA_LEGACY_RECEIPT_DEFAULTED=true
  fi
}

run_stdin_with_receipt_compatibility() {
  local stdin_copy
  local status
  stdin_copy="$(
    mktemp "${RUNNER_TEMP:-/tmp}/deform360-v6-dispatch-stdin.XXXXXX"
  )"
  chmod 600 "${stdin_copy}"
  cat > "${stdin_copy}"

  if ! grep -Fq 'path = Path(os.environ["RECEIPT_PATH"])' "${stdin_copy}" \
    || ! grep -Fq 'receipt["runtime_cuda_host_compiler_repair"] = {' "${stdin_copy}" \
    || ! grep -Fq 'receipt["runtime_ninja_build_tool_repair"] = {' "${stdin_copy}"; then
    set +e
    "${BPT_PRIMARY_PYTHON}" "$@" < "${stdin_copy}"
    status=$?
    set -e
    rm -f "${stdin_copy}"
    return "${status}"
  fi

  : "${RECEIPT_PATH:?RECEIPT_PATH is required for legacy receipt compatibility}"
  BPT_CUDA_LEGACY_RECEIPT_DEFAULTED=false
  BPT_NINJA_LEGACY_RECEIPT_DEFAULTED=false
  export BPT_CUDA_LEGACY_RECEIPT_DEFAULTED
  export BPT_NINJA_LEGACY_RECEIPT_DEFAULTED

  set_legacy_receipt_default cuda CUDA_HOST_COMPILER_REPAIR_ID "not-observed"
  set_legacy_receipt_default cuda CUDA_HOST_COMPILER_REPAIR_PATH "not-observed"
  set_legacy_receipt_default cuda CUDA_HOST_COMPILER_REPAIR_SHA256 "not-observed"
  set_legacy_receipt_default cuda CUDA_HOST_COMPILER_VERSION "not-observed"
  set_legacy_receipt_default cuda CUDA_HOST_CC_PACKAGE "not-observed"
  set_legacy_receipt_default cuda CUDA_HOST_CXX_PACKAGE "not-observed"
  set_legacy_receipt_default cuda CUDA_HOST_CC_RESOLVED "not-observed"
  set_legacy_receipt_default cuda CUDA_HOST_CXX_RESOLVED "not-observed"
  set_legacy_receipt_default cuda CUDA_HOST_CC_SHA256 "not-observed"
  set_legacy_receipt_default cuda CUDA_HOST_CXX_SHA256 "not-observed"
  set_legacy_receipt_default cuda CUDA_HOST_COMPILER_PROBE_PASSED "false"
  set_legacy_receipt_default ninja NINJA_BUILD_TOOL_REPAIR_ID "not-observed"
  set_legacy_receipt_default ninja NINJA_BUILD_TOOL_REPAIR_PATH "not-observed"
  set_legacy_receipt_default ninja NINJA_BUILD_TOOL_REPAIR_SHA256 "not-observed"
  set_legacy_receipt_default ninja NINJA_DISTRIBUTION_VERSION "not-observed"
  set_legacy_receipt_default ninja NINJA_EXECUTABLE_PATH "not-observed"
  set_legacy_receipt_default ninja NINJA_EXECUTABLE_SHA256 "not-observed"
  set_legacy_receipt_default ninja NINJA_EXECUTABLE_VERSION "not-observed"
  set_legacy_receipt_default ninja NINJA_PYTORCH_PROBE_PASSED "false"

  set +e
  "${BPT_PRIMARY_PYTHON}" "$@" < "${stdin_copy}"
  status=$?
  set -e
  rm -f "${stdin_copy}"
  [[ "${status}" -eq 0 ]] || return "${status}"

  if [[ "${BPT_CUDA_LEGACY_RECEIPT_DEFAULTED}" == "false" \
    && "${BPT_NINJA_LEGACY_RECEIPT_DEFAULTED}" == "false" ]]; then
    return 0
  fi

  "${BPT_PRIMARY_PYTHON}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

path = Path(os.environ["RECEIPT_PATH"])
if path.is_symlink() or not path.is_file():
    raise SystemExit("legacy receipt compatibility output is unavailable or unsafe")
receipt = json.loads(path.read_text(encoding="utf-8"))
receipt.pop("receipt_id", None)
if os.environ["BPT_CUDA_LEGACY_RECEIPT_DEFAULTED"] == "true":
    receipt.pop("runtime_cuda_host_compiler_repair", None)
if os.environ["BPT_NINJA_LEGACY_RECEIPT_DEFAULTED"] == "true":
    receipt.pop("runtime_ninja_build_tool_repair", None)
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
}

if [[ "${1:-}" == "${PHYSICAL_TARGET}" ]]; then
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
  [[ "${stage_count}" -eq 1 ]] || {
    echo "physical source stage binding is not unique" >&2
    exit 2
  }
  if [[ "${stage_value}" == "${FRAME_ZERO_STAGE}" ]]; then
    fallback_config_count=0
    for ((index = 0; index < ${#arguments[@]}; index++)); do
      if [[ "${arguments[index]}" == "${FALLBACK_CONFIG_FLAG}" ]]; then
        ((index + 1 < ${#arguments[@]})) || {
          echo "frame-zero fallback config lacks a value" >&2
          exit 2
        }
        [[ "${arguments[index + 1]}" == "${PREVIOUS_FALLBACK_CONFIG}" ]] || {
          echo "frame-zero fallback config no longer matches the retained failure" >&2
          exit 2
        }
        fallback_config_count=$((fallback_config_count + 1))
        arguments[index + 1]="${CORRECTED_FALLBACK_CONFIG}"
      fi
    done
    [[ "${fallback_config_count}" -eq 1 ]] || {
      echo "frame-zero fallback config binding is not unique" >&2
      exit 2
    }
    [[ -f "${PREVIOUS_FALLBACK_CONFIG}" && ! -L "${PREVIOUS_FALLBACK_CONFIG}" ]] || {
      echo "previous fallback config is unavailable" >&2
      exit 2
    }
    [[ -f "${CORRECTED_FALLBACK_CONFIG}" && ! -L "${CORRECTED_FALLBACK_CONFIG}" ]] || {
      echo "corrected fallback config is unavailable" >&2
      exit 2
    }
    [[ "$(sha256sum "${PREVIOUS_FALLBACK_CONFIG}" | awk '{print $1}')" == "${PREVIOUS_FALLBACK_CONFIG_FILE_SHA256}" ]] || {
      echo "previous fallback config bytes changed" >&2
      exit 2
    }
    [[ "$(sha256sum "${CORRECTED_FALLBACK_CONFIG}" | awk '{print $1}')" == "${CORRECTED_FALLBACK_CONFIG_FILE_SHA256}" ]] || {
      echo "corrected fallback config bytes changed" >&2
      exit 2
    }
    mark_frame_zero_runtime
    mark_fallback_config_repair
    mark_case_stdin_isolation
    exec "${BPT_FRAME_ZERO_PYTHON}" "${arguments[@]}" </dev/null
  fi
  if [[ "${stage_value}" == "${PHYSICAL_PRIOR_STAGE}" ]]; then
    mark_official_phystwin_runtime
    mark_case_stdin_isolation
    exec "${BPT_FRAME_ZERO_PYTHON}" "${arguments[@]}" </dev/null
  fi
  mark_case_stdin_isolation
  exec "${BPT_PRIMARY_PYTHON}" "$@" </dev/null
fi

if [[ "${1:-}" == "${STAGE_SELECTOR_HELPER_TARGET}" \
  || "${1:-}" == "${MATERIALIZER_TARGET}" ]]; then
  mark_case_stdin_isolation
  exec "${BPT_PRIMARY_PYTHON}" "$@" </dev/null
fi

if [[ "${1:-}" == "-" ]]; then
  run_stdin_with_receipt_compatibility "$@"
  exit $?
fi

exec "${BPT_PRIMARY_PYTHON}" "$@"
