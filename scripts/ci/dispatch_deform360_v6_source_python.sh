#!/usr/bin/env bash
set -euo pipefail

: "${BPT_PRIMARY_PYTHON:?BPT_PRIMARY_PYTHON is required}"
: "${BPT_FRAME_ZERO_PYTHON:?BPT_FRAME_ZERO_PYTHON is required}"
: "${BPT_FRAME_ZERO_RUNTIME_MARKER:?BPT_FRAME_ZERO_RUNTIME_MARKER is required}"

readonly REPAIR_ID="6524b544bb59d06fee3388906d680b8f1436a0c6a36555cd8f3de0c76074deb8"
readonly PHYSICAL_TARGET="scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"
readonly FRAME_ZERO_STAGE="frame-zero"
readonly EXPECTED_MARKER="{\"repair_id\":\"${REPAIR_ID}\",\"stage\":\"${FRAME_ZERO_STAGE}\"}"

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
    mark_frame_zero_runtime
    exec "${BPT_FRAME_ZERO_PYTHON}" "$@"
  fi
fi

exec "${BPT_PRIMARY_PYTHON}" "$@"
