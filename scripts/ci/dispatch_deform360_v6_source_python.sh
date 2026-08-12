#!/usr/bin/env bash
set -euo pipefail

: "${BPT_PRIMARY_PYTHON:?BPT_PRIMARY_PYTHON is required}"
: "${BPT_FRAME_ZERO_PYTHON:?BPT_FRAME_ZERO_PYTHON is required}"
: "${BPT_FRAME_ZERO_RUNTIME_MARKER:?BPT_FRAME_ZERO_RUNTIME_MARKER is required}"
: "${BPT_FRAME_ZERO_FALLBACK_CONFIG_REPAIR_MARKER:?BPT_FRAME_ZERO_FALLBACK_CONFIG_REPAIR_MARKER is required}"
: "${BPT_OFFICIAL_PHYSTWIN_RUNTIME_MARKER:?BPT_OFFICIAL_PHYSTWIN_RUNTIME_MARKER is required}"

readonly FRAME_ZERO_DISPATCH_REPAIR_ID="6524b544bb59d06fee3388906d680b8f1436a0c6a36555cd8f3de0c76074deb8"
readonly FALLBACK_CONFIG_ROUTE_REPAIR_ID="df4fd52c65acc25c70c4cde650dd021f704e799dceda3323f3aa28af6fd99e0e"
readonly OFFICIAL_PHYSTWIN_RUNTIME_REPAIR_ID="72db4752194340a4e8122332ec7483e7d397240c714b3aeec771b1e043369deb"
readonly PHYSICAL_TARGET="scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"
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
    exec "${BPT_FRAME_ZERO_PYTHON}" "${arguments[@]}"
  fi
  if [[ "${stage_value}" == "${PHYSICAL_PRIOR_STAGE}" ]]; then
    mark_official_phystwin_runtime
    exec "${BPT_FRAME_ZERO_PYTHON}" "${arguments[@]}"
  fi
fi

exec "${BPT_PRIMARY_PYTHON}" "$@"
