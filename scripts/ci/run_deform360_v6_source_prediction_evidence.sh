#!/usr/bin/env bash
set -euo pipefail

# Preserve the reviewed source-evidence runner byte-for-byte and repair only the
# discovery transport for its checksum-frozen Causal4D selector dependency.
RUNNER_IMPL="$(dirname "${BASH_SOURCE[0]}")/run_deform360_v6_source_prediction_evidence_locked.sh"
SELECTOR_RELATIVE_PATH="src/causal4d_public/deform360_object_sam2.py"
SELECTOR_SHA256="79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
DISCOVERY_REVISION="${CAUSAL4D_DISCOVERY_REVISION:-50e3682a5dbf976b20cc9115b6e7a975d0144ea5}"

# Security boundary: recovery is anchored to DISCOVERY_REVISION. Do not replace
# it with '+refs/heads/*:refs/remotes/origin/*' or "--all --format='%H'".
materialize_frozen_selector_history() {
  local repository
  local current_source
  local revision=""
  local candidate
  local history_root
  local scratch_root

  repository="${GITHUB_WORKSPACE:-}/_causal4d_discovery"
  current_source="${repository}/${SELECTOR_RELATIVE_PATH}"
  if [[ -f "${current_source}" && ! -L "${current_source}" ]] \
    && [[ "$(sha256sum "${current_source}" | awk '{print $1}')" \
      = "${SELECTOR_SHA256}" ]]; then
    return 0
  fi
  if [[ ! -d "${repository}/.git" ]]; then
    echo "Causal4D discovery checkout is unavailable; retaining bounded runner failure" >&2
    return 0
  fi
  if ! git -C "${repository}" cat-file -e "${DISCOVERY_REVISION}^{commit}"; then
    echo "configured Causal4D discovery revision is unavailable" >&2
    return 0
  fi

  if [[ "$(git -C "${repository}" rev-parse --is-shallow-repository)" = "true" ]]; then
    if ! git -C "${repository}" fetch \
      --no-tags --prune --unshallow origin \
      '+refs/heads/main:refs/remotes/origin/main'; then
      echo "cannot expand Causal4D discovery history; retaining bounded runner failure" >&2
      return 0
    fi
  fi

  candidate="${RUNNER_TEMP:-/tmp}/deform360-frozen-selector-candidate-$$.py"
  while IFS= read -r commit; do
    [[ -n "${commit}" ]] || continue
    if git -C "${repository}" show \
      "${commit}:${SELECTOR_RELATIVE_PATH}" \
      > "${candidate}" 2>/dev/null \
      && [[ "$(sha256sum "${candidate}" | awk '{print $1}')" \
        = "${SELECTOR_SHA256}" ]]; then
      revision="${commit}"
      break
    fi
  done < <(
    git -C "${repository}" log \
      --format='%H' "${DISCOVERY_REVISION}" -- "${SELECTOR_RELATIVE_PATH}"
  )
  rm -f "${candidate}"

  if [[ -z "${revision}" ]]; then
    echo "frozen Causal4D selector bytes are absent from anchored history" >&2
    return 0
  fi

  history_root="${GITHUB_WORKSPACE}/_frozen_causal4d_selector"
  scratch_root="${history_root}.tmp-$$"
  rm -rf "${scratch_root}"
  mkdir -p "${scratch_root}"
  if ! git -C "${repository}" archive \
    "${revision}" src/causal4d_public \
    | tar -xf - -C "${scratch_root}"; then
    rm -rf "${scratch_root}"
    echo "cannot materialize frozen Causal4D selector source" >&2
    return 0
  fi

  candidate="${scratch_root}/${SELECTOR_RELATIVE_PATH}"
  if [[ ! -f "${candidate}" || -L "${candidate}" ]] \
    || [[ "$(sha256sum "${candidate}" | awk '{print $1}')" \
      != "${SELECTOR_SHA256}" ]]; then
    rm -rf "${scratch_root}"
    echo "materialized Causal4D selector failed checksum verification" >&2
    return 0
  fi

  rm -rf "${history_root}"
  mv "${scratch_root}" "${history_root}"
  echo "materialized frozen Causal4D selector revision=${revision}"
}

materialize_frozen_selector_history
bash -n "${RUNNER_IMPL}"
exec bash "${RUNNER_IMPL}" "$@"
