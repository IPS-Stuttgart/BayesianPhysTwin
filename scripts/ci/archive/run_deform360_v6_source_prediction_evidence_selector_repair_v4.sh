#!/usr/bin/env bash
set -euo pipefail

: "${BPT_PYTHON:?BPT_PYTHON is required}"
: "${EVIDENCE_ROOT:?EVIDENCE_ROOT is required}"

DELEGATED_WRAPPER="scripts/ci/archive/run_deform360_v6_source_prediction_evidence_selector_repair_v3.sh"
DELEGATED_WRAPPER_BLOB_SHA="5958db6362917e6bc355b194abdac4736e39a5a4"

test -f "${DELEGATED_WRAPPER}"
test ! -L "${DELEGATED_WRAPPER}"
test "$(git hash-object "${DELEGATED_WRAPPER}")" = "${DELEGATED_WRAPPER_BLOB_SHA}"

COMPAT_PYTHON="$(
  mktemp "${RUNNER_TEMP:-/tmp}/deform360-v6-stage-prefix-python.XXXXXX"
)"
cleanup() {
  rm -f "${COMPAT_PYTHON}"
}
trap cleanup EXIT

cat > "${COMPAT_PYTHON}" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
: "${DEFORM360_V6_DELEGATE_PYTHON:?delegate Python is required}"

target="scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"
if [[ "${1:-}" != "${target}" ]]; then
  exec "${DEFORM360_V6_DELEGATE_PYTHON}" "$@"
fi

arguments=("$@")
stage=""
execution_repo=""
for ((index = 1; index < ${#arguments[@]}; index++)); do
  value="${arguments[index]}"
  if [[ "${value}" == "--stage" ]]; then
    next=$((index + 1))
    [[ "${next}" -lt "${#arguments[@]}" ]] || {
      echo "physical-source --stage lacks a value" >&2
      exit 2
    }
    stage="${arguments[next]}"
  elif [[ "${value}" == --stage=* ]]; then
    stage="${value#--stage=}"
  elif [[ "${value}" == "--execution-repo" ]]; then
    next=$((index + 1))
    [[ "${next}" -lt "${#arguments[@]}" ]] || {
      echo "physical-source --execution-repo lacks a value" >&2
      exit 2
    }
    execution_repo="${arguments[next]}"
  elif [[ "${value}" == --execution-repo=* ]]; then
    execution_repo="${value#--execution-repo=}"
  fi
done

if [[ "${stage}" != "stage-prefix" ]]; then
  exec "${DEFORM360_V6_DELEGATE_PYTHON}" "$@"
fi
[[ -n "${execution_repo}" ]] || {
  echo "stage-prefix compatibility repair lacks --execution-repo" >&2
  exit 2
}

rewritten=("${arguments[0]}")
repo_removed=0
role_removed=0
index=1
while [[ "${index}" -lt "${#arguments[@]}" ]]; do
  value="${arguments[index]}"
  case "${value}" in
    --repo)
      next=$((index + 1))
      [[ "${next}" -lt "${#arguments[@]}" ]] || {
        echo "legacy stage-prefix --repo lacks a value" >&2
        exit 2
      }
      [[ "${arguments[next]}" == "${execution_repo}" ]] || {
        echo "legacy stage-prefix --repo does not match --execution-repo" >&2
        exit 2
      }
      repo_removed=$((repo_removed + 1))
      index=$((index + 2))
      ;;
    --repo=*)
      [[ "${value#--repo=}" == "${execution_repo}" ]] || {
        echo "legacy stage-prefix --repo does not match --execution-repo" >&2
        exit 2
      }
      repo_removed=$((repo_removed + 1))
      index=$((index + 1))
      ;;
    --role)
      next=$((index + 1))
      [[ "${next}" -lt "${#arguments[@]}" ]] || {
        echo "legacy stage-prefix --role lacks a value" >&2
        exit 2
      }
      [[ "${arguments[next]}" == "calibration" ]] || {
        echo "legacy stage-prefix role is not calibration" >&2
        exit 2
      }
      role_removed=$((role_removed + 1))
      index=$((index + 2))
      ;;
    --role=*)
      [[ "${value#--role=}" == "calibration" ]] || {
        echo "legacy stage-prefix role is not calibration" >&2
        exit 2
      }
      role_removed=$((role_removed + 1))
      index=$((index + 1))
      ;;
    *)
      rewritten+=("${value}")
      index=$((index + 1))
      ;;
  esac
done

if [[ "${repo_removed}" -ne 1 || "${role_removed}" -ne 1 ]]; then
  echo "legacy stage-prefix compatibility arguments are not unique" >&2
  exit 2
fi

exec "${DEFORM360_V6_DELEGATE_PYTHON}" "${rewritten[@]}"
SH
chmod 700 "${COMPAT_PYTHON}"

DEFORM360_V6_DELEGATE_PYTHON="${BPT_PYTHON}" \
BPT_PYTHON="${COMPAT_PYTHON}" \
  bash "${DELEGATED_WRAPPER}"
