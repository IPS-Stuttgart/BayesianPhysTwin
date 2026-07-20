#!/bin/bash

# Normalize before inspecting the deployment or confirmation artifact tree.
if [[ "${BPT_HELD_V6_CONFIRMATION_SHARD_ENV_NORMALIZED:-}" != "1" ]]; then
  exec env -i \
    HOME=/home/florianpfaff \
    USER=florianpfaff \
    LOGNAME=florianpfaff \
    PATH=/usr/local/bin:/usr/bin:/bin \
    TMPDIR=/tmp \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    BPT_HELD_V6_CODE="${BPT_HELD_V6_CODE:-}" \
    BPT_HELD_V6_CONFIRMATION_SHARD_ENV_NORMALIZED=1 \
    /bin/bash "$0" "$@"
fi

set -Eeuo pipefail
umask 077
export PATH=/usr/local/bin:/usr/bin:/bin
unset BASH_ENV ENV CDPATH

# The audited launcher must invoke this file as `env -i ... /bin/bash FILE ...`
# after checking its deployed source digest.  Bash can process BASH_ENV before
# the first line of a script, so a running shell cannot bootstrap its own trust.

die() {
  echo "ERROR: $*" >&2
  exit 2
}

while IFS='=' read -r environment_name _value; do
  case "$environment_name" in
    HOME|USER|LOGNAME|PATH|TMPDIR|LANG|LC_ALL|PWD|SHLVL|_|\
    BPT_HELD_V6_CODE|BPT_HELD_V6_CONFIRMATION_SHARD_ENV_NORMALIZED)
      ;;
    *) die "normalized confirmation-shard environment contains $environment_name" ;;
  esac
done < <(env)

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  die "usage: run_deform360_v6_confirmation_shard.sh SHARD_INDEX [CUDA_DEVICE]"
fi

readonly SHARD_INDEX="$1"
readonly CUDA_DEVICE="${2:-0}"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SHARD_RUNNER="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
readonly CASE_RUNNER="$SCRIPT_DIR/run_deform360_v6_confirmation_case.sh"

case "$SHARD_INDEX" in
  0|1) ;;
  *) die "shard index must be 0 or 1" ;;
esac
[[ "$CUDA_DEVICE" =~ ^[0-9]+$ ]] || die "CUDA device must be a non-negative integer"
[[ "$(hostname)" == "workstation2" ]] || \
  die "formal held-v6 shards must run together on gpuserver6000/workstation2"
if [[ "$SHARD_INDEX" == "0" ]]; then
  [[ "$CUDA_DEVICE" == "0" ]] || die "formal shard 0 is bound to CUDA device 0"
else
  [[ "$CUDA_DEVICE" == "1" ]] || die "formal shard 1 is bound to CUDA device 1"
fi

# The canonical cohort and both disjoint shards live together so startup can
# prove exact coverage before a prediction process starts.
readonly -a ALL_CASE_SPECS=(
  "002-rope-silk-ep0001:002-rope-silk:0001"
  "081-stripe-rope-ep0005:081-stripe-rope:0005"
  "085-scarf-cloth-ep0002:085-scarf-cloth:0002"
  "083-blanket-cloth-ep0007:083-blanket-cloth:0007"
  "092-squirrel-ep0001:092-squirrel:0001"
  "170-spider-ep0006:170-spider:0006"
)
readonly -a SHARD_0_CASE_SPECS=(
  "002-rope-silk-ep0001:002-rope-silk:0001"
  "085-scarf-cloth-ep0002:085-scarf-cloth:0002"
  "092-squirrel-ep0001:092-squirrel:0001"
)
readonly -a SHARD_1_CASE_SPECS=(
  "081-stripe-rope-ep0005:081-stripe-rope:0005"
  "083-blanket-cloth-ep0007:083-blanket-cloth:0007"
  "170-spider-ep0006:170-spider:0006"
)

[[ "${#ALL_CASE_SPECS[@]}" -eq 6 ]] || die "canonical cohort does not contain six cases"
[[ "${#SHARD_0_CASE_SPECS[@]}" -eq 3 && "${#SHARD_1_CASE_SPECS[@]}" -eq 3 ]] || \
  die "shard sizes changed from the fixed 3/3 split"

declare -A COHORT_COUNTS=()
for spec in "${ALL_CASE_SPECS[@]}"; do
  [[ ! -v "COHORT_COUNTS[$spec]" ]] || die "duplicate in canonical cohort: $spec"
  COHORT_COUNTS["$spec"]=0
done
for spec in "${SHARD_0_CASE_SPECS[@]}" "${SHARD_1_CASE_SPECS[@]}"; do
  [[ -v "COHORT_COUNTS[$spec]" ]] || die "shard contains a non-cohort case: $spec"
  COHORT_COUNTS["$spec"]=$((COHORT_COUNTS["$spec"] + 1))
done
for spec in "${ALL_CASE_SPECS[@]}"; do
  [[ "${COHORT_COUNTS[$spec]}" -eq 1 ]] || die "cohort case is missing or duplicated: $spec"
done

if [[ "$SHARD_INDEX" == "0" ]]; then
  readonly -a CASE_SPECS=("${SHARD_0_CASE_SPECS[@]}")
else
  readonly -a CASE_SPECS=("${SHARD_1_CASE_SPECS[@]}")
fi

readonly HELD="/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v6"
readonly V1_LOCK="/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v1/calibration-lock.json"
readonly V1_REPORT="/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v1/v1-preoutcome-feasibility-report.json"
readonly V4_HELD="/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v4"
readonly V2_WITHDRAWAL_REPORT="$V4_HELD/v2-design-withdrawal-report.json"
readonly V3_BOUNDARY_INCIDENT_REPORT="$V4_HELD/v3-prelock-boundary-incident-report.json"
readonly V4_LOCK="$V4_HELD/calibration-lock.json"
readonly V4_EXECUTION_WITHDRAWAL_REPORT="$V4_HELD/v4-execution-withdrawal-report.json"
readonly V5_HELD="/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v5"
readonly V5_LOCK="$V5_HELD/calibration-lock.json"
readonly V5_OUTCOME_WITHDRAWAL_REPORT="$V5_HELD/v5-outcome-withdrawal-report.json"
readonly CALIBRATION_LOCK="$HELD/calibration-lock.json"
readonly CALIBRATION_DECISION="$HELD/calibration/calibration-gate-decision.json"
readonly CONFIRMATION_LOCK="$HELD/confirmation-lock.json"
readonly BPT_HELD_V6_CODE="${BPT_HELD_V6_CODE:?set BPT_HELD_V6_CODE to the immutable v6 deployment}"
readonly LOCK_OPERATOR="$SCRIPT_DIR/prepare_deform360_v6_lock.py"
readonly OUTCOME_DRIVER="$SCRIPT_DIR/run_deform360_v6_confirmation_outcomes.py"
# Updated mechanically after the final lock-operator source is frozen.
readonly EXPECTED_LOCK_OPERATOR_SHA256="7622e8a4338c9a76da3d114bde4fa5407374396f4943a838b13dd2214eebe329"
readonly PY="/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004/bin/python"
readonly PYCACHE_PREFIX="/nonexistent/bpt-held-v6-pycache"
readonly RUN="$HELD/confirmation"
readonly LOG_DIR="$RUN/logs"

cd -- "$BPT_HELD_V6_CODE"
[[ "$PWD" == "$BPT_HELD_V6_CODE" ]] || die "failed to enter immutable deployed code root"

[[ -x "$SHARD_RUNNER" && ! -L "$SHARD_RUNNER" ]] || die "shard runner is absent, non-executable, or a symlink"
[[ -x "$CASE_RUNNER" && ! -L "$CASE_RUNNER" ]] || die "case runner is absent, non-executable, or a symlink"
[[ -x "$LOCK_OPERATOR" && ! -L "$LOCK_OPERATOR" ]] || die "lock operator is absent, non-executable, or a symlink"
[[ -x "$OUTCOME_DRIVER" && ! -L "$OUTCOME_DRIVER" ]] || die "outcome driver is absent, non-executable, or a symlink"
for operator in "$SHARD_RUNNER" "$CASE_RUNNER" "$LOCK_OPERATOR" "$OUTCOME_DRIVER"; do
  [[ "$(readlink -f -- "$operator")" == "$operator" ]] || die "operator path is not canonical: $operator"
done
[[ -z "$(find "$SHARD_RUNNER" "$CASE_RUNNER" "$LOCK_OPERATOR" "$OUTCOME_DRIVER" -maxdepth 0 -perm /222 -print -quit)" ]] || \
  die "operator runners are writable; deploy a read-only copy before execution"
[[ -d "$HELD" && ! -L "$HELD" && "$(readlink -f -- "$HELD")" == "$HELD" ]] || \
  die "held-v6 root is absent, a symlink, or non-canonical"
[[ -d "$BPT_HELD_V6_CODE" && ! -L "$BPT_HELD_V6_CODE" ]] || \
  die "v6 deployed code is absent or a symlink"
[[ "$(readlink -f -- "$BPT_HELD_V6_CODE")" == "$BPT_HELD_V6_CODE" ]] || \
  die "v6 deployed code is not canonical"
[[ "$BPT_HELD_V6_CODE" =~ ^${HELD}/code-([0-9a-f]{40}|[0-9a-f]{64})$ ]] || \
  die "v6 deployed code is outside the canonical immutable snapshot path"
[[ "$SCRIPT_DIR" == "$BPT_HELD_V6_CODE/scripts/held" ]] || \
  die "operator bundle is outside the deployed v6 snapshot"
[[ -z "$(find "$BPT_HELD_V6_CODE" -xdev -perm /222 -print -quit)" ]] || \
  die "v6 deployed code tree is writable"
[[ -f "$CALIBRATION_LOCK" && ! -L "$CALIBRATION_LOCK" && "$(stat -c '%a' -- "$CALIBRATION_LOCK")" == "400" ]] || \
  die "held-v6 calibration parent lock is absent, mutable, or a symlink"
[[ -f "$CALIBRATION_DECISION" && ! -L "$CALIBRATION_DECISION" ]] || \
  die "held-v6 calibration GO decision is absent or a symlink"
[[ -f "$CONFIRMATION_LOCK" && ! -L "$CONFIRMATION_LOCK" ]] || die "held-v6 confirmation lock is absent or a symlink"
[[ "$(readlink -f -- "$CONFIRMATION_LOCK")" == "$CONFIRMATION_LOCK" ]] || \
  die "held-v6 confirmation lock is not canonical"
[[ "$(stat -c '%a' -- "$CONFIRMATION_LOCK")" == "400" ]] || die "held-v6 confirmation lock mode is not 0400"
[[ -f "$V1_LOCK" && ! -L "$V1_LOCK" && "$(stat -c '%a' -- "$V1_LOCK")" == "400" ]] || \
  die "sealed v1 parent lock is absent, mutable, or a symlink"
[[ -f "$V1_REPORT" && ! -L "$V1_REPORT" && "$(stat -c '%a' -- "$V1_REPORT")" == "400" ]] || \
  die "sealed v1 pre-outcome report is absent, mutable, or a symlink"
[[ -f "$V2_WITHDRAWAL_REPORT" && ! -L "$V2_WITHDRAWAL_REPORT" && "$(stat -c '%a' -- "$V2_WITHDRAWAL_REPORT")" == "400" ]] || \
  die "sealed v2 design-withdrawal report is absent, mutable, or a symlink"
[[ -f "$V3_BOUNDARY_INCIDENT_REPORT" && ! -L "$V3_BOUNDARY_INCIDENT_REPORT" && "$(stat -c '%a' -- "$V3_BOUNDARY_INCIDENT_REPORT")" == "400" ]] || \
  die "sealed v3 boundary-incident report is absent, mutable, or a symlink"
[[ -f "$V4_LOCK" && ! -L "$V4_LOCK" && "$(stat -c '%a' -- "$V4_LOCK")" == "400" ]] || \
  die "sealed v4 calibration lock is absent, mutable, or a symlink"
[[ -f "$V4_EXECUTION_WITHDRAWAL_REPORT" && ! -L "$V4_EXECUTION_WITHDRAWAL_REPORT" && "$(stat -c '%a' -- "$V4_EXECUTION_WITHDRAWAL_REPORT")" == "400" ]] || \
  die "sealed v4 execution-withdrawal report is absent, mutable, or a symlink"
[[ -f "$V5_LOCK" && ! -L "$V5_LOCK" && "$(stat -c '%a' -- "$V5_LOCK")" == "400" ]] || \
  die "sealed v5 calibration lock is absent, mutable, or a symlink"
[[ -f "$V5_OUTCOME_WITHDRAWAL_REPORT" && ! -L "$V5_OUTCOME_WITHDRAWAL_REPORT" && "$(stat -c '%a' -- "$V5_OUTCOME_WITHDRAWAL_REPORT")" == "400" ]] || \
  die "sealed v5 outcome-withdrawal report is absent, mutable, or a symlink"
readonly ACTUAL_LOCK_OPERATOR_SHA256="$(sha256sum -- "$LOCK_OPERATOR" | awk '{print $1}')"
[[ "$ACTUAL_LOCK_OPERATOR_SHA256" == "$EXPECTED_LOCK_OPERATOR_SHA256" ]] || \
  die "lock operator checksum differs from the audited verifier"
[[ -x "$PY" ]] || die "locked virtualenv Python entry point is not executable"
[[ ! -e /nonexistent && ! -L /nonexistent && ! -e "$PYCACHE_PREFIX" && ! -L "$PYCACHE_PREFIX" ]] || \
  die "reserved held-v6 pycache prefix is no longer unavailable"

# Refuse a pre-existing symlink at either output-parent level before mkdir -p
# can follow it outside held-v6.
for output_parent in "$RUN" "$LOG_DIR"; do
  if [[ -e "$output_parent" || -L "$output_parent" ]]; then
    [[ -d "$output_parent" && ! -L "$output_parent" ]] || \
      die "output parent is not a real directory: $output_parent"
    [[ "$(readlink -f -- "$output_parent")" == "$output_parent" ]] || \
      die "output parent is non-canonical: $output_parent"
  fi
done
mkdir -p -- "$LOG_DIR"
for output_parent in "$RUN" "$LOG_DIR"; do
  [[ -d "$output_parent" && ! -L "$output_parent" && "$(readlink -f -- "$output_parent")" == "$output_parent" ]] || \
    die "failed to create a canonical output parent: $output_parent"
done
readonly SHARD_CLAIM="$RUN/.shard-$SHARD_INDEX.claim"
readonly VERIFY_LOG="$LOG_DIR/shard-$SHARD_INDEX.lock-verification.log"
readonly VERIFY_PARTIAL="$VERIFY_LOG.partial.$$"
readonly VERIFY_FAILED="$LOG_DIR/shard-$SHARD_INDEX.lock-verification.failed.log"
[[ ! -e "$VERIFY_LOG" && ! -e "$VERIFY_PARTIAL" && ! -e "$VERIFY_FAILED" ]] || \
  die "shard lock-verification log already exists"

# Mandatory, non-mutating verification recomputes all 113 bindings and the
# canonical lock bytes from the immutable deployment and audited v1-v5 ancestry.
if env -i \
  HOME=/home/florianpfaff \
  USER=florianpfaff \
  LOGNAME=florianpfaff \
  PATH=/usr/local/bin:/usr/bin:/bin \
  TMPDIR=/tmp \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  PYTHONHASHSEED=0 \
  PYTHONPYCACHEPREFIX="$PYCACHE_PREFIX" \
  GIT_OPTIONAL_LOCKS=0 \
  "$PY" -B -X "pycache_prefix=$PYCACHE_PREFIX" -I "$LOCK_OPERATOR" \
  --v1-lock "$V1_LOCK" \
  --v1-report "$V1_REPORT" \
  --v2-withdrawal-report "$V2_WITHDRAWAL_REPORT" \
  --v3-boundary-incident-report "$V3_BOUNDARY_INCIDENT_REPORT" \
  --v4-lock "$V4_LOCK" \
  --v4-execution-withdrawal-report "$V4_EXECUTION_WITHDRAWAL_REPORT" \
  --v5-lock "$V5_LOCK" \
  --v5-outcome-withdrawal-report "$V5_OUTCOME_WITHDRAWAL_REPORT" \
  --deployed-code "$BPT_HELD_V6_CODE" \
  --output-lock "$CALIBRATION_LOCK" \
  --verify-existing-lock \
  >"$VERIFY_PARTIAL" 2>&1; then
  chmod 600 -- "$VERIFY_PARTIAL"
  mv -- "$VERIFY_PARTIAL" "$VERIFY_LOG"
else
  status="$?"
  chmod 600 -- "$VERIFY_PARTIAL"
  mv -- "$VERIFY_PARTIAL" "$VERIFY_FAILED"
  echo "mandatory lock verification failed; log retained at $VERIFY_FAILED" >&2
  exit "$status"
fi

if ! env -i \
  HOME=/home/florianpfaff \
  USER=florianpfaff \
  LOGNAME=florianpfaff \
  PATH=/usr/local/bin:/usr/bin:/bin \
  TMPDIR=/tmp \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  PYTHONHASHSEED=0 \
  PYTHONPYCACHEPREFIX="$PYCACHE_PREFIX" \
  "$PY" -B -X "pycache_prefix=$PYCACHE_PREFIX" -I - "$BPT_HELD_V6_CODE/src" "$CONFIRMATION_LOCK" "$CALIBRATION_LOCK" "$CALIBRATION_DECISION" \
  >>"$VERIFY_LOG" 2>&1 <<'PY'
import json
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
from bayesian_phystwin.deform360_held_protocol import load_held_protocol_lock

lock = load_held_protocol_lock(sys.argv[2])
if lock.get("stage") != "confirmation":
    raise RuntimeError("held-v6 confirmation lock has another stage")
if Path(lock["parent_calibration_lock"]["path"]) != Path(sys.argv[3]):
    raise RuntimeError("confirmation lock binds another calibration parent")
if Path(lock["calibration_gate_evidence"]["path"]) != Path(sys.argv[4]):
    raise RuntimeError("confirmation lock binds another calibration decision")
print(json.dumps({"artifact_sha256": lock["artifact_sha256"]}, sort_keys=True))
PY
then
  die "derived confirmation-lock verification failed"
fi

readonly BPT_HELD_V6_CONFIRMATION_LOCK_VERIFIED_SHA256="$(sha256sum -- "$CONFIRMATION_LOCK" | awk '{print $1}')"
[[ "$BPT_HELD_V6_CONFIRMATION_LOCK_VERIFIED_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "invalid verified confirmation-lock digest"
export BPT_HELD_V6_CODE BPT_HELD_V6_CONFIRMATION_LOCK_VERIFIED_SHA256

# Refuse a partial/reused v6 shard up front.  No v1, v2, or v3 prediction path is
# ever considered as an input or resume source.
for spec in "${CASE_SPECS[@]}"; do
  IFS=: read -r case_name object_id episode_id <<<"$spec"
  [[ "$case_name:$object_id:$episode_id" == "$spec" ]] || die "malformed case tuple: $spec"
  [[ ! -e "$RUN/cases/$case_name" && ! -L "$RUN/cases/$case_name" ]] || \
    die "v6 case output already exists: $case_name"
  if compgen -G "$LOG_DIR/$case_name.*" >/dev/null; then
    die "v6 case log already exists: $case_name"
  fi
done

# Claim only after every non-mutating verifier and stale-output preflight has
# succeeded.  Keep the claim after formal start; a held shard is never resumed.
mkdir -- "$SHARD_CLAIM" 2>/dev/null || die "this v6 shard was already claimed"

echo "SHARD_START shard=$SHARD_INDEX cuda_device=$CUDA_DEVICE case_count=${#CASE_SPECS[@]} lock_sha256=$BPT_HELD_V6_CONFIRMATION_LOCK_VERIFIED_SHA256"
for spec in "${CASE_SPECS[@]}"; do
  IFS=: read -r case_name object_id episode_id <<<"$spec"
  env -i \
    HOME=/home/florianpfaff \
    USER=florianpfaff \
    LOGNAME=florianpfaff \
    PATH="$PATH" \
    TMPDIR=/tmp \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONPYCACHEPREFIX="$PYCACHE_PREFIX" \
    BPT_HELD_V6_CODE="$BPT_HELD_V6_CODE" \
    BPT_HELD_V6_CONFIRMATION_LOCK_VERIFIED_SHA256="$BPT_HELD_V6_CONFIRMATION_LOCK_VERIFIED_SHA256" \
    /bin/bash "$CASE_RUNNER" "$CUDA_DEVICE" "$case_name" "$object_id" "$episode_id"
done
echo "SHARD_COMPLETE shard=$SHARD_INDEX cuda_device=$CUDA_DEVICE case_count=${#CASE_SPECS[@]}"
