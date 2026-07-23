#!/bin/bash

if [[ "${BPT_HELD_V8_CONFIRMATION_SHARD_ENV_NORMALIZED:-}" != "1" ]]; then
  exec env -i \
    HOME=/home/florianpfaff USER=florianpfaff LOGNAME=florianpfaff \
    PATH=/usr/local/bin:/usr/bin:/bin TMPDIR=/tmp LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    BPT_HELD_V8_CODE="${BPT_HELD_V8_CODE:-}" \
    BPT_HELD_V8_CONFIRMATION_SHARD_ENV_NORMALIZED=1 \
    /bin/bash "$0" "$@"
fi

set -Eeuo pipefail
umask 077
export PATH=/usr/local/bin:/usr/bin:/bin
unset BASH_ENV ENV CDPATH

die() { echo "ERROR: $*" >&2; exit 2; }
while IFS='=' read -r name _value; do
  case "$name" in
    HOME|USER|LOGNAME|PATH|TMPDIR|LANG|LC_ALL|PWD|SHLVL|_|BPT_HELD_V8_CODE|\
    BPT_HELD_V8_CONFIRMATION_SHARD_ENV_NORMALIZED) ;;
    *) die "normalized confirmation-shard environment contains $name" ;;
  esac
done < <(env)

[[ "$#" -eq 3 ]] || die "usage: run_deform360_v8_confirmation_shard.sh SHARD_INDEX CUDA_DEVICE CONFIRMATION_SOURCE_MANIFEST"
readonly SHARD_INDEX="$1" CUDA_DEVICE="$2" CONFIRMATION_SOURCE_MANIFEST="$3"
case "$SHARD_INDEX:$CUDA_DEVICE" in 0:0|1:1) ;; *) die "formal shards are fixed to 0:0 and 1:1" ;; esac
[[ "$(hostname)" == "workstation2" ]] || \
  die "formal held-v8 shards must run together on gpuserver6000/workstation2"

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
[[ "${#ALL_CASE_SPECS[@]}" -eq 6 ]] || die "confirmation cohort cardinality changed"
[[ "${#SHARD_0_CASE_SPECS[@]}" -eq 3 && "${#SHARD_1_CASE_SPECS[@]}" -eq 3 ]] || \
  die "confirmation shard cardinalities changed"
declare -A COUNTS=()
for spec in "${ALL_CASE_SPECS[@]}"; do
  [[ ! -v "COUNTS[$spec]" ]] || die "duplicate confirmation cohort case: $spec"
  COUNTS["$spec"]=0
done
for spec in "${SHARD_0_CASE_SPECS[@]}" "${SHARD_1_CASE_SPECS[@]}"; do
  [[ -v "COUNTS[$spec]" ]] || die "confirmation shard contains non-cohort case: $spec"
  COUNTS["$spec"]=$((COUNTS["$spec"] + 1))
done
for spec in "${ALL_CASE_SPECS[@]}"; do
  [[ "${COUNTS[$spec]}" -eq 1 ]] || die "confirmation case missing or duplicated: $spec"
done
if [[ "$SHARD_INDEX" == "0" ]]; then
  readonly -a CASE_SPECS=("${SHARD_0_CASE_SPECS[@]}")
else
  readonly -a CASE_SPECS=("${SHARD_1_CASE_SPECS[@]}")
fi

readonly HELD="/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v8"
readonly LOCK="$HELD/confirmation-lock.json"
readonly CANONICAL_CONFIRMATION_SOURCE_MANIFEST="$HELD/confirmation-source/manifests/aligned-source-cohort.json"
readonly CODE="${BPT_HELD_V8_CODE:?set BPT_HELD_V8_CODE to immutable v8 deployment}"
readonly PY="/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004/bin/python"
readonly PYCACHE_PREFIX="/nonexistent/bpt-held-v8-pycache"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SHARD_RUNNER="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
readonly CASE_RUNNER="$SCRIPT_DIR/run_deform360_v8_confirmation_case.sh"
readonly RUN="$HELD/confirmation"
readonly LOG_DIR="$RUN/logs"
readonly CLAIM="$RUN/.shard-$SHARD_INDEX.claim"

[[ -d "$CODE" && ! -L "$CODE" && "$(readlink -f -- "$CODE")" == "$CODE" ]] || \
  die "immutable v8 deployment is absent, linked, or non-canonical"
[[ -d "$HELD" && ! -L "$HELD" && "$(readlink -f -- "$HELD")" == "$HELD" ]] || \
  die "held-v8 root is absent, linked, or non-canonical"
[[ "$CODE" =~ ^${HELD}/code-([0-9a-f]{40}|[0-9a-f]{64})$ ]] || die "deployment path changed"
[[ "$SCRIPT_DIR" == "$CODE/scripts/held" ]] || die "shard is outside v8 deployment"
for operator in "$SHARD_RUNNER" "$CASE_RUNNER" "$SCRIPT_DIR/run_deform360_v8_case_common.sh"; do
  [[ -f "$operator" && ! -L "$operator" && "$(readlink -f -- "$operator")" == "$operator" ]] || \
    die "v8 runner is absent, linked, or non-canonical: $operator"
done
[[ -z "$(find "$CODE" -xdev -perm /222 -print -quit)" ]] || die "v8 deployment is writable"
[[ -f "$LOCK" && ! -L "$LOCK" && "$(stat -c '%a' -- "$LOCK")" == "400" ]] || \
  die "confirmation lock is absent, linked, or not mode 0400"
[[ "$CONFIRMATION_SOURCE_MANIFEST" == "$CANONICAL_CONFIRMATION_SOURCE_MANIFEST" ]] || \
  die "confirmation source manifest path changed"
[[ -f "$CONFIRMATION_SOURCE_MANIFEST" && ! -L "$CONFIRMATION_SOURCE_MANIFEST" && \
  "$(stat -c '%a' -- "$CONFIRMATION_SOURCE_MANIFEST")" == "400" ]] || \
  die "confirmation source manifest is absent, linked, or not mode 0400"
[[ ! -e /nonexistent && ! -L /nonexistent && ! -e "$PYCACHE_PREFIX" ]] || \
  die "reserved v8 pycache prefix is available"
cd -- "$CODE"

# A shard creates no role output until the entire exact-six source cohort has
# been recursively rehashed against the canonical confirmation lock.
readonly SOURCE_VALIDATION="$(
  env -i HOME=/home/florianpfaff USER=florianpfaff LOGNAME=florianpfaff \
    PATH=/usr/local/bin:/usr/bin:/bin TMPDIR=/tmp LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 \
    PYTHONPYCACHEPREFIX="$PYCACHE_PREFIX" \
    "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" - \
    "$CODE/src" "$LOCK" "$CONFIRMATION_SOURCE_MANIFEST" <<'PY'
import hashlib
from pathlib import Path
import sys
sys.path.insert(0, sys.argv.pop(1))
from bayesian_phystwin.deform360_held_v8_protocol import (
    confirmation_source_permit_evidence,
    validate_protocol_lock,
)
from bayesian_phystwin.deform360_held_v8_confirmation_source import (
    validate_confirmation_source_cohort_manifest,
)
lock = validate_protocol_lock(sys.argv[1])
if lock.get("stage") != "confirmation":
    raise RuntimeError("confirmation lock did not recursively validate GO")
validate_confirmation_source_cohort_manifest(
    sys.argv[2],
    expected_source_permit=confirmation_source_permit_evidence(sys.argv[1]),
    verify_content=True,
)
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
[[ "$SOURCE_VALIDATION" =~ ^[0-9a-f]{64}$ ]] || \
  die "confirmation source validation did not return the lock digest"

readonly VERIFY_PARTIAL="$RUN/shard-$SHARD_INDEX.lock-verify.partial.$$"
readonly VERIFY_LOG="$RUN/shard-$SHARD_INDEX.lock-verify.log"
for parent in "$RUN" "$LOG_DIR"; do
  if [[ -e "$parent" || -L "$parent" ]]; then
    [[ -d "$parent" && ! -L "$parent" && "$(readlink -f -- "$parent")" == "$parent" ]] || \
      die "confirmation output parent is linked or non-canonical: $parent"
  fi
done
mkdir -p -- "$RUN" "$LOG_DIR"
for parent in "$RUN" "$LOG_DIR"; do
  [[ -d "$parent" && ! -L "$parent" && "$(readlink -f -- "$parent")" == "$parent" ]] || \
    die "failed to create a canonical confirmation output parent: $parent"
done
[[ ! -e "$VERIFY_PARTIAL" && ! -e "$VERIFY_LOG" && ! -e "$CLAIM" && ! -L "$CLAIM" ]] || \
  die "confirmation shard already started"
if env -i HOME=/home/florianpfaff USER=florianpfaff LOGNAME=florianpfaff \
  PATH=/usr/local/bin:/usr/bin:/bin TMPDIR=/tmp LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 \
  PYTHONPYCACHEPREFIX="$PYCACHE_PREFIX" \
  "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" - \
  "$CODE/src" "$LOCK" >"$VERIFY_PARTIAL" 2>&1 <<'PY'
import hashlib
from pathlib import Path
import sys
sys.path.insert(0, sys.argv.pop(1))
from bayesian_phystwin.deform360_held_v8_protocol import (
    CONFIRMATION_CASE_NAMES,
    locked_case_names,
    validate_protocol_lock,
)
lock = validate_protocol_lock(sys.argv[1])
if lock["protocol_id"] != "deform360-held-online-belief-v8.1":
    raise RuntimeError("not a v8 confirmation lock")
if tuple(locked_case_names(sys.argv[1], role="confirmation")) != tuple(CONFIRMATION_CASE_NAMES):
    raise RuntimeError("locked confirmation cohort changed")
# validate_protocol_lock recursively validates the parent calibration lock and
# its sealed GO decision before it returns a confirmation-stage lock.
if lock.get("stage") != "confirmation" or lock.get("confirmation_access_authorized") is not True:
    raise RuntimeError("confirmation lock lacks a recursively validated calibration GO")
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
then
  chmod 600 -- "$VERIFY_PARTIAL"
  mv -- "$VERIFY_PARTIAL" "$VERIFY_LOG"
else
  status="$?"
  chmod 600 -- "$VERIFY_PARTIAL"
  mv -- "$VERIFY_PARTIAL" "$VERIFY_LOG.failed"
  exit "$status"
fi
readonly BPT_HELD_V8_LOCK_VERIFIED_SHA256="$(tail -n 1 -- "$VERIFY_LOG")"
[[ "$BPT_HELD_V8_LOCK_VERIFIED_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "invalid verified lock hash"

for spec in "${CASE_SPECS[@]}"; do
  IFS=: read -r case_name object_id episode_id <<<"$spec"
  [[ "$case_name:$object_id:$episode_id" == "$spec" ]] || die "malformed tuple: $spec"
  [[ ! -e "$RUN/cases/$case_name" && ! -L "$RUN/cases/$case_name" ]] || \
    die "fresh confirmation case already exists: $case_name"
done
mkdir -- "$CLAIM" 2>/dev/null || die "confirmation shard was already claimed"

echo "SHARD_START role=confirmation shard=$SHARD_INDEX cuda_device=$CUDA_DEVICE case_count=${#CASE_SPECS[@]}"
for spec in "${CASE_SPECS[@]}"; do
  IFS=: read -r case_name object_id episode_id <<<"$spec"
  env -i HOME=/home/florianpfaff USER=florianpfaff LOGNAME=florianpfaff \
    PATH=/usr/local/bin:/usr/bin:/bin TMPDIR=/tmp LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    BPT_HELD_V8_CODE="$CODE" \
    BPT_HELD_V8_LOCK_VERIFIED_SHA256="$BPT_HELD_V8_LOCK_VERIFIED_SHA256" \
    /bin/bash "$CASE_RUNNER" "$CUDA_DEVICE" "$case_name" "$object_id" "$episode_id" \
      "$CONFIRMATION_SOURCE_MANIFEST"
done
echo "SHARD_COMPLETE role=confirmation shard=$SHARD_INDEX cuda_device=$CUDA_DEVICE case_count=${#CASE_SPECS[@]}"
