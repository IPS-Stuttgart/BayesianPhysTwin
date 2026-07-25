#!/bin/bash

if [[ "${BPT_HELD_V8_CALIBRATION_SHARD_ENV_NORMALIZED:-}" != "1" ]]; then
  exec env -i \
    HOME=/home/florianpfaff USER=florianpfaff LOGNAME=florianpfaff \
    PATH=/usr/local/bin:/usr/bin:/bin TMPDIR=/tmp LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    BPT_HELD_V8_CODE="${BPT_HELD_V8_CODE:-}" \
    BPT_HELD_V8_CALIBRATION_SHARD_ENV_NORMALIZED=1 \
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
    BPT_HELD_V8_CALIBRATION_SHARD_ENV_NORMALIZED) ;;
    *) die "normalized calibration-shard environment contains $name" ;;
  esac
done < <(env)

[[ "$#" -eq 3 ]] || \
  die "usage: run_deform360_v8_calibration_shard.sh SHARD_INDEX CUDA_DEVICE REPLACEMENT_SOURCE_MANIFEST"
readonly SHARD_INDEX="$1" CUDA_DEVICE="$2" REPLACEMENT_SOURCE_MANIFEST="$3"
case "$SHARD_INDEX:$CUDA_DEVICE" in 0:0|1:1) ;; *) die "formal shards are fixed to 0:0 and 1:1" ;; esac
[[ "$(hostname)" == "workstation2" ]] || \
  die "formal held-v8 shards must run together on gpuserver6000/workstation2"

readonly -a ALL_CASE_SPECS=(
  "072-cotton-clohesline-ep0003:072-cotton-clohesline:0003"
  "002-rope-silk-ep0004:002-rope-silk:0004"
  "002-rope-silk-ep0008:002-rope-silk:0008"
  "083-blanket-cloth-ep0000:083-blanket-cloth:0000"
  "083-blanket-cloth-ep0003:083-blanket-cloth:0003"
  "083-blanket-cloth-ep0006:083-blanket-cloth:0006"
  "085-scarf-cloth-ep0000:085-scarf-cloth:0000"
  "085-scarf-cloth-ep0005:085-scarf-cloth:0005"
  "085-scarf-cloth-ep0007:085-scarf-cloth:0007"
  "092-squirrel-ep0002:092-squirrel:0002"
  "092-squirrel-ep0003:092-squirrel:0003"
  "092-squirrel-ep0006:092-squirrel:0006"
  "170-spider-ep0002:170-spider:0002"
  "170-spider-ep0004:170-spider:0004"
  "170-spider-ep0007:170-spider:0007"
)
readonly -a SHARD_0_CASE_SPECS=(
  "072-cotton-clohesline-ep0003:072-cotton-clohesline:0003"
  "002-rope-silk-ep0004:002-rope-silk:0004"
  "002-rope-silk-ep0008:002-rope-silk:0008"
  "083-blanket-cloth-ep0000:083-blanket-cloth:0000"
  "085-scarf-cloth-ep0000:085-scarf-cloth:0000"
  "085-scarf-cloth-ep0005:085-scarf-cloth:0005"
  "085-scarf-cloth-ep0007:085-scarf-cloth:0007"
  "170-spider-ep0002:170-spider:0002"
)
readonly -a SHARD_1_CASE_SPECS=(
  "083-blanket-cloth-ep0003:083-blanket-cloth:0003"
  "083-blanket-cloth-ep0006:083-blanket-cloth:0006"
  "092-squirrel-ep0002:092-squirrel:0002"
  "092-squirrel-ep0003:092-squirrel:0003"
  "092-squirrel-ep0006:092-squirrel:0006"
  "170-spider-ep0004:170-spider:0004"
  "170-spider-ep0007:170-spider:0007"
)
[[ "${#ALL_CASE_SPECS[@]}" -eq 15 ]] || die "calibration cohort cardinality changed"
[[ "${#SHARD_0_CASE_SPECS[@]}" -eq 8 && "${#SHARD_1_CASE_SPECS[@]}" -eq 7 ]] || \
  die "calibration shard cardinalities changed"
declare -A COUNTS=()
for spec in "${ALL_CASE_SPECS[@]}"; do
  [[ ! -v "COUNTS[$spec]" ]] || die "duplicate calibration cohort case: $spec"
  COUNTS["$spec"]=0
done
for spec in "${SHARD_0_CASE_SPECS[@]}" "${SHARD_1_CASE_SPECS[@]}"; do
  [[ -v "COUNTS[$spec]" ]] || die "calibration shard contains a non-cohort case: $spec"
  COUNTS["$spec"]=$((COUNTS["$spec"] + 1))
done
for spec in "${ALL_CASE_SPECS[@]}"; do
  [[ "${COUNTS[$spec]}" -eq 1 ]] || die "calibration case missing or duplicated: $spec"
done
if [[ "$SHARD_INDEX" == "0" ]]; then
  readonly -a CASE_SPECS=("${SHARD_0_CASE_SPECS[@]}")
else
  readonly -a CASE_SPECS=("${SHARD_1_CASE_SPECS[@]}")
fi

readonly HELD="/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v83"
readonly LOCK="$HELD/calibration-lock.json"
readonly CODE="${BPT_HELD_V8_CODE:?set BPT_HELD_V8_CODE to immutable v8 deployment}"
readonly PY="/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004/bin/python"
readonly PYCACHE_PREFIX="/nonexistent/bpt-held-v83-pycache"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SHARD_RUNNER="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
readonly CASE_RUNNER="$SCRIPT_DIR/run_deform360_v8_calibration_case.sh"
readonly RUN="$HELD/calibration"
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
  die "calibration lock is absent, linked, or not mode 0400"
[[ -f "$REPLACEMENT_SOURCE_MANIFEST" && ! -L "$REPLACEMENT_SOURCE_MANIFEST" && \
   "$(stat -c '%a' -- "$REPLACEMENT_SOURCE_MANIFEST")" == "400" ]] || \
  die "replacement source manifest is absent, linked, or not mode 0400"
[[ ! -e /nonexistent && ! -L /nonexistent && ! -e "$PYCACHE_PREFIX" ]] || \
  die "reserved v8 pycache prefix is available"
cd -- "$CODE"

readonly VERIFY_PARTIAL="$RUN/shard-$SHARD_INDEX.lock-verify.partial.$$"
readonly VERIFY_LOG="$RUN/shard-$SHARD_INDEX.lock-verify.log"
for parent in "$RUN" "$LOG_DIR"; do
  if [[ -e "$parent" || -L "$parent" ]]; then
    [[ -d "$parent" && ! -L "$parent" && "$(readlink -f -- "$parent")" == "$parent" ]] || \
      die "calibration output parent is linked or non-canonical: $parent"
  fi
done
mkdir -p -- "$RUN" "$LOG_DIR"
for parent in "$RUN" "$LOG_DIR"; do
  [[ -d "$parent" && ! -L "$parent" && "$(readlink -f -- "$parent")" == "$parent" ]] || \
    die "failed to create a canonical calibration output parent: $parent"
done
[[ ! -e "$VERIFY_PARTIAL" && ! -e "$VERIFY_LOG" && ! -e "$CLAIM" && ! -L "$CLAIM" ]] || \
  die "calibration shard already started"
if env -i HOME=/home/florianpfaff USER=florianpfaff LOGNAME=florianpfaff \
  PATH=/usr/local/bin:/usr/bin:/bin TMPDIR=/tmp LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 \
  PYTHONPYCACHEPREFIX="$PYCACHE_PREFIX" \
  "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" - \
  "$CODE/src" "$LOCK" "$REPLACEMENT_SOURCE_MANIFEST" >"$VERIFY_PARTIAL" 2>&1 <<'PY'
import hashlib
from pathlib import Path
import sys
sys.path.insert(0, sys.argv.pop(1))
from bayesian_phystwin.deform360_held_v8_protocol import (
    CALIBRATION_CASE_NAMES,
    locked_case_names,
    replacement_source_permit_evidence,
    validate_protocol_lock,
)
from bayesian_phystwin.deform360_held_v8_replacement_source import validate_aligned_source_manifest
lock = validate_protocol_lock(sys.argv[1])
if lock["protocol_id"] != "deform360-held-online-belief-v8.3":
    raise RuntimeError("not a v8 calibration lock")
if tuple(locked_case_names(sys.argv[1], role="calibration")) != tuple(CALIBRATION_CASE_NAMES):
    raise RuntimeError("locked calibration cohort changed")
source = validate_aligned_source_manifest(
    sys.argv[2],
    expected_source_permit=replacement_source_permit_evidence(sys.argv[1]),
)
if source["case_name"] != "072-cotton-clohesline-ep0003":
    raise RuntimeError("replacement source case changed")
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
    die "fresh v8 case already exists: $case_name"
done
mkdir -- "$CLAIM" 2>/dev/null || die "calibration shard was already claimed"

echo "SHARD_START role=calibration shard=$SHARD_INDEX cuda_device=$CUDA_DEVICE case_count=${#CASE_SPECS[@]}"
for spec in "${CASE_SPECS[@]}"; do
  IFS=: read -r case_name object_id episode_id <<<"$spec"
  case_args=("$CUDA_DEVICE" "$case_name" "$object_id" "$episode_id")
  if [[ "$case_name" == "072-cotton-clohesline-ep0003" ]]; then
    env -i HOME=/home/florianpfaff USER=florianpfaff LOGNAME=florianpfaff \
      PATH=/usr/local/bin:/usr/bin:/bin TMPDIR=/tmp LANG=C.UTF-8 LC_ALL=C.UTF-8 \
      BPT_HELD_V8_CODE="$CODE" \
      BPT_HELD_V8_LOCK_VERIFIED_SHA256="$BPT_HELD_V8_LOCK_VERIFIED_SHA256" \
      /bin/bash "$CASE_RUNNER" "${case_args[@]}" "$REPLACEMENT_SOURCE_MANIFEST"
  else
    env -i HOME=/home/florianpfaff USER=florianpfaff LOGNAME=florianpfaff \
      PATH=/usr/local/bin:/usr/bin:/bin TMPDIR=/tmp LANG=C.UTF-8 LC_ALL=C.UTF-8 \
      BPT_HELD_V8_CODE="$CODE" \
      BPT_HELD_V8_LOCK_VERIFIED_SHA256="$BPT_HELD_V8_LOCK_VERIFIED_SHA256" \
      /bin/bash "$CASE_RUNNER" "${case_args[@]}"
  fi
done
echo "SHARD_COMPLETE role=calibration shard=$SHARD_INDEX cuda_device=$CUDA_DEVICE case_count=${#CASE_SPECS[@]}"
