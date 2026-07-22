#!/bin/bash

if [[ "${BPT_HELD_V8_CONFIRMATION_CASE_ENV_NORMALIZED:-}" != "1" ]]; then
  exec env -i \
    HOME=/home/florianpfaff USER=florianpfaff LOGNAME=florianpfaff \
    PATH=/usr/local/bin:/usr/bin:/bin TMPDIR=/tmp LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    BPT_HELD_V8_CODE="${BPT_HELD_V8_CODE:-}" \
    BPT_HELD_V8_LOCK_VERIFIED_SHA256="${BPT_HELD_V8_LOCK_VERIFIED_SHA256:-}" \
    BPT_HELD_V8_CONFIRMATION_CASE_ENV_NORMALIZED=1 \
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
    BPT_HELD_V8_LOCK_VERIFIED_SHA256|BPT_HELD_V8_CONFIRMATION_CASE_ENV_NORMALIZED) ;;
    *) die "normalized confirmation-case environment contains $name" ;;
  esac
done < <(env)

[[ "$#" -eq 4 ]] || \
  die "usage: run_deform360_v8_confirmation_case.sh CUDA_DEVICE CASE OBJECT EPISODE"
readonly CUDA_DEVICE="$1" CASE_NAME="$2" OBJECT="$3" EPISODE="$4"
readonly REPLACEMENT_SOURCE_MANIFEST=""

case "$CASE_NAME:$OBJECT:$EPISODE" in
  002-rope-silk-ep0001:002-rope-silk:0001|\
  081-stripe-rope-ep0005:081-stripe-rope:0005|\
  085-scarf-cloth-ep0002:085-scarf-cloth:0002|\
  083-blanket-cloth-ep0007:083-blanket-cloth:0007|\
  092-squirrel-ep0001:092-squirrel:0001|\
  170-spider-ep0006:170-spider:0006) ;;
  *) die "case tuple is outside the exact six-case v8 confirmation cohort" ;;
esac

readonly V8_ROLE="confirmation"
readonly CASE_RUNNER="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"
source "$(dirname -- "$CASE_RUNNER")/run_deform360_v8_case_common.sh"
