#!/bin/bash

if [[ "${BPT_HELD_V8_CALIBRATION_CASE_ENV_NORMALIZED:-}" != "1" ]]; then
  exec env -i \
    HOME=/home/florianpfaff USER=florianpfaff LOGNAME=florianpfaff \
    PATH=/usr/local/bin:/usr/bin:/bin TMPDIR=/tmp LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    BPT_HELD_V8_CODE="${BPT_HELD_V8_CODE:-}" \
    BPT_HELD_V8_LOCK_VERIFIED_SHA256="${BPT_HELD_V8_LOCK_VERIFIED_SHA256:-}" \
    BPT_HELD_V8_CALIBRATION_CASE_ENV_NORMALIZED=1 \
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
    BPT_HELD_V8_LOCK_VERIFIED_SHA256|BPT_HELD_V8_CALIBRATION_CASE_ENV_NORMALIZED) ;;
    *) die "normalized calibration-case environment contains $name" ;;
  esac
done < <(env)

[[ "$#" -ge 4 && "$#" -le 5 ]] || \
  die "usage: run_deform360_v8_calibration_case.sh CUDA_DEVICE CASE OBJECT EPISODE [REPLACEMENT_SOURCE_MANIFEST]"
readonly CUDA_DEVICE="$1" CASE_NAME="$2" OBJECT="$3" EPISODE="$4"
readonly REPLACEMENT_SOURCE_MANIFEST="${5:-}"
readonly CONFIRMATION_SOURCE_MANIFEST=""

case "$CASE_NAME:$OBJECT:$EPISODE" in
  072-cotton-clohesline-ep0003:072-cotton-clohesline:0003|\
  002-rope-silk-ep0004:002-rope-silk:0004|\
  002-rope-silk-ep0008:002-rope-silk:0008|\
  083-blanket-cloth-ep0000:083-blanket-cloth:0000|\
  083-blanket-cloth-ep0003:083-blanket-cloth:0003|\
  083-blanket-cloth-ep0006:083-blanket-cloth:0006|\
  085-scarf-cloth-ep0000:085-scarf-cloth:0000|\
  085-scarf-cloth-ep0005:085-scarf-cloth:0005|\
  085-scarf-cloth-ep0007:085-scarf-cloth:0007|\
  092-squirrel-ep0002:092-squirrel:0002|\
  092-squirrel-ep0003:092-squirrel:0003|\
  092-squirrel-ep0006:092-squirrel:0006|\
  170-spider-ep0002:170-spider:0002|\
  170-spider-ep0004:170-spider:0004|\
  170-spider-ep0007:170-spider:0007) ;;
  *) die "case tuple is outside the exact fresh 15-case v8 calibration cohort" ;;
esac
if [[ "$CASE_NAME" == "072-cotton-clohesline-ep0003" ]]; then
  [[ "$#" -eq 5 && -n "$REPLACEMENT_SOURCE_MANIFEST" ]] || \
    die "replacement case requires its explicit aligned-source manifest"
else
  [[ "$#" -eq 4 ]] || die "non-replacement case rejects a source manifest"
fi

readonly V8_ROLE="calibration"
readonly CASE_RUNNER="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"
source "$(dirname -- "$CASE_RUNNER")/run_deform360_v8_case_common.sh"
