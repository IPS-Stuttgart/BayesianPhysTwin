#!/bin/bash

# Shared implementation for the two exact-cohort v8 case entry points.  This
# file may only be sourced by an immutable calibration/confirmation wrapper.
[[ -n "${V8_ROLE:-}" && -n "${CASE_RUNNER:-}" ]] || {
  echo "ERROR: v8 case common code is not a direct entry point" >&2
  exit 2
}

set -Eeuo pipefail
umask 077
export PATH=/usr/local/bin:/usr/bin:/bin
unset BASH_ENV ENV CDPATH

die() {
  echo "ERROR: $*" >&2
  exit 2
}

case "$V8_ROLE" in
  calibration|confirmation) ;;
  *) die "invalid v8 case role" ;;
esac

readonly HELD="/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v8"
readonly CODE="${BPT_HELD_V8_CODE:?set BPT_HELD_V8_CODE to the immutable v8 deployment}"
readonly VERIFIED_LOCK_SHA256="${BPT_HELD_V8_LOCK_VERIFIED_SHA256:?run cases through a verified v8 shard}"
readonly PY="/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004/bin/python"
readonly PYCACHE_PREFIX="/nonexistent/bpt-held-v8-pycache"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly COMMON_RUNNER="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
readonly ALIGNED="/mnt/lexar4tb/datasets/deform360/data-7fea8e2/replication-v1/aligned"
readonly SAM2="/mnt/lexar4tb/datasets/deform360/sam2-2b90b9f5"
readonly SAM2_CHECKPOINT="$SAM2/checkpoints/sam2.1_hiera_small.pt"
readonly UPSTREAM="/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/Bayesian-PhysTwin-upstream-58ab4808e59d"
readonly OFFICIAL="/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/PhysTwin-upstream-2b6630528141"
readonly DEFORM360="/mnt/lexar4tb/datasets/deform360/code"
readonly SEMANTIC_MODEL="/mnt/corsair/florianpfaff/model-cache/siglip2-base-patch16-224-75de2d55"
readonly SEMANTIC_MODEL_LOCK="/mnt/corsair/florianpfaff/bpt-framezero-field-dev-20260720/scratch_siglip2_model_lock.json"
readonly ALLTRACKER="/mnt/corsair/florianpfaff/alltracker-molmomotion-61f5b21"
readonly ALLTRACKER_CHECKPOINT="/mnt/corsair/florianpfaff/model-cache/alltracker.pth"
readonly DEVELOPMENT_DECISION="/mnt/corsair/florianpfaff/bpt-online-belief-v1/runs/deform360-query-field-open27-v1-development/decision.json"
readonly DEVELOPMENT_DECISION_SHA256="110b3c1831898ff6b333f35236401761222f85eafac1dcbcea7b7183d5b434bd"

if [[ "$V8_ROLE" == "calibration" ]]; then
  readonly LOCK="$HELD/calibration-lock.json"
  readonly RUN="$HELD/calibration"
else
  readonly LOCK="$HELD/confirmation-lock.json"
  readonly RUN="$HELD/confirmation"
fi
readonly LOG_DIR="$RUN/logs"
readonly ROOT="$RUN/cases/$CASE_NAME"
readonly FZ="$ROOT/frame-zero"
readonly PHYS="$ROOT/physical"
readonly AUTH="$ROOT/prefix-authorization.json"
readonly ONLINE="$ROOT/online"
readonly FIELD="$ROOT/frozen-field"
readonly FROZEN_FIELD_MANIFEST="$FIELD/preoutcome-frozen-field-manifest.json"
readonly FZ_MANIFEST="$FZ/frame_zero_bundle.manifest.json"
readonly PHYSICAL_SEAL="$PHYS/physical_prior_seal.json"
readonly ONLINE_ARCHIVE="$ONLINE/online_prediction.npz"
readonly ONLINE_SEAL="$ONLINE/online_prediction_seal.json"

[[ "$(hostname)" == "workstation2" ]] || \
  die "formal held-v8 cases must run on gpuserver6000/workstation2"
case "$CUDA_DEVICE" in 0|1) ;; *) die "formal CUDA device must be 0 or 1" ;; esac
[[ "$VERIFIED_LOCK_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "invalid verified lock digest"
[[ -d "$HELD" && ! -L "$HELD" && "$(readlink -f -- "$HELD")" == "$HELD" ]] || \
  die "held-v8 root is absent, linked, or non-canonical"
[[ -d "$CODE" && ! -L "$CODE" && "$(readlink -f -- "$CODE")" == "$CODE" ]] || \
  die "v8 deployed code is absent, linked, or non-canonical"
[[ "$CODE" =~ ^${HELD}/code-([0-9a-f]{40}|[0-9a-f]{64})$ ]] || \
  die "v8 deployed code is outside the canonical immutable snapshot path"
[[ "$SCRIPT_DIR" == "$CODE/scripts/held" ]] || die "case code is outside the deployment"
[[ "$CASE_RUNNER" == "$SCRIPT_DIR/run_deform360_v8_${V8_ROLE}_case.sh" ]] || \
  die "case wrapper path changed"
[[ -f "$COMMON_RUNNER" && ! -L "$COMMON_RUNNER" ]] || die "common runner is absent or linked"
[[ -z "$(find "$CODE" -xdev -perm /222 -print -quit)" ]] || \
  die "v8 deployed code tree is writable"
[[ -f "$LOCK" && ! -L "$LOCK" && "$(stat -c '%a' -- "$LOCK")" == "400" ]] || \
  die "v8 role lock is absent, linked, or not mode 0400"
[[ "$(sha256sum -- "$LOCK" | awk '{print $1}')" == "$VERIFIED_LOCK_SHA256" ]] || \
  die "v8 role lock changed after shard verification"
[[ -x "$PY" ]] || die "locked Python runtime is not executable"
[[ ! -e /nonexistent && ! -L /nonexistent && ! -e "$PYCACHE_PREFIX" && ! -L "$PYCACHE_PREFIX" ]] || \
  die "reserved held-v8 pycache prefix is available"
[[ -f "$DEVELOPMENT_DECISION" && ! -L "$DEVELOPMENT_DECISION" ]] || \
  die "frozen open27 decision is absent or linked"
[[ "$(stat -c '%a' -- "$DEVELOPMENT_DECISION")" == "400" ]] || \
  die "frozen open27 decision is not mode 0400"
[[ "$(sha256sum -- "$DEVELOPMENT_DECISION" | awk '{print $1}')" == "$DEVELOPMENT_DECISION_SHA256" ]] || \
  die "frozen open27 decision changed"
cd -- "$CODE"
[[ "$PWD" == "$CODE" ]] || die "failed to enter immutable v8 deployment"

readonly -a CLEAN_ENV=(
  env -i
  HOME=/home/florianpfaff
  USER=florianpfaff
  LOGNAME=florianpfaff
  PATH=/usr/local/bin:/usr/bin:/bin
  TMPDIR=/tmp
  LANG=C.UTF-8
  LC_ALL=C.UTF-8
  PYTHONDONTWRITEBYTECODE=1
  PYTHONNOUSERSITE=1
  PYTHONSAFEPATH=1
  PYTHONHASHSEED=0
  PYTHONPYCACHEPREFIX="$PYCACHE_PREFIX"
  PYNPUT_BACKEND=dummy
  PYOPENGL_PLATFORM=egl
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  WANDB_MODE=disabled
  CUBLAS_WORKSPACE_CONFIG=:4096:8
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
)

# Revalidate the v8 lock and exact case membership in an isolated interpreter.
"${CLEAN_ENV[@]}" "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" - \
  "$CODE/src" "$LOCK" "$CASE_NAME" "$V8_ROLE" <<'PY'
import sys
sys.path.insert(0, sys.argv.pop(1))
from bayesian_phystwin.deform360_held_v8_protocol import (
    locked_case_names,
    validate_protocol_lock,
)
lock = validate_protocol_lock(sys.argv[1])
case_name, role = sys.argv[2:4]
if lock["protocol_id"] != "deform360-held-online-belief-v8":
    raise RuntimeError("lock is not held-v8")
if tuple(locked_case_names(sys.argv[1], role=role)).count(case_name) != 1:
    raise RuntimeError("case is not present exactly once in the locked role")
PY
[[ "$(sha256sum -- "$LOCK" | awk '{print $1}')" == "$VERIFIED_LOCK_SHA256" ]] || \
  die "v8 role lock changed during case-level validation"

# The replacement source is validated before an RGB frame is decoded.  The
# other fourteen calibration cases and all confirmation cases reject a source
# manifest argument, so no source can be substituted implicitly.
if [[ "$CASE_NAME" == "072-cotton-clohesline-ep0003" ]]; then
  [[ "$V8_ROLE" == "calibration" ]] || die "replacement case cannot be confirmation"
  [[ -n "$REPLACEMENT_SOURCE_MANIFEST" ]] || die "replacement source manifest is required"
  [[ -f "$REPLACEMENT_SOURCE_MANIFEST" && ! -L "$REPLACEMENT_SOURCE_MANIFEST" ]] || \
    die "replacement source manifest is absent or linked"
  [[ "$(stat -c '%a' -- "$REPLACEMENT_SOURCE_MANIFEST")" == "400" ]] || \
    die "replacement source manifest is not sealed mode 0400"
  EPDIR="$("${CLEAN_ENV[@]}" "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" - \
    "$CODE/src" "$LOCK" "$REPLACEMENT_SOURCE_MANIFEST" <<'PY'
import sys
sys.path.insert(0, sys.argv.pop(1))
from bayesian_phystwin.deform360_held_v8_protocol import (
    replacement_source_permit_evidence,
)
from bayesian_phystwin.deform360_held_v8_replacement_source import (
    validate_aligned_source_manifest,
)
manifest = validate_aligned_source_manifest(
    sys.argv[2],
    expected_source_permit=replacement_source_permit_evidence(sys.argv[1]),
)
expected = {
    "semantic_label": "rope",
    "action": "drag",
    "action_location": "center",
    "bimanual": False,
    "prehensile": True,
}
if manifest["case_name"] != "072-cotton-clohesline-ep0003":
    raise RuntimeError("replacement source binds another case")
if manifest["object_id"] != "072-cotton-clohesline" or manifest["episode_id"] != "0003":
    raise RuntimeError("replacement source tuple changed")
if manifest["semantics"] != expected:
    raise RuntimeError("replacement source semantics changed")
print(manifest["aligned_episode_dir"])
PY
  )"
else
  [[ -z "$REPLACEMENT_SOURCE_MANIFEST" ]] || \
    die "replacement source manifest supplied for a non-replacement case"
  EPDIR="$ALIGNED/$OBJECT/episode_$EPISODE"
fi
readonly EPDIR
readonly OBJECT_DIR="$(dirname -- "$EPDIR")"
[[ -d "$EPDIR" && ! -L "$EPDIR" && "$(readlink -f -- "$EPDIR")" == "$EPDIR" ]] || \
  die "aligned episode directory is absent, linked, or non-canonical"
[[ "$(basename -- "$OBJECT_DIR")" == "$OBJECT" && "$(basename -- "$EPDIR")" == "episode_$EPISODE" ]] || \
  die "aligned episode mapping changed"
[[ -f "$SAM2_CHECKPOINT" ]] || die "SAM2 checkpoint is absent"
[[ -d "$UPSTREAM" && -d "$OFFICIAL" && -f "$OFFICIAL/configs/real.yaml" ]] || \
  die "frozen physical runtime is absent"
[[ -d "$DEFORM360" && -d "$SEMANTIC_MODEL" && -f "$SEMANTIC_MODEL_LOCK" ]] || \
  die "frozen frame-zero runtime is absent"
[[ -d "$ALLTRACKER" && -f "$ALLTRACKER_CHECKPOINT" ]] || \
  die "frozen online runtime is absent"

for parent in "$RUN" "$RUN/cases" "$LOG_DIR"; do
  if [[ -e "$parent" || -L "$parent" ]]; then
    [[ -d "$parent" && ! -L "$parent" && "$(readlink -f -- "$parent")" == "$parent" ]] || \
      die "output parent is linked or non-canonical: $parent"
  fi
done
[[ ! -e "$ROOT" && ! -L "$ROOT" ]] || die "fresh v8 case root already exists"
mkdir -p -- "$RUN/cases" "$LOG_DIR"
for parent in "$RUN" "$RUN/cases" "$LOG_DIR"; do
  [[ -d "$parent" && ! -L "$parent" && "$(readlink -f -- "$parent")" == "$parent" ]] || \
    die "failed to create a canonical output parent: $parent"
done
if compgen -G "$LOG_DIR/$CASE_NAME.*" >/dev/null; then
  die "case log already exists"
fi
mkdir -- "$ROOT"

CURRENT_PHASE="initialization"
case_exit() {
  local status="$?"
  trap - EXIT
  if [[ "$status" -ne 0 ]]; then
    echo "CASE_FAILED role=$V8_ROLE cuda_device=$CUDA_DEVICE case=$CASE_NAME phase=$CURRENT_PHASE status=$status" >&2
  fi
  exit "$status"
}
trap case_exit EXIT

run_logged() {
  local final_log="$1"
  shift
  local partial_log="${final_log}.partial.$$"
  local failed_log="${final_log%.log}.failed.log"
  local status
  [[ ! -e "$final_log" && ! -e "$partial_log" && ! -e "$failed_log" ]] || \
    die "refusing to overwrite stage log: $final_log"
  if "$@" >"$partial_log" 2>&1; then
    chmod 600 -- "$partial_log"
    mv -- "$partial_log" "$final_log"
  else
    status="$?"
    chmod 600 -- "$partial_log"
    mv -- "$partial_log" "$failed_log"
    echo "stage failed; diagnostic retained at $failed_log" >&2
    return "$status"
  fi
}

echo "CASE_START role=$V8_ROLE cuda_device=$CUDA_DEVICE case=$CASE_NAME lock_sha256=$VERIFIED_LOCK_SHA256"

CURRENT_PHASE="frame-zero-build"
run_logged "$LOG_DIR/$CASE_NAME.frame-zero.log" \
  "${CLEAN_ENV[@]}" "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" -c \
  'import runpy,sys; root=sys.argv.pop(1); module=sys.argv.pop(1); sys.path.insert(0,root); sys.argv[0]=module; runpy.run_module(module,run_name="__main__",alter_sys=False)' \
  "$CODE/src" bayesian_phystwin.cli.deform360_held_v8_frame_zero_assets \
  --lock "$LOCK" \
  --episode-dir "$EPDIR" \
  --case-name "$CASE_NAME" \
  --output-dir "$FZ" \
  --sam2-repository "$SAM2" \
  --checkpoint "$SAM2_CHECKPOINT" \
  --semantic-model "$SEMANTIC_MODEL" \
  --semantic-model-lock "$SEMANTIC_MODEL_LOCK" \
  --deform360-code "$DEFORM360" \
  --role "$V8_ROLE" \
  --device cuda

CURRENT_PHASE="frame-zero-sealing"
run_logged "$LOG_DIR/$CASE_NAME.frame-zero-seal.log" \
  "${CLEAN_ENV[@]}" "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" - \
  "$FZ" <<'PY_FRAME_ZERO_SEAL'
import os
from pathlib import Path
import stat
import sys


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


root = Path(os.path.abspath(sys.argv[1]))
root_state = os.lstat(root)
require(
    stat.S_ISDIR(root_state.st_mode)
    and not stat.S_ISLNK(root_state.st_mode)
    and root.resolve(strict=True) == root,
    "frame-zero output root is not a canonical regular directory",
)

# Payloads are frozen first and the manifest last.  The manifest therefore
# remains an invalid commit marker if sealing either payload fails.
artifact_names = (
    "known_action_76.npz",
    "frame_zero_bundle.npz",
    "frame_zero_bundle.manifest.json",
)
observed_names = tuple(sorted(entry.name for entry in root.iterdir()))
require(
    observed_names == tuple(sorted(artifact_names)),
    "frame-zero output differs from the exact three-artifact allowlist: "
    f"{observed_names}",
)

artifacts = tuple(root / name for name in artifact_names)
for path in artifacts:
    observed = os.lstat(path)
    require(
        stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and path.resolve(strict=True) == path,
        f"frame-zero artifact is not a canonical regular file: {path}",
    )

for path in artifacts:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(path)
        identity = (opened.st_dev, opened.st_ino)
        require(
            stat.S_ISREG(opened.st_mode)
            and (current.st_dev, current.st_ino) == identity,
            f"frame-zero artifact changed before sealing: {path}",
        )
        os.fchmod(descriptor, 0o400)
        sealed = os.fstat(descriptor)
        current = os.lstat(path)
        require(
            (sealed.st_dev, sealed.st_ino) == identity
            and (current.st_dev, current.st_ino) == identity
            and stat.S_IMODE(sealed.st_mode) == 0o400
            and stat.S_IMODE(current.st_mode) == 0o400,
            f"frame-zero artifact was not sealed to exact mode 0400: {path}",
        )
    finally:
        os.close(descriptor)

print("FRAME_ZERO_ARTIFACTS_SEALED mode=0400 count=3 manifest_last=true")
PY_FRAME_ZERO_SEAL

CURRENT_PHASE="frame-zero-validation"
run_logged "$LOG_DIR/$CASE_NAME.frame-zero-validate.log" \
  "${CLEAN_ENV[@]}" "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" - \
  "$CODE/src" "$FZ_MANIFEST" "$LOCK" "$CASE_NAME" "$V8_ROLE" <<'PY'
import sys
sys.path.insert(0, sys.argv.pop(1))
from bayesian_phystwin.deform360_held_v8_protocol import validate_frame_zero_bundle_manifest
validate_frame_zero_bundle_manifest(
    sys.argv[1], sys.argv[2], expected_case_name=sys.argv[3], expected_role=sys.argv[4]
)
PY

CURRENT_PHASE="physical-build"
run_logged "$LOG_DIR/$CASE_NAME.physical.log" \
  "${CLEAN_ENV[@]}" "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" -c \
  'import runpy,sys; root=sys.argv.pop(1); module=sys.argv.pop(1); sys.path.insert(0,root); sys.argv[0]=module; runpy.run_module(module,run_name="__main__",alter_sys=False)' \
  "$CODE/src" bayesian_phystwin.cli.deform360_held_v8_physical_prior \
  --frame-zero-manifest "$FZ_MANIFEST" \
  --lock "$LOCK" \
  --output-dir "$PHYS" \
  --case-name "$CASE_NAME" \
  --role "$V8_ROLE" \
  --upstream-repo "$UPSTREAM" \
  --official-phystwin-repo "$OFFICIAL" \
  --official-config "$OFFICIAL/configs/real.yaml" \
  --deform360-repo "$DEFORM360" \
  --python "$PY" \
  --device cuda:0

CURRENT_PHASE="physical-seal-validation"
run_logged "$LOG_DIR/$CASE_NAME.physical-validate.log" \
  "${CLEAN_ENV[@]}" "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" - \
  "$CODE/src" "$PHYSICAL_SEAL" "$LOCK" "$CASE_NAME" "$V8_ROLE" <<'PY'
import sys
sys.path.insert(0, sys.argv.pop(1))
from bayesian_phystwin.deform360_held_v8_protocol import validate_physical_prior_seal
seal = validate_physical_prior_seal(
    sys.argv[1], sys.argv[2], expected_case_name=sys.argv[3], expected_role=sys.argv[4]
)
if seal["protocol_id"] != "deform360-held-online-belief-v8":
    raise RuntimeError("physical seal is not v8")
PY

CURRENT_PHASE="prefix-authorization"
run_logged "$LOG_DIR/$CASE_NAME.prefix-authorization.log" \
  "${CLEAN_ENV[@]}" "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" - \
  "$CODE/src" "$AUTH" "$LOCK" "$PHYSICAL_SEAL" <<'PY'
import sys
sys.path.insert(0, sys.argv.pop(1))
from bayesian_phystwin.deform360_held_v8_protocol import create_prefix_stage_authorization
create_prefix_stage_authorization(sys.argv[1], sys.argv[2], sys.argv[3])
PY

CURRENT_PHASE="online-prefix-build"
run_logged "$LOG_DIR/$CASE_NAME.online.log" \
  "${CLEAN_ENV[@]}" "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" -c \
  'import runpy,sys; root=sys.argv.pop(1); module=sys.argv.pop(1); sys.path.insert(0,root); sys.argv[0]=module; runpy.run_module(module,run_name="__main__",alter_sys=False)' \
  "$CODE/src" bayesian_phystwin.cli.deform360_held_v8_online_prefix \
  --lock "$LOCK" \
  --frame-zero-manifest "$FZ_MANIFEST" \
  --physical-prior-seal "$PHYSICAL_SEAL" \
  --prefix-authorization "$AUTH" \
  --aligned-episode-dir "$EPDIR" \
  --output-dir "$ONLINE" \
  --case-name "$CASE_NAME" \
  --role "$V8_ROLE" \
  --alltracker-source "$ALLTRACKER" \
  --checkpoint "$ALLTRACKER_CHECKPOINT" \
  --device cuda:0

CURRENT_PHASE="online-seal-validation"
run_logged "$LOG_DIR/$CASE_NAME.online-validate.log" \
  "${CLEAN_ENV[@]}" "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" - \
  "$CODE/src" "$ONLINE_SEAL" "$LOCK" "$CASE_NAME" "$V8_ROLE" <<'PY'
import sys
sys.path.insert(0, sys.argv.pop(1))
from bayesian_phystwin.deform360_held_v8_protocol import validate_online_prediction_seal
seal = validate_online_prediction_seal(
    sys.argv[1], sys.argv[2], expected_case_name=sys.argv[3], expected_role=sys.argv[4]
)
if seal["protocol_id"] != "deform360-held-online-belief-v8":
    raise RuntimeError("online seal is not v8")
PY

# Freeze the selected query field immediately after the fresh online archive.
# This happens before the complete-cohort first barrier can authorize any
# reconstruction process.
CURRENT_PHASE="frozen-field-seal"
mkdir -- "$FIELD"
run_logged "$LOG_DIR/$CASE_NAME.frozen-field.log" \
  "${CLEAN_ENV[@]}" "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" - \
  "$CODE/src" "$FROZEN_FIELD_MANIFEST" "$LOCK" "$VERIFIED_LOCK_SHA256" \
  "$ONLINE_ARCHIVE" "$ONLINE_SEAL" "$DEVELOPMENT_DECISION" \
  "$DEVELOPMENT_DECISION_SHA256" "$CASE_NAME" <<'PY'
import hashlib
from pathlib import Path
import sys
root = Path(sys.argv.pop(1))
sys.path.insert(0, str(root))
from bayesian_phystwin import deform360_frozen_query_field as field
from bayesian_phystwin import deform360_held_v8_query_artifacts as artifacts
(
    manifest_path,
    lock_path,
    lock_sha256,
    online_archive,
    online_seal,
    decision_path,
    decision_sha256,
    case_name,
) = sys.argv[1:]
field_source = Path(field.__file__).resolve()
artifact_source = Path(artifacts.__file__).resolve()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
result = artifacts.write_preoutcome_frozen_field_manifest(
    manifest_path,
    lock_path=lock_path,
    lock_sha256=lock_sha256,
    online_prediction_archive_path=online_archive,
    online_prediction_seal_path=online_seal,
    field_source_path=field_source,
    field_source_sha256=sha(field_source),
    artifact_module_source_path=artifact_source,
    artifact_module_source_sha256=sha(artifact_source),
    development_decision_path=decision_path,
    development_decision_sha256=decision_sha256,
    case_name=case_name,
)
if result["protocol_id"] != "deform360-held-online-belief-v8":
    raise RuntimeError("frozen field manifest is not v8")
PY

CURRENT_PHASE="complete"
echo "CASE_COMPLETE role=$V8_ROLE cuda_device=$CUDA_DEVICE case=$CASE_NAME online_seal=$ONLINE_SEAL frozen_field_manifest=$FROZEN_FIELD_MANIFEST"
