#!/bin/bash

# Normalize before any deployment, dataset, or prediction path is inspected.
# The shard normally supplies the two trust anchors; direct invocation is
# permitted only when it supplies the same anchors and is then re-executed.
if [[ "${BPT_HELD_V6_CONFIRMATION_CASE_ENV_NORMALIZED:-}" != "1" ]]; then
  exec env -i \
    HOME=/home/florianpfaff \
    USER=florianpfaff \
    LOGNAME=florianpfaff \
    PATH=/usr/local/bin:/usr/bin:/bin \
    TMPDIR=/tmp \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    BPT_HELD_V6_CODE="${BPT_HELD_V6_CODE:-}" \
    BPT_HELD_V6_CONFIRMATION_LOCK_VERIFIED_SHA256="${BPT_HELD_V6_CONFIRMATION_LOCK_VERIFIED_SHA256:-}" \
    BPT_HELD_V6_CONFIRMATION_CASE_ENV_NORMALIZED=1 \
    /bin/bash "$0" "$@"
fi

set -Eeuo pipefail
umask 077
export PATH=/usr/local/bin:/usr/bin:/bin
unset BASH_ENV ENV CDPATH

# This is a confirmation-prefix runner only.  It deliberately accepts no path
# for an outcome, target, tactile stream, or confirmation payload.

die() {
  echo "ERROR: $*" >&2
  exit 2
}

while IFS='=' read -r environment_name _value; do
  case "$environment_name" in
    HOME|USER|LOGNAME|PATH|TMPDIR|LANG|LC_ALL|PWD|SHLVL|_|\
    BPT_HELD_V6_CODE|BPT_HELD_V6_CONFIRMATION_LOCK_VERIFIED_SHA256|\
    BPT_HELD_V6_CONFIRMATION_CASE_ENV_NORMALIZED)
      ;;
    *) die "normalized confirmation-case environment contains $environment_name" ;;
  esac
done < <(env)

if [[ "$#" -ne 4 ]]; then
  die "usage: run_deform360_v6_confirmation_case.sh CUDA_DEVICE CASE OBJECT EPISODE"
fi

readonly CUDA_DEVICE="$1"
readonly CASE_NAME="$2"
readonly OBJECT="$3"
readonly EPISODE="$4"

[[ "$CUDA_DEVICE" =~ ^[0-9]+$ ]] || die "CUDA device must be a non-negative integer"

# Bind the redundant tuple, rather than accepting independently variable case,
# object, and episode arguments.  These are the exact six confirmation cases.
case "$CASE_NAME:$OBJECT:$EPISODE" in
  002-rope-silk-ep0001:002-rope-silk:0001|\
  081-stripe-rope-ep0005:081-stripe-rope:0005|\
  085-scarf-cloth-ep0002:085-scarf-cloth:0002|\
  083-blanket-cloth-ep0007:083-blanket-cloth:0007|\
  092-squirrel-ep0001:092-squirrel:0001|\
  170-spider-ep0006:170-spider:0006)
    ;;
  *) die "case tuple is outside the exact six-case v6 confirmation cohort" ;;
esac

readonly HELD="/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v6"
readonly CODE="${BPT_HELD_V6_CODE:?set BPT_HELD_V6_CODE to the immutable v6 deployment}"
readonly CALIBRATION_LOCK="$HELD/calibration-lock.json"
readonly LOCK="$HELD/confirmation-lock.json"
readonly CALIBRATION_DECISION="$HELD/calibration/calibration-gate-decision.json"
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
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly CASE_RUNNER="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
readonly LOCK_OPERATOR="$SCRIPT_DIR/prepare_deform360_v6_lock.py"
# Updated mechanically after the final lock-operator source is frozen.
readonly EXPECTED_LOCK_OPERATOR_SHA256="7622e8a4338c9a76da3d114bde4fa5407374396f4943a838b13dd2214eebe329"
readonly RUN="$HELD/confirmation"
readonly LOG_DIR="$RUN/logs"
readonly ALIGNED="/mnt/lexar4tb/datasets/deform360/data-7fea8e2/replication-v1/aligned"
# Keep this literal virtualenv entry point.  Do not resolve it to the system
# interpreter: the held physical builder validates this exact symlink/runtime.
readonly PY="/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004/bin/python"
readonly PYCACHE_PREFIX="/nonexistent/bpt-held-v6-pycache"
readonly SAM2="/mnt/lexar4tb/datasets/deform360/sam2-2b90b9f5"
readonly SAM2_CHECKPOINT="$SAM2/checkpoints/sam2.1_hiera_small.pt"
readonly UPSTREAM="/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/Bayesian-PhysTwin-upstream-58ab4808e59d"
readonly OFFICIAL="/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/PhysTwin-upstream-2b6630528141"
readonly DEFORM360="/mnt/lexar4tb/datasets/deform360/code"
readonly SEMANTIC_MODEL="/mnt/corsair/florianpfaff/model-cache/siglip2-base-patch16-224-75de2d55"
readonly SEMANTIC_MODEL_LOCK="/mnt/corsair/florianpfaff/bpt-framezero-field-dev-20260720/scratch_siglip2_model_lock.json"
readonly ALLTRACKER="/mnt/corsair/florianpfaff/alltracker-molmomotion-61f5b21"
readonly ALLTRACKER_CHECKPOINT="/mnt/corsair/florianpfaff/model-cache/alltracker.pth"

readonly ROOT="$RUN/cases/$CASE_NAME"
readonly FZ="$ROOT/frame-zero"
readonly PHYS="$ROOT/physical"
readonly AUTH="$ROOT/prefix-authorization.json"
readonly ONLINE="$ROOT/online"
# Deform360 aligned episodes are object/episode_NNNN, not object-epNNNN.
readonly OBJECT_DIR="$ALIGNED/$OBJECT"
readonly EPDIR="$ALIGNED/$OBJECT/episode_$EPISODE"

[[ -d "$HELD" && ! -L "$HELD" ]] || die "held-v6 root is absent or a symlink"
[[ "$(readlink -f -- "$HELD")" == "$HELD" ]] || die "held-v6 root is not canonical"
[[ -d "$CODE" && ! -L "$CODE" ]] || die "v6 deployed code is absent or a symlink"
[[ "$(readlink -f -- "$CODE")" == "$CODE" ]] || die "v6 deployed code is not canonical"
[[ "$CODE" =~ ^${HELD}/code-([0-9a-f]{40}|[0-9a-f]{64})$ ]] || \
  die "v6 deployed code is outside the canonical immutable snapshot path"
[[ "$SCRIPT_DIR" == "$CODE/scripts/held" ]] || die "case runner is outside the deployed snapshot"
[[ "$CASE_RUNNER" == "$CODE/scripts/held/run_deform360_v6_confirmation_case.sh" ]] || \
  die "case runner path is not the canonical tracked source"
[[ -z "$(find "$CODE" -xdev -perm /222 -print -quit)" ]] || \
  die "v6 deployed code tree is writable"
[[ -f "$LOCK_OPERATOR" && ! -L "$LOCK_OPERATOR" ]] || \
  die "tracked lock operator is absent or a symlink"
[[ -z "$(find "$CASE_RUNNER" "$LOCK_OPERATOR" -maxdepth 0 -perm /222 -print -quit)" ]] || \
  die "tracked v6 operators are writable"
readonly ACTUAL_LOCK_OPERATOR_SHA256="$(sha256sum -- "$LOCK_OPERATOR" | awk '{print $1}')"
[[ "$ACTUAL_LOCK_OPERATOR_SHA256" == "$EXPECTED_LOCK_OPERATOR_SHA256" ]] || \
  die "lock operator checksum differs from the audited v6 verifier"
[[ -f "$LOCK" && ! -L "$LOCK" ]] || die "held-v6 lock is absent or a symlink"
[[ "$(readlink -f -- "$LOCK")" == "$LOCK" ]] || die "held-v6 lock is not canonical"
[[ "$(stat -c '%a' -- "$LOCK")" == "400" ]] || die "held-v6 lock mode is not 0400"
[[ -f "$CALIBRATION_LOCK" && ! -L "$CALIBRATION_LOCK" && "$(stat -c '%a' -- "$CALIBRATION_LOCK")" == "400" ]] || \
  die "held-v6 calibration parent lock is absent, mutable, or a symlink"
[[ -f "$CALIBRATION_DECISION" && ! -L "$CALIBRATION_DECISION" ]] || \
  die "held-v6 calibration GO decision is absent or a symlink"
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
[[ -x "$PY" ]] || die "locked virtualenv Python entry point is not executable"
[[ ! -e /nonexistent && ! -L /nonexistent && ! -e "$PYCACHE_PREFIX" && ! -L "$PYCACHE_PREFIX" ]] || \
  die "reserved held-v6 pycache prefix is no longer unavailable"
cd -- "$CODE"
[[ "$PWD" == "$CODE" ]] || die "failed to enter the immutable deployed code root"

# The shard exports this only after the non-mutating lock verifier succeeds.
# Re-hashing here catches lock replacement between shard startup and this case.
readonly VERIFIED_LOCK_SHA256="${BPT_HELD_V6_CONFIRMATION_LOCK_VERIFIED_SHA256:?run cases through the verified v6 confirmation shard runner}"
[[ "$VERIFIED_LOCK_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "invalid verified lock digest"
readonly ACTUAL_LOCK_SHA256="$(sha256sum -- "$LOCK" | awk '{print $1}')"
[[ "$ACTUAL_LOCK_SHA256" == "$VERIFIED_LOCK_SHA256" ]] || \
  die "held-v6 lock changed after startup verification"

# Recompute the full deployed Git snapshot, all 113 immutable bindings, and
# exact v1-through-v5 ancestry immediately before this case accesses payloads.
env -i \
  HOME=/home/florianpfaff \
  USER=florianpfaff \
  LOGNAME=florianpfaff \
  PATH="$PATH" \
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
  --deployed-code "$CODE" \
  --output-lock "$CALIBRATION_LOCK" \
  --verify-existing-lock \
  >/dev/null

# The preparer validates the immutable 113-binding calibration parent.  The
# protocol validator then proves this exact confirmation lock is its derived
# child and binds the exact immutable calibration GO decision.
env -i \
  HOME=/home/florianpfaff \
  USER=florianpfaff \
  LOGNAME=florianpfaff \
  PATH="$PATH" \
  TMPDIR=/tmp \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  PYTHONHASHSEED=0 \
  PYTHONPYCACHEPREFIX="$PYCACHE_PREFIX" \
  "$PY" -B -X "pycache_prefix=$PYCACHE_PREFIX" -I - "$CODE/src" "$LOCK" "$CALIBRATION_LOCK" "$CALIBRATION_DECISION" <<'PY' \
  >/dev/null
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
readonly REVERIFIED_LOCK_SHA256="$(sha256sum -- "$LOCK" | awk '{print $1}')"
[[ "$REVERIFIED_LOCK_SHA256" == "$VERIFIED_LOCK_SHA256" ]] || \
  die "held-v6 lock changed during case-level deployment verification"

[[ -d "$ALIGNED" && ! -L "$ALIGNED" && "$(readlink -f -- "$ALIGNED")" == "$ALIGNED" ]] || \
  die "aligned dataset root is absent, a symlink, or non-canonical"
[[ -d "$OBJECT_DIR" && ! -L "$OBJECT_DIR" && "$(readlink -f -- "$OBJECT_DIR")" == "$OBJECT_DIR" ]] || \
  die "aligned object directory is absent, a symlink, or non-canonical"
[[ -d "$EPDIR" && ! -L "$EPDIR" && "$(readlink -f -- "$EPDIR")" == "$EPDIR" ]] || \
  die "aligned episode directory is absent, a symlink, or non-canonical"
[[ "$EPDIR" == "$ALIGNED/$OBJECT/episode_$EPISODE" ]] || die "episode mapping changed"
[[ -f "$SAM2_CHECKPOINT" ]] || die "SAM2 checkpoint is absent"
[[ -d "$UPSTREAM" ]] || die "source-only upstream runtime is absent"
[[ -d "$OFFICIAL" && -f "$OFFICIAL/configs/real.yaml" ]] || \
  die "official PhysTwin runtime/config is absent"
[[ -d "$DEFORM360" ]] || die "Deform360 code runtime is absent"
[[ -d "$SEMANTIC_MODEL" && ! -L "$SEMANTIC_MODEL" && "$(readlink -f -- "$SEMANTIC_MODEL")" == "$SEMANTIC_MODEL" ]] || \
  die "pinned SigLIP2 model tree is absent, linked, or non-canonical"
[[ -f "$SEMANTIC_MODEL_LOCK" && ! -L "$SEMANTIC_MODEL_LOCK" && "$(readlink -f -- "$SEMANTIC_MODEL_LOCK")" == "$SEMANTIC_MODEL_LOCK" ]] || \
  die "pinned SigLIP2 model lock is absent, linked, or non-canonical"
[[ -d "$ALLTRACKER" && -f "$ALLTRACKER_CHECKPOINT" ]] || \
  die "AllTracker runtime/checkpoint is absent"

# Refuse symlinks in every output parent.  This prevents a nominal held-v6
# path from being redirected into held-v1 or any other prediction tree.
for output_parent in "$RUN" "$RUN/cases" "$LOG_DIR"; do
  if [[ -e "$output_parent" || -L "$output_parent" ]]; then
    [[ -d "$output_parent" && ! -L "$output_parent" ]] || \
      die "output parent is not a real directory: $output_parent"
    [[ "$(readlink -f -- "$output_parent")" == "$output_parent" ]] || \
      die "output parent is non-canonical: $output_parent"
  fi
done
[[ ! -e "$ROOT" && ! -L "$ROOT" ]] || die "case output root already exists"
mkdir -p -- "$RUN/cases" "$LOG_DIR"
for output_parent in "$RUN" "$RUN/cases" "$LOG_DIR"; do
  [[ -d "$output_parent" && ! -L "$output_parent" && "$(readlink -f -- "$output_parent")" == "$output_parent" ]] || \
    die "failed to create a canonical output parent: $output_parent"
done
if compgen -G "$LOG_DIR/$CASE_NAME.*" >/dev/null; then
  die "case log already exists"
fi
# mkdir, not mkdir -p: claiming this case is an atomic, one-attempt operation.
mkdir -- "$ROOT"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONPYCACHEPREFIX="$PYCACHE_PREFIX"
export PYNPUT_BACKEND=dummy
export PYOPENGL_PLATFORM=egl
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# Discard the inherited environment for every prediction subprocess.  This is
# stronger than trying to enumerate possible protected-payload pointer names.
# The locked APIs also accept no protected path arguments and validate their
# information-boundary declarations in every seal.
readonly -a CLEAN_ENV=(
  env
  -i
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
  WANDB_MODE=disabled
  CUBLAS_WORKSPACE_CONFIG=:4096:8
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
)

CURRENT_PHASE="initialization"
case_exit() {
  local status="$?"
  trap - EXIT
  if [[ "$status" -ne 0 ]]; then
    echo "CASE_FAILED cuda_device=$CUDA_DEVICE case=$CASE_NAME phase=$CURRENT_PHASE status=$status" >&2
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
    die "refusing to overwrite a stage log: $final_log"
  if "$@" >"$partial_log" 2>&1; then
    chmod 600 -- "$partial_log"
    mv -- "$partial_log" "$final_log"
  else
    status="$?"
    chmod 600 -- "$partial_log"
    mv -- "$partial_log" "$failed_log"
    echo "stage failed; diagnostic log retained at $failed_log" >&2
    return "$status"
  fi
}

echo "CASE_START cuda_device=$CUDA_DEVICE case=$CASE_NAME lock_sha256=$REVERIFIED_LOCK_SHA256"

CURRENT_PHASE="frame-zero-build"
run_logged "$LOG_DIR/$CASE_NAME.frame-zero.log" \
  "${CLEAN_ENV[@]}" "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" -c \
  'import runpy,sys; root=sys.argv.pop(1); module=sys.argv.pop(1); sys.path.insert(0,root); sys.argv[0]=module; runpy.run_module(module,run_name="__main__",alter_sys=False)' \
  "$CODE/src" bayesian_phystwin.cli.deform360_frame_zero_assets \
  --lock "$LOCK" \
  --episode-dir "$EPDIR" \
  --case-name "$CASE_NAME" \
  --output-dir "$FZ" \
  --sam2-repository "$SAM2" \
  --checkpoint "$SAM2_CHECKPOINT" \
  --semantic-model "$SEMANTIC_MODEL" \
  --semantic-model-lock "$SEMANTIC_MODEL_LOCK" \
  --deform360-code "$DEFORM360" \
  --role confirmation \
  --device cuda

readonly FZ_MANIFEST="$FZ/frame_zero_bundle.manifest.json"
[[ -f "$FZ_MANIFEST" && ! -L "$FZ_MANIFEST" ]] || die "frame-zero manifest is absent or a symlink"
CURRENT_PHASE="frame-zero-validation"
run_logged "$LOG_DIR/$CASE_NAME.frame-zero-validate.log" \
  "${CLEAN_ENV[@]}" "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" - \
  "$CODE/src" "$FZ_MANIFEST" "$LOCK" "$CASE_NAME" <<'PY'
import json
import sys

sys.path.insert(0, sys.argv.pop(1))

from bayesian_phystwin.deform360_held_protocol import validate_frame_zero_bundle_manifest

result = validate_frame_zero_bundle_manifest(
    sys.argv[1],
    sys.argv[2],
    expected_case_name=sys.argv[3],
    expected_role="confirmation",
)
print(json.dumps({"artifact_sha256": result["artifact_sha256"], "case_name": result["case_name"]}, sort_keys=True))
PY

[[ ! -e "$PHYS" && ! -L "$PHYS" ]] || die "physical output exists before physical phase"
CURRENT_PHASE="physical-build"
run_logged "$LOG_DIR/$CASE_NAME.physical.log" \
  "${CLEAN_ENV[@]}" "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" -c \
  'import runpy,sys; root=sys.argv.pop(1); module=sys.argv.pop(1); sys.path.insert(0,root); sys.argv[0]=module; runpy.run_module(module,run_name="__main__",alter_sys=False)' \
  "$CODE/src" bayesian_phystwin.cli.deform360_held_physical_prior \
  --frame-zero-manifest "$FZ_MANIFEST" \
  --lock "$LOCK" \
  --output-dir "$PHYS" \
  --case-name "$CASE_NAME" \
  --role confirmation \
  --upstream-repo "$UPSTREAM" \
  --official-phystwin-repo "$OFFICIAL" \
  --official-config "$OFFICIAL/configs/real.yaml" \
  --deform360-repo "$DEFORM360" \
  --python "$PY" \
  --device cuda:0

readonly PHYSICAL_SEAL="$PHYS/physical_prior_seal.json"
[[ -f "$PHYSICAL_SEAL" && ! -L "$PHYSICAL_SEAL" ]] || die "physical seal is absent or a symlink"
CURRENT_PHASE="physical-seal-validation"
run_logged "$LOG_DIR/$CASE_NAME.physical-validate.log" \
  "${CLEAN_ENV[@]}" "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" - \
  "$CODE/src" "$PHYSICAL_SEAL" "$LOCK" "$CASE_NAME" <<'PY'
import json
import sys

sys.path.insert(0, sys.argv.pop(1))

from bayesian_phystwin.deform360_held_protocol import validate_physical_prior_seal

result = validate_physical_prior_seal(
    sys.argv[1],
    sys.argv[2],
    expected_case_name=sys.argv[3],
    expected_role="confirmation",
)
print(json.dumps({"artifact_sha256": result["artifact_sha256"], "case_name": result["case_name"]}, sort_keys=True))
PY

[[ ! -e "$AUTH" && ! -L "$AUTH" ]] || die "prefix authorization already exists"
CURRENT_PHASE="prefix-authorization-create"
run_logged "$LOG_DIR/$CASE_NAME.prefix-authorization.log" \
  "${CLEAN_ENV[@]}" "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" - \
  "$CODE/src" "$AUTH" "$LOCK" "$PHYSICAL_SEAL" <<'PY'
import json
import sys

sys.path.insert(0, sys.argv.pop(1))

from bayesian_phystwin.deform360_held_protocol import create_prefix_stage_authorization

result = create_prefix_stage_authorization(sys.argv[1], sys.argv[2], sys.argv[3])
print(json.dumps({"artifact_sha256": result["artifact_sha256"], "case_name": result["case_name"]}, sort_keys=True))
PY

[[ -f "$AUTH" && ! -L "$AUTH" ]] || die "prefix authorization is absent or a symlink"
CURRENT_PHASE="prefix-authorization-validation"
run_logged "$LOG_DIR/$CASE_NAME.prefix-validate.log" \
  "${CLEAN_ENV[@]}" "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" - \
  "$CODE/src" "$AUTH" "$LOCK" "$CASE_NAME" <<'PY'
import json
import sys

sys.path.insert(0, sys.argv.pop(1))

from bayesian_phystwin.deform360_held_protocol import validate_prefix_stage_authorization

result = validate_prefix_stage_authorization(sys.argv[1], sys.argv[2])
if result["case_name"] != sys.argv[3] or result["role"] != "confirmation":
    raise RuntimeError("prefix authorization binds another case or role")
print(json.dumps({"artifact_sha256": result["artifact_sha256"], "case_name": result["case_name"]}, sort_keys=True))
PY

[[ ! -e "$ONLINE" && ! -L "$ONLINE" ]] || die "online output exists before prefix phase"
CURRENT_PHASE="online-prefix-build"
run_logged "$LOG_DIR/$CASE_NAME.online.log" \
  "${CLEAN_ENV[@]}" "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" -c \
  'import runpy,sys; root=sys.argv.pop(1); module=sys.argv.pop(1); sys.path.insert(0,root); sys.argv[0]=module; runpy.run_module(module,run_name="__main__",alter_sys=False)' \
  "$CODE/src" bayesian_phystwin.cli.deform360_held_online_prefix \
  --lock "$LOCK" \
  --frame-zero-manifest "$FZ_MANIFEST" \
  --physical-prior-seal "$PHYSICAL_SEAL" \
  --prefix-authorization "$AUTH" \
  --aligned-episode-dir "$EPDIR" \
  --output-dir "$ONLINE" \
  --case-name "$CASE_NAME" \
  --role confirmation \
  --alltracker-source "$ALLTRACKER" \
  --checkpoint "$ALLTRACKER_CHECKPOINT" \
  --device cuda:0

readonly ONLINE_SEAL="$ONLINE/online_prediction_seal.json"
[[ -f "$ONLINE_SEAL" && ! -L "$ONLINE_SEAL" ]] || die "online prediction seal is absent or a symlink"
CURRENT_PHASE="online-seal-validation"
run_logged "$LOG_DIR/$CASE_NAME.online-validate.log" \
  "${CLEAN_ENV[@]}" "$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX" - \
  "$CODE/src" "$ONLINE_SEAL" "$LOCK" "$CASE_NAME" <<'PY'
import json
import sys

sys.path.insert(0, sys.argv.pop(1))

from bayesian_phystwin.deform360_held_protocol import validate_online_prediction_seal

result = validate_online_prediction_seal(
    sys.argv[1],
    sys.argv[2],
    expected_case_name=sys.argv[3],
    expected_role="confirmation",
)
print(json.dumps({"artifact_sha256": result["artifact_sha256"], "case_name": result["case_name"]}, sort_keys=True))
PY

CURRENT_PHASE="complete"
echo "CASE_COMPLETE cuda_device=$CUDA_DEVICE case=$CASE_NAME seal=$ONLINE_SEAL"
