#!/usr/bin/env bash
set -euo pipefail

: "${BPT_PYTHON:?BPT_PYTHON is required}"
: "${BPT_SOURCE_SHA:?BPT_SOURCE_SHA is required}"
: "${SOURCE_ARTIFACT_DIR:?SOURCE_ARTIFACT_DIR is required}"
: "${RESULTS_ROOT:?RESULTS_ROOT is required}"
: "${PROCESSED_ROOT:?PROCESSED_ROOT is required}"
: "${VISUAL_PRODUCTION_ROOT:?VISUAL_PRODUCTION_ROOT is required}"
: "${METRIC_BATCH_ROOT:?METRIC_BATCH_ROOT is required}"
: "${DEFORM360_PHYSICAL_REPO:?DEFORM360_PHYSICAL_REPO is required}"
: "${OFFICIAL_PHYSTWIN_REPO:?OFFICIAL_PHYSTWIN_REPO is required}"
: "${SAM2_REPO:?SAM2_REPO is required}"
: "${EVIDENCE_ROOT:?EVIDENCE_ROOT is required}"

AMENDMENT_ID="f8ed525480a6a96265af3cd58e62a96bf1ed748294d0af02aa6386763b993b7f"
AMENDMENT_PATH="protocols/amendments/deform360_official_hub_fresh_object_session_v6_source_prediction_execution.json"
SELECTOR_REPAIR_ID="41f3580de5ca7e09bcd4c2623569c293e29ed796634c60c84ededdbd945af042"
SELECTOR_REPAIR_PATH="protocols/amendments/deform360_official_hub_fresh_object_session_v6_selector_identity_repair.json"
LOCK_PATH="protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json"
SOURCE_PROTOCOL="protocols/deform360_official_hub_calibration_source_v1.json"
STAGE0_PROTOCOL="protocols/deform360_official_hub_visuotactile_v1.json"
SELECTION_LOCK="protocols/locks/deform360_official_hub_visuotactile_v1_selection.json"
VISUAL_PROVIDER_LOCK="protocols/locks/deform360_official_hub_visuotactile_v1_visual_provider/visual-provider-lock.json"
FALLBACK_CONFIG="configs/sota/deform360_reconstruction_failure_persistence_fallback_v1.json"
SAM2_CHECKPOINT_NAME="sam2.1_hiera_small.pt"
SAM2_CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/${SAM2_CHECKPOINT_NAME}"
SAM2_SHA256="6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
SELECTOR_SHA256="c10391578c73dde47fbce160312559a7e638007e9053ec89373fe575cc64d7e5"
SELECTOR_BYTE_COUNT="17310"
CAUSAL4D_SELECTOR_REVISION="50e3682a5dbf976b20cc9115b6e7a975d0144ea5"
OFFICIAL_CONFIG_SHA256="a40a5ec2f5c978c1290810f20ed56db7cab99dc0c227adfe6b7434dfc95ead48"

RUN_ROOT="${RESULTS_ROOT}/bayesian-phystwin/deform360-v6-source-prediction/${AMENDMENT_ID}/${BPT_SOURCE_SHA}"
PHYSICAL_SOURCE_ROOT="${RUN_ROOT}/physical-source"
STAGED_ROOT="${RUN_ROOT}/staged-prefix"
PHYSICAL_WORK_ROOT="${RUN_ROOT}/physical-work"
BACKBONE_ROOT="${RUN_ROOT}/backbone-seals"
PREDICTION_ROOT="${RUN_ROOT}/prediction-panel"
LOG_ROOT="${RUN_ROOT}/logs"
COMPACT_ROOT="${EVIDENCE_ROOT}/deform360-v6-source-prediction-evidence"
STATUS_FILE="${RUN_ROOT}/terminal-status.txt"
STAGE_FILE="${RUN_ROOT}/current-stage.txt"
ERROR_FILE="${RUN_ROOT}/error.txt"

mkdir -p \
  "${RUN_ROOT}" \
  "${PHYSICAL_SOURCE_ROOT}" \
  "${STAGED_ROOT}" \
  "${PHYSICAL_WORK_ROOT}" \
  "${BACKBONE_ROOT}" \
  "${LOG_ROOT}" \
  "${COMPACT_ROOT}"
printf '%s\n' "starting" > "${STATUS_FILE}"
printf '%s\n' "preflight" > "${STAGE_FILE}"
: > "${ERROR_FILE}"

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

set_stage() {
  printf '%s\n' "$1" > "${STAGE_FILE}"
  echo "stage=$1"
}

fail_incomplete() {
  printf '%s\n' "source-inputs-incomplete" > "${STATUS_FILE}"
  printf '%s\n' "$1" > "${ERROR_FILE}"
  return 3
}

fail_invalid() {
  printf '%s\n' "invalid" > "${STATUS_FILE}"
  printf '%s\n' "$1" > "${ERROR_FILE}"
  return 4
}

write_receipt() {
  local exit_code="$1"
  local status
  local stage
  status="$(cat "${STATUS_FILE}")"
  stage="$(cat "${STAGE_FILE}")"
  if [[ "${exit_code}" -ne 0 && "${status}" == "starting" ]]; then
    status="source-technical-failure-retained"
  fi
  if [[ "${exit_code}" -eq 0 ]]; then
    status="source-prediction-evidence-sealed"
  fi
  export RECEIPT_STATUS="${status}"
  export RECEIPT_STAGE="${stage}"
  export RECEIPT_EXIT_CODE="${exit_code}"
  export RECEIPT_ERROR="$(tail -c 16000 "${ERROR_FILE}" 2>/dev/null || true)"
  export RUN_ROOT PHYSICAL_WORK_ROOT PREDICTION_ROOT COMPACT_ROOT
  "${BPT_PYTHON}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def digest(path: Path) -> str | None:
    if not path.is_file() or path.is_symlink():
        return None
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


run_root = Path(os.environ["RUN_ROOT"])
prediction_root = Path(os.environ["PREDICTION_ROOT"])
compact_root = Path(os.environ["COMPACT_ROOT"])
compact_root.mkdir(parents=True, exist_ok=True)
paths = {
    "prepared_source_inventory": run_root / "prepared-source-inventory.json",
    "source_plan_inputs": run_root / "source-plan-inputs.json",
    "source_plan": run_root / "source-plan.json",
    "prediction_batch": prediction_root / "source-prediction-batch.json",
    "prediction_receipt": prediction_root / "source-prediction-receipt.json",
}
physical_manifests = sorted(
    Path(os.environ["PHYSICAL_WORK_ROOT"]).glob(
        "*/physical_prediction_manifest.json"
    )
)
source_seals = sorted((prediction_root / "source-seals").glob("*.json"))
receipt = {
    "schema": (
        "bayesian-phystwin.deform360-v6-source-prediction-execution-receipt"
    ),
    "schema_version": 1,
    "amendment_id": (
        "f8ed525480a6a96265af3cd58e62a96bf1ed748294d0af02aa6386763b993b7f"
    ),
    "selector_identity_repair_id": (
        "41f3580de5ca7e09bcd4c2623569c293e29ed796634c60c84ededdbd945af042"
    ),
    "source_revision": os.environ["BPT_SOURCE_SHA"],
    "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
    "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
    "runner_name": os.environ.get("RUNNER_NAME"),
    "status": os.environ["RECEIPT_STATUS"],
    "terminal_stage": os.environ["RECEIPT_STAGE"],
    "exit_code": int(os.environ["RECEIPT_EXIT_CODE"]),
    "error": os.environ.get("RECEIPT_ERROR") or None,
    "artifacts": {
        name: {
            "path": (
                path.relative_to(run_root).as_posix()
                if path.is_relative_to(run_root)
                else str(path)
            ),
            "sha256": digest(path),
            "byte_count": path.stat().st_size if path.is_file() else None,
        }
        for name, path in paths.items()
    },
    "physical_manifest_count": len(physical_manifests),
    "source_prediction_seal_count": len(source_seals),
    "information_boundary": {
        "development_suffix_opened": False,
        "future_object_observations_used_for_prediction": False,
        "v5_confirmation_payloads_opened": False,
        "v5_confirmation_outcomes_used": False,
        "v6_fresh_target_selected": False,
        "v6_target_payloads_opened": False,
        "v6_target_outcomes_used": False,
        "replacement_allowed": False,
    },
    "claim_authorized": False,
    "fresh_target_selection_authorized": False,
    "fresh_target_payload_access_authorized": False,
}
canonical = json.dumps(
    receipt,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
receipt["receipt_id"] = hashlib.sha256(canonical).hexdigest()
output = compact_root / "execution-receipt.json"
output.write_text(
    json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
for path in paths.values():
    if path.is_file() and not path.is_symlink():
        target = compact_root / path.name
        target.write_bytes(path.read_bytes())
for path in physical_manifests:
    target = compact_root / "physical-manifests" / path.parent.name / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
for path in source_seals:
    target = compact_root / "source-seals" / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
PY
  find "${COMPACT_ROOT}" -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > "${COMPACT_ROOT}/SHA256SUMS"
}

finish() {
  local exit_code="$?"
  set +e
  write_receipt "${exit_code}"
  exit "${exit_code}"
}
trap finish EXIT

set_stage "validate-reviewed-source"
test "${RUNNER_NAME}" = "workstation2" || fail_invalid "wrong runner"
test "$(git rev-parse HEAD)" = "${BPT_SOURCE_SHA}" \
  || fail_invalid "checked-out revision changed"
for checkout in \
  _deform360_physical \
  _official_phystwin \
  _sam2 \
  _causal4d_discovery
do
  grep -qxF "/${checkout}/" .git/info/exclude \
    || printf '/%s/\n' "${checkout}" >> .git/info/exclude
done
test -z "$(git status --porcelain=v1)" \
  || fail_invalid "repository is dirty before source access"
test -d "${RESULTS_ROOT}" || fail_incomplete "results root is missing"
test -d "${PROCESSED_ROOT}/aligned" \
  || fail_incomplete "processed source root is missing"
test -d "${VISUAL_PRODUCTION_ROOT}" \
  || fail_incomplete "visual production root is missing"
test -d "${METRIC_BATCH_ROOT}" \
  || fail_incomplete "v4 metric batch root is missing"
test -d "${SOURCE_ARTIFACT_DIR}" \
  || fail_incomplete "calibration source artifact is missing"
test -f "${AMENDMENT_PATH}" || fail_invalid "execution amendment is missing"
test -f "${SELECTOR_REPAIR_PATH}" \
  || fail_invalid "selector identity repair is missing"
test -f "${LOCK_PATH}" || fail_invalid "v5 source lock is missing"

set_stage "verify-execution-amendment"
"${BPT_PYTHON}" - "${SELECTOR_REPAIR_PATH}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from scripts.remote.run_deform360_v6_selector_identity_repair import (
    load_selector_identity_repair,
)

path = Path(
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "source_prediction_execution.json"
)
payload = json.loads(path.read_text(encoding="utf-8"))
declared = payload.pop("amendment_id")
canonical = json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
observed = hashlib.sha256(canonical).hexdigest()
expected = "f8ed525480a6a96265af3cd58e62a96bf1ed748294d0af02aa6386763b993b7f"
if observed != declared or observed != expected:
    raise SystemExit("source prediction execution amendment changed")
repair = load_selector_identity_repair(sys.argv[1])
if repair["repair_id"] != (
    "41f3580de5ca7e09bcd4c2623569c293e29ed796634c60c84ededdbd945af042"
):
    raise SystemExit("selector identity repair changed")
PY

set_stage "materialize-prepared-source-inventory"
"${BPT_PYTHON}" \
  scripts/science/inventory_deform360_calibration_prepared_source.py \
  --source-protocol "${SOURCE_PROTOCOL}" \
  --stage0-protocol "${STAGE0_PROTOCOL}" \
  --selection-lock "${SELECTION_LOCK}" \
  --visual-provider-lock "${VISUAL_PROVIDER_LOCK}" \
  --plan "${SOURCE_ARTIFACT_DIR}/calibration-source-plan.json" \
  --download-manifest \
    "${SOURCE_ARTIFACT_DIR}/calibration-download-manifest.json" \
  --result "${SOURCE_ARTIFACT_DIR}/calibration-source-result.json" \
  --run-record "${SOURCE_ARTIFACT_DIR}/execution-manifest.json" \
  --processed-root "${PROCESSED_ROOT}/aligned" \
  --implementation-revision "${BPT_SOURCE_SHA}" \
  --output "${RUN_ROOT}/prepared-source-inventory.json" \
  > "${LOG_ROOT}/prepared-source-inventory.log" 2>&1

set_stage "locate-frozen-sam2-checkpoint"
SAM2_CHECKPOINT="${RUN_ROOT}/${SAM2_CHECKPOINT_NAME}"
if [[ ! -f "${SAM2_CHECKPOINT}" ]]; then
  found_checkpoint=""
  while IFS= read -r candidate; do
    [[ -f "${candidate}" && ! -L "${candidate}" ]] || continue
    if [[ "$(sha256_file "${candidate}")" = "${SAM2_SHA256}" ]]; then
      found_checkpoint="${candidate}"
      break
    fi
  done < <(
    find \
      "${RUNNER_WORKSPACE:-/home/github-runner}" \
      /home/github-runner \
      /mnt/lexar4tb \
      -type f \
      -name "${SAM2_CHECKPOINT_NAME}" \
      2>/dev/null | sort -u
  )
  if [[ -n "${found_checkpoint}" ]]; then
    cp --reflink=auto "${found_checkpoint}" "${SAM2_CHECKPOINT}"
  else
    curl --fail --location --retry 3 \
      --output "${SAM2_CHECKPOINT}.part" \
      "${SAM2_CHECKPOINT_URL}" \
      || fail_incomplete "frozen SAM2 checkpoint is unavailable"
    mv "${SAM2_CHECKPOINT}.part" "${SAM2_CHECKPOINT}"
  fi
fi
test "$(sha256_file "${SAM2_CHECKPOINT}")" = "${SAM2_SHA256}" \
  || fail_invalid "SAM2 checkpoint identity changed"

set_stage "locate-frozen-generic-selector"
GENERIC_SELECTOR_REPOSITORY="${GITHUB_WORKSPACE}/_causal4d_discovery"
GENERIC_SELECTOR_SOURCE="${GENERIC_SELECTOR_REPOSITORY}/src/causal4d_public/deform360_object_sam2.py"
test -d "${GENERIC_SELECTOR_REPOSITORY}" \
  || fail_incomplete "frozen Causal4D selector repository is unavailable"
test "$(git -C "${GENERIC_SELECTOR_REPOSITORY}" rev-parse HEAD)" \
  = "${CAUSAL4D_SELECTOR_REVISION}" \
  || fail_invalid "Causal4D selector revision changed"
test -z "$(git -C "${GENERIC_SELECTOR_REPOSITORY}" status --porcelain)" \
  || fail_invalid "Causal4D selector repository is dirty"
test -f "${GENERIC_SELECTOR_SOURCE}" \
  && test ! -L "${GENERIC_SELECTOR_SOURCE}" \
  || fail_incomplete "frozen generic SAM2 selector source is unavailable"
test "$(stat -c '%s' "${GENERIC_SELECTOR_SOURCE}")" = "${SELECTOR_BYTE_COUNT}" \
  || fail_invalid "generic SAM2 selector byte count changed"
test "$(sha256_file "${GENERIC_SELECTOR_SOURCE}")" = "${SELECTOR_SHA256}" \
  || fail_invalid "generic SAM2 selector identity changed"
GENERIC_SELECTOR_ROOT="${GENERIC_SELECTOR_REPOSITORY}/src"

set_stage "locate-frozen-physical-upstream"
UPSTREAM_ROOT=""
while IFS= read -r smoke; do
  [[ -f "${smoke}" && ! -L "${smoke}" ]] || continue
  candidate_root="${smoke%/scripts/remote/run_deform360_official_phystwin_smoke.py}"
  if CANDIDATE_ROOT="${candidate_root}" "${BPT_PYTHON}" - <<'PY'
import hashlib
import os
from pathlib import Path

required = {
    "scripts/remote/run_deform360_official_phystwin_smoke.py": (
        "e7bf6a6c06e074ac3cdefe259c1cf5eecf8cd905dae1b710a81107ab166ca535"
    ),
    "src/causal4d_public/deform360_reusable_graph.py": (
        "97b93e32c5009f5783b2f36be7e03d4acda33f0608c9694797e8e5c72d3dd8a5"
    ),
    "src/causal4d_public/deform360_partial_graph_state.py": (
        "81536d81ce4cfd0e61074d2f4096b3160624b6afa2e1dda1d0dab16c113192a3"
    ),
    "src/causal4d_public/deform360_dense_reusable_panel.py": (
        "0861831b9ab3cf6d64833efe533073f4f444f2315c04057377f243efffd8b17e"
    ),
    "src/causal4d_public/deform360_action_support.py": (
        "132283722400ac102ec84e9b7d21974edcdac0ff750168d70860cd89c8446783"
    ),
    "src/causal4d_public/deform360_contact_conditioned_action.py": (
        "1d4e2bbd4389d8d7055d0803f3feda3ea540d45123e0aa3f646bccf2cfa6c57e"
    ),
    "src/causal4d_public/deform360_dense_source.py": (
        "6c9ffa0043302079acf303f23af9e9ebb895f0aa8cf03930effe8936a879bb29"
    ),
    "src/bayesian_phystwin/phystwin_graph.py": (
        "f6f1ef8d3a1fb95fc069a550ae7db12d6b32efe80582f479efb411452062b6fb"
    ),
    "configs/causal4d_public/deform360_dense_reusable_panel_v1.json": (
        "8a90705dd38c6c90b042ed8f450e2bc7e3cffc54b965765b004d0385999d40ea"
    ),
    "configs/causal4d_public/deform360_independent_source_split_v1.json": (
        "c150b2c8ea3947fe2ffe359c5da45d321b5086cd67141c2da9f912aac154ff4a"
    ),
}
root = Path(os.environ["CANDIDATE_ROOT"])
for relative, expected in required.items():
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise SystemExit(1)
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    if value != expected:
        raise SystemExit(1)
PY
  then
    UPSTREAM_ROOT="${candidate_root}"
    break
  fi
done < <(
  find \
    "${RUNNER_WORKSPACE:-/home/github-runner}" \
    /home/github-runner \
    -type f \
    -path '*/scripts/remote/run_deform360_official_phystwin_smoke.py' \
    2>/dev/null | sort -u
)
test -n "${UPSTREAM_ROOT}" \
  || fail_incomplete "frozen automatic-twin upstream source is unavailable"

set_stage "locate-official-phystwin-config"
OFFICIAL_CONFIG=""
while IFS= read -r candidate; do
  [[ -f "${candidate}" && ! -L "${candidate}" ]] || continue
  if [[ "$(sha256_file "${candidate}")" = "${OFFICIAL_CONFIG_SHA256}" ]]; then
    OFFICIAL_CONFIG="${candidate}"
    break
  fi
done < <(find "${OFFICIAL_PHYSTWIN_REPO}" -type f 2>/dev/null | sort)
test -n "${OFFICIAL_CONFIG}" \
  || fail_incomplete "frozen official PhysTwin real config is unavailable"

export PYTHONPATH="${GENERIC_SELECTOR_ROOT}:${DEFORM360_PHYSICAL_REPO}:${PYTHONPATH:-}"

set_stage "materialize-physical-source"
OBJECT_ROWS="${RUN_ROOT}/objects.tsv"
"${BPT_PYTHON}" - "${LOCK_PATH}" > "${OBJECT_ROWS}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for row in sorted(
    payload["cohort"]["development_objects"],
    key=lambda value: value["object_id"],
):
    print(row["object_id"], row["episode_id"], row["stratum"], sep="\t")
PY

while IFS=$'\t' read -r object_id episode_id stratum; do
  case_id="${object_id}-ep$(printf '%04d' "${episode_id}")"
  set_stage "physical-source:${case_id}"
  "${BPT_PYTHON}" \
    scripts/science/materialize_deform360_joint_sparse_physical_source_v5.py \
    --execution-lock "${LOCK_PATH}" \
    --prepared-source-inventory "${RUN_ROOT}/prepared-source-inventory.json" \
    --processed-root "${PROCESSED_ROOT}/aligned" \
    --object-id "${object_id}" \
    --output-root "${PHYSICAL_SOURCE_ROOT}" \
    > "${LOG_ROOT}/${case_id}-materialize.log" 2>&1

  set_stage "stage-prefix:${case_id}"
  "${BPT_PYTHON}" \
    scripts/remote/run_deform360_v6_selector_identity_repair.py \
    --execution-repo "${GITHUB_WORKSPACE}" \
    --execution-lock "${LOCK_PATH}" \
    --runtime-repair "${SELECTOR_REPAIR_PATH}" \
    --selector-repository "${GENERIC_SELECTOR_REPOSITORY}" \
    --stage stage-prefix \
    --repo "${GITHUB_WORKSPACE}" \
    --protocol "${LOCK_PATH}" \
    --role calibration \
    --source-aligned-root "${PHYSICAL_SOURCE_ROOT}" \
    --object-id "${object_id}" \
    --episode-id "${episode_id}" \
    --output-root "${STAGED_ROOT}" \
    --generic-selector-source "${GENERIC_SELECTOR_SOURCE}" \
    --sam2-repository "${SAM2_REPO}" \
    --sam2-checkpoint "${SAM2_CHECKPOINT}" \
    --device cuda:0 \
    > "${LOG_ROOT}/${case_id}-stage-prefix.log" 2>&1

  set_stage "frame-zero:${case_id}"
  "${BPT_PYTHON}" \
    scripts/remote/run_deform360_joint_sparse_physical_source_v5.py \
    --execution-repo "${GITHUB_WORKSPACE}" \
    --execution-lock "${LOCK_PATH}" \
    --stage frame-zero \
    --protocol "${LOCK_PATH}" \
    --deform360-repo "${DEFORM360_PHYSICAL_REPO}" \
    --staged-case-dir "${STAGED_ROOT}/${case_id}" \
    --persistence-fallback-source-config "${FALLBACK_CONFIG}" \
    > "${LOG_ROOT}/${case_id}-frame-zero.log" 2>&1

  set_stage "physical-prior:${case_id}"
  "${BPT_PYTHON}" \
    scripts/remote/run_deform360_joint_sparse_physical_source_v5.py \
    --execution-repo "${GITHUB_WORKSPACE}" \
    --execution-lock "${LOCK_PATH}" \
    --stage physical-prior \
    --repo "${GITHUB_WORKSPACE}" \
    --protocol "${LOCK_PATH}" \
    --staged-case-dir "${STAGED_ROOT}/${case_id}" \
    --work-root "${PHYSICAL_WORK_ROOT}" \
    --backbone-root "${BACKBONE_ROOT}" \
    --upstream-repo "${UPSTREAM_ROOT}" \
    --official-phystwin-repo "${OFFICIAL_PHYSTWIN_REPO}" \
    --official-config "${OFFICIAL_CONFIG}" \
    --deform360-repo "${DEFORM360_PHYSICAL_REPO}" \
    --python "${BPT_PYTHON}" \
    --device cuda:0 \
    > "${LOG_ROOT}/${case_id}-physical-prior.log" 2>&1

done < "${OBJECT_ROWS}"

set_stage "materialize-source-plan"
"${BPT_PYTHON}" \
  scripts/science/materialize_deform360_v6_source_plan_inputs.py \
  --execution-lock "${LOCK_PATH}" \
  --execution-amendment "${AMENDMENT_PATH}" \
  --metric-batch-root "${METRIC_BATCH_ROOT}" \
  --prediction-root "${VISUAL_PRODUCTION_ROOT}" \
  --physical-work-root "${PHYSICAL_WORK_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --implementation-revision "${BPT_SOURCE_SHA}" \
  --output "${RUN_ROOT}/source-plan-inputs.json" \
  > "${LOG_ROOT}/source-plan-inputs.log" 2>&1

"${BPT_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
wrapper = json.loads(
    (root / "source-plan-inputs.json").read_text(encoding="utf-8")
)
path = root / "source-plan.json"
if path.exists():
    raise SystemExit("source plan already exists")
path.write_text(
    json.dumps(
        wrapper["source_prediction_plan"],
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    + "\n",
    encoding="utf-8",
)
PY

set_stage "generate-nested-source-predictions"
"${BPT_PYTHON}" \
  scripts/science/run_deform360_joint_sparse_source_predictions_v5.py \
  --execution-lock "${LOCK_PATH}" \
  --source-plan "${RUN_ROOT}/source-plan.json" \
  --input-root "${RESULTS_ROOT}" \
  --output-root "${PREDICTION_ROOT}" \
  > "${LOG_ROOT}/source-predictions.log" 2>&1

set_stage "verify-prediction-evidence"
"${BPT_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PREDICTION_ROOT"])
receipt = json.loads(
    (root / "source-prediction-receipt.json").read_text(encoding="utf-8")
)
batch = json.loads(
    (root / "source-prediction-batch.json").read_text(encoding="utf-8")
)
if receipt.get("prediction_record_count") != 100:
    raise SystemExit("source prediction receipt does not contain 100 records")
if batch.get("record_count") != 100 or batch.get("fold_count") != 10:
    raise SystemExit("source prediction batch is incomplete")
if len(list((root / "source-seals").glob("*.json"))) != 100:
    raise SystemExit("source prediction seal roster is incomplete")
boundary = receipt.get("information_boundary", {})
if boundary.get("development_suffix_opened") is not False:
    raise SystemExit("development suffix was opened")
if boundary.get("target_outcomes_used") is not False:
    raise SystemExit("target outcomes were used")
PY

printf '%s\n' "source-prediction-evidence-sealed" > "${STATUS_FILE}"
set_stage "completed"
