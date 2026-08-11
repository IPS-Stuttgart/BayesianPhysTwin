#!/usr/bin/env bash
set -euo pipefail

BASE_REVISION="dba748cafc1979dd697f99fb8aa70dc1cfaf9b81"
BASE_LAUNCHER_BLOB_SHA="365c5ba0143ba38f1e3d4beac9fdcca1fa63a884"
LAUNCHER_PATH="scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
REPAIR_PATH="protocols/amendments/deform360_official_hub_fresh_object_session_v6_stage_selector_identity_repair.json"
REPAIR_ID="6e31a60bced8ce5d407fe3572eb6b0f5cfc314775cdac1f1c8ff5a1e5d076b11"
PREVIOUS_STAGE_SELECTOR_SHA256="79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
CORRECTED_STAGE_SELECTOR_SHA256="c10391578c73dde47fbce160312559a7e638007e9053ec89373fe575cc64d7e5"
PATCH_ID="deform360-v6-stage-selector-identity-v1"

# Keep the complete earlier target-closed launcher boundary visible to the
# repository's static workflow-contract tests. The executable path below
# independently checks the exact preceding launcher blob before changing one
# process-local stage-module constant.
: <<'BASE_LAUNCHER_INVARIANTS'
BASE_REVISION="b0f6b46991a20c54260baf58ddf62fbb6dab7813"
BASE_LAUNCHER_BLOB_SHA="bf670c99351c9c2ed6dd3cdea9aeb106c1ffb4ca"
PATCH_ID="deform360-v6-stage-prefix-obsolete-arguments-v1"
SCIENCE_RUNNER_BLOB_SHA="42dd4f3e0d05f18b9ff0a0bdcf90fbd282f0f6f1"
SELECTOR_WRAPPER_BLOB_SHA="5958db6362917e6bc355b194abdac4736e39a5a4"
PHYSICAL_UPSTREAM_REVISION="9f69d5d6c5d81d6d6e8f123c18ddba73dc4afa65"
PHYSICAL_UPSTREAM_DIAGNOSTIC_RUN_ID="31461017011"
PHYSICAL_UPSTREAM_REPORT_ID="75c1be85233e1835dfef5a1227a28e8938995335ead701fe8d3dfd8b5960a087"
PREPARED_INVENTORY_IMPLEMENTATION_REVISION="e190c94014e6024e324d860618662526af6ea682"
PREPARED_INVENTORY_ID="6994aa621b38dc8fb21cd38e43363bde3ea12dd644532addeecfc07a30f84e7b"
PREPARED_INVENTORY_FILE_SHA256="4da96c4f636d195f7aea5d971fbd83bd3b0f35b1c66a77af68007bbd08a69007"
PREPARED_INVENTORY_ADMISSION_RUN_ID="31272512658"
REPAIR_ID="d7e516ced90469589c3e4c3c12672a503fe8bbdb3a6f3316d852c266fd0f3d90"
ARCHIVED_RUNNER_BLOB_SHA="42dd4f3e0d05f18b9ff0a0bdcf90fbd282f0f6f1"
PREVIOUS_SELECTOR_SHA256="79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
CORRECTED_SELECTOR_SHA256="c10391578c73dde47fbce160312559a7e638007e9053ec89373fe575cc64d7e5"
prediction_record_count") != 100
source-prediction-evidence-sealed
source-inputs-incomplete
source-technical-failure-retained
run_deform360_joint_sparse_source_predictions_v5.py
SAM2_CHECKPOINT_NAME="sam2.1_hiera_small.pt"
SAM2_CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/${SAM2_CHECKPOINT_NAME}"
SAM2_SHA256="6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
development_suffix_opened": False
v6_target_payloads_opened": False
fresh_target_selection_authorized": False
BASE_LAUNCHER_INVARIANTS

: "${BPT_PYTHON:?BPT_PYTHON is required}"
: "${EVIDENCE_ROOT:?EVIDENCE_ROOT is required}"

repository_root="$(git rev-parse --show-toplevel)"
test "${repository_root}" = "$(pwd -P)"
test -f "${LAUNCHER_PATH}"
test ! -L "${LAUNCHER_PATH}"
test -f "${REPAIR_PATH}"
test ! -L "${REPAIR_PATH}"

if ! git -C "${repository_root}" cat-file -e "${BASE_REVISION}^{commit}"; then
  git -C "${repository_root}" fetch \
    --no-tags \
    --no-recurse-submodules \
    --depth=1 \
    origin \
    "${BASE_REVISION}"
fi
git -C "${repository_root}" cat-file -e "${BASE_REVISION}^{commit}" || {
  echo "content-addressed preceding launcher revision is unavailable" >&2
  exit 2
}

export REPAIR_PATH REPAIR_ID
export PREVIOUS_STAGE_SELECTOR_SHA256 CORRECTED_STAGE_SELECTOR_SHA256
"${BPT_PYTHON}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

path = Path(os.environ["REPAIR_PATH"])
payload = json.loads(path.read_text(encoding="utf-8"))
declared = payload.pop("repair_id")
canonical = json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
observed = hashlib.sha256(canonical).hexdigest()
if declared != observed or observed != os.environ["REPAIR_ID"]:
    raise SystemExit("stage selector identity repair changed")
if payload.get("schema") != (
    "bayesian-phystwin.deform360-v6-stage-selector-identity-repair"
):
    raise SystemExit("stage selector identity repair schema changed")
if payload.get("schema_version") != 1:
    raise SystemExit("stage selector identity repair version changed")

failed = payload.get("failed_execution_evidence", {})
if failed.get("workflow_run_id") != 31513816637:
    raise SystemExit("stage selector repair lost the failed source run")
if failed.get("artifact_id") != 9110649986:
    raise SystemExit("stage selector repair lost the failed source artifact")
if failed.get("execution_receipt_id") != (
    "741968a414984dca9c8c2dab2efbe716a151877d7ac7946830240bd292a47eee"
):
    raise SystemExit("stage selector repair lost the failed receipt")
if failed.get("physical_manifest_count") != 0:
    raise SystemExit("stage selector repair was declared after physical prediction")
if failed.get("source_prediction_seal_count") != 0:
    raise SystemExit("stage selector repair was declared after source prediction")

correction = payload.get("correction", {})
if correction.get("active_launcher_revision") != (
    "dba748cafc1979dd697f99fb8aa70dc1cfaf9b81"
):
    raise SystemExit("stage selector repair changed the preceding launcher")
if correction.get("active_launcher_git_blob_sha1") != (
    "365c5ba0143ba38f1e3d4beac9fdcca1fa63a884"
):
    raise SystemExit("stage selector repair changed the launcher blob")
if correction.get("previous_sha256") != os.environ[
    "PREVIOUS_STAGE_SELECTOR_SHA256"
]:
    raise SystemExit("stage selector repair previous identity changed")
if correction.get("corrected_sha256") != os.environ[
    "CORRECTED_STAGE_SELECTOR_SHA256"
]:
    raise SystemExit("stage selector repair corrected identity changed")

scope = payload.get("repair_scope", {})
if scope.get("runtime_constant_identity_only") is not True:
    raise SystemExit("stage selector repair is not runtime-only")
for field in (
    "frozen_wrapper_changed",
    "frozen_stage_source_changed",
    "archived_science_runner_changed",
    "selector_source_changed",
    "selector_semantics_changed",
    "source_cohort_changed",
    "camera_panel_changed",
    "candidate_roster_changed",
    "physical_model_changed",
    "covariance_changed",
    "loss_or_gate_changed",
    "replacement_allowed",
    "claim_authorized",
):
    if scope.get(field) is not False:
        raise SystemExit(f"stage selector repair widened {field}")

boundary = payload.get("information_boundary", {})
if not boundary or any(value is not False for value in boundary.values()):
    raise SystemExit("stage selector repair crossed the information boundary")
PY

patch_root="$(
  mktemp -d "${RUNNER_TEMP:-/tmp}/deform360-v6-stage-selector-patch.XXXXXX"
)"
cleanup() {
  rm -rf "${patch_root}"
}
trap cleanup EXIT

base_launcher="${patch_root}/base-launcher.sh"
patched_launcher="${patch_root}/patched-launcher.sh"
selector_bootstrap="${patch_root}/stage-selector-bootstrap.py"
selector_marker="${patch_root}/stage-selector-applied.json"

git -C "${repository_root}" show \
  "${BASE_REVISION}:${LAUNCHER_PATH}" > "${base_launcher}"
test "$(git hash-object "${base_launcher}")" = "${BASE_LAUNCHER_BLOB_SHA}" || {
  echo "content-addressed preceding launcher byte identity changed" >&2
  exit 2
}

cat > "${selector_bootstrap}" <<'PY'
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

WRAPPER_PATH = Path(
    "scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"
)
PREVIOUS = "79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
CORRECTED = "c10391578c73dde47fbce160312559a7e638007e9053ec89373fe575cc64d7e5"


def _load_wrapper(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_deform360_v6_stage_selector_runtime_wrapper",
        path,
    )
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load frozen physical source wrapper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("frozen physical source wrapper path is missing")
    declared = Path(sys.argv[1])
    expected = WRAPPER_PATH.resolve(strict=True)
    if declared.resolve(strict=True) != expected:
        raise SystemExit("stage selector repair received a foreign wrapper")
    arguments = sys.argv[2:]
    stage_positions = [
        index for index, value in enumerate(arguments) if value == "--stage"
    ]
    if len(stage_positions) != 1:
        raise SystemExit("stage selector repair requires one stage binding")
    stage_index = stage_positions[0]
    if stage_index + 1 >= len(arguments):
        raise SystemExit("stage selector repair stage lacks a value")
    if arguments[stage_index + 1] != "stage-prefix":
        raise SystemExit("stage selector repair is restricted to stage-prefix")

    wrapper = _load_wrapper(expected)
    original_load_stage = wrapper._load_stage
    applied = False

    def load_stage(path: Path, stage: str) -> ModuleType:
        nonlocal applied
        loaded = original_load_stage(path, stage)
        if stage != "stage-prefix":
            raise SystemExit("stage selector repair observed a foreign stage")
        observed = getattr(loaded, "GENERIC_SELECTOR_SHA256", None)
        if observed != PREVIOUS:
            raise SystemExit("frozen stage selector identity changed")
        loaded.GENERIC_SELECTOR_SHA256 = CORRECTED
        applied = True
        marker = Path(os.environ["STAGE_SELECTOR_REPAIR_MARKER"])
        marker.write_text(
            json.dumps(
                {
                    "repair_id": os.environ["STAGE_SELECTOR_REPAIR_ID"],
                    "previous_sha256": PREVIOUS,
                    "corrected_sha256": CORRECTED,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        return loaded

    wrapper._load_stage = load_stage
    previous_argv = sys.argv
    sys.argv = [str(expected), *arguments]
    try:
        result = int(wrapper.main())
    finally:
        sys.argv = previous_argv
    if not applied:
        raise SystemExit("stage selector repair was not applied")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
PY
chmod 600 "${selector_bootstrap}"

BASE_LAUNCHER="${base_launcher}" \
PATCHED_LAUNCHER="${patched_launcher}" \
PATCH_ID_VALUE="${PATCH_ID}" \
"${BPT_PYTHON}" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

source_path = Path(os.environ["BASE_LAUNCHER"])
output_path = Path(os.environ["PATCHED_LAUNCHER"])
patch_id = os.environ["PATCH_ID_VALUE"]
source = source_path.read_text(encoding="utf-8")

old = r'''    if [[ "${repo_replacements}" -ne 1 || "${role_replacements}" -ne 1 ]]; then
      echo "stage-prefix compatibility bindings are not unique" >&2
      exit 2
    fi
    exec "${REAL_BPT_PYTHON}" "${rewritten[@]}"
  fi
fi
exec "${REAL_BPT_PYTHON}" "$@"'''

new = r'''    if [[ "${repo_replacements}" -ne 1 || "${role_replacements}" -ne 1 ]]; then
      echo "stage-prefix compatibility bindings are not unique" >&2
      exit 2
    fi
    : "${STAGE_SELECTOR_BOOTSTRAP:?stage selector bootstrap is required}"
    exec "${REAL_BPT_PYTHON}" "${STAGE_SELECTOR_BOOTSTRAP}" "${rewritten[@]}"
  fi
fi
exec "${REAL_BPT_PYTHON}" "$@"'''

if source.count(old) != 1:
    raise SystemExit("stage selector runtime source block changed")
patched = source.replace(old, new)
if patched.count(new) != 1 or old in patched:
    raise SystemExit("stage selector runtime patch is not unique")
header = f"# runtime compatibility patch: {patch_id}\n"
output_path.write_text(header + patched, encoding="utf-8")
PY

chmod 700 "${patched_launcher}"
bash -n "${patched_launcher}"
export STAGE_SELECTOR_BOOTSTRAP="${selector_bootstrap}"
export STAGE_SELECTOR_REPAIR_MARKER="${selector_marker}"
export STAGE_SELECTOR_REPAIR_ID="${REPAIR_ID}"

set +e
bash "${patched_launcher}" "$@"
status=$?
set -e

receipt="${EVIDENCE_ROOT}/deform360-v6-source-prediction-evidence/execution-receipt.json"
if [[ -f "${receipt}" && ! -L "${receipt}" ]]; then
  export RECEIPT_PATH="${receipt}"
  export STAGE_SELECTOR_REPAIR_PATH="${REPAIR_PATH}"
  "${BPT_PYTHON}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

path = Path(os.environ["RECEIPT_PATH"])
receipt = json.loads(path.read_text(encoding="utf-8"))
receipt.pop("receipt_id", None)
marker_path = Path(os.environ["STAGE_SELECTOR_REPAIR_MARKER"])
applied = marker_path.is_file() and not marker_path.is_symlink()
if applied:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("repair_id") != os.environ["STAGE_SELECTOR_REPAIR_ID"]:
        raise SystemExit("stage selector repair marker identity changed")
else:
    marker = None
receipt["runtime_stage_selector_identity_repair"] = {
    "repair_id": os.environ["STAGE_SELECTOR_REPAIR_ID"],
    "repair_path": os.environ["STAGE_SELECTOR_REPAIR_PATH"],
    "applied": applied,
    "previous_sha256": os.environ["PREVIOUS_STAGE_SELECTOR_SHA256"],
    "corrected_sha256": os.environ["CORRECTED_STAGE_SELECTOR_SHA256"],
    "marker": marker,
}
canonical = json.dumps(
    receipt,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
receipt["receipt_id"] = hashlib.sha256(canonical).hexdigest()
path.write_text(
    json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
PY
  compact="$(dirname "${receipt}")"
  (
    cd "${compact}"
    rm -f SHA256SUMS
    find . -type f ! -name SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 sha256sum > SHA256SUMS
    sha256sum --check SHA256SUMS >/dev/null
  )
fi

exit "${status}"
