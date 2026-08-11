#!/usr/bin/env bash
set -euo pipefail

SELECTOR_WRAPPER="scripts/ci/archive/run_deform360_v6_source_prediction_evidence_selector_repair_v3.sh"
SELECTOR_WRAPPER_BLOB_SHA="5958db6362917e6bc355b194abdac4736e39a5a4"
SCIENCE_RUNNER="scripts/ci/archive/run_deform360_v6_source_prediction_evidence_v2.sh"
SCIENCE_RUNNER_BLOB_SHA="42dd4f3e0d05f18b9ff0a0bdcf90fbd282f0f6f1"
PHYSICAL_UPSTREAM_REVISION="9f69d5d6c5d81d6d6e8f123c18ddba73dc4afa65"
PHYSICAL_UPSTREAM_DIAGNOSTIC_RUN_ID="31461017011"
PHYSICAL_UPSTREAM_REPORT_ID="75c1be85233e1835dfef5a1227a28e8938995335ead701fe8d3dfd8b5960a087"
PREPARED_INVENTORY_IMPLEMENTATION_REVISION="e190c94014e6024e324d860618662526af6ea682"
PREPARED_INVENTORY_ID="6994aa621b38dc8fb21cd38e43363bde3ea12dd644532addeecfc07a30f84e7b"
PREPARED_INVENTORY_FILE_SHA256="4da96c4f636d195f7aea5d971fbd83bd3b0f35b1c66a77af68007bbd08a69007"
PREPARED_INVENTORY_ADMISSION_RUN_ID="31272512658"

# The delegated selector wrapper preserves these reviewed invariants verbatim:
# REPAIR_ID="d7e516ced90469589c3e4c3c12672a503fe8bbdb3a6f3316d852c266fd0f3d90"
# ARCHIVED_RUNNER_BLOB_SHA="42dd4f3e0d05f18b9ff0a0bdcf90fbd282f0f6f1"
# PREVIOUS_SELECTOR_SHA256="79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
# CORRECTED_SELECTOR_SHA256="c10391578c73dde47fbce160312559a7e638007e9053ec89373fe575cc64d7e5"
# text.count(old) != 1
# patched.count(new) != 1
# "runtime_identity_repair_id"
# "runtime_selector_identity"
# prediction_record_count") != 100
# source-prediction-evidence-sealed
# source-inputs-incomplete
# source-technical-failure-retained
# run_deform360_joint_sparse_source_predictions_v5.py
# SAM2_CHECKPOINT_NAME="sam2.1_hiera_small.pt"
# SAM2_CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/${SAM2_CHECKPOINT_NAME}"
# SAM2_SHA256="6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
# development_suffix_opened": False
# v6_target_payloads_opened": False
# fresh_target_selection_authorized": False

materialize_frozen_physical_upstream() {
  local repository="$1"
  local output_root="$2"
  local python_bin="${BPT_PYTHON:-python}"

  REPOSITORY_ROOT="${repository}" \
  OUTPUT_ROOT="${output_root}" \
  SCIENCE_RUNNER_PATH="${SCIENCE_RUNNER}" \
  PHYSICAL_UPSTREAM_REVISION_VALUE="${PHYSICAL_UPSTREAM_REVISION}" \
  "${python_bin}" - <<'PY'
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import subprocess

repository = Path(os.environ["REPOSITORY_ROOT"]).resolve()
output_root = Path(os.environ["OUTPUT_ROOT"]).resolve()
science_runner = Path(os.environ["SCIENCE_RUNNER_PATH"]).resolve()
revision = os.environ["PHYSICAL_UPSTREAM_REVISION_VALUE"]

if not (repository / ".git").exists():
    raise SystemExit("BayesianPhysTwin git repository is unavailable")
commit_object = f"{revision}^{{commit}}"
available = subprocess.run(
    ["git", "-C", str(repository), "cat-file", "-e", commit_object],
    check=False,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
if available.returncode != 0:
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "fetch",
            "--no-tags",
            "--no-recurse-submodules",
            "--depth=1",
            "origin",
            revision,
        ],
        check=True,
    )
subprocess.run(
    ["git", "-C", str(repository), "cat-file", "-e", commit_object],
    check=True,
)

source = science_runner.read_text(encoding="utf-8")
stage = source.index('set_stage "locate-frozen-physical-upstream"')
start = source.index("required = {", stage) + len("required = ")
end = source.index("\nroot = Path(", start)
required = ast.literal_eval(source[start:end])
expected_paths = {
    "scripts/remote/run_deform360_official_phystwin_smoke.py",
    "src/causal4d_public/deform360_reusable_graph.py",
    "src/causal4d_public/deform360_partial_graph_state.py",
    "src/causal4d_public/deform360_dense_reusable_panel.py",
    "src/causal4d_public/deform360_action_support.py",
    "src/causal4d_public/deform360_contact_conditioned_action.py",
    "src/causal4d_public/deform360_dense_source.py",
    "src/bayesian_phystwin/phystwin_graph.py",
    "configs/causal4d_public/deform360_dense_reusable_panel_v1.json",
    "configs/causal4d_public/deform360_independent_source_split_v1.json",
}
if set(required) != expected_paths or len(required) != 10:
    raise SystemExit("frozen physical-upstream roster changed")

for relative, expected in sorted(required.items()):
    content = subprocess.check_output(
        ["git", "-C", str(repository), "show", f"{revision}:{relative}"],
        stderr=subprocess.DEVNULL,
    )
    if hashlib.sha256(content).hexdigest() != expected:
        raise SystemExit(f"pinned physical-upstream byte identity changed: {relative}")

if output_root.exists() and any(output_root.iterdir()):
    raise SystemExit("physical-upstream output root is not empty")
output_root.mkdir(parents=True, exist_ok=True)
archive = subprocess.Popen(
    [
        "git",
        "-C",
        str(repository),
        "archive",
        revision,
        "--",
        "configs/causal4d_public",
        "scripts/remote",
        "src/bayesian_phystwin",
        "src/causal4d_public",
    ],
    stdout=subprocess.PIPE,
)
if archive.stdout is None:
    raise SystemExit("cannot open historical archive stream")
extract = subprocess.run(
    ["tar", "-xf", "-", "-C", str(output_root)],
    stdin=archive.stdout,
    check=False,
)
archive.stdout.close()
archive_status = archive.wait()
if archive_status != 0 or extract.returncode != 0:
    raise SystemExit("cannot materialize frozen physical-upstream tree")

for relative, expected in sorted(required.items()):
    path = output_root / relative
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"materialized frozen source is invalid: {relative}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"materialized frozen source changed: {relative}")
PY
}

if [[ "${1:-}" == "--materialize-physical-upstream" ]]; then
  test -n "${2:-}"
  test -n "${3:-}"
  materialize_frozen_physical_upstream "$2" "$3"
  printf '%s\n' "${PHYSICAL_UPSTREAM_REVISION}"
  exit 0
fi

: "${BPT_PYTHON:?BPT_PYTHON is required}"
: "${EVIDENCE_ROOT:?EVIDENCE_ROOT is required}"

test -f "${SELECTOR_WRAPPER}"
test ! -L "${SELECTOR_WRAPPER}"
test "$(git hash-object "${SELECTOR_WRAPPER}")" = "${SELECTOR_WRAPPER_BLOB_SHA}"
test -f "${SCIENCE_RUNNER}"
test ! -L "${SCIENCE_RUNNER}"
test "$(git hash-object "${SCIENCE_RUNNER}")" = "${SCIENCE_RUNNER_BLOB_SHA}"

export PREPARED_INVENTORY_IMPLEMENTATION_REVISION
export PREPARED_INVENTORY_ID
export PREPARED_INVENTORY_FILE_SHA256
export PREPARED_INVENTORY_ADMISSION_RUN_ID
export SCIENCE_RUNNER
"${BPT_PYTHON}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

lock_path = Path(
    "protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json"
)
lock = json.loads(lock_path.read_text(encoding="utf-8"))
prepared = lock.get("physical_baseline", {}).get("prepared_source_inventory", {})
if prepared != {
    "file_sha256": os.environ["PREPARED_INVENTORY_FILE_SHA256"],
    "inventory_id": os.environ["PREPARED_INVENTORY_ID"],
}:
    raise SystemExit("locked prepared-source inventory identity changed")

runner = Path(os.environ["SCIENCE_RUNNER"]).read_text(encoding="utf-8")
needle = '--implementation-revision "${BPT_SOURCE_SHA}"'
if runner.count(needle) != 1:
    raise SystemExit("archived prepared-inventory implementation binding changed")
PY

PHYSICAL_UPSTREAM_ROOT="$(
  mktemp -d "${RUNNER_TEMP:-/tmp}/deform360-v6-physical-upstream.XXXXXX"
)"
PYTHON_SHIM="$(
  mktemp "${RUNNER_TEMP:-/tmp}/deform360-v6-python-shim.XXXXXX"
)"
REAL_BPT_PYTHON="${BPT_PYTHON}"
cat > "${PYTHON_SHIM}" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
: "${REAL_BPT_PYTHON:?REAL_BPT_PYTHON is required}"
: "${PREPARED_INVENTORY_IMPLEMENTATION_REVISION:?prepared inventory revision is required}"

target="scripts/science/inventory_deform360_calibration_prepared_source.py"
if [[ "${1:-}" == "${target}" ]]; then
  rewritten=()
  replacements=0
  while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == "--implementation-revision" ]]; then
      [[ "$#" -ge 2 ]] || {
        echo "prepared inventory implementation revision lacks a value" >&2
        exit 2
      }
      rewritten+=("$1" "${PREPARED_INVENTORY_IMPLEMENTATION_REVISION}")
      replacements=$((replacements + 1))
      shift 2
    else
      rewritten+=("$1")
      shift
    fi
  done
  if [[ "${replacements}" -ne 1 ]]; then
    echo "prepared inventory implementation binding is not unique" >&2
    exit 2
  fi
  exec "${REAL_BPT_PYTHON}" "${rewritten[@]}"
fi
exec "${REAL_BPT_PYTHON}" "$@"
SH
chmod 700 "${PYTHON_SHIM}"

cleanup() {
  rm -rf "${PHYSICAL_UPSTREAM_ROOT}"
  rm -f "${PYTHON_SHIM}"
}
trap cleanup EXIT

materialize_frozen_physical_upstream \
  "${GITHUB_WORKSPACE:-.}" \
  "${PHYSICAL_UPSTREAM_ROOT}"
echo "materialized frozen physical upstream revision=${PHYSICAL_UPSTREAM_REVISION}"
echo "bound prepared inventory to authoritative admission revision=${PREPARED_INVENTORY_IMPLEMENTATION_REVISION}"

export PHYSICAL_UPSTREAM_REVISION
export PHYSICAL_UPSTREAM_DIAGNOSTIC_RUN_ID
export PHYSICAL_UPSTREAM_REPORT_ID
export REAL_BPT_PYTHON

set +e
BPT_PYTHON="${PYTHON_SHIM}" \
RUNNER_WORKSPACE="${PHYSICAL_UPSTREAM_ROOT}" \
  bash "${SELECTOR_WRAPPER}"
status=$?
set -e

receipt="${EVIDENCE_ROOT}/deform360-v6-source-prediction-evidence/execution-receipt.json"
if [[ -f "${receipt}" && ! -L "${receipt}" ]]; then
  export RECEIPT_PATH="${receipt}"
  "${BPT_PYTHON}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

path = Path(os.environ["RECEIPT_PATH"])
receipt = json.loads(path.read_text(encoding="utf-8"))
receipt.pop("receipt_id", None)
receipt["runtime_physical_upstream"] = {
    "repository": "IPS-Stuttgart/BayesianPhysTwin",
    "repository_revision": os.environ["PHYSICAL_UPSTREAM_REVISION"],
    "diagnostic_workflow_run_id": int(os.environ["PHYSICAL_UPSTREAM_DIAGNOSTIC_RUN_ID"]),
    "diagnostic_report_id": os.environ["PHYSICAL_UPSTREAM_REPORT_ID"],
    "selection": "unique-complete-history-exact-ten-file-sha256-match",
    "required_file_count": 10,
}
receipt["runtime_prepared_inventory_identity"] = {
    "authoritative_admission_workflow_run_id": int(
        os.environ["PREPARED_INVENTORY_ADMISSION_RUN_ID"]
    ),
    "implementation_revision": os.environ[
        "PREPARED_INVENTORY_IMPLEMENTATION_REVISION"
    ],
    "inventory_id": os.environ["PREPARED_INVENTORY_ID"],
    "file_sha256": os.environ["PREPARED_INVENTORY_FILE_SHA256"],
    "selection": "frozen-authoritative-retained-source-admission",
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
