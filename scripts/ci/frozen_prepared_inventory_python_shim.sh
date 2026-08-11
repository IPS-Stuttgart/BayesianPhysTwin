#!/usr/bin/env bash
set -euo pipefail

: "${REAL_BPT_PYTHON:?REAL_BPT_PYTHON is required}"
: "${FROZEN_PREPARED_INVENTORY:?FROZEN_PREPARED_INVENTORY is required}"
: "${FROZEN_INVENTORY_REUSE_AMENDMENT:?FROZEN_INVENTORY_REUSE_AMENDMENT is required}"

EXPECTED_FILE_SHA256="4da96c4f636d195f7aea5d971fbd83bd3b0f35b1c66a77af68007bbd08a69007"
EXPECTED_INVENTORY_ID="6994aa621b38dc8fb21cd38e43363bde3ea12dd644532addeecfc07a30f84e7b"
EXPECTED_AMENDMENT_ID="1d4087e22d7c7cd3fcec09c6f392427c90c0eaa5adbb8d12e35b89e215dd5ed9"
INVENTORY_COMMAND="scripts/science/inventory_deform360_calibration_prepared_source.py"

if [[ "${1:-}" != "${INVENTORY_COMMAND}" ]]; then
  exec "${REAL_BPT_PYTHON}" "$@"
fi

output=""
arguments=("$@")
index=0
while [[ "${index}" -lt "${#arguments[@]}" ]]; do
  if [[ "${arguments[${index}]}" == "--output" ]]; then
    next=$((index + 1))
    [[ "${next}" -lt "${#arguments[@]}" ]] \
      || { echo "prepared inventory output argument is missing" >&2; exit 2; }
    output="${arguments[${next}]}"
  fi
  index=$((index + 1))
done
[[ -n "${output}" ]] \
  || { echo "prepared inventory output path is missing" >&2; exit 2; }

test -f "${FROZEN_PREPARED_INVENTORY}"
test ! -L "${FROZEN_PREPARED_INVENTORY}"
test -f "${FROZEN_INVENTORY_REUSE_AMENDMENT}"
test ! -L "${FROZEN_INVENTORY_REUSE_AMENDMENT}"

export EXPECTED_FILE_SHA256 EXPECTED_INVENTORY_ID EXPECTED_AMENDMENT_ID
"${REAL_BPT_PYTHON}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

inventory_path = Path(os.environ["FROZEN_PREPARED_INVENTORY"])
amendment_path = Path(os.environ["FROZEN_INVENTORY_REUSE_AMENDMENT"])

inventory_bytes = inventory_path.read_bytes()
if hashlib.sha256(inventory_bytes).hexdigest() != os.environ["EXPECTED_FILE_SHA256"]:
    raise SystemExit("frozen prepared inventory file identity changed")
inventory = json.loads(inventory_bytes)
if inventory.get("inventory_id") != os.environ["EXPECTED_INVENTORY_ID"]:
    raise SystemExit("frozen prepared inventory content identity changed")
if inventory.get("implementation_revision") != (
    "e190c94014e6024e324d860618662526af6ea682"
):
    raise SystemExit("frozen prepared inventory implementation revision changed")
if inventory.get("object_count") != 10 or len(inventory.get("objects", [])) != 10:
    raise SystemExit("frozen prepared inventory cohort changed")
if inventory.get("status") != "complete-calibration-only-prepared-source":
    raise SystemExit("frozen prepared inventory status changed")
boundary = inventory.get("information_boundary", {})
expected_boundary = {
    "calibration_camera_payloads_opened": True,
    "calibration_robot_state_opened": True,
    "calibration_tactile_payloads_opened": True,
    "calibration_target_metrics_computed": False,
    "confirmation_payloads_opened": False,
    "replacement_allowed": False,
    "target_outcomes_used": False,
}
if boundary != expected_boundary:
    raise SystemExit("frozen prepared inventory information boundary changed")

amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
declared = amendment.pop("amendment_id")
canonical = json.dumps(
    amendment,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
observed = hashlib.sha256(canonical).hexdigest()
if declared != observed or observed != os.environ["EXPECTED_AMENDMENT_ID"]:
    raise SystemExit("frozen inventory reuse amendment identity changed")
if amendment.get("claim_authorized") is not False:
    raise SystemExit("frozen inventory reuse authorized a claim")
if any(amendment.get("information_boundary", {}).values()):
    raise SystemExit("frozen inventory reuse crossed an information boundary")
frozen = amendment.get("frozen_inventory", {})
if frozen.get("file_sha256") != os.environ["EXPECTED_FILE_SHA256"]:
    raise SystemExit("frozen inventory reuse file binding changed")
if frozen.get("inventory_id") != os.environ["EXPECTED_INVENTORY_ID"]:
    raise SystemExit("frozen inventory reuse content binding changed")
comparison = amendment.get("semantic_comparison", {})
if comparison.get("differing_paths") != ["implementation_revision", "inventory_id"]:
    raise SystemExit("frozen inventory semantic comparison changed")
for field in (
    "all_object_records_identical",
    "all_source_artifact_hashes_identical",
    "all_selection_and_provider_locks_identical",
    "all_information_boundary_fields_identical",
):
    if comparison.get(field) is not True:
        raise SystemExit(f"frozen inventory comparison lost {field}")
if comparison.get("cohort_or_payload_change_detected") is not False:
    raise SystemExit("frozen inventory comparison reports a payload change")
PY

destination="${output}"
mkdir -p "$(dirname "${destination}")"
test ! -e "${destination}"
cp --reflink=auto --preserve=mode,timestamps \
  "${FROZEN_PREPARED_INVENTORY}" \
  "${destination}"
test ! -L "${destination}"
test "$(sha256sum "${destination}" | awk '{print $1}')" = \
  "${EXPECTED_FILE_SHA256}"
printf '%s\n' \
  "{\"complete\":true,\"inventory_id\":\"${EXPECTED_INVENTORY_ID}\",\"object_count\":10,\"reuse_amendment_id\":\"${EXPECTED_AMENDMENT_ID}\",\"status\":\"exact-frozen-prepared-inventory-reused\"}"
