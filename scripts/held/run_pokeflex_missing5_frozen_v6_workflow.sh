#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_ROOT:?DATASET_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${GITHUB_SHA:?GITHUB_SHA is required}"

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
V5_SCRIPT="$REPOSITORY_ROOT/scripts/held/run_pokeflex_missing5_v5.py"
V6_SCRIPT="$REPOSITORY_ROOT/scripts/held/run_pokeflex_missing5_v6.py"
PARENT_PROTOCOL="$REPOSITORY_ROOT/configs/sota/pokeflex_action_robust_official18_v4.json"
V5_EXECUTION="$REPOSITORY_ROOT/configs/sota/pokeflex_missing5_execution_v5.json"
V6_EXECUTION="$REPOSITORY_ROOT/configs/sota/pokeflex_missing5_execution_v6.json"
COMPLETION_PROTOCOL="$REPOSITORY_ROOT/configs/sota/pokeflex_missing5_scale_completion_v5.json"
CAUSAL_MODEL="$REPOSITORY_ROOT/configs/sota/pokeflex_missing5_causal_scale_v6.json"
CAUSAL_SOURCE_RESULT="$REPOSITORY_ROOT/results/sota/pokeflex_missing5_causal_scale_v6/source_result.json"
PUBLIC13_RESULT="$REPOSITORY_ROOT/results/sota/pokeflex_action_robust_all18_v4_public13_retrospective/result.json"
PREDICTION_ENV_ROOT="/home/florianpfaff/pokeflex-action-robust-official13-v1-a9e38c2"
UPSTREAM_COMMIT="aaa8726072834a95bbe97e1a113588968c36e185"

TAKES=(
  Pillow_T8
  3dPrintedCylinder_T7
  3dPrintedHeart_T14
  Sponge_T10
  3dPrintedPizza_T13
)

mkdir -p "$RUN_ROOT"
for child in manifest stages v5-predictions v6-predictions barriers results logs runtime; do
  test ! -e "$RUN_ROOT/$child"
  mkdir "$RUN_ROOT/$child"
done

STATUS_PATH="$RUN_ROOT/status.json"
write_status() {
  local stage="$1"
  local target_payload_opened="$2"
  python3 - "$STATUS_PATH" "$stage" "$target_payload_opened" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
value = {
    "schema": "bayesian-phystwin/pokeflex-missing5-v6-workflow-status-v1",
    "stage": sys.argv[2],
    "target_payload_opened": sys.argv[3].lower() == "true",
    "target_metric_computed": sys.argv[2] == "complete",
    "github_sha": os.environ["GITHUB_SHA"],
    "github_run_id": os.environ.get("GITHUB_RUN_ID"),
    "runner_name": os.environ.get("RUNNER_NAME"),
    "dataset_root": os.environ["DATASET_ROOT"],
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
}
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

write_status initialized false

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "dataset root is absent: $DATASET_ROOT" >&2
  write_status dataset-root-absent false
  exit 2
fi
zip_count="$(find "$DATASET_ROOT" -type f -iname '*.zip' -printf '.' | wc -c)"
if [[ "$zip_count" -ne 170 ]]; then
  echo "expected 170 verified ZIPs, found $zip_count" >&2
  write_status archive-count-mismatch false
  exit 3
fi
for take in "${TAKES[@]}"; do
  mapfile -t matches < <(find "$DATASET_ROOT" -type f -name "$take.zip" -print)
  if [[ "${#matches[@]}" -ne 1 ]]; then
    echo "expected exactly one $take.zip, found ${#matches[@]}" >&2
    write_status registered-archive-unavailable false
    exit 4
  fi
done
write_status registered-archives-present false

# Reuse a numerical environment that has already executed the sealed public-13
# predictor. No environment is mutated by this workflow.
science_python=""
python_candidates=(
  "$PREDICTION_ENV_ROOT/venv/bin/python"
  "$PREDICTION_ENV_ROOT/.venv/bin/python"
  /home/florianpfaff/.venvs/pokeflex/bin/python
  /home/florianpfaff/miniconda3/envs/pokeflex/bin/python
  /home/florianpfaff/anaconda3/envs/pokeflex/bin/python
  /usr/local/bin/python3.11
  /usr/bin/python3.11
  /usr/local/bin/python3
  /usr/bin/python3
)
while IFS= read -r candidate; do
  python_candidates+=("$candidate")
done < <(
  find /home/florianpfaff /home/github-runner/.cache \
    -path '*/datasets' -prune -o \
    -path '*/datasets/*' -prune -o \
    -path '*/bin/python' -type f -print 2>/dev/null | sort -u
)
for candidate in "${python_candidates[@]}"; do
  if [[ -x "$candidate" ]] && "$candidate" - <<'PY' >/dev/null 2>&1
import cv2
import numpy
import scipy
import torch
import trimesh
assert torch.cuda.is_available()
PY
  then
    science_python="$candidate"
    break
  fi
done
if [[ -z "$science_python" ]]; then
  echo "no retained CUDA PokeFlex Python environment was found" >&2
  write_status numerical-environment-unavailable false
  exit 5
fi
export SCIENCE_PYTHON="$science_python"
export PYTHONPATH="$REPOSITORY_ROOT/src:$REPOSITORY_ROOT/scripts/held:$REPOSITORY_ROOT/scripts/remote${PYTHONPATH:+:$PYTHONPATH}"

# Resolve the three exact released checkpoint files by their frozen hashes.
checkpoint_root="$($SCIENCE_PYTHON - <<'PY'
from __future__ import annotations

import hashlib
import os
from pathlib import Path

expected = {
    "attention_model.pth": "51181c22d7ad9fcc194a48411fda64759bdfb491c73abfa94f63d0a7167284fe",
    "decoder.pth": "34a29ab89912ffdd0ea2a4436bcaca0e843d2c51a19f77c88844702b596b46cf",
    "pointcloud_encoder.pth": "3053f0656e4ca61645aa194e2d33540f68953efe9fd0cbab062ec561c405609b",
}

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

roots = [
    Path("/home/florianpfaff/pokeflex-action-robust-official13-v1-a9e38c2"),
    Path("/home/florianpfaff"),
    Path("/home/github-runner/.cache"),
]
candidates: set[Path] = set()
for root in roots:
    if not root.exists():
        continue
    for directory, names, files in os.walk(root, followlinks=False):
        current = Path(directory)
        relative_depth = len(current.relative_to(root).parts)
        names[:] = [
            name
            for name in names
            if name not in {".git", "datasets"}
            and not (current / name).is_symlink()
            and relative_depth < 9
        ]
        if "pointcloud_encoder.pth" in files:
            candidates.add(current)
valid = []
for candidate in sorted(candidates):
    if all((candidate / name).is_file() for name in expected) and all(
        digest(candidate / name) == expected[name] for name in expected
    ):
        valid.append(candidate.resolve())
if not valid:
    raise SystemExit("no exact released PokeFlex checkpoint root found")
print(valid[0])
PY
)" || {
  write_status checkpoint-unavailable false
  exit 6
}
export CHECKPOINT_ROOT="$checkpoint_root"

# Use a fresh clean checkout of the exact registered upstream implementation.
UPSTREAM_CHECKOUT="$RUN_ROOT/runtime/reconstruction"
git clone --quiet https://github.com/pokeflex-dataset/reconstruction.git "$UPSTREAM_CHECKOUT"
git -C "$UPSTREAM_CHECKOUT" checkout --quiet --detach "$UPSTREAM_COMMIT"
test "$(git -C "$UPSTREAM_CHECKOUT" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_CHECKOUT" status --porcelain)"
export UPSTREAM_CHECKOUT

{
  echo "runner_name=$RUNNER_NAME"
  echo "runner_os=$RUNNER_OS"
  echo "runner_arch=$RUNNER_ARCH"
  echo "github_sha=$GITHUB_SHA"
  echo "dataset_root=$DATASET_ROOT"
  echo "zip_count=$zip_count"
  echo "science_python=$SCIENCE_PYTHON"
  echo "checkpoint_root=$CHECKPOINT_ROOT"
  echo "upstream_checkout=$UPSTREAM_CHECKOUT"
  echo "upstream_commit=$(git -C "$UPSTREAM_CHECKOUT" rev-parse HEAD)"
  "$SCIENCE_PYTHON" --version
  "$SCIENCE_PYTHON" - <<'PY'
import cv2
import numpy
import scipy
import torch
import trimesh
print(f"numpy={numpy.__version__}")
print(f"scipy={scipy.__version__}")
print(f"torch={torch.__version__}")
print(f"trimesh={trimesh.__version__}")
print(f"opencv={cv2.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
PY
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  df -h "$DATASET_ROOT" "$RUN_ROOT"
} | tee "$RUN_ROOT/environment.txt"

"$SCIENCE_PYTHON" "$V5_SCRIPT" --help >/dev/null
"$SCIENCE_PYTHON" "$V6_SCRIPT" --help >/dev/null
write_status runtime-bound false

# Build the immutable author-source manifest from ZIP names, central-directory
# metadata, and archive hashes. No member payload is decoded in this stage.
SOURCE_MANIFEST="$RUN_ROOT/manifest/source-manifest.json"
export SOURCE_MANIFEST PARENT_PROTOCOL
"$SCIENCE_PYTHON" - <<'PY'
import json
import os
from pathlib import Path
from bayesian_phystwin.pokeflex_action_robust_official18_v4 import (
    build_author_source_manifest,
    load_official18_v4_protocol,
)
protocol = load_official18_v4_protocol(Path(os.environ["PARENT_PROTOCOL"]))
manifest = build_author_source_manifest(Path(os.environ["DATASET_ROOT"]), protocol)
Path(os.environ["SOURCE_MANIFEST"]).write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({
    "source_manifest_sha256": manifest["source_manifest_sha256"],
    "take_count": len(manifest["takes"]),
    "member_payload_decoded": manifest["member_payload_decoded"],
    "target_geometry_decoded": manifest["target_geometry_decoded"],
}, indent=2))
PY
write_status source-manifest-sealed false

# Stage only causal prediction inputs and the single task-authorized template
# mesh for each take. Future target meshes remain unopened.
for take in "${TAKES[@]}"; do
  archive="$(find "$DATASET_ROOT" -type f -name "$take.zip" -print -quit)"
  "$SCIENCE_PYTHON" "$V5_SCRIPT" \
    --source-manifest "$SOURCE_MANIFEST" \
    stage "$archive" "$RUN_ROOT/stages/$take" \
    >"$RUN_ROOT/logs/$take-stage.log" 2>&1
done
write_status causal-inputs-staged true

run_pairwise() {
  local operation="$1"
  local -a pids=()
  local -a names=()
  local index=0
  local failure=0
  for take in "${TAKES[@]}"; do
    local gpu=$((index % 2))
    if [[ "$operation" == "v5" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" "$SCIENCE_PYTHON" "$V5_SCRIPT" \
        --source-manifest "$SOURCE_MANIFEST" \
        predict "$RUN_ROOT/stages/$take" "$RUN_ROOT/v5-predictions/$take" \
        --upstream-checkout "$UPSTREAM_CHECKOUT" \
        --checkpoint-root "$CHECKPOINT_ROOT" \
        >"$RUN_ROOT/logs/$take-v5-predict.log" 2>&1 &
    else
      CUDA_VISIBLE_DEVICES="$gpu" "$SCIENCE_PYTHON" "$V6_SCRIPT" \
        --source-manifest "$SOURCE_MANIFEST" \
        augment "$RUN_ROOT/stages/$take" \
        "$RUN_ROOT/v5-predictions/$take" \
        "$RUN_ROOT/v6-predictions/$take" \
        >"$RUN_ROOT/logs/$take-v6-augment.log" 2>&1 &
    fi
    pids+=("$!")
    names+=("$take")
    index=$((index + 1))
    if [[ "${#pids[@]}" -eq 2 ]]; then
      for offset in "${!pids[@]}"; do
        if ! wait "${pids[$offset]}"; then
          echo "$operation failed for ${names[$offset]}" >&2
          failure=1
        fi
      done
      pids=()
      names=()
      test "$failure" -eq 0
    fi
  done
  for offset in "${!pids[@]}"; do
    if ! wait "${pids[$offset]}"; then
      echo "$operation failed for ${names[$offset]}" >&2
      failure=1
    fi
  done
  test "$failure" -eq 0
}

run_pairwise v5
"$SCIENCE_PYTHON" "$V5_SCRIPT" \
  --source-manifest "$SOURCE_MANIFEST" \
  barrier "$RUN_ROOT/v5-predictions" "$RUN_ROOT/barriers/v5-barrier.json" \
  >"$RUN_ROOT/logs/v5-barrier.log" 2>&1
write_status v5-all-five-barrier-sealed true

run_pairwise v6
"$SCIENCE_PYTHON" "$V6_SCRIPT" \
  --source-manifest "$SOURCE_MANIFEST" \
  barrier "$RUN_ROOT/v5-predictions" "$RUN_ROOT/v6-predictions" \
  "$RUN_ROOT/barriers/v6-barrier.json" \
  >"$RUN_ROOT/logs/v6-barrier.log" 2>&1
write_status v6-all-five-barrier-sealed true

# Record a joint pre-scoring seal over every prediction and both barriers before
# any future target mesh is decoded.
export RUN_ROOT
"$SCIENCE_PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
paths = [root / "manifest/source-manifest.json"]
paths.extend(sorted((root / "v5-predictions").glob("*/seal.json")))
paths.extend(sorted((root / "v5-predictions").glob("*/prediction.npz")))
paths.append(root / "barriers/v5-barrier.json")
paths.extend(sorted((root / "v6-predictions").glob("*/seal.json")))
paths.extend(sorted((root / "v6-predictions").glob("*/prediction.npz")))
paths.append(root / "barriers/v6-barrier.json")
records = []
for path in paths:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    records.append({
        "path": str(path.relative_to(root)),
        "sha256": digest,
        "byte_size": path.stat().st_size,
    })
if len(list((root / "v5-predictions").glob("*/seal.json"))) != 5:
    raise SystemExit("V5 prediction set is incomplete")
if len(list((root / "v6-predictions").glob("*/seal.json"))) != 5:
    raise SystemExit("V6 prediction set is incomplete")
payload = {
    "schema": "bayesian-phystwin/pokeflex-missing5-v5-v6-joint-pre-scoring-seal-v1",
    "github_sha": os.environ["GITHUB_SHA"],
    "future_target_mesh_decoded": False,
    "target_metric_computed": False,
    "records": records,
}
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
payload["joint_seal_sha256"] = hashlib.sha256(canonical).hexdigest()
(root / "barriers/joint-pre-scoring-seal.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({
    "joint_seal_sha256": payload["joint_seal_sha256"],
    "record_count": len(records),
    "future_target_mesh_decoded": False,
}, indent=2))
PY
write_status joint-v5-v6-pre-scoring-seal true

# Only now may the scorer decode future target meshes. Score V5 and V6 from the
# same five archives and immutable public-13 component.
"$SCIENCE_PYTHON" "$V5_SCRIPT" \
  --source-manifest "$SOURCE_MANIFEST" \
  score "$DATASET_ROOT" "$RUN_ROOT/v5-predictions" \
  "$RUN_ROOT/barriers/v5-barrier.json" "$RUN_ROOT/results/v5-result.json" \
  --public13-result "$PUBLIC13_RESULT" \
  >"$RUN_ROOT/logs/v5-score.log" 2>&1

"$SCIENCE_PYTHON" "$V6_SCRIPT" \
  --source-manifest "$SOURCE_MANIFEST" \
  score "$DATASET_ROOT" "$RUN_ROOT/v5-predictions" \
  "$RUN_ROOT/v6-predictions" "$RUN_ROOT/barriers/v6-barrier.json" \
  "$RUN_ROOT/results/v6-result.json" \
  --public13-result "$PUBLIC13_RESULT" \
  >"$RUN_ROOT/logs/v6-score.log" 2>&1

write_status complete true

"$SCIENCE_PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
v5 = json.loads((root / "results/v5-result.json").read_text())
v6 = json.loads((root / "results/v6-result.json").read_text())
status = json.loads((root / "status.json").read_text())
summary = {
    "schema": "bayesian-phystwin/pokeflex-missing5-v5-v6-execution-summary-v1",
    "status": "complete",
    "github_sha": os.environ["GITHUB_SHA"],
    "github_run_id": os.environ.get("GITHUB_RUN_ID"),
    "runner_name": os.environ.get("RUNNER_NAME"),
    "dataset_root": os.environ["DATASET_ROOT"],
    "source_manifest_sha256": json.loads(
        (root / "manifest/source-manifest.json").read_text()
    )["source_manifest_sha256"],
    "joint_pre_scoring_seal_sha256": json.loads(
        (root / "barriers/joint-pre-scoring-seal.json").read_text()
    )["joint_seal_sha256"],
    "v5_result_sha256": v5["result_sha256"],
    "v6_result_sha256": v6["result_sha256"],
    "v5_aggregate": v5["aggregate"],
    "v6_aggregate": v6["aggregate"],
    "target_payload_opened_only_after_joint_seal": True,
    "target_metric_computed": True,
    "target_adaptation_used": False,
    "replacement_used": False,
    "status_receipt": status,
}
canonical = json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
summary["summary_sha256"] = hashlib.sha256(canonical).hexdigest()
(root / "results/summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
report = [
    "# Frozen PokeFlex missing-five V5/V6 execution",
    "",
    f"- GitHub revision: `{summary['github_sha']}`",
    f"- Source manifest: `{summary['source_manifest_sha256']}`",
    f"- Joint pre-scoring seal: `{summary['joint_pre_scoring_seal_sha256']}`",
    "- Target adaptation: `false`",
    "- Replacement: `false`",
    "- Target meshes opened only after both five-prediction barriers: `true`",
    "",
    "## V5 aggregate",
    "",
    "```json",
    json.dumps(summary["v5_aggregate"], indent=2, sort_keys=True),
    "```",
    "",
    "## V6 aggregate",
    "",
    "```json",
    json.dumps(summary["v6_aggregate"], indent=2, sort_keys=True),
    "```",
]
(root / "results/report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
PY

(
  cd "$RUN_ROOT"
  find manifest barriers results logs -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum > ALL_FILES_SHA256SUMS
)
chmod -R u+rwX,go-rwx "$RUN_ROOT"

echo "Frozen PokeFlex missing-five V5/V6 execution completed at $RUN_ROOT"
