#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_TOKEN:?missing GITHUB_TOKEN}"
: "${GITHUB_REPOSITORY:?missing GITHUB_REPOSITORY}"
: "${GITHUB_RUN_ID:?missing GITHUB_RUN_ID}"
: "${GITHUB_RUN_ATTEMPT:?missing GITHUB_RUN_ATTEMPT}"
: "${GITHUB_SHA:?missing GITHUB_SHA}"
: "${RUNNER_NAME:?missing RUNNER_NAME}"
: "${TCN_ARTIFACT_ID:?missing TCN_ARTIFACT_ID}"
: "${TCN_ARTIFACT_SHA256:?missing TCN_ARTIFACT_SHA256}"
: "${TCN_EXECUTION_REVISION:?missing TCN_EXECUTION_REVISION}"
: "${TCN_CACHE_ROOT:?missing TCN_CACHE_ROOT}"
: "${QUERY_ARTIFACT_ID:?missing QUERY_ARTIFACT_ID}"
: "${QUERY_ARTIFACT_SHA256:?missing QUERY_ARTIFACT_SHA256}"
: "${QUERY_RESULT_SHA256:?missing QUERY_RESULT_SHA256}"
: "${DROPOUT_DRAWS:?missing DROPOUT_DRAWS}"
: "${FOLD_COUNT:?missing FOLD_COUNT}"
: "${MATURITY_LAG_WINDOWS:?missing MATURITY_LAG_WINDOWS}"
: "${BOOTSTRAP_REPETITIONS:?missing BOOTSTRAP_REPETITIONS}"
: "${REQUEST_PATH:?missing REQUEST_PATH}"

OUT="/home/github-runner/.cache/workflows/deform360-exact-tcn-query-uncertainty-v9/${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
VENV="${RUNNER_TEMP}/deform360-exact-tcn-query-uncertainty-v9-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
export OUT VENV

cleanup() {
  rm -rf -- "$VENV"
  if test -d "$OUT"; then
    chmod -R go-rwx "$OUT" || true
  fi
}
trap cleanup EXIT

select_exact_python() {
  local candidate version
  for candidate in python3.12 python3 /usr/bin/python3.12 /usr/bin/python3; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    version="$($candidate -c 'import platform; print(platform.python_version())')"
    if test "$version" = 3.12.3; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(select_exact_python)"
export PYTHON_BIN

test "$(git rev-parse HEAD)" = "$GITHUB_SHA"
test "$(git -C _exact_tcn_v6 rev-parse HEAD)" = "$TCN_EXECUTION_REVISION"
test "$(realpath -e "$TCN_CACHE_ROOT")" = "$TCN_CACHE_ROOT"
test -d "$TCN_CACHE_ROOT/raw-repository/raw"
test -d "$TCN_CACHE_ROOT/processed-repository/processed"
test ! -e "$OUT"
test ! -e "$VENV"
mkdir -p "$OUT/tcn-artifact" "$OUT/query-artifact"
chmod 700 "$OUT"
{
  echo "runner_name=${RUNNER_NAME}"
  echo "machine_name=$(hostname)"
  echo "python_bin=${PYTHON_BIN}"
  "$PYTHON_BIN" --version
  uname -a
  nvidia-smi -L
  realpath -e "$TCN_CACHE_ROOT"
} | tee "$OUT/compatibility.txt"

curl --fail --location --retry 4 --retry-all-errors \
  -H 'Accept: application/vnd.github+json' \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H 'X-GitHub-Api-Version: 2022-11-28' \
  "https://api.github.com/repos/${GITHUB_REPOSITORY}/actions/artifacts/${TCN_ARTIFACT_ID}/zip" \
  --output "$OUT/tcn-artifact.zip"
printf '%s  %s\n' "$TCN_ARTIFACT_SHA256" "$OUT/tcn-artifact.zip" \
  | sha256sum --check --strict
unzip -q "$OUT/tcn-artifact.zip" -d "$OUT/tcn-artifact"
mapfile -t tcn_results < <(find "$OUT/tcn-artifact" -type f -name result.json -print)
test "${#tcn_results[@]}" -eq 1
TCN_ARTIFACT_ROOT="$(dirname "${tcn_results[0]}")"
export TCN_ARTIFACT_ROOT
test -s "$TCN_ARTIFACT_ROOT/effective-protocol.json"
test -s "$TCN_ARTIFACT_ROOT/carrier-manifest.json"

curl --fail --location --retry 4 --retry-all-errors \
  -H 'Accept: application/vnd.github+json' \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H 'X-GitHub-Api-Version: 2022-11-28' \
  "https://api.github.com/repos/${GITHUB_REPOSITORY}/actions/artifacts/${QUERY_ARTIFACT_ID}/zip" \
  --output "$OUT/query-artifact.zip"
printf '%s  %s\n' "$QUERY_ARTIFACT_SHA256" "$OUT/query-artifact.zip" \
  | sha256sum --check --strict
unzip -q "$OUT/query-artifact.zip" -d "$OUT/query-artifact"
mapfile -t query_files < <(find "$OUT/query-artifact" -type f -name query-sufficient-statistics-v7.npz -print)
mapfile -t query_results < <(find "$OUT/query-artifact" -type f -name result.json -print)
test "${#query_files[@]}" -eq 1
test "${#query_results[@]}" -eq 1
QUERY_SOURCE_NPZ="${query_files[0]}"
export QUERY_SOURCE_NPZ
"$PYTHON_BIN" - "${query_results[0]}" <<'PY'
import json
import os
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
assert result["result_sha256"] == os.environ["QUERY_RESULT_SHA256"]
assert result["exact_reproduction"]["object_count"] == 92
assert result["same_mean_contract"]["predictive_mean_changed"] is False
PY

"$PYTHON_BIN" -m venv --without-pip "$VENV"
curl --fail --location --retry 4 --retry-all-errors \
  https://bootstrap.pypa.io/get-pip.py \
  --output "$OUT/get-pip.py"
"$VENV/bin/python" "$OUT/get-pip.py"
"$VENV/bin/python" -m pip install numpy==1.26.4 ruff==0.14.14
"$VENV/bin/python" -m pip install \
  'torch==2.8.0+cu128' \
  --index-url https://download.pytorch.org/whl/cu128

files=(
  scripts/science/deform360_exact_tcn_capture_v9.py
  scripts/science/deform360_tcn_query_uncertainty_v9.py
  scripts/science/run_deform360_exact_tcn_query_uncertainty_v9.py
)
"$VENV/bin/python" -m ruff check "${files[@]}"
"$VENV/bin/python" -m ruff format --check "${files[@]}"
"$VENV/bin/python" \
  scripts/science/run_deform360_exact_tcn_query_uncertainty_v9.py \
  --self-test
"$VENV/bin/python" - <<'PY' | tee "$OUT/runtime.txt"
import platform
import torch

assert platform.python_version() == "3.12.3"
assert torch.__version__ == "2.8.0+cu128"
assert torch.cuda.is_available()
value = torch.nn.Conv1d(8, 8, 3, padding=1).cuda()(
    torch.zeros((2, 8, 8), device="cuda")
)
assert torch.isfinite(value).all()
print(platform.python_version())
print(torch.__version__)
print(torch.cuda.get_device_name(0))
PY

"$VENV/bin/python" \
  scripts/science/run_deform360_exact_tcn_query_uncertainty_v9.py \
  --tcn-artifact-root "$TCN_ARTIFACT_ROOT" \
  --exact-tcn-root _exact_tcn_v6 \
  --data-root "$TCN_CACHE_ROOT" \
  --query-source-npz "$QUERY_SOURCE_NPZ" \
  --output-root "$OUT/study" \
  --dropout-draws "$DROPOUT_DRAWS" \
  --fold-count "$FOLD_COUNT" \
  --maturity-lag-windows "$MATURITY_LAG_WINDOWS" \
  --bootstrap-repetitions "$BOOTSTRAP_REPETITIONS" \
  2>&1 | tee "$OUT/console.log"

"$VENV/bin/python" - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ["OUT"]) / "study"
result = json.loads((root / "result.json").read_text())
manifest = json.loads((root / "capture-manifest.json").read_text())
assert result["status"] == "completed"
assert result["exact_tcn_reproduction"]["passed"] is True
assert result["exact_tcn_reproduction"]["maximum_absolute_numeric_difference"] <= 1e-10
assert math.isclose(
    result["point_prediction"]["active_field_rmse"],
    0.6163561621007425,
    rel_tol=0.0,
    abs_tol=1e-10,
)
assert result["point_prediction"]["predictive_mean_changed"] is False
assert result["uncertainty_study"]["object_count"] == 92
assert result["uncertainty_study"]["query_dimension"] == 12
assert result["uncertainty_study"]["gates"][
    "zero_future_or_current_outcome_uses"
] is True
assert result["information_boundary"][
    "online_update_uses_current_or_future_outcome"
] is False
assert result["paper_claim_authorized"] is False
assert result["globally_fresh_confirmation_authorized"] is False
assert manifest["exact_scientific_result_reproduction"] is True
assert manifest["carrier_audit"]["file_count"] == 6176
assert manifest["carrier_audit"]["all_sha256_verified"] is True
PY

cat "$OUT/study/report.md" >> "$GITHUB_STEP_SUMMARY"
sha256sum \
  "$OUT/study"/* \
  "$OUT/runtime.txt" \
  "$OUT/compatibility.txt" \
  > "$OUT/sha256sums.txt"
git diff --exit-code
test -z "$(git status --porcelain --untracked-files=no)"
git -C _exact_tcn_v6 diff --exit-code
test -z "$(git -C _exact_tcn_v6 status --porcelain --untracked-files=all)"
