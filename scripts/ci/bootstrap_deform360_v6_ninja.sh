#!/usr/bin/env bash
# Source after creating the isolated Deform360 v6 runtime and installing pip.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "bootstrap_deform360_v6_ninja.sh must be sourced" >&2
  exit 2
fi

_bpt_ninja_require_env() {
  local name=$1
  if [[ -z "${!name:-}" ]]; then
    echo "required Ninja environment variable is unset: ${name}" >&2
    return 1
  fi
}

_bpt_ninja_require_env GITHUB_ENV
_bpt_ninja_require_env GITHUB_PATH
_bpt_ninja_require_env NINJA_BUILD_TOOL_REPAIR_ID
_bpt_ninja_require_env NINJA_BUILD_TOOL_REPAIR_PATH
_bpt_ninja_require_env NINJA_BUILD_TOOL_REPAIR_SHA256

runtime=${1:?runtime directory argument is required}
if [[ -L "${runtime}" || ! -d "${runtime}" ]]; then
  echo "Ninja runtime directory is missing, not a directory, or symlinked" >&2
  return 1
fi
runtime_python="${runtime}/bin/python"
if [[ -L "${runtime_python}" || ! -x "${runtime_python}" ]]; then
  echo "Ninja runtime Python is missing, not executable, or symlinked" >&2
  return 1
fi

if [[ -L "${NINJA_BUILD_TOOL_REPAIR_PATH}" || \
      ! -f "${NINJA_BUILD_TOOL_REPAIR_PATH}" ]]; then
  echo "Ninja build-tool repair is missing, not a file, or symlinked" >&2
  return 1
fi
observed_repair_sha=$(
  sha256sum "${NINJA_BUILD_TOOL_REPAIR_PATH}" | awk '{print $1}'
)
if [[ "${observed_repair_sha}" != "${NINJA_BUILD_TOOL_REPAIR_SHA256}" ]]; then
  echo "Ninja build-tool repair bytes changed" >&2
  return 1
fi

"${runtime_python}" - \
  "${NINJA_BUILD_TOOL_REPAIR_PATH}" \
  "${NINJA_BUILD_TOOL_REPAIR_ID}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_id = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
declared_id = payload.pop("repair_id")
canonical = json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
observed_id = hashlib.sha256(canonical).hexdigest()
if declared_id != expected_id or observed_id != expected_id:
    raise SystemExit("Ninja build-tool repair identity changed")
if any(payload["information_boundary"].values()):
    raise SystemExit("Ninja build-tool repair opens a forbidden boundary")
correction = payload["correction"]
expected = {
    "distribution": "ninja",
    "executable_relative_to_runtime": "bin/ninja",
    "executable_sha256": (
        "696f9628a79d9ce50314cf9556d7cd1a1d1ec52b8fd52828f6f9db1719565b67"
    ),
    "executable_version_output": "1.13.0.git.kitware.jobserver-pipe-1",
    "install_mode": "ignore-installed, no-deps, wheel-only, require-hashes",
    "pytorch_probe": "torch.utils.cpp_extension.is_ninja_available",
    "runtime_bin_prepended_to_path": True,
    "version": "1.13.0",
    "wheel_filename": (
        "ninja-1.13.0-py3-none-manylinux2014_x86_64."
        "manylinux_2_17_x86_64.whl"
    ),
    "wheel_sha256": (
        "fb46acf6b93b8dd0322adc3a4945452a4e774b75b91293bafcc7b7f8e6517dfa"
    ),
}
if correction != expected:
    raise SystemExit("Ninja build-tool correction changed")
PY

requirements="${runtime}/ninja-requirements.txt"
cat > "${requirements}" <<'REQ'
ninja==1.13.0 --hash=sha256:fb46acf6b93b8dd0322adc3a4945452a4e774b75b91293bafcc7b7f8e6517dfa
REQ
"${runtime_python}" -m pip install \
  --ignore-installed \
  --no-deps \
  --only-binary=:all: \
  --require-hashes \
  --requirement "${requirements}"
rm -- "${requirements}"

export PATH="${runtime}/bin:${PATH}"
ninja_path=$(command -v ninja)
if [[ "${ninja_path}" != "${runtime}/bin/ninja" || \
      -L "${ninja_path}" || ! -x "${ninja_path}" ]]; then
  echo "installed Ninja executable identity changed" >&2
  return 1
fi
ninja_sha256=$(sha256sum "${ninja_path}" | awk '{print $1}')
if [[ "${ninja_sha256}" != \
      "696f9628a79d9ce50314cf9556d7cd1a1d1ec52b8fd52828f6f9db1719565b67" ]]; then
  echo "installed Ninja executable bytes changed" >&2
  return 1
fi
ninja_version=$(ninja --version)
if [[ "${ninja_version}" != "1.13.0.git.kitware.jobserver-pipe-1" ]]; then
  echo "installed Ninja executable version changed" >&2
  return 1
fi

"${runtime_python}" - <<'PY'
from importlib.metadata import version

from torch.utils.cpp_extension import is_ninja_available

if version("ninja") != "1.13.0":
    raise SystemExit("Ninja distribution identity changed")
if not is_ninja_available():
    raise SystemExit("PyTorch cannot discover the registered Ninja executable")
PY

{
  printf 'NINJA_BUILD_TOOL_REPAIR_ID=%s\n' "${NINJA_BUILD_TOOL_REPAIR_ID}"
  printf 'NINJA_BUILD_TOOL_REPAIR_PATH=%s\n' "${NINJA_BUILD_TOOL_REPAIR_PATH}"
  printf 'NINJA_BUILD_TOOL_REPAIR_SHA256=%s\n' \
    "${NINJA_BUILD_TOOL_REPAIR_SHA256}"
  printf 'NINJA_DISTRIBUTION_VERSION=1.13.0\n'
  printf 'NINJA_EXECUTABLE_PATH=%s\n' "${ninja_path}"
  printf 'NINJA_EXECUTABLE_SHA256=%s\n' "${ninja_sha256}"
  printf 'NINJA_EXECUTABLE_VERSION=%s\n' "${ninja_version}"
  printf 'NINJA_PYTORCH_PROBE_PASSED=true\n'
} >> "${GITHUB_ENV}"
printf '%s\n' "${runtime}/bin" >> "${GITHUB_PATH}"

unset -f _bpt_ninja_require_env
