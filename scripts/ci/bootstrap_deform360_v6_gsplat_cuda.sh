#!/usr/bin/env bash
# Source this file from the Deform360 v6 GPU runtime bootstrap.
# It installs only the checksum-pinned CUDA 12.1 compiler/runtime headers
# required to compile gsplat 1.4.0 against the inherited PyTorch cu121 build,
# and binds nvcc to the registered GNU 11 host compiler available on the sole
# authorized runner.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "bootstrap_deform360_v6_gsplat_cuda.sh must be sourced" >&2
  exit 2
fi

_bpt_require_env() {
  local name=$1
  if [[ -z "${!name:-}" ]]; then
    echo "required environment variable is unset: ${name}" >&2
    return 1
  fi
}

_bpt_require_env RUNNER_TEMP
_bpt_require_env GITHUB_RUN_ID
_bpt_require_env GITHUB_RUN_ATTEMPT
_bpt_require_env GITHUB_ENV
_bpt_require_env GITHUB_PATH
_bpt_require_env GSPLAT_CUDA_RUNTIME_REPAIR_ID
_bpt_require_env GSPLAT_CUDA_RUNTIME_REPAIR_PATH
_bpt_require_env GSPLAT_CUDA_RUNTIME_REPAIR_SHA256

runtime=${1:?runtime directory argument is required}
if [[ -L "${runtime}" || ! -d "${runtime}" ]]; then
  echo "runtime directory is missing, not a directory, or symlinked" >&2
  return 1
fi
runtime_python="${runtime}/bin/python"
if [[ -L "${runtime_python}" || ! -x "${runtime_python}" ]]; then
  echo "runtime Python is missing, not executable, or symlinked" >&2
  return 1
fi

observed_repair_sha=$(
  sha256sum "${GSPLAT_CUDA_RUNTIME_REPAIR_PATH}" | awk '{print $1}'
)
if [[ "${observed_repair_sha}" != "${GSPLAT_CUDA_RUNTIME_REPAIR_SHA256}" ]]; then
  echo "gsplat CUDA runtime repair bytes changed" >&2
  return 1
fi

"${runtime_python}" - \
  "${GSPLAT_CUDA_RUNTIME_REPAIR_PATH}" \
  "${GSPLAT_CUDA_RUNTIME_REPAIR_ID}" <<'PY'
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
    raise SystemExit("gsplat CUDA runtime repair identity changed")
if any(payload["information_boundary"].values()):
    raise SystemExit("gsplat CUDA runtime repair opens a forbidden boundary")
if payload["correction"]["torch_cuda_version"] != "12.1":
    raise SystemExit("gsplat CUDA runtime repair no longer targets cu121")
PY

host_compiler_repair_path="protocols/amendments/deform360_official_hub_fresh_object_session_v6_cuda_host_compiler_gnu11.json"
host_compiler_repair_id="01a5b25972e5b254bfd0ed40fadfd3417532519869d70f404acedf64b98147e0"
host_compiler_repair_sha256="4771a44c9c38158e54659ec2c420fe33e2c22f725adf977d891c7b9b978109e5"
predecessor_host_compiler_repair_id="e935a990cd380b10f225617d4b439ff609593d63a93e44c27e8fcba5e1dec721"

if [[ -L "${host_compiler_repair_path}" || \
      ! -f "${host_compiler_repair_path}" ]]; then
  echo "CUDA host-compiler repair is missing, not a file, or symlinked" >&2
  return 1
fi
observed_host_compiler_repair_sha=$(
  sha256sum "${host_compiler_repair_path}" | awk '{print $1}'
)
if [[ "${observed_host_compiler_repair_sha}" != \
      "${host_compiler_repair_sha256}" ]]; then
  echo "CUDA host-compiler repair bytes changed" >&2
  return 1
fi

"${runtime_python}" - \
  "${host_compiler_repair_path}" \
  "${host_compiler_repair_id}" \
  "${predecessor_host_compiler_repair_id}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_id = sys.argv[2]
expected_predecessor = sys.argv[3]
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
    raise SystemExit("CUDA host-compiler repair identity changed")
if payload["predecessor_host_compiler_repair_id"] != expected_predecessor:
    raise SystemExit("CUDA host-compiler repair predecessor changed")
if any(payload["information_boundary"].values()):
    raise SystemExit("CUDA host-compiler repair opens a forbidden boundary")
correction = payload["correction"]
expected_correction = {
    "compiler_family": "GNU",
    "compiler_major_version": 11,
    "compiler_package_versions": {
        "cc": "gcc-11 11.5.0-1ubuntu1~24.04.1",
        "cxx": "g++-11 11.5.0-1ubuntu1~24.04.1",
    },
    "compiler_paths": {
        "cc": "/usr/bin/gcc-11",
        "cxx": "/usr/bin/g++-11",
    },
    "compiler_versions": {"cc": "11.5.0", "cxx": "11.5.0"},
    "cuda_host_compiler_environment": [
        "CC",
        "CXX",
        "CUDAHOSTCXX",
        "NVCC_CCBIN",
    ],
    "nvcc_compiler_bindir_probe_required": True,
    "resolved_compiler_paths": {
        "cc": "/usr/bin/x86_64-linux-gnu-gcc-11",
        "cxx": "/usr/bin/x86_64-linux-gnu-g++-11",
    },
    "resolved_compiler_sha256": {
        "cc": "920b82bda223384ee558b43dd2a6e4c465b40ba268380f12ea59df45eeb7609d",
        "cxx": "02ba98cc5feefe173cfb8c28c98089817737800537dc7189138ed66b07cf56ec",
    },
    "source_independent_probe": {
        "cuda_toolkit_release": "12.1.1",
        "host": "workstation2",
        "probe": (
            "compile a trivial CUDA translation unit with nvcc "
            "--compiler-bindir /usr/bin/g++-11"
        ),
        "result": "passed",
    },
    "unsupported_compiler_override_allowed": False,
}
if correction != expected_correction:
    raise SystemExit("CUDA host-compiler correction changed")
if correction["unsupported_compiler_override_allowed"] is not False:
    raise SystemExit("unsupported CUDA host-compiler override was authorized")
PY

host_cc=/usr/bin/gcc-11
host_cxx=/usr/bin/g++-11
for compiler in "${host_cc}" "${host_cxx}"; do
  if [[ ! -x "${compiler}" ]]; then
    echo "registered CUDA host compiler is unavailable: ${compiler}" >&2
    return 1
  fi
  resolved_compiler=$(readlink -f -- "${compiler}")
  if [[ -z "${resolved_compiler}" || ! -x "${resolved_compiler}" ]]; then
    echo "registered CUDA host compiler does not resolve safely: ${compiler}" >&2
    return 1
  fi
done
host_cc_resolved=$(readlink -f -- "${host_cc}")
host_cxx_resolved=$(readlink -f -- "${host_cxx}")
host_cc_version=$("${host_cc}" -dumpfullversion -dumpversion)
host_cxx_version=$("${host_cxx}" -dumpfullversion -dumpversion)
host_cc_package=$(dpkg-query -W -f='${Package} ${Version}' gcc-11)
host_cxx_package=$(dpkg-query -W -f='${Package} ${Version}' g++-11)
if [[ "${host_cc_version}" != "11.5.0" || \
      "${host_cxx_version}" != "11.5.0" ]]; then
  echo "registered CUDA host compilers must report GNU version 11.5.0" >&2
  return 1
fi
if [[ "${host_cc_version}" != "${host_cxx_version}" ]]; then
  echo "registered CUDA C and C++ compiler versions differ" >&2
  return 1
fi
if [[ "${host_cc_package}" != "gcc-11 11.5.0-1ubuntu1~24.04.1" || \
      "${host_cxx_package}" != "g++-11 11.5.0-1ubuntu1~24.04.1" ]]; then
  echo "registered CUDA host compiler package identity changed" >&2
  return 1
fi
if [[ "${host_cc_resolved}" != "/usr/bin/x86_64-linux-gnu-gcc-11" || \
      "${host_cxx_resolved}" != "/usr/bin/x86_64-linux-gnu-g++-11" ]]; then
  echo "registered CUDA host compiler resolution changed" >&2
  return 1
fi
host_cc_sha256=$(sha256sum "${host_cc_resolved}" | awk '{print $1}')
host_cxx_sha256=$(sha256sum "${host_cxx_resolved}" | awk '{print $1}')
if [[ "${host_cc_sha256}" != \
      "920b82bda223384ee558b43dd2a6e4c465b40ba268380f12ea59df45eeb7609d" || \
      "${host_cxx_sha256}" != \
      "02ba98cc5feefe173cfb8c28c98089817737800537dc7189138ed66b07cf56ec" ]]; then
  echo "registered CUDA host compiler byte identity changed" >&2
  return 1
fi

export CC="${host_cc}"
export CXX="${host_cxx}"
export CUDAHOSTCXX="${host_cxx}"
export NVCC_CCBIN="${host_cxx}"
printf 'CUDA host compiler repair ID: %s\n' "${host_compiler_repair_id}"
printf 'CUDA host C compiler: %s (%s)\n' "${CC}" "${host_cc_version}"
printf 'CUDA host C++ compiler: %s (%s)\n' "${CXX}" "${host_cxx_version}"

host_probe_source="${runtime}/cuda-host-compiler-probe.cpp"
host_probe_object="${runtime}/cuda-host-compiler-probe.o"
cat > "${host_probe_source}" <<'CPP'
#include <cstdint>

static_assert(sizeof(std::int64_t) == 8);
CPP
"${CXX}" \
  --std=c++17 \
  --compile \
  "${host_probe_source}" \
  -o "${host_probe_object}"
test -s "${host_probe_object}"
rm -- "${host_probe_source}" "${host_probe_object}"

cuda_root="${RUNNER_TEMP}/deform360-v6-cuda-12.1.1-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
if [[ -L "${cuda_root}" || -e "${cuda_root}" ]]; then
  echo "refusing pre-existing CUDA toolkit root" >&2
  return 1
fi
mkdir -p "${cuda_root}/downloads"
if [[ -L "${cuda_root}" || ! -d "${cuda_root}" ]]; then
  echo "CUDA toolkit root failed real-directory validation" >&2
  return 1
fi

_bpt_download_cuda_component() {
  local component=$1
  local relative_path=$2
  local expected_sha=$3
  local archive="${cuda_root}/downloads/${component}.tar.xz"
  local url="https://developer.download.nvidia.com/compute/cuda/redist/${relative_path}"

  if [[ -L "${archive}" || -e "${archive}" ]]; then
    echo "refusing pre-existing CUDA component archive: ${archive}" >&2
    return 1
  fi

  "${runtime_python}" - "${url}" "${archive}" <<'PY'
from __future__ import annotations

import shutil
import sys
import time
import urllib.request
from pathlib import Path

url = sys.argv[1]
destination = Path(sys.argv[2])
last_error: BaseException | None = None
for attempt in range(1, 4):
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "BayesianPhysTwin-Deform360-v6-runtime/1"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 200:
                raise RuntimeError(f"unexpected download status {response.status}")
            with destination.open("xb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
        break
    except BaseException as error:
        last_error = error
        destination.unlink(missing_ok=True)
        if attempt == 3:
            raise
        time.sleep(2**attempt)
if not destination.is_file() or destination.is_symlink():
    raise SystemExit(f"CUDA archive was not materialized safely: {last_error}")
PY

  printf '%s  %s\n' "${expected_sha}" "${archive}" | sha256sum --check -
  tar \
    --extract \
    --xz \
    --file "${archive}" \
    --directory "${cuda_root}" \
    --strip-components=1 \
    --no-same-owner \
    --no-same-permissions
  rm -- "${archive}"
}

_bpt_download_cuda_component \
  cuda_cccl \
  cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.1.109-archive.tar.xz \
  b84ef3ec3dc1b4891267be25846f0c3ed7f9fa84154d59eba805402b86991baa
_bpt_download_cuda_component \
  cuda_cudart \
  cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.1.105-archive.tar.xz \
  6096ec878c8c443258d39c6e9cf2decef127f8aa8da594fdc5a336d047ab6bd9
_bpt_download_cuda_component \
  cuda_nvcc \
  cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.1.105-archive.tar.xz \
  0b85f7eee17788abbd170b0b493c74ce2e9fd5a9604461b99c2c378165e1083b

rmdir "${cuda_root}/downloads"

if [[ ! -x "${cuda_root}/bin/nvcc" ]]; then
  echo "checksum-pinned CUDA archive did not provide bin/nvcc" >&2
  return 1
fi

if [[ ! -e "${cuda_root}/include" ]]; then
  target_include="${cuda_root}/targets/x86_64-linux/include"
  if [[ ! -d "${target_include}" || -L "${target_include}" ]]; then
    echo "checksum-pinned CUDA archive did not provide headers" >&2
    return 1
  fi
  ln -s targets/x86_64-linux/include "${cuda_root}/include"
fi
if [[ ! -e "${cuda_root}/lib64" ]]; then
  if [[ -d "${cuda_root}/lib" && ! -L "${cuda_root}/lib" ]]; then
    ln -s lib "${cuda_root}/lib64"
  elif [[ -d "${cuda_root}/targets/x86_64-linux/lib" && \
          ! -L "${cuda_root}/targets/x86_64-linux/lib" ]]; then
    ln -s targets/x86_64-linux/lib "${cuda_root}/lib64"
  else
    echo "checksum-pinned CUDA archive did not provide runtime libraries" >&2
    return 1
  fi
fi

cat > "${cuda_root}/version.json" <<'JSON'
{
  "cuda": {
    "name": "CUDA SDK",
    "version": "12.1.1"
  },
  "cuda_cccl": {
    "version": "12.1.109"
  },
  "cuda_cudart": {
    "version": "12.1.105"
  },
  "cuda_nvcc": {
    "version": "12.1.105"
  }
}
JSON

export CUDA_HOME="${cuda_root}"
export CUDA_PATH="${cuda_root}"
export PATH="${cuda_root}/bin:${PATH}"
export LD_LIBRARY_PATH="${cuda_root}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TORCH_EXTENSIONS_DIR="${runtime}/torch-extensions"
mkdir -p "${TORCH_EXTENSIONS_DIR}"
if [[ -L "${TORCH_EXTENSIONS_DIR}" || ! -d "${TORCH_EXTENSIONS_DIR}" ]]; then
  echo "Torch extension directory failed real-directory validation" >&2
  return 1
fi

"${CUDA_HOME}/bin/nvcc" --version
if ! "${CUDA_HOME}/bin/nvcc" --version | grep -Fq "release 12.1"; then
  echo "CUDA compiler version does not match the inherited cu121 runtime" >&2
  return 1
fi

probe_source="${runtime}/cuda-runtime-probe.cu"
probe_object="${runtime}/cuda-runtime-probe.o"
cat > "${probe_source}" <<'CU'
#include <cuda_runtime.h>

__global__ void bpt_probe_kernel() {}

int main() {
  bpt_probe_kernel<<<1, 1>>>();
  return 0;
}
CU
"${CUDA_HOME}/bin/nvcc" \
  --std=c++17 \
  --compiler-bindir "${CUDAHOSTCXX}" \
  --compile \
  "${probe_source}" \
  --output-file "${probe_object}"
test -s "${probe_object}"
rm -- "${probe_source}" "${probe_object}"

{
  printf 'CC=%s\n' "${CC}"
  printf 'CXX=%s\n' "${CXX}"
  printf 'CUDAHOSTCXX=%s\n' "${CUDAHOSTCXX}"
  printf 'NVCC_CCBIN=%s\n' "${NVCC_CCBIN}"
  printf 'CUDA_HOST_COMPILER_REPAIR_ID=%s\n' "${host_compiler_repair_id}"
  printf 'CUDA_HOST_COMPILER_REPAIR_PATH=%s\n' "${host_compiler_repair_path}"
  printf 'CUDA_HOST_COMPILER_REPAIR_SHA256=%s\n' \
    "${host_compiler_repair_sha256}"
  printf 'CUDA_HOST_COMPILER_VERSION=%s\n' "${host_cxx_version}"
  printf 'CUDA_HOST_CC_PACKAGE=%s\n' "${host_cc_package}"
  printf 'CUDA_HOST_CXX_PACKAGE=%s\n' "${host_cxx_package}"
  printf 'CUDA_HOST_CC_RESOLVED=%s\n' "${host_cc_resolved}"
  printf 'CUDA_HOST_CXX_RESOLVED=%s\n' "${host_cxx_resolved}"
  printf 'CUDA_HOST_CC_SHA256=%s\n' "${host_cc_sha256}"
  printf 'CUDA_HOST_CXX_SHA256=%s\n' "${host_cxx_sha256}"
  printf 'CUDA_HOST_COMPILER_PROBE_PASSED=true\n'
  printf 'CUDA_HOME=%s\n' "${CUDA_HOME}"
  printf 'CUDA_PATH=%s\n' "${CUDA_PATH}"
  printf 'LD_LIBRARY_PATH=%s\n' "${LD_LIBRARY_PATH}"
  printf 'TORCH_EXTENSIONS_DIR=%s\n' "${TORCH_EXTENSIONS_DIR}"
} >> "${GITHUB_ENV}"
printf '%s\n' "${CUDA_HOME}/bin" >> "${GITHUB_PATH}"

unset -f _bpt_download_cuda_component
unset -f _bpt_require_env
