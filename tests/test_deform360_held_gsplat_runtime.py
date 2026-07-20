from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin import deform360_held_gsplat_runtime as runtime


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PREPARER = ROOT / "scripts" / "held" / "prepare_deform360_v7_gsplat_runtime.py"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_runtime_preparer() -> object:
    spec = importlib.util.spec_from_file_location(
        "deform360_v7_gsplat_runtime_preparer", RUNTIME_PREPARER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gsplat_contracts_have_fixed_canonical_identities() -> None:
    extension = runtime.GSPLAT_CUDA_EXTENSION_CONTRACT
    smoke = runtime.GSPLAT_RUNTIME_SMOKE_CONTRACT

    assert runtime.GSPLAT_CUDA_EXTENSION_CONTRACT_SHA256 == _canonical_sha256(extension)
    assert runtime.GSPLAT_RUNTIME_SMOKE_CONTRACT_SHA256 == _canonical_sha256(smoke)
    assert extension["canonical_path"] == os.fspath(runtime.GSPLAT_CUDA_EXTENSION_PATH)
    assert extension["sha256"] == (
        "2dd5e0c2a349619e1afc3dd041086eca900b387602bc76627b7f54264fffec64"
    )
    assert extension["file_size_bytes"] == 6_982_312
    assert extension["file_mode_octal"] == "0444"
    assert extension["parent_mode_octal"] == "0555"
    assert extension["jit_compilation_permitted"] is False
    assert smoke["logical_device"] == "cuda:0"
    assert smoke["formal_physical_gpu_indices"] == [0, 1]
    assert smoke["rasterization"]["packed"] is False
    assert runtime.GSPLAT_RUNTIME_SUPPLEMENT_MANIFEST_PATH.name == (
        "runtime-supplement-manifest.json"
    )
    assert runtime.GSPLAT_RUNTIME_SMOKE_EVIDENCE_PATH.name == (
        "gsplat-runtime-smoke-evidence.json"
    )


def test_exact_readonly_extension_is_validated_nofollow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        supplement = Path(directory) / "supplement"
        supplement.mkdir()
        extension = supplement / "gsplat_cuda.so"
        payload = b"fixed-aot-extension"
        extension.write_bytes(payload)
        extension.chmod(0o444)
        supplement.chmod(0o555)
        contract = {
            **runtime.GSPLAT_CUDA_EXTENSION_CONTRACT,
            "canonical_path": os.fspath(extension),
            "file_size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        monkeypatch.setattr(runtime, "GSPLAT_CUDA_EXTENSION_CONTRACT", contract)

        try:
            identity, digest = runtime._validate_extension_file(extension)
            assert identity[2] == len(payload)
            assert digest == contract["sha256"]
        finally:
            supplement.chmod(0o755)


def test_extension_validation_rejects_wrong_bytes_and_symlink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory)
        supplement = root / "supplement"
        supplement.mkdir()
        extension = supplement / "gsplat_cuda.so"
        extension.write_bytes(b"wrong")
        extension.chmod(0o444)
        supplement.chmod(0o555)
        contract = {
            **runtime.GSPLAT_CUDA_EXTENSION_CONTRACT,
            "canonical_path": os.fspath(extension),
            "file_size_bytes": 5,
            "sha256": "0" * 64,
        }
        monkeypatch.setattr(runtime, "GSPLAT_CUDA_EXTENSION_CONTRACT", contract)
        try:
            with pytest.raises(RuntimeError, match="checksum changed"):
                runtime._validate_extension_file(extension)
        finally:
            supplement.chmod(0o755)

        target = root / "target.so"
        target.write_bytes(b"fixed")
        linked = root / "linked.so"
        linked.symlink_to(target)
        linked_contract = {
            **contract,
            "canonical_path": os.fspath(linked),
            "file_size_bytes": 5,
            "sha256": hashlib.sha256(b"fixed").hexdigest(),
            "parent_mode_octal": format(root.stat().st_mode & 0o777, "04o"),
        }
        monkeypatch.setattr(runtime, "GSPLAT_CUDA_EXTENSION_CONTRACT", linked_contract)
        with pytest.raises(RuntimeError, match="aliased"):
            runtime._validate_extension_file(linked)


def test_exact_extension_loader_installs_and_retains_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extension_path = tmp_path / "gsplat_cuda.so"
    extension_path.write_bytes(b"placeholder")
    exported = {
        name: object()
        for name in runtime.GSPLAT_RUNTIME_SMOKE_CONTRACT["extension_required_exports"]
    }
    module = SimpleNamespace(__file__=os.fspath(extension_path), **exported)

    class Loader:
        def exec_module(self, observed: object) -> None:
            assert observed is module

    spec = SimpleNamespace(loader=Loader())
    monkeypatch.setattr(
        runtime.importlib.util,
        "spec_from_file_location",
        lambda name, path: spec,
    )
    monkeypatch.setattr(
        runtime.importlib.util, "module_from_spec", lambda observed: module
    )
    monkeypatch.setattr(runtime, "_LOADED_GSPLAT_CUDA", None)
    monkeypatch.delitem(runtime.sys.modules, "gsplat_cuda", raising=False)
    backend = SimpleNamespace(_C=None)

    loaded = runtime._load_exact_extension(extension_path, backend)

    assert loaded is module
    assert backend._C is module
    assert runtime._LOADED_GSPLAT_CUDA is module
    assert runtime.sys.modules["gsplat_cuda"] is module


def test_extension_loader_rejects_an_ambient_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "_LOADED_GSPLAT_CUDA", None)
    monkeypatch.delitem(runtime.sys.modules, "gsplat_cuda", raising=False)

    with pytest.raises(RuntimeError, match="ambient gsplat CUDA backend"):
        runtime._load_exact_extension(
            tmp_path / "gsplat_cuda.so", SimpleNamespace(_C=object())
        )


class _FakeTensor:
    def __init__(
        self,
        value: object,
        *,
        leaves: list[_FakeTensor] | None = None,
        zero_gradient: bool = False,
    ) -> None:
        self.value = np.asarray(value)
        self.leaves = list(leaves or [])
        self.zero_gradient = zero_gradient
        self.grad: _FakeTensor | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    def _combined(self, other: _FakeTensor) -> list[_FakeTensor]:
        return list(dict.fromkeys([*self.leaves, *other.leaves]))

    def __add__(self, other: object) -> _FakeTensor:
        if isinstance(other, _FakeTensor):
            return _FakeTensor(self.value + other.value, leaves=self._combined(other))
        return _FakeTensor(self.value + other, leaves=self.leaves)

    __radd__ = __add__

    def __mul__(self, other: object) -> _FakeTensor:
        if isinstance(other, _FakeTensor):
            return _FakeTensor(self.value * other.value, leaves=self._combined(other))
        return _FakeTensor(self.value * other, leaves=self.leaves)

    __rmul__ = __mul__

    def __gt__(self, other: object) -> _FakeTensor:
        return _FakeTensor(self.value > other)

    def unsqueeze(self, axis: int) -> _FakeTensor:
        return _FakeTensor(np.expand_dims(self.value, axis), leaves=self.leaves)

    def reshape(self, shape: tuple[int, ...]) -> _FakeTensor:
        return _FakeTensor(self.value.reshape(shape), leaves=self.leaves)

    def detach(self) -> _FakeTensor:
        return self

    def abs(self) -> _FakeTensor:
        return _FakeTensor(np.abs(self.value), leaves=self.leaves)

    def sum(self) -> _FakeTensor:
        return _FakeTensor(self.value.sum(), leaves=self.leaves)

    def all(self) -> _FakeTensor:
        return _FakeTensor(self.value.all())

    def item(self) -> Any:
        return self.value.item()

    def numel(self) -> int:
        return int(self.value.size)

    def backward(self) -> None:
        for leaf in self.leaves:
            gradient = (
                np.zeros_like(leaf.value)
                if leaf.zero_gradient
                else np.ones_like(leaf.value)
            )
            leaf.grad = _FakeTensor(gradient)


class _MockTorchFacade:
    def __init__(self) -> None:
        self.float32 = np.float32
        self.cuda = SimpleNamespace(synchronize=lambda device: None)

    def tensor(self, value: object, **kwargs: object) -> _FakeTensor:
        requires_grad = bool(kwargs.get("requires_grad", False))
        tensor = _FakeTensor(
            np.asarray(value, dtype=np.float32),
            zero_gradient=requires_grad
            and len(np.asarray(value).shape) == 2
            and np.asarray(value).shape == (2, 3)
            and np.asarray(value)[0, 0] == pytest.approx(0.82),
        )
        if requires_grad:
            tensor.leaves = [tensor]
        return tensor

    def eye(self, size: int, **kwargs: object) -> _FakeTensor:
        return _FakeTensor(np.eye(size, dtype=np.float32))

    def linspace(
        self, start: float, stop: float, count: int, **kwargs: object
    ) -> _FakeTensor:
        return _FakeTensor(np.linspace(start, stop, count, dtype=np.float32))

    def isfinite(self, value: _FakeTensor) -> _FakeTensor:
        return _FakeTensor(np.isfinite(value.value))


def test_fixed_smoke_rejects_a_zero_parameter_gradient_without_cuda() -> None:
    facade = _MockTorchFacade()

    def rasterization(
        **inputs: _FakeTensor,
    ) -> tuple[_FakeTensor, _FakeTensor, dict[str, _FakeTensor]]:
        leaves = [
            inputs[name] for name in ("means", "quats", "scales", "opacities", "colors")
        ]
        render = _FakeTensor(np.ones((1, 16, 16, 3)), leaves=leaves)
        alpha = _FakeTensor(np.full((1, 16, 16, 1), 0.5), leaves=leaves)
        return render, alpha, {"radii": _FakeTensor([[2, 3]])}

    with pytest.raises(RuntimeError, match="zero colors gradient"):
        runtime._run_fixed_rasterization_smoke(facade, rasterization)


def test_smoke_artifact_signature_covers_all_semantic_predicates() -> None:
    unsigned = {
        "artifact_kind": "Deform360HeldGsplatRuntimeSmokeV1",
        "schema_version": 1,
        "predicates": {
            "backward_complete": True,
            "cuda_synchronized": True,
        },
    }
    signed = runtime._artifact(unsigned)

    assert signed["artifact_sha256"] == _canonical_sha256(unsigned)
    changed = dict(signed)
    changed["predicates"] = {**signed["predicates"], "backward_complete": False}
    assert changed["artifact_sha256"] != _canonical_sha256(
        {key: value for key, value in changed.items() if key != "artifact_sha256"}
    )


def _device_smoke(preparer: object, physical_gpu_index: int) -> dict[str, object]:
    return preparer._artifact(
        {
            "artifact_kind": "Deform360HeldGsplatRuntimeSmokeV1",
            "compute_capability": "8.9",
            "contract_sha256": runtime.GSPLAT_RUNTIME_SMOKE_CONTRACT_SHA256,
            "extension_loaded_and_retained": True,
            "extension_path": os.fspath(runtime.GSPLAT_CUDA_EXTENSION_PATH),
            "extension_sha256": runtime.GSPLAT_CUDA_EXTENSION_CONTRACT["sha256"],
            "gpu_name": "NVIDIA RTX 6000 Ada Generation",
            "gsplat_version": "1.4.0",
            "logical_device": "cuda:0",
            "ninja_visible": False,
            "nvcc_visible": False,
            "physical_gpu_index": physical_gpu_index,
            "predicates": dict(preparer._EXPECTED_PREDICATES),
            "python_version": "3.12",
            "schema_version": 1,
            "target_or_outcome_path_accessed": False,
            "torch_cuda_version": "12.1",
            "torch_version": "2.4.0+cu121",
        }
    )


def test_runtime_preparer_builds_exact_signed_manifest_and_two_gpu_evidence() -> None:
    preparer = _load_runtime_preparer()
    extension_contract = dict(runtime.GSPLAT_CUDA_EXTENSION_CONTRACT)
    manifest = preparer._expected_manifest(
        extension_contract, runtime.GSPLAT_CUDA_EXTENSION_CONTRACT_SHA256
    )
    manifest_payload = preparer._canonical_payload(manifest)
    smokes = [_device_smoke(preparer, index) for index in (0, 1)]

    for index, smoke in enumerate(smokes):
        preparer._validate_smoke(
            smoke,
            physical_gpu_index=index,
            smoke_contract_sha256=runtime.GSPLAT_RUNTIME_SMOKE_CONTRACT_SHA256,
        )
    evidence = preparer._expected_evidence(
        smoke_contract_sha256=runtime.GSPLAT_RUNTIME_SMOKE_CONTRACT_SHA256,
        method_head="a" * 40,
        source_sha256="b" * 64,
        manifest_payload=manifest_payload,
        smokes=smokes,
    )

    assert manifest["artifact_kind"] == (
        "Deform360HeldGsplatRuntimeSupplementManifestV1"
    )
    assert manifest["artifact_sha256"] == preparer._canonical_sha256(
        {key: value for key, value in manifest.items() if key != "artifact_sha256"}
    )
    assert [smoke["physical_gpu_index"] for smoke in evidence["smokes"]] == [0, 1]
    assert evidence["operator_source_sha256"] == "b" * 64
    assert (
        evidence["runtime_supplement_manifest_sha256"]
        == hashlib.sha256(manifest_payload).hexdigest()
    )
    assert evidence["artifact_sha256"] == preparer._canonical_sha256(
        {key: value for key, value in evidence.items() if key != "artifact_sha256"}
    )
    preparer_source = RUNTIME_PREPARER.read_text(encoding="utf-8")
    assert 'f"pycache_prefix={_PYCACHE_PREFIX}"' in preparer_source
    assert '"PYNPUT_BACKEND": "dummy"' in preparer_source


def test_runtime_preparer_rejects_rechecksummed_false_smoke_predicate() -> None:
    preparer = _load_runtime_preparer()
    smoke = _device_smoke(preparer, 0)
    smoke["predicates"] = {
        **smoke["predicates"],
        "cuda_synchronized": False,
    }
    smoke["artifact_sha256"] = preparer._canonical_sha256(
        {key: value for key, value in smoke.items() if key != "artifact_sha256"}
    )

    with pytest.raises(ValueError, match="identity or predicates changed"):
        preparer._validate_smoke(
            smoke,
            physical_gpu_index=0,
            smoke_contract_sha256=runtime.GSPLAT_RUNTIME_SMOKE_CONTRACT_SHA256,
        )
