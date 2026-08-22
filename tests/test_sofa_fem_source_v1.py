from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import bayesian_phystwin.sofa_fem_source_v1 as module
from bayesian_phystwin.sofa_fem_source_v1 import (
    ATTACHMENT_MODEL,
    CONSTITUTIVE_MODEL,
    SOFA_ARCHIVE_FILENAME,
    SOFA_ARCHIVE_SHA256,
    SOFA_REPORTED_VERSION,
    SOFA_REQUIRED_PLUGINS,
    SOFA_REVISION,
    SOFA_VERSION,
    load_native_sofa_fem_modules_v1,
    stable_neo_hookean_lame_parameters_v1,
)


def test_module_import_freezes_runtime_and_native_models() -> None:
    assert SOFA_VERSION == "26.06.00"
    assert SOFA_REVISION == "7c18e95d5c5f2839079892c69e7d89a313c79603"
    assert SOFA_ARCHIVE_SHA256 == (
        "129211fd01781bdd5ba3f28f1c3617a2f3792a71b62dc609cf866eec4ac745e2"
    )
    assert "Sofa.Component.Constraint.Projective" in SOFA_REQUIRED_PLUGINS
    assert "Sofa.Component.SolidMechanics.FEM.HyperElastic" in (SOFA_REQUIRED_PLUGINS)
    assert ATTACHMENT_MODEL == "AttachProjectiveConstraint-moving-Dirichlet-v1"
    assert "stable-Neo-Hookean" in CONSTITUTIVE_MODEL


def test_stable_neo_hookean_uses_physical_lame_parameters() -> None:
    shear, first_lame = stable_neo_hookean_lame_parameters_v1(1000.0, 0.3)
    assert shear == pytest.approx(384.6153846153846)
    assert first_lame == pytest.approx(576.9230769230769)


@pytest.mark.parametrize("poisson", [-0.1, 0.0, 0.5, 0.7])
def test_nonphysical_poisson_ratio_is_rejected(poisson: float) -> None:
    with pytest.raises(ValueError, match="poisson_ratio"):
        stable_neo_hookean_lame_parameters_v1(1000.0, poisson)


def test_native_loader_verifies_archive_environment_and_installed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / SOFA_ARCHIVE_FILENAME
    archive.write_bytes(b"pinned archive")
    root = tmp_path / "sofa"
    for relative in module.SOFA_INSTALLED_FILE_SHA256:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    (root / "git-info.txt").write_text(f"revision {SOFA_REVISION}\n", encoding="utf-8")
    sofa = SimpleNamespace(
        __file__=str(root / "plugins" / "SofaPython3" / "Sofa" / "__init__.py"),
        GetVersion=lambda: SOFA_REPORTED_VERSION,
    )
    imported_plugins: list[str] = []
    sofa_runtime = SimpleNamespace(
        __file__=str(root / "plugins" / "SofaPython3" / "SofaRuntime.py"),
        importPlugin=imported_plugins.append,
    )

    monkeypatch.setenv("SOFA_ROOT", str(root))
    monkeypatch.setenv("SOFA_PLUGIN_PATH", str(root / "plugins"))
    monkeypatch.setenv(
        "LD_LIBRARY_PATH",
        os.pathsep.join(
            (str(root / "lib"), str(root / "plugins" / "SofaPython3" / "lib"))
        ),
    )
    monkeypatch.setattr(module.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(
        module.platform, "python_version_tuple", lambda: ("3", "10", "0")
    )
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        {"Sofa": sofa, "SofaRuntime": sofa_runtime}.__getitem__,
    )

    def pinned_digest(path: Path) -> str:
        if path == archive:
            return SOFA_ARCHIVE_SHA256
        return module.SOFA_INSTALLED_FILE_SHA256[path.relative_to(root).as_posix()]

    monkeypatch.setattr(module, "_sha256_file", pinned_digest)
    native = load_native_sofa_fem_modules_v1(
        distribution_archive=archive,
        sofa_root=root,
    )

    assert native.sofa is sofa
    assert native.sofa_runtime is sofa_runtime
    assert native.root == root
    assert imported_plugins == list(SOFA_REQUIRED_PLUGINS)


def test_runtime_helpers_cover_hash_validation_and_scene_identity(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"payload")
    assert module._sha256_file(payload) == hashlib.sha256(b"payload").hexdigest()
    identity = module._scene_identity(
        points=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        cells=np.asarray([[0, 0, 0, 0]], dtype=np.int64),
        attachment_indices=np.asarray([0], dtype=np.int64),
        parameters={"young_modulus_pa": 1.0},
    )
    assert len(identity) == 64

    with pytest.raises(ValueError, match="young_modulus_pa"):
        stable_neo_hookean_lame_parameters_v1(True, 0.3)
    with pytest.raises(ValueError, match="young_modulus_pa"):
        stable_neo_hookean_lame_parameters_v1(0.0, 0.3)
    with pytest.raises(ValueError, match="poisson_ratio"):
        stable_neo_hookean_lame_parameters_v1(1.0, float("nan"))
