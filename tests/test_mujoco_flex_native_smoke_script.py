from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "remote"
    / "run_mujoco_flex_native_smoke.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bpt_mujoco_flex_native_smoke",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_script_import_does_not_require_mujoco() -> None:
    module = _load_script()
    assert module.MUJOCO_VERSION == "3.9.0"
    assert module.MUJOCO_REVISION == "237c17e48539b6c90bf90d3161547cbdcbfaa1e0"
    assert module.MUJOCO_WHEEL_SHA256 == (
        "c148824d73487fe5ee29c371eff981645f372ccada1f20ea331288323e37c65e"
    )
    assert module._topology_descriptor() == {
        "flex_name": "soft",
        "dimension": 3,
        "grid_count": [5, 3, 3],
        "grid_spacing_m": [0.1, 0.1, 0.1],
        "vertex_count": 45,
        "tetrahedron_count": 96,
        "pinned_grid_range": [0, 0, 0, 0, 2, 2],
        "driven_face": "maximum-x-nine-vertices",
    }


def test_content_id_is_mapping_order_independent() -> None:
    module = _load_script()
    assert module._content_id({"b": 2, "a": 1}) == module._content_id({"a": 1, "b": 2})


def test_scene_is_volumetric_and_pins_the_left_face() -> None:
    module = _load_script()
    xml = module._scene_xml(
        integrator_time_step_s=1e-5,
        young_modulus_pa=1000.0,
        poisson_ratio=0.3,
        total_mass_kg=7.0,
    )
    assert 'dim="3"' in xml
    assert 'count="5 3 3"' in xml
    assert '<elasticity young="1000"' in xml
    assert '<pin gridrange="0 0 0 0 2 2"/>' in xml
    assert 'contype="0" conaffinity="0"' in xml


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frame_count": 1}, "frame_count"),
        ({"output_time_step_s": 0.0}, "output_time_step_s"),
        ({"integrator_time_step_s": 0.0}, "integrator_time_step_s"),
        ({"force_per_vertex_n": 0.0}, "force_per_vertex_n"),
        ({"young_modulus_pa": 0.0}, "young_modulus_pa"),
        ({"poisson_ratio": 0.5}, "poisson_ratio"),
        ({"total_mass_kg": 0.0}, "total_mass_kg"),
        (
            {"output_time_step_s": 0.025, "integrator_time_step_s": 0.00003},
            "exact integrator-step multiple",
        ),
    ],
)
def test_invalid_arguments_fail_before_native_import(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    module = _load_script()
    output = tmp_path / "result"
    with pytest.raises(ValueError, match=message):
        module.run_smoke(output, wheel_path=tmp_path / "missing.whl", **kwargs)
    assert not output.exists()
