from __future__ import annotations

import pytest

from bayesian_phystwin.sofa_fem_source_v1 import (
    ATTACHMENT_MODEL,
    CONSTITUTIVE_MODEL,
    SOFA_ARCHIVE_SHA256,
    SOFA_REQUIRED_PLUGINS,
    SOFA_REVISION,
    SOFA_VERSION,
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
