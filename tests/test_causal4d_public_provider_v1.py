from __future__ import annotations

import bayesian_phystwin.causal4d_public_provider_v1 as public_provider


def test_public_provider_manifest_and_lazy_exports() -> None:
    manifest = public_provider.causal4d_public_provider_manifest(
        provider_revision="abc123"
    )
    assert manifest["provider_revision"] == "abc123"
    assert manifest["schema_version"] == 1
    assert set(manifest["capabilities"]) == {
        "deform360_selective_virtual_sensing",
        "phystwin_track_objective",
    }
    assert manifest["metadata"] == {
        "provider_api": "bayesian_phystwin.causal4d_public_provider_v1",
        "provider_api_version": 1,
    }
    assert public_provider.PROTOCOL_ID == "deform360-selective-virtual-sensing-v1"
    assert public_provider.MANIFEST_FILENAME == "measurement_manifest.json"
    assert callable(public_provider.build_phystwin_track_objective)


def test_public_provider_lazy_registry_is_closed() -> None:
    assert "dynamic_window_source_case" in dir(public_provider)
    try:
        public_provider.__getattr__("not_a_public_provider_operation")
    except AttributeError as error:
        assert "no attribute" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("unknown provider operation was accepted")
