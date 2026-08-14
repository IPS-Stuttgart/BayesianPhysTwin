from __future__ import annotations

from dataclasses import replace

import pytest

from bayesian_phystwin.physics_backend_registry_v1 import (
    BUILTIN_BACKEND_PROFILES,
    PHYSICS_BACKEND_ENTRY_POINT_GROUP,
    PhysicsBackendProfileV1,
    discover_backend_profiles,
    get_backend_profile,
    profile_from_mapping,
)


class _EntryPoint:
    def __init__(self, name: str, value: object) -> None:
        self.name = name
        self.group = PHYSICS_BACKEND_ENTRY_POINT_GROUP
        self._value = value

    def load(self) -> object:
        return self._value


def _plugin_profile() -> PhysicsBackendProfileV1:
    return PhysicsBackendProfileV1(
        profile_id="custom-pbd-v1",
        display_name="Custom PBD",
        engine_repository="example/custom-pbd",
        solver_family="position-based-dynamics",
        state_representation="persistent-particles",
        query_identity="particle-index",
        differentiability="producer-declared",
        contact_model="native-contact",
        priority=10,
        role="test plugin",
        rationale="Exercises opt-in profile discovery without runtime imports.",
    )


def test_builtin_profiles_are_ranked_and_cover_complementary_mechanisms() -> None:
    profiles = discover_backend_profiles()
    assert profiles == BUILTIN_BACKEND_PROFILES
    assert [profile.profile_id for profile in profiles] == [
        "genesis-mpm-v1",
        "jax-fem-v1",
        "warp-fem-v1",
        "physx-fem-v1",
        "sofa-fem-v1",
        "mujoco-flex-v1",
        "position-based-dynamics-v1",
        "drake-fem-v1",
    ]
    assert [profile.priority for profile in profiles] == list(range(1, 9))
    assert len({profile.engine_repository for profile in profiles}) == len(profiles)
    assert {profile.solver_family for profile in profiles} >= {
        "material-point-method",
        "finite-element-method",
        "flexible-body-dynamics",
        "position-based-dynamics",
    }
    assert all("persistent" in profile.state_representation for profile in profiles)
    assert all(profile.query_identity.endswith("-index") for profile in profiles)


@pytest.mark.parametrize("profile", BUILTIN_BACKEND_PROFILES)
def test_profile_mapping_round_trip_is_exact(
    profile: PhysicsBackendProfileV1,
) -> None:
    assert profile_from_mapping(profile.to_dict()) == profile
    assert get_backend_profile(profile.profile_id) == profile


def test_plugin_discovery_is_explicit_and_deterministic() -> None:
    plugin = _plugin_profile()
    entry_points = [_EntryPoint("zeta", lambda: plugin)]
    assert (
        get_backend_profile(
            plugin.profile_id,
            include_plugins=True,
            entry_points=entry_points,
        )
        == plugin
    )
    with pytest.raises(ValueError, match="entry_points require"):
        discover_backend_profiles(entry_points=entry_points)
    with pytest.raises(ValueError, match="unknown backend profile"):
        get_backend_profile(plugin.profile_id)


def test_plugin_accepts_portable_mappings_and_sequences() -> None:
    plugin = _plugin_profile()
    entry_points = [_EntryPoint("mapping", [plugin.to_dict()])]
    profiles = discover_backend_profiles(
        include_plugins=True,
        entry_points=entry_points,
    )
    assert profiles[-1] == plugin


def test_plugin_cannot_override_a_builtin_profile() -> None:
    duplicate = replace(BUILTIN_BACKEND_PROFILES[0], display_name="Override")
    with pytest.raises(ValueError, match="duplicate backend profile id"):
        discover_backend_profiles(
            include_plugins=True,
            entry_points=[_EntryPoint("duplicate", duplicate)],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("profile_id", "Bad Profile", "lowercase versioned slug"),
        ("engine_repository", "missing-slash", "canonical owner/name"),
        ("priority", 0, "positive integer"),
        ("rationale", " trailing ", "surrounding whitespace"),
    ],
)
def test_profile_rejects_malformed_descriptors(
    field: str,
    value: object,
    message: str,
) -> None:
    profile = BUILTIN_BACKEND_PROFILES[0]
    with pytest.raises(ValueError, match=message):
        replace(profile, **{field: value})


def test_plugin_rejects_empty_and_invalid_values() -> None:
    with pytest.raises(ValueError, match="returned no profiles"):
        discover_backend_profiles(
            include_plugins=True,
            entry_points=[_EntryPoint("empty", [])],
        )
    with pytest.raises(ValueError, match="returned an invalid value"):
        discover_backend_profiles(
            include_plugins=True,
            entry_points=[_EntryPoint("invalid", 5)],
        )
