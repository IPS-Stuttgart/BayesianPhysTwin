"""Profiles and opt-in plugin discovery for external physics producers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Final, cast

from ._portable_contracts import nonempty_string, repository_name, require_exact_fields

PHYSICS_BACKEND_PROFILE_SCHEMA: Final = "bayesian-phystwin.physics-backend-profile"
PHYSICS_BACKEND_PROFILE_VERSION: Final = 1
PHYSICS_BACKEND_ENTRY_POINT_GROUP: Final = "bayesian_phystwin.physics_backends.v1"

_PROFILE_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "profile_id",
        "display_name",
        "engine_repository",
        "solver_family",
        "state_representation",
        "query_identity",
        "differentiability",
        "contact_model",
        "priority",
        "role",
        "rationale",
    }
)
_PROFILE_ID_PATTERN: Final = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*-v[1-9][0-9]*\Z")


@dataclass(frozen=True, slots=True)
class PhysicsBackendProfileV1:
    """One self-contained external-producer compatibility profile."""

    profile_id: str
    display_name: str
    engine_repository: str
    solver_family: str
    state_representation: str
    query_identity: str
    differentiability: str
    contact_model: str
    priority: int
    role: str
    rationale: str

    def __post_init__(self) -> None:
        if not _PROFILE_ID_PATTERN.fullmatch(self.profile_id):
            raise ValueError("profile_id must be a lowercase versioned slug")
        for name in (
            "display_name",
            "solver_family",
            "state_representation",
            "query_identity",
            "differentiability",
            "contact_model",
            "role",
            "rationale",
        ):
            value = getattr(self, name)
            nonempty_string(value, name=name)
            if value.strip() != value:
                raise ValueError(f"{name} must not contain surrounding whitespace")
        repository_name(self.engine_repository, name="engine_repository")
        if (
            isinstance(self.priority, bool)
            or not isinstance(self.priority, int)
            or self.priority < 1
        ):
            raise ValueError("priority must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical portable profile descriptor."""

        return {
            "schema": PHYSICS_BACKEND_PROFILE_SCHEMA,
            "schema_version": PHYSICS_BACKEND_PROFILE_VERSION,
            "profile_id": self.profile_id,
            "display_name": self.display_name,
            "engine_repository": self.engine_repository,
            "solver_family": self.solver_family,
            "state_representation": self.state_representation,
            "query_identity": self.query_identity,
            "differentiability": self.differentiability,
            "contact_model": self.contact_model,
            "priority": self.priority,
            "role": self.role,
            "rationale": self.rationale,
        }


def profile_from_mapping(value: Mapping[str, Any]) -> PhysicsBackendProfileV1:
    """Validate and construct one profile from a strict JSON mapping."""

    require_exact_fields(value, expected=_PROFILE_FIELDS, name="backend profile")
    if value.get("schema") != PHYSICS_BACKEND_PROFILE_SCHEMA:
        raise ValueError("backend profile schema changed")
    if value.get("schema_version") != PHYSICS_BACKEND_PROFILE_VERSION:
        raise ValueError("backend profile schema version changed")
    priority = value.get("priority")
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise ValueError("priority must be a positive integer")
    return PhysicsBackendProfileV1(
        profile_id=nonempty_string(value.get("profile_id"), name="profile_id"),
        display_name=nonempty_string(value.get("display_name"), name="display_name"),
        engine_repository=repository_name(
            value.get("engine_repository"), name="engine_repository"
        ),
        solver_family=nonempty_string(value.get("solver_family"), name="solver_family"),
        state_representation=nonempty_string(
            value.get("state_representation"), name="state_representation"
        ),
        query_identity=nonempty_string(
            value.get("query_identity"), name="query_identity"
        ),
        differentiability=nonempty_string(
            value.get("differentiability"), name="differentiability"
        ),
        contact_model=nonempty_string(value.get("contact_model"), name="contact_model"),
        priority=priority,
        role=nonempty_string(value.get("role"), name="role"),
        rationale=nonempty_string(value.get("rationale"), name="rationale"),
    )


BUILTIN_BACKEND_PROFILES: Final[tuple[PhysicsBackendProfileV1, ...]] = (
    PhysicsBackendProfileV1(
        profile_id="genesis-mpm-v1",
        display_name="Genesis World MPM",
        engine_repository="Genesis-Embodied-AI/genesis-world",
        solver_family="material-point-method",
        state_representation="persistent-material-particles",
        query_identity="material-particle-index",
        differentiability="native-autodiff",
        contact_model="coupled-multiphysics-contact",
        priority=1,
        role="primary broad-coverage deformable candidate",
        rationale=(
            "Combines persistent MPM material state, differentiable simulation, "
            "and coupled contact in one actively maintained engine."
        ),
    ),
    PhysicsBackendProfileV1(
        profile_id="jax-fem-v1",
        display_name="JAX-FEM",
        engine_repository="deepmodeling/jax-fem",
        solver_family="finite-element-method",
        state_representation="persistent-mesh-nodes",
        query_identity="mesh-node-index",
        differentiability="jax-autodiff",
        contact_model="producer-declared-contact",
        priority=2,
        role="differentiable material-identification candidate",
        rationale=(
            "Provides a compact differentiable FEM route for parameter inference, "
            "inverse problems, and uncertainty ensembles."
        ),
    ),
    PhysicsBackendProfileV1(
        profile_id="warp-fem-v1",
        display_name="NVIDIA Warp FEM",
        engine_repository="NVIDIA/warp",
        solver_family="finite-element-method",
        state_representation="persistent-mesh-nodes",
        query_identity="mesh-node-index",
        differentiability="native-kernel-autodiff",
        contact_model="producer-declared-fem-contact",
        priority=3,
        role="gpu-differentiable custom-mechanics candidate",
        rationale=(
            "Adds GPU-native differentiable kernels and an extensible FEM toolkit "
            "for custom constitutive, contact, and batched inference experiments."
        ),
    ),
    PhysicsBackendProfileV1(
        profile_id="physx-fem-v1",
        display_name="NVIDIA PhysX Deformables",
        engine_repository="NVIDIA-Omniverse/PhysX",
        solver_family="finite-element-method",
        state_representation="persistent-simulation-mesh-vertices",
        query_identity="simulation-mesh-vertex-index",
        differentiability="external-or-gradient-free",
        contact_model="native-gpu-deformable-contact",
        priority=4,
        role="high-throughput operational deformable reference",
        rationale=(
            "Provides GPU FEM surface and volume deformables with native contact "
            "for scalable operational and simulator-transfer comparisons."
        ),
    ),
    PhysicsBackendProfileV1(
        profile_id="sofa-fem-v1",
        display_name="SOFA FEM",
        engine_repository="sofa-framework/sofa",
        solver_family="finite-element-method",
        state_representation="persistent-mesh-nodes",
        query_identity="mechanical-state-index",
        differentiability="inverse-plugin-or-external",
        contact_model="native-fem-contact",
        priority=5,
        role="mature contact-rich FEM reference",
        rationale=(
            "Adds a mature deformable-contact and constitutive-model reference "
            "that is complementary to differentiable research engines."
        ),
    ),
    PhysicsBackendProfileV1(
        profile_id="mujoco-flex-v1",
        display_name="MuJoCo Flex",
        engine_repository="google-deepmind/mujoco",
        solver_family="flexible-body-dynamics",
        state_representation="persistent-flex-vertices",
        query_identity="flex-vertex-index",
        differentiability="producer-declared-derivatives",
        contact_model="native-flex-contact",
        priority=6,
        role="fast deployment and contact baseline",
        rationale=(
            "Offers a lightweight, fast, and widely deployed flexible-body path "
            "for controls-oriented comparisons and operational fallbacks."
        ),
    ),
    PhysicsBackendProfileV1(
        profile_id="position-based-dynamics-v1",
        display_name="PositionBasedDynamics XPBD",
        engine_repository="InteractiveComputerGraphics/PositionBasedDynamics",
        solver_family="position-based-dynamics",
        state_representation="persistent-particles-and-mesh-vertices",
        query_identity="particle-or-vertex-index",
        differentiability="external-or-gradient-free",
        contact_model="native-pbd-xpbd-contact",
        priority=7,
        role="fast rope cloth and soft-body baseline",
        rationale=(
            "Adds stable interactive coverage for rods, ropes, cloth, and soft "
            "solids where throughput and controllability are primary concerns."
        ),
    ),
    PhysicsBackendProfileV1(
        profile_id="drake-fem-v1",
        display_name="Drake Deformable FEM",
        engine_repository="RobotLocomotion/drake",
        solver_family="finite-element-method",
        state_representation="persistent-volumetric-mesh-vertices",
        query_identity="deformable-body-vertex-index",
        differentiability="double-only-external-inference",
        contact_model="experimental-native-deformable-contact",
        priority=8,
        role="robotics-systems integration candidate",
        rationale=(
            "Connects persistent-topology FEM bodies and deformable contact to a "
            "systems-and-controls stack, while retaining its experimental status."
        ),
    ),
)


def _profile_values(
    value: object,
    *,
    source: str,
) -> tuple[PhysicsBackendProfileV1, ...]:
    if callable(value) and not isinstance(value, PhysicsBackendProfileV1):
        value = value()
    if isinstance(value, PhysicsBackendProfileV1):
        return (value,)
    if isinstance(value, Mapping):
        return (profile_from_mapping(cast(Mapping[str, Any], value)),)
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"backend profile plugin {source} returned an invalid value")
    profiles: list[PhysicsBackendProfileV1] = []
    for index, item in enumerate(value):
        profiles.extend(_profile_values(item, source=f"{source}[{index}]"))
    if not profiles:
        raise ValueError(f"backend profile plugin {source} returned no profiles")
    return tuple(profiles)


def _installed_entry_points() -> tuple[Any, ...]:
    installed = metadata.entry_points()
    if hasattr(installed, "select"):
        return tuple(installed.select(group=PHYSICS_BACKEND_ENTRY_POINT_GROUP))
    return tuple(
        entry_point
        for entry_point in installed
        if getattr(entry_point, "group", None) == PHYSICS_BACKEND_ENTRY_POINT_GROUP
    )


def discover_backend_profiles(
    *,
    include_plugins: bool = False,
    entry_points: Sequence[Any] | None = None,
) -> tuple[PhysicsBackendProfileV1, ...]:
    """Return validated profiles in priority order.

    Third-party code is imported only when ``include_plugins`` is true. Tests and
    controlled callers may inject an explicit entry-point sequence.
    """

    profiles = list(BUILTIN_BACKEND_PROFILES)
    selected = tuple(entry_points or ())
    if include_plugins:
        selected = _installed_entry_points() if entry_points is None else selected
        for entry_point in sorted(selected, key=lambda item: str(item.name)):
            loaded = entry_point.load()
            profiles.extend(
                _profile_values(loaded, source=f"entry point {entry_point.name!r}")
            )
    elif selected:
        raise ValueError("entry_points require include_plugins=True")

    by_id: dict[str, PhysicsBackendProfileV1] = {}
    for profile in profiles:
        if profile.profile_id in by_id:
            raise ValueError(f"duplicate backend profile id: {profile.profile_id}")
        by_id[profile.profile_id] = profile
    return tuple(
        sorted(by_id.values(), key=lambda item: (item.priority, item.profile_id))
    )


def get_backend_profile(
    profile_id: str,
    *,
    include_plugins: bool = False,
    entry_points: Sequence[Any] | None = None,
) -> PhysicsBackendProfileV1:
    """Resolve one exact profile id or fail closed."""

    nonempty_string(profile_id, name="profile_id")
    for profile in discover_backend_profiles(
        include_plugins=include_plugins,
        entry_points=entry_points,
    ):
        if profile.profile_id == profile_id:
            return profile
    raise ValueError(f"unknown backend profile: {profile_id}")


__all__ = [
    "BUILTIN_BACKEND_PROFILES",
    "PHYSICS_BACKEND_ENTRY_POINT_GROUP",
    "PHYSICS_BACKEND_PROFILE_SCHEMA",
    "PHYSICS_BACKEND_PROFILE_VERSION",
    "PhysicsBackendProfileV1",
    "discover_backend_profiles",
    "get_backend_profile",
    "profile_from_mapping",
]
