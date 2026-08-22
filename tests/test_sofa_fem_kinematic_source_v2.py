from __future__ import annotations

import numpy as np

from bayesian_phystwin.jax_fem_source_qualification_v1 import (
    RigidContactProjectionV1,
)
from bayesian_phystwin.native_tet_fem_source_v1 import (
    prepare_native_tet_source_geometry_v1,
)
from bayesian_phystwin.sofa_fem_kinematic_source_v2 import (
    ATTACHMENT_MODEL,
    BACKEND_VARIANT,
    CONTINUATION_POLICY,
    build_sofa_kinematic_schedule_v2,
)


def _geometry_and_contact() -> tuple[object, RigidContactProjectionV1]:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
        ],
        dtype=np.float64,
    )
    translation = np.asarray([0.002, -0.001, 0.0005], dtype=np.float64)
    contact = RigidContactProjectionV1(
        projected_targets_m=np.stack((points, points + translation)),
        rotations=np.repeat(np.eye(3)[None, None], 2, axis=0),
        translations_m=np.asarray(
            [[[0.0, 0.0, 0.0]], [translation]],
            dtype=np.float64,
        ),
        patch_local_indices=(np.arange(4, dtype=np.int64),),
        patch_ranks=(3,),
    )
    geometry = prepare_native_tet_source_geometry_v1(
        points_m=points,
        cells=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        attachment_indices=np.arange(4, dtype=np.int64),
        contact=contact,
    )
    return geometry, contact


def test_schedule_preserves_every_registered_substep_target() -> None:
    geometry, contact = _geometry_and_contact()
    schedule = build_sofa_kinematic_schedule_v2(
        geometry,
        contact,
        driven=True,
        integrator_time_step_s=0.005,
        interval_substeps=4,
    )
    assert BACKEND_VARIANT == "sofa-stable-neo-hookean-keyed-dirichlet-v2"
    assert ATTACHMENT_MODEL.startswith("LinearMovementProjectiveConstraint")
    assert CONTINUATION_POLICY == "registered-rigid-patch-substep-key-schedule-v2"
    np.testing.assert_array_equal(schedule.key_times_s, [0.0, 0.005, 0.01, 0.015, 0.02])
    np.testing.assert_array_equal(schedule.frame_step_indices, [0, 4])
    np.testing.assert_allclose(
        schedule.attached_targets_m[-1],
        contact.projected_targets_m[-1],
        atol=1.0e-15,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        schedule.relative_movements_m[-1],
        contact.projected_targets_m[-1] - geometry.points_m,
        atol=1.0e-15,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        schedule.relative_movements_m[0],
        (contact.projected_targets_m[-1] - geometry.points_m) / 4.0,
        atol=1.0e-15,
        rtol=0.0,
    )


def test_schedule_is_deterministic_and_zero_action_stays_at_rest() -> None:
    geometry, contact = _geometry_and_contact()
    kwargs = {
        "driven": False,
        "integrator_time_step_s": 0.005,
        "interval_substeps": 4,
    }
    left = build_sofa_kinematic_schedule_v2(geometry, contact, **kwargs)
    right = build_sofa_kinematic_schedule_v2(geometry, contact, **kwargs)
    assert left.schedule_sha256 == right.schedule_sha256
    np.testing.assert_array_equal(left.relative_movements_m, 0.0)
