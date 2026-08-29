"""Source-only HOOD mesh-sequence qualification contracts."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Any, Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from ._portable_contracts import (
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
    write_atomic_json,
)

PLAN_SCHEMA: Final = "bayesian-phystwin.hood-mesh-source-qualification-plan"
RESULT_SCHEMA: Final = "bayesian-phystwin.hood-mesh-source-qualification-result"
ATTEMPT_SCHEMA: Final = "bayesian-phystwin.hood-mesh-source-attempt"
SCHEMA_VERSION: Final = 1

HOOD_REPOSITORY: Final = "Dolorousrtur/HOOD"
HOOD_REVISION: Final = "9bc1076195979ac6c027fdd729c6e960cad62f2a"
PUBLIC_ARCHIVE_SHA256: Final = (
    "3b68239bea3f298f9456680e34cf0204c90512ba1e43233febb375a90038a2a4"
)
PUBLIC_ARCHIVE_BYTE_COUNT: Final = 604_239_517
POSTCVPR_CHECKPOINT_SHA256: Final = (
    "155d2dd25e54756fc04b0d27996ebca3446b2a59d3a715bb1fb73407753ce5ea"
)
MESH_SEQUENCE_SHA256: Final = (
    "1ad213334bf1bdb01bcc831f3c579afc063e832f0d7b99407a6f187d07b3059a"
)
TSHIRT_TEMPLATE_SHA256: Final = (
    "57717c231b80d5c6f9eeb26fc350aa6060eb1ad7584a1f4224cda02025d99454"
)
TSHIRT_OBJ_SHA256: Final = (
    "33d82264415a0bd30faf894ef0c8dcda4ce994d2e8682147d410990f95ae93bc"
)

RANDOM_SEED: Final = 20_260_830
REPLAY_COUNT: Final = 2
ROLLOUT_STEPS: Final = 30

CLAIM_BOUNDARY: Final = (
    "A source-only numerical and interface qualification of the exact HOOD "
    "post-CVPR checkpoint on its public arbitrary-mesh example. A pass shows "
    "repeatable finite short-horizon execution with nontrivial motion. It is "
    "not physical accuracy, 4D-DRESS access, source competence, a selective "
    "risk certificate, deployment evidence, or state of the art."
)

_PLAN_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "plan_id",
        "protocol_label",
        "claim_boundary",
        "implementation",
        "upstream",
        "public_source",
        "runtime",
        "execution",
        "gates",
        "information_boundary",
    }
)
_IMPLEMENTATION_FIELDS: Final = frozenset(
    {
        "repository",
        "revision",
        "source_archive_sha256",
        "source_files",
    }
)
_UPSTREAM_FIELDS: Final = frozenset(
    {
        "repository",
        "revision",
        "git_archive_sha256",
        "config_relative_path",
        "config_sha256",
    }
)
_PUBLIC_SOURCE_FIELDS: Final = frozenset(
    {
        "archive_url",
        "archive_sha256",
        "archive_byte_count",
        "checkpoint_relative_path",
        "checkpoint_sha256",
        "mesh_sequence_relative_path",
        "mesh_sequence_sha256",
        "garment_template_relative_path",
        "garment_template_sha256",
        "garment_obj_relative_path",
        "garment_obj_sha256",
    }
)
_RUNTIME_FIELDS: Final = frozenset(
    {
        "base_python_path",
        "base_python_sha256",
        "base_freeze_sha256",
        "python_overlay_path",
        "python_overlay_tree_sha256",
        "cuda_visible_device",
        "torch_version",
        "torch_cuda_version",
        "torch_geometric_version",
        "pytorch3d_version",
    }
)
_EXECUTION_FIELDS: Final = frozenset(
    {
        "output_root",
        "attempt_ledger_path",
        "attempt_limit",
        "random_seed",
        "replay_count",
        "rollout_steps",
        "configuration_name",
        "pose_sequence_type",
        "source_execution_authorized",
    }
)
_GATE_FIELDS: Final = frozenset(
    {
        "maximum_repeat_rmse_m",
        "minimum_cloth_motion_m",
        "minimum_obstacle_motion_m",
        "maximum_absolute_coordinate_m",
        "all_values_finite_required",
        "exact_frame_count_required",
        "topology_identity_required",
    }
)
_BOUNDARY_FIELDS: Final = frozenset(
    {
        "public_hood_source_read",
        "fourddress_payload_read",
        "fourddress_participant_roster_read",
        "physical_outcomes_read",
        "certification_outcomes_read",
        "held_v8_read",
        "dlo4_or_dlo5_read",
        "certification_execution_authorized",
        "replacement_allowed",
    }
)

_EXPECTED_PUBLIC_SOURCE: Final = {
    "archive_url": (
        "https://drive.google.com/file/d/"
        "1RdA4L6Fy50VsKZ8k7ySp5ps5YtWoHSgs/view?usp=sharing"
    ),
    "archive_sha256": PUBLIC_ARCHIVE_SHA256,
    "archive_byte_count": PUBLIC_ARCHIVE_BYTE_COUNT,
    "checkpoint_relative_path": "hood_data/trained_models/postcvpr.pth",
    "checkpoint_sha256": POSTCVPR_CHECKPOINT_SHA256,
    "mesh_sequence_relative_path": "hood_data/fromanypose/mesh_sequence.pkl",
    "mesh_sequence_sha256": MESH_SEQUENCE_SHA256,
    "garment_template_relative_path": "hood_data/fromanypose/tshirt.pkl",
    "garment_template_sha256": TSHIRT_TEMPLATE_SHA256,
    "garment_obj_relative_path": "hood_data/fromanypose/tshirt.obj",
    "garment_obj_sha256": TSHIRT_OBJ_SHA256,
}
_EXPECTED_EXECUTION_POLICY: Final = {
    "attempt_limit": 1,
    "random_seed": RANDOM_SEED,
    "replay_count": REPLAY_COUNT,
    "rollout_steps": ROLLOUT_STEPS,
    "configuration_name": "aux/from_any_pose",
    "pose_sequence_type": "mesh",
    "source_execution_authorized": True,
}
_EXPECTED_GATES: Final = {
    "maximum_repeat_rmse_m": 1e-7,
    "minimum_cloth_motion_m": 1e-5,
    "minimum_obstacle_motion_m": 1e-5,
    "maximum_absolute_coordinate_m": 10.0,
    "all_values_finite_required": True,
    "exact_frame_count_required": True,
    "topology_identity_required": True,
}
_EXPECTED_BOUNDARY: Final = {
    "public_hood_source_read": True,
    "fourddress_payload_read": False,
    "fourddress_participant_roster_read": False,
    "physical_outcomes_read": False,
    "certification_outcomes_read": False,
    "held_v8_read": False,
    "dlo4_or_dlo5_read": False,
    "certification_execution_authorized": False,
    "replacement_allowed": False,
}

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]


def file_sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _finite_real(value: object, *, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be a finite real >= {minimum}")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _absolute_path(value: object, *, name: str) -> Path:
    text = nonempty_string(value, name=name)
    path = Path(text)
    if not path.is_absolute() or str(path) != text or os.path.normpath(text) != text:
        raise ValueError(f"{name} must be a canonical absolute path")
    return path


def _relative_path(value: object, *, name: str) -> str:
    text = cast(str, nonempty_string(value, name=name))
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or not path.parts
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{name} must be a canonical relative POSIX path")
    return text


@dataclass(frozen=True, slots=True)
class HoodSourceQualificationPlanV1:
    value: Mapping[str, Any]
    plan_id: str
    implementation_source_files: Mapping[str, Any]
    output_root: Path
    attempt_ledger_path: Path
    base_python_path: Path
    python_overlay_path: Path
    cuda_visible_device: int


def load_hood_source_qualification_plan_v1(
    path: str | Path,
) -> HoodSourceQualificationPlanV1:
    value = load_strict_json_object(path, label="HOOD source qualification plan")
    require_exact_fields(value, expected=_PLAN_FIELDS, name="HOOD source plan")
    if (
        value["schema"] != PLAN_SCHEMA
        or type(value["schema_version"]) is not int
        or value["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("HOOD source plan schema changed")
    nonempty_string(value["protocol_label"], name="protocol_label")
    plan_id = sha256_digest(value["plan_id"], name="plan_id")
    descriptor = dict(value)
    descriptor.pop("plan_id")
    if content_id(descriptor) != plan_id:
        raise ValueError("HOOD source plan_id changed")
    if value["claim_boundary"] != CLAIM_BOUNDARY:
        raise ValueError("HOOD source claim boundary changed")

    implementation = _mapping(value["implementation"], name="implementation")
    require_exact_fields(
        implementation,
        expected=_IMPLEMENTATION_FIELDS,
        name="implementation",
    )
    if implementation["repository"] != "IPS-Stuttgart/BayesianPhysTwin":
        raise ValueError("implementation repository changed")
    exact_revision(implementation["revision"], name="implementation revision")
    sha256_digest(
        implementation["source_archive_sha256"],
        name="implementation source archive",
    )
    source_files = source_artifact_mapping(
        _mapping(implementation["source_files"], name="source files"),
        name="implementation source files",
    )

    upstream = _mapping(value["upstream"], name="upstream")
    require_exact_fields(upstream, expected=_UPSTREAM_FIELDS, name="upstream")
    if upstream["repository"] != HOOD_REPOSITORY:
        raise ValueError("HOOD repository changed")
    if upstream["revision"] != HOOD_REVISION:
        raise ValueError("HOOD revision changed")
    sha256_digest(upstream["git_archive_sha256"], name="HOOD git archive")
    if upstream["config_relative_path"] != "configs/aux/from_any_pose.yaml":
        raise ValueError("HOOD configuration path changed")
    _relative_path(upstream["config_relative_path"], name="configuration path")
    sha256_digest(upstream["config_sha256"], name="configuration sha256")

    public_source = _mapping(value["public_source"], name="public source")
    require_exact_fields(
        public_source,
        expected=_PUBLIC_SOURCE_FIELDS,
        name="public source",
    )
    if dict(public_source) != _EXPECTED_PUBLIC_SOURCE:
        raise ValueError("public HOOD source changed")

    runtime = _mapping(value["runtime"], name="runtime")
    require_exact_fields(runtime, expected=_RUNTIME_FIELDS, name="runtime")
    base_python_path = _absolute_path(
        runtime["base_python_path"],
        name="base_python_path",
    )
    python_overlay_path = _absolute_path(
        runtime["python_overlay_path"],
        name="python_overlay_path",
    )
    for field in (
        "base_python_sha256",
        "base_freeze_sha256",
        "python_overlay_tree_sha256",
    ):
        sha256_digest(runtime[field], name=field)
    if runtime["torch_version"] != "2.0.1+cu118":
        raise ValueError("torch version changed")
    if runtime["torch_cuda_version"] != "11.8":
        raise ValueError("torch CUDA version changed")
    if runtime["torch_geometric_version"] != "2.4.0":
        raise ValueError("torch-geometric version changed")
    if runtime["pytorch3d_version"] != "0.7.4":
        raise ValueError("PyTorch3D version changed")
    cuda_visible_device = runtime["cuda_visible_device"]
    if (
        isinstance(cuda_visible_device, bool)
        or not isinstance(cuda_visible_device, int)
        or cuda_visible_device not in {0, 1}
    ):
        raise ValueError("cuda_visible_device must be 0 or 1")

    execution = _mapping(value["execution"], name="execution")
    require_exact_fields(execution, expected=_EXECUTION_FIELDS, name="execution")
    for key, expected in _EXPECTED_EXECUTION_POLICY.items():
        if type(execution[key]) is not type(expected) or execution[key] != expected:
            raise ValueError(f"execution policy changed: {key}")
    output_root = _absolute_path(execution["output_root"], name="output_root")
    attempt_ledger_path = _absolute_path(
        execution["attempt_ledger_path"],
        name="attempt_ledger_path",
    )
    if output_root == attempt_ledger_path or output_root in attempt_ledger_path.parents:
        raise ValueError("attempt ledger must be separate from the output root")

    gates = _mapping(value["gates"], name="gates")
    require_exact_fields(gates, expected=_GATE_FIELDS, name="gates")
    if content_id(gates) != content_id(_EXPECTED_GATES):
        raise ValueError("HOOD source gates changed")

    boundary = _mapping(value["information_boundary"], name="boundary")
    require_exact_fields(boundary, expected=_BOUNDARY_FIELDS, name="boundary")
    if content_id(boundary) != content_id(_EXPECTED_BOUNDARY):
        raise ValueError("HOOD source information boundary changed")

    return HoodSourceQualificationPlanV1(
        value=value,
        plan_id=plan_id,
        implementation_source_files=source_files,
        output_root=output_root,
        attempt_ledger_path=attempt_ledger_path,
        base_python_path=base_python_path,
        python_overlay_path=python_overlay_path,
        cuda_visible_device=cuda_visible_device,
    )


def consume_hood_source_attempt_v1(
    plan: HoodSourceQualificationPlanV1,
) -> Mapping[str, Any]:
    """Atomically consume the one allowed source attempt."""

    ledger = plan.attempt_ledger_path
    if ledger.parent.exists() and not ledger.parent.is_dir():
        raise ValueError("attempt ledger parent must be a directory")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": ATTEMPT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "plan_id": plan.plan_id,
        "attempt_index": 1,
        "attempt_limit": 1,
        "output_root": str(plan.output_root),
        "information_boundary": plain_json(plan.value["information_boundary"]),
    }
    write_atomic_json(payload, ledger, overwrite=False)
    return cast(
        Mapping[str, Any], frozen_finite_json_mapping(payload, name="attempt ledger")
    )


@dataclass(frozen=True, slots=True)
class HoodSourceReplayAssessmentV1:
    passed: bool
    decisions: Mapping[str, bool]
    metrics: Mapping[str, float | int]


def assess_hood_source_replays_v1(
    predictions: Sequence[FloatArray],
    obstacles: Sequence[FloatArray],
    cloth_faces: Sequence[npt.NDArray[np.integer[Any]]],
    obstacle_faces: Sequence[npt.NDArray[np.integer[Any]]],
    *,
    rollout_steps: int = ROLLOUT_STEPS,
    maximum_repeat_rmse_m: float = 1e-7,
    minimum_cloth_motion_m: float = 1e-5,
    minimum_obstacle_motion_m: float = 1e-5,
    maximum_absolute_coordinate_m: float = 10.0,
) -> HoodSourceReplayAssessmentV1:
    """Assess structural validity and numerical repeatability of two replays."""

    if not (
        len(predictions)
        == len(obstacles)
        == len(cloth_faces)
        == len(obstacle_faces)
        == REPLAY_COUNT
    ):
        raise ValueError(f"exactly {REPLAY_COUNT} complete replays are required")
    rollout_steps = _positive_integer(rollout_steps, name="rollout_steps")
    maximum_repeat_rmse_m = _finite_real(
        maximum_repeat_rmse_m,
        name="maximum_repeat_rmse_m",
    )
    minimum_cloth_motion_m = _finite_real(
        minimum_cloth_motion_m,
        name="minimum_cloth_motion_m",
    )
    minimum_obstacle_motion_m = _finite_real(
        minimum_obstacle_motion_m,
        name="minimum_obstacle_motion_m",
    )
    maximum_absolute_coordinate_m = _finite_real(
        maximum_absolute_coordinate_m,
        name="maximum_absolute_coordinate_m",
    )

    prediction_arrays = tuple(
        np.asarray(value, dtype=np.float64) for value in predictions
    )
    obstacle_arrays = tuple(np.asarray(value, dtype=np.float64) for value in obstacles)
    cloth_face_arrays = tuple(np.asarray(value) for value in cloth_faces)
    obstacle_face_arrays = tuple(np.asarray(value) for value in obstacle_faces)
    if any(value.ndim != 3 or value.shape[-1] != 3 for value in prediction_arrays):
        raise ValueError("predictions must have shape (T, V, 3)")
    if any(value.ndim != 3 or value.shape[-1] != 3 for value in obstacle_arrays):
        raise ValueError("obstacles must have shape (T, V, 3)")
    if any(value.ndim != 2 or value.shape[-1] != 3 for value in cloth_face_arrays):
        raise ValueError("cloth faces must have shape (F, 3)")
    if any(value.ndim != 2 or value.shape[-1] != 3 for value in obstacle_face_arrays):
        raise ValueError("obstacle faces must have shape (F, 3)")
    if any(value.shape != prediction_arrays[0].shape for value in prediction_arrays):
        raise ValueError("prediction shapes changed between replays")
    if any(value.shape != obstacle_arrays[0].shape for value in obstacle_arrays):
        raise ValueError("obstacle shapes changed between replays")
    if any(
        value.size == 0
        for value in (
            *prediction_arrays,
            *obstacle_arrays,
            *cloth_face_arrays,
            *obstacle_face_arrays,
        )
    ):
        raise ValueError("empty mesh or trajectory is not a complete replay")
    for face_arrays, trajectories in (
        (cloth_face_arrays, prediction_arrays),
        (obstacle_face_arrays, obstacle_arrays),
    ):
        if any(
            not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0)
            or np.any(faces >= trajectory.shape[1])
            for faces, trajectory in zip(face_arrays, trajectories, strict=True)
        ):
            raise ValueError("faces must contain valid integer vertex indices")

    finite = all(
        bool(np.all(np.isfinite(value)))
        for value in (*prediction_arrays, *obstacle_arrays)
    )
    nonfinite_value_count = sum(
        int(np.size(value) - np.count_nonzero(np.isfinite(value)))
        for value in (*prediction_arrays, *obstacle_arrays)
    )
    exact_frame_count = all(
        value.shape[0] == rollout_steps
        for value in (*prediction_arrays, *obstacle_arrays)
    )
    topology_identity = bool(
        np.array_equal(cloth_face_arrays[0], cloth_face_arrays[1])
        and np.array_equal(obstacle_face_arrays[0], obstacle_face_arrays[1])
    )
    maximum_coordinate = (
        float(
            max(
                np.max(np.abs(value))
                for value in (*prediction_arrays, *obstacle_arrays)
            )
        )
        if finite
        else maximum_absolute_coordinate_m + 1.0
    )
    bounded = finite and maximum_coordinate <= maximum_absolute_coordinate_m
    repeat_rmse = (
        float(np.sqrt(np.mean(np.square(prediction_arrays[0] - prediction_arrays[1]))))
        if bounded
        else maximum_repeat_rmse_m + 1.0
    )
    obstacle_repeat_rmse = (
        float(np.sqrt(np.mean(np.square(obstacle_arrays[0] - obstacle_arrays[1]))))
        if bounded
        else maximum_repeat_rmse_m + 1.0
    )
    cloth_motion = (
        float(
            np.mean(
                np.linalg.norm(
                    prediction_arrays[0][-1] - prediction_arrays[0][0],
                    axis=-1,
                )
            )
        )
        if bounded and exact_frame_count
        else 0.0
    )
    obstacle_motion = (
        float(
            np.mean(
                np.linalg.norm(
                    obstacle_arrays[0][-1] - obstacle_arrays[0][0],
                    axis=-1,
                )
            )
        )
        if bounded and exact_frame_count
        else 0.0
    )
    decisions = {
        "all_values_finite": finite,
        "exact_frame_count": exact_frame_count,
        "topology_identity": topology_identity,
        "repeatability": (
            repeat_rmse <= maximum_repeat_rmse_m
            and obstacle_repeat_rmse <= maximum_repeat_rmse_m
        ),
        "nontrivial_cloth_motion": cloth_motion >= minimum_cloth_motion_m,
        "nontrivial_obstacle_motion": obstacle_motion >= minimum_obstacle_motion_m,
        "coordinates_bounded": bounded,
    }
    metrics: dict[str, float | int] = {
        "rollout_steps": int(prediction_arrays[0].shape[0]),
        "cloth_vertex_count": int(prediction_arrays[0].shape[1]),
        "obstacle_vertex_count": int(obstacle_arrays[0].shape[1]),
        "cloth_face_count": int(cloth_face_arrays[0].shape[0]),
        "obstacle_face_count": int(obstacle_face_arrays[0].shape[0]),
        "nonfinite_value_count": nonfinite_value_count,
        "repeat_rmse_m": repeat_rmse,
        "obstacle_repeat_rmse_m": obstacle_repeat_rmse,
        "cloth_mean_displacement_m": cloth_motion,
        "obstacle_mean_displacement_m": obstacle_motion,
        "maximum_absolute_coordinate_m": maximum_coordinate,
    }
    return HoodSourceReplayAssessmentV1(
        passed=all(decisions.values()),
        decisions=frozen_finite_json_mapping(decisions, name="source decisions"),
        metrics=frozen_finite_json_mapping(metrics, name="source metrics"),
    )


def build_hood_source_result_v1(
    *,
    plan: HoodSourceQualificationPlanV1,
    assessment: HoodSourceReplayAssessmentV1,
    replay_archive_sha256: str,
    elapsed_seconds: float,
) -> Mapping[str, Any]:
    descriptor: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "plan_id": plan.plan_id,
        "passed": assessment.passed,
        "decisions": plain_json(assessment.decisions),
        "metrics": plain_json(assessment.metrics),
        "replay_archive_sha256": sha256_digest(
            replay_archive_sha256,
            name="replay archive sha256",
        ),
        "elapsed_seconds": _finite_real(
            elapsed_seconds,
            name="elapsed_seconds",
        ),
        "information_boundary": plain_json(plan.value["information_boundary"]),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    descriptor["result_id"] = content_id(descriptor)
    return cast(
        Mapping[str, Any],
        frozen_finite_json_mapping(descriptor, name="HOOD source result"),
    )


def save_hood_source_result_v1(
    value: Mapping[str, Any],
    path: str | Path,
) -> Path:
    destination = Path(path)
    write_atomic_json(value, destination, overwrite=False)
    return destination
