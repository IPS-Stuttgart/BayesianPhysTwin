"""Guarded MatPhys spring proposals as a Bayesian-PhysTwin physical backend.

MatPhys predicts a spring field; the trajectory is still produced by the
official PhysTwin Warp simulator.  This module makes that distinction explicit
and adapts a future-blind MatPhys/Warp replay to the six-array physical archive
consumed by Bayesian-PhysTwin.  Rejected proposals copy the incumbent archive
byte for byte.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any, Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from ._canonical_contracts import plain_json
from ._portable_contracts import (
    canonical_relative_posix_path,
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    repository_name,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
    write_atomic_json,
)
from .deform360_bias_aware_prospective_artifacts import PHYSICAL_ARRAY_NAMES
from .phystwin_backbone_family_gate import (
    choose_guarded_backbone_family,
    trajectory_coordinate_rmse,
)

MATPHYS_BACKEND_PROPOSAL_SCHEMA: Final = (
    "bayesian-phystwin.matphys-backend-proposal"
)
MATPHYS_BACKEND_GATE_SCHEMA: Final = "bayesian-phystwin.matphys-backend-gate"
MATPHYS_BACKEND_ARTIFACT_SCHEMA: Final = (
    "bayesian-phystwin.matphys-backend-artifact"
)
MATPHYS_BACKEND_VERSION: Final = 1
MATPHYS_BACKEND_KIND: Final = "matphys-spring-proposal-phystwin-warp-v1"
MATPHYS_PARAMETERIZATION: Final = "spring_Y-log-space-overlay"
MATPHYS_ROLLOUT_BACKEND: Final = "official-phystwin-warp"
MATPHYS_SOURCE_REPOSITORY: Final = "Yrainy0615/MatPhys"
PHYSTWIN_SOURCE_REPOSITORY: Final = "Jianghanxiao/PhysTwin"
SELECTED_ARCHIVE_FILENAME: Final = "physical-prediction.npz"
ARTIFACT_FILENAME: Final = "matphys-backend.json"
PROPOSAL_FILENAME: Final = "matphys-proposal.json"
GATE_FILENAME: Final = "matphys-gate.json"
CHECKSUMS_FILENAME: Final = "SHA256SUMS"
_ROOT_ROSTER: Final = frozenset(
    {
        ARTIFACT_FILENAME,
        CHECKSUMS_FILENAME,
        SELECTED_ARCHIVE_FILENAME,
        "provenance",
    }
)
_PROVENANCE_ROSTER: Final = frozenset({PROPOSAL_FILENAME, GATE_FILENAME})

MATPHYS_BACKEND_CLAIM_BOUNDARY: Final = (
    "A MatPhys spring-field proposal replayed by the official PhysTwin Warp "
    "simulator. Selection uses only the declared causal validation prefix and "
    "has an exact incumbent fallback. This artifact alone does not establish "
    "target transfer, predictive calibration, safety, or state of the art."
)

_PROPOSAL_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "backend_kind",
        "parameterization",
        "rollout_backend",
        "source_repository",
        "source_revision",
        "simulator_repository",
        "simulator_revision",
        "target_object_id",
        "training_object_ids",
        "target_object_excluded",
        "target_evidence_end_frame_exclusive",
        "target_future_observations_used",
        "known_future_robot_action_used",
        "proposal_strength",
        "checkpoint",
        "spring_field",
        "source_artifacts",
        "claim_boundary",
        "proposal_id",
    }
)
_GATE_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "proposal_id",
        "target_object_id",
        "case_id",
        "validation_frame_range_half_open",
        "future_frame_start",
        "future_outcomes_opened",
        "evaluated_archive_sha256s",
        "incumbent_metrics",
        "candidate_metrics",
        "minimum_relative_improvement",
        "maximum_metric_regression",
        "maximum_identity_replay_rmse_m",
        "source_artifacts",
        "gate_id",
    }
)
_ARTIFACT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "backend_kind",
        "proposal_id",
        "gate_id",
        "case_id",
        "selected_backend",
        "candidate_accepted",
        "selection",
        "inputs",
        "output",
        "information_boundary",
        "claim_boundary",
        "artifact_id",
    }
)
_FILE_FIELDS: Final = frozenset({"path", "sha256"})
_SPRING_FIELD_FIELDS: Final = frozenset({"path", "sha256", "count"})
_ARTIFACT_INPUT_FIELDS: Final = frozenset(
    {
        "proposal_manifest",
        "gate_manifest",
        "incumbent_archive",
        "candidate_archive",
        "identity_replay_archive",
    }
)
_ARTIFACT_OUTPUT_FIELDS: Final = frozenset(
    {
        "path",
        "sha256",
        "byte_count",
        "source_archive_sha256",
        "byte_exact_source_copy",
        "exact_incumbent_fallback_verified",
    }
)
_SELECTION_FIELDS: Final = frozenset(
    {
        "identity_replay_coordinate_rmse_m",
        "maximum_identity_replay_rmse_m",
        "identity_replay_stable",
        "scores",
        "decisions",
        "minimum_relative_improvement",
        "maximum_metric_regression",
    }
)
_METRIC_FIELDS: Final = frozenset(
    {"chamfer_distance_m", "track_error_m"}
)
_EVALUATED_ARCHIVE_FIELDS: Final = frozenset(
    {
        "incumbent",
        "matphys_warp_proposal",
        "zero_strength_identity_replay",
    }
)
_INFORMATION_BOUNDARY: Final = {
    "target_future_observations_used_for_proposal": False,
    "known_future_robot_action_used": True,
    "future_outcomes_opened_for_gate": False,
    "selection_uses_declared_validation_prefix_only": True,
    "prediction_hashed_before_future_outcome_scoring": True,
    "rejected_proposal_uses_byte_exact_incumbent": True,
}

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[object], value)


def _finite_number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_exclusive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None:
        invalid = result <= minimum if minimum_exclusive else result < minimum
        if invalid:
            relation = ">" if minimum_exclusive else ">="
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _frame_range(value: object, *, name: str) -> tuple[int, int]:
    raw = _sequence(value, name=name)
    if len(raw) != 2:
        raise ValueError(f"{name} must contain exactly two frame indices")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
        raise ValueError(f"{name} must contain integer frame indices")
    start, stop = cast(int, raw[0]), cast(int, raw[1])
    if not 0 <= start < stop:
        raise ValueError(f"{name} must be a nonempty half-open range")
    return start, stop


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordinary_file(path: str | Path, *, name: str) -> Path:
    source = Path(path).absolute()
    _require(
        source.is_file()
        and not source.is_symlink()
        and not any(parent.is_symlink() for parent in source.parents),
        f"{name} must be an ordinary non-symlink file",
    )
    return source.resolve(strict=True)


def _file_identity(path: str | Path, *, name: str) -> dict[str, str]:
    source = _ordinary_file(path, name=name)
    return {"path": str(source), "sha256": _file_sha256(source)}


def _normalize_file_identity(
    value: object,
    *,
    name: str,
    verify_file: bool,
) -> dict[str, str]:
    record = _mapping(value, name=name)
    require_exact_fields(record, expected=_FILE_FIELDS, name=name)
    path = nonempty_string(record.get("path"), name=f"{name}.path")
    digest = sha256_digest(record.get("sha256"), name=f"{name}.sha256")
    if verify_file:
        source = _ordinary_file(path, name=name)
        _require(_file_sha256(source) == digest, f"{name} SHA-256 changed")
        path = str(source)
    return {"path": path, "sha256": digest}


def _normalize_spring_field_identity(
    value: object,
    *,
    verify_file: bool,
) -> dict[str, object]:
    record = _mapping(value, name="spring_field")
    require_exact_fields(
        record, expected=_SPRING_FIELD_FIELDS, name="spring_field"
    )
    identity = _normalize_file_identity(
        {"path": record.get("path"), "sha256": record.get("sha256")},
        name="spring_field",
        verify_file=verify_file,
    )
    count = _positive_integer(record.get("count"), name="spring_field.count")
    if verify_file:
        stored = np.asarray(np.load(identity["path"], allow_pickle=False))
        _require(
            stored.ndim == 1 and np.issubdtype(stored.dtype, np.floating),
            "spring_field must be a one-dimensional floating array",
        )
        values = np.asarray(stored, dtype=np.float64)
        _require(len(values) == count, "spring_field count changed")
        _require(
            np.all(np.isfinite(values)) and np.all(values > 0.0),
            "spring_field must be finite and positive",
        )
    return {**identity, "count": count}


def _normalized_object_ids(value: object) -> tuple[str, ...]:
    raw = _sequence(value, name="training_object_ids")
    items = tuple(
        nonempty_string(item, name="training_object_ids entry") for item in raw
    )
    _require(bool(items), "training_object_ids must not be empty")
    _require(
        items == tuple(sorted(set(items))),
        "training_object_ids must be sorted and unique",
    )
    return items


def _normalized_metrics(value: object, *, name: str) -> dict[str, float]:
    metrics = _mapping(value, name=name)
    require_exact_fields(metrics, expected=_METRIC_FIELDS, name=name)
    return {
        metric: _finite_number(
            metrics.get(metric), name=f"{name}.{metric}", minimum=0.0
        )
        for metric in sorted(_METRIC_FIELDS)
    }


def build_matphys_backend_proposal(
    *,
    source_revision: str,
    simulator_revision: str,
    target_object_id: str,
    training_object_ids: Sequence[str],
    target_evidence_end_frame_exclusive: int,
    proposal_strength: float,
    checkpoint_path: str | Path,
    spring_field_path: str | Path,
    source_artifacts: Mapping[str, str],
) -> dict[str, Any]:
    """Build a strict future-blind MatPhys proposal record."""

    checkpoint = _file_identity(checkpoint_path, name="checkpoint")
    spring_path = _ordinary_file(spring_field_path, name="spring_field")
    spring = np.asarray(np.load(spring_path, allow_pickle=False))
    spring_identity: dict[str, object] = {
        **_file_identity(spring_path, name="spring_field"),
        "count": int(len(spring)),
    }
    identity: dict[str, Any] = {
        "schema": MATPHYS_BACKEND_PROPOSAL_SCHEMA,
        "schema_version": MATPHYS_BACKEND_VERSION,
        "backend_kind": MATPHYS_BACKEND_KIND,
        "parameterization": MATPHYS_PARAMETERIZATION,
        "rollout_backend": MATPHYS_ROLLOUT_BACKEND,
        "source_repository": MATPHYS_SOURCE_REPOSITORY,
        "source_revision": source_revision,
        "simulator_repository": PHYSTWIN_SOURCE_REPOSITORY,
        "simulator_revision": simulator_revision,
        "target_object_id": target_object_id,
        "training_object_ids": list(training_object_ids),
        "target_object_excluded": True,
        "target_evidence_end_frame_exclusive": target_evidence_end_frame_exclusive,
        "target_future_observations_used": False,
        "known_future_robot_action_used": True,
        "proposal_strength": proposal_strength,
        "checkpoint": checkpoint,
        "spring_field": spring_identity,
        "source_artifacts": dict(source_artifacts),
        "claim_boundary": MATPHYS_BACKEND_CLAIM_BOUNDARY,
    }
    proposal = {**identity, "proposal_id": content_id(identity)}
    return validate_matphys_backend_proposal(proposal, verify_files=True)


def validate_matphys_backend_proposal(
    value: object,
    *,
    verify_files: bool,
) -> dict[str, Any]:
    """Validate a MatPhys proposal and optionally rehash its large inputs."""

    proposal = _mapping(value, name="MatPhys backend proposal")
    require_exact_fields(
        proposal,
        expected=_PROPOSAL_FIELDS,
        name="MatPhys backend proposal",
    )
    _require(
        proposal.get("schema") == MATPHYS_BACKEND_PROPOSAL_SCHEMA
        and proposal.get("schema_version") == MATPHYS_BACKEND_VERSION,
        "MatPhys proposal schema changed",
    )
    _require(
        proposal.get("backend_kind") == MATPHYS_BACKEND_KIND
        and proposal.get("parameterization") == MATPHYS_PARAMETERIZATION
        and proposal.get("rollout_backend") == MATPHYS_ROLLOUT_BACKEND,
        "MatPhys backend semantics changed",
    )
    source_repository = repository_name(
        proposal.get("source_repository"), name="source_repository"
    )
    simulator_repository = repository_name(
        proposal.get("simulator_repository"), name="simulator_repository"
    )
    _require(
        source_repository == MATPHYS_SOURCE_REPOSITORY,
        "MatPhys source repository changed",
    )
    _require(
        simulator_repository == PHYSTWIN_SOURCE_REPOSITORY,
        "MatPhys rollout must use the official PhysTwin repository",
    )
    target = nonempty_string(
        proposal.get("target_object_id"), name="target_object_id"
    )
    training = _normalized_object_ids(proposal.get("training_object_ids"))
    _require(target not in training, "MatPhys training includes the target object")
    _require(
        proposal.get("target_object_excluded") is True,
        "MatPhys proposal does not declare target-object exclusion",
    )
    _require(
        proposal.get("target_future_observations_used") is False,
        "MatPhys proposal used target future observations",
    )
    _require(
        proposal.get("known_future_robot_action_used") is True,
        "MatPhys proposal changed the action-conditioned task",
    )
    checkpoint = _normalize_file_identity(
        proposal.get("checkpoint"), name="checkpoint", verify_file=verify_files
    )
    spring_field = _normalize_spring_field_identity(
        proposal.get("spring_field"), verify_file=verify_files
    )
    artifacts = source_artifact_mapping(
        _mapping(proposal.get("source_artifacts"), name="source_artifacts"),
        name="source_artifacts",
    )
    identity: dict[str, Any] = {
        "schema": MATPHYS_BACKEND_PROPOSAL_SCHEMA,
        "schema_version": MATPHYS_BACKEND_VERSION,
        "backend_kind": MATPHYS_BACKEND_KIND,
        "parameterization": MATPHYS_PARAMETERIZATION,
        "rollout_backend": MATPHYS_ROLLOUT_BACKEND,
        "source_repository": source_repository,
        "source_revision": exact_revision(
            proposal.get("source_revision"), name="source_revision"
        ),
        "simulator_repository": simulator_repository,
        "simulator_revision": exact_revision(
            proposal.get("simulator_revision"), name="simulator_revision"
        ),
        "target_object_id": target,
        "training_object_ids": list(training),
        "target_object_excluded": True,
        "target_evidence_end_frame_exclusive": _nonnegative_integer(
            proposal.get("target_evidence_end_frame_exclusive"),
            name="target_evidence_end_frame_exclusive",
        ),
        "target_future_observations_used": False,
        "known_future_robot_action_used": True,
        "proposal_strength": _finite_number(
            proposal.get("proposal_strength"),
            name="proposal_strength",
            minimum=0.0,
            maximum=1.0,
            minimum_exclusive=True,
        ),
        "checkpoint": checkpoint,
        "spring_field": spring_field,
        "source_artifacts": dict(artifacts),
        "claim_boundary": nonempty_string(
            proposal.get("claim_boundary"), name="claim_boundary"
        ),
    }
    _require(
        identity["claim_boundary"] == MATPHYS_BACKEND_CLAIM_BOUNDARY,
        "MatPhys proposal claim boundary changed",
    )
    normalized = {**identity, "proposal_id": content_id(identity)}
    _require(
        proposal.get("proposal_id") == normalized["proposal_id"],
        "MatPhys proposal content identity changed",
    )
    return cast(dict[str, Any], plain_json(normalized))


def build_matphys_backend_gate(
    *,
    proposal_id: str,
    target_object_id: str,
    case_id: str,
    validation_frame_range_half_open: tuple[int, int],
    future_frame_start: int,
    incumbent_archive_path: str | Path,
    candidate_archive_path: str | Path,
    identity_replay_archive_path: str | Path,
    incumbent_metrics: Mapping[str, float],
    candidate_metrics: Mapping[str, float],
    minimum_relative_improvement: float,
    maximum_metric_regression: float,
    maximum_identity_replay_rmse_m: float,
    source_artifacts: Mapping[str, str],
) -> dict[str, Any]:
    """Build the prefix-only gate input consumed by the backend selector."""

    identity: dict[str, Any] = {
        "schema": MATPHYS_BACKEND_GATE_SCHEMA,
        "schema_version": MATPHYS_BACKEND_VERSION,
        "proposal_id": proposal_id,
        "target_object_id": target_object_id,
        "case_id": case_id,
        "validation_frame_range_half_open": list(
            validation_frame_range_half_open
        ),
        "future_frame_start": future_frame_start,
        "future_outcomes_opened": False,
        "evaluated_archive_sha256s": {
            "incumbent": _file_sha256(
                _ordinary_file(incumbent_archive_path, name="incumbent archive")
            ),
            "matphys_warp_proposal": _file_sha256(
                _ordinary_file(candidate_archive_path, name="candidate archive")
            ),
            "zero_strength_identity_replay": _file_sha256(
                _ordinary_file(
                    identity_replay_archive_path, name="identity replay archive"
                )
            ),
        },
        "incumbent_metrics": dict(incumbent_metrics),
        "candidate_metrics": dict(candidate_metrics),
        "minimum_relative_improvement": minimum_relative_improvement,
        "maximum_metric_regression": maximum_metric_regression,
        "maximum_identity_replay_rmse_m": maximum_identity_replay_rmse_m,
        "source_artifacts": dict(source_artifacts),
    }
    gate = {**identity, "gate_id": content_id(identity)}
    return validate_matphys_backend_gate(gate)


def validate_matphys_backend_gate(
    value: object,
    *,
    proposal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a causal-prefix gate and its relation to one proposal."""

    gate = _mapping(value, name="MatPhys backend gate")
    require_exact_fields(gate, expected=_GATE_FIELDS, name="MatPhys backend gate")
    _require(
        gate.get("schema") == MATPHYS_BACKEND_GATE_SCHEMA
        and gate.get("schema_version") == MATPHYS_BACKEND_VERSION,
        "MatPhys gate schema changed",
    )
    validation_start, validation_stop = _frame_range(
        gate.get("validation_frame_range_half_open"),
        name="validation_frame_range_half_open",
    )
    future_start = _positive_integer(
        gate.get("future_frame_start"), name="future_frame_start"
    )
    _require(
        validation_stop <= future_start,
        "MatPhys validation interval crosses the future boundary",
    )
    _require(
        gate.get("future_outcomes_opened") is False,
        "MatPhys gate was built after future outcomes opened",
    )
    proposal_id = sha256_digest(gate.get("proposal_id"), name="proposal_id")
    target_object_id = nonempty_string(
        gate.get("target_object_id"), name="target_object_id"
    )
    if proposal is not None:
        _require(
            proposal_id == proposal.get("proposal_id"),
            "MatPhys gate names another proposal",
        )
        _require(
            target_object_id == proposal.get("target_object_id"),
            "MatPhys gate names another target object",
        )
        _require(
            validation_start
            >= int(proposal["target_evidence_end_frame_exclusive"]),
            "MatPhys validation overlaps proposal fitting evidence",
        )
    artifacts = source_artifact_mapping(
        _mapping(gate.get("source_artifacts"), name="source_artifacts"),
        name="source_artifacts",
    )
    archive_digests = _mapping(
        gate.get("evaluated_archive_sha256s"),
        name="evaluated_archive_sha256s",
    )
    require_exact_fields(
        archive_digests,
        expected=_EVALUATED_ARCHIVE_FIELDS,
        name="evaluated_archive_sha256s",
    )
    identity: dict[str, Any] = {
        "schema": MATPHYS_BACKEND_GATE_SCHEMA,
        "schema_version": MATPHYS_BACKEND_VERSION,
        "proposal_id": proposal_id,
        "target_object_id": target_object_id,
        "case_id": nonempty_string(gate.get("case_id"), name="case_id"),
        "validation_frame_range_half_open": [validation_start, validation_stop],
        "future_frame_start": future_start,
        "future_outcomes_opened": False,
        "evaluated_archive_sha256s": {
            name: sha256_digest(
                archive_digests.get(name),
                name=f"evaluated_archive_sha256s.{name}",
            )
            for name in sorted(_EVALUATED_ARCHIVE_FIELDS)
        },
        "incumbent_metrics": _normalized_metrics(
            gate.get("incumbent_metrics"), name="incumbent_metrics"
        ),
        "candidate_metrics": _normalized_metrics(
            gate.get("candidate_metrics"), name="candidate_metrics"
        ),
        "minimum_relative_improvement": _finite_number(
            gate.get("minimum_relative_improvement"),
            name="minimum_relative_improvement",
            minimum=0.0,
        ),
        "maximum_metric_regression": _finite_number(
            gate.get("maximum_metric_regression"),
            name="maximum_metric_regression",
            minimum=0.0,
        ),
        "maximum_identity_replay_rmse_m": _finite_number(
            gate.get("maximum_identity_replay_rmse_m"),
            name="maximum_identity_replay_rmse_m",
            minimum=0.0,
            minimum_exclusive=True,
        ),
        "source_artifacts": dict(artifacts),
    }
    normalized = {**identity, "gate_id": content_id(identity)}
    _require(
        gate.get("gate_id") == normalized["gate_id"],
        "MatPhys gate content identity changed",
    )
    return cast(dict[str, Any], plain_json(normalized))


def _load_physical_archive(
    path: str | Path, *, name: str
) -> tuple[Path, dict[str, FloatArray]]:
    source = _ordinary_file(path, name=name)
    try:
        with np.load(source, allow_pickle=False) as stored:
            _require(
                set(stored.files) == set(PHYSICAL_ARRAY_NAMES),
                f"{name} array roster changed",
            )
            arrays = {key: np.asarray(stored[key]).copy() for key in stored.files}
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load {name}") from error
    prediction = arrays["prediction_m"]
    persistence = arrays["persistence_m"]
    driven = arrays["driven_readout_m"]
    zero = arrays["zero_action_readout_m"]
    support = arrays["action_support"]
    frame_zero = arrays["frame_zero_points_m"]
    _require(
        prediction.ndim == 3
        and prediction.shape[0] >= 2
        and prediction.shape[1] >= 1
        and prediction.shape[2] == 3,
        f"{name} prediction must have shape (T,N,3)",
    )
    _require(
        persistence.shape == prediction.shape
        and driven.shape == prediction.shape
        and zero.shape == prediction.shape
        and frame_zero.shape == prediction.shape[1:]
        and support.shape == (prediction.shape[1],),
        f"{name} physical array shapes changed",
    )
    _require(
        all(np.issubdtype(value.dtype, np.floating) for value in arrays.values()),
        f"{name} arrays must be floating point",
    )
    _require(
        all(np.all(np.isfinite(value)) for value in arrays.values()),
        f"{name} contains non-finite values",
    )
    _require(
        np.all((support >= 0.0) & (support <= 1.0)),
        f"{name} action support is invalid",
    )
    _require(
        np.array_equal(
            persistence,
            np.repeat(frame_zero[None], prediction.shape[0], axis=0),
        ),
        f"{name} persistence is not exact",
    )
    _require(
        np.array_equal(prediction[0], frame_zero)
        and np.array_equal(driven[0], frame_zero)
        and np.array_equal(zero[0], frame_zero),
        f"{name} changes frame-zero identity",
    )
    return source, arrays


def _require_compatible_archives(
    incumbent: Mapping[str, FloatArray],
    candidate: Mapping[str, FloatArray],
    identity: Mapping[str, FloatArray],
) -> None:
    for name in sorted(PHYSICAL_ARRAY_NAMES):
        _require(
            incumbent[name].shape == candidate[name].shape == identity[name].shape,
            f"MatPhys archive shape differs for {name}",
        )
        _require(
            incumbent[name].dtype == candidate[name].dtype == identity[name].dtype,
            f"MatPhys archive dtype differs for {name}",
        )
    for name in ("frame_zero_points_m", "persistence_m", "action_support"):
        _require(
            np.array_equal(incumbent[name], candidate[name])
            and np.array_equal(incumbent[name], identity[name]),
            f"MatPhys proposal changed the {name} contract",
        )


def _copy_no_overwrite(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(destination) from None
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_no_overwrite(
    value: str,
    destination: Path,
    *,
    encoding: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(destination) from None
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def materialize_matphys_backend(
    *,
    proposal_manifest_path: str | Path,
    gate_manifest_path: str | Path,
    incumbent_archive_path: str | Path,
    candidate_archive_path: str | Path,
    identity_replay_archive_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Select and publish one MatPhys/Warp proposal with exact fallback."""

    proposal_path = _ordinary_file(
        proposal_manifest_path, name="MatPhys proposal manifest"
    )
    proposal = validate_matphys_backend_proposal(
        load_strict_json_object(proposal_path, label="MatPhys proposal manifest"),
        verify_files=True,
    )
    gate_path = _ordinary_file(gate_manifest_path, name="MatPhys gate manifest")
    gate = validate_matphys_backend_gate(
        load_strict_json_object(gate_path, label="MatPhys gate manifest"),
        proposal=proposal,
    )
    incumbent_path, incumbent = _load_physical_archive(
        incumbent_archive_path, name="incumbent physical archive"
    )
    candidate_path, candidate = _load_physical_archive(
        candidate_archive_path, name="MatPhys candidate physical archive"
    )
    identity_path, identity = _load_physical_archive(
        identity_replay_archive_path, name="MatPhys identity replay archive"
    )
    evaluated_digests = gate["evaluated_archive_sha256s"]
    _require(
        _file_sha256(incumbent_path) == evaluated_digests["incumbent"]
        and _file_sha256(candidate_path)
        == evaluated_digests["matphys_warp_proposal"]
        and _file_sha256(identity_path)
        == evaluated_digests["zero_strength_identity_replay"],
        "MatPhys gate was evaluated on different physical archive bytes",
    )
    _require_compatible_archives(incumbent, candidate, identity)
    identity_rmse_m = trajectory_coordinate_rmse(
        incumbent["prediction_m"], identity["prediction_m"]
    )
    stability_eligible = bool(
        identity_rmse_m <= gate["maximum_identity_replay_rmse_m"]
    )
    metrics = {
        "incumbent": cast(Mapping[str, object], gate["incumbent_metrics"]),
        "matphys_warp_proposal": cast(
            Mapping[str, object], gate["candidate_metrics"]
        ),
    }
    selected, scores, decisions = choose_guarded_backbone_family(
        metrics,
        cast(Mapping[str, object], gate["incumbent_metrics"]),
        fallback_family="incumbent",
        minimum_relative_improvement=float(gate["minimum_relative_improvement"]),
        maximum_metric_regression=float(gate["maximum_metric_regression"]),
        eligible_families={
            "incumbent": True,
            "matphys_warp_proposal": stability_eligible,
        },
    )
    selected_path = (
        candidate_path if selected == "matphys_warp_proposal" else incumbent_path
    )
    output = Path(output_dir).absolute()
    _require(not os.path.lexists(output), "MatPhys backend output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    )
    try:
        archive = temporary / SELECTED_ARCHIVE_FILENAME
        _copy_no_overwrite(selected_path, archive)
        _require(
            _file_sha256(archive) == _file_sha256(selected_path),
            "selected MatPhys backend archive changed while publishing",
        )
        provenance = temporary / "provenance"
        proposal_copy = provenance / PROPOSAL_FILENAME
        gate_copy = provenance / GATE_FILENAME
        _copy_no_overwrite(proposal_path, proposal_copy)
        _copy_no_overwrite(gate_path, gate_copy)
        candidate_accepted = selected == "matphys_warp_proposal"
        exact_fallback = (
            candidate_accepted is False
            and _file_sha256(archive) == _file_sha256(incumbent_path)
        )
        _require(
            candidate_accepted or exact_fallback,
            "rejected MatPhys proposal did not preserve exact incumbent bytes",
        )
        artifact_identity: dict[str, Any] = {
            "schema": MATPHYS_BACKEND_ARTIFACT_SCHEMA,
            "schema_version": MATPHYS_BACKEND_VERSION,
            "backend_kind": MATPHYS_BACKEND_KIND,
            "proposal_id": proposal["proposal_id"],
            "gate_id": gate["gate_id"],
            "case_id": gate["case_id"],
            "selected_backend": selected,
            "candidate_accepted": candidate_accepted,
            "selection": {
                "identity_replay_coordinate_rmse_m": identity_rmse_m,
                "maximum_identity_replay_rmse_m": gate[
                    "maximum_identity_replay_rmse_m"
                ],
                "identity_replay_stable": stability_eligible,
                "scores": scores,
                "decisions": decisions,
                "minimum_relative_improvement": gate[
                    "minimum_relative_improvement"
                ],
                "maximum_metric_regression": gate["maximum_metric_regression"],
            },
            "inputs": {
                "proposal_manifest": {
                    "path": str(proposal_copy.relative_to(temporary)),
                    "sha256": _file_sha256(proposal_copy),
                },
                "gate_manifest": {
                    "path": str(gate_copy.relative_to(temporary)),
                    "sha256": _file_sha256(gate_copy),
                },
                "incumbent_archive": _file_identity(
                    incumbent_path, name="incumbent physical archive"
                ),
                "candidate_archive": _file_identity(
                    candidate_path, name="MatPhys candidate physical archive"
                ),
                "identity_replay_archive": _file_identity(
                    identity_path, name="MatPhys identity replay archive"
                ),
            },
            "output": {
                "path": SELECTED_ARCHIVE_FILENAME,
                "sha256": _file_sha256(archive),
                "byte_count": archive.stat().st_size,
                "source_archive_sha256": _file_sha256(selected_path),
                "byte_exact_source_copy": True,
                "exact_incumbent_fallback_verified": exact_fallback,
            },
            "information_boundary": dict(_INFORMATION_BOUNDARY),
            "claim_boundary": MATPHYS_BACKEND_CLAIM_BOUNDARY,
        }
        artifact = {
            **artifact_identity,
            "artifact_id": content_id(artifact_identity),
        }
        artifact_path = temporary / ARTIFACT_FILENAME
        write_atomic_json(artifact, artifact_path, overwrite=False)
        checksums = {
            SELECTED_ARCHIVE_FILENAME: _file_sha256(archive),
            ARTIFACT_FILENAME: _file_sha256(artifact_path),
            f"provenance/{PROPOSAL_FILENAME}": _file_sha256(proposal_copy),
            f"provenance/{GATE_FILENAME}": _file_sha256(gate_copy),
        }
        _write_text_no_overwrite(
            "".join(
                f"{digest}  {name}\n" for name, digest in sorted(checksums.items())
            ),
            temporary / CHECKSUMS_FILENAME,
            encoding="ascii",
        )
        validate_matphys_backend_artifact(temporary)
        _require(not os.path.lexists(output), "MatPhys backend output already exists")
        os.rename(temporary, output)
        _fsync_directory(output.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_matphys_backend_artifact(output)


def validate_matphys_backend_artifact(output_dir: str | Path) -> dict[str, Any]:
    """Revalidate one materialized MatPhys backend without large source files."""

    raw_output = Path(output_dir).absolute()
    _require(
        raw_output.is_dir()
        and not raw_output.is_symlink()
        and not any(parent.is_symlink() for parent in raw_output.parents),
        "backend root is invalid",
    )
    output = raw_output.resolve(strict=True)
    _require(
        {entry.name for entry in output.iterdir()} == _ROOT_ROSTER,
        "MatPhys backend root roster changed",
    )
    provenance_root = output / "provenance"
    _require(
        provenance_root.is_dir()
        and not provenance_root.is_symlink()
        and {entry.name for entry in provenance_root.iterdir()}
        == _PROVENANCE_ROSTER,
        "MatPhys backend provenance roster changed",
    )
    artifact_path = _ordinary_file(
        output / ARTIFACT_FILENAME, name="MatPhys backend artifact"
    )
    artifact = _mapping(
        load_strict_json_object(artifact_path, label="MatPhys backend artifact"),
        name="MatPhys backend artifact",
    )
    require_exact_fields(
        artifact, expected=_ARTIFACT_FIELDS, name="MatPhys backend artifact"
    )
    _require(
        artifact.get("schema") == MATPHYS_BACKEND_ARTIFACT_SCHEMA
        and artifact.get("schema_version") == MATPHYS_BACKEND_VERSION
        and artifact.get("backend_kind") == MATPHYS_BACKEND_KIND,
        "MatPhys backend artifact schema changed",
    )
    identity = {key: value for key, value in artifact.items() if key != "artifact_id"}
    _require(
        artifact.get("artifact_id") == content_id(identity),
        "MatPhys backend artifact content identity changed",
    )
    _require(
        artifact.get("information_boundary") == _INFORMATION_BOUNDARY
        and artifact.get("claim_boundary") == MATPHYS_BACKEND_CLAIM_BOUNDARY,
        "MatPhys backend artifact boundary changed",
    )
    inputs = _mapping(artifact.get("inputs"), name="inputs")
    require_exact_fields(
        inputs, expected=_ARTIFACT_INPUT_FIELDS, name="artifact inputs"
    )
    proposal_record = _mapping(
        inputs.get("proposal_manifest"),
        name="proposal_manifest",
    )
    gate_record = _mapping(
        inputs.get("gate_manifest"),
        name="gate_manifest",
    )
    require_exact_fields(
        proposal_record, expected=_FILE_FIELDS, name="proposal_manifest"
    )
    require_exact_fields(gate_record, expected=_FILE_FIELDS, name="gate_manifest")
    proposal_relative = canonical_relative_posix_path(
        proposal_record.get("path"), name="proposal_manifest.path"
    )
    gate_relative = canonical_relative_posix_path(
        gate_record.get("path"), name="gate_manifest.path"
    )
    _require(
        proposal_relative == f"provenance/{PROPOSAL_FILENAME}"
        and gate_relative == f"provenance/{GATE_FILENAME}",
        "MatPhys compact provenance paths changed",
    )
    proposal_path = _ordinary_file(
        output / proposal_relative, name="copied MatPhys proposal"
    )
    gate_path = _ordinary_file(output / gate_relative, name="copied MatPhys gate")
    _require(
        proposal_path.is_file()
        and _file_sha256(proposal_path) == proposal_record.get("sha256"),
        "copied MatPhys proposal changed",
    )
    _require(
        gate_path.is_file() and _file_sha256(gate_path) == gate_record.get("sha256"),
        "copied MatPhys gate changed",
    )
    proposal = validate_matphys_backend_proposal(
        load_strict_json_object(proposal_path, label="copied MatPhys proposal"),
        verify_files=False,
    )
    gate = validate_matphys_backend_gate(
        load_strict_json_object(gate_path, label="copied MatPhys gate"),
        proposal=proposal,
    )
    _require(
        proposal["proposal_id"] == artifact.get("proposal_id")
        and gate["gate_id"] == artifact.get("gate_id")
        and gate["case_id"] == artifact.get("case_id"),
        "MatPhys backend lineage changed",
    )
    input_identities = {
        "incumbent": _mapping(
            inputs.get("incumbent_archive"), name="incumbent_archive"
        ),
        "matphys_warp_proposal": _mapping(
            inputs.get("candidate_archive"), name="candidate_archive"
        ),
        "zero_strength_identity_replay": _mapping(
            inputs.get("identity_replay_archive"), name="identity_replay_archive"
        ),
    }
    for name, record in input_identities.items():
        require_exact_fields(record, expected=_FILE_FIELDS, name=name)
        nonempty_string(record.get("path"), name=f"{name}.path")
        _require(
            sha256_digest(record.get("sha256"), name=f"{name}.sha256")
            == gate["evaluated_archive_sha256s"][name],
            f"{name} differs from the evaluated archive identity",
        )
    selected = nonempty_string(
        artifact.get("selected_backend"), name="selected_backend"
    )
    _require(
        selected in {"incumbent", "matphys_warp_proposal"},
        "selected MatPhys backend changed",
    )
    accepted = artifact.get("candidate_accepted")
    _require(
        type(accepted) is bool
        and accepted == (selected == "matphys_warp_proposal"),
        "MatPhys backend acceptance disagrees with selection",
    )
    selection = _mapping(artifact.get("selection"), name="selection")
    require_exact_fields(selection, expected=_SELECTION_FIELDS, name="selection")
    identity_rmse_m = _finite_number(
        selection.get("identity_replay_coordinate_rmse_m"),
        name="identity_replay_coordinate_rmse_m",
        minimum=0.0,
    )
    _require(
        selection.get("maximum_identity_replay_rmse_m")
        == gate["maximum_identity_replay_rmse_m"]
        and selection.get("minimum_relative_improvement")
        == gate["minimum_relative_improvement"]
        and selection.get("maximum_metric_regression")
        == gate["maximum_metric_regression"],
        "MatPhys selection thresholds differ from the gate",
    )
    identity_stable = bool(
        identity_rmse_m <= gate["maximum_identity_replay_rmse_m"]
    )
    expected_selected, expected_scores, expected_decisions = (
        choose_guarded_backbone_family(
            {
                "incumbent": cast(
                    Mapping[str, object], gate["incumbent_metrics"]
                ),
                "matphys_warp_proposal": cast(
                    Mapping[str, object], gate["candidate_metrics"]
                ),
            },
            cast(Mapping[str, object], gate["incumbent_metrics"]),
            fallback_family="incumbent",
            minimum_relative_improvement=float(
                gate["minimum_relative_improvement"]
            ),
            maximum_metric_regression=float(gate["maximum_metric_regression"]),
            eligible_families={
                "incumbent": True,
                "matphys_warp_proposal": identity_stable,
            },
        )
    )
    _require(
        selection.get("identity_replay_stable") is identity_stable
        and selection.get("scores") == expected_scores
        and selection.get("decisions") == expected_decisions
        and selected == expected_selected,
        "MatPhys backend guard decision changed",
    )
    output_record = _mapping(artifact.get("output"), name="output")
    require_exact_fields(
        output_record, expected=_ARTIFACT_OUTPUT_FIELDS, name="output"
    )
    _require(
        output_record.get("byte_exact_source_copy") is True,
        "MatPhys backend output is not a byte-exact source copy",
    )
    archive_relative = canonical_relative_posix_path(
        output_record.get("path"), name="output.path"
    )
    _require(
        archive_relative == SELECTED_ARCHIVE_FILENAME,
        "selected MatPhys output path changed",
    )
    archive = _ordinary_file(
        output / archive_relative, name="selected MatPhys backend archive"
    )
    archive_digest = sha256_digest(
        output_record.get("sha256"), name="output.sha256"
    )
    source_digest = sha256_digest(
        output_record.get("source_archive_sha256"),
        name="output.source_archive_sha256",
    )
    byte_count = _positive_integer(
        output_record.get("byte_count"), name="output.byte_count"
    )
    _require(
        _file_sha256(archive) == archive_digest
        and archive.stat().st_size == byte_count,
        "selected MatPhys backend archive changed",
    )
    _load_physical_archive(archive, name="selected MatPhys backend archive")
    expected_checksums = {
        SELECTED_ARCHIVE_FILENAME: _file_sha256(archive),
        ARTIFACT_FILENAME: _file_sha256(artifact_path),
        f"provenance/{PROPOSAL_FILENAME}": _file_sha256(proposal_path),
        f"provenance/{GATE_FILENAME}": _file_sha256(gate_path),
    }
    checksums_path = _ordinary_file(
        output / CHECKSUMS_FILENAME, name="MatPhys checksum manifest"
    )
    _require(
        checksums_path.is_file()
        and checksums_path.read_text(encoding="ascii")
        == "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(expected_checksums.items())
        ),
        "MatPhys backend checksum manifest changed",
    )
    if not accepted:
        _require(
            output_record.get("exact_incumbent_fallback_verified") is True,
            "rejected MatPhys proposal lacks exact fallback verification",
        )
        _require(
            archive_digest == input_identities["incumbent"].get("sha256"),
            "rejected MatPhys output differs from incumbent bytes",
        )
    else:
        _require(
            output_record.get("exact_incumbent_fallback_verified") is False
            and archive_digest
            == input_identities["matphys_warp_proposal"].get("sha256"),
            "accepted MatPhys output differs from candidate bytes",
        )
    _require(
        source_digest == archive_digest,
        "selected MatPhys source archive identity changed",
    )
    return cast(dict[str, Any], plain_json(artifact))


__all__ = [
    "ARTIFACT_FILENAME",
    "MATPHYS_BACKEND_ARTIFACT_SCHEMA",
    "MATPHYS_BACKEND_CLAIM_BOUNDARY",
    "MATPHYS_BACKEND_GATE_SCHEMA",
    "MATPHYS_BACKEND_KIND",
    "MATPHYS_BACKEND_PROPOSAL_SCHEMA",
    "MATPHYS_BACKEND_VERSION",
    "SELECTED_ARCHIVE_FILENAME",
    "build_matphys_backend_gate",
    "build_matphys_backend_proposal",
    "materialize_matphys_backend",
    "validate_matphys_backend_artifact",
    "validate_matphys_backend_gate",
    "validate_matphys_backend_proposal",
]
