"""Materialize calibration-only Deform360 visual/contact information states.

The public Deform360 tactile arrays are unitless contact responses, not forces
or Cartesian positions.  This module uses them only to weight the known taxel
geometry from the synchronized robot trajectory.  One sensor/contact episode
becomes one correlated family of displacement-equivalent rows; raw taxels are
never counted as independent observations.

The posterior path accepts only a claim-bearing Prob4D observation belief.  It
forms the visual innovation once, runs the existing robust gauge-aware solver,
and emits the nuisance-marginalized state precisions consumed by the locked
calibration observability report.  Confirmation payloads and future outcomes
are outside this module's interface.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._canonical_contracts import (
    canonical_string_tuple,
    frozen_finite_json_mapping,
    genuine_integer,
    immutable_array,
    literal_lower_hex,
    plain_json,
)
from .claim_bearing_prob4d import (
    build_claim_bearing_gauge_aware_batch_from_artifacts,
)
from .deform360_contact_anchor import (
    Deform360ContactAnchorV1,
    attach_deform360_contact_anchor,
    save_deform360_contact_anchor,
)
from .gauge_aware_belief import (
    GaugeAwareBeliefConfig,
    GaugeAwareBeliefResult,
    update_gauge_aware_belief,
)
from .observation_belief import ObservationBeliefV1
from .physical_linearization import PhysicalLinearizationV1

DEFORM360_CALIBRATION_FACTOR_SCHEMA = (
    "bayesian-phystwin.deform360-calibration-factor-materialization"
)
DEFORM360_CALIBRATION_FACTOR_VERSION = 1
DEFORM360_CALIBRATION_FACTOR_SEMANTICS = (
    "claim-bearing-visual-reference-plus-grouped-kinematic-contact-v1"
)
DEFORM360_KINEMATIC_CONTACT_SEMANTICS = (
    "positive-taxel-weighted-world-patch-with-shared-sensor-bias-v1"
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _array_record(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _finite_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    _require(array.ndim == ndim, f"{name} must have {ndim} dimensions")
    _require(np.all(np.isfinite(array)), f"{name} contains non-finite values")
    return array


def _probability_vector(value: object, count: int, *, name: str) -> np.ndarray:
    array = _finite_array(value, name=name, ndim=1)
    _require(array.shape == (count,), f"{name} must have shape ({count},)")
    _require(
        np.all((array >= 0.0) & (array <= 1.0)),
        f"{name} must lie in [0, 1]",
    )
    return array


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    raw = np.asarray(value)
    _require(
        raw.shape == () and raw.dtype.kind in "iuf",
        f"{name} must be finite and positive",
    )
    result = float(raw.item())
    _require(
        np.isfinite(result) and result > 0.0,
        f"{name} must be finite and positive",
    )
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _spd_inverse(value: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    _require(
        matrix.ndim == 2
        and matrix.shape[0] == matrix.shape[1]
        and len(matrix) > 0,
        f"{name} must be a nonempty square matrix",
    )
    matrix = 0.5 * (matrix + matrix.T)
    try:
        cholesky = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    identity = np.eye(len(matrix), dtype=np.float64)
    inverse = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, identity))
    return 0.5 * (inverse + inverse.T)


@dataclass(frozen=True)
class Deform360KinematicContactConfig:
    """Frozen conversion from public tactile geometry to metric patch rows."""

    localization_floor_m: float = 0.005
    sensor_bias_prior_std_m: float = 0.010
    minimum_unique_active_taxels: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "localization_floor_m",
            _positive_float(
                self.localization_floor_m,
                name="localization_floor_m",
            ),
        )
        object.__setattr__(
            self,
            "sensor_bias_prior_std_m",
            _positive_float(
                self.sensor_bias_prior_std_m,
                name="sensor_bias_prior_std_m",
            ),
        )
        object.__setattr__(
            self,
            "minimum_unique_active_taxels",
            genuine_integer(
                self.minimum_unique_active_taxels,
                name="minimum_unique_active_taxels",
                minimum=2,
            ),
        )


DEFAULT_DEFORM360_KINEMATIC_CONTACT_CONFIG = Deform360KinematicContactConfig()
DEFAULT_DEFORM360_CALIBRATION_BELIEF_CONFIG = GaugeAwareBeliefConfig(
    effective_samples_per_anchor_correlation_group=1.0,
)


def build_deform360_kinematic_contact_anchor(
    *,
    object_id: str,
    observation_case_id: str,
    episode_id: int,
    causal_frame_stop: int,
    frame_ids: object,
    sensor_names: Sequence[str],
    contact_episode_ids: Sequence[str],
    tactile_response: object,
    taxel_world_positions_m: object,
    physical_patch_prediction_m: object,
    state_jacobian: object,
    source_reliability: object,
    source_revision: str,
    source_artifacts: Mapping[str, str],
    config: Deform360KinematicContactConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Deform360ContactAnchorV1:
    """Reduce synchronized public tactile/robot evidence to contact-patch rows.

    Positive unitless responses weight known world-frame taxel positions.  The
    weighted patch scatter is retained as covariance without dividing by the
    number of taxels, so duplicating a pixel/taxel block cannot increase
    confidence.  Rows from the same sensor/contact episode share one likelihood
    group, and every sensor receives a three-dimensional common-bias nuisance.
    """

    cfg = config or DEFAULT_DEFORM360_KINEMATIC_CONTACT_CONFIG
    _require(
        isinstance(observation_case_id, str) and bool(observation_case_id),
        "observation_case_id must be nonempty",
    )
    frames_raw = np.asarray(frame_ids)
    _require(
        frames_raw.ndim == 1
        and frames_raw.dtype.kind in "iu"
        and len(frames_raw) > 0,
        "frame_ids must be a nonempty integer vector",
    )
    frames = np.asarray(frames_raw, dtype=np.int64)
    count = len(frames)
    sensors = canonical_string_tuple(
        sensor_names,
        name="sensor_names",
        allow_empty=False,
    )
    episodes = canonical_string_tuple(
        contact_episode_ids,
        name="contact_episode_ids",
        allow_empty=False,
    )
    _require(len(sensors) == count, "sensor_names must identify every row")
    _require(
        len(episodes) == count,
        "contact_episode_ids must identify every row",
    )
    _require(
        len(set(zip(frames.tolist(), sensors, strict=True))) == count,
        "contact rows repeat a frame/sensor identity",
    )

    response = _finite_array(tactile_response, name="tactile_response", ndim=2)
    positions = _finite_array(
        taxel_world_positions_m,
        name="taxel_world_positions_m",
        ndim=3,
    )
    prediction = _finite_array(
        physical_patch_prediction_m,
        name="physical_patch_prediction_m",
        ndim=2,
    )
    state = _finite_array(state_jacobian, name="state_jacobian", ndim=3)
    _require(
        response.shape[0] == count and response.shape[1] >= 2,
        "tactile_response must have shape (A, K) with K >= 2",
    )
    _require(
        positions.shape == (count, response.shape[1], 3),
        "taxel_world_positions_m must have shape (A, K, 3)",
    )
    _require(
        prediction.shape == (count, 3),
        "physical_patch_prediction_m must have shape (A, 3)",
    )
    _require(
        state.shape[:2] == (count, 3) and state.shape[2] >= 1,
        "state_jacobian must have shape (A, 3, S) with S >= 1",
    )
    reliability = _probability_vector(
        source_reliability,
        count,
        name="source_reliability",
    )

    patch_centroid: np.ndarray = np.empty((count, 3), dtype=np.float64)
    covariance: np.ndarray = np.empty((count, 3, 3), dtype=np.float64)
    active_counts: np.ndarray = np.empty(count, dtype=np.int64)
    unique_active_counts: np.ndarray = np.empty(count, dtype=np.int64)
    floor = cfg.localization_floor_m**2 * np.eye(3, dtype=np.float64)
    for row in range(count):
        positive = np.maximum(response[row], 0.0)
        active = positive > 0.0
        active_counts[row] = int(np.sum(active))
        active_positions = positions[row, active]
        unique_active_counts[row] = len(np.unique(active_positions, axis=0))
        _require(
            unique_active_counts[row] >= cfg.minimum_unique_active_taxels,
            "contact row has too few unique active taxels",
        )
        weights = positive[active]
        weights = weights / np.sum(weights)
        centroid = np.einsum("k,kc->c", weights, active_positions, optimize=True)
        centered = active_positions - centroid
        scatter = np.einsum(
            "k,ki,kj->ij",
            weights,
            centered,
            centered,
            optimize=True,
        )
        patch_centroid[row] = centroid
        covariance[row] = 0.5 * (scatter + scatter.T) + floor

    sensor_order = tuple(sorted(set(sensors)))
    sensor_position = {name: index for index, name in enumerate(sensor_order)}
    bias: np.ndarray = np.zeros(
        (count, 3, 3 * len(sensor_order)),
        dtype=np.float64,
    )
    for row, sensor in enumerate(sensors):
        start = 3 * sensor_position[sensor]
        bias[row, :, start : start + 3] = np.eye(3)
    bias_prior = np.eye(bias.shape[2], dtype=np.float64)
    bias_prior *= cfg.sensor_bias_prior_std_m**2

    caller_metadata = plain_json(
        frozen_finite_json_mapping(metadata, name="contact materializer metadata")
    )
    if not isinstance(caller_metadata, dict):
        raise TypeError("contact materializer metadata lost its mapping type")
    reserved = {
        "materialization_semantics": DEFORM360_KINEMATIC_CONTACT_SEMANTICS,
        "observation_case_id": observation_case_id,
        "source_tactile_units": "unitless-peak-relative",
        "taxel_geometry_units": "m",
        "positive_response_used_only_as_patch_weight": True,
        "raw_taxels_used_as_independent_rows": False,
        "patch_covariance_divided_by_taxel_count": False,
        "duplicate_taxel_block_increases_confidence": False,
        "source_reliability_depends_on_state_innovation": False,
        "contact_nominal_probability_depends_on_state_innovation": False,
        "correlation_group_semantics": "sensor-contact-episode",
        "active_taxel_threshold": 0.0,
        "minimum_unique_active_taxels": cfg.minimum_unique_active_taxels,
        "localization_floor_m": cfg.localization_floor_m,
        "sensor_bias_prior_std_m": cfg.sensor_bias_prior_std_m,
        "active_taxel_counts": active_counts.tolist(),
        "unique_active_taxel_counts": unique_active_counts.tolist(),
        "sensor_bias_order": list(sensor_order),
    }
    collisions = sorted(set(caller_metadata) & set(reserved))
    _require(
        not collisions,
        f"contact materializer metadata overrides reserved fields: {collisions}",
    )

    return Deform360ContactAnchorV1(
        object_id=object_id,
        episode_id=episode_id,
        causal_frame_stop=causal_frame_stop,
        sensor_names=sensors,
        frame_ids=frames,
        innovation_m=patch_centroid - prediction,
        covariance_m2=covariance,
        state_jacobian=state,
        correlation_group_ids=tuple(
            f"deform360-contact:{sensor}:{contact_episode}"
            for sensor, contact_episode in zip(sensors, episodes, strict=True)
        ),
        source_revision=source_revision,
        source_artifacts=source_artifacts,
        prior_reliability=reliability,
        prior_nominal_probability=np.ones(count, dtype=np.float64),
        composite_weight=np.ones(count, dtype=np.float64),
        bias_jacobian=bias,
        bias_prior_covariance=bias_prior,
        metadata={**caller_metadata, **reserved},
    )


def _state_marginal_precision(
    result: GaugeAwareBeliefResult,
    *,
    state_count: int,
) -> np.ndarray:
    covariance = np.asarray(result.posterior_covariance[:state_count, :state_count])
    return _spd_inverse(covariance, name="state marginal covariance")


@dataclass(frozen=True)
class Deform360CalibrationFactorMaterializationV1:
    """Exact visual-reference and visual-plus-contact state information."""

    object_id: str
    episode_id: int
    causal_frame_stop: int
    observation_artifact_id: str
    linearization_artifact_id: str
    contact_anchor_artifact_id: str
    reference_inference_admissible: bool
    reference_reason: str
    candidate_inference_admissible: bool
    candidate_reason: str
    reference_marginal_precision: np.ndarray
    candidate_marginal_precision: np.ndarray
    physical_query_jacobian: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require(bool(self.object_id), "object_id must be nonempty")
        object.__setattr__(
            self,
            "episode_id",
            genuine_integer(self.episode_id, name="episode_id", minimum=0),
        )
        object.__setattr__(
            self,
            "causal_frame_stop",
            genuine_integer(
                self.causal_frame_stop,
                name="causal_frame_stop",
                minimum=1,
            ),
        )
        for name in (
            "observation_artifact_id",
            "linearization_artifact_id",
            "contact_anchor_artifact_id",
        ):
            literal_lower_hex(getattr(self, name), name=name, lengths={64})
        _require(
            isinstance(self.reference_inference_admissible, bool)
            and isinstance(self.candidate_inference_admissible, bool),
            "inference-admissible fields must be Boolean",
        )
        _require(
            bool(self.reference_reason) and bool(self.candidate_reason),
            "solver reasons must be nonempty",
        )
        reference = _finite_array(
            self.reference_marginal_precision,
            name="reference_marginal_precision",
            ndim=2,
        )
        candidate = _finite_array(
            self.candidate_marginal_precision,
            name="candidate_marginal_precision",
            ndim=2,
        )
        _require(
            reference.shape == candidate.shape
            and reference.shape[0] == reference.shape[1]
            and len(reference) > 0,
            "marginal precision matrices must be matching nonempty squares",
        )
        _spd_inverse(reference, name="reference marginal precision")
        _spd_inverse(candidate, name="candidate marginal precision")
        query = _finite_array(
            self.physical_query_jacobian,
            name="physical_query_jacobian",
            ndim=2,
        )
        _require(
            query.shape[1] == len(reference) and query.shape[0] >= 1,
            "physical_query_jacobian has changed state dimension",
        )
        singular_values = np.linalg.svd(query, compute_uv=False)
        tolerance = (
            max(query.shape)
            * np.finfo(np.float64).eps
            * max(1.0, float(singular_values[0]))
        )
        _require(
            int(np.sum(singular_values > tolerance)) == len(query),
            "physical_query_jacobian rows must be independent",
        )
        object.__setattr__(
            self,
            "reference_marginal_precision",
            immutable_array(reference, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "candidate_marginal_precision",
            immutable_array(candidate, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "physical_query_jacobian",
            immutable_array(query, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="materialization metadata"),
        )

    @property
    def minimum_information_increment_eigenvalue(self) -> float:
        increment = self.candidate_marginal_precision
        increment = increment - self.reference_marginal_precision
        increment = 0.5 * (increment + increment.T)
        return float(np.min(np.linalg.eigvalsh(increment)))

    @property
    def observability_evaluable(self) -> bool:
        scale = max(
            1.0,
            float(np.linalg.norm(self.reference_marginal_precision, ord=2)),
            float(np.linalg.norm(self.candidate_marginal_precision, ord=2)),
        )
        return self.minimum_information_increment_eigenvalue >= -1e-9 * scale

    def _descriptor(self) -> dict[str, object]:
        return {
            "schema": DEFORM360_CALIBRATION_FACTOR_SCHEMA,
            "schema_version": DEFORM360_CALIBRATION_FACTOR_VERSION,
            "semantics": DEFORM360_CALIBRATION_FACTOR_SEMANTICS,
            "object_id": self.object_id,
            "episode_id": self.episode_id,
            "causal_frame_stop": self.causal_frame_stop,
            "observation_artifact_id": self.observation_artifact_id,
            "linearization_artifact_id": self.linearization_artifact_id,
            "contact_anchor_artifact_id": self.contact_anchor_artifact_id,
            "reference_inference_admissible": (
                self.reference_inference_admissible
            ),
            "reference_reason": self.reference_reason,
            "candidate_inference_admissible": (
                self.candidate_inference_admissible
            ),
            "candidate_reason": self.candidate_reason,
            "observability_evaluable": self.observability_evaluable,
            "minimum_information_increment_eigenvalue": (
                self.minimum_information_increment_eigenvalue
            ),
            "metadata": self.metadata,
            "arrays": {
                "reference_marginal_precision": _array_record(
                    self.reference_marginal_precision
                ),
                "candidate_marginal_precision": _array_record(
                    self.candidate_marginal_precision
                ),
                "physical_query_jacobian": _array_record(
                    self.physical_query_jacobian
                ),
            },
        }

    @property
    def materialization_id(self) -> str:
        return hashlib.sha256(_canonical_json(self._descriptor())).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {"materialization_id": self.materialization_id, **self._descriptor()}


def materialize_deform360_calibration_factors(
    observation_belief: ObservationBeliefV1,
    linearization: PhysicalLinearizationV1,
    contact_anchor: Deform360ContactAnchorV1,
    *,
    physical_prediction_xyz_m: object,
    physical_query_jacobian: object,
    state_prior_covariance_m2: object | None = None,
    config: GaugeAwareBeliefConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Deform360CalibrationFactorMaterializationV1:
    """Run the strict visual reference and grouped contact candidate updates."""

    cfg = config or DEFAULT_DEFORM360_CALIBRATION_BELIEF_CONFIG
    anchor_case_id = (contact_anchor.metadata or {}).get("observation_case_id")
    _require(
        anchor_case_id == observation_belief.case_id,
        "contact anchor and visual observation identify different cases",
    )
    _require(
        contact_anchor.causal_frame_stop == observation_belief.causal_frame_stop,
        "contact anchor and visual observation use different causal cutoffs",
    )
    prediction = _finite_array(
        physical_prediction_xyz_m,
        name="physical_prediction_xyz_m",
        ndim=2,
    )
    state_prior = (
        None
        if state_prior_covariance_m2 is None
        else _finite_array(
            state_prior_covariance_m2,
            name="state_prior_covariance_m2",
            ndim=2,
        )
    )
    adapted = build_claim_bearing_gauge_aware_batch_from_artifacts(
        observation_belief,
        linearization,
        physical_prediction_xyz_m=prediction,
        state_prior_covariance_m2=state_prior,
    )
    visual_batch = adapted.batch
    candidate_batch = attach_deform360_contact_anchor(visual_batch, contact_anchor)
    reference = update_gauge_aware_belief(visual_batch, config=cfg)
    candidate = update_gauge_aware_belief(candidate_batch, config=cfg)
    state_count = visual_batch.state_jacobian.shape[2]

    caller_metadata = plain_json(
        frozen_finite_json_mapping(metadata, name="materialization metadata")
    )
    if not isinstance(caller_metadata, dict):
        raise TypeError("materialization metadata lost its mapping type")
    reserved = {
        "calibration_only": True,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
        "prob4d_claim_bearing_provider_v2_required": True,
        "prob4d_claim_bearing_provider_v2_validated": True,
        "visual_innovation_formed_once": True,
        "association_probability_used_as_prior_reliability": False,
        "raw_taxels_used_as_independent_rows": False,
        "contact_effective_samples_per_correlation_group": (
            cfg.effective_samples_per_anchor_correlation_group
        ),
        "belief_config": asdict(cfg),
        "reference_diagnostics": dict(reference.diagnostics),
        "candidate_diagnostics": dict(candidate.diagnostics),
        "claim_boundary": (
            "Calibration-source observability only; this artifact does not "
            "establish target accuracy, confirmation, deployment safety, or "
            "state of the art."
        ),
    }
    collisions = sorted(set(caller_metadata) & set(reserved))
    _require(
        not collisions,
        f"materialization metadata overrides reserved fields: {collisions}",
    )

    return Deform360CalibrationFactorMaterializationV1(
        object_id=contact_anchor.object_id,
        episode_id=contact_anchor.episode_id,
        causal_frame_stop=contact_anchor.causal_frame_stop,
        observation_artifact_id=observation_belief.artifact_id,
        linearization_artifact_id=linearization.artifact_id,
        contact_anchor_artifact_id=contact_anchor.artifact_id,
        reference_inference_admissible=reference.inference_admissible,
        reference_reason=reference.reason,
        candidate_inference_admissible=candidate.inference_admissible,
        candidate_reason=candidate.reason,
        reference_marginal_precision=_state_marginal_precision(
            reference,
            state_count=state_count,
        ),
        candidate_marginal_precision=_state_marginal_precision(
            candidate,
            state_count=state_count,
        ),
        physical_query_jacobian=np.asarray(
            physical_query_jacobian,
            dtype=np.float64,
        ),
        metadata={**caller_metadata, **reserved},
    )


def publish_deform360_calibration_factor_materialization(
    output_dir: str | Path,
    materialization: Deform360CalibrationFactorMaterializationV1,
    contact_anchor: Deform360ContactAnchorV1,
) -> Path:
    """Atomically publish matrices, contact anchor, manifest, and checksums."""

    _require(
        materialization.contact_anchor_artifact_id == contact_anchor.artifact_id,
        "materialization identifies another contact anchor",
    )
    target = Path(output_dir).absolute()
    _require(not target.exists(), f"output directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.partial"
    lock = target.parent / f".{target.name}.publish.lock"
    temporary.mkdir(mode=0o700)
    lock_descriptor: int | None = None
    lock_created = False
    try:
        reference_path = temporary / "reference-marginal-precision.npy"
        candidate_path = temporary / "candidate-marginal-precision.npy"
        query_path = temporary / "physical-query-jacobian.npy"
        anchor_path = temporary / "contact-anchor.npz"
        np.save(
            reference_path,
            materialization.reference_marginal_precision,
            allow_pickle=False,
        )
        np.save(
            candidate_path,
            materialization.candidate_marginal_precision,
            allow_pickle=False,
        )
        np.save(
            query_path,
            materialization.physical_query_jacobian,
            allow_pickle=False,
        )
        save_deform360_contact_anchor(anchor_path, contact_anchor)

        data_files = tuple(sorted(temporary.iterdir(), key=lambda path: path.name))
        files = {path.name: _sha256_file(path) for path in data_files}
        manifest = {
            **materialization.to_record(),
            "files": files,
            "information_boundary": {
                "calibration_payloads_opened": True,
                "confirmation_payloads_opened": False,
                "target_outcomes_used": False,
            },
        }
        manifest_path = temporary / "materialization.json"
        manifest_path.write_text(
            json.dumps(
                plain_json(manifest),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        checksum_files = tuple(
            sorted(temporary.iterdir(), key=lambda path: path.name)
        )
        checksum_path = temporary / "SHA256SUMS"
        checksum_path.write_text(
            "".join(
                f"{_sha256_file(path)}  {path.name}\n" for path in checksum_files
            ),
            encoding="ascii",
        )
        lock_descriptor = os.open(
            lock,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        lock_created = True
        os.write(lock_descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(lock_descriptor)
        lock_descriptor = None
        if os.path.lexists(target):
            raise FileExistsError(target)
        os.rename(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        if lock_created:
            lock.unlink(missing_ok=True)
    return target


__all__ = [
    "DEFAULT_DEFORM360_CALIBRATION_BELIEF_CONFIG",
    "DEFAULT_DEFORM360_KINEMATIC_CONTACT_CONFIG",
    "DEFORM360_CALIBRATION_FACTOR_SCHEMA",
    "DEFORM360_CALIBRATION_FACTOR_SEMANTICS",
    "DEFORM360_CALIBRATION_FACTOR_VERSION",
    "DEFORM360_KINEMATIC_CONTACT_SEMANTICS",
    "Deform360CalibrationFactorMaterializationV1",
    "Deform360KinematicContactConfig",
    "build_deform360_kinematic_contact_anchor",
    "materialize_deform360_calibration_factors",
    "publish_deform360_calibration_factor_materialization",
]
