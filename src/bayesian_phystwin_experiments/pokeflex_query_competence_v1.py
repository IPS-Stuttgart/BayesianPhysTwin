"""Retrospective public PokeFlex competence validation.

The instrument consumes immutable per-take artifacts produced by the causal
PokeFlex checkpoint-registration runner.  Every routing feature is computed
from frame-prefix diagnostics.  Target-mesh errors are used only to fit or
evaluate the risk model after the take split has been fixed.

All 116 public PokeFlex poking takes were exposed by earlier project work.  The
78-take cohort used here is therefore external real-world retrospective
evidence, not a fresh prospective confirmation or an official PokeFlex score.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from bayesian_phystwin._portable_contracts import content_id

SCHEMA: Final = "bayesian-phystwin.pokeflex-query-competence"
SCHEMA_VERSION: Final = 1
PROTOCOL_ID: Final = "pokeflex-query-competence-retrospective-v1"
IMPLEMENTATION_MODULE_PATH: Final = (
    "src/bayesian_phystwin_experiments/pokeflex_query_competence_v1.py"
)
PARENT_PUBLIC78_PROTOCOL_SHA256: Final = (
    "f108baede896f32ee7150efc7dd2fe54fb51bfe374cc5e4e97f4969dca381eec"
)
SPLIT_NAMESPACE: Final = "pokeflex-query-competence-retrospective-v1"
SOURCE_ARTIFACT_ROLE: Final = (
    "previously exposed public action; fixed all18 scale; never prospective evidence"
)
CLAIM_BOUNDARY: Final = (
    "Public physical PokeFlex evidence on previously exposed actions. The "
    "method, split, and routing rule are frozen before this reanalysis, but "
    "the data are not fresh prospective confirmation. The claim is limited "
    "to causal one-frame geometry forecasts for the registered 18-object "
    "cohort; it is not an official PokeFlex score, unseen-object guarantee, "
    "deployment-safety claim, or state-of-the-art claim."
)

HARM_MARGIN_RELATIVE: Final = 0.01
TARGET_HARM_PROBABILITY: Final = 0.10
MINIMUM_OBJECT_BALANCED_COVERAGE: Final = 0.25
MINIMUM_ACCEPTED_OBJECTS: Final = 12
THRESHOLD_GRID: Final = (
    0.01,
    0.025,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.25,
    0.33,
    0.50,
)
LOGISTIC_L2_PENALTY: Final = 1.0
LOGISTIC_MAX_ITERATIONS: Final = 100
LOGISTIC_TOLERANCE: Final = 1e-9
BOOTSTRAP_REPLICATES: Final = 20_000
SOURCE_BOOTSTRAP_SEED: Final = 20260831
VALIDATION_BOOTSTRAP_SEED: Final = 20260901

FEATURE_NAMES: Final = (
    "candidate_disagreement_mm",
    "candidate_motion_ratio_log1p",
    "candidate_scale",
    "prior_motion_mm",
    "update_rms_mm",
    "update_maximum_mm",
    "force_y_abs",
    "force_y_delta_abs",
    "log1p_associated_points",
    "log1p_information_mass",
    "median_robust_weight",
    "downweighted_fraction",
    "log10_assignment_variance_m2",
    "log10_condition_number",
    "camera_bias_rms_mm",
    "correction_prior_motion_cosine",
    "previous_correction_cosine",
    "correction_prior_motion_cosine_missing",
    "previous_correction_cosine_missing",
    "update_accepted",
    "action_supported",
)
PRIMARY_FEATURES: Final = (
    "candidate_disagreement_mm",
    "candidate_motion_ratio_log1p",
)
CONTEXT_FEATURES: Final = FEATURE_NAMES
FORBIDDEN_BOUNDARIES: Final = (
    "claiming fresh or prospective confirmation",
    "claiming an official PokeFlex benchmark score",
    "opening validation artifacts before the source gate passes",
    "retuning features, margin, split, threshold grid, or candidate from validation outcomes",
    "replacing failed or adverse takes",
    "using frame f or later observations to route target frame f",
    "using object identity as a risk feature",
    "touching held-v8, DLO4, DLO5, or Deform360 protected artifacts",
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Mapping[str, object]) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _object_name(take_id: str) -> str:
    object_name, separator, take_number = take_id.rpartition("_T")
    if not separator or not object_name or not take_number.isdigit():
        raise ValueError(f"invalid PokeFlex take id: {take_id}")
    return object_name


def deterministic_split_v1(
    take_ids: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    """Assign one take per object to training and threshold selection."""

    grouped: dict[str, list[str]] = {}
    for take_id in sorted(set(take_ids)):
        grouped.setdefault(_object_name(take_id), []).append(take_id)
    if any(len(values) < 3 for values in grouped.values()):
        raise ValueError("every PokeFlex object needs at least three audit takes")

    def split_key(take_id: str) -> str:
        payload = f"{SPLIT_NAMESPACE}\0{take_id}".encode()
        return hashlib.sha256(payload).hexdigest()

    result: dict[str, list[str]] = {
        "risk_train": [],
        "threshold_select": [],
        "validation": [],
    }
    for values in grouped.values():
        ordered = sorted(values, key=split_key)
        result["risk_train"].append(ordered[0])
        result["threshold_select"].append(ordered[1])
        result["validation"].extend(ordered[2:])
    return {key: tuple(sorted(values)) for key, values in result.items()}


def load_protocol_v1(path: Path) -> dict[str, Any]:
    raw_protocol = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_protocol, dict):
        raise ValueError("PokeFlex competence protocol must be an object")
    protocol = cast(dict[str, Any], raw_protocol)
    required = {
        "schema",
        "schema_version",
        "protocol_id",
        "protocol_sha256",
        "locked_at_utc",
        "claim_boundary",
        "parent_public78_protocol_sha256",
        "source_artifact_role",
        "implementation",
        "artifact_inventory",
        "split",
        "method",
        "gates",
        "forbidden",
    }
    if set(protocol) != required:
        raise ValueError("PokeFlex competence protocol fields changed")
    if protocol["schema"] != SCHEMA or protocol["schema_version"] != SCHEMA_VERSION:
        raise ValueError("PokeFlex competence protocol schema changed")
    if protocol["protocol_id"] != PROTOCOL_ID:
        raise ValueError("PokeFlex competence protocol id changed")
    try:
        locked_at = datetime.fromisoformat(
            str(protocol["locked_at_utc"]).replace("Z", "+00:00")
        )
    except ValueError as error:
        raise ValueError("PokeFlex protocol lock time changed") from error
    if locked_at.tzinfo is None or locked_at.utcoffset() != timezone.utc.utcoffset(
        None
    ):
        raise ValueError("PokeFlex protocol lock time is not UTC")
    if protocol["claim_boundary"] != CLAIM_BOUNDARY:
        raise ValueError("PokeFlex competence claim boundary changed")
    if protocol["parent_public78_protocol_sha256"] != PARENT_PUBLIC78_PROTOCOL_SHA256:
        raise ValueError("PokeFlex parent protocol changed")
    if protocol["source_artifact_role"] != SOURCE_ARTIFACT_ROLE:
        raise ValueError("PokeFlex source artifact role changed")
    if protocol["forbidden"] != list(FORBIDDEN_BOUNDARIES):
        raise ValueError("PokeFlex forbidden boundaries changed")

    implementation = protocol["implementation"]
    if set(implementation) != {"git_commit", "module_path", "module_sha256"}:
        raise ValueError("PokeFlex implementation binding fields changed")
    git_commit = str(implementation["git_commit"])
    if len(git_commit) != 40 or any(
        character not in "0123456789abcdef" for character in git_commit
    ):
        raise ValueError("PokeFlex implementation commit changed")
    if implementation["module_path"] != IMPLEMENTATION_MODULE_PATH:
        raise ValueError("PokeFlex implementation module path changed")
    if file_sha256(Path(__file__)) != implementation["module_sha256"]:
        raise ValueError("PokeFlex analysis implementation bytes changed")
    without_identity = dict(protocol)
    protocol_sha256 = without_identity.pop("protocol_sha256")
    if protocol_sha256 != _canonical_json_sha256(without_identity):
        raise ValueError("PokeFlex competence protocol identity changed")

    inventory = protocol["artifact_inventory"]
    if not isinstance(inventory, dict) or len(inventory) != 78:
        raise ValueError("PokeFlex artifact inventory must contain 78 takes")
    for take_id, row in inventory.items():
        _object_name(take_id)
        if set(row) != {"filename", "bytes", "sha256"}:
            raise ValueError("PokeFlex artifact inventory fields changed")
        if row["filename"] != f"{take_id}.json":
            raise ValueError("PokeFlex artifact filename changed")
        if int(row["bytes"]) <= 0 or len(str(row["sha256"])) != 64:
            raise ValueError("PokeFlex artifact inventory value changed")

    expected_split = deterministic_split_v1(inventory)
    split = {
        key: tuple(protocol["split"][key])
        for key in ("risk_train", "threshold_select", "validation")
    }
    if split != expected_split:
        raise ValueError("PokeFlex deterministic split changed")
    if tuple(len(split[name]) for name in split) != (18, 18, 42):
        raise ValueError("PokeFlex split counts changed")
    method = protocol["method"]
    if set(method) != {
        "split_namespace",
        "primary_arm",
        "primary_feature_names",
        "context_feature_names",
        "candidate",
        "fallback",
        "harm_margin_relative",
        "threshold_grid",
        "risk_fit",
        "uncertainty",
    }:
        raise ValueError("PokeFlex method fields changed")
    if method["split_namespace"] != SPLIT_NAMESPACE:
        raise ValueError("PokeFlex split namespace changed")
    if method["primary_arm"] != "model_disagreement_only":
        raise ValueError("PokeFlex primary arm changed")
    if method["primary_feature_names"] != list(PRIMARY_FEATURES):
        raise ValueError("PokeFlex primary feature set changed")
    if method["context_feature_names"] != list(CONTEXT_FEATURES):
        raise ValueError("PokeFlex contextual feature set changed")
    if method["threshold_grid"] != list(THRESHOLD_GRID):
        raise ValueError("PokeFlex threshold grid changed")
    if float(method["harm_margin_relative"]) != HARM_MARGIN_RELATIVE:
        raise ValueError("PokeFlex harm margin changed")
    return protocol


@dataclass(frozen=True, slots=True)
class PokeFlexFrameV1:
    take_id: str
    object_name: str
    target_frame: int
    feature_vector: FloatArray
    candidate_available: bool
    fallback_error_mm: float
    candidate_error_mm: float

    def __post_init__(self) -> None:
        features = np.asarray(self.feature_vector, dtype=np.float64)
        if features.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(features)):
            raise ValueError("PokeFlex feature vector changed")
        immutable = features.copy()
        immutable.setflags(write=False)
        object.__setattr__(self, "feature_vector", immutable)
        if self.object_name != _object_name(self.take_id):
            raise ValueError("PokeFlex object identity changed")
        if self.target_frame < 1:
            raise ValueError("PokeFlex target frame changed")
        if not (
            math.isfinite(self.fallback_error_mm)
            and math.isfinite(self.candidate_error_mm)
            and self.fallback_error_mm > 0.0
            and self.candidate_error_mm >= 0.0
        ):
            raise ValueError("PokeFlex target error changed")

    @property
    def normalized_regret(self) -> float:
        return (self.candidate_error_mm - self.fallback_error_mm) / max(
            self.fallback_error_mm, 1e-12
        )

    @property
    def harmful_candidate(self) -> bool:
        return self.normalized_regret > HARM_MARGIN_RELATIVE


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"non-numeric PokeFlex {name}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite PokeFlex {name}")
    return result


def _cosine_features(value: object) -> tuple[float, float]:
    if value is None:
        return 0.0, 1.0
    return max(-1.0, min(1.0, _finite_float(value, name="cosine"))), 0.0


def _feature_vector(update: Mapping[str, object], candidate_scale: float) -> FloatArray:
    correction_cosine, correction_missing = _cosine_features(
        update["correction_prior_motion_cosine"]
    )
    previous_cosine, previous_missing = _cosine_features(
        update["previous_correction_cosine"]
    )
    rms_update_m = max(0.0, _finite_float(update["rms_update_m"], name="rms"))
    maximum_update_m = max(
        0.0, _finite_float(update["maximum_update_m"], name="maximum update")
    )
    prior_motion_m = max(
        0.0, _finite_float(update["prior_motion_rms_m"], name="prior motion")
    )
    ratio = max(
        0.0,
        _finite_float(update["correction_to_prior_motion_ratio"], name="motion ratio"),
    )
    biases = np.asarray(update["camera_biases_m"], dtype=np.float64)
    if biases.shape != (2, 3) or not np.all(np.isfinite(biases)):
        raise ValueError("PokeFlex camera bias shape changed")
    assignment_variance = max(
        1e-15,
        _finite_float(
            update["assignment_variance_m2_mean"], name="assignment variance"
        ),
    )
    condition_number = max(
        1.0, _finite_float(update["condition_number"], name="condition number")
    )
    values = (
        candidate_scale * rms_update_m * 1000.0,
        math.log1p(candidate_scale * ratio),
        candidate_scale,
        prior_motion_m * 1000.0,
        rms_update_m * 1000.0,
        maximum_update_m * 1000.0,
        abs(_finite_float(update["force_y"], name="force")),
        abs(_finite_float(update["force_y_delta"], name="force delta")),
        math.log1p(max(0.0, _finite_float(update["associated_points"], name="points"))),
        math.log1p(
            max(
                0.0,
                _finite_float(update["effective_information_mass"], name="information"),
            )
        ),
        _finite_float(update["median_robust_weight"], name="robust weight"),
        _finite_float(update["downweighted_fraction"], name="downweighting"),
        math.log10(assignment_variance),
        math.log10(condition_number),
        float(np.sqrt(np.mean(biases * biases))) * 1000.0,
        correction_cosine,
        previous_cosine,
        correction_missing,
        previous_missing,
        float(bool(update["accepted"])),
        float(bool(update["action_supported"])),
    )
    return cast(FloatArray, np.asarray(values, dtype=np.float64))


def _candidate_key(candidate_scale: float) -> str:
    return (
        f"checkpoint_action_local_state_relative_0.4_residual_scale_{candidate_scale:g}"
    )


def load_take_artifact_v1(
    path: Path,
    *,
    take_id: str,
    expected_sha256: str,
    expected_bytes: int,
) -> tuple[PokeFlexFrameV1, ...]:
    if path.name != f"{take_id}.json":
        raise ValueError("PokeFlex artifact path changed")
    if path.stat().st_size != expected_bytes or file_sha256(path) != expected_sha256:
        raise ValueError(f"PokeFlex artifact bytes changed: {take_id}")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("future_observation_used") is not False:
        raise ValueError("PokeFlex artifact used future observations")
    if artifact.get("retrospective_prediction_role") != (
        "previously exposed public action; fixed all18 scale; never prospective evidence"
    ):
        raise ValueError("PokeFlex retrospective role changed")
    if artifact.get("public_transfer_protocol_sha256") != (
        PARENT_PUBLIC78_PROTOCOL_SHA256
    ):
        raise ValueError("PokeFlex artifact parent protocol changed")
    if artifact.get("take", {}).get("id") != take_id:
        raise ValueError("PokeFlex artifact take identity changed")
    candidate_scale = _finite_float(
        artifact.get("candidate_effective_scale"), name="candidate scale"
    )
    if candidate_scale <= 0.0:
        raise ValueError("PokeFlex candidate scale changed")
    targets = artifact.get("targets")
    updates = artifact.get("updates")
    if not isinstance(targets, list) or not isinstance(updates, list):
        raise ValueError("PokeFlex frame records changed")
    update_by_target = {int(row["target_frame"]): row for row in updates}
    if len(update_by_target) != len(updates):
        raise ValueError("PokeFlex update frame identity changed")
    candidate_key = _candidate_key(candidate_scale)
    frames: list[PokeFlexFrameV1] = []
    for target in targets:
        target_frame = int(target["target_frame"])
        if target_frame not in update_by_target:
            raise ValueError("PokeFlex target lacks pre-outcome update")
        update = update_by_target[target_frame]
        fallback_error = _finite_float(
            target["released_checkpoint_CD_UL1_mm"], name="fallback error"
        )
        candidate_error = _finite_float(target[candidate_key], name="candidate error")
        frames.append(
            PokeFlexFrameV1(
                take_id=take_id,
                object_name=_object_name(take_id),
                target_frame=target_frame,
                feature_vector=_feature_vector(update, candidate_scale),
                candidate_available=bool(update["accepted"])
                and bool(update["action_supported"]),
                fallback_error_mm=fallback_error,
                candidate_error_mm=candidate_error,
            )
        )
    if not frames:
        raise ValueError("PokeFlex artifact has no target frames")
    return tuple(frames)


def load_partition_v1(
    protocol: Mapping[str, Any],
    artifact_root: Path,
    partition: str,
) -> tuple[PokeFlexFrameV1, ...]:
    if partition not in {"risk_train", "threshold_select", "validation"}:
        raise ValueError("unknown PokeFlex partition")
    frames: list[PokeFlexFrameV1] = []
    inventory = protocol["artifact_inventory"]
    for take_id in protocol["split"][partition]:
        row = inventory[take_id]
        frames.extend(
            load_take_artifact_v1(
                artifact_root / row["filename"],
                take_id=take_id,
                expected_sha256=row["sha256"],
                expected_bytes=int(row["bytes"]),
            )
        )
    return tuple(frames)


@dataclass(frozen=True, slots=True)
class FrozenPhysicalRiskModelV1:
    model_name: str
    selected_feature_names: tuple[str, ...]
    feature_center: FloatArray
    feature_scale: FloatArray
    coefficients: FloatArray
    l2_penalty: float
    iteration_count: int
    converged: bool
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        expected_features = {
            "model_disagreement_only": PRIMARY_FEATURES,
            "physical_context": CONTEXT_FEATURES,
        }
        if expected_features.get(self.model_name) != self.selected_feature_names:
            raise ValueError("PokeFlex risk feature set changed")
        if self.l2_penalty != LOGISTIC_L2_PENALTY:
            raise ValueError("PokeFlex risk penalty changed")
        if (
            not self.converged
            or not 1 <= self.iteration_count <= LOGISTIC_MAX_ITERATIONS
        ):
            raise ValueError("PokeFlex risk convergence record changed")
        count = len(self.selected_feature_names)
        center = np.array(self.feature_center, dtype=np.float64, copy=True)
        scale = np.array(self.feature_scale, dtype=np.float64, copy=True)
        coefficients = np.array(self.coefficients, dtype=np.float64, copy=True)
        if center.shape != (count,) or scale.shape != (count,):
            raise ValueError("PokeFlex risk normalization shape changed")
        if coefficients.shape != (count + 1,):
            raise ValueError("PokeFlex risk coefficient shape changed")
        if not (
            np.all(np.isfinite(center))
            and np.all(np.isfinite(scale))
            and np.all(scale > 0.0)
            and np.all(np.isfinite(coefficients))
        ):
            raise ValueError("PokeFlex risk model contains invalid values")
        for value in (center, scale, coefficients):
            value.setflags(write=False)
        object.__setattr__(self, "feature_center", center)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        expected = content_id(self.descriptor())
        if self.artifact_id is not None and self.artifact_id != expected:
            raise ValueError("PokeFlex risk model identity changed")
        object.__setattr__(self, "artifact_id", expected)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": "bayesian-phystwin.pokeflex-physical-risk-model",
            "schema_version": 1,
            "model_name": self.model_name,
            "selected_feature_names": list(self.selected_feature_names),
            "feature_center": self.feature_center.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "l2_penalty": self.l2_penalty,
            "iteration_count": self.iteration_count,
            "converged": self.converged,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> FrozenPhysicalRiskModelV1:
        required = {
            "schema",
            "schema_version",
            "model_name",
            "selected_feature_names",
            "feature_center",
            "feature_scale",
            "coefficients",
            "l2_penalty",
            "iteration_count",
            "converged",
            "artifact_id",
        }
        if set(record) != required:
            raise ValueError("PokeFlex risk model record fields changed")
        if (
            record["schema"] != "bayesian-phystwin.pokeflex-physical-risk-model"
            or record["schema_version"] != 1
            or record["converged"] is not True
        ):
            raise ValueError("PokeFlex risk model record schema changed")
        return cls(
            model_name=str(record["model_name"]),
            selected_feature_names=tuple(record["selected_feature_names"]),
            feature_center=np.asarray(record["feature_center"], dtype=np.float64),
            feature_scale=np.asarray(record["feature_scale"], dtype=np.float64),
            coefficients=np.asarray(record["coefficients"], dtype=np.float64),
            l2_penalty=float(record["l2_penalty"]),
            iteration_count=int(record["iteration_count"]),
            converged=True,
            artifact_id=str(record["artifact_id"]),
        )

    def score(self, feature_vector: FloatArray) -> float:
        positions = [FEATURE_NAMES.index(name) for name in self.selected_feature_names]
        selected = np.asarray(feature_vector, dtype=np.float64)[positions]
        standardized = (selected - self.feature_center) / self.feature_scale
        linear = float(self.coefficients[0] + standardized @ self.coefficients[1:])
        return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, linear)))))


def fit_risk_model_v1(
    frames: Sequence[PokeFlexFrameV1],
    *,
    model_name: str,
    selected_feature_names: tuple[str, ...],
) -> FrozenPhysicalRiskModelV1:
    eligible = [frame for frame in frames if frame.candidate_available]
    if len(eligible) < 100:
        raise ValueError("too few PokeFlex frames for risk fitting")
    labels = np.asarray([frame.harmful_candidate for frame in eligible], dtype=float)
    if len(np.unique(labels)) != 2:
        raise ValueError("PokeFlex risk labels must contain both classes")
    positions = [FEATURE_NAMES.index(name) for name in selected_feature_names]
    features = np.stack([frame.feature_vector[positions] for frame in eligible])
    center = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale < 1e-8] = 1.0
    design = np.column_stack((np.ones(len(features)), (features - center) / scale))

    object_counts: dict[str, int] = {}
    for frame in eligible:
        object_counts[frame.object_name] = object_counts.get(frame.object_name, 0) + 1
    weights = np.asarray([1.0 / object_counts[frame.object_name] for frame in eligible])
    weights *= len(weights) / float(np.sum(weights))

    coefficients = np.zeros(design.shape[1])
    penalty = LOGISTIC_L2_PENALTY * np.diag(
        np.concatenate((np.zeros(1), np.ones(design.shape[1] - 1)))
    )
    converged = False
    iteration_count = 0
    for iteration in range(LOGISTIC_MAX_ITERATIONS):
        linear = np.clip(design @ coefficients, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-linear))
        variance = probabilities * (1.0 - probabilities)
        gradient = design.T @ (weights * (probabilities - labels))
        gradient += penalty @ coefficients
        hessian = design.T @ ((weights * variance)[:, None] * design)
        hessian += penalty
        try:
            step = np.linalg.solve(hessian + np.eye(hessian.shape[0]) * 1e-9, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        coefficients -= step
        iteration_count = iteration + 1
        if float(np.max(np.abs(step))) < LOGISTIC_TOLERANCE:
            converged = True
            break
    if not converged:
        raise ValueError("PokeFlex risk fit did not converge")
    return FrozenPhysicalRiskModelV1(
        model_name=model_name,
        selected_feature_names=selected_feature_names,
        feature_center=center,
        feature_scale=scale,
        coefficients=coefficients,
        l2_penalty=LOGISTIC_L2_PENALTY,
        iteration_count=iteration_count,
        converged=True,
    )


def _binary_auc(scores: FloatArray, labels: BoolArray) -> float | None:
    positive = scores[labels]
    negative = scores[~labels]
    if len(positive) == 0 or len(negative) == 0:
        return None
    differences = positive[:, None] - negative[None, :]
    return float(np.mean(differences > 0.0) + 0.5 * np.mean(differences == 0.0))


def _object_rows(
    frames: Sequence[PokeFlexFrameV1], accepted: BoolArray
) -> list[dict[str, float | int | str]]:
    harmful = np.asarray([frame.harmful_candidate for frame in frames])
    regrets = np.asarray([frame.normalized_regret for frame in frames])
    rows: list[dict[str, float | int | str]] = []
    for object_name in sorted({frame.object_name for frame in frames}):
        mask = np.asarray([frame.object_name == object_name for frame in frames])
        selected = mask & accepted
        accepted_count = int(np.sum(selected))
        rows.append(
            {
                "object_name": object_name,
                "frame_count": int(np.sum(mask)),
                "accepted_count": accepted_count,
                "coverage": float(np.mean(accepted[mask])),
                "harm_rate": (
                    0.0 if accepted_count == 0 else float(np.mean(harmful[selected]))
                ),
                "policy_regret": float(
                    np.mean(np.where(accepted[mask], regrets[mask], 0.0))
                ),
                "accepted_regret": (
                    0.0 if accepted_count == 0 else float(np.mean(regrets[selected]))
                ),
            }
        )
    return rows


def _cluster_intervals(
    object_rows: Sequence[Mapping[str, float | int | str]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, dict[str, float]]:
    if replicates < 100:
        raise ValueError("too few PokeFlex bootstrap replicates")
    coverage = np.asarray([float(row["coverage"]) for row in object_rows])
    harm = np.asarray([float(row["harm_rate"]) for row in object_rows])
    regret = np.asarray([float(row["policy_regret"]) for row in object_rows])
    accepted = np.asarray([int(row["accepted_count"]) > 0 for row in object_rows])
    generator = np.random.default_rng(seed)
    estimates = {
        "coverage": np.empty(replicates),
        "harm_rate": np.empty(replicates),
        "policy_regret": np.empty(replicates),
    }
    for index in range(replicates):
        sample = generator.integers(0, len(object_rows), size=len(object_rows))
        estimates["coverage"][index] = float(np.mean(coverage[sample]))
        sampled_accepted = accepted[sample]
        estimates["harm_rate"][index] = (
            1.0
            if not np.any(sampled_accepted)
            else float(np.mean(harm[sample][sampled_accepted]))
        )
        estimates["policy_regret"][index] = float(np.mean(regret[sample]))
    result: dict[str, dict[str, float]] = {}
    point_values = {
        "coverage": float(np.mean(coverage)),
        "harm_rate": (1.0 if not np.any(accepted) else float(np.mean(harm[accepted]))),
        "policy_regret": float(np.mean(regret)),
    }
    for name, values in estimates.items():
        lower = float(np.quantile(values, 0.025))
        upper = float(np.quantile(values, 0.975))
        result[name] = {
            "estimate": point_values[name],
            "ci95_lower": lower,
            "ci95_upper": upper,
        }
    return result


def evaluate_policy_v1(
    frames: Sequence[PokeFlexFrameV1],
    scores: FloatArray,
    threshold: float,
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    if scores.shape != (len(frames),) or not np.all(np.isfinite(scores)):
        raise ValueError("PokeFlex risk scores changed")
    available = np.asarray([frame.candidate_available for frame in frames])
    accepted = available & (scores <= threshold)
    harmful = np.asarray([frame.harmful_candidate for frame in frames])
    regrets = np.asarray([frame.normalized_regret for frame in frames])
    rows = _object_rows(frames, accepted)
    intervals = _cluster_intervals(
        rows, seed=bootstrap_seed, replicates=bootstrap_replicates
    )
    accepted_count = int(np.sum(accepted))
    accepted_objects = sum(int(row["accepted_count"]) > 0 for row in rows)
    checks = {
        "minimum_object_balanced_coverage": (
            intervals["coverage"]["estimate"] >= MINIMUM_OBJECT_BALANCED_COVERAGE
        ),
        "minimum_accepted_objects": accepted_objects >= MINIMUM_ACCEPTED_OBJECTS,
        "object_cluster_harm_upper_bound": (
            intervals["harm_rate"]["ci95_upper"] <= TARGET_HARM_PROBABILITY
        ),
        "object_cluster_policy_regret_improves_fallback": (
            intervals["policy_regret"]["ci95_upper"] < 0.0
        ),
        "exact_fallback_identity": True,
    }
    available_scores = scores[available]
    available_labels = harmful[available]
    return {
        "frame_count": len(frames),
        "take_count": len({frame.take_id for frame in frames}),
        "object_count": len(rows),
        "candidate_available_count": int(np.sum(available)),
        "accepted_count": accepted_count,
        "frame_coverage": float(np.mean(accepted)),
        "accepted_object_count": accepted_objects,
        "harmful_accepted_count": int(np.sum(accepted & harmful)),
        "frame_harm_rate": (
            None if accepted_count == 0 else float(np.mean(harmful[accepted]))
        ),
        "accepted_frame_mean_regret": (
            None if accepted_count == 0 else float(np.mean(regrets[accepted]))
        ),
        "object_cluster_intervals": intervals,
        "risk_brier_score": float(
            np.mean((available_scores - available_labels.astype(float)) ** 2)
        ),
        "risk_auc": _binary_auc(available_scores, available_labels),
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
        "objects": rows,
    }


def select_threshold_v1(
    frames: Sequence[PokeFlexFrameV1],
    scores: FloatArray,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    for threshold in THRESHOLD_GRID:
        evaluation = evaluate_policy_v1(
            frames,
            scores,
            threshold,
            bootstrap_seed=SOURCE_BOOTSTRAP_SEED,
            bootstrap_replicates=bootstrap_replicates,
        )
        candidate = {
            "threshold": threshold,
            "frame_coverage": evaluation["frame_coverage"],
            "accepted_count": evaluation["accepted_count"],
            "accepted_object_count": evaluation["accepted_object_count"],
            "object_cluster_intervals": evaluation["object_cluster_intervals"],
            "eligible": evaluation["gate_passed"],
        }
        candidates.append(candidate)
        if candidate["eligible"]:
            selected = candidate
    return {
        "threshold_grid": list(THRESHOLD_GRID),
        "candidate_summaries": candidates,
        "selected_threshold": None if selected is None else selected["threshold"],
        "selection_passed": selected is not None,
    }


def score_frames_v1(
    model: FrozenPhysicalRiskModelV1, frames: Sequence[PokeFlexFrameV1]
) -> FloatArray:
    return cast(
        FloatArray,
        np.asarray([model.score(frame.feature_vector) for frame in frames]),
    )


def run_source_stage_v1(
    protocol: Mapping[str, Any],
    artifact_root: Path,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    risk_frames = load_partition_v1(protocol, artifact_root, "risk_train")
    threshold_frames = load_partition_v1(protocol, artifact_root, "threshold_select")
    models = {
        "model_disagreement_only": fit_risk_model_v1(
            risk_frames,
            model_name="model_disagreement_only",
            selected_feature_names=PRIMARY_FEATURES,
        ),
        "physical_context": fit_risk_model_v1(
            risk_frames,
            model_name="physical_context",
            selected_feature_names=CONTEXT_FEATURES,
        ),
    }
    threshold_results = {
        name: select_threshold_v1(
            threshold_frames,
            score_frames_v1(model, threshold_frames),
            bootstrap_replicates=bootstrap_replicates,
        )
        for name, model in models.items()
    }
    primary = threshold_results["model_disagreement_only"]
    result = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "stage": "source",
        "protocol_sha256": protocol["protocol_sha256"],
        "claim_boundary": CLAIM_BOUNDARY,
        "risk_train_take_count": len(protocol["split"]["risk_train"]),
        "threshold_select_take_count": len(protocol["split"]["threshold_select"]),
        "validation_take_count_opened": 0,
        "risk_models": {name: model.to_record() for name, model in models.items()},
        "threshold_selection": threshold_results,
        "primary_arm": "model_disagreement_only",
        "source_gate_passed": bool(primary["selection_passed"]),
        "validation_authorized": bool(primary["selection_passed"]),
    }
    result["result_sha256"] = _canonical_json_sha256(result)
    return result


def validate_source_result_v1(
    source_result: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    required = {
        "schema",
        "schema_version",
        "stage",
        "protocol_sha256",
        "claim_boundary",
        "risk_train_take_count",
        "threshold_select_take_count",
        "validation_take_count_opened",
        "risk_models",
        "threshold_selection",
        "primary_arm",
        "source_gate_passed",
        "validation_authorized",
        "result_sha256",
    }
    if set(source_result) != required:
        raise ValueError("PokeFlex source result fields changed")
    if source_result["schema"] != SCHEMA or source_result["stage"] != "source":
        raise ValueError("PokeFlex source result schema changed")
    if source_result["schema_version"] != SCHEMA_VERSION:
        raise ValueError("PokeFlex source result version changed")
    if source_result["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError("PokeFlex source result protocol changed")
    without_identity = dict(source_result)
    result_sha256 = without_identity.pop("result_sha256")
    if result_sha256 != _canonical_json_sha256(without_identity):
        raise ValueError("PokeFlex source result identity changed")
    if source_result["validation_take_count_opened"] != 0:
        raise ValueError("PokeFlex source stage opened validation data")
    if source_result["claim_boundary"] != CLAIM_BOUNDARY:
        raise ValueError("PokeFlex source claim boundary changed")
    if (
        source_result["risk_train_take_count"] != 18
        or source_result["threshold_select_take_count"] != 18
    ):
        raise ValueError("PokeFlex source roster counts changed")
    if source_result["primary_arm"] != "model_disagreement_only":
        raise ValueError("PokeFlex primary arm changed")
    if (
        source_result["source_gate_passed"] is not True
        or source_result["validation_authorized"] is not True
    ):
        raise ValueError("PokeFlex source gate did not authorize validation")
    expected_models = {"model_disagreement_only", "physical_context"}
    if set(source_result["risk_models"]) != expected_models:
        raise ValueError("PokeFlex source risk model arms changed")
    if set(source_result["threshold_selection"]) != expected_models:
        raise ValueError("PokeFlex source threshold arms changed")
    for name in sorted(expected_models):
        model = FrozenPhysicalRiskModelV1.from_record(
            source_result["risk_models"][name]
        )
        if model.model_name != name:
            raise ValueError("PokeFlex source model name changed")
        selection = source_result["threshold_selection"][name]
        if set(selection) != {
            "threshold_grid",
            "candidate_summaries",
            "selected_threshold",
            "selection_passed",
        }:
            raise ValueError("PokeFlex threshold selection fields changed")
        if selection["threshold_grid"] != list(THRESHOLD_GRID):
            raise ValueError("PokeFlex source threshold grid changed")
        threshold = selection["selected_threshold"]
        if threshold is not None and float(threshold) not in THRESHOLD_GRID:
            raise ValueError("PokeFlex selected threshold changed")
    if (
        source_result["threshold_selection"]["model_disagreement_only"][
            "selection_passed"
        ]
        is not True
    ):
        raise ValueError("PokeFlex primary source selection did not pass")


def run_validation_stage_v1(
    protocol: Mapping[str, Any],
    source_result: Mapping[str, Any],
    artifact_root: Path,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    validate_source_result_v1(source_result, protocol)
    frames = load_partition_v1(protocol, artifact_root, "validation")
    evaluations: dict[str, dict[str, Any]] = {}
    for name, record in source_result["risk_models"].items():
        model = FrozenPhysicalRiskModelV1.from_record(record)
        threshold = source_result["threshold_selection"][name]["selected_threshold"]
        if threshold is None:
            evaluations[name] = {"source_qualified": False}
            continue
        evaluations[name] = {
            "source_qualified": True,
            "evaluation": evaluate_policy_v1(
                frames,
                score_frames_v1(model, frames),
                float(threshold),
                bootstrap_seed=VALIDATION_BOOTSTRAP_SEED,
                bootstrap_replicates=bootstrap_replicates,
            ),
        }
    primary = evaluations["model_disagreement_only"]
    primary_evaluation = cast(dict[str, object], primary["evaluation"])
    result = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "stage": "validation",
        "protocol_sha256": protocol["protocol_sha256"],
        "source_result_sha256": source_result["result_sha256"],
        "claim_boundary": CLAIM_BOUNDARY,
        "retrospective_data": True,
        "prospective_confirmation": False,
        "validation_take_count": len(protocol["split"]["validation"]),
        "validation_object_count": 18,
        "primary_arm": "model_disagreement_only",
        "evaluations": evaluations,
        "primary_gate_passed": bool(primary_evaluation["gate_passed"]),
    }
    result["result_sha256"] = _canonical_json_sha256(result)
    return result
