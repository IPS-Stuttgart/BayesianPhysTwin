"""V14 adaptive-panel causal direct-depth belief update.

V14 combines the target-free V13 camera carrier with the V12 causal-response
event, admission, and robust belief update. It does not reuse V13's failed
fixed-identity tracker provider: metric depth is re-associated at each tested
causal endpoint.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_causal_response_adaptive_query import (
    ABSTAINED_ARM,
    INFLATED_FALLBACK_ARM,
    STRICT_ARM,
    AdaptiveCausalResponseQuerySchedule,
)
from .deform360_causal_response_admission import (
    CausalResponseAdmissionConfig,
)
from .deform360_causal_response_event import (
    CausalResponseEventConfig,
    CausalResponseEventScan,
    predict_scanned_causal_response,
    scan_causal_response_event,
)
from .deform360_causal_response_update import (
    BASELINE_ARM,
    CANDIDATE_ARM,
    CausalResponseMeasurementConfig,
)
from .deform360_direct_depth_provider import DirectDepthEndpointConfig
from .observation_belief import array_sha256, file_sha256
from .phystwin_online_belief import RecursiveRbfBeliefConfig

CONTRACT = "deform360-causal-response-direct-depth-v14"
ARTIFACT_KIND = "Deform360CausalResponseDirectDepthPredictionV14"
ARCHIVE_FILENAME = "causal_response_direct_depth_v14.npz"
REPORT_FILENAME = "causal_response_direct_depth_v14.json"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _valid_digest(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(payload: Mapping[str, Any], *, key: str) -> str:
    canonical = dict(payload)
    canonical.pop(key, None)
    return hashlib.sha256(
        b"deform360-causal-response-direct-depth-v14\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _expected_support_and_inflation(
    carrier: AdaptiveCausalResponseQuerySchedule,
) -> tuple[int, float]:
    if carrier.arm == STRICT_ARM:
        return carrier.config.strict_minimum_support_per_panel, 1.0
    _require(
        carrier.arm in {INFLATED_FALLBACK_ARM, ABSTAINED_ARM},
        "adaptive carrier arm is unsupported",
    )
    return (
        carrier.config.fallback_minimum_support_per_panel,
        carrier.config.fallback_covariance_inflation,
    )


@dataclass(frozen=True)
class AdaptiveDirectDepthScanV14:
    """One V14 causal scan bound to its adaptive carrier and uncertainty."""

    carrier_artifact_sha256: str
    carrier_arm: str
    selected_camera_ids: tuple[str, ...]
    depth_config: DirectDepthEndpointConfig
    admission_config: CausalResponseAdmissionConfig
    scan: CausalResponseEventScan
    artifact_sha256: str

    def __post_init__(self) -> None:
        _require(
            _valid_digest(self.carrier_artifact_sha256)
            and _valid_digest(self.artifact_sha256),
            "V14 scan digest is invalid",
        )
        _require(
            self.carrier_arm
            in {STRICT_ARM, INFLATED_FALLBACK_ARM, ABSTAINED_ARM},
            "V14 carrier arm is invalid",
        )
        _require(
            len(self.selected_camera_ids) == 8
            and len(set(self.selected_camera_ids)) == 8,
            "V14 selected camera panel is invalid",
        )
        expected_support = 3 if self.carrier_arm == STRICT_ARM else 2
        expected_inflation = 1.0 if self.carrier_arm == STRICT_ARM else 4.0
        _require(
            self.depth_config.minimum_camera_support == expected_support
            and np.isclose(
                self.depth_config.correlation_covariance_inflation,
                expected_inflation,
            ),
            "V14 depth uncertainty differs from its carrier arm",
        )
        _require(
            np.isclose(
                self.admission_config.shared_bias_variance_m2,
                25e-6,
            ),
            "V14 shared-bias nuisance changed",
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_kind": "Deform360AdaptiveDirectDepthScanV14",
            "contract": CONTRACT,
            "carrier_artifact_sha256": self.carrier_artifact_sha256,
            "carrier_arm": self.carrier_arm,
            "selected_camera_ids": list(self.selected_camera_ids),
            "depth_config": asdict(self.depth_config),
            "admission_config": asdict(self.admission_config),
            "event_scan": self.scan.descriptor(),
            "information_boundary": {
                "maximum_object_observation_frame": (
                    self.scan.maximum_observation_frame
                ),
                "frame_zero_carrier_selection_only": True,
                "metric_depth_reassociated_at_causal_endpoints": True,
                "prefix_tactile_used": True,
                "measured_prefix_actuator_used": True,
                "proposal_and_validation_panels_disjoint": True,
                "validation_panel_formed_update": False,
                "future_object_observation_read": False,
                "future_identity_or_metric_read": False,
                "held_v8_artifact_or_process_access": False,
            },
            "artifact_sha256": self.artifact_sha256,
        }


def scan_adaptive_direct_depth_v14(
    case_id: str,
    physical_prediction_m: np.ndarray,
    carrier: AdaptiveCausalResponseQuerySchedule,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    prefix_depths_m: np.ndarray,
    prefix_object_masks: np.ndarray,
    action_support: np.ndarray,
    tactile_contact_probability: np.ndarray,
    measured_actuator_positions_m: np.ndarray,
    *,
    persistence_prediction_m: np.ndarray | None = None,
    event_config: CausalResponseEventConfig | None = None,
    depth_config: DirectDepthEndpointConfig | None = None,
    admission_config: CausalResponseAdmissionConfig | None = None,
) -> AdaptiveDirectDepthScanV14:
    """Run V12 causal abduction through the frozen V13 adaptive carrier."""

    matrices = np.asarray(intrinsics)
    poses = np.asarray(camera_to_world)
    depths = np.asarray(prefix_depths_m)
    masks = np.asarray(prefix_object_masks)
    available_count = len(carrier.available_camera_ids)
    _require(
        carrier.config.selected_camera_count == 8
        and carrier.config.panel_camera_count == 4
        and carrier.config.strict_minimum_support_per_panel == 3
        and carrier.config.fallback_minimum_support_per_panel == 2
        and np.isclose(carrier.config.fallback_covariance_inflation, 4.0)
        and np.isclose(carrier.config.shared_bias_std_m, 0.005),
        "V14 requires the frozen V13 4+4 support and uncertainty contract",
    )
    _require(
        matrices.shape == (available_count, 3, 3)
        and poses.shape == (available_count, 4, 4)
        and depths.ndim == 4
        and depths.shape[0] == available_count
        and masks.shape == depths.shape,
        "V14 camera prefix differs from the carrier's available panel",
    )
    selected = carrier.panels.selected_indices
    selected_names = tuple(carrier.available_camera_ids[index] for index in selected)
    _require(
        selected_names == carrier.selected_camera_ids
        and selected_names == carrier.query_schedule.camera_ids,
        "V14 selected camera order differs from the carrier",
    )
    support_count, covariance_inflation = _expected_support_and_inflation(carrier)
    base_depth = depth_config or DirectDepthEndpointConfig()
    depth_cfg = replace(
        base_depth,
        minimum_camera_support=support_count,
        correlation_covariance_inflation=covariance_inflation,
    )
    shared_bias_variance = carrier.config.shared_bias_std_m**2
    base_admission = admission_config or CausalResponseAdmissionConfig()
    admission_cfg = replace(
        base_admission,
        shared_bias_variance_m2=shared_bias_variance,
    )
    scan = scan_causal_response_event(
        case_id,
        physical_prediction_m,
        carrier.query_schedule,
        matrices[selected],
        poses[selected],
        depths[selected],
        masks[selected],
        action_support,
        tactile_contact_probability,
        measured_actuator_positions_m,
        persistence_prediction_m=persistence_prediction_m,
        event_config=event_config,
        depth_config=depth_cfg,
        admission_config=admission_cfg,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": CONTRACT,
        "carrier_artifact_sha256": carrier.artifact_sha256,
        "carrier_arm": carrier.arm,
        "selected_camera_ids": list(selected_names),
        "depth_config": asdict(depth_cfg),
        "admission_config": asdict(admission_cfg),
        "event_scan_sha256": scan.artifact_sha256,
    }
    digest = _canonical_sha256(payload, key="artifact_sha256")
    result = AdaptiveDirectDepthScanV14(
        carrier_artifact_sha256=carrier.artifact_sha256,
        carrier_arm=carrier.arm,
        selected_camera_ids=selected_names,
        depth_config=depth_cfg,
        admission_config=admission_cfg,
        scan=scan,
        artifact_sha256=digest,
    )
    _require(
        _canonical_sha256(
            {
                "schema_version": 1,
                "contract": CONTRACT,
                "carrier_artifact_sha256": result.carrier_artifact_sha256,
                "carrier_arm": result.carrier_arm,
                "selected_camera_ids": list(result.selected_camera_ids),
                "depth_config": asdict(result.depth_config),
                "admission_config": asdict(result.admission_config),
                "event_scan_sha256": result.scan.artifact_sha256,
            },
            key="artifact_sha256",
        )
        == result.artifact_sha256,
        "V14 scan descriptor changed after construction",
    )
    return result


def predict_adaptive_direct_depth_v14(
    physical_prediction_m: np.ndarray,
    scan: AdaptiveDirectDepthScanV14,
    *,
    persistence_prediction_m: np.ndarray | None = None,
    measurement_config: CausalResponseMeasurementConfig | None = None,
    belief_config: RecursiveRbfBeliefConfig | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Form the V14 candidate or preserve the selected baseline exactly."""

    base_measurement = measurement_config or CausalResponseMeasurementConfig()
    measurement_cfg = replace(
        base_measurement,
        shared_bias_variance_m2=scan.admission_config.shared_bias_variance_m2,
    )
    candidate, arrays = predict_scanned_causal_response(
        physical_prediction_m,
        scan.scan,
        persistence_prediction_m=persistence_prediction_m,
        measurement_config=measurement_cfg,
        belief_config=belief_config,
    )
    applied = bool(candidate["candidate_applied"])
    if not applied:
        _require(
            arrays[CANDIDATE_ARM].dtype == arrays[BASELINE_ARM].dtype
            and arrays[CANDIDATE_ARM].shape == arrays[BASELINE_ARM].shape
            and arrays[CANDIDATE_ARM].tobytes() == arrays[BASELINE_ARM].tobytes(),
            "rejected V14 candidate changed the selected baseline",
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360CausalResponseDirectDepthCandidateV14",
        "contract": CONTRACT,
        "adaptive_scan_sha256": scan.artifact_sha256,
        "carrier_artifact_sha256": scan.carrier_artifact_sha256,
        "carrier_arm": scan.carrier_arm,
        "candidate": dict(candidate),
        "candidate_applied": applied,
        "bit_exact_baseline_fallback": not applied,
        "baseline_array_sha256": array_sha256(arrays[BASELINE_ARM]),
        "candidate_array_sha256": array_sha256(arrays[CANDIDATE_ARM]),
        "information_boundary": {
            "maximum_object_observation_frame": (
                scan.scan.maximum_observation_frame
            ),
            "metric_depth_reassociated_at_causal_endpoints": True,
            "shared_camera_bias_retained_as_nuisance": True,
            "innovation_robustified_once": True,
            "validation_panel_formed_update": False,
            "future_object_observation_read": False,
            "future_identity_or_metric_read": False,
            "held_v8_artifact_or_process_access": False,
        },
    }
    report["result_sha256"] = _canonical_sha256(report, key="result_sha256")
    return report, arrays


def write_adaptive_direct_depth_v14_artifacts(
    output_dir: str | Path,
    carrier: AdaptiveCausalResponseQuerySchedule,
    scan: AdaptiveDirectDepthScanV14,
    candidate_report: Mapping[str, Any],
    candidate_arrays: Mapping[str, np.ndarray],
    *,
    case_id: str,
    repository_revision: str,
    protocol_path: str | Path,
    input_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Seal one V14 candidate before any future outcome is authorized."""

    _require(scan.scan.case_id == case_id, "V14 case differs from its scan")
    _require(
        carrier.artifact_sha256 == scan.carrier_artifact_sha256,
        "V14 carrier differs from its scan",
    )
    _require(
        len(repository_revision) == 40
        and all(character in "0123456789abcdef" for character in repository_revision),
        "repository revision is invalid",
    )
    supplied_inputs = dict(sorted(input_sha256.items()))
    _require(
        supplied_inputs
        and all(_valid_digest(value) for value in supplied_inputs.values()),
        "V14 input provenance is invalid",
    )
    _require(
        candidate_report.get("result_sha256")
        == _canonical_sha256(candidate_report, key="result_sha256")
        and candidate_report.get("adaptive_scan_sha256") == scan.artifact_sha256,
        "V14 candidate report is invalid",
    )
    arrays = {
        BASELINE_ARM: np.asarray(candidate_arrays[BASELINE_ARM]),
        CANDIDATE_ARM: np.asarray(candidate_arrays[CANDIDATE_ARM]),
        "candidate_correction_variance_m2": np.asarray(
            candidate_arrays["candidate_correction_variance_m2"]
        ),
    }
    output = Path(output_dir).resolve()
    _require(not output.exists(), "V14 output directory already exists")
    output.mkdir(parents=True)
    archive_path = output / ARCHIVE_FILENAME
    temporary = archive_path.with_name(archive_path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(archive_path)
    applied = bool(candidate_report["candidate_applied"])
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "contract": CONTRACT,
        "case": case_id,
        "status": (
            "candidate_prediction_sealed"
            if applied
            else "exact_baseline_fallback_sealed"
        ),
        "repository_revision": repository_revision,
        "protocol": {
            "path": str(Path(protocol_path)),
            "file_sha256": file_sha256(protocol_path),
        },
        "carrier": carrier.descriptor(),
        "adaptive_scan": scan.descriptor(),
        "candidate": dict(candidate_report),
        "inputs_sha256": supplied_inputs,
        "archive": {
            "filename": ARCHIVE_FILENAME,
            "file_sha256": file_sha256(archive_path),
            "array_sha256": {
                name: array_sha256(values) for name, values in sorted(arrays.items())
            },
        },
        "information_boundary": {
            **candidate_report["information_boundary"],
            "prediction_sealed_before_outcome_authorization": True,
            "query_or_event_rejection_is_exact_fallback": True,
        },
    }
    report["result_sha256"] = _canonical_sha256(report, key="result_sha256")
    (output / REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_adaptive_direct_depth_v14_artifacts(output)
    return report


def validate_adaptive_direct_depth_v14_artifacts(
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate one sealed V14 prediction without opening an outcome."""

    output = Path(output_dir).resolve()
    report = json.loads((output / REPORT_FILENAME).read_text(encoding="utf-8"))
    _require(
        report.get("artifact_kind") == ARTIFACT_KIND
        and report.get("contract") == CONTRACT
        and report.get("status")
        in {"candidate_prediction_sealed", "exact_baseline_fallback_sealed"}
        and report.get("result_sha256")
        == _canonical_sha256(report, key="result_sha256"),
        "V14 prediction report is invalid",
    )
    boundary = report["information_boundary"]
    _require(
        boundary.get("future_object_observation_read") is False
        and boundary.get("future_identity_or_metric_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False
        and boundary.get("prediction_sealed_before_outcome_authorization") is True,
        "V14 prediction crossed its information boundary",
    )
    archive_path = output / ARCHIVE_FILENAME
    _require(
        report["archive"]["filename"] == ARCHIVE_FILENAME
        and report["archive"]["file_sha256"] == file_sha256(archive_path),
        "V14 archive checksum changed",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    observed = {name: array_sha256(values) for name, values in sorted(arrays.items())}
    _require(
        observed == report["archive"]["array_sha256"],
        "V14 archive arrays changed",
    )
    fallback = report["status"] == "exact_baseline_fallback_sealed"
    _require(
        fallback == (not bool(report["candidate"]["candidate_applied"])),
        "V14 status differs from its candidate",
    )
    if fallback:
        _require(
            arrays[CANDIDATE_ARM].dtype == arrays[BASELINE_ARM].dtype
            and arrays[CANDIDATE_ARM].shape == arrays[BASELINE_ARM].shape
            and arrays[CANDIDATE_ARM].tobytes() == arrays[BASELINE_ARM].tobytes(),
            "V14 fallback is not bit exact",
        )
    return report, arrays


__all__ = [
    "ARCHIVE_FILENAME",
    "ARTIFACT_KIND",
    "CONTRACT",
    "REPORT_FILENAME",
    "AdaptiveDirectDepthScanV14",
    "predict_adaptive_direct_depth_v14",
    "scan_adaptive_direct_depth_v14",
    "validate_adaptive_direct_depth_v14_artifacts",
    "write_adaptive_direct_depth_v14_artifacts",
]
