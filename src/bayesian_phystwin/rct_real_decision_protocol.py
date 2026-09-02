"""Validate the pre-outcome RCT real-decision probe protocol."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

SCHEMA_VERSION = 1
PROTOCOL_ID = "rct-real-decision-probe-protocol-v1"
PREOUTCOME_CLARIFICATION_ID = "rct-real-decision-probe-preoutcome-clarification-v1"
PREOUTCOME_AMENDMENT_V2_ID = "rct-real-decision-probe-preoutcome-amendment-v2"
RCT_CODE_REVISION = "8d2f2de96b08d7c1e4d754e327b974f3e41283b8"
RCT_ARCHIVE_FILE_ID = 65037834
RCT_ARCHIVE_SIZE_BYTES = 9_905_561_734
REGISTERED_INDENTATIONS_MM = (0.4, 0.8, 1.2)
MANDATORY_ANCHOR = (1, 1)
SELECTABLE_PROBES = ((1, 2), (2, 1), (2, 2))
HELD_INTERVENTION = (3, 3)
CALIBRATION_MATERIALS = (
    "0700120",
    "0700300",
    "0700340",
    "0700490",
    "0700680",
    "0700750",
    "0700830",
    "0700970",
    "0701320",
    "0701430",
    "0701650",
    "0701850",
    "0710070",
    "0710100",
    "0710200",
    "0710310",
    "0710330",
    "0710530",
    "0710550",
    "710320",
)
SOURCE_TEST_MATERIALS = (
    "0700140",
    "0700170",
    "070020",
    "0700210",
    "0700320",
    "0700610",
    "0700660",
    "0701280",
    "0701290",
    "0701460",
    "0701990",
    "0710140",
    "0710150",
    "0710240",
    "0710280",
    "0710360",
    "0710520",
    "0710580",
    "700870",
    "700920",
)
CONFIRMATION_MATERIALS = (
    "0700070",
    "0700160",
    "0700220",
    "0700510",
    "0700520",
    "0700600",
    "0700640",
    "0700690",
    "0700790",
    "0700800",
    "0701110",
    "0701380",
    "0701420",
    "0701490",
    "0701610",
    "0701710",
    "0710210",
    "0710380",
    "0710390",
    "0710510",
)
OFFICIAL_SPLIT_SHA256 = {
    "matholdout_K20_bal_seed0": (
        "2813cbf7f76cfe4fc4116ce3abb5b1fc7edc8f5a0a2996f5016952cce6531111"
    ),
    "matholdout_K20_bal_seed42": (
        "862d5c51d67265f661f330c93a116c6abb4c0eee6b4c3c53bbc8069b6b965421"
    ),
    "matholdout_K20_bal_seed7": (
        "921c1799c3f70c4ff5e6c27bd6081deda8ed931fed17a80896c709ae65084adb"
    ),
}
_CANONICAL_CONFIG_SHA256 = (
    "6a6d0d0b52ed71cb530e0ad5cb5fe5898f202d6dd9ad099cab6b035fa063a140"
)
_PROTOCOL_FILE_SHA256 = (
    "c6eac3371e379956c285fe0ea0743c2ba9b67eb40d09fe18a3642839188ba8bd"
)
_PREOUTCOME_CLARIFICATION_CONFIG_SHA256 = (
    "f9248258e40cd42cd718f1244658234777731681720afa1733c3c05f68346b05"
)
_PREOUTCOME_PROTOCOL_COMMIT = "43b47d26a57c1340873a4136f4aea735e1febdd3"
_PREOUTCOME_CLARIFICATION_FILE_SHA256 = (
    "e05f77b571e3676cfd63fa8efcc73028859921a3b7c0f6516996220c4f5de87f"
)
_PREOUTCOME_CLARIFICATION_COMMIT = "46c44e22906196979bc27598181e31ab6046dfd8"
_PREOUTCOME_AMENDMENT_V2_CONFIG_SHA256 = (
    "c5dda44afc698e7bab0ff897ff8e58cda7afa1e8e8a69c2284da01e73bf1e6fc"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: object, *, message: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), message)
    return cast(Mapping[str, Any], value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def protocol_config_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical digest of a parsed RCT protocol."""

    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def protocol_file_sha256(path: str | Path) -> str:
    """Return the byte digest of a protocol file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RCTRealDecisionCohort:
    """Material identities assigned before any registered force outcome is read."""

    calibration: tuple[str, ...]
    source_test: tuple[str, ...]
    confirmation: tuple[str, ...]
    expected_fit_count: int

    @property
    def reserved(self) -> frozenset[str]:
        return frozenset(self.calibration + self.source_test + self.confirmation)

    def role(self, material_id: str) -> str:
        """Return the predeclared role of a released RCT material ID."""

        if material_id in self.calibration:
            return "calibration"
        if material_id in self.source_test:
            return "source_test"
        if material_id in self.confirmation:
            return "confirmation"
        return "fit"


def _material_ids(record: object, *, role: str) -> tuple[str, ...]:
    record_mapping = _mapping(record, message=f"{role} cohort record is missing")
    raw = record_mapping.get("material_ids")
    _require(
        isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)),
        f"{role} material IDs must be a list",
    )
    values = tuple(str(value) for value in cast(Sequence[object], raw))
    _require(len(values) == 20, f"{role} material count changed")
    _require(len(values) == len(set(values)), f"{role} contains duplicate materials")
    return values


def _validate_dataset(payload: Mapping[str, Any]) -> None:
    dataset = _mapping(payload.get("dataset"), message="dataset lock is missing")
    _require(
        dataset.get("code_revision") == RCT_CODE_REVISION,
        "RCT code revision changed",
    )
    _require(
        dataset.get("official_split_manifests") == OFFICIAL_SPLIT_SHA256,
        "official RCT split digests changed",
    )
    _require(dataset.get("license") == "CC BY 4.0", "dataset license changed")
    counts = dataset.get("counts")
    _require(
        counts
        == {
            "force_traces": 1827,
            "material_samples": 122,
            "robot_press_sequences": 1832,
            "tactile_frames": 29279,
            "tactile_sensors": 3,
        },
        "RCT release counts changed",
    )
    archive = _mapping(dataset.get("archive"), message="archive lock is missing")
    _require(
        int(archive.get("file_id", -1)) == RCT_ARCHIVE_FILE_ID,
        "archive file ID changed",
    )
    _require(
        int(archive.get("size_bytes", -1)) == RCT_ARCHIVE_SIZE_BYTES,
        "archive size changed",
    )
    _require(archive.get("expected_sha256") is None, "preaccess archive hash changed")
    _require(
        archive.get("status") == "download-in-progress-hash-required-before-extraction",
        "preaccess archive status changed",
    )


def _validate_cohort(payload: Mapping[str, Any]) -> RCTRealDecisionCohort:
    cohort = _mapping(payload.get("cohort"), message="cohort lock is missing")
    calibration = _material_ids(cohort.get("calibration"), role="calibration")
    source_test = _material_ids(cohort.get("source_test"), role="source_test")
    confirmation = _material_ids(cohort.get("confirmation"), role="confirmation")
    _require(calibration == CALIBRATION_MATERIALS, "calibration cohort changed")
    _require(source_test == SOURCE_TEST_MATERIALS, "source-test cohort changed")
    _require(confirmation == CONFIRMATION_MATERIALS, "confirmation cohort changed")
    _require(cohort.get("pairwise_disjoint") is True, "cohort disjointness changed")
    reserved = calibration + source_test + confirmation
    _require(len(set(reserved)) == 60, "registered cohorts overlap")
    fit = _mapping(cohort.get("fit"), message="fit cohort rule is missing")
    _require(int(fit.get("expected_material_count", -1)) == 62, "fit count changed")
    _require(
        fit.get("rule")
        == "all released material IDs outside calibration, source_test, and confirmation",
        "fit cohort rule changed",
    )
    return RCTRealDecisionCohort(
        calibration=calibration,
        source_test=source_test,
        confirmation=confirmation,
        expected_fit_count=62,
    )


def _probe_tuple(value: object, *, name: str) -> tuple[int, int]:
    value_mapping = _mapping(value, message=f"{name} record is missing")
    return int(value_mapping.get("position", -1)), int(value_mapping.get("sensor", -1))


def _validate_method(payload: Mapping[str, Any]) -> None:
    method = _mapping(payload.get("method"), message="method lock is missing")
    _require(
        _probe_tuple(method.get("mandatory_anchor"), name="mandatory anchor")
        == MANDATORY_ANCHOR,
        "mandatory anchor changed",
    )
    raw_probes = method.get("selectable_probes")
    _require(isinstance(raw_probes, Sequence), "selectable probes are missing")
    probes = tuple(
        _probe_tuple(value, name="selectable probe")
        for value in cast(Sequence[object], raw_probes)
    )
    _require(probes == SELECTABLE_PROBES, "selectable probe roster changed")
    fit = _mapping(method.get("fit"), message="fit method is missing")
    _require(
        float(fit.get("covariance_diagonal_shrinkage", -1.0)) == 0.25,
        "covariance shrinkage changed",
    )
    _require(
        float(fit.get("jitter_fraction_of_median_variance", -1.0)) == 1e-8,
        "covariance jitter changed",
    )
    selectors = _mapping(
        method.get("probe_selectors"), message="probe selectors are missing"
    )
    decision_directed = _mapping(
        selectors.get("decision_directed"),
        message="decision-directed selector is missing",
    )
    _require(
        int(decision_directed.get("predictive_draw_count", -1)) == 4096,
        "predictive draw count changed",
    )
    _require(
        int(decision_directed.get("predictive_seed", -1)) == 20260902,
        "predictive seed changed",
    )
    guard = _mapping(
        method.get("universal_conformal_guard"), message="conformal guard is missing"
    )
    _require(float(guard.get("coverage", 0.0)) == 0.9, "guard coverage changed")
    _require(guard.get("shared_across_methods") is True, "guard sharing changed")
    _require(guard.get("uses_confirmation") is False, "confirmation entered guard")


def _validate_decision(payload: Mapping[str, Any]) -> None:
    decision = _mapping(payload.get("decision"), message="decision lock is missing")
    action_grid = tuple(
        float(value) for value in decision.get("action_indentation_mm", ())
    )
    _require(action_grid == (0.0, *REGISTERED_INDENTATIONS_MM), "action grid changed")
    _require(
        _probe_tuple(decision.get("held_intervention"), name="held intervention")
        == HELD_INTERVENTION,
        "held intervention changed",
    )
    force_limit = _mapping(
        decision.get("force_limit"), message="force-limit rule is missing"
    )
    _require(float(force_limit.get("quantile", -1.0)) == 0.6, "force limit changed")
    _require(force_limit.get("fit_only") is True, "force limit is not fit-only")


def _validate_boundary(payload: Mapping[str, Any]) -> None:
    boundary = _mapping(
        payload.get("information_boundary"), message="information boundary is missing"
    )
    for key in (
        "calibration_force_rows_open_before_protocol_freeze",
        "confirmation_descriptor_labels_used",
        "confirmation_force_rows_open_before_authorization",
        "confirmation_tactile_frames_used",
        "held_v8_access_allowed",
        "method_changes_after_source_test",
        "source_test_force_rows_open_before_protocol_freeze",
    ):
        _require(boundary.get(key) is False, f"information boundary changed: {key}")
    for key in (
        "confirmation_material_ids_are_public_split_metadata",
        "target_open_requires_archive_hash",
        "target_open_requires_implementation_seal",
        "target_open_requires_source_gate",
        "target_open_requires_write_once_authorization",
    ):
        _require(boundary.get(key) is True, f"information boundary changed: {key}")
    promotion = _mapping(payload.get("promotion"), message="promotion lock is missing")
    _require(promotion.get("target_authorized") is False, "target was authorized early")
    _require(
        int(promotion.get("target_attempt_limit", -1)) == 1, "attempt limit changed"
    )


def load_rct_real_decision_protocol(path: str | Path) -> dict[str, Any]:
    """Load the exact pre-outcome RCT protocol and reject any drift."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "protocol must be a JSON object")
    _require(payload.get("schema_version") == SCHEMA_VERSION, "schema changed")
    _require(payload.get("contract") == PROTOCOL_ID, "protocol ID changed")
    _require(
        payload.get("status") == "frozen-before-force-outcome-access",
        "protocol status changed",
    )
    _validate_dataset(payload)
    _validate_cohort(payload)
    _validate_method(payload)
    _validate_decision(payload)
    _validate_boundary(payload)
    _require(
        protocol_config_sha256(payload) == _CANONICAL_CONFIG_SHA256,
        "canonical RCT protocol digest changed",
    )
    return dict(payload)


def load_rct_preoutcome_clarification(path: str | Path) -> dict[str, Any]:
    """Load the source-independent clarification bound to the frozen protocol."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "clarification must be a JSON object")
    _require(payload.get("schema_version") == 1, "clarification schema changed")
    _require(
        payload.get("contract") == PREOUTCOME_CLARIFICATION_ID,
        "clarification ID changed",
    )
    _require(
        payload.get("status") == "frozen-before-force-outcome-access",
        "clarification status changed",
    )
    _require(
        payload.get("parent")
        == {
            "protocol_commit": _PREOUTCOME_PROTOCOL_COMMIT,
            "protocol_config_sha256": _CANONICAL_CONFIG_SHA256,
            "protocol_file_sha256": _PROTOCOL_FILE_SHA256,
        },
        "clarification parent lock changed",
    )
    _require(
        payload.get("information_boundary")
        == {
            "archive_download_complete": False,
            "calibration_force_rows_opened": False,
            "confirmation_force_rows_opened": False,
            "held_v8_accessed": False,
            "source_test_force_rows_opened": False,
        },
        "clarification information boundary changed",
    )
    _require(
        payload.get("scientific_effect")
        == {
            "changes_action_grid": False,
            "changes_cohort": False,
            "changes_force_limit": False,
            "changes_method_parameters": False,
            "changes_probe_roster": False,
            "changes_promotion_thresholds": False,
            "changes_statistical_test": False,
        },
        "clarification scientific effect changed",
    )
    clarifications = payload.get("clarifications")
    _require(isinstance(clarifications, Mapping), "clarifications are missing")
    _require(
        clarifications.get("force_increment_n")
        == (
            "abs(raw_fz_frame - "
            "raw_fz_at_the_maximum_z_frame_within_the_same_trajectory)"
        ),
        "force-increment clarification changed",
    )
    _require(
        protocol_config_sha256(payload) == _PREOUTCOME_CLARIFICATION_CONFIG_SHA256,
        "canonical clarification digest changed",
    )
    return dict(payload)


def load_rct_preoutcome_amendment_v2(path: str | Path) -> dict[str, Any]:
    """Load the practical-significance amendment frozen before force access."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "amendment must be a JSON object")
    _require(payload.get("schema_version") == 1, "amendment schema changed")
    _require(
        payload.get("contract") == PREOUTCOME_AMENDMENT_V2_ID,
        "amendment ID changed",
    )
    _require(
        payload.get("status") == "frozen-before-force-outcome-access",
        "amendment status changed",
    )
    _require(
        payload.get("parent")
        == {
            "clarification_commit": _PREOUTCOME_CLARIFICATION_COMMIT,
            "clarification_config_sha256": (_PREOUTCOME_CLARIFICATION_CONFIG_SHA256),
            "clarification_file_sha256": _PREOUTCOME_CLARIFICATION_FILE_SHA256,
            "protocol_config_sha256": _CANONICAL_CONFIG_SHA256,
            "protocol_file_sha256": _PROTOCOL_FILE_SHA256,
        },
        "amendment parent lock changed",
    )
    _require(
        payload.get("information_boundary")
        == {
            "archive_download_complete": False,
            "calibration_force_rows_opened": False,
            "confirmation_force_rows_opened": False,
            "held_v8_accessed": False,
            "source_test_force_rows_opened": False,
        },
        "amendment information boundary changed",
    )
    change = payload.get("change")
    _require(isinstance(change, Mapping), "amendment change is missing")
    _require(
        float(change.get("confirmation_minimum_relative_auc_improvement", -1.0))
        == 0.05,
        "confirmation practical threshold changed",
    )
    _require(
        protocol_config_sha256(payload) == _PREOUTCOME_AMENDMENT_V2_CONFIG_SHA256,
        "canonical amendment digest changed",
    )
    return dict(payload)


def cohort_from_protocol(payload: Mapping[str, Any]) -> RCTRealDecisionCohort:
    """Return the validated material roles from an already parsed protocol."""

    _require(payload.get("contract") == PROTOCOL_ID, "protocol ID changed")
    return _validate_cohort(payload)


__all__ = [
    "CALIBRATION_MATERIALS",
    "CONFIRMATION_MATERIALS",
    "HELD_INTERVENTION",
    "MANDATORY_ANCHOR",
    "OFFICIAL_SPLIT_SHA256",
    "PREOUTCOME_AMENDMENT_V2_ID",
    "PREOUTCOME_CLARIFICATION_ID",
    "PROTOCOL_ID",
    "RCT_ARCHIVE_FILE_ID",
    "RCT_ARCHIVE_SIZE_BYTES",
    "RCT_CODE_REVISION",
    "RCTRealDecisionCohort",
    "SELECTABLE_PROBES",
    "SOURCE_TEST_MATERIALS",
    "cohort_from_protocol",
    "load_rct_preoutcome_amendment_v2",
    "load_rct_preoutcome_clarification",
    "load_rct_real_decision_protocol",
    "protocol_config_sha256",
    "protocol_file_sha256",
]
