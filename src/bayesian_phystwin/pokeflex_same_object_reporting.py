"""Paper-facing diagnostics for the frozen PokeFlex same-object replication."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from .pokeflex_independent_depth_regret_guard import (
    _bound_from_dict,
    _certificate_from_dict,
    _select_candidates,
    extract_pokeflex_regret_guard_rows,
)

EXPECTED_ARTIFACT_KIND = "PokeFlexIndependentDepthRegretGuardProspectiveEvaluation"
EXPECTED_CLAIM_STATUS = "prospective development-take replication"
BOUNDED_CLAIM = (
    "A source-calibrated independent-depth regret guard improved the released "
    "PokeFlex checkpoint on three prospective new takes of two previously seen "
    "objects while returning the released checkpoint exactly when it abstained."
)
EXCLUDED_CLAIMS = (
    "independent-object generalization",
    "PokeFlex target-object performance",
    "state of the art",
    "general deployment calibration",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping_list(
    value: object,
    *,
    name: str,
    expected_length: int | None = None,
    nonempty: bool = False,
) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    if expected_length is not None and len(value) != expected_length:
        raise ValueError(f"{name} must contain exactly {expected_length} entries")
    if nonempty and not value:
        raise ValueError(f"{name} must not be empty")
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"{name} entries must be objects")
    return cast(list[Mapping[str, Any]], value)


def _string_list(value: object, *, name: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{name} must contain nonempty strings")
    return cast(list[str], value)


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _close(first: object, second: object, *, atol: float = 1e-10) -> bool:
    if first is None or second is None:
        return first is second
    return bool(
        np.isclose(
            _number(first, name="first comparison value"),
            _number(second, name="second comparison value"),
            atol=atol,
            rtol=0.0,
        )
    )


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    """Load one JSON object and reject non-object roots."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def validate_bounded_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and summarize the frozen prospective result without retuning it."""

    _require(
        result.get("artifact_kind") == EXPECTED_ARTIFACT_KIND,
        "unexpected prospective result kind",
    )
    _require(
        result.get("claim_status") == EXPECTED_CLAIM_STATUS,
        "prospective claim status changed",
    )
    _require(result.get("gate_passed") is True, "prospective gate did not pass")
    takes = _mapping_list(result.get("takes"), name="takes", expected_length=3)
    objects = _mapping_list(result.get("objects"), name="objects", expected_length=2)
    decisions = _mapping_list(result.get("decisions"), name="decisions", nonempty=True)
    _require(
        _integer(result.get("take_count", -1), name="take_count") == len(takes),
        "take count changed",
    )
    _require(
        _integer(result.get("object_count", -1), name="object_count") == len(objects),
        "object count changed",
    )
    _require(
        _integer(result.get("object_wins", -1), name="object_wins") == 2,
        "object-win result changed",
    )
    _require(
        _integer(result.get("object_losses", -1), name="object_losses") == 0,
        "object-loss result changed",
    )

    frame_count = sum(
        _integer(value.get("target_frame_count"), name="target_frame_count")
        for value in takes
    )
    accepted_count = _integer(
        result.get("accepted_frame_count"), name="accepted_frame_count"
    )
    fallback_count = _integer(
        result.get("exact_fallback_frame_count"), name="exact_fallback_frame_count"
    )
    _require(frame_count == len(decisions), "take/frame accounting changed")
    _require(
        accepted_count + fallback_count == frame_count,
        "accept/fallback accounting changed",
    )
    accepted_wins = _integer(
        result.get("accepted_frame_wins"), name="accepted_frame_wins"
    )
    accepted_losses = _integer(
        result.get("accepted_frame_losses"), name="accepted_frame_losses"
    )
    _require(
        accepted_wins + accepted_losses == accepted_count,
        "accepted-frame accounting changed",
    )
    for take in takes:
        baseline = _number(
            take.get("baseline_mean_CD_UL1_mm"),
            name="baseline_mean_CD_UL1_mm",
        )
        selected = _number(
            take.get("selected_mean_CD_UL1_mm"),
            name="selected_mean_CD_UL1_mm",
        )
        _require(selected <= baseline + 1e-12, "a prospective take now regresses")

    baseline = _number(
        result.get("baseline_object_mean_CD_UL1_mm"),
        name="baseline_object_mean_CD_UL1_mm",
    )
    selected = _number(
        result.get("selected_object_mean_CD_UL1_mm"),
        name="selected_object_mean_CD_UL1_mm",
    )
    relative = _number(
        result.get("object_balanced_relative_improvement"),
        name="object_balanced_relative_improvement",
    )
    _require(
        _close(relative, (baseline - selected) / baseline),
        "object-balanced improvement is inconsistent",
    )
    return {
        "claim": BOUNDED_CLAIM,
        "excluded_claims": list(EXCLUDED_CLAIMS),
        "aggregation": result["aggregation"],
        "object_count": len(objects),
        "take_count": len(takes),
        "frame_count": frame_count,
        "baseline_object_mean_CD_UL1_mm": baseline,
        "guarded_object_mean_CD_UL1_mm": selected,
        "object_balanced_relative_improvement": relative,
        "object_wins": 2,
        "object_losses": 0,
        "accepted_frame_count": accepted_count,
        "accepted_frame_wins": accepted_wins,
        "accepted_frame_losses": accepted_losses,
        "exact_fallback_frame_count": fallback_count,
        "takes": takes,
        "objects": objects,
    }


def build_candidate_diagnostics(
    candidate_payloads: Sequence[Mapping[str, Any]],
    prospective_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct frozen pre-fallback choices for a post-outcome diagnostic."""

    bounded = validate_bounded_result(prospective_result)
    extracted_rows, extracted_frames = extract_pokeflex_regret_guard_rows(
        candidate_payloads
    )
    rows = cast(list[dict[str, Any]], extracted_rows)
    frames = cast(list[dict[str, Any]], extracted_frames)
    observed_takes = sorted({str(frame["take_id"]) for frame in frames})
    expected_takes = sorted(
        _string_list(prospective_result.get("take_ids"), name="take_ids")
    )
    _require(observed_takes == expected_takes, "candidate take inventory changed")

    deployment_value = prospective_result.get("deployment_artifact")
    _require(isinstance(deployment_value, Mapping), "deployment artifact is missing")
    deployment = cast(Mapping[str, Any], deployment_value)
    certificate = _certificate_from_dict(deployment["candidate_certificate"])
    selector_bound = _bound_from_dict(deployment["selector_correction_bound"])
    upper_by_index = {
        index: certificate.upper_regret(row["features"])
        for index, row in enumerate(rows)
    }
    selected = _select_candidates(rows, upper_by_index)
    committed_records = _mapping_list(
        prospective_result.get("decisions"), name="decisions", nonempty=True
    )
    committed = {str(value["frame_id"]): value for value in committed_records}
    _require(len(committed) == len(frames), "committed decision inventory changed")

    diagnostics: list[dict[str, Any]] = []
    for frame in frames:
        frame_id = str(frame["frame_id"])
        recorded = committed.get(frame_id)
        if recorded is None:
            raise ValueError(f"missing committed decision: {frame_id}")
        baseline = _number(frame.get("baseline_error_mm"), name="baseline_error_mm")
        candidate = selected.get(frame_id)
        if candidate is None:
            _require(recorded["accepted"] is False, "unsupported frame was accepted")
            _require(
                recorded["candidate_upper_regret_mm"] is None,
                "unsupported frame has a candidate bound",
            )
            diagnostics.append(
                {
                    **frame,
                    "candidate_supported": False,
                    "accepted": False,
                    "selected_arm": "released_checkpoint",
                    "baseline_error_mm": baseline,
                    "deployed_error_mm": baseline,
                }
            )
            continue

        candidate_upper, row_index = candidate
        row = rows[row_index]
        adjusted_upper = float(candidate_upper + selector_bound.upper_regret_m)
        candidate_error = _number(
            row.get("candidate_error_mm"), name="candidate_error_mm"
        )
        candidate_regret = candidate_error - baseline
        accepted = adjusted_upper < -certificate.minimum_improvement
        deployed_error = candidate_error if accepted else baseline
        deployed_arm = str(row["candidate"]) if accepted else "released_checkpoint"

        _require(
            _close(recorded["candidate_upper_regret_mm"], candidate_upper),
            f"candidate bound changed: {frame_id}",
        )
        _require(
            _close(recorded["selector_adjusted_upper_regret_mm"], adjusted_upper),
            f"selector-adjusted bound changed: {frame_id}",
        )
        _require(
            bool(recorded["accepted"]) is accepted,
            f"decision changed: {frame_id}",
        )
        _require(
            str(recorded["selected_arm"]) == deployed_arm,
            f"selected arm changed: {frame_id}",
        )
        _require(
            _close(recorded["selected_error_mm"], deployed_error),
            f"deployed error changed: {frame_id}",
        )
        diagnostics.append(
            {
                **frame,
                "candidate_supported": True,
                "candidate": str(row["candidate"]),
                "candidate_upper_regret_mm": float(candidate_upper),
                "selector_adjusted_upper_regret_mm": adjusted_upper,
                "candidate_error_mm": candidate_error,
                "candidate_regret_mm": candidate_regret,
                "upper_bound_covered": candidate_regret <= adjusted_upper + 1e-12,
                "accepted": accepted,
                "selected_arm": deployed_arm,
                "baseline_error_mm": baseline,
                "deployed_error_mm": deployed_error,
                "deployed_regret_mm": deployed_error - baseline,
            }
        )

    supported = [value for value in diagnostics if value["candidate_supported"]]
    accepted_rows = [value for value in supported if value["accepted"]]
    rejected_rows = [value for value in supported if not value["accepted"]]
    harmful_accepted = [
        value for value in accepted_rows if value["candidate_regret_mm"] > 1e-12
    ]
    safe_accepted = [
        value for value in accepted_rows if value["candidate_regret_mm"] < -1e-12
    ]
    harmful_prevented = [
        value for value in rejected_rows if value["candidate_regret_mm"] > 1e-12
    ]
    conservative_fallbacks = [
        value for value in rejected_rows if value["candidate_regret_mm"] < -1e-12
    ]
    coverage = [bool(value["upper_bound_covered"]) for value in supported]
    _require(
        len(accepted_rows)
        == _integer(
            prospective_result.get("accepted_frame_count"),
            name="accepted_frame_count",
        ),
        "accepted diagnostic count changed",
    )
    _require(
        len(safe_accepted)
        == _integer(
            prospective_result.get("accepted_frame_wins"),
            name="accepted_frame_wins",
        ),
        "accepted-win diagnostic count changed",
    )
    _require(
        len(harmful_accepted)
        == _integer(
            prospective_result.get("accepted_frame_losses"),
            name="accepted_frame_losses",
        ),
        "accepted-loss diagnostic count changed",
    )

    return {
        "schema_version": 1,
        "artifact_kind": "PokeFlexSameObjectPaperDiagnosticV1",
        "analysis_role": (
            "post-outcome visualization of a frozen prospective decision rule; "
            "not method selection or calibration"
        ),
        "bounded_result": bounded,
        "candidate_diagnostic": {
            "candidate_supported_frame_count": len(supported),
            "candidate_unsupported_frame_count": len(diagnostics) - len(supported),
            "adjusted_upper_bound_coverage": (
                float(np.mean(coverage)) if coverage else None
            ),
            "accepted_frame_count": len(accepted_rows),
            "safe_accepted_frame_count": len(safe_accepted),
            "harmful_accepted_frame_count": len(harmful_accepted),
            "accepted_harmful_fraction": (
                len(harmful_accepted) / len(accepted_rows) if accepted_rows else 0.0
            ),
            "harmful_candidate_fallback_count": len(harmful_prevented),
            "beneficial_candidate_fallback_count": len(conservative_fallbacks),
            "candidate_nominal_coverage": certificate.nominal_coverage,
            "candidate_finite_sample_coverage": certificate.finite_sample_coverage,
            "selector_nominal_coverage": selector_bound.nominal_coverage,
            "selector_finite_sample_coverage": selector_bound.finite_sample_coverage,
            "selector_correction_mm": selector_bound.upper_regret_m,
            "frozen_acceptance_threshold_mm": -certificate.minimum_improvement,
        },
        "rows": diagnostics,
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write deterministic finite JSON."""

    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
