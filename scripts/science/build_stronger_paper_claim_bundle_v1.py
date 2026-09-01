#!/usr/bin/env python3
"""Build a conservative paper-claim ladder from three retained evidence streams."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

SCHEMA = "bayesian-phystwin/stronger-paper-claim-bundle-v1"
DIRECT_CONTRACT = "deform-dlo3-cross-backend-coefficient-transfer-result-v1"
SCALAR_CONTRACT = "deform-dlo3-cross-backend-scalar-transport-result-v1"
COLLECTION_SCHEMA = "bayesian-phystwin/deform-dlo45-run-evidence-collection-v1"
RELEVANT_BOOLEAN_KEYS = {
    "passed",
    "supported",
    "source_gate_passed",
    "target_gate_passed",
    "promotion_gate_passed",
    "replication_supported",
    "joint_gate_passed",
}
NEGATIVE_DECISION_TOKENS = (
    "not-supported",
    "unsupported",
    "failed",
    "failure",
    "reject",
    "blocked",
    "not-replicated",
    "negative",
)
POSITIVE_DECISION_TOKENS = (
    "supported",
    "replicated",
    "passed",
    "success",
    "promote",
    "confirmed",
)


class ClaimBundleError(RuntimeError):
    """Raised when evidence violates the claim-bundle contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ClaimBundleError(message)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    _require(isinstance(value, Mapping), f"{label} must be a JSON object")
    return value


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, object]:
    source = path.resolve(strict=True)
    return {
        "path": str(source),
        "sha256": _sha256(source),
        "size_bytes": source.stat().st_size,
    }


def _validate_direct(value: Mapping[str, object]) -> dict[str, object]:
    _require(value.get("contract") == DIRECT_CONTRACT, "direct result contract differs")
    boundary = _mapping(value.get("information_boundary"), label="direct boundary")
    _require(boundary.get("pyelastica_refit") is False, "direct result used a refit")
    _require(boundary.get("deform_refit") is False, "direct result changed DEFORM")
    _require(
        boundary.get("dlo3_official_evaluation_read") is False,
        "direct result opened DLO3 official evaluation",
    )
    _require(
        boundary.get("dlo4_or_dlo5_read") is False, "direct result read reserve DLOs"
    )
    gate = _mapping(value.get("promotion_gate"), label="direct promotion gate")
    primary = _mapping(
        value.get("primary_vs_raw_pyelastica"), label="direct primary comparison"
    )
    supported = (
        value.get("decision") == "no-refit-cross-backend-transfer-supported"
        and gate.get("supported") is True
    )
    return {
        "supported": supported,
        "decision": value.get("decision"),
        "relative_improvement": primary.get("relative_improvement"),
        "wins": primary.get("wins"),
        "losses": primary.get("losses"),
        "maximum_case_ratio": primary.get("maximum_case_ratio"),
        "improving_seed_models": gate.get("improving_seed_models"),
        "backend_specific_gain_retained_fraction": value.get(
            "backend_specific_gain_retained_fraction"
        ),
    }


def _validate_scalar(value: Mapping[str, object]) -> dict[str, object]:
    _require(value.get("contract") == SCALAR_CONTRACT, "scalar result contract differs")
    boundary = _mapping(value.get("information_boundary"), label="scalar boundary")
    _require(
        boundary.get("same_trajectory_label_used_for_its_scalar") is False,
        "scalar result leaked the held-out label into its own fold",
    )
    _require(
        boundary.get("pyelastica_high_dimensional_refit") is False,
        "scalar result used a high-dimensional PyElastica refit",
    )
    _require(boundary.get("deform_refit") is False, "scalar result changed DEFORM")
    _require(
        boundary.get("dlo3_official_evaluation_read") is False,
        "scalar result opened DLO3 official evaluation",
    )
    _require(
        boundary.get("dlo4_or_dlo5_read") is False, "scalar result read reserve DLOs"
    )
    gate = _mapping(value.get("promotion_gate"), label="scalar promotion gate")
    point = _mapping(value.get("scalar_vs_raw_pyelastica"), label="scalar comparison")
    alignment = _mapping(value.get("directional_alignment"), label="alignment")
    supported = (
        value.get("decision") == "cross-backend-shared-residual-geometry-supported"
        and gate.get("supported") is True
    )
    return {
        "supported": supported,
        "decision": value.get("decision"),
        "relative_improvement": point.get("relative_improvement"),
        "wins": point.get("wins"),
        "losses": point.get("losses"),
        "maximum_case_ratio": point.get("maximum_case_ratio"),
        "positive_alignment_cases": alignment.get("positive_cases"),
        "median_alignment_cosine": alignment.get("median_cosine"),
        "backend_specific_gain_retained_fraction": value.get(
            "backend_specific_gain_retained_fraction"
        ),
    }


def _walk_relevant(
    value: object,
    *,
    path: tuple[str, ...] = (),
) -> tuple[list[tuple[str, bool]], list[tuple[str, str]]]:
    booleans: list[tuple[str, bool]] = []
    decisions: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            child_path = (*path, key)
            lower = key.casefold()
            rendered_path = ".".join(child_path)
            if lower in RELEVANT_BOOLEAN_KEYS and isinstance(item, bool):
                booleans.append((rendered_path, item))
            if "decision" in lower and isinstance(item, str):
                decisions.append((rendered_path, item))
            child_booleans, child_decisions = _walk_relevant(item, path=child_path)
            booleans.extend(child_booleans)
            decisions.extend(child_decisions)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            child_booleans, child_decisions = _walk_relevant(
                item, path=(*path, str(index))
            )
            booleans.extend(child_booleans)
            decisions.extend(child_decisions)
    return booleans, decisions


def _choose_dlo45_result(
    retained_root: Path,
) -> tuple[Path | None, dict[str, object] | None]:
    candidates: list[tuple[int, Path, dict[str, object]]] = []
    for path in sorted(retained_root.rglob("result.json")):
        try:
            value = _load_json(path, label="DLO45 result candidate")
        except (ClaimBundleError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        normalized = path.as_posix().casefold()
        contract = str(value.get("contract", value.get("schema", ""))).casefold()
        score = 0
        if "/score/" in normalized or path.parent.name.casefold() == "score":
            score += 100
        if "dlo45" in contract or "dlo4" in contract or "dlo5" in contract:
            score += 40
        if "decision" in value:
            score += 20
        if (
            "dlo4" in json.dumps(value).casefold()
            and "dlo5" in json.dumps(value).casefold()
        ):
            score += 10
        candidates.append((score, path, value))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: (-item[0], item[1].as_posix()))
    _, path, value = candidates[0]
    return path, value


def _interpret_dlo45(value: Mapping[str, object] | None) -> dict[str, object]:
    if value is None:
        return {
            "status": "manual-review-required",
            "reason": "no retained DLO45 score result was found",
            "result_contract": None,
            "decisions": [],
            "relevant_booleans": [],
        }
    booleans, decisions = _walk_relevant(value)
    decision_values = [text.casefold() for _, text in decisions]
    has_negative_decision = any(
        token in text for text in decision_values for token in NEGATIVE_DECISION_TOKENS
    )
    has_positive_decision = any(
        token in text for text in decision_values for token in POSITIVE_DECISION_TOKENS
    )
    dlo4 = [(path, flag) for path, flag in booleans if "dlo4" in path.casefold()]
    dlo5 = [(path, flag) for path, flag in booleans if "dlo5" in path.casefold()]
    any_false = any(not flag for _, flag in (*dlo4, *dlo5))
    both_explicitly_positive = (
        bool(dlo4)
        and bool(dlo5)
        and all(flag for _, flag in dlo4)
        and all(flag for _, flag in dlo5)
    )
    serialized = json.dumps(value, sort_keys=True).casefold()
    names_both_dlos = "dlo4" in serialized and "dlo5" in serialized
    if has_negative_decision or any_false:
        status = "not-supported"
        reason = "a retained decision or DLO-specific gate is explicitly negative"
    elif both_explicitly_positive:
        status = "supported"
        reason = "DLO4 and DLO5 each have explicit positive machine-readable gates"
    elif (
        has_positive_decision
        and names_both_dlos
        and booleans
        and all(flag for _, flag in booleans)
    ):
        status = "supported"
        reason = (
            "the retained joint decision is positive and all explicit gates are true"
        )
    else:
        status = "manual-review-required"
        reason = (
            "the retained schema does not provide an unambiguous two-DLO positive gate"
        )
    return {
        "status": status,
        "reason": reason,
        "result_contract": value.get("contract", value.get("schema")),
        "decisions": [{"path": path, "value": text} for path, text in decisions],
        "relevant_booleans": [{"path": path, "value": flag} for path, flag in booleans],
    }


def _claim_text(
    *,
    tier: int,
    dlo45: Mapping[str, object],
    direct: Mapping[str, object],
    scalar: Mapping[str, object],
) -> dict[str, str]:
    limitation = (
        "The cross-backend analyses are retrospective DLO3 source-panel diagnostics; "
        "they do not establish arbitrary-backend transfer, zero-shot object "
        "generalization, calibrated target uncertainty, deployment safety, or state "
        "of the art."
    )
    if tier == 1:
        headline = (
            "Frozen discrepancy correction replicates across additional official DLO "
            "operators and transfers without coefficient refitting across physical "
            "backends."
        )
        result_sentence = (
            "The protected DLO4/DLO5 replication had an explicit positive joint gate, "
            f"while unchanged DEFORM-fitted coefficients improved sealed PyElastica "
            f"predictions by {100.0 * float(direct['relative_improvement']):.2f}% with "
            f"{direct['wins']} complete-trajectory wins."
        )
        contribution = (
            "We demonstrate both procedure-level replication on additional official "
            "deformable-object operators and coefficient-level no-refit transfer to an "
            "independently implemented physical backend."
        )
    elif tier == 2:
        headline = (
            "Frozen discrepancy correction replicates across additional official DLO "
            "operators, and its residual geometry transfers across physical backends "
            "with one-dimensional amplitude recalibration."
        )
        result_sentence = (
            "The protected DLO4/DLO5 replication had an explicit positive joint gate, "
            f"and the cross-validated scalar transport improved PyElastica by "
            f"{100.0 * float(scalar['relative_improvement']):.2f}% with "
            f"{scalar['wins']} held-out trajectory wins and median residual cosine "
            f"{float(scalar['median_alignment_cosine']):.3f}."
        )
        contribution = (
            "We show procedure-level replication across additional official DLO "
            "operators and a shared high-dimensional discrepancy direction across two "
            "physical backends, requiring only one scalar amplitude calibration."
        )
    elif tier == 3:
        headline = (
            "DEFORM-fitted discrepancy coefficients transfer without refitting to an "
            "independently implemented PyElastica backend on the DLO3 source panel."
        )
        result_sentence = (
            f"Unchanged equal-seed DEFORM coefficients improved sealed PyElastica "
            f"predictions by {100.0 * float(direct['relative_improvement']):.2f}% with "
            f"{direct['wins']} complete-trajectory wins."
        )
        contribution = (
            "We provide coefficient-level evidence that part of the learned discrepancy "
            "is shared across independently implemented physical backends."
        )
    elif tier == 4:
        headline = (
            "The DEFORM discrepancy field exhibits residual geometry shared with "
            "PyElastica after one-dimensional amplitude recalibration."
        )
        result_sentence = (
            f"Cross-validated scalar transport improved PyElastica by "
            f"{100.0 * float(scalar['relative_improvement']):.2f}% with "
            f"{scalar['wins']} held-out trajectory wins and median residual cosine "
            f"{float(scalar['median_alignment_cosine']):.3f}."
        )
        contribution = (
            "We identify a shared high-dimensional residual direction across two "
            "physical backends while explicitly separating transferable geometry from "
            "backend-specific amplitude."
        )
    else:
        headline = "No stronger cross-operator or cross-backend claim is authorized."
        result_sentence = (
            "The registered evidence gates did not jointly support a stronger claim; "
            "the existing per-backend procedure-portability statement remains the "
            "defensible boundary."
        )
        contribution = (
            "The negative or ambiguous evidence is retained as a boundary on "
            "generalization rather than converted into a positive claim."
        )
    return {
        "headline": headline,
        "abstract_sentence": result_sentence,
        "contribution_bullet": contribution,
        "results_sentence": result_sentence,
        "limitation_sentence": limitation,
        "dlo45_interpretation": str(dlo45["status"]),
    }


def build_bundle(
    *,
    direct_path: Path,
    scalar_path: Path,
    collection_path: Path,
    dlo45_retained_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Validate evidence and produce a conservative claim recommendation."""

    destination = output_root.resolve()
    _require(not destination.exists(), "claim output root already exists")
    direct_value = _load_json(direct_path, label="direct result")
    scalar_value = _load_json(scalar_path, label="scalar result")
    collection_value = _load_json(collection_path, label="DLO45 collection")
    _require(
        collection_value.get("schema") == COLLECTION_SCHEMA,
        "DLO45 collection schema differs",
    )
    run = _mapping(collection_value.get("run"), label="DLO45 collected run")
    _require(int(run.get("id", -1)) == 33361441865, "DLO45 run identity differs")
    _require(run.get("status") == "completed", "DLO45 run is not terminal")
    _require(
        _mapping(
            collection_value.get("information_boundary"),
            label="collection boundary",
        ).get("self_hosted_runner_used")
        is False,
        "DLO45 collection unexpectedly used a self-hosted runner",
    )

    direct = _validate_direct(direct_value)
    scalar = _validate_scalar(scalar_value)
    selected_path, selected_dlo45 = _choose_dlo45_result(
        dlo45_retained_root.resolve(strict=True)
    )
    dlo45 = _interpret_dlo45(selected_dlo45)
    dlo45_supported = dlo45["status"] == "supported"
    if dlo45_supported and direct["supported"]:
        tier = 1
        tier_name = "cross-operator-replication-and-exact-cross-backend-transfer"
    elif dlo45_supported and scalar["supported"]:
        tier = 2
        tier_name = "cross-operator-replication-and-shared-cross-backend-geometry"
    elif direct["supported"]:
        tier = 3
        tier_name = "exact-cross-backend-transfer-only"
    elif scalar["supported"]:
        tier = 4
        tier_name = "shared-cross-backend-geometry-only"
    else:
        tier = 0
        tier_name = "no-stronger-claim"
    text = _claim_text(tier=tier, dlo45=dlo45, direct=direct, scalar=scalar)

    destination.mkdir(parents=True)
    bundle: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "recommended_claim_tier": tier,
        "recommended_claim_tier_name": tier_name,
        "claim_ready_for_manual_authorization": tier > 0,
        "paper_claim_authorized": False,
        "evidence": {
            "direct": {"identity": _identity(direct_path), **direct},
            "scalar": {"identity": _identity(scalar_path), **scalar},
            "dlo45_collection": _identity(collection_path),
            "dlo45_selected_result": (
                None
                if selected_path is None
                else {
                    "identity": _identity(selected_path),
                    "relative_path": selected_path.resolve()
                    .relative_to(dlo45_retained_root.resolve())
                    .as_posix(),
                }
            ),
            "dlo45_interpretation": dlo45,
        },
        "paper_text": text,
        "claim_boundary": (
            "This bundle recommends bounded wording from machine-readable gates but "
            "does not authorize manuscript use. A human author must verify the DLO45 "
            "result schema, retained prediction seals, and compatibility with the "
            "previously retained DLO2/DLO3 evidence before adopting any combined claim."
        ),
    }
    (destination / "claim.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (destination / "evidence-table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["evidence", "status", "decision", "primary_result"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "evidence": "DLO4/DLO5 protected replication",
                "status": dlo45["status"],
                "decision": "; ".join(
                    str(item["value"]) for item in dlo45["decisions"]
                ),
                "primary_result": dlo45["reason"],
            }
        )
        writer.writerow(
            {
                "evidence": "exact DEFORM-to-PyElastica transfer",
                "status": "supported" if direct["supported"] else "not-supported",
                "decision": direct["decision"],
                "primary_result": (
                    f"relative_improvement={direct['relative_improvement']}; "
                    f"wins={direct['wins']}; max_ratio={direct['maximum_case_ratio']}"
                ),
            }
        )
        writer.writerow(
            {
                "evidence": "one-scalar shared residual geometry",
                "status": "supported" if scalar["supported"] else "not-supported",
                "decision": scalar["decision"],
                "primary_result": (
                    f"relative_improvement={scalar['relative_improvement']}; "
                    f"wins={scalar['wins']}; "
                    f"median_cosine={scalar['median_alignment_cosine']}"
                ),
            }
        )
    report_lines = [
        "# Stronger BayesianPhysTwin paper claim bundle",
        "",
        f"- Recommended tier: **{tier} — `{tier_name}`**",
        f"- Ready for manual authorization: **{tier > 0}**",
        "- Paper claim authorized by this tool: **false**",
        "",
        "## Recommended headline",
        "",
        text["headline"],
        "",
        "## Abstract/results sentence",
        "",
        text["abstract_sentence"],
        "",
        "## Contribution bullet",
        "",
        text["contribution_bullet"],
        "",
        "## Required limitation",
        "",
        text["limitation_sentence"],
        "",
        "## Evidence decisions",
        "",
        f"- DLO4/DLO5: `{dlo45['status']}` — {dlo45['reason']}",
        f"- Exact no-refit transfer: `{direct['supported']}` — `{direct['decision']}`",
        f"- One-scalar residual geometry: `{scalar['supported']}` — `{scalar['decision']}`",
        "",
        "## Claim boundary",
        "",
        str(bundle["claim_boundary"]),
    ]
    (destination / "paper-text.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    checksum_rows = []
    for path in sorted(destination.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            checksum_rows.append(f"{_sha256(path)}  {path.name}")
    (destination / "SHA256SUMS").write_text(
        "\n".join(checksum_rows) + "\n", encoding="utf-8"
    )
    return bundle


def _fixture_direct(*, supported: bool) -> dict[str, object]:
    return {
        "contract": DIRECT_CONTRACT,
        "decision": (
            "no-refit-cross-backend-transfer-supported"
            if supported
            else "no-refit-cross-backend-transfer-not-supported"
        ),
        "promotion_gate": {
            "supported": supported,
            "improving_seed_models": 3 if supported else 0,
        },
        "primary_vs_raw_pyelastica": {
            "relative_improvement": 0.12 if supported else -0.02,
            "wins": 8 if supported else 1,
            "losses": 0 if supported else 7,
            "maximum_case_ratio": 0.95 if supported else 1.3,
        },
        "backend_specific_gain_retained_fraction": 0.5,
        "information_boundary": {
            "pyelastica_refit": False,
            "deform_refit": False,
            "dlo3_official_evaluation_read": False,
            "dlo4_or_dlo5_read": False,
        },
    }


def _fixture_scalar(*, supported: bool) -> dict[str, object]:
    return {
        "contract": SCALAR_CONTRACT,
        "decision": (
            "cross-backend-shared-residual-geometry-supported"
            if supported
            else "cross-backend-shared-residual-geometry-not-supported"
        ),
        "promotion_gate": {"supported": supported},
        "scalar_vs_raw_pyelastica": {
            "relative_improvement": 0.09 if supported else 0.0,
            "wins": 7 if supported else 0,
            "losses": 1 if supported else 0,
            "maximum_case_ratio": 1.02 if supported else 1.0,
        },
        "directional_alignment": {
            "positive_cases": 7 if supported else 2,
            "median_cosine": 0.3 if supported else -0.1,
        },
        "backend_specific_gain_retained_fraction": 0.4,
        "information_boundary": {
            "same_trajectory_label_used_for_its_scalar": False,
            "pyelastica_high_dimensional_refit": False,
            "deform_refit": False,
            "dlo3_official_evaluation_read": False,
            "dlo4_or_dlo5_read": False,
        },
    }


def _self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        direct_path = root / "direct.json"
        scalar_path = root / "scalar.json"
        collection_path = root / "collection.json"
        retained = root / "retained" / "artifact" / "score"
        retained.mkdir(parents=True)
        direct_path.write_text(json.dumps(_fixture_direct(supported=True)))
        scalar_path.write_text(json.dumps(_fixture_scalar(supported=True)))
        collection_path.write_text(
            json.dumps(
                {
                    "schema": COLLECTION_SCHEMA,
                    "run": {"id": 33361441865, "status": "completed"},
                    "information_boundary": {"self_hosted_runner_used": False},
                }
            )
        )
        (retained / "result.json").write_text(
            json.dumps(
                {
                    "contract": "fixture-dlo45-result-v1",
                    "decision": "replication-supported",
                    "dlo4": {"promotion_gate": {"supported": True}},
                    "dlo5": {"promotion_gate": {"supported": True}},
                }
            )
        )
        output = root / "output"
        bundle = build_bundle(
            direct_path=direct_path,
            scalar_path=scalar_path,
            collection_path=collection_path,
            dlo45_retained_root=root / "retained",
            output_root=output,
        )
        _require(bundle["recommended_claim_tier"] == 1, "fixture tier 1 failed")
        _require(bundle["paper_claim_authorized"] is False, "fixture over-authorized")

        negative_root = root / "negative-retained" / "score"
        negative_root.mkdir(parents=True)
        (negative_root / "result.json").write_text(
            json.dumps(
                {
                    "decision": "replication-not-supported",
                    "dlo4": {"supported": False},
                    "dlo5": {"supported": True},
                }
            )
        )
        direct_path.write_text(json.dumps(_fixture_direct(supported=False)))
        second = build_bundle(
            direct_path=direct_path,
            scalar_path=scalar_path,
            collection_path=collection_path,
            dlo45_retained_root=root / "negative-retained",
            output_root=root / "output-negative",
        )
        _require(second["recommended_claim_tier"] == 4, "fixture tier 4 failed")
    print("self-test passed")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-result", type=Path)
    parser.add_argument("--scalar-result", type=Path)
    parser.add_argument("--dlo45-collection", type=Path)
    parser.add_argument("--dlo45-retained-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.self_test:
        _self_test()
        return 0
    _require(args.direct_result is not None, "--direct-result is required")
    _require(args.scalar_result is not None, "--scalar-result is required")
    _require(args.dlo45_collection is not None, "--dlo45-collection is required")
    _require(args.dlo45_retained_root is not None, "--dlo45-retained-root is required")
    _require(args.output_root is not None, "--output-root is required")
    bundle = build_bundle(
        direct_path=args.direct_result,
        scalar_path=args.scalar_result,
        collection_path=args.dlo45_collection,
        dlo45_retained_root=args.dlo45_retained_root,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "recommended_claim_tier": bundle["recommended_claim_tier"],
                "recommended_claim_tier_name": bundle["recommended_claim_tier_name"],
                "claim_ready_for_manual_authorization": bundle[
                    "claim_ready_for_manual_authorization"
                ],
                "paper_claim_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ClaimBundleError as error:
        raise SystemExit(f"claim bundle failed: {error}") from error
