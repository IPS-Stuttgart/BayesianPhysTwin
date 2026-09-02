#!/usr/bin/env python3
"""Replay a fixed DLO3 correction across PyElastica and fresh DLO operators."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from bayesian_phystwin_experiments.anytime_valid_admission_v1 import (
    AnytimeAdmissionConfig,
    AnytimeAdmissionController,
    DeploymentState,
)


@dataclass(frozen=True)
class ArrayRecord:
    path: Path
    key: str
    values: np.ndarray
    digest: str

    @property
    def label(self) -> str:
        return f"{self.path.as_posix()}::{self.key}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dlo45-artifact", type=Path, required=True)
    parser.add_argument("--hierarchical-artifact", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported shift-replay schema")
    if payload.get("contract") != "anytime-valid-dlo-backend-operator-shift-replay-v1":
        raise ValueError("unexpected shift-replay contract")
    if payload.get("status") != "frozen-before-retrospective-replay-execution":
        raise ValueError("shift-replay protocol is not frozen")
    boundary = payload["information_boundary"]
    if (
        boundary.get("all_prediction_outcomes_previously_opened") is not True
        or boundary.get("target_tuning") is not False
        or boundary.get("new_physical_acquisition") is not False
        or boundary.get("fresh_sequential_confirmation") is not False
        or boundary.get("paper_claim_authorized") is not False
    ):
        raise ValueError("information boundary changed")
    stream = payload["stream"]
    if stream.get("allow_reentry") is not False:
        raise ValueError("registered replay is non-reentrant")
    if not math.isclose(float(stream["loss_cap_m"]), 0.05):
        raise ValueError("registered loss cap changed")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize(path: Path, destination: Path) -> Path:
    source = path.resolve(strict=True)
    destination.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        return source
    if not zipfile.is_zipfile(source):
        raise ValueError(f"artifact is neither a directory nor a ZIP: {source}")
    with zipfile.ZipFile(source) as archive:
        archive.extractall(destination)
    return destination


def array_digest(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(json.dumps(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def collect_arrays(*roots: Path) -> list[ArrayRecord]:
    records: list[ArrayRecord] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for root in roots:
        for path in sorted(root.rglob("*.npz")):
            try:
                with np.load(path, allow_pickle=False) as archive:
                    for key in archive.files:
                        raw = np.asarray(archive[key])
                        if (
                            raw.ndim < 2
                            or raw.size <= 100
                            or not np.issubdtype(raw.dtype, np.number)
                        ):
                            continue
                        values = raw.astype(np.float64, copy=False)
                        if not np.isfinite(values).all():
                            continue
                        digest = array_digest(values)
                        identity = (digest, values.shape)
                        if identity in seen:
                            continue
                        seen.add(identity)
                        records.append(
                            ArrayRecord(
                                path=path.relative_to(root.parent),
                                key=key,
                                values=values,
                                digest=digest,
                            )
                        )
            except (OSError, ValueError, zipfile.BadZipFile):
                continue
    if not records:
        raise ValueError("no numeric prediction arrays found in retained artifacts")
    return records


def mean_l1(prediction: np.ndarray, truth: np.ndarray) -> float:
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth shapes differ")
    return float(np.mean(np.abs(prediction - truth)))


def canonical(values: np.ndarray) -> np.ndarray:
    result = values
    if result.ndim == 3:
        result = result[None, ...]
    if result.ndim != 4:
        raise ValueError(f"expected a case/time/node/coordinate array, got {result.shape}")
    if result.shape[1] < 20 and result.shape[2] > 100:
        result = np.swapaxes(result, 1, 2)
    if result.shape[1] < 100 or result.shape[-1] != 3:
        raise ValueError(f"array does not have a plausible DLO time/coordinate layout: {result.shape}")
    return result


def case_errors(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    prediction = canonical(prediction)
    truth = canonical(truth)
    if prediction.shape != truth.shape:
        raise ValueError("canonical prediction and truth shapes differ")
    return np.mean(np.abs(prediction - truth), axis=(1, 2, 3))


def semantic_score(record: ArrayRecord, terms: tuple[str, ...]) -> int:
    label = record.label.casefold()
    return sum(term in label for term in terms)


def identify_triple(
    records: list[ArrayRecord],
    *,
    expected_baseline: float,
    expected_candidate: float,
    tolerance: float,
    case_count: int,
    context_terms: tuple[str, ...],
) -> tuple[ArrayRecord, ArrayRecord, ArrayRecord]:
    by_shape: dict[tuple[int, ...], list[ArrayRecord]] = {}
    for record in records:
        try:
            values = canonical(record.values)
        except ValueError:
            continue
        if values.shape[0] == case_count:
            by_shape.setdefault(values.shape, []).append(record)

    matches: list[tuple[int, float, ArrayRecord, ArrayRecord, ArrayRecord]] = []
    for shape_records in by_shape.values():
        if len(shape_records) > 40:
            continue
        for truth, baseline, candidate in itertools.permutations(shape_records, 3):
            baseline_error = mean_l1(canonical(baseline.values), canonical(truth.values))
            candidate_error = mean_l1(canonical(candidate.values), canonical(truth.values))
            discrepancy = abs(baseline_error - expected_baseline) + abs(
                candidate_error - expected_candidate
            )
            if discrepancy > tolerance:
                continue
            semantic = sum(
                semantic_score(record, context_terms)
                for record in (truth, baseline, candidate)
            )
            matches.append((semantic, discrepancy, truth, baseline, candidate))
    if not matches:
        raise ValueError(
            "no array triple matches the registered aggregate identity: "
            f"baseline={expected_baseline}, candidate={expected_candidate}"
        )
    matches.sort(key=lambda item: (-item[0], item[1], item[2].label, item[3].label, item[4].label))
    best = matches[0]
    if len(matches) > 1 and matches[1][:2] == best[:2]:
        first = tuple(record.digest for record in best[2:])
        second = tuple(record.digest for record in matches[1][2:])
        if first != second:
            raise ValueError("aggregate identity does not uniquely select an array triple")
    return best[2], best[3], best[4]


def identify_cross_operator_candidates(
    records: list[ArrayRecord],
    *,
    dlo4: tuple[ArrayRecord, ArrayRecord, ArrayRecord],
    dlo5: tuple[ArrayRecord, ArrayRecord, ArrayRecord],
    expected_relative_improvement: float,
    tolerance: float = 1e-8,
) -> tuple[ArrayRecord, ArrayRecord]:
    terms = (
        "transfer",
        "dlo3",
        "no_refit",
        "no-refit",
        "cross_operator",
        "cross-operator",
        "equal_seed",
        "ensemble",
    )
    options: dict[str, list[tuple[int, float, int, ArrayRecord]]] = {}
    for name, triple in (("DLO4", dlo4), ("DLO5", dlo5)):
        truth, baseline, own_candidate = triple
        truth_values = canonical(truth.values)
        baseline_error = mean_l1(canonical(baseline.values), truth_values)
        baseline_cases = case_errors(baseline.values, truth.values)
        rows: list[tuple[int, float, int, ArrayRecord]] = []
        excluded = {truth.digest, baseline.digest, own_candidate.digest}
        for record in records:
            if record.digest in excluded:
                continue
            try:
                values = canonical(record.values)
            except ValueError:
                continue
            if values.shape != truth_values.shape:
                continue
            semantic = semantic_score(record, terms)
            if semantic == 0:
                continue
            error = mean_l1(values, truth_values)
            wins = int(np.sum(case_errors(values, truth_values) < baseline_cases - 1e-12))
            if error > baseline_error:
                rows.append((semantic, error, wins, record))
        if not rows:
            raise ValueError(f"no semantically identified cross-operator candidate for {name}")
        options[name] = rows

    baseline_mean = 0.5 * (
        mean_l1(canonical(dlo4[1].values), canonical(dlo4[0].values))
        + mean_l1(canonical(dlo5[1].values), canonical(dlo5[0].values))
    )
    matches: list[tuple[int, float, ArrayRecord, ArrayRecord]] = []
    for option4 in options["DLO4"]:
        for option5 in options["DLO5"]:
            candidate_mean = 0.5 * (option4[1] + option5[1])
            relative = 1.0 - candidate_mean / baseline_mean
            discrepancy = abs(relative - expected_relative_improvement)
            if discrepancy <= tolerance and option4[2] + option5[2] == 0:
                matches.append(
                    (
                        option4[0] + option5[0],
                        discrepancy,
                        option4[3],
                        option5[3],
                    )
                )
    if not matches:
        raise ValueError("cross-operator arrays do not reproduce the registered decision")
    matches.sort(key=lambda item: (-item[0], item[1], item[2].label, item[3].label))
    best = matches[0]
    if len(matches) > 1 and matches[1][:2] == best[:2]:
        if (matches[1][2].digest, matches[1][3].digest) != (
            best[2].digest,
            best[3].digest,
        ):
            raise ValueError("cross-operator candidate identity is not unique")
    return best[2], best[3]


def method_identity(records: tuple[ArrayRecord, ...]) -> list[dict[str, Any]]:
    return [
        {
            "path": record.path.as_posix(),
            "key": record.key,
            "sha256": record.digest,
            "shape": list(canonical(record.values).shape),
        }
        for record in records
    ]


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    protocol = load_protocol(protocol_path)
    work_root = args.work_root.resolve()
    output_root = args.output_root.resolve()
    if work_root.exists() and any(work_root.iterdir()):
        raise RuntimeError(f"work root is not empty: {work_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    work_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    dlo45_archive = args.dlo45_artifact.resolve(strict=True)
    hierarchical_archive = args.hierarchical_artifact.resolve(strict=True)
    expected_artifacts = protocol["artifacts"]
    for path, identity in (
        (dlo45_archive, expected_artifacts["dlo45"]),
        (hierarchical_archive, expected_artifacts["hierarchical_transfer"]),
    ):
        if path.is_file() and sha256_file(path) != identity["artifact_sha256"]:
            raise ValueError(f"artifact digest differs: {path}")

    dlo45_root = materialize(dlo45_archive, work_root / "dlo45")
    hierarchical_root = materialize(
        hierarchical_archive,
        work_root / "hierarchical",
    )
    records = collect_arrays(dlo45_root, hierarchical_root)
    identities = protocol["registered_aggregate_identities"]

    py_identity = identities["pyelastica"]
    pyelastica = identify_triple(
        records,
        expected_baseline=float(py_identity["fallback_mean_l1_m"]),
        expected_candidate=float(py_identity["candidate_mean_l1_m"]),
        tolerance=5e-4,
        case_count=int(py_identity["trajectory_count"]),
        context_terms=("pyelastica", "source", "transfer", "backend"),
    )
    dlo4_identity = identities["dlo4_procedure"]
    dlo4 = identify_triple(
        records,
        expected_baseline=float(dlo4_identity["fallback_mean_l1_m"]),
        expected_candidate=float(dlo4_identity["candidate_mean_l1_m"]),
        tolerance=5e-7,
        case_count=int(dlo4_identity["trajectory_count"]),
        context_terms=("dlo4", "target", "candidate", "physical"),
    )
    dlo5_identity = identities["dlo5_procedure"]
    dlo5 = identify_triple(
        records,
        expected_baseline=float(dlo5_identity["fallback_mean_l1_m"]),
        expected_candidate=float(dlo5_identity["candidate_mean_l1_m"]),
        tolerance=5e-7,
        case_count=int(dlo5_identity["trajectory_count"]),
        context_terms=("dlo5", "target", "candidate", "physical"),
    )
    cross_identity = identities["dlo3_coefficients_on_dlo45"]
    transfer4, transfer5 = identify_cross_operator_candidates(
        records,
        dlo4=dlo4,
        dlo5=dlo5,
        expected_relative_improvement=float(
            cross_identity["equal_dlo_relative_improvement"]
        ),
    )

    stream = protocol["stream"]
    config = AnytimeAdmissionConfig(
        alpha=float(stream["alpha"]),
        beta=float(stream["beta"]),
        loss_cap=float(stream["loss_cap_m"]),
        gain_margin=float(stream["gain_margin"]),
        harm_margin=float(stream["harm_margin"]),
        allow_reentry=bool(stream["allow_reentry"]),
        lambdas=tuple(float(value) for value in stream["lambdas"]),
    )
    controller = AnytimeAdmissionController(
        config,
        candidate_id=protocol["candidate"]["identity"],
    )
    fallback_token = object()
    candidate_token = object()
    identity_violations = 0
    event_rows: list[dict[str, Any]] = []
    admission: dict[str, Any] | None = None
    revocation: dict[str, Any] | None = None
    post_shift_exposures = 0
    post_shift_harmful_exposures = 0
    guarded_post_shift_regret = 0.0
    unguarded_post_shift_regret = 0.0
    reveal_order = 0

    panels = (
        ("PYELASTICA", pyelastica[0], pyelastica[1], pyelastica[2]),
        ("DLO4", dlo4[0], dlo4[1], transfer4),
        ("DLO5", dlo5[0], dlo5[1], transfer5),
    )
    for panel_index, (domain, truth_record, fallback_record, candidate_record) in enumerate(
        panels
    ):
        truth = canonical(truth_record.values)
        fallback = canonical(fallback_record.values)
        candidate = canonical(candidate_record.values)
        if not (truth.shape == fallback.shape == candidate.shape):
            raise ValueError(f"panel arrays do not align for {domain}")
        for case_index in range(truth.shape[0]):
            for frame_index in range(truth.shape[1]):
                reveal_order += 1
                candidate_loss = float(
                    np.mean(np.abs(candidate[case_index, frame_index] - truth[case_index, frame_index]))
                )
                fallback_loss = float(
                    np.mean(np.abs(fallback[case_index, frame_index] - truth[case_index, frame_index]))
                )
                state_before = controller.state
                selected = controller.select(
                    fallback=fallback_token,
                    candidate=candidate_token,
                )
                if state_before is DeploymentState.FALLBACK:
                    if selected is not fallback_token:
                        identity_violations += 1
                else:
                    if selected is not candidate_token:
                        raise AssertionError("candidate identity changed")
                after_shift = panel_index > 0
                regret = candidate_loss - fallback_loss
                if after_shift:
                    unguarded_post_shift_regret += regret
                    if state_before is DeploymentState.CANDIDATE:
                        post_shift_exposures += 1
                        guarded_post_shift_regret += regret
                        if regret > 0.0:
                            post_shift_harmful_exposures += 1
                record = controller.observe(
                    candidate_loss=candidate_loss,
                    fallback_loss=fallback_loss,
                )
                row = {
                    "reveal_order": reveal_order,
                    "domain": domain,
                    "case_index": case_index,
                    "frame_index": frame_index,
                    "state_before": state_before.value,
                    "event": record.event,
                    "candidate_loss_m": candidate_loss,
                    "fallback_loss_m": fallback_loss,
                    "raw_gain_m": record.raw_gain,
                    "clipped_normalized_gain": record.clipped_normalized_gain,
                    "e_value": record.e_value,
                    "boundary": record.boundary,
                    "epoch": record.epoch,
                }
                event_rows.append(row)
                if record.event == "admit" and admission is None:
                    admission = row
                if record.event == "revoke" and revocation is None:
                    revocation = row

    gates = {
        "admitted_during_pyelastica": admission is not None
        and admission["domain"] == "PYELASTICA",
        "revoked_after_operator_shift": revocation is not None
        and revocation["domain"] in {"DLO4", "DLO5"},
        "zero_exact_fallback_identity_violations": identity_violations == 0,
        "guarded_post_shift_exposure_below_unguarded": post_shift_exposures
        < sum(
            canonical(panel[1].values).shape[0]
            * canonical(panel[1].values).shape[1]
            for panel in panels[1:]
        ),
        "guarded_post_shift_regret_below_unguarded": guarded_post_shift_regret
        < unguarded_post_shift_regret,
    }
    decision = (
        "retrospective-admit-then-revoke-mechanism-supported"
        if all(gates.values())
        else "retrospective-admit-then-revoke-mechanism-not-supported"
    )
    result = {
        "schema_version": 1,
        "contract": "anytime-valid-dlo-backend-operator-shift-replay-result-v1",
        "status": "complete",
        "decision": decision,
        "gates": gates,
        "candidate_id": protocol["candidate"]["identity"],
        "observation_count": len(event_rows),
        "admission": admission,
        "revocation": revocation,
        "post_shift": {
            "guarded_candidate_exposures": post_shift_exposures,
            "guarded_harmful_candidate_exposures": post_shift_harmful_exposures,
            "guarded_cumulative_regret_m": guarded_post_shift_regret,
            "unguarded_cumulative_regret_m": unguarded_post_shift_regret,
            "regret_avoided_m": unguarded_post_shift_regret
            - guarded_post_shift_regret,
        },
        "exact_fallback_identity_violations": identity_violations,
        "selected_arrays": {
            "pyelastica": method_identity(pyelastica),
            "dlo4_procedure_identity": method_identity(dlo4),
            "dlo5_procedure_identity": method_identity(dlo5),
            "dlo4_exact_transfer_candidate": method_identity((transfer4,)),
            "dlo5_exact_transfer_candidate": method_identity((transfer5,)),
        },
        "artifact_identity": {
            "dlo45": {
                "path": str(dlo45_archive),
                "sha256": sha256_file(dlo45_archive)
                if dlo45_archive.is_file()
                else None,
            },
            "hierarchical_transfer": {
                "path": str(hierarchical_archive),
                "sha256": sha256_file(hierarchical_archive)
                if hierarchical_archive.is_file()
                else None,
            },
        },
        "evidence_class": protocol["evidence_class"],
        "claim_boundary": protocol["claim_boundary"],
        "paper_claim_authorized": False,
    }
    (output_root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_root / "events.csv").open("w", newline="", encoding="utf-8") as stream_handle:
        writer = csv.DictWriter(stream_handle, fieldnames=list(event_rows[0]))
        writer.writeheader()
        writer.writerows(event_rows)
    report = [
        "# Anytime-valid DLO backend-to-operator shift replay",
        "",
        f"- Decision: **{decision}**",
        f"- Observations: **{len(event_rows)}**",
        f"- Admission: **{admission}**",
        f"- Revocation: **{revocation}**",
        f"- Exact fallback identity violations: **{identity_violations}**",
        f"- Guarded post-shift exposures: **{post_shift_exposures}**",
        f"- Guarded post-shift harmful exposures: **{post_shift_harmful_exposures}**",
        f"- Post-shift regret avoided: **{1000.0 * (unguarded_post_shift_regret - guarded_post_shift_regret):.4f} mm-sum**",
        "",
        "## Claim boundary",
        "",
        protocol["claim_boundary"],
    ]
    (output_root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
