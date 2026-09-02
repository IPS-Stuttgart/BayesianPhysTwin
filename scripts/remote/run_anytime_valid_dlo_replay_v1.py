#!/usr/bin/env python3
"""Replay terminal DLO loss pairs through the anytime-valid guard."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from bayesian_phystwin_experiments.anytime_valid_admission_v1 import (
    BettingMixtureConfig,
    MixtureBettingEProcess,
    geometric_alpha,
    symmetric_relative_gain,
)

CONTRACT = "anytime-valid-dlo-retrospective-replay-v1"
RESULT_CONTRACT = "anytime-valid-dlo-retrospective-replay-result-v1"

CANDIDATE_TOKENS = ("candidate", "corrected", "transfer", "ensemble")
BASELINE_TOKENS = ("baseline", "physical", "backend", "raw", "fallback")
METRIC_TOKENS = ("l1", "error", "rmse", "loss")
EXCLUDED_TOKENS = (
    "ratio",
    "relative",
    "improvement",
    "difference",
    "interval",
    "coverage",
    "nees",
)
NAME_KEYS = (
    "name",
    "trajectory",
    "trajectory_name",
    "case",
    "case_name",
    "id",
    "object",
)


@dataclass(frozen=True)
class PairRecord:
    name: str
    candidate: float
    baseline: float
    source_file: str
    source_group: str
    candidate_key: str
    baseline_key: str


@dataclass(frozen=True)
class PairGroup:
    source_file: str
    source_group: str
    candidate_key: str
    baseline_key: str
    records: tuple[PairRecord, ...]

    @property
    def candidate_mean(self) -> float:
        return float(np.mean([record.candidate for record in self.records]))

    @property
    def baseline_mean(self) -> float:
        return float(np.mean([record.baseline for record in self.records]))

    @property
    def relative_improvement(self) -> float:
        return 1.0 - self.candidate_mean / self.baseline_mean

    def counts(self, tolerance: float = 1e-12) -> tuple[int, int, int]:
        differences = np.asarray(
            [record.candidate - record.baseline for record in self.records],
            dtype=np.float64,
        )
        return (
            int(np.sum(differences < -tolerance)),
            int(np.sum(np.abs(differences) <= tolerance)),
            int(np.sum(differences > tolerance)),
        )


@dataclass(frozen=True)
class ExpectedGroup:
    case_count: int
    relative_improvement: float
    wins: int
    ties: int
    losses: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dlo45-root", type=Path, required=True)
    parser.add_argument("--hierarchy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def normalize_group_path(value: str) -> str:
    return re.sub(r"\[\d+\]", "[]", value)


def finite_scalar(value: object) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def metric_key(key: object, tokens: Sequence[str]) -> bool:
    normalized = normalize_key(str(key))
    return (
        any(token in normalized for token in tokens)
        and any(token in normalized for token in METRIC_TOKENS)
        and not any(token in normalized for token in EXCLUDED_TOKENS)
    )


def record_name(value: Mapping[str, object]) -> str | None:
    for key in NAME_KEYS:
        raw = value.get(key)
        if isinstance(raw, (str, int)) and str(raw):
            return str(raw)
    return None


def append_scalar_records(
    value: Mapping[str, object],
    *,
    source_file: str,
    json_path: str,
    destination: dict[tuple[str, str, str, str], list[PairRecord]],
) -> None:
    name = record_name(value)
    if name is None:
        return
    candidate_keys = [
        str(key)
        for key, raw in value.items()
        if finite_scalar(raw) and metric_key(key, CANDIDATE_TOKENS)
    ]
    baseline_keys = [
        str(key)
        for key, raw in value.items()
        if finite_scalar(raw) and metric_key(key, BASELINE_TOKENS)
    ]
    group_path = normalize_group_path(json_path)
    for candidate_key in candidate_keys:
        for baseline_key in baseline_keys:
            candidate = float(cast(Any, value[candidate_key]))
            baseline = float(cast(Any, value[baseline_key]))
            if candidate < 0.0 or baseline <= 0.0:
                continue
            group_key = (
                source_file,
                group_path,
                candidate_key,
                baseline_key,
            )
            destination[group_key].append(
                PairRecord(
                    name=name,
                    candidate=candidate,
                    baseline=baseline,
                    source_file=source_file,
                    source_group=group_path,
                    candidate_key=candidate_key,
                    baseline_key=baseline_key,
                )
            )


def numeric_sequence(value: object) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    if not all(finite_scalar(item) for item in value):
        return None
    return [float(cast(Any, item)) for item in value]


def append_parallel_array_records(
    value: Mapping[str, object],
    *,
    source_file: str,
    json_path: str,
    destination: dict[tuple[str, str, str, str], list[PairRecord]],
) -> None:
    names: list[str] | None = None
    for key in ("names", "trajectory_names", "case_names", "ids", "objects"):
        raw = value.get(key)
        if (
            isinstance(raw, list)
            and raw
            and all(isinstance(item, (str, int)) for item in raw)
        ):
            names = [str(item) for item in raw]
            break
    if names is None or len(set(names)) != len(names):
        return
    candidate_arrays = {
        str(key): sequence
        for key, raw in value.items()
        if metric_key(key, CANDIDATE_TOKENS)
        and (sequence := numeric_sequence(raw)) is not None
        and len(sequence) == len(names)
    }
    baseline_arrays = {
        str(key): sequence
        for key, raw in value.items()
        if metric_key(key, BASELINE_TOKENS)
        and (sequence := numeric_sequence(raw)) is not None
        and len(sequence) == len(names)
    }
    group_path = normalize_group_path(json_path)
    for candidate_key, candidates in candidate_arrays.items():
        for baseline_key, baselines in baseline_arrays.items():
            group_key = (
                source_file,
                group_path,
                candidate_key,
                baseline_key,
            )
            for name, candidate, baseline in zip(
                names,
                candidates,
                baselines,
                strict=True,
            ):
                if candidate < 0.0 or baseline <= 0.0:
                    continue
                destination[group_key].append(
                    PairRecord(
                        name=name,
                        candidate=candidate,
                        baseline=baseline,
                        source_file=source_file,
                        source_group=group_path,
                        candidate_key=candidate_key,
                        baseline_key=baseline_key,
                    )
                )


def walk_json(
    value: object,
    *,
    source_file: str,
    json_path: str,
    destination: dict[tuple[str, str, str, str], list[PairRecord]],
) -> None:
    if isinstance(value, Mapping):
        typed = cast(Mapping[str, object], value)
        append_scalar_records(
            typed,
            source_file=source_file,
            json_path=json_path,
            destination=destination,
        )
        append_parallel_array_records(
            typed,
            source_file=source_file,
            json_path=json_path,
            destination=destination,
        )
        for key, child in typed.items():
            walk_json(
                child,
                source_file=source_file,
                json_path=f"{json_path}.{key}",
                destination=destination,
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            walk_json(
                child,
                source_file=source_file,
                json_path=f"{json_path}[{index}]",
                destination=destination,
            )


def append_csv_records(
    path: Path,
    *,
    source_file: str,
    destination: dict[tuple[str, str, str, str], list[PairRecord]],
) -> None:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            return
        candidate_keys = [
            key for key in reader.fieldnames if metric_key(key, CANDIDATE_TOKENS)
        ]
        baseline_keys = [
            key for key in reader.fieldnames if metric_key(key, BASELINE_TOKENS)
        ]
        name_key = next(
            (key for key in reader.fieldnames if normalize_key(key) in NAME_KEYS),
            None,
        )
        if not candidate_keys or not baseline_keys or name_key is None:
            return
        rows = list(reader)
    for candidate_key in candidate_keys:
        for baseline_key in baseline_keys:
            group_key = (source_file, "$", candidate_key, baseline_key)
            for row in rows:
                try:
                    candidate = float(row[candidate_key])
                    baseline = float(row[baseline_key])
                except (KeyError, TypeError, ValueError):
                    continue
                name = row.get(name_key, "")
                if (
                    not name
                    or not math.isfinite(candidate)
                    or not math.isfinite(baseline)
                    or candidate < 0.0
                    or baseline <= 0.0
                ):
                    continue
                destination[group_key].append(
                    PairRecord(
                        name=name,
                        candidate=candidate,
                        baseline=baseline,
                        source_file=source_file,
                        source_group="$",
                        candidate_key=candidate_key,
                        baseline_key=baseline_key,
                    )
                )


def append_npz_records(
    path: Path,
    *,
    source_file: str,
    destination: dict[tuple[str, str, str, str], list[PairRecord]],
) -> None:
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return
    with archive:
        keys = set(archive.files)
        name_key = next(
            (key for key in ("names", "trajectory_names", "case_names") if key in keys),
            None,
        )
        truth_key = next(
            (key for key in ("truth", "target", "targets") if key in keys), None
        )
        candidate_key = next(
            (
                key
                for key in archive.files
                if any(token in normalize_key(key) for token in CANDIDATE_TOKENS)
                and normalize_key(key) not in {"candidate_l1", "candidate_error"}
            ),
            None,
        )
        baseline_key = next(
            (
                key
                for key in archive.files
                if any(token in normalize_key(key) for token in BASELINE_TOKENS)
                and normalize_key(key) not in {"baseline_l1", "baseline_error"}
            ),
            None,
        )
        if None in {name_key, truth_key, candidate_key, baseline_key}:
            return
        names = [
            str(item) for item in np.asarray(archive[cast(str, name_key)]).tolist()
        ]
        truth = np.asarray(archive[cast(str, truth_key)], dtype=np.float64)
        candidate = np.asarray(archive[cast(str, candidate_key)], dtype=np.float64)
        baseline = np.asarray(archive[cast(str, baseline_key)], dtype=np.float64)
    if (
        candidate.shape != baseline.shape
        or candidate.shape != truth.shape
        or candidate.ndim < 2
        or candidate.shape[0] != len(names)
        or not np.isfinite(candidate).all()
        or not np.isfinite(baseline).all()
        or not np.isfinite(truth).all()
    ):
        return
    axes = tuple(range(1, candidate.ndim))
    candidate_errors = np.mean(np.abs(candidate - truth), axis=axes)
    baseline_errors = np.mean(np.abs(baseline - truth), axis=axes)
    group_key = (
        source_file,
        "$npz",
        f"{candidate_key}_derived_l1",
        f"{baseline_key}_derived_l1",
    )
    for name, candidate_error, baseline_error in zip(
        names,
        candidate_errors,
        baseline_errors,
        strict=True,
    ):
        destination[group_key].append(
            PairRecord(
                name=name,
                candidate=float(candidate_error),
                baseline=float(baseline_error),
                source_file=source_file,
                source_group="$npz",
                candidate_key=f"{candidate_key}_derived_l1",
                baseline_key=f"{baseline_key}_derived_l1",
            )
        )


def collect_groups(root: Path) -> list[PairGroup]:
    source = root.resolve(strict=True)
    destination: dict[tuple[str, str, str, str], list[PairRecord]] = defaultdict(list)
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        source_file = str(path.relative_to(source))
        if path.suffix.casefold() == ".json" and path.stat().st_size < 25_000_000:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            walk_json(
                payload,
                source_file=source_file,
                json_path="$",
                destination=destination,
            )
        elif path.suffix.casefold() == ".csv":
            try:
                append_csv_records(
                    path,
                    source_file=source_file,
                    destination=destination,
                )
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
        elif path.suffix.casefold() == ".npz":
            append_npz_records(
                path,
                source_file=source_file,
                destination=destination,
            )

    groups: list[PairGroup] = []
    for key, records in destination.items():
        unique = {
            (record.name, record.candidate, record.baseline): record
            for record in records
        }
        ordered = sorted(unique.values(), key=lambda record: normalize_key(record.name))
        if len({record.name for record in ordered}) != len(ordered):
            continue
        if not ordered:
            continue
        groups.append(
            PairGroup(
                source_file=key[0],
                source_group=key[1],
                candidate_key=key[2],
                baseline_key=key[3],
                records=tuple(ordered),
            )
        )
    return groups


def expected_group(value: Mapping[str, object]) -> ExpectedGroup:
    return ExpectedGroup(
        case_count=int(cast(Any, value["expected_case_count"])),
        relative_improvement=float(cast(Any, value["expected_relative_improvement"])),
        wins=int(cast(Any, value["expected_wins"])),
        ties=int(cast(Any, value["expected_ties"])),
        losses=int(cast(Any, value["expected_losses"])),
    )


def canonical_records(group: PairGroup) -> tuple[tuple[str, float, float], ...]:
    return tuple(
        (normalize_key(record.name), record.candidate, record.baseline)
        for record in group.records
    )


def select_group(
    groups: Sequence[PairGroup],
    expected: ExpectedGroup,
    *,
    label: str,
) -> PairGroup:
    matches = []
    for group in groups:
        wins, ties, losses = group.counts()
        if (
            len(group.records) == expected.case_count
            and wins == expected.wins
            and ties == expected.ties
            and losses == expected.losses
            and math.isclose(
                group.relative_improvement,
                expected.relative_improvement,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            matches.append(group)
    if not matches:
        candidates = sorted(
            (
                len(group.records),
                group.relative_improvement,
                group.counts(),
                group.source_file,
                group.source_group,
                group.candidate_key,
                group.baseline_key,
            )
            for group in groups
            if len(group.records) == expected.case_count
        )
        raise ValueError(f"{label}: no exact pair group; candidates={candidates}")
    canonical = {canonical_records(group) for group in matches}
    if len(canonical) != 1:
        raise ValueError(f"{label}: multiple nonidentical exact pair groups")
    return sorted(
        matches,
        key=lambda group: (
            group.source_file,
            group.source_group,
            group.candidate_key,
            group.baseline_key,
        ),
    )[0]


def load_protocol(path: Path) -> dict[str, object]:
    payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("protocol must be a JSON object")
    if payload.get("schema_version") != 1 or payload.get("contract") != CONTRACT:
        raise ValueError("protocol identity changed")
    boundary = mapping(payload.get("information_boundary"), label="boundary")
    if (
        boundary.get("outcomes_previously_opened") is not True
        or boundary.get("retrospective_replay") is not True
        or boundary.get("order_selected_from_outcomes") is not False
        or boundary.get("fresh_validation_claim_authorized") is not False
        or boundary.get("deployment_safety_claim_authorized") is not False
        or boundary.get("paper_claim_authorized") is not False
    ):
        raise ValueError("replay information boundary changed")
    e_process = mapping(payload.get("e_process"), label="e-process")
    if (
        [float(value) for value in cast(list[object], e_process["lambdas"])]
        != [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
        or float(cast(Any, e_process["global_promotion_alpha"])) != 0.05
        or float(cast(Any, e_process["global_revocation_alpha"])) != 0.05
        or int(cast(Any, e_process["minimum_promotion_observations"])) != 5
        or int(cast(Any, e_process["minimum_revocation_observations"])) != 5
    ):
        raise ValueError("replay e-process contract changed")
    return payload


def sign_gain(record: PairRecord, tolerance: float = 1e-12) -> float:
    difference = record.candidate - record.baseline
    if difference < -tolerance:
        return 1.0
    if difference > tolerance:
        return -1.0
    return 0.0


def magnitude_gain(record: PairRecord) -> float:
    return symmetric_relative_gain(
        candidate_loss=record.candidate,
        fallback_loss=record.baseline,
    )


def run_e_process(
    scores: Sequence[float],
    *,
    alpha: float,
    minimum_observations: int,
    betting: BettingMixtureConfig,
) -> dict[str, object]:
    process = MixtureBettingEProcess(betting)
    e_values = []
    first_crossing: int | None = None
    for index, score in enumerate(scores):
        e_values.append(process.update(float(score)).e_value)
        if (
            first_crossing is None
            and index + 1 >= minimum_observations
            and process.crossed(alpha)
        ):
            first_crossing = index + 1
    return {
        "alpha": alpha,
        "threshold": 1.0 / alpha,
        "observation_count": len(scores),
        "first_crossing_observation": first_crossing,
        "crossed": first_crossing is not None,
        "final_e_value": process.e_value,
        "maximum_e_value": process.maximum_e_value,
        "e_values": e_values,
    }


def replay_guard(
    scores: Sequence[float],
    *,
    promotion_alpha: float,
    revocation_alpha: float,
    minimum_promotion: int,
    minimum_revocation: int,
    betting: BettingMixtureConfig,
) -> dict[str, object]:
    promotion = MixtureBettingEProcess(betting)
    revocation: MixtureBettingEProcess | None = None
    active = False
    promotion_observation: int | None = None
    revocation_observation: int | None = None
    deployments = []
    harmful_candidate_deployments = 0
    fallback_object = object()
    candidate_object = object()
    identity_violations = 0

    for index, raw_score in enumerate(scores):
        score = float(raw_score)
        deployed = candidate_object if active else fallback_object
        expected = candidate_object if active else fallback_object
        if deployed is not expected:
            identity_violations += 1
        deployments.append("candidate" if active else "exact-fallback")
        if active and score < 0.0:
            harmful_candidate_deployments += 1

        if not active and promotion_observation is None:
            promotion.update(score)
            if index + 1 >= minimum_promotion and promotion.crossed(promotion_alpha):
                active = True
                promotion_observation = index + 1
                revocation = MixtureBettingEProcess(betting)
        elif active:
            if revocation is None:
                raise RuntimeError("active replay lacks revocation e-process")
            revocation.update(-score)
            if revocation.count >= minimum_revocation and revocation.crossed(
                revocation_alpha
            ):
                active = False
                revocation_observation = index + 1

    return {
        "promotion_observation": promotion_observation,
        "revocation_observation": revocation_observation,
        "candidate_deployment_count": deployments.count("candidate"),
        "fallback_deployment_count": deployments.count("exact-fallback"),
        "harmful_candidate_deployment_count": harmful_candidate_deployments,
        "exact_fallback_identity_violations": identity_violations,
        "final_state": "candidate" if active else "exact-fallback",
        "deployments": deployments,
        "promotion_e_process": promotion.snapshot(),
        "revocation_e_process": None if revocation is None else revocation.snapshot(),
    }


def group_summary(group: PairGroup) -> dict[str, object]:
    wins, ties, losses = group.counts()
    return {
        "source_file": group.source_file,
        "source_group": group.source_group,
        "candidate_key": group.candidate_key,
        "baseline_key": group.baseline_key,
        "case_count": len(group.records),
        "candidate_mean": group.candidate_mean,
        "baseline_mean": group.baseline_mean,
        "relative_improvement": group.relative_improvement,
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_cases(
    path: Path,
    streams: Mapping[str, Sequence[PairRecord]],
    results: Mapping[str, Mapping[str, object]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "stream",
            "sequence_index",
            "name",
            "candidate_loss",
            "fallback_loss",
            "sign_gain",
            "symmetric_gain",
            "deployment",
            "candidate_harmful",
            "source_file",
            "source_group",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for stream_name, records in streams.items():
            deployments = cast(list[str], results[stream_name]["guard"]["deployments"])
            for index, (record, deployment) in enumerate(
                zip(records, deployments, strict=True),
                start=1,
            ):
                writer.writerow(
                    {
                        "stream": stream_name,
                        "sequence_index": index,
                        "name": record.name,
                        "candidate_loss": record.candidate,
                        "fallback_loss": record.baseline,
                        "sign_gain": sign_gain(record),
                        "symmetric_gain": magnitude_gain(record),
                        "deployment": deployment,
                        "candidate_harmful": deployment == "candidate"
                        and record.candidate > record.baseline,
                        "source_file": record.source_file,
                        "source_group": record.source_group,
                    }
                )


def write_report(path: Path, result: Mapping[str, object]) -> None:
    streams = mapping(result["streams"], label="streams")
    procedure = mapping(streams["procedure_replication"], label="procedure")
    universal = mapping(
        streams["universal_coefficient_transport"],
        label="universal",
    )
    procedure_guard = mapping(procedure["guard"], label="procedure guard")
    universal_guard = mapping(universal["guard"], label="universal guard")
    procedure_sign = mapping(procedure["sign_e_process"], label="procedure sign")
    universal_sign = mapping(universal["sign_e_process"], label="universal sign")
    lines = [
        "# Anytime-valid retrospective DLO replay v1",
        "",
        f"- Decision: **{result['decision']}**",
        "- Evidence class: retrospective shadow replay of previously opened outcomes",
        "- Primary sequential score: paired win/loss sign",
        "",
        "## Operator-specific correction procedure",
        "",
        f"- Cases: **{procedure['case_count']}**",
        f"- Wins/ties/losses: **{procedure['wins']}/{procedure['ties']}/{procedure['losses']}**",
        (
            "- First anytime-valid sign-evidence crossing: observation "
            f"**{procedure_sign['first_crossing_observation']}**"
        ),
        (
            "- Guard promotion applies after observation "
            f"**{procedure_guard['promotion_observation']}**"
        ),
        f"- Post-promotion harmful candidate deployments: **{procedure_guard['harmful_candidate_deployment_count']}**",
        "",
        "## Unchanged DLO3 coefficient field",
        "",
        (
            "The first eight cases are the successful no-refit PyElastica panel; "
            "the following 28 are the failed DLO4/DLO5 transfer panel."
        ),
        f"- Primary sign process crossed: **{universal_sign['crossed']}**",
        (
            "- Candidate active at the cross-operator boundary: "
            f"**{universal['candidate_active_at_shift_boundary']}**"
        ),
        f"- Cross-operator harmful candidate deployments: **{universal['cross_operator_harmful_candidate_deployments']}**",
        f"- Final deployment state: **{universal_guard['final_state']}**",
        "",
        "## Exact fallback",
        "",
        f"- Total object-identity violations: **{result['exact_fallback_identity_violations']}**",
        "",
        "## Interpretation",
        "",
        str(result["interpretation"]),
        "",
        "## Claim boundary",
        "",
        str(result["claim_boundary"]),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    artifact_contracts = mapping(protocol["artifacts"], label="artifacts")
    dlo45_contract = mapping(
        artifact_contracts["dlo45_prospective_target"],
        label="DLO45 artifact",
    )
    hierarchy_contract = mapping(
        artifact_contracts["hierarchical_transfer"],
        label="hierarchy artifact",
    )
    dlo45_groups = collect_groups(args.dlo45_root)
    hierarchy_groups = collect_groups(args.hierarchy_root)
    procedure_group = select_group(
        dlo45_groups,
        expected_group(dlo45_contract),
        label="DLO4/DLO5 procedure target",
    )
    pyelastica_group = select_group(
        hierarchy_groups,
        expected_group(mapping(hierarchy_contract["pyelastica"], label="PyElastica")),
        label="PyElastica exact coefficient transfer",
    )
    cross_operator_group = select_group(
        hierarchy_groups,
        expected_group(
            mapping(hierarchy_contract["cross_operator"], label="cross operator")
        ),
        label="DLO3-to-DLO45 exact coefficient transfer",
    )

    e_contract = mapping(protocol["e_process"], label="e-process")
    betting = BettingMixtureConfig(
        lambdas=tuple(
            float(value) for value in cast(list[object], e_contract["lambdas"])
        )
    )
    total_promotion_alpha = float(cast(Any, e_contract["global_promotion_alpha"]))
    total_revocation_alpha = float(cast(Any, e_contract["global_revocation_alpha"]))
    promotion_alpha = geometric_alpha(total_promotion_alpha, 0)
    revocation_alpha = geometric_alpha(total_revocation_alpha, 0)
    minimum_promotion = int(cast(Any, e_contract["minimum_promotion_observations"]))
    minimum_revocation = int(cast(Any, e_contract["minimum_revocation_observations"]))

    procedure_records = procedure_group.records
    universal_records = pyelastica_group.records + cross_operator_group.records
    stream_records = {
        "procedure_replication": procedure_records,
        "universal_coefficient_transport": universal_records,
    }
    stream_results: dict[str, dict[str, object]] = {}
    for name, records in stream_records.items():
        sign_scores = [sign_gain(record) for record in records]
        magnitude_scores = [magnitude_gain(record) for record in records]
        stream_results[name] = {
            **group_summary(
                procedure_group if name == "procedure_replication" else pyelastica_group
            ),
            "case_count": len(records),
            "sign_e_process": run_e_process(
                sign_scores,
                alpha=promotion_alpha,
                minimum_observations=minimum_promotion,
                betting=betting,
            ),
            "symmetric_gain_e_process": run_e_process(
                magnitude_scores,
                alpha=promotion_alpha,
                minimum_observations=minimum_promotion,
                betting=betting,
            ),
            "single_epoch_full_alpha_sign_e_process": run_e_process(
                sign_scores,
                alpha=float(
                    cast(
                        Any,
                        mapping(protocol["comparators"], label="comparators")[
                            "single_epoch_full_alpha"
                        ],
                    )
                ),
                minimum_observations=minimum_promotion,
                betting=betting,
            ),
            "guard": replay_guard(
                sign_scores,
                promotion_alpha=promotion_alpha,
                revocation_alpha=revocation_alpha,
                minimum_promotion=minimum_promotion,
                minimum_revocation=minimum_revocation,
                betting=betting,
            ),
        }

    procedure = stream_results["procedure_replication"]
    universal = stream_results["universal_coefficient_transport"]
    universal_guard = cast(dict[str, object], universal["guard"])
    shift_boundary = int(
        cast(
            Any,
            mapping(
                mapping(protocol["streams"], label="streams")[
                    "universal_coefficient_transport"
                ],
                label="universal stream",
            )["shift_boundary_after_case"],
        )
    )
    deployments = cast(list[str], universal_guard["deployments"])
    candidate_active_at_boundary = (
        shift_boundary < len(deployments) and deployments[shift_boundary] == "candidate"
    )
    cross_harm = sum(
        deployment == "candidate" and sign_gain(record) < 0.0
        for deployment, record in zip(
            deployments[shift_boundary:],
            cross_operator_group.records,
            strict=True,
        )
    )
    universal["pyelastica_group"] = group_summary(pyelastica_group)
    universal["cross_operator_group"] = group_summary(cross_operator_group)
    universal["candidate_active_at_shift_boundary"] = candidate_active_at_boundary
    universal["cross_operator_harmful_candidate_deployments"] = cross_harm

    exact_violations = sum(
        int(
            cast(
                Any,
                cast(dict[str, object], value["guard"])[
                    "exact_fallback_identity_violations"
                ],
            )
        )
        for value in stream_results.values()
    )
    expected = mapping(
        protocol["expected_mechanism_outcomes"],
        label="expected outcomes",
    )
    procedure_sign = cast(dict[str, object], procedure["sign_e_process"])
    checks = {
        "procedure_sign_process_promotes": bool(procedure_sign["crossed"])
        is bool(expected["procedure_sign_process_promotes"]),
        "procedure_post_promotion_losses": int(
            cast(
                Any,
                cast(dict[str, object], procedure["guard"])[
                    "harmful_candidate_deployment_count"
                ],
            )
        )
        == int(cast(Any, expected["procedure_post_promotion_losses"])),
        "universal_field_not_active_at_cross_operator_boundary": (
            not candidate_active_at_boundary
        )
        is bool(expected["universal_field_not_active_at_cross_operator_boundary"]),
        "universal_field_harmful_cross_operator_deployments": cross_harm
        == int(
            cast(
                Any,
                expected["universal_field_harmful_cross_operator_deployments"],
            )
        ),
        "exact_fallback_identity": exact_violations
        <= int(cast(Any, expected["maximum_exact_fallback_identity_violations"])),
    }
    supported = all(checks.values())
    result = {
        "schema_version": 1,
        "contract": RESULT_CONTRACT,
        "status": "complete",
        "decision": (
            "retrospective-anytime-dlo-mechanism-supported"
            if supported
            else "retrospective-anytime-dlo-mechanism-gate-failed"
        ),
        "streams": stream_results,
        "mechanism_gate": {"passed": supported, "checks": checks},
        "exact_fallback_identity_violations": exact_violations,
        "interpretation": (
            "Under the frozen reveal order, directional evidence for the "
            "operator-specific correction procedure earns promotion, whereas "
            "the unchanged DLO3 coefficient field remains on exact fallback "
            "before the DLO4/DLO5 transport failures arrive. This is a "
            "retrospective illustration of evidence accumulation and refusal, "
            "not fresh validation of the stopping rule."
        ),
        "information_boundary": protocol["information_boundary"],
        "claim_boundary": protocol["claim_boundary"],
        "selected_groups": {
            "procedure": group_summary(procedure_group),
            "pyelastica": group_summary(pyelastica_group),
            "cross_operator": group_summary(cross_operator_group),
        },
        "protocol": protocol,
    }
    write_json(output_root / "result.json", result)
    write_report(output_root / "report.md", result)
    write_cases(output_root / "case-sequence.csv", stream_records, stream_results)
    print(json.dumps(result["mechanism_gate"], indent=2, sort_keys=True))
    return 0 if supported else 2


if __name__ == "__main__":
    raise SystemExit(main())
