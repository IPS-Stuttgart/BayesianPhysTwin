#!/usr/bin/env python3
"""Bind official Deform360 action metadata to timestamp-clustered episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from itertools import permutations
from pathlib import Path
from typing import Any, Final

INPUT_SCHEMA: Final = "bayesian-phystwin/deform360-same-object-rope-pilot-result-v2"
OUTPUT_SCHEMA: Final = "bayesian-phystwin/deform360-cross-action-pair-plan-v1"
VERB_RE: Final = re.compile(r"^[a-z]+")


class BindError(ValueError):
    """Raised when the target-blind action binding cannot be certified."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BindError(message)


def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BindError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs_hook,
        )
    except BindError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BindError(f"cannot read JSON: {path}") from error
    require(type(value) is dict, f"JSON root must be an object: {path}")
    return value


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_id(value: Mapping[str, object], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def yes_no(value: object, *, field: str) -> bool:
    require(type(value) is str, f"{field} must be a string")
    normalized = value.strip().casefold()
    require(normalized in {"yes", "no"}, f"{field} must be yes/no")
    return normalized == "yes"


def action_family(action: str) -> str:
    normalized = " ".join(action.casefold().split())
    match = VERB_RE.match(normalized)
    require(match is not None, f"cannot parse action family: {action!r}")
    return match.group(0)


def contact_anchor(action: str) -> str:
    normalized = " ".join(action.casefold().split())
    tokens = set(normalized.split())
    if "edges" in tokens or "sides" in tokens:
        return "bilateral-edge"
    if "edge" in tokens or "side" in tokens:
        return "single-edge"
    if "center" in tokens or "middle" in tokens:
        return "center"
    if normalized in {"curve", "fold"}:
        return "global-shape"
    return "unresolved"


def load_sequences(path: Path) -> tuple[dict[int, dict[str, object]], str]:
    raw = path.read_bytes()
    require(
        0 < len(raw) <= 1_048_576,
        f"metadata size outside policy: {path}",
    )
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs_hook)
    require(type(value) is dict, f"metadata root must be object: {path}")
    sequences = value.get("sequences")
    require(type(sequences) is dict, f"metadata sequences missing: {path}")
    result: dict[int, dict[str, object]] = {}
    for raw_index, record in sequences.items():
        require(
            type(raw_index) is str and raw_index.isdigit(),
            "sequence key must be numeric",
        )
        require(type(record) is dict, f"sequence {raw_index} must be object")
        index = int(raw_index)
        action = record.get("action")
        require(
            type(action) is str and action.strip(),
            f"action missing for sequence {index}",
        )
        result[index] = {
            "action": " ".join(action.split()),
            "action_family": action_family(action),
            "contact_anchor": contact_anchor(action),
            "bimanual": yes_no(record.get("bimanual"), field="bimanual"),
            "nonprehensile": yes_no(
                record.get("nonprehensile"),
                field="nonprehensile",
            ),
        }
    require(result, f"no sequences in {path}")
    return result, hashlib.sha256(raw).hexdigest()


def candidate_pairs(
    object_id: str,
    episodes: Sequence[Mapping[str, object]],
    sequences: Mapping[int, Mapping[str, object]],
) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for source in episodes:
        index = int(source["episode_index"])
        require(index in sequences, f"{object_id}: no metadata sequence {index}")
        row = dict(source)
        row.update(sequences[index])
        enriched.append(row)
    eligible = [
        row for row in enriched if row.get("visuotactile_eligible") is True
    ]
    pairs: list[dict[str, object]] = []
    for source, target in permutations(eligible, 2):
        same_manuality = source["bimanual"] == target["bimanual"]
        same_contact_mode = source["nonprehensile"] == target["nonprehensile"]
        same_anchor = (
            source["contact_anchor"] == target["contact_anchor"]
            and source["contact_anchor"] != "unresolved"
        )
        different_family = source["action_family"] != target["action_family"]
        if not (
            same_manuality
            and same_contact_mode
            and same_anchor
            and different_family
        ):
            continue
        common_cameras = sorted(
            set(source["camera_streams"])
            & set(target["camera_streams"])
        )
        common_tactile = sorted(
            set(source["tactile_streams"])
            & set(target["tactile_streams"])
        )
        if len(common_cameras) < 12 or len(common_tactile) < 2:
            continue
        pair: dict[str, object] = {
            "object_id": object_id,
            "source_episode_index": int(source["episode_index"]),
            "target_episode_index": int(target["episode_index"]),
            "source_episode_key": source["episode_key"],
            "target_episode_key": target["episode_key"],
            "source_action": source["action"],
            "target_action": target["action"],
            "source_action_family": source["action_family"],
            "target_action_family": target["action_family"],
            "contact_anchor": source["contact_anchor"],
            "bimanual": source["bimanual"],
            "nonprehensile": source["nonprehensile"],
            "common_camera_count": len(common_cameras),
            "common_cameras": common_cameras,
            "common_tactile_count": len(common_tactile),
            "common_tactile_streams": common_tactile,
            "source_capture_timestamp_min": source[
                "capture_timestamp_min"
            ],
            "target_capture_timestamp_min": target[
                "capture_timestamp_min"
            ],
            "different_action_family": True,
            "same_manuality": True,
            "same_contact_mode": True,
            "same_contact_anchor": True,
            "target_future_opened": False,
        }
        pair["pair_id"] = content_id(pair, "pair_id")
        pairs.append(pair)
    pairs.sort(
        key=lambda pair: (
            int(bool(pair["nonprehensile"])),
            int(bool(pair["bimanual"])),
            -int(pair["common_camera_count"]),
            -int(pair["common_tactile_count"]),
            int(pair["source_episode_index"]),
            int(pair["target_episode_index"]),
        )
    )
    return pairs


def build_plan(
    roster: Mapping[str, object],
    metadata_root: Path,
    *,
    repository_revision: str,
    source_roster_run_id: int,
    metadata_probe_run_id: int,
) -> dict[str, object]:
    require(roster.get("schema") == INPUT_SCHEMA, "unexpected roster schema")
    require(roster.get("schema_version") == 2, "unexpected roster version")
    require(
        roster.get("result_id") == content_id(roster, "result_id"),
        "roster ID mismatch",
    )
    boundary = roster.get("information_boundary")
    require(type(boundary) is dict, "roster information boundary missing")
    require(
        boundary.get("target_future_opened") is False,
        "roster opened target future",
    )
    require(
        boundary.get("score_bearing_outcomes_used") is False,
        "roster used outcomes",
    )
    objects = roster.get("objects")
    require(type(objects) is list and len(objects) == 4, "expected four objects")
    selected: list[dict[str, object]] = []
    object_records: list[dict[str, object]] = []
    for record in objects:
        require(type(record) is dict, "malformed roster object")
        object_id = record.get("object_id")
        require(type(object_id) is str, "object ID missing")
        sequences, metadata_sha256 = load_sequences(
            metadata_root / object_id / "metadata.json"
        )
        episodes = record.get("episodes")
        require(
            type(episodes) is list and len(episodes) >= 2,
            f"{object_id}: episodes missing",
        )
        candidates = candidate_pairs(object_id, episodes, sequences)
        require(
            candidates,
            f"{object_id}: no matched-nuisance cross-action pair",
        )
        choice = candidates[0]
        selected.append(choice)
        object_records.append(
            {
                "object_id": object_id,
                "metadata_sha256": metadata_sha256,
                "available_sequence_indices": sorted(sequences),
                "bound_episode_indices": sorted(
                    int(row["episode_index"]) for row in episodes
                ),
                "candidate_pair_count": len(candidates),
                "selected_pair": choice,
            }
        )
    selected.sort(key=lambda row: str(row["object_id"]))
    expected = [
        "001-rope",
        "002-rope-silk",
        "003-cable",
        "081-stripe-rope",
    ]
    require(
        [row["object_id"] for row in selected] == expected,
        "object roster changed",
    )
    require(
        all(row["different_action_family"] is True for row in selected),
        "action family not changed",
    )
    require(
        all(row["same_manuality"] is True for row in selected),
        "manuality not matched",
    )
    require(
        all(row["same_contact_mode"] is True for row in selected),
        "contact mode not matched",
    )
    require(
        all(row["same_contact_anchor"] is True for row in selected),
        "contact anchor not matched",
    )
    plan: dict[str, object] = {
        "schema": OUTPUT_SCHEMA,
        "schema_version": 1,
        "repository_revision": repository_revision,
        "source_roster_run_id": source_roster_run_id,
        "source_roster_result_id": roster["result_id"],
        "metadata_probe_run_id": metadata_probe_run_id,
        "metadata_root": str(metadata_root),
        "object_records": object_records,
        "selected_pairs": selected,
        "summary": {
            "object_count": len(selected),
            "cross_action_pair_count": len(selected),
            "matched_manuality_pair_count": sum(
                row["same_manuality"] is True for row in selected
            ),
            "matched_contact_mode_pair_count": sum(
                row["same_contact_mode"] is True for row in selected
            ),
            "matched_contact_anchor_pair_count": sum(
                row["same_contact_anchor"] is True for row in selected
            ),
            "minimum_common_camera_count": min(
                int(row["common_camera_count"]) for row in selected
            ),
            "minimum_common_tactile_count": min(
                int(row["common_tactile_count"]) for row in selected
            ),
            "decision": (
                "four-object-matched-nuisance-cross-action-plan-ready"
            ),
        },
        "information_boundary": {
            "directory_and_filename_inventory_only": True,
            "small_metadata_json_allowed": True,
            "media_payload_decoded": False,
            "numeric_arrays_loaded": False,
            "target_future_opened": False,
            "score_bearing_outcomes_used": False,
        },
        "claim_boundary": (
            "Retrospective target-blind action-binding and pair-selection plan "
            "only. It does not establish model accuracy, calibration, physical "
            "transport, fresh confirmation, or a paper-level result."
        ),
    }
    plan["plan_id"] = content_id(plan, "plan_id")
    return plan


def write_outputs(output: Path, plan: Mapping[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "cross_action_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "selected_pairs.json").write_text(
        json.dumps(
            plan["selected_pairs"],
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "## Matched-nuisance Deform360 cross-action plan",
        "",
        f"- Decision: `{plan['summary']['decision']}`",
        f"- Plan ID: `{plan['plan_id']}`",
        (
            "- Minimum common cameras: "
            f"`{plan['summary']['minimum_common_camera_count']}`"
        ),
        (
            "- Minimum common tactile streams: "
            f"`{plan['summary']['minimum_common_tactile_count']}`"
        ),
        "- Target futures opened: `false`",
        "",
        "| Object | Source | Target | Anchor | Cameras |",
        "|---|---|---|---|---:|",
    ]
    for pair in plan["selected_pairs"]:
        lines.append(
            f"| `{pair['object_id']}` | {pair['source_action']} "
            f"(ep {pair['source_episode_index']}) | {pair['target_action']} "
            f"(ep {pair['target_episode_index']}) | {pair['contact_anchor']} | "
            f"{pair['common_camera_count']} |"
        )
    lines.extend(["", str(plan["claim_boundary"])])
    (output / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        metadata_root = root / "raw"
        objects = []
        object_ids = [
            "001-rope",
            "002-rope-silk",
            "003-cable",
            "081-stripe-rope",
        ]
        for object_id in object_ids:
            object_dir = metadata_root / object_id
            object_dir.mkdir(parents=True)
            metadata = {
                "sequences": {
                    "0": {
                        "action": "move edge",
                        "bimanual": "no",
                        "nonprehensile": "no",
                    },
                    "1": {
                        "action": "move center",
                        "bimanual": "no",
                        "nonprehensile": "no",
                    },
                    "2": {
                        "action": "lift edge",
                        "bimanual": "no",
                        "nonprehensile": "no",
                    },
                }
            }
            (object_dir / "metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            episodes = []
            for index in range(3):
                episodes.append(
                    {
                        "episode_index": index,
                        "episode_key": f"episode_{index:04d}",
                        "camera_pairs": 12,
                        "tactile_pairs": 4,
                        "camera_streams": [
                            f"cam{camera:02d}" for camera in range(12)
                        ],
                        "tactile_streams": ["tl", "tr", "rl", "rr"],
                        "capture_timestamp_min": 1000 + index * 100,
                        "visuotactile_eligible": True,
                    }
                )
            objects.append({"object_id": object_id, "episodes": episodes})
        roster: dict[str, object] = {
            "schema": INPUT_SCHEMA,
            "schema_version": 2,
            "objects": objects,
            "information_boundary": {
                "target_future_opened": False,
                "score_bearing_outcomes_used": False,
            },
        }
        roster["result_id"] = content_id(roster, "result_id")
        plan = build_plan(
            roster,
            metadata_root,
            repository_revision="a" * 40,
            source_roster_run_id=1,
            metadata_probe_run_id=2,
        )
        require(
            plan["summary"]["cross_action_pair_count"] == 4,
            "fixture count changed",
        )
        require(
            all(
                pair["source_episode_index"] == 0
                for pair in plan["selected_pairs"]
            ),
            "bad source",
        )
        require(
            all(
                pair["target_episode_index"] == 2
                for pair in plan["selected_pairs"]
            ),
            "bad target",
        )
    print("self-test passed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--repository-revision")
    parser.add_argument(
        "--source-roster-run-id",
        type=int,
        default=33335964618,
    )
    parser.add_argument(
        "--metadata-probe-run-id",
        type=int,
        default=33337433438,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    require(args.roster is not None, "--roster is required")
    require(args.metadata_root is not None, "--metadata-root is required")
    require(
        args.repository_revision is not None,
        "--repository-revision is required",
    )
    require(args.output is not None, "--output is required")
    plan = build_plan(
        read_json(args.roster),
        args.metadata_root.resolve(),
        repository_revision=args.repository_revision,
        source_roster_run_id=args.source_roster_run_id,
        metadata_probe_run_id=args.metadata_probe_run_id,
    )
    write_outputs(args.output.resolve(), plan)
    print(
        json.dumps(
            {
                "decision": plan["summary"]["decision"],
                "plan_id": plan["plan_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BindError as error:
        print(f"Deform360 action binding failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
