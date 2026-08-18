#!/usr/bin/env python3
"""Render the release-facing README claim table from one pinned snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = ROOT / "evidence" / "public_claim_snapshot_v1.json"
DEFAULT_README = ROOT / "README.md"
START_MARKER = "<!-- public-claim-status:begin -->"
END_MARKER = "<!-- public-claim-status:end -->"
ALLOWED_STATUSES = frozenset(
    {
        "confirmed",
        "confirmed_with_boundary",
        "confirmed_with_cost",
        "not_confirmed",
        "refuted",
        "not_established",
        "terminal_without_claim",
    }
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def load_snapshot(path: Path) -> dict[str, Any]:
    """Load strict JSON without duplicate keys or non-finite constants."""

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    if not isinstance(value, dict):
        raise ValueError("claim snapshot must be a JSON object")
    return value


def _expect_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} fields changed; "
            f"expected={sorted(expected)}, observed={sorted(value)}"
        )


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _git_blob_sha1(path: Path) -> str:
    value = path.read_bytes()
    return hashlib.sha1(f"blob {len(value)}\0".encode("ascii") + value).hexdigest()


def validate_snapshot(snapshot: dict[str, Any], *, root: Path = ROOT) -> None:
    """Validate structure, source identity, claim order, and public boundaries."""

    _expect_fields(
        snapshot,
        {
            "contract",
            "contract_version",
            "snapshot_date",
            "evidence_source",
            "claims",
            "not_authorized",
        },
        name="public claim snapshot",
    )
    if snapshot["contract"] != "bayesian-phystwin.public-claim-snapshot":
        raise ValueError("unexpected public claim snapshot contract")
    if type(snapshot["contract_version"]) is not int or (
        snapshot["contract_version"] != 1
    ):
        raise ValueError("unexpected public claim snapshot version")
    try:
        date.fromisoformat(
            _nonempty_string(snapshot["snapshot_date"], name="snapshot_date")
        )
    except ValueError as error:
        raise ValueError("snapshot_date must be an ISO date") from error

    source = snapshot["evidence_source"]
    if not isinstance(source, dict):
        raise ValueError("evidence_source must be an object")
    _expect_fields(
        source,
        {"path", "git_blob_sha1", "role"},
        name="evidence_source",
    )
    source_path = _nonempty_string(source["path"], name="evidence_source.path")
    expected_sha = _nonempty_string(
        source["git_blob_sha1"],
        name="evidence_source.git_blob_sha1",
    )
    _nonempty_string(source["role"], name="evidence_source.role")
    if len(expected_sha) != 40 or any(
        character not in "0123456789abcdef" for character in expected_sha
    ):
        raise ValueError("evidence_source.git_blob_sha1 must be a SHA-1 digest")
    actual_sha = _git_blob_sha1(root / source_path)
    if actual_sha != expected_sha:
        raise ValueError(
            "pinned evidence source changed: "
            f"expected {expected_sha}, observed {actual_sha}"
        )

    claims = snapshot["claims"]
    if not isinstance(claims, list) or not claims:
        raise ValueError("claims must be a nonempty list")
    seen_ids: set[str] = set()
    seen_orders: set[int] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("each claim must be an object")
        _expect_fields(
            claim,
            {
                "id",
                "table_order",
                "question",
                "status",
                "display_status",
                "boundary",
                "metrics",
            },
            name="claim",
        )
        claim_id = _nonempty_string(claim["id"], name="claim.id")
        if claim_id in seen_ids:
            raise ValueError(f"duplicate claim id: {claim_id}")
        seen_ids.add(claim_id)
        order = claim["table_order"]
        if type(order) is not int or order < 1 or order in seen_orders:
            raise ValueError(f"invalid or duplicate table_order for {claim_id}")
        seen_orders.add(order)
        if claim["status"] not in ALLOWED_STATUSES:
            raise ValueError(f"unsupported status for {claim_id}: {claim['status']}")
        for field in ("question", "display_status", "boundary"):
            _nonempty_string(claim[field], name=f"{claim_id}.{field}")
        if not isinstance(claim["metrics"], dict):
            raise ValueError(f"{claim_id}.metrics must be an object")

    not_authorized = snapshot["not_authorized"]
    if not isinstance(not_authorized, list) or not not_authorized:
        raise ValueError("not_authorized must be a nonempty list")
    values = [
        _nonempty_string(item, name=f"not_authorized[{index}]")
        for index, item in enumerate(not_authorized)
    ]
    if len(values) != len(set(values)):
        raise ValueError("not_authorized contains duplicate values")


def _escape_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def render_status_block(snapshot: dict[str, Any]) -> str:
    """Render the canonical Markdown block and its provenance note."""

    claims: Sequence[Mapping[str, Any]] = sorted(
        snapshot["claims"],
        key=lambda item: item["table_order"],
    )
    lines = [
        START_MARKER,
        "| Question | Current status | Boundary |",
        "| --- | --- | --- |",
    ]
    for claim in claims:
        lines.append(
            "| "
            + " | ".join(
                (
                    _escape_cell(claim["question"]),
                    f"**{_escape_cell(claim['display_status'])}**",
                    _escape_cell(claim["boundary"]),
                )
            )
            + " |"
        )
    lines.extend(
        (
            END_MARKER,
            "",
            "This table is generated from",
            "[`evidence/public_claim_snapshot_v1.json`]"
            "(evidence/public_claim_snapshot_v1.json),",
            "which pins the release-facing claim contract by Git blob identity. "
            "Regenerate it",
            "with `python scripts/render_public_claim_status.py --write`; CI checks "
            "that the",
            "snapshot, source document, and README stay synchronized.",
        )
    )
    return "\n".join(lines)


def replace_status_block(readme: str, block: str) -> str:
    """Replace exactly one generated status block and provenance note."""

    if readme.count(START_MARKER) != 1 or readme.count(END_MARKER) != 1:
        raise ValueError("README must contain exactly one claim-status marker pair")
    start = readme.index(START_MARKER)
    end = readme.index(END_MARKER, start) + len(END_MARKER)
    suffix = readme[end:]
    if suffix.startswith("\n\nThis table is generated from"):
        next_section = suffix.find("\n\n## ", 2)
        suffix = suffix[next_section:] if next_section >= 0 else "\n"
    return readme[:start] + block + suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="update README in place")
    mode.add_argument(
        "--check",
        action="store_true",
        help="fail when README is not synchronized (default)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = load_snapshot(args.snapshot)
    validate_snapshot(snapshot, root=ROOT)
    current = args.readme.read_text(encoding="utf-8")
    expected = replace_status_block(current, render_status_block(snapshot))
    if args.write:
        args.readme.write_text(expected, encoding="utf-8")
        return 0
    if current != expected:
        raise SystemExit(
            "README public claim status is stale; run "
            "python scripts/render_public_claim_status.py --write"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
