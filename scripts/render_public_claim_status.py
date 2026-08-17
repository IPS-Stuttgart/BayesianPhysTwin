#!/usr/bin/env python3
"""Render the release-facing README claim table from one pinned snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(
            handle,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    if not isinstance(raw, dict):
        raise ValueError("claim snapshot must be a JSON object")
    return raw


def validate_snapshot(snapshot: dict[str, Any], *, root: Path = ROOT) -> None:
    expected_fields = {
        "contract",
        "contract_version",
        "snapshot_date",
        "evidence_source",
        "claims",
        "not_authorized",
    }
    if set(snapshot) != expected_fields:
        raise ValueError(
            "public claim snapshot fields changed; "
            f"expected={sorted(expected_fields)}, observed={sorted(snapshot)}"
        )
    if snapshot.get("contract") != "bayesian-phystwin.public-claim-snapshot":
        raise ValueError("unexpected public claim snapshot contract")
    if snapshot.get("contract_version") != 1:
        raise ValueError("unexpected public claim snapshot version")
    snapshot_date = snapshot.get("snapshot_date")
    if not isinstance(snapshot_date, str):
        raise ValueError("snapshot_date must be an ISO date")
    try:
        date.fromisoformat(snapshot_date)
    except ValueError as error:
        raise ValueError("snapshot_date must be an ISO date") from error

    source = snapshot.get("evidence_source")
    if not isinstance(source, dict):
        raise ValueError("evidence_source must be an object")
    if set(source) != {"path", "git_blob_sha1", "role"}:
        raise ValueError("evidence_source fields changed")
    source_path = source.get("path")
    expected_sha = source.get("git_blob_sha1")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("evidence_source.path must be nonempty")
    if (
        not isinstance(expected_sha, str)
        or len(expected_sha) != 40
        or any(character not in "0123456789abcdef" for character in expected_sha)
    ):
        raise ValueError("evidence_source.git_blob_sha1 must be a SHA-1 digest")
    source_bytes = (root / source_path).read_bytes()
    header = f"blob {len(source_bytes)}\0".encode("ascii")
    actual_sha = hashlib.sha1(header + source_bytes).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(
            "pinned evidence source changed: "
            f"expected {expected_sha}, observed {actual_sha}"
        )

    claims = snapshot.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("claims must be a nonempty list")
    seen_ids: set[str] = set()
    seen_orders: set[int] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("each claim must be an object")
        expected_claim_fields = {
            "id",
            "table_order",
            "question",
            "status",
            "display_status",
            "boundary",
            "metrics",
        }
        if set(claim) != expected_claim_fields:
            raise ValueError("claim fields changed")
        if not isinstance(claim.get("metrics"), dict):
            raise ValueError("claim metrics must be an object")
        claim_id = claim.get("id")
        order = claim.get("table_order")
        status = claim.get("status")
        if not isinstance(claim_id, str) or not claim_id:
            raise ValueError("every claim needs a nonempty id")
        if claim_id in seen_ids:
            raise ValueError(f"duplicate claim id: {claim_id}")
        seen_ids.add(claim_id)
        if not isinstance(order, int) or order < 1 or order in seen_orders:
            raise ValueError(f"invalid or duplicate table_order for {claim_id}")
        seen_orders.add(order)
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"unsupported status for {claim_id}: {status}")
        for field in ("question", "display_status", "boundary"):
            if not isinstance(claim.get(field), str) or not claim[field].strip():
                raise ValueError(f"{claim_id}.{field} must be nonempty")

    not_authorized = snapshot.get("not_authorized")
    if (
        not isinstance(not_authorized, list)
        or not not_authorized
        or any(not isinstance(item, str) or not item for item in not_authorized)
        or len(not_authorized) != len(set(not_authorized))
    ):
        raise ValueError("not_authorized must contain unique nonempty strings")


def _escape_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def render_status_block(snapshot: dict[str, Any]) -> str:
    claims = sorted(snapshot["claims"], key=lambda item: item["table_order"])
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
            "This table is generated from "
            "[`evidence/public_claim_snapshot_v1.json`]"
            "(evidence/public_claim_snapshot_v1.json), which pins the release-facing "
            "claim contract by Git blob identity. Regenerate it with "
            "`python scripts/render_public_claim_status.py --write`; CI checks that "
            "the snapshot, source document, and README stay synchronized.",
        )
    )
    return "\n".join(lines)


def replace_status_block(readme: str, block: str) -> str:
    start = readme.find(START_MARKER)
    end = readme.find(END_MARKER)
    if start < 0 or end < 0 or end < start:
        raise ValueError("README claim-status markers are missing or malformed")
    end += len(END_MARKER)
    suffix = readme[end:]
    generated_note_start = suffix.find("\n\nThis table is generated from ")
    if generated_note_start == 0:
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
    expected = replace_status_block(
        args.readme.read_text(encoding="utf-8"),
        render_status_block(snapshot),
    )
    current = args.readme.read_text(encoding="utf-8")
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
