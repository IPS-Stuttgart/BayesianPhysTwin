"""JSON-safe execution wrapper for the DEFORM conformal regret envelope.

A split-conformal rank can exceed the number of calibration trajectories.  The
mathematically correct radius is then positive infinity, which means exact
fallback for every finite regret budget.  Strict JSON has no infinity literal.
This wrapper serializes that value as the explicit token ``"infinite"`` and
restores it before target evaluation.  It changes no protocol, score, rank,
radius, action, or target access rule.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from . import support_envelope as implementation

INFINITE_RADIUS_TOKEN: Final = "infinite"


def json_safe(value: Any) -> Any:
    """Return a strict-JSON representation retaining explicit infinity semantics."""

    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("NaN cannot be represented in scientific evidence")
        if math.isinf(value):
            if value < 0.0:
                raise ValueError("negative infinity is not a valid regret radius")
            return INFINITE_RADIUS_TOKEN
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    """Write strict JSON with explicit positive-infinity tokens."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            json_safe(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def canonical_sha256(value: Any) -> str:
    """Hash the same JSON-safe scientific representation that is persisted."""

    payload = json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def radius_from_record(
    source: Mapping[str, object],
    *,
    dlo: str,
    grouping: str,
    alpha_key: str,
    envelope_kind: str,
) -> float:
    """Read a finite radius or restore the explicit infinity token."""

    if grouping == "per_dlo":
        dlos = source.get("dlos")
        if not isinstance(dlos, dict) or not isinstance(dlos.get(dlo), dict):
            raise ValueError(f"missing source envelope for {dlo}")
        record = dlos[dlo]
    elif grouping == "pooled":
        record = source.get("pooled_normalized_regret")
    else:
        raise ValueError(f"unknown grouping {grouping!r}")
    if not isinstance(record, dict):
        raise ValueError("invalid source envelope record")
    envelopes = record.get("envelopes")
    if not isinstance(envelopes, dict):
        raise ValueError("missing source envelopes")
    level = envelopes.get(alpha_key)
    if not isinstance(level, dict):
        raise ValueError("missing source envelope level")
    kind = level.get(envelope_kind)
    if not isinstance(kind, dict):
        raise ValueError("missing source envelope kind")
    radius = kind.get("radius")
    has_finite_radius = kind.get("has_finite_radius")
    if radius == INFINITE_RADIUS_TOKEN:
        if has_finite_radius is not False:
            raise ValueError("infinite radius record lacks fail-closed flag")
        return float("inf")
    if isinstance(radius, bool) or not isinstance(radius, (int, float)):
        raise ValueError("source envelope radius is neither finite nor explicit infinity")
    result = float(radius)
    if not math.isfinite(result) or result < 0.0 or has_finite_radius is not True:
        raise ValueError("invalid finite source envelope radius")
    return result


def _install_json_contract() -> None:
    implementation.write_json = write_json
    implementation.canonical_sha256 = canonical_sha256
    implementation._radius = radius_from_record


def _run(argv: Sequence[str] | None = None) -> dict[str, object]:
    _install_json_contract()
    args = implementation._parser().parse_args(argv)
    if args.stage == "source":
        return implementation.run_source(
            parent_protocol_path=args.parent_protocol,
            envelope_protocol_path=args.envelope_protocol,
            parent_source_result_path=args.parent_source_result,
            request_path=args.request,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
        )
    return implementation.run_target(
        parent_protocol_path=args.parent_protocol,
        envelope_protocol_path=args.envelope_protocol,
        request_path=args.request,
        dataset_root=args.dataset_root,
        source_root=args.source_root,
        output_root=args.output_root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    result = _run(argv)
    print(
        json.dumps(
            json_safe(result),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
