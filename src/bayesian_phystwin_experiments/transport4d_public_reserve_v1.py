"""Metadata-only reservation of public Deform360 objects for Transport4D.

The audit protects every object already used or reserved by the bound Deform360
protocols, reads only each remaining object's public metadata JSON, and assigns
all remaining metadata namespaces to calibration or confirmation by a frozen
hash rule.  Robot, tactile, image, geometry, and outcome payloads are never
opened by this module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

RESERVE_SCHEMA: Final = "bayesian-phystwin.transport4d_deform360_reserve"
RESERVE_VERSION: Final = 1
RESERVE_PROTOCOL_SCHEMA: Final = (
    "bayesian-phystwin.transport4d_deform360_reserve_protocol"
)
RESERVE_PROTOCOL_VERSION: Final = 1


def canonical_id(value: object) -> str:
    """Return a stable SHA-256 identifier for JSON-compatible content."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _literal_strings(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(value)
    if any(type(item) is not str or not item for item in result):
        raise ValueError(f"{name} must contain nonempty literal strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _metadata_actions(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    raw = metadata.get("sequences", metadata.get("episodes", metadata.get("takes")))
    if isinstance(raw, Mapping):
        values = [raw[key] for key in sorted(raw, key=lambda item: str(item))]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        values = list(raw)
    else:
        return ()
    actions: list[str] = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        action = value.get("action")
        if isinstance(action, str) and action.strip():
            actions.append(action.strip())
    return tuple(actions)


def validate_protocol(protocol: dict[str, Any], data_root: Path) -> None:
    if protocol.get("schema") != RESERVE_PROTOCOL_SCHEMA:
        raise ValueError("unexpected Transport4D reserve protocol schema")
    if protocol.get("schema_version") != RESERVE_PROTOCOL_VERSION:
        raise ValueError("unexpected Transport4D reserve protocol version")
    if protocol.get("status") != "frozen-before-reserve-metadata-access":
        raise ValueError("Transport4D reserve protocol is not frozen")
    if Path(str(protocol.get("dataset_root"))) != data_root:
        raise ValueError("Transport4D reserve dataset root changed")
    supplied = protocol.get("protocol_id")
    if not isinstance(supplied, str):
        raise ValueError("Transport4D reserve protocol_id is missing")
    unsigned = {key: value for key, value in protocol.items() if key != "protocol_id"}
    if canonical_id(unsigned) != supplied:
        raise ValueError("Transport4D reserve protocol_id does not match content")

    reservation = protocol.get("reservation")
    if not isinstance(reservation, dict):
        raise ValueError("reservation rule is missing")
    if reservation.get("include_every_remaining_metadata_object") is not True:
        raise ValueError("all remaining metadata objects must be reserved")
    if reservation.get("replacement_allowed") is not False:
        raise ValueError("reserve replacement must remain forbidden")
    if reservation.get("split_rule") != (
        "sha256-ranked-first-calibration-remainder-confirmation-v1"
    ):
        raise ValueError("reserve split rule changed")
    for name in (
        "minimum_calibration_objects",
        "minimum_confirmation_objects",
        "maximum_metadata_bytes",
    ):
        value = reservation.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    salt = reservation.get("split_salt")
    if type(salt) is not str or not salt:
        raise ValueError("reserve split_salt must be a nonempty string")

    boundary = protocol.get("information_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("reserve information boundary is missing")
    required_false = (
        "robot_numeric_payload_opened",
        "tactile_numeric_payload_opened",
        "camera_pixel_opened",
        "geometry_or_point_cloud_opened",
        "target_outcome_opened",
        "confirmation_authorized",
        "paper_claim_authorized",
    )
    if any(boundary.get(key) is not False for key in required_false):
        raise ValueError("reserve information boundary was weakened")


def protected_object_ids(
    action_kernel_protocol: Mapping[str, Any],
    untouched_protocol: Mapping[str, Any],
    reserve_protocol: Mapping[str, Any],
) -> tuple[str, ...]:
    upstream = reserve_protocol.get("upstream_bindings")
    if not isinstance(upstream, Mapping):
        raise ValueError("reserve upstream bindings are missing")
    action_binding = upstream.get("action_kernel_v3")
    untouched_binding = upstream.get("untouched_confirmation_v5")
    causal4d_binding = upstream.get("causal4d_deform360_holdings_v1")
    if (
        not isinstance(action_binding, Mapping)
        or not isinstance(untouched_binding, Mapping)
        or not isinstance(causal4d_binding, Mapping)
    ):
        raise ValueError("reserve upstream protocol bindings are malformed")
    if (
        action_kernel_protocol.get("schema")
        != "bayesian-phystwin/deform360-action-kernel-protocol-v3"
        or action_kernel_protocol.get("protocol_id")
        != action_binding.get("protocol_id")
    ):
        raise ValueError("bound Deform360 action-kernel protocol changed")
    if (
        untouched_protocol.get("schema")
        != "bayesian-phystwin/deform360-untouched-confirmation-protocol-v5"
        or untouched_protocol.get("protocol_id") != untouched_binding.get("protocol_id")
    ):
        raise ValueError("bound Deform360 untouched-confirmation protocol changed")
    if causal4d_binding.get("repository") != "IPS-Stuttgart/Causal4D":
        raise ValueError("Causal4D reserve repository binding changed")
    if causal4d_binding.get("path") != (
        "configs/causal4d_public/deform360_gpuserver6000_holdings_v1.json"
    ):
        raise ValueError("Causal4D reserve path binding changed")
    for field_name in ("revision", "git_blob_sha1"):
        value = causal4d_binding.get(field_name)
        if (
            type(value) is not str
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"invalid Causal4D reserve {field_name}")
    eligible = _literal_strings(
        untouched_protocol.get("eligible_object_ids"),
        name="untouched eligible_object_ids",
    )
    required_count = untouched_binding.get("eligible_object_count")
    if required_count != len(eligible):
        raise ValueError("bound untouched object count changed")
    protected = set(
        _literal_strings(
            action_kernel_protocol.get("development_object_ids"),
            name="development_object_ids",
        )
    )
    protected.update(
        _literal_strings(
            action_kernel_protocol.get("reserved_object_ids"),
            name="reserved_object_ids",
        )
    )
    protected.update(eligible)
    additional = _literal_strings(
        reserve_protocol.get("additional_protected_object_ids", []),
        name="additional protected objects",
    )
    causal4d_additional = _literal_strings(
        causal4d_binding.get("additional_protected_object_ids"),
        name="Causal4D additional protected objects",
    )
    if tuple(sorted(additional)) != tuple(sorted(causal4d_additional)):
        raise ValueError("cross-repository protected object roster differs")
    protected.update(additional)
    return tuple(sorted(protected))


def audit_deform360_transport_reserve(
    *,
    data_root: Path,
    reserve_protocol_path: Path,
    action_kernel_protocol_path: Path,
    untouched_protocol_path: Path,
) -> dict[str, Any]:
    """Reserve every remaining metadata namespace without opening numeric payloads."""

    root = data_root.resolve(strict=True)
    reserve_protocol = read_json(reserve_protocol_path)
    validate_protocol(reserve_protocol, root)
    action_kernel_protocol = read_json(action_kernel_protocol_path)
    untouched_protocol = read_json(untouched_protocol_path)
    protected = protected_object_ids(
        action_kernel_protocol,
        untouched_protocol,
        reserve_protocol,
    )
    protected_set = set(protected)

    raw_root = root / "raw-repository" / "raw"
    if not raw_root.is_dir() or raw_root.is_symlink():
        raise ValueError(f"Deform360 raw metadata root unavailable: {raw_root}")
    metadata_objects = sorted(
        path.name
        for path in raw_root.iterdir()
        if path.is_dir()
        and not path.is_symlink()
        and (path / "metadata.json").is_file()
        and not (path / "metadata.json").is_symlink()
    )
    remaining = [
        object_id for object_id in metadata_objects if object_id not in protected_set
    ]

    reservation = reserve_protocol["reservation"]
    salt = str(reservation["split_salt"])
    ranked = sorted(
        remaining,
        key=lambda object_id: (
            hashlib.sha256(f"{salt}:{object_id}".encode()).hexdigest(),
            object_id,
        ),
    )
    minimum_calibration = int(reservation["minimum_calibration_objects"])
    minimum_confirmation = int(reservation["minimum_confirmation_objects"])
    preferred_calibration_fraction = float(
        reservation["preferred_calibration_fraction"]
    )
    if not 0.0 < preferred_calibration_fraction < 1.0:
        raise ValueError("preferred calibration fraction must lie in (0, 1)")
    calibration_count = min(
        len(ranked),
        max(
            minimum_calibration,
            int(round(preferred_calibration_fraction * len(ranked))),
        ),
    )
    calibration_ids = ranked[:calibration_count]
    confirmation_ids = ranked[calibration_count:]

    rows: list[dict[str, Any]] = []
    split_by_id = {
        **{object_id: "calibration" for object_id in calibration_ids},
        **{object_id: "confirmation" for object_id in confirmation_ids},
    }
    maximum_metadata_bytes = int(reservation["maximum_metadata_bytes"])
    for object_id in ranked:
        metadata_path = raw_root / object_id / "metadata.json"
        metadata_size = int(metadata_path.stat().st_size)
        if metadata_size > maximum_metadata_bytes:
            raise ValueError(f"metadata exceeds registered byte limit: {metadata_path}")
        metadata_bytes = metadata_path.read_bytes()
        metadata = json.loads(metadata_bytes)
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata must be a JSON object: {metadata_path}")
        actions = _metadata_actions(metadata)
        rows.append(
            {
                "object_id": object_id,
                "split": split_by_id[object_id],
                "metadata_relative_path": metadata_path.relative_to(root).as_posix(),
                "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                "metadata_size_bytes": metadata_size,
                "declared_action_count": len(actions),
                "declared_actions": list(actions),
                "numeric_payload_opened": False,
            }
        )

    enough_calibration = len(calibration_ids) >= minimum_calibration
    enough_confirmation = len(confirmation_ids) >= minimum_confirmation
    ready = enough_calibration and enough_confirmation
    result: dict[str, Any] = {
        "schema": RESERVE_SCHEMA,
        "schema_version": RESERVE_VERSION,
        "protocol_id": reserve_protocol["protocol_id"],
        "status": (
            "metadata-reserve-ready"
            if ready
            else "metadata-reserve-too-small-for-registered-confirmation"
        ),
        "dataset_root": str(root),
        "metadata_object_count": len(metadata_objects),
        "protected_object_count": len(protected),
        "protected_object_ids": list(protected),
        "remaining_metadata_object_count": len(ranked),
        "calibration_object_ids": calibration_ids,
        "confirmation_object_ids": confirmation_ids,
        "calibration_object_count": len(calibration_ids),
        "confirmation_object_count": len(confirmation_ids),
        "registered_minimum_calibration_objects": minimum_calibration,
        "registered_minimum_confirmation_objects": minimum_confirmation,
        "reservation_ready": ready,
        "objects": rows,
        "information_boundary": {
            "metadata_json_opened": True,
            "directory_names_opened": True,
            "robot_numeric_payload_opened": False,
            "tactile_numeric_payload_opened": False,
            "camera_pixel_opened": False,
            "geometry_or_point_cloud_opened": False,
            "target_outcome_opened": False,
            "confirmation_authorized": False,
            "paper_claim_authorized": False,
        },
        "future_carrier_qualification": reserve_protocol[
            "future_carrier_qualification"
        ],
        "claim_boundary": reserve_protocol["claim_boundary"],
    }
    result["reserve_id"] = canonical_id(result)
    return result


def render_reserve_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Transport4D Deform360 metadata reserve",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- Metadata-bearing objects: **{result['metadata_object_count']}**",
        f"- Prior-used or protected namespaces: **{result['protected_object_count']}**",
        "- Remaining reserved namespaces: "
        f"**{result['remaining_metadata_object_count']}**",
        f"- Calibration objects: **{result['calibration_object_count']}**",
        f"- Confirmation objects: **{result['confirmation_object_count']}**",
        "",
        "| Object | Split | Declared actions | Metadata SHA-256 |",
        "|---|---|---:|---|",
    ]
    for row in result["objects"]:
        lines.append(
            f"| `{row['object_id']}` | `{row['split']}` | "
            f"{row['declared_action_count']} | `{row['metadata_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "Every metadata-bearing object outside the conservatively protected",
            "namespace is assigned before numeric payload access. Later carrier",
            "qualification may mark an assigned object support-negative, but may not",
            "replace it or move an object between calibration and confirmation.",
            "",
            str(result["claim_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)
