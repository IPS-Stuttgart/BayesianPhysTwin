from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPOSITORY = Path(__file__).resolve().parents[1]
_SCRIPT = (
    _REPOSITORY
    / "scripts"
    / "science"
    / "materialize_deform360_covariance_camera_partitions_v1.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_deform360_covariance_camera_partitions_v1",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_PROTOCOL = (
    _REPOSITORY
    / "protocols"
    / "locks"
    / "deform360_covariance_only_target_v1_5.json"
)
_EXACT_PLAN = (
    _REPOSITORY
    / "results"
    / "science"
    / "deform360_covariance_only_target_v1"
    / "exact_file_plan_v1.json"
)


def _canonical_digest(value: dict[str, object]) -> str:
    payload = dict(value)
    payload.pop("partition_sha256", None)
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def test_registered_names_only_partition_covers_all_locked_sessions() -> None:
    artifact = _MODULE.build_partition_artifact(
        protocol_path=_PROTOCOL,
        exact_plan_path=_EXACT_PLAN,
        implementation_revision="a" * 40,
    )

    assert artifact["target_count"] == 24
    assert len(artifact["rows"]) == 24
    assert artifact["partition_sha256"] == _canonical_digest(artifact)
    assert artifact["summary"]["source_plan_status_counts"] == {
        "planned": 20,
        "unsupported_without_replacement": 4,
    }
    assert artifact["information_boundary"] == {
        "camera_media_decoded": False,
        "geometry_or_tracks_opened": False,
        "names_only_exact_plan_read": True,
        "payload_path_opened": False,
        "predictions_run": False,
        "reconstructions_built": False,
        "replacement_allowed": False,
        "scoring_run": False,
        "sensor_arrays_loaded": False,
        "target_outcomes_opened": False,
    }
    for row in artifact["rows"]:
        provider = set(row["provider_camera_ids"])
        scoring = set(row["scoring_camera_ids"])
        assert provider.isdisjoint(scoring)
        assert len(provider) == row["provider_camera_count"]
        assert len(scoring) == row["scoring_camera_count"]
        assert len(provider | scoring) == row["camera_count"]
        assert "object_id" not in row


def test_partition_is_deterministic_and_session_domain_separated() -> None:
    first = _MODULE.build_partition_artifact(
        protocol_path=_PROTOCOL,
        exact_plan_path=_EXACT_PLAN,
        implementation_revision="b" * 40,
    )
    second = _MODULE.build_partition_artifact(
        protocol_path=_PROTOCOL,
        exact_plan_path=_EXACT_PLAN,
        implementation_revision="b" * 40,
    )

    assert first == second
    first_row = first["rows"][0]
    changed_episode = _MODULE._object_session_hash(
        object_hash=first_row["object_hash"],
        episode_id=first_row["episode_id"] + 1,
    )
    assert changed_episode != first_row["object_session_hash"]


def test_camera_names_must_be_unique_and_multiview() -> None:
    with pytest.raises(ValueError, match="at least four"):
        _MODULE._camera_ids(["a", "b", "c"])
    with pytest.raises(ValueError, match="repeat"):
        _MODULE._camera_ids(["a", "b", "c", "a"])


def test_cli_writes_once_and_refuses_changed_output(tmp_path: Path) -> None:
    output = tmp_path / "partitions.json"
    arguments = [
        "--protocol",
        str(_PROTOCOL),
        "--exact-plan",
        str(_EXACT_PLAN),
        "--output",
        str(output),
        "--implementation-revision",
        "c" * 40,
    ]

    assert _MODULE.main(arguments) == 0
    first = output.read_bytes()
    assert _MODULE.main(arguments) == 0
    assert output.read_bytes() == first
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output differs"):
        _MODULE.main(arguments)
