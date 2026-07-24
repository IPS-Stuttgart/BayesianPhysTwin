"""Immutable source inputs for the equivariant-force official-Warp gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .phystwin_equivariant_force_data import load_equivariant_force_episode
from .phystwin_equivariant_force_source import (
    load_equivariant_force_source_protocol,
)
from .phystwin_residual_dynamics import _sha256


EQUIVARIANT_FORCE_STAGE2_SOURCE_CONTRACT = (
    "phystwin-equivariant-force-stage2-source-manifest-v1"
)
OFFICIAL_SIMULATOR_RELATIVE_PATH = (
    "qqtt/model/diff_simulator/spring_mass_warp.py"
)
STAGE2_SOURCE_FILES = {
    "baseline_trajectory": "inference.pkl",
    "final_data": "final_data.pkl",
    "optimal_params": "optimal_params.pkl",
    "manual_tracks": "gt_track_3d.pkl",
    "checkpoint": "checkpoint.pth",
    "split": "split.json",
}


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    return payload


def _case_source_record(
    case_id: str,
    data_root: Path,
    episode_root: Path,
) -> dict[str, Any]:
    episode = load_equivariant_force_episode(
        episode_root / case_id / "force_episode"
    )
    case_root = data_root / case_id
    files = {
        role: {
            "relative_path": filename,
            "sha256": _sha256(case_root / filename),
        }
        for role, filename in STAGE2_SOURCE_FILES.items()
    }
    expected_episode_sources = {
        role: files[role]["sha256"]
        for role in ("baseline_trajectory", "final_data", "optimal_params")
    }
    if episode.source_checksums != expected_episode_sources:
        raise ValueError(f"{case_id}: episode and source files disagree")
    split = _load_json_object(case_root / STAGE2_SOURCE_FILES["split"])
    if split.get("train") != [0, episode.validation_end_frame]:
        raise ValueError(f"{case_id}: released split and episode disagree")
    return {
        "case_id": case_id,
        "episode_artifact_id": episode.artifact_id,
        "fit_end_frame": episode.fit_end_frame,
        "validation_end_frame": episode.validation_end_frame,
        "files": files,
    }


def build_equivariant_force_stage2_source_manifest(
    official_repo: str | Path,
    data_root: str | Path,
    episode_root: str | Path,
    source_protocol_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Hash every registered Stage-2 source without resolving a target path."""

    protocol = load_equivariant_force_source_protocol(source_protocol_path)
    official = Path(official_repo).resolve()
    data = Path(data_root).resolve()
    episodes = Path(episode_root).resolve()
    simulator_source = official / OFFICIAL_SIMULATOR_RELATIVE_PATH
    records = [
        _case_source_record(str(case), data, episodes)
        for case in protocol.payload["source_cases"]
    ]
    payload = {
        "schema_version": 1,
        "contract": EQUIVARIANT_FORCE_STAGE2_SOURCE_CONTRACT,
        "source_protocol_sha256": _sha256(source_protocol_path),
        "official_simulator": {
            "relative_path": OFFICIAL_SIMULATOR_RELATIVE_PATH,
            "sha256": _sha256(simulator_source),
        },
        "source_cases": records,
        "target_artifacts_opened": False,
        "claim_boundary": (
            "Source-input identity only. This artifact contains no Stage-1 or "
            "Stage-2 outcome and authorizes no target access."
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return {
        **payload,
        "manifest_path": str(output.resolve()),
        "manifest_sha256": _sha256(output),
    }


def load_equivariant_force_stage2_source_manifest(
    path: str | Path,
    *,
    source_protocol_path: str | Path,
) -> dict[str, Any]:
    """Load a source manifest and verify its closed information boundary."""

    payload = _load_json_object(path)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported Stage-2 source-manifest schema")
    if payload.get("contract") != EQUIVARIANT_FORCE_STAGE2_SOURCE_CONTRACT:
        raise ValueError("unsupported Stage-2 source-manifest contract")
    if payload.get("target_artifacts_opened") is not False:
        raise ValueError("Stage-2 source manifest crossed the target boundary")
    if payload.get("source_protocol_sha256") != _sha256(source_protocol_path):
        raise ValueError("Stage-2 source manifest used another source protocol")
    protocol = load_equivariant_force_source_protocol(source_protocol_path)
    raw_cases = payload.get("source_cases")
    if not isinstance(raw_cases, list):
        raise ValueError("Stage-2 source manifest omits source cases")
    cases = [str(record.get("case_id", "")) for record in raw_cases]
    if cases != list(protocol.payload["source_cases"]):
        raise ValueError("Stage-2 source case order changed")
    official = payload.get("official_simulator")
    if (
        not isinstance(official, Mapping)
        or official.get("relative_path") != OFFICIAL_SIMULATOR_RELATIVE_PATH
        or not _valid_sha256(official.get("sha256"))
    ):
        raise ValueError("Stage-2 official simulator identity is invalid")
    return payload


def validate_equivariant_force_stage2_source_case(
    manifest_path: str | Path,
    official_repo: str | Path,
    data_root: str | Path,
    episode_root: str | Path,
    source_protocol_path: str | Path,
    case_id: str,
) -> dict[str, Any]:
    """Validate one registered case against the immutable source manifest."""

    manifest = load_equivariant_force_stage2_source_manifest(
        manifest_path,
        source_protocol_path=source_protocol_path,
    )
    protocol = load_equivariant_force_source_protocol(source_protocol_path)
    if case_id not in protocol.payload["source_cases"]:
        raise ValueError("Stage-2 source validation accepts source cases only")
    official_source = (
        Path(official_repo).resolve() / OFFICIAL_SIMULATOR_RELATIVE_PATH
    )
    expected_official = manifest["official_simulator"]["sha256"]
    if _sha256(official_source) != expected_official:
        raise ValueError("official PhysTwin simulator source changed")
    matches = [
        record
        for record in manifest["source_cases"]
        if record.get("case_id") == case_id
    ]
    if len(matches) != 1:
        raise ValueError(f"{case_id}: source manifest has no unique record")
    expected = matches[0]
    actual = _case_source_record(
        case_id,
        Path(data_root).resolve(),
        Path(episode_root).resolve(),
    )
    if actual != expected:
        raise ValueError(f"{case_id}: Stage-2 source inputs changed")
    return {
        "case_id": case_id,
        "source_manifest_sha256": _sha256(manifest_path),
        "official_simulator_sha256": expected_official,
        "episode_artifact_id": actual["episode_artifact_id"],
        "files": actual["files"],
        "target_artifacts_opened": False,
    }


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
