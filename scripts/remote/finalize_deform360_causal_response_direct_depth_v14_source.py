#!/usr/bin/env python3
"""Freeze the first twelve admitted V14 sources without outcome access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bayesian_phystwin.deform360_causal_response_direct_depth_admission_v14 import (
    ADMISSION_REPORT_FILENAME,
    PREFLIGHT_FILENAME,
    load_v14_admission_prelock_protocol,
    validate_v14_admission_report,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_cohort import (
    validate_v14_staging_queue,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_preflight import (
    deform360_v14_case_hash,
    load_adaptive_direct_depth_source_preflight_v14,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_selection_v14 import (
    SELECTION_FILENAME,
    V14SelectionDisposition,
    build_v14_selection_ledger,
    load_v14_source_finalizer_protocol,
    write_v14_selection_ledger,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_source_lock import (
    AdaptiveDirectDepthSourceCaseV14,
    build_adaptive_direct_depth_source_lock_v14,
    write_adaptive_direct_depth_source_lock_v14,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_synthetic import (
    validate_adaptive_direct_depth_synthetic_v14,
)
from bayesian_phystwin.deform360_causal_response_preflight import (
    deform360_object_hash,
)
from bayesian_phystwin.deform360_object_exclusion import (
    file_sha256,
    load_object_exclusion_manifest,
)

SOURCE_LOCK_FILENAME = "deform360_causal_response_direct_depth_source_lock_v14.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {source}")
    return payload


def _canonical_config_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_repository(repository: Path) -> str:
    revision = _git_output(repository, "rev-parse", "HEAD")
    _require(
        not _git_output(repository, "status", "--porcelain", "--untracked-files=normal"),
        "V14 source finalizer repository is dirty",
    )
    return revision


def _validate_window_failure(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    digest = hashlib.sha256(
        b"deform360-causal-response-direct-depth-window-v14\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    _require(
        payload.get("artifact_kind") == "Deform360CausalDirectDepthWindowStageV14"
        and payload.get("contract")
        == "deform360-causal-response-direct-depth-window-v14"
        and payload.get("status") == "technical_preflight_failure"
        and payload.get("artifact_sha256") == digest,
        "V14 technical pre-lock disposition is invalid",
    )
    return payload


def _verify_parent(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    role: str,
    semantic_sha256: str,
) -> None:
    expected = protocol["parent_artifacts"][role]
    _require(
        expected["semantic_sha256"] == semantic_sha256
        and expected["file_sha256"] == file_sha256(path),
        f"V14 source finalizer parent changed: {role}",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--finalizer-protocol", type=Path, required=True)
    parser.add_argument("--method-protocol", type=Path, required=True)
    parser.add_argument("--admission-prelock", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--exclusion-manifest", type=Path, required=True)
    parser.add_argument("--synthetic-control", type=Path, required=True)
    parser.add_argument(
        "--technical-disposition",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--admission-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repository = args.repo.resolve()
    revision = _require_clean_repository(repository)
    finalizer_path = args.finalizer_protocol.resolve()
    finalizer = load_v14_source_finalizer_protocol(finalizer_path)
    implementation_paths = {
        "admission_module": (
            repository
            / "src/bayesian_phystwin/"
            "deform360_causal_response_direct_depth_admission_v14.py"
        ),
        "finalizer_runner": Path(__file__).resolve(),
        "selection_module": (
            repository
            / "src/bayesian_phystwin/"
            "deform360_causal_response_direct_depth_selection_v14.py"
        ),
        "source_lock_module": (
            repository
            / "src/bayesian_phystwin/"
            "deform360_causal_response_direct_depth_source_lock.py"
        ),
    }
    _require(
        all(
            file_sha256(path)
            == finalizer["implementation"]["file_sha256"][name]
            for name, path in implementation_paths.items()
        )
        and _git_output(
            repository,
            "merge-base",
            "--is-ancestor",
            finalizer["implementation"]["parent_commit"],
            revision,
        )
        == "",
        "V14 source finalizer implementation changed",
    )

    method_path = args.method_protocol.resolve()
    method = _read_json(method_path)
    _require(
        method.get("config_sha256") == _canonical_config_sha256(method),
        "V14 method protocol checksum changed",
    )
    admission_prelock_path = args.admission_prelock.resolve()
    admission_prelock = load_v14_admission_prelock_protocol(
        admission_prelock_path
    )
    queue_path = args.queue.resolve()
    queue = validate_v14_staging_queue(queue_path)
    exclusion_path = args.exclusion_manifest.resolve()
    exclusion = load_object_exclusion_manifest(exclusion_path)
    synthetic_path = args.synthetic_control.resolve()
    synthetic = validate_adaptive_direct_depth_synthetic_v14(synthetic_path)
    for path, role, semantic in (
        (method_path, "method_protocol", method["config_sha256"]),
        (
            admission_prelock_path,
            "admission_prelock",
            admission_prelock["config_sha256"],
        ),
        (queue_path, "staging_queue", queue["queue_sha256"]),
        (exclusion_path, "exclusion_manifest", exclusion["exclusion_sha256"]),
        (synthetic_path, "synthetic_control", synthetic.artifact_sha256),
    ):
        _verify_parent(
            path,
            protocol=finalizer,
            role=role,
            semantic_sha256=semantic,
        )

    by_rank: dict[int, tuple[dict[str, Any], Path]] = {}
    for path in args.technical_disposition:
        resolved = path.resolve()
        payload = _validate_window_failure(resolved)
        rank = int(payload["queue_rank"])
        _require(rank not in by_rank, "V14 source rank has duplicate dispositions")
        by_rank[rank] = (payload, resolved)
    admission_root = args.admission_root.resolve()
    for directory in sorted(admission_root.glob("rank-*")):
        if not directory.is_dir():
            continue
        payload = validate_v14_admission_report(directory)
        rank = int(payload["queue_rank"])
        _require(rank not in by_rank, "V14 source rank has duplicate dispositions")
        by_rank[rank] = (payload, directory / ADMISSION_REPORT_FILENAME)

    admitted_ranks = sorted(
        rank for rank, (payload, _) in by_rank.items() if payload["status"] == "admitted"
    )
    _require(
        len(admitted_ranks) >= 12,
        "V14 source panel has fewer than twelve admitted preflights",
    )
    final_rank = admitted_ranks[11]
    _require(
        set(by_rank) == set(range(1, final_rank + 1))
        and not any(rank > final_rank for rank in by_rank),
        "V14 source dispositions do not stop at a contiguous twelfth admission",
    )

    dispositions: list[V14SelectionDisposition] = []
    for rank in range(1, final_rank + 1):
        payload, path = by_rank[rank]
        candidate = queue["candidates"][rank - 1]
        _require(
            payload["queue_rank"] == rank
            and payload["object_hash"] == deform360_object_hash(
                str(candidate["object_id"])
            )
            and payload["case_hash"] == deform360_v14_case_hash(
                str(candidate["object_id"]),
                int(candidate["episode_id"]),
            ),
            "V14 disposition differs from the immutable queue",
        )
        dispositions.append(
            V14SelectionDisposition(
                queue_rank=rank,
                object_hash=payload["object_hash"],
                case_hash=payload["case_hash"],
                status=payload["status"],
                disposition_artifact_sha256=payload["artifact_sha256"],
                disposition_file_sha256=file_sha256(path),
                selected=payload["status"] == "admitted",
            )
        )
    ledger = build_v14_selection_ledger(
        dispositions,
        repository_revision=revision,
        queue_sha256=queue["queue_sha256"],
        queue_path=queue_path,
        admission_prelock_config_sha256=admission_prelock["config_sha256"],
        admission_prelock_path=admission_prelock_path,
    )

    selected_dispositions = [item for item in dispositions if item.selected]
    source_cases: list[AdaptiveDirectDepthSourceCaseV14] = []
    preflights = []
    for index, disposition in enumerate(selected_dispositions):
        payload, _ = by_rank[disposition.queue_rank]
        admission_dir = admission_root / f"rank-{disposition.queue_rank:03d}"
        preflight = load_adaptive_direct_depth_source_preflight_v14(
            admission_dir / PREFLIGHT_FILENAME
        )
        candidate = queue["candidates"][disposition.queue_rank - 1]
        _require(
            preflight.admitted
            and preflight.object_hash == disposition.object_hash
            and preflight.case_hash == disposition.case_hash,
            "V14 selected disposition lacks its accepted preflight",
        )
        preflights.append(preflight)
        source_cases.append(
            AdaptiveDirectDepthSourceCaseV14(
                case_id=preflight.case_hash,
                case_hash=preflight.case_hash,
                object_hash=preflight.object_hash,
                metadata_sha256=str(candidate["metadata_sha256"]),
                source_preflight_sha256=preflight.artifact_sha256,
                carrier_artifact_sha256=preflight.carrier_artifact_sha256,
                carrier_arm=preflight.carrier_arm,
                fold=index % 3,
            )
        )
    source_lock = build_adaptive_direct_depth_source_lock_v14(
        source_cases,
        repository_revision=revision,
        method_config_sha256=method["config_sha256"],
        exclusion_manifest_path=exclusion_path,
        synthetic_control_result_path=synthetic_path,
        selection_metadata_sha256=ledger.artifact_sha256,
        source_preflights=preflights,
    )

    output = args.output_dir.resolve()
    _require(not output.exists(), "V14 source finalizer output already exists")
    scratch = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    _require(not scratch.exists(), "V14 source finalizer scratch already exists")
    scratch.mkdir(parents=True)
    write_v14_selection_ledger(scratch / SELECTION_FILENAME, ledger)
    write_adaptive_direct_depth_source_lock_v14(
        scratch / SOURCE_LOCK_FILENAME,
        source_lock,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch.rename(output)
    print(
        json.dumps(
            {
                "disposition_count": len(dispositions),
                "final_queue_rank": final_rank,
                "selected_source_count": len(source_cases),
                "selection_artifact_sha256": ledger.artifact_sha256,
                "source_lock_artifact_sha256": source_lock.artifact_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
