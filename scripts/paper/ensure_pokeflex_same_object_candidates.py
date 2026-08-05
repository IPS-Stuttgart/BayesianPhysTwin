#!/usr/bin/env python3
"""Stage exact frozen PokeFlex candidates and attest the video take."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import prepare_pokeflex_same_object_assets as asset_resolver


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def _candidate_take_id(record: Mapping[str, Any]) -> str:
    frozen = Path(str(record["path"]))
    suffix = "_candidates.json"
    asset_resolver._require(
        frozen.name.endswith(suffix),
        f"candidate artifact name is not canonical: {frozen.name}",
    )
    return str(record.get("take_id", frozen.name.removesuffix(suffix)))


def _candidate_paths(
    record: Mapping[str, Any],
    *,
    candidate_root: Path | None,
    search_roots: Iterable[Path],
) -> Iterable[tuple[Path, str]]:
    frozen = Path(str(record["path"]))
    if candidate_root is not None:
        yield candidate_root / frozen.name, "configured"
    yield frozen, "frozen"
    yield from (
        (path, "discovered")
        for path in asset_resolver._walk_files(search_roots, filename=frozen.name)
    )


def _find_exact_candidate(
    record: Mapping[str, Any],
    *,
    candidate_root: Path | None,
    search_roots: Iterable[Path],
) -> tuple[Path | None, str | None, list[str]]:
    expected_sha = str(record["sha256"])
    attempts: list[str] = []
    seen: set[str] = set()
    for path, source_kind in _candidate_paths(
        record,
        candidate_root=candidate_root,
        search_roots=search_roots,
    ):
        key = str(path.absolute())
        if key in seen:
            continue
        seen.add(key)
        try:
            if not path.is_file():
                attempts.append(f"{path}: missing or unreadable")
                continue
            observed = asset_resolver._sha256(path)
        except OSError as error:
            attempts.append(f"{path}: {error}")
            continue
        if observed != expected_sha:
            attempts.append(f"{path}: sha256={observed}")
            continue
        return path, source_kind, attempts
    return None, None, attempts


def _archive_verification(record: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(record["path"]))
    expected_size = int(record["size_bytes"])
    expected_sha = str(record["sha256"])
    try:
        if not path.is_file():
            return {
                "status": "unavailable",
                "path": str(path.absolute()),
                "expected_size_bytes": expected_size,
                "expected_sha256": expected_sha,
                "error": "archive is missing or unreadable",
            }
        observed_size = path.stat().st_size
    except OSError as error:
        return {
            "status": "unavailable",
            "path": str(path.absolute()),
            "expected_size_bytes": expected_size,
            "expected_sha256": expected_sha,
            "error": str(error),
        }
    asset_resolver._require(
        observed_size == expected_size,
        f"PokeFlex archive size changed: {path}",
    )
    try:
        observed_sha = asset_resolver._sha256(path)
    except OSError as error:
        return {
            "status": "unavailable",
            "path": str(path.absolute()),
            "expected_size_bytes": expected_size,
            "expected_sha256": expected_sha,
            "observed_size_bytes": observed_size,
            "error": str(error),
        }
    asset_resolver._require(
        observed_sha == expected_sha,
        f"PokeFlex archive checksum changed: {path}",
    )
    return {
        "status": "verified",
        "path": str(path.resolve()),
        "expected_size_bytes": expected_size,
        "expected_sha256": expected_sha,
        "observed_size_bytes": observed_size,
        "observed_sha256": observed_sha,
    }


def _replace_staged_take(staged_take: Path, take_root: Path) -> None:
    if staged_take.is_symlink() or staged_take.is_file():
        staged_take.unlink()
    elif staged_take.exists():
        shutil.rmtree(staged_take)
    staged_take.symlink_to(take_root, target_is_directory=True)
    asset_resolver._require(
        (staged_take / "robot_data.json").is_file(),
        "staged take is invalid",
    )


def _stage_take_for_regeneration(
    *,
    take_id: str,
    archive: Mapping[str, Any],
    archive_verification: Mapping[str, Any],
    configured_dataset_root: Path | None,
    search_roots: tuple[Path, ...],
    stage_root: Path,
) -> tuple[Path, dict[str, Any]]:
    if archive_verification["status"] == "verified":
        staged_take, evidence = asset_resolver._stage_take(
            take_id=take_id,
            archive=archive,
            configured_dataset_root=configured_dataset_root,
            search_roots=search_roots,
            stage_root=stage_root,
        )
        return staged_take, {
            **evidence,
            "archive_validation": "verified",
        }

    take_root = asset_resolver._find_take_root(
        take_id,
        configured_dataset_root=configured_dataset_root,
        search_roots=search_roots,
    )
    asset_resolver._require(
        take_root is not None,
        f"no readable extracted take is available for {take_id}",
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    staged_take = stage_root / take_id
    _replace_staged_take(staged_take, take_root)
    return staged_take, {
        "take_id": take_id,
        "archive": str(archive_verification["path"]),
        "archive_validation": "unavailable-exact-regeneration-required",
        "archive_verification": dict(archive_verification),
        "take_root": str(take_root),
        "staged_take": str(staged_take),
        "extraction_performed": False,
    }


def _validated_inputs(
    *,
    prospective_result: Path,
    execution_manifest: Path,
    prospective_protocol: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = asset_resolver._load_json(prospective_result)
    manifest = asset_resolver._load_json(execution_manifest)
    protocol = asset_resolver._load_json(prospective_protocol)
    asset_resolver._require(
        result.get("gate_passed") is True, "prospective result failed"
    )
    asset_resolver._require(
        manifest.get("replacement_performed") is False,
        "prospective candidate replacement was recorded",
    )
    asset_resolver._require(
        manifest["prospective_evaluation"]["sha256"]
        == asset_resolver._sha256(prospective_result),
        "prospective result checksum changed",
    )
    asset_resolver._require(
        manifest["protocol_sha256"] == protocol["protocol_sha256"],
        "prospective protocol checksum changed",
    )
    result_records = result.get("candidate_artifacts")
    manifest_records = manifest.get("candidate_artifacts")
    asset_resolver._require(
        isinstance(result_records, list) and result_records,
        "prospective candidate records are missing",
    )
    asset_resolver._require(
        isinstance(manifest_records, list) and manifest_records,
        "execution-manifest candidate records are missing",
    )
    manifest_by_name = {
        Path(str(record["path"])).name: record for record in manifest_records
    }
    for record in result_records:
        name = Path(str(record["path"])).name
        asset_resolver._require(
            name in manifest_by_name, f"candidate is unregistered: {name}"
        )
        asset_resolver._require(
            str(record["sha256"]) == str(manifest_by_name[name]["sha256"]),
            f"candidate checksum differs between frozen records: {name}",
        )
    return result, manifest


def _resolve_runtime(
    *,
    registration_protocol: Path,
    configured_upstream: Path | None,
    configured_checkpoints: Path | None,
    search_roots: tuple[Path, ...],
    software_root: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    protocol = asset_resolver._load_json(registration_protocol)
    payload = protocol.get("payload", protocol)
    upstream, upstream_evidence = asset_resolver._resolve_upstream(
        payload["upstream"],
        configured_checkout=configured_upstream,
        search_roots=search_roots,
        software_root=software_root,
    )
    checkpoint_records = payload["upstream"]["released_kinect_checkpoint"]
    checkpoints, checkpoint_evidence = asset_resolver._resolve_checkpoints(
        checkpoint_records,
        configured_root=configured_checkpoints,
        search_roots=search_roots,
        software_root=software_root,
    )
    return (
        upstream,
        checkpoints,
        {
            "upstream": {**upstream_evidence, "checkout": str(upstream)},
            "checkpoints": {**checkpoint_evidence, "root": str(checkpoints)},
        },
    )


def ensure_candidates(args: argparse.Namespace) -> dict[str, Any]:
    prospective_result = args.prospective_result.resolve()
    execution_manifest = args.execution_manifest.resolve()
    prospective_protocol = args.prospective_protocol.resolve()
    independent_depth_protocol = args.independent_depth_protocol.resolve()
    registration_protocol = args.registration_protocol.resolve()
    result, manifest = _validated_inputs(
        prospective_result=prospective_result,
        execution_manifest=execution_manifest,
        prospective_protocol=prospective_protocol,
    )
    requested_attestation = (
        args.attest_take_id.strip() if args.attest_take_id else None
    )
    attested_take_id = asset_resolver._select_take(result, requested_attestation)

    output_root = args.output_root.resolve()
    workspace_root = args.workspace_root.resolve()
    dataset_stage = workspace_root / "dataset"
    software_root = workspace_root / "software"
    for path in (output_root, dataset_stage, software_root):
        path.mkdir(parents=True, exist_ok=True)

    candidate_root = args.candidate_root.resolve() if args.candidate_root else None
    configured_dataset = args.dataset_root.resolve() if args.dataset_root else None
    configured_upstream = (
        args.upstream_checkout.resolve() if args.upstream_checkout else None
    )
    configured_checkpoints = (
        args.checkpoint_root.resolve() if args.checkpoint_root else None
    )
    search_roots = tuple(
        path.resolve()
        for path in (args.search_root or asset_resolver.DEFAULT_SEARCH_ROOTS)
    )

    candidate_records = result["candidate_artifacts"]
    archive_by_take = {
        str(record["take_id"]): record for record in manifest["archives"]
    }
    frozen_take_ids = {str(value) for value in result["take_ids"]}
    runtime: tuple[Path, Path, dict[str, Any]] | None = None
    prepared: list[dict[str, Any]] = []
    regeneration_performed = False

    for record in candidate_records:
        take_id = _candidate_take_id(record)
        asset_resolver._require(
            take_id in frozen_take_ids,
            f"candidate take is outside the prospective panel: {take_id}",
        )
        asset_resolver._require(
            take_id in archive_by_take,
            f"candidate take has no frozen archive: {take_id}",
        )
        archive = archive_by_take[take_id]
        archive_verification = _archive_verification(archive)
        raw_take_attestation_required = (
            take_id == attested_take_id
            and archive_verification["status"] != "verified"
        )
        frozen = Path(str(record["path"]))
        target = output_root / frozen.name
        expected_sha = str(record["sha256"])

        if (
            not raw_take_attestation_required
            and target.is_file()
            and asset_resolver._sha256(target) == expected_sha
        ):
            prepared.append(
                {
                    "take_id": take_id,
                    "source": str(target),
                    "source_resolution": "existing-output",
                    "staged": str(target),
                    "sha256": expected_sha,
                    "regenerated": False,
                    "raw_take_attestation_required": False,
                    "archive_verification": archive_verification,
                }
            )
            continue
        if target.exists() or target.is_symlink():
            target.unlink()

        source, source_kind, attempts = _find_exact_candidate(
            record,
            candidate_root=candidate_root,
            search_roots=search_roots,
        )
        if (
            not raw_take_attestation_required
            and source is not None
            and source_kind is not None
        ):
            shutil.copy2(source, target)
            observed = asset_resolver._sha256(target)
            asset_resolver._require(
                observed == expected_sha,
                f"candidate changed while copying: {frozen.name}",
            )
            prepared.append(
                {
                    "take_id": take_id,
                    "source": str(source.resolve()),
                    "source_resolution": source_kind,
                    "staged": str(target),
                    "sha256": observed,
                    "regenerated": False,
                    "raw_take_attestation_required": False,
                    "failed_attempts": attempts,
                    "archive_verification": archive_verification,
                }
            )
            continue
        if source is not None and raw_take_attestation_required:
            attempts.append(
                "exact candidate copy found, but the selected video take requires "
                "raw-take attestation because its frozen archive is unavailable"
            )

        if runtime is None:
            runtime = _resolve_runtime(
                registration_protocol=registration_protocol,
                configured_upstream=configured_upstream,
                configured_checkpoints=configured_checkpoints,
                search_roots=search_roots,
                software_root=software_root,
            )
        upstream, checkpoints, _ = runtime
        _, take_evidence = _stage_take_for_regeneration(
            take_id=take_id,
            archive=archive,
            archive_verification=archive_verification,
            configured_dataset_root=configured_dataset,
            search_roots=search_roots,
            stage_root=dataset_stage,
        )
        take_root = Path(str(take_evidence["take_root"]))
        runner = (
            _repository_root()
            / "scripts/remote/run_pokeflex_independent_depth_regret_guard_prospective.py"
        )
        command = [
            sys.executable,
            str(runner),
            str(take_root),
            str(target),
            "--upstream-checkout",
            str(upstream),
            "--checkpoint-root",
            str(checkpoints),
            "--prospective-protocol",
            str(prospective_protocol),
            "--independent-depth-protocol",
            str(independent_depth_protocol),
            "--registration-protocol",
            str(registration_protocol),
        ]
        asset_resolver._run(command)
        observed = asset_resolver._sha256(target)
        asset_resolver._require(
            observed == expected_sha,
            f"regenerated candidate checksum differs for {take_id}: {observed}",
        )
        regeneration_performed = True
        prepared.append(
            {
                "take_id": take_id,
                "source": str(take_root),
                "source_resolution": "regenerated-from-frozen-inputs",
                "staged": str(target),
                "sha256": observed,
                "regenerated": True,
                "raw_take_attestation_required": raw_take_attestation_required,
                "failed_attempts": attempts,
                "archive_verification": archive_verification,
                "take": take_evidence,
                "command": command,
            }
        )

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexSameObjectCandidatePreparationV1",
        "attested_take_id": attested_take_id,
        "regeneration_performed": regeneration_performed,
        "candidate_artifacts": prepared,
        "inputs": {
            "prospective_result": str(prospective_result),
            "prospective_result_sha256": asset_resolver._sha256(prospective_result),
            "execution_manifest": str(execution_manifest),
            "execution_manifest_sha256": asset_resolver._sha256(execution_manifest),
            "prospective_protocol": str(prospective_protocol),
            "prospective_protocol_sha256": asset_resolver._sha256(prospective_protocol),
            "independent_depth_protocol": str(independent_depth_protocol),
            "independent_depth_protocol_sha256": asset_resolver._sha256(
                independent_depth_protocol
            ),
            "registration_protocol": str(registration_protocol),
            "registration_protocol_sha256": asset_resolver._sha256(
                registration_protocol
            ),
        },
    }
    if runtime is not None:
        upstream, checkpoints, runtime_evidence = runtime
        evidence["runtime"] = runtime_evidence
        evidence["dataset_root"] = str(dataset_stage)
        if args.github_env is not None:
            asset_resolver._append_environment(
                args.github_env.resolve(),
                {
                    "POKEFLEX_REGENERATED_DATA_ROOT": dataset_stage,
                    "POKEFLEX_REGENERATED_UPSTREAM_CHECKOUT": upstream,
                    "POKEFLEX_REGENERATED_CHECKPOINT_ROOT": checkpoints,
                },
            )
    if args.github_env is not None:
        asset_resolver._append_environment(
            args.github_env.resolve(),
            {"POKEFLEX_VERIFIED_CANDIDATE_ROOT": output_root},
        )
    if args.evidence_output is not None:
        asset_resolver._write_json(args.evidence_output.resolve(), evidence)
    return evidence


def main() -> None:
    repository_root = _repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prospective-result",
        type=Path,
        default=(
            repository_root
            / "results/sota/pokeflex_independent_depth_regret_guard_prospective_v1"
            / "prospective_evaluation.json"
        ),
    )
    parser.add_argument(
        "--execution-manifest",
        type=Path,
        default=(
            repository_root
            / "results/sota/pokeflex_independent_depth_regret_guard_prospective_v1"
            / "execution_manifest.json"
        ),
    )
    parser.add_argument(
        "--prospective-protocol",
        type=Path,
        default=(
            repository_root
            / "configs/sota"
            / "pokeflex_independent_depth_regret_guard_prospective_v1.json"
        ),
    )
    parser.add_argument(
        "--independent-depth-protocol",
        type=Path,
        default=(
            repository_root
            / "configs/sota/pokeflex_independent_depth_source_validation_v2.json"
        ),
    )
    parser.add_argument(
        "--registration-protocol",
        type=Path,
        default=(
            repository_root / "configs/sota/pokeflex_bayesian_registration_v1.json"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path)
    parser.add_argument("--candidate-root", type=_optional_path)
    parser.add_argument("--dataset-root", type=_optional_path)
    parser.add_argument("--upstream-checkout", type=_optional_path)
    parser.add_argument("--checkpoint-root", type=_optional_path)
    parser.add_argument("--search-root", type=Path, action="append")
    parser.add_argument(
        "--attest-take-id",
        default=os.environ.get("INPUT_TAKE_ID", ""),
        help=(
            "Take requiring raw-data attestation; blank selects the strongest "
            "prospective take"
        ),
    )
    parser.add_argument("--github-env", type=Path)
    args = parser.parse_args()
    try:
        evidence = ensure_candidates(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
