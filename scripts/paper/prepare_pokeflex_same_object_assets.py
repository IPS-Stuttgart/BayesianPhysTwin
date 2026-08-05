#!/usr/bin/env python3
"""Resolve immutable PokeFlex assets for the paper-artifact workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_SEARCH_ROOTS = (
    Path("/mnt/corsair"),
    Path("/mnt/lexar4tb"),
    Path("/home/github-runner"),
    Path("/opt"),
)
MAXIMUM_SEARCH_DEPTH = 10


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _append_environment(path: Path, values: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            rendered = str(value)
            if "\n" in rendered or "\r" in rendered:
                raise ValueError(f"environment value contains a newline: {key}")
            handle.write(f"{key}={rendered}\n")


def _relative_depth(path: Path, root: Path) -> int:
    try:
        return len(path.relative_to(root).parts)
    except ValueError:
        return MAXIMUM_SEARCH_DEPTH + 1


def _walk_files(
    roots: Iterable[Path],
    *,
    filename: str,
    maximum_depth: int = MAXIMUM_SEARCH_DEPTH,
) -> Iterable[Path]:
    """Yield matching files without descending indefinitely through large mounts."""

    ignored_names = {
        ".cache",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        "__pycache__",
        "node_modules",
    }
    for root in roots:
        if not root.is_dir():
            continue
        for current, directories, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            depth = _relative_depth(current_path, root)
            directories[:] = [
                name
                for name in directories
                if name not in ignored_names and depth < maximum_depth
            ]
            if filename in files:
                yield current_path / filename


def _run(command: Sequence[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _git_head(checkout: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _select_take(result: Mapping[str, Any], requested: str | None) -> str:
    takes = result.get("takes")
    _require(isinstance(takes, list) and takes, "prospective takes are missing")
    available = {str(value["take_id"]): value for value in takes}
    if requested:
        _require(
            requested in available, f"take is outside the frozen panel: {requested}"
        )
        return requested
    return str(
        max(available.values(), key=lambda value: value["relative_improvement"])[
            "take_id"
        ]
    )


def _stage_candidate_artifacts(
    result: Mapping[str, Any],
    *,
    candidate_root: Path | None,
    search_roots: Iterable[Path],
    output_root: Path,
) -> list[dict[str, Any]]:
    records = result.get("candidate_artifacts")
    _require(isinstance(records, list) and records, "candidate artifacts are missing")
    take_ids = result.get("take_ids")
    _require(isinstance(take_ids, list) and take_ids, "prospective take IDs are missing")
    frozen_take_ids = {str(value) for value in take_ids}
    output_root.mkdir(parents=True, exist_ok=True)
    staged: list[dict[str, Any]] = []
    for record in records:
        frozen = Path(str(record["path"]))
        expected_sha = str(record["sha256"])
        suffix = "_candidates.json"
        _require(
            frozen.name.endswith(suffix),
            f"candidate artifact name is not canonical: {frozen.name}",
        )
        take_id = str(record.get("take_id", frozen.name.removesuffix(suffix)))
        _require(take_id in frozen_take_ids, f"candidate take is outside panel: {take_id}")

        candidates: list[tuple[Path, str]] = []
        if candidate_root is not None:
            candidates.append((candidate_root / frozen.name, "configured"))
        candidates.append((frozen, "frozen"))
        candidates.extend(
            (path, "discovered")
            for path in _walk_files(search_roots, filename=frozen.name)
        )

        source: Path | None = None
        digest: str | None = None
        resolution: str | None = None
        attempts: list[str] = []
        seen: set[str] = set()
        for path, source_kind in candidates:
            key = str(path.absolute())
            if key in seen:
                continue
            seen.add(key)
            try:
                if not path.is_file():
                    attempts.append(f"{path}: missing")
                    continue
                observed = _sha256(path)
            except OSError as error:
                attempts.append(f"{path}: {error}")
                continue
            if observed != expected_sha:
                attempts.append(f"{path}: sha256={observed}")
                continue
            source = path
            digest = observed
            resolution = source_kind
            break
        if source is None or digest is None or resolution is None:
            detail = "; ".join(attempts) if attempts else "no candidate paths found"
            raise FileNotFoundError(
                f"no readable exact-SHA candidate artifact for {take_id}: {detail}"
            )

        target = output_root / frozen.name
        shutil.copy2(source, target)
        staged.append(
            {
                "take_id": take_id,
                "source": str(source.resolve()),
                "source_resolution": resolution,
                "staged": str(target.resolve()),
                "sha256": digest,
                "byte_identical_to_frozen": True,
            }
        )
    return staged


def _find_take_root(
    take_id: str,
    *,
    configured_dataset_root: Path | None,
    search_roots: Iterable[Path],
) -> Path | None:
    if configured_dataset_root is not None:
        candidate = configured_dataset_root / take_id
        if (candidate / "robot_data.json").is_file():
            return candidate.resolve()
    suffix = Path(take_id) / "robot_data.json"
    for path in _walk_files(search_roots, filename="robot_data.json"):
        if path.parts[-2:] == suffix.parts:
            return path.parent.resolve()
    return None


def _locate_extracted_take(extract_root: Path, take_id: str) -> Path:
    direct = extract_root / take_id
    if (direct / "robot_data.json").is_file():
        return direct.resolve()
    matches = [
        path.parent
        for path in extract_root.rglob("robot_data.json")
        if path.parent.name == take_id
    ]
    _require(len(matches) == 1, f"archive contains {len(matches)} roots for {take_id}")
    return matches[0].resolve()


def _stage_take(
    *,
    take_id: str,
    archive: Mapping[str, Any],
    configured_dataset_root: Path | None,
    search_roots: Iterable[Path],
    stage_root: Path,
) -> tuple[Path, dict[str, Any]]:
    archive_path = Path(str(archive["path"]))
    _require(archive_path.is_file(), f"PokeFlex archive is missing: {archive_path}")
    observed_size = archive_path.stat().st_size
    _require(
        observed_size == int(archive["size_bytes"]),
        f"PokeFlex archive size changed: {archive_path}",
    )
    observed_sha = _sha256(archive_path)
    _require(
        observed_sha == str(archive["sha256"]),
        f"PokeFlex archive checksum changed: {archive_path}",
    )

    take_root = _find_take_root(
        take_id,
        configured_dataset_root=configured_dataset_root,
        search_roots=search_roots,
    )
    extraction_performed = False
    if take_root is None:
        extract_root = stage_root / "extracted"
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as handle:
            handle.extractall(extract_root)
        take_root = _locate_extracted_take(extract_root, take_id)
        extraction_performed = True

    staged_take = stage_root / take_id
    if staged_take.exists() or staged_take.is_symlink():
        staged_take.unlink()
    staged_take.symlink_to(take_root, target_is_directory=True)
    _require((staged_take / "robot_data.json").is_file(), "staged take is invalid")
    return staged_take, {
        "take_id": take_id,
        "archive": str(archive_path.resolve()),
        "archive_size_bytes": observed_size,
        "archive_sha256": observed_sha,
        "take_root": str(take_root),
        "staged_take": str(staged_take),
        "extraction_performed": extraction_performed,
    }


def _valid_upstream(checkout: Path, expected_commit: str) -> bool:
    return (checkout / "models").is_dir() and _git_head(checkout) == expected_commit


def _resolve_upstream(
    upstream: Mapping[str, Any],
    *,
    configured_checkout: Path | None,
    search_roots: Iterable[Path],
    software_root: Path,
) -> tuple[Path, dict[str, Any]]:
    expected_commit = str(upstream["code_commit"])
    repository = str(upstream["repository"])
    if configured_checkout is not None and _valid_upstream(
        configured_checkout, expected_commit
    ):
        return configured_checkout.resolve(), {
            "source": "configured",
            "repository": repository,
            "commit": expected_commit,
        }

    for models_path in _walk_files(search_roots, filename="__init__.py"):
        if models_path.parent.name != "models":
            continue
        checkout = models_path.parent.parent
        if _valid_upstream(checkout, expected_commit):
            return checkout.resolve(), {
                "source": "discovered",
                "repository": repository,
                "commit": expected_commit,
            }

    checkout = software_root / "reconstruction"
    if checkout.exists():
        shutil.rmtree(checkout)
    _run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            repository,
            str(checkout),
        ]
    )
    _run(["git", "checkout", "--detach", expected_commit], cwd=checkout)
    _require(_valid_upstream(checkout, expected_commit), "cloned upstream is invalid")
    return checkout.resolve(), {
        "source": "cloned",
        "repository": repository,
        "commit": expected_commit,
    }


def _valid_checkpoint_root(
    root: Path, checkpoint_records: Mapping[str, Mapping[str, Any]]
) -> bool:
    for filename, metadata in checkpoint_records.items():
        path = root / filename
        if not path.is_file() or _sha256(path) != str(metadata["sha256"]):
            return False
    return True


def _resolve_checkpoints(
    checkpoint_records: Mapping[str, Mapping[str, Any]],
    *,
    configured_root: Path | None,
    search_roots: Iterable[Path],
    software_root: Path,
) -> tuple[Path, dict[str, Any]]:
    if configured_root is not None and _valid_checkpoint_root(
        configured_root, checkpoint_records
    ):
        return configured_root.resolve(), {"source": "configured"}

    anchor_filename = sorted(checkpoint_records)[0]
    for anchor in _walk_files(search_roots, filename=anchor_filename):
        candidate = anchor.parent
        if _valid_checkpoint_root(candidate, checkpoint_records):
            return candidate.resolve(), {"source": "discovered"}

    try:
        import gdown
    except ImportError as error:
        raise RuntimeError(
            "gdown is required to retrieve released checkpoints"
        ) from error

    root = software_root / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    downloads = []
    for filename, metadata in sorted(checkpoint_records.items()):
        target = root / filename
        result = gdown.download(
            id=str(metadata["google_drive_id"]),
            output=str(target),
            quiet=False,
        )
        _require(
            result is not None and target.is_file(), f"download failed: {filename}"
        )
        digest = _sha256(target)
        _require(digest == str(metadata["sha256"]), f"checksum failed: {filename}")
        downloads.append({"filename": filename, "sha256": digest})
    _require(_valid_checkpoint_root(root, checkpoint_records), "downloads are invalid")
    return root.resolve(), {"source": "downloaded", "files": downloads}


def prepare_assets(args: argparse.Namespace) -> dict[str, Any]:
    result_path = args.prospective_result.resolve()
    manifest_path = args.execution_manifest.resolve()
    protocol_path = args.registration_protocol.resolve()
    result = _load_json(result_path)
    manifest = _load_json(manifest_path)
    protocol = _load_json(protocol_path)

    _require(result.get("gate_passed") is True, "prospective result did not pass")
    _require(manifest.get("replacement_performed") is False, "replacement occurred")
    _require(
        manifest["prospective_evaluation"]["sha256"] == _sha256(result_path),
        "prospective result checksum changed",
    )
    _require(
        manifest["protocol_sha256"]
        == _load_json(args.prospective_protocol.resolve())["protocol_sha256"],
        "prospective protocol checksum changed",
    )

    output_root = args.output_root.resolve()
    candidate_stage = output_root / "candidates"
    dataset_stage = output_root / "dataset"
    software_root = output_root / "software"
    for path in (candidate_stage, dataset_stage, software_root):
        path.mkdir(parents=True, exist_ok=True)

    requested = args.take_id.strip() if args.take_id else None
    take_id = _select_take(result, requested)
    archive = next(
        value for value in manifest["archives"] if value["take_id"] == take_id
    )
    candidate_root = args.candidate_root.resolve() if args.candidate_root else None
    configured_dataset = args.dataset_root.resolve() if args.dataset_root else None
    configured_upstream = (
        args.upstream_checkout.resolve() if args.upstream_checkout else None
    )
    configured_checkpoints = (
        args.checkpoint_root.resolve() if args.checkpoint_root else None
    )
    search_roots = tuple(
        path.resolve() for path in (args.search_root or DEFAULT_SEARCH_ROOTS)
    )

    staged_candidates = _stage_candidate_artifacts(
        result,
        candidate_root=candidate_root,
        search_roots=search_roots,
        output_root=candidate_stage,
    )
    staged_take, take_evidence = _stage_take(
        take_id=take_id,
        archive=archive,
        configured_dataset_root=configured_dataset,
        search_roots=search_roots,
        stage_root=dataset_stage,
    )
    registration_payload = protocol.get("payload", protocol)
    upstream, upstream_evidence = _resolve_upstream(
        registration_payload["upstream"],
        configured_checkout=configured_upstream,
        search_roots=search_roots,
        software_root=software_root,
    )
    checkpoint_records = registration_payload["upstream"]["released_kinect_checkpoint"]
    checkpoints, checkpoint_evidence = _resolve_checkpoints(
        checkpoint_records,
        configured_root=configured_checkpoints,
        search_roots=search_roots,
        software_root=software_root,
    )

    assets = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexSameObjectWorkflowAssetsV1",
        "selected_take": take_id,
        "candidate_artifacts": staged_candidates,
        "take": take_evidence,
        "upstream": {
            **upstream_evidence,
            "checkout": str(upstream),
        },
        "checkpoints": {
            **checkpoint_evidence,
            "root": str(checkpoints),
            "sha256": {
                filename: _sha256(checkpoints / filename)
                for filename in sorted(checkpoint_records)
            },
        },
        "inputs": {
            "prospective_result": str(result_path),
            "prospective_result_sha256": _sha256(result_path),
            "execution_manifest": str(manifest_path),
            "execution_manifest_sha256": _sha256(manifest_path),
            "registration_protocol": str(protocol_path),
            "registration_protocol_sha256": protocol["protocol_sha256"],
        },
    }
    manifest_output = output_root / "workflow_assets.json"
    _write_json(manifest_output, assets)

    if args.github_env is not None:
        _append_environment(
            args.github_env.resolve(),
            {
                "POKEFLEX_SELECTED_TAKE": take_id,
                "POKEFLEX_SELECTED_ARCHIVE": archive["path"],
                "POKEFLEX_DATA_ROOT": dataset_stage,
                "POKEFLEX_TAKE_ROOT": staged_take,
                "POKEFLEX_UPSTREAM_CHECKOUT": upstream,
                "POKEFLEX_CHECKPOINT_ROOT": checkpoints,
                "POKEFLEX_WORKFLOW_ASSETS": manifest_output,
            },
        )
    return assets


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
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
        "--registration-protocol",
        type=Path,
        default=(
            repository_root / "configs/sota/pokeflex_bayesian_registration_v1.json"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--take-id", default="")
    parser.add_argument("--candidate-root", type=_optional_path)
    parser.add_argument("--dataset-root", type=_optional_path)
    parser.add_argument("--upstream-checkout", type=_optional_path)
    parser.add_argument("--checkpoint-root", type=_optional_path)
    parser.add_argument("--search-root", type=Path, action="append")
    parser.add_argument("--github-env", type=Path)
    args = parser.parse_args()
    try:
        assets = prepare_assets(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(assets, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
