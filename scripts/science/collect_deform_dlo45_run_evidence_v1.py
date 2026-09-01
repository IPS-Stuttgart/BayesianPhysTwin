#!/usr/bin/env python3
"""Collect bounded, text-only evidence from one DEFORM DLO4/DLO5 Actions run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

SCHEMA = "bayesian-phystwin/deform-dlo45-run-evidence-collection-v1"
TEXT_EXTENSIONS = {".json", ".md", ".csv", ".txt", ".jsonl", ".log", ".yml", ".yaml"}
PRIORITY_NAMES = {
    "result.json",
    "report.md",
    "trajectory-results.csv",
    "source_result.json",
    "prediction_seal.json",
    "joint_prediction_seal.json",
    "target_authorization.json",
    "progress.json",
    "failure_receipt.json",
    "preflight.json",
    "method_seal.json",
    "protocol.json",
}
MAX_COPY_BYTES = 5 * 1024 * 1024
MAX_LOG_BYTES = 2 * 1024 * 1024
LOG_TAIL_LINES = 250


class CollectionError(RuntimeError):
    """Raised when the retained artifact bundle violates the collector contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CollectionError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "suffix": path.suffix.lower(),
            }
        )
    return rows


def _safe_relative(raw: str) -> Path:
    relative = Path(raw)
    _require(not relative.is_absolute(), f"artifact path is absolute: {raw}")
    _require(".." not in relative.parts, f"artifact path escapes root: {raw}")
    return relative


def _copy_bounded(source: Path, destination: Path) -> dict[str, object] | None:
    size = source.stat().st_size
    suffix = source.suffix.lower()
    if suffix not in TEXT_EXTENSIONS:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".log" and size > MAX_LOG_BYTES:
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
        rendered = "\n".join(lines[-LOG_TAIL_LINES:]) + "\n"
        tail = destination.with_suffix(destination.suffix + ".tail.txt")
        tail.write_text(rendered, encoding="utf-8")
        return {
            "source_size_bytes": size,
            "retained_path": str(tail),
            "retention": f"last-{LOG_TAIL_LINES}-lines",
            "retained_size_bytes": tail.stat().st_size,
            "retained_sha256": _sha256(tail),
        }
    if size > MAX_COPY_BYTES and source.name not in PRIORITY_NAMES:
        return None
    shutil.copyfile(source, destination)
    return {
        "source_size_bytes": size,
        "retained_path": str(destination),
        "retention": "complete",
        "retained_size_bytes": destination.stat().st_size,
        "retained_sha256": _sha256(destination),
    }


def _interesting(value: object, *, depth: int = 0) -> object:
    if depth > 4:
        return None
    if isinstance(value, Mapping):
        selected: dict[str, object] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            lower = key.casefold()
            keep = any(
                token in lower
                for token in (
                    "decision",
                    "passed",
                    "l1",
                    "improvement",
                    "wins",
                    "losses",
                    "ties",
                    "ratio",
                    "coverage",
                    "nees",
                    "dlo4",
                    "dlo5",
                    "candidate",
                    "baseline",
                    "contract",
                )
            )
            nested = _interesting(item, depth=depth + 1)
            if keep and isinstance(item, (str, int, float, bool, type(None))):
                selected[key] = item
            elif nested not in (None, {}, []):
                selected[key] = nested
        return selected
    if isinstance(value, list):
        if len(value) > 24:
            return None
        nested = [_interesting(item, depth=depth + 1) for item in value]
        return [item for item in nested if item not in (None, {}, [])]
    return None


def _json_candidates(root: Path) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(value, Mapping):
            continue
        relative = path.relative_to(root).as_posix()
        contract = value.get("contract", value.get("schema"))
        decision = value.get("decision")
        is_result = path.name == "result.json" or decision is not None
        if not is_result:
            continue
        candidates.append(
            {
                "path": relative,
                "contract": contract,
                "decision": decision,
                "summary": _interesting(value),
            }
        )
    return candidates


def collect(
    artifact_root: Path,
    output_root: Path,
    run_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Collect a content-addressed, text-only evidence subset."""

    source = artifact_root.resolve(strict=True)
    destination = output_root.resolve()
    _require(source.is_dir(), "artifact root must be a directory")
    _require(not destination.exists(), "output root already exists")
    inventory = _inventory(source)
    _require(inventory, "artifact download is empty")
    destination.mkdir(parents=True)
    retained_root = destination / "retained"
    retained: list[dict[str, object]] = []
    for row in inventory:
        relative = _safe_relative(str(row["path"]))
        source_path = source / relative
        retained_path = retained_root / relative
        copied = _copy_bounded(source_path, retained_path)
        if copied is None:
            continue
        retained.append(
            {
                "source_path": relative.as_posix(),
                "source_sha256": row["sha256"],
                **copied,
                "retained_path": Path(str(copied["retained_path"]))
                .relative_to(destination)
                .as_posix(),
            }
        )
    candidates = _json_candidates(retained_root)
    terminal = run_metadata.get("status") == "completed"
    result: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "run": dict(run_metadata),
        "blocking_run_terminal": terminal,
        "source_artifact_file_count": len(inventory),
        "retained_text_file_count": len(retained),
        "result_candidate_count": len(candidates),
        "result_candidates": candidates,
        "inventory": inventory,
        "retained": retained,
        "information_boundary": {
            "artifact_download_only": True,
            "self_hosted_runner_used": False,
            "target_rerun": False,
            "method_change": False,
            "paper_claim_authorized": False,
        },
        "claim_boundary": (
            "This collector preserves bounded text evidence from the already-"
            "executed DLO4/DLO5 workflow. It neither reruns the experiment nor "
            "authorizes a paper claim; scientific interpretation must follow the "
            "frozen result contract and retained prediction seals."
        ),
    }
    (destination / "collection.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# DEFORM DLO4/DLO5 run evidence collection",
        "",
        f"- Run ID: `{run_metadata.get('id')}`",
        f"- Status/conclusion: `{run_metadata.get('status')}` / "
        f"`{run_metadata.get('conclusion')}`",
        f"- Source artifact files: `{len(inventory)}`",
        f"- Retained text files: `{len(retained)}`",
        f"- Result candidates: `{len(candidates)}`",
        "",
        "## Result candidates",
        "",
    ]
    if not candidates:
        lines.append(
            "No JSON result candidate was present in the downloaded artifacts."
        )
    for candidate in candidates:
        lines.extend(
            [
                f"### `{candidate['path']}`",
                "",
                f"- Contract: `{candidate['contract']}`",
                f"- Decision: `{candidate['decision']}`",
                "",
                "```json",
                json.dumps(candidate["summary"], indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    lines.extend(["## Claim boundary", "", str(result["claim_boundary"])])
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    checksum_rows = []
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksum_rows.append(
                f"{_sha256(path)}  {path.relative_to(destination).as_posix()}"
            )
    (destination / "SHA256SUMS").write_text(
        "\n".join(checksum_rows) + "\n", encoding="utf-8"
    )
    return result


def _self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        artifacts = root / "artifacts" / "bundle"
        artifacts.mkdir(parents=True)
        (artifacts / "score").mkdir()
        (artifacts / "score" / "result.json").write_text(
            json.dumps(
                {
                    "contract": "fixture-deform-dlo45-result-v1",
                    "decision": "fixture-pass",
                    "dlo4": {"relative_improvement": 0.1, "wins": 12},
                    "dlo5": {"relative_improvement": 0.2, "wins": 13},
                    "paper_claim_authorized": False,
                }
            ),
            encoding="utf-8",
        )
        (artifacts / "large.log").write_text(
            "".join(f"line-{index}\n" for index in range(300000)),
            encoding="utf-8",
        )
        output = root / "output"
        result = collect(
            root / "artifacts",
            output,
            {"id": 1, "status": "completed", "conclusion": "success"},
        )
        _require(result["result_candidate_count"] == 1, "fixture result count")
        _require(
            any(
                str(row["retained_path"]).endswith(".tail.txt")
                for row in result["retained"]
            ),
            "fixture log was not bounded",
        )
        _require((output / "SHA256SUMS").is_file(), "fixture checksums missing")
    print("self-test passed")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-metadata", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.self_test:
        _self_test()
        return 0
    _require(args.artifact_root is not None, "--artifact-root is required")
    _require(args.output_root is not None, "--output-root is required")
    _require(args.run_metadata is not None, "--run-metadata is required")
    metadata = json.loads(
        args.run_metadata.resolve(strict=True).read_text(encoding="utf-8")
    )
    _require(isinstance(metadata, Mapping), "run metadata must be a JSON object")
    result = collect(args.artifact_root, args.output_root, metadata)
    print(
        json.dumps(
            {
                "run_id": result["run"]["id"],
                "result_candidate_count": result["result_candidate_count"],
                "retained_text_file_count": result["retained_text_file_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CollectionError as error:
        raise SystemExit(f"evidence collection failed: {error}") from error
