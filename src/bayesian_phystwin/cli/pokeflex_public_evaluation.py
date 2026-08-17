"""Run the frozen, source-only PokeFlex public evaluation boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bayesian_phystwin.pokeflex_independent_depth_protocol import (
    load_pokeflex_independent_depth_protocol,
)
from bayesian_phystwin.pokeflex_registration_protocol import (
    load_pokeflex_registration_protocol,
)

PROFILE_CONTRACTS = "contracts"
PROFILE_SOURCE_VALIDATION = "source-validation"
PROFILES = (PROFILE_CONTRACTS, PROFILE_SOURCE_VALIDATION)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_source_protocol() -> Path:
    return (
        _repository_root()
        / "configs"
        / "sota"
        / "pokeflex_independent_depth_source_validation_v2.json"
    )


def _default_registration_protocol() -> Path:
    return (
        _repository_root()
        / "configs"
        / "sota"
        / "pokeflex_bayesian_registration_v1.json"
    )


def _source_runner() -> Path:
    return (
        _repository_root()
        / "scripts"
        / "remote"
        / "run_pokeflex_independent_depth_source_validation.py"
    )


def _candidate_runner() -> Path:
    return (
        _repository_root()
        / "scripts"
        / "remote"
        / "run_pokeflex_checkpoint_registration_independent_depth.py"
    )


def _analysis_runner() -> Path:
    return (
        _repository_root()
        / "scripts"
        / "remote"
        / "evaluate_pokeflex_independent_depth_source.py"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _git_head(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _expected_take_ids(source: dict[str, Any]) -> tuple[str, ...]:
    boundary = source["payload"]["evidence_boundary"]
    return tuple(
        f"{object_name}_{take}"
        for object_name in boundary["development_objects"]
        for take in boundary["source_validation_takes"]
    )


def _load_contracts(
    source_protocol: Path,
    registration_protocol: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = load_pokeflex_independent_depth_protocol(source_protocol)
    registration = load_pokeflex_registration_protocol(registration_protocol)
    parent = source["payload"]["parent_protocol"]
    if parent["protocol_sha256"] != registration["protocol_sha256"]:
        raise ValueError("PokeFlex source protocol parent changed")

    causal = source["payload"]["causal_input_contract"]
    required = {
        "kinect_history": "f-5 through f-1 only",
        "realsense_history": "f-5 through f-1 only",
        "robot_history": "through f-1 only",
        "frame_f_kinect_or_realsense_allowed_before_prediction": False,
        "frame_f_mesh_allowed_before_scoring": False,
    }
    for key, expected in required.items():
        if causal.get(key) != expected:
            raise ValueError(f"PokeFlex causal contract changed: {key}")
    if source["payload"]["evidence_boundary"].get("replacement_allowed") is not False:
        raise ValueError("PokeFlex technical failures may not be replaced")
    if (
        source["payload"]["method_lock"].get("state_innovation_processed_once")
        is not True
    ):
        raise ValueError("PokeFlex state innovation must be processed once")
    if (
        registration["payload"]["methods"]["candidate_constraints"].get(
            "observation_reliability_is_residual_independent"
        )
        is not True
    ):
        raise ValueError("PokeFlex reliability became outcome-dependent")
    for path in (_source_runner(), _candidate_runner(), _analysis_runner()):
        if not path.is_file():
            raise FileNotFoundError(f"checked-in PokeFlex runner is missing: {path}")
    return source, registration


def _contract_report(
    source: dict[str, Any], registration: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_kind": "PokeFlexPublicEvaluationContract",
        "profile": PROFILE_CONTRACTS,
        "authorized_profiles": list(PROFILES),
        "source_protocol_id": source["payload"]["protocol_id"],
        "source_protocol_sha256": source["protocol_sha256"],
        "registration_protocol_id": registration["payload"]["protocol_id"],
        "registration_protocol_sha256": registration["protocol_sha256"],
        "source_runner_sha256": _sha256(_source_runner()),
        "candidate_runner_sha256": _sha256(_candidate_runner()),
        "analysis_runner_sha256": _sha256(_analysis_runner()),
        "authorized_take_count": len(_expected_take_ids(source)),
        "causal_history": "f-5 through f-1 only",
        "target_geometry_role": "scoring only",
        "replacement_allowed": False,
        "claim_boundary": (
            "source-only retrospective/exploratory evaluation; not official-18, "
            "prospective confirmation, or a state-of-the-art claim"
        ),
    }


def _require_directory(path: Path | None, label: str) -> Path:
    if path is None:
        raise ValueError(f"--{label.replace('_', '-')} is required")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def _verify_checkpoint_files(
    checkpoint_root: Path, registration: dict[str, Any]
) -> dict[str, str]:
    expected = registration["payload"]["upstream"]["released_kinect_checkpoint"]
    observed: dict[str, str] = {}
    for filename, metadata in sorted(expected.items()):
        path = checkpoint_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing PokeFlex checkpoint: {path}")
        digest = _sha256(path)
        if digest != metadata["sha256"]:
            raise ValueError(f"PokeFlex checkpoint checksum changed: {filename}")
        observed[filename] = digest
    return observed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_source_validation(
    args: argparse.Namespace,
    source: dict[str, Any],
    registration: dict[str, Any],
) -> int:
    dataset_root = _require_directory(args.dataset_root, "dataset_root")
    upstream_checkout = _require_directory(args.upstream_checkout, "upstream_checkout")
    checkpoint_root = _require_directory(args.checkpoint_root, "checkpoint_root")
    if args.output_root is None:
        raise ValueError("--output-root is required")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output root must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    expected_take_ids = _expected_take_ids(source)
    selected_take_ids = tuple(args.take_id or expected_take_ids)
    unknown = sorted(set(selected_take_ids) - set(expected_take_ids))
    if unknown:
        raise ValueError(f"takes are outside the frozen source panel: {unknown}")
    missing = [
        take_id
        for take_id in selected_take_ids
        if not (dataset_root / take_id).is_dir()
    ]
    if missing:
        raise ValueError(f"PokeFlex source takes are missing: {missing}")

    expected_upstream = registration["payload"]["upstream"]["code_commit"]
    observed_upstream = _git_head(upstream_checkout)
    if observed_upstream != expected_upstream:
        raise ValueError(
            "PokeFlex upstream revision changed: "
            f"expected {expected_upstream}, received {observed_upstream}"
        )
    checkpoint_hashes = _verify_checkpoint_files(checkpoint_root, registration)
    repository_revision = _git_head(_repository_root())

    manifest = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexPublicEvaluationExecutionManifest",
        "profile": PROFILE_SOURCE_VALIDATION,
        "repository_revision": repository_revision,
        "source_protocol_sha256": source["protocol_sha256"],
        "registration_protocol_sha256": registration["protocol_sha256"],
        "source_runner_sha256": _sha256(_source_runner()),
        "candidate_runner_sha256": _sha256(_candidate_runner()),
        "analysis_runner_sha256": _sha256(_analysis_runner()),
        "upstream_revision": observed_upstream,
        "checkpoint_sha256": checkpoint_hashes,
        "selected_take_ids": list(selected_take_ids),
        "causal_history": "f-5 through f-1 only",
        "target_geometry_role": "scoring only after the method and take panel are sealed",
        "replacement_allowed": False,
        "claim_boundary": (
            "source-only retrospective/exploratory evaluation; not official-18, "
            "prospective confirmation, or a state-of-the-art claim"
        ),
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    _write_json(output_root / "execution_manifest.json", manifest)

    command = [
        sys.executable,
        str(_source_runner()),
        str(dataset_root),
        str(output_root),
        "--upstream-checkout",
        str(upstream_checkout),
        "--checkpoint-root",
        str(checkpoint_root),
        "--source-protocol",
        str(args.source_protocol.resolve()),
        "--parent-protocol",
        str(args.registration_protocol.resolve()),
    ]
    for take_id in selected_take_ids:
        command.extend(("--take-id", take_id))

    result = subprocess.run(command, check=False)
    progress_path = output_root / "source_validation_progress_v2.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexPublicEvaluationSummary",
        "profile": PROFILE_SOURCE_VALIDATION,
        "execution_manifest_sha256": manifest["manifest_sha256"],
        "runner_exit_code": result.returncode,
        "replacement_allowed": False,
    }
    final_exit_code = int(result.returncode)
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        records = progress.get("records", [])
        if progress.get("protocol_sha256") != source["protocol_sha256"]:
            raise ValueError("PokeFlex progress protocol changed")
        if progress.get("replacement_allowed") is not False:
            raise ValueError("PokeFlex progress permits replacement")
        if [record.get("take_id") for record in records] != list(selected_take_ids):
            raise ValueError("PokeFlex progress take order or membership changed")
        summary["record_count"] = len(records)
        summary["status_counts"] = {
            status: sum(record.get("status") == status for record in records)
            for status in ("completed", "existing", "failed-no-replacement")
        }
        successful = all(
            record.get("status") in {"completed", "existing"} for record in records
        ) and len(records) == len(selected_take_ids)
        summary["run_complete"] = successful
        if not successful:
            final_exit_code = final_exit_code or 1

        full_panel = selected_take_ids == expected_take_ids
        if successful and full_panel:
            analysis_path = output_root / "source_validation_analysis.json"
            analysis_command = [
                sys.executable,
                str(_analysis_runner()),
                *(str(Path(record["output"]).resolve()) for record in records),
                "--output",
                str(analysis_path),
                "--source-validation-protocol",
                str(args.source_protocol.resolve()),
                "--compact",
            ]
            analysis_result = subprocess.run(analysis_command, check=False)
            summary["analysis_status"] = (
                "completed" if analysis_result.returncode == 0 else "failed"
            )
            summary["analysis_exit_code"] = analysis_result.returncode
            if analysis_path.is_file():
                analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
                summary["analysis_sha256"] = _sha256(analysis_path)
                summary["object_balanced_selector"] = analysis.get(
                    "object_balanced_selector"
                )
                summary["registered_gate"] = analysis.get("registered_gate")
            final_exit_code = final_exit_code or int(analysis_result.returncode)
        elif successful:
            summary["analysis_status"] = "not-run-subset"
        else:
            summary["analysis_status"] = "not-run-incomplete"
    else:
        summary["record_count"] = 0
        summary["status_counts"] = {}
        summary["run_complete"] = False
        summary["analysis_status"] = "not-run-missing-progress"
        final_exit_code = final_exit_code or 1
    _write_json(output_root / "evaluation_summary.json", summary)
    return final_exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default=PROFILE_CONTRACTS)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--upstream-checkout", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument(
        "--source-protocol", type=Path, default=_default_source_protocol()
    )
    parser.add_argument(
        "--registration-protocol",
        type=Path,
        default=_default_registration_protocol(),
    )
    parser.add_argument(
        "--take-id",
        action="append",
        help="Run an authorized source take; repeat to select multiple takes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source, registration = _load_contracts(
        args.source_protocol.resolve(),
        args.registration_protocol.resolve(),
    )
    if args.profile == PROFILE_CONTRACTS:
        print(
            json.dumps(_contract_report(source, registration), indent=2, sort_keys=True)
        )
        return 0
    return _run_source_validation(args, source, registration)


if __name__ == "__main__":
    raise SystemExit(main())
