#!/usr/bin/env python3
# ruff: noqa: E402, F403, F405
"""Acquire and prepare only the locked Deform360 calibration cohort."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _deform360_calibration_acquisition_runtime_common import *
from _deform360_calibration_acquisition_runtime_download import *
from _deform360_calibration_acquisition_runtime_process import *

def _write_status(
    path: Path,
    *,
    result: Mapping[str, Any],
    cases: Sequence[Deform360CalibrationAcquisitionCaseV1],
) -> None:
    lines = [
        "# Deform360 calibration acquisition",
        "",
        f"- Status: `{result['status']}`",
        f"- Prepared objects: {result['prepared_object_count']}/10",
        f"- Technical failures: {result['technical_failure_count']}/10",
        f"- Result ID: `{result['result_id']}`",
        f"- Evidence-use ledger: `{result['evidence_use_ledger_id']}`",
        "- Calibration payloads opened: `true`",
        "- Confirmation payloads opened: `false`",
        "- Target outcomes used: `false`",
        "- Replacement allowed: `false`",
        "",
        "## Per-object disposition",
        "",
    ]
    for case in sorted(cases, key=lambda item: (item.stratum, item.object_id)):
        detail = (
            f"{case.aligned_frame_count} frames"
            if case.status == "prepared"
            else f"{case.failure_stage}/{case.failure_type}"
        )
        lines.append(
            f"- `{case.object_id}` episode {case.episode_id}: "
            f"**{case.status}** ({detail})"
        )
    lines.extend(
        [
            "",
            "This artifact records source acquisition and information order only; ",
            "it is not empirical performance evidence.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--stage0-selection", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--visual-provider-lock", type=Path, required=True)
    parser.add_argument("--deform360-checkout", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument(
        "--token-environment-variable",
        default="HF_TOKEN",
        help="environment variable containing an optional Hugging Face token",
    )
    parser.add_argument(
        "--open-calibration-payloads",
        action="store_true",
        help="required acknowledgement that the ten locked calibration units open",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if not arguments.open_calibration_payloads:
        raise SystemExit("--open-calibration-payloads is required")

    repository = arguments.repository.resolve()
    deform360_checkout = arguments.deform360_checkout.resolve()
    data_root = arguments.data_root.expanduser().resolve()
    output = arguments.output_dir.resolve()
    _require_separate_data_and_output_roots(
        repository=repository,
        deform360_checkout=deform360_checkout,
        data_root=data_root,
        output=output,
    )
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cases_root = output / "cases"
    failures_root = output / "failures"
    cases_root.mkdir()

    observed_implementation = _git_head(repository, name="BayesianPhysTwin")
    if observed_implementation != arguments.implementation_revision:
        raise ValueError(
            "implementation revision differs from the exact BayesianPhysTwin checkout"
        )
    allowed_nested_checkouts: tuple[str, ...] = ()
    if deform360_checkout.is_relative_to(repository):
        relative_checkout = deform360_checkout.relative_to(repository).as_posix()
        if relative_checkout != "_deform360":
            raise ValueError(
                "nested Deform360 checkout must use the isolated _deform360 path"
            )
        allowed_nested_checkouts = (relative_checkout,)
    _require_clean(
        repository,
        name="BayesianPhysTwin",
        allowed_untracked=allowed_nested_checkouts,
    )
    _require_bayesian_phystwin_import(repository)

    plan = build_calibration_acquisition_plan(
        stage0_selection_path=arguments.stage0_selection,
        visual_provider_lock_path=arguments.visual_provider_lock,
        implementation_revision=arguments.implementation_revision,
        protocol_path=arguments.protocol,
    )
    observed_processing = _git_head(deform360_checkout, name="Deform360")
    if observed_processing != plan.processing_revision:
        raise ValueError(
            "Deform360 checkout differs from the locked processing revision"
        )
    _require_clean(deform360_checkout, name="Deform360")
    _require_deform360_import(deform360_checkout)
    plan_path = output / "acquisition-plan.json"
    save_calibration_acquisition_plan(plan_path, plan)

    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "raw").mkdir(exist_ok=True)
    validate_calibration_download_root(data_root, plan, require_complete=False)

    token = os.environ.get(str(arguments.token_environment_variable))
    selected_paths_by_object: dict[str, tuple[str, ...]] = {}
    for unit in plan.calibration_units:
        selected_paths_by_object[unit.object_id] = _selected_unit_paths(
            repository=plan.dataset_repository,
            revision=plan.dataset_revision,
            object_id=unit.object_id,
            episode_id=unit.episode_id,
            token=token,
        )
    allowlist = _payload_allowlist(plan, selected_paths_by_object)
    allowlist_path = output / "payload-allowlist.json"
    write_atomic_json(allowlist, allowlist_path, overwrite=False)
    validate_calibration_download_root(
        data_root,
        plan,
        require_complete=False,
        expected_paths_by_object=selected_paths_by_object,
    )

    downloads: list[dict[str, Any]] = []
    for unit in plan.calibration_units:
        downloads.append(
            _download_unit(
                repository=plan.dataset_repository,
                revision=plan.dataset_revision,
                object_id=unit.object_id,
                episode_id=unit.episode_id,
                selected_paths=selected_paths_by_object[unit.object_id],
                expected_metadata_sha256=unit.metadata_sha256,
                data_root=data_root,
                token=token,
            )
        )
    validate_calibration_download_root(
        data_root,
        plan,
        require_complete=True,
        expected_paths_by_object=selected_paths_by_object,
    )
    download_manifest = _download_manifest(
        plan,
        payload_allowlist_id=str(allowlist["allowlist_id"]),
        downloads=downloads,
    )
    download_path = output / "download-manifest.json"
    write_atomic_json(download_manifest, download_path, overwrite=False)
    downloads_by_object = {str(item["object_id"]): item for item in downloads}

    cases: list[Deform360CalibrationAcquisitionCaseV1] = []
    for unit in plan.calibration_units:
        raw_artifacts = _raw_artifacts_from_download(
            downloads_by_object[unit.object_id]
        )
        case = _process_case(
            plan_id=plan.plan_id,
            object_id=unit.object_id,
            episode_id=unit.episode_id,
            stratum=unit.stratum,
            data_root=data_root,
            raw_artifacts=raw_artifacts,
            failure_root=failures_root,
        )
        save_calibration_acquisition_case(
            cases_root / f"{unit.object_id}-episode-{unit.episode_id:04d}.json",
            case,
        )
        cases.append(case)

    validate_calibration_download_root(
        data_root,
        plan,
        require_complete=True,
        expected_paths_by_object=selected_paths_by_object,
    )
    _require_clean(
        repository,
        name="BayesianPhysTwin",
        allowed_untracked=allowed_nested_checkouts,
    )
    _require_clean(deform360_checkout, name="Deform360")
    ledger = build_calibration_evidence_ledger(plan, cases)
    ledger_path = output / "evidence-use-ledger.json"
    save_calibration_evidence_ledger(ledger_path, ledger)

    source_paths = [
        plan_path,
        allowlist_path,
        download_path,
        ledger_path,
        *_all_files(cases_root),
    ]
    source_artifacts = _relative_artifacts(output, source_paths)
    source_artifacts.update(
        {
            "inputs/protocol.json": file_sha256(arguments.protocol),
            "inputs/stage0-selection.json": file_sha256(
                arguments.stage0_selection
            ),
            "inputs/visual-provider-lock.json": file_sha256(
                arguments.visual_provider_lock
            ),
            "implementation/deform360_calibration_acquisition.py": file_sha256(
                repository
                / "src"
                / "bayesian_phystwin"
                / "deform360_calibration_acquisition.py"
            ),
            "implementation/run_deform360_calibration_acquisition.py": file_sha256(
                repository
                / "scripts"
                / "science"
                / "run_deform360_calibration_acquisition.py"
            ),
        }
    )
    result = build_calibration_acquisition_result(
        plan,
        cases,
        ledger,
        source_artifacts=source_artifacts,
        metadata={
            "processing_stages": ["undistort", "tactile", "robot"],
            "processing_repository": plan.processing_repository,
            "processing_revision": plan.processing_revision,
            "payload_allowlist_id": allowlist["allowlist_id"],
            "download_manifest_id": download_manifest["manifest_id"],
            "data_root_recorded": False,
            "raw_or_processed_payload_bytes_published": False,
        },
    )
    result_path = output / "acquisition-result.json"
    save_calibration_acquisition_result(result_path, result)
    _write_status(output / "STATUS.md", result=result, cases=cases)

    checksum_paths = [
        plan_path,
        allowlist_path,
        download_path,
        ledger_path,
        result_path,
        output / "STATUS.md",
        *_all_files(cases_root),
        *_all_files(failures_root),
    ]
    checksums = "".join(
        f"{file_sha256(path)}  {path.relative_to(output).as_posix()}\n"
        for path in sorted(set(checksum_paths))
        if path.is_file()
    )
    (output / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
