#!/usr/bin/env python3
"""Run and seal one dynamic TAPNext++ provider prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_dynamic_query import (
    build_dynamic_query_schedule,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_artifacts import (
    build_prediction_seal,
    record_technical_failure,
    validate_source_admission,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_assimilation import (
    build_birth_anchored_measurements,
    predict_dynamic_tapnextpp_candidate,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_cohort import (
    dynamic_provider_case_record,
    load_dynamic_provider_cohort_lock,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_physical import (
    PHYSICAL_MANIFEST_FILENAME,
    validate_dynamic_physical_artifacts,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_provider import (
    ASSIMILATION_REPORT_FILENAME,
    QUERY_SCHEDULE_FILENAME,
    RUNTIME_REPORT_FILENAME,
    load_camera_geometry,
    load_selected_causal_inputs,
    write_assimilation_artifacts,
    write_provider_artifacts,
    write_query_schedule_artifact,
)
from bayesian_phystwin.deform360_object_exclusion import file_sha256
from bayesian_phystwin.observation_belief import save_observation_belief
from bayesian_phystwin.tapnextpp_dynamic_multiview import (
    PROTOCOL_ID,
    TAPNEXT_CHECKPOINT_SHA256,
    TAPNEXT_REVISION,
    build_dynamic_tapnextpp_observation_belief,
    fuse_dynamic_tapnextpp_multiview,
)
from bayesian_phystwin.tapnextpp_dynamic_runtime import (
    build_dynamic_birth_associations,
    run_dynamic_tapnextpp_births,
)

EXPECTED_QUERY_COUNT = 72
OBSERVATION_BELIEF_FILENAME = "dynamic_tapnextpp_observation_belief.npz"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "JSON artifact must contain an object")
    return payload


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_output(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_repository(repo: Path) -> str:
    revision = _git_output(repo, "rev-parse", "HEAD")
    _require(
        len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision),
        "provider repository revision is invalid",
    )
    _require(
        not _git_output(repo, "status", "--porcelain"),
        "provider repository contains uncommitted changes",
    )
    return revision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cohort-lock", type=Path, required=True)
    parser.add_argument("--partition", choices=("source", "target"), required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--processed-episode-dir", type=Path, required=True)
    parser.add_argument("--physical-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tapnet-root", type=Path, required=True)
    parser.add_argument("--tapnextpp-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _load_model(
    tapnet_root: Path,
    checkpoint: Path,
    device_name: str,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    revision = _git_output(tapnet_root, "rev-parse", "HEAD")
    _require(revision == TAPNEXT_REVISION, "TAPNet repository revision changed")
    _require(
        file_sha256(checkpoint) == TAPNEXT_CHECKPOINT_SHA256,
        "TAPNext++ checkpoint checksum changed",
    )
    if str(tapnet_root) not in sys.path:
        sys.path.insert(0, str(tapnet_root))
    from tapnet.tapnextpp.votsp2026 import utils as tapnext_utils
    from tapnet.tapnextpp.votsp2026.model import TAPNextPP

    device = torch.device(device_name)
    _require(device.type == "cuda", "frozen provider requires CUDA")
    torch.manual_seed(72)
    torch.cuda.manual_seed_all(72)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = TAPNextPP.from_checkpoint(
        checkpoint,
        device=device,
        half_precision=False,
        compile_model=False,
        input_resolution=512,
    )
    return (
        model,
        tapnext_utils,
        {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "tapnet_revision": revision,
            "tapnextpp_checkpoint_sha256": TAPNEXT_CHECKPOINT_SHA256,
            "deterministic_algorithms_requested": True,
        },
    )


def _environment_sha256(runtime: dict[str, Any], code_revision: str) -> str:
    import cv2
    import h5py
    import torch

    descriptor = {
        "code_revision": code_revision,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "h5py": h5py.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "runtime": runtime,
    }
    return _canonical_sha256(descriptor)


def _validate_locked_case_inputs(
    args: argparse.Namespace,
    code_revision: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cohort_path = args.cohort_lock.resolve()
    cohort = load_dynamic_provider_cohort_lock(cohort_path)
    _require(
        cohort["bindings"]["provider_commit"] == code_revision,
        "runtime revision differs from the locked provider commit",
    )
    protocol_path = args.protocol.resolve()
    protocol = _load_json(protocol_path)
    _require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol ID changed")
    _require(
        file_sha256(protocol_path)
        == cohort["bindings"]["provider_protocol_file_sha256"],
        "protocol differs from the cohort lock",
    )
    admission_source = _load_json(args.admission.resolve())
    admission = validate_source_admission(admission_source)
    _require(admission["admitted"] is True, "source admission did not pass")
    record = dynamic_provider_case_record(
        cohort,
        object_id=str(admission_source["object_id"]),
        episode_id=int(admission_source["episode_id"]),
        partition=args.partition,
    )
    _require(
        admission["case_hash"] == record["case_hash"]
        and admission["object_hash"] == record["object_hash"]
        and admission["source_admission_sha256"]
        == record["admission_sha256"],
        "source admission differs from the cohort lock",
    )
    return cohort, record


def _validate_physical_case_inputs(
    args: argparse.Namespace,
    record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    protocol_path = args.protocol.resolve()
    cohort_path = args.cohort_lock.resolve()
    physical_manifest, physical_arrays = validate_dynamic_physical_artifacts(
        args.physical_dir.resolve()
    )
    _require(
        physical_manifest["partition"] == args.partition
        and physical_manifest["case_hash"] == record["case_hash"]
        and physical_manifest["object_hash"] == record["object_hash"],
        "physical backbone differs from the cohort case",
    )
    _require(
        physical_manifest["inputs_sha256"]["protocol"]
        == file_sha256(protocol_path)
        and physical_manifest["inputs_sha256"]["cohort_lock"]
        == file_sha256(cohort_path),
        "physical backbone differs from the locked protocol",
    )
    return physical_manifest, physical_arrays


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    repo = args.repo.resolve()
    code_revision = _require_clean_repository(repo)
    output = args.output_dir.resolve()
    _require(not output.exists(), "provider output directory already exists")
    output.mkdir(parents=True)
    stage = "source-admission"
    try:
        _, record = _validate_locked_case_inputs(
            args,
            code_revision,
        )
        stage = "physical-backbone"
        _, physical = _validate_physical_case_inputs(
            args,
            record,
        )
        case_hash = str(record["case_hash"])
        protocol_path = args.protocol.resolve()
        cohort_path = args.cohort_lock.resolve()
        admission_path = args.admission.resolve()
        physical_dir = args.physical_dir.resolve()
        processed = args.processed_episode_dir.resolve()

        stage = "query-schedule"
        (
            camera_names,
            intrinsics,
            camera_to_world,
            image_shapes_hw,
            calibration_hashes,
        ) = load_camera_geometry(processed)
        schedule = build_dynamic_query_schedule(
            physical["physical_prediction_m"],
            physical["graph_basis"],
            intrinsics,
            camera_to_world,
            image_shapes_hw,
            camera_names,
        )
        _require(
            len(schedule.entity_ids) == EXPECTED_QUERY_COUNT,
            "query scheduler did not fill every frozen birth wave",
        )
        schedule_path = output / QUERY_SCHEDULE_FILENAME
        schedule_artifact = write_query_schedule_artifact(
            schedule_path,
            schedule,
            case_hash=case_hash,
            input_sha256={
                "physical_manifest": file_sha256(
                    physical_dir / PHYSICAL_MANIFEST_FILENAME
                ),
                "intrinsics": calibration_hashes["intrinsics"],
                "extrinsics": calibration_hashes["extrinsics"],
            },
        )

        stage = "birth-association"
        camera_inputs = load_selected_causal_inputs(
            processed,
            schedule.camera_panel.camera_indices,
        )
        associations = build_dynamic_birth_associations(
            schedule,
            physical["physical_prediction_m"],
            camera_inputs.intrinsics,
            camera_inputs.camera_to_world,
            camera_inputs.depths_m,
            camera_inputs.object_masks,
            input_camera_indices=camera_inputs.camera_indices,
        )

        stage = "tapnextpp-runtime"
        model, tapnext_utils, model_runtime = _load_model(
            args.tapnet_root.resolve(),
            args.tapnextpp_checkpoint.resolve(),
            args.device,
        )
        runtime = run_dynamic_tapnextpp_births(
            model,
            camera_inputs.rgbs,
            associations,
            schedule.birth_frames,
            schedule.update_frames,
            tapnext_utils,
        )
        torch.cuda.synchronize(model.device)
        model_runtime.update(
            {
                "peak_gpu_memory_gib": (
                    torch.cuda.max_memory_allocated(model.device) / (1024**3)
                ),
                "rollout_count": runtime.rollout_count,
                "model_frame_count": runtime.model_frame_count,
                "elapsed_seconds": runtime.elapsed_seconds,
            }
        )

        stage = "multiview-lift"
        result = fuse_dynamic_tapnextpp_multiview(
            runtime.tracks_xy,
            runtime.visibility_probability,
            camera_inputs.depths_m,
            camera_inputs.object_masks,
            camera_inputs.intrinsics,
            camera_inputs.camera_to_world,
            associations.query_points_world_m,
            association_valid=associations.valid,
            association_probability=associations.association_probability,
            association_entropy=associations.association_entropy,
            assignment_pixel_covariance_px2=(
                associations.candidate_pixel_covariance_px2
            ),
        )
        provider_archive, provider_report_path, provider_report = (
            write_provider_artifacts(
                output,
                result,
                runtime,
                associations,
                schedule,
                case_hash=case_hash,
                input_sha256={
                    "protocol": file_sha256(protocol_path),
                    "cohort_lock": file_sha256(cohort_path),
                    "admission": file_sha256(admission_path),
                    "physical_manifest": file_sha256(
                        physical_dir / PHYSICAL_MANIFEST_FILENAME
                    ),
                    "query_schedule": file_sha256(schedule_path),
                },
                runtime_provenance={
                    **model_runtime,
                    "causal_camera_inputs": camera_inputs.provenance,
                },
            )
        )

        stage = "observation-belief"
        belief = build_dynamic_tapnextpp_observation_belief(
            result,
            case_id=case_hash,
            frame_ids=np.arange(58, dtype=np.int64),
            entity_ids=schedule.entity_ids,
            entity_birth_frames=schedule.birth_frames,
            entity_update_frames=schedule.update_frames,
            camera_names=camera_inputs.camera_names,
            query_schedule_sha256=file_sha256(schedule_path),
        )
        belief_path = output / OBSERVATION_BELIEF_FILENAME
        save_observation_belief(belief_path, belief)

        stage = "discrepancy-update"
        measurements = build_birth_anchored_measurements(
            result,
            schedule,
            physical["physical_prediction_m"],
        )
        assimilation_report, assimilation_arrays = (
            predict_dynamic_tapnextpp_candidate(
                physical["physical_prediction_m"],
                physical["persistence_prediction_m"],
                measurements,
            )
        )
        (
            prediction_input,
            assimilation_archive,
            assimilation_artifact,
        ) = write_assimilation_artifacts(
            output,
            assimilation_report,
            assimilation_arrays,
            case_hash=case_hash,
            measurement_entity_ids=schedule.entity_ids,
            update_frames=np.asarray([19, 38, 57]),
            input_sha256={
                "physical_manifest": file_sha256(
                    physical_dir / PHYSICAL_MANIFEST_FILENAME
                ),
                "query_schedule": file_sha256(schedule_path),
                "provider_archive": file_sha256(provider_archive),
                "observation_belief": file_sha256(belief_path),
            },
        )

        runtime_report = {
            "schema_version": 1,
            "artifact_kind": "Deform360DynamicTAPNextPPRuntime",
            "protocol_id": PROTOCOL_ID,
            "case_hash": case_hash,
            "code_revision": code_revision,
            "environment_sha256": _environment_sha256(
                model_runtime,
                code_revision,
            ),
            "provider_result_sha256": provider_report[
                "provider_result_sha256"
            ],
            "assimilation_result_sha256": assimilation_artifact[
                "result_sha256"
            ],
            "implementation_sha256": {
                "runner": file_sha256(Path(__file__).resolve()),
                "provider": file_sha256(
                    Path(load_selected_causal_inputs.__code__.co_filename)
                ),
                "runtime": file_sha256(
                    Path(run_dynamic_tapnextpp_births.__code__.co_filename)
                ),
                "multiview": file_sha256(
                    Path(fuse_dynamic_tapnextpp_multiview.__code__.co_filename)
                ),
                "assimilation": file_sha256(
                    Path(
                        predict_dynamic_tapnextpp_candidate.__code__.co_filename
                    )
                ),
            },
            "information_boundary": {
                "maximum_rgb_depth_mask_frame_read": 57,
                "future_object_geometry_read": False,
                "target_metric_read": False,
                "held_v8_access": False,
            },
        }
        runtime_report["result_sha256"] = _canonical_sha256(runtime_report)
        runtime_report_path = output / RUNTIME_REPORT_FILENAME
        _write_json_atomic(runtime_report_path, runtime_report)

        stage = "prediction-seal"
        seal = build_prediction_seal(
            output,
            protocol_path=protocol_path,
            admission_path=admission_path,
            query_schedule_path=schedule_path,
            observation_belief_path=belief_path,
            prediction_archive_path=prediction_input,
            code_revision=code_revision,
            environment_sha256=runtime_report["environment_sha256"],
            additional_input_paths={
                "physical_manifest": (
                    physical_dir / PHYSICAL_MANIFEST_FILENAME
                ),
                "provider_archive": provider_archive,
                "provider_report": provider_report_path,
                "assimilation_archive": assimilation_archive,
                "assimilation_report": output / ASSIMILATION_REPORT_FILENAME,
                "runtime_report": runtime_report_path,
            },
        )
        return {
            "status": "prediction_sealed",
            "case_hash": case_hash,
            "prediction_seal_sha256": seal["result_sha256"],
            "query_schedule_sha256": schedule_artifact["result_sha256"],
            "provider_sha256": provider_report["result_sha256"],
            "assimilation_sha256": assimilation_artifact["result_sha256"],
            "observation_count": belief.observation_count,
        }
    except Exception as error:
        if stage != "source-admission":
            try:
                failure = record_technical_failure(
                    output,
                    protocol_path=args.protocol.resolve(),
                    admission_path=args.admission.resolve(),
                    stage=stage,
                    reason_code=type(error).__name__,
                    code_revision=code_revision,
                )
                print(
                    json.dumps(
                        {
                            "status": "technical_failure_recorded",
                            "stage": stage,
                            "failure_sha256": failure["result_sha256"],
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                )
            except Exception as disposition_error:
                print(
                    f"could not record technical failure: {disposition_error}",
                    file=sys.stderr,
                )
        raise


def main() -> int:
    result = _run(_parse_args())
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
