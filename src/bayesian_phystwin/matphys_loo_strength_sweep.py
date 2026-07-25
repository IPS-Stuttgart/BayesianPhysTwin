"""Sealed PhysTwin replay and family selection for LOO MatPhys proposals."""

from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .matphys_causal_bridge import sha256_file
from .matphys_loo_spring_fields import validate_loo_spring_fields
from .phystwin_backbone_family_gate import run_backbone_family_gate
from .phystwin_external_backbone import (
    EXTERNAL_COORDINATE_FRAME,
    EXTERNAL_VERTEX_CONTRACT,
    run_external_backbone_overlay,
    validate_external_backbone_manifest,
)


LOO_STRENGTH_SWEEP_CONTRACT = "matphys-object-disjoint-loo-strength-sweep-v1"


def strength_family_name(strength: float) -> str:
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("proposal strength must lie in [0, 1]")
    return f"alpha_{int(round(strength * 1000)):04d}"


def _identity(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _resolve_identity(manifest: Path, value: object, label: str) -> Path:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a file identity")
    path = Path(str(value.get("path", "")))
    if not path.is_absolute():
        path = manifest.parent / path
    path = path.resolve()
    if not path.is_file() or sha256_file(path) != str(value.get("sha256", "")):
        raise ValueError(f"{label} bytes changed")
    return path


def _run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(f"$ {shlex.join(command)}\n")
        stream.flush()
        completed = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
            text=True,
        )
    if completed.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        raise RuntimeError(
            f"command failed with exit {completed.returncode}: {shlex.join(command)}\n"
            + "\n".join(tail)
        )


def _refit_command(
    python: Path,
    official_repo: Path,
    case_root: Path,
    cues: Path,
    checkpoint: Path,
    output: Path,
    *,
    train_end: int,
    fit_end: int,
    optimal_params: Path | None = None,
    released_trajectory: Path | None = None,
) -> list[str]:
    """Build a future-sealed epochs-zero replay command."""

    optimal = (
        case_root / "optimal_params.pkl" if optimal_params is None else optimal_params
    )
    released = (
        case_root / "inference.pkl"
        if released_trajectory is None
        else released_trajectory
    )

    return [
        str(python),
        "-m",
        "bayesian_phystwin.cli.phystwin_refit",
        str(official_repo),
        str(case_root / "final_data.pkl"),
        str(optimal),
        str(checkpoint),
        str(cues),
        str(output),
        "--variant",
        "mixture",
        "--train-end-frame",
        str(train_end),
        "--fit-end-frame",
        str(fit_end),
        "--epochs",
        "0",
        "--learning-rate",
        "0.0001",
        "--observation-variance",
        "2.5e-5",
        "--model-discrepancy-variance",
        "0",
        "--outlier-variance-multiplier",
        "100",
        "--flow-scale",
        "0.005",
        "--boundary-scale",
        "0.03",
        "--dt",
        "5e-5",
        "--num-substeps",
        "667",
        "--track-weight",
        "1",
        "--acceleration-weight",
        "0.01",
        "--freeze-collision",
        "--spring-parameterization",
        "dense",
        "--device",
        "cuda:0",
        "--released-trajectory",
        str(released),
        "--selection-only",
    ]


def _validate_replay_cache(
    replay_root: Path,
    overlay_summary: Path,
    *,
    source_checkpoint: Path,
    candidate_field: Path,
    strength: float,
) -> bool:
    replay_summary = replay_root / "summary.json"
    if not overlay_summary.is_file() or not replay_summary.is_file():
        return False
    overlay = json.loads(overlay_summary.read_text(encoding="utf-8"))
    replay = json.loads(replay_summary.read_text(encoding="utf-8"))
    checks = (
        overlay.get("proposal_strength") == strength,
        overlay.get("source_checkpoint", {}).get("sha256")
        == sha256_file(source_checkpoint),
        overlay.get("candidate_spring_y", {}).get("sha256")
        == sha256_file(candidate_field),
        replay.get("future_metrics_opened") is False,
        replay.get("config", {}).get("evaluate_future") is False,
        replay.get("inputs", {}).get("checkpoint", {}).get("sha256")
        == overlay.get("output_checkpoint", {}).get("sha256"),
    )
    trajectory = Path(str(replay.get("outputs", {}).get("trajectory", "")))
    return all(checks) and trajectory.is_file()


def _resolve_released_checkpoint(
    case_data: Path,
    official_repo: Path,
    case: str,
) -> Path:
    """Resolve one authoritative released PhysTwin checkpoint."""

    extracted = case_data / "checkpoint.pth"
    if extracted.is_file():
        return extracted
    upstream = sorted(
        (official_repo / "experiments" / case / "train").glob("best_*.pth")
    )
    if len(upstream) != 1:
        raise FileNotFoundError(
            f"{case}: expected checkpoint.pth or exactly one upstream best_*.pth; "
            f"found {len(upstream)}"
        )
    return upstream[0]


def _resolve_released_artifact(
    case_data: Path,
    official_repo: Path,
    case: str,
    filename: str,
) -> Path:
    """Resolve a released runtime artifact from compact or upstream layouts."""

    extracted = case_data / filename
    if extracted.is_file():
        return extracted
    upstream_paths = {
        "inference.pkl": official_repo / "experiments" / case / "inference.pkl",
        "optimal_params.pkl": (
            official_repo / "experiments_optimization" / case / "optimal_params.pkl"
        ),
    }
    try:
        upstream = upstream_paths[filename]
    except KeyError as error:
        raise ValueError(f"unsupported released artifact: {filename}") from error
    if not upstream.is_file():
        raise FileNotFoundError(f"{case}: missing released {filename}")
    return upstream


def build_strength_external_manifest(
    spring_fields: dict[str, object],
    replay_root: str | Path,
    output_path: str | Path,
    *,
    strength: float,
) -> dict[str, object]:
    """Bind one strength arm to complete replay trajectories without scoring it."""

    family = strength_family_name(strength)
    root = Path(replay_root).resolve()
    destination = Path(output_path).resolve()
    cases = []
    for entry in spring_fields["cases"]:
        case = str(entry["name"])
        trajectory = root / family / "cases" / case / "replay" / "trajectory.pkl"
        replay_summary = root / family / "cases" / case / "replay" / "summary.json"
        overlay_summary = root / family / "cases" / case / "overlay.json"
        if not trajectory.is_file() or not replay_summary.is_file():
            raise FileNotFoundError(f"missing replay for {family}/{case}")
        replay = json.loads(replay_summary.read_text(encoding="utf-8"))
        if replay.get("future_metrics_opened") is not False:
            raise ValueError(f"{family}/{case}: replay opened future metrics")
        cases.append(
            {
                "name": case,
                "trajectory": str(trajectory),
                "sha256": sha256_file(trajectory),
                "evidence_end_frame_exclusive": int(
                    entry["evidence_end_frame_exclusive"]
                ),
                "initial_alignment_tolerance_m": 1e-6,
                "proposal_strength": strength,
                "spring_overlay": _identity(overlay_summary),
                "replay_summary": _identity(replay_summary),
            }
        )
    source_manifest = Path(str(spring_fields["manifest"]["path"]))
    source_backbone = spring_fields["backbone"]
    payload = {
        "schema_version": 1,
        "backbone": {
            "name": f"LOO MatPhys spring proposal replay ({family})",
            "source_repository": source_backbone["source_repository"],
            "source_commit": source_backbone["source_commit"],
            "future_observations_used": False,
            "coordinate_frame": EXTERNAL_COORDINATE_FRAME,
            "vertex_contract": EXTERNAL_VERTEX_CONTRACT,
            "proxy_contract": source_backbone.get("proxy_contract"),
            "claim_boundary": (
                "Object-disjoint source training; full action-conditioned Warp rollout "
                "generated without scoring future observations."
            ),
            "proposal_strength": strength,
            "spring_interpolation": "log-space geodesic from released to MatPhys field",
            "loo_spring_fields": _identity(source_manifest),
        },
        "cases": cases,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**payload, "manifest_path": str(destination)}


def _write_stability_control(
    family: str,
    case_order: list[str],
    data_root: Path,
    official_repo: Path,
    destination: Path,
) -> Path:
    payload = {
        "schema_version": 1,
        "contract": "phystwin-family-stability-control-v1",
        "family": family,
        "future_observations_used": False,
        "cases": [
            {
                "name": case,
                "trajectory": _identity(
                    _resolve_released_artifact(
                        data_root / case,
                        official_repo,
                        case,
                        "inference.pkl",
                    )
                ),
            }
            for case in case_order
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def run_loo_strength_sweep(
    spring_field_manifest: str | Path,
    output_dir: str | Path,
    *,
    python: str | Path,
    official_repo: str | Path,
    data_root: str | Path,
    cues_root: str | Path,
    gpu_ids: tuple[str, ...] = ("0", "1"),
    overlay_workers: int = 2,
    resume: bool = False,
) -> dict[str, object]:
    """Replay all arms, fit prefix overlays, and seal family choices."""

    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("GPU ids must be a nonempty unique tuple")
    if overlay_workers < 1:
        raise ValueError("overlay workers must be positive")
    fields = validate_loo_spring_fields(spring_field_manifest)
    fields_manifest = Path(str(fields["manifest"]["path"]))
    protocol_path = _resolve_identity(fields_manifest, fields["protocol"], "protocol")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    strengths = [
        float(protocol["proposal_families"]["identity_fallback"]),
        *[float(value) for value in protocol["proposal_families"]["strengths"]],
    ]
    if strengths[0] != 0.0 or len(strengths) != len(set(strengths)):
        raise ValueError("protocol must define one exact identity fallback")
    case_order = [str(case) for case in protocol["case_order"]]
    if fields["case_order"] != case_order:
        raise ValueError("spring-field cohort differs from the protocol")

    output = Path(output_dir).resolve()
    replay_root = output / "replays"
    runtime_python = Path(python).expanduser().absolute()
    official = Path(official_repo).resolve()
    data = Path(data_root).resolve()
    cues = Path(cues_root).resolve()
    for path in (runtime_python, official, data, cues):
        if not path.exists():
            raise FileNotFoundError(path)

    task_queue: queue.Queue[str] = queue.Queue()
    for gpu_id in gpu_ids:
        task_queue.put(gpu_id)

    tasks = [(strength, entry) for strength in strengths for entry in fields["cases"]]

    def replay_task(task: tuple[float, dict[str, Any]]) -> dict[str, object]:
        strength, entry = task
        case = str(entry["name"])
        family = strength_family_name(strength)
        case_output = replay_root / family / "cases" / case
        overlay_checkpoint = case_output / "overlay_checkpoint.pt"
        overlay_summary = case_output / "overlay.json"
        replay_output = case_output / "replay"
        case_data = data / case
        candidate = Path(str(entry["candidate_spring_y_path"]))
        source_checkpoint = _resolve_released_checkpoint(case_data, official, case)
        optimal_params = _resolve_released_artifact(
            case_data,
            official,
            case,
            "optimal_params.pkl",
        )
        released_trajectory = _resolve_released_artifact(
            case_data,
            official,
            case,
            "inference.pkl",
        )
        split = json.loads((case_data / "split.json").read_text(encoding="utf-8"))
        train_end = int(split["train"][1])
        fit_end = int(entry["evidence_end_frame_exclusive"])
        if not 1 < fit_end < train_end:
            raise ValueError(f"{case}: invalid fit/validation boundary")
        if resume and _validate_replay_cache(
            replay_output,
            overlay_summary,
            source_checkpoint=source_checkpoint,
            candidate_field=candidate,
            strength=strength,
        ):
            return {"case": case, "family": family, "cached": True}

        gpu_id = task_queue.get()
        try:
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["PYTHONUNBUFFERED"] = "1"
            _run_logged(
                [
                    str(runtime_python),
                    "-m",
                    "bayesian_phystwin.cli.phystwin_spring_overlay",
                    str(source_checkpoint),
                    str(candidate),
                    str(overlay_checkpoint),
                    "--summary",
                    str(overlay_summary),
                    "--strength",
                    str(strength),
                ],
                case_output / "overlay.log",
                env,
            )
            _run_logged(
                _refit_command(
                    runtime_python,
                    official,
                    case_data,
                    cues / case / "cues.npz",
                    overlay_checkpoint,
                    replay_output,
                    train_end=train_end,
                    fit_end=fit_end,
                    optimal_params=optimal_params,
                    released_trajectory=released_trajectory,
                ),
                case_output / "replay.log",
                env,
            )
        finally:
            task_queue.put(gpu_id)
        if not _validate_replay_cache(
            replay_output,
            overlay_summary,
            source_checkpoint=source_checkpoint,
            candidate_field=candidate,
            strength=strength,
        ):
            raise RuntimeError(f"{family}/{case}: replay did not seal correctly")
        return {"case": case, "family": family, "cached": False}

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        replay_records = list(executor.map(replay_task, tasks))

    families: OrderedDict[str, Path] = OrderedDict()
    stability_controls: OrderedDict[str, Path] = OrderedDict()
    overlay_summaries: OrderedDict[str, Path] = OrderedDict()
    for strength in strengths:
        family = strength_family_name(strength)
        manifest = output / "families" / family / "external_backbone_manifest.json"
        build_strength_external_manifest(
            fields,
            replay_root,
            manifest,
            strength=strength,
        )
        validate_external_backbone_manifest(data, manifest)
        families[family] = manifest
        if strength != 0.0:
            stability_controls[family] = _write_stability_control(
                family,
                case_order,
                data,
                official,
                output / "stability_controls" / f"{family}.json",
            )

        overlay_root = output / "prefix_overlays" / family
        overlay_summary = overlay_root / "external_backbone_selection_summary.json"
        if resume and overlay_summary.is_file():
            cached = json.loads(overlay_summary.read_text(encoding="utf-8"))
            if cached.get("future_metrics_opened") is not False:
                raise ValueError(f"{family}: cached overlay opened future metrics")
            expected_hash = sha256_file(manifest)
            actual_hash = cached.get("manifest", {}).get("sha256")
            if actual_hash != expected_hash:
                raise ValueError(f"{family}: cached overlay manifest changed")
        else:
            result = run_external_backbone_overlay(
                data,
                overlay_root,
                manifest,
                workers=overlay_workers,
                evaluate_future=False,
            )
            overlay_summary = Path(str(result["summary_path"]))
        overlay_summaries[family] = overlay_summary

    selection_root = output / "family_selection"
    selection_path = selection_root / "backbone_family_selection.json"
    if resume and selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("future_metrics_opened") is not False:
            raise ValueError("cached family selection opened future metrics")
    else:
        selection = run_backbone_family_gate(
            data,
            selection_root,
            overlay_summaries,
            minimum_relative_improvement=float(
                protocol["selection"]["minimum_relative_score_improvement"]
            ),
            maximum_metric_regression=float(
                protocol["selection"]["maximum_per_metric_regression"]
            ),
            stability_control_manifests=stability_controls,
            maximum_stability_rmse_m=float(
                protocol["selection"]["maximum_identity_replay_coordinate_rmse_m"]
            ),
            evaluate_future=False,
        )
        selection_path = Path(str(selection["summary_path"]))

    summary = {
        "schema_version": 1,
        "contract": LOO_STRENGTH_SWEEP_CONTRACT,
        "future_metrics_opened": False,
        "spring_fields": _identity(fields_manifest),
        "protocol": _identity(protocol_path),
        "strengths": strengths,
        "replays": replay_records,
        "family_manifests": {
            family: _identity(path) for family, path in families.items()
        },
        "prefix_overlay_summaries": {
            family: _identity(path) for family, path in overlay_summaries.items()
        },
        "selection": _identity(selection_path),
    }
    destination = output / "loo_strength_sweep.json"
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**summary, "summary_path": str(destination)}
