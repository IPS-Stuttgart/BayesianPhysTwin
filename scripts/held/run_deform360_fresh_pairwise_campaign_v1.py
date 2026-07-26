#!/usr/bin/env python3
"""Run the frozen pre-outcome fresh pairwise campaign on gpuserver4090 GPU 0."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path("/home/florianpfaff/bpt-fresh-pairwise-32f0fe1")
PROTOCOL = REPO / "configs/sota/deform360_fresh_pairwise_belief_v1.json"
COHORT = (
    REPO
    / "results/sota/deform360_fresh_source_lock_v1"
    / "deform360_fresh_object_cohort_lock_v1.json"
)
ADMISSIONS = REPO / "results/sota/deform360_fresh_source_lock_v1/admissions"
PROCESSED = Path("/home/florianpfaff/deform360-fresh-source-processed-v1-1a3f9b1")
ROOT = Path("/home/florianpfaff/deform360-fresh-pairwise-eval-32f0fe1")
SOURCE_REPO = Path("/home/florianpfaff/bpt-open27-runtime-4e2cdbd")
OFFICIAL = Path("/home/florianpfaff/PhysTwin-upstream")
DEFORM360 = Path("/home/florianpfaff/deform360-code")
PYTHON = Path("/home/florianpfaff/.venvs/deform360-processing-v1/bin/python")
ALLTRACKER = Path("/home/florianpfaff/alltracker-molmomotion-61f5b21")
CHECKPOINT = Path("/home/florianpfaff/model-cache/alltracker.pth")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(stage: str, case: str, command: list[str]) -> dict[str, object]:
    log = ROOT / "campaign_logs" / f"{case}.{stage}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=REPO,
            env={
                **os.environ,
                "PYTHONPATH": str(REPO / "src"),
                "CUDA_VISIBLE_DEVICES": "0",
            },
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return {
        "stage": stage,
        "returncode": completed.returncode,
        "runtime_seconds": time.perf_counter() - started,
        "log": str(log),
        "log_sha256": sha256(log),
    }


def validate_completed_case(case: str) -> None:
    sys.path.insert(0, str(REPO / "src"))
    from bayesian_phystwin.deform360_fresh_pairwise_protocol import (
        file_sha256,
        load_bound_cohort,
        load_fresh_pairwise_protocol,
        load_json,
        validate_backbone_seal,
        validate_belief_prediction_seal,
    )

    protocol = load_fresh_pairwise_protocol(PROTOCOL, repository_root=REPO)
    cohort = load_bound_cohort(COHORT, protocol)
    backbone = load_json(ROOT / "backbones" / case / "prediction_seal.json")
    validate_backbone_seal(
        backbone,
        protocol_config_sha256=file_sha256(PROTOCOL),
        cohort_lock_sha256=cohort["cohort_lock_sha256"],
    )
    belief = load_json(
        ROOT / "predictions" / case / "belief_prediction_seal.json"
    )
    validate_belief_prediction_seal(
        belief,
        protocol_config_sha256=file_sha256(PROTOCOL),
        cohort_lock_sha256=cohort["cohort_lock_sha256"],
    )


def main() -> int:
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    failed = False
    for item in cohort["cases"]:
        case = str(item["case"])
        object_id = str(item["object_id"])
        episode_id = int(item["episode_id"])
        completed_seal = (
            ROOT / "predictions" / case / "belief_prediction_seal.json"
        )
        if completed_seal.is_file():
            validate_completed_case(case)
            records.append({"case": case, "status": "preexisting_valid"})
            print(f"{case}: preexisting valid", flush=True)
            continue
        admission = ADMISSIONS / f"{case}.admission.json"
        processed = PROCESSED / object_id / f"episode_{episode_id:04d}"
        stages: list[dict[str, object]] = []
        physical = [
            str(PYTHON),
            str(REPO / "scripts/remote/run_deform360_fresh_pairwise_physical.py"),
            "--repo",
            str(REPO),
            "--protocol",
            str(PROTOCOL),
            "--cohort-lock",
            str(COHORT),
            "--admission",
            str(admission),
            "--processed-episode-dir",
            str(processed),
            "--output-dir",
            str(ROOT / "backbones" / case),
            "--source-repo",
            str(SOURCE_REPO),
            "--official-phystwin-repo",
            str(OFFICIAL),
            "--official-config",
            str(OFFICIAL / "configs/real.yaml"),
            "--deform360-repo",
            str(DEFORM360),
            "--python",
            str(PYTHON),
            "--device",
            "cuda:0",
        ]
        measurement = [
            str(PYTHON),
            str(REPO / "scripts/remote/run_deform360_fresh_pairwise_measurement.py"),
            "--repo",
            str(REPO),
            "--protocol",
            str(PROTOCOL),
            "--cohort-lock",
            str(COHORT),
            "--backbone-case-dir",
            str(ROOT / "backbones" / case),
            "--processed-episode-dir",
            str(processed),
            "--output-dir",
            str(ROOT / "measurements" / case),
            "--alltracker-source",
            str(ALLTRACKER),
            "--alltracker-checkpoint",
            str(CHECKPOINT),
            "--device",
            "cuda:0",
        ]
        prediction = [
            str(PYTHON),
            str(REPO / "scripts/remote/run_deform360_fresh_pairwise_prediction.py"),
            "--repo",
            str(REPO),
            "--protocol",
            str(PROTOCOL),
            "--cohort-lock",
            str(COHORT),
            "--backbone-case-dir",
            str(ROOT / "backbones" / case),
            "--measurement-dir",
            str(ROOT / "measurements" / case),
            "--output-dir",
            str(ROOT / "predictions" / case),
        ]
        for stage, command in (
            ("physical", physical),
            ("measurement", measurement),
            ("prediction", prediction),
        ):
            result = run(stage, case, command)
            stages.append(result)
            print(
                f"{case}: {stage} rc={result['returncode']} "
                f"t={result['runtime_seconds']:.1f}s",
                flush=True,
            )
            if result["returncode"] != 0:
                failed = True
                break
        records.append(
            {
                "case": case,
                "status": "technical_failure" if failed else "predicted",
                "stages": stages,
            }
        )
        if failed:
            break
    summary = {
        "schema_version": 1,
        "artifact_kind": "Deform360FreshPairwisePreOutcomeCampaign",
        "repository_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "protocol_file_sha256": sha256(PROTOCOL),
        "cohort_lock_file_sha256": sha256(COHORT),
        "operator_script_sha256": sha256(Path(__file__)),
        "records": records,
        "passed": not failed and len(records) == len(cohort["cases"]),
        "information_boundary": {
            "future_target_read": False,
            "outcome_manifest_read": False,
            "outcome_path_argument_available": False,
        },
    }
    summary_path = ROOT / "preoutcome_campaign_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": summary["passed"], "records": len(records)}))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
