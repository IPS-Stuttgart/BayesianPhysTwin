#!/usr/bin/env python3
"""Run source-only GP diagnostics; never open official evaluation datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bayesian_phystwin_experiments.gp_discrepancy_study_v1 import (  # noqa: E402
    StudyData,
    run_study,
    synthetic_data,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_development(archive: Path, manifest_path: Path) -> StudyData:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "gp-discrepancy-development-archive-v1"
        or manifest.get("scope") != "DLO1/train"
        or manifest.get("official_evaluation_opened") is not False
        or manifest.get("archive_sha256") != sha256(archive)
    ):
        raise ValueError("input is not an intact source-only DLO1 development archive")
    with np.load(archive, allow_pickle=False) as source:
        keys = (
            "features",
            "times",
            "ids",
            "split",
            "baseline",
            "ridge",
            "truth",
            "basis",
            "frames",
        )
        if set(source.files) != set(keys):
            raise ValueError("unexpected or missing development archive arrays")
        arrays = {key: source[key] for key in keys}
    data = StudyData(**arrays, kind="retrospective-development")
    data.validate()
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/diagnostics/gp_discrepancy_development_v1.json",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--archive", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.archive is not None and args.manifest is None:
        parser.error("--archive requires --manifest")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("study") != "gp-discrepancy-development-v1"
        or protocol.get("official_evaluation_allowed") is not False
    ):
        raise ValueError("wrong protocol or forbidden official-evaluation permission")
    sources = [
        ROOT / "src/bayesian_phystwin_experiments/gp_discrepancy_v1.py",
        ROOT / "src/bayesian_phystwin_experiments/gp_discrepancy_study_v1.py",
        Path(__file__),
    ]
    runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol_sha256": sha256(args.protocol),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    try:
        runtime["git_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        runtime["git_revision"] = None
    try:
        if args.synthetic:
            datasets = [
                (
                    f"{arm}-seed{seed}",
                    synthetic_data(seed, adverse=arm == "hidden-regime-shift"),
                )
                for arm in protocol["synthetic_arms"]
                for seed in protocol["synthetic_seeds"]
            ]
        else:
            datasets = [
                (
                    "deform-dlo1-development",
                    load_development(args.archive, args.manifest),
                )
            ]
            runtime["input_archive_sha256"] = sha256(args.archive)
            runtime["input_manifest_sha256"] = sha256(args.manifest)
        results = []
        for name, data in datasets:
            result, arrays = run_study(data, protocol)
            result["name"] = name
            result["runtime"] = runtime
            prediction_path = output / f"{name}-predictions.npz"
            np.savez_compressed(prediction_path, **arrays)
            result["prediction_archive_sha256"] = sha256(prediction_path)
            (output / f"{name}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            results.append(
                {
                    "name": name,
                    "evidence_kind": result["evidence_kind"],
                    "point_summary": result["point_summary"],
                    "covariance_summary": result["covariance_summary"],
                    "prediction_sha256": result["prediction_sha256"],
                }
            )
        summary = {
            "study": protocol["study"],
            "runtime": runtime,
            "results": results,
            "real_object_gp_improvement_established": False,
            "official_evaluation_opened": False,
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except Exception as error:
        failure = {
            "exception": type(error).__name__,
            "message": str(error),
            "runtime": runtime,
            "evidence_status": "technical failure; no promotion",
            "official_evaluation_opened": False,
        }
        (output / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
