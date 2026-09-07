#!/usr/bin/env python3
"""Run retained residual implementation tests without opening recorded datasets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "experiments/residual-models/imports-2026-09-07.json"


def local_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT) or not path.exists():
        raise ValueError(f"Invalid repository-local path: {value}")
    return path


def run_suite(study: dict, output: Path) -> dict:
    cwd = local_path(study["test_working_directory"])
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join(
            str(p) for p in (ROOT, ROOT / "src", ROOT / "scripts/remote", cwd)
        ),
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    number = study["pr"]
    xml = output / f"pr-{number}.xml"
    if "self_test_script" in study:
        script = local_path(str(cwd.relative_to(ROOT) / study["self_test_script"]))
        command = [sys.executable, str(script)]
    else:
        tests = [
            str(local_path(str(cwd.relative_to(ROOT) / p))) for p in study["test_paths"]
        ]
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            *tests,
            f"--junitxml={xml}",
        ]
    completed = subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True, timeout=180
    )
    log = completed.stdout + completed.stderr
    (output / f"pr-{number}.log").write_text(log, encoding="utf-8")
    print(f"PR #{number}: {log.strip()}", flush=True)
    if completed.returncode:
        raise RuntimeError(
            f"PR #{number} implementation checks failed ({completed.returncode})"
        )
    record = {
        "pr": number,
        "status": "passed",
        "command": command,
        "scientific_data_run": False,
    }
    if xml.exists():
        tree = ET.parse(xml)
        cases = tree.findall(".//testcase")
        if not cases or any(case.find("skipped") is not None for case in cases):
            raise RuntimeError(
                f"PR #{number}: missing tests or unexpected optional-dependency skip"
            )
        if any(
            case.find("failure") is not None or case.find("error") is not None
            for case in cases
        ):
            raise RuntimeError(f"PR #{number}: failure in test receipt")
        record["tests"] = len(cases)
        record["skipped"] = 0
    else:
        # This historical script raises if any of its six named assertions fails.
        record["self_test_script"] = study["self_test_script"]
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pr", type=int, action="append", help="Run only these catalog PRs"
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    studies = json.loads(CATALOG.read_text(encoding="utf-8"))["experiments"]
    if args.pr:
        unknown = set(args.pr) - {s["pr"] for s in studies}
        if unknown:
            parser.error(f"Unknown PRs: {sorted(unknown)}")
        studies = [s for s in studies if s["pr"] in args.pr]
    with tempfile.TemporaryDirectory(prefix="residual-checks-") as temporary:
        output = (
            Path(temporary) if args.output_dir is None else args.output_dir.resolve()
        )
        output.mkdir(parents=True, exist_ok=True)
        receipts = [run_suite(s, output) for s in studies]
        result = {
            "schema_version": 1,
            "status": "passed",
            "scientific_data_run": False,
            "suites": receipts,
            "pytest_tests": sum(r.get("tests", 0) for r in receipts),
            "skipped": 0,
        }
        (output / "summary.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({k: v for k, v in result.items() if k != "suites"}, indent=2))


if __name__ == "__main__":
    main()
