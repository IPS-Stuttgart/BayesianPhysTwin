#!/usr/bin/env python3
"""Reproduce and evidence-bind the frozen Bayesian-PhysTwin full-22 result."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

CAPSULE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = CAPSULE_DIR.parents[1]
EXPECTED_PATH = CAPSULE_DIR / "expected_metrics.json"
EXPECTED_SOURCE_REVISION = "e393bb6ff61d44815afd8d09dfc5334cb55d5524"
EXPECTED_PROTOCOL_ID = (
    "ee11310a84b92ff2158018a13ef09989e641e7c0ea84733fe8a6abf267093c65"
)
EXPECTED_DATA_MANIFEST_SHA256 = (
    "c986f9fffe99e63f842bb48eb1d394a6b87663f5c4a4fb99f2a58855875fb125"
)
OFFICIAL_PHYSTWIN_REVISION = "2b6630528141b9cba5a7677c8b88b2129b4a8390"
PAPER_EVIDENCE_REVISION = "71729c6cab784d7471995457269fbec0a2b9ea33"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    subprocess.run(
        list(command),
        cwd=None if cwd is None else str(cwd),
        env=None if env is None else dict(env),
        check=True,
    )


def _capture(command: Sequence[str], *, cwd: Path) -> str:
    result = subprocess.run(
        list(command),
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_environment(source_checkout: Path) -> dict[str, str]:
    environment = dict(os.environ)
    source_path = str(source_checkout / "src")
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path if not previous else os.pathsep.join((source_path, previous))
    )
    return environment


def validate_source_checkout(source_checkout: Path) -> None:
    revision = _capture(("git", "rev-parse", "HEAD"), cwd=source_checkout)
    if revision != EXPECTED_SOURCE_REVISION:
        raise ValueError(
            "full-22 reproduction requires Bayesian-PhysTwin revision "
            f"{EXPECTED_SOURCE_REVISION}; received {revision}"
        )
    dirty = _capture(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=source_checkout,
    )
    if dirty:
        raise ValueError("the frozen source checkout must be clean")


def validate_data_root(data_root: Path) -> Path:
    manifest = data_root / "evaluation_subset_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    actual_digest = _sha256(manifest)
    if actual_digest != EXPECTED_DATA_MANIFEST_SHA256:
        raise ValueError(
            "evaluation subset manifest digest changed: "
            f"expected {EXPECTED_DATA_MANIFEST_SHA256}, received {actual_digest}"
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    selected = payload.get("selected_cases")
    available = payload.get("available_cases")
    if not isinstance(selected, list) or len(selected) != 22:
        raise ValueError("the frozen data manifest must select exactly 22 cases")
    if available != selected:
        raise ValueError(
            "available_cases must equal the ordered selected 22-case cohort"
        )
    if len(set(map(str, selected))) != 22:
        raise ValueError("the frozen data manifest contains duplicate cases")
    return manifest


def _cohort_metric(
    comparison: Mapping[str, Any],
    method: str,
    aggregation: str,
    metric: str,
) -> float:
    methods = comparison.get("methods")
    if not isinstance(methods, Mapping) or method not in methods:
        raise ValueError(f"comparison is missing method {method!r}")
    method_record = methods[method]
    if not isinstance(method_record, Mapping):
        raise ValueError(f"comparison method {method!r} is malformed")
    cohorts = method_record.get("cohorts")
    if not isinstance(cohorts, Mapping):
        raise ValueError(f"comparison method {method!r} has no cohorts")
    cohort = cohorts.get("all_22_table_compatible")
    if not isinstance(cohort, Mapping):
        raise ValueError("comparison has no official all-22 cohort")
    metric_record = cohort.get(metric)
    if not isinstance(metric_record, Mapping):
        raise ValueError(f"comparison has no metric {metric!r}")
    field = {
        "equal_case": "equal_case_mean_m",
        "frame_weighted": "frame_weighted_mean_m",
    }.get(aggregation)
    if field is None:
        raise ValueError(f"unknown aggregation {aggregation!r}")
    value = metric_record.get(field)
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"comparison metric {method}/{aggregation}/{metric} is not numeric"
        )
    return float(value)


def verify_comparison(
    comparison: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    tolerance = float(expected["absolute_tolerance_m"])
    expected_methods = expected["methods"]
    if not isinstance(expected_methods, Mapping):
        raise ValueError("expected metrics file has no method mapping")
    checks: list[dict[str, Any]] = []
    for method, method_record in expected_methods.items():
        if not isinstance(method_record, Mapping):
            raise ValueError(f"expected method {method!r} is malformed")
        for aggregation, metric_record in method_record.items():
            if not isinstance(metric_record, Mapping):
                raise ValueError(
                    f"expected aggregation {method}/{aggregation} is malformed"
                )
            for metric, expected_value in metric_record.items():
                actual = _cohort_metric(
                    comparison, str(method), str(aggregation), str(metric)
                )
                delta = actual - float(expected_value)
                checks.append(
                    {
                        "method": str(method),
                        "aggregation": str(aggregation),
                        "metric": str(metric),
                        "expected_m": float(expected_value),
                        "actual_m": actual,
                        "absolute_delta_m": abs(delta),
                        "passed": abs(delta) <= tolerance,
                    }
                )
    failed = [record for record in checks if not record["passed"]]
    if failed:
        first = failed[0]
        raise ValueError(
            "full-22 metric verification failed for "
            f"{first['method']}/{first['aggregation']}/{first['metric']}: "
            f"expected {first['expected_m']}, received {first['actual_m']}"
        )
    return {
        "schema_version": 1,
        "status": "verified",
        "absolute_tolerance_m": tolerance,
        "check_count": len(checks),
        "checks": checks,
    }


def verify_confirmation_summary(summary: Mapping[str, Any]) -> None:
    protocol_id = summary.get("protocol_id")
    if protocol_id != EXPECTED_PROTOCOL_ID:
        raise ValueError(
            "Bayesian anchor protocol ID changed: "
            f"expected {EXPECTED_PROTOCOL_ID}, received {protocol_id}"
        )
    case_results = summary.get("case_results")
    if not isinstance(case_results, Mapping) or len(case_results) != 22:
        raise ValueError("Bayesian anchor confirmation must contain all 22 cases")


def _shell_command(command: Sequence[str], *, pythonpath: Path) -> str:
    prefix = f"PYTHONPATH={shlex.quote(str(pythonpath))}"
    return prefix + " " + shlex.join(list(command))


def _runtime_payload(source_checkout: Path, workers: int) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in ("bayesian-phystwin", "numpy", "scipy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "capsule_schema_version": 1,
        "orchestrator_python": platform.python_version(),
        "platform": platform.platform(),
        "source_checkout": str(source_checkout),
        "workers": workers,
        "orchestrator_packages": packages,
    }


def _manifest_command(
    output: Path,
    source_checkout: Path,
    command_line: str,
) -> list[str]:
    configuration = output / "configuration.lock.json"
    boundary = output / "information_boundary.json"
    repositories = output / "repositories.lock.json"
    runtime = output / "runtime.json"
    return [
        sys.executable,
        "-m",
        "bayesian_phystwin.cli.run_manifest",
        "create",
        str(output / "run_manifest.json"),
        "--run-id",
        "phystwin-full22-anchor-reproduction-v1",
        "--repository-root",
        str(source_checkout),
        "--related-repositories-json",
        str(repositories),
        "--classification",
        "confirmatory",
        "--statistical-unit",
        "interaction with equal-case aggregation",
        "--command-line",
        command_line,
        "--configuration-json",
        str(configuration),
        "--information-boundary-json",
        str(boundary),
        "--runtime-json",
        str(runtime),
        "--environment-variable",
        "CUDA_VISIBLE_DEVICES",
        "--claim-id",
        "bpt.full22_anchor_released_contract",
        "--method-freeze-id",
        "full22-anchor-method-v1",
        "--protocol-id",
        EXPECTED_PROTOCOL_ID,
        "--split-id",
        "development3-confirmation19-v1",
        "--baseline-id",
        "released-phystwin-reproduction-v1",
        "--seed",
        "20260710",
        "--artifact-root",
        str(output),
        "--input",
        "capsule=reproduction_capsule/reproduce.py",
        "--input",
        "expected_metrics=reproduction_capsule/expected_metrics.json",
        "--input",
        "data_manifest=input/evaluation_subset_manifest.json",
        "--input",
        "method_lock=method_lock.json",
        "--output-artifact",
        "confirmation_summary=run/bayesian_anchor_confirmation_summary.json",
        "--output-artifact",
        "full22_comparison=full22_comparison.json",
        "--output-artifact",
        "verification=verification.json",
        "--output-artifact",
        "source_command=source-command.txt",
        "--package",
        "scipy",
        "--notes",
        (
            "Portable orchestration record for the frozen full-22 Bayesian anchor; "
            "the original inverse-physics optimization is outside this claim."
        ),
    ]


def reproduce(args: argparse.Namespace) -> None:
    source_checkout = args.source_checkout.resolve()
    data_root = args.data_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(
            f"output directory is not empty: {output}; pass --force to replace it"
        )
    if output.exists() and args.force:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    validate_source_checkout(source_checkout)
    data_manifest = validate_data_root(data_root)
    expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    if expected.get("source_revision") != EXPECTED_SOURCE_REVISION:
        raise ValueError("expected metrics source revision changed")
    if expected.get("source_protocol_id") != EXPECTED_PROTOCOL_ID:
        raise ValueError("expected metrics protocol ID changed")
    if expected.get("data_manifest_sha256") != EXPECTED_DATA_MANIFEST_SHA256:
        raise ValueError("expected metrics data-manifest digest changed")

    capsule_copy = output / "reproduction_capsule"
    capsule_copy.mkdir(parents=True)
    shutil.copy2(Path(__file__).resolve(), capsule_copy / "reproduce.py")
    shutil.copy2(EXPECTED_PATH, capsule_copy / "expected_metrics.json")
    input_dir = output / "input"
    input_dir.mkdir(parents=True)
    shutil.copy2(data_manifest, input_dir / data_manifest.name)

    run_dir = output / "run"
    comparison_path = output / "full22_comparison.json"
    source_env = _source_environment(source_checkout)
    confirm_command = [
        sys.executable,
        "-m",
        "bayesian_phystwin.cli.phystwin_bayesian_confirmation",
        str(data_root),
        str(run_dir),
        "--workers",
        str(args.workers),
    ]
    compare_command = [
        sys.executable,
        "-m",
        "bayesian_phystwin.cli.phystwin_sota_comparison",
        str(data_root),
        str(comparison_path),
        "--method",
        f"released_phystwin={data_root / '{case}' / 'inference.pkl'}",
        "--method",
        f"bayesian_anchor={run_dir / 'cases' / '{case}' / 'trajectory.pkl'}",
    ]
    source_pythonpath = source_checkout / "src"
    source_command = " && ".join(
        (
            _shell_command(confirm_command, pythonpath=source_pythonpath),
            _shell_command(compare_command, pythonpath=source_pythonpath),
        )
    )
    (output / "source-command.txt").write_text(source_command + "\n", encoding="utf-8")

    _run(confirm_command, cwd=source_checkout, env=source_env)
    summary_path = run_dir / "bayesian_anchor_confirmation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    verify_confirmation_summary(summary)
    _run(compare_command, cwd=source_checkout, env=source_env)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    verification = verify_comparison(comparison, expected)
    _json_write(output / "verification.json", verification)

    _json_write(
        output / "method_lock.json",
        {
            "schema_version": 1,
            "method": "robust Bayesian random-walk endpoint anchoring",
            "source_revision": EXPECTED_SOURCE_REVISION,
            "protocol_id": EXPECTED_PROTOCOL_ID,
            "maximum_residual_m": 0.01,
            "fit_fraction": 0.75,
            "development_case_count": 3,
            "confirmation_case_count": 19,
        },
    )
    _json_write(
        output / "configuration.lock.json",
        {
            "schema_version": 1,
            "workers": args.workers,
            "source_checkout": str(source_checkout),
            "data_root": str(data_root),
            "data_manifest_sha256": _sha256(data_manifest),
            "metric_tolerance_m": expected["absolute_tolerance_m"],
        },
    )
    _json_write(
        output / "information_boundary.json",
        {
            "schema_version": 1,
            "official_ordered_case_count": 22,
            "development_case_count": 3,
            "confirmation_case_count": 19,
            "future_frames_used_for_method_selection": 0,
            "starting_point": "released inference.pkl trajectories",
            "excluded_operation": "original PhysTwin inverse-physics optimization",
            "permitted_claim": (
                "better than re-evaluated released PhysTwin under the recorded "
                "official 22-case contract"
            ),
            "prohibited_claims": [
                "overall state of the art",
                "calibrated raw posterior covariance",
                "dynamically admissible simulator-state correction",
            ],
        },
    )
    _json_write(
        output / "repositories.lock.json",
        [
            {
                "repository": "Jianghanxiao/PhysTwin",
                "revision": OFFICIAL_PHYSTWIN_REVISION,
                "dirty": False,
                "role": "upstream",
            },
            {
                "repository": "FlorianPfaff/BayesianPhysTwin-Paper",
                "revision": PAPER_EVIDENCE_REVISION,
                "dirty": False,
                "role": "paper",
            },
        ],
    )
    _json_write(
        output / "runtime.json",
        _runtime_payload(source_checkout, args.workers),
    )

    current_env = dict(os.environ)
    current_source = REPOSITORY_ROOT / "src"
    previous = current_env.get("PYTHONPATH")
    current_env["PYTHONPATH"] = (
        str(current_source)
        if not previous
        else os.pathsep.join((str(current_source), previous))
    )
    manifest_command = _manifest_command(output, source_checkout, source_command)
    _run(manifest_command, cwd=REPOSITORY_ROOT, env=current_env)
    _run(
        [
            sys.executable,
            "-m",
            "bayesian_phystwin.cli.run_manifest",
            "validate",
            str(output / "run_manifest.json"),
            "--artifact-root",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        env=current_env,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_checkout",
        type=Path,
        help=f"clean Bayesian-PhysTwin checkout at {EXPECTED_SOURCE_REVISION}",
    )
    parser.add_argument("data_root", type=Path, help="frozen official 22-case subset")
    parser.add_argument("output_dir", type=Path, help="new evidence bundle directory")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    reproduce(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
