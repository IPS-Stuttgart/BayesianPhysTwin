"""Validate installed ecosystem packages against the committed compatibility lock."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.ecosystem_compatibility import (
    EcosystemCompatibilityReportV1,
    load_ecosystem_compatibility_lock,
    normalize_ecosystem_component_id,
    validate_installed_ecosystem,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bpt ecosystem validate",
        description=(
            "Validate installed BayesianPhysTwin, Prob4D, and Causal4D package "
            "lines and optional exact source revisions."
        ),
    )
    parser.add_argument(
        "--lock",
        type=Path,
        help="explicit compatibility-lock JSON; defaults to the bundled lock",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="require Prob4D and Causal4D to be installed as well as BayesianPhysTwin",
    )
    parser.add_argument(
        "--exact-versions",
        action="store_true",
        help="require exact locked package versions instead of compatible minor lines",
    )
    parser.add_argument(
        "--revision",
        action="append",
        default=[],
        metavar="COMPONENT=COMMIT",
        help=(
            "verify an exact lowercase 40-character source commit; may be repeated "
            "for bpt, prob4d, or causal4d"
        ),
    )
    parser.add_argument("--json", action="store_true", help="print JSON to stdout")
    parser.add_argument(
        "--output-json",
        type=Path,
        help="write the complete validation report atomically",
    )
    return parser


def _revision_map(values: Sequence[str]) -> dict[str, str]:
    revisions: dict[str, str] = {}
    for value in values:
        if value.count("=") != 1:
            raise ValueError("--revision must use COMPONENT=COMMIT form")
        selector, revision = value.split("=", 1)
        component_id = normalize_ecosystem_component_id(selector)
        if component_id in revisions:
            raise ValueError(f"duplicate --revision for {component_id}")
        revisions[component_id] = revision
    return revisions


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _print_human(report: EcosystemCompatibilityReportV1) -> None:
    print(f"ecosystem lock: {report.lock_name}")
    print(f"lock sha256: {report.lock_sha256}")
    for status in report.components:
        if not status.installed:
            state = (
                "missing (required)"
                if status.required
                else "not installed (optional)"
            )
        elif status.compatible:
            state = f"compatible {status.installed_version}"
        else:
            state = f"incompatible {status.installed_version}"
        if status.supplied_revision is not None:
            revision_state = "match" if status.revision_compatible else "mismatch"
            state += f", revision {revision_state}"
        print(f"{status.component_id}: {state}")
    print("compatible: " + ("yes" if report.compatible else "no"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        revisions = _revision_map(arguments.revision)
        lock = load_ecosystem_compatibility_lock(arguments.lock)
        report = validate_installed_ecosystem(
            lock,
            require_all=arguments.require_all,
            exact_versions=arguments.exact_versions,
            revisions=revisions,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    payload = report.to_dict()
    if arguments.output_json is not None:
        _write_json(arguments.output_json, payload)
    if arguments.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0 if report.compatible else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
