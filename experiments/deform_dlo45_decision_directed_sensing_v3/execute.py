"""Execution wrapper that preserves the v2 core's no-clobber output contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.deform_dlo45_decision_directed_sensing_v2 import (
    evaluate as core,
)
from experiments.deform_dlo45_decision_directed_sensing_v3 import (
    evaluate as analysis,
)


def run(args: argparse.Namespace) -> int:
    protocol_path = Path(args.protocol).resolve()
    output = Path(args.output_dir).resolve()
    core_output = output.with_name(f"{output.name}-core")
    if output.exists() or core_output.exists():
        raise ValueError("output directory already exists")

    protocol = analysis.load_protocol(protocol_path)
    repository_root = protocol_path.parents[2]
    core_protocol = (repository_root / str(protocol["core_protocol_path"])).resolve()
    if not core_protocol.is_file():
        raise ValueError(f"missing core protocol: {core_protocol}")

    core_status = core.run(
        argparse.Namespace(
            dataset_root=args.dataset_root,
            protocol=str(core_protocol),
            output_dir=str(core_output),
            source_revision=args.source_revision,
        )
    )
    if core_status != 0:
        raise RuntimeError(f"core replication returned {core_status}")

    output.mkdir(parents=True)
    core_output.rename(output / "core")
    nested_core = output / "core"
    core_result = analysis.read_json(nested_core / "result.json")
    selected = core_result["selected_calibration"]
    predecessor = protocol["predecessor"]
    if (
        float(selected["sensor_log_likelihood_scale"])
        != float(predecessor["fixed_likelihood_scale"])
        or float(selected["action_prototype_scale"])
        != float(predecessor["fixed_action_prototype_scale"])
        or float(selected["regret_tolerance"])
        != float(predecessor["fixed_support_regret_tolerance"])
    ):
        raise ValueError("core operating point differs from frozen predecessor")

    calibration_rows = analysis.read_jsonl(nested_core / "calibration_cases.jsonl")
    test_rows = analysis.read_jsonl(nested_core / "source_test_cases.jsonl")
    overlap = analysis.overlap_audit(core_result, protocol)
    transport = analysis.transport_calibration(
        calibration_rows,
        test_rows,
        protocol,
    )
    budget = int(predecessor["fixed_measurement_budget"])
    sign_tests = analysis.sign_test_summary(core_result, budget)
    acceptance = analysis.acceptance_summary(
        core_result,
        transport,
        overlap,
        protocol,
    )
    operating_point = core_result["aggregate"]["decision_regret"][str(budget)]
    classification = (
        "strong-nonoverlapping-source-replication"
        if acceptance["passed"]
        else "mixed-nonoverlapping-source-replication"
    )
    result: dict[str, Any] = {
        "contract": analysis.CONTRACT,
        "schema_version": 1,
        "status": "source-test-only-nonoverlapping-replication",
        "classification": classification,
        "protocol_sha256": analysis.sha256_file(protocol_path),
        "core_protocol_sha256": analysis.sha256_file(core_protocol),
        "source_revision": args.source_revision,
        "predecessor": predecessor,
        "core_result_id": core_result["result_id"],
        "core_classification": core_result["classification"],
        "core_selected_calibration": selected,
        "overlap_audit": overlap,
        "replication_operating_point": operating_point,
        "transport_calibration": transport,
        "sign_tests": sign_tests,
        "acceptance": acceptance,
        "accounting": {
            "calibration_trajectory_scores": len(
                transport["calibration_trajectory_scores"]
            ),
            "source_test_trajectory_scores": len(
                transport["source_test_trajectory_scores"]
            ),
            "predecessor_source_test_overlap": overlap["total_overlap_count"],
            "official_evaluation_files_opened": False,
            "new_data_collected": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = analysis.canonical_sha256(result)
    analysis.write_json(output / "result.json", result)
    analysis.write_json(
        output / "compact_result.json",
        analysis.compact_result(result),
    )
    with (output / "transport_trajectory_scores.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for role in ("calibration", "source_test"):
            for row in transport[f"{role}_trajectory_scores"]:
                stream.write(
                    json.dumps(
                        {"role": role, **row},
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
    (output / "SUMMARY.md").write_text(
        analysis.render_summary(result),
        encoding="utf-8",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
