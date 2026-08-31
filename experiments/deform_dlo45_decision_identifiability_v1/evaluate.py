"""Source-frozen decision-identifiability evaluation on DEFORM DLO4/DLO5."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from ._common import (
    CONTRACT,
    DLOS,
    FRAME_COUNT,
    NODE_COUNT,
    Model,
    Protocol,
    canonical_sha256,
    extract_observation,
    file_manifest,
    load_protocol,
    partition_names,
    read_json,
    sha256_file,
    trajectory_paths,
    validate_request,
    write_json,
)
from ._evaluation import (
    bootstrap_interval,
    choose_model_for_dlo,
    evaluate_paths,
    load_models,
    save_models,
)
from ._model import decide, deterministic_kmeans, fit_model

__all__ = [
    "FRAME_COUNT",
    "NODE_COUNT",
    "Model",
    "Protocol",
    "decide",
    "deterministic_kmeans",
    "extract_observation",
    "fit_model",
    "main",
    "partition_names",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("source", "target"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--source-seal", type=Path)
    parser.add_argument("--request", type=Path)
    return parser.parse_args()

def source_command(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.protocol)
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    models: dict[str, Model] = {}
    dlo_results: dict[str, object] = {}
    manifests: dict[str, object] = {}
    for dlo in DLOS:
        paths = trajectory_paths(dataset_root, dlo, "train")
        manifests[dlo] = file_manifest(paths)
        model, result = choose_model_for_dlo(paths, dlo, protocol)
        models[dlo] = model
        dlo_results[dlo] = result
    model_path = output_root / "source_model.npz"
    save_models(model_path, models)
    source_result = {
        "contract": CONTRACT,
        "schema_version": 1,
        "stage": "source",
        "protocol_sha256": sha256_file(args.protocol),
        "train_manifest": manifests,
        "train_manifest_sha256": canonical_sha256(manifests),
        "dlos": dlo_results,
        "all_source_gates_passed": all(
            bool(result["source_gate"]["passed"])
            for result in dlo_results.values()
            if isinstance(result, dict)
        ),
        "target_data_read": False,
        "claim_boundary": (
            "Source selection uses only official DLO4/DLO5 train trajectories. "
            "It does not establish target performance, unique physical state, "
            "calibrated probabilities, or deployment safety."
        ),
    }
    source_result_path = output_root / "source_result.json"
    write_json(source_result_path, source_result)
    seal = {
        "contract": CONTRACT,
        "schema_version": 1,
        "stage": "source-seal",
        "protocol_sha256": sha256_file(args.protocol),
        "source_model_sha256": sha256_file(model_path),
        "source_result_sha256": sha256_file(source_result_path),
        "train_manifest_sha256": source_result["train_manifest_sha256"],
    }
    write_json(output_root / "source_seal.json", seal)
    return 0


def render_summary(result: dict[str, object]) -> str:
    lines = [
        "# DEFORM DLO4/DLO5 decision-identifiability result",
        "",
        "This is a source-frozen, within-DLO held-trajectory evaluation on the ",
        "official DEFORM DLO4 and DLO5 evaluation trajectories.",
        "",
        "| DLO | Baseline RMSE [mm] | Certificate RMSE [mm] | Ratio | "
        "Nonfallback | Mean regret |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    dlos = result["dlos"]
    assert isinstance(dlos, dict)
    for dlo in DLOS:
        item = dlos[dlo]
        assert isinstance(item, dict)
        aggregate = item["aggregate"]
        assert isinstance(aggregate, dict)
        fallback = aggregate["fallback"]
        certificate = aggregate["certificate"]
        assert isinstance(fallback, dict)
        assert isinstance(certificate, dict)
        lines.append(
            f"| {dlo} | {float(fallback['rmse_mm']):.3f} | "
            f"{float(certificate['rmse_mm']):.3f} | "
            f"{float(certificate['rmse_ratio_to_fallback']):.3f} | "
            f"{float(item['certificate_nonfallback_fraction']):.3f} | "
            f"{float(certificate['mean_normalized_regret']):.4f} |"
        )
    aggregate = result["aggregate"]
    assert isinstance(aggregate, dict)
    lines.extend(
        (
            "",
            "## Combined",
            "",
            f"- Certificate RMSE ratio: "
            f"`{float(aggregate['certificate_rmse_ratio']):.4f}`",
            f"- Mean paired trajectory improvement: "
            f"`{100.0 * float(aggregate['mean_trajectory_improvement']):.2f}%`",
            f"- 95% trajectory-bootstrap interval: "
            f"`[{100.0 * float(aggregate['improvent_ci95'][0]):.2f}%, "
            f"{float(aggregate['improvement_ci95'][1]):.2f}%]`",
            f"- Nonfallback decisions: "
            f"`{int(aggregate['certificate_nonfallback_count'])}` / "
            f"`{int(aggregate['decision_count'])}`",
            "",
            "## Claim boundary",
            "",
            str(result["claim_boundary"]),
            "",
        )
    )
    return "\n".join(lines)


def target_command(args: argparse.Namespace) -> int:
    if args.model is None or args.source_seal is None or args.request is None:
        raise ValueError("target command requires model, source seal, and request")
    protocol = load_protocol(args.protocol)
    request = validate_request(args.request)
    seal = read_json(args.source_seal)
    if (
        seal.get("contract") != CONTRACT
        or seal.get("stage") != "source-seal"
        or seal.get("protocol_sha256") != sha256_file(args.protocol)
        or seal.get("source_model_sha256") != sha256_file(args.model)
    ):
        raise ValueError("source model seal does not verify")
    models = load_models(args.model)
    dataset_root = args.dataset_root.resolve()
    dlo_results: dict[str, object] = {}
    eval_manifests: dict[str, object] = {}
    trajectory_improvements: list[float] = []
    decision_count = 0
    nonfallback_count = 0
    total_baseline_sse = 0.0
    total_certificate_sse = 0.0
    for dlo in DLOS:
        eval_paths = trajectory_paths(dataset_root, dlo, "eval")
        eval_manifests[dlo] = file_manifest(eval_paths)
        result = evaluate_paths(eval_paths, models[dlo], protocol)
        dlo_results[dlo] = result
        decision_count += int(result["decision_count"])
        aggregate = result["aggregate"]
        assert isinstance(aggregate, dict)
        fallback = aggregate["fallback"]
        certificate = aggregate["certificate"]
        assert isinstance(fallback, dict)
        assert isinstance(certificate, dict)
        count = int(result["decision_count"])
        total_baseline_sse += count * (float(fallback["rmse_mm"]) / 1000.0) ** 2
        total_certificate_sse += count * (
            float(certificate["rmse_mm"]) / 1000.0
        ) ** 2
        certificate_actions = certificate["action_counts"]
        assert isinstance(certificate_actions, list)
        nonfallback_count += count - int(certificate_actions[0])
        per_trajectory = result["per_trajectory"]
        assert isinstance(per_trajectory, list)
        trajectory_improvements.extend(
            1.0 - float(item["certificate_ratio"])
            for item in per_trajectory
            if isinstance(item, dict)
        )
    combined_ratio = math.sqrt(total_certificate_sse / total_baseline_sse)
    improvements = np.asarray(trajectory_improvements, dtype=np.float64)
    interval = bootstrap_interval(
        improvements,
        protocol.bootstrap_replicates,
        protocol.bootstrap_seed,
    )
    result = {
        "contract": CONTRACT,
        "schema_version": 1,
        "stage": "target-result",
        "run_key": request["run_key"],
        "protocol_sha256": sha256_file(args.protocol),
        "request_sha256": sha256_file(args.request),
        "source_seal_sha256": sha256_file(args.source_seal),
        "source_model_sha256": sha256_file(args.model),
        "eval_manifest": eval_manifests,
        "eval_manifest_sha256": canonical_sha256(eval_manifests),
        "dlos": dlo_results,
        "aggregate": {
            "decision_count": decision_count,
            "certificate_nonfallback_count": nonfallback_count,
            "certificate_rmse_ratio": combined_ratio,
            "mean_trajectory_improvement": float(np.mean(improvements)),
            "improvement_ci95": list(interval),
        },
        "target_tuning": False,
        "target_retries": False,
        "raw_predictions_published": False,
        "claim_boundary": (
            "This public-data result evaluates a frozen finite-action policy "
            "within DLO4 and DLO5. The exact certificate is conditional on the "
            "registered finite source-window support, quotient partition, and "
            "loss matrix. The pickle carrier co-locates permitted prefixes, "
            "future endpoint actions, and held internal-node outcomes; the code "
            "enforces semantic slicing but cannot provide byte-level channel "
            "separation. The result does not identify a unique physical state, "
            "prove the quotient physically correct, establish unseen-object "
            "generalization, calibrate uncertainty, or authorize deployment."
        ),
    }
    output_root = args.output_root.resolve()
    write_json(output_root / "target_result.json", result)
    (output_root / "summary.md").write_text(
        render_summary(result), encoding="utf-8"
    )
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "source":
        return source_command(args)
    return target_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
