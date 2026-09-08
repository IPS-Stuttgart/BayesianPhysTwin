"""Verify a raw-data reproduction against PR 940, without changing its gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

ARCHIVE_SHA256 = "e7e182e009ac234fb611afb6ca03eb1006ee65da2688ec2a084105667fa4e591"
METRICS = ("nll", "crps_m", "coverage90", "width90_m", "brier")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(left, right, path="root", differences=None):
    """Compare complete numerical trees, returning absolute numeric errors."""
    if differences is None:
        differences = []
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError(f"Mapping mismatch: {path}")
        for key in left:
            compare(left[key], right[key], f"{path}.{key}", differences)
    elif isinstance(left, list):
        if not isinstance(right, list) or len(left) != len(right):
            raise ValueError(f"List mismatch: {path}")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            compare(a, b, f"{path}[{index}]", differences)
    elif isinstance(left, (int, float)) and not isinstance(left, bool):
        if not isinstance(right, (int, float)) or isinstance(right, bool):
            raise ValueError(f"Numeric type mismatch: {path}")
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError(f"Nonfinite number: {path}")
        if not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"Numeric mismatch: {path}: {left} != {right}")
        differences.append(abs(left - right))
    elif left != right:
        raise ValueError(f"Value mismatch: {path}")
    return differences


def verify(root: Path, historical_archive: Path) -> dict:
    if sha256(historical_archive) != ARCHIVE_SHA256:
        raise ValueError("Historical archive digest mismatch")
    with zipfile.ZipFile(historical_archive) as archive:
        names = [n for n in archive.namelist() if n == "result.json" or n.endswith("/result.json")]
        if len(names) != 1:
            raise ValueError("Expected exactly one historical result.json")
        old = json.loads(archive.read(names[0]))
    new = json.loads((root / "result.json").read_text())
    seal = json.loads((root / "prediction_seal.json").read_text())
    seal_files = {"protocol_sha256": "protocol.json", "models_sha256": "frozen_models.json",
                  "predictions_sha256": "sealed_predictions.json", "input_manifest_sha256": "input_manifest.json"}
    for field, name in seal_files.items():
        if sha256(root / name) != seal[field]:
            raise ValueError(f"New seal mismatch: {field}")
    if seal["input_manifest_sha256"] != old["seal"]["input_manifest_sha256"]:
        raise ValueError("Input recordings differ from original experiment")
    errors = []
    for field in ("config", "accounting", "aggregate", "posterior_minus_comparator",
                  "primary_hypothesis_supported", "decision", "statistical_unit",
                  "common_mean_coordinate_l1_m_descriptive"):
        compare(new[field], old[field], field, errors)
    rows = [json.loads(line) for line in (root / "trajectory_panel_scores.jsonl").read_text().splitlines()]
    groups = {}
    for row in rows:
        key = (str(row["source_size"]), row["arm"], row["dlo"], row["trajectory"])
        groups.setdefault(key, []).append(row)
    reaggregation_errors = []
    for size, arms in new["aggregate"].items():
        for arm, scores in arms.items():
            selected = [v for k, v in groups.items() if k[:2] == (size, arm)]
            if len(selected) != 24 or any(len(v) != 15 for v in selected):
                raise ValueError("Incorrect independent-unit accounting")
            for metric in METRICS:
                per_case = [math.fsum(r[metric] for r in case) / len(case) for case in selected]
                value = math.fsum(per_case) / len(per_case)
                compare(scores[metric], value, f"reaggregation.{size}.{arm}.{metric}", reaggregation_errors)
    receipt = {"status": "raw-data-reproduction-verified", "historical_run_id": 33985128557,
               "historical_artifact_sha256": ARCHIVE_SHA256, "fresh_confirmation": False,
               "source_revision": seal["source_revision"], "input_manifest_identical": True,
               "scientific_code_modified": False, "retuning": False,
               "historical_numeric_values_compared": len(errors),
               "maximum_numeric_difference": max(errors, default=0.0),
               "score_rows": len(rows), "independent_reaggregated_metrics": len(reaggregation_errors),
               "maximum_reaggregation_difference": max(reaggregation_errors, default=0.0),
               "primary_hypothesis_supported": new["primary_hypothesis_supported"],
               "current_result_sha256": sha256(root / "result.json"),
               "current_prediction_seal_sha256": sha256(root / "prediction_seal.json")}
    (root / "reproduction_verification.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    text = ["# Raw-data reproduction of PR 940", "", "Frozen experiment rerun on canonical DLO4/DLO5 source recordings.",
            "This is reproducibility evidence, not independent confirmation or a new parameter search.", "",
            f"Primary hypothesis supported: **{new['primary_hypothesis_supported']}**", "",
            "| Fit trajectories per object (+12 calibration) | Posterior CRPS (mm) | Plug-in CRPS (mm) | Difference (mm), 95% trajectory CI |",
            "|---|---:|---:|---:|---|" ]
    for size in ("8", "16", "32"):
        means = new["aggregate"][size]
        c = new["posterior_minus_comparator"][size]["plugin_gaussian"]["crps_m"]
        low, high = c["trajectory_bootstrap_95_ci"]
        text.append(f"| {size} | {1000*means['posterior_student']['crps_m']:.5f} | {1000*means['plugin_gaussian']['crps_m']:.5f} | {1000*c['difference']:+.6f} [{1000*low:+.6f}, {1000*high:+.6f}] |")
    text += ["", "Primary setting is 32 fitting recordings; 8 and 16 are prespecified secondary settings.",
             "The mean is a compact action-conditioned ridge surrogate, not the DEFORM physical simulator.",
             "Bootstrap prediction controls preserve whole future-shape vectors; this is not a temporally joint trajectory-query test.",
             "Two fixed physical objects; intervals resample complete source-test recordings within each object.",
             "", "```json", json.dumps(receipt, indent=2), "```", ""]
    (root / "REPRODUCTION.md").write_text("\n".join(text))
    print("\n".join(text), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-archive", type=Path, required=True)
    args = parser.parse_args()
    verify(args.output, args.historical_archive)
