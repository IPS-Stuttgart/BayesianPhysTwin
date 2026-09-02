"""Verify the source-only release without loading predictions or protected data."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/sota/deform_dlo3_cross_backend_source_release_v1"
ARMS = ("cross-backend", "cross-backend-scalar")
MEMBERS = {
    f"{arm}/{name}"
    for arm in ARMS
    for name in (
        "preflight.json",
        "method_seal.json",
        "result.json",
        "report.md",
        "trajectory-results.csv",
    )
}


def _read(path: str) -> dict[str, Any]:
    value = json.loads((EVIDENCE / path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(actual: Any, expected: Any) -> None:
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_source_export_is_exact_and_does_not_claim_whole_archive_verification() -> None:
    receipt = _read("source_export_receipt.json")
    identity = dict(receipt)
    digest = identity.pop("receipt_sha256")
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == digest
    assert digest == "f0d4ca0ff98bcfd730a8a4036ae7a6165dd3667552724a28b3f5f5a9039e2116"
    assert receipt["run_id"] == 33536420739
    assert receipt["artifact_id"] == 9811886089
    assert set(receipt["source_members"]) == MEMBERS
    for name, expected in receipt["source_members"].items():
        assert _sha(EVIDENCE / name) == expected
    assert _sha(EVIDENCE / "source_export.py.txt") == receipt["retrieval_code_sha256"]
    assert receipt["whole_zip_sha256_recomputed"] is False
    assert receipt["protected_member_payload_read"] is False
    assert receipt["source_member_sha256_matches_archive_manifest"] is True
    assert sum(item["length"] for item in receipt["ranges"]) == 20906
    allowed_purposes = {"zip-end-metadata", "zip-central-directory-metadata"}
    allowed_purposes.update(
        f"{name}:{stage}"
        for name in MEMBERS | {"SHA256SUMS"}
        for stage in ("local-header", "local-name", "allowed-payload")
    )
    assert {item["purpose"] for item in receipt["ranges"]} == allowed_purposes


@pytest.mark.parametrize("arm", ARMS)
def test_seals_bind_protocol_inputs_and_closed_boundaries(arm: str) -> None:
    result = _read(f"{arm}/result.json")
    seal = _read(f"{arm}/method_seal.json")
    preflight = _read(f"{arm}/preflight.json")
    protocol_name = Path(seal["protocol"]["path"]).name
    protocol_path = ROOT / "protocols" / protocol_name
    assert _sha(protocol_path) == seal["protocol"]["sha256"]
    assert protocol_path.stat().st_size == seal["protocol"]["size_bytes"]
    assert _sha(EVIDENCE / arm / "method_seal.json") == result["method_seal"]["sha256"]
    assert result["source_payload_loaded_after_method_seal"] is True
    assert len(set(seal["source_names"])) == 8
    assert seal["source_names"] == preflight["source_names"]
    direct_protocol = json.loads(
        (ROOT / "protocols/deform_dlo3_cross_backend_transfer_v1.json").read_text()
    )
    registered = direct_protocol["artifacts"]
    for name in ("source_manifest", "pyelastica_source_predictions"):
        assert seal[name] == registered[name]
    for model in registered["deform_local_residual_models"]:
        expected = {key: model[key] for key in ("path", "sha256", "size_bytes")}
        assert seal["deform_local_residual_models"][str(model["seed"])] == expected
    for name in (
        "protocol",
        "source_manifest",
        "pyelastica_source_predictions",
        "deform_local_residual_models",
    ):
        assert result[name] == seal[name] == preflight[name]
    for key in ("dlo3_official_evaluation_read", "dlo4_or_dlo5_read"):
        assert result["information_boundary"][key] is False
        assert seal[key] is preflight[key] is False
    assert seal["source_numeric_payload_opened"] is False
    assert result["information_boundary"]["paper_claim_authorized"] is False
    assert result["information_boundary"]["deform_refit"] is False
    if arm == "cross-backend-scalar":
        assert (
            result["information_boundary"]["same_trajectory_label_used_for_its_scalar"]
            is False
        )
        assert (
            result["information_boundary"]["pyelastica_high_dimensional_refit"] is False
        )


COMPARISONS = [
    ("cross-backend", "primary_vs_raw_pyelastica", 20260901),
    ("cross-backend", "pyelastica_specific_vs_raw_pyelastica", 20260902),
    ("cross-backend", "direct_transfer_vs_pyelastica_specific", 20260903),
    *[
        ("cross-backend", f"individual_seed_vs_raw_pyelastica/{seed}", 20260901 + seed)
        for seed in (42, 43, 44)
    ],
    ("cross-backend-scalar", "direct_vs_raw_pyelastica", 20260901),
    ("cross-backend-scalar", "scalar_vs_raw_pyelastica", 20260902),
    ("cross-backend-scalar", "pyelastica_specific_vs_raw_pyelastica", 20260903),
    ("cross-backend-scalar", "scalar_vs_pyelastica_specific", 20260904),
]


@pytest.mark.parametrize(("arm", "comparison", "seed"), COMPARISONS)
def test_recompute_all_paired_trajectory_summaries(
    arm: str, comparison: str, seed: int
) -> None:
    summary = _read(f"{arm}/result.json")
    for part in comparison.split("/"):
        summary = summary[part]
    rows = summary["cases"]
    assert len(rows) == len({row["name"] for row in rows}) == 8
    candidate = np.array([row["candidate_l1_m"] for row in rows])
    baseline = np.array([row["baseline_l1_m"] for row in rows])
    assert np.isfinite(candidate).all() and np.isfinite(baseline).all()
    assert (candidate >= 0).all() and (baseline > 0).all()
    difference = candidate - baseline
    _close(summary["candidate_mean_l1_m"], float(candidate.mean()))
    _close(summary["baseline_mean_l1_m"], float(baseline.mean()))
    _close(summary["mean_difference_m"], float(difference.mean()))
    _close(
        summary["relative_improvement"], float(1 - candidate.mean() / baseline.mean())
    )
    _close(summary["maximum_case_ratio"], float((candidate / baseline).max()))
    assert summary["wins"] == int((difference < -1e-12).sum())
    assert summary["ties"] == int((np.abs(difference) <= 1e-12).sum())
    assert summary["losses"] == int((difference > 1e-12).sum())
    indices = np.random.default_rng(seed).integers(0, 8, size=(10000, 8))
    interval = np.quantile(difference[indices].mean(axis=1), (0.025, 0.975))
    _close(summary["object_bootstrap_95_interval_m"], interval)
    for index, row in enumerate(rows):
        _close(row["difference_m"], float(difference[index]))
        _close(
            row["candidate_to_baseline_ratio"],
            float(candidate[index] / baseline[index]),
        )
        assert row["candidate_wins"] == bool(difference[index] < -1e-12)


def _point_gate(summary: dict[str, Any]) -> bool:
    return bool(
        summary["relative_improvement"] >= 0.01
        and summary["wins"] >= 6
        and summary["maximum_case_ratio"] <= 1.10
    )


def test_rederive_both_registered_gates_without_promoting_scalar_over_direct() -> None:
    direct = _read("cross-backend/result.json")
    scalar = _read("cross-backend-scalar/result.json")
    primary = direct["primary_vs_raw_pyelastica"]
    improving_seeds = sum(
        row["relative_improvement"] > 0
        for row in direct["individual_seed_vs_raw_pyelastica"].values()
    )
    assert direct["promotion_gate"]["supported"] == (
        _point_gate(primary) and improving_seeds >= 2
    )
    assert improving_seeds == direct["promotion_gate"]["improving_seed_models"] == 3
    assert primary == scalar["direct_vs_raw_pyelastica"]
    alignment = np.array(scalar["directional_alignment"]["trajectory_cosines"])
    scalars = np.array(scalar["fold_scalars"]["values"])
    direction_gate = bool(
        (alignment > 0).sum() >= 6
        and np.median(alignment) >= 0.05
        and (scalars > 0).sum() >= 6
    )
    assert np.isfinite(alignment).all() and np.isfinite(scalars).all()
    assert (np.abs(alignment) <= 1).all() and ((scalars >= 0) & (scalars <= 4)).all()
    assert scalar["promotion_gate"]["supported"] == (
        _point_gate(scalar["scalar_vs_raw_pyelastica"]) and direction_gate
    )
    assert primary["wins"] == 8
    assert scalar["scalar_vs_raw_pyelastica"]["wins"] == 6
    assert (
        primary["candidate_mean_l1_m"]
        < scalar["methods"]["leave_one_trajectory_out_one_scalar"]
    )
    assert scalar["scalar_vs_raw_pyelastica"]["object_bootstrap_95_interval_m"][1] > 0
    specific = direct["pyelastica_specific_vs_raw_pyelastica"]
    retained = primary["mean_difference_m"] / specific["mean_difference_m"]
    _close(direct["backend_specific_gain_retained_fraction"], retained)


@pytest.mark.parametrize("arm", ARMS)
def test_csvs_match_complete_paired_result_rows(arm: str) -> None:
    result = _read(f"{arm}/result.json")
    with (EVIDENCE / arm / "trajectory-results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert [row["name"] for row in rows] == _read(f"{arm}/method_seal.json")[
        "source_names"
    ]
    if arm == "cross-backend-scalar":
        expected = result["cases"]
        assert [row["name"] for row in rows] == [row["name"] for row in expected]
        for row, recorded in zip(rows, expected, strict=True):
            for key in (
                "fold_scalar",
                "alignment_cosine",
                "baseline_l1_m",
                "direct_l1_m",
                "scalar_l1_m",
            ):
                _close(float(row[key]), recorded[key])
    else:
        expected = result["primary_vs_raw_pyelastica"]["cases"]
        assert [row["name"] for row in rows] == [row["name"] for row in expected]
        for row, recorded in zip(rows, expected, strict=True):
            _close(float(row["raw_pyelastica_l1_m"]), recorded["baseline_l1_m"])
            _close(
                float(row["deform_no_refit_equal_seed_l1_m"]),
                recorded["candidate_l1_m"],
            )
