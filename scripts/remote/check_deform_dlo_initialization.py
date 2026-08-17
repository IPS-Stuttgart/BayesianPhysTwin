#!/usr/bin/env python3
"""Verify DLO-specific DEFORM construction without reading trajectory data."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import run_deform_dlo_source as source_runtime

from bayesian_phystwin.deform_dlo_source import sha256_file
from bayesian_phystwin.deform_dlo_upstream import load_deform_dlo_initialization


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise RuntimeError(f"locked output differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main() -> int:
    args = _parse_args()
    upstream_root = args.upstream_root.resolve()
    data_root = upstream_root / "data_set"
    for dlo_type in ("DLO1", "DLO2"):
        for partition in ("train", "eval"):
            source_runtime._install_eval_read_guard(data_root / dlo_type / partition)

    upstream = source_runtime._assert_upstream(
        upstream_root,
        args.upstream_commit,
    )
    import torch

    modules = source_runtime._load_upstream(upstream_root)
    raw_dlo1 = modules.DEFORM_sim(
        n_vert=13,
        n_edge=12,
        pbd_iter=10,
        device=args.device,
    )
    _, dlo1 = source_runtime._build_dlo_model(
        modules,
        torch,
        args.device,
        dlo_type="DLO1",
        node_count=13,
    )
    _, dlo2 = source_runtime._build_dlo_model(
        modules,
        torch,
        args.device,
        dlo_type="DLO2",
        node_count=12,
    )
    dlo2_initialization = load_deform_dlo_initialization(
        upstream_root / "train_DEFORM.py",
        "DLO2",
    )
    checks = {
        "dlo1_rest_geometry_constructor_parity": bool(
            torch.equal(raw_dlo1.rest_vert, dlo1.rest_vert)
        ),
        "dlo1_rest_lengths_constructor_parity": bool(
            torch.equal(raw_dlo1.m_restEdgeL, dlo1.m_restEdgeL)
            and torch.equal(raw_dlo1.m_restRegionL, dlo1.m_restRegionL)
        ),
        "dlo2_rest_shape": list(dlo2.rest_vert.shape),
        "dlo2_rest_edge_length_shape": list(dlo2.m_restEdgeL.shape),
        "dlo2_rest_region_length_shape": list(dlo2.m_restRegionL.shape),
        "dlo2_bend_stiffness_expected": dlo2_initialization.bend_stiffness,
        "dlo2_bend_stiffness_observed_mean": float(
            dlo2.DEFORM_func.bend_stiffness.mean().item()
        ),
        "dlo2_twist_stiffness_expected": dlo2_initialization.twist_stiffness,
        "dlo2_twist_stiffness_observed_mean": float(
            dlo2.DEFORM_func.twist_stiffness.mean().item()
        ),
        "source_hash_verified": (
            dlo2_initialization.source_sha256
            == upstream["source_files"]["train_DEFORM.py"]["sha256"]
        ),
    }
    passed = (
        checks["dlo1_rest_geometry_constructor_parity"] is True
        and checks["dlo1_rest_lengths_constructor_parity"] is True
        and checks["dlo2_rest_shape"] == [1, 12, 3]
        and checks["dlo2_rest_edge_length_shape"] == [1, 11]
        and checks["dlo2_rest_region_length_shape"] == [1, 11]
        and abs(
            float(checks["dlo2_bend_stiffness_observed_mean"])
            - dlo2_initialization.bend_stiffness
        )
        <= 1e-10
        and abs(
            float(checks["dlo2_twist_stiffness_observed_mean"])
            - dlo2_initialization.twist_stiffness
        )
        <= 1e-10
        and checks["source_hash_verified"] is True
    )
    script_path = Path(__file__).resolve()
    parser_path = (
        script_path.parents[2] / "src" / "bayesian_phystwin" / "deform_dlo_upstream.py"
    )
    runner_path = script_path.with_name("run_deform_dlo_source.py")
    payload = {
        "schema_version": 1,
        "contract": "deform-dlo2-initialization-construction-smoke-v1",
        "claim_boundary": (
            "Target-free model-construction verification only; no DLO1/DLO2 "
            "trajectory or official-evaluation artifact was read."
        ),
        "executed_on": f"{socket.gethostname()}:{args.device}",
        "dlo1_source_read": False,
        "dlo2_source_read": False,
        "official_eval_read": False,
        "upstream": {
            "commit": upstream["commit"],
            "train_deform_sha256": upstream["source_files"]["train_DEFORM.py"][
                "sha256"
            ],
        },
        "implementation": {
            "parser_sha256": sha256_file(parser_path),
            "runner_sha256": sha256_file(runner_path),
            "verifier_sha256": sha256_file(script_path),
        },
        "checks": checks,
        "passed": passed,
    }
    _write_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
