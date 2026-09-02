#!/usr/bin/env python3
"""Reproduce one complete component and export its world-level portfolio record."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

PORTFOLIO_PROTOCOL_ID = (
    "595ea3752b83ad8403555bd5f237a8b0ccf764b7e1bfa094fe8a36664182a7c2"
)
BASE = Path("/home/florianpfaff/source-only/dlolab-query-portfolio-replication-v1")
WRAPPING_SOURCE = Path(
    "/home/florianpfaff/source-only/bpt-query-portfolio-replication-v1-56d24906"
)
SLINGSHOT_SOURCE = Path(
    "/home/florianpfaff/source-only/bpt-query-portfolio-replication-v1-c9dd44b3"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_id(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _component_record(
    *,
    query_id: str,
    gain: NDArray[np.float64],
    deployed: NDArray[np.bool_],
    component_result_id: str,
    component_result_sha256: str,
) -> dict[str, Any]:
    values = np.asarray(gain, dtype=np.float64)
    selected = np.asarray(deployed, dtype=np.bool_)
    if (
        values.shape != (320,)
        or selected.shape != values.shape
        or not np.isfinite(values).all()
        or np.any(~selected & (values != 0.0))
        or query_id not in {"dlolab_wrapping_v9", "dlolab_slingshot_v4"}
    ):
        raise ValueError("invalid complete component outcome")
    record: dict[str, Any] = {
        "schema": "bayesian_phystwin.query_portfolio_component_evidence",
        "version": 1,
        "portfolio_protocol_id": PORTFOLIO_PROTOCOL_ID,
        "query_id": query_id,
        "component_result_id": component_result_id,
        "component_result_sha256": component_result_sha256,
        "gain": values.tolist(),
        "candidate_deployed": selected.tolist(),
        "ordinary_success": np.ones(320, dtype=np.bool_).tolist(),
        "worlds": 320,
        "partial_result": False,
    }
    record["artifact_id"] = _content_id(record)
    return record


def _wrapping() -> dict[str, Any]:
    wrapper = _load(
        WRAPPING_SOURCE
        / "scripts/remote/run_dlolab_wrapping_portfolio_replication_v1.py",
        "portfolio_wrapping_export",
    )
    runner = wrapper._load_runner()
    output = BASE / "wrapping"
    lock, _, bank = runner._validate(output)
    result = runner.read_record(output / "result.json")
    if (
        result.get("status") != "complete"
        or result.get("source_gate_passed") is not True
        or result.get("technical_failures") != 0
        or result.get("ordinary_worlds") != 320
    ):
        raise ValueError("complete passing Wrapping result required")
    _, decisions, _ = runner._load_decisions(output, lock, bank)
    rows = [runner._load_future(output, lock, index) for index in range(320)]
    rewards = np.asarray([row[1] for row in rows], dtype=np.float64)
    metrics = runner.score(
        decisions["decisions"],
        rewards,
        all_native_qa=all(row[2]["qa_passed"] for row in rows),
        calibration_certificate_valid=True,
    )
    if any(result.get(key) != value for key, value in metrics.items()):
        raise ValueError("Wrapping result does not reproduce")
    decision = np.asarray(decisions["decisions"], dtype=np.int64)
    selected = np.take_along_axis(rewards[:, None, :], decision, axis=2).mean(axis=1)
    primary = runner.ARM_NAMES.index("posterior_975_guard")
    gain = selected[:, primary] - selected[:, 0]
    deployed = np.any(decision[:, :, primary] != decision[:, :, 0], axis=1)
    return _component_record(
        query_id="dlolab_wrapping_v9",
        gain=gain,
        deployed=deployed,
        component_result_id=result["artifact_id"],
        component_result_sha256=_sha256(output / "result.json"),
    )


def _slingshot() -> dict[str, Any]:
    wrapper = _load(
        SLINGSHOT_SOURCE
        / "scripts/remote/run_dlolab_slingshot_portfolio_replication_v4.py",
        "portfolio_slingshot_export",
    )
    configured = wrapper._load_runner()
    runner = configured.runner
    output = BASE / "slingshot-v4"
    result = runner.verify_result(output)
    if (
        result.get("source_gate_passed") is not True
        or result.get("technical_failures") != 0
        or result.get("ordinary_evaluation_worlds") != 320
    ):
        raise ValueError("complete passing Slingshot result required")
    lock = runner.validate_lock(output)
    _, decisions, _ = runner.load_evaluation_decisions(output, lock)
    rewards, _, _, all_qa = runner._future_rewards(
        output, lock, "evaluation", write=False
    )
    if not all_qa:
        raise ValueError("complete Slingshot native QA required")
    decision = np.asarray(decisions["decisions"], dtype=np.int64)
    selected = np.take_along_axis(rewards, decision, axis=1)
    gain = selected[:, 3] - selected[:, 0]
    deployed = decision[:, 3] != decision[:, 0]
    return _component_record(
        query_id="dlolab_slingshot_v4",
        gain=gain,
        deployed=deployed,
        component_result_id=result["artifact_id"],
        component_result_sha256=_sha256(output / "result.json"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", choices=("wrapping", "slingshot"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("component evidence output must be fresh")
    record = _wrapping() if args.query == "wrapping" else _slingshot()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
