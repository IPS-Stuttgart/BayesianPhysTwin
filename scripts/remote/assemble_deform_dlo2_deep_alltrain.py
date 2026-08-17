#!/usr/bin/env python3
"""Assemble two verified all-train DLO2 members into one frozen method."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bayesian_phystwin.deform_dlo_alltrain import (
    DEFORM_DLO2_DEEP_ALLTRAIN_RESULT_CONTRACT,
    load_deform_dlo2_deep_alltrain_protocol,
    validate_deform_dlo2_deep_alltrain_authorization,
)
from bayesian_phystwin.deform_dlo_source import (
    load_deform_dlo_source_protocol,
    sha256_file,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--ensemble-result", type=Path, required=True)
    parser.add_argument("--seed42-source-protocol", type=Path, required=True)
    parser.add_argument("--seed43-source-protocol", type=Path, required=True)
    parser.add_argument("--seed42-alltrain-result", type=Path, required=True)
    parser.add_argument("--seed43-alltrain-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"locked output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _verified_json(identity: Mapping[str, object], *, label: str) -> dict[str, object]:
    path = Path(str(identity.get("path", ""))).resolve()
    if not path.is_file() or sha256_file(path) != identity.get("sha256"):
        raise ValueError(f"{label} identity does not verify")
    return _read_json(path)


def _verified_checkpoint(
    identity: Mapping[str, object],
    *,
    seed: int,
    update: int,
    protocol_sha256: str,
    schedule_sha256: str,
    method_spec_sha256: str,
    torch: Any,
) -> None:
    path = Path(str(identity.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(identity.get("size_bytes", -1))
        or sha256_file(path) != identity.get("sha256")
    ):
        raise ValueError("deep all-train checkpoint identity does not verify")
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(bundle, Mapping)
        or not isinstance(bundle.get("model_state_dict"), Mapping)
        or int(bundle.get("seed", -1)) != seed
        or int(bundle.get("update", -1)) != update
        or bundle.get("deep_alltrain_protocol_sha256") != protocol_sha256
        or bundle.get("schedule_sha256") != schedule_sha256
        or bundle.get("method_spec_sha256") != method_spec_sha256
    ):
        raise ValueError("deep all-train checkpoint lineage differs")


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_deform_dlo2_deep_alltrain_protocol(protocol_path)
    protocol_sha256 = sha256_file(protocol_path)
    source_protocol_paths = {
        42: args.seed42_source_protocol.resolve(),
        43: args.seed43_source_protocol.resolve(),
    }
    source_protocol_sha256s = {}
    for seed in (42, 43):
        source_protocol = load_deform_dlo_source_protocol(
            source_protocol_paths[seed]
        )
        source_protocol_sha256s[seed] = sha256_file(source_protocol_paths[seed])
        if (
            source_protocol_sha256s[seed]
            != protocol["parents"][f"seed{seed}_source_protocol"]["sha256"]
            or int(source_protocol["training"]["random_seed"]) != seed
        ):
            raise ValueError(f"deep all-train seed-{seed} source protocol differs")

    ensemble_result_path = args.ensemble_result.resolve()
    ensemble_result = _read_json(ensemble_result_path)
    selection_identity = ensemble_result.get("selection_seal")
    if not isinstance(selection_identity, Mapping):
        raise ValueError("DLO2 ensemble result omits its selection seal")
    selection_path = Path(str(selection_identity.get("path", ""))).resolve()
    if (
        not selection_path.is_file()
        or sha256_file(selection_path) != selection_identity.get("sha256")
    ):
        raise ValueError("DLO2 ensemble selection seal does not verify")
    selection_seal = _read_json(selection_path)

    alltrain_result_paths = {
        42: args.seed42_alltrain_result.resolve(),
        43: args.seed43_alltrain_result.resolve(),
    }
    alltrain_results = {
        seed: _read_json(alltrain_result_paths[seed]) for seed in (42, 43)
    }
    method_specs = {}
    final_members = {}
    source_results = None
    for seed in (42, 43):
        result = alltrain_results[seed]
        protocol_identity = result.get("protocol")
        ensemble_identity = result.get("ensemble_result")
        method_identity = result.get("method_spec")
        final_identity = result.get("final_member")
        if (
            result.get("contract")
            != "deform-dlo2-deep-alltrain-seed-result-v1"
            or result.get("official_eval_read") is not False
            or result.get("assembly_authorized") is not True
            or int(result.get("seed", -1)) != seed
            or not isinstance(protocol_identity, Mapping)
            or protocol_identity.get("sha256") != protocol_sha256
            or not isinstance(ensemble_identity, Mapping)
            or ensemble_identity.get("sha256") != sha256_file(ensemble_result_path)
            or not isinstance(method_identity, Mapping)
            or not isinstance(final_identity, Mapping)
        ):
            raise ValueError(f"deep all-train seed-{seed} result differs")
        method_specs[seed] = _verified_json(
            method_identity, label=f"seed-{seed} method spec"
        )
        final_members[seed] = _verified_json(
            final_identity, label=f"seed-{seed} final member"
        )
        raw_sources = method_specs[seed].get("source_results")
        if not isinstance(raw_sources, Mapping):
            raise ValueError(f"seed-{seed} method spec omits source results")
        candidate_sources = {}
        for source_seed in (42, 43):
            identity = raw_sources.get(str(source_seed))
            if not isinstance(identity, Mapping):
                raise ValueError("deep all-train source identity is malformed")
            candidate_sources[source_seed] = _verified_json(
                identity, label=f"source seed-{source_seed} result"
            )
        if source_results is None:
            source_results = candidate_sources
        elif candidate_sources != source_results:
            raise ValueError("deep all-train seed runs bind different source results")
    if source_results is None:
        raise RuntimeError("deep all-train source results are absent")
    source_result_sha256s = {
        seed: str(method_specs[42]["source_results"][str(seed)]["sha256"])
        for seed in (42, 43)
    }
    selected = validate_deform_dlo2_deep_alltrain_authorization(
        protocol,
        source_results,
        ensemble_result,
        selection_seal,
        source_protocol_sha256s=source_protocol_sha256s,
        source_result_sha256s=source_result_sha256s,
        ensemble_protocol_sha256=protocol["parents"]["ensemble_protocol"][
            "sha256"
        ],
        selection_seal_sha256=sha256_file(selection_path),
    )

    import torch

    member_checkpoints = {}
    runtimes = []
    for seed in (42, 43):
        member = final_members[seed]
        checkpoint = member.get("selected_checkpoint")
        schedule = alltrain_results[seed].get("window_schedule")
        if (
            member.get("contract") != "deform-dlo2-deep-alltrain-seed-final-v1"
            or member.get("official_eval_read") is not False
            or int(member.get("seed", -1)) != seed
            or member.get("operator") != selected["operator"]
            or float(member.get("weight", -1.0)) != selected["weights"][seed]
            or int(member.get("selected_update", -1))
            != selected["member_updates"][seed]
            or not isinstance(checkpoint, Mapping)
            or not isinstance(schedule, Mapping)
        ):
            raise ValueError(f"deep all-train seed-{seed} final member differs")
        _verified_checkpoint(
            checkpoint,
            seed=seed,
            update=selected["member_updates"][seed],
            protocol_sha256=protocol_sha256,
            schedule_sha256=str(schedule.get("sha256", "")),
            method_spec_sha256=str(
                alltrain_results[seed]["method_spec"]["sha256"]
            ),
            torch=torch,
        )
        member_checkpoints[str(seed)] = checkpoint
        runtimes.append(alltrain_results[seed].get("runtime"))
    if not all(isinstance(runtime, Mapping) for runtime in runtimes):
        raise ValueError("deep all-train seed runtime is malformed")
    runtime = {
        key: runtimes[0].get(key) for key in ("python", "torch", "cuda")
    }
    if any(
        {key: candidate.get(key) for key in runtime} != runtime
        for candidate in runtimes[1:]
    ):
        raise ValueError("deep all-train seed software runtimes differ")

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    final_method = {
        "schema_version": 1,
        "contract": "deform-dlo2-deep-alltrain-final-method-v1",
        "official_eval_read": False,
        "operator": selected["operator"],
        "seed_weights": selected["weights"],
        "member_updates": selected["member_updates"],
        "comparison_baseline_seed": selected["comparison_baseline_seed"],
        "comparison_baseline_checkpoint": member_checkpoints[
            str(selected["comparison_baseline_seed"])
        ],
        "member_checkpoints": member_checkpoints,
        "variance_calibration": {
            "scale": selected["validation_fitted_variance_scale"],
            "floor_m2": selected["variance_floor_m2"],
            "nominal_coordinate_coverage": selected[
                "nominal_coordinate_coverage"
            ],
        },
        "seed_results": {
            str(seed): {
                "path": str(alltrain_result_paths[seed]),
                "sha256": sha256_file(alltrain_result_paths[seed]),
            }
            for seed in (42, 43)
        },
    }
    final_method_path = output_root / "final_method.json"
    _write_json(final_method_path, final_method)
    result = {
        "schema_version": 1,
        "contract": DEFORM_DLO2_DEEP_ALLTRAIN_RESULT_CONTRACT,
        "claim_boundary": protocol["claim_boundary"],
        "official_eval_read": False,
        "official_eval_execution_authorized": True,
        "protocol": {
            "path": str(protocol_path),
            "sha256": protocol_sha256,
        },
        "ensemble_result": {
            "path": str(ensemble_result_path),
            "sha256": sha256_file(ensemble_result_path),
        },
        "final_method": {
            "path": str(final_method_path),
            "sha256": sha256_file(final_method_path),
        },
        "selected_method": {
            "operator": selected["operator"],
            "seed_weights": selected["weights"],
            "member_updates": selected["member_updates"],
            "comparison_baseline_seed": selected["comparison_baseline_seed"],
            "variance_calibration": final_method["variance_calibration"],
        },
        "seed_results": final_method["seed_results"],
        "runtime": runtime,
    }
    result_path = output_root / "alltrain_result.json"
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
