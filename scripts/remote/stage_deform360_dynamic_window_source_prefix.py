#!/usr/bin/env python3
"""Stage a source-only dynamic window without changing the frozen v1 runner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

from bayesian_phystwin.deform360_online_belief_evaluation import _sha256
from bayesian_phystwin.deform360_selective_virtual_sensing_staging import (
    dynamic_window_source_case,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_frozen_runner(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "deform360_selective_prediction_prefix_frozen_runner", path
    )
    _require(spec is not None and spec.loader is not None, "cannot load frozen runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical_config_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("config_sha256", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _consume_source_arguments(argv: list[str]) -> tuple[Path, Path, list[str]]:
    remaining = list(argv)
    values = []
    for option in ("--source-window-selection-seal", "--source-config"):
        _require(option in remaining, f"missing required argument: {option}")
        index = remaining.index(option)
        _require(index + 1 < len(remaining), f"missing value for {option}")
        values.append(Path(remaining[index + 1]).resolve())
        del remaining[index : index + 2]
    return values[0], values[1], remaining


def main() -> int:
    selection_path, source_config_path, frozen_argv = _consume_source_arguments(
        sys.argv
    )
    runner_path = Path(__file__).with_name(
        "stage_deform360_selective_prediction_prefix.py"
    )
    frozen = _load_frozen_runner(runner_path)
    original_argv = sys.argv
    try:
        sys.argv = frozen_argv
        args = frozen._parse_args()
    finally:
        sys.argv = original_argv
    protocol = frozen.load_selective_virtual_sensing_protocol(args.protocol)
    record = frozen._case(args.protocol, args.object_id, args.episode_id)
    selection_seal = json.loads(selection_path.read_text(encoding="utf-8"))
    source_row = dynamic_window_source_case(selection_seal, str(record["case"]))
    selection = source_row["translation_contact_v2"]
    _require(
        selection.get("has_contact_supported_future_motion") is True,
        "source window has no contact-supported future motion",
    )
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    _require(
        source_config.get("config_sha256") == _canonical_config_sha256(source_config),
        "source config checksum changed",
    )
    _require(
        _sha256(source_config_path) == selection_seal.get("source_config_sha256")
        and source_config["config"]["source_cohort"]["protocol_config_sha256"]
        == protocol["config_sha256"],
        "source-window seal belongs to another cohort",
    )
    wrapper_path = Path(__file__).resolve()
    original_canonical = frozen._canonical_sha256

    def amend_and_hash(payload: dict[str, object]) -> str:
        if payload.get("artifact_kind") == "Deform360SelectivePredictionPrefix":
            payload["source_window_selection"] = {
                "protocol_id": selection_seal["protocol_id"],
                "result_sha256": selection_seal["result_sha256"],
                "file_sha256": _sha256(selection_path),
                "claim_boundary": (
                    "exploratory source-window diagnostic; not the frozen v1 "
                    "prospective selection"
                ),
            }
            payload["inputs_sha256"]["source_window_selection"] = _sha256(
                selection_path
            )
            payload["inputs_sha256"]["dynamic_window_source_config"] = _sha256(
                source_config_path
            )
            payload["inputs_sha256"]["dynamic_window_staging_wrapper"] = _sha256(
                wrapper_path
            )
            payload["information_boundary"].update(
                {
                    "tactile_read": True,
                    "tactile_read_for_dataset_window_selection": True,
                    "tactile_exposed_to_prediction_method": False,
                }
            )
        return original_canonical(payload)

    frozen.select_action_only_window = lambda *_args, **_kwargs: dict(selection)
    frozen._canonical_sha256 = amend_and_hash
    frozen._parse_args = lambda: args
    return int(frozen.main())


if __name__ == "__main__":
    raise SystemExit(main())
