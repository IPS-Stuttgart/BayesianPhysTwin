"""Frozen protocol checks for the exploratory DEFORM long-run continuation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

DEFORM_DLO_LONGRUN_SCHEMA_VERSION = 1
DEFORM_DLO_LONGRUN_CONTRACT = "deform-dlo-longrun-continuation-v2"


def load_deform_dlo_longrun_protocol(path: str | Path) -> dict[str, object]:
    """Load and validate the post-open DLO1 continuation protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DEFORM_DLO_LONGRUN_SCHEMA_VERSION:
        raise ValueError("unsupported DEFORM long-run schema")
    if payload.get("contract") != DEFORM_DLO_LONGRUN_CONTRACT:
        raise ValueError("unsupported DEFORM long-run contract")
    if payload.get("dlo_type") != "DLO1":
        raise ValueError("long-run development must remain on DLO1")
    if payload.get("source_test_status") != "post-open-exploratory-only":
        raise ValueError("long-run DLO1 source test must remain exploratory")
    if payload.get("official_eval_policy") != "forbidden":
        raise ValueError("long-run protocol must forbid official evaluation")
    if payload.get("fresh_confirmation_dlo") != "DLO2":
        raise ValueError("long-run confirmation must use fresh DLO2")

    source_result = payload.get("source_result")
    if not isinstance(source_result, Mapping):
        raise ValueError("long-run protocol omits its frozen source result")
    digest = str(source_result.get("sha256", ""))
    if len(digest) != 64:
        raise ValueError("long-run source-result digest is invalid")
    if source_result.get("required_advancement_authorized") is not False:
        raise ValueError("long-run must bind the failed short-budget decision")

    starting = payload.get("starting_checkpoint")
    if not isinstance(starting, Mapping):
        raise ValueError("long-run protocol omits its starting checkpoint")
    starting_update = int(starting.get("global_update", -1))
    if starting_update != 280:
        raise ValueError("long-run must resume the frozen update-280 checkpoint")
    if len(str(starting.get("sha256", ""))) != 64:
        raise ValueError("long-run starting-checkpoint digest is invalid")
    if starting.get("optimizer_state_required") is not True:
        raise ValueError("long-run must resume the optimizer state")

    training = payload.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("long-run protocol omits training settings")
    continuation_updates = int(training.get("continuation_updates", -1))
    final_update = int(training.get("final_global_update", -1))
    if (
        continuation_updates <= 0
        or starting_update + continuation_updates != final_update
    ):
        raise ValueError("long-run continuation length is inconsistent")
    checkpoints = tuple(int(value) for value in training.get("checkpoint_updates", ()))
    if (
        not checkpoints
        or tuple(sorted(set(checkpoints))) != checkpoints
        or checkpoints[-1] != final_update
        or any(update <= starting_update for update in checkpoints)
    ):
        raise ValueError("long-run checkpoint schedule is invalid")
    if int(training.get("batch_size", -1)) != 32:
        raise ValueError("long-run batch size must preserve the source run")
    if int(training.get("unroll_horizon_frames", -1)) != 50:
        raise ValueError("long-run horizon must preserve the source run")
    if int(training.get("continuation_random_seed", -1)) < 0:
        raise ValueError("long-run continuation seed is invalid")

    gate = payload.get("source_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("long-run protocol omits its source gate")
    reference = float(gate.get("published_reference_l1_m", math.nan))
    multiplier = float(gate.get("published_error_multiplier_max", math.nan))
    wins = int(gate.get("minimum_persistence_wins", -1))
    if (
        not math.isfinite(reference)
        or reference <= 0.0
        or not math.isfinite(multiplier)
        or multiplier <= 1.0
        or not 0 <= wins <= 8
    ):
        raise ValueError("long-run source gate is invalid")

    result = dict(payload)
    result["protocol_path"] = str(source)
    result["training"] = {**training, "checkpoint_updates": checkpoints}
    return result
