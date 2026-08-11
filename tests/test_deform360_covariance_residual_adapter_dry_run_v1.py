from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "science"
    / "run_deform360_covariance_residual_adapter_dry_run_v1.py"
)
PROTOCOL = (
    ROOT
    / "protocols"
    / "locks"
    / "deform360_covariance_residual_adapter_dry_run_v1.json"
)


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("residual_adapter_dry_run", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_locked_dry_run_passes_all_source_only_gates() -> None:
    module = _module()
    protocol = module._load_protocol(PROTOCOL)
    first = module.run_dry_run(protocol)
    second = module.run_dry_run(protocol)

    assert first == second
    assert first["dry_run_passed"] is True
    boundary = first["information_boundary"]
    assert boundary["source_only"] is True
    for field in (
        "camera_media_decoded",
        "sensor_arrays_loaded_from_target",
        "target_payload_opened",
        "target_outcomes_opened",
        "predictions_run_on_target",
        "claim_authorized",
    ):
        assert boundary[field] is False
    gates = first["gates"]
    for field in (
        "validity_hash_preserved",
        "masked_values_do_not_change_output",
        "minimum_support_enforced",
        "candidate_mean_is_reference_object",
        "unsupported_tracks_are_byte_identical_to_physical_fallback",
        "provider_failure_is_exact_zero_covariance_fallback",
        "insufficient_support_is_exact_physical_fallback",
        "output_covariance_is_psd",
        "horizon_scales_are_exact",
        "camera_sets_are_disjoint",
        "artifact_sets_are_disjoint",
    ):
        assert gates[field] is True


def test_cli_writes_a_content_addressed_result(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "result.json"

    assert module.main(["--protocol", str(PROTOCOL), "--output", str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    declared = result.pop("result_sha256")
    assert declared == module._canonical_sha256(result)
    assert result["dry_run_passed"] is True
    with pytest.raises(FileExistsError):
        module.main(["--protocol", str(PROTOCOL), "--output", str(output)])


def test_protocol_tampering_fails_before_dry_run(tmp_path: Path) -> None:
    module = _module()
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["method"]["minimum_valid_observations_per_track"] = 2
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 changed"):
        module._load_protocol(path)
