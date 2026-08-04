from __future__ import annotations

from pathlib import Path

REPRODUCE = Path("reproductions/full22_anchor_v1/reproduce.py")
TEST = Path("tests/test_full22_reproduction_capsule.py")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one {label} occurrence, found {count}")
    return text.replace(old, new, 1)


source = REPRODUCE.read_text(encoding="utf-8")
source = replace_once(
    source,
    '''EXPECTED_PROTOCOL_ID = (\n    "ee11310a84b92ff2158018a13ef09989e641e7c0ea84733fe8a6abf267093c65"\n)''',
    '''EXPECTED_SOURCE_PROTOCOL_ID = (\n    "ee11310a84b92ff2158018a13ef09989e641e7c0ea84733fe8a6abf267093c65"\n)\nEXPECTED_PORTABLE_PROTOCOL_ID = (\n    "5d31b04c464478d839ac3919ec04209b5ff22c75cf7fe8de54a6d189a4802872"\n)''',
    label="protocol constants",
)
source = replace_once(
    source,
    '''def _json_write(path: Path, payload: object) -> None:\n''',
    '''def _portable_protocol_specification(\n    specification: Mapping[str, Any],\n    *,\n    data_manifest_identity: str,\n) -> dict[str, Any]:\n    """Remove the retrieval path while retaining every scientific choice."""\n\n    normalized = json.loads(\n        json.dumps(\n            dict(specification),\n            sort_keys=True,\n            separators=(",", ":"),\n            allow_nan=False,\n        )\n    )\n    data_manifest = normalized.get("data_manifest")\n    if not isinstance(data_manifest, dict):\n        raise ValueError("locked protocol has no data_manifest mapping")\n    path = data_manifest.get("path")\n    if not isinstance(path, str) or not path:\n        raise ValueError("locked protocol data-manifest path is missing")\n    selected_cases = data_manifest.get("selected_cases")\n    if (\n        not isinstance(selected_cases, list)\n        or len(selected_cases) != 22\n        or any(not isinstance(case, str) or not case for case in selected_cases)\n        or len(set(selected_cases)) != 22\n    ):\n        raise ValueError("locked protocol must bind 22 ordered selected cases")\n    if (\n        len(data_manifest_identity) != 64\n        or any(\n            character not in "0123456789abcdef"\n            for character in data_manifest_identity\n        )\n    ):\n        raise ValueError("data manifest identity must be a lowercase SHA-256 digest")\n    normalized["data_manifest"] = {\n        "identity_sha256": data_manifest_identity,\n        "selected_cases": selected_cases,\n    }\n    return normalized\n\n\ndef _portable_protocol_id(\n    specification: Mapping[str, Any],\n    *,\n    data_manifest_identity: str,\n) -> str:\n    return _canonical_sha256(\n        _portable_protocol_specification(\n            specification,\n            data_manifest_identity=data_manifest_identity,\n        )\n    )\n\n\ndef _json_write(path: Path, payload: object) -> None:\n''',
    label="portable protocol helpers",
)
source = replace_once(
    source,
    '''def verify_confirmation_summary(summary: Mapping[str, Any]) -> None:\n    protocol_id = summary.get("protocol_id")\n    if protocol_id != EXPECTED_PROTOCOL_ID:\n        raise ValueError(\n            "Bayesian anchor protocol ID changed: "\n            f"expected {EXPECTED_PROTOCOL_ID}, received {protocol_id}"\n        )\n    case_results = summary.get("case_results")\n    if not isinstance(case_results, Mapping) or len(case_results) != 22:\n        raise ValueError("Bayesian anchor confirmation must contain all 22 cases")\n''',
    '''def verify_confirmation_summary(\n    summary: Mapping[str, Any],\n    locked_protocol: Mapping[str, Any],\n    *,\n    data_manifest_identity: str,\n) -> str:\n    """Bind the source output to a path-normalized scientific protocol."""\n\n    protocol_id = summary.get("protocol_id")\n    locked_protocol_id = locked_protocol.get("protocol_id")\n    if protocol_id != locked_protocol_id:\n        raise ValueError(\n            "confirmation summary and locked protocol IDs disagree: "\n            f"{protocol_id!r} != {locked_protocol_id!r}"\n        )\n    specification = locked_protocol.get("specification")\n    if not isinstance(specification, Mapping):\n        raise ValueError("locked protocol specification is missing")\n    if locked_protocol_id != _canonical_sha256(specification):\n        raise ValueError("locked protocol raw ID does not bind its specification")\n    portable_protocol_id = _portable_protocol_id(\n        specification,\n        data_manifest_identity=data_manifest_identity,\n    )\n    if portable_protocol_id != EXPECTED_PORTABLE_PROTOCOL_ID:\n        raise ValueError(\n            "Bayesian anchor portable protocol ID changed: "\n            f"expected {EXPECTED_PORTABLE_PROTOCOL_ID}, "\n            f"received {portable_protocol_id}"\n        )\n    case_results = summary.get("case_results")\n    if not isinstance(case_results, Mapping) or len(case_results) != 22:\n        raise ValueError("Bayesian anchor confirmation must contain all 22 cases")\n    return portable_protocol_id\n''',
    label="confirmation verifier",
)
source = source.replace("EXPECTED_PROTOCOL_ID", "EXPECTED_PORTABLE_PROTOCOL_ID")
source = replace_once(
    source,
    '''    if expected.get("source_protocol_id") != EXPECTED_PORTABLE_PROTOCOL_ID:\n        raise ValueError("expected metrics protocol ID changed")\n''',
    '''    if expected.get("source_protocol_id") != EXPECTED_SOURCE_PROTOCOL_ID:\n        raise ValueError("expected metrics source protocol ID changed")\n''',
    label="expected source protocol check",
)
source = replace_once(
    source,
    '''    summary = json.loads(summary_path.read_text(encoding="utf-8"))\n    verify_confirmation_summary(summary)\n    _run(compare_command, cwd=source_checkout, env=source_env)\n''',
    '''    summary = json.loads(summary_path.read_text(encoding="utf-8"))\n    locked_protocol_path = run_dir / "locked_protocol.json"\n    locked_protocol = json.loads(\n        locked_protocol_path.read_text(encoding="utf-8")\n    )\n    portable_protocol_id = verify_confirmation_summary(\n        summary,\n        locked_protocol,\n        data_manifest_identity=data_identity,\n    )\n    _run(compare_command, cwd=source_checkout, env=source_env)\n''',
    label="runtime protocol verification call",
)
source = replace_once(
    source,
    '''            "source_revision": EXPECTED_SOURCE_REVISION,\n            "protocol_id": EXPECTED_PORTABLE_PROTOCOL_ID,\n            "maximum_residual_m": 0.01,\n''',
    '''            "source_revision": EXPECTED_SOURCE_REVISION,\n            "source_protocol_id": EXPECTED_SOURCE_PROTOCOL_ID,\n            "runtime_protocol_id": summary["protocol_id"],\n            "portable_protocol_id": portable_protocol_id,\n            "maximum_residual_m": 0.01,\n''',
    label="method lock protocol identities",
)
source = replace_once(
    source,
    '''            "data_manifest_identity_sha256": data_identity,\n            "metric_tolerance_m": expected["absolute_tolerance_m"],\n''',
    '''            "data_manifest_identity_sha256": data_identity,\n            "runtime_protocol_id": summary["protocol_id"],\n            "portable_protocol_id": portable_protocol_id,\n            "metric_tolerance_m": expected["absolute_tolerance_m"],\n''',
    label="configuration protocol identities",
)
REPRODUCE.write_text(source, encoding="utf-8")


test = TEST.read_text(encoding="utf-8")
test = replace_once(
    test,
    '''def test_confirmation_summary_requires_protocol_and_complete_cohort() -> None:\n    capsule = _load_capsule()\n    summary = {\n        "protocol_id": capsule.EXPECTED_PROTOCOL_ID,\n        "case_results": {f"case-{index:02d}": {} for index in range(22)},\n    }\n    capsule.verify_confirmation_summary(summary)\n\n    summary["protocol_id"] = "changed"\n    with pytest.raises(ValueError, match="protocol ID changed"):\n        capsule.verify_confirmation_summary(summary)\n''',
    '''def _locked_protocol(\n    capsule: ModuleType,\n    *,\n    manifest_path: str,\n) -> dict[str, object]:\n    cases = [f"case-{index:02d}" for index in range(22)]\n    specification = {\n        "method": "robust Bayesian random-walk endpoint anchoring",\n        "protocol": {\n            "bootstrap_block_length": 5,\n            "bootstrap_samples": 10000,\n            "bootstrap_seed": 20260710,\n            "development_cases": [\n                "single_lift_sloth",\n                "double_lift_sloth",\n                "double_stretch_sloth",\n            ],\n            "fit_fraction": 0.75,\n            "initial_std_m": 0.01,\n            "inlier_prior": 0.95,\n            "interpolation_neighbors": 4,\n            "maximum_residual_m": 0.01,\n            "minimum_validation_improvement": 0.0,\n            "observation_std_candidates_m": [0.001, 0.0025, 0.005],\n            "outlier_variance_multiplier": 100.0,\n            "process_std_candidates_m": [0.0, 0.0005, 0.001, 0.0025, 0.005],\n        },\n        "data_manifest": {\n            "path": manifest_path,\n            "selected_cases": cases,\n        },\n        "cohorts": {\n            "development": [\n                "double_lift_sloth",\n                "double_stretch_sloth",\n                "single_lift_sloth",\n            ],\n            "confirmation": [case for case in cases if "sloth" not in case],\n        },\n        "status": "exploratory extension after the deterministic confirmation",\n    }\n    return {\n        "schema_version": 1,\n        "protocol_id": capsule._canonical_sha256(specification),\n        "specification": specification,\n    }\n\n\ndef test_confirmation_summary_uses_path_normalized_protocol_identity(\n    monkeypatch: pytest.MonkeyPatch,\n) -> None:\n    capsule = _load_capsule()\n    first = _locked_protocol(capsule, manifest_path="/cache-a/manifest.json")\n    second = _locked_protocol(capsule, manifest_path="/cache-b/manifest.json")\n    first_portable = capsule._portable_protocol_id(\n        first["specification"],\n        data_manifest_identity=capsule.EXPECTED_DATA_MANIFEST_IDENTITY_SHA256,\n    )\n    second_portable = capsule._portable_protocol_id(\n        second["specification"],\n        data_manifest_identity=capsule.EXPECTED_DATA_MANIFEST_IDENTITY_SHA256,\n    )\n    assert first_portable == second_portable\n    monkeypatch.setattr(capsule, "EXPECTED_PORTABLE_PROTOCOL_ID", first_portable)\n\n    summary = {\n        "protocol_id": first["protocol_id"],\n        "case_results": {f"case-{index:02d}": {} for index in range(22)},\n    }\n    assert (\n        capsule.verify_confirmation_summary(\n            summary,\n            first,\n            data_manifest_identity=capsule.EXPECTED_DATA_MANIFEST_IDENTITY_SHA256,\n        )\n        == first_portable\n    )\n\n    summary["protocol_id"] = second["protocol_id"]\n    with pytest.raises(ValueError, match="summary and locked protocol IDs disagree"):\n        capsule.verify_confirmation_summary(\n            summary,\n            first,\n            data_manifest_identity=capsule.EXPECTED_DATA_MANIFEST_IDENTITY_SHA256,\n        )\n\n\ndef test_confirmation_summary_rejects_portable_protocol_drift(\n    monkeypatch: pytest.MonkeyPatch,\n) -> None:\n    capsule = _load_capsule()\n    locked = _locked_protocol(capsule, manifest_path="/cache/manifest.json")\n    portable = capsule._portable_protocol_id(\n        locked["specification"],\n        data_manifest_identity=capsule.EXPECTED_DATA_MANIFEST_IDENTITY_SHA256,\n    )\n    monkeypatch.setattr(capsule, "EXPECTED_PORTABLE_PROTOCOL_ID", portable)\n    summary = {\n        "protocol_id": locked["protocol_id"],\n        "case_results": {f"case-{index:02d}": {} for index in range(22)},\n    }\n    changed = json.loads(json.dumps(locked))\n    changed["specification"]["protocol"]["fit_fraction"] = 0.5\n    changed["protocol_id"] = capsule._canonical_sha256(changed["specification"])\n    summary["protocol_id"] = changed["protocol_id"]\n    with pytest.raises(ValueError, match="portable protocol ID changed"):\n        capsule.verify_confirmation_summary(\n            summary,\n            changed,\n            data_manifest_identity=capsule.EXPECTED_DATA_MANIFEST_IDENTITY_SHA256,\n        )\n''',
    label="protocol tests",
)
test = replace_once(
    test,
    '''    assert capsule.EXPECTED_PROTOCOL_ID in command\n''',
    '''    assert capsule.EXPECTED_PORTABLE_PROTOCOL_ID in command\n    assert capsule.EXPECTED_SOURCE_PROTOCOL_ID not in command\n''',
    label="manifest command protocol assertion",
)
TEST.write_text(test, encoding="utf-8")
