"""One-shot claim-boundary correction for PR #187; deleted by its publisher."""

from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one replacement target in {path}, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "src/bayesian_phystwin/prob4d_observation_timestamp_admission.py",
    "checks the raw\nsource identity against a separate verification artifact, and rejects concurrent\n",
    "checks the raw\nsource identity against a separately supplied verification-artifact ID, and rejects\nconcurrent\n",
)
replace_once(
    "src/bayesian_phystwin/prob4d_observation_timestamp_admission.py",
    '        "prob4d_timestamp_source_independently_verified",\n',
    "",
)
replace_once(
    "src/bayesian_phystwin/prob4d_observation_timestamp_admission.py",
    "    ``timestamp_source_verification_artifact_id`` must come from an independently\n    frozen source/calibration manifest, not from the timestamp sidecar being\n    admitted. The verification artifact must be distinct from the sidecar itself.\n",
    "    ``timestamp_source_verification_artifact_id`` must come from a separately\n    frozen upstream source/calibration manifest, not from the timestamp sidecar\n    being admitted. This function binds that artifact's content ID but does not\n    open it or establish the verifier's competence. The verification artifact\n    must be distinct from the raw source, bundle, and sidecar identities.\n",
)
replace_once(
    "src/bayesian_phystwin/prob4d_observation_timestamp_admission.py",
    "    expected_bundle = sha256_digest(\n"
    "        expected_bundle_manifest_sha256,\n"
    "        name=\"expected_bundle_manifest_sha256\",\n"
    "    )\n"
    "    timestamp_path, timestamp_bytes, timestamp_sha_before = _ordinary_snapshot(\n",
    "    expected_bundle = sha256_digest(\n"
    "        expected_bundle_manifest_sha256,\n"
    "        name=\"expected_bundle_manifest_sha256\",\n"
    "    )\n"
    "    if verification_id in {expected_source, expected_bundle}:\n"
    "        raise ValueError(\n"
    "            \"timestamp verification artifact must be distinct from source and bundle\"\n"
    "        )\n"
    "    timestamp_path, timestamp_bytes, timestamp_sha_before = _ordinary_snapshot(\n",
)
replace_once(
    "src/bayesian_phystwin/prob4d_observation_timestamp_admission.py",
    "    if bundle_sha_before != expected_bundle:\n"
    "        raise ValueError(\"Prob4D bundle manifest checksum mismatch\")\n\n"
    "    with tempfile.TemporaryDirectory(\n",
    "    if bundle_sha_before != expected_bundle:\n"
    "        raise ValueError(\"Prob4D bundle manifest checksum mismatch\")\n"
    "    if verification_id in {timestamp_sha_before, bundle_sha_before}:\n"
    "        raise ValueError(\n"
    "            \"timestamp verification artifact must be distinct from admitted files\"\n"
    "        )\n\n"
    "    with tempfile.TemporaryDirectory(\n",
)
replace_once(
    "src/bayesian_phystwin/prob4d_observation_timestamp_admission.py",
    "                \"Prob4D timestamp source artifact differs from independent evidence\"\n",
    "                \"Prob4D timestamp source artifact differs from separately frozen evidence\"\n",
)
replace_once(
    "src/bayesian_phystwin/prob4d_observation_timestamp_admission.py",
    '            "prob4d_timestamp_source_independently_verified": True,\n',
    "",
)

replace_once(
    "tests/test_prob4d_observation_timestamp_admission.py",
    "def test_admission_binds_independent_source_and_private_exact_snapshots(\n",
    "def test_admission_binds_separately_frozen_source_and_private_snapshots(\n",
)
replace_once(
    "tests/test_prob4d_observation_timestamp_admission.py",
    '    assert admitted["prob4d_timestamp_source_independently_verified"] is True\n',
    "",
)
replace_once(
    "tests/test_prob4d_observation_timestamp_admission.py",
    "def test_wrong_independent_source_digest_fails_before_binding(\n",
    "def test_wrong_separately_frozen_source_digest_fails_before_binding(\n",
)
replace_once(
    "tests/test_prob4d_observation_timestamp_admission.py",
    '    with pytest.raises(ValueError, match="independent evidence"):\n',
    '    with pytest.raises(ValueError, match="separately frozen evidence"):\n',
)
replace_once(
    "tests/test_prob4d_observation_timestamp_admission.py",
    "\n\ndef test_sidecar_cannot_verify_itself(\n",
    '''


def test_verification_artifact_is_distinct_from_source_bundle_and_files(
    tmp_path: Path,
) -> None:
    path = tmp_path / "timestamps.json"
    sidecar_bytes = b"separate sidecar bytes"
    path.write_bytes(sidecar_bytes)

    with pytest.raises(ValueError, match="distinct from source and bundle"):
        _call(path, verification_id=SOURCE_SHA)
    with pytest.raises(ValueError, match="distinct from source and bundle"):
        _call(path, verification_id=BUNDLE_SHA)
    with pytest.raises(ValueError, match="distinct from admitted files"):
        _call(
            path,
            verification_id=hashlib.sha256(sidecar_bytes).hexdigest(),
        )


def test_sidecar_cannot_verify_itself(
''',
)

replace_once(
    "docs/prob4d_observation_timestamps.md",
    "the raw timestamp-source digest, its independent verification artifact, the\n",
    "the raw timestamp-source digest, its separately frozen verification-artifact ID, the\n",
)
replace_once(
    "docs/prob4d_observation_timestamps.md",
    "factor-to-row mapping, independently evidenced source-byte admission, and\n",
    "factor-to-row mapping, separately bound source-byte evidence, and\n",
)
