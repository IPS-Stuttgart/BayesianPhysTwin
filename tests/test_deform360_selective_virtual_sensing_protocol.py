import hashlib
import json
from pathlib import Path

import pytest

import bayesian_phystwin.deform360_selective_virtual_sensing_protocol as protocol


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "sota"
    / "deform360_selective_virtual_sensing_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_rehashed(tmp_path: Path, payload: dict) -> Path:
    payload["config_sha256"] = protocol.protocol_config_sha256(payload)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_registered_virtual_sensing_protocol_is_fresh_and_object_disjoint() -> None:
    loaded = protocol.load_selective_virtual_sensing_protocol(PROTOCOL_PATH)

    cohort = loaded["normalized_cohort"]
    objects = [object_id for records in cohort.values() for object_id in records]
    episodes = [episodes for records in cohort.values() for episodes in records.values()]
    assert len(objects) == len(set(objects)) == 12
    assert sum(len(value) for value in episodes) == 24
    assert not set(objects) & protocol.INELIGIBLE_OBJECTS
    assert all(len(records) == 4 for records in cohort.values())
    for records in cohort.values():
        for object_id, episode_ids in records.items():
            assert episode_ids == protocol.metadata_ranked_episode_ids(object_id)


def test_registered_protocol_pins_algorithm_and_information_boundary() -> None:
    loaded = protocol.load_selective_virtual_sensing_protocol(PROTOCOL_PATH)
    config = loaded["config"]

    assert config["method"]["primary_arm"] == (
        "persistence_full_blend_rbf_pairwise_clique"
    )
    assert config["method"]["insufficient_support_fallback"] == (
        "bit-exact persistence"
    )
    boundary = config["information_boundary"]
    assert boundary["predictions_hashed_before_target_open"] is True
    assert boundary["future_dense_reconstruction_allowed_before_prediction_seal"] is (
        False
    )
    assert boundary["existing_frame_zero_confirmation_remains_sealed"] is True

    source_hashes = config["source_provenance"]
    assert source_hashes["explicit_arm_algorithm_sha256"] == _sha256(
        REPOSITORY_ROOT
        / "src"
        / "bayesian_phystwin"
        / "deform360_raw_pairwise_correspondence_diagnostic.py"
    )
    assert source_hashes["pairwise_gate_sha256"] == _sha256(
        REPOSITORY_ROOT
        / "src"
        / "bayesian_phystwin"
        / "phystwin_correspondence_gate.py"
    )
    assert source_hashes["online_belief_sha256"] == _sha256(
        REPOSITORY_ROOT
        / "src"
        / "bayesian_phystwin"
        / "phystwin_online_belief.py"
    )


def test_protocol_rejects_unhashed_edit(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["config"]["observation"]["center_count"] = 32
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        protocol.load_selective_virtual_sensing_protocol(path)


def test_protocol_rejects_previously_accessed_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    record = payload["config"]["cohort"]["strata"]["filament"][0]
    record["object_id"] = "002-rope-silk"
    record["episode_ids"] = list(
        protocol.metadata_ranked_episode_ids("002-rope-silk")
    )
    path = _write_rehashed(tmp_path, payload)
    monkeypatch.setattr(protocol, "_CANONICAL_CONFIG_SHA256", "")

    with pytest.raises(ValueError, match="previously accessed or reserved"):
        protocol.load_selective_virtual_sensing_protocol(path)


def test_protocol_rejects_nonranked_episode_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["config"]["cohort"]["strata"]["sheet"][0]["episode_ids"] = [0, 1]
    path = _write_rehashed(tmp_path, payload)
    monkeypatch.setattr(protocol, "_CANONICAL_CONFIG_SHA256", "")

    with pytest.raises(ValueError, match="not metadata-ranked"):
        protocol.load_selective_virtual_sensing_protocol(path)
