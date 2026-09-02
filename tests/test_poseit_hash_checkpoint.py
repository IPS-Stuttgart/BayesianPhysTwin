from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin_experiments.poseit_hash_checkpoint import (
    RHashCheckpointEngine,
)


@pytest.fixture(scope="module")
def engine() -> RHashCheckpointEngine:
    configured = os.environ.get("POSEIT_TEST_RHASH_LIBRARY")
    if not configured:
        pytest.skip("requires an explicitly built local RHash v1.4.6 test library")
    library = Path(configured)
    return RHashCheckpointEngine(
        library,
        expected_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
    )


def _payload(size: int) -> bytes:
    return random.Random(20260903 + size).randbytes(size)


def _repin(checkpoint: dict[str, Any]) -> dict[str, Any]:
    identity = dict(checkpoint)
    identity.pop("checkpoint_id")
    return {**identity, "checkpoint_id": content_id(identity)}


@pytest.mark.parametrize(
    "size", [0, 1, 55, 56, 63, 64, 65, 127, 128, 129, 4096, 1048603]
)
def test_checkpoint_matches_hashlib_with_every_final_tail_shape(
    engine: RHashCheckpointEngine, size: int
) -> None:
    data = _payload(size)
    boundary = size - size % 64
    with engine.new() as state:
        state.update_blocks(data[:boundary])
        checkpoint = state.checkpoint()
        assert checkpoint["bytes_hashed"] == boundary
        assert (
            checkpoint["prefix_sha256"] == hashlib.sha256(data[:boundary]).hexdigest()
        )
        assert state.hexdigest(data[boundary:]) == hashlib.sha256(data).hexdigest()
        assert state.bytes_hashed == boundary
    with engine.restore(
        json.loads(json.dumps(checkpoint)),
        expected_checkpoint_id=checkpoint["checkpoint_id"],
    ) as restored:
        assert restored.hexdigest(data[boundary:]) == hashlib.sha256(data).hexdigest()


def test_many_restarts_preserve_one_ordered_hash(engine: RHashCheckpointEngine) -> None:
    data = _payload(131079)
    processed = 0
    state = engine.new()
    try:
        for length in (64, 4096, 128, 32768, 256, 65536):
            state.update_blocks(data[processed : processed + length])
            processed += length
            checkpoint = state.checkpoint()
            assert state.hexdigest() == hashlib.sha256(data[:processed]).hexdigest()
            state.close()
            state = engine.restore(
                checkpoint, expected_checkpoint_id=checkpoint["checkpoint_id"]
            )
        aligned_end = len(data) - len(data) % 64
        state.update_blocks(data[processed:aligned_end])
        assert state.hexdigest(data[aligned_end:]) == hashlib.sha256(data).hexdigest()
    finally:
        state.close()


def test_checkpoint_contains_no_pending_input_or_native_padding(
    engine: RHashCheckpointEngine,
) -> None:
    data = _payload(32768)
    with engine.new() as state:
        state.update_blocks(data)
        checkpoint = state.checkpoint()
    native = bytes.fromhex(checkpoint["state_hex"])
    assert len(native) == 132
    assert native[20:84] == bytes(64)
    assert native[128:132] == bytes(4)
    assert checkpoint["input_bytes_retained"] is False
    assert checkpoint["acquisition_receipt"] is False
    assert checkpoint["execution_authorized"] is False


def test_rejects_partial_updates_and_long_tails(engine: RHashCheckpointEngine) -> None:
    with engine.new() as state:
        with pytest.raises(ValueError, match="complete SHA-256 blocks"):
            state.update_blocks(b"not a block")
        with pytest.raises(ValueError, match="shorter than one block"):
            state.hexdigest(bytes(64))
        assert state.bytes_hashed == 0


def test_rejects_library_change_before_native_loading(tmp_path: Path) -> None:
    path = tmp_path / "not-a-library.so"
    path.write_bytes(b"not a native library")
    with pytest.raises(ValueError, match="library SHA-256 changed"):
        RHashCheckpointEngine(path, expected_library_sha256="a" * 64)


def test_rejects_rebound_checkpoint_id(engine: RHashCheckpointEngine) -> None:
    with engine.new() as state:
        checkpoint = state.checkpoint()
    checkpoint["bytes_hashed"] = 64
    with pytest.raises(ValueError, match="content binding"):
        engine.restore(checkpoint, expected_checkpoint_id=checkpoint["checkpoint_id"])
    with pytest.raises(ValueError, match="content binding"):
        engine.restore(
            _repin(checkpoint), expected_checkpoint_id=checkpoint["checkpoint_id"]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "another schema"),
        ("schema_version", 2),
        ("schema_version", True),
        ("unexpected_field", "not admitted"),
        ("library_sha256", "0" * 64),
        ("rhash_source_commit", "0" * 40),
        ("abi", "another ABI"),
        ("input_bytes_retained", True),
        ("acquisition_receipt", True),
        ("execution_authorized", True),
        ("bytes_hashed", True),
        ("bytes_hashed", -64),
        ("bytes_hashed", 1),
        ("bytes_hashed", 1 << 61),
        ("prefix_sha256", "0" * 64),
        ("state_hex", "00"),
    ],
)
def test_rejects_invalid_checkpoint_metadata(
    engine: RHashCheckpointEngine, field: str, value: Any
) -> None:
    with engine.new() as state:
        checkpoint = state.checkpoint()
    checkpoint[field] = value
    checkpoint = _repin(checkpoint)
    with pytest.raises(ValueError):
        engine.restore(checkpoint, expected_checkpoint_id=checkpoint["checkpoint_id"])


@pytest.mark.parametrize("offset", [0, 4, 6, 8, 16, 20, 84, 92, 124, 128])
def test_rejects_corrupt_native_state_before_it_can_supply_a_digest(
    engine: RHashCheckpointEngine, offset: int
) -> None:
    with engine.new() as state:
        state.update_blocks(bytes(128))
        checkpoint = copy.deepcopy(state.checkpoint())
    blob = bytearray.fromhex(checkpoint["state_hex"])
    blob[offset] ^= 1
    checkpoint["state_hex"] = blob.hex()
    checkpoint = _repin(checkpoint)
    with pytest.raises(ValueError):
        engine.restore(checkpoint, expected_checkpoint_id=checkpoint["checkpoint_id"])


def test_finalizing_copy_does_not_finalize_live_state(
    engine: RHashCheckpointEngine,
) -> None:
    data = _payload(128)
    state = engine.new()
    state.update_blocks(data[:64])
    assert state.hexdigest(b"tail") == hashlib.sha256(data[:64] + b"tail").hexdigest()
    state.update_blocks(data[64:])
    assert state.hexdigest() == hashlib.sha256(data).hexdigest()
    state.close()
    state.close()
    with pytest.raises(ValueError, match="closed"):
        state.checkpoint()


def test_checkpoint_is_deterministic_across_block_partitions(
    engine: RHashCheckpointEngine,
) -> None:
    data = _payload(16384)
    with engine.new() as one, engine.new() as many:
        one.update_blocks(data)
        for offset in range(0, len(data), 64):
            many.update_blocks(data[offset : offset + 64])
        assert one.checkpoint() == many.checkpoint()


def test_registered_chunk_size_matches_hashlib(engine: RHashCheckpointEngine) -> None:
    data = _payload(33554432)
    with engine.new() as state:
        state.update_blocks(data)
        checkpoint = state.checkpoint()
    with engine.restore(
        checkpoint, expected_checkpoint_id=checkpoint["checkpoint_id"]
    ) as restored:
        assert restored.bytes_hashed == len(data)
        assert restored.hexdigest() == hashlib.sha256(data).hexdigest()


def test_native_counter_survives_restart_across_four_gibibytes(
    engine: RHashCheckpointEngine,
) -> None:
    block = _payload(1048576)
    reference = hashlib.sha256()
    with engine.new() as state:
        for _ in range(4095):
            state.update_blocks(block)
            reference.update(block)
        checkpoint = state.checkpoint()
    with engine.restore(
        checkpoint, expected_checkpoint_id=checkpoint["checkpoint_id"]
    ) as restored:
        restored.update_blocks(block)
        restored.update_blocks(block)
        reference.update(block)
        reference.update(block)
        assert restored.bytes_hashed == 4097 * len(block)
        assert restored.checkpoint()["prefix_sha256"] == reference.hexdigest()
