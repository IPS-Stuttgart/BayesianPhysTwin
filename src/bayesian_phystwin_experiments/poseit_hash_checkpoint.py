"""Prospective, local-only SHA-256 checkpoints; no acquisition authorization."""

from __future__ import annotations

import ctypes
import hashlib
import platform
import sys
from pathlib import Path
from typing import Any

from bayesian_phystwin._portable_contracts import content_id

RHASH_COMMIT = "6562de382954d9893442b89b0e8b5c513eea6a88"
RHASH_VERSION = 0x01040600
SHA256_ID = 0x20000
BLOCK_SIZE = 64
_MAX_BYTES = (1 << 61) - 1
_HEADER_SIZE = 20


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class _PublicContext(ctypes.Structure):
    _fields_ = [("msg_size", ctypes.c_uint64), ("hash_mask", ctypes.c_uint64)]


class _ShaContext(ctypes.Structure):
    # The pinned, non-OpenSSL v1.4.6 layout from librhash/sha256.h.
    _fields_ = [
        ("message", ctypes.c_uint32 * 16),
        ("length", ctypes.c_uint64),
        ("hash", ctypes.c_uint32 * 8),
        ("digest_length", ctypes.c_uint32),
    ]


def _canonical_context(state: _ShaContext) -> bytes:
    _require(not any(state.message), "checkpoint would retain an input buffer")
    _require(state.digest_length == 32, "SHA-256 digest size changed")
    _require(state.length % BLOCK_SIZE == 0, "checkpoint is not block-aligned")
    clean = _ShaContext()
    clean.length = state.length
    clean.hash[:] = state.hash[:]
    clean.digest_length = state.digest_length
    return bytes(clean)


class RHashCheckpointEngine:
    """Use an explicitly hash-bound native build, never a discovered system library."""

    def __init__(self, library_path: Path, *, expected_library_sha256: str) -> None:
        _require(
            len(expected_library_sha256) == 64,
            "expected library SHA-256 is malformed",
        )
        library_path = library_path.resolve(strict=True)
        digest = hashlib.sha256(library_path.read_bytes()).hexdigest()
        _require(digest == expected_library_sha256, "native library SHA-256 changed")
        _require(
            sys.byteorder == "little"
            and platform.machine() == "x86_64"
            and ctypes.sizeof(ctypes.c_void_p) == 8
            and ctypes.sizeof(_PublicContext) == 16
            and ctypes.sizeof(_ShaContext) == 112,
            "unsupported native checkpoint ABI",
        )
        self.library_sha256 = digest
        self._lib = ctypes.CDLL(str(library_path), use_errno=True)
        for name, result, arguments in (
            ("rhash_library_init", None, []),
            ("rhash_init", ctypes.c_void_p, [ctypes.c_uint]),
            ("rhash_free", None, [ctypes.c_void_p]),
            (
                "rhash_update",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t],
            ),
            ("rhash_final", ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]),
            (
                "rhash_export",
                ctypes.c_size_t,
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t],
            ),
            ("rhash_import", ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_size_t]),
            (
                "rhash_ctrl",
                ctypes.c_size_t,
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p],
            ),
        ):
            function = getattr(self._lib, name)
            function.restype = result
            function.argtypes = arguments
        self._lib.rhash_library_init()
        _require(
            self._lib.rhash_ctrl(None, 20, 0, None) == RHASH_VERSION,
            "RHash version changed",
        )
        _require(
            self._lib.rhash_ctrl(None, 18, 0, None) == 0,
            "OpenSSL-backed contexts are not admitted",
        )
        context = self._new()
        try:
            self._template = self._raw_export(context)
        finally:
            self._lib.rhash_free(context)
        _require(
            len(self._template) == _HEADER_SIZE + ctypes.sizeof(_ShaContext),
            "native export layout changed",
        )

    def _new(self) -> int:
        context = self._lib.rhash_init(SHA256_ID)
        _require(bool(context), "could not initialize SHA-256")
        return int(context)

    def _raw_export(self, context: int) -> bytes:
        size = self._lib.rhash_export(context, None, 0)
        _require(
            size == _HEADER_SIZE + ctypes.sizeof(_ShaContext),
            "native export size changed",
        )
        buffer = ctypes.create_string_buffer(size)
        _require(
            self._lib.rhash_export(context, buffer, size) == size,
            "native export failed",
        )
        return buffer.raw

    def _validate_blob(self, blob: bytes, byte_count: int) -> None:
        _require(len(blob) == len(self._template), "checkpoint byte size changed")
        _require(blob[:8] == self._template[:8], "checkpoint state or flags changed")
        _require(blob[16:20] == self._template[16:20], "checkpoint algorithm changed")
        _require(
            int.from_bytes(blob[8:16], "little") == byte_count,
            "checkpoint public byte count changed",
        )
        state = _ShaContext.from_buffer_copy(blob[_HEADER_SIZE:])
        _require(state.length == byte_count, "checkpoint SHA byte count changed")
        _require(
            blob[_HEADER_SIZE:] == _canonical_context(state),
            "checkpoint native padding is not canonical",
        )

    def _canonical_export(self, context: int, byte_count: int) -> bytes:
        blob = self._raw_export(context)
        state = _ShaContext.from_buffer_copy(blob[_HEADER_SIZE:])
        canonical = blob[:_HEADER_SIZE] + _canonical_context(state)
        self._validate_blob(canonical, byte_count)
        return canonical

    def _import(self, blob: bytes) -> int:
        buffer = ctypes.create_string_buffer(blob, len(blob))
        context = self._lib.rhash_import(buffer, len(blob))
        _require(bool(context), "native checkpoint import failed")
        return int(context)

    def _update(self, context: int, data: bytes) -> None:
        if not data:
            return
        # Aligned, complete blocks bypass the native leftover-input buffer.
        buffer = ctypes.create_string_buffer(len(data) + BLOCK_SIZE - 1)
        address = (ctypes.addressof(buffer) + BLOCK_SIZE - 1) & -BLOCK_SIZE
        ctypes.memmove(address, data, len(data))
        _require(
            self._lib.rhash_update(context, address, len(data)) == 0,
            "native update failed",
        )

    def _digest(self, blob: bytes, tail: bytes = b"") -> str:
        context = self._import(blob)
        try:
            self._update(context, tail)
            output = ctypes.create_string_buffer(32)
            _require(
                self._lib.rhash_final(context, output) == 0,
                "native finalization failed",
            )
            return output.raw.hex()
        finally:
            self._lib.rhash_free(context)

    def new(self) -> CheckpointedSha256:
        return CheckpointedSha256(self, self._new())

    def restore(
        self, checkpoint: dict[str, Any], *, expected_checkpoint_id: str
    ) -> CheckpointedSha256:
        identity = dict(checkpoint)
        checkpoint_id = identity.pop("checkpoint_id", None)
        _require(
            checkpoint_id == expected_checkpoint_id == content_id(identity),
            "checkpoint content binding changed",
        )
        _require(
            set(identity)
            == {
                "schema",
                "schema_version",
                "library_sha256",
                "rhash_source_commit",
                "abi",
                "bytes_hashed",
                "state_hex",
                "prefix_sha256",
                "input_bytes_retained",
                "acquisition_receipt",
                "execution_authorized",
            },
            "checkpoint fields changed",
        )
        _require(
            identity["schema"] == "bayesian-phystwin.poseit-sha256-checkpoint",
            "checkpoint schema changed",
        )
        _require(
            type(identity["schema_version"]) is int and identity["schema_version"] == 1,
            "checkpoint version changed",
        )
        _require(
            identity["library_sha256"] == self.library_sha256,
            "checkpoint library changed",
        )
        _require(
            identity["rhash_source_commit"] == RHASH_COMMIT, "checkpoint source changed"
        )
        _require(
            identity["abi"] == "rhash-1.4.6-native-sha256-linux-x86_64-le",
            "checkpoint ABI changed",
        )
        _require(
            identity["input_bytes_retained"] is False,
            "checkpoint retention flag changed",
        )
        _require(
            identity["acquisition_receipt"] is False,
            "checkpoint is not an acquisition receipt",
        )
        _require(
            identity["execution_authorized"] is False,
            "checkpoint cannot authorize an execution",
        )
        byte_count = identity["bytes_hashed"]
        _require(
            type(byte_count) is int
            and 0 <= byte_count <= _MAX_BYTES
            and byte_count % BLOCK_SIZE == 0,
            "checkpoint byte count is invalid",
        )
        state_hex = identity["state_hex"]
        _require(
            type(state_hex) is str and len(state_hex) == 2 * len(self._template),
            "checkpoint state is malformed",
        )
        blob = bytes.fromhex(state_hex)
        _require(blob.hex() == state_hex, "checkpoint state encoding is not canonical")
        self._validate_blob(blob, byte_count)
        _require(
            self._digest(blob) == identity["prefix_sha256"],
            "checkpoint prefix digest changed",
        )
        return CheckpointedSha256(self, self._import(blob))


class CheckpointedSha256:
    def __init__(self, engine: RHashCheckpointEngine, context: int) -> None:
        self._engine = engine
        self._context: int | None = context

    def __enter__(self) -> CheckpointedSha256:
        self._live()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _live(self) -> int:
        _require(self._context is not None, "SHA-256 state is closed")
        assert self._context is not None
        return self._context

    @property
    def bytes_hashed(self) -> int:
        state = ctypes.cast(self._live(), ctypes.POINTER(_PublicContext)).contents
        _require(state.hash_mask == SHA256_ID, "native hash algorithm changed")
        return int(state.msg_size)

    def update_blocks(self, data: bytes) -> None:
        _require(
            type(data) is bytes and len(data) % BLOCK_SIZE == 0,
            "update must contain complete SHA-256 blocks",
        )
        _require(
            self.bytes_hashed + len(data) <= _MAX_BYTES,
            "SHA-256 length exceeds its limit",
        )
        self._engine._update(self._live(), data)

    def hexdigest(self, tail: bytes = b"") -> str:
        _require(
            type(tail) is bytes and len(tail) < BLOCK_SIZE,
            "final tail must be shorter than one block",
        )
        _require(
            self.bytes_hashed + len(tail) <= _MAX_BYTES,
            "SHA-256 length exceeds its limit",
        )
        blob = self._engine._canonical_export(self._live(), self.bytes_hashed)
        return self._engine._digest(blob, tail)

    def checkpoint(self) -> dict[str, Any]:
        blob = self._engine._canonical_export(self._live(), self.bytes_hashed)
        identity = {
            "schema": "bayesian-phystwin.poseit-sha256-checkpoint",
            "schema_version": 1,
            "library_sha256": self._engine.library_sha256,
            "rhash_source_commit": RHASH_COMMIT,
            "abi": "rhash-1.4.6-native-sha256-linux-x86_64-le",
            "bytes_hashed": self.bytes_hashed,
            "state_hex": blob.hex(),
            "prefix_sha256": self._engine._digest(blob),
            "input_bytes_retained": False,
            "acquisition_receipt": False,
            "execution_authorized": False,
        }
        return {**identity, "checkpoint_id": content_id(identity)}

    def close(self) -> None:
        if self._context is not None:
            self._engine._lib.rhash_free(self._context)
            self._context = None
