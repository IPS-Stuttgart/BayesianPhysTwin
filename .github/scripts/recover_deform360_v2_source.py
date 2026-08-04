from __future__ import annotations

import base64
import hashlib
import io
import sys
import tarfile
import zlib
from pathlib import Path, PurePosixPath

_COMMITTED_PREFIX_SHA256 = (
    "17c2d0749f2de3c925cf5d94d136a8f6076f67e15c5dd41e38a51fac976d66e2"
)
_ALLOWED = {
    ".github/workflows/deform360-group-evaluation.yml",
    ".github/workflows/deform360-public-evaluation.yml",
    "docs/deform360_group_evaluation_v2.md",
    "scripts/science/prepare_deform360_archive_manifest.py",
    "scripts/science/prepare_deform360_name_inventory.py",
    "scripts/science/prepare_deform360_public_protocol.py",
    "scripts/science/run_deform360_group_evaluation.py",
    "tests/test_deform360_archive_manifest.py",
    "tests/test_deform360_group_evaluation.py",
    "tests/test_deform360_name_inventory.py",
    "tests/test_deform360_preflight_workflow.py",
    "tests/test_deform360_public_protocol.py",
}


def _decode_locked_prefix() -> bytes:
    parts = tuple(
        Path(f"tools/deform360-v2-bundle-{index:02d}.b64")
        for index in range(3)
    )
    if any(not path.is_file() for path in parts):
        raise SystemExit("one or more locked source-bundle chunks are missing")
    encoded = "".join(
        "".join(path.read_text(encoding="ascii").split()) for path in parts
    )
    encoded += "=" * ((-len(encoded)) % 4)
    committed = base64.b64decode(encoded, validate=True)
    digest = hashlib.sha256(committed).hexdigest()
    print(f"locked compressed prefix bytes={len(committed)} sha256={digest}")
    if digest != _COMMITTED_PREFIX_SHA256:
        raise SystemExit("committed source-bundle prefix changed")
    return committed


def _decompress_locked_prefix(committed: bytes) -> bytes:
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    unpacked = decoder.decompress(committed)
    unpacked += decoder.flush()
    if not unpacked:
        raise SystemExit("locked compressed prefix yielded no tar bytes")
    print(
        f"locked tar prefix bytes={len(unpacked)} "
        f"sha256={hashlib.sha256(unpacked).hexdigest()} "
        f"gzip_stream_complete={decoder.eof}"
    )
    return unpacked


def _relative_product_path(member_name: str) -> str:
    raw = PurePosixPath(member_name)
    if raw.is_absolute() or ".." in raw.parts:
        raise SystemExit(f"unsafe archive path: {member_name}")
    if len(raw.parts) < 2:
        raise SystemExit(f"unexpected top-level archive member: {member_name}")
    return PurePosixPath(*raw.parts[1:]).as_posix()


def _extract_products(unpacked: bytes, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    terminal_error: tarfile.TarError | None = None
    try:
        with tarfile.open(fileobj=io.BytesIO(unpacked), mode="r|") as archive:
            for member in archive:
                if member.issym() or member.islnk() or member.isdev():
                    raise SystemExit(
                        f"unsafe archive member type: {member.name}"
                    )
                if member.isdir():
                    continue
                if not member.isfile():
                    raise SystemExit(f"unexpected archive member: {member.name}")
                relative = _relative_product_path(member.name)
                if relative not in _ALLOWED or relative in found:
                    raise SystemExit(
                        f"unexpected or duplicate product file: {relative}"
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise SystemExit(f"cannot read archive member: {relative}")
                data = stream.read()
                if len(data) != member.size:
                    raise SystemExit(f"short archive member: {relative}")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                found.add(relative)
                print(
                    f"product {relative} bytes={len(data)} "
                    f"sha256={hashlib.sha256(data).hexdigest()}"
                )
    except tarfile.TarError as error:
        terminal_error = error

    if found != _ALLOWED:
        raise SystemExit(
            "locked tar prefix does not contain the exact product set: "
            f"missing={sorted(_ALLOWED - found)}, "
            f"extra={sorted(found - _ALLOWED)}, "
            f"terminal_error={terminal_error!r}"
        )
    if terminal_error is not None:
        print(
            "accepted terminal tar truncation only after every declared member "
            f"was read completely: {terminal_error}"
        )
    print(f"verified exact product file count: {len(found)}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: recover_deform360_v2_source.py OUTPUT_ROOT")
    committed = _decode_locked_prefix()
    unpacked = _decompress_locked_prefix(committed)
    _extract_products(unpacked, Path(sys.argv[1]))


if __name__ == "__main__":
    main()
