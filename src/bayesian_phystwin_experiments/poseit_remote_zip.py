"""Structure-only parsing for a remote PoseIt ZIP archive."""

from __future__ import annotations

import hashlib
import stat
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, cast

from bayesian_phystwin_experiments.poseit_remote_archive import (
    RangeOpener,
    RemoteArchiveExpectation,
    fetch_exact_range,
)

_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
_ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_CENTRAL_SIGNATURE = b"PK\x01\x02"
_DIGITAL_SIGNATURE = b"PK\x05\x05"
_EOCD = struct.Struct("<4s4H2LH")
_ZIP64_LOCATOR = struct.Struct("<4sLQL")
_ZIP64_EOCD = struct.Struct("<4sQ2H2L4Q")
_CENTRAL = struct.Struct("<4s6H3L5H2L")
_EOCD_SEARCH_BYTES = 65_557
_UINT16_MAX = (1 << 16) - 1
_UINT32_MAX = (1 << 32) - 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class RemoteZipLayout:
    archive_size_bytes: int
    central_directory_offset: int
    central_directory_size: int
    entry_count: int
    eocd_offset: int
    archive_comment: bytes
    zip64: bool
    zip64_eocd_offset: int | None


@dataclass(frozen=True)
class RemoteZipMember:
    name: str
    crc32: int
    compressed_size: int
    uncompressed_size: int
    compression_method: int
    flag_bits: int
    external_attr: int
    create_system: int
    local_header_offset: int
    is_directory: bool

    def as_record(self) -> dict[str, Any]:
        return {
            "compressed_size": self.compressed_size,
            "compression_method": self.compression_method,
            "crc32": f"{self.crc32:08x}",
            "encrypted": bool(self.flag_bits & 0x1),
            "is_directory": self.is_directory,
            "local_header_offset": self.local_header_offset,
            "name": self.name,
            "uncompressed_size": self.uncompressed_size,
        }


def _fetch(
    expectation: RemoteArchiveExpectation,
    *,
    start: int,
    end: int,
    index: int,
    opener: RangeOpener | None,
) -> bytes:
    if opener is None:
        return cast(
            bytes,
            fetch_exact_range(
                expectation,
                index=index,
                start=start,
                end=end,
            ).data,
        )
    return cast(
        bytes,
        fetch_exact_range(
            expectation,
            index=index,
            start=start,
            end=end,
            opener=opener,
        ).data,
    )


def _find_eocd(
    tail: bytes,
) -> tuple[int, tuple[bytes, int, int, int, int, int, int, int]]:
    position = tail.rfind(_EOCD_SIGNATURE)
    while position >= 0:
        if position + _EOCD.size <= len(tail):
            record = _EOCD.unpack_from(tail, position)
            comment_length = record[-1]
            if position + _EOCD.size + comment_length == len(tail):
                return position, record
        position = tail.rfind(_EOCD_SIGNATURE, 0, position)
    raise ValueError("PoseIt ZIP end-of-central-directory record is missing")


def read_remote_zip_layout(
    expectation: RemoteArchiveExpectation,
    *,
    opener: RangeOpener | None = None,
) -> RemoteZipLayout:
    """Read only ZIP end records and return the central-directory layout."""

    tail_size = min(expectation.size_bytes, _EOCD_SEARCH_BYTES)
    tail_start = expectation.size_bytes - tail_size
    tail = _fetch(
        expectation,
        start=tail_start,
        end=expectation.size_bytes - 1,
        index=0,
        opener=opener,
    )
    relative_eocd, eocd = _find_eocd(tail)
    (
        signature,
        disk_number,
        central_disk,
        disk_entries,
        total_entries,
        central_size32,
        central_offset32,
        comment_length,
    ) = eocd
    _require(signature == _EOCD_SIGNATURE, "PoseIt EOCD signature changed")
    _require(disk_number == 0 and central_disk == 0, "multi-disk ZIP is unsupported")
    _require(disk_entries == total_entries, "ZIP entry counts disagree across disks")
    eocd_offset = tail_start + relative_eocd
    comment = tail[
        relative_eocd + _EOCD.size : relative_eocd + _EOCD.size + comment_length
    ]
    needs_zip64 = (
        total_entries == _UINT16_MAX
        or central_size32 == _UINT32_MAX
        or central_offset32 == _UINT32_MAX
    )
    zip64_offset: int | None = None
    if needs_zip64:
        locator_offset = eocd_offset - _ZIP64_LOCATOR.size
        _require(locator_offset >= 0, "ZIP64 locator offset is invalid")
        if locator_offset >= tail_start:
            locator = tail[
                locator_offset - tail_start : locator_offset
                - tail_start
                + _ZIP64_LOCATOR.size
            ]
        else:
            locator = _fetch(
                expectation,
                start=locator_offset,
                end=locator_offset + _ZIP64_LOCATOR.size - 1,
                index=1,
                opener=opener,
            )
        locator_signature, zip64_disk, zip64_offset, disk_count = _ZIP64_LOCATOR.unpack(
            locator
        )
        _require(
            locator_signature == _ZIP64_LOCATOR_SIGNATURE,
            "ZIP64 locator signature is missing",
        )
        _require(zip64_disk == 0 and disk_count == 1, "multi-disk ZIP64 is unsupported")
        _require(zip64_offset < locator_offset, "ZIP64 EOCD offset is invalid")
        zip64_header = _fetch(
            expectation,
            start=zip64_offset,
            end=zip64_offset + _ZIP64_EOCD.size - 1,
            index=2,
            opener=opener,
        )
        (
            zip64_signature,
            zip64_record_size,
            _version_made,
            _version_needed,
            zip64_disk_number,
            zip64_central_disk,
            zip64_disk_entries,
            zip64_total_entries,
            central_size,
            central_offset,
        ) = _ZIP64_EOCD.unpack(zip64_header)
        _require(
            zip64_signature == _ZIP64_EOCD_SIGNATURE,
            "ZIP64 EOCD signature is missing",
        )
        _require(zip64_record_size >= 44, "ZIP64 EOCD record is truncated")
        _require(
            zip64_offset + 12 + zip64_record_size <= locator_offset,
            "ZIP64 EOCD record overlaps its locator",
        )
        _require(
            zip64_disk_number == 0 and zip64_central_disk == 0,
            "multi-disk ZIP64 is unsupported",
        )
        _require(
            zip64_disk_entries == zip64_total_entries,
            "ZIP64 entry counts disagree across disks",
        )
        if total_entries != _UINT16_MAX:
            _require(
                total_entries == zip64_total_entries,
                "ZIP64 total entry count disagrees with EOCD",
            )
        if central_size32 != _UINT32_MAX:
            _require(
                central_size32 == central_size,
                "ZIP64 central-directory size disagrees with EOCD",
            )
        if central_offset32 != _UINT32_MAX:
            _require(
                central_offset32 == central_offset,
                "ZIP64 central-directory offset disagrees with EOCD",
            )
        entry_count = zip64_total_entries
    else:
        central_size = central_size32
        central_offset = central_offset32
        entry_count = total_entries

    _require(entry_count > 0, "PoseIt ZIP has no central-directory entries")
    _require(central_size > 0, "PoseIt ZIP central directory is empty")
    structure_start = zip64_offset if zip64_offset is not None else eocd_offset
    _require(
        central_offset + central_size <= structure_start,
        "PoseIt ZIP central directory overlaps its end records",
    )
    return RemoteZipLayout(
        archive_size_bytes=expectation.size_bytes,
        central_directory_offset=central_offset,
        central_directory_size=central_size,
        entry_count=entry_count,
        eocd_offset=eocd_offset,
        archive_comment=comment,
        zip64=needs_zip64,
        zip64_eocd_offset=zip64_offset,
    )


def fetch_remote_central_directory(
    expectation: RemoteArchiveExpectation,
    layout: RemoteZipLayout,
    *,
    opener: RangeOpener | None = None,
    chunk_size_bytes: int = 32 * 1024 * 1024,
    maximum_size_bytes: int = 8 * 1024 * 1024 * 1024,
) -> bytes:
    """Fetch exactly the declared central-directory byte range."""

    _require(layout.archive_size_bytes == expectation.size_bytes, "layout size changed")
    _require(chunk_size_bytes > 0, "central-directory chunk size must be positive")
    _require(
        0 < layout.central_directory_size <= maximum_size_bytes,
        "central directory exceeds its registered size bound",
    )
    chunks: list[bytes] = []
    position = layout.central_directory_offset
    stop = position + layout.central_directory_size
    index = 10
    while position < stop:
        end = min(position + chunk_size_bytes, stop) - 1
        chunks.append(
            _fetch(
                expectation,
                start=position,
                end=end,
                index=index,
                opener=opener,
            )
        )
        position = end + 1
        index += 1
    result = b"".join(chunks)
    _require(
        len(result) == layout.central_directory_size,
        "central-directory transfer is incomplete",
    )
    return result


def _zip64_values(
    extra: bytes,
    *,
    uncompressed_size32: int,
    compressed_size32: int,
    local_offset32: int,
    disk_start16: int,
) -> tuple[int, int, int, int]:
    position = 0
    zip64: bytes | None = None
    while position < len(extra):
        _require(position + 4 <= len(extra), "ZIP extra field header is truncated")
        field_id, field_size = struct.unpack_from("<HH", extra, position)
        position += 4
        _require(position + field_size <= len(extra), "ZIP extra field is truncated")
        if field_id == 0x0001:
            _require(zip64 is None, "ZIP64 extra field is duplicated")
            zip64 = extra[position : position + field_size]
        position += field_size

    needs_zip64 = (
        uncompressed_size32 == _UINT32_MAX
        or compressed_size32 == _UINT32_MAX
        or local_offset32 == _UINT32_MAX
        or disk_start16 == _UINT16_MAX
    )
    if not needs_zip64:
        return uncompressed_size32, compressed_size32, local_offset32, disk_start16
    _require(zip64 is not None, "required ZIP64 extra field is missing")
    assert zip64 is not None
    position = 0

    def take(size: int) -> int:
        nonlocal position
        _require(position + size <= len(zip64), "ZIP64 extra field is truncated")
        fmt = "<Q" if size == 8 else "<L"
        value = int(struct.unpack_from(fmt, zip64, position)[0])
        position += size
        return value

    uncompressed_size = (
        take(8) if uncompressed_size32 == _UINT32_MAX else uncompressed_size32
    )
    compressed_size = take(8) if compressed_size32 == _UINT32_MAX else compressed_size32
    local_offset = take(8) if local_offset32 == _UINT32_MAX else local_offset32
    disk_start = take(4) if disk_start16 == _UINT16_MAX else disk_start16
    return uncompressed_size, compressed_size, local_offset, disk_start


def _safe_member_name(name: str, *, is_directory: bool) -> str:
    _require(bool(name), "ZIP member name is empty")
    _require("\x00" not in name, "ZIP member name contains NUL")
    _require("\\" not in name, "ZIP member name uses a backslash")
    _require(not name.startswith("/"), "ZIP member path is absolute")
    parts = PurePosixPath(name).parts
    _require(bool(parts), "ZIP member path is empty")
    _require(".." not in parts, "ZIP member path traverses its root")
    first = parts[0]
    _require(
        not (len(first) >= 2 and first[0].isalpha() and first[1] == ":"),
        "ZIP member path has a drive prefix",
    )
    normalized = PurePosixPath(*parts).as_posix()
    if is_directory and not normalized.endswith("/"):
        normalized += "/"
    _require(name == normalized, "ZIP member path is not canonical")
    return normalized


def parse_remote_central_directory(
    payload: bytes,
    *,
    expected_entries: int,
) -> tuple[RemoteZipMember, ...]:
    """Parse central metadata without opening a local member header or payload."""

    _require(expected_entries > 0, "expected ZIP entry count must be positive")
    members: list[RemoteZipMember] = []
    names: set[str] = set()
    position = 0
    for _ in range(expected_entries):
        _require(
            position + _CENTRAL.size <= len(payload), "central record is truncated"
        )
        (
            signature,
            version_made,
            _version_needed,
            flag_bits,
            compression_method,
            _modified_time,
            _modified_date,
            crc32,
            compressed_size32,
            uncompressed_size32,
            name_length,
            extra_length,
            comment_length,
            disk_start16,
            _internal_attr,
            external_attr,
            local_offset32,
        ) = _CENTRAL.unpack_from(payload, position)
        _require(signature == _CENTRAL_SIGNATURE, "central record signature changed")
        position += _CENTRAL.size
        variable_size = name_length + extra_length + comment_length
        _require(
            position + variable_size <= len(payload), "central record data is truncated"
        )
        name_bytes = payload[position : position + name_length]
        position += name_length
        extra = payload[position : position + extra_length]
        position += extra_length + comment_length
        encoding = "utf-8" if flag_bits & 0x800 else "cp437"
        name = name_bytes.decode(encoding, errors="strict")
        uncompressed_size, compressed_size, local_offset, disk_start = _zip64_values(
            extra,
            uncompressed_size32=uncompressed_size32,
            compressed_size32=compressed_size32,
            local_offset32=local_offset32,
            disk_start16=disk_start16,
        )
        _require(disk_start == 0, "multi-disk member is unsupported")
        _require(flag_bits & 0x1 == 0, "ZIP member is encrypted")
        create_system = version_made >> 8
        mode = external_attr >> 16
        kind = stat.S_IFMT(mode)
        name_directory = name.endswith("/")
        if create_system == 3:
            _require(
                kind in (0, stat.S_IFREG, stat.S_IFDIR),
                "ZIP member is a link or special file",
            )
            _require(
                not (name_directory and kind == stat.S_IFREG),
                "ZIP directory has regular-file mode",
            )
            _require(
                not (not name_directory and kind == stat.S_IFDIR),
                "ZIP regular member has directory mode",
            )
        normalized = _safe_member_name(name, is_directory=name_directory)
        _require(normalized not in names, "PoseIt ZIP has duplicate members")
        names.add(normalized)
        members.append(
            RemoteZipMember(
                name=normalized,
                crc32=crc32,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                compression_method=compression_method,
                flag_bits=flag_bits,
                external_attr=external_attr,
                create_system=create_system,
                local_header_offset=local_offset,
                is_directory=name_directory,
            )
        )

    remaining = payload[position:]
    if remaining:
        _require(
            len(remaining) >= 6 and remaining.startswith(_DIGITAL_SIGNATURE),
            "central directory has unregistered trailing bytes",
        )
        signature_size = struct.unpack_from("<H", remaining, 4)[0]
        _require(
            len(remaining) == 6 + signature_size,
            "central-directory digital signature is malformed",
        )
    return tuple(members)


def remote_zip_structure_summary(
    layout: RemoteZipLayout,
    central_directory: bytes,
    members: tuple[RemoteZipMember, ...],
) -> dict[str, Any]:
    """Return a compact, identity-bound structure summary."""

    _require(len(members) == layout.entry_count, "ZIP member count changed")
    records = sorted(
        (member.as_record() for member in members), key=lambda item: item["name"]
    )
    regular = [record for record in records if not record["is_directory"]]
    _require(bool(regular), "PoseIt ZIP has no regular members")
    local_offsets = [int(record["local_header_offset"]) for record in records]
    _require(
        all(0 <= offset < layout.central_directory_offset for offset in local_offsets),
        "ZIP local-header offset lies outside the member region",
    )
    _require(
        len(set(local_offsets)) == len(local_offsets),
        "ZIP members share a local-header offset",
    )
    extensions = Counter(
        PurePosixPath(str(record["name"])).suffix.casefold() or "<none>"
        for record in regular
    )
    depths = Counter(
        len(PurePosixPath(str(record["name"])).parts) for record in records
    )
    top_level = {PurePosixPath(str(record["name"])).parts[0] for record in records}
    identity = hashlib.sha256(b"poseit-remote-zip-central-directory-v1\0")
    names = hashlib.sha256(b"poseit-zip-member-names-v1\0")
    for record in records:
        fields = (
            str(record["name"]),
            str(record["crc32"]),
            str(record["compressed_size"]),
            str(record["uncompressed_size"]),
            str(record["compression_method"]),
            str(record["local_header_offset"]),
            "1" if record["is_directory"] else "0",
        )
        identity.update(("\0".join(fields) + "\n").encode())
        names.update((str(record["name"]) + "\n").encode())
    return {
        "archive_comment_length": len(layout.archive_comment),
        "archive_comment_sha256": hashlib.sha256(layout.archive_comment).hexdigest(),
        "central_directory_offset": layout.central_directory_offset,
        "central_directory_size": layout.central_directory_size,
        "central_directory_bytes_sha256": hashlib.sha256(central_directory).hexdigest(),
        "central_directory_identity_sha256": identity.hexdigest(),
        "directory_member_count": len(records) - len(regular),
        "extension_counts": dict(sorted(extensions.items())),
        "member_depth_counts": {
            str(depth): count for depth, count in sorted(depths.items())
        },
        "member_names_sha256": names.hexdigest(),
        "regular_member_count": len(regular),
        "top_level_component_count": len(top_level),
        "total_compressed_member_bytes": sum(
            int(record["compressed_size"]) for record in regular
        ),
        "total_uncompressed_member_bytes": sum(
            int(record["uncompressed_size"]) for record in regular
        ),
        "zip64": layout.zip64,
    }


__all__ = [
    "RemoteZipLayout",
    "RemoteZipMember",
    "fetch_remote_central_directory",
    "parse_remote_central_directory",
    "read_remote_zip_layout",
    "remote_zip_structure_summary",
]
