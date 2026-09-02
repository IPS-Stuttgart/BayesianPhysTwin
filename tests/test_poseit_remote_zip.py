from __future__ import annotations

import io
import stat
import struct
import urllib.request
import zipfile
from collections.abc import Mapping

import pytest

from bayesian_phystwin_experiments.poseit_remote_archive import (
    RemoteArchiveExpectation,
)
from bayesian_phystwin_experiments.poseit_remote_zip import (
    fetch_remote_central_directory,
    parse_remote_central_directory,
    read_remote_zip_layout,
    remote_zip_structure_summary,
)


class _Response:
    def __init__(self, data: bytes, start: int, end: int) -> None:
        payload = data[start : end + 1]
        self._payload = payload
        self._position = 0
        self.status = 206
        self.headers: Mapping[str, str] = {
            "Accept-Ranges": "bytes",
            "Content-Disposition": 'attachment; filename="gelsight.zip"',
            "Content-Length": str(len(payload)),
            "Content-Range": f"bytes {start}-{end}/{len(data)}",
            "Content-Type": "application/octet-stream",
            "Last-Modified": "Sat, 20 Aug 2022 02:26:04 GMT",
        }

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            amount = len(self._payload) - self._position
        block = self._payload[self._position : self._position + amount]
        self._position += len(block)
        return block

    def geturl(self) -> str:
        return "https://drive.usercontent.google.com/download"

    def close(self) -> None:
        pass


def _range(request: urllib.request.Request) -> tuple[int, int]:
    header = request.get_header("Range")
    assert header is not None
    start, end = header.removeprefix("bytes=").split("-", maxsplit=1)
    return int(start), int(end)


def _expectation(data: bytes) -> RemoteArchiveExpectation:
    return RemoteArchiveExpectation(
        source_url="https://drive.usercontent.google.com/download?id=frozen",
        file_name="gelsight.zip",
        size_bytes=len(data),
        last_modified="Sat, 20 Aug 2022 02:26:04 GMT",
        chunk_size_bytes=32,
        max_workers=1,
        max_attempts_per_range=1,
        timeout_seconds=5.0,
    )


def _opener(data: bytes, requests: list[tuple[int, int]] | None = None):
    def open_range(request: urllib.request.Request, timeout: float) -> _Response:
        assert timeout == 5.0
        start, end = _range(request)
        if requests is not None:
            requests.append((start, end))
        return _Response(data, start, end)

    return open_range


def _archive(
    *,
    unsafe_name: str | None = None,
    link: bool = False,
    duplicate: bool = False,
) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.comment = b"structure-only"
        bundle.writestr("root/", b"")
        bundle.writestr(unsafe_name or "root/first.txt", b"first payload")
        if link:
            info = zipfile.ZipInfo("root/link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            bundle.writestr(info, b"target")
        else:
            second_name = "root/first.txt" if duplicate else "root/second.bin"
            bundle.writestr(second_name, b"second payload")
    return stream.getvalue()


def _standard_central_offset(data: bytes) -> int:
    eocd_offset = data.rfind(b"PK\x05\x06")
    assert eocd_offset >= 0
    return int(struct.unpack_from("<L", data, eocd_offset + 16)[0])


def _zip64_archive() -> bytes:
    data = _archive()
    eocd_offset = data.rfind(b"PK\x05\x06")
    assert eocd_offset >= 0
    eocd = struct.unpack_from("<4s4H2LH", data, eocd_offset)
    _, _, _, _, entry_count, central_size, central_offset, comment_size = eocd
    comment = data[eocd_offset + 22 : eocd_offset + 22 + comment_size]
    prefix = data[:eocd_offset]
    zip64_offset = len(prefix)
    zip64 = struct.pack(
        "<4sQ2H2L4Q",
        b"PK\x06\x06",
        44,
        45,
        45,
        0,
        0,
        entry_count,
        entry_count,
        central_size,
        central_offset,
    )
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, zip64_offset, 1)
    sentinel = struct.pack(
        "<4s4H2LH",
        b"PK\x05\x06",
        0,
        0,
        0xFFFF,
        0xFFFF,
        0xFFFFFFFF,
        0xFFFFFFFF,
        len(comment),
    )
    return prefix + zip64 + locator + sentinel + comment


@pytest.mark.parametrize("zip64", (False, True))
def test_remote_structure_inventory_reads_only_ranges_and_matches_zip(
    zip64: bool,
) -> None:
    data = _zip64_archive() if zip64 else _archive()
    expectation = _expectation(data)
    requests: list[tuple[int, int]] = []
    opener = _opener(data, requests)

    layout = read_remote_zip_layout(expectation, opener=opener)
    central = fetch_remote_central_directory(
        expectation,
        layout,
        opener=opener,
        chunk_size_bytes=31,
    )
    members = parse_remote_central_directory(
        central,
        expected_entries=layout.entry_count,
    )
    summary = remote_zip_structure_summary(layout, central, members)

    assert layout.zip64 is zip64
    assert [member.name for member in members] == [
        "root/",
        "root/first.txt",
        "root/second.bin",
    ]
    assert all(
        member.local_header_offset < layout.central_directory_offset
        for member in members
    )
    assert summary["regular_member_count"] == 2
    assert summary["directory_member_count"] == 1
    assert summary["archive_comment_length"] == len(b"structure-only")
    assert requests
    assert all(0 <= start <= end < len(data) for start, end in requests)


@pytest.mark.parametrize(
    ("name", "message"),
    (
        ("../outside", "traverses"),
        ("/absolute", "absolute"),
        ("folder\\member", "backslash"),
        ("C:/member", "drive prefix"),
        ("C:member", "drive prefix"),
        ("root//member", "not canonical"),
        ("root/./member", "not canonical"),
    ),
)
def test_remote_structure_rejects_unsafe_member_names(name: str, message: str) -> None:
    data = _archive(unsafe_name=name)
    expectation = _expectation(data)
    opener = _opener(data)
    layout = read_remote_zip_layout(expectation, opener=opener)
    central = fetch_remote_central_directory(expectation, layout, opener=opener)

    with pytest.raises(ValueError, match=message):
        parse_remote_central_directory(central, expected_entries=layout.entry_count)


def test_remote_structure_rejects_links() -> None:
    data = _archive(link=True)
    expectation = _expectation(data)
    opener = _opener(data)
    layout = read_remote_zip_layout(expectation, opener=opener)
    central = fetch_remote_central_directory(expectation, layout, opener=opener)

    with pytest.raises(ValueError, match="link or special"):
        parse_remote_central_directory(central, expected_entries=layout.entry_count)


def test_remote_structure_rejects_encrypted_members() -> None:
    data = bytearray(_archive())
    central_offset = _standard_central_offset(data)
    flag_bits = struct.unpack_from("<H", data, central_offset + 8)[0]
    struct.pack_into("<H", data, central_offset + 8, flag_bits | 0x1)
    frozen = bytes(data)
    expectation = _expectation(frozen)
    opener = _opener(frozen)
    layout = read_remote_zip_layout(expectation, opener=opener)
    central = fetch_remote_central_directory(expectation, layout, opener=opener)

    with pytest.raises(ValueError, match="encrypted"):
        parse_remote_central_directory(central, expected_entries=layout.entry_count)


def test_remote_structure_rejects_duplicate_members() -> None:
    with pytest.warns(UserWarning, match="Duplicate name"):
        data = _archive(duplicate=True)
    expectation = _expectation(data)
    opener = _opener(data)
    layout = read_remote_zip_layout(expectation, opener=opener)
    central = fetch_remote_central_directory(expectation, layout, opener=opener)

    with pytest.raises(ValueError, match="duplicate"):
        parse_remote_central_directory(central, expected_entries=layout.entry_count)


def test_remote_structure_rejects_multi_disk_eocd() -> None:
    data = bytearray(_archive())
    eocd_offset = data.rfind(b"PK\x05\x06")
    assert eocd_offset >= 0
    struct.pack_into("<H", data, eocd_offset + 4, 1)
    frozen = bytes(data)

    with pytest.raises(ValueError, match="multi-disk"):
        read_remote_zip_layout(_expectation(frozen), opener=_opener(frozen))


def test_remote_structure_rejects_zip64_eocd_disagreement() -> None:
    data = bytearray(_zip64_archive())
    eocd_offset = data.rfind(b"PK\x05\x06")
    assert eocd_offset >= 0
    struct.pack_into("<L", data, eocd_offset + 12, 1)
    frozen = bytes(data)

    with pytest.raises(ValueError, match="size disagrees"):
        read_remote_zip_layout(_expectation(frozen), opener=_opener(frozen))


def test_remote_structure_parses_zip64_member_values() -> None:
    name = b"root/file.bin"
    values = struct.pack("<3Q", 123, 45, 7)
    extra = struct.pack("<HH", 0x0001, len(values)) + values
    payload = (
        struct.pack(
            "<4s6H3L5H2L",
            b"PK\x01\x02",
            (3 << 8) | 45,
            45,
            0,
            zipfile.ZIP_DEFLATED,
            0,
            0,
            0x10203040,
            0xFFFFFFFF,
            0xFFFFFFFF,
            len(name),
            len(extra),
            0,
            0,
            0,
            (stat.S_IFREG | 0o644) << 16,
            0xFFFFFFFF,
        )
        + name
        + extra
    )

    (member,) = parse_remote_central_directory(payload, expected_entries=1)

    assert member.uncompressed_size == 123
    assert member.compressed_size == 45
    assert member.local_header_offset == 7


def test_remote_structure_rejects_member_offset_outside_member_region() -> None:
    data = bytearray(_archive())
    central_offset = _standard_central_offset(data)
    struct.pack_into("<L", data, central_offset + 42, central_offset)
    frozen = bytes(data)
    expectation = _expectation(frozen)
    opener = _opener(frozen)
    layout = read_remote_zip_layout(expectation, opener=opener)
    central = fetch_remote_central_directory(expectation, layout, opener=opener)
    members = parse_remote_central_directory(
        central,
        expected_entries=layout.entry_count,
    )

    with pytest.raises(ValueError, match="outside the member region"):
        remote_zip_structure_summary(layout, central, members)


def test_remote_structure_rejects_central_directory_over_bound() -> None:
    data = _archive()
    expectation = _expectation(data)
    opener = _opener(data)
    layout = read_remote_zip_layout(expectation, opener=opener)

    with pytest.raises(ValueError, match="size bound"):
        fetch_remote_central_directory(
            expectation,
            layout,
            opener=opener,
            maximum_size_bytes=layout.central_directory_size - 1,
        )
