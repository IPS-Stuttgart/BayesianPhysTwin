from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin_experiments.poseit_real_decision_protocol import (
    POSEIT_GELSIGHT_FILE_ID,
    poseit_mapping_constraints_file_sha256,
    poseit_protocol_file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/acquire_poseit_gelsight_archive_v1.py"
PROTOCOL = ROOT / "protocols/poseit_real_decision_probe_v1.json"
MAPPING_CONSTRAINTS = (
    ROOT
    / "protocols"
    / "poseit_real_decision_probe_v1_preaccess_mapping_constraints.json"
)


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("poseit_archive_acquisition", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(
        self,
        payload: bytes,
        *,
        content_type: str,
        disposition: str = "",
        final_url: str = "https://drive.usercontent.google.com/download",
    ) -> None:
        self._payload = payload
        self._position = 0
        self.headers = {
            "Content-Disposition": disposition,
            "Content-Type": content_type,
        }
        self.status = 200
        self._final_url = final_url
        self.closed = False

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            amount = len(self._payload) - self._position
        block = self._payload[self._position : self._position + amount]
        self._position += len(block)
        return block

    def geturl(self) -> str:
        return self._final_url

    def close(self) -> None:
        self.closed = True


def test_acquisition_streams_exact_file_opaquely_and_writes_receipt(
    tmp_path: Path,
) -> None:
    acquisition = _module()
    payload = b"PK\x03\x04opaque-archive-content"
    response = _Response(
        payload,
        content_type="application/octet-stream",
        disposition='attachment; filename="gelsight.zip"',
    )
    seen_url = ""

    def opener(request: object, timeout: float) -> _Response:
        nonlocal seen_url
        seen_url = request.full_url  # type: ignore[attr-defined]
        assert timeout == 7.0
        return response

    archive = tmp_path / "gelsight.zip"
    receipt = tmp_path / "receipt.json"
    result = acquisition._acquire(
        archive,
        receipt,
        PROTOCOL,
        MAPPING_CONSTRAINTS,
        expected_protocol_sha256=poseit_protocol_file_sha256(PROTOCOL),
        expected_mapping_constraints_sha256=(
            poseit_mapping_constraints_file_sha256(MAPPING_CONSTRAINTS)
        ),
        opener=opener,
        timeout_seconds=7.0,
    )

    assert seen_url == acquisition.SOURCE_URL
    assert f"id={POSEIT_GELSIGHT_FILE_ID}" in seen_url
    assert archive.read_bytes() == payload
    assert json.loads(receipt.read_text(encoding="utf-8")) == result
    identity = dict(result)
    receipt_id = identity.pop("receipt_id")
    assert receipt_id == content_id(identity)
    assert result["archive_bytes_streamed_opaquely"] is True
    assert result["zip_central_directory_parsed"] is False
    assert result["mapping_constraints_file_sha256"] == (
        poseit_mapping_constraints_file_sha256(MAPPING_CONSTRAINTS)
    )
    assert result["archive_member_names_opened"] is False
    assert result["member_payload_bytes_opened"] is False
    assert result["shake_outcomes_opened"] is False
    assert result["confirmation_opened"] is False
    assert result["held_v8_accessed"] is False
    assert response.closed is True


def test_quota_html_is_not_persisted(tmp_path: Path) -> None:
    acquisition = _module()
    response = _Response(
        b"Too many users have viewed or downloaded this file recently.",
        content_type="text/html; charset=utf-8",
    )

    with pytest.raises(RuntimeError, match="quota blocked"):
        acquisition._acquire(
            tmp_path / "gelsight.zip",
            tmp_path / "receipt.json",
            PROTOCOL,
            MAPPING_CONSTRAINTS,
            expected_protocol_sha256=poseit_protocol_file_sha256(PROTOCOL),
            expected_mapping_constraints_sha256=(
                poseit_mapping_constraints_file_sha256(MAPPING_CONSTRAINTS)
            ),
            opener=lambda request, timeout: response,
        )

    assert not (tmp_path / "gelsight.zip").exists()
    assert not (tmp_path / "receipt.json").exists()
    assert not list(tmp_path.glob("*.partial"))
    assert response.closed is True


def test_interrupted_transfer_is_not_persisted(tmp_path: Path) -> None:
    acquisition = _module()

    class InterruptedResponse(_Response):
        def read(self, amount: int = -1) -> bytes:
            block = super().read(amount)
            if block:
                return block
            raise OSError("simulated transport interruption")

    response = InterruptedResponse(
        b"partial archive bytes",
        content_type="application/octet-stream",
        disposition='attachment; filename="gelsight.zip"',
    )

    with pytest.raises(OSError, match="transport interruption"):
        acquisition._acquire(
            tmp_path / "gelsight.zip",
            tmp_path / "receipt.json",
            PROTOCOL,
            MAPPING_CONSTRAINTS,
            expected_protocol_sha256=poseit_protocol_file_sha256(PROTOCOL),
            expected_mapping_constraints_sha256=(
                poseit_mapping_constraints_file_sha256(MAPPING_CONSTRAINTS)
            ),
            opener=lambda request, timeout: response,
        )

    assert not (tmp_path / "gelsight.zip").exists()
    assert not (tmp_path / "receipt.json").exists()
    assert not list(tmp_path.glob("*.partial"))
    assert response.closed is True


@pytest.mark.parametrize(
    ("content_type", "disposition", "message"),
    (
        ("text/html", "", "archive attachment"),
        ("text/html", 'attachment; filename="gelsight.zip"', "response is HTML"),
        (
            "application/octet-stream",
            'attachment; filename="different.zip"',
            "name changed",
        ),
    ),
)
def test_acquisition_rejects_unregistered_responses(
    tmp_path: Path, content_type: str, disposition: str, message: str
) -> None:
    acquisition = _module()
    response = _Response(
        b"response",
        content_type=content_type,
        disposition=disposition,
    )

    with pytest.raises((RuntimeError, ValueError), match=message):
        acquisition._acquire(
            tmp_path / "gelsight.zip",
            tmp_path / "receipt.json",
            PROTOCOL,
            MAPPING_CONSTRAINTS,
            expected_protocol_sha256=poseit_protocol_file_sha256(PROTOCOL),
            expected_mapping_constraints_sha256=(
                poseit_mapping_constraints_file_sha256(MAPPING_CONSTRAINTS)
            ),
            opener=lambda request, timeout: response,
        )

    assert not (tmp_path / "gelsight.zip").exists()
    assert not (tmp_path / "receipt.json").exists()


def test_acquisition_is_write_once(tmp_path: Path) -> None:
    acquisition = _module()
    archive = tmp_path / "gelsight.zip"
    archive.write_bytes(b"reserved")

    with pytest.raises(ValueError, match="already exists"):
        acquisition._acquire(
            archive,
            tmp_path / "receipt.json",
            PROTOCOL,
            MAPPING_CONSTRAINTS,
            expected_protocol_sha256=poseit_protocol_file_sha256(PROTOCOL),
            expected_mapping_constraints_sha256=(
                poseit_mapping_constraints_file_sha256(MAPPING_CONSTRAINTS)
            ),
            opener=lambda request, timeout: pytest.fail("network was opened"),
        )


def test_acquisition_rejects_mapping_constraint_drift_before_network(
    tmp_path: Path,
) -> None:
    acquisition = _module()

    with pytest.raises(ValueError, match="mapping-constraint file SHA-256"):
        acquisition._acquire(
            tmp_path / "gelsight.zip",
            tmp_path / "receipt.json",
            PROTOCOL,
            MAPPING_CONSTRAINTS,
            expected_protocol_sha256=poseit_protocol_file_sha256(PROTOCOL),
            expected_mapping_constraints_sha256="a" * 64,
            opener=lambda request, timeout: pytest.fail("network was opened"),
        )


def test_acquisition_rejects_unregistered_redirect_host(tmp_path: Path) -> None:
    acquisition = _module()
    response = _Response(
        b"archive",
        content_type="application/octet-stream",
        disposition='attachment; filename="gelsight.zip"',
        final_url="https://example.invalid/archive",
    )

    with pytest.raises(ValueError, match="unregistered host"):
        acquisition._acquire(
            tmp_path / "gelsight.zip",
            tmp_path / "receipt.json",
            PROTOCOL,
            MAPPING_CONSTRAINTS,
            expected_protocol_sha256=poseit_protocol_file_sha256(PROTOCOL),
            expected_mapping_constraints_sha256=(
                poseit_mapping_constraints_file_sha256(MAPPING_CONSTRAINTS)
            ),
            opener=lambda request, timeout: response,
        )

    assert not (tmp_path / "gelsight.zip").exists()
    assert not (tmp_path / "receipt.json").exists()
