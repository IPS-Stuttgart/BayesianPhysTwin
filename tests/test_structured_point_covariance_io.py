from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.structured_point_covariance import (
    StructuredPointCovarianceV1,
)
from bayesian_phystwin.structured_point_covariance_io import (
    STRUCTURED_POINT_COVARIANCE_ARCHIVE_SCHEMA,
    STRUCTURED_POINT_COVARIANCE_ARCHIVE_VERSION,
    StructuredPointCovarianceIOLimitsV1,
    load_structured_point_covariance,
    write_structured_point_covariance,
)

_SOURCE_ID = "1" * 64


def _artifact(point_count: int = 3) -> StructuredPointCovarianceV1:
    local = np.repeat(
        (np.eye(3, dtype=np.float64) * 0.01)[None, :, :],
        point_count,
        axis=0,
    )
    gauge = np.arange(point_count * 3 * 2, dtype=np.float64).reshape(
        point_count,
        3,
        2,
    )
    process = np.ones((point_count, 3, 1), dtype=np.float64)
    return StructuredPointCovarianceV1(
        point_ids=tuple(f"point-{index}" for index in range(point_count)),
        local_covariance_m2=local,
        shared_factors_m={
            "process": process / 100.0,
            "gauge": gauge / 1000.0,
        },
        coordinate_frame="world",
        source_artifact_id=_SOURCE_ID,
        metadata={"purpose": "portable-roundtrip-test"},
    )


def _payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _write_payload(path: Path, payload: dict[str, np.ndarray]) -> None:
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)


def _descriptor(payload: dict[str, np.ndarray]) -> dict[str, object]:
    return json.loads(str(payload["descriptor_json"].item()))


def _replace_descriptor(
    payload: dict[str, np.ndarray],
    descriptor: dict[str, object],
) -> None:
    payload["descriptor_json"] = np.asarray(
        json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def test_roundtrip_preserves_complete_decomposition(tmp_path: Path) -> None:
    covariance = _artifact()
    path = tmp_path / "covariance.npz"

    write_structured_point_covariance(path, covariance)
    loaded = load_structured_point_covariance(path)

    assert loaded.artifact_id == covariance.artifact_id
    assert loaded.descriptor() == covariance.descriptor()
    assert tuple(loaded.shared_factors_m) == ("gauge", "process")
    assert not loaded.local_covariance_m2.flags.writeable
    assert all(not value.flags.writeable for value in loaded.shared_factors_m.values())


def test_publication_is_no_clobber_with_explicit_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "covariance.npz"
    first = _artifact()
    second = _artifact(point_count=2)
    write_structured_point_covariance(path, first)

    with pytest.raises(FileExistsError):
        write_structured_point_covariance(path, second)

    write_structured_point_covariance(path, second, overwrite=True)
    assert load_structured_point_covariance(path).artifact_id == second.artifact_id


def test_tampered_array_fails_content_identity(tmp_path: Path) -> None:
    path = tmp_path / "covariance.npz"
    write_structured_point_covariance(path, _artifact())
    payload = _payload(path)
    payload["shared_factor__gauge"][0, 0, 0] += 1.0
    _write_payload(path, payload)

    with pytest.raises(ValueError, match="artifact_id"):
        load_structured_point_covariance(path)


@pytest.mark.parametrize(
    ("member", "replacement", "message"),
    [
        (
            "local_covariance_m2",
            lambda value: value.astype(np.float32),
            "exact dtype",
        ),
        (
            "shared_factor__gauge",
            lambda value: value.astype(np.float32),
            "exact dtype",
        ),
        (
            "shared_factor__gauge",
            lambda value: value[:, :, :0],
            "positive rank",
        ),
        (
            "shared_factor__gauge",
            lambda value: value[:-1],
            "shape",
        ),
    ],
)
def test_array_container_contract_is_strict(
    tmp_path: Path,
    member: str,
    replacement: object,
    message: str,
) -> None:
    path = tmp_path / "covariance.npz"
    write_structured_point_covariance(path, _artifact())
    payload = _payload(path)
    payload[member] = replacement(payload[member])  # type: ignore[operator]
    _write_payload(path, payload)

    with pytest.raises(ValueError, match=message):
        load_structured_point_covariance(path)


def test_archive_member_set_is_exact(tmp_path: Path) -> None:
    path = tmp_path / "covariance.npz"
    write_structured_point_covariance(path, _artifact())
    payload = _payload(path)
    payload["unexpected"] = np.zeros(1, dtype=np.float64)
    _write_payload(path, payload)

    with pytest.raises(ValueError, match="member set"):
        load_structured_point_covariance(path)


def test_resource_limits_fail_before_acceptance(tmp_path: Path) -> None:
    path = tmp_path / "covariance.npz"
    write_structured_point_covariance(path, _artifact())

    with pytest.raises(ValueError, match="archive byte budget"):
        load_structured_point_covariance(
            path,
            limits=StructuredPointCovarianceIOLimitsV1(maximum_archive_bytes=1),
        )
    with pytest.raises(ValueError, match="descriptor_json exceeds"):
        load_structured_point_covariance(
            path,
            limits=StructuredPointCovarianceIOLimitsV1(maximum_descriptor_bytes=1),
        )
    with pytest.raises(ValueError, match="point count"):
        load_structured_point_covariance(
            path,
            limits=StructuredPointCovarianceIOLimitsV1(maximum_points=2),
        )
    with pytest.raises(ValueError, match="shared rank"):
        load_structured_point_covariance(
            path,
            limits=StructuredPointCovarianceIOLimitsV1(
                maximum_total_shared_rank=2
            ),
        )


def test_duplicate_and_nonfinite_json_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "covariance.npz"
    write_structured_point_covariance(path, _artifact())
    payload = _payload(path)
    raw = str(payload["descriptor_json"].item())
    payload["descriptor_json"] = np.asarray(
        raw[:-1] + ',"schema":"duplicate"}'
    )
    _write_payload(path, payload)
    with pytest.raises(ValueError, match="duplicate JSON"):
        load_structured_point_covariance(path)

    write_structured_point_covariance(path, _artifact(), overwrite=True)
    payload = _payload(path)
    raw = str(payload["descriptor_json"].item())
    payload["descriptor_json"] = np.asarray(raw[:-1] + ',"invalid":NaN}')
    _write_payload(path, payload)
    with pytest.raises(ValueError, match="non-finite JSON"):
        load_structured_point_covariance(path)


def test_schema_and_component_roster_are_frozen(tmp_path: Path) -> None:
    path = tmp_path / "covariance.npz"
    write_structured_point_covariance(path, _artifact())
    payload = _payload(path)
    descriptor = _descriptor(payload)
    assert descriptor["schema"] == STRUCTURED_POINT_COVARIANCE_ARCHIVE_SCHEMA
    assert descriptor["schema_version"] == (
        STRUCTURED_POINT_COVARIANCE_ARCHIVE_VERSION
    )
    descriptor["schema_version"] = 2
    _replace_descriptor(payload, descriptor)
    _write_payload(path, payload)
    with pytest.raises(ValueError, match="version changed"):
        load_structured_point_covariance(path)

    write_structured_point_covariance(path, _artifact(), overwrite=True)
    payload = _payload(path)
    descriptor = _descriptor(payload)
    covariance_descriptor = descriptor["covariance_descriptor"]
    assert isinstance(covariance_descriptor, dict)
    factors = covariance_descriptor["shared_factors_m"]
    assert isinstance(factors, dict)
    factors["unknown"] = factors["gauge"]
    _replace_descriptor(payload, descriptor)
    _write_payload(path, payload)
    with pytest.raises(ValueError, match="unsupported"):
        load_structured_point_covariance(path)


def test_nonordinary_inputs_and_invalid_targets_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "covariance.npz"
    write_structured_point_covariance(path, _artifact())
    link = tmp_path / "link.npz"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="ordinary file"):
        load_structured_point_covariance(link)

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="must not be a directory"):
        write_structured_point_covariance(directory, _artifact())

    invalid = tmp_path / "invalid.npz"
    invalid.write_bytes(b"not an npz")
    with pytest.raises(ValueError, match="valid NPZ"):
        load_structured_point_covariance(invalid)


def test_argument_types_and_limit_values_are_not_coerced(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="covariance"):
        write_structured_point_covariance(
            tmp_path / "x",
            object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="overwrite"):
        write_structured_point_covariance(
            tmp_path / "x",
            _artifact(),
            overwrite=1,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="limits"):
        load_structured_point_covariance(
            tmp_path / "x",
            limits=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="positive integer"):
        StructuredPointCovarianceIOLimitsV1(
            maximum_points=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="at least one"):
        StructuredPointCovarianceIOLimitsV1(maximum_compression_ratio=0.5)
