import json
from pathlib import Path

import numpy as np
import pytest
from test_query_calibration import _IDS, _fit, _groups

from bayesian_phystwin.query_calibration import (
    QueryCalibrationV1,
    calibrate_query_covariance,
    fit_query_calibration,
    group_mahalanobis_nonconformity,
    load_query_calibration,
)


def _constructor_kwargs(
    calibration: QueryCalibrationV1 | None = None,
) -> dict[str, object]:
    record = (calibration or _fit()).as_dict()
    groups = record.pop("calibration_groups")
    record.pop("artifact_id")
    record.pop("schema")
    record.pop("schema_version")
    record.pop("score")
    record["calibration_group_ids"] = tuple(
        group["group_id"] for group in groups
    )
    record["calibration_group_scores"] = np.asarray(
        [group["score"] for group in groups]
    )
    return record


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"covariance_scale": True}, "finite real"),
        ({"covariance_scale": "1"}, "finite real"),
        ({"covariance_scale": np.nan}, "finite real"),
        ({"covariance_scale": 0.0}, "positive"),
        ({"isotropic_variance": -1.0}, "nonnegative"),
    ],
)
def test_covariance_transform_scalars_fail_closed(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        group_mahalanobis_nonconformity(
            np.ones(2),
            np.eye(2),
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("residual", "covariance", "message"),
    [
        (np.asarray(["1"]), np.eye(1), "residual.*real numeric"),
        (np.ones(1), np.asarray([["1"]]), "covariance.*real numeric"),
        (np.ones((1, 1, 1)), np.ones((1, 1, 1)), "residual must have shape"),
        (np.ones((1, 1)), np.ones((1, 1, 1, 1)), "covariance must have shape"),
        (np.empty((0, 1)), np.empty((0, 1, 1)), "cannot be empty"),
        (np.empty((1, 0)), np.empty((1, 0, 0)), "cannot be empty"),
        (np.ones(1), np.asarray([[np.inf]]), "covariance must be finite"),
    ],
)
def test_query_array_contract_rejects_malformed_inputs(
    residual: np.ndarray,
    covariance: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        group_mahalanobis_nonconformity(residual, covariance)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {
                "calibration_group_ids": (
                    "object-00",
                    "object-00",
                    *tuple(f"object-{index:02d}" for index in range(2, 9)),
                )
            },
            "unique",
        ),
        ({"calibration_group_scores": np.asarray(["1"] * 9)}, "real array"),
        ({"calibration_group_scores": np.ones((9, 1))}, "real array"),
        ({"calibration_group_scores": np.ones(8)}, "one score per group"),
        (
            {"calibration_group_scores": np.asarray([*range(1, 9), np.nan])},
            "finite and nonnegative",
        ),
        (
            {"calibration_group_scores": np.asarray([*range(1, 9), -1.0])},
            "finite and nonnegative",
        ),
        ({"predictor_frozen_before_scores": np.bool_(True)}, "must be a boolean"),
        (
            {"calibration_outcomes_used_for_selection": np.bool_(False)},
            "must be a boolean",
        ),
        ({"finite_sample_rank": 8}, "finite-group conformal rank"),
    ],
)
def test_query_calibration_contract_rejects_malformed_fields(
    changes: dict[str, object],
    message: str,
) -> None:
    values = _constructor_kwargs()
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        QueryCalibrationV1(**values)  # type: ignore[arg-type]


def test_from_dict_rejects_nonmapping_or_incomplete_records() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        QueryCalibrationV1.from_dict([])  # type: ignore[arg-type]

    record = _fit().as_dict()
    record.pop("guard_id")
    with pytest.raises(ValueError, match="missing or unknown"):
        QueryCalibrationV1.from_dict(record)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "other", "unsupported.*schema"),
        ("schema_version", 2, "schema version"),
        ("score", "other", "unsupported.*score"),
        ("calibration_groups", (), "nonempty list"),
        ("calibration_groups", [], "nonempty list"),
        ("calibration_groups", ["bad"], "group_id and score"),
        ("calibration_groups", [{"group_id": "only"}], "group_id and score"),
    ],
)
def test_from_dict_rejects_unsupported_or_malformed_values(
    field: str,
    value: object,
    message: str,
) -> None:
    record = _fit().as_dict()
    record[field] = value
    with pytest.raises(ValueError, match=message):
        QueryCalibrationV1.from_dict(record)


def test_from_dict_rejects_valid_but_content_tampered_record() -> None:
    record = _fit().as_dict()
    record["query_set_id"] = "6" * 64

    with pytest.raises(ValueError, match="artifact_id does not match content"):
        QueryCalibrationV1.from_dict(record)


@pytest.mark.parametrize("missing", ["residual", "covariance"])
def test_fit_requires_one_residual_and_covariance_entry_per_group(
    missing: str,
) -> None:
    group_ids, residuals, covariances = _groups()
    if missing == "residual":
        residuals = residuals[:-1]
    else:
        covariances = covariances[:-1]

    with pytest.raises(ValueError, match=f"{missing}_groups"):
        fit_query_calibration(
            group_ids,
            residuals,
            covariances,
            nominal_coverage=0.9,
            covariance_scale=1.0,
            isotropic_variance=0.0,
            predictor_frozen_before_scores=True,
            calibration_outcomes_used_for_selection=False,
            **_IDS,
        )


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (np.asarray(["1"]), "real numeric"),
        (np.asarray(1.0), "one or more square"),
        (np.empty((0, 0)), "one or more square"),
        (np.ones((2, 3)), "must be square"),
        (np.empty((0, 2, 2)), "nonempty and finite"),
        (np.asarray([[np.inf]]), "nonempty and finite"),
    ],
)
def test_calibrate_query_covariance_rejects_malformed_inputs(
    covariance: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        calibrate_query_covariance(covariance, _fit())


def test_calibrate_query_covariance_supports_batches() -> None:
    calibration = _fit()
    covariance = np.repeat(np.eye(2)[None, :, :], 3, axis=0)

    calibrated = calibrate_query_covariance(covariance, calibration)

    np.testing.assert_allclose(calibrated, 81.0 * covariance)
    assert calibrated.shape == (3, 2, 2)


def test_load_rejects_nonobject_json(tmp_path: Path) -> None:
    path = tmp_path / "query_calibration.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="one JSON object"):
        load_query_calibration(path)
