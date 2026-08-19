from __future__ import annotations

import json

import numpy as np
import pytest

from bayesian_phystwin.predictive_query_mixture import (
    compose_same_mean_gaussian_mixture,
)
from bayesian_phystwin.query_density_calibration import (
    QueryDensityCalibrationV1,
    density_region_contains,
    fit_query_density_calibration,
    group_density_nonconformity,
    group_density_region_covered,
    load_query_density_calibration,
    save_query_density_calibration,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64
DIGEST_E = "e" * 64


def _prediction(endpoint_count: int = 2):
    mean = np.zeros((endpoint_count, 1), dtype=np.float64)
    nominal = np.ones((endpoint_count, 1, 1), dtype=np.float64)
    tail = np.ones((endpoint_count, 1, 1), dtype=np.float64) * 9.0
    return compose_same_mean_gaussian_mixture(
        mean,
        nominal,
        tail,
        reference_predictor_id="last-residual-v1",
        nominal_covariance_id="core-v1",
        tail_covariance_id="tail-v1",
        complete_predictor_id=DIGEST_A,
        nominal_probability=0.9,
    )


def _fit(scores: list[float] | None = None) -> QueryDensityCalibrationV1:
    prediction = _prediction(endpoint_count=1)
    values = list(range(9)) if scores is None else scores
    residual_groups = [np.asarray([[float(value)]]) for value in values]
    prediction_groups = [prediction] * len(values)
    return fit_query_density_calibration(
        calibration_group_ids=[f"object-{index:02d}" for index in range(len(values))],
        residual_groups=residual_groups,
        prediction_groups=prediction_groups,
        nominal_coverage=0.9,
        predictor_id=DIGEST_A,
        query_set_id=DIGEST_B,
        grouping_rule_id=DIGEST_C,
        guard_id=DIGEST_D,
        calibration_evidence_id=DIGEST_E,
    )


def test_group_density_nonconformity_is_the_maximum_endpoint_score() -> None:
    prediction = _prediction(endpoint_count=3)
    residual = np.asarray([[0.0], [1.0], [4.0]])

    score = group_density_nonconformity(residual, prediction)

    endpoint_scores = -np.log(
        0.9 * np.exp(-(residual[:, 0] ** 2) / 2.0) / np.sqrt(2.0 * np.pi)
        + 0.1 * np.exp(-(residual[:, 0] ** 2) / 18.0) / np.sqrt(18.0 * np.pi)
    )
    assert score == pytest.approx(float(np.max(endpoint_scores)))


def test_fit_uses_one_finite_group_order_statistic() -> None:
    calibration = _fit()

    assert calibration.finite_sample_rank == 9
    assert calibration.density_score_threshold == float(
        np.max(calibration.calibration_group_scores)
    )
    assert calibration.calibration_group_scores.flags.writeable is False
    assert calibration.calibration_group_ids == tuple(
        f"object-{index:02d}" for index in range(9)
    )


def test_density_region_membership_and_group_coverage() -> None:
    prediction = _prediction(endpoint_count=2)
    calibration = _fit()
    small = np.asarray([[0.0], [0.5]])
    huge = np.asarray([[0.0], [100.0]])

    membership = density_region_contains(
        small,
        prediction,
        calibration,
        predictor_id=DIGEST_A,
    )

    assert membership.shape == (2,)
    assert np.all(membership)
    assert group_density_region_covered(
        small,
        prediction,
        calibration,
        predictor_id=DIGEST_A,
    )
    assert not group_density_region_covered(
        huge,
        prediction,
        calibration,
        predictor_id=DIGEST_A,
    )


def test_predictor_identity_is_bound_to_calibration() -> None:
    prediction = _prediction()
    calibration = _fit()

    with pytest.raises(ValueError, match="does not match"):
        density_region_contains(
            np.zeros((2, 1)),
            prediction,
            calibration,
            predictor_id=DIGEST_B,
        )


def test_requested_coverage_must_have_a_finite_rank() -> None:
    prediction = _prediction(endpoint_count=1)

    with pytest.raises(ValueError, match="no finite split-conformal rank"):
        fit_query_density_calibration(
            calibration_group_ids=[f"g-{index}" for index in range(8)],
            residual_groups=[np.asarray([[0.0]])] * 8,
            prediction_groups=[prediction] * 8,
            nominal_coverage=0.9,
            predictor_id=DIGEST_A,
            query_set_id=DIGEST_B,
            grouping_rule_id=DIGEST_C,
            guard_id=DIGEST_D,
            calibration_evidence_id=DIGEST_E,
        )


def test_calibration_cannot_select_its_predictor_or_use_dependent_groups() -> None:
    prediction = _prediction(endpoint_count=1)
    common = dict(
        calibration_group_ids=[f"g-{index}" for index in range(9)],
        residual_groups=[np.asarray([[0.0]])] * 9,
        prediction_groups=[prediction] * 9,
        nominal_coverage=0.9,
        predictor_id=DIGEST_A,
        query_set_id=DIGEST_B,
        grouping_rule_id=DIGEST_C,
        guard_id=DIGEST_D,
        calibration_evidence_id=DIGEST_E,
    )

    with pytest.raises(ValueError, match="cannot select"):
        fit_query_density_calibration(
            **common,
            calibration_outcomes_used_for_selection=True,
        )
    with pytest.raises(ValueError, match="independent physical"):
        fit_query_density_calibration(
            **common,
            calibration_groups_independent=False,
        )


def test_strict_round_trip_is_content_addressed_and_idempotent(tmp_path) -> None:
    calibration = _fit()
    path = tmp_path / "density-calibration.json"

    save_query_density_calibration(calibration, path)
    save_query_density_calibration(calibration, path)
    loaded = load_query_density_calibration(path)

    assert loaded.artifact_id == calibration.artifact_id
    np.testing.assert_array_equal(
        loaded.calibration_group_scores,
        calibration.calibration_group_scores,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["density_score_threshold"] += 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="order statistic"):
        load_query_density_calibration(path)


def test_existing_different_artifact_is_not_replaced(tmp_path) -> None:
    first = _fit()
    second = _fit(scores=[float(value) / 10.0 for value in range(9)])
    path = tmp_path / "density-calibration.json"

    save_query_density_calibration(first, path)

    with pytest.raises(FileExistsError, match="different"):
        save_query_density_calibration(second, path)


def test_duplicate_json_keys_and_symlinks_fail_closed(tmp_path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":1,"schema":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_query_density_calibration(path)

    calibration = _fit()
    target = tmp_path / "target.json"
    save_query_density_calibration(calibration, target)
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        load_query_density_calibration(link)


def test_record_rejects_forged_rank_threshold_and_artifact_identity() -> None:
    calibration = _fit()
    values = calibration.as_dict()

    with pytest.raises(ValueError, match="finite-group conformal rank"):
        QueryDensityCalibrationV1(
            predictor_id=DIGEST_A,
            query_set_id=DIGEST_B,
            grouping_rule_id=DIGEST_C,
            guard_id=DIGEST_D,
            calibration_evidence_id=DIGEST_E,
            calibration_group_ids=calibration.calibration_group_ids,
            calibration_group_scores=calibration.calibration_group_scores,
            nominal_coverage=0.9,
            finite_sample_rank=8,
            density_score_threshold=calibration.density_score_threshold,
            predictor_frozen_before_scores=True,
            calibration_outcomes_used_for_selection=False,
            calibration_groups_independent=True,
        )

    values["artifact_id"] = DIGEST_A
    with pytest.raises(ValueError, match="artifact_id"):
        QueryDensityCalibrationV1.from_dict(values)


def test_private_density_validators_cover_scalar_json_and_group_edges() -> None:
    import bayesian_phystwin.query_density_calibration as module

    for value in ("", " padded "):
        with pytest.raises(ValueError, match="canonical string"):
            module._canonical_string(value, name="value")
    with pytest.raises(ValueError, match="single canonical line"):
        module._canonical_string("a\nb", name="value")
    with pytest.raises(ValueError, match="SHA-256"):
        module._sha256("abc", name="digest")
    for value in (True, float("inf")):
        with pytest.raises(ValueError, match="finite real"):
            module._finite_real(value, name="value")
    with pytest.raises(ValueError, match="at least"):
        module._finite_real(-1.0, name="value", minimum=0.0)
    with pytest.raises(ValueError, match="strictly inside"):
        module._open_probability(1.0, name="coverage")
    assert module._plain_json(np.int64(3)) == 3
    with pytest.raises(ValueError, match="finite JSON"):
        module._plain_json(float("nan"))
    with pytest.raises(ValueError, match="finite JSON"):
        module._plain_json(object())
    with pytest.raises(ValueError, match="sequence"):
        module._canonical_group_ids("group", count=1)
    with pytest.raises(ValueError, match="length"):
        module._canonical_group_ids(["a"], count=2)
    with pytest.raises(ValueError, match="unique"):
        module._canonical_group_ids(["a", "a"], count=2)


def test_fsync_directory_tolerates_open_and_fsync_failures(
    monkeypatch,
    tmp_path,
) -> None:
    import bayesian_phystwin.query_density_calibration as module

    def fail_open(*args, **kwargs):
        raise OSError

    monkeypatch.setattr(module.os, "open", fail_open)
    module._fsync_directory(tmp_path)

    monkeypatch.setattr(module.os, "open", lambda *args, **kwargs: 7)

    def fail_fsync(descriptor):
        raise OSError

    monkeypatch.setattr(module.os, "fsync", fail_fsync)
    closed: list[int] = []
    monkeypatch.setattr(module.os, "close", closed.append)
    module._fsync_directory(tmp_path)
    assert closed == [7]


def test_group_density_nonconformity_rejects_empty_and_nonfinite_scores(
    monkeypatch,
) -> None:
    import bayesian_phystwin.query_density_calibration as module

    prediction = _prediction(endpoint_count=1)
    monkeypatch.setattr(
        module,
        "gaussian_mixture_negative_log_density",
        lambda residual, value: np.asarray([], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="at least one endpoint"):
        group_density_nonconformity(np.zeros((1, 1)), prediction)
    monkeypatch.setattr(
        module,
        "gaussian_mixture_negative_log_density",
        lambda residual, value: np.asarray([np.inf]),
    )
    with pytest.raises(ValueError, match="must be finite"):
        group_density_nonconformity(np.zeros((1, 1)), prediction)


def test_calibration_record_rejects_score_shape_empty_rank_type_and_unfrozen() -> None:
    common = dict(
        predictor_id=DIGEST_A,
        query_set_id=DIGEST_B,
        grouping_rule_id=DIGEST_C,
        guard_id=DIGEST_D,
        calibration_evidence_id=DIGEST_E,
        calibration_group_ids=[f"g-{index}" for index in range(9)],
        calibration_group_scores=np.arange(9.0),
        nominal_coverage=0.9,
        finite_sample_rank=9,
        density_score_threshold=8.0,
        predictor_frozen_before_scores=True,
        calibration_outcomes_used_for_selection=False,
        calibration_groups_independent=True,
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        QueryDensityCalibrationV1(
            **{**common, "calibration_group_scores": np.zeros((3, 3))}
        )
    with pytest.raises(ValueError, match="nonempty"):
        QueryDensityCalibrationV1(
            **{
                **common,
                "calibration_group_ids": [],
                "calibration_group_scores": np.asarray([]),
                "finite_sample_rank": 1,
                "density_score_threshold": 0.0,
            }
        )
    with pytest.raises(ValueError, match="must be an integer"):
        QueryDensityCalibrationV1(**{**common, "finite_sample_rank": True})
    with pytest.raises(ValueError, match="must be frozen"):
        QueryDensityCalibrationV1(**{**common, "predictor_frozen_before_scores": False})


def test_from_dict_rejects_schema_field_and_group_drift() -> None:
    calibration = _fit()
    valid = calibration.as_dict()

    with pytest.raises(ValueError, match="must be a mapping"):
        QueryDensityCalibrationV1.from_dict([])
    missing = dict(valid)
    missing.pop("metadata")
    with pytest.raises(ValueError, match="missing or unknown"):
        QueryDensityCalibrationV1.from_dict(missing)
    for field_name, replacement, match in (
        ("schema", "other", "unsupported.*schema"),
        ("schema_version", 2, "unsupported.*version"),
        ("score", "other", "unsupported.*score"),
    ):
        changed = dict(valid)
        changed[field_name] = replacement
        with pytest.raises(ValueError, match=match):
            QueryDensityCalibrationV1.from_dict(changed)
    bad_groups = dict(valid)
    bad_groups["calibration_groups"] = []
    with pytest.raises(ValueError, match="nonempty list"):
        QueryDensityCalibrationV1.from_dict(bad_groups)
    bad_item = dict(valid)
    bad_item["calibration_groups"] = [{"group_id": "g"}]
    with pytest.raises(ValueError, match="group_id and score"):
        QueryDensityCalibrationV1.from_dict(bad_item)


def test_fit_and_region_type_contracts_fail_closed() -> None:
    prediction = _prediction(endpoint_count=1)
    with pytest.raises(ValueError, match="equal nonzero"):
        fit_query_density_calibration(
            calibration_group_ids=[],
            residual_groups=[],
            prediction_groups=[],
            nominal_coverage=0.9,
            predictor_id=DIGEST_A,
            query_set_id=DIGEST_B,
            grouping_rule_id=DIGEST_C,
            guard_id=DIGEST_D,
            calibration_evidence_id=DIGEST_E,
        )
    with pytest.raises(TypeError, match="calibration"):
        density_region_contains(
            np.zeros((1, 1)),
            prediction,
            object(),
            predictor_id=DIGEST_A,
        )


def test_save_contract_rejects_wrong_type_symlink_and_directory(tmp_path) -> None:
    calibration = _fit()
    with pytest.raises(TypeError, match="calibration"):
        save_query_density_calibration(object(), tmp_path / "x.json")

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="cannot be a symlink"):
        save_query_density_calibration(calibration, link)

    directory = tmp_path / "directory.json"
    directory.mkdir()
    with pytest.raises(ValueError, match="must be a file"):
        save_query_density_calibration(calibration, directory)


def test_load_rejects_unreadable_json_constant_and_nonfile(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="regular file"):
        load_query_density_calibration(missing)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        load_query_density_calibration(malformed)

    constant = tmp_path / "constant.json"
    constant.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON constant"):
        load_query_density_calibration(constant)


def test_concurrent_same_artifact_publication_path_is_idempotent(
    monkeypatch,
    tmp_path,
) -> None:
    import bayesian_phystwin.query_density_calibration as module

    calibration = _fit()
    target = tmp_path / "density.json"
    original_link = module.os.link

    def publish_then_report_exists(source, destination):
        target.write_bytes(Path(source).read_bytes())
        raise FileExistsError

    from pathlib import Path

    monkeypatch.setattr(module.os, "link", publish_then_report_exists)
    save_query_density_calibration(calibration, target)
    monkeypatch.setattr(module.os, "link", original_link)
    assert load_query_density_calibration(target).artifact_id == calibration.artifact_id
