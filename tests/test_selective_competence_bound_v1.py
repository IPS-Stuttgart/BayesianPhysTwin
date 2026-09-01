from __future__ import annotations

import math

import pytest

from bayesian_phystwin.selective_competence_bound_v1 import (
    SelectiveCompetenceCertificateV1,
    clopper_pearson_upper,
    hoeffding_upper,
)


def test_zero_harm_closed_form() -> None:
    expected = 1.0 - 0.05 ** (1.0 / 29.0)
    assert clopper_pearson_upper(0, 29) == pytest.approx(expected, abs=1e-12)


def test_clopper_pearson_is_monotone() -> None:
    assert clopper_pearson_upper(0, 40) < clopper_pearson_upper(1, 40)
    assert clopper_pearson_upper(1, 80) < clopper_pearson_upper(1, 40)
    assert clopper_pearson_upper(0, 0) == 1.0


def test_simultaneous_hoeffding_penalty_grows_with_family() -> None:
    single = hoeffding_upper(-0.5, 50, lower=-1.0, upper=1.0)
    family = hoeffding_upper(
        -0.5,
        50,
        lower=-1.0,
        upper=1.0,
        family_size=20,
    )
    assert -0.5 < single < family <= 1.0


def test_certificate_recomputes_and_authorizes() -> None:
    certificate = SelectiveCompetenceCertificateV1(
        accepted_count=100,
        harmful_count=0,
        empirical_excess_loss=-0.8,
        harm_alpha=0.05,
        regret_alpha=0.05,
        loss_lower=-1.0,
        loss_upper=1.0,
        group_count=100,
        family_size=1,
    )
    assert certificate.authorizes(maximum_harm=0.05)
    record = certificate.to_record()
    assert record["schema_version"] == 1
    assert math.isfinite(float(record["regret_upper_bound"]))


def test_rejects_tampered_endpoint() -> None:
    with pytest.raises(ValueError, match="does not match"):
        SelectiveCompetenceCertificateV1(
            accepted_count=20,
            harmful_count=0,
            empirical_excess_loss=-0.1,
            harm_alpha=0.05,
            regret_alpha=0.05,
            loss_lower=-1.0,
            loss_upper=1.0,
            group_count=20,
            family_size=1,
            harm_upper_bound=0.0,
        )
