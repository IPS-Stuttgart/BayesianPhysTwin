import math

import numpy as np
import pytest

from bayesian_phystwin.anytime_factor_envelope_v4 import (
    LowerEnvelopeMixtureEProcess,
    bernoulli_harm_factor,
    bounded_gain_factor,
    lower_envelope_factor,
)


def _process() -> LowerEnvelopeMixtureEProcess:
    return LowerEnvelopeMixtureEProcess(
        gain_bet_fractions=(0.1, 0.4, 0.8),
        maximum_harm_rate=0.1,
        harm_alternative_fractions=(0.1, 0.5, 0.9),
    )


def test_gain_factor_is_fair_on_the_gain_boundary() -> None:
    positive = bounded_gain_factor(gain_score=1.0, bet_fraction=0.4)
    negative = bounded_gain_factor(gain_score=-1.0, bet_fraction=0.4)

    assert 0.5 * positive + 0.5 * negative == pytest.approx(1.0)
    with pytest.raises(ValueError, match="gain_score"):
        bounded_gain_factor(gain_score=1.1, bet_fraction=0.4)


def test_harm_factor_is_fair_at_the_registered_ceiling() -> None:
    ceiling = 0.1
    safe = bernoulli_harm_factor(
        harmful=False,
        maximum_harm_rate=ceiling,
        alternative_fraction=0.5,
    )
    harmful = bernoulli_harm_factor(
        harmful=True,
        maximum_harm_rate=ceiling,
        alternative_fraction=0.5,
    )

    assert (1.0 - ceiling) * safe + ceiling * harmful == pytest.approx(1.0)
    assert 0.8 * safe + 0.2 * harmful < 1.0
    with pytest.raises(ValueError, match="literal bool"):
        bernoulli_harm_factor(
            harmful=1,  # type: ignore[arg-type]
            maximum_harm_rate=ceiling,
            alternative_fraction=0.5,
        )


def test_lower_envelope_is_dominated_by_every_component() -> None:
    value = lower_envelope_factor((1.4, 0.8, 1.1))

    assert value == 0.8
    with pytest.raises(ValueError, match="at least one"):
        lower_envelope_factor(())
    with pytest.raises(ValueError, match="positive"):
        lower_envelope_factor((1.0, 0.0))


def test_one_step_process_equals_cartesian_factor_average() -> None:
    process = _process()
    update = process.update(gain_score=0.2, harmful=False)

    gain = np.asarray([1.0 + bet * 0.2 for bet in (0.1, 0.4, 0.8)])
    harm = np.asarray(
        [
            bernoulli_harm_factor(
                harmful=False,
                maximum_harm_rate=0.1,
                alternative_fraction=fraction,
            )
            for fraction in (0.1, 0.5, 0.9)
        ]
    )
    expected = float(np.mean(np.minimum(gain[:, None], harm[None, :])))

    assert math.exp(update.log_e_value) == pytest.approx(expected)
    assert process.component_count == 9
    assert process.count == 1


def test_independent_tuning_can_beat_shared_fraction_scalarization() -> None:
    process = LowerEnvelopeMixtureEProcess(
        gain_bet_fractions=(0.8,),
        maximum_harm_rate=0.1,
        harm_alternative_fractions=(0.1,),
    )
    for _ in range(40):
        process.update(gain_score=0.2, harmful=False)

    # Version 3 with a shared lambda=0.8 uses the bounded harm score
    # S=(0.1-0)/0.9 and per-step factor 1+0.8*S.
    shared_log_wealth = 40 * math.log1p(0.8 * (0.1 / 0.9))
    assert process.log_e_value > shared_log_wealth


def test_strong_safe_benefit_crosses_registered_threshold() -> None:
    process = _process()
    threshold = -math.log(0.025)

    for _ in range(500):
        process.update(gain_score=0.3, harmful=False)
        if process.maximum_log_e_value >= threshold:
            break

    assert process.maximum_log_e_value >= threshold
    assert process.count < 500


def test_process_rejects_malformed_grids_and_updates() -> None:
    with pytest.raises(ValueError, match="unique"):
        LowerEnvelopeMixtureEProcess(
            gain_bet_fractions=(0.2, 0.2),
            maximum_harm_rate=0.1,
            harm_alternative_fractions=(0.5,),
        )
    with pytest.raises(ValueError, match="harm_alternative_fractions"):
        LowerEnvelopeMixtureEProcess(
            gain_bet_fractions=(0.2,),
            maximum_harm_rate=0.1,
            harm_alternative_fractions=(),
        )
    process = _process()
    with pytest.raises(ValueError, match="literal bool"):
        process.update(gain_score=0.0, harmful=0)  # type: ignore[arg-type]


def test_theorem_boundary_records_switching_union_scope() -> None:
    boundary = _process().theorem_boundary()

    assert boundary["schema_version"] == 4
    assert boundary["component_count"] == 9
    assert "active component may change arbitrarily" in str(
        boundary["pointwise_union_null"]
    )
