import pytest
from test_grouped_likelihood import _belief

from bayesian_phystwin.grouped_likelihood import (
    conditional_grouped_student_t_mixture_objective,
    grouped_student_t_mixture_likelihood,
)


@pytest.mark.parametrize("bad_config", [False, 0, {}, object()])
def test_covariance_marginal_score_rejects_invalid_config(bad_config: object) -> None:
    belief = _belief()

    with pytest.raises(TypeError, match="GroupedStudentTLikelihoodConfig"):
        grouped_student_t_mixture_likelihood(
            belief,
            belief.mean_xyz_m,
            config=bad_config,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad_config", [False, 0, {}, object()])
def test_conditional_objective_rejects_invalid_config(bad_config: object) -> None:
    belief = _belief()

    with pytest.raises(TypeError, match="ConditionalGroupedStudentTObjectiveConfig"):
        conditional_grouped_student_t_mixture_objective(
            belief,
            belief.mean_xyz_m,
            config=bad_config,  # type: ignore[arg-type]
        )
