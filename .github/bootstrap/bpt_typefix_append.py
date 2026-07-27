replace_once(
    "src/bayesian_phystwin/observation_belief.py",
    '''    def __ior__(self, other):
        self._immutable(other)
''',
    '''    def __ior__(self, other) -> _FrozenDict:
        self._immutable(other)
        return self
''',
)
replace_once(
    "src/bayesian_phystwin/observation_belief.py",
    '''    def __iadd__(self, other):
        self._immutable(other)

    def __imul__(self, other):
        self._immutable(other)
''',
    '''    def __iadd__(self, other) -> _FrozenList:
        self._immutable(other)
        return self

    def __imul__(self, other) -> _FrozenList:
        self._immutable(other)
        return self
''',
)
replace_once(
    "src/bayesian_phystwin/observation_belief.py",
    '''    np.savez_compressed(
        target,
        descriptor_json=np.asarray(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
        **belief._arrays(),
    )
''',
    '''    archive_payload: dict[str, Any] = {
        "descriptor_json": np.asarray(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    }
    archive_payload.update(belief._arrays())
    np.savez_compressed(target, **archive_payload)
''',
)

replace_once(
    "src/bayesian_phystwin/grouped_likelihood.py",
    '''import numpy as np

from .observation_belief import ObservationBeliefV1
''',
    '''import numpy as np
from numpy.typing import NDArray

from .observation_belief import ObservationBeliefV1
''',
)
replace_once(
    "src/bayesian_phystwin/grouped_likelihood.py",
    '''    dimension = np.empty(group_count, dtype=np.int64)
    nll = np.empty(group_count, dtype=np.float64)
    weighted_nll = np.empty(group_count, dtype=np.float64)
    posterior_nominal = np.empty(group_count, dtype=np.float64)
    log_nominal = np.empty(group_count, dtype=np.float64)
    log_outlier = np.empty(group_count, dtype=np.float64)
    mean_association = np.empty(group_count, dtype=np.float64)
    covariance_logdet = np.empty(group_count, dtype=np.float64)
    covariance_mahalanobis = np.empty(group_count, dtype=np.float64)
''',
    '''    dimension: NDArray[np.int64] = np.empty(group_count, dtype=np.int64)
    nll: NDArray[np.float64] = np.empty(group_count, dtype=np.float64)
    weighted_nll: NDArray[np.float64] = np.empty(group_count, dtype=np.float64)
    posterior_nominal: NDArray[np.float64] = np.empty(
        group_count,
        dtype=np.float64,
    )
    log_nominal: NDArray[np.float64] = np.empty(group_count, dtype=np.float64)
    log_outlier: NDArray[np.float64] = np.empty(group_count, dtype=np.float64)
    mean_association: NDArray[np.float64] = np.empty(
        group_count,
        dtype=np.float64,
    )
    covariance_logdet: NDArray[np.float64] = np.empty(
        group_count,
        dtype=np.float64,
    )
    covariance_mahalanobis: NDArray[np.float64] = np.empty(
        group_count,
        dtype=np.float64,
    )
''',
)
target = Path("src/bayesian_phystwin/grouped_likelihood.py")
text = target.read_text(encoding="utf-8")
old_covariance_arguments = '''            covariance_log_determinant=covariance_logdet[position],
            covariance_mahalanobis_squared=(
                covariance_mahalanobis[position]
            ),
'''
new_covariance_arguments = '''            covariance_log_determinant=float(covariance_logdet[position]),
            covariance_mahalanobis_squared=float(
                covariance_mahalanobis[position]
            ),
'''
if text.count(old_covariance_arguments) != 2:
    raise SystemExit(
        "src/bayesian_phystwin/grouped_likelihood.py: expected two covariance argument targets"
    )
target.write_text(
    text.replace(old_covariance_arguments, new_covariance_arguments),
    encoding="utf-8",
)

replace_once(
    "src/bayesian_phystwin/phystwin_bayesian_anchor.py",
    '''from pathlib import Path

import numpy as np
''',
    '''from pathlib import Path
from typing import cast

import numpy as np
''',
)
replace_once(
    "src/bayesian_phystwin/phystwin_bayesian_anchor.py",
    '''    validation_improvement = 1.0 - float(selected_candidate["selection_score"])
    accepted = validation_improvement > config.minimum_validation_improvement
    process_std = float(selected_candidate["process_std_m"])
    observation_std = float(selected_candidate["observation_std_m"])
''',
    '''    validation_improvement = 1.0 - cast(
        float,
        selected_candidate["selection_score"],
    )
    accepted = validation_improvement > config.minimum_validation_improvement
    process_std = cast(float, selected_candidate["process_std_m"])
    observation_std = cast(float, selected_candidate["observation_std_m"])
''',
)
replace_once(
    "src/bayesian_phystwin/phystwin_bayesian_anchor.py",
    '''    summary["outputs"]["summary"] = str(summary_path.resolve())
    return summary
''',
    '''    summary_outputs = cast(dict[str, str], summary["outputs"])
    summary_outputs["summary"] = str(summary_path.resolve())
    return summary
''',
)
