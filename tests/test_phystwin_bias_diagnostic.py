import pickle
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_bias_diagnostic import (
    PhysTwinBiasDiagnosticConfig,
    diagnose_phystwin_bias_forecast,
)


def test_simulator_residual_bias_can_remove_real_model_discrepancy(
    tmp_path: Path,
) -> None:
    frame_count = 8
    manual = np.zeros((frame_count, 1, 3))
    observed = manual.copy()
    baseline = manual.copy()
    baseline[1:, 0, 0] = -0.01
    data = {
        "object_points": observed,
        "object_visibilities": np.ones((frame_count, 1), dtype=bool),
        "object_motions_valid": np.ones((frame_count - 1, 1), dtype=bool),
    }
    paths = {}
    for name, value in (
        ("final", data),
        ("baseline", baseline),
        ("manual", manual),
    ):
        paths[name] = tmp_path / f"{name}.pkl"
        with paths[name].open("wb") as handle:
            pickle.dump(value, handle)

    result = diagnose_phystwin_bias_forecast(
        paths["final"],
        paths["baseline"],
        paths["manual"],
        config=PhysTwinBiasDiagnosticConfig(
            fit_end_frame=4,
            train_end_frame=6,
            minimum_fit_measurements=2,
        ),
    )

    assert result["inferred_bias_rms_m"] > 0.0
    assert result["validation"]["corrected_error_m"] > 0.0
    assert result["validation"]["raw_error_m"] == 0.0
