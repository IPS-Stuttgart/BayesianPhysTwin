"""Apply the reviewed BPT quality, coverage, and provider-gate repair."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement anchor, found {count}")
    target.write_text(text.replace(old, new), encoding="utf-8")


def append_once(path: str, marker: str, addition: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if marker in text:
        raise RuntimeError(f"{path}: repair marker already present")
    target.write_text(text.rstrip() + "\n\n\n" + addition.strip() + "\n", encoding="utf-8")


def repair_contracts() -> None:
    path = "src/bayesian_phystwin/_gauge_aware_contracts.py"
    replace_once(path, "from typing import Any\n", "from typing import Any, cast\n")
    replace_once(
        path,
        "def _require(condition: bool, message: str) -> None:\n"
        "    if not condition:\n"
        "        raise ValueError(message)\n",
        "def _require(condition: bool | np.bool_, message: str) -> None:\n"
        "    if not bool(condition):\n"
        "        raise ValueError(message)\n",
    )
    replace_once(
        path,
        "            query.ndim == 3 and query.shape[1:] == (3, state_count) and len(query),\n",
        "            query.ndim == 3\n"
        "            and query.shape[1:] == (3, state_count)\n"
        "            and len(query) > 0,\n",
    )
    replace_once(
        path,
        "        anchor_innovation = None\n"
        "        anchor_covariance = None\n"
        "        anchor_state = None\n"
        "        anchor_groups: tuple[str, ...] | None = None\n"
        "        anchor_reliability = None\n"
        "        anchor_nominal_probability = None\n"
        "        anchor_composite_weight = None\n"
        "        anchor_bias = None\n"
        "        anchor_bias_prior = None\n",
        "        anchor_innovation: np.ndarray | None = None\n"
        "        anchor_covariance: np.ndarray | None = None\n"
        "        anchor_state: np.ndarray | None = None\n"
        "        anchor_groups: tuple[str, ...] | None = None\n"
        "        anchor_reliability: np.ndarray | None = None\n"
        "        anchor_nominal_probability: np.ndarray | None = None\n"
        "        anchor_composite_weight: np.ndarray | None = None\n"
        "        anchor_bias: np.ndarray | None = None\n"
        "        anchor_bias_prior: np.ndarray | None = None\n",
    )
    replace_once(
        path,
        "                self.anchor_innovation_m, \"anchor_innovation_m\", 2\n",
        "                cast(np.ndarray, self.anchor_innovation_m),\n"
        "                \"anchor_innovation_m\",\n"
        "                2,\n",
    )
    replace_once(
        path,
        "                self.anchor_covariance_m2, \"anchor_covariance_m2\", 3\n",
        "                cast(np.ndarray, self.anchor_covariance_m2),\n"
        "                \"anchor_covariance_m2\",\n"
        "                3,\n",
    )
    replace_once(
        path,
        "                self.anchor_state_jacobian, \"anchor_state_jacobian\", 3\n",
        "                cast(np.ndarray, self.anchor_state_jacobian),\n"
        "                \"anchor_state_jacobian\",\n"
        "                3,\n",
    )
    replace_once(
        path,
        "                    self.anchor_bias_prior_covariance,\n"
        "                    \"anchor_bias_prior_covariance\",\n",
        "                    cast(np.ndarray, self.anchor_bias_prior_covariance),\n"
        "                    \"anchor_bias_prior_covariance\",\n",
    )
    replace_once(
        path,
        "        input_lineage=batch.metadata,\n",
        "        input_lineage=batch.metadata or {},\n",
    )


def repair_solver() -> None:
    path = "src/bayesian_phystwin/_gauge_aware_solver.py"
    replace_once(
        path,
        "from typing import Any, Protocol\n",
        "from typing import Any, Protocol, cast\n",
    )
    replace_once(
        path,
        "    weights = np.zeros(len(group_ids), dtype=np.float64)\n",
        "    weights: np.ndarray = np.zeros(len(group_ids), dtype=np.float64)\n",
    )
    replace_once(
        path,
        "    whitened_target = np.empty((count, 3), dtype=np.float64)\n",
        "    whitened_target: np.ndarray = np.empty((count, 3), dtype=np.float64)\n",
    )
    replace_once(
        path,
        "    whiteners = np.empty((count, 3, 3), dtype=np.float64)\n",
        "    whiteners: np.ndarray = np.empty((count, 3, 3), dtype=np.float64)\n",
    )
    replace_once(
        path,
        "    full_covariance = np.zeros(\n",
        "    full_covariance: np.ndarray = np.zeros(\n",
    )
    replace_once(
        path,
        "        batch.prior_nominal_probability,\n"
        "        batch.composite_weight,\n",
        "        cast(np.ndarray, batch.prior_nominal_probability),\n"
        "        cast(np.ndarray, batch.composite_weight),\n",
    )
    replace_once(
        path,
        "    if batch.anchor_innovation_m is None:\n"
        "        anchor_base_weight = np.zeros(0, dtype=np.float64)\n",
        "    anchor_base_weight: np.ndarray\n"
        "    if batch.anchor_innovation_m is None:\n"
        "        anchor_base_weight = np.zeros(0, dtype=np.float64)\n",
    )
    replace_once(
        path,
        "            batch.anchor_covariance_m2,\n"
        "            (batch.anchor_state_jacobian, raw_anchor_nuisance),\n",
        "            cast(np.ndarray, batch.anchor_covariance_m2),\n"
        "            (cast(np.ndarray, batch.anchor_state_jacobian), raw_anchor_nuisance),\n",
    )
    replace_once(
        path,
        "        input_lineage=batch.metadata,\n",
        "        input_lineage=batch.metadata or {},\n",
    )


def repair_prior_math() -> None:
    path = "src/bayesian_phystwin/_prior_aware_gauge_math.py"
    replace_once(
        path,
        "    base = np.zeros(len(labels), dtype=np.float64)\n"
        "    prior = np.empty(len(ordered), dtype=np.float64)\n"
        "    group_power = np.zeros(len(ordered), dtype=np.float64)\n",
        "    base: np.ndarray = np.zeros(len(labels), dtype=np.float64)\n"
        "    prior: np.ndarray = np.empty(len(ordered), dtype=np.float64)\n"
        "    group_power: np.ndarray = np.zeros(len(ordered), dtype=np.float64)\n",
    )


def repair_prior_aware_solver() -> None:
    path = "src/bayesian_phystwin/prior_aware_gauge_belief.py"
    replace_once(
        path,
        "    observation_floor_active = np.zeros(len(observation_groups), dtype=bool)\n"
        "    anchor_floor_active = np.zeros(len(anchor_groups), dtype=bool)\n",
        "    observation_floor_active: np.ndarray = np.zeros(\n"
        "        len(observation_groups), dtype=bool\n"
        "    )\n"
        "    anchor_floor_active: np.ndarray = np.zeros(len(anchor_groups), dtype=bool)\n",
    )
    replace_once(
        path,
        "                observation_prior[position],\n"
        "                cfg,\n",
        "                float(observation_prior[position]),\n"
        "                cfg,\n",
    )
    replace_once(
        path,
        "                anchor_prior[position],\n"
        "                cfg,\n",
        "                float(anchor_prior[position]),\n"
        "                cfg,\n",
    )


def repair_adapter() -> None:
    path = "src/bayesian_phystwin/observation_belief_gauge_adapter.py"
    replace_once(
        path,
        "def _require(condition: bool, message: str) -> None:\n"
        "    if not condition:\n"
        "        raise ValueError(message)\n",
        "def _require(condition: bool | np.bool_, message: str) -> None:\n"
        "    if not bool(condition):\n"
        "        raise ValueError(message)\n",
    )
    replace_once(
        path,
        "    _require(views.ndim == 1 and len(views), \"view_indices must be nonempty\")\n",
        "    _require(\n"
        "        views.ndim == 1 and len(views) > 0,\n"
        "        \"view_indices must be nonempty\",\n"
        "    )\n",
    )
    replace_once(
        path,
        "    def __post_init__(self) -> None:\n"
        "        _require(\n"
        "            self.observation_artifact_id\n"
        "            == self.batch.metadata.get(\"observation_artifact_id\"),\n",
        "    def __post_init__(self) -> None:\n"
        "        metadata = self.batch.metadata or {}\n"
        "        _require(\n"
        "            self.observation_artifact_id\n"
        "            == metadata.get(\"observation_artifact_id\"),\n",
    )
    replace_once(
        path,
        "    def summary(self) -> dict[str, object]:\n"
        "        return {\n",
        "    def summary(self) -> dict[str, object]:\n"
        "        metadata = self.batch.metadata or {}\n"
        "        return {\n",
    )
    replace_once(
        path,
        "            \"prob4d_causal_lineage_validated\": self.batch.metadata.get(\n",
        "            \"prob4d_causal_lineage_validated\": metadata.get(\n",
    )
    replace_once(
        path,
        "        query.ndim == 3 and query.shape[1:] == (3, state.shape[2]) and len(query),\n",
        "        query.ndim == 3\n"
        "        and query.shape[1:] == (3, state.shape[2])\n"
        "        and len(query) > 0,\n",
    )


def repair_provider_annotations() -> None:
    replace_once(
        "src/bayesian_phystwin/phystwin/geometry.py",
        "    lifted = np.zeros((len(tracked), state_count, 3), dtype=float)\n",
        "    lifted: np.ndarray = np.zeros(\n"
        "        (len(tracked), state_count, 3), dtype=float\n"
        "    )\n",
    )
    replace_once(
        "src/bayesian_phystwin/phystwin/replay.py",
        "        if isinstance(request, InitialReplayRequestV1):\n",
        "        frame_ids: np.ndarray\n"
        "        if isinstance(request, InitialReplayRequestV1):\n",
    )


def repair_adapter_tests() -> None:
    path = "tests/test_observation_belief_gauge_adapter.py"
    replace_once(
        path,
        "import numpy as np\n",
        "from dataclasses import replace\n\n"
        "import numpy as np\n"
        "import pytest\n",
    )
    replace_once(
        path,
        "from bayesian_phystwin.gauge_aware_belief import (\n"
        "    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,\n",
        "from bayesian_phystwin.gauge_aware_belief import (\n"
        "    COMPOSITE_WEIGHT_MODE_CONSUMER_CAP,\n"
        "    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,\n",
    )
    append_once(
        path,
        "test_adapter_rejects_unknown_prob4d_composite_weight_semantics",
        '''

def test_adapter_rejects_unknown_prob4d_composite_weight_semantics() -> None:
    belief = _belief(
        metadata={"group_composite_weight_semantics": "unsupported-semantics"}
    )

    with pytest.raises(ValueError, match="unsupported Prob4D"):
        _adapt(belief)


def test_non_prob4d_without_weight_semantics_uses_consumer_cap() -> None:
    belief = replace(
        _belief(),
        source_repository="Example/ObservationProvider",
        metadata={},
    )

    adapted = _adapt(belief)

    assert adapted.batch.composite_weight_mode == COMPOSITE_WEIGHT_MODE_CONSUMER_CAP
    assert adapted.batch.metadata["composite_weight_mode_source"] == "consumer-default"


def test_zero_rank_belief_has_no_gauge_parameters() -> None:
    belief = replace(
        _belief(),
        factor_names=(),
        low_rank_factor_m=np.zeros((4, 3, 0)),
    )

    adapted = _adapt(belief)

    assert adapted.batch.gauge_jacobian.shape == (4, 3, 0)
    assert adapted.gauge_parameter_names == ()
    assert adapted.gauge_parameter_group_ids.shape == (0,)


def test_global_translation_bias_rejects_empty_observation_set() -> None:
    with pytest.raises(ValueError, match="observation_count must be positive"):
        global_translation_bias_jacobian(0)
''',
    )


def main() -> None:
    repair_contracts()
    repair_solver()
    repair_prior_math()
    repair_prior_aware_solver()
    repair_adapter()
    repair_provider_annotations()
    repair_adapter_tests()


if __name__ == "__main__":
    main()
