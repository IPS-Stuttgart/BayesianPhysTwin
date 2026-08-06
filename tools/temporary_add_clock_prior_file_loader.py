"""One-shot strict clock-prior loader patch for PR #187."""

from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one replacement target in {path}, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "src/bayesian_phystwin/causal4d_observation_clock_prior.py",
    "from dataclasses import dataclass\nfrom typing import Any\n",
    "from dataclasses import dataclass\nfrom pathlib import Path\nfrom typing import Any\n",
)
replace_once(
    "src/bayesian_phystwin/causal4d_observation_clock_prior.py",
    "    exact_revision,\n    nonempty_string,\n",
    "    exact_revision,\n    load_strict_json_object,\n    nonempty_string,\n",
)
replace_once(
    "src/bayesian_phystwin/causal4d_observation_clock_prior.py",
    "\n\n__all__ = [\n",
    '''


def load_causal4d_observation_timing_prior(
    path: str | Path,
    *,
    expected_artifact_id: str,
    expected_clock_domain: str,
    expected_time_scale: str,
) -> ObservationTimingPrior:
    """Strictly load, reconstruct, and bind one Causal4D prior artifact."""

    value = load_strict_json_object(
        path,
        label="Causal4D observation clock-offset prior",
    )
    return causal4d_observation_timing_prior_from_record(
        value,
        expected_artifact_id=expected_artifact_id,
        expected_clock_domain=expected_clock_domain,
        expected_time_scale=expected_time_scale,
    )


__all__ = [
''',
)
replace_once(
    "src/bayesian_phystwin/causal4d_observation_clock_prior.py",
    '    "causal4d_observation_timing_prior_from_record",\n]',
    '    "causal4d_observation_timing_prior_from_record",\n'
    '    "load_causal4d_observation_timing_prior",\n]',
)

replace_once(
    "tests/test_prob4d_observation_timestamps_causal4d.py",
    "import math\n",
    "import json\nimport math\n",
)
replace_once(
    "tests/test_prob4d_observation_timestamps_causal4d.py",
    "from dataclasses import replace\nfrom typing import Any\n",
    "from dataclasses import replace\nfrom pathlib import Path\nfrom typing import Any\n",
)
replace_once(
    "tests/test_prob4d_observation_timestamps_causal4d.py",
    "    causal4d_observation_timing_prior_from_record,\n",
    "    causal4d_observation_timing_prior_from_record,\n"
    "    load_causal4d_observation_timing_prior,\n",
)
replace_once(
    "tests/test_prob4d_observation_timestamps_causal4d.py",
    "\n\ndef test_actual_causal4d_prior_record_round_trips_exactly() -> None:\n",
    '''


def test_strict_clock_prior_file_loader_round_trip_and_duplicates(
    tmp_path: Path,
) -> None:
    producer = _causal4d_prior()
    assert producer.artifact_id is not None
    binding = _binding(producer.artifact_id)
    path = tmp_path / "clock-prior.json"
    path.write_text(json.dumps(producer.to_record()), encoding="utf-8")

    prior = load_causal4d_observation_timing_prior(
        path,
        expected_artifact_id=producer.artifact_id,
        expected_clock_domain=binding.clock_domain,
        expected_time_scale=binding.time_scale,
    )
    assert prior.source_artifact_id == producer.artifact_id

    path.write_text(
        '{"schema":"first","schema":"second"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_causal4d_observation_timing_prior(
            path,
            expected_artifact_id=producer.artifact_id,
            expected_clock_domain=binding.clock_domain,
            expected_time_scale=binding.time_scale,
        )


def test_actual_causal4d_prior_record_round_trips_exactly() -> None:
''',
)

replace_once(
    "docs/prob4d_observation_timestamps.md",
    "    causal4d_observation_timing_prior_from_record,\n",
    "    load_causal4d_observation_timing_prior,\n",
)
replace_once(
    "docs/prob4d_observation_timestamps.md",
    "prior = causal4d_observation_timing_prior_from_record(\n"
    "    source_only_prior_record,\n",
    "prior = load_causal4d_observation_timing_prior(\n"
    "    source_only_prior_path,\n",
)
