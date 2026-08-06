"""One-shot exact patch for PR #187; deleted by its publisher workflow."""

from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one replacement target in {path}, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_all(path: str, old: str, new: str, *, expected: int) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected:
        raise SystemExit(
            f"expected {expected} replacement targets in {path}, found {count}"
        )
    target.write_text(text.replace(old, new), encoding="utf-8")


def replace_between(
    path: str,
    *,
    start: str,
    end: str,
    replacement: str,
) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    start_count = text.count(start)
    end_count = text.count(end)
    if start_count != 1 or end_count != 1:
        raise SystemExit(
            f"replacement markers in {path} are not unique: "
            f"start={start_count}, end={end_count}"
        )
    first = text.index(start)
    last = text.index(end, first + len(start))
    target.write_text(text[:first] + replacement + text[last:], encoding="utf-8")


replace_once(
    "src/bayesian_phystwin/prob4d_observation_timestamps.py",
    '        """Bind a source-only prior without merging it into local jitter."""\n',
    '        """Construct an exploratory compact prior; not claim-bearing."""\n',
)

replace_once(
    "tests/test_prob4d_observation_timestamps_causal4d.py",
    "from __future__ import annotations\n\n"
    "import math\n"
    "from dataclasses import replace\n\n"
    "import numpy as np\n"
    "import pytest\n\n"
    "from bayesian_phystwin.causal4d_observation_clock_binding import (\n"
    "    bind_causal4d_observation_clock_prior,\n"
    ")\n"
    "from bayesian_phystwin.causal4d_observation_clock_prior import (\n"
    "    Causal4DObservationClockOffsetPriorV1,\n"
    ")\n",
    "from __future__ import annotations\n\n"
    "import math\n"
    "from collections.abc import Mapping\n"
    "from dataclasses import replace\n"
    "from typing import Any\n\n"
    "import numpy as np\n"
    "import pytest\n\n"
    "from bayesian_phystwin.causal4d_observation_clock_prior import (\n"
    "    Causal4DObservationClockOffsetPriorV1,\n"
    "    causal4d_observation_timing_prior_from_record,\n"
    ")\n"
    "from bayesian_phystwin.observation_timing_nuisance import (\n"
    "    ObservationTimingPrior,\n"
    ")\n",
)

replace_once(
    "tests/test_prob4d_observation_timestamps_causal4d.py",
    "    )\n\n\ndef test_actual_causal4d_prior_record_round_trips_exactly() -> None:\n",
    "    )\n\n\n"
    "def _consume(\n"
    "    binding: Prob4DObservationTimestampBindingV1,\n"
    "    record: Mapping[str, Any],\n"
    ") -> ObservationTimingPrior:\n"
    "    expected_id = binding.shared_clock_offset_prior_artifact_id\n"
    "    if expected_id is None:\n"
    "        raise ValueError(\"timestamp lineage declares no shared clock prior\")\n"
    "    return causal4d_observation_timing_prior_from_record(\n"
    "        record,\n"
    "        expected_artifact_id=expected_id,\n"
    "        expected_clock_domain=binding.clock_domain,\n"
    "        expected_time_scale=binding.time_scale,\n"
    "    )\n\n\n"
    "def test_actual_causal4d_prior_record_round_trips_exactly() -> None:\n",
)

replace_all(
    "tests/test_prob4d_observation_timestamps_causal4d.py",
    "bind_causal4d_observation_clock_prior(",
    "_consume(",
    expected=10,
)

replace_between(
    "docs/prob4d_observation_timestamps.md",
    start="## Shared clock design and prior\n",
    end="\n## Information-order boundary\n",
    replacement=(
        "## Shared clock design and prior\n"
        "\n"
        "```python\n"
        "from bayesian_phystwin.causal4d_observation_clock_prior import (\n"
        "    causal4d_observation_timing_prior_from_record,\n"
        ")\n"
        "\n"
        "clock_design = binding.shared_clock_design(\n"
        "    observation_derivative_xyz_per_s\n"
        ")\n"
        "prior = causal4d_observation_timing_prior_from_record(\n"
        "    source_only_prior_record,\n"
        "    expected_artifact_id=(\n"
        "        binding.shared_clock_offset_prior_artifact_id\n"
        "    ),\n"
        "    expected_clock_domain=binding.clock_domain,\n"
        "    expected_time_scale=binding.time_scale,\n"
        ")\n"
        "```\n"
        "\n"
        "The design has shape `(3N, 1)` and uses the same coordinate flattening as the\n"
        "physical and nuisance Jacobians. Claim-bearing consumption requires the complete\n"
        "content-addressed Causal4D prior record, not only its compact Gaussian payload.\n"
        "BayesianPhysTwin independently checks the closed schema, source-only information\n"
        "boundary, source execution count and ordering, source offsets, predictive-width\n"
        "formula and floor, exact content ID, clock domain, time scale, and correction\n"
        "convention:\n"
        "\n"
        "```text\n"
        "aligned_observation_time_s = observation_time_s + offset_s\n"
        "```\n"
        "\n"
        "A compact payload containing an artifact ID, mean, and standard deviation cannot\n"
        "tie those numeric values to that ID. The legacy\n"
        "`binding.shared_clock_prior_from_payload(...)` helper is therefore exploratory\n"
        "only and must not authorize a claim-bearing run. The full-record validator\n"
        "rejects compact payloads and returns an `ObservationTimingPrior` only after the\n"
        "complete record has been reconstructed successfully.\n"
        "\n"
        "Timing identifiability must still be assessed against physical-state, gauge,\n"
        "visual-bias, and material-lag modes. A source timestamp sidecar and a valid\n"
        "source-only prior do not by themselves distinguish hardware clock error from\n"
        "physical relaxation.\n"
    ),
)
