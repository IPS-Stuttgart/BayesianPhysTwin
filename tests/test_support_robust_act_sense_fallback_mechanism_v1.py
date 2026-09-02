from __future__ import annotations

import json
from pathlib import Path

from experiments.support_robust_act_sense_fallback_v1.run import build


def test_support_robust_mechanism_result_is_current() -> None:
    root = Path(__file__).parents[1]
    path = root / "experiments/support_robust_act_sense_fallback_v1/result.json"
    retained = json.loads(path.read_text(encoding="utf-8"))
    assert retained == build()
    assert retained["result_id"] == (
        "c31b84076cde53bb925beb5563190f97132481b431b09baf0a2332f9f953dae9"
    )
    assert [row["output_mode"] for row in retained["phase_diagram"]] == [
        "act",
        "sense",
        "sense",
        "fallback",
    ]
