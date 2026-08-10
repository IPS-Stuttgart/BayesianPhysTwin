from __future__ import annotations

import json
from importlib.resources import files

from bayesian_phystwin.evidence_decision_v1 import EvidenceDecisionV1


def test_packaged_schema_matches_closed_python_wire_shape() -> None:
    schema_path = files("bayesian_phystwin").joinpath(
        "contract_data",
        "evidence_decision_v1",
        "evidence-decision-v1.schema.json",
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    python_fields = {
        "decision_id",
        "schema_name",
        "schema_version",
        *EvidenceDecisionV1.__dataclass_fields__,
    }

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == python_fields
    assert set(schema["properties"]) == python_fields
    assert schema["properties"]["schema_name"]["const"] == (
        "bayesian_phystwin.evidence_decision"
    )
    assert schema["properties"]["schema_version"]["const"] == 1
