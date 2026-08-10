"""Reviewed one-shot execution marker for Deform360 geometric v4.

Importing this module has no side effects.  Its presence records the exact
reviewed request to execute the already-merged, development-only geometric-v4
materializer and observability evaluator on the protected workstation2 runner.
"""

from __future__ import annotations

from typing import Final

MERGED_IMPLEMENTATION_REVISION: Final = (
    "5439d16bbcd051d4b9908152263c44e3d562000e"
)
MATERIALIZER_POLICY_ID: Final = (
    "08405c7e85a4730b1affb0110f9d50bcb02db26462ce95bda374c8df83ef845b"
)
EXECUTION_REQUEST_ID: Final = (
    "deform360-joint-sparse-geometric-v4-reviewed-execution-v1"
)
CLAIM_BOUNDARY: Final = (
    "Development-only structural observability on the ten already-opened "
    "source objects. This request opens no adaptive-confirmation or "
    "confirmation payload, uses no target outcome, authorizes no confirmation, "
    "and establishes no physical-query benefit, deployment safety, Causal4D "
    "benefit, or state of the art."
)

__all__ = [
    "CLAIM_BOUNDARY",
    "EXECUTION_REQUEST_ID",
    "MERGED_IMPLEMENTATION_REVISION",
    "MATERIALIZER_POLICY_ID",
]
