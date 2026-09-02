"""Run the registered Tracking Cloth stages with the stable-bank model wrapper."""

from __future__ import annotations

from experiments.tracking_cloth_self_collision_selective_twin_v1 import run as parent_run
from experiments.tracking_cloth_self_collision_stable_bank_v1.model import (
    PhysicsFit,
    all_predictions,
    fit_physics,
)


def main() -> int:
    """Patch only the physical-bank interface, then invoke the parent CLI."""

    parent_run.PhysicsFit = PhysicsFit
    parent_run.fit_physics = fit_physics
    parent_run.all_predictions = all_predictions
    return parent_run.main()


if __name__ == "__main__":
    raise SystemExit(main())
