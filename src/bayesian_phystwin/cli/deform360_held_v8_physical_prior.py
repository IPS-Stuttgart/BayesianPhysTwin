"""Run the frozen physical-prior numerical builder with explicit v8 hooks."""

from bayesian_phystwin.deform360_held_v8_builders import main_for


if __name__ == "__main__":
    main_for("bayesian_phystwin.cli.deform360_held_physical_prior")
