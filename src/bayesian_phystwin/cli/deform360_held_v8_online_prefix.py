"""Run the frozen online-prefix numerical builder with explicit v8 hooks."""

from bayesian_phystwin.deform360_held_v8_builders import main_for


if __name__ == "__main__":
    main_for("bayesian_phystwin.cli.deform360_held_online_prefix")
