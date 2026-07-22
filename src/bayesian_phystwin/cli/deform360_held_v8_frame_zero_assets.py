"""Run the frozen frame-zero numerical builder with explicit held-v8 hooks."""

from bayesian_phystwin.deform360_held_v8_builders import main_for


if __name__ == "__main__":
    main_for("bayesian_phystwin.cli.deform360_frame_zero_assets")
