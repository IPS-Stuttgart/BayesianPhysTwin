from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.cli.phystwin_discrepancy as discrepancy_cli
import bayesian_phystwin.legacy_artifacts as legacy_artifacts
import bayesian_phystwin.phystwin_discrepancy as discrepancy
from bayesian_phystwin.phystwin_discrepancy import (
    PhysTwinDiscrepancyConfig,
    calibrate_phystwin_profile_discrepancy,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pickle(path: Path, value: object) -> str:
    path.write_bytes(pickle.dumps(value))
    return _sha256(path)


def _fixture(tmp_path: Path) -> tuple[Path, str, Path, str, Path, str]:
    frame_count = 6
    track_count = 2
    observed = np.empty((frame_count, track_count, 3), dtype=float)
    for frame in range(frame_count):
        observed[frame] = 0.001 * (frame + 1)
    final_data = tmp_path / "final_data.pkl"
    final_digest = _write_pickle(
        final_data,
        {
            "object_points": observed,
            "object_visibilities": np.ones((frame_count, track_count), dtype=bool),
            "object_motions_valid": np.ones((frame_count, track_count), dtype=bool),
        },
    )
    profile = tmp_path / "parameter_profile.npz"
    np.savez_compressed(
        profile,
        posterior_mean_trajectory=np.zeros_like(observed),
        epistemic_variance=np.full_like(observed, 1.0e-6),
    )
    reference = tmp_path / "reference.pkl"
    reference_digest = _write_pickle(reference, np.zeros_like(observed))
    return (
        final_data,
        final_digest,
        profile,
        _sha256(profile),
        reference,
        reference_digest,
    )


def _config() -> PhysTwinDiscrepancyConfig:
    return PhysTwinDiscrepancyConfig(
        fit_end_frame=3,
        test_start_frame=5,
        observation_variance=1.0e-4,
        decay_candidates=(0.0, 0.9),
    )


def _cli_arguments() -> list[str]:
    return [
        "calibrate-phystwin-discrepancy",
        "final_data.pkl",
        "profile.npz",
        "summary.json",
        "--final-data-sha256",
        "a" * 64,
        "--parameter-profile-sha256",
        "b" * 64,
        "--fit-end-frame",
        "3",
        "--test-start-frame",
        "5",
    ]


def test_calibration_binds_exact_inputs_without_host_paths(tmp_path: Path) -> None:
    (
        final_data,
        final_digest,
        profile,
        profile_digest,
        reference,
        reference_digest,
    ) = _fixture(tmp_path)

    summary = calibrate_phystwin_profile_discrepancy(
        final_data,
        profile,
        config=_config(),
        final_data_sha256=final_digest,
        profile_sha256=profile_digest,
        reference_trajectory_path=reference,
        reference_trajectory_sha256=reference_digest,
    )

    assert summary["schema_version"] == 2
    assert summary["inputs"] == {
        "final_data": {
            "format": "trusted_legacy_pickle_mapping",
            "sha256": final_digest,
        },
        "profile": {
            "format": "numpy_npz_no_pickle",
            "sha256": profile_digest,
        },
        "reference_trajectory": {
            "format": "trusted_legacy_pickle_ndarray",
            "sha256": reference_digest,
        },
    }
    serialized = json.dumps(summary, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert '"path"' not in serialized


def test_final_data_digest_mismatch_prevents_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_data, _, profile, profile_digest, _, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        legacy_artifacts.pickle,
        "load",
        lambda stream: pytest.fail("pickle must not load after a digest mismatch"),
    )

    with pytest.raises(ValueError, match="refusing to deserialize"):
        calibrate_phystwin_profile_discrepancy(
            final_data,
            profile,
            config=_config(),
            final_data_sha256="0" * 64,
            profile_sha256=profile_digest,
        )


def test_profile_loading_uses_the_verified_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, profile, profile_digest, _, _ = _fixture(tmp_path)
    original_load = discrepancy.np.load

    def replace_source_then_load(stream, *args, **kwargs):
        np.savez_compressed(
            profile,
            posterior_mean_trajectory=np.full((6, 2, 3), 99.0),
            epistemic_variance=np.full((6, 2, 3), 77.0),
        )
        assert kwargs["allow_pickle"] is False
        return original_load(stream, *args, **kwargs)

    monkeypatch.setattr(discrepancy.np, "load", replace_source_then_load)
    mean, epistemic = discrepancy._load_verified_profile(
        profile,
        expected_sha256=profile_digest,
    )

    np.testing.assert_array_equal(mean, np.zeros((6, 2, 3)))
    np.testing.assert_array_equal(epistemic, np.full((6, 2, 3), 1.0e-6))
    with np.load(profile, allow_pickle=False) as replacement:
        assert float(replacement["posterior_mean_trajectory"][0, 0, 0]) == 99.0


def test_profile_digest_mismatch_prevents_numpy_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_data, final_digest, profile, _, _, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        discrepancy.np,
        "load",
        lambda *args, **kwargs: pytest.fail(
            "NumPy archive must not load after a digest mismatch"
        ),
    )

    with pytest.raises(ValueError, match="profile SHA-256 mismatch"):
        calibrate_phystwin_profile_discrepancy(
            final_data,
            profile,
            config=_config(),
            final_data_sha256=final_digest,
            profile_sha256="0" * 64,
        )


def test_profile_requires_both_registered_arrays(tmp_path: Path) -> None:
    profile = tmp_path / "incomplete_profile.npz"
    np.savez_compressed(
        profile,
        posterior_mean_trajectory=np.zeros((6, 2, 3)),
    )

    with pytest.raises(ValueError, match="epistemic_variance"):
        discrepancy._load_verified_profile(
            profile,
            expected_sha256=_sha256(profile),
        )


@pytest.mark.parametrize(
    ("key", "invalid_value"),
    (
        ("object_visibilities", float("nan")),
        ("object_motions_valid", 2.0),
    ),
)
def test_masks_reject_truth_coercion(
    tmp_path: Path,
    key: str,
    invalid_value: float,
) -> None:
    final_data, _, profile, profile_digest, _, _ = _fixture(tmp_path)
    payload = pickle.loads(final_data.read_bytes())
    invalid_mask = np.asarray(payload[key], dtype=float)
    invalid_mask[0, 0] = invalid_value
    payload[key] = invalid_mask
    final_digest = _write_pickle(final_data, payload)

    with pytest.raises(ValueError, match="exact numeric 0/1 values"):
        calibrate_phystwin_profile_discrepancy(
            final_data,
            profile,
            config=_config(),
            final_data_sha256=final_digest,
            profile_sha256=profile_digest,
        )


def test_masks_reject_nonnumeric_values() -> None:
    with pytest.raises(ValueError, match="exact numeric 0/1 values"):
        discrepancy._strict_boolean_array(
            np.array(["visible"], dtype=object),
            name="mask",
        )


def test_masks_accept_exact_numeric_zero_one(tmp_path: Path) -> None:
    final_data, _, profile, profile_digest, _, _ = _fixture(tmp_path)
    payload = pickle.loads(final_data.read_bytes())
    payload["object_visibilities"] = np.asarray(
        payload["object_visibilities"],
        dtype=float,
    )
    payload["object_motions_valid"] = np.asarray(
        payload["object_motions_valid"],
        dtype=np.int64,
    )
    final_digest = _write_pickle(final_data, payload)

    summary = calibrate_phystwin_profile_discrepancy(
        final_data,
        profile,
        config=_config(),
        final_data_sha256=final_digest,
        profile_sha256=profile_digest,
    )

    assert summary["schema_version"] == 2


@pytest.mark.parametrize(
    "invalid_digest",
    (
        "A" * 64,
        "0" * 63,
        "g" * 64,
    ),
)
def test_digest_identifiers_are_strict(invalid_digest: str) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        discrepancy._validate_sha256(invalid_digest, name="artifact_sha256")


def test_reference_path_and_digest_are_conjunctive(tmp_path: Path) -> None:
    final_data, final_digest, profile, profile_digest, reference, _ = _fixture(tmp_path)

    with pytest.raises(ValueError, match="must be supplied together"):
        calibrate_phystwin_profile_discrepancy(
            final_data,
            profile,
            config=_config(),
            final_data_sha256=final_digest,
            profile_sha256=profile_digest,
            reference_trajectory_path=reference,
        )


def test_write_discrepancy_summary_is_canonical_json(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"

    discrepancy.write_discrepancy_summary(
        {"schema_version": 2, "inputs": {"profile": {"sha256": "a" * 64}}},
        output,
    )

    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 2
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_cli_requires_digest_bindings() -> None:
    parser = discrepancy_cli.build_parser()
    args = parser.parse_args(_cli_arguments()[1:])

    assert args.final_data_sha256 == "a" * 64
    assert args.parameter_profile_sha256 == "b" * 64


def test_cli_rejects_unpaired_reference_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [*_cli_arguments(), "--reference-trajectory", "reference.pkl"],
    )

    with pytest.raises(SystemExit) as error:
        discrepancy_cli.main()

    assert error.value.code == 2


def test_cli_forwards_verified_input_identities(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_calibrate(final_data, parameter_profile, **kwargs):
        captured["final_data"] = final_data
        captured["parameter_profile"] = parameter_profile
        captured.update(kwargs)
        return {"schema_version": 2}

    def fake_write(summary, output_path):
        captured["summary"] = summary
        captured["output_path"] = output_path

    monkeypatch.setattr(
        discrepancy_cli, "calibrate_phystwin_profile_discrepancy", fake_calibrate
    )
    monkeypatch.setattr(discrepancy_cli, "write_discrepancy_summary", fake_write)
    monkeypatch.setattr(sys, "argv", _cli_arguments())

    discrepancy_cli.main()

    assert captured["final_data_sha256"] == "a" * 64
    assert captured["profile_sha256"] == "b" * 64
    assert captured["reference_trajectory_path"] is None
    assert captured["reference_trajectory_sha256"] is None
    assert captured["output_path"] == "summary.json"
    assert json.loads(capsys.readouterr().out) == {"schema_version": 2}
