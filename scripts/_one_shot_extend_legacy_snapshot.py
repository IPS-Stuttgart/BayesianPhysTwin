#!/usr/bin/env python3
"""Propagate verified legacy mappings into downstream correspondence builders."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _replace(path: str, old: str, new: str, *, expected_count: int = 1) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected_count:
        raise SystemExit(
            f"{path}: expected {expected_count} occurrence(s), found {count}: {old!r}"
        )
    target.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    _replace(
        "src/bayesian_phystwin/phystwin_raw_cues.py",
        "import pickle\nfrom dataclasses import asdict, dataclass\n",
        "import pickle\nfrom collections.abc import Mapping\n"
        "from dataclasses import asdict, dataclass\n",
    )
    _replace(
        "src/bayesian_phystwin/phystwin_raw_cues.py",
        """def load_phystwin_raw_track_map(
    final_data_path: str | Path,
    raw_case_dir: str | Path,
    *,
    config: PhysTwinRawCueConfig | None = None,
) -> PhysTwinRawTrackMap:
    \"\"\"Recover the release preprocessing's exact raw-query correspondence.\"\"\"
""",
        """def load_phystwin_raw_track_map(
    final_data_path: str | Path,
    raw_case_dir: str | Path,
    *,
    config: PhysTwinRawCueConfig | None = None,
    final_data_payload: Mapping[str, Any] | None = None,
) -> PhysTwinRawTrackMap:
    \"\"\"Recover the release preprocessing's exact raw-query correspondence.

    ``final_data_payload`` lets digest-bound callers pass the exact mapping they
    already verified and deserialized. When omitted, the historical path-based
    behavior is retained for ordinary development callers.
    \"\"\"
""",
    )
    _replace(
        "src/bayesian_phystwin/phystwin_raw_cues.py",
        "    final_data = _load_pickle(final_data_path)\n",
        """    final_data = (
        _load_pickle(final_data_path)
        if final_data_payload is None
        else final_data_payload
    )
    if not isinstance(final_data, Mapping):
        raise TypeError("final_data_payload must contain a mapping")
""",
    )

    _replace(
        "src/bayesian_phystwin/causal4d_artifacts_v1.py",
        """    load_trusted_legacy_phystwin_pickle(
        final_data_path,
        expected_sha256=final_data_sha256,
        artifact_kind="mapping",
        required_keys=("object_points", "object_visibilities"),
    )
""",
        """    final_data = load_trusted_legacy_phystwin_pickle(
        final_data_path,
        expected_sha256=final_data_sha256,
        artifact_kind="mapping",
        required_keys=("object_points", "object_visibilities"),
    )
""",
    )
    _replace(
        "src/bayesian_phystwin/causal4d_artifacts_v1.py",
        """    mapping = module.load_phystwin_raw_track_map(
        final_data_path,
        raw_case_dir,
        config=config,
    )
""",
        """    mapping = module.load_phystwin_raw_track_map(
        final_data_path,
        raw_case_dir,
        config=config,
        final_data_payload=final_data,
    )
""",
    )

    _replace(
        "src/bayesian_phystwin/causal4d_artifacts_v2.py",
        """    mapping = module.load_phystwin_raw_track_map(
        final_path,
        raw_path,
        config=config,
    )
""",
        """    mapping = module.load_phystwin_raw_track_map(
        final_path,
        raw_path,
        config=config,
        final_data_payload=final_data,
    )
""",
    )

    fake_signature = (
        "def load_phystwin_raw_track_map(final_data_path, raw_case_dir, *, config):"
    )
    expanded_signature = """def load_phystwin_raw_track_map(
            final_data_path,
            raw_case_dir,
            *,
            config,
            final_data_payload,
        ):"""
    _replace(
        "tests/test_legacy_artifacts.py",
        fake_signature,
        expanded_signature,
        expected_count=2,
    )
    _replace(
        "tests/test_legacy_artifacts.py",
        """            assert isinstance(config, FakeConfig)
            return FakeMapping()
""",
        """            assert isinstance(config, FakeConfig)
            assert final_data_payload is not None
            return FakeMapping()
""",
        expected_count=2,
    )
    _replace(
        "tests/test_causal4d_artifacts_v2.py",
        fake_signature,
        expanded_signature,
    )
    _replace(
        "tests/test_causal4d_artifacts_v2.py",
        """            assert isinstance(config, FakeConfig)
            return FakeMapping()
""",
        """            assert isinstance(config, FakeConfig)
            assert final_data_payload is not None
            return FakeMapping()
""",
    )

    _replace(
        "tests/test_phystwin_raw_cues.py",
        "import numpy as np\n\nfrom bayesian_phystwin.phystwin_raw_cues import (\n",
        "import numpy as np\nimport pytest\n\n"
        "import bayesian_phystwin.phystwin_raw_cues as raw_cues\n"
        "from bayesian_phystwin.phystwin_raw_cues import (\n",
    )
    raw_test = """


def test_preloaded_final_data_avoids_pickle_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    (raw / "cotracker").mkdir(parents=True)
    (raw / "pcd").mkdir()
    np.savez(
        raw / "cotracker" / "0.npz",
        tracks=np.zeros((1, 1, 2)),
        visibility=np.ones((1, 1), dtype=bool),
    )
    np.savez(raw / "pcd" / "0.npz", points=np.zeros((1, 1, 1, 3)))
    final_data = {
        "object_points": np.zeros((1, 1, 3)),
        "object_visibilities": np.ones((1, 1), dtype=bool),
    }
    final_path = tmp_path / "final.pkl"
    final_path.write_bytes(b"must not be opened")
    monkeypatch.setattr(
        raw_cues,
        "_load_pickle",
        lambda _path: pytest.fail("verified final data must not be reopened"),
    )

    mapping = load_phystwin_raw_track_map(
        final_path,
        raw,
        final_data_payload=final_data,
    )

    np.testing.assert_array_equal(mapping.final_points, final_data["object_points"])
    np.testing.assert_array_equal(mapping.source_camera, [0])
    np.testing.assert_array_equal(mapping.source_track, [0])
"""
    target = ROOT / "tests/test_phystwin_raw_cues.py"
    text = target.read_text(encoding="utf-8")
    if "test_preloaded_final_data_avoids_pickle_reopen" in text:
        raise SystemExit("raw-cue regression test already exists")
    target.write_text(text.rstrip() + raw_test + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
