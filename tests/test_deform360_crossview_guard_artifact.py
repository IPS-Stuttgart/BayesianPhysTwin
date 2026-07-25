from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    canonical_sha256,
    file_sha256,
)
from bayesian_phystwin.deform360_crossview_guard_artifact import (
    ARCHIVE_FILENAME,
    ARTIFACT_KIND,
    PROTOCOL_ID,
    REPORT_FILENAME,
    build_crossview_guard_prediction,
    load_crossview_guard_prediction,
)


def _write_artifact(root: Path) -> None:
    root.mkdir()
    archive = root / ARCHIVE_FILENAME
    baseline = np.zeros((4, 6, 3), dtype=np.float32)
    np.savez_compressed(
        archive,
        baseline_m=baseline,
        crossview_guarded_m=baseline.copy(),
        center_ids=np.asarray([0, 3, 5]),
        update_frames=np.asarray([1, 2]),
    )
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "output": {"archive_file_sha256": file_sha256(archive)},
    }
    report["result_sha256"] = canonical_sha256(
        report, digest_key="result_sha256"
    )
    (root / REPORT_FILENAME).write_text(
        json.dumps(report, sort_keys=True), encoding="utf-8"
    )


def test_prediction_builder_accepts_no_target_or_outcome() -> None:
    parameters = inspect.signature(build_crossview_guard_prediction).parameters

    assert "target" not in parameters
    assert "outcome" not in parameters


def test_guarded_prediction_loads_only_checksummed_arrays(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)

    report, arrays = load_crossview_guard_prediction(artifact)

    assert report["protocol_id"] == PROTOCOL_ID
    assert arrays["crossview_guarded_m"].shape == (4, 6, 3)


def test_guarded_prediction_rejects_mutated_archive(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)
    with (artifact / ARCHIVE_FILENAME).open("ab") as stream:
        stream.write(b"changed")

    with pytest.raises(ValueError, match="archive checksum"):
        load_crossview_guard_prediction(artifact)
