import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.phystwin_tapnextpp_competence import file_sha256
from bayesian_phystwin.tapnextpp_sparse_assimilation import (
    SparseAssimilationConfig,
)


def _load_staging_script():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "remote"
        / "prepare_phystwin_tapnextpp_sparse_assimilation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "tapnextpp_material_transport_assimilation_staging_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _protocol() -> dict:
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "sota"
        / "phystwin_tapnextpp_material_transport_assimilation_source_v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_material_transport_assimilation_protocol_is_locked() -> None:
    staging = _load_staging_script()
    protocol = _protocol()

    staging._validate_protocol(protocol)
    assert len(protocol["fixed_source_cases"]) == 14
    assert protocol["sparse_assimilation_mode"] == (
        "fixed_frame_zero_material_displacement"
    )
    SparseAssimilationConfig(**protocol["sparse_assimilation_config"])


def test_material_transport_protocol_rejects_case_panel_change() -> None:
    staging = _load_staging_script()
    protocol = _protocol()
    protocol["fixed_source_cases"] = protocol["fixed_source_cases"][:-1]

    with pytest.raises(ValueError, match="case panel changed"):
        staging._validate_protocol(protocol)


def test_material_attachment_is_immutable_and_identity_aligned(
    tmp_path: Path,
) -> None:
    staging = _load_staging_script()
    identity_ids = np.asarray([3, 8], dtype=np.int64)
    node_indices = np.asarray([2, 5], dtype=np.int64)
    attachment = tmp_path / staging.MATERIAL_ATTACHMENT_FILENAME
    np.savez_compressed(
        attachment,
        identity_ids=identity_ids,
        material_node_indices=node_indices,
        frame_zero_attachment_distance_m=np.asarray(
            [0.002, 0.004],
            dtype=np.float32,
        ),
    )
    source_record = {
        "material_attachment_sha256": file_sha256(attachment),
        "material_node_indices": node_indices.tolist(),
    }

    result = staging._validate_material_attachment(
        tmp_path,
        source_record,
        identity_ids,
        node_count=6,
    )

    np.testing.assert_array_equal(result["node_indices"], node_indices)
    np.testing.assert_allclose(result["distance_m"], [0.002, 0.004])

    with pytest.raises(ValueError, match="identities differ"):
        staging._validate_material_attachment(
            tmp_path,
            source_record,
            identity_ids[::-1],
            node_count=6,
        )
