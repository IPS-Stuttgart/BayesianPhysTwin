import importlib.util
import json
import sys
from pathlib import Path

import pytest

from bayesian_phystwin.deform_dlo_alltrain import (
    load_deform_dlo2_alltrain_protocol,
    validate_deform_dlo2_alltrain_authorization,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "configs" / "sota" / "deform_dlo2_alltrain_refit_v1.json"
RUNNER = REPOSITORY_ROOT / "scripts" / "remote" / "run_deform_dlo2_alltrain.py"


def _load_runner():
    scripts_root = str(RUNNER.parent)
    sys.path.insert(0, scripts_root)
    try:
        spec = importlib.util.spec_from_file_location(
            "deform_dlo2_alltrain_runner",
            RUNNER,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts_root)


def _parents() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    selected_spec = {
        "operator": "predictive_mean",
        "weights": {"6040": 0.4, "6400": 0.6},
    }
    selection_record = {
        "selected_arm": "predictive_mean::tail_2_uniform",
        "fallback_used": False,
    }
    source = {
        "contract": "deform-dlo-source-reproduction-result-v1",
        "official_eval_read": False,
        "advancement_authorized": True,
        "source_gate": {"passed": True},
    }
    posterior = {
        "contract": "deform-dlo2-posterior-result-v1",
        "official_eval_read": False,
        "exact_fallback": False,
        "identical_information_official_eval_authorized": True,
        "selected_arm": "predictive_mean::tail_2_uniform",
        "selected_spec": selected_spec,
        "selection": selection_record,
        "uncertainty": {
            "validation_fitted_variance_scale": 2.0,
            "variance_floor_m2": 0.000025,
            "nominal_coordinate_coverage": 0.9,
        },
    }
    selection = {
        "contract": "deform-dlo2-posterior-selection-v1",
        "official_eval_read": False,
        "source_result": {"sha256": "a" * 64},
        "protocol": {"sha256": "b" * 64},
        "selection": selection_record,
        "candidate_specs": {
            "predictive_mean::tail_2_uniform": selected_spec,
        },
    }
    return source, posterior, selection


def test_dlo2_alltrain_protocol_uses_all_training_data_and_no_reselection() -> None:
    protocol = load_deform_dlo2_alltrain_protocol(PROTOCOL)

    assert protocol["data"]["trajectory_count"] == 56
    assert protocol["data"]["use_every_train_trajectory"] is True
    assert protocol["data"]["official_eval_read"] is False
    assert protocol["method_transfer"]["validation_reselection"] is False
    assert protocol["method_transfer"]["target_reselection"] is False
    assert protocol["checkpoint_updates"][-1] == 6400


def test_dlo2_alltrain_protocol_rejects_target_reselection(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["method_transfer"]["target_reselection"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="transfer contract"):
        load_deform_dlo2_alltrain_protocol(changed)


def test_dlo2_alltrain_authorization_preserves_selected_spec() -> None:
    protocol = load_deform_dlo2_alltrain_protocol(PROTOCOL)
    source, posterior, selection = _parents()

    selected = validate_deform_dlo2_alltrain_authorization(
        protocol,
        source,
        posterior,
        selection,
        source_protocol_sha256="b" * 64,
        source_result_sha256="a" * 64,
    )

    assert selected["operator"] == "predictive_mean"
    assert selected["weights"] == {6040: 0.4, 6400: 0.6}
    assert selected["validation_fitted_variance_scale"] == 2.0


def test_dlo2_alltrain_authorization_rejects_fallback() -> None:
    protocol = load_deform_dlo2_alltrain_protocol(PROTOCOL)
    source, posterior, selection = _parents()
    posterior["exact_fallback"] = True

    with pytest.raises(ValueError, match="did not authorize"):
        validate_deform_dlo2_alltrain_authorization(
            protocol,
            source,
            posterior,
            selection,
            source_protocol_sha256="b" * 64,
            source_result_sha256="a" * 64,
        )


def test_dlo2_alltrain_authorization_rejects_a_changed_selected_spec() -> None:
    protocol = load_deform_dlo2_alltrain_protocol(PROTOCOL)
    source, posterior, selection = _parents()
    posterior["selected_spec"] = {
        "operator": "parameter_mean",
        "weights": {"6040": 0.4, "6400": 0.6},
    }

    with pytest.raises(ValueError, match="does not match"):
        validate_deform_dlo2_alltrain_authorization(
            protocol,
            source,
            posterior,
            selection,
            source_protocol_sha256="b" * 64,
            source_result_sha256="a" * 64,
        )


def test_dlo2_alltrain_progress_writer_updates_only_mutable_outputs(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    progress = tmp_path / "progress.json"

    runner._write_json(progress, {"update": 1}, immutable=False)
    runner._write_json(progress, {"update": 10}, immutable=False)
    assert json.loads(progress.read_text(encoding="utf-8")) == {"update": 10}

    sealed = tmp_path / "sealed.json"
    runner._write_json(sealed, {"value": 1})
    with pytest.raises(RuntimeError, match="locked output differs"):
        runner._write_json(sealed, {"value": 2})
