import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_runner():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "remote"
        / "run_matphys_transductive_reconstruction.py"
    )
    spec = importlib.util.spec_from_file_location("matphys_transductive_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _common_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        case="case_a",
        output_dir=str(tmp_path / "output"),
        data_root=str(tmp_path / "data"),
        experiments_dir=str(tmp_path / "experiments"),
        experiments_optimization_dir=str(tmp_path / "optimization"),
        case_to_material=str(tmp_path / "case_to_material.json"),
        results_dir=str(tmp_path / "results"),
        sem_cache_dir=str(tmp_path / "cache"),
        device="cuda:0",
        epochs=200,
        eval_every=10,
        videomae_model="MCG-NJU/videomae-base",
    )


def test_future_observation_acknowledgement_is_mandatory() -> None:
    runner = _load_runner()

    with pytest.raises(ValueError, match="acknowledge-future-observations"):
        runner._require_acknowledgement(False)
    runner._require_acknowledgement(True)


def test_working_directory_is_restored(tmp_path: Path) -> None:
    runner = _load_runner()
    original = Path.cwd()
    target = tmp_path / "target"
    target.mkdir()

    with runner._working_directory(target):
        assert Path.cwd() == target

    assert Path.cwd() == original


def test_lexicographic_selection_exactly_matches_upstream(tmp_path: Path) -> None:
    runner = _load_runner()
    color = tmp_path / "case_a" / "color" / "0"
    color.mkdir(parents=True)
    for name in ("0.png", "1.png", "2.png", "10.png", "11.png"):
        (color / name).write_bytes(name.encode("ascii"))

    selected = runner._lexicographic_frame_selection(tmp_path, "case_a", 3)

    assert [record["filename"] for record in selected] == [
        "0.png",
        "10.png",
        "2.png",
    ]
    assert [record["lexicographic_position"] for record in selected] == [0, 2, 4]
    assert [record["slot"] for record in selected] == [0, 1, 2]


def test_training_argv_is_explicitly_transductive_and_matches_recipe(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    argv = runner._training_argv(_common_args(tmp_path))

    assert "--fit_all_frames" in argv
    assert argv[argv.index("--lambda_track") + 1] == "1.0"
    assert argv[argv.index("--lambda_geo") + 1] == "1.0"
    assert argv[argv.index("--lambda_acc_smooth") + 1] == "0.01"
    assert argv[argv.index("--grad_scale") + 1] == "1000.0"
    assert argv[argv.index("--logk_soft_clamp") + 1] == "0.25"
    assert argv[argv.index("--epochs") + 1] == "200"
    assert "--save_best_only" in argv


def test_training_schedule_must_write_terminal_checkpoint() -> None:
    runner = _load_runner()

    runner._validate_training_schedule(200, 10)
    runner._validate_training_schedule(1, 1)
    with pytest.raises(ValueError, match="divisible"):
        runner._validate_training_schedule(3, 2)


def test_checkpoint_namespace_keeps_all_frame_semantics(tmp_path: Path) -> None:
    runner = _load_runner()
    args = _common_args(tmp_path)

    namespace = runner._checkpoint_namespace(
        {"fit_all_frames": False, "logk_residual_scale": 1.0}, args
    )

    assert namespace.fit_all_frames is True
    assert namespace.lambda_render == 0.0
    assert namespace.base_path == str(Path(args.data_root).resolve())


def test_audit_validation_rejects_missing_future_disclosure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    args = _common_args(tmp_path)
    args.matphys_root = str(tmp_path / "matphys")
    Path(args.matphys_root).mkdir()
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "contract": runner.TRANSDUCTIVE_CONTRACT,
                "future_observations_used": False,
                "released_test_outcomes_used_in_objective": True,
                "claim_boundary": runner.CLAIM_BOUNDARY,
                "case_name": args.case,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "_validated_source_commit", lambda path: "source")

    with pytest.raises(ValueError, match="future observations"):
        runner._load_training_audit(audit_path, args)


def test_source_validation_requires_pinned_clean_commit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "_source_commit", lambda path: "wrong")

    with pytest.raises(ValueError, match="expected pinned"):
        runner._validated_source_commit(tmp_path)

    monkeypatch.setattr(
        runner, "_source_commit", lambda path: runner.PINNED_MATPHYS_COMMIT
    )
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda *args, **kwargs: " M semantic/train_models.py\n",
    )
    with pytest.raises(ValueError, match="tracked source tree is dirty"):
        runner._validated_source_commit(tmp_path)


def test_identity_validation_detects_mutation(tmp_path: Path) -> None:
    runner = _load_runner()
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"before")
    identity = runner._file_identity(artifact)
    runner._validate_file_identity(identity, "artifact")

    artifact.write_bytes(b"after")
    with pytest.raises(ValueError, match="changed after training"):
        runner._validate_file_identity(identity, "artifact")


def test_parser_does_not_imply_acknowledgement(tmp_path: Path) -> None:
    runner = _load_runner()
    common = [
        "--matphys-root",
        str(tmp_path / "matphys"),
        "--data-root",
        str(tmp_path / "data"),
        "--experiments-dir",
        str(tmp_path / "experiments"),
        "--experiments-optimization-dir",
        str(tmp_path / "optimization"),
        "--case-to-material",
        str(tmp_path / "mapping.json"),
        "--results-dir",
        str(tmp_path / "results"),
        "--sem-cache-dir",
        str(tmp_path / "cache"),
        "--case",
        "case_a",
    ]

    args = runner._parse_args(["train", *common, "--output-dir", str(tmp_path / "out")])

    assert args.acknowledge_future_observations is False
