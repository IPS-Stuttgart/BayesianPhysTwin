import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def _load_runner():
    path = Path(__file__).parents[1] / "scripts" / "remote" / "run_matphys_causal.py"
    spec = importlib.util.spec_from_file_location("run_matphys_causal_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ddp_runtime_access_logs_merge_disjoint_case_shards(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    frame_a = tmp_path / "0.png"
    frame_b = tmp_path / "1.png"
    frame_a.write_bytes(b"a")
    frame_b.write_bytes(b"b")

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    runner._ACCESSED_FRAMES.clear()
    runner._ACCESSED_FRAMES["case_b"] = {1}
    runner._ACCESSED_FRAME_PATHS.clear()
    runner._ACCESSED_FRAME_PATHS["case_b"] = {1: frame_b}
    runner._OBJECTIVE_END_FRAMES.clear()
    runner._OBJECTIVE_END_FRAMES["case_b"] = 8
    assert runner._collect_distributed_access_logs(tmp_path) is None

    monkeypatch.setenv("RANK", "0")
    runner._ACCESSED_FRAMES.clear()
    runner._ACCESSED_FRAMES["case_a"] = {0}
    runner._ACCESSED_FRAME_PATHS.clear()
    runner._ACCESSED_FRAME_PATHS["case_a"] = {0: frame_a}
    runner._OBJECTIVE_END_FRAMES.clear()
    runner._OBJECTIVE_END_FRAMES["case_a"] = 6
    merged = runner._collect_distributed_access_logs(tmp_path)

    assert merged is not None
    frames, paths, objectives, logs, optimizer_summaries = merged
    assert frames == {"case_a": {0}, "case_b": {1}}
    assert paths == {
        "case_a": {0: frame_a.resolve()},
        "case_b": {1: frame_b.resolve()},
    }
    assert objectives == {"case_a": 6, "case_b": 8}
    assert len(logs) == 2
    assert optimizer_summaries == [
        {
            "accepted_steps": 0,
            "attempted_steps": 0,
            "rejected_post_step": 0,
            "rejected_pre_step": 0,
        },
        {
            "accepted_steps": 0,
            "attempted_steps": 0,
            "rejected_post_step": 0,
            "rejected_pre_step": 0,
        },
    ]


def test_open3d_stub_kdtree_matches_distance_then_index_order() -> None:
    runner = _load_runner()
    original = sys.modules.get("open3d")
    try:
        runner._install_open3d_stub()
        import open3d

        cloud = open3d.geometry.PointCloud()
        cloud.points = open3d.utility.Vector3dVector(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.25, 0.0]]
        )
        tree = open3d.geometry.KDTreeFlann(cloud)

        count, indices, distance_sq = tree.search_knn_vector_3d([0, 0, 0], 3)
        assert count == 3
        assert indices == [2, 0, 1]
        np.testing.assert_allclose(distance_sq, [0.0625, 1.0, 1.0])

        count, indices, distance_sq = tree.search_hybrid_vector_3d(
            [0, 0, 0], 1.0, 2
        )
        assert count == 2
        assert indices == [2, 0]
        np.testing.assert_allclose(distance_sq, [0.0625, 1.0])
    finally:
        if original is None:
            sys.modules.pop("open3d", None)
        else:
            sys.modules["open3d"] = original


def test_uneven_ddp_authority_uses_last_rank_with_most_forwards() -> None:
    runner = _load_runner()

    assert runner._authoritative_uneven_ddp_rank([993, 994]) == 1
    assert runner._authoritative_uneven_ddp_rank([994, 993]) == 0
    assert runner._authoritative_uneven_ddp_rank([994, 994]) == 1


def test_distributed_training_uses_rank_local_simulator_device(monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")

    assert runner._rank_local_training_device("cuda:0") == "cuda:1"

    monkeypatch.setenv("WORLD_SIZE", "1")
    assert runner._rank_local_training_device("cuda:7") == "cuda:7"


def test_distributed_wrapper_forces_unused_parameter_detection() -> None:
    runner = _load_runner()
    received = {}

    def ddp_factory(*args, **kwargs):
        received.update(kwargs)
        return args

    training = SimpleNamespace(DDP=ddp_factory)
    runner._enable_unused_parameter_ddp(training)

    training.DDP("model", find_unused_parameters=False)

    assert received["find_unused_parameters"] is True


def test_transactional_optimizer_rolls_back_nonfinite_update() -> None:
    torch = pytest.importorskip("torch")

    class BadOptimizer(torch.optim.Optimizer):
        def __init__(self, params):
            super().__init__(params, {})

        @torch.no_grad()
        def step(self, closure=None):
            del closure
            for group in self.param_groups:
                for parameter in group["params"]:
                    parameter.fill_(float("nan"))
                    self.state[parameter]["moment"] = torch.tensor(float("nan"))

    diagnostics = {
        "attempted_steps": 0,
        "accepted_steps": 0,
        "rejected_pre_step": 0,
        "rejected_post_step": 0,
    }
    guarded_type = _load_runner()._transactional_finite_optimizer(
        BadOptimizer, diagnostics
    )
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([2.0])
    optimizer = guarded_type([parameter])

    optimizer.step()

    torch.testing.assert_close(parameter, torch.tensor([1.0]))
    assert optimizer.state == {}
    assert diagnostics == {
        "attempted_steps": 1,
        "accepted_steps": 0,
        "rejected_pre_step": 0,
        "rejected_post_step": 1,
    }


def test_single_backward_auxiliary_hook_adds_teacher_gradient_once() -> None:
    torch = pytest.importorskip("torch")
    runner = _load_runner()
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    applied = parameter * 1.0
    training = SimpleNamespace(
        _rollout_aux_loss=lambda *args: (
            (args[0]["log_k"] - 1.0).pow(2).sum(),
            {"teacher_log_k": 1.0},
        )
    )
    runner._install_single_backward_auxiliary_loss(training)

    auxiliary, _ = training._rollout_aux_loss(
        {"log_k": applied}, None, 0, None, None, 0
    )
    (3.0 * applied).sum().backward()

    assert auxiliary.requires_grad is False
    torch.testing.assert_close(parameter.grad, torch.tensor([5.0]))
    assert (
        training._single_backward_auxiliary_contract
        == runner.SINGLE_BACKWARD_AUXILIARY_CONTRACT
    )


def test_checkpoint_finiteness_report_detects_model_and_optimizer_nan(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "model_state_dict": {"weight": torch.tensor([1.0, float("nan")])},
            "optimizer_state_dict": {
                "state": {0: {"moment": torch.tensor([float("inf")])}}
            },
        },
        checkpoint,
    )

    report = _load_runner()._checkpoint_finiteness_report(checkpoint)

    assert report["finite"] is False
    assert report["model_nonfinite_count"] == 1
    assert report["optimizer_nonfinite_count"] == 1


def test_model_spring_y_exports_complete_positive_field() -> None:
    torch = pytest.importorskip("torch")
    runner = _load_runner()
    expected_logk = torch.log(torch.tensor([1000.0, 2000.0, 3000.0]))
    training = SimpleNamespace(
        _build_model_logk=lambda model_out, runtime, sim, device: (
            expected_logk
        )
    )
    runtime = SimpleNamespace(
        sim=SimpleNamespace(wp_spring_Y=SimpleNamespace(shape=(3,)))
    )

    model_logk, spring_y = runner._model_spring_y(
        training, runtime, {"log_k": expected_logk}, "cpu"
    )

    torch.testing.assert_close(model_logk, expected_logk)
    assert spring_y.dtype == np.float32
    np.testing.assert_allclose(
        spring_y, [1000.0, 2000.0, 3000.0], rtol=1e-6
    )


def test_model_spring_y_rejects_topology_mismatch() -> None:
    torch = pytest.importorskip("torch")
    runner = _load_runner()
    training = SimpleNamespace(
        _build_model_logk=lambda model_out, runtime, sim, device: torch.zeros(2)
    )
    runtime = SimpleNamespace(
        sim=SimpleNamespace(wp_spring_Y=SimpleNamespace(shape=(3,)))
    )

    with pytest.raises(RuntimeError, match="invalid complete spring field"):
        runner._model_spring_y(training, runtime, {}, "cpu")
