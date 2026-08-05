import numpy as np
import pytest

pytest.importorskip("torch")
wp = pytest.importorskip("warp")

from bayesian_phystwin._phystwin_warp_backend import (  # noqa: E402
    expand_sparse_spring_basis_log_y,
)


@wp.kernel
def _squared_loss(
    values: wp.array(dtype=wp.float32),
    targets: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
):
    index = wp.tid()
    residual = values[index] - targets[index]
    wp.atomic_add(loss, 0, 0.5 * residual * residual)


def _loss(
    coefficients: np.ndarray,
    reference: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    targets: np.ndarray,
) -> float:
    output = wp.zeros(len(reference), dtype=wp.float32, device="cpu")
    loss = wp.zeros(1, dtype=wp.float32, device="cpu")
    wp.launch(
        expand_sparse_spring_basis_log_y,
        dim=len(reference),
        inputs=[
            wp.array(reference, dtype=wp.float32, device="cpu"),
            wp.array(indices.reshape(-1), dtype=wp.int32, device="cpu"),
            wp.array(weights.reshape(-1), dtype=wp.float32, device="cpu"),
            wp.array(coefficients, dtype=wp.float32, device="cpu"),
            indices.shape[1],
        ],
        outputs=[output],
        device="cpu",
    )
    wp.launch(
        _squared_loss,
        dim=len(reference),
        inputs=[
            output,
            wp.array(targets, dtype=wp.float32, device="cpu"),
        ],
        outputs=[loss],
        device="cpu",
    )
    return float(loss.numpy()[0])


def test_sparse_triplane_warp_gradient_matches_finite_difference_direction():
    reference = np.array([0.2, -0.3, 0.1], dtype=np.float32)
    indices = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    weights = np.array([[0.75, 0.25], [0.4, 0.6], [0.2, 0.8]], dtype=np.float32)
    coefficients_value = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    direction = np.array([0.3, -0.5, 0.2, 0.7], dtype=np.float32)
    targets = np.array([0.0, 0.25, -0.2], dtype=np.float32)

    coefficients = wp.array(
        coefficients_value,
        dtype=wp.float32,
        device="cpu",
        requires_grad=True,
    )
    output = wp.zeros(
        len(reference),
        dtype=wp.float32,
        device="cpu",
        requires_grad=True,
    )
    loss = wp.zeros(1, dtype=wp.float32, device="cpu", requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(
            expand_sparse_spring_basis_log_y,
            dim=len(reference),
            inputs=[
                wp.array(reference, dtype=wp.float32, device="cpu"),
                wp.array(indices.reshape(-1), dtype=wp.int32, device="cpu"),
                wp.array(weights.reshape(-1), dtype=wp.float32, device="cpu"),
                coefficients,
                indices.shape[1],
            ],
            outputs=[output],
            device="cpu",
        )
        wp.launch(
            _squared_loss,
            dim=len(reference),
            inputs=[
                output,
                wp.array(targets, dtype=wp.float32, device="cpu"),
            ],
            outputs=[loss],
            device="cpu",
        )
    tape.backward(loss)

    analytic = float(np.dot(coefficients.grad.numpy(), direction))
    epsilon = 1.0e-3
    finite_difference = (
        _loss(
            coefficients_value + epsilon * direction,
            reference,
            indices,
            weights,
            targets,
        )
        - _loss(
            coefficients_value - epsilon * direction,
            reference,
            indices,
            weights,
            targets,
        )
    ) / (2.0 * epsilon)

    assert abs(analytic) > 1.0e-4
    assert np.sign(analytic) == np.sign(finite_difference)
    assert analytic == pytest.approx(finite_difference, rel=5.0e-3, abs=1.0e-5)
