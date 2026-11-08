"""Differentiable optimization of optical surfaces to eliminate aberrations."""

from __future__ import annotations
import numpy as np
from typing import Callable, List, Tuple
from .ray import RayBundle
from .system import OpticalSystem


class LensOptimizer:
    """Optimizes optical surface curvatures, conics, and aspherics to minimize RMS spot size."""

    def __init__(
        self,
        system: OpticalSystem,
        ray_generator: Callable[[], RayBundle],
        param_targets: List[Tuple[int, str]],  # List of (surface_index, attribute_name)
    ):
        self.system = system
        self.ray_generator = ray_generator
        self.param_targets = param_targets

    def _get_params(self) -> np.ndarray:
        vals = []
        for s_idx, attr in self.param_targets:
            surf = self.system.surfaces[s_idx]
            vals.append(getattr(surf, attr))
        return np.array(vals, dtype=np.float64)

    def _set_params(self, params: np.ndarray) -> None:
        for (s_idx, attr), val in zip(self.param_targets, params):
            surf = self.system.surfaces[s_idx]
            setattr(surf, attr, float(val))

    def evaluate_loss(self, params: np.ndarray) -> float:
        """Evaluates RMS spot radius for given parameter vector."""
        self._set_params(params)
        rays = self.ray_generator()
        self.system.trace(rays)
        return self.system.compute_rms_spot_radius(rays)

    def compute_gradient(self, params: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Computes numerical gradient dLoss/dParams via central finite differences."""
        grad = np.zeros_like(params)
        for i in range(len(params)):
            p_plus = params.copy()
            p_minus = params.copy()
            h = max(eps, abs(params[i]) * 1e-4) if abs(params[i]) > 1e-4 else eps

            p_plus[i] += h
            p_minus[i] -= h

            loss_plus = self.evaluate_loss(p_plus)
            loss_minus = self.evaluate_loss(p_minus)

            grad[i] = (loss_plus - loss_minus) / (2.0 * h)

        return grad

    def optimize_adam(
        self,
        learning_rate: float = 0.01,
        iterations: int = 50,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> Tuple[np.ndarray, List[float]]:
        """Minimizes spot radius using the Adam gradient descent algorithm."""
        params = self._get_params()
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        loss_history = []

        for step in range(1, iterations + 1):
            loss = self.evaluate_loss(params)
            loss_history.append(loss)

            grad = self.compute_gradient(params)

            # Gradient clipping to prevent optical surface explosion
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 10.0:
                grad = (grad / grad_norm) * 10.0

            # Adam moment updates
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad ** 2)

            m_hat = m / (1.0 - beta1 ** step)
            v_hat = v / (1.0 - beta2 ** step)

            # Parameter update
            params -= (learning_rate / (np.sqrt(v_hat) + epsilon)) * m_hat

        self._set_params(params)
        final_loss = self.evaluate_loss(params)
        loss_history.append(final_loss)

        return params, loss_history
