"""Vector 3D Snell's law refraction and total internal reflection."""

from __future__ import annotations
import numpy as np
from typing import Tuple


def refract_ray_bundle(
    direction: np.ndarray,
    normal: np.ndarray,
    n1: float,
    n2: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Applies vectorized 3D Snell's law of refraction.

    Returns: (new_direction of shape (N, 3), tir_mask of shape (N,))
    """
    d = direction
    n = normal
    eta = n1 / n2

    # Ensure normal opposes incoming ray
    d_dot_n = np.sum(d * n, axis=1, keepdims=True)
    n_orient = np.where(d_dot_n > 0.0, -n, n)
    cos_theta1 = -np.sum(d * n_orient, axis=1, keepdims=True)
    cos_theta1 = np.clip(cos_theta1, -1.0, 1.0)

    # Snell's radicand: 1 - eta^2 * (1 - cos^2(theta1))
    sin2_theta2 = (eta * eta) * (1.0 - cos_theta1 * cos_theta1)
    tir_mask = (sin2_theta2 > 1.0).ravel()

    # Compute refracted direction
    cos_theta2 = np.sqrt(np.maximum(0.0, 1.0 - sin2_theta2))
    refracted = eta * d + (eta * cos_theta1 - cos_theta2) * n_orient

    # Normalize
    norms = np.linalg.norm(refracted, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    refracted /= norms

    return refracted, tir_mask
