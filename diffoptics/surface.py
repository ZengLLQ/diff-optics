"""Optical surfaces: planar, spherical, conic, and high-order aspheric."""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from .ray import RayBundle


class OpticalSurface:
    """Represents a refractive or reflective optical surface in 3D."""

    def __init__(
        self,
        z_position: float,
        curvature: float = 0.0,      # c = 1/R [mm^-1]
        conic: float = 0.0,          # k (0 = spherical, -1 = paraboloid)
        aspheric_a4: float = 0.0,    # r^4 coefficient
        aspheric_a6: float = 0.0,    # r^6 coefficient
        clear_aperture: float = 25.0,# Semi-diameter [mm]
        refractive_index: float = 1.0,
    ):
        self.z_position = float(z_position)
        self.curvature = float(curvature)
        self.conic = float(conic)
        self.aspheric_a4 = float(aspheric_a4)
        self.aspheric_a6 = float(aspheric_a6)
        self.clear_aperture = float(clear_aperture)
        self.refractive_index = float(refractive_index)

    def sag(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculates surface sag z(x, y) relative to vertex."""
        r2 = x * x + y * y
        c = self.curvature
        k = self.conic

        if abs(c) < 1e-12:
            base_sag = np.zeros_like(r2)
        else:
            arg = 1.0 - (1.0 + k) * (c * c) * r2
            arg = np.maximum(0.0, arg)
            base_sag = (c * r2) / (1.0 + np.sqrt(arg))

        higher_order = self.aspheric_a4 * (r2 ** 2) + self.aspheric_a6 * (r2 ** 3)
        return base_sag + higher_order

    def surface_normal(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Computes inward unit normal vector field n = (-dz/dx, -dz/dy, 1) / norm."""
        r2 = x * x + y * y
        c = self.curvature
        k = self.conic

        if abs(c) < 1e-12:
            dz_dr2 = np.zeros_like(r2)
        else:
            arg = np.maximum(1e-12, 1.0 - (1.0 + k) * (c * c) * r2)
            sqrt_arg = np.sqrt(arg)
            dz_dr2 = c / (sqrt_arg * (1.0 + sqrt_arg))

        dz_dr2 += 2.0 * self.aspheric_a4 * r2 + 3.0 * self.aspheric_a6 * (r2 ** 2)

        dz_dx = 2.0 * x * dz_dr2
        dz_dy = 2.0 * y * dz_dr2

        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(r2)

        normals = np.column_stack([nx.ravel(), ny.ravel(), nz.ravel()])
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        return normals / norms

    def intersect(self, rays: RayBundle, max_iter: int = 15, tol: float = 1e-10) -> np.ndarray:
        r"""Performs vectorized Newton-Raphson line-surface intersection.

        Returns: distance t along each ray to intersection point.
        """
        p0 = rays.origin
        d = rays.direction

        # Initial planar tangent guess: t0 = (z_vertex - p_z) / d_z
        dz = np.where(abs(d[:, 2]) < 1e-12, 1e-12, d[:, 2])
        t = (self.z_position - p0[:, 2]) / dz

        for _ in range(max_iter):
            # Current point: p = p0 + t * d
            p = p0 + t[:, None] * d
            x, y, z = p[:, 0], p[:, 1], p[:, 2]

            # Residual: F = z - z_vertex - sag(x, y)
            f_val = z - self.z_position - self.sag(x, y)

            # Gradient: dF/dt = d_z - dz/dx * d_x - dz/dy * d_y
            r2 = x * x + y * y
            c = self.curvature
            if abs(c) < 1e-12:
                dz_dr2 = np.zeros_like(r2)
            else:
                arg = np.maximum(1e-12, 1.0 - (1.0 + self.conic) * (c * c) * r2)
                sqrt_arg = np.sqrt(arg)
                dz_dr2 = c / (sqrt_arg * (1.0 + sqrt_arg))
            dz_dr2 += 2.0 * self.aspheric_a4 * r2 + 3.0 * self.aspheric_a6 * (r2 ** 2)

            df_dt = d[:, 2] - 2.0 * x * dz_dr2 * d[:, 0] - 2.0 * y * dz_dr2 * d[:, 1]
            df_dt = np.where(abs(df_dt) < 1e-12, 1e-12, df_dt)

            dt = f_val / df_dt
            t -= dt

            if np.max(np.abs(dt)) < tol:
                break

        # Check aperture vignetting
        intersections = p0 + t[:, None] * d
        r_intersect = np.hypot(intersections[:, 0], intersections[:, 1])
        vignetted = r_intersect > self.clear_aperture
        rays.active &= ~vignetted

        return t
