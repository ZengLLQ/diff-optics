"""Optical system sequentially tracing ray bundles through optical elements."""

from __future__ import annotations
import numpy as np
from typing import List, Tuple
from .ray import RayBundle
from .surface import OpticalSurface
from .snell import refract_ray_bundle


class OpticalSystem:
    """Manages sequential optical surfaces terminating on an image plane detector."""

    def __init__(self, surfaces: List[OpticalSurface], z_image: float):
        self.surfaces = surfaces
        self.z_image = float(z_image)

    def trace(self, rays: RayBundle) -> RayBundle:
        """Traces ray bundle through all sequential refractive interfaces onto detector."""
        current_index = 1.0  # Initial medium (air)

        for surface in self.surfaces:
            # 1. Intersect surface
            t = surface.intersect(rays)

            # 2. Advance rays to intersection points
            rays.propagate(t, refractive_index=current_index)

            # 3. Compute surface normals at intersection points
            x = rays.origin[:, 0]
            y = rays.origin[:, 1]
            normals = surface.surface_normal(x, y)

            # 4. Refract rays
            refracted_dir, tir_mask = refract_ray_bundle(
                rays.direction, normals, current_index, surface.refractive_index
            )

            rays.active &= ~tir_mask
            rays.direction = np.where(rays.active[:, None], refracted_dir, rays.direction)
            current_index = surface.refractive_index

        # Final propagation to detector image plane
        dz = np.where(abs(rays.direction[:, 2]) < 1e-12, 1e-12, rays.direction[:, 2])
        t_detector = (self.z_image - rays.origin[:, 2]) / dz
        rays.propagate(t_detector, refractive_index=current_index)

        return rays

    def get_spot_coordinates(self, rays: RayBundle) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (x, y) positions of active rays at image detector plane."""
        active_mask = rays.active
        if not np.any(active_mask):
            return np.array([]), np.array([])
        x = rays.origin[active_mask, 0]
        y = rays.origin[active_mask, 1]
        return x, y

    def compute_rms_spot_radius(self, rays: RayBundle) -> float:
        """Computes root-mean-square spot radius relative to centroid on detector."""
        x, y = self.get_spot_coordinates(rays)
        if len(x) == 0:
            return 1e6  # Large penalty for complete vignetting/TIR

        xc = np.mean(x)
        yc = np.mean(y)
        r2 = (x - xc) ** 2 + (y - yc) ** 2
        return float(np.sqrt(np.mean(r2)))
