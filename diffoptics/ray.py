"""3D optical ray bundles and geometric propagation."""

from __future__ import annotations
import numpy as np


class RayBundle:
    """Represents a vectorized bundle of optical rays in 3D Euclidean space."""

    def __init__(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        wavelength: float = 0.5876,  # Helium d-line [micrometers]
    ):
        self.origin = np.asarray(origin, dtype=np.float64).reshape(-1, 3)
        self.direction = np.asarray(direction, dtype=np.float64).reshape(-1, 3)
        # Normalize directions
        norms = np.linalg.norm(self.direction, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        self.direction /= norms

        self.num_rays = self.origin.shape[0]
        self.wavelength = float(wavelength)
        self.opl = np.zeros(self.num_rays, dtype=np.float64)
        self.active = np.ones(self.num_rays, dtype=bool)

    @classmethod
    def create_collimated_bundle(
        cls,
        radius: float,
        num_rings: int = 4,
        z_start: float = -10.0,
        wavelength: float = 0.5876,
    ) -> RayBundle:
        """Generates polar concentric rings of parallel collimated rays along +Z axis."""
        origins = [[0.0, 0.0, z_start]]
        directions = [[0.0, 0.0, 1.0]]

        for r_idx in range(1, num_rings + 1):
            r = radius * (r_idx / num_rings)
            num_pts = 6 * r_idx
            for theta_idx in range(num_pts):
                theta = 2.0 * np.pi * (theta_idx / num_pts)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                origins.append([x, y, z_start])
                directions.append([0.0, 0.0, 1.0])

        return cls(origin=np.array(origins), direction=np.array(directions), wavelength=wavelength)

    def propagate(self, distance: np.ndarray, refractive_index: float = 1.0) -> None:
        """Propagates active rays along their direction by specified distance."""
        dist = np.asarray(distance, dtype=np.float64).reshape(-1, 1)
        self.origin += np.where(self.active[:, None], dist * self.direction, 0.0)
        self.opl += np.where(self.active, dist.ravel() * refractive_index, 0.0)
