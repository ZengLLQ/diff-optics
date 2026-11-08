"""DiffOptics: Differentiable Ray Tracing & Aspheric Lens Surface Optimization."""

__version__ = "0.1.0"

from .ray import RayBundle
from .surface import OpticalSurface
from .snell import refract_ray_bundle
from .system import OpticalSystem
from .optimizer import LensOptimizer

__all__ = [
    "RayBundle",
    "OpticalSurface",
    "refract_ray_bundle",
    "OpticalSystem",
    "LensOptimizer",
]
