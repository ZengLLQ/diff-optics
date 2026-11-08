"""Tests for OpticalSystem sequential ray tracing and spot diagrams."""

import pytest
import numpy as np
from diffoptics.ray import RayBundle
from diffoptics.surface import OpticalSurface
from diffoptics.system import OpticalSystem


def test_planar_window_trace():
    """Rays passing through a flat glass plate emerge parallel to incident rays."""
    s1 = OpticalSurface(z_position=0.0, curvature=0.0, refractive_index=1.5)
    s2 = OpticalSurface(z_position=10.0, curvature=0.0, refractive_index=1.0)
    system = OpticalSystem(surfaces=[s1, s2], z_image=20.0)

    angle = np.radians(20.0)
    d_init = np.array([[np.sin(angle), 0.0, np.cos(angle)]])
    rays = RayBundle(origin=[[0.0, 0.0, -5.0]], direction=d_init)

    system.trace(rays)
    # Emergent direction must equal incident direction
    np.testing.assert_allclose(rays.direction, d_init, atol=1e-12)


def test_convex_lens_focusing():
    """Planoconvex lens focuses parallel collimated rays towards focal point."""
    # Front convex R = 50mm, rear flat
    s1 = OpticalSurface(z_position=0.0, curvature=1.0 / 50.0, refractive_index=1.5)
    s2 = OpticalSurface(z_position=5.0, curvature=0.0, refractive_index=1.0)

    # Focal length: f ~ R / (n - 1) = 50 / 0.5 = 100mm from lens
    system = OpticalSystem(surfaces=[s1, s2], z_image=105.0)

    rays = RayBundle.create_collimated_bundle(radius=5.0, num_rings=3, z_start=-10.0)
    system.trace(rays)

    rms = system.compute_rms_spot_radius(rays)
    # Near focus, RMS spot size should be small (< 0.5 mm)
    assert rms < 0.5
