"""Tests for optical surface sag, normal vectors, and ray intersection."""

import pytest
import numpy as np
from diffoptics.ray import RayBundle
from diffoptics.surface import OpticalSurface


def test_planar_surface():
    surf = OpticalSurface(z_position=10.0, curvature=0.0)
    x = np.array([0.0, 5.0, -3.0])
    y = np.array([0.0, 2.0, 4.0])

    # Planar sag is zero everywhere
    np.testing.assert_allclose(surf.sag(x, y), 0.0)

    # Normal is always (0, 0, 1)
    normals = surf.surface_normal(x, y)
    np.testing.assert_allclose(normals[:, 2], 1.0)


def test_spherical_sag_and_intersection():
    R = 100.0  # Radius 100mm
    c = 1.0 / R
    surf = OpticalSurface(z_position=0.0, curvature=c)

    # Test ray entering at (x=10, y=0, z=-10) directed along +Z
    ray = RayBundle(origin=[[10.0, 0.0, -10.0]], direction=[[0.0, 0.0, 1.0]])
    t = surf.intersect(ray)

    # Analytical sphere: z = R - sqrt(R^2 - x^2) = 100 - sqrt(100^2 - 10^2) = 100 - 99.49874 = 0.50126
    expected_z = R - np.sqrt(R ** 2 - 100.0)
    expected_t = 10.0 + expected_z  # started at z=-10

    assert pytest.approx(t[0], rel=1e-5) == expected_t


def test_aperture_vignetting():
    surf = OpticalSurface(z_position=0.0, clear_aperture=5.0)
    ray_in = RayBundle(origin=[[3.0, 0.0, -5.0]], direction=[[0.0, 0.0, 1.0]])
    ray_out = RayBundle(origin=[[8.0, 0.0, -5.0]], direction=[[0.0, 0.0, 1.0]])

    surf.intersect(ray_in)
    surf.intersect(ray_out)

    assert ray_in.active[0] == True
    assert ray_out.active[0] == False
