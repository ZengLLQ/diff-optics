"""Tests for RayBundle creation and 3D propagation."""

import pytest
import numpy as np
from diffoptics.ray import RayBundle


def test_ray_bundle_creation_and_normalization():
    origins = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    directions = np.array([[0.0, 0.0, 5.0], [1.0, 1.0, 1.0]])

    bundle = RayBundle(origins, directions)
    assert bundle.num_rays == 2

    # All direction vectors must be unit norm
    norms = np.linalg.norm(bundle.direction, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-12)


def test_collimated_bundle():
    bundle = RayBundle.create_collimated_bundle(radius=5.0, num_rings=3, z_start=-20.0)
    assert bundle.num_rays > 1
    # Chief ray at origin
    np.testing.assert_allclose(bundle.origin[0], [0.0, 0.0, -20.0])
    # All rays point along +Z
    np.testing.assert_allclose(bundle.direction[:, 2], 1.0)


def test_ray_propagation():
    bundle = RayBundle(origin=[[0.0, 0.0, 0.0]], direction=[[0.0, 0.0, 1.0]])
    bundle.propagate(distance=np.array([15.0]), refractive_index=1.5)

    np.testing.assert_allclose(bundle.origin[0], [0.0, 0.0, 15.0])
    assert bundle.opl[0] == pytest.approx(15.0 * 1.5)
