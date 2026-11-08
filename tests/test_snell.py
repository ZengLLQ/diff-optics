"""Tests for 3D vector Snell's law refraction and total internal reflection."""

import pytest
import numpy as np
from diffoptics.snell import refract_ray_bundle


def test_normal_incidence():
    d = np.array([[0.0, 0.0, 1.0]])
    n = np.array([[0.0, 0.0, 1.0]])

    refr, tir = refract_ray_bundle(d, n, n1=1.0, n2=1.5)
    np.testing.assert_allclose(refr, [[0.0, 0.0, 1.0]], atol=1e-12)
    assert tir[0] == False


def test_oblique_snells_law():
    # Ray at 45 degrees in X-Z plane: d = (sin(45), 0, cos(45))
    angle1 = np.radians(45.0)
    d = np.array([[np.sin(angle1), 0.0, np.cos(angle1)]])
    n = np.array([[0.0, 0.0, 1.0]])

    n1 = 1.0
    n2 = 1.5
    refr, tir = refract_ray_bundle(d, n, n1=n1, n2=n2)

    assert tir[0] == False
    # Snell's law: n1 * sin(theta1) = n2 * sin(theta2)
    expected_sin_theta2 = (n1 / n2) * np.sin(angle1)
    actual_sin_theta2 = refr[0, 0]
    assert pytest.approx(actual_sin_theta2, rel=1e-6) == expected_sin_theta2


def test_total_internal_reflection():
    # High to low index: n1=1.5, n2=1.0. Critical angle = arcsin(1/1.5) = 41.81 deg
    angle_tir = np.radians(60.0)  # Beyond critical angle
    d = np.array([[np.sin(angle_tir), 0.0, np.cos(angle_tir)]])
    n = np.array([[0.0, 0.0, 1.0]])

    _, tir = refract_ray_bundle(d, n, n1=1.5, n2=1.0)
    assert tir[0] == True
