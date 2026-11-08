"""Tests for LensOptimizer gradient descent aberration reduction."""

import pytest
import numpy as np
from diffoptics.ray import RayBundle
from diffoptics.surface import OpticalSurface
from diffoptics.system import OpticalSystem
from diffoptics.optimizer import LensOptimizer


def test_lens_optimizer_reduces_rms():
    s1 = OpticalSurface(z_position=0.0, curvature=0.02, conic=0.0, refractive_index=1.5)
    s2 = OpticalSurface(z_position=5.0, curvature=0.0, refractive_index=1.0)
    system = OpticalSystem(surfaces=[s1, s2], z_image=105.0)

    def ray_gen():
        return RayBundle.create_collimated_bundle(radius=8.0, num_rings=3, z_start=-5.0)

    opt = LensOptimizer(
        system=system,
        ray_generator=ray_gen,
        param_targets=[(0, "conic")],
    )

    init_loss = opt.evaluate_loss(opt._get_params())
    params, history = opt.optimize_adam(learning_rate=0.1, iterations=15)
    final_loss = history[-1]

    # Conic optimization must reduce spherical aberration spot size
    assert final_loss < init_loss
