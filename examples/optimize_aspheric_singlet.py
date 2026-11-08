"""Example: Aspheric singlet lens optimization eliminating spherical aberration."""

import numpy as np
from diffoptics.ray import RayBundle
from diffoptics.surface import OpticalSurface
from diffoptics.system import OpticalSystem
from diffoptics.optimizer import LensOptimizer


def run():
    print("Setting up initial spherical planoconvex lens (R = 40mm, N-BK7 n=1.5168)...")
    s1 = OpticalSurface(z_position=0.0, curvature=0.025, conic=0.0, refractive_index=1.5168)
    s2 = OpticalSurface(z_position=6.0, curvature=0.0, refractive_index=1.0)
    system = OpticalSystem(surfaces=[s1, s2], z_image=84.0)

    def ray_gen():
        return RayBundle.create_collimated_bundle(radius=10.0, num_rings=4, z_start=-10.0)

    optimizer = LensOptimizer(
        system=system,
        ray_generator=ray_gen,
        param_targets=[(0, "curvature"), (0, "conic")],
    )

    init_rms = optimizer.evaluate_loss(optimizer._get_params())
    print(f"Initial Spherical Lens RMS Spot Radius: {init_rms:.5f} mm")

    print("\nRunning Adam gradient descent on (curvature, conic constant k)...")
    opt_params, history = optimizer.optimize_adam(learning_rate=0.04, iterations=25)
    final_rms = history[-1]

    print(f"Optimization Results:")
    print(f"  Optimized Front Curvature:  {opt_params[0]:.6f} (R = {1.0/opt_params[0]:.2f} mm)")
    print(f"  Optimized Conic Constant k: {opt_params[1]:.4f}")
    print(f"  Final Aspheric Spot Radius: {final_rms:.5f} mm")
    print(f"  Aberration Reduction:       {((init_rms - final_rms)/init_rms * 100):.1f}%")


if __name__ == "__main__":
    run()
