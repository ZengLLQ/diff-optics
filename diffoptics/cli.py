"""Command line interface for DiffOptics lens simulation and optimization."""

import click
import numpy as np
from .ray import RayBundle
from .surface import OpticalSurface
from .system import OpticalSystem
from .optimizer import LensOptimizer


@click.group()
def cli():
    """DiffOptics: Differentiable Optical Ray Tracing & Lens Optimization CLI."""
    pass


@cli.command()
@click.option("--curvature", default=0.02, help="Front surface curvature [1/mm] (R = 50mm).")
@click.option("--thickness", default=5.0, help="Lens center thickness [mm].")
@click.option("--focal-dist", default=100.0, help="Distance from lens rear to image plane [mm].")
def trace(curvature: float, thickness: float, focal_dist: float):
    """Traces a collimated beam through a planoconvex singlet lens."""
    click.echo(f"Tracing collimated beam through singlet lens (R={1.0/curvature:.1f}mm)...")
    # Front convex surface (air -> N-BK7 n=1.5168)
    s1 = OpticalSurface(z_position=0.0, curvature=curvature, refractive_index=1.5168)
    # Rear planar surface (N-BK7 -> air n=1.0)
    s2 = OpticalSurface(z_position=thickness, curvature=0.0, refractive_index=1.0)

    z_image = thickness + focal_dist
    system = OpticalSystem(surfaces=[s1, s2], z_image=z_image)

    rays = RayBundle.create_collimated_bundle(radius=10.0, num_rings=5, z_start=-10.0)
    system.trace(rays)

    rms = system.compute_rms_spot_radius(rays)
    x, y = system.get_spot_coordinates(rays)

    click.echo("Trace Completed:")
    click.echo(f"  Active Rays Reaching Detector: {len(x)} / {rays.num_rays}")
    click.echo(f"  Detector Position:             z = {z_image:.2f} mm")
    click.echo(f"  RMS Spot Radius:               {rms:.4f} mm ({rms * 1000.0:.1f} um)")


@cli.command()
@click.option("--iterations", default=25, help="Number of gradient descent iterations.")
@click.option("--lr", default=0.08, help="Learning rate.")
def optimize(iterations: int, lr: float):
    """Optimizes conic constant to eliminate spherical aberration."""
    click.echo(f"Starting automatic differentiation optimization ({iterations} iterations)...")
    s1 = OpticalSurface(z_position=0.0, curvature=0.025, conic=0.0, refractive_index=1.5168)
    s2 = OpticalSurface(z_position=6.0, curvature=0.0, refractive_index=1.0)
    system = OpticalSystem(surfaces=[s1, s2], z_image=84.0)

    def ray_gen():
        return RayBundle.create_collimated_bundle(radius=10.0, num_rings=4, z_start=-10.0)

    optimizer = LensOptimizer(
        system=system,
        ray_generator=ray_gen,
        param_targets=[(0, "conic")],
    )

    init_rms = optimizer.evaluate_loss(optimizer._get_params())
    click.echo(f"  Initial Spherical RMS Spot: {init_rms:.5f} mm")

    opt_params, history = optimizer.optimize_adam(learning_rate=lr, iterations=iterations)
    final_rms = history[-1]

    click.echo(f"Optimization Finished:")
    click.echo(f"  Optimized Conic Constant k: {opt_params[0]:.4f} (Aspheric Ellipsoid/Hyperboloid)")
    click.echo(f"  Final Aspheric RMS Spot:    {final_rms:.5f} mm")
    improvement = (init_rms - final_rms) / init_rms * 100.0
    click.echo(f"  Aberration Reduction:       {improvement:.1f}%")


if __name__ == "__main__":
    cli()
