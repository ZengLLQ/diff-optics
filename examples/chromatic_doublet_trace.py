"""Example: Tracing multi-element achromatic doublet lens across visible wavelengths."""

import numpy as np
from diffoptics.ray import RayBundle
from diffoptics.surface import OpticalSurface
from diffoptics.system import OpticalSystem


def run():
    print("Setting up Fraunhofer Achromatic Doublet (Crown N-BK7 + Flint N-SF11)...")

    # Doublet prescription:
    # Surface 1: Crown front R1 = 62.0 mm
    # Surface 2: Cemented interface R2 = -44.0 mm
    # Surface 3: Flint rear R3 = -120.0 mm
    s1 = OpticalSurface(z_position=0.0, curvature=1.0 / 62.0, refractive_index=1.5168)
    s2 = OpticalSurface(z_position=7.0, curvature=-1.0 / 44.0, refractive_index=1.7847)
    s3 = OpticalSurface(z_position=10.0, curvature=-1.0 / 120.0, refractive_index=1.0)

    z_image = 105.0
    system = OpticalSystem(surfaces=[s1, s2, s3], z_image=z_image)

    # Trace multi-wavelength rays (d-line 587nm, F-line 486nm, C-line 656nm)
    wavelengths = [0.4861, 0.5876, 0.6563]
    colors = ["Blue (F)", "Yellow (d)", "Red (C)"]

    for wl, name in zip(wavelengths, colors):
        rays = RayBundle.create_collimated_bundle(radius=8.0, num_rings=4, z_start=-10.0, wavelength=wl)
        system.trace(rays)
        rms = system.compute_rms_spot_radius(rays)
        print(f"  {name:10s} (lambda={wl*1000:.1f}nm): RMS Spot Radius = {rms:.5f} mm")

    print("\nAchromatic doublet tracing completed successfully!")


if __name__ == "__main__":
    run()
