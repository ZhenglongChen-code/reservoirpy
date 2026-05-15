"""
Multi-resolution single-phase simulation with JAX.

One 64x64 permeability field + five-spot wells → simulations at 16x16, 32x32,
64x64.  Coarse-grid permeability is obtained via geometric-mean block
averaging; well positions are remapped through normalised coordinates.

Run with:
    python examples/jax_multiresolution.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import numpy as np

from reservoirpy.models.jax_single_phase import (
    interpolate_to_fine,
    run_multiresolution,
)
from reservoirpy.utils.units import uc


def main():
    nx_fine = ny_fine = 64
    Lx = nx_fine * 50.0
    Ly = ny_fine * 50.0

    rng = np.random.default_rng(2026)
    perm_fine = np.exp(rng.normal(np.log(100.0), 0.5, size=(1, ny_fine, nx_fine)))

    wells_fine = [
        {"location": [0, 32, 32], "control_type": "bhp", "value": uc.mpa_to_pa(40), "rw": 0.1},
        {"location": [0, 0, 0], "control_type": "bhp", "value": uc.mpa_to_pa(20), "rw": 0.1},
        {"location": [0, 0, 63], "control_type": "bhp", "value": uc.mpa_to_pa(20), "rw": 0.1},
        {"location": [0, 63, 0], "control_type": "bhp", "value": uc.mpa_to_pa(20), "rw": 0.1},
        {"location": [0, 63, 63], "control_type": "bhp", "value": uc.mpa_to_pa(20), "rw": 0.1},
    ]

    resolutions = [(16, 16), (32, 32), (64, 64)]

    t0 = time.time()
    results = run_multiresolution(
        perm_fine=perm_fine,
        wells_fine=wells_fine,
        Lx=Lx,
        Ly=Ly,
        dz=10.0,
        resolutions=resolutions,
        dt=uc.d_to_s(10),
        n_steps=10,
        initial_pressure=uc.mpa_to_pa(30),
        viscosity=1e-3,
        compressibility=1e-9,
        porosity=0.2,
    )
    elapsed = time.time() - t0
    print(f"Total elapsed (incl. JIT): {elapsed:.2f} s\n")

    for (nx, ny), res in results.items():
        p = uc.pa_to_mpa(res["pressure"])
        last_cg = res["cg_info"][-1]
        print(f"  {nx}x{ny}: P=[{p.min():.2f}, {p.max():.2f}] MPa, "
              f"CG iters={last_cg.iterations}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Multi-Resolution Single-Phase Pressure (JAX CG)", fontsize=15)

    for col, (nx, ny) in enumerate(resolutions):
        p_final = uc.pa_to_mpa(results[(nx, ny)]["pressure"])
        ax = axes[0, col]
        im = ax.imshow(p_final, origin="lower", cmap="RdYlBu_r", aspect="equal")
        ax.set_title(f"{nx}×{ny} final P (MPa)")
        plt.colorbar(im, ax=ax, shrink=0.8)

        p_on_fine = interpolate_to_fine(p_final, nx_fine, ny_fine)
        ax = axes[1, col]
        im = ax.imshow(p_on_fine, origin="lower", cmap="RdYlBu_r", aspect="equal")
        ax.set_title(f"{nx}×{ny} → {nx_fine}×{ny_fine} interp")
        plt.colorbar(im, ax=ax, shrink=0.8)

    p_ref = uc.pa_to_mpa(results[(64, 64)]["pressure"])
    fig2, axes2 = plt.subplots(1, len(resolutions) - 1, figsize=(14, 4.5))
    fig2.suptitle("Difference from 64×64 reference (interpolated to fine grid)", fontsize=13)
    for col, (nx, ny) in enumerate(resolutions[:-1]):
        p_coarse = uc.pa_to_mpa(results[(nx, ny)]["pressure"])
        p_coarse_fine = interpolate_to_fine(p_coarse, nx_fine, ny_fine)
        diff = p_coarse_fine - p_ref
        ax = axes2[col]
        vmax = max(np.abs(diff).max(), 0.01)
        im = ax.imshow(diff, origin="lower", cmap="bwr", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_title(f"{nx}×{ny} − 64×64 (MPa)")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    out = __import__("os").path.join(
        __import__("os").path.dirname(__import__("os").path.abspath(__file__)),
        "jax_multiresolution_result.png",
    )
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {out}")


if __name__ == "__main__":
    main()
