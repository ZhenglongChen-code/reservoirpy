"""
Minimal 64x64 single-phase JAX prototype.

Run with:
    python examples/jax_single_phase_64x64.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from reservoirpy.models.jax_single_phase import JaxSinglePhaseCG
from reservoirpy.utils.units import uc


def main():
    nx = ny = 64
    wells_config = [
        {"location": [0, 32, 32], "control_type": "bhp", "value": uc.mpa_to_pa(40), "rw": 0.1},
        {"location": [0, 0, 0], "control_type": "bhp", "value": uc.mpa_to_pa(20), "rw": 0.1},
        {"location": [0, 0, 63], "control_type": "bhp", "value": uc.mpa_to_pa(20), "rw": 0.1},
        {"location": [0, 63, 0], "control_type": "bhp", "value": uc.mpa_to_pa(20), "rw": 0.1},
        {"location": [0, 63, 63], "control_type": "bhp", "value": uc.mpa_to_pa(20), "rw": 0.1},
    ]

    rng = np.random.default_rng(2026)
    permeability_mD = np.exp(rng.normal(np.log(100.0), 0.5, size=(ny, nx)))

    solver = JaxSinglePhaseCG(
        nx=nx,
        ny=ny,
        dx=50.0,
        dy=50.0,
        dz=10.0,
        permeability_mD=permeability_mD,
        porosity=0.2,
        viscosity=1e-3,
        compressibility=1e-9,
        wells_config=wells_config,
        cg_tolerance=1e-8,
        cg_maxiter=1000,
    )

    t0 = time.time()
    result = solver.run(initial_pressure=uc.mpa_to_pa(30), dt=uc.d_to_s(10), n_steps=10)
    elapsed = time.time() - t0

    pressure_mpa = uc.pa_to_mpa(result["pressure"])
    last_info = result["cg_info"][-1]
    print(f"JAX devices: {', '.join(str(d) for d in __import__('jax').devices())}")
    print(f"Final pressure range: {pressure_mpa.min():.3f} - {pressure_mpa.max():.3f} MPa")
    print(f"Last CG: iterations={last_info.iterations}, residual={last_info.residual_norm:.3e}")
    print(f"Elapsed including first JIT compile: {elapsed:.3f} s")


if __name__ == "__main__":
    main()
