"""
Multi-resolution training data generator using JAX.

Generates 3000 samples at 16×16, 32×32, 64×64, 128×128 from the *same*
fine-scale (128×128) permeability field.  Each sample:
  - random log-normal permeability field
  - random single-well BHP position
  - 1-year simulation with 50 time slices (early-dense + late-sparse)

Output layout:
    /mnt/bn/zzl-lf/czl/Dataset/
        16x16/  data.npz  metadata.json
        32x32/  data.npz  metadata.json
        64x64/  data.npz  metadata.json
        128x128/ data.npz metadata.json

Run:
    python scripts/generate_jax_multires_data.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import logging
import time

import numpy as np

from reservoirpy.models.jax_single_phase import (
    JaxSinglePhaseCG,
    remap_well_positions,
    resample_perm_field,
)
from reservoirpy.utils.units import uc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RESOLUTIONS = [16, 32, 64, 128]
NX_FINE = NY_FINE = 128
DX = DY = 50.0
DZ = 10.0
N_SAMPLES = 3000
N_TIME_SLICES = 50
TOTAL_TIME_DAYS = 365.0
TOTAL_TIME = uc.d_to_s(TOTAL_TIME_DAYS)
N_SIM_STEPS = 200
CG_TOLERANCE = 1e-8
CG_MAXITER = 2000

OUTPUT_ROOT = "/mnt/bn/zzl-lf/czl/Dataset"


def make_time_fractions(n: int) -> np.ndarray:
    if n == 1:
        return np.array([1.0])
    n_early = n * 2 // 3
    n_late = n - n_early
    early = np.geomspace(0.001, 0.25, n_early, endpoint=False)
    late = np.linspace(0.25, 1.0, n_late)
    t = np.concatenate([early, late])
    return np.unique(np.round(t, 6))[:n]


TIME_FRACTIONS = make_time_fractions(N_TIME_SLICES)


def random_config(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    mean_log_perm = rng.uniform(1.5, 2.8)
    std_log_perm = rng.uniform(0.2, 0.5)
    viscosity_mPas = float(10 ** rng.uniform(-0.5, 0.7))
    porosity = rng.uniform(0.1, 0.3)
    compressibility = 10 ** rng.uniform(-10, -8)
    initial_pressure_MPa = rng.uniform(20, 35)
    x = int(rng.integers(0, NX_FINE))
    y = int(rng.integers(0, NY_FINE))
    bhp_MPa = initial_pressure_MPa - rng.uniform(5, 20)
    bhp_MPa = max(bhp_MPa, 5.0)
    return {
        "seed": seed,
        "mean_log_perm": mean_log_perm,
        "std_log_perm": std_log_perm,
        "viscosity_mPas": viscosity_mPas,
        "porosity": porosity,
        "compressibility": compressibility,
        "initial_pressure_MPa": initial_pressure_MPa,
        "well_x": x,
        "well_y": y,
        "bhp_MPa": bhp_MPa,
    }


def generate_perm_field_128(cfg: dict) -> np.ndarray:
    rng = np.random.default_rng(cfg["seed"])
    log_k = rng.normal(cfg["mean_log_perm"], cfg["std_log_perm"],
                       size=(NY_FINE, NX_FINE))
    return np.power(10.0, log_k).astype(np.float64)


def run_jax_simulation(
    perm_mD: np.ndarray,
    well_y: int,
    well_x: int,
    bhp_Pa: float,
    initial_pressure_Pa: float,
    viscosity_Pa_s: float,
    compressibility: float,
    porosity: float,
    nx: int,
    ny: int,
) -> np.ndarray:
    Lx = NX_FINE * DX
    Ly = NY_FINE * DY
    dx = Lx / nx
    dy = Ly / ny

    wells_fine = [{"location": [0, well_y, well_x],
                   "control_type": "bhp", "value": bhp_Pa, "rw": 0.1}]
    wells = remap_well_positions(wells_fine, NX_FINE, NY_FINE, nx, ny)

    if nx == NX_FINE and ny == NY_FINE:
        perm = perm_mD
    else:
        perm = resample_perm_field(perm_mD, nx, ny)
        if perm.ndim == 3:
            perm = perm.reshape(ny, nx)

    solver = JaxSinglePhaseCG(
        nx=nx, ny=ny, dx=dx, dy=dy, dz=DZ,
        permeability_mD=perm, porosity=porosity,
        viscosity=viscosity_Pa_s, compressibility=compressibility,
        wells_config=wells,
        cg_tolerance=CG_TOLERANCE, cg_maxiter=CG_MAXITER,
    )

    snap_times = TIME_FRACTIONS * TOTAL_TIME
    pressure = solver.initialize_pressure(initial_pressure_Pa)
    results = [pressure.copy()]
    current_time = 0.0
    next_snap = 1
    dt = TOTAL_TIME / N_SIM_STEPS

    while current_time < TOTAL_TIME and next_snap < N_TIME_SLICES:
        target = snap_times[next_snap]
        actual_dt = min(dt, target - current_time)
        if actual_dt <= 0:
            next_snap += 1
            continue
        pressure, _ = solver.solve_timestep(pressure, actual_dt)
        current_time += actual_dt
        while next_snap < N_TIME_SLICES and current_time >= snap_times[next_snap] - 1.0:
            results.append(pressure.copy())
            next_snap += 1

    while len(results) < N_TIME_SLICES:
        results.append(results[-1])

    return np.stack(results)


def encode_well(well_y: int, well_x: int, bhp_MPa: float,
                nx: int, ny: int) -> tuple:
    well_mask = np.zeros((ny, nx), dtype=np.int8)
    well_value = np.zeros((ny, nx), dtype=np.float32)
    wells_fine = [{"location": [0, well_y, well_x]}]
    w = remap_well_positions(wells_fine, NX_FINE, NY_FINE, nx, ny)[0]
    _, y_new, x_new = w["location"]
    well_mask[y_new, x_new] = 2
    well_value[y_new, x_new] = bhp_MPa
    return well_mask, well_value


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    logger.info(f"Generating {N_SAMPLES} samples at resolutions {RESOLUTIONS}")
    logger.info(f"Fine grid: {NX_FINE}×{NY_FINE}, domain: "
                f"{NX_FINE*DX:.0f}×{NY_FINE*DY:.0f} m")
    logger.info(f"Output root: {OUTPUT_ROOT}")

    accumulators = {}
    for res in RESOLUTIONS:
        accumulators[res] = {
            "pressure_MPa": [],
            "permeability_mD": [],
            "well_mask": [],
            "well_value": [],
        }

    t0 = time.time()

    for i in range(N_SAMPLES):
        cfg = random_config(i)

        try:
            perm_128 = generate_perm_field_128(cfg)
            bhp_Pa = uc.mpa_to_pa(cfg["bhp_MPa"])
            init_Pa = uc.mpa_to_pa(cfg["initial_pressure_MPa"])
            visc = uc.mpas_to_pas(cfg["viscosity_mPas"])

            for res in RESOLUTIONS:
                pressure = run_jax_simulation(
                    perm_mD=perm_128,
                    well_y=cfg["well_y"],
                    well_x=cfg["well_x"],
                    bhp_Pa=bhp_Pa,
                    initial_pressure_Pa=init_Pa,
                    viscosity_Pa_s=visc,
                    compressibility=cfg["compressibility"],
                    porosity=cfg["porosity"],
                    nx=res, ny=res,
                )
                pressure_MPa = uc.pa_to_mpa(pressure).astype(np.float32)

                if res == NX_FINE:
                    perm_res = perm_128.astype(np.float32)
                else:
                    perm_res = resample_perm_field(perm_128, res, res)
                    if perm_res.ndim == 3:
                        perm_res = perm_res.reshape(res, res)
                    perm_res = perm_res.astype(np.float32)

                well_mask, well_value = encode_well(
                    cfg["well_y"], cfg["well_x"], cfg["bhp_MPa"], res, res)

                acc = accumulators[res]
                acc["pressure_MPa"].append(pressure_MPa)
                acc["permeability_mD"].append(perm_res)
                acc["well_mask"].append(well_mask)
                acc["well_value"].append(well_value)

        except Exception as e:
            logger.error(f"Sample {i} failed: {e}")
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N_SAMPLES - i - 1) / rate
            logger.info(f"  [{i+1}/{N_SAMPLES}] "
                        f"elapsed={elapsed:.0f}s, rate={rate:.1f} samples/s, "
                        f"ETA={eta:.0f}s")

    total_elapsed = time.time() - t0
    logger.info(f"Simulation done in {total_elapsed:.0f}s, saving...")

    for res in RESOLUTIONS:
        acc = accumulators[res]
        n = len(acc["pressure_MPa"])
        if n == 0:
            logger.warning(f"  {res}x{res}: no samples, skipping")
            continue

        out_dir = os.path.join(OUTPUT_ROOT, f"{res}x{res}")
        os.makedirs(out_dir, exist_ok=True)

        pressure_MPa = np.stack(acc["pressure_MPa"])
        permeability_mD = np.stack(acc["permeability_mD"])
        well_mask = np.stack(acc["well_mask"])
        well_value = np.stack(acc["well_value"])

        npz_path = os.path.join(out_dir, "data.npz")
        np.savez_compressed(
            npz_path,
            pressure_MPa=pressure_MPa,
            permeability_mD=permeability_mD,
            well_mask=well_mask,
            well_value=well_value,
            t_normalized=TIME_FRACTIONS.astype(np.float32),
            time_days=(TIME_FRACTIONS * TOTAL_TIME_DAYS).astype(np.float32),
        )
        npz_size = os.path.getsize(npz_path) / 1e6

        metadata = {
            "n_samples": n,
            "grid_size": res,
            "n_time_slices": N_TIME_SLICES,
            "total_time_days": TOTAL_TIME_DAYS,
            "dx": NX_FINE * DX / res,
            "dy": NY_FINE * DY / res,
            "dz": DZ,
            "Lx": NX_FINE * DX,
            "Ly": NY_FINE * DY,
            "time_fractions": TIME_FRACTIONS.tolist(),
            "fine_grid_size": NX_FINE,
            "upscaling_method": "geometric_mean_block_average",
            "shapes": {
                "pressure_MPa": f"({n}, {N_TIME_SLICES}, {res}, {res})",
                "permeability_mD": f"({n}, {res}, {res})",
                "well_mask": f"({n}, {res}, {res})",
                "well_value": f"({n}, {res}, {res})",
                "t_normalized": f"({N_TIME_SLICES},)",
                "time_days": f"({N_TIME_SLICES},)",
            },
        }
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"  {res}x{res}: {n} samples, {npz_size:.1f} MB → {out_dir}")

    logger.info(f"All done! Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
