"""
Two-phase IMPES multi-resolution training data generator (JAX GPU/CPU).

Generates 3000 samples at 16×16, 32×32, 64×64, 128×128 from the *same*
fine-scale (128×128) permeability field using JAX-accelerated IMPES solver.

Output layout:
    /mnt/bn/zzl-lf/czl/Dataset/two_phase/
        16x16/  data.npz  metadata.json
        32x32/  data.npz  metadata.json
        64x64/  data.npz  metadata.json
        128x128/ data.npz metadata.json

Single GPU:
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_jax_twophase_data.py

Multi-GPU (2 cards):
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_jax_twophase_data.py --shard 0 --num-shards 2 &
    CUDA_VISIBLE_DEVICES=1 python scripts/generate_jax_twophase_data.py --shard 1 --num-shards 2 &
    wait
    python scripts/generate_jax_twophase_data.py --merge --num-shards 2

Background:
    nohup python scripts/generate_jax_twophase_data.py > generate.log 2>&1 &
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import json
import logging
import time

import numpy as np

from reservoirpy.models.jax_two_phase import JaxTwoPhaseIMPES
from reservoirpy.models.jax_single_phase import resample_perm_field, remap_well_positions
from reservoirpy.utils.units import uc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RESOLUTIONS = [16, 32, 64, 128]
NX_FINE = NY_FINE = 128
DX = DY = 50.0
DZ = 10.0
N_SAMPLES = 3000
N_TIME_SLICES = 100
TOTAL_TIME_DAYS = 365.0
TOTAL_TIME = uc.d_to_s(TOTAL_TIME_DAYS)

OUTPUT_ROOT = "/mnt/bn/zzl-lf/czl/Dataset/two_phase"


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
    porosity = rng.uniform(0.1, 0.3)
    compressibility = 10 ** rng.uniform(-10, -8)
    initial_pressure_MPa = rng.uniform(20, 35)
    mu_o_mPas = float(10 ** rng.uniform(-0.3, 1.0))
    mu_w_mPas = float(10 ** rng.uniform(-1.0, 0.0))
    n_o = rng.uniform(1.5, 4.0)
    n_w = rng.uniform(1.5, 4.0)
    S_or = rng.uniform(0.1, 0.3)
    S_wr = rng.uniform(0.1, 0.3)
    inj_y = int(rng.integers(NY_FINE // 4, 3 * NY_FINE // 4))
    inj_x = int(rng.integers(NX_FINE // 4, 3 * NX_FINE // 4))
    bhp_inj_MPa = initial_pressure_MPa + rng.uniform(5, 15)
    bhp_prod_MPa = initial_pressure_MPa - rng.uniform(5, 15)
    bhp_prod_MPa = max(bhp_prod_MPa, 5.0)
    return {
        "seed": seed,
        "mean_log_perm": mean_log_perm,
        "std_log_perm": std_log_perm,
        "porosity": porosity,
        "compressibility": compressibility,
        "initial_pressure_MPa": initial_pressure_MPa,
        "mu_o_mPas": mu_o_mPas,
        "mu_w_mPas": mu_w_mPas,
        "n_o": n_o,
        "n_w": n_w,
        "S_or": S_or,
        "S_wr": S_wr,
        "inj_y": inj_y,
        "inj_x": inj_x,
        "bhp_inj_MPa": bhp_inj_MPa,
        "bhp_prod_MPa": bhp_prod_MPa,
    }


def generate_perm_field_128(cfg: dict) -> np.ndarray:
    rng = np.random.default_rng(cfg["seed"])
    log_k = rng.normal(cfg["mean_log_perm"], cfg["std_log_perm"],
                       size=(NY_FINE, NX_FINE))
    return np.power(10.0, log_k).astype(np.float64)


def make_wells_fine(cfg: dict) -> list:
    iy, ix = cfg["inj_y"], cfg["inj_x"]
    bhp_inj_Pa = uc.mpa_to_pa(cfg["bhp_inj_MPa"])
    bhp_prod_Pa = uc.mpa_to_pa(cfg["bhp_prod_MPa"])
    return [
        {"location": [0, iy, ix], "control_type": "bhp", "value": bhp_inj_Pa, "rw": 0.1, "skin_factor": 0},
        {"location": [0, 0, 0], "control_type": "bhp", "value": bhp_prod_Pa, "rw": 0.1, "skin_factor": 0},
        {"location": [0, 0, NX_FINE - 1], "control_type": "bhp", "value": bhp_prod_Pa, "rw": 0.1, "skin_factor": 0},
        {"location": [0, NY_FINE - 1, 0], "control_type": "bhp", "value": bhp_prod_Pa, "rw": 0.1, "skin_factor": 0},
        {"location": [0, NY_FINE - 1, NX_FINE - 1], "control_type": "bhp", "value": bhp_prod_Pa, "rw": 0.1, "skin_factor": 0},
    ]


def encode_wells(wells_fine: list, bhp_inj_MPa: float, bhp_prod_MPa: float,
                 nx: int, ny: int) -> tuple:
    well_mask = np.zeros((ny, nx), dtype=np.int8)
    well_value = np.zeros((ny, nx), dtype=np.float32)
    wells = remap_well_positions(wells_fine, NX_FINE, NY_FINE, nx, ny)
    for i, w in enumerate(wells):
        z, y, x = w["location"]
        if i == 0:
            well_mask[y, x] = 1
            well_value[y, x] = bhp_inj_MPa
        else:
            well_mask[y, x] = 2
            well_value[y, x] = bhp_prod_MPa
    return well_mask, well_value


def run_jax_twophase(perm_mD, wells_fine, cfg, nx, ny):
    Lx = NX_FINE * DX
    Ly = NY_FINE * DY
    dx = Lx / nx
    dy = Ly / ny

    wells = remap_well_positions(wells_fine, NX_FINE, NY_FINE, nx, ny)

    if nx == NX_FINE and ny == NY_FINE:
        perm = perm_mD
    else:
        perm = resample_perm_field(perm_mD, nx, ny)
        if perm.ndim == 3:
            perm = perm.reshape(ny, nx)

    solver = JaxTwoPhaseIMPES(
        nx=nx, ny=ny, dx=dx, dy=dy, dz=DZ,
        permeability_mD=perm,
        porosity=cfg["porosity"],
        mu_o=uc.mpas_to_pas(cfg["mu_o_mPas"]),
        mu_w=uc.mpas_to_pas(cfg["mu_w_mPas"]),
        compressibility=cfg["compressibility"],
        kro_params={"n_o": cfg["n_o"], "S_or": cfg["S_or"]},
        krw_params={"n_w": cfg["n_w"], "S_wr": cfg["S_wr"]},
        wells_config=wells,
    )

    result = solver.run(
        initial_pressure_Pa=uc.mpa_to_pa(cfg["initial_pressure_MPa"]),
        initial_Sw=cfg["S_wr"],
        total_time=TOTAL_TIME,
        n_snapshots=N_TIME_SLICES,
    )
    return result["pressure"], result["saturation"]


def generate_shard(shard: int, num_shards: int, output_root: str):
    sample_indices = list(range(shard, N_SAMPLES, num_shards))
    n_shard = len(sample_indices)

    log_fmt = logging.Formatter(f"%(asctime)s [shard {shard}] %(message)s")
    for h in logging.getLogger().handlers:
        h.setFormatter(log_fmt)

    logger.info(f"Generating {n_shard} two-phase samples at resolutions {RESOLUTIONS} [JAX IMPES]")

    accumulators = {}
    for res in RESOLUTIONS:
        accumulators[res] = {
            "pressure_MPa": [],
            "saturation": [],
            "permeability_mD": [],
            "well_mask": [],
            "well_value": [],
        }

    t0 = time.time()

    for count, i in enumerate(sample_indices):
        cfg = random_config(i)

        try:
            perm_128 = generate_perm_field_128(cfg)
            wells_fine = make_wells_fine(cfg)

            for res in RESOLUTIONS:
                pressure, saturation = run_jax_twophase(
                    perm_128, wells_fine, cfg, res, res)
                pressure_MPa = uc.pa_to_mpa(pressure).astype(np.float32)
                saturation = saturation.astype(np.float32)

                if res == NX_FINE:
                    perm_res = perm_128.astype(np.float32)
                else:
                    perm_res = resample_perm_field(perm_128, res, res)
                    if perm_res.ndim == 3:
                        perm_res = perm_res.reshape(res, res)
                    perm_res = perm_res.astype(np.float32)

                well_mask, well_value = encode_wells(
                    wells_fine, cfg["bhp_inj_MPa"], cfg["bhp_prod_MPa"], res, res)

                acc = accumulators[res]
                acc["pressure_MPa"].append(pressure_MPa)
                acc["saturation"].append(saturation)
                acc["permeability_mD"].append(perm_res)
                acc["well_mask"].append(well_mask)
                acc["well_value"].append(well_value)

        except Exception as e:
            logger.error(f"Sample {i} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        if (count + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (count + 1) / elapsed
            eta = (n_shard - count - 1) / rate
            logger.info(f"  [{count+1}/{n_shard}] "
                        f"elapsed={elapsed:.0f}s, rate={rate:.2f} samples/s, "
                        f"ETA={eta:.0f}s")

    logger.info(f"Simulation done in {time.time()-t0:.0f}s, saving shard...")

    for res in RESOLUTIONS:
        acc = accumulators[res]
        n = len(acc["pressure_MPa"])
        if n == 0:
            continue

        out_dir = os.path.join(output_root, f"{res}x{res}")
        os.makedirs(out_dir, exist_ok=True)

        npz_path = os.path.join(out_dir, f"shard_{shard}.npz")
        np.savez_compressed(
            npz_path,
            pressure_MPa=np.stack(acc["pressure_MPa"]),
            saturation=np.stack(acc["saturation"]),
            permeability_mD=np.stack(acc["permeability_mD"]),
            well_mask=np.stack(acc["well_mask"]),
            well_value=np.stack(acc["well_value"]),
        )
        npz_size = os.path.getsize(npz_path) / 1e6
        logger.info(f"  {res}x{res}: {n} samples, {npz_size:.1f} MB → {npz_path}")

    logger.info(f"Shard {shard} done!")


def merge_shards(num_shards: int, output_root: str):
    logger.info(f"Merging {num_shards} shards into final datasets...")

    for res in RESOLUTIONS:
        out_dir = os.path.join(output_root, f"{res}x{res}")
        shard_files = [os.path.join(out_dir, f"shard_{s}.npz")
                       for s in range(num_shards)]
        existing = [f for f in shard_files if os.path.exists(f)]

        if not existing:
            logger.warning(f"  {res}x{res}: no shard files found, skipping")
            continue

        all_p, all_sw, all_perm, all_mask, all_val = [], [], [], [], []
        for f in existing:
            d = np.load(f)
            all_p.append(d["pressure_MPa"])
            all_sw.append(d["saturation"])
            all_perm.append(d["permeability_mD"])
            all_mask.append(d["well_mask"])
            all_val.append(d["well_value"])

        pressure_MPa = np.concatenate(all_p, axis=0)
        saturation = np.concatenate(all_sw, axis=0)
        permeability_mD = np.concatenate(all_perm, axis=0)
        well_mask = np.concatenate(all_mask, axis=0)
        well_value = np.concatenate(all_val, axis=0)

        n = pressure_MPa.shape[0]
        npz_path = os.path.join(out_dir, "data.npz")
        np.savez_compressed(
            npz_path,
            pressure_MPa=pressure_MPa,
            saturation=saturation,
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
            "solver": "jax_impes_cg",
            "physics": "two_phase",
            "num_shards": num_shards,
            "shapes": {
                "pressure_MPa": f"({n}, {N_TIME_SLICES}, {res}, {res})",
                "saturation": f"({n}, {N_TIME_SLICES}, {res}, {res})",
                "permeability_mD": f"({n}, {res}, {res})",
                "well_mask": f"({n}, {res}, {res})",
                "well_value": f"({n}, {res}, {res})",
                "t_normalized": f"({N_TIME_SLICES},)",
                "time_days": f"({N_TIME_SLICES},)",
            },
        }
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        for sf in existing:
            os.remove(sf)

        logger.info(f"  {res}x{res}: {n} samples, {npz_size:.1f} MB → {npz_path}")

    logger.info("Merge done!")


def main():
    parser = argparse.ArgumentParser(
        description="Two-phase IMPES multi-resolution data generator (JAX)")
    parser.add_argument("--shard", type=int, default=None,
                        help="Shard index (0-based).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards. Default: 1")
    parser.add_argument("--merge", action="store_true",
                        help="Merge shards into final data.npz")
    parser.add_argument("--output", type=str, default=OUTPUT_ROOT,
                        help=f"Output root directory. Default: {OUTPUT_ROOT}")
    args = parser.parse_args()

    if args.merge:
        merge_shards(args.num_shards, args.output)
        return

    if args.shard is not None:
        generate_shard(args.shard, args.num_shards, args.output)
    else:
        generate_shard(0, 1, args.output)


if __name__ == "__main__":
    main()
