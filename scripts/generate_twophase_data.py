"""
Two-phase IMPES multi-resolution training data generator (CPU, no JAX).

Generates 3000 samples at 16×16, 32×32, 64×64, 128×128 from the *same*
fine-scale (128×128) permeability field.  Each sample:
  - random log-normal permeability field
  - five-spot waterflood pattern (1 injector + 4 producers)
  - IMPES simulation with CFL-adaptive time stepping
  - 100 time slices of pressure + water saturation

Output layout:
    /mnt/bn/zzl-lf/czl/Dataset/
        16x16/  data.npz  metadata.json
        32x32/  data.npz  metadata.json
        64x64/  data.npz  metadata.json
        128x128/ data.npz metadata.json

Single process:
    python scripts/generate_twophase_data.py

Multi-process:
    python scripts/generate_twophase_data.py --shard 0 --num-shards 4 &
    python scripts/generate_twophase_data.py --shard 1 --num-shards 4 &
    python scripts/generate_twophase_data.py --shard 2 --num-shards 4 &
    python scripts/generate_twophase_data.py --shard 3 --num-shards 4 &
    wait
    python scripts/generate_twophase_data.py --merge --num-shards 4

Background:
    nohup python scripts/generate_twophase_data.py > generate.log 2>&1 &
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import json
import logging
import time

import numpy as np
from scipy.sparse import csr_matrix

from reservoirpy.mesh.mesh import StructuredMesh
from reservoirpy.physics.physics import TwoPhaseProperties
from reservoirpy.core.well_model import WellManager
from reservoirpy.core.discretization import FVMDiscretizer
from reservoirpy.core.linear_solver import LinearSolver
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
CFL_FACTOR = 0.8
MAX_SUB_STEPS = 5000
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


def resample_perm_field(perm: np.ndarray, target_nx: int, target_ny: int) -> np.ndarray:
    arr = np.asarray(perm, dtype=np.float64)
    squeeze = False
    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[1], arr.shape[2])
        squeeze = True
    ny_src, nx_src = arr.shape
    by = ny_src // target_ny
    bx = nx_src // target_nx
    log_k = np.log(np.maximum(arr, 1e-30))
    log_k_coarse = log_k.reshape(target_ny, by, target_nx, bx).mean(axis=(1, 3))
    coarse = np.exp(log_k_coarse)
    if squeeze:
        coarse = coarse[np.newaxis, :, :]
    return coarse


def remap_well(well_y: int, well_x: int, nx_src: int, ny_src: int,
               nx_dst: int, ny_dst: int):
    y_frac = (well_y + 0.5) / ny_src
    x_frac = (well_x + 0.5) / nx_src
    y_new = min(int(y_frac * ny_dst), ny_dst - 1)
    x_new = min(int(x_frac * nx_dst), nx_dst - 1)
    return y_new, x_new


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
    wells = [
        {"location": [0, iy, ix], "control_type": "bhp", "value": bhp_inj_Pa, "rw": 0.1, "skin_factor": 0},
        {"location": [0, 0, 0], "control_type": "bhp", "value": bhp_prod_Pa, "rw": 0.1, "skin_factor": 0},
        {"location": [0, 0, NX_FINE - 1], "control_type": "bhp", "value": bhp_prod_Pa, "rw": 0.1, "skin_factor": 0},
        {"location": [0, NY_FINE - 1, 0], "control_type": "bhp", "value": bhp_prod_Pa, "rw": 0.1, "skin_factor": 0},
        {"location": [0, NY_FINE - 1, NX_FINE - 1], "control_type": "bhp", "value": bhp_prod_Pa, "rw": 0.1, "skin_factor": 0},
    ]
    return wells


def remap_wells(wells_fine: list, nx: int, ny: int) -> list:
    out = []
    for w in wells_fine:
        z, y, x = w["location"]
        y_new, x_new = remap_well(y, x, NX_FINE, NY_FINE, nx, ny)
        w_new = dict(w)
        w_new["location"] = [0, y_new, x_new]
        out.append(w_new)
    return out


def compute_mobility(Sw: np.ndarray, physics) -> tuple:
    S_or = physics.kro_params['S_or']
    S_wr = physics.krw_params['S_wr']
    n_o = physics.kro_params['n_o']
    n_w = physics.krw_params['n_w']

    S_o_norm = np.clip((1.0 - Sw - S_or) / (1.0 - S_wr - S_or), 0.0, 1.0)
    S_w_norm = np.clip((Sw - S_wr) / (1.0 - S_wr - S_or), 0.0, 1.0)

    kro = np.where(Sw <= S_wr, 1.0, np.where(Sw >= 1.0 - S_or, 0.0, S_o_norm ** n_o))
    krw = np.where(Sw <= S_wr, 0.0, np.where(Sw >= 1.0 - S_or, 1.0, S_w_norm ** n_w))

    lambda_w = krw / physics.mu_w
    lambda_o = kro / physics.mu_o
    lambda_t = lambda_w + lambda_o
    f_w = lambda_w / (lambda_t + 1e-30)

    return lambda_w, lambda_o, lambda_t, f_w


def update_saturation(discretizer, pressure_old: np.ndarray,
                      pressure_new: np.ndarray, saturation_old: np.ndarray,
                      dt: float, well_manager, physics, mesh) -> np.ndarray:
    n = mesh.n_cells
    lambda_w, lambda_o, lambda_t, f_w = compute_mobility(saturation_old, physics)
    mu_ref = getattr(physics, 'viscosity', 1e-3)
    mobility_scale = mu_ref * lambda_t

    dSw = np.zeros(n)

    for d in range(6):
        ni = discretizer.neighbor_indices[d]
        valid = ni >= 0
        if not np.any(valid):
            continue
        ci = np.where(valid)[0]
        tv = discretizer.trans_matrix[d, ci]
        cj = ni[ci]

        ms_i = mobility_scale[ci]
        ms_j = mobility_scale[cj]
        ms_face = 2.0 * ms_i * ms_j / (ms_i + ms_j + 1e-30)

        dp = pressure_new[cj] - pressure_new[ci]
        upstream_j = dp >= 0
        fw_up = np.where(upstream_j, f_w[cj], f_w[ci])

        T_w = tv * ms_face * fw_up
        flux = T_w * dp
        np.add.at(dSw, ci, flux)

    for well in well_manager.wells:
        z, y, x = well.location
        cell_index = mesh.get_cell_index(z, y, x)
        ms_cell = mobility_scale[cell_index]

        if well.control_type == 'bhp':
            effective_wi = well.well_index * ms_cell
            q_total = effective_wi * (pressure_new[cell_index] - well.value)
        else:
            q_total = well.value

        if q_total < 0:
            dSw[cell_index] += abs(q_total)
        else:
            fw_cell = f_w[cell_index]
            dSw[cell_index] += fw_cell * q_total

    volumes = discretizer.volumes
    porosity_flat = discretizer.porosity_flat
    saturation_new = saturation_old + dt * dSw / (volumes * porosity_flat + 1e-30)
    saturation_new = np.clip(saturation_new, 0.0, 1.0)

    return saturation_new


def compute_cfl_dt(discretizer, pressure: np.ndarray, saturation: np.ndarray,
                   physics, mesh) -> float:
    lambda_w, lambda_o, lambda_t, f_w = compute_mobility(saturation, physics)
    mu_ref = getattr(physics, 'viscosity', 1e-3)
    mobility_scale = mu_ref * lambda_t

    dt_min = np.inf
    porosity_flat = discretizer.porosity_flat
    volumes = discretizer.volumes

    for d in range(6):
        ni = discretizer.neighbor_indices[d]
        valid = ni >= 0
        if not np.any(valid):
            continue
        ci = np.where(valid)[0]
        tv = discretizer.trans_matrix[d, ci]
        cj = ni[ci]

        ms_i = mobility_scale[ci]
        ms_j = mobility_scale[cj]
        ms_face = 2.0 * ms_i * ms_j / (ms_i + ms_j + 1e-30)

        dp = np.abs(pressure[cj] - pressure[ci])
        v_total = tv * ms_face * dp

        dSw_fd = 0.01
        Sw_i = saturation[ci]
        Sw_p = np.clip(Sw_i + dSw_fd, 0.0, 1.0)
        Sw_m = np.clip(Sw_i - dSw_fd, 0.0, 1.0)

        _, _, lt_p, _ = compute_mobility(Sw_p, physics)
        lw_p = compute_mobility(Sw_p, physics)[0]
        _, _, lt_m, _ = compute_mobility(Sw_m, physics)
        lw_m = compute_mobility(Sw_m, physics)[0]

        fw_p = lw_p / (lt_p + 1e-30)
        fw_m = lw_m / (lt_m + 1e-30)
        dfw_dSw = (fw_p - fw_m) / (2 * dSw_fd)

        active = v_total * np.abs(dfw_dSw) > 1e-30
        if np.any(active):
            dt_cells = porosity_flat[ci[active]] * volumes[ci[active]] / (
                v_total[active] * np.abs(dfw_dSw[active]))
            dt_min = min(dt_min, dt_cells.min())

    return dt_min * CFL_FACTOR if dt_min < np.inf else np.inf


def run_twophase_simulation(
    perm_mD: np.ndarray,
    wells_fine: list,
    cfg: dict,
    nx: int,
    ny: int,
) -> tuple:
    Lx = NX_FINE * DX
    Ly = NY_FINE * DY
    dx = Lx / nx
    dy = Ly / ny

    wells = remap_wells(wells_fine, nx, ny)

    if nx == NX_FINE and ny == NY_FINE:
        perm = perm_mD
    else:
        perm = resample_perm_field(perm_mD, nx, ny)
        if perm.ndim == 3:
            perm = perm.reshape(ny, nx)

    mesh = StructuredMesh(nx=nx, ny=ny, nz=1, dx=dx, dy=dy, dz=DZ)

    physics = TwoPhaseProperties(mesh, {
        "type": "two_phase",
        "permeability": perm.reshape(1, ny, nx),
        "porosity": cfg["porosity"],
        "compressibility": cfg["compressibility"],
        "oil_viscosity": uc.mpas_to_pas(cfg["mu_o_mPas"]),
        "water_viscosity": uc.mpas_to_pas(cfg["mu_w_mPas"]),
        "kro_params": {"n_o": cfg["n_o"], "S_or": cfg["S_or"]},
        "krw_params": {"n_w": cfg["n_w"], "S_wr": cfg["S_wr"]},
    })

    well_manager = WellManager(mesh, wells)
    k_field = physics.property_manager.properties["permeability"]
    if isinstance(k_field, float):
        k_field = np.full((1, ny, nx), k_field)
    well_manager.initialize_wells(k_field, physics.mu_w)

    discretizer = FVMDiscretizer(mesh, physics)
    solver = LinearSolver({"method": "direct"})

    initial_Pa = uc.mpa_to_pa(cfg["initial_pressure_MPa"])
    n_cells = nx * ny
    pressure = np.full(n_cells, initial_Pa, dtype=np.float64)
    saturation = np.full(n_cells, cfg["S_wr"], dtype=np.float64)

    snap_times = TIME_FRACTIONS * TOTAL_TIME
    p_results = [pressure.reshape(ny, nx).copy()]
    sw_results = [saturation.reshape(ny, nx).copy()]

    current_time = 0.0
    next_snap = 1
    sub_steps = 0

    while current_time < TOTAL_TIME and sub_steps < MAX_SUB_STEPS:
        if next_snap < N_TIME_SLICES:
            target = snap_times[next_snap]
        else:
            target = TOTAL_TIME

        remaining = target - current_time
        if remaining <= 0:
            next_snap += 1
            continue

        cfl_dt = compute_cfl_dt(discretizer, pressure, saturation, physics, mesh)
        if cfl_dt < 1.0:
            cfl_dt = 1.0
        if cfl_dt > remaining:
            cfl_dt = remaining

        A, b = discretizer.discretize_two_phase(cfl_dt, pressure, saturation, well_manager)
        new_pressure = solver.solve(A, b)
        new_saturation = update_saturation(
            discretizer, pressure, new_pressure, saturation,
            cfl_dt, well_manager, physics, mesh)

        pressure = new_pressure
        saturation = new_saturation
        current_time += cfl_dt
        sub_steps += 1

        while next_snap < N_TIME_SLICES and current_time >= snap_times[next_snap] - 1.0:
            p_results.append(pressure.reshape(ny, nx).copy())
            sw_results.append(saturation.reshape(ny, nx).copy())
            next_snap += 1

    while len(p_results) < N_TIME_SLICES:
        p_results.append(p_results[-1])
        sw_results.append(sw_results[-1])

    return np.stack(p_results), np.stack(sw_results)


def encode_wells(wells_fine: list, bhp_inj_MPa: float, bhp_prod_MPa: float,
                 nx: int, ny: int) -> tuple:
    well_mask = np.zeros((ny, nx), dtype=np.int8)
    well_value = np.zeros((ny, nx), dtype=np.float32)
    wells = remap_wells(wells_fine, nx, ny)
    for i, w in enumerate(wells):
        z, y, x = w["location"]
        if i == 0:
            well_mask[y, x] = 1
            well_value[y, x] = bhp_inj_MPa
        else:
            well_mask[y, x] = 2
            well_value[y, x] = bhp_prod_MPa
    return well_mask, well_value


def generate_shard(shard: int, num_shards: int, output_root: str):
    sample_indices = list(range(shard, N_SAMPLES, num_shards))
    n_shard = len(sample_indices)

    log_fmt = logging.Formatter(f"%(asctime)s [shard {shard}] %(message)s")
    for h in logging.getLogger().handlers:
        h.setFormatter(log_fmt)

    logger.info(f"Generating {n_shard} two-phase samples at resolutions {RESOLUTIONS} [CPU, IMPES]")

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
                pressure, saturation = run_twophase_simulation(
                    perm_mD=perm_128,
                    wells_fine=wells_fine,
                    cfg=cfg,
                    nx=res, ny=res,
                )
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

        pressure_MPa = np.stack(acc["pressure_MPa"])
        saturation = np.stack(acc["saturation"])
        permeability_mD = np.stack(acc["permeability_mD"])
        well_mask = np.stack(acc["well_mask"])
        well_value = np.stack(acc["well_value"])

        npz_path = os.path.join(out_dir, f"shard_{shard}.npz")
        np.savez_compressed(
            npz_path,
            pressure_MPa=pressure_MPa,
            saturation=saturation,
            permeability_mD=permeability_mD,
            well_mask=well_mask,
            well_value=well_value,
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

        all_pressure = []
        all_saturation = []
        all_perm = []
        all_mask = []
        all_value = []

        for f in existing:
            d = np.load(f)
            all_pressure.append(d["pressure_MPa"])
            all_saturation.append(d["saturation"])
            all_perm.append(d["permeability_mD"])
            all_mask.append(d["well_mask"])
            all_value.append(d["well_value"])

        pressure_MPa = np.concatenate(all_pressure, axis=0)
        saturation = np.concatenate(all_saturation, axis=0)
        permeability_mD = np.concatenate(all_perm, axis=0)
        well_mask = np.concatenate(all_mask, axis=0)
        well_value = np.concatenate(all_value, axis=0)

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
            "solver": "scipy_direct_impes",
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
        description="Two-phase IMPES multi-resolution data generator (CPU)")
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
