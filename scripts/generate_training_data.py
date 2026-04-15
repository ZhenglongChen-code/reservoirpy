"""
训练数据集生成器 — 32×32 单相流

生成 100 条样本，每条包含:
  - 随机渗透率场 (SGSIM)
  - 随机单井位置 + BHP
  - 3 年模拟，100 个时间切片（早期密集 + 晚期稀疏）

输出: data/training_32x32.npz
"""

import os
import json
import numpy as np
import logging
import time
from reservoirpy.utils.units import uc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

NX = NY = 32
NZ = 1
DX = DY = 50.0
DZ = 10.0
N_TIME_SLICES = 50
TOTAL_TIME_DAYS = 365.0
TOTAL_TIME = uc.d_to_s(TOTAL_TIME_DAYS)
N_SIM_STEPS = 200
N_SAMPLES = 100


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

    major_range = rng.uniform(150, 500)
    minor_range = rng.uniform(80, major_range)
    azimuth = rng.uniform(0, 180)
    sill = rng.uniform(0.3, 1.0)
    nugget = rng.uniform(0.0, 0.05)
    mean_log_perm = rng.uniform(1.5, 2.8)
    std_log_perm = rng.uniform(0.2, 0.5)

    viscosity_mPas = float(10 ** rng.uniform(-0.5, 0.7))
    porosity = rng.uniform(0.1, 0.3)
    compressibility = 10 ** rng.uniform(-10, -8)
    initial_pressure_MPa = rng.uniform(20, 35)

    x = int(rng.integers(0, NX))
    y = int(rng.integers(0, NY))
    is_injector = False
    bhp_MPa = initial_pressure_MPa - rng.uniform(5, 20)
    bhp_MPa = max(bhp_MPa, 5.0)

    return {
        "seed": seed,
        "major_range": major_range,
        "minor_range": minor_range,
        "azimuth": azimuth,
        "sill": sill,
        "nugget": nugget,
        "mean_log_perm": mean_log_perm,
        "std_log_perm": std_log_perm,
        "viscosity_mPas": viscosity_mPas,
        "porosity": porosity,
        "compressibility": compressibility,
        "initial_pressure_MPa": initial_pressure_MPa,
        "well_x": x,
        "well_y": y,
        "is_injector": is_injector,
        "bhp_MPa": bhp_MPa,
    }


def generate_perm_field(cfg: dict) -> np.ndarray:
    from reservoirpy.geostatistics import PermeabilityGenerator

    gen = PermeabilityGenerator(nx=NX, ny=NY, dx=DX, dy=DY)
    perm = gen.generate(
        major_range=cfg["major_range"],
        minor_range=cfg["minor_range"],
        azimuth=cfg["azimuth"],
        sill=cfg["sill"],
        nugget=cfg["nugget"],
        vtype="exponential",
        n_realizations=1,
        seed=cfg["seed"],
        mean_log_perm=cfg["mean_log_perm"],
        std_log_perm=cfg["std_log_perm"],
    )
    return perm.squeeze()


def run_simulation(cfg: dict, perm: np.ndarray) -> np.ndarray:
    from reservoirpy.mesh.mesh import StructuredMesh
    from reservoirpy.physics.physics import SinglePhaseProperties
    from reservoirpy.core.well_model import WellManager
    from reservoirpy.models.single_phase.single_phase_model import SinglePhaseModel

    mesh = StructuredMesh(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ)

    physics = SinglePhaseProperties(mesh, {
        "type": "single_phase",
        "permeability": perm.reshape(1, NY, NX),
        "porosity": cfg["porosity"],
        "viscosity": uc.mpas_to_pas(cfg["viscosity_mPas"]),
        "compressibility": cfg["compressibility"],
    })

    wells_config = [{
        "location": [0, cfg["well_y"], cfg["well_x"]],
        "control_type": "bhp",
        "value": uc.mpa_to_pa(cfg["bhp_MPa"]),
        "rw": 0.1,
        "skin_factor": 0,
    }]
    well_manager = WellManager(mesh, wells_config)
    k_field = physics.property_manager.properties["permeability"]
    if isinstance(k_field, float):
        k_field = np.full((1, NY, NX), k_field)
    well_manager.initialize_wells(k_field, physics.viscosity)

    model = SinglePhaseModel(mesh, physics, {})
    state = model.initialize_state({
        "initial_pressure": uc.mpa_to_pa(cfg["initial_pressure_MPa"]),
    })

    snap_times = TIME_FRACTIONS * TOTAL_TIME
    results = [state["pressure"].reshape(NY, NX).copy()]
    current_time = 0.0
    next_snap = 1
    dt = TOTAL_TIME / N_SIM_STEPS

    while current_time < TOTAL_TIME and next_snap < N_TIME_SLICES:
        target = snap_times[next_snap]
        actual_dt = min(dt, target - current_time)
        if actual_dt <= 0:
            next_snap += 1
            continue
        state = model.solve_timestep(actual_dt, state, well_manager)
        model.update_properties(state)
        current_time += actual_dt
        while next_snap < N_TIME_SLICES and current_time >= snap_times[next_snap] - 1.0:
            results.append(state["pressure"].reshape(NY, NX).copy())
            next_snap += 1

    while len(results) < N_TIME_SLICES:
        results.append(results[-1])

    return np.stack(results)


def encode_well(cfg: dict) -> tuple:
    well_mask = np.zeros((NY, NX), dtype=np.int8)
    well_value = np.zeros((NY, NX), dtype=np.float32)
    well_mask[cfg["well_y"], cfg["well_x"]] = 1 if cfg["is_injector"] else 2
    well_value[cfg["well_y"], cfg["well_x"]] = cfg["bhp_MPa"]
    return well_mask, well_value


def generate_one_sample(seed: int) -> dict:
    cfg = random_config(seed)
    perm = generate_perm_field(cfg)
    pressure = run_simulation(cfg, perm)
    well_mask, well_value = encode_well(cfg)

    log_perm = np.log10(perm + 1e-10)
    perm_mean = log_perm.mean()
    perm_std = max(log_perm.std(), 1e-6)
    log_perm_norm = (log_perm - perm_mean) / perm_std

    p_mean = pressure.mean()
    p_std = max(pressure.std(), 1e-6)
    pressure_norm = (pressure - p_mean) / p_std

    return {
        "log_perm_norm": log_perm_norm.astype(np.float32),
        "pressure_norm": pressure_norm.astype(np.float32),
        "pressure_MPa": uc.pa_to_mpa(pressure).astype(np.float32),
        "permeability_mD": perm.astype(np.float32),
        "well_mask": well_mask,
        "well_value": well_value,
        "t_normalized": TIME_FRACTIONS.astype(np.float32),
        "time_days": (TIME_FRACTIONS * TOTAL_TIME_DAYS).astype(np.float32),
        "perm_mean": float(perm_mean),
        "perm_std": float(perm_std),
        "p_mean": float(p_mean),
        "p_std": float(p_std),
        "config": cfg,
    }


def main():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Generating {N_SAMPLES} samples, {NX}x{NY}, {N_TIME_SLICES} slices, {TOTAL_TIME_DAYS:.0f} days")
    logger.info(f"Time fractions: first={TIME_FRACTIONS[0]:.4f}, last={TIME_FRACTIONS[-1]:.4f}")
    logger.info(f"Output: {output_dir}")

    all_samples = []
    t0 = time.time()

    for i in range(N_SAMPLES):
        seed = i
        try:
            sample = generate_one_sample(seed)
            all_samples.append(sample)
        except Exception as e:
            logger.error(f"Sample {seed} failed: {e}")
            continue

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N_SAMPLES - i - 1)
            logger.info(f"  [{i+1}/{N_SAMPLES}] elapsed={elapsed:.1f}s, ETA={eta:.0f}s")

    logger.info(f"Generated {len(all_samples)} samples in {time.time()-t0:.1f}s")

    log_perm_norm = np.stack([s["log_perm_norm"] for s in all_samples])
    pressure_norm = np.stack([s["pressure_norm"] for s in all_samples])
    pressure_MPa = np.stack([s["pressure_MPa"] for s in all_samples])
    permeability_mD = np.stack([s["permeability_mD"] for s in all_samples])
    well_mask = np.stack([s["well_mask"] for s in all_samples])
    well_value = np.stack([s["well_value"] for s in all_samples])
    t_normalized = np.stack([s["t_normalized"] for s in all_samples])
    time_days = np.stack([s["time_days"] for s in all_samples])

    npz_path = os.path.join(output_dir, "training_32x32.npz")
    np.savez_compressed(
        npz_path,
        log_perm_norm=log_perm_norm,
        pressure_norm=pressure_norm,
        pressure_MPa=pressure_MPa,
        permeability_mD=permeability_mD,
        well_mask=well_mask,
        well_value=well_value,
        t_normalized=t_normalized,
        time_days=time_days,
    )
    npz_size = os.path.getsize(npz_path) / 1e6
    logger.info(f"Saved {npz_path} ({npz_size:.1f} MB)")

    configs = [s["config"] for s in all_samples]
    stats = {
        "perm_mean": [s["perm_mean"] for s in all_samples],
        "perm_std": [s["perm_std"] for s in all_samples],
        "p_mean": [s["p_mean"] for s in all_samples],
        "p_std": [s["p_std"] for s in all_samples],
    }
    with open(os.path.join(output_dir, "training_32x32_configs.json"), "w") as f:
        json.dump({"configs": configs, "stats": stats}, f, indent=2, default=float)

    metadata = {
        "n_samples": len(all_samples),
        "grid_size": NX,
        "n_time_slices": N_TIME_SLICES,
        "time_fractions": TIME_FRACTIONS.tolist(),
        "total_time_days": TOTAL_TIME_DAYS,
        "dx": DX, "dy": DY, "dz": DZ,
        "shapes": {
            "log_perm_norm": f"({len(all_samples)}, {NY}, {NX})",
            "pressure_norm": f"({len(all_samples)}, {N_TIME_SLICES}, {NY}, {NX})",
            "pressure_MPa": f"({len(all_samples)}, {N_TIME_SLICES}, {NY}, {NX})",
            "permeability_mD": f"({len(all_samples)}, {NY}, {NX})",
            "well_mask": f"({len(all_samples)}, {NY}, {NX})",
            "well_value": f"({len(all_samples)}, {NY}, {NX})",
            "t_normalized": f"({len(all_samples)}, {N_TIME_SLICES})",
            "time_days": f"({len(all_samples)}, {N_TIME_SLICES})",
        },
    }
    with open(os.path.join(output_dir, "training_32x32_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Done!")


if __name__ == "__main__":
    main()
