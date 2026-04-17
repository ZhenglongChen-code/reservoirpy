"""
两相流并行生成示例

20 个样本，32×32 网格，2 口对角井（1注1采），随机渗透率，1 年模拟。
对比单线程 vs 4 workers 并行的速度。

用法:
    uv run --extra geostat python scripts/example_parallel_two_phase.py
"""

import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from reservoirpy.utils.units import uc

NX = NY = 32
NZ = 1
DX = DY = 50.0
DZ = 10.0
TOTAL_TIME_DAYS = 365.0
TOTAL_TIME = uc.d_to_s(TOTAL_TIME_DAYS)
N_TIME_SLICES = 50
N_SAMPLES = 20
N_WORKERS = 4


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


def generate_one_two_phase(seed: int) -> dict:
    from reservoirpy.mesh.mesh import StructuredMesh
    from reservoirpy.physics.physics import TwoPhaseProperties
    from reservoirpy.core.well_model import WellManager
    from reservoirpy.models.two_phase_impes import TwoPhaseIMPES
    from reservoirpy.geostatistics import PermeabilityGenerator

    rng = np.random.default_rng(seed)
    mesh = StructuredMesh(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ)

    major_range = rng.uniform(150, 500)
    minor_range = rng.uniform(80, major_range)
    azimuth = rng.uniform(0, 180)
    sill = rng.uniform(0.3, 1.0)
    nugget = rng.uniform(0.0, 0.05)
    mean_log_perm = rng.uniform(1.5, 2.8)
    std_log_perm = rng.uniform(0.2, 0.5)

    gen = PermeabilityGenerator(nx=NX, ny=NY, dx=DX, dy=DY)
    perm = gen.generate(
        major_range=major_range,
        minor_range=minor_range,
        azimuth=azimuth,
        sill=sill,
        nugget=nugget,
        vtype="exponential",
        n_realizations=1,
        seed=seed,
        mean_log_perm=mean_log_perm,
        std_log_perm=std_log_perm,
    ).squeeze()

    porosity = rng.uniform(0.15, 0.25)
    initial_pressure_MPa = rng.uniform(25, 35)
    initial_pressure = uc.mpa_to_pa(initial_pressure_MPa)
    compressibility = 10 ** rng.uniform(-10, -8)

    oil_viscosity = uc.mpas_to_pas(rng.uniform(2.0, 8.0))
    water_viscosity = uc.mpas_to_pas(rng.uniform(0.5, 1.5))

    inj_bhp_MPa = initial_pressure_MPa + rng.uniform(5, 15)
    prod_bhp_MPa = initial_pressure_MPa - rng.uniform(5, 15)
    prod_bhp_MPa = max(prod_bhp_MPa, 5.0)

    wells_config = [
        {"location": [0, 2, 2], "control_type": "bhp",
         "value": uc.mpa_to_pa(inj_bhp_MPa), "rw": 0.1, "skin_factor": 0},
        {"location": [0, NY - 3, NX - 3], "control_type": "bhp",
         "value": uc.mpa_to_pa(prod_bhp_MPa), "rw": 0.1, "skin_factor": 0},
    ]

    physics = TwoPhaseProperties(mesh, {
        "type": "two_phase_impes",
        "permeability": perm.reshape(1, NY, NX),
        "porosity": porosity,
        "compressibility": compressibility,
        "oil_viscosity": oil_viscosity,
        "water_viscosity": water_viscosity,
    })

    well_manager = WellManager(mesh, wells_config)
    k_field = physics.property_manager.properties["permeability"]
    if isinstance(k_field, float):
        k_field = np.full((1, NY, NX), k_field)
    well_manager.initialize_wells(k_field, physics.viscosity)

    model = TwoPhaseIMPES(mesh, physics, {"cfl_factor": 0.8})
    state = model.initialize_state({
        "initial_pressure": initial_pressure,
        "initial_saturation": 0.2,
    })

    snap_times = TIME_FRACTIONS * TOTAL_TIME
    pressure_results = [state["pressure"].reshape(NY, NX).copy()]
    saturation_results = [state["saturation"].reshape(NY, NX).copy()]
    current_time = 0.0
    next_snap = 1

    while current_time < TOTAL_TIME and next_snap < N_TIME_SLICES:
        cfl_dt = model.compute_cfl_timestep(
            state["pressure"], state["saturation"], well_manager
        )
        target = snap_times[next_snap]
        actual_dt = min(cfl_dt, target - current_time, uc.d_to_s(30))
        if actual_dt <= 0:
            next_snap += 1
            continue

        state = model.solve_timestep(actual_dt, state, well_manager)
        model.update_properties(state)
        current_time += actual_dt

        while next_snap < N_TIME_SLICES and current_time >= snap_times[next_snap] - 1.0:
            pressure_results.append(state["pressure"].reshape(NY, NX).copy())
            saturation_results.append(state["saturation"].reshape(NY, NX).copy())
            next_snap += 1

    while len(pressure_results) < N_TIME_SLICES:
        pressure_results.append(pressure_results[-1])
        saturation_results.append(saturation_results[-1])

    pressure_arr = np.stack(pressure_results)
    saturation_arr = np.stack(saturation_results)

    return {
        "seed": seed,
        "pressure_MPa": uc.pa_to_mpa(pressure_arr).astype(np.float32),
        "saturation": saturation_arr.astype(np.float32),
        "permeability_mD": perm.astype(np.float32),
        "inj_bhp_MPa": inj_bhp_MPa,
        "prod_bhp_MPa": prod_bhp_MPa,
        "initial_pressure_MPa": initial_pressure_MPa,
    }


def run_serial() -> list:
    print(f"\n{'='*60}")
    print(f"  串行模式: {N_SAMPLES} 个样本, 单线程")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    results = []
    for i in range(N_SAMPLES):
        r = generate_one_two_phase(i)
        results.append(r)
        print(f"  [{i+1}/{N_SAMPLES}] seed={i} "
              f"P=[{r['pressure_MPa'].min():.1f}, {r['pressure_MPa'].max():.1f}] MPa "
              f"Sw=[{r['saturation'].min():.3f}, {r['saturation'].max():.3f}]")
    elapsed = time.perf_counter() - t0
    print(f"  串行耗时: {elapsed:.1f}s")
    return results, elapsed


def run_parallel(n_workers: int) -> list:
    print(f"\n{'='*60}")
    print(f"  并行模式: {N_SAMPLES} 个样本, {n_workers} workers")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(generate_one_two_phase, range(N_SAMPLES)))
    elapsed = time.perf_counter() - t0
    for i, r in enumerate(results):
        print(f"  [{i+1}/{N_SAMPLES}] seed={i} "
              f"P=[{r['pressure_MPa'].min():.1f}, {r['pressure_MPa'].max():.1f}] MPa "
              f"Sw=[{r['saturation'].min():.3f}, {r['saturation'].max():.3f}]")
    print(f"  并行耗时 ({n_workers} workers): {elapsed:.1f}s")
    return results, elapsed


def main():
    print("两相流并行生成示例")
    print(f"  网格: {NX}x{NY}x{NZ}")
    print(f"  井位: 注入井(2,2) + 生产井({NX-3},{NY-3})")
    print(f"  模拟时间: {TOTAL_TIME_DAYS:.0f} 天")
    print(f"  时间切片: {N_TIME_SLICES}")
    print(f"  样本数: {N_SAMPLES}")

    results_serial, t_serial = run_serial()
    results_parallel, t_parallel = run_parallel(N_WORKERS)

    speedup = t_serial / t_parallel
    print(f"\n{'='*60}")
    print(f"  性能对比")
    print(f"{'='*60}")
    print(f"  串行:      {t_serial:.1f}s")
    print(f"  并行({N_WORKERS}w): {t_parallel:.1f}s")
    print(f"  加速比:    {speedup:.2f}x")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(output_dir, exist_ok=True)

    pressure = np.stack([r["pressure_MPa"] for r in results_parallel])
    saturation = np.stack([r["saturation"] for r in results_parallel])
    perm = np.stack([r["permeability_mD"] for r in results_parallel])

    npz_path = os.path.join(output_dir, "two_phase_parallel_example.npz")
    np.savez_compressed(
        npz_path,
        pressure_MPa=pressure,
        saturation=saturation,
        permeability_mD=perm,
        time_days=(TIME_FRACTIONS * TOTAL_TIME_DAYS).astype(np.float32),
    )
    print(f"\n  数据已保存: {npz_path} ({os.path.getsize(npz_path)/1e6:.1f} MB)")
    print(f"  pressure_MPa shape: {pressure.shape}")
    print(f"  saturation shape:   {saturation.shape}")
    print(f"  permeability_mD shape: {perm.shape}")


if __name__ == "__main__":
    main()
