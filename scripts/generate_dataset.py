"""
PDE 数据集生成器 — 单相流 64×64

生成用于 Flow Matching / Diffusion 模型训练的数据集：
  条件 (ControlNet): 渗透率场 + 井配置 + 归一化时间
  目标: 压力场

每条样本包含 10 个时间切片，早期密集采样（变化快），晚期稀疏采样（趋于稳态）。

差异性策略：
  - 渗透率场：变差函数参数随机化（变程、方位角、基台值）
  - 井配置：随机数量(1~4)、随机位置、随机 BHP
  - 物理参数：粘度随机

输出格式：HuggingFace Dataset (Arrow)
"""

import os
import json
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from reservoirpy.utils.units import mpa_to_pa, mpas_to_pas, d_to_s, s_to_d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

GRID_SIZE = 64
DX = DY = 50.0
DZ = 10.0
N_TIME_SLICES = 10
TIME_FRACTIONS = np.array([0.01, 0.03, 0.06, 0.10, 0.15, 0.22, 0.32, 0.45, 0.65, 1.0])
TOTAL_TIME = d_to_s(365)


@dataclass
class SampleConfig:
    """单条样本的随机配置"""
    seed: int = 0
    major_range: float = 100.0
    minor_range: float = 80.0
    azimuth: float = 0.0
    sill: float = 1.0
    nugget: float = 0.05
    vtype: str = 'exponential'
    mean_log_perm: float = 2.0
    std_log_perm: float = 0.6
    viscosity_mPas: float = 1.0
    porosity: float = 0.2
    compressibility: float = 1e-9
    initial_pressure_MPa: float = 30.0
    n_wells: int = 2
    wells: List[Dict] = field(default_factory=list)


def random_config(seed: int, rng: np.random.Generator) -> SampleConfig:
    """生成随机样本配置"""
    cfg = SampleConfig(seed=seed)

    cfg.major_range = rng.uniform(50, 250)
    cfg.minor_range = rng.uniform(30, cfg.major_range)
    cfg.azimuth = rng.uniform(0, 180)
    cfg.sill = rng.uniform(0.3, 2.0)
    cfg.nugget = rng.uniform(0.0, 0.15)
    cfg.mean_log_perm = rng.uniform(1.0, 3.0)
    cfg.std_log_perm = rng.uniform(0.3, 0.8)
    cfg.viscosity_mPas = float(10 ** rng.uniform(-0.3, 1.0))
    cfg.porosity = rng.uniform(0.1, 0.35)
    cfg.initial_pressure_MPa = rng.uniform(25, 35)

    n_wells = rng.integers(1, 5)
    cfg.n_wells = n_wells

    wells = []
    used_positions = set()
    for _ in range(n_wells):
        for _attempt in range(50):
            x = int(rng.integers(0, GRID_SIZE))
            y = int(rng.integers(0, GRID_SIZE))
            if (x, y) not in used_positions:
                used_positions.add((x, y))
                break

        is_injector = bool(rng.random() < 0.4)
        if is_injector:
            bhp = cfg.initial_pressure_MPa + rng.uniform(5, 15)
        else:
            bhp = cfg.initial_pressure_MPa - rng.uniform(5, 15)
            bhp = max(bhp, 10.0)

        wells.append({
            'x': x, 'y': y,
            'is_injector': is_injector,
            'bhp_MPa': bhp,
            'rw': 0.1,
            'skin': 0.0,
        })
    cfg.wells = wells
    return cfg


def generate_perm_field(cfg: SampleConfig) -> np.ndarray:
    """生成渗透率场 (64, 64)，单位 mD"""
    from reservoirpy.geostatistics import PermeabilityGenerator

    gen = PermeabilityGenerator(nx=GRID_SIZE, ny=GRID_SIZE, dx=DX, dy=DY)
    perm = gen.generate(
        major_range=cfg.major_range,
        minor_range=cfg.minor_range,
        azimuth=cfg.azimuth,
        sill=cfg.sill,
        nugget=cfg.nugget,
        vtype=cfg.vtype,
        n_realizations=1,
        seed=cfg.seed,
        mean_log_perm=cfg.mean_log_perm,
        std_log_perm=cfg.std_log_perm,
    )
    return perm.squeeze()


def run_simulation(cfg: SampleConfig, perm_field: np.ndarray) -> List[np.ndarray]:
    """运行单相流模拟，返回 10 个时间切片的压力场"""
    from reservoirpy.mesh.mesh import StructuredMesh
    from reservoirpy.physics.physics import SinglePhaseProperties
    from reservoirpy.core.well_model import WellManager
    from reservoirpy.models.single_phase.single_phase_model import SinglePhaseModel

    nx = ny = GRID_SIZE
    mesh = StructuredMesh(nx=nx, ny=ny, nz=1, dx=DX, dy=DY, dz=DZ)

    physics = SinglePhaseProperties(mesh, {
        'type': 'single_phase',
        'permeability': perm_field.reshape(1, ny, nx),
        'porosity': cfg.porosity,
        'viscosity': mpas_to_pas(cfg.viscosity_mPas),
        'compressibility': cfg.compressibility,
    })

    wells_config = []
    for w in cfg.wells:
        wells_config.append({
            'location': [0, w['y'], w['x']],
            'control_type': 'bhp',
            'value': mpa_to_pa(w['bhp_MPa']),
            'rw': w['rw'],
            'skin_factor': w['skin'],
        })

    well_manager = WellManager(mesh, wells_config)
    k_field = physics.property_manager.properties['permeability']
    if isinstance(k_field, float):
        k_field = np.full((1, ny, nx), k_field)
    well_manager.initialize_wells(k_field, physics.viscosity)

    model = SinglePhaseModel(mesh, physics, {})
    state = model.initialize_state({
        'initial_pressure': mpa_to_pa(cfg.initial_pressure_MPa),
    })

    snap_times = TIME_FRACTIONS * TOTAL_TIME
    snap_set = set(range(N_TIME_SLICES))
    results = [None] * N_TIME_SLICES
    results[0] = state['pressure'].reshape(ny, nx).copy()

    current_time = 0.0
    step = 0
    dt = TOTAL_TIME / 100

    next_snap_idx = 1

    while current_time < TOTAL_TIME and next_snap_idx < N_TIME_SLICES:
        target_time = snap_times[next_snap_idx]
        actual_dt = min(dt, target_time - current_time)
        if actual_dt <= 0:
            next_snap_idx += 1
            continue

        state = model.solve_timestep(actual_dt, state, well_manager)
        model.update_properties(state)
        current_time += actual_dt
        step += 1

        while next_snap_idx < N_TIME_SLICES and current_time >= snap_times[next_snap_idx] - 1.0:
            results[next_snap_idx] = state['pressure'].reshape(ny, nx).copy()
            next_snap_idx += 1

    for i in range(N_TIME_SLICES):
        if results[i] is None:
            results[i] = state['pressure'].reshape(ny, nx).copy()

    return results


def encode_wells(cfg: SampleConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    编码井配置为网格张量

    Returns:
        well_mask: (64, 64, 2) — [is_injector, is_producer] one-hot
        well_value: (64, 64) — BHP 值 (MPa)，非井位置为 0
    """
    well_mask = np.zeros((GRID_SIZE, GRID_SIZE, 2), dtype=np.float32)
    well_value = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    for w in cfg.wells:
        x, y = w['x'], w['y']
        if w['is_injector']:
            well_mask[y, x, 0] = 1.0
        else:
            well_mask[y, x, 1] = 1.0
        well_value[y, x] = w['bhp_MPa']

    return well_mask, well_value


def generate_one_sample(seed: int) -> Dict:
    """生成一条完整样本"""
    rng = np.random.default_rng(seed)
    cfg = random_config(seed, rng)

    perm_field = generate_perm_field(cfg)
    pressure_slices = run_simulation(cfg, perm_field)
    well_mask, well_value = encode_wells(cfg)

    log_perm = np.log10(perm_field + 1e-10)
    perm_mean = log_perm.mean()
    perm_std = max(log_perm.std(), 1e-6)
    log_perm_norm = (log_perm - perm_mean) / perm_std

    p_all = np.stack(pressure_slices)
    p_mean = p_all.mean()
    p_std = max(p_all.std(), 1e-6)
    pressure_norm = (p_all - p_mean) / p_std

    t_normalized = TIME_FRACTIONS.astype(np.float32)

    sample = {
        'log_perm_norm': log_perm_norm.astype(np.float32),
        'well_mask': well_mask,
        'well_value': well_value,
        't_normalized': t_normalized,
        'pressure_norm': pressure_norm.astype(np.float32),
        'perm_mean': float(perm_mean),
        'perm_std': float(perm_std),
        'p_mean': float(p_mean),
        'p_std': float(p_std),
        'config': {
            'seed': cfg.seed,
            'major_range': cfg.major_range,
            'minor_range': cfg.minor_range,
            'azimuth': cfg.azimuth,
            'sill': cfg.sill,
            'mean_log_perm': cfg.mean_log_perm,
            'std_log_perm': cfg.std_log_perm,
            'viscosity_mPas': cfg.viscosity_mPas,
            'porosity': cfg.porosity,
            'initial_pressure_MPa': cfg.initial_pressure_MPa,
            'n_wells': cfg.n_wells,
            'wells': cfg.wells,
        },
    }
    return sample


def generate_dataset(
    n_samples: int = 1000,
    start_seed: int = 0,
    output_dir: str = './dataset_single_phase',
    n_workers: int = 1,
):
    """
    生成完整数据集

    Args:
        n_samples: 样本数
        start_seed: 起始随机种子
        output_dir: 输出目录
        n_workers: 并行进程数
    """
    os.makedirs(output_dir, exist_ok=True)

    seeds = list(range(start_seed, start_seed + n_samples))

    all_samples = []
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(generate_one_sample, s): s for s in seeds}
            for future in tqdm(futures, desc="Generating samples", total=n_samples):
                try:
                    sample = future.result()
                    all_samples.append(sample)
                except Exception as e:
                    logger.error(f"Sample failed: {e}")
    else:
        for seed in tqdm(seeds, desc="Generating samples"):
            try:
                sample = generate_one_sample(seed)
                all_samples.append(sample)
            except Exception as e:
                logger.error(f"Sample seed={seed} failed: {e}")

    logger.info(f"Generated {len(all_samples)} samples, saving...")

    log_perm = np.stack([s['log_perm_norm'] for s in all_samples])
    well_mask = np.stack([s['well_mask'] for s in all_samples])
    well_value = np.stack([s['well_value'] for s in all_samples])
    t_normalized = np.stack([s['t_normalized'] for s in all_samples])
    pressure = np.stack([s['pressure_norm'] for s in all_samples])

    np.savez_compressed(
        os.path.join(output_dir, 'data.npz'),
        log_perm_norm=log_perm,
        well_mask=well_mask,
        well_value=well_value,
        t_normalized=t_normalized,
        pressure_norm=pressure,
    )

    configs = [s['config'] for s in all_samples]
    stats = {
        'perm_mean': [s['perm_mean'] for s in all_samples],
        'perm_std': [s['perm_std'] for s in all_samples],
        'p_mean': [s['p_mean'] for s in all_samples],
        'p_std': [s['p_std'] for s in all_samples],
    }
    with open(os.path.join(output_dir, 'configs.json'), 'w') as f:
        json.dump({'configs': configs, 'stats': stats}, f, indent=2, default=_json_default)

    metadata = {
        'n_samples': len(all_samples),
        'grid_size': GRID_SIZE,
        'n_time_slices': N_TIME_SLICES,
        'time_fractions': TIME_FRACTIONS.tolist(),
        'total_time_days': s_to_d(TOTAL_TIME),
        'dx': DX, 'dy': DY, 'dz': DZ,
        'shapes': {
            'log_perm_norm': f'({len(all_samples)}, {GRID_SIZE}, {GRID_SIZE})',
            'well_mask': f'({len(all_samples)}, {GRID_SIZE}, {GRID_SIZE}, 2)',
            'well_value': f'({len(all_samples)}, {GRID_SIZE}, {GRID_SIZE})',
            't_normalized': f'({len(all_samples)}, {N_TIME_SLICES})',
            'pressure_norm': f'({len(all_samples)}, {N_TIME_SLICES}, {GRID_SIZE}, {GRID_SIZE})',
        },
        'description': {
            'log_perm_norm': 'Normalized log10(permeability) field',
            'well_mask': 'Well location mask [injector, producer]',
            'well_value': 'Well BHP values in MPa (0 where no well)',
            't_normalized': 'Normalized time fractions for each slice',
            'pressure_norm': 'Normalized pressure field at each time slice',
        },
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dataset saved to {output_dir}/")
    logger.info(f"  data.npz: {os.path.getsize(os.path.join(output_dir, 'data.npz')) / 1e6:.1f} MB")
    logger.info(f"  configs.json: well configurations")
    logger.info(f"  metadata.json: dataset metadata")

    return all_samples


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate single-phase PDE dataset')
    parser.add_argument('--n-samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--start-seed', type=int, default=0, help='Starting seed')
    parser.add_argument('--output-dir', type=str, default='./dataset_single_phase',
                        help='Output directory')
    parser.add_argument('--n-workers', type=int, default=1, help='Parallel workers')
    args = parser.parse_args()

    generate_dataset(
        n_samples=args.n_samples,
        start_seed=args.start_seed,
        output_dir=args.output_dir,
        n_workers=args.n_workers,
    )
