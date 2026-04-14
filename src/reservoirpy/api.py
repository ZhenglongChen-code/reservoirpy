"""
ReservoirPy API 服务

提供渗透率场生成、单相流/两相流模拟的 REST API。
支持阿里云 OSS 存储结果数据。

启动方式：
    uv run uvicorn reservoirpy.api:app --host 0.0.0.0 --port 8000

环境变量（OSS 配置）：
    OSS_ACCESS_KEY_ID      阿里云 AccessKey ID
    OSS_ACCESS_KEY_SECRET  阿里云 AccessKey Secret
    OSS_BUCKET_NAME        Bucket 名称
    OSS_ENDPOINT           OSS Endpoint，如 oss-cn-hangzhou.aliyuncs.com
    OSS_PREFIX             存储路径前缀，默认 reservoirpy/

API 文档：
    http://localhost:8000/docs
"""

import io
import os
import time
import json
import uuid
import logging
import numpy as np
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_tasks: Dict[str, Dict[str, Any]] = {}

OSS_CONFIG = {
    'access_key_id': os.environ.get('OSS_ACCESS_KEY_ID', ''),
    'access_key_secret': os.environ.get('OSS_ACCESS_KEY_SECRET', ''),
    'bucket_name': os.environ.get('OSS_BUCKET_NAME', ''),
    'endpoint': os.environ.get('OSS_ENDPOINT', ''),
    'prefix': os.environ.get('OSS_PREFIX', 'reservoirpy/'),
}


def _load_secrets():
    secrets_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), '.secrets.yaml'),
        os.path.join(os.getcwd(), '.secrets.yaml'),
    ]
    for path in secrets_paths:
        if os.path.exists(path):
            try:
                import yaml
                with open(path) as f:
                    secrets = yaml.safe_load(f) or {}
                for key in ['access_key_id', 'access_key_secret', 'bucket_name',
                            'endpoint', 'prefix']:
                    yaml_key = f'OSS_{key.upper()}'
                    if not OSS_CONFIG[key] and secrets.get(yaml_key):
                        OSS_CONFIG[key] = str(secrets[yaml_key])
                logger.info(f"Loaded secrets from {path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load secrets from {path}: {e}")


_load_secrets()


def _oss_available() -> bool:
    return bool(OSS_CONFIG['access_key_id'] and OSS_CONFIG['bucket_name']
                and OSS_CONFIG['endpoint'])


def _upload_to_oss(data: bytes, object_key: str) -> Optional[str]:
    if not _oss_available():
        return None
    try:
        import oss2
        auth = oss2.Auth(OSS_CONFIG['access_key_id'], OSS_CONFIG['access_key_secret'])
        bucket = oss2.Bucket(auth, OSS_CONFIG['endpoint'], OSS_CONFIG['bucket_name'])
        full_key = OSS_CONFIG['prefix'] + object_key
        bucket.put_object(full_key, data)
        signed_url = bucket.sign_url('GET', full_key, 3600, slash_safe=True)
        logger.info(f"Uploaded to OSS: {full_key}")
        return signed_url
    except Exception as e:
        logger.error(f"OSS upload failed: {e}")
        return None


def _save_result_to_oss(result: Dict, prefix: str) -> Dict:
    """将模拟结果存为 npz 上传 OSS，返回轻量元数据"""
    arrays = {}
    metadata = {'grid': result.get('grid', {}), 'time_fractions': result.get('time_fractions', [])}

    if 'pressure_MPa' in result:
        arrays['pressure_MPa'] = np.array(result['pressure_MPa'], dtype=np.float32)
    if 'saturation' in result:
        arrays['saturation'] = np.array(result['saturation'], dtype=np.float32)
    if 'permeability_mD' in result:
        arrays['permeability_mD'] = np.array(result['permeability_mD'], dtype=np.float32)
    if 'well_mask' in result:
        arrays['well_mask'] = np.array(result['well_mask'], dtype=np.int8)
    if 'well_bhp_MPa' in result:
        arrays['well_bhp_MPa'] = np.array(result['well_bhp_MPa'], dtype=np.float32)
    if 'time_days' in result:
        arrays['time_days'] = np.array(result['time_days'], dtype=np.float32)

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    buf.seek(0)
    data = buf.read()

    object_key = f"{prefix}/{uuid.uuid4().hex[:8]}.npz"
    oss_url = _upload_to_oss(data, object_key)

    response = {
        'elapsed_seconds': result.get('elapsed_seconds', 0),
        'grid': result.get('grid', {}),
        'time_fractions': result.get('time_fractions', []),
        'data_size_bytes': len(data),
    }

    if oss_url:
        response['download_url'] = oss_url
        response['storage'] = 'oss'
    else:
        task_id = uuid.uuid4().hex[:8]
        _tasks[task_id] = {"status": "completed", "data": arrays}
        response['download_task_id'] = task_id
        response['storage'] = 'memory'

    if 'permeability_mD' in result:
        perm = np.array(result['permeability_mD'])
        log_perm = np.log10(perm + 1e-10)
        response['perm_stats'] = {
            'log_mean': float(log_perm.mean()),
            'log_std': float(log_perm.std()),
            'min_mD': float(perm.min()),
            'max_mD': float(perm.max()),
        }

    if 'pressure_MPa' in result:
        p = np.array(result['pressure_MPa'])
        response['pressure_stats'] = {
            'min_MPa': float(p.min()),
            'max_MPa': float(p.max()),
            'mean_MPa': float(p.mean()),
        }

    if 'saturation' in result:
        sw = np.array(result['saturation'])
        response['saturation_stats'] = {
            'min': float(sw.min()),
            'max': float(sw.max()),
            'mean': float(sw.mean()),
        }

    return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _oss_available():
        logger.info(f"OSS enabled: bucket={OSS_CONFIG['bucket_name']}, "
                    f"endpoint={OSS_CONFIG['endpoint']}")
    else:
        logger.warning("OSS not configured, using in-memory storage")
    logger.info("ReservoirPy API started")
    yield
    logger.info("ReservoirPy API stopped")


app = FastAPI(
    title="ReservoirPy API",
    description="油藏数值模拟云服务 — 渗透率场生成 + 单相流/两相流模拟 + OSS 存储",
    version="0.2.0",
    lifespan=lifespan,
)


class WellConfig(BaseModel):
    x: int = Field(ge=0, description="X 索引")
    y: int = Field(ge=0, description="Y 索引")
    is_injector: bool = Field(default=False, description="是否注入井")
    bhp_MPa: float = Field(gt=0, description="井底压力 (MPa)")


class PermFieldRequest(BaseModel):
    nx: int = Field(default=64, ge=8, le=128)
    ny: int = Field(default=64, ge=8, le=128)
    dx: float = Field(default=50.0, gt=0)
    dy: float = Field(default=50.0, gt=0)
    major_range: float = Field(default=120.0, gt=0)
    minor_range: float = Field(default=80.0, gt=0)
    azimuth: float = Field(default=0.0, ge=0, lt=180)
    sill: float = Field(default=1.0, gt=0)
    nugget: float = Field(default=0.05, ge=0)
    mean_log_perm: float = Field(default=2.0)
    std_log_perm: float = Field(default=0.6, gt=0)
    seed: int = Field(default=42)


class SinglePhaseRequest(BaseModel):
    nx: int = Field(default=64, ge=8, le=128)
    ny: int = Field(default=64, ge=8, le=128)
    dx: float = Field(default=50.0, gt=0)
    dy: float = Field(default=50.0, gt=0)
    permeability_mD: Optional[List[List[float]]] = Field(default=None)
    perm_seed: int = Field(default=42)
    major_range: float = Field(default=120.0, gt=0)
    minor_range: float = Field(default=80.0, gt=0)
    azimuth: float = Field(default=0.0, ge=0, lt=180)
    mean_log_perm: float = Field(default=2.0)
    std_log_perm: float = Field(default=0.6)
    viscosity_mPas: float = Field(default=1.0, gt=0)
    porosity: float = Field(default=0.2, gt=0, lt=1)
    initial_pressure_MPa: float = Field(default=30.0, gt=0)
    total_time_days: float = Field(default=365.0, gt=0)
    n_time_slices: int = Field(default=10, ge=1, le=50)
    wells: List[WellConfig] = Field(min_length=1)


class TwoPhaseRequest(BaseModel):
    nx: int = Field(default=64, ge=8, le=128)
    ny: int = Field(default=64, ge=8, le=128)
    dx: float = Field(default=50.0, gt=0)
    dy: float = Field(default=50.0, gt=0)
    permeability_mD: Optional[List[List[float]]] = Field(default=None)
    perm_seed: int = Field(default=42)
    major_range: float = Field(default=120.0, gt=0)
    minor_range: float = Field(default=80.0, gt=0)
    azimuth: float = Field(default=0.0, ge=0, lt=180)
    mean_log_perm: float = Field(default=2.0)
    std_log_perm: float = Field(default=0.6)
    oil_viscosity_mPas: float = Field(default=5.0, gt=0)
    water_viscosity_mPas: float = Field(default=1.0, gt=0)
    porosity: float = Field(default=0.2, gt=0, lt=1)
    initial_pressure_MPa: float = Field(default=30.0, gt=0)
    initial_saturation: float = Field(default=0.2, ge=0, le=1)
    total_time_days: float = Field(default=365.0, gt=0)
    n_time_slices: int = Field(default=10, ge=1, le=50)
    wells: List[WellConfig] = Field(min_length=1)


class DatasetRequest(BaseModel):
    n_samples: int = Field(default=10, ge=1, le=1000)
    grid_size: int = Field(default=64, ge=16, le=128)
    start_seed: int = Field(default=0)
    simulation_type: str = Field(default="single_phase",
                                 pattern="^(single_phase|two_phase)$")


def _generate_perm_field(req: PermFieldRequest) -> np.ndarray:
    from reservoirpy.geostatistics import PermeabilityGenerator
    gen = PermeabilityGenerator(nx=req.nx, ny=req.ny, dx=req.dx, dy=req.dy)
    perm = gen.generate(
        major_range=req.major_range, minor_range=req.minor_range,
        azimuth=req.azimuth, sill=req.sill, nugget=req.nugget,
        vtype='exponential', n_realizations=1, seed=req.seed,
        mean_log_perm=req.mean_log_perm, std_log_perm=req.std_log_perm,
    )
    return perm.squeeze()


def _get_time_fractions(n: int) -> np.ndarray:
    if n == 1:
        return np.array([1.0])
    t = np.concatenate([
        np.geomspace(0.01, 0.3, max(n // 2, 1), endpoint=False),
        np.linspace(0.3, 1.0, n - n // 2),
    ])
    return np.unique(np.round(t, 6))[:n]


def _run_single_phase(req: SinglePhaseRequest) -> Dict:
    from reservoirpy.mesh.mesh import StructuredMesh
    from reservoirpy.physics.physics import SinglePhaseProperties
    from reservoirpy.core.well_model import WellManager
    from reservoirpy.models.single_phase.single_phase_model import SinglePhaseModel

    nx, ny = req.nx, req.ny
    mesh = StructuredMesh(nx=nx, ny=ny, nz=1, dx=req.dx, dy=req.dy, dz=10.0)

    if req.permeability_mD is not None:
        perm = np.array(req.permeability_mD)
    else:
        perm = _generate_perm_field(PermFieldRequest(
            nx=nx, ny=ny, dx=req.dx, dy=req.dy,
            major_range=req.major_range, minor_range=req.minor_range,
            azimuth=req.azimuth, mean_log_perm=req.mean_log_perm,
            std_log_perm=req.std_log_perm, seed=req.perm_seed,
        ))

    physics = SinglePhaseProperties(mesh, {
        'type': 'single_phase',
        'permeability': perm.reshape(1, ny, nx),
        'porosity': req.porosity,
        'viscosity': req.viscosity_mPas * 1e-3,
        'compressibility': 1e-9,
    })

    wells_config = [{'location': [0, w.y, w.x], 'control_type': 'bhp',
                     'value': w.bhp_MPa * 1e6, 'rw': 0.1, 'skin_factor': 0}
                    for w in req.wells]

    well_manager = WellManager(mesh, wells_config)
    k_field = physics.property_manager.properties['permeability']
    if isinstance(k_field, float):
        k_field = np.full((1, ny, nx), k_field)
    well_manager.initialize_wells(k_field, physics.viscosity)

    model = SinglePhaseModel(mesh, physics, {})
    state = model.initialize_state({'initial_pressure': req.initial_pressure_MPa * 1e6})

    total_time = req.total_time_days * 86400.0
    time_fracs = _get_time_fractions(req.n_time_slices)
    snap_times = time_fracs * total_time

    results = [state['pressure'].reshape(ny, nx).copy()]
    current_time = 0.0
    next_snap = 1
    dt = total_time / 100

    while current_time < total_time and next_snap < len(snap_times):
        target = snap_times[next_snap]
        actual_dt = min(dt, target - current_time)
        if actual_dt <= 0:
            next_snap += 1
            continue
        state = model.solve_timestep(actual_dt, state, well_manager)
        model.update_properties(state)
        current_time += actual_dt
        while next_snap < len(snap_times) and current_time >= snap_times[next_snap] - 1.0:
            results.append(state['pressure'].reshape(ny, nx).copy())
            next_snap += 1

    while len(results) < len(snap_times):
        results.append(results[-1])

    well_mask = np.zeros((ny, nx), dtype=np.int8)
    well_bhp = np.zeros((ny, nx), dtype=np.float32)
    for w in req.wells:
        well_mask[w.y, w.x] = 1 if w.is_injector else 2
        well_bhp[w.y, w.x] = w.bhp_MPa

    p_all = np.stack(results)
    return {
        'pressure_MPa': (p_all / 1e6),
        'time_days': (time_fracs * req.total_time_days).astype(np.float32),
        'time_fractions': time_fracs.tolist(),
        'permeability_mD': perm,
        'well_mask': well_mask,
        'well_bhp_MPa': well_bhp,
        'grid': {'nx': nx, 'ny': ny, 'dx': req.dx, 'dy': req.dy},
    }


def _run_two_phase(req: TwoPhaseRequest) -> Dict:
    from reservoirpy.mesh.mesh import StructuredMesh
    from reservoirpy.physics.physics import TwoPhaseProperties
    from reservoirpy.core.well_model import WellManager
    from reservoirpy.models.two_phase_impes import TwoPhaseIMPES

    nx, ny = req.nx, req.ny
    mesh = StructuredMesh(nx=nx, ny=ny, nz=1, dx=req.dx, dy=req.dy, dz=10.0)

    if req.permeability_mD is not None:
        perm = np.array(req.permeability_mD)
    else:
        perm = _generate_perm_field(PermFieldRequest(
            nx=nx, ny=ny, dx=req.dx, dy=req.dy,
            major_range=req.major_range, minor_range=req.minor_range,
            azimuth=req.azimuth, mean_log_perm=req.mean_log_perm,
            std_log_perm=req.std_log_perm, seed=req.perm_seed,
        ))

    physics = TwoPhaseProperties(mesh, {
        'type': 'two_phase_impes',
        'permeability': perm.reshape(1, ny, nx),
        'porosity': req.porosity,
        'compressibility': 1e-9,
        'oil_viscosity': req.oil_viscosity_mPas * 1e-3,
        'water_viscosity': req.water_viscosity_mPas * 1e-3,
    })

    wells_config = [{'location': [0, w.y, w.x], 'control_type': 'bhp',
                     'value': w.bhp_MPa * 1e6, 'rw': 0.1, 'skin_factor': 0}
                    for w in req.wells]

    well_manager = WellManager(mesh, wells_config)
    k_field = physics.property_manager.properties['permeability']
    if isinstance(k_field, float):
        k_field = np.full((1, ny, nx), k_field)
    well_manager.initialize_wells(k_field, physics.viscosity)

    model = TwoPhaseIMPES(mesh, physics, {'cfl_factor': 0.8})
    state = model.initialize_state({
        'initial_pressure': req.initial_pressure_MPa * 1e6,
        'initial_saturation': req.initial_saturation,
    })

    total_time = req.total_time_days * 86400.0
    dt_max = 30 * 86400.0
    time_fracs = _get_time_fractions(req.n_time_slices)
    snap_times = time_fracs * total_time

    p_results = [state['pressure'].reshape(ny, nx).copy()]
    sw_results = [state['saturation'].reshape(ny, nx).copy()]
    current_time = 0.0
    next_snap = 1

    while current_time < total_time and next_snap < len(snap_times):
        cfl_dt = model.compute_cfl_timestep(
            state['pressure'], state['saturation'], well_manager)
        target = snap_times[next_snap]
        actual_dt = min(dt_max, cfl_dt, target - current_time, total_time - current_time)
        actual_dt = max(actual_dt, 3600.0)
        state = model.solve_timestep(actual_dt, state, well_manager)
        model.update_properties(state)
        current_time += actual_dt
        while next_snap < len(snap_times) and current_time >= snap_times[next_snap] - 1.0:
            p_results.append(state['pressure'].reshape(ny, nx).copy())
            sw_results.append(state['saturation'].reshape(ny, nx).copy())
            next_snap += 1

    while len(p_results) < len(snap_times):
        p_results.append(p_results[-1])
        sw_results.append(sw_results[-1])

    well_mask = np.zeros((ny, nx), dtype=np.int8)
    well_bhp = np.zeros((ny, nx), dtype=np.float32)
    for w in req.wells:
        well_mask[w.y, w.x] = 1 if w.is_injector else 2
        well_bhp[w.y, w.x] = w.bhp_MPa

    return {
        'pressure_MPa': np.stack(p_results) / 1e6,
        'saturation': np.stack(sw_results),
        'time_days': (time_fracs * req.total_time_days).astype(np.float32),
        'time_fractions': time_fracs.tolist(),
        'permeability_mD': perm,
        'well_mask': well_mask,
        'well_bhp_MPa': well_bhp,
        'grid': {'nx': nx, 'ny': ny, 'dx': req.dx, 'dy': req.dy},
    }


@app.get("/")
async def root():
    return {
        "service": "ReservoirPy API",
        "version": "0.2.0",
        "docs": "/docs",
        "oss_enabled": _oss_available(),
    }


@app.get("/health")
async def health():
    import psutil
    mem = psutil.virtual_memory()
    return {
        "status": "ok",
        "memory_total_GB": round(mem.total / 1e9, 2),
        "memory_available_GB": round(mem.available / 1e9, 2),
        "memory_percent": mem.percent,
        "cpu_count": psutil.cpu_count(),
        "oss_enabled": _oss_available(),
    }


@app.post("/generate-permeability")
async def generate_permeability(req: PermFieldRequest):
    t0 = time.time()
    try:
        perm = _generate_perm_field(req)
        log_perm = np.log10(perm + 1e-10)

        buf = io.BytesIO()
        np.savez_compressed(buf, permeability_mD=perm.astype(np.float32))
        buf.seek(0)
        data = buf.read()

        object_key = f"perm/{req.nx}x{req.ny}_seed{req.seed}/{uuid.uuid4().hex[:8]}.npz"
        oss_url = _upload_to_oss(data, object_key)

        response = {
            "log_perm_mean": float(log_perm.mean()),
            "log_perm_std": float(log_perm.std()),
            "perm_min_mD": float(perm.min()),
            "perm_max_mD": float(perm.max()),
            "grid": {"nx": req.nx, "ny": req.ny, "dx": req.dx, "dy": req.dy},
            "data_size_bytes": len(data),
            "elapsed_seconds": round(time.time() - t0, 3),
        }

        if oss_url:
            response['download_url'] = oss_url
            response['storage'] = 'oss'
        else:
            task_id = uuid.uuid4().hex[:8]
            _tasks[task_id] = {"status": "completed", "data": {"permeability_mD": perm}}
            response['download_task_id'] = task_id
            response['storage'] = 'memory'

        return response
    except Exception as e:
        logger.error(f"Permeability generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate/single-phase")
async def simulate_single_phase(req: SinglePhaseRequest):
    t0 = time.time()
    try:
        result = _run_single_phase(req)
        result['elapsed_seconds'] = round(time.time() - t0, 3)
        return _save_result_to_oss(result, "single-phase")
    except Exception as e:
        logger.error(f"Single-phase simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate/two-phase")
async def simulate_two_phase(req: TwoPhaseRequest):
    t0 = time.time()
    try:
        result = _run_two_phase(req)
        result['elapsed_seconds'] = round(time.time() - t0, 3)
        return _save_result_to_oss(result, "two-phase")
    except Exception as e:
        logger.error(f"Two-phase simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dataset/generate")
async def generate_dataset(req: DatasetRequest, background_tasks: BackgroundTasks):
    task_id = uuid.uuid4().hex[:8]
    _tasks[task_id] = {"status": "pending", "progress": 0, "n_samples": req.n_samples}
    background_tasks.add_task(_generate_dataset_task, task_id, req)
    return {"task_id": task_id, "status": "started", "n_samples": req.n_samples}


@app.get("/dataset/status/{task_id}")
async def dataset_status(task_id: str):
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = _tasks[task_id]
    resp = {k: v for k, v in task.items() if k != 'data'}
    return resp


@app.get("/dataset/download/{task_id}")
async def dataset_download(task_id: str):
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = _tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task status: {task['status']}")
    buf = io.BytesIO()
    np.savez_compressed(buf, **task["data"])
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=dataset_{task_id}.npz"},
    )


def _generate_dataset_task(task_id: str, req: DatasetRequest):
    try:
        _tasks[task_id]["status"] = "running"
        from scripts.generate_dataset import generate_one_sample

        samples = []
        for i in range(req.n_samples):
            sample = generate_one_sample(req.start_seed + i)
            samples.append(sample)
            _tasks[task_id]["progress"] = (i + 1) / req.n_samples

        data = {
            'log_perm_norm': np.stack([s['log_perm_norm'] for s in samples]),
            'pressure_norm': np.stack([s['pressure_norm'] for s in samples]),
            'well_mask': np.stack([s['well_mask'] for s in samples]),
            'well_value': np.stack([s['well_value'] for s in samples]),
            't_normalized': np.stack([s['t_normalized'] for s in samples]),
        }

        buf = io.BytesIO()
        np.savez_compressed(buf, **data)
        buf.seek(0)
        npz_bytes = buf.read()

        object_key = f"dataset/{req.simulation_type}/{req.grid_size}x{req.grid_size}/{task_id}.npz"
        oss_url = _upload_to_oss(npz_bytes, object_key)

        _tasks[task_id]["status"] = "completed"
        _tasks[task_id]["n_completed"] = req.n_samples
        _tasks[task_id]["data_size_MB"] = round(len(npz_bytes) / 1e6, 2)

        if oss_url:
            _tasks[task_id]["download_url"] = oss_url
            _tasks[task_id]["storage"] = "oss"
        else:
            _tasks[task_id]["data"] = data
            _tasks[task_id]["storage"] = "memory"

    except Exception as e:
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)
