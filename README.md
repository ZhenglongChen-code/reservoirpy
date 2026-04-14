# ReservoirPy

轻量级、模块化、可扩展的油藏数值模拟器，支持单相流和两相流（IMPES）渗流方程求解。

## 🚀 主要特性

- **模块化架构**: BaseModel 抽象接口 + ModelFactory 工厂模式，易于扩展新模型
- **配置驱动**: YAML / 字典配置，灵活管理不同模拟场景
- **单相流模拟**: 隐式压力求解，支持多种线性求解器（Direct/CG/BiCGSTAB/GMRES）
- **两相流 IMPES**: 隐式压力-显式饱和度方法，支持 CFL 自适应时间步
- **Peaceman 井模型**: 定压/定流量控制方式
- **2D/3D 可视化**: matplotlib + PyVista
- **地质统计**: 变差函数建模 + 克里金插值 + SGSIM 渗透率场生成
- **完整文档**: MkDocs + mkdocstrings 自动生成 API 文档

## 📦 安装

### 使用 UV（推荐）

```bash
git clone https://github.com/yourusername/reservoirpy.git
cd reservoirpy
uv sync --extra dev --extra viz --extra geostat
```

### 使用 pip

```bash
pip install -e ".[dev,viz]"
```

## 🏗️ 项目结构

```
reservoirpy/
├── src/reservoirpy/           # 源代码（src layout）
│   ├── core/                  # 核心模块
│   │   ├── discretization.py  # FVM 离散化（单相/两相）
│   │   ├── linear_solver.py   # 线性求解器
│   │   ├── nonlinear_solver.py# 非线性求解器
│   │   ├── simulator.py       # 主模拟器
│   │   ├── time_integration.py# 时间积分
│   │   ├── well_model.py      # Peaceman 井模型
│   │   └── output_manager.py  # 输出管理
│   ├── mesh/                  # 结构化网格
│   │   └── mesh.py
│   ├── models/                # 数学模型
│   │   ├── base_model.py      # 抽象基类
│   │   ├── model_factory.py   # 工厂模式
│   │   ├── single_phase.py    # 单相流模型
│   │   ├── two_phase_impes.py # 两相流 IMPES 模型
│   │   └── two_phase_fim.py   # 两相流 FIM 模型（框架）
│   ├── physics/               # 物理属性
│   │   └── physics.py         # Corey 相渗 + 两相物性
│   ├── geostatistics/         # 地质统计
│   │   ├── variogram.py       # 变差函数建模
│   │   ├── kriging.py         # 克里金插值
│   │   ├── sgsim.py           # 序贯高斯模拟
│   │   └── perm_generator.py  # 渗透率场生成器
│   ├── utils/                 # 工具函数
│   └── visualization/         # 可视化
├── config/                    # 配置文件
├── docs/                      # MkDocs 文档
├── examples/                  # 示例脚本
├── tests/                     # 测试套件（pytest）
└── pyproject.toml             # 项目配置（hatchling）
```

## 🧮 数学模型

### 单相流模型

求解单相渗流方程：`φ·c·∂p/∂t = ∇·(k/μ·∇p) + q`

- 隐式时间积分，无条件稳定
- 支持多种线性求解器 + 预条件
- **渗透率各向异性**：kx = ky ≠ kz（`kz_kx_ratio` 参数控制，默认 0.1）

### 两相流 IMPES 模型

隐式压力-显式饱和度方法：

1. **隐式求解压力方程**（总流度形式）：`φ·c_t·∂p/∂t = ∇·(λ_t·k·∇p) + q_t`
2. **显式更新饱和度**（分流方程 + 上游权重）：`φ·∂S_w/∂t + ∇·(f_w·v_t) = q_w`

关键特性：
- 界面总流度采用调和平均
- 分流函数使用上游权重（upstream weighting）
- 注入井注入纯水（f_w=1.0），生产井按分流函数产出
- 支持 CFL 自适应时间步长
- **渗透率各向异性**：3D 模拟中 z 方向使用 kz = kx × kz_kx_ratio

## 🚀 快速开始

### 单相流模拟

```python
from reservoirpy import ReservoirSimulator

config = {
    'mesh': {'nx': 10, 'ny': 10, 'nz': 1, 'dx': 10, 'dy': 10, 'dz': 10},
    'physics': {
        'type': 'single_phase',
        'permeability': 100.0,   # mD
        'porosity': 0.2,
        'viscosity': 0.001,      # Pa·s
        'compressibility': 1e-9  # 1/Pa
    },
    'wells': [
        {'location': [0, 5, 5], 'control_type': 'bhp',
         'value': 1e6, 'rw': 0.05, 'skin_factor': 0}
    ],
    'simulation': {
        'dt': 86400, 'total_time': 864000,
        'initial_pressure': 30e6
    }
}

sim = ReservoirSimulator(config_dict=config)
results = sim.run_simulation()
```

### 两相流水驱模拟

```python
from reservoirpy import ReservoirSimulator

config = {
    'mesh': {'nx': 5, 'ny': 5, 'nz': 1, 'dx': 10, 'dy': 10, 'dz': 10},
    'physics': {
        'type': 'two_phase_impes',
        'permeability': 100.0,
        'porosity': 0.2,
        'compressibility': 1e-9,
        'oil_viscosity': 5e-3,     # Pa·s
        'water_viscosity': 1e-3    # Pa·s
    },
    'wells': [
        {'location': [0, 0, 0], 'control_type': 'bhp',
         'value': 35e6, 'rw': 0.05, 'skin_factor': 0},  # 注入井
        {'location': [0, 4, 4], 'control_type': 'bhp',
         'value': 25e6, 'rw': 0.05, 'skin_factor': 0}   # 生产井
    ],
    'simulation': {
        'dt': 86400, 'total_time': 864000,
        'initial_pressure': 30e6,
        'initial_saturation': 0.2
    }
}

sim = ReservoirSimulator(config_dict=config)
results = sim.run_simulation()
```

### CFL 自适应时间步

IMPES 显式饱和度更新受 CFL 条件限制，可使用自适应时间步：

```python
from reservoirpy.models.two_phase_impes import TwoPhaseIMPES
from reservoirpy.mesh.mesh import StructuredMesh
from reservoirpy.physics.physics import TwoPhaseProperties
from reservoirpy.core.well_model import WellManager
import numpy as np

mesh = StructuredMesh(nx=10, ny=10, nz=1, dx=10, dy=10, dz=10)
physics = TwoPhaseProperties(mesh, {
    'permeability': 100.0, 'porosity': 0.2,
    'compressibility': 1e-9,
    'oil_viscosity': 5e-3, 'water_viscosity': 1e-3
})

well_manager = WellManager(mesh, wells_config)
well_manager.initialize_wells(k, physics.viscosity)

model = TwoPhaseIMPES(mesh, physics, {'cfl_factor': 0.8})
state = model.initialize_state({
    'initial_pressure': 30e6,
    'initial_saturation': 0.2
})

dt_max = 86400.0
total_time = 864000.0
current_time = 0.0

while current_time < total_time:
    cfl_dt = model.compute_cfl_timestep(
        state['pressure'], state['saturation'], well_manager)
    actual_dt = min(dt_max, cfl_dt, total_time - current_time)
    state = model.solve_timestep(actual_dt, state, well_manager)
    model.update_properties(state)
    current_time += actual_dt
```

### 地质统计 — 渗透率场生成

```python
from reservoirpy.geostatistics import PermeabilityGenerator

gen = PermeabilityGenerator(nx=20, ny=20, dx=10, dy=10)

# 非条件模拟
perm_field = gen.generate(
    major_range=50, minor_range=30,     # 变差函数变程 (m)
    sill=1.0, vtype='exponential',      # 变差函数模型
    n_realizations=1, seed=42,
    mean_log_perm=2.0,                  # log10(K) 均值 → ~100 mD
    std_log_perm=0.5,                   # log10(K) 标准差
)
# perm_field.shape = (1, 20, 20)，单位 mD

# 条件模拟（带井位硬数据）
import numpy as np
hard_data = np.array([
    [5.0, 5.0, 100.0],    # (x, y, perm_mD)
    [195.0, 195.0, 50.0],
])
perm_field = gen.generate(
    hard_data=hard_data,
    major_range=50, minor_range=30,
    sill=1.0, vtype='exponential',
    n_realizations=5, seed=42,
)

# 直接传入模拟器
config = {
    'physics': {
        'type': 'two_phase_impes',
        'permeability': perm_field,  # 直接使用生成的渗透率场
        ...
    }
}
```

## 🧪 测试

```bash
# 运行全部测试
uv run pytest tests/ -v

# 运行两相流测试
uv run pytest tests/test_two_phase.py -v
```

当前测试覆盖：网格、物理属性、井模型、线性求解器、单相流模型、两相流 IMPES、模拟器集成。

## 📚 文档

```bash
# 本地启动文档服务
uv run mkdocs serve

# 部署到 GitHub Pages
uv run mkdocs gh-deploy --force
```

文档包含：
- **理论基础**: 有限体积法、单相流方程、两相流方程、IMPES 方法、井模型
- **求解全流程**: 从 PDE 到离散到求解的完整推导
- **API 参考**: 自动从 docstrings 生成
- **示例**: 单相流模拟、两相流水驱模拟

## 🤝 贡献

欢迎贡献代码！请确保：

1. 新功能附带测试
2. 通过 `uv run pytest tests/ -v` 验证
3. 遵循现有代码风格

## 📄 许可证

MIT License
