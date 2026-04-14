# 快速开始

## 安装

### 使用 UV（推荐）

```bash
git clone https://github.com/yourusername/reservoirpy.git
cd reservoirpy
uv sync --extra dev --extra viz
```

### 使用 pip

```bash
pip install -e ".[dev,viz]"
```

## 核心概念

ReservoirPy 的架构围绕以下核心组件：

| 组件 | 职责 | 对应模块 |
|------|------|---------|
| **Mesh** | 网格生成与管理 | `reservoirpy.mesh` |
| **Physics** | 物理属性建模 | `reservoirpy.physics` |
| **Model** | 数学模型与求解 | `reservoirpy.models` |
| **Discretizer** | FVM 离散化 | `reservoirpy.core.discretization` |
| **LinearSolver** | 线性系统求解 | `reservoirpy.core.linear_solver` |
| **WellManager** | 井模型管理 | `reservoirpy.core.well_model` |
| **Simulator** | 顶层模拟器 | `reservoirpy.core.simulator` |

## 运行第一个模拟

### 方式一：配置驱动（推荐）

```python
from reservoirpy import ReservoirSimulator

config = {
    'mesh': {'nx': 10, 'ny': 10, 'nz': 1, 'dx': 10, 'dy': 10, 'dz': 10},
    'physics': {
        'type': 'single_phase',
        'permeability': 100.0,
        'porosity': 0.2,
        'viscosity': 0.001,
        'compressibility': 1e-9
    },
    'wells': [
        {'location': [0, 5, 5], 'control_type': 'bhp',
         'value': 1e6, 'rw': 0.05, 'skin_factor': 0}
    ],
    'simulation': {
        'dt': 86400,
        'total_time': 864000,
        'initial_pressure': 30e6
    }
}

sim = ReservoirSimulator(config_dict=config)
results = sim.run_simulation()
```

### 方式二：组件式使用

```python
from reservoirpy import StructuredMesh, SinglePhaseProperties, WellManager
from reservoirpy.core.discretization import FVMDiscretizer
from reservoirpy.core.linear_solver import LinearSolver
import numpy as np

mesh = StructuredMesh(nx=10, ny=10, nz=1, dx=10, dy=10, dz=10)
physics = SinglePhaseProperties(mesh, {
    'permeability': 100.0, 'porosity': 0.2,
    'viscosity': 0.001, 'compressibility': 1e-9
})

well_manager = WellManager(mesh, [
    {'location': [0, 5, 5], 'control_type': 'bhp', 'value': 1e6}
])
well_manager.initialize_wells(
    np.full((1, 10, 10), 1e-13), physics.viscosity
)

discretizer = FVMDiscretizer(mesh, physics)
solver = LinearSolver()

pressure = np.full(mesh.n_cells, 30e6)
A, b = discretizer.discretize_single_phase(86400, pressure, well_manager)
pressure = solver.solve(A, b)
```

## 运行测试

```bash
uv run pytest tests/ -v
```
