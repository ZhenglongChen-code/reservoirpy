# 单相流模拟示例

本示例展示如何使用 ReservoirPy 进行单相流压力场模拟。

## 问题描述

二维油藏（10×10×1 网格），中心有一口生产井：

- 网格：10×10×1，单元尺寸 10m×10m×10m
- 渗透率：100 mD，孔隙度：0.2
- 初始压力：30 MPa
- 生产井：中心位置，定井底流压 1 MPa
- 模拟时间：10 天，时间步长 1 天

## 方式一：配置驱动（推荐）

```python
from reservoirpy import ReservoirSimulator

config = {
    'mesh': {'nx': 10, 'ny': 10, 'nz': 1, 'dx': 10, 'dy': 10, 'dz': 10},
    'physics': {
        'type': 'single_phase',
        'permeability': 100.0,   # mD
        'porosity': 0.2,
        'viscosity': 0.001,      # Pa·s
        'compressibility': 1e-9   # 1/Pa
    },
    'wells': [
        {'location': [0, 5, 5], 'control_type': 'bhp',
         'value': 1e6, 'rw': 0.05, 'skin_factor': 0}
    ],
    'simulation': {
        'dt': 86400,              # 1 天
        'total_time': 864000,     # 10 天
        'initial_pressure': 30e6  # 30 MPa
    }
}

sim = ReservoirSimulator(config_dict=config)
results = sim.run_simulation()
```

## 方式二：组件式使用

适合需要精细控制每个步骤的场景：

```python
import numpy as np
from reservoirpy import StructuredMesh, SinglePhaseProperties, WellManager
from reservoirpy.core.discretization import FVMDiscretizer
from reservoirpy.core.linear_solver import LinearSolver

# 1. 创建网格
mesh = StructuredMesh(nx=10, ny=10, nz=1, dx=10, dy=10, dz=10)

# 2. 设置物理属性
physics = SinglePhaseProperties(mesh, {
    'permeability': 100.0, 'porosity': 0.2,
    'viscosity': 0.001, 'compressibility': 1e-9
})

# 3. 配置井
well_manager = WellManager(mesh, [
    {'location': [0, 5, 5], 'control_type': 'bhp',
     'value': 1e6, 'rw': 0.05, 'skin_factor': 0}
])

# 初始化井（计算产能指数）
k = np.full((1, 10, 10), 9.869e-14)  # 100 mD 转 m²
well_manager.initialize_wells(k, physics.viscosity)

# 4. 离散化 + 求解
discretizer = FVMDiscretizer(mesh, physics)
solver = LinearSolver()

pressure = np.full(mesh.n_cells, 30e6)  # 初始压力 30 MPa
dt = 86400.0                             # 1 天

# 单步求解
A, b = discretizer.discretize_single_phase(dt, pressure, well_manager)
pressure_new = solver.solve(A, b)

print(f"压力范围: {np.min(pressure_new)/1e6:.2f} - {np.max(pressure_new)/1e6:.2f} MPa")
```

## 方式三：从 YAML 配置文件

创建配置文件 `config.yaml`：

```yaml
mesh:
  nx: 10
  ny: 10
  nz: 1
  dx: 10
  dy: 10
  dz: 10

physics:
  type: single_phase
  permeability: 100.0
  porosity: 0.2
  viscosity: 0.001
  compressibility: 1.0e-9

wells:
  - location: [0, 5, 5]
    control_type: bhp
    value: 1000000
    rw: 0.05
    skin_factor: 0

simulation:
  dt: 86400
  total_time: 864000
  initial_pressure: 30000000
```

然后加载运行：

```python
from reservoirpy import ReservoirSimulator

sim = ReservoirSimulator(config_file='config.yaml')
results = sim.run_simulation()
```

## 结果分析

```python
import numpy as np

# 时间历史
print(f"模拟步数: {len(results['time_history'])}")

# 最终压力场
final_pressure = results['field_data']['pressure'][-1]
print(f"最终压力: {np.min(final_pressure)/1e6:.2f} - {np.max(final_pressure)/1e6:.2f} MPa")

# 井数据
if results['well_data']:
    well_data = results['well_data'][-1]
    print(f"井数据: {well_data}")
```

## 可视化

```python
from reservoirpy.visualization.plot_2d import create_2d_plotter

plotter = create_2d_plotter(mesh)
fig = plotter.plot_pressure_field(final_pressure, "Pressure Field (MPa)")
fig.savefig("pressure_field.png", dpi=150)
```
