# 两相流水驱模拟示例

本示例展示如何使用 ReservoirPy 进行油水两相流水驱模拟。

## 问题描述

5×5 二维油藏，一角注入、对角生产：

- 注入井（左上角）：BHP = 35 MPa，注水驱油
- 生产井（右下角）：BHP = 25 MPa
- 初始水饱和度 Sw = 0.2（束缚水）
- 油粘度 5 mPa·s，水粘度 1 mPa·s

## 完整代码

```python
from reservoirpy import ReservoirSimulator

config = {
    'mesh': {'nx': 5, 'ny': 5, 'nz': 1, 'dx': 10, 'dy': 10, 'dz': 10},
    'physics': {
        'type': 'two_phase_impes',
        'permeability': 100.0,       # mD
        'porosity': 0.2,
        'compressibility': 1e-9,     # 1/Pa
        'oil_viscosity': 5e-3,       # Pa·s
        'water_viscosity': 1e-3      # Pa·s
    },
    'wells': [
        # 注入井（左上角）- BHP > 地层压力 → 注入
        {'location': [0, 0, 0], 'control_type': 'bhp',
         'value': 35e6, 'rw': 0.05, 'skin_factor': 0},
        # 生产井（右下角）- BHP < 地层压力 → 生产
        {'location': [0, 4, 4], 'control_type': 'bhp',
         'value': 25e6, 'rw': 0.05, 'skin_factor': 0}
    ],
    'simulation': {
        'dt': 86400,                  # 1 天
        'total_time': 864000,         # 10 天
        'initial_pressure': 30e6,     # 30 MPa
        'initial_saturation': 0.2,    # 束缚水饱和度
        'output_interval': 5
    },
    'output': {'output_interval': 5}
}

sim = ReservoirSimulator(config_dict=config)
results = sim.run_simulation()
```

## 使用 CFL 自适应时间步

IMPES 方法的显式饱和度更新受 CFL 条件限制，可以使用自适应时间步：

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

wells_config = [
    {'location': [0, 1, 1], 'control_type': 'bhp', 'value': 35e6,
     'rw': 0.05, 'skin_factor': 0},
    {'location': [0, 8, 8], 'control_type': 'bhp', 'value': 25e6,
     'rw': 0.05, 'skin_factor': 0}
]

well_manager = WellManager(mesh, wells_config)
k = np.full((1, 10, 10), 9.869e-14)
well_manager.initialize_wells(k, physics.viscosity)

model = TwoPhaseIMPES(mesh, physics, {'cfl_factor': 0.8})
state = model.initialize_state({
    'initial_pressure': 30e6,
    'initial_saturation': 0.2
})

dt_max = 86400.0  # 最大时间步 1 天
total_time = 864000.0
current_time = 0.0

while current_time < total_time:
    # 计算 CFL 限制的时间步
    cfl_dt = model.compute_cfl_timestep(
        state['pressure'], state['saturation'], well_manager)
    actual_dt = min(dt_max, cfl_dt, total_time - current_time)

    state = model.solve_timestep(actual_dt, state, well_manager)
    model.update_properties(state)
    current_time += actual_dt
```

## 关键物理机制

### 注入井处理

注入井注入纯水（f_w = 1.0），直接增加地层含水饱和度：

$$\Delta S_w = \frac{|Q_{inj}| \cdot \Delta t}{V \cdot \phi}$$

### 生产井处理

生产井产出按分流函数分配油水比例：

$$Q_w = f_w \cdot Q_{total}, \quad f_w = \frac{\lambda_w}{\lambda_w + \lambda_o}$$

### 上游权重

界面分流函数取上游单元的值，确保物理一致性：

$$f_w^{up} = \begin{cases} f_w(S_{w,i}) & \text{if } p_i > p_j \\ f_w(S_{w,j}) & \text{if } p_j > p_i \end{cases}$$
