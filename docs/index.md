# ReservoirPy

轻量级、模块化、可扩展的油藏数值模拟器。

## 特性

- 🧮 **有限体积法离散化** — 支持结构化网格上的 FVM 离散
- 🔄 **多物理模型** — 单相流、两相流 IMPES/FIM
- 🏗️ **模块化架构** — BaseModel 抽象接口 + ModelFactory 工厂模式
- 🌐 **多种线性求解器** — Direct/CG/BiCGSTAB/GMRES + 预条件
- 🛢️ **Peaceman 井模型** — 支持定压/定流量控制
- 📊 **2D/3D 可视化** — matplotlib + PyVista

## 快速示例

```python
from reservoirpy import ReservoirSimulator

config = {
    'mesh': {'nx': 10, 'ny': 10, 'nz': 1, 'dx': 10, 'dy': 10, 'dz': 10},
    'physics': {
        'type': 'single_phase',
        'permeability': 100.0,  # mD
        'porosity': 0.2,
        'viscosity': 0.001,     # Pa·s
        'compressibility': 1e-9 # 1/Pa
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

## 安装

```bash
# 使用 UV（推荐）
uv sync --extra dev --extra viz

# 或使用 pip
pip install -e ".[dev,viz]"
```
