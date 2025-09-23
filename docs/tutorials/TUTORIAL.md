# 油藏数值模拟器使用教程

本文档提供了油藏数值模拟器的详细使用教程，包括基础使用、高级功能和实际算例。

## 📚 目录

1. [基础使用](#基础使用)
2. [配置文件驱动模拟](#配置文件驱动模拟)
3. [程序化配置模拟](#程序化配置模拟)
4. [单相流模拟算例](#单相流模拟算例)
5. [两相流模拟算例](#两相流模拟算例)
6. [可视化功能](#可视化功能)
7. [高级功能](#高级功能)

## 基础使用

### 安装

```bash
# 克隆项目
git clone https://github.com/yourusername/reservoir-sim.git
cd reservoir-sim

# 安装基础包
pip install -e .

# 安装开发依赖（可选）
pip install -e .[dev]

# 安装可视化依赖（可选）
pip install -e .[viz]
```

### 运行简单示例

```bash
python examples/simple_example.py
```

### 基本代码使用

```python
from reservoir_sim import ReservoirSimulator

# 使用配置文件创建模拟器
simulator = ReservoirSimulator('config/default_config.yaml')

# 运行模拟
results = simulator.run_simulation()

# 获取结果
pressure_field = simulator.get_pressure_field()
```

## 配置文件驱动模拟

### 配置文件结构

```yaml
# config/default_config.yaml
mesh:
  nx: 20
  ny: 20
  nz: 1
  dx: 10.0
  dy: 10.0
  dz: 5.0

physics:
  permeability: 100
  porosity: 0.2
  viscosity: 0.001
  compressibility: 1e-9

wells:
  - location: [0, 5, 5]
    control_type: "rate"
    value: 0.001
  - location: [0, 15, 15]
    control_type: "bhp"
    value: 1e6

simulation:
  dt: 86400
  total_time: 31536000
  initial_pressure: 30e6
```

### 运行配置文件驱动模拟

```python
from reservoir_sim import ReservoirSimulator

# 使用配置文件创建模拟器
simulator = ReservoirSimulator('config/default_config.yaml')

# 运行模拟
results = simulator.run_simulation()
```

## 程序化配置模拟

### 创建程序化配置

```python
from reservoir_sim import ReservoirSimulator

# 使用字典配置
config = {
    'mesh': {
        'nx': 20, 'ny': 20, 'nz': 1,
        'dx': 10.0, 'dy': 10.0, 'dz': 5.0
    },
    'physics': {
        'permeability': 100.0,
        'porosity': 0.2,
        'viscosity': 0.001,
        'compressibility': 1e-9
    },
    'wells': [
        {'location': [0, 5, 5], 'control_type': 'rate', 'value': 0.001},
        {'location': [0, 15, 15], 'control_type': 'bhp', 'value': 1e6}
    ],
    'simulation': {
        'dt': 86400,
        'total_time': 31536000,
        'initial_pressure': 30e6
    }
}

simulator = ReservoirSimulator(config_dict=config)
results = simulator.run_simulation()
```

## 单相流模拟算例

### 算例1: 简单注采井模型

```python
# examples/single_phase_2d.py
import numpy as np
from reservoir_sim import ReservoirSimulator

def single_phase_2d_example():
    """2D单相流注采井模型"""
    config = {
        'mesh': {
            'nx': 20, 'ny': 20, 'nz': 1,
            'dx': 50.0, 'dy': 50.0, 'dz': 10.0
        },
        'physics': {
            'permeability': 100.0,  # mD
            'porosity': 0.2,
            'viscosity': 0.001,     # Pa·s
            'compressibility': 1e-9  # 1/Pa
        },
        'wells': [
            # 注入井
            {'location': [0, 10, 5], 'control_type': 'rate', 'value': 0.002},
            # 生产井
            {'location': [0, 10, 15], 'control_type': 'bhp', 'value': 20e6}
        ],
        'simulation': {
            'dt': 86400,        # 1天
            'total_time': 86400 * 30,  # 30天
            'output_interval': 5,
            'initial_pressure': 30e6
        }
    }
    
    simulator = ReservoirSimulator(config_dict=config)
    results = simulator.run_simulation()
    
    return simulator, results

if __name__ == "__main__":
    simulator, results = single_phase_2d_example()
    print(f"模拟完成，共{len(results['time_history'])}个时间步")
```

### 算例2: 3D单相流模型

```python
# examples/single_phase_3d.py
import numpy as np
from reservoir_sim import ReservoirSimulator

def single_phase_3d_example():
    """3D单相流模型"""
    config = {
        'mesh': {
            'nx': 10, 'ny': 10, 'nz': 5,
            'dx': 100.0, 'dy': 100.0, 'dz': 20.0
        },
        'physics': {
            'permeability': 150.0,  # mD
            'porosity': 0.25,
            'viscosity': 0.0008,    # Pa·s
            'compressibility': 1e-9  # 1/Pa
        },
        'wells': [
            # 底部注入井
            {'location': [0, 5, 5], 'control_type': 'rate', 'value': 0.003},
            # 顶部生产井
            {'location': [4, 5, 5], 'control_type': 'bhp', 'value': 25e6}
        ],
        'simulation': {
            'dt': 86400,        # 1天
            'total_time': 86400 * 60,  # 60天
            'output_interval': 10,
            'initial_pressure': 35e6
        }
    }
    
    simulator = ReservoirSimulator(config_dict=config)
    results = simulator.run_simulation()
    
    return simulator, results

if __name__ == "__main__":
    simulator, results = single_phase_3d_example()
    print(f"模拟完成，共{len(results['time_history'])}个时间步")
```

## 两相流模拟算例

### 算例3: 2D两相流模型

```python
# examples/two_phase_2d.py
import numpy as np
from reservoir_sim import ReservoirSimulator

def two_phase_2d_example():
    """2D两相流模型"""
    config = {
        'mesh': {
            'nx': 15, 'ny': 15, 'nz': 1,
            'dx': 75.0, 'dy': 75.0, 'dz': 15.0
        },
        'physics': {
            'permeability': 200.0,  # mD
            'porosity': 0.3,
            'viscosity': 0.001,     # Pa·s (水相)
            'compressibility': 1e-9, # 1/Pa
            'oil_viscosity': 0.002,  # Pa·s (油相)
            'water_viscosity': 0.001 # Pa·s (水相)
        },
        'two_phase': {
            'kro_model': 'corey',
            'krw_model': 'corey',
            'pc_model': 'brooks_corey',
            'kro_params': {'n_o': 2.0, 'S_or': 0.2},
            'krw_params': {'n_w': 2.0, 'S_wr': 0.2},
            'pc_params': {'P_c0': 1000.0, 'lambda': 2.0}
        },
        'wells': [
            # 注入井（注水）
            {'location': [0, 7, 3], 'control_type': 'rate', 'value': 0.0015},
            # 生产井
            {'location': [0, 7, 12], 'control_type': 'bhp', 'value': 22e6}
        ],
        'simulation': {
            'dt': 86400,        # 1天
            'total_time': 86400 * 45,  # 45天
            'output_interval': 5,
            'initial_pressure': 28e6,
            'initial_saturation': 0.2  # 初始水饱和度
        }
    }
    
    simulator = ReservoirSimulator(config_dict=config)
    results = simulator.run_simulation()
    
    return simulator, results

if __name__ == "__main__":
    simulator, results = two_phase_2d_example()
    print(f"模拟完成，共{len(results['time_history'])}个时间步")
```

## 可视化功能

### 2D可视化

```python
# examples/visualization_example.py
import numpy as np
import matplotlib.pyplot as plt
from reservoir_sim import ResulatorSimulator

def visualization_example():
    """可视化示例"""
    config = {
        'mesh': {
            'nx': 20, 'ny': 20, 'nz': 1,
            'dx': 50.0, 'dy': 50.0, 'dz': 10.0
        },
        'physics': {
            'permeability': 100.0,
            'porosity': 0.2,
            'viscosity': 0.001,
            'compressibility': 1e-9
        },
        'wells': [
            {'location': [0, 10, 5], 'control_type': 'rate', 'value': 0.002},
            {'location': [0, 10, 15], 'control_type': 'bhp', 'value': 20e6}
        ],
        'simulation': {
            'dt': 86400,
            'total_time': 86400 * 20,
            'output_interval': 2,
            'initial_pressure': 30e6
        }
    }
    
    simulator = ReservoirSimulator(config_dict=config)
    results = simulator.run_simulation()
    
    # 获取最终压力场
    final_pressure = results['pressure_history'][-1]
    pressure_2d = final_pressure.reshape(20, 20)
    
    # 绘制压力场
    plt.figure(figsize=(10, 8))
    plt.imshow(pressure_2d, cmap='viridis', origin='lower')
    plt.colorbar(label='Pressure (Pa)')
    plt.title('Final Pressure Field')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

if __name__ == "__main__":
    visualization_example()
```

## 高级功能

### 自定义网格和物理属性

```python
from reservoir_sim import StructuredMesh, SinglePhaseProperties

# 创建自定义网格
mesh = StructuredMesh(nx=25, ny=25, nz=1, dx=40.0, dy=40.0, dz=8.0)

# 创建自定义物理属性
config = {
    'permeability': 120.0,  # mD
    'porosity': 0.22,
    'viscosity': 0.0009,    # Pa·s
    'compressibility': 1e-9  # 1/Pa
}
physics = SinglePhaseProperties(mesh, config)
```

### 获取单元属性

```python
# 获取特定单元的属性
cell_props = simulator.get_cell_properties(0, 10, 10)
print(f"单元压力: {cell_props['pressure']}")
print(f"单元孔隙度: {cell_props['porosity']}")
```

### 井生产数据分析

```python
# 获取井生产数据
well_data = simulator.get_well_production(0)  # 第一口井
times = well_data['time']
pressures = well_data['pressure']

# 绘制井底压力变化曲线
plt.plot(times/86400, pressures/1e6)
plt.xlabel('Time (days)')
plt.ylabel('Pressure (MPa)')
plt.title('Well Bottom Hole Pressure')
plt.grid(True)
plt.show()
```

## 📌 总结

本教程涵盖了油藏数值模拟器的主要使用方法，从基础使用到高级功能。通过这些示例，您可以快速上手并开始进行油藏模拟研究。

如需更多帮助，请参考：
- [API文档](../api/API文档.md)
- [理论背景](../theory/)
- [测试指南](../../src/reservoirpy/tests/TESTING.md)