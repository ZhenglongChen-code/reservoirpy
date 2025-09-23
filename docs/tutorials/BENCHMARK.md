# 油藏数值模拟器基准测试

本文档介绍了油藏数值模拟器的基准测试算例，用于验证模拟器的正确性和性能。

## 📚 目录

1. [解析解对比](#解析解对比)
2. [Eclipse对比算例](#eclipse对比算例)
3. [性能基准测试](#性能基准测试)
4. [收敛性测试](#收敛性测试)

## 解析解对比

### 算例1: 无限大油藏中的点源解

对于无限大均质油藏中的点源注入问题，存在解析解：

```
P(r,t) = P0 + (qμ/4πKh) * Ei(-r²/4ηt)
```

其中：
- P0: 初始压力
- q: 注入流量
- μ: 流体粘度
- K: 渗透率
- h: 厚度
- Ei: 指数积分函数
- r: 到井的距离
- η: 压力扩散系数

```python
# benchmarks/analytical_comparison.py
import numpy as np
from scipy.special import expi
from reservoir_sim import ReservoirSimulator

def point_source_analytical_solution(r, t, q, mu, K, h, phi, ct, P0=30e6):
    """
    点源解析解
    """
    eta = K / (phi * mu * ct)  # 压力扩散系数
    term = -r**2 / (4 * eta * t)
    if term < -700:  # 避免数值下溢
        return P0
    else:
        return P0 + (q * mu / (4 * np.pi * K * h)) * expi(term)

def point_source_numerical_simulation():
    """
    点源数值模拟
    """
    config = {
        'mesh': {
            'nx': 50, 'ny': 50, 'nz': 1,
            'dx': 10.0, 'dy': 10.0, 'dz': 10.0
        },
        'physics': {
            'permeability': 100.0,  # mD
            'porosity': 0.2,
            'viscosity': 0.001,     # Pa·s
            'compressibility': 1e-9  # 1/Pa
        },
        'wells': [
            {'location': [0, 25, 25], 'control_type': 'rate', 'value': 0.001}
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

# 运行基准测试
if __name__ == "__main__":
    simulator, results = point_source_numerical_simulation()
    print("点源数值模拟完成")
```

### 算例2: 稳态径向流解

对于稳态径向流，存在解析解：

```
P(r) = Pw + (qμ/2πKh) * ln(r/rw)
```

```python
# benchmarks/steady_state_comparison.py
import numpy as np
from reservoir_sim import ReservoirSimulator

def steady_state_analytical_solution(r, rw, Pw, q, mu, K, h):
    """
    稳态径向流解析解
    """
    return Pw + (q * mu / (2 * np.pi * K * h)) * np.log(r / rw)

def steady_state_numerical_simulation():
    """
    稳态径向流数值模拟
    """
    config = {
        'mesh': {
            'nx': 40, 'ny': 40, 'nz': 1,
            'dx': 25.0, 'dy': 25.0, 'dz': 20.0
        },
        'physics': {
            'permeability': 150.0,  # mD
            'porosity': 0.25,
            'viscosity': 0.0008,    # Pa·s
            'compressibility': 1e-12 # 1/Pa (接近稳态)
        },
        'wells': [
            {'location': [0, 20, 20], 'control_type': 'bhp', 'value': 25e6}
        ],
        'simulation': {
            'dt': 86400,        # 1天
            'total_time': 86400 * 100,  # 100天（足够达到稳态）
            'output_interval': 20,
            'initial_pressure': 30e6
        }
    }
    
    simulator = ReservoirSimulator(config_dict=config)
    results = simulator.run_simulation()
    
    return simulator, results

# 运行基准测试
if __name__ == "__main__":
    simulator, results = steady_state_numerical_simulation()
    print("稳态径向流数值模拟完成")
```

## Eclipse对比算例

### 算例3: SPE1问题

SPE1是比较油藏模拟器的标准测试问题。

```python
# benchmarks/spe1_comparison.py
import numpy as np
from reservoir_sim import ReservoirSimulator

def spe1_numerical_simulation():
    """
    SPE1数值模拟
    """
    # SPE1网格参数
    nx, ny, nz = 10, 10, 10
    dx, dy, dz = 100.0, 100.0, 100.0  # ft转换为米: 1 ft = 0.3048 m
    
    config = {
        'mesh': {
            'nx': nx, 'ny': ny, 'nz': nz,
            'dx': dx * 0.3048, 'dy': dy * 0.3048, 'dz': dz * 0.3048
        },
        'physics': {
            'permeability': 100.0,  # mD
            'porosity': 0.2,
            'viscosity': 0.001,     # Pa·s
            'compressibility': 1e-9  # 1/Pa
        },
        'wells': [
            # 生产井
            {'location': [0, 4, 4], 'control_type': 'bhp', 'value': 1000*6894.76}  # 1000 psi
        ],
        'simulation': {
            'dt': 86400,        # 1天
            'total_time': 86400 * 365,  # 1年
            'output_interval': 30,
            'initial_pressure': 3000*6894.76  # 3000 psi
        }
    }
    
    simulator = ReservoirSimulator(config_dict=config)
    results = simulator.run_simulation()
    
    return simulator, results

# 运行基准测试
if __name__ == "__main__":
    simulator, results = spe1_numerical_simulation()
    print("SPE1数值模拟完成")
```

## 性能基准测试

### 算例4: 网格尺寸收敛性测试

```python
# benchmarks/performance_benchmark.py
import time
import numpy as np
from reservoir_sim import ReservoirSimulator

def performance_benchmark():
    """
    性能基准测试
    """
    grid_sizes = [10, 20, 30, 40]
    run_times = []
    
    for nx in grid_sizes:
        ny, nz = nx, 1
        print(f"测试网格尺寸: {nx}x{ny}x{nz}")
        
        config = {
            'mesh': {
                'nx': nx, 'ny': ny, 'nz': nz,
                'dx': 50.0, 'dy': 50.0, 'dz': 10.0
            },
            'physics': {
                'permeability': 100.0,
                'porosity': 0.2,
                'viscosity': 0.001,
                'compressibility': 1e-9
            },
            'wells': [
                {'location': [0, nx//2, nx//4], 'control_type': 'rate', 'value': 0.001},
                {'location': [0, nx//2, 3*nx//4], 'control_type': 'bhp', 'value': 25e6}
            ],
            'simulation': {
                'dt': 86400,
                'total_time': 86400 * 10,
                'output_interval': 2,
                'initial_pressure': 30e6
            }
        }
        
        # 记录运行时间
        start_time = time.time()
        simulator = ReservoirSimulator(config_dict=config)
        results = simulator.run_simulation()
        end_time = time.time()
        
        run_time = end_time - start_time
        run_times.append(run_time)
        
        print(f"  运行时间: {run_time:.2f} 秒")
        print(f"  时间步数: {len(results['time_history'])}")
    
    # 输出性能结果
    print("\n性能基准测试结果:")
    for i, nx in enumerate(grid_sizes):
        ny, nz = nx, 1
        cells = nx * ny * nz
        print(f"  {nx}x{ny}x{nz} ({cells} cells): {run_times[i]:.2f} 秒")
    
    return grid_sizes, run_times

# 运行性能基准测试
if __name__ == "__main__":
    grid_sizes, run_times = performance_benchmark()
```

## 收敛性测试

### 算例5: 时间步长收敛性测试

```python
# benchmarks/convergence_benchmark.py
import numpy as np
from reservoir_sim import ReservoirSimulator

def time_step_convergence_benchmark():
    """
    时间步长收敛性测试
    """
    time_steps = [86400, 43200, 21600, 10800]  # 1天, 12小时, 6小时, 3小时
    final_pressures = []
    
    for dt in time_steps:
        print(f"测试时间步长: {dt/3600:.0f} 小时")
        
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
                {'location': [0, 10, 5], 'control_type': 'rate', 'value': 0.001},
                {'location': [0, 10, 15], 'control_type': 'bhp', 'value': 25e6}
            ],
            'simulation': {
                'dt': dt,
                'total_time': 86400 * 20,  # 20天
                'output_interval': 1,
                'initial_pressure': 30e6
            }
        }
        
        simulator = ReservoirSimulator(config_dict=config)
        results = simulator.run_simulation()
        
        # 记录最终压力
        final_pressure = results['pressure_history'][-1].mean()
        final_pressures.append(final_pressure)
        
        print(f"  最终平均压力: {final_pressure/1e6:.2f} MPa")
        print(f"  时间步数: {len(results['time_history'])}")
    
    # 计算收敛性误差
    print("\n时间步长收敛性测试结果:")
    for i, dt in enumerate(time_steps):
        print(f"  时间步长 {dt/3600:.0f} 小时: {final_pressures[i]/1e6:.6f} MPa")
    
    # 计算相对于最小时步的误差
    if len(final_pressures) > 1:
        reference_pressure = final_pressures[-1]  # 最小时步的结果作为参考
        print(f"\n相对于最小时步({time_steps[-1]/3600:.0f}小时)的误差:")
        for i, dt in enumerate(time_steps):
            error = abs(final_pressures[i] - reference_pressure) / reference_pressure * 100
            print(f"  时间步长 {dt/3600:.0f} 小时: {error:.4f}%")
    
    return time_steps, final_pressures

# 运行收敛性测试
if __name__ == "__main__":
    time_steps, final_pressures = time_step_convergence_benchmark()
```

## 📌 总结

本基准测试文档提供了多种验证油藏数值模拟器正确性和性能的方法：

1. **解析解对比**: 验证数值方法的正确性
2. **标准算例对比**: 与行业标准进行对比
3. **性能基准测试**: 评估不同网格尺寸下的性能
4. **收敛性测试**: 验证时间步长和网格尺寸的收敛性

这些基准测试有助于确保模拟器的准确性和可靠性，并为后续的开发和优化提供参考依据。