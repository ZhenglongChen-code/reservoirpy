# 添加新数学模型指南

本文档说明如何在新架构下添加自定义的数学模型。

## 1. 基本步骤

### 步骤1：继承BaseModel

```python
from reservoirpy.models.base_model import BaseModel
from typing import Dict, List, Tuple
import numpy as np
from scipy.sparse import csr_matrix

class ThermicalModel(BaseModel):
    """
    热采模型
    
    求解耦合的压力-温度-饱和度方程组
    """
    
    def __init__(self, mesh, physics, config):
        super().__init__(mesh, physics, config)
        
        # 热采特有参数
        self.thermal_conductivity = config.get('thermal_conductivity', 2.0)  # W/(m·K)
        self.heat_capacity = config.get('heat_capacity', 2000.0)  # J/(kg·K)
        
        # 初始化求解器
        from reservoirpy.core.discretization import FVMDiscretizer
        from reservoirpy.core.nonlinear_solver import NewtonRaphsonSolver
        
        self.discretizer = FVMDiscretizer(mesh, physics)
        self.nonlinear_solver = NewtonRaphsonSolver(config.get('nonlinear_solver', {}))
```

### 步骤2：实现抽象方法

```python
    def get_state_variables(self) -> List[str]:
        """热采模型的状态变量"""
        return ['pressure', 'temperature', 'saturation']
        
    def assemble_system(self, dt: float, state_vars: Dict[str, np.ndarray], 
                       well_manager) -> Tuple[csr_matrix, np.ndarray]:
        """组装耦合的线性系统"""
        pressure = state_vars['pressure']
        temperature = state_vars['temperature'] 
        saturation = state_vars['saturation']
        
        # 这里实现具体的数学组装逻辑
        # 返回雅可比矩阵和残差向量
        pass
        
    def solve_timestep(self, dt: float, state_vars: Dict[str, np.ndarray],
                      well_manager) -> Dict[str, np.ndarray]:
        """求解一个时间步"""
        # 使用Newton-Raphson方法求解非线性系统
        def residual_function(x):
            # 将解向量分解为各个状态变量
            n_cells = self.mesh.n_cells
            new_pressure = x[:n_cells]
            new_temperature = x[n_cells:2*n_cells]
            new_saturation = x[2*n_cells:3*n_cells]
            
            new_state = {
                'pressure': new_pressure,
                'temperature': new_temperature,
                'saturation': new_saturation
            }
            
            # 计算残差
            A, b = self.assemble_system(dt, new_state, well_manager)
            residual = A @ x - b
            return residual
            
        # 初始猜测
        n_cells = self.mesh.n_cells
        x0 = np.concatenate([
            state_vars['pressure'],
            state_vars['temperature'],
            state_vars['saturation']
        ])
        
        # 求解
        solution = self.nonlinear_solver.solve(residual_function, x0)
        
        # 分解解向量
        return {
            'pressure': solution[:n_cells],
            'temperature': solution[n_cells:2*n_cells],
            'saturation': solution[2*n_cells:3*n_cells]
        }
        
    def update_properties(self, state_vars: Dict[str, np.ndarray]) -> None:
        """更新温度相关的物理属性"""
        pressure = state_vars['pressure']
        temperature = state_vars['temperature']
        saturation = state_vars['saturation']
        
        for i, cell in enumerate(self.mesh.cell_list):
            cell.press = pressure[i]
            cell.temperature = temperature[i]
            cell.Sw = saturation[i]
            
            # 更新温度依赖的粘度
            self._update_viscosity(cell, temperature[i])
            
    def validate_solution(self, state_vars: Dict[str, np.ndarray]) -> bool:
        """验证热采解的合理性"""
        pressure = state_vars['pressure'] 
        temperature = state_vars['temperature']
        saturation = state_vars['saturation']
        
        # 基本检查
        if (np.any(np.isnan(pressure)) or np.any(np.isnan(temperature)) or 
            np.any(np.isnan(saturation))):
            return False
            
        # 物理范围检查
        if (np.any(pressure <= 0) or 
            np.any(temperature < 273.15) or np.any(temperature > 1000) or
            np.any(saturation < 0) or np.any(saturation > 1)):
            return False
            
        return True
        
    def _update_viscosity(self, cell, temperature):
        """根据温度更新粘度"""
        # 使用Arrhenius关系或其他经验公式
        # μ(T) = μ0 * exp(E/(RT))
        pass
```

### 步骤3：注册新模型

```python
# 在模型定义文件末尾
from reservoirpy.models.model_factory import ModelFactory
ModelFactory.register('thermal', ThermicalModel) 
```

### 步骤4：更新配置

```yaml
physics:
  type: 'thermal'  # 指定新的模型类型
  permeability: 100.0
  porosity: 0.2
  viscosity: 0.001
  thermal_conductivity: 2.0
  heat_capacity: 2000.0

simulation:
  dt: 3600
  total_time: 86400
  initial_pressure: 30e6
  initial_temperature: 353.15  # 80°C
  initial_saturation: 0.2

model:
  thermal_conductivity: 2.0
  heat_capacity: 2000.0
  nonlinear_solver:
    method: 'newton_raphson'
    tolerance: 1e-6
    max_iterations: 50
```

## 2. 完整示例：组分模型

```python
class CompositionalModel(BaseModel):
    """
    组分模型
    
    求解多组分流体的相平衡和流动
    """
    
    def __init__(self, mesh, physics, config):
        super().__init__(mesh, physics, config)
        
        # 组分模型参数
        self.n_components = config.get('n_components', 3)
        self.component_names = config.get('component_names', ['C1', 'C7+', 'CO2'])
        
        # EOS参数
        self.eos_parameters = config.get('eos_parameters', {})
        
    def get_state_variables(self) -> List[str]:
        """组分模型状态变量：压力 + 各组分摩尔分数"""
        variables = ['pressure']
        for i in range(self.n_components - 1):  # 最后一个组分由约束确定
            variables.append(f'mole_fraction_{i}')
        return variables
        
    def assemble_system(self, dt: float, state_vars: Dict[str, np.ndarray], 
                       well_manager) -> Tuple[csr_matrix, np.ndarray]:
        """组装组分守恒方程组"""
        
        # 压力方程：总的质量守恒
        # ∂/∂t(φρ) + ∇·(ρv) = q
        
        # 组分方程：各组分的摩尔守恒  
        # ∂/∂t(φ∑ρⱼxᵢⱼ) + ∇·(∑ρⱼxᵢⱼvⱼ) = qᵢ
        
        # 这里需要实现复杂的相平衡计算和雅可比矩阵组装
        pass
        
    def solve_timestep(self, dt: float, state_vars: Dict[str, np.ndarray],
                      well_manager) -> Dict[str, np.ndarray]:
        """使用Newton-Raphson求解强非线性组分系统"""
        
        # 组分模型通常需要多次Newton迭代
        # 包含相平衡计算（闪蒸计算）
        pass

# 注册组分模型
ModelFactory.register('compositional', CompositionalModel)
```

## 3. 高级特性

### 自适应时间步长

```python
class AdaptiveTimeStepModel(BaseModel):
    """支持自适应时间步长的模型"""
    
    def solve_simulation(self, initial_state, dt, total_time, 
                        well_manager, output_manager):
        """重写solve_simulation以支持自适应时间步"""
        current_time = 0.0
        time_step = 0
        state_vars = initial_state.copy()
        
        # 自适应参数
        dt_min = self.config.get('dt_min', dt / 100)
        dt_max = self.config.get('dt_max', dt * 10) 
        target_iterations = self.config.get('target_iterations', 5)
        
        while current_time < total_time:
            time_step += 1
            
            # 尝试求解
            try:
                new_state = self.solve_timestep(dt, state_vars, well_manager)
                
                # 根据收敛性调整时间步长
                if self.last_iterations < target_iterations:
                    dt = min(dt * 1.2, dt_max)  # 增大时间步
                elif self.last_iterations > target_iterations * 2:
                    dt = max(dt * 0.8, dt_min)  # 减小时间步
                    
                state_vars = new_state
                current_time += dt
                
            except ConvergenceError:
                # 收敛失败，减小时间步重试
                dt = max(dt * 0.5, dt_min)
                if dt == dt_min:
                    raise RuntimeError("Cannot converge even with minimum time step")
                continue
                
            # 保存结果
            if time_step % output_manager.output_interval == 0:
                output_manager.save_timestep(time_step, current_time, state_vars)
                
        return output_manager.get_results()
```

### 并行求解支持

```python
class ParallelModel(BaseModel):
    """支持并行计算的模型"""
    
    def __init__(self, mesh, physics, config):
        super().__init__(mesh, physics, config)
        
        # 并行参数
        self.use_parallel = config.get('use_parallel', False)
        self.n_processes = config.get('n_processes', 4)
        
        if self.use_parallel:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            
    def assemble_system(self, dt, state_vars, well_manager):
        """并行组装系数矩阵"""
        if self.use_parallel:
            # 将网格分解到不同进程
            # 使用MPI进行通信
            return self._parallel_assemble(dt, state_vars, well_manager)
        else:
            return super().assemble_system(dt, state_vars, well_manager)
```

## 4. 测试新模型

```python
def test_thermal_model():
    """测试热采模型"""
    config = {
        'mesh': {'nx': 10, 'ny': 10, 'nz': 1, 'dx': 10, 'dy': 10, 'dz': 5},
        'physics': {
            'type': 'thermal',
            'permeability': 100.0,
            'porosity': 0.2,
            'thermal_conductivity': 2.0
        },
        'simulation': {
            'dt': 3600,
            'total_time': 86400,
            'initial_pressure': 30e6,
            'initial_temperature': 353.15,
            'initial_saturation': 0.2
        }
    }
    
    simulator = ReservoirSimulator(config_dict=config)
    results = simulator.run_simulation()
    
    # 验证结果
    assert 'temperature' in results['field_data']
    assert np.all(results['field_data']['temperature'][-1] > 273.15)
    
    print("热采模型测试通过")

if __name__ == "__main__":
    test_thermal_model()
```

## 5. 最佳实践

### 5.1 模型设计原则

1. **单一职责**：每个模型专注于特定的物理过程
2. **接口一致**：严格遵循BaseModel接口
3. **配置驱动**：所有参数通过配置文件管理
4. **错误处理**：提供清晰的错误信息和恢复机制

### 5.2 性能优化

1. **稀疏矩阵**：使用scipy.sparse进行大型线性系统
2. **内存管理**：避免不必要的数组复制
3. **编译加速**：关键循环使用numba或cython
4. **并行计算**：支持MPI或多线程

### 5.3 验证和测试

1. **单元测试**：每个方法都要有对应测试
2. **基准测试**：与解析解或商业软件对比
3. **收敛性测试**：验证网格和时间步长收敛性
4. **物理合理性**：检查守恒定律和物理约束

通过遵循这个指南，您可以轻松地为ReservoirPy添加新的数学模型，同时保持代码的一致性和可维护性。