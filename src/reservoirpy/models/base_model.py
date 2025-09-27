"""
油藏数值模拟模型抽象基类

定义所有数学模型必须实现的接口和通用流程
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.sparse import csr_matrix


class BaseModel(ABC):
    """
    油藏数值模拟模型抽象基类
    
    采用模板方法模式，定义求解的通用流程，
    具体的数学计算由子类实现。
    """
    
    def __init__(self, mesh, physics, config: Dict[str, Any]):
        """
        初始化模型
        
        Args:
            mesh: 网格对象
            physics: 物理属性对象
            config: 模型配置
        """
        self.mesh = mesh
        self.physics = physics
        self.config = config
        self.discretizer = None
        self.solver = None
        self.nonlinear_solver = None
        
        # 模拟参数
        self.dt = config.get('dt', 86400.0)
        self.tolerance = config.get('tolerance', 1e-6)
        self.max_iterations = config.get('max_iterations', 100)
        
    @abstractmethod
    def get_state_variables(self) -> List[str]:
        """
        返回模型的状态变量列表
        
        Returns:
            状态变量名称列表，如 ['pressure'] 或 ['pressure', 'saturation']
        """
        pass
        
    @abstractmethod
    def assemble_system(self, dt: float, state_vars: Dict[str, np.ndarray], 
                       well_manager) -> Tuple[csr_matrix, np.ndarray]:
        """
        组装线性系统 A*x = b
        
        Args:
            dt: 时间步长
            state_vars: 当前状态变量字典
            well_manager: 井管理器
            
        Returns:
            (系数矩阵A, 右端向量b)
        """
        pass
        
    @abstractmethod
    def solve_timestep(self, dt: float, state_vars: Dict[str, np.ndarray],
                      well_manager) -> Dict[str, np.ndarray]:
        """
        求解单个时间步
        
        Args:
            dt: 时间步长
            state_vars: 当前状态变量字典
            well_manager: 井管理器
            
        Returns:
            新的状态变量字典
        """
        pass
        
    @abstractmethod
    def update_properties(self, state_vars: Dict[str, np.ndarray]) -> None:
        """
        更新网格单元的物理属性
        
        Args:
            state_vars: 状态变量字典
        """
        pass
        
    @abstractmethod
    def validate_solution(self, state_vars: Dict[str, np.ndarray]) -> bool:
        """
        验证解的合理性
        
        Args:
            state_vars: 状态变量字典
            
        Returns:
            解是否合理
        """
        pass
        
    def initialize_state(self, config: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        初始化状态变量
        
        Args:
            config: 模拟配置
            
        Returns:
            初始状态变量字典
        """
        state_vars = {}
        
        for var_name in self.get_state_variables():
            if var_name == 'pressure':
                initial_pressure = config.get('initial_pressure', 30e6)
                state_vars['pressure'] = np.full(self.mesh.n_cells, initial_pressure)
            elif var_name == 'saturation':
                initial_saturation = config.get('initial_saturation', 0.2)
                state_vars['saturation'] = np.full(self.mesh.n_cells, initial_saturation)
            elif var_name == 'temperature':
                initial_temperature = config.get('initial_temperature', 353.15)  # 80°C
                state_vars['temperature'] = np.full(self.mesh.n_cells, initial_temperature)
                
        return state_vars
        
    def solve_simulation(self, initial_state: Dict[str, np.ndarray],
                        dt: float, total_time: float, 
                        well_manager, output_manager) -> Dict[str, Any]:
        """
        模拟求解的模板方法
        
        定义通用的时间循环和输出流程，具体的时间步求解由子类实现。
        
        Args:
            initial_state: 初始状态变量
            dt: 时间步长
            total_time: 总模拟时间
            well_manager: 井管理器
            output_manager: 输出管理器
            
        Returns:
            模拟结果字典
        """
        current_time = 0.0
        time_step = 0
        state_vars = initial_state.copy()
        
        # 初始化物理属性
        self.update_properties(state_vars)
        
        # 保存初始状态
        output_manager.save_timestep(0, current_time, state_vars)
        
        print(f"Starting simulation: dt={dt:.0f}s, total_time={total_time:.0f}s")
        
        while current_time < total_time:
            time_step += 1
            current_time += dt
            
            try:
                # 求解一个时间步（由子类实现）
                state_vars = self.solve_timestep(dt, state_vars, well_manager)
                
                # 验证解
                if not self.validate_solution(state_vars):
                    raise RuntimeError(f"Solution validation failed at timestep {time_step}")
                
                # 更新物理属性
                self.update_properties(state_vars)
                
                # 保存结果
                if time_step % output_manager.output_interval == 0:
                    output_manager.save_timestep(time_step, current_time, state_vars)
                    print(f"Timestep {time_step}: t={current_time:.1f}s")
                    
            except Exception as e:
                print(f"Error at timestep {time_step}: {e}")
                break
                
        print(f"Simulation completed: {time_step} timesteps")
        return output_manager.get_results()
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_type': self.__class__.__name__,
            'state_variables': self.get_state_variables(),
            'mesh_size': self.mesh.grid_shape,
            'total_cells': self.mesh.n_cells,
            'dt': self.dt,
            'tolerance': self.tolerance,
            'max_iterations': self.max_iterations
        }
        
    def solve_steady_state(self, well_manager, tolerance: float = 1e-8) -> Dict[str, np.ndarray]:
        """
        求解稳态解（默认实现）
        
        子类可以重写此方法以提供更高效的稳态求解
        
        Args:
            well_manager: 井管理器
            tolerance: 收敛容差
            
        Returns:
            稳态状态变量字典
            
        Raises:
            NotImplementedError: 如果子类不支持稳态求解
        """
        raise NotImplementedError(f"Steady state solving not implemented for {self.__class__.__name__}")
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.mesh.nx}x{self.mesh.ny}x{self.mesh.nz})"