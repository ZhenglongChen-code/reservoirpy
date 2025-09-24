"""
单相流求解器

实现单相流油藏模拟的完整求解流程
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from ..core.mesh import StructuredMesh
from ..core.physics import SinglePhaseProperties
from ..core.discretization import FVMDiscretizer
from ..core.well_model import WellManager
from ..core.linear_solver import LinearSolver
from ..core.time_integration import ImplicitEulerIntegrator


class SinglePhaseSolver:
    """
    单相流求解器
    
    协调所有模块，提供单相流油藏模拟的完整求解流程
    """
    
    def __init__(self, mesh: StructuredMesh, physics: SinglePhaseProperties, 
                 config: Dict[str, Any] = None):
        """
        初始化单相流求解器
        
        Args:
            mesh: 结构化网格
            physics: 单相流物理属性
            config: 求解器配置
        """
        self.mesh = mesh
        self.physics = physics
        self.config = config or {}
        
        # 初始化子模块
        self.discretizer = FVMDiscretizer(mesh, physics)
        self.linear_solver = LinearSolver(
            self.config.get('linear_solver', {}))
        self.time_integrator = ImplicitEulerIntegrator(
            mesh, physics, self.discretizer)
        
        # 模拟参数
        self.dt = self.config.get('dt', 86400.0)  # 默认1天
        self.total_time = self.config.get('total_time', 31536000.0)  # 默认1年
        self.output_interval = self.config.get('output_interval', 10)
        self.initial_pressure = self.config.get('initial_pressure', 30e6)  # 默认30MPa
    
    def solve_steady_state(self, wells_config: List[Dict[str, Any]]) -> np.ndarray:
        """
        求解稳态压力分布
        
        Args:
            wells_config: 井配置列表
            
        Returns:
            稳态压力场
        """
        # 初始化井管理器
        well_manager = WellManager(self.mesh, wells_config)
        well_manager.initialize_wells(
            self.physics.permeability, self.physics.viscosity)
        
        # 设置初始压力场
        pressure = np.full(self.mesh.n_cells, self.initial_pressure)
        
        # 更新单元压力
        for i, cell in enumerate(self.mesh.cell_list):
            cell.press = pressure[i]
        
        # 稳态求解（时间步长设为很大）
        dt_steady = 1e20
        A, b = self.discretizer.discretize_single_phase(dt_steady, pressure, well_manager)
        
        # 求解线性系统
        steady_pressure = self.linear_solver.solve(A, b)
        
        return steady_pressure
    
    def solve_transient(self, wells_config: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        求解瞬态压力分布
        
        Args:
            wells_config: 井配置列表
            
        Returns:
            模拟结果字典
        """
        # 初始化井管理器
        well_manager = WellManager(self.mesh, wells_config)
        well_manager.initialize_wells(
            self.physics.permeability, self.physics.viscosity)
        
        # 设置初始压力场
        initial_pressure = np.full(self.mesh.n_cells, self.initial_pressure)
        
        # 更新单元压力
        for i, cell in enumerate(self.mesh.cell_list):
            cell.press = initial_pressure[i]
        
        # 执行时间积分
        results = self.time_integrator.integrate(
            initial_pressure, self.dt, self.total_time, 
            well_manager, self.output_interval)
        
        return results
    
    def update_config(self, config: Dict[str, Any]):
        """
        更新求解器配置
        
        Args:
            config: 新的配置字典
        """
        self.config.update(config)
        self.dt = self.config.get('dt', self.dt)
        self.total_time = self.config.get('total_time', self.total_time)
        self.output_interval = self.config.get('output_interval', self.output_interval)
        self.initial_pressure = self.config.get('initial_pressure', self.initial_pressure)
        
        # 更新子模块配置
        if 'linear_solver' in config:
            self.linear_solver.update_config(config['linear_solver'])
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取求解器信息
        
        Returns:
            求解器信息字典
        """
        return {
            'mesh_size': self.mesh.grid_shape,
            'total_cells': self.mesh.n_cells,
            'dt': self.dt,
            'total_time': self.total_time,
            'output_interval': self.output_interval,
            'initial_pressure': self.initial_pressure,
            'linear_solver_info': self.linear_solver.get_info()
        }
    
    def __repr__(self):
        return f"SinglePhaseSolver({self.mesh.nx}x{self.mesh.ny}x{self.mesh.nz})"


def create_single_phase_solver(config: Dict[str, Any]) -> SinglePhaseSolver:
    """
    根据配置创建单相流求解器
    
    Args:
        config: 配置字典，包含网格和物理属性配置
        
    Returns:
        单相流求解器实例
    """
    from ..core.mesh import StructuredMesh
    from ..core.physics import SinglePhaseProperties
    
    # 创建网格
    mesh_config = config['mesh']
    mesh = StructuredMesh(
        nx=mesh_config['nx'],
        ny=mesh_config['ny'],
        nz=mesh_config['nz'],
        dx=mesh_config['dx'],
        dy=mesh_config['dy'],
        dz=mesh_config['dz']
    )
    
    # 创建物理属性
    physics_config = config['physics']
    physics = SinglePhaseProperties(mesh, physics_config)
    
    # 更新单元物理属性
    for i, cell in enumerate(mesh.cell_list):
        physics.update_cell_properties(cell, i)
    
    # 创建求解器
    solver_config = config.get('solver', {})
    solver = SinglePhaseSolver(mesh, physics, solver_config)
    
    return solver


def run_single_phase_simulation(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行单相流模拟
    
    Args:
        config: 配置字典
        
    Returns:
        模拟结果
    """
    # 创建求解器
    solver = create_single_phase_solver(config)
    
    # 获取井配置
    wells_config = config.get('wells', [])
    
    # 判断是稳态还是瞬态模拟
    if config.get('simulation_type', 'transient') == 'steady_state':
        # 稳态模拟
        steady_pressure = solver.solve_steady_state(wells_config)
        results = {
            'pressure_field': steady_pressure,
            'solver_info': solver.get_info()
        }
    else:
        # 瞬态模拟
        results = solver.solve_transient(wells_config)
        results['solver_info'] = solver.get_info()
    
    return results
