"""
两相流IMPES求解器

实现隐式压力-显式饱和度方法求解两相流问题
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from reservoirpy.mesh.mesh import StructuredMesh
from reservoirpy.physics.physics import TwoPhaseProperties
from ..core.discretization import FVMDiscretizer
from ..core.well_model import WellManager
from ..core.linear_solver import LinearSolver


class TwoPhaseIMPES:
    """
    两相流IMPES求解器
    
    实现隐式压力-显式饱和度方法求解两相流问题
    """
    
    def __init__(self, mesh: StructuredMesh, physics: TwoPhaseProperties, 
                 config: Dict[str, Any] = None):
        """
        初始化IMPES求解器
        
        Args:
            mesh: 结构化网格
            physics: 两相流物理属性
            config: 求解器配置
        """
        self.mesh = mesh
        self.physics = physics
        self.config = config or {}
        self.total_cells = mesh.total_cells
        
        # 初始化子模块
        self.discretizer = FVMDiscretizer(mesh, physics)
        self.linear_solver = LinearSolver(
            self.config.get('linear_solver', {}))
        
        # 模拟参数
        self.dt = self.config.get('dt', 86400.0)  # 默认1天
        self.total_time = self.config.get('total_time', 31536000.0)  # 默认1年
        self.output_interval = self.config.get('output_interval', 10)
        self.initial_pressure = self.config.get('initial_pressure', 30e6)  # 默认30MPa
        self.initial_saturation = self.config.get('initial_saturation', 0.2)  # 默认水饱和度0.2
    
    def solve_time_step(self, pressure: np.ndarray, saturation: np.ndarray,
                       dt: float, well_manager: WellManager) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解一个时间步
        
        Args:
            pressure: 当前压力场
            saturation: 当前饱和度场
            dt: 时间步长
            well_manager: 井管理器
            
        Returns:
            (新压力场, 新饱和度场)
        """
        # 第一步：隐式求解压力方程
        A, b = self.discretizer.discretize_single_phase(dt, pressure, well_manager)
        new_pressure = self.linear_solver.solve(A, b)
        
        # 第二步：显式更新饱和度
        new_saturation = self._update_saturation_explicit(
            pressure, new_pressure, saturation, dt, well_manager)
        
        return new_pressure, new_saturation
    
    def _update_saturation_explicit(self, pressure_old: np.ndarray,
                                  pressure_new: np.ndarray,
                                  saturation_old: np.ndarray,
                                  dt: float, well_manager: WellManager) -> np.ndarray:
        """
        显式更新饱和度
        
        Args:
            pressure_old: 旧压力场
            pressure_new: 新压力场
            saturation_old: 旧饱和度场
            dt: 时间步长
            well_manager: 井管理器
            
        Returns:
            新饱和度场
        """
        saturation_new = saturation_old.copy()
        
        # 对每个单元更新饱和度
        for i in range(self.total_cells):
            z, y, x = self.mesh.get_cell_coords(i)
            cell = self.mesh.cell_list[i]
            
            # 计算饱和度变化率
            dS_dt = self._compute_saturation_rate(
                cell, pressure_old[i], pressure_new[i], 
                saturation_old[i], dt)
            
            # 更新饱和度
            saturation_new[i] = saturation_old[i] + dS_dt * dt
            
            # 限制饱和度在物理范围内
            saturation_new[i] = np.clip(saturation_new[i], 0.0, 1.0)
        
        return saturation_new
    
    def _compute_saturation_rate(self, cell, pressure_old: float, 
                               pressure_new: float, saturation: float, 
                               dt: float) -> float:
        """
        计算饱和度变化率
        
        Args:
            cell: 单元对象
            pressure_old: 旧压力
            pressure_new: 新压力
            saturation: 当前饱和度
            dt: 时间步长
            
        Returns:
            饱和度变化率 dS/dt
        """
        # 这里应该实现完整的饱和度变化率计算
        # 简化处理：基于压力变化和相对渗透率变化
        dp = pressure_new - pressure_old
        
        # 计算相对渗透率
        kro = self.physics.get_relative_permeability(saturation, 'oil')
        krw = self.physics.get_relative_permeability(saturation, 'water')
        
        # 计算流度比
        lambda_o = kro / self.physics.mu_o
        lambda_w = krw / self.physics.mu_w
        lambda_t = lambda_o + lambda_w
        
        # 简化的饱和度变化率（实际应基于流量计算）
        if lambda_t > 0:
            dS_dt = 0.01 * dp / (lambda_t * cell.porosity)
        else:
            dS_dt = 0.0
        
        return dS_dt
    
    def solve_simulation(self, wells_config: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        运行完整的模拟
        
        Args:
            wells_config: 井配置列表
            
        Returns:
            模拟结果字典
        """
        # 初始化井管理器
        well_manager = WellManager(self.mesh, wells_config)
        well_manager.initialize_wells(
            self.physics.permeability, self.physics.mu_w)  # 使用水相粘度初始化
        
        # 设置初始条件
        pressure = np.full(self.total_cells, self.initial_pressure)
        saturation = np.full(self.total_cells, self.initial_saturation)
        
        # 更新单元属性
        for i, cell in enumerate(self.mesh.cell_list):
            cell.press = pressure[i]
            cell.Sw = saturation[i]
            # 计算两相流参数
            self.physics.compute_2phase_param(cell, saturation[i])
        
        # 结果存储
        results = {
            'pressure_history': [pressure.copy()],
            'saturation_history': [saturation.copy()],
            'time_history': [0.0],
            'well_data': []
        }
        
        # 时间循环
        current_time = 0.0
        time_step = 0
        
        while current_time < self.total_time:
            time_step += 1
            current_time += self.dt
            
            # 求解一个时间步
            pressure, saturation = self.solve_time_step(
                pressure, saturation, self.dt, well_manager)
            
            # 更新单元属性
            for i, cell in enumerate(self.mesh.cell_list):
                cell.press = pressure[i]
                cell.Sw = saturation[i]
                # 计算两相流参数
                self.physics.compute_2phase_param(cell, saturation[i])
            
            # 保存输出
            if time_step % self.output_interval == 0:
                results['pressure_history'].append(pressure.copy())
                results['saturation_history'].append(saturation.copy())
                results['time_history'].append(current_time)
                
                # 保存井数据
                well_data = {}
                for i, well in enumerate(well_manager.wells):
                    z, y, x = well.location
                    cell_index = self.mesh.get_cell_index(z, y, x)
                    well_data[f'well_{i}'] = {
                        'pressure': pressure[cell_index],
                        'saturation': saturation[cell_index],
                        'location': well.location,
                        'control_type': well.control_type,
                        'value': well.value
                    }
                results['well_data'].append(well_data)
        
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
        self.initial_saturation = self.config.get('initial_saturation', self.initial_saturation)
        
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
            'total_cells': self.total_cells,
            'dt': self.dt,
            'total_time': self.total_time,
            'output_interval': self.output_interval,
            'initial_pressure': self.initial_pressure,
            'initial_saturation': self.initial_saturation,
            'linear_solver_info': self.linear_solver.get_info()
        }
    
    def __repr__(self):
        return f"TwoPhaseIMPES({self.mesh.nx}x{self.mesh.ny}x{self.mesh.nz})"


def create_impes_solver(config: Dict[str, Any]) -> TwoPhaseIMPES:
    """
    根据配置创建IMPES求解器
    
    Args:
        config: 配置字典，包含网格和物理属性配置
        
    Returns:
        IMPES求解器实例
    """
    from reservoirpy.mesh.mesh import StructuredMesh
    from reservoirpy.physics.physics import TwoPhaseProperties
    
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
    physics = TwoPhaseProperties(mesh, physics_config)
    
    # 更新单元物理属性
    for i, cell in enumerate(mesh.cell_list):
        physics.update_cell_properties(cell, i)
    
    # 创建求解器
    solver_config = config.get('solver', {})
    solver = TwoPhaseIMPES(mesh, physics, solver_config)
    
    return solver


def run_impes_simulation(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行IMPES模拟
    
    Args:
        config: 配置字典
        
    Returns:
        模拟结果
    """
    # 创建求解器
    solver = create_impes_solver(config)
    
    # 获取井配置
    wells_config = config.get('wells', [])
    
    # 运行模拟
    results = solver.solve_simulation(wells_config)
    results['solver_info'] = solver.get_info()
    
    return results
