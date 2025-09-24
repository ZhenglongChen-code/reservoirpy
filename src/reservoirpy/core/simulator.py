"""
主模拟器模块

协调所有模块，提供用户友好的运行接口。
"""

import yaml
import numpy as np
from typing import Dict, Any, Optional, List, Union
from reservoirpy.mesh.mesh import StructuredMesh
from reservoirpy.physics.physics import SinglePhaseProperties
from .discretization import FVMDiscretizer
from .well_model import WellManager


class ReservoirSimulator:
    """
    油藏数值模拟器主类
    
    协调所有模块，提供用户友好的运行接口。
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        """
        初始化模拟器
        
        Args:
            config_path: 配置文件路径
            config_dict: 配置字典（可选，用于程序化配置）
        """
        # 加载配置
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")
        
        # 初始化网格
        mesh_config = self.config['mesh']
        self.mesh = StructuredMesh(
            nx=mesh_config['nx'],
            ny=mesh_config['ny'],
            nz=mesh_config['nz'],
            dx=mesh_config['dx'],
            dy=mesh_config['dy'],
            dz=mesh_config['dz']
        )
        
        # 初始化物理属性
        physics_config = self.config['physics']
        self.physics = SinglePhaseProperties(self.mesh, physics_config)
        
        # 更新单元物理属性
        for i, cell in enumerate(self.mesh.cell_list):
            self.physics.update_cell_properties(cell, i)
        
        # 初始化井
        self.well_manager = self._init_well_manager()
        self.wells = self.well_manager.wells  # 为了向后兼容
        
        # 初始化离散化器
        self.discretizer = FVMDiscretizer(self.mesh, self.physics)
        
        # 初始化压力场
        self.pressure = self._init_pressure()
        
        # 模拟状态
        self.current_time = 0.0
        self.time_step = 0
        self.results = {
            'pressure_history': [],
            'time_history': [],
            'well_data': []
        }
    
    def _init_well_manager(self) -> WellManager:
        """
        初始化井管理器
        
        Returns:
            井管理器对象
        """
        wells_config = self.config.get('wells', [])
        well_manager = WellManager(self.mesh, wells_config)
        
        # 初始化井的产能指数
        well_manager.initialize_wells(
            self.physics.permeability, 
            self.physics.viscosity
        )
        
        return well_manager
    
    def _init_pressure(self) -> np.ndarray:
        """
        初始化压力场
        
        Returns:
            初始压力场
        """
        initial_pressure = self.config['simulation'].get('initial_pressure', 30e6)
        return np.full(self.mesh.n_cells, initial_pressure)
    
    def set_boundary_conditions(self, boundary_conditions: List[Dict[str, Any]]):
        """
        设置边界条件
        
        Args:
            boundary_conditions: 边界条件列表
        """
        for bc in boundary_conditions:
            cell_index = bc['index']
            cell = self.mesh.cell_list[cell_index]
            
            if 'pressure' in bc['type']:
                cell.press = bc['values'][0]
                cell.mark_bc = 1
            elif 'saturation' in bc['type']:
                cell.Sw = bc['values'][1]
    
    def set_well_mark(self, well_mark: List[Union[Dict[str, Any], int]]):
        """
        设置井标记
        
        Args:
            well_mark: 井标记列表
        """
        if well_mark and isinstance(well_mark[0], int):
            # 列表 of 索引
            for idx in well_mark:
                if isinstance(idx, int):
                    cell = self.mesh.cell_list[idx]
                    cell.markwell = 2  # 生产井
        else:
            # 列表 of 字典
            for well in well_mark:
                if isinstance(well, dict) and 'index' in well:
                    cell = self.mesh.cell_list[well['index']]
                    if well['type'] == 'injector':
                        cell.markwell = 1
                    elif well['type'] == 'producer':
                        cell.markwell = 2
    
    def update_cell_pressure(self, pressure_new: np.ndarray):
        """
        更新单元压力
        
        Args:
            pressure_new: 新的压力场
        """
        self.pressure = pressure_new.copy()
        for i, cell in enumerate(self.mesh.cell_list):
            cell.press = pressure_new[i]
    
    def get_pressure_field(self) -> np.ndarray:
        """
        获取当前压力场
        
        Returns:
            压力场数组
        """
        return self.pressure.copy()
    
    def save_output(self, timestep: int, pressure_field: np.ndarray):
        """
        保存输出结果
        
        Args:
            timestep: 时间步
            pressure_field: 压力场
        """
        self.results['pressure_history'].append(pressure_field.copy())
        self.results['time_history'].append(self.current_time)
        
        # 保存井数据
        well_data = {}
        for i, well in enumerate(self.well_manager.wells):
            cell_index = self.mesh.get_cell_index(*well.location)
            well_data[f'well_{i}'] = {
                'pressure': pressure_field[cell_index],
                'location': well.location,
                'control_type': well.control_type,
                'value': well.value
            }
        self.results['well_data'].append(well_data)
    
    def run_simulation(self) -> Dict[str, Any]:
        """
        运行模拟
        
        Returns:
            模拟结果
        """
        sim_config = self.config['simulation']
        dt = sim_config['dt']
        total_time = sim_config['total_time']
        output_interval = sim_config['output_interval']
        
        # 保存初始状态
        self.save_output(0, self.pressure)
        
        # 时间循环
        while self.current_time < total_time:
            self.time_step += 1
            self.current_time += dt
            
            # 使用真正的压力求解器
            self.pressure = self._solve_pressure(dt)
            
            # 更新单元压力
            self.update_cell_pressure(self.pressure)
            
            # 保存输出
            if self.time_step % output_interval == 0:
                self.save_output(self.time_step, self.pressure)
        
        return self.results
    
    def _solve_pressure(self, dt: float) -> np.ndarray:
        """
        求解压力场
        
        Args:
            dt: 时间步长
            
        Returns:
            新的压力场
        """
        # 离散化方程
        A, b = self.discretizer.discretize_single_phase(dt, self.pressure, self.well_manager)
        
        # 求解线性系统
        solver_config = self.config.get('linear_solver', {})
        method = solver_config.get('method', 'bicgstab')
        tolerance = solver_config.get('tolerance', 1e-8)
        max_iterations = solver_config.get('max_iterations', 1000)
        
        new_pressure = self.discretizer.solve_linear_system(
            A, b, method=method, tolerance=tolerance, max_iterations=max_iterations
        )
        
        return new_pressure
    
    def get_well_production(self, well_index: int) -> Dict[str, np.ndarray]:
        """
        获取井生产数据
        
        Args:
            well_index: 井索引
            
        Returns:
            井生产数据字典
        """
        if well_index >= len(self.wells):
            raise ValueError(f"Invalid well index: {well_index}")
        
        times = np.array(self.results['time_history'])
        pressures = np.array([data[f'well_{well_index}']['pressure'] 
                             for data in self.results['well_data']])
        
        return {
            'time': times,
            'pressure': pressures
        }
    
    def get_pressure_at_location(self, i: int, j: int, k: int) -> float:
        """
        获取指定位置的压力
        
        Args:
            i: Z方向索引
            j: Y方向索引
            k: X方向索引
            
        Returns:
            压力值
        """
        cell_index = self.mesh.get_cell_index(i, j, k)
        return self.pressure[cell_index]
    
    def get_cell_properties(self, i: int, j: int, k: int) -> Dict[str, Any]:
        """
        获取指定单元的属性
        
        Args:
            i: Z方向索引
            j: Y方向索引
            k: X方向索引
            
        Returns:
            单元属性字典
        """
        cell_index = self.mesh.get_cell_index(i, j, k)
        cell = self.mesh.cell_list[cell_index]
        
        return {
            'index': cell_index,
            'center': cell.center,
            'volume': cell.volume,
            'pressure': cell.press,
            'porosity': cell.porosity,
            'permeability': [cell.kx, cell.ky, cell.kz],
            'neighbors': cell.neighbors,
            'boundary_type': cell.boundary_type,
            'well_mark': cell.markwell
        }
    
    def __repr__(self):
        return f"ReservoirSimulator({self.mesh.nx}x{self.mesh.ny}x{self.mesh.nz}, wells={len(self.wells)})"
