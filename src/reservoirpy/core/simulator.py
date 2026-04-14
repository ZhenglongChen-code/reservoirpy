"""
主模拟器模块

采用策略模式，将具体的数学模型委托给Model对象处理。
协调所有模块，提供用户友好的运行接口。
"""

import yaml
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Union
from reservoirpy.mesh.mesh import StructuredMesh
from reservoirpy.physics.physics import SinglePhaseProperties, TwoPhaseProperties
from .well_model import WellManager
from .output_manager import OutputManager
from ..models.model_factory import ModelFactory
from ..models.base_model import BaseModel

logger = logging.getLogger(__name__)


class ReservoirSimulator:
    """
    油藏模拟器顶层接口

    提供配置驱动的模拟流程，自动创建和管理 Mesh、Physics、Model、WellManager 等组件。

    使用方式:
        1. 从 YAML 文件创建: ReservoirSimulator('config.yaml')
        2. 从字典创建: ReservoirSimulator(config_dict={...})

    支持的物理模型:
        - single_phase: 单相流
        - two_phase_impes: 两相流IMPES
        - two_phase_fim: 两相流全隐式

    Example:
        >>> sim = ReservoirSimulator(config_dict={
        ...     'mesh': {'nx': 10, 'ny': 10, 'nz': 1, 'dx': 10, 'dy': 10, 'dz': 10},
        ...     'physics': {'type': 'single_phase', 'permeability': 100, ...},
        ...     'wells': [...],
        ...     'simulation': {'dt': 86400, 'total_time': 864000, ...}
        ... })
        >>> results = sim.run_simulation()
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        """
        初始化模拟器
        
        Args:
            config_path: 配置文件路径
            config_dict: 配置字典（可选，用于程序化配置）
        """
        # 加载和验证配置
        self.config = self._load_config(config_path, config_dict)
        
        # 创建基础组件
        self.mesh = self._create_mesh()
        self.physics = self._create_physics()
        self.well_manager = self._create_well_manager()
        self.output_manager = self._create_output_manager()
        
        # 通过工厂创建数学模型
        model_type = self.config['physics']['type']
        model_config = self.config.get('model', {})
        model_config.update(self.config.get('simulation', {}))  # 合并模拟参数
        
        self.model = ModelFactory.create_model(
            model_type, self.mesh, self.physics, model_config)
            
        # 为了向后兼容保留的属性
        self.wells = self.well_manager.wells
        
        logger.info(f"Created ReservoirSimulator with {model_type} model")
        logger.info(f"Mesh: {self.mesh.nx}x{self.mesh.ny}x{self.mesh.nz} = {self.mesh.n_cells} cells")
        logger.info(f"Wells: {len(self.wells)}")
    
    def _load_config(self, config_path: Optional[str], config_dict: Optional[Dict]) -> Dict[str, Any]:
        """
        加载和验证配置
        
        Args:
            config_path: 配置文件路径
            config_dict: 配置字典
            
        Returns:
            配置字典
        """
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        elif config_dict:
            config = config_dict.copy()
        else:
            raise ValueError("Either config_path or config_dict must be provided")
            
        # 验证配置
        self._validate_config(config)
        return config
        
    def _validate_config(self, config: Dict[str, Any]):
        """
        验证配置的有效性
        
        Args:
            config: 配置字典
        """
        required_sections = ['mesh', 'physics', 'simulation']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
                
        # 验证网格配置
        mesh_config = config['mesh']
        required_mesh_params = ['nx', 'ny', 'nz', 'dx', 'dy', 'dz']
        for param in required_mesh_params:
            if param not in mesh_config:
                raise ValueError(f"Missing required mesh parameter: {param}")
                
        # 验证物理配置
        physics_config = config['physics']
        if 'type' not in physics_config:
            raise ValueError("Missing physics type")
            
        if not ModelFactory.is_registered(physics_config['type']):
            available_types = list(ModelFactory.get_registered_models().keys())
            raise ValueError(f"Unknown physics type: {physics_config['type']}. "
                           f"Available types: {available_types}")
    
    def _create_mesh(self) -> StructuredMesh:
        """
        创建网格对象
        
        Returns:
            网格对象
        """
        mesh_config = self.config['mesh']
        return StructuredMesh(
            nx=mesh_config['nx'],
            ny=mesh_config['ny'],
            nz=mesh_config['nz'],
            dx=mesh_config['dx'],
            dy=mesh_config['dy'],
            dz=mesh_config['dz']
        )
        
    def _create_physics(self):
        """
        创建物理属性对象
        
        Returns:
            物理属性对象
        """
        physics_config = self.config['physics']
        physics_type = physics_config['type']
        
        if physics_type in ['single_phase']:
            return SinglePhaseProperties(self.mesh, physics_config)
        elif physics_type in ['two_phase_impes', 'two_phase_fim']:
            return TwoPhaseProperties(self.mesh, physics_config)
        else:
            raise ValueError(f"Unsupported physics type for property creation: {physics_type}")
            
    def _create_well_manager(self) -> WellManager:
        """
        创建井管理器
        
        Returns:
            井管理器对象
        """
        wells_config = self.config.get('wells', [])
        well_manager = WellManager(self.mesh, wells_config)
        
        # 初始化井的产能指数
        # 注意：这里使用的是物理属性管理器的接口
        permeability = self.physics.property_manager.properties['permeability']
        if isinstance(permeability, float):
            # 均质渗透率
            import numpy as np
            nx, ny, nz = self.mesh.grid_shape
            permeability = np.full((nz, ny, nx), permeability)
            
        well_manager.initialize_wells(permeability, self.physics.viscosity)
        
        return well_manager
        
    def _create_output_manager(self) -> OutputManager:
        """
        创建输出管理器
        
        Returns:
            输出管理器对象
        """
        output_config = self.config.get('output', {})
        return OutputManager(output_config)
    
    def run_simulation(self) -> Dict[str, Any]:
        """
        运行模拟
        
        委托给数学模型执行具体的求解过程
        
        Returns:
            模拟结果
        """
        # 初始化状态变量
        initial_state = self.model.initialize_state(self.config['simulation'])
        
        # 获取模拟参数
        sim_config = self.config['simulation']
        dt = sim_config['dt']
        total_time = sim_config['total_time']
        
        # 委托给模型执行求解
        logger.info("Starting reservoir simulation...")
        results = self.model.solve_simulation(
            initial_state, dt, total_time, 
            self.well_manager, self.output_manager)
            
        logger.info("Simulation completed successfully")
        return results
        
    def run_steady_state(self) -> Dict[str, np.ndarray]:
        """
        运行稳态模拟
        
        Returns:
            稳态解
        """
        try:
            return self.model.solve_steady_state(self.well_manager)
        except NotImplementedError:
            raise NotImplementedError(f"Steady state solving not supported for {type(self.model).__name__}")
    
    # 为了向后兼容的方法
    def get_pressure_field(self) -> np.ndarray:
        """
        获取当前压力场
        
        Returns:
            压力场数组
        """
        final_state = self.output_manager.get_final_state()
        return final_state.get('pressure', np.array([]))
        
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
        pressure_field = self.get_pressure_field()
        if len(pressure_field) > cell_index:
            return pressure_field[cell_index]
        else:
            return 0.0
            
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
            
        results = self.output_manager.get_results()
        times = results.get('time_history', np.array([]))
        
        # 这里可以扩展为从输出管理器中获取井数据
        well_pressure = []
        pressure_history = results.get('field_data', {}).get('pressure', [])
        
        if len(pressure_history) > 0:
            well = self.wells[well_index]
            z, y, x = well.location
            cell_index = self.mesh.get_cell_index(z, y, x)
            
            for pressure_field in pressure_history:
                well_pressure.append(pressure_field[cell_index])
                
        return {
            'time': times,
            'pressure': np.array(well_pressure)
        }
    
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
        
        # 从物理属性管理器获取属性
        porosity = self.physics.property_manager.get_cell_property(cell_index, 'porosity')
        permeability = self.physics.property_manager.get_cell_property(cell_index, 'permeability')
        
        return {
            'index': cell_index,
            'center': cell.center,
            'volume': cell.volume,
            'pressure': getattr(cell, 'press', 0.0),
            'porosity': porosity,
            'permeability': permeability,
            'neighbors': getattr(cell, 'neighbors', []),
            'boundary_type': getattr(cell, 'boundary_type', 0),
            'well_mark': getattr(cell, 'markwell', 0)
        }
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return self.model.get_model_info()
        
    def save_results(self, filename: str, format: str = 'npz'):
        """
        保存结果到文件
        
        Args:
            filename: 文件名
            format: 文件格式
        """
        self.output_manager.save_to_file(filename, format)
        
    def get_simulation_results(self) -> Dict[str, Any]:
        """
        获取模拟结果
        
        Returns:
            模拟结果字典
        """
        return self.output_manager.get_results()
        
    def __repr__(self):
        model_type = self.config['physics']['type']
        return f"ReservoirSimulator({self.mesh.nx}x{self.mesh.ny}x{self.mesh.nz}, model={model_type}, wells={len(self.wells)})"
