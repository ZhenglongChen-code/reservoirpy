"""
井模型模块
基于现有井模型实现重构，提供完整的井模型功能
"""

import numpy as np
import math
from typing import Dict, Any, List
from reservoirpy.mesh.mesh import StructuredMesh


class Well:
    """井模型类"""
    
    def __init__(self, location: List[int], control_type: str, value: float, 
                 rw: float = 0.05, skin_factor: float = 0, well_length: float = 1000):
        """
        初始化井模型
        
        Args:
            location: 井所在网格索引 (z, y, x)
            control_type: 控制类型 ('rate' 或 'bhp')
            value: 控制值（流量 m³/s 或压力 Pa）
            rw: 井筒半径（m）
            skin_factor: 表皮因子
            well_length: 井长度（m）
        """
        self.location = location
        self.control_type = control_type
        self.value = value
        self.rw = rw
        self.skin_factor = skin_factor
        self.well_length = well_length
        self.re = None  # 等效半径（由Peaceman公式计算）
        self.well_index = None  # 产能指数
        
    def compute_well_index(self, mesh: StructuredMesh, permeability: float, 
                          viscosity: float=0.001) -> float:
        """
        计算产能指数 WI = 2πKh / (μ(ln(re/rw) + S))
        
        Args:
            mesh: 网格对象
            permeability: 渗透率 (m²)
            viscosity: 粘度 (Pa·s) 默认1e-3
            
        Returns:
            产能指数 WI
        """
        # 计算等效半径（Peaceman公式）
        dx = float(mesh.dx)
        dy = float(mesh.dy)
        dz = float(mesh.dz)
        self.re = 0.14 * (dx**2 + dy**2)**0.5
        
        # 计算产能指数 - 使用实际的渗透率值
        self.well_index = (2 * np.pi * float(permeability) * float(dz) / 
                          (float(viscosity) * (math.log(self.re / self.rw) + self.skin_factor)))
        
        return self.well_index

    def compute_well_term(self, pressure: float) -> float:
        """
        计算井项贡献

        Args:
            pressure: 当前压力 (Pa)

        Returns:
            井项值
        """
        if self.well_index is None:
            raise ValueError("Well index not computed. Call compute_well_index first.")

        if self.control_type == 'bhp':
            # 定井底流压: q = PI * (p - bhp)
            bhp = self.value
            return self.well_index * (pressure - bhp)
        elif self.control_type == 'rate':
            # 定流量: 直接返回流量值
            return self.value
        else:
            raise ValueError(f"Unknown control type: {self.control_type}")

    def add_to_rhs(self, b: np.ndarray, cell_index: int, pressure: float):
        """
        将井项添加到右端向量

        Args:
            b: 右端向量
            cell_index: 井所在单元索引
            pressure: 当前压力
        """
        if self.control_type == 'bhp':
            # 定井底流压：添加 +PI * bhp 到右端向量
            # 因为离散方程为: (accumulation + transmissibility) * p_new = accumulation * p_old + q
            # 对于BHP井: q = PI * (p_new - bhp)，移项后得到: (accumulation + transmissibility + PI) * p_new = accumulation * p_old + PI * bhp
            b[cell_index] += self.well_index * self.value
        elif self.control_type == 'rate':
            # 定流量：直接添加到右端向量
            b[cell_index] += self.value

    def add_to_matrix(self, A, cell_index: int):
        """
        将井项添加到系数矩阵（仅用于定井底流压）

        Args:
            A: 系数矩阵
            cell_index: 单元索引
        """
        if self.control_type == 'bhp':
            # 定井底流压：在对角线上加上产能指数项
            # 对应离散方程中的 (accumulation + transmissibility + PI) * p_new 项
            A[cell_index, cell_index] += self.well_index


class WellManager:
    """井管理器类"""
    
    def __init__(self, mesh: StructuredMesh, wells_config: List[Dict[str, Any]]):
        """
        初始化井管理器
        
        Args:
            mesh: 网格对象
            wells_config: 井配置列表
        """
        self.mesh = mesh
        self.wells = []
        
        for well_config in wells_config:
            well = Well(
                location=well_config['location'],
                control_type=well_config['control_type'],
                value=well_config['value'],
                rw=well_config.get('rw', 0.05),
                skin_factor=well_config.get('skin_factor', 0),
                well_length=well_config.get('well_length', 1000)
            )
            self.wells.append(well)
    
    def initialize_wells(self, permeability: np.ndarray, viscosity: float):
        """
        初始化所有井的产能指数
        
        Args:
            permeability: 渗透率场
            viscosity: 粘度
        """
        for well in self.wells:
            z, y, x = well.location
            # cell_index = self.mesh.get_cell_index(z, y, x)
            
            # 获取井所在单元的渗透率
            # 修正坐标顺序：permeability 形状为 (nz, ny, nx, 3)
            if permeability.ndim == 4:  # (nz, ny, nx, 3)
                k = permeability[z, y, x, 0]  # Kx - 使用正确的坐标顺序 (z, y, x)
            else:  # (nz, ny, nx)
                k = permeability[z, y, x]  # 使用正确的坐标顺序 (z, y, x)
            
            well.compute_well_index(self.mesh, k, viscosity)
    
    def get_well_cells(self) -> List[int]:
        """获取所有井所在单元的索引"""
        well_cells = []
        for well in self.wells:
            z, y, x = well.location
            cell_index = self.mesh.get_cell_index(z, y, x)
            well_cells.append(cell_index)
        return well_cells

    def apply_well_terms(self, A, b: np.ndarray, pressure: np.ndarray, dt: float):
        """
        将所有井项应用到线性系统

        Args:
            A: 系数矩阵
            b: 右端向量
            pressure: 当前压力场
            dt: 时间步长
        """
        for well in self.wells:
            z, y, x = well.location
            cell_index = self.mesh.get_cell_index(z, y, x)

            # 添加井项到右端向量
            well.add_to_rhs(b, cell_index, pressure[cell_index])

            # 添加井项到系数矩阵（仅定井底流压）
            well.add_to_matrix(A, cell_index)

    def get_well_production(self, pressure: np.ndarray, dt: float) -> Dict[str, float]:
        """
        计算井的生产数据
        
        Args:
            pressure: 当前压力场
            dt: 时间步长
            
        Returns:
            井生产数据字典
        """
        production = {}
        
        for i, well in enumerate(self.wells):
            z, y, x = well.location
            cell_index = self.mesh.get_cell_index(z, y, x)
            
            if well.control_type == 'bhp':
                # 定井底流压：计算流量
                well_term = well.compute_well_term(pressure[cell_index], dt)
                production[f'well_{i+1}_rate'] = well_term
                production[f'well_{i+1}_bhp'] = well.value
            else:
                # 定流量：计算井底流压
                production[f'well_{i+1}_rate'] = well.value
                # 需要求解井底流压（这里简化处理）
                production[f'well_{i+1}_bhp'] = pressure[cell_index]
        
        return production


def create_well_from_config(well_config: Dict[str, Any]) -> Well:
    """
    从配置创建井对象
    
    Args:
        well_config: 井配置字典
        
    Returns:
        井对象
    """
    return Well(
        location=well_config['location'],
        control_type=well_config['control_type'],
        value=well_config['value'],
        rw=well_config.get('rw', 0.05),
        skin_factor=well_config.get('skin_factor', 0),
        well_length=well_config.get('well_length', 1000)
    )


def validate_well_config(well_config: Dict[str, Any], mesh: StructuredMesh) -> bool:
    """
    验证井配置的有效性
    
    Args:
        well_config: 井配置字典
        mesh: 网格对象
        
    Returns:
        配置是否有效
    """
    try:
        location = well_config['location']
        if len(location) != 3:
            return False
        
        z, y, x = location
        if not (0 <= z < mesh.nz and 0 <= y < mesh.ny and 0 <= x < mesh.nx):
            return False
        
        control_type = well_config['control_type']
        if control_type not in ['rate', 'bhp']:
            return False
        
        value = well_config['value']
        if not isinstance(value, (int, float)) or value < 0:
            return False
        
        return True
    except (KeyError, TypeError):
        return False
