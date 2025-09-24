"""
有限体积法离散化模块

实现单相流和两相流的有限体积法离散化
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, cg, bicgstab
import warnings

from reservoirpy.mesh.mesh import StructuredMesh, CubeCell
from reservoirpy.physics.physics import SinglePhaseProperties
from .well_model import WellManager


class FVMDiscretizer:
    """
    有限体积法离散化器
    
    实现单相流和两相流的有限体积法离散化
    """
    
    def __init__(self, mesh: StructuredMesh, physics: SinglePhaseProperties):
        """
        初始化离散化器
        
        Args:
            mesh: 结构化网格
            physics: 物理属性
        """
        self.mesh = mesh
        self.physics = physics
        self.total_cells = mesh.total_cells
        
        # 计算传导率矩阵
        self._compute_transmissibilities()
        
    def _compute_transmissibilities(self):
        """
        计算传导率矩阵
        """
        # 初始化传导率矩阵 (6个方向: W, E, N, S, B, T)
        self.trans_matrix = np.zeros((6, self.total_cells))
        
        for i in range(self.total_cells):
            z, y, x = self.mesh.get_cell_coords(i)
            cell = self.mesh.cell_list[i]
            
            # 获取邻居单元
            neighbors = self.mesh.get_neighbors(z, y, x)
            
            # 计算各方向的传导率
            for direction, neighbor_idx in enumerate(neighbors):
                if neighbor_idx != -1:
                    # 计算传导率
                    trans = self._compute_transmissibility(i, neighbor_idx, direction)
                    self.trans_matrix[direction, i] = trans
                    
    def _compute_transmissibility(self, cell_idx1: int, cell_idx2: int, direction: int) -> float:
        """
        计算两个相邻单元之间的传导率
        
        Args:
            cell_idx1: 单元1索引
            cell_idx2: 单元2索引
            direction: 方向 (0:W, 1:E, 2:N, 3:S, 4:B, 5:T)
            
        Returns:
            传导率值
        """
        cell1 = self.mesh.cell_list[cell_idx1]
        cell2 = self.mesh.cell_list[cell_idx2]
        
        # 获取渗透率
        k1 = self._get_permeability(cell1, direction)
        k2 = self._get_permeability(cell2, direction)
        
        # 调和平均渗透率
        k_harmonic = 2 * k1 * k2 / (k1 + k2) if (k1 + k2) > 0 else 0
        
        # 获取接触面积和距离
        area = self._get_contact_area(direction)
        distance = self._get_contact_distance(direction)
        
        # 传导率 = k * A / d / μ
        if k_harmonic > 0:
            transmissibility = k_harmonic * float(area) / float(distance) / float(self.physics.viscosity)
        else:
            transmissibility = 0.0
        
        return transmissibility
    
    def _get_permeability(self, cell: CubeCell, direction: int) -> float:
        """
        获取指定方向的渗透率
        
        Args:
            cell: 单元对象
            direction: 方向
            
        Returns:
            渗透率值
        """
        if direction in [0, 1]:  # W, E (x方向)
            return cell.kx
        elif direction in [2, 3]:  # N, S (y方向)
            return cell.ky
        else:  # B, T (z方向)
            return cell.kz
    
    def _get_contact_area(self, direction: int) -> float:
        """
        获取接触面积
        
        Args:
            direction: 方向
            
        Returns:
            接触面积
        """
        if direction in [0, 1]:  # W, E (x方向)
            return self.mesh.dy * self.mesh.dz
        elif direction in [2, 3]:  # N, S (y方向)
            return self.mesh.dx * self.mesh.dz
        else:  # B, T (z方向)
            return self.mesh.dx * self.mesh.dy
    
    def _get_contact_distance(self, direction: int) -> float:
        """
        获取接触距离
        
        Args:
            direction: 方向
            
        Returns:
            接触距离
        """
        if direction in [0, 1]:  # W, E (x方向)
            return self.mesh.dx
        elif direction in [2, 3]:  # N, S (y方向)
            return self.mesh.dy
        else:  # B, T (z方向)
            return self.mesh.dz
    
    def discretize_single_phase(self, dt: float, pressure: np.ndarray, 
                               well_manager: WellManager) -> Tuple[csr_matrix, np.ndarray]:
        """
        离散化单相流方程
        
        Args:
            dt: 时间步长
            pressure: 当前压力场
            well_manager: 井管理器
            
        Returns:
            (系数矩阵, 右端向量)
        """
        # 初始化系数矩阵和右端向量
        A_data = []
        A_row = []
        A_col = []
        b = np.zeros(self.total_cells)
        
        for i in range(self.total_cells):
            cell = self.mesh.cell_list[i]
            z, y, x = self.mesh.get_cell_coords(i)
            
            # 累积项
            accumulation = float(cell.volume) * float(cell.porosity) * float(self.physics.compressibility)
            A_data.append(accumulation)
            A_row.append(i)
            A_col.append(i)
            
            # 右端向量（时间项）
            b[i] = float(cell.volume) * float(cell.porosity) * float(self.physics.compressibility) * float(pressure[i])
            
            # 井项（通过井管理器处理）
            pass  # 井项将在后面统一处理
            
            # 通量项
            neighbors = self.mesh.get_neighbors(z, y, x)
            for direction, neighbor_idx in enumerate(neighbors):
                if neighbor_idx != -1:
                    trans = self.trans_matrix[direction, i]
                    
                    # 对角项
                    A_data.append(trans * dt)
                    A_row.append(i)
                    A_col.append(i)
                    
                    # 非对角项
                    A_data.append(-trans * dt)
                    A_row.append(i)
                    A_col.append(neighbor_idx)
        
        # 构建稀疏矩阵
        A = csr_matrix((A_data, (A_row, A_col)), shape=(self.total_cells, self.total_cells))
        
        # 应用井项
        well_manager.apply_well_terms(A, b, pressure, dt)
        
        return A, b
    
    
    def solve_linear_system(self, A: csr_matrix, b: np.ndarray, 
                           method: str = 'bicgstab', 
                           tolerance: float = 1e-8,
                           max_iterations: int = 1000) -> np.ndarray:
        """
        求解线性系统
        
        Args:
            A: 系数矩阵
            b: 右端向量
            method: 求解方法
            tolerance: 收敛容差
            max_iterations: 最大迭代次数
            
        Returns:
            解向量
        """
        if method == 'direct':
            # 直接求解
            return spsolve(A, b)
        elif method == 'cg':
            # 共轭梯度法
            x, info = cg(A, b, tol=tolerance, maxiter=max_iterations)
            if info != 0:
                warnings.warn(f"CG solver did not converge: info={info}")
            return x
        elif method == 'bicgstab':
            # 双共轭梯度稳定法
            x, info = bicgstab(A, b, tol=tolerance, maxiter=max_iterations)
            if info != 0:
                warnings.warn(f"BiCGSTAB solver did not converge: info={info}")
            return x
        else:
            raise ValueError(f"Unknown solver method: {method}")
    
    def discretize_two_phase(self, dt: float, pressure: np.ndarray, 
                            saturation: np.ndarray,
                            wells: List[Dict[str, Any]]) -> Tuple[csr_matrix, np.ndarray]:
        """
        离散化两相流方程（IMPES方法）
        
        Args:
            dt: 时间步长
            pressure: 当前压力场
            saturation: 当前饱和度场
            wells: 井配置列表
            
        Returns:
            (系数矩阵, 右端向量)
        """
        # 这里实现两相流的离散化
        # 暂时返回单相流的结果
        return self.discretize_single_phase(dt, pressure, wells)
    
    def __repr__(self):
        return f"FVMDiscretizer({self.mesh.nx}x{self.mesh.ny}x{self.mesh.nz})"
