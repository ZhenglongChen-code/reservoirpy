"""
有限体积法离散化模块

实现单相流和两相流的有限体积法离散化
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Union
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, cg, bicgstab
import warnings

from reservoirpy.mesh.mesh import StructuredMesh, CubeCell
from reservoirpy.physics.physics import BasePhysics, SinglePhaseProperties, TwoPhaseProperties
from .well_model import WellManager


class FVMDiscretizer:
    """
    有限体积法离散化器

    将渗流微分方程离散化为线性系统 Ax=b。

    支持的离散化:
        - 单相流: discretize_single_phase()
        - 两相流: discretize_two_phase() (待实现)

    内部使用 COO 稀疏矩阵格式批量构建，最终转为 CSR 格式供求解器使用。

    Attributes:
        mesh: 结构化网格对象
        physics: 物理属性对象（单相或两相）
        total_cells: 总单元数
        trans_matrix: 传导率矩阵，形状 (6, n_cells)，6个方向 [W, E, N, S, B, T]

    Example:
        >>> discretizer = FVMDiscretizer(mesh, physics)
        >>> A, b = discretizer.discretize_single_phase(dt, pressure, well_manager)
    """
    
    def __init__(self, mesh: StructuredMesh, physics: BasePhysics):
        """
        初始化离散化器
        
        Args:
            mesh: 结构化网格
            physics: 物理属性（可以是单相或两相）
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
        # 将方向索引转换为字符串表示
        direction_map = {0: 'x', 1: 'x', 2: 'y', 3: 'y', 4: 'z', 5: 'z'}
        direction_str = direction_map[direction]
        
        # 使用物理属性对象计算传导率
        transmissibility = self.physics.get_transmissibility(cell_idx1, cell_idx2, direction_str)
        
        return transmissibility
    
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
        from scipy.sparse import coo_matrix, lil_matrix
        
        n = self.total_cells
        b = np.zeros(n)
        
        compressibility = getattr(self.physics, 'compressibility', 1e-9)
        
        porosity = np.array([
            self.physics.property_manager.get_cell_property(i, 'porosity')
            for i in range(n)
        ])
        volumes = np.array([float(self.mesh.cell_list[i].volume) for i in range(n)])
        
        acc_coeff = volumes * porosity * compressibility / dt
        b = acc_coeff * pressure
        
        rows = []
        cols = []
        data = []
        
        diag = acc_coeff.copy()
        
        for direction in range(6):
            trans = self.trans_matrix[direction]
            mask = trans != 0.0
            if not np.any(mask):
                continue
                
            cell_indices = np.where(mask)[0]
            trans_vals = trans[cell_indices]
            
            neighbors = np.array([
                self.mesh.cell_list[i].neighbors[direction] 
                for i in cell_indices
            ])
            
            valid = neighbors != -1
            cell_indices = cell_indices[valid]
            trans_vals = trans_vals[valid]
            neighbor_indices = neighbors[valid]
            
            rows.extend(cell_indices.tolist())
            cols.extend(neighbor_indices.tolist())
            data.extend((-trans_vals).tolist())
            
            np.add.at(diag, cell_indices, trans_vals)
        
        rows.extend(range(n))
        cols.extend(range(n))
        data.extend(diag.tolist())
        
        A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        
        A_lil = A.tolil()
        well_manager.apply_well_terms(A_lil, b, pressure, dt)
        A_csr = A_lil.tocsr()
        
        return A_csr, b
    
    def _verify_matrix_properties(self, A: Union[np.ndarray, csr_matrix]) -> Dict[str, Any]:
        """
        验证FVM离散矩阵的数值性质
        
        对于稳定的FVM离散化，系数矩阵应该满足：
        1. 对角占优性 (Diagonal Dominance): |A_ii| >= Σ|A_ij| for j≠i
        2. 良好的条件数 (Condition Number): cond(A) < 1e12
        3. 对称性 (可选，取决于边界条件)
        
        Args:
            A: 系数矩阵
            
        Returns:
            矩阵性质字典
        """
        if isinstance(A, csr_matrix):
            A_dense = A.toarray()
        else:
            A_dense = A
            
        properties = {}
        
        # 检查对角占优性 - FVM物理守恒的重要条件
        # |A_ii| >= Σ|A_ij| 确保数值稳定性
        diagonal = np.abs(np.diag(A_dense))
        off_diagonal_sum = np.sum(np.abs(A_dense), axis=1) - diagonal
        diagonal_dominance = np.all(diagonal >= off_diagonal_sum)
        properties['diagonal_dominant'] = diagonal_dominance
        properties['diagonal_dominance_ratio'] = np.min(diagonal / (off_diagonal_sum + 1e-15))
        
        # 检查条件数 - 线性求解器收敛性的关键指标
        try:
            cond_num = np.linalg.cond(A_dense)
            properties['condition_number'] = cond_num
            properties['well_conditioned'] = cond_num < 1e12
        except:
            properties['condition_number'] = np.inf
            properties['well_conditioned'] = False
            
        # 检查对称性 - 某些边界条件下FVM矩阵可能对称
        properties['symmetric'] = np.allclose(A_dense, A_dense.T, rtol=1e-10)
        
        # 检查是否有零对角元素 - 会导致奇异矩阵
        zero_diagonal = np.any(np.abs(diagonal) < 1e-15)
        properties['has_zero_diagonal'] = zero_diagonal
        
        # 矩阵稠密度
        nonzero_ratio = np.count_nonzero(A_dense) / A_dense.size
        properties['sparsity'] = 1.0 - nonzero_ratio
        
        return properties

    def discretize_two_phase(self, dt: float, pressure: np.ndarray,
                            saturation: np.ndarray,
                            well_manager) -> Tuple[csr_matrix, np.ndarray]:
        from scipy.sparse import coo_matrix, lil_matrix

        n = self.total_cells
        physics = self.physics

        lambda_w = np.zeros(n)
        lambda_o = np.zeros(n)
        for i in range(n):
            Sw = saturation[i]
            kro = physics.get_relative_permeability(Sw, 'oil')
            krw = physics.get_relative_permeability(Sw, 'water')
            lambda_w[i] = krw / physics.mu_w
            lambda_o[i] = kro / physics.mu_o
        lambda_t = lambda_w + lambda_o

        porosity = np.array([
            physics.property_manager.get_cell_property(i, 'porosity')
            for i in range(n)
        ])
        volumes = np.array([float(self.mesh.cell_list[i].volume) for i in range(n)])
        compressibility = getattr(physics, 'compressibility', 1e-9)

        acc_coeff = volumes * porosity * compressibility / dt
        b = acc_coeff * pressure

        rows = []
        cols = []
        data = []
        diag = acc_coeff.copy()

        for direction in range(6):
            trans = self.trans_matrix[direction]
            mask = trans != 0.0
            if not np.any(mask):
                continue

            cell_indices = np.where(mask)[0]
            trans_vals = trans[cell_indices]

            neighbors = np.array([
                self.mesh.cell_list[i].neighbors[direction]
                for i in cell_indices
            ])

            valid = neighbors != -1
            ci = cell_indices[valid]
            tv = trans_vals[valid]
            ni = neighbors[valid]

            for idx in range(len(ci)):
                i_cell = ci[idx]
                j_cell = ni[idx]
                lt_i = lambda_t[i_cell]
                lt_j = lambda_t[j_cell]
                lt_face = 2.0 * lt_i * lt_j / (lt_i + lt_j + 1e-30)

                T_total = tv[idx] * lt_face

                rows.append(i_cell)
                cols.append(j_cell)
                data.append(-T_total)
                diag[i_cell] += T_total

        rows.extend(range(n))
        cols.extend(range(n))
        data.extend(diag.tolist())

        A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

        A_lil = A.tolil()
        well_manager.apply_well_terms(A_lil, b, pressure, dt)
        A_csr = A_lil.tocsr()

        return A_csr, b
    
    def __repr__(self):
        return f"FVMDiscretizer({self.mesh.nx}x{self.mesh.ny}x{self.mesh.nz})"


class SinglePhaseFVMDiscretizer(FVMDiscretizer):
    """
    单相流有限体积法离散化器
    """
    
    def __init__(self, mesh: StructuredMesh, physics: SinglePhaseProperties):
        """
        初始化单相流离散化器
        
        Args:
            mesh: 结构化网格
            physics: 单相流物理属性
        """
        super().__init__(mesh, physics)
    
    def discretize(self, dt: float, pressure: np.ndarray, 
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
        return self.discretize_single_phase(dt, pressure, well_manager)


class TwoPhaseFVMDiscretizer(FVMDiscretizer):
    """
    两相流有限体积法离散化器
    """
    
    def __init__(self, mesh: StructuredMesh, physics: TwoPhaseProperties):
        """
        初始化两相流离散化器
        
        Args:
            mesh: 结构化网格
            physics: 两相流物理属性
        """
        super().__init__(mesh, physics)
    
    def discretize_pressure(self, dt: float, pressure: np.ndarray, 
                           saturation: np.ndarray,
                           well_manager: WellManager) -> Tuple[csr_matrix, np.ndarray]:
        """
        离散化两相流压力方程（IMPES方法）
        
        Args:
            dt: 时间步长
            pressure: 当前压力场
            saturation: 当前饱和度场
            well_manager: 井管理器
            
        Returns:
            (系数矩阵, 右端向量)
        """
        # 实现两相流压力方程的离散化
        # 这里需要考虑相对渗透率和相粘度的影响
        return self.discretize_single_phase(dt, pressure, well_manager)  # 简化实现
    
    def discretize_saturation(self, dt: float, pressure_new: np.ndarray,
                             pressure_old: np.ndarray, saturation_old: np.ndarray) -> np.ndarray:
        """
        离散化饱和度方程（IMPES方法）
        
        Args:
            dt: 时间步长
            pressure_new: 新压力场
            pressure_old: 旧压力场
            saturation_old: 旧饱和度场
            
        Returns:
            新饱和度场
        """
        # 实现饱和度方程的显式求解
        saturation_new = saturation_old.copy()
        # 这里需要实现饱和度的更新逻辑
        return saturation_new