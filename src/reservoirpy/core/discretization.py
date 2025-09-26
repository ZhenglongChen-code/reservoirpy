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
    
    实现单相流和两相流的有限体积法离散化
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
        """
        单相流FVM离散化 - 使用有限体积法离散微分方程
        
        原始微分方程：
        ∂/∂t(φ·ρ) + ∇·(ρ·v) = q
        
        其中 v = -(k/μ)·∇p (达西定律)
        
        对于微可压缩流体，结合连续性方程得到压力方程：
        φ·c·∂p/∂t = ∇·(k/μ·∇p) + q/ρ
        
        FVM积分形式（对控制体积V积分）：
        ∫∫∫_V φ·c·∂p/∂t dV = ∫∫∫_V ∇·(k/μ·∇p) dV + ∫∫∫_V q/ρ dV
        
        应用散度定理和中点规则离散化：
        V·φ·c·(p^{n+1} - p^n)/Δt = Σ_faces T_face·(p_neighbor - p_center) + Q_well
        
        重组为线性系统 A·p^{n+1} = b：
        [V·φ·c/Δt + Σ_faces T_face]·p_i^{n+1} - Σ_neighbors T_ij·p_j^{n+1} = V·φ·c·p_i^n/Δt + Q_well
        
        Args:
            dt: 时间步长 Δt
            pressure: 当前时间步压力场 p^n
            well_manager: 井管理器
            
        Returns:
            (系数矩阵A, 右端向量b)
        """
        # 直接使用稀疏矩阵初始化，提高内存效率
        # 使用COO格式构建矩阵，便于增量更新
        from scipy.sparse import lil_matrix
        A = lil_matrix((self.total_cells, self.total_cells))
        b = np.zeros(self.total_cells)
        
        # 对每个控制体积（网格单元）应用FVM离散化
        for i in range(self.total_cells):
            cell = self.mesh.cell_list[i]
            z, y, x = self.mesh.get_cell_coords(i)
            
            # ====== 1. 累积项离散化 ======
            # 原始项: ∫∫∫_V φ·c·∂p/∂t dV ≈ V·φ·c·(p^{n+1} - p^n)/Δt
            # 系数矩阵贡献: V·φ·c/Δt (对角项)
            compressibility = getattr(self.physics, 'compressibility', 1e-9)  # 默认值
                
            # 从属性管理器获取孔隙度
            porosity = self.physics.property_manager.get_cell_property(i, 'porosity')
                
            accumulation_coeff = (float(cell.volume) * float(porosity) * 
                                 float(compressibility) / dt)
            
            A[i, i] += accumulation_coeff  # 对角项: V·φ·c/Δt
            
            # 右端向量贡献: V·φ·c·p^n/Δt (前一时间步的贡献)
            b[i] += accumulation_coeff * float(pressure[i])
            
            # ====== 2. 扩散项离散化 ======
            # 原始项: ∫∫∫_V ∇·(k/μ·∇p) dV
            # 应用散度定理: ∫∫_∂V (k/μ·∇p)·n̂ dA
            # 离散化: Σ_faces T_face·(p_neighbor - p_center)
            # 其中 T_face = k_harmonic·A_face/(μ·d_face) 是传导率
            
            neighbors = self.mesh.get_neighbors(z, y, x)
            total_transmissibility = 0.0
            
            for direction, neighbor_idx in enumerate(neighbors):
                if neighbor_idx != -1:  # 内部界面
                    # 传导率 T_ij = k_harmonic·A_ij/(μ·d_ij)
                    trans = self.trans_matrix[direction, i]
                    total_transmissibility += trans
                    
                    # 非对角项: -T_ij·p_j (相邻单元的贡献)
                    # 来自 T_ij·(p_j - p_i) 中的 T_ij·p_j 项
                    A[i, neighbor_idx] -= trans
                # else: 边界界面，零通量边界条件（不需要额外处理）
            
            # 对角项: +Σ_faces T_face (来自 -T_ij·(-p_i) = +T_ij·p_i)
            A[i, i] += total_transmissibility
        
        # ====== 3. 应用井项 ======
        # Q_well 项通过井管理器添加到线性系统中
        well_manager.apply_well_terms(A, b, pressure, dt)
        
        # ====== 4. 转换为CSR格式提高求解效率 ======
        A_csr = A.tocsr()
        
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
            solution = spsolve(A, b)
            return np.array(solution)  # 确保返回numpy数组
        elif method == 'cg':
            # 共轭梯度法 - 使用正确的参数名称
            x, info = cg(A, b, rtol=tolerance, maxiter=max_iterations)
            if info != 0:
                warnings.warn(f"CG solver did not converge: info={info}")
            return x
        elif method == 'bicgstab':
            # 双共轭梯度稳定法 - 使用正确的参数名称
            x, info = bicgstab(A, b, rtol=tolerance, maxiter=max_iterations)
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
        # 暂时简化处理，需要创建临时井管理器
        from .well_model import WellManager
        temp_well_manager = WellManager(self.mesh, wells)
        temp_well_manager.initialize_wells(getattr(self.physics, 'permeability', np.ones((1,1,1,3))), 
                                          getattr(self.physics, 'viscosity', 0.001))
        return self.discretize_single_phase(dt, pressure, temp_well_manager)
    
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