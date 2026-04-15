"""
有限体积法离散化模块

实现单相流和两相流的有限体积法离散化。
所有矩阵组装逻辑均使用 NumPy 向量化操作，避免 Python for 循环。
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Union
from scipy.sparse import csr_matrix, coo_matrix
import warnings

from reservoirpy.mesh.mesh import StructuredMesh, CubeCell
from reservoirpy.physics.physics import BasePhysics, SinglePhaseProperties, TwoPhaseProperties
from .well_model import WellManager


class FVMDiscretizer:
    """
    有限体积法离散化器

    将渗流微分方程离散化为线性系统 Ax=b。

    内部使用 NumPy 向量化操作 + COO 稀疏矩阵批量构建，
    最终转为 CSR 格式供求解器使用。

    Attributes:
        mesh: 结构化网格对象
        physics: 物理属性对象（单相或两相）
        total_cells: 总单元数
        trans_matrix: 传导率矩阵，形状 (6, n_cells)，6个方向 [W, E, N, S, B, T]
        neighbor_indices: 邻居索引矩阵，形状 (6, n_cells)，-1 表示边界
        perm_flat: 展平的渗透率数组 (n_cells,)，SI 单位
        perm_z_flat: 展平的垂向渗透率数组 (n_cells,)，SI 单位
        porosity_flat: 展平的孔隙度数组 (n_cells,)
        volumes: 单元体积数组 (n_cells,)

    Example:
        >>> discretizer = FVMDiscretizer(mesh, physics)
        >>> A, b = discretizer.discretize_single_phase(dt, pressure, well_manager)
    """

    def __init__(self, mesh: StructuredMesh, physics: BasePhysics):
        self.mesh = mesh
        self.physics = physics
        self.total_cells = mesh.total_cells

        self._build_neighbor_table()
        self._extract_property_arrays()
        self._compute_transmissibilities()

    def _build_neighbor_table(self):
        """向量化构建邻居索引表，避免逐单元访问 cell.neighbors"""
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz
        n = self.total_cells
        idx = np.arange(n)
        k = idx % nx
        j = (idx // nx) % ny
        i = idx // (nx * ny)

        self.neighbor_indices = np.full((6, n), -1, dtype=np.int32)

        self.neighbor_indices[0] = np.where(k > 0, idx - 1, -1)
        self.neighbor_indices[1] = np.where(k < nx - 1, idx + 1, -1)
        self.neighbor_indices[2] = np.where(j > 0, idx - nx, -1)
        self.neighbor_indices[3] = np.where(j < ny - 1, idx + nx, -1)
        self.neighbor_indices[4] = np.where(i > 0, idx - nx * ny, -1)
        self.neighbor_indices[5] = np.where(i < nz - 1, idx + nx * ny, -1)

        self._dir_axis = np.array([0, 0, 1, 1, 2, 2])

    def _extract_property_arrays(self):
        """向量化提取物理属性为 1D 数组，避免逐单元调用 get_cell_property"""
        pm = self.physics.property_manager
        n = self.total_cells

        def _flatten_prop(name):
            prop = pm.properties.get(name)
            if prop is None:
                return np.zeros(n)
            if isinstance(prop, (int, float)):
                return np.full(n, float(prop))
            if isinstance(prop, np.ndarray):
                return prop.ravel().astype(np.float64)
            return np.zeros(n)

        self.perm_flat = _flatten_prop('permeability')
        self.perm_z_flat = _flatten_prop('permeability_z')
        self.porosity_flat = _flatten_prop('porosity')
        self.volumes = np.full(n, float(self.mesh.dx * self.mesh.dy * self.mesh.dz))

    def _compute_transmissibilities(self):
        """向量化计算传导率矩阵，替代逐单元循环"""
        n = self.total_cells
        self.trans_matrix = np.zeros((6, n))

        area_map = np.array([
            self.mesh.dy * self.mesh.dz,
            self.mesh.dy * self.mesh.dz,
            self.mesh.dx * self.mesh.dz,
            self.mesh.dx * self.mesh.dz,
            self.mesh.dx * self.mesh.dy,
            self.mesh.dx * self.mesh.dy,
        ])
        dist_map = np.array([
            self.mesh.dx, self.mesh.dx,
            self.mesh.dy, self.mesh.dy,
            self.mesh.dz, self.mesh.dz,
        ])

        mu = getattr(self.physics, 'viscosity', 1e-3)

        for d in range(6):
            ni = self.neighbor_indices[d]
            valid = ni >= 0
            if not np.any(valid):
                continue

            ci = np.where(valid)[0]
            cj = ni[ci]

            axis = self._dir_axis[d]
            if axis < 2:
                k_i = self.perm_flat[ci]
                k_j = self.perm_flat[cj]
            else:
                k_i = self.perm_z_flat[ci]
                k_j = self.perm_z_flat[cj]

            k_harm = np.where(
                (k_i + k_j) > 0,
                2.0 * k_i * k_j / (k_i + k_j),
                0.0,
            )

            self.trans_matrix[d, ci] = k_harm * area_map[d] / (mu * dist_map[d])

    def discretize_single_phase(self, dt: float, pressure: np.ndarray,
                                well_manager: WellManager) -> Tuple[csr_matrix, np.ndarray]:
        """向量化组装单相流线性系统"""
        n = self.total_cells

        compressibility = getattr(self.physics, 'compressibility', 1e-9)
        acc_coeff = self.volumes * self.porosity_flat * compressibility / dt
        b = acc_coeff * pressure

        rows_list = []
        cols_list = []
        data_list = []
        diag = acc_coeff.copy()

        for d in range(6):
            ni = self.neighbor_indices[d]
            valid = ni >= 0
            if not np.any(valid):
                continue

            ci = np.where(valid)[0]
            tv = self.trans_matrix[d, ci]
            cj = ni[ci]

            rows_list.append(ci)
            cols_list.append(cj)
            data_list.append(-tv)

            np.add.at(diag, ci, tv)

        rows_list.append(np.arange(n))
        cols_list.append(np.arange(n))
        data_list.append(diag)

        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        data = np.concatenate(data_list)

        A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

        A_lil = A.tolil()
        well_manager.apply_well_terms(A_lil, b, pressure, dt)
        A_csr = A_lil.tocsr()

        return A_csr, b

    def _verify_matrix_properties(self, A: Union[np.ndarray, csr_matrix]) -> Dict[str, Any]:
        if isinstance(A, csr_matrix):
            A_dense = A.toarray()
        else:
            A_dense = A

        properties = {}
        diagonal = np.abs(np.diag(A_dense))
        off_diagonal_sum = np.sum(np.abs(A_dense), axis=1) - diagonal
        diagonal_dominance = np.all(diagonal >= off_diagonal_sum)
        properties['diagonal_dominant'] = diagonal_dominance
        properties['diagonal_dominance_ratio'] = np.min(diagonal / (off_diagonal_sum + 1e-15))

        try:
            cond_num = np.linalg.cond(A_dense)
            properties['condition_number'] = cond_num
            properties['well_conditioned'] = cond_num < 1e12
        except Exception:
            properties['condition_number'] = np.inf
            properties['well_conditioned'] = False

        properties['symmetric'] = np.allclose(A_dense, A_dense.T, rtol=1e-10)
        properties['has_zero_diagonal'] = np.any(np.abs(diagonal) < 1e-15)
        properties['sparsity'] = 1.0 - np.count_nonzero(A_dense) / A_dense.size

        return properties

    def discretize_two_phase(self, dt: float, pressure: np.ndarray,
                             saturation: np.ndarray,
                             well_manager) -> Tuple[csr_matrix, np.ndarray]:
        """向量化组装两相流 IMPES 压力方程线性系统"""
        n = self.total_cells
        physics = self.physics

        Sw = saturation
        S_or = physics.kro_params['S_or']
        S_wr = physics.krw_params['S_wr']
        n_o = physics.kro_params['n_o']
        n_w = physics.krw_params['n_w']

        S_o_norm = np.clip((1.0 - Sw - S_or) / (1.0 - S_wr - S_or), 0.0, 1.0)
        S_w_norm = np.clip((Sw - S_wr) / (1.0 - S_wr - S_or), 0.0, 1.0)

        kro = np.where(Sw <= S_wr, 1.0, np.where(Sw >= 1.0 - S_or, 0.0, S_o_norm ** n_o))
        krw = np.where(Sw <= S_wr, 0.0, np.where(Sw >= 1.0 - S_or, 1.0, S_w_norm ** n_w))

        lambda_w = krw / physics.mu_w
        lambda_o = kro / physics.mu_o
        lambda_t = lambda_w + lambda_o

        compressibility = getattr(physics, 'compressibility', 1e-9)
        acc_coeff = self.volumes * self.porosity_flat * compressibility / dt
        b = acc_coeff * pressure

        rows_list = []
        cols_list = []
        data_list = []
        diag = acc_coeff.copy()

        for d in range(6):
            ni = self.neighbor_indices[d]
            valid = ni >= 0
            if not np.any(valid):
                continue

            ci = np.where(valid)[0]
            tv = self.trans_matrix[d, ci]
            cj = ni[ci]

            lt_i = lambda_t[ci]
            lt_j = lambda_t[cj]
            lt_face = 2.0 * lt_i * lt_j / (lt_i + lt_j + 1e-30)

            T_total = tv * lt_face

            rows_list.append(ci)
            cols_list.append(cj)
            data_list.append(-T_total)

            np.add.at(diag, ci, T_total)

        rows_list.append(np.arange(n))
        cols_list.append(np.arange(n))
        data_list.append(diag)

        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        data = np.concatenate(data_list)

        A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

        A_lil = A.tolil()
        well_manager.apply_well_terms(A_lil, b, pressure, dt)
        A_csr = A_lil.tocsr()

        return A_csr, b

    def __repr__(self):
        return f"FVMDiscretizer({self.mesh.nx}x{self.mesh.ny}x{self.mesh.nz})"


class SinglePhaseFVMDiscretizer(FVMDiscretizer):

    def __init__(self, mesh: StructuredMesh, physics: SinglePhaseProperties):
        super().__init__(mesh, physics)

    def discretize(self, dt: float, pressure: np.ndarray,
                   well_manager: WellManager) -> Tuple[csr_matrix, np.ndarray]:
        return self.discretize_single_phase(dt, pressure, well_manager)


class TwoPhaseFVMDiscretizer(FVMDiscretizer):

    def __init__(self, mesh: StructuredMesh, physics: TwoPhaseProperties):
        super().__init__(mesh, physics)

    def discretize_pressure(self, dt: float, pressure: np.ndarray,
                            saturation: np.ndarray,
                            well_manager: WellManager) -> Tuple[csr_matrix, np.ndarray]:
        return self.discretize_single_phase(dt, pressure, well_manager)

    def discretize_saturation(self, dt: float, pressure_new: np.ndarray,
                              pressure_old: np.ndarray, saturation_old: np.ndarray) -> np.ndarray:
        saturation_new = saturation_old.copy()
        return saturation_new
