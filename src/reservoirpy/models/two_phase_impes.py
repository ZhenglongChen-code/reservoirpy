"""
两相流IMPES模型

实现隐式压力-显式饱和度方法求解两相流问题
"""

import numpy as np
from typing import Dict, Any, Tuple, List
import logging
from scipy.sparse import csr_matrix

from .base_model import BaseModel
from ..core.discretization import FVMDiscretizer
from ..core.linear_solver import LinearSolver
from ..core.well_model import WellManager

logger = logging.getLogger(__name__)


class TwoPhaseIMPES(BaseModel):
    """
    两相流IMPES模型

    实现隐式压力-显式饱和度方法:
    1. 隐式求解压力方程（总流度形式）
    2. 显式更新水相饱和度（分流方程+上游权重）

    Attributes:
        discretizer: FVM离散化器
        solver: 线性求解器
        initial_saturation: 初始水相饱和度

    Example:
        >>> model = TwoPhaseIMPES(mesh, physics, config)
        >>> state = model.initialize_state({'initial_pressure': 30e6, 'initial_saturation': 0.2})
        >>> new_state = model.solve_timestep(dt, state, well_manager)
    """

    def __init__(self, mesh, physics, config: Dict[str, Any]):
        super().__init__(mesh, physics, config)

        self.discretizer = FVMDiscretizer(mesh, physics)
        solver_config = config.get('linear_solver', {})
        if 'method' not in solver_config:
            solver_config['method'] = 'direct'
        self.solver = LinearSolver(solver_config)

        self.initial_saturation = config.get('initial_saturation', 0.2)
        self.cfl_factor = config.get('cfl_factor', 0.8)

        logger.info(f"Initialized TwoPhaseIMPES: {self.mesh.nx}x{self.mesh.ny}x{self.mesh.nz}")

    def get_state_variables(self) -> List[str]:
        return ['pressure', 'saturation']

    def initialize_state(self, config: Dict[str, Any]) -> Dict[str, np.ndarray]:
        state_vars = super().initialize_state(config)

        if 'saturation' not in state_vars:
            initial_saturation = config.get('initial_saturation', self.initial_saturation)
            state_vars['saturation'] = np.full(self.mesh.n_cells, initial_saturation)

        return state_vars

    def assemble_system(self, dt: float, state_vars: Dict[str, np.ndarray],
                       well_manager) -> Tuple[csr_matrix, np.ndarray]:
        pressure = state_vars['pressure']
        saturation = state_vars['saturation']
        A, b = self.discretizer.discretize_two_phase(dt, pressure, saturation, well_manager)
        return A, b

    def solve_timestep(self, dt: float, state_vars: Dict[str, np.ndarray],
                      well_manager) -> Dict[str, np.ndarray]:
        pressure = state_vars['pressure']
        saturation = state_vars['saturation']

        A, b = self.assemble_system(dt, state_vars, well_manager)
        new_pressure = self.solver.solve(A, b)

        new_saturation = self._update_saturation_explicit(
            pressure, new_pressure, saturation, dt, well_manager)

        return {'pressure': new_pressure, 'saturation': new_saturation}

    def _update_saturation_explicit(self, pressure_old: np.ndarray,
                                  pressure_new: np.ndarray,
                                  saturation_old: np.ndarray,
                                  dt: float, well_manager) -> np.ndarray:
        n = self.mesh.n_cells
        physics = self.physics
        mesh = self.mesh

        porosity = np.array([
            physics.property_manager.get_cell_property(i, 'porosity')
            for i in range(n)
        ])
        volumes = np.array([float(mesh.cell_list[i].volume) for i in range(n)])

        Sw = saturation_old
        krw = np.array([physics.get_relative_permeability(s, 'water') for s in Sw])
        kro = np.array([physics.get_relative_permeability(s, 'oil') for s in Sw])
        lambda_w = krw / physics.mu_w
        lambda_o = kro / physics.mu_o
        lambda_t = lambda_w + lambda_o

        dSw = np.zeros(n)

        for direction in range(6):
            trans = self.discretizer.trans_matrix[direction]
            mask = trans != 0.0
            if not np.any(mask):
                continue

            cell_indices = np.where(mask)[0]
            j_cells = np.array([mesh.cell_list[i].neighbors[direction] for i in cell_indices])
            valid = j_cells >= 0
            ci = cell_indices[valid]
            cj = j_cells[valid]

            if len(ci) == 0:
                continue

            lt_i = lambda_t[ci]
            lt_j = lambda_t[cj]
            lt_face = 2.0 * lt_i * lt_j / (lt_i + lt_j + 1e-30)

            dp = pressure_new[cj] - pressure_new[ci]

            upstream_j = dp >= 0
            f_w_up = np.where(upstream_j,
                              lambda_w[cj] / (lt_j + 1e-30),
                              lambda_w[ci] / (lt_i + 1e-30))

            T_w = trans[ci] * lt_face * f_w_up
            flux = T_w * dp

            np.add.at(dSw, ci, flux)

        for well in well_manager.wells:
            z, y, x = well.location
            cell_index = mesh.get_cell_index(z, y, x)
            Sw_cell = saturation_old[cell_index]

            if well.control_type == 'bhp':
                q_total = well.compute_well_term(pressure_new[cell_index])
            else:
                q_total = well.value

            if q_total < 0:
                dSw[cell_index] += abs(q_total)
            else:
                lambda_w_cell = physics.get_relative_permeability(Sw_cell, 'water') / physics.mu_w
                lambda_o_cell = physics.get_relative_permeability(Sw_cell, 'oil') / physics.mu_o
                lambda_t_cell = lambda_w_cell + lambda_o_cell
                f_w = lambda_w_cell / (lambda_t_cell + 1e-30)
                dSw[cell_index] += f_w * q_total

        saturation_new = saturation_old + dt * dSw / (volumes * porosity + 1e-30)
        saturation_new = np.clip(saturation_new, 0.0, 1.0)

        return saturation_new

    def compute_cfl_timestep(self, pressure: np.ndarray,
                            saturation: np.ndarray,
                            well_manager) -> float:
        n = self.mesh.n_cells
        physics = self.physics
        mesh = self.mesh

        porosity = np.array([
            physics.property_manager.get_cell_property(i, 'porosity')
            for i in range(n)
        ])
        volumes = np.array([float(mesh.cell_list[i].volume) for i in range(n)])

        Sw = saturation
        krw = np.array([physics.get_relative_permeability(s, 'water') for s in Sw])
        kro = np.array([physics.get_relative_permeability(s, 'oil') for s in Sw])
        lambda_w = krw / physics.mu_w
        lambda_o = kro / physics.mu_o
        lambda_t = lambda_w + lambda_o

        dt_min = np.inf

        for direction in range(6):
            trans = self.discretizer.trans_matrix[direction]
            mask = trans != 0.0
            if not np.any(mask):
                continue

            cell_indices = np.where(mask)[0]
            j_cells = np.array([mesh.cell_list[i].neighbors[direction] for i in cell_indices])
            valid = j_cells >= 0
            ci = cell_indices[valid]
            cj = j_cells[valid]

            if len(ci) == 0:
                continue

            lt_i = lambda_t[ci]
            lt_j = lambda_t[cj]
            lt_face = 2.0 * lt_i * lt_j / (lt_i + lt_j + 1e-30)

            dp = np.abs(pressure[cj] - pressure[ci])
            v_total = trans[ci] * lt_face * dp

            dSw_fd = 0.01
            Sw_i = Sw[ci]
            krw_p = np.array([physics.get_relative_permeability(s + dSw_fd, 'water') for s in Sw_i])
            kro_p = np.array([physics.get_relative_permeability(s + dSw_fd, 'oil') for s in Sw_i])
            fw_p = (krw_p / physics.mu_w) / (krw_p / physics.mu_w + kro_p / physics.mu_o + 1e-30)

            krw_m = np.array([physics.get_relative_permeability(s - dSw_fd, 'water') for s in Sw_i])
            kro_m = np.array([physics.get_relative_permeability(s - dSw_fd, 'oil') for s in Sw_i])
            fw_m = (krw_m / physics.mu_w) / (krw_m / physics.mu_w + kro_m / physics.mu_o + 1e-30)

            dfw_dSw = (fw_p - fw_m) / (2 * dSw_fd)

            active = v_total * np.abs(dfw_dSw) > 1e-30
            if np.any(active):
                dt_cells = porosity[ci[active]] * volumes[ci[active]] / (
                    v_total[active] * np.abs(dfw_dSw[active]))
                dt_min = min(dt_min, dt_cells.min())

        return dt_min * self.cfl_factor if dt_min < np.inf else np.inf

    def update_properties(self, state_vars: Dict[str, np.ndarray]) -> None:
        pressure = state_vars['pressure']
        saturation = state_vars['saturation']

        for i, cell in enumerate(self.mesh.cell_list):
            cell.press = pressure[i]
            cell.Sw = saturation[i]
            self.physics.compute_2phase_param(cell, saturation[i])

    def validate_solution(self, state_vars: Dict[str, np.ndarray]) -> bool:
        pressure = state_vars['pressure']
        saturation = state_vars['saturation']

        if np.any(np.isnan(pressure)) or np.any(np.isinf(pressure)):
            logger.error("Error: NaN or Inf values in pressure field")
            return False

        if np.any(pressure <= 0):
            logger.error("Error: Non-positive pressure values detected")
            return False

        if np.any(np.isnan(saturation)) or np.any(np.isinf(saturation)):
            logger.error("Error: NaN or Inf values in saturation field")
            return False

        if np.any(saturation < 0) or np.any(saturation > 1):
            logger.warning("Warning: Saturation out of [0, 1] range")
            state_vars['saturation'] = np.clip(saturation, 0.0, 1.0)

        return True

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            'physics_type': 'two_phase_impes',
            'equations': ['pressure', 'saturation'],
            'discretization': 'FVM',
            'time_integration': 'implicit_pressure_explicit_saturation',
            'cfl_factor': self.cfl_factor,
            'linear_solver': self.solver.get_info() if hasattr(self.solver, 'get_info') else 'unknown'
        })
        return info

    def __repr__(self):
        return f"TwoPhaseIMPES({self.mesh.nx}x{self.mesh.ny}x{self.mesh.nz})"


def create_impes_solver(config: Dict[str, Any]) -> TwoPhaseIMPES:
    from reservoirpy.mesh.mesh import StructuredMesh
    from reservoirpy.physics.physics import TwoPhaseProperties

    mesh_config = config['mesh']
    mesh = StructuredMesh(
        nx=mesh_config['nx'], ny=mesh_config['ny'], nz=mesh_config['nz'],
        dx=mesh_config['dx'], dy=mesh_config['dy'], dz=mesh_config['dz']
    )

    physics_config = config['physics']
    physics = TwoPhaseProperties(mesh, physics_config)

    solver_config = config.get('solver', {})
    solver = TwoPhaseIMPES(mesh, physics, solver_config)

    return solver


def run_impes_simulation(config: Dict[str, Any]) -> Dict[str, Any]:
    solver = create_impes_solver(config)
    wells_config = config.get('wells', [])
    results = solver.solve_simulation(wells_config)
    results['solver_info'] = solver.get_info()
    return results
