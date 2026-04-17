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

    def __init__(self, mesh, physics, config: Dict[str, Any]):
        super().__init__(mesh, physics, config)

        self.discretizer = FVMDiscretizer(mesh, physics)
        solver_config = config.get('linear_solver', {})
        if 'method' not in solver_config:
            solver_config['method'] = 'direct'
        self.solver = LinearSolver(solver_config)

        self.initial_saturation = config.get('initial_saturation', 0.2)
        self.cfl_factor = config.get('cfl_factor', 0.8)

        self.porosity_flat = self.discretizer.porosity_flat
        self.volumes = self.discretizer.volumes
        self.mu_ref = getattr(physics, 'viscosity', 1e-3)

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

    def _compute_mobility(self, Sw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        physics = self.physics
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
        f_w = lambda_w / (lambda_t + 1e-30)

        return lambda_w, lambda_o, lambda_t, f_w

    def _update_saturation_explicit(self, pressure_old: np.ndarray,
                                  pressure_new: np.ndarray,
                                  saturation_old: np.ndarray,
                                  dt: float, well_manager) -> np.ndarray:
        n = self.mesh.n_cells
        physics = self.physics

        lambda_w, lambda_o, lambda_t, f_w = self._compute_mobility(saturation_old)
        mobility_scale = self.mu_ref * lambda_t

        dSw = np.zeros(n)

        for d in range(6):
            ni = self.discretizer.neighbor_indices[d]
            valid = ni >= 0
            if not np.any(valid):
                continue

            ci = np.where(valid)[0]
            tv = self.discretizer.trans_matrix[d, ci]
            cj = ni[ci]

            ms_i = mobility_scale[ci]
            ms_j = mobility_scale[cj]
            ms_face = 2.0 * ms_i * ms_j / (ms_i + ms_j + 1e-30)

            dp = pressure_new[cj] - pressure_new[ci]

            upstream_j = dp >= 0
            fw_up = np.where(upstream_j, f_w[cj], f_w[ci])

            T_w = tv * ms_face * fw_up
            flux = T_w * dp

            np.add.at(dSw, ci, flux)

        for well in well_manager.wells:
            z, y, x = well.location
            cell_index = self.mesh.get_cell_index(z, y, x)
            Sw_cell = saturation_old[cell_index]
            ms_cell = mobility_scale[cell_index]

            if well.control_type == 'bhp':
                effective_wi = well.well_index * ms_cell
                q_total = effective_wi * (pressure_new[cell_index] - well.value)
            else:
                q_total = well.value

            if q_total < 0:
                dSw[cell_index] += abs(q_total)
            else:
                fw_cell = f_w[cell_index]
                dSw[cell_index] += fw_cell * q_total

        saturation_new = saturation_old + dt * dSw / (self.volumes * self.porosity_flat + 1e-30)
        saturation_new = np.clip(saturation_new, 0.0, 1.0)

        return saturation_new

    def compute_cfl_timestep(self, pressure: np.ndarray,
                            saturation: np.ndarray,
                            well_manager) -> float:
        physics = self.physics

        lambda_w, lambda_o, lambda_t, f_w = self._compute_mobility(saturation)
        mobility_scale = self.mu_ref * lambda_t

        dt_min = np.inf

        for d in range(6):
            ni = self.discretizer.neighbor_indices[d]
            valid = ni >= 0
            if not np.any(valid):
                continue

            ci = np.where(valid)[0]
            tv = self.discretizer.trans_matrix[d, ci]
            cj = ni[ci]

            ms_i = mobility_scale[ci]
            ms_j = mobility_scale[cj]
            ms_face = 2.0 * ms_i * ms_j / (ms_i + ms_j + 1e-30)

            dp = np.abs(pressure[cj] - pressure[ci])
            v_total = tv * ms_face * dp

            dSw_fd = 0.01
            Sw_i = saturation[ci]
            Sw_p = np.clip(Sw_i + dSw_fd, 0.0, 1.0)
            Sw_m = np.clip(Sw_i - dSw_fd, 0.0, 1.0)

            _, _, lt_p, _ = self._compute_mobility(Sw_p)
            _, _, lt_m, _ = self._compute_mobility(Sw_m)
            lw_p = self._compute_mobility(Sw_p)[0]
            lw_m = self._compute_mobility(Sw_m)[0]

            fw_p = lw_p / (lt_p + 1e-30)
            fw_m = lw_m / (lt_m + 1e-30)
            dfw_dSw = (fw_p - fw_m) / (2 * dSw_fd)

            active = v_total * np.abs(dfw_dSw) > 1e-30
            if np.any(active):
                dt_cells = self.porosity_flat[ci[active]] * self.volumes[ci[active]] / (
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
