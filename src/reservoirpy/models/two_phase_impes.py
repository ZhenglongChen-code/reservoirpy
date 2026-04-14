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

    实现隐式压力-显式饱和度方法求解两相流问题
    """

    def __init__(self, mesh, physics, config: Dict[str, Any]):
        super().__init__(mesh, physics, config)

        self.discretizer = FVMDiscretizer(mesh, physics)
        solver_config = config.get('linear_solver', {})
        self.solver = LinearSolver(solver_config)

        self.initial_saturation = config.get('initial_saturation', 0.2)

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
        A, b = self.discretizer.discretize_single_phase(dt, pressure, well_manager)
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
        saturation_new = saturation_old.copy()

        for i in range(self.mesh.n_cells):
            dS_dt = self._compute_saturation_rate(
                pressure_old[i], pressure_new[i],
                saturation_old[i])

            saturation_new[i] = saturation_old[i] + dS_dt * dt
            saturation_new[i] = np.clip(saturation_new[i], 0.0, 1.0)

        return saturation_new

    def _compute_saturation_rate(self, pressure_old: float,
                               pressure_new: float, saturation: float) -> float:
        dp = pressure_new - pressure_old

        kro = self.physics.get_relative_permeability(saturation, 'oil')
        krw = self.physics.get_relative_permeability(saturation, 'water')

        lambda_o = kro / self.physics.mu_o
        lambda_w = krw / self.physics.mu_w
        lambda_t = lambda_o + lambda_w

        if lambda_t > 0:
            porosity = self.physics.property_manager.get_cell_property(0, 'porosity')
            dS_dt = 0.01 * dp / (lambda_t * porosity)
        else:
            dS_dt = 0.0

        return dS_dt

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
            saturation = np.clip(saturation, 0.0, 1.0)
            state_vars['saturation'] = saturation

        return True

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            'physics_type': 'two_phase_impes',
            'equations': ['pressure', 'saturation'],
            'discretization': 'FVM',
            'time_integration': 'implicit_pressure_explicit_saturation',
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
