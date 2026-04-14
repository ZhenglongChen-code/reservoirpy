"""
两相流FIM模型

实现全隐式方法求解两相流问题
"""

import numpy as np
from typing import Dict, Any, Tuple, List
import logging
from scipy.sparse import csr_matrix, eye

from .base_model import BaseModel
from ..core.discretization import FVMDiscretizer
from ..core.linear_solver import LinearSolver
from ..core.nonlinear_solver import NewtonRaphsonSolver
from ..core.well_model import WellManager

logger = logging.getLogger(__name__)


class TwoPhaseFIM(BaseModel):
    """
    两相流FIM模型

    实现全隐式方法求解两相流问题
    """

    def __init__(self, mesh, physics, config: Dict[str, Any]):
        super().__init__(mesh, physics, config)

        self.discretizer = FVMDiscretizer(mesh, physics)
        solver_config = config.get('linear_solver', {})
        self.solver = LinearSolver(solver_config)
        self.newton_solver = NewtonRaphsonSolver(
            config.get('newton_solver', {}))

        self.initial_saturation = config.get('initial_saturation', 0.2)

        logger.info(f"Initialized TwoPhaseFIM: {self.mesh.nx}x{self.mesh.ny}x{self.mesh.nz}")

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
        raise NotImplementedError(
            "FIM uses Newton-Raphson iteration, not direct assembly. "
            "Use solve_timestep instead.")

    def solve_timestep(self, dt: float, state_vars: Dict[str, np.ndarray],
                      well_manager) -> Dict[str, np.ndarray]:
        pressure = state_vars['pressure']
        saturation = state_vars['saturation']

        x0 = np.concatenate([pressure, saturation])

        def residual_function(x):
            p = x[:self.mesh.n_cells]
            s = x[self.mesh.n_cells:]
            return self._compute_residual(p, s, dt, well_manager)

        def jacobian_function(x):
            p = x[:self.mesh.n_cells]
            s = x[self.mesh.n_cells:]
            return self._compute_jacobian(p, s, dt, well_manager)

        x_new, info = self.newton_solver.solve(x0, residual_function, jacobian_function)

        new_pressure = x_new[:self.mesh.n_cells]
        new_saturation = x_new[self.mesh.n_cells:]

        return {'pressure': new_pressure, 'saturation': new_saturation}

    def _compute_residual(self, pressure: np.ndarray, saturation: np.ndarray,
                         dt: float, well_manager) -> np.ndarray:
        raise NotImplementedError(
            "TwoPhaseFIM residual computation is not yet implemented. "
            "This requires full two-phase flow discretization.")

    def _compute_jacobian(self, pressure: np.ndarray, saturation: np.ndarray,
                         dt: float, well_manager) -> csr_matrix:
        raise NotImplementedError(
            "TwoPhaseFIM Jacobian computation is not yet implemented. "
            "This requires full two-phase flow discretization.")

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
            'physics_type': 'two_phase_fim',
            'equations': ['pressure', 'saturation'],
            'discretization': 'FVM',
            'time_integration': 'fully_implicit',
            'linear_solver': self.solver.get_info() if hasattr(self.solver, 'get_info') else 'unknown',
            'newton_solver': self.newton_solver.get_info() if hasattr(self.newton_solver, 'get_info') else 'unknown'
        })
        return info

    def __repr__(self):
        return f"TwoPhaseFIM({self.mesh.nx}x{self.mesh.ny}x{self.mesh.nz})"


def create_fim_solver(config: Dict[str, Any]) -> TwoPhaseFIM:
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
    solver = TwoPhaseFIM(mesh, physics, solver_config)

    return solver


def run_fim_simulation(config: Dict[str, Any]) -> Dict[str, Any]:
    solver = create_fim_solver(config)
    wells_config = config.get('wells', [])
    results = solver.solve_simulation(wells_config)
    results['solver_info'] = solver.get_info()
    return results
