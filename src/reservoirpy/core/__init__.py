"""
核心模块包

包含油藏模拟的核心组件，如离散化、求解器、时间积分等。
"""

from .discretization import FVMDiscretizer
from .linear_solver import LinearSolver
from .nonlinear_solver import NewtonRaphsonSolver
from .simulator import ReservoirSimulator
from .time_integration import ImplicitEulerIntegrator
from .well_model import Well, WellManager

__all__ = [
    'FVMDiscretizer',
    'LinearSolver',
    'NewtonRaphsonSolver',
    'ReservoirSimulator',
    'ImplicitEulerIntegrator',
    'Well',
    'WellManager'
]