"""
数学模型包

包含各种物理模型的求解器实现，采用统一的BaseModel接口。
支持的模型：
- SinglePhaseModel: 单相流模型
- TwoPhaseIMPES: 两相流IMPES模型
- TwoPhaseFIM: 两相流FIM模型

设计特点：
- 统一的抽象接口 (BaseModel)
- 工厂模式创建 (ModelFactory)
- Strategy模式与Simulator集成
- 易于扩展新模型
"""

from .base_model import BaseModel
from .model_factory import ModelFactory

try:
    from .single_phase import SinglePhaseModel
except ImportError:
    SinglePhaseModel = None

try:
    from .two_phase_impes import TwoPhaseIMPES, create_impes_solver, run_impes_simulation
except ImportError:
    TwoPhaseIMPES = None
    create_impes_solver = None
    run_impes_simulation = None

try:
    from .two_phase_fim import TwoPhaseFIM, create_fim_solver, run_fim_simulation
except ImportError:
    TwoPhaseFIM = None
    create_fim_solver = None
    run_fim_simulation = None

from .single_phase_sim import SinglePhaseSolver, create_single_phase_solver

__all__ = [
    'BaseModel',
    'ModelFactory',
    'SinglePhaseModel',
    'TwoPhaseIMPES',
    'TwoPhaseFIM',
    'SinglePhaseSolver',
    'create_single_phase_solver',
    'create_impes_solver',
    'run_impes_simulation',
    'create_fim_solver',
    'run_fim_simulation',
]
