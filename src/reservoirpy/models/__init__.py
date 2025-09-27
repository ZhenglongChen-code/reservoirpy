"""
数学模型包

包含各种物理模型的求解器实现，采用统一的BaseModel接口。
支持的模型：
- SinglePhaseModel: 单相流模型
- IMPESModel: 两相流IMPES模型
- FIMModel: 两相流FIM模型

设计特点：
- 统一的抽象接口 (BaseModel)
- 工厂模式创建 (ModelFactory)
- Strategy模式与Simulator集成
- 易于扩展新模型
"""

from .base_model import BaseModel
from .model_factory import ModelFactory

# 导入具体模型（可选，主要通过工厂创建）
try:
    from .single_phase import SinglePhaseModel
except ImportError:
    SinglePhaseModel = None

try:
    from .two_phase import IMPESModel, FIMModel
except ImportError:
    IMPESModel = None
    FIMModel = None

# 为了向后兼容，保留原有导入
from .single_phase_sim import SinglePhaseSolver, create_single_phase_solver
from .two_phase_impes import TwoPhaseIMPES, create_impes_solver, run_impes_simulation
from .two_phase_fim import TwoPhaseFIM

__all__ = [
    # 新架构
    'BaseModel',
    'ModelFactory',
    'SinglePhaseModel',
    'IMPESModel', 
    'FIMModel',
    
    # 向后兼容
    'SinglePhaseSolver',
    'create_single_phase_solver',
    'TwoPhaseIMPES',
    'TwoPhaseFIM',
    'create_impes_solver',
    'run_impes_simulation'
]