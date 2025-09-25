"""
油藏数值模拟器软件包

一个轻量级、模块化、可扩展的油藏数值模拟器，支持单相流和两相流渗流方程求解。

主要功能：
- 结构化网格生成和管理
- 单相流和两相流物理属性建模
- 有限体积法离散化
- 多种线性求解器支持
- 井模型（Peaceman模型）
- 2D/3D可视化
- 配置驱动的模拟流程

使用示例：
    >>> from reservoirpy import ReservoirSimulator
    >>> simulator = ReservoirSimulator('../../config/default_config.yaml')
    >>> results = simulator.run_simulation()

文档：
- API文档：reservoir_sim/docs/api/API文档.md
- 使用教程：reservoir_sim/docs/tutorials/
- 理论背景：reservoir_sim/docs/theory/
"""

__version__ = "0.1.0"
__author__ = "Reservoir Simulation Team"
__email__ = "reservoir@example.com"

# 导入主要类
from reservoirpy.mesh.mesh import StructuredMesh, CubeCell, Node
from reservoirpy.physics.physics import SinglePhaseProperties, TwoPhaseProperties
from .core.simulator import ReservoirSimulator
from .core.well_model import Well, WellManager, create_well_from_config, validate_well_config
from .core.discretization import FVMDiscretizer

# 向后兼容
MeshGrid = StructuredMesh

__all__ = [
    'FVMDiscretizer',
    'StructuredMesh',
    'CubeCell',
    'Node',
    'SinglePhaseProperties',
    'TwoPhaseProperties',
    'ReservoirSimulator',
    'Well',
    'WellManager',
    'create_well_from_config',
    'validate_well_config',
    'MeshGrid'  # 向后兼容
]