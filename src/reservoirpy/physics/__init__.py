"""
物理模型包

包含油藏模拟中使用的物理属性和模型。
"""

from .physics import SinglePhaseProperties, TwoPhaseProperties

__all__ = [
    'SinglePhaseProperties',
    'TwoPhaseProperties'
]