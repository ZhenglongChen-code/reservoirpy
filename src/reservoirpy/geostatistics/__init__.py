"""
地质统计模块

提供变差函数建模、克里金插值和序贯高斯模拟功能，
用于生成非均质渗透率场等油藏属性。

基于 GStatSim 和 SciKit-GStat 封装。
"""

from reservoirpy.geostatistics.variogram import VariogramModel
from reservoirpy.geostatistics.kriging import KrigingEstimator
from reservoirpy.geostatistics.sgsim import SGSimulator
from reservoirpy.geostatistics.perm_generator import PermeabilityGenerator

__all__ = [
    'VariogramModel',
    'KrigingEstimator',
    'SGSimulator',
    'PermeabilityGenerator',
]
