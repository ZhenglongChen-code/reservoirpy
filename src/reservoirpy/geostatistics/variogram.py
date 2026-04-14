"""
变差函数建模模块

封装 SciKit-GStat 的变差函数计算和拟合功能，
提供面向油藏模拟的变差函数参数接口。
"""

import numpy as np
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

VARIogram_MODELS = ['spherical', 'exponential', 'gaussian', 'matern', 'stable']


@dataclass
class VariogramParams:
    """
    变差函数参数

    Attributes:
        azimuth: 方位角（度），0 为 x 方向，90 为 y 方向
        nugget: 块金值
        major_range: 主变程（沿 azimuth 方向的相关长度）
        minor_range: 次变程（垂直于 azimuth 方向的相关长度）
        sill: 基台值
        vtype: 变差函数模型类型
    """
    azimuth: float = 0.0
    nugget: float = 0.0
    major_range: float = 100.0
    minor_range: float = 100.0
    sill: float = 1.0
    vtype: str = 'exponential'

    def to_list(self) -> List:
        """转换为 GStatSim 所需的变差函数参数列表"""
        return [self.azimuth, self.nugget, self.major_range,
                self.minor_range, self.sill, self.vtype.capitalize()]

    @classmethod
    def from_list(cls, params: List) -> 'VariogramParams':
        """从 GStatSim 参数列表创建"""
        return cls(
            azimuth=params[0],
            nugget=params[1],
            major_range=params[2],
            minor_range=params[3],
            sill=params[4],
            vtype=params[5].lower() if isinstance(params[5], str) else 'exponential',
        )


class VariogramModel:
    """
    变差函数建模器

    从硬数据计算实验变差函数并拟合理论模型。
    支持各向同性/各向异性分析。

    Args:
        coords: 硬数据坐标，形状 (n, 2)，列为 [x, y]
        values: 硬数据属性值，形状 (n,)

    Examples:
        >>> coords = np.random.rand(50, 2) * 200
        >>> values = np.random.randn(50)
        >>> vm = VariogramModel(coords, values)
        >>> vm.fit(model='exponential')
        >>> print(vm.params)
        >>> vm.plot()
    """

    def __init__(self, coords: np.ndarray, values: np.ndarray):
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must be (n, 2) array, got {coords.shape}")
        if values.ndim != 1:
            raise ValueError(f"values must be 1D array, got {values.ndim}D")
        if coords.shape[0] != values.shape[0]:
            raise ValueError(f"coords and values must have same length, "
                             f"got {coords.shape[0]} and {values.shape[0]}")

        self.coords = coords.astype(np.float64)
        self.values = values.astype(np.float64)
        self._variogram = None
        self.params: Optional[VariogramParams] = None

    def fit(
        self,
        model: str = 'exponential',
        n_lags: int = 20,
        maxlag: Optional[float] = None,
        azimuth: float = 0.0,
        minor_range_ratio: float = 1.0,
    ) -> VariogramParams:
        """
        计算实验变差函数并拟合理论模型

        Args:
            model: 理论模型类型 ('spherical', 'exponential', 'gaussian')
            n_lags: 滞后距离分组数
            maxlag: 最大滞后距离，None 则自动计算
            azimuth: 方位角（度），用于各向异性
            minor_range_ratio: 次/主变程比，1.0 为各向同性

        Returns:
            拟合后的变差函数参数
        """
        import skgstat as skg

        if model.lower() not in VARIogram_MODELS:
            raise ValueError(f"Unknown variogram model: {model}. "
                             f"Supported: {VARIogram_MODELS}")

        if maxlag is None:
            diffs = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=2))
            maxlag = np.median(dists[dists > 0]) * 2

        self._variogram = skg.Variogram(
            coordinates=self.coords,
            values=self.values,
            bin_func='even',
            n_lags=n_lags,
            maxlag=maxlag,
            normalize=False,
        )
        self._variogram.model = model.lower()

        fitted_range = self._variogram.parameters[0]
        fitted_sill = self._variogram.parameters[1]
        fitted_nugget = self._variogram.parameters[2] if len(self._variogram.parameters) > 2 else 0.0

        self.params = VariogramParams(
            azimuth=azimuth,
            nugget=fitted_nugget,
            major_range=fitted_range,
            minor_range=fitted_range * minor_range_ratio,
            sill=fitted_sill,
            vtype=model.lower(),
        )

        logger.info(f"Variogram fitted: range={fitted_range:.2f}, "
                    f"sill={fitted_sill:.4f}, nugget={fitted_nugget:.4f}")
        return self.params

    def set_params(self, params: VariogramParams) -> None:
        """手动设置变差函数参数（跳过拟合）"""
        self.params = params

    def set_params_manual(
        self,
        azimuth: float = 0.0,
        nugget: float = 0.0,
        major_range: float = 100.0,
        minor_range: float = 100.0,
        sill: float = 1.0,
        vtype: str = 'exponential',
    ) -> VariogramParams:
        """手动设置变差函数参数（便捷方法）"""
        self.params = VariogramParams(
            azimuth=azimuth, nugget=nugget,
            major_range=major_range, minor_range=minor_range,
            sill=sill, vtype=vtype,
        )
        return self.params

    def plot(self, figsize: Tuple[int, int] = (8, 5)) -> None:
        """绘制实验变差函数和拟合模型"""
        import matplotlib.pyplot as plt

        if self._variogram is None:
            raise RuntimeError("Must call fit() before plotting")

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self._variogram.plot(axes=ax)
        ax.set_title(f'Variogram ({self.params.vtype})\n'
                     f'Range={self.params.major_range:.1f}, '
                     f'Sill={self.params.sill:.4f}, '
                     f'Nugget={self.params.nugget:.4f}')
        plt.tight_layout()
        plt.show()
