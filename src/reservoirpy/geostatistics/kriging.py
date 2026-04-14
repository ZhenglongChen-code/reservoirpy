"""
克里金插值模块

封装 GStatSim 的克里金插值功能，
支持简单克里金和普通克里金。
"""

import numpy as np
import logging
from typing import Optional, Tuple
from reservoirpy.geostatistics.variogram import VariogramParams

logger = logging.getLogger(__name__)


class KrigingEstimator:
    """
    克里金插值器

    基于变差函数参数和硬数据，对目标网格进行克里金插值。

    Args:
        params: 变差函数参数
        k: 用于估计的邻近数据点数
        search_radius: 搜索半径

    Examples:
        >>> est, var = estimator.predict(pred_grid, df_hard_data)
    """

    def __init__(
        self,
        params: VariogramParams,
        k: int = 50,
        search_radius: float = 1e10,
    ):
        self.params = params
        self.k = k
        self.search_radius = search_radius

    def predict(
        self,
        pred_grid: np.ndarray,
        hard_data: np.ndarray,
        method: str = 'ordinary',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行克里金插值

        Args:
            pred_grid: 目标网格坐标，形状 (m, 2)，列为 [x, y]
            hard_data: 硬数据，形状 (n, 3)，列为 [x, y, value]
            method: 克里金方法 ('simple' 或 'ordinary')

        Returns:
            (estimate, variance) 估计值和方差，各形状 (m,)
        """
        import gstatsim as gs
        import pandas as pd

        df = pd.DataFrame(hard_data, columns=['X', 'Y', 'Z'])
        vario = self.params.to_list()

        if method.lower() == 'simple':
            est, var = gs.Interpolation.skrige(
                pred_grid, df, 'X', 'Y', 'Z',
                self.k, vario, self.search_radius,
            )
        elif method.lower() == 'ordinary':
            est, var = gs.Interpolation.okrige(
                pred_grid, df, 'X', 'Y', 'Z',
                self.k, vario, self.search_radius,
            )
        else:
            raise ValueError(f"Unknown kriging method: {method}. "
                             f"Supported: 'simple', 'ordinary'")

        logger.info(f"Kriging ({method}) completed: "
                    f"est range [{np.nanmin(est):.4f}, {np.nanmax(est):.4f}]")
        return est, var
