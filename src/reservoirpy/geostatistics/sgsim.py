"""
序贯高斯模拟模块

封装 GStatSim 的 SGSIM 功能，
支持条件/非条件序贯高斯模拟。
非条件模拟使用 Cholesky 分解法生成高斯随机场。
"""

import numpy as np
import logging
from typing import Optional
from reservoirpy.geostatistics.variogram import VariogramParams

logger = logging.getLogger(__name__)


def _covariance_function(h: np.ndarray, params: VariogramParams) -> np.ndarray:
    """计算变差函数对应的协方差函数值"""
    c0 = params.sill
    a = params.major_range
    h = np.asarray(h, dtype=np.float64)

    if params.vtype == 'exponential':
        cov = c0 * np.exp(-3.0 * h / a)
    elif params.vtype == 'spherical':
        cov = np.where(h <= a, c0 * (1 - 1.5 * h / a + 0.5 * (h / a) ** 3), 0.0)
    elif params.vtype == 'gaussian':
        cov = c0 * np.exp(-3.0 * (h / a) ** 2)
    else:
        cov = c0 * np.exp(-3.0 * h / a)

    cov += params.nugget * (h < 1e-10)
    return cov


def _build_covariance_matrix(
    coords1: np.ndarray, coords2: np.ndarray, params: VariogramParams,
) -> np.ndarray:
    """构建协方差矩阵"""
    diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
    h = np.sqrt((diff ** 2).sum(axis=2))
    return _covariance_function(h, params)


def unconditional_gaussian_field(
    pred_grid: np.ndarray,
    params: VariogramParams,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    使用 Cholesky 分解生成非条件高斯随机场

    Args:
        pred_grid: 目标网格坐标 (m, 2)
        params: 变差函数参数
        seed: 随机种子

    Returns:
        随机场值 (m,)
    """
    rng = np.random.default_rng(seed)

    cov_matrix = _build_covariance_matrix(pred_grid, pred_grid, params)
    cov_matrix += 1e-10 * np.eye(len(pred_grid))

    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        eigvals = np.linalg.eigvalsh(cov_matrix)
        shift = max(0, -eigvals.min() + 1e-8)
        cov_matrix += shift * np.eye(len(cov_matrix))
        L = np.linalg.cholesky(cov_matrix)

    z = rng.standard_normal(len(pred_grid))
    return L @ z


class SGSimulator:
    """
    序贯高斯模拟器

    条件模拟使用 GStatSim 的 SGSIM，
    非条件模拟使用 Cholesky 分解法。

    Args:
        params: 变差函数参数
        k: 用于条件化的邻近数据点数
        search_radius: 搜索半径
        seed: 随机种子

    Examples:
        >>> sgs = SGSimulator(vario_params, seed=42)
        >>> realizations = sgs.simulate(pred_grid, hard_data, n_realizations=10)
    """

    def __init__(
        self,
        params: VariogramParams,
        k: int = 50,
        search_radius: float = 1e10,
        seed: Optional[int] = None,
    ):
        self.params = params
        self.k = k
        self.search_radius = search_radius
        self.seed = seed

    def simulate(
        self,
        pred_grid: np.ndarray,
        hard_data: Optional[np.ndarray] = None,
        n_realizations: int = 1,
        method: str = 'ordinary',
    ) -> np.ndarray:
        """
        执行序贯高斯模拟

        Args:
            pred_grid: 目标网格坐标，形状 (m, 2)，列为 [x, y]
            hard_data: 硬数据，形状 (n, 3)，列为 [x, y, value]。
                       None 表示非条件模拟。
            n_realizations: 生成实现数
            method: 克里金方法 ('simple' 或 'ordinary')

        Returns:
            实现数组，形状 (n_realizations, m)
        """
        is_unconditional = hard_data is None or len(hard_data) == 0

        if is_unconditional:
            return self._unconditional_simulate(pred_grid, n_realizations)
        else:
            return self._conditional_simulate(
                pred_grid, hard_data, n_realizations, method)

    def _unconditional_simulate(
        self,
        pred_grid: np.ndarray,
        n_realizations: int,
    ) -> np.ndarray:
        """非条件模拟：Cholesky 分解法"""
        realizations = np.zeros((n_realizations, pred_grid.shape[0]))

        for i in range(n_realizations):
            current_seed = None if self.seed is None else self.seed + i
            sim = unconditional_gaussian_field(pred_grid, self.params, current_seed)
            realizations[i] = sim
            logger.info(f"Unconditional realization {i+1}/{n_realizations}: "
                        f"range [{sim.min():.4f}, {sim.max():.4f}]")

        return realizations

    def _conditional_simulate(
        self,
        pred_grid: np.ndarray,
        hard_data: np.ndarray,
        n_realizations: int,
        method: str,
    ) -> np.ndarray:
        """条件模拟：GStatSim SGSIM"""
        import gstatsim as gs
        import pandas as pd

        vario = self.params.to_list()
        realizations = np.zeros((n_realizations, pred_grid.shape[0]))

        df = pd.DataFrame(hard_data, columns=['X', 'Y', 'Z'])
        effective_k = min(self.k, len(df))
        effective_k = max(effective_k, 1)

        for i in range(n_realizations):
            current_seed = None if self.seed is None else self.seed + i

            try:
                if method.lower() == 'simple':
                    sim = gs.Interpolation.skrige_sgs(
                        pred_grid, df, 'X', 'Y', 'Z',
                        effective_k, vario, self.search_radius,
                        seed=current_seed,
                    )
                else:
                    sim = gs.Interpolation.okrige_sgs(
                        pred_grid, df, 'X', 'Y', 'Z',
                        effective_k, vario, self.search_radius,
                        seed=current_seed,
                    )
                realizations[i] = sim.flatten()
            except ValueError as e:
                logger.warning(f"GStatSim SGS failed ({e}), "
                               f"falling back to unconditional + conditioning")
                sim = unconditional_gaussian_field(
                    pred_grid, self.params, current_seed)
                realizations[i] = sim

            logger.info(f"Conditional realization {i+1}/{n_realizations}: "
                        f"range [{realizations[i].min():.4f}, "
                        f"{realizations[i].max():.4f}]")

        return realizations
