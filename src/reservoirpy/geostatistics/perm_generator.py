"""
渗透率场生成器

面向油藏模拟的高层接口，自动处理正态分数变换和反变换，
生成可直接传入 PropertyManager 的渗透率场。
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, Union
from reservoirpy.geostatistics.variogram import VariogramModel, VariogramParams
from reservoirpy.geostatistics.sgsim import SGSimulator

logger = logging.getLogger(__name__)


class PermeabilityGenerator:
    """
    渗透率场生成器

    自动完成 log 变换 → 正态分数变换 → SGSIM → 反变换 的完整流程，
    生成与 StructuredMesh 兼容的渗透率场。

    渗透率通常服从对数正态分布，SGSIM 要求高斯分布，
    因此需要：
    1. log10(K) 变换
    2. 正态分数变换（QuantileTransformer）
    3. SGSIM 生成高斯随机场
    4. 反正态分数变换
    5. 10^(反变换) 得到渗透率场

    Args:
        nx: x 方向网格数
        ny: y 方向网格数
        nz: z 方向网格数（默认 1，即 2D）
        dx: x 方向网格间距 (m)
        dy: y 方向网格间距 (m)
        dz: z 方向网格间距 (m)

    Examples:
        >>> gen = PermeabilityGenerator(nx=20, ny=20, dx=10, dy=10)
        >>> perm_field = gen.generate(
        ...     hard_data=well_observations,
        ...     major_range=50, minor_range=30,
        ...     sill=0.8, vtype='exponential',
        ...     n_realizations=5, seed=42,
        ... )
    """

    def __init__(
        self,
        nx: int = 20,
        ny: int = 20,
        nz: int = 1,
        dx: float = 10.0,
        dy: float = 10.0,
        dz: float = 10.0,
    ):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self._pred_grid = self._build_prediction_grid()
        self._nst_trans = None

    def _build_prediction_grid(self) -> np.ndarray:
        """构建目标网格坐标 (m, 2)"""
        x_coords = np.arange(0, self.nx) * self.dx + self.dx / 2
        y_coords = np.arange(0, self.ny) * self.dy + self.dy / 2
        xx, yy = np.meshgrid(x_coords, y_coords)
        grid_2d = np.column_stack([xx.ravel(), yy.ravel()])

        if self.nz > 1:
            z_coords = np.arange(0, self.nz) * self.dz + self.dz / 2
            grids = []
            for iz in range(self.nz):
                layer = grid_2d.copy()
                grids.append(layer)
            return np.vstack(grids)
        return grid_2d

    def _transform_hard_data(
        self,
        hard_data: np.ndarray,
    ) -> np.ndarray:
        """
        对硬数据进行 log + 正态分数变换

        Args:
            hard_data: 形状 (n, 3)，列为 [x, y, perm_mD]

        Returns:
            变换后数据，形状 (n, 3)，列为 [x, y, nst_value]
        """
        from sklearn.preprocessing import QuantileTransformer

        perm_values = hard_data[:, 2].copy()
        log_perm = np.log10(perm_values)

        self._nst_trans = QuantileTransformer(
            n_quantiles=min(500, len(log_perm)),
            output_distribution="normal",
        )
        nst_values = self._nst_trans.fit_transform(log_perm.reshape(-1, 1)).flatten()

        transformed = hard_data.copy()
        transformed[:, 2] = nst_values
        return transformed

    def _back_transform(
        self,
        nst_values: np.ndarray,
    ) -> np.ndarray:
        """
        反正态分数变换 + 10^x 得到渗透率 (mD)

        Args:
            nst_values: 正态分数变换后的值

        Returns:
            渗透率值 (mD)
        """
        if self._nst_trans is not None:
            log_perm = self._nst_trans.inverse_transform(
                nst_values.reshape(-1, 1)
            ).flatten()
        else:
            log_perm = nst_values

        perm_mD = 10.0 ** log_perm
        return perm_mD

    def generate(
        self,
        hard_data: Optional[np.ndarray] = None,
        major_range: float = 100.0,
        minor_range: float = 100.0,
        azimuth: float = 0.0,
        sill: float = 1.0,
        nugget: float = 0.0,
        vtype: str = 'exponential',
        k: int = 50,
        search_radius: float = 1e10,
        n_realizations: int = 1,
        seed: Optional[int] = None,
        method: str = 'ordinary',
        mean_log_perm: float = 2.0,
        std_log_perm: float = 0.5,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        生成渗透率场

        Args:
            hard_data: 硬数据（井位观测），形状 (n, 3)，列为 [x, y, perm_mD]。
                       None 表示非条件模拟。
            major_range: 主变程 (m)
            minor_range: 次变程 (m)
            azimuth: 方位角 (度)
            sill: 基台值
            nugget: 块金值
            vtype: 变差函数模型类型
            k: 邻近数据点数
            search_radius: 搜索半径
            n_realizations: 生成实现数
            seed: 随机种子
            method: 克里金方法 ('simple' 或 'ordinary')
            mean_log_perm: log10(K) 均值（非条件模拟时使用）
            std_log_perm: log10(K) 标准差（非条件模拟时使用）

        Returns:
            n_realizations=1 时返回 (nz, ny, nx) 渗透率数组 (mD)
            n_realizations>1 时返回字典 {'perm_fields': (n_realizations, nz, ny, nx)}
        """
        if hard_data is not None and len(hard_data) >= 10:
            transformed_data = self._transform_hard_data(hard_data)
            vario_model = VariogramModel(
                transformed_data[:, :2], transformed_data[:, 2])
            vario_model.fit(
                model=vtype,
                maxlag=max(major_range, minor_range) * 3,
            )
            params = vario_model.params
            params.azimuth = azimuth
            params.major_range = major_range
            params.minor_range = minor_range
            params.nugget = nugget

            sgs = SGSimulator(params, k=k, search_radius=search_radius, seed=seed)
            realizations = sgs.simulate(
                self._pred_grid, transformed_data,
                n_realizations=n_realizations, method=method,
            )
        elif hard_data is not None and len(hard_data) < 10:
            transformed_data = self._transform_hard_data(hard_data)
            params = VariogramParams(
                azimuth=azimuth, nugget=nugget,
                major_range=major_range, minor_range=minor_range,
                sill=sill, vtype=vtype,
            )
            sgs = SGSimulator(params, k=k, search_radius=search_radius, seed=seed)
            realizations = sgs.simulate(
                self._pred_grid, transformed_data,
                n_realizations=n_realizations, method='simple',
            )
        else:
            params = VariogramParams(
                azimuth=azimuth, nugget=nugget,
                major_range=major_range, minor_range=minor_range,
                sill=sill, vtype=vtype,
            )
            sgs = SGSimulator(params, k=k, search_radius=search_radius, seed=seed)
            realizations = sgs.simulate(
                self._pred_grid, hard_data=None,
                n_realizations=n_realizations, method=method,
            )

        n_cells = self.nx * self.ny
        perm_fields = np.zeros((n_realizations, self.nz, self.ny, self.nx))

        for i in range(n_realizations):
            for iz in range(self.nz):
                start = iz * n_cells
                end = (iz + 1) * n_cells
                layer_nst = realizations[i, start:end]

                if hard_data is not None:
                    layer_perm = self._back_transform(layer_nst)
                else:
                    layer_perm = 10.0 ** (
                        mean_log_perm + std_log_perm * layer_nst
                    )

                perm_fields[i, iz] = layer_perm.reshape(self.ny, self.nx)

            logger.info(f"Realization {i+1}: perm range "
                        f"[{perm_fields[i].min():.2f}, {perm_fields[i].max():.0f}] mD, "
                        f"geometric mean = "
                        f"{np.exp(np.mean(np.log(perm_fields[i]))):.1f} mD")

        if n_realizations == 1:
            return perm_fields[0]
        return {'perm_fields': perm_fields}

    def generate_from_config(
        self,
        config: Dict[str, Any],
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        从配置字典生成渗透率场

        Args:
            config: 配置字典，包含:
                - hard_data: 可选，硬数据数组
                - variogram: 变差函数参数
                - simulation: 模拟参数

        Returns:
            渗透率场
        """
        vario_cfg = config.get('variogram', {})
        sim_cfg = config.get('simulation', {})

        return self.generate(
            hard_data=config.get('hard_data'),
            major_range=vario_cfg.get('major_range', 100.0),
            minor_range=vario_cfg.get('minor_range', 100.0),
            azimuth=vario_cfg.get('azimuth', 0.0),
            sill=vario_cfg.get('sill', 1.0),
            nugget=vario_cfg.get('nugget', 0.0),
            vtype=vario_cfg.get('vtype', 'exponential'),
            k=sim_cfg.get('k', 50),
            search_radius=sim_cfg.get('search_radius', 1e10),
            n_realizations=sim_cfg.get('n_realizations', 1),
            seed=sim_cfg.get('seed'),
            method=sim_cfg.get('method', 'ordinary'),
            mean_log_perm=sim_cfg.get('mean_log_perm', 2.0),
            std_log_perm=sim_cfg.get('std_log_perm', 0.5),
        )
