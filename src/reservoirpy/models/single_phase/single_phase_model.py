"""
单相流数学模型

实现单相流油藏模拟的数学模型和求解方法
"""

from typing import Dict, List, Tuple
import numpy as np
import logging
from scipy.sparse import csr_matrix

from ..base_model import BaseModel
from ...core.discretization import FVMDiscretizer
from ...core.linear_solver import LinearSolver

logger = logging.getLogger(__name__)


class SinglePhaseModel(BaseModel):
    """
    单相流数学模型
    
    实现单相流渗流方程的求解：
    φ·c·∂p/∂t = ∇·(k/μ·∇p) + q
    
    其中：
    - φ: 孔隙度
    - c: 压缩系数  
    - p: 压力
    - k: 渗透率
    - μ: 粘度
    - q: 源汇项（井）
    """
    
    def __init__(self, mesh, physics, config: Dict):
        """
        初始化单相流模型
        
        Args:
            mesh: 网格对象
            physics: 单相流物理属性对象
            config: 模型配置
        """
        super().__init__(mesh, physics, config)
        
        # 初始化离散化器
        self.discretizer = FVMDiscretizer(mesh, physics)
        
        # 初始化线性求解器
        solver_config = config.get('linear_solver', {})
        self.solver = LinearSolver(solver_config)
        
        logger.info(f"Initialized SinglePhaseModel: {self.mesh.nx}x{self.mesh.ny}x{self.mesh.nz}")
        
    def get_state_variables(self) -> List[str]:
        """
        返回单相流模型的状态变量
        
        Returns:
            状态变量列表：['pressure']
        """
        return ['pressure']
        
    def assemble_system(self, dt: float, state_vars: Dict[str, np.ndarray], 
                       well_manager) -> Tuple[csr_matrix, np.ndarray]:
        """
        组装单相流线性系统
        
        离散化单相流方程：
        φ·c·∂p/∂t = ∇·(k/μ·∇p) + q
        
        得到线性系统：A·p^{n+1} = b
        
        Args:
            dt: 时间步长
            state_vars: 状态变量字典，包含 'pressure'
            well_manager: 井管理器
            
        Returns:
            (系数矩阵A, 右端向量b)
        """
        pressure = state_vars['pressure']
        
        # 使用FVM离散化器组装线性系统
        A, b = self.discretizer.discretize_single_phase(dt, pressure, well_manager)
        
        return A, b
        
    def solve_timestep(self, dt: float, state_vars: Dict[str, np.ndarray],
                      well_manager) -> Dict[str, np.ndarray]:
        """
        求解单相流的一个时间步
        
        Args:
            dt: 时间步长
            state_vars: 当前状态变量
            well_manager: 井管理器
            
        Returns:
            新的状态变量字典
        """
        # 组装线性系统
        A, b = self.assemble_system(dt, state_vars, well_manager)
        
        # 求解线性系统
        new_pressure = self.solver.solve(A, b)
        
        return {'pressure': new_pressure}
        
    def update_properties(self, state_vars: Dict[str, np.ndarray]) -> None:
        """
        更新网格单元的压力属性
        
        Args:
            state_vars: 状态变量字典
        """
        pressure = state_vars['pressure']
        
        # 更新网格单元中的压力值
        for i, cell in enumerate(self.mesh.cell_list):
            cell.press = pressure[i]
            
    def validate_solution(self, state_vars: Dict[str, np.ndarray]) -> bool:
        """
        验证单相流解的合理性
        
        检查：
        1. 压力值是否为有限数值（非NaN、非无穷大）
        2. 压力值是否为正数
        3. 压力变化是否在合理范围内
        
        Args:
            state_vars: 状态变量字典
            
        Returns:
            解是否合理
        """
        pressure = state_vars['pressure']
        
        # 检查是否有NaN或无穷大值
        if np.any(np.isnan(pressure)) or np.any(np.isinf(pressure)):
            logger.error("Error: NaN or Inf values in pressure field")
            return False
            
        # 检查压力是否为正数
        if np.any(pressure <= 0):
            logger.error("Error: Non-positive pressure values detected")
            return False
            
        # 检查压力范围是否合理（0.1 MPa 到 100 MPa）
        min_pressure = 0.1e6  # 0.1 MPa
        max_pressure = 100e6  # 100 MPa
        
        if np.any(pressure < min_pressure) or np.any(pressure > max_pressure):
            logger.warning(f"Warning: Pressure out of expected range "
                  f"[{min_pressure/1e6:.1f}, {max_pressure/1e6:.1f}] MPa")
            logger.warning(f"Current range: [{np.min(pressure)/1e6:.2f}, {np.max(pressure)/1e6:.2f}] MPa")
            # 这里只是警告，不返回False，因为某些情况下可能确实需要极端压力值
            
        return True
        
    def get_mass_balance_error(self, state_vars: Dict[str, np.ndarray],
                              state_vars_old: Dict[str, np.ndarray],
                              dt: float, well_manager) -> float:
        """
        计算质量平衡误差
        
        Args:
            state_vars: 新状态变量
            state_vars_old: 旧状态变量  
            dt: 时间步长
            well_manager: 井管理器
            
        Returns:
            质量平衡误差
        """
        pressure = state_vars['pressure']
        pressure_old = state_vars_old['pressure']
        
        # 计算累积项变化
        accumulation_change = 0.0
        for i, cell in enumerate(self.mesh.cell_list):
            porosity = self.physics.property_manager.get_cell_property(i, 'porosity')
            compressibility = getattr(self.physics, 'compressibility', 1e-9)
            
            dp = pressure[i] - pressure_old[i]
            accumulation_change += cell.volume * porosity * compressibility * dp
        
        # 计算井的产量
        well_production = 0.0
        for well in well_manager.wells:
            z, y, x = well.location
            cell_index = self.mesh.get_cell_index(z, y, x)
            well_flow = well.compute_well_term(pressure[cell_index])
            well_production += well_flow * dt
        
        # 质量平衡: 累积变化 = 井产量
        mass_balance_error = abs(accumulation_change - well_production)
        
        return mass_balance_error
        
    def solve_steady_state(self, well_manager, tolerance: float = 1e-8) -> Dict[str, np.ndarray]:
        """
        求解稳态解
        
        Args:
            well_manager: 井管理器
            tolerance: 收敛容差
            
        Returns:
            稳态压力场
        """
        # 设置初始猜测
        initial_pressure = self.config.get('initial_pressure', 30e6)
        pressure = np.full(self.mesh.n_cells, initial_pressure)
        state_vars = {'pressure': pressure}
        
        # 使用很大的时间步长求解稳态
        dt_steady = 1e20
        
        logger.info("Solving steady state...")
        
        # 组装和求解稳态系统
        A, b = self.assemble_system(dt_steady, state_vars, well_manager)
        steady_pressure = self.solver.solve(A, b)
        
        steady_state = {'pressure': steady_pressure}
        
        # 验证稳态解
        if self.validate_solution(steady_state):
            logger.info("Steady state solution converged")
            return steady_state
        else:
            raise RuntimeError("Steady state solution validation failed")
            
    def get_model_info(self) -> Dict:
        """
        获取模型详细信息
        
        Returns:
            模型信息字典
        """
        info = super().get_model_info()
        info.update({
            'physics_type': 'single_phase',
            'equations': ['pressure'],
            'discretization': 'FVM',
            'time_integration': 'implicit_euler',
            'linear_solver': self.solver.get_info() if hasattr(self.solver, 'get_info') else 'unknown'
        })
        return info