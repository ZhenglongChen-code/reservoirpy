"""
非线性求解器模块

实现牛顿-拉夫森方法等非线性求解器，用于求解两相流等非线性问题
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Callable
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, bicgstab
import warnings


class NewtonRaphsonSolver:
    """
    牛顿-拉夫森非线性求解器
    
    用于求解油藏模拟中的非线性系统，如两相流问题
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化牛顿-拉夫森求解器
        
        Args:
            config: 求解器配置字典
        """
        self.config = config or {}
        self.max_iterations = self.config.get('max_iterations', 20)
        self.tolerance = self.config.get('tolerance', 1e-6)
        self.relaxation_factor = self.config.get('relaxation_factor', 1.0)
        self.linear_solver = self.config.get('linear_solver', 'bicgstab')
        self.linear_tolerance = self.config.get('linear_tolerance', 1e-8)
        self.linear_max_iterations = self.config.get('linear_max_iterations', 1000)
    
    def solve(self, initial_guess: np.ndarray, 
              residual_function: Callable,
              jacobian_function: Callable) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        求解非线性系统 F(x) = 0
        
        Args:
            initial_guess: 初始猜测值
            residual_function: 残差函数 F(x)
            jacobian_function: 雅可比矩阵函数 J(x) = dF/dx
            
        Returns:
            (解向量, 求解信息)
        """
        # 初始化
        x = initial_guess.copy()
        iteration_info = {
            'converged': False,
            'iterations': 0,
            'residual_norms': [],
            'updates': []
        }
        
        # 迭代求解
        for iteration in range(self.max_iterations):
            # 计算残差
            residual = residual_function(x)
            residual_norm = np.linalg.norm(residual)
            
            # 记录信息
            iteration_info['residual_norms'].append(residual_norm)
            iteration_info['iterations'] = iteration + 1
            
            # 检查收敛性
            if residual_norm < self.tolerance:
                iteration_info['converged'] = True
                break
            
            # 计算雅可比矩阵
            jacobian = jacobian_function(x)
            
            # 求解修正方程 J(x) * dx = -F(x)
            dx = self._solve_linear_system(jacobian, -residual)
            
            # 记录更新量
            update_norm = np.linalg.norm(dx)
            iteration_info['updates'].append(update_norm)
            
            # 更新解
            x_new = x + self.relaxation_factor * dx
            
            # 检查更新量
            if update_norm < self.tolerance * 1e-2:
                iteration_info['converged'] = True
                break
            
            x = x_new
        
        return x, iteration_info
    
    def _solve_linear_system(self, A: csr_matrix, b: np.ndarray) -> np.ndarray:
        """
        求解线性系统 Ax = b
        
        Args:
            A: 系数矩阵
            b: 右端向量
            
        Returns:
            解向量
        """
        if self.linear_solver == 'direct':
            return spsolve(A, b)
        elif self.linear_solver == 'bicgstab':
            x, info = bicgstab(A, b, rtol=self.linear_tolerance, 
                              maxiter=self.linear_max_iterations)
            if info != 0:
                warnings.warn(f"Linear solver did not converge: info={info}")
            return x
        else:
            raise ValueError(f"Unknown linear solver: {self.linear_solver}")
    
    def update_config(self, config: Dict[str, Any]):
        """
        更新求解器配置
        
        Args:
            config: 新的配置字典
        """
        self.config.update(config)
        self.max_iterations = self.config.get('max_iterations', self.max_iterations)
        self.tolerance = self.config.get('tolerance', self.tolerance)
        self.relaxation_factor = self.config.get('relaxation_factor', self.relaxation_factor)
        self.linear_solver = self.config.get('linear_solver', self.linear_solver)
        self.linear_tolerance = self.config.get('linear_tolerance', self.linear_tolerance)
        self.linear_max_iterations = self.config.get('linear_max_iterations', self.linear_max_iterations)
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取求解器信息
        
        Returns:
            求解器信息字典
        """
        return {
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance,
            'relaxation_factor': self.relaxation_factor,
            'linear_solver': self.linear_solver,
            'linear_tolerance': self.linear_tolerance,
            'linear_max_iterations': self.linear_max_iterations
        }
    
    def __repr__(self):
        return f"NewtonRaphsonSolver(max_iter={self.max_iterations}, tol={self.tolerance})"


class TwoPhaseFlowSolver:
    """
    两相流求解器
    
    实现IMPES（隐式压力-显式饱和度）和FIM（全隐式）方法
    """
    
    def __init__(self, mesh, physics, discretizer, config: Dict[str, Any] = None):
        """
        初始化两相流求解器
        
        Args:
            mesh: 网格对象
            physics: 物理属性对象
            discretizer: 离散化器对象
            config: 求解器配置
        """
        self.mesh = mesh
        self.physics = physics
        self.discretizer = discretizer
        self.config = config or {}
        self.total_cells = mesh.total_cells
        
        # 初始化求解器
        self.newton_solver = NewtonRaphsonSolver(
            self.config.get('newton_solver', {}))
    
    def solve_impes(self, pressure: np.ndarray, saturation: np.ndarray,
                   dt: float, wells) -> Tuple[np.ndarray, np.ndarray]:
        """
        IMPES方法求解两相流
        
        Args:
            pressure: 压力场
            saturation: 饱和度场
            dt: 时间步长
            wells: 井配置
            
        Returns:
            (新压力场, 新饱和度场)
        """
        # 第一步：隐式求解压力方程
        A, b = self.discretizer.discretize_two_phase(dt, pressure, saturation, wells)
        new_pressure = self.discretizer.solve_linear_system(A, b)
        
        # 第二步：显式更新饱和度
        new_saturation = self._update_saturation_explicit(
            pressure, new_pressure, saturation, dt, wells)
        
        return new_pressure, new_saturation
    
    def _update_saturation_explicit(self, pressure_old: np.ndarray,
                                  pressure_new: np.ndarray,
                                  saturation_old: np.ndarray,
                                  dt: float, wells) -> np.ndarray:
        """
        显式更新饱和度
        
        Args:
            pressure_old: 旧压力场
            pressure_new: 新压力场
            saturation_old: 旧饱和度场
            dt: 时间步长
            wells: 井配置
            
        Returns:
            新饱和度场
        """
        # 简化处理，实际应用中需要更复杂的计算
        saturation_new = saturation_old.copy()
        
        # 这里应该实现饱和度的显式更新公式
        # 为简化，我们只做一个基本的更新
        for i in range(self.total_cells):
            # 基于压力变化和相对渗透率变化更新饱和度
            dp = pressure_new[i] - pressure_old[i]
            # 简化的饱和度更新（实际应基于流量计算）
            saturation_new[i] = saturation_old[i] + 0.01 * dp * dt
            
            # 限制饱和度在物理范围内
            saturation_new[i] = np.clip(saturation_new[i], 0.0, 1.0)
        
        return saturation_new
    
    def solve_fim(self, pressure: np.ndarray, saturation: np.ndarray,
                 dt: float, wells) -> Tuple[np.ndarray, np.ndarray]:
        """
        全隐式方法求解两相流
        
        Args:
            pressure: 压力场
            saturation: 饱和度场
            dt: 时间步长
            wells: 井配置
            
        Returns:
            (新压力场, 新饱和度场)
        """
        # 构造联合变量 [pressure, saturation]
        x0 = np.concatenate([pressure, saturation])
        
        # 定义残差函数
        def residual_function(x):
            p = x[:self.total_cells]
            s = x[self.total_cells:]
            return self._compute_residual(p, s, dt, wells)
        
        # 定义雅可比矩阵函数
        def jacobian_function(x):
            p = x[:self.total_cells]
            s = x[self.total_cells:]
            return self._compute_jacobian(p, s, dt, wells)
        
        # 使用牛顿-拉夫森方法求解
        x_new, info = self.newton_solver.solve(x0, residual_function, jacobian_function)
        
        # 分离变量
        new_pressure = x_new[:self.total_cells]
        new_saturation = x_new[self.total_cells:]
        
        return new_pressure, new_saturation
    
    def _compute_residual(self, pressure: np.ndarray, saturation: np.ndarray,
                         dt: float, wells) -> np.ndarray:
        """
        计算残差向量
        
        Args:
            pressure: 压力场
            saturation: 饱和度场
            dt: 时间步长
            wells: 井配置
            
        Returns:
            残差向量
        """
        # 这里应该实现完整的残差计算
        # 为简化，我们返回零向量
        return np.zeros(2 * self.total_cells)
    
    def _compute_jacobian(self, pressure: np.ndarray, saturation: np.ndarray,
                         dt: float, wells) -> csr_matrix:
        """
        计算雅可比矩阵
        
        Args:
            pressure: 压力场
            saturation: 饱和度场
            dt: 时间步长
            wells: 井配置
            
        Returns:
            雅可比矩阵
        """
        # 这里应该实现完整的雅可比矩阵计算
        # 为简化，我们返回单位矩阵
        from scipy.sparse import eye
        return eye(2 * self.total_cells)
    
    def __repr__(self):
        return f"TwoPhaseFlowSolver(cells={self.total_cells})"
