"""
线性求解器模块

提供多种线性求解器方法，用于求解油藏模拟中的线性系统
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, cg, bicgstab, gmres, lgmres, minres
import warnings


def solve_linear_system(A: csr_matrix, b: np.ndarray, 
                       method: str = 'bicgstab', 
                       tolerance: float = 1e-8,
                       max_iterations: int = 1000,
                       preconditioner: Optional[str] = None) -> np.ndarray:
    """
    求解线性系统 Ax = b
    
    Args:
        A: 系数矩阵 (稀疏矩阵)
        b: 右端向量
        method: 求解方法
        tolerance: 收敛容差
        max_iterations: 最大迭代次数
        preconditioner: 预条件子 ('jacobi', 'ilu', None)
        
    Returns:
        解向量 x
    """
    # 根据方法选择求解器
    if method == 'direct':
        # 直接求解 (适用于小规模问题)
        return spsolve(A, b)
    elif method == 'cg':
        # 共轭梯度法 (适用于对称正定矩阵)
        x, info = cg(A, b, rtol=tolerance, maxiter=max_iterations)
        if info != 0:
            warnings.warn(f"CG solver did not converge: info={info}")
        return x
    elif method == 'bicgstab':
        x, info = bicgstab(A, b, rtol=tolerance, maxiter=max_iterations)
        if info != 0:
            warnings.warn(f"BiCGSTAB solver did not converge: info={info}")
        return x
    elif method == 'gmres':
        # 广义最小残差法
        x, info = gmres(A, b, rtol=tolerance, maxiter=max_iterations)
        if info != 0:
            warnings.warn(f"GMRES solver did not converge: info={info}")
        return x
    elif method == 'lgmres':
        # 重启GMRES
        x, info = lgmres(A, b, rtol=tolerance, maxiter=max_iterations)
        if info != 0:
            warnings.warn(f"LGMRES solver did not converge: info={info}")
        return x
    elif method == 'minres':
        # 最小残差法 (适用于对称矩阵)
        x, info = minres(A, b, rtol=tolerance, maxiter=max_iterations)
        if info != 0:
            warnings.warn(f"MINRES solver did not converge: info={info}")
        return x
    else:
        raise ValueError(f"Unknown solver method: {method}")


def solve_linear_system_with_preconditioner(A: csr_matrix, b: np.ndarray,
                                          method: str = 'bicgstab',
                                          tolerance: float = 1e-8,
                                          max_iterations: int = 1000,
                                          preconditioner: str = 'ilu') -> np.ndarray:
    """
    使用预条件子求解线性系统
    
    Args:
        A: 系数矩阵 (稀疏矩阵)
        b: 右端向量
        method: 求解方法
        tolerance: 收敛容差
        max_iterations: 最大迭代次数
        preconditioner: 预条件子 ('jacobi', 'ilu')
        
    Returns:
        解向量 x
    """
    from scipy.sparse.linalg import spilu, LinearOperator
    
    if preconditioner == 'jacobi':
        # Jacobi预条件子 (对角预条件子)
        M = LinearOperator(A.shape, lambda x: x / A.diagonal())
    elif preconditioner == 'ilu':
        # 不完全LU分解预条件子
        try:
            ilu = spilu(A.tocsc(), fill_factor=10)
            M = LinearOperator(A.shape, ilu.solve)
        except:
            # 如果ILU失败，回退到无预条件子
            warnings.warn("ILU preconditioner failed, using no preconditioner")
            return solve_linear_system(A, b, method, tolerance, max_iterations)
    else:
        raise ValueError(f"Unknown preconditioner: {preconditioner}")
    
    # 根据方法选择求解器
    if method == 'cg':
        x, info = cg(A, b, rtol=tolerance, maxiter=max_iterations, M=M)
        if info != 0:
            warnings.warn(f"CG solver with preconditioner did not converge: info={info}")
        return x
    elif method == 'bicgstab':
        x, info = bicgstab(A, b, rtol=tolerance, maxiter=max_iterations, M=M)
        if info != 0:
            warnings.warn(f"BiCGSTAB solver with preconditioner did not converge: info={info}")
        return x
    elif method == 'gmres':
        x, info = gmres(A, b, rtol=tolerance, maxiter=max_iterations, M=M)
        if info != 0:
            warnings.warn(f"GMRES solver with preconditioner did not converge: info={info}")
        return x
    else:
        raise ValueError(f"Method {method} not supported with preconditioner")


class LinearSolver:
    """
    线性求解器类

    封装多种稀疏线性系统求解方法，提供统一接口。

    支持的求解方法:
        - ``direct``: 直接求解（超级LU分解），适用于小规模问题
        - ``cg``: 共轭梯度法，适用于对称正定矩阵
        - ``bicgstab``: 双共轭梯度稳定法，适用于一般非对称矩阵（默认）
        - ``gmres``: 广义最小残差法
        - ``lgmres``: 重启GMRES，适用于大规模问题
        - ``minres``: 最小残差法，适用于对称矩阵

    支持的预条件子:
        - ``jacobi``: 对角预条件子
        - ``ilu``: 不完全LU分解预条件子

    Attributes:
        method: 求解方法名称
        tolerance: 收敛容差
        max_iterations: 最大迭代次数
        preconditioner: 预条件子类型

    Example:
        >>> solver = LinearSolver({'method': 'bicgstab', 'tolerance': 1e-10})
        >>> x = solver.solve(A, b)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化线性求解器
        
        Args:
            config: 求解器配置字典
        """
        self.config = config or {}
        self.method = self.config.get('method', 'bicgstab')
        self.tolerance = self.config.get('tolerance', 1e-8)
        self.max_iterations = self.config.get('max_iterations', 1000)
        self.preconditioner = self.config.get('preconditioner', None)
    
    def solve(self, A: csr_matrix, b: np.ndarray) -> np.ndarray:
        """
        求解线性系统
        
        Args:
            A: 系数矩阵
            b: 右端向量
            
        Returns:
            解向量
        """
        if self.preconditioner:
            return solve_linear_system_with_preconditioner(
                A, b, self.method, self.tolerance, 
                self.max_iterations, self.preconditioner)
        else:
            return solve_linear_system(
                A, b, self.method, self.tolerance, self.max_iterations)
    
    def update_config(self, config: Dict[str, Any]):
        """
        更新求解器配置
        
        Args:
            config: 新的配置字典
        """
        self.config.update(config)
        self.method = self.config.get('method', self.method)
        self.tolerance = self.config.get('tolerance', self.tolerance)
        self.max_iterations = self.config.get('max_iterations', self.max_iterations)
        self.preconditioner = self.config.get('preconditioner', self.preconditioner)
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取求解器信息
        
        Returns:
            求解器信息字典
        """
        return {
            'method': self.method,
            'tolerance': self.tolerance,
            'max_iterations': self.max_iterations,
            'preconditioner': self.preconditioner
        }
    
    def __repr__(self):
        return f"LinearSolver(method={self.method}, tol={self.tolerance})"
