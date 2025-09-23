"""
3D可视化模块

提供3D压力场、饱和度场等的可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Any, List, Optional
import os


class Plot3D:
    """
    3D可视化类
    
    提供3D压力场、饱和度场等的可视化功能
    """
    
    def __init__(self, mesh, config: Dict[str, Any] = None):
        """
        初始化3D可视化器
        
        Args:
            mesh: 网格对象
            config: 可视化配置
        """
        self.mesh = mesh
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', 'results')
        self.dpi = self.config.get('dpi', 150)
        self.figsize = self.config.get('figsize', (12, 10))
        
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def plot_pressure_volume(self, pressure: np.ndarray, 
                           title: str = "3D Pressure Field",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制3D压力场
        
        Args:
            pressure: 压力场数据
            title: 图标题
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        # 重塑压力场为3D网格
        pressure_3d = pressure.reshape(self.mesh.nz, self.mesh.ny, self.mesh.nx)
        
        # 创建网格坐标
        x = np.linspace(0, self.mesh.nx*self.mesh.dx, self.mesh.nx)
        y = np.linspace(0, self.mesh.ny*self.mesh.dy, self.mesh.ny)
        z = np.linspace(0, self.mesh.nz*self.mesh.dz, self.mesh.nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 创建图形
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制体积渲染
        scatter = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), 
                           c=pressure_3d.flatten(), cmap='viridis', 
                           alpha=0.6, s=20)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Pressure (Pa)', rotation=270, labelpad=20)
        
        # 设置标签和标题
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        
        # 保存图形
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), 
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_isosurface(self, field: np.ndarray, 
                       isovalue: float,
                       title: str = "Isosurface",
                       field_name: str = "Field",
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制等值面
        
        Args:
            field: 场数据
            isovalue: 等值
            title: 图标题
            field_name: 场名称
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        # 重塑场数据为3D网格
        field_3d = field.reshape(self.mesh.nz, self.mesh.ny, self.mesh.nx)
        
        # 创建网格坐标
        x = np.linspace(0, self.mesh.nx*self.mesh.dx, self.mesh.nx)
        y = np.linspace(0, self.mesh.ny*self.mesh.dy, self.mesh.ny)
        z = np.linspace(0, self.mesh.nz*self.mesh.dz, self.mesh.nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 创建图形
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制等值面
        # 注意：matplotlib的等值面绘制功能有限，这里使用散点图近似
        mask = np.abs(field_3d - isovalue) < (field_3d.max() - field_3d.min()) * 0.05
        if np.any(mask):
            ax.scatter(X[mask], Y[mask], Z[mask], 
                      c=field_3d[mask], cmap='viridis', 
                      alpha=0.7, s=30)
        
        # 设置标签和标题
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f"{title} (Isovalue: {isovalue:.2e})")
        
        # 保存图形
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), 
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_slice(self, field: np.ndarray, 
                  slice_index: int = None,
                  direction: str = 'z',
                  title: str = "Slice Plot",
                  field_name: str = "Field",
                  save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制切片图
        
        Args:
            field: 场数据
            slice_index: 切片索引
            direction: 切片方向 ('x', 'y', 'z')
            title: 图标题
            field_name: 场名称
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        # 重塑场数据为3D网格
        field_3d = field.reshape(self.mesh.nz, self.mesh.ny, self.mesh.nx)
        
        # 确定切片索引
        if slice_index is None:
            if direction == 'x':
                slice_index = self.mesh.nx // 2
            elif direction == 'y':
                slice_index = self.mesh.ny // 2
            else:  # z
                slice_index = self.mesh.nz // 2
        
        # 提取切片
        if direction == 'x':
            slice_data = field_3d[:, :, slice_index]
            extent = [0, self.mesh.ny*self.mesh.dy, 0, self.mesh.nz*self.mesh.dz]
            xlabel, ylabel = 'Y (m)', 'Z (m)'
        elif direction == 'y':
            slice_data = field_3d[:, slice_index, :]
            extent = [0, self.mesh.nx*self.mesh.dx, 0, self.mesh.nz*self.mesh.dz]
            xlabel, ylabel = 'X (m)', 'Z (m)'
        else:  # z
            slice_data = field_3d[slice_index, :, :]
            extent = [0, self.mesh.nx*self.mesh.dx, 0, self.mesh.ny*self.mesh.dy]
            xlabel, ylabel = 'X (m)', 'Y (m)'
        
        # 创建图形
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 绘制切片
        im = ax.imshow(slice_data, cmap='viridis', origin='lower', 
                      extent=extent)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(field_name, rotation=270, labelpad=20)
        
        # 设置标签和标题
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} ({direction.upper()} slice at index {slice_index})")
        
        # 保存图形
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), 
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_3d_scatter(self, field: np.ndarray, 
                       sample_rate: float = 0.1,
                       title: str = "3D Scatter Plot",
                       field_name: str = "Field",
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制3D散点图
        
        Args:
            field: 场数据
            sample_rate: 采样率
            title: 图标题
            field_name: 场名称
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        # 重塑场数据为3D网格
        field_3d = field.reshape(self.mesh.nz, self.mesh.ny, self.mesh.nx)
        
        # 创建网格坐标
        x = np.linspace(0, self.mesh.nx*self.mesh.dx, self.mesh.nx)
        y = np.linspace(0, self.mesh.ny*self.mesh.dy, self.mesh.ny)
        z = np.linspace(0, self.mesh.nz*self.mesh.dz, self.mesh.nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 采样数据点
        total_points = X.size
        sample_points = int(total_points * sample_rate)
        indices = np.random.choice(total_points, sample_points, replace=False)
        
        # 创建图形
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制散点图
        scatter = ax.scatter(X.flatten()[indices], Y.flatten()[indices], Z.flatten()[indices], 
                           c=field_3d.flatten()[indices], cmap='viridis', 
                           alpha=0.6, s=20)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label(field_name, rotation=270, labelpad=20)
        
        # 设置标签和标题
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        
        # 保存图形
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), 
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_vector_field(self, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray = None,
                         sample_rate: float = 0.2,
                         title: str = "3D Vector Field",
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制3D矢量场
        
        Args:
            vx: X方向矢量分量
            vy: Y方向矢量分量
            vz: Z方向矢量分量
            sample_rate: 采样率
            title: 图标题
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        # 重塑矢量场为3D网格
        vx_3d = vx.reshape(self.mesh.nz, self.mesh.ny, self.mesh.nx)
        vy_3d = vy.reshape(self.mesh.nz, self.mesh.ny, self.mesh.nx)
        if vz is not None:
            vz_3d = vz.reshape(self.mesh.nz, self.mesh.ny, self.mesh.nx)
        else:
            vz_3d = np.zeros_like(vx_3d)
        
        # 创建网格坐标
        x = np.linspace(0, self.mesh.nx*self.mesh.dx, self.mesh.nx)
        y = np.linspace(0, self.mesh.ny*self.mesh.dy, self.mesh.ny)
        z = np.linspace(0, self.mesh.nz*self.mesh.dz, self.mesh.nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 采样数据点
        total_points = X.size
        sample_points = int(total_points * sample_rate)
        indices = np.random.choice(total_points, sample_points, replace=False)
        
        # 创建图形
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制矢量场
        ax.quiver(X.flatten()[indices], Y.flatten()[indices], Z.flatten()[indices],
                 vx_3d.flatten()[indices], vy_3d.flatten()[indices], vz_3d.flatten()[indices],
                 length=0.1*np.max([self.mesh.dx, self.mesh.dy, self.mesh.dz]),
                 normalize=True, alpha=0.7)
        
        # 设置标签和标题
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        
        # 保存图形
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), 
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def __repr__(self):
        return f"Plot3D(mesh={self.mesh.nx}x{self.mesh.ny}x{self.mesh.nz})"


def create_3d_plotter(mesh, config: Dict[str, Any] = None) -> Plot3D:
    """
    创建3D可视化器
    
    Args:
        mesh: 网格对象
        config: 可视化配置
        
    Returns:
        Plot3D实例
    """
    return Plot3D(mesh, config)


def plot_3d_simulation_results(mesh, results: Dict[str, Any], 
                             config: Dict[str, Any] = None) -> List[plt.Figure]:
    """
    绘制3D模拟结果
    
    Args:
        mesh: 网格对象
        results: 模拟结果
        config: 可视化配置
        
    Returns:
        matplotlib图形对象列表
    """
    # 创建可视化器
    plotter = create_3d_plotter(mesh, config)
    
    figures = []
    
    # 绘制3D压力场
    if 'pressure_history' in results and results['pressure_history']:
        final_pressure = results['pressure_history'][-1]
        fig = plotter.plot_pressure_volume(
            final_pressure,
            title="3D Pressure Field",
            save_path="3d_pressure_field.png"
        )
        figures.append(fig)
    
    # 绘制中间切片
    if 'pressure_history' in results and results['pressure_history']:
        final_pressure = results['pressure_history'][-1]
        fig = plotter.plot_slice(
            final_pressure,
            direction='z',
            title="Pressure Field (Z-slice)",
            field_name="Pressure (Pa)",
            save_path="pressure_z_slice.png"
        )
        figures.append(fig)
    
    return figures
