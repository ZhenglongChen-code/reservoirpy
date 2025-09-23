"""
2D可视化模块

提供2D压力场、饱和度场等的可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
import os


class Plot2D:
    """
    2D可视化类
    
    提供2D压力场、饱和度场等的可视化功能
    """
    
    def __init__(self, mesh, config: Dict[str, Any] = None):
        """
        初始化2D可视化器
        
        Args:
            mesh: 网格对象
            config: 可视化配置
        """
        self.mesh = mesh
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', 'results')
        self.dpi = self.config.get('dpi', 150)
        self.figsize = self.config.get('figsize', (10, 8))
        
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def plot_pressure_field(self, pressure: np.ndarray, 
                          title: str = "Pressure Field", 
                          save_path: Optional[str] = None,
                          show_wells: bool = True,
                          wells: Optional[List] = None) -> plt.Figure:
        """
        绘制压力场
        
        Args:
            pressure: 压力场数据
            title: 图标题
            save_path: 保存路径
            show_wells: 是否显示井位置
            wells: 井列表
            
        Returns:
            matplotlib图形对象
        """
        # 重塑压力场为2D网格
        pressure_2d = pressure.reshape(self.mesh.ny, self.mesh.nx)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 绘制压力场
        im = ax.imshow(pressure_2d, cmap='viridis', origin='lower', 
                      extent=[0, self.mesh.nx*self.mesh.dx, 
                             0, self.mesh.ny*self.mesh.dy])
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Pressure (Pa)', rotation=270, labelpad=20)
        
        # 设置标签和标题
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        
        # 显示井位置
        if show_wells and wells:
            for i, well in enumerate(wells):
                z, y, x = well.location
                # 转换为实际坐标
                x_coord = (x + 0.5) * self.mesh.dx
                y_coord = (y + 0.5) * self.mesh.dy
                
                if well.control_type == 'rate':
                    # 注入井用绿色三角形
                    ax.plot(x_coord, y_coord, '^', color='green', 
                           markersize=10, markeredgecolor='black')
                else:
                    # 生产井用红色倒三角形
                    ax.plot(x_coord, y_coord, 'v', color='red', 
                           markersize=10, markeredgecolor='black')
                
                # 添加井标签
                ax.text(x_coord + self.mesh.dx*0.1, y_coord + self.mesh.dy*0.1, 
                       f'W{i+1}', fontsize=8, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        # 保存图形
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), 
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_saturation_field(self, saturation: np.ndarray, 
                            title: str = "Saturation Field",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制饱和度场
        
        Args:
            saturation: 饱和度场数据
            title: 图标题
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        # 重塑饱和度场为2D网格
        saturation_2d = saturation.reshape(self.mesh.ny, self.mesh.nx)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 绘制饱和度场
        im = ax.imshow(saturation_2d, cmap='Blues', origin='lower', 
                      extent=[0, self.mesh.nx*self.mesh.dx, 
                             0, self.mesh.ny*self.mesh.dy],
                      vmin=0, vmax=1)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Saturation', rotation=270, labelpad=20)
        
        # 设置标签和标题
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        
        # 保存图形
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), 
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_well_curves(self, well_data: List[Dict[str, Any]], 
                        title: str = "Well Performance",
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制井曲线
        
        Args:
            well_data: 井数据列表
            title: 图标题
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        # 创建图形
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 绘制每口井的曲线
        for i, data in enumerate(well_data):
            times = np.array(data['time_history']) / 86400  # 转换为天
            pressures = np.array(data['pressure_history']) / 1e6  # 转换为MPa
            
            ax.plot(times, pressures, label=f'Well {i+1}', marker='o', markersize=3)
        
        # 设置标签和标题
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Pressure (MPa)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 保存图形
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), 
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_pressure_saturation(self, pressure: np.ndarray, 
                               saturation: np.ndarray,
                               title: str = "Pressure vs Saturation",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制压力-饱和度关系图
        
        Args:
            pressure: 压力场数据
            saturation: 饱和度场数据
            title: 图标题
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        # 创建图形
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 绘制散点图
        ax.scatter(pressure/1e6, saturation, alpha=0.6, s=20)
        
        # 设置标签和标题
        ax.set_xlabel('Pressure (MPa)')
        ax.set_ylabel('Saturation')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # 保存图形
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), 
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_contour(self, field: np.ndarray, 
                    title: str = "Contour Plot",
                    field_name: str = "Field",
                    save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制等值线图
        
        Args:
            field: 场数据
            title: 图标题
            field_name: 场名称
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        # 重塑场数据为2D网格
        field_2d = field.reshape(self.mesh.ny, self.mesh.nx)
        
        # 创建网格坐标
        x = np.linspace(0, self.mesh.nx*self.mesh.dx, self.mesh.nx)
        y = np.linspace(0, self.mesh.ny*self.mesh.dy, self.mesh.ny)
        X, Y = np.meshgrid(x, y)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 绘制等值线
        contour = ax.contour(X, Y, field_2d, levels=20, colors='black', alpha=0.5, linewidths=0.5)
        contourf = ax.contourf(X, Y, field_2d, levels=20, cmap='viridis', alpha=0.8)
        
        # 添加颜色条
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label(field_name, rotation=270, labelpad=20)
        
        # 添加等值线标签
        ax.clabel(contour, inline=True, fontsize=8)
        
        # 设置标签和标题
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        
        # 保存图形
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), 
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def create_comparison_plot(self, fields: List[np.ndarray], 
                             labels: List[str],
                             title: str = "Field Comparison",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        创建场比较图
        
        Args:
            fields: 场数据列表
            labels: 标签列表
            title: 图标题
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        n_fields = len(fields)
        cols = min(3, n_fields)
        rows = (n_fields + cols - 1) // cols
        
        # 创建图形
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if n_fields == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # 绘制每个场
        for i, (field, label) in enumerate(zip(fields, labels)):
            if i < len(axes):
                ax = axes[i]
                # 重塑场数据为2D网格
                field_2d = field.reshape(self.mesh.ny, self.mesh.nx)
                
                # 绘制场
                im = ax.imshow(field_2d, cmap='viridis', origin='lower',
                              extent=[0, self.mesh.nx*self.mesh.dx,
                                     0, self.mesh.ny*self.mesh.dy])
                
                # 添加颜色条
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(label, rotation=270, labelpad=15)
                
                # 设置标签和标题
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_title(label)
        
        # 隐藏多余的子图
        for i in range(n_fields, len(axes)):
            axes[i].set_visible(False)
        
        # 设置总标题
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # 保存图形
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), 
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def __repr__(self):
        return f"Plot2D(mesh={self.mesh.nx}x{self.mesh.ny})"


def create_2d_plotter(mesh, config: Dict[str, Any] = None) -> Plot2D:
    """
    创建2D可视化器
    
    Args:
        mesh: 网格对象
        config: 可视化配置
        
    Returns:
        Plot2D实例
    """
    return Plot2D(mesh, config)


def plot_simulation_results(mesh, results: Dict[str, Any], 
                          wells: Optional[List] = None,
                          config: Dict[str, Any] = None) -> List[plt.Figure]:
    """
    绘制模拟结果
    
    Args:
        mesh: 网格对象
        results: 模拟结果
        wells: 井列表
        config: 可视化配置
        
    Returns:
        matplotlib图形对象列表
    """
    # 创建可视化器
    plotter = create_2d_plotter(mesh, config)
    
    figures = []
    
    # 绘制最终压力场
    if 'pressure_history' in results and results['pressure_history']:
        final_pressure = results['pressure_history'][-1]
        fig = plotter.plot_pressure_field(
            final_pressure, 
            title="Final Pressure Field",
            save_path="final_pressure_field.png",
            show_wells=True,
            wells=wells
        )
        figures.append(fig)
    
    # 绘制最终饱和度场（如果是两相流）
    if 'saturation_history' in results and results['saturation_history']:
        final_saturation = results['saturation_history'][-1]
        fig = plotter.plot_saturation_field(
            final_saturation,
            title="Final Saturation Field",
            save_path="final_saturation_field.png"
        )
        figures.append(fig)
    
    # 绘制井曲线
    if 'well_data' in results and results['well_data']:
        fig = plotter.plot_well_curves(
            results['well_data'],
            title="Well Performance",
            save_path="well_performance.png"
        )
        figures.append(fig)
    
    return figures
