"""
动画可视化模块

提供模拟结果的动画生成功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, Any, List, Optional
import os
from .plot_2d import Plot2D


class AnimationGenerator:
    """
    动画生成器类
    
    提供模拟结果的动画生成功能
    """
    
    def __init__(self, mesh, config: Dict[str, Any] = None):
        """
        初始化动画生成器
        
        Args:
            mesh: 网格对象
            config: 动画配置
        """
        self.mesh = mesh
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', 'results')
        self.fps = self.config.get('fps', 5)
        self.dpi = self.config.get('dpi', 150)
        self.figsize = self.config.get('figsize', (10, 8))
        
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def create_pressure_animation(self, pressure_history: List[np.ndarray], 
                                title: str = "Pressure Field Evolution",
                                save_path: Optional[str] = None) -> animation.FuncAnimation:
        """
        创建压力场演变动画
        
        Args:
            pressure_history: 压力场历史数据
            title: 动画标题
            save_path: 保存路径
            
        Returns:
            matplotlib动画对象
        """
        # 创建图形和轴
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 初始化第一帧
        pressure_2d = pressure_history[0].reshape(self.mesh.ny, self.mesh.nx)
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
        
        # 时间文本
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        def animate(frame):
            """动画帧函数"""
            pressure_2d = pressure_history[frame].reshape(self.mesh.ny, self.mesh.nx)
            im.set_array(pressure_2d)
            time_text.set_text(f'Frame: {frame+1}/{len(pressure_history)}')
            return [im, time_text]
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=len(pressure_history),
                                     interval=1000/self.fps, blit=True, repeat=True)
        
        # 保存动画
        if save_path:
            anim.save(os.path.join(self.output_dir, save_path), 
                     writer='pillow', fps=self.fps, dpi=self.dpi)
        
        return anim
    
    def create_saturation_animation(self, saturation_history: List[np.ndarray],
                                  title: str = "Saturation Field Evolution",
                                  save_path: Optional[str] = None) -> animation.FuncAnimation:
        """
        创建饱和度场演变动画
        
        Args:
            saturation_history: 饱和度场历史数据
            title: 动画标题
            save_path: 保存路径
            
        Returns:
            matplotlib动画对象
        """
        # 创建图形和轴
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 初始化第一帧
        saturation_2d = saturation_history[0].reshape(self.mesh.ny, self.mesh.nx)
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
        
        # 时间文本
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        def animate(frame):
            """动画帧函数"""
            saturation_2d = saturation_history[frame].reshape(self.mesh.ny, self.mesh.nx)
            im.set_array(saturation_2d)
            time_text.set_text(f'Frame: {frame+1}/{len(saturation_history)}')
            return [im, time_text]
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=len(saturation_history),
                                     interval=1000/self.fps, blit=True, repeat=True)
        
        # 保存动画
        if save_path:
            anim.save(os.path.join(self.output_dir, save_path), 
                     writer='pillow', fps=self.fps, dpi=self.dpi)
        
        return anim
    
    def create_combined_animation(self, pressure_history: List[np.ndarray],
                                saturation_history: List[np.ndarray],
                                title: str = "Pressure and Saturation Evolution",
                                save_path: Optional[str] = None) -> animation.FuncAnimation:
        """
        创建压力和饱和度联合演变动画
        
        Args:
            pressure_history: 压力场历史数据
            saturation_history: 饱和度场历史数据
            title: 动画标题
            save_path: 保存路径
            
        Returns:
            matplotlib动画对象
        """
        # 创建图形和轴
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 初始化压力场子图
        pressure_2d = pressure_history[0].reshape(self.mesh.ny, self.mesh.nx)
        im1 = ax1.imshow(pressure_2d, cmap='viridis', origin='lower',
                        extent=[0, self.mesh.nx*self.mesh.dx,
                               0, self.mesh.ny*self.mesh.dy])
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Pressure (Pa)', rotation=270, labelpad=20)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Pressure Field')
        
        # 初始化饱和度场子图
        saturation_2d = saturation_history[0].reshape(self.mesh.ny, self.mesh.nx)
        im2 = ax2.imshow(saturation_2d, cmap='Blues', origin='lower',
                        extent=[0, self.mesh.nx*self.mesh.dx,
                               0, self.mesh.ny*self.mesh.dy],
                        vmin=0, vmax=1)
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Saturation', rotation=270, labelpad=20)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Saturation Field')
        
        # 总标题
        fig.suptitle(title, fontsize=16)
        
        # 时间文本
        time_text = fig.text(0.02, 0.95, '', transform=fig.transFigure,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        def animate(frame):
            """动画帧函数"""
            # 更新压力场
            pressure_2d = pressure_history[frame].reshape(self.mesh.ny, self.mesh.nx)
            im1.set_array(pressure_2d)
            
            # 更新饱和度场
            saturation_2d = saturation_history[frame].reshape(self.mesh.ny, self.mesh.nx)
            im2.set_array(saturation_2d)
            
            time_text.set_text(f'Frame: {frame+1}/{len(pressure_history)}')
            return [im1, im2, time_text]
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=len(pressure_history),
                                     interval=1000/self.fps, blit=True, repeat=True)
        
        # 保存动画
        if save_path:
            anim.save(os.path.join(self.output_dir, save_path), 
                     writer='pillow', fps=self.fps, dpi=self.dpi)
        
        return anim
    
    def create_contour_animation(self, field_history: List[np.ndarray],
                               title: str = "Contour Evolution",
                               field_name: str = "Field",
                               save_path: Optional[str] = None) -> animation.FuncAnimation:
        """
        创建等值线演变动画
        
        Args:
            field_history: 场历史数据
            title: 动画标题
            field_name: 场名称
            save_path: 保存路径
            
        Returns:
            matplotlib动画对象
        """
        # 创建图形和轴
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 创建网格坐标
        x = np.linspace(0, self.mesh.nx*self.mesh.dx, self.mesh.nx)
        y = np.linspace(0, self.mesh.ny*self.mesh.dy, self.mesh.ny)
        X, Y = np.meshgrid(x, y)
        
        # 初始化第一帧
        field_2d = field_history[0].reshape(self.mesh.ny, self.mesh.nx)
        contour = ax.contour(X, Y, field_2d, levels=20, colors='black', alpha=0.5)
        contourf = ax.contourf(X, Y, field_2d, levels=20, cmap='viridis', alpha=0.8)
        
        # 添加颜色条
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label(field_name, rotation=270, labelpad=20)
        
        # 设置标签和标题
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        
        # 时间文本
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        def animate(frame):
            """动画帧函数"""
            # 清除之前的等值线
            for coll in ax.collections:
                coll.remove()
            
            # 绘制新的等值线
            field_2d = field_history[frame].reshape(self.mesh.ny, self.mesh.nx)
            contour = ax.contour(X, Y, field_2d, levels=20, colors='black', alpha=0.5)
            contourf = ax.contourf(X, Y, field_2d, levels=20, cmap='viridis', alpha=0.8)
            
            time_text.set_text(f'Frame: {frame+1}/{len(field_history)}')
            return [time_text]
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=len(field_history),
                                     interval=1000/self.fps, blit=True, repeat=True)
        
        # 保存动画
        if save_path:
            anim.save(os.path.join(self.output_dir, save_path), 
                     writer='pillow', fps=self.fps, dpi=self.dpi)
        
        return anim
    
    def __repr__(self):
        return f"AnimationGenerator(mesh={self.mesh.nx}x{self.mesh.ny}, fps={self.fps})"


def create_animation_generator(mesh, config: Dict[str, Any] = None) -> AnimationGenerator:
    """
    创建动画生成器
    
    Args:
        mesh: 网格对象
        config: 动画配置
        
    Returns:
        AnimationGenerator实例
    """
    return AnimationGenerator(mesh, config)


def generate_simulation_animation(mesh, results: Dict[str, Any],
                                animation_type: str = 'pressure',
                                config: Dict[str, Any] = None) -> animation.FuncAnimation:
    """
    生成模拟结果动画
    
    Args:
        mesh: 网格对象
        results: 模拟结果
        animation_type: 动画类型 ('pressure', 'saturation', 'combined', 'contour')
        config: 动画配置
        
    Returns:
        matplotlib动画对象
    """
    # 创建动画生成器
    animator = create_animation_generator(mesh, config)
    
    # 根据类型生成动画
    if animation_type == 'pressure' and 'pressure_history' in results:
        anim = animator.create_pressure_animation(
            results['pressure_history'],
            title="Pressure Field Evolution",
            save_path="pressure_evolution.gif"
        )
    elif animation_type == 'saturation' and 'saturation_history' in results:
        anim = animator.create_saturation_animation(
            results['saturation_history'],
            title="Saturation Field Evolution",
            save_path="saturation_evolution.gif"
        )
    elif animation_type == 'combined' and 'pressure_history' in results and 'saturation_history' in results:
        anim = animator.create_combined_animation(
            results['pressure_history'],
            results['saturation_history'],
            title="Pressure and Saturation Evolution",
            save_path="combined_evolution.gif"
        )
    elif animation_type == 'contour' and 'pressure_history' in results:
        anim = animator.create_contour_animation(
            results['pressure_history'],
            title="Pressure Contour Evolution",
            field_name="Pressure (Pa)",
            save_path="pressure_contour_evolution.gif"
        )
    else:
        raise ValueError(f"Unsupported animation type: {animation_type}")
    
    return anim
