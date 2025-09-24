"""
3D可视化模块

提供交互式3D压力场、饱和度场等的可视化功能，类似Paraview效果
"""

import numpy as np
import pyvista as pv
from typing import Dict, Any, List, Optional
import os


class Plot3D:
    """
    3D可视化类

    提供交互式3D压力场、饱和度场等的可视化功能
    """

    def __init__(self, mesh, config: Dict[str, Any] = None):
        """
        初始化3D可视化器

        Args:
            mesh: 网格对象
            config: 可视化配置字典，可包含以下键值对：
                - 'window_size': 窗口大小，格式为(width, height)的元组，默认为(1600, 1200)
                - 'output_dir': 输出目录，用于保存截图和结果文件，默认为'results'
        """
        self.mesh = mesh
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', 'results')
        self.window_size = self.config.get('window_size', (1600, 1200))  # 默认窗口大小为1600x1200

        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 设置PyVista主题
        pv.set_plot_theme('document')

    def _create_grid_mesh(self, field_data: np.ndarray = None) -> pv.UnstructuredGrid:
        """
        创建网格网格

        Args:
            field_data: 场数据（可选）

        Returns:
            PyVista网格对象
        """
        # 使用网格中已有的节点坐标
        nodes = self.mesh.node_list
        grid_points = np.array([node.coord for node in nodes])
        
        # 创建单元格连接信息
        cells = []
        cell_types = []
        
        # 使用网格中已有的单元顶点信息
        for cell in self.mesh.cell_list:
            # 获取单元的8个顶点索引
            points = cell.vertices
            
            # 添加单元格信息 (npoints, point0, point1, ...)
            cells.extend([8] + points)
            cell_types.append(pv.CellType.HEXAHEDRON)
        
        # 创建网格
        grid = pv.UnstructuredGrid(np.array(cells), np.array(cell_types), grid_points)
        
        # 添加场数据
        if field_data is not None:
            grid.cell_data["field"] = field_data.flatten()
            
        return grid

    def plot_pressure_field_3d(self, pressure: np.ndarray,
                              title: str = "3D Pressure Field",
                              save_path: Optional[str] = None,
                              show_wells: bool = True,
                              wells: Optional[List] = None) -> pv.Plotter:
        """
        绘制3D压力场

        Args:
            pressure: 压力场数据
            title: 图标题
            save_path: 保存路径
            show_wells: 是否显示井位置
            wells: 井列表

        Returns:
            PyVista绘图器对象
        """
        # 创建网格
        grid = self._create_grid_mesh(pressure)

        # 创建绘图器
        plotter = pv.Plotter()
        plotter.window_size = self.window_size  # 设置窗口大小
        plotter.add_mesh(grid, scalars="field", cmap="viridis", show_edges=True,
                        edge_color="gray", line_width=1,
                        scalar_bar_args={'title': 'Pressure (Pa)'})

        # 添加坐标轴
        plotter.show_axes()
        plotter.add_axes()

        # 设置标题
        plotter.add_title(title, font_size=14)

        # 显示井位置
        if show_wells and wells:
            for i, well in enumerate(wells):
                z, y, x = well.location
                # 转换为实际坐标
                x_coord = (x + 0.5) * self.mesh.dx
                y_coord = (y + 0.5) * self.mesh.dy
                z_coord = (z + 0.5) * self.mesh.dz

                # 添加井点
                if well.control_type == 'rate':
                    # 注入井用绿色球体
                    sphere = pv.Sphere(radius=min(self.mesh.dx, self.mesh.dy, self.mesh.dz)*0.3,
                                     center=[x_coord, y_coord, z_coord])
                    plotter.add_mesh(sphere, color='green',
                                   render_points_as_spheres=True)
                else:
                    # 生产井用红色球体
                    sphere = pv.Sphere(radius=min(self.mesh.dx, self.mesh.dy, self.mesh.dz)*0.3,
                                     center=[x_coord, y_coord, z_coord])
                    plotter.add_mesh(sphere, color='red',
                                   render_points_as_spheres=True)

        # 设置背景
        plotter.set_background('white')

        # 保存图形
        if save_path:
            plotter.screenshot(os.path.join(self.output_dir, save_path), 
                             window_size=self.window_size)  # 指定窗口大小

        return plotter

    def plot_saturation_field_3d(self, saturation: np.ndarray,
                                title: str = "3D Saturation Field",
                                save_path: Optional[str] = None) -> pv.Plotter:
        """
        绘制3D饱和度场

        Args:
            saturation: 饱和度场数据
            title: 图标题
            save_path: 保存路径

        Returns:
            PyVista绘图器对象
        """
        # 创建网格
        grid = self._create_grid_mesh(saturation)

        # 创建绘图器
        plotter = pv.Plotter()
        plotter.window_size = self.window_size  # 设置窗口大小
        plotter.add_mesh(grid, scalars="field", cmap="Blues", show_edges=True,
                        edge_color="gray", line_width=1,
                        scalar_bar_args={'title': 'Saturation'})

        # 添加坐标轴
        plotter.show_axes()
        plotter.add_axes()

        # 设置标题
        plotter.add_title(title, font_size=14)

        # 设置背景
        plotter.set_background('white')

        # 保存图形
        if save_path:
            plotter.screenshot(os.path.join(self.output_dir, save_path),
                             window_size=self.window_size)  # 指定窗口大小

        return plotter

    def plot_permeability_field_3d(self, permeability: np.ndarray,
                                  title: str = "3D Permeability Field",
                                  save_path: Optional[str] = None) -> pv.Plotter:
        """
        绘制3D渗透率场

        Args:
            permeability: 渗透率场数据
            title: 图标题
            save_path: 保存路径

        Returns:
            PyVista绘图器对象
        """
        # 创建网格
        grid = self._create_grid_mesh(permeability)

        # 创建绘图器
        plotter = pv.Plotter()
        plotter.window_size = self.window_size  # 设置窗口大小
        plotter.add_mesh(grid, scalars="field", cmap="plasma", show_edges=True,
                        edge_color="gray", line_width=1,
                        scalar_bar_args={'title': 'Permeability (mD)'})

        # 添加坐标轴
        plotter.show_axes()
        plotter.add_axes()

        # 设置标题
        plotter.add_title(title, font_size=14)

        # 设置背景
        plotter.set_background('white')

        # 保存图形
        if save_path:
            plotter.screenshot(os.path.join(self.output_dir, save_path),
                             window_size=self.window_size)  # 指定窗口大小

        return plotter

    def interactive_plot(self, field: np.ndarray,
                        field_name: str = "Field",
                        colormap: str = "viridis") -> pv.Plotter:
        """
        创建交互式3D可视化

        Args:
            field: 场数据
            field_name: 场名称
            colormap: 颜色映射

        Returns:
            PyVista绘图器对象
        """
        # 创建网格
        grid = self._create_grid_mesh(field)

        # 创建交互式绘图器
        plotter = pv.Plotter()
        plotter.window_size = self.window_size  # 设置窗口大小
        plotter.add_mesh(grid, scalars="field", cmap=colormap, show_edges=True,
                        edge_color="gray", line_width=0.5,
                        scalar_bar_args={'title': field_name})

        # 添加坐标轴
        plotter.show_axes()
        plotter.add_axes()

        # 设置背景
        plotter.set_background('white')

        return plotter


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
                             wells: Optional[List] = None,
                             config: Dict[str, Any] = None) -> List[pv.Plotter]:
    """
    绘制3D模拟结果

    Args:
        mesh: 网格对象
        results: 模拟结果
        wells: 井列表
        config: 可视化配置

    Returns:
        PyVista绘图器对象列表
    """
    # 创建可视化器
    plotter = create_3d_plotter(mesh, config)

    plotters = []

    # 绘制最终压力场
    if 'pressure_history' in results and results['pressure_history']:
        final_pressure = results['pressure_history'][-1]
        p = plotter.plot_pressure_field_3d(
            final_pressure,
            title="3D Final Pressure Field",
            save_path="3d_final_pressure_field.png",
            show_wells=True,
            wells=wells
        )
        plotters.append(p)

    # 绘制最终饱和度场（如果是两相流）
    if 'saturation_history' in results and results['saturation_history']:
        final_saturation = results['saturation_history'][-1]
        p = plotter.plot_saturation_field_3d(
            final_saturation,
            title="3D Final Saturation Field",
            save_path="3d_final_saturation_field.png"
        )
        plotters.append(p)

    return plotters
