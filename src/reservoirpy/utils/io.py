"""
输入输出工具模块

提供配置文件读写、数据导入导出等功能
"""

import yaml
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import os
import pyvista as pv
import numpy as np


def mesh_to_vtk(mesh, pressure=None, saturation=None, permeability=None, filename="reservoir.vtk", file_format="binary"):
    """
    将结构化网格和场数据保存为VTK文件

    Args:
        mesh: StructuredMesh对象
        pressure: 压力场数据数组
        saturation: 饱和度场数据数组
        permeability: 渗透率场数据数组
        filename: 保存的VTK文件名
        file_format: 文件格式，"binary"（默认，文件小）或"ascii"（可用文本编辑器查看）
    """
    # 创建节点坐标数组
    points = np.array([node.coord for node in mesh.node_list])

    # 创建单元连接信息
    cells = []
    cell_types = []

    for cell in mesh.cell_list:
        # 每个六面体单元有8个顶点
        node_indices = cell.vertices  # 使用vertices属性而不是nodes
        cells.extend([8] + node_indices)  # VTK格式: [n_points, point0, point1, ...]
        cell_types.append(pv.CellType.HEXAHEDRON)

    # 创建非结构化网格
    grid = pv.UnstructuredGrid(np.array(cells), np.array(cell_types), points)

    # 添加场数据
    if pressure is not None:
        grid.cell_data["Pressure"] = pressure

    if saturation is not None:
        grid.cell_data["Saturation"] = saturation

    if permeability is not None:
        grid.cell_data["Permeability"] = permeability

    # 保存为VTK文件
    if file_format.lower() == "ascii":
        grid.save(filename, binary=False)
        print(f"网格数据已保存到 {filename} (ASCII格式，可用文本编辑器查看)")
    else:
        grid.save(filename, binary=True)
        print(f"网格数据已保存到 {filename} (二进制格式)")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    保存配置文件
    
    Args:
        config: 配置字典
        config_path: 配置文件路径
    """
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        elif config_path.endswith('.json'):
            json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")


def load_grid_data(file_path: str) -> np.ndarray:
    """
    加载网格数据
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        网格数据数组
    """
    if file_path.endswith('.npy'):
        data = np.load(file_path)
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path).values
    elif file_path.endswith('.txt'):
        data = np.loadtxt(file_path)
    else:
        raise ValueError(f"Unsupported grid data file format: {file_path}")
    
    return data


def save_grid_data(data: np.ndarray, file_path: str):
    """
    保存网格数据
    
    Args:
        data: 网格数据数组
        file_path: 数据文件路径
    """
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if file_path.endswith('.npy'):
        np.save(file_path, data)
    elif file_path.endswith('.csv'):
        pd.DataFrame(data).to_csv(file_path, index=False)
    elif file_path.endswith('.txt'):
        np.savetxt(file_path, data)
    else:
        raise ValueError(f"Unsupported grid data file format: {file_path}")


def load_well_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载井数据
    
    Args:
        file_path: 井数据文件路径
        
    Returns:
        井数据列表
    """
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            well_data = json.load(f)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        well_data = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported well data file format: {file_path}")
    
    return well_data


def save_well_data(well_data: List[Dict[str, Any]], file_path: str):
    """
    保存井数据
    
    Args:
        well_data: 井数据列表
        file_path: 井数据文件路径
    """
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if file_path.endswith('.json'):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(well_data, f, indent=2, ensure_ascii=False)
    elif file_path.endswith('.csv'):
        df = pd.DataFrame(well_data)
        df.to_csv(file_path, index=False)
    else:
        raise ValueError(f"Unsupported well data file format: {file_path}")


def load_simulation_results(file_path: str) -> Dict[str, Any]:
    """
    加载模拟结果
    
    Args:
        file_path: 模拟结果文件路径
        
    Returns:
        模拟结果字典
    """
    if file_path.endswith('.npz'):
        data = np.load(file_path)
        results = {key: data[key] for key in data.files}
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        raise ValueError(f"Unsupported simulation results file format: {file_path}")
    
    return results


def save_simulation_results(results: Dict[str, Any], file_path: str):
    """
    保存模拟结果
    
    Args:
        results: 模拟结果字典
        file_path: 模拟结果文件路径
    """
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if file_path.endswith('.npz'):
        # 将列表转换为数组以便保存
        save_dict = {}
        for key, value in results.items():
            if isinstance(value, list):
                save_dict[key] = np.array(value)
            else:
                save_dict[key] = value
        np.savez_compressed(file_path, **save_dict)
    elif file_path.endswith('.json'):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    else:
        raise ValueError(f"Unsupported simulation results file format: {file_path}")


def export_to_vtk(mesh, pressure: np.ndarray, saturation: Optional[np.ndarray] = None,
                 file_path: str = "simulation_results.vtk"):
    """
    导出到VTK格式（用于可视化软件）
    
    Args:
        mesh: 网格对象
        pressure: 压力场
        saturation: 饱和度场（可选）
        file_path: VTK文件路径
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista is required for VTK export. Install it with: pip install pyvista")
    
    # 创建结构化网格
    x = np.linspace(0, mesh.nx*mesh.dx, mesh.nx+1)
    y = np.linspace(0, mesh.ny*mesh.dy, mesh.ny+1)
    z = np.linspace(0, mesh.nz*mesh.dz, mesh.nz+1)
    
    grid = pv.StructuredGrid()
    
    # 设置网格点
    grid.points = np.array([[xi, yi, zi] for zi in z for yi in y for xi in x])
    
    # 设置维度
    grid.dimensions = [mesh.nx+1, mesh.ny+1, mesh.nz+1]
    
    # 添加压力数据
    grid["pressure"] = pressure
    
    # 添加饱和度数据（如果提供）
    if saturation is not None:
        grid["saturation"] = saturation
    
    # 保存到文件
    grid.save(file_path)


class ConfigManager:
    """
    配置管理器
    
    提供配置文件的加载、保存和管理功能
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = {}
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = load_config(config_path)
    
    def save_config(self, config_path: Optional[str] = None):
        """
        保存配置文件
        
        Args:
            config_path: 配置文件路径（可选）
        """
        path = config_path or self.config_path
        if path:
            save_config(self.config, path)
        else:
            raise ValueError("No config path specified")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项
        
        Args:
            key: 配置项键
            default: 默认值
            
        Returns:
            配置项值
        """
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        设置配置项
        
        Args:
            key: 配置项键
            value: 配置项值
        """
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def update(self, config_dict: Dict[str, Any]):
        """
        更新配置
        
        Args:
            config_dict: 配置字典
        """
        self.config.update(config_dict)
    
    def __repr__(self):
        return f"ConfigManager(config_path={self.config_path})"


def create_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    创建配置管理器
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        ConfigManager实例
    """
    return ConfigManager(config_path)
