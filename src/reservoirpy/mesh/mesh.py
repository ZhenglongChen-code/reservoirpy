"""
网格管理模块 - 基于现有meshgrid.py重构

提供结构化矩形网格（2D/3D）的拓扑与几何信息，支持后续FVM离散。
"""

import numpy as np
from typing import Tuple, List, Optional, Union
from abc import ABC, abstractmethod


class BaseMesh(ABC):
    """
    网格基类
    
    定义网格类的基本接口，为结构化和非结构化网格提供统一的抽象。
    """
    
    def __init__(self):
        """初始化网格基类"""
        self.n_cells = 0
        self.node_list = []
        self.cell_list = []
    
    @abstractmethod
    def get_cell_volume(self, cell_index: int) -> float:
        """
        获取单元体积
        
        Args:
            cell_index: 单元索引
            
        Returns:
            单元体积
        """
        pass
    
    @abstractmethod
    def get_face_area(self, face_index: int) -> float:
        """
        获取界面面积
        
        Args:
            face_index: 界面索引
            
        Returns:
            界面面积
        """
        pass
    
    @abstractmethod
    def get_neighbors(self, cell_index: int) -> List[int]:
        """
        获取相邻单元索引
        
        Args:
            cell_index: 单元索引
            
        Returns:
            相邻单元索引列表
        """
        pass
    
    @abstractmethod
    def is_boundary_cell(self, cell_index: int) -> bool:
        """
        判断是否为边界单元
        
        Args:
            cell_index: 单元索引
            
        Returns:
            是否为边界单元
        """
        pass
    
    @property
    @abstractmethod
    def total_cells(self) -> int:
        """总单元数"""
        pass
    
    def get_cell_centers(self) -> np.ndarray:
        """
        获取所有单元中心坐标
        
        Returns:
            形状为 (ncell, 3) 的数组，每行为 [x, y, z]
        """
        centers = np.zeros((self.total_cells, 3))
        for i, cell in enumerate(self.cell_list):
            centers[i] = cell.center
        return centers
    
    def get_cell_volumes(self) -> np.ndarray:
        """
        获取所有单元体积
        
        Returns:
            形状为 (ncell,) 的数组
        """
        volumes = np.zeros(self.total_cells)
        for i, cell in enumerate(self.cell_list):
            volumes[i] = cell.volume
        return volumes


class Node:
    """
    网格节点类
    
    表示网格中的一个节点，包含坐标信息。
    """
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """
        初始化节点
        
        Args:
            x: X坐标
            y: Y坐标  
            z: Z坐标
        """
        self.x = x
        self.y = y
        self.z = z
        self.coord = [x, y, z]
    
    def __repr__(self):
        return f"Node({self.coord[0]:.2f}, {self.coord[1]:.2f}, {self.coord[2]:.2f})"


class BaseCell(ABC):
    """
    网格单元基类
    
    定义网格单元的基本接口，为不同类型的单元（如立方体、四面体等）提供统一的抽象。
    """
    
    def __init__(self, index: int, center: List[float], volume: float):
        """
        初始化单元基类
        
        Args:
            index: 单元索引
            center: 单元中心坐标 [x, y, z]
            volume: 单元体积
        """
        self.index = index
        self.center = center
        self.volume = volume
        self.neighbors = []  # 邻居列表，大小取决于单元类型
        self.boundary_type = None
        
        # 几何属性
        self.vertices = []  # 顶点列表，大小取决于单元类型
        
        # 物理属性（将在physics模块中设置）
        self.porosity = 0.0
        self.kx = 0.0
        self.ky = 0.0
        self.kz = 0.0
        self.trans = []  # 传导率列表，大小取决于单元类型
        
        # 边界和井条件
        self.mark_bc = 0
        self.markwell = 0
        self.well_id = -1
        
        # 流体属性
        self.press = 0.0
        self.Sw = 0.0
        self.So = 0.0
        self.mu_o = 1.8e-3
        self.mu_w = 1e-3
        self.kro_div_mu_o = 0.0
        self.krw_div_mu_w = 0.0
        self.sum_k_div_mu = 0.0
    
    @abstractmethod
    def get_face_count(self) -> int:
        """
        获取单元的面数
        
        Returns:
            面单数
        """
        pass
    
    @abstractmethod
    def get_vertex_count(self) -> int:
        """
        获取单元的顶点数
        
        Returns:
            顶点数
        """
        pass


class CubeCell(BaseCell):
    """
    结构化网格单元类（六面体）
    
    表示结构化网格中的一个六面体单元，包含几何和物理属性信息。
    """
    
    def __init__(self, index: int, center: List[float], volume: float):
        """
        初始化六面体单元
        
        Args:
            index: 单元索引
            center: 单元中心坐标 [x, y, z]
            volume: 单元体积
        """
        super().__init__(index, center, volume)
        
        # 为六面体单元初始化特定属性
        self.neighbors = [-1] * 6  # W, E, N, S, F, B
        self.vertices = [-1] * 8
        self.trans = [0.0] * 6  # 6个面的传导率
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
    
    def get_face_count(self) -> int:
        """获取六面体的面数"""
        return 6
    
    def get_vertex_count(self) -> int:
        """获取六面体的顶点数"""
        return 8
    
    def __repr__(self):
        return f"Cell({self.index}, center={self.center}, volume={self.volume:.2e})"


class StructuredMesh(BaseMesh):
    """
    结构化网格管理类
    
    提供结构化矩形网格的几何和拓扑信息，支持2D和3D网格。
    """
    
    def __init__(self, nx: int, ny: int, nz: int, dx: float, dy: float, dz: float):
        """
        初始化结构化网格
        
        Args:
            nx: X方向单元数
            ny: Y方向单元数
            nz: Z方向单元数
            dx: X方向单元尺寸
            dy: Y方向单元尺寸
            dz: Z方向单元尺寸
        """
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        
        # 计算总单元数
        self.n_cells = nx * ny * nz
        
        # 构建网格
        self.node_list = self._generate_nodes()
        self.cell_list = self._generate_cells()
    
    def _generate_nodes(self) -> List[Node]:
        """
        生成网格节点
        
        Returns:
            节点列表
        """
        nodes = []
        for i in range(self.nz + 1):
            current_z = i * self.dz
            for j in range(self.ny + 1):
                current_y = j * self.dy
                for k in range(self.nx + 1):
                    current_x = k * self.dx
                    nodes.append(Node(current_x, current_y, current_z))
        return nodes
    
    def _generate_cells(self) -> List[CubeCell]:
        """
        生成网格单元
        
        Returns:
            单元列表
        """
        cells = []
        for i in range(self.nz):
            for j in range(self.ny):
                for k in range(self.nx):
                    idx = i * self.nx * self.ny + j * self.nx + k
                    
                    # 计算单元中心坐标
                    center_x = (k + 0.5) * self.dx
                    center_y = (j + 0.5) * self.dy
                    center_z = (i + 0.5) * self.dz
                    
                    # 创建单元
                    cell = CubeCell(
                        index=idx,
                        center=[center_x, center_y, center_z],
                        volume=self.dx * self.dy * self.dz
                    )
                    
                    # 设置单元尺寸
                    cell.dx = self.dx
                    cell.dy = self.dy
                    cell.dz = self.dz
                    
                    # 设置相邻单元索引
                    cell.neighbors[0] = idx - 1 if k > 0 else -1  # West
                    cell.neighbors[1] = idx + 1 if k < self.nx - 1 else -1  # East
                    cell.neighbors[2] = idx - self.nx if j > 0 else -1  # North
                    cell.neighbors[3] = idx + self.nx if j < self.ny - 1 else -1  # South
                    cell.neighbors[4] = idx - self.nx * self.ny if i > 0 else -1  # Front
                    cell.neighbors[5] = idx + self.nx * self.ny if i < self.nz - 1 else -1  # Back
                    
                    # 设置顶点索引
                    # PyVista/VTK六面体顶点顺序:
                    # 底面 (按逆时针方向，从角落开始): 0, 1, 2, 3
                    # 顶面 (按逆时针方向，从角落开始): 4, 5, 6, 7
                    # 0-1-2-3 构成底面，4-5-6-7 构成顶面，且 0与4, 1与5 等垂直对齐
                    i0 = i * (self.ny + 1) * (self.nx + 1) + j * (self.nx + 1) + k          # 0
                    i1 = i * (self.ny + 1) * (self.nx + 1) + j * (self.nx + 1) + k + 1      # 1
                    i2 = i * (self.ny + 1) * (self.nx + 1) + (j + 1) * (self.nx + 1) + k + 1  # 2
                    i3 = i * (self.ny + 1) * (self.nx + 1) + (j + 1) * (self.nx + 1) + k      # 3
                    i4 = (i + 1) * (self.ny + 1) * (self.nx + 1) + j * (self.nx + 1) + k      # 4
                    i5 = (i + 1) * (self.ny + 1) * (self.nx + 1) + j * (self.nx + 1) + k + 1  # 5
                    i6 = (i + 1) * (self.ny + 1) * (self.nx + 1) + (j + 1) * (self.nx + 1) + k + 1  # 6
                    i7 = (i + 1) * (self.ny + 1) * (self.nx + 1) + (j + 1) * (self.nx + 1) + k      # 7
                    cell.vertices = [i0, i1, i2, i3, i4, i5, i6, i7]
                    
                    cells.append(cell)
        
        return cells
    
    def get_cell_volume(self, i: int, j: int, k: int) -> float:
        """
        获取单元体积
        
        Args:
            i: Z方向索引
            j: Y方向索引
            k: X方向索引
            
        Returns:
            单元体积
        """
        return self.dx * self.dy * self.dz
    
    def get_face_area(self, direction: str, i: int, j: int, k: int) -> float:
        """
        获取界面面积
        
        Args:
            direction: 方向 ('x', 'y', 'z')
            i: Z方向索引
            j: Y方向索引
            k: X方向索引
            
        Returns:
            界面面积
        """
        if direction == 'x':
            return self.dy * self.dz
        elif direction == 'y':
            return self.dx * self.dz
        elif direction == 'z':
            return self.dx * self.dy
        else:
            raise ValueError(f"Invalid direction: {direction}")
    
    def get_face_distance(self, direction: str, i: int, j: int, k: int) -> float:
        """
        获取到相邻单元中心的距离
        
        Args:
            direction: 方向 ('x', 'y', 'z')
            i: Z方向索引
            j: Y方向索引
            k: X方向索引
            
        Returns:
            距离
        """
        if direction == 'x':
            return self.dx
        elif direction == 'y':
            return self.dy
        elif direction == 'z':
            return self.dz
        else:
            raise ValueError(f"Invalid direction: {direction}")
    
    def get_neighbors(self, i: int, j: int, k: int) -> List[int]:
        """
        获取相邻单元索引
        
        Args:
            i: Z方向索引
            j: Y方向索引
            k: X方向索引
            
        Returns:
            相邻单元索引列表 [W, E, N, S, F, B]
        """
        idx = i * self.nx * self.ny + j * self.nx + k
        cell = self.cell_list[idx]
        return cell.neighbors.copy()
    
    def is_boundary_cell(self, i: int, j: int, k: int) -> bool:
        """
        判断是否为边界单元
        
        Args:
            i: Z方向索引
            j: Y方向索引
            k: X方向索引
            
        Returns:
            是否为边界单元
        """
        # 对于2D网格（nz=1），只检查x和y方向的边界
        if self.nz == 1:
            return (j == 0 or j == self.ny - 1 or 
                    k == 0 or k == self.nx - 1)
        else:
            return (i == 0 or i == self.nz - 1 or 
                    j == 0 or j == self.ny - 1 or 
                    k == 0 or k == self.nx - 1)
    
    def get_cell_index(self, i: int, j: int, k: int) -> int:
        """
        获取单元的一维索引
        
        Args:
            i: Z方向索引
            j: Y方向索引
            k: X方向索引
            
        Returns:
            单元索引
        """
        return i * self.nx * self.ny + j * self.nx + k
    
    def get_cell_coords(self, index: int) -> Tuple[int, int, int]:
        """
        从一维索引获取三维坐标
        
        Args:
            index: 单元索引
            
        Returns:
            (i, j, k) 三维坐标
        """
        k = index % self.nx
        j = (index // self.nx) % self.ny
        i = index // (self.nx * self.ny)
        return i, j, k
    
    @property
    def total_cells(self) -> int:
        """总单元数"""
        return self.n_cells
    
    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """网格形状 (nx, ny, nz)"""
        return (self.nx, self.ny, self.nz)
    
    def __repr__(self):
        return f"StructuredMesh({self.nx}x{self.ny}x{self.nz}, cells={self.n_cells})"


# 向后兼容的别名
MeshGrid = StructuredMesh