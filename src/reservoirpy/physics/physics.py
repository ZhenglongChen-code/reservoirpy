"""
物理属性模块

封装单相流和两相流动物理参数，为FVM提供物性输入。
"""

import numpy as np
from typing import Dict, Any, Union
from abc import ABC, abstractmethod
from reservoirpy.mesh.mesh import StructuredMesh


class BasePhysics(ABC):
    """
    物理属性基类
    
    定义所有物理属性类的通用接口
    """
    
    def __init__(self, mesh: StructuredMesh, config: Dict[str, Any]):
        """
        初始化基类
        
        Args:
            mesh: 网格对象
            config: 配置字典
        """
        self.mesh = mesh
        self.config = config
        self.property_manager = PropertyManager(mesh, config)
    
    @abstractmethod
    def get_transmissibility(self, cell_i: int, cell_j: int, direction: str) -> float:
        """
        计算传导率
        
        Args:
            cell_i: 当前单元索引
            cell_j: 相邻单元索引
            direction: 方向 ('x', 'y', 'z')
            
        Returns:
            传导率值
        """
        pass


class PropertyManager:
    """
    属性管理器
    
    负责物理属性的存储和访问，解耦物理属性与网格单元的直接关联。
    """
    
    def __init__(self, mesh: StructuredMesh, config: Dict[str, Any]):
        """
        初始化属性管理器
        
        Args:
            mesh: 网格对象
            config: 配置字典
        """
        self.mesh = mesh
        self.properties = self._initialize_properties(config)
    
    def _initialize_properties(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        初始化所有物理属性
        
        Args:
            config: 配置字典
            
        Returns:
            包含所有物理属性的字典
        """
        properties = {}
        properties['permeability'] = self._init_permeability(config)
        properties['porosity'] = self._init_porosity(config)
        return properties
    
    def _init_permeability(self, config: Dict[str, Any]) -> Union[float, np.ndarray]:
        """
        初始化渗透率
        
        Args:
            config: 配置字典
            
        Returns:
            渗透率值或数组
        """
        # 获取渗透率值
        perm_value = config.get('permeability', 100.0)  # 默认100 mD
        
        # 单位转换因子
        mD_to_m2 = 9.869233e-16  # mD to m²
        
        if isinstance(perm_value, (int, float)):
            # 均质渗透率，直接返回数值（已转换为SI单位）
            return float(perm_value) * mD_to_m2
        elif isinstance(perm_value, np.ndarray):
            # 非均质渗透率
            if perm_value.ndim == 1:
                # 一维数组，展平为三维
                nx, ny, nz = self.mesh.grid_shape
                perm_3d = perm_value.reshape(nz, ny, nx)
            elif perm_value.ndim == 3:
                # 三维数组
                perm_3d = perm_value
            else:
                raise ValueError("Permeability array must be 1D or 3D")
            
            # 转换为SI单位 (m²)
            return perm_3d * mD_to_m2
        else:
            raise ValueError("Permeability must be a number or numpy array")
    
    def _init_porosity(self, config: Dict[str, Any]) -> Union[float, np.ndarray]:
        """
        初始化孔隙度
        
        Args:
            config: 配置字典
            
        Returns:
            孔隙度值或数组
        """
        # 获取孔隙度值
        poro_value = config.get('porosity', 0.2)  # 默认0.2
        
        if isinstance(poro_value, (int, float)):
            # 均质孔隙度，直接返回数值
            return float(poro_value)
        elif isinstance(poro_value, np.ndarray):
            # 非均质孔隙度
            if poro_value.ndim == 1:
                # 一维数组，展平为三维
                nx, ny, nz = self.mesh.grid_shape
                return poro_value.reshape(nz, ny, nx)
            elif poro_value.ndim == 3:
                # 三维数组
                return poro_value
            else:
                raise ValueError("Porosity array must be 1D or 3D")
        else:
            raise ValueError("Porosity must be a number or numpy array")
    
    def get_cell_property(self, cell_index: int, property_name: str) -> float:
        """
        获取指定单元的属性值
        
        Args:
            cell_index: 单元索引
            property_name: 属性名称 ('permeability', 'porosity')
            
        Returns:
            指定单元的属性值
        """
        if property_name not in self.properties:
            raise ValueError(f"Property {property_name} not found")
            
        prop = self.properties[property_name]
        
        # 如果是标量，直接返回
        if isinstance(prop, (int, float)):
            return float(prop)
        
        # 如果是数组，根据索引获取值
        if isinstance(prop, np.ndarray):
            i, j, k = self.mesh.get_cell_coords(cell_index)
            if property_name == 'permeability':
                # 渗透率是四维数组 (nz, ny, nx, 3)
                # 为了简化，这里假设只考虑x方向的渗透率
                return float(prop[i, j, k, 0]) if prop.ndim == 4 else float(prop[i, j, k])
            else:
                # 其他属性是三维数组 (nz, ny, nx)
                return float(prop[i, j, k])
        
        raise ValueError(f"Unsupported property type for {property_name}")


class SinglePhaseProperties(BasePhysics):
    """
    单相流物理属性类
    
    封装单相流动物理参数，包括渗透率、孔隙度、粘度、压缩系数等。
    """
    
    def __init__(self, mesh: StructuredMesh, config: Dict[str, Any]):
        """
        初始化单相流物理属性
        
        Args:
            mesh: 网格对象
            config: 配置字典，包含以下键值对（所有参数均使用SI标准单位）：
                - permeability (float or np.ndarray): 渗透率，单位为mD，默认100.0
                - porosity (float or np.ndarray): 孔隙度，默认0.2
                - viscosity (float): 粘度，单位为Pa·s，默认0.001
                - compressibility (float): 压缩系数，单位为1/Pa，默认1e-9
                - reference_pressure (float): 参考压力，单位为Pa，默认0.0
        """
        super().__init__(mesh, config)
        
        # 初始化物理参数
        self.viscosity = self._init_viscosity(config)
        self.compressibility = self._init_compressibility(config)
        self.reference_pressure = config.get('reference_pressure', 30e6)
    
    def _init_viscosity(self, config: Dict[str, Any]) -> float:
        """
        初始化粘度
        
        Args:
            config: 配置字典
            
        Returns:
            粘度值 (Pa·s)
        """
        # 直接使用SI单位，不进行单位转换
        viscosity = config.get('viscosity', 0.002)  # 默认0.002 Pa·s
        return viscosity
    
    def _init_compressibility(self, config: Dict[str, Any]) -> float:
        """
        初始化压缩系数
        
        Args:
            config: 配置字典
            
        Returns:
            压缩系数值 (1/Pa)
        """
        return config.get('compressibility', 1e-9)  # 默认1e-9 1/Pa
    
    def get_transmissibility(self, cell_i: int, cell_j: int, direction: str) -> float:
        """
        计算传导率
        
        Args:
            cell_i: 当前单元索引
            cell_j: 相邻单元索引
            direction: 方向 ('x', 'y', 'z')
            
        Returns:
            传导率值
        """
        # 获取单元属性
        if direction == 'x':
            K1 = self.property_manager.get_cell_property(cell_i, 'permeability')
            K2 = self.property_manager.get_cell_property(cell_j, 'permeability')
            area = self.mesh.get_face_area('x', *self.mesh.get_cell_coords(cell_i))
            distance = self.mesh.get_face_distance('x', *self.mesh.get_cell_coords(cell_i))
        elif direction == 'y':
            K1 = self.property_manager.get_cell_property(cell_i, 'permeability')
            K2 = self.property_manager.get_cell_property(cell_j, 'permeability')
            area = self.mesh.get_face_area('y', *self.mesh.get_cell_coords(cell_i))
            distance = self.mesh.get_face_distance('y', *self.mesh.get_cell_coords(cell_i))
        elif direction == 'z':
            K1 = self.property_manager.get_cell_property(cell_i, 'permeability')
            K2 = self.property_manager.get_cell_property(cell_j, 'permeability')
            area = self.mesh.get_face_area('z', *self.mesh.get_cell_coords(cell_i))
            distance = self.mesh.get_face_distance('z', *self.mesh.get_cell_coords(cell_i))
        else:
            raise ValueError(f"Invalid direction: {direction}")
        
        # 调和平均渗透率
        K_harmonic = 2 * K1 * K2 / (K1 + K2) if (K1 + K2) > 0 else 0
        
        # 传导率 T = K * A / (μ * d)
        transmissibility = K_harmonic * area / (self.viscosity * distance)
        
        return transmissibility
    
    def get_fluid_density(self, pressure: float) -> float:
        """
        计算流体密度（微可压缩模型）
        
        Args:
            pressure: 压力 (Pa)
            
        Returns:
            密度 (kg/m³)
        """
        # 简化的微可压缩模型
        # ρ = ρ0 * exp(c * (P - P0))
        rho0 = 1000.0  # 参考密度 kg/m³
        return rho0 * np.exp(self.compressibility * (pressure - self.reference_pressure))


class TwoPhaseProperties(SinglePhaseProperties):
    """
    两相流物理属性类
    
    继承单相流属性，添加相对渗透率和毛管压力模型。
    """
    
    def __init__(self, mesh: StructuredMesh, config: Dict[str, Any]):
        """
        初始化两相流物理属性
        
        Args:
            mesh: 网格对象
            config: 配置字典，包含以下键值对（所有参数均使用SI标准单位）：
                - permeability (float or np.ndarray): 渗透率，单位为mD，默认100.0
                - porosity (float or np.ndarray): 孔隙度，默认0.2
                - viscosity (float): 粘度，单位为Pa·s，默认0.001
                - compressibility (float): 压缩系数，单位为1/Pa，默认1e-9
                - reference_pressure (float): 参考压力，单位为Pa，默认0.0
                - oil_viscosity (float): 油相粘度，单位为Pa·s，默认2e-3
                - water_viscosity (float): 水相粘度，单位为Pa·s，默认1e-3
                - kro_model (str): 油相相对渗透率模型，'corey'等，默认'corey'
                - krw_model (str): 水相相对渗透率模型，'corey'等，默认'corey'
                - pc_model (str): 毛管压力模型，'brooks_corey'等，默认'brooks_corey'
                - kro_params (dict): 油相相对渗透率模型参数，默认{'n_o': 2.0, 'S_or': 0.2}
                - krw_params (dict): 水相相对渗透率模型参数，默认{'n_w': 2.0, 'S_wr': 0.2}
                - pc_params (dict): 毛管压力模型参数，默认{'P_c0': 1000.0, 'lambda': 2.0}
        """
        super().__init__(mesh, config)
        
        # 两相流特有属性
        self.mu_o = config.get('oil_viscosity', 2e-3)
        self.mu_w = config.get('water_viscosity', 1e-3)
        self.viscosity = self.mu_w
        
        # 相对渗透率模型
        self.kro_model = config.get('kro_model', 'corey')
        self.krw_model = config.get('krw_model', 'corey')
        
        # 毛管压力模型
        self.pc_model = config.get('pc_model', 'brooks_corey')
        
        # 模型参数
        self.kro_params = config.get('kro_params', {'n_o': 2.0, 'S_or': 0.2})
        self.krw_params = config.get('krw_params', {'n_w': 2.0, 'S_wr': 0.2})
        self.pc_params = config.get('pc_params', {'P_c0': 1000.0, 'lambda': 2.0})
    
    def get_relative_permeability(self, saturation: float, phase: str) -> float:
        """
        获取相对渗透率
        
        Args:
            saturation: 饱和度
            phase: 相 ('oil' 或 'water')
            
        Returns:
            相对渗透率
        """
        if phase == 'oil':
            return self._get_kro(saturation)
        elif phase == 'water':
            return self._get_krw(saturation)
        else:
            raise ValueError(f"Invalid phase: {phase}")
    
    def _get_kro(self, Sw: float) -> float:
        """
        计算油相相对渗透率
        
        Args:
            Sw: 水相饱和度
            
        Returns:
            油相相对渗透率
        """
        if self.kro_model == 'corey':
            n_o = self.kro_params['n_o']
            S_or = self.kro_params['S_or']
            S_wr = self.krw_params['S_wr']
            
            if Sw <= S_wr:
                return 1.0
            elif Sw >= 1 - S_or:
                return 0.0
            else:
                S_o = 1 - Sw
                S_o_norm = (S_o - S_or) / (1 - S_wr - S_or)
                return S_o_norm ** n_o
        else:
            raise ValueError(f"Unsupported kro model: {self.kro_model}")
    
    def _get_krw(self, Sw: float) -> float:
        """
        计算水相相对渗透率
        
        Args:
            Sw: 水相饱和度
            
        Returns:
            水相相对渗透率
        """
        if self.krw_model == 'corey':
            n_w = self.krw_params['n_w']
            S_wr = self.krw_params['S_wr']
            S_or = self.kro_params['S_or']
            
            if Sw <= S_wr:
                return 0.0
            elif Sw >= 1 - S_or:
                return 1.0
            else:
                S_w_norm = (Sw - S_wr) / (1 - S_wr - S_or)
                return S_w_norm ** n_w
        else:
            raise ValueError(f"Unsupported krw model: {self.krw_model}")
    
    def get_capillary_pressure(self, Sw: float) -> float:
        """
        获取毛管压力
        
        Args:
            Sw: 水相饱和度
            
        Returns:
            毛管压力 (Pa)
        """
        if self.pc_model == 'brooks_corey':
            P_c0 = self.pc_params['P_c0']
            lambda_pc = self.pc_params['lambda']
            S_wr = self.krw_params['S_wr']
            S_or = self.kro_params['S_or']
            
            if Sw <= S_wr:
                return P_c0
            elif Sw >= 1 - S_or:
                return 0.0
            else:
                S_w_norm = (Sw - S_wr) / (1 - S_wr - S_or)
                return P_c0 * (S_w_norm ** (-1/lambda_pc))
        else:
            raise ValueError(f"Unsupported pc model: {self.pc_model}")
    
    def get_phase_viscosity(self, phase: str) -> float:
        """
        获取相粘度
        
        Args:
            phase: 相 ('oil' 或 'water')
            
        Returns:
            相粘度 (Pa·s)
        """
        if phase == 'oil':
            return self.mu_o
        elif phase == 'water':
            return self.mu_w
        else:
            raise ValueError(f"Invalid phase: {phase}")
    
    def compute_2phase_param(self, cell: 'Cell', S_iw: float):
        """
        计算两相流参数
        
        Args:
            cell: 单元对象
            S_iw: 初始水饱和度
        """
        # 计算相对渗透率
        kro = self.get_relative_permeability(cell.Sw, 'oil')
        krw = self.get_relative_permeability(cell.Sw, 'water')
        
        # 计算流度
        cell.kro_div_mu_o = kro / self.mu_o
        cell.krw_div_mu_w = krw / self.mu_w
        cell.sum_k_div_mu = cell.krw_div_mu_w + cell.kro_div_mu_o