"""
数据验证工具模块

提供输入数据验证和结果验证功能
"""

import numpy as np
from typing import Dict, Any, List, Union, Optional
import warnings


def validate_mesh_config(config: Dict[str, Any]) -> bool:
    """
    验证网格配置
    
    Args:
        config: 网格配置字典
        
    Returns:
        验证是否通过
    """
    required_keys = ['nx', 'ny', 'nz', 'dx', 'dy', 'dz']
    
    # 检查必需的键
    for key in required_keys:
        if key not in config:
            warnings.warn(f"Missing required mesh config key: {key}")
            return False
    
    # 检查数据类型和值
    try:
        nx, ny, nz = int(config['nx']), int(config['ny']), int(config['nz'])
        dx, dy, dz = float(config['dx']), float(config['dy']), float(config['dz'])
        
        if nx <= 0 or ny <= 0 or nz <= 0:
            warnings.warn("Mesh dimensions must be positive integers")
            return False
            
        if dx <= 0 or dy <= 0 or dz <= 0:
            warnings.warn("Mesh cell sizes must be positive")
            return False
            
        if nx * ny * nz > 1e6:
            warnings.warn("Mesh size is very large, may cause memory issues")
            
    except (ValueError, TypeError):
        warnings.warn("Invalid mesh configuration values")
        return False
    
    return True


def validate_physics_config(config: Dict[str, Any]) -> bool:
    """
    验证物理属性配置
    
    Args:
        config: 物理属性配置字典
        
    Returns:
        验证是否通过
    """
    required_keys = ['permeability', 'porosity', 'viscosity', 'compressibility']
    
    # 检查必需的键
    for key in required_keys:
        if key not in config:
            warnings.warn(f"Missing required physics config key: {key}")
            return False
    
    # 检查数据类型和值
    try:
        permeability = float(config['permeability'])
        porosity = float(config['porosity'])
        viscosity = float(config['viscosity'])
        compressibility = float(config['compressibility'])
        
        if permeability <= 0:
            warnings.warn("Permeability must be positive")
            return False
            
        if not (0 < porosity < 1):
            warnings.warn("Porosity must be between 0 and 1")
            return False
            
        if viscosity <= 0:
            warnings.warn("Viscosity must be positive")
            return False
            
        if compressibility < 0:
            warnings.warn("Compressibility must be non-negative")
            return False
            
    except (ValueError, TypeError):
        warnings.warn("Invalid physics configuration values")
        return False
    
    return True


def validate_well_config(config: Dict[str, Any], mesh_config: Dict[str, Any]) -> bool:
    """
    验证井配置
    
    Args:
        config: 井配置字典
        mesh_config: 网格配置字典
        
    Returns:
        验证是否通过
    """
    if 'wells' not in config:
        return True  # 没有井配置是允许的
    
    wells = config['wells']
    if not isinstance(wells, list):
        warnings.warn("Wells config must be a list")
        return False
    
    # 获取网格尺寸
    nx, ny, nz = mesh_config['nx'], mesh_config['ny'], mesh_config['nz']
    
    for i, well in enumerate(wells):
        # 检查必需的键
        required_keys = ['location', 'control_type', 'value']
        for key in required_keys:
            if key not in well:
                warnings.warn(f"Missing required well config key in well {i}: {key}")
                return False
        
        # 验证位置
        location = well['location']
        if not isinstance(location, list) or len(location) != 3:
            warnings.warn(f"Invalid location format in well {i}")
            return False
        
        z, y, x = location
        if not (0 <= z < nz and 0 <= y < ny and 0 <= x < nx):
            warnings.warn(f"Invalid well location in well {i}: {location}")
            return False
        
        # 验证控制类型
        control_type = well['control_type']
        if control_type not in ['rate', 'bhp']:
            warnings.warn(f"Invalid control type in well {i}: {control_type}")
            return False
        
        # 验证值
        try:
            value = float(well['value'])
            if value < 0:
                warnings.warn(f"Negative well value in well {i}: {value}")
                return False
        except (ValueError, TypeError):
            warnings.warn(f"Invalid well value in well {i}: {well['value']}")
            return False
    
    return True


def validate_simulation_config(config: Dict[str, Any]) -> bool:
    """
    验证模拟配置
    
    Args:
        config: 模拟配置字典
        
    Returns:
        验证是否通过
    """
    required_keys = ['dt', 'total_time']
    
    # 检查必需的键
    for key in required_keys:
        if key not in config:
            warnings.warn(f"Missing required simulation config key: {key}")
            return False
    
    # 检查数据类型和值
    try:
        dt = float(config['dt'])
        total_time = float(config['total_time'])
        
        if dt <= 0:
            warnings.warn("Time step must be positive")
            return False
            
        if total_time <= 0:
            warnings.warn("Total simulation time must be positive")
            return False
            
        if dt > total_time:
            warnings.warn("Time step is larger than total simulation time")
            
    except (ValueError, TypeError):
        warnings.warn("Invalid simulation configuration values")
        return False
    
    return True


def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证完整配置
    
    Args:
        config: 完整配置字典
        
    Returns:
        验证是否通过
    """
    # 检查必需的顶级键
    required_sections = ['mesh', 'physics', 'simulation']
    for section in required_sections:
        if section not in config:
            warnings.warn(f"Missing required config section: {section}")
            return False
    
    # 验证各个部分
    if not validate_mesh_config(config['mesh']):
        return False
        
    if not validate_physics_config(config['physics']):
        return False
        
    if not validate_simulation_config(config['simulation']):
        return False
        
    if not validate_well_config(config, config['mesh']):
        return False
    
    return True


def check_mass_conservation(pressure_field: np.ndarray, 
                          mesh, physics, dt: float, 
                          wells: Optional[List] = None) -> Dict[str, float]:
    """
    检查质量守恒
    
    Args:
        pressure_field: 压力场
        mesh: 网格对象
        physics: 物理属性对象
        dt: 时间步长
        wells: 井列表
        
    Returns:
        质量守恒检查结果
    """
    # 计算总质量变化
    total_mass_change = 0.0
    total_inflow = 0.0
    total_outflow = 0.0
    
    # 对每个单元计算质量变化
    for i in range(mesh.ncell):
        z, y, x = mesh.get_cell_coords(i)
        cell = mesh.cell_list[i]
        
        # 计算单元质量变化
        volume = cell.volume
        porosity = cell.porosity
        compressibility = physics.compressibility
        
        # 简化的质量变化计算
        mass_change = volume * porosity * compressibility * pressure_field[i]
        total_mass_change += mass_change
        
        # 计算流入和流出
        neighbors = mesh.get_neighbors(z, y, x)
        for direction, neighbor_idx in enumerate(neighbors):
            if neighbor_idx != -1:
                # 计算传导率
                trans = physics.get_transmissibility(i, neighbor_idx, 
                                                   ['x', 'x', 'y', 'y', 'z', 'z'][direction])
                # 简化的流量计算
                flow = trans * (pressure_field[neighbor_idx] - pressure_field[i])
                if flow > 0:
                    total_inflow += flow
                else:
                    total_outflow += abs(flow)
    
    # 计算井贡献
    total_well_rate = 0.0
    if wells:
        for well in wells:
            if well.control_type == 'rate':
                total_well_rate += well.value
    
    # 检查质量守恒
    mass_balance = total_mass_change - (total_inflow - total_outflow) * dt - total_well_rate * dt
    
    return {
        'mass_change': total_mass_change,
        'inflow': total_inflow,
        'outflow': total_outflow,
        'well_rate': total_well_rate,
        'mass_balance': mass_balance,
        'balance_error': abs(mass_balance) / (abs(total_mass_change) + 1e-10)
    }


def check_numerical_stability(mesh, physics, dt: float) -> Dict[str, Union[bool, float]]:
    """
    检查数值稳定性
    
    Args:
        mesh: 网格对象
        physics: 物理属性对象
        dt: 时间步长
        
    Returns:
        稳定性检查结果
    """
    # 计算CFL条件
    max_cfl = 0.0
    stable = True
    
    for i in range(mesh.ncell):
        z, y, x = mesh.get_cell_coords(i)
        cell = mesh.cell_list[i]
        
        # 计算局部CFL数
        volume = cell.volume
        porosity = cell.porosity
        compressibility = physics.compressibility
        
        # 计算总传导率
        neighbors = mesh.get_neighbors(z, y, x)
        total_trans = 0.0
        for direction, neighbor_idx in enumerate(neighbors):
            if neighbor_idx != -1:
                trans = physics.get_transmissibility(i, neighbor_idx, 
                                                   ['x', 'x', 'y', 'y', 'z', 'z'][direction])
                total_trans += abs(trans)
        
        # 计算局部时间步长限制
        if total_trans > 0:
            accumulation = volume * porosity * compressibility
            if accumulation > 0:
                dt_limit = accumulation / total_trans
                cfl = dt / dt_limit
                max_cfl = max(max_cfl, cfl)
                
                if cfl > 1.0:
                    stable = False
    
    return {
        'stable': stable,
        'max_cfl': max_cfl,
        'recommended_dt': dt / max_cfl if max_cfl > 0 else dt
    }


def validate_results(results: Dict[str, Any], 
                    mesh, physics, wells: Optional[List] = None) -> Dict[str, Any]:
    """
    验证模拟结果
    
    Args:
        results: 模拟结果字典
        mesh: 网格对象
        physics: 物理属性对象
        wells: 井列表
        
    Returns:
        验证结果
    """
    validation_results = {}
    
    # 检查压力场
    if 'pressure_history' in results and results['pressure_history']:
        final_pressure = results['pressure_history'][-1]
        
        # 检查压力值范围
        pressure_min, pressure_max = final_pressure.min(), final_pressure.max()
        validation_results['pressure_range'] = {
            'min': pressure_min,
            'max': pressure_max,
            'valid': pressure_min >= 0 and pressure_max < 1e10  # 合理的压力范围
        }
        
        # 检查是否有NaN或无穷大值
        has_nan = np.isnan(final_pressure).any()
        has_inf = np.isinf(final_pressure).any()
        validation_results['pressure_validity'] = {
            'no_nan': not has_nan,
            'no_inf': not has_inf,
            'valid': not (has_nan or has_inf)
        }
    
    # 检查饱和度场（如果是两相流）
    if 'saturation_history' in results and results['saturation_history']:
        final_saturation = results['saturation_history'][-1]
        
        # 检查饱和度值范围
        saturation_min, saturation_max = final_saturation.min(), final_saturation.max()
        validation_results['saturation_range'] = {
            'min': saturation_min,
            'max': saturation_max,
            'valid': saturation_min >= 0 and saturation_max <= 1  # 饱和度应在0-1之间
        }
        
        # 检查是否有NaN或无穷大值
        has_nan = np.isnan(final_saturation).any()
        has_inf = np.isinf(final_saturation).any()
        validation_results['saturation_validity'] = {
            'no_nan': not has_nan,
            'no_inf': not has_inf,
            'valid': not (has_nan or has_inf)
        }
    
    # 检查质量守恒（如果提供了最终压力场）
    if 'pressure_history' in results and results['pressure_history']:
        final_pressure = results['pressure_history'][-1]
        mass_conservation = check_mass_conservation(
            final_pressure, mesh, physics, 
            results.get('dt', 86400.0), wells)
        validation_results['mass_conservation'] = mass_conservation
    
    # 检查数值稳定性
    dt = results.get('dt', 86400.0)
    stability = check_numerical_stability(mesh, physics, dt)
    validation_results['numerical_stability'] = stability
    
    # 总体验证结果
    all_valid = all(
        result.get('valid', True) 
        for result in validation_results.values() 
        if isinstance(result, dict)
    )
    validation_results['overall_valid'] = all_valid
    
    return validation_results


class ConfigValidator:
    """
    配置验证器类
    
    提供配置文件的验证功能
    """
    
    def __init__(self):
        """初始化配置验证器"""
        pass
    
    def validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证配置
        
        Args:
            config: 配置字典
            
        Returns:
            验证结果字典
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # 验证网格配置
        if 'mesh' in config:
            if not validate_mesh_config(config['mesh']):
                results['valid'] = False
                results['errors'].append("Invalid mesh configuration")
        else:
            results['valid'] = False
            results['errors'].append("Missing mesh configuration")
        
        # 验证物理属性配置
        if 'physics' in config:
            if not validate_physics_config(config['physics']):
                results['valid'] = False
                results['errors'].append("Invalid physics configuration")
        else:
            results['valid'] = False
            results['errors'].append("Missing physics configuration")
        
        # 验证模拟配置
        if 'simulation' in config:
            if not validate_simulation_config(config['simulation']):
                results['valid'] = False
                results['errors'].append("Invalid simulation configuration")
        else:
            results['valid'] = False
            results['errors'].append("Missing simulation configuration")
        
        # 验证井配置
        if not validate_well_config(config, config.get('mesh', {})):
            results['valid'] = False
            results['errors'].append("Invalid well configuration")
        
        return results
    
    def __repr__(self):
        return "ConfigValidator()"


def create_config_validator() -> ConfigValidator:
    """
    创建配置验证器
    
    Returns:
        ConfigValidator实例
    """
    return ConfigValidator()
