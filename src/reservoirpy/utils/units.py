"""
单位转换工具模块

提供常用的油藏工程单位转换功能
"""

import numpy as np
from typing import Union, List


class UnitConverter:
    """
    单位转换器类
    
    提供常用的油藏工程单位转换功能
    """
    
    # 长度单位转换因子
    LENGTH_UNITS = {
        'm': 1.0,        # 米
        'cm': 0.01,      # 厘米
        'mm': 0.001,     # 毫米
        'km': 1000.0,    # 千米
        'in': 0.0254,    # 英寸
        'ft': 0.3048,    # 英尺
        'yd': 0.9144,    # 码
        'mi': 1609.344   # 英里
    }
    
    # 面积单位转换因子
    AREA_UNITS = {
        'm2': 1.0,           # 平方米
        'cm2': 1e-4,         # 平方厘米
        'mm2': 1e-6,         # 平方毫米
        'km2': 1e6,          # 平方千米
        'in2': 0.00064516,   # 平方英寸
        'ft2': 0.092903,     # 平方英尺
        'acre': 4046.86,     # 英亩
    }
    
    # 体积单位转换因子
    VOLUME_UNITS = {
        'm3': 1.0,           # 立方米
        'cm3': 1e-6,         # 立方厘米
        'mm3': 1e-9,         # 立方毫米
        'L': 0.001,          # 升
        'mL': 1e-6,          # 毫升
        'ft3': 0.0283168,    # 立方英尺
        'bbl': 0.158987,     # 桶
        'gal': 0.00378541    # 加仑
    }
    
    # 压力单位转换因子
    PRESSURE_UNITS = {
        'Pa': 1.0,           # 帕斯卡
        'kPa': 1000.0,       # 千帕
        'MPa': 1e6,          # 兆帕
        'bar': 1e5,          # 巴
        'atm': 101325.0,     # 标准大气压
        'psi': 6894.76,      # 磅力每平方英寸
        'ksi': 6894760.0     # 千磅力每平方英寸
    }
    
    # 时间单位转换因子
    TIME_UNITS = {
        's': 1.0,            # 秒
        'min': 60.0,         # 分钟
        'h': 3600.0,         # 小时
        'd': 86400.0,        # 天
        'week': 604800.0,    # 周
        'month': 2628000.0,  # 月（平均）
        'year': 31536000.0   # 年
    }
    
    # 粘度单位转换因子
    VISCOSITY_UNITS = {
        'Pa.s': 1.0,         # 帕斯卡秒
        'cP': 0.001,         # 厘泊
        'mPa.s': 0.001       # 毫帕斯卡秒
    }
    
    # 渗透率单位转换因子
    PERMEABILITY_UNITS = {
        'm2': 1.0,           # 平方米
        'D': 9.869233e-13,   # 达西
        'mD': 9.869233e-16,  # 毫达西
        'um2': 1e-12         # 平方微米
    }
    
    # 密度单位转换因子
    DENSITY_UNITS = {
        'kg/m3': 1.0,        # 千克每立方米
        'g/cm3': 1000.0,     # 克每立方厘米
        'lb/ft3': 16.0185,   # 磅每立方英尺
        'lb/in3': 27679.9    # 磅每立方英寸
    }
    
    def convert(self, value: Union[float, np.ndarray], 
                from_unit: str, to_unit: str,
                unit_type: str = 'length') -> Union[float, np.ndarray]:
        """
        单位转换
        
        Args:
            value: 要转换的值
            from_unit: 源单位
            to_unit: 目标单位
            unit_type: 单位类型
            
        Returns:
            转换后的值
        """
        # 获取单位转换因子字典
        if unit_type == 'length':
            units = self.LENGTH_UNITS
        elif unit_type == 'area':
            units = self.AREA_UNITS
        elif unit_type == 'volume':
            units = self.VOLUME_UNITS
        elif unit_type == 'pressure':
            units = self.PRESSURE_UNITS
        elif unit_type == 'time':
            units = self.TIME_UNITS
        elif unit_type == 'viscosity':
            units = self.VISCOSITY_UNITS
        elif unit_type == 'permeability':
            units = self.PERMEABILITY_UNITS
        elif unit_type == 'density':
            units = self.DENSITY_UNITS
        else:
            raise ValueError(f"Unsupported unit type: {unit_type}")
        
        # 检查单位是否支持
        if from_unit not in units:
            raise ValueError(f"Unsupported from_unit: {from_unit}")
        if to_unit not in units:
            raise ValueError(f"Unsupported to_unit: {to_unit}")
        
        # 执行转换
        from_factor = units[from_unit]
        to_factor = units[to_unit]
        
        return value * from_factor / to_factor
    
    def convert_length(self, value: Union[float, np.ndarray], 
                      from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
        """
        长度单位转换
        
        Args:
            value: 要转换的值
            from_unit: 源单位
            to_unit: 目标单位
            
        Returns:
            转换后的值
        """
        return self.convert(value, from_unit, to_unit, 'length')
    
    def convert_area(self, value: Union[float, np.ndarray], 
                    from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
        """
        面积单位转换
        
        Args:
            value: 要转换的值
            from_unit: 源单位
            to_unit: 目标单位
            
        Returns:
            转换后的值
        """
        return self.convert(value, from_unit, to_unit, 'area')
    
    def convert_volume(self, value: Union[float, np.ndarray], 
                      from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
        """
        体积单位转换
        
        Args:
            value: 要转换的值
            from_unit: 源单位
            to_unit: 目标单位
            
        Returns:
            转换后的值
        """
        return self.convert(value, from_unit, to_unit, 'volume')
    
    def convert_pressure(self, value: Union[float, np.ndarray], 
                        from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
        """
        压力单位转换
        
        Args:
            value: 要转换的值
            from_unit: 源单位
            to_unit: 目标单位
            
        Returns:
            转换后的值
        """
        return self.convert(value, from_unit, to_unit, 'pressure')
    
    def convert_time(self, value: Union[float, np.ndarray], 
                    from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
        """
        时间单位转换
        
        Args:
            value: 要转换的值
            from_unit: 源单位
            to_unit: 目标单位
            
        Returns:
            转换后的值
        """
        return self.convert(value, from_unit, to_unit, 'time')
    
    def convert_viscosity(self, value: Union[float, np.ndarray], 
                         from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
        """
        粘度单位转换
        
        Args:
            value: 要转换的值
            from_unit: 源单位
            to_unit: 目标单位
            
        Returns:
            转换后的值
        """
        return self.convert(value, from_unit, to_unit, 'viscosity')
    
    def convert_permeability(self, value: Union[float, np.ndarray], 
                            from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
        """
        渗透率单位转换
        
        Args:
            value: 要转换的值
            from_unit: 源单位
            to_unit: 目标单位
            
        Returns:
            转换后的值
        """
        return self.convert(value, from_unit, to_unit, 'permeability')
    
    def convert_density(self, value: Union[float, np.ndarray], 
                       from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
        """
        密度单位转换
        
        Args:
            value: 要转换的值
            from_unit: 源单位
            to_unit: 目标单位
            
        Returns:
            转换后的值
        """
        return self.convert(value, from_unit, to_unit, 'density')
    
    def get_supported_units(self, unit_type: str) -> List[str]:
        """
        获取支持的单位列表
        
        Args:
            unit_type: 单位类型
            
        Returns:
            支持的单位列表
        """
        if unit_type == 'length':
            return list(self.LENGTH_UNITS.keys())
        elif unit_type == 'area':
            return list(self.AREA_UNITS.keys())
        elif unit_type == 'volume':
            return list(self.VOLUME_UNITS.keys())
        elif unit_type == 'pressure':
            return list(self.PRESSURE_UNITS.keys())
        elif unit_type == 'time':
            return list(self.TIME_UNITS.keys())
        elif unit_type == 'viscosity':
            return list(self.VISCOSITY_UNITS.keys())
        elif unit_type == 'permeability':
            return list(self.PERMEABILITY_UNITS.keys())
        elif unit_type == 'density':
            return list(self.DENSITY_UNITS.keys())
        else:
            raise ValueError(f"Unsupported unit type: {unit_type}")


# 创建全局单位转换器实例
unit_converter = UnitConverter()


def convert_units(value: Union[float, np.ndarray], 
                 from_unit: str, to_unit: str,
                 unit_type: str = 'length') -> Union[float, np.ndarray]:
    """
    单位转换函数
    
    Args:
        value: 要转换的值
        from_unit: 源单位
        to_unit: 目标单位
        unit_type: 单位类型
        
    Returns:
        转换后的值
    """
    return unit_converter.convert(value, from_unit, to_unit, unit_type)


def convert_length(value: Union[float, np.ndarray], 
                  from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """
    长度单位转换函数
    
    Args:
        value: 要转换的值
        from_unit: 源单位
        to_unit: 目标单位
        
    Returns:
        转换后的值
    """
    return unit_converter.convert_length(value, from_unit, to_unit)


def convert_pressure(value: Union[float, np.ndarray], 
                    from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """
    压力单位转换函数
    
    Args:
        value: 要转换的值
        from_unit: 源单位
        to_unit: 目标单位
        
    Returns:
        转换后的值
    """
    return unit_converter.convert_pressure(value, from_unit, to_unit)


def convert_time(value: Union[float, np.ndarray], 
                from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """
    时间单位转换函数
    
    Args:
        value: 要转换的值
        from_unit: 源单位
        to_unit: 目标单位
        
    Returns:
        转换后的值
    """
    return unit_converter.convert_time(value, from_unit, to_unit)


def convert_viscosity(value: Union[float, np.ndarray], 
                     from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """
    粘度单位转换函数
    
    Args:
        value: 要转换的值
        from_unit: 源单位
        to_unit: 目标单位
        
    Returns:
        转换后的值
    """
    return unit_converter.convert_viscosity(value, from_unit, to_unit)


def convert_permeability(value: Union[float, np.ndarray], 
                        from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """
    渗透率单位转换函数
    
    Args:
        value: 要转换的值
        from_unit: 源单位
        to_unit: 目标单位
        
    Returns:
        转换后的值
    """
    return unit_converter.convert_permeability(value, from_unit, to_unit)


# 常用单位转换的便捷函数
def m_to_ft(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """米转英尺"""
    return convert_length(value, 'm', 'ft')


def ft_to_m(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """英尺转米"""
    return convert_length(value, 'ft', 'm')


def pa_to_psi(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """帕斯卡转磅力每平方英寸"""
    return convert_pressure(value, 'Pa', 'psi')


def psi_to_pa(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """磅力每平方英寸转帕斯卡"""
    return convert_pressure(value, 'psi', 'Pa')


def s_to_d(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """秒转天"""
    return convert_time(value, 's', 'd')


def d_to_s(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """天转秒"""
    return convert_time(value, 'd', 's')


def cp_to_pas(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """厘泊转帕斯卡秒"""
    return convert_viscosity(value, 'cP', 'Pa.s')


def pas_to_cp(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """帕斯卡秒转厘泊"""
    return convert_viscosity(value, 'Pa.s', 'cP')


def md_to_m2(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """毫达西转平方米"""
    return convert_permeability(value, 'mD', 'm2')


def m2_to_md(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """平方米转毫达西"""
    return convert_permeability(value, 'm2', 'mD')
