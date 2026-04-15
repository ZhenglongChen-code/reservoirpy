"""
单位转换工具模块

核心类 UnitConverter 提供油藏工程中常用的单位转换功能。
内部计算统一使用 SI 单位，对外通过 uc 实例进行工业单位 ↔ SI 单位转换。

用法:
    from reservoirpy.utils.units import uc

    # 通用转换
    uc.convert(30, 'MPa', 'Pa', 'pressure')

    # 快捷方法: 工业单位 → SI
    uc.mpa_to_pa(30)        # 30e6
    uc.md_to_m2(100)        # 9.869e-14
    uc.mpas_to_pas(1.0)     # 0.001
    uc.d_to_s(365)          # 3.154e7

    # 快捷方法: SI → 工业单位
    uc.pa_to_mpa(30e6)      # 30.0
    uc.m2_to_md(1e-12)      # 1013.25
    uc.pas_to_mpas(0.001)   # 1.0
    uc.s_to_d(86400)        # 1.0
"""

import numpy as np
from typing import Union, List


class UnitConverter:
    """油藏工程单位转换器"""

    LENGTH = {
        'm': 1.0, 'cm': 0.01, 'mm': 0.001, 'km': 1000.0,
        'in': 0.0254, 'ft': 0.3048, 'yd': 0.9144, 'mi': 1609.344,
    }

    AREA = {
        'm2': 1.0, 'cm2': 1e-4, 'mm2': 1e-6, 'km2': 1e6,
        'in2': 0.00064516, 'ft2': 0.092903, 'acre': 4046.86,
    }

    VOLUME = {
        'm3': 1.0, 'cm3': 1e-6, 'mm3': 1e-9, 'L': 0.001, 'mL': 1e-6,
        'ft3': 0.0283168, 'bbl': 0.158987, 'gal': 0.00378541,
    }

    PRESSURE = {
        'Pa': 1.0, 'kPa': 1000.0, 'MPa': 1e6,
        'bar': 1e5, 'atm': 101325.0, 'psi': 6894.76, 'ksi': 6894760.0,
    }

    TIME = {
        's': 1.0, 'min': 60.0, 'h': 3600.0, 'd': 86400.0,
        'week': 604800.0, 'month': 2628000.0, 'year': 31536000.0,
    }

    VISCOSITY = {
        'Pa.s': 1.0, 'cP': 0.001, 'mPa.s': 0.001,
    }

    PERMEABILITY = {
        'm2': 1.0, 'D': 9.869233e-13, 'mD': 9.869233e-16, 'um2': 1e-12,
    }

    DENSITY = {
        'kg/m3': 1.0, 'g/cm3': 1000.0, 'lb/ft3': 16.0185, 'lb/in3': 27679.9,
    }

    _UNIT_MAP = {
        'length': 'LENGTH', 'area': 'AREA', 'volume': 'VOLUME',
        'pressure': 'PRESSURE', 'time': 'TIME', 'viscosity': 'VISCOSITY',
        'permeability': 'PERMEABILITY', 'density': 'DENSITY',
    }

    def convert(self, value: Union[float, np.ndarray],
                from_unit: str, to_unit: str,
                unit_type: str) -> Union[float, np.ndarray]:
        attr = self._UNIT_MAP.get(unit_type)
        if attr is None:
            raise ValueError(f"Unsupported unit type: {unit_type}, "
                             f"choose from {list(self._UNIT_MAP.keys())}")
        units = getattr(self, attr)
        if from_unit not in units:
            raise ValueError(f"Unsupported unit: {from_unit}, "
                             f"choose from {list(units.keys())}")
        if to_unit not in units:
            raise ValueError(f"Unsupported unit: {to_unit}, "
                             f"choose from {list(units.keys())}")
        return value * units[from_unit] / units[to_unit]

    def supported_units(self, unit_type: str) -> List[str]:
        attr = self._UNIT_MAP.get(unit_type)
        if attr is None:
            raise ValueError(f"Unsupported unit type: {unit_type}")
        return list(getattr(self, attr).keys())

    # ── 压力: 工业单位 ↔ SI ──────────────────────────────

    def mpa_to_pa(self, value):
        return self.convert(value, 'MPa', 'Pa', 'pressure')

    def pa_to_mpa(self, value):
        return self.convert(value, 'Pa', 'MPa', 'pressure')

    def psi_to_pa(self, value):
        return self.convert(value, 'psi', 'Pa', 'pressure')

    def pa_to_psi(self, value):
        return self.convert(value, 'Pa', 'psi', 'pressure')

    # ── 渗透率: 工业单位 ↔ SI ────────────────────────────

    def md_to_m2(self, value):
        return self.convert(value, 'mD', 'm2', 'permeability')

    def m2_to_md(self, value):
        return self.convert(value, 'm2', 'mD', 'permeability')

    def darcy_to_m2(self, value):
        return self.convert(value, 'D', 'm2', 'permeability')

    # ── 粘度: 工业单位 ↔ SI ──────────────────────────────

    def mpas_to_pas(self, value):
        return self.convert(value, 'mPa.s', 'Pa.s', 'viscosity')

    def pas_to_mpas(self, value):
        return self.convert(value, 'Pa.s', 'mPa.s', 'viscosity')

    def cp_to_pas(self, value):
        return self.convert(value, 'cP', 'Pa.s', 'viscosity')

    def pas_to_cp(self, value):
        return self.convert(value, 'Pa.s', 'cP', 'viscosity')

    # ── 时间: 工业单位 ↔ SI ──────────────────────────────

    def d_to_s(self, value):
        return self.convert(value, 'd', 's', 'time')

    def s_to_d(self, value):
        return self.convert(value, 's', 'd', 'time')

    def h_to_s(self, value):
        return self.convert(value, 'h', 's', 'time')

    def year_to_s(self, value):
        return self.convert(value, 'year', 's', 'time')

    # ── 长度: 工业单位 ↔ SI ──────────────────────────────

    def ft_to_m(self, value):
        return self.convert(value, 'ft', 'm', 'length')

    def m_to_ft(self, value):
        return self.convert(value, 'm', 'ft', 'length')

    # ── 体积: 工业单位 ↔ SI ──────────────────────────────

    def bbl_to_m3(self, value):
        return self.convert(value, 'bbl', 'm3', 'volume')

    def m3_to_bbl(self, value):
        return self.convert(value, 'm3', 'bbl', 'volume')

    def ft3_to_m3(self, value):
        return self.convert(value, 'ft3', 'm3', 'volume')


uc = UnitConverter()
