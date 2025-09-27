"""
输出管理器

负责管理模拟结果的保存和输出
"""

from typing import Dict, Any, List
import numpy as np


class OutputManager:
    """
    输出管理器
    
    负责模拟结果的收集、存储和输出
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化输出管理器
        
        Args:
            config: 输出配置
        """
        self.config = config
        self.output_interval = config.get('output_interval', 10)
        self.save_pressure = config.get('save_pressure', True)
        self.save_saturation = config.get('save_saturation', True)
        self.save_temperature = config.get('save_temperature', False)
        self.save_well_data = config.get('save_well_data', True)
        
        # 结果存储
        self.results = {
            'time_history': [],
            'timestep_history': [],
            'field_data': {},  # 存储场变量历史
            'well_data': [],
            'simulation_info': {}
        }
        
        # 初始化场数据存储
        if self.save_pressure:
            self.results['field_data']['pressure'] = []
        if self.save_saturation:
            self.results['field_data']['saturation'] = []
        if self.save_temperature:
            self.results['field_data']['temperature'] = []
    
    def save_timestep(self, timestep: int, current_time: float, 
                     state_vars: Dict[str, np.ndarray]):
        """
        保存时间步结果
        
        Args:
            timestep: 时间步编号
            current_time: 当前时间
            state_vars: 状态变量字典
        """
        # 保存时间信息
        self.results['time_history'].append(current_time)
        self.results['timestep_history'].append(timestep)
        
        # 保存场变量
        for var_name, var_data in state_vars.items():
            if var_name in self.results['field_data']:
                self.results['field_data'][var_name].append(var_data.copy())
        
        # TODO: 保存井数据
        if self.save_well_data:
            # 这里可以添加井数据的保存逻辑
            pass
    
    def get_results(self) -> Dict[str, Any]:
        """
        获取完整的模拟结果
        
        Returns:
            模拟结果字典
        """
        # 转换为numpy数组以便后续处理
        for var_name in self.results['field_data']:
            if len(self.results['field_data'][var_name]) > 0:
                self.results['field_data'][var_name] = np.array(
                    self.results['field_data'][var_name])
        
        self.results['time_history'] = np.array(self.results['time_history'])
        self.results['timestep_history'] = np.array(self.results['timestep_history'])
        
        return self.results
    
    def get_final_state(self) -> Dict[str, np.ndarray]:
        """
        获取最终状态
        
        Returns:
            最终状态变量字典
        """
        final_state = {}
        for var_name, var_history in self.results['field_data'].items():
            if len(var_history) > 0:
                final_state[var_name] = var_history[-1]
        return final_state
    
    def get_variable_history(self, var_name: str) -> np.ndarray:
        """
        获取指定变量的历史数据
        
        Args:
            var_name: 变量名称
            
        Returns:
            变量历史数据数组
        """
        if var_name not in self.results['field_data']:
            raise ValueError(f"Variable {var_name} not found in results")
        return self.results['field_data'][var_name]
    
    def save_to_file(self, filename: str, format: str = 'npz'):
        """
        保存结果到文件
        
        Args:
            filename: 文件名
            format: 文件格式 ('npz', 'mat', 'hdf5')
        """
        results = self.get_results()
        
        if format == 'npz':
            # 保存为NumPy压缩格式
            save_dict = {
                'time_history': results['time_history'],
                'timestep_history': results['timestep_history']
            }
            # 添加场变量
            for var_name, var_data in results['field_data'].items():
                save_dict[var_name] = var_data
            
            np.savez_compressed(filename, **save_dict)
            
        elif format == 'mat':
            # 保存为MATLAB格式（需要scipy）
            from scipy.io import savemat
            savemat(filename, results)
            
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def __repr__(self):
        num_timesteps = len(self.results['time_history'])
        variables = list(self.results['field_data'].keys())
        return f"OutputManager(timesteps={num_timesteps}, variables={variables})"