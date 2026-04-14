"""
新架构测试脚本

测试新的BaseModel架构和ModelFactory工厂模式
"""
import numpy as np
from reservoirpy.core.simulator import ReservoirSimulator
from reservoirpy.models.model_factory import ModelFactory

def test_new_architecture():
    """测试新架构的基本功能"""
    
    # 配置字典
    config = {
        'mesh': {
            'nx': 5, 'ny': 5, 'nz': 1,
            'dx': 10, 'dy': 10, 'dz': 10
        },
        'physics': {
            'type': 'single_phase',
            'permeability': 10.0,  # mD
            'porosity': 0.2,
            'viscosity': 0.001,     # Pa·s
            'compressibility': 1e-9  # 1/Pa
        },
        'wells': [
            {
                'location': [0, 2, 2],      # 井位置（中心）
                'control_type': 'bhp',      # 定井底流压控制
                'value': 100000,            # 井底流压值（Pa）
                'rw': 0.05,                 # 井筒半径
                'skin_factor': 0            # 表皮因子
            }
        ],
        'simulation': {
            'dt': 3600,           # 时间步长(秒) = 1小时
            'total_time': 36000,  # 总时间 = 10小时
            'initial_pressure': 300000, # 初始压力(Pa)
            'output_interval': 2  # 输出间隔
        },
        'output': {
            'save_pressure': True,
            'output_interval': 2
        }
    }
    
    print("=== 新架构测试 ===")
    
    # 1. 测试ModelFactory
    print("\n1. 测试ModelFactory")
    registered_models = ModelFactory.get_registered_models()
    print(f"已注册的模型: {list(registered_models.keys())}")
    
    # 2. 创建模拟器
    print("\n2. 创建ReservoirSimulator")
    simulator = ReservoirSimulator(config_dict=config)
    
    # 3. 获取模型信息
    print("\n3. 模型信息")
    model_info = simulator.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # 4. 运行模拟
    print("\n4. 运行模拟")
    results = simulator.run_simulation()
    
    # 5. 分析结果
    print("\n5. 模拟结果分析")
    print(f"时间步数: {len(results['time_history'])}")
    print(f"模拟时间范围: {results['time_history'][0]:.1f}s - {results['time_history'][-1]:.1f}s")
    
    # 检查压力场
    if 'pressure' in results['field_data']:
        pressure_history = results['field_data']['pressure']
        initial_pressure = pressure_history[0]
        final_pressure = pressure_history[-1]
        
        print(f"初始压力范围: {np.min(initial_pressure)/1000:.1f} - {np.max(initial_pressure)/1000:.1f} kPa")
        print(f"最终压力范围: {np.min(final_pressure)/1000:.1f} - {np.max(final_pressure)/1000:.1f} kPa")
        
        # 检查井位置的压力
        well = simulator.wells[0]
        z, y, x = well.location
        cell_index = simulator.mesh.get_cell_index(z, y, x)
        
        well_pressure_initial = initial_pressure[cell_index]
        well_pressure_final = final_pressure[cell_index]
        
        print(f"井压力变化: {well_pressure_initial/1000:.1f} -> {well_pressure_final/1000:.1f} kPa")
        print(f"井底流压设定: {well.value/1000:.1f} kPa")
        
        # 验证井压力是否合理
        if well.control_type == 'bhp':
            expected_bhp = well.value
            if abs(well_pressure_final - expected_bhp) < 50000:  # 50 kPa容差
                print("✓ 井底流压控制正常")
            else:
                print(f"✗ 井底流压控制异常，期望{expected_bhp/1000:.1f}，实际{well_pressure_final/1000:.1f}")
        
        # 检查压力是否在合理范围内
        if np.all(final_pressure > 0) and np.all(final_pressure < 1e7):
            print("✓ 压力范围合理")
        else:
            print("✗ 压力范围异常")
    
    # 6. 测试稳态求解（如果支持）
    print("\n6. 测试稳态求解")
    try:
        steady_state = simulator.run_steady_state()
        steady_pressure = steady_state['pressure']
        print(f"稳态压力范围: {np.min(steady_pressure)/1000:.1f} - {np.max(steady_pressure)/1000:.1f} kPa")
        print("✓ 稳态求解成功")
    except Exception as e:
        print(f"稳态求解失败: {e}")
    
    print("\n=== 测试完成 ===")
    return results

if __name__ == "__main__":
    try:
        results = test_new_architecture()
        print("新架构测试成功！")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()