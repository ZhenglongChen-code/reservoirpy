# 油藏数值模拟器 API 文档

## 概述

油藏数值模拟器是一个轻量级、模块化的软件包，用于求解单相流和两相流渗流方程。该软件包提供了结构化网格生成、物理属性建模、有限体积法离散化、线性求解器和井模型等功能。

## 核心模块

### 1. StructuredMesh (网格模块)

结构化网格管理类，提供结构化矩形网格的几何和拓扑信息，支持2D和3D网格。

#### 类定义
```python
class StructuredMesh(BaseMesh):
    def __init__(self, nx: int, ny: int, nz: int, dx: float, dy: float, dz: float)
```

#### 参数说明
- `nx`: X方向单元数
- `ny`: Y方向单元数
- `nz`: Z方向单元数
- `dx`: X方向单元尺寸 (米)
- `dy`: Y方向单元尺寸 (米)
- `dz`: Z方向单元尺寸 (米)

#### 属性
- `nx`: X方向单元数
- `ny`: Y方向单元数
- `nz`: Z方向单元数
- `dx`: X方向单元尺寸
- `dy`: Y方向单元尺寸
- `dz`: Z方向单元尺寸
- `n_cells`: 总单元数
- `node_list`: 节点列表，包含所有网格节点
- `cell_list`: 单元列表，包含所有网格单元
- `total_cells`: 总单元数（只读属性）
- `grid_shape`: 网格形状 (nx, ny, nz)（只读属性）

#### 方法

##### `__init__(self, nx: int, ny: int, nz: int, dx: float, dy: float, dz: float)`
初始化结构化网格

##### `_generate_nodes(self) -> List[Node]`
生成网格节点

Returns:
    节点列表

##### `_generate_cells(self) -> List[CubeCell]`
生成网格单元

Returns:
    单元列表

##### `get_cell_volume(self, i: int, j: int, k: int) -> float`
获取指定单元的体积

Args:
    i: Z方向索引
    j: Y方向索引
    k: X方向索引

Returns:
    单元体积

##### `get_face_area(self, direction: str, i: int, j: int, k: int) -> float`
获取指定方向界面的面积

Args:
    direction: 方向 ('x', 'y', 'z')
    i: Z方向索引
    j: Y方向索引
    k: X方向索引

Returns:
    界面面积

##### `get_face_distance(self, direction: str, i: int, j: int, k: int) -> float`
获取到相邻单元中心的距离

Args:
    direction: 方向 ('x', 'y', 'z')
    i: Z方向索引
    j: Y方向索引
    k: X方向索引

Returns:
    距离

##### `get_neighbors(self, i: int, j: int, k: int) -> List[int]`
获取相邻单元索引

Args:
    i: Z方向索引
    j: Y方向索引
    k: X方向索引

Returns:
    相邻单元索引列表 [W, E, N, S, F, B]

##### `is_boundary_cell(self, i: int, j: int, k: int) -> bool`
判断是否为边界单元

Args:
    i: Z方向索引
    j: Y方向索引
    k: X方向索引

Returns:
    是否为边界单元

##### `get_cell_index(self, i: int, j: int, k: int) -> int`
获取单元的一维索引

Args:
    i: Z方向索引
    j: Y方向索引
    k: X方向索引

Returns:
    单元索引

##### `get_cell_coords(self, index: int) -> Tuple[int, int, int]`
从一维索引获取三维坐标

Args:
    index: 单元索引(0 -- self.n_cells-1)

Returns:
    (i, j, k) 三维坐标,对应（z, y, x)方向索引

##### `get_cell_centers(self) -> np.ndarray`
获取所有单元中心坐标

Returns:
    形状为 (ncell, 3) 的数组，每行为 [x, y, z]

##### `get_cell_volumes(self) -> np.ndarray`
获取所有单元体积

Returns:
    形状为 (ncell,) 的数组

##### `__repr__(self)`
返回网格对象的字符串表示

### 2. PropertyManager (属性管理器)

属性管理器类，负责物理属性的存储和访问，解耦物理属性与网格单元的直接关联。

#### 类定义
```python
class PropertyManager:
    def __init__(self, mesh: StructuredMesh, config: Dict[str, Any])
```

#### 参数说明
- `mesh`: 网格对象
- `config`: 配置字典

#### 属性
- `mesh`: 网格对象
- `properties`: 包含所有物理属性的字典

#### 方法

##### `__init__(self, mesh: StructuredMesh, config: Dict[str, Any])`
初始化属性管理器

##### `_initialize_properties(self, config: Dict[str, Any]) -> Dict[str, Any]`
初始化所有物理属性

Args:
    config: 配置字典

Returns:
    包含所有物理属性的字典

##### `_init_permeability(self, config: Dict[str, Any]) -> Union[float, np.ndarray]`
初始化渗透率

Args:
    config: 配置字典

Returns:
    渗透率值或数组

##### `_init_porosity(self, config: Dict[str, Any]) -> Union[float, np.ndarray]`
初始化孔隙度

Args:
    config: 配置字典

Returns:
    孔隙度值或数组

##### `get_cell_property(self, cell_index: int, property_name: str) -> float`
获取指定单元的属性值

Args:
    cell_index: 单元索引
    property_name: 属性名称 ('permeability', 'porosity')

Returns:
    指定单元的属性值

### 3. SinglePhaseProperties (单相流物理属性)

单相流物理属性类，封装单相流动物理参数，包括渗透率、孔隙度、粘度、压缩系数等。

#### 类定义
```python
class SinglePhaseProperties(BasePhysics):
    def __init__(self, mesh: StructuredMesh, config: Dict[str, Any])
```

#### 参数说明
- `mesh`: 网格对象
- `config`: 配置字典，包含以下键值对（所有参数均使用SI标准单位）：
  - `permeability` (float or np.ndarray): 渗透率，单位为mD，默认100.0
  - `porosity` (float or np.ndarray): 孔隙度，默认0.2
  - `viscosity` (float): 粘度，单位为Pa·s，默认0.001
  - `compressibility` (float): 压缩系数，单位为1/Pa，默认1e-9
  - `reference_pressure` (float): 参考压力，单位为Pa，默认0.0

#### 属性
- `mesh`: 网格对象
- `config`: 配置字典
- `property_manager`: 属性管理器
- `viscosity`: 粘度值 (Pa·s)
- `compressibility`: 压缩系数 (1/Pa)
- `reference_pressure`: 参考压力 (Pa)

#### 方法

##### `__init__(self, mesh: StructuredMesh, config: Dict[str, Any])`
初始化单相流物理属性

##### `_init_viscosity(self, config: Dict[str, Any]) -> float`
初始化粘度

Args:
    config: 配置字典

Returns:
    粘度值 (Pa·s)

##### `_init_compressibility(self, config: Dict[str, Any]) -> float`
初始化压缩系数

Args:
    config: 配置字典

Returns:
    压缩系数值 (1/Pa)

##### `get_transmissibility(self, cell_i: int, cell_j: int, direction: str) -> float`
计算传导率

Args:
    cell_i: 当前单元索引
    cell_j: 相邻单元索引
    direction: 方向 ('x', 'y', 'z')

Returns:
    传导率值

##### `get_fluid_density(self, pressure: float) -> float`
计算流体密度（微可压缩模型）

Args:
    pressure: 压力 (Pa)

Returns:
    密度 (kg/m³)

### 4. Well (井模型)

#### 类定义
```python
class Well:
    def __init__(self, location: List[int], control_type: str, value: float, 
                 rw: float = 0.05, skin_factor: float = 0, well_length: float = 1000)
```

#### 参数说明
- `location`: 井所在网格索引 [z, y, x]
- `control_type`: 控制类型 ('rate' 或 'bhp')
- `value`: 控制值（流量 m³/s 或压力 Pa）
- `rw`: 井筒半径 (米)
- `skin_factor`: 表皮因子
- `well_length`: 井长度 (米)

#### 属性
- `location`: 井位置
- `control_type`: 控制类型
- `value`: 控制值
- `rw`: 井筒半径
- `skin_factor`: 表皮因子
- `well_length`: 井长度

#### 方法

##### `compute_well_index(mesh: StructuredMesh, permeability: float, viscosity: float) -> float`
计算产能指数

##### `compute_well_term(pressure: float, dt: float) -> float`
计算井项贡献

##### `add_to_rhs(b: np.ndarray, cell_index: int, pressure: float, dt: float) -> None`
将井项添加到右端向量

### 5. FVMDiscretizer (有限体积法离散化器)

#### 类定义
```python
class FVMDiscretizer:
    def __init__(self, mesh: StructuredMesh, physics: SinglePhaseProperties)
```

#### 参数说明
- `mesh`: 结构化网格
- `physics`: 物理属性

#### 方法

##### `discretize_single_phase(dt: float, pressure: np.ndarray, well_manager: WellManager) -> Tuple[csr_matrix, np.ndarray]`
离散化单相流方程

##### `solve_linear_system(A: csr_matrix, b: np.ndarray, method: str = 'bicgstab', tolerance: float = 1e-8, max_iterations: int = 1000) -> np.ndarray`
求解线性系统

### 6. ReservoirSimulator (主模拟器)

#### 类定义
```python
class ReservoirSimulator:
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None)
```

#### 参数说明
- `config_path`: 配置文件路径
- `config_dict`: 配置字典

#### 配置文件格式
```
mesh:
  nx: 20                    # X方向单元数
  ny: 20                    # Y方向单元数
  nz: 1                     # Z方向单元数
  dx: 10.0                  # X方向单元尺寸 (m)
  dy: 10.0                  # Y方向单元尺寸 (m)
  dz: 5.0                   # Z方向单元尺寸 (m)

physics:
  permeability: 100         # 渗透率 (mD)
  porosity: 0.2             # 孔隙度
  viscosity: 0.001          # 粘度 (Pa·s)
  compressibility: 1e-9     # 压缩系数 (1/Pa)
  reference_pressure: 0.0   # 参考压力 (Pa)

wells:
  - location: [0, 5, 5]     # 井位置 [z, y, x]
    control_type: "rate"    # 控制类型: "rate" 或 "bhp"
    value: 0.001            # 控制值 (m³/s 或 Pa)
    rw: 0.05                # 井筒半径 (m)

simulation:
  dt: 86400                 # 时间步长 (s)
  total_time: 31536000      # 总时间 (s)
  output_interval: 10       # 输出间隔
  initial_pressure: 30e6    # 初始压力 (Pa)

linear_solver:
  method: "bicgstab"        # 求解器方法: "bicgstab", "gmres", "direct"
  tolerance: 1e-8           # 收敛容差
  max_iterations: 1000      # 最大迭代次数
```

#### 方法

##### `run_simulation() -> Dict[str, Any]`
运行模拟

##### `get_pressure_field() -> np.ndarray`
获取当前压力场

##### `get_pressure_at_location(i: int, j: int, k: int) -> float`
获取指定位置的压力

##### `get_cell_properties(i: int, j: int, k: int) -> Dict[str, Any]`
获取指定单元的属性

## 使用示例

### 基本使用
```
from reservoir_sim import ReservoirSimulator

# 使用配置文件创建模拟器
simulator = ReservoirSimulator('config/default_config.yaml')

# 运行模拟
results = simulator.run_simulation()

# 获取结果
pressure_field = simulator.get_pressure_field()
```

### 程序化配置
```
from reservoir_sim import ReservoirSimulator

# 使用字典配置
config = {
    'mesh': {
        'nx': 20, 'ny': 20, 'nz': 1,
        'dx': 10.0, 'dy': 10.0, 'dz': 5.0
    },
    'physics': {
        'permeability': 100.0,
        'porosity': 0.2,
        'viscosity': 0.001,
        'compressibility': 1e-9
    },
    'wells': [
        {'location': [0, 5, 5], 'control_type': 'rate', 'value': 0.001},
        {'location': [0, 15, 15], 'control_type': 'bhp', 'value': 1e6}
    ],
    'simulation': {
        'dt': 86400,
        'total_time': 31536000,
        'initial_pressure': 30e6
    }
}

simulator = ReservoirSimulator(config_dict=config)
results = simulator.run_simulation()
```